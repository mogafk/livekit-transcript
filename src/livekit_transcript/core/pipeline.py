"""Wires components together. Knows only the interfaces, never concrete implementations."""

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, aclosing
from operator import methodcaller

from livekit_transcript.core.interfaces import Recognizer, Saver, Sink, Source, Vad
from livekit_transcript.core.models import (
    AudioFrame,
    SessionInfo,
    Speaker,
    SpeakerTrack,
    TranscriptSegment,
    utc_now,
)

logger = logging.getLogger(__name__)

# Queued segments per sink after which we warn that the sink can't keep up
SINK_BACKLOG_WARNING = 1000


class _SinkWriter:
    """Feeds one sink from its own queue, so a slow sink doesn't stall recognition."""

    def __init__(self, sink: Sink) -> None:
        self._sink = sink
        self._queue: asyncio.Queue[TranscriptSegment | None] = asyncio.Queue()
        self._warned = False

    def put(self, segment: TranscriptSegment) -> None:
        self._queue.put_nowait(segment)
        if self._queue.qsize() > SINK_BACKLOG_WARNING and not self._warned:
            self._warned = True
            logger.warning("Sink %s не успевает обрабатывать сегменты", type(self._sink).__name__)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def run(self) -> None:
        while (segment := await self._queue.get()) is not None:
            try:
                await self._sink.on_segment(segment)
            except Exception:
                logger.exception("Ошибка в sink %s", type(self._sink).__name__)


class Pipeline:
    def __init__(
        self,
        source: Source,
        vad: Vad,
        recognizer: Recognizer,
        sinks: Sequence[Sink] = (),
        savers: Sequence[Saver] = (),
    ) -> None:
        self._source = source
        self._vad = vad
        self._recognizer = recognizer
        self._sinks = list(sinks)
        self._savers = list(savers)
        self._writers: list[_SinkWriter] = []
        self._stopping = False

    def stop(self) -> None:
        """Graceful stop: the source ends its tracks, pending phrases are still recognized."""
        self._stopping = True
        self._source.stop()

    async def run(self) -> None:
        async with AsyncExitStack() as stack:
            # The source goes last: it starts producing audio only when everything else is ready
            components = [*self._sinks, *self._savers, self._vad, self._recognizer, self._source]
            for component in components:
                if self._stopping:
                    return
                await component.start()
                stack.push_async_callback(component.aclose)
            await self._run_session()

    async def _run_session(self) -> None:
        session = SessionInfo(name=self._source.session_name, started_at=utc_now())
        logger.info("Сессия %s начата", session.name)
        await self._notify_savers("on_session_start", session)
        self._writers = [_SinkWriter(sink) for sink in self._sinks]
        try:
            async with asyncio.TaskGroup() as writers:
                for writer in self._writers:
                    writers.create_task(writer.run())
                try:
                    async with asyncio.TaskGroup() as tracks:
                        async for track in self._source.tracks():
                            tracks.create_task(self._process_track(track))
                finally:
                    for writer in self._writers:
                        writer.close()
        finally:
            await self._notify_savers("finalize")
            logger.info("Сессия %s завершена", session.name)

    async def _process_track(self, track: SpeakerTrack) -> None:
        speaker = track.speaker
        logger.info("Спикер %s (%s): трек начат", speaker.name, speaker.id)
        await self._notify_savers("on_track_start", speaker)
        try:
            events = self._vad.process(self._tee_to_savers(speaker, track.frames))
            async with aclosing(self._recognizer.transcribe(speaker, events)) as segments:
                async for segment in segments:
                    await self._publish(segment)
        except Exception:
            # One broken track must not end the whole session
            logger.exception("Спикер %s: ошибка обработки трека", speaker.id)
        finally:
            await self._notify_savers("on_track_end", speaker)
            logger.info("Спикер %s (%s): трек завершён", speaker.name, speaker.id)

    async def _tee_to_savers(
        self, speaker: Speaker, frames: AsyncIterator[AudioFrame]
    ) -> AsyncIterator[AudioFrame]:
        async for frame in frames:
            await self._notify_savers("on_frame", speaker, frame)
            yield frame

    async def _publish(self, segment: TranscriptSegment) -> None:
        for writer in self._writers:
            writer.put(segment)
        if segment.is_final:
            await self._notify_savers("on_segment", segment)

    async def _notify_savers(self, method: str, *args: object) -> None:
        call = methodcaller(method, *args)
        for saver in self._savers:
            try:
                await call(saver)
            except Exception:
                logger.exception("Ошибка в saver %s", type(saver).__name__)
