"""Recognizes nothing: reports how long each phrase is. For debugging pipelines without a GPU."""

from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime, timedelta

from pydantic import Field

from livekit_transcript.core.interfaces import ComponentParams, Recognizer
from livekit_transcript.core.models import (
    Speaker,
    SpeechEnd,
    SpeechEvent,
    SpeechFrames,
    SpeechStart,
    TranscriptSegment,
)


class DummyRecognizerParams(ComponentParams):
    # Emit an interim segment every N ms of speech, like a streaming recognizer; None — finals only
    interim_every_ms: int | None = Field(default=None, gt=0)


class DummyRecognizer(Recognizer):
    Params = DummyRecognizerParams

    def __init__(self, params: DummyRecognizerParams) -> None:
        self._params = params

    async def transcribe(
        self, speaker: Speaker, events: AsyncIterator[SpeechEvent]
    ) -> AsyncGenerator[TranscriptSegment]:
        interim_every = (
            timedelta(milliseconds=self._params.interim_every_ms)
            if self._params.interim_every_ms
            else None
        )
        phrase = 0
        start: datetime | None = None
        duration = timedelta()
        reported = timedelta()

        def segment(is_final: bool, end: datetime) -> TranscriptSegment:
            assert start is not None
            return TranscriptSegment(
                speaker=speaker,
                segment_id=f"{speaker.id}:{phrase}",
                is_final=is_final,
                start=start,
                end=end,
                text=f"[речь {(end - start).total_seconds():.1f} с]",
            )

        async for event in events:
            match event:
                case SpeechStart(time=time):
                    phrase += 1
                    start, duration, reported = time, timedelta(), timedelta()
                case SpeechFrames(frames=frames) if start is not None:
                    duration += sum((frame.duration for frame in frames), timedelta())
                    if interim_every and duration - reported >= interim_every:
                        reported = duration
                        yield segment(is_final=False, end=start + duration)
                case SpeechEnd(time=time) if start is not None:
                    yield segment(is_final=True, end=time)
                    start = None
