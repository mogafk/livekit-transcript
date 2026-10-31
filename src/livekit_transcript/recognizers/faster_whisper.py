"""Whisper via faster-whisper (CTranslate2). Recognizes a phrase as a whole on SpeechEnd;
optionally re-recognizes the phrase so far for interim segments.
"""

import asyncio
import logging
import os
import sys
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from itertools import count
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from pydantic import Field

from livekit_transcript.core.interfaces import ComponentParams, Recognizer
from livekit_transcript.core.models import (
    SAMPLE_RATE,
    Speaker,
    SpeechEnd,
    SpeechEvent,
    SpeechFrames,
    SpeechStart,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


class FasterWhisperParams(ComponentParams):
    # Name from the faster-whisper catalog (downloaded to the HuggingFace cache) or a local path
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    # None — detect per phrase (slower and less stable on short phrases)
    language: str | None = None
    beam_size: int = Field(default=5, ge=1)
    # Hints spelling of names and terms: "LiveKit, SFU, diarization"
    initial_prompt: str | None = None
    # Shorter phrases are skipped: on clicks and breaths Whisper tends to hallucinate
    min_phrase_ms: int = Field(default=300, ge=0)
    # Whisper's own "no speech" filter for its segments
    no_speech_threshold: float = Field(default=0.6, ge=0, le=1)
    # Re-recognize the unfinished phrase every N ms of speech for interim segments; None — off
    interim_every_ms: int | None = Field(default=None, gt=0)


class FasterWhisperRecognizer(Recognizer):
    Params = FasterWhisperParams

    def __init__(self, params: FasterWhisperParams) -> None:
        self._params = params
        # Per speaker across tracks: a rejoined speaker must not reuse segment ids
        self._phrase_ids: defaultdict[str, count[int]] = defaultdict(lambda: count(1))
        self._model: WhisperModel | None = None
        # One worker: the GPU runs one phrase at a time, tracks queue up for it
        self._executor: ThreadPoolExecutor | None = None

    async def start(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        logger.info(
            "Загрузка модели %s (%s, %s), первый запуск скачивает её",
            self._params.model,
            self._params.device,
            self._params.compute_type,
        )
        await self._in_worker(self._load)
        logger.info("Модель %s загружена", self._params.model)

    async def aclose(self) -> None:
        if self._executor is not None:
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
            self._executor = None
        self._model = None

    def _load(self) -> None:
        _add_cuda_dll_dirs()
        self._model = WhisperModel(
            self._params.model,
            device=self._params.device,
            compute_type=self._params.compute_type,
        )
        # The first inference is several times slower: pay for it now, not on the first phrase
        self._recognize(bytes(SAMPLE_RATE * 2))

    async def transcribe(
        self, speaker: Speaker, events: AsyncIterator[SpeechEvent]
    ) -> AsyncGenerator[TranscriptSegment]:
        min_phrase = timedelta(milliseconds=self._params.min_phrase_ms)
        interim_every = (
            timedelta(milliseconds=self._params.interim_every_ms)
            if self._params.interim_every_ms
            else None
        )
        phrase = 0
        start: datetime | None = None
        chunks: list[bytes] = []
        duration = timedelta()
        reported = timedelta()
        has_interim = False

        def segment(is_final: bool, end: datetime, text: str) -> TranscriptSegment:
            assert start is not None
            return TranscriptSegment(
                speaker=speaker,
                segment_id=f"{speaker.id}:{phrase}",
                is_final=is_final,
                start=start,
                end=end,
                text=text,
            )

        async for event in events:
            match event:
                case SpeechStart(time=time):
                    phrase = next(self._phrase_ids[speaker.id])
                    start, chunks, has_interim = time, [], False
                    duration, reported = timedelta(), timedelta()
                case SpeechFrames(frames=frames) if start is not None:
                    for frame in frames:
                        chunks.append(frame.data)
                        duration += frame.duration
                    if (
                        interim_every
                        and duration >= min_phrase
                        and duration - reported >= interim_every
                    ):
                        reported = duration
                        text = await self._in_worker(self._recognize, b"".join(chunks))
                        if text:
                            has_interim = True
                            yield segment(is_final=False, end=start + duration, text=text)
                case SpeechEnd(time=time) if start is not None:
                    text = ""
                    if duration >= min_phrase:
                        text = await self._in_worker(self._recognize, b"".join(chunks))
                    # An empty final still closes interims already shown for this phrase
                    if text or has_interim:
                        yield segment(is_final=True, end=time, text=text)
                    start, chunks = None, []

    def _recognize(self, pcm: bytes) -> str:
        """Runs in the worker thread."""
        assert self._model is not None
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        segments, _ = self._model.transcribe(
            audio,
            language=self._params.language,
            beam_size=self._params.beam_size,
            initial_prompt=self._params.initial_prompt,
            no_speech_threshold=self._params.no_speech_threshold,
            # Phrases are independent: an error must not leak into the next one
            condition_on_previous_text=False,
            # Phrases are already cut by our VAD
            vad_filter=False,
            without_timestamps=True,
        )
        # `segments` is lazy: decoding happens while iterating, so here in the worker
        return " ".join(text for s in segments if (text := s.text.strip()))

    async def _in_worker[T](self, fn: Callable[..., T], *args: object) -> T:
        assert self._executor is not None
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)


_dll_dirs_added = False


def _add_cuda_dll_dirs() -> None:
    """On Windows the CUDA libraries from pip (nvidia-cublas-cu12, nvidia-cudnn-cu12)
    are not on the DLL search path, and CTranslate2 fails with "cublas64_12.dll is not found".
    """
    global _dll_dirs_added
    if sys.platform != "win32" or _dll_dirs_added:
        return
    _dll_dirs_added = True
    try:
        import nvidia
    except ImportError:
        return
    for base in nvidia.__path__:
        for bin_dir in Path(base).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
