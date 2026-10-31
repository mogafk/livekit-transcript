"""WAV files as a source: one file per speaker, all of them start at the same moment."""

import asyncio
import wave
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import Field

from livekit_transcript.core.interfaces import ComponentParams, ConfigError, Source
from livekit_transcript.core.models import (
    NUM_CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    AudioFrame,
    Speaker,
    SpeakerTrack,
    utc_now,
)

# Disk is read in chunks of this many frames, then sliced into frames
FRAMES_PER_READ = 50


class WavFile(ComponentParams):
    path: Path
    speaker_id: str = Field(min_length=1)
    speaker_name: str | None = None


class WavSourceParams(ComponentParams):
    files: list[WavFile] = Field(min_length=1)
    # Pace frames like a live call instead of reading as fast as possible
    realtime: bool = False
    frame_ms: int = Field(default=20, gt=0, le=1000)
    session_name: str = "wav"


class WavSource(Source):
    Params = WavSourceParams

    def __init__(self, params: WavSourceParams) -> None:
        self._params = params
        self.session_name = params.session_name
        self._stopped = False
        speaker_ids = [file.speaker_id for file in params.files]
        if len(set(speaker_ids)) != len(speaker_ids):
            raise ConfigError("source (wav): speaker_id файлов должны быть уникальны")
        for file in params.files:
            _check_format(file.path)

    def stop(self) -> None:
        self._stopped = True

    async def tracks(self) -> AsyncIterator[SpeakerTrack]:
        started_at = utc_now()
        clock_start = asyncio.get_running_loop().time()
        for file in self._params.files:
            speaker = Speaker(id=file.speaker_id, name=file.speaker_name or file.speaker_id)
            yield SpeakerTrack(speaker, self._frames(file.path, started_at, clock_start))

    async def _frames(
        self, path: Path, started_at: datetime, clock_start: float
    ) -> AsyncIterator[AudioFrame]:
        loop = asyncio.get_running_loop()
        frame_bytes = SAMPLE_RATE * self._params.frame_ms // 1000 * SAMPLE_WIDTH
        offset = 0  # samples from the start of the file
        with wave.open(str(path), "rb") as wav:
            while not self._stopped:
                chunk = await asyncio.to_thread(
                    wav.readframes, frame_bytes // SAMPLE_WIDTH * FRAMES_PER_READ
                )
                if not chunk:
                    return
                for i in range(0, len(chunk), frame_bytes):
                    if self._stopped:
                        return
                    data = chunk[i : i + frame_bytes]
                    frame = AudioFrame(data, started_at + timedelta(seconds=offset / SAMPLE_RATE))
                    offset += frame.num_samples
                    if self._params.realtime:
                        # A live frame is available only once it has been fully spoken
                        delay = clock_start + offset / SAMPLE_RATE - loop.time()
                        if delay > 0:
                            await asyncio.sleep(delay)
                    yield frame


def _check_format(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as wav:
            actual = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
    except (OSError, EOFError, wave.Error) as e:
        raise ConfigError(f"source (wav): не удалось открыть {path}: {e}") from e
    if actual != (NUM_CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE):
        channels, width, rate = actual
        raise ConfigError(
            f"source (wav): {path} — {channels} кан., {width * 8} бит, {rate} Гц; "
            f"нужен PCM 16 бит, 16000 Гц, моно. Конвертировать: "
            f"ffmpeg -i <вход> -ar 16000 -ac 1 -c:a pcm_s16le <выход>.wav"
        )
