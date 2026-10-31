import wave
from pathlib import Path

import numpy as np

from livekit_transcript.core.interfaces import Saver, Sink
from livekit_transcript.core.models import (
    SAMPLE_RATE,
    AudioFrame,
    SessionInfo,
    Speaker,
    TranscriptSegment,
)


def make_wav(path: Path, parts: list[tuple[float, bool]], rate: int = SAMPLE_RATE) -> Path:
    """Writes mono 16-bit WAV of (seconds, loud) parts: a 440 Hz tone or digital silence."""
    chunks = []
    for seconds, loud in parts:
        t = np.arange(int(seconds * rate)) / rate
        chunks.append(0.3 * np.sin(2 * np.pi * 440 * t) if loud else np.zeros_like(t))
    samples = (np.concatenate(chunks) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return path


class CollectingSink(Sink):
    def __init__(self) -> None:
        self.segments: list[TranscriptSegment] = []
        self.closed = False

    async def on_segment(self, segment: TranscriptSegment) -> None:
        self.segments.append(segment)

    async def aclose(self) -> None:
        self.closed = True


class RecordingSaver(Saver):
    def __init__(self) -> None:
        self.session: SessionInfo | None = None
        self.calls: list[tuple[str, str]] = []
        self.frames: dict[str, list[AudioFrame]] = {}
        self.segments: list[TranscriptSegment] = []
        self.finalized = False

    async def on_session_start(self, session: SessionInfo) -> None:
        self.session = session

    async def on_track_start(self, speaker: Speaker) -> None:
        self.calls.append(("start", speaker.id))
        self.frames[speaker.id] = []

    async def on_frame(self, speaker: Speaker, frame: AudioFrame) -> None:
        self.frames[speaker.id].append(frame)

    async def on_segment(self, segment: TranscriptSegment) -> None:
        self.segments.append(segment)

    async def on_track_end(self, speaker: Speaker) -> None:
        self.calls.append(("end", speaker.id))

    async def finalize(self) -> None:
        self.finalized = True
