from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import numpy as np

from livekit_transcript.core.models import (
    SAMPLE_RATE,
    AudioFrame,
    SpeechEnd,
    SpeechEvent,
    SpeechFrames,
    SpeechStart,
)
from livekit_transcript.vad.energy import EnergyVad, EnergyVadParams

T0 = datetime(2026, 1, 1, tzinfo=UTC)
FRAME = timedelta(milliseconds=20)


async def frames_of(pattern: str) -> AsyncIterator[AudioFrame]:
    """One 20 ms frame per char: '#' — tone, '.' — silence."""
    n = SAMPLE_RATE // 50
    tone = (0.3 * 32767 * np.sin(2 * np.pi * 440 * np.arange(n) / SAMPLE_RATE)).astype("<i2")
    silence = np.zeros(n, dtype="<i2")
    for i, char in enumerate(pattern):
        yield AudioFrame((tone if char == "#" else silence).tobytes(), T0 + i * FRAME)


async def run_vad(pattern: str, **params: int) -> list[SpeechEvent]:
    vad = EnergyVad(EnergyVadParams(**params))
    return [event async for event in vad.process(frames_of(pattern))]


async def test_phrase_ends_after_min_silence() -> None:
    events = await run_vad("..###......", min_silence_ms=100, pre_roll_ms=40)

    assert events[0] == SpeechStart(T0)  # pre-roll pulls the start 2 frames back
    assert events[-1] == SpeechEnd(T0 + 5 * FRAME)  # end of the last loud frame
    frames = [f for e in events if isinstance(e, SpeechFrames) for f in e.frames]
    assert [f.start for f in frames] == [T0 + i * FRAME for i in range(10)]
    assert sum(isinstance(e, SpeechEnd) for e in events) == 1


async def test_short_pause_does_not_split_phrase() -> None:
    events = await run_vad("##..##......", min_silence_ms=100, pre_roll_ms=0)

    assert [type(e) for e in events].count(SpeechStart) == 1
    assert events[-1] == SpeechEnd(T0 + 6 * FRAME)


async def test_long_pause_splits_phrases() -> None:
    events = await run_vad("##......##", min_silence_ms=100, pre_roll_ms=0)

    starts = [e for e in events if isinstance(e, SpeechStart)]
    ends = [e for e in events if isinstance(e, SpeechEnd)]
    assert starts == [SpeechStart(T0), SpeechStart(T0 + 8 * FRAME)]
    # The second phrase is closed by the end of input
    assert ends == [SpeechEnd(T0 + 2 * FRAME), SpeechEnd(T0 + 10 * FRAME)]


async def test_silence_only_yields_nothing() -> None:
    assert await run_vad("..........") == []
