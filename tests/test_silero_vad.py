from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from livekit_transcript.core.models import (
    SAMPLE_RATE,
    AudioFrame,
    SpeechEnd,
    SpeechEvent,
    SpeechFrames,
    SpeechStart,
)
from livekit_transcript.vad import silero
from livekit_transcript.vad.silero import WINDOW_SAMPLES, SileroVad, SileroVadParams

T0 = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW = timedelta(seconds=WINDOW_SAMPLES / SAMPLE_RATE)


class FakeSession:
    """Returns scripted speech probabilities, one per window."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = iter(probabilities)
        self.inputs: list[np.ndarray] = []

    def run(self, _: None, feed: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        self.inputs.append(feed["input"])
        return np.array([[next(self.probabilities)]], dtype=np.float32), feed["state"]


async def frames_of(samples: np.ndarray, frame_samples: int = 160) -> AsyncIterator[AudioFrame]:
    """Cuts audio into 10 ms frames, like LiveKit delivers it."""
    pcm = samples.astype("<i2").tobytes()
    step = frame_samples * 2
    for i in range(0, len(pcm), step):
        yield AudioFrame(pcm[i : i + step], T0 + timedelta(seconds=i / 2 / SAMPLE_RATE))


async def run_vad(vad: SileroVad, samples: np.ndarray) -> list[SpeechEvent]:
    await vad.start()
    try:
        return [event async for event in vad.process(frames_of(samples))]
    finally:
        await vad.aclose()


async def fake_vad(
    monkeypatch: pytest.MonkeyPatch, probabilities: list[float], **params: Any
) -> tuple[SileroVad, FakeSession]:
    session = FakeSession(probabilities)
    monkeypatch.setattr(silero, "_load_session", lambda: session)
    return SileroVad(SileroVadParams(pre_roll_ms=0, **params)), session


async def test_frames_are_recut_into_model_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    vad, session = await fake_vad(monkeypatch, [0.9] * 3, min_silence_ms=1000)

    # 3.5 windows of audio in 10 ms frames: three windows go to the model, the tail doesn't
    events = await run_vad(vad, np.arange(WINDOW_SAMPLES * 7 // 2) % 1000)

    assert [x.shape for x in session.inputs] == [(1, 64 + WINDOW_SAMPLES)] * 3
    # Context is the end of the previous window
    assert np.array_equal(session.inputs[1][0, :64], session.inputs[0][0, -64:])
    frames = [f for e in events if isinstance(e, SpeechFrames) for f in e.frames]
    assert [f.start for f in frames] == [T0 + i * WINDOW for i in range(4)]
    # The tail is kept inside the phrase, all audio is passed on
    assert sum(f.num_samples for f in frames) == WINDOW_SAMPLES * 7 // 2
    assert events[-1] == SpeechEnd(T0 + 3 * WINDOW)


async def test_hysteresis_keeps_phrase_between_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 0.4 doesn't start a phrase (< 0.5) but doesn't end one either (>= 0.35)
    probabilities = [0.4, 0.6, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]
    vad, _ = await fake_vad(monkeypatch, probabilities, min_silence_ms=100)

    events = await run_vad(vad, np.zeros(WINDOW_SAMPLES * len(probabilities)))

    assert [e for e in events if not isinstance(e, SpeechFrames)] == [
        SpeechStart(T0 + WINDOW),
        # Ends after the last window above neg_threshold, once 100 ms of silence passed
        SpeechEnd(T0 + 4 * WINDOW),
    ]


def test_neg_threshold_above_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="neg_threshold"):
        SileroVadParams(threshold=0.5, neg_threshold=0.6)


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(SAMPLE_RATE * 2),
        0.3 * 32767 * np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE),
        np.random.default_rng(0).normal(0, 3000, SAMPLE_RATE * 2),
    ],
    ids=["silence", "tone", "white-noise"],
)
async def test_real_model_ignores_non_speech(samples: np.ndarray) -> None:
    # Unlike the energy VAD, a loud tone or noise is not speech
    assert await run_vad(SileroVad(SileroVadParams()), samples) == []
