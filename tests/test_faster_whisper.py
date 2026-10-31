from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from livekit_transcript.core.models import (
    SAMPLE_RATE,
    AudioFrame,
    Speaker,
    SpeechEnd,
    SpeechEvent,
    SpeechFrames,
    SpeechStart,
    TranscriptSegment,
)
from livekit_transcript.recognizers.faster_whisper import (
    FasterWhisperParams,
    FasterWhisperRecognizer,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
FRAME = timedelta(milliseconds=20)
SPEAKER = Speaker(id="alice", name="Алиса")
SAMPLES = Path(__file__).parent.parent / "samples"


@dataclass
class FakeSegment:
    text: str


class FakeModel:
    """Answers with the audio length, or with what `texts` says for that length in ms."""

    def __init__(self, texts: dict[int, str] | None = None) -> None:
        self.texts = texts or {}
        self.calls: list[tuple[int, dict[str, Any]]] = []

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[list[FakeSegment], None]:
        assert audio.dtype == np.float32
        ms = len(audio) * 1000 // SAMPLE_RATE
        self.calls.append((ms, kwargs))
        return [FakeSegment(f" {self.texts.get(ms, f'{ms} ms')} ")], None


async def make_recognizer(
    monkeypatch: pytest.MonkeyPatch, model: FakeModel, **params: Any
) -> FasterWhisperRecognizer:
    def fake_load(self: FasterWhisperRecognizer) -> None:
        self._model = model

    monkeypatch.setattr(FasterWhisperRecognizer, "_load", fake_load)
    recognizer = FasterWhisperRecognizer(FasterWhisperParams(**params))
    await recognizer.start()
    return recognizer


def phrase(start_frame: int, num_frames: int) -> list[SpeechEvent]:
    frames = tuple(
        AudioFrame(bytes(SAMPLE_RATE // 50 * 2), T0 + (start_frame + i) * FRAME)
        for i in range(num_frames)
    )
    return [
        SpeechStart(frames[0].start),
        *(SpeechFrames((frame,)) for frame in frames),
        SpeechEnd(frames[-1].end),
    ]


async def run(
    recognizer: FasterWhisperRecognizer, events: list[SpeechEvent]
) -> list[TranscriptSegment]:
    async def stream() -> AsyncIterator[SpeechEvent]:
        for event in events:
            yield event

    try:
        return [s async for s in recognizer.transcribe(SPEAKER, stream())]
    finally:
        await recognizer.aclose()


async def test_one_final_segment_per_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    recognizer = await make_recognizer(monkeypatch, model, language="ru", initial_prompt="LiveKit")

    segments = await run(recognizer, phrase(0, 25) + phrase(50, 40))

    assert [(s.segment_id, s.is_final, s.text) for s in segments] == [
        ("alice:1", True, "500 ms"),
        ("alice:2", True, "800 ms"),
    ]
    assert (segments[1].start, segments[1].end) == (T0 + 50 * FRAME, T0 + 90 * FRAME)
    kwargs = model.calls[0][1]
    assert kwargs["language"] == "ru"
    assert kwargs["initial_prompt"] == "LiveKit"
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["vad_filter"] is False


async def test_short_phrase_is_not_recognized(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    recognizer = await make_recognizer(monkeypatch, model, min_phrase_ms=300)

    segments = await run(recognizer, phrase(0, 10))

    assert segments == []
    assert model.calls == []


async def test_empty_text_gives_no_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    recognizer = await make_recognizer(monkeypatch, FakeModel({500: ""}))

    assert await run(recognizer, phrase(0, 25)) == []


async def test_interim_segments_then_final(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel({600: ""})
    recognizer = await make_recognizer(monkeypatch, model, interim_every_ms=200, min_phrase_ms=100)

    segments = await run(recognizer, phrase(0, 30))

    assert [(s.is_final, s.text) for s in segments] == [
        (False, "200 ms"),
        (False, "400 ms"),
        # Recognized as empty, but still closes the interims shown before
        (True, ""),
    ]
    assert {s.segment_id for s in segments} == {"alice:1"}
    assert segments[1].end == T0 + 20 * FRAME


@pytest.mark.gpu
@pytest.mark.skipif(not (SAMPLES / "alice.wav").exists(), reason="нет samples/alice.wav")
async def test_real_model_on_samples() -> None:
    from livekit_transcript.core.pipeline import Pipeline
    from livekit_transcript.sources.wav import WavFile, WavSource, WavSourceParams
    from livekit_transcript.vad.energy import EnergyVad, EnergyVadParams
    from tests.helpers import CollectingSink

    sink = CollectingSink()
    source = WavSource(
        WavSourceParams(
            files=[
                WavFile(path=SAMPLES / "alice.wav", speaker_id="alice"),
                WavFile(path=SAMPLES / "bob.wav", speaker_id="bob"),
            ]
        )
    )
    pipeline = Pipeline(
        source=source,
        vad=EnergyVad(EnergyVadParams()),
        recognizer=FasterWhisperRecognizer(FasterWhisperParams(language="en")),
        sinks=[sink],
    )

    await pipeline.run()

    text = {
        speaker: " ".join(s.text for s in sink.segments if s.speaker.id == speaker).lower()
        for speaker in ("alice", "bob")
    }
    assert "can you hear me" in text["alice"]
    assert "loud and clear" in text["bob"]


async def test_rejoined_speaker_does_not_reuse_segment_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    recognizer = await make_recognizer(monkeypatch, FakeModel())

    async def stream() -> AsyncIterator[SpeechEvent]:
        for event in phrase(0, 25):
            yield event

    first = [s async for s in recognizer.transcribe(SPEAKER, stream())]
    second = [s async for s in recognizer.transcribe(SPEAKER, stream())]
    await recognizer.aclose()

    assert [s.segment_id for s in first + second] == ["alice:1", "alice:2"]
