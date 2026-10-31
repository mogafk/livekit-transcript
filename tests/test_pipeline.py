import asyncio
import json
from datetime import timedelta
from pathlib import Path

import pytest

from livekit_transcript.core.models import SAMPLE_RATE
from livekit_transcript.core.pipeline import Pipeline
from livekit_transcript.recognizers.dummy import DummyRecognizer, DummyRecognizerParams
from livekit_transcript.sinks.stdout import StdoutSink, StdoutSinkParams
from livekit_transcript.sources.wav import WavFile, WavSource, WavSourceParams
from livekit_transcript.vad.energy import EnergyVad, EnergyVadParams
from tests.helpers import CollectingSink, RecordingSaver, make_wav


def make_pipeline(
    tmp_path: Path, *, realtime: bool = False, interim_every_ms: int | None = None
) -> tuple[Pipeline, CollectingSink, RecordingSaver]:
    alice = make_wav(tmp_path / "alice.wav", [(0.5, True), (1.0, False), (0.3, True), (0.2, False)])
    bob = make_wav(tmp_path / "bob.wav", [(0.6, False), (0.4, True), (0.4, False)])
    source = WavSource(
        WavSourceParams(
            files=[
                WavFile(path=alice, speaker_id="alice", speaker_name="Алиса"),
                WavFile(path=bob, speaker_id="bob"),
            ],
            realtime=realtime,
        )
    )
    sink, saver = CollectingSink(), RecordingSaver()
    pipeline = Pipeline(
        source=source,
        vad=EnergyVad(EnergyVadParams(min_silence_ms=300, pre_roll_ms=0)),
        recognizer=DummyRecognizer(DummyRecognizerParams(interim_every_ms=interim_every_ms)),
        sinks=[sink],
        savers=[saver],
    )
    return pipeline, sink, saver


async def test_segments_are_per_speaker_with_source_timestamps(tmp_path: Path) -> None:
    pipeline, sink, saver = make_pipeline(tmp_path)

    await pipeline.run()

    finals = [s for s in sink.segments if s.is_final]
    assert [(s.speaker.id, s.segment_id) for s in sorted(finals, key=lambda s: s.start)] == [
        ("alice", "alice:1"),
        ("bob", "bob:1"),
        ("alice", "alice:2"),
    ]
    session_start = saver.session.started_at
    offsets = {
        s.segment_id: (
            (s.start - session_start).total_seconds(),
            (s.end - session_start).total_seconds(),
        )
        for s in finals
    }
    # Both files start at the session start; tolerance covers utc_now() differences
    assert offsets["alice:1"] == pytest.approx((0.0, 0.5), abs=0.05)
    assert offsets["bob:1"] == pytest.approx((0.6, 1.0), abs=0.05)
    assert offsets["alice:2"] == pytest.approx((1.5, 1.8), abs=0.05)
    assert sink.closed


async def test_savers_get_all_audio_and_final_segments(tmp_path: Path) -> None:
    pipeline, sink, saver = make_pipeline(tmp_path, interim_every_ms=100)

    await pipeline.run()

    assert any(not s.is_final for s in sink.segments)
    assert all(s.is_final for s in saver.segments)
    assert len(saver.segments) == 3
    samples = {k: sum(f.num_samples for f in v) for k, v in saver.frames.items()}
    assert samples == {"alice": 2 * SAMPLE_RATE, "bob": int(1.4 * SAMPLE_RATE)}
    assert sorted(saver.calls) == [
        ("end", "alice"),
        ("end", "bob"),
        ("start", "alice"),
        ("start", "bob"),
    ]
    assert saver.finalized


async def test_stop_ends_realtime_session_gracefully(tmp_path: Path) -> None:
    pipeline, sink, saver = make_pipeline(tmp_path, realtime=True)

    task = asyncio.create_task(pipeline.run())
    await asyncio.sleep(0.7)  # alice's first phrase is over, the second hasn't started
    pipeline.stop()
    await asyncio.wait_for(task, timeout=2)

    finals = {s.segment_id for s in sink.segments if s.is_final}
    assert "alice:1" in finals
    assert "alice:2" not in finals
    assert saver.finalized
    assert sum((f.duration for f in saver.frames["alice"]), timedelta()) < timedelta(seconds=1)


async def test_stdout_sink_writes_jsonl(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    pipeline, _, _ = make_pipeline(tmp_path)
    pipeline._sinks = [StdoutSink(StdoutSinkParams(format="jsonl", interim=False))]

    await pipeline.run()

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {line["segment_id"] for line in lines} == {"alice:1", "alice:2", "bob:1"}
    alice = next(line for line in lines if line["speaker_id"] == "alice")
    assert alice["speaker_name"] == "Алиса"
    assert alice["start"].endswith("Z")
