import json
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from livekit_transcript.config import ComponentConfig, load_config
from livekit_transcript.core.models import (
    SAMPLE_RATE,
    AudioFrame,
    SessionInfo,
    Speaker,
    TranscriptSegment,
)
from livekit_transcript.core.pipeline import Pipeline
from livekit_transcript.recognizers.dummy import DummyRecognizer, DummyRecognizerParams
from livekit_transcript.registry import build_pipeline
from livekit_transcript.savers.filesystem import FilesystemSaver, FilesystemSaverParams
from livekit_transcript.sources.wav import WavFile, WavSource, WavSourceParams
from livekit_transcript.vad.energy import EnergyVad, EnergyVadParams
from tests.helpers import CollectingSink, make_wav

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
ALICE = Speaker(id="alice", name="Алиса")
BOB = Speaker(id="bob/1", name="Боб")


def frame(at_ms: int, ms: int = 20, value: int = 1000) -> AudioFrame:
    samples = SAMPLE_RATE * ms // 1000
    return AudioFrame(
        value.to_bytes(2, "little", signed=True) * samples, T0 + timedelta(milliseconds=at_ms)
    )


def wav_samples(path: Path) -> int:
    with wave.open(str(path)) as w:
        return w.getnframes()


def session_dir(root: Path) -> Path:
    [path] = root.iterdir()
    return path


async def test_gaps_reconnects_offsets_and_transcript(tmp_path: Path) -> None:
    saver = FilesystemSaver(FilesystemSaverParams(dir=tmp_path, min_gap_ms=500))
    await saver.on_session_start(SessionInfo(name="room/1", started_at=T0))
    await saver.on_track_start(ALICE)
    await saver.on_frame(ALICE, frame(0))
    # Jitter below min_gap is not filled: it would push the file ahead of real time
    await saver.on_frame(ALICE, frame(120))
    await saver.on_track_end(ALICE)
    # Rejoined after a pause: same file, the pause is filled with silence
    await saver.on_track_start(ALICE)
    await saver.on_frame(ALICE, frame(1040))
    await saver.on_track_start(BOB)
    await saver.on_frame(BOB, frame(2500))
    late = TranscriptSegment(
        BOB, "bob/1:1", True, T0 + timedelta(seconds=3), T0 + timedelta(seconds=4), "Привет"
    )
    early = TranscriptSegment(ALICE, "alice:1", True, T0, T0 + timedelta(seconds=1), "Слышно?")
    silent = TranscriptSegment(ALICE, "alice:2", True, T0, T0, "")
    for segment in (late, early, silent):
        await saver.on_segment(segment)
    await saver.finalize()
    await saver.aclose()

    directory = session_dir(tmp_path)
    assert directory.name.startswith("room_1-")
    # 20 + 20 ms of frames, the 1000 ms gap filled up to the third frame, + 20 ms
    assert wav_samples(directory / "alice.wav") == SAMPLE_RATE * 1060 // 1000
    assert wav_samples(directory / "bob_1.wav") == SAMPLE_RATE * 20 // 1000
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    files = {f["speaker_id"]: f for f in source["params"]["files"]}
    assert files["alice"]["offset_ms"] == 0
    assert files["bob/1"]["offset_ms"] == 2500
    assert files["bob/1"]["speaker_name"] == "Боб"
    assert Path(files["bob/1"]["path"]).is_absolute()
    transcript = (directory / "transcript.txt").read_text(encoding="utf-8").splitlines()
    assert [line.split("] ")[1] for line in transcript] == ["Алиса: Слышно?", "Боб: Привет"]
    lines = (directory / "segments.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["segment_id"] for line in lines] == ["bob/1:1", "alice:1", "alice:2"]


def relative(segments: list[TranscriptSegment], start: datetime) -> list[tuple[str, float, float]]:
    return sorted(
        (s.speaker.id, (s.start - start).total_seconds(), (s.end - start).total_seconds())
        for s in segments
        if s.is_final
    )


async def test_saved_session_replays_the_same(tmp_path: Path) -> None:
    alice = make_wav(tmp_path / "a.wav", [(0.4, True), (0.6, False), (0.3, True), (0.2, False)])
    bob = make_wav(tmp_path / "b.wav", [(0.2, False), (0.5, True), (0.3, False)])
    sink = CollectingSink()
    saver = FilesystemSaver(FilesystemSaverParams(dir=tmp_path / "sessions"))
    pipeline = Pipeline(
        source=WavSource(
            WavSourceParams(
                files=[
                    WavFile(path=alice, speaker_id="alice"),
                    WavFile(path=bob, speaker_id="bob", offset_ms=700),
                ]
            )
        ),
        vad=EnergyVad(EnergyVadParams(min_silence_ms=300, pre_roll_ms=0)),
        recognizer=DummyRecognizer(DummyRecognizerParams()),
        sinks=[sink],
        savers=[saver],
    )
    await pipeline.run()

    # Replay: any config with the saved source in place of its own
    config_path = tmp_path / "replay-config.json"
    config_path.write_text(
        json.dumps(
            {
                "source": {"type": "livekit", "params": {"room": "unused"}},
                "vad": {"type": "energy", "params": {"min_silence_ms": 300, "pre_roll_ms": 0}},
                "recognizer": {"type": "dummy"},
            }
        ),
        encoding="utf-8",
    )
    source_path = session_dir(tmp_path / "sessions") / "source.json"
    config = load_config(config_path, source=source_path)
    assert config.source == ComponentConfig.model_validate_json(source_path.read_text("utf-8"))
    replay = build_pipeline(config)
    replay_sink = CollectingSink()
    replay._sinks = [replay_sink]
    await replay.run()
    # The replay session starts at its own moment: compare relative to the first phrase
    replayed = relative(replay_sink.segments, min(s.start for s in replay_sink.segments))
    recorded = relative(sink.segments, min(s.start for s in sink.segments))

    assert [r[0] for r in recorded] == [r[0] for r in replayed] == ["alice", "alice", "bob"]
    # Bob's phrase is where it was: offset 0.7 s + 0.2 s of silence in his file
    assert recorded[2][1:] == pytest.approx((0.9, 1.4), abs=0.05)
    assert [t for r in replayed for t in r[1:]] == pytest.approx(
        [t for r in recorded for t in r[1:]], abs=0.01
    )
