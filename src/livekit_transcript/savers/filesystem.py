"""Saves the session to a directory, so a call can be replayed through the wav source:

<dir>/<session>-<local start time>/
    <speaker_id>.wav   raw audio of the speaker (before VAD), from their first frame
    segments.jsonl     final segments as they appear
    transcript.txt     all speakers by time, written at the end
    source.json        `source` section for the wav source: --source <path>
"""

import json
import logging
import re
import wave
from datetime import datetime, timedelta
from io import TextIOWrapper
from pathlib import Path

from pydantic import Field

from livekit_transcript.core.interfaces import ComponentParams, Saver
from livekit_transcript.core.models import (
    NUM_CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    AudioFrame,
    SessionInfo,
    Speaker,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

# How often the WAV header is updated: a crashed process leaves a readable file
HEADER_UPDATE_SAMPLES = SAMPLE_RATE


class FilesystemSaverParams(ComponentParams):
    dir: Path = Path("sessions")
    # A gap between frames longer than this (mute, reconnect) is filled with silence,
    # so the file stays on the session timeline. Shorter ones are jitter of arrival-time
    # timestamps: filling them would make the file drift ahead of real time
    min_gap_ms: int = Field(default=500, gt=0)


class _SpeakerAudio:
    def __init__(self, path: Path, speaker: Speaker) -> None:
        self.path = path
        self.speaker = speaker
        self.first_start: datetime | None = None
        self._samples = 0
        self._since_header_update = 0
        # Open for the whole session, closed in close()
        self._wav: wave.Wave_write | None = wave.open(str(path), "wb")  # noqa: SIM115
        self._wav.setnchannels(NUM_CHANNELS)
        self._wav.setsampwidth(SAMPLE_WIDTH)
        self._wav.setframerate(SAMPLE_RATE)

    def write(self, frame: AudioFrame, min_gap: timedelta) -> None:
        if self.first_start is None:
            self.first_start = frame.start
        else:
            expected = self.first_start + timedelta(seconds=self._samples / SAMPLE_RATE)
            gap = frame.start - expected
            if gap >= min_gap:
                self._write(bytes(int(gap.total_seconds() * SAMPLE_RATE) * SAMPLE_WIDTH))
        self._write(frame.data)

    def _write(self, data: bytes) -> None:
        assert self._wav is not None
        self._wav.writeframesraw(data)
        written = len(data) // SAMPLE_WIDTH
        self._samples += written
        self._since_header_update += written
        if self._since_header_update >= HEADER_UPDATE_SAMPLES:
            # writeframes() patches the header with the current length
            self._wav.writeframes(b"")
            self._since_header_update = 0

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None


class FilesystemSaver(Saver):
    Params = FilesystemSaverParams

    def __init__(self, params: FilesystemSaverParams) -> None:
        self._params = params
        self._min_gap = timedelta(milliseconds=params.min_gap_ms)
        self._session: SessionInfo | None = None
        self._dir: Path | None = None
        self._audio: dict[str, _SpeakerAudio] = {}
        self._segments: list[TranscriptSegment] = []
        self._segments_file: TextIOWrapper | None = None

    async def on_session_start(self, session: SessionInfo) -> None:
        self._session = session
        base = f"{_safe_name(session.name)}-{session.started_at.astimezone():%Y-%m-%dT%H-%M-%S}"
        self._dir = self._params.dir / base
        suffix = 1
        while self._dir.exists():
            suffix += 1
            self._dir = self._params.dir / f"{base}-{suffix}"
        self._dir.mkdir(parents=True)
        self._segments_file = (self._dir / "segments.jsonl").open("w", encoding="utf-8")
        logger.info("Сессия сохраняется в %s", self._dir)

    async def on_track_start(self, speaker: Speaker) -> None:
        # A rejoined speaker continues their file: the gap is filled by timestamps
        if speaker.id in self._audio:
            return
        assert self._dir is not None
        taken = {audio.path.name for audio in self._audio.values()}
        name, n = _safe_name(speaker.id), 1
        while f"{name}.wav" in taken:
            n += 1
            name = f"{_safe_name(speaker.id)}-{n}"
        self._audio[speaker.id] = _SpeakerAudio(self._dir / f"{name}.wav", speaker)

    async def on_frame(self, speaker: Speaker, frame: AudioFrame) -> None:
        audio = self._audio[speaker.id]
        first = audio.first_start is None
        audio.write(frame, self._min_gap)
        if first:
            # Keep source.json usable even if the process dies before finalize()
            self._write_source()

    async def on_segment(self, segment: TranscriptSegment) -> None:
        assert self._segments_file is not None
        self._segments.append(segment)
        self._segments_file.write(json.dumps(segment.to_dict(), ensure_ascii=False) + "\n")
        self._segments_file.flush()

    async def finalize(self) -> None:
        if self._dir is None:
            return
        self._close_files()
        self._write_source()
        lines = [
            f"[{s.start.astimezone():%H:%M:%S}] {s.speaker.name}: {s.text}"
            for s in sorted(self._segments, key=lambda s: s.start)
            # An empty final only closes interims of a phrase that turned out to be silence
            if s.text
        ]
        (self._dir / "transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(
            "Сессия сохранена в %s, повторить: --source %s", self._dir, self._dir / "source.json"
        )

    async def aclose(self) -> None:
        self._close_files()

    def _close_files(self) -> None:
        for audio in self._audio.values():
            audio.close()
        if self._segments_file is not None:
            self._segments_file.close()
            self._segments_file = None

    def _write_source(self) -> None:
        assert self._dir is not None and self._session is not None
        files = [
            {
                "path": str(audio.path.resolve()),
                "speaker_id": audio.speaker.id,
                "speaker_name": audio.speaker.name,
                "offset_ms": max(
                    0, round((audio.first_start - self._session.started_at).total_seconds() * 1000)
                ),
            }
            for audio in self._audio.values()
            if audio.first_start is not None
        ]
        if not files:
            return
        source = {
            "type": "wav",
            "params": {"files": files, "realtime": False, "session_name": self._session.name},
        }
        (self._dir / "source.json").write_text(
            json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]", "_", value) or "_"
