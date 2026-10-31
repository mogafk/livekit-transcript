"""Data passed between components. Only the standard library here."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import uuid4

# The single audio format of the pipeline: sources convert to it, everyone else relies on it
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # bytes, PCM int16 little-endian
NUM_CHANNELS = 1


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class Speaker:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class AudioFrame:
    data: bytes
    # UTC wall-clock time of the first sample
    start: datetime

    @property
    def num_samples(self) -> int:
        return len(self.data) // SAMPLE_WIDTH

    @property
    def duration(self) -> timedelta:
        return timedelta(seconds=self.num_samples / SAMPLE_RATE)

    @property
    def end(self) -> datetime:
        return self.start + self.duration


@dataclass(frozen=True, slots=True)
class SpeakerTrack:
    speaker: Speaker
    # Ends when the speaker's track is gone (left the room, unpublished, end of file)
    frames: AsyncIterator[AudioFrame]


@dataclass(frozen=True, slots=True)
class SessionInfo:
    name: str
    started_at: datetime
    # Unique per run: the same room can be transcribed many times
    id: str = field(default_factory=lambda: str(uuid4()))


# VAD output: SpeechStart, then one or more SpeechFrames, then SpeechEnd — per phrase


@dataclass(frozen=True, slots=True)
class SpeechStart:
    time: datetime


@dataclass(frozen=True, slots=True)
class SpeechFrames:
    frames: tuple[AudioFrame, ...]


@dataclass(frozen=True, slots=True)
class SpeechEnd:
    time: datetime


type SpeechEvent = SpeechStart | SpeechFrames | SpeechEnd


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    speaker: Speaker
    # Stable for one phrase: interim segments with the same id are replaced by later ones
    segment_id: str
    is_final: bool
    start: datetime
    end: datetime
    text: str

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "speaker_id": self.speaker.id,
            "speaker_name": self.speaker.name,
            "segment_id": self.segment_id,
            "is_final": self.is_final,
            "start": format_utc(self.start),
            "end": format_utc(self.end),
            "text": self.text,
        }
