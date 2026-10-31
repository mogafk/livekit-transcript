"""Interfaces of the five component kinds. The pipeline and CLI work only through them."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from livekit_transcript.core.models import (
    AudioFrame,
    SessionInfo,
    Speaker,
    SpeakerTrack,
    SpeechEvent,
    TranscriptSegment,
)


class ConfigError(Exception):
    """Invalid session config or environment: reported to the user without a traceback."""


class ComponentParams(BaseModel):
    """Base for `params` of an implementation: unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Component:
    """Common lifecycle. The registry creates an implementation as `cls(cls.Params(...))`.

    `__init__` must be cheap and may only validate (files exist, env vars set):
    the whole config is checked before any `start()`. Heavy work (model loading,
    connecting) belongs to `start()`.
    """

    Params: ClassVar[type[ComponentParams]] = ComponentParams

    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class Source(Component, ABC):
    """Where speaker tracks come from."""

    # Used by savers for naming the session (room name, file set name)
    session_name: str

    @abstractmethod
    def tracks(self) -> AsyncIterator[SpeakerTrack]:
        """Yields tracks as speakers appear. Exhausted = end of the session."""

    @abstractmethod
    def stop(self) -> None:
        """Requests a graceful end: `tracks()` and every track's frames finish soon."""


class Vad(Component, ABC):
    @abstractmethod
    def process(self, frames: AsyncIterator[AudioFrame]) -> AsyncGenerator[SpeechEvent]:
        """Marks speech in one track. Called once per track, calls must not share state.

        Every SpeechStart is followed by SpeechEnd, including at the end of `frames`.
        """


class Recognizer(Component, ABC):
    @abstractmethod
    def transcribe(
        self, speaker: Speaker, events: AsyncIterator[SpeechEvent]
    ) -> AsyncGenerator[TranscriptSegment]:
        """Turns speech events of one track into segments. Called once per track.

        The model is shared by all calls: load it in `start()`, run inference off the event loop.
        """


class Sink(Component, ABC):
    """Realtime output: receives every segment, interim and final, as soon as it appears."""

    @abstractmethod
    async def on_segment(self, segment: TranscriptSegment) -> None: ...


class Saver(Component, ABC):
    """Saves the whole session. Writes incrementally, `finalize()` is called once at the end."""

    async def on_session_start(self, session: SessionInfo) -> None:
        return None

    async def on_track_start(self, speaker: Speaker) -> None:
        return None

    async def on_frame(self, speaker: Speaker, frame: AudioFrame) -> None:
        return None

    async def on_segment(self, segment: TranscriptSegment) -> None:
        """Final segments only."""
        return None

    async def on_track_end(self, speaker: Speaker) -> None:
        return None

    @abstractmethod
    async def finalize(self) -> None: ...
