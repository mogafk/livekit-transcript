"""LiveKit SFU as a source. NOT IMPLEMENTED YET: params, settings and the planned behavior only.

Plan (see "Источник livekit" in AGENTS.md):
- start(): generate a token with `livekit-api` (`AccessToken`: room_join to this room only,
  can_subscribe, no publishing, hidden, short TTL) and connect with the `livekit` rtc SDK;
- subscribe to MICROPHONE audio of every participant, including late joiners;
  ignore SCREEN_SHARE_AUDIO, ourselves and agents (ParticipantKind.AGENT);
- one SpeakerTrack per participant, speaker id = identity; frames via
  `rtc.AudioStream(track, sample_rate=16000, num_channels=1)` with UTC arrival time;
- track unpublished / participant left -> its frames end; mute -> just no frames;
- room closed or disconnected -> tracks() ends; stop() -> disconnect gracefully.
"""

from collections.abc import AsyncIterator

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from livekit_transcript.core.interfaces import ComponentParams, ConfigError, Source
from livekit_transcript.core.models import SpeakerTrack


class LiveKitSettings(BaseSettings):
    """Secrets live in the environment / .env, never in the session config."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: SecretStr


class LiveKitSourceParams(ComponentParams):
    room: str = Field(min_length=1)
    identity: str = Field(default="transcriber", min_length=1)


class LiveKitSource(Source):
    Params = LiveKitSourceParams

    def __init__(self, params: LiveKitSourceParams) -> None:
        self._params = params
        self.session_name = params.room
        try:
            self._settings = LiveKitSettings()
        except ValidationError as e:
            raise ConfigError(
                "source (livekit): задайте LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET "
                "в окружении или в .env (шаблон — .env.example)"
            ) from e

    async def start(self) -> None:
        raise NotImplementedError("Источник livekit ещё не реализован")

    async def tracks(self) -> AsyncIterator[SpeakerTrack]:
        raise NotImplementedError("Источник livekit ещё не реализован")
        yield  # makes this an async generator, like the real implementation

    def stop(self) -> None:
        return None
