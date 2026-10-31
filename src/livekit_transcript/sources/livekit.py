"""LiveKit SFU as a source: a hidden participant that subscribes to every speaker's microphone."""

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from datetime import timedelta
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from livekit import api, rtc
from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from livekit_transcript.core.interfaces import ComponentParams, ConfigError, Source
from livekit_transcript.core.models import (
    NUM_CHANNELS,
    SAMPLE_RATE,
    AudioFrame,
    SessionInfo,
    Speaker,
    SpeakerTrack,
    utc_now,
)

logger = logging.getLogger(__name__)

# The token is only for connecting: while connected the SFU refreshes it itself
TOKEN_TTL = timedelta(minutes=10)
# How long to wait for the SDK to close an audio stream on shutdown
STREAM_CLOSE_TIMEOUT_S = 2.0
# Chat topic of LiveKit components (LiveKit Meet) for text streams
CHAT_TOPIC = "lk.chat"
CHAT_SEND_TIMEOUT_S = 5.0


class LiveKitSettings(BaseSettings):
    """Secrets live in the environment / .env, never in the session config."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Any of http(s):// or ws(s)://, may include a reverse-proxy path
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: SecretStr
    # Base URL of the display UI; the chat announcement links to {url}/sessions/{session id}
    transcript_view_url: str | None = None

    @field_validator("livekit_url")
    @classmethod
    def _check_scheme(cls, value: str) -> str:
        if urlsplit(value).scheme not in {"http", "https", "ws", "wss"}:
            raise ValueError("ожидается http(s):// или ws(s)://")
        return value

    @field_validator("transcript_view_url")
    @classmethod
    def _check_view_url(cls, value: str | None) -> str | None:
        if not value:
            return None
        if urlsplit(value).scheme not in {"http", "https"}:
            raise ValueError("ожидается http(s)://")
        return value.rstrip("/")

    def view_url(self, session_id: str) -> str | None:
        if self.transcript_view_url is None:
            return None
        return f"{self.transcript_view_url}/sessions/{session_id}"

    def url(self, *, websocket: bool) -> str:
        parts = urlsplit(self.livekit_url)
        secure = parts.scheme in {"https", "wss"}
        scheme = ("wss" if secure else "ws") if websocket else ("https" if secure else "http")
        return urlunsplit(parts._replace(scheme=scheme))


class LiveKitSourceParams(ComponentParams):
    room: str = Field(min_length=1)
    # A random suffix is appended: a second participant with the same identity would kick us out
    identity: str = Field(default="transcriber", min_length=1)
    # Tell participants in the room chat that they are transcribed (and where to watch)
    announce: bool = True


class _TrackFeed:
    """Frames of one subscribed track.

    Ends by our own signal (unsubscribed, stop, disconnect) rather than relying on the SDK's
    end-of-stream: `rtc.AudioStream.aclose()` waits for it and may hang without it.
    """

    def __init__(self, track: rtc.Track, speaker: Speaker) -> None:
        self._speaker = speaker
        self._stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
        self._queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._ended = False
        self._pump = asyncio.create_task(self._run(), name=f"livekit-audio-{speaker.id}")

    async def _run(self) -> None:
        try:
            async for event in self._stream:
                frame = event.frame
                # The frame has just been received, so it started `duration` ago
                duration = timedelta(seconds=frame.samples_per_channel / frame.sample_rate)
                self._queue.put_nowait(AudioFrame(bytes(frame.data), utc_now() - duration))
        except Exception:
            logger.exception("Спикер %s: ошибка чтения аудио", self._speaker.id)
        finally:
            self.end()

    def end(self) -> None:
        if not self._ended:
            self._ended = True
            self._queue.put_nowait(None)

    async def frames(self) -> AsyncIterator[AudioFrame]:
        while (frame := await self._queue.get()) is not None:
            yield frame

    async def aclose(self) -> None:
        self.end()
        self._pump.cancel()
        await asyncio.wait([self._pump])
        try:
            await asyncio.wait_for(self._stream.aclose(), STREAM_CLOSE_TIMEOUT_S)
        except TimeoutError:
            logger.debug("Спикер %s: аудиопоток SDK не закрылся вовремя", self._speaker.id)


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
                f"в окружении или в .env (шаблон — .env.example):\n{e}"
            ) from e
        self._room: rtc.Room | None = None
        self._tracks: asyncio.Queue[SpeakerTrack | None] = asyncio.Queue()
        # By track sid; ended feeds stay here until aclose() releases their SDK streams
        self._feeds: dict[str, _TrackFeed] = {}
        self._finished = False
        # Set at session start; participants joining later get it personally
        self._announcement: str | None = None
        self._chat_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        await self._check_room_exists()
        identity = f"{self._params.identity}-{secrets.token_hex(3)}"
        token = (
            api.AccessToken(
                self._settings.livekit_api_key,
                self._settings.livekit_api_secret.get_secret_value(),
            )
            .with_identity(identity)
            .with_name("Transcriber")
            .with_ttl(TOKEN_TTL)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self._params.room,
                    can_subscribe=True,
                    can_publish=False,
                    # Only for the chat announcement
                    can_publish_data=self._params.announce,
                    hidden=True,
                )
            )
            .to_jwt()
        )
        room = rtc.Room()
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("disconnected", self._on_disconnected)
        room.on("participant_connected", self._on_participant_connected)
        room.on("reconnecting", lambda: logger.warning("Связь с SFU потеряна, переподключаемся"))
        room.on("reconnected", lambda: logger.info("Связь с SFU восстановлена"))
        try:
            await room.connect(self._settings.url(websocket=True), token)
        except rtc.ConnectError as e:
            raise ConfigError(f"source (livekit): не удалось подключиться к комнате: {e}") from e
        self._room = room
        logger.info(
            "Подключены к комнате %s как %s (скрытый участник)", self._params.room, identity
        )

    async def _check_room_exists(self) -> None:
        """Joining a missing room would silently create an empty one, so check first."""
        try:
            async with api.LiveKitAPI(
                self._settings.url(websocket=False),
                self._settings.livekit_api_key,
                self._settings.livekit_api_secret.get_secret_value(),
            ) as lkapi:
                response = await lkapi.room.list_rooms(
                    api.ListRoomsRequest(names=[self._params.room])
                )
        except api.TwirpError as e:
            raise ConfigError(f"source (livekit): SFU отклонил запрос: {e.message}") from e
        except (aiohttp.ClientError, OSError) as e:
            raise ConfigError(f"source (livekit): SFU недоступен: {e}") from e
        if not response.rooms:
            raise ConfigError(f"source (livekit): комнаты {self._params.room!r} нет на SFU")

    async def on_session_start(self, session: SessionInfo) -> None:
        if not self._params.announce:
            return
        url = self._settings.view_url(session.id)
        if url is None:
            logger.info("TRANSCRIPT_VIEW_URL не задан: сообщение в чат будет без ссылки")
            self._announcement = "Audio transcription is on."
        else:
            self._announcement = f"Audio transcription is on. Live transcript: {url}"
        await self._send_chat(self._announcement)
        logger.info("В чат комнаты отправлено сообщение о транскрипции")

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        # Chat history isn't replayed to newcomers, so tell each of them personally
        if (
            self._announcement is None
            or self._finished
            or participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        ):
            return
        task = asyncio.create_task(
            self._send_chat_logged(self._announcement, participant.identity),
            name=f"livekit-chat-{participant.identity}",
        )
        self._chat_tasks.add(task)
        task.add_done_callback(self._chat_tasks.discard)

    async def _send_chat(self, text: str, identity: str | None = None) -> None:
        assert self._room is not None
        await asyncio.wait_for(
            self._room.local_participant.send_text(
                text,
                topic=CHAT_TOPIC,
                destination_identities=[identity] if identity else None,
                # Browser clients may not support compressed streams
                compress=False,
            ),
            CHAT_SEND_TIMEOUT_S,
        )

    async def _send_chat_logged(self, text: str, identity: str) -> None:
        try:
            await self._send_chat(text, identity)
        except Exception:
            logger.exception("Не удалось отправить сообщение в чат участнику %s", identity)

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            self._finished
            or track.kind != rtc.TrackKind.KIND_AUDIO
            or publication.source != rtc.TrackSource.SOURCE_MICROPHONE
            or participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        ):
            return
        speaker = Speaker(id=participant.identity, name=participant.name or participant.identity)
        feed = _TrackFeed(track, speaker)
        self._feeds[publication.sid] = feed
        self._tracks.put_nowait(SpeakerTrack(speaker, feed.frames()))

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        feed = self._feeds.get(publication.sid)
        if feed is not None:
            feed.end()

    def _on_disconnected(self, reason: rtc.DisconnectReason.ValueType) -> None:
        if not self._finished:
            logger.info("Отключены от комнаты: %s", rtc.DisconnectReason.Name(reason))
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        for feed in self._feeds.values():
            feed.end()
        self._tracks.put_nowait(None)

    async def tracks(self) -> AsyncIterator[SpeakerTrack]:
        while (track := await self._tracks.get()) is not None:
            yield track

    def stop(self) -> None:
        self._finish()

    async def aclose(self) -> None:
        self._finish()
        for task in self._chat_tasks:
            task.cancel()
        await asyncio.gather(*self._chat_tasks, return_exceptions=True)
        for feed in self._feeds.values():
            await feed.aclose()
        self._feeds.clear()
        if self._room is not None:
            await self._room.disconnect()
            self._room = None
