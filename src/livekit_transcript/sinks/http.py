"""Sends session events to a display server with HTTP POST. Protocol: docs/http-sink.md.

One request in flight at a time and a failed batch is retried before anything newer,
so the server receives events in order; `seq` lets it drop duplicates of a retried batch.
"""

import asyncio
import logging
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from livekit_transcript.core.interfaces import ComponentParams, ConfigError, Sink
from livekit_transcript.core.models import (
    SessionInfo,
    Speaker,
    TranscriptSegment,
    format_utc,
    utc_now,
)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
RETRY_MIN_S = 0.5
RETRY_MAX_S = 30.0
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class HttpSinkSettings(BaseSettings):
    """The address and the token live in the environment / .env, never in the session config."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    http_sink_url: str
    http_sink_token: SecretStr


class HttpSinkParams(ComponentParams):
    # Send interim segments too (the display updates a phrase while it is spoken)
    interim: bool = True
    timeout_s: float = Field(default=5.0, gt=0)
    max_batch: int = Field(default=50, ge=1)
    # How long to keep delivering queued events on shutdown before giving up
    shutdown_timeout_s: float = Field(default=10.0, ge=0)
    # Plain http:// to a non-local host sends the token in clear text: only on request
    allow_insecure: bool = False


class HttpSink(Sink):
    Params = HttpSinkParams

    def __init__(self, params: HttpSinkParams) -> None:
        self._params = params
        try:
            self._settings = HttpSinkSettings()
        except ValidationError as e:
            raise ConfigError(
                "sink (http): задайте HTTP_SINK_URL и HTTP_SINK_TOKEN в окружении или в .env"
            ) from e
        url = urlsplit(self._settings.http_sink_url)
        if url.scheme not in {"http", "https"}:
            raise ConfigError("sink (http): HTTP_SINK_URL должен начинаться с http(s)://")
        if url.scheme == "http" and url.hostname not in LOCAL_HOSTS and not params.allow_insecure:
            raise ConfigError(
                "sink (http): токен по http:// уйдёт открытым текстом — используйте https:// "
                "или явно разрешите allow_insecure: true"
            )
        self._session: dict[str, str] | None = None
        self._seq = 0
        # Events not yet delivered, oldest first; a batch in flight is taken out of it
        self._pending: list[dict[str, Any]] = []
        self._wakeup = asyncio.Event()
        self._closing = False
        self._http: aiohttp.ClientSession | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._params.timeout_s),
            headers={
                "Authorization": f"Bearer {self._settings.http_sink_token.get_secret_value()}"
            },
        )
        self._task = asyncio.create_task(self._deliver(), name="http-sink")

    async def aclose(self) -> None:
        self._closing = True
        self._wakeup.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, self._params.shutdown_timeout_s)
            except TimeoutError:
                logger.warning("Sink http: не доставлено событий: %d", len(self._pending))
            self._task = None
        if self._http is not None:
            await self._http.close()
            self._http = None

    async def on_session_start(self, session: SessionInfo) -> None:
        self._session = {
            "id": session.id,
            "name": session.name,
            "started_at": format_utc(session.started_at),
        }
        self._add("session_started")

    async def on_track_start(self, speaker: Speaker) -> None:
        self._add("speaker_joined", speaker={"id": speaker.id, "name": speaker.name})

    async def on_segment(self, segment: TranscriptSegment) -> None:
        if not segment.is_final and not self._params.interim:
            return
        # A queued interim of the same phrase is outdated by this segment: don't send it
        self._pending = [
            event
            for event in self._pending
            if not (
                event["type"] == "segment"
                and event["segment"]["segment_id"] == segment.segment_id
                and not event["segment"]["is_final"]
            )
        ]
        self._add("segment", segment=segment.to_dict())

    async def on_track_end(self, speaker: Speaker) -> None:
        self._add("speaker_left", speaker={"id": speaker.id, "name": speaker.name})

    async def on_session_end(self) -> None:
        self._add("session_ended")

    def _add(self, event_type: str, **payload: Any) -> None:
        self._seq += 1
        self._pending.append(
            {"seq": self._seq, "type": event_type, "at": format_utc(utc_now()), **payload}
        )
        self._wakeup.set()

    async def _deliver(self) -> None:
        retry_delay = RETRY_MIN_S
        failing = False
        while True:
            if not self._pending:
                if self._closing:
                    return
                self._wakeup.clear()
                await self._wakeup.wait()
                continue

            batch = self._pending[: self._params.max_batch]
            self._pending = self._pending[len(batch) :]
            error = await self._post(batch)
            if error is None:
                if failing:
                    logger.info("Sink http: доставка восстановлена")
                failing, retry_delay = False, RETRY_MIN_S
                continue

            # Retry this batch before anything newer, so the order holds
            self._pending = batch + self._pending
            if not failing:
                logger.warning("Sink http: %s, повторяем", error)
                failing = True
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, RETRY_MAX_S)

    async def _post(self, batch: list[dict[str, Any]]) -> str | None:
        """None — delivered (or rejected for good and dropped); otherwise a retryable error."""
        assert self._http is not None and self._session is not None
        body = {"version": PROTOCOL_VERSION, "session": self._session, "events": batch}
        try:
            async with self._http.post(self._settings.http_sink_url, json=body) as response:
                if response.ok:
                    return None
                text = (await response.text())[:200]
                if response.status in {401, 403}:
                    # Fixed on the server or in .env, not in the data: keep the events
                    return f"сервер не принял токен ({response.status})"
                if response.status in {408, 429} or response.status >= 500:
                    return f"сервер ответил {response.status}"
                # 4xx won't fix itself on retry: a retry loop would block every later event
                logger.error(
                    "Sink http: сервер отклонил %d событий (%d): %s",
                    len(batch),
                    response.status,
                    text,
                )
                return None
        except (aiohttp.ClientError, TimeoutError) as e:
            return f"сервер недоступен ({type(e).__name__}: {e})"
