from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from aiohttp import web

from livekit_transcript.core.interfaces import ConfigError
from livekit_transcript.core.models import SessionInfo, Speaker, TranscriptSegment
from livekit_transcript.sinks import http
from livekit_transcript.sinks.http import HttpSink, HttpSinkParams

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
ALICE = Speaker(id="alice", name="Alice")
TOKEN = "test-token"


@dataclass
class FakeServer:
    url: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    auth: list[str] = field(default_factory=list)
    # Statuses to answer with, in order; then 204
    statuses: list[int] = field(default_factory=list)

    def events(self) -> list[dict[str, Any]]:
        return [e for r in self.requests for e in r["events"]]


@pytest.fixture
async def server(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeServer]:
    state = FakeServer(url="")

    async def handler(request: web.Request) -> web.Response:
        state.auth.append(request.headers.get("Authorization", ""))
        status = state.statuses.pop(0) if state.statuses else 204
        if status < 300:
            state.requests.append(await request.json())
        return web.Response(status=status)

    app = web.Application()
    app.router.add_post("/events", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    state.url = f"http://127.0.0.1:{port}/events"
    monkeypatch.setenv("HTTP_SINK_URL", state.url)
    monkeypatch.setenv("HTTP_SINK_TOKEN", TOKEN)
    # Retries in milliseconds, not seconds
    monkeypatch.setattr(http, "RETRY_MIN_S", 0.01)
    monkeypatch.setattr(http, "RETRY_MAX_S", 0.02)
    yield state
    await runner.cleanup()


def segment(n: int, *, final: bool, text: str = "hi") -> TranscriptSegment:
    start = T0 + timedelta(seconds=n)
    return TranscriptSegment(ALICE, f"alice:{n}", final, start, start + timedelta(seconds=1), text)


async def run_session(sink: HttpSink, *segments: TranscriptSegment) -> None:
    await sink.start()
    await sink.on_session_start(SessionInfo(name="room", started_at=T0, id="s-1"))
    await sink.on_track_start(ALICE)
    for s in segments:
        await sink.on_segment(s)
    await sink.on_track_end(ALICE)
    await sink.on_session_end()
    await sink.aclose()


async def test_events_arrive_in_order_with_session_and_token(server: FakeServer) -> None:
    await run_session(HttpSink(HttpSinkParams()), segment(1, final=True, text="Hello"))

    assert server.auth[0] == f"Bearer {TOKEN}"
    request = server.requests[0]
    assert request["version"] == 1
    assert request["session"] == {
        "id": "s-1",
        "name": "room",
        "started_at": "2026-01-01T12:00:00.000Z",
    }
    events = server.events()
    assert [(e["seq"], e["type"]) for e in events] == [
        (1, "session_started"),
        (2, "speaker_joined"),
        (3, "segment"),
        (4, "speaker_left"),
        (5, "session_ended"),
    ]
    assert events[1]["speaker"] == {"id": "alice", "name": "Alice"}
    assert events[2]["segment"]["text"] == "Hello"
    assert events[2]["segment"]["start"] == "2026-01-01T12:00:01.000Z"


@pytest.mark.parametrize("status", [500, 503, 429, 401])
async def test_failed_batch_is_retried_without_loss(server: FakeServer, status: int) -> None:
    server.statuses = [status, status]

    await run_session(HttpSink(HttpSinkParams()), segment(1, final=True), segment(2, final=True))

    assert len(server.auth) >= 3  # two failures, then delivered
    assert [e["seq"] for e in server.events()] == [1, 2, 3, 4, 5, 6]


async def test_rejected_batch_is_dropped_and_delivery_goes_on(server: FakeServer) -> None:
    server.statuses = [400]
    sink = HttpSink(HttpSinkParams(max_batch=2))

    await run_session(sink, segment(1, final=True))

    # The first batch (seq 1–2) is lost, the rest still arrives
    assert [e["seq"] for e in server.events()] == [3, 4, 5]


async def test_queued_interim_is_replaced_by_newer_version(server: FakeServer) -> None:
    # Not started: nothing is sent, so everything stays queued
    sink = HttpSink(HttpSinkParams())

    await sink.on_segment(segment(1, final=False, text="Hi"))
    await sink.on_segment(segment(2, final=False, text="Other"))
    await sink.on_segment(segment(1, final=False, text="Hi Bob"))
    await sink.on_segment(segment(1, final=True, text="Hi Bob, hear me?"))

    assert [(e["seq"], e["segment"]["text"]) for e in sink._pending] == [
        (2, "Other"),
        (4, "Hi Bob, hear me?"),
    ]


async def test_interim_off_sends_finals_only(server: FakeServer) -> None:
    await run_session(
        HttpSink(HttpSinkParams(interim=False)), segment(1, final=False), segment(1, final=True)
    )

    segments = [e["segment"] for e in server.events() if e["type"] == "segment"]
    assert [s["is_final"] for s in segments] == [True]


async def test_unreachable_server_gives_up_after_shutdown_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_SINK_URL", "http://127.0.0.1:9/events")  # nothing listens
    monkeypatch.setenv("HTTP_SINK_TOKEN", TOKEN)

    # Must return, not hang: events are reported as lost
    await run_session(HttpSink(HttpSinkParams(shutdown_timeout_s=0.3, timeout_s=0.2)))


@pytest.mark.parametrize(
    ("url", "params", "ok"),
    [
        ("https://display.example.com/events", {}, True),
        ("http://localhost:8787/events", {}, True),
        ("http://display.example.com/events", {}, False),
        ("http://display.example.com/events", {"allow_insecure": True}, True),
        ("ftp://display.example.com/events", {}, False),
    ],
)
def test_insecure_url_needs_explicit_permission(
    monkeypatch: pytest.MonkeyPatch, url: str, params: dict[str, bool], ok: bool
) -> None:
    monkeypatch.setenv("HTTP_SINK_URL", url)
    monkeypatch.setenv("HTTP_SINK_TOKEN", TOKEN)

    if ok:
        HttpSink(HttpSinkParams(**params))
    else:
        with pytest.raises(ConfigError):
            HttpSink(HttpSinkParams(**params))


def test_missing_env_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HTTP_SINK_URL", raising=False)
    monkeypatch.delenv("HTTP_SINK_TOKEN", raising=False)
    monkeypatch.chdir("/")  # no .env here

    with pytest.raises(ConfigError, match="HTTP_SINK_URL"):
        HttpSink(HttpSinkParams())
