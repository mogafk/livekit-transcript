"""Debug display server for the http sink: accepts the protocol from docs/http-sink.md
and prints the transcript. Also a reference for the server-side rules (duplicates, final wins).

    uv run python scripts/http_sink_server.py --port 8787

Token — HTTP_SINK_TOKEN from .env; in .env of the transcriber:
HTTP_SINK_URL=http://localhost:8787/events
"""

import argparse
import hmac
import io
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from livekit_transcript.sinks.http import HttpSinkSettings

logger = logging.getLogger("http_sink_server")


@dataclass
class Session:
    name: str
    last_seq: int = 0
    finals: set[str] = field(default_factory=set)


sessions: dict[str, Session] = {}


def handle_event(session: Session, event: dict[str, Any]) -> None:
    match event["type"]:
        case "session_started":
            logger.info("[%s] сессия началась", session.name)
        case "speaker_joined":
            logger.info("[%s] + %s", session.name, event["speaker"]["name"])
        case "speaker_left":
            logger.info("[%s] - %s", session.name, event["speaker"]["name"])
        case "session_ended":
            logger.info("[%s] сессия завершена", session.name)
        case "segment":
            segment = event["segment"]
            if segment["segment_id"] in session.finals:
                return  # final wins
            if segment["is_final"]:
                session.finals.add(segment["segment_id"])
            mark = "" if segment["is_final"] else "… "
            logger.info(
                "[%s] %s%s: %s", session.name, mark, segment["speaker_name"], segment["text"]
            )
        case _:
            pass  # unknown types are allowed by the protocol


async def events(request: web.Request) -> web.Response:
    auth = request.headers.get("Authorization", "")
    if not hmac.compare_digest(auth.encode(), f"Bearer {request.app['token']}".encode()):
        return web.Response(status=401, text="bad token")
    body = await request.json()
    if body.get("version") != 1:
        return web.Response(status=400, text="unsupported version")
    info = body["session"]
    session = sessions.setdefault(info["id"], Session(name=info["name"]))
    for event in body["events"]:
        if event["seq"] <= session.last_seq:
            continue  # duplicate of a retried batch
        session.last_seq = event["seq"]
        handle_event(session, event)
    return web.Response(status=204)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info("Слушаю http://%s:%d/events", args.host, args.port)

    app = web.Application()
    app["token"] = HttpSinkSettings().http_sink_token.get_secret_value()
    app.router.add_post("/events", events)
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
