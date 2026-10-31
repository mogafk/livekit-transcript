import argparse
import asyncio
import io
import logging
import sys
from pathlib import Path

from livekit_transcript.config import load_config
from livekit_transcript.core.interfaces import ConfigError
from livekit_transcript.core.pipeline import Pipeline
from livekit_transcript.registry import build_pipeline

logger = logging.getLogger("livekit_transcript")

# "livekit" is the Rust core of the SDK: connection steps at INFO
NOISY_LOGGERS = ("faster_whisper", "httpx", "httpx2", "huggingface_hub", "livekit")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="livekit-transcript",
        description="Транскрибация созвона по спикерам в реальном времени",
    )
    parser.add_argument("--config", type=Path, required=True, help="JSON-конфиг сессии")
    parser.add_argument("--room", help="имя комнаты, переопределяет source.params.room")
    parser.add_argument("--log-level", default="INFO", help="уровень логов в stderr")
    args = parser.parse_args(argv)

    # Redirected streams on Windows default to the ANSI code page, which can't hold Cyrillic
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    # stdout belongs to sinks, logs go to stderr
    logging.basicConfig(
        stream=sys.stderr,
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if logging.getLogger().level > logging.DEBUG:
        # Libraries log every phrase and HTTP request at INFO
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    try:
        pipeline = build_pipeline(load_config(args.config, room=args.room))
    except ConfigError as e:
        logger.error("%s", e)
        sys.exit(2)

    try:
        asyncio.run(_run(pipeline))
    except KeyboardInterrupt:
        logger.warning("Прервано без дожидания финальных результатов")
        sys.exit(130)
    except ConfigError as e:
        logger.error("%s", e)
        sys.exit(2)
    except Exception:
        logger.exception("Сессия завершилась с ошибкой")
        sys.exit(1)


async def _run(pipeline: Pipeline) -> None:
    # First Ctrl+C cancels this task (asyncio.run does that): we turn it into a graceful stop.
    # Second Ctrl+C makes asyncio.run raise KeyboardInterrupt and drop everything.
    task = asyncio.create_task(pipeline.run())
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None:
            current.uncancel()
        logger.info("Остановка: дожидаемся финальных результатов (повторный Ctrl+C — сразу)")
        pipeline.stop()
        await task
