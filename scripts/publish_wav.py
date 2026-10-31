"""Plays WAV files into a LiveKit room as microphones: one participant per file, like a real call.
For testing the livekit source without people. Uses LIVEKIT_* from .env.

    uv run python scripts/publish_wav.py --room test
        samples/alice.wav:alice:Алиса samples/bob.wav:bob:Боб

The room is created via the Server API if it doesn't exist. Audio starts after `--delay`
seconds, so there is time to start the transcriber. `--delete-room` deletes the room at the end:
the transcriber then sees the room closed and finishes the session.
"""

import argparse
import asyncio
import io
import logging
import sys
import wave
from pathlib import Path

from livekit import api, rtc

from livekit_transcript.core.models import SAMPLE_RATE
from livekit_transcript.sources.livekit import LiveKitSettings

FRAME_MS = 10
logger = logging.getLogger("publish_wav")


async def connect(settings: LiveKitSettings, room_name: str, identity: str, name: str) -> rtc.Room:
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret.get_secret_value())
        .with_identity(identity)
        .with_name(name)
        .with_grants(
            api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=False)
        )
        .to_jwt()
    )
    room = rtc.Room()
    await room.connect(settings.url(websocket=True), token)
    return room


async def play(room: rtc.Room, path: Path, start_at: float) -> None:
    source = rtc.AudioSource(SAMPLE_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("microphone", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, options)

    loop = asyncio.get_running_loop()
    samples_per_frame = SAMPLE_RATE * FRAME_MS // 1000
    with wave.open(str(path), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    step = samples_per_frame * 2
    for i, offset in enumerate(range(0, len(pcm), step)):
        delay = start_at + i * FRAME_MS / 1000 - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        data = pcm[offset : offset + step].ljust(step, b"\0")
        await source.capture_frame(rtc.AudioFrame(data, SAMPLE_RATE, 1, samples_per_frame))
    await source.wait_for_playout()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--room", required=True)
    parser.add_argument("--delay", type=float, default=10.0, help="секунд до начала речи")
    parser.add_argument("--delete-room", action="store_true", help="удалить комнату в конце")
    parser.add_argument("files", nargs="+", help="path.wav:identity[:name]")
    args = parser.parse_args()

    settings = LiveKitSettings()
    speakers = []
    for spec in args.files:
        path, identity, *name = spec.split(":")
        speakers.append((Path(path), identity, name[0] if name else identity))

    lkapi = api.LiveKitAPI(
        settings.url(websocket=False),
        settings.livekit_api_key,
        settings.livekit_api_secret.get_secret_value(),
    )
    rooms: list[rtc.Room] = []
    try:
        # The SFU may have auto-create disabled; for an existing room this is a no-op
        await lkapi.room.create_room(api.CreateRoomRequest(name=args.room))
        for _, identity, name in speakers:
            rooms.append(await connect(settings, args.room, identity, name))
        logger.info("В комнате %s: %s", args.room, ", ".join(i for _, i, _ in speakers))
        logger.info("Речь начнётся через %.0f с", args.delay)
        start_at = asyncio.get_running_loop().time() + args.delay
        await asyncio.gather(
            *(
                play(room, path, start_at)
                for room, (path, _, _) in zip(rooms, speakers, strict=True)
            )
        )
        logger.info("Записи проиграны")
        await asyncio.sleep(1)
    finally:
        for room in rooms:
            await room.disconnect()
        if args.delete_room:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=args.room))
            logger.info("Комната %s удалена", args.room)
        await lkapi.aclose()


if __name__ == "__main__":
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("livekit").setLevel(logging.WARNING)
    asyncio.run(main())
