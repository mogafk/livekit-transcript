# livekit-transcript

Транскрибация созвона в LiveKit SFU в реальном времени: каждый участник — отдельный трек,
поэтому реплики сразу привязаны к спикеру.

Состав pipeline (источник, VAD, распознаватель, выходы) задаётся JSON-конфигом,
см. `configs/` и раздел «Архитектура» в [AGENTS.md](AGENTS.md).

```bash
uv sync
uv run python -m livekit_transcript --config configs/wav.json
uv run python -m livekit_transcript --config configs/livekit.json --room <room_name>
```

Источник `wav` принимает WAV PCM 16 бит, 16 кГц, моно — по файлу на спикера:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le samples/alice.wav
```

Ctrl+C — корректная остановка с дожиданием финальных результатов, повторный Ctrl+C — сразу.
