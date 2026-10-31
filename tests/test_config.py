import json
from pathlib import Path

import pytest

from livekit_transcript.config import load_config
from livekit_transcript.core.interfaces import ConfigError
from livekit_transcript.registry import build_pipeline
from tests.helpers import make_wav


def write_config(path: Path, source: dict) -> Path:
    config = {
        "source": source,
        "vad": {"type": "energy"},
        "recognizer": {"type": "dummy"},
        "sinks": [{"type": "stdout", "params": {"format": "text"}}],
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def wav_source(path: Path) -> dict:
    return {"type": "wav", "params": {"files": [{"path": str(path), "speaker_id": "a"}]}}


def test_valid_wav_config_builds(tmp_path: Path) -> None:
    wav = make_wav(tmp_path / "a.wav", [(0.1, True)])

    build_pipeline(load_config(write_config(tmp_path / "c.json", wav_source(wav))))


def test_unknown_type_lists_available(tmp_path: Path) -> None:
    config = write_config(tmp_path / "c.json", {"type": "zoom"})

    with pytest.raises(ConfigError, match=r"source: неизвестный type 'zoom'.*livekit, wav"):
        build_pipeline(load_config(config))


def test_unknown_param_is_rejected(tmp_path: Path) -> None:
    wav = make_wav(tmp_path / "a.wav", [(0.1, True)])
    source = wav_source(wav)
    source["params"]["realtme"] = True

    with pytest.raises(ConfigError, match="realtme"):
        build_pipeline(load_config(write_config(tmp_path / "c.json", source)))


def test_room_override_is_rejected_by_wav_source(tmp_path: Path) -> None:
    wav = make_wav(tmp_path / "a.wav", [(0.1, True)])
    config = load_config(write_config(tmp_path / "c.json", wav_source(wav)), room="r1")

    with pytest.raises(ConfigError, match="room"):
        build_pipeline(config)


def test_room_override_reaches_livekit_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    path = write_config(tmp_path / "c.json", {"type": "livekit", "params": {"room": "old"}})

    pipeline = build_pipeline(load_config(path, room="new"))

    assert pipeline._source.session_name == "new"


def test_wav_with_wrong_rate_is_rejected(tmp_path: Path) -> None:
    wav = make_wav(tmp_path / "a.wav", [(0.1, True)], rate=48_000)

    with pytest.raises(ConfigError, match="48000 Гц"):
        build_pipeline(load_config(write_config(tmp_path / "c.json", wav_source(wav))))


def test_missing_wav_is_rejected(tmp_path: Path) -> None:
    config = write_config(tmp_path / "c.json", wav_source(tmp_path / "nope.wav"))

    with pytest.raises(ConfigError, match="nope.wav"):
        build_pipeline(load_config(config))


@pytest.mark.parametrize(
    ("url", "api_url", "rtc_url"),
    [
        (
            "https://sfu.example.com/livekit",
            "https://sfu.example.com/livekit",
            "wss://sfu.example.com/livekit",
        ),
        ("wss://sfu.example.com", "https://sfu.example.com", "wss://sfu.example.com"),
        ("http://localhost:7880", "http://localhost:7880", "ws://localhost:7880"),
    ],
)
def test_livekit_url_gives_api_and_rtc_urls(
    monkeypatch: pytest.MonkeyPatch, url: str, api_url: str, rtc_url: str
) -> None:
    from livekit_transcript.sources.livekit import LiveKitSettings

    settings = LiveKitSettings(livekit_url=url, livekit_api_key="k", livekit_api_secret="s")

    assert settings.url(websocket=False) == api_url
    assert settings.url(websocket=True) == rtc_url


@pytest.mark.parametrize(
    ("base", "link"),
    [
        ("http://localhost:5173", "http://localhost:5173/sessions/s-1"),
        ("https://ui.example.com/transcripts/", "https://ui.example.com/transcripts/sessions/s-1"),
        ("", None),
    ],
)
def test_transcript_view_url_links_to_session(base: str, link: str | None) -> None:
    from livekit_transcript.sources.livekit import LiveKitSettings

    settings = LiveKitSettings(
        livekit_url="https://sfu.example.com",
        livekit_api_key="k",
        livekit_api_secret="s",
        transcript_view_url=base,
    )

    assert settings.view_url("s-1") == link
