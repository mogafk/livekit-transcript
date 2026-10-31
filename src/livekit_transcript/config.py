"""Session config: which implementation of each component to use and with which params."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from livekit_transcript.core.interfaces import ConfigError


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    # Validated later by the chosen implementation's own Params model
    params: dict[str, Any] = Field(default_factory=dict)


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: ComponentConfig
    vad: ComponentConfig
    recognizer: ComponentConfig
    sinks: list[ComponentConfig] = Field(default_factory=list)
    savers: list[ComponentConfig] = Field(default_factory=list)


def load_config(
    path: Path, *, room: str | None = None, source: Path | None = None
) -> SessionConfig:
    """Reads the JSON config. `source` — a file with a `source` section replacing the config's
    one (e.g. `source.json` of a saved session); `room` overrides `source.params.room`.
    """
    config = _load(path, SessionConfig)
    if source is not None:
        config.source = _load(source, ComponentConfig)
    if room is not None:
        # A source without a `room` param rejects it during params validation
        config.source.params["room"] = room
    return config


def _load[T: BaseModel](path: Path, model: type[T]) -> T:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"Не удалось прочитать {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path} — невалидный JSON: {e}") from e
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"{path}:\n{e}") from e
