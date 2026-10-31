"""Maps config `type` to implementations. The only place that knows them by name.

Modules are imported lazily: only the implementations chosen in the config,
so heavy dependencies of the others don't have to be installed.
"""

from importlib import import_module

from pydantic import ValidationError

from livekit_transcript.config import ComponentConfig, SessionConfig
from livekit_transcript.core.interfaces import (
    Component,
    ConfigError,
    Recognizer,
    Saver,
    Sink,
    Source,
    Vad,
)
from livekit_transcript.core.pipeline import Pipeline

IMPLEMENTATIONS: dict[type[Component], dict[str, str]] = {
    Source: {
        "wav": "livekit_transcript.sources.wav:WavSource",
        "livekit": "livekit_transcript.sources.livekit:LiveKitSource",
    },
    Vad: {
        "energy": "livekit_transcript.vad.energy:EnergyVad",
        "silero": "livekit_transcript.vad.silero:SileroVad",
    },
    Recognizer: {
        "dummy": "livekit_transcript.recognizers.dummy:DummyRecognizer",
        "faster-whisper": "livekit_transcript.recognizers.faster_whisper:FasterWhisperRecognizer",
    },
    Sink: {
        "stdout": "livekit_transcript.sinks.stdout:StdoutSink",
    },
    Saver: {},
}


def create[T: Component](kind: type[T], config: ComponentConfig, where: str) -> T:
    """Validates params and instantiates the implementation. `where` names it in errors."""
    known = IMPLEMENTATIONS[kind]
    target = known.get(config.type)
    if target is None:
        available = ", ".join(sorted(known)) or "нет"
        raise ConfigError(f"{where}: неизвестный type {config.type!r}, доступны: {available}")
    module_name, class_name = target.split(":")
    cls = getattr(import_module(module_name), class_name)
    if not issubclass(cls, kind):
        raise TypeError(f"{target} не реализует {kind.__name__}")
    try:
        params = cls.Params.model_validate(config.params)
    except ValidationError as e:
        raise ConfigError(f"{where} ({config.type}): неверные params:\n{e}") from e
    return cls(params)


def build_pipeline(config: SessionConfig) -> Pipeline:
    """Creates every component before anything starts, so config errors surface immediately."""
    return Pipeline(
        source=create(Source, config.source, "source"),
        vad=create(Vad, config.vad, "vad"),
        recognizer=create(Recognizer, config.recognizer, "recognizer"),
        sinks=[create(Sink, sink, f"sinks[{i}]") for i, sink in enumerate(config.sinks)],
        savers=[create(Saver, saver, f"savers[{i}]") for i, saver in enumerate(config.savers)],
    )
