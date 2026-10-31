"""Silero VAD (neural, streaming) via onnxruntime on CPU, without PyTorch.

The model is bundled: `silero_vad.onnx` next to this file, see `silero_vad.LICENSE`.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import numpy as np
import onnxruntime as ort
from pydantic import Field, model_validator

from livekit_transcript.core.interfaces import ComponentParams, Vad
from livekit_transcript.core.models import SAMPLE_RATE, SAMPLE_WIDTH, AudioFrame, SpeechEvent
from livekit_transcript.vad.segmenter import PhraseSegmenter

MODEL_PATH = Path(__file__).with_name("silero_vad.onnx")
# The model takes 512-sample windows at 16 kHz plus 64 samples of the previous window as context
WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64


class SileroVadParams(ComponentParams):
    # Speech probability at which a phrase starts
    threshold: float = Field(default=0.5, gt=0, lt=1)
    # ...and below which it counts as silence inside a phrase. None — threshold - 0.15,
    # like Silero's own VADIterator: the gap keeps it from flapping around one value
    neg_threshold: float | None = Field(default=None, gt=0, lt=1)
    min_silence_ms: int = Field(default=500, gt=0)
    pre_roll_ms: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def _check_thresholds(self) -> Self:
        if self.neg_threshold is not None and self.neg_threshold > self.threshold:
            raise ValueError("neg_threshold должен быть не больше threshold")
        return self


class SileroVad(Vad):
    Params = SileroVadParams

    def __init__(self, params: SileroVadParams) -> None:
        self._params = params
        self._neg_threshold = (
            params.neg_threshold
            if params.neg_threshold is not None
            else max(params.threshold - 0.15, 0.01)
        )
        self._session: ort.InferenceSession | None = None

    async def start(self) -> None:
        self._session = await asyncio.to_thread(_load_session)

    async def aclose(self) -> None:
        self._session = None

    async def process(self, frames: AsyncIterator[AudioFrame]) -> AsyncGenerator[SpeechEvent]:
        assert self._session is not None
        segmenter = PhraseSegmenter(
            min_silence=timedelta(milliseconds=self._params.min_silence_ms),
            pre_roll=timedelta(milliseconds=self._params.pre_roll_ms),
        )
        # Model state is per track; the session itself is shared and stateless
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
        window_bytes = WINDOW_SAMPLES * SAMPLE_WIDTH
        # Incoming frames (10–20 ms) are re-cut into model windows (32 ms)
        pending = bytearray()
        pending_start: datetime | None = None

        async for frame in frames:
            if not pending:
                pending_start = frame.start
            pending += frame.data
            while len(pending) >= window_bytes:
                assert pending_start is not None
                window = AudioFrame(bytes(pending[:window_bytes]), pending_start)
                del pending[:window_bytes]
                pending_start = window.end

                samples = np.frombuffer(window.data, dtype=np.int16).astype(np.float32) / 32768
                model_input = np.concatenate([context, samples[np.newaxis, :]], axis=1)
                # ~0.1 ms per window on CPU: cheaper inline than a hop to a thread
                output, state = self._session.run(
                    None, {"input": model_input, "state": state, "sr": sample_rate}
                )
                context = model_input[:, -CONTEXT_SAMPLES:]

                threshold = self._neg_threshold if segmenter.in_speech else self._params.threshold
                for event in segmenter.push(window, float(output[0, 0]) >= threshold):
                    yield event

        # A tail shorter than a window can't be classified: keep it only inside a phrase
        if pending and pending_start is not None:
            for event in segmenter.push(AudioFrame(bytes(pending), pending_start), speech=False):
                yield event
        for event in segmenter.flush():
            yield event


def _load_session() -> ort.InferenceSession:
    options = ort.SessionOptions()
    # Tiny model: threads only add overhead, and many tracks share one session
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
    )
