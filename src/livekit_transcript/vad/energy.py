"""Loudness-threshold VAD. Crude (any noise is "speech"), but needs no model:
for debugging pipelines and tests. For real calls use a model-based VAD.
"""

from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import timedelta

import numpy as np
from pydantic import Field

from livekit_transcript.core.interfaces import ComponentParams, Vad
from livekit_transcript.core.models import (
    AudioFrame,
    SpeechEnd,
    SpeechEvent,
    SpeechFrames,
    SpeechStart,
)


class EnergyVadParams(ComponentParams):
    # Frame RMS level (dB relative to full scale) at which it counts as speech
    threshold_dbfs: float = Field(default=-45.0, le=0)
    # Silence that ends a phrase
    min_silence_ms: int = Field(default=500, gt=0)
    # Audio before the detected start that is added to the phrase, so the first sound isn't cut
    pre_roll_ms: int = Field(default=200, ge=0)


class EnergyVad(Vad):
    Params = EnergyVadParams

    def __init__(self, params: EnergyVadParams) -> None:
        self._params = params

    async def process(self, frames: AsyncIterator[AudioFrame]) -> AsyncGenerator[SpeechEvent]:
        min_silence = timedelta(milliseconds=self._params.min_silence_ms)
        max_pre_roll = timedelta(milliseconds=self._params.pre_roll_ms)
        pre_roll: deque[AudioFrame] = deque()
        pre_roll_duration = timedelta()
        in_speech = False
        silence = timedelta()
        last_loud_end = None

        async for frame in frames:
            loud = level_dbfs(frame) >= self._params.threshold_dbfs
            if not in_speech:
                if not loud:
                    pre_roll.append(frame)
                    pre_roll_duration += frame.duration
                    while pre_roll and pre_roll_duration > max_pre_roll:
                        pre_roll_duration -= pre_roll.popleft().duration
                    continue
                head = (*pre_roll, frame)
                pre_roll.clear()
                pre_roll_duration = timedelta()
                in_speech, silence, last_loud_end = True, timedelta(), frame.end
                yield SpeechStart(head[0].start)
                yield SpeechFrames(head)
                continue

            yield SpeechFrames((frame,))
            if loud:
                silence, last_loud_end = timedelta(), frame.end
                continue
            silence += frame.duration
            if silence >= min_silence:
                in_speech = False
                yield SpeechEnd(last_loud_end)

        if in_speech:
            yield SpeechEnd(last_loud_end)


def level_dbfs(frame: AudioFrame) -> float:
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return float("-inf")
    rms = np.sqrt(np.mean(samples**2))
    return float(20 * np.log10(rms / 32768)) if rms > 0 else float("-inf")
