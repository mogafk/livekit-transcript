"""Turns per-frame "is it speech" decisions into phrases. Shared by the VAD implementations:
they differ only in how they decide.
"""

from collections import deque
from datetime import datetime, timedelta

from livekit_transcript.core.models import (
    AudioFrame,
    SpeechEnd,
    SpeechEvent,
    SpeechFrames,
    SpeechStart,
)


class PhraseSegmenter:
    def __init__(self, min_silence: timedelta, pre_roll: timedelta) -> None:
        # Silence that ends a phrase
        self._min_silence = min_silence
        # Audio before the detected start added to the phrase, so the first sound isn't cut
        self._max_pre_roll = pre_roll
        self._pre_roll: deque[AudioFrame] = deque()
        self._pre_roll_duration = timedelta()
        self._in_speech = False
        self._silence = timedelta()
        self._last_speech_end: datetime | None = None

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def push(self, frame: AudioFrame, speech: bool) -> list[SpeechEvent]:
        if not self._in_speech:
            if not speech:
                self._pre_roll.append(frame)
                self._pre_roll_duration += frame.duration
                while self._pre_roll and self._pre_roll_duration > self._max_pre_roll:
                    self._pre_roll_duration -= self._pre_roll.popleft().duration
                return []
            head = (*self._pre_roll, frame)
            self._pre_roll.clear()
            self._pre_roll_duration = timedelta()
            self._in_speech, self._silence = True, timedelta()
            self._last_speech_end = frame.end
            return [SpeechStart(head[0].start), SpeechFrames(head)]

        events: list[SpeechEvent] = [SpeechFrames((frame,))]
        if speech:
            self._silence, self._last_speech_end = timedelta(), frame.end
            return events
        self._silence += frame.duration
        if self._silence >= self._min_silence:
            events.append(self._end())
        return events

    def flush(self) -> list[SpeechEvent]:
        """End of input: closes an unfinished phrase."""
        return [self._end()] if self._in_speech else []

    def _end(self) -> SpeechEnd:
        assert self._last_speech_end is not None
        self._in_speech = False
        return SpeechEnd(self._last_speech_end)
