import json
import sys
from typing import Literal

from livekit_transcript.core.interfaces import ComponentParams, Sink
from livekit_transcript.core.models import TranscriptSegment


class StdoutSinkParams(ComponentParams):
    # jsonl — one JSON object per segment (for programs), text — "[hh:mm:ss] Name: text"
    format: Literal["jsonl", "text"] = "jsonl"
    interim: bool = True


class StdoutSink(Sink):
    Params = StdoutSinkParams

    def __init__(self, params: StdoutSinkParams) -> None:
        self._params = params

    async def on_segment(self, segment: TranscriptSegment) -> None:
        if not segment.is_final and not self._params.interim:
            return
        sys.stdout.write(self._format(segment) + "\n")
        sys.stdout.flush()

    def _format(self, segment: TranscriptSegment) -> str:
        if self._params.format == "jsonl":
            return json.dumps(segment.to_dict(), ensure_ascii=False)
        time = segment.start.astimezone().strftime("%H:%M:%S")
        mark = "" if segment.is_final else "… "
        return f"{mark}[{time}] {segment.speaker.name}: {segment.text}"
