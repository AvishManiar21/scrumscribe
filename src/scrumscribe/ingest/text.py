"""Plain-text transcript parser.

Handles the 'Speaker: line' convention and falls back to treating each
paragraph as an untimed utterance.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..transcript import Transcript, Utterance

_SPEAKER = re.compile(r"^\s*(?:\[(?P<ts>[\d:.]+)\]\s*)?(?P<who>[^:\n]{1,40}?):\s+(?P<text>.+)$")


def _to_seconds(stamp: str | None) -> float | None:
    if not stamp:
        return None
    try:
        parts = [float(p) for p in stamp.split(":")]
    except ValueError:
        return None
    total = 0.0
    for part in parts:
        total = total * 60 + part
    return total


def parse(text: str, source: str = "text") -> Transcript:
    utterances: list[Utterance] = []

    for block in re.split(r"\n\s*\n", text):
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            match = _SPEAKER.match(line)
            if match:
                start = _to_seconds(match.group("ts"))
                utterances.append(
                    Utterance(match.group("text").strip(), start, None, match.group("who").strip())
                )
            elif utterances:
                # Continuation of the previous speaker's turn.
                utterances[-1].text += " " + line
            else:
                utterances.append(Utterance(line))

    return Transcript(utterances, source=source)


def load(path: Path) -> Transcript:
    t = parse(path.read_text(encoding="utf-8", errors="replace"), source="text")
    t.title = path.stem
    return t
