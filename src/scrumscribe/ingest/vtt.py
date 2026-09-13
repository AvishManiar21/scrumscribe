"""WebVTT parser — the format Microsoft Teams exports transcripts in."""

from __future__ import annotations

import re
from pathlib import Path

from ..transcript import Transcript, Utterance

# 00:01:02.345 --> 00:01:05.678  (hours optional)
_CUE = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)
# Teams wraps speaker names in a voice span: <v Avish Maniar>text</v>
_VOICE = re.compile(r"<v\s+([^>]+?)\s*>(.*?)(?:</v>|$)", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
# Fallback speaker prefix: "Avish Maniar: text"
_PREFIX = re.compile(r"^([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3}):\s+(.*)", re.DOTALL)


def _seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse(text: str, source: str = "vtt") -> Transcript:
    utterances: list[Utterance] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        match = _CUE.search(lines[i])
        if not match:
            i += 1
            continue

        g = match.groups()
        start, end = _seconds(*g[0:4]), _seconds(*g[4:8])

        # Cue payload runs until the next blank line.
        i += 1
        payload: list[str] = []
        while i < len(lines) and lines[i].strip():
            payload.append(lines[i])
            i += 1
        raw = " ".join(payload).strip()
        if not raw:
            continue

        speaker = None
        voice = _VOICE.search(raw)
        if voice:
            speaker, raw = voice.group(1).strip(), voice.group(2)
        raw = _TAG.sub("", raw).strip()

        if speaker is None:
            prefixed = _PREFIX.match(raw)
            if prefixed:
                speaker, raw = prefixed.group(1).strip(), prefixed.group(2).strip()

        if raw:
            utterances.append(Utterance(raw, start, end, speaker))

    return Transcript(utterances, source=source)


def load(path: Path) -> Transcript:
    t = parse(path.read_text(encoding="utf-8", errors="replace"), source="vtt")
    t.title = path.stem
    return t
