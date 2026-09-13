"""JSON transcript parser — Whisper / faster-whisper / Meetily exports.

Rather than hardcode one schema, we look for the first list of dicts that
carries recognisable text keys, then map fields by alias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..transcript import Transcript, Utterance

TEXT_KEYS = ("text", "transcript", "content", "sentence", "value", "body")
START_KEYS = ("start", "start_time", "startTime", "begin", "from", "offset", "ts")
END_KEYS = ("end", "end_time", "endTime", "stop", "to")
SPEAKER_KEYS = ("speaker", "speaker_label", "speakerName", "who", "name", "participant")


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Heuristic: values this large are milliseconds, not seconds.
        return float(value) / 1000 if value > 100_000 else float(value)
    if isinstance(value, str):
        try:
            parts = [float(p) for p in value.replace(",", ".").split(":")]
        except ValueError:
            return None
        total = 0.0
        for part in parts:
            total = total * 60 + part
        return total
    return None


def _find_segments(node: Any, depth: int = 0) -> list[dict] | None:
    """Depth-first search for the segment list, whatever it is nested under."""
    if depth > 6:
        return None
    if isinstance(node, list):
        rows = [r for r in node if isinstance(r, dict)]
        if rows and any(_first(r, TEXT_KEYS) for r in rows):
            return rows
        for item in node:
            found = _find_segments(item, depth + 1)
            if found:
                return found
    elif isinstance(node, dict):
        # Prefer conventionally-named containers before brute-forcing.
        for key in ("segments", "utterances", "transcript", "results", "items", "chunks"):
            if key in node:
                found = _find_segments(node[key], depth + 1)
                if found:
                    return found
        for value in node.values():
            found = _find_segments(value, depth + 1)
            if found:
                return found
    return None


def parse(raw: str, source: str = "json") -> Transcript:
    data = json.loads(raw)
    rows = _find_segments(data) or []

    utterances = []
    for row in rows:
        text = _first(row, TEXT_KEYS)
        if not isinstance(text, str) or not text.strip():
            continue
        speaker = _first(row, SPEAKER_KEYS)
        utterances.append(
            Utterance(
                text.strip(),
                _seconds(_first(row, START_KEYS)),
                _seconds(_first(row, END_KEYS)),
                str(speaker) if speaker else None,
            )
        )

    title = data.get("title") or data.get("name") if isinstance(data, dict) else None
    return Transcript(utterances, source=source, title=title)


def load(path: Path) -> Transcript:
    t = parse(path.read_text(encoding="utf-8", errors="replace"), source="json")
    t.title = t.title or path.stem
    return t
