"""Meetily recording-folder reader.

Meetily writes each session to its own directory::

    Meeting 13_09_26_14_06_14_2026-09-13_18-06/
        audio.mp4          the captured audio, mic and system mixed
        metadata.json      devices, timestamps, status
        transcripts.json   {"segments": [...], "total_segments": N}

This is the path that matters in practice. The SQLite store is only
populated once a meeting is saved in the app -- a recording that was never
saved leaves ``meeting_id: null`` in its metadata and every database table
empty, while the folder on disk holds the full transcript. Reading the folder
works either way, and it works while the app is still open, which the database
does not reliably do.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..transcript import Transcript
from . import json_ingest

METADATA = "metadata.json"
TRANSCRIPTS = "transcripts.json"


def is_recording_dir(path: Path) -> bool:
    """True when a directory looks like one Meetily recording."""
    path = Path(path)
    if not path.is_dir():
        return False
    return (path / TRANSCRIPTS).is_file() or (path / METADATA).is_file()


def is_recordings_root(path: Path) -> bool:
    """True when a directory contains recording directories."""
    path = Path(path)
    if not path.is_dir():
        return False
    return any(is_recording_dir(child) for child in path.iterdir() if child.is_dir())


def list_recordings(root: Path) -> list[Path]:
    """Every recording under a root directory, newest first."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = [c for c in root.iterdir() if c.is_dir() and is_recording_dir(c)]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def read_metadata(path: Path) -> dict:
    meta_path = Path(path) / METADATA
    if not meta_path.is_file():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def describe(path: Path) -> dict:
    """Summarise a recording without parsing the whole transcript."""
    path = Path(path)
    meta = read_metadata(path)

    segments = 0
    transcripts = path / TRANSCRIPTS
    if transcripts.is_file():
        try:
            payload = json.loads(transcripts.read_text(encoding="utf-8"))
            segments = int(payload.get("total_segments") or len(payload.get("segments") or []))
        except (ValueError, OSError, TypeError):
            segments = 0

    audio = path / (meta.get("audio_file") or "audio.mp4")
    return {
        "path": path,
        "name": meta.get("meeting_name") or path.name,
        "created": (meta.get("created_at") or "")[:19],
        "status": meta.get("status") or "unknown",
        "segments": segments,
        "audio_bytes": audio.stat().st_size if audio.is_file() else 0,
        "saved_to_db": bool(meta.get("meeting_id")),
        "devices": meta.get("devices") or {},
    }


def load(path: Path) -> Transcript:
    """Load one recording directory as a Transcript."""
    path = Path(path)
    transcripts = path / TRANSCRIPTS
    if not transcripts.is_file():
        raise ValueError(f"{path.name} has no {TRANSCRIPTS}")

    # Segment field names are mapped by alias rather than assumed, the same way
    # the plain JSON reader works, so a schema change in Meetily does not
    # require a change here.
    parsed = json_ingest.parse(
        transcripts.read_text(encoding="utf-8"), source=f"meetily-recording:{path.name}"
    )

    meta = read_metadata(path)
    parsed.title = meta.get("meeting_name") or path.name
    created = meta.get("created_at")
    if created:
        parsed.meeting_date = str(created)[:10]

    if not parsed:
        status = meta.get("status", "unknown")
        raise ValueError(
            f"'{parsed.title}' contains no transcript segments (status: {status}).\n"
            "  Meetily records audio and transcribes separately, so this happens when\n"
            "  the recording was silent, very short, or the transcription model was\n"
            "  still downloading. The audio is still in the folder."
        )

    return parsed
