"""Transcript ingest — format detection and dispatch.

ScrumScribe is deliberately agnostic about where a transcript came from.
Meetily handles capture today; a Teams export, a Whisper JSON dump, or a
hand-typed text file work identically. Anything that can be reduced to
speaker turns is a valid input.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..transcript import Transcript, clean as clean_text
from . import json_ingest, meetily, meetily_folder, text, vtt

__all__ = ["load", "detect", "SUPPORTED"]

SUPPORTED = (
    "meetily-recording",
    "meetily-sqlite",
    "vtt",
    "srt",
    "json",
    "text",
)

_SQLITE_MAGIC = b"SQLite format 3\x00"


def detect(path: Path) -> str:
    """Identify a transcript source by content first, extension second.

    Content sniffing matters because Meetily's database has been shipped with
    several extensions across releases (.db, .sqlite, .sqlite3).
    """
    if path.is_dir():
        # Meetily writes one directory per recording. Pointing at the recording,
        # or at the folder holding all of them, are both reasonable things to do.
        if meetily_folder.is_recording_dir(path):
            return "meetily-recording"
        if meetily_folder.is_recordings_root(path):
            return "meetily-recordings-root"
        raise IsADirectoryError(
            f"{path} is a directory, and does not look like a Meetily recording"
        )

    try:
        with path.open("rb") as fh:
            header = fh.read(16)
    except OSError as exc:
        raise FileNotFoundError(f"cannot read {path}: {exc}") from exc

    if header == _SQLITE_MAGIC:
        return "meetily-sqlite"

    suffix = path.suffix.lower()
    if suffix == ".vtt":
        return "vtt"
    if suffix == ".srt":
        return "srt"
    if suffix == ".json":
        return "json"

    # Extension is missing or lying -- look at the actual bytes.
    head = path.read_text(encoding="utf-8", errors="replace")[:2000].lstrip()
    if head.upper().startswith("WEBVTT"):
        return "vtt"
    if head.startswith(("{", "[")):
        try:
            json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return "json"
        except ValueError:
            pass
    if "-->" in head:
        return "srt"
    return "text"


def load(
    path: Path, meeting_id: str | None = None, strip_fillers: bool = True
) -> Transcript:
    """Load any supported transcript into the common Transcript model.

    Two normalisations happen here, both for the benefit of a small model.

    Utterances are merged into speaker turns: raw cue-level segments shred
    sentences across boundaries and measurably degrade summarisation quality.

    Filler words are stripped. Speech-to-text output is full of "um", "uh" and
    stutter repeats, and on a 4B model every wasted token is context that is
    not available for the actual meeting. Pass ``strip_fillers=False`` to keep
    the transcript verbatim.
    """
    path = Path(path)
    kind = detect(path)

    if kind == "meetily-recording":
        transcript = meetily_folder.load(path)
    elif kind == "meetily-recordings-root":
        recordings = meetily_folder.list_recordings(path)
        if not recordings:
            raise ValueError(f"no recordings found under {path}")
        transcript = meetily_folder.load(recordings[0])
    elif kind == "meetily-sqlite":
        transcript = meetily.load(path, meeting_id)
    elif kind in ("vtt", "srt"):
        # The SRT cue grammar is a subset of what the VTT parser accepts.
        transcript = vtt.load(path)
        if not transcript:
            transcript = text.load(path)
    elif kind == "json":
        transcript = json_ingest.load(path)
    else:
        transcript = text.load(path)

    if not transcript:
        raise ValueError(
            f"Parsed {path.name} as '{kind}' but found no utterances. "
            "If this is a Meetily database, run `scrumscribe doctor` to inspect it."
        )

    transcript = transcript.merge_consecutive()

    if strip_fillers:
        for utt in transcript.utterances:
            stripped = clean_text(utt.text)
            # Never let cleaning empty an utterance -- an all-filler turn
            # ("um, yeah") still carries that someone spoke.
            if stripped:
                utt.text = stripped

    return transcript
