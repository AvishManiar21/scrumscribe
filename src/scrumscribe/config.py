"""Paths and defaults.

Memory lives outside the repo by default. Meeting notes are personal data and
the single worst failure mode for this project would be helpfully committing a
transcript of your supervisor to a public GitHub repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from .model.ollama import DEFAULT_HOST, DEFAULT_MODEL

ENV_HOME = "SCRUMSCRIBE_HOME"
ENV_MODEL = "SCRUMSCRIBE_MODEL"
ENV_HOST = "SCRUMSCRIBE_HOST"
ENV_RECORDINGS = "SCRUMSCRIBE_RECORDINGS"


def home() -> Path:
    """Where memory and generated notes live."""
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".scrumscribe"


def memory_path() -> Path:
    return home() / "memory.json"


def notes_dir() -> Path:
    return home() / "notes"


def default_model() -> str:
    return os.environ.get(ENV_MODEL, DEFAULT_MODEL)


def default_host() -> str:
    return os.environ.get(ENV_HOST, DEFAULT_HOST)


def recordings_dir() -> Path | None:
    """Where Meetily writes its per-meeting recording folders.

    The location is configurable inside Meetily, so this is a best guess over
    the usual places. Override with SCRUMSCRIBE_RECORDINGS, or pass a path
    explicitly on the command line.
    """
    override = os.environ.get(ENV_RECORDINGS)
    if override:
        path = Path(override).expanduser()
        return path if path.is_dir() else None

    candidates = [
        Path.home() / "Music" / "meetily-recordings",
        Path.home() / "Documents" / "meetily-recordings",
        Path.home() / "meetily-recordings",
        Path.home() / "Videos" / "meetily-recordings",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return None
