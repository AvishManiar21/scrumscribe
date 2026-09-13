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
