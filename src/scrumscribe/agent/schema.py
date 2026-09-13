"""Structured output schema for scrum notes.

Two representations live here and they serve different masters:

* dataclasses -- what the rest of the program passes around and renders.
* JSON Schema dicts -- handed to Ollama's `format` parameter so decoding is
  constrained at the sampler. On a 4B model this is the single highest-leverage
  reliability decision in the project: asking politely for JSON in the prompt
  produces parse failures on a meaningful fraction of chunks, whereas a
  constrained grammar produces valid JSON every time.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

UNASSIGNED = "unassigned"


@dataclass
class Item:
    """A single attributed statement: who said/owns it, and what it was."""

    owner: str = UNASSIGNED
    detail: str = ""

    def key(self) -> str:
        # Keyed on content alone. The same point is often attributed in one
        # chunk and left unattributed in another; including the owner in the
        # identity would make those two look like different points.
        return _fingerprint(self.detail)


@dataclass
class ActionItem:
    """A commitment made in the meeting, tracked across meetings."""

    owner: str = UNASSIGNED
    task: str = ""
    due: str = ""
    status: str = "open"  # open | done | dropped
    first_seen: str = ""  # ISO date of the meeting it was first raised in
    meetings: int = 1  # how many meetings it has survived

    def key(self) -> str:
        # Keyed on the task alone, which also makes cross-meeting identity
        # survive a week where the model attributed the task differently.
        return _fingerprint(self.task)

    def label(self) -> str:
        bits = [self.task.strip()]
        if self.due:
            bits.append(f"due {self.due}")
        return " — ".join(bits)


@dataclass
class ScrumNotes:
    """The finished product for one meeting."""

    title: str = "Scrum meeting"
    date: str = ""
    participants: list[str] = field(default_factory=list)
    summary: str = ""
    progress: list[Item] = field(default_factory=list)
    blockers: list[Item] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    feedback: list[Item] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)

    # Populated from cross-meeting memory, not from this transcript.
    carried_over: list[ActionItem] = field(default_factory=list)
    resolved: list[ActionItem] = field(default_factory=list)

    # Provenance, so a reader can judge how much to trust the notes.
    source: str = ""
    model: str = ""
    chunks: int = 0
    failed_chunks: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _fingerprint(text: str) -> str:
    """Normalised hash used to detect the same item restated differently.

    Lowercase, strip punctuation and stopwords, keep the content words sorted.
    Cheap, deterministic, and good enough to stop the same action item
    appearing three times because it was mentioned in three chunks.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    meaningful = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    signature = " ".join(sorted(set(meaningful)))
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "will", "was", "are", "has",
    "have", "had", "his", "her", "its", "our", "their", "can", "should",
    "would", "could", "into", "from", "about", "than", "then", "there",
    "been", "but", "not", "you", "your", "she", "him", "they", "them",
    "some", "what", "when", "which", "who", "how", "need", "needs",
    "going", "get", "got", "make", "made", "also", "just", "still", "very",
}


# --- JSON Schemas handed to Ollama -------------------------------------------

_ITEM = {
    "type": "object",
    "properties": {
        "owner": {"type": "string"},
        "detail": {"type": "string"},
    },
    "required": ["owner", "detail"],
}

_ACTION = {
    "type": "object",
    "properties": {
        "owner": {"type": "string"},
        "task": {"type": "string"},
        "due": {"type": "string"},
    },
    "required": ["owner", "task"],
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "progress": {"type": "array", "items": _ITEM},
        "blockers": {"type": "array", "items": _ITEM},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "feedback": {"type": "array", "items": _ITEM},
        "action_items": {"type": "array", "items": _ACTION},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["progress", "blockers", "decisions", "feedback", "action_items", "questions"],
}

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {"action_items": {"type": "array", "items": _ACTION}},
    "required": ["action_items"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

RESOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "resolved": {"type": "boolean"},
        "evidence": {"type": "string"},
    },
    "required": ["resolved", "evidence"],
}

EMPTY_EXTRACT = {
    "progress": [],
    "blockers": [],
    "decisions": [],
    "feedback": [],
    "action_items": [],
    "questions": [],
}
