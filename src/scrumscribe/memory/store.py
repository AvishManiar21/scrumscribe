"""Cross-meeting memory.

This is the part that makes ScrumScribe an assistant rather than a summariser.
A transcript tool answers "what was said today". Memory answers the question
that actually matters in a weekly scrum: "what did I promise three weeks ago
that I still have not done."

Storage is a single JSON file. It is small, diffable, trivially inspectable,
and survives every version of this program. Writes are atomic: we render to a
temporary file in the same directory and replace, so an interrupted write
cannot leave you with a truncated memory of your own project.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import date as _date
from pathlib import Path

from ..agent.schema import ActionItem, ScrumNotes

SCHEMA_VERSION = 1


class Memory:
    """Durable record of past meetings and the commitments made in them."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = self._load()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> dict:
        if not self.path.is_file():
            return {"version": SCHEMA_VERSION, "meetings": [], "action_items": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            # A corrupt memory file must not block note-taking. Preserve it for
            # inspection and start fresh rather than dying on startup.
            backup = self.path.with_suffix(".corrupt.json")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            return {"version": SCHEMA_VERSION, "meetings": [], "action_items": []}

        data.setdefault("version", SCHEMA_VERSION)
        data.setdefault("meetings", [])
        data.setdefault("action_items", [])
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, indent=2, ensure_ascii=False)

        handle, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".memory-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    # -- queries --------------------------------------------------------------

    def open_items(self) -> list[ActionItem]:
        return [
            _to_item(row)
            for row in self.data["action_items"]
            if row.get("status", "open") == "open"
        ]

    def all_items(self) -> list[ActionItem]:
        return [_to_item(row) for row in self.data["action_items"]]

    def meetings(self) -> list[dict]:
        return sorted(self.data["meetings"], key=lambda m: m.get("date", ""), reverse=True)

    def last_meeting(self) -> dict | None:
        found = self.meetings()
        return found[0] if found else None

    # -- mutation -------------------------------------------------------------

    def record(self, notes: ScrumNotes, notes_path: Path | None = None) -> dict:
        """Fold one meeting's notes into memory and return a change report."""
        today = notes.date or _date.today().isoformat()
        index = {row["key"]: row for row in self.data["action_items"]}

        closed = 0
        for item in notes.resolved:
            row = index.get(item.key())
            if row and row.get("status") == "open":
                row["status"] = "done"
                row["closed_at"] = today
                closed += 1

        added, carried = 0, 0
        for item in notes.action_items:
            key = item.key()
            row = index.get(key)
            if row is None:
                record = asdict(item)
                record["key"] = key
                record["first_seen"] = today
                record["meetings"] = 1
                record["status"] = "open"
                self.data["action_items"].append(record)
                index[key] = record
                added += 1
            elif row.get("status") == "open":
                # Re-stated in a later meeting: it has now survived another week.
                if row.get("last_seen") != today:
                    row["meetings"] = int(row.get("meetings", 1)) + 1
                    carried += 1
                if item.due and not row.get("due"):
                    row["due"] = item.due
            row = index[key]
            row["last_seen"] = today

        # Re-running the same meeting replaces its record rather than duplicating it.
        self.data["meetings"] = [
            m
            for m in self.data["meetings"]
            if not (m.get("date") == today and m.get("title") == notes.title)
        ]
        self.data["meetings"].append(
            {
                "date": today,
                "title": notes.title,
                "summary": notes.summary,
                "participants": notes.participants,
                "source": notes.source,
                "notes_path": str(notes_path) if notes_path else None,
                "counts": {
                    "progress": len(notes.progress),
                    "blockers": len(notes.blockers),
                    "actions": len(notes.action_items),
                },
            }
        )

        self.save()
        return {"added": added, "closed": closed, "carried": carried}

    def close_item(self, needle: str) -> ActionItem | None:
        """Manually close an action item by key prefix or task substring."""
        needle_lower = needle.lower()
        for row in self.data["action_items"]:
            if row.get("status") != "open":
                continue
            if row["key"].startswith(needle) or needle_lower in row.get("task", "").lower():
                row["status"] = "done"
                row["closed_at"] = _date.today().isoformat()
                self.save()
                return _to_item(row)
        return None


def _to_item(row: dict) -> ActionItem:
    """Build an ActionItem from a stored row, ignoring persistence-only fields.

    Stored rows carry bookkeeping the dataclass does not model (key, last_seen,
    closed_at). Filtering by field name keeps old memory files loadable after
    the schema grows.
    """
    fields = set(ActionItem.__dataclass_fields__)
    return ActionItem(**{k: v for k, v in row.items() if k in fields})
