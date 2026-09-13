"""Meetily SQLite reader with runtime schema discovery.

Meetily stores meetings in a local SQLite database, but its schema is not a
documented public interface and has changed across releases. Hardcoding table
and column names would make this integration break on their next update, so
instead we inspect the database at runtime, score every table on how much it
looks like a transcript store, and map columns by alias.

The database is always opened read-only. We never write to Meetily's store.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..transcript import Transcript, Utterance

TEXT_COLS = ("text", "transcript", "content", "sentence", "body", "message", "chunk")
START_COLS = ("start", "start_time", "starttime", "begin", "timestamp", "offset", "ts", "time")
END_COLS = ("end", "end_time", "endtime", "stop")
SPEAKER_COLS = ("speaker", "speaker_label", "speaker_name", "who", "participant", "source")
MEETING_COLS = ("meeting_id", "meetingid", "session_id", "sessionid", "recording_id", "call_id")
TITLE_COLS = ("title", "name", "subject", "meeting_name")
DATE_COLS = ("created_at", "createdat", "date", "started_at", "timestamp", "updated_at")


def default_db_paths() -> list[Path]:
    """Probable Meetily database locations on this machine.

    Tauri apps store data under an identifier-named folder in APPDATA, so we
    glob rather than guess the exact bundle id.
    """
    roots: list[Path] = []
    for env in ("APPDATA", "LOCALAPPDATA"):
        value = os.environ.get(env)
        if value:
            roots.append(Path(value))
    roots += [Path.home(), Path.home() / ".meetily"]

    found: list[Path] = []
    patterns = ("*meetily*/**/*.db", "*meetily*/**/*.sqlite*", "*meeting*/**/*.db")
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            try:
                found.extend(p for p in root.glob(pattern) if p.is_file())
            except OSError:
                continue

    # De-duplicate, newest first -- the active database is the one just written.
    unique = {p.resolve(): p for p in found}
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def find_db(explicit: Path | None = None) -> Path | None:
    if explicit:
        return explicit if explicit.is_file() else None
    candidates = default_db_paths()
    return candidates[0] if candidates else None


@dataclass
class TableMap:
    """A discovered transcript table and the column roles we matched in it."""

    table: str
    text: str
    start: str | None = None
    end: str | None = None
    speaker: str | None = None
    meeting: str | None = None
    order_by: str | None = None
    score: int = 0


def _match(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    # Fall back to substring containment (e.g. "transcript_text").
    for alias in aliases:
        for low, original in lowered.items():
            if alias in low:
                return original
    return None


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # Heuristic: values this large are milliseconds, not seconds.
    return number / 1000 if number > 100_000 else number


class MeetilyStore:
    def __init__(self, path: Path):
        self.path = path
        self.conn = _connect(path)
        self.tables = self._discover()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> MeetilyStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _table_names(self) -> list[str]:
        sql = (
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%'"
        )
        return [r["name"] for r in self.conn.execute(sql).fetchall()]

    def _columns(self, table: str) -> list[str]:
        return [r["name"] for r in self.conn.execute(f'PRAGMA table_info("{table}")')]

    def _discover(self) -> list[TableMap]:
        """Score every table on how transcript-shaped it is."""
        maps: list[TableMap] = []

        for table in self._table_names():
            columns = self._columns(table)
            if not columns:
                continue
            text_col = _match(columns, TEXT_COLS)
            if not text_col:
                continue

            try:
                rows = self.conn.execute(
                    f'SELECT "{text_col}" AS t FROM "{table}" '
                    f'WHERE "{text_col}" IS NOT NULL LIMIT 50'
                ).fetchall()
            except sqlite3.Error:
                continue

            samples = [r["t"] for r in rows if isinstance(r["t"], str) and r["t"].strip()]
            if not samples:
                continue

            tm = TableMap(
                table=table,
                text=text_col,
                start=_match(columns, START_COLS),
                end=_match(columns, END_COLS),
                speaker=_match(columns, SPEAKER_COLS),
                meeting=_match(columns, MEETING_COLS),
            )
            lower_cols = [c.lower() for c in columns]
            tm.order_by = tm.start or ("id" if "id" in lower_cols else None)

            # A transcript table has many rows of short-to-medium utterances.
            avg_len = sum(len(s) for s in samples) / len(samples)
            score = len(samples)
            if "transcript" in table.lower():
                score += 100
            if tm.speaker:
                score += 30
            if tm.start:
                score += 20
            if tm.meeting:
                score += 10
            if 10 <= avg_len <= 600:
                score += 25
            elif avg_len > 5000:
                score -= 50  # this is a summary/notes blob, not utterances
            tm.score = score
            maps.append(tm)

        return sorted(maps, key=lambda m: m.score, reverse=True)

    @property
    def best(self) -> TableMap | None:
        return self.tables[0] if self.tables else None

    def _meeting_metadata(self) -> dict[str, dict]:
        """Best-effort title/date per meeting id from any meetings-like table."""
        meta: dict[str, dict] = {}
        for table in self._table_names():
            lowered = table.lower()
            if "meeting" not in lowered and "session" not in lowered:
                continue
            columns = self._columns(table)
            id_col = _match(columns, ("id",) + MEETING_COLS)
            if not id_col:
                continue

            select = [f'"{id_col}" AS mid']
            title_col = _match(columns, TITLE_COLS)
            date_col = _match(columns, DATE_COLS)
            if title_col:
                select.append(f'"{title_col}" AS title')
            if date_col:
                select.append(f'"{date_col}" AS date')

            try:
                rows = self.conn.execute(f'SELECT {", ".join(select)} FROM "{table}"')
                for row in rows:
                    keys = row.keys()
                    meta[str(row["mid"])] = {
                        "title": row["title"] if "title" in keys else None,
                        "date": str(row["date"]) if "date" in keys and row["date"] else None,
                    }
            except sqlite3.Error:
                continue
        return meta

    def list_meetings(self) -> list[dict]:
        """Every meeting in the store, newest first."""
        tm = self.best
        if not tm:
            return []
        meta = self._meeting_metadata()

        if not tm.meeting:
            row = self.conn.execute(f'SELECT COUNT(*) AS n FROM "{tm.table}"').fetchone()
            return [{"id": None, "title": self.path.stem, "date": None, "segments": row["n"]}]

        rows = self.conn.execute(
            f'SELECT "{tm.meeting}" AS mid, COUNT(*) AS n FROM "{tm.table}" '
            f'GROUP BY "{tm.meeting}"'
        ).fetchall()

        meetings = []
        for row in rows:
            mid = str(row["mid"])
            info = meta.get(mid, {})
            meetings.append(
                {
                    "id": mid,
                    "title": info.get("title") or f"meeting {mid[:8]}",
                    "date": info.get("date"),
                    "segments": row["n"],
                }
            )
        meetings.sort(key=lambda m: (m["date"] or "", m["id"] or ""), reverse=True)
        return meetings

    def load(self, meeting_id: str | None = None) -> Transcript:
        tm = self.best
        if not tm:
            raise ValueError(
                f"No transcript-shaped table found in {self.path}. "
                f"Run `scrumscribe doctor --db {self.path}` to inspect the schema."
            )

        select = [f'"{tm.text}" AS text']
        for role, col in (("start", tm.start), ("end", tm.end), ("speaker", tm.speaker)):
            if col:
                select.append(f'"{col}" AS {role}')

        sql = f'SELECT {", ".join(select)} FROM "{tm.table}"'
        params: list = []
        if meeting_id and tm.meeting:
            sql += f' WHERE "{tm.meeting}" = ?'
            params.append(meeting_id)
        if tm.order_by:
            sql += f' ORDER BY "{tm.order_by}"'

        utterances = []
        for row in self.conn.execute(sql, params):
            keys = row.keys()
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                continue
            utterances.append(
                Utterance(
                    text.strip(),
                    _num(row["start"]) if "start" in keys else None,
                    _num(row["end"]) if "end" in keys else None,
                    str(row["speaker"]) if "speaker" in keys and row["speaker"] else None,
                )
            )

        meta = self._meeting_metadata().get(str(meeting_id), {}) if meeting_id else {}
        return Transcript(
            utterances,
            source=f"meetily:{self.path.name}",
            title=meta.get("title") or self.path.stem,
            meeting_date=meta.get("date"),
        )


def load(path: Path, meeting_id: str | None = None) -> Transcript:
    with MeetilyStore(path) as store:
        return store.load(meeting_id)
