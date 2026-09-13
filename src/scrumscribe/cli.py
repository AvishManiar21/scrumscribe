"""Command line interface.

Each stage is a separate command on purpose. Note generation is the slow,
failure-prone part, and it must be re-runnable against a transcript you already
have without redoing anything upstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date as _date
from pathlib import Path

from . import config
from .agent import ScrumAgent
from .agent.prompts import STANDUP_SYSTEM, standup_prompt
from .context import git_context
from .ingest import detect, load
from .ingest import meetily_folder
from .ingest.meetily import MeetilyStore, find_db
from .memory import Memory
from .model import Ollama, OllamaError
from .render import to_markdown, to_terminal


def _use_utf8() -> None:
    """Force UTF-8 on stdout/stderr.

    The Windows console defaults to cp1252, which turns every em-dash and
    check mark in the notes into a replacement character. The notes file is
    always written as UTF-8; this makes the terminal agree with it.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _echo(message: str = "") -> None:
    print(message, file=sys.stderr)


def _client(args) -> Ollama:
    return Ollama(model=args.model, host=args.host)


def _slug(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_ " else "" for ch in text)
    return "-".join(safe.split()).lower()[:60] or "meeting"


# -- commands -----------------------------------------------------------------


def cmd_notes(args) -> int:
    source = Path(args.transcript)
    if not source.exists():
        _echo(f"error: {source} does not exist")
        return 2

    client = _client(args)
    try:
        client.preflight()
    except OllamaError as exc:
        _echo(f"error: {exc}")
        return 3

    _echo(f"reading {source.name} (detected: {detect(source)})")
    try:
        transcript = load(source, meeting_id=args.meeting)
    except (ValueError, OSError) as exc:
        _echo(f"error: {exc}")
        return 2

    memory = Memory(config.memory_path())
    open_items = [] if args.no_memory else memory.open_items()

    agent = ScrumAgent(client, max_tokens=args.chunk_tokens, on_progress=_echo)
    notes = agent.run(
        transcript,
        title=args.title,
        meeting_date=args.date,
        open_items=open_items,
    )

    markdown = to_markdown(notes)

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = config.notes_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{notes.date}-{_slug(notes.title)}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    if not args.no_memory:
        change = memory.record(notes, out_path)
        _echo(
            f"memory: +{change['added']} new, {change['closed']} closed, "
            f"{change['carried']} carried over"
        )

    _echo("")
    print(to_terminal(notes) if not args.markdown else markdown)
    _echo("")
    _echo(f"notes written to {out_path}")
    return 0


def _from_recordings(args, root: Path, recordings: list[Path]) -> int:
    _echo(f"recordings: {root}")

    if args.list:
        for i, rec in enumerate(recordings):
            info = meetily_folder.describe(rec)
            marker = "*" if i == 0 else " "
            saved = "saved" if info["saved_to_db"] else "unsaved"
            print(
                f"{marker} {info['created']}  {info['segments']:>5} segments  "
                f"{info['audio_bytes'] // 1024:>6} KB  {saved:>7}  {info['name']}"
            )
        _echo("\n(* = most recent)")
        return 0

    target = recordings[0]
    if args.meeting:
        matches = [r for r in recordings if args.meeting.lower() in r.name.lower()]
        if not matches:
            _echo(f"error: no recording matching '{args.meeting}'")
            return 2
        target = matches[0]

    try:
        transcript = load(target)
    except ValueError as exc:
        _echo(f"error: {exc}")
        return 1

    out = Path(args.output) if args.output else config.home() / "transcripts" / f"{target.name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(transcript.render(), encoding="utf-8")
    _echo(
        f"exported {len(transcript)} speaker turns "
        f"({transcript.word_count()} words) to {out}"
    )
    _echo(f'\nnext:  scrumscribe notes "{out}"')
    return 0


def cmd_meetily(args) -> int:
    """Read from Meetily.

    Recording folders are preferred over the database. Meetily only writes rows
    to SQLite once a meeting is saved in the app, so a recording that was made
    but never saved is on disk and absent from the database entirely.
    """
    root = Path(args.recordings) if args.recordings else config.recordings_dir()
    if root and not args.db:
        recordings = meetily_folder.list_recordings(root)
        if recordings:
            return _from_recordings(args, root, recordings)
        _echo(f"note: {root} exists but holds no recordings yet")

    db = find_db(Path(args.db) if args.db else None)
    if not db:
        _echo(
            "error: no Meetily database found.\n"
            "  Pass one explicitly:  scrumscribe meetily --db <path-to.db>\n"
            "  Meetily stores it under %APPDATA% or %LOCALAPPDATA%."
        )
        return 2

    _echo(f"database: {db}")
    with MeetilyStore(db) as store:
        meetings = store.list_meetings()
        if not meetings:
            _echo("no meetings found in this database")
            return 1

        if args.list:
            for i, meeting in enumerate(meetings):
                marker = "*" if i == 0 else " "
                date = (meeting["date"] or "")[:10] or "unknown date"
                print(f"{marker} {date}  {meeting['segments']:>5} segments  {meeting['title']}")
            _echo("\n(* = most recent)")
            return 0

        target = args.meeting or meetings[0]["id"]
        transcript = store.load(target)

    if not transcript:
        _echo("error: that meeting has no transcript segments")
        return 1

    transcript = transcript.merge_consecutive()
    out = Path(args.output) if args.output else config.home() / "transcripts" / f"{target or 'latest'}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(transcript.render(), encoding="utf-8")
    _echo(
        f"exported {len(transcript)} speaker turns "
        f"({transcript.word_count()} words) to {out}"
    )
    _echo(f"\nnext:  scrumscribe notes {out}")
    return 0


def cmd_open(args) -> int:
    memory = Memory(config.memory_path())
    items = memory.open_items()
    if not items:
        print("No open action items.")
        return 0

    for item in sorted(items, key=lambda i: -i.meetings):
        age = f"  (open {item.meetings} meetings)" if item.meetings > 1 else ""
        due = f"  due {item.due}" if item.due else ""
        print(f"[{item.key()[:8]}] {item.owner}: {item.task}{due}{age}")
    return 0


def cmd_close(args) -> int:
    memory = Memory(config.memory_path())
    item = memory.close_item(args.item)
    if not item:
        _echo(f"no open action item matching '{args.item}'")
        return 1
    print(f"closed: {item.owner}: {item.task}")
    return 0


def cmd_history(args) -> int:
    memory = Memory(config.memory_path())
    meetings = memory.meetings()
    if not meetings:
        print("No meetings recorded yet.")
        return 0
    for meeting in meetings[: args.limit]:
        counts = meeting.get("counts", {})
        print(f"{meeting.get('date', '?')}  {meeting.get('title', 'untitled')}")
        print(
            f"    {counts.get('progress', 0)} progress, "
            f"{counts.get('blockers', 0)} blockers, {counts.get('actions', 0)} actions"
        )
        if meeting.get("summary"):
            print(f"    {meeting['summary'][:160]}")
    return 0


def cmd_standup(args) -> int:
    client = _client(args)
    try:
        client.preflight()
    except OllamaError as exc:
        _echo(f"error: {exc}")
        return 3

    memory = Memory(config.memory_path())
    last = memory.last_meeting()
    open_items = memory.open_items()

    since = args.since or (last or {}).get("date")
    commits: list[str] = []
    for repo in args.repo or []:
        found = git_context.commits_since(Path(repo), since=since, author=args.author)
        name = git_context.repo_name(Path(repo))
        commits.extend(f"[{name}] {line}" for line in found)

    if args.repo and not commits:
        _echo(f"note: no commits found since {since or 'the beginning'}")

    draft = client.chat(
        standup_prompt(
            [f"{i.owner}: {i.label()}" for i in open_items],
            commits,
            (last or {}).get("summary", ""),
        ),
        system=STANDUP_SYSTEM,
    )
    print(draft.strip())
    return 0


def cmd_doctor(args) -> int:
    ok = True
    print("ScrumScribe diagnostics")
    print("=" * 60)

    client = _client(args)
    if client.available():
        print(f"[ok]   Ollama reachable at {client.host}")
        installed = client.models()
        if any(m == client.model or m.startswith(f"{client.model}:") for m in installed):
            print(f"[ok]   Model '{client.model}' installed")
        else:
            ok = False
            print(f"[FAIL] Model '{client.model}' missing — run: ollama pull {client.model}")
        print(f"       installed: {', '.join(installed) or '(none)'}")
    else:
        ok = False
        print(f"[FAIL] Ollama unreachable at {client.host} — run: ollama serve")

    home = config.home()
    print(f"[ok]   Memory home: {home}")
    memory_file = config.memory_path()
    if memory_file.is_file():
        memory = Memory(memory_file)
        print(
            f"[ok]   Memory: {len(memory.meetings())} meetings, "
            f"{len(memory.open_items())} open action items"
        )
    else:
        print("[--]   Memory: not created yet (first run will create it)")

    root = Path(args.recordings) if args.recordings else config.recordings_dir()
    if root:
        recordings = meetily_folder.list_recordings(root)
        print(f"[ok]   Meetily recordings: {root}")
        if recordings:
            usable = 0
            for rec in recordings[:5]:
                info = meetily_folder.describe(rec)
                flag = "ok" if info["segments"] else "--"
                usable += bool(info["segments"])
                print(
                    f"[{flag}]     {info['name']}  {info['segments']} segments, "
                    f"{info['audio_bytes'] // 1024} KB audio, {info['status']}"
                )
            if not usable:
                print(
                    "       none of these have a transcript yet. Meetily records audio\n"
                    "       and transcribes separately -- a silent or very short recording,\n"
                    "       or one made while the model was still downloading, produces none."
                )
        else:
            print("[--]     no recordings yet")
    else:
        print("[--]   Meetily recordings folder not found")

    db = find_db(Path(args.db) if args.db else None)
    if db:
        print(f"[ok]   Meetily database: {db}")
        try:
            with MeetilyStore(db) as store:
                if store.best:
                    tm = store.best
                    print(
                        f"[ok]   Discovered transcript table '{tm.table}' "
                        f"(text={tm.text}, speaker={tm.speaker}, "
                        f"start={tm.start}, meeting={tm.meeting}, score={tm.score})"
                    )
                    print(f"[ok]   Meetings in store: {len(store.list_meetings())}")
                    if args.verbose:
                        for candidate in store.tables:
                            print(f"         candidate: {candidate}")
                else:
                    print(
                        "[--]   No transcript table found in that database "
                        "(normal before the first meeting is saved)"
                    )
        except Exception as exc:  # noqa: BLE001 - diagnostics must never crash
            ok = False
            print(f"[FAIL] Could not read database: {exc}")
    else:
        print("[--]   Meetily database not found (fine if you use .vtt or .txt inputs)")

    print("=" * 60)
    print("all good" if ok else "problems found — see [FAIL] lines above")
    return 0 if ok else 1


def cmd_export(args) -> int:
    """Dump memory as JSON for use elsewhere."""
    memory = Memory(config.memory_path())
    print(json.dumps(memory.data, indent=2, ensure_ascii=False))
    return 0


# -- argument parsing ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scrumscribe",
        description="Local-first scrum meeting agent. No API keys, no cloud.",
    )
    parser.add_argument("--model", default=config.default_model(), help="Ollama model")
    parser.add_argument("--host", default=config.default_host(), help="Ollama host URL")

    sub = parser.add_subparsers(dest="command", required=True)

    p_notes = sub.add_parser("notes", help="generate scrum notes from a transcript")
    p_notes.add_argument("transcript", help="path to a transcript (.vtt/.txt/.json/.db)")
    p_notes.add_argument("-o", "--output", help="write notes here (default: ~/.scrumscribe/notes)")
    p_notes.add_argument("--title", help="meeting title")
    p_notes.add_argument("--date", help="meeting date (ISO), default today")
    p_notes.add_argument("--meeting", help="meeting id, when reading a Meetily database")
    p_notes.add_argument("--chunk-tokens", type=int, default=1800, help="section size")
    p_notes.add_argument("--no-memory", action="store_true", help="do not read or update memory")
    p_notes.add_argument("--markdown", action="store_true", help="print markdown to stdout")
    p_notes.set_defaults(func=cmd_notes)

    p_meetily = sub.add_parser("meetily", help="read transcripts from Meetily's database")
    p_meetily.add_argument("--db", help="path to the Meetily database")
    p_meetily.add_argument("--recordings", help="path to Meetily's recordings folder")
    p_meetily.add_argument("--list", action="store_true", help="list meetings and exit")
    p_meetily.add_argument("--meeting", help="meeting id to export (default: most recent)")
    p_meetily.add_argument("-o", "--output", help="write the transcript here")
    p_meetily.set_defaults(func=cmd_meetily)

    p_open = sub.add_parser("open", help="list open action items")
    p_open.set_defaults(func=cmd_open)

    p_close = sub.add_parser("close", help="close an action item by id or text")
    p_close.add_argument("item")
    p_close.set_defaults(func=cmd_close)

    p_history = sub.add_parser("history", help="list past meetings")
    p_history.add_argument("-n", "--limit", type=int, default=10)
    p_history.set_defaults(func=cmd_history)

    p_standup = sub.add_parser("standup", help="draft your next status update")
    p_standup.add_argument("--repo", action="append", help="repo to read commits from (repeatable)")
    p_standup.add_argument("--since", help="ISO date (default: last meeting)")
    p_standup.add_argument("--author", help="filter commits by author")
    p_standup.set_defaults(func=cmd_standup)

    p_doctor = sub.add_parser("doctor", help="check the setup end to end")
    p_doctor.add_argument("--db", help="path to a Meetily database to inspect")
    p_doctor.add_argument("--recordings", help="path to Meetily's recordings folder")
    p_doctor.add_argument("-v", "--verbose", action="store_true")
    p_doctor.set_defaults(func=cmd_doctor)

    p_export = sub.add_parser("export", help="dump memory as JSON")
    p_export.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    _use_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _echo("\ninterrupted")
        return 130
    except OllamaError as exc:
        _echo(f"error: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
