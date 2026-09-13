"""Markdown rendering.

Notes are written to read well in a terminal, in a text editor, and pasted
into Notion or a GitHub issue without touch-up. Sections with no content are
omitted entirely rather than left as empty headings -- an empty "Blockers"
heading reads as "we checked and there were none", which is a claim the model
did not make.
"""

from __future__ import annotations

from .agent.schema import ActionItem, Item, ScrumNotes, UNASSIGNED


def _owned(item: Item) -> str:
    if item.owner and item.owner != UNASSIGNED:
        return f"**{item.owner}** — {item.detail}"
    return item.detail


def _action_line(action: ActionItem, show_age: bool = False) -> str:
    box = "[x]" if action.status == "done" else "[ ]"
    owner = f"**{action.owner}**: " if action.owner != UNASSIGNED else ""
    line = f"- {box} {owner}{action.task}"
    if action.due:
        line += f" _(due {action.due})_"
    if show_age and action.meetings > 1:
        line += f" ⚠️ _open for {action.meetings} meetings_"
    return line


def _prefixed(owner: str, text: str) -> str:
    """Terminal line, without a dangling owner when nobody was attributed."""
    return f"{owner}: {text}" if owner and owner != UNASSIGNED else text


def to_markdown(notes: ScrumNotes) -> str:
    out: list[str] = []
    out.append(f"# {notes.title}")
    out.append("")

    meta = [f"**Date:** {notes.date}"]
    if notes.participants:
        meta.append(f"**Participants:** {', '.join(notes.participants)}")
    else:
        # Be explicit rather than leaving the reader to wonder why nothing is
        # attributed. Meetily's recording export carries no speaker labels.
        meta.append("**Participants:** _not labelled in this transcript_")
    out.append("  \n".join(meta))
    out.append("")

    if notes.summary:
        out.append(notes.summary)
        out.append("")

    if notes.carried_over:
        out.append("## ⚠️ Still open from previous meetings")
        out.append("")
        for action in sorted(notes.carried_over, key=lambda a: -a.meetings):
            out.append(_action_line(action, show_age=True))
        out.append("")

    if notes.resolved:
        out.append("## ✅ Closed this meeting")
        out.append("")
        for action in notes.resolved:
            out.append(f"- [x] {_prefixed(action.owner, action.task)}")
        out.append("")

    if notes.progress:
        out.append("## Progress")
        out.append("")
        out.extend(f"- {_owned(i)}" for i in notes.progress)
        out.append("")

    if notes.blockers:
        out.append("## Blockers")
        out.append("")
        out.extend(f"- {_owned(i)}" for i in notes.blockers)
        out.append("")

    if notes.feedback:
        out.append("## Feedback & guidance")
        out.append("")
        out.extend(f"- {_owned(i)}" for i in notes.feedback)
        out.append("")

    if notes.decisions:
        out.append("## Decisions")
        out.append("")
        out.extend(f"- {d}" for d in notes.decisions)
        out.append("")

    if notes.action_items:
        out.append("## Action items")
        out.append("")
        out.extend(_action_line(a) for a in notes.action_items)
        out.append("")

    if notes.questions:
        out.append("## Open questions")
        out.append("")
        out.extend(f"- {q}" for q in notes.questions)
        out.append("")

    out.append("---")
    provenance = (
        f"_Generated locally by ScrumScribe using `{notes.model}` "
        f"from {notes.chunks} transcript section(s)._"
    )
    if notes.failed_chunks:
        # Never hide partial coverage: a note that silently dropped 3 of 17
        # sections looks identical to a complete one.
        provenance += (
            f"  \n_⚠️ {notes.failed_chunks} of {notes.chunks} section(s) failed to "
            "process — these notes are incomplete. Re-run to retry._"
        )
    out.append(provenance)
    out.append("")

    return "\n".join(out)


def to_terminal(notes: ScrumNotes) -> str:
    """Compact rendering for stdout."""
    lines = [f"{notes.title}  ({notes.date})"]
    if notes.participants:
        lines.append(f"  with: {', '.join(notes.participants)}")
    lines.append("")
    if notes.summary:
        lines.append(notes.summary)
        lines.append("")

    def block(label: str, values: list[str]) -> None:
        if values:
            lines.append(label)
            lines.extend(f"  - {v}" for v in values)
            lines.append("")

    if notes.carried_over:
        block(
            "STILL OPEN FROM BEFORE",
            [
                _prefixed(a.owner, a.task)
                + (f" (open {a.meetings} meetings)" if a.meetings > 1 else "")
                for a in notes.carried_over
            ],
        )
    block("PROGRESS", [_prefixed(i.owner, i.detail) for i in notes.progress])
    block("BLOCKERS", [_prefixed(i.owner, i.detail) for i in notes.blockers])
    block("FEEDBACK", [_prefixed(i.owner, i.detail) for i in notes.feedback])
    block("DECISIONS", notes.decisions)
    block("ACTION ITEMS", [_prefixed(a.owner, a.label()) for a in notes.action_items])
    block("OPEN QUESTIONS", notes.questions)

    if notes.failed_chunks:
        lines.append(
            f"WARNING: {notes.failed_chunks}/{notes.chunks} sections failed — notes incomplete."
        )
    return "\n".join(lines)
