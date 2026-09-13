"""The agent loop: map over chunks, reduce to notes, reconcile with memory.

The shape is map-reduce rather than one big prompt, because a 4B model given a
whole meeting produces confident notes about the first and last five minutes
and quietly ignores everything between. Extracting per-section and merging
afterwards costs more calls but is the difference between notes you can trust
and notes you have to re-read the transcript to check.

Every stage degrades rather than fails. A chunk that errors is counted and
skipped; a summary call that fails falls back to a mechanical summary. The
transcript is already on disk, so any single bad run can simply be re-run --
but a run that dies at chunk 14 of 17 and throws away the first thirteen
extractions is pure waste.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date as _date

from ..model import Chunk, Ollama, chunk as split_chunks, estimate_tokens
from ..transcript import Transcript
from . import prompts
from .schema import (
    CONSOLIDATE_SCHEMA,
    EMPTY_EXTRACT,
    EXTRACT_SCHEMA,
    RESOLUTION_SCHEMA,
    SUMMARY_SCHEMA,
    UNASSIGNED,
    ActionItem,
    Item,
    ScrumNotes,
)

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _as_text(value) -> str:
    """Coerce whatever the model produced into a clean string.

    Constrained decoding guarantees the JSON *shape*, not that every string
    field is non-empty or that a list field holds strings rather than objects.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("detail", "text", "task", "description", "value"):
            if isinstance(value.get(key), str):
                return value[key].strip()
    if value is None:
        return ""
    return str(value).strip()


def _parse_items(rows) -> list[Item]:
    items: list[Item] = []
    if not isinstance(rows, list):
        return items
    for row in rows:
        if isinstance(row, dict):
            detail = _as_text(row.get("detail") or row.get("text"))
            owner = _as_text(row.get("owner")) or UNASSIGNED
        else:
            detail, owner = _as_text(row), UNASSIGNED
        if detail:
            items.append(Item(owner=owner, detail=detail))
    return items


_NON_DATES = {
    "", "none", "n/a", "na", "null", "unspecified", "tbd", "unassigned",
    "not specified", "no due date", "unknown", "asap", "nil", "undefined",
}


def _parse_actions(rows) -> list[ActionItem]:
    actions: list[ActionItem] = []
    if not isinstance(rows, list):
        return actions
    for row in rows:
        if isinstance(row, dict):
            task = _as_text(row.get("task") or row.get("detail"))
            owner = _as_text(row.get("owner")) or UNASSIGNED
            due = _as_text(row.get("due"))
        else:
            task, owner, due = _as_text(row), UNASSIGNED, ""
        if task:
            # Models echo placeholder words rather than omitting an absent
            # date -- including "unassigned", which they copy from the owner
            # field of the examples we showed them.
            if due.lower().strip(" .-") in _NON_DATES:
                due = ""
            actions.append(ActionItem(owner=owner, task=task, due=due))
    return actions


def _parse_strings(rows) -> list[str]:
    if not isinstance(rows, list):
        return []
    return [text for text in (_as_text(r) for r in rows) if text]


def _dedupe_items(items: list[Item]) -> list[Item]:
    """Collapse restatements of the same point across chunks.

    Prefers the longer phrasing, and prefers a named owner over "unassigned"
    -- the same point often gets attributed in one chunk and not another.
    """
    best: dict[str, Item] = {}
    for item in items:
        key = item.key()
        current = best.get(key)
        if current is None:
            best[key] = item
            continue
        if current.owner == UNASSIGNED and item.owner != UNASSIGNED:
            current.owner = item.owner
        if len(item.detail) > len(current.detail):
            current.detail = item.detail
    return list(best.values())


def _dedupe_actions(actions: list[ActionItem]) -> list[ActionItem]:
    """Collapse restatements of the same commitment.

    Identity is the task, not the owner: the same commitment routinely appears
    attributed to the person doing it and to the person who asked for it. When
    owners conflict, the one who owns more of the meeting's action items wins
    -- in a supervisor/student scrum that is reliably the person doing the work.
    """
    tally: dict[str, int] = {}
    for action in actions:
        if action.owner != UNASSIGNED:
            tally[action.owner] = tally.get(action.owner, 0) + 1

    def rank(owner: str) -> int:
        return -1 if owner == UNASSIGNED else tally.get(owner, 0)

    best: dict[str, ActionItem] = {}
    for action in actions:
        key = action.key()
        current = best.get(key)
        if current is None:
            best[key] = action
            continue
        if rank(action.owner) > rank(current.owner):
            current.owner = action.owner
        if not current.due and action.due:
            current.due = action.due
        if len(action.task) > len(current.task):
            current.task = action.task
    return list(best.values())


def _dedupe_strings(values: list[str]) -> list[str]:
    from .schema import _fingerprint

    best: dict[str, str] = {}
    for value in values:
        key = _fingerprint(value)
        if key not in best or len(value) > len(best[key]):
            best[key] = value
    return list(best.values())


class ScrumAgent:
    """Turns a transcript into structured notes using a local model."""

    def __init__(
        self,
        client: Ollama,
        max_tokens: int = 1800,
        on_progress: ProgressFn | None = None,
    ):
        self.client = client
        self.max_tokens = max_tokens
        self.progress = on_progress or _noop

    # -- stage 1: map ---------------------------------------------------------

    def extract(self, chunks: list[Chunk]) -> tuple[dict, int]:
        """Run extraction over every chunk, tolerating individual failures."""
        collected = {key: [] for key in EMPTY_EXTRACT}
        failures = 0

        for c in chunks:
            self.progress(f"  reading section {c.index + 1}/{c.total} ({c.time_range()})")
            try:
                result = self.client.chat_json(
                    prompts.extract_prompt(
                        c.render_with_context(), f"{c.index + 1}", c.total
                    ),
                    schema=EXTRACT_SCHEMA,
                    system=prompts.EXTRACT_SYSTEM,
                    default=EMPTY_EXTRACT,
                )
            except Exception as exc:  # noqa: BLE001 - a bad chunk must not kill the run
                self.progress(f"    ! section {c.index + 1} failed: {exc}")
                failures += 1
                continue

            for key in collected:
                collected[key].extend(result.get(key) or [])

        return collected, failures

    # -- stage 2: reduce ------------------------------------------------------

    def reduce(self, collected: dict) -> ScrumNotes:
        notes = ScrumNotes()
        notes.progress = _dedupe_items(_parse_items(collected["progress"]))
        notes.blockers = _dedupe_items(_parse_items(collected["blockers"]))
        notes.feedback = _dedupe_items(_parse_items(collected["feedback"]))
        notes.action_items = _dedupe_actions(_parse_actions(collected["action_items"]))
        notes.decisions = _dedupe_strings(_parse_strings(collected["decisions"]))
        notes.questions = _dedupe_strings(_parse_strings(collected["questions"]))
        return notes

    def consolidate(self, actions: list[ActionItem], participants: list[str]) -> list[ActionItem]:
        """Semantic merge of near-duplicate action items.

        Lexical fingerprinting catches verbatim restatements but not the common
        case: a commitment made mid-meeting ("send me the schema document
        before Friday") and the same commitment in the closing recap ("schema
        doc"). Those share almost no tokens, so only the model can tell they are
        one item.

        The same pass fixes ownership. Extraction attributes an action to
        whoever was speaking, which makes the supervisor the owner of every task
        they assign.
        """
        if len(actions) < 2:
            return actions

        labels = [f"{a.owner}: {a.label()}" for a in actions]
        try:
            result = self.client.chat_json(
                prompts.consolidate_prompt(labels, participants),
                schema=CONSOLIDATE_SCHEMA,
                system=prompts.CONSOLIDATE_SYSTEM,
                default={},
            )
        except Exception as exc:  # noqa: BLE001
            self.progress(f"  ! consolidation failed, keeping raw list: {exc}")
            return actions

        merged = _dedupe_actions(_parse_actions(result.get("action_items")))

        # Guard against a bad merge. Consolidation should remove duplicates,
        # not content: if the model returns nothing, or collapses the list to a
        # fraction of its input, we distrust it and keep what we extracted.
        # Losing a real commitment is far worse than showing a duplicate.
        if not merged or len(merged) < max(1, len(actions) // 3):
            self.progress(
                f"  ! consolidation returned {len(merged)} of {len(actions)} items "
                "— looks lossy, keeping the raw list"
            )
            return actions

        # Preserve due dates the merge may have dropped.
        for item in merged:
            if item.due:
                continue
            for original in actions:
                if original.due and _overlaps(item.task, original.task):
                    item.due = original.due
                    break

        if len(merged) < len(actions):
            self.progress(f"  merged {len(actions)} action items into {len(merged)}")
        return merged

    # -- stage 3: synthesise --------------------------------------------------

    def summarise(self, notes: ScrumNotes) -> str:
        facts = facts_digest(notes)
        if not facts.strip():
            return "No substantive discussion was captured in this transcript."
        try:
            result = self.client.chat_json(
                prompts.summary_prompt(facts, notes.participants),
                schema=SUMMARY_SCHEMA,
                system=prompts.SUMMARY_SYSTEM,
                default={},
            )
            summary = _as_text(result.get("summary"))
            if summary:
                return summary
        except Exception as exc:  # noqa: BLE001
            self.progress(f"  ! summary generation failed: {exc}")

        # Mechanical fallback -- factual, if inelegant.
        return (
            f"{len(notes.progress)} progress update(s), {len(notes.blockers)} blocker(s) "
            f"and {len(notes.action_items)} action item(s) were recorded."
        )

    # -- stage 4: reconcile with memory --------------------------------------

    def detect_resolved(
        self, open_items: list[ActionItem], chunks: list[Chunk]
    ) -> list[ActionItem]:
        """Find which carried-over commitments this meeting closed.

        Two design decisions carry this feature.

        First, it reads the raw transcript rather than the extracted facts.
        Evidence that a commitment is done is conversational -- "yes, I sent it
        Thursday night" -- and extraction classifies that as neither progress
        nor decision, so it never reaches the facts digest.

        Second, it asks one yes/no question per item instead of asking which of
        N numbered items were resolved. A 4B model given the list-of-indices
        form returns an empty list almost every time, regardless of how the
        prompt is worded: the indirection through numbers plus the scan over a
        whole meeting is simply past what it can do in one step. Decomposed
        into a single focused question it answers correctly. This costs one
        call per open item and is the difference between the feature working
        and silently never firing.

        Every claimed resolution must come with a quote, and the quote is
        checked against the transcript before it is believed.
        """
        if not open_items or not chunks:
            return []

        transcript_text = "\n".join(c.render() for c in chunks)
        resolved: list[ActionItem] = []

        for item in open_items:
            verdict = self._check_one(item, chunks, transcript_text)
            if verdict:
                resolved.append(item)

        return resolved

    def _check_one(
        self, item: ActionItem, chunks: list[Chunk], transcript_text: str
    ) -> bool:
        """Ask whether one commitment was completed, and verify the answer."""
        label = f"{item.owner}: {item.label()}"

        # Short meetings go in whole; long ones are scanned section by section
        # until evidence turns up.
        sections = [transcript_text]
        if estimate_tokens(transcript_text) > self.max_tokens:
            sections = [c.render() for c in chunks]

        for section in sections:
            try:
                result = self.client.chat_json(
                    prompts.resolution_prompt(label, section),
                    schema=RESOLUTION_SCHEMA,
                    system=prompts.RESOLUTION_SYSTEM,
                    default={},
                )
            except Exception as exc:  # noqa: BLE001
                self.progress(f"  ! resolution check failed for '{item.task}': {exc}")
                continue

            if not result.get("resolved"):
                continue

            evidence = _as_text(result.get("evidence"))
            if not _evidence_supports_completion(evidence):
                self.progress(
                    f"  ! '{item.task}' claimed done, but the quoted evidence "
                    "reads as unfinished — keeping it open"
                )
                continue
            if not _quote_is_real(evidence, transcript_text):
                # The model said yes but could not point at anything that was
                # actually said. Closing an item on invented evidence is the
                # worst outcome here -- it makes real work disappear from the
                # open list -- so an unverifiable claim is refused.
                self.progress(
                    f"  ! '{item.task}' claimed done, but the quoted evidence "
                    "is not in the transcript — keeping it open"
                )
                continue

            self.progress(f"  resolved: {item.task}")
            return True

        return False

    # -- orchestration --------------------------------------------------------

    def run(
        self,
        transcript: Transcript,
        title: str | None = None,
        meeting_date: str | None = None,
        open_items: list[ActionItem] | None = None,
    ) -> ScrumNotes:
        chunks = split_chunks(transcript, max_tokens=self.max_tokens)
        self.progress(
            f"{len(transcript)} speaker turns, {transcript.word_count()} words "
            f"-> {len(chunks)} section(s)"
        )

        collected, failures = self.extract(chunks)
        notes = self.reduce(collected)

        if notes.action_items:
            self.progress(f"  consolidating {len(notes.action_items)} action item(s)")
            notes.action_items = self.consolidate(
                notes.action_items, transcript.speakers
            )

        notes.title = title or transcript.title or "Scrum meeting"
        notes.date = meeting_date or transcript.meeting_date or _date.today().isoformat()
        notes.participants = transcript.speakers
        notes.source = transcript.source
        notes.model = self.client.model
        notes.chunks = len(chunks)
        notes.failed_chunks = failures

        self.progress("  writing summary")
        notes.summary = self.summarise(notes)

        if open_items:
            self.progress(f"  checking {len(open_items)} carried-over action item(s)")
            resolved = self.detect_resolved(open_items, chunks)
            resolved_keys = {item.key() for item in resolved}
            for item in resolved:
                item.status = "done"
            notes.resolved = resolved
            notes.carried_over = [i for i in open_items if i.key() not in resolved_keys]

        return notes


def _evidence_supports_completion(quote: str) -> bool:
    """Reject evidence that proves the opposite of what was claimed.

    Provenance checking alone is not enough. Asked whether "start on RLS" was
    done, the model answers true and quotes "I haven not started it yet" -- a
    real sentence from the real transcript, which is why the quote check passes
    it. The quote is genuine; it simply proves the opposite.

    Rather than spend another model call on entailment, we screen the quote for
    negation and future tense. This is deliberately trigger-happy: a false
    negative leaves a finished item on the open list, which is mildly annoying,
    while a false positive deletes outstanding work from the only place it is
    recorded. When in doubt, stay open.
    """
    lowered = " " + quote.lower().replace("\\u2019", "'") + " "
    return not any(marker in lowered for marker in _NOT_DONE_MARKERS)


_NOT_DONE_MARKERS = (
    "haven't", "haven t", "have not", "hasn't", "has not", "didn't", "did not",
    "not started", "not yet", "isn't", "is not", "wasn't", "won't", "will not",
    "i'll", "we'll", "i will", "we will", "going to", "plan to", "planning to",
    "need to", "needs to", "should ", "next week", "this week", "still on",
    "still need", "yet", "promise", "first thing", "prioriti",
)


def _quote_is_real(quote: str, transcript: str, threshold: float = 0.8) -> bool:
    """Check a model-supplied quote actually appears in the transcript.

    Exact substring matching is too strict -- models normalise punctuation and
    drop filler when quoting. Requiring most of the quote's content words to be
    present catches fabrication while tolerating faithful paraphrase.
    """
    from .schema import _STOPWORDS

    def words(text: str) -> list[str]:
        cleaned = "".join(c if c.isalnum() else " " for c in text.lower())
        return [w for w in cleaned.split() if len(w) > 2 and w not in _STOPWORDS]

    quoted = words(quote)
    if len(quoted) < 3:
        return False
    haystack = set(words(transcript))
    hits = sum(1 for w in quoted if w in haystack)
    return hits / len(quoted) >= threshold


def _overlaps(a: str, b: str, threshold: float = 0.4) -> bool:
    """Loose word-overlap test used only to rescue dropped due dates."""
    from .schema import _STOPWORDS

    def words(text: str) -> set[str]:
        return {
            w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split()
            if len(w) > 2 and w not in _STOPWORDS
        }

    first, second = words(a), words(b)
    if not first or not second:
        return False
    return len(first & second) / min(len(first), len(second)) >= threshold


def facts_digest(notes: ScrumNotes, limit: int = 40) -> str:
    """Compact, model-readable rendering of the extracted facts."""
    lines: list[str] = []

    def add(label: str, values: list[str]) -> None:
        for value in values[:limit]:
            lines.append(f"{label}: {value}")

    add("Progress", [f"{i.owner} — {i.detail}" for i in notes.progress])
    add("Blocker", [f"{i.owner} — {i.detail}" for i in notes.blockers])
    add("Decision", notes.decisions)
    add("Feedback", [f"{i.owner} — {i.detail}" for i in notes.feedback])
    add("Action", [f"{a.owner} — {a.label()}" for a in notes.action_items])
    add("Question", notes.questions)

    return "\n".join(lines)
