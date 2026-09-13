"""Transcript chunking for small-context models.

A 45-minute meeting is roughly 7-9k tokens of messy spoken text. A 4B model
technically accepts that in one prompt, but quality collapses well before the
context limit: it latches onto the first and last few minutes and silently
drops the middle. Chunking with overlap is what makes a small model produce
notes comparable to a large one.

Design note: windows are *disjoint*. Overlap is supplied separately as
read-only lead-in context rather than by copying utterances into the next
window. Copying is the obvious implementation and it is wrong -- when a window
holds few utterances the carried tail can equal the whole window, so the
splitter stops advancing and later chunks silently re-contain the entire
transcript. Disjoint ownership makes forward progress structural, and means
each chunk's findings map to exactly one span of the meeting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..transcript import Transcript, Utterance

# Rough English heuristic. Deliberately conservative -- underestimating the
# token count is what causes silent truncation mid-chunk.
CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass
class Chunk:
    """A window of consecutive utterances small enough to reason over.

    `utterances` are owned exclusively by this chunk. `context` is the tail of
    the previous chunk, shown to the model for continuity but never attributed
    to this chunk.
    """

    index: int
    total: int
    utterances: list[Utterance]
    context: list[Utterance] = field(default_factory=list)

    @property
    def start(self) -> float | None:
        starts = [u.start for u in self.utterances if u.start is not None]
        return min(starts) if starts else None

    @property
    def end(self) -> float | None:
        ends = [u.end for u in self.utterances if u.end is not None]
        return max(ends) if ends else None

    def time_range(self) -> str:
        if not self.utterances or self.start is None:
            return f"part {self.index + 1} of {self.total}"
        return f"{self.utterances[0].timestamp()}-{self.utterances[-1].timestamp()}"

    def render(self) -> str:
        """The chunk's own content."""
        return "\n".join(u.render() for u in self.utterances)

    def render_with_context(self) -> str:
        """Prompt-ready text, with any lead-in clearly marked as prior context."""
        body = self.render()
        if not self.context:
            return body
        lead = "\n".join(u.render() for u in self.context)
        return (
            "--- earlier context (already summarised, do not re-report) ---\n"
            f"{lead}\n"
            "--- this section ---\n"
            f"{body}"
        )

    def tokens(self) -> int:
        return estimate_tokens(self.render_with_context())


def chunk(
    transcript: Transcript,
    max_tokens: int = 1800,
    overlap_utterances: int = 2,
) -> list[Chunk]:
    """Split a transcript into disjoint windows on speaker-turn boundaries.

    Chunks never split mid-utterance. An utterance larger than `max_tokens` on
    its own becomes its own chunk rather than being dropped. Overlap is capped
    at a quarter of the budget so lead-in context can never crowd out content.
    """
    if not transcript.utterances:
        return []

    windows: list[list[Utterance]] = []
    current: list[Utterance] = []
    current_tokens = 0

    for utt in transcript.utterances:
        cost = estimate_tokens(utt.render())

        # Close the window only if it already holds something; a single
        # oversized utterance gets a window to itself.
        if current and current_tokens + cost > max_tokens:
            windows.append(current)
            current = []
            current_tokens = 0

        current.append(utt)
        current_tokens += cost

    if current:
        windows.append(current)

    overlap_budget = max_tokens // 4
    chunks: list[Chunk] = []
    for i, window in enumerate(windows):
        context: list[Utterance] = []
        if i > 0 and overlap_utterances > 0:
            spent = 0
            for utt in reversed(windows[i - 1][-overlap_utterances:]):
                cost = estimate_tokens(utt.render())
                if spent + cost > overlap_budget:
                    break
                context.insert(0, utt)
                spent += cost

        chunks.append(
            Chunk(index=i, total=len(windows), utterances=window, context=context)
        )

    return chunks
