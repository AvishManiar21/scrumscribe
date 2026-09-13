"""Core transcript data model.

Every ingest source normalises down to a Transcript of Utterances, so the
agent layer never needs to know whether the text came from Meetily's SQLite
store, a Teams .vtt export, or a plain text file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Utterance:
    """A single continuous chunk of speech by one speaker."""

    text: str
    start: float | None = None  # seconds from meeting start
    end: float | None = None
    speaker: str | None = None

    @property
    def duration(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return max(0.0, self.end - self.start)

    def timestamp(self) -> str:
        """Human-readable mm:ss, or empty string when untimed."""
        if self.start is None:
            return ""
        minutes, seconds = divmod(int(self.start), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def render(self) -> str:
        ts = self.timestamp()
        who = self.speaker or "Unknown"
        prefix = f"[{ts}] {who}:" if ts else f"{who}:"
        return f"{prefix} {self.text}"


@dataclass
class Transcript:
    """A full meeting transcript plus whatever provenance we could recover."""

    utterances: list[Utterance] = field(default_factory=list)
    source: str = "unknown"
    title: str | None = None
    meeting_date: str | None = None  # ISO date, best effort

    def __len__(self) -> int:
        return len(self.utterances)

    def __bool__(self) -> bool:
        return bool(self.utterances)

    @property
    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for utt in self.utterances:
            if utt.speaker:
                seen.setdefault(utt.speaker, None)
        return list(seen)

    @property
    def duration(self) -> float | None:
        ends = [u.end for u in self.utterances if u.end is not None]
        return max(ends) if ends else None

    def word_count(self) -> int:
        return sum(len(u.text.split()) for u in self.utterances)

    def render(self) -> str:
        return "\n".join(u.render() for u in self.utterances)

    def merge_consecutive(self, max_gap: float = 2.0) -> "Transcript":
        """Collapse adjacent utterances from the same speaker.

        Whisper and VTT sources emit one cue every few seconds, which shreds
        sentences across cues and wrecks summarisation quality. Merging back
        into speaker turns gives the model coherent paragraphs to read.
        """
        merged: list[Utterance] = []
        for utt in self.utterances:
            prev = merged[-1] if merged else None
            same_speaker = prev is not None and prev.speaker == utt.speaker
            gap_ok = (
                prev is not None
                and prev.end is not None
                and utt.start is not None
                and (utt.start - prev.end) <= max_gap
            )
            # Untimed sources (plain text) merge on speaker alone.
            untimed = prev is not None and prev.end is None and utt.start is None

            if same_speaker and (gap_ok or untimed):
                prev.text = f"{prev.text} {utt.text}".strip()
                if utt.end is not None:
                    prev.end = utt.end
            else:
                merged.append(
                    Utterance(
                        text=utt.text,
                        start=utt.start,
                        end=utt.end,
                        speaker=utt.speaker,
                    )
                )

        return Transcript(
            utterances=merged,
            source=self.source,
            title=self.title,
            meeting_date=self.meeting_date,
        )


_FILLER = re.compile(
    r"\b(?:um+|uh+|erm+|ah+|hmm+|mm+|like,|you know,|i mean,)\b",
    re.IGNORECASE,
)
_REPEAT = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Strip transcription noise that costs tokens and adds nothing.

    Speech-to-text output is full of fillers and stutter repeats. On a 4B
    model every wasted token is context we cannot spare.
    """
    text = _FILLER.sub(" ", text)
    text = _REPEAT.sub(r"\1", text)
    text = _WS.sub(" ", text)
    return text.strip(" ,.-")
