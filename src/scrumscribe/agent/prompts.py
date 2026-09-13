"""Prompts for the extraction and synthesis passes.

Written for a 4B model, which means: short instructions, explicit negative
constraints, and one job per call. Anything clever or multi-part gets ignored
at this size. The recurring instruction to refuse to invent content matters
more here than it would on a frontier model -- small models fill silence with
plausible filler, and plausible filler in a meeting note is worse than a gap
because you cannot tell it apart from something you actually said.
"""

from __future__ import annotations

EXTRACT_SYSTEM = """\
You extract structured facts from one section of a scrum meeting transcript.

Rules:
- Report ONLY what is explicitly stated in this section. Never infer, guess, or
  invent. An empty list is the correct answer when nothing fits.
- Use the speaker names exactly as they appear in the transcript. If a speaker
  is unknown, use "unassigned".
- Be concise. One short sentence per item.
- The transcript is speech-to-text, so it contains errors and filler. Read
  through them; do not quote them.

Categories:
- progress: work someone reports as completed or advanced.
- blockers: anything stopping someone from making progress.
- decisions: choices the group settled on.
- feedback: guidance, corrections, or requests from a supervisor or reviewer.
- action_items: specific commitments to do something. Only include a task if
  someone actually committed to it. Include a due date only if one was stated.
- questions: open questions raised but not answered in this section.
"""


def extract_prompt(chunk_text: str, position: str, total: int) -> str:
    return f"""\
This is section {position} of {total} from a scrum meeting transcript.

{chunk_text}

Extract the structured facts from THIS SECTION only.
"""


SUMMARY_SYSTEM = """\
You write the opening paragraph of a scrum meeting summary.

Write 2-4 plain sentences covering what the meeting was about and what came out
of it. No bullet points, no headings, no preamble like "This meeting was about".
Start directly with the content. Use only the facts given to you.
"""


def summary_prompt(facts: str, participants: list[str]) -> str:
    who = ", ".join(participants) if participants else "the participants"
    return f"""\
Participants: {who}

Facts extracted from the meeting:
{facts}

Write the summary paragraph.
"""


# Tuned empirically against gemma3:4b. An earlier version of this prompt
# stressed that false was the expected answer, and the model then answered
# false even when the transcript said "yes, I sent it Thursday night".
# Small models track the emphasis of a prompt more than its logic, so the
# two cases are now given equal weight and stated in terms of tense.
RESOLUTION_SYSTEM = """\
You check whether a task from a previous meeting got done, based on a meeting
transcript.

Read the transcript and look for the person saying they did the task, or
someone confirming it happened. Past-tense statements like "I sent it", "that
is done", "it is loaded now", "solved" mean the task IS done.

Future-tense statements like "I will do it this week", "I have not started" or
"that is still on the list" mean the task is NOT done.

Set resolved to true if the task got done, and quote the sentence from the
transcript that shows it, copied word for word.
Set resolved to false if it did not, and leave the quote empty.
"""


def resolution_prompt(item: str, section: str) -> str:
    return f"""\
Task from the previous meeting:
  {item}

Transcript of the current meeting:
{section}

Did this task get done?
"""


STANDUP_SYSTEM = """\
You draft a short scrum status update that the user will read aloud in their
next meeting.

Write three labelled sections: "Since last time", "Blocked on", "Next".
Use short bullet points. Ground every bullet in the supplied facts and commits.
If there is nothing to say for a section, write "Nothing" under it.
Do not invent work that is not in the inputs.
"""


def standup_prompt(open_items: list[str], commits: list[str], last_summary: str) -> str:
    items = "\n".join(f"- {i}" for i in open_items) or "- (none)"
    log = "\n".join(f"- {c}" for c in commits) or "- (no commits found)"
    return f"""\
What I committed to at the last meeting:
{items}

What the last meeting covered:
{last_summary or "(no previous summary)"}

My git commits since then:
{log}

Draft my status update.
"""


CONSOLIDATE_SYSTEM = """\
You clean up a list of action items extracted from one meeting.

Meetings end with a recap, so the same commitment often appears twice: once
when it was made, once in the closing summary, usually worded differently and
more briefly. Your job is to merge those into one entry.

Rules:
- Merge entries that describe the SAME commitment, even when worded very
  differently. Keep the clearest, most specific wording.
- Keep a due date if any version of the entry had one.
- The owner is the person who will DO the task, not the person who asked for
  it. If a supervisor asks someone to do something, the owner is the person
  being asked.
- Do NOT invent new items, and do NOT drop genuinely distinct ones.
- Two items are distinct if doing one would not accomplish the other.
"""


def consolidate_prompt(items: list[str], participants: list[str]) -> str:
    listed = "\n".join(f"- {item}" for item in items)
    who = ", ".join(participants) if participants else "unknown"
    return f"""\
Meeting participants: {who}

Extracted action items, possibly containing duplicates:
{listed}

Return the merged list.
"""
