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
# Resolution judges retrieved candidate lines rather than searching the
# whole transcript. Asked to find evidence in a full meeting, a 4B model
# reliably returns a real, completion-sounding sentence about a different
# task -- every downstream check passes it because the quote is genuine.
# Deterministic retrieval removes the search problem and leaves the model
# only the judgement, which it does well.
RESOLUTION_SYSTEM = """\
You decide whether a task from a previous meeting has been completed.

You are shown the lines from the current meeting that mention the task. Read
them and decide.

Completed means someone states in the past tense that it happened: "I sent it",
"it is loaded now", "that is done", "solved", "I emailed her".

Not completed means it was only planned, promised, requested, or explicitly not
started: "I will do it this week", "I have not started", "that needs to happen",
"so X first, then Y".

If the lines only mention the task without saying it happened, it is not
completed. Quote the exact line that decided it.
"""


def resolution_prompt(item: str, candidates: str) -> str:
    return f"""\
Task from the previous meeting:
  {item}

Lines from the current meeting that mention this task:
{candidates}

Has this task been completed?
"""


STANDUP_SYSTEM = """\
You draft a short scrum status update the user will read aloud in their next
meeting.

Write three labelled sections: "Since last time", "Blocked on", "Next".

The outstanding commitments you are given are things the user has NOT done yet.
Never describe them as completed or started. They belong under "Next", or under
"Blocked on" if something is stopping them.

"Since last time" may only contain work evidenced by the git commits. If there
are no commits, write "Nothing" under it rather than inventing progress.

Use short bullets. Do not invent work that is not in the inputs.
"""


def standup_prompt(open_items: list[str], commits: list[str], last_summary: str) -> str:
    items = """\n""".join(f"- {i}" for i in open_items) or "- (none)"
    log = """\n""".join(f"- {c}" for c in commits) or "- (no commits found)"
    return f"""\
Outstanding commitments I have NOT completed yet:
{items}

What the last meeting covered:
{last_summary}

My git commits since then (this is the only evidence of completed work):
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

# The second half of resolution. The search step is given the whole
# transcript and is prone to returning a real sentence that mentions the
# task without stating it was done -- a closing agenda line like "so RLS
# first, then pricing" reads as relevant, and keyword screening cannot
# tell it apart. This judge sees only the task and the quoted sentence,
# with no surrounding context to be swayed by, and is reliably correct on
# exactly the cases the search step gets wrong.
CONFIRM_SYSTEM = """\
You are given a task and a single sentence from a meeting.

Decide whether that sentence, on its own, states that the task was ALREADY
COMPLETED in the past.

Answer true only for past-tense completion: "I sent it", "it is loaded now",
"that is done", "solved".
Answer false for anything else, including plans, reminders, agendas, requests,
lists of what to do next, or statements that the task has not been started.
"""


def confirm_prompt(item: str, quote: str) -> str:
    return f"""\
Task:
  {item}

Sentence from the meeting:
  {quote}

Does this sentence say the task was already completed?
"""
