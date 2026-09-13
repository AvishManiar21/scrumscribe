# ScrumScribe

Turns meeting transcripts into structured scrum notes, and remembers what you
promised last week.

Runs entirely on your own machine against a local Ollama model. No API keys, no
cloud service, no subscription, and no audio of your supervisor leaving your
laptop.

```
$ scrumscribe notes weekly-scrum.vtt

26 speaker turns, 440 words -> 2 section(s)
  reading section 1/2 (00:12-03:44)
  reading section 2/2 (04:00-04:36)
  consolidating 7 action item(s)
  merged 7 action items into 4
  checking 4 carried-over action item(s)
  resolved: Send the schema document
  resolved: Batch requests and appointment history

STILL OPEN FROM BEFORE
  - Avish: Start on RLS (open 3 meetings)

PROGRESS
  - Avish: Appointment history fully loaded, four years of it
  - Avish: Batching working, rate limit solved

BLOCKERS
  - Avish: Supabase free tier nearing its row limit

ACTION ITEMS
  - Avish: Start row level security — due this week
  - Avish: Send Dr. Chen the Supabase pricing — due this week
```

## Why this exists

There are good open-source tools that record a meeting and summarise it —
[Meetily](https://github.com/Zackriya-Solutions/meetily) is excellent and is
what I use for capture. What none of them do is carry state between meetings.

A summariser answers *what was said today*. In a weekly scrum with a supervisor,
the question that actually matters is *what did I commit to three weeks ago that
I still have not done*. ScrumScribe tracks that: every commitment is recorded,
matched against later meetings, and flagged when it has been open too long.

## Install

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
git clone https://github.com/<you>/scrumscribe
cd scrumscribe
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -e .

ollama pull gemma3:4b
scrumscribe doctor
```

`doctor` checks Ollama, the model, memory, and any Meetily database it can find,
and tells you exactly what to run if something is missing.

## Use

```bash
# From any transcript: Teams .vtt export, Whisper .json, plain text
scrumscribe notes weekly-scrum.vtt

# Straight from Meetily's database
scrumscribe meetily --list
scrumscribe meetily                    # exports the most recent meeting
scrumscribe notes ~/.scrumscribe/transcripts/<id>.txt

# Cross-meeting memory
scrumscribe open                       # what is still outstanding
scrumscribe close "schema doc"         # close one by hand
scrumscribe history                    # past meetings

# Draft your next status update from open items + your actual commits
scrumscribe standup --repo ~/projects/dashboard
```

Notes and memory live in `~/.scrumscribe` by default — deliberately outside the
repo, so meeting content is never at risk of being committed. Override with
`SCRUMSCRIBE_HOME`.

## Input formats

| Source | Detection |
|---|---|
| Meetily database | SQLite magic bytes, schema discovered at runtime |
| Microsoft Teams `.vtt` | `WEBVTT` header, `<v Speaker>` tags |
| Whisper / faster-whisper `.json` | segment list found by structure, fields by alias |
| `.srt` | cue timestamps |
| Plain text | `Speaker: line` convention |

Format is detected from content, not the file extension.

## How it works

```
transcript ──► ingest ──► speaker turns ──► disjoint chunks
                                                  │
                                    ┌─────────────┴─────────────┐
                                    │  MAP: extract per section │
                                    └─────────────┬─────────────┘
                                                  ▼
                                    REDUCE: dedupe + consolidate
                                                  ▼
                                    RECONCILE: against memory ──► memory.json
                                                  ▼
                                             notes.md
```

Design decisions that matter, and why:

**Capture is somebody else's job.** Meetily already solves recording well.
Building another audio pipeline would have been the fragile part of this project
for no gain. ScrumScribe starts where a transcript exists.

**Map-reduce, not one big prompt.** A 4B model handed a whole meeting writes
confident notes about the first and last five minutes and quietly ignores the
middle. Extracting per-section and merging afterwards costs more calls and is
the difference between notes you trust and notes you have to check.

**Chunks are disjoint; overlap is context only.** The obvious implementation
copies a tail of utterances into the next window. When a window holds few
utterances that carried tail can equal the whole window, the splitter stops
advancing, and later chunks silently re-contain the entire transcript. Windows
here own their content exclusively, so forward progress is structural.

**Decoding is schema-constrained.** Ollama's `format` parameter is given a real
JSON Schema, so output is valid by construction. Asking a small model politely
for JSON produces parse failures on a meaningful fraction of chunks.

**Resolution is retrieval first, judgement second.** Deciding which past
commitments a meeting closed is the hardest part of this, and the obvious
designs all fail at 4B. Asked *"which of these six numbered items were
resolved"*, the model returns an empty list essentially always. Asked about one
item and told to quote its evidence, it returns a real, completion-sounding
sentence about a **different** task — asked whether "start on RLS" was done, it
quotes *"Yes, I sent it Thursday night"*, which passes every check because the
quote is genuine and does describe a completion.

So the search is done deterministically. Lines sharing vocabulary with the task
are retrieved by stem match, ranked by strength, and pulled in with their
neighbours — the proof is usually the reply, not the line that matched. The
model then only judges lines already known to be on topic, which it does well.

**Every gate fails closed.** A claimed resolution must quote real transcript
text, must not read as future tense or negation, and must survive a second
model call that sees the quote alone with no surrounding context to be swayed
by. Any failure leaves the item open. Leaving finished work on the list is
mildly annoying; deleting unfinished work from the only place it is recorded is
the failure this tool must not have — so the bias is deliberate and one-sided.

**Partial coverage is never hidden.** If sections fail, the count appears in the
notes. A note that silently dropped three of seventeen sections looks exactly
like a complete one, and that is how you end up trusting a lie.

## Known limitations

Resolution is tuned to under-close rather than over-close. On the test meetings
it correctly closes clear completions ("yes, I sent it Thursday night") and
correctly refuses explicit non-completions ("I haven't started it yet"), but
leaves genuinely-finished-but-hedged items open — *"I emailed her but no reply
yet, I'll follow up"* stays on the list. Close those by hand with
`scrumscribe close`.

Extraction quality varies run to run at this model size; consolidation usually
merges duplicate action items but not always. Re-running is cheap and the
transcript is unchanged, so a poor run costs nothing but time.

## Model choice

Default is `gemma3:4b` (~3.3 GB), which runs comfortably on a 6 GB laptop GPU
alongside a Whisper model. Anything Ollama serves works:

```bash
scrumscribe --model qwen3:8b notes meeting.vtt
export SCRUMSCRIBE_MODEL=llama3.2   # or set a default
```

Larger models give better summaries. The extraction prompts are tuned for the
4B tier, so they hold up on anything bigger.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite is entirely offline — no model calls. It covers parsing, the chunk
partition invariant (every utterance in exactly one chunk, at every window
size), deduplication, atomic persistence, and corruption recovery. The
deterministic machinery is what must not regress; model output is checked by
running it.

## Licence

MIT
