"""Offline test suite.

Nothing here touches Ollama. These tests cover the deterministic machinery --
parsing, chunking, deduplication, persistence -- which is exactly the part that
must not regress, and the part a model-dependent test could never pin down.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from scrumscribe.agent.loop import (
    _dedupe_actions,
    _dedupe_items,
    _evidence_supports_completion,
    _parse_actions,
    _retrieve,
    _parse_items,
    _quote_is_real,
)
from scrumscribe.agent.schema import ActionItem, Item, ScrumNotes
from scrumscribe.ingest import detect, load
from scrumscribe.ingest import meetily_folder
from scrumscribe.ingest.meetily import (
    MeetilyStore,
    default_db_paths,
    find_db,
    is_sqlite,
)
from scrumscribe.memory import Memory
from scrumscribe.model import chunk, estimate_tokens
from scrumscribe.model.ollama import _salvage_json
from scrumscribe.render import to_markdown, to_terminal
from scrumscribe.transcript import Transcript, Utterance, clean

FIXTURES = Path(__file__).parent / "fixtures"


# -- transcript model ---------------------------------------------------------


def test_merge_consecutive_joins_same_speaker():
    t = Transcript(
        [
            Utterance("I finished the parser", 0, 3, "Avish"),
            Utterance("and the eval script.", 3.5, 6, "Avish"),
            Utterance("Good.", 7, 8, "Chen"),
        ]
    )
    merged = t.merge_consecutive()
    assert len(merged) == 2
    assert merged.utterances[0].text == "I finished the parser and the eval script."
    assert merged.utterances[0].end == 6


def test_merge_respects_gap_threshold():
    t = Transcript(
        [
            Utterance("First thought.", 0, 3, "Avish"),
            Utterance("Much later thought.", 60, 63, "Avish"),
        ]
    )
    assert len(t.merge_consecutive(max_gap=2.0)) == 2


def test_clean_strips_fillers_and_stutters():
    assert "um" not in clean("Um, I I finished it").lower()
    assert clean("I I finished it") == "I finished it"


def test_speakers_preserve_order():
    t = Transcript(
        [
            Utterance("a", speaker="Chen"),
            Utterance("b", speaker="Avish"),
            Utterance("c", speaker="Chen"),
        ]
    )
    assert t.speakers == ["Chen", "Avish"]


# -- ingest -------------------------------------------------------------------


def test_detect_and_load_vtt():
    path = FIXTURES / "teams.vtt"
    assert detect(path) == "vtt"
    t = load(path)
    assert "Avish Maniar" in t.speakers
    assert t.utterances[0].start == pytest.approx(1.0)


def test_vtt_voice_tags_are_stripped():
    t = load(FIXTURES / "teams.vtt")
    assert all("<v" not in u.text for u in t.utterances)


def test_detect_and_load_json():
    assert detect(FIXTURES / "whisper.json") == "json"
    t = load(FIXTURES / "whisper.json")
    assert len(t) >= 2
    assert t.speakers


def test_detect_and_load_text():
    assert detect(FIXTURES / "plain.txt") == "text"
    t = load(FIXTURES / "plain.txt")
    assert t.speakers == ["Avish", "Dr. Chen"]


def test_sqlite_detected_by_magic_bytes(tmp_path):
    """Extension is irrelevant; Meetily has shipped several."""
    odd = tmp_path / "store.bin"
    odd.write_bytes((FIXTURES / "fake_meetily.db").read_bytes())
    assert detect(odd) == "meetily-sqlite"


def test_meetily_schema_discovery_picks_transcript_table():
    with MeetilyStore(FIXTURES / "fake_meetily.db") as store:
        best = store.best
        assert best.table == "transcript_chunks"
        assert best.text == "transcript_text"
        assert best.speaker == "speaker_label"
        # The long-blob summaries table must never outrank the real one.
        assert best.score > max(t.score for t in store.tables[1:])


def test_meetily_lists_and_filters_meetings():
    with MeetilyStore(FIXTURES / "fake_meetily.db") as store:
        meetings = store.list_meetings()
        assert [m["id"] for m in meetings] == ["m1", "m2"]  # newest first
        assert len(store.load("m1")) == 4
        assert len(store.load("m2")) == 1


def test_meetily_opened_read_only(tmp_path):
    copy = tmp_path / "ro.db"
    copy.write_bytes((FIXTURES / "fake_meetily.db").read_bytes())
    with MeetilyStore(copy) as store:
        with pytest.raises(sqlite3.OperationalError):
            store.conn.execute("CREATE TABLE nope (x INTEGER)")


def test_empty_transcript_raises_useful_error(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(ValueError, match="no utterances"):
        load(empty)


# -- chunking -----------------------------------------------------------------


def _long_transcript(n: int) -> Transcript:
    return Transcript(
        [Utterance("word " * 40, i * 10.0, i * 10.0 + 8, f"S{i % 3}") for i in range(n)]
    )


@pytest.mark.parametrize("n,max_tokens", [(1, 1800), (5, 100), (50, 400), (500, 1800)])
def test_chunks_partition_transcript_exactly(n, max_tokens):
    """Every utterance appears in exactly one chunk. No loss, no duplication."""
    t = _long_transcript(n)
    chunks = chunk(t, max_tokens=max_tokens)
    owned = [u for c in chunks for u in c.utterances]
    assert len(owned) == n
    assert [id(u) for u in owned] == [id(u) for u in t.utterances]


def test_oversized_utterance_gets_its_own_chunk():
    t = Transcript([Utterance("word " * 5000, 0, 10, "Avish")])
    chunks = chunk(t, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].utterances


def test_overlap_is_context_not_content():
    t = _long_transcript(20)
    chunks = chunk(t, max_tokens=300, overlap_utterances=2)
    assert len(chunks) > 1
    assert chunks[1].context
    # Context must be lead-in from the previous chunk, never owned content.
    assert not set(id(u) for u in chunks[1].context) & set(
        id(u) for u in chunks[1].utterances
    )
    assert "earlier context" in chunks[1].render_with_context()


def test_empty_transcript_yields_no_chunks():
    assert chunk(Transcript([])) == []


def test_token_estimate_is_monotonic():
    assert estimate_tokens("a" * 100) < estimate_tokens("a" * 200)


# -- extraction parsing and dedupe --------------------------------------------

def test_placeholder_due_dates_are_dropped():
    actions = _parse_actions(
        [
            {"owner": "Avish", "task": "Email Patel", "due": "unassigned"},
            {"owner": "Avish", "task": "Start RLS", "due": "N/A"},
            {"owner": "Avish", "task": "Send schema", "due": "Friday"},
        ]
    )
    assert [a.due for a in actions] == ["", "", "Friday"]


def test_parse_actions_tolerates_bare_strings():
    actions = _parse_actions(["do the thing", {"task": "other thing"}])
    assert [a.task for a in actions] == ["do the thing", "other thing"]


def test_parse_items_ignores_empty_details():
    assert _parse_items([{"owner": "A", "detail": ""}, {"owner": "A", "detail": "x"}]) == [
        Item("A", "x")
    ]


def test_dedupe_prefers_named_owner_and_longer_text():
    merged = _dedupe_items(
        [
            Item("unassigned", "finished the WCOnline sync"),
            Item("Avish", "Finished the WCOnline sync."),
        ]
    )
    assert len(merged) == 1
    assert merged[0].owner == "Avish"


def test_dedupe_actions_keeps_due_date():
    merged = _dedupe_actions(
        [
            ActionItem("Avish", "send the schema doc", ""),
            ActionItem("Avish", "Send the schema doc", "Friday"),
        ]
    )
    assert len(merged) == 1
    assert merged[0].due == "Friday"


def test_dedupe_resolves_conflicting_owners():
    """The doer owns more items than the requester, so the doer wins."""
    merged = _dedupe_actions(
        [
            ActionItem("Avish", "Email Dr. Patel"),
            ActionItem("Dr. Chen", "Email Dr. Patel"),
            ActionItem("Avish", "Start on RLS"),
            ActionItem("Avish", "Send schema doc", "Friday"),
        ]
    )
    tasks = {a.task: a.owner for a in merged}
    assert len(merged) == 3
    assert tasks["Email Dr. Patel"] == "Avish"


def test_distinct_tasks_survive_dedupe():
    merged = _dedupe_actions(
        [ActionItem("Avish", "Start on RLS"), ActionItem("Avish", "Email Dr. Patel")]
    )
    assert len(merged) == 2


# -- memory -------------------------------------------------------------------


def test_memory_tracks_carry_over_and_resolution(tmp_path):
    memory = Memory(tmp_path / "memory.json")

    week1 = ScrumNotes(
        title="Scrum",
        date="2026-09-01",
        action_items=[
            ActionItem("Avish", "send schema doc", "Friday"),
            ActionItem("Avish", "batch the API requests"),
        ],
    )
    assert memory.record(week1) == {"added": 2, "closed": 0, "carried": 0}

    week2 = ScrumNotes(
        title="Scrum",
        date="2026-09-08",
        action_items=[ActionItem("Avish", "send schema doc", "Friday")],
        resolved=[ActionItem("Avish", "batch the API requests")],
    )
    report = memory.record(week2)
    assert report["closed"] == 1
    assert report["carried"] == 1

    open_items = memory.open_items()
    assert len(open_items) == 1
    assert open_items[0].meetings == 2  # survived two meetings


def test_memory_survives_reload(tmp_path):
    path = tmp_path / "memory.json"
    Memory(path).record(
        ScrumNotes(title="S", date="2026-09-01", action_items=[ActionItem("A", "task one")])
    )
    assert len(Memory(path).open_items()) == 1


def test_memory_recovers_from_corruption(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ this is not json", encoding="utf-8")
    memory = Memory(path)  # must not raise
    assert memory.open_items() == []
    assert path.with_suffix(".corrupt.json").exists()


def test_rerunning_a_meeting_does_not_duplicate_it(tmp_path):
    memory = Memory(tmp_path / "memory.json")
    notes = ScrumNotes(title="Scrum", date="2026-09-01", action_items=[ActionItem("A", "t")])
    memory.record(notes)
    memory.record(notes)
    assert len(memory.meetings()) == 1
    assert len(memory.open_items()) == 1


def test_close_item_by_text_and_by_key(tmp_path):
    memory = Memory(tmp_path / "memory.json")
    memory.record(
        ScrumNotes(
            title="S",
            date="2026-09-01",
            action_items=[ActionItem("A", "email Dr. Patel"), ActionItem("A", "start RLS")],
        )
    )
    assert memory.close_item("patel") is not None
    remaining = memory.open_items()
    assert len(remaining) == 1
    assert memory.close_item(remaining[0].key()) is not None
    assert memory.open_items() == []


def test_memory_write_is_atomic(tmp_path):
    path = tmp_path / "memory.json"
    memory = Memory(path)
    memory.record(ScrumNotes(title="S", date="2026-09-01"))
    # No temporary files left behind, and the result is valid JSON.
    assert not list(tmp_path.glob(".memory-*.tmp"))
    json.loads(path.read_text(encoding="utf-8"))


# -- rendering ----------------------------------------------------------------


def test_markdown_omits_empty_sections():
    md = to_markdown(ScrumNotes(title="S", date="2026-09-01", summary="Nothing much."))
    assert "## Blockers" not in md
    assert "## Action items" not in md


def test_markdown_flags_partial_coverage():
    notes = ScrumNotes(title="S", date="2026-09-01", chunks=17, failed_chunks=3)
    md = to_markdown(notes)
    assert "incomplete" in md
    assert "3 of 17" in md


def test_markdown_marks_long_running_items():
    notes = ScrumNotes(
        title="S",
        date="2026-09-01",
        carried_over=[ActionItem("Avish", "start RLS", meetings=4)],
    )
    md = to_markdown(notes)
    assert "open for 4 meetings" in md
    assert "Still open from previous meetings" in md


def test_terminal_render_warns_on_failed_chunks():
    out = to_terminal(ScrumNotes(title="S", date="x", chunks=5, failed_chunks=2))
    assert "WARNING" in out


# -- resolution safety rails --------------------------------------------------


@pytest.mark.parametrize(
    "quote,expected",
    [
        ("Avish: Yes, I sent it Thursday night.", True),
        ("Avish: Batching is working, appointment history is fully loaded now.", True),
        ("Avish: Solved, yeah. I pulled it in monthly windows.", True),
        ("Avish: I haven't started it yet, it's been on the list two weeks.", False),
        ("Avish: I'll do it this week, I promise.", False),
        ("Avish: I emailed her but no reply yet. I'll follow up.", False),
        ("Dr. Chen: That one needs to happen before the demo.", False),
    ],
)
def test_evidence_screen_rejects_unfinished_work(quote, expected):
    """The screen fails closed: ambiguity leaves an item open."""
    assert _evidence_supports_completion(quote) is expected


def test_fabricated_quotes_are_rejected():
    transcript = "[00:14] Avish: Yes, I sent the schema document Thursday night."
    assert _quote_is_real("I sent the schema document Thursday", transcript)
    assert not _quote_is_real(
        "I deployed the authentication middleware to production", transcript
    )


def test_quote_must_be_substantial():
    """Too short to verify is treated as unverifiable."""
    assert not _quote_is_real("yes", "Avish: yes it is done")


# -- srt and filler stripping -------------------------------------------------


def test_detect_and_load_srt():
    path = FIXTURES / "meeting.srt"
    assert detect(path) == "srt"
    t = load(path)
    assert t.speakers == ["Avish", "Dr. Chen"]
    assert t.utterances[0].start == pytest.approx(1.0)


def test_srt_detected_without_extension(tmp_path):
    odd = tmp_path / "transcript.log"
    odd.write_text((FIXTURES / "meeting.srt").read_text(encoding="utf-8"), encoding="utf-8")
    assert detect(odd) == "srt"


def test_pipeline_strips_fillers_by_default():
    t = load(FIXTURES / "meeting.srt")
    assert not t.utterances[0].text.lower().startswith("um")


def test_verbatim_mode_keeps_fillers():
    t = load(FIXTURES / "meeting.srt", strip_fillers=False)
    assert t.utterances[0].text.lower().startswith("um")


def test_cleaning_keeps_terminal_punctuation():
    assert clean("Um, so I finished the parser.").endswith(".")
    assert clean("Uh, what is blocking you?").endswith("?")


def test_cleaning_never_empties_an_utterance():
    """An all-filler turn still records that somebody spoke."""
    t = Transcript([Utterance("Um, uh, um", 0, 2, "Avish")])
    for utt in t.utterances:
        stripped = clean(utt.text)
        if stripped:
            utt.text = stripped
    assert t.utterances[0].text


# -- malformed model output ---------------------------------------------------


def test_salvage_recovers_truncated_string():
    """A model that runs out of output budget stops mid-string."""
    raw = '{\n  "summary": "The team discussed the WCOnline integration and decided to store raw payl'
    recovered = _salvage_json(raw)
    assert recovered is not None
    assert recovered["summary"].startswith("The team discussed")


def test_salvage_recovers_truncated_array():
    raw = '{"action_items": [{"owner": "Avish", "task": "Send sch'
    recovered = _salvage_json(raw)
    assert recovered["action_items"][0]["owner"] == "Avish"


def test_salvage_strips_surrounding_prose():
    assert _salvage_json('Sure! {"summary": "done"} hope that helps') == {"summary": "done"}


def test_salvage_gives_up_on_garbage():
    assert _salvage_json("no json here at all") is None


def test_salvage_handles_escaped_quotes():
    raw = '{"summary": "he said \\"done\\" and left'
    recovered = _salvage_json(raw)
    assert recovered is not None
    assert "done" in recovered["summary"]


def test_lenient_parsing_allows_raw_newlines():
    """Constrained decoding fixes the shape, not control-character escaping."""
    assert _salvage_json('{"summary": "line one\nline two"}')["summary"].count("\n") == 1


# -- evidence retrieval -------------------------------------------------------


def _week2():
    return load(FIXTURES / "scrum_week2.txt")


def test_retrieval_finds_the_reply_not_just_the_mention():
    """The proof is usually the answer to the question that matched."""
    lines = _retrieve("Send the schema document", _week2().utterances)
    joined = " ".join(lines)
    assert "schema document over" in joined      # the mention
    assert "I sent it Thursday night" in joined  # the proof


def test_retrieval_window_is_forward_only():
    """Preceding lines belong to the previous topic and contaminate the judge.

    "Solved, yeah..." sits immediately before the RLS exchange and refers to
    the rate limit. Pulled in as context it is enough to convince the model
    that RLS was finished.
    """
    lines = _retrieve("Start on RLS", _week2().utterances)
    joined = " ".join(lines)
    assert "haven't started it yet" in joined
    assert "Solved, yeah" not in joined


def test_retrieval_returns_nothing_for_unrelated_task():
    assert _retrieve("Deploy the Kubernetes ingress controller", _week2().utterances) == []


def test_retrieval_ignores_stopword_only_tasks():
    assert _retrieve("do the it", _week2().utterances) == []


def test_retrieval_matches_on_word_stems():
    """"batch" must find "batching"."""
    lines = _retrieve("Batch appointment history", _week2().utterances)
    assert any("batching working" in line for line in lines)


# -- database discovery on disk -----------------------------------------------


def test_wal_and_shm_sidecars_are_not_databases(tmp_path):
    """SQLite's sidecars match a *.sqlite* glob and the -wal is the newest file.

    Sorting candidates by modification time therefore puts the write-ahead log
    first, and opening it fails with "file is not a database".
    """
    real = tmp_path / "meeting_minutes.sqlite"
    real.write_bytes((FIXTURES / "fake_meetily.db").read_bytes())
    (tmp_path / "meeting_minutes.sqlite-wal").write_bytes(b"\x00" * 512)
    (tmp_path / "meeting_minutes.sqlite-shm").write_bytes(b"\x00" * 512)

    assert is_sqlite(real)
    assert not is_sqlite(tmp_path / "meeting_minutes.sqlite-wal")
    assert not is_sqlite(tmp_path / "meeting_minutes.sqlite-shm")


def test_find_db_rejects_a_non_database(tmp_path):
    impostor = tmp_path / "notes.sqlite"
    impostor.write_text("this is not a database", encoding="utf-8")
    assert find_db(impostor) is None


def test_empty_schema_is_still_recognised(tmp_path):
    """A fresh Meetily install has the right tables and no rows in them."""
    db = tmp_path / "empty.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE transcripts (id TEXT, meeting_id TEXT, transcript TEXT, "
        "timestamp TEXT, audio_start_time REAL, audio_end_time REAL, speaker TEXT)"
    )
    conn.execute("CREATE TABLE settings (id TEXT, content TEXT)")
    conn.commit()
    conn.close()

    with MeetilyStore(db) as store:
        assert store.best is not None
        assert store.best.table == "transcripts"
        # audio_start_time must win over the wall-clock `timestamp` column.
        assert store.best.start == "audio_start_time"
        assert store.best.speaker == "speaker"


def test_webview_storage_is_excluded(tmp_path, monkeypatch):
    """Meetily embeds WebView2; its Chromium profile is full of SQLite files.

    Filename heuristics alone picked `declarative_performance_observer.db` out
    of `EBWebView/Default/` on a real installation.
    """
    appdata = tmp_path / "Roaming"
    real = appdata / "com.meetily.ai" / "meeting_minutes.sqlite"
    webview = appdata / "com.meetily.ai" / "EBWebView" / "Default" / "perf.db"
    for target in (real, webview):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((FIXTURES / "fake_meetily.db").read_bytes())
    # Make the decoy the most recently written file.
    os.utime(webview, (2_000_000_000, 2_000_000_000))

    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("LOCALAPPDATA", str(appdata))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert webview not in default_db_paths()
    assert find_db() == real.resolve()


# -- meetily recording folders ------------------------------------------------


def _make_recording(root: Path, name: str, segments: list[dict], status: str = "completed"):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "meeting_name": name,
                "created_at": "2026-09-13T18:06:16.228372800+00:00",
                "status": status,
                "audio_file": "audio.mp4",
                "meeting_id": None,
                "devices": {"microphone": "Mic", "system_audio": "Speakers"},
            }
        ),
        encoding="utf-8",
    )
    (folder / "transcripts.json").write_text(
        json.dumps({"segments": segments, "total_segments": len(segments), "version": "1.0"}),
        encoding="utf-8",
    )
    (folder / "audio.mp4").write_bytes(b"\x00" * 2048)
    return folder


def test_recording_folder_is_detected_and_loaded(tmp_path):
    folder = _make_recording(
        tmp_path,
        "Meeting 13_09_26",
        [
            {"start": 0.0, "end": 4.0, "text": "I finished the sync", "speaker": "Avish"},
            {"start": 4.5, "end": 8.0, "text": "Good work.", "speaker": "Dr. Chen"},
        ],
    )
    assert detect(folder) == "meetily-recording"
    t = load(folder)
    assert t.speakers == ["Avish", "Dr. Chen"]
    assert t.title == "Meeting 13_09_26"
    assert t.meeting_date == "2026-09-13"


def test_recordings_root_picks_the_newest(tmp_path):
    _make_recording(tmp_path, "older", [{"text": "old one", "speaker": "A"}])
    newer = _make_recording(tmp_path, "newer", [{"text": "new one", "speaker": "A"}])
    os.utime(newer, (2_000_000_000, 2_000_000_000))

    assert detect(tmp_path) == "meetily-recordings-root"
    assert "new one" in load(tmp_path).render()


def test_empty_recording_explains_itself(tmp_path):
    """A silent or too-short recording produces audio but no segments."""
    folder = _make_recording(tmp_path, "silent", [])
    with pytest.raises(ValueError, match="no transcript segments"):
        load(folder)


def test_describe_reports_recording_state(tmp_path):
    folder = _make_recording(tmp_path, "Meeting X", [{"text": "hello", "speaker": "A"}])
    info = meetily_folder.describe(folder)
    assert info["segments"] == 1
    assert info["status"] == "completed"
    assert info["saved_to_db"] is False
    assert info["audio_bytes"] == 2048


def test_plain_directory_is_rejected(tmp_path):
    (tmp_path / "notes").mkdir()
    with pytest.raises(IsADirectoryError, match="Meetily recording"):
        detect(tmp_path / "notes")
