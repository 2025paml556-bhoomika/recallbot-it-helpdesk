"""Semantic, episodic and working memory.

No test calls Ollama or the embedding model. embed() is replaced by a
deterministic stand-in, and every store writes under pytest's tmp_path, so
the shared memory/ folder is never touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import EMBED_DIM, EPISODIC_DECAY
from src.episodic_memory import EpisodicMemory, score
from src.history import WorkingMemory, select
from src.semantic_memory import SemanticMemory


def _fake_embed(texts, **kwargs):
    """One axis per character, so texts about the same thing point together.

    log=False is a rebuild from vectors already stored, and writes nothing.
    Anything else writes the embed line the real embedder would, so a test
    can check that a new fact or a query was logged.
    """
    if kwargs.get("log", True):
        import time

        from src.logger import log_call

        log_call(
            call_type="embed",
            model="test-embed",
            input_tokens=len(texts),
            output_tokens=0,
            start=time.time(),
            config=kwargs.get("config"),
            user=kwargs.get("user"),
            session=kwargs.get("session"),
            turn=kwargs.get("turn"),
            run_id=kwargs.get("run_id"),
            extra={"n_texts": len(texts)},
        )
    width = EMBED_DIM
    rows = []
    for text in texts:
        vector = np.zeros(width, dtype=np.float32)
        for character in text.lower():
            vector[ord(character) % width] += 1.0
        norm = np.linalg.norm(vector)
        rows.append(vector / norm if norm else vector)
    return np.vstack(rows)


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)


def test_extracting_a_fact_is_logged_and_small_talk_is_not(tmp_path, monkeypatch):
    """call_type "extract" is written once per extraction, never for small talk.

    Storing the fact also logs its embedding. Recalling it logs the query
    embedding. Rebuilding the index from stored vectors logs nothing more.
    """
    import json

    import src.logger as logger

    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(logger, "LOG_PATH", path)

    store = SemanticMemory(tmp_path / "memory")
    written = store.extract(
        "U1",
        "I sit at the Pune office.",
        {"location": "Pune office"},
        turn=2,
        session="A",
        run_id="abl-hybrid_sem",
        config="hybrid_sem",
    )
    assert written
    assert store.extract("U1", "Thanks!", {}, turn=3, session="A", config="hybrid_sem") == []
    # The same fact is already active, so it is not embedded again. The
    # extraction itself is still logged: the turn was examined.
    again = store.extract(
        "U1",
        "I sit at the Pune office.",
        {"location": "Pune office"},
        turn=4,
        session="A",
        config="hybrid_sem",
    )
    assert again[0]["id"] == written[0]["id"]
    found = store.recall(
        "U1", "Where do I sit?",
        session="B", turn=1, run_id="abl-hybrid_sem", config="hybrid_sem",
    )
    assert found

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["call_type"] for row in lines] == ["embed", "extract", "extract", "embed"]
    stored, extracted, repeat, query = lines
    assert repeat["call_type"] == "extract" and repeat["turn"] == 4
    assert repeat["keys"] == ["location"]
    assert stored["user"] == "U1" and stored["session"] == "A" and stored["turn"] == 2
    assert stored["run_id"] == "abl-hybrid_sem" and stored["config"] == "hybrid_sem"
    assert stored["retrieved_tokens"] == 0
    assert query["session"] == "B" and query["turn"] == 1
    assert query["run_id"] == "abl-hybrid_sem"
    assert extracted["call_type"] == "extract"
    assert extracted["model"] == "rule"
    assert extracted["input_tokens"] == 0
    assert extracted["output_tokens"] == 0
    assert extracted["retrieved_tokens"] == 0
    assert extracted["user"] == "U1"
    assert extracted["session"] == "A"
    assert extracted["turn"] == 2
    assert extracted["config"] == "hybrid_sem"
    assert "location" in extracted["keys"]


def test_a_stated_fact_is_stored_and_small_talk_is_not(tmp_path):
    store = SemanticMemory(tmp_path)
    written = store.extract(
        "U1",
        "I sit at the Pune office and I use LAP-T14G3.",
        {"location": "Pune", "asset_id": "LAP-T14G3"},
        turn=3,
    )
    assert {row["key"] for row in written} == {"location", "asset_id"}
    assert all(row["user"] == "U1" and row["status"] == "active" for row in written)
    assert all(row["source_turn"] == 3 for row in written)

    assert store.extract("U1", "Thanks, have a good day!", {}, turn=4) == []
    assert len(store.active("U1")) == 2


def test_moving_office_supersedes_the_old_one(tmp_path):
    from src.memory import Memory

    memory = Memory("U1", "A", root=tmp_path)
    memory.record_turn(
        "I sit at the Pune office.",
        "provide_info",
        {"location": "Pune office"},
        turn=1,
    )
    written = memory.record_turn(
        "I have moved from Pune to Hyderabad.",
        "provide_info",
        {"location": "Hyderabad"},
        turn=2,
    )
    assert [row["value"] for row in written] == ["Hyderabad"]


def test_a_visit_does_not_store_its_place_as_where_the_user_sits(tmp_path):
    from src.memory import Memory

    memory = Memory("U1", "A", root=tmp_path)
    written = memory.record_turn(
        "Can you come to my desk?",
        "book_slot",
        {"location": "desk 14"},
        turn=4,
    )
    assert written == []
    assert memory.semantic.active("U1") == []

    # Saying where they sit is still a standing fact.
    memory.record_turn(
        "I sit at the Pune office.",
        "provide_info",
        {"location": "Pune office"},
        turn=5,
    )
    assert [row["value"] for row in memory.semantic.active("U1")] == ["Pune office"]


def test_an_update_leaves_one_active_and_one_superseded(tmp_path):
    store = SemanticMemory(tmp_path)
    store.remember("U1", "location", "Pune", turn=2)
    store.remember("U1", "location", "Hyderabad", turn=9)

    active = store.active("U1")
    assert len(active) == 1
    assert active[0]["value"] == "Hyderabad"
    assert active[0]["status"] == "active"

    rows = [row for row in store._read() if row["key"] == "location"]
    # One line per value. The old value is superseded; the new one is active.
    assert [(row["value"], row["status"]) for row in rows] == [
        ("Pune", "superseded"),
        ("Hyderabad", "active"),
    ]


def test_a_forget_removes_the_fact_from_the_file_and_the_index(tmp_path):
    store = SemanticMemory(tmp_path)
    store.remember("U1", "location", "Pune", turn=2)
    store.remember("U1", "asset_id", "LAP-T14G3", turn=2)
    assert store.vector_count() == 2

    removed = store.forget("U1", "location")
    assert removed == 1

    # A fresh object is a fresh process: it reads only what is on disk.
    reloaded = SemanticMemory(tmp_path)
    assert reloaded.active("U1")[0]["key"] == "asset_id"
    assert reloaded.vector_count() == len(reloaded.active_records())
    assert "Pune" not in reloaded.path.read_text(encoding="utf-8")


def test_a_second_user_never_receives_the_first_users_facts(tmp_path):
    store = SemanticMemory(tmp_path)
    store.remember("U1", "location", "Pune", turn=2)
    store.remember("U2", "location", "Bengaluru", turn=1)

    recalled = store.recall("U2", "which office am I in")
    assert recalled
    assert {row["user"] for row in recalled} == {"U2"}
    assert recalled[0]["value"] == "Bengaluru"

    store.remember("U2", "location", "Mumbai", turn=5)
    assert store.active("U1")[0]["value"] == "Pune"


def test_a_new_episode_and_a_query_are_logged_and_a_rebuild_is_not(tmp_path, monkeypatch):
    """Embedding a new episode and a recall query writes an embed line each.

    The index rebuild copies the stored vector, so it adds no third line.
    """
    import json

    import src.logger as logger

    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(logger, "LOG_PATH", path)

    store = EpisodicMemory(tmp_path / "episodes")
    written = store.add(
        "U1",
        "Raised TCK-1042 for the laptop screen.",
        "ticket",
        session="A",
        turn=9,
        run_id="abl-hybrid_mem",
        config="hybrid_mem",
    )
    assert written is not None
    store._reindex(store._read())
    found = store.recall(
        "U1",
        "what ticket did I raise",
        session="B",
        turn=2,
        run_id="abl-hybrid_mem",
        config="hybrid_mem",
    )
    assert found

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["call_type"] for row in lines] == ["embed", "embed"]
    stored, query = lines
    assert stored["user"] == "U1" and stored["session"] == "A" and stored["turn"] == 9
    assert stored["run_id"] == "abl-hybrid_mem" and stored["config"] == "hybrid_mem"
    assert stored["retrieved_tokens"] == 0
    assert query["session"] == "B" and query["turn"] == 2
    assert query["run_id"] == "abl-hybrid_mem"


def test_score_matches_the_worked_example():
    value = score(10, 8, 0.6)
    assert value == pytest.approx(EPISODIC_DECAY**10 + 0.8 + 0.6)
    assert value == pytest.approx(2.351, abs=0.001)


def test_a_newer_episode_outranks_an_older_one_with_the_same_text(tmp_path):
    from datetime import datetime, timedelta, timezone

    store = EpisodicMemory(tmp_path)
    now = datetime.now(timezone.utc)
    store.add("U1", "Raised TCK-1042 for the laptop screen.", "ticket", when=now - timedelta(hours=10))
    store.add("U1", "Raised TCK-1042 for the laptop screen.", "ticket", when=now - timedelta(hours=1))

    ranked = store.recall("U1", "what ticket did I raise", now=now)
    assert len(ranked) == 2
    assert ranked[0]["age_hours"] < ranked[1]["age_hours"]


def test_small_talk_is_not_stored_as_an_episode(tmp_path):
    store = EpisodicMemory(tmp_path)
    assert store.add("U1", "Thanks!", "note") is None
    assert store.for_user("U1") == []


def test_an_episode_is_recalled_only_for_its_owner(tmp_path):
    store = EpisodicMemory(tmp_path)
    store.add("U1", "Raised TCK-1042 for the broken laptop screen.", "ticket", session="A")
    store.add("U2", "Reported the Wi-Fi dropping in Bengaluru.", "event", session="B")

    mine = store.recall("U2", "what did I report about the wifi")
    assert mine
    assert {row["user"] for row in mine} == {"U2"}
    assert "Wi-Fi" in mine[0]["text"]

    assert store.recall("U2", "what laptop did the other person raise") == [] or all(
        row["user"] == "U2" for row in store.recall("U2", "laptop")
    )


def test_a_forget_removes_the_episode_and_a_restart_cannot_find_it(tmp_path):
    store = EpisodicMemory(tmp_path)
    store.add("U1", "My personal phone number is 98xxxxxx12.", "note")
    store.reflect("U1", "A", "U1 asked to be reached on 98xxxxxx12.")
    assert store.forget("U1", "98xxxxxx12") == 2

    reloaded = EpisodicMemory(tmp_path)
    assert reloaded.for_user("U1") == []
    assert "98xxxxxx12" not in reloaded.path.read_text(encoding="utf-8")
    assert reloaded.vector_count() == 0


def test_a_decision_scores_nine_and_a_complaint_scores_six(tmp_path):
    """The refresh decision outranks the VPN complaint on importance alone.

    Recency-heavy ranking puts the complaint from an hour ago first.
    Importance-heavy ranking puts the decision first, even though it is old.
    """
    from datetime import datetime, timedelta, timezone

    store = EpisodicMemory(tmp_path)
    now = datetime.now(timezone.utc)
    complaint = store.add(
        "U1",
        "The VPN keeps failing with ERR-VPN-809.",
        "event",
        when=now - timedelta(hours=1),
    )
    decision = store.add(
        "U1",
        "I'd like to go ahead with the refresh rather than another repair.",
        "event",
        when=now - timedelta(hours=400),
    )
    assert complaint["importance"] == 6
    assert decision["importance"] == 9

    by_recency = store.recall("U1", "what happened", now=now, alpha=1.0, beta=0.0, gamma=0.0)
    by_importance = store.recall("U1", "what happened", now=now, alpha=0.0, beta=1.0, gamma=0.0)
    assert by_recency[0]["importance"] == 6
    assert by_importance[0]["importance"] == 9


def test_two_weightings_rank_a_probe_differently(tmp_path):
    from datetime import datetime, timedelta, timezone

    store = EpisodicMemory(tmp_path)
    now = datetime.now(timezone.utc)
    store.add(
        "U1",
        "Complained that the printer on the third floor jams.",
        "event",
        importance=4,
        when=now - timedelta(hours=1),
    )
    store.add(
        "U1",
        "Decided to upgrade the laptop next quarter.",
        "event",
        importance=9,
        when=now - timedelta(hours=400),
    )

    # Recency-heavy: the complaint is from an hour ago, so it leads.
    by_recency = store.recall("U1", "what happened", now=now, alpha=1.0, beta=0.0, gamma=0.0)
    # Importance-heavy: the upgrade decision outranks the recent complaint.
    by_importance = store.recall("U1", "what happened", now=now, alpha=0.0, beta=1.0, gamma=0.0)

    assert by_recency[0]["text"].startswith("Complained")
    assert by_importance[0]["text"].startswith("Decided")
    assert [row["id"] for row in by_recency] != [row["id"] for row in by_importance]


def test_trim_keeps_the_last_four_turns_and_full_keeps_all():
    history = [{"user": f"turn {n}", "bot": "ok"} for n in range(10)]
    assert len(select(history, "full")) == 10
    trimmed = select(history, "trim")
    assert len(trimmed) == 4
    assert trimmed[0]["user"] == "turn 6"
    assert len(select(history, "summary")) == 2


def test_asking_where_i_sit_does_not_write_a_fact(tmp_path):
    from src.memory import Memory

    memory = Memory("U1", "B", root=tmp_path)
    memory.semantic.remember("U1", "location", "Pune office", turn=1)
    written = memory.record_turn("Where do I sit?", "recall_fact", {}, turn=2)
    assert written == []
    assert memory.semantic.active("U1")[0]["value"] == "Pune office"


def test_ending_a_session_writes_a_reflection_the_next_session_reads(monkeypatch, tmp_path):
    """No reflection is typed in. The model writes it, and it is logged.

    A second Memory is a second process: it can only find the reflection
    because the first one wrote it to the file.
    """
    import json
    import time

    from src.config import LOG_PATH
    from src.logger import log_call
    from src.memory import Memory

    def fake_chat(prompt, **kwargs):
        if kwargs["call_type"] == "summary":
            return "U1 has a VPN fault and wants a Thursday visit."
        assert kwargs["call_type"] == "reflect"
        assert "Thursday" in prompt
        # The real chat() writes the reflect line. A stub has to as well,
        # or the log would not show the step the brief asks for.
        log_call(
            call_type="reflect",
            model="stub",
            input_tokens=0,
            output_tokens=0,
            start=time.time(),
            config=kwargs.get("config"),
            user=kwargs.get("user"),
            session=kwargs.get("session"),
            turn=kwargs.get("turn"),
        )
        return "U1 booked a Thursday visit for the VPN fault and chose a refresh."

    monkeypatch.setattr("src.history.chat", fake_chat)
    monkeypatch.setattr("src.memory.chat", fake_chat)
    before = len(LOG_PATH.read_text(encoding="utf-8").splitlines()) if LOG_PATH.exists() else 0

    memory = Memory("U1", "A", root=tmp_path)
    memory.working.update([{"user": "Book Thursday.", "bot": "Booked."}], turn=16)
    written = memory.end_session(
        None,
        turns=[{"user": "Book the Thursday visit.", "bot": "Booked Thursday at 4 pm."}],
        turn=25,
        config="full",
    )
    assert written["type"] == "reflection"
    assert "Thursday" in written["text"]
    assert written["importance"] == 5
    assert not memory.working.path.exists()

    added = [
        json.loads(line)
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines()[before:]
    ]
    assert any(row["call_type"] == "reflect" for row in added)
    assert any(
        row["call_type"] == "importance" and row.get("importance") == 5
        and row.get("episode_type") == "reflection"
        for row in added
    )

    later = Memory("U1", "B", root=tmp_path)
    hits = later.episodic.recall("U1", "what did I book last time")
    assert hits
    assert hits[0]["type"] == "reflection"
    assert "Thursday" in hits[0]["text"]


def test_ending_a_session_deletes_the_working_summary(monkeypatch, tmp_path):
    def fake_chat(prompt, **kwargs):
        assert kwargs["call_type"] == "summary"
        return "U1 has a VPN fault and wants a Thursday visit."

    monkeypatch.setattr("src.history.chat", fake_chat)
    notes = WorkingMemory("U1", "A", root=tmp_path)
    notes.update([{"user": "The VPN is down.", "bot": "Which day works?"}], turn=2)

    assert notes.path.exists()
    assert "VPN" in notes.summary

    text = notes.end()
    assert "VPN" in text
    assert not notes.path.exists()
    assert notes.summary == ""

def test_each_session_is_its_own_conversation_file(tmp_path):
    from src.memory import ConversationLog, _session_filename

    first = ConversationLog("U1", "S-1", root=tmp_path)
    first.append(
        "I sit at the Pune office.",
        "Thanks, I'll keep that in mind.",
        turn=1,
        intent="provide_info",
        calls=[{
            "call_type": "nlu",
            "model": "qwen2.5:3b",
            "messages": [
                {"role": "system", "content": "label the message"},
                {"role": "user", "content": "I sit at the Pune office."},
                {"role": "assistant", "content": '{"intent": "provide_info"}'},
            ],
        }],
    )
    second = ConversationLog("U1", "S-2", root=tmp_path)
    second.append("Where do I sit?", "You sit at Pune office.", turn=1, intent="recall_fact")

    assert first.path != second.path
    assert first.path.name == _session_filename("U1", "S-1")
    assert second.path.name == _session_filename("U1", "S-2")

    # A new object is a new process: it reads the file the first one wrote.
    reloaded = ConversationLog("U1", "S-1", root=tmp_path)
    assert reloaded.turns[0]["turn"] == 1
    assert reloaded.turns[0]["user"] == "I sit at the Pune office."
    assert reloaded.turns[0]["bot"] == "Thanks, I'll keep that in mind."
    assert reloaded.turns[0]["intent"] == "provide_info"
    assert reloaded.turns[0]["calls"][0]["messages"][0]["role"] == "system"
    assert reloaded.turns[0]["calls"][0]["messages"][2]["role"] == "assistant"
    # The second session's question is not in the first session's file.
    assert "Where do I sit?" not in first.path.read_text(encoding="utf-8")


def test_a_session_id_with_odd_characters_still_makes_one_file(tmp_path):
    from src.memory import ConversationLog

    log = ConversationLog("U1", "S 1/tab", root=tmp_path)
    log.append("Hi.", "Hello.", turn=1, intent="out_of_scope")
    assert log.path.parent == tmp_path
    assert log.path.exists()
    assert "/" not in log.path.name


def test_clear_user_removes_only_that_users_memory(tmp_path):
    from src.memory import Memory

    first = Memory("U1", "S-1", root=tmp_path)
    first.semantic.remember("U1", "location", "Pune office", 1)
    first.episodic.add("U1", "Raised TCK-0001 for VPN.", "ticket", session="S-1")
    first.conversation.append("I sit at Pune.", "Noted.", turn=1, intent="provide_info")
    other_session = Memory("U1", "S-2", root=tmp_path)
    other_session.conversation.append("Where do I sit?", "Pune.", turn=1, intent="recall_fact")

    second = Memory("U2", "S-1", root=tmp_path)
    second.semantic.remember("U2", "location", "Hyderabad", 1)
    second.episodic.add("U2", "Raised TCK-0002 for Wi-Fi.", "ticket", session="S-1")
    second.conversation.append("I sit at Hyderabad.", "Noted.", turn=1, intent="provide_info")

    removed = first.clear_user()
    assert removed["facts"] == 1
    assert removed["episodes"] == 1
    assert removed["sessions"] == 2

    assert first.semantic.active("U1") == []
    assert first.episodic.for_user("U1") == []
    assert list((tmp_path / "sessions").glob("U1__*.json")) == []

    assert second.semantic.active("U2")[0]["value"] == "Hyderabad"
    assert second.episodic.for_user("U2")[0]["text"] == "Raised TCK-0002 for Wi-Fi."
    assert (tmp_path / "sessions" / "U2__S-1.json").exists()


def test_location_wording_is_searched_as_an_office(tmp_path):
    store = SemanticMemory(tmp_path)
    store.remember("U1", "location", "Hyderabad office", turn=1)
    for question in (
        "where is my location",
        "Where do I sit?",
        "what is my office",
        "where am I based",
        "Where am I located",
    ):
        hits = store.recall("U1", question)
        assert hits, question
        assert hits[0]["key"] == "location"
        assert hits[0]["value"] == "Hyderabad office"


def test_a_colour_and_a_name_are_stored_as_notes(tmp_path):
    """A sentence with no office and no laptop id is still a fact.

    It is one note, not forced into asset_id. A badge number the model
    parked in asset_id is rejected there and kept as the sentence instead,
    so it cannot replace the laptop id.
    """
    from src.memory import Memory

    memory = Memory("U1", "A", root=tmp_path)
    colour = memory.record_turn(
        "My laptop bag is blue.",
        "provide_info",
        {},
        turn=1,
    )
    assert [row["key"] for row in colour] == ["note"]
    assert colour[0]["value"] == "My laptop bag is blue."

    badge = memory.record_turn(
        "My badge number is 4412.",
        "provide_info",
        {"asset_id": "4412"},
        turn=2,
    )
    assert [row["key"] for row in badge] == ["note"]

    laptop = memory.record_turn(
        "My laptop is LAP-T14G3.",
        "provide_info",
        {"asset_id": "LAP-T14G3"},
        turn=3,
    )
    assert [row["key"] for row in laptop] == ["asset_id"]

    active = memory.semantic.active("U1")
    assert {row["key"] for row in active} == {"note", "asset_id"}
    assert [row["value"] for row in active if row["key"] == "asset_id"] == ["LAP-T14G3"]
    notes = [row["value"] for row in active if row["key"] == "note"]
    assert "My laptop bag is blue." in notes
    assert "My badge number is 4412." in notes

    # A second copy of the same sentence is not a second note.
    again = memory.record_turn("My laptop bag is blue.", "provide_info", {}, turn=4)
    assert again[0]["id"] == colour[0]["id"]
    assert len([row for row in memory.semantic.active("U1") if row["key"] == "note"]) == 2


def test_the_nine_profile_keys_are_extracted_from_the_scripted_turns(tmp_path):
    """Turns 1 to 7 state the profile. Each lands under its own key.

    The NLU slots are empty here, which is what the model returns for these
    sentences. The facts still have to be stored, because they are read off
    the sentence rather than off a slot.
    """
    import json

    from src.config import ROOT
    from src.memory import Memory
    from src.semantic_memory import PROFILE_KEYS

    turns = json.loads((ROOT / "data" / "test_conversation.json").read_text())["turns"]
    memory = Memory("U1", "A", root=tmp_path)
    for turn in turns:
        if turn["session"] != "A" or turn["turn"] > 7:
            continue
        memory.record_turn(turn["text"], "provide_info", {}, turn["turn"])

    active = memory.semantic.active("U1")
    stored = {row["key"]: row["value"] for row in active}
    expected = {}
    for turn in turns:
        if turn["session"] == "A" and turn["turn"] <= 7:
            for fact in turn.get("facts", []):
                expected[fact["key"]] = fact["value"]
    assert set(expected) == set(PROFILE_KEYS)
    assert stored == expected
    # A profile sentence is the fact, not also a free-text note.
    assert "note" not in stored


def test_a_profile_update_supersedes_the_old_value(tmp_path):
    from src.memory import Memory

    memory = Memory("U1", "A", root=tmp_path)
    memory.record_turn(
        "I work in the finance department, and my manager is Rohan Deshpande.",
        "provide_info",
        {},
        turn=4,
    )
    written = memory.record_turn(
        "And my manager changed too. It's Priya Nair now, not Rohan Deshpande.",
        "provide_info",
        {},
        turn=21,
    )
    assert [row["value"] for row in written] == ["Priya Nair"]
    managers = [row for row in memory.semantic.active("U1") if row["key"] == "manager"]
    assert [row["value"] for row in managers] == ["Priya Nair"]
    rows = [row for row in memory.semantic._read() if row["key"] == "manager"]
    assert [(row["value"], row["status"]) for row in rows] == [
        ("Rohan Deshpande", "superseded"),
        ("Priya Nair", "active"),
    ]


def test_a_move_replaces_the_office_and_keeps_the_floor(tmp_path):
    from src.memory import Memory

    memory = Memory("U1", "A", root=tmp_path)
    memory.record_turn(
        "I sit at the Pune office, on the third floor.",
        "provide_info",
        {},
        turn=3,
    )
    memory.record_turn(
        "One more thing: I've moved. I'm at the Hyderabad office now, not Pune.",
        "provide_info",
        {},
        turn=20,
    )
    active = {row["key"]: row["value"] for row in memory.semantic.active("U1")}
    assert active["office"] == "Hyderabad"
    assert active["floor"] == "3rd floor"


def test_who_is_my_manager_recalls_the_manager_and_not_the_office(tmp_path):
    store = SemanticMemory(tmp_path)
    store.remember("U1", "office", "Hyderabad", turn=20)
    store.remember("U1", "manager", "Priya Nair", turn=21)
    store.remember("U1", "department", "finance", turn=4)
    hits = store.recall("U1", "Who is my manager now?")
    assert hits
    assert hits[0]["key"] == "manager"
    assert hits[0]["value"] == "Priya Nair"


def test_turn_22_forgets_the_phone_and_a_restart_cannot_read_it(tmp_path):
    """The scripted forget names the key and the number.

    Both have to leave the file. The office stated on an earlier turn stays,
    and a store opened afterwards, as a new session would, finds no phone.
    """
    from src.memory import Memory

    memory = Memory("U1", "A", root=tmp_path)
    memory.record_turn(
        "I sit at the Pune office, on the third floor.",
        "provide_info",
        {},
        turn=3,
    )
    memory.record_turn(
        "If the desk has to reach me, my personal number is 98220 11435.",
        "provide_info",
        {},
        turn=7,
    )
    memory.episodic.add("U1", "Meera can be reached on 98220 11435.", "note", session="A")
    assert any(row["key"] == "personal_phone" for row in memory.semantic.active("U1"))

    removed = memory.forget_request("personal_phone", "98220 11435")
    assert removed["facts"] >= 1
    assert removed["episodes"] == 1

    reloaded = Memory("U1", "B", root=tmp_path)
    active = {row["key"]: row["value"] for row in reloaded.semantic.active("U1")}
    assert "personal_phone" not in active
    assert active["office"] == "Pune"
    text = reloaded.semantic.path.read_text(encoding="utf-8")
    assert "98220 11435" not in text
    assert "98220 11435" not in reloaded.episodic.path.read_text(encoding="utf-8")
    assert reloaded.semantic.recall("U1", "What number did I give you for my personal phone?") == []


def test_a_booking_sentence_does_not_overwrite_the_profile_office(tmp_path):
    """"The Pune office." while booking a visit is not where the user sits."""
    from src.memory import Memory

    memory = Memory("U1", "A", root=tmp_path)
    memory.record_turn(
        "I sit at the Pune office, on the third floor.",
        "provide_info",
        {},
        turn=3,
    )
    written = memory.record_turn("The Pune office.", "book_slot", {"location": "Pune office"}, turn=12)
    assert written == []
    active = {row["key"]: row["value"] for row in memory.semantic.active("U1")}
    assert active == {"office": "Pune", "floor": "3rd floor"}


def test_a_fault_and_a_visit_are_not_stored_as_notes(tmp_path):
    from src.memory import Memory

    memory = Memory("U1", "A", root=tmp_path)
    assert memory.record_turn(
        "The printer on my floor jams every morning.",
        "fix_issue",
        {"location": "floor"},
        turn=1,
    ) == []
    assert memory.record_turn("Monday", "book_slot", {"date": "Monday"}, turn=2) == []
    assert memory.semantic.active("U1") == []


def test_a_colour_question_finds_the_note_not_the_laptop(tmp_path):
    store = SemanticMemory(tmp_path)
    store.remember("U1", "asset_id", "LAP-T14G3", turn=1)
    store.remember("U1", "note", "My laptop bag is blue.", turn=2)
    store.remember("U1", "note", "My badge number is 4412.", turn=3)
    hits = store.recall("U1", "What colour is my laptop bag?")
    assert hits
    assert hits[0]["key"] == "note"
    assert hits[0]["value"] == "My laptop bag is blue."
    # The laptop id is still found by a laptop question.
    laptop = store.recall("U1", "Whats my asset Id ?")
    assert laptop[0]["key"] == "asset_id"
    assert laptop[0]["value"] == "LAP-T14G3"


def test_asset_wording_is_searched_as_a_laptop(tmp_path):
    store = SemanticMemory(tmp_path)
    store.remember("U1", "asset_id", "LAP-T14G3", turn=1)
    for question in (
        "which laptop do I use",
        "Whats my asset Id ?",
        "what is my asset",
        "what device do I have",
    ):
        hits = store.recall("U1", question)
        assert hits, question
        assert hits[0]["key"] == "asset_id"
        assert hits[0]["value"] == "LAP-T14G3"


def test_the_window_quotes_only_completed_turns():
    from src.history import earlier_turns, window_text

    history = [
        {"text": "I sit at the Pune office.", "bot": "Noted."},
        {"text": "Where do I sit?", "bot": "Pune."},
        {"text": "And the floor?", "intent": "recall_fact"},
    ]
    done = earlier_turns(history)
    assert [row["user"] for row in done] == ["I sit at the Pune office.", "Where do I sit?"]
    assert "And the floor?" not in window_text(history, "full")
    assert "Pune office" in window_text(history, "full")
    # summary with no notes yet still quotes the last two completed turns.
    shown = window_text(history, "summary", summary="")
    assert "Earlier turns:" in shown
    assert "Notes so far:" not in shown
    noted = window_text(history, "summary", summary="U1 sits in Pune.")
    assert noted.startswith("Notes so far:")
    assert "U1 sits in Pune." in noted


def test_a_summary_folds_in_only_the_turns_it_has_not_covered(monkeypatch, tmp_path):
    """Turns 1 and 2 are summarised once. Turn 3 does not send them again."""
    from src.history import WorkingMemory

    sent = []

    def fake_chat(prompt, **kwargs):
        sent.append(prompt)
        return "notes: " + prompt

    monkeypatch.setattr("src.history.chat", fake_chat)
    notes = WorkingMemory("U1", "A", root=tmp_path)
    notes.update([{"user": "turn one", "bot": "ok"}], turn=3)
    assert notes.covered == 1
    notes.update([{"user": "turn two", "bot": "ok"}], turn=4)
    assert notes.covered == 2
    assert "turn one" in sent[0]
    # The second call is handed the notes so far, then only the new turn.
    assert sent[1].rstrip().endswith("User: turn two\nBot: ok")
    # The notes so far repeat the first call, so "New turns" appears twice.
    # The new-turn block itself names only turn two.
    head, _, fresh = sent[1].rpartition("New turns:")
    assert "turn two" in fresh
    assert "turn one" not in fresh

    reloaded = WorkingMemory("U1", "A", root=tmp_path)
    assert reloaded.covered == 2
    assert "turn two" in reloaded.summary
