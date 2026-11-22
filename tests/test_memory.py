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
    """One axis per character, so texts about the same thing point together."""
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


def test_ending_a_session_deletes_the_working_summary(monkeypatch, tmp_path):
    def fake_chat(prompt, **kwargs):
        assert kwargs["call_type"] == "summarise"
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
