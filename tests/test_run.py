"""The ablation switch and the Session A then Session B runner.

No test calls Ollama. The replies are fixed by stubbing the two functions
that would, so what is being checked is which store a configuration reads
and that Session B reads it from disk rather than from the Session A object.
"""

from __future__ import annotations

import json

import pytest

from src.agent import FALLBACK, Session
from src.nlu import _rule_based
from src.config import ABLATION_CONFIGS, ABLATION_USES
from src.memory import Memory
from src.run import run_config


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """Label by a rule stand-in and phrase a tool result as itself."""
    def understand(text, **kwargs):
        ruled = _rule_based(text)
        return ruled if ruled is not None else _label(text)

    def chat(prompt, **kwargs):
        return "Stub reply."

    monkeypatch.setattr("src.agent.understand", understand)
    monkeypatch.setattr("src.nlu.understand", understand)
    monkeypatch.setattr("src.agent.chat", chat)
    monkeypatch.setattr("src.memory.chat", chat)


def _label(text: str) -> dict:
    """Enough of an intent to drive the turns the tests use."""
    lowered = text.lower()
    slots: dict[str, str] = {}
    if "windows 11" in lowered:
        slots["os"] = "Windows 11"
    if "hyderabad" in lowered:
        slots["office"] = "Hyderabad"
    if lowered.startswith("what") or lowered.startswith("which") or lowered.startswith("who"):
        intent = "recall_fact"
    elif "err-vpn-809" in lowered and "what does" in lowered:
        intent = "ask_kb"
    elif "last time" in lowered or "what did i" in lowered:
        intent = "recall_fact"
    elif slots:
        intent = "provide_info"
    else:
        intent = "out_of_scope"
    return {"intent": intent, "source": "rule", "slots": slots}


def test_the_four_configurations_add_one_memory_type_at_a_time():
    assert ABLATION_CONFIGS == ("no_rag", "hybrid", "hybrid_sem", "hybrid_mem")
    assert ABLATION_USES["no_rag"] == {"kb": False, "semantic": False, "episodic": False}
    assert ABLATION_USES["hybrid"]["kb"] and not ABLATION_USES["hybrid"]["semantic"]
    assert ABLATION_USES["hybrid_sem"]["semantic"] and not ABLATION_USES["hybrid_sem"]["episodic"]
    assert all(ABLATION_USES["hybrid_mem"].values())


def test_a_configuration_name_is_rejected_when_unknown():
    with pytest.raises(ValueError):
        Session("U1", "A", config="dense")


def test_no_rag_does_not_search_the_knowledge_base(tmp_path, monkeypatch):
    searched = []
    monkeypatch.setattr("src.agent.search_kb", lambda *a, **k: searched.append(a) or [])

    session = _session(tmp_path, "no_rag")
    text = "What does ERR-VPN-809 mean?"
    session.ingest(text)
    decision = session.decide(text)

    assert decision["action"] == "none"
    assert searched == []
    assert session.reply(text, decision) == FALLBACK


def test_hybrid_searches_and_the_earlier_config_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.agent.search_kb",
        lambda *a, **k: [{"doc_id": "KB-VPN", "chunk_id": "c0", "text": "certificate check failed", "score": 0.9}],
    )
    session = _session(tmp_path, "hybrid")
    # The last turn is planted as a knowledge-base question. What is being
    # checked is the gate in decide(), not the labelling.
    session.history.append({"turn": 1, "text": "What does ERR-VPN-809 mean?", "intent": "ask_kb", "bot": ""})
    session.turn = 1
    decision = session.decide("What does ERR-VPN-809 mean?")
    assert decision["action"] == "tool"
    assert decision["name"] == "search_kb"


def test_facts_are_stored_under_no_rag_but_only_read_from_hybrid_sem(tmp_path):
    """Writing is not gated. The read is."""
    text = "My laptop runs Windows 11."
    no_rag = _session(tmp_path, "no_rag")
    no_rag.ingest(text)
    question = "What operating system did I say my laptop runs?"
    no_rag.ingest(question)
    assert no_rag.decide(question)["action"] == "none"

    # A new Session, same folder. hybrid still does not read facts.
    hybrid = _session(tmp_path, "hybrid", user="U1")
    hybrid.ingest(question)
    assert hybrid.decide(question)["action"] == "none"

    # hybrid_sem is the first config that reads the fact store.
    sem = _session(tmp_path, "hybrid_sem")
    sem.ingest(question)
    decision = sem.decide(question)
    assert decision["action"] == "tool"
    assert decision["name"] == "recall_fact"
    assert decision["result"]["value"] == "Windows 11"


def test_episodes_are_read_only_by_hybrid_mem(tmp_path, monkeypatch):
    # The stub embedder scores nothing above the cut-off, so the episode
    # would be dropped for every config. This checks the gate, not the rank.
    monkeypatch.setattr(
        "src.episodic_memory.EpisodicMemory.recall",
        lambda self, user, query, **kwargs: [{"id": "ep-0001", "text": "Booked a visit on Thursday at 4 pm.", "type": "booking"}],
    )
    question = "What did I book last time?"

    sem = _session(tmp_path, "hybrid_sem")
    sem.ingest(question)
    assert sem.decide(question)["action"] == "none"

    mem = _session(tmp_path, "hybrid_mem")
    mem.ingest(question)
    decision = mem.decide(question)
    assert decision["action"] == "tool"
    assert decision["name"] == "recall_episode"
    assert "Thursday" in decision["result"]["text"]


def test_session_b_reads_what_session_a_wrote_to_disk(tmp_path, monkeypatch):
    """The runner closes A before it builds B, so B has nothing in memory."""
    monkeypatch.setattr("src.run.load_turns", _script)
    monkeypatch.setattr("src.run.load_probes", lambda session: [])
    monkeypatch.setattr("src.run.RUNS_ROOT", tmp_path)

    transcript = run_config("hybrid_mem")

    assert [s["session"] for s in transcript["sessions"]] == ["A", "B"]
    # B answered from the store, and the reply names the fact.
    answer = transcript["sessions"][1]["turns"][0]
    assert answer["user"] == "U1"
    assert "Windows 11" in answer["bot"]
    # The fact is on disk, which is the only place B could have read it.
    stored = (tmp_path / "hybrid_mem" / "memory" / "semantic.jsonl").read_text(encoding="utf-8")
    assert "Windows 11" in stored


def test_a_second_user_does_not_read_the_first_users_fact(tmp_path, monkeypatch):
    monkeypatch.setattr("src.run.load_turns", _two_users)
    monkeypatch.setattr("src.run.load_probes", lambda session: [])
    monkeypatch.setattr("src.run.RUNS_ROOT", tmp_path)

    transcript = run_config("hybrid_mem", sessions=("B",))
    turns = transcript["sessions"][0]["turns"]
    # U1 stated the fact and was told so. U2 asked and was not told it.
    assert turns[0]["user"] == "U1"
    assert turns[1]["user"] == "U2"
    assert "Windows 11" not in turns[1]["bot"]
    stored = (tmp_path / "hybrid_mem" / "memory" / "semantic.jsonl").read_text(encoding="utf-8")
    assert stored.count('"user": "U1"') >= 1
    assert '"user": "U2"' not in stored


def test_the_run_writes_a_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr("src.run.load_turns", _script)
    monkeypatch.setattr("src.run.load_probes", lambda session: [])
    monkeypatch.setattr("src.run.RUNS_ROOT", tmp_path)

    run_config("no_rag", sessions=("A",))
    written = json.loads((tmp_path / "no_rag" / "transcript.json").read_text(encoding="utf-8"))
    assert written["config"] == "no_rag"
    assert written["sessions"][0]["session"] == "A"


def _session(tmp_path, config: str, user: str = "U1") -> Session:
    return Session(
        user,
        "A",
        db_path=tmp_path / "helpdesk.sqlite",
        memory=Memory(user, "A", root=tmp_path / "memory"),
        config=config,
    )


def _script(session: str) -> list[dict]:
    if session == "A":
        return [{"turn": 1, "session": "A", "user": "U1", "text": "My laptop runs Windows 11."}]
    return [{"turn": 2, "session": "B", "user": "U1", "text": "What operating system did I say my laptop runs?"}]


def _two_users(session: str) -> list[dict]:
    return [
        {"turn": 1, "session": "B", "user": "U1", "text": "My laptop runs Windows 11."},
        {"turn": 2, "session": "B", "user": "U2", "text": "What operating system did I say my laptop runs?"},
    ]
