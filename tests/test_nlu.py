"""NLU checks: rules fire without a model call, the LLM path returns JSON.

The live tests talk to the local Ollama model and are marked `live`.
The accuracy figure over all 20+ labels is `python -m src.eval_nlu`, which
is too slow to run on every pytest invocation.
"""

from __future__ import annotations

import json

import pytest

from src.config import LOG_PATH
from src.eval_nlu import load_labels, slots_match
from src.nlu import INTENTS, SLOT_NAMES, NLUError, understand


def _log_len() -> int:
    if not LOG_PATH.exists():
        return 0
    return len(LOG_PATH.read_text(encoding="utf-8").splitlines())


def test_labelled_file_has_20_plus_messages_covering_every_intent():
    labels = load_labels()
    assert len(labels) >= 20
    seen = {item["intent"] for item in labels}
    assert seen == set(INTENTS)
    for item in labels:
        assert item["text"].strip()
        assert set(item["slots"]) <= set(SLOT_NAMES)


def test_rules_label_yes_no_and_a_bare_ticket_without_logging():
    before = _log_len()

    yes = understand("yes")
    assert yes["intent"] == "confirm"
    assert yes["source"] == "rule"
    assert all(value is None for value in yes["slots"].values())

    no = understand("no")
    assert no["intent"] == "reject"
    assert no["source"] == "rule"

    ticket = understand("TCK-1001")
    assert ticket["intent"] == "check_ticket"
    assert ticket["source"] == "rule"
    assert ticket["slots"]["ticket_id"] == "TCK-1001"

    assert _log_len() == before


def test_slots_match_ignores_slots_the_label_does_not_mention():
    expected = {"error_code": "ERR-VPN-809"}
    assert slots_match({"error_code": "ERR-VPN-809", "issue_type": "VPN"}, expected)
    assert not slots_match({"error_code": "ERR-VPN-104"}, expected)
    assert not slots_match({}, expected)


def test_understand_rejects_an_unknown_intent(monkeypatch):
    monkeypatch.setattr("src.nlu.chat", lambda *args, **kwargs: '{"intent": "chat", "slots": {}}')
    with pytest.raises(NLUError, match="unknown intent"):
        understand("What does ERR-VPN-809 mean?")


def test_understand_rejects_a_reply_that_is_not_json(monkeypatch):
    monkeypatch.setattr("src.nlu.chat", lambda *args, **kwargs: "I think this is a VPN question.")
    with pytest.raises(NLUError):
        understand("What does ERR-VPN-809 mean?")


def test_a_known_code_is_copied_when_the_model_leaves_it_null(monkeypatch):
    payload = json.dumps({"intent": "ask_kb", "slots": {"error_code": None}})
    monkeypatch.setattr("src.nlu.chat", lambda *args, **kwargs: payload)
    result = understand("What does ERR-VPN-809 mean?")
    assert result["intent"] == "ask_kb"
    assert result["slots"]["error_code"] == "ERR-VPN-809"
    assert result["slots"]["location"] is None
    assert set(result["slots"]) == set(SLOT_NAMES)


@pytest.mark.live
def test_live_vpn_question_is_ask_kb_with_the_code():
    result = understand(
        "The VPN keeps failing with ERR-VPN-809. What does that code mean?",
        user="U1",
        turn=9,
    )
    assert result["intent"] == "ask_kb"
    assert result["source"] == "llm"
    assert result["slots"]["error_code"] == "ERR-VPN-809"
    assert result["slots"]["date"] is None
    assert result["slots"]["time"] is None


@pytest.mark.live
def test_live_same_sentence_twice_gives_the_same_intent():
    text = "How long does a guest Wi-Fi pass last?"
    first = understand(text)
    second = understand(text)
    assert first["intent"] == second["intent"] == "ask_kb"
    assert first["slots"] == second["slots"]

def test_a_pending_question_is_sent_with_the_reply(monkeypatch):
    seen = {}

    def fake_chat(prompt, **kwargs):
        seen["prompt"] = prompt
        return '{"intent": "book_slot", "slots": {"location": "Pune office, floor 2, desk 1"}}'

    monkeypatch.setattr("src.nlu.chat", fake_chat)
    result = understand(
        "Pune office, floor 2 and desk 1.",
        pending_question="Which office, floor, and desk should the engineer come to?",
    )
    assert result["intent"] == "book_slot"
    assert result["slots"]["location"] == "Pune office, floor 2, desk 1"
    assert seen["prompt"] == (
        "The assistant just asked: Which office, floor, and desk should the engineer come to?\n"
        "The user replied: Pune office, floor 2 and desk 1."
    )


def test_no_pending_question_sends_the_sentence_alone(monkeypatch):
    seen = {}

    def fake_chat(prompt, **kwargs):
        seen["prompt"] = prompt
        return '{"intent": "provide_info", "slots": {"location": "Pune office"}}'

    monkeypatch.setattr("src.nlu.chat", fake_chat)
    understand("I sit at the Pune office.")
    assert seen["prompt"] == "I sit at the Pune office."


def test_a_reschedule_labelled_out_of_scope_becomes_book_slot(monkeypatch):
    payload = json.dumps({"intent": "out_of_scope", "slots": {}})
    monkeypatch.setattr("src.nlu.chat", lambda *args, **kwargs: payload)
    result = understand("Can I change the timing of my booking?")
    assert result["intent"] == "book_slot"
    assert result["slots"]["time"] is None


def test_moving_office_stays_provide_info_and_is_not_a_reschedule(monkeypatch):
    payload = json.dumps({"intent": "out_of_scope", "slots": {}})
    monkeypatch.setattr("src.nlu.chat", lambda *args, **kwargs: payload)
    result = understand("Update my location from Pune to Gurugram")
    assert result["intent"] == "provide_info"
    assert result["slots"]["location"] == "Gurugram"

def test_a_last_issue_labelled_recall_fact_becomes_check_ticket(monkeypatch):
    payload = json.dumps({"intent": "recall_fact", "slots": {}})
    monkeypatch.setattr("src.nlu.chat", lambda *args, **kwargs: payload)
    result = understand("Whats the last issue I raised?")
    assert result["intent"] == "check_ticket"
    assert result["slots"]["ticket_id"] is None


def test_where_do_i_sit_stays_recall_fact(monkeypatch):
    payload = json.dumps({"intent": "recall_fact", "slots": {}})
    monkeypatch.setattr("src.nlu.chat", lambda *args, **kwargs: payload)
    result = understand("Where do I sit?")
    assert result["intent"] == "recall_fact"

def test_sharing_booking_details_labelled_book_slot_becomes_check_ticket(monkeypatch):
    payload = json.dumps({
        "intent": "book_slot",
        "slots": {"location": None, "date": "tuesday", "time": None},
    })
    monkeypatch.setattr("src.nlu.chat", lambda *args, **kwargs: payload)
    result = understand("Share the details of booking")
    assert result["intent"] == "check_ticket"
    assert result["slots"]["date"] is None
    assert result["slots"]["time"] is None


def test_changing_the_timing_is_not_treated_as_a_details_request(monkeypatch):
    payload = json.dumps({"intent": "book_slot", "slots": {}})
    monkeypatch.setattr("src.nlu.chat", lambda *args, **kwargs: payload)
    result = understand("change the timing of my last booking")
    assert result["intent"] == "book_slot"

