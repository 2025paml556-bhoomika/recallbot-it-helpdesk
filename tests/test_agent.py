"""Parts 1 to 4 of the agent: label a turn, carry its slots, act, then reply.

No test here calls Ollama. understand() and chat() are stubbed, and the tools
write to a temporary database, so nothing touches the shared helpdesk file
or the model.
"""

from __future__ import annotations

import json

import pytest

from src.agent import (
    ACTION_ASK,
    ACTION_NONE,
    ACTION_TOOL,
    BOOKING_ASKS,
    BOOKING_SLOTS,
    CONFIRM_BOOKING,
    NO_PASSAGE,
    REPLIES,
    Session,
)
from src.config import LOG_PATH
from src.db import connect
from src.memory import Memory
from src.nlu import SLOT_NAMES, understand


def _log_lines() -> list[str]:
    if not LOG_PATH.exists():
        return []
    return LOG_PATH.read_text(encoding="utf-8").splitlines()


def _log_len() -> int:
    return len(_log_lines())


def _scripted(replies: dict[str, dict]):
    """Stand in for understand(), keyed by the user text.

    A sentence with no scripted reply falls through to the real function, so
    "yes" and "no" still hit the rules and make no model call.
    """

    def fake(text, **kwargs):
        found = replies.get(text)
        if found is None:
            return understand(text, **kwargs)
        slots = {name: None for name in SLOT_NAMES}
        slots.update(found.get("slots", {}))
        return {
            "intent": found["intent"],
            "slots": slots,
            "source": found.get("source", "llm"),
        }

    return fake


def test_rules_are_ingested_without_a_model_call_or_a_log_line(tmp_path):
    before = _log_len()
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))

    yes = session.ingest("yes")
    assert yes["intent"] == "confirm"
    assert yes["source"] == "rule"
    assert yes["turn"] == 1

    no = session.ingest("no")
    assert no["intent"] == "reject"
    assert no["source"] == "rule"
    assert no["turn"] == 2

    ticket = session.ingest("TCK-1001")
    assert ticket["intent"] == "check_ticket"
    assert ticket["source"] == "rule"
    assert ticket["slots"]["ticket_id"] == "TCK-1001"

    assert _log_len() == before


def test_a_later_turn_keeps_slots_from_earlier_turns(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "The VPN keeps failing.": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "VPN failure", "error_code": "ERR-VPN-809"},
                },
                "The Pune office.": {
                    "intent": "book_slot",
                    "slots": {"location": "Pune office"},
                },
                "Thursday, please.": {
                    "intent": "book_slot",
                    "slots": {"date": "Thursday"},
                },
            }
        ),
    )
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))

    session.ingest("The VPN keeps failing.")
    session.ingest("The Pune office.")
    latest = session.ingest("Thursday, please.")

    # The message itself only said the day. The session still holds the rest.
    assert latest["message_slots"]["date"] == "Thursday"
    assert latest["message_slots"]["location"] is None
    assert latest["slots"]["issue_type"] == "VPN failure"
    assert latest["slots"]["error_code"] == "ERR-VPN-809"
    assert latest["slots"]["location"] == "Pune office"
    assert latest["slots"]["date"] == "Thursday"
    assert latest["slots"]["time"] is None
    assert session.missing_booking_slots() == ["time"]
    assert session.pending_booking is True


def test_a_new_value_replaces_the_slot_it_names(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Thursday, please.": {"intent": "book_slot", "slots": {"date": "Thursday"}},
                "Make it Friday.": {"intent": "book_slot", "slots": {"date": "Friday"}},
            }
        ),
    )
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Thursday, please.")
    latest = session.ingest("Make it Friday.")

    assert latest["slots"]["date"] == "Friday"
    assert session.slots["location"] is None


def test_booking_answer_does_not_wipe_the_fault(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Please fix the VPN.": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "VPN failure", "error_code": "ERR-VPN-809"},
                },
                "Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {
                        "issue_type": None,
                        "error_code": None,
                        "date": "Thursday",
                        "time": "4 pm",
                    },
                },
            }
        ),
    )
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Please fix the VPN.")
    latest = session.ingest("Thursday at 4 pm.")

    assert latest["slots"]["issue_type"] == "VPN failure"
    assert latest["slots"]["error_code"] == "ERR-VPN-809"
    assert latest["slots"]["date"] == "Thursday"
    assert latest["slots"]["time"] == "4 pm"


def test_confirm_holds_the_gathered_slots(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Come to the Pune office on Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {
                        "location": "Pune office",
                        "date": "Thursday",
                        "time": "4 pm",
                    },
                },
            }
        ),
    )
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Come to the Pune office on Thursday at 4 pm.")
    assert session.missing_booking_slots() == []

    before = _log_len()
    confirmed = session.ingest("yes")
    assert confirmed["intent"] == "confirm"
    assert confirmed["source"] == "rule"
    assert confirmed["slots"]["location"] == "Pune office"
    assert confirmed["slots"]["date"] == "Thursday"
    assert confirmed["slots"]["time"] == "4 pm"
    assert session.pending_booking is True
    assert _log_len() == before


def test_reject_drops_the_pending_booking_and_changes_no_slot(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Thursday, please.": {
                    "intent": "book_slot",
                    "slots": {"date": "Thursday", "location": "Pune office"},
                },
            }
        ),
    )
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Thursday, please.")
    rejected = session.ingest("no")

    assert rejected["intent"] == "reject"
    assert session.pending_booking is False
    assert rejected["slots"]["date"] == "Thursday"
    assert rejected["slots"]["location"] == "Pune office"


def test_note_booked_clears_the_day_but_keeps_the_place(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Pune office, Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {
                        "location": "Pune office",
                        "date": "Thursday",
                        "time": "4 pm",
                        "issue_type": "VPN failure",
                    },
                },
            }
        ),
    )
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Pune office, Thursday at 4 pm.")
    session.note_booked()

    assert session.pending_booking is False
    assert session.slots["date"] is None
    assert session.slots["time"] is None
    assert session.slots["location"] == "Pune office"
    assert session.slots["issue_type"] == "VPN failure"
    assert session.missing_booking_slots() == ["date", "time"]


def test_a_second_user_does_not_see_the_first_users_slots(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "I sit at the Pune office.": {
                    "intent": "provide_info",
                    "slots": {"location": "Pune office"},
                },
                "I have moved from Pune to Hyderabad.": {
                    "intent": "provide_info",
                    "slots": {"location": "Hyderabad"},
                },
            }
        ),
    )
    meera = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    meera.ingest("I sit at the Pune office.")

    arjun = Session("U2", "B", memory=Memory("U2", "B", root=tmp_path))
    assert arjun.slots_so_far() == {name: None for name in SLOT_NAMES}
    assert arjun.user == "U2"
    assert meera.slots["location"] == "Pune office"


def test_callers_cannot_mutate_the_session_through_the_returned_dict(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Thursday.": {"intent": "book_slot", "slots": {"date": "Thursday"}}}),
    )
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    latest = session.ingest("Thursday.")
    latest["slots"]["date"] = "Friday"
    assert session.slots["date"] == "Thursday"


def test_missing_booking_slots_follow_location_date_time_order(tmp_path):
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    assert session.missing_booking_slots() == list(BOOKING_SLOTS)
    session.slots["date"] = "Thursday"
    assert session.missing_booking_slots() == ["location", "time"]


def _rows(db_path, table: str) -> list[dict]:
    conn = connect(db_path)
    try:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}").fetchall()]
    finally:
        conn.close()


def _tool_lines(lines: list[str]) -> list[dict]:
    return [json.loads(line) for line in lines if json.loads(line)["call_type"] == "tool"]


def test_ask_kb_searches_and_logs_one_tool_call(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"What does ERR-VPN-809 mean?": {"intent": "ask_kb"}}),
    )
    before = _log_len()
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    text = "What does ERR-VPN-809 mean?"
    session.ingest(text)
    decision = session.decide(text)

    assert decision["action"] == ACTION_TOOL
    assert decision["name"] == "search_kb"
    assert decision["result"]
    assert decision["result"][0]["doc_id"] == "KB-VPN"

    added = _tool_lines(_log_lines()[before:])
    assert len(added) == 1
    assert added[0]["tool"] == "search_kb"
    assert added[0]["input_tokens"] == 0
    assert added[0]["output_tokens"] == 0
    assert added[0]["user"] == "U1"


def test_check_ticket_asks_when_no_id_was_given(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"What happened to my ticket?": {"intent": "check_ticket"}}),
    )
    before = _log_len()
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    text = "What happened to my ticket?"
    session.ingest(text)
    decision = session.decide(text)

    assert decision == {
        "action": ACTION_ASK,
        "slot": None,
        "prompt": "Which ticket should I look up?",
    }
    assert _log_len() == before


def test_check_ticket_returns_only_the_owners_row(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Please raise the VPN failure.": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "VPN failure"},
                },
            }
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    owner = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    owner.ingest("Please raise the VPN failure.")
    raised = owner.decide("Please raise the VPN failure.")
    ticket_id = raised["result"]["ticket_id"]

    other = Session("U2", "B", db_path=db_path, memory=Memory("U2", "B", root=tmp_path))
    other.slots["ticket_id"] = ticket_id
    other.history.append({"intent": "check_ticket"})
    other.turn = 1
    before = _log_len()
    decision = other.decide("TCK")

    assert decision["action"] == ACTION_TOOL
    assert decision["name"] == "get_ticket"
    assert decision["result"] == "not found"
    # A ticket owned by someone else is not logged.
    assert _log_len() == before


def test_fix_issue_raises_one_ticket_and_remembers_its_id(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Please raise the VPN failure.": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "VPN failure ERR-VPN-809"},
                },
            }
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    before = _log_len()
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    text = "Please raise the VPN failure."
    session.ingest(text)
    decision = session.decide(text)

    assert decision["action"] == ACTION_TOOL
    assert decision["name"] == "raise_ticket"
    ticket = decision["result"]
    assert ticket["owner_user"] == "U1"
    assert ticket["issue_type"] == "VPN failure ERR-VPN-809"
    assert ticket["status"] == "open"
    assert session.slots["ticket_id"] == ticket["ticket_id"]

    rows = _rows(db_path, "tickets")
    assert len(rows) == 1
    assert rows[0]["ticket_id"] == ticket["ticket_id"]
    assert _rows(db_path, "slots") == []

    added = _tool_lines(_log_lines()[before:])
    assert len(added) == 1
    assert added[0]["tool"] == "raise_ticket"
    assert added[0]["input_tokens"] == 0
    assert added[0]["output_tokens"] == 0


def test_the_same_fault_is_not_raised_twice(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Please raise the VPN failure.": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "VPN failure"},
                },
            }
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    text = "Please raise the VPN failure."
    session.ingest(text)
    session.decide(text)

    before = _log_len()
    session.ingest(text)
    second = session.decide(text)
    assert second == {"action": ACTION_NONE}
    assert len(_rows(db_path, "tickets")) == 1
    assert _log_len() == before


def test_booking_asks_for_the_first_missing_slot_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Can someone come to my desk?": {
                    "intent": "book_slot",
                    "slots": {"issue_type": "VPN failure"},
                },
                "The Pune office.": {"intent": "book_slot", "slots": {"location": "Pune office"}},
                "Thursday, please.": {"intent": "book_slot", "slots": {"date": "Thursday"}},
            }
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    before = _log_len()
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))

    session.ingest("Can someone come to my desk?")
    first = session.decide("Can someone come to my desk?")
    assert first["action"] == ACTION_ASK
    assert first["slot"] == "location"

    session.ingest("The Pune office.")
    second = session.decide("The Pune office.")
    assert second["action"] == ACTION_ASK
    assert second["slot"] == "date"

    session.ingest("Thursday, please.")
    third = session.decide("Thursday, please.")
    assert third["action"] == ACTION_ASK
    assert third["slot"] == "time"

    assert _rows(db_path, "slots") == []
    assert _log_len() == before


def test_a_complete_booking_asks_for_confirmation_and_still_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Pune office, Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {
                        "location": "Pune office, 3rd floor, desk 14",
                        "date": "Thursday",
                        "time": "4 pm",
                        "issue_type": "VPN failure",
                    },
                },
            }
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    before = _log_len()
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    text = "Pune office, Thursday at 4 pm."
    session.ingest(text)
    decision = session.decide(text)

    assert decision == {"action": ACTION_ASK, "slot": None, "prompt": CONFIRM_BOOKING}
    assert session.pending_booking is True
    assert session.awaiting_confirmation is True
    assert _rows(db_path, "slots") == []
    assert _log_len() == before


def test_yes_after_a_complete_booking_writes_exactly_one_row(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Pune office, Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {
                        "location": "Pune office, 3rd floor, desk 14",
                        "date": "Thursday",
                        "time": "4 pm",
                        "issue_type": "VPN failure ERR-VPN-809",
                    },
                },
            }
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Pune office, Thursday at 4 pm.")
    session.decide("Pune office, Thursday at 4 pm.")
    assert _rows(db_path, "slots") == []

    before = _log_len()
    session.ingest("yes")
    decision = session.decide("yes")

    assert decision["action"] == ACTION_TOOL
    assert decision["name"] == "book_it_slot"
    booked = decision["result"]
    assert booked["owner_user"] == "U1"
    assert booked["location"] == "Pune office, 3rd floor, desk 14"
    assert booked["date"] == "Thursday"
    assert booked["time"] == "4 pm"
    assert booked["status"] == "booked"

    rows = _rows(db_path, "slots")
    assert len(rows) == 1
    assert rows[0]["slot_id"] == booked["slot_id"]

    # The day is spent. The place and the fault remain for a follow-up.
    assert session.pending_booking is False
    assert session.awaiting_confirmation is False
    assert session.slots["date"] is None
    assert session.slots["time"] is None
    assert session.slots["location"] == "Pune office, 3rd floor, desk 14"
    assert session.slots["issue_type"] == "VPN failure ERR-VPN-809"

    added = _tool_lines(_log_lines()[before:])
    assert len(added) == 1
    assert added[0]["tool"] == "book_it_slot"
    assert added[0]["confirmed"] is True
    assert added[0]["input_tokens"] == 0
    assert added[0]["output_tokens"] == 0


def test_yes_with_nothing_pending_calls_no_tool(monkeypatch, tmp_path):
    db_path = tmp_path / "helpdesk.sqlite"
    before = _log_len()
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    session.ingest("yes")
    decision = session.decide("yes")

    assert decision == {"action": ACTION_NONE}
    assert _rows(db_path, "slots") == []
    assert _log_len() == before


def test_no_drops_the_booking_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Pune office, Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {
                        "location": "Pune office",
                        "date": "Thursday",
                        "time": "4 pm",
                    },
                },
            }
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Pune office, Thursday at 4 pm.")
    session.decide("Pune office, Thursday at 4 pm.")

    before = _log_len()
    session.ingest("no")
    decision = session.decide("no")

    assert decision == {"action": ACTION_NONE}
    assert session.pending_booking is False
    assert session.awaiting_confirmation is False
    assert _rows(db_path, "slots") == []
    assert _log_len() == before


def test_provide_info_and_a_greeting_call_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "I sit at the Pune office.": {
                    "intent": "provide_info",
                    "slots": {"location": "Pune office"},
                },
                "Hope your day is going well!": {"intent": "out_of_scope"},
            }
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    before = _log_len()
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))

    session.ingest("I sit at the Pune office.")
    assert session.decide("I sit at the Pune office.") == {"action": ACTION_NONE}
    session.ingest("Hope your day is going well!")
    assert session.decide("Hope your day is going well!") == {"action": ACTION_NONE}

    assert session.slots["location"] == "Pune office"
    assert _rows(db_path, "tickets") == []
    assert _rows(db_path, "slots") == []
    assert _log_len() == before


def _scripted_chat(replies: dict[str, str]):
    """Stand in for chat(). The reply is chosen by a marker in the prompt.

    A prompt that matches nothing returns a fixed line, so a test that forgets
    to script one still never reaches Ollama.
    """

    def fake(prompt, **kwargs):
        for marker, reply in replies.items():
            if marker in prompt:
                return reply
        return "scripted reply"

    return fake


def test_an_ask_is_the_reply_and_makes_no_model_call(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Can someone come to my desk?": {"intent": "book_slot"}}),
    )

    def fail(*args, **kwargs):
        raise AssertionError("an ask must not call the model")

    monkeypatch.setattr("src.agent.chat", fail)
    before = _log_len()
    session = Session("U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path))
    text = "Can someone come to my desk?"
    session.ingest(text)
    decision = session.decide(text)
    said = session.reply(text, decision)

    assert decision["slot"] == "location"
    assert said == BOOKING_ASKS["location"]
    assert _log_len() == before


def test_a_tool_result_is_phrased_by_one_model_call(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Please raise the VPN failure.": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "VPN failure"},
                },
            }
        ),
    )
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(kwargs)
        seen["prompt"] = prompt
        return "I've raised TCK-0001 for the VPN failure."

    monkeypatch.setattr("src.agent.chat", fake)
    db_path = tmp_path / "helpdesk.sqlite"
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    text = "Please raise the VPN failure."
    session.ingest(text)
    decision = session.decide(text)
    said = session.reply(text, decision)

    assert said == "I've raised TCK-0001 for the VPN failure."
    assert seen["temperature"] == 0.0
    assert seen["call_type"] == "reply"
    assert seen["user"] == "U1"
    assert seen["session"] == "A"
    assert seen["turn"] == 1
    # The model is shown the row the tool actually wrote, not the user's words alone.
    assert decision["result"]["ticket_id"] in seen["prompt"]
    assert "VPN failure" in seen["prompt"]


def test_a_search_with_nothing_relevant_is_not_handed_a_passage(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"What's on the cafeteria menu?": {"intent": "ask_kb"}}),
    )
    monkeypatch.setattr(
        "src.agent.search_kb",
        lambda *args, **kwargs: [
            {
                "rank": 1,
                "score": 0.05,
                "mode": "dense",
                "chunk_id": "c1",
                "doc_id": "KB-VPN",
                "text": "ERR-VPN-809 means the tunnel dropped.",
            }
        ],
    )
    seen = {}

    def fake(prompt, **kwargs):
        seen["prompt"] = prompt
        return "I don't know."

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    text = "What's on the cafeteria menu?"
    session.ingest(text)
    said = session.reply(text, session.decide(text))

    assert said == "I don't know."
    assert NO_PASSAGE in seen["prompt"]
    assert "ERR-VPN-809" not in seen["prompt"]


def test_a_search_hands_the_model_the_passage_it_may_quote(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"What does ERR-VPN-809 mean?": {"intent": "ask_kb"}}),
    )
    seen = {}

    def fake(prompt, **kwargs):
        seen["prompt"] = prompt
        return "ERR-VPN-809 means the tunnel dropped, from KB-VPN."

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    text = "What does ERR-VPN-809 mean?"
    session.ingest(text)
    said = session.reply(text, session.decide(text))

    assert "KB-VPN" in said
    assert "[KB-VPN]" in seen["prompt"]


def test_a_ticket_that_is_not_the_users_is_phrased_as_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {"Please raise the VPN failure.": {"intent": "fix_issue", "slots": {"issue_type": "VPN"}}}
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    owner = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    owner.ingest("Please raise the VPN failure.")
    ticket_id = owner.decide("Please raise the VPN failure.")["result"]["ticket_id"]

    seen = {}

    def fake(prompt, **kwargs):
        seen["prompt"] = prompt
        return "I couldn't find that ticket."

    monkeypatch.setattr("src.agent.chat", fake)
    other = Session("U2", "B", db_path=db_path, memory=Memory("U2", "B", root=tmp_path))
    other.slots["ticket_id"] = ticket_id
    other.history.append({"intent": "check_ticket"})
    other.turn = 1
    said = other.reply("TCK", other.decide("TCK"))

    assert said == "I couldn't find that ticket."
    assert "No ticket was found" in seen["prompt"]
    assert ticket_id not in seen["prompt"]


def test_a_turn_with_nothing_to_do_uses_a_fixed_line(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Hope your day is going well!": {"intent": "out_of_scope"},
                "I sit at the Pune office.": {
                    "intent": "provide_info",
                    "slots": {"location": "Pune office"},
                },
            }
        ),
    )

    def fail(*args, **kwargs):
        raise AssertionError("a turn with nothing to do must not call the model")

    monkeypatch.setattr("src.agent.chat", fail)
    before = _log_len()
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))

    session.ingest("Hope your day is going well!")
    greeting = session.reply("Hope your day is going well!", session.decide("Hope your day is going well!"))
    session.ingest("I sit at the Pune office.")
    noted = session.reply("I sit at the Pune office.", session.decide("I sit at the Pune office."))
    session.ingest("no")
    declined = session.reply("no", session.decide("no"))

    assert greeting == REPLIES["out_of_scope"]
    assert noted == REPLIES["provide_info"]
    assert declined == REPLIES["reject"]
    assert _log_len() == before


def test_yes_is_phrased_from_the_row_that_was_written(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Pune office, Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {
                        "location": "Pune office, 3rd floor, desk 14",
                        "date": "Thursday",
                        "time": "4 pm",
                    },
                },
            }
        ),
    )
    seen = {}

    def fake(prompt, **kwargs):
        seen["prompt"] = prompt
        return "Booked for Thursday at 4 pm."

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session(
        "U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path)
    )
    session.ingest("Pune office, Thursday at 4 pm.")
    asked = session.reply(
        "Pune office, Thursday at 4 pm.",
        session.decide("Pune office, Thursday at 4 pm."),
    )
    assert asked == CONFIRM_BOOKING
    assert "prompt" not in seen

    session.ingest("yes")
    said = session.reply("yes", session.decide("yes"))
    assert said == "Booked for Thursday at 4 pm."
    assert "Thursday" in seen["prompt"]
    assert "4 pm" in seen["prompt"]
    assert "Pune office, 3rd floor, desk 14" in seen["prompt"]


def test_decide_before_ingest_is_an_error(tmp_path):
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    with pytest.raises(RuntimeError):
        session.decide("Hello.")


def test_a_move_replaces_the_office_a_later_session_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "I sit at the Pune office.": {
                    "intent": "provide_info",
                    "slots": {"location": "Pune office"},
                },
                "I have moved from Pune to Hyderabad.": {
                    "intent": "provide_info",
                    "slots": {"location": "Hyderabad"},
                },
                "Where do I sit?": {"intent": "recall_fact"},
            }
        ),
    )
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)

    first = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    first.ingest("I sit at the Pune office.")

    second = Session("U1", "B", memory=Memory("U1", "B", root=tmp_path))
    moved = second.ingest("I have moved from Pune to Hyderabad.")
    assert moved["intent"] == "provide_info"
    assert moved["slots"]["location"] == "Hyderabad"
    assert [row["value"] for row in second.memory.semantic.active("U1")] == ["Hyderabad"]

    third = Session("U1", "C", memory=Memory("U1", "C", root=tmp_path))
    asked = third.ingest("Where do I sit?")
    assert [fact["value"] for fact in asked["recalled"]["facts"]] == ["Hyderabad"]


def test_a_stated_fact_is_recalled_by_a_later_session(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "I sit at the Pune office.": {
                    "intent": "provide_info",
                    "slots": {"location": "Pune office, 3rd floor, desk 14"},
                },
                "Where do I sit?": {"intent": "recall_fact"},
            }
        ),
    )
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)

    def fail(*args, **kwargs):
        raise AssertionError("a recalled fact is a fixed line, not a model call")

    monkeypatch.setattr("src.agent.chat", fail)

    first = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    first.ingest("I sit at the Pune office.")

    # A new session, and a new process would do the same: it reads the file.
    second = Session("U1", "B", memory=Memory("U1", "B", root=tmp_path))
    text = "Where do I sit?"
    recalled = second.ingest(text)
    offices = [fact["value"] for fact in recalled["recalled"]["facts"]]
    assert "Pune office, 3rd floor, desk 14" in offices
    # The question states nothing, so the stored fact is not rewritten.
    assert len(second.memory.semantic.active("U1")) == 1

    said = second.reply(text, second.decide(text))
    assert said == "You sit at Pune office, 3rd floor, desk 14."

    # A different user reads the same folder and finds nothing of U1's.
    other = Session("U2", "B", memory=Memory("U2", "B", root=tmp_path))
    theirs = other.ingest(text)
    assert theirs["recalled"]["facts"] == []
    assert theirs["recalled"]["episodes"] == []
    assert other.reply(text, other.decide(text)) == REPLIES["recall_fact"]


def _fake_embed(texts, **kwargs):
    """A stand-in for the embedding model, so memory tests make no model call.

    Each text becomes a unit vector with one axis per character it contains.
    The width matches the real model, because the stored index is built at
    that width. Two texts about the same place then point the same way.
    """
    import numpy as np

    from src.config import EMBED_DIM

    rows = []
    for text in texts:
        vector = np.zeros(EMBED_DIM, dtype=np.float32)
        for character in text.lower():
            vector[ord(character) % EMBED_DIM] += 1.0
        norm = np.linalg.norm(vector)
        rows.append(vector / norm if norm else vector)
    return np.vstack(rows)


def test_ingest_passes_user_session_and_turn_to_nlu(monkeypatch, tmp_path):
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)
    seen = {}

    def fake(text, **kwargs):
        seen.update(kwargs)
        slots = {name: None for name in SLOT_NAMES}
        return {"intent": "out_of_scope", "slots": slots, "source": "llm"}

    monkeypatch.setattr("src.agent.understand", fake)
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Hello.")
    session.ingest("Thanks.")

    assert seen == {"user": "U1", "session": "A", "turn": 2}
    assert [item["turn"] for item in session.history] == [1, 2]
    assert json.loads(json.dumps(session.history))[1]["text"] == "Thanks."
