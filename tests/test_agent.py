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
    FALLBACK,
    BOOKING_ASKS,
    BOOKING_SLOTS,
    BOOKING_STEPS_PATH,
    CONFIRM_BOOKING,
    OPEN_TEMPERATURE,
    OPEN_TOP_P,
    REPLIES,
    is_greeting,
    SYSTEM_PROMPT_PATH,
    Session,
    load_skill,
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
        seen = replies.get("_seen")
        if isinstance(seen, list):
            seen.append(kwargs.get("pending_question"))
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
                    "slots": {
                        "issue_type": "VPN failure",
                        "location": "Pune office",
                        "date": "Thursday",
                        "time": "4 pm",
                    },
                },
            }
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    owner = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    owner.ingest("Please raise the VPN failure.")
    owner.decide("Please raise the VPN failure.")
    owner.ingest("yes")
    raised = owner.decide("yes")
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


def test_fix_issue_asks_for_the_visit_before_raising_a_ticket(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Please raise the VPN failure.": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "VPN failure ERR-VPN-809"},
                },
                "The Pune office.": {"intent": "book_slot", "slots": {"location": "Pune office"}},
                "Thursday, please.": {"intent": "book_slot", "slots": {"date": "Thursday"}},
                "4 pm.": {"intent": "book_slot", "slots": {"time": "4 pm"}},
            }
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    before = _log_len()
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))

    text = "Please raise the VPN failure."
    session.ingest(text)
    decision = session.decide(text)
    assert decision == {
        "action": ACTION_ASK,
        "slot": "location",
        "prompt": BOOKING_ASKS["location"],
    }
    assert session.pending_booking is True
    assert session.slots["ticket_id"] is None
    assert _rows(db_path, "tickets") == []
    assert _log_len() == before

    session.ingest("The Pune office.")
    assert session.decide("The Pune office.")["slot"] == "date"
    session.ingest("Thursday, please.")
    assert session.decide("Thursday, please.")["slot"] == "time"
    session.ingest("4 pm.")
    confirm = session.decide("4 pm.")
    assert confirm == {"action": ACTION_ASK, "slot": None, "prompt": CONFIRM_BOOKING}
    assert _rows(db_path, "tickets") == []
    assert _rows(db_path, "slots") == []

    session.ingest("yes")
    booked = session.decide("yes")
    assert booked["action"] == ACTION_TOOL
    assert booked["name"] == "book_it_slot"
    assert booked["result"]["location"] == "Pune office"
    assert booked["result"]["date"] == "Thursday"
    assert booked["result"]["time"] == "4 pm"
    assert booked["result"]["ticket_id"] == session.slots["ticket_id"]

    tickets = _rows(db_path, "tickets")
    assert len(tickets) == 1
    assert tickets[0]["ticket_id"] == session.slots["ticket_id"]
    assert tickets[0]["owner_user"] == "U1"
    assert tickets[0]["issue_type"] == "VPN failure ERR-VPN-809"
    assert tickets[0]["status"] == "open"
    assert len(_rows(db_path, "slots")) == 1

    added = _tool_lines(_log_lines()[before:])
    assert [row["tool"] for row in added] == ["raise_ticket", "book_it_slot"]


def test_the_same_fault_is_not_raised_twice(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Please raise the VPN failure.": {
                    "intent": "fix_issue",
                    "slots": {
                        "issue_type": "VPN failure",
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
    text = "Please raise the VPN failure."
    session.ingest(text)
    asked = session.decide(text)
    assert asked["prompt"] == CONFIRM_BOOKING
    session.ingest("yes")
    session.decide("yes")
    assert len(_rows(db_path, "tickets")) == 1

    before = _log_len()
    session.ingest(text)
    second = session.decide(text)
    assert second == {"action": ACTION_NONE}
    assert len(_rows(db_path, "tickets")) == 1
    # No second ticket, so no second tool line. Recalling the repeated
    # sentence still embeds the query.
    added = [json.loads(line) for line in _log_lines()[before:]]
    assert [row["call_type"] for row in added] == ["embed"]
    assert "search_kb" not in {row.get("tool") for row in added}


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


def test_a_fix_issue_answer_labelled_provide_info_still_asks_for_the_day(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "My downloads are slow on laptop, what to do?": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "slow downloads", "asset_id": "laptop"},
                },
                "Pune office, floor 2 and desk 14.": {
                    "intent": "provide_info",
                    "slots": {"location": "Pune office, floor 2, desk 14"},
                },
            }
        ),
    )
    session = Session("U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("My downloads are slow on laptop, what to do?")
    asked = session.decide("My downloads are slow on laptop, what to do?")
    assert asked["slot"] == "location"

    session.ingest("Pune office, floor 2 and desk 14.")
    nxt = session.decide("Pune office, floor 2 and desk 14.")
    assert nxt == {"action": ACTION_ASK, "slot": "date", "prompt": BOOKING_ASKS["date"]}
    assert session.pending_booking is True
    assert session.pending_ticket is True
    assert session.slots["location"] == "Pune office, floor 2, desk 14"
    assert _rows(tmp_path / "helpdesk.sqlite", "tickets") == []


def test_a_time_labelled_ask_kb_fills_the_slot_instead_of_searching(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "My laptop is facing screen issues.": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "screen issues", "asset_id": "laptop"},
                },
                "I sit a pune office at floor 2 and desk 14.": {
                    "intent": "provide_info",
                    "slots": {"location": "Pune office, floor 2, desk 14"},
                },
                "Thursday, next week.": {
                    "intent": "book_slot",
                    "slots": {"date": "Thursday, next week"},
                },
                "12 PM": {"intent": "ask_kb", "slots": {"time": "12 PM"}},
            }
        ),
    )
    session = Session("U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path))
    for text in (
        "My laptop is facing screen issues.",
        "I sit a pune office at floor 2 and desk 14.",
        "Thursday, next week.",
    ):
        session.ingest(text)
        session.decide(text)

    session.ingest("12 PM")
    decision = session.decide("12 PM")
    assert decision == {"action": ACTION_ASK, "slot": None, "prompt": CONFIRM_BOOKING}
    assert session.slots["time"] == "12 PM"
    assert session.tool_call is None


def test_ask_kb_during_a_booking_still_searches_when_no_slot_was_filled(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "My laptop is facing screen issues.": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "screen issues"},
                },
                "What does ERR-VPN-809 mean?": {"intent": "ask_kb"},
            }
        ),
    )
    session = Session("U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("My laptop is facing screen issues.")
    session.decide("My laptop is facing screen issues.")

    session.ingest("What does ERR-VPN-809 mean?")
    decision = session.decide("What does ERR-VPN-809 mean?")
    assert decision["action"] == ACTION_TOOL
    assert decision["name"] == "search_kb"


def test_provide_info_logs_the_extraction_and_a_greeting_calls_nothing(monkeypatch, tmp_path):
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
    # The greeting ran nothing. Stating the office logs the new fact's
    # embedding and the extraction. Recalling memory for the second turn
    # embeds the query. Neither line is a tool call.
    added = [json.loads(line) for line in _log_lines()[before:]]
    assert [row["call_type"] for row in added] == ["embed", "extract", "embed"]
    fact, extract, query = added
    assert fact["user"] == "U1" and fact["session"] == "A" and fact["turn"] == 1
    # No ablation run was named, so the line keeps the process run id.
    assert fact["run_id"]
    assert extract["call_type"] == "extract"
    assert extract["user"] == "U1"
    assert extract["turn"] == 1
    assert extract["retrieved_tokens"] == 0
    assert "location" in extract["keys"]
    assert query["turn"] == 2
    assert query["user"] == "U1" and query["session"] == "A"


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
                    "slots": {
                        "issue_type": "VPN failure",
                        "location": "Pune office",
                        "date": "Thursday",
                        "time": "4 pm",
                    },
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
    asked = session.decide(text)
    assert asked["action"] == ACTION_ASK
    session.ingest("yes")
    decision = session.decide("yes")
    said = session.reply("yes", decision)

    assert said == "I've raised TCK-0001 for the VPN failure."
    assert seen["temperature"] == 0.0
    assert seen["call_type"] == "chat"
    assert seen["user"] == "U1"
    assert seen["session"] == "A"
    assert seen["turn"] == 2
    # The model is shown the row the tool actually wrote, not the user's words alone.
    assert "Booked slot" in seen["prompt"]
    assert "Pune office" in seen["prompt"]
    assert "Thursday" in seen["prompt"]
    assert decision["result"]["ticket_id"]


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

    # Nothing cleared the cut-off, so the model is not called at all.
    assert said == FALLBACK
    assert seen == {}


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
    # The model named the document and not the chunk. The reply still has to
    # carry the chunk it was actually shown.
    assert "[KB-VPN#" in said
    assert said.endswith("]")

    assert "KB-VPN" in said
    # The prompt labels the passage with the chunk, not the bare document.
    assert "[KB-VPN#" in seen["prompt"]
    assert "[KB-VPN]" not in seen["prompt"]


def test_a_ticket_that_is_not_the_users_is_phrased_as_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {"Please raise the VPN failure.": {"intent": "fix_issue", "slots": {"issue_type": "VPN", "location": "Pune office", "date": "Thursday", "time": "4 pm"}}}
        ),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    owner = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    owner.ingest("Please raise the VPN failure.")
    owner.decide("Please raise the VPN failure.")
    owner.ingest("yes")
    ticket_id = owner.decide("yes")["result"]["ticket_id"]

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


def test_a_kb_reply_drops_a_citation_for_a_chunk_it_was_not_shown(monkeypatch, tmp_path):
    """The model cites a document the search did not return. That cite goes."""
    from src.agent import chunk_cite, ground_citations

    allowed = "[KB-VPN#KB-VPN@2026.1:128:0]"
    invented = "The tunnel dropped [KB-PASSWORD#KB-PASSWORD@2025.1:128:0]."
    grounded = ground_citations(invented, [allowed])
    assert "[KB-PASSWORD#" not in grounded
    assert grounded.endswith(allowed)

    # A cite the model copied from the passage it was shown is kept, and not
    # doubled by the one that gets appended when none survived.
    copied = ground_citations(f"The tunnel dropped {allowed}", [allowed])
    assert copied.count(allowed) == 1

    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"What does ERR-VPN-809 mean?": {"intent": "ask_kb"}}),
    )
    monkeypatch.setattr(
        "src.agent.search_kb",
        lambda *args, **kwargs: [
            {
                "rank": 1,
                "score": 0.9,
                "mode": "hybrid",
                "chunk_id": "KB-VPN@2026.1:128:0",
                "doc_id": "KB-VPN",
                "text": "ERR-VPN-809 means the tunnel dropped.",
            }
        ],
    )

    def fake(prompt, **kwargs):
        return "The tunnel dropped [KB-SLA#KB-SLA@2026.1:128:0]."

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    text = "What does ERR-VPN-809 mean?"
    session.ingest(text)
    said = session.reply(text, session.decide(text))
    assert "[KB-SLA#" not in said
    assert said.endswith(chunk_cite({
        "doc_id": "KB-VPN",
        "chunk_id": "KB-VPN@2026.1:128:0",
    }))


def test_an_open_turn_is_reworded_and_reads_the_skill_file(monkeypatch, tmp_path):
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
    seen = []

    def fake(prompt, **kwargs):
        seen.append({"prompt": prompt, **kwargs})
        return "reworded: " + prompt.splitlines()[-1]

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))

    session.ingest("Hope your day is going well!")
    greeting = session.reply("Hope your day is going well!", session.decide("Hope your day is going well!"))
    session.ingest("I sit at the Pune office.")
    noted = session.reply("I sit at the Pune office.", session.decide("I sit at the Pune office."))
    session.ingest("no")
    declined = session.reply("no", session.decide("no"))

    # "Hope your day is going well!" is out of scope but not a hello, so it
    # gets the plain refusal rather than the greeting line.
    assert REPLIES["out_of_scope"] in greeting
    assert REPLIES["greeting"] not in greeting
    # A stored fact is the fixed thanks line. It is not reworded, because
    # the persona's "I don't know." rule was winning over the required reply.
    assert noted == REPLIES["provide_info"]
    assert REPLIES["reject"] in declined
    # The stored fact returns without a model call. The refusal and the
    # rejection are still reworded.
    assert len(seen) == 2
    for call in seen:
        assert call["temperature"] == 0.7
        assert call["top_p"] == 0.9
        assert call["call_type"] == "chat"
        # The persona was read from skills/system_prompt_v1.md, not inlined.
        assert "corporate IT helpdesk" in call["system"]
        # An open turn is not a booking, so the slot steps stay out.
        assert "desk-side visit" not in call["system"]


def test_only_a_bare_hello_is_answered_with_hi(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Hi": {"intent": "out_of_scope"},
                "Good morning!": {"intent": "out_of_scope"},
                "Hi, my VPN is down.": {"intent": "out_of_scope"},
                "What's for lunch?": {"intent": "out_of_scope"},
            }
        ),
    )

    def fake(prompt, **kwargs):
        return prompt.splitlines()[-1]

    monkeypatch.setattr("src.agent.chat", fake)

    def said(text: str) -> str:
        session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
        session.ingest(text)
        return session.reply(text, session.decide(text))

    assert is_greeting("Hi")
    assert is_greeting("Hello!")
    assert is_greeting("Hey there")
    assert is_greeting("Good morning!")
    assert not is_greeting("Hi, my VPN is down.")
    assert not is_greeting("What's for lunch?")

    assert said("Hi") == REPLIES["greeting"]
    assert said("Good morning!") == REPLIES["greeting"]
    # A hello with a request after it, and a plain off-topic question, both
    # stay refusals. Neither is answered with "Hi".
    assert said("Hi, my VPN is down.") == REPLIES["out_of_scope"]
    assert said("What's for lunch?") == REPLIES["out_of_scope"]


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


def test_located_and_asset_id_are_recalled_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "I sit at the Hyderabad office.": {
                    "intent": "provide_info",
                    "slots": {"location": "Hyderabad office"},
                },
                "My laptop is LAP-T14G3.": {
                    "intent": "provide_info",
                    "slots": {"asset_id": "LAP-T14G3"},
                },
                "Where am I located": {"intent": "recall_fact"},
                "Whats my asset Id ?": {"intent": "recall_fact"},
            }
        ),
    )
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)

    first = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    first.ingest("I sit at the Hyderabad office.")
    first.ingest("My laptop is LAP-T14G3.")

    second = Session("U1", "B", memory=Memory("U1", "B", root=tmp_path))
    place = "Where am I located"
    second.ingest(place)
    place_reply = second.reply(place, second.decide(place))
    assert place_reply.startswith("You sit at Hyderabad office.")
    assert "[mem:" in place_reply
    asset = "Whats my asset Id ?"
    second.ingest(asset)
    asset_reply = second.reply(asset, second.decide(asset))
    assert asset_reply.startswith("You use LAP-T14G3.")
    assert "[mem:" in asset_reply


def test_a_colour_is_remembered_and_a_badge_does_not_become_the_laptop(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "My name is Bhoomika.": {"intent": "provide_info", "slots": {}},
                "My laptop bag is blue.": {"intent": "provide_info", "slots": {}},
                "My badge number is 4412.": {
                    "intent": "provide_info",
                    "slots": {"asset_id": "4412"},
                },
                "My laptop is LAP-T14G3.": {
                    "intent": "provide_info",
                    "slots": {"asset_id": "LAP-T14G3"},
                },
                "What colour is my laptop bag?": {"intent": "recall_fact"},
                "Whats my asset Id ?": {"intent": "recall_fact"},
            }
        ),
    )
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)

    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("My name is Bhoomika.")
    session.ingest("My laptop bag is blue.")
    session.ingest("My badge number is 4412.")
    session.ingest("My laptop is LAP-T14G3.")
    active = session.memory.semantic.active("U1")
    assert [row["value"] for row in active if row["key"] == "asset_id"] == ["LAP-T14G3"]
    assert "4412" not in [row["value"] for row in active if row["key"] == "asset_id"]

    later = Session("U1", "B", memory=Memory("U1", "B", root=tmp_path))
    colour = "What colour is my laptop bag?"
    later.ingest(colour)
    colour_reply = later.reply(colour, later.decide(colour))
    assert colour_reply.startswith("You told me: My laptop bag is blue.")
    assert "[mem:" in colour_reply
    asset = "Whats my asset Id ?"
    later.ingest(asset)
    asset_reply = later.reply(asset, later.decide(asset))
    assert asset_reply.startswith("You use LAP-T14G3.")
    assert "[mem:" in asset_reply


def test_provide_info_that_stores_nothing_does_not_claim_to_remember(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Thanks.": {"intent": "provide_info", "slots": {}}}),
    )
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)

    def fail_chat(*args, **kwargs):
        raise AssertionError("a memory miss must not call the chat model")

    monkeypatch.setattr("src.agent.chat", fail_chat)
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    # Small talk is labelled provide_info here and writes no row.
    text = "Thanks."
    session.ingest(text)
    assert session.facts_written == []
    assert session.reply(text, session.decide(text)) == REPLIES["recall_fact"]


def test_a_past_event_is_read_from_the_episode_log(monkeypatch, tmp_path):
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"What did we do last time?": {"intent": "out_of_scope"}}),
    )
    memory = Memory("U1", "B", root=tmp_path)
    memory.episodic.add(
        "U1",
        "Raised TCK-1042 for the laptop screen.",
        "ticket",
        session="A",
    )
    session = Session("U1", "B", memory=memory)
    text = "What did we do last time?"
    labelled = session.ingest(text)
    assert labelled["recalled"]["episodes"]
    decision = session.decide(text)
    assert decision["action"] == ACTION_TOOL
    assert decision["name"] == "recall_episode"
    said = session.reply(text, decision)
    assert said.startswith("Last time, Raised TCK-1042 for the laptop screen.")
    assert "[mem:" in said

    other = Session("U2", "B", memory=Memory("U2", "B", root=tmp_path))
    other.ingest(text)
    missed = other.decide(text)
    assert missed["action"] == ACTION_NONE
    # No model call: the miss is the fixed line, not a reworded refusal.
    def fail_chat(*args, **kwargs):
        raise AssertionError("a memory miss must not call the chat model")

    monkeypatch.setattr("src.agent.chat", fail_chat)
    assert other.reply(text, missed) == FALLBACK


def _fake_embed(texts, **kwargs):
    """A stand-in for the embedding model, so memory tests make no model call.

    Each text becomes a unit vector with one axis per character it contains.
    The width matches the real model, because the stored index is built at
    that width. Two texts about the same place then point the same way.
    A logged call writes the embed line, so the log tests see it.
    """
    import time

    import numpy as np

    from src.config import EMBED_DIM

    if kwargs.get("log", True):
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

    assert seen["user"] == "U1"
    assert seen["session"] == "A"
    assert seen["turn"] == 2
    # The same list the session file is filled from, so the labelling call
    # can be recorded beside the reply.
    assert seen["trace"] is session.calls
    assert [item["turn"] for item in session.history] == [1, 2]
    assert json.loads(json.dumps(session.history))[1]["text"] == "Thanks."

def test_a_reply_is_appended_to_that_sessions_json_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Hello.": {"intent": "out_of_scope"}}),
    )
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)

    session = Session("U1", "S-1", memory=Memory("U1", "S-1", root=tmp_path))
    text = "Hello."
    session.ingest(text)
    said = session.reply(text, session.decide(text))

    saved = json.loads(session.memory.conversation.path.read_text(encoding="utf-8"))
    assert saved["user"] == "U1"
    assert saved["session"] == "S-1"
    assert saved["turns"][0]["turn"] == 1
    assert saved["turns"][0]["user"] == text
    assert saved["turns"][0]["bot"] == said
    assert saved["turns"][0]["intent"] == "out_of_scope"
    # An open turn is one reply call, logged beside the sentence.
    assert saved["turns"][0]["calls"][0]["call_type"] == "chat"

    other = Session("U1", "S-2", memory=Memory("U1", "S-2", root=tmp_path))
    other.ingest(text)
    other.reply(text, other.decide(text))
    assert session.memory.conversation.path != other.memory.conversation.path
    # Ending the session keeps the transcript. Only the working summary goes.
    session.end("Nothing to keep.")
    assert session.memory.conversation.path.exists()


def test_a_tool_turn_logs_the_prompts_and_the_tool_call(monkeypatch, tmp_path):
    from src.llm import ChatResult

    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"What does ERR-VPN-809 mean?": {"intent": "ask_kb"}}),
    )
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)
    monkeypatch.setattr(
        "src.agent.search_kb",
        lambda query, **kwargs: [{"doc_id": "vpn.md", "text": "ERR-VPN-809 means the tunnel dropped.", "mode": "hybrid", "score": 1.0}],
    )

    def fake_chat(prompt, **kwargs):
        result = ChatResult(
            "ERR-VPN-809 means the tunnel dropped, from vpn.md.",
            system=kwargs.get("system"),
            user=prompt,
            call_type=kwargs.get("call_type", "chat"),
            model="stub",
        )
        trace = kwargs.get("trace")
        if trace is not None:
            trace.append(result.trace())
        return result

    monkeypatch.setattr("src.agent.chat", fake_chat)
    session = Session("U1", "S-9", memory=Memory("U1", "S-9", root=tmp_path))
    text = "What does ERR-VPN-809 mean?"
    session.ingest(text)
    said = session.reply(text, session.decide(text))

    saved = json.loads(session.memory.conversation.path.read_text(encoding="utf-8"))
    turn = saved["turns"][0]
    assert turn["bot"] == said
    assert turn["tool"]["name"] == "search_kb"
    assert turn["tool"]["arguments"]["query"] == text
    assert "ERR-VPN-809" in turn["tool"]["result"][0]["text"]
    roles = [message["role"] for message in turn["calls"][0]["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "Tool result:" in turn["calls"][0]["messages"][1]["content"]
    assert turn["calls"][0]["messages"][2]["content"] == "ERR-VPN-809 means the tunnel dropped, from vpn.md."
    # The stub named no chunk, so the reply carries the chunk it was shown.
    assert "[vpn.md#" in said

def test_the_office_question_is_sent_when_labelling_the_answer(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "_seen": seen,
                "My downloads are slow on laptop, what to do?": {
                    "intent": "fix_issue",
                    "slots": {"issue_type": "slow downloads"},
                },
                "Pune office, floor 2 and desk 1.": {
                    "intent": "provide_info",
                    "slots": {"location": "Pune office, floor 2, desk 1"},
                },
            }
        ),
    )
    session = Session("U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("My downloads are slow on laptop, what to do?")
    asked = session.decide("My downloads are slow on laptop, what to do?")
    assert asked["prompt"] == BOOKING_ASKS["location"]

    session.ingest("Pune office, floor 2 and desk 1.")
    nxt = session.decide("Pune office, floor 2 and desk 1.")
    assert seen == [None, BOOKING_ASKS["location"]]
    assert nxt == {"action": ACTION_ASK, "slot": "date", "prompt": BOOKING_ASKS["date"]}
    assert session.slots["location"] == "Pune office, floor 2, desk 1"


def _booked_visit(monkeypatch, tmp_path):
    """A confirmed visit: Pune office, Thursday, 4 pm. Returns session, db path."""
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
    db_path = tmp_path / "helpdesk.sqlite"
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Pune office, Thursday at 4 pm.")
    session.decide("Pune office, Thursday at 4 pm.")
    session.ingest("yes")
    session.decide("yes")
    return session, db_path


def test_changing_the_timing_asks_for_a_new_time(monkeypatch, tmp_path):
    session, db_path = _booked_visit(monkeypatch, tmp_path)
    assert len(_rows(db_path, "slots")) == 1
    # understand() now returns book_slot for this sentence. The script stands
    # in for that, including the case where the model said out_of_scope and
    # the NLU override corrected it.
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Can I change the timing of my booking?": {"intent": "book_slot", "slots": {}},
            }
        ),
    )
    text = "Can I change the timing of my booking?"
    latest = session.ingest(text)
    decision = session.decide(text)

    assert latest["intent"] == "book_slot"
    # The day stays. Only the time the user asked to change is collected again.
    assert latest["slots"]["time"] is None
    assert latest["slots"]["date"] == "Thursday"
    assert latest["slots"]["location"] == "Pune office"
    assert decision == {
        "action": ACTION_ASK,
        "slot": "time",
        "prompt": BOOKING_ASKS["time"],
    }
    assert session.pending_booking is True
    assert session.awaiting_confirmation is False
    assert session.reschedule_slot_id == _rows(db_path, "slots")[0]["slot_id"]
    assert len(_rows(db_path, "slots")) == 1


def test_a_new_time_on_a_reschedule_asks_to_confirm(monkeypatch, tmp_path):
    session, db_path = _booked_visit(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Move it to 4 pm.": {"intent": "book_slot", "slots": {"time": "5 pm"}}}),
    )
    text = "Move it to 4 pm."
    latest = session.ingest(text)
    decision = session.decide(text)

    assert latest["slots"]["time"] == "5 pm"
    # The day is restored from the visit being changed, so only a confirm
    # is left. Saying yes updates that row instead of inserting another.
    assert latest["slots"]["date"] == "Thursday"
    assert latest["slots"]["location"] == "Pune office"
    assert decision == {
        "action": ACTION_ASK,
        "slot": None,
        "prompt": CONFIRM_BOOKING,
    }
    assert len(_rows(db_path, "slots")) == 1

    session.ingest("yes")
    updated = session.decide("yes")
    assert updated["name"] == "update_slot"
    rows = _rows(db_path, "slots")
    assert len(rows) == 1
    assert rows[0]["time"] == "5 pm"
    assert rows[0]["date"] == "Thursday"
    assert rows[0]["location"] == "Pune office"
    assert rows[0]["slot_id"] == updated["result"]["slot_id"]

def test_last_issue_with_no_id_reads_the_newest_ticket(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Please raise the VPN failure.": {
            "intent": "fix_issue",
            "slots": {
                "issue_type": "VPN connection",
                "location": "Gurugram office",
                "date": "Sunday",
                "time": "2 pm",
            },
        }}),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Please raise the VPN failure.")
    session.decide("Please raise the VPN failure.")
    session.ingest("yes")
    raised = session.decide("yes")
    ticket_id = raised["result"]["ticket_id"]

    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"what was the last issue I raised?": {"intent": "check_ticket"}}),
    )
    text = "what was the last issue I raised?"
    session.ingest(text)
    decision = session.decide(text)

    assert decision["action"] == ACTION_TOOL
    assert decision["name"] == "get_ticket"
    assert decision["result"]["ticket_id"] == ticket_id
    assert decision["result"]["issue_type"] == "VPN connection"
    # The id was not in the message, so the carried slot stays empty.
    assert session.slots["ticket_id"] == ticket_id


def test_last_issue_labelled_recall_fact_still_reads_the_ticket(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Please raise the VPN failure.": {
            "intent": "fix_issue",
            "slots": {
                "issue_type": "VPN connection",
                "location": "Gurugram office",
                "date": "Sunday",
                "time": "2 pm",
            },
        }}),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    session.ingest("Please raise the VPN failure.")
    session.decide("Please raise the VPN failure.")
    session.ingest("yes")
    session.decide("yes")

    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Whats the last issue I raised?": {"intent": "recall_fact"}}),
    )
    text = "Whats the last issue I raised?"
    session.ingest(text)
    decision = session.decide(text)
    assert decision["name"] == "get_ticket"
    assert decision["result"]["owner_user"] == "U1"
    assert decision["result"]["issue_type"] == "VPN connection"


def test_last_booking_reads_the_visit_not_the_ticket(monkeypatch, tmp_path):
    session, _db_path = _booked_visit(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"what is my last booking?": {"intent": "check_ticket"}}),
    )
    text = "what is my last booking?"
    session.ingest(text)
    decision = session.decide(text)

    assert decision["action"] == ACTION_TOOL
    assert decision["name"] == "get_booking"
    assert decision["result"]["location"] == "Pune office"
    assert decision["result"]["date"] == "Thursday"
    assert decision["result"]["time"] == "4 pm"
    assert decision["result"]["ticket_id"] is None


def test_another_users_last_issue_is_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Please raise the VPN failure.": {
            "intent": "fix_issue",
            "slots": {
                "issue_type": "VPN connection",
                "location": "Gurugram office",
                "date": "Sunday",
                "time": "2 pm",
            },
        }}),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    owner = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    owner.ingest("Please raise the VPN failure.")
    owner.decide("Please raise the VPN failure.")
    owner.ingest("yes")
    owner.decide("yes")

    other = Session("U2", "B", db_path=db_path, memory=Memory("U2", "B", root=tmp_path / "u2"))
    other.history.append({"intent": "check_ticket"})
    other.turn = 1
    decision = other.decide("what was the last issue I raised?")
    assert decision["name"] == "get_ticket"
    assert decision["result"] == "not found"


def test_a_status_question_that_names_nothing_still_asks(monkeypatch, tmp_path):
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    session.history.append({"intent": "check_ticket"})
    session.turn = 1
    decision = session.decide("How is it going?")
    assert decision == {
        "action": ACTION_ASK,
        "slot": None,
        "prompt": "Which ticket should I look up?",
    }

def test_share_the_details_of_booking_reads_the_visit(monkeypatch, tmp_path):
    session, db_path = _booked_visit(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Share the details of booking": {"intent": "book_slot", "slots": {}}}),
    )
    # The live path corrects book_slot before decide(). This test covers the
    # lookup itself when the sentence still arrives labelled book_slot, which
    # is what used to ask "Which day works for the visit?".
    text = "Share the details of booking"
    # Re-label the way understand() does, so a book_slot mislabel is corrected.
    from src.nlu import _apply_known_codes, empty_slots
    corrected = _apply_known_codes(text, {"intent": "book_slot", "slots": empty_slots()})
    assert corrected["intent"] == "check_ticket"
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({text: corrected}),
    )
    session.ingest(text)
    decision = session.decide(text)
    assert decision["action"] == ACTION_TOOL
    assert decision["name"] == "get_booking"
    assert decision["result"]["location"] == "Pune office"
    assert decision["result"]["time"] == "4 pm"
    assert len(_rows(db_path, "slots")) == 1
    assert session.pending_booking is False

def test_booking_details_are_phrased_without_the_issue(monkeypatch, tmp_path):
    session, _db_path = _booked_visit(monkeypatch, tmp_path)
    session.slots["issue_type"] = "network_connection"
    seen = {}

    def fake(prompt, **kwargs):
        seen["prompt"] = prompt
        return "Booking 1 at Pune office on Thursday at 4 pm, status booked."

    monkeypatch.setattr("src.agent.chat", fake)
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"Details of my last booking, please": {"intent": "check_ticket"}}),
    )
    text = "Details of my last booking, please"
    session.ingest(text)
    said = session.reply(text, session.decide(text))
    assert said.startswith("Booking")
    assert "network_connection" not in seen["prompt"]
    assert "issue" not in seen["prompt"].lower()
    assert "Pune office" in seen["prompt"]
    assert "Thursday" in seen["prompt"]
    assert "4 pm" in seen["prompt"]



def test_a_tool_reply_loads_both_skill_files_while_booking(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Pune office, Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {"location": "Pune office", "date": "Thursday", "time": "4 pm"},
                },
            }
        ),
    )
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(kwargs)
        return "Booked."

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session(
        "U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path)
    )
    session.ingest("Pune office, Thursday at 4 pm.")
    session.decide("Pune office, Thursday at 4 pm.")
    session.ingest("yes")
    session.reply("yes", session.decide("yes"))

    persona = load_skill(SYSTEM_PROMPT_PATH)
    steps = load_skill(BOOKING_STEPS_PATH)
    assert persona in seen["system"]
    assert steps in seen["system"]
    assert seen["temperature"] == 0.0
    assert "top_p" not in seen


def test_a_written_visit_is_reported_as_booked(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Pune office, Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {"location": "Pune office", "date": "Thursday", "time": "4 pm"},
                },
            }
        ),
    )

    def fake(prompt, **kwargs):
        # What the model must be told: the visit is already written, and the
        # confirmation question is not the reply to that result.
        assert "Booked slot" in prompt
        assert "already booked" in kwargs["system"]
        return "Booked slot 1 at Pune office on Thursday at 4 pm."

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session(
        "U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path)
    )
    session.ingest("Pune office, Thursday at 4 pm.")
    session.decide("Pune office, Thursday at 4 pm.")
    session.ingest("yes")
    said = session.reply("yes", session.decide("yes"))
    assert said.startswith("Booked slot")
    # A second yes has nothing left to confirm.
    session.ingest("yes")
    assert session.decide("yes") == {"action": ACTION_NONE}


def test_a_knowledge_reply_loads_the_persona_but_not_the_booking_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"What does ERR-VPN-809 mean?": {"intent": "ask_kb"}}),
    )
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(kwargs)
        return "From the VPN guide."

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    text = "What does ERR-VPN-809 mean?"
    session.ingest(text)
    session.reply(text, session.decide(text))

    assert load_skill(SYSTEM_PROMPT_PATH) in seen["system"]
    assert "desk-side visit" not in seen["system"]
    assert seen["temperature"] == 0.0


def test_a_booking_reply_that_asks_again_is_replaced_by_the_written_row(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Pune office, 3rd floor, desk 14 on Thursday at 4 pm.": {
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

    def fake(prompt, **kwargs):
        return "The engineer will come to the Pune office on Thursday at 4 pm. Shall I book that visit?"

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session(
        "U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path)
    )
    session.ingest("Pune office, 3rd floor, desk 14 on Thursday at 4 pm.")
    session.decide("Pune office, 3rd floor, desk 14 on Thursday at 4 pm.")
    session.ingest("yes")
    said = session.reply("yes", session.decide("yes"))
    assert said == "Booked the visit at Pune office, 3rd floor, desk 14 on Thursday at 4 pm."
    assert "?" not in said
    session.ingest("yes")
    assert session.decide("yes") == {"action": ACTION_NONE}


def test_a_booking_reply_that_already_reports_the_row_is_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Pune office, Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {"location": "Pune office", "date": "Thursday", "time": "4 pm"},
                },
            }
        ),
    )

    def fake(prompt, **kwargs):
        return "Booked slot 1 at Pune office on Thursday at 4 pm."

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session(
        "U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path)
    )
    session.ingest("Pune office, Thursday at 4 pm.")
    session.decide("Pune office, Thursday at 4 pm.")
    session.ingest("yes")
    said = session.reply("yes", session.decide("yes"))
    assert said == "Booked slot 1 at Pune office on Thursday at 4 pm."


def test_ok_after_a_booking_is_thanks_not_nothing_waiting(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted(
            {
                "Pune office, Thursday at 4 pm.": {
                    "intent": "book_slot",
                    "slots": {"location": "Pune office", "date": "Thursday", "time": "4 pm"},
                },
            }
        ),
    )

    def fake(prompt, **kwargs):
        return prompt.splitlines()[-1]

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session(
        "U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path)
    )
    session.ingest("Pune office, Thursday at 4 pm.")
    session.decide("Pune office, Thursday at 4 pm.")
    session.ingest("yes")
    session.reply("yes", session.decide("yes"))

    session.ingest("ok")
    decision = session.decide("ok")
    assert decision == {"action": ACTION_NONE}
    said = session.reply("ok", decision)
    assert said == REPLIES["ack"]
    assert "nothing waiting" not in said.lower()
    assert _rows(tmp_path / "helpdesk.sqlite", "slots").__len__() == 1


def test_ok_with_nothing_booked_is_still_not_a_refusal(monkeypatch, tmp_path):
    def fake(prompt, **kwargs):
        return prompt.splitlines()[-1]

    monkeypatch.setattr("src.agent.chat", fake)
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    session.ingest("ok")
    said = session.reply("ok", session.decide("ok"))
    assert said == REPLIES["ack"]
    assert REPLIES["out_of_scope"] not in said


def test_trim_shows_only_the_last_four_turns(monkeypatch, tmp_path):
    """Turn 10's reply sees turns 6 to 9, and not turn 1.

    The window is attached to a tool reply. An open turn has a required
    line and is not shown the earlier turns, so it cannot prove the window.
    """
    monkeypatch.setattr(
        "src.agent.understand",
        lambda text, **kwargs: {"intent": "out_of_scope", "slots": {}, "source": "rule"},
    )
    monkeypatch.setattr("src.agent.chat", lambda prompt, **kwargs: "Noted.")
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path), strategy="trim")
    for number in range(1, 10):
        text = f"Message number {number} about nothing in particular."
        session.ingest(text)
        session.reply(text, {"action": "none"})
    seen = {}

    def capture(prompt, **kwargs):
        seen["prompt"] = prompt
        return "Noted."

    monkeypatch.setattr("src.agent.chat", capture)
    text = "Message number 10 about nothing in particular."
    session.ingest(text)
    session.reply(text, {"action": "tool", "name": "search_kb", "result": []})

    assert "Message number 1 " not in seen["prompt"]
    assert "Message number 5 " not in seen["prompt"]
    for number in (6, 7, 8, 9):
        assert f"Message number {number} " in seen["prompt"]
    # The current message is the question, not a line of earlier history.
    assert seen["prompt"].count("Message number 10 ") == 1


def test_full_shows_every_earlier_turn(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        lambda text, **kwargs: {"intent": "out_of_scope", "slots": {}, "source": "rule"},
    )
    monkeypatch.setattr("src.agent.chat", lambda prompt, **kwargs: "Noted.")
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path), strategy="full")
    for number in range(1, 6):
        text = f"Message number {number} about nothing in particular."
        session.ingest(text)
        session.reply(text, {"action": "none"})
    seen = {}
    monkeypatch.setattr(
        "src.agent.chat",
        lambda prompt, **kwargs: seen.setdefault("prompt", prompt) or "Noted.",
    )
    text = "Message number 6 about nothing in particular."
    session.ingest(text)
    session.reply(text, {"action": "tool", "name": "search_kb", "result": []})
    for number in range(1, 6):
        assert f"Message number {number} " in seen["prompt"]


def test_summary_quotes_the_notes_and_the_last_two_turns(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agent.understand",
        lambda text, **kwargs: {"intent": "out_of_scope", "slots": {}, "source": "rule"},
    )
    summarised = []

    def reply(prompt, **kwargs):
        return "Noted."

    def summarise(prompt, **kwargs):
        summarised.append(prompt)
        return "U1 mentioned the VPN once."

    monkeypatch.setattr("src.agent.chat", reply)
    monkeypatch.setattr("src.history.chat", summarise)
    session = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path), strategy="summary")
    for number in range(1, 5):
        text = f"Message number {number} about nothing in particular."
        session.ingest(text)
        session.reply(text, {"action": "none"})
    seen = {}

    def capture(prompt, **kwargs):
        seen["prompt"] = prompt
        return "Noted."

    monkeypatch.setattr("src.agent.chat", capture)
    text = "Message number 5 about nothing in particular."
    session.ingest(text)
    session.reply(text, {"action": "tool", "name": "search_kb", "result": []})

    assert "Notes so far:" in seen["prompt"]
    assert "U1 mentioned the VPN once." in seen["prompt"]
    assert "Message number 3 " in seen["prompt"]
    assert "Message number 4 " in seen["prompt"]
    # Turn 1 has fallen out of the quoted window. It was summarised once,
    # on the turn it first fell out, and not again on the next turn.
    assert "User: Message number 1 " not in seen["prompt"]
    # Turn 5's reply is a tool call, so it builds no summary of its own.
    # The one summarise call was made when turn 1 first fell out of the window.
    assert len(summarised) == 1
    assert "Message number 1 " in summarised[0]


def test_an_unknown_strategy_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        Session("U1", "A", memory=Memory("U1", "A", root=tmp_path), strategy="window")


def test_the_same_time_reuses_the_booked_time(monkeypatch, tmp_path):
    """'The same time is fine' is not a yes. It fills the time from the visit."""
    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({"And for Friday?": {"intent": "book_slot", "slots": {"date": "Friday"}}}),
    )
    db_path = tmp_path / "helpdesk.sqlite"
    session = Session("U1", "A", db_path=db_path, memory=Memory("U1", "A", root=tmp_path))
    session.slots["location"] = "Pune office, 3rd floor, desk 14"
    session.last_booked = {
        "location": "Pune office, 3rd floor, desk 14",
        "date": "Thursday",
        "time": "4 pm",
    }
    session.ingest("And for Friday?")
    asked = session.decide("And for Friday?")
    assert asked["slot"] == "time"

    session.ingest("The same time is fine.")
    decision = session.decide("The same time is fine.")
    assert decision["action"] == ACTION_ASK
    assert session.slots["time"] == "4 pm"
    assert session.slots["date"] == "Friday"
    assert _rows(db_path, "slots") == []


def test_closing_the_session_stores_the_models_reflection(monkeypatch, tmp_path):
    """End with no typed note. The model writes one, and Session B reads it."""
    def fake_chat(prompt, **kwargs):
        assert kwargs["call_type"] == "reflect"
        return "Session A booked a Thursday visit for the VPN fault."

    monkeypatch.setattr("src.memory.chat", fake_chat)
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)

    first = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    first.history.append({"text": "Yes, book it.", "bot": "Booked Thursday at 4 pm."})
    written = first.end()
    assert written["type"] == "reflection"
    assert "Thursday" in written["text"]

    later = Session("U1", "B", memory=Memory("U1", "B", root=tmp_path))
    hits = later.memory.episodic.recall("U1", "what did I book last time")
    assert hits
    assert hits[0]["type"] == "reflection"
    assert hits[0]["session"] == "A"


def test_a_booked_visit_logs_its_importance(monkeypatch, tmp_path):
    import json

    from src.config import LOG_PATH

    monkeypatch.setattr(
        "src.agent.understand",
        _scripted({
            "Please raise the VPN failure.": {
                "intent": "fix_issue",
                "slots": {
                    "issue_type": "VPN failure",
                    "location": "Pune office",
                    "date": "Thursday",
                    "time": "4 pm",
                },
            },
        }),
    )
    before = len(LOG_PATH.read_text(encoding="utf-8").splitlines()) if LOG_PATH.exists() else 0
    session = Session("U1", "A", db_path=tmp_path / "helpdesk.sqlite", memory=Memory("U1", "A", root=tmp_path))
    session.run_id = "imp-test"
    text = "Please raise the VPN failure."
    session.ingest(text)
    session.decide(text)
    session.ingest("yes")
    session.decide("yes")

    added = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()[before:]]
    scores = [row for row in added if row["call_type"] == "importance"]
    assert {row["episode_type"] for row in scores} == {"ticket", "booking"}
    assert {row["importance"] for row in scores} == {8, 7}
    assert all(row["run_id"] == "imp-test" and row["config"] == "full" for row in scores)


def test_please_forget_deletes_the_phone_before_any_model_call(monkeypatch, tmp_path):
    """Turn 22. The sentence must never reach understand().

    If it did, the model could label it provide_info and store the number
    again. The reply is the fixed line, and a later session finds nothing.
    """
    def boom(*args, **kwargs):
        raise AssertionError("understand() must not be called for a forget")

    monkeypatch.setattr("src.agent.understand", boom)
    monkeypatch.setattr("src.semantic_memory.embed", _fake_embed)
    monkeypatch.setattr("src.episodic_memory.embed", _fake_embed)

    first = Session("U1", "A", memory=Memory("U1", "A", root=tmp_path))
    first.memory.record_turn(
        "If the desk has to reach me, my personal number is 98220 11435.",
        "provide_info",
        {},
        turn=7,
    )
    first.memory.semantic.remember("U1", "office", "Hyderabad", turn=20)

    text = "Please forget my personal phone number. Don't keep 98220 11435."
    seen = first.ingest(text)
    assert seen["intent"] == "forget"
    assert seen["source"] == "rule"
    assert seen["forgotten"]["key"] == "personal_phone"
    assert seen["forgotten"]["facts"] >= 1
    assert first.decide(text) == {"action": ACTION_NONE}
    assert first.reply(text, {"action": ACTION_NONE}) == "I've forgotten that."
    assert "98220 11435" not in first.memory.semantic.path.read_text(encoding="utf-8")

    later = Session("U1", "B", memory=Memory("U1", "B", root=tmp_path))
    assert later.memory.semantic.recall("U1", "What number did I give you for my personal phone?") == []
    # The office was not part of the request, so it is still there.
    assert any(row["key"] == "office" for row in later.memory.semantic.active("U1"))


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
    assert said.startswith("You sit at Pune office, 3rd floor, desk 14.")
    assert "[mem:" in said

    # A different user reads the same folder and finds nothing of U1's.
    other = Session("U2", "B", memory=Memory("U2", "B", root=tmp_path))
    theirs = other.ingest(text)
    assert theirs["recalled"]["facts"] == []
    assert theirs["recalled"]["episodes"] == []
    assert other.reply(text, other.decide(text)) == FALLBACK

