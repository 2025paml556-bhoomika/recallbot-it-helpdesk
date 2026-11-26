"""Record what a turn actually did, and compare it with the turn's `expect`.

The check runs after the turn is finished. It reads the Session and the
reply, builds one `actual` dict, and returns the list of fields that differ.
A turn with no `expect` block has nothing to compare and passes.

How the contract is read:
  - a list means any one of the values is acceptable
        (intent, tool, citation); inside reply_contains an inner list is
        any-of and each outer item is required
  - null (or a missing key) means the field is not checked
  - citation "none" means the reply carries no [..] citation
  - session_slots and db_rows are subset matches, extra keys are ignored
"""

from __future__ import annotations

from src.agent import ACTION_ASK, ACTION_TOOL, FALLBACK, _CITATION
from src.db import connect
from src.run_history import UNKNOWN


def capture(session, decision: dict, reply: str) -> dict:
    """What happened on the turn that just finished."""
    latest = session.history[-1] if session.history else {}
    is_tool = decision.get("action") == ACTION_TOOL
    return {
        "intent": latest.get("intent"),
        "action": decision.get("action"),
        "ask_slot": decision.get("slot") if decision.get("action") == ACTION_ASK else None,
        "tool": decision.get("name") if is_tool else None,
        "reply": reply,
        "reply_is_unknown": _is_unknown(reply),
        "citations": _CITATION.findall(reply or ""),
        "facts_written": len(session.facts_written or []),
        "state": {
            "pending_booking": session.pending_booking,
            "awaiting_confirmation": session.awaiting_confirmation,
        },
        "session_slots": dict(session.slots),
        "db": _db_snapshot(session._db_path()),
    }


def compare(expect: dict, actual: dict) -> list[dict]:
    """One entry per field that differs: {field, expected, actual}."""
    misses: list[dict] = []

    def miss(field, expected, got):
        misses.append({"field": field, "expected": expected, "actual": got})

    for field in ("intent", "tool"):
        if expect.get(field) is not None and not _any_of(expect[field], actual[field]):
            miss(field, expect[field], actual[field])
    if expect.get("action") is not None and expect["action"] != actual["action"]:
        miss("action", expect["action"], actual["action"])
    # ask_slot null on an ask means "no particular slot", so it is checked
    # only when the contract names one.
    if expect.get("ask_slot") is not None and expect["ask_slot"] != actual["ask_slot"]:
        miss("ask_slot", expect["ask_slot"], actual["ask_slot"])

    reply = (actual["reply"] or "").lower()
    for want in expect.get("reply_contains", []):
        options = [want] if isinstance(want, str) else want
        if not any(opt.lower() in reply for opt in options):
            miss("reply_contains", want, actual["reply"])
    for bad in expect.get("reply_not_contains", []):
        if bad.lower() in reply:
            miss("reply_not_contains", bad, actual["reply"])
    if "reply_is_unknown" in expect and expect["reply_is_unknown"] != actual["reply_is_unknown"]:
        miss("reply_is_unknown", expect["reply_is_unknown"], actual["reply_is_unknown"])

    if "citation" in expect and expect["citation"] is not None:
        if not _citation_ok(expect["citation"], actual["citations"]):
            miss("citation", expect["citation"], actual["citations"])
    if expect.get("facts_written") is not None and expect["facts_written"] != actual["facts_written"]:
        miss("facts_written", expect["facts_written"], actual["facts_written"])

    for key, want in (expect.get("state") or {}).items():
        if actual["state"].get(key) != want:
            miss(f"state.{key}", want, actual["state"].get(key))
    for key, want in (expect.get("session_slots") or {}).items():
        got = actual["session_slots"].get(key)
        if (got or "").strip().lower() != str(want).strip().lower():
            miss(f"session_slots.{key}", want, got)
    for table, want in (expect.get("db") or {}).items():
        got = len(actual["db"].get(table, []))
        if got != want:
            miss(f"db.{table}", want, got)
    for table, wanted_rows in (expect.get("db_rows") or {}).items():
        rows = actual["db"].get(table, [])
        for wanted in wanted_rows:
            if not any(_row_matches(wanted, row) for row in rows):
                miss(f"db_rows.{table}", wanted, rows)
    return misses


def check_turn(turn: dict, session, decision: dict, reply: str) -> dict:
    """The record stored on the transcript for one turn."""
    actual = capture(session, decision, reply)
    expect = turn.get("expect") or {}
    misses = compare(expect, actual)
    return {"expect": expect, "actual": actual, "misses": misses, "passed": not misses}


def _any_of(expected, got) -> bool:
    return got in expected if isinstance(expected, list) else got == expected


def _is_unknown(reply: str) -> bool:
    text = reply or ""
    return text.strip() == FALLBACK or UNKNOWN.search(text) is not None


def _citation_ok(expected, citations: list[str]) -> bool:
    if expected == "none":
        return not citations
    options = expected if isinstance(expected, list) else [expected]
    # "mem" is any [mem:..] token; a KB name is matched inside [doc#chunk].
    for option in options:
        needle = "mem:" if option == "mem" else option
        if any(needle in cite for cite in citations):
            return True
    return False


def _db_snapshot(db_path) -> dict:
    """Every ticket and booking row, read after the turn."""
    conn = connect(db_path)
    try:
        tickets = [dict(r) for r in conn.execute("SELECT * FROM tickets ORDER BY ticket_id")]
        bookings = [dict(r) for r in conn.execute("SELECT * FROM slots ORDER BY slot_id")]
    finally:
        conn.close()
    for row in tickets + bookings:
        row["user"] = row.get("owner_user")
    return {"tickets": tickets, "bookings": bookings}


def _row_matches(wanted: dict, row: dict) -> bool:
    for key, value in wanted.items():
        if key == "issue_contains":
            if str(value).lower() not in str(row.get("issue_type", "")).lower():
                return False
        elif str(row.get(key)).lower() != str(value).lower():
            return False
    return True
