"""Tool checks for the dialogue pipeline: search, ticket isolation, booking.

get_ticket as U2 on U1's ticket returns "not found". book_it_slot writes no
row until confirmed=True, then exactly one. Successful tool calls log
call_type "tool" with 0 input and 0 output tokens. A refused call logs nothing.
"""

from __future__ import annotations

import json

import pytest

from src.config import LOG_PATH
from src.db import connect
from src.tools import NOT_FOUND, book_it_slot, get_ticket, raise_ticket, search_kb


def _log_lines() -> list[str]:
    if not LOG_PATH.exists():
        return []
    return LOG_PATH.read_text(encoding="utf-8").splitlines()


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "helpdesk.sqlite"


def test_search_kb_returns_hits_and_logs_zero_tokens():
    before = _log_lines()
    hits = search_kb("What does ERR-VPN-809 mean?", user="U1", turn=1)
    assert hits
    assert hits[0]["doc_id"] == "KB-VPN"

    # The embedder logs its own embedding line. The tool adds exactly one more,
    # and that line is the one with 0 tokens.
    added = [json.loads(line) for line in _log_lines()[len(before):]]
    tool_lines = [row for row in added if row["call_type"] == "tool"]
    assert len(tool_lines) == 1
    new = tool_lines[0]
    assert new["model"] == "tool"
    assert new["tool"] == "search_kb"
    assert new["input_tokens"] == 0
    assert new["output_tokens"] == 0
    assert new["user"] == "U1"


def test_get_ticket_returns_only_the_owners_row(db_path):
    raised = raise_ticket(
        "U1", "VPN failure ERR-VPN-809", ticket_id="TCK-1001", db_path=db_path
    )
    before = _log_lines()

    own = get_ticket("TCK-1001", "U1", db_path=db_path)
    assert own["ticket_id"] == "TCK-1001"
    assert own["owner_user"] == "U1"
    assert own["issue_type"] == raised["issue_type"]

    other = get_ticket("TCK-1001", "U2", db_path=db_path)
    missing = get_ticket("TCK-9999", "U1", db_path=db_path)
    assert other == NOT_FOUND
    assert missing == NOT_FOUND

    # The owner's read is logged. The two refusals are not.
    after = _log_lines()
    assert len(after) == len(before) + 1
    logged = json.loads(after[-1])
    assert logged["call_type"] == "tool"
    assert logged["tool"] == "get_ticket"
    assert logged["input_tokens"] == 0
    assert logged["output_tokens"] == 0
    assert logged["user"] == "U1"


def test_book_it_slot_writes_nothing_until_confirmed(db_path):
    before = _log_lines()
    fields = dict(
        user="U1",
        issue_type="VPN failure ERR-VPN-809",
        location="Pune office, 3rd floor, desk 14",
        date="Thursday",
        time_of_day="4 pm",
        db_path=db_path,
    )

    refused = book_it_slot(**fields, confirmed=False)
    assert refused is None
    conn = connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM slots").fetchone()["n"] == 0
    finally:
        conn.close()
    assert _log_lines() == before

    booked = book_it_slot(**fields, confirmed=True)
    assert booked is not None
    assert booked["owner_user"] == "U1"
    assert booked["date"] == "Thursday"
    assert booked["time"] == "4 pm"
    assert booked["status"] == "booked"

    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM slots").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["slot_id"] == booked["slot_id"]

    logged = json.loads(_log_lines()[-1])
    assert len(_log_lines()) == len(before) + 1
    assert logged["call_type"] == "tool"
    assert logged["tool"] == "book_it_slot"
    assert logged["input_tokens"] == 0
    assert logged["output_tokens"] == 0
    assert logged["confirmed"] is True
