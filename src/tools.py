"""Helpdesk tools: search the knowledge base, read a ticket, book a visit.

get_ticket returns a ticket only when owner_user matches the caller. A ticket
that belongs to someone else is reported as "not found", the same answer as a
ticket that does not exist, so one user cannot tell the other's tickets apart.

book_it_slot writes a row only when confirmed is true. The dialogue manager
sets that flag after the user says yes; calling the tool earlier inserts nothing.

Every successful tool call appends one line to logs/calls.jsonl with
input_tokens and output_tokens set to 0. A refused call (missing confirmation,
missing ticket, wrong owner) writes nothing.

    python -m src.tools search "What does ERR-VPN-809 mean?"
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from src.config import DB_PATH
from src.db import connect
from src.logger import log_call
from src.search_kb import search_kb as search_kb_index

NOT_FOUND = "not found"

# What a tool call records. Tools do not spend model tokens.
TOOL_MODEL = "tool"
TOOL_INPUT_TOKENS = 0
TOOL_OUTPUT_TOKENS = 0


def _log_tool(
    name: str,
    start: float,
    *,
    config: str | None = None,
    user: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"tool": name}
    if extra:
        payload.update(extra)
    log_call(
        call_type="tool",
        model=TOOL_MODEL,
        input_tokens=TOOL_INPUT_TOKENS,
        output_tokens=TOOL_OUTPUT_TOKENS,
        start=start,
        config=config,
        user=user,
        session=session,
        turn=turn,
        run_id=run_id,
        extra=payload,
    )


def search_kb(
    query: str,
    *,
    mode: str = "hybrid",
    top_k: int = 5,
    size: int = 128,
    config: str | None = None,
    user: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
) -> list[dict]:
    """Top chunks for a query. The knowledge base is shared across users.

    Returns the same hit dicts as src.search_kb.search_kb. Logs one tool call
    with 0 tokens after the search succeeds.
    """
    start = time.time()
    hits = search_kb_index(query, mode=mode, top_k=top_k, size=size)
    _log_tool(
        "search_kb",
        start,
        config=config,
        user=user,
        session=session,
        turn=turn,
        run_id=run_id,
        extra={"query": query, "mode": mode, "size": size, "n_hits": len(hits)},
    )
    return hits


def get_ticket(
    ticket_id: str,
    user: str,
    *,
    config: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
    db_path=DB_PATH,
) -> dict | str:
    """The ticket if it belongs to `user`, otherwise "not found".

    A missing id and a ticket owned by someone else both return "not found".
    Only a ticket that is returned is logged; a refusal leaves the log unchanged.
    """
    start = time.time()
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ? AND owner_user = ?",
            (ticket_id, user),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return NOT_FOUND

    ticket = dict(row)
    _log_tool(
        "get_ticket",
        start,
        config=config,
        user=user,
        session=session,
        turn=turn,
        run_id=run_id,
        extra={"ticket_id": ticket_id},
    )
    return ticket


def latest_ticket(
    user: str,
    *,
    config: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
    db_path=DB_PATH,
) -> dict | str:
    """This user's newest ticket, or "not found" when they have none.

    "What was the last issue I raised?" names no ticket id. The newest row
    owned by `user` is the one that answers it. Another user's tickets are
    not visible. A user with no tickets returns "not found" and logs nothing.
    """
    start = time.time()
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT * FROM tickets
            WHERE owner_user = ?
            ORDER BY created_at DESC, ticket_id DESC
            LIMIT 1
            """,
            (user,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return NOT_FOUND

    ticket = dict(row)
    _log_tool(
        "get_ticket",
        start,
        config=config,
        user=user,
        session=session,
        turn=turn,
        run_id=run_id,
        extra={"ticket_id": ticket["ticket_id"], "latest": True},
    )
    return ticket


def latest_booking(
    user: str,
    *,
    config: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
    db_path=DB_PATH,
) -> dict | str:
    """This user's newest booked visit, or "not found" when they have none.

    A booking question is not answered with a ticket row. The newest slots
    row owned by `user` is the visit. Another user's visits are not visible.
    """
    start = time.time()
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT * FROM slots
            WHERE owner_user = ?
            ORDER BY created_at DESC, slot_id DESC
            LIMIT 1
            """,
            (user,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return NOT_FOUND

    booked = dict(row)
    _log_tool(
        "get_booking",
        start,
        config=config,
        user=user,
        session=session,
        turn=turn,
        run_id=run_id,
        extra={"slot_id": booked["slot_id"], "latest": True},
    )
    return booked


def book_it_slot(
    user: str,
    issue_type: str,
    location: str,
    date: str,
    time_of_day: str,
    *,
    confirmed: bool = False,
    ticket_id: str | None = None,
    asset_id: str | None = None,
    config: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
    db_path=DB_PATH,
) -> dict | None:
    """Insert one slots row, but only after the user has confirmed.

    confirmed=False inserts nothing and logs nothing, and returns None.
    confirmed=True inserts one row owned by `user` and logs the tool call
    with 0 tokens. Returns the inserted row.
    """
    if not confirmed:
        return None

    start = time.time()
    conn = connect(db_path)
    try:
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        cursor = conn.execute(
            """
            INSERT INTO slots (
                owner_user, ticket_id, asset_id, issue_type,
                location, date, time, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'booked', ?)
            """,
            (
                user,
                ticket_id,
                asset_id,
                issue_type,
                location,
                date,
                time_of_day,
                created_at,
            ),
        )
        conn.commit()
        slot_id = cursor.lastrowid
        row = conn.execute(
            "SELECT * FROM slots WHERE slot_id = ?",
            (slot_id,),
        ).fetchone()
        booked = dict(row)
    finally:
        conn.close()

    _log_tool(
        "book_it_slot",
        start,
        config=config,
        user=user,
        session=session,
        turn=turn,
        run_id=run_id,
        extra={"slot_id": booked["slot_id"], "confirmed": True},
    )
    return booked


def update_slot(
    user: str,
    slot_id: int,
    *,
    location: str | None = None,
    date: str | None = None,
    time_of_day: str | None = None,
    config: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
    db_path=DB_PATH,
) -> dict | str:
    """Change one of this user's visits. A missing or other-owned row is "not found".

    Only the fields the caller passes are written. The row stays the same
    slot, so a time change does not create a second booking.
    """
    start = time.time()
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM slots WHERE slot_id = ? AND owner_user = ?",
            (slot_id, user),
        ).fetchone()
        if row is None:
            return NOT_FOUND
        current = dict(row)
        new_location = location if location is not None else current["location"]
        new_date = date if date is not None else current["date"]
        new_time = time_of_day if time_of_day is not None else current["time"]
        conn.execute(
            """
            UPDATE slots
            SET location = ?, date = ?, time = ?
            WHERE slot_id = ? AND owner_user = ?
            """,
            (new_location, new_date, new_time, slot_id, user),
        )
        conn.commit()
        updated = dict(
            conn.execute(
                "SELECT * FROM slots WHERE slot_id = ?",
                (slot_id,),
            ).fetchone()
        )
    finally:
        conn.close()

    _log_tool(
        "update_slot",
        start,
        config=config,
        user=user,
        session=session,
        turn=turn,
        run_id=run_id,
        extra={"slot_id": slot_id, "confirmed": True},
    )
    return updated


def raise_ticket(
    user: str,
    issue_type: str,
    *,
    ticket_id: str | None = None,
    asset_id: str | None = None,
    config: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
    db_path=DB_PATH,
) -> dict:
    """Insert one ticket owned by `user`. Used by tests and the dialogue loop.

    ticket_id is generated as TCK-<n> when the caller does not supply one.
    """
    start = time.time()
    conn = connect(db_path)
    try:
        if ticket_id is None:
            count = conn.execute("SELECT COUNT(*) AS n FROM tickets").fetchone()["n"]
            ticket_id = f"TCK-{count + 1:04d}"
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        conn.execute(
            """
            INSERT INTO tickets (
                ticket_id, owner_user, asset_id, issue_type, status, created_at
            )
            VALUES (?, ?, ?, ?, 'open', ?)
            """,
            (ticket_id, user, asset_id, issue_type, created_at),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()
        ticket = dict(row)
    finally:
        conn.close()

    _log_tool(
        "raise_ticket",
        start,
        config=config,
        user=user,
        session=session,
        turn=turn,
        run_id=run_id,
        extra={"ticket_id": ticket["ticket_id"]},
    )
    return ticket


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one helpdesk tool")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="search the knowledge base")
    search.add_argument("query")
    search.add_argument("--mode", default="hybrid")
    search.add_argument("--size", type=int, default=128)

    args = parser.parse_args()
    if args.command == "search":
        hits = search_kb(args.query, mode=args.mode, size=args.size)
        if not hits:
            print("(no hits)")
            return
        for hit in hits:
            print(f"{hit['rank']}. {hit['doc_id']}  {hit['score']:.5f}  {hit['chunk_id']}")


if __name__ == "__main__":
    main()
