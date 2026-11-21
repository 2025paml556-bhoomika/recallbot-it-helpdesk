"""SQLite schema for the helpdesk.

Three tables, each with an owner_user column so a later tool can refuse a
row that belongs to someone else:

    tickets  a raised fault (ERR-VPN-809 and the like)
    slots    a desk-side visit, written only after the user confirms
    assets   a laptop or other device tied to one user

This module creates the file and the tables. It does not insert rows;
booking and ticket tools do that in the next step.

    python -m src.db
"""

from __future__ import annotations

import sqlite3

from src.config import DB_PATH

# Columns every table shares. owner_user is what the isolation check reads.
_OWNER = "owner_user TEXT NOT NULL"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id   TEXT PRIMARY KEY,
    {owner},
    asset_id    TEXT,
    issue_type  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slots (
    slot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    {owner},
    ticket_id   TEXT,
    asset_id    TEXT,
    issue_type  TEXT NOT NULL,
    location    TEXT NOT NULL,
    date        TEXT NOT NULL,
    time        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'booked',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id    TEXT PRIMARY KEY,
    {owner},
    model       TEXT,
    location    TEXT,
    status      TEXT NOT NULL DEFAULT 'active'
);
""".format(owner=_OWNER)

TABLES = ("tickets", "slots", "assets")


def connect(path=DB_PATH) -> sqlite3.Connection:
    """Open the database, creating the file and the three tables if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row["name"] for row in rows]


def init_db(path=DB_PATH) -> None:
    """Create db/helpdesk.sqlite and leave it empty."""
    conn = connect(path)
    try:
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    init_db()
    conn = connect()
    try:
        print(f"database: {DB_PATH}")
        for table in TABLES:
            columns = table_columns(conn, table)
            count = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            print(f"  {table}: {count} rows, columns={', '.join(columns)}")
            if "owner_user" not in columns:
                raise SystemExit(f"{table} is missing owner_user")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
