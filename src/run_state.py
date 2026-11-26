"""What is on disk after one step of a run, for the step-by-step replay.

run.py calls snapshot() after every turn, probe and session close, and stores
the result on the transcript. The page then shows the state as it stood at the
step the reader has reached, without playing the model again.

The files hold every user, so rows keep their user and the page picks the one
it is showing. Embeddings are dropped: they are long and are not facts.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.db import connect


def snapshot(memory_root, db_path) -> dict:
    """Facts, episodes, tickets and bookings as they are right now."""
    root = Path(memory_root)
    return {
        "facts": _read_jsonl(root / "semantic.jsonl"),
        "episodes": [
            {key: value for key, value in row.items() if key != "embedding"}
            for row in _read_jsonl(root / "episodic.jsonl")
        ],
        "tickets": _table(db_path, "tickets"),
        "bookings": _table(db_path, "slots"),
    }


def recalled_view(recalled: dict) -> dict:
    """What load() returned for a turn, cut down to what the page shows."""
    return {
        "facts": [
            {"key": row.get("key"), "value": row.get("value"), "score": round(row.get("score", 0), 3)}
            for row in recalled.get("facts", [])
        ],
        "episodes": [
            {"text": row.get("text"), "type": row.get("type"), "score": round(row.get("score", 0), 3)}
            for row in recalled.get("episodes", [])
        ],
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _table(db_path, name: str) -> list[dict]:
    if not Path(db_path).exists():
        return []
    conn = connect(Path(db_path))
    try:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {name}").fetchall()]
    finally:
        conn.close()
