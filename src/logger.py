"""Append-only call log.

Every LLM, embedding and tool call writes exactly one JSON line to
logs/calls.jsonl. This module only ever opens that file in append mode.
A failed call must not be logged: callers log after the call succeeds.
"""

from __future__ import annotations

import datetime
import json
import time
from typing import Any

from src.config import CALL_TYPES, CONFIGS, LOG_PATH, RUN_ID

# Fields the brief's sample line and TESTING.md T1.5 both require.
REQUIRED_FIELDS = (
    "ts",
    "run_id",
    "config",
    "user",
    "session",
    "turn",
    "call_type",
    "model",
    "input_tokens",
    "output_tokens",
    "retrieved_tokens",
    "latency_ms",
)


def log_call(
    call_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    start: float,
    retrieved_tokens: int = 0,
    config: str | None = None,
    user: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one line to logs/calls.jsonl.

    `start` is time.time() taken just before the call. Token counts must
    come from the API usage field when the backend reports them.
    Returns the record that was written.

    call_type must be one of CALL_TYPES. The names the old log lines use
    ("reply", "summarise", "embedding") are refused, so a new line cannot
    be written in the old vocabulary. config, when given, must be one of
    CONFIGS. Passing None is still allowed: a call made outside a measured
    run (a smoke test, a one-off search) has no configuration to name.
    """
    if call_type not in CALL_TYPES:
        raise ValueError(f"call_type must be one of {CALL_TYPES}, got {call_type!r}")
    if config is not None and config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}, got {config!r}")

    record: dict[str, Any] = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "run_id": run_id if run_id is not None else RUN_ID,
        "config": config,
        "user": user,
        "session": session,
        "turn": turn,
        "call_type": call_type,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "retrieved_tokens": int(retrieved_tokens),
        "latency_ms": int((time.time() - start) * 1000),
    }
    if extra:
        # Extra keys never replace a required field.
        for key, value in extra.items():
            if key not in record:
                record[key] = value

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    # Append only. Never open with "w", "r+" or seek back to edit a line.
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return record
