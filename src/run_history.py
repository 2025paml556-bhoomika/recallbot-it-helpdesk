"""Run Session A once under each short-term memory strategy.

Each strategy gets its own run_id, stm-<strategy>, so the three runs can be
told apart in logs/calls.jsonl. The state tracker stays on: it is the same
Session either way. Memory is written to a folder for this run, not to the
shared memory/, so a measurement run does not change what the page recalls.

    python -m src.run_history
    python -m src.run_history --strategy trim
"""

from __future__ import annotations

import argparse
import json
import re

from src.agent import Session
from src.config import ROOT
from src.history import STRATEGIES
from src.memory import Memory

CONVERSATION_PATH = ROOT / "data" / "test_conversation.json"
PROBES_PATH = ROOT / "data" / "probes.json"
RUN_ROOT = ROOT / "results" / "history_runs"

# A probe passes when the expected phrase appears. A forgotten fact passes
# when the reply says it is not known, and fails when the value comes back.
UNKNOWN = re.compile(r"don'?t know|don'?t have|no record|not (?:have|know)|cannot find", re.IGNORECASE)

# Session A probes the short-term window can actually answer. A fact asked
# in Session B is long-term memory, which is a later step.
SESSION = "A"


def load_turns(session: str = SESSION) -> list[dict]:
    payload = json.loads(CONVERSATION_PATH.read_text(encoding="utf-8"))
    return [turn for turn in payload["turns"] if turn["session"] == session]


def load_probes(session: str = SESSION) -> list[dict]:
    payload = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
    return [probe for probe in payload["probes"] if probe["session"] == session]


def run_strategy(strategy: str, turns: list[dict], probes: list[dict]) -> dict:
    """Play Session A, then ask each probe against the state it left."""
    root = RUN_ROOT / strategy
    run_id = f"stm-{strategy}"
    replies = []
    session = Session(
        "U1",
        "A",
        db_path=root / "helpdesk.sqlite",
        memory=Memory("U1", "A", root=root / "memory"),
        strategy=strategy,
    )
    session.run_id = run_id
    for turn in turns:
        text = turn["text"]
        session.ingest(text)
        decision = session.decide(text)
        said = session.reply(text, decision)
        replies.append({"turn": turn["turn"], "user": text, "bot": said, "action": decision.get("action")})

    asked = []
    for probe in probes:
        text = probe["question"]
        session.ingest(text)
        decision = session.decide(text)
        said = session.reply(text, decision)
        asked.append(
            {
                "id": probe["id"],
                "category": probe["category"],
                "question": text,
                "expected": probe["expected"],
                "evidence_turn": probe.get("evidence_turn"),
                "reply": said,
                "passed": _passed(probe, said),
            }
        )
    transcript = {"strategy": strategy, "run_id": run_id, "turns": replies, "probes": asked}
    root.mkdir(parents=True, exist_ok=True)
    (root / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return transcript


def _passed(probe: dict, reply: str) -> bool:
    """True when the reply contains the expected phrase.

    Category F is a forgotten fact: the reply passes by not repeating it.
    """
    expected = str(probe.get("expected") or "")
    if probe.get("category") == "F":
        return UNKNOWN.search(reply or "") is not None and expected.lower() not in (reply or "").lower()
    return expected.lower() in (reply or "").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Session A under one or all history strategies")
    parser.add_argument("--strategy", choices=STRATEGIES)
    args = parser.parse_args()
    turns = load_turns()
    probes = load_probes()
    chosen = (args.strategy,) if args.strategy else STRATEGIES
    for strategy in chosen:
        transcript = run_strategy(strategy, turns, probes)
        passed = sum(1 for probe in transcript["probes"] if probe["passed"])
        print(f"{strategy}: {len(transcript['turns'])} turns, probes {passed}/{len(transcript['probes'])}")


if __name__ == "__main__":
    main()
