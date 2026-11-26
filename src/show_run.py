"""Print a finished run as a conversation, with the expect/actual verdict per turn.

    python -m src.show_run                      # hybrid_mem, every turn
    python -m src.show_run --failed             # only turns that missed
    python -m src.show_run --turn 24            # one turn in full
    python -m src.show_run --config no_rag
"""

from __future__ import annotations

import argparse
import json

from src.config import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a recorded run as a conversation")
    parser.add_argument("--config", default="hybrid_mem")
    parser.add_argument("--failed", action="store_true")
    parser.add_argument("--turn", type=int)
    args = parser.parse_args()

    path = ROOT / "results" / "runs" / args.config / "transcript.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    for session in run["sessions"]:
        print(f"\n===== Session {session['session']} =====")
        for turn in session["turns"]:
            if args.turn is not None and turn["turn"] != args.turn:
                continue
            if args.failed and turn.get("passed", True):
                continue
            mark = "ok  " if turn.get("passed", True) else "MISS"
            print(f"\n[{mark}] turn {turn['turn']} ({turn['user']})")
            print(f"  user: {turn['text']}")
            print(f"  bot : {turn['bot']}")
            for miss in turn.get("misses", []):
                print(f"    - {miss['field']}: expected {miss['expected']!r}, got {miss['actual']!r}")
            if args.turn is not None:
                print(json.dumps(turn["actual"], indent=2, ensure_ascii=False))
        if args.turn is None:
            for probe in session["probes"]:
                mark = "ok  " if probe["passed"] else "MISS"
                print(f"\n[{mark}] probe {probe['id']} ({probe['user']}): {probe['question']}")
                print(f"  expected: {probe['expected']!r}")
                print(f"  reply   : {probe['reply']}")


if __name__ == "__main__":
    main()
