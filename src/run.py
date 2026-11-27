"""Run Sessions A and B under one ablation configuration.

Session A is played, the session is closed, and then the process stops.
Session B is a new process: it builds a fresh Session against the same
memory folder and the same database, so anything it recalls has to have
been read back from disk. Nothing is kept in memory between the two.

    python -m src.run --config hybrid_mem
    python -m src.run --config no_rag --session B

Each configuration writes under results/runs/<config>/, with its own
database and its own memory folder, so one run cannot change another's
answers. The log lines carry the config name.
"""

from __future__ import annotations

import argparse
import json
import shutil

from src.agent import Session
from src.config import ABLATION_CONFIGS, ROOT, RUN_ID
from src.memory import Memory
from src.run_history import _passed, load_probes, load_turns
from src.run_state import recalled_view, snapshot
from src.turn_check import check_turn

RUNS_ROOT = ROOT / "results" / "runs"

# The history strategy the ablation holds fixed. The thing being varied is
# which memory type is read, not how the window is cut.
STRATEGY = "trim"


def run_config(config: str, *, sessions: tuple[str, ...] = ("A", "B"), raw: bool = False) -> dict:
    """Play the named sessions in order, each in its own Session object.

    A and B share the folder. They share nothing else: B's Session is built
    after A's has been closed and dropped, which is as close as one process
    gets to a restart. Run with --session A and then --session B to restart
    for real.
    """
    if config not in ABLATION_CONFIGS:
        raise ValueError(f"config must be one of {ABLATION_CONFIGS}, got {config!r}")

    root = RUNS_ROOT / config
    memory_root = root / "memory"
    db_path = root / "helpdesk.sqlite"
    # RUN_ID changes per run (RECALLBOT_RUN_ID), so a rerun under another
    # model never shares an id with the lines already in the call log.
    run_id = f"abl-{config}-{RUN_ID}"

    # A run that starts at Session A starts from nothing. Without this a
    # second run would begin with the facts and bookings of the first, and
    # the step-by-step replay could not start from an empty state. Running
    # Session B alone keeps what A left behind, which is the point of it.
    if "A" in sessions:
        shutil.rmtree(memory_root, ignore_errors=True)
        db_path.unlink(missing_ok=True)

    transcript = {
        "config": config,
        "run_id": run_id,
        # Step 0 of the replay: nothing stored yet.
        "initial_state": snapshot(memory_root, db_path),
        "sessions": [],
    }
    for name in sessions:
        played = _play_session(config, name, memory_root, db_path, run_id, raw)
        transcript["sessions"].append(played)

    root.mkdir(parents=True, exist_ok=True)
    (root / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return transcript


def _play_session(config: str, name: str, memory_root, db_path, run_id: str, raw: bool = False) -> dict:
    """One session, from a Session built fresh against the folder on disk."""
    turns = load_turns(name)
    probes = load_probes(name)
    replies = []
    # One user at a time. U2's turns are a separate session, so they cannot
    # see the slots U1 gathered. Both point at the same folder and database.
    open_sessions: dict[tuple[str, str], Session] = {}

    def get(user: str) -> Session:
        key = (user, name)
        if key not in open_sessions:
            session = Session(
                user,
                name,
                db_path=db_path,
                memory=Memory(user, name, root=memory_root),
                strategy=STRATEGY,
                config=config,
            )
            session.run_id = run_id
            open_sessions[key] = session
        return open_sessions[key]

    for turn in turns:
        session = get(turn["user"])
        text = turn["text"]
        labelled = session.ingest(text)
        decision = session.decide(text)
        said = session.reply(text, decision)
        # Recorded after the turn is done: what happened, next to what the
        # script said should happen, and the fields that differ.
        row = {
            "turn": turn["turn"],
            "user": turn["user"],
            "text": text,
            "bot": said,
            "action": decision.get("action"),
            "tool": decision.get("name"),
            "intent": labelled.get("intent"),
            "slots": {k: v for k, v in labelled.get("slots", {}).items() if v},
            "recalled": recalled_view(labelled.get("recalled", {})),
            # What is on disk now, so the page can show this step on its own.
            "state": snapshot(memory_root, db_path),
        }
        if not raw:
            checked = check_turn(turn, session, decision, said)
            row.update(
                expect=checked["expect"],
                actual=checked["actual"],
                misses=checked["misses"],
                passed=checked["passed"],
            )
        replies.append(row)

    for session in open_sessions.values():
        # The reflection and the summary episode are written here, before
        # the object is dropped. Session B reads them back from the file.
        session.memory.end_session(
            turns=[
                {"user": row.get("text", ""), "bot": row.get("bot", "")}
                for row in session.history
                if row.get("bot")
            ],
            turn=session.turn,
            trace=session.calls,
            run_id=run_id,
            config=config,
        )

    # The reflection and the summary episode exist only after the close above.
    closed_state = snapshot(memory_root, db_path)
    asked = _ask_probes(probes, memory_root, db_path, run_id, config, name, raw)
    return {"session": name, "turns": replies, "closed_state": closed_state, "probes": asked}


def _ask_probes(probes, memory_root, db_path, run_id, config, name, raw: bool = False) -> list[dict]:
    """Ask each probe against a Session built after the conversation closed.

    A new Session has empty slots. A probe that passes is passing on what
    was stored, not on what the conversation object still held.
    """
    asked = []
    for probe in probes:
        user = probe["user"]
        session = Session(
            user,
            name,
            db_path=db_path,
            memory=Memory(user, name, root=memory_root),
            strategy=STRATEGY,
            config=config,
        )
        session.run_id = run_id
        text = probe["question"]
        labelled = session.ingest(text)
        decision = session.decide(text)
        said = session.reply(text, decision)
        entry = {
            "intent": labelled.get("intent"),
            "recalled": recalled_view(labelled.get("recalled", {})),
            "state": snapshot(memory_root, db_path),
            "id": probe["id"],
            "category": probe["category"],
            "user": user,
            "question": text,
            "expected": probe["expected"],
            "evidence_turn": probe.get("evidence_turn"),
            "reply": said,
        }
        if not raw:
            entry["passed"] = _passed(probe, said)
        asked.append(entry)
    return asked


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sessions A and B under one configuration")
    parser.add_argument("--config", choices=ABLATION_CONFIGS, default="hybrid_mem")
    parser.add_argument("--session", choices=("A", "B"))
    parser.add_argument("--raw", action="store_true", help="record the conversation only, no expect comparison")
    args = parser.parse_args()
    sessions = (args.session,) if args.session else ("A", "B")
    transcript = run_config(args.config, sessions=sessions, raw=args.raw)
    for played in transcript["sessions"]:
        if args.raw:
            print(f"{args.config} session {played['session']}: {len(played['turns'])} turns, {len(played['probes'])} probes (raw)")
            continue
        probes = played["probes"]
        passed = sum(1 for probe in probes if probe["passed"])
        turns = played["turns"]
        ok = sum(1 for turn in turns if turn["passed"])
        print(
            f"{args.config} session {played['session']}: turns {ok}/{len(turns)} match expect, "
            f"probes {passed}/{len(probes)}"
        )
        for turn in turns:
            for miss in turn["misses"]:
                print(f"  turn {turn['turn']} {miss['field']}: expected {miss['expected']!r}, got {miss['actual']!r}")


if __name__ == "__main__":
    main()
