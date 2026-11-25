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

from src.agent import Session
from src.config import ABLATION_CONFIGS, ROOT
from src.memory import Memory
from src.run_history import _passed, load_probes, load_turns

RUNS_ROOT = ROOT / "results" / "runs"

# The history strategy the ablation holds fixed. The thing being varied is
# which memory type is read, not how the window is cut.
STRATEGY = "trim"


def run_config(config: str, *, sessions: tuple[str, ...] = ("A", "B")) -> dict:
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
    run_id = f"abl-{config}"

    transcript = {"config": config, "run_id": run_id, "sessions": []}
    for name in sessions:
        played = _play_session(config, name, memory_root, db_path, run_id)
        transcript["sessions"].append(played)

    root.mkdir(parents=True, exist_ok=True)
    (root / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return transcript


def _play_session(config: str, name: str, memory_root, db_path, run_id: str) -> dict:
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
        session.ingest(text)
        decision = session.decide(text)
        said = session.reply(text, decision)
        replies.append(
            {
                "turn": turn["turn"],
                "user": turn["user"],
                "text": text,
                "bot": said,
                "action": decision.get("action"),
            }
        )

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

    asked = _ask_probes(probes, memory_root, db_path, run_id, config, name)
    return {"session": name, "turns": replies, "probes": asked}


def _ask_probes(probes, memory_root, db_path, run_id, config, name) -> list[dict]:
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
        session.ingest(text)
        decision = session.decide(text)
        said = session.reply(text, decision)
        asked.append(
            {
                "id": probe["id"],
                "category": probe["category"],
                "user": user,
                "question": text,
                "expected": probe["expected"],
                "evidence_turn": probe.get("evidence_turn"),
                "reply": said,
                "passed": _passed(probe, said),
            }
        )
    return asked


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sessions A and B under one configuration")
    parser.add_argument("--config", choices=ABLATION_CONFIGS, default="hybrid_mem")
    parser.add_argument("--session", choices=("A", "B"))
    args = parser.parse_args()
    sessions = (args.session,) if args.session else ("A", "B")
    transcript = run_config(args.config, sessions=sessions)
    for played in transcript["sessions"]:
        probes = played["probes"]
        passed = sum(1 for probe in probes if probe["passed"])
        print(f"{args.config} session {played['session']}: {len(played['turns'])} turns, probes {passed}/{len(probes)}")


if __name__ == "__main__":
    main()
