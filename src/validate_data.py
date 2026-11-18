"""Check the Task 1 test data against every minimum in Step 3.

Run from the repository root:

    python -m src.validate_data

Exits with status 1 if any minimum is missed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KB_DIR = DATA / "kb"

REQUIRED_CATEGORIES = ("R", "U", "F", "E", "K", "N", "S")
KB_SUBTYPES = ("code", "paraphrase", "versioned")


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def documents() -> dict[tuple[str, str], str]:
    """Map (doc_id, version) to the full text of that file."""
    found: dict[tuple[str, str], str] = {}
    for path in sorted(KB_DIR.glob("*.md")):
        header: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() == "":
                break
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
        found[(header["doc_id"], header["version"])] = path.read_text(encoding="utf-8")
    return found


def check(condition: bool, message: str, failures: list[str]) -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)


def validate_conversation(failures: list[str]) -> None:
    print("test_conversation.json")
    turns = load("test_conversation.json")["turns"]
    numbers = [turn["turn"] for turn in turns]
    users = {turn["user"] for turn in turns}

    def count(tag: str) -> int:
        return sum(1 for turn in turns if tag in turn["tags"])

    check(len(turns) >= 40, f"{len(turns)} user turns (need 40+)", failures)
    check(numbers == list(range(1, len(turns) + 1)), "turns are numbered 1..n with no gaps", failures)

    session_a = [t for t in turns if t["turn"] <= 25]
    session_b = [t for t in turns if t["turn"] >= 26]
    check(bool(session_a) and all(t["session"] == "A" for t in session_a), "turns 1-25 are Session A", failures)
    check(bool(session_b) and all(t["session"] == "B" for t in session_b), "turns 26 onward are Session B", failures)
    check(users == {"U1", "U2"}, f"users present: {sorted(users)} (need U1 and U2)", failures)

    facts = [fact for turn in turns for fact in turn.get("facts", [])]
    updates = [item for turn in turns for item in turn.get("updates", [])]
    forgets = [item for turn in turns for item in turn.get("forgets", [])]
    events = [item for turn in turns for item in turn.get("events", [])]

    check(len(facts) >= 8, f"{len(facts)} user facts (need 8+)", failures)
    check(len(updates) >= 2, f"{len(updates)} fact updates (need 2+)", failures)
    check(len(forgets) >= 1, f"{len(forgets)} forget requests (need 1+)", failures)
    check(len(events) >= 3, f"{len(events)} notable events (need 3+)", failures)
    check(count("small_talk") >= 5, f"{count('small_talk')} small-talk turns (need 5+)", failures)
    check(count("kb_question") >= 5, f"{count('kb_question')} knowledge-base questions (need 5+)", failures)
    check(count("state_followup") >= 2, f"{count('state_followup')} state follow-ups (need 2+)", failures)
    check(count("needs_clarification") >= 1, f"{count('needs_clarification')} clarification turns (need 1+)", failures)

    confirmed = [t for t in turns if "confirmation" in t["tags"]]
    check(len(confirmed) == 1, f"{len(confirmed)} confirmed database write (need 1)", failures)

    kb_sessions = {t["session"] for t in turns if "kb_question" in t["tags"]}
    check(kb_sessions == {"A", "B"}, "knowledge-base questions appear in both sessions", failures)


def validate_questions(failures: list[str]) -> None:
    print("kb_questions.json")
    questions = load("kb_questions.json")["questions"]
    docs = documents()
    subtypes: Counter[str] = Counter()
    spans_ok = True

    for question in questions:
        subtypes.update(question["subtypes"])
        text = docs.get((question["doc_id"], question["version"]))
        if text is None or question["gold_span"] not in text:
            spans_ok = False
            print(f"    span missing for {question['id']} in {question['doc_id']} {question['version']}")
        if "versioned" in question["subtypes"]:
            older = [
                body
                for (doc_id, version), body in docs.items()
                if doc_id == question["doc_id"] and version != question["version"]
            ]
            if any(question["gold_span"] in body for body in older):
                spans_ok = False
                print(f"    {question['id']} is marked versioned but its span is also in another version")

    check(len(questions) >= 25, f"{len(questions)} questions (need 25+)", failures)
    for subtype in KB_SUBTYPES:
        minimum = 5 if subtype != "versioned" else 2
        check(subtypes[subtype] >= minimum, f"{subtypes[subtype]} {subtype} questions (need {minimum}+)", failures)
    check(spans_ok, "every gold span appears verbatim in its doc_id and version", failures)


def validate_probes(failures: list[str]) -> None:
    print("probes.json")
    probes = load("probes.json")["probes"]
    categories = Counter(probe["category"] for probe in probes)

    check(len(probes) >= 14, f"{len(probes)} probes (need 14+)", failures)
    for category in REQUIRED_CATEGORIES:
        check(categories[category] >= 1, f"category {category}: {categories[category]} (need 1+)", failures)
    check(categories["isolation"] >= 2, f"{categories['isolation']} isolation probes (need 2+)", failures)

    isolation_ok = all(
        probe["expected"] == "does not know" and probe["user"] == "U2"
        for probe in probes
        if probe["category"] == "isolation"
    )
    check(isolation_ok, "every isolation probe is asked by U2 and expects 'does not know'", failures)


def main() -> int:
    failures: list[str] = []
    validate_conversation(failures)
    validate_questions(failures)
    validate_probes(failures)
    print("result:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
