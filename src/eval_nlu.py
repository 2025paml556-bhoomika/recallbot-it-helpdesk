"""Accuracy of src.nlu on data/nlu_labelled.json.

A message counts as correct only when the intent matches and every labelled
slot is present with the labelled value. Extra slots the label does not
mention are ignored, so a model that also fills issue_type is not penalised.
A slot the label explicitly sets must match.

    python -m src.eval_nlu
"""

from __future__ import annotations

import json

from src.config import ROOT
from src.nlu import understand

LABELS_PATH = ROOT / "data" / "nlu_labelled.json"


def load_labels(path=LABELS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["labels"]


def slots_match(predicted: dict, expected: dict) -> bool:
    from src.nlu import _same_place

    for name, value in expected.items():
        got = predicted.get(name)
        if name == "location" and isinstance(got, str) and isinstance(value, str):
            if not _same_place(got, value):
                return False
            continue
        if got != value:
            return False
    return True


def score(labels: list[dict] | None = None) -> dict:
    """Run understand() on every label and return the accuracy breakdown."""
    labels = load_labels() if labels is None else labels
    rows = []
    for item in labels:
        predicted = understand(item["text"])
        intent_ok = predicted["intent"] == item["intent"]
        slots_ok = slots_match(predicted["slots"], item["slots"])
        rows.append(
            {
                "id": item["id"],
                "intent_ok": intent_ok,
                "slots_ok": slots_ok,
                "correct": intent_ok and slots_ok,
                "expected_intent": item["intent"],
                "predicted_intent": predicted["intent"],
                "source": predicted["source"],
            }
        )
    n = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "intent_accuracy": sum(1 for row in rows if row["intent_ok"]) / n if n else 0.0,
        "rows": rows,
    }


def main() -> None:
    result = score()
    print(f"labelled: {result['n']}")
    print(f"correct:  {result['correct']}")
    print(f"accuracy: {result['accuracy']:.3f}")
    print(f"intent:   {result['intent_accuracy']:.3f}")
    missed = [row for row in result["rows"] if not row["correct"]]
    if not missed:
        print("misses:   none")
        return
    print("misses:")
    for row in missed:
        print(
            f"  {row['id']}: expected {row['expected_intent']}, "
            f"got {row['predicted_intent']} ({row['source']}), "
            f"intent_ok={row['intent_ok']} slots_ok={row['slots_ok']}"
        )


if __name__ == "__main__":
    main()
