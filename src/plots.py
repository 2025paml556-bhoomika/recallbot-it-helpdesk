"""Input tokens per turn for the three short-term memory strategies.

Reads logs/calls.jsonl. One line per strategy, reply calls only, so the
shape of the prompt is what the chart shows. The printed totals add the
summarise calls too, because those tokens were spent to keep the notes.

    python -m src.plots
"""

from __future__ import annotations

import json
from collections import defaultdict

from src.config import LOG_PATH, ROOT

PLOT_PATH = ROOT / "results" / "plots" / "history_tokens.png"
STRATEGIES = ("full", "trim", "summary")


def load_calls(path=LOG_PATH) -> list[dict]:
    """Every logged call. A missing log is an empty run, not an error."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def strategy_of(run_id: str) -> str | None:
    """The strategy a run id belongs to.

    A strategy run is named stm-<strategy>, or stm-<strategy>-<date>. Any
    other run id is not one of these three and is left out of the chart.
    """
    if not isinstance(run_id, str) or not run_id.startswith("stm-"):
        return None
    name = run_id.split("-", 2)[1]
    return name if name in STRATEGIES else None


def reply_series(calls: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """(turn, input tokens) for each strategy's reply calls, in turn order."""
    series: dict[str, list[tuple[int, int]]] = {name: [] for name in STRATEGIES}
    for row in calls:
        if row.get("call_type") != "reply":
            continue
        name = strategy_of(row.get("run_id", ""))
        if name is None or row.get("turn") is None:
            continue
        series[name].append((int(row["turn"]), int(row["input_tokens"])))
    for name in series:
        series[name].sort()
    return series


def totals(calls: list[dict]) -> dict[str, dict[str, int]]:
    """Input tokens per strategy, split into reply calls and summarise calls.

    nlu calls are not part of the short-term memory cost, so they are left
    out. A tool call logs 0 tokens and does not change the total.
    """
    counted: dict[str, dict[str, int]] = {
        name: {"reply": 0, "summarise": 0} for name in STRATEGIES
    }
    for row in calls:
        name = strategy_of(row.get("run_id", ""))
        kind = row.get("call_type")
        if name is None or kind not in ("reply", "summarise"):
            continue
        counted[name][kind] += int(row.get("input_tokens") or 0)
    return counted


def plot(calls: list[dict], path=PLOT_PATH) -> dict:
    """Draw the three lines and return the totals the chart was drawn from."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = reply_series(calls)
    figure, ax = plt.subplots(figsize=(8, 4.5))
    for name in STRATEGIES:
        points = series[name]
        if not points:
            continue
        turns = [turn for turn, _tokens in points]
        tokens = [tokens for _turn, tokens in points]
        ax.plot(turns, tokens, marker="o", label=name)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Input tokens (reply calls)")
    ax.set_title("Short-term memory: input tokens per turn")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return totals(calls)


def main() -> None:
    calls = load_calls()
    counted = plot(calls)
    for name in STRATEGIES:
        reply = counted[name]["reply"]
        summary = counted[name]["summarise"]
        print(f"{name}: reply {reply} + summarise {summary} = {reply + summary}")
    print(f"wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
