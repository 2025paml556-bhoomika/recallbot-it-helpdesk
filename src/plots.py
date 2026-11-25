"""Input tokens per turn for the three short-term memory strategies.

Reads logs/calls.jsonl. One line per strategy, chat calls only, so the
shape of the prompt is what the chart shows. The printed totals add the
summary calls too, because those tokens were spent to keep the notes.

Lines written before the vocabulary change are labelled "reply" and
"summarise". Those names are read as "chat" and "summary", so a log that
holds both the old runs and a new one still plots. The file itself is not
rewritten.

    python -m src.plots
"""

from __future__ import annotations

import json
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


# Names the log used before the vocabulary was fixed. Read as the new name.
# The lines stay as they were written.
_OLD_CALL_TYPES = {"reply": "chat", "summarise": "summary", "embedding": "embed"}


def canonical_call_type(row: dict) -> str:
    """The call type, with the retired names mapped onto the current ones."""
    kind = row.get("call_type")
    return _OLD_CALL_TYPES.get(kind, kind)


def reply_series(calls: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """(turn, input tokens) for each strategy's chat calls, in turn order."""
    series: dict[str, list[tuple[int, int]]] = {name: [] for name in STRATEGIES}
    for row in calls:
        if canonical_call_type(row) != "chat":
            continue
        name = strategy_of(row.get("run_id", ""))
        if name is None or row.get("turn") is None:
            continue
        series[name].append((int(row["turn"]), int(row["input_tokens"])))
    for name in series:
        series[name].sort()
    return series


def totals(calls: list[dict]) -> dict[str, dict[str, int]]:
    """Input tokens per strategy, split into chat calls and summary calls.

    nlu calls are not part of the short-term memory cost, so they are left
    out. A tool call logs 0 tokens and does not change the total. A line
    logged as "reply" or "summarise" is counted under the name it maps to.
    """
    counted: dict[str, dict[str, int]] = {
        name: {"chat": 0, "summary": 0} for name in STRATEGIES
    }
    for row in calls:
        name = strategy_of(row.get("run_id", ""))
        kind = canonical_call_type(row)
        if name is None or kind not in ("chat", "summary"):
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
    ax.set_ylabel("Input tokens (chat calls)")
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
        reply = counted[name]["chat"]
        summary = counted[name]["summary"]
        print(f"{name}: chat {reply} + summary {summary} = {reply + summary}")
    print(f"wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
