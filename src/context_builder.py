"""Assemble one reply prompt inside the 4,096-token window.

The window is CONTEXT_WINDOW. REPLY_RESERVE_TOKENS of it is kept for the
reply, so the prompt itself must stay at or under PROMPT_BUDGET. The parts
go in in a fixed order, and the ones that can be dropped go in last:

    system and instructions   kept whole
    tool result               kept whole, it is the thing being answered
    long-term memory          facts, then episodes, newest dropped first
    conversation history      oldest turns dropped first
    retrieved knowledge       lowest-ranked chunks dropped first
    current task              kept whole

A part that does not fit is left out entirely. A chunk is never cut in half,
because a half chunk cannot be cited. The system prompt, the tool result and
the current message are not trimmable: if those three already pass the
budget, the prompt is over and assemble() raises rather than sending it.

What is left out stays where it lives. Facts and episodes stay in the memory
files, turns stay in the session file, and chunks stay in the index. The
table records each part as "prompt" or "external" so the report can say which.

    python -m src.context_builder
"""

from __future__ import annotations

import csv

from src.grounding import chunk_cite, memory_cite
from src.chunker import load_tokenizer
from src.config import CONTEXT_WINDOW, PROMPT_BUDGET, REPLY_RESERVE_TOKENS, ROOT

# The order the parts are written into the prompt. Trimming never reorders.
PARTS = (
    "system",
    "tool_result",
    "long_term_memory",
    "history",
    "retrieved_knowledge",
    "current_task",
)

# Where each part comes from. "prompt" means the model sees it. "external"
# means it lives in a store and is only copied in when it fits.
MEMORY_TYPE = {
    "system": "procedural",
    "tool_result": "tool",
    "long_term_memory": "semantic+episodic",
    "history": "short-term",
    "retrieved_knowledge": "knowledge base",
    "current_task": "current turn",
}

# Dropped from the end, so the highest-ranked chunk and the newest memory
# survive. History is a single block already windowed by the strategy, so it
# is kept or dropped as a whole.
_TRIMMABLE = ("retrieved_knowledge", "history", "long_term_memory")

# Written by `python -m src.context_builder` for the report.
TABLE_PATH = ROOT / "results" / "context_budget.csv"

_tokenizer = None


def count_tokens(text: str) -> int:
    """How many tokens the chat model's tokenizer makes of this text.

    The same tokenizer the chunks were cut with, so a chunk counted here
    matches the token_count stored on it. An empty string is zero.
    """
    if not text:
        return 0
    return len(_load_tokenizer()(text, add_special_tokens=False)["input_ids"])


def _load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = load_tokenizer()
    return _tokenizer


def _render_facts(facts: list[dict]) -> list[str]:
    """One line per fact, each with its [mem:id] so the reply can cite it."""
    lines = []
    for fact in facts or []:
        lines.append(f"{memory_cite(fact.get('id'))} {fact.get('key')}: {fact.get('value')}")
    return lines


def _render_episodes(episodes: list[dict]) -> list[str]:
    lines = []
    for episode in episodes or []:
        lines.append(f"{memory_cite(episode.get('id'))} {episode.get('text')}")
    return lines


def _render_chunks(chunks: list[dict]) -> list[str]:
    lines = []
    for hit in chunks or []:
        lines.append(f"{chunk_cite(hit)} {hit.get('text', '')}")
    return lines


class Context:
    """One assembled prompt, plus the token count of every part.

    prompt is the user message. The system prompt is separate, because the
    chat call sends it as its own message. total counts both, since the
    window holds both.
    """

    def __init__(self, prompt: str, system: str, rows: list[dict]) -> None:
        self.prompt = prompt
        self.system = system
        self.rows = rows

    @property
    def total(self) -> int:
        """Tokens actually sent: the system message plus the user prompt.

        The rows are per section and each is counted on its own, so their sum
        is a little higher than the joined prompt wherever the tokenizer
        merges across a boundary. The joined text is what the model receives,
        so that is the number the budget is checked against.
        """
        return count_tokens(self.system) + count_tokens(self.prompt)

    def within_budget(self) -> bool:
        return self.total <= PROMPT_BUDGET

    def table(self) -> list[dict]:
        """One row per part: tokens, memory type, prompt or external."""
        return [
            {
                "part": row["part"],
                "tokens": row["tokens"],
                "memory_type": row["memory_type"],
                "location": row["location"],
            }
            for row in self.rows
        ]


def assemble(
    *,
    system: str,
    tool_result: str = "",
    facts: list[dict] | None = None,
    episodes: list[dict] | None = None,
    history: str = "",
    chunks: list[dict] | None = None,
    current: str,
    budget: int = PROMPT_BUDGET,
) -> Context:
    """Build the prompt, dropping the lowest-priority parts until it fits.

    facts and episodes are this user's only. The caller has already filtered
    them. chunks are the passages the reply may quote, best first.

    Raises ValueError when the system prompt, the tool result and the current
    message alone exceed the budget. Those three are the turn, and there is
    nothing left to drop.
    """
    memory_lines = _render_facts(facts) + _render_episodes(episodes)
    chunk_lines = _render_chunks(chunks)

    # Each trimmable part is a list of lines. Dropping removes a whole line.
    blocks = {
        "system": [system.strip()] if system.strip() else [],
        "tool_result": [tool_result.strip()] if tool_result.strip() else [],
        "long_term_memory": list(memory_lines),
        "history": [history.strip()] if history.strip() else [],
        "retrieved_knowledge": list(chunk_lines),
        "current_task": [current.strip()] if current.strip() else [],
    }
    dropped = {name: [] for name in PARTS}

    def cost() -> int:
        # Counted the way the prompt is joined, headings included, so the
        # number that decides a cut is the number the table reports.
        return _measure(system, blocks)

    while cost() > budget:
        for name in _TRIMMABLE:
            if not blocks[name]:
                continue
            # Facts and chunks are best first, so the last line is the one
            # to lose. History is one block and goes all at once.
            dropped[name].insert(0, blocks[name].pop())
            break
        else:
            raise ValueError(
                f"prompt is {cost()} tokens and nothing trimmable remains; "
                f"budget is {budget}"
            )

    rows = []
    for name in PARTS:
        kept = "\n".join(blocks[name])
        left_out = "\n".join(dropped[name])
        rows.append(
            {
                "part": name,
                "text": kept,
                "tokens": _section_tokens(name, blocks),
                "memory_type": MEMORY_TYPE[name],
                "location": "prompt" if kept else "external",
            }
        )
        # A part that was cut still has a size. Record it as external, so the
        # table shows the tokens that stayed on disk.
        if left_out:
            rows.append(
                {
                    "part": name,
                    "text": left_out,
                    "tokens": count_tokens(left_out),
                    "memory_type": MEMORY_TYPE[name],
                    "location": "external",
                }
            )

    prompt = _join(
        ("Long-term memory:", blocks["long_term_memory"]),
        ("Earlier conversation:", blocks["history"]),
        ("Retrieved knowledge:", blocks["retrieved_knowledge"]),
        ("Tool result:", blocks["tool_result"]),
        ("Current message:", blocks["current_task"]),
    )
    return Context(prompt, system.strip(), rows)


def assert_budget(context: Context) -> None:
    """Raise when the prompt leaves fewer than REPLY_RESERVE_TOKENS free.

    Called on every turn that builds a prompt. A prompt that fits is returned
    unchanged, so the caller can chain it.
    """
    if not context.within_budget():
        raise AssertionError(
            f"prompt is {context.total} tokens, over the {PROMPT_BUDGET} budget "
            f"({CONTEXT_WINDOW} window, {REPLY_RESERVE_TOKENS} held for the reply)"
        )


def write_table(context: Context, path=None) -> None:
    """Write the assembly table as CSV. One row per part."""
    target = path or TABLE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = context.table()
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["part", "tokens", "memory_type", "location"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "part": "total_in_prompt",
                "tokens": context.total,
                "memory_type": "",
                "location": "prompt",
            }
        )


def _measure(system: str, blocks: dict[str, list[str]]) -> int:
    """Tokens of the system message plus the joined user prompt.

    The headings the prompt adds are part of the count, because they are
    part of what is sent.
    """
    prompt = _join(
        ("Long-term memory:", blocks["long_term_memory"]),
        ("Earlier conversation:", blocks["history"]),
        ("Retrieved knowledge:", blocks["retrieved_knowledge"]),
        ("Tool result:", blocks["tool_result"]),
        ("Current message:", blocks["current_task"]),
    )
    return count_tokens(system.strip()) + count_tokens(prompt)


def _section_tokens(name: str, blocks: dict[str, list[str]]) -> int:
    """Tokens of this section on its own.

    The system message is tokenized by itself. A user section is tokenized
    with its heading, which is how it appears in the prompt. The rows are
    not expected to sum to the total: the tokenizer merges across the
    boundary when the sections are joined, and the total is that joined text.
    """
    if name == "system":
        return count_tokens("\n".join(blocks[name]))
    headings = {
        "long_term_memory": "Long-term memory:",
        "history": "Earlier conversation:",
        "retrieved_knowledge": "Retrieved knowledge:",
        "tool_result": "Tool result:",
        "current_task": "Current message:",
    }
    if not blocks[name]:
        return 0
    return count_tokens(headings[name] + "\n" + "\n".join(blocks[name]))


def _join(*sections: tuple[str, list[str]]) -> str:
    """The labelled sections that have any lines, separated by a blank line."""
    parts = []
    for heading, lines in sections:
        if not lines:
            continue
        parts.append(heading + "\n" + "\n".join(lines))
    return "\n\n".join(parts)


def main() -> None:
    """One example turn, so the budget can be checked without a session."""
    built = assemble(
        system="You are the corporate IT helpdesk.",
        tool_result="[KB-VPN#KB-VPN@2026.1:128:0] ERR-VPN-809 means the tunnel dropped.",
        facts=[{"id": "sem-0001", "key": "office", "value": "Hyderabad"}],
        episodes=[{"id": "ep-0007", "text": "Booked a visit at the Pune office."}],
        history="User: The VPN keeps failing.\nBot: Raised TCK-0001.",
        chunks=[{"doc_id": "KB-VPN", "chunk_id": "KB-VPN@2026.1:128:0", "text": "ERR-VPN-809."}],
        current="What does ERR-VPN-809 mean?",
    )
    assert_budget(built)
    write_table(built)
    print(f"prompt tokens: {built.total}  budget: {PROMPT_BUDGET}")
    for row in built.table():
        print(f"  {row['part']:24} {row['tokens']:5}  {row['memory_type']:20} {row['location']}")


if __name__ == "__main__":
    main()
