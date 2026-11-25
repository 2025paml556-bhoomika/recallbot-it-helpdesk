"""The prompt stays inside 4,096 tokens with 500 held back for the reply.

No test calls Ollama. The tokenizer is the chat model's, which is what the
budget is counted with. Memory and retrieval are handed in as plain dicts.
"""

from __future__ import annotations

import csv

import pytest

from src.config import CONTEXT_WINDOW, PROMPT_BUDGET, REPLY_RESERVE_TOKENS
from src.context_builder import (
    PARTS,
    assert_budget,
    assemble,
    count_tokens,
    write_table,
)


def _big(label: str, words: int) -> str:
    return " ".join(f"{label}{i}" for i in range(words))


def test_the_budget_is_the_window_minus_the_reply_reserve():
    assert CONTEXT_WINDOW == 4096
    assert REPLY_RESERVE_TOKENS == 500
    assert PROMPT_BUDGET == CONTEXT_WINDOW - REPLY_RESERVE_TOKENS


def test_a_small_prompt_keeps_every_part_and_fits():
    built = assemble(
        system="You are the IT helpdesk.",
        tool_result="Ticket TCK-0001 is open.",
        facts=[{"id": "sem-0001", "key": "office", "value": "Hyderabad"}],
        episodes=[{"id": "ep-0007", "text": "Booked a visit."}],
        history="User: Hi.\nBot: Hi.",
        chunks=[{"doc_id": "KB-VPN", "chunk_id": "KB-VPN@2026.1:128:0", "text": "The tunnel dropped."}],
        current="What does ERR-VPN-809 mean?",
    )

    assert built.within_budget()
    assert built.total <= PROMPT_BUDGET
    assert "[mem:sem-0001]" in built.prompt
    assert "[mem:ep-0007]" in built.prompt
    assert "[KB-VPN#KB-VPN@2026.1:128:0]" in built.prompt
    assert "What does ERR-VPN-809 mean?" in built.prompt
    # The system prompt is its own message, not copied into the user prompt.
    assert "You are the IT helpdesk." not in built.prompt
    assert built.system == "You are the IT helpdesk."
    # Nothing was dropped, so every part is in the prompt.
    assert {row["location"] for row in built.rows} == {"prompt"}


def test_the_token_table_matches_a_recount():
    """The budget number is a fresh tokenize of the text that is sent.

    The per-section rows are each a fresh tokenize of that section. Joining
    the sections changes the count by a few tokens, because the tokenizer
    merges across a boundary, so the rows are not required to add up to the
    total. The total is the joined text, which is what is sent.
    """
    built = assemble(
        system="You are the IT helpdesk.",
        facts=[{"id": "sem-0001", "key": "office", "value": "Hyderabad"}],
        history="User: The VPN keeps failing.\nBot: Raised TCK-0001.",
        current="Which office am I based at?",
    )
    assert built.total == count_tokens(built.system) + count_tokens(built.prompt)
    for row in built.rows:
        if row["location"] == "prompt" and row["part"] != "system":
            assert row["tokens"] > 0


def test_the_table_names_the_memory_type_and_where_the_part_lives():
    built = assemble(
        system="You are the IT helpdesk.",
        tool_result="No ticket was found.",
        facts=[{"id": "sem-0001", "key": "office", "value": "Hyderabad"}],
        current="Where do I sit?",
    )
    kinds = {row["part"]: row for row in built.table()}
    assert list(kinds) == list(PARTS)
    assert kinds["system"]["memory_type"] == "procedural"
    assert kinds["long_term_memory"]["memory_type"] == "semantic+episodic"
    assert kinds["history"]["memory_type"] == "short-term"
    assert kinds["retrieved_knowledge"]["memory_type"] == "knowledge base"
    # No history and no chunks were given, so those parts stay external.
    assert kinds["history"]["location"] == "external"
    assert kinds["history"]["tokens"] == 0
    assert kinds["retrieved_knowledge"]["location"] == "external"
    assert kinds["long_term_memory"]["location"] == "prompt"


def test_chunks_are_dropped_lowest_rank_first_until_the_prompt_fits():
    chunks = [
        {"doc_id": "KB-VPN", "chunk_id": f"c{i}", "text": _big(f"chunk{i}", 400)}
        for i in range(6)
    ]
    built = assemble(
        system="You are the IT helpdesk.",
        chunks=chunks,
        current="What does ERR-VPN-809 mean?",
    )

    assert built.within_budget()
    # The best chunk is the one that has to survive.
    assert "[KB-VPN#c0]" in built.prompt
    kept = [row for row in built.rows if row["part"] == "retrieved_knowledge" and row["location"] == "prompt"]
    left = [row for row in built.rows if row["part"] == "retrieved_knowledge" and row["location"] == "external"]
    assert kept and left
    # What was cut is the tail, not the head.
    assert "[KB-VPN#c0]" not in left[0]["text"]
    assert "[KB-VPN#c5]" in left[0]["text"]


def test_history_is_dropped_before_the_newest_fact_when_the_budget_is_tight():
    """History is one block and goes whole, before the last fact does."""
    facts = [{"id": f"sem-{i:04d}", "key": "note", "value": _big(f"fact{i}", 80)} for i in range(3)]
    built = assemble(
        system="You are the IT helpdesk.",
        facts=facts,
        history=_big("turn", 400),
        current="Where do I sit?",
        budget=600,
    )

    assert built.within_budget()
    assert "turn0" not in built.prompt
    # The first fact is the newest and the most relevant, so it is last to go.
    assert "[mem:sem-0000]" in built.prompt
    assert any(row["part"] == "history" and row["location"] == "external" and row["tokens"] for row in built.rows)


def test_a_prompt_that_cannot_be_trimmed_raises():
    with pytest.raises(ValueError):
        assemble(
            system=_big("system", 3000),
            tool_result=_big("tool", 2000),
            current="Hi.",
            budget=1000,
        )


def test_assert_budget_rejects_a_prompt_over_the_limit():
    built = assemble(system="You are the IT helpdesk.", current="Hi.", budget=PROMPT_BUDGET)
    assert_budget(built)

    over = assemble(system=_big("s", 3000), current="Hi.", budget=20_000)
    # The builder trusted the budget it was given. The check does not.
    assert over.total > PROMPT_BUDGET
    with pytest.raises(AssertionError):
        assert_budget(over)


def test_the_table_is_written_as_csv(tmp_path):
    built = assemble(
        system="You are the IT helpdesk.",
        facts=[{"id": "sem-0001", "key": "office", "value": "Hyderabad"}],
        current="Which office am I based at now?",
    )
    path = tmp_path / "context_budget.csv"
    write_table(built, path)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[-1]["part"] == "total_in_prompt"
    assert int(rows[-1]["tokens"]) == built.total
    assert int(rows[-1]["tokens"]) <= PROMPT_BUDGET
    parts = [row["part"] for row in rows[:-1]]
    assert parts[0] == "system"
    assert "long_term_memory" in parts
