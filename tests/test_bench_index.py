"""Task 5(b): the index sizing sums, and flat vs HNSW recall.

The latency run over 100,000 vectors is `python -m src.bench_index`. It is
not part of the suite because it takes minutes. What is checked here is the
arithmetic the report quotes, and the recall property on a small index:
flat recall is 1 and HNSW recall is at most 1.
"""

from __future__ import annotations

import json

import numpy as np

from src.bench_index import (
    DAILY_USERS,
    DAYS,
    EMBED_DIM,
    HNSW_M,
    bytes_per_vector,
    conversation_rates,
    embedding_recall,
    estimate_bytes,
    estimate_vectors,
    recall_at_k,
)
from src.config import ROOT
from src.eval_retrieval import load_questions


def test_bytes_follow_the_faiss_formula():
    # flat is the float32 vector. HNSW adds M forward and M back links, each
    # an int32. Both are the formula on the FAISS index-selection page.
    per = bytes_per_vector(dim=384, m=16)
    assert per["flat"] == 384 * 4
    assert per["hnsw"] == 384 * 4 + 16 * 2 * 4
    assert per["hnsw"] > per["flat"]

    # The constants the report uses.
    assert bytes_per_vector() == {"flat": EMBED_DIM * 4, "hnsw": EMBED_DIM * 4 + HNSW_M * 2 * 4}


def test_one_year_count_uses_the_planted_conversation():
    rates = conversation_rates()
    # U1 only: 9 facts (turn 3 states the office and the floor), 3 updates,
    # 1 forget, 4 distinct events. U2's facts are a second user, not more
    # memory for the one user being scaled. Counted from the file, so
    # editing a turn changes this test.
    conversation = json.loads((ROOT / "data" / "test_conversation.json").read_text())
    assert rates == conversation_rates(conversation)
    assert rates["facts_per_user"] == 12
    assert rates["forgotten_per_user"] == 1
    assert rates["facts_stored_per_user"] == 11
    assert rates["episodes_per_user"] == 4

    chunks = 77
    estimate = estimate_vectors(chunks, rates=rates)
    assert estimate["kb_chunks"] == chunks
    assert estimate["semantic"] == DAILY_USERS * rates["facts_stored_per_user"]
    assert estimate["episodic"] == DAILY_USERS * DAYS * 4
    assert estimate["total"] == chunks + estimate["semantic"] + estimate["episodic"]

    sized = estimate_bytes(estimate["total"])
    assert sized["flat"] == estimate["total"] * 384 * 4
    assert sized["hnsw"] == estimate["total"] * (384 * 4 + 16 * 8)
    assert sized["hnsw"] > sized["flat"]


def test_recall_is_the_overlap_with_flat():
    flat = np.array([[0, 1, 2, 3, 4]])
    assert recall_at_k(flat, flat, 5) == 1.0
    # Three of the five flat neighbours returned: 0.6.
    partial = np.array([[0, 1, 2, 9, 8]])
    assert recall_at_k(flat, partial, 5) == 0.6
    # A longer list does not help: only the top k is counted.
    assert recall_at_k(flat, np.array([[9, 8, 7, 6, 5, 0]]), 5) == 0.0
    # Mean over queries.
    both = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    mixed = np.array([[0, 1, 2, 3, 4], [0, 9, 8, 7, 6]])
    assert recall_at_k(both, mixed, 5) == 0.6


def test_hnsw_recall_does_not_exceed_flat_on_the_real_questions():
    questions = load_questions()
    measured = embedding_recall(questions, ef_search=32)
    assert measured["flat"] == 1.0
    assert 0.0 <= measured["hnsw"] <= 1.0
    assert measured["n_questions"] == len(questions)
    assert measured["n_chunks"] > 0
