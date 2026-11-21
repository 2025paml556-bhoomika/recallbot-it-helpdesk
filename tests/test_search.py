"""T4.8 to T4.15: search modes and the worked user questions.

Dense, BM25 and hybrid each return a ranked list. Hybrid is RRF over the two
real rankings, not a sum of their scores. The user questions check the
document that should come back, not a particular chunk inside it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ROOT
from src.search_kb import (
    MODES,
    RELEVANCE_THRESHOLD,
    search_bm25,
    search_dense,
    search_hybrid,
    search_kb,
)

# Chunk size the user examples are checked at. 512-token chunks swallow a
# whole document, so a code and an unrelated mention of it land in one vector.
EXAMPLE_SIZE = 128


def _doc_ids(hits: list[dict]) -> list[str]:
    return [hit["doc_id"] for hit in hits]


def test_t4_8_each_mode_returns_ranked_hits_with_doc_ids_and_scores():
    query = "What does ERR-VPN-809 mean?"
    for mode in MODES:
        hits = search_kb(query, mode=mode, top_k=3, size=EXAMPLE_SIZE)
        assert hits, mode
        assert [hit["rank"] for hit in hits] == [1, 2, 3]
        assert all(hit["mode"] == mode for hit in hits)
        assert all(hit["doc_id"] and hit["chunk_id"] for hit in hits)
        assert all(isinstance(hit["score"], float) for hit in hits)
        # Best first. Ties are allowed; a later hit must not outscore an earlier one.
        scores = [hit["score"] for hit in hits]
        assert scores == sorted(scores, reverse=True)


def test_t4_8_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        search_kb("vpn", mode="keyword")


def test_hybrid_score_is_rrf_not_a_sum_of_the_other_scores():
    """Cosine is at most 1 and BM25 here is above 1, so a sum cannot be ~0.03."""
    query = "What does ERR-VPN-809 mean?"
    hybrid = search_hybrid(query, top_k=1, size=EXAMPLE_SIZE)[0]
    assert hybrid["mode"] == "hybrid"
    assert 0 < hybrid["score"] < 0.1


def test_t4_9_src_imports_no_retrieval_library():
    banned = ("rank_bm25", "langchain", "llama_index")
    for path in Path(ROOT / "src").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in banned:
            assert name not in text, f"{path.name} mentions {name}"


def test_t4_10_error_code_ranks_the_vpn_document_first():
    """The code question from the plan, and the wording T4.10 actually prints.

    "What does ERR-VPN-809 mean?" is the question whose gold span is the
    definition. BM25 and hybrid both rank that VPN chunk first. The T4.10
    wording adds "fix", a word the VPN page never uses, so BM25's top chunk
    is the incident page (it says "how" and "error") and the VPN page is
    second. Hybrid still puts the VPN page first, which is the fusion doing
    the job dense alone also does.
    """
    defined = "What does ERR-VPN-809 mean?"
    for mode in ("bm25", "hybrid"):
        hits = search_kb(defined, mode=mode, top_k=1, size=EXAMPLE_SIZE)
        assert hits[0]["doc_id"] == "KB-VPN", mode
        assert "ERR-VPN-809" in hits[0]["text"]

    asked = "How do I fix error ERR-VPN-809?"
    hybrid = search_kb(asked, mode="hybrid", top_k=1, size=EXAMPLE_SIZE)
    assert hybrid[0]["doc_id"] == "KB-VPN"
    bm25 = search_kb(asked, mode="bm25", top_k=3, size=EXAMPLE_SIZE)
    assert bm25[0]["doc_id"] == "KB-INCIDENT"
    assert bm25[1]["doc_id"] == "KB-VPN"


def test_t4_11_paraphrase_puts_vpn_in_the_top_3():
    query = "My internet tunnel keeps dropping when I work from home"
    for mode in ("dense", "hybrid"):
        hits = search_kb(query, mode=mode, top_k=3, size=EXAMPLE_SIZE)
        assert "KB-VPN" in _doc_ids(hits), mode


def test_t4_12_current_password_policy_outranks_the_retired_one():
    query = "How often must I change my password?"
    hits = search_kb(query, mode="hybrid", top_k=5, size=EXAMPLE_SIZE)
    versions = [hit["version"] for hit in hits if hit["doc_id"] == "KB-PASSWORD"]
    assert "2026.1" in versions
    assert versions.index("2026.1") < versions.index("2025.1")


def test_t4_13_laptop_refresh_2026_is_in_the_top_3():
    query = "When am I due a new laptop?"
    hits = search_dense(query, top_k=3, size=EXAMPLE_SIZE)
    assert any(
        hit["doc_id"] == "KB-LAPTOP" and hit["version"] == "2026.1" for hit in hits
    )


def test_t4_14_sla_code_ranks_the_sla_document_first():
    query = "What is SLA-P1?"
    for mode in MODES:
        hits = search_kb(query, mode=mode, top_k=1, size=EXAMPLE_SIZE)
        assert hits[0]["doc_id"] == "KB-SLA", mode


def test_t4_15_out_of_scope_question_is_below_the_relevance_threshold():
    query = "What's on the cafeteria menu?"
    top = search_dense(query, top_k=1, size=EXAMPLE_SIZE)[0]
    assert top["score"] < RELEVANCE_THRESHOLD
    # An in-scope question clears the same threshold, so the cut is not vacuous.
    in_scope = search_dense("What does ERR-VPN-809 mean?", top_k=1, size=EXAMPLE_SIZE)[0]
    assert in_scope["score"] >= RELEVANCE_THRESHOLD


def test_bm25_and_dense_agree_on_the_shape_of_a_hit():
    query = "What is SLA-P1?"
    dense = search_dense(query, top_k=1, size=EXAMPLE_SIZE)[0]
    sparse = search_bm25(query, top_k=1, size=EXAMPLE_SIZE)[0]
    assert set(dense) == set(sparse)
    assert dense["mode"] == "dense"
    assert sparse["mode"] == "bm25"
