"""T4.2: RRF reproduces the CS4 worked example.

Slide 45 gives two rankings. Slide 46 fuses them with k=60 into
A, B, C, D, then E and F tied.
"""

from __future__ import annotations

import pytest

from src.rrf import CS4_DENSE, CS4_RANKING, CS4_SPARSE, K, fuse, reciprocal_rank


def test_t4_2_cs4_rrf_ranking():
    fused = fuse([CS4_DENSE, CS4_SPARSE])
    assert [doc_id for doc_id, _ in fused] == CS4_RANKING
    # E and F each appear in only one list, both at rank 4.
    scores = dict(fused)
    assert scores["E"] == scores["F"] == pytest.approx(1 / 64)


def test_cs4_scores_match_the_slide():
    """The six contributions printed on slide 46, before rounding."""
    scores = dict(fuse([CS4_DENSE, CS4_SPARSE]))
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)  # 0.03252
    assert scores["B"] == pytest.approx(1 / 63 + 1 / 61)  # 0.03226
    assert scores["C"] == pytest.approx(1 / 62 + 1 / 65)  # 0.03151
    assert scores["D"] == pytest.approx(1 / 65 + 1 / 63)  # 0.03125
    assert scores["E"] == pytest.approx(1 / 64)           # 0.01562
    assert scores["F"] == pytest.approx(1 / 64)           # 0.01562
    assert K == 60


def test_missing_from_one_list_adds_nothing_from_it():
    scores = dict(fuse([["A"], ["B", "A"]]))
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["B"] == pytest.approx(1 / 61)


def test_a_repeated_document_keeps_its_first_rank():
    scores = dict(fuse([["A", "B", "A"]]))
    assert scores["A"] == pytest.approx(reciprocal_rank(1))
    assert list(scores) == ["A", "B"]


def test_ties_keep_the_document_seen_first():
    # Both are rank 1 of their own list, so the scores match. A was seen first.
    fused = fuse([["A"], ["B"]])
    assert [doc_id for doc_id, _ in fused] == ["A", "B"]
    assert fused[0][1] == fused[1][1]


def test_empty_rankings_fuse_to_nothing():
    assert fuse([]) == []
    assert fuse([[], []]) == []


def test_rank_and_k_are_checked():
    with pytest.raises(ValueError):
        reciprocal_rank(0)
    with pytest.raises(ValueError):
        fuse([["A"]], k=-1)
