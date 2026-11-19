"""T4.1: BM25 reproduces the CS4 worked example.

The slide rounds each IDF and each term contribution to 3 decimal places and
prints 4.159. The same inputs with no intermediate rounding score 4.160.
"""

from __future__ import annotations

import math

from src.bm25 import (
    CS4_AVGDL,
    CS4_DF,
    CS4_DOC_LEN,
    CS4_N_DOCS,
    CS4_QUERY,
    CS4_SCORE,
    CS4_TF,
    BM25,
    idf,
    score_document,
    tokenize,
)


def test_t4_1_cs4_bm25_score():
    score = score_document(
        CS4_TF, CS4_DOC_LEN, CS4_AVGDL, CS4_DF, CS4_N_DOCS, CS4_QUERY
    )
    assert abs(score - CS4_SCORE) <= 0.001
    assert round(score, 3) == CS4_SCORE


def test_cs4_score_term_by_term():
    """Each term's IDF and saturated weight, checked before they are summed."""
    expected_idf = {
        "retrieval": math.log((20 - 2 + 0.5) / (2 + 0.5)),
        "system": math.log((20 - 5 + 0.5) / (5 + 0.5)),
        "search": math.log((20 - 8 + 0.5) / (8 + 0.5)),
    }
    total = 0.0
    for term, tf in CS4_TF.items():
        assert idf(CS4_N_DOCS, CS4_DF[term]) == expected_idf[term]
        # Documents are the average length, so the length factor is 1 and the
        # weight is tf*(k1+1)/(tf+k1) with k1 = 1.2.
        weight = tf * 2.2 / (tf + 1.2)
        total += expected_idf[term] * weight
    assert abs(total - CS4_SCORE) <= 0.001


def test_bm25_index_matches_the_direct_score():
    """The index scores a real token corpus with the same formula."""
    # 20 documents, each 8 tokens. Document 0 has the slide's term frequencies.
    # The other occurrences (one retrieval, four system, seven search) sit in
    # documents 1 to 7, which brings the document frequencies to 2, 5 and 8.
    pad = ["word"] * 8
    docs = [pad[:] for _ in range(CS4_N_DOCS)]
    docs[0] = ["retrieval", "system", "system", "system", "search", "search", "word", "word"]
    docs[1] = ["retrieval", "system", "search", "word", "word", "word", "word", "word"]
    for i in range(2, 5):
        docs[i] = ["system", "search", "word", "word", "word", "word", "word", "word"]
    for i in range(5, 8):
        docs[i] = ["search", "word", "word", "word", "word", "word", "word", "word"]

    index = BM25(docs)
    assert index.n_docs == CS4_N_DOCS
    assert index.avgdl == CS4_AVGDL
    assert index.doc_freq["retrieval"] == 2
    assert index.doc_freq["system"] == 5
    assert index.doc_freq["search"] == 8

    scores = index.score(CS4_QUERY)
    assert abs(scores[0] - CS4_SCORE) <= 0.001
    assert scores[0] == max(scores)


def test_missing_term_adds_nothing_and_query_repeats_count_once():
    docs = [["cat", "sat"], ["dog", "sat", "sat"]]
    index = BM25(docs)
    once = index.score(["sat"])
    twice = index.score(["sat", "sat"])
    assert once == twice
    assert index.score(["missing"]) == [0.0, 0.0]


def test_tokenize_keeps_codes_and_lowercases():
    assert tokenize("How do I fix error ERR-VPN-809?") == [
        "how",
        "do",
        "i",
        "fix",
        "error",
        "err-vpn-809",
    ]


def test_search_orders_by_score_and_skips_zeros():
    docs = [
        ["vpn", "error"],
        ["password", "policy"],
        ["vpn", "vpn", "setup"],
        ["email"],
        ["printer"],
    ]
    found = BM25(docs).search("vpn", top_k=5)
    # Two mentions outrank one. Documents with no shared term are left out.
    assert [i for i, _ in found] == [2, 0]
    assert found[0][1] > found[1][1]
