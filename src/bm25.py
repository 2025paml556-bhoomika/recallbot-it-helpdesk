"""BM25 scoring, written out rather than taken from a library.

The score of one document against a query is the sum over query terms of

    IDF(t) * f(t, d) * (k1 + 1)
    ---------------------------------------------
    f(t, d) + k1 * (1 - b + b * |d| / avgdl)

with the Robertson-Sparck Jones IDF

    IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) )

N is the number of documents, n(t) is how many of them contain t, f(t, d) is
how often t occurs in document d, and avgdl is the mean document length.
k1 (1.2) saturates term frequency and b (0.75) is the length normalisation.
Both are the usual defaults. A term missing from the document adds nothing,
and the same query term is counted once however often it is repeated.

    python -m src.bm25
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Lucene / Elasticsearch defaults.
K1 = 1.2
B = 0.75

# A token is a run of letters, digits, dots or hyphens, so codes such as
# ERR-VPN-809 and SLA-P1 stay in one piece. Everything else is a separator.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase and keep codes intact. Punctuation and whitespace separate tokens."""
    return _TOKEN.findall(text.lower())


def idf(n_docs: int, doc_freq: int) -> float:
    """Robertson-Sparck Jones IDF, floored at zero.

    The plain formula is negative once a term sits in more than half the
    documents, and a negative weight would push a document that contains the
    term below one that does not. Zero keeps the term from hurting a match
    while still giving it no reward. A term in no document is also zero.
    """
    if doc_freq <= 0 or n_docs <= 0:
        return 0.0
    return max(0.0, math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5)))


def term_weight(tf: int, doc_len: int, avgdl: float, k1: float = K1, b: float = B) -> float:
    """The saturated, length-normalised frequency of one term in one document."""
    if tf <= 0 or avgdl <= 0:
        return 0.0
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))


class BM25:
    """An in-memory BM25 index over a fixed list of documents.

    Documents are token lists. Text passed to search() is tokenised here;
    token lists passed to score() are used as they are, which is what the
    worked example needs.
    """

    def __init__(self, documents: list[list[str]], k1: float = K1, b: float = B):
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.n_docs = len(documents)
        self.doc_len = [len(doc) for doc in documents]
        total = sum(self.doc_len)
        self.avgdl = total / self.n_docs if self.n_docs else 0.0

        self.tf: list[Counter] = []
        doc_freq: Counter = Counter()
        for doc in documents:
            counts = Counter(doc)
            self.tf.append(counts)
            doc_freq.update(counts.keys())
        self.doc_freq = dict(doc_freq)
        self._idf = {term: idf(self.n_docs, df) for term, df in self.doc_freq.items()}

    def score(self, query_terms: list[str]) -> list[float]:
        """BM25 score of every document for these query terms, in index order."""
        scores = [0.0] * self.n_docs
        # Repeated query terms must not be counted twice. A term with no
        # positive IDF (missing, or common enough that its weight floored at
        # zero) adds nothing.
        for term in set(query_terms):
            term_idf = self._idf.get(term, 0.0)
            if term_idf == 0.0:
                continue
            for i, counts in enumerate(self.tf):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                scores[i] += term_idf * term_weight(
                    tf, self.doc_len[i], self.avgdl, self.k1, self.b
                )
        return scores

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """The top_k documents for a query string, best score first.

        Ties keep the lower document index first.
        """
        if top_k <= 0 or self.n_docs == 0:
            return []
        scored = self.score(tokenize(query))
        order = sorted(range(self.n_docs), key=lambda i: (-scored[i], i))
        return [(i, scored[i]) for i in order[:top_k] if scored[i] > 0.0]


def score_document(
    tf_in_doc: dict[str, int],
    doc_len: int,
    avgdl: float,
    doc_freq: dict[str, int],
    n_docs: int,
    query_terms: list[str],
    k1: float = K1,
    b: float = B,
) -> float:
    """Score one document from raw counts.

    This is the form the CS4 worked example uses: the slide gives frequencies
    and lengths directly, with no corpus of tokens to index.
    """
    total = 0.0
    for term in set(query_terms):
        tf = tf_in_doc.get(term, 0)
        if tf <= 0:
            continue
        total += idf(n_docs, doc_freq.get(term, 0)) * term_weight(tf, doc_len, avgdl, k1, b)
    return total


# The CS4 worked example. Three query terms against one document in a
# collection of 20 documents whose average length is 8, and this document is
# also 8 tokens, so |d| / avgdl is 1 and the length term drops out.
#
# The slide rounds each term's contribution to 3 decimal places before adding:
# 2.001 + 1.628 + 0.530 = 4.159. The same inputs with no intermediate
# rounding sum to 4.160.
CS4_N_DOCS = 20
CS4_DOC_LEN = 8
CS4_AVGDL = 8.0
CS4_QUERY = ["retrieval", "system", "search"]
CS4_TF = {"retrieval": 1, "system": 3, "search": 2}
CS4_DF = {"retrieval": 2, "system": 5, "search": 8}
CS4_SCORE = 4.160


def cs4_example_score() -> float:
    """The unrounded BM25 score of the CS4 worked example."""
    return score_document(
        CS4_TF, CS4_DOC_LEN, CS4_AVGDL, CS4_DF, CS4_N_DOCS, CS4_QUERY
    )


def main() -> None:
    score = cs4_example_score()
    print(f"CS4 BM25 score: {score:.3f}")
    if abs(score - CS4_SCORE) > 0.001:
        raise SystemExit(f"expected {CS4_SCORE}, got {score}")


if __name__ == "__main__":
    main()
