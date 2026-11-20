"""Reciprocal rank fusion, written out rather than taken from a library.

Dense scores are cosine similarities in [0, 1] and BM25 scores are unbounded,
so the two cannot be added. RRF ignores the values and uses only the rank
each document was given:

    RRF(d) = sum over rankings  1 / (k + rank(d))

rank starts at 1. k is 60, the value Cormack, Clarke and Büttcher (2009)
found keeps the top rank from overwhelming the rest: 1/61 and 1/62 are close,
whereas 1/1 and 1/2 are not. A document missing from a ranking adds nothing
from that ranking.

    python -m src.rrf
"""

from __future__ import annotations

# Cormack et al. (2009). Large enough that rank 1 does not dominate rank 5.
K = 60

# The CS4 worked example (slide 45). Two rankings of five documents each.
# Dense:  A, C, B, E, D
# Sparse: B, A, D, F, C
# Fused with k=60 the order is A, B, C, D, then E and F tied (slide 46).
CS4_DENSE = ["A", "C", "B", "E", "D"]
CS4_SPARSE = ["B", "A", "D", "F", "C"]
CS4_RANKING = ["A", "B", "C", "D", "E", "F"]


def reciprocal_rank(rank: int, k: int = K) -> float:
    """The contribution of one position. Rank 1 is the first result."""
    if rank < 1:
        raise ValueError(f"rank starts at 1, got {rank}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    return 1.0 / (k + rank)


def fuse(
    rankings: list[list[str]],
    k: int = K,
) -> list[tuple[str, float]]:
    """Fuse ranked lists into one list of (document, score), best first.

    Each inner list is a ranking, best result first, and may be a different
    length. A document that appears twice in one ranking is scored at its
    first position only. Ties keep the document that was seen first, which
    for the CS4 lists puts E before F.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    seen_at = 0
    for ranking in rankings:
        placed: set[str] = set()
        for position, doc_id in enumerate(ranking):
            if doc_id in placed:
                continue
            placed.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + reciprocal_rank(position + 1, k)
            if doc_id not in first_seen:
                first_seen[doc_id] = seen_at
                seen_at += 1

    order = sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id]))
    return [(doc_id, scores[doc_id]) for doc_id in order]


def cs4_example() -> list[tuple[str, float]]:
    """The CS4 dense and sparse lists fused with k=60."""
    return fuse([CS4_DENSE, CS4_SPARSE])


def main() -> None:
    fused = cs4_example()
    ranking = [doc_id for doc_id, _ in fused]
    for doc_id, score in fused:
        print(f"{doc_id}  {score:.5f}")
    if ranking != CS4_RANKING:
        raise SystemExit(f"expected {CS4_RANKING}, got {ranking}")
    # E and F are the tie the slide prints on one line.
    if fused[-2][1] != fused[-1][1]:
        raise SystemExit("E and F should be tied")
    print("order: A, B, C, D, then E and F tied")


if __name__ == "__main__":
    main()
