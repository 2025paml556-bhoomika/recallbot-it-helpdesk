"""Search the knowledge base by dense retrieval, BM25, or both.

Dense scores are cosine similarities and BM25 scores are unbounded, so hybrid
does not add them. It asks each method for a ranking and fuses those rankings
with reciprocal rank fusion (src.rrf, k=60). The ids that go into fuse() are
chunk ids from the index, not the A-F labels in the worked example.

    python -m src.search_kb "What does ERR-VPN-809 mean?"
    python -m src.search_kb "What does ERR-VPN-809 mean?" --mode hybrid --size 128
"""

from __future__ import annotations

import argparse

from src.bm25 import BM25, tokenize
from src.rrf import fuse
from src.vector_index import META_PATH, load, search as dense_search

MODES = ("dense", "bm25", "hybrid")

# How many chunks each method may contribute to a fusion. RRF only looks at
# rank, so this has to be deep enough that a chunk ranked well by one method
# and poorly by the other still receives both contributions.
FUSE_DEPTH = 20

# Cosine similarity below this is treated as "nothing relevant" (T4.15).
# In-scope questions on this index sit well above it; the cafeteria question
# does not. Step 9 uses the same cut-off for the "I don't know" fallback.
RELEVANCE_THRESHOLD = 0.30

_bm25_cache: dict[int | None, tuple[BM25, list[dict]]] = {}


def clear_cache() -> None:
    """Drop the cached BM25 indexes. Tests call this after swapping metadata."""
    _bm25_cache.clear()


def _chunks_of(meta: list[dict], size: int | None) -> list[dict]:
    if size is None:
        return list(meta)
    return [row for row in meta if row["size"] == size]


def bm25_index(size: int | None = None, meta: list[dict] | None = None) -> tuple[BM25, list[dict]]:
    """BM25 over the chunks of one size. Built once per size and reused.

    BM25 stores token lists. Passing the raw string would index one document
    per character, so every chunk is tokenised here.
    """
    if size not in _bm25_cache or meta is not None:
        if meta is None:
            _, meta = load()
        rows = _chunks_of(meta, size)
        index = BM25([tokenize(row["text"]) for row in rows])
        _bm25_cache[size] = (index, rows)
    return _bm25_cache[size]


def _hit(row: dict, rank: int, score: float, mode: str) -> dict:
    return {
        "rank": rank,
        "score": score,
        "mode": mode,
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "version": row["version"],
        "section": row["section"],
        "status": row["status"],
        "text": row["text"],
    }


def search_dense(
    query: str,
    top_k: int = 5,
    size: int | None = None,
    *,
    config: str | None = None,
    user: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
) -> list[dict]:
    """Nearest chunks by cosine similarity. Scores lie in [-1, 1]."""
    hits = dense_search(
        query,
        top_k=top_k,
        size=size,
        config=config,
        user=user,
        session=session,
        turn=turn,
        run_id=run_id,
    )
    return [_hit(row, rank, row["score"], "dense") for rank, row in enumerate(hits, start=1)]


def search_bm25(
    query: str,
    top_k: int = 5,
    size: int | None = None,
    meta: list[dict] | None = None,
) -> list[dict]:
    """Top chunks by BM25. A chunk that shares no term with the query is dropped."""
    index, rows = bm25_index(size, meta)
    found = index.search(query, top_k=top_k)
    return [_hit(rows[i], rank, score, "bm25") for rank, (i, score) in enumerate(found, start=1)]


def search_hybrid(
    query: str,
    top_k: int = 5,
    size: int | None = None,
    meta: list[dict] | None = None,
    *,
    config: str | None = None,
    user: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
) -> list[dict]:
    """Fuse the dense ranking and the BM25 ranking with RRF, k=60.

    Each method contributes up to FUSE_DEPTH chunks. A chunk missing from one
    ranking adds nothing from that ranking, which is the same rule as the
    worked example. The score on each hit is the fused RRF score, not a cosine
    and not a BM25 score.
    """
    if meta is None:
        _, meta = load()
    by_id = {row["chunk_id"]: row for row in _chunks_of(meta, size)}

    dense_ids = [
        row["chunk_id"]
        for row in search_dense(
            query,
            top_k=FUSE_DEPTH,
            size=size,
            config=config,
            user=user,
            session=session,
            turn=turn,
            run_id=run_id,
        )
        if row["chunk_id"] in by_id
    ]
    bm25, bm25_rows = bm25_index(size, meta)
    bm25_ids = [bm25_rows[i]["chunk_id"] for i, _ in bm25.search(query, top_k=FUSE_DEPTH)]

    fused = fuse([dense_ids, bm25_ids])
    return [
        _hit(by_id[chunk_id], rank, score, "hybrid")
        for rank, (chunk_id, score) in enumerate(fused[:top_k], start=1)
    ]


def search_kb(
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    size: int | None = None,
    *,
    config: str | None = None,
    user: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    run_id: str | None = None,
) -> list[dict]:
    """Top-k chunks for one query.

    mode is "dense", "bm25" or "hybrid". size is 128 or 512; omit it to
    search every stored chunk. Each hit has rank, score, mode, chunk_id and
    doc_id.

    The identity fields are forwarded to the embedding call. BM25 never
    embeds, so a bm25 search writes no embed line. Hybrid embeds once.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if top_k <= 0:
        return []
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"no index at {META_PATH}; build it with python -m src.vector_index"
        )

    identity = dict(
        config=config, user=user, session=session, turn=turn, run_id=run_id,
    )
    if mode == "dense":
        return search_dense(query, top_k=top_k, size=size, **identity)
    if mode == "bm25":
        return search_bm25(query, top_k=top_k, size=size)
    return search_hybrid(query, top_k=top_k, size=size, **identity)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the IT knowledge base")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--size", type=int, choices=(128, 512), default=128)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    hits = search_kb(args.query, mode=args.mode, top_k=args.top_k, size=args.size)
    print(f"mode={args.mode} size={args.size}")
    if not hits:
        print("(no hits)")
        return
    for hit in hits:
        print(
            f"{hit['rank']}. {hit['doc_id']}@{hit['version']}  "
            f"{hit['score']:.5f}  {hit['chunk_id']}"
        )


if __name__ == "__main__":
    main()
