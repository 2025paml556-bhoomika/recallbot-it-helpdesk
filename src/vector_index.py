"""A flat FAISS index over the knowledge-base chunks.

Flat, meaning IndexFlatIP, compares the query with every stored vector.
With 77 chunks that is both cheap and exact, so there is no reason to accept
the recall loss of an approximate index such as HNSW. The embedder returns
L2-normalised vectors, so the inner product it computes is the cosine
similarity.

The index holds only vectors. The chunk text and its metadata are written
beside it, in the same order, and the position of a vector is the position of
its chunk.

    python -m src.vector_index
"""

from __future__ import annotations

import json

import faiss
import numpy as np

from src.config import CHUNKS_PATH, EMBED_DIM, INDEX_DIR
from src.embedder import embed

INDEX_PATH = INDEX_DIR / "kb.faiss"
META_PATH = INDEX_DIR / "kb_meta.jsonl"

# What a search result needs to carry back. The text stays in the metadata
# file so the index itself is only vectors.
META_FIELDS = (
    "chunk_id",
    "doc_id",
    "version",
    "valid_from",
    "status",
    "section",
    "size",
    "index",
    "text",
)


def load_chunks(path=CHUNKS_PATH) -> list[dict]:
    """Read the chunks written by src.chunker, one JSON object per line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build(chunks: list[dict] | None = None) -> tuple[faiss.Index, list[dict]]:
    """Embed every chunk and add it to a new flat index.

    The metadata list is in the same order as the vectors, so search results
    can be joined back to their chunk by position.
    """
    chunks = load_chunks() if chunks is None else chunks
    if not chunks:
        raise ValueError(f"no chunks to index in {CHUNKS_PATH}")

    vectors = embed([chunk["text"] for chunk in chunks])
    if vectors.shape != (len(chunks), EMBED_DIM):
        raise ValueError(f"expected {(len(chunks), EMBED_DIM)} vectors, got {vectors.shape}")

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    meta = [{field: chunk[field] for field in META_FIELDS} for chunk in chunks]
    return index, meta


def save(index: faiss.Index, meta: list[dict]) -> None:
    """Write the index and its metadata. The two files must stay in step."""
    if index.ntotal != len(meta):
        raise ValueError(f"index has {index.ntotal} vectors but {len(meta)} metadata rows")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    lines = [json.dumps(row, ensure_ascii=False) for row in meta]
    META_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load() -> tuple[faiss.Index, list[dict]]:
    """Read the index and metadata back. Refuses a pair that has drifted."""
    index = faiss.read_index(str(INDEX_PATH))
    meta = [json.loads(line) for line in META_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if index.ntotal != len(meta):
        raise ValueError(f"index has {index.ntotal} vectors but {len(meta)} metadata rows")
    return index, meta


def search(
    query: str,
    top_k: int = 5,
    *,
    index: faiss.Index | None = None,
    meta: list[dict] | None = None,
    size: int | None = None,
) -> list[dict]:
    """The nearest chunks to a query, best first.

    `size` keeps only the chunks cut at that many tokens, so the 128 and the
    512 indexes can be compared from the one file. Scores are cosine
    similarities, because the stored vectors are normalised.
    """
    if index is None or meta is None:
        index, meta = load()

    # Which positions are eligible. Filtering before the search keeps the
    # cut-off honest: top 5 means 5 chunks of the requested size.
    positions = [i for i, row in enumerate(meta) if size is None or row["size"] == size]
    if not positions or top_k <= 0:
        return []

    query_vector = embed([query])
    scores, ids = index.search(np.ascontiguousarray(query_vector, dtype=np.float32), index.ntotal)

    allowed = set(positions)
    hits = []
    for score, position in zip(scores[0], ids[0]):
        position = int(position)
        if position < 0 or position not in allowed:
            continue
        hit = dict(meta[position])
        hit["score"] = float(score)
        hits.append(hit)
        if len(hits) == top_k:
            break
    return hits


def main() -> None:
    chunks = load_chunks()
    index, meta = build(chunks)
    save(index, meta)

    sizes = sorted({row["size"] for row in meta})
    print(f"vectors: {index.ntotal}")
    print(f"dim:     {index.d}")
    print(f"type:    {type(index).__name__}")
    for size in sizes:
        print(f"size {size}: {sum(1 for row in meta if row['size'] == size)} vectors")
    print(f"wrote {INDEX_PATH.name} and {META_PATH.name} to {INDEX_DIR}")

    # One query, to show the index actually returns the chunk it should.
    hits = search("What does ERR-VPN-809 mean?", top_k=3, index=index, meta=meta, size=128)
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank}. {hit['chunk_id']}  {hit['score']:.3f}")


if __name__ == "__main__":
    main()
