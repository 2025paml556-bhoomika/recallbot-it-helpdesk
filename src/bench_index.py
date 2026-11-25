"""Size the index after one year, then time flat search against HNSW.

Task 5(b). The byte counts use the FAISS guideline formula, not a measurement
of the file on disk:

    flat float32:  d * 4
    HNSW M:        d * 4 + M * 2 * 4

(https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)

The one-year count is the knowledge-base chunks plus one year's semantic and
episodic memories for the variant's 100,000 daily users. How many memories a
user keeps is counted from data/test_conversation.json, which is the one
conversation the brief says to scale up. It is not a guess typed in here.

Latency is measured on at least 100,000 vectors. The real chunks are embedded
once and the rest of the index is random vectors on the unit sphere, so the
timing is about the index and not about the embedder. Recall is measured only
on the real chunks: HNSW's top-k against the flat index's top-k, for the
questions in data/kb_questions.json. Flat compares every vector, so its
recall against itself is 1. HNSW recall cannot exceed 1.

    python -m src.bench_index
    python -m src.bench_index --n 100000 --queries 50

Writes results/bench_index.csv. Nothing is appended to logs/calls.jsonl:
the padding vectors are random, and the chunk embeddings are computed with
logging off because the same calls were already logged when the index was
built.
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import faiss
import numpy as np

from src.config import EMBED_DIM, ROOT
from src.embedder import embed
from src.eval_retrieval import load_questions
from src.vector_index import load_chunks

# The variant. 100,000 conversations a day, each one the length of the test
# conversation, for a year.
DAILY_USERS = 100_000
DAYS = 365

# HNSW. M is the number of neighbours stored per vector. ef_search is the
# candidate list walked at query time; ef_construction is the same list used
# while the graph is built. The three ef values are what the recommendation
# is chosen from.
HNSW_M = 16
EF_CONSTRUCTION = 40
EF_SEARCHES = (16, 32, 64)

# Recall and the reported latency use the top 5, matching eval_retrieval.
TOP_K = 5

# Below this many vectors the timing says nothing about the 100,000 target.
MIN_VECTORS = 100_000

OUT_PATH = ROOT / "results" / "bench_index.csv"

# One query is embedded once and reused, so the latency column is the search,
# not the embedding call.
_QUERY = "What does ERR-VPN-809 mean?"


def bytes_per_vector(dim: int = EMBED_DIM, m: int = HNSW_M) -> dict[str, int]:
    """Bytes for one stored vector, flat and HNSW, at this M.

    Flat stores the float32 vector and nothing else. HNSW stores the same
    vector plus M links out and the matching links back, each link an int32,
    which is the M * 2 * 4 term in the FAISS formula.
    """
    flat = dim * 4
    hnsw = dim * 4 + m * 2 * 4
    return {"flat": flat, "hnsw": hnsw}


def conversation_rates(conversation: dict | None = None) -> dict[str, int]:
    """Facts and episodes one user plants in the test conversation.

    Counted for U1 only. U2 appears in the same file, but the figure being
    scaled is "one user, one conversation", and adding U2's facts would
    double-count a single user's store.

    A fact is one key/value stated, and an update adds one more: the old
    value is kept and marked superseded, so it still takes a row and a
    vector. A forget is a row written and then removed, so it counts in the
    planted total but not in the stored total. An episode is one tagged
    event; the same event listed twice would count once.
    """
    if conversation is None:
        path = ROOT / "data" / "test_conversation.json"
        conversation = json.loads(path.read_text(encoding="utf-8"))

    facts = 0
    forgotten = 0
    seen_events: set[tuple[str, str]] = set()
    for turn in conversation["turns"]:
        if turn["user"] != "U1":
            continue
        facts += len(turn.get("facts", []))
        for update in turn.get("updates", []):
            facts += 1
        forgotten += len(turn.get("forgets", []))
        for event in turn.get("events", []):
            seen_events.add((event["type"], event["summary"]))
    return {
        "facts_per_user": facts,
        "forgotten_per_user": forgotten,
        "facts_stored_per_user": facts - forgotten,
        "episodes_per_user": len(seen_events),
    }


def estimate_vectors(
    n_chunks: int,
    *,
    users: int = DAILY_USERS,
    days: int = DAYS,
    rates: dict[str, int] | None = None,
) -> dict[str, int]:
    """Vectors after one year: chunks, plus every user's facts and episodes.

    The knowledge base is shared, so its chunks are counted once. Memory is
    not shared: each of the daily users has their own.
    """
    rates = conversation_rates() if rates is None else rates
    semantic = users * rates["facts_stored_per_user"]
    episodic = users * days * rates["episodes_per_user"]
    return {
        "kb_chunks": n_chunks,
        "semantic": semantic,
        "episodic": episodic,
        "total": n_chunks + semantic + episodic,
        **rates,
    }


def estimate_bytes(n_vectors: int, dim: int = EMBED_DIM, m: int = HNSW_M) -> dict[str, int]:
    """Bytes for an index of n_vectors. Same formula as bytes_per_vector."""
    per = bytes_per_vector(dim, m)
    return {kind: n_vectors * size for kind, size in per.items()}


def make_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    """n random unit vectors. Inner product on these is cosine similarity."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def build_flat(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_hnsw(vectors: np.ndarray, m: int = HNSW_M, ef_construction: int = EF_CONSTRUCTION) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def _query_matrix(n_queries: int, dim: int) -> np.ndarray:
    """One embedded question, repeated, so every timed query is the same."""
    vector = embed([_QUERY], log=False).astype(np.float32)
    return np.repeat(vector, n_queries, axis=0)


def measure_latency(
    index: faiss.Index,
    queries: np.ndarray,
    *,
    top_k: int = TOP_K,
    repeats: int = 3,
) -> float:
    """Median per-query latency in milliseconds, over `repeats` passes.

    The first search is discarded: HNSW's first call pays for page faults
    that later calls do not.
    """
    index.search(queries[:1], top_k)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        index.search(queries, top_k)
        elapsed = time.perf_counter() - start
        samples.append(elapsed * 1000.0 / len(queries))
    samples.sort()
    return samples[len(samples) // 2]


def recall_at_k(flat_ids: np.ndarray, other_ids: np.ndarray, k: int) -> float:
    """Share of the flat top-k that the other index also returned, mean over queries.

    Order does not matter and duplicates do not count twice. A perfect index
    scores 1. Nothing scores more than 1, because the denominator is the
    flat result itself.
    """
    scores = []
    for flat_row, other_row in zip(flat_ids, other_ids):
        truth = {int(i) for i in flat_row[:k] if int(i) >= 0}
        if not truth:
            continue
        found = sum(1 for i in other_row[:k] if int(i) in truth)
        scores.append(found / len(truth))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def embedding_recall(
    questions: list[dict],
    *,
    m: int = HNSW_M,
    ef_search: int = 64,
    ef_construction: int = EF_CONSTRUCTION,
    top_k: int = TOP_K,
    size: int = 128,
) -> dict[str, float]:
    """HNSW recall against flat on the real chunk embeddings.

    Only the chunks of one size are indexed, so a question's neighbours are
    the same chunks eval_retrieval scores. The questions are embedded with
    logging off.
    """
    chunks = [chunk for chunk in load_chunks() if chunk["size"] == size]
    if len(chunks) < top_k:
        raise ValueError(f"need at least {top_k} chunks of size {size}, got {len(chunks)}")

    vectors = embed([chunk["text"] for chunk in chunks], log=False).astype(np.float32)
    flat = build_flat(vectors)
    hnsw = build_hnsw(vectors, m=m, ef_construction=ef_construction)
    hnsw.hnsw.efSearch = ef_search

    queries = embed([q["question"] for q in questions], log=False).astype(np.float32)
    _, flat_ids = flat.search(queries, top_k)
    _, hnsw_ids = hnsw.search(queries, top_k)
    return {
        "flat": recall_at_k(flat_ids, flat_ids, top_k),
        "hnsw": recall_at_k(flat_ids, hnsw_ids, top_k),
        "n_chunks": float(len(chunks)),
        "n_questions": float(len(questions)),
    }


def recommend(recall_by_ef: dict[int, float], target: float = 0.95) -> dict[str, int | float]:
    """The smallest ef_search that reaches `target` recall, else the best one.

    M stays at 16: the brief asks for the M=16 figure, and raising M grows
    every vector's bytes for a recall gain that ef_search already buys.
    """
    chosen = max(EF_SEARCHES)
    for ef in EF_SEARCHES:
        if recall_by_ef[ef] >= target:
            chosen = ef
            break
    return {"M": HNSW_M, "ef_search": chosen, "recall": recall_by_ef[chosen]}


def run(
    n: int = MIN_VECTORS,
    n_queries: int = 50,
    seed: int = 0,
) -> dict:
    """Estimates plus one latency/recall measurement. See the module docstring."""
    if n < MIN_VECTORS:
        raise ValueError(f"n must be at least {MIN_VECTORS}, got {n}")

    chunks = load_chunks()
    # The shipped index holds both chunk sizes. The one-year count uses that
    # figure, since it is the index that would actually be kept.
    yearly = estimate_vectors(len(chunks))
    per_vector = bytes_per_vector()
    yearly_bytes = estimate_bytes(yearly["total"])

    real = embed(
        [chunk["text"] for chunk in chunks if chunk["size"] == 128],
        log=False,
    ).astype(np.float32)
    padding = make_vectors(n - len(real), EMBED_DIM, seed)
    vectors = np.vstack([real, padding])

    flat = build_flat(vectors)
    hnsw = build_hnsw(vectors)
    queries = _query_matrix(n_queries, EMBED_DIM)

    flat_ms = measure_latency(flat, queries)
    latency = {"flat": flat_ms}
    recall_by_ef: dict[int, float] = {}
    _, flat_ids = flat.search(queries, TOP_K)
    for ef in EF_SEARCHES:
        hnsw.hnsw.efSearch = ef
        latency[f"hnsw_ef{ef}"] = measure_latency(hnsw, queries)
        _, hnsw_ids = hnsw.search(queries, TOP_K)
        recall_by_ef[ef] = recall_at_k(flat_ids, hnsw_ids, TOP_K)

    questions = load_questions()
    question_recall = {
        ef: embedding_recall(questions, ef_search=ef) for ef in EF_SEARCHES
    }
    choice = recommend({ef: question_recall[ef]["hnsw"] for ef in EF_SEARCHES})

    return {
        "dim": EMBED_DIM,
        "M": HNSW_M,
        "ef_construction": EF_CONSTRUCTION,
        "daily_users": DAILY_USERS,
        "days": DAYS,
        "yearly": yearly,
        "bytes_per_vector": per_vector,
        "yearly_bytes": yearly_bytes,
        "bench_vectors": int(vectors.shape[0]),
        "latency_ms": latency,
        "recall_padded": recall_by_ef,
        "recall_questions": {ef: question_recall[ef]["hnsw"] for ef in EF_SEARCHES},
        "flat_recall": question_recall[EF_SEARCHES[0]]["flat"],
        "n_questions": int(question_recall[EF_SEARCHES[0]]["n_questions"]),
        "recommendation": choice,
    }


def write_csv(result: dict, path=OUT_PATH) -> None:
    """One row per index configuration, so the report table is a straight copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "M",
        "ef_search",
        "vectors",
        "bytes_per_vector",
        "yearly_vectors",
        "yearly_bytes",
        "latency_ms",
        "recall_at_5_padded",
        "recall_at_5_questions",
    ]
    yearly = result["yearly"]["total"]
    rows = [
        {
            "index": "flat",
            "M": "",
            "ef_search": "",
            "vectors": result["bench_vectors"],
            "bytes_per_vector": result["bytes_per_vector"]["flat"],
            "yearly_vectors": yearly,
            "yearly_bytes": result["yearly_bytes"]["flat"],
            "latency_ms": round(result["latency_ms"]["flat"], 4),
            "recall_at_5_padded": 1.0,
            "recall_at_5_questions": result["flat_recall"],
        }
    ]
    for ef in EF_SEARCHES:
        rows.append(
            {
                "index": "hnsw",
                "M": result["M"],
                "ef_search": ef,
                "vectors": result["bench_vectors"],
                "bytes_per_vector": result["bytes_per_vector"]["hnsw"],
                "yearly_vectors": yearly,
                "yearly_bytes": result["yearly_bytes"]["hnsw"],
                "latency_ms": round(result["latency_ms"][f"hnsw_ef{ef}"], 4),
                "recall_at_5_padded": round(result["recall_padded"][ef], 4),
                "recall_at_5_questions": round(result["recall_questions"][ef], 4),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Flat vs HNSW size, latency and recall")
    parser.add_argument("--n", type=int, default=MIN_VECTORS, help="vectors in the timed index")
    parser.add_argument("--queries", type=int, default=50, help="queries per latency pass")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run(n=args.n, n_queries=args.queries, seed=args.seed)
    write_csv(result)

    yearly = result["yearly"]
    print(f"dim: {result['dim']}")
    print(
        f"one year: {yearly['kb_chunks']} chunks + "
        f"{yearly['semantic']} facts + {yearly['episodic']} episodes "
        f"= {yearly['total']} vectors"
    )
    print(
        f"  per user: {yearly['facts_stored_per_user']} facts kept "
        f"({yearly['forgotten_per_user']} forgotten), "
        f"{yearly['episodes_per_user']} episodes per conversation"
    )
    print(
        f"bytes/vector: flat {result['bytes_per_vector']['flat']}, "
        f"HNSW M={result['M']} {result['bytes_per_vector']['hnsw']}"
    )
    print(
        f"one year on disk: flat {_gb(result['yearly_bytes']['flat'])}, "
        f"HNSW {_gb(result['yearly_bytes']['hnsw'])}"
    )
    print(f"timed on {result['bench_vectors']} vectors, top {TOP_K}:")
    print(f"  flat            {result['latency_ms']['flat']:.3f} ms/query")
    for ef in EF_SEARCHES:
        print(
            f"  hnsw ef={ef:<4}    {result['latency_ms'][f'hnsw_ef{ef}']:.3f} ms/query"
            f"   recall@padded {result['recall_padded'][ef]:.3f}"
            f"   recall@questions {result['recall_questions'][ef]:.3f}"
        )
    choice = result["recommendation"]
    print(
        f"flat recall on the questions: {result['flat_recall']:.3f} "
        f"({result['n_questions']} questions)"
    )
    print(
        f"recommend HNSW M={choice['M']} ef_search={choice['ef_search']} "
        f"(question recall {choice['recall']:.3f})"
    )
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
