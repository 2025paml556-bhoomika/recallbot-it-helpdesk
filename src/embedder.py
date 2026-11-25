"""Embed text with all-MiniLM-L6-v2.

The model mean-pools token vectors and this module L2-normalises each row,
so a later cosine search is just a dot product. Every successful call is
appended to logs/calls.jsonl; a failed call writes nothing.

    python -m src.embedder
"""

from __future__ import annotations

import time

import numpy as np

from src.config import EMBED_BATCH_SIZE, EMBED_DIM, EMBED_MODEL
from src.logger import log_call

_model = None


def load_model():
    """Load the embedding model once and keep it for later calls."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def count_tokens(texts: list[str]) -> int:
    """Token count from the embedding model's own tokenizer."""
    tokenizer = load_model().tokenizer
    encoded = tokenizer(texts, add_special_tokens=True, verbose=False)
    return sum(len(ids) for ids in encoded["input_ids"])


def embed(
    texts: list[str],
    *,
    config: str | None = None,
    user: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    log: bool = True,
) -> np.ndarray:
    """Return one normalised vector per text, shape (len(texts), EMBED_DIM).

    An empty input returns an empty (0, EMBED_DIM) array and logs nothing,
    because no embedding call was made.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    model = load_model()
    # Token counting loads the tokenizer, so it is skipped when the caller
    # has asked for no log line. There is nothing to count for.
    input_tokens = count_tokens(texts) if log else 0
    start = time.time()
    # normalize_embeddings does the L2 normalisation. The model's default
    # pooling is mean pooling, which is what all-MiniLM-L6-v2 was trained with.
    vectors = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    # Log only after the call succeeds, so a failure leaves no line. A caller
    # that is re-embedding stored memory passes log=False: that work was
    # already logged when the vector was first computed.
    if not log:
        return vectors
    log_call(
        call_type="embed",
        model=EMBED_MODEL,
        input_tokens=input_tokens,
        output_tokens=0,
        start=start,
        config=config,
        user=user,
        session=session,
        turn=turn,
        extra={"n_texts": len(texts)},
    )
    return vectors


def main() -> None:
    samples = [
        "The VPN drops with error ERR-VPN-809.",
        "How do I reset my laptop password?",
        "A priority one incident has a one hour response.",
    ]
    vectors = embed(samples)
    norms = np.linalg.norm(vectors, axis=1)
    print(f"model:  {EMBED_MODEL}")
    print(f"shape:  {vectors.shape}")
    print(f"dtype:  {vectors.dtype}")
    print(f"norms:  {np.round(norms, 6).tolist()}")


if __name__ == "__main__":
    main()
