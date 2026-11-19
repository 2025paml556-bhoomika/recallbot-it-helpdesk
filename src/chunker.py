"""Split the knowledge base into fixed-size token chunks.

Two sizes, both with about 15% overlap, using the chat model's tokenizer so
the chunk counts match the 4,096-token context budget counted later.

    python -m src.chunker

Writes one JSON object per chunk to data/chunks.jsonl.
"""

from __future__ import annotations

import json
import re
import sys

from src.config import ROOT, TOKENIZER_MODEL
from src.kb_stats import HEADER_FIELDS, KB_DIR, parse_header

CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

# Two chunk sizes, each with an overlap of about 15% of the size.
CHUNK_SIZES = (128, 512)
OVERLAP_RATIO = 0.15

HEADING_RE = re.compile(r"^#{1,3} +(.+)$", re.MULTILINE)


def load_tokenizer():
    """The same tokenizer the chat model uses, so token counts agree."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def overlap_for(size: int) -> int:
    return round(size * OVERLAP_RATIO)


def _section_at(text: str, offset: int) -> str:
    """The Markdown heading in force at this point in the document.

    A chunk may open on the heading line itself, so the heading is taken from
    the last match whose start is at or before `offset`, not strictly before.
    """
    heading = ""
    for match in HEADING_RE.finditer(text):
        if match.start() > offset:
            break
        heading = match.group(1).strip()
    return heading


def chunk_document(text: str, tokenizer, size: int, overlap: int) -> list[dict]:
    """Cut one document into chunks of `size` tokens, stepping by size-overlap."""
    header = parse_header(text)
    # The header is metadata, not content, so it is not part of any chunk.
    body = text.split("\n", len(header))[-1].lstrip("\n")

    # Character offsets let us recover the section heading for each span.
    encoded = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    step = size - overlap
    chunks = []
    index = 0
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        char_start = offsets[start][0]
        # The span starts at the first token, so it begins mid-word whenever the
        # overlap cut falls inside one. Extend back to the word boundary.
        while char_start > 0 and body[char_start - 1].isalnum():
            char_start -= 1
        char_end = offsets[end - 1][1]
        # The span ends at the last token, so it stops mid-word when the cut
        # falls inside one. Extend forward to the word boundary.
        while char_end < len(body) and body[char_end].isalnum():
            char_end += 1
        # A sentence cut off right after its last word loses the closing
        # punctuation; keep it, since gold spans include the full stop.
        if char_end < len(body) and body[char_end] in ".!?":
            char_end += 1
        chunks.append(
            {
                "chunk_id": f"{header['doc_id']}@{header['version']}:{size}:{index}",
                "doc_id": header["doc_id"],
                "version": header["version"],
                "valid_from": header["valid_from"],
                "status": header["status"],
                "section": _section_at(body, char_start),
                "size": size,
                "index": index,
                "token_count": end - start,
                "text": body[char_start:char_end],
            }
        )
        index += 1
        if end == len(tokens):
            break
        start += step
    return chunks


def build_chunks(tokenizer=None) -> list[dict]:
    tokenizer = tokenizer or load_tokenizer()
    chunks: list[dict] = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for size in CHUNK_SIZES:
            chunks.extend(chunk_document(text, tokenizer, size, overlap_for(size)))
    return chunks


def main() -> int:
    chunks = build_chunks()
    lines = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
    CHUNKS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for size in CHUNK_SIZES:
        n = sum(1 for c in chunks if c["size"] == size)
        print(f"size {size}: {n} chunks (overlap {overlap_for(size)} tokens)")
    print(f"wrote {len(chunks)} chunks to {CHUNKS_PATH.relative_to(ROOT)}")

    # Every chunk must carry the fields the rest of the pipeline relies on.
    # `section` may legitimately be empty when a chunk begins before the first
    # heading, so it is checked for presence rather than for being non-empty.
    required = ("chunk_id", "text", "token_count", *HEADER_FIELDS)
    bad = [
        c["chunk_id"]
        for c in chunks
        if not all(c.get(f) for f in required) or "section" not in c
    ]
    if bad:
        print(f"chunks missing metadata: {bad[:5]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
