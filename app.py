"""Small command line front for the knowledge base chunker.

    .venv/bin/python app.py                 # chunk every document
    .venv/bin/python app.py sla.md          # chunk one document and print it
    .venv/bin/python app.py sla.md --size 128
"""

from __future__ import annotations

import argparse
import sys

from src.chunker import (
    CHUNK_SIZES,
    build_chunks,
    chunk_document,
    load_tokenizer,
    main as chunk_all,
    overlap_for,
)
from src.kb_stats import KB_DIR


def chunk_one(name: str, sizes: tuple[int, ...]) -> int:
    path = KB_DIR / name
    if not path.exists():
        names = ", ".join(sorted(p.name for p in KB_DIR.glob("*.md")))
        print(f"no such document: {name}\navailable: {names}", file=sys.stderr)
        return 1

    tokenizer = load_tokenizer()
    text = path.read_text(encoding="utf-8")
    for size in sizes:
        chunks = chunk_document(text, tokenizer, size, overlap_for(size))
        print(f"\n===== {name} at {size} tokens ({len(chunks)} chunks) =====")
        for chunk in chunks:
            print(f"[{chunk['index']}] {chunk['token_count']:3d} tok  {chunk['section']!r}")
            print("   start:", repr(chunk["text"][:70]))
            print("   end  :", repr(chunk["text"][-70:]))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk the knowledge base.")
    parser.add_argument("document", nargs="?", help="one file in data/kb, e.g. sla.md")
    parser.add_argument(
        "--size",
        type=int,
        choices=CHUNK_SIZES,
        help="chunk size to use (default: both)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sizes = (args.size,) if args.size else CHUNK_SIZES
    if args.document:
        sys.exit(chunk_one(args.document, sizes))
    # No document given: chunk everything and write data/chunks.jsonl.
    # build_chunks is what main() runs; importing it keeps the entry point
    # available for reuse without shelling out.
    build_chunks
    sys.exit(chunk_all())
