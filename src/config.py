"""Run configuration for RecallBot V06.

Later steps add their own constants (database, skills) when they start.
Change RUN_ID at the start of each run; never reuse an id.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (this file lives in src/).
ROOT = Path(__file__).resolve().parent.parent

# Short-term window and model context, fixed by the variant brief.
K_TURNS = 4
CONTEXT_WINDOW = 4096

# Chat model. Ollama serves it locally; the Hugging Face id is the matching
# tokenizer used when an API response has no usage field.
CHAT_MODEL = "qwen2.5:3b"
TOKENIZER_MODEL = "Qwen/Qwen2.5-3B"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Change this string at the start of every run. Do not edit past log lines.
RUN_ID = os.environ.get("RECALLBOT_RUN_ID", "r01")

# Embeddings. all-MiniLM-L6-v2 mean-pools to 384 dimensions; embedder.py
# L2-normalises each vector so cosine similarity is a dot product.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
EMBED_BATCH_SIZE = 32

# Paths Step 1 actually writes or checks.
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "calls.jsonl"

# Chunks and the dense index built from them (Step 4).
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
INDEX_DIR = ROOT / "index"

# Folders Step 1 must create. Kept here so the check and the README agree.
REQUIRED_DIRS = (
    ROOT / "src",
    ROOT / "skills",
    ROOT / "tests",
    ROOT / "data" / "kb",
    ROOT / "db",
    ROOT / "index",
    ROOT / "memory",
    LOG_DIR,
    ROOT / "results" / "plots",
)
