"""Run configuration for RecallBot V06.

One chat model is used for every configuration so later comparisons stay fair.
Change RUN_ID at the start of each run; never reuse an id for a new run.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (this file lives in src/).
ROOT = Path(__file__).resolve().parent.parent

# Short-term window and model context, fixed by the variant brief.
K_TURNS = 4
CONTEXT_WINDOW = 4096
REPLY_RESERVE_TOKENS = 500

# Chat model. Ollama serves it locally; the Hugging Face id is the matching
# tokenizer used when an API response has no usage field.
CHAT_BACKEND = "ollama"
CHAT_MODEL = "qwen2.5:3b"
TOKENIZER_MODEL = "Qwen/Qwen2.5-3B"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Embeddings (used from Step 4 onwards; named here so every step shares one config).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Change this string at the start of every run. Do not edit past log lines.
RUN_ID = os.environ.get("RECALLBOT_RUN_ID", "r01")

# Paths. Directories are created by setup; files are written at run time.
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "kb"
DB_DIR = ROOT / "db"
DB_PATH = DB_DIR / "helpdesk.sqlite"
INDEX_DIR = ROOT / "index"
MEMORY_DIR = ROOT / "memory"
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "calls.jsonl"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SKILLS_DIR = ROOT / "skills"

# Folders Step 1 must create. Kept here so the check and the README agree.
REQUIRED_DIRS = (
    ROOT / "src",
    SKILLS_DIR,
    ROOT / "tests",
    KB_DIR,
    DB_DIR,
    INDEX_DIR,
    MEMORY_DIR,
    LOG_DIR,
    PLOTS_DIR,
)
