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
#
# The default is qwen3:4b-instruct, the non-thinking build. Plain qwen3:4b is
# the thinking-only build and spends thousands of tokens reasoning per call,
# and no setting turns that off. To use another model that Ollama has, set
# RECALLBOT_CHAT_MODEL, for example RECALLBOT_CHAT_MODEL=qwen2.5:3b. The logs
# record the model used, so runs made with different models are not mixed up.
#
# TOKENIZER_MODEL is only used to count tokens against the prompt budget.
# Qwen3 uses the same vocabulary as Qwen2.5, so the Qwen2.5 tokenizer counts
# its prompts correctly and no second tokenizer has to be downloaded.
CHAT_MODEL = os.environ.get("RECALLBOT_CHAT_MODEL", "qwen3:4b-instruct")
TOKENIZER_MODEL = "Qwen/Qwen2.5-3B"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Change this string at the start of every run. Do not edit past log lines.
RUN_ID = os.environ.get("RECALLBOT_RUN_ID", "r01")

# What call_type a new log line may use. Old lines in logs/calls.jsonl use
# "reply", "summarise" and "embedding"; those names are not written any more,
# and the old lines are not rewritten. extract, importance, reflect and judge
# are the steps that do not exist yet. Listing them here means a later step
# can log one without inventing a new name.
CALL_TYPES = (
    "chat",
    "nlu",
    "embed",
    "tool",
    "summary",
    "extract",
    "importance",
    "reflect",
    "judge",
)

# The configuration a run is measuring. Short-term memory is full, trim or
# summary. The ablation runs are no_rag, hybrid, hybrid_sem and hybrid_mem.
# A new log line carries one of these. A line written before this field was
# filled in has config null, and is left that way.
CONFIGS = (
    "full",
    "trim",
    "summary",
    "no_rag",
    "hybrid",
    "hybrid_sem",
    "hybrid_mem",
)

# The ablation. Each one adds exactly one thing to the one before it:
# no_rag is the short-term window alone, hybrid adds the knowledge base,
# hybrid_sem adds the fact store, hybrid_mem adds the episode store.
ABLATION_CONFIGS = ("no_rag", "hybrid", "hybrid_sem", "hybrid_mem")

# Which memory type each ablation config may read. Writing is not gated:
# a fact stated under no_rag is still stored, so the later config can be
# compared against the same store. Only the read changes.
ABLATION_USES = {
    "no_rag": {"kb": False, "semantic": False, "episodic": False},
    "hybrid": {"kb": True, "semantic": False, "episodic": False},
    "hybrid_sem": {"kb": True, "semantic": True, "episodic": False},
    "hybrid_mem": {"kb": True, "semantic": True, "episodic": True},
}

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

# Helpdesk database (Step 5). Recreated by `python -m src.db`.
DB_DIR = ROOT / "db"
DB_PATH = DB_DIR / "helpdesk.sqlite"

# Long-term memory (Steps 7 and 8). One JSONL stream each, shared by every
# user; every read filters on the user column. The vectors sit beside the
# stream so a forget can drop the row and its vector together.
MEMORY_DIR = ROOT / "memory"
SEMANTIC_PATH = MEMORY_DIR / "semantic.jsonl"
SEMANTIC_INDEX_PATH = MEMORY_DIR / "semantic.faiss"
SEMANTIC_IDS_PATH = MEMORY_DIR / "semantic_ids.json"
EPISODIC_PATH = MEMORY_DIR / "episodic.jsonl"
EPISODIC_INDEX_PATH = MEMORY_DIR / "episodic.faiss"
EPISODIC_IDS_PATH = MEMORY_DIR / "episodic_ids.json"

# Working memory is one file per session, thrown away when the session ends.
WORKING_DIR = MEMORY_DIR / "working"

# The transcript of each conversation. One JSON file per (user, session),
# kept after the session ends. Not the fact store: semantic.jsonl is that.
SESSIONS_DIR = MEMORY_DIR / "sessions"

# How many past episodes a turn may load. The rest stay on disk.
EPISODIC_TOP_K = 3
# Cosine below this is not relevant. A user with no related history gets
# nothing back, not their least-unrelated episode.
MEMORY_RELEVANCE = 0.30

# Episodic score = ALPHA * recency + BETA * importance + GAMMA * cosine.
# Recency is exponential decay per hour. Importance is stored 1-10 and
# scaled to 0-1 before it is added, so the three terms share a scale.
EPISODIC_ALPHA = 1.0
EPISODIC_BETA = 1.0
EPISODIC_GAMMA = 1.0
EPISODIC_DECAY = 0.995

# The rolling summary is capped so it cannot crowd the retrieved passage.
WORKING_SUMMARY_TOKENS = 150

# The chat context is 4,096 tokens. About 500 are kept back for the reply,
# so the assembled prompt must stay at or under this. Asserted on every turn.
REPLY_RESERVE_TOKENS = 500
PROMPT_BUDGET = CONTEXT_WINDOW - REPLY_RESERVE_TOKENS

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
