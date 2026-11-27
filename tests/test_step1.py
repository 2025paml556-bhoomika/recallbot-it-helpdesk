"""Step 1 checks from TESTING.md: T1.2–T1.8.

T1.1 (pip install) is run outside pytest and recorded in the step report.
T1.4–T1.7 talk to the local Ollama model and are marked `live`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src import config
from src.llm import LLMError, chat
from src.logger import REQUIRED_FIELDS

ROOT = config.ROOT
LOG_PATH = config.LOG_PATH


def _read_log() -> list[str]:
    if not LOG_PATH.exists():
        return []
    return LOG_PATH.read_text(encoding="utf-8").splitlines()


def _sha256_lines(lines: list[str]) -> str:
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_t1_2_required_folders_exist():
    missing = [str(path.relative_to(ROOT)) for path in config.REQUIRED_DIRS if not path.is_dir()]
    assert missing == []


def test_t1_3_config_constants():
    assert config.K_TURNS == 4
    assert config.CONTEXT_WINDOW == 4096
    assert isinstance(config.CHAT_MODEL, str) and config.CHAT_MODEL.strip()
    assert isinstance(config.RUN_ID, str) and config.RUN_ID.strip()
    # One model name is shared; the tokenizer id is the matching checkpoint.
    assert config.CHAT_MODEL == "qwen3:4b-instruct"
    assert config.TOKENIZER_MODEL == "Qwen/Qwen2.5-3B"


@pytest.mark.live
def test_t1_4_to_t1_7_chat_call_logs_usage_and_never_edits(tmp_path, monkeypatch):
    """T1.4 reply, T1.5 one complete line, T1.6 usage tokens, T1.7 append-only."""
    before = _read_log()
    before_hash = _sha256_lines(before)

    reply = chat("Say hello")
    assert isinstance(reply, str) and reply.strip()

    after_first = _read_log()
    assert len(after_first) == len(before) + 1
    new_line = after_first[-1]
    record = json.loads(new_line)
    for field in ("ts", "run_id", "call_type", "model", "input_tokens", "output_tokens", "latency_ms"):
        assert field in record
        assert record[field] not in (None, "")
    assert record["call_type"] == "chat"
    assert record["model"] == config.CHAT_MODEL
    assert record["run_id"] == config.RUN_ID
    assert isinstance(record["input_tokens"], int) and record["input_tokens"] > 0
    assert isinstance(record["output_tokens"], int) and record["output_tokens"] > 0
    assert isinstance(record["latency_ms"], int) and record["latency_ms"] >= 0
    # T1.6: the logged counts are the API usage field, copied through unchanged.
    usage = record["usage"]
    assert record["input_tokens"] == usage["prompt_eval_count"]
    assert record["output_tokens"] == usage["eval_count"]
    for field in REQUIRED_FIELDS:
        assert field in record

    # T1.7: three further calls only append. The earlier bytes do not change.
    frozen = list(after_first)
    frozen_hash = _sha256_lines(frozen)
    for _ in range(3):
        extra = chat("Say hello again in one short sentence.")
        assert extra.strip()
    grown = _read_log()
    assert len(grown) == len(frozen) + 3
    assert grown[: len(frozen)] == frozen
    assert _sha256_lines(grown[: len(frozen)]) == frozen_hash
    assert _sha256_lines(before) == before_hash


@pytest.mark.live
def test_t1_8_missing_model_logs_nothing():
    before = _read_log()
    with pytest.raises(LLMError, match="qwen2.5:does-not-exist"):
        chat("Say hello", model="qwen2.5:does-not-exist")
    assert _read_log() == before
