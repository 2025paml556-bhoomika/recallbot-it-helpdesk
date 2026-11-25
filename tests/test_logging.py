"""The log vocabulary and the config field, for lines written from now on.

Old lines are not rewritten. What is checked is that a new line cannot use
the retired names, that a new line can use each current name, and that the
config it carries is one of the seven the runs measure.
"""

from __future__ import annotations

import json

import pytest

from src.config import CALL_TYPES, CONFIGS
from src.logger import log_call


def test_the_vocabulary_is_the_one_the_brief_names():
    assert set(CALL_TYPES) == {
        "chat",
        "nlu",
        "embed",
        "tool",
        "summary",
        "extract",
        "importance",
        "reflect",
        "judge",
    }
    # The three names the existing log lines use. A new line must not.
    for retired in ("reply", "summarise", "embedding"):
        assert retired not in CALL_TYPES


def test_the_seven_configs():
    assert CONFIGS == (
        "full",
        "trim",
        "summary",
        "no_rag",
        "hybrid",
        "hybrid_sem",
        "hybrid_mem",
    )


def test_a_new_line_uses_the_current_name_and_a_config(tmp_path, monkeypatch):
    import src.logger as logger

    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(logger, "LOG_PATH", path)

    written = log_call(
        "chat",
        model="qwen2.5:3b",
        input_tokens=10,
        output_tokens=2,
        start=0.0,
        config="trim",
        user="U1",
        session="A",
        turn=3,
    )
    assert written["call_type"] == "chat"
    assert written["config"] == "trim"

    line = json.loads(path.read_text(encoding="utf-8"))
    assert line["call_type"] == "chat"
    assert line["config"] == "trim"
    assert line["user"] == "U1"
    assert line["turn"] == 3


def test_every_current_call_type_can_be_logged(tmp_path, monkeypatch):
    import src.logger as logger

    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(logger, "LOG_PATH", path)
    for call_type in CALL_TYPES:
        log_call(call_type, model="m", input_tokens=0, output_tokens=0, start=0.0, config="hybrid")
    logged = [json.loads(line)["call_type"] for line in path.read_text().splitlines()]
    assert logged == list(CALL_TYPES)


def test_a_retired_name_is_refused_and_writes_nothing(tmp_path, monkeypatch):
    import src.logger as logger

    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(logger, "LOG_PATH", path)
    for retired in ("reply", "summarise", "embedding"):
        with pytest.raises(ValueError):
            log_call(retired, model="m", input_tokens=1, output_tokens=1, start=0.0, config="full")
    assert not path.exists()


def test_an_unknown_config_is_refused(tmp_path, monkeypatch):
    import src.logger as logger

    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(logger, "LOG_PATH", path)
    with pytest.raises(ValueError):
        log_call("chat", model="m", input_tokens=1, output_tokens=1, start=0.0, config="window")
    assert not path.exists()


def test_a_call_outside_a_measured_run_may_omit_the_config(tmp_path, monkeypatch):
    """A smoke test has no configuration. None is allowed; a typo is not."""
    import src.logger as logger

    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(logger, "LOG_PATH", path)
    log_call("chat", model="m", input_tokens=1, output_tokens=1, start=0.0)
    assert json.loads(path.read_text())["config"] is None


def test_plots_read_the_old_names_as_the_new_ones():
    """A log line is not rewritten, so the plot has to understand both."""
    from src.plots import canonical_call_type, totals

    assert canonical_call_type({"call_type": "reply"}) == "chat"
    assert canonical_call_type({"call_type": "summarise"}) == "summary"
    assert canonical_call_type({"call_type": "embedding"}) == "embed"
    assert canonical_call_type({"call_type": "chat"}) == "chat"

    calls = [
        {"run_id": "stm-trim", "call_type": "reply", "input_tokens": 100},
        {"run_id": "stm-trim", "call_type": "chat", "input_tokens": 40},
        {"run_id": "stm-summary", "call_type": "summarise", "input_tokens": 30},
        {"run_id": "stm-summary", "call_type": "summary", "input_tokens": 15},
        {"run_id": "stm-trim", "call_type": "nlu", "input_tokens": 999},
    ]
    counted = totals(calls)
    assert counted["trim"] == {"chat": 140, "summary": 0}
    assert counted["summary"] == {"chat": 0, "summary": 45}
