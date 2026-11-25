"""Keep the test suite out of the real call log.

logs/calls.jsonl is the measurement record for the reported runs and is
append-only, so a line written by a test can never be taken out again. Every
test run therefore writes to a throwaway file instead.

The redirect happens here, at import time, because pytest loads this file
before any test module. Test modules that do `from src.config import
LOG_PATH`, src.logger (which binds LOG_PATH by name) and src.plots (which uses
it as a default argument) all pick up the temporary path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import src.config as config

REAL_LOG_PATH = config.LOG_PATH
TEST_LOG_PATH = Path(tempfile.mkdtemp(prefix="recallbot-test-log-")) / "calls.jsonl"

config.LOG_PATH = TEST_LOG_PATH

import src.logger as logger  # noqa: E402  (must follow the redirect)

logger.LOG_PATH = TEST_LOG_PATH


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


_REAL_LOG_SIZE_AT_START = _size(REAL_LOG_PATH)


@pytest.fixture(autouse=True)
def _log_goes_to_temp_file(monkeypatch):
    """Hold the redirect for every test, even one that patched it and failed."""
    monkeypatch.setattr(config, "LOG_PATH", TEST_LOG_PATH)
    monkeypatch.setattr(logger, "LOG_PATH", TEST_LOG_PATH)


def pytest_sessionfinish(session, exitstatus):
    """Fail the run if anything still reached the real log."""
    grown = _size(REAL_LOG_PATH) - _REAL_LOG_SIZE_AT_START
    if grown:
        session.exitstatus = 1
        print(f"\nERROR: the test run wrote {grown} bytes to {REAL_LOG_PATH}")
