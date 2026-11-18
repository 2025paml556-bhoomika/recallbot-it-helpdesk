"""Check the knowledge base against the Step 2 minimums.

Run from the repository root:

    python -m src.kb_stats

Exits with status 1 if any minimum is missed, so the check can gate the step.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"

MIN_FILES = 15
MIN_WORDS = 4000
MAX_WORDS = 8000
MIN_CODES = 10
MIN_VERSIONED_DOC_IDS = 2

HEADER_FIELDS = ("doc_id", "version", "valid_from", "status")

# Codes are uppercase tokens with at least one hyphen: ERR-VPN-809, SLA-P1.
CODE_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'+._-]*")


def parse_header(text: str) -> dict[str, str]:
    """Read the leading key: value lines. Stops at the first blank line."""
    header: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip() == "":
            break
        if ":" not in line:
            break
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    return header


def main() -> int:
    files = sorted(KB_DIR.glob("*.md"))
    total_words = 0
    codes: set[str] = set()
    by_doc: dict[str, list[str]] = defaultdict(list)
    missing_headers: list[str] = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        total_words += len(WORD_RE.findall(text))
        codes.update(CODE_RE.findall(text))

        header = parse_header(text)
        absent = [field for field in HEADER_FIELDS if not header.get(field)]
        if absent:
            missing_headers.append(f"{path.name}: missing {', '.join(absent)}")
        else:
            by_doc[header["doc_id"]].append(header["version"])

    versioned = sorted(
        doc_id
        for doc_id, versions in by_doc.items()
        if any(v.startswith("2025") for v in versions)
        and any(v.startswith("2026") for v in versions)
    )

    print(f"files:        {len(files)}")
    print(f"words:        {total_words}")
    print(f"unique codes: {len(codes)}")
    for code in sorted(codes):
        print(f"  {code}")
    print(f"doc_ids with both a 2025 and a 2026 version: {len(versioned)}")
    for doc_id in versioned:
        print(f"  {doc_id}: {', '.join(by_doc[doc_id])}")
    if missing_headers:
        print("header problems:")
        for problem in missing_headers:
            print(f"  {problem}")
    else:
        print("headers: every file has version, valid_from and status")

    ok = (
        len(files) >= MIN_FILES
        and MIN_WORDS <= total_words <= MAX_WORDS
        and len(codes) >= MIN_CODES
        and len(versioned) >= MIN_VERSIONED_DOC_IDS
        and not missing_headers
    )
    print("result:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
