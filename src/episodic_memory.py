"""Episodic memory: a record of a session that has finished.

An episode is one thing that happened, so it is written once and never
edited. Records live in memory/episodic.jsonl:

    {"id", "user", "session", "ts", "text", "importance", "type", "embedding"}

The embedding is stored on the record so a later session can score it
without rebuilding the index. A flat index of the same vectors is kept
beside the file for the same reason the knowledge base has one.

Relevance is not cosine alone. The score is written out in score():

    alpha * recency + beta * importance + gamma * cosine

recency is EXP_DECAY raised to the episode's age in hours, importance is
the stored 1-10 scaled to 0-1, and cosine is the similarity to the current
message. A reflection is an episode of type "reflection", written at the
end of a session, and a later session retrieves it the same way.

Every read filters on user. A forget removes or redacts the user's episodes
whose text contains the value, and rebuilds the index.

    python -m src.episodic_memory
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone

import faiss
import numpy as np

from src.config import (
    EMBED_DIM,
    EPISODIC_ALPHA,
    EPISODIC_BETA,
    EPISODIC_DECAY,
    EPISODIC_GAMMA,
    EPISODIC_IDS_PATH,
    EPISODIC_INDEX_PATH,
    EPISODIC_PATH,
    EPISODIC_TOP_K,
    MEMORY_RELEVANCE,
)
from src.embedder import embed

# What an episode can be. A reflection is the session-end note.
TYPES = ("event", "booking", "ticket", "decision", "reflection", "note")

# Importance is a rule, not a model call: a ticket or a booking is the thing
# a later session asks about, and small talk is not. The rule is deterministic,
# so the same session always yields the same importance. A decision (a refresh
# rather than another repair) outranks a complaint, which is what the two
# weightings in the tests compare.
IMPORTANCE = {
    "ticket": 8,
    "booking": 7,
    "decision": 9,
    "event": 6,
    "reflection": 5,
    "note": 2,
}

# Words that make an event a decision rather than a complaint. The scripted
# turn is "I'd like to go ahead with the refresh rather than another repair."
_DECISION = re.compile(r"\b(rather than|decided|decision|go ahead with)\b", re.IGNORECASE)

_SMALL_TALK = re.compile(
    r"^(thanks|thank you|thx|ok|okay|bye|goodbye|hi|hello|hey|"
    r"good morning|good night|have a (good|nice) day|cheers)[\s!.]*$",
    re.IGNORECASE,
)


def rate(text: str, episode_type: str) -> int:
    """The 1-10 importance of one episode, from the type and the wording.

    A decision is the highest: the user chose something a later session has
    to remember. A ticket and a booking come next. A complaint is an event.
    Anything the rule does not recognise stays a low note. The result is
    always between 1 and 10.
    """
    kind = episode_type
    if kind == "event" and _DECISION.search(text or ""):
        kind = "decision"
    value = IMPORTANCE.get(kind, IMPORTANCE["note"])
    return max(1, min(10, int(value)))


def score(
    age_hours: float,
    importance: float,
    cosine: float,
    alpha: float = EPISODIC_ALPHA,
    beta: float = EPISODIC_BETA,
    gamma: float = EPISODIC_GAMMA,
    decay: float = EPISODIC_DECAY,
) -> float:
    """alpha * decay**hours + beta * (importance / 10) + gamma * cosine.

    importance is on the 1-10 scale the records use. Dividing by 10 puts it
    on the same 0-1 scale as the cosine and the decay term.
    """
    recency = decay**age_hours
    return alpha * recency + beta * (importance / 10.0) + gamma * cosine


class EpisodicMemory:
    """The episode store. root overrides the paths for tests."""

    def __init__(self, root=None) -> None:
        if root is None:
            self.path = EPISODIC_PATH
            self.index_path = EPISODIC_INDEX_PATH
            self.ids_path = EPISODIC_IDS_PATH
        else:
            root.mkdir(parents=True, exist_ok=True)
            self.path = root / "episodic.jsonl"
            self.index_path = root / "episodic.faiss"
            self.ids_path = root / "episodic_ids.json"

    def add(
        self,
        user: str,
        text: str,
        episode_type: str = "event",
        *,
        session: str | None = None,
        importance: int | None = None,
        when: datetime | None = None,
    ) -> dict | None:
        """Append one episode and its vector. Small talk is not stored.

        importance defaults to the rule for the type. Returns the record, or
        None when the text was small talk and nothing was written.
        """
        if _SMALL_TALK.match(text.strip()):
            return None
        if episode_type not in TYPES:
            raise ValueError(f"unknown episode type: {episode_type!r}")

        stamp = when or datetime.now(timezone.utc)
        record = {
            "id": self._next_id(),
            "user": user,
            "session": session,
            "ts": stamp.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "text": text.strip(),
            "importance": int(importance if importance is not None else rate(text, episode_type)),
            "type": episode_type,
        }
        vector = embed([record["text"]], log=False)[0]
        record["embedding"] = [float(value) for value in vector]
        self._append(record)
        self._reindex(self._read())
        return record

    def reflect(self, user: str, session: str, text: str) -> dict:
        """The note written when a session ends. Retrieved like any episode."""
        written = self.add(user, text, "reflection", session=session)
        if written is None:
            raise ValueError("a reflection cannot be empty or small talk")
        return written

    def recall(
        self,
        user: str,
        query: str,
        *,
        top_k: int = EPISODIC_TOP_K,
        now: datetime | None = None,
        alpha: float = EPISODIC_ALPHA,
        beta: float = EPISODIC_BETA,
        gamma: float = EPISODIC_GAMMA,
        min_score: float | None = None,
    ) -> list[dict]:
        """This user's episodes closest to the query, highest score first.

        Other users are excluded before scoring. An episode whose cosine is
        below the relevance cut-off is dropped even if it is recent, so an
        unrelated message does not surface the user's newest episode.
        """
        rows = [row for row in self._read() if row["user"] == user and not row.get("redacted")]
        if not rows or not query.strip():
            return []

        moment = now or datetime.now(timezone.utc)
        query_vector = embed([query], log=False)[0]
        ranked = []
        for row in rows:
            cosine = _cosine(query_vector, row["embedding"])
            if cosine < MEMORY_RELEVANCE:
                continue
            age = _age_hours(row["ts"], moment)
            value = score(age, row["importance"], cosine, alpha, beta, gamma)
            if min_score is not None and value < min_score:
                continue
            hit = _public(row)
            hit["score"] = value
            hit["cosine"] = cosine
            hit["age_hours"] = age
            ranked.append(hit)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def for_user(self, user: str) -> list[dict]:
        """Every episode of this user, in the order they were written."""
        return [_public(row) for row in self._read() if row["user"] == user]

    def forget_user(self, user: str) -> int:
        """Drop every episode of this user and rebuild the index.

        Another user's episodes stay. Returns how many were removed.
        """
        kept = []
        removed = 0
        for row in self._read():
            if row["user"] == user:
                removed += 1
                continue
            kept.append(row)
        self._write(kept)
        self._reindex(kept)
        return removed

    def forget(self, user: str, value: str) -> int:
        """Drop this user's episodes whose text contains the value.

        The line is removed and the index rebuilt, so a restart cannot bring
        it back. Returns how many episodes were removed.
        """
        needle = value.strip().lower()
        kept = []
        removed = 0
        for row in self._read():
            if row["user"] == user and needle and needle in row["text"].lower():
                removed += 1
                continue
            kept.append(row)
        self._write(kept)
        self._reindex(kept)
        return removed

    def redact(self, user: str, value: str) -> int:
        """Blank the value inside this user's episodes instead of deleting them.

        Used when the episode should survive but the forgotten fact must not
        be readable from it. Returns how many episodes were changed.
        """
        needle = value.strip()
        if not needle:
            return 0
        pattern = re.compile(re.escape(needle), re.IGNORECASE)
        changed = 0
        rows = []
        for row in self._read():
            if row["user"] == user and pattern.search(row["text"]):
                row["text"] = pattern.sub("[forgotten]", row["text"])
                row["redacted"] = True
                row["embedding"] = [float(value) for value in embed([row["text"]], log=False)[0]]
                changed += 1
            rows.append(row)
        self._write(rows)
        self._reindex(rows)
        return changed

    def vector_count(self) -> int:
        if not self.index_path.exists():
            return 0
        return int(faiss.read_index(str(self.index_path)).ntotal)

    def _reindex(self, rows: list[dict]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(EMBED_DIM)
        visible = [row for row in rows if not row.get("redacted")]
        if visible:
            vectors = np.asarray([row["embedding"] for row in visible], dtype=np.float32)
            index.add(np.ascontiguousarray(vectors))
        faiss.write_index(index, str(self.index_path))
        self.ids_path.write_text(
            json.dumps([row["id"] for row in visible]),
            encoding="utf-8",
        )

    def _next_id(self) -> str:
        return f"ep-{len(self._read()) + 1:04d}"

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _write(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        text = ("\n".join(lines) + "\n") if lines else ""
        self.path.write_text(text, encoding="utf-8")


def _public(row: dict) -> dict:
    """The record without its embedding, which is an index, not a fact."""
    return {key: value for key, value in row.items() if key != "embedding"}


def _cosine(query: np.ndarray, stored: list[float]) -> float:
    vector = np.asarray(stored, dtype=np.float32)
    return float(np.dot(query, vector))


def _age_hours(stamp: str, now: datetime) -> float:
    then = datetime.fromisoformat(stamp)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    delta = now.astimezone(timezone.utc) - then.astimezone(timezone.utc)
    return max(0.0, delta.total_seconds() / 3600.0)


def _check_score() -> None:
    """The worked example: 10 hours old, importance 8, cosine 0.6.

    0.995**10 + 0.8 + 0.6 = 2.351. Printed so the figure can be checked
    without a test runner.
    """
    value = score(10, 8, 0.6)
    expected = 0.995**10 + 0.8 + 0.6
    if not math.isclose(value, expected, abs_tol=1e-9) or abs(value - 2.351) > 0.001:
        raise SystemExit(f"score drifted: {value} != {expected}")
    print(f"score(10h, importance 8, cosine 0.6) = {value:.3f}")


def main() -> None:
    _check_score()
    store = EpisodicMemory()
    print(f"records: {store.path}")
    print(f"vectors: {store.vector_count()}")


if __name__ == "__main__":
    main()
