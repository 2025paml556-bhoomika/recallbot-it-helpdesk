"""Semantic memory: stable facts about one user.

A fact is true of the user, not of one visit, so it outlives the session.
Records live in memory/semantic.jsonl, one per line:

    {"id", "user", "key", "value", "source_turn", "date", "status"}

status is "active" or "superseded". An update marks the old line superseded
and appends the new value; it never deletes the history of the change. A
forget does delete: the key's lines leave the file and its vectors leave the
index, and the deletion survives a restart because it is the file itself.

Small talk writes nothing. A fact is written only from a slot the user
stated. Every read filters on user, so one user cannot retrieve another's.

    python -m src.semantic_memory
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import faiss
import numpy as np

from src.config import (
    EMBED_DIM,
    MEMORY_RELEVANCE,
    SEMANTIC_IDS_PATH,
    SEMANTIC_INDEX_PATH,
    SEMANTIC_PATH,
)
from src.embedder import embed

# Slots that name a standing fact. A booking day is not one of them.
# A personal line that fits neither slot (a name, a colour, a badge) is
# stored as its own note instead, so it does not have to borrow a slot.
FACT_KEYS = ("location", "asset_id")
NOTE_KEY = "note"

# One note per sentence. "My badge number is 4412" must not replace
# "My laptop bag is blue", and neither must replace the laptop id.
# location and asset_id stay one active value each: a new office replaces
# the old office.
_SINGLE_VALUE_KEYS = FACT_KEYS

# How a slot is phrased when it is embedded and when it is read back.
FACT_LABELS = {
    "location": "office",
    "asset_id": "laptop",
    "note": "note",
}

# asset_id is a laptop id such as LAP-T14G3. A badge number is digits only,
# and qwen2.5:3b parks that in asset_id. It is not a laptop.
_ASSET_ID = re.compile(r"^LAP-[A-Z0-9]+$", re.IGNORECASE)

# A greeting, thanks, or a short acknowledgement. None of these states a fact.
_SMALL_TALK = re.compile(
    r"^(thanks|thank you|thx|ok|okay|bye|goodbye|hi|hello|hey|"
    r"good morning|good night|have a (good|nice) day|cheers)[\s!.]*$",
    re.IGNORECASE,
)


class SemanticMemory:
    """The fact store. root overrides the paths so a test uses its own files."""

    def __init__(self, root=None) -> None:
        if root is None:
            self.path = SEMANTIC_PATH
            self.index_path = SEMANTIC_INDEX_PATH
            self.ids_path = SEMANTIC_IDS_PATH
        else:
            root.mkdir(parents=True, exist_ok=True)
            self.path = root / "semantic.jsonl"
            self.index_path = root / "semantic.faiss"
            self.ids_path = root / "semantic_ids.json"

    def is_small_talk(self, text: str) -> bool:
        return bool(_SMALL_TALK.match(text.strip()))

    def extract(self, user: str, text: str, slots: dict, turn: int) -> list[dict]:
        """Write one record per stated fact. Returns the records written.

        A small-talk turn writes nothing. location and asset_id are written
        from the slots. asset_id is written only when it is a laptop id
        (LAP-...), so a badge number the model parked there is not stored
        as the laptop. Anything else the user stated on this turn is one
        note, kept beside the slots rather than forced into one of them.
        A new office replaces the old office. A new note does not.
        """
        if self.is_small_talk(text):
            return []

        slots = slots if isinstance(slots, dict) else {}
        written = []
        for key in FACT_KEYS:
            value = slots.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            if key == "asset_id" and not _ASSET_ID.match(value.strip()):
                continue
            written.append(self.remember(user, key, value.strip(), turn))

        note = text.strip()
        if note and not self._covered_by_slots(note, slots):
            written.append(self.remember(user, NOTE_KEY, note, turn))
        return written

    def _covered_by_slots(self, text: str, slots: dict) -> bool:
        """True when the sentence only restates the office or the laptop id.

        "I sit at the Pune office" with location "Pune office" is that
        fact, so it is not also stored as a note. "My laptop bag is blue"
        slots nothing, so it is a note. "My badge number is 4412" may have
        been slotted as asset_id, but that slot was rejected, so it is a
        note too. "Thanks, have a good day!" states neither, and is not a
        fact just because the model called it provide_info.
        """
        lowered = text.lower()
        covered = False
        for key in FACT_KEYS:
            value = slots.get(key)
            if key == "asset_id" and isinstance(value, str) and not _ASSET_ID.match(value.strip()):
                value = None
            if isinstance(value, str) and value.strip() and value.strip().lower() in lowered:
                covered = True
        if covered:
            return True
        # A personal line names whose fact it is, and states something
        # beyond the office or the laptop. "I sit at the Pune office" is
        # already covered above. "Thanks" names nothing.
        if re.search(r"\b(my|i'm|i am|i've|mine)\b", lowered):
            return False
        if re.search(r"\bi\b", lowered) and re.search(
            r"\b(name|bag|badge|colour|color|manager|phone|number)\b", lowered
        ):
            return False
        return True

    def remember(self, user: str, key: str, value: str, turn: int | None) -> dict:
        """Append one active fact.

        location and asset_id supersede this user's previous value, so
        there is one office and one laptop. A note does not: each sentence
        stays active, and an identical sentence is not written twice.
        """
        if key in _SINGLE_VALUE_KEYS:
            previous = self._latest(user).get(key)
            if previous is not None and previous["value"] == value and previous["status"] == "active":
                return previous
        else:
            previous = None
            for row in self.active(user):
                if row["key"] == key and row["value"] == value:
                    return row

        record = {
            "id": self._next_id(),
            "user": user,
            "key": key,
            "value": value,
            "source_turn": turn,
            "date": _now(),
            "status": "active",
        }
        rows = self._read()
        if previous is not None and previous["status"] == "active":
            for row in rows:
                if row["id"] == previous["id"] and row["status"] == "active":
                    row["status"] = "superseded"
        rows.append(record)
        self._write(rows)
        self._reindex(self.active_records())
        return record

    def active(self, user: str) -> list[dict]:
        """This user's active facts.

        location and asset_id contribute their latest line only. Every
        active note is kept, because each one is a different sentence.
        """
        found = []
        for row in self._read():
            if row["user"] != user or row["status"] != "active":
                continue
            if row["key"] in _SINGLE_VALUE_KEYS:
                continue
            found.append(row)
        for row in self._latest(user).values():
            if row["status"] == "active" and row["key"] in _SINGLE_VALUE_KEYS:
                found.append(row)
        return found

    def active_records(self) -> list[dict]:
        """Every user's active facts.

        location and asset_id are one row per user. Notes are one row per
        sentence, so a colour and a badge are both in the index.
        """
        found = []
        for user in self._users():
            found.extend(self.active(user))
        return found

    def _latest(self, user: str) -> dict[str, dict]:
        """The last line written for each of this user's keys."""
        latest: dict[str, dict] = {}
        for row in self._read():
            if row["user"] == user:
                latest[row["key"]] = row
        return latest

    def _users(self) -> list[str]:
        seen = []
        for row in self._read():
            if row["user"] not in seen:
                seen.append(row["user"])
        return seen

    def recall(self, user: str, query: str, top_k: int = 3) -> list[dict]:
        """Active facts of this user whose text is close to the query.

        Other users' facts are excluded before scoring. A fact below the
        relevance cut-off is dropped, so an unrelated query returns nothing.
        The query is also rephrased in the same "office: ..." form the fact
        was embedded with. "Where do I sit?" is about an office, but those
        words are far from "office: Pune office" in the embedding space.
        A question that names one kind of fact (the office, the laptop, or
        a note) is not answered from a different kind.
        """
        rows = self.active(user)
        if not rows or not query.strip():
            return []

        wanted = self._wanted_key(query)
        if wanted is not None:
            rows = [row for row in rows if row["key"] == wanted]
        if not rows:
            return []

        matrix = self._vectors_for(rows)
        query_vector = embed([self._query_text(query)], log=False)[0]
        scores = matrix @ query_vector
        order = np.argsort(-scores)
        found = []
        for position in order[:top_k]:
            score = float(scores[position])
            if score < MEMORY_RELEVANCE:
                continue
            hit = dict(rows[int(position)])
            hit["score"] = score
            found.append(hit)
        return found

    def _wanted_key(self, query: str) -> str | None:
        """Which fact key this question is about, or None to search all.

        "What colour is my laptop bag?" says laptop, but it is about the
        bag. That question searches notes, not asset_id. "Whats my asset
        Id?" searches the laptop id only, so a note that also says laptop
        cannot outscore it.
        """
        lowered = query.lower()
        if re.search(r"\b(colour|color|bag|badge|name)\b", lowered):
            return NOTE_KEY
        if re.search(
            r"\b(sit|seat|desk|office|work|location|located|based|workplace)\b",
            lowered,
        ):
            return "location"
        if re.search(r"\b(laptops?|assets?|machines?|computers?|devices?)\b", lowered):
            return "asset_id"
        return None

    def forget(self, user: str, key: str) -> int:
        """Delete this user's records for one key, file and index both.

        Returns how many records were removed. Another user's records for the
        same key are left untouched.
        """
        kept = []
        removed = 0
        for row in self._read():
            if row["user"] == user and row["key"] == key:
                removed += 1
                continue
            kept.append(row)
        self._write(kept)
        self._reindex(self.active_records())
        return removed

    def forget_user(self, user: str) -> int:
        """Delete every fact this user has, file and index both.

        Another user's facts stay. Returns how many records were removed,
        including ones already superseded.
        """
        kept = []
        removed = 0
        for row in self._read():
            if row["user"] == user:
                removed += 1
                continue
            kept.append(row)
        self._write(kept)
        self._reindex(self.active_records())
        return removed

    def forget_value(self, user: str, value: str) -> int:
        """Delete this user's records whose value contains the text."""
        kept = []
        removed = 0
        needle = value.strip().lower()
        for row in self._read():
            if row["user"] == user and needle and needle in row["value"].lower():
                removed += 1
                continue
            kept.append(row)
        self._write(kept)
        self._reindex(self.active_records())
        return removed

    def vector_count(self) -> int:
        """Vectors in the index. Equals the number of active records."""
        if not self.index_path.exists():
            return 0
        return int(faiss.read_index(str(self.index_path)).ntotal)

    def _vectors_for(self, rows: list[dict]) -> np.ndarray:
        """The stored vectors for these rows, rebuilt if the index is stale."""
        stored_ids, index = self._load_index()
        active_ids = [row["id"] for row in self.active_records()]
        if stored_ids != active_ids or index.ntotal != len(active_ids):
            self._reindex(self.active_records())
            stored_ids, index = self._load_index()
        positions = [stored_ids.index(row["id"]) for row in rows]
        vectors = np.vstack([index.reconstruct(int(position)) for position in positions])
        return np.asarray(vectors, dtype=np.float32)

    def _reindex(self, rows: list[dict]) -> faiss.Index:
        """Rebuild the flat index from the active records, in their order."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(EMBED_DIM)
        if rows:
            texts = [self._text(row) for row in rows]
            vectors = embed(texts, log=False)
            index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        faiss.write_index(index, str(self.index_path))
        self.ids_path.write_text(
            json.dumps([row["id"] for row in rows]),
            encoding="utf-8",
        )
        return index

    def _load_index(self) -> tuple[list[str], faiss.Index]:
        if not self.index_path.exists() or not self.ids_path.exists():
            return [], faiss.IndexFlatIP(EMBED_DIM)
        ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
        return ids, faiss.read_index(str(self.index_path))

    def _text(self, row: dict) -> str:
        label = FACT_LABELS.get(row["key"], row["key"])
        return f"{label}: {row['value']}"

    def _query_text(self, query: str) -> str:
        """Phrase the question like the stored fact, when it is about one.

        "Where do I sit?", "where is my location", and "where am I located"
        become "office: ...", which is the shape the location vector was
        built from. "Which laptop do I use?", "Whats my asset Id ?", and
        "what device do I have" become "laptop: ...", the shape the asset
        vector was built from. Anything else is left unchanged, so a
        password question does not get pulled toward a stored fact.
        """
        lowered = query.lower()
        if re.search(
            r"\b(sit|seat|desk|office|work|location|located|based|workplace)\b",
            lowered,
        ):
            return f"office: {query}"
        # "What colour is my laptop bag?" is about the bag, not the laptop
        # id. Routing it to "laptop:" made the badge number (stored under
        # asset_id) the answer. A colour, a bag, or a badge stays as the
        # question, so it can match the note that stated it.
        if re.search(r"\b(colour|color|bag|badge|name)\b", lowered):
            return query
        if re.search(r"\b(laptops?|assets?|machines?|computers?|devices?)\b", lowered):
            return f"laptop: {query}"
        # "Where do I sit?" contains "do", which is not a manager question.
        # The word has to be the subject of the question, not a helper verb.
        if re.search(r"\b(manager|boss|supervisor)\b", lowered):
            return f"manager: {query}"
        return query

    def _next_id(self) -> str:
        return f"sem-{len(self._read()) + 1:04d}"

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
        """Rewrite the stream. An update rewrites one line's status; a forget drops lines."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        text = ("\n".join(lines) + "\n") if lines else ""
        self.path.write_text(text, encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    store = SemanticMemory()
    print(f"records: {store.path}")
    print(f"active:  {len(store.active_records())}")
    print(f"vectors: {store.vector_count()}")


if __name__ == "__main__":
    main()
