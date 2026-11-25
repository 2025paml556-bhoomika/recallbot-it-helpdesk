"""What a turn loads from memory, and what it writes back.

Three stores, and they do not share a lifetime:

    working   this session's summary, discarded when the session ends
    semantic  facts about the user, kept until the user changes or forgets them
    episodic  one record per finished session, kept and never edited

load() reads before the turn is handled. It never reads another user's files,
and it reads the working summary only for the session that is open.

record_turn() writes what this turn stated. A small-talk turn writes nothing.
A fact the user stated becomes a semantic record. The working summary is left
to the caller, because summarising is a model call and not every turn wants one.

end_session() folds the summary into one episode, writes the reflection, and
then deletes the summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import time

from src.config import CHAT_MODEL
from src.episodic_memory import EpisodicMemory, rate
from src.history import WorkingMemory, render
from src.llm import chat
from src.logger import log_call
from src.semantic_memory import FACT_KEYS, SemanticMemory

# Written once, when the session ends. The model is handed the turns and
# asked for the note a later session should be able to read back. It is not
# typed by hand: end_session() calls this and stores whatever comes back.
REFLECT_SYSTEM = """You write the closing note for one IT helpdesk session.

You are given the turns. Write one short paragraph a later session can read.
- Keep the ticket, the booking (place, day and time), and any decision the user made.
- Do not include a personal phone number, even if one was said.
- Write only the note. No preamble, no JSON.
"""

# A session id becomes a filename, so anything that is not a plain character
# is flattened. Two ids that flatten to the same name would share a file.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

# An intent that states nothing worth remembering. recall_fact is a question
# about a fact already stored, so writing it would store the question itself.
_SKIP_INTENTS = ("confirm", "reject", "out_of_scope", "recall_fact")


def _session_filename(user: str, session: str) -> str:
    """One safe filename for this user and session.

    The user is part of the name so two users can both have a session
    called S-1 without sharing a file.
    """
    safe_user = _UNSAFE.sub("-", user.strip() or "user").strip("-") or "user"
    safe_session = _UNSAFE.sub("-", session.strip() or "session").strip("-") or "session"
    return f"{safe_user}__{safe_session}.json"


class ConversationLog:
    """The transcript of one conversation, one JSON file per session.

    The file is memory/sessions/<user>__<session>.json. It is created on the
    first turn and rewritten after each later turn, so a restart of the page
    keeps the same conversation in the same file. Ending the session leaves
    the file where it is: it is the record of the conversation, not the
    working summary, which is deleted.
    """

    def __init__(self, user: str, session: str, root=None) -> None:
        from src.config import SESSIONS_DIR

        self.user = user
        self.session = session
        base = root if root is not None else SESSIONS_DIR
        self.path = Path(base) / _session_filename(user, session)
        self.turns: list[dict] = []
        self._load()

    def append(
        self,
        user_text: str,
        reply: str,
        *,
        turn: int,
        intent: str,
        calls: list[dict] | None = None,
        tool: dict | None = None,
    ) -> dict:
        """Add one exchange and write the whole file back.

        calls is every model call made for this turn, each one already in
        message order: system prompt, user prompt, assistant response.
        tool is the tool that ran, when one did: its name, the arguments it
        was given, and the result it returned. A turn that called neither
        still records the user text and the reply.

        A session is a few dozen turns, so rewriting the file is cheaper
        than keeping a correct JSON array by appending. A crash mid-write
        can lose the last turn; it cannot leave a truncated array that the
        next load refuses to parse, because the write is of the whole file.
        """
        row = {
            "turn": turn,
            "user": user_text,
            "bot": reply,
            "intent": intent,
            "calls": list(calls or []),
        }
        if tool is not None:
            row["tool"] = tool
        self.turns.append(row)
        self._save()
        return row

    def _load(self) -> None:
        if not self.path.exists():
            return
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        loaded = saved.get("turns", [])
        self.turns = loaded if isinstance(loaded, list) else []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "user": self.user,
            "session": self.session,
            "turns": self.turns,
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class Memory:
    """The three stores for one user in one session."""

    def __init__(self, user: str, session: str, root=None) -> None:
        self.user = user
        self.session = session
        self.semantic = SemanticMemory(root)
        self.episodic = EpisodicMemory(root)
        self.working = WorkingMemory(user, session, root)
        # Beside the memory root when a test passes one, otherwise the shared
        # sessions folder. An isolated test must not write a real transcript.
        sessions_root = None if root is None else Path(root) / "sessions"
        self.conversation = ConversationLog(user, session, sessions_root)

    def load(self, text: str, *, semantic: bool = True, episodic: bool = True) -> dict:
        """The memories relevant to this message.

        Returns {"summary", "facts", "episodes"}. facts are the user's active
        facts, episodes are the few past episodes close to the message, and
        summary is this session's notes. A new user gets three empty values.

        semantic and episodic gate the read, not the store. The ablation
        turns one off at a time: no_rag reads neither, hybrid_sem reads facts
        but not episodes. What was written stays written either way.
        """
        return {
            "summary": self.working.summary,
            "facts": self.semantic.recall(self.user, text) if semantic else [],
            "episodes": self.episodic.recall(self.user, text) if episodic else [],
        }

    def forget_request(self, key: str, value: str | None) -> dict:
        """Delete one fact the user asked to be forgotten.

        The key's rows go, including a value that was superseded earlier.
        The value is also removed from this user's episodes, so a reflection
        that quoted the phone number does not keep it. Another user's rows
        are not touched. Returns the counts removed.
        """
        removed = {
            "key": key,
            "facts": self.semantic.forget(self.user, key),
            "episodes": 0,
        }
        if value:
            removed["episodes"] = self.episodic.forget(self.user, value)
            # A note that quoted the number is not under the key.
            removed["facts"] += self.semantic.forget_value(self.user, value)
        return removed

    def record_turn(self, text: str, intent: str, slots: dict, turn: int) -> list[dict]:
        """Write the facts this turn stated. Returns the records written.

        confirm, reject, out_of_scope and recall_fact state no fact. Anything
        else is handed to the semantic store, which itself writes nothing for
        small talk or for a turn with no fact slot.
        """
        if intent in _SKIP_INTENTS:
            return []
        # book_slot states where a visit should happen, not where the user
        # sits. Writing it would make the next "come to my desk" skip the
        # office question, and "where do I sit?" would answer with a desk
        # the user never claimed.
        remembered = dict(slots) if isinstance(slots, dict) else {}
        if intent == "book_slot":
            # A visit place is not where the user sits, and the sentence is
            # not a standing note either. "Monday" must not be stored.
            remembered.pop("location", None)
            return self.semantic.extract(self.user, "", remembered, turn)
        if intent != "provide_info":
            # A fault report and a ticket check state no personal fact.
            # A laptop id stated along the way is still a laptop id. The
            # place in a fault ("on my floor") is the visit, not the office.
            asset = remembered.get("asset_id")
            only_asset = {"asset_id": asset} if isinstance(asset, str) else {}
            return self.semantic.extract(self.user, "", only_asset, turn)
        return self.semantic.extract(self.user, text, remembered, turn)

    def remember_event(
        self,
        text: str,
        episode_type: str,
        turn: int | None = None,
        *,
        run_id: str | None = None,
        config: str | None = None,
    ) -> dict | None:
        """Append one episode for something that just happened, such as a ticket.

        The importance is the rule in rate(), and it is logged as its own
        call_type so the log shows the score that was stored. Small talk
        writes nothing and logs nothing.
        """
        importance = rate(text, episode_type)
        written = self.episodic.add(
            self.user,
            text,
            episode_type,
            session=self.session,
            importance=importance,
        )
        if written is not None:
            self._log_importance(written, turn, run_id=run_id, config=config)
        return written

    def _log_importance(
        self,
        record: dict,
        turn: int | None,
        *,
        run_id: str | None,
        config: str | None,
    ) -> None:
        """One log line per episode, carrying the 1-10 score that was stored."""
        log_call(
            call_type="importance",
            model="rule",
            input_tokens=0,
            output_tokens=0,
            start=time.time(),
            config=config,
            user=self.user,
            session=self.session,
            turn=turn,
            run_id=run_id,
            extra={
                "episode_id": record["id"],
                "episode_type": record["type"],
                "importance": record["importance"],
            },
        )

    def reflect(
        self,
        turns: list[dict],
        *,
        turn: int | None = None,
        trace: list | None = None,
        run_id: str | None = None,
        config: str | None = None,
    ) -> str:
        """Ask the model for the session note. Logged as call_type "reflect".

        turns are the completed exchanges, each {"user", "bot"}. An empty
        list still produces a note, so a session that only booked something
        the model never phrased still closes.
        """
        body = render(turns) if turns else "(no turns)"
        # chat() logs this as call_type "reflect". log stays on, so the line
        # carries the API's own token counts rather than a zero.
        return chat(
            f"Turns:\n{body}",
            system=REFLECT_SYSTEM,
            temperature=0.0,
            model=CHAT_MODEL,
            call_type="reflect",
            config=config,
            user=self.user,
            session=self.session,
            turn=turn,
            trace=trace,
            run_id=run_id,
        ).strip()

    def end_session(
        self,
        reflection: str | None = None,
        turns: list[dict] | None = None,
        *,
        turn: int | None = None,
        trace: list | None = None,
        run_id: str | None = None,
        config: str | None = None,
    ) -> dict:
        """Write the reflection, keep one episode of the summary, drop it.

        reflection given is stored as it stands, which is how a test plants
        a known note. reflection omitted is generated from the turns, which
        is what the session does when it closes. The working file is removed
        either way.
        """
        summary = self.working.end()
        if summary:
            self.remember_event(summary, "note", turn, run_id=run_id, config=config)
        if not (reflection or "").strip():
            reflection = self.reflect(
                turns or [],
                turn=turn,
                trace=trace,
                run_id=run_id,
                config=config,
            )
        written = self.episodic.reflect(self.user, self.session, reflection)
        self._log_importance(written, turn, run_id=run_id, config=config)
        return written

    def forget(self, value: str) -> dict:
        """Remove a fact from the semantic store and from this user's episodes."""
        return {
            "facts": self.semantic.forget_value(self.user, value),
            "episodes": self.episodic.forget(self.user, value),
        }

    def clear_user(self) -> dict:
        """Delete everything stored about this user.

        Facts, episodes, this session's working notes, and every conversation
        file of theirs. Another user's files are not touched. Tickets and
        bookings live in the helpdesk database, so the caller clears those.
        """
        facts = self.semantic.forget_user(self.user)
        episodes = self.episodic.forget_user(self.user)
        self.working.end()
        sessions = _delete_user_sessions(self.user, self.conversation.path.parent)
        return {"facts": facts, "episodes": episodes, "sessions": sessions}


def _delete_user_sessions(user: str, folder: Path) -> int:
    """Remove memory/sessions files that belong to this user.

    The filename starts with the user's safe name and a double underscore,
    which is how ConversationLog names them. A user called "U1" must not
    take "U10"'s files, so the separator is part of the match.
    """
    if not folder.exists():
        return 0
    prefix = _session_filename(user, "x").split("__", 1)[0] + "__"
    removed = 0
    for path in folder.glob(prefix + "*.json"):
        path.unlink()
        removed += 1
    return removed


def fact_text(slots: dict) -> list[str]:
    """The standing facts present in a slot dict, phrased as a sentence each."""
    lines = []
    for key in FACT_KEYS:
        value = slots.get(key) if isinstance(slots, dict) else None
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value.strip()}")
    return lines
