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

from src.episodic_memory import EpisodicMemory
from src.history import WorkingMemory
from src.semantic_memory import FACT_KEYS, SemanticMemory

# An intent that states nothing worth remembering.
_SKIP_INTENTS = ("confirm", "reject", "out_of_scope")


class Memory:
    """The three stores for one user in one session."""

    def __init__(self, user: str, session: str, root=None) -> None:
        self.user = user
        self.session = session
        self.semantic = SemanticMemory(root)
        self.episodic = EpisodicMemory(root)
        self.working = WorkingMemory(user, session, root)

    def load(self, text: str) -> dict:
        """The memories relevant to this message.

        Returns {"summary", "facts", "episodes"}. facts are the user's active
        facts, episodes are the few past episodes close to the message, and
        summary is this session's notes. A new user gets three empty values.
        """
        return {
            "summary": self.working.summary,
            "facts": self.semantic.recall(self.user, text),
            "episodes": self.episodic.recall(self.user, text),
        }

    def record_turn(self, text: str, intent: str, slots: dict, turn: int) -> list[dict]:
        """Write the facts this turn stated. Returns the records written.

        confirm, reject and out_of_scope state no fact. Anything else is
        handed to the semantic store, which itself writes nothing for small
        talk or for a turn with no fact slot.
        """
        if intent in _SKIP_INTENTS:
            return []
        return self.semantic.extract(self.user, text, slots, turn)

    def remember_event(self, text: str, episode_type: str, turn: int | None = None) -> dict | None:
        """Append one episode for something that just happened, such as a ticket."""
        return self.episodic.add(
            self.user,
            text,
            episode_type,
            session=self.session,
        )

    def end_session(self, reflection: str) -> dict:
        """Write the reflection, keep one episode of the summary, drop it.

        The summary episode is how a later session reads what this one was
        about. The reflection is the separate note the brief asks for. The
        working file is removed either way.
        """
        summary = self.working.end()
        if summary:
            self.episodic.add(self.user, summary, "note", session=self.session)
        return self.episodic.reflect(self.user, self.session, reflection)

    def forget(self, value: str) -> dict:
        """Remove a fact from the semantic store and from this user's episodes."""
        return {
            "facts": self.semantic.forget_value(self.user, value),
            "episodes": self.episodic.forget(self.user, value),
        }


def fact_text(slots: dict) -> list[str]:
    """The standing facts present in a slot dict, phrased as a sentence each."""
    lines = []
    for key in FACT_KEYS:
        value = slots.get(key) if isinstance(slots, dict) else None
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value.strip()}")
    return lines
