"""Short-term memory: what of the current session is shown to the model.

Three strategies, selected by name:

    full     every turn so far
    trim     the last K_TURNS turns
    summary  a rolling summary plus the last two turns

The summary is working memory. It belongs to one session, it is capped, and
it is discarded when the session ends. The slot store is separate and is not
trimmed: a booking started early in the session can still be confirmed after
its turns have fallen out of the window.

Summarising is the one model call this module makes. The other two strategies
only slice a list.
"""

from __future__ import annotations

import json

from src.config import K_TURNS, WORKING_DIR, WORKING_SUMMARY_TOKENS
from src.llm import chat

STRATEGIES = ("full", "trim", "summary")

# Turns kept verbatim alongside the summary. The summary covers the rest.
SUMMARY_VERBATIM = 2

SUMMARY_SYSTEM = f"""You keep the running notes for one IT helpdesk session.

You are given the notes so far and the turns that have happened since. Write the updated notes.
- Keep facts, decisions, ticket ids, and anything still unresolved. Drop greetings and repetition.
- Use at most {WORKING_SUMMARY_TOKENS} tokens.
- Write only the notes. No preamble, no JSON.
"""


def select(history: list[dict], strategy: str) -> list[dict]:
    """The turns the strategy shows. The list it returns is a copy.

    full returns everything. trim returns the last K_TURNS. summary returns
    the last two, because the rest is carried by the summary rather than by
    the turns themselves.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if strategy == "full":
        window = history
    elif strategy == "trim":
        window = history[-K_TURNS:]
    else:
        window = history[-SUMMARY_VERBATIM:]
    return [dict(turn) for turn in window]


def render(turns: list[dict]) -> str:
    """The turns as plain lines: one "User:" and one "Bot:" per turn."""
    lines = []
    for turn in turns:
        lines.append(f"User: {turn.get('user', '')}")
        lines.append(f"Bot: {turn.get('bot', '')}")
    return "\n".join(lines)


class WorkingMemory:
    """The rolling summary for one user in one session.

    The summary is saved under memory/working/<user>/<session>.json so a
    restart mid-session continues it. end() deletes that file: the summary
    was only about this conversation.
    """

    def __init__(self, user: str, session: str, root=None) -> None:
        self.user = user
        self.session = session
        base = root if root is not None else WORKING_DIR
        self.path = base / user / f"{session}.json"
        self.summary = ""
        self._load()

    def update(
        self,
        history: list[dict],
        *,
        turn: int | None = None,
    ) -> str:
        """Rewrite the summary to include the turns since the last one.

        Returns the new summary. One chat call, logged as call_type
        "summarise". With no turns, nothing is written and no call is made.
        """
        if not history:
            return self.summary
        prompt = (
            f"Notes so far:\n{self.summary or '(none)'}\n\n"
            f"New turns:\n{render(history)}"
        )
        self.summary = chat(
            prompt,
            system=SUMMARY_SYSTEM,
            temperature=0.0,
            call_type="summarise",
            user=self.user,
            session=self.session,
            turn=turn,
        ).strip()
        self._save()
        return self.summary

    def end(self) -> str:
        """Drop the summary. Returns its text, for the episode written after."""
        text = self.summary
        self.summary = ""
        if self.path.exists():
            self.path.unlink()
        return text

    def _load(self) -> None:
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.summary = saved.get("summary", "")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"user": self.user, "session": self.session, "summary": self.summary},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
