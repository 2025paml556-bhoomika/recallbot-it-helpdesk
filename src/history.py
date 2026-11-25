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


def earlier_turns(history: list[dict]) -> list[dict]:
    """The completed exchanges, not the message being answered.

    The last row is the current turn. It has the user text and no reply yet,
    so it is not part of the history the strategy windows over. A session
    with only the current turn has no earlier history.
    """
    done = []
    for turn in history[:-1]:
        if not turn.get("bot"):
            continue
        done.append({"user": turn.get("text", ""), "bot": turn.get("bot", "")})
    return done


def window_text(history: list[dict], strategy: str, summary: str = "") -> str:
    """The earlier conversation the reply may see, under one strategy.

    full sends every completed turn. trim sends the last K_TURNS. summary
    sends the rolling notes plus the last two turns. The current message is
    not included: the caller already puts it in the prompt. An empty window
    is an empty string, so the first turn of a session adds nothing.
    """
    shown = select(earlier_turns(history), strategy)
    lines = []
    if strategy == "summary" and summary.strip():
        lines.append("Notes so far:")
        lines.append(summary.strip())
    rendered = render(shown)
    if rendered:
        if lines:
            lines.append("")
        lines.append("Earlier turns:")
        lines.append(rendered)
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
        # How many completed turns the summary already covers. The two newest
        # turns are quoted verbatim, so they are not counted here.
        self.covered = 0
        self._load()

    def update(
        self,
        history: list[dict],
        *,
        turn: int | None = None,
        trace: list | None = None,
        run_id: str | None = None,
        config: str | None = None,
    ) -> str:
        """Rewrite the summary to include the turns since the last one.

        Returns the new summary. One chat call, logged as call_type
        "summary". With no turns, nothing is written and no call is made.
        """
        if not history:
            return self.summary
        prompt = (
            f"Notes so far:\n{self.summary or '(none)'}\n\n"
            f"New turns:\n{render(history)}"
        )
        # The notes are the session's own turns, not a retrieved memory, so
        # retrieved_tokens is 0.
        self.summary = chat(
            prompt,
            system=SUMMARY_SYSTEM,
            temperature=0.0,
            call_type="summary",
            config=config,
            user=self.user,
            session=self.session,
            turn=turn,
            trace=trace,
            run_id=run_id,
            retrieved_tokens=0,
        ).strip()
        self.covered += len(history)
        self._save()
        return self.summary

    def end(self) -> str:
        """Drop the summary. Returns its text, for the episode written after."""
        text = self.summary
        self.summary = ""
        self.covered = 0
        if self.path.exists():
            self.path.unlink()
        return text

    def _load(self) -> None:
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.summary = saved.get("summary", "")
            self.covered = int(saved.get("covered", 0))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "user": self.user,
                    "session": self.session,
                    "summary": self.summary,
                    "covered": self.covered,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
