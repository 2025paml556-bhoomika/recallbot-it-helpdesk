"""Dialogue state for one helpdesk session.

Part 1 labels the latest user message with src.nlu.understand.
Part 2 keeps the slots from earlier turns and fills them in as the user
answers, so "Thursday, please." is remembered when they later say yes.
Part 3 decides what to do with that turn. It asks for a slot that is still
missing, or it calls one tool. It never writes a booking on the turn that
supplied the last slot: the user must say yes first.
Part 4 turns that decision into the one sentence the user reads. An ask is
already a sentence, so it is returned as it stands and no model is called.
A tool result is phrased by the chat model, which is the only model call
this module makes besides the one in understand().

    python -m src.agent "The Pune office."
"""

from __future__ import annotations

import argparse
import json

from src.llm import chat
from src.memory import Memory
from src.nlu import SLOT_NAMES, understand
from src.search_kb import RELEVANCE_THRESHOLD
from src.tools import NOT_FOUND, book_it_slot, get_ticket, raise_ticket, search_kb

# A booking is not complete until the user has named all three.
BOOKING_SLOTS = ("location", "date", "time")

# Asked in this order, one per turn. location covers site, floor and desk:
# the user often gives them in one answer, so they are one slot.
BOOKING_ASKS = {
    "location": "Which office, floor, and desk should the engineer come to?",
    "date": "Which day works for the visit?",
    "time": "What time of day works?",
}

# The question asked once location, date and time are all known. The booking
# tool is not called on this turn. It is called on the later turn where the
# user says yes.
CONFIRM_BOOKING = "Shall I book that visit?"

# What decide() returns in its "action" field.
ACTION_ASK = "ask"
ACTION_TOOL = "tool"
ACTION_NONE = "none"

# Fixed lines. These turns have nothing to phrase, so they never call the model.
REPLIES = {
    "reject": "Okay, I won't book that.",
    "provide_info": "Thanks, I'll keep that in mind.",
    "out_of_scope": "I can only help with IT issues.",
    "confirm": "There is nothing waiting to book.",
    "fix_issue": "That is already raised.",
    # Said when the user asks about themselves and nothing stored matches.
    "recall_fact": "I don't have that stored for you.",
}

# How a stored fact is read back. The key is the slot name, not the sentence
# the user asked, so "Where do I sit?" and "Which office am I based at?" share
# one line. A key with no entry is read as "Your <key> is <value>."
FACT_REPLIES = {
    "location": "You sit at {value}.",
    "asset_id": "You use {value}.",
}

# The model phrases a tool result. It must not call a tool, and it must not
# add a fact that is not in the result it is shown.
REPLY_SYSTEM = """You are the IT helpdesk. Write the one reply the user will read.

You are given the user's message and the result of a tool that has already run. Use only that result.
- Do not call a tool, do not invent a ticket id, a time, or a place, and do not add a step the result does not state.
- If the result is a knowledge-base passage, answer from it in one or two sentences and name the document it came from.
- If the result says nothing relevant was found, say you don't know. Do not guess.
- If the result is a ticket or a booking, report its id and its status.
- Write the reply only. No JSON, no preamble.
"""

# Shown to the model when retrieval found nothing above the relevance cut-off.
# The model is not handed a weak passage it might quote as if it were an answer.
NO_PASSAGE = "Nothing relevant was found in the knowledge base."

# Slots a later message is allowed to refresh. issue_type and error_code
# describe the fault being discussed, so a booking answer must not wipe them.
CARRIED_SLOTS = (
    "issue_type",
    "error_code",
    "urgency",
    "location",
    "date",
    "time",
    "asset_id",
    "ticket_id",
)


class Session:
    """Slots gathered across the turns of one user in one session.

    Slots belong to (user, session). A second user, or the same user in a
    later session, starts empty and cannot read these.
    """

    def __init__(self, user: str, session: str, db_path=None, memory=None) -> None:
        self.user = user
        self.session = session
        self.turn = 0
        self.slots: dict[str, str | None] = {name: None for name in SLOT_NAMES}
        self.pending_booking = False
        self.awaiting_confirmation = False
        self.history: list[dict] = []
        # None uses the shared helpdesk database. Tests pass their own file.
        self.db_path = db_path
        # None uses the shared memory folder. Tests pass an isolated store.
        self.memory = memory if memory is not None else Memory(user, session)
        # What load() returned for the latest turn. Empty until the first one.
        self.recalled: dict = {"summary": "", "facts": [], "episodes": []}

    def ingest(self, text: str) -> dict:
        """Label one message and merge any new slot values into this session.

        Returns the NLU result plus the session slots after the merge.
        understand() is the only model call. A rule hit makes none.
        """
        self.turn += 1
        # Loaded before the sentence is labelled, because a later session
        # answers from what earlier sessions stored. The label itself still
        # comes from the sentence alone.
        self.recalled = self.memory.load(text)
        understood = understand(
            text,
            user=self.user,
            session=self.session,
            turn=self.turn,
        )
        self._merge(understood["intent"], understood["slots"])
        snapshot = {
            "turn": self.turn,
            "text": text,
            "intent": understood["intent"],
            "source": understood["source"],
            "message_slots": dict(understood["slots"]),
            "slots": self.slots_so_far(),
        }
        self.history.append(snapshot)
        # A stated fact is written now, so a later session can read it. A
        # greeting, a yes, or a no writes nothing.
        self.memory.record_turn(
            text,
            understood["intent"],
            understood["slots"],
            self.turn,
        )
        return {
            "intent": understood["intent"],
            "source": understood["source"],
            "message_slots": dict(understood["slots"]),
            "slots": self.slots_so_far(),
            "recalled": self.recalled,
            "turn": self.turn,
        }

    def slots_so_far(self) -> dict[str, str | None]:
        """A copy of the carried slots. Callers cannot mutate the session."""
        return dict(self.slots)

    def missing_booking_slots(self) -> list[str]:
        """location, date, time that are still empty, in that order."""
        return [name for name in BOOKING_SLOTS if not self.slots.get(name)]

    def _merge(self, intent: str, message_slots: dict) -> None:
        """Copy stated slots in. A new value replaces the one carried so far.

        confirm and reject are answers about the booking, not new facts, so
        they change no slot. reject also drops the unfinished booking: the
        user backed out, and the next request must collect its own day.
        """
        if intent == "reject":
            self.pending_booking = False
            self.awaiting_confirmation = False
            return

        if intent != "confirm":
            for name in CARRIED_SLOTS:
                value = message_slots.get(name)
                if isinstance(value, str) and value.strip():
                    self.slots[name] = value.strip()

        if intent == "book_slot":
            self.pending_booking = True
        elif intent == "confirm" and self.pending_booking:
            # Held, not cleared. The tool call that writes the row comes later;
            # clearing it here would forget which booking the user accepted.
            self.pending_booking = True

    def decide(self, text: str) -> dict:
        """Choose the action for the latest ingested turn.

        Returns one of:
        - {"action": "ask", "slot": name or None, "prompt": text}
        - {"action": "tool", "name": tool, "result": value}
        - {"action": "none"}

        A booking is only written when the user confirms a visit that was
        already fully described. The turn that filled the last slot asks
        instead of writing. A tool that refuses (no confirmation, unknown
        ticket) is not called at all, so it cannot log a line.
        """
        if not self.history:
            raise RuntimeError("decide() needs ingest() to have run first")
        intent = self.history[-1]["intent"]

        if intent == "ask_kb":
            return self._tool("search_kb", search_kb(text, user=self.user, session=self.session, turn=self.turn))

        if intent == "check_ticket":
            ticket_id = self.slots.get("ticket_id")
            if not ticket_id:
                return self._ask(None, "Which ticket should I look up?")
            found = get_ticket(
                ticket_id,
                self.user,
                session=self.session,
                turn=self.turn,
                db_path=self._db_path(),
            )
            return self._tool("get_ticket", found)

        if intent == "fix_issue":
            if self.slots.get("ticket_id"):
                return {"action": ACTION_NONE}
            issue = self.slots.get("issue_type") or self.slots.get("error_code") or text
            ticket = raise_ticket(
                self.user,
                issue,
                asset_id=self.slots.get("asset_id"),
                session=self.session,
                turn=self.turn,
                db_path=self._db_path(),
            )
            self.slots["ticket_id"] = ticket["ticket_id"]
            self.memory.remember_event(
                f"Raised {ticket['ticket_id']} for {ticket['issue_type']}.",
                "ticket",
            )
            return self._tool("raise_ticket", ticket)

        if intent == "book_slot":
            return self._advance_booking()

        if intent == "recall_fact":
            # Answered from memory, not from the knowledge base. A question
            # states no new fact, so nothing is written on this turn either.
            fact = self._fact_for(text)
            if fact is None:
                return {"action": ACTION_NONE}
            return self._tool("recall_fact", fact)

        if intent == "confirm":
            if self.pending_booking and self.awaiting_confirmation and not self.missing_booking_slots():
                booked = book_it_slot(
                    self.user,
                    self.slots.get("issue_type") or "unspecified",
                    self.slots["location"],
                    self.slots["date"],
                    self.slots["time"],
                    confirmed=True,
                    ticket_id=self.slots.get("ticket_id"),
                    asset_id=self.slots.get("asset_id"),
                    session=self.session,
                    turn=self.turn,
                    db_path=self._db_path(),
                )
                self.note_booked()
                self.memory.remember_event(
                    (
                        f"Booked a visit at {booked['location']} on {booked['date']} "
                        f"at {booked['time']}."
                    ),
                    "booking",
                )
                return self._tool("book_it_slot", booked)
            return {"action": ACTION_NONE}

        # reject, provide_info, out_of_scope: nothing to call and nothing to ask.
        return {"action": ACTION_NONE}

    def _fact_for(self, text: str) -> dict | None:
        """The stored fact this question is about, or None.

        Recall already dropped facts that are not close to the question, and
        it already dropped every other user's facts. The closest remaining
        one is the answer. Ties keep the order recall returned them in.
        """
        facts = self.recalled.get("facts") or []
        if not facts:
            return None
        return max(facts, key=lambda row: row.get("score", 0))

    def _advance_booking(self) -> dict:
        """Ask for the next missing slot, or ask the user to confirm.

        Never calls book_it_slot. That happens on the confirm turn.
        """
        missing = self.missing_booking_slots()
        if missing:
            self.awaiting_confirmation = False
            name = missing[0]
            return self._ask(name, BOOKING_ASKS[name])
        self.awaiting_confirmation = True
        return self._ask(None, CONFIRM_BOOKING)

    def _ask(self, slot: str | None, prompt: str) -> dict:
        return {"action": ACTION_ASK, "slot": slot, "prompt": prompt}

    def _tool(self, name: str, result) -> dict:
        return {"action": ACTION_TOOL, "name": name, "result": result}

    def _db_path(self):
        # Imported here so a test that never books pays no import-time cost,
        # and so the default stays the single path named in src.config.
        from src.config import DB_PATH

        return self.db_path if self.db_path is not None else DB_PATH

    def reply(self, text: str, decision: dict) -> str:
        """The one sentence the user reads for this turn.

        An ask is already that sentence, and a turn that called nothing has
        a fixed line. Both return without a model call and without a log
        line. A tool result is phrased by chat() at temperature 0, which logs
        one line with call_type "reply" and the API's own token counts.
        """
        action = decision.get("action")
        if action == ACTION_ASK:
            return decision["prompt"]
        if action == ACTION_NONE:
            intent = self.history[-1]["intent"] if self.history else ""
            return REPLIES.get(intent, REPLIES["out_of_scope"])
        if action != ACTION_TOOL:
            raise ValueError(f"unknown action: {action!r}")

        name = decision["name"]
        result = decision["result"]
        # A recalled fact is already a sentence. Phrasing it with the model
        # would only risk adding a place or a laptop the user never stated.
        if name == "recall_fact":
            template = FACT_REPLIES.get(result["key"], "Your {key} is {value}.")
            return template.format(key=result["key"], value=result["value"])
        if name == "search_kb":
            shown = self._passages(result)
        elif name == "get_ticket":
            shown = self._ticket_text(result)
        elif name == "raise_ticket":
            shown = (
                f"Raised {result['ticket_id']} for {self.user}, "
                f"status {result['status']}, issue: {result['issue_type']}."
            )
        elif name == "book_it_slot":
            shown = (
                f"Booked slot {result['slot_id']} at {result['location']} "
                f"on {result['date']} at {result['time']}, status {result['status']}."
            )
        else:
            raise ValueError(f"unknown tool: {name!r}")

        prompt = f"User message: {text}\n\nTool result:\n{shown}"
        return chat(
            prompt,
            system=REPLY_SYSTEM,
            temperature=0.0,
            call_type="reply",
            user=self.user,
            session=self.session,
            turn=self.turn,
        ).strip()

    def _passages(self, hits: list[dict]) -> str:
        """The passages the reply may quote.

        A hit whose dense score is below the relevance cut-off is dropped.
        Hybrid scores are rank-fusion scores, not cosines, so only a dense
        score is compared. When nothing remains, the model is told that
        rather than handed a weak passage.
        """
        kept = []
        for hit in hits:
            if hit.get("mode") == "dense" and hit.get("score", 0) < RELEVANCE_THRESHOLD:
                continue
            kept.append(hit)
        if not kept:
            return NO_PASSAGE
        lines = []
        for hit in kept:
            lines.append(f"[{hit['doc_id']}] {hit['text']}")
        return "\n".join(lines)

    def _ticket_text(self, result) -> str:
        if result == NOT_FOUND or not isinstance(result, dict):
            return "No ticket was found for this user."
        return (
            f"Ticket {result['ticket_id']} for {result['owner_user']}, "
            f"status {result['status']}, issue: {result['issue_type']}."
        )

    def note_booked(self) -> None:
        """Call after a confirmed booking has been written.

        The place and the fault stay, because a follow-up such as "and for
        Friday?" still refers to them. The day and the time do not, so the
        next booking cannot reuse a slot the user did not ask for.
        """
        self.pending_booking = False
        self.awaiting_confirmation = False
        self.slots["date"] = None
        self.slots["time"] = None

    def end(self, reflection: str) -> dict:
        """Close the session: keep the summary as an episode, then drop it."""
        return self.memory.end_session(reflection)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one or more helpdesk turns")
    parser.add_argument("text", nargs="+", help="one or more user messages, in order")
    parser.add_argument("--user", default="U1")
    parser.add_argument("--session", default="A")
    parser.add_argument(
        "--forget",
        action="append",
        default=[],
        help="forget this value for the user before the turns run",
    )
    parser.add_argument(
        "--end",
        metavar="REFLECTION",
        help="close the session afterwards and store this reflection",
    )
    args = parser.parse_args()

    session = Session(args.user, args.session)
    for value in args.forget:
        forgotten = session.memory.forget(value)
        print(json.dumps({"forgot": value, "removed": forgotten}))

    for text in args.text:
        labelled = session.ingest(text)
        decision = session.decide(text)
        said = session.reply(text, decision)
        recalled = labelled.pop("recalled")
        print(
            json.dumps(
                {
                    "nlu": labelled,
                    "decision": decision,
                    "reply": said,
                    "recalled": {
                        "facts": [row["value"] for row in recalled["facts"]],
                        "episodes": [row["text"] for row in recalled["episodes"]],
                    },
                },
                indent=2,
                default=str,
            )
        )

    if args.end:
        reflection = session.end(args.end)
        print(json.dumps({"reflection_id": reflection["id"], "text": reflection["text"]}))


if __name__ == "__main__":
    main()
