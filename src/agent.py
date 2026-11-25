"""Dialogue state for one helpdesk session.

Part 1 labels the latest user message with src.nlu.understand.
Part 2 keeps the slots from earlier turns and fills them in as the user
answers, so "Thursday, please." is remembered when they later say yes.
Part 3 decides what to do with that turn. It asks for a slot that is still
missing, or it calls one tool. A fault is not raised on the turn it is
reported: location, date and time are collected first, and the ticket and
the visit are written together on the turn the user says yes. It never
writes a booking on the turn that supplied the last slot.
Part 4 turns that decision into the one sentence the user reads. An ask is
already a sentence, so it is returned as it stands and no model is called.
A tool result is phrased by the chat model, which is the only model call
this module makes besides the one in understand().

    python -m src.agent "The Pune office."
"""

from __future__ import annotations

import argparse
import json

import re

from src.context_builder import assert_budget, assemble
from src.history import STRATEGIES, window_text
from src.llm import chat
from src.memory import Memory
from src.nlu import SLOT_NAMES, _SAME_TIME, forget_request, understand
from src.search_kb import RELEVANCE_THRESHOLD
from src.tools import (
    NOT_FOUND,
    book_it_slot,
    get_ticket,
    latest_booking,
    latest_ticket,
    raise_ticket,
    search_kb,
    update_slot,
)

# A booking is not complete until the user has named all three.
# fix_issue collects the same three before a ticket is raised, so a fault
# report and a visit share one set of questions.
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
    "out_of_scope": "I can only help with IT issues. What can I help you with?",
    # Used instead of out_of_scope when the message is nothing but a hello,
    # so "Hi" is answered with "Hi" and a payroll question is not.
    "greeting": "Hi. I can only help with IT issues. What can I help you with?",
    # "ok" after a visit was written. Not a new booking, and not a refusal.
    "ack": "You're welcome. Let me know if you need anything else.",
    "confirm": "There is nothing waiting to book.",
    "fix_issue": "That is already raised.",
    # Said when the user asks about themselves and nothing stored matches.
    "recall_fact": "I don't have that stored for you.",
    # Said when they ask what happened before and no episode is close enough.
    "recall_episode": "I don't have a record of that.",
    # Said after a "please forget" turn has deleted the fact.
    "forget": "I've forgotten that.",
}

# How a stored fact is read back. The key is the slot name, not the sentence
# the user asked, so "Where do I sit?" and "Which office am I based at?" share
# one line. A key with no entry is read as "Your <key> is <value>."
FACT_REPLIES = {
    "location": "You sit at {value}.",
    "office": "You sit at the {value} office.",
    "asset_id": "You use {value}.",
    "laptop_model": "Your laptop is a {value}.",
    "os": "Your laptop runs {value}.",
    "floor": "You are on the {value}.",
    "department": "You work in {value}.",
    "manager": "Your manager is {value}.",
    "contact_time": "The best time to call you is {value}.",
    "vpn_client": "You connect with {value}.",
    "personal_phone": "Your personal number is {value}.",
    # A name, a colour, or a badge was stored as the sentence itself.
    "note": "You told me: {value}",
}

# A past event is not a standing fact. "What did we do last time?" is about
# an episode, not the office or the laptop, so it is not scored as either.
_PAST_EVENT = re.compile(
    r"\b(?:last time|previously|earlier|before)\b"
    r"|\bwhat did (?:we|i|you)\b"
    r"|\bwhat happened\b",
    re.IGNORECASE,
)

# The reply instructions live in skills/ and are read from disk on each
# reply, so editing the file changes the next session without a code change.
# system_prompt_v1.md is the persona. book_slot_steps_v1.md is how a visit
# is gathered; it is included only while a booking is in progress, because
# a password question has no use for it.
from src.config import ABLATION_USES, ROOT

SKILLS_DIR = ROOT / "skills"
SYSTEM_PROMPT_PATH = SKILLS_DIR / "system_prompt_v1.md"
BOOKING_STEPS_PATH = SKILLS_DIR / "book_slot_steps_v1.md"


def load_skill(path) -> str:
    """The text of one skill file. Missing or empty is an error, not a blank prompt."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"skill file is empty: {path}")
    return text


def _with_history(prompt: str, history: str) -> str:
    """Put the strategy's window in front of the current turn.

    An empty window leaves the prompt unchanged, so the first turn of a
    session reads exactly as it did before short-term memory existed.
    """
    if not history.strip():
        return prompt
    return f"{history}\n\n{prompt}"


def reply_system(booking: bool) -> str:
    """The system prompt for one reply, loaded from skills/ at call time."""
    text = load_skill(SYSTEM_PROMPT_PATH)
    if booking:
        text = f"{text}\n\n{load_skill(BOOKING_STEPS_PATH)}"
    return text


# Task turns are phrased at temperature 0: a ticket id or a booked time must
# come out the same way on every run. An open turn (a greeting, a fact just
# noted, a refusal) has no fact to pin down, so it uses a little variety.
TASK_TEMPERATURE = 0.0
OPEN_TEMPERATURE = 0.7
OPEN_TOP_P = 0.9

# Shown to the model when retrieval found nothing above the relevance cut-off.
# The model is not handed a weak passage it might quote as if it were an answer.
NO_PASSAGE = "Nothing relevant was found in the knowledge base."

# The sentence a question gets when neither the knowledge base nor this
# user's memory has anything relevant. It is returned as written. The model
# is not called, because a call is what invents an answer.
FALLBACK = "I don't know."

# A citation the reply already carries. Grounding adds one only when the
# model left it off, and never a second copy.
_CITATION = re.compile(r"\[(?:mem:[A-Za-z0-9._-]+|[A-Za-z0-9._-]+#[A-Za-z0-9_.:@-]+)\]")

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

# "What was the last issue I raised?" and "my last booking" name no id.
# The model files both under check_ticket or recall_fact. A booking word
# reads the newest visit; anything else about a ticket reads the newest ticket.
# A message that is only a hello. "Hi, my VPN is down" is a fault, not a
# greeting, because the words after the hello are still there.
_GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|hiya|howdy|good\s+(?:morning|afternoon|evening))"
    r"(?:\s+there)?\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_ACK = re.compile(
    r"^\s*(?:ok|okay|thanks|thank you|thx|cheers|got it|alright|cool)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    """True when the message is only a hello, with no request after it."""
    return _GREETING.match(text or "") is not None


def is_ack(text: str) -> bool:
    """True when the message is only thanks or "ok", with no request after it."""
    return _ACK.match(text or "") is not None


def _booking_confirmation(said: str, result: dict, updated: bool) -> str:
    """Replace a reply that asks to confirm a visit already written.

    The model is told the row exists and still sometimes ends with
    "Shall I book that visit?". A reply with no question is kept, even when
    it shortens the place: the tool text already named the full location.
    """
    if not isinstance(result, dict):
        return said
    lowered = (said or "").lower()
    asks_again = "?" in (said or "") or "shall i book" in lowered
    if said and not asks_again:
        return said
    place = result.get("location") or ""
    day = result.get("date") or ""
    clock = result.get("time") or ""
    verb = "Updated" if updated else "Booked"
    return f"{verb} the visit at {place} on {day} at {clock}."


_LAST_BOOKING = re.compile(
    r"\b(?:last|latest|recent|previous|my)\b.{0,40}\b(?:booking|visit|appointment|slot)\b"
    r"|\b(?:booking|visit|appointment|slot)\b.{0,40}\b(?:last|latest|recent|previous)\b"
    r"|\b(?:share|show|give|tell|what|whats|what's)\b.{0,40}\b(?:detail|details|status|info|information)\b.{0,40}\b(?:booking|visit|appointment|slot)\b"
    r"|\b(?:detail|details|status)\b.{0,30}\b(?:booking|visit|appointment|slot)\b"
    r"|\b(?:booking|visit|appointment|slot)\b.{0,30}\b(?:detail|details|status)\b",
    re.IGNORECASE,
)
_LAST_TICKET = re.compile(
    r"\b(?:last|latest|recent|previous|any)\b.{0,40}\b(?:ticket|issue|problem|fault)s?\b"
    r"|\b(?:ticket|issue|problem|fault)s?\b.{0,40}\b(?:last|latest|recent|previous|raised|for me)\b"
    r"|\braised\b.{0,20}\b(?:ticket|issue|problem|fault)s?\b",
    re.IGNORECASE,
)

# A request to change a visit that is already booked. "change my office" is a
# standing fact and is not matched here.
_RESCHEDULE_TIME = re.compile(
    r"\b(?:timing|time of day|what time|the time)\b|\b(?:change|move|reschedule|shift)\s+(?:the\s+|my\s+)?time\b",
    re.IGNORECASE,
)
_RESCHEDULE_DATE = re.compile(
    r"\b(?:change|move|reschedule|shift)\s+(?:the\s+|my\s+)?(?:day|date)\b",
    re.IGNORECASE,
)
_RESCHEDULE_PLACE = re.compile(
    r"\b(?:change|move|reschedule|shift)\s+(?:the\s+)?(?:visit\s+)?(?:to\s+)?(?:location|place|office)\b",
    re.IGNORECASE,
)
_RESCHEDULE = re.compile(
    r"\b(?:"
    r"(?:change|move|reschedule|shift|update)\s+(?:the\s+|my\s+)?(?:timing|time|day|date|slot|booking|visit|appointment)"
    r"|(?:change|move|reschedule|shift)\s+it\b"
    r"|reschedule\b"
    r")",
    re.IGNORECASE,
)


def chunk_cite(hit: dict) -> str:
    """The citation for one retrieved chunk: [doc_id#chunk_id]."""
    from src.grounding import chunk_cite as _chunk_cite

    return _chunk_cite(hit)


def memory_cite(memory_id: str | None) -> str:
    """The citation for one fact or episode: [mem:id]."""
    from src.grounding import memory_cite as _memory_cite

    return _memory_cite(memory_id)


def _with_citation(text: str, cite: str) -> str:
    """Append a citation when the sentence does not already carry one."""
    body = (text or "").rstrip()
    if cite in body:
        return body
    return f"{body} {cite}"


def ground_citations(text: str, allowed: list[str]) -> str:
    """Keep only citations that name a passage the reply was shown.

    The model is told to cite, and it sometimes cites a document it was not
    given. Those tokens are removed. When none of the allowed citations
    survived, the first allowed one is appended, so the reply still names
    the passage it was written from.
    """
    body = text or ""
    kept = [cite for cite in allowed if cite and cite in body]
    for found in _CITATION.findall(body):
        if found not in allowed:
            body = body.replace(found, "")
    body = re.sub(r"[ \t]{2,}", " ", body).strip()
    body = re.sub(r"\s+([.!?])", r"\1", body)
    if not kept and allowed:
        body = _with_citation(body, allowed[0])
    return body


def _jsonable(value):
    """A tool result safe to write into the session JSON file.

    Search hits carry the chunk text. Embeddings are not on these dicts,
    but a value that is not JSON (a Path, a set) is stringified rather than
    crashing the turn that already succeeded.
    """
    try:
        json.dumps(value, default=str)
    except TypeError:
        return str(value)
    return json.loads(json.dumps(value, default=str))


class Session:
    """Slots gathered across the turns of one user in one session.

    Slots belong to (user, session). A second user, or the same user in a
    later session, starts empty and cannot read these.
    """

    def __init__(
        self,
        user: str,
        session: str,
        db_path=None,
        memory=None,
        strategy: str = "full",
        config: str | None = None,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if config is not None and config not in ABLATION_USES:
            raise ValueError(f"config must be one of {tuple(ABLATION_USES)}, got {config!r}")
        self.user = user
        self.session = session
        # Which earlier turns a reply is shown. The slot store is not trimmed.
        self.strategy = strategy
        # Which memory types this run may read. None means all of them, which
        # is how the page and the tests behave. An ablation run names one.
        self.config = config
        self.uses = ABLATION_USES.get(config, {"kb": True, "semantic": True, "episodic": True})
        # None logs under the process RUN_ID. A strategy run passes its own.
        self.run_id: str | None = None
        self.turn = 0
        self.slots: dict[str, str | None] = {name: None for name in SLOT_NAMES}
        self.pending_booking = False
        self.awaiting_confirmation = False
        # True only after a fix_issue turn. Confirm raises a ticket then.
        # A plain booking, even one that named the fault, does not.
        self.pending_ticket = False
        # The visit a "change the timing" turn is editing. None means the
        # next confirm inserts a new row rather than changing one.
        self.reschedule_slot_id: int | None = None
        # Set when a visit row was written this session. A later "ok" is
        # thanks for that, not a request to book something else.
        self.last_booked: dict | None = None
        self.history: list[dict] = []
        # None uses the shared helpdesk database. Tests pass their own file.
        self.db_path = db_path
        # None uses the shared memory folder. Tests pass an isolated store.
        self.memory = memory if memory is not None else Memory(user, session)
        # What load() returned for the latest turn. Empty until the first one.
        self.recalled: dict = {"summary": "", "facts": [], "episodes": []}
        # Model calls for the turn currently being answered. ingest() resets it.
        self.calls: list[dict] = []
        # The tool decide() ran for this turn, or None when it ran none.
        self.tool_call: dict | None = None
        # Facts written by the latest ingest. The thanks line is only said
        # when this is not empty, so "I'll remember that" matches a real row.
        self.facts_written: list[dict] = []
        # Set when this turn asked to delete a fact. reply() reads it.
        self.forgotten: dict | None = None
        # The prompt built for the latest model call, with its token table.
        # None until a turn actually calls the model.
        self.context = None

    def ingest(self, text: str) -> dict:
        """Label one message and merge any new slot values into this session.

        Returns the NLU result plus the session slots after the merge.
        understand() is the only model call. A rule hit makes none.
        """
        self.turn += 1
        # Loaded before the sentence is labelled, because a later session
        # answers from what earlier sessions stored. The label itself still
        # comes from the sentence alone.
        self.recalled = self.memory.load(
            text,
            semantic=self.uses["semantic"],
            episodic=self.uses["episodic"],
            turn=self.turn,
            run_id=self.run_id,
            config=self._config_name(),
        )
        # Calls made while labelling and answering this turn. reply() writes
        # them into the session file. A new turn starts empty.
        self.calls = []
        self.tool_call = None
        self.forgotten = None
        # "Please forget my personal phone number" is a deletion, not a fact
        # and not a question. It is handled here so the model cannot label
        # it as provide_info and store the number a second time.
        requested = forget_request(text)
        if requested is not None:
            self.forgotten = self.memory.forget_request(requested["key"], requested["value"])
            self.facts_written = []
            self.history.append({
                "turn": self.turn,
                "text": text,
                "intent": "forget",
                "source": "rule",
                "message_slots": {},
                "slots": self.slots_so_far(),
            })
            return {
                "intent": "forget",
                "source": "rule",
                "message_slots": {},
                "slots": self.slots_so_far(),
                "recalled": self.recalled,
                "turn": self.turn,
                "forgotten": self.forgotten,
            }
        understood = understand(
            text,
            user=self.user,
            session=self.session,
            turn=self.turn,
            trace=self.calls,
            pending_question=self._pending_question(),
            run_id=self.run_id,
            config=self._config_name(),
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
        if understood["intent"] == "book_slot" and self._reschedule(understood["slots"]):
            # The sentence asked to change a visit. The slot it named is
            # cleared only when no new value was given, and the existing
            # visit is remembered so confirm updates that row.
            self.history[-1]["slots"] = self.slots_so_far()
            self.pending_booking = True
            self.awaiting_confirmation = False
            self._arm_reschedule()
        # A stated fact is written now, so a later session can read it. A
        # greeting, a yes, or a no writes nothing.
        self.facts_written = self.memory.record_turn(
            text,
            understood["intent"],
            understood["slots"],
            self.turn,
            run_id=self.run_id,
            config=self._config_name(),
        )
        self._note_decision(text, understood["intent"])
        return {
            "intent": understood["intent"],
            "source": understood["source"],
            "message_slots": dict(understood["slots"]),
            "slots": self.slots_so_far(),
            "recalled": self.recalled,
            "turn": self.turn,
        }


    def _note_decision(self, text: str, intent: str) -> None:
        """Store a choice the user just made, such as a refresh over a repair.

        It is an episode, not a standing fact. The next session asks what was
        decided, and that question reads episodes. A profile sentence is not
        a decision.
        """
        if intent != "provide_info":
            return
        if not re.search(r"\brather than\b|\bgo ahead with the refresh\b", text, re.IGNORECASE):
            return
        self.memory.remember_event(
            text,
            "event",
            self.turn,
            run_id=self.run_id,
            config=self._config_name(),
        )

    def _config_name(self) -> str:
        """The name written on this turn's log lines.

        An ablation run names its config, so the four runs can be split apart
        in the log. Anything else logs the history strategy, as before.
        """
        return self.config or self.strategy

    def slots_so_far(self) -> dict[str, str | None]:
        """A copy of the carried slots. Callers cannot mutate the session."""
        return dict(self.slots)

    def missing_booking_slots(self) -> list[str]:
        """location, date, time that are still empty, in that order."""
        return [name for name in BOOKING_SLOTS if not self.slots.get(name)]

    def _reschedule(self, message_slots: dict) -> bool:
        """True when this sentence asks to change a visit that is already booked.

        The new value, when the sentence names one, is already in the slots.
        The part that was not named is restored from the booked visit and
        then cleared, so the next question asks only for that part.
        """
        if not self.history:
            return False
        text = self.history[-1].get("text") or ""
        if not _RESCHEDULE.search(text):
            return False
        # The existing day and time stay. _arm_reschedule() drops only the
        # part this sentence asked to change, after it has copied them back
        # from the booked visit. Clearing them here would forget the day of
        # a time change, and the confirm turn would insert a second visit.
        return True

    def _arm_reschedule(self) -> None:
        """Point this change at the user's newest visit, and refill its day and time.

        note_booked() clears the day and the time so a brand new visit cannot
        reuse them. A change is not a new visit: the day and the time stay,
        and only the part the user is replacing is collected again.
        """
        if self.reschedule_slot_id is not None:
            return
        found = latest_booking(
            self.user,
            config=self._config_name(),
            session=self.session,
            turn=self.turn,
            run_id=self.run_id,
            db_path=self._db_path(),
        )
        if not isinstance(found, dict):
            return
        self.reschedule_slot_id = found["slot_id"]
        if not self.slots.get("date"):
            self.slots["date"] = found["date"]
        if not self.slots.get("time"):
            self.slots["time"] = found["time"]
        if not self.slots.get("location"):
            self.slots["location"] = found["location"]
        # The sentence already said which part to replace, and named no new
        # value. Drop that part now that the rest has been restored.
        text = self.history[-1].get("text") or "" if self.history else ""
        message_slots = self.history[-1].get("message_slots") or {} if self.history else {}

        def stated(name: str) -> bool:
            value = message_slots.get(name)
            return isinstance(value, str) and bool(value.strip())

        if _RESCHEDULE_TIME.search(text) and not stated("time"):
            self.slots["time"] = None
        elif _RESCHEDULE_DATE.search(text) and not stated("date"):
            self.slots["date"] = None
        elif _RESCHEDULE_PLACE.search(text) and not stated("location"):
            self.slots["location"] = None

    def _filled_open_booking_slot(self) -> bool:
        """True when this turn put a value in location, date, or time.

        Used so a mislabelled answer still advances the visit. A question
        that names no booking slot does not.
        """
        if not self.history:
            return False
        message_slots = self.history[-1].get("message_slots") or {}
        return any(
            isinstance(message_slots.get(name), str) and message_slots.get(name).strip()
            for name in BOOKING_SLOTS
        )

    def _merge(self, intent: str, message_slots: dict) -> None:
        """Copy stated slots in. A new value replaces the one carried so far.

        confirm and reject are answers about the booking, not new facts, so
        they change no slot. reject also drops the unfinished booking: the
        user backed out, and the next request must collect its own day.
        """
        if intent == "reject":
            self.pending_booking = False
            self.awaiting_confirmation = False
            self.pending_ticket = False
            self.reschedule_slot_id = None
            return

        if intent != "confirm":
            for name in CARRIED_SLOTS:
                value = message_slots.get(name)
                if isinstance(value, str) and value.strip():
                    self.slots[name] = value.strip()

        if intent == "book_slot":
            self.pending_booking = True
        elif intent == "fix_issue" and not self.slots.get("ticket_id"):
            # The visit questions start on this turn, not on the later
            # decide() call. The next sentence is an answer to one of them,
            # so a provide_info mislabel still has a pending booking to fill.
            self.pending_booking = True
            self.pending_ticket = True
        elif intent in ("provide_info", "ask_kb") and self.pending_booking:
            # An answer to "which office / which day / what time" is sometimes
            # labelled provide_info, or ask_kb when the reply is a bare time
            # such as "12 PM". The visit is already underway, so the value
            # still fills the slot and the next question is still asked.
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
        instead of writing. A fault report is that same visit: the ticket
        is raised on the confirm turn, not on the turn that named the fault.
        A tool that refuses (no confirmation, unknown ticket) is not called
        at all, so it cannot log a line.
        """
        if not self.history:
            raise RuntimeError("decide() needs ingest() to have run first")
        intent = self.history[-1]["intent"]

        if intent == "forget":
            # The deletion already happened in ingest(). Nothing to call.
            return {"action": ACTION_NONE}

        if intent == "ask_kb":
            # "12 PM" after "What time of day works?" is the open slot, not a
            # knowledge-base question. Search only when it filled nothing the
            # visit is still waiting for.
            if self.pending_booking and (
                self._filled_open_booking_slot() or self.awaiting_confirmation
            ):
                return self._advance_booking()
            # no_rag answers from the conversation alone. The index is not
            # searched, and the reply falls back rather than guessing.
            if not self.uses["kb"]:
                return {"action": ACTION_NONE}
            return self._tool(
                "search_kb",
                search_kb(
                    text,
                    config=self._config_name(),
                    user=self.user,
                    session=self.session,
                    turn=self.turn,
                    run_id=self.run_id,
                ),
                {"query": text},
            )

        if intent == "check_ticket":
            return self._lookup_record(text)

        if intent == "recall_fact" and self._asks_for_record(text):
            # "Last ticket" is sometimes labelled as a memory question. The
            # ticket and the visit live in the helpdesk tables, not in facts.
            return self._lookup_record(text)

        if intent == "fix_issue":
            if self.slots.get("ticket_id"):
                return {"action": ACTION_NONE}
            # Same questions as a visit. The ticket is raised only when the
            # user confirms a place, a day and a time.
            self.pending_booking = True
            self.pending_ticket = True
            return self._advance_booking()

        if intent == "book_slot":
            # "The same time is fine" names no clock time. The visit being
            # changed still has one, and that is the time being accepted.
            if _SAME_TIME.search(text) and not self.slots.get("time") and self.last_booked:
                self.slots["time"] = self.last_booked.get("time")
            return self._advance_booking()

        if intent == "provide_info" and self.pending_booking and self.missing_booking_slots():
            # The user answered the question the previous turn asked. Keep
            # collecting. A provide_info turn with nothing left to ask is a
            # standing fact, not a yes, so it does not confirm the visit.
            return self._advance_booking()

        if intent == "recall_fact" or self._asks_for_episode(text):
            # A past event lives in the episode log, not in the fact store
            # and not in the ticket table. "What did we do last time?" is
            # often labelled out_of_scope, so the wording is checked too.
            if self._asks_for_episode(text):
                # Episodes are off until hybrid_mem. The question is then a
                # miss, which is the point of the earlier configurations.
                if not self.uses["episodic"]:
                    return {"action": ACTION_NONE}
                episode = self._episode_for(text)
                if episode is None:
                    return {"action": ACTION_NONE}
                return self._tool(
                    "recall_episode",
                    episode,
                    {"text": episode.get("text"), "type": episode.get("type")},
                )
            # Answered from memory, not from the knowledge base. A question
            # states no new fact, so nothing is written on this turn either.
            # Facts are off for no_rag and hybrid. hybrid_sem is the first
            # configuration that may answer from the fact store.
            if not self.uses["semantic"]:
                return {"action": ACTION_NONE}
            fact = self._fact_for(text)
            if fact is None:
                return {"action": ACTION_NONE}
            return self._tool("recall_fact", fact, {"key": fact.get("key"), "value": fact.get("value")})

        if intent == "confirm":
            if self.pending_booking and self.awaiting_confirmation and not self.missing_booking_slots():
                # A fault report has an issue and no ticket yet, so the
                # ticket is raised now and the visit is attached to it. A
                # plain booking has no fault, and must not invent a ticket.
                # A visit that followed an already-raised ticket reuses it.
                if self.pending_ticket and not self.slots.get("ticket_id"):
                    self._raise_current_issue(text)
                if self.reschedule_slot_id is not None:
                    booked = update_slot(
                        self.user,
                        self.reschedule_slot_id,
                        location=self.slots["location"],
                        date=self.slots["date"],
                        time_of_day=self.slots["time"],
                        config=self._config_name(),
                        session=self.session,
                        turn=self.turn,
                        run_id=self.run_id,
                        db_path=self._db_path(),
                    )
                    tool_name = "update_slot"
                else:
                    booked = book_it_slot(
                        self.user,
                        self.slots.get("issue_type") or "unspecified",
                        self.slots["location"],
                        self.slots["date"],
                        self.slots["time"],
                        confirmed=True,
                        ticket_id=self.slots.get("ticket_id"),
                        asset_id=self.slots.get("asset_id"),
                        config=self._config_name(),
                        session=self.session,
                        turn=self.turn,
                        run_id=self.run_id,
                        db_path=self._db_path(),
                    )
                    tool_name = "book_it_slot"
                self.note_booked(booked if isinstance(booked, dict) else None)
                # The arguments below were copied before this call. They still
                # name the day and the time; the slots themselves no longer do.
                self.memory.remember_event(
                    (
                        f"Booked a visit at {booked['location']} on {booked['date']} "
                        f"at {booked['time']}."
                    ),
                    "booking",
                    self.turn,
                    run_id=self.run_id,
                    config=self._config_name(),
                )
                return self._tool(
                    tool_name,
                    booked,
                    {
                        "location": self.slots["location"],
                        "date": self.slots["date"],
                        "time": self.slots["time"],
                        "ticket_id": booked.get("ticket_id") if isinstance(booked, dict) else None,
                        "slot_id": booked.get("slot_id") if isinstance(booked, dict) else None,
                        "confirmed": True,
                    },
                )
            return {"action": ACTION_NONE}

        # reject, provide_info, out_of_scope: nothing to call and nothing to ask.
        return {"action": ACTION_NONE}

    def _asks_for_record(self, text: str) -> bool:
        """True when the sentence asks for a ticket or a booking, not a fact."""
        return bool(_LAST_TICKET.search(text) or _LAST_BOOKING.search(text))

    def _lookup_record(self, text: str) -> dict:
        """Read the ticket or visit this sentence is asking about.

        A named ticket id still reads that ticket. "My last booking" reads
        the newest visit, because a ticket row cannot answer it. "My last
        issue" names no id, so it reads the newest ticket owned by this user.
        A status question that names neither is asked back.
        """
        if _LAST_BOOKING.search(text):
            found = latest_booking(
                self.user,
                config=self._config_name(),
                session=self.session,
                turn=self.turn,
                run_id=self.run_id,
                db_path=self._db_path(),
            )
            return self._tool("get_booking", found, {"user": self.user, "latest": True})

        ticket_id = self.slots.get("ticket_id")
        if not ticket_id and _LAST_TICKET.search(text):
            found = latest_ticket(
                self.user,
                config=self._config_name(),
                session=self.session,
                turn=self.turn,
                run_id=self.run_id,
                db_path=self._db_path(),
            )
            ticket_id = found["ticket_id"] if isinstance(found, dict) else None
            return self._tool(
                "get_ticket",
                found,
                {"ticket_id": ticket_id, "user": self.user, "latest": True},
            )
        if not ticket_id:
            return self._ask(None, "Which ticket should I look up?")
        found = get_ticket(
            ticket_id,
            self.user,
            config=self._config_name(),
            session=self.session,
            turn=self.turn,
            run_id=self.run_id,
            db_path=self._db_path(),
        )
        return self._tool("get_ticket", found, {"ticket_id": ticket_id, "user": self.user})

    def _asks_for_episode(self, text: str) -> bool:
        """True when the sentence asks what happened before, not for a fact.

        "Where do I sit?" is a standing fact. "What did we do last time?"
        is an episode. A ticket or a booking still goes to the helpdesk
        tables, so those sentences are not treated as episodes.
        """
        if self._asks_for_record(text):
            return False
        return _PAST_EVENT.search(text or "") is not None

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

    def _episode_for(self, text: str) -> dict | None:
        """The past event this question is about, or None.

        load() already dropped other users and episodes below the cosine
        cut-off, then ranked what remained. The first one is the answer.
        """
        episodes = self.recalled.get("episodes") or []
        if not episodes:
            return None
        return episodes[0]

    def _raise_current_issue(self, text: str) -> dict:
        """Insert the ticket for the fault this session is holding."""
        issue = self.slots.get("issue_type") or self.slots.get("error_code") or text
        ticket = raise_ticket(
            self.user,
            issue,
            asset_id=self.slots.get("asset_id"),
            config=self._config_name(),
            session=self.session,
            turn=self.turn,
            run_id=self.run_id,
            db_path=self._db_path(),
        )
        self.slots["ticket_id"] = ticket["ticket_id"]
        self.memory.remember_event(
            f"Raised {ticket['ticket_id']} for {ticket['issue_type']}.",
            "ticket",
            self.turn,
            run_id=self.run_id,
            config=self._config_name(),
        )
        return ticket

    def _pending_question(self) -> str | None:
        """The question the previous turn asked, if it is still unanswered.

        Sent with the next sentence so the label sees what the user is
        answering. Nothing is pending before the first ask, and a finished
        or rejected visit has no question left.
        """
        if not self.pending_booking:
            return None
        if self.awaiting_confirmation:
            return CONFIRM_BOOKING
        missing = self.missing_booking_slots()
        if not missing:
            return None
        return BOOKING_ASKS[missing[0]]

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

    def _tool(self, name: str, result, arguments: dict | None = None) -> dict:
        # Kept for the transcript. The result is what the tool returned,
        # before the model phrases it into the sentence the user reads.
        self.tool_call = {
            "name": name,
            "arguments": arguments or {},
            "result": _jsonable(result),
        }
        return {"action": ACTION_TOOL, "name": name, "result": result}

    def _db_path(self):
        # Imported here so a test that never books pays no import-time cost,
        # and so the default stays the single path named in src.config.
        from src.config import DB_PATH

        return self.db_path if self.db_path is not None else DB_PATH

    def reply(self, text: str, decision: dict) -> str:
        """The one sentence the user reads for this turn.

        An ask is already that sentence, so it is returned with no model
        call. A turn that called nothing is an open turn: the fixed line is
        reworded by chat() at temperature 0.7 and top-p 0.9. A tool result
        is a task turn, phrased by chat() at temperature 0. Both model calls
        log one line with call_type "reply" and the API's own token counts,
        and both read their instructions from skills/.
        """
        action = decision.get("action")
        if action == ACTION_ASK:
            said = decision["prompt"]
            self._log_exchange(text, said)
            return said
        if action == ACTION_NONE:
            said = self._open_reply(text)
            self._log_exchange(text, said)
            return said
        if action != ACTION_TOOL:
            raise ValueError(f"unknown action: {action!r}")

        name = decision["name"]
        result = decision["result"]
        # A recalled fact is already a sentence. Phrasing it with the model
        # would only risk adding a place or a laptop the user never stated.
        if name == "recall_fact":
            template = FACT_REPLIES.get(result["key"], "Your {key} is {value}.")
            said = template.format(key=result["key"], value=result["value"])
            said = _with_citation(said, memory_cite(result.get("id")))
            self._log_exchange(text, said)
            return said
        if name == "recall_episode":
            # The episode text is already the sentence. The model is not
            # asked to rephrase it, for the same reason a fact is not.
            said = f"Last time, {result['text']}"
            said = _with_citation(said, memory_cite(result.get("id")))
            self._log_exchange(text, said)
            return said
        if name == "search_kb":
            kept = self._kept_hits(result)
            # A dense hit below the cut-off is not a passage. An empty list
            # is not that: the caller handed no hits to judge, and the model
            # is told so rather than skipped.
            if result and not kept:
                self._log_exchange(text, FALLBACK)
                return FALLBACK
            shown = self._passages(kept)
        elif name == "get_ticket":
            shown = self._ticket_text(result)
        elif name == "get_booking":
            shown = self._booking_text(result)
        elif name == "raise_ticket":
            shown = (
                f"Raised {result['ticket_id']} for {self.user}, "
                f"status {result['status']}, issue: {result['issue_type']}."
            )
        elif name in ("book_it_slot", "update_slot"):
            verb = "Updated" if name == "update_slot" else "Booked"
            shown = (
                f"{verb} slot {result['slot_id']} at {result['location']} "
                f"on {result['date']} at {result['time']}, status {result['status']}. "
                "This visit is already written. Tell the user it is booked. "
                "Do not ask a question."
            )
        else:
            raise ValueError(f"unknown tool: {name!r}")

        # A tool result is a task turn: temperature 0. The booking steps are
        # included when the reply confirms a visit that was just written, or
        # while one is still being collected. A ticket lookup is neither.
        booking = self.pending_booking or name in ("book_it_slot", "update_slot")
        # The prompt is built to the budget before it is sent. The system
        # prompt, the tool result and the current message are kept. Memory,
        # history and the extra chunks are dropped, lowest priority first,
        # until the prompt leaves 500 tokens for the reply.
        built = assemble(
            system=reply_system(booking=booking),
            tool_result=shown,
            facts=self.recalled.get("facts") if self.uses["semantic"] else None,
            episodes=self.recalled.get("episodes") if self.uses["episodic"] else None,
            history=self._history_block(),
            chunks=kept if name == "search_kb" else None,
            current=text,
        )
        assert_budget(built)
        self.context = built
        said = chat(
            built.prompt,
            system=reply_system(booking=booking),
            temperature=TASK_TEMPERATURE,
            call_type="chat",
            config=self._config_name(),
            user=self.user,
            session=self.session,
            turn=self.turn,
            trace=self.calls,
            run_id=self.run_id,
            context=built,
        ).strip()
        if name in ("book_it_slot", "update_slot"):
            # The model is told not to ask again, and still sometimes repeats
            # "Shall I book that visit?". The booking is already done, so
            # that sentence is replaced by the row that was written.
            said = _booking_confirmation(said, result, updated=name == "update_slot")
        if name == "search_kb" and kept:
            # The passage shown to the model is the one the citation names.
            # A citation the model invented, for a chunk it was not shown,
            # is dropped and replaced by that passage's own id. No passage
            # means no citation, so an empty result is left as the model wrote it.
            said = ground_citations(said, [chunk_cite(hit) for hit in kept])
        self._log_exchange(text, said)
        return said

    def _open_reply(self, text: str) -> str:
        """Phrase a turn that called no tool.

        The fixed line is what must be said (a refusal, a stored "thanks").
        It is handed to the model as the required reply, at temperature 0.7
        and top-p 0.9, so the wording can vary while the meaning cannot.
        The persona is read from skills/ for this call too.
        """
        intent = self.history[-1]["intent"] if self.history else ""
        if intent == "forget":
            return REPLIES["forget"]
        # A bare hello shares the out_of_scope intent with a payroll question.
        # Only the hello gets the greeting line, so the refusal stays a refusal.
        if intent == "out_of_scope" and is_greeting(text):
            intent = "greeting"
        # "What did we do last time?" with nothing stored is not a refusal
        # of a payroll question. It is a memory miss.
        if self._asks_for_episode(text) and intent in ("out_of_scope", "recall_fact"):
            # Nothing stored is close enough. The fallback is exact, and the
            # model is not asked to soften it into a guess.
            return FALLBACK
        # A question about the user that matched nothing stored. Same reason:
        # the model invents an office when it is asked to reword the miss.
        if intent == "recall_fact":
            return FALLBACK
        # "I'll remember that" is only true when a row was written. A
        # provide_info turn that stored nothing gets the refusal instead,
        # so the bot does not claim to have kept a colour it dropped.
        # A turn that did store a fact returns the line as written. The
        # persona tells the model to say "I don't know." whenever no tool
        # result was fetched, and the model obeys that instead of the
        # required thanks.
        if intent == "provide_info":
            if not self.facts_written:
                return REPLIES["recall_fact"]
            return REPLIES["provide_info"]
        # "ok" after a written visit is thanks. The same word with nothing
        # booked stays a short acknowledgement, not a refusal and not
        # "there is nothing waiting to book".
        if is_ack(text) and intent in ("out_of_scope", "confirm"):
            intent = "ack"
        required = REPLIES.get(intent, REPLIES["out_of_scope"])
        # The window is not shown here. A required line is a refusal, a
        # thanks, or "nothing waiting to book", and the model was copying an
        # earlier "Booked slot" line out of the window instead of saying it.
        # Memory is left out for the same reason: a stored office would be
        # worked into a reply that is meant to be the fixed line.
        built = assemble(
            system=reply_system(booking=False),
            current=(
                f"User message: {text}\n\n"
                "Required reply, which you may reword but must not change the meaning of:\n"
                f"{required}"
            ),
        )
        assert_budget(built)
        self.context = built
        return chat(
            built.prompt,
            system=reply_system(booking=False),
            temperature=OPEN_TEMPERATURE,
            top_p=OPEN_TOP_P,
            call_type="chat",
            config=self._config_name(),
            user=self.user,
            session=self.session,
            turn=self.turn,
            trace=self.calls,
            run_id=self.run_id,
            context=built,
        ).strip()

    def _history_block(self) -> str:
        """The earlier turns this strategy shows, not the message being answered.

        summary also folds in the turns the window no longer quotes, so the
        notes stay current. That is one model call, logged as "summary".
        full and trim only slice the list and make no call.
        """
        if self.strategy == "summary":
            done = [
                {"user": turn.get("text", ""), "bot": turn.get("bot", "")}
                for turn in self.history[:-1]
                if turn.get("bot")
            ]
            # The rolling notes cover everything except the two turns the
            # prompt still quotes. Each new turn beyond those two is folded
            # in once, rather than the whole session being re-summarised.
            pending = done[: max(0, len(done) - 2)]
            fresh = pending[self.memory.working.covered :]
            if fresh:
                self.memory.working.update(
                    fresh,
                    turn=self.turn,
                    trace=self.calls,
                    run_id=self.run_id,
                    config=self._config_name(),
                )
        return window_text(self.history, self.strategy, self.memory.working.summary)

    def _log_exchange(self, user_text: str, reply: str) -> None:
        """Append this turn to the session's JSON file.

        The intent is the one ingest() stored for this turn. A reply made
        with no ingest has nothing to file, so it is skipped rather than
        inventing a turn number.
        """
        if not self.history:
            return
        latest = self.history[-1]
        # A test, or a caller, may hand a partial history row. The transcript
        # still wants a turn number, and the session counter is the one this
        # reply was made under.
        turn = latest["turn"] if "turn" in latest else self.turn
        # The next turn's window reads this reply. A row handed in by a test
        # may have no text, and still gets the reply so the window is honest.
        latest["bot"] = reply
        self.memory.conversation.append(
            user_text,
            reply,
            turn=turn,
            intent=latest.get("intent", ""),
            calls=self.calls,
            tool=self.tool_call,
        )

    def _kept_hits(self, hits: list[dict]) -> list[dict]:
        """The hits the reply may quote.

        A dense hit below the relevance cut-off is dropped. Hybrid scores
        are rank-fusion scores, not cosines, so only a dense score is
        compared. An empty list means the question has no passage.
        """
        kept = []
        for hit in hits or []:
            if hit.get("mode") == "dense" and hit.get("score", 0) < RELEVANCE_THRESHOLD:
                continue
            kept.append(hit)
        return kept

    def _passages(self, hits: list[dict]) -> str:
        """The passages the reply may quote, each labelled with its cite id.

        The id is [doc_id#chunk_id]. The reply has to carry the same token,
        so a reader can check it against the index.
        """
        if not hits:
            return NO_PASSAGE
        lines = []
        for hit in hits:
            lines.append(f"{chunk_cite(hit)} {hit['text']}")
        return "\n".join(lines)

    def _ticket_text(self, result) -> str:
        if result == NOT_FOUND or not isinstance(result, dict):
            return "No ticket was found for this user."
        return (
            f"Ticket {result['ticket_id']} for {result['owner_user']}, "
            f"status {result['status']}, issue: {result['issue_type']}."
        )

    def _booking_text(self, result) -> str:
        if result == NOT_FOUND or not isinstance(result, dict):
            return "No booking was found for this user."
        return (
            f"Booking {result['slot_id']} for {result['owner_user']} "
            f"at {result['location']} on {result['date']} at {result['time']}, "
            f"status {result['status']}."
        )

    def note_booked(self, booked: dict | None = None) -> None:
        """Call after a confirmed booking has been written.

        The place and the fault stay, because a follow-up such as "and for
        Friday?" still refers to them. The day and the time do not, so the
        next booking cannot reuse a slot the user did not ask for.
        The written row is kept so a later "ok" can be answered as thanks
        rather than as a second confirmation.
        """
        self.pending_booking = False
        self.awaiting_confirmation = False
        self.pending_ticket = False
        self.reschedule_slot_id = None
        if isinstance(booked, dict):
            self.last_booked = {
                "location": booked.get("location"),
                "date": booked.get("date"),
                "time": booked.get("time"),
            }
        self.slots["date"] = None
        self.slots["time"] = None

    def end(self, reflection: str | None = None) -> dict:
        """Close the session: write the reflection and drop the working summary.

        A reflection passed in is stored as given. None means the session
        writes its own from the turns, which is what the page does when the
        box is left empty.
        """
        from src.history import earlier_turns

        return self.memory.end_session(
            reflection,
            earlier_turns(self.history),
            turn=self.turn,
            trace=self.calls,
            run_id=self.run_id,
            config=self._config_name(),
        )


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
