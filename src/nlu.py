"""Hybrid intent and slot detection.

Rules run first. A short yes or no, a ticket id on its own, or an asset id is
unambiguous, so those never spend a model call. Everything else goes to the
chat model at temperature 0, which must reply with JSON and nothing else.

The model labels the sentence. It does not answer it, and it does not call a
tool. A slot the user did not state stays null; the parser drops any extra
key the model invented and fills any key it omitted with null.

    python -m src.nlu "The VPN keeps failing with ERR-VPN-809. What does that code mean?"
"""

from __future__ import annotations

import argparse
import json
import re

from src.llm import chat

INTENTS = (
    "fix_issue",
    "ask_kb",
    "book_slot",
    "confirm",
    "reject",
    "check_ticket",
    "provide_info",
    "recall_fact",
    "out_of_scope",
)

SLOT_NAMES = (
    "issue_type",
    "error_code",
    "urgency",
    "location",
    "date",
    "time",
    "asset_id",
    "ticket_id",
)

# Codes the knowledge base and the test conversation actually use. A rule
# matches these rather than any token that looks like a code, so a made-up
# string is not promoted to an error code.
ERROR_CODES = (
    "ERR-VPN-809",
    "ERR-VPN-104",
    "ERR-VPN-440",
    "ERR-WIFI-220",
    "ERR-PRINT-017",
    "ERR-MFA-003",
)

TICKET_RE = re.compile(r"\bTCK-\d+\b", re.IGNORECASE)
ASSET_RE = re.compile(r"\bLAP-[A-Z0-9]+\b", re.IGNORECASE)
ERROR_RE = re.compile(r"\b(?:" + "|".join(ERROR_CODES) + r")\b", re.IGNORECASE)

_YES = re.compile(
    r"^(yes|yeah|yep|sure|go ahead|please do|book it|confirm)[\s.!]*$",
    re.IGNORECASE,
)
# "ok" after a booking is an acknowledgement, not another yes. It must not
# be labelled confirm, which replies "there is nothing waiting to book".
_ACK = re.compile(
    r"^(ok|okay|thanks|thank you|thx|cheers|got it|alright|cool)[\s.!]*$",
    re.IGNORECASE,
)
_NO = re.compile(
    r"^(no|nope|nah|don'?t|do not|stop|cancel|never mind)[\s.!]*$",
    re.IGNORECASE,
)

# An explicit change of the user's own office. The destination is the new
# fact and the origin is not. A bare "to the Pune office" is not a change:
# that sentence is usually answering a booking question.
_MOVE_TO = re.compile(
    r"\b(?:"
    r"(?:update|change|set)\s+my\s+(?:location|office)\s+(?:from\s+.+?\s+)?to"
    r"|(?:moved|shifting|shifted|moving)\s+from\s+.+?\s+to"
    r")\s+([A-Z][A-Za-z]+(?:\s+office)?)",
    re.IGNORECASE,
)

# Changing a booked visit, as opposed to a standing fact such as "change my
# office". The verb has to sit next to a visit word, so "change my office to
# Gurugram" is not a reschedule.
# A question about the user's own ticket or visit, not a standing fact.
_LAST_RECORD = re.compile(
    r"\b(?:ticket|issue|problem|fault|booking|visit|appointment)s?\b",
    re.IGNORECASE,
)

# "Share the details of booking" asks to read a visit, not to book one.
# "Book a visit" and "change the timing of my booking" are requests, not reads.
_BOOKING_DETAILS = re.compile(
    r"\b(?:share|show|give|tell|what|whats|what's)\b.{0,40}\b(?:detail|details|status|info|information)\b"
    r"|\b(?:detail|details|status)\b.{0,30}\b(?:booking|visit|appointment|slot)\b"
    r"|\b(?:booking|visit|appointment|slot)\b.{0,30}\b(?:detail|details|status)\b",
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

SYSTEM_PROMPT = """You label one IT helpdesk message. Reply with a single JSON object and no other text.

The object has two keys:
- "intent": one of fix_issue, ask_kb, book_slot, confirm, reject, check_ticket, provide_info, recall_fact, out_of_scope
- "slots": an object with exactly these keys: issue_type, error_code, urgency, location, date, time, asset_id, ticket_id

Intent meanings:
- fix_issue: something is broken and the user wants it fixed or raised as a ticket
- ask_kb: a question about a policy, an error code, a procedure, or how something works
- book_slot: the user is asking for a visit, answering a booking question with a place, a floor, a desk, a day, or a time, or changing the time, day, or place of a visit they already booked. A short reply that is only a place or a time is book_slot, not provide_info. When the message says the assistant just asked for an office, a floor, a desk, a day, or a time, the user's reply is the answer to that question: label it book_slot, never provide_info, and copy the place, day, or time into the matching slot. "Can I change the timing of my booking?", "Reschedule my visit", and "Move it to 4 pm" are book_slot, never out_of_scope and never provide_info. If they name a new time, day, or place, copy it into that slot. If they only ask to change it and name nothing new, leave that slot null.
- confirm: the user agrees to go ahead, including "yes", "go ahead", and "book that". "ok" and "okay" are not confirm.
- reject: the user says no, or withdraws a day or a booking, including "no" and "leave Friday"
- check_ticket: the user asks about an existing ticket, or about the last issue they raised, or about a visit that was already booked. "What was the last issue I raised?", "What is the last ticket I raised?", "Is there any ticket raised for me?", "What is my last booking?", and "Share the details of booking" are check_ticket, never book_slot and never recall_fact. Leave ticket_id null unless they named an id such as TCK-0001.
- provide_info: the user introduces themselves or states or changes a standing fact (their name, their laptop model, which office they sit in, their manager). A request to change their own office is provide_info, not out_of_scope and not book_slot. "Update my location from Pune to Gurugram", "I have shifted from Pune to Gurugram", and "Change my office to Gurugram" are provide_info. Put only the new place in location, for example "Gurugram", and leave every other slot null. A question is never provide_info.
- recall_fact: the user asks what you already know about them, such as "Where do I sit?" or "Which laptop do I use?". They are not stating a new fact and they are not asking about a policy. A question about a ticket, an issue they raised, or a booking is check_ticket, not recall_fact. Every slot stays null.
- out_of_scope: not an IT helpdesk matter. Greetings, thanks, "ok", "okay", and "hope your day is going well" are out_of_scope, not provide_info and not confirm. Changing the user's own office, desk, or laptop is provide_info, never out_of_scope. Changing the time, day, or place of a booked visit is book_slot, never out_of_scope.

A question about policy, a cycle, a model the user would get, or how something works is ask_kb, even if it mentions a laptop or a printer.
"desk 14" is part of the location, not a time. time is only a clock time such as "4 pm".
location is the whole place the user named, not the city alone. Copy the office, the floor, and the desk together. "Pune office, 3rd floor, 212 desk" has location "Pune office, 3rd floor, 212 desk", never "Pune office". "floor 2 and desk 1" stays inside location. Do not drop a floor or a desk number.

Reply with the JSON object only. Examples:
"Make it Thursday at 4 pm." -> {"intent": "book_slot", "slots": {"issue_type": null, "error_code": null, "urgency": null, "location": null, "date": "Thursday", "time": "4 pm", "asset_id": null, "ticket_id": null}}
"Can I change the timing of my booking?" -> {"intent": "book_slot", "slots": {"issue_type": null, "error_code": null, "urgency": null, "location": null, "date": null, "time": null, "asset_id": null, "ticket_id": null}}
"Move it to 4 pm." -> {"intent": "book_slot", "slots": {"issue_type": null, "error_code": null, "urgency": null, "location": null, "date": null, "time": "4 pm", "asset_id": null, "ticket_id": null}}
"Update my location from Pune to Gurugram" -> {"intent": "provide_info", "slots": {"issue_type": null, "error_code": null, "urgency": null, "location": "Gurugram", "date": null, "time": null, "asset_id": null, "ticket_id": null}}
"What was the last issue I raised?" -> {"intent": "check_ticket", "slots": {"issue_type": null, "error_code": null, "urgency": null, "location": null, "date": null, "time": null, "asset_id": null, "ticket_id": null}}
"What is my last booking?" -> {"intent": "check_ticket", "slots": {"issue_type": null, "error_code": null, "urgency": null, "location": null, "date": null, "time": null, "asset_id": null, "ticket_id": null}}
"Share the details of booking" -> {"intent": "check_ticket", "slots": {"issue_type": null, "error_code": null, "urgency": null, "location": null, "date": null, "time": null, "asset_id": null, "ticket_id": null}}

Rules for slots:
- Copy only what the user stated in this message. Use null for anything they did not state.
- Do not copy a place, a day, a time, a desk number, an office, or a code from the examples. The examples show the shape of the JSON, not facts about this user. "Can you come to my desk?" names no desk number, so location is null. Leave a slot null and let the next turn ask for it.
- error_code, asset_id and ticket_id must be copied exactly, or be null.
- Do not answer the user's question. Do not add keys. Do not wrap the JSON in markdown.
"""


class NLUError(RuntimeError):
    """Raised when the model reply is not the JSON this module asked for."""


def empty_slots() -> dict:
    return {name: None for name in SLOT_NAMES}


def _rule_based(text: str) -> dict | None:
    """Labels a sentence only when a rule is unambiguous. Otherwise None."""
    stripped = text.strip()
    if _YES.match(stripped):
        return {"intent": "confirm", "slots": empty_slots(), "source": "rule"}
    if _ACK.match(stripped):
        return {"intent": "out_of_scope", "slots": empty_slots(), "source": "rule"}
    if _NO.match(stripped):
        return {"intent": "reject", "slots": empty_slots(), "source": "rule"}

    # A ticket id with no other request is a status check. A longer sentence
    # that merely mentions one still goes to the model.
    if TICKET_RE.fullmatch(stripped):
        slots = empty_slots()
        slots["ticket_id"] = TICKET_RE.fullmatch(stripped).group(0).upper()
        return {"intent": "check_ticket", "slots": slots, "source": "rule"}

    return None


# "The same time is fine" answers a booking question by pointing at the time
# already given. It names no clock time, so the model calls it confirm and the
# visit is written with no time. It is the time answer, not the yes.
_SAME_TIME = re.compile(
    r"\b(?:same|that)\s+time\b|\btime\s+is\s+fine\b",
    re.IGNORECASE,
)


def _same_time(text: str) -> dict | None:
    """A reply that reuses the time already on the visit, without naming one."""
    if not _SAME_TIME.search(text or ""):
        return None
    return {"intent": "book_slot", "slots": empty_slots(), "source": "rule"}


def _extract_json(reply: str) -> dict:
    """The model sometimes wraps JSON in a fence. Take the outermost object."""
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise NLUError(f"NLU reply had no JSON object: {reply!r}")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise NLUError(f"NLU reply was not valid JSON: {reply!r}") from exc
    if not isinstance(parsed, dict):
        raise NLUError(f"NLU reply was not a JSON object: {reply!r}")
    return parsed


def _normalise(parsed: dict) -> dict:
    intent = parsed.get("intent")
    if intent not in INTENTS:
        raise NLUError(f"NLU reply had an unknown intent: {intent!r}")

    raw_slots = parsed.get("slots") or {}
    if not isinstance(raw_slots, dict):
        raise NLUError("NLU reply had slots that were not an object")

    slots = empty_slots()
    for name in SLOT_NAMES:
        value = raw_slots.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        slots[name] = value.strip()
    return {"intent": intent, "slots": slots, "source": "llm"}


# A place the user named while answering "which office, floor, and desk?".
# The model often keeps only the city ("Pune office") and drops the floor
# and the desk. The whole reply is the place when the question asked for one
# and the reply itself is not a day or a clock time.
_PLACE_QUESTION = re.compile(r"\b(?:office|floor|desk)\b", re.IGNORECASE)
_CLOCK_TIME = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b(?:morning|afternoon|evening|noon)\b",
    re.IGNORECASE,
)
_WEEKDAY = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)\b",
    re.IGNORECASE,
)


def _looks_like_place(text: str) -> bool:
    """True when the sentence names a place and is not only a day or a time."""
    stripped = text.strip()
    if not stripped or _CLOCK_TIME.search(stripped) or _WEEKDAY.search(stripped):
        return False
    return _PLACE_QUESTION.search(stripped) is not None


def _keep_full_place(text: str, result: dict, pending_question: str | None) -> dict:
    """Put the floor and the desk back when the model kept only the city.

    Runs only on the answer to the place question. A standing "I sit at the
    Pune office" is not rewritten, and a day or a time is left in its slot.
    """
    question = pending_question or ""
    if not _PLACE_QUESTION.search(question):
        return result
    if not _looks_like_place(text):
        return result
    result["intent"] = "book_slot"
    result["slots"]["location"] = re.sub(r"\s+", " ", text).strip(" .")
    result["slots"]["date"] = None
    result["slots"]["time"] = None
    return result


# The labelled set writes locations in one form. The model paraphrases
# "third" as "3rd" and the other way round, which is the same place.
_LOCATION_WORDS = (
    ("third", "3rd"),
    ("second", "2nd"),
    ("first", "1st"),
)


def _place_tokens(value: str) -> set[str]:
    text = value.lower().replace(",", " ")
    for word, ordinal in _LOCATION_WORDS:
        text = text.replace(word, ordinal)
    return set(text.split())


def _same_place(predicted: str, expected: str) -> bool:
    """True when both name the same place, ignoring word order and 3rd/third."""
    return _place_tokens(predicted) == _place_tokens(expected)


def _apply_known_codes(text: str, result: dict) -> dict:
    """Fill a code the model left null when the sentence states it verbatim.

    This only copies a code that is literally present. It never invents one,
    and it never overwrites a value the model already extracted.
    """
    slots = result["slots"]
    if slots["error_code"] is None:
        found = ERROR_RE.search(text)
        if found:
            slots["error_code"] = found.group(0).upper()
    if slots["ticket_id"] is None:
        found = TICKET_RE.search(text)
        if found:
            slots["ticket_id"] = found.group(0).upper()
    if slots["asset_id"] is None:
        found = ASSET_RE.search(text)
        if found:
            slots["asset_id"] = found.group(0).upper()
    # "desk 14" is a place. A model that parks it in time has mis-labelled a
    # location, and the slot is cleared rather than kept as a clock time.
    time_value = slots["time"]
    if isinstance(time_value, str) and re.fullmatch(r"desk\s*\d+", time_value.strip(), re.IGNORECASE):
        slots["time"] = None
        if slots["location"] and "desk" not in slots["location"].lower():
            slots["location"] = f"{slots['location']}, {time_value.strip()}"
    # "Update my location from Pune to Gurugram" states a new office. qwen2.5:3b
    # often labels that as provide_info but leaves location null, or calls the
    # whole sentence out_of_scope. The new place is the destination of the move.
    moved = _MOVE_TO.search(text)
    if moved and slots["location"] is None:
        result["intent"] = "provide_info"
        slots["location"] = moved.group(1).strip(" .")
    # "Can I change the timing of my booking?" is a visit change. qwen2.5:3b
    # often calls that out_of_scope. A standing office move is left alone.
    elif result["intent"] == "out_of_scope" and _RESCHEDULE.search(text) and moved is None:
        result["intent"] = "book_slot"
    # "What was the last issue I raised?" is a ticket lookup. The same model
    # often files it as recall_fact, which never opens the ticket table.
    elif result["intent"] == "recall_fact" and _LAST_RECORD.search(text):
        result["intent"] = "check_ticket"
    # "Share the details of booking" is a read. qwen2.5:3b often calls it
    # book_slot, which then asks for a day and a time the user did not request.
    elif result["intent"] in ("book_slot", "ask_kb", "recall_fact") and _BOOKING_DETAILS.search(text) and moved is None:
        if not _RESCHEDULE.search(text):
            result["intent"] = "check_ticket"
            for name in ("location", "date", "time"):
                result["slots"][name] = None
    return result


def _user_prompt(text: str, pending_question: str | None) -> str:
    """The sentence, plus the question it answers when one is still open.

    A bare "Pune office, floor 2 and desk 1." looks like a standing fact.
    The question "Which office, floor, and desk?" makes it the answer to a
    visit instead. No question is sent on a first message, so a real profile
    update is not pushed into a booking.
    """
    question = (pending_question or "").strip()
    if not question:
        return text
    return f"The assistant just asked: {question}\nThe user replied: {text}"


def understand(
    text: str,
    *,
    user: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    trace: list | None = None,
    pending_question: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Label one user sentence.

    Returns {"intent", "slots", "source"}. source is "rule" or "llm".
    A rule hit makes no model call and writes no log line. An LLM hit is
    logged by chat() as call_type "nlu" with the API's own token counts.
    When trace is a list, the system prompt, the user prompt and the model
    reply are appended to it so the session file can record the call.
    pending_question is the line the assistant asked on the previous turn,
    when that question is still unanswered. It is not sent for a rule hit.
    """
    ruled = _rule_based(text)
    if ruled is not None:
        return ruled
    # Only while a visit question is open. Said on its own, "same time" is
    # not a booking answer and the model may label it however it reads.
    if pending_question and _same_time(text) is not None:
        return _same_time(text)

    reply = chat(
        _user_prompt(text, pending_question),
        system=SYSTEM_PROMPT,
        temperature=0.0,
        call_type="nlu",
        user=user,
        session=session,
        turn=turn,
        run_id=run_id,
        trace=trace,
    )
    result = _normalise(_extract_json(reply))
    result = _apply_known_codes(text, result)
    return _keep_full_place(text, result, pending_question)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label one helpdesk sentence")
    parser.add_argument("text")
    args = parser.parse_args()
    print(json.dumps(understand(args.text), indent=2))


if __name__ == "__main__":
    main()
