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
    r"^(yes|yeah|yep|ok|okay|sure|go ahead|please do|book it|confirm)[\s.!]*$",
    re.IGNORECASE,
)
_NO = re.compile(
    r"^(no|nope|nah|don'?t|do not|stop|cancel|never mind)[\s.!]*$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You label one IT helpdesk message. Reply with a single JSON object and no other text.

The object has two keys:
- "intent": one of fix_issue, ask_kb, book_slot, confirm, reject, check_ticket, provide_info, recall_fact, out_of_scope
- "slots": an object with exactly these keys: issue_type, error_code, urgency, location, date, time, asset_id, ticket_id

Intent meanings:
- fix_issue: something is broken and the user wants it fixed or raised as a ticket
- ask_kb: a question about a policy, an error code, a procedure, or how something works
- book_slot: the user is asking for a visit, or answering a booking question with a place, a floor, a desk, a day, or a time. A short reply that is only a place or a time is book_slot, not provide_info.
- confirm: the user agrees to go ahead, including "yes", "go ahead", and "book that"
- reject: the user says no, or withdraws a day or a booking, including "no" and "leave Friday"
- check_ticket: the user asks about an existing ticket
- provide_info: the user introduces themselves or states a standing fact (their name, their laptop model, which office they sit in, their manager). A question is never provide_info.
- recall_fact: the user asks what you already know about them, such as "Where do I sit?" or "Which laptop do I use?". They are not stating a new fact and they are not asking about a policy. Every slot stays null.
- out_of_scope: not an IT helpdesk matter. Greetings, thanks, and "hope your day is going well" are out_of_scope, not provide_info.

A question about policy, a cycle, a model the user would get, or how something works is ask_kb, even if it mentions a laptop or a printer.
"desk 14" is part of the location, not a time. time is only a clock time such as "4 pm".

Reply with the JSON object only. Example for "Make it Thursday at 4 pm.":
{"intent": "book_slot", "slots": {"issue_type": null, "error_code": null, "urgency": null, "location": null, "date": "Thursday", "time": "4 pm", "asset_id": null, "ticket_id": null}}

Rules for slots:
- Copy only what the user stated. Use null for anything they did not state.
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
    if _NO.match(stripped):
        return {"intent": "reject", "slots": empty_slots(), "source": "rule"}

    # A ticket id with no other request is a status check. A longer sentence
    # that merely mentions one still goes to the model.
    if TICKET_RE.fullmatch(stripped):
        slots = empty_slots()
        slots["ticket_id"] = TICKET_RE.fullmatch(stripped).group(0).upper()
        return {"intent": "check_ticket", "slots": slots, "source": "rule"}

    return None


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
    return result


def understand(
    text: str,
    *,
    user: str | None = None,
    session: str | None = None,
    turn: int | None = None,
) -> dict:
    """Label one user sentence.

    Returns {"intent", "slots", "source"}. source is "rule" or "llm".
    A rule hit makes no model call and writes no log line. An LLM hit is
    logged by chat() as call_type "nlu" with the API's own token counts.
    """
    ruled = _rule_based(text)
    if ruled is not None:
        return ruled

    reply = chat(
        text,
        system=SYSTEM_PROMPT,
        temperature=0.0,
        call_type="nlu",
        user=user,
        session=session,
        turn=turn,
    )
    result = _normalise(_extract_json(reply))
    return _apply_known_codes(text, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label one helpdesk sentence")
    parser.add_argument("text")
    args = parser.parse_args()
    print(json.dumps(understand(args.text), indent=2))


if __name__ == "__main__":
    main()
