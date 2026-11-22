"""One wrapper for chat calls.

Tokens are taken from Ollama's usage-style counts (prompt_eval_count and
eval_count), which are the API's own token counts, not a local estimate.
A call that fails raises and writes nothing to the log.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from src.config import CHAT_MODEL, OLLAMA_HOST
from src.logger import log_call

_CHAT_URL = f"{OLLAMA_HOST.rstrip('/')}/api/chat"


class LLMError(RuntimeError):
    """Raised when the chat backend rejects or cannot complete a call."""


class ChatResult(str):
    """The assistant text, plus the prompts that produced it.

    It is a str, so `reply.strip()` and f-strings keep working. system is
    the system prompt (None when the call sent none), user is the user
    prompt, and call_type says which step this was (nlu, reply, summarise).
    """

    def __new__(
        cls,
        text: str,
        *,
        system: str | None,
        user: str,
        call_type: str,
        model: str,
    ):
        obj = str.__new__(cls, text)
        obj.system = system
        obj.user = user
        obj.call_type = call_type
        obj.model = model
        return obj

    def trace(self) -> dict:
        """One model call, in the order the messages were sent."""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.user})
        messages.append({"role": "assistant", "content": str(self)})
        return {
            "call_type": self.call_type,
            "model": self.model,
            "messages": messages,
        }


def chat(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    model: str | None = None,
    call_type: str = "chat",
    config: str | None = None,
    user: str | None = None,
    session: str | None = None,
    turn: int | None = None,
    retrieved_tokens: int = 0,
    timeout: float = 120.0,
    trace: list | None = None,
) -> str:
    """Send one chat turn and return the assistant text.

    The return value is a str, so callers can treat it as text. It also
    carries the system prompt and the user prompt that were sent, on
    .system and .user, for the conversation log.

    Logs exactly one line after a successful response. On any HTTP, network
    or payload error, raises LLMError and leaves the log unchanged.
    """
    model_name = model or CHAT_MODEL
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options: dict[str, Any] = {"temperature": temperature}
    if top_p is not None:
        options["top_p"] = top_p

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    start = time.time()
    try:
        response = requests.post(_CHAT_URL, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError(f"Chat call to {model_name} failed before a response: {exc}") from exc

    if response.status_code != 200:
        detail = response.text.strip().splitlines()[0] if response.text.strip() else response.reason
        raise LLMError(
            f"Chat call to {model_name} failed: HTTP {response.status_code} {detail}"
        )

    try:
        body = response.json()
        reply = body["message"]["content"]
        # Ollama reports token counts in the response, equivalent to an
        # OpenAI-style usage field. Do not recount them with a tokenizer.
        input_tokens = int(body["prompt_eval_count"])
        output_tokens = int(body["eval_count"])
    except (KeyError, TypeError, ValueError, requests.JSONDecodeError) as exc:
        raise LLMError(
            f"Chat call to {model_name} returned a response with no usage counts"
        ) from exc

    if not isinstance(reply, str) or not reply.strip():
        raise LLMError(f"Chat call to {model_name} returned an empty reply")

    log_call(
        call_type=call_type,
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        start=start,
        retrieved_tokens=retrieved_tokens,
        config=config,
        user=user,
        session=session,
        turn=turn,
        extra={
            "temperature": temperature,
            "top_p": top_p,
            "usage": {
                "prompt_eval_count": input_tokens,
                "eval_count": output_tokens,
            },
        },
    )
    # The transcript wants the prompts that were actually sent, not a second
    # copy assembled by the caller. trace is that record. The reply text is
    # still `text`, so every existing caller keeps working.
    result = ChatResult(reply, system=system, user=prompt, call_type=call_type, model=model_name)
    if trace is not None:
        trace.append(result.trace())
    return result
