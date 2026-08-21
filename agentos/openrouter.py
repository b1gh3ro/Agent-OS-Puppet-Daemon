"""OpenRouter transport for the brain, shaped like the google-genai client.

Gemini's computer-use tool is a *server-side* built-in: the model is handed
``Tool(computer_use=...)`` and answers with UI actions it was trained to emit.
OpenRouter proxies models over the OpenAI chat-completions API and exposes no
such built-in — it does not even list a computer-use model. So the action
vocabulary `GeminiBrain._execute` dispatches on is declared here as ordinary
function tools, keeping the same names and the same 0-1000 coordinate grid
`scaling.denormalize` expects, so nothing downstream of the model call changes.

Everything else in brain.py — history repair, elision, screenshot trimming,
retries, pacing — is written against google-genai's Content/Part types. Rather
than fork that logic, this module speaks those types on both ends and
translates only in the middle: Content/Part in, OpenAI messages out, OpenAI
response back into Content/Part.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field

import aiohttp
from google.genai import types

API_URL = "https://openrouter.ai/api/v1/chat/completions"

#: Vision + function calling + a large context are all required; every id here
#: has them. 3.7-flash leads on price (~$0.38/$1.88 per Mtok vs 3.5-flash's
#: $1.50/$9.00). Override with AGENT_MODEL to use any other OpenRouter id.
MODEL_CANDIDATES = [
    "google/gemini-3.7-flash",
    "google/gemini-3.5-flash",
]

#: A tool result is JSON *text* on this API — an image cannot ride inside one
#: the way it does in a Gemini FunctionResponse, so screenshots follow as a
#: separate user turn (see `_to_messages`). This caps the text half so a
#: runaway `run_command` cannot blow out the prompt on its own.
TOOL_RESULT_LIMIT = 8000

_GRID = (
    "Coordinates are on a normalized 0-1000 grid over the screen, NOT pixels: "
    "x=0 is the left edge, x=1000 the right, y=0 the top, y=1000 the bottom."
)


def _point(extra: dict | None = None, *, required: bool = True) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": f"Horizontal position. {_GRID}"},
            "y": {"type": "integer", "description": f"Vertical position. {_GRID}"},
            **(extra or {}),
        },
    }
    if required:
        schema["required"] = ["x", "y"]
    return schema


#: The UI half of the toolset — what `Tool(computer_use=...)` would have supplied
#: natively. Names must match the `match` arms in `GeminiBrain._execute`.
UI_TOOLS = [
    {"name": "click_at", "description": "Left-click a point on the screen.",
     "parameters": _point()},
    {"name": "double_click", "description": "Double-click a point (open a file, select a word).",
     "parameters": _point()},
    {"name": "right_click", "description": "Right-click a point to open a context menu.",
     "parameters": _point()},
    {"name": "middle_click", "description": "Middle-click a point (paste selection, open in new tab).",
     "parameters": _point()},
    {"name": "hover_at", "description": "Move the pointer to a point without clicking, e.g. to reveal a tooltip or hover menu.",
     "parameters": _point()},
    {"name": "type_text_at", "description": (
        "Click a field and type into it. This is the reliable way to fill an input: "
        "it focuses the field first. By default it clears any existing content."),
     "parameters": _point({
         "text": {"type": "string", "description": "The text to type."},
         "press_enter": {"type": "boolean", "description": "Press Enter after typing. Default false."},
         "clear_before_typing": {"type": "boolean", "description": "Select-all and overwrite first. Default true."},
     } )},
    {"name": "type_text", "description": (
        "Type text wherever focus already is, without clicking first. Prefer "
        "type_text_at unless the field is already focused."),
     "parameters": {"type": "object", "properties": {
         "text": {"type": "string", "description": "The text to type."},
         "press_enter": {"type": "boolean", "description": "Press Enter after typing. Default false."},
     }, "required": ["text"]}},
    {"name": "key_combination", "description": (
        "Press a key or chord, e.g. 'Return', 'ctrl+s', 'alt+Tab', 'ctrl+shift+t'. "
        "Use X keysym names for non-printing keys (Return, Escape, Tab, BackSpace, "
        "Up, Down, Left, Right, Page_Up, Page_Down, Home, End)."),
     "parameters": {"type": "object", "properties": {
         "keys": {"type": "string", "description": "The key or '+'-joined chord to press."},
     }, "required": ["keys"]}},
    {"name": "scroll_document", "description": (
        "Scroll the window under the pointer. Omit x/y to scroll at the centre of "
        "the screen; pass them to scroll a specific pane."),
     "parameters": _point({
         "direction": {"type": "string", "enum": ["up", "down", "left", "right"],
                       "description": "Which way to scroll. Default down."},
         "magnitude": {"type": "integer", "description":
                       "Roughly how far, in normalized units. ~300 is a few wheel notches. Default 300."},
     }, required=False)},
    {"name": "drag_and_drop", "description": "Press at one point, drag to another, release.",
     "parameters": _point({
         "destination_x": {"type": "integer", "description": f"Where to drop, horizontally. {_GRID}"},
         "destination_y": {"type": "integer", "description": f"Where to drop, vertically. {_GRID}"},
     })},
    {"name": "navigate", "description": (
        "In the focused browser window, go to a URL (focuses the address bar and "
        "loads it). Only works if a browser is already open and focused."),
     "parameters": {"type": "object", "properties": {
         "url": {"type": "string", "description": "The full URL to load."},
     }, "required": ["url"]}},
    {"name": "go_back", "description": "Browser back.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "go_forward", "description": "Browser forward.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "open_web_browser", "description": "Launch the sandbox's Firefox and wait for it to appear.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "take_screenshot", "description": (
        "Take a fresh screenshot without acting. You already get one after every "
        "action, so use this only to re-look at an unchanged screen."),
     "parameters": {"type": "object", "properties": {}}},
    {"name": "wait", "description": (
        "Pause a few seconds for the UI to settle, then screenshot. For anything "
        "longer than ~15s use sleep or wait_for_screen_change instead."),
     "parameters": {"type": "object", "properties": {
         "seconds": {"type": "number", "description": "How long to pause, capped at 15. Default 5."},
     }}},
]

#: Every UI action carries one line of rationale. Reasoning tokens are not a
#: dependable substitute: gemini-3.7-flash returns most of its thinking as an
#: opaque `reasoning.encrypted` blob and only sometimes as readable text
#: (measured 2026-08-21: 0 of 4 turns at effort=medium, 1 of 4 at high), so a
#: feed that relied on them would show bare coordinates for most of a run. This
#: is a required argument instead — always present, ~10 tokens, and it makes the
#: activity feed narrate itself: "click_at — accept the AWS Builder ID terms".
_INTENT = {"type": "string", "description":
           "One short line, for the human watching: what you are trying to "
           "accomplish with this action and why now."}


def _with_intent(tool: dict) -> dict:
    params = tool["parameters"]
    properties = {**params.get("properties", {}), "intent": _INTENT}
    required = [*params.get("required", []), "intent"]
    return {**tool, "parameters": {**params, "properties": properties,
                                   "required": required}}


UI_TOOLS = [_with_intent(t) for t in UI_TOOLS]


@dataclass
class Config:
    """What `GeminiBrain._config` hands the client. Deliberately not
    `types.GenerateContentConfig`: that carries Gemini-only fields (the
    computer_use built-in, safety thresholds) whose Schema objects would have to
    be reverse-engineered back into JSON schema here. The brain builds this
    instead, so the tool declarations arrive as the plain dicts they started as."""
    system_instruction: str | None = None
    tools: list[dict] = field(default_factory=list)
    timeout_ms: int = 180_000
    #: OpenRouter's `reasoning` request field, or None to leave it off. When set,
    #: the model returns its chain of thought in `message.reasoning`, which the
    #: brain logs so the operator can see WHY it did what it did. Reasoning
    #: tokens bill as output, hence the effort knob rather than a hard-coded on.
    reasoning: dict | None = None


class OpenRouterError(RuntimeError):
    """An API-level failure. The message embeds the HTTP status because
    `GeminiBrain._is_rate_limit`/`_is_transient` classify by string match."""


class _Usage:
    """Duck-types google-genai's usage_metadata for `GeminiBrain._usage_fields`."""

    def __init__(self, usage: dict) -> None:
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        self.prompt_token_count = usage.get("prompt_tokens") or 0
        self.cached_content_token_count = prompt_details.get("cached_tokens") or 0
        self.candidates_token_count = usage.get("completion_tokens") or 0
        self.thoughts_token_count = completion_details.get("reasoning_tokens") or 0
        self.tool_use_prompt_token_count = 0
        self.total_token_count = usage.get("total_tokens") or 0


class _Candidate:
    def __init__(self, content: types.Content | None, finish_reason: str | None) -> None:
        self.content = content
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, candidates: list[_Candidate], usage: dict, feedback: str | None) -> None:
        self.candidates = candidates
        self.usage_metadata = _Usage(usage)
        self.prompt_feedback = feedback


def _image_url(blob) -> dict:
    mime = getattr(blob, "mime_type", None) or "image/png"
    data = base64.b64encode(blob.data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _to_messages(contents: list[types.Content], system: str | None) -> list[dict]:
    """Gemini Content/Part turns -> OpenAI chat messages.

    Two shape mismatches drive the work here:

    * Calls are matched to results by **id**, not by position and name. The ids
      are synthesized from the turn/slot index, which is sound precisely because
      `_repair_dangling_calls` has already guaranteed the responses in a turn
      mirror the calls in the turn before it, one for one and in order.
    * A `tool` message is text-only, so the screenshot that Gemini attaches
      *inside* the function response is re-emitted as a user turn right after
      the tool results it belongs to.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    pending_ids: list[str] = []
    for index, content in enumerate(contents):
        parts = content.parts or []

        if content.role == "model":
            # Thought parts are for the operator's feed, not for the wire: this
            # API wants reasoning echoed back as `reasoning_details`, not as
            # assistant content, and pasting it into `content` would both inflate
            # the prompt and read to the model as something it said out loud.
            parts = [p for p in parts if not p.thought]
            text = "".join(p.text or "" for p in parts if p.text)
            calls = [p.function_call for p in parts if p.function_call]
            message: dict = {"role": "assistant", "content": text or None}
            if calls:
                pending_ids = [f"call_{index}_{slot}" for slot in range(len(calls))]
                message["tool_calls"] = [
                    {"id": call_id, "type": "function", "function": {
                        "name": fc.name or "",
                        "arguments": json.dumps(dict(fc.args or {}), default=str)}}
                    for call_id, fc in zip(pending_ids, calls)
                ]
            else:
                pending_ids = []
            messages.append(message)
            continue

        # A user turn: tool results first (they must directly follow the
        # assistant turn that called them), then whatever else it carries.
        chunks: list[dict] = []
        orphaned: list[str] = []
        available = list(pending_ids)
        for part in parts:
            fr = part.function_response
            if fr is None:
                continue
            payload = json.dumps(fr.response or {}, default=str)[:TOOL_RESULT_LIMIT]
            if available:
                messages.append({"role": "tool", "tool_call_id": available.pop(0),
                                 "content": payload})
            else:
                # No call to attach to (a trimmed history, a hand-built turn).
                # Fold it into the user text rather than emit a tool message the
                # API would reject.
                orphaned.append(f"{fr.name}: {payload}")
            for response_part in (fr.parts or []):
                blob = getattr(response_part, "inline_data", None)
                if blob is not None and blob.data:
                    chunks.append({"type": "text", "text":
                                   "(screenshot of the screen after the actions above)"})
                    chunks.append(_image_url(blob))
        pending_ids = []

        if orphaned:
            chunks.append({"type": "text", "text": "Earlier results:\n" + "\n".join(orphaned)})
        for part in parts:
            if part.function_response is not None:
                continue
            if part.text:
                chunks.append({"type": "text", "text": part.text})
            blob = part.inline_data
            if blob is not None and blob.data:
                chunks.append(_image_url(blob))
        if chunks:
            messages.append({"role": "user", "content": chunks})

    return messages


def _to_content(message: dict) -> types.Content | None:
    """OpenAI assistant message -> a Gemini model turn the brain can append.

    Returns None for a turn with neither text nor calls, which routes into the
    brain's empty-response recovery instead of being mistaken for a final answer.
    """
    parts: list[types.Part] = []
    # Reasoning first, flagged `thought=True` so everything downstream can tell
    # the model's thinking apart from what it is actually saying. It carries no
    # thought_signature (that is a Gemini-native concept), which is also how
    # `GeminiBrain._strip_reasoning` knows this part is display-only and must
    # not be replayed to the model.
    reasoning = message.get("reasoning")
    if not reasoning:
        # Some providers only populate the structured form. Skip
        # `reasoning.encrypted` entries: they carry a `data` blob meant for
        # replay to the provider, not text a human can read.
        reasoning = "".join(
            d.get("text") or d.get("summary") or ""
            for d in (message.get("reasoning_details") or [])
            if isinstance(d, dict) and d.get("type") != "reasoning.encrypted")
    if reasoning and reasoning.strip():
        parts.append(types.Part(text=reasoning.strip(), thought=True))
    text = message.get("content")
    if isinstance(text, list):  # some providers return content as chunks
        text = "".join(c.get("text", "") for c in text if isinstance(c, dict))
    if text:
        parts.append(types.Part(text=text))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError:
            args = {"_unparsed_arguments": raw}
        parts.append(types.Part(function_call=types.FunctionCall(
            name=function.get("name") or "", args=args)))
    return types.Content(role="model", parts=parts) if parts else None


class _Models:
    def __init__(self, client: OpenRouterClient) -> None:
        self._client = client

    async def generate_content(self, *, model: str, contents: list[types.Content],
                               config: Config):
        return await self._client.generate_content(model, contents, config)


class _Aio:
    def __init__(self, client: OpenRouterClient) -> None:
        self.models = _Models(client)


class OpenRouterClient:
    """The slice of `genai.Client` the brain actually uses:
    `client.aio.models.generate_content(model=…, contents=…, config=…)`."""

    def __init__(self, api_key: str | None = None, *, url: str = API_URL) -> None:
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.url = url
        self.aio = _Aio(self)
        self._session: aiohttp.ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _session_for_loop(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        if self._session is None or self._session.closed or self._loop is not loop:
            self._session = aiohttp.ClientSession()
            self._loop = loop
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def generate_content(self, model: str, contents: list[types.Content],
                               config: Config) -> _Response:
        body = {
            "model": model,
            "messages": _to_messages(contents, config.system_instruction),
            "tools": [{"type": "function", "function": tool} for tool in config.tools],
            "tool_choice": "auto",
            "usage": {"include": True},   # ask for the token accounting we log
        }
        if config.reasoning:
            body["reasoning"] = config.reasoning
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "agentos",
        }
        session = await self._session_for_loop()
        timeout = aiohttp.ClientTimeout(total=config.timeout_ms / 1000)
        async with session.post(self.url, json=body, headers=headers,
                                timeout=timeout) as response:
            text = await response.text()
            if response.status >= 400:
                raise OpenRouterError(_error_message(response, text))
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                raise OpenRouterError(f"OpenRouter 200: unparsable body: {text[:400]}") from e

        # A failure can also arrive inside a 200 body, with the real status in it.
        error = data.get("error")
        if error:
            code = error.get("code") if isinstance(error, dict) else ""
            detail = error.get("message") if isinstance(error, dict) else str(error)
            raise OpenRouterError(f"OpenRouter {code}: {detail}")

        choices = data.get("choices") or []
        if not choices:
            return _Response([], data.get("usage") or {}, "no choices returned")
        choice = choices[0]
        content = _to_content(choice.get("message") or {})
        feedback = None if content else f"empty message (finish_reason={choice.get('finish_reason')})"
        candidates = [_Candidate(content, choice.get("finish_reason"))] if content else []
        return _Response(candidates, data.get("usage") or {}, feedback)


def _error_message(response: aiohttp.ClientResponse, body: str) -> str:
    """Format an HTTP failure so the brain's retry predicates can read it.

    The status goes in verbatim (they match on '429', '503', …), and a
    Retry-After header is rewritten into the `retryDelay: Ns` spelling
    `GeminiBrain._retry_after` already parses, so OpenRouter's own backoff
    advice is honored instead of the blind exponential fallback.
    """
    detail = body[:400]
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            detail = f"retryDelay: {float(retry_after)}s {detail}"
        except ValueError:
            pass
    return f"OpenRouter {response.status}: {detail}"
