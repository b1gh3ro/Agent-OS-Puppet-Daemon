"""The model's reasoning reaches the operator's feed, and only the feed.

Actions alone don't say WHY the agent did something. OpenRouter returns the
chain of thought in `message.reasoning`; this pins down that it is asked for,
parsed into a `thought` part, logged as a `thinking` event, and then stripped
before the turn is replayed — reasoning is display-only on this transport, so
it must not ride back out as assistant content or inflate the token estimate
that drives history trimming.
"""

from __future__ import annotations

import asyncio
import json
import types as pytypes

import pytest
from google.genai import types

from agentos import openrouter
from agentos.brain import GeminiBrain, OpenRouterBrain, _reasoning_request
from agentos.logs import RunLog
from agentos.models import Task

from test_completion_signal import PNG, _brain, _events, _Sandbox


def _thought(text: str) -> types.Part:
    return types.Part(text=text, thought=True)


# ---- the request ------------------------------------------------------------

def test_reasoning_effort_maps_to_the_api_field():
    assert _reasoning_request("low") == {"effort": "low"}
    assert _reasoning_request("high") == {"effort": "high"}
    assert _reasoning_request("2048") == {"max_tokens": 2048}
    assert _reasoning_request("off") is None
    assert _reasoning_request("") is None
    with pytest.raises(ValueError):
        _reasoning_request("lots")


def test_config_asks_for_reasoning(monkeypatch):
    brain = OpenRouterBrain.__new__(OpenRouterBrain)
    brain._tools = []
    monkeypatch.setattr("agentos.brain.REASONING_EFFORT", "high")
    assert brain._config().reasoning == {"effort": "high"}


def test_reasoning_is_off_by_default_but_thinking_still_surfaces():
    """The dial defaults off because it is not what makes thinking visible.

    The model reasons whether or not the field is sent; `reasoning` only raises
    the budget. Paying for it by default would buy a partial view (45% of turns
    vs 29%) at 2.3x the reasoning tokens."""
    from agentos.brain import REASONING_EFFORT
    assert _reasoning_request(REASONING_EFFORT) is None


# ---- per-action intent: the part that is always there -----------------------

def test_every_ui_action_must_state_its_intent():
    for tool in openrouter.UI_TOOLS:
        params = tool["parameters"]
        assert "intent" in params["properties"], tool["name"]
        assert "intent" in params["required"], tool["name"]


def test_intent_does_not_disturb_the_action_itself():
    """It is metadata for the human; _execute must ignore it."""
    async def inner():
        clicks = []

        class SB(_Sandbox):
            async def click(self, x, y, repeat=1):
                clicks.append((x, y, repeat))

        brain = GeminiBrain.__new__(GeminiBrain)
        await brain._execute("click_at", {"x": 500, "y": 500, "intent": "accept terms"},
                             SB(), Task(goal="x"))
        assert clicks == [(640, 400, 1)]

    asyncio.run(inner())


# ---- the response -----------------------------------------------------------

def test_reasoning_becomes_a_thought_part():
    content = openrouter._to_content({
        "content": "Opening the signup form.",
        "reasoning": "Account 9 of 48 is next; the form needs the grad year first.",
        "tool_calls": [{"function": {"name": "click", "arguments": '{"x": 10, "y": 20}'}}],
    })
    thoughts = [p for p in content.parts if p.thought]
    assert len(thoughts) == 1
    assert "Account 9 of 48" in thoughts[0].text
    assert thoughts[0].thought_signature is None  # display-only, not replayable
    assert [p.text for p in content.parts if p.text and not p.thought] == \
        ["Opening the signup form."]


def test_structured_reasoning_details_are_read_too():
    content = openrouter._to_content({
        "content": "",
        "reasoning_details": [{"type": "reasoning.text", "text": "step one"},
                              {"type": "reasoning.text", "text": " step two"}],
        "tool_calls": [{"function": {"name": "wait", "arguments": "{}"}}],
    })
    assert [p.text for p in content.parts if p.thought] == ["step one step two"]


def test_encrypted_reasoning_is_not_shown_as_text():
    """gemini-3.7-flash at effort=low returns ONLY this: a replay blob, not prose.

    Rendering its base64 in the feed would be worse than showing nothing."""
    content = openrouter._to_content({
        "content": "",
        "reasoning_details": [{"type": "reasoning.encrypted", "data": "AY89a1/22JEX",
                               "format": "google-gemini-v1"}],
        "tool_calls": [{"function": {"name": "wait", "arguments": "{}"}}],
    })
    assert not [p for p in content.parts if p.thought]


def test_reasoning_is_never_sent_back_to_the_model():
    contents = [types.Content(role="model", parts=[
        _thought("private deliberation"),
        types.Part(text="out loud"),
    ])]
    messages = openrouter._to_messages(contents, None)
    body = json.dumps(messages)
    assert "private deliberation" not in body
    assert "out loud" in body


# ---- the loop ---------------------------------------------------------------

def test_thinking_and_narration_are_logged(tmp_path):
    task = Task(goal="x", max_steps=5)
    log = RunLog(task.id, root=tmp_path)
    turn = types.Content(role="model", parts=[
        _thought("42 accounts left; do the next one."),
        types.Part(text="Starting account 9."),
        types.Part(function_call=types.FunctionCall(name="run_command",
                                                    args={"command": "true"})),
    ])
    fin = types.Content(role="model", parts=[types.Part(
        function_call=types.FunctionCall(name="finish", args={"summary": "ok"}))])
    brain = _brain([turn, fin, fin])

    asyncio.run(brain._loop(task, _Sandbox(), log, []))

    by_kind = {e["kind"]: e for e in _events(log)}
    assert by_kind["thinking"]["text"] == "42 accounts left; do the next one."
    assert by_kind["narration"]["text"] == "Starting account 9."


def test_unsigned_thought_parts_leave_the_conversation(tmp_path):
    """Display-only reasoning must not accumulate in history..."""
    task = Task(goal="x", max_steps=5)
    log = RunLog(task.id, root=tmp_path)
    contents: list[types.Content] = []
    fin = types.Content(role="model", parts=[
        _thought("a" * 5000),
        types.Part(function_call=types.FunctionCall(name="finish",
                                                    args={"summary": "ok"}))])
    brain = _brain([fin, fin])

    asyncio.run(brain._loop(task, _Sandbox(), log, contents))

    assert not any(p.thought for c in contents for p in (c.parts or []))
    # The 5000-char thought alone would be ~1400 tokens; what is left is the
    # finish exchange and the challenge turn.
    assert GeminiBrain._estimate_tokens(contents) < 600


def test_signed_thought_parts_are_preserved():
    """...but a Gemini-native thought carries a signature its call needs on replay."""
    content = types.Content(role="model", parts=[
        types.Part(text="native reasoning", thought=True, thought_signature=b"sig"),
        types.Part(function_call=types.FunctionCall(name="click", args={})),
    ])
    GeminiBrain._strip_reasoning(content)
    assert len(content.parts) == 2


def test_reasoning_is_not_mistaken_for_a_final_answer(tmp_path):
    """A turn that is pure reasoning has said nothing — it must not end the task."""
    task = Task(goal="x", max_steps=5)
    log = RunLog(task.id, root=tmp_path)
    fin = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
        name="finish", args={"summary": "real answer"}))])
    brain = _brain([
        types.Content(role="model", parts=[_thought("hmm, let me think")]),
        fin, fin,
    ])

    result = asyncio.run(brain._loop(task, _Sandbox(), log, []))

    assert result == "real answer"
    kinds = [e["kind"] for e in _events(log)]
    assert "thinking" in kinds and "text_only" in kinds
