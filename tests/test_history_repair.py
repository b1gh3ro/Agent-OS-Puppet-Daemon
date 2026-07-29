"""Every function_call must keep a matching function_response.

Regression test for the continued task that died instantly with
"400 INVALID_ARGUMENT: Each Function Response must be matched to a Function Call
by name". A batch is answered only at the END of a step, so an interruption
between the model turn and its response turn persists a conversation Gemini
rejects on every subsequent request — permanently, since the bad turn is saved
to history.json and reloaded on continue.
"""

from __future__ import annotations

from google.genai import types

from agentos.brain import GeminiBrain


def _call(name: str) -> types.Content:
    return types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name=name, args={}))])


def _resp(*names: str, text: str | None = None) -> types.Content:
    parts = [types.Part(function_response=types.FunctionResponse(name=n, response={"ok": True}))
             for n in names]
    if text is not None:
        parts.append(types.Part(text=text))
    return types.Content(role="user", parts=parts)


def _shape(contents):
    out = []
    for c in contents:
        kinds = []
        for p in (c.parts or []):
            if p.function_call:
                kinds.append("CALL:" + p.function_call.name)
            elif p.function_response:
                kinds.append("RESP:" + p.function_response.name)
            elif p.text is not None:
                kinds.append("text")
        out.append((c.role, kinds))
    return out


def _mismatches(contents) -> int:
    bad = 0
    for i, c in enumerate(contents):
        if c.role != "model":
            continue
        calls = [p.function_call.name for p in (c.parts or []) if p.function_call]
        if not calls:
            continue
        nxt = contents[i + 1] if i + 1 < len(contents) else None
        resp = [p.function_response.name for p in (nxt.parts or []) if p.function_response] if nxt else []
        if calls != resp:
            bad += 1
    return bad


def test_healthy_conversation_is_left_alone():
    contents = [_call("click"), _resp("click", text="[budget]"),
                _call("scroll"), _resp("scroll", text="[budget]")]
    before = _shape(contents)
    assert GeminiBrain._repair_dangling_calls(contents) == 0
    assert _shape(contents) == before


def test_call_at_end_of_conversation_gets_a_response():
    """Interrupted with the model turn appended and nothing after it."""
    contents = [_call("click"), _resp("click"), _call("scroll")]
    assert GeminiBrain._repair_dangling_calls(contents) == 1
    assert _mismatches(contents) == 0
    assert _shape(contents)[-1] == ("user", ["RESP:scroll"])


def test_continue_hint_turn_absorbs_the_missing_response():
    """The real failure: `continue` appended text+screenshot after a live call."""
    contents = [_call("scroll"),
                types.Content(role="user", parts=[types.Part(text="operator follow-up")])]
    assert GeminiBrain._repair_dangling_calls(contents) == 1
    assert _mismatches(contents) == 0
    # Response leads the turn, the hint text is preserved after it.
    assert _shape(contents)[1] == ("user", ["RESP:scroll", "text"])


def test_partially_answered_batch_is_completed():
    """Crash midway through a batch: some responses real, the rest invented."""
    batch = types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="click", args={})),
        types.Part(function_call=types.FunctionCall(name="type_text", args={})),
        types.Part(function_call=types.FunctionCall(name="scroll", args={})),
    ])
    contents = [batch, _resp("click")]
    assert GeminiBrain._repair_dangling_calls(contents) == 2
    assert _mismatches(contents) == 0
    assert _shape(contents)[1][1] == ["RESP:click", "RESP:type_text", "RESP:scroll"]


def test_repeated_tool_in_one_batch_lines_up():
    """A real response is consumed once, so duplicates still match positionally."""
    batch = types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="click", args={})),
        types.Part(function_call=types.FunctionCall(name="click", args={})),
    ])
    contents = [batch, _resp("click")]
    assert GeminiBrain._repair_dangling_calls(contents) == 1
    assert _shape(contents)[1][1] == ["RESP:click", "RESP:click"]


def test_invented_response_says_the_action_may_not_have_run():
    contents = [_call("scroll")]
    GeminiBrain._repair_dangling_calls(contents)
    payload = contents[1].parts[0].function_response.response
    assert "interrupted" in payload["error"].lower()


def test_repair_is_idempotent():
    contents = [_call("click"), _resp("click"), _call("scroll")]
    assert GeminiBrain._repair_dangling_calls(contents) == 1
    assert GeminiBrain._repair_dangling_calls(contents) == 0
    assert _mismatches(contents) == 0


def test_text_only_model_turns_are_ignored():
    contents = [types.Content(role="model", parts=[types.Part(text="final answer")])]
    assert GeminiBrain._repair_dangling_calls(contents) == 0
    assert len(contents) == 1
