"""Conversation growth must stay bounded, without corrupting the transcript.

_trim_screenshots capped images but nothing capped turns, so a long or continued
run grew its prompt forever — a real 1828-turn history billed 157k tokens per
step. Elision has to move in call/response pairs: splitting one would recreate
the dangling-call 400 that _repair_dangling_calls exists to fix.
"""

from __future__ import annotations

from google.genai import types

from agentos.brain import (HISTORY_TOKEN_LIMIT, HISTORY_TOKEN_TARGET,
                           GeminiBrain)

FILLER = "x" * 4000  # ~1k tokens per turn, so conversations get big quickly


def _pair(name: str = "click") -> list[types.Content]:
    return [
        types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name=name, args={"pad": FILLER}))]),
        types.Content(role="user", parts=[
            types.Part(function_response=types.FunctionResponse(
                name=name, response={"pad": FILLER})),
            types.Part(text="[budget: 100 of 600 actions remaining]")]),
    ]


def _conversation(pairs: int) -> list[types.Content]:
    anchor = types.Content(role="user", parts=[types.Part(text="GOAL: find the thing")])
    out = [anchor]
    for _ in range(pairs):
        out += _pair()
    return out


def _dangling(contents) -> int:
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


def test_short_conversation_is_untouched():
    contents = _conversation(5)
    assert GeminiBrain._estimate_tokens(contents) < HISTORY_TOKEN_LIMIT
    assert GeminiBrain._trim_history(contents) == 0
    assert len(contents) == 11


def test_long_conversation_is_cut_to_target():
    contents = _conversation(200)
    assert GeminiBrain._estimate_tokens(contents) > HISTORY_TOKEN_LIMIT
    assert GeminiBrain._trim_history(contents) > 0
    assert GeminiBrain._estimate_tokens(contents) <= HISTORY_TOKEN_TARGET


def test_trim_never_splits_a_call_from_its_response():
    contents = _conversation(200)
    GeminiBrain._trim_history(contents)
    assert _dangling(contents) == 0


def test_goal_anchor_survives():
    """Dropping the opening turn would lose the task itself."""
    contents = _conversation(200)
    GeminiBrain._trim_history(contents)
    assert "GOAL: find the thing" in (contents[0].parts[0].text or "")


def test_marker_warns_the_model_its_memory_is_gone():
    contents = _conversation(200)
    GeminiBrain._trim_history(contents)
    marker = contents[1].parts[0].text or ""
    assert "elided" in marker
    assert "do not" in marker.lower()  # must not silently invent continuity


def test_trim_is_idempotent():
    contents = _conversation(200)
    assert GeminiBrain._trim_history(contents) > 0
    assert GeminiBrain._trim_history(contents) == 0


def test_recent_turns_are_the_ones_kept():
    """The tail is what the model needs; the middle is what goes."""
    contents = _conversation(200)
    contents += _pair("scroll")  # most recent action
    GeminiBrain._trim_history(contents)
    last_calls = [p.function_call.name for c in contents for p in (c.parts or [])
                  if p.function_call]
    assert last_calls[-1] == "scroll"


def test_estimator_counts_thought_signatures():
    """They are ~half the prompt on a long run and count_tokens omits them."""
    plain = types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="click", args={}))])
    signed = types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="click", args={}),
                   thought_signature=b"s" * 4000)])
    assert (GeminiBrain._estimate_tokens([signed])
            > GeminiBrain._estimate_tokens([plain]) + 500)


def test_atomic_blocks_pair_calls_with_responses():
    contents = _conversation(2)
    blocks = GeminiBrain._atomic_blocks(contents)
    assert blocks[0] == (0, 1)      # anchor: lone user turn
    assert blocks[1] == (1, 3)      # call + response held together
    assert blocks[2] == (3, 5)
