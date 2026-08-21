"""The task's scope must survive a long run, and finish() must answer to it.

Two failures with one cause. `_trim_history` elides the middle of a long
conversation and protects only the ANCHOR turn — which, on a task continued 48
times, belongs to the first run, not to the follow-up goal the operator typed
afterwards. A 48-item queue handed over as a follow-up therefore scrolled out of
the model's view entirely. What was left looked like a tidy, finished tail, so
the model called finish() and meant it: pushing back on text turns cannot help,
because a deliberate finish IS a tool call.

So the goal is re-sent as system_instruction on every round-trip (untrimmable),
and the first finish() after real work is challenged against it rather than
taken at its word.
"""

from __future__ import annotations

import asyncio

from google.genai import types

from agentos.brain import GeminiBrain
from agentos.instructions import system_instruction
from agentos.logs import RunLog
from agentos.models import Task

from test_completion_signal import _brain, _call_turn, _events, _Sandbox

GOAL = "a@duck.com b@duck.com c@duck.com — register all three."


# ---- the goal is pinned outside the conversation ----------------------------

def test_system_instruction_carries_the_goal():
    text = system_instruction("be quick", GOAL)
    assert GOAL in text
    assert "be quick" in text


def test_goal_survives_with_no_standing_instructions():
    """The operator box is usually empty; the goal must ride anyway."""
    text = system_instruction("", GOAL)
    assert GOAL in text
    assert system_instruction("", "") is None


def test_every_request_resends_the_goal(tmp_path):
    """Not just the first call — the point is that trimming cannot reach it."""
    task = Task(goal=GOAL, max_steps=4)
    log = RunLog(task.id, root=tmp_path)
    seen: list[str | None] = []
    brain = _brain([_call_turn("run_command", command="true"),
                    _call_turn("finish", summary="done"),
                    _call_turn("finish", summary="done")])
    inner = brain._generate

    async def spy(contents, log=None, step=0, instructions=None, task=None,
                  attempt=0, goal=None):
        seen.append(goal)
        return await inner(contents, log, step, instructions, task, attempt, goal)

    brain._generate = spy
    asyncio.run(brain._loop(task, _Sandbox(), log, []))

    assert seen == [GOAL, GOAL, GOAL]


def test_a_continued_task_pins_its_new_goal(tmp_path):
    """Continuing replaces task.goal; the follow-up scope must be what is sent."""
    task = Task(goal="the original ask")
    task.goal = GOAL  # what POST /tasks/{id}/continue does
    assert GOAL in system_instruction(task.instructions, task.goal)


# ---- finish() has to answer for the whole scope -----------------------------

def test_first_finish_is_challenged_not_honoured(tmp_path):
    task = Task(goal=GOAL, max_steps=6)
    log = RunLog(task.id, root=tmp_path)
    brain = _brain([
        _call_turn("finish", summary="did the ones that were asked for"),
        _call_turn("run_command", command="register c@duck.com"),
        _call_turn("finish", summary="all three registered"),
        _call_turn("finish", summary="all three registered", checked="3 of 3"),
    ])

    result = asyncio.run(brain._loop(task, _Sandbox(), log, []))

    assert result == "all three registered"
    kinds = [e["kind"] for e in _events(log)]
    assert kinds.count("finish_challenged") == 2  # once per attempt to end
    assert kinds.count("done") == 1


def test_the_challenge_turn_points_at_the_pinned_scope(tmp_path):
    task = Task(goal=GOAL, max_steps=4)
    log = RunLog(task.id, root=tmp_path)
    contents: list[types.Content] = []
    brain = _brain([_call_turn("finish", summary="done"),
                    _call_turn("finish", summary="done")])

    asyncio.run(brain._loop(task, _Sandbox(), log, contents))

    challenge = "".join(p.text or "" for c in contents for p in (c.parts or [])
                        if p.text and "system instructions" in (p.text or ""))
    assert "item by item" in challenge
    assert "do not finish" in challenge.lower()


def test_a_repeated_finish_is_accepted_so_the_loop_terminates(tmp_path):
    """The challenge must not be a trap: insisting twice ends the task."""
    task = Task(goal=GOAL, max_steps=8)
    log = RunLog(task.id, root=tmp_path)
    brain = _brain([_call_turn("finish", summary="really done")] * 2)

    result = asyncio.run(brain._loop(task, _Sandbox(), log, []))

    assert result == "really done"
    assert task.steps_taken == 2


def test_working_again_re_arms_the_challenge(tmp_path):
    """Every attempt to end gets checked, not just the first of the run."""
    task = Task(goal=GOAL, max_steps=10)
    log = RunLog(task.id, root=tmp_path)
    brain = _brain([
        _call_turn("finish", summary="attempt one"),      # challenged
        _call_turn("run_command", command="more work"),   # re-arms
        _call_turn("finish", summary="attempt two"),      # challenged again
        _call_turn("finish", summary="attempt two"),      # honoured
    ])

    result = asyncio.run(brain._loop(task, _Sandbox(), log, []))

    assert result == "attempt two"
    assert [e["kind"] for e in _events(log)].count("finish_challenged") == 2


def test_challenged_turn_leaves_a_replayable_history(tmp_path):
    """The refused finish still needs a function_response, or the next request
    400s on a dangling call."""
    task = Task(goal=GOAL, max_steps=4)
    log = RunLog(task.id, root=tmp_path)
    contents: list[types.Content] = []
    brain = _brain([_call_turn("finish", summary="done"),
                    _call_turn("finish", summary="done")])

    asyncio.run(brain._loop(task, _Sandbox(), log, contents))

    assert GeminiBrain._repair_dangling_calls(contents) == 0
