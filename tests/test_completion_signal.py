"""Only finish() ends a task; a text-only turn is a progress note.

Regression test for the failure that ended long jobs early: the loop treated
ANY model turn without function calls as the final answer. On a many-item job
the model periodically narrates its progress ("8 of 48 done, 42 remaining"),
and the first such turn closed the task as DONE with most of the work
outstanding — 35 of 44 historical `done` runs stopped that way, against only 9
that exhausted their step budget. No Docker, no Gemini: the client is a stub
that replays a scripted sequence of turns.
"""

from __future__ import annotations

import asyncio
import json
import types as pytypes

from agentos.brain import GeminiBrain, TEXT_ONLY_LIMIT
from agentos.logs import RunLog
from agentos.models import Task

from google.genai import types

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class _Sandbox:
    width, height = 1280, 800

    async def screenshot(self) -> bytes:
        return PNG

    async def exec_shell(self, command: str):
        return 0, f"ran: {command}"


def _text_turn(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


def _finish(**args) -> list[types.Content]:
    """The two turns it now takes to end a task: the first finish is challenged
    against the goal, the second is honoured."""
    return [_call_turn("finish", **args), _call_turn("finish", **args)]


def _call_turn(name: str, **args) -> types.Content:
    return types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name=name, args=args))])


def _brain(script: list[types.Content]) -> GeminiBrain:
    """A brain whose model replays `script`, one turn per _generate call."""
    brain = GeminiBrain.__new__(GeminiBrain)
    turns = iter(script)

    async def generate(contents, log=None, step=0, instructions=None, task=None,
                       attempt=0, goal=None):
        brain.seen.append(len(contents))
        candidate = pytypes.SimpleNamespace(content=next(turns))
        return pytypes.SimpleNamespace(candidates=[candidate], usage_metadata=None)

    brain.seen = []
    brain._generate = generate
    brain._settled_screenshot = lambda sandbox, delay=1.0: _ready(PNG)
    return brain


def _ready(value):
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    fut.set_result(value)
    return fut


def _events(log: RunLog) -> list[dict]:
    return [json.loads(line) for line in
            (log.dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]


def test_progress_note_does_not_end_the_task(tmp_path):
    """A status summary is pushed back on, and the task keeps working."""
    task = Task(goal="onboard all 48 accounts", max_steps=10)
    log = RunLog(task.id, root=tmp_path)
    brain = _brain([
        _text_turn("### Queue Status\n8 done, 42 remaining."),
        _call_turn("run_command", command="echo next"),
        *_finish(summary="all 48 onboarded"),
    ])

    result = asyncio.run(brain._loop(task, _Sandbox(), log, []))

    assert result == "all 48 onboarded"
    kinds = [e["kind"] for e in _events(log)]
    assert "text_only" in kinds          # the note was recorded...
    assert kinds.index("text_only") < kinds.index("done")  # ...but did not end it
    assert task.steps_taken == 4


def test_finish_ends_the_task_and_reports_leftovers(tmp_path):
    task = Task(goal="onboard all 48 accounts", max_steps=10)
    log = RunLog(task.id, root=tmp_path)
    brain = _brain(_finish(summary="40 onboarded",
                           remaining_work="8 blocked on captcha"))

    result = asyncio.run(brain._loop(task, _Sandbox(), log, []))

    assert "40 onboarded" in result
    assert "8 blocked on captcha" in result
    done = [e for e in _events(log) if e["kind"] == "done"][-1]
    assert done["remaining_work"] == "8 blocked on captcha"


def test_repeated_text_only_turns_are_eventually_accepted(tmp_path):
    """A model that will not call finish() must not spin the whole budget."""
    task = Task(goal="x", max_steps=20)
    log = RunLog(task.id, root=tmp_path)
    brain = _brain([_text_turn(f"note {i}") for i in range(TEXT_ONLY_LIMIT)])

    result = asyncio.run(brain._loop(task, _Sandbox(), log, []))

    assert result == f"note {TEXT_ONLY_LIMIT - 1}"
    assert task.steps_taken == TEXT_ONLY_LIMIT
    done = [e for e in _events(log) if e["kind"] == "done"][-1]
    assert done["reason"] == "text_only_limit"


def test_streak_resets_after_real_work(tmp_path):
    """Notes separated by actions never hit the limit — the run keeps going."""
    task = Task(goal="x", max_steps=20)
    log = RunLog(task.id, root=tmp_path)
    script: list[types.Content] = []
    for i in range(TEXT_ONLY_LIMIT + 2):
        script += [_text_turn(f"note {i}"), _call_turn("run_command", command="true")]
    script += _finish(summary="done for real")
    brain = _brain(script)

    result = asyncio.run(brain._loop(task, _Sandbox(), log, []))

    assert result == "done for real"
    assert len([e for e in _events(log) if e["kind"] == "text_only"]) == TEXT_ONLY_LIMIT + 2
