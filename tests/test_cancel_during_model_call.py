"""A cancel must interrupt a hung model call, not wait for it to return.

Regression test for the wedge where a task sat at RUNNING with
cancel_requested=true and dead Pause/Cancel buttons: cancel is only honored by
pause_gate at the top of each iteration, so a `generate_content` that never
returned parked the loop forever. No Docker, no Gemini — the client is a stub
whose call hangs.
"""

from __future__ import annotations

import asyncio
import types as pytypes

import pytest

from agentos.brain import GeminiBrain, MODEL_CALL_TIMEOUT_MS, _await_cancelable
from agentos.logs import RunLog
from agentos.models import Task, TaskCancelled


def test_await_cancelable_raises_promptly_on_cancel():
    async def inner():
        task = Task(goal="x")

        async def never_returns():
            await asyncio.sleep(3600)

        async def cancel_soon():
            await asyncio.sleep(0.1)
            task.cancel_requested = True

        asyncio.create_task(cancel_soon())
        loop = asyncio.get_event_loop()
        started = loop.time()
        with pytest.raises(TaskCancelled):
            await _await_cancelable(task, never_returns())
        assert loop.time() - started < 2.0

    asyncio.run(inner())


def test_await_cancelable_tears_down_an_in_flight_operation():
    """A call already in flight must be torn down, not left running orphaned."""
    async def inner():
        task = Task(goal="x")
        observed: dict = {}
        running = asyncio.Event()

        async def never_returns():
            running.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                observed["cancelled"] = True
                raise

        async def cancel_once_running():
            await running.wait()
            task.cancel_requested = True

        asyncio.create_task(cancel_once_running())
        with pytest.raises(TaskCancelled):
            await _await_cancelable(task, never_returns())
        await asyncio.sleep(0.05)
        assert observed.get("cancelled") is True

    asyncio.run(inner())


def test_await_cancelable_skips_work_already_cancelled():
    """Cancel set before the call starts: the operation must never run at all."""
    async def inner():
        task = Task(goal="x")
        observed: dict = {}

        async def never_returns():
            observed["ran"] = True
            await asyncio.sleep(3600)

        task.cancel_requested = True
        with pytest.raises(TaskCancelled):
            await _await_cancelable(task, never_returns())
        assert "ran" not in observed

    asyncio.run(inner())


def test_generate_is_interrupted_by_cancel(tmp_path):
    """The actual wedge: _generate on a hanging client must yield to a cancel."""
    async def inner():
        task = Task(goal="x")
        brain = GeminiBrain.__new__(GeminiBrain)  # bypass __init__ (wants an API key)
        brain._models = ["fake-model"]
        brain._config = lambda instructions=None: None
        brain._repair_safety_acks = lambda contents: None

        hung = asyncio.Event()

        async def generate_content(**kwargs):
            hung.set()
            await asyncio.sleep(3600)

        brain.client = pytypes.SimpleNamespace(
            aio=pytypes.SimpleNamespace(models=pytypes.SimpleNamespace(
                generate_content=generate_content)))

        log = RunLog(task.id, root=tmp_path)

        async def cancel_once_hung():
            await hung.wait()
            await asyncio.sleep(0.1)
            task.cancel_requested = True

        asyncio.create_task(cancel_once_hung())
        loop = asyncio.get_event_loop()
        started = loop.time()
        with pytest.raises(TaskCancelled):
            await brain._generate([], log, 1, None, task)
        assert loop.time() - started < 3.0

    asyncio.run(inner())


def test_generate_without_task_still_works(tmp_path):
    """The budget-synthesis call passes no task; it must not regress."""
    async def inner():
        brain = GeminiBrain.__new__(GeminiBrain)
        brain._models = ["fake-model"]
        brain._config = lambda instructions=None: None
        brain._repair_safety_acks = lambda contents: None
        sentinel = pytypes.SimpleNamespace(candidates=[], usage_metadata=None)

        async def generate_content(**kwargs):
            return sentinel

        brain.client = pytypes.SimpleNamespace(
            aio=pytypes.SimpleNamespace(models=pytypes.SimpleNamespace(
                generate_content=generate_content)))
        log = RunLog("t", root=tmp_path)
        assert await brain._generate([], log, 1, None) is sentinel

    asyncio.run(inner())


def test_model_call_has_a_bounded_timeout():
    """A stalled connection must not be able to hang the loop forever."""
    assert 0 < MODEL_CALL_TIMEOUT_MS <= 600_000


def test_transient_errors_are_retryable_but_bugs_are_not():
    assert GeminiBrain._is_transient(TimeoutError("timed out"))
    assert GeminiBrain._is_transient(Exception("503 Service Unavailable"))
    assert GeminiBrain._is_transient(Exception("connection reset by peer"))
    assert not GeminiBrain._is_transient(NameError("name 'task' is not defined"))
    assert not GeminiBrain._is_transient(ValueError("bad argument"))
