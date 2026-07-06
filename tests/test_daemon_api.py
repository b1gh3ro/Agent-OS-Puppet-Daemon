"""Pause / resume / guidance / steps-feed API tests. No Docker, no Gemini —
StubBrain against a fake sandbox, aiohttp's in-process test client."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentos.brain import GeminiBrain, StubBrain, pause_gate
from agentos.daemon import Daemon
from agentos.logs import RunLog
from agentos.models import Task, TaskCancelled, TaskStatus

# Smallest valid PNG (1x1 transparent pixel).
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63fcff9fa11e0000078400816cb0dc0000000049454e44ae426082"
)


class FakeSandbox:
    width = 1280
    height = 800

    async def screenshot(self) -> bytes:
        return TINY_PNG


async def _wait_for(predicate, timeout=10.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met in time")


def _client(tmp_path, steps=3):
    daemon = Daemon(brain=StubBrain(steps=steps), sandbox=FakeSandbox(),
                    runs_root=tmp_path)
    return daemon, TestClient(TestServer(daemon.build_app()))


def test_to_dict_new_fields():
    task = Task(goal="x")
    d = task.to_dict()
    for key in ("cancel_requested", "pause_requested", "paused", "pending_guidance"):
        assert key in d
    json.dumps(d)  # must stay JSON-serializable
    task.guidance.append("hint")
    assert task.to_dict()["pending_guidance"] == 1


def test_is_terminal():
    task = Task(goal="x")
    assert not task.is_terminal
    for status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED):
        task.status = status
        assert task.is_terminal


def test_pause_gate_unit(tmp_path):
    async def inner():
        task = Task(goal="x")
        log = RunLog(task.id, root=tmp_path)

        # No pause requested: returns immediately.
        await pause_gate(task, log, 1)
        assert not task.paused

        # Pause requested: blocks, sets paused, unblocks when cleared.
        task.pause_requested = True
        gate = asyncio.ensure_future(pause_gate(task, log, 2))
        await _wait_for(lambda: task.paused)
        assert not gate.done()
        task.pause_requested = False
        await gate
        assert not task.paused

        # Cancel while paused: raises TaskCancelled.
        task.pause_requested = True
        gate = asyncio.ensure_future(pause_gate(task, log, 3))
        await _wait_for(lambda: task.paused)
        task.cancel_requested = True
        with pytest.raises(TaskCancelled):
            await gate
        assert not task.paused

        kinds = [json.loads(line)["kind"]
                 for line in (tmp_path / task.id / "steps.jsonl").read_text().splitlines()]
        assert kinds == ["paused", "resumed", "paused"]

    asyncio.run(inner())


def test_wait_for_user_unit(tmp_path):
    async def inner():
        task = Task(goal="x")
        log = RunLog(task.id, root=tmp_path)

        waiter = asyncio.ensure_future(
            GeminiBrain._wait_for_user(task, {"message": "log in for me"}, log, 1))
        await _wait_for(lambda: task.paused)
        assert task.wait_message == "log in for me"
        assert task.pause_requested and not waiter.done()

        task.pause_requested = False  # operator hits Resume
        result = await waiter
        assert result["resumed"] is True
        assert task.wait_message is None and not task.paused

        kinds = [json.loads(line)["kind"]
                 for line in (tmp_path / task.id / "steps.jsonl").read_text().splitlines()]
        assert kinds == ["wait_for_user", "paused", "resumed"]

    asyncio.run(inner())


def test_drain_guidance_unit(tmp_path):
    task = Task(goal="x")
    log = RunLog(task.id, root=tmp_path)
    contents: list = []

    GeminiBrain._drain_guidance(task, contents, log, 1)
    assert contents == []  # nothing pending, nothing appended

    task.guidance += ["click the second link", "then scroll down"]
    GeminiBrain._drain_guidance(task, contents, log, 1)
    assert task.guidance == []
    assert len(contents) == 1
    assert contents[0].role == "user"
    text = contents[0].parts[0].text
    assert "click the second link" in text and "then scroll down" in text

    events = [json.loads(line)
              for line in (tmp_path / task.id / "steps.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events] == ["guidance", "guidance"]


def test_pause_resume_flow(tmp_path):
    async def inner():
        daemon, client = _client(tmp_path, steps=2)
        async with client:
            resp = await client.post("/tasks", json={"goal": "stub"})
            tid = (await resp.json())["id"]

            resp = await client.post(f"/tasks/{tid}/pause")
            assert resp.status == 200
            task = daemon.tasks[tid]
            await _wait_for(lambda: task.paused)
            assert task.status == TaskStatus.RUNNING

            # Frozen at the gate: steps don't advance while paused.
            frozen = task.steps_taken
            await asyncio.sleep(0.6)
            assert task.steps_taken == frozen

            resp = await client.post(f"/tasks/{tid}/resume")
            assert resp.status == 200
            await _wait_for(lambda: task.status == TaskStatus.DONE)
            assert not task.paused

    asyncio.run(inner())


def test_cancel_while_paused(tmp_path):
    async def inner():
        daemon, client = _client(tmp_path, steps=50)
        async with client:
            resp = await client.post("/tasks", json={"goal": "stub"})
            tid = (await resp.json())["id"]
            await client.post(f"/tasks/{tid}/pause")
            task = daemon.tasks[tid]
            await _wait_for(lambda: task.paused)

            await client.post(f"/tasks/{tid}/cancel")
            await _wait_for(lambda: task.status == TaskStatus.CANCELLED)
            assert not task.paused

    asyncio.run(inner())


def test_guidance_endpoint(tmp_path):
    async def inner():
        daemon, client = _client(tmp_path, steps=2)
        async with client:
            resp = await client.post("/tasks/nope/guidance", json={"text": "hi"})
            assert resp.status == 404

            resp = await client.post("/tasks", json={"goal": "stub"})
            tid = (await resp.json())["id"]
            task = daemon.tasks[tid]

            resp = await client.post(f"/tasks/{tid}/guidance", json={"text": "  "})
            assert resp.status == 400
            resp = await client.post(f"/tasks/{tid}/guidance", json={"text": "try harder"})
            assert resp.status == 200

            await _wait_for(lambda: task.status == TaskStatus.DONE)
            for path in ("guidance", "pause", "resume"):
                resp = await client.post(f"/tasks/{tid}/{path}", json={"text": "late"})
                assert resp.status == 409, path

            # StubBrain logs consumed guidance to the run log.
            kinds = [json.loads(line)["kind"]
                     for line in (tmp_path / tid / "steps.jsonl").read_text().splitlines()]
            assert "guidance" in kinds

    asyncio.run(inner())


def test_steps_endpoint(tmp_path):
    async def inner():
        daemon, client = _client(tmp_path)
        async with client:
            # Unknown id -> 404; known task with no run dir -> empty.
            resp = await client.get("/tasks/nope/steps")
            assert resp.status == 404

            task = Task(goal="x")
            daemon.tasks[task.id] = task  # registered but never run
            resp = await client.get(f"/tasks/{task.id}/steps")
            assert await resp.json() == {"events": [], "next": 0}

            run_dir = tmp_path / task.id
            run_dir.mkdir()
            lines = [json.dumps({"ts": i, "step": i, "kind": "action"}) for i in range(3)]
            (run_dir / "steps.jsonl").write_text("\n".join(lines) + "\n{half-writ")

            resp = await client.get(f"/tasks/{task.id}/steps")
            body = await resp.json()
            assert len(body["events"]) == 3
            assert body["next"] == 3  # corrupt tail line not consumed

            resp = await client.get(f"/tasks/{task.id}/steps?after=2")
            body = await resp.json()
            assert len(body["events"]) == 1 and body["events"][0]["step"] == 2

            resp = await client.get(f"/tasks/{task.id}/steps?after=oops")
            assert resp.status == 400

    asyncio.run(inner())


def test_screenshot_route(tmp_path):
    async def inner():
        daemon, client = _client(tmp_path)
        async with client:
            task = Task(goal="x")
            daemon.tasks[task.id] = task
            run_dir = tmp_path / task.id
            run_dir.mkdir()
            (run_dir / "step_001.png").write_bytes(TINY_PNG)

            resp = await client.get(f"/runs/{task.id}/step_001.png")
            assert resp.status == 200
            assert await resp.read() == TINY_PNG

            for bad in ("steps.jsonl", "step_1.png", "..%2F..%2Fetc%2Fpasswd"):
                resp = await client.get(f"/runs/{task.id}/{bad}")
                assert resp.status == 404, bad
            resp = await client.get("/runs/unknown/step_001.png")
            assert resp.status == 404

    asyncio.run(inner())
