"""Model picker: GET/POST /models, key gating, and the promise that a task
already running keeps the brain it started with. StubBrain + aiohttp's
in-process client, so no Docker, no API keys, no calls."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentos import daemon as daemon_mod
from agentos.brain import MODEL_CATALOG, StubBrain
from agentos.daemon import Daemon
from agentos.logs import RunLog
from agentos.models import Task

TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63fcff9fa11e0000078400816cb0dc0000000049454e44ae426082"
)


class FakeSandbox:
    width = 1280
    height = 800

    async def screenshot(self) -> bytes:
        return TINY_PNG


class MarkerBrain(StubBrain):
    """A stub that records which instance actually ran a task."""

    def __init__(self, name: str, steps: int = 1):
        super().__init__(steps=steps)
        self.name = name
        self.model = name


def _client(tmp_path, brain=None):
    daemon = Daemon(brain=brain or StubBrain(steps=1), sandbox=FakeSandbox(),
                    runs_root=tmp_path)
    return daemon, TestClient(TestServer(daemon.build_app()))


@pytest.fixture(autouse=True)
def _no_provider_keys(monkeypatch):
    """Tests must not depend on the developer's own .env."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_catalog_entries_are_well_formed():
    for spec in MODEL_CATALOG:
        assert spec["provider"] in ("gemini", "openrouter", "stub")
        assert spec["id"] and spec["label"] and spec["note"]


def test_get_models_marks_current_and_availability(tmp_path):
    async def inner():
        daemon, client = _client(tmp_path)
        async with client:
            payload = await (await client.get("/models")).json()
            assert payload["current"] == {"provider": "stub", "id": "stub",
                                          "brain": "StubBrain"}
            by_id = {m["id"]: m for m in payload["models"]}
            assert by_id["stub"]["current"] and by_id["stub"]["available"]
            # No keys in the environment, so every hosted model is still offered
            # but marked unavailable, with the variable to set.
            assert not by_id["gemini-3.5-flash"]["available"]
            assert by_id["gemini-3.5-flash"]["requires"] == "GEMINI_API_KEY"
            assert by_id["google/gemma-4-31b-it"]["requires"] == "OPENROUTER_API_KEY"

    asyncio.run(inner())


def test_model_outside_catalog_still_shows_as_current(tmp_path):
    async def inner():
        daemon, client = _client(tmp_path, brain=MarkerBrain("some/custom-model"))
        async with client:
            payload = await (await client.get("/models")).json()
            current = [m for m in payload["models"] if m["current"]]
            assert [m["id"] for m in current] == ["some/custom-model"]

    asyncio.run(inner())


def test_switch_requires_a_key(tmp_path):
    async def inner():
        daemon, client = _client(tmp_path)
        async with client:
            resp = await client.post("/models", json={"id": "gemini-3.5-flash"})
            assert resp.status == 400
            assert "GEMINI_API_KEY" in await resp.text()
            assert daemon.brain.model == "stub"          # unchanged

    asyncio.run(inner())


def test_unknown_id_is_rejected_unless_a_provider_is_given(tmp_path, monkeypatch):
    async def inner():
        daemon, client = _client(tmp_path)
        async with client:
            resp = await client.post("/models", json={"id": "mystery/model"})
            assert resp.status == 400
            assert "provider" in await resp.text()

            # With a provider (and its key) an id outside the catalog is allowed:
            # the catalog is a shortlist, not a whitelist.
            monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
            built = []
            monkeypatch.setattr(
                daemon_mod, "build_brain",
                lambda provider, model: (built.append((provider, model))
                                         or MarkerBrain(model)))
            resp = await client.post(
                "/models", json={"id": "mystery/model", "provider": "openrouter"})
            assert resp.status == 200
            assert built == [("openrouter", "mystery/model")]
            assert daemon.brain.model == "mystery/model"

    asyncio.run(inner())


def test_switch_swaps_the_brain_and_reuses_it_on_the_way_back(tmp_path, monkeypatch):
    async def inner():
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        daemon, client = _client(tmp_path)
        original = daemon.brain
        monkeypatch.setattr(daemon_mod, "build_brain",
                            lambda provider, model: MarkerBrain(model or provider))
        async with client:
            payload = await (await client.post(
                "/models", json={"id": "gemini-3.5-flash"})).json()
            assert payload["current"]["id"] == "gemini-3.5-flash"
            switched = daemon.brain
            assert switched is not original

            # Back to the stub, then forward again: the second switch hands back
            # the very same instance rather than building a new transport.
            await client.post("/models", json={"id": "stub"})
            assert daemon.brain is original
            await client.post("/models", json={"id": "gemini-3.5-flash"})
            assert daemon.brain is switched

            health = await (await client.get("/health")).json()
            assert health["model"] == "gemini-3.5-flash"

    asyncio.run(inner())


def test_running_task_keeps_the_brain_it_started_with(tmp_path):
    """The point of pinning: swapping models must not hand a new model somebody
    else's half-finished conversation."""
    async def inner():
        daemon = Daemon(brain=MarkerBrain("first"), sandbox=FakeSandbox(),
                        runs_root=tmp_path)
        task = Task(goal="x")
        started = daemon.brain

        async def swap_midway():
            await asyncio.sleep(0.05)
            daemon.brain = MarkerBrain("second")

        runner = asyncio.ensure_future(
            daemon._run_with_deadline(task, RunLog(task.id, root=tmp_path), started))
        result, _ = await asyncio.gather(runner, swap_midway())
        assert result.startswith("stub brain")
        assert daemon.brain.name == "second"    # the picker did move on
        assert task.steps_taken == 1            # …but this run finished on `first`

    asyncio.run(inner())
