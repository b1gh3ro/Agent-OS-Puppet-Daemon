"""Agent daemon: aiohttp API on 127.0.0.1:8420 + autonomous worker loop.

Tasks enter via POST /tasks, land on an in-memory asyncio.Queue, and are
executed fully autonomously (auto mode) by worker tasks. This module knows
nothing about Gemini or Docker beyond constructing the configured Brain and
Sandbox implementations at startup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv
from google.genai import types

from .brain import GeminiBrain, StubBrain
from .logs import RunLog
from .models import Task, TaskCancelled, TaskStatus
from .sandbox import DockerSandbox, Sandbox, ensure_container

log = logging.getLogger("agentos")


_SCREENSHOT_NAME = re.compile(r"^step_\d{3,}\.png$")


class Daemon:
    def __init__(self, brain, sandbox: Sandbox, workers: int = 1,
                 runs_root: str | Path = "runs"):
        self.brain = brain
        self.sandbox = sandbox
        self.workers = workers
        self.runs_root = Path(runs_root)
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.tasks: dict[str, Task] = {}
        self._load_persisted()  # bring back finished tasks (with their memory) after a restart

    # -- worker loop ---------------------------------------------------------

    async def worker(self, n: int) -> None:
        while True:
            task = await self.queue.get()
            if task.cancel_requested:
                task.status = TaskStatus.CANCELLED
                task.finished_at = time.time()
                continue
            task.status = TaskStatus.RUNNING
            run_log = RunLog(task.id, root=self.runs_root, base=task.prior_steps)
            run_log.event(0, "start", goal=task.goal, worker=n)
            log.info("worker %d: task %s started: %s", n, task.id, task.goal)
            try:
                task.result = await self._run_with_deadline(task, run_log)
                task.status = TaskStatus.DONE
            except TaskCancelled:
                task.status = TaskStatus.CANCELLED
            except TimeoutError:
                task.status = TaskStatus.FAILED
                task.error = f"timed out after {task.timeout_seconds}s"
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                log.exception("task %s failed", task.id)
            finally:
                task.finished_at = time.time()
                run_log.event(task.steps_taken, "final", status=task.status.value,
                              result=task.result, error=task.error)
                # Save the task + its model conversation so a restart can pick up
                # exactly where it left off (continue reuses task.history).
                self._persist_task(task)
                log.info("task %s finished: %s", task.id, task.status.value)

    async def _run_with_deadline(self, task: Task, run_log: RunLog) -> str | None:
        """Run the brain against a *mutable* wall-clock deadline.

        asyncio.wait_for freezes its timeout at call time, so a long sleep
        mid-task could never extend it. Instead we track task.deadline (a
        monotonic cutoff the sleep tool can push out) and poll the runner
        against it, cancelling only once the current deadline actually passes.
        """
        task.deadline = time.monotonic() + task.timeout_seconds
        runner = asyncio.ensure_future(self.brain.run_task(task, self.sandbox, run_log))
        try:
            while True:
                remaining = task.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                done, _ = await asyncio.wait({runner}, timeout=min(remaining, 30))
                if runner in done:
                    return runner.result()
        finally:
            if not runner.done():
                runner.cancel()
                try:
                    await runner
                except (asyncio.CancelledError, Exception):
                    pass

    # -- persistence ---------------------------------------------------------

    _PERSIST_FIELDS = ("id", "goal", "instructions", "result", "error",
                       "steps_taken", "prior_steps", "max_steps",
                       "timeout_seconds", "created_at", "finished_at")

    def _persist_task(self, task: Task) -> None:
        """Snapshot a finished task and its model conversation to its run dir.

        history.json holds the exact Gemini contents (types.Content is pydantic),
        so a reloaded task can be continued with its full memory intact."""
        d = self.runs_root / task.id
        try:
            d.mkdir(parents=True, exist_ok=True)
            meta = {k: getattr(task, k) for k in self._PERSIST_FIELDS}
            meta["status"] = task.status.value
            (d / "task.json").write_text(json.dumps(meta, default=str), encoding="utf-8")
            if task.history is not None:
                hist = [c.model_dump(mode="json") for c in task.history]
                (d / "history.json").write_text(json.dumps(hist), encoding="utf-8")
        except Exception:
            log.exception("could not persist task %s", task.id)

    def _load_persisted(self) -> None:
        """Rebuild self.tasks from task.json snapshots left by earlier runs."""
        if not self.runs_root.exists():
            return
        valid = {s.value for s in TaskStatus}
        for meta_path in sorted(self.runs_root.glob("*/task.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                log.exception("could not read %s", meta_path)
                continue
            task = Task(goal=meta.get("goal", ""), id=meta.get("id") or meta_path.parent.name)
            for k in ("instructions", "result", "error", "steps_taken",
                      "prior_steps", "max_steps", "timeout_seconds",
                      "created_at", "finished_at"):
                if meta.get(k) is not None:
                    setattr(task, k, meta[k])
            status = meta.get("status", "done")
            task.status = TaskStatus(status) if status in valid else TaskStatus.DONE
            # A snapshot saved as running/pending means the daemon died mid-run;
            # mark it terminal so the operator can still continue it.
            if not task.is_terminal:
                task.status = TaskStatus.FAILED
                task.error = task.error or "daemon restarted while this task was running"
            hist_path = meta_path.parent / "history.json"
            if hist_path.exists():
                try:
                    raw = json.loads(hist_path.read_text(encoding="utf-8"))
                    task.history = [types.Content.model_validate(c) for c in raw]
                except Exception:
                    log.exception("could not load history for %s", task.id)
            self.tasks[task.id] = task
        if self.tasks:
            log.info("restored %d task(s) from %s", len(self.tasks), self.runs_root)

    # -- HTTP API ------------------------------------------------------------

    async def post_task(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="body must be JSON")
        goal = (body.get("goal") or "").strip()
        if not goal:
            raise web.HTTPBadRequest(text='"goal" is required')
        task = Task(
            goal=goal,
            instructions=(body.get("instructions") or "").strip(),
            max_steps=int(body.get("max_steps", 300)),
            timeout_seconds=float(body.get("timeout_seconds", 3600)),
        )
        self.tasks[task.id] = task
        await self.queue.put(task)
        return web.json_response(task.to_dict(), status=201)

    async def get_tasks(self, request: web.Request) -> web.Response:
        return web.json_response([t.to_dict() for t in self.tasks.values()])

    async def get_task(self, request: web.Request) -> web.Response:
        task = self.tasks.get(request.match_info["id"])
        if not task:
            raise web.HTTPNotFound(text="no such task")
        return web.json_response(task.to_dict())

    async def cancel_task(self, request: web.Request) -> web.Response:
        task = self.tasks.get(request.match_info["id"])
        if not task:
            raise web.HTTPNotFound(text="no such task")
        task.cancel_requested = True
        return web.json_response(task.to_dict())

    def _get_live_task(self, request: web.Request) -> Task:
        task = self.tasks.get(request.match_info["id"])
        if not task:
            raise web.HTTPNotFound(text="no such task")
        if task.is_terminal:
            raise web.HTTPConflict(text=f"task is already {task.status.value}")
        return task

    async def pause_task(self, request: web.Request) -> web.Response:
        task = self._get_live_task(request)
        task.pause_requested = True
        return web.json_response(task.to_dict())

    async def resume_task(self, request: web.Request) -> web.Response:
        task = self._get_live_task(request)
        task.pause_requested = False   # un-pause / release a wait_for_user
        task.wake_requested = True     # ...and cut a running sleep short
        return web.json_response(task.to_dict())

    async def post_guidance(self, request: web.Request) -> web.Response:
        task = self._get_live_task(request)
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="body must be JSON")
        text = (body.get("text") or "").strip()
        if not text:
            raise web.HTTPBadRequest(text='"text" is required')
        task.guidance.append(text)
        return web.json_response(task.to_dict())

    async def continue_task(self, request: web.Request) -> web.Response:
        """Re-queue a finished task with a follow-up goal. The brain resumes
        from the saved conversation, so the model keeps everything it did and
        saw in the earlier run(s)."""
        task = self.tasks.get(request.match_info["id"])
        if not task:
            raise web.HTTPNotFound(text="no such task")
        if not task.is_terminal:
            raise web.HTTPConflict(text="task is still running — steer it with /guidance")
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="body must be JSON")
        goal = (body.get("goal") or "").strip()
        if not goal:
            raise web.HTTPBadRequest(text='"goal" is required')
        # Shift follow-up step numbers past this run's (+1 so its step-0
        # screenshot doesn't overwrite the previous run's last one).
        task.prior_steps += task.steps_taken + 1
        task.steps_taken = 0
        task.goal = goal
        task.max_steps = int(body.get("max_steps", task.max_steps))
        task.timeout_seconds = float(body.get("timeout_seconds", task.timeout_seconds))
        task.status = TaskStatus.PENDING
        task.result = None
        task.error = None
        task.finished_at = None
        task.cancel_requested = False
        task.pause_requested = False
        task.wake_requested = False
        task.wait_message = None
        task.wait_kind = None
        task.guidance.clear()
        await self.queue.put(task)
        return web.json_response(task.to_dict())

    async def get_steps(self, request: web.Request) -> web.Response:
        task = self.tasks.get(request.match_info["id"])
        if not task:
            raise web.HTTPNotFound(text="no such task")
        try:
            after = int(request.query.get("after", "0"))
        except ValueError:
            raise web.HTTPBadRequest(text='"after" must be an integer')
        path = self.runs_root / task.id / "steps.jsonl"
        if not path.exists():  # pending task, or cancelled before it started
            return web.json_response({"events": [], "next": after})
        events = []
        consumed = after
        with path.open() as f:
            for i, line in enumerate(f):
                if i < after:
                    continue
                if i != consumed:  # don't skip past an unparsed line
                    break
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # half-written tail line; retry it on the next poll
                consumed = i + 1
        return web.json_response({"events": events, "next": consumed})

    async def get_screenshot(self, request: web.Request) -> web.Response:
        task_id, name = request.match_info["id"], request.match_info["name"]
        # Whitelist both segments so the route can't be used for traversal.
        if task_id not in self.tasks or not _SCREENSHOT_NAME.match(name):
            raise web.HTTPNotFound()
        path = self.runs_root / task_id / name
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    async def index(self, request: web.Request) -> web.Response:
        return web.FileResponse(
            Path(__file__).parent / "static" / "index.html",
            headers={"Cache-Control": "no-cache"},
        )

    async def put_task_instructions(self, request: web.Request) -> web.Response:
        """Set/replace a job's standing instructions. Takes effect on the job's
        next step (the brain reads task.instructions fresh each call), so editing
        a running job permanently steers it from here on."""
        task = self.tasks.get(request.match_info["id"])
        if not task:
            raise web.HTTPNotFound(text="no such task")
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="body must be JSON")
        task.instructions = (body.get("instructions") or "").strip()
        log.info("task %s instructions updated (%d chars)", task.id, len(task.instructions))
        return web.json_response(task.to_dict())

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "ok": True,
            "queued": self.queue.qsize(),
            "tasks": len(self.tasks),
            "brain": type(self.brain).__name__,
        })

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/tasks", self.post_task)
        app.router.add_get("/tasks", self.get_tasks)
        app.router.add_get("/tasks/{id}", self.get_task)
        app.router.add_post("/tasks/{id}/cancel", self.cancel_task)
        app.router.add_post("/tasks/{id}/pause", self.pause_task)
        app.router.add_post("/tasks/{id}/resume", self.resume_task)
        app.router.add_post("/tasks/{id}/guidance", self.post_guidance)
        app.router.add_post("/tasks/{id}/continue", self.continue_task)
        app.router.add_get("/tasks/{id}/steps", self.get_steps)
        app.router.add_get("/runs/{id}/{name}", self.get_screenshot)
        app.router.add_put("/tasks/{id}/instructions", self.put_task_instructions)
        app.router.add_get("/health", self.health)
        app.router.add_get("/", self.index)

        async def start_workers(app: web.Application):
            app["workers"] = [asyncio.create_task(self.worker(i)) for i in range(self.workers)]

        async def stop_workers(app: web.Application):
            for w in app["workers"]:
                w.cancel()

        app.on_startup.append(start_workers)
        app.on_cleanup.append(stop_workers)
        return app


def make_brain(kind: str):
    stub_steps = int(os.getenv("AGENT_STUB_STEPS", "3"))
    if kind == "stub":
        return StubBrain(steps=stub_steps)
    if not os.getenv("GEMINI_API_KEY"):
        log.warning("GEMINI_API_KEY not set — falling back to stub brain")
        return StubBrain(steps=stub_steps)
    # Expose BOTH waiting primitives with neutral, matched-length wording so the
    # model picks sleep vs wait_for_screen_change on the task's merits, not
    # because the prompt or tool text steers it. (The biased production wording
    # remains available via waiting_tools=None for the deliberate A/B, but the
    # live daemon runs unbiased.)
    return GeminiBrain(model=os.getenv("AGENT_MODEL") or None,
                       waiting_tools=GeminiBrain.WAITING_TOOLS)


async def main() -> None:
    parser = argparse.ArgumentParser(prog="agentos", description="Autonomous OS-level agent daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--container", default="agent-sandbox")
    parser.add_argument("--brain", choices=["auto", "stub"], default="auto")
    parser.add_argument("--no-container-autostart", action="store_true",
                        help="assume the sandbox container is already running")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    if not args.no_container_autostart:
        await ensure_container(container=args.container, image="agent-sandbox")

    daemon = Daemon(brain=make_brain(args.brain), sandbox=DockerSandbox(container=args.container))
    runner = web.AppRunner(daemon.build_app())
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    log.info("agentos daemon in auto mode on http://%s:%d (sandbox: %s, brain: %s)",
             args.host, args.port, args.container, type(daemon.brain).__name__)
    await asyncio.Event().wait()  # run until killed


if __name__ == "__main__":
    asyncio.run(main())
