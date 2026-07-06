"""Agent daemon: aiohttp API on 127.0.0.1:8420 + autonomous worker loop.

Tasks enter via POST /tasks, land on an in-memory asyncio.Queue, and are
executed fully autonomously (auto mode) by worker tasks. This module knows
nothing about Gemini or Docker beyond constructing the configured Brain and
Sandbox implementations at startup.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from aiohttp import web
from dotenv import load_dotenv

from .brain import GeminiBrain, StubBrain
from .logs import RunLog
from .models import Task, TaskCancelled, TaskStatus
from .sandbox import DockerSandbox, Sandbox, ensure_container

log = logging.getLogger("agentos")


class Daemon:
    def __init__(self, brain, sandbox: Sandbox, workers: int = 1):
        self.brain = brain
        self.sandbox = sandbox
        self.workers = workers
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.tasks: dict[str, Task] = {}

    # -- worker loop ---------------------------------------------------------

    async def worker(self, n: int) -> None:
        while True:
            task = await self.queue.get()
            if task.cancel_requested:
                task.status = TaskStatus.CANCELLED
                task.finished_at = time.time()
                continue
            task.status = TaskStatus.RUNNING
            run_log = RunLog(task.id)
            run_log.event(0, "start", goal=task.goal, worker=n)
            log.info("worker %d: task %s started: %s", n, task.id, task.goal)
            try:
                task.result = await asyncio.wait_for(
                    self.brain.run_task(task, self.sandbox, run_log),
                    timeout=task.timeout_seconds,
                )
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
                log.info("task %s finished: %s", task.id, task.status.value)

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
            max_steps=int(body.get("max_steps", 40)),
            timeout_seconds=float(body.get("timeout_seconds", 600)),
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
        app.router.add_get("/health", self.health)

        async def start_workers(app: web.Application):
            app["workers"] = [asyncio.create_task(self.worker(i)) for i in range(self.workers)]

        async def stop_workers(app: web.Application):
            for w in app["workers"]:
                w.cancel()

        app.on_startup.append(start_workers)
        app.on_cleanup.append(stop_workers)
        return app


def make_brain(kind: str):
    if kind == "stub":
        return StubBrain()
    if not os.getenv("GEMINI_API_KEY"):
        log.warning("GEMINI_API_KEY not set — falling back to stub brain")
        return StubBrain()
    return GeminiBrain(model=os.getenv("AGENT_MODEL") or None)


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
