"""Standardized RQ1 driver: cost vs latency across waiting strategies (paper §V).

For each (arm, task, repeat) it:
  1. resets the sandbox and runs the task's setup script,
  2. runs the agent with exactly ONE waiting primitive exposed (the arm), so the
     comparison is between architectures, never between promptings,
  3. fires the task's ground-truth event at event_at on the monotonic clock,
  4. records model calls, tokens (cached/uncached/output/total), reaction latency
     (event -> first agent action), wall-clock, and codeword success.

Everything runs against a dedicated container (default agent-sandbox-exp) so a
live daemon is untouched. Durations come only from the run log's `mono` field,
never `ts` (see paper §V-A on suspend contamination). The run is resumable: a
completed (arm, task, repeat) already in the results file is skipped.

Run:  .venv-wsl/bin/python -m experiments.run_rq1 --repeats 5
      .venv-wsl/bin/python -m experiments.run_rq1 --arms poll,sleep,event,free --repeats 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from agentos.brain import GeminiBrain
from agentos.logs import RunLog
from agentos.models import Task, TaskCancelled
from agentos.sandbox import DockerSandbox, ensure_container
from experiments.tasks import ARMS, FREE_CHOICE, RUN_TIMEOUT, TASKS

logging.getLogger("google_genai.types").setLevel(logging.ERROR)  # silence AFC notices

CONTAINER = "agent-sandbox-exp"
OUT = Path("experiments/results")
MAX_STEPS = 80  # generous: even the poll arm reaches a 90 s event in <20 calls

# Experimental control: strip the escape hatches so the agent must actually
# WATCH the screen and use its one waiting primitive. With run_command in hand,
# the model just writes a shell loop that blocks until the window appears —
# bypassing the visual-waiting regime the experiment exists to compare. Removing
# these leaves exactly the GUI computer-use actions (click/type/take_screenshot/
# the built-in `wait`) plus whichever waiting primitive the arm exposes.
_NON_WAITING_CUSTOM_TOOLS = ("run_command", "open_app", "wait_for_user")


async def _sh(sandbox: DockerSandbox, cmds) -> None:
    for c in cmds:
        if c and c.strip():
            try:
                await sandbox.exec_shell(c, timeout=30)
            except Exception:
                pass  # a teardown pkill with nothing to kill is not an error


def _parse_metrics(logdir: Path, event_mono: float | None) -> dict:
    calls = 0
    tok = {"prompt": 0, "cached": 0, "uncached": 0, "output": 0, "total": 0}
    first_action_after = None
    monos: list[float] = []
    for line in (logdir / "steps.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = e.get("mono")
        if m is not None:
            monos.append(m)
        if e.get("kind") == "model_call":
            calls += 1
            tok["prompt"] += e.get("prompt_tokens", 0) or 0
            tok["cached"] += e.get("cached_tokens", 0) or 0
            tok["uncached"] += e.get("uncached_prompt_tokens", 0) or 0
            tok["output"] += e.get("output_tokens", 0) or 0
            tok["total"] += e.get("total_tokens", 0) or 0
        # Reaction = the agent's first committed response to the change. That is
        # an `action` OR a `done` (a final text answer, e.g. reporting the banner
        # text, emits `done` with no preceding action), whichever is first.
        if (e.get("kind") in ("action", "done") and event_mono is not None
                and m is not None and m >= event_mono and first_action_after is None):
            first_action_after = m
    latency = (round(first_action_after - event_mono, 2)
               if first_action_after is not None and event_mono is not None else None)
    wall = round(max(monos) - min(monos), 2) if len(monos) >= 2 else 0.0
    return {"model_calls": calls, **{f"tok_{k}": v for k, v in tok.items()},
            "reaction_latency_s": latency, "run_wall_s": wall}


async def run_one(arm: str, brain: GeminiBrain, task, rep: int,
                  sandbox: DockerSandbox) -> dict:
    run_id = f"{arm}-{task.name}-r{rep}"
    logdir = OUT / "runs" / run_id

    await _sh(sandbox, task.teardown)
    await _sh(sandbox, task.setup)

    t = Task(goal=task.goal, max_steps=MAX_STEPS, timeout_seconds=RUN_TIMEOUT)
    log = RunLog(run_id, root=str(OUT / "runs"))
    log.event(0, "exp_start", arm=arm, task=task.name, regime=task.regime,
              repeat=rep, noise_x_phi=task.noise_x_phi, event_at=task.event_at)

    event_mono: float | None = None
    started = time.monotonic()

    async def fire() -> None:
        nonlocal event_mono
        if not task.event:
            return  # control: nothing ever happens, so there is no reaction to time
        await asyncio.sleep(task.event_at)
        event_mono = time.monotonic()
        log.event(0, "event_fired", after=round(event_mono - started, 2))
        await _sh(sandbox, task.event)

    firing = asyncio.ensure_future(fire())
    result, err, timed_out = None, None, False
    try:
        result = await asyncio.wait_for(brain.run_task(t, sandbox, log),
                                        timeout=RUN_TIMEOUT + 30)
    except asyncio.TimeoutError:
        timed_out, result = True, "(timed out)"
    except TaskCancelled:
        result = "(cancelled)"
    except Exception as e:  # keep the batch alive; record the failure
        err = f"{type(e).__name__}: {e}"
    finally:
        firing.cancel()
        try:
            await firing
        except (asyncio.CancelledError, Exception):
            pass
        await _sh(sandbox, task.teardown)

    m = _parse_metrics(logdir, event_mono)
    answer = result or ""
    if task.codeword:
        success = task.codeword.lower() in answer.lower()
    else:  # control: the only correct outcome is the explicit negative
        success = "nothing appeared" in answer.lower()
    return {"arm": arm, "task": task.name, "regime": task.regime, "repeat": rep,
            "noise_x_phi": task.noise_x_phi, "steps": t.steps_taken,
            "success": success, "timed_out": timed_out, "error": err,
            "answer": answer[:200], **m}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--arms", default="poll,sleep,event,free")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", default="rq1.jsonl")
    ap.add_argument("--container", default=CONTAINER)
    a = ap.parse_args()

    load_dotenv()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    await ensure_container(container=a.container, image="agent-sandbox")
    sandbox = DockerSandbox(container=a.container)

    arm_tools = {**ARMS, "free": FREE_CHOICE}
    arms = [x for x in a.arms.split(",") if x]
    brains = {}
    for arm in arms:
        b = GeminiBrain(waiting_tools=arm_tools[arm])
        # Enforce the control: drop shell / app-launch / handoff so the only way
        # to wait is the visual channel + the arm's primitive.
        b._tools = [t for t in b._tools if t["name"] not in _NON_WAITING_CUSTOM_TOOLS]
        brains[arm] = b
    tasks = [t for t in TASKS if not a.tasks or t.name in a.tasks.split(",")]

    outpath = OUT / a.out
    done: set[tuple[str, str, int]] = set()
    if outpath.exists():
        for line in outpath.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add((r["arm"], r["task"], r["repeat"]))
            except (json.JSONDecodeError, KeyError):
                pass

    total = len(arms) * len(tasks) * a.repeats
    n = 0
    with outpath.open("a", encoding="utf-8") as f:
        for rep in range(a.repeats):          # one full sweep of every cell per rep
            for arm in arms:
                for task in tasks:
                    n += 1
                    key = (arm, task.name, rep)
                    if key in done:
                        print(f"[{n}/{total}] skip {arm}/{task.name}/r{rep}", flush=True)
                        continue
                    print(f"[{n}/{total}] {arm}/{task.name}/r{rep} ...", flush=True)
                    t0 = time.monotonic()
                    try:
                        row = await run_one(arm, brains[arm], task, rep, sandbox)
                    except Exception as e:
                        row = {"arm": arm, "task": task.name, "repeat": rep,
                               "error": f"driver: {type(e).__name__}: {e}"}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"    -> calls={row.get('model_calls')} "
                          f"lat={row.get('reaction_latency_s')} ok={row.get('success')} "
                          f"tout={row.get('timed_out')} err={row.get('error')} "
                          f"[{time.monotonic()-t0:.0f}s]", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    asyncio.run(main())
