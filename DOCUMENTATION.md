# agentos — Complete Guide

An autonomous OS-level AI agent: a persistent daemon that looks at a virtual Linux desktop through screenshots, decides what to do using Gemini's computer-use model, and executes real mouse/keyboard actions inside an isolated Docker container. You give it a goal in plain English over HTTP; it drives a browser like a person would and reports back.

Everything runs inside WSL2 (Ubuntu) with Docker Desktop. Nothing the agent does can touch your actual machine — it lives entirely inside the container.

---

## 1. The big picture

```
you ──POST /tasks──▶ ┌─ daemon (WSL2 host, one asyncio process, 127.0.0.1:8420) ─┐
                     │                                                            │
                     │   aiohttp HTTP API ──▶ asyncio.Queue ──▶ worker            │
                     │                                            │               │
                     │        the ReAct loop, once per step:      │               │
                     │   screenshot ─▶ Gemini ─▶ action ─▶ repeat ◀┘              │
                     └───────────────────┬────────────────────────────────────────┘
                                         │ docker exec (scrot / xdotool)
                     ┌─ Docker container "agent-sandbox" ─────────────────────────┐
                     │  Xvfb :99 (invisible 1280x800 display)                     │
                     │  openbox (window manager) · Firefox ESR (pre-launched)     │
                     │  scrot (screen capture) · xdotool (mouse/keyboard inject)  │
                     │  x11vnc :5900 · noVNC http://localhost:6080 (live view)    │
                     └────────────────────────────────────────────────────────────┘
```

**Life of a task, step by step:**

1. You `POST /tasks` with a goal. The daemon creates a `Task`, puts it on an in-memory queue, returns you an id immediately.
2. A worker picks the task off the queue and hands it to the **brain**.
3. The brain takes a screenshot of the virtual desktop (via `docker exec ... scrot`), sends it to Gemini together with the goal.
4. Gemini replies with **one action** — e.g. `click {x: 350, y: 108}` — with coordinates on a normalized 0–1000 grid.
5. The brain denormalizes the coordinates to real pixels, executes the action via `docker exec ... xdotool`, waits a beat for the UI to settle, takes a fresh screenshot, and sends it back to Gemini.
6. Repeat 3–5 until Gemini answers in plain text instead of an action → that text becomes the task's `result`.
7. Every step (action, intent, screenshot) is logged to `runs/<task-id>/` as evidence.

Safety rails bound the loop: a step budget, a wall-clock timeout, and a cancel endpoint checked between every step.

---

## 2. Requirements & setup

- **WSL2** (Ubuntu) with **Docker Desktop** using the WSL2 backend (`docker` works inside Ubuntu)
- **Python 3.12+** on the WSL side
- A **Gemini API key** (free tier works) in `.env`:

```
GEMINI_API_KEY=your-key-here
```

One-time setup:

```bash
cd /mnt/d/saadm/Documents/agent

docker build -t agent-sandbox sandbox/     # build the virtual desktop image

python3 -m venv .venv-wsl                  # Linux venv (.venv is an old Windows venv — ignore it)
.venv-wsl/bin/pip install -r requirements.txt
```

Dependencies (requirements.txt): `aiohttp` (HTTP server), `google-genai` (Gemini SDK), `python-dotenv` (.env loading), `pillow` (screenshot downscaling), `pytest` (tests).

---

## 3. Running it

**Start the daemon** (starts the sandbox container automatically if it isn't running):

```bash
.venv-wsl/bin/python -m agentos.daemon
```

You'll see: `agentos daemon in auto mode on http://127.0.0.1:8420 (sandbox: agent-sandbox, brain: GeminiBrain)`. Ctrl+C stops it. "Auto mode" means tasks execute end-to-end with no per-action confirmation — the safety rails are what keep that sane.

**The dashboard**: open **http://localhost:8420** in your Windows browser. It's a single-page mission control served by the daemon itself: submit tasks, watch the live activity feed (every action with the model's stated intent), pause/steer/cancel running tasks, see the latest screenshot, and view the live desktop in an embedded noVNC frame. The raw noVNC view is still at **http://localhost:6080/vnc.html** (click Connect, no password).

**Submit a task:**

```bash
curl -X POST localhost:8420/tasks -H 'Content-Type: application/json' \
  -d '{"goal": "Go to en.wikipedia.org, search for Alan Turing, and report his date of birth.", "max_steps": 25}'
```

**Full API:**

| Endpoint | What it does |
|---|---|
| `POST /tasks` | Submit. Body: `{"goal": "...", "max_steps": 40, "timeout_seconds": 600}` (last two optional). Returns the task with its `id`. |
| `GET /tasks/<id>` | Status, steps taken, and `result` when done. |
| `GET /tasks` | All tasks this daemon has seen since it started. |
| `POST /tasks/<id>/cancel` | Kill switch — checked between every step. |
| `POST /tasks/<id>/pause` | Freeze the agent at its next step boundary. 409 if the task already finished. |
| `POST /tasks/<id>/resume` | Clear the pause; the agent continues (pending guidance is injected first). |
| `POST /tasks/<id>/guidance` | Body: `{"text": "..."}`. Queue a steering message — injected into the model's conversation as a user turn before its next step. Works paused or mid-flight. |
| `GET /tasks/<id>/steps?after=N` | The run's step events as JSON (parsed from `steps.jsonl`), skipping the first `N` lines. Returns `{"events": [...], "next": M}` — pass `M` back as `after` to poll incrementally. |
| `GET /runs/<id>/step_NNN.png` | A run's screenshots (whitelisted filenames only). |
| `GET /` | The dashboard (single static page, no build step). |
| `GET /health` | Queue depth, task count, which brain is loaded. |

**Pause semantics worth knowing:** pause is cooperative — it takes effect at the next step boundary, never mid-action or mid-model-call, and `paused` in `GET /tasks/<id>` tells you when the freeze has actually happened (vs `pause_requested`, which is just your intent). The wall-clock `timeout_seconds` keeps ticking while paused (see §6). Cancelling a paused task works. Guidance sent while the model call is already in flight lands on the step after.

**Useful daemon flags:** `--brain stub` (test the whole pipeline without API calls — the stub just takes screenshots), `--port`, `--container`, `--no-container-autostart`.

**Tests:**

```bash
.venv-wsl/bin/python -m pytest tests/ -q
```

**Goal-writing tips learned from real runs:**
- End with an explicit *"report X"* so the model knows what the final answer must contain.
- Name the site: "go to site Z and do X" converges; "find me a good Y" wanders.
- Give a fallback for walled content: *"if the story links to a site you cannot read (like x.com), summarize from the comments instead"* — this exact clause is what turned a failed run into a successful one.
- Budget 40–50 steps for anything multi-site.

---

## 4. Repo layout

```
agentos/                 the Python package (the daemon side)
  models.py              Task dataclass + statuses — pure data, no I/O
  daemon.py              HTTP ingress + queue + workers (entrypoint)
  brain.py               Gemini computer-use ReAct loop + StubBrain
  sandbox.py             Sandbox protocol + DockerSandbox (docker exec)
  scaling.py             0–1000 grid → pixel coordinate math
  logs.py                per-run JSONL + screenshot evidence trail
  static/index.html      the dashboard — one self-contained page, no build step
sandbox/                 the Docker image (the desktop side)
  Dockerfile             Debian + Xvfb/openbox/xdotool/scrot/Firefox/noVNC
  entrypoint.sh          boots the virtual desktop
  policies.json          kills Firefox's first-run wizard & telemetry
tests/test_scaling.py    unit tests for the coordinate math
tests/test_daemon_api.py pause/steer/cancel + steps-feed API tests (no Docker/Gemini)
runs/<task-id>/          created at runtime: steps.jsonl + step screenshots
```

The dependency rule that keeps it modular: **daemon → brain → sandbox**, never sideways or backwards. `daemon.py` knows nothing about Gemini or Docker; `brain.py` knows nothing about HTTP or Docker (only the abstract `Sandbox` interface); `sandbox.py` knows nothing about models or tasks. Any layer can be swapped without touching the others.

---

## 5. Module walkthrough

### 5.1 `agentos/models.py` — the shared vocabulary

```python
@dataclass
class Task:
    goal: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    steps_taken: int = 0
    max_steps: int = 40
    timeout_seconds: float = 600.0
    cancel_requested: bool = False
    pause_requested: bool = False    # operator intent, set/cleared over HTTP
    paused: bool = False             # observed state, set/cleared only by the brain
    guidance: list[str] = field(default_factory=list)   # pending steering messages
```

One dataclass, one enum (`pending / running / done / failed / cancelled`), one exception (`TaskCancelled`). Deliberately boring: it's the contract every other module speaks, so it has zero dependencies and does zero I/O. Tasks live **only in memory** — a daemon restart forgets them. That's by design (see §6); the durable record is the run log.

The three control fields at the bottom are the operator's steering wheel, and they all follow the same pattern as `cancel_requested`: an HTTP handler mutates them, the brain reads them at each step boundary, and because both sides share one event loop no locking is needed. The `pause_requested`/`paused` split matters for the UI: the first is "the user asked", the second is "the agent actually froze" — a pause requested mid-model-call takes a few seconds to land.

### 5.2 `agentos/daemon.py` — ingress and dispatch

The heart is ~15 lines. Ingress:

```python
async def post_task(self, request):
    body = await request.json()
    task = Task(goal=body["goal"].strip(), ...)
    self.tasks[task.id] = task
    await self.queue.put(task)          # ← the entire "scheduler"
    return web.json_response(task.to_dict(), status=201)
```

And the worker loop:

```python
async def worker(self, n):
    while True:
        task = await self.queue.get()   # blocks until work arrives — no polling
        task.status = TaskStatus.RUNNING
        try:
            task.result = await asyncio.wait_for(
                self.brain.run_task(task, self.sandbox, run_log),
                timeout=task.timeout_seconds,          # wall-clock rail
            )
            task.status = TaskStatus.DONE
        except TaskCancelled:  task.status = TaskStatus.CANCELLED
        except TimeoutError:   task.status = TaskStatus.FAILED
        except Exception as e: task.status, task.error = TaskStatus.FAILED, str(e)
```

Why this shape: the original design brief called for SQLite in WAL mode with atomic locks, polled every second. All of that existed to coordinate *multiple processes*. Since the API handler and the workers live in **one asyncio event loop**, `await queue.get()` is already atomic — no locks, no polling, no race conditions, no database. The `asyncio.wait_for` wrapper is the timeout rail; the status transitions are what you see in `GET /tasks/<id>`.

The daemon runs **one worker by default** — there's one desktop, and two tasks fighting over one mouse would be chaos. `main()` also calls `ensure_container()` so a bare `python -m agentos.daemon` brings up the whole system.

Beyond the queue, the daemon is also the **control plane and file server for the dashboard**: pause/resume/guidance handlers just flip fields on the shared `Task` object (the brain notices at its next step), `GET /tasks/<id>/steps` parses `runs/<id>/steps.jsonl` into JSON with an `after=` cursor for cheap incremental polling (a half-written tail line is skipped and retried on the next poll), and `GET /runs/<id>/<name>` serves screenshots with both path segments whitelisted (`id` must be a known task, `name` must match `step_\d{3}\.png`) so the route can't be used for path traversal. Pause/resume/guidance return **409** on finished tasks — silently accepting steering for a task that will never read it would be lying to the operator. The `runs_root` constructor parameter keeps all of this testable against a tmp dir.

### 5.3 `agentos/sandbox.py` — the only place actions become real

The interface (a `Protocol`, so implementations are swappable):

```python
class Sandbox(Protocol):
    async def screenshot(self) -> bytes: ...
    async def click(self, x, y, button=1, repeat=1): ...
    async def move(self, x, y): ...
    async def type_text(self, text): ...
    async def key(self, combo): ...
    async def scroll(self, x, y, direction, magnitude=3): ...
    async def drag(self, x, y, dest_x, dest_y): ...
    async def mouse_button(self, x, y, down, button=1): ...
    async def key_state(self, key, down): ...
    async def launch(self, command): ...
```

`DockerSandbox` implements it with `docker exec` as the transport:

```python
async def screenshot(self) -> bytes:
    png = await self._exec("bash", "-c", "scrot -o /tmp/screen.png && cat /tmp/screen.png")
    if not png.startswith(b"\x89PNG"):
        raise SandboxError("screenshot did not return a PNG")
    return png

async def click(self, x, y, button=1, repeat=1):
    await self._exec("xdotool", "mousemove", "--sync", str(x), str(y),
                     "click", "--repeat", str(repeat), str(button))
```

`scrot` captures the virtual display to a PNG; `xdotool` injects synthetic X11 input events (XTEST extension) — the same mechanism a screen-reader or automation tool uses, no root needed. `mousemove --sync` waits until the pointer actually arrives before clicking. Each `docker exec` costs ~50–100 ms, which is nothing at ReAct cadence (the Gemini call is seconds). If that ever matters, the upgrade path is a tiny HTTP server inside the container implementing the same `Sandbox` protocol — no other file changes.

`ensure_container()` at the bottom inspects/starts/creates the container (`docker run -d --name agent-sandbox -p 127.0.0.1:5900:5900 -p 127.0.0.1:6080:6080 agent-sandbox`) so startup is self-healing.

### 5.4 `agentos/scaling.py` — the coordinate math

Gemini returns click coordinates on a **normalized 0–1000 grid** regardless of actual screen size, so `y: 937` on our 800-pixel-tall screen means "93.7% of the way down." Before xdotool sees it:

```python
def denormalize(x, y, width, height):
    px = round(x / GRID * width)     # GRID = 1000
    py = round(y / GRID * height)
    return min(max(px, 0), width - 1), min(max(py, 0), height - 1)
```

This replaced the original brief's inverse-scale-factor formula (`X_native = ⌊X_model × 1/S⌋`) — the normalized grid means there's no scale factor to track at all. `round` instead of `floor` (floor biases every click up-left), and clamping so coordinate `1000` can't land one pixel off-screen. Five unit tests in `tests/test_scaling.py` cover origin, max, midpoint, negative, and rounding cases.

### 5.5 `agentos/brain.py` — the cognitive layer (the biggest file)

**Declaring the tool.** One config block tells Gemini "you are operating a desktop":

```python
types.GenerateContentConfig(
    tools=[types.Tool(computer_use=types.ComputerUse(
        environment=types.Environment.ENVIRONMENT_DESKTOP,
        enable_prompt_injection_detection=True,
    ))],
)
```

**Model fallback.** `_generate()` tries `gemini-3.5-flash` first and falls back to the legacy `gemini-2.5-computer-use-preview-10-2025` on a model-not-found error, then locks in whichever worked. APIs drift; this keeps the daemon running across that drift.

**The ReAct loop** (`run_task`), condensed:

```python
contents = [goal_text + initial_screenshot]
for step in range(1, task.max_steps + 1):          # step-budget rail
    await pause_gate(task, log, step)              # cancel + pause rails
    self._drain_guidance(task, contents, log, step)  # operator steering

    response = await self._generate(contents)
    calls = [p.function_call for p in parts if p.function_call]
    if not calls:
        return text_of(parts)                       # plain text = final answer

    for fc in calls:
        await self._execute(fc.name, dict(fc.args), sandbox)   # act
        png = await self._settled_screenshot(sandbox)          # perceive
        contents.append(function_response_with(png))           # close the loop
    self._trim_screenshots(contents)
```

**The action dispatcher** (`_execute`) maps every action name Gemini can emit onto sandbox primitives. Discovered the hard way: today's `gemini-3.5-flash` emits `click / hotkey / press_key / wait / take_screenshot / move / type` (each with an `intent` string explaining itself), while the documented legacy vocabulary was `click_at / key_combination / type_text_at / wait_5_seconds`. The dispatcher handles **both**. Highlights:

```python
case "click" | "click_at":
    await sandbox.click(*point())                       # point() denormalizes
case "hotkey" | "key_combination":
    await sandbox.key(_to_xdotool_combo(args["keys"]))  # ["Ctrl","l"] → "ctrl+l"
case "navigate":
    await sandbox.key("ctrl+l"); await sandbox.type_text(args["url"]); await sandbox.key("Return")
case "take_screenshot":
    pass    # a fresh screenshot is returned after every action anyway
```

`_to_xdotool_combo` translates model key names to X11 keysyms (`Enter → Return`, `page_down → Page_Down`, ...). Unknown actions raise, and the error is **sent back to the model** in the function response — so it can read the error and try a different approach instead of the task dying.

**Custom tools: the agent has hands beyond the mouse.** Alongside the `computer_use` tool, `_config()` declares two ordinary function tools (`_CUSTOM_TOOLS`), and the system hint tells the model they exist:
- `run_command(command)` → `sandbox.exec_shell()`: runs any bash command in the container and returns `{exit_code, output}` (output capped at 4 KB) to the model in the function response. Combined with the container's passwordless sudo this is what makes "install a package / download a file / fix the environment" possible — sudo without a shell channel was a capability the model couldn't reach.
- `open_app(command)` → `sandbox.launch()`: starts a GUI program detached (e.g. `firefox-esr`, `xterm`). GUI programs must not go through `run_command` — they'd block until its timeout.

This exists because of a real failure: Firefox died in a long-lived container, and the model burned steps trying Ubuntu shortcuts (Alt+F2, Super) that plain openbox doesn't have — it had sudo but no way to *use* it, and no way to launch anything. Unknown-action errors also return to the model, so even a wrong tool call is recoverable.

**Screenshots are on-demand, not streamed.** `_settled_screenshot()` sleeps ~1 s after each action (letting the UI settle), captures, and downscales anything wider than 1366 px with Pillow. The original brief called for a 1 Hz screenshot stream to the model; that would burn tokens on frames where nothing changed. One screenshot per decision is all a ReAct loop needs.

**History trimming.** Each screenshot is ~200 KB. Forty steps of history would blow past request limits, so `_trim_screenshots()` blanks the image bytes out of all but the last 3 function responses, leaving `{"screenshot": "elided"}` markers. The model keeps its full *action* history but only recent *vision* — enough to stay oriented.

**Pause and steering.** `pause_gate` (shared by both brains) is the per-step control point: it raises `TaskCancelled` on cancel, and if `pause_requested` is set it sets `task.paused = True`, logs a `paused` event, and sleeps in 0.25 s ticks until the flag clears — still honoring cancel while frozen. `_drain_guidance` runs right after the gate: any queued operator messages are logged as `guidance` events and appended to `contents` as a single `role="user"` turn prefixed *"Operator guidance (incorporate into your next actions):"* — to the model it reads as fresh instruction between the last tool result and its next decision. Gate-then-drain ordering means guidance typed while paused is injected the moment you resume, before the next model call. `_trim_screenshots` never touches these turns (it only elides image blobs), so steering survives history trimming.

**Budget synthesis.** If the step budget runs out, the brain doesn't just give up — it sends one final message: *"You have run out of action budget... give your best final answer in plain text now"* and returns that as a clearly-labeled best-effort result. Added after a real run spent its budget mid-investigation and threw away everything it had learned.

**Safety acknowledgements.** When Gemini flags an action as needing confirmation (its own safety layer), auto mode acknowledges it — but logs `safety_auto_acknowledged` to the run log so there's always a record.

**`StubBrain`** at the top of the file is a 15-line fake brain that just takes N screenshots. It exists so `--brain stub` can smoke-test daemon + queue + sandbox + logging with zero API calls.

### 5.6 `agentos/logs.py` — the evidence trail

```python
class RunLog:
    def event(self, step, kind, **data):       # one JSON line per event
    def save_screenshot(self, step, png):      # runs/<id>/step_NNN.png
```

Every run produces `runs/<task-id>/steps.jsonl` — start, every action with the model's stated *intent*, every error, every screenshot path, the final status, plus the operator-control events `paused`, `resumed`, and `guidance` (with the message text), and `action_result` (a `run_command`'s exit code and first 1 KB of output — `run_command` never touches the visible desktop, so this is the only place a human sees what the shell returned). When the agent does something weird, this file is how you replay its reasoning — including what you told it mid-run. Tasks are in-memory, but the evidence is forever. The dashboard's activity feed is just this file, polled through `GET /tasks/<id>/steps`.

### 5.7 `sandbox/` — the virtual desktop image

**Dockerfile**, annotated:

```dockerfile
FROM debian:bookworm-slim      # NOT ubuntu:24.04 — Ubuntu ships Firefox only as
                               # a snap, which can't run in a container; Debian
                               # has firefox-esr as a normal apt package
RUN apt-get install ... xvfb openbox x11vnc xdotool scrot xterm firefox-esr novnc websockify
COPY policies.json /usr/share/firefox-esr/distribution/policies.json
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix   # Xvfb can't create
                               # this as non-root; pre-create it as root
RUN useradd -m agent && echo "agent ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/agent
USER agent                     # desktop processes run unprivileged (Firefox
                               # misbehaves as root), but the agent has
                               # passwordless sudo — it can install packages or
                               # change anything in its sandbox when a task needs it
```

The entrypoint also runs the browser in a **respawn loop** (`while true; do firefox-esr; sleep 2; done`) rather than launching it once: a crashed or model-closed Firefox used to leave an empty desktop for the rest of the container's life, stranding every later task.

What each package is: **xvfb** = an X server that renders to memory instead of a monitor (the "invisible screen"); **openbox** = minimal window manager (chosen over the brief's `mutter`, which drags in DBus/GNOME and is flaky headless); **x11vnc** = mirrors the display over VNC; **novnc + websockify** = serves that VNC session as a web page at :6080; **scrot/xdotool** = the agent's eye and hand; **firefox-esr** = what the agent actually drives.

**entrypoint.sh** boots them in order, with a real readiness check instead of a blind sleep:

```bash
Xvfb :99 -screen 0 1280x800x24 -nolisten tcp &
for _ in $(seq 1 50); do                       # wait for the display socket
    [ -S /tmp/.X11-unix/X99 ] && break; sleep 0.1
done
openbox &
x11vnc -display :99 -forever -shared -nopw &
websockify --web /usr/share/novnc 6080 localhost:5900 &
firefox-esr &        # pre-launched so tasks don't burn steps opening a browser
tail -f /dev/null    # PID 1 idles; the container lives until stopped
```

**policies.json** uses Firefox's enterprise-policy mechanism to disable the first-run wizard, telemetry, default-browser nag, and profile import — so the browser is instantly usable. Added after a real run wasted 9 of its 30 steps clicking through onboarding dialogs.

---

### 5.8 `agentos/static/index.html` — the dashboard

One self-contained HTML file (inline CSS + vanilla JS), served by the daemon at `GET /` with `Cache-Control: no-cache`. No framework, no build step, no new dependencies — the same philosophy as the rest of the repo.

Layout: header with a daemon-health dot; a left column with the new-task form and the task list (status badges, with distinct "pausing…"/"paused" states derived from `pause_requested` vs `paused`); a main pane for the selected task with Pause/Resume, Cancel, and a steering input, plus three tabs — **Activity** (the live step feed), **Screenshot** (the latest `step_NNN.png`), and **Live desktop** (an embedded noVNC iframe, lazily attached on first view so idle dashboards don't hold a VNC connection).

Mechanics worth knowing:
- Everything is polling: `GET /tasks` every 1.5 s, `GET /health` every 5 s, and the feed via `GET /tasks/<id>/steps?after=<cursor>` — the cursor makes each poll return only new events. The newest task is auto-selected on load so opening the page during a run drops you straight into the action.
- Screenshot `<img>` URLs carry a `?t=<now>` cache-buster because the brain **overwrites** `step_NNN.png` within a step (one file per step, one save per action).
- The noVNC URL (`http://localhost:6080/vnc.html`) is a single constant at the top of the script — there is one container and one fixed port today; per-task views only become a thing if multi-worker/multi-container lands.
- All user-controlled text (goals, guidance, event bodies) is rendered via `textContent`, never `innerHTML`, so a task goal containing HTML can't script the dashboard.

## 6. Design decisions and why

| Decision | Why |
|---|---|
| **No database** (in-memory queue + dict) | The brief's SQLite/WAL/atomic-locks design solved multi-process coordination — but everything lives in one asyncio process, where `queue.get()` is already atomic. Tasks don't need to survive restarts; run logs are the durable record. Deleted complexity: WAL mode, poll loops, lock choreography. |
| **HTTP on 127.0.0.1:8420 as the only ingress** | The queue is private process memory; an HTTP endpoint is the cheapest door into it, gives status/cancel for free, and localhost-only binding means nothing on your network can command the agent. |
| **Screenshots on demand, not 1 Hz** | A ReAct loop needs one screenshot per decision, after the previous action settles. A fixed stream burns tokens on unchanged frames. |
| **`docker exec` as transport** | Zero infrastructure inside the container. ~100 ms overhead is irrelevant next to multi-second model calls. Swappable later via the `Sandbox` protocol. |
| **openbox, not mutter** | mutter needs DBus/GNOME session scaffolding and is the reference implementations' main container headache; openbox just works. |
| **One worker** | One desktop, one mouse. Parallelism would need one container per worker (a clean future extension — `DockerSandbox` already takes a container name). |
| **Auto mode + rails instead of confirmations** | The point is autonomy. Step budgets, timeouts, cancel, and an unprivileged localhost-only sandbox make failures boring instead of dangerous. Pause/steer adds a human hand on the wheel *without* reintroducing per-action confirmations: the agent never waits for you unless you ask it to. |
| **Steering via flags + a list, not channels/locks** | Pause, cancel, and guidance are plain fields on `Task`, mutated by HTTP handlers and read by the brain at step boundaries — same event loop, so it's race-free by construction. An `asyncio.Queue` for guidance would add nothing but a JSON-serialization headache. |
| **Pause burns the timeout (v1)** | `asyncio.wait_for`'s deadline can't be extended mid-flight. Crediting paused time back needs a supervisor loop around the brain call — deferred (§10) rather than complicating the daemon's most failure-sensitive code path. A long pause can therefore time a task out; the dashboard shows both numbers. |
| **Dashboard feed = polling JSONL, not SSE/websockets** | Single local operator, and the append-only `steps.jsonl` already exists. A cursor-based poll every 1.5 s is ~40 lines total and survives page reloads for free; streaming infra would be complexity with no observable benefit here. |
| **Unprivileged container user, no host mounts** | Even a fully hijacked agent (e.g. via prompt injection on a malicious page) can only click around inside its own disposable desktop. |

---

## 7. Where the inspiration came from

- **[Anthropic's computer-use reference implementation](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)** — the single biggest influence on the sandbox: their container proved the exact Xvfb + window manager + xdotool + scrot recipe, and their agent loop (act → screenshot → feed back as tool result) is the shape our brain loop follows. We deviated on the window manager (openbox vs their mutter) and transport (docker exec vs in-container tooling).
- **[Google's Gemini Computer Use docs](https://ai.google.dev/gemini-api/docs/computer-use)** — the cognitive layer: the `computer_use` tool declaration, the normalized 0–1000 coordinate grid (which replaced the original scale-factor math), the function-response-with-screenshot loop contract, and `enable_prompt_injection_detection`.
- **[ScreenAgent (IJCAI-24)](https://github.com/niuzaisheng/ScreenAgent)** — academic validation of the whole concept: a VLM controlling a desktop in a Docker container via screenshots and mouse/keyboard, with a plan-act-reflect loop.
- **[Hugging Face ScreenEnv](https://huggingface.co/blog/screenenv)** — validation for the "isolated Ubuntu desktop in Docker as an agent environment" pattern and the idea that the sandbox should be a reusable, disposable unit.
- **[e2b open-computer-use](https://github.com/e2b-dev/open-computer-use)** — the swappable-brain idea: their config-driven LLM switching is why `brain.py` is isolated behind a tiny interface (swap Gemini for Claude or a local Ollama model by writing one new class).
- **The original engineering brief** — the daemon-with-task-queue architecture, port 8420, the sandbox isolation requirement, and the coordinate-scaling requirement all came from it; the implementation simplified the memory layer (no SQLite) and the perception model (no 1 Hz stream) after design review.

---

## 8. Field notes (things real runs taught us)

1. **Action vocabulary drift.** The current `gemini-3.5-flash` emits different action names than the documented preview model. First E2E run produced 8 actions, all "unsupported". Fix: dispatcher accepts both vocabularies. Lesson: log every action; send errors back to the model.
2. **Cold desktops waste budget.** A fresh container cost one run 9 steps of "how do I open a browser" (it tried Alt+F2, a terminal, the openbox menu...). Fix: pre-launch Firefox at boot + policies.json to kill onboarding.
3. **Budget death loses everything.** A run that exhausted 30 steps mid-investigation originally returned nothing. Fix: final synthesis pass asks for a best-effort answer from what was already seen.
4. **Walled content needs a fallback in the goal.** Top HN story was an x.com link (login wall). The run that failed had no instructions for this; the run that succeeded was told "if unreadable, summarize from the HN comments" — and did exactly that, including a headline-vs-reality verdict.

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `POST /tasks` connection refused | Daemon not running: `.venv-wsl/bin/python -m agentos.daemon` |
| Task stuck `pending` | Worker busy with an earlier task (one desktop = one task at a time) |
| Every action errors in `steps.jsonl` | Gemini changed action names again — extend `_execute` in `brain.py` |
| Screenshot errors | Container down: `docker start agent-sandbox` (or let the daemon autostart it) |
| noVNC won't connect | Container was restarted — click Connect again; check `docker ps` |
| Firefox shows first-run wizard | Old image — `docker build -t agent-sandbox sandbox/` and recreate the container |
| Want to test without spending API calls | `--brain stub` |

## 10. Extension points

- **New brain** (Claude computer-use, local Ollama VLM): one new class in `brain.py` implementing `run_task(task, sandbox, log)`.
- **Faster transport**: FastAPI server inside the container implementing the `Sandbox` protocol over HTTP.
- **Parallel tasks**: one container per worker; `DockerSandbox` already takes a container name.
- **Persistence**: if tasks ever must survive restarts, put SQLite *behind* the `Task` model — the daemon's queue logic doesn't change.
- **Pause that doesn't burn the timeout**: replace the worker's `asyncio.wait_for` with a small supervisor loop that tracks accumulated paused seconds and extends the deadline accordingly.
- **Always-on**: wrap the daemon in a systemd user unit or tmux session.
