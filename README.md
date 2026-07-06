# agentos

An AI agent that controls a real (but sandboxed) Linux desktop. You give it a goal in plain English; it looks at the screen, moves the mouse, types, and reports back what it found. Everything the AI touches lives inside a Docker container — it can never reach your actual machine.

Full architecture explanation: [DOCUMENTATION.md](DOCUMENTATION.md)

## What you need

- WSL2 (Ubuntu) with Docker Desktop connected to it (`docker ps` works inside Ubuntu)
- Python 3.12+
- A Gemini API key (free tier is fine): https://aistudio.google.com/apikey

## Make it work

All commands run inside WSL, from this folder.

**Step 1 — your API key.** Create a file named `.env` containing:

```
GEMINI_API_KEY=paste-your-key-here
```

**Step 2 — build the virtual desktop** (one time, ~2 minutes):

```bash
docker build -t agent-sandbox sandbox/
```

**Step 3 — install the Python side** (one time):

```bash
python3 -m venv .venv-wsl
.venv-wsl/bin/pip install -r requirements.txt
```

**Step 4 — start the agent:**

```bash
.venv-wsl/bin/python -m agentos.daemon
```

Leave this terminal open — it's the agent's brain stem and prints a log line for everything it does. It starts the desktop container automatically. You're ready when you see:

```
agentos daemon in auto mode on http://127.0.0.1:8420
```

**Step 5 — open the dashboard.** In your normal Windows browser, open:

```
http://localhost:8420
```

This is mission control: submit tasks, watch the live activity feed, see the
agent's latest screenshot, and open the **Live desktop** tab (the embedded
noVNC view of the sandbox — also available raw at `http://localhost:6080/vnc.html`).

**Step 6 — give it a job.** Type a goal into the dashboard and hit **Run**, or from a second terminal:

```bash
curl -X POST localhost:8420/tasks -H 'Content-Type: application/json' \
  -d '{"goal": "Go to en.wikipedia.org, search for Alan Turing, and report his date of birth.", "max_steps": 25}'
```

You get back an `id`. Now watch the Live desktop tab — the mouse will start moving on its own. Check the answer with:

```bash
curl localhost:8420/tasks/PUT-THE-ID-HERE
```

When `status` is `done`, the `result` field is the agent's answer.

**Steering a running task.** If the agent is going down the wrong path, you
don't have to kill it. From the dashboard: hit **Pause** (it freezes at its
next step), type a hint into the steer box, then **Resume** — or just send the
hint without pausing and it's injected before the agent's next move. Same
thing over HTTP:

```bash
curl -X POST localhost:8420/tasks/<id>/pause
curl -X POST localhost:8420/tasks/<id>/guidance -H 'Content-Type: application/json' \
  -d '{"text": "The first result is an ad — use the second link instead."}'
curl -X POST localhost:8420/tasks/<id>/resume
```

Note: the wall-clock timeout keeps ticking while paused.

## Everyday commands

```bash
curl localhost:8420/health                        # is it alive?
curl localhost:8420/tasks                         # all tasks this session
curl -X POST localhost:8420/tasks/<id>/cancel     # stop a runaway task
curl -X POST localhost:8420/tasks/<id>/pause      # freeze at the next step
curl -X POST localhost:8420/tasks/<id>/resume     # let it continue
curl "localhost:8420/tasks/<id>/steps?after=0"    # the step feed as JSON
cat runs/<id>/steps.jsonl                         # replay every action it took
.venv-wsl/bin/python -m pytest tests/ -q          # run the tests
```

## Stopping it

- **Ctrl+C** in the daemon terminal stops the agent.
- `docker stop agent-sandbox` stops the virtual desktop (the daemon restarts it next launch).
- Task history is forgotten on restart by design; the `runs/` folders keep the evidence.

