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

**Step 5 — watch its screen.** In your normal Windows browser, open:

```
http://localhost:6080/vnc.html
```

and click **Connect** (no password). This is the agent's desktop, live.

**Step 6 — give it a job.** In a second terminal:

```bash
curl -X POST localhost:8420/tasks -H 'Content-Type: application/json' \
  -d '{"goal": "Go to en.wikipedia.org, search for Alan Turing, and report his date of birth.", "max_steps": 25}'
```

You get back an `id`. Now watch the browser tab from step 5 — the mouse will start moving on its own. Check the answer with:

```bash
curl localhost:8420/tasks/PUT-THE-ID-HERE
```

When `status` is `done`, the `result` field is the agent's answer.

## Everyday commands

```bash
curl localhost:8420/health                        # is it alive?
curl localhost:8420/tasks                         # all tasks this session
curl -X POST localhost:8420/tasks/<id>/cancel     # stop a runaway task
cat runs/<id>/steps.jsonl                         # replay every action it took
.venv-wsl/bin/python -m pytest tests/ -q          # run the tests
```

## Stopping it

- **Ctrl+C** in the daemon terminal stops the agent.
- `docker stop agent-sandbox` stops the virtual desktop (the daemon restarts it next launch).
- Task history is forgotten on restart by design; the `runs/` folders keep the evidence.

## Writing good goals

- Always end with **"report X"** — tell it what the answer should contain.
- Name the website. "Go to site Z and do X" works; "find me something good" wanders.
- Give a fallback for sites that need login: *"if the page can't be read, summarize from the comments instead."*
- Multi-site tasks need `"max_steps": 40` or more.

## If something breaks

| Problem | Fix |
|---|---|
| `curl: connection refused` | Daemon isn't running (step 4) |
| Task stuck on `pending` | It runs one task at a time — wait or cancel the earlier one |
| noVNC won't connect | Click Connect again; check `docker ps` shows agent-sandbox |
| Want to test without using the API | `.venv-wsl/bin/python -m agentos.daemon --brain stub` |
