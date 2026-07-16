"""Cognitive layer: the ReAct loop between screenshots and sandbox actions.

GeminiBrain drives the Gemini computer-use tool: send goal + screenshot,
receive one UI action, execute it in the sandbox, send back a fresh
screenshot, repeat until the model answers in plain text. It depends only on
the Sandbox protocol and the Task model — no Docker, no HTTP.
"""

from __future__ import annotations

import asyncio
import io
import os
import time

from google import genai
from google.genai import types
from PIL import Image, ImageChops

from .logs import RunLog
from .models import Task, TaskCancelled
from .sandbox import Sandbox
from .scaling import denormalize

MODEL_CANDIDATES = [
    "gemini-3.5-flash",
    "gemini-2.5-computer-use-preview-10-2025",
]

MAX_SCREENSHOTS_IN_HISTORY = 3

SYSTEM_HINT = (
    "You are operating a Linux desktop ({w}x{h}) inside a sandbox running the "
    "openbox window manager (no taskbar, no app launcher, no Ubuntu/GNOME "
    "keyboard shortcuts; right-clicking the desktop opens a small menu). "
    "Besides the GUI actions you have two extra tools — use them freely:\n"
    "- run_command(command): run any bash command in the sandbox and get its "
    "output. You have passwordless sudo, so you can install packages "
    "('sudo apt-get install -y <pkg>'), download files, and inspect or fix "
    "anything without the GUI. This returns text only and does NOT take a "
    "screenshot — chain several shell commands cheaply this way. When you "
    "actually need to see the screen, call take_screenshot (or pass "
    "screenshot=true to run_command); don't waste actions screenshotting "
    "after pure shell work.\n"
    "- open_app(command): launch a GUI program detached, e.g. "
    "open_app(command='firefox-esr') or open_app(command='xterm').\n"
    "- wait_for_user(message): when you hit something only the human can do "
    "(a login, captcha, payment, or a decision), call this with clear "
    "instructions; the task pauses until they resume, then you get a fresh "
    "screenshot of what they did. The operator only sees this same desktop "
    "through a remote view — they cannot open apps, terminals, or files "
    "themselves. Before handing off, open whatever window they need, and in "
    "your message say exactly which window to use and where to click or "
    "type.\n"
    "Firefox ESR is installed and usually already open; if the screen looks "
    "empty, call open_app(command='firefox-esr') instead of hunting for a "
    "launcher. You have a budget of {max_steps} actions for this task; each "
    "action result tells you how many remain — make sure you deliver your "
    "final answer before it runs out.\n"
    "You can also issue SEVERAL actions in a single turn when they follow "
    "predictably and none depends on seeing the result of the one before — e.g. "
    "click a field, type text, then press Enter; or fire off several "
    "run_command calls at once. They execute in order and you get ONE "
    "screenshot after the last one, which saves round-trips. Split them back "
    "into separate turns whenever you genuinely need to see the screen before "
    "deciding the next move (a click that opens an unpredictable dialog, a "
    "search whose results you must read).\n"
    "- sleep(seconds): idle for minutes up to an hour (a download, a build, a "
    "scheduled event) with no cost while you wait — one call sleeps the whole "
    "time and returns a single screenshot. Use this, not repeated 'wait' "
    "actions, whenever you must wait longer than ~15 seconds.\n"
    "- wait_for_screen_change(timeout_seconds): when you're waiting for the "
    "display to update but don't know how long it takes (a page loading, a "
    "spinner, a dialog), use this instead of sleep — it returns the instant the "
    "screen changes, so you wait exactly as long as needed and no longer.\n"
    "Complete this task, then answer in plain text with the outcome:\n\n{goal}"
)

# Prepended instead of SYSTEM_HINT when the operator continues a finished
# task: the model already carries the session in its history.
CONTINUE_HINT = (
    "The operator has a follow-up for you in this same session. The desktop "
    "is as you left it; a fresh screenshot is attached. You have a new budget "
    "of {max_steps} actions. Complete the follow-up, then answer in plain "
    "text with the outcome:\n\n{goal}"
)

_CUSTOM_TOOLS = [
    {
        "name": "run_command",
        "description": (
            "Run a bash command inside the sandbox and return its exit code and "
            "combined stdout/stderr. Passwordless sudo is available. Use this for "
            "installing packages, downloading files, reading/writing files, and "
            "anything faster done in a shell than by clicking. Returns text only "
            "and takes NO screenshot by default (that keeps shell work fast) — "
            "set screenshot=true only if the command changes what's on screen "
            "and you need to see the result. Do not start GUI programs with this "
            "(they block); use open_app for those."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."},
                "intent": {"type": "string", "description": "One line: why you are running it."},
                "screenshot": {"type": "boolean", "description":
                    "Set true only if you need to see the screen after this command. Default false."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "open_app",
        "description": (
            "Launch a GUI application on the desktop, detached, e.g. "
            "'firefox-esr' or 'xterm'. Returns immediately; a screenshot "
            "follows so you can see the app once it is up."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The program (and args) to launch."},
                "intent": {"type": "string", "description": "One line: why you are launching it."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "wait_for_user",
        "description": (
            "Pause and hand control to the human operator when you are blocked "
            "on something only they can do — logging in, solving a captcha, "
            "making a payment, a yes/no decision, plugging in a device. Put "
            "clear instructions in 'message'. The task freezes and the operator "
            "sees your message; it resumes only when they click Resume, after "
            "which you get a fresh screenshot of whatever they changed. The "
            "operator works through a remote view of this same desktop and "
            "CANNOT open apps, terminals, or files — they can only use windows "
            "already on screen. Open the window they need BEFORE calling this, "
            "and spell out in 'message' which window to use and where to click "
            "or type. Use sparingly — prefer doing things yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description":
                    "Exactly what you need the operator to do before you can continue."},
            },
            "required": ["message"],
        },
    },
    {
        "name": "sleep",
        "description": (
            "Wait a long, fixed amount of time (up to 1 hour) with NO model "
            "round-trips while it sleeps — use this instead of polling with the "
            "short 'wait' action when you must idle for minutes: a download, a "
            "build, a countdown, a scheduled event. This is the token-cheap way "
            "to wait: one call sleeps server-side for the whole duration, then "
            "returns a single fresh screenshot. The task deadline is pushed out "
            "so a legitimately long sleep won't time the task out. The operator "
            "can cut the wait short at any time (you'll get 'woken_early': true "
            "in the result, so check it rather than assuming the full time "
            "passed). Do NOT use it as a substitute for wait_for_user (a human "
            "action) or for sub-15s UI settling (use 'wait' for that)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description":
                    "How long to sleep, in seconds (capped at 3600). E.g. 1800 for 30 minutes."},
                "reason": {"type": "string", "description": "One line: what you are waiting for."},
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "wait_for_screen_change",
        "description": (
            "Block until the screen visibly changes from how it looks right now "
            "(or a timeout), with NO model round-trips while waiting. This is the "
            "smart way to wait when you don't know how long something takes but "
            "you'll know it's done because the display updates: a page finishing "
            "loading, a spinner resolving, a dialog appearing, a download that "
            "refreshes the view. It watches the screen server-side and returns "
            "the instant it changes, then hands you one fresh screenshot of the "
            "new state — so you wait exactly as long as needed and no longer. "
            "Prefer this over sleep() whenever the thing you're waiting for will "
            "change the screen. It wakes eagerly on ANY visible change, so the "
            "wake may not be what you wanted (a 'typing...' indicator, a clock "
            "tick, an unrelated chat lighting up): look at the screenshot, act if "
            "it's relevant, and if it isn't just call wait_for_screen_change again "
            "to keep waiting. If it returns 'changed': false it hit the timeout "
            "with no visible change. The operator can also skip the wait. (Use "
            "sleep() when what you're waiting for does NOT alter the display.)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timeout_seconds": {"type": "number", "description":
                    "Give up after this many seconds if nothing changes (capped at 3600). Default 30."},
                "reason": {"type": "string", "description": "One line: what update you are waiting for."},
            },
            "required": [],
        },
    },
]

# Hard cap for a single sleep call, and how far past the sleep we push the
# task deadline so waking up still leaves room to act.
MAX_SLEEP_SECONDS = 3600.0
SLEEP_DEADLINE_MARGIN = 60.0

RUN_COMMAND_OUTPUT_LIMIT = 4000

# After this many *consecutive* blocked/empty responses, stop silently
# retrying and hand control to the operator instead of spinning.
BLOCK_PAUSE_THRESHOLD = 3

# Model key names -> xdotool keysym names (pass-through when unmapped).
_KEYMAP = {
    "enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape",
    "backspace": "BackSpace", "delete": "Delete", "tab": "Tab", "space": "space",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "page_down": "Page_Down", "pagedown": "Page_Down",
    "page_up": "Page_Up", "pageup": "Page_Up",
    "home": "Home", "end": "End", "control": "ctrl", "ctrl": "ctrl",
    "alt": "alt", "shift": "shift", "super": "super", "meta": "super",
}


def _to_xdotool_combo(keys: str | list[str]) -> str:
    parts = keys.split("+") if isinstance(keys, str) else list(keys)
    return "+".join(_KEYMAP.get(p.strip().lower(), p.strip()) for p in parts if p.strip())


async def pause_gate(task: Task, log: RunLog, step: int) -> None:
    """Per-step control gate: honor cancel, then block while pause is requested.

    Pause is cooperative — it only takes effect at step boundaries, and the
    wait still burns the task's wall-clock timeout (unlike sleep(), pausing
    does not push task.deadline out, so a long pause can still time the task
    out).
    """
    if task.cancel_requested:
        raise TaskCancelled()
    if not task.pause_requested:
        return
    task.paused = True
    log.event(step, "paused")
    try:
        while task.pause_requested:
            if task.cancel_requested:
                raise TaskCancelled()
            await asyncio.sleep(0.25)
    finally:
        task.paused = False
    log.event(step, "resumed")


class StubBrain:
    """Screenshot-only brain for smoke-testing the daemon without an API key."""

    def __init__(self, steps: int = 3):
        self.steps = steps

    async def run_task(self, task: Task, sandbox: Sandbox, log: RunLog) -> str:
        for step in range(1, self.steps + 1):
            await pause_gate(task, log, step)
            if task.guidance:
                for text in task.guidance:
                    log.event(step, "guidance", text=text)
                task.guidance.clear()
            png = await sandbox.screenshot()
            path = log.save_screenshot(step, png)
            log.event(step, "screenshot", path=path, bytes=len(png))
            task.steps_taken = step
            await asyncio.sleep(1)
        return f"stub brain took {self.steps} screenshots"


class GeminiBrain:
    def __init__(self, model: str | None = None):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._models = [model] if model else list(MODEL_CANDIDATES)
        self._environment = self._pick_environment()

    @staticmethod
    def _pick_environment():
        for name in ("ENVIRONMENT_DESKTOP", "ENVIRONMENT_BROWSER"):
            if hasattr(types.Environment, name):
                return getattr(types.Environment, name)
        raise RuntimeError("installed google-genai SDK has no computer-use environments")

    @staticmethod
    def _safety_settings() -> list[types.SafetySetting]:
        """Default thresholds nondeterministically block benign computer-use
        turns (BlockedReason.SAFETY on 'play music on spotify');"""
        names = ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                 "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
                 "HARM_CATEGORY_CIVIC_INTEGRITY")
        return [
            types.SafetySetting(category=getattr(types.HarmCategory, name),
                                threshold=types.HarmBlockThreshold.BLOCK_NONE)
            for name in names if hasattr(types.HarmCategory, name)
        ]

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            tools=[
                types.Tool(computer_use=types.ComputerUse(
                    environment=self._environment,
                    enable_prompt_injection_detection=True,
                )),
                types.Tool(function_declarations=_CUSTOM_TOOLS),
            ],
            safety_settings=self._safety_settings(),
        )

    @staticmethod
    def _usage_fields(response) -> dict:
        """Per-call token accounting. Cached and uncached prompt tokens are
        reported separately: Gemini implicitly caches the repeated prefix, so a
        poll costs far less than its raw prompt size suggests, and collapsing
        the two hides that. `uncached_prompt_tokens` is the part a poll actually
        pays for — mostly the new screenshot."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        prompt = getattr(usage, "prompt_token_count", None) or 0
        cached = getattr(usage, "cached_content_token_count", None) or 0
        return {
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "uncached_prompt_tokens": max(prompt - cached, 0),
            "output_tokens": getattr(usage, "candidates_token_count", None) or 0,
            "thoughts_tokens": getattr(usage, "thoughts_token_count", None) or 0,
            "tool_use_prompt_tokens": getattr(usage, "tool_use_prompt_token_count", None) or 0,
            "total_tokens": getattr(usage, "total_token_count", None) or 0,
        }

    async def _generate(self, contents, log: RunLog | None = None, step: int = 0):
        """Call the model, falling back through MODEL_CANDIDATES once.

        Every model call in the system funnels through here, so this is where
        cost is measured: one `model_call` event per round-trip, carrying token
        counts and wall-clock latency. Model *calls* and latency are the metrics
        that caching cannot deflate, so they are recorded alongside tokens."""
        last_error: Exception | None = None
        for model in list(self._models):
            started = time.perf_counter()
            try:
                response = await self.client.aio.models.generate_content(
                    model=model, contents=contents, config=self._config(),
                )
            except Exception as e:
                last_error = e
                message = str(e).lower()
                if not any(hint in message for hint in ("not found", "not supported", "invalid model", "404")):
                    raise
                continue
            self._models = [model]
            if log is not None:
                log.event(step, "model_call", model=model,
                          latency_s=round(time.perf_counter() - started, 3),
                          **self._usage_fields(response))
            return response
        raise last_error  # type: ignore[misc]

    _TEXT_ONLY_ACTIONS = {"run_command"}

    @classmethod
    def _wants_screenshot(cls, name: str, args: dict) -> bool:
        """A GUI action needs a screenshot to show its effect; a shell command
        does not. `take_screenshot`/`wait`/`open_app` stay visual. The model
        can force one after a shell command with screenshot=true."""
        if name in cls._TEXT_ONLY_ACTIONS:
            return bool(args.get("screenshot"))
        return True

    @staticmethod
    async def _pause_for_block(task: Task, log: RunLog, step: int, streak: int) -> None:
        """Google's safety filter keeps blocking the turn: hand control to the
        operator with instructions instead of retrying forever."""
        task.wait_message = (
            f"Google's safety filter blocked my last {streak} turns "
            "(block_reason=SAFETY), so I can't continue on my own. You can: "
            "1) rephrase the task without sensitive wording and send it in the "
            "steer box, 2) do the blocked step yourself on the desktop, then "
            "Resume, or 3) just Resume to let me try again."
        )
        task.pause_requested = True
        log.event(step, "blocked_pause", streak=streak)
        try:
            await pause_gate(task, log, step)  # blocks until Resume; raises on Cancel
        finally:
            task.wait_message = None

    @staticmethod
    async def _wait_for_user(task: Task, args: dict, log: RunLog, step: int) -> dict:
        """Model-initiated pause: surface a message and block until the operator
        resumes (reusing the same pause flag the Resume button flips)."""
        msg = str(args.get("message") or "Waiting for you to continue.").strip()
        task.wait_message = msg
        task.wait_kind = "user"
        task.pause_requested = True
        log.event(step, "wait_for_user", message=msg)
        try:
            await pause_gate(task, log, step)  # blocks until Resume; raises on Cancel
        finally:
            task.wait_message = None
            task.wait_kind = None
        return {"resumed": True, "note": "Operator resumed; screen reflects their changes."}

    @staticmethod
    def _open_wait(task: Task, budget: float, message: str) -> None:
        """Start an operator-skippable timed wait: push the mutable deadline out
        so a long wait can't trip the daemon timeout, and raise a wait box (the
        'sleep' kind renders a Skip button). Clearing wake_requested first drops
        any stale skip from an earlier resume so the wait doesn't end instantly."""
        if task.deadline is not None:
            task.deadline = max(task.deadline,
                                time.monotonic() + budget + SLEEP_DEADLINE_MARGIN)
        task.timeout_seconds += budget
        task.wake_requested = False
        task.wait_kind = "sleep"
        task.wait_message = message

    @staticmethod
    def _close_wait(task: Task) -> None:
        task.wait_message = None
        task.wait_kind = None
        task.wake_requested = False

    @staticmethod
    async def _sleep(task: Task, args: dict, log: RunLog, step: int) -> dict:
        """Model-initiated timed wait: sleep server-side for up to an hour with
        no model round-trips. The operator can end it early by clicking Resume
        (wake_requested); also cancellable at 1s granularity."""
        seconds = max(0.0, min(float(args.get("seconds", 60)), MAX_SLEEP_SECONDS))
        reason = str(args.get("reason") or "").strip()
        mins = f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f}s"
        GeminiBrain._open_wait(task, seconds, f"Sleeping ~{mins}"
                               + (f" ({reason})" if reason else "")
                               + ". Click Resume to skip the wait and continue now.")
        log.event(step, "sleep", seconds=seconds, reason=reason or None)
        elapsed = 0.0
        try:
            while elapsed < seconds:
                if task.cancel_requested:
                    raise TaskCancelled()
                if task.wake_requested:
                    log.event(step, "sleep_woken", after=round(elapsed, 1))
                    break
                await asyncio.sleep(min(seconds - elapsed, 1.0))
                elapsed += 1.0
        finally:
            GeminiBrain._close_wait(task)
        return {"slept_seconds": round(min(elapsed, seconds), 1),
                "woken_early": elapsed < seconds}

    # Screen-change detection: compare grayscale thumbnails. A pixel counts as
    # changed if it shifts more than _CHANGE_PIXEL_DELTA (of 255); the frame
    # counts as changed if more than _CHANGE_FRACTION of pixels do. Tuned to be
    # eager — a single new chat bubble should trip it — while still ignoring a
    # blinking caret or mouse cursor (a few pixels). The model decides whether a
    # wake actually matters and can just wait again if it doesn't.
    _CHANGE_PIXEL_DELTA = 14
    _CHANGE_FRACTION = 0.004

    @staticmethod
    def _signature(png: bytes) -> Image.Image:
        return Image.open(io.BytesIO(png)).convert("L").resize((240, 152))

    @classmethod
    def _frames_differ(cls, a: Image.Image, b: Image.Image) -> bool:
        hist = ImageChops.difference(a, b).histogram()
        changed = sum(hist[cls._CHANGE_PIXEL_DELTA + 1:])
        return changed > cls._CHANGE_FRACTION * (a.width * a.height)

    @staticmethod
    async def _wait_for_change(task: Task, args: dict, sandbox: Sandbox,
                               log: RunLog, step: int) -> dict:
        """Block until the screen visibly changes from now (or a timeout), polling
        server-side with no model round-trips. Returns as soon as a change is
        seen; the batch's trailing screenshot then shows the model the new state.
        Operator-skippable and cancellable."""
        timeout = max(1.0, min(float(args.get("timeout_seconds", 30)), MAX_SLEEP_SECONDS))
        reason = str(args.get("reason") or "").strip()
        GeminiBrain._open_wait(task, timeout, f"Waiting up to {timeout:.0f}s for the "
                               "screen to change" + (f" ({reason})" if reason else "")
                               + ". Click Resume to continue now.")
        baseline = GeminiBrain._signature(await sandbox.screenshot())
        log.event(step, "wait_for_change", timeout=timeout, reason=reason or None)
        elapsed = 0.0
        changed = woken = False
        try:
            while elapsed < timeout:
                if task.cancel_requested:
                    raise TaskCancelled()
                if task.wake_requested:
                    woken = True
                    log.event(step, "wait_woken", after=round(elapsed, 1))
                    break
                await asyncio.sleep(min(timeout - elapsed, 1.0))
                elapsed += 1.0
                if GeminiBrain._frames_differ(
                        baseline, GeminiBrain._signature(await sandbox.screenshot())):
                    changed = True
                    log.event(step, "screen_changed", after=round(elapsed, 1))
                    break
        finally:
            GeminiBrain._close_wait(task)
        return {"changed": changed, "waited_seconds": round(min(elapsed, timeout), 1),
                "timed_out": not changed and not woken,
                "skipped_by_operator": woken}

    @staticmethod
    def _drain_guidance(task: Task, contents: list[types.Content],
                        log: RunLog, step: int) -> None:
        """Inject pending operator guidance as a user turn before the next call."""
        if not task.guidance:
            return
        notes = list(task.guidance)
        task.guidance.clear()
        for text in notes:
            log.event(step, "guidance", text=text)
        contents.append(types.Content(role="user", parts=[types.Part(text=(
            "Operator guidance (incorporate into your next actions):\n"
            + "\n".join(notes)
        ))]))

    async def run_task(self, task: Task, sandbox: Sandbox, log: RunLog) -> str:
        png = await self._settled_screenshot(sandbox, delay=0)
        log.save_screenshot(0, png)
        # A follow-up run resumes the saved conversation; a fresh task starts one.
        hint = SYSTEM_HINT if task.history is None else CONTINUE_HINT
        contents: list[types.Content] = task.history or []
        contents.append(types.Content(role="user", parts=[
            types.Part(text=hint.format(w=sandbox.width, h=sandbox.height,
                                        goal=task.goal, max_steps=task.max_steps)),
            types.Part.from_bytes(data=png, mime_type="image/png"),
        ]))
        try:
            return await self._loop(task, sandbox, log, contents)
        finally:
            # Keep the conversation (even after cancel/timeout) so the
            # operator can continue the task from where it stopped.
            task.history = contents

    async def _loop(self, task: Task, sandbox: Sandbox, log: RunLog,
                    contents: list[types.Content]) -> str:
        blocked_streak = 0
        for step in range(1, task.max_steps + 1):
            await pause_gate(task, log, step)
            self._drain_guidance(task, contents, log, step)
            task.steps_taken = step

            response = await self._generate(contents, log, step)
            candidate = response.candidates[0] if response.candidates else None
            if candidate is None or candidate.content is None:
                # Filtered/empty responses happen; nudge instead of crashing.
                detail = str(getattr(response, "prompt_feedback", None) or "no detail")
                log.event(step, "empty_response", detail=detail)
                blocked_streak += 1
                if blocked_streak >= BLOCK_PAUSE_THRESHOLD:
                    # Google keeps blocking us — stop spinning, ask the human.
                    await self._pause_for_block(task, log, step, blocked_streak)
                    blocked_streak = 0
                contents.append(types.Content(role="user", parts=[types.Part(text=(
                    "(Your previous turn came back empty — possibly filtered. "
                    "Continue the task, taking a different approach if the last "
                    "action was the trigger.)"))]))
                continue
            blocked_streak = 0
            contents.append(candidate.content)

            calls = [p.function_call for p in (candidate.content.parts or []) if p.function_call]
            if not calls:
                text = "".join(p.text or "" for p in (candidate.content.parts or [])).strip()
                log.event(step, "done", result=text)
                return text or "(model finished without a text answer)"

            # The model may batch several actions in one turn (see SYSTEM_HINT).
            # Run them in order, but capture just ONE screenshot for the whole
            # batch — the intermediate states are the model's own to predict, and
            # N screenshots per turn would undo the point of batching.
            response_parts: list[types.Part] = []
            need_screenshot = False
            for fc in calls:
                args = dict(fc.args or {})
                log.event(step, "action", name=fc.name, args=args)

                if fc.name in ("wait_for_user", "sleep", "wait_for_screen_change"):
                    # Block here (until Resume, a timer, or a screen change); the
                    # trailing screenshot below shows whatever changed meanwhile.
                    if fc.name == "wait_for_user":
                        payload: dict = await self._wait_for_user(task, args, log, step)
                    elif fc.name == "sleep":
                        payload = await self._sleep(task, args, log, step)
                    else:
                        payload = await self._wait_for_change(task, args, sandbox, log, step)
                    need_screenshot = True
                    response_parts.append(types.Part(function_response=types.FunctionResponse(
                        name=fc.name, response=payload)))
                    continue

                acknowledged = self._auto_acknowledge_safety(args, log, step)
                try:
                    payload = await self._execute(fc.name, args, sandbox) or {}
                except Exception as e:
                    payload = {"error": str(e)}
                    log.event(step, "action_error", name=fc.name, error=str(e))
                if acknowledged:
                    payload["safety_acknowledgement"] = "true"
                if "output" in payload:  # surface shell results in the run log/feed
                    log.event(step, "action_result", name=fc.name,
                              exit_code=payload.get("exit_code"),
                              output=payload["output"][:1000])

                # A screenshot is the feedback for GUI actions, but pure noise
                # after a shell command whose answer is text. Defer it to the end
                # of the batch so several actions cost a single frame.
                if self._wants_screenshot(fc.name, args):
                    need_screenshot = True
                response_parts.append(types.Part(
                    function_response=types.FunctionResponse(name=fc.name, response=payload)))

            # One settle + one screenshot for the batch, hung off the last
            # response so the model sees the final state after every action ran.
            if need_screenshot:
                png = await self._settled_screenshot(sandbox)
                log.event(step, "screenshot", path=log.save_screenshot(step, png))
                response_parts[-1].function_response.parts = [types.FunctionResponsePart(
                    inline_data=types.FunctionResponseBlob(mime_type="image/png", data=png))]

            remaining = task.max_steps - step
            response_parts.append(types.Part(text=(
                f"[budget: {remaining} of {task.max_steps} actions remaining"
                + (" — give your final answer now]" if remaining <= 3 else "]")
            )))
            contents.append(types.Content(role="user", parts=response_parts))
            self._trim_screenshots(contents)

        # Budget exhausted: ask for a best-effort answer instead of dropping
        # everything the model has already seen.
        contents.append(types.Content(role="user", parts=[types.Part(text=(
            "You have run out of action budget and cannot take any more actions. "
            "Based on everything you have seen so far, give your best final answer "
            "to the task in plain text now."
        ))]))
        response = await self._generate(contents, log, task.max_steps)
        candidates = response.candidates or []
        parts = (candidates[0].content.parts if candidates and candidates[0].content else None) or []
        text = "".join(p.text or "" for p in parts if p.text).strip()
        log.event(task.max_steps, "budget_synthesis", result=text)
        if text:
            return f"(step budget of {task.max_steps} exhausted; best-effort answer)\n{text}"
        return (f"(step budget of {task.max_steps} exhausted with no answer — "
                "raise max_steps if the task needs more actions)")

    @staticmethod
    def _auto_acknowledge_safety(args: dict, log: RunLog, step: int) -> bool:
        """Auto mode: acknowledge model safety confirmations, but log them."""
        decision = args.pop("safety_decision", None)
        if decision:
            log.event(step, "safety_auto_acknowledged", decision=decision)
            return True
        return False

    async def _execute(self, name: str, args: dict, sandbox: Sandbox) -> dict | None:
        """Dispatch a model action; a returned dict goes back to the model in
        the function response. Covers both the current gemini-3.5 action
        names (click, hotkey, wait, ...), the legacy computer-use-preview
        vocabulary (click_at, key_combination, ...), and our custom tools
        (run_command, open_app)."""
        w, h = sandbox.width, sandbox.height

        def point(kx: str = "x", ky: str = "y", default_center: bool = False) -> tuple[int, int]:
            if default_center and kx not in args:
                return w // 2, h // 2
            return denormalize(args[kx], args[ky], w, h)

        match name:
            case "run_command":
                code, output = await sandbox.exec_shell(str(args.get("command", "")))
                if len(output) > RUN_COMMAND_OUTPUT_LIMIT:
                    output = output[:RUN_COMMAND_OUTPUT_LIMIT] + "\n…(output truncated)"
                return {"exit_code": code, "output": output}
            case "open_app":
                await sandbox.launch(str(args.get("command", "")))
                await asyncio.sleep(3)
            case "open_web_browser":
                await sandbox.launch("firefox-esr")
                await asyncio.sleep(6)
            case "click" | "click_at":
                await sandbox.click(*point())
            case "double_click":
                await sandbox.click(*point(), repeat=2)
            case "triple_click":
                await sandbox.click(*point(), repeat=3)
            case "middle_click":
                await sandbox.click(*point(), button=2)
            case "right_click":
                await sandbox.click(*point(), button=3)
            case "mouse_down":
                await sandbox.mouse_button(*point(), down=True)
            case "mouse_up":
                await sandbox.mouse_button(*point(), down=False)
            case "move" | "hover_at":
                await sandbox.move(*point())
            case "type" | "type_text":
                await sandbox.type_text(args.get("text", ""))
                if args.get("press_enter"):
                    await sandbox.key("Return")
            case "type_text_at":
                await sandbox.click(*point())
                await asyncio.sleep(0.3)
                if args.get("clear_before_typing", True):
                    await sandbox.key("ctrl+a")
                await sandbox.type_text(args.get("text", ""))
                if args.get("press_enter"):
                    await sandbox.key("Return")
            case "press_key":
                await sandbox.key(_to_xdotool_combo(args.get("key") or args.get("keys", "")))
            case "hotkey" | "key_combination":
                await sandbox.key(_to_xdotool_combo(args.get("keys", "")))
            case "key_down":
                await sandbox.key_state(_to_xdotool_combo(args.get("key", "")), down=True)
            case "key_up":
                await sandbox.key_state(_to_xdotool_combo(args.get("key", "")), down=False)
            case "scroll" | "scroll_document" | "scroll_at":
                x, y = point(default_center=True)
                magnitude = int(args.get("magnitude") or args.get("amount") or 300)
                await sandbox.scroll(x, y, args.get("direction", "down"),
                                     max(1, magnitude // 100))
            case "drag_and_drop":
                x, y = point()
                dx, dy = point("destination_x", "destination_y")
                await sandbox.drag(x, y, dx, dy)
            case "take_screenshot":
                pass  # a fresh screenshot is returned after every action anyway
            case "wait" | "wait_5_seconds":
                await asyncio.sleep(min(float(args.get("seconds", 5)), 15))
            case "go_back":
                await sandbox.key("alt+Left")
            case "go_forward":
                await sandbox.key("alt+Right")
            case "navigate":
                await sandbox.key("ctrl+l")
                await asyncio.sleep(0.3)
                await sandbox.type_text(args.get("url", ""))
                await sandbox.key("Return")
            case "search":
                await sandbox.key("ctrl+l")
                await asyncio.sleep(0.3)
                await sandbox.type_text("https://duckduckgo.com")
                await sandbox.key("Return")
            case _:
                raise ValueError(f"unsupported action: {name}")

    async def _settled_screenshot(self, sandbox: Sandbox, delay: float = 1.0) -> bytes:
        """Screenshot after the UI has had a moment to settle; downscale if wide."""
        if delay:
            await asyncio.sleep(delay)
        png = await sandbox.screenshot()
        image = Image.open(io.BytesIO(png))
        if image.width > 1366:
            image = image.resize((1366, round(image.height * 1366 / image.width)))
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            png = buf.getvalue()
        return png

    @staticmethod
    def _trim_screenshots(contents: list[types.Content], keep: int = MAX_SCREENSHOTS_IN_HISTORY) -> None:
        """Blank inline screenshot bytes everywhere except the last `keep`
        occurrences, so history token cost stays bounded."""
        holders = []
        for content in contents:
            for part in content.parts or []:
                if part.function_response and part.function_response.parts:
                    holders.append(part.function_response)
                elif part.inline_data and (part.inline_data.mime_type or "").startswith("image/"):
                    holders.append(part)
        for holder in holders[:-keep] if keep else holders:
            if isinstance(holder, types.FunctionResponse):
                holder.parts = None
                holder.response = {**(holder.response or {}), "screenshot": "elided"}
            else:
                holder.inline_data = None
                holder.text = "(earlier screenshot elided)"
