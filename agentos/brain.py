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

from google import genai
from google.genai import types
from PIL import Image

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
    "anything without the GUI.\n"
    "- open_app(command): launch a GUI program detached, e.g. "
    "open_app(command='firefox-esr') or open_app(command='xterm').\n"
    "Firefox ESR is installed and usually already open; if the screen looks "
    "empty, call open_app(command='firefox-esr') instead of hunting for a "
    "launcher. Complete this task, then answer in plain text with the "
    "outcome:\n\n{goal}"
)

_CUSTOM_TOOLS = [
    {
        "name": "run_command",
        "description": (
            "Run a bash command inside the sandbox and return its exit code and "
            "combined stdout/stderr. Passwordless sudo is available. Use this for "
            "installing packages, downloading files, reading/writing files, and "
            "anything faster done in a shell than by clicking. Do not start GUI "
            "programs with this (they block); use open_app for those."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."},
                "intent": {"type": "string", "description": "One line: why you are running it."},
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
]

RUN_COMMAND_OUTPUT_LIMIT = 4000

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
    wait still burns the task's wall-clock timeout (asyncio.wait_for's
    deadline in the daemon cannot be extended mid-flight).
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
            if task.guidance:  # no model to steer; just log and clear
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

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            tools=[
                types.Tool(computer_use=types.ComputerUse(
                    environment=self._environment,
                    enable_prompt_injection_detection=True,
                )),
                types.Tool(function_declarations=_CUSTOM_TOOLS),
            ],
        )

    async def _generate(self, contents):
        """Call the model, falling back through MODEL_CANDIDATES once."""
        last_error: Exception | None = None
        for model in list(self._models):
            try:
                response = await self.client.aio.models.generate_content(
                    model=model, contents=contents, config=self._config(),
                )
                self._models = [model]  # lock in the working model
                return response
            except Exception as e:  # try next candidate only for model-availability errors
                last_error = e
                message = str(e).lower()
                if not any(hint in message for hint in ("not found", "not supported", "invalid model", "404")):
                    raise
        raise last_error  # type: ignore[misc]

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
        contents: list[types.Content] = [
            types.Content(role="user", parts=[
                types.Part(text=SYSTEM_HINT.format(w=sandbox.width, h=sandbox.height, goal=task.goal)),
                types.Part.from_bytes(data=png, mime_type="image/png"),
            ])
        ]

        for step in range(1, task.max_steps + 1):
            await pause_gate(task, log, step)
            self._drain_guidance(task, contents, log, step)
            task.steps_taken = step

            response = await self._generate(contents)
            candidate = response.candidates[0]
            contents.append(candidate.content)

            calls = [p.function_call for p in (candidate.content.parts or []) if p.function_call]
            if not calls:
                text = "".join(p.text or "" for p in (candidate.content.parts or [])).strip()
                log.event(step, "done", result=text)
                return text or "(model finished without a text answer)"

            response_parts: list[types.Part] = []
            for fc in calls:
                args = dict(fc.args or {})
                log.event(step, "action", name=fc.name, args=args)
                acknowledged = self._auto_acknowledge_safety(args, log, step)
                try:
                    payload: dict = await self._execute(fc.name, args, sandbox) or {}
                except Exception as e:
                    payload = {"error": str(e)}
                    log.event(step, "action_error", name=fc.name, error=str(e))
                if acknowledged:
                    payload["safety_acknowledgement"] = "true"
                if "output" in payload:  # surface shell results in the run log/feed
                    log.event(step, "action_result", name=fc.name,
                              exit_code=payload.get("exit_code"),
                              output=payload["output"][:1000])

                png = await self._settled_screenshot(sandbox)
                path = log.save_screenshot(step, png)
                log.event(step, "screenshot", path=path)
                response_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response=payload,
                        parts=[types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(
                                mime_type="image/png", data=png,
                            )
                        )],
                    )
                ))

            contents.append(types.Content(role="user", parts=response_parts))
            self._trim_screenshots(contents)

        # Budget exhausted: ask for a best-effort answer instead of dropping
        # everything the model has already seen.
        contents.append(types.Content(role="user", parts=[types.Part(text=(
            "You have run out of action budget and cannot take any more actions. "
            "Based on everything you have seen so far, give your best final answer "
            "to the task in plain text now."
        ))]))
        response = await self._generate(contents)
        parts = response.candidates[0].content.parts or []
        text = "".join(p.text or "" for p in parts if p.text).strip()
        log.event(task.max_steps, "budget_synthesis", result=text)
        if text:
            return f"(step budget of {task.max_steps} exhausted; best-effort answer)\n{text}"
        raise RuntimeError(f"step budget of {task.max_steps} exhausted with no answer")

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
