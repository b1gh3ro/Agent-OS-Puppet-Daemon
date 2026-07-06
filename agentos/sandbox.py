"""Execution sandbox interface and the Docker implementation.

Everything the agent physically does happens through this module, and only
inside the container — the daemon host is never touched. The v1 transport is
`docker exec`; a future in-container HTTP bridge can implement the same
Sandbox protocol without changing any caller.
"""

from __future__ import annotations

import asyncio
import shlex
from typing import Protocol


class SandboxError(RuntimeError):
    pass


class Sandbox(Protocol):
    width: int
    height: int

    async def screenshot(self) -> bytes: ...
    async def click(self, x: int, y: int, button: int = 1, repeat: int = 1) -> None: ...
    async def move(self, x: int, y: int) -> None: ...
    async def type_text(self, text: str) -> None: ...
    async def key(self, combo: str) -> None: ...
    async def scroll(self, x: int, y: int, direction: str, magnitude: int = 3) -> None: ...
    async def drag(self, x: int, y: int, dest_x: int, dest_y: int) -> None: ...
    async def mouse_button(self, x: int, y: int, down: bool, button: int = 1) -> None: ...
    async def key_state(self, key: str, down: bool) -> None: ...
    async def launch(self, command: str) -> None: ...
    async def exec_shell(self, command: str, timeout: float = 60.0) -> tuple[int, str]: ...


class DockerSandbox:
    """Drives a running `agent-sandbox` container via docker exec."""

    def __init__(self, container: str = "agent-sandbox", width: int = 1280, height: int = 800):
        self.container = container
        self.width = width
        self.height = height

    async def _exec(self, *argv: str, binary_stdout: bool = False, check: bool = True) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", self.container, *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if check and proc.returncode != 0:
            cmd = " ".join(argv)
            raise SandboxError(f"sandbox command failed ({cmd}): {stderr.decode(errors='replace').strip()}")
        return stdout if binary_stdout else stdout

    async def screenshot(self) -> bytes:
        # scrot writes to a file inside the container; cat streams it back out.
        png = await self._exec(
            "bash", "-c", "scrot -o /tmp/screen.png && cat /tmp/screen.png",
            binary_stdout=True,
        )
        if not png.startswith(b"\x89PNG"):
            raise SandboxError("screenshot did not return a PNG")
        return png

    async def click(self, x: int, y: int, button: int = 1, repeat: int = 1) -> None:
        await self._exec(
            "xdotool", "mousemove", "--sync", str(x), str(y),
            "click", "--repeat", str(repeat), str(button),
        )

    async def move(self, x: int, y: int) -> None:
        await self._exec("xdotool", "mousemove", "--sync", str(x), str(y))

    async def type_text(self, text: str) -> None:
        await self._exec("xdotool", "type", "--delay", "25", "--", text)

    async def key(self, combo: str) -> None:
        await self._exec("xdotool", "key", "--", combo)

    async def scroll(self, x: int, y: int, direction: str, magnitude: int = 3) -> None:
        wheel = {"up": "4", "down": "5", "left": "6", "right": "7"}.get(direction, "5")
        await self._exec(
            "xdotool", "mousemove", "--sync", str(x), str(y),
            "click", "--repeat", str(max(1, magnitude)), wheel,
        )

    async def drag(self, x: int, y: int, dest_x: int, dest_y: int) -> None:
        await self._exec("xdotool", "mousemove", "--sync", str(x), str(y), "mousedown", "1")
        await self._exec("xdotool", "mousemove", "--sync", str(dest_x), str(dest_y), "mouseup", "1")

    async def mouse_button(self, x: int, y: int, down: bool, button: int = 1) -> None:
        action = "mousedown" if down else "mouseup"
        await self._exec("xdotool", "mousemove", "--sync", str(x), str(y), action, str(button))

    async def key_state(self, key: str, down: bool) -> None:
        action = "keydown" if down else "keyup"
        await self._exec("xdotool", action, "--", key)

    async def launch(self, command: str) -> None:
        """Start a GUI app detached inside the container (e.g. firefox-esr)."""
        await self._exec("bash", "-c", f"nohup {shlex.quote(command)} >/dev/null 2>&1 & disown")

    async def exec_shell(self, command: str, timeout: float = 60.0) -> tuple[int, str]:
        """Run a shell command in the container and return (exit_code, output).

        Unlike _exec this never raises on a nonzero exit — the code and the
        combined stdout/stderr go back to the caller (ultimately the model),
        which can read the error and adapt.
        """
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", self.container, "bash", "-lc", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            proc.kill()
            return 124, f"(command still running after {timeout:.0f}s and was killed; use open_app for GUI programs)"
        return proc.returncode or 0, stdout.decode(errors="replace")


async def ensure_container(
    container: str = "agent-sandbox",
    image: str = "agent-sandbox",
    vnc_port: int = 5900,
    novnc_port: int = 6080,
) -> None:
    """Start the sandbox container if it isn't already running."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "inspect", "-f", "{{.State.Running}}", container,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode == 0 and stdout.strip() == b"true":
        return
    if proc.returncode == 0:  # exists but stopped
        argv = ["docker", "start", container]
    else:
        argv = [
            "docker", "run", "-d", "--name", container,
            "-p", f"127.0.0.1:{vnc_port}:5900",
            "-p", f"127.0.0.1:{novnc_port}:6080",
            # Brakes off: full kernel capabilities so the agent can do anything
            # a root user could (mount, modify sysctl, run nested containers…),
            # and a big /dev/shm so heavy browser tabs don't crash.
            "--privileged",
            "--shm-size=2g",
            image,
        ]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise SandboxError(f"could not start sandbox container: {stderr.decode(errors='replace').strip()}")
    await asyncio.sleep(2)  # give Xvfb/openbox a moment to boot
