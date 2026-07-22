"""Operator-editable *standing instructions* — one free-text block that applies
to every task and every step.

Unlike a task goal (which scrolls to the top of a long conversation and drifts
out of the model's attention after a few hundred turns) this block is injected
as the model's ``system_instruction`` on every round-trip, so it is always in
front of the model. It is persisted to a plain text file so it outlives daemon
restarts, and it is read fresh on every model call — an edit takes effect on the
very next step, even for a task that is already running.

The store is deliberately tiny: a single global string behind a file. There is
one set of standing instructions for the whole daemon, not one per task.
"""

from __future__ import annotations

import os
from pathlib import Path

# Where the standing instructions live. Overridable for tests / alternate
# deployments; defaults to a file in the working directory (the project root).
_PATH = Path(os.getenv("AGENT_INSTRUCTIONS_FILE", "general_instructions.txt"))

# Wraps the operator's text when handed to the model, so the model knows what
# this block is and how much weight it carries relative to the task prompt.
PREAMBLE = (
    "These are the operator's standing instructions. They apply to EVERY task "
    "and every step, and take precedence over conflicting wording in the task "
    "prompt. Keep following them for the whole session:\n\n"
)


def get_instructions() -> str:
    """The current standing instructions, or '' if none are set."""
    try:
        return _PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def set_instructions(text: str) -> str:
    """Replace the standing instructions. Empty text clears them. Returns the
    stored (stripped) value."""
    text = (text or "").strip()
    if text:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(text, encoding="utf-8")
    elif _PATH.exists():
        _PATH.unlink()
    return text


def system_instruction() -> str | None:
    """The standing instructions wrapped in PREAMBLE, ready to hand to the model
    as ``system_instruction`` — or None when nothing is set, so callers can omit
    the field entirely rather than send an empty system turn."""
    text = get_instructions()
    return PREAMBLE + text if text else None
