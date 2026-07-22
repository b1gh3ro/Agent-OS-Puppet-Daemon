"""Wrapping for a job's *standing instructions* — one free-text block, specific
to each task, that applies to every step of that job.

Unlike a task goal (which scrolls to the top of a long conversation and drifts
out of the model's attention after a few hundred turns) this block is injected
as the model's ``system_instruction`` on every round-trip, so it is always in
front of the model. The brain reads it fresh from the task on every step, so
editing a running job's instructions permanently steers it from the next step
on. The text itself lives on the Task (and is persisted with it); this module
only decides how to present it to the model.
"""

from __future__ import annotations

# Wraps the operator's text when handed to the model, so the model knows what
# this block is and how much weight it carries relative to the task prompt.
PREAMBLE = (
    "These are the operator's standing instructions for THIS job. They apply to "
    "every step and take precedence over conflicting wording in the task prompt. "
    "The operator may update them mid-run; always follow the latest version:\n\n"
)


def system_instruction(text: str | None) -> str | None:
    """The job's instructions wrapped in PREAMBLE, ready to hand to the model as
    ``system_instruction`` — or None when the job has none, so callers can omit
    the field entirely rather than send an empty system turn."""
    text = (text or "").strip()
    return PREAMBLE + text if text else None
