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

# The goal rides here too, for the same reason the instructions do. It is also
# in the conversation, but the conversation is finite: `_trim_history` elides
# the middle of a long run, and on a *continued* task the anchor turn it
# protects belongs to the FIRST run, not to the follow-up goal the operator
# typed afterwards. A 48-item queue given as a follow-up therefore scrolled out
# of the model's view entirely, after which it saw a tidy recent tail, judged
# the job complete, and finished with 40 items untouched. Re-sending the goal
# every round-trip makes the scope impossible to forget or trim.
GOAL_PREAMBLE = (
    "THE TASK YOU ARE WORKING ON, in full. This is the authoritative scope — "
    "the conversation gets trimmed on a long run, this does not. Before you "
    "even consider finishing, re-read it and check every part is done:\n\n"
)


def system_instruction(text: str | None, goal: str | None = None) -> str | None:
    """The job's goal and standing instructions, ready to hand to the model as
    ``system_instruction`` — or None when it would be empty, so callers can omit
    the field entirely rather than send an empty system turn."""
    blocks = []
    goal = (goal or "").strip()
    if goal:
        blocks.append(GOAL_PREAMBLE + goal)
    text = (text or "").strip()
    if text:
        blocks.append(PREAMBLE + text)
    return "\n\n---\n\n".join(blocks) if blocks else None
