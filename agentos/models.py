"""Task model shared by the daemon, brain, and logs. No I/O here."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class TaskCancelled(Exception):
    """Raised inside a brain loop when task.cancel_requested is set."""


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    goal: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    steps_taken: int = 0
    max_steps: int = 300
    timeout_seconds: float = 3600.0
    deadline: float | None = None  # monotonic wall-clock cutoff; mutable so a long sleep can push it out
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_requested: bool = False
    pause_requested: bool = False  # operator intent, set/cleared over HTTP
    paused: bool = False           # observed state, set/cleared only by the brain
    wake_requested: bool = False   # operator asked to end the current wait/sleep early
    guidance: list[str] = field(default_factory=list)
    wait_message: str | None = None  # set when the agent itself asks the operator to act
    wait_kind: str | None = None     # "user" (needs a human) or "sleep" (a timer) — drives the UI button
    prior_steps: int = 0     # steps consumed by earlier runs (offsets the run log)
    history: list | None = None  # model conversation kept for follow-ups; never serialized

    @property
    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "cancel_requested": self.cancel_requested,
            "pause_requested": self.pause_requested,
            "paused": self.paused,
            "pending_guidance": len(self.guidance),
            "wait_message": self.wait_message,
            "wait_kind": self.wait_kind,
        }
