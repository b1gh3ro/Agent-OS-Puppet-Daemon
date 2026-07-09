"""Append-only evidence trail: one directory per task run with a JSONL step
log and the screenshot each decision was based on. Pure filesystem, no
dependencies on the rest of the package."""

from __future__ import annotations

import json
import time
from pathlib import Path


class RunLog:
    def __init__(self, task_id: str, root: str | Path = "runs", base: int = 0):
        self.dir = Path(root) / task_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.dir / "steps.jsonl"
        # Follow-up runs of the same task append to the same directory; `base`
        # shifts their step numbers past the earlier runs' so nothing collides.
        self.base = base

    def event(self, step: int, kind: str, **data) -> None:
        record = {"ts": time.time(), "step": self.base + step, "kind": kind, **data}
        with self._jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def save_screenshot(self, step: int, png: bytes) -> str:
        path = self.dir / f"step_{self.base + step:03d}.png"
        path.write_bytes(png)
        return str(path)
