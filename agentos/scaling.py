"""Coordinate mapping between the model's normalized grid and screen pixels.

Gemini computer-use returns coordinates on a 0-1000 grid regardless of the
actual screen size; they must be denormalized to pixels before xdotool sees
them.
"""

from __future__ import annotations

GRID = 1000


def denormalize(x: int | float, y: int | float, width: int, height: int) -> tuple[int, int]:
    """Map a (x, y) point on the 0-1000 model grid to pixel coordinates,
    clamped to the screen bounds."""
    px = round(x / GRID * width)
    py = round(y / GRID * height)
    px = min(max(px, 0), width - 1)
    py = min(max(py, 0), height - 1)
    return px, py
