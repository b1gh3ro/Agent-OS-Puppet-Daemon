"""Measure how much of the screen each candidate scene changes per second.

The task suite splits scenes into "quiescent" and "noisy", but that split is only
meaningful relative to the detector's threshold: a scene whose per-second change
sits below _CHANGE_FRACTION never wakes an event-driven wait, and is therefore
quiescent *in effect* no matter how busy it looks to a human. This probe reports
the measured fraction so scenes can be chosen against real numbers rather than
intuition.

Run:  .venv-wsl/bin/python -m experiments.probe_noise
"""

from __future__ import annotations

import asyncio
import io

from dotenv import load_dotenv
from PIL import ImageChops

from agentos.brain import GeminiBrain
from agentos.sandbox import DockerSandbox

CLEAN = "pkill -f 'xmessage|xclock|xterm|mpv|feh' 2>/dev/null || true"

SPIN_SH = r"""cat > /tmp/spin.sh <<'EOS'
i=0
while true; do
  case $((i % 4)) in 0) c='|';; 1) c='/';; 2) c='-';; 3) c='\\';; esac
  printf '\r  Working %s  ' "$c"
  i=$((i+1)); sleep 0.25
done
EOS"""

# Scenes to characterise: (label, setup shell, settle seconds)
SCENES: list[tuple[str, str, float]] = [
    ("empty desktop", "", 1.0),
    ("xclock, second hand", "DISPLAY=:99 xclock -update 1 -geometry 200x200+900+80 &", 2.0),
    ("xclock, large", "DISPLAY=:99 xclock -update 1 -geometry 600x600+300+80 &", 2.0),
    ("spinner, 28pt", "DISPLAY=:99 xterm -T W -fa Monospace -fs 28 -geometry 24x3+300+120 -e 'bash /tmp/spin.sh' &", 3.0),
    ("spinner, 72pt", "DISPLAY=:99 xterm -T W -fa Monospace -fs 72 -geometry 20x3+200+120 -e 'bash /tmp/spin.sh' &", 3.0),
    ("scrolling log", "DISPLAY=:99 xterm -T L -fa Monospace -fs 18 -geometry 100x40+80+60 -e 'bash /tmp/log.sh' &", 3.0),
]

LOG_SH = r"""cat > /tmp/log.sh <<'EOS'
i=0
while true; do
  echo "line $i  $(date +%s.%N)  processing batch $((i * 7 % 991))"
  i=$((i+1)); sleep 0.2
done
EOS"""


async def change_fraction(sb: DockerSandbox, gap: float = 1.0) -> float:
    """Fraction of thumbnail pixels that change over `gap` seconds, using the
    production signature/threshold pipeline."""
    a = GeminiBrain._signature(await sb.screenshot())
    await asyncio.sleep(gap)
    b = GeminiBrain._signature(await sb.screenshot())
    hist = ImageChops.difference(a, b).histogram()
    changed = sum(hist[GeminiBrain._CHANGE_PIXEL_DELTA + 1:])
    return changed / (a.width * a.height)


async def main() -> None:
    load_dotenv()
    sb = DockerSandbox()
    phi = GeminiBrain._CHANGE_FRACTION
    print(f"threshold phi = {phi:.4f}  ({phi*100:.1f}% of pixels)\n")
    print(f"{'scene':<24}{'change/s':>10}{'wakes?':>9}   {'':>6}")
    print("-" * 52)

    await sb.exec_shell(CLEAN)
    await sb.exec_shell(SPIN_SH)
    await sb.exec_shell(LOG_SH)

    for label, setup, settle in SCENES:
        await sb.exec_shell(CLEAN)
        await asyncio.sleep(1.0)
        if setup:
            await sb.exec_shell(setup)
            await asyncio.sleep(settle)
        samples = [await change_fraction(sb) for _ in range(3)]
        f = sum(samples) / len(samples)
        wakes = f > phi
        margin = f / phi if phi else 0
        print(f"{label:<24}{f*100:9.3f}%{str(wakes):>9}   {margin:5.2f}x phi")

    # The event itself must clear the threshold by a wide margin, or the
    # experiment cannot detect the thing it is timing.
    await sb.exec_shell(CLEAN)
    await asyncio.sleep(1.0)
    a = GeminiBrain._signature(await sb.screenshot())
    await sb.exec_shell('DISPLAY=:99 xmessage -center -geometry 600x200+340+300 DONE-7Q2 &')
    await asyncio.sleep(1.5)
    b = GeminiBrain._signature(await sb.screenshot())
    hist = ImageChops.difference(a, b).histogram()
    changed = sum(hist[GeminiBrain._CHANGE_PIXEL_DELTA + 1:]) / (a.width * a.height)
    print("-" * 52)
    print(f"{'EVENT: banner appears':<24}{changed*100:9.3f}%"
          f"{str(GeminiBrain._frames_differ(a, b)):>9}   {changed/phi:5.2f}x phi")
    await sb.exec_shell(CLEAN)


if __name__ == "__main__":
    asyncio.run(main())
