"""RQ2: characterize the change threshold phi (paper Fig. 3).

The trade-off phi governs is false wakes (noise trips the detector) versus missed
changes (the real signal is too small to trip it). Rather than re-run a scene per
phi, we capture frames ONCE from each scene and replay the production wake loop
offline at every phi:

  * for a noisy scene, simulate wait_for_screen_change over the captured stream
    (baseline resets on each wake, exactly as the runtime does) and count wakes;
  * for the signal, measure how large a change the real event produces, so any
    phi above that amplitude MISSES it.

The result locates the usable band of phi: above every noise scene's amplitude
(no false wakes) yet below the signal's (no misses). Non-AI and self-contained,
so it runs on its own container in parallel with RQ1.

Run:  .venv-wsl/bin/python -m experiments.rq2_sweep --container agent-sandbox-rq2
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import ImageChops

from agentos.brain import GeminiBrain
from agentos.sandbox import DockerSandbox

OUT = Path("experiments/results")
PHI0 = GeminiBrain._CHANGE_FRACTION          # production threshold (0.004)
FRAMES = 34                                  # ~34 s of 1 Hz sampling per scene

CLEAN = "pkill -f 'xmessage|xclock|xterm|feh' 2>/dev/null || true"
SPIN = r"""cat > /tmp/spin.sh <<'EOS'
i=0
while true; do case $((i%4)) in 0) c='|';;1) c='/';;2) c='-';;3) c='\\';; esac
printf '\r  Working %s  ' "$c"; i=$((i+1)); sleep 0.25; done
EOS"""
LOG = r"""cat > /tmp/log.sh <<'EOS'
i=0
while true; do echo "line $i  $(date +%s.%N)  processing batch $((i*7%991))"
i=$((i+1)); sleep 0.2; done
EOS"""

# (label, x*phi from the RQ1 ladder, setup, settle s)
SCENES = [
    ("spinner (0.58x)", 0.58,
     "DISPLAY=:99 xterm -T W -fa Monospace -fs 28 -geometry 24x3+300+120 -e 'bash /tmp/spin.sh' &", 3.0),
    ("clock (1.54x)", 1.54,
     "DISPLAY=:99 xclock -update 1 -geometry 600x600+300+80 &", 2.0),
    ("log (7.87x)", 7.87,
     "DISPLAY=:99 xterm -T L -fa Monospace -fs 18 -geometry 100x40+80+60 -e 'bash /tmp/log.sh' &", 3.0),
]


def frac(a, b) -> float:
    hist = ImageChops.difference(a, b).histogram()
    changed = sum(hist[GeminiBrain._CHANGE_PIXEL_DELTA + 1:])
    return changed / (a.width * a.height)


async def capture_stream(sb, n) -> list:
    sigs = []
    for _ in range(n):
        sigs.append(GeminiBrain._signature(await sb.screenshot()))
        await asyncio.sleep(1.0)
    return sigs


def wakes_at(sigs, phi) -> int:
    """Simulate the runtime wait loop over a captured stream: count wakes, with
    the baseline resetting after each wake, exactly as Algorithm 1 does."""
    base, w = sigs[0], 0
    for f in sigs[1:]:
        if frac(base, f) > phi:
            w += 1
            base = f
    return w


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="agent-sandbox-rq2")
    a = ap.parse_args()
    load_dotenv()
    sb = DockerSandbox(container=a.container)
    OUT.mkdir(parents=True, exist_ok=True)

    await sb.exec_shell(CLEAN)
    await sb.exec_shell(SPIN)
    await sb.exec_shell(LOG)

    # signal amplitude: banner appears on a quiet desktop (the pure event)
    await sb.exec_shell(CLEAN)
    await asyncio.sleep(1.5)
    before = GeminiBrain._signature(await sb.screenshot())
    await sb.exec_shell("DISPLAY=:99 xmessage -center -geometry 600x200+340+300 DONE-7Q2 &")
    await asyncio.sleep(1.5)
    after = GeminiBrain._signature(await sb.screenshot())
    signal = frac(before, after)
    print(f"phi0 = {PHI0:.4f}   signal amplitude = {signal*100:.2f}%  ({signal/PHI0:.1f}x phi0)\n")

    streams = {}
    for label, xphi, setup, settle in SCENES:
        await sb.exec_shell(CLEAN)
        await asyncio.sleep(1.0)
        await sb.exec_shell(setup)
        await asyncio.sleep(settle)
        sigs = await capture_stream(sb, FRAMES)
        streams[label] = sigs
        # reconfirm the ladder amplitude while we have the frames
        amp = sum(frac(sigs[i], sigs[i + 1]) for i in range(len(sigs) - 1)) / (len(sigs) - 1)
        print(f"{label:<18} mean change/frame = {amp*100:.3f}%  ({amp/PHI0:.2f}x phi0)")
    await sb.exec_shell(CLEAN)

    # sweep phi and record false wakes per scene + missed-signal boundary
    phis = [0.0004 * (1.4 ** k) for k in range(20)]     # ~0.0004 .. ~0.24
    rows = []
    for phi in phis:
        row = {"phi": phi, "phi_x": phi / PHI0, "missed_signal": signal <= phi}
        for label, sigs in streams.items():
            row[label] = wakes_at(sigs, phi)
        rows.append(row)
    (OUT / "rq2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    # ---------------- Fig. 3 ----------------
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    colors = {"spinner (0.58x)": "#2f7c79", "clock (1.54x)": "#b4740f", "log (7.87x)": "#8a3b2e"}
    for label in streams:
        ax.plot([r["phi_x"] for r in rows], [r[label] for r in rows], "-o", ms=3, lw=1.2,
                color=colors[label], label=label)
    ax.axvline(1.0, ls="--", lw=0.9, color="#666")
    ax.text(1.05, ax.get_ylim()[1] * 0.9, r"$\phi_0=0.4\%$", fontsize=6.6, color="#444", rotation=90, va="top")
    ax.axvline(signal / PHI0, ls=":", lw=1.1, color="#333")
    ax.text(signal / PHI0 * 1.05, ax.get_ylim()[1] * 0.55, "signal\nmissed →", fontsize=6.6, color="#333")
    ax.set_xscale("log")
    ax.set_xlabel(r"threshold $\phi$ (multiples of $\phi_0$, log scale)")
    ax.set_ylabel(f"false wakes over {FRAMES}s")
    ax.set_title("Fig. 3  false wakes vs. missed changes", fontsize=8, loc="left")
    ax.grid(True, lw=0.3, alpha=0.4, which="both")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(Path("paper/fig3.pdf"), bbox_inches="tight")
    print(f"\nsignal missed once phi > {signal/PHI0:.1f}x phi0; wrote paper/fig3.pdf")


if __name__ == "__main__":
    asyncio.run(main())
