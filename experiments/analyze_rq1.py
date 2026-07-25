"""Turn the RQ1 run log into the paper's numbers and Fig. 2.

Reads experiments/results/rq1.jsonl (balanced: only reps present for EVERY cell)
plus each run's steps.jsonl, and emits:
  * per-arm cost/latency/success table (medians),
  * the event-arm noise ladder (model calls vs noise x phi),
  * reaction-latency decomposition (detection vs model round-trip),
  * the free-choice primitive split (the unbiased replacement for 359:7),
  * paper/fig2.pdf  — cost-latency + degradation, single IEEE column.

Run:  .venv-wsl/bin/python -m experiments.analyze_rq1
"""
from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("experiments/results")
RUNS = OUT / "runs"
ARMS = ["poll", "sleep", "event", "free"]
TASKS = ["dialog", "terminal", "download", "noise-0.26x", "noise-0.58x",
         "noise-1.54x", "noise-7.87x", "control"]
NOISE_X = {"dialog": 0.0, "terminal": 0.0, "download": 0.0, "noise-0.26x": 0.26,
           "noise-0.58x": 0.58, "noise-1.54x": 1.54, "noise-7.87x": 7.87, "control": 0.0}


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def load_balanced():
    rows = [json.loads(l) for l in (OUT / "rq1.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "arm" in r and "task" in r and r.get("model_calls") is not None]
    # keep the largest R such that every (arm,task) has all reps 0..R-1
    per = defaultdict(set)
    for r in rows:
        per[(r["arm"], r["task"])].add(r["repeat"])
    R = 0
    while all(R in per[(a, t)] for a in ARMS for t in TASKS):
        R += 1
    rows = [r for r in rows if r["repeat"] < R]
    return rows, R


def steps(run_id):
    p = RUNS / run_id / "steps.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def decompose(run_id):
    """(detection_s, model_s, total_s) for a run: event->change, change->act."""
    ev = det = act = None
    for e in steps(run_id):
        m, k = e.get("mono"), e.get("kind")
        if k == "event_fired":
            ev = m
        elif k == "screen_changed" and ev is not None and det is None:
            det = m
        elif k in ("action", "done") and ev is not None and m is not None and m >= ev and act is None:
            act = m
    if ev is None or act is None:
        return None
    detection = (det - ev) if det is not None else None
    return detection, (act - (det if det is not None else ev)), act - ev


def main():
    rows, R = load_balanced()
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)

    print(f"=== RQ1 balanced n={R}  ({len(rows)} runs) ===\n")
    print(f"{'arm':<7}{'calls_med':>10}{'react_med':>11}{'success':>9}")
    for a in ARMS:
        rs = by[a]
        print(f"{a:<7}{med([r['model_calls'] for r in rs]):>10.0f}"
              f"{med([r.get('reaction_latency_s') for r in rs]):>10.1f}s"
              f"{100*sum(bool(r.get('success')) for r in rs)/len(rs):>8.0f}%")

    # event ladder
    ladder = {}
    for t in TASKS:
        rs = [r for r in by["event"] if r["task"] == t]
        ladder[t] = med([r["model_calls"] for r in rs])
    print("\nevent ladder (calls_med by noise x phi):")
    for t in TASKS:
        print(f"  {t:<12} {NOISE_X[t]:>5.2f}x -> {ladder[t]:.0f}")

    # latency decomposition on the event arm (quiescent scenes: clean detection)
    det, mod, tot = [], [], []
    for r in by["event"]:
        if r["regime"] == "control":
            continue
        d = decompose(f"event-{r['task']}-r{r['repeat']}")
        if d and d[0] is not None:
            det.append(d[0]); mod.append(d[1]); tot.append(d[2])
    print(f"\nevent reaction decomposition (n={len(tot)}):")
    print(f"  detection (event->screen_changed): med {med(det):.1f}s")
    print(f"  model+capture (change->action):    med {med(mod):.1f}s")
    print(f"  end-to-end:                        med {med(tot):.1f}s")

    # free-choice split
    pick = Counter()
    for r in by["free"]:
        for e in steps(f"free-{r['task']}-r{r['repeat']}"):
            if e.get("kind") == "action" and e.get("name") in ("wait_for_screen_change", "sleep"):
                pick[e["name"]] += 1
    tot_pick = sum(pick.values()) or 1
    print("\nfree-choice primitive split (unbiased prompt):")
    for k in ("wait_for_screen_change", "sleep"):
        print(f"  {k:<24} {pick[k]:>3}  ({100*pick[k]/tot_pick:.0f}%)")

    # ---------------- Fig. 2 ----------------
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.dpi": 200,
    })
    INK, ACC, QUIET = "#1c2b30", "#b4740f", "#2f7c79"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.15))

    # (a) cost-latency on quiescent scenes
    mk = {"poll": ("o", INK), "sleep": ("s", "#6b6b6b"),
          "event": ("D", ACC), "free": ("^", QUIET)}
    for a in ARMS:
        rs = [r for r in by[a] if r["regime"] == "quiescent"]
        x = med([r.get("reaction_latency_s") for r in rs])
        y = med([r["model_calls"] for r in rs])
        m, c = mk[a]
        ax1.scatter([x], [y], marker=m, s=52, color=c, zorder=3,
                    edgecolor="white", linewidth=0.6, label=a)
        ax1.annotate(a, (x, y), textcoords="offset points", xytext=(7, 4),
                     fontsize=7.5, color=c)
    ax1.set_xlabel("reaction latency (s, median)")
    ax1.set_ylabel("model calls (median)")
    ax1.set_title("(a) cost vs latency, quiescent screens", fontsize=8, loc="left")
    ax1.margins(0.22)
    ax1.grid(True, lw=0.3, alpha=0.4)
    ax1.text(0.03, 0.06, "cheap + fast", transform=ax1.transAxes, fontsize=6.8,
             style="italic", color=QUIET)

    # (b) event-arm degradation across the noise ladder
    xs = [NOISE_X[t] for t in TASKS if NOISE_X[t] > 0]
    ys = [ladder[t] for t in TASKS if NOISE_X[t] > 0]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
    poll_calls = med([r["model_calls"] for r in by["poll"] if r["regime"] == "quiescent"])
    ax2.axhline(poll_calls, ls=":", lw=1, color=INK, alpha=0.7)
    ax2.text(xs[0], poll_calls + 0.12, "polling baseline", fontsize=6.6, color=INK)
    ax2.axvline(1.0, ls="--", lw=0.9, color="#999")
    ax2.text(1.06, min(ys) + 0.1, r"wake threshold $\phi$", fontsize=6.6,
             rotation=90, va="bottom", color="#777")
    ax2.plot(xs, ys, "-D", color=ACC, ms=5, lw=1.3, mec="white", mew=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"background noise ($\times\,\phi$, log scale)")
    ax2.set_ylabel("event-driven calls (median)")
    ax2.set_title("(b) event-driven waiting degrades with noise", fontsize=8, loc="left")
    ax2.grid(True, lw=0.3, alpha=0.4, which="both")

    fig.tight_layout(pad=0.4)
    dest = Path("paper/fig2.pdf")
    fig.savefig(dest, bbox_inches="tight")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
