"""The waiting-task suite for the cost/latency experiment (paper §V).

Each task puts the agent in front of a screen where exactly one meaningful thing
happens, at a moment the *driver* chooses. The driver — not the sandbox, not the
agent — fires the event and records when it did, on the same monotonic clock the
run log uses. Reaction latency is then a subtraction rather than an inference.

Two properties of these tasks are load-bearing:

1. `goal` never mentions waiting, tools, cost, or efficiency. Which primitive the
   agent may use is controlled by ablating its toolbox (GeminiBrain's
   `waiting_tools`), never by asking. A goal that hints at a strategy measures
   the hint.

2. Every event reveals a `codeword` the agent must echo back. Without it, an
   agent that never looked at the screen and simply reported "done" would be
   scored as a success.

Scenes are not split into "quiet" and "noisy" by eye. Whether a scene is noisy is
only meaningful relative to the detector's threshold phi, and intuition is a poor
guide: a ticking clock and a small spinner both *look* busy yet change only
0.26x and 0.58x phi per second, so they never wake an event-driven wait at all.
Choosing them as the "noisy" condition would have produced a false negative --
no degradation, because no false wakes.

So `noise` here is a measured ladder spanning phi, from 0.26x to 7.87x (see
experiments/probe_noise.py, which produced these figures against the production
detector). This turns the strategy comparison into a dose-response curve: the
cost model predicts event-driven waiting is O(1) in model calls while noise sits
below phi, and degrades to O(rT) above it. The ladder measures where that
transition actually happens rather than asserting it.

The event itself changes 28.9x phi -- far above every noise level -- so it is
unambiguously detectable in every scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: When the event fires, in seconds after the agent starts.
EVENT_AT = 90.0

#: How long a run may last before the driver gives up, in seconds.
RUN_TIMEOUT = 300.0


@dataclass(frozen=True)
class WaitTask:
    name: str
    regime: str                 # "quiescent" | "noisy" | "control"
    goal: str                   # verbatim prompt; must not hint at a strategy
    setup: tuple[str, ...]      # shell, run in the sandbox before the agent starts
    event: tuple[str, ...]      # shell, run by the driver at EVENT_AT
    codeword: str | None        # must appear in the agent's final answer
    #: Measured background change per second, as a multiple of _CHANGE_FRACTION.
    #: Below 1.0 the detector ignores this scene; above it, false wakes accrue.
    noise_x_phi: float = 0.0
    teardown: tuple[str, ...] = field(default=(
        "pkill -f 'xmessage|xclock|xterm|feh' 2>/dev/null || true",
        "rm -f /tmp/done",
    ))
    event_at: float = EVENT_AT


def _banner(text: str, geometry: str = "600x200+340+300") -> str:
    """A plain window showing `text`. xmessage is unstyled, static, and has no
    animation of its own — so any pixel change it causes is the event itself.
    Measured at 28.9x phi: unambiguous against every scene in the ladder."""
    return f'DISPLAY=:99 xmessage -center -geometry {geometry} {text} &'


#: One prompt for every ladder scene, so the only variable across them is the
#: background noise. Says nothing about how to wait: the available primitive is
#: controlled by ablating the toolbox, never by asking (see ARMS).
_WATCH_GOAL = (
    "A notification window will appear on this desktop within the next few "
    "minutes. Do not do anything else in the meantime. As soon as it appears, "
    "report the exact text it displays."
)

# Background-animation scripts, written to the sandbox during setup. Heredocs
# rather than inline quoting: these strings pass through Python, docker exec and
# bash, and nested quotes silently mangle into a scene that renders nothing —
# which reads in the results as "no noise" rather than as a failure.
_SPIN_SH = r"""cat > /tmp/spin.sh <<'EOS'
i=0
while true; do
  case $((i % 4)) in 0) c='|';; 1) c='/';; 2) c='-';; 3) c='\\';; esac
  printf '\r  Working %s  ' "$c"
  i=$((i+1)); sleep 0.25
done
EOS"""

_LOG_SH = r"""cat > /tmp/log.sh <<'EOS'
i=0
while true; do
  echo "line $i  $(date +%s.%N)  processing batch $((i * 7 % 991))"
  i=$((i+1)); sleep 0.2
done
EOS"""


TASKS: tuple[WaitTask, ...] = (

    # ---- quiescent: screen is still until the event ----------------------

    WaitTask(
        name="dialog",
        regime="quiescent",
        goal=("A notification window will appear on this desktop within the next "
              "few minutes. Do not do anything else in the meantime. As soon as it "
              "appears, report the exact text it displays."),
        setup=(),
        event=(_banner("DONE-7Q2"),),
        codeword="DONE-7Q2",
    ),

    WaitTask(
        name="terminal",
        regime="quiescent",
        # The xterm spins on a file that only the driver creates, so the moment
        # the completion line appears is the moment the driver chose (plus <0.5s
        # of poll granularity, which is identical across every arm).
        goal=("An xterm on this desktop is running a build. Do not do anything "
              "else. When the build prints its completion message, report that "
              "message exactly."),
        setup=(
            "rm -f /tmp/done",
            "DISPLAY=:99 xterm -T Build -geometry 80x24+100+100 -e "
            "\"bash -c 'echo Building project...; "
            "while [ ! -f /tmp/done ]; do sleep 0.5; done; "
            "echo DONE-K4M; sleep 99999'\" &",
        ),
        event=("touch /tmp/done",),
        codeword="DONE-K4M",
    ),

    WaitTask(
        name="download",
        regime="quiescent",
        # A static "Downloading..." banner is replaced by a completion banner.
        # No progress bar: this is the case where nothing moves until it is done.
        goal=("A file is downloading on this desktop. Do not do anything else. "
              "When the download reports that it has finished, report the exact "
              "completion text shown."),
        setup=(_banner("Downloading...  please wait"),),
        event=(
            "pkill xmessage 2>/dev/null || true",
            _banner("DONE-P8X"),
        ),
        codeword="DONE-P8X",
    ),

    # ---- the noise ladder ------------------------------------------------
    #
    # Same task, same event, same prompt: only the background activity varies.
    # Two scenes sit below phi and two above, bracketing the predicted O(1) ->
    # O(rT) transition. Amplitudes are measured, not assumed; re-run
    # experiments/probe_noise.py after any change to the detector or the screen
    # resolution, since every one of these numbers is relative to both.

    WaitTask(
        name="noise-0.26x",
        regime="quiescent",     # below phi: looks busy, detector ignores it
        goal=_WATCH_GOAL,
        setup=(_SPIN_SH, "DISPLAY=:99 xclock -update 1 -geometry 200x200+900+80 &"),
        event=(_banner("DONE-R3T"),),
        codeword="DONE-R3T",
        noise_x_phi=0.26,
    ),

    WaitTask(
        name="noise-0.58x",
        regime="quiescent",     # below phi
        goal=_WATCH_GOAL,
        setup=(_SPIN_SH,
               "DISPLAY=:99 xterm -T W -fa Monospace -fs 28 -geometry 24x3+300+120 "
               "-e 'bash /tmp/spin.sh' &"),
        event=(_banner("DONE-W9L"),),
        codeword="DONE-W9L",
        noise_x_phi=0.58,
    ),

    WaitTask(
        name="noise-1.54x",
        regime="noisy",         # above phi: false wakes begin
        goal=_WATCH_GOAL,
        setup=("DISPLAY=:99 xclock -update 1 -geometry 600x600+300+80 &",),
        event=(_banner("DONE-M2V", geometry="600x200+340+560"),),
        codeword="DONE-M2V",
        noise_x_phi=1.54,
    ),

    WaitTask(
        name="noise-7.87x",
        regime="noisy",         # well above phi: the degenerate case
        goal=_WATCH_GOAL,
        setup=(_LOG_SH,
               "DISPLAY=:99 xterm -T L -fa Monospace -fs 18 -geometry 100x40+80+60 "
               "-e 'bash /tmp/log.sh' &"),
        event=(_banner("DONE-J5H", geometry="600x200+600+520"),),
        codeword="DONE-J5H",
        noise_x_phi=7.87,
    ),

    # ---- control: the event never comes ----------------------------------

    WaitTask(
        name="control",
        regime="control",
        # Nothing ever happens. Measures what each strategy costs when the thing
        # being waited for does not arrive — the case where polling burns calls
        # for the entire duration and event-driven waiting should burn one.
        # There is no codeword: the correct outcome is a reported timeout, and
        # any run claiming to have seen a notification has hallucinated it.
        goal=("A notification window may appear on this desktop. Do not do "
              "anything else. Watch for up to two minutes. If it appears, report "
              "its exact text; if nothing appears in that time, report exactly: "
              "NOTHING APPEARED."),
        setup=(),
        event=(),
        codeword=None,
        event_at=0.0,
    ),
)

#: The three ways of waiting under comparison, as `waiting_tools` arguments to
#: GeminiBrain. Each arm leaves the model exactly one way to express a wait, so
#: the comparison is between architectures rather than between promptings.
ARMS: dict[str, tuple[str, ...]] = {
    "poll": (),                                # only the built-in short `wait`
    "sleep": ("sleep",),                       # fixed server-side wait
    "event": ("wait_for_screen_change",),      # event-driven wait
}

#: A fourth, separate condition: both primitives exposed, neutrally described,
#: to observe which the model actually chooses. This is the controlled
#: replacement for the deployment's 359:7 split, which only measured a prompt
#: that said "prefer this over sleep()".
FREE_CHOICE: tuple[str, ...] = ("sleep", "wait_for_screen_change")

REPEATS = 5
