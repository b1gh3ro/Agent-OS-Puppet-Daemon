#!/bin/bash
# Boot the virtual desktop: Xvfb -> openbox -> x11vnc, then idle forever.
set -e

RESOLUTION="${RESOLUTION:-1280x800x24}"

# A container that gets stopped uncleanly (host sleep, Docker restart, or a
# plain `docker start` of a stopped container) leaves the X socket AND lock
# behind; either one makes Xvfb refuse to start on :99, so a restarted sandbox
# comes up with a dead display. Clear both before launching.
rm -f /tmp/.X11-unix/X99 /tmp/.X99-lock

Xvfb :99 -screen 0 "$RESOLUTION" -nolisten tcp &

# Wait for the display socket before starting clients.
for _ in $(seq 1 50); do
    if [ -S /tmp/.X11-unix/X99 ]; then
        break
    fi
    sleep 0.1
done

openbox &
x11vnc -display :99 -forever -shared -nopw -quiet &
# Browser-based live view: http://localhost:6080 proxies to the VNC server.
websockify --web /usr/share/novnc 6080 localhost:5900 &

# Pre-launch the browser so tasks don't burn steps figuring out how to open
# it — and respawn it if it ever dies, so the desktop is never left empty.
# Set NO_BROWSER=1 for a bare desktop (used by the waiting-cost experiments,
# whose scenes require that the only thing that changes is the injected event).
if [ -z "$NO_BROWSER" ]; then
    (while true; do firefox-esr >/dev/null 2>&1; sleep 2; done) &
fi

echo "sandbox ready on :99 (${RESOLUTION}), watch at http://localhost:6080"
tail -f /dev/null
