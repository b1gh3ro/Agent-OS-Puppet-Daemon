# Debian rather than Ubuntu: 24.04 only ships Firefox as a snap, which
# doesn't work in containers; Debian has firefox-esr as a plain apt package.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    openbox \
    x11vnc \
    xdotool \
    scrot \
    xterm \
    firefox-esr \
    novnc \
    websockify \
    fonts-dejavu-core \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Skip Firefox's first-run wizard/telemetry so tasks start on a clean page.
COPY policies.json /usr/share/firefox-esr/distribution/policies.json

# Xvfb can't create /tmp/.X11-unix when running unprivileged; pre-create it.
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Run the desktop as an unprivileged user; the agent never gets root here.
RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /home/agent

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
