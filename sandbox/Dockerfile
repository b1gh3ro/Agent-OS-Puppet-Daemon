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
    sudo \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Skip Firefox's first-run wizard/telemetry so tasks start on a clean page.
COPY policies.json /usr/share/firefox-esr/distribution/policies.json

# Xvfb can't create /tmp/.X11-unix when running unprivileged; pre-create it.
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Desktop processes run unprivileged (Firefox misbehaves as root), but the
# agent gets passwordless sudo so it can apt-get install or touch system
# files whenever a task needs it.
RUN useradd -m -s /bin/bash agent \
    && echo "agent ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/agent \
    && chmod 0440 /etc/sudoers.d/agent
USER agent
WORKDIR /home/agent

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
