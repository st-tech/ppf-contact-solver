#!/bin/bash
# File: .github/workflows/scripts/linux/user-data-build.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# First-boot user data for the Linux release BUILD instance: install Docker,
# let the ssh user run it, and leave a marker the workflow waits for.
#
# The build itself is `container-build.sh` inside almalinux:8.10, exactly as
# it ran on the GitHub-hosted runner; only the machine changed. Installing
# Docker here rather than over ssh lets it overlap the instance's own boot
# and the workflow's other launch work. `usermod -aG` takes effect at the
# user's next login, and every ssh session the workflow opens comes after
# this script, so `docker` runs without sudo, as it did on the runner.
set -eux
# EVERY CI INSTANCE ENDS ITSELF AFTER EIGHT HOURS, AND THIS ONE MAY REBOOT
# FIRST (the driver install reboots it), so the deadline is an absolute time
# in a persistent systemd timer rather than `shutdown -h +480`, which a
# reboot would cancel. Every launch sets --instance-initiated-shutdown-
# behavior terminate, so the shutdown is a termination.
DEADLINE="$(date -u -d '+480 min' '+%Y-%m-%d %H:%M:%S UTC')"
cat > /etc/systemd/system/ci-selfdestruct.service <<UNIT
[Unit]
Description=CI self-destruct: this instance outlived its workflow
[Service]
Type=oneshot
ExecStart=/sbin/shutdown -h now "CI self-destruct: this instance outlived its workflow"
UNIT
cat > /etc/systemd/system/ci-selfdestruct.timer <<UNIT
[Unit]
Description=CI self-destruct at $DEADLINE
[Timer]
OnCalendar=$DEADLINE
Persistent=true
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now ci-selfdestruct.timer
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y docker.io
systemctl enable --now docker
usermod -aG docker ubuntu
touch /home/ubuntu/docker_ready
chown ubuntu:ubuntu /home/ubuntu/docker_ready
