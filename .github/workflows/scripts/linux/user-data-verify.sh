#!/bin/bash
# File: .github/workflows/scripts/linux/user-data-verify.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# First-boot user data for a Linux verification instance, whose whole setup
# the workflow does over ssh. Its one job is the self-destruct below.
#
# The workflows terminate their instances in `if: always()` cleanup steps,
# which run on success, failure and cancellation, but nothing runs if the
# RUNNER dies or GitHub kills a cancelled job before its cleanup finishes,
# and no reaper covers CI instances. Eight hours is past every job's
# timeout-minutes with margin; a healthy run never sees it.
set -eu
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
