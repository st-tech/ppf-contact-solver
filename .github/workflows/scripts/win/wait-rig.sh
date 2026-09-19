#!/usr/bin/env bash
# File: .github/workflows/scripts/win/wait-rig.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Polls the detached Windows rig for at most MINUTES, through one tunnel that
# is reopened whenever a poll comes back empty:
#
#     wait-rig.sh MINUTES
#
# Needs the instance id in /tmp/instance_id.txt, the key at /tmp/ec2key.pem and
# poll-rig.ps1 already copied to C:/poll_rig.ps1 on the instance, which is what
# the launch step in blender.yml leaves behind. Those defaults are the rig's;
# every one of them is an environment variable so the same wait serves any
# detached Windows job that answers a `STATE|<code>|<progress>` poll:
#
#     WAIT_INSTANCE_ID_FILE  (/tmp/instance_id.txt)  the instance to tunnel to
#     WAIT_KEY_FILE          (/tmp/ec2key.pem)       its ssh key
#     WAIT_POLL_SCRIPT       (C:/poll_rig.ps1)       the poll on the instance
#     WAIT_LOCAL_PORT        (2222)                  the tunnel's local port
#     WAIT_RC_VAR            (RIG_RC)                the GITHUB_ENV name written
#     WAIT_LABEL             (rig)                   the word in the log lines
#
# release.yml's Windows verification sets all six.
#
# blender.yml runs it in several steps with an AWS re-authentication between
# them, because the OIDC credential that opens a tunnel lasts an hour and the
# rig's real subset runs longer (measured, 204 of 211 scenarios at 60 minutes).
# `aws ec2-instance-connect open-tunnel` authenticates when it opens and runs in
# the background unchecked, so a credential that expired inside one long step
# shows only as the next tunnel never binding, which reads as the instance
# being down. A window is also shorter than the tunnel's own one hour session
# cap, so no tunnel here lives long enough to hit it.
#
# A step ends early once the rig has written its exit code, and records it as
# RIG_RC (or WAIT_RC_VAR) in GITHUB_ENV, which the later windows read to skip
# themselves.
#
# Every poll prints the rig's newest progress line, not only a change: a
# repeated slot number is how a wedged rig shows itself, and an advancing one is
# the only evidence this wait has that the run is alive. poll-rig.ps1 records
# why the slot number is an identifier rather than a monotonic counter.

set -uo pipefail

MINUTES="${1:?usage: wait-rig.sh MINUTES}"
INSTANCE_ID="$(cat "${WAIT_INSTANCE_ID_FILE:-/tmp/instance_id.txt}")"
KEY_FILE="${WAIT_KEY_FILE:-/tmp/ec2key.pem}"
POLL_SCRIPT="${WAIT_POLL_SCRIPT:-C:/poll_rig.ps1}"
LOCAL_PORT="${WAIT_LOCAL_PORT:-2222}"
RC_VAR="${WAIT_RC_VAR:-RIG_RC}"
LABEL="${WAIT_LABEL:-rig}"
SSH_OPTS=(-p "$LOCAL_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
    -o ConnectTimeout=15 -i "$KEY_FILE")
TUNNEL_PID=""

# A tunnel is confirmed by BINDING, not by a sleep: a failed bind leaves local
# port 2222 held and the next open silently reaches the wrong thing.
open_tunnel() {
    local attempt
    for attempt in $(seq 1 10); do
        aws ec2-instance-connect open-tunnel --instance-id "$INSTANCE_ID" \
            --remote-port 22 --local-port "$LOCAL_PORT" &
        TUNNEL_PID=$!
        sleep 5
        if nc -z localhost "$LOCAL_PORT" 2>/dev/null; then
            return 0
        fi
        close_tunnel
        printf '[tunnel] attempt %s did not bind\n' "$attempt"
        sleep 5
    done
    return 1
}

close_tunnel() {
    [ -n "$TUNNEL_PID" ] || return 0
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
    TUNNEL_PID=""
}
trap close_tunnel EXIT

STARTED=$(date +%s)
END=$(( STARTED + MINUTES * 60 ))
while [ "$(date +%s)" -lt "$END" ]; do
    if [ -z "$TUNNEL_PID" ] && ! open_tunnel; then
        echo "[wait] the tunnel would not open; retrying"
        sleep 30
        continue
    fi
    LINE="$(ssh "${SSH_OPTS[@]}" Administrator@localhost \
        "powershell -ExecutionPolicy Bypass -File $POLL_SCRIPT" 2>/dev/null \
        | tr -d '\r' | grep '^STATE|' | head -1)" || LINE=""
    if [ -z "$LINE" ]; then
        echo "[wait] the poll returned nothing; reopening the tunnel"
        close_tunnel
        sleep 10
        continue
    fi
    CODE="$(printf '%s' "$LINE" | cut -d'|' -f2)"
    PROGRESS="$(printf '%s' "$LINE" | cut -d'|' -f3-)"
    echo "[$LABEL +$(( ( $(date +%s) - STARTED ) / 60 ))m] ${PROGRESS:-no progress line yet}"
    if [ "$CODE" != "RUNNING" ]; then
        echo "$RC_VAR=$CODE" >> "$GITHUB_ENV"
        echo "The $LABEL finished with exit code $CODE."
        exit 0
    fi
    sleep 45
done
echo "This window of $MINUTES minutes ended with the $LABEL still running."
