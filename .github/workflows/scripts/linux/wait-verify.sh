#!/usr/bin/env bash
# File: .github/workflows/scripts/linux/wait-verify.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Polls the detached verification on the instance for at most MINUTES, through
# one tunnel that is reopened whenever a poll comes back empty:
#
#     wait-verify.sh MINUTES
#
# release.yml runs it in several steps with an AWS re-authentication
# between them, because the OIDC credentials that open a tunnel last an hour and
# the verification can take longer. A step ends early once the verification has
# written its exit code, and records it as VERIFY_RC in GITHUB_ENV, which the
# later windows read to skip themselves.
#
# Every poll prints the check in flight and the last line of its log, not only
# a change: the same line repeated is how a wedged run shows itself.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/workflows/scripts/linux/tunnel.sh
. "$HERE/tunnel.sh"
trap close_tunnel EXIT

MINUTES="${1:?usage: wait-verify.sh MINUTES}"
# The defaults are the verification's. All three are environment variables so
# the same wait serves the detached BUILD on the build instance, whose poll is
# `bash poll-build.sh` and whose verdict is BUILD_RC; the instance polled is
# whatever INSTANCE_ID tunnel.sh reads.
POLL="${WAIT_POLL:-bash poll-verify.sh}"
RC_VAR="${WAIT_RC_VAR:-VERIFY_RC}"
LABEL="${WAIT_LABEL:-verify}"
END=$(( $(date +%s) + MINUTES * 60 ))
while [ "$(date +%s)" -lt "$END" ]; do
    if [ -z "$TUNNEL_PID" ] && ! open_tunnel; then
        echo "[wait] the tunnel would not open; retrying"
        sleep 30
        continue
    fi
    LINE="$(remote "$POLL" 2>/dev/null | tr -d '\r' | grep '^STATE|' | head -1)" || LINE=""
    if [ -z "$LINE" ]; then
        echo "[wait] the poll returned nothing; reopening the tunnel"
        close_tunnel
        sleep 10
        continue
    fi
    CODE="$(printf '%s' "$LINE" | cut -d'|' -f2)"
    PROGRESS="$(printf '%s' "$LINE" | cut -d'|' -f3-)"
    echo "[$LABEL $(date -u +%H:%M:%S)] $PROGRESS"
    if [ "$CODE" != "RUNNING" ]; then
        echo "$RC_VAR=$CODE" >> "$GITHUB_ENV"
        echo "The $LABEL finished with exit code $CODE."
        exit 0
    fi
    sleep 60
done
echo "This window of $MINUTES minutes ended with the $LABEL still running."
