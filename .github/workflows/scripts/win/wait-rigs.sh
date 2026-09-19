#!/usr/bin/env bash
# File: .github/workflows/scripts/win/wait-rigs.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Polls EVERY detached Windows rig of a sharded run for at most MINUTES:
#
#     wait-rigs.sh MINUTES
#
# Needs the rig instance ids in /tmp/rig_ids.txt, one per line, in shard
# order; the key at /tmp/ec2key.pem; and poll-rig.ps1 at C:/poll_rig.ps1 on
# each instance. It is wait-rig.sh over N instances: ONE TUNNEL PER RIG, on
# local port 2222 + shard index, opened at the start of the window the way
# wait-rig.sh opens its one (up to ten bind attempts) and held for the whole
# window; a poll that fails closes that rig's tunnel and the next round opens
# it again. Each round asks every rig that has not reported yet for its poll
# line, prints its newest progress line labeled by shard, and records a
# verdict as RIG_RC_<shard>=<code> in GITHUB_ENV the moment that rig writes
# its exit file. The window ends early once every rig has reported. Later
# windows skip rigs already recorded, by reading /tmp/rig_rc.txt.
#
# A POLL THAT FAILS SAYS WHY. A tunnel opened and closed per round bound in
# five seconds often enough to pass the pipeline test and seldom enough on a
# real run that a window read as "no answer this round" four times over
# (run 35183239004), with ssh's stderr discarded, so a tunnel that had not
# bound, a refused key and a poll script that did not parse all printed the
# same line. Now the tunnel's bind is retried, and a failed ssh prints the
# last line it wrote to stderr.
#
# blender.yml runs it in several windows with an AWS re-authentication
# between them, for the reasons wait-rig.sh records: the credential that
# opens a tunnel lasts an hour, and so does a tunnel.

set -uo pipefail

MINUTES="${1:?usage: wait-rigs.sh MINUTES}"
mapfile -t RIGS < /tmp/rig_ids.txt
RC_FILE=/tmp/rig_rc.txt
touch "$RC_FILE"

reported() { grep -q "^$1=" "$RC_FILE"; }

# One tunnel pid per shard, "" when that shard has no tunnel open.
TUNNEL_PIDS=()
for i in "${!RIGS[@]}"; do TUNNEL_PIDS[$i]=""; done

close_tunnel() {
    # One name per `local`: the expansions of a single `local a=.. b=$a`
    # are performed before either is assigned, so `b` reads an unbound `a`.
    local shard="$1"
    local pid="${TUNNEL_PIDS[$shard]}"
    [ -n "$pid" ] || return 0
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    TUNNEL_PIDS[$shard]=""
}
close_all() { local i; for i in "${!RIGS[@]}"; do close_tunnel "$i"; done; }
trap close_all EXIT

open_tunnel() {
    # open_tunnel SHARD -> 0 once the local port is bound, 1 after ten tries
    local shard="$1"
    local port=$((2222 + shard)) attempt
    [ -n "${TUNNEL_PIDS[$shard]}" ] && return 0
    for attempt in $(seq 1 10); do
        aws ec2-instance-connect open-tunnel --instance-id "${RIGS[$shard]}" \
            --remote-port 22 --local-port "$port" >/dev/null 2>&1 &
        TUNNEL_PIDS[$shard]=$!
        sleep 5
        nc -z localhost "$port" 2>/dev/null && return 0
        close_tunnel "$shard"
        sleep 5
    done
    return 1
}

poll_one() {
    # poll_one SHARD -> prints STATE|code|progress; on failure prints the
    # reason to stderr and returns 1, with that rig's tunnel closed.
    local shard="$1"
    local port=$((2222 + shard)) line err
    if ! open_tunnel "$shard"; then
        echo "tunnel to ${RIGS[$shard]} did not bind on port $port in ten attempts" >&2
        return 1
    fi
    err=$(mktemp)
    line="$(ssh -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=15 -i /tmp/ec2key.pem Administrator@localhost \
        "powershell -ExecutionPolicy Bypass -File C:/poll_rig.ps1" 2>"$err" \
        | tr -d '\r' | grep '^STATE|' | head -1)"
    if [ -z "$line" ]; then
        echo "ssh poll answered no STATE line; stderr: $(tail -n 1 "$err" 2>/dev/null)" >&2
        rm -f "$err"
        close_tunnel "$shard"
        return 1
    fi
    rm -f "$err"
    printf '%s\n' "$line"
}

STARTED=$(date +%s)
END=$(( STARTED + MINUTES * 60 ))
while [ "$(date +%s)" -lt "$END" ]; do
    pending=0
    for i in "${!RIGS[@]}"; do
        reported "$i" && continue
        pending=$((pending + 1))
        # Not `$(poll_one ...)`: a subshell would lose the tunnel pid the
        # poll records, and the next round would open a second tunnel on a
        # port the first still holds.
        if ! poll_one "$i" >/tmp/rig-poll-line.txt 2>/tmp/rig-poll-err.txt; then
            echo "[rig $i +$(( ( $(date +%s) - STARTED ) / 60 ))m] poll failed: $(cat /tmp/rig-poll-err.txt)"
            continue
        fi
        LINE="$(cat /tmp/rig-poll-line.txt)"
        CODE="$(printf '%s' "$LINE" | cut -d'|' -f2)"
        PROGRESS="$(printf '%s' "$LINE" | cut -d'|' -f3-)"
        echo "[rig $i +$(( ( $(date +%s) - STARTED ) / 60 ))m] ${PROGRESS:-no progress line yet}"
        if [ -n "$CODE" ] && [ "$CODE" != "RUNNING" ]; then
            echo "$i=$CODE" >> "$RC_FILE"
            echo "RIG_RC_$i=$CODE" >> "$GITHUB_ENV"
            echo "Rig $i finished with exit code $CODE."
        fi
    done
    if [ "$pending" -eq 0 ] || [ "$(wc -l < "$RC_FILE")" -ge "${#RIGS[@]}" ]; then
        echo "Every rig has reported."
        exit 0
    fi
    sleep 45
done
echo "This window of $MINUTES minutes ended with $(( ${#RIGS[@]} - $(wc -l < "$RC_FILE") )) rig(s) still running."
