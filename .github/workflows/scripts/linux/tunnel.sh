# File: .github/workflows/scripts/linux/tunnel.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Sourced by release.yml's steps that reach the verification instance.
# Needs INSTANCE_ID and AWS_REGION in the environment and the key at
# /tmp/ec2key.
#
# EVERY CONNECTION IS SHORT. An ec2-instance-connect websocket closes at 60
# minutes whatever it carries, so long work runs detached on the instance and
# each call here lasts seconds.
#
# A TUNNEL IS CONFIRMED BY BINDING, not by a sleep: a failed bind leaves local
# port 2222 held, and the next open would silently reach the wrong thing.

# shellcheck shell=bash

SSH_OPTS=(-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
    -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i /tmp/ec2key)
SCP_OPTS=(-P 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
    -o ConnectTimeout=15 -i /tmp/ec2key)
TUNNEL_PID=""

open_tunnel() {
    local attempt
    for attempt in $(seq 1 10); do
        aws ec2-instance-connect open-tunnel --instance-id "$INSTANCE_ID" \
            --remote-port 22 --local-port 2222 --region "$AWS_REGION" >/dev/null 2>&1 &
        TUNNEL_PID=$!
        sleep 5
        if nc -z localhost 2222 2>/dev/null; then
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

remote() {
    # The command is built by the caller, already quoted for the remote shell.
    # shellcheck disable=SC2029
    ssh "${SSH_OPTS[@]}" ubuntu@localhost "$@"
}

to_remote() {
    # to_remote LOCAL... REMOTE_PATH
    scp "${SCP_OPTS[@]}" "$@"
}

from_remote() {
    # from_remote REMOTE_PATH LOCAL
    scp -r "${SCP_OPTS[@]}" "ubuntu@localhost:$1" "$2"
}

# wait_for_ssh TRIES: a fresh tunnel per try, since an instance that is still
# booting refuses the tunnel's own connection.
wait_for_ssh() {
    local tries="$1" n
    for n in $(seq 1 "$tries"); do
        if open_tunnel && remote true >/dev/null 2>&1; then
            close_tunnel
            printf '[ssh] ready after %s tries\n' "$n"
            return 0
        fi
        close_tunnel
        printf '[ssh] not ready (try %s of %s)\n' "$n" "$tries"
        sleep 20
    done
    return 1
}
