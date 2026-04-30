#!/bin/bash
# Stop and restart the ppf-contact-solver server process.
#
# Invoked by blender_addon/debug/main.py restart-server. Runs on the
# remote host: kills any running server.py, then relaunches it with
# the same venv-activation pattern the addon uses. Writes stdout/stderr
# to the per-server log and waits for SERVER_READY in progress.log
# before returning so the caller sees a clean exit code only after the
# new server is actually accepting connections.

set -u

PORT="${1:-9090}"
HOST="${HOST:-127.0.0.1}"
REPO_DIR="${REPO_DIR:-$HOME/dev}"
VENV_DIR="${VENV_DIR:-$HOME/.local/share/ppf-cts/venv}"
SERVER_LOG="${SERVER_LOG:-$REPO_DIR/server.log}"
PROGRESS_FILE="${PROGRESS_FILE:-$REPO_DIR/progress.log}"
READY_TIMEOUT="${READY_TIMEOUT:-20}"

cd "$REPO_DIR" || { echo "restart.sh: cannot cd to $REPO_DIR" >&2; exit 2; }

pkill -f "python3 server.py" 2>/dev/null
for _ in 1 2 3 4 5; do
    pgrep -f "python3 server.py" >/dev/null || break
    sleep 0.5
done
pkill -9 -f "python3 server.py" 2>/dev/null || true

# Catch anything still holding the port (pattern-based pkill misses
# processes launched with an absolute interpreter path or unusual args).
# fuser -k sends SIGKILL to the pid(s) bound to the TCP port.
if command -v fuser >/dev/null 2>&1; then
    fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
    # Wait for the kernel to release the socket before bind()ing again.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        fuser -s "$PORT/tcp" 2>/dev/null || break
        sleep 0.3
    done
fi

rm -f "$PROGRESS_FILE"

if [ -d "$VENV_DIR" ]; then
    ACTIVATE="source \"$VENV_DIR/bin/activate\" && "
else
    ACTIVATE=""
fi

nohup bash -c "${ACTIVATE}python3 server.py --host $HOST --port $PORT" \
    > "$SERVER_LOG" 2>&1 &
NEW_PID=$!
disown "$NEW_PID" 2>/dev/null || true

for _ in $(seq 1 "$READY_TIMEOUT"); do
    if [ -f "$PROGRESS_FILE" ] && grep -q "SERVER_READY" "$PROGRESS_FILE"; then
        echo "restart.sh: server ready (pid $NEW_PID, port $PORT)"
        exit 0
    fi
    sleep 1
done

echo "restart.sh: server did not become ready within ${READY_TIMEOUT}s" >&2
echo "--- last 20 lines of $SERVER_LOG ---" >&2
tail -20 "$SERVER_LOG" >&2 2>/dev/null || true
exit 1
