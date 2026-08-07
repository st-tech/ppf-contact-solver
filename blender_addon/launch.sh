#!/usr/bin/env bash
# Launch Blender with the addon and auto-start Debug + MCP servers.

set -u

# Resolve a Blender binary for the running OS. PPF_BLENDER_BIN overrides
# everything, which is also what the test rig honors, so one export drives
# both. The per-OS order differs because the platforms disagree about what
# the canonical install IS: on macOS it is the .app bundle, while on Linux
# it is whatever is on PATH (often a wrapper that resolves DISPLAY or
# selects a GPU launcher), with the unpacked tarball as the fallback.
find_blender() {
    if [ -n "${PPF_BLENDER_BIN:-}" ] && [ -x "${PPF_BLENDER_BIN}" ]; then
        echo "$PPF_BLENDER_BIN"
        return 0
    fi
    case "$(uname -s)" in
    Darwin)
        local bundle="/Applications/Blender.app/Contents/MacOS/Blender"
        [ -x "$bundle" ] && { echo "$bundle"; return 0; }
        command -v blender 2>/dev/null && return 0
        ;;
    Linux)
        command -v blender 2>/dev/null && return 0
        local candidate
        for candidate in $(ls -d /opt/blender-*/blender \
                                 /usr/local/blender-*/blender \
                                 /opt/blender/blender 2>/dev/null | sort -rV); do
            [ -x "$candidate" ] && { echo "$candidate"; return 0; }
        done
        ;;
    esac
    return 1
}

BLENDER="$(find_blender)" || {
    echo "Error: no Blender found for $(uname -s)." >&2
    echo "       Set PPF_BLENDER_BIN, or install Blender (./install-blender.sh)." >&2
    exit 1
}

# This launcher is for a session you sit in front of, so a display is a
# hard requirement rather than something to synthesize: an Xvfb started
# here would host a Blender nobody can see. The rig is the other case and
# starts its own (blender_addon/debug/blender_harness.py:ensure_display).
if [ "$(uname -s)" = Linux ] && [ -z "${DISPLAY:-}" ]; then
    echo "Error: DISPLAY is not set, so Blender has nowhere to open." >&2
    echo "       Connect to the desktop session first, or export DISPLAY." >&2
    exit 1
fi

DEBUG_PORT="${DEBUG_PORT:-8765}"
MCP_PORT="${MCP_PORT:-9633}"

# Window size is left to Blender unless PPF_BLENDER_WINDOW asks otherwise,
# using the same WxH / X,Y,W,H spelling the rig accepts.
WINDOW_ARGS=()
if [ -n "${PPF_BLENDER_WINDOW:-}" ]; then
    case "$PPF_BLENDER_WINDOW" in
    *,*,*,*)
        IFS=, read -r _wx _wy _ww _wh <<<"$PPF_BLENDER_WINDOW"
        WINDOW_ARGS=(--window-geometry "$_wx" "$_wy" "$_ww" "$_wh")
        ;;
    *x*)
        WINDOW_ARGS=(--window-geometry 0 0 "${PPF_BLENDER_WINDOW%x*}" \
                     "${PPF_BLENDER_WINDOW#*x}")
        ;;
    *)
        echo "Error: PPF_BLENDER_WINDOW='$PPF_BLENDER_WINDOW' is not WxH or X,Y,W,H" >&2
        exit 1
        ;;
    esac
fi

exec "$BLENDER" "${WINDOW_ARGS[@]}" \
    --addons bl_ext.user_default.ppf_contact_solver --python-expr "
import bpy, sys

def _start_servers():
    # Resolve the addon's root package by stripping a known leaf module
    # off whatever sys.modules entry contains it. Works for both the
    # extension layout (bl_ext.user_default.ppf_contact_solver.ui.solver)
    # and any legacy single-segment layout.
    try:
        pkg = next(n.removesuffix('.ui.solver') for n in sys.modules
                   if n.endswith('.ui.solver'))
    except StopIteration:
        print('launch.sh: addon not loaded, servers not started')
        return None
    rl = __import__(pkg + '.core.reload_server', fromlist=['start_reload_server'])
    rl.start_reload_server(${DEBUG_PORT})
    mc = __import__(pkg + '.mcp.mcp_server', fromlist=['start_mcp_server'])
    mc.start_mcp_server(${MCP_PORT})
    print(f'Debug server on port ${DEBUG_PORT}, MCP on port ${MCP_PORT}')
    return None

bpy.app.timers.register(_start_servers, first_interval=2.0)
" "$@"
