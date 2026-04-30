#!/usr/bin/env bash
# Launch Blender with the addon and auto-start Debug + MCP servers.

BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"

if [ ! -x "$BLENDER" ]; then
    echo "Error: Blender not found at $BLENDER" >&2
    exit 1
fi

DEBUG_PORT="${DEBUG_PORT:-8765}"
MCP_PORT="${MCP_PORT:-9633}"

exec "$BLENDER" --addons bl_ext.user_default.ppf_contact_solver --python-expr "
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
