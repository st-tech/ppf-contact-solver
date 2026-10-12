# File: defaults.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Canonical port defaults shared across the addon.
# Every module that needs these values should import from here
# instead of hardcoding literals.

DEFAULT_SERVER_PORT = 9090
# Seconds a server the add-on starts waits, with no request and no solve or
# build running, before stopping itself (`--idle-timeout`). Short, because a
# connected add-on pings it every DEFAULT_PING_INTERVAL_S from a thread of its
# own, so only a Blender that closed, disconnected or lost the network stops
# pinging. A server started by hand keeps the server's own default of an hour.
DEFAULT_IDLE_STOP_S = 60
DEFAULT_PING_INTERVAL_S = 10
DEFAULT_MCP_PORT = 9633
DEFAULT_RELOAD_PORT = 8765

# SSH transport keepalive interval (seconds). Shared between the facade's
# connect_ssh default and the create_backend fallback so the two cannot
# diverge.
DEFAULT_SSH_KEEPALIVE_INTERVAL = 30

# Scene parameter aliases for backward compatibility. Shared between the
# read path (_SceneProxy.__getattr__) and the write path (the set operator).
SCENE_PARAM_ALIASES = {"gravity": "gravity_3d"}

# Maximum number of collision windows per object. Shared between the MCP
# handler and the UI operator so the two enforcement paths cannot drift.
MAX_COLLISION_WINDOWS = 8
