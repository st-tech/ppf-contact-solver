# File: defaults.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Canonical port defaults shared across the addon.
# Every module that needs these values should import from here
# instead of hardcoding literals.

DEFAULT_SERVER_PORT = 9090
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
