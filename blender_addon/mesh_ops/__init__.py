# File: __init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from . import merge_ops, snap_ops


def register():
    snap_ops.register()
    merge_ops.register()


def unregister():
    merge_ops.unregister()
    snap_ops.unregister()
