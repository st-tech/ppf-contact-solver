# File: __init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from . import (
    cleaning_ops,
    geometry_ref_ops,
    intersection_ops,
    merge_ops,
    shell_intersection_ops,
    snap_ops,
    utility_ops,
)


def register():
    snap_ops.register()
    merge_ops.register()
    geometry_ref_ops.register()
    utility_ops.register()
    cleaning_ops.register()
    intersection_ops.register()
    shell_intersection_ops.register()


def unregister():
    shell_intersection_ops.unregister()
    intersection_ops.unregister()
    cleaning_ops.unregister()
    utility_ops.unregister()
    geometry_ref_ops.unregister()
    merge_ops.unregister()
    snap_ops.unregister()
