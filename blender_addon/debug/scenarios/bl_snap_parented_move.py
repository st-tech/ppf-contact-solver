# File: scenarios/bl_snap_parented_move.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Snap repositioning must move a PARENTED object by the intended world
# distance, even when the parent is scaled. _apply_world_translation maps a
# world-space translation into the child's local ``location``; a mesh parented
# to a scaled rig (an armature imported at 0.01, common for game characters)
# carries a compensating ``matrix_parent_inverse`` (100x here), so the mapping
# must invert ``parent.matrix_world @ matrix_parent_inverse``, not the parent
# alone. Omitting matrix_parent_inverse moves the object 100x too far: a ~2 cm
# snap nudge becomes a ~2 m fling, dragging the whole object (and the stitch
# built afterward) to the wrong part of the body.
#
# Blender-only (no solver build / server run).
#
# Subtests:
#   A. parented_scaled_move_is_world_exact: _apply_world_translation on a child
#      of a 0.01-scaled empty moves it by the requested world vector (~cm), not
#      100x that (~m).

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

import bpy
from mathutils import Vector

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    snap_mod = __import__(pkg + ".mesh_ops.snap_ops",
                          fromlist=["_apply_world_translation"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # A rig scaled to 0.01 (game-asset convention) and a mesh parented to it
    # with the matrix_parent_inverse that Ctrl+P assigns, i.e.
    # parent.matrix_world.inverted() == 100x. This is the exact configuration
    # that made a snap fling the cape metres in the batman scene.
    rig = bpy.data.objects.new("Rig", None)
    bpy.context.collection.objects.link(rig)
    rig.scale = (0.01, 0.01, 0.01)
    bpy.context.view_layer.update()

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    child = bpy.context.active_object
    child.name = "Child"
    child.parent = rig
    child.matrix_parent_inverse = rig.matrix_world.inverted()
    bpy.context.view_layer.update()

    mpi_scale = child.matrix_parent_inverse.to_scale().x

    intended = Vector((0.0, 0.0, 0.05))  # 5 cm up
    before = child.matrix_world.translation.copy()
    snap_mod._apply_world_translation(child, intended)
    bpy.context.view_layer.update()
    after = child.matrix_world.translation.copy()
    moved = after - before
    err = (moved - intended).length

    ok = (
        mpi_scale > 50.0        # confirms the 100x parent-inverse is really set
        and err < 1e-4          # moved exactly the intended world distance
        and moved.length < 0.2  # sanity: not a 100x (~5 m) fling
    )
    dh.record(
        "A_parented_scaled_move_is_world_exact",
        ok,
        {
            "mpi_scale": mpi_scale,
            "intended": list(intended),
            "moved": list(moved),
            "world_error": err,
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 120.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
