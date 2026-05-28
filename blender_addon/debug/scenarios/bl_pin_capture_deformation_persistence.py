# File: scenarios/bl_pin_capture_deformation_persistence.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Persistence + load_post reconciliation for Capture Pin Deformation.
#
# Subtests:
#   A. flag_survives_save_reopen: capture, save the .blend, open it
#      again; has_captured_anim is still True and the EMBEDDED_MOVE
#      sentinel is in place.
#   B. flag_reconciles_to_false_when_pc2_missing: between save and
#      reopen, delete the on-disk .pc2 file; the load_post handler
#      reconciles has_captured_anim back to False and drops the
#      orphan EMBEDDED_MOVE sentinel.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def _build_pinned_plane():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "PersistPlane"
    n = len(plane.data.vertices)
    plane.vertex_groups.new(name="AllPin").add(list(range(n)), 1.0, "REPLACE")
    return plane


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    plane = _build_pinned_plane()
    api = dh.api.solver
    cloth = api.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "AllPin")

    root = dh.groups.get_addon_data(bpy.context.scene)
    group = root.object_group_0
    group.pin_vertex_groups_index = 0
    pin_item = group.pin_vertex_groups[0]

    pc2 = __import__(pkg + ".core.pc2", fromlist=[
        "write_pin_anim_pc2", "remove_pin_anim_pc2",
        "has_pin_anim_pc2", "pin_anim_pc2_key", "get_pc2_path",
    ])
    pin_ops = __import__(pkg + ".ui.dynamics.pin_ops",
                        fromlist=["_ensure_embedded_move_op"])

    # Synthetic 3-frame cache (no depsgraph walk needed; the
    # persistence path is the same regardless of the cache content).
    n_pin = len(plane.data.vertices)
    frames = np.zeros((3, n_pin, 3), dtype=np.float32)
    frames[1, :, 0] = 0.1
    frames[2, :, 0] = 0.2
    pc2.write_pin_anim_pc2(plane, "AllPin", frames)
    pin_item.has_captured_anim = True
    pin_ops._ensure_embedded_move_op(pin_item)

    # Save the .blend; migrate_pc2_on_save relocates the cache from
    # tempdir to data/.
    blend_path = dh.save_blend(PROBE_DIR, "pin_capture_persistence.blend")
    dh.log(f"saved {blend_path}")

    # Capture the PC2 path AFTER save: migrate_pc2_on_save has
    # relocated the file into data/<blend_basename>/ so the same key
    # now resolves to a different filesystem location.
    pc2_path_after_save = pc2.get_pc2_path(pc2.pin_anim_pc2_key(plane, "AllPin"))
    dh.log(f"pc2 path after save = {pc2_path_after_save}")

    # ---- A: re-open the file, flag + sentinel survive -------------
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    root_A = dh.groups.get_addon_data(bpy.context.scene)
    group_A = root_A.object_group_0
    pin_item_A = group_A.pin_vertex_groups[0]
    flag_A = bool(pin_item_A.has_captured_anim)
    em_present_A = any(
        op.op_type == "EMBEDDED_MOVE" for op in pin_item_A.operations
    )
    plane_A = next(
        (o for o in bpy.data.objects if o.name == "PersistPlane"), None,
    )
    cache_on_disk_A = bool(
        plane_A is not None and pc2.has_pin_anim_pc2(plane_A, "AllPin")
    )
    dh.record(
        "A_flag_survives_save_reopen",
        flag_A and em_present_A and cache_on_disk_A,
        {
            "has_captured_anim": flag_A,
            "embedded_move_present": em_present_A,
            "pc2_present_on_disk": cache_on_disk_A,
        },
    )

    # ---- B: delete PC2 between save and reopen, reconciler clears
    #         the flag and drops the orphan sentinel.
    if plane_A is not None:
        pc2.remove_pin_anim_pc2(plane_A, "AllPin")
        # remove_pin_anim_pc2 only drops the in-memory entry; the file
        # has already been deleted by its own call. Confirm.
        path_B = pc2.get_pc2_path(pc2.pin_anim_pc2_key(plane_A, "AllPin"))
        file_gone_B = not os.path.exists(path_B)
    else:
        file_gone_B = False

    # Re-open the same file. The load_post reconciler should walk
    # every pin and reset has_captured_anim to False since the cache
    # is no longer on disk. With no manual fcurves either, the
    # EMBEDDED_MOVE sentinel should also be removed.
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    root_B = dh.groups.get_addon_data(bpy.context.scene)
    group_B = root_B.object_group_0
    pin_item_B = group_B.pin_vertex_groups[0]
    flag_B = bool(pin_item_B.has_captured_anim)
    em_present_B = any(
        op.op_type == "EMBEDDED_MOVE" for op in pin_item_B.operations
    )
    dh.record(
        "B_flag_reconciles_to_false_when_pc2_missing",
        file_gone_B
        and flag_B is False
        and em_present_B is False,
        {
            "pc2_file_gone": file_gone_B,
            "has_captured_anim": flag_B,
            "embedded_move_present": em_present_B,
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
