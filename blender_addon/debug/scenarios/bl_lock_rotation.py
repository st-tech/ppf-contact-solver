# File: scenarios/bl_lock_rotation.py
# License: Apache v2.0
#
# Per-object Lock Rotation encoding round-trip.
#
# Lock Rotation is a PER-OBJECT setting (mirrors Lock Translation in
# bl_lock_translation.py, and the PDRD hinge in bl_pdrd_hinge.py): each
# AssignedObject carries ``lock_rotation_enable`` + ``lock_rotation_axis``,
# and the params encoder emits a per-UUID dict
#   param["group"][i][0]["lock-rotation"][uuid] = [ax, ay, az]
# containing ONLY the lock-enabled bodies, with the axis swapped from
# Blender (Z-up) to solver (Y-up) axes and normalized to unit length.
# This is available on every dynamic type; a STATIC group must never
# carry the key (a static object has no free rotation to constrain),
# enforced by the `active_entries` allowlist rather than a per-type RNA
# restriction (the RNA field exists on every AssignedObject regardless
# of group type). It coexists independently of "lock-translation": an
# object may set either, both, or neither. The panel draws both locks
# in the SAME box as one shared object picker
# (``lock_translation_object_selection``) and one shared eye toggle
# (``preview_lock_translation``); there is no separate
# lock_rotation_object_selection or preview_lock_rotation property.
#
# This scenario asserts:
#   A) enabling lock-rotation on a SOLID body encodes its axis, swapped
#      Blender->solver and normalized to unit length;
#   B) a second SOLID body in the same group with lock-rotation disabled
#      is absent from the dict (per-object gating);
#   C) a SAND group also encodes correctly, proving the feature reaches
#      every dynamic type, not just SOLID/PDRD;
#   D) a STATIC group's lock-rotation is pruned by `active_entries` even
#      though the RNA field itself is set (STATIC has no free rotation
#      to lock);
#   E) enabling with a zero axis raises loudly at encode time instead of
#      silently emitting a zero vector or skipping validation;
#   F) the shared `preview_lock_translation` eye toggle builds a
#      rotation-ring/arc overlay only while enabled;
#   G) Lock Translation and Lock Rotation are independent: enabling one
#      on a body does not add it to the other's dict, and both can be
#      enabled on the same body simultaneously;
#   H) `lock_rotation_prohibit_axis` encodes a per-UUID
#      "lock-rotation-prohibit-axis" boolean dict, gated by the same
#      "rotation lock enabled" filter as "lock-rotation": true for a
#      body with the checkbox set, false for a lock-enabled body that
#      left it unchecked (present, not omitted, mirroring how
#      "lock-rotation" itself is gated on enablement rather than on the
#      axis value), absent entirely for a rotation-lock-DISABLED body,
#      and pruned for STATIC groups by the same active_entries
#      allowlist as "lock-rotation".
#   I) both lock overlays use evaluated animation geometry.
#
# Encoding-only: no build / run / fetch (mirrors bl_pdrd_hinge /
# bl_velocity_keyframes / bl_lock_translation). The solver-side
# constraint is future work and out of scope for this repo layer.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import math
import traceback
from mathutils import Vector

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def _resolve(group, obj_name):
    uuid_registry = __import__(pkg + ".core.uuid_registry",
                               fromlist=["resolve_assigned"])
    for assigned in group.assigned_objects:
        uuid_registry.resolve_assigned(assigned)
        if assigned.name == obj_name:
            return assigned
    raise RuntimeError(f"could not resolve assigned object '{obj_name}'")


def _lock_dicts(dh, uuids):
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = dh.decode_addon_blob(param_bytes)
    for params, _objs, object_uuids in decoded["group"]:
        if any(u in object_uuids for u in uuids):
            return (
                params.get("lock-rotation", {}),
                params.get("lock-translation", {}),
                params.get("lock-rotation-prohibit-axis", {}),
            )
    raise RuntimeError("could not locate the group in decoded params")


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(-1.0, 0.0, 0.0))
    cube_a = bpy.context.active_object
    cube_a.name = "RotLockA"
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(1.0, 0.0, 0.0))
    cube_b = bpy.context.active_object
    cube_b.name = "RotLockB"

    root = dh.configure_state(project_name="lock_rotation", frame_count=4)
    solids = dh.api.solver.create_group("Solids", "SOLID")
    solids.add("RotLockA", "RotLockB")
    group = root.object_group_0

    a_assigned = _resolve(group, "RotLockA")
    b_assigned = _resolve(group, "RotLockB")

    # RotLockA is locked to world Y (Blender axis), RotLockB stays free.
    a_assigned.lock_rotation_enable = True
    a_assigned.lock_rotation_axis = (0.0, 3.0, 0.0)
    b_assigned.lock_rotation_enable = False

    uuid_a = a_assigned.uuid
    uuid_b = b_assigned.uuid

    # ----- A: enabled body's axis is swapped Blender->solver, unit ---
    rot_lock, _trans_lock, _prohibit_lock = _lock_dicts(dh, [uuid_a, uuid_b])
    axis_a = rot_lock.get(uuid_a)
    # Blender (x, y, z) -> solver (x, z, -y): (0, 3, 0) -> (0, 0, -3),
    # normalized -> (0, 0, -1).
    a_ok = (
        axis_a is not None
        and math.isclose(axis_a[0], 0.0, abs_tol=1e-9)
        and math.isclose(axis_a[1], 0.0, abs_tol=1e-9)
        and math.isclose(axis_a[2], -1.0, abs_tol=1e-9)
    )
    dh.record(
        "A_enabled_axis_swapped_and_normalized",
        a_ok,
        {"axis_a": list(axis_a) if axis_a is not None else None},
    )

    # ----- B: disabled body is absent from the dict -------------------
    dh.record(
        "B_disabled_body_absent",
        uuid_b not in rot_lock and len(rot_lock) == 1,
        {"lock_keys": list(rot_lock.keys())},
    )

    # ----- C: SAND group also encodes lock-rotation --------------------
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.5, location=(3.0, 0.0, 0.0))
    sand_src = bpy.context.object
    sand_src.name = "RotLockSand"
    sand_ops = __import__(pkg + ".ui.dynamics.sand_ops",
                          fromlist=["build_and_commit_particle_mesh"])
    sand_ops.build_and_commit_particle_mesh(sand_src, 0.08, 0.0, rng_seed=0)
    sand_group_api = dh.api.solver.create_group("Sand", "SAND")
    sand_group_api.add(sand_src.name)
    sand_group = root.object_group_1
    sand_assigned = _resolve(sand_group, "RotLockSand")
    sand_assigned.lock_rotation_enable = True
    sand_assigned.lock_rotation_axis = (1.0, 0.0, 0.0)
    sand_rot_lock, _, _ = _lock_dicts(dh, [sand_assigned.uuid])
    sand_axis = sand_rot_lock.get(sand_assigned.uuid)
    sand_ok = (
        sand_axis is not None
        and math.isclose(sand_axis[0], 1.0, abs_tol=1e-9)
        and math.isclose(sand_axis[1], 0.0, abs_tol=1e-9)
        and math.isclose(sand_axis[2], 0.0, abs_tol=1e-9)
    )
    dh.record(
        "C_sand_group_encodes_lock_rotation",
        sand_ok,
        {"sand_axis": list(sand_axis) if sand_axis is not None else None},
    )

    # ----- D: STATIC group prunes lock-rotation via active_entries -----
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, -1.0, 0.0))
    static_obj = bpy.context.active_object
    static_obj.name = "RotLockGround"
    static_group_api = dh.api.solver.create_group("Ground", "STATIC")
    static_group_api.add("RotLockGround")
    static_group = root.object_group_2
    static_assigned = _resolve(static_group, "RotLockGround")
    # Set the RNA field directly (no UI call site draws it for STATIC);
    # active_entries must still prune it since a STATIC object has no
    # free rotation to lock.
    static_assigned.lock_rotation_enable = True
    static_assigned.lock_rotation_axis = (1.0, 0.0, 0.0)
    static_rot_lock, _, static_prohibit_lock = _lock_dicts(dh, [static_assigned.uuid])
    dh.record(
        "D_static_group_prunes_lock_rotation",
        static_assigned.uuid not in static_rot_lock,
        {"static_lock_keys": list(static_rot_lock.keys())},
    )

    # ----- E: zero axis raises loudly at encode time -------------------
    a_assigned.lock_rotation_axis = (0.0, 0.0, 0.0)
    zero_axis_raised = False
    zero_axis_message = ""
    try:
        dh.encoder_param.encode_param(bpy.context)
    except Exception as exc:  # noqa: BLE001 - we want to see any exception type
        zero_axis_raised = True
        zero_axis_message = str(exc)
    dh.record(
        "E_zero_axis_raises_loudly",
        zero_axis_raised,
        {"message": zero_axis_message},
    )
    # Restore a valid axis so nothing downstream (e.g. a save) trips on it.
    a_assigned.lock_rotation_axis = (0.0, 3.0, 0.0)

    # ----- F: eye toggle builds the selected group's rotation ring ----
    previews = __import__(
        pkg + ".ui.dynamics.overlay_geometry.previews",
        fromlist=["_build_rotation_lock_batches"],
    )
    # There is no separate preview_lock_rotation: the rotation-ring
    # overlay is gated by the same preview_lock_translation eye toggle
    # that drives the translation arrows, since both locks now share
    # one box, one object picker and one eye toggle.
    group.preview_lock_translation = True
    shown_batches = previews._build_rotation_lock_batches(
        bpy.context.scene, 10.0,
    )
    group.preview_lock_translation = False
    hidden_batches = previews._build_rotation_lock_batches(
        bpy.context.scene, 10.0,
    )
    dh.record(
        "F_eye_toggle_controls_rotation_overlay",
        len(shown_batches) == 1 and len(hidden_batches) == 0,
        {
            "shown_batch_count": len(shown_batches),
            "hidden_batch_count": len(hidden_batches),
        },
    )

    # ----- G: Lock Translation and Lock Rotation are independent -------
    a_assigned.lock_translation_enable = True
    a_assigned.lock_translation_axis = (1.0, 0.0, 0.0)
    rot_lock2, trans_lock2, _prohibit_lock2 = _lock_dicts(dh, [uuid_a, uuid_b])
    both_ok = (
        uuid_a in rot_lock2
        and uuid_a in trans_lock2
        and uuid_b not in rot_lock2
        and uuid_b not in trans_lock2
    )
    dh.record(
        "G_lock_translation_and_lock_rotation_independent",
        both_ok,
        {
            "rot_lock_keys": list(rot_lock2.keys()),
            "trans_lock_keys": list(trans_lock2.keys()),
        },
    )

    # ----- H: lock_rotation_prohibit_axis encodes a per-UUID bool ------
    a_assigned.lock_rotation_prohibit_axis = True
    b_assigned.lock_rotation_enable = True
    b_assigned.lock_rotation_axis = (0.0, 0.0, 2.0)
    b_assigned.lock_rotation_prohibit_axis = False
    _rot_lock3, _trans_lock3, prohibit_lock = _lock_dicts(dh, [uuid_a, uuid_b])
    h_ok = (
        prohibit_lock.get(uuid_a) is True
        and prohibit_lock.get(uuid_b) is False
        and static_assigned.uuid not in static_prohibit_lock
    )
    dh.record(
        "H_prohibit_axis_encodes_per_uuid_bool",
        h_ok,
        {
            "prohibit_lock_items": list(prohibit_lock.items()),
            "static_prohibit_lock_keys": list(static_prohibit_lock.keys()),
        },
    )

    # ----- I: overlay geometry follows evaluated animation --------------
    animated_obj = _resolve(group, "RotLockA")
    animated_obj = __import__(
        pkg + ".core.uuid_registry", fromlist=["resolve_assigned"]
    ).resolve_assigned(animated_obj)
    animated_obj.shape_key_add(name="Basis")
    moved_key = animated_obj.shape_key_add(name="OverlayMove")
    for vertex in moved_key.data:
        vertex.co.z += 2.0
    moved_key.value = 0.0
    moved_key.keyframe_insert(data_path="value", frame=1)
    moved_key.value = 1.0
    moved_key.keyframe_insert(data_path="value", frame=2)
    bpy.context.scene.frame_set(1)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    center_1 = sum(
        previews._evaluated_world_vertices(animated_obj, depsgraph),
        Vector((0.0, 0.0, 0.0)),
    ) / len(animated_obj.data.vertices)
    bpy.context.scene.frame_set(2)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    center_2 = sum(
        previews._evaluated_world_vertices(animated_obj, depsgraph),
        Vector((0.0, 0.0, 0.0)),
    ) / len(animated_obj.data.vertices)
    center_shift = tuple(center_2 - center_1)
    dh.record(
        "I_lock_overlays_follow_evaluated_animation",
        abs(center_shift[2] - 2.0) < 1e-4,
        {"center_shift": center_shift},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
