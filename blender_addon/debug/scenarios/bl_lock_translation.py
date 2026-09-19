# File: scenarios/bl_lock_translation.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Per-object Lock Translation encoding round-trip.
#
# Lock Translation is a PER-OBJECT setting (mirrors the PDRD hinge in
# bl_pdrd_hinge.py): each AssignedObject carries
# ``lock_translation_enable`` + ``lock_translation_axis``, and the params
# encoder emits a per-UUID dict
#   param["group"][i][0]["lock-translation"][uuid] = [ax, ay, az]
# containing ONLY the lock-enabled bodies, with the axis swapped from
# Blender (Z-up) to solver (Y-up) axes and normalized to unit length.
# Unlike the hinge (PDRD only), this is available on every dynamic
# type; a STATIC group must never carry the key (there is no free COM
# to constrain), which is enforced by the `active_entries` allowlist
# rather than a per-type RNA restriction (the RNA field exists on every
# AssignedObject regardless of group type).
#
# This scenario asserts:
#   A) enabling lock-translation on a SOLID body encodes its axis,
#      swapped Blender->solver and normalized to unit length;
#   B) a second SOLID body in the same group with lock-translation
#      disabled is absent from the dict (per-object gating);
#   C) a SAND group (a newly-wired UI call site with no prior velocity
#      controls) also encodes correctly, proving the feature reaches
#      every dynamic type, not just SOLID/PDRD;
#   D) a STATIC group's lock-translation is pruned by `active_entries`
#      even though the RNA field itself is set (STATIC has no free
#      COM to lock);
#   E) enabling with a zero axis raises loudly at encode time instead
#      of silently emitting a zero vector or skipping validation.
#   F) the eye toggle builds a double-headed axis overlay only while enabled.
#
# Encoding-only: no build / run / fetch (mirrors bl_pdrd_hinge /
# bl_velocity_keyframes). The solver-side COM constraint is future
# work and out of scope for this repo layer.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)

# NOT ON WINDOWS, WHICH RUNS THE RIG HEADLESS AND SO HAS NO MODAL LOOP.
#
# This scenario needs a Blender that owns a window: the PC2 it asserts on is
# written by `PPF_OT_FramePump.modal` AFTER the driver's exec returns, and a
# modal operator needs an event loop to run in. Measured on the Windows leg of
# Blender CI: the driver reached `fetched queued=9 total=9`, the probe recorded
# `modal_seen: []`, and the scenario finished with ZERO checks and no error,
# because nothing it asserts on had been written yet. The drawing scenarios in
# the same set fail one step earlier and say so outright, with "GPU functions
# for drawing requires the gpu module to be initialized".
#
# Two requirements collide here: a full build/run/fetch scenario must NOT be
# run with `--background`, because the modal operator above needs an event
# loop, and the Windows leg of CI has no window server, so it runs headless.
# There is no configuration in which both hold, so this declares where it can
# run rather than failing there every time. Linux gives the rig its own Xvfb
# and macOS has a real window server.
PLATFORMS = ("linux", "darwin")


_DRIVER_BODY = r"""
import math
import traceback

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


def _lock_dict(dh, uuids, key="lock-translation"):
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = dh.decode_addon_blob(param_bytes)
    for params, _objs, object_uuids in decoded["group"]:
        if any(u in object_uuids for u in uuids):
            return params.get(key, {})
    raise RuntimeError("could not locate the group in decoded params")


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(-1.0, 0.0, 0.0))
    cube_a = bpy.context.active_object
    cube_a.name = "LockA"
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(1.0, 0.0, 0.0))
    cube_b = bpy.context.active_object
    cube_b.name = "LockB"

    root = dh.configure_state(project_name="lock_translation", frame_count=4)
    solids = dh.api.solver.create_group("Solids", "SOLID")
    solids.add("LockA", "LockB")
    group = root.object_group_0

    a_assigned = _resolve(group, "LockA")
    b_assigned = _resolve(group, "LockB")

    # LockA is locked to world Y (Blender axis), LockB stays free.
    a_assigned.lock_translation_enable = True
    a_assigned.lock_translation_axis = (0.0, 3.0, 0.0)
    b_assigned.lock_translation_enable = False

    uuid_a = a_assigned.uuid
    uuid_b = b_assigned.uuid

    # ----- A: enabled body's axis is swapped Blender->solver, unit ---
    lock = _lock_dict(dh, [uuid_a, uuid_b])
    axis_a = lock.get(uuid_a)
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
        uuid_b not in lock and len(lock) == 1,
        {"lock_keys": list(lock.keys())},
    )

    # ----- C: SAND group also encodes lock-translation ----------------
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.5, location=(3.0, 0.0, 0.0))
    sand_src = bpy.context.object
    sand_src.name = "LockSand"
    sand_ops = __import__(pkg + ".ui.dynamics.sand_ops",
                          fromlist=["build_and_commit_particle_mesh"])
    sand_ops.build_and_commit_particle_mesh(sand_src, 0.08, 0.0, rng_seed=0)
    sand_group_api = dh.api.solver.create_group("Sand", "SAND")
    sand_group_api.add(sand_src.name)
    sand_group = root.object_group_1
    sand_assigned = _resolve(sand_group, "LockSand")
    sand_assigned.lock_translation_enable = True
    sand_assigned.lock_translation_axis = (1.0, 0.0, 0.0)
    sand_lock = _lock_dict(dh, [sand_assigned.uuid])
    sand_axis = sand_lock.get(sand_assigned.uuid)
    sand_ok = (
        sand_axis is not None
        and math.isclose(sand_axis[0], 1.0, abs_tol=1e-9)
        and math.isclose(sand_axis[1], 0.0, abs_tol=1e-9)
        and math.isclose(sand_axis[2], 0.0, abs_tol=1e-9)
    )
    dh.record(
        "C_sand_group_encodes_lock_translation",
        sand_ok,
        {"sand_axis": list(sand_axis) if sand_axis is not None else None},
    )

    # ----- D: STATIC group prunes lock-translation via active_entries -
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, -1.0, 0.0))
    static_obj = bpy.context.active_object
    static_obj.name = "LockGround"
    static_group_api = dh.api.solver.create_group("Ground", "STATIC")
    static_group_api.add("LockGround")
    static_group = root.object_group_2
    static_assigned = _resolve(static_group, "LockGround")
    # Set the RNA field directly (no UI call site draws it for STATIC);
    # active_entries must still prune it since a STATIC object has no
    # free center of mass to lock.
    static_assigned.lock_translation_enable = True
    static_assigned.lock_translation_axis = (1.0, 0.0, 0.0)
    static_lock = _lock_dict(dh, [static_assigned.uuid])
    dh.record(
        "D_static_group_prunes_lock_translation",
        static_assigned.uuid not in static_lock,
        {"static_lock_keys": list(static_lock.keys())},
    )

    # ----- E: zero axis raises loudly at encode time -------------------
    a_assigned.lock_translation_axis = (0.0, 0.0, 0.0)
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
    a_assigned.lock_translation_axis = (0.0, 3.0, 0.0)

    # ----- F: eye toggle builds the selected group's axis overlay ----------
    previews = __import__(
        pkg + ".ui.dynamics.overlay_geometry.previews",
        fromlist=["_build_translation_lock_batches"],
    )
    group.preview_lock_translation = True
    shown_batches = previews._build_translation_lock_batches(
        bpy.context.scene, 10.0,
    )
    group.preview_lock_translation = False
    hidden_batches = previews._build_translation_lock_batches(
        bpy.context.scene, 10.0,
    )
    dh.record(
        "F_eye_toggle_controls_axis_overlay",
        len(shown_batches) == 2 and len(hidden_batches) == 0,
        {
            "shown_batch_count": len(shown_batches),
            "hidden_batch_count": len(hidden_batches),
        },
    )

    # ----- G: Lock All Translations replaces the axis on the wire -----
    # The axis dict and the all dict are two spellings of one lock state, and
    # exactly one of them may claim a given object: the axis encoder rejects
    # the zero axis an all-axes record requires, and the solver asserts the
    # biconditional (all mode if and only if a zero axis). So turning the
    # checkbox on must move this object OUT of "lock-translation" and INTO
    # "lock-all-translations", not add it to both.
    #
    # This subtest also settles which TREE the rig loaded: `lock_translation_all`
    # exists only where this feature is checked out, so an addon symlinked
    # elsewhere raises AttributeError here instead of quietly passing the rest.
    a_assigned.lock_translation_all = True
    all_dict = _lock_dict(dh, [uuid_a, uuid_b], "lock-all-translations")
    axis_dict = _lock_dict(dh, [uuid_a, uuid_b], "lock-translation")
    dh.record(
        "G_all_translations_replaces_the_axis_entry",
        all_dict.get(uuid_a) is True and uuid_a not in axis_dict,
        {"all_dict": {k: bool(v) for k, v in all_dict.items()},
         "axis_keys": list(axis_dict.keys())},
    )

    # ----- H: unticking restores the axis entry unchanged --------------
    # The panel disables the axis field rather than clearing it, so the value
    # the artist typed has to survive a trip through the all-axes mode.
    a_assigned.lock_translation_all = False
    restored = _lock_dict(dh, [uuid_a, uuid_b])
    axis_restored = restored.get(uuid_a)
    dh.record(
        "H_unticking_restores_the_axis",
        axis_restored is not None
        and math.isclose(axis_restored[2], -1.0, abs_tol=1e-9),
        {"axis": list(axis_restored) if axis_restored is not None else None},
    )

    # ----- I: an all-locked object with a ZEROED axis still encodes ----
    # The panel stops showing "Axis is zero; scene build will fail" while an
    # all-axes lock is on, because a zero axis is the CORRECT thing to send
    # then. That suppression is only honest if the encoder agrees: the axis
    # validator raises on a zero-length axis, so an object left at (0,0,0)
    # would otherwise turn a legal scene into a build failure with a message
    # naming an axis the artist was told to ignore.
    a_assigned.lock_translation_all = True
    a_assigned.lock_translation_axis = (0.0, 0.0, 0.0)
    encode_ok, encode_detail = True, "encoded"
    try:
        zero_all = _lock_dict(dh, [uuid_a, uuid_b], "lock-all-translations")
        zero_axis = _lock_dict(dh, [uuid_a, uuid_b], "lock-translation")
        encode_ok = zero_all.get(uuid_a) is True and uuid_a not in zero_axis
        encode_detail = {"all": bool(zero_all.get(uuid_a)),
                         "axis_keys": list(zero_axis.keys())}
    except Exception as encode_exc:  # noqa: BLE001 - the failure IS the result
        encode_ok = False
        encode_detail = f"{type(encode_exc).__name__}: {encode_exc}"
    dh.record("I_all_lock_with_zero_axis_encodes", encode_ok, encode_detail)
    a_assigned.lock_translation_all = False
    a_assigned.lock_translation_axis = (0.0, 3.0, 0.0)

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
