# File: scenarios/bl_hair_curves_fur.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Blender's own Fur, simulated (issue #146). Object > Quick Effects > Quick
# Fur builds the stack an artist actually has: a few guide curves in the
# object's data and modifiers that set the profile, interpolate thousands
# of children from the guides, add noise and frizz, and end in Blender's
# Hair Dynamics, which is pinned to the end of the stack. This scenario
# adds that object to a ROD group as it is, with the head it grows on as a
# STATIC collider, and runs it.
#
# bl_hair_curves_run and bl_hair_curves_scalp author their hair point by
# point. What only this one covers is the generated stack: the solve runs
# on the guides, is played back in front of the modifier that generates
# the children, and the children are rebuilt from it on every frame.
#
# Quick Fur also puts a Geometry Nodes modifier on the HEAD, Capture Rest
# Geometry, which records the head's rest shape and moves nothing. The
# add-on reads a node modifier that contains a Set Position node as a
# deformer without running it, so the transfer asks for a Capture
# Deformation on the head, and the scenario does what that message says.
#
# Coverage targets:
#   A. fur_joins_as_it_is:       the fur joins a ROD group with its Roots
#                                pin, the rest angle from its groom and a
#                                zero contact offset; the head joins a
#                                STATIC group. Nothing else is set.
#   B. builds_with_no_allowance: the transfer names the head and the
#                                capture it wants; once the head is
#                                captured, with every guide rooted on it,
#                                the scene builds with no intersection
#                                allowance and exempts no pair.
#   C. guides_hold_and_fall:     each guide's first segment stays exactly
#                                where the groom put it, its tip ends
#                                lower, and no guide point is inside the
#                                head at any frame.
#   D. children_follow_the_guides: the stack still generates every child,
#                                on roots that have not moved, and the
#                                children's tips have come down with the
#                                guides'.
#   E. clear_shows_the_fur_again: Clear Local Animation leaves the evaluated
#                                fur exactly as it was before the solve.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND: the contact and the pins are the solver's.
BACKENDS = ("real",)


_RADIUS = 0.5
_LENGTH = 0.15
_FRAMES = 12


_DRIVER_BODY = r"""
import os
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
RADIUS = <<RADIUS>>
LENGTH = <<LENGTH>>
FRAMES = <<FRAMES>>


def evaluated(obj):
    # (positions, curve offsets) of the evaluated object: the children.
    deps = bpy.context.evaluated_depsgraph_get()
    deps.update()
    data = obj.evaluated_get(deps).data
    flat = np.empty(len(data.points) * 3, dtype=np.float32)
    data.attributes["position"].data.foreach_get("vector", flat)
    offsets = np.empty(len(data.curve_offset_data), dtype=np.int32)
    data.curve_offset_data.foreach_get("value", offsets)
    return flat.reshape(-1, 3), offsets


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    hair_mod = __import__(pkg + ".core.hair_curves", fromlist=["x"])
    pc2_mod = __import__(pkg + ".core.pc2", fromlist=["x"])
    client_mod = __import__(pkg + ".core.client", fromlist=["x"])

    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=RADIUS, segments=24, ring_count=12, location=(0.0, 0.0, 0.0))
    head = bpy.context.active_object
    head.name = "HeadObj"
    # The head is convex, so a point's signed distance to it is bounded
    # below by its largest signed distance to any face plane.
    face_normals = np.array([list(p.normal) for p in head.data.polygons],
                            dtype=np.float64)
    face_offsets = np.array([p.normal.dot(p.center)
                             for p in head.data.polygons], dtype=np.float64)

    def clearance(points):
        planes = np.asarray(points, dtype=np.float64) @ face_normals.T
        return (planes - face_offsets).max(axis=-1)
    bpy.ops.object.quick_fur(density="MEDIUM", length=LENGTH)
    fur = next(o for o in bpy.data.objects
               if o.type == "CURVES" and o.data.surface == head)
    stack = [m.name for m in fur.modifiers]
    guides = hair_mod.local_positions(fur).copy()
    offsets = hair_mod.curve_offsets(fur.data)
    n_guides = len(offsets) - 1
    held = hair_mod.root_segment_indices(fur).tolist()
    tips = (offsets[1:] - 1).tolist()
    children_before, child_offsets_before = evaluated(fur)

    dh.save_blend(PROBE_DIR, "hair_curves_fur.blend")
    root_state = dh.configure_state(
        project_name="hair_curves_fur", frame_count=FRAMES,
        gravity=(0.0, 0.0, -9.8),
    )
    root_state.state.disable_contact = False

    dh.api.solver.create_group("Fur", "ROD")
    dh.api.solver.create_group("Head", "STATIC")
    group = root_state.object_group_0
    bpy.ops.object.select_all(action="DESELECT")
    fur.select_set(True)
    bpy.context.view_layer.objects.active = fur
    bpy.ops.object.add_objects_to_group(group_index=0)
    bpy.ops.object.select_all(action="DESELECT")
    head.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.add_objects_to_group(group_index=1)

    # ----- A: the fur joins as it is -------------------------------
    pins = [p.name for p in group.pin_vertex_groups]
    dh.record(
        "A_fur_joins_as_it_is",
        pins == [f"[{fur.name}][{hair_mod.ROOTS_PIN_NAME}]"]
        and group.bend_rest_angle_source == "FROM_GEOMETRY"
        and float(group.contact_offset) == 0.0
        and not group.allow_existing_intersection
        and [m.name for m in fur.modifiers] == stack
        and len(root_state.object_group_1.assigned_objects) == 1,
        {"pins": pins, "rest": group.bend_rest_angle_source,
         "contact_offset": float(group.contact_offset), "stack": stack,
         "guides": n_guides, "guide_points": int(len(guides)),
         "children": int(len(child_offsets_before) - 1)},
    )

    # ----- B: builds with no allowance -----------------------------
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj"])
    try:
        encoder_mesh.encode_obj(bpy.context)
        asked = ""
    except ValueError as exc:
        asked = str(exc)
    sd = __import__(pkg + ".ui.dynamics.static_deform_ops",
                    fromlist=["start_capture_for_objects", "advance_capture",
                              "finalize_capture", "cleanup_capture"])
    ok, err = sd.start_capture_for_objects(bpy.context, [head])
    if not ok:
        raise RuntimeError(f"head capture did not start: {err}")
    while True:
        more, aborted, capture_error = sd.advance_capture(bpy.context)
        if aborted:
            raise RuntimeError(f"head capture aborted: {capture_error}")
        if not more:
            break
    sd.finalize_capture(bpy.context)
    sd.cleanup_capture(bpy.context)

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root_state.state.project_name)
    solver, error = dh.build_verdict("hair_fur:build")
    exemptions = list(dh.facade.engine.state.exemptions)
    dh.record(
        "B_builds_with_no_allowance",
        f"'{head.name}'" in asked and "Capture Deformation" in asked
        and solver in ("READY", "RESUMABLE") and not exemptions,
        {"asked_for": asked[:300], "solver": solver, "error": error[:600],
         "exemptions": len(exemptions)},
    )
    if solver not in ("READY", "RESUMABLE"):
        raise RuntimeError(f"build failed: {error}")

    dh.run_and_wait(timeout=900.0)
    run_state = dh.facade.engine.state
    if run_state.solver.name == "FAILED":
        raise RuntimeError(
            "the solve failed: "
            f"{run_state.server_error or run_state.error}"
        )
    dh.force_frame_query(expected_frames=1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain(fetch_timeout=120.0, drain_timeout=120.0)
    dh.log("fetched")

    pc2_path = pc2_mod.get_pc2_path(client_mod.object_pc2_key(fur))
    pc2_arr = dh.read_pc2(pc2_path) if os.path.isfile(pc2_path) else None
    if pc2_arr is None or pc2_arr.shape[1] != len(guides):
        raise RuntimeError(
            f"no usable PC2 at {pc2_path}: "
            f"{None if pc2_arr is None else pc2_arr.shape}"
        )

    # ----- C: the guides hold at the root and fall -----------------
    # The head sits at the origin, unrotated, so the fur's local frame is
    # the world's and the head's: -Z is down.
    root_err = float(np.max(np.abs(pc2_arr[:, held, :] - guides[held])))
    tip_drop = guides[tips, 2] - pc2_arr[-1, tips, 2]
    held_set = set(held)
    free = [i for i in range(len(guides)) if i not in held_set]
    closest = float(clearance(pc2_arr[:, free, :]).min())
    dh.record(
        "C_guides_hold_and_fall",
        pc2_arr.shape[0] >= FRAMES - 1
        and root_err <= 1.0e-5
        and float(np.mean(tip_drop)) > 5.0e-3
        and closest >= -1.0e-5,
        {"frames": int(pc2_arr.shape[0]), "max_root_error": root_err,
         "mean_tip_drop": float(np.mean(tip_drop)),
         "min_tip_drop": float(np.min(tip_drop)),
         "closest_to_the_head": closest},
    )

    # ----- D: the children follow the guides -----------------------
    start_frame = int(bpy.context.scene.frame_start)
    bpy.context.scene.frame_set(start_frame + pc2_arr.shape[0] - 1)
    bound = [m.name for m in fur.modifiers]
    children_after, child_offsets_after = evaluated(fur)
    same_children = bool(
        np.array_equal(child_offsets_after, child_offsets_before))
    child_roots_before = children_before[child_offsets_before[:-1]]
    child_tips_before = children_before[child_offsets_before[1:] - 1]
    root_shift = tip_fall = float("nan")
    if same_children:
        root_shift = float(np.max(np.abs(
            children_after[child_offsets_after[:-1]] - child_roots_before)))
        tip_fall = float(np.mean(
            child_tips_before[:, 2]
            - children_after[child_offsets_after[1:] - 1][:, 2]))
    dh.record(
        "D_children_follow_the_guides",
        same_children
        and len(child_offsets_after) - 1 > 10 * n_guides
        and bound.index(pc2_mod.MODIFIER_NAME) + 1
        == bound.index("Interpolate Hair Curves")
        and root_shift <= 1.0e-4
        and tip_fall > 5.0e-3
        and np.array_equal(hair_mod.local_positions(fur), guides),
        {"children": int(len(child_offsets_after) - 1), "stack": bound,
         "max_child_root_shift": root_shift,
         "mean_child_tip_fall": tip_fall},
    )

    # ----- E: clear shows the fur again ----------------------------
    animation_mod = __import__(pkg + ".core.animation",
                               fromlist=["clear_animation_data"])
    animation_mod.clear_animation_data(bpy.context)
    children_cleared, child_offsets_cleared = evaluated(fur)
    restored = (
        np.array_equal(child_offsets_cleared, child_offsets_before)
        and float(np.max(np.abs(children_cleared - children_before))) <= 1.0e-6
    )
    dh.record(
        "E_clear_shows_the_fur_again",
        bool(restored)
        and [m.name for m in fur.modifiers] == stack
        and hair_mod.SIM_POSITION_ATTRIBUTE not in fur.data.attributes
        and not os.path.exists(pc2_path),
        {"restored": bool(restored),
         "stack": [m.name for m in fur.modifiers],
         "pc2_left": os.path.exists(pc2_path)},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<RADIUS>>", str(_RADIUS))
        .replace("<<LENGTH>>", str(_LENGTH))
        .replace("<<FRAMES>>", str(_FRAMES))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 1500.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
