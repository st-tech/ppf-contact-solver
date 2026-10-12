# File: scenarios/bl_hair_curves_run.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Hair Curves simulated as rods, end to end (issue #146). A Hair Curves
# object (Blender object type CURVES) joins a ROD group, is transferred,
# solved, fetched and played back, and its animation is then cleared.
#
# A Hair Curves point is a rod vertex, so one index space runs through
# the encoder, the pin, the solver output and the playback cache. What
# differs from a mesh rod is everything Blender refuses on this object
# type: it has no vertex groups, so the pin here is the derived Roots
# pin, which holds the first segment of every strand, and it takes no
# Mesh Cache modifier, so playback is a point attribute plus a Geometry
# Nodes ContactSolverCache.
#
# Coverage targets:
#   A. encoded_as_rod:          the object ships one vertex per point,
#                               one edge per consecutive pair within a
#                               curve, no faces, and the Roots pin as the
#                               first two points of every curve.
#   B. joining_sets_hair_defaults: the first object of the group being
#                               hair switches Rest Angle to From Initial
#                               Geometry and the Contact Offset to zero;
#                               with no surface, no pin is made.
#   C. pc2_has_point_positions: the fetched PC2 holds (frames, points, 3).
#   D. roots_hold_tips_fall:    the first segment of every strand stays
#                               where it started on every frame, the third
#                               point has left the line of that segment,
#                               and every tip ends lower. A strand turning
#                               about its second point passes this as one
#                               bending away from a clamp would: a rod has
#                               no bending term at a held vertex
#                               (bl_hair_curves_weights measures it).
#   E. playback_reaches_the_evaluated_object: at the last frame the
#                               evaluated points equal the PC2 frame, the
#                               ContactSolverCache is a bound NODES
#                               modifier, and the groom in ``position`` is
#                               byte-identical to what it was.
#   F. playback_follows_the_timeline: stepping back to the starting frame
#                               shows the starting pose again.
#   G. clear_takes_everything_back: Clear Local Animation removes the
#                               modifier, the playback attribute, the node
#                               group and the PC2, and the evaluated object
#                               is the groom.
#   P. solve_survives_a_reopen: saved, reopened and with nothing left in
#                               memory, the file plays the solve back from
#                               its PC2 on the first frame change. Runs
#                               before G, which removes what it reads.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND: it solves, and the checks read the solve.
BACKENDS = ("real",)


# Strands of _N_POINTS points laid out along +X, one above the other, so
# gravity bends each one about its pinned root.
_N_STRANDS = 3
_N_POINTS = 8
_SEGMENT = 0.05


_DRIVER_BODY = r"""
import os
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
N_STRANDS = <<N_STRANDS>>
N_POINTS = <<N_POINTS>>
SEGMENT = <<SEGMENT>>


def evaluated_points(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    deps.update()
    data = obj.evaluated_get(deps).data
    flat = np.empty(len(data.points) * 3, dtype=np.float32)
    data.attributes["position"].data.foreach_get("vector", flat)
    return flat.reshape(-1, 3)


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    hair_mod = __import__(pkg + ".core.hair_curves", fromlist=["x"])
    pc2_mod = __import__(pkg + ".core.pc2", fromlist=["x"])
    client_mod = __import__(pkg + ".core.client", fromlist=["x"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    curves = bpy.data.hair_curves.new("Hair")
    curves.add_curves([N_POINTS] * N_STRANDS)
    groom = np.array(
        [[i * SEGMENT, 0.0, 0.1 * s]
         for s in range(N_STRANDS) for i in range(N_POINTS)],
        dtype=np.float32,
    )
    curves.attributes["position"].data.foreach_set("vector", groom.ravel())
    hair = bpy.data.objects.new("HairObj", curves)
    bpy.context.collection.objects.link(hair)
    n_points = N_STRANDS * N_POINTS
    # The Roots pin: the first segment of every strand.
    roots = [s * N_POINTS + i for s in range(N_STRANDS) for i in (0, 1)]
    thirds = [s * N_POINTS + 2 for s in range(N_STRANDS)]
    tips = [s * N_POINTS + N_POINTS - 1 for s in range(N_STRANDS)]

    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "hair_curves.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    # The rig's state defaults switch gravity off; this scenario is about
    # strands bending under it.
    root = dh.configure_state(
        project_name="hair_curves_run", frame_count=12,
        gravity=(0.0, 0.0, -9.8),
    )
    rod_group = dh.api.solver.create_group("Hair", "ROD")
    group = root.object_group_0
    rest_before = group.bend_rest_angle_source
    offset_before = float(group.contact_offset)

    bpy.ops.object.select_all(action="DESELECT")
    hair.select_set(True)
    bpy.context.view_layer.objects.active = hair
    bpy.ops.object.add_objects_to_group(group_index=0)

    # ----- B: what joining the group set ---------------------------
    dh.record(
        "B_joining_sets_hair_defaults",
        rest_before == "FLAT"
        and group.bend_rest_angle_source == "FROM_GEOMETRY"
        and offset_before > 0.0
        and float(group.contact_offset) == 0.0
        and float(group.contact_offset_rat) == 0.0
        and len(group.assigned_objects) == 1
        and len(group.pin_vertex_groups) == 0,
        {
            "rest_before": rest_before,
            "rest_after": group.bend_rest_angle_source,
            "offset_before": offset_before,
            "offset_after": float(group.contact_offset),
            "assigned": len(group.assigned_objects),
            "pins": len(group.pin_vertex_groups),
        },
    )

    # The derived Roots pin, through the public mutation gate.
    mutation_mod = __import__(pkg + ".core.mutation", fromlist=["create_pin"])
    mutation_mod.create_pin(rod_group._uuid, hair.name, hair_mod.ROOTS_PIN_NAME)

    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj", "compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash", "encode_param"])

    # ----- A: the encoded object -----------------------------------
    raw_data = encoder_mesh.encode_obj(bpy.context)
    decoded = dh.decode_addon_blob(raw_data)
    info = None
    for entry in decoded:
        if entry.get("type") == "ROD":
            for candidate in entry.get("object", []):
                if candidate.get("name") == hair.name:
                    info = candidate
    edges = [tuple(int(v) for v in e) for e in (info or {}).get("edge", [])]
    expected_edges = [
        (s * N_POINTS + i, s * N_POINTS + i + 1)
        for s in range(N_STRANDS) for i in range(N_POINTS - 1)
    ]
    dh.record(
        "A_encoded_as_rod",
        info is not None
        and len(info.get("vert", [])) == n_points
        and edges == expected_edges
        and "face" not in info
        and [int(i) for i in info.get("pin", [])] == roots,
        {
            "vertex_count": len(info.get("vert", [])) if info else 0,
            "edge_count": len(edges),
            "edges_match": edges == expected_edges,
            "has_face_key": bool(info and "face" in info),
            "pin": [int(i) for i in (info or {}).get("pin", [])],
            "expected_pin": roots,
        },
    )

    # ----- transfer + run + fetch ----------------------------------
    param_bytes = encoder_params.encode_param(bpy.context)
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=raw_data, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="hair_curves_run:build",
    ))
    deadline = time.time() + 180.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.3)
    if dh.facade.engine.state.solver.name == "FAILED":
        raise RuntimeError(build_failure_message(dh.facade, dh.com))
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    dh.run_and_wait(timeout=300.0)
    run_state = dh.facade.engine.state
    if run_state.solver.name == "FAILED":
        raise RuntimeError(
            "the solve failed: "
            f"{run_state.server_error or run_state.error}"
        )
    dh.force_frame_query(expected_frames=1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    dh.log("fetched")

    # ----- C: the PC2 ----------------------------------------------
    pc2_path = pc2_mod.get_pc2_path(client_mod.object_pc2_key(hair))
    pc2_arr = dh.read_pc2(pc2_path) if os.path.isfile(pc2_path) else None
    dh.record(
        "C_pc2_has_point_positions",
        pc2_arr is not None and pc2_arr.ndim == 3
        and pc2_arr.shape[1] == n_points and pc2_arr.shape[0] >= 3,
        {"pc2_path": pc2_path,
         "pc2_shape": list(pc2_arr.shape) if pc2_arr is not None else None,
         "n_points": n_points},
    )

    # ----- D: roots hold, tips fall --------------------------------
    # The object sits at the origin with no rotation, so its local frame
    # is the world frame and -Z is down.
    root_err = float("inf")
    tip_drop = []
    third_drop = []
    if pc2_arr is not None:
        root_err = float(np.max(np.abs(pc2_arr[:, roots, :] - groom[roots])))
        tip_drop = [
            float(groom[t, 2] - pc2_arr[-1, t, 2]) for t in tips
        ]
        third_drop = [
            float(groom[t, 2] - pc2_arr[-1, t, 2]) for t in thirds
        ]
    # Every strand turns about the second point of its roots, so its third
    # point drops and its tip drops further. A rod has no bending term at a
    # held vertex, so that is what a hinge there gives, and this check does
    # not tell it from a clamp.
    dh.record(
        "D_roots_hold_tips_fall",
        root_err <= 1.0e-5 and len(tip_drop) == N_STRANDS
        and all(d > 1.0e-3 for d in tip_drop)
        and all(0.0 < d < t for d, t in zip(third_drop, tip_drop)),
        {"max_root_error": root_err, "tip_drop": tip_drop,
         "third_point_drop": third_drop},
    )

    # ----- E: playback reaches the evaluated object ----------------
    start_frame = int(bpy.context.scene.frame_start)
    last = pc2_arr.shape[0] - 1 if pc2_arr is not None else 0
    bpy.context.scene.frame_set(start_frame + last)
    shown = evaluated_points(hair)
    mod = hair.modifiers.get(pc2_mod.MODIFIER_NAME)
    stored = np.empty(n_points * 3, dtype=np.float32)
    curves.attributes["position"].data.foreach_get("vector", stored)
    play_err = (
        float(np.max(np.abs(shown - pc2_arr[last])))
        if pc2_arr is not None and len(shown) == n_points else float("inf")
    )
    dh.record(
        "E_playback_reaches_the_evaluated_object",
        play_err <= 1.0e-6
        and mod is not None and mod.type == "NODES"
        and hair_mod.cache_modifier_is_bound(hair)
        and np.array_equal(stored.reshape(-1, 3), groom)
        and pc2_mod.has_mesh_cache(hair),
        {
            "playback_error": play_err,
            "modifier_type": mod.type if mod is not None else None,
            "bound": hair_mod.cache_modifier_is_bound(hair),
            "groom_untouched": bool(np.array_equal(stored.reshape(-1, 3), groom)),
            "has_mesh_cache": pc2_mod.has_mesh_cache(hair),
        },
    )

    # ----- F: playback follows the timeline ------------------------
    bpy.context.scene.frame_set(start_frame)
    at_start = evaluated_points(hair)
    start_err = (
        float(np.max(np.abs(at_start - pc2_arr[0])))
        if pc2_arr is not None else float("inf")
    )
    dh.record(
        "F_playback_follows_the_timeline",
        start_err <= 1.0e-6 and play_err <= 1.0e-6
        and float(np.max(np.abs(pc2_arr[0] - pc2_arr[last]))) > 1.0e-3,
        {"start_error": start_err,
         "first_to_last": float(np.max(np.abs(pc2_arr[0] - pc2_arr[last])))},
    )

    # ----- P: the solve survives a save and a reopen ---------------
    # A solve is the PC2 on disk plus what the .blend carries: the
    # modifier, its node group and the pin. The registry of loaded caches
    # is per session, so it is emptied as a fresh Blender would find it,
    # and the frame-change handler has to load the cache again on its own.
    bpy.ops.wm.save_mainfile()
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    pc2_mod.unload_curve_cache()
    hair = bpy.data.objects["HairObj"]
    curves = hair.data
    reopened_mod = hair.modifiers.get(pc2_mod.MODIFIER_NAME)
    registered_before = pc2_mod.has_curve_cache(
        client_mod.object_pc2_key(hair))
    bpy.context.scene.frame_set(start_frame + 1)
    bpy.context.scene.frame_set(start_frame + last)
    reopened = evaluated_points(hair)
    reopen_err = (
        float(np.max(np.abs(reopened - pc2_arr[last])))
        if len(reopened) == n_points else float("inf")
    )
    dh.record(
        "P_solve_survives_a_reopen",
        not registered_before
        and reopened_mod is not None and reopened_mod.type == "NODES"
        and hair_mod.cache_modifier_is_bound(hair)
        and reopen_err <= 1.0e-6
        and pc2_mod.has_curve_cache(client_mod.object_pc2_key(hair)),
        {
            "registered_before_the_frame_change": registered_before,
            "modifier_type":
                reopened_mod.type if reopened_mod is not None else None,
            "bound": hair_mod.cache_modifier_is_bound(hair),
            "playback_error": reopen_err,
        },
    )

    # ----- G: clear takes everything back --------------------------
    bpy.context.scene.frame_set(start_frame + last)
    animation_mod = __import__(pkg + ".core.animation",
                               fromlist=["clear_animation_data"])
    animation_mod.clear_animation_data(bpy.context)
    cleared = evaluated_points(hair)
    tagged = [g.name for g in bpy.data.node_groups
              if g.get("_solver_hair_cache")]
    dh.record(
        "G_clear_takes_everything_back",
        hair.modifiers.get(pc2_mod.MODIFIER_NAME) is None
        and hair_mod.SIM_POSITION_ATTRIBUTE not in curves.attributes
        and not tagged
        and not os.path.exists(pc2_path)
        and not pc2_mod.has_mesh_cache(hair)
        and np.array_equal(cleared, groom),
        {
            "modifier_left": hair.modifiers.get(pc2_mod.MODIFIER_NAME) is not None,
            "attribute_left": hair_mod.SIM_POSITION_ATTRIBUTE in curves.attributes,
            "node_groups_left": tagged,
            "pc2_left": os.path.exists(pc2_path),
            "shows_groom": bool(np.array_equal(cleared, groom)),
        },
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
        .replace("<<N_STRANDS>>", str(_N_STRANDS))
        .replace("<<N_POINTS>>", str(_N_POINTS))
        .replace("<<SEGMENT>>", str(_SEGMENT))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
