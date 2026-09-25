# File: scenarios/bl_timing_refusals.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Frame windows and keyframes the solve cannot place are REFUSED at encode, by
# name, rather than reshaped into timing the artist did not author.
#
# The simulation begins at the starting frame (`resolve_start_frame`), which is
# simulated time zero. A window whose End frame is not after its Start frame
# covers no time, and one that starts before the starting frame would have its
# first part cut off; either way the solver would run something other than what
# the panel shows. A collider's or a dynamic parameter's initial keyframe IS the
# state at the starting frame, so a later keyframe at or before that frame has
# nowhere to go, and the frontend would stop on a time-ordering error naming
# nothing. Encode-only: no build and no run.
#
# Every subtest runs with the starting frame at START (10), well above 1, since
# that is where clipping, a frame-1 anchor and an overlay that ignores the
# starting frame all show.
#
# Subtests:
#   A. pin_op_inverted_window_refused: a pin Move By, Spin, Scale and Torque
#      whose End frame is before its Start frame are each refused naming the
#      pin, the object, the operation type and both frames.
#   B. pin_op_zero_length_window_refused: the same with End equal to Start.
#   C. pin_op_window_before_start_refused: each of the four starting before
#      the starting frame is refused, saying to move the window or the start.
#   D. pin_op_window_at_start_accepted: each of the four starting exactly on
#      the starting frame encodes, with t_start 0 and t_end the window length.
#   E. static_op_inverted_window_refused: a STATIC Move By, Spin and Scale
#      with an End frame before or equal to the Start frame are refused naming
#      the object, its group, the operation type and both frames.
#   F. static_op_window_before_start_refused: each starting before the
#      starting frame is refused.
#   G. static_op_window_at_start_accepted: each starting on the starting frame
#      encodes with frame offsets 0 and the window length.
#   H. collision_window_refused: an inverted, a zero-length and a pre-start
#      collision window are refused naming the object, its group and both
#      frames.
#   I. collision_window_accepted: a window starting on the starting frame
#      encodes to (0, length), and a bad window on a group whose collision
#      windows are off reaches nothing and is not refused.
#   J. collider_keyframe_at_or_before_start_refused: an invisible collider
#      keyframe on the starting frame, and one before it, are refused naming
#      the collider and both frames.
#   K. collider_keyframe_out_of_order_refused: a keyframe whose frame was
#      edited to sit at or before the keyframe listed ahead of it is refused
#      naming both frames.
#   L. collider_keyframe_after_start_accepted: a keyframe after the starting
#      frame encodes, the initial keyframe at time 0 with the base position;
#      with the starting frame at 0 (below the initial keyframe's stored frame
#      1) the initial keyframe is still time 0.
#   M. collider_overlay_initial_at_start: the viewport overlay's collider
#      state places the initial keyframe at the starting frame, so halfway to
#      the next keyframe in frames is halfway in position, which is what the
#      encoded times say.
#   N. active_until_not_after_start_refused: an invisible collider Active
#      Until the starting frame, or earlier, is refused naming the collider;
#      one frame later encodes.
#   O. dyn_param_keyframe_at_or_before_start_refused: a legacy dynamic
#      parameter keyframe authored through MCP on the starting frame is
#      refused naming the parameter; after it, it encodes at its time.
#   P. mcp_initial_collider_keyframe_not_removable: MCP refuses to remove a
#      collider's initial keyframe, which would otherwise hand its role to the
#      next keyframe and drop that keyframe's position.
#   Q. new_default_windows_start_at_start: a pin operation and a STATIC
#      operation added over MCP without frames, and a collision window seeded
#      by the same helper the panel's add button uses, start at the starting
#      frame with their default length, and encode; frames a caller gives
#      explicitly are kept as given.
#   R. checkpoint_before_start_refused: a Save Checkpoints frame at or before
#      the starting frame is refused at encode (and by the MCP setter) naming
#      the frame, where it was dropped; one after the start encodes.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# Encode-only: it builds nothing and asks the solver for nothing.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

START = 10
FPS = 100
PIN_OP_TYPES = (
    ("MOVE_BY", "Move By"),
    ("SPIN", "Spin"),
    ("SCALE", "Scale"),
    ("TORQUE", "Torque"),
)
STATIC_OP_TYPES = (
    ("MOVE_BY", "Move By"),
    ("SPIN", "Spin"),
    ("SCALE", "Scale"),
)


def error_of(fn):
    try:
        fn(bpy.context)
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def param_error():
    return error_of(params_mod.encode_param)


def data_error():
    return error_of(mesh_mod.compute_data_hash)


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


try:
    dh = DriverHelpers(pkg, result)
    params_mod = __import__(pkg + ".core.encoder.params",
                            fromlist=["encode_param", "_build_param_dict"])
    mesh_mod = __import__(pkg + ".core.encoder.mesh",
                          fromlist=["compute_data_hash", "_build_obj_data"])
    uuid_registry = __import__(pkg + ".core.uuid_registry",
                               fromlist=["get_or_create_object_uuid"])
    colliders_mod = __import__(pkg + ".ui.dynamics.overlay_geometry.colliders",
                               fromlist=["_resolve_collider_state"])
    mcp_dyn = __import__(pkg + ".mcp.handlers.dyn_params",
                         fromlist=["add_dynamic_param", "remove_collider_keyframe"])

    plane = dh.reset_scene_to_pinned_plane(name="Sheet")
    dh.save_blend(PROBE_DIR, "timing_refusals.blend")
    root = dh.configure_state(project_name="timing_refusals",
                              frame_count=30, frame_rate=FPS)
    state = root.state
    state.use_scene_frame_start = False
    state.frame_start = START
    state.time_scale = 1.0
    scene = bpy.context.scene

    cloth_api = dh.api.solver.create_group("Cloth", "SHELL")
    cloth_api.add(plane.name)
    cloth_api.create_pin(plane.name, "AllPin")
    cloth = dh.groups.get_active_group_by_uuid(scene, cloth_api.uuid)
    pin_item = cloth.pin_vertex_groups[0]
    plane_uuid = uuid_registry.get_or_create_object_uuid(plane)

    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(3.0, 0.0, 0.0))
    mover = bpy.context.active_object
    mover.name = "Mover"
    static_api = dh.api.solver.create_group("Collider", "STATIC")
    static_api.add(mover.name)
    static_group = dh.groups.get_active_group_by_uuid(scene, static_api.uuid)
    mover_uuid = uuid_registry.get_or_create_object_uuid(mover)
    mover_assigned = next(
        a for a in static_group.assigned_objects if a.uuid == mover_uuid
    )

    baseline_param = param_error()
    baseline_data = data_error()
    dh.log(f"baseline: param={baseline_param!r} data={baseline_data!r}")

    def set_pin_op(op_type, frame_start, frame_end):
        pin_item.operations.clear()
        op = pin_item.operations.add()
        op.op_type = op_type
        op.spin_axis = (0.0, 0.0, 1.0)
        op.spin_center_mode = "CENTROID"
        op.scale_center_mode = "CENTROID"
        op.scale_factor = 0.5
        op.delta = (0.0, 0.0, 0.1)
        op.frame_start = frame_start
        op.frame_end = frame_end
        return op

    def pin_label(label):
        return f"Pin 'AllPin' on 'Sheet': its {label} operation"

    # ---- A: inverted pin windows --------------------------------------
    a_details = {}
    a_ok = baseline_param == "" and baseline_data == ""
    for op_type, label in PIN_OP_TYPES:
        set_pin_op(op_type, START + 10, START + 5)
        err = param_error()
        a_details[op_type] = err[:300]
        a_ok = a_ok and (
            err.startswith("ValueError: " + pin_label(label))
            and f"runs from frame {START + 10} to frame {START + 5}" in err
            and "has to end after it starts" in err
        )
    dh.record("A_pin_op_inverted_window_refused", a_ok,
              {"baseline_param": baseline_param,
               "baseline_data": baseline_data, "errors": a_details})

    # ---- B: zero-length pin windows -----------------------------------
    b_details = {}
    b_ok = True
    for op_type, label in PIN_OP_TYPES:
        set_pin_op(op_type, START + 5, START + 5)
        err = param_error()
        b_details[op_type] = err[:300]
        b_ok = b_ok and (
            err.startswith("ValueError: " + pin_label(label))
            and f"runs from frame {START + 5} to frame {START + 5}" in err
            and "has to end after it starts" in err
        )
    dh.record("B_pin_op_zero_length_window_refused", b_ok,
              {"errors": b_details})

    # ---- C: pin windows starting before the starting frame -----------
    c_details = {}
    c_ok = True
    for op_type, label in PIN_OP_TYPES:
        set_pin_op(op_type, START - 3, START + 10)
        err = param_error()
        c_details[op_type] = err[:400]
        c_ok = c_ok and (
            err.startswith("ValueError: " + pin_label(label))
            and f"runs from frame {START - 3} to frame {START + 10}" in err
            and f"starts before the starting frame {START}" in err
            and f"Move its Start frame to {START} or later" in err
            and f"start the simulation at frame {START - 3} or earlier" in err
        )
    dh.record("C_pin_op_window_before_start_refused", c_ok,
              {"errors": c_details})

    # ---- D: pin windows starting on the starting frame ----------------
    d_details = {}
    d_ok = True
    for op_type, _label in PIN_OP_TYPES:
        set_pin_op(op_type, START, START + 10)
        err = param_error()
        times = None
        if err == "":
            built = params_mod._build_param_dict(bpy.context)
            ops = built["pin_config"][plane_uuid][0].get("operations") or []
            if len(ops) == 1:
                times = (float(ops[0]["t_start"]), float(ops[0]["t_end"]))
        d_details[op_type] = {"error": err[:300], "times": times}
        d_ok = d_ok and (
            err == "" and times is not None
            and close(times[0], 0.0) and close(times[1], 10.0 / FPS)
        )
    pin_item.operations.clear()
    dh.record("D_pin_op_window_at_start_accepted", d_ok,
              {"results": d_details})

    # ---- E / F / G: STATIC operations ----------------------------------
    def set_static_op(op_type, frame_start, frame_end):
        mover_assigned.static_ops.clear()
        op = mover_assigned.static_ops.add()
        op.op_type = op_type
        op.spin_axis = (0.0, 0.0, 1.0)
        op.delta = (0.0, 0.0, 0.1)
        op.scale_factor = 0.5
        op.frame_start = frame_start
        op.frame_end = frame_end
        return op

    def static_label(label):
        return (f"STATIC object 'Mover' in group 'Collider': its {label} "
                "operation")

    e_details = {}
    e_ok = True
    for op_type, label in STATIC_OP_TYPES:
        for frame_start, frame_end in ((START + 10, START + 5),
                                       (START + 5, START + 5)):
            set_static_op(op_type, frame_start, frame_end)
            err = data_error()
            e_details[f"{op_type} {frame_start}-{frame_end}"] = err[:300]
            e_ok = e_ok and (
                err.startswith("ValueError: " + static_label(label))
                and f"runs from frame {frame_start} to frame {frame_end}" in err
                and "has to end after it starts" in err
            )
    dh.record("E_static_op_inverted_window_refused", e_ok,
              {"errors": e_details})

    f_details = {}
    f_ok = True
    for op_type, label in STATIC_OP_TYPES:
        set_static_op(op_type, START - 3, START + 10)
        err = data_error()
        f_details[op_type] = err[:400]
        f_ok = f_ok and (
            err.startswith("ValueError: " + static_label(label))
            and f"runs from frame {START - 3} to frame {START + 10}" in err
            and f"starts before the starting frame {START}" in err
        )
    dh.record("F_static_op_window_before_start_refused", f_ok,
              {"errors": f_details})

    g_details = {}
    g_ok = True
    for op_type, _label in STATIC_OP_TYPES:
        set_static_op(op_type, START, START + 10)
        err = data_error()
        offsets = None
        if err == "":
            for group_data in mesh_mod._build_obj_data(
                    bpy.context, persist_topology_hash=False):
                for info in group_data["object"]:
                    if info["uuid"] == mover_uuid and info.get("static_ops"):
                        entry = info["static_ops"][0]
                        offsets = (float(entry["frame_offset_start"]),
                                   float(entry["frame_offset_end"]))
        g_details[op_type] = {"error": err[:300], "offsets": offsets}
        g_ok = g_ok and (
            err == "" and offsets is not None
            and close(offsets[0], 0.0) and close(offsets[1], 10.0)
        )
    mover_assigned.static_ops.clear()
    dh.record("G_static_op_window_at_start_accepted", g_ok,
              {"results": g_details})

    # ---- H / I: collision windows --------------------------------------
    sheet_assigned = cloth.assigned_objects[0]
    cloth.use_collision_windows = True
    window_label = "Object 'Sheet' in group 'Cloth': its collision window"

    def set_window(frame_start, frame_end):
        sheet_assigned.collision_windows.clear()
        cw = sheet_assigned.collision_windows.add()
        cw.frame_start = frame_start
        cw.frame_end = frame_end

    h_details = {}
    h_ok = True
    for frame_start, frame_end, needle in (
            (START + 10, START + 5, "has to end after it starts"),
            (START + 5, START + 5, "has to end after it starts"),
            (START - 3, START + 10, f"starts before the starting frame {START}"),
    ):
        set_window(frame_start, frame_end)
        err = param_error()
        h_details[f"{frame_start}-{frame_end}"] = err[:400]
        h_ok = h_ok and (
            err.startswith("ValueError: " + window_label)
            and f"runs from frame {frame_start} to frame {frame_end}" in err
            and needle in err
        )
    dh.record("H_collision_window_refused", h_ok, {"errors": h_details})

    set_window(START, START + 10)
    err_i = param_error()
    windows = None
    if err_i == "":
        built = params_mod._build_param_dict(bpy.context)
        for params, _names, _uuids in built["group"]:
            cw_map = params.get("collision-windows") or {}
            if sheet_assigned.uuid in cw_map:
                windows = [tuple(float(t) for t in w)
                           for w in cw_map[sheet_assigned.uuid]]
    set_window(START + 10, START + 5)
    cloth.use_collision_windows = False
    err_i_off = param_error()
    sheet_assigned.collision_windows.clear()
    dh.record(
        "I_collision_window_accepted",
        err_i == "" and windows is not None and len(windows) == 1
        and close(windows[0][0], 0.0) and close(windows[0][1], 10.0 / FPS)
        and err_i_off == "",
        {"error": err_i[:300], "windows": windows,
         "feature_off_error": err_i_off[:300]},
    )

    # ---- J / K / L: invisible collider keyframes -----------------------
    wall = dh.api.solver.add_wall(position=(0.0, 0.0, -1.0),
                                  normal=(0.0, 0.0, 1.0))
    wall_item = wall._item
    wall_name = wall_item.name
    collider_label = f"Invisible collider '{wall_name}'"

    def clear_later_keyframes():
        while len(wall_item.keyframes) > 1:
            wall_item.keyframes.remove(len(wall_item.keyframes) - 1)

    j_details = {}
    j_ok = True
    for frame in (START, START - 4):
        clear_later_keyframes()
        kf = wall_item.keyframes.add()
        kf.frame = frame
        kf.position = (0.0, 0.0, 0.0)
        err = param_error()
        j_details[str(frame)] = err[:400]
        j_ok = j_ok and (
            err.startswith("ValueError: " + collider_label)
            and f"has a keyframe at frame {frame}, at or before the "
                f"starting frame {START}" in err
            and f"start the simulation before frame {frame}" in err
        )
    clear_later_keyframes()
    dh.record("J_collider_keyframe_at_or_before_start_refused", j_ok,
              {"errors": j_details})

    for frame in (START + 5, START + 10):
        kf = wall_item.keyframes.add()
        kf.frame = frame
        kf.position = (0.0, 0.0, 0.0)
    wall_item.keyframes[2].frame = START + 3
    err_k = param_error()
    wall_item.keyframes[2].frame = START + 5
    err_k_equal = param_error()
    clear_later_keyframes()
    dh.record(
        "K_collider_keyframe_out_of_order_refused",
        err_k.startswith("ValueError: " + collider_label)
        and f"has a keyframe at frame {START + 3} listed after the keyframe "
            f"at frame {START + 5}" in err_k
        and f"has a keyframe at frame {START + 5} listed after the keyframe "
            f"at frame {START + 5}" in err_k_equal,
        {"error": err_k[:400], "equal_error": err_k_equal[:400]},
    )

    wall.time(START + 10).move_to((0.0, 0.0, 1.0))
    err_l = param_error()
    encoded = None
    if err_l == "":
        built = params_mod._build_param_dict(bpy.context)
        walls = (built.get("invisible_colliders") or {}).get("walls") or []
        if walls:
            encoded = [(float(k["time"]), [float(c) for c in k["position"]])
                       for k in walls[0]["keyframes"]]
    state.frame_start = 0
    err_l_zero = param_error()
    zero_times = None
    if err_l_zero == "":
        built = params_mod._build_param_dict(bpy.context)
        walls = (built.get("invisible_colliders") or {}).get("walls") or []
        if walls:
            zero_times = [float(k["time"]) for k in walls[0]["keyframes"]]
    state.frame_start = START
    # Solver space is Y-up: the base Blender Z of -1 is the second component.
    dh.record(
        "L_collider_keyframe_after_start_accepted",
        err_l == "" and encoded is not None and len(encoded) == 2
        and close(encoded[0][0], 0.0) and close(encoded[0][1][1], -1.0, 1e-6)
        and close(encoded[1][0], 10.0 / FPS)
        and close(encoded[1][1][1], 1.0, 1e-6)
        and err_l_zero == "" and zero_times is not None
        and close(zero_times[0], 0.0) and close(zero_times[1], (START + 10) / FPS),
        {"error": err_l[:300], "encoded": encoded,
         "start_zero_error": err_l_zero[:300], "start_zero_times": zero_times},
    )

    # ---- M: the overlay places the initial state at the starting frame -
    resolve = colliders_mod._resolve_collider_state
    states = {}
    for frame in (1, START, START + 5, START + 10, START + 20):
        pos, _radius = resolve(wall_item, frame, START)
        states[frame] = float(pos.z)
    dh.record(
        "M_collider_overlay_initial_at_start",
        close(states[1], -1.0, 1e-6)
        and close(states[START], -1.0, 1e-6)
        # Halfway in frames between the starting frame and the keyframe is
        # halfway in encoded time, so halfway in position.
        and close(states[START + 5], 0.0, 1e-6)
        and close(states[START + 10], 1.0, 1e-6)
        and close(states[START + 20], 1.0, 1e-6),
        {"z_by_frame": states,
         "anchored_at_frame_1_would_be": -1.0 + 2.0 * (START + 4) / (START + 9)},
    )
    clear_later_keyframes()

    # ---- N: Active Until not after the starting frame -------------------
    wall_item.enable_active_duration = True
    n_details = {}
    n_ok = True
    for frame in (START, START - 5):
        wall_item.active_duration = frame
        err = param_error()
        n_details[str(frame)] = err[:400]
        n_ok = n_ok and (
            err.startswith("ValueError: " + collider_label)
            and f"is Active Until frame {frame}, which is not after the "
                f"starting frame {START}" in err
            and "turn Active Duration off" in err
        )
    wall_item.active_duration = START + 1
    err_n_ok = param_error()
    cutoff = None
    if err_n_ok == "":
        built = params_mod._build_param_dict(bpy.context)
        walls = (built.get("invisible_colliders") or {}).get("walls") or []
        if walls:
            cutoff = float(walls[0]["active_duration"])
    wall_item.enable_active_duration = False
    dh.record(
        "N_active_until_not_after_start_refused",
        n_ok and err_n_ok == "" and cutoff is not None
        and close(cutoff, 0.5 / FPS),
        {"errors": n_details, "one_after_error": err_n_ok[:300],
         "cutoff": cutoff},
    )

    # ---- O: legacy dynamic parameter keyframes --------------------------
    added = mcp_dyn.add_dynamic_param({"param_type": "GRAVITY"})
    on_start = mcp_dyn.add_dynamic_param_keyframe(
        {"param_type": "GRAVITY", "frame": START, "gravity": [0.0, 0.0, -5.0]})
    err_o = param_error()
    mcp_dyn.remove_dynamic_param_keyframe(
        {"param_type": "GRAVITY", "frame": START})
    after = mcp_dyn.add_dynamic_param_keyframe(
        {"param_type": "GRAVITY", "frame": START + 10,
         "gravity": [0.0, 0.0, -5.0]})
    err_o_after = param_error()
    gravity_times = None
    if err_o_after == "":
        built = params_mod._build_param_dict(bpy.context)
        entries = (built.get("dyn_param") or {}).get("gravity") or []
        gravity_times = [float(e[0]) for e in entries]
    mcp_dyn.remove_dynamic_param({"param_type": "GRAVITY"})
    dh.record(
        "O_dyn_param_keyframe_at_or_before_start_refused",
        added.get("status") == "success"
        and on_start.get("status") == "success"
        and err_o.startswith("ValueError: Dynamic parameter GRAVITY")
        and f"has a keyframe at frame {START}, at or before the starting "
            f"frame {START}" in err_o
        and after.get("status") == "success" and err_o_after == ""
        and gravity_times is not None and len(gravity_times) == 2
        and close(gravity_times[0], 0.0)
        and close(gravity_times[1], 10.0 / FPS),
        {"error": err_o[:400], "after_error": err_o_after[:300],
         "gravity_times": gravity_times},
    )

    # ---- P: MCP keeps a collider's initial keyframe ---------------------
    wall_index = list(state.invisible_colliders).index(wall_item)
    wall.time(START + 12).move_to((0.0, 0.0, 2.0))
    before = [int(k.frame) for k in wall_item.keyframes]
    refused = mcp_dyn.remove_collider_keyframe(
        {"index": wall_index, "frame": 1})
    kept = [int(k.frame) for k in wall_item.keyframes]
    removed = mcp_dyn.remove_collider_keyframe(
        {"index": wall_index, "frame": START + 12})
    after_remove = [int(k.frame) for k in wall_item.keyframes]
    dh.record(
        "P_mcp_initial_collider_keyframe_not_removable",
        before == [1, START + 12]
        and refused.get("status") == "error"
        and "Cannot remove the initial keyframe" in refused.get("message", "")
        and kept == before
        and removed.get("status") == "success" and after_remove == [1],
        {"before": before, "refused": refused, "kept": kept,
         "removed": removed, "after_remove": after_remove},
    )

    # ---- Q: new items from defaults start at the starting frame --------
    mcp_obj = __import__(pkg + ".mcp.handlers.object_ops",
                         fromlist=["add_pin_operation", "add_static_op"])
    encoder_pkg = __import__(pkg + ".core.encoder",
                             fromlist=["seed_window_at_start"])
    pin_item.operations.clear()
    mover_assigned.static_ops.clear()
    pin_added = mcp_obj.add_pin_operation({
        "group_uuid": cloth_api.uuid,
        "vertex_group_identifier": "Sheet::AllPin",
        "op_type": "MOVE_BY", "delta": [0.1, 0.0, 0.0],
    })
    pin_window = [(int(o.frame_start), int(o.frame_end)) for o in pin_item.operations]
    static_added = mcp_obj.add_static_op({
        "group_uuid": static_api.uuid, "object_name": "Mover",
        "op_type": "MOVE_BY", "delta": [0.1, 0.0, 0.0],
    })
    static_window = [(int(o.frame_start), int(o.frame_end))
                     for o in mover_assigned.static_ops]
    encoded_q = param_error() or data_error()
    probe = mover_assigned.collision_windows.add()
    probe.frame_start, probe.frame_end = 1, 60
    encoder_pkg.seed_window_at_start(probe, state)
    seeded_window = (int(probe.frame_start), int(probe.frame_end))
    mover_assigned.collision_windows.remove(len(mover_assigned.collision_windows) - 1)
    pin_item.operations.clear()
    explicit = mcp_obj.add_pin_operation({
        "group_uuid": cloth_api.uuid,
        "vertex_group_identifier": "Sheet::AllPin",
        "op_type": "MOVE_BY", "delta": [0.1, 0.0, 0.0],
        "frame_start": START + 3, "frame_end": START + 9,
    })
    explicit_window = [(int(o.frame_start), int(o.frame_end)) for o in pin_item.operations]
    dh.record(
        "Q_new_default_windows_start_at_start",
        pin_added.get("status") == "success"
        and pin_window == [(START, START + 59)]
        and static_added.get("status") == "success"
        and static_window == [(START, START + 59)]
        and seeded_window == (START, START + 59)
        and encoded_q == ""
        and explicit.get("status") == "success"
        and explicit_window == [(START + 3, START + 9)],
        {"pin": pin_window, "static": static_window, "seeded": seeded_window,
         "explicit": explicit_window, "encode_error": encoded_q[:300]},
    )
    pin_item.operations.clear()
    mover_assigned.static_ops.clear()

    # ---- R: a checkpoint frame at or before the starting frame ---------
    mcp_remote = __import__(pkg + ".mcp.handlers.remote",
                            fromlist=["set_save_checkpoint_frames"])
    state.save_checkpoint_frames.clear()
    for frame in (START, START + 5):
        item = state.save_checkpoint_frames.add()
        item.frame = frame
    err_r = param_error()
    state.save_checkpoint_frames.clear()
    item = state.save_checkpoint_frames.add()
    item.frame = START + 5
    ok_r = param_error()
    refused_mcp = mcp_remote.set_save_checkpoint_frames({"frames": [START - 1, START + 5]})
    state.save_checkpoint_frames.clear()
    dh.record(
        "R_checkpoint_before_start_refused",
        f"Checkpoint frame {START} is at or before the starting frame {START}" in err_r
        and ok_r == ""
        and refused_mcp.get("status") == "error"
        and f"Checkpoint frame {START - 1}" in refused_mcp.get("message", ""),
        {"error": err_r[:300], "after_start_error": ok_r[:300],
         "mcp": refused_mcp},
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
