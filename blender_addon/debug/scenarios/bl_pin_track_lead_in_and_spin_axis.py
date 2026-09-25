# File: scenarios/bl_pin_track_lead_in_and_spin_axis.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Two pin encodings where a plausible payload turns into motion the artist
# never authored. Encode-only: no build and no run.
#
# A keyframed pin's track is integrated by the decoder as deltas from the pose
# each vertex ships at, which is the pose at the starting frame, so the track
# has to start from that pose. For a pin keyed across a lead-in (a key before
# the starting frame and one after it), a track that started at the earliest
# key instead would be offset by the motion between that key and the starting
# frame, and the pin would end away from its last key by exactly that much.
#
# A spin with a zero-length axis names no rotation. The solver normalizes the
# axis inside its rotation formula, and a zero one leaves the pinned vertices
# contracting toward the center instead of turning, so a spin that moves at all
# is refused without an axis, on a pin and on a STATIC collider alike.
#
# Subtests:
#   A. keyed_track_starts_at_the_starting_frame: with the starting frame
#      between two pin keys, the track's first sample is at time zero and holds
#      the pose the data payload ships the vertex at (the curve's value at the
#      starting frame), and its last sample is the last key, so the deltas the
#      decoder integrates end on it.
#   B. zero_axis_pin_spin_refused: a pin Spin with a zero-length axis and a
#      nonzero angular velocity is refused naming the pin and the object; at
#      zero angular velocity, or with an axis, it encodes.
#   C. zero_axis_static_spin_refused: the same for a STATIC collider's Spin
#      operation, refused naming the object and its group.
#   D. zero_max_towards_direction_refused: a pin Spin centered Max Towards a
#      zero-length direction picks no vertex as furthest, so it is refused
#      naming the pin, where it would otherwise spin about a point the artist
#      never picked; with a direction it encodes.

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
FIRST_KEY = 1
LAST_KEY = 20
SHIFT_X = 0.6
FPS = 100


def encode_error(fn):
    try:
        fn(bpy.context)
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def add_plane(name, x):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(x, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = name
    vg = obj.vertex_groups.new(name="AllPin")
    vg.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    return obj


try:
    dh = DriverHelpers(pkg, result)
    params_mod = __import__(pkg + ".core.encoder.params",
                            fromlist=["encode_param", "_build_param_dict"])
    mesh_mod = __import__(pkg + ".core.encoder.mesh",
                          fromlist=["compute_data_hash", "_build_obj_data"])
    uuid_registry = __import__(pkg + ".core.uuid_registry",
                               fromlist=["get_or_create_object_uuid"])
    plane = dh.reset_scene_to_pinned_plane(name="LeadInSheet")
    dh.save_blend(PROBE_DIR, "pin_track_lead_in.blend")
    root = dh.configure_state(project_name="pin_track_lead_in",
                              frame_count=30, frame_rate=FPS)
    state = root.state
    state.use_scene_frame_start = False
    state.frame_start = START
    state.time_scale = 1.0
    scene = bpy.context.scene
    scene.frame_end = 60

    # ---- A: a keyed pin whose keys straddle the starting frame ------------
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "AllPin")
    slot = dh.groups.get_group_slot_index(scene, cloth.uuid)
    cloth_group = dh.groups.get_active_group_by_uuid(scene, cloth.uuid)
    cloth_group.pin_vertex_groups_index = 0

    scene.frame_set(FIRST_KEY)
    rest_x = float(plane.data.vertices[0].co.x)
    bpy.ops.object.make_pin_keyframe(group_index=slot)
    scene.frame_set(LAST_KEY)
    for v in plane.data.vertices:
        v.co.x += SHIFT_X
    plane.data.update()
    bpy.ops.object.make_pin_keyframe(group_index=slot)
    scene.frame_set(FIRST_KEY)

    plane_uuid = uuid_registry.get_or_create_object_uuid(plane)
    built = params_mod._build_param_dict(bpy.context)
    cfg = built["pin_config"][plane_uuid][0]
    track = cfg["pin_anim"][0]
    times = [float(t) for t in track["time"]]
    first_x = float(track["position"][0][0])
    last_x = float(track["position"][-1][0])

    shipped_x = None
    for group_data in mesh_mod._build_obj_data(
            bpy.context, persist_topology_hash=False):
        for info in group_data["object"]:
            if info["uuid"] == plane_uuid:
                shipped_x = float(info["vert"][0][0])
    fraction = (START - FIRST_KEY) / (LAST_KEY - FIRST_KEY)
    expected_first_x = rest_x + SHIFT_X * fraction
    expected_last_x = rest_x + SHIFT_X
    expected_times = [0.0, (LAST_KEY - START) / FPS]
    dh.record(
        "A_keyed_track_starts_at_the_starting_frame",
        len(times) == 2
        and all(abs(a - b) < 1e-9 for a, b in zip(times, expected_times))
        and abs(first_x - expected_first_x) < 1e-5
        and shipped_x is not None and abs(first_x - shipped_x) < 1e-5
        and abs(last_x - expected_last_x) < 1e-5,
        {"times": times, "expected_times": expected_times,
         "first_x": first_x, "expected_first_x": expected_first_x,
         "shipped_x": shipped_x, "last_x": last_x,
         "expected_last_x": expected_last_x,
         "embedded_move_index": cfg.get("embedded_move_index")},
    )

    # ---- B: a pin Spin with no axis ---------------------------------------
    spun = add_plane("SpunSheet", 3.0)
    spun_api = dh.api.solver.create_group("Spun", "SHELL")
    spun_api.add(spun.name)
    spun_api.create_pin(spun.name, "AllPin").spin(
        axis=(0.0, 0.0, 0.0), angular_velocity=90.0,
        frame_start=START, frame_end=START + 10,
    )
    spun_group = dh.groups.get_active_group_by_uuid(scene, spun_api.uuid)
    spin_op = spun_group.pin_vertex_groups[0].operations[0]
    err_b = encode_error(params_mod.encode_param)
    spin_op.spin_angular_velocity = 0.0
    err_b_still = encode_error(params_mod.encode_param)
    spin_op.spin_angular_velocity = 90.0
    spin_op.spin_axis = (0.0, 0.0, 1.0)
    err_b_axis = encode_error(params_mod.encode_param)
    dh.record(
        "B_zero_axis_pin_spin_refused",
        "Pin 'AllPin' on 'SpunSheet'" in err_b
        and "zero-length axis" in err_b and "90.0" in err_b
        and err_b_still == "" and err_b_axis == "",
        {"error": err_b[:300], "zero_velocity_error": err_b_still[:300],
         "with_axis_error": err_b_axis[:300]},
    )

    # ---- D: a Max Towards center with no direction --------------------------
    spin_op.spin_center_mode = "MAX_TOWARDS"
    spin_op.spin_center_direction = (0.0, 0.0, 0.0)
    err_d = encode_error(params_mod.encode_param)
    spin_op.spin_center_direction = (0.0, 0.0, -1.0)
    err_d_direction = encode_error(params_mod.encode_param)
    spin_op.spin_center_mode = "CENTROID"
    dh.record(
        "D_zero_max_towards_direction_refused",
        "Pin 'AllPin' on 'SpunSheet'" in err_d
        and "Max Towards center has a zero-length direction" in err_d
        and err_d_direction == "",
        {"error": err_d[:300], "with_direction_error": err_d_direction[:300]},
    )

    # ---- C: a STATIC collider's Spin with no axis --------------------------
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(6.0, 0.0, 0.0))
    spinner = bpy.context.active_object
    spinner.name = "Spinner"
    stat_api = dh.api.solver.create_group("Stat", "STATIC")
    stat_api.add(spinner.name)
    stat_group = dh.groups.get_active_group_by_uuid(scene, stat_api.uuid)
    spinner_uuid = uuid_registry.get_or_create_object_uuid(spinner)
    assigned = next(
        a for a in stat_group.assigned_objects if a.uuid == spinner_uuid
    )
    static_op = assigned.static_ops.add()
    static_op.op_type = "SPIN"
    static_op.spin_axis = (0.0, 0.0, 0.0)
    static_op.spin_angular_velocity = 45.0
    static_op.frame_start = START
    static_op.frame_end = START + 10
    err_c = encode_error(mesh_mod.compute_data_hash)
    static_op.spin_angular_velocity = 0.0
    err_c_still = encode_error(mesh_mod.compute_data_hash)
    static_op.spin_angular_velocity = 45.0
    static_op.spin_axis = (0.0, 0.0, 1.0)
    err_c_axis = encode_error(mesh_mod.compute_data_hash)
    dh.record(
        "C_zero_axis_static_spin_refused",
        "STATIC object 'Spinner' in group 'Stat'" in err_c
        and "zero-length axis" in err_c and "45.0" in err_c
        and err_c_still == "" and err_c_axis == "",
        {"error": err_c[:300], "zero_velocity_error": err_c_still[:300],
         "with_axis_error": err_c_axis[:300]},
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
