# File: scenarios/bl_velocity_keyframes.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Velocity-keyframe encoding round-trip.
#
# Each AssignedObject carries a ``velocity_keyframes`` collection
# (frame, direction, speed) that the params encoder splits across two
# fields in ``param.pickle``:
#   - ``param["group"][i][0]["velocity"][uuid]``: the swapped, scaled
#     velocity for the ``frame == 1`` entry (the initial velocity).
#   - ``param["group"][i][0]["velocity-schedule"][uuid]``: a list of
#     ``(time_seconds, swapped_velocity_vec3)`` tuples for every
#     entry with ``frame > 1``. ``time_seconds`` is
#     ``(frame - 1) / fps``. The swap converts Blender Z-up directions
#     to solver Y-up: ``(x, y, z) -> (x, z, -y)``. The vector is
#     normalized then multiplied by ``speed`` before the swap.
#   - ``param["group"][i][0]["angular-velocity-schedule"][uuid]``: a
#     list of ``(time_seconds, pca_index, speed_rad)`` for every entry
#     with ``angular_speed != 0`` (including ``frame == 1`` at t=0).
#     There is NO axis swap: ``pca_index`` is a principal-axis selector
#     the solver resolves to a world axis from the live geometry, and
#     ``speed_rad`` is ``radians(angular_speed_deg)``.
#
# This scenario authors a small velocity schedule on a single SHELL
# assigned plane via the production
# ``OBJECT_OT_AddVelocityKeyframe`` operator (so the post-add
# ``sort_keyframes_by_frame`` actually runs), encodes the params,
# decodes the pickle, and asserts the encoded payload matches
# **hardcoded ground-truth** vectors computed by hand from
# axis-aligned inputs. We deliberately avoid re-implementing the
# encoder formula so a regression in the upstream swap or scale
# trips the test instead of silently agreeing with a mirrored copy.
#
# No build / run / fetch is involved; this is encoding-only.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import pickle
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="VelMesh")
    # frame_rate=100 keeps fps deterministic and independent of any
    # scene render override; the encoder reads frame_rate when
    # use_frame_rate_in_output is False (the default we set in
    # configure_state's caller chain). It also makes the hardcoded
    # schedule times below trivially exact: t = (frame - 1) / 100.
    root = dh.configure_state(project_name="velocity_keyframes",
                              frame_count=12, frame_rate=100)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    # A pin makes the group encodable; without one the group still
    # encodes but the test stays closer to a real cloth scene.
    cloth.create_pin(plane.name, "AllPin")

    group = root.object_group_0
    assigned = group.assigned_objects[0]
    # Resolve UUIDs eagerly so we can wire ``velocity_object_selection``
    # (the EnumProperty the AddVelocityKeyframe op reads to find the
    # active assigned object) before invoking the op.
    uuid_registry = __import__(pkg + ".core.uuid_registry",
                               fromlist=["resolve_assigned"])
    uuid_registry.resolve_assigned(assigned)
    group.velocity_object_selection = assigned.uuid

    # Authored inputs: axis-aligned directions chosen so the swap
    # ``(x, y, z) -> (x, z, -y)`` produces exact integer / half values
    # we can hardcode without re-running the encoder formula.
    # Frames are intentionally added out of order to exercise
    # ``sort_keyframes_by_frame`` in OBJECT_OT_AddVelocityKeyframe.
    INSERT_ORDER = [8, 1, 10, 5]  # non-monotonic by design
    PER_FRAME_AUTHOR = {
        # frame: (direction (Blender Z-up), speed)
        1:  ((0.0, 1.0, 0.0), 2.0),
        5:  ((0.0, 0.0, 1.0), 3.0),
        8:  ((1.0, 0.0, 0.0), 4.0),
        10: ((-1.0, 0.0, 0.0), 1.5),
    }
    # Angular (spin) component per frame: (axis_mode, speed_deg). All use
    # principal-axis modes here (the world/custom modes are checked in F).
    # Frame 1's nonzero spin proves frame-1 angular routes through the
    # schedule at t=0 (not a separate initial key); frame 10's zero is
    # omitted by the encoder.
    PER_FRAME_ANGULAR = {
        1:  ("PC3", 90.0),
        5:  ("PC1", -45.0),
        8:  ("PC2", 180.0),
        10: ("PC3", 0.0),
    }

    assigned.velocity_keyframes.clear()
    # Drive the production operator. It reads frame from
    # ``context.scene.frame_current`` and calls
    # sort_keyframes_by_frame after each add, so a non-monotonic
    # insert sequence is the load-bearing input.
    for frame in INSERT_ORDER:
        bpy.context.scene.frame_current = frame
        op_verdict = bpy.ops.object.add_velocity_keyframe(group_index=0)
        if "FINISHED" not in op_verdict:
            raise RuntimeError(
                f"add_velocity_keyframe(frame={frame}) returned {op_verdict!r}"
            )
        # The op only sets .frame; direction and speed are still the
        # PropertyGroup defaults. Locate the row we just added (it
        # may have been moved by sort) and stamp the authored values.
        added = None
        for kf in assigned.velocity_keyframes:
            if kf.frame == frame:
                added = kf
                break
        if added is None:
            raise RuntimeError(
                f"could not find newly added keyframe at frame {frame}"
            )
        direction, speed = PER_FRAME_AUTHOR[frame]
        added.direction = direction
        added.speed = speed
        axis_mode, ang_speed = PER_FRAME_ANGULAR[frame]
        added.angular_axis = axis_mode
        added.angular_speed = ang_speed
        # enable_translational stays at its default (True) so the linear
        # checks below see every keyframe; angular is enabled only where a
        # nonzero spin was authored.
        added.enable_angular = ang_speed != 0.0

    # ----- A: keyframes_authored_via_operator -------------------
    # Sanity check that the operator path produced one row per
    # insert and that direction / speed assignments stuck. This is
    # not the load-bearing assertion; B and C are.
    expected_count = len(INSERT_ORDER)
    by_frame = {kf.frame: kf for kf in assigned.velocity_keyframes}
    authored_ok = (
        len(assigned.velocity_keyframes) == expected_count
        and set(by_frame.keys()) == set(PER_FRAME_AUTHOR.keys())
        and all(
            tuple(round(c, 6) for c in by_frame[f].direction)
                == tuple(round(c, 6) for c in PER_FRAME_AUTHOR[f][0])
            and abs(by_frame[f].speed - PER_FRAME_AUTHOR[f][1]) < 1e-6
            for f in PER_FRAME_AUTHOR
        )
    )
    dh.record(
        "A_keyframes_authored_via_operator",
        authored_ok,
        {
            "count": len(assigned.velocity_keyframes),
            "expected_count": expected_count,
            "entries": [
                {
                    "frame": kf.frame,
                    "direction": tuple(kf.direction),
                    "speed": kf.speed,
                }
                for kf in assigned.velocity_keyframes
            ],
            "insert_order": INSERT_ORDER,
        },
    )

    # ----- Encode params and decode the CBOR envelope ------------
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = dh.decode_addon_blob(param_bytes)
    dh.log(f"decoded keys={sorted(decoded.keys())}")

    # group is encoded as [(params_dict, objects, object_uuids), ...].
    # Locate our group's params by UUID match.
    assigned_uuid = assigned.uuid
    group_params = None
    for params, _objects, object_uuids in decoded["group"]:
        if assigned_uuid in object_uuids:
            group_params = params
            break
    if group_params is None:
        raise RuntimeError(
            f"could not locate group entry for uuid {assigned_uuid!r} "
            f"in decoded param.pickle"
        )

    # ----- B: param_pickle_carries_velocity_keyframes -----------
    # Hardcoded ground truth. The encoder pipeline is
    # ``swap(normalize_and_scale(direction, speed))`` with the swap
    # ``(x, y, z) -> (x, z, -y)``. We chose axis-aligned inputs and
    # unit-length directions so the expected vectors are exact.
    #
    #   frame  direction (Z-up)  speed   -> normalize*speed (Z-up)
    #   1      (0, 1, 0)         2.0     -> (0, 2, 0)
    #   5      (0, 0, 1)         3.0     -> (0, 0, 3)
    #   8      (1, 0, 0)         4.0     -> (4, 0, 0)
    #   10     (-1, 0, 0)        1.5     -> (-1.5, 0, 0)
    #
    # Apply swap (x, y, z) -> (x, z, -y):
    #
    #   frame  swapped (Y-up, solver)
    #   1      (0, 0, -2)
    #   5      (0, 3, 0)
    #   8      (4, 0, 0)
    #   10     (-1.5, 0, 0)
    #
    # Schedule times are (frame - 1) / fps with fps == 100.
    EXPECTED_INITIAL = (0.0, 0.0, -2.0)
    EXPECTED_SCHEDULE = [
        (0.04, (0.0, 3.0, 0.0)),    # frame 5
        (0.07, (4.0, 0.0, 0.0)),    # frame 8
        (0.09, (-1.5, 0.0, 0.0)),   # frame 10
    ]

    init_payload = group_params["velocity"][assigned_uuid]
    init_actual = tuple(float(c) for c in init_payload)
    init_ok = all(
        abs(a - e) < 1e-5 for a, e in zip(init_actual, EXPECTED_INITIAL)
    )

    schedule_payload = group_params["velocity-schedule"][assigned_uuid]
    schedule_actual = [
        (float(t), tuple(float(c) for c in v)) for t, v in schedule_payload
    ]
    schedule_len_ok = len(schedule_actual) == len(EXPECTED_SCHEDULE)
    schedule_match_ok = schedule_len_ok and all(
        abs(schedule_actual[i][0] - EXPECTED_SCHEDULE[i][0]) < 1e-6
        and all(
            abs(schedule_actual[i][1][j] - EXPECTED_SCHEDULE[i][1][j]) < 1e-5
            for j in range(3)
        )
        for i in range(len(EXPECTED_SCHEDULE))
    )

    dh.record(
        "B_param_pickle_carries_velocity_keyframes",
        init_ok and schedule_match_ok,
        {
            "initial_velocity": init_actual,
            "initial_expected": EXPECTED_INITIAL,
            "schedule_actual": schedule_actual,
            "schedule_expected": EXPECTED_SCHEDULE,
        },
    )

    # ----- C: keyframes_in_correct_order ------------------------
    # The keyframes were added through OBJECT_OT_AddVelocityKeyframe
    # with INSERT_ORDER = [8, 1, 10, 5] (non-monotonic). The
    # operator calls sort_keyframes_by_frame after each add, so
    # both the source collection and the encoded schedule must
    # come out ascending. If sort_keyframes_by_frame regresses
    # (e.g. someone removes the call), this assertion fires
    # because the raw insert order is not sorted.
    src_frames = [kf.frame for kf in assigned.velocity_keyframes]
    src_sorted_ok = src_frames == sorted(src_frames)
    src_nontrivial = src_frames != INSERT_ORDER  # proves a sort happened

    times = [t for t, _ in schedule_actual]
    times_sorted_ok = times == sorted(times)
    # Schedule covers frames > 1, expected ascending: 0.04, 0.07, 0.09.
    expected_times = [0.04, 0.07, 0.09]
    times_value_ok = (
        len(times) == len(expected_times)
        and all(abs(times[i] - expected_times[i]) < 1e-6
                for i in range(len(expected_times)))
    )

    dh.record(
        "C_keyframes_in_correct_order",
        src_sorted_ok and src_nontrivial and times_sorted_ok and times_value_ok,
        {
            "insert_order": INSERT_ORDER,
            "source_frames": src_frames,
            "source_sorted": src_sorted_ok,
            "source_nontrivial": src_nontrivial,
            "schedule_times": times,
            "schedule_times_sorted": times_sorted_ok,
            "schedule_times_expected": expected_times,
        },
    )

    # ----- D: param_pickle_carries_angular_velocity_schedule ----
    # Every keyframe with angular_speed != 0 is emitted into
    # "angular-velocity-schedule" as (t, pca_index, speed_rad) with
    # t = (frame - 1) / fps (fps == 100) and speed_rad = radians(deg).
    # There is NO axis swap: pca_index is a principal-axis selector the
    # solver resolves to a world axis from the live geometry. Frame 10
    # (speed 0) is omitted; frame 1's spin lands at t=0.
    import math
    EXPECTED_ANGULAR = [
        (0.00, 2, math.radians(90.0)),    # frame 1  -> t=0
        (0.04, 0, math.radians(-45.0)),   # frame 5
        (0.07, 1, math.radians(180.0)),   # frame 8
    ]
    ang_payload = group_params.get("angular-velocity-schedule", {}).get(
        assigned_uuid, []
    )
    ang_actual = [
        (float(t), int(round(pca)), float(speed)) for t, pca, speed in ang_payload
    ]
    ang_len_ok = len(ang_actual) == len(EXPECTED_ANGULAR)
    ang_match_ok = ang_len_ok and all(
        abs(ang_actual[i][0] - EXPECTED_ANGULAR[i][0]) < 1e-6
        and ang_actual[i][1] == EXPECTED_ANGULAR[i][1]
        and abs(ang_actual[i][2] - EXPECTED_ANGULAR[i][2]) < 1e-5
        for i in range(len(EXPECTED_ANGULAR))
    )
    dh.record(
        "D_param_pickle_carries_angular_velocity_schedule",
        ang_match_ok,
        {
            "angular_actual": ang_actual,
            "angular_expected": EXPECTED_ANGULAR,
        },
    )

    # ----- E: checkboxes gate emission --------------------------
    # The "Enable Translational / Angular Velocity Overwrite" checkboxes
    # control emission independently of the stored vectors. Unchecking
    # both on every keyframe must drop ALL linear and angular entries and
    # zero the initial velocity; re-checking only angular must bring back
    # exactly the angular schedule (and keep the linear side empty). This
    # is the load-bearing assertion for the per-component gating.
    for kf in assigned.velocity_keyframes:
        kf.enable_translational = False
        kf.enable_angular = False
    decoded_off = dh.decode_addon_blob(dh.encoder_param.encode_param(bpy.context))
    gp_off = next(
        params for params, _o, uuids in decoded_off["group"]
        if assigned_uuid in uuids
    )
    init_off = tuple(float(c) for c in gp_off["velocity"][assigned_uuid])
    sched_off = gp_off["velocity-schedule"][assigned_uuid]
    ang_off = gp_off.get("angular-velocity-schedule", {}).get(assigned_uuid, [])
    all_off_ok = (
        all(abs(c) < 1e-9 for c in init_off)
        and len(sched_off) == 0
        and len(ang_off) == 0
    )

    # Re-enable only the angular component; linear must stay empty.
    for kf in assigned.velocity_keyframes:
        kf.enable_angular = kf.angular_speed != 0.0
    decoded_ang = dh.decode_addon_blob(dh.encoder_param.encode_param(bpy.context))
    gp_ang = next(
        params for params, _o, uuids in decoded_ang["group"]
        if assigned_uuid in uuids
    )
    init_ang = tuple(float(c) for c in gp_ang["velocity"][assigned_uuid])
    sched_ang = gp_ang["velocity-schedule"][assigned_uuid]
    ang_ang = gp_ang.get("angular-velocity-schedule", {}).get(assigned_uuid, [])
    angular_only_ok = (
        all(abs(c) < 1e-9 for c in init_ang)
        and len(sched_ang) == 0
        and len(ang_ang) == len(EXPECTED_ANGULAR)
    )

    dh.record(
        "E_checkboxes_gate_emission",
        all_off_ok and angular_only_ok,
        {
            "all_off": {
                "initial": init_off,
                "schedule_len": len(sched_off),
                "angular_len": len(ang_off),
            },
            "angular_only": {
                "initial": init_ang,
                "schedule_len": len(sched_ang),
                "angular_len": len(ang_ang),
            },
        },
    )

    # ----- F: world / custom axis modes -------------------------
    # World X/Y/Z and Custom axes are FIXED world directions, pre-resolved
    # by the encoder into a world-space ω vector (axis swapped Blender->
    # solver via (x,y,z)->(x,z,-y), normalized, scaled by radians(speed)).
    # They land in "angular-velocity-world-schedule" and must NOT appear in
    # the principal-axis "angular-velocity-schedule".
    assigned.velocity_keyframes.clear()
    WORLD_KFS = [
        # frame, axis_mode, custom_vec (Blender), speed_deg
        (1, "X", (0.0, 0.0, 1.0), 90.0),
        (3, "Z", (0.0, 0.0, 1.0), 180.0),
        (5, "CUSTOM", (0.0, 2.0, 0.0), -60.0),
    ]
    for fr, mode, custom, spd in WORLD_KFS:
        kf = assigned.velocity_keyframes.add()
        kf.frame = fr
        kf.angular_axis = mode
        kf.angular_axis_custom = custom
        kf.angular_speed = spd
        kf.enable_translational = False
        kf.enable_angular = True
    assigned.velocity_keyframes_index = 0

    EXPECTED_WORLD = [
        (0.00, (math.radians(90.0), 0.0, 0.0)),     # World X
        (0.02, (0.0, math.radians(180.0), 0.0)),    # World Z -> solver Y
        (0.04, (0.0, 0.0, -math.radians(-60.0))),   # Custom (0,2,0) -> solver (0,0,-1)
    ]
    decoded_w = dh.decode_addon_blob(dh.encoder_param.encode_param(bpy.context))
    gp_w = next(
        params for params, _o, uuids in decoded_w["group"]
        if assigned_uuid in uuids
    )
    world_payload = gp_w.get("angular-velocity-world-schedule", {}).get(
        assigned_uuid, []
    )
    world_actual = [
        (float(t), tuple(float(c) for c in v)) for t, v in world_payload
    ]
    pca_leak = gp_w.get("angular-velocity-schedule", {}).get(assigned_uuid, [])
    world_ok = (
        len(world_actual) == len(EXPECTED_WORLD)
        and len(pca_leak) == 0
        and all(
            abs(world_actual[i][0] - EXPECTED_WORLD[i][0]) < 1e-6
            and all(
                abs(world_actual[i][1][j] - EXPECTED_WORLD[i][1][j]) < 1e-4
                for j in range(3)
            )
            for i in range(len(EXPECTED_WORLD))
        )
    )
    dh.record(
        "F_param_pickle_carries_world_axis_schedule",
        world_ok,
        {
            "world_actual": world_actual,
            "world_expected": EXPECTED_WORLD,
            "pca_leak_len": len(pca_leak),
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
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
