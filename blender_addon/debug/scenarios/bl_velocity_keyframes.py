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

    # ----- Encode params and decode the pickle ------------------
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = pickle.loads(param_bytes)
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
