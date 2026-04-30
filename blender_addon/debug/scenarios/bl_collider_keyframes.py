# File: scenarios/bl_collider_keyframes.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Invisible-collider keyframe encoding.
#
# When a user adds a wall and authors a keyframed timeline of
# positions, ``encode_param`` must emit a ``invisible_colliders``
# entry whose ``walls[0].keyframes`` list mirrors that timeline:
# one record per authored frame, each with ``time = (frame - 1) /
# fps`` and ``position`` axis-swapped via ``_to_solver``. The first
# keyframe always reads from ``item.position`` (the wall's base
# position); subsequent keyframes read from ``kf.position`` unless
# ``use_hold`` is set.
#
# This scenario walks the encode path only -- no build, no run, no
# fetch. We add a wall via the public ``solver.add_wall`` API,
# author keyframes at frames 1, 5, 10, 15, 20, 25, 30, then encode
# and decode the param.pickle to assert the schema and values.
#
# Subtests:
#   A. ``wall_added_to_state``: exactly one wall lives in
#      ``state.invisible_colliders``.
#   B. ``keyframes_authored``: the authored timeline has the
#      expected frame range on the PropertyGroup.
#   C. ``param_pickle_carries_keyframes``: the decoded pickle has
#      ``invisible_colliders.walls[0].keyframes`` with matching
#      times and positions (within float tolerance).

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

    # Cloth scene for context. The encoder iterates active groups
    # for scene_params even when only the wall path is exercised, so
    # a minimal pinned plane keeps that side of encode_param happy.
    plane = dh.reset_scene_to_pinned_plane(name="ColliderKfMesh")
    root = dh.configure_state(project_name="collider_keyframes",
                              frame_count=30, frame_rate=100)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "AllPin")

    # Author the wall timeline. The wall's base position is the
    # frame-1 anchor; the encoder reads it (NOT kf[0].position) for
    # the first keyframe entry. Subsequent ``move_to`` calls write
    # their position to ``kf.position`` and the encoder picks them
    # up as-is.
    base_position = (0.0, 0.0, 0.0)
    timeline = [
        (1, base_position),
        (5, (0.1, 0.0, 0.0)),
        (10, (0.2, 0.0, 0.0)),
        (15, (0.3, 0.0, 0.0)),
        (20, (0.4, 0.0, 0.0)),
        (25, (0.5, 0.0, 0.0)),
        (30, (0.6, 0.0, 0.0)),
    ]
    wall = dh.api.solver.add_wall(position=base_position,
                                  normal=(0.0, 0.0, 1.0))
    for frame, pos in timeline[1:]:
        wall.time(frame).move_to(pos)
    dh.log("wall_authored")

    state = dh.groups.get_addon_data(bpy.context.scene).state
    walls_in_state = [
        c for c in state.invisible_colliders if c.collider_type == "WALL"
    ]

    # ----- A: state has exactly one wall ----------------------
    dh.record(
        "A_wall_added_to_state",
        len(walls_in_state) == 1,
        {
            "n_walls": len(walls_in_state),
            "n_total_colliders": len(state.invisible_colliders),
        },
    )

    # ----- B: authored timeline matches what we asked for ----
    item = walls_in_state[0] if walls_in_state else None
    authored_frames = [int(kf.frame) for kf in item.keyframes] if item else []
    expected_frames = [f for f, _ in timeline]
    dh.record(
        "B_keyframes_authored",
        item is not None
        and authored_frames == expected_frames
        and len(item.keyframes) == len(timeline),
        {
            "authored_frames": authored_frames,
            "expected_frames": expected_frames,
        },
    )

    # ----- C: encoded param.pickle carries the timeline ------
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    data = pickle.loads(param_bytes)

    ic = data.get("invisible_colliders")
    walls = ic.get("walls") if ic else None
    encoded_wall = walls[0] if walls else None
    encoded_kfs = encoded_wall.get("keyframes") if encoded_wall else None

    fps = int(root.state.frame_rate)
    # The encoder applies ``_to_solver`` on each position:
    # Blender (x, y, z) -> solver (x, z, -y).
    expected_kfs = []
    for frame, (x, y, z) in timeline:
        expected_kfs.append({
            "time": float(frame - 1) / fps,
            "position": [float(x), float(z), float(-y)],
        })

    schema_ok = (
        ic is not None
        and isinstance(walls, list)
        and len(walls) == 1
        and isinstance(encoded_kfs, list)
        and len(encoded_kfs) == len(timeline)
    )

    times_ok = positions_ok = False
    if schema_ok:
        times_ok = all(
            abs(float(encoded_kfs[i]["time"]) - expected_kfs[i]["time"])
            < 1e-6
            for i in range(len(timeline))
        )
        positions_ok = all(
            all(
                abs(float(encoded_kfs[i]["position"][j])
                    - expected_kfs[i]["position"][j]) < 1e-5
                for j in range(3)
            )
            for i in range(len(timeline))
        )

    dh.record(
        "C_param_pickle_carries_keyframes",
        schema_ok and times_ok and positions_ok,
        {
            "schema_ok": schema_ok,
            "times_ok": times_ok,
            "positions_ok": positions_ok,
            "n_encoded_walls": len(walls) if isinstance(walls, list) else None,
            "n_encoded_keyframes": (
                len(encoded_kfs) if isinstance(encoded_kfs, list) else None
            ),
            "first_encoded_kf": encoded_kfs[0] if encoded_kfs else None,
            "last_encoded_kf": encoded_kfs[-1] if encoded_kfs else None,
            "expected_first_kf": expected_kfs[0],
            "expected_last_kf": expected_kfs[-1],
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 60.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
