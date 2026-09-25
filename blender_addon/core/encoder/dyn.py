# File: encoder/dyn.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from . import (
    _swap_axes,
    _to_solver,
    frame_to_time,
    solver_gravity,
    solver_wind,
)


_DYN_PARAM_SOLVER_KEYS = {
    "GRAVITY": "gravity",
    "WIND": "wind",
    "AIR_DENSITY": "air-density",
    "AIR_FRICTION": "air-friction",
    "VERTEX_AIR_DAMP": "isotropic-air-friction",
}


# "Active Until frame N" means frame < N is active, frame >= N is inactive.
# The cutoff sits half a frame before N's time, so all substeps of the
# transition TO frame N see the collider as off. The displayed state at frame N
# carries no residual collider effect, and f32 drift at the boundary can't leak
# an extra frame through.
_COLLIDER_CUTOFF_MARGIN_FRAMES = 0.5


def _active_duration_cutoff(item, fps, start_frame):
    """Encode a collider's active-until cutoff in seconds.

    Despite the property's name, ``active_duration`` is an absolute Blender
    frame ("Active Until (frame)"), not a duration, so it goes through the same
    frame-to-time conversion as every keyframe: the starting frame is the
    origin. Getting that wrong keeps the collider alive for the whole solve
    whenever the solve does not start at frame 1, and puts the viewport overlay
    (which compares against the raw frame) out of step with the run.

    Shared by wall and sphere encoding so the half-frame boundary margin
    (see _COLLIDER_CUTOFF_MARGIN_FRAMES) stays identical for both kinds.
    Returns -1.0 when the collider has no active-duration limit.

    A cutoff at or before the starting frame is refused by name: the collider
    would never act in the simulation. Its active span runs from the starting
    frame to Active Until, so that is a window that ends no later than it
    starts, the shape ``check_frame_window`` refuses for operations.
    """
    if not item.enable_active_duration:
        return -1.0
    if int(item.active_duration) <= int(start_frame):
        raise ValueError(
            f"Invisible collider '{item.name}' is Active Until frame "
            f"{int(item.active_duration)}, which is not after the starting "
            f"frame {int(start_frame)}, so it would never act in this "
            f"simulation. Set Active Until after frame {int(start_frame)}, "
            "turn Active Duration off, or start the simulation before frame "
            f"{int(item.active_duration)}."
        )
    cutoff_frame = float(item.active_duration) - _COLLIDER_CUTOFF_MARGIN_FRAMES
    return frame_to_time(cutoff_frame, fps, start_frame)


def _keyframe_times(owner, keyframes, fps, start_frame):
    """Seconds for each keyframe of a list whose first entry is the initial one.

    The initial keyframe (index 0) is the state at the starting frame, which
    is simulated time zero, whatever frame it is stored at: its value comes
    from the owner's own settings rather than from the keyframe, and the
    frontend creates the owner in that state and never reads the entry's time.
    The viewport overlay places it at the starting frame for the same reason.

    Every later keyframe is refused by name when the frontend could not place
    it. One at or before the starting frame would land at or before the
    initial keyframe's instant, and one at or before the keyframe listed ahead
    of it would run time backward; either way the frontend stops with a
    time-ordering error that names nothing. ``owner`` names the collider or
    parameter in the artist's terms.
    """
    start_frame = int(start_frame)
    times = []
    prev_frame = None
    for i, kf in enumerate(keyframes):
        if i == 0:
            times.append(0.0)
            continue
        frame = int(kf.frame)
        if frame <= start_frame:
            raise ValueError(
                f"{owner} has a keyframe at frame {frame}, at or before the "
                f"starting frame {start_frame}. Its initial keyframe already "
                "holds the state at the starting frame, so every later "
                f"keyframe has to come after frame {start_frame}: move this "
                f"one past it, or start the simulation before frame {frame}."
            )
        if prev_frame is not None and frame <= prev_frame:
            raise ValueError(
                f"{owner} has a keyframe at frame {frame} listed after the "
                f"keyframe at frame {prev_frame}, and each keyframe has to "
                "come at a later frame than the one listed before it. Change "
                "one of the two frames."
            )
        times.append(frame_to_time(frame, fps, start_frame))
        prev_frame = frame
    return times


def _encode_dyn_params(state, fps, start_frame):
    """Encode dynamic scene parameters as dyn_param dict.

    ``start_frame`` is the Blender frame that is simulated time zero (see
    ``resolve_start_frame``); keyframe times are relative to it.

    Gravity and wind take the same transform as the static settings
    (``solver_gravity`` / ``solver_wind``).

    Returns:
        dict mapping solver param key to list of (time_seconds, value_list) entries.
    """
    dyn_param = {}
    for dyn_item in state.dyn_params:
        solver_key = _DYN_PARAM_SOLVER_KEYS.get(dyn_item.param_type)
        if solver_key is None or len(dyn_item.keyframes) < 2:
            continue

        times = _keyframe_times(
            f"Dynamic parameter {dyn_item.param_type}", dyn_item.keyframes,
            fps, start_frame,
        )
        entries = []
        for i, kf in enumerate(dyn_item.keyframes):
            time_seconds = times[i]

            if i == 0:
                # First keyframe (t=0): read from global State params
                if dyn_item.param_type == "GRAVITY":
                    value = solver_gravity(state.gravity_3d)
                elif dyn_item.param_type == "WIND":
                    value = solver_wind(
                        state.wind_direction, state.wind_strength, "Wind")
                elif dyn_item.param_type == "AIR_DENSITY":
                    value = [float(state.air_density)]
                elif dyn_item.param_type == "AIR_FRICTION":
                    value = [float(state.air_friction)]
                elif dyn_item.param_type == "VERTEX_AIR_DAMP":
                    value = [float(state.vertex_air_damp)]
                else:
                    continue
            else:
                if kf.use_hold:
                    # Hold: repeat the previous keyframe's value
                    value = entries[-1][1] if entries else [0.0]
                elif dyn_item.param_type == "GRAVITY":
                    value = solver_gravity(kf.gravity_value)
                elif dyn_item.param_type == "WIND":
                    value = solver_wind(
                        kf.wind_direction_value, kf.wind_strength_value,
                        f"The wind keyframe at frame {kf.frame}")
                else:
                    value = [float(kf.scalar_value)]

            is_hold = kf.use_hold if i > 0 else False
            entries.append((time_seconds, value, is_hold))

        if entries:
            dyn_param[solver_key] = entries

    return dyn_param


def _encode_invisible_colliders(state, fps, start_frame):
    """Encode invisible colliders as a dict for the CBOR PARAM payload (lands in param.pickle via _build_param_dict, stays seconds).

    ``start_frame`` is the Blender frame that is simulated time zero (see
    ``resolve_start_frame``); keyframe times are relative to it.

    Returns:
        dict with "walls" and "spheres" lists, or None if empty.
    """
    result = {"walls": [], "spheres": []}
    # A collider's contact gap is authored as a world-space distance, like a
    # group's absolute gap (params.py scales that one). The solver scales the
    # collider's position, radius and thickness by world_scaling on ingest and
    # never scales a gap, so the gap is scaled here and only here, or a wall
    # at World Scaling 0.1 holds cloth ten times its authored gap away.
    gap_scale = float(state.world_scaling)
    for item in state.invisible_colliders:
        if item.collider_type == "WALL":
            wall = {
                "position": _to_solver(item.position),
                "normal": _swap_axes(item.normal),
                "contact_gap": float(item.contact_gap) * gap_scale,
                "friction": float(item.friction),
                "thickness": float(item.thickness),
                # Half-frame boundary margin shared with sphere encoding; see
                # _active_duration_cutoff / _COLLIDER_CUTOFF_MARGIN_FRAMES.
                "active_duration": _active_duration_cutoff(item, fps, start_frame),
                "keyframes": [],
            }
            times = _keyframe_times(
                f"Invisible collider '{item.name}'", item.keyframes,
                fps, start_frame,
            )
            for i, kf in enumerate(item.keyframes):
                time_seconds = times[i]
                if i == 0:
                    pos = _to_solver(item.position)
                elif kf.use_hold and wall["keyframes"]:
                    pos = wall["keyframes"][-1]["position"]
                else:
                    pos = _to_solver(kf.position)
                wall["keyframes"].append({"time": time_seconds, "position": pos})
            result["walls"].append(wall)
        elif item.collider_type == "SPHERE":
            sphere = {
                "position": _to_solver(item.position),
                "radius": float(item.radius),
                "hemisphere": bool(item.hemisphere),
                "invert": bool(item.invert),
                "contact_gap": float(item.contact_gap) * gap_scale,
                "friction": float(item.friction),
                "thickness": float(item.thickness),
                # Same half-frame boundary margin as walls; see
                # _active_duration_cutoff / _COLLIDER_CUTOFF_MARGIN_FRAMES.
                "active_duration": _active_duration_cutoff(item, fps, start_frame),
                "keyframes": [],
            }
            times = _keyframe_times(
                f"Invisible collider '{item.name}'", item.keyframes,
                fps, start_frame,
            )
            for i, kf in enumerate(item.keyframes):
                time_seconds = times[i]
                if i == 0:
                    pos = _to_solver(item.position)
                    r = float(item.radius)
                elif kf.use_hold and sphere["keyframes"]:
                    prev = sphere["keyframes"][-1]
                    pos, r = prev["position"], prev["radius"]
                else:
                    pos = _to_solver(kf.position)
                    r = float(kf.radius)
                sphere["keyframes"].append({
                    "time": time_seconds, "position": pos, "radius": r,
                })
            result["spheres"].append(sphere)
    if result["walls"] or result["spheres"]:
        return result
    return None
