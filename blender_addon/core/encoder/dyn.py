# File: encoder/dyn.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from . import _normalize_and_scale, _swap_axes, _to_solver


_DYN_PARAM_SOLVER_KEYS = {
    "GRAVITY": "gravity",
    "WIND": "wind",
    "AIR_DENSITY": "air-density",
    "AIR_FRICTION": "air-friction",
    "VERTEX_AIR_DAMP": "isotropic-air-friction",
}


# "Active Until frame N" means frame < N is active, frame >= N is inactive.
# The cutoff sits half a frame before N's time: 1 frame to map the 1-based
# "Active Until N" to a < N boundary, plus half a frame margin so all substeps
# of the transition TO frame N see the collider as off. The displayed state at
# frame N carries no residual collider effect, and f32 drift at the boundary
# can't leak an extra frame through.
_COLLIDER_CUTOFF_MARGIN_FRAMES = 1.5


def _active_duration_cutoff(item, fps):
    """Encode a collider's active-duration cutoff in seconds.

    Shared by wall and sphere encoding so the half-frame boundary margin
    (see _COLLIDER_CUTOFF_MARGIN_FRAMES) stays identical for both kinds.
    Returns -1.0 when the collider has no active-duration limit.
    """
    if not item.enable_active_duration:
        return -1.0
    return max(0.0, (float(item.active_duration) - _COLLIDER_CUTOFF_MARGIN_FRAMES) / fps)


def _encode_dyn_params(state, fps):
    """Encode dynamic scene parameters as dyn_param dict.

    Returns:
        dict mapping solver param key to list of (time_seconds, value_list) entries.
    """
    dyn_param = {}
    for dyn_item in state.dyn_params:
        solver_key = _DYN_PARAM_SOLVER_KEYS.get(dyn_item.param_type)
        if solver_key is None or len(dyn_item.keyframes) < 2:
            continue

        entries = []
        for i, kf in enumerate(dyn_item.keyframes):
            time_seconds = float(kf.frame - 1) / fps

            if i == 0:
                # Frame 1: read from global State params
                if dyn_item.param_type == "GRAVITY":
                    value = _swap_axes(state.gravity_3d)
                elif dyn_item.param_type == "WIND":
                    value = _swap_axes(_normalize_and_scale(state.wind_direction, state.wind_strength))
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
                    value = _swap_axes(kf.gravity_value)
                elif dyn_item.param_type == "WIND":
                    value = _swap_axes(_normalize_and_scale(kf.wind_direction_value, kf.wind_strength_value))
                else:
                    value = [float(kf.scalar_value)]

            is_hold = kf.use_hold if i > 0 else False
            entries.append((time_seconds, value, is_hold))

        if entries:
            dyn_param[solver_key] = entries

    return dyn_param


def _encode_invisible_colliders(state, fps):
    """Encode invisible colliders as a dict for the CBOR scene payload.

    Returns:
        dict with "walls" and "spheres" lists, or None if empty.
    """
    result = {"walls": [], "spheres": []}
    for item in state.invisible_colliders:
        if item.collider_type == "WALL":
            wall = {
                "position": _to_solver(item.position),
                "normal": _swap_axes(item.normal),
                "contact_gap": float(item.contact_gap),
                "friction": float(item.friction),
                "thickness": float(item.thickness),
                # Half-frame boundary margin shared with sphere encoding; see
                # _active_duration_cutoff / _COLLIDER_CUTOFF_MARGIN_FRAMES.
                "active_duration": _active_duration_cutoff(item, fps),
                "keyframes": [],
            }
            for i, kf in enumerate(item.keyframes):
                time_seconds = float(kf.frame - 1) / fps
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
                "contact_gap": float(item.contact_gap),
                "friction": float(item.friction),
                "thickness": float(item.thickness),
                # Same half-frame boundary margin as walls; see
                # _active_duration_cutoff / _COLLIDER_CUTOFF_MARGIN_FRAMES.
                "active_duration": _active_duration_cutoff(item, fps),
                "keyframes": [],
            }
            for i, kf in enumerate(item.keyframes):
                time_seconds = float(kf.frame - 1) / fps
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
