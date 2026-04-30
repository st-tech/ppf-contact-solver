"""Dynamic scene parameter and invisible-collider keyframe handlers."""

from typing import Optional

import bpy  # pyright: ignore

from ...models.collection_utils import safe_update_index, sort_keyframes_by_frame
from ...models.groups import get_addon_data
from ..decorators import MCPError, ValidationError, mcp_handler


_DYN_PARAM_TYPES = {
    "GRAVITY",
    "WIND",
    "AIR_DENSITY",
    "AIR_FRICTION",
    "VERTEX_AIR_DAMP",
}


def _check_vec3(name: str, v):
    from ...core.utils import check_vec3
    return check_vec3(name, v, ValidationError)


def _find_dyn_param(state, param_type: str):
    for i, item in enumerate(state.dyn_params):
        if item.param_type == param_type:
            return i, item
    return -1, None


# ---------------------------------------------------------------------------
# Dynamic parameter entries
# ---------------------------------------------------------------------------


@mcp_handler
def add_dynamic_param(param_type: str):
    """Add a dynamic (time-varying) scene parameter.

    Creates an initial keyframe at frame 1 seeded from the current static
    scene value.

    Args:
        param_type: One of GRAVITY, WIND, AIR_DENSITY, AIR_FRICTION, VERTEX_AIR_DAMP
    """
    if param_type not in _DYN_PARAM_TYPES:
        raise ValidationError(
            f"Invalid param_type '{param_type}'. Must be one of {sorted(_DYN_PARAM_TYPES)}"
        )
    state = get_addon_data(bpy.context.scene).state
    for item in state.dyn_params:
        if item.param_type == param_type:
            raise MCPError(f"{param_type} already added")

    item = state.dyn_params.add()
    item.param_type = param_type
    kf = item.keyframes.add()
    kf.frame = 1
    # Seed the initial keyframe from the static state value so the LLM
    # doesn't silently zero-out the parameter on first add.
    if param_type == "GRAVITY":
        kf.gravity_value = tuple(state.gravity_3d)
    elif param_type == "WIND":
        kf.wind_direction_value = tuple(state.wind_direction)
        kf.wind_strength_value = state.wind_strength
    elif param_type == "AIR_DENSITY":
        kf.scalar_value = state.air_density
    elif param_type == "AIR_FRICTION":
        kf.scalar_value = state.air_friction
    elif param_type == "VERTEX_AIR_DAMP":
        kf.scalar_value = state.vertex_air_damp

    state.dyn_params_index = len(state.dyn_params) - 1
    return {
        "message": f"Added dynamic param {param_type}",
        "param_type": param_type,
        "index": state.dyn_params_index,
        "keyframe_count": len(item.keyframes),
    }


@mcp_handler
def remove_dynamic_param(param_type: str):
    """Remove a dynamic scene parameter entry.

    Args:
        param_type: One of GRAVITY, WIND, AIR_DENSITY, AIR_FRICTION, VERTEX_AIR_DAMP
    """
    state = get_addon_data(bpy.context.scene).state
    idx, _ = _find_dyn_param(state, param_type)
    if idx < 0:
        raise MCPError(f"Dynamic param {param_type} not found")
    state.dyn_params.remove(idx)
    state.dyn_params_index = safe_update_index(idx, len(state.dyn_params))
    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": f"Removed dynamic param {param_type}",
        "param_type": param_type,
    }


@mcp_handler
def list_dynamic_params():
    """List all dynamic scene parameters and their keyframes."""
    state = get_addon_data(bpy.context.scene).state
    out = []
    for item in state.dyn_params:
        keyframes = []
        for kf in item.keyframes:
            entry = {"frame": kf.frame, "use_hold": kf.use_hold}
            if item.param_type == "GRAVITY":
                entry["gravity"] = list(kf.gravity_value)
            elif item.param_type == "WIND":
                entry["wind_direction"] = list(kf.wind_direction_value)
                entry["wind_strength"] = kf.wind_strength_value
            else:
                entry["value"] = kf.scalar_value
            keyframes.append(entry)
        out.append({"param_type": item.param_type, "keyframes": keyframes})
    return {"dynamic_params": out}


@mcp_handler
def add_dynamic_param_keyframe(
    param_type: str,
    frame: int,
    gravity: Optional[list[float]] = None,
    wind_direction: Optional[list[float]] = None,
    wind_strength: Optional[float] = None,
    value: Optional[float] = None,
    use_hold: Optional[bool] = None,
):
    """Add a keyframe to a dynamic scene parameter.

    Supply the field matching the param_type (gravity for GRAVITY; wind_direction
    + wind_strength for WIND; value for the scalar params).

    Args:
        param_type: GRAVITY, WIND, AIR_DENSITY, AIR_FRICTION, or VERTEX_AIR_DAMP
        frame: Blender frame (>= 1)
        gravity: [x, y, z] for GRAVITY param
        wind_direction: [x, y, z] for WIND param
        wind_strength: Scalar speed (m/s) for WIND param
        value: Scalar for AIR_DENSITY, AIR_FRICTION, or VERTEX_AIR_DAMP
        use_hold: Hold previous keyframe value (step function)
    """
    if frame < 1:
        raise ValidationError("frame must be >= 1")
    state = get_addon_data(bpy.context.scene).state
    idx, item = _find_dyn_param(state, param_type)
    if item is None:
        raise MCPError(f"Dynamic param {param_type} not found; call add_dynamic_param first")

    for kf in item.keyframes:
        if kf.frame == frame:
            raise MCPError(f"Keyframe at frame {frame} already exists")

    kf = item.keyframes.add()
    kf.frame = frame
    if param_type == "GRAVITY":
        if gravity is None:
            raise ValidationError("GRAVITY keyframe requires 'gravity' argument")
        kf.gravity_value = _check_vec3("gravity", gravity)
    elif param_type == "WIND":
        if wind_direction is not None:
            kf.wind_direction_value = _check_vec3("wind_direction", wind_direction)
        if wind_strength is not None:
            kf.wind_strength_value = float(wind_strength)
    else:
        if value is None:
            raise ValidationError(f"{param_type} keyframe requires 'value' argument")
        kf.scalar_value = float(value)

    if use_hold is not None:
        kf.use_hold = bool(use_hold)

    item.keyframes_index = sort_keyframes_by_frame(item.keyframes)
    state.dyn_params_index = idx
    return {
        "message": f"Added keyframe at frame {frame} for {param_type}",
        "param_type": param_type,
        "frame": frame,
        "keyframe_count": len(item.keyframes),
    }


@mcp_handler
def remove_dynamic_param_keyframe(param_type: str, frame: int):
    """Remove a keyframe from a dynamic scene parameter.

    The initial keyframe (frame 1) cannot be removed.

    Args:
        param_type: GRAVITY, WIND, AIR_DENSITY, AIR_FRICTION, or VERTEX_AIR_DAMP
        frame: Frame number of the keyframe to remove
    """
    state = get_addon_data(bpy.context.scene).state
    _, item = _find_dyn_param(state, param_type)
    if item is None:
        raise MCPError(f"Dynamic param {param_type} not found")
    for i, kf in enumerate(item.keyframes):
        if kf.frame == frame:
            if i == 0:
                raise MCPError("Cannot remove the initial keyframe")
            item.keyframes.remove(i)
            item.keyframes_index = safe_update_index(i, len(item.keyframes))
            from ...models.groups import invalidate_overlays
            invalidate_overlays()
            return {
                "message": f"Removed keyframe at frame {frame}",
                "param_type": param_type,
                "frame": frame,
                "keyframe_count": len(item.keyframes),
            }
    raise MCPError(f"No keyframe at frame {frame} for {param_type}")


# ---------------------------------------------------------------------------
# Invisible collider keyframes + property updates
# ---------------------------------------------------------------------------


def _get_collider(index: int):
    state = get_addon_data(bpy.context.scene).state
    n = len(state.invisible_colliders)
    if index < 0 or index >= n:
        raise ValidationError(f"Collider index {index} out of range (0..{n - 1})")
    return state, state.invisible_colliders[index]


@mcp_handler
def set_collider_properties(
    index: int,
    name: Optional[str] = None,
    position: Optional[list[float]] = None,
    normal: Optional[list[float]] = None,
    radius: Optional[float] = None,
    contact_gap: Optional[float] = None,
    friction: Optional[float] = None,
    thickness: Optional[float] = None,
    invert: Optional[bool] = None,
    hemisphere: Optional[bool] = None,
    enable_active_duration: Optional[bool] = None,
    active_duration: Optional[int] = None,
):
    """Update properties on an invisible collider.

    Pass only the fields you want to change. `normal` is wall-only;
    `radius`/`invert`/`hemisphere` are sphere-only.

    Args:
        index: Zero-based collider index as reported by list_invisible_colliders
        name: Display name
        position: [x, y, z] origin
        normal: [x, y, z] outward normal (WALL only)
        radius: Sphere radius (SPHERE only)
        contact_gap: Contact gap tolerance
        friction: Friction coefficient [0, 1]
        thickness: Max penetration depth (> 0)
        invert: Flip contact direction (SPHERE only)
        hemisphere: Restrict to upper half (SPHERE only)
        enable_active_duration: Enable per-collider active-until frame
        active_duration: First frame the collider is no longer active
    """
    state, col = _get_collider(index)

    if normal is not None and col.collider_type != "WALL":
        raise ValidationError("'normal' is valid only for WALL colliders")
    if (radius is not None or invert is not None or hemisphere is not None) and col.collider_type != "SPHERE":
        raise ValidationError("'radius', 'invert', and 'hemisphere' are valid only for SPHERE colliders")

    if name is not None:
        col.name = name
    if position is not None:
        col.position = _check_vec3("position", position)
    if normal is not None:
        col.normal = _check_vec3("normal", normal)
    if radius is not None:
        col.radius = float(radius)
    if contact_gap is not None:
        col.contact_gap = float(contact_gap)
    if friction is not None:
        col.friction = float(friction)
    if thickness is not None:
        col.thickness = float(thickness)
    if invert is not None:
        col.invert = bool(invert)
    if hemisphere is not None:
        col.hemisphere = bool(hemisphere)
    if enable_active_duration is not None:
        col.enable_active_duration = bool(enable_active_duration)
    if active_duration is not None:
        col.active_duration = int(active_duration)

    return {
        "message": f"Updated collider {index}",
        "index": index,
        "collider_type": col.collider_type,
    }


@mcp_handler
def add_collider_keyframe(
    index: int,
    frame: int,
    position: Optional[list[float]] = None,
    radius: Optional[float] = None,
    use_hold: Optional[bool] = None,
):
    """Add a keyframe to an invisible collider.

    Args:
        index: Zero-based collider index
        frame: Blender frame (>= 1)
        position: [x, y, z] at this keyframe
        radius: Sphere radius at this keyframe (SPHERE only)
        use_hold: Hold the previous keyframe value (step function)
    """
    if frame < 1:
        raise ValidationError("frame must be >= 1")
    _, col = _get_collider(index)
    for kf in col.keyframes:
        if kf.frame == frame:
            raise MCPError(f"Keyframe at frame {frame} already exists on collider {index}")

    kf = col.keyframes.add()
    kf.frame = frame
    if position is not None:
        kf.position = _check_vec3("position", position)
    else:
        kf.position = tuple(col.position)
    if radius is not None:
        if col.collider_type != "SPHERE":
            raise ValidationError("radius keyframe is valid only on SPHERE colliders")
        kf.radius = float(radius)
    else:
        kf.radius = float(col.radius)
    if use_hold is not None:
        kf.use_hold = bool(use_hold)

    col.keyframes_index = sort_keyframes_by_frame(col.keyframes)
    return {
        "message": f"Added keyframe at frame {frame} on collider {index}",
        "index": index,
        "frame": frame,
        "keyframe_count": len(col.keyframes),
    }


@mcp_handler
def remove_collider_keyframe(index: int, frame: int):
    """Remove a keyframe from an invisible collider.

    Args:
        index: Zero-based collider index
        frame: Frame number of the keyframe to remove
    """
    _, col = _get_collider(index)
    for i, kf in enumerate(col.keyframes):
        if kf.frame == frame:
            col.keyframes.remove(i)
            col.keyframes_index = safe_update_index(i, len(col.keyframes))
            from ...models.groups import invalidate_overlays
            invalidate_overlays()
            return {
                "message": f"Removed keyframe at frame {frame} from collider {index}",
                "index": index,
                "frame": frame,
                "keyframe_count": len(col.keyframes),
            }
    raise MCPError(f"No keyframe at frame {frame} on collider {index}")


@mcp_handler
def list_collider_keyframes(index: int):
    """List keyframes on an invisible collider.

    Args:
        index: Zero-based collider index
    """
    _, col = _get_collider(index)
    keyframes = [
        {
            "frame": kf.frame,
            "position": list(kf.position),
            "radius": kf.radius,
            "use_hold": kf.use_hold,
        }
        for kf in col.keyframes
    ]
    return {"index": index, "collider_type": col.collider_type, "keyframes": keyframes}
