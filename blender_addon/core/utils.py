# File: utils.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore
import numpy as np

from ..models.groups import decode_vertex_group_identifier, get_addon_data, iterate_active_object_groups
from .transform import inv_world_matrix, world_matrix, zup_to_yup


def get_category_name():
    """Get the category name for the add-on."""
    return "ZOZO's Contact Solver"


def count_ngon_faces(obj) -> int:
    """Return the number of N-gon faces (polygons with > 4 vertices)
    on *obj*'s mesh. ``0`` means the mesh has only triangles and
    quads, the only face shapes the solver supports.

    Returns ``0`` for non-mesh objects (curves, etc.) — they don't
    have polygons in the Blender mesh sense.

    Used by:
      * ``OBJECT_OT_AddObjectsToGroup`` to refuse N-gon meshes at
        assignment time, so the user sees an explicit error popup
        rather than silently fan-triangulated geometry on the wire.
      * ``encoder.mesh._build_obj_data`` to fail the Transfer if an
        N-gon got assigned through any path that bypasses the
        operator (older saves, MCP scripts).
    """
    if obj is None or obj.type != "MESH" or obj.data is None:
        return 0
    return sum(1 for p in obj.data.polygons if len(p.vertices) > 4)


def find_linked_duplicate_siblings(obj) -> list[str]:
    """Return the names of other Blender objects that share *obj*'s data
    block (i.e. ``obj`` is a Linked Duplicate / shallow copy, typically
    created by Alt-D).

    The simulator works under the assumption that each object owns its
    own mesh data: shared data means a vertex coordinate written for
    one assigned object would silently propagate to its sibling, which
    in turn would cause the encoder to ship inconsistent geometry,
    corrupt PC2 playback, and break the topology hash. The dynamics
    pipeline rejects these objects up-front.

    Returns an empty list when ``obj`` has its own data block, or when
    ``obj.data`` is None (curves with no spline yet, etc.).
    """
    if obj is None or obj.data is None:
        return []
    # ``users`` counts every reference to the data block, including
    # the active object itself. A solo owner has ``users == 1``.
    if obj.data.users <= 1:
        return []
    return [
        o.name
        for o in bpy.data.objects
        if o is not obj and o.data is obj.data
    ]


def get_timer_wait_time():
    """Get the wait time for the timer."""
    return 0.25


def redraw_all_areas(context):
    """Tag all screen areas for redraw."""
    for area in context.screen.areas:
        area.tag_redraw()


def check_vec3(name: str, v, error_cls) -> tuple[float, float, float]:
    """Coerce `v` to a length-3 tuple of floats, raising `error_cls` on failure.

    Callers plug in their layer's exception type (MCPError, ValidationError,
    MutationError, ...) so the message vocabulary is shared but the error
    class stays specific to the boundary that raised.
    """
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise error_cls(f"{name} must be a length-3 list/tuple, got {v!r}")
    try:
        return tuple(float(x) for x in v)
    except (TypeError, ValueError) as e:
        raise error_cls(f"{name} components must be numeric: {e}")


def parse_vertex_index(data_path: str) -> int | None:
    """Parse the vertex index from a data path string."""
    start = data_path.find("[") + 1
    end = data_path.find("]")
    if start >= 0 and end > start:
        try:
            return int(data_path[start:end])
        except ValueError:
            pass
    return None


def _get_fcurves(action):
    """Get fcurves from an action (Blender 5.0+ layered API)."""
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                if bag.fcurves:
                    return bag.fcurves
    return []


_TRANSFORM_PATHS = (
    "location",
    "rotation_euler",
    "rotation_quaternion",
    "rotation_axis_angle",
    "scale",
)


def has_transform_fcurves(obj) -> bool:
    """True if *obj* has any object-level transform fcurve (loc/rot/scale).

    Used by the static-ops UI and encoder to enforce mutual exclusion:
    a static object with Blender keyframe animation cannot also use
    UI-assigned move/spin/scale ops (only one source of motion at a
    time).
    """
    if obj is None or not hasattr(obj, "animation_data"):
        return False
    ad = obj.animation_data
    if not ad or not ad.action:
        return False
    for fc in _get_fcurves(ad.action):
        path = getattr(fc, "data_path", "") or ""
        if any(path == p or path.endswith(f".{p}") for p in _TRANSFORM_PATHS):
            return True
    return False


def get_vertices_in_group(obj, vg) -> list[int]:
    """Return vertex indices belonging to the given vertex group.

    For MESH objects, reads from Blender vertex groups.
    For CURVE objects, reads from custom property ``_pin_{vg.name}``.

    Args:
        obj: Blender object (MESH or CURVE).
        vg: Blender vertex group (or object with .name for curve lookup).

    Returns:
        List of vertex indices that belong to *vg*.
    """
    if obj.type == "CURVE":
        import json
        key = f"_pin_{vg.name}"
        raw = obj.get(key)
        if raw:
            return json.loads(raw)
        return []
    indices = []
    if not hasattr(obj.data, "vertices"):
        return indices
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == vg.index:
                indices.append(v.index)
                break
    return indices


def set_linear_interpolation(action):
    """Set LINEAR interpolation on all keyframe points in *action*.

    Args:
        action: Blender action containing fcurves.
    """
    for fc in _get_fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def get_moving_vertex_indices(obj, exclude=None) -> list[int]:
    from .pc2 import has_mesh_cache

    if exclude is None:
        exclude = []
    # MESH_CACHE modifier means all vertices are animated
    if obj and obj.type == "MESH" and has_mesh_cache(obj):
        return [i for i in range(len(obj.data.vertices)) if i not in exclude]
    return []


def get_pin_vertex_indices(obj, context, frame: int | None = None) -> list[int]:
    """List vertex indices that are pinned (active) at the given frame.

    Args:
        obj: Blender mesh object.
        context: Blender context.
        frame: Current frame number. If given, pins with duration that have
            expired by this frame are excluded (their vertices are no longer
            considered pinned). If ``None``, all pin vertices are returned
            regardless of duration.
    """
    indices = set()
    if obj and hasattr(obj, "vertex_groups") and hasattr(obj.data, "vertices"):
        pin_vg_names = set()

        from .uuid_registry import get_object_uuid
        _obj_uid = get_object_uuid(obj)
        for group in iterate_active_object_groups(context.scene):
            if hasattr(group, "pin_vertex_groups"):
                from .uuid_registry import resolve_pin
                for pin_item in group.pin_vertex_groups:
                    resolve_pin(pin_item)
                    if pin_item.object_uuid != _obj_uid:
                        continue
                    _, vg_name = decode_vertex_group_identifier(pin_item.name)
                    if vg_name:
                        # Pull pins are not hard-pinned — exclude them
                        if pin_item.use_pull:
                            continue
                        # Pins with explicit operations (spin/scale/move_by)
                        # move during simulation — exclude them
                        if any(op.op_type in ("SPIN", "SCALE", "MOVE_BY", "TORQUE") for op in pin_item.operations):
                            continue
                        # If frame is given, skip expired duration-limited pins
                        if frame is not None and pin_item.use_pin_duration:
                            if frame > pin_item.pin_duration:
                                continue
                        pin_vg_names.add(vg_name)

        for vg_name in pin_vg_names:
            vg = obj.vertex_groups.get(vg_name)
            if vg:
                for idx in get_vertices_in_group(obj, vg):
                    indices.add(idx)

    return list(indices)


def get_transform_keyframes(obj, context, fps: float) -> dict | None:
    """Extract sparse object-level transform keyframes for a STATIC object.

    Only extracts keyframes from object-level animation (location, rotation, scale).
    Raises RuntimeError if the object has mesh-level-only animation (shape keys).

    Args:
        obj: The Blender object.
        context: The Blender context.
        fps: Frames per second for time conversion.

    Returns:
        dict with keys "time", "translation", "quaternion", "scale", "segments",
        or None if no animation.
    """
    if not obj or not hasattr(obj, "animation_data"):
        return None

    has_mesh_anim = (
        obj.data
        and hasattr(obj.data, "animation_data")
        and obj.data.animation_data
        and obj.data.animation_data.action
        and any(_get_fcurves(obj.data.animation_data.action))
    )
    if has_mesh_anim:
        raise RuntimeError(
            f"STATIC object '{obj.name}' has mesh-level animation (shape keys). "
            "Only object-level transform animation is supported for STATIC objects."
        )

    if not obj.animation_data or not obj.animation_data.action:
        return None

    fcurves = _get_fcurves(obj.animation_data.action)
    if not fcurves:
        return None

    keyframe_frames = set()
    for fc in fcurves:
        keyframe_frames.update(int(kp.co[0]) for kp in fc.keyframe_points)
    if not keyframe_frames:
        return None

    sorted_frames = sorted(keyframe_frames)
    scene = context.scene
    current_frame = scene.frame_current

    times = []
    translations = []
    quaternions = []
    scales = []
    # Per-segment interpolation between keyframe[i] and keyframe[i+1].
    # Bezier handles are normalized to the segment's [0,1] time range.
    segments = []

    for frame in sorted_frames:
        scene.frame_set(frame)
        mat = world_matrix(obj)
        loc, quat, scale = mat.decompose()
        times.append((frame - 1) / fps)
        translations.append([float(loc.x), float(loc.y), float(loc.z)])
        quaternions.append([float(quat.w), float(quat.x), float(quat.y), float(quat.z)])
        scales.append([float(scale.x), float(scale.y), float(scale.z)])

    # Extract interpolation info from fcurves (use location X as representative)
    loc_fc = None
    for fc in fcurves:
        if "location" in fc.data_path:
            loc_fc = fc
            break
    if loc_fc is None:
        loc_fc = fcurves[0]

    kp_by_frame = {int(kp.co[0]): kp for kp in loc_fc.keyframe_points}
    for i in range(len(sorted_frames) - 1):
        f0 = sorted_frames[i]
        f1 = sorted_frames[i + 1]
        kp0 = kp_by_frame.get(f0)
        kp1 = kp_by_frame.get(f1)
        interp = "LINEAR"
        handle_right = [1.0 / 3.0, 0.0]
        handle_left = [2.0 / 3.0, 1.0]
        if kp0 is not None:
            interp = kp0.interpolation
            if interp == "BEZIER" and kp1 is not None:
                dt = f1 - f0
                dv = kp1.co[1] - kp0.co[1]
                hr = kp0.handle_right
                hl = kp1.handle_left
                hr_x = float((hr[0] - f0) / dt) if dt > 0 else 1.0 / 3.0
                hl_x = float((hl[0] - f0) / dt) if dt > 0 else 2.0 / 3.0
                if abs(dv) > 1e-10:
                    hr_y = float((hr[1] - kp0.co[1]) / dv)
                    hl_y = float((hl[1] - kp0.co[1]) / dv)
                else:
                    hr_y = 0.0
                    hl_y = 1.0
                handle_right = [hr_x, hr_y]
                handle_left = [hl_x, hl_y]
        segments.append({
            "interpolation": interp,
            "handle_right": handle_right,
            "handle_left": handle_left,
        })

    scene.frame_set(current_frame)

    return {
        "time": times,
        "translation": translations,
        "quaternion": quaternions,
        "scale": scales,
        "segments": segments,
    }
