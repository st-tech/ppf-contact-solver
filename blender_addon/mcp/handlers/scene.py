# File: handlers/scene.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP handlers for scene-level setup: invisible colliders, merge pairs,
# and snap-to-vertices.  Thin adapters over ``core.mutation``: validate
# input via schema, translate ``MutationError`` into ``MCPError``,
# return a result dict.  All business logic and locking is in the
# mutation module; this file is a one-line-per-call translation.

import bpy  # pyright: ignore

from ..decorators import MCPError, mcp_handler
from ...core import mutation


def _call(fn, *args, **kwargs):
    """Run a core.mutation call, surfacing MutationError as MCPError."""
    try:
        return fn(*args, **kwargs)
    except mutation.MutationError as e:
        raise MCPError(str(e))


@mcp_handler
def clear_solver():
    """Reset the entire solver state to defaults."""
    from ...ops.api import solver as solver_api

    solver_api.clear()
    return "Solver state cleared"


# ---------------------------------------------------------------------------
# Invisible colliders
# ---------------------------------------------------------------------------


@mcp_handler
def add_invisible_wall(position: list[float], normal: list[float]):
    """Add an invisible wall collider at a given position and normal.

    Args:
        position: Wall origin in Blender world space [x, y, z].
        normal: Outward-facing normal vector [x, y, z].
    """
    _call(mutation.add_invisible_wall, position, normal)
    return "Invisible wall added"


@mcp_handler
def add_invisible_sphere(
    position: list[float],
    radius: float,
    invert: bool = False,
    hemisphere: bool = False,
):
    """Add an invisible sphere collider.

    Args:
        position: Center in Blender world space [x, y, z].
        radius: Sphere radius.
        invert: If true, acts as an inverted sphere (contact from inside).
        hemisphere: If true, only the upper half acts as a collider.
    """
    _call(
        mutation.add_invisible_sphere,
        position,
        radius,
        invert=invert,
        hemisphere=hemisphere,
    )
    return "Invisible sphere added"


@mcp_handler
def list_invisible_colliders():
    """Return a list of all invisible colliders currently in the scene."""
    out = []
    for item in bpy.context.scene.zozo_contact_solver.state.invisible_colliders:
        entry = {
            "index": len(out),
            "type": item.collider_type,
            "name": item.name,
            "position": list(item.position),
            "contact_gap": float(item.contact_gap),
            "friction": float(item.friction),
            "thickness": float(item.thickness),
        }
        if item.collider_type == "WALL":
            entry["normal"] = list(item.normal)
        else:
            entry["radius"] = float(item.radius)
            entry["invert"] = bool(item.invert)
            entry["hemisphere"] = bool(item.hemisphere)
        out.append(entry)
    return {"colliders": out}


@mcp_handler
def remove_invisible_collider(index: int):
    """Remove an invisible collider by its index in the scene list.

    Args:
        index: Zero-based index as reported by list_invisible_colliders.
    """
    _call(mutation.remove_invisible_collider, index)
    return f"Removed collider at index {index}"


@mcp_handler
def clear_invisible_colliders():
    """Remove every invisible collider from the scene."""
    _call(mutation.clear_invisible_colliders)
    return "All invisible colliders cleared"


# ---------------------------------------------------------------------------
# Merge pairs (cross-object stitching)
# ---------------------------------------------------------------------------


def _resolve_name_to_uuid(object_name: str) -> tuple[str, str]:
    """Resolve caller-supplied object name to (name, uuid) once.

    The LLM passes names; internally we carry UUID+name side-by-side so
    later identity checks never rely on the display name.
    """
    from ...core.uuid_registry import get_or_create_object_uuid

    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise MCPError(f"Object '{object_name}' not found in scene")
    uid = get_or_create_object_uuid(obj)
    if not uid:
        raise MCPError(f"Object '{object_name}' is library-linked and not writable")
    return obj.name, uid


@mcp_handler
def add_merge_pair(object_a: str, object_b: str):
    """Stitch two objects together along their nearest overlapping vertices.

    Args:
        object_a: Name of the source object.
        object_b: Name of the target object.
    """
    name_a, uuid_a = _resolve_name_to_uuid(object_a)
    name_b, uuid_b = _resolve_name_to_uuid(object_b)
    _call(mutation.add_merge_pair, name_a, name_b)
    return {
        "message": f"Merge pair added: {name_a} <-> {name_b}",
        "object_a": name_a,
        "object_b": name_b,
        "object_a_uuid": uuid_a,
        "object_b_uuid": uuid_b,
    }


@mcp_handler
def remove_merge_pair(object_a: str, object_b: str):
    """Remove a merge pair by the two object names.

    Args:
        object_a: Name of the source object.
        object_b: Name of the target object.
    """
    name_a, uuid_a = _resolve_name_to_uuid(object_a)
    name_b, uuid_b = _resolve_name_to_uuid(object_b)
    _call(mutation.remove_merge_pair, name_a, name_b)
    return {
        "message": f"Merge pair removed: {name_a} <-> {name_b}",
        "object_a": name_a,
        "object_b": name_b,
        "object_a_uuid": uuid_a,
        "object_b_uuid": uuid_b,
    }


@mcp_handler
def list_merge_pairs():
    """Return every stored merge pair with its names, UUIDs and stitch state.

    ``stitch_row_count`` is how many stitch anchors the pair currently
    carries. A pair reporting 0 contributes no stitch to the solve whatever
    its ``stitch_stiffness``, and resnap_merge_pair rebuilds the anchors.
    Anchors invalidated by a later mesh edit are dropped when the scene is
    encoded, so a nonzero count reports what is stored rather than a fresh
    check against the current topology.
    """
    from ...mesh_ops.merge_ops import pair_stitch_row_count
    from ...models.groups import get_addon_data

    state = get_addon_data(bpy.context.scene).state
    pairs = [
        {
            "object_a": p.object_a,
            "object_b": p.object_b,
            "object_a_uuid": p.object_a_uuid,
            "object_b_uuid": p.object_b_uuid,
            "stitch_stiffness": float(p.stitch_stiffness),
            "show_stitch": bool(p.show_stitch),
            "stitch_row_count": pair_stitch_row_count(p),
        }
        for p in state.merge_pairs
    ]
    return {"pairs": pairs}


@mcp_handler
def clear_merge_pairs():
    """Remove every merge pair from the scene."""
    _call(mutation.clear_merge_pairs)
    return "All merge pairs cleared"


# ---------------------------------------------------------------------------
# Snap
# ---------------------------------------------------------------------------


@mcp_handler
def snap_to_vertices(object_a: str, object_b: str):
    """Move object A so its nearest vertex matches object B's nearest vertex.

    Args:
        object_a: Name of the object that will move.
        object_b: Name of the target object (stays put).
    """
    name_a, uuid_a = _resolve_name_to_uuid(object_a)
    name_b, uuid_b = _resolve_name_to_uuid(object_b)
    _call(mutation.snap_to_vertices, name_a, name_b)
    return {
        "message": f"Snapped {name_a} to {name_b}",
        "object_a": name_a,
        "object_b": name_b,
        "object_a_uuid": uuid_a,
        "object_b_uuid": uuid_b,
    }


# ---------------------------------------------------------------------------
# Baking (animation export to Blender keyframes)
# ---------------------------------------------------------------------------


@mcp_handler
def bake_all_animation():
    """Bake simulated animation for every dynamic group to Blender keyframes."""
    # Call via bpy.ops so the operator's modal/confirm flow runs in the
    # correct context; the operator itself skips the confirm dialog when
    # invoked programmatically ('EXEC_DEFAULT').
    bpy.ops.solver.bake_all_animation("EXEC_DEFAULT")
    return "Bake-all initiated"


@mcp_handler
def bake_all_single_frame():
    """Bake the current frame as frame 1 for every dynamic group."""
    bpy.ops.solver.bake_all_single_frame("EXEC_DEFAULT")
    return "Bake-all-single-frame initiated"


# ---------------------------------------------------------------------------
# Scene enumeration
# ---------------------------------------------------------------------------


def _geometry_counts(obj) -> dict:
    """Vertex and face counts for the object types the solver accepts."""
    data = getattr(obj, "data", None)
    if obj.type == "MESH" and data is not None:
        return {
            "vertex_count": len(data.vertices),
            "face_count": len(data.polygons),
        }
    if obj.type == "CURVE" and data is not None:
        return {
            "spline_count": len(data.splines),
            "point_count": sum(
                len(spline.points) or len(spline.bezier_points)
                for spline in data.splines
            ),
        }
    return {}


@mcp_handler
def get_scene_info():
    """Enumerate the current Blender scene: objects, frame range, and groups.

    This is the starting point for an agent that did not create the scene: it
    reports what is in the file and which objects are already assigned to a
    dynamics group, so the caller can tell setup work that remains from work
    already done.

    Returns the scene's frame range as Blender holds it, alongside the
    simulation frame count and fps the solver will actually use, which are
    separate values and are resolved differently.
    """
    from ...core.encoder import resolve_fps
    from ...models.groups import get_addon_data, iterate_active_object_groups

    scene = bpy.context.scene
    if scene is None:
        raise MCPError("No active Blender scene")

    state = get_addon_data(scene).state

    assigned_names: set[str] = set()
    groups = []
    for group in iterate_active_object_groups(scene):
        members = [assigned.name for assigned in group.assigned_objects]
        assigned_names.update(members)
        groups.append(
            {
                "name": group.name,
                "uuid": group.uuid or "",
                "object_type": group.object_type,
                "object_names": members,
            }
        )

    objects = []
    for obj in scene.objects:
        entry = {
            "name": obj.name,
            "type": obj.type,
            "visible": bool(obj.visible_get()),
            "in_group": obj.name in assigned_names,
        }
        entry.update(_geometry_counts(obj))
        objects.append(entry)
    objects.sort(key=lambda item: item["name"])

    return {
        "scene": {
            "name": scene.name,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "frame_current": scene.frame_current,
            "blender_fps": scene.render.fps,
        },
        "simulation": {
            "frame_count": state.frame_count,
            "fps": resolve_fps(state),
        },
        "groups": groups,
        "group_count": len(groups),
        "objects": objects,
        "object_count": len(objects),
    }


# ---------------------------------------------------------------------------
# Merge pair stitch settings
# ---------------------------------------------------------------------------


def _resolve_merge_pair(object_a: str, object_b: str):
    """Return the stored merge pair that joins two named objects.

    The stored pair is matched by UUID in either direction, so the caller may
    name the two objects in the order opposite to the one the pair carries.
    """
    from ...models.groups import get_addon_data

    name_a, uuid_a = _resolve_name_to_uuid(object_a)
    name_b, uuid_b = _resolve_name_to_uuid(object_b)
    state = get_addon_data(bpy.context.scene).state
    for pair in state.merge_pairs:
        if (pair.object_a_uuid == uuid_a and pair.object_b_uuid == uuid_b) or (
            pair.object_a_uuid == uuid_b and pair.object_b_uuid == uuid_a
        ):
            return pair
    raise MCPError(f"No merge pair exists between '{name_a}' and '{name_b}'")


def _checked_stitch_stiffness(value) -> float:
    """Validate a caller-supplied stitch stiffness before it is written.

    ``MergePairItem.stitch_stiffness`` declares ``min=0.0``, and Blender
    clamps a write below a property's minimum without reporting it, so a
    negative value would land as 0.0 and read back as a number the caller
    never asked for.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MCPError(f"stitch_stiffness must be a number, got {value!r}")
    number = float(value)
    if number < 0.0:
        raise MCPError(f"stitch_stiffness must be 0 or greater, got {number}")
    return number


@mcp_handler
def set_merge_pair_properties(
    object_a: str,
    object_b: str,
    stitch_stiffness: float | None = None,
    show_stitch: bool | None = None,
):
    """Set one merge pair's own stitch stiffness and stitch visualization.

    This ``stitch_stiffness`` belongs to the PAIR and is a separate solver
    input from the group parameter of the same name that
    set_group_material_properties writes: the solver scales this pair's
    stitch gradient and Hessian by it directly, with no mass or dt
    normalization, so raise it to hold this one seam harder.

    The value reaches the solver only through the stitch anchors captured at
    snap time, so it stays inert on a pair whose ``stitch_row_count`` (see
    list_merge_pairs) is 0; call resnap_merge_pair to build the anchors. An
    argument left out is not written.

    Args:
        object_a: Name of one object in the pair.
        object_b: Name of the other object in the pair, in either order.
        stitch_stiffness: Stiffness of this pair's stitch, 0 or greater.
        show_stitch: Draw this pair's stitch in the viewport.
    """
    if stitch_stiffness is None and show_stitch is None:
        raise MCPError("Nothing to set: pass stitch_stiffness, show_stitch, or both")
    # Validate every argument before writing any of them, so a rejected value
    # cannot leave the pair half updated.
    if show_stitch is not None and not isinstance(show_stitch, bool):
        raise MCPError(f"show_stitch must be a boolean, got {show_stitch!r}")
    stiffness = (
        None
        if stitch_stiffness is None
        else _checked_stitch_stiffness(stitch_stiffness)
    )

    pair = _resolve_merge_pair(object_a, object_b)
    updates = {}
    if stiffness is not None:
        pair.stitch_stiffness = stiffness
        # Report what Blender stored: the property is single precision.
        updates["stitch_stiffness"] = float(pair.stitch_stiffness)
    if show_stitch is not None:
        pair.show_stitch = show_stitch
        updates["show_stitch"] = bool(pair.show_stitch)

    from ...mesh_ops.merge_ops import pair_stitch_row_count

    return {
        "message": f"Merge pair updated: {pair.object_a} <-> {pair.object_b}",
        "object_a": pair.object_a,
        "object_b": pair.object_b,
        "object_a_uuid": pair.object_a_uuid,
        "object_b_uuid": pair.object_b_uuid,
        "updates": updates,
        "stitch_row_count": pair_stitch_row_count(pair),
    }


class _SnapReports:
    """Collects what ``_snap_pair`` reports, standing in for an operator.

    ``mesh_ops.snap_ops._snap_pair`` returns only an operator status set; the
    reason for a status goes to ``report(level, message)``, which the UI shows
    in the status bar. A tool call has no operator to receive those, so they
    are kept here and the ERROR text is what a failed re-snap carries back.
    """

    def __init__(self):
        self.entries: list[tuple[set[str], str]] = []

    def report(self, level, message):
        self.entries.append((set(level), str(message)))

    def messages(self, level: str) -> list[str]:
        return [text for levels, text in self.entries if level in levels]


@mcp_handler
def resnap_merge_pair(object_a: str, object_b: str):
    """Re-run the snap on an existing merge pair to rebuild its stitch.

    The two objects must already form a merge pair (add_merge_pair or
    snap_to_vertices). The snap MOVES one of them: object A of the STORED
    pair, unless that side is in a STATIC group, in which case the other side
    moves instead. Which object moves therefore follows the stored pair, not
    the argument order used here. The two are left a small gap apart, sized
    from their contact offsets, rather than coincident.

    This is what makes a pair's stitch anchors current after either mesh was
    edited, and what gives a pair anchors at all when it was created without
    a snap. A pair whose ``stitch_row_count`` stays 0 forms no stitch at
    solve time.

    Args:
        object_a: Name of one object in the pair.
        object_b: Name of the other object in the pair, in either order.
    """
    from ...core.uuid_registry import get_object_by_uuid
    from ...mesh_ops.merge_ops import pair_stitch_row_count
    from ...mesh_ops.snap_ops import _snap_pair

    pair = _resolve_merge_pair(object_a, object_b)
    # Resolve the objects from the pair's own UUIDs, in the pair's own order.
    # For a pair of one type (shell-shell, solid-solid, rod-rod) that order is
    # what picks the stitch source and target, so it must be the stored order
    # and not the order this call named them in.
    obj_a = get_object_by_uuid(pair.object_a_uuid)
    obj_b = get_object_by_uuid(pair.object_b_uuid)

    reports = _SnapReports()
    status = _snap_pair(reports, bpy.context, obj_a, obj_b)
    if "FINISHED" not in status:
        errors = reports.messages("ERROR")
        raise MCPError(
            "; ".join(errors)
            or f"Re-snap returned {sorted(status)} without reporting a reason"
        )

    return {
        "message": f"Merge pair re-snapped: {obj_a.name} <-> {obj_b.name}",
        "object_a": obj_a.name,
        "object_b": obj_b.name,
        "object_a_uuid": pair.object_a_uuid,
        "object_b_uuid": pair.object_b_uuid,
        "stitch_row_count": pair_stitch_row_count(pair),
        "report": reports.messages("INFO"),
    }
