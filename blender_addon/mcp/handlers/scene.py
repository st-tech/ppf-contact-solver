# File: handlers/scene.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP handlers for scene-level setup: invisible colliders, merge pairs,
# and snap-to-vertices.  Thin adapters over ``core.mutation.service``:
# validate input via schema, translate ``MutationError`` into
# ``MCPError``, return a result dict.  All business logic and locking
# is in the service; this file is a one-line-per-call translation.

from typing import Optional

import bpy  # pyright: ignore

from ..decorators import MCPError, mcp_handler
from ...core.mutation import MutationError, service


def _call(fn, *args, **kwargs):
    """Run a MutationService call, surfacing MutationError as MCPError."""
    try:
        return fn(*args, **kwargs)
    except MutationError as e:
        raise MCPError(str(e))


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
    _call(service.add_invisible_wall, position, normal)
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
    _call(service.add_invisible_sphere, position, radius, invert=invert, hemisphere=hemisphere)
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
    state = bpy.context.scene.zozo_contact_solver.state
    n = len(state.invisible_colliders)
    if index < 0 or index >= n:
        raise MCPError(f"Collider index {index} out of range (0..{n - 1})")
    state.invisible_colliders.remove(index)
    if state.invisible_colliders_index >= len(state.invisible_colliders):
        state.invisible_colliders_index = max(0, len(state.invisible_colliders) - 1)
    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return f"Removed collider at index {index}"


@mcp_handler
def clear_invisible_colliders():
    """Remove every invisible collider from the scene."""
    _call(service.clear_invisible_colliders)
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
    _call(service.add_merge_pair, name_a, name_b)
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
    _call(service.remove_merge_pair, name_a, name_b)
    return {
        "message": f"Merge pair removed: {name_a} <-> {name_b}",
        "object_a": name_a,
        "object_b": name_b,
        "object_a_uuid": uuid_a,
        "object_b_uuid": uuid_b,
    }


@mcp_handler
def list_merge_pairs():
    """Return all stored merge pairs with both display names and UUIDs."""
    from ...models.groups import get_addon_data
    state = get_addon_data(bpy.context.scene).state
    pairs = [
        {
            "object_a": p.object_a,
            "object_b": p.object_b,
            "object_a_uuid": p.object_a_uuid,
            "object_b_uuid": p.object_b_uuid,
        }
        for p in state.merge_pairs
    ]
    return {"pairs": pairs}


@mcp_handler
def clear_merge_pairs():
    """Remove every merge pair from the scene."""
    _call(service.clear_merge_pairs)
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
    _call(service.snap_to_vertices, name_a, name_b)
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
