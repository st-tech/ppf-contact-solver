# File: mutation.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Unified state-mutation surface.  The three user-visible callers
# (Blender UI operators, MCP handlers for LLM clients, Python API for
# Jupyter) all dispatch through these module-level functions instead
# of duplicating validation and side-effect logic in three places.
#
# Each public function:
#   * validates input (raises ``MutationError`` for contract violations)
#   * mutates scene state under a single-writer lock so UI clicks and
#     MCP task-queue drains can't interleave mid-operation
#   * returns a simple result dict so adapters (UI / MCP / API) can
#     translate it into their native error shape
#
# Adapters are thin:
#   UI:   ``mutation.add_merge_pair(a, b); self.report(...)``
#   MCP:  ``try: mutation.add_merge_pair(...) except MutationError: raise MCPError``
#   API:  ``try: mutation.add_merge_pair(...) except MutationError as e: raise ValueError(str(e))``

from __future__ import annotations

import threading
from typing import Any


class MutationError(Exception):
    """Raised by the mutation API for contract violations (bad input,
    missing object, unmet precondition).  Adapters translate this to
    their surface-specific error shape."""


# Single-writer lock: MCP handlers run on a worker thread and post
# tasks to a queue drained from a Blender timer; UI operators run
# synchronously on the main thread.  Without this lock, the two can
# race on the same PropertyGroup mutation.  The lock is cheap because
# mutations are short-lived.
_mutation_lock = threading.Lock()


def _resolve_writable(name: str) -> tuple[Any, str]:
    """Look up an object by name and return (obj, uuid).

    Raises ``MutationError`` if the object is missing or library-linked
    (no writable UUID).  Caller is expected to hold ``_mutation_lock``.
    """
    import bpy  # pyright: ignore
    from .uuid_registry import get_or_create_object_uuid

    obj = bpy.data.objects.get(name)
    if obj is None:
        raise MutationError(f"object '{name}' not found in scene")
    uuid = get_or_create_object_uuid(obj)
    if not uuid:
        raise MutationError(f"object '{name}' is library-linked (unwritable)")
    return obj, uuid


# -- merge pairs ---------------------------------------------------------
#
# For each operation:
#   * public function: validates, holds the lock, calls the raw impl
#   * raw impl (``_raw_*``): untyped, unvalidated, touches state
# This keeps both the scripting API (``_Solver``) and the MCP handler
# pointed at the same gate.


def add_merge_pair(object_a: str, object_b: str) -> dict[str, Any]:
    """Add a merge pair between two named objects in the active scene.

    Validations applied identically on every surface:
    * both names non-empty
    * both refer to existing objects
    * not a self-merge
    """
    if not object_a or not object_b:
        raise MutationError("object_a and object_b are both required")
    if object_a == object_b:
        raise MutationError("cannot merge an object with itself")

    with _mutation_lock:
        _, uuid_a = _resolve_writable(object_a)
        _, uuid_b = _resolve_writable(object_b)
        _raw_add_merge_pair(object_a, object_b)
        return {
            "object_a": object_a,
            "object_b": object_b,
            "object_a_uuid": uuid_a,
            "object_b_uuid": uuid_b,
        }


def remove_merge_pair(object_a: str, object_b: str) -> dict[str, Any]:
    """Remove an existing merge pair.  Raises when no such pair."""
    import bpy  # pyright: ignore

    if not object_a or not object_b:
        raise MutationError("object_a and object_b are both required")

    with _mutation_lock:
        from ..models.groups import get_addon_data

        _, uuid_a = _resolve_writable(object_a)
        _, uuid_b = _resolve_writable(object_b)

        # Check existence by UUID, not name; rename-safe.
        state = get_addon_data(bpy.context.scene).state
        found = False
        for pair in state.merge_pairs:
            if (pair.object_a_uuid == uuid_a and pair.object_b_uuid == uuid_b) or \
               (pair.object_a_uuid == uuid_b and pair.object_b_uuid == uuid_a):
                found = True
                break
        if not found:
            raise MutationError(
                f"no merge pair exists between '{object_a}' and '{object_b}'"
            )
        _raw_remove_merge_pair(object_a, object_b)
        return {
            "object_a": object_a,
            "object_b": object_b,
            "object_a_uuid": uuid_a,
            "object_b_uuid": uuid_b,
        }


def clear_merge_pairs() -> dict[str, Any]:
    """Drop every merge pair from the active scene."""
    with _mutation_lock:
        _raw_clear_merge_pairs()
        return {"cleared": True}


# -- snap ----------------------------------------------------------------


def snap_to_vertices(object_a: str, object_b: str) -> dict[str, Any]:
    """Snap *object_a* to the nearest vertices of *object_b*."""
    if not object_a or not object_b:
        raise MutationError("object_a and object_b are both required")
    if object_a == object_b:
        raise MutationError("cannot snap an object to itself")

    with _mutation_lock:
        _, uuid_a = _resolve_writable(object_a)
        _, uuid_b = _resolve_writable(object_b)
        _raw_snap(object_a, object_b)
        return {
            "object_a": object_a,
            "object_b": object_b,
            "object_a_uuid": uuid_a,
            "object_b_uuid": uuid_b,
        }


# -- invisible colliders -------------------------------------------------


def add_invisible_wall(position, normal) -> dict[str, Any]:
    """Add an infinite-plane invisible collider."""
    from .utils import check_vec3
    pos = check_vec3("position", position, MutationError)
    nrm = check_vec3("normal", normal, MutationError)
    with _mutation_lock:
        _raw_add_wall(pos, nrm)
        return {"type": "WALL", "position": pos, "normal": nrm}


def add_invisible_sphere(
    position,
    radius: float,
    invert: bool = False,
    hemisphere: bool = False,
) -> dict[str, Any]:
    """Add a sphere invisible collider.  ``invert=True`` flips the
    contact direction so the sphere acts as a cavity; ``hemisphere=True``
    keeps only the upper half active."""
    from .utils import check_vec3
    pos = check_vec3("position", position, MutationError)
    try:
        r = float(radius)
    except (TypeError, ValueError):
        raise MutationError(f"radius must be numeric, got {radius!r}")
    if r <= 0.0:
        raise MutationError(f"radius must be positive, got {r}")
    with _mutation_lock:
        _raw_add_sphere(pos, r, invert=invert, hemisphere=hemisphere)
        return {
            "type": "SPHERE",
            "position": pos,
            "radius": r,
            "invert": bool(invert),
            "hemisphere": bool(hemisphere),
        }


def clear_invisible_colliders() -> dict[str, Any]:
    """Drop every invisible collider."""
    with _mutation_lock:
        _raw_clear_invisible_colliders()
        return {"cleared": True}


# -- pins ----------------------------------------------------------------


def create_pin(group_uuid: str, object_name: str, vertex_group_name: str) -> dict[str, Any]:
    """Add a vertex group to a dynamics group's pin list."""
    import bpy  # pyright: ignore

    if not group_uuid:
        raise MutationError("group_uuid is required")
    if not object_name or not vertex_group_name:
        raise MutationError("object_name and vertex_group_name are both required")

    with _mutation_lock:
        from ..models.groups import get_group_by_uuid
        if get_group_by_uuid(bpy.context.scene, group_uuid) is None:
            raise MutationError(f"group '{group_uuid}' not found")
        obj, obj_uuid = _resolve_writable(object_name)
        if obj.type != "MESH":
            raise MutationError(f"object '{object_name}' is not a mesh")
        try:
            _raw_create_pin(group_uuid, object_name, vertex_group_name)
        except ValueError as e:
            raise MutationError(str(e))
        return {
            "group_uuid": group_uuid,
            "object_name": object_name,
            "object_uuid": obj_uuid,
            "vertex_group_name": vertex_group_name,
        }


def _raw_add_merge_pair(object_a: str, object_b: str) -> None:
    import bpy  # pyright: ignore
    from .uuid_registry import get_or_create_object_uuid
    from ..models.groups import get_addon_data
    state = get_addon_data(bpy.context.scene).state
    obj_a = bpy.data.objects.get(object_a)
    obj_b = bpy.data.objects.get(object_b)
    uuid_a = get_or_create_object_uuid(obj_a) if obj_a else ""
    uuid_b = get_or_create_object_uuid(obj_b) if obj_b else ""
    if not uuid_a or not uuid_b:
        return
    for pair in state.merge_pairs:
        if (pair.object_a_uuid == uuid_a and pair.object_b_uuid == uuid_b) or (
            pair.object_a_uuid == uuid_b and pair.object_b_uuid == uuid_a
        ):
            return
    item = state.merge_pairs.add()
    item.object_a = object_a
    item.object_b = object_b
    item.object_a_uuid = uuid_a
    item.object_b_uuid = uuid_b
    state.merge_pairs_index = len(state.merge_pairs) - 1
    from ..ui.dynamics.overlay import apply_object_overlays
    apply_object_overlays()


def _raw_remove_merge_pair(object_a: str, object_b: str) -> None:
    import bpy  # pyright: ignore
    from .uuid_registry import get_or_create_object_uuid
    from ..models.groups import get_addon_data
    from ..models.collection_utils import safe_update_index
    state = get_addon_data(bpy.context.scene).state
    # Resolve names to UUIDs so the comparison is rename-safe.
    obj_a = bpy.data.objects.get(object_a)
    obj_b = bpy.data.objects.get(object_b)
    uuid_a = get_or_create_object_uuid(obj_a) if obj_a else ""
    uuid_b = get_or_create_object_uuid(obj_b) if obj_b else ""
    if not uuid_a or not uuid_b:
        return
    for i in range(len(state.merge_pairs)):
        pair = state.merge_pairs[i]
        if (pair.object_a_uuid == uuid_a and pair.object_b_uuid == uuid_b) or (
            pair.object_a_uuid == uuid_b and pair.object_b_uuid == uuid_a
        ):
            state.merge_pairs.remove(i)
            state.merge_pairs_index = safe_update_index(
                state.merge_pairs_index, len(state.merge_pairs)
            )
            from ..ui.dynamics.overlay import apply_object_overlays
            apply_object_overlays()
            return


def _raw_clear_merge_pairs() -> None:
    import bpy  # pyright: ignore
    from ..models.groups import get_addon_data
    state = get_addon_data(bpy.context.scene).state
    state.merge_pairs.clear()
    state.merge_pairs_index = -1
    from ..ui.dynamics.overlay import apply_object_overlays
    apply_object_overlays()


def _raw_get_merge_pairs() -> list[tuple[str, str]]:
    """Return existing merge pairs as (name_a, name_b) tuples."""
    import bpy  # pyright: ignore
    from ..models.groups import get_addon_data
    state = get_addon_data(bpy.context.scene).state
    return [(p.object_a, p.object_b) for p in state.merge_pairs]


def _raw_snap(object_a: str, object_b: str) -> None:
    import bpy  # pyright: ignore
    from .uuid_registry import get_or_create_object_uuid
    from ..models.groups import get_addon_data
    state = get_addon_data(bpy.context.scene).state
    obj_a = bpy.data.objects.get(object_a)
    obj_b = bpy.data.objects.get(object_b)
    if not obj_a or not obj_b:
        return
    uid_a = get_or_create_object_uuid(obj_a)
    uid_b = get_or_create_object_uuid(obj_b)
    if not uid_a or not uid_b:
        return
    state.snap_object_a = uid_a
    state.snap_object_b = uid_b
    bpy.ops.object.snap_to_vertices()


def _raw_add_wall(position, normal) -> None:
    from ..ops.api import _InvisibleWallBuilder
    _InvisibleWallBuilder(position, normal)


def _raw_add_sphere(position, radius, invert: bool = False, hemisphere: bool = False) -> None:
    from ..ops.api import _InvisibleSphereBuilder
    builder = _InvisibleSphereBuilder(position, radius)
    if invert:
        builder.invert()
    if hemisphere:
        builder.hemisphere()


def _raw_clear_invisible_colliders() -> None:
    import bpy  # pyright: ignore
    from ..models.groups import get_addon_data, invalidate_overlays
    state = get_addon_data(bpy.context.scene).state
    state.invisible_colliders.clear()
    state.invisible_colliders_index = -1
    invalidate_overlays()


def _raw_create_pin(group_uuid: str, object_name: str, vertex_group_name: str) -> None:
    """Add a pin without validation (caller has validated)."""
    import bpy  # pyright: ignore
    from ..models.groups import (
        decode_vertex_group_identifier,
        encode_vertex_group_identifier,
        get_group_by_uuid,
        invalidate_overlays,
    )
    from .uuid_registry import compute_vg_hash, get_or_create_object_uuid
    group = get_group_by_uuid(bpy.context.scene, group_uuid)
    if group is None:
        raise ValueError(f"group '{group_uuid}' not found")
    obj = bpy.data.objects.get(object_name)
    identifier = encode_vertex_group_identifier(object_name, vertex_group_name)
    uuid_val = get_or_create_object_uuid(obj)
    if not uuid_val:
        raise ValueError(f"object '{object_name}' is not writable (library-linked)")
    vg_hash_val = str(compute_vg_hash(obj, vertex_group_name))
    # Match-by-(UUID, vg_name); rename-safe duplicate detection.
    for p in group.pin_vertex_groups:
        if p.object_uuid != uuid_val:
            continue
        _, p_vg = decode_vertex_group_identifier(p.name)
        if p_vg == vertex_group_name:
            return
    item = group.pin_vertex_groups.add()
    try:
        item.name = identifier
        item.object_uuid = uuid_val
        item.vg_hash = vg_hash_val
        group.pin_vertex_groups_index = len(group.pin_vertex_groups) - 1
    except Exception:
        group.pin_vertex_groups.remove(len(group.pin_vertex_groups) - 1)
        raise
    invalidate_overlays()
