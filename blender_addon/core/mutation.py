# File: mutation.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Unified state-mutation service.  The three user-visible surfaces
# (Blender UI operators, MCP handlers for LLM clients, Python API for
# Jupyter) all call through this service instead of duplicating
# validation and side-effect logic in three places.
#
# Each method:
#   * validates input (raises ``MutationError`` for contract violations)
#   * mutates scene state under a single-writer lock so UI clicks and
#     MCP task-queue drains can't interleave mid-operation
#   * returns a simple result dict so adapters (UI / MCP / API) can
#     translate it into their native error shape
#
# Adapters are thin:
#   UI:   ``MutationService.add_merge_pair(a, b); self.report(...)``
#   MCP:  ``try: MutationService.add_merge_pair(...) except MutationError: raise MCPError``
#   API:  ``try: MutationService.add_merge_pair(...) except MutationError as e: raise ValueError(str(e))``
#
# This pass adds the infrastructure and one canonical operation
# (``add_merge_pair``) as a migration template.  Additional operations
# move in over time as their surface consolidates.

from __future__ import annotations

import threading
from typing import Any


class MutationError(Exception):
    """Raised by MutationService for contract violations (bad input,
    missing object, unmet precondition).  Adapters translate this to
    their surface-specific error shape."""


# Single-writer lock: MCP handlers run on a worker thread and post
# tasks to a queue drained from a Blender timer; UI operators run
# synchronously on the main thread.  Without this lock, the two can
# race on the same PropertyGroup mutation.  The lock is cheap because
# mutations are short-lived.
_mutation_lock = threading.Lock()


class MutationService:
    """Central mutation API.  All methods are thread-safe via the
    module-level ``_mutation_lock``.

    Instances are stateless; methods are effectively static but
    exposed as methods so future work can inject per-scene context
    without breaking call sites.
    """

    # -- merge pairs -----------------------------------------------
    #
    # For each operation:
    #   * public method: validates, holds the lock, calls the raw impl
    #   * raw impl (on ``_Solver``): untyped, unvalidated, touches state
    # This keeps both the scripting API (``_Solver``) and the MCP handler
    # pointed at the same gate.

    @staticmethod
    def add_merge_pair(object_a: str, object_b: str) -> dict[str, Any]:
        """Add a merge pair between two named objects in the active scene.

        Validations applied identically on every surface:
        * both names non-empty
        * both refer to existing objects
        * not a self-merge
        """
        import bpy  # pyright: ignore

        if not object_a or not object_b:
            raise MutationError("object_a and object_b are both required")
        if object_a == object_b:
            raise MutationError("cannot merge an object with itself")

        with _mutation_lock:
            from .uuid_registry import get_or_create_object_uuid

            obj_a = bpy.data.objects.get(object_a)
            obj_b = bpy.data.objects.get(object_b)
            if obj_a is None:
                raise MutationError(f"object '{object_a}' not found in scene")
            if obj_b is None:
                raise MutationError(f"object '{object_b}' not found in scene")
            uuid_a = get_or_create_object_uuid(obj_a)
            uuid_b = get_or_create_object_uuid(obj_b)
            if not uuid_a or not uuid_b:
                raise MutationError("one or both objects are library-linked (unwritable)")
            from ..ops.api import _raw_add_merge_pair
            _raw_add_merge_pair(object_a, object_b)
            return {
                "object_a": object_a,
                "object_b": object_b,
                "object_a_uuid": uuid_a,
                "object_b_uuid": uuid_b,
            }

    @staticmethod
    def remove_merge_pair(object_a: str, object_b: str) -> dict[str, Any]:
        """Remove an existing merge pair.  Raises when no such pair."""
        import bpy  # pyright: ignore

        if not object_a or not object_b:
            raise MutationError("object_a and object_b are both required")

        with _mutation_lock:
            from .uuid_registry import get_or_create_object_uuid
            from ..models.groups import get_addon_data
            from ..ops.api import _raw_remove_merge_pair

            obj_a = bpy.data.objects.get(object_a)
            obj_b = bpy.data.objects.get(object_b)
            if obj_a is None:
                raise MutationError(f"object '{object_a}' not found in scene")
            if obj_b is None:
                raise MutationError(f"object '{object_b}' not found in scene")

            uuid_a = get_or_create_object_uuid(obj_a)
            uuid_b = get_or_create_object_uuid(obj_b)
            if not uuid_a or not uuid_b:
                raise MutationError("one or both objects are library-linked (unwritable)")

            # Check existence by UUID, not name — rename-safe.
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

    @staticmethod
    def clear_merge_pairs() -> dict[str, Any]:
        """Drop every merge pair from the active scene."""
        with _mutation_lock:
            from ..ops.api import _raw_clear_merge_pairs
            _raw_clear_merge_pairs()
            return {"cleared": True}

    # -- snap -----------------------------------------------------

    @staticmethod
    def snap_to_vertices(object_a: str, object_b: str) -> dict[str, Any]:
        """Snap *object_a* to the nearest vertices of *object_b*."""
        import bpy  # pyright: ignore

        if not object_a or not object_b:
            raise MutationError("object_a and object_b are both required")
        if object_a == object_b:
            raise MutationError("cannot snap an object to itself")

        with _mutation_lock:
            from .uuid_registry import get_or_create_object_uuid

            obj_a = bpy.data.objects.get(object_a)
            obj_b = bpy.data.objects.get(object_b)
            if obj_a is None:
                raise MutationError(f"object '{object_a}' not found in scene")
            if obj_b is None:
                raise MutationError(f"object '{object_b}' not found in scene")
            uuid_a = get_or_create_object_uuid(obj_a)
            uuid_b = get_or_create_object_uuid(obj_b)
            if not uuid_a or not uuid_b:
                raise MutationError("one or both objects are library-linked (unwritable)")
            from ..ops.api import _raw_snap
            _raw_snap(object_a, object_b)
            return {
                "object_a": object_a,
                "object_b": object_b,
                "object_a_uuid": uuid_a,
                "object_b_uuid": uuid_b,
            }

    # -- invisible colliders --------------------------------------

    @staticmethod
    def add_invisible_wall(position, normal) -> dict[str, Any]:
        """Add an infinite-plane invisible collider."""
        pos = _check_vec3("position", position)
        nrm = _check_vec3("normal", normal)
        with _mutation_lock:
            from ..ops.api import _raw_add_wall
            _raw_add_wall(pos, nrm)
            return {"type": "WALL", "position": pos, "normal": nrm}

    @staticmethod
    def add_invisible_sphere(
        position,
        radius: float,
        invert: bool = False,
        hemisphere: bool = False,
    ) -> dict[str, Any]:
        """Add a sphere invisible collider.  ``invert=True`` flips the
        contact direction so the sphere acts as a cavity; ``hemisphere=True``
        keeps only the upper half active."""
        pos = _check_vec3("position", position)
        try:
            r = float(radius)
        except (TypeError, ValueError):
            raise MutationError(f"radius must be numeric, got {radius!r}")
        if r <= 0.0:
            raise MutationError(f"radius must be positive, got {r}")
        with _mutation_lock:
            from ..ops.api import _raw_add_sphere
            _raw_add_sphere(pos, r, invert=invert, hemisphere=hemisphere)
            return {
                "type": "SPHERE",
                "position": pos,
                "radius": r,
                "invert": bool(invert),
                "hemisphere": bool(hemisphere),
            }

    @staticmethod
    def clear_invisible_colliders() -> dict[str, Any]:
        """Drop every invisible collider."""
        with _mutation_lock:
            from ..ops.api import _raw_clear_invisible_colliders
            _raw_clear_invisible_colliders()
            return {"cleared": True}

    # -- pins -----------------------------------------------------

    @staticmethod
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
            obj = bpy.data.objects.get(object_name)
            if obj is None:
                raise MutationError(f"object '{object_name}' not found in scene")
            if obj.type != "MESH":
                raise MutationError(f"object '{object_name}' is not a mesh")
            from .uuid_registry import get_or_create_object_uuid
            from ..ops.api import _raw_create_pin
            obj_uuid = get_or_create_object_uuid(obj)
            if not obj_uuid:
                raise MutationError(f"object '{object_name}' is library-linked (unwritable)")
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


def _check_vec3(name: str, v) -> tuple[float, float, float]:
    from .utils import check_vec3
    return check_vec3(name, v, MutationError)


# Module-level singleton so adapters don't have to construct their own.
service = MutationService()
