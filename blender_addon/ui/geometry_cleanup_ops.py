# File: geometry_cleanup_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Operators that clean up mesh geometry the solver rejects at Transfer.
# Currently: removing stray (isolated, faceless) vertices from STATIC
# colliders, surfaced as a button under the Transfer error label.

import bmesh
import bpy  # pyright: ignore

from ..core.client import communicator as com
from ..core.utils import redraw_all_areas


def _static_isolated_offenders(context):
    """Map ``{object: [isolated vertex indices]}`` for active STATIC colliders.

    Mirrors the encoder's STATIC isolated-vertex check
    (``encoder.mesh.detect_isolated_vertices``) so the button removes exactly
    the vertices that fail Transfer.
    """
    from ..core.encoder.mesh import detect_isolated_vertices
    from ..core.uuid_registry import resolve_assigned
    from ..models.groups import iterate_object_groups

    offenders = {}
    for group in iterate_object_groups(context.scene):
        if not group.active or group.object_type != "STATIC":
            continue
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            obj = resolve_assigned(assigned)
            if obj is None or obj.type != "MESH":
                continue
            isolated = detect_isolated_vertices(obj.data)
            if isolated:
                offenders[obj] = isolated
    return offenders


def _delete_vertices(obj, indices):
    """Delete the given vertex indices (and their incident loose edges).

    Operates on the object's mesh datablock via bmesh; faces are untouched
    (the indices are faceless by construction).
    """
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    n = len(bm.verts)
    to_del = [bm.verts[i] for i in indices if 0 <= i < n]
    if to_del:
        bmesh.ops.delete(bm, geom=to_del, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    me.update()
    return len(to_del)


class MESH_OT_RemoveIsolatedVertices(bpy.types.Operator):
    """Remove stray vertices that belong to no face from the assigned STATIC
    collider meshes, so the scene can be transferred. Only vertices in no
    triangle are deleted (with their loose edges); faces are untouched."""

    bl_idname = "ssh.remove_isolated_vertices"
    bl_label = "Remove Isolated Vertices"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # bmesh from_mesh/to_mesh read/write the mesh datablock, which is
        # stale while an object is in Edit Mode; leave it first.
        if context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass

        offenders = _static_isolated_offenders(context)
        if not offenders:
            self.report({"INFO"}, "No isolated vertices found on STATIC colliders.")
            return {"CANCELLED"}

        total = 0
        parts = []
        for obj, indices in offenders.items():
            removed = _delete_vertices(obj, indices)
            total += removed
            parts.append(f"{obj.name} ({removed})")

        com.set_error("")
        redraw_all_areas(context)
        self.report(
            {"INFO"},
            f"Removed {total} isolated vertex(es) from {', '.join(parts)}. "
            f"Transfer again.",
        )
        return {"FINISHED"}


classes = (MESH_OT_RemoveIsolatedVertices,)
