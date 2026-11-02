# File: geometry_cleanup_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Operators that clean up mesh geometry the solver rejects at Transfer:
# removing stray (isolated, faceless) vertices from STATIC colliders, and
# triangulating the faces whose tessellation has no usable rest shape. Both are
# surfaced as a button under the Transfer error label; the second also opens a
# dialog on the failed Transfer, because that error costs the artist an upload
# and reads as a wall of text in the status bar.

import textwrap

import bmesh
import bpy  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_
from bpy.props import BoolProperty, StringProperty  # pyright: ignore

from ..core.client import communicator as com
from ..core.utils import (
    DegenerateTessellationError,
    find_degenerate_tessellation,
    redraw_all_areas,
    triangulate_degenerate_faces,
)


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
            self.report(
                {"INFO"}, iface_("No isolated vertices found on STATIC colliders.")
            )
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
            iface_(
                "Removed {count} isolated vertex(es) from {objects}. Transfer again."
            ).format(count=total, objects=", ".join(parts)),
        )
        return {"FINISHED"}


def _degenerate_tessellation_offenders(context):
    """``[(object, found)]`` for every assigned object whose tessellation has
    triangles with no usable rest shape.

    Mirrors the encoder's gate in ``_build_obj_data`` exactly, SAND exemption
    included, so the button repairs precisely what fails Transfer. Scanning the
    whole scene rather than only the object the error named means one click
    clears the scene instead of one object per Transfer attempt.
    """
    from ..core.uuid_registry import resolve_assigned
    from ..models.groups import iterate_object_groups

    offenders = {}
    for group in iterate_object_groups(context.scene):
        if not group.active or group.object_type == "SAND":
            continue
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            obj = resolve_assigned(assigned)
            if obj is None or obj.type != "MESH":
                continue
            found = find_degenerate_tessellation(obj)
            if found["count"]:
                # By DATABLOCK: an object in two active groups, and two objects
                # sharing one mesh, would otherwise be repaired twice, the
                # second time through stale pre-repair polygon indices, and
                # counted twice in the report.
                offenders[obj.data] = (obj, found)
    return list(offenders.values())


class MESH_OT_TriangulateDegenerateFaces(bpy.types.Operator):
    """Triangulate only the faces whose tessellation leaves the solver no
    usable rest shape, so the scene can be transferred. Every other face keeps
    its shape: Blender splits a quad along the diagonal that puts three nearly
    collinear vertices in one triangle, and the other diagonal is sound."""

    bl_idname = "ssh.triangulate_degenerate_faces"
    bl_label = "Triangulate Faces"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # bmesh from_mesh/to_mesh read and write the mesh datablock, which is
        # stale while an object is in Edit Mode; leave it first.
        if context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError as exc:
                # bmesh writes the mesh datablock, which Blender discards on
                # leaving Edit Mode, so a repair applied from there is lost
                # and reported as done. Refuse instead.
                self.report(
                    {"ERROR"},
                    iface_(
                        "Switch to Object Mode before repairing: {error}"
                    ).format(error=str(exc)),
                )
                return {"CANCELLED"}

        offenders = _degenerate_tessellation_offenders(context)
        if not offenders:
            self.report(
                {"INFO"},
                iface_("No degenerate tessellation found on the assigned meshes."),
            )
            return {"CANCELLED"}

        total = 0
        stuck = []
        parts = []
        for obj, found in offenders:
            split, unrepairable = triangulate_degenerate_faces(obj, found=found)
            total += split
            if split:
                parts.append(f"{obj.name} ({split})")
            if unrepairable:
                stuck.append(f"{obj.name} ({unrepairable})")

        # Clear the standing refusal on the RESULT, not on the intent. A
        # partially repaired scene still refuses the Transfer, and the panel's
        # whole error block, repair row included, draws off `com.error`.
        remaining = _degenerate_tessellation_offenders(context)
        if total:
            redraw_all_areas(context)
        if remaining:
            obj, found = remaining[0]
            com.set_error(
                iface_(
                    "Object '{name}' still tessellates into {count} "
                    "triangle(s) with no usable rest shape, from face index "
                    "{faces}."
                ).format(
                    name=obj.name,
                    count=found["count"],
                    faces=", ".join(str(p) for p in found["polygons"][:8]),
                )
                + (
                    " Select the mesh in Edit Mode and run Face > Triangulate "
                    "Faces (Ctrl+T) before transferring."
                    if found["all_repairable_by_triangulation"] else ""
                )
            )
        elif total:
            com.set_error("")

        if stuck:
            # Faces no split can rescue: one that is ALREADY a triangle, so it
            # is its own only triangulation, or a polygon of no area, or one
            # with a zero-length boundary edge, since every boundary edge
            # belongs to a triangle of every triangulation.
            self.report(
                {"WARNING"},
                iface_(
                    "Triangulated {count} face(s) on {objects}. {stuck} still "
                    "cannot be repaired this way: run Mesh > Merge > By "
                    "Distance to weld coincident vertices, or dissolve those "
                    "faces."
                ).format(
                    count=total,
                    objects=", ".join(parts),
                    stuck=", ".join(stuck),
                ) if total else iface_(
                    "{stuck} cannot be repaired by triangulating: a face that "
                    "is already a triangle has no other split, and one with no "
                    "area or a zero-length boundary edge has no sound split at "
                    "all. Move the offending vertex, weld coincident vertices "
                    "with Mesh > Merge > By Distance, or dissolve those faces."
                ).format(stuck=", ".join(stuck)),
            )
            return {"FINISHED"} if total else {"CANCELLED"}

        self.report(
            {"INFO"},
            iface_(
                "Triangulated {count} face(s) on {objects}. Transfer again."
            ).format(count=total, objects=", ".join(parts)),
        )
        return {"FINISHED"}


class SOLVER_OT_DegenerateTessellationDialog(bpy.types.Operator):
    """Report, in a dialog, that an object tessellates into triangles the
    solver has no usable rest shape for, and offer the repair."""

    bl_idname = "ssh.degenerate_tessellation_dialog"
    bl_label = "Cannot Transfer This Mesh"
    # INTERNAL keeps it out of the operator search: it is only ever opened by
    # the failing Transfer, and it carries the message as a property, so an
    # F3 invocation would draw an empty box.
    bl_options = {"INTERNAL"}

    message: StringProperty(name="Message", default="")  # pyright: ignore
    repairable: BoolProperty(default=True)  # pyright: ignore

    def invoke(self, context, event):
        # A props dialog attaches a modal handler for as long as it is open,
        # and Blender writes no auto-save while one is attached (issue #145).
        # This one is transient by construction: it ends when the user answers
        # it. Do NOT convert it into anything that stays up.
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        # `layout.label` draws a multi-line string on one line with the breaks
        # as glyphs, so wrap it here. The icon goes on the first line only, so
        # the rest reads as one paragraph rather than a list of alerts.
        lines = textwrap.wrap(self.message, width=88) or [self.message]
        for i, line in enumerate(lines):
            col.label(text=line, icon="ERROR" if i == 0 else "BLANK1")
        if self.repairable:
            layout.separator()
            row = layout.row()
            row.operator(
                MESH_OT_TriangulateDegenerateFaces.bl_idname,
                icon="MOD_TRIANGULATE",
            )

    def execute(self, context):
        return {"FINISHED"}


def wants_transfer_dialog(exc) -> bool:
    """Whether *exc* is an encode refusal the dialog can explain and repair.

    Split from :func:`popup_transfer_error` so the decision can be asserted
    without opening a modal dialog, which a test driver cannot answer.
    """
    return isinstance(exc, DegenerateTessellationError)


def popup_transfer_error(exc) -> bool:
    """Open the dialog when *exc* is the degenerate-tessellation refusal.

    Returns whether a dialog was scheduled, so a caller can tell an error it
    has explained from one it has only reported.

    The open is deferred through a timer rather than invoked inline: every
    caller is inside a staged modal tick, and opening a dialog from there runs
    a nested modal handler underneath the operator that is about to end.
    """
    if not wants_transfer_dialog(exc):
        return False

    message = str(exc)
    repairable = exc.repairable

    def _open():
        try:
            bpy.ops.ssh.degenerate_tessellation_dialog(
                "INVOKE_DEFAULT", message=message, repairable=repairable,
            )
        except RuntimeError:
            # No window to parent a dialog to (background Blender). The
            # message is already reported through the operator and the panel.
            pass
        return None

    bpy.app.timers.register(_open, first_interval=0.0)
    return True


classes = (
    MESH_OT_RemoveIsolatedVertices,
    MESH_OT_TriangulateDegenerateFaces,
    SOLVER_OT_DegenerateTessellationDialog,
)
