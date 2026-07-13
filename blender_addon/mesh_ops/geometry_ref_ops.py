# File: geometry_ref_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Operators backing the per-object "bending reference rest angle" UI in
# the SHELL material panel. The picker sets a reference object (a
# topological copy of a shell whose vertices were moved, e.g. by a
# modifier or geometry nodes) for the object currently focused in the
# group's reference-object pulldown; the clear operator removes it. The
# picked reference is validated against the source object's topology at
# pick time (and re-validated at encode time) so a deviating mesh is
# rejected early with a clear message.

import bpy  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_
from bpy.props import IntProperty, StringProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ..ui.dynamics.utils import get_assigned_by_selection_uuid, get_group_from_index


def _resolve_target_assigned(self, context):
    """Return ``(group, assigned)`` for the focused reference object or
    ``(None, None)`` after reporting a warning."""
    group = get_group_from_index(context.scene, self.group_index)
    if group is None:
        self.report({"WARNING"}, iface_("Object group not found."))
        return None, None
    # Prefer the explicit operator argument; fall back to the pulldown.
    if self.object_uuid:
        assigned = None
        for a in group.assigned_objects:
            if a.uuid == self.object_uuid:
                assigned = a
                break
    else:
        assigned = get_assigned_by_selection_uuid(group, "bend_ref_object_selection")
    if assigned is None:
        self.report(
            {"WARNING"}, iface_("Select a shell object in the pulldown first.")
        )
        return None, None
    return group, assigned


class OBJECT_OT_PickBendReference(Operator):
    """Set the reference object (from the active selection) used to compute
    this object's bending rest angle. The reference must be a topological
    copy whose vertices were moved (modifiers / geometry nodes are
    evaluated)."""

    bl_idname = "object.pick_bend_reference"
    bl_label = "Pick Reference Object"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore
    object_uuid: StringProperty(options={"HIDDEN"})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        from ..core.uuid_registry import get_object_by_uuid, get_or_create_object_uuid
        from ..core.utils import validate_bend_reference

        group, assigned = _resolve_target_assigned(self, context)
        if assigned is None:
            return {"CANCELLED"}

        source_obj = get_object_by_uuid(assigned.uuid)
        if source_obj is None:
            self.report({"WARNING"}, iface_("Source object could not be resolved."))
            return {"CANCELLED"}

        ref_obj = context.active_object
        ref_uuid = get_or_create_object_uuid(ref_obj)
        if not ref_uuid:
            self.report(
                {"WARNING"},
                iface_("'{name}' is not writable (library-linked).").format(
                    name=ref_obj.name
                ),
            )
            return {"CANCELLED"}

        ok, msg = validate_bend_reference(
            source_obj, ref_obj, context, group.object_type
        )
        if not ok:
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        assigned.bend_ref_uuid = ref_uuid
        assigned.bend_ref_name = ref_obj.name
        self.report(
            {"INFO"},
            iface_("Reference rest angle for '{source}' set to '{name}'.").format(
                source=source_obj.name, name=ref_obj.name
            ),
        )
        return {"FINISHED"}


class OBJECT_OT_ClearBendReference(Operator):
    """Clear the bending reference object for the focused shell object."""

    bl_idname = "object.clear_bend_reference"
    bl_label = "Clear Reference Object"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore
    object_uuid: StringProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        _group, assigned = _resolve_target_assigned(self, context)
        if assigned is None:
            return {"CANCELLED"}
        assigned.bend_ref_uuid = ""
        assigned.bend_ref_name = ""
        self.report({"INFO"}, iface_("Reference rest angle cleared."))
        return {"FINISHED"}


classes = (
    OBJECT_OT_PickBendReference,
    OBJECT_OT_ClearBendReference,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
