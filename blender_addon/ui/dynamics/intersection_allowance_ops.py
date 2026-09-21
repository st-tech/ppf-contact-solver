# File: intersection_allowance_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Add / remove / remove-all for the object subset each intersection allowance
# is narrowed to. See models/intersection_allowances.py for what the subset
# means and why it exists.

import bpy  # pyright: ignore
from bpy.types import Operator  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_  # pyright: ignore

from ...core.utils import redraw_all_areas
from ...models.collection_utils import safe_update_index
from ...models.enum_props import EnumProperty
from ...models.intersection_allowances import (
    INTERSECTION_ALLOWANCE_ITEMS,
    allowance_by_key,
    allowance_objects,
)
from .utils import get_group_from_index


def _resolve(self, context):
    """(group, spec, collection) for this operator call, or None on refusal.

    Reports the refusal on the operator so the UI says which half is missing
    rather than failing silently, which is what a CANCELLED with no report
    looks like from a panel.
    """
    group = get_group_from_index(context.scene, self.group_index)
    if group is None:
        self.report({"ERROR"}, iface_("Group not found"))
        return None
    try:
        spec = allowance_by_key(self.allowance)
    except ValueError as exc:
        self.report({"ERROR"}, str(exc))
        return None
    return group, spec, allowance_objects(group, spec)


class OBJECT_OT_AddIntersectionAllowanceObjects(Operator):
    """Narrow this allowance to the selected objects"""

    bl_idname = "object.add_intersection_allowance_objects"
    bl_label = "Add Selected Objects"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    allowance: EnumProperty(items=INTERSECTION_ALLOWANCE_ITEMS,
                            options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return len(context.selected_objects) > 0

    def execute(self, context):
        resolved = _resolve(self, context)
        if resolved is None:
            return {"CANCELLED"}
        group, spec, collection = resolved

        from ...core.uuid_registry import get_object_uuid

        # The list names objects of THIS group. An object the group does not
        # hold would encode to nothing, so it is refused here where the user
        # can see why, rather than accepted and dropped at build time.
        member_uuids = {
            assigned.uuid for assigned in group.assigned_objects if assigned.uuid
        }
        listed = {item.uuid for item in collection if item.uuid}

        added = 0
        outsiders = []
        for obj in context.selected_objects:
            obj_uuid = get_object_uuid(obj)
            if not obj_uuid or obj_uuid not in member_uuids:
                outsiders.append(obj.name)
                continue
            if obj_uuid in listed:
                continue
            item = collection.add()
            item.name = obj.name
            item.uuid = obj_uuid
            listed.add(obj_uuid)
            added += 1

        if added:
            setattr(group, spec.index_prop, len(collection) - 1)
        if outsiders:
            self.report(
                {"WARNING"},
                iface_(
                    "Not in group '{group}': {names}"
                ).format(group=group.name, names=", ".join(sorted(outsiders))),
            )
        if not added and not outsiders:
            self.report({"INFO"}, iface_("Already listed"))

        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_RemoveIntersectionAllowanceObject(Operator):
    """Remove the selected object from this allowance"""

    bl_idname = "object.remove_intersection_allowance_object"
    bl_label = "Remove"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    allowance: EnumProperty(items=INTERSECTION_ALLOWANCE_ITEMS,
                            options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        resolved = _resolve(self, context)
        if resolved is None:
            return {"CANCELLED"}
        group, spec, collection = resolved

        index = getattr(group, spec.index_prop)
        if not 0 <= index < len(collection):
            self.report({"ERROR"}, iface_("No object selected"))
            return {"CANCELLED"}
        collection.remove(index)
        setattr(group, spec.index_prop,
                safe_update_index(index, len(collection)))

        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_ClearIntersectionAllowanceObjects(Operator):
    """Remove every object from this allowance"""

    bl_idname = "object.clear_intersection_allowance_objects"
    bl_label = "Remove All"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    allowance: EnumProperty(items=INTERSECTION_ALLOWANCE_ITEMS,
                            options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        resolved = _resolve(self, context)
        if resolved is None:
            return {"CANCELLED"}
        group, spec, collection = resolved

        collection.clear()
        setattr(group, spec.index_prop, -1)

        redraw_all_areas(context)
        return {"FINISHED"}


classes = (
    OBJECT_OT_AddIntersectionAllowanceObjects,
    OBJECT_OT_RemoveIntersectionAllowanceObject,
    OBJECT_OT_ClearIntersectionAllowanceObjects,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
