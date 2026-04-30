# File: group_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore

from bpy.types import Operator  # pyright: ignore

from ...models.collection_utils import safe_update_index
from ...models.groups import get_addon_data
from ..state import (
    ObjectGroup,
    assign_display_indices,
    find_available_group_slot,
    iterate_active_object_groups,
)
from .utils import cleanup_pin_vertex_groups_for_object, get_group_from_index, reset_object_display


class OBJECT_OT_CreateGroup(Operator):
    """Create a new dynamics group"""

    bl_idname = "object.create_group"
    bl_label = "Create Group"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        slot_index = find_available_group_slot(scene)
        if slot_index is None:
            self.report({"WARNING"}, "Maximum number of groups reached")
            return {"CANCELLED"}

        prop_name = f"object_group_{slot_index}"
        group: ObjectGroup = getattr(get_addon_data(scene), prop_name)
        group.reset_to_defaults()
        group.active = True
        group.ensure_uuid()
        # Don't set name - leave it empty so UI shows "Group N" dynamically

        assign_display_indices(scene)

        # Update overlays when creating a new group
        from .overlay import apply_object_overlays

        apply_object_overlays()

        return {"FINISHED"}


class OBJECT_OT_DeleteGroup(Operator):
    """Delete the active group"""

    bl_idname = "object.delete_group"
    bl_label = "Delete Group"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def invoke(self, context, event):
        group = get_group_from_index(context.scene, self.group_index)
        if group and len(group.assigned_objects) > 0:
            return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found or not active")
            return {"CANCELLED"}

        # Reset object display properties
        for obj_ref in group.assigned_objects:
            from ...core.uuid_registry import resolve_assigned
            obj = resolve_assigned(obj_ref)
            if obj:
                reset_object_display(obj)

        # Delete the group by resetting to defaults (which now sets active=False)
        group.reset_to_defaults()
        assign_display_indices(scene)
        apply_object_overlays()

        return {"FINISHED"}


class OBJECT_OT_DuplicateGroup(Operator):
    """Duplicate a dynamics group (material params only, no objects or pins)"""

    bl_idname = "object.duplicate_group"
    bl_label = "Duplicate Group"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @staticmethod
    def _next_name(name: str) -> str:
        """Generate incremented name: foo -> foo-1, foo-1 -> foo-2."""
        import re

        m = re.match(r"^(.*)-(\d+)$", name)
        if m:
            return f"{m.group(1)}-{int(m.group(2)) + 1}"
        return f"{name}-1" if name else ""

    def execute(self, context):
        from ...core.profile import apply_material_profile, read_material_profile
        from .overlay import apply_object_overlays

        scene = context.scene
        src = get_group_from_index(scene, self.group_index)
        if src is None:
            self.report({"ERROR"}, "Source group not found")
            return {"CANCELLED"}

        slot_index = find_available_group_slot(scene)
        if slot_index is None:
            self.report({"WARNING"}, "Maximum number of groups reached")
            return {"CANCELLED"}

        addon_data = get_addon_data(scene)
        dst = getattr(addon_data, f"object_group_{slot_index}")
        dst.reset_to_defaults()
        dst.active = True
        dst.ensure_uuid()
        dst.name = self._next_name(src.name)

        # Copy material params only
        mat_data = read_material_profile(src)
        apply_material_profile(mat_data, dst)

        assign_display_indices(scene)
        apply_object_overlays()
        return {"FINISHED"}


class OBJECT_OT_DeleteAllGroups(Operator):
    """Delete all groups"""

    bl_idname = "object.delete_all_groups"
    bl_label = "Delete All Groups"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(
            group.active for group in iterate_active_object_groups(context.scene)
        )

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene

        for i in range(32):  # N_MAX_GROUPS
            prop_name = f"object_group_{i}"
            group: ObjectGroup | None = getattr(get_addon_data(scene), prop_name, None)
            if group and group.active:
                for obj_ref in group.assigned_objects:
                    from ...core.uuid_registry import resolve_assigned
                    obj = resolve_assigned(obj_ref)
                    if obj:
                        reset_object_display(obj)

                group.reset_to_defaults()

        get_addon_data(scene).current_group_uuid = ""
        apply_object_overlays()

        return {"FINISHED"}


class OBJECT_OT_AddObjectsToGroup(Operator):
    """Add selected objects to the active group"""

    bl_idname = "object.add_objects_to_group"
    bl_label = "Add Selected Objects"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return len(context.selected_objects) > 0

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is not None:
            from ...core.uuid_registry import get_or_create_object_uuid
            existing_uuids = {a.uuid for a in group.assigned_objects if a.uuid}
            group_uuid = group.ensure_uuid()

            from ...core.utils import (
                count_ngon_faces, find_linked_duplicate_siblings,
            )
            for obj in list(context.selected_objects):
                is_acceptable = obj.type == "MESH" or (
                    obj.type == "CURVE" and group.object_type == "ROD"
                )
                obj_uid = get_or_create_object_uuid(obj)
                if not obj_uid:
                    self.report(
                        {"WARNING"},
                        f"Object '{obj.name}' is library-linked and cannot be assigned",
                    )
                    continue
                if not is_acceptable or obj_uid in existing_uuids:
                    continue

                # Reject shallow-copied (Alt-D / Linked Duplicate) objects
                # before they can be assigned. Shared mesh data would let
                # the encoder ship inconsistent vertex coords across two
                # "different" assigned objects and silently corrupt PC2
                # playback at fetch time; the only safe fix is for the
                # user to make the duplicate single-user via Object > Make
                # Single User > Object & Data.
                siblings = find_linked_duplicate_siblings(obj)
                if siblings:
                    self.report(
                        {"ERROR"},
                        f"Object '{obj.name}' is a Linked Duplicate (shares mesh "
                        f"data with {siblings[0]!r}). Make it single-user "
                        f"(Object > Relations > Make Single User > Object & Data) "
                        f"before assigning.",
                    )
                    continue

                # Reject N-gons (polygons with > 4 vertices). The solver
                # operates on triangles + quads only; the encoder used to
                # silently fan-triangulate N-gons, which broke vertex-
                # group / pin / UV correspondence. Make the user
                # triangulate explicitly so the geometry they see in
                # the viewport is the geometry the solver runs on.
                ngon_count = count_ngon_faces(obj)
                if ngon_count > 0:
                    self.report(
                        {"ERROR"},
                        f"Object '{obj.name}' has {ngon_count} N-gon face(s) "
                        f"(polygon with > 4 vertices). The solver supports "
                        f"only triangles and quads. Apply a Triangulate "
                        f"modifier or use Mesh > Faces > Triangulate Faces "
                        f"in Edit Mode before assigning.",
                    )
                    continue

                in_other = False
                for other_group in iterate_active_object_groups(scene):
                    if other_group.ensure_uuid() != group_uuid:
                        other_uuids = {a.uuid for a in other_group.assigned_objects if a.uuid}
                        if obj_uid in other_uuids:
                            self.report(
                                {"WARNING"},
                                f"Object '{obj.name}' is already in another group",
                            )
                            in_other = True
                            break
                if in_other:
                    continue

                item = group.assigned_objects.add()
                item.name = obj.name
                item.uuid = get_or_create_object_uuid(obj)
                if obj.type == "MESH":
                    obj.show_wire = True
                    obj.show_all_edges = True
                    if group.object_type == "ROD" and not any(
                        m.type == "WIREFRAME" for m in obj.modifiers
                    ):
                        wire = obj.modifiers.new(name="Wireframe", type="WIREFRAME")
                        # Even-offset compensates sharp dihedrals by extruding
                        # further; rod simulations fold/crease faces freely
                        # (only edge length is constrained), and that turns
                        # into wild visual jumps in the wireframe geometry.
                        # Uniform thickness keeps the visualization stable.
                        wire.use_even_offset = False
                        new_idx = len(obj.modifiers) - 1
                        if new_idx > 0:
                            obj.modifiers.move(new_idx, 0)
                # Set object color to group color if overlay is enabled
                if group.show_overlay_color:
                    obj.color = group.color

            apply_object_overlays()
            return {"FINISHED"}

        self.report({"ERROR"}, "Group not found")
        return {"CANCELLED"}


class OBJECT_OT_RemoveObjectFromGroup(Operator):
    """Remove the selected object from the group"""

    bl_idname = "object.remove_object_from_group"
    bl_label = "Remove Object"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from .overlay import apply_object_overlays
        from ...core.pc2 import cleanup_mesh_cache

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is not None:
            index = group.assigned_objects_index
            if 0 <= index < len(group.assigned_objects):
                assigned = group.assigned_objects[index]
                from ...core.uuid_registry import resolve_assigned
                obj = resolve_assigned(assigned)
                obj_uuid = assigned.uuid
                if obj:
                    reset_object_display(obj)
                    cleanup_mesh_cache(obj)

                group.assigned_objects.remove(index)
                cleanup_pin_vertex_groups_for_object(group, obj_uuid)
                group.assigned_objects_index = safe_update_index(
                    index, len(group.assigned_objects)
                )

            apply_object_overlays()
            return {"FINISHED"}

        self.report({"ERROR"}, "Group not found")
        return {"CANCELLED"}


_cleanup_scheduled = False


def _apply_cleanup():
    """Deferred cleanup body — runs from a timer, outside depsgraph context.

    Blender 5.x forbids ID writes (collection mutation, obj.color) from
    depsgraph_update_post. This timer tick fires after the depsgraph
    evaluation completes, so the writes are allowed.
    """
    global _cleanup_scheduled
    _cleanup_scheduled = False

    from .overlay import apply_object_overlays
    from ...core.uuid_registry import resolve_assigned

    scene = bpy.context.scene
    if scene is None:
        return None
    changed = False

    assigned_names = set()
    assigned_uuids = set()
    for group in iterate_active_object_groups(scene):
        indices_to_remove = []
        for i in range(len(group.assigned_objects)):
            assigned = group.assigned_objects[i]
            obj = resolve_assigned(assigned)
            if obj is None:
                indices_to_remove.append(i)
            else:
                # resolve_assigned() already calls _sync_all_names()
                # when assigned.name drifts from obj.name, so every
                # downstream reference (pin_items, merge_pairs, saved
                # keyframes) is kept coherent through the single
                # reconciler in core/uuid_registry. Do NOT write
                # assigned.name directly here — it would skip siblings.
                assigned_names.add(obj.name)
                if assigned.uuid:
                    assigned_uuids.add(assigned.uuid)
        for i in reversed(indices_to_remove):
            obj_uuid = group.assigned_objects[i].uuid
            group.assigned_objects.remove(i)
            cleanup_pin_vertex_groups_for_object(group, obj_uuid)
            changed = True
        if indices_to_remove:
            group.assigned_objects_index = safe_update_index(
                group.assigned_objects_index, len(group.assigned_objects)
            )

    from ...core.uuid_registry import get_object_uuid
    for obj in bpy.data.objects:
        if obj.type in ("MESH", "CURVE"):
            uid = get_object_uuid(obj)
            if uid and uid in assigned_uuids:
                continue
            r, g, b, a = obj.color
            if abs(r - 1.0) > 0.01 or abs(g - 1.0) > 0.01 or abs(b - 1.0) > 0.01:
                obj.color = (1.0, 1.0, 1.0, 1.0)
                changed = True

    state = get_addon_data(scene).state
    for i in range(len(state.merge_pairs) - 1, -1, -1):
        pair = state.merge_pairs[i]
        if pair.object_a_uuid not in assigned_uuids or pair.object_b_uuid not in assigned_uuids:
            state.merge_pairs.remove(i)
            changed = True
    if state.merge_pairs_index >= len(state.merge_pairs):
        state.merge_pairs_index = max(0, len(state.merge_pairs) - 1)

    if changed:
        apply_object_overlays()
    return None


@bpy.app.handlers.persistent
def _cleanup_deleted_objects(scene, depsgraph):
    """Schedule deferred cleanup of stale group references.

    Blender 5.x disallows ID writes from depsgraph handlers, so we only
    mark that cleanup is needed and let a one-shot timer run the actual
    mutations from a permissive context.
    """
    global _cleanup_scheduled
    if _cleanup_scheduled:
        return
    _cleanup_scheduled = True
    bpy.app.timers.register(_apply_cleanup, first_interval=0.0)


classes = (
    OBJECT_OT_CreateGroup,
    OBJECT_OT_DeleteGroup,
    OBJECT_OT_DuplicateGroup,
    OBJECT_OT_DeleteAllGroups,
    OBJECT_OT_AddObjectsToGroup,
    OBJECT_OT_RemoveObjectFromGroup,
)


@bpy.app.handlers.persistent
def _reset_cleanup_flag_on_load(*_args):
    """Clear the single-shot guard when a new .blend loads.

    If a file is saved mid-cycle (flag True), then reopened, the stale
    flag would suppress the next legitimate cleanup scheduling.
    """
    global _cleanup_scheduled
    _cleanup_scheduled = False


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.app.handlers.depsgraph_update_post.append(_cleanup_deleted_objects)
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "_reset_cleanup_flag_on_load":
            bpy.app.handlers.load_post.remove(h)
    bpy.app.handlers.load_post.append(_reset_cleanup_flag_on_load)


def unregister():
    # Remove all instances (handles reload where function identity changes)
    handlers = bpy.app.handlers.depsgraph_update_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_cleanup_deleted_objects":
            handlers.remove(h)
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "_reset_cleanup_flag_on_load":
            bpy.app.handlers.load_post.remove(h)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
