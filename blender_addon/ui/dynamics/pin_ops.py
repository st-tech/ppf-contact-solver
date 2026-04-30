# File: pin_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import json

import bpy  # pyright: ignore

from bpy.types import Operator  # pyright: ignore

from ...core.utils import get_vertices_in_group, parse_vertex_index, redraw_all_areas, set_linear_interpolation
from ...models.collection_utils import safe_update_index
from ..state import iterate_active_object_groups
from .utils import get_group_from_index


class OBJECT_OT_CreatePinVertexGroup(Operator):
    """Create a vertex group from selected vertices and add it to the pin list"""

    bl_idname = "object.create_pin_vertex_group"
    bl_label = "Create"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    vg_name: bpy.props.StringProperty(  # pyright: ignore
        name="Vertex Group Name",
        default="pin",
        description="Name for the new vertex group",
    )

    @classmethod
    def poll(cls, context):
        if context.mode == "EDIT_MESH":
            obj = context.edit_object
            if obj is None or obj.type != "MESH":
                return False
            import bmesh
            bm = bmesh.from_edit_mesh(obj.data)
            if not any(v.select for v in bm.verts):
                return False
        elif context.mode == "EDIT_CURVE":
            obj = context.edit_object
            if obj is None or obj.type != "CURVE":
                return False
            # Check if any control points are selected
            has_selected = False
            for s in obj.data.splines:
                if s.type == "BEZIER":
                    if any(bp.select_control_point for bp in s.bezier_points):
                        has_selected = True
                        break
                elif s.type in ("NURBS", "POLY"):
                    if any(p.select for p in s.points):
                        has_selected = True
                        break
            if not has_selected:
                return False
        else:
            return False
        obj = context.edit_object
        # Check the object belongs to at least one active group
        from ...core.uuid_registry import get_object_uuid
        obj_uuid = get_object_uuid(obj)
        if not obj_uuid:
            return False
        scene = context.scene
        for group in iterate_active_object_groups(scene):
            if group.object_type == "STATIC":
                continue
            for assigned in group.assigned_objects:
                if assigned.uuid == obj_uuid:
                    return True
        return False

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, "vg_name")

    def execute(self, context):
        from .overlay import apply_object_overlays

        obj = context.edit_object
        if obj is None:
            self.report({"ERROR"}, "No active edit object")
            return {"CANCELLED"}

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        # Check this object is in the group (UUID comparison)
        from ...core.uuid_registry import get_object_uuid as _get_uuid
        _obj_uid = _get_uuid(obj)
        if not _obj_uid or not any(a.uuid == _obj_uid for a in group.assigned_objects):
            self.report({"ERROR"}, f"'{obj.name}' is not in this group")
            return {"CANCELLED"}

        name = self.vg_name.strip()
        if not name:
            self.report({"ERROR"}, "Name cannot be empty")
            return {"CANCELLED"}

        # Get selected indices
        if obj.type == "MESH":
            import bmesh
            bm = bmesh.from_edit_mesh(obj.data)
            selected = [v.index for v in bm.verts if v.select]
        elif obj.type == "CURVE":
            selected = []
            idx = 0
            for s in obj.data.splines:
                if s.type == "BEZIER":
                    for bp in s.bezier_points:
                        if bp.select_control_point:
                            selected.append(idx)
                        idx += 1
                elif s.type in ("NURBS", "POLY"):
                    for p in s.points:
                        if p.select:
                            selected.append(idx)
                        idx += 1
        else:
            selected = []

        if not selected:
            self.report({"ERROR"}, "No vertices selected")
            return {"CANCELLED"}

        # Switch to object mode
        bpy.ops.object.mode_set(mode="OBJECT")

        if obj.type == "MESH":
            # Use Blender vertex groups for meshes
            vg = obj.vertex_groups.get(name)
            if vg is None:
                vg = obj.vertex_groups.new(name=name)
            vg.add(selected, 1.0, "REPLACE")
        elif obj.type == "CURVE":
            # Store as custom property for curves (no native vertex group support)
            key = f"_pin_{name}"
            obj[key] = json.dumps(selected)

        # Add to pin list
        from ...core.uuid_registry import get_or_create_object_uuid, compute_vg_hash
        from ...models.groups import encode_vertex_group_identifier, decode_vertex_group_identifier
        obj_uuid = get_or_create_object_uuid(obj)
        if not obj_uuid:
            self.report({"ERROR"}, f"'{obj.name}' is not writable (library-linked)")
            return {"CANCELLED"}
        new_hash = str(compute_vg_hash(obj, name))
        # Duplicate check by UUID + vg_name — consistent with _raw_create_pin
        # and MCP handler add_pin_vertex_group.
        already_exists = False
        for item in group.pin_vertex_groups:
            if item.object_uuid != obj_uuid:
                continue
            _, item_vg = decode_vertex_group_identifier(item.name)
            if item_vg == name:
                already_exists = True
                break
        if not already_exists:
            identifier = encode_vertex_group_identifier(obj.name, name)
            item = group.pin_vertex_groups.add()
            item.name = identifier
            item.object_uuid = obj_uuid
            item.vg_hash = new_hash

        bpy.ops.object.mode_set(mode="EDIT")
        apply_object_overlays()
        self.report({"INFO"}, f"Created pin '{name}' with {len(selected)} points")
        return {"FINISHED"}


class OBJECT_OT_AddPinVertexGroup(Operator):
    """Add selected vertex group to pin list"""

    bl_idname = "object.add_pin_vertex_group"
    bl_label = "Add"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        identifier = group.pin_vertex_group_items
        if identifier == "NONE":
            return {"CANCELLED"}

        existing_identifiers = {vg.name for vg in group.pin_vertex_groups}
        if identifier not in existing_identifiers:
            from ...models.groups import decode_vertex_group_identifier
            from ...core.uuid_registry import get_or_create_object_uuid, compute_vg_hash, get_object_by_uuid

            item = group.pin_vertex_groups.add()
            item.name = identifier

            obj_name, vg_name = decode_vertex_group_identifier(identifier)
            # The identifier came from the UUID-keyed vertex-group dropdown
            # (get_vertex_group_items).  Resolve by iterating assigned
            # objects purely by UUID + vertex-group existence.
            # The decoded obj_name is a display label only — never the
            # identity gate.
            obj = None
            if obj_name and vg_name:
                for assigned in group.assigned_objects:
                    if not assigned.uuid:
                        raise ValueError(
                            f"pin_ops: assigned object has empty UUID"
                            f" (name={getattr(assigned, 'name', '?')})"
                        )
                    candidate = get_object_by_uuid(assigned.uuid)
                    if candidate is None:
                        continue
                    # Curves store pins as custom properties, not vertex groups
                    if candidate.type == "CURVE":
                        has_vg = f"_pin_{vg_name}" in candidate
                    else:
                        has_vg = (
                            hasattr(candidate, "vertex_groups")
                            and vg_name in (vg.name for vg in candidate.vertex_groups)
                        )
                    if has_vg:
                        obj = candidate
                        if candidate.name == obj_name:
                            break  # exact name still matches — prefer it
            if obj:
                resolved_uuid = get_or_create_object_uuid(obj)
                if not resolved_uuid:
                    # Object is library-linked; remove the half-built item
                    group.pin_vertex_groups.remove(len(group.pin_vertex_groups) - 1)
                    self.report({"ERROR"}, f"'{obj.name}' is not writable (library-linked)")
                    return {"CANCELLED"}
                item.object_uuid = resolved_uuid
                if vg_name:
                    item.vg_hash = str(compute_vg_hash(obj, vg_name))
                # Recover EMBEDDED_MOVE from object marker (survives remove/re-add)
                if vg_name and obj.get(f"_embedded_move_{vg_name}"):
                    op = item.operations.add()
                    op.op_type = "EMBEDDED_MOVE"
            else:
                # No assigned object resolved — remove the half-built item
                group.pin_vertex_groups.remove(len(group.pin_vertex_groups) - 1)
                self.report({"ERROR"}, "Could not resolve pin object by UUID")
                return {"CANCELLED"}

        apply_object_overlays()
        return {"FINISHED"}


class OBJECT_OT_RemovePinVertexGroup(Operator):
    """Remove selected vertex group from pin list"""

    bl_idname = "object.remove_pin_vertex_group"
    bl_label = "Remove"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from .overlay import apply_object_overlays

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        index = group.pin_vertex_groups_index
        if 0 <= index < len(group.pin_vertex_groups):
            group.pin_vertex_groups.remove(index)
            group.pin_vertex_groups_index = safe_update_index(index, len(group.pin_vertex_groups))

        apply_object_overlays()
        return {"FINISHED"}


class OBJECT_OT_RenamePinVertexGroup(Operator):
    """Rename a pin vertex group"""

    bl_idname = "object.rename_pin_vertex_group"
    bl_label = "Rename Pin"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    new_name: bpy.props.StringProperty(name="New Name")  # pyright: ignore

    def invoke(self, context, event):
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            return {"CANCELLED"}
        from ...core.uuid_registry import resolve_pin
        pin_item = group.pin_vertex_groups[idx]
        resolve_pin(pin_item)
        from ...models.groups import decode_vertex_group_identifier
        _, vg_name = decode_vertex_group_identifier(pin_item.name)
        self.new_name = vg_name or ""
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        from .overlay import apply_object_overlays
        from ...models.groups import decode_vertex_group_identifier
        from ...core.uuid_registry import resolve_pin, compute_vg_hash

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            self.report({"WARNING"}, "No pin vertex group selected")
            return {"CANCELLED"}

        pin_item = group.pin_vertex_groups[idx]
        obj = resolve_pin(pin_item)
        obj_name, old_vg = decode_vertex_group_identifier(pin_item.name)
        if not obj or not old_vg:
            self.report({"ERROR"}, "Cannot resolve pin")
            return {"CANCELLED"}

        new_vg = self.new_name.strip()
        if not new_vg or new_vg == old_vg:
            return {"CANCELLED"}

        # Rename the actual vertex group or curve custom property
        if obj.type == "MESH":
            vg = obj.vertex_groups.get(old_vg)
            if not vg:
                self.report({"ERROR"}, f"Vertex group '{old_vg}' not found")
                return {"CANCELLED"}
            vg.name = new_vg
        elif obj.type == "CURVE":
            key_old = f"_pin_{old_vg}"
            if key_old in obj:
                obj[f"_pin_{new_vg}"] = obj[key_old]
                del obj[key_old]

        # Migrate _embedded_move_ marker
        em_old = f"_embedded_move_{old_vg}"
        em_new = f"_embedded_move_{new_vg}"
        if em_old in obj and em_new not in obj:
            obj[em_new] = obj[em_old]
            del obj[em_old]

        # Update pin identifier and hash
        from ...models.groups import encode_vertex_group_identifier
        pin_item.name = encode_vertex_group_identifier(obj.name, new_vg)
        pin_item.vg_hash = str(compute_vg_hash(obj, new_vg))

        # Sync saved keyframe cache entries
        from ...core.uuid_registry import _sync_vg_name
        _sync_vg_name(pin_item.object_uuid, old_vg, new_vg)

        apply_object_overlays()
        self.report({"INFO"}, f"Renamed '{old_vg}' to '{new_vg}'")
        return {"FINISHED"}


def _pin_select(operator, context, select):
    """Select or deselect pin vertices in edit mode (like Blender's built-in)."""
    import bmesh

    from ...models.groups import decode_vertex_group_identifier

    scene = context.scene
    group = get_group_from_index(scene, operator.group_index)
    if group is None:
        operator.report({"ERROR"}, "Group not found")
        return {"CANCELLED"}

    idx = group.pin_vertex_groups_index
    if idx < 0 or idx >= len(group.pin_vertex_groups):
        operator.report({"WARNING"}, "No pin vertex group selected")
        return {"CANCELLED"}

    from ...core.uuid_registry import resolve_pin, get_object_uuid, get_object_by_uuid
    pin_item = group.pin_vertex_groups[idx]
    resolve_pin(pin_item)
    _, vg_name = decode_vertex_group_identifier(pin_item.name)
    if not pin_item.object_uuid or not vg_name:
        operator.report({"ERROR"}, "Invalid pin identifier")
        return {"CANCELLED"}

    obj = context.edit_object
    if obj is None or get_object_uuid(obj) != pin_item.object_uuid:
        expected = get_object_by_uuid(pin_item.object_uuid)
        expected_name = expected.name if expected else "<unknown>"
        operator.report({"ERROR"}, f"Edit object must be '{expected_name}'")
        return {"CANCELLED"}

    # Get pin indices
    if obj.type == "CURVE":
        key = f"_pin_{vg_name}"
        raw = obj.get(key)
        pin_indices = set(json.loads(raw)) if raw else set()
    else:
        vg = obj.vertex_groups.get(vg_name)
        if not vg:
            operator.report({"ERROR"}, f"Vertex group '{vg_name}' not found")
            return {"CANCELLED"}
        pin_indices = set(get_vertices_in_group(obj, vg))

    if not pin_indices:
        operator.report({"WARNING"}, "No pin vertices found")
        return {"CANCELLED"}

    count = 0
    if obj.type == "CURVE":
        idx = 0
        for s in obj.data.splines:
            if s.type == "BEZIER":
                for bp in s.bezier_points:
                    if idx in pin_indices:
                        bp.select_control_point = select
                        count += 1
                    idx += 1
            elif s.type in ("NURBS", "POLY"):
                for p in s.points:
                    if idx in pin_indices:
                        p.select = select
                        count += 1
                    idx += 1
    else:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        for idx in pin_indices:
            if idx < len(bm.verts):
                bm.verts[idx].select = select
                count += 1
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)

    action = "Selected" if select else "Deselected"
    operator.report({"INFO"}, f"{action} {count} points")
    return {"FINISHED"}


class OBJECT_OT_SelectPinVertices(Operator):
    """Select vertices in the active pin vertex group"""

    bl_idname = "object.select_pin_vertices"
    bl_label = "Select"

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return context.mode in ("EDIT_MESH", "EDIT_CURVE")

    def execute(self, context):
        return _pin_select(self, context, select=True)


class OBJECT_OT_DeselectPinVertices(Operator):
    """Deselect vertices in the active pin vertex group"""

    bl_idname = "object.deselect_pin_vertices"
    bl_label = "Deselect"

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return context.mode in ("EDIT_MESH", "EDIT_CURVE")

    def execute(self, context):
        return _pin_select(self, context, select=False)


class OBJECT_OT_MakePinKeyframe(Operator):
    """Insert a positional keyframe for vertices in the selected pin vertex group"""

    bl_idname = "object.make_pin_keyframe"
    bl_label = "Make Keyframe"
    bl_options = {"UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from ...models.groups import decode_vertex_group_identifier

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            self.report({"WARNING"}, "No pin vertex group selected")
            return {"CANCELLED"}

        from ...core.uuid_registry import resolve_pin, get_object_by_uuid
        pin_item = group.pin_vertex_groups[idx]
        resolve_pin(pin_item)
        _, vg_name = decode_vertex_group_identifier(pin_item.name)
        if not pin_item.object_uuid or not vg_name:
            self.report({"ERROR"}, "Invalid pin identifier")
            return {"CANCELLED"}

        obj = get_object_by_uuid(pin_item.object_uuid)
        if not obj or obj.type not in ("MESH", "CURVE"):
            self.report({"ERROR"}, "Pin object not found")
            return {"CANCELLED"}

        if obj.type == "CURVE":
            self.report({"INFO"}, "Keyframing not supported for curve pins")
            return {"FINISHED"}

        vg = obj.vertex_groups.get(vg_name)
        if not vg:
            self.report({"ERROR"}, f"Vertex group '{vg_name}' not found")
            return {"CANCELLED"}

        # Flush edit mode changes to mesh data before keyframing
        was_edit = False
        if context.active_object and context.active_object.mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")
            was_edit = True

        count = 0
        for vi in get_vertices_in_group(obj, vg):
            obj.data.vertices[vi].keyframe_insert(data_path="co")
            count += 1

        if was_edit:
            bpy.ops.object.mode_set(mode="EDIT")

        if obj.data.animation_data and obj.data.animation_data.action:
            set_linear_interpolation(obj.data.animation_data.action)

        # Add EMBEDDED_MOVE to operations list if not already present
        if not any(op.op_type == "EMBEDDED_MOVE" for op in pin_item.operations):
            op = pin_item.operations.add()
            op.op_type = "EMBEDDED_MOVE"
            for i in range(len(pin_item.operations) - 1, 0, -1):
                pin_item.operations.move(i, i - 1)
        # Store marker on object for recovery after remove/re-add
        obj[f"_embedded_move_{vg_name}"] = True

        self.report({"INFO"}, f"Keyframed {count} vertices at frame {scene.frame_current}")

        from ...core.animation import save_pin_keyframes
        save_pin_keyframes(context)
        from ...models.groups import invalidate_overlays
        invalidate_overlays()

        return {"FINISHED"}


class OBJECT_OT_DeletePinKeyframes(Operator):
    """Delete all positional keyframes for vertices in the selected pin vertex group"""

    bl_idname = "object.delete_pin_keyframes"
    bl_label = "Delete All Keyframes"
    bl_options = {"UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        from ...core.utils import _get_fcurves
        from ...models.groups import decode_vertex_group_identifier

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            self.report({"WARNING"}, "No pin vertex group selected")
            return {"CANCELLED"}

        from ...core.uuid_registry import resolve_pin, get_object_by_uuid
        pin_item = group.pin_vertex_groups[idx]
        resolve_pin(pin_item)
        _, vg_name = decode_vertex_group_identifier(pin_item.name)
        if not pin_item.object_uuid or not vg_name:
            self.report({"ERROR"}, "Invalid pin identifier")
            return {"CANCELLED"}

        obj = get_object_by_uuid(pin_item.object_uuid)
        if not obj or obj.type not in ("MESH", "CURVE"):
            self.report({"ERROR"}, "Pin object not found")
            return {"CANCELLED"}

        if obj.type == "CURVE":
            self.report({"INFO"}, "Keyframe deletion not applicable for curve pins")
            return {"FINISHED"}

        vg = obj.vertex_groups.get(vg_name)
        if not vg:
            self.report({"ERROR"}, f"Vertex group '{vg_name}' not found")
            return {"CANCELLED"}

        indices = set(get_vertices_in_group(obj, vg))
        removed = 0
        if obj.data.animation_data and obj.data.animation_data.action:
            fcurves = _get_fcurves(obj.data.animation_data.action)
            to_remove = []
            for fc in fcurves:
                if fc.data_path.startswith("vertices[") and ".co" in fc.data_path:
                    vi = parse_vertex_index(fc.data_path)
                    if vi is not None and vi in indices:
                        to_remove.append(fc)
            for fc in to_remove:
                fcurves.remove(fc)
                removed += 1

        # Remove EMBEDDED_MOVE from operations list
        for i in range(len(pin_item.operations) - 1, -1, -1):
            if pin_item.operations[i].op_type == "EMBEDDED_MOVE":
                pin_item.operations.remove(i)
                break
        from ...models.collection_utils import safe_update_index
        pin_item.operations_index = safe_update_index(pin_item.operations_index, len(pin_item.operations))
        # Remove marker from object
        key = f"_embedded_move_{vg_name}"
        if key in obj:
            del obj[key]

        self.report({"INFO"}, f"Removed {removed} fcurves")

        from ...core.animation import save_pin_keyframes
        save_pin_keyframes(context)
        from ...models.groups import invalidate_overlays
        invalidate_overlays()

        return {"FINISHED"}


class OBJECT_OT_AddPinOperation(Operator):
    """Add an operation to the selected pin"""
    bl_idname = "object.add_pin_operation"
    bl_label = "Add Operation"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    op_type: bpy.props.EnumProperty(
        items=[
            ("MOVE_BY", "Move By", "Move by delta"),
            ("SPIN", "Spin", "Rotate around axis"),
            ("SCALE", "Scale", "Scale from center"),
            ("TORQUE", "Torque", "Apply rotational force"),
        ],
    )  # pyright: ignore

    def execute(self, context):
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[idx]
        # Torque cannot be mixed with other operation types
        existing_types = {o.op_type for o in pin_item.operations}
        if self.op_type == "TORQUE" and existing_types - {"TORQUE", "EMBEDDED_MOVE"}:
            self.report({"ERROR"}, "Torque cannot be mixed with Move/Spin/Scale operations")
            return {"CANCELLED"}
        if self.op_type != "TORQUE" and "TORQUE" in existing_types:
            self.report({"ERROR"}, "Cannot add Move/Spin/Scale to a pin that has Torque")
            return {"CANCELLED"}
        op = pin_item.operations.add()
        op.op_type = self.op_type
        # Default center mode to CENTROID for scale and spin
        # (the solver computes the center from vertex positions at runtime)
        if self.op_type == "SCALE":
            op.scale_center_mode = "CENTROID"
        elif self.op_type == "SPIN":
            op.spin_center_mode = "CENTROID"
        elif self.op_type == "TORQUE":
            op.torque_axis_component = "PC3"
        # Insert at head
        pin_item.operations.move(len(pin_item.operations) - 1, 0)
        pin_item.operations_index = 0
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_RemovePinOperation(Operator):
    """Remove the selected operation from the pin"""
    bl_idname = "object.remove_pin_operation"
    bl_label = "Remove Operation"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[idx]
        op_idx = pin_item.operations_index
        if op_idx < 0 or op_idx >= len(pin_item.operations):
            return {"CANCELLED"}
        if pin_item.operations[op_idx].op_type == "EMBEDDED_MOVE":
            # Delegate to delete_pin_keyframes operator
            bpy.ops.object.delete_pin_keyframes(group_index=self.group_index)
            return {"FINISHED"}
        pin_item.operations.remove(op_idx)
        pin_item.operations_index = safe_update_index(op_idx, len(pin_item.operations))
        from ...models.groups import invalidate_overlays
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_MovePinOperation(Operator):
    """Move the selected operation up or down"""
    bl_idname = "object.move_pin_operation"
    bl_label = "Move Operation"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    direction: bpy.props.IntProperty()  # pyright: ignore  # -1 = up, 1 = down

    def execute(self, context):
        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[idx]
        op_idx = pin_item.operations_index
        new_idx = op_idx + self.direction
        if op_idx < 0 or op_idx >= len(pin_item.operations):
            return {"CANCELLED"}
        if new_idx < 0 or new_idx >= len(pin_item.operations):
            return {"CANCELLED"}
        pin_item.operations.move(op_idx, new_idx)
        pin_item.operations_index = new_idx
        return {"FINISHED"}


class OBJECT_OT_PickCenterFromSelected(Operator):
    """Set the absolute center from the centroid of selected vertices"""

    bl_idname = "object.pick_center_from_selected"
    bl_label = "Pick from Selected"
    bl_options = {"UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    target: bpy.props.StringProperty()  # pyright: ignore  # "spin" or "scale"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == "MESH"
            and obj.mode == "EDIT"
        )

    def execute(self, context):
        import numpy as np

        obj = context.active_object
        bpy.ops.object.mode_set(mode="OBJECT")
        selected = [v for v in obj.data.vertices if v.select]
        bpy.ops.object.mode_set(mode="EDIT")

        if not selected:
            self.report({"WARNING"}, "No vertices selected")
            return {"CANCELLED"}

        mat = obj.matrix_world
        centroid = np.mean([list(mat @ v.co) for v in selected], axis=0)

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[idx]
        op_idx = pin_item.operations_index
        if op_idx < 0 or op_idx >= len(pin_item.operations):
            return {"CANCELLED"}
        op = pin_item.operations[op_idx]

        if self.target == "spin":
            op.spin_center = tuple(centroid)
        elif self.target == "scale":
            op.scale_center = tuple(centroid)

        self.report({"INFO"}, f"Center set from {len(selected)} vertices")
        return {"FINISHED"}


class OBJECT_OT_PickVertexCenter(Operator):
    """Set the center vertex from a single selected vertex in Edit Mode"""

    bl_idname = "object.pick_vertex_center"
    bl_label = "Pick Vertex"
    bl_options = {"UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore
    target: bpy.props.StringProperty()  # pyright: ignore  # "spin" or "scale"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == "MESH"
            and obj.mode == "EDIT"
        )

    def execute(self, context):
        obj = context.active_object
        bpy.ops.object.mode_set(mode="OBJECT")
        selected = [v for v in obj.data.vertices if v.select]
        bpy.ops.object.mode_set(mode="EDIT")

        if len(selected) != 1:
            self.report({"WARNING"}, "Select exactly one vertex")
            return {"CANCELLED"}

        vi = selected[0].index

        scene = context.scene
        group = get_group_from_index(scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        idx = group.pin_vertex_groups_index
        if idx < 0 or idx >= len(group.pin_vertex_groups):
            return {"CANCELLED"}
        pin_item = group.pin_vertex_groups[idx]
        op_idx = pin_item.operations_index
        if op_idx < 0 or op_idx >= len(pin_item.operations):
            return {"CANCELLED"}
        op = pin_item.operations[op_idx]

        if self.target == "spin":
            op.spin_center_vertex = vi
        elif self.target == "scale":
            op.scale_center_vertex = vi

        self.report({"INFO"}, f"Center vertex set to {vi}")
        return {"FINISHED"}


classes = (
    OBJECT_OT_CreatePinVertexGroup,
    OBJECT_OT_AddPinVertexGroup,
    OBJECT_OT_RemovePinVertexGroup,
    OBJECT_OT_RenamePinVertexGroup,
    OBJECT_OT_SelectPinVertices,
    OBJECT_OT_DeselectPinVertices,
    OBJECT_OT_MakePinKeyframe,
    OBJECT_OT_DeletePinKeyframes,
    OBJECT_OT_AddPinOperation,
    OBJECT_OT_RemovePinOperation,
    OBJECT_OT_MovePinOperation,
    OBJECT_OT_PickCenterFromSelected,
    OBJECT_OT_PickVertexCenter,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
