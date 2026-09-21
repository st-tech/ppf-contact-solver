"""Group management handlers (create, delete, objects, pins, materials)."""

import contextlib
import struct

import bpy  # pyright: ignore

from ...models.collection_utils import safe_update_index
from ...models.intersection_allowances import (
    INTERSECTION_ALLOWANCES,
    allowance_by_key,
    allowance_objects,
    allowed_object_uuids,
)
from ...models.groups import (
    assign_display_indices,
    decode_vertex_group_identifier,
    get_active_group_by_uuid,
    get_addon_data,
    get_group_slot_index,
    iterate_active_object_groups,
    iterate_object_groups,
)
from ..decorators import (
    MCPError,
    ValidationError,
    group_handler,
)


def get_active_group_by_uuid_helper(group_uuid: str):
    """Helper function to get and validate active group by UUID."""
    scene = bpy.context.scene
    group = get_active_group_by_uuid(scene, group_uuid)

    if not group:
        raise ValidationError(f"Group with UUID {group_uuid} not found or not active")

    return group


def get_group_index_by_uuid(group_uuid: str):
    """Return the ``object_group_N`` slot of an active group, or raise.

    The UI operators take that slot as their ``group_index`` property. It is
    not ``ObjectGroup.index``, which numbers only the active groups for
    display and so disagrees with the slot once any group has been deleted,
    so the slot comes from ``models.groups.get_group_slot_index``, the one
    resolver for it. Used by the object_ops handlers; the handlers in this
    module resolve the slot at their call sites.
    """
    scene = bpy.context.scene
    slot = get_group_slot_index(scene, group_uuid)
    if slot is None or not get_active_group_by_uuid(scene, group_uuid):
        raise ValidationError(f"Active group with UUID {group_uuid} not found")
    return slot


def resolve_assigned_with_index(group_uuid: str, object_name: str):
    """Return (group, assigned, idx, obj_uuid) for an object in a group.

    Resolves the group by UUID, the object by name -> UUID, then locates the
    assigned-object row by a single scan. Raises MCPError if the group is
    unknown, the object is missing/has no UUID, or it is not a member.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    from ...core.uuid_registry import get_object_uuid

    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise MCPError(f"Object '{object_name}' not found in scene")
    obj_uuid = get_object_uuid(obj)
    if not obj_uuid:
        raise MCPError(f"Object '{object_name}' has no UUID")
    for i, assigned in enumerate(group.assigned_objects):
        if assigned.uuid == obj_uuid:
            return group, assigned, i, obj_uuid
    raise MCPError(f"Object '{object_name}' not in group {group_uuid}")


def _serialize_group(group) -> dict:
    """Return a stable MCP-facing representation of an active group."""
    return {
        "uuid": group.uuid or "",
        "name": group.name,
        "object_type": group.object_type,
        "active": group.active,
        "assigned_objects": [
            {
                "name": obj.name,
                "uuid": obj.uuid,
                "included": bool(obj.included),
            }
            for obj in group.assigned_objects
        ],
        "object_count": len(group.assigned_objects),
        "pin_count": len(group.pin_vertex_groups),
        "show_overlay_color": bool(group.show_overlay_color),
    }


def _list_groups() -> list[dict]:
    """Serialize every active group in display order.

    This does not normalize the stored display indices. Renumbering them
    would write to the scene, and `get_active_groups` is annotated
    readOnlyHint, which is a factual claim a client may act on. The order is
    the iteration order either way: renumbering assigns indices FROM that
    order rather than deriving it, and nothing serialized here reads an index.
    """
    scene = bpy.context.scene
    return [_serialize_group(group) for group in iterate_active_object_groups(scene)]


def _add_pin_vertex_group_impl(
    group_uuid: str,
    vertex_group_identifier: str,
    indices: list[int] | None,
) -> dict:
    """Shared implementation for mesh and curve pin registration."""
    group = get_active_group_by_uuid_helper(group_uuid)
    obj_name, vg_name = _parse_pin_identifier(vertex_group_identifier)

    # Mirror the scripting API (ops/api/group.py::Group.create_pin): the only
    # handler-side responsibility is writing the curve's _pin_<vg> property when
    # indices are supplied. Existence/type/vertex-group/curve-pin validation is
    # owned by mutation.create_pin so the LLM sees one unified error text.
    obj = bpy.data.objects.get(obj_name)
    if obj is not None and indices is not None:
        if obj.type != "CURVE":
            raise ValidationError(
                f"'indices' is only valid for curve objects; "
                f"'{obj_name}' is {obj.type}. A mesh pin names a vertex "
                "group that already holds the vertices: create it with "
                "create_vertex_group, then pin it by name."
            )
        import json

        try:
            encoded_indices = [int(index) for index in indices]
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "indices must be a list of integers for curve pins"
            ) from exc
        obj[f"_pin_{vg_name}"] = json.dumps(encoded_indices)

    from ...core import mutation

    try:
        result = mutation.create_pin(group_uuid, obj_name, vg_name)
    except mutation.MutationError as exc:
        raise MCPError(str(exc)) from exc

    return {
        "message": f"Added pin '{obj_name}::{vg_name}' to group {group_uuid}",
        "group_uuid": group_uuid,
        "object_name": obj_name,
        "object_uuid": result["object_uuid"],
        "vertex_group_name": vg_name,
        "pin_count": len(group.pin_vertex_groups),
    }


@group_handler
def create_group(name: str = "", type: str = "SOLID"):
    """Create a new dynamics group.

    Args:
        name: Display name for the new group (optional)
        type: Group type (SOLID, SHELL, ROD, STATIC, PDRD, SAND)
    """
    valid_types = {"SOLID", "SHELL", "ROD", "STATIC", "PDRD", "SAND"}
    if type not in valid_types:
        raise ValidationError(
            f"Invalid type '{type}'. Valid types: {sorted(valid_types)}"
        )

    # Use UI operator instead of direct manipulation
    bpy.ops.object.create_group()

    # Get the newly created group info
    scene = bpy.context.scene

    # Assign display indices to ensure consistency
    assign_display_indices(scene)

    # The operator publishes the group it allocated. Read that rather than
    # scanning for the highest active slot: the allocated slot is the lowest
    # free one, so after any deletion the two are different groups and the
    # scan would rename and retype a bystander.
    group_uuid = get_addon_data(scene).state.current_group_uuid
    created = get_active_group_by_uuid(scene, group_uuid) if group_uuid else None

    if created:
        if name:
            created.name = name
        if type != "SOLID":
            created.object_type = type
        return {
            "message": f"Created group: {created.name}",
            "group": _serialize_group(created),
            "group_uuid": group_uuid,
        }

    raise MCPError("Failed to create group")


@group_handler
def delete_group(group_uuid: str):
    """Delete a specific group by UUID.

    Args:
        group_uuid: UUID of group to delete
    """
    scene = bpy.context.scene
    group = get_active_group_by_uuid(scene, group_uuid)
    if not group:
        raise MCPError(f"Group with UUID {group_uuid} not found or not active")

    # Set the group UUID in scene state for the operator
    get_addon_data(scene).state.current_group_uuid = group_uuid

    # Use UI operator for group deletion. It addresses a group by its
    # object_group_N slot, so resolve the slot rather than passing the UUID
    # (which it has no property for) or ObjectGroup.index (which counts only
    # active groups and so drifts from the slot once one has been deleted).
    slot = get_group_slot_index(scene, group_uuid)
    if slot is None:
        raise MCPError(f"Group with UUID {group_uuid} has no slot")
    bpy.ops.object.delete_group(group_index=slot)

    return {
        "message": f"Deleted group with UUID {group_uuid}",
        "group_uuid": group_uuid,
    }


@group_handler
def delete_all_groups():
    """Delete all active groups."""
    # Count active groups before deletion
    scene = bpy.context.scene
    deleted_count = len([g for g in iterate_object_groups(scene) if g.active])

    # Use UI operator for group deletion
    bpy.ops.object.delete_all_groups()

    return {
        "message": f"Deleted {deleted_count} groups",
        "deleted_count": deleted_count,
    }


@group_handler
def duplicate_group(group_uuid: str):
    """Duplicate a dynamics group (material params only, no objects or pins).

    Args:
        group_uuid: UUID of the source group to duplicate
    """
    scene = bpy.context.scene
    src = get_active_group_by_uuid_helper(group_uuid)
    src_name = src.name
    src_index = get_group_slot_index(scene, group_uuid)
    if src_index is None:
        raise MCPError(f"Group with UUID {group_uuid} has no slot")

    before_uuids = {
        g.uuid for g in iterate_active_object_groups(scene) if g.uuid
    }
    bpy.ops.object.duplicate_group(group_index=src_index)

    assign_display_indices(scene)
    new_group = None
    for g in iterate_active_object_groups(scene):
        if g.uuid and g.uuid not in before_uuids:
            new_group = g
            break
    if new_group is None:
        raise MCPError(f"Failed to duplicate group {group_uuid}")
    return {
        "message": f"Duplicated group '{src_name}' -> '{new_group.name}'",
        "source_group_uuid": group_uuid,
        "group_uuid": new_group.uuid,
        "group_name": new_group.name,
    }


@group_handler
def rename_group(group_uuid: str, name: str):
    """Rename a dynamics group.

    Args:
        group_uuid: UUID of group to rename
        name: New display name (empty string falls back to 'Group N')
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    old_name = group.name
    group.name = name
    return {
        "message": f"Renamed group '{old_name}' -> '{group.name}'",
        "group_uuid": group_uuid,
        "name": group.name,
    }


@group_handler
def bake_group_animation(group_uuid: str, object_name: str):
    """Bake simulated animation for one object in a group to Blender keyframes.

    The object is removed from the group and keeps its baked animation.

    Args:
        group_uuid: UUID of group containing the object
        object_name: Name of the object to bake
    """
    group, _, idx, obj_uuid = resolve_assigned_with_index(group_uuid, object_name)
    group.assigned_objects_index = idx

    group_index = get_group_slot_index(bpy.context.scene, group_uuid)
    if group_index is None:
        raise MCPError(f"Group with UUID {group_uuid} has no slot")
    bpy.ops.object.bake_animation("EXEC_DEFAULT", group_index=group_index)
    return {
        "message": f"Baked animation for '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
    }


@group_handler
def bake_group_single_frame(group_uuid: str, object_name: str):
    """Bake the current frame as frame 1 for one object and drop it from the group.

    Args:
        group_uuid: UUID of group containing the object
        object_name: Name of the object to bake
    """
    group, _, idx, obj_uuid = resolve_assigned_with_index(group_uuid, object_name)
    group.assigned_objects_index = idx

    group_index = get_group_slot_index(bpy.context.scene, group_uuid)
    if group_index is None:
        raise MCPError(f"Group with UUID {group_uuid} has no slot")
    bpy.ops.object.bake_single_frame("EXEC_DEFAULT", group_index=group_index)
    return {
        "message": f"Baked single frame for '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
    }


@group_handler
def set_object_included(group_uuid: str, object_name: str, included: bool):
    """Toggle whether an assigned object is included in the simulation.

    Args:
        group_uuid: UUID of the group
        object_name: Name of the assigned object
        included: True to include, False to mute
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    from ...core.uuid_registry import get_object_uuid
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise MCPError(f"Object '{object_name}' not found in scene")
    obj_uuid = get_object_uuid(obj)
    if not obj_uuid:
        raise MCPError(f"Object '{object_name}' has no UUID")

    for assigned in group.assigned_objects:
        if assigned.uuid == obj_uuid:
            assigned.included = bool(included)
            return {
                "message": f"Set '{object_name}' included={included}",
                "group_uuid": group_uuid,
                "object_name": object_name,
                "object_uuid": obj_uuid,
                "included": bool(included),
            }
    raise MCPError(f"Object '{object_name}' not in group {group_uuid}")


@group_handler
def get_group(group_uuid: str):
    """Get one active group by UUID.

    Args:
        group_uuid: UUID of group
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    return {"group": _serialize_group(group)}


@group_handler
def get_active_groups():
    """Get list of all active groups with their properties."""
    groups = _list_groups()
    return {"groups": groups, "group_count": len(groups)}


@group_handler
def add_objects_to_group(group_uuid: str, object_names: list[str]):
    """Add objects to a dynamics group.

    Args:
        group_uuid: UUID of target group
        object_names: List of object names to add
    """
    scene = bpy.context.scene
    from ...core.uuid_registry import get_or_create_object_uuid, get_object_by_uuid

    # Validate the target group up front so we don't partially
    # mutate selection/state before discovering the UUID is bad.
    target_group = get_active_group_by_uuid(scene, group_uuid)
    if not target_group:
        raise MCPError(f"Group with UUID {group_uuid} not found or not active")

    # Build UUID-based assignment set so rename-after-assign does not
    # cause spurious "not assigned" or duplicate-assign errors.
    all_assigned_uuids = {
        obj.uuid
        for grp in iterate_active_object_groups(scene)
        for obj in grp.assigned_objects
        if obj.uuid
    }

    # Clear current selection
    bpy.ops.object.select_all(action="DESELECT")

    valid_objects = []
    warnings = []

    # Resolve each caller-supplied name to its UUID, then work off UUID.
    for obj_name in object_names:
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            warnings.append(f"Object '{obj_name}' not found")
            continue

        # Mirror the UI operator's acceptance rule
        # (ui/dynamics/group_ops.py::OBJECT_OT_AddObjectsToGroup): a
        # ROD group also accepts Bezier curves; every other group type
        # is mesh-only.
        is_acceptable = obj.type == "MESH" or (
            obj.type == "CURVE" and target_group.object_type == "ROD"
        )
        if not is_acceptable:
            warnings.append(
                f"Object '{obj_name}' has type {obj.type!r} which is not "
                f"accepted by a {target_group.object_type} group"
            )
            continue

        obj_uuid = get_or_create_object_uuid(obj)
        if not obj_uuid:
            raise MCPError(
                f"Object '{obj_name}' has no UUID and one could not be created; "
                "the object may be library-linked or read-only"
            )
        # Re-resolve via UUID so subsequent ops don't rely on name identity.
        obj = get_object_by_uuid(obj_uuid) or obj

        if obj_uuid in all_assigned_uuids:
            warnings.append(
                f"Object '{obj.name}' (uuid={obj_uuid}) already assigned to a group"
            )
            continue

        # Note: Triangulation is now handled during data transfer

        # Select object for the UI operator
        obj.select_set(True)
        valid_objects.append({"name": obj.name, "uuid": obj_uuid})

    if valid_objects:
        # Resolve UUID to the object_group_N slot the operator addresses
        group_index = get_group_slot_index(scene, group_uuid)
        if group_index is None:
            raise MCPError(f"Group with UUID {group_uuid} has no slot")

        # Use UI operator to add selected objects to group
        bpy.ops.object.add_objects_to_group(group_index=group_index)

    return {
        "message": f"Added {len(valid_objects)} objects to group {group_uuid}",
        "added_objects": valid_objects,
        "warnings": warnings,
        "group_uuid": group_uuid,
    }


@group_handler
def remove_object_from_group(group_uuid: str, object_name: str):
    """Remove an object from a dynamics group.

    Args:
        group_uuid: UUID of group
        object_name: Name of object to remove
    """
    # Verify object exists in group and get its index
    group = get_active_group_by_uuid_helper(group_uuid)
    from ...core.uuid_registry import get_object_uuid, get_object_by_uuid
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise MCPError(f"Object '{object_name}' not found in scene")
    obj_uuid = get_object_uuid(obj)
    if not obj_uuid:
        raise MCPError(
            f"Object '{object_name}' has no UUID; "
            "save and reload the .blend to trigger auto-migration"
        )
    # Re-resolve via UUID so we work off UUID identity from here on.
    obj = get_object_by_uuid(obj_uuid) or obj

    object_index = -1
    for i, assigned_obj in enumerate(group.assigned_objects):
        if assigned_obj.uuid == obj_uuid:
            object_index = i
            break

    if object_index == -1:
        raise MCPError(
            f"Object '{obj.name}' (uuid={obj_uuid}) not found in group {group_uuid}"
        )

    # Direct manipulation instead of using UI operator to avoid poll issues
    # Clean up pin vertex groups for this object first
    from ...ui.dynamics.utils import (
        cleanup_group_references_for_object,
        reset_object_display,
    )

    cleanup_group_references_for_object(group, obj_uuid)

    # Clear the display state the add-on put on this object while it is
    # still a member. Once the assignment is gone the object is the
    # user's again and nothing may write to it, so this is the last
    # point at which the tint and wireframe can be taken back.
    reset_object_display(obj)

    # Remove the object from the group
    group.assigned_objects.remove(object_index)

    # Adjust assigned_objects_index if needed
    if len(group.assigned_objects) == 0:
        group.assigned_objects_index = -1
    else:
        group.assigned_objects_index = safe_update_index(
            group.assigned_objects_index, len(group.assigned_objects)
        )

    # Clean up merge pairs that reference the removed object
    scene = bpy.context.scene
    state = get_addon_data(scene).state
    for i in range(len(state.merge_pairs) - 1, -1, -1):
        pair = state.merge_pairs[i]
        if pair.object_a_uuid == obj_uuid or pair.object_b_uuid == obj_uuid:
            state.merge_pairs.remove(i)
    if state.merge_pairs_index >= len(state.merge_pairs):
        state.merge_pairs_index = max(0, len(state.merge_pairs) - 1)

    # Apply overlay updates
    from ...ui.dynamics import apply_object_overlays

    apply_object_overlays()

    return {
        "message": f"Removed object '{obj.name}' from group {group_uuid}",
        "object_name": obj.name,
        "object_uuid": obj_uuid,
        "group_uuid": group_uuid,
    }


@group_handler
def remove_all_objects_from_group(group_uuid: str):
    """Remove all objects from a dynamics group.

    Args:
        group_uuid: UUID of group to clear
    """
    # Verify group exists first
    group = get_active_group_by_uuid_helper(group_uuid)

    # Check if there are objects to remove
    if len(group.assigned_objects) == 0:
        return {
            "message": f"Group {group_uuid} is already empty",
            "group_uuid": group_uuid,
            "objects_removed": 0,
        }

    # Store count before removal
    object_count = len(group.assigned_objects)

    # Clear the add-on's display state on every member while they are
    # still members; after the clear below they are the user's objects
    # and nothing may write to them.
    from ...core.uuid_registry import resolve_assigned
    from ...ui.dynamics.utils import reset_object_display

    for assigned in group.assigned_objects:
        member = resolve_assigned(assigned)
        if member is not None:
            reset_object_display(member)

    # Direct manipulation - clear all objects and related data
    group.pin_vertex_groups.clear()
    group.pin_vertex_groups_index = -1
    # Both intersection-allowance subsets name members of this group, so
    # emptying the group empties them: an entry that outlived its object
    # would be drawn in the panel and would name a different object once
    # this slot was reused.
    for spec in INTERSECTION_ALLOWANCES:
        allowance_objects(group, spec).clear()
        setattr(group, spec.index_prop, -1)
    group.assigned_objects.clear()
    group.assigned_objects_index = -1

    # Apply overlay updates
    from ...ui.dynamics import apply_object_overlays

    apply_object_overlays()

    return {
        "message": f"Removed all {object_count} objects from group {group_uuid}",
        "group_uuid": group_uuid,
        "objects_removed": object_count,
    }


def _bend_reference(assigned) -> dict:
    """The per-object bending rest-angle reference, as a caller sees it.

    The reference object is stored by UUID so it survives a rename, and the
    name beside it is only the label the panel last drew.
    """
    from ...core.uuid_registry import get_object_by_uuid

    object_uuid = assigned.bend_ref_uuid or ""
    # get_object_by_uuid raises on an empty UUID, which is what an object with
    # no reference picked yet carries.
    reference = get_object_by_uuid(object_uuid) if object_uuid else None
    return {
        "enabled": bool(assigned.bend_ref_enable),
        "object_uuid": object_uuid,
        "object_name": reference.name if reference is not None else None,
        "stored_name": assigned.bend_ref_name,
    }


def _tet_settings(assigned) -> dict:
    """The per-object tetrahedralizer backend and the overrides switched on."""
    # Which flag switches on which value is set_object_tet_settings's own
    # table, so both tools read one source and cannot disagree about what an
    # override is.
    from .object_ops import _TET_OVERRIDE_FIELDS

    return {
        "backend": assigned.tet_backend,
        # Only the overrides that are on. A field absent here is left to the
        # backend's own default, whatever value the record still holds for it.
        "overrides": {
            field: getattr(assigned, field)
            for field, flag in _TET_OVERRIDE_FIELDS.items()
            if getattr(assigned, flag)
        },
    }


def _intersection_allowances(group, assigned) -> dict:
    """Which intersection allowances reach THIS object, and by which route.

    A group's allowance checkbox is one of two halves: the allowance covers
    every object of the group, or only the objects named in its subset. So a
    per-object answer cannot be read off the group's booleans alone, and
    "allowed" here is what the scene build will actually give this object,
    with "scope" saying whether that came from the group-wide switch or from
    the object being named. set_intersection_allowance_objects writes the
    subset; set_group_material_properties writes the two switches.
    """
    report = {}
    for spec in INTERSECTION_ALLOWANCES:
        allowed = allowed_object_uuids(group, spec)
        report[spec.key] = {
            "allowed": assigned.uuid in allowed,
            "scope": (
                "all_objects"
                if getattr(group, spec.all_objects_prop)
                else "listed_objects"
            ),
        }
    return report


def _serialize_assigned_object(group, assigned, obj) -> dict:
    """One assigned object, with the per-object state the object tools write.

    The lock field names are the ones set_object_locks takes, and pca_axis is
    the one set_pdrd_hinge takes, so a value read here can be passed straight
    back to the tool that writes it.
    """
    return {
        "name": obj.name,
        "uuid": assigned.uuid,
        "type": obj.type,
        "vertex_count": len(obj.data.vertices) if obj.type == "MESH" else 0,
        "face_count": len(obj.data.polygons) if obj.type == "MESH" else 0,
        "included": bool(assigned.included),
        "locks": {
            "lock_translation_enable": bool(assigned.lock_translation_enable),
            "lock_translation_all": bool(assigned.lock_translation_all),
            "lock_translation_axis": [
                float(component) for component in assigned.lock_translation_axis
            ],
            "lock_rotation_enable": bool(assigned.lock_rotation_enable),
            "lock_rotation_all": bool(assigned.lock_rotation_all),
            "lock_rotation_axis": [
                float(component) for component in assigned.lock_rotation_axis
            ],
            "lock_rotation_prohibit_axis": bool(
                assigned.lock_rotation_prohibit_axis
            ),
        },
        "pdrd_hinge": {
            "enabled": bool(assigned.pdrd_hinge_enable),
            "pca_axis": int(assigned.pdrd_hinge_axis),
        },
        "bend_reference": _bend_reference(assigned),
        "intersection_allowances": _intersection_allowances(group, assigned),
        "tet": _tet_settings(assigned),
        "static_op_count": len(assigned.static_ops),
        "velocity_keyframe_count": len(assigned.velocity_keyframes),
        "collision_window_count": len(assigned.collision_windows),
    }


@group_handler
def get_group_objects(group_uuid: str):
    """Get the objects assigned to a dynamics group, with their own state.

    Every per-object field the object tools write is reported: "included"
    from set_object_included, "locks" from set_object_locks, "pdrd_hinge"
    from set_pdrd_hinge, and "tet" from set_object_tet_settings. The three
    counts say how many rows each per-object list holds; read the rows with
    list_static_ops, list_velocity_keyframes and list_collision_windows.

    Material parameters are per GROUP, not per object, and are reported by
    get_group_material_properties instead.

    A field is stored on every assigned object whatever the group type, and
    the scene build reads it only where it applies: a hinge on PDRD, a
    tetrahedralizer setting on SOLID, a bending reference on SHELL and ROD.
    So a value reported for a type that does not use it is stored and unread.

    "bend_reference" is this object's own bending rest angle source, which
    replaces the group's for that object, and which the scene build reads
    only while the group's bend_rest_from_reference is on. Its object is
    stored by UUID: "object_name" is what that UUID resolves to in the scene
    now, and is null when it resolves to nothing, which is a reference whose
    object has left the scene.

    Args:
        group_uuid: UUID of group
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    objects = []

    from ...core.uuid_registry import resolve_assigned
    for assigned_obj in group.assigned_objects:
        obj = resolve_assigned(assigned_obj)
        if obj is None:
            raise MCPError(
                f"Cannot resolve object with UUID '{assigned_obj.uuid}' "
                f"in group {group_uuid}; the object may have been deleted "
                "or the .blend needs re-migration"
            )
        objects.append(_serialize_assigned_object(group, assigned_obj, obj))

    return {
        "group_uuid": group_uuid,
        "group_type": group.object_type,
        "objects": objects,
        "object_count": len(objects),
    }


@group_handler
def set_intersection_allowance_objects(
    group_uuid: str,
    allowance: str,
    object_names: list[str],
):
    """Narrow one intersection allowance to named objects of a group.

    An allowance ("allow_self_intersection" or
    "allow_inter_object_intersection", both set by
    set_group_material_properties) reaches every object of its group while
    the matching "..._all_objects" switch is on. This tool writes the subset
    the allowance reaches instead, and turns that switch OFF, so the
    allowance covers exactly the objects named here and no others. Pass an
    empty list to clear the subset, which leaves the allowance reaching
    nothing; turn "..._all_objects" back on with
    set_group_material_properties to go back to covering the whole group.

    The subset is stored whether or not the allowance itself is enabled, and
    the allowance is not enabled as a side effect: an allowance that is off
    reports every intersection whatever this list holds.
    get_group_objects reports, per object, whether an allowance reaches it
    and by which of the two routes.

    Every name must be an object currently assigned to this group. A name
    that is not is refused and nothing is written, because an allowance
    stored for a non-member would reach no vertex at build time while this
    call reported success.

    Args:
        group_uuid: UUID of group
        allowance: "self" or "inter_object"
        object_names: Objects of this group the allowance is narrowed to
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    try:
        spec = allowance_by_key(allowance)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    if not isinstance(object_names, list):
        raise ValidationError(
            "object_names must be a list of object names, "
            f"not {type(object_names).__name__}"
        )

    from ...core.uuid_registry import resolve_assigned

    # Resolve names through the group's own membership, not through
    # bpy.data, so the one refusal below covers an object that is missing
    # and one that is present but belongs to another group.
    member_uuid_by_name = {}
    for assigned in group.assigned_objects:
        member = resolve_assigned(assigned)
        if member is not None:
            member_uuid_by_name[member.name] = assigned.uuid

    resolved: list[tuple[str, str]] = []
    outsiders: list[str] = []
    seen: set[str] = set()
    for name in object_names:
        obj_uuid = member_uuid_by_name.get(name)
        if obj_uuid is None:
            outsiders.append(name)
            continue
        if obj_uuid in seen:
            continue
        seen.add(obj_uuid)
        resolved.append((name, obj_uuid))

    if outsiders:
        raise ValidationError(
            f"Not assigned to group {group_uuid}: {sorted(outsiders)}. "
            f"Its members are {sorted(member_uuid_by_name)}"
        )

    collection = allowance_objects(group, spec)
    collection.clear()
    for name, obj_uuid in resolved:
        item = collection.add()
        item.name = name
        item.uuid = obj_uuid
    setattr(group, spec.index_prop, len(collection) - 1)
    setattr(group, spec.all_objects_prop, False)

    return {
        "message": (
            f"Narrowed {spec.enable_prop} on group {group_uuid} to "
            f"{len(resolved)} object(s)"
        ),
        "group_uuid": group_uuid,
        "allowance": spec.key,
        "enabled": bool(getattr(group, spec.enable_prop)),
        "applies_to_all_objects": False,
        "object_names": [name for name, _uuid in resolved],
        "object_uuids": [obj_uuid for _name, obj_uuid in resolved],
    }


@group_handler
def set_group_type(group_uuid: str, type: str):
    """Set the type of a dynamics group.

    Args:
        group_uuid: UUID of group
        type: Group type (SOLID, SHELL, ROD, STATIC, PDRD, SAND)
    """
    valid_types = ["SOLID", "SHELL", "ROD", "STATIC", "PDRD", "SAND"]
    if type not in valid_types:
        raise ValidationError(f"Invalid type '{type}'. Valid types: {valid_types}")

    group = get_active_group_by_uuid_helper(group_uuid)
    group.object_type = type

    return {
        "message": f"Set group {group_uuid} type to {type}",
        "group_uuid": group_uuid,
        "type": type,
    }


# ---------------------------------------------------------------------------
# Blender vertex groups (the membership a mesh pin names)
# ---------------------------------------------------------------------------


def _resolve_mesh_object(object_name: str):
    """Return the named object, raising unless it exists and is a MESH.

    Vertex groups are a mesh property. A curve carries its pinned control
    points in a ``_pin_<name>`` custom property instead, written by
    add_pin_vertex_group's ``indices`` argument.
    """
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise MCPError(f"Object '{object_name}' not found in scene")
    if obj.type != "MESH":
        raise MCPError(
            f"Object '{object_name}' is type {obj.type}; only a MESH carries "
            "vertex groups. A curve pins control points instead: pass 'indices' "
            "to add_pin_vertex_group."
        )
    return obj


@contextlib.contextmanager
def _object_mode(obj):
    """Hold *obj* in Object Mode for the body, then return it to its own mode.

    ``VertexGroups.add`` refuses an object that is in Edit Mode, and an Edit
    Mode session holds its geometry and its weights in a BMesh that
    ``obj.data.vertices`` does not reflect until the mode is left, so both the
    index range checked against the mesh and the write itself require Object
    Mode. Leaving Edit Mode writes the session back to the mesh, so the edit
    the caller made is what the indices then address.

    ``mode_set`` acts on the active object, and an object in Edit Mode is
    either that object or a participant in the same multi-object edit session,
    so leaving the mode covers *obj* either way.
    """
    saved_mode = obj.mode
    if saved_mode == "OBJECT":
        yield
        return
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError as exc:
        raise MCPError(
            f"Could not leave {saved_mode} mode on '{obj.name}': {exc}"
        ) from exc
    completed = False
    try:
        yield
        completed = True
    finally:
        try:
            bpy.ops.object.mode_set(mode=saved_mode)
        except RuntimeError as exc:
            # Only reportable when the body succeeded. Raising here after a
            # failed body would replace the error the caller has to act on.
            if completed:
                raise MCPError(
                    f"'{obj.name}' was left in Object Mode: returning it to "
                    f"{saved_mode} mode failed ({exc}). The work it was taken "
                    "out of that mode for did complete."
                ) from exc


def _vertex_group_counts(obj) -> dict[int, int]:
    """Assigned vertex count per vertex-group index, in one pass over the mesh.

    A vertex counts toward every group that lists it, at whatever weight,
    which is the membership ``core.utils.get_vertices_in_group`` reports for a
    single group and the membership a pin's content hash is built from. The
    counts are read from ``obj.data``, so the object has to be out of Edit
    Mode for them to describe what the caller sees.
    """
    counts = dict.fromkeys(range(len(obj.vertex_groups)), 0)
    for vertex in obj.data.vertices:
        for element in vertex.groups:
            # A group index the mesh holds but the object does not own is a
            # broken datablock; let the KeyError surface it.
            counts[element.group] += 1
    return counts


def _create_mesh_vertex_group(
    obj, name: str, indices: list[int], weight: float
) -> dict:
    """Create a vertex group on *obj* holding *indices* at *weight*.

    Every check that does not need the mesh runs before the mode is touched,
    so a rejected call leaves Blender exactly as it was found.
    """
    vg_name = name.strip()
    if not vg_name:
        raise MCPError("Vertex group name cannot be empty")
    if obj.library is not None or obj.data.library is not None:
        raise MCPError(
            f"Object '{obj.name}' is library-linked, so its vertex groups and "
            "weights cannot be written. Make it local first."
        )
    if obj.vertex_groups.get(vg_name) is not None:
        raise MCPError(
            f"Object '{obj.name}' already has a vertex group named '{vg_name}'. "
            "An existing group can be driven by an armature or a modifier, so "
            "it is never overwritten; pick another name."
        )
    if not indices:
        raise MCPError(
            f"No vertex indices given for vertex group '{vg_name}'. An empty "
            "group pins nothing, so the indices to assign are required."
        )
    if not 0.0 <= weight <= 1.0:
        raise MCPError(f"weight must be in [0, 1]; got {weight}")
    try:
        wanted = sorted({int(index) for index in indices})
    except (TypeError, ValueError) as exc:
        raise MCPError(f"indices must be a list of integers: {exc}") from exc

    with _object_mode(obj):
        vertex_count = len(obj.data.vertices)
        out_of_range = [index for index in wanted if not 0 <= index < vertex_count]
        if out_of_range:
            shown = ", ".join(str(index) for index in out_of_range[:10])
            if len(out_of_range) > 10:
                shown += f", and {len(out_of_range) - 10} more"
            bounds = (
                "The mesh has no vertices."
                if vertex_count == 0
                else f"The mesh has {vertex_count} vertices, so valid indices "
                f"are 0 to {vertex_count - 1}."
            )
            raise MCPError(
                f"Vertex index out of range for '{obj.name}': {shown}. {bounds}"
            )
        vertex_group = obj.vertex_groups.new(name=vg_name)
        vertex_group.add(wanted, weight, "REPLACE")
        # Report the name Blender kept and the membership it recorded, which
        # is what a pin then names and hashes.
        created = {
            "object_name": obj.name,
            "vertex_group_name": vertex_group.name,
            "vertex_group_index": vertex_group.index,
            "vertex_count": _vertex_group_counts(obj)[vertex_group.index],
            "weight": float(weight),
        }
    return created


@group_handler
def list_vertex_groups(object_name: str):
    """List a mesh's vertex groups and how many vertices each one holds.

    A mesh pin names a vertex group that already exists on the object, so this
    reports the names add_pin_vertex_group accepts and create_vertex_group
    will refuse as duplicates. ``vertex_count`` counts the vertices assigned
    to the group at any weight; a group holding zero vertices pins nothing.

    Only a MESH carries vertex groups. A curve's pinned control points live on
    the curve object and are reported by list_pins once they are pinned.

    Refuses an object that is in Edit Mode. That session holds the geometry
    and the weights in a BMesh the mesh datablock does not receive until the
    mode is left, so the counts would describe the mesh as it stood before the
    session. Leave Edit Mode and call again.

    Args:
        object_name: Name of the mesh object to inspect.
    """
    obj = _resolve_mesh_object(object_name)
    if obj.mode == "EDIT":
        raise MCPError(
            f"Object '{object_name}' is in Edit Mode; its vertex counts are "
            "only readable in Object Mode. Leave Edit Mode and call again."
        )
    counts = _vertex_group_counts(obj)
    vertex_groups = [
        {
            "name": vertex_group.name,
            "index": vertex_group.index,
            "vertex_count": counts[vertex_group.index],
        }
        for vertex_group in obj.vertex_groups
    ]
    return {
        "object_name": obj.name,
        "vertex_groups": vertex_groups,
        "vertex_group_count": len(vertex_groups),
        "mesh_vertex_count": len(obj.data.vertices),
    }


@group_handler
def create_vertex_group(
    object_name: str,
    name: str,
    indices: list[int],
    weight: float = 1.0,
):
    """Create a vertex group on a mesh and assign the given vertices to it.

    This is the membership a mesh pin names: create the group here, then pass
    "object_name::name" to add_pin_vertex_group to pin it. Call
    list_vertex_groups first to see which names the object already carries.

    The object does not have to be the active one and Blender can be in any
    mode. An object in Edit Mode is taken to Object Mode for the write and put
    back, which also writes the edit session to the mesh, so the indices below
    address the geometry the caller can see.

    Fails before creating anything, leaving Blender as it was found, when the
    object is not a MESH, when it is library-linked, when indices is empty,
    when any index is outside the mesh, or when the object already carries a
    group of that name. An existing group can be driven by an armature or a
    modifier, so it is never overwritten.

    Args:
        object_name: Name of the mesh object to create the vertex group on.
        name: Name for the new vertex group; must not already exist on the object.
        indices: Vertex indices to assign, each in 0 to vertex_count - 1.
            Repeated indices are assigned once.
        weight: Weight for every assigned vertex, in [0, 1]. Defaults to 1.0,
            which is what the panel's Create button assigns.
    """
    obj = _resolve_mesh_object(object_name)
    created = _create_mesh_vertex_group(obj, name, indices, weight)
    assigned = created["vertex_count"]
    return {
        "message": (
            f"Created vertex group '{created['vertex_group_name']}' on "
            f"'{created['object_name']}' with {assigned} "
            f"{'vertex' if assigned == 1 else 'vertices'} assigned"
        ),
        **created,
    }


def _parse_pin_identifier(vertex_group_identifier: str) -> tuple[str, str]:
    from ...models.groups import parse_pin_identifier
    return parse_pin_identifier(vertex_group_identifier, ValidationError)


@group_handler
def add_pin_vertex_group(
    group_uuid: str,
    vertex_group_identifier: str,
    indices: list[int] | None = None,
):
    """Add a vertex group to the pin list of a dynamics group.

    For a MESH, the vertex group the identifier names has to exist on the
    object already: create it with create_vertex_group, or read what the
    object carries with list_vertex_groups. Passing 'indices' for a mesh is
    refused, because the vertex group holds the membership, not this call.

    For a CURVE, which has no vertex groups, 'indices' is how the pinned
    control points are defined: they are written onto the curve as
    "_pin_<vertex_group_name>" and pinned in the same call.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Identifier in format "object_name::vertex_group_name"
        indices: Curve control-point indices, for CURVE objects only
    """
    return _add_pin_vertex_group_impl(group_uuid, vertex_group_identifier, indices)


def _find_pin(group, group_uuid: str, vertex_group_identifier: str):
    """Locate one pin in a group's pin list.

    Returns (index, pin_item, object_name, object_uuid, vertex_group_name).
    The object is resolved by name to its UUID and the pin is matched on that
    UUID plus the vertex group name, so a renamed object still resolves.
    """
    obj_name, vg_name = _parse_pin_identifier(vertex_group_identifier)
    from ...core.uuid_registry import get_object_uuid

    obj = bpy.data.objects.get(obj_name)
    obj_uuid = get_object_uuid(obj) if obj else None
    if not obj_uuid:
        raise ValidationError(
            f"Object '{obj_name}' not found or has no UUID; "
            "cannot identify pin without a valid object UUID"
        )

    for index, item in enumerate(group.pin_vertex_groups):
        if item.object_uuid != obj_uuid:
            continue
        _, item_vg = decode_vertex_group_identifier(item.name)
        if item_vg == vg_name:
            return index, item, obj_name, obj_uuid, vg_name

    raise ValidationError(
        f"Pin '{obj_name}::{vg_name}' (object_uuid={obj_uuid}) "
        f"not found in group {group_uuid}"
    )


def _pin_order(group) -> list[str]:
    """Every pin identifier in the group, in list order."""
    order = []
    for item in group.pin_vertex_groups:
        item_obj, item_vg = decode_vertex_group_identifier(item.name)
        order.append(f"{item_obj}::{item_vg}")
    return order


@group_handler
def remove_pin_vertex_group(group_uuid: str, vertex_group_identifier: str):
    """Remove a vertex group from the pin list of a dynamics group.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Identifier in format "object_name::vertex_group_name"
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pin_index, _pin_item, obj_name, obj_uuid, vg_name = _find_pin(
        group, group_uuid, vertex_group_identifier
    )

    group.pin_vertex_groups.remove(pin_index)
    group.pin_vertex_groups_index = safe_update_index(
        pin_index, len(group.pin_vertex_groups)
    )

    # Refresh viewport overlays (consistent with the UI operator)
    from ...ui.dynamics import apply_object_overlays

    apply_object_overlays()

    return {
        "message": f"Removed pin '{obj_name}::{vg_name}' from group {group_uuid}",
        "group_uuid": group_uuid,
        "object_name": obj_name,
        "object_uuid": obj_uuid,
        "vertex_group_name": vg_name,
        "pin_count": len(group.pin_vertex_groups),
    }


@group_handler
def list_pins(group_uuid: str):
    """List all pins in a dynamics group.

    Args:
        group_uuid: UUID of group
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pins = []
    for pin_item in group.pin_vertex_groups:
        obj_name, vg_name = decode_vertex_group_identifier(pin_item.name)
        pins.append(
            {
                "object_name": obj_name,
                "object_uuid": pin_item.object_uuid,
                "vertex_group_name": vg_name,
                "vertex_group_identifier": f"{obj_name}::{vg_name}",
                "included": bool(pin_item.included),
                "use_pin_duration": bool(pin_item.use_pin_duration),
                "pin_duration": int(pin_item.pin_duration),
                "use_pull": bool(pin_item.use_pull),
                "pull_strength": float(pin_item.pull_strength),
                "operation_count": len(pin_item.operations),
            }
        )
    return {"group_uuid": group_uuid, "pins": pins, "pin_count": len(pins)}


def _pin_membership_name(obj, vg_name: str) -> str:
    """Where a pin's vertices live on *obj*, named for an error message."""
    if obj.type == "CURVE":
        return f"custom property '_pin_{vg_name}'"
    return f"vertex group '{vg_name}'"


def _check_pin_rename_target(obj, obj_name: str, vg_name: str, new_name: str):
    """Refuse a pin rename Blender would not perform as asked.

    A mesh pin names a vertex group and a curve pin names a "_pin_<name>"
    custom property, so both the group that is being renamed and the name it
    is renamed to are checked on the object itself. Blender makes a duplicate
    vertex group name unique by appending a suffix, and a duplicate custom
    property key overwrites what is already there, so a collision is refused
    here rather than resolved silently by Blender.
    """
    if obj.type == "MESH":
        if obj.vertex_groups.get(vg_name) is None:
            raise MCPError(
                f"Object '{obj_name}' has no vertex group named '{vg_name}', "
                f"so the pin '{obj_name}::{vg_name}' names nothing to rename. "
                "Call list_vertex_groups to see what the object carries."
            )
        if obj.vertex_groups.get(new_name) is not None:
            raise MCPError(
                f"Object '{obj_name}' already has a vertex group named "
                f"'{new_name}'. Blender would keep both by giving the renamed "
                "one a numbered suffix, which is not the name asked for, so "
                "pick a name the object does not already carry."
            )
        # A vertex group name lives in a fixed-width field that Blender
        # truncates to silently, so a name too long for it would be stored
        # under a name other than the one asked for. The field holds BYTES of
        # UTF-8, not characters, and its capacity counts the terminator, so
        # length_max reports one more byte than a name can occupy.
        capacity = bpy.types.VertexGroup.bl_rna.properties["name"].length_max - 1
        encoded = len(new_name.encode("utf-8"))
        if capacity > 0 and encoded > capacity:
            raise MCPError(
                f"'{new_name}' is {encoded} bytes of UTF-8 and a vertex group "
                f"name holds at most {capacity} bytes. Blender truncates a "
                "longer name instead of refusing it, which would leave the "
                "pin naming a group that does not exist. Pick a shorter name."
            )
        return
    if obj.type == "CURVE":
        if f"_pin_{vg_name}" not in obj:
            raise MCPError(
                f"Curve '{obj_name}' carries no '_pin_{vg_name}' property, so "
                f"the pin '{obj_name}::{vg_name}' holds no control-point "
                "indices to rename. Re-add the pin with add_pin_vertex_group "
                "and its indices."
            )
        if f"_pin_{new_name}" in obj:
            raise MCPError(
                f"Curve '{obj_name}' already carries a '_pin_{new_name}' "
                "property holding another pin's control points, and renaming "
                "onto it would overwrite them. Pick a name the curve does not "
                "already carry."
            )
        # No length check on this branch: Blender refuses a custom property
        # name longer than it can store, naming the limit, and the curve keeps
        # its indices. Only the mesh field above truncates instead.
        return
    raise MCPError(
        f"Object '{obj_name}' is type {obj.type}; a pin names a mesh vertex "
        "group or a curve control-point set, so there is nothing to rename on "
        "this object."
    )


@group_handler
def rename_pin_vertex_group(
    group_uuid: str,
    vertex_group_identifier: str,
    new_name: str,
):
    """Rename the vertex group a pin names, on the object and in the pin list.

    This renames the membership as well as the pin entry: on a mesh the
    object's vertex group is renamed, and on a curve the "_pin_<name>"
    property holding the pinned control points is. Anything else that names
    that vertex group, an armature or a modifier for instance, refers to it by
    name and stops finding it, so rename a group only the solver pin uses.

    The object keeps its name; only the vertex group half of the identifier
    changes. Refused before anything is renamed when the new name is empty,
    when it is the name the pin already has, when the object already carries
    a vertex group (on a curve, a pin property) under that name, or when the
    name is longer in UTF-8 bytes than a mesh vertex group name field holds,
    since Blender would in the last two cases store a name other than the one
    asked for.

    Args:
        group_uuid: UUID of the group holding the pin.
        vertex_group_identifier: The pin to rename, in the format
            "object_name::vertex_group_name".
        new_name: New name for the vertex group half of the identifier.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    index, _pin_item, obj_name, obj_uuid, vg_name = _find_pin(
        group, group_uuid, vertex_group_identifier
    )

    wanted = new_name.strip()
    if not wanted:
        raise MCPError(
            "new_name is empty; a pin has to name a vertex group, so pass the "
            "name the group is to take."
        )
    if wanted == vg_name:
        raise MCPError(
            f"Pin '{obj_name}::{vg_name}' already carries that name, and the "
            "rename operator declines a no-op rename."
        )

    from ...core.uuid_registry import get_object_by_uuid

    obj = get_object_by_uuid(obj_uuid)
    if obj is None:
        raise MCPError(
            f"Pin '{obj_name}::{vg_name}' names object_uuid={obj_uuid}, which "
            "resolves to no object in the scene; the object was deleted after "
            "the pin was made. Remove the pin with remove_pin_vertex_group."
        )
    _check_pin_rename_target(obj, obj_name, vg_name, wanted)

    slot = get_group_slot_index(bpy.context.scene, group_uuid)
    if slot is None:
        raise MCPError(f"Group with UUID {group_uuid} has no slot")

    # The operator renames whichever pin the group has selected, and asks for
    # the new name in a dialog when it is invoked, so select the pin first and
    # run it with "EXEC_DEFAULT", which runs execute and skips the dialog. The
    # selected row is the artist's own UI state and a rename is not a change of
    # selection, so put it back afterwards: a rename changes neither the length
    # of the list nor its order, so the saved row still names the pin it named
    # before.
    selected = group.pin_vertex_groups_index
    try:
        group.pin_vertex_groups_index = index
        result = bpy.ops.object.rename_pin_vertex_group(
            "EXEC_DEFAULT", group_index=slot, new_name=wanted
        )
    finally:
        group.pin_vertex_groups_index = selected
    if "FINISHED" not in result:
        raise MCPError(
            f"object.rename_pin_vertex_group returned {sorted(result)} for "
            f"'{obj_name}::{vg_name}', so the pin was not renamed."
        )

    # Report the name that was stored, and confirm the membership answers to
    # it: the pin entry and the object are written separately, so a name
    # Blender altered on the object would leave the pin naming nothing.
    _, kept = decode_vertex_group_identifier(group.pin_vertex_groups[index].name)
    if obj.type == "MESH":
        stored = obj.vertex_groups.get(kept) is not None
    else:
        stored = f"_pin_{kept}" in obj
    if not stored:
        raise MCPError(
            f"Pin '{obj_name}::{kept}' was renamed, but '{obj_name}' has no "
            f"{_pin_membership_name(obj, kept)}, so the pin now names no "
            "vertices. Rename it to a name the object can store as written, "
            "before running the simulation."
        )

    return {
        "message": (
            f"Renamed pin '{obj_name}::{vg_name}' to '{obj_name}::{kept}'"
        ),
        "group_uuid": group_uuid,
        "object_name": obj_name,
        "object_uuid": obj_uuid,
        "previous_vertex_group_name": vg_name,
        "vertex_group_name": kept,
        "vertex_group_identifier": f"{obj_name}::{kept}",
    }


@group_handler
def move_pin_vertex_group(
    group_uuid: str,
    vertex_group_identifier: str,
    direction: str,
):
    """Move a pin one place up or down its group's pin list.

    Pin order decides what two pins of one group do where they hold the same
    vertex: the scene build writes each pin's settings in list order, so for a
    shared vertex the pin lower in the list is the one whose duration, pull
    and operations that vertex takes. Order says nothing about pins that share
    no vertex.

    One place per call. list_pins reports the pins in list order, so the
    position of a pin in that array is the position this moves it from. A pin
    already at the top cannot move up and one already at the bottom cannot
    move down; either is refused rather than reported as a move that did
    nothing.

    Args:
        group_uuid: UUID of the group holding the pin.
        vertex_group_identifier: The pin to move, in the format
            "object_name::vertex_group_name".
        direction: "UP" to move it one place toward the start of the list,
            "DOWN" to move it one place toward the end.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    index, _pin_item, obj_name, obj_uuid, vg_name = _find_pin(
        group, group_uuid, vertex_group_identifier
    )

    steps = {"UP": -1, "DOWN": 1}.get(direction.strip().upper())
    if steps is None:
        raise MCPError(
            f"direction must be 'UP' or 'DOWN', got {direction!r}."
        )

    count = len(group.pin_vertex_groups)
    target = index + steps
    if not 0 <= target < count:
        edge = "first" if steps < 0 else "last"
        raise MCPError(
            f"Pin '{obj_name}::{vg_name}' is already the {edge} of the "
            f"{count} pins in group {group_uuid}, so it cannot move "
            f"{direction.strip().upper()}."
        )

    slot = get_group_slot_index(bpy.context.scene, group_uuid)
    if slot is None:
        raise MCPError(f"Group with UUID {group_uuid} has no slot")

    # The operator moves whichever pin the group has selected, by one place in
    # the direction it is given (-1 up, 1 down). "EXEC_DEFAULT" runs execute
    # directly, which is the path a tool call takes.
    group.pin_vertex_groups_index = index
    result = bpy.ops.object.move_pin_vertex_group(
        "EXEC_DEFAULT", group_index=slot, direction=steps
    )
    if "FINISHED" not in result:
        raise MCPError(
            f"object.move_pin_vertex_group returned {sorted(result)} for "
            f"'{obj_name}::{vg_name}', so the pin was not moved."
        )

    moved, _item, _name, _uuid, _vg = _find_pin(
        group, group_uuid, vertex_group_identifier
    )
    return {
        "message": (
            f"Moved pin '{obj_name}::{vg_name}' from position {index} to "
            f"{moved} in group {group_uuid}"
        ),
        "group_uuid": group_uuid,
        "object_name": obj_name,
        "object_uuid": obj_uuid,
        "vertex_group_name": vg_name,
        "previous_index": index,
        "index": moved,
        "pin_order": _pin_order(group),
        "pin_count": count,
    }


@group_handler
def set_group_overlay_color(
    group_uuid: str,
    r: float,
    g: float,
    b: float,
    a: float = 1.0,
):
    """Set the viewport overlay color for a dynamics group.

    Args:
        group_uuid: UUID of group
        r: Red channel in [0, 1]
        g: Green channel in [0, 1]
        b: Blue channel in [0, 1]
        a: Alpha channel in [0, 1]
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    group.color = (float(r), float(g), float(b), float(a))
    group.show_overlay_color = True

    from ...ui.dynamics.overlay import apply_object_overlays

    apply_object_overlays()
    return {
        "message": f"Updated overlay color for group {group_uuid}",
        "group_uuid": group_uuid,
        "color": [float(r), float(g), float(b), float(a)],
        "show_overlay_color": True,
    }


# ---------------------------------------------------------------------------
# Group material parameters
# ---------------------------------------------------------------------------

# Which material parameters each group type carries. This is the one source
# for both directions: set_group_material_properties validates a write
# against the set for the group's own type, and get_group_material_properties
# reports that same set, so a parameter added for a type is offered and
# accepted in the same edit.
_MATERIAL_PROPERTIES_BY_TYPE: dict[str, set[str]] = {
    "SHELL": {
        "enable_strain_limit",
        "strain_limit_percent",
        "shell_density",
        "shell_young_modulus",
        "shell_poisson_ratio",
        "shell_model",
        "bend",
        "bend_warp",
        "bend_weft",
        "shrink_x",
        "shrink_y",
        "deformation_damping",
        "bending_damping",
        "young_mod_density_normalized",
        "friction",
        "enable_inflate",
        "inflate_pressure",
        "stitch_stiffness",
        "enable_plasticity",
        "plasticity",
        "plasticity_threshold",
        "enable_bend_plasticity",
        "bend_plasticity",
        "bend_plasticity_threshold",
        "bend_rest_angle_source",
        "bend_rest_from_reference",
        "allow_self_intersection",
        "allow_self_intersection_all_objects",
        "allow_inter_object_intersection",
        "allow_inter_object_intersection_all_objects",
        "contact_gap",
        "contact_offset",
        "contact_gap_rat",
        "contact_offset_rat",
        "use_group_bounding_box_diagonal",
    },
    "SOLID": {
        "solid_density",
        "solid_young_modulus",
        "solid_poisson_ratio",
        "solid_model",
        "shrink",
        "deformation_damping",
        "young_mod_density_normalized",
        "friction",
        "stitch_stiffness",
        "enable_plasticity",
        "plasticity",
        "plasticity_threshold",
        "allow_self_intersection",
        "allow_self_intersection_all_objects",
        "allow_inter_object_intersection",
        "allow_inter_object_intersection_all_objects",
        "contact_gap",
        "contact_offset",
        "contact_gap_rat",
        "contact_offset_rat",
        "use_group_bounding_box_diagonal",
    },
    "ROD": {
        "rod_density",
        "rod_young_modulus",
        "rod_model",
        "deformation_damping",
        "bending_damping",
        "young_mod_density_normalized",
        "friction",
        "bend",
        "length_factor",
        "enable_strain_limit",
        "strain_limit_percent",
        "stitch_stiffness",
        "enable_bend_plasticity",
        "bend_plasticity",
        "bend_plasticity_threshold",
        "bend_rest_angle_source",
        "bend_rest_from_reference",
        "allow_self_intersection",
        "allow_self_intersection_all_objects",
        "allow_inter_object_intersection",
        "allow_inter_object_intersection_all_objects",
        "contact_gap",
        "contact_offset",
        "contact_gap_rat",
        "contact_offset_rat",
        "use_group_bounding_box_diagonal",
    },
    "STATIC": {
        "friction",
        "enable_soft_constraint",
        "soft_constraint_stiffness",
        "allow_self_intersection",
        "allow_self_intersection_all_objects",
        "allow_inter_object_intersection",
        "allow_inter_object_intersection_all_objects",
        "contact_gap",
        "contact_offset",
        "contact_gap_rat",
        "contact_offset_rat",
        "use_group_bounding_box_diagonal",
    },
    "PDRD": {
        "pdrd_density",
        "friction",
        "stitch_stiffness",
        "allow_self_intersection",
        "allow_self_intersection_all_objects",
        "allow_inter_object_intersection",
        "allow_inter_object_intersection_all_objects",
        "contact_gap",
        "contact_offset",
        "contact_gap_rat",
        "contact_offset_rat",
        "use_group_bounding_box_diagonal",
    },
    "SAND": {
        "sand_grain_radius",
        "sand_particle_mass",
        "sand_friction",
        "allow_self_intersection",
        "allow_self_intersection_all_objects",
        "allow_inter_object_intersection",
        "allow_inter_object_intersection_all_objects",
        "contact_gap",
        "contact_offset",
        "contact_gap_rat",
        "contact_offset_rat",
        "use_group_bounding_box_diagonal",
    },
}


def _material_property(group, name: str):
    """The RNA property a material parameter is stored in."""
    prop = group.bl_rna.properties.get(name)
    if prop is None:
        raise MCPError(
            f"The parameter table lists '{name}' for a {group.object_type} "
            "group, but a dynamics group carries no property of that name. "
            "The table and the group's own properties disagree, which is a "
            "defect in the add-on rather than something the call can correct."
        )
    return prop


def _offered_enum_identifiers(prop) -> list[str]:
    """The identifiers an enum property offers as a choice.

    An item whose UI name is empty is not one of them. ui/object_group.py
    registers a withdrawn identifier that way, with an explicit item number
    so a `.blend` holding it still loads as the same identifier, and an empty
    name is what suppresses the item from the panel's picker.
    core/encoder/params.py substitutes a supported identifier for it when the
    scene is encoded, so it is not a model a caller can ask the solver to run.
    """
    return [item.identifier for item in prop.enum_items if item.name]


def _material_property_description(prop) -> str:
    """What a parameter means, for a caller that reads only this report.

    A property registered without a description of its own carries an empty
    one, which explains nothing to a caller who has no panel in front of
    them. An enum's items each carry a description, so an enum falls back to
    those, and any property falls back to the name its panel row is labeled
    with.
    """
    if prop.description:
        return prop.description
    if prop.type == "ENUM":
        described = [
            f"{item.identifier}: {item.description}"
            for item in prop.enum_items
            if item.name and item.description
        ]
        if described:
            return f"{prop.name}. {'; '.join(described)}"
    return prop.name


def _material_property_report(group, name: str) -> dict:
    """One material parameter as an MCP caller sees it.

    The current value, the add-on default, the description and the limits
    all come from the property's own registration on the group, which is
    also what set_group_material_properties holds a write to, so a caller
    reads the same limits a later write is subject to.
    """
    prop = _material_property(group, name)

    value = getattr(group, name)
    report: dict = {
        "type": prop.type.lower(),
        "description": _material_property_description(prop),
    }
    if prop.type == "BOOLEAN":
        report["value"] = bool(value)
        report["default"] = bool(prop.default)
    elif prop.type == "ENUM":
        options = _offered_enum_identifiers(prop)
        report["value"] = str(value)
        report["default"] = str(prop.default)
        report["options"] = options
        # A group can hold an identifier the picker does not offer, loaded
        # from a `.blend` that carries it. Report it as the value it is, and
        # say that it is not among the options, so it is never advertised as
        # a choice a caller could ask for.
        report["value_withdrawn"] = report["value"] not in options
    elif prop.type in ("FLOAT", "INT"):
        cast = float if prop.type == "FLOAT" else int
        report["value"] = cast(value)
        report["default"] = cast(prop.default)
        # The bounds the property itself enforces. A property declared with
        # no bound reports the extreme its number type can hold.
        report["min"] = cast(prop.hard_min)
        report["max"] = cast(prop.hard_max)
    else:
        raise MCPError(
            f"Material parameter '{name}' is a {prop.type} property, which "
            "this tool has no reporting rule for. A material parameter is a "
            "number, a boolean or an enum."
        )
    return report


def _property_image(prop, value: float) -> float:
    """*value* in the property's own number type, before its range applies.

    A float property is a C float, and so are the bounds it reports, so a
    Python double is rounded to single precision on the way in and it is
    that image the range applies to. Comparing the double instead would
    refuse a value written exactly at a bound: a minimum registered as 0.001
    is stored as the nearest float, 0.0010000000474974513, which the double
    0.001 sits below. A magnitude too large for a float has no image, and
    its double value lies outside the range of any float property, so the
    comparison refuses it on that.
    """
    if prop.type == "INT":
        return float(value)
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return float(value)


def _check_material_value(group, name: str, value) -> None:
    """Refuse a value the property itself would not store as written.

    Blender clamps a number written past a property's own minimum or
    maximum and keeps the clamp, and it keeps an enum identifier its picker
    does not offer, so either write would leave the group holding something
    other than what was asked for while the call reported success. Both are
    refused here, naming the property, the value and what the property
    takes. A value of the wrong Python type is left to the assignment
    itself, which names the type it cannot take.
    """
    prop = _material_property(group, name)
    if prop.type in ("FLOAT", "INT") and isinstance(value, (int, float)):
        low = float(prop.hard_min)
        high = float(prop.hard_max)
        if not low <= _property_image(prop, value) <= high:
            raise MCPError(
                f"Property '{name}' on a {group.object_type} group holds "
                f"{low} to {high}, and {value} is outside it. Blender stores "
                "the nearest limit for a write past one, so the group would "
                "not carry the value asked for. Pass a value within the "
                "range, which get_group_material_properties reports as min "
                "and max."
            )
    elif prop.type == "ENUM" and isinstance(value, str):
        options = _offered_enum_identifiers(prop)
        if value not in options:
            registered = any(
                item.identifier == value for item in prop.enum_items
            )
            because = (
                "The property still stores that identifier, so a `.blend` "
                "holding it keeps loading, but the scene build does not "
                "encode it as asked."
                if registered
                else "It is not an identifier the property carries."
            )
            raise MCPError(
                f"Property '{name}' on a {group.object_type} group accepts "
                f"{', '.join(options)}, and '{value}' is not one of them. "
                f"{because} get_group_material_properties reports the "
                "accepted identifiers as options."
            )


@group_handler
def get_group_material_properties(group_uuid: str):
    """Report every material parameter a group accepts, with its value.

    Which parameters a group carries is decided by its object_type, and the
    set reported here is exactly the set set_group_material_properties
    accepts for this group, so a name absent from this report is refused by
    that tool. A parameter another type carries is reachable only by
    retyping the group with set_group_type first.

    Each entry carries the current value, the add-on default, the
    description the panel shows for it (or, where the property carries none,
    the descriptions its own options carry), and the limits the property
    enforces: min and max for a number, the accepted identifiers for an enum.
    set_group_material_properties refuses a value outside those limits
    instead of storing a clamp of it, so a value within them is one that tool
    stores as written.

    An enum reports only the identifiers its picker offers. A group loaded
    from a `.blend` holding a withdrawn identifier reports it as the value,
    with "value_withdrawn": true and the identifier absent from "options";
    the scene build substitutes a supported identifier for it, so set the
    parameter to one of the offered identifiers to decide what the solver
    runs.

    The values are the authored ones, not what the solver derives from them.
    Contact distances in particular are stored as an absolute pair
    (contact_gap, contact_offset) and a relative pair (contact_gap_rat,
    contact_offset_rat), and both pairs are reported whichever one
    use_group_bounding_box_diagonal currently selects.

    Per-object state (inclusion, locks, hinge, bending reference,
    tetrahedralizer) is reported by get_group_objects, and a parameter driven
    across the surface by a weight map is reported by list_material_maps.

    Args:
        group_uuid: UUID of the group to report.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    names = _MATERIAL_PROPERTIES_BY_TYPE.get(group.object_type)
    if names is None:
        raise MCPError(
            f"Group '{group.name}' has object_type '{group.object_type}', "
            "which carries no material parameters. The types that do are "
            f"{', '.join(sorted(_MATERIAL_PROPERTIES_BY_TYPE))}."
        )

    properties = {
        name: _material_property_report(group, name) for name in sorted(names)
    }
    return {
        "group_uuid": group_uuid,
        "group_name": group.name,
        "group_type": group.object_type,
        "properties": properties,
        "property_count": len(properties),
    }


@group_handler
def set_group_material_properties(group_uuid: str, properties: dict):
    """Set material properties for a dynamics group.

    A value the property itself cannot hold is refused, and nothing is
    written when one is: a number past the property's own minimum or
    maximum, or an enum identifier its picker does not offer. Blender stores
    the nearest limit for the first and keeps the second for the scene build
    to substitute for, so either would leave the group holding something
    other than what was asked for while this call reported success.
    get_group_material_properties reports the range and the accepted
    identifiers of every parameter this tool takes.

    Args:
        group_uuid: UUID of target group
        properties: Dict of property_name -> value mappings

    Supported properties by group type:

    - SHELL: enable_strain_limit, strain_limit_percent, shell_density, shell_young_modulus, shell_poisson_ratio, shell_model, bend, bend_warp, bend_weft, shrink_x, shrink_y, deformation_damping, bending_damping, young_mod_density_normalized, friction, enable_inflate, inflate_pressure, stitch_stiffness
    - SOLID: solid_density, solid_young_modulus, solid_poisson_ratio, solid_model, shrink, deformation_damping, young_mod_density_normalized, friction, stitch_stiffness
    - ROD: rod_density, rod_young_modulus, rod_model, deformation_damping, bending_damping, young_mod_density_normalized, friction, bend, length_factor, enable_strain_limit, strain_limit_percent, stitch_stiffness
    - PDRD: pdrd_density, friction, stitch_stiffness (the hinge joint is per-object; use the set_pdrd_hinge tool)
    - SAND: sand_grain_radius, sand_particle_mass, sand_friction (faceless granular body of loose grain-center vertices)
    - STATIC: friction, enable_soft_constraint, soft_constraint_stiffness (a collider tracks its animation exactly unless soft constraints are on, which holds it with springs of that stiffness so contact can push it off its path)

    Rayleigh damping (deformation_damping on Solid/Shell/Rod, bending_damping on
    Shell/Rod only) and young_mod_density_normalized (interpret Young's modulus
    as true pascals when False) are per-group. Solid has no bending term, so
    bending_damping is rejected for Solid. PDRD groups carry only density,
    friction, contact, and stitch settings.

    length_factor (Rod only) multiplies every rod edge's rest length, so below
    1.0 it tensions a pinned rod and above 1.0 it slackens it; mass is taken
    from the drawn length and does not move with it. Rod bending stiffness is
    normalized against that same rest length and varies as its inverse square,
    so halving length_factor also makes the rod about four times stiffer in
    bending.

    Intersection allowances (accepted on every group type: SOLID, SHELL, ROD,
    PDRD, SAND, STATIC). Each reaches every object assigned to the group while
    its allow_self_intersection_all_objects /
    allow_inter_object_intersection_all_objects switch is on, and both switches
    default on; set_intersection_allowance_objects narrows one to named objects
    and turns its switch off. Self versus inter-object is decided per Blender
    object, not per group, whichever way the allowance is narrowed:

    - allow_self_intersection: an overlap of one object with itself is
      simulated instead of reported, so a run starts and keeps going through a
      pose that object is tangled in. An overlap between two objects assigned
      to the same group is an inter-object pair, which this key does not cover.
    - allow_inter_object_intersection: the same for an overlap between two
      different objects, including two objects of this group. Either side is
      enough, so setting it on a garment also covers the body it is fitted to.

    On a STATIC group both keys reach the solver whenever the collider is part
    of the solved scene, which covers an animated collider, a soft-constrained
    one, and one named as a cross-stitch endpoint: each of those decodes to a
    pin shell whose vertices carry the policy. A collider that is none of them
    stays a contact-only collision mesh, its vertices carry no object id and an
    empty policy, and a pair involving it is tolerated only when the opposing
    dynamic side opts in. Contact and CCD are unaffected; only the report is
    suppressed.

    Contact properties (mutually exclusive modes):

    - Absolute mode: contact_gap, contact_offset (sets use_group_bounding_box_diagonal=False)
    - Relative mode: contact_gap_rat, contact_offset_rat (sets use_group_bounding_box_diagonal=True)

    Returns:
        Dict with success message and properties set
    """

    def validate_contact_properties(props: dict):
        """Validate contact property combinations and resolve conflicts."""
        absolute_props = {"contact_gap", "contact_offset"}
        relative_props = {"contact_gap_rat", "contact_offset_rat"}

        has_absolute = any(prop in props for prop in absolute_props)
        has_relative = any(prop in props for prop in relative_props)

        if has_absolute and has_relative:
            raise ValidationError(
                "Cannot set both absolute (contact_gap/contact_offset) and "
                + "relative (contact_gap_rat/contact_offset_rat) properties simultaneously. "
                + "Use separate calls to switch contact modes."
            )

        # Auto-set the mode flag based on which properties are provided
        if has_absolute:
            props["use_group_bounding_box_diagonal"] = False
        elif has_relative:
            props["use_group_bounding_box_diagonal"] = True

    def validate_properties_for_group_type(props: dict, object_type: str):
        """Validate that properties are appropriate for the group type."""
        valid_props = _MATERIAL_PROPERTIES_BY_TYPE.get(object_type, set())
        invalid_props = set(props.keys()) - valid_props

        if invalid_props:
            raise ValidationError(
                f"Invalid properties for {object_type} group: {invalid_props}. "
                + f"Valid properties: {sorted(valid_props)}"
            )

    # Get and validate group
    group = get_active_group_by_uuid_helper(group_uuid)

    if not properties:
        raise ValidationError("Properties dictionary cannot be empty")

    # Create working copy for atomic updates
    props_to_set = properties.copy()

    # Validate contact property combinations
    validate_contact_properties(props_to_set)

    # Validate properties against group type
    validate_properties_for_group_type(props_to_set, group.object_type)

    # Validate that all properties exist on the ObjectGroup class
    invalid_attrs = []
    for prop_name in props_to_set:
        if not hasattr(group, prop_name):
            invalid_attrs.append(prop_name)

    if invalid_attrs:
        raise ValidationError(f"Invalid property names: {invalid_attrs}")

    # Refuse every value the properties would not store as written before
    # writing any of them, so a refused call leaves the group as it was.
    for prop_name, value in props_to_set.items():
        _check_material_value(group, prop_name, value)

    # Apply properties atomically
    updated_props = []
    for prop_name, value in props_to_set.items():
        try:
            # Get current value for logging
            old_value = getattr(group, prop_name, None)

            # Set new value (PropertyGroup validation will occur automatically)
            setattr(group, prop_name, value)

            # Report what the group now holds, read back from the property,
            # so the report is the stored value and not the requested one.
            updated_props.append(
                f"{prop_name}: {old_value} -> {getattr(group, prop_name)}"
            )

        except Exception as e:
            raise ValidationError(
                f"Failed to set property '{prop_name}' to '{value}': {str(e)}"
            ) from e

    return {
        "message": f"Updated {len(updated_props)} properties for group {group_uuid}",
        "group_uuid": group_uuid,
        "group_type": group.object_type,
        "properties_set": list(props_to_set.keys()),
        "updates": updated_props,
    }
