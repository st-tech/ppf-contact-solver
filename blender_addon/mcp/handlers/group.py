"""Group management handlers (create, delete, objects, pins, materials)."""

import bpy  # pyright: ignore

from ...models.collection_utils import safe_update_index
from ...models.groups import (
    N_MAX_GROUPS,
    assign_display_indices,
    decode_vertex_group_identifier,
    get_active_group_by_uuid,
    get_addon_data,
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
    """Helper function to get group slot index by UUID for UI operators that require index."""
    scene = bpy.context.scene
    for i in range(N_MAX_GROUPS):
        group = getattr(get_addon_data(scene), f"object_group_{i}", None)
        if group and group.active and group.uuid == group_uuid:
            return i

    raise ValidationError(f"Active group with UUID {group_uuid} not found")


@group_handler
def create_group():
    """Create a new dynamics group."""
    # Use UI operator instead of direct manipulation
    bpy.ops.object.create_group()

    # Get the newly created group info
    scene = bpy.context.scene

    # Assign display indices to ensure consistency
    assign_display_indices(scene)

    # Find the most recently created group (should be the last one created)
    newest_group = None
    for group in iterate_object_groups(scene):
        if group.active:
            newest_group = group

    if newest_group:
        group_uuid = newest_group.uuid  # already set by the operator
        get_addon_data(scene).state.current_group_uuid = group_uuid
        return {
            "message": f"Created group: {newest_group.name}",
            "group_name": newest_group.name,
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

    # Use UI operator for group deletion
    bpy.ops.object.delete_group(group_uuid=group_uuid)

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
    src_index = get_group_index_by_uuid(group_uuid)

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
    group = get_active_group_by_uuid_helper(group_uuid)
    from ...core.uuid_registry import get_object_uuid
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise MCPError(f"Object '{object_name}' not found in scene")
    obj_uuid = get_object_uuid(obj)
    if not obj_uuid:
        raise MCPError(f"Object '{object_name}' has no UUID")

    idx = -1
    for i, a in enumerate(group.assigned_objects):
        if a.uuid == obj_uuid:
            idx = i
            break
    if idx < 0:
        raise MCPError(f"Object '{object_name}' not in group {group_uuid}")
    group.assigned_objects_index = idx

    group_index = get_group_index_by_uuid(group_uuid)
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
    group = get_active_group_by_uuid_helper(group_uuid)
    from ...core.uuid_registry import get_object_uuid
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise MCPError(f"Object '{object_name}' not found in scene")
    obj_uuid = get_object_uuid(obj)
    if not obj_uuid:
        raise MCPError(f"Object '{object_name}' has no UUID")

    idx = -1
    for i, a in enumerate(group.assigned_objects):
        if a.uuid == obj_uuid:
            idx = i
            break
    if idx < 0:
        raise MCPError(f"Object '{object_name}' not in group {group_uuid}")
    group.assigned_objects_index = idx

    group_index = get_group_index_by_uuid(group_uuid)
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
def get_active_groups():
    """Get list of all active groups with their properties."""
    scene = bpy.context.scene
    groups = []

    # Ensure display indices are assigned
    assign_display_indices(scene)

    for group in iterate_active_object_groups(scene):
        groups.append(
            {
                "uuid": group.uuid or "",
                "name": group.name,
                "object_type": group.object_type,
                "active": group.active,
                "assigned_objects": [
                    {"name": obj.name, "uuid": obj.uuid}
                    for obj in group.assigned_objects
                ],
                "object_count": len(group.assigned_objects),
            }
        )

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

        if obj.type != "MESH":
            warnings.append(f"Object '{obj_name}' is not a mesh")
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
        # Resolve UUID to slot index for the operator
        group_index = get_group_index_by_uuid(group_uuid)

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
    from ...ui.dynamics.utils import cleanup_pin_vertex_groups_for_object

    cleanup_pin_vertex_groups_for_object(group, obj_uuid)

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

    # Direct manipulation - clear all objects and related data
    group.pin_vertex_groups.clear()
    group.pin_vertex_groups_index = -1
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


@group_handler
def get_group_objects(group_uuid: str):
    """Get objects assigned to a dynamics group.

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
        objects.append(
            {
                "name": obj.name,
                "uuid": assigned_obj.uuid,
                "type": obj.type,
                "vertex_count": len(obj.data.vertices) if obj.type == "MESH" else 0,
                "face_count": len(obj.data.polygons) if obj.type == "MESH" else 0,
            }
        )

    return {
        "group_uuid": group_uuid,
        "group_type": group.object_type,
        "objects": objects,
        "object_count": len(objects),
    }


@group_handler
def set_group_type(group_uuid: str, type: str):
    """Set the type of a dynamics group.

    Args:
        group_uuid: UUID of group
        type: Group type (SOLID, SHELL, ROD, STATIC)
    """
    valid_types = ["SOLID", "SHELL", "ROD", "STATIC"]
    if type not in valid_types:
        raise ValidationError(f"Invalid type '{type}'. Valid types: {valid_types}")

    group = get_active_group_by_uuid_helper(group_uuid)
    group.object_type = type

    return {
        "message": f"Set group {group_uuid} type to {type}",
        "group_uuid": group_uuid,
        "type": type,
    }


def _parse_pin_identifier(vertex_group_identifier: str) -> tuple[str, str]:
    from ...models.groups import parse_pin_identifier
    return parse_pin_identifier(vertex_group_identifier, ValidationError)


@group_handler
def add_pin_vertex_group(group_uuid: str, vertex_group_identifier: str):
    """Add a vertex group to the pin list of a dynamics group.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Identifier in format "object_name::vertex_group_name"
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    obj_name, vg_name = _parse_pin_identifier(vertex_group_identifier)

    # Validate that the object exists and is a mesh
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        raise ValidationError(f"Object '{obj_name}' not found")
    if obj.type != "MESH":
        raise ValidationError(f"Object '{obj_name}' is not a mesh")

    # Validate that the vertex group exists
    if vg_name not in [vg.name for vg in obj.vertex_groups]:
        raise ValidationError(
            f"Vertex group '{vg_name}' not found in object '{obj_name}'"
        )

    # Internal format: [ObjectName][VertexGroupName]
    from ...models.groups import encode_vertex_group_identifier, decode_vertex_group_identifier
    from ...core.uuid_registry import get_or_create_object_uuid, compute_vg_hash
    internal_id = encode_vertex_group_identifier(obj_name, vg_name)
    obj_uuid = get_or_create_object_uuid(obj)
    if not obj_uuid:
        raise MCPError(f"Object '{obj_name}' is library-linked and not writable")

    # Check if this vertex group is already in the pin list (UUID + vg_name match)
    for item in group.pin_vertex_groups:
        if item.object_uuid != obj_uuid:
            continue
        _, item_vg = decode_vertex_group_identifier(item.name)
        if item_vg == vg_name:
            return {
                "message": f"Vertex group already pinned",
                "group_uuid": group_uuid,
                "pin_count": len(group.pin_vertex_groups),
            }

    # Add the vertex group to the pin list
    item = group.pin_vertex_groups.add()
    item.name = internal_id
    item.object_uuid = obj_uuid
    item.vg_hash = str(compute_vg_hash(obj, vg_name))
    group.pin_vertex_groups_index = len(group.pin_vertex_groups) - 1

    return {
        "message": f"Added pin '{obj_name}::{vg_name}' to group {group_uuid}",
        "group_uuid": group_uuid,
        "object_name": obj_name,
        "object_uuid": obj_uuid,
        "vertex_group_name": vg_name,
        "pin_count": len(group.pin_vertex_groups),
    }


@group_handler
def remove_pin_vertex_group(group_uuid: str, vertex_group_identifier: str):
    """Remove a vertex group from the pin list of a dynamics group.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Identifier in format "object_name::vertex_group_name"
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    obj_name, vg_name = _parse_pin_identifier(vertex_group_identifier)
    from ...models.groups import decode_vertex_group_identifier
    from ...core.uuid_registry import get_object_uuid

    # Resolve the target object via name -> UUID up front; match pins by UUID
    _obj = bpy.data.objects.get(obj_name)
    _obj_uuid = get_object_uuid(_obj) if _obj else None

    if not _obj_uuid:
        raise ValidationError(
            f"Object '{obj_name}' not found or has no UUID; "
            "cannot identify pin without a valid object UUID"
        )

    # Find the vertex group in the pin list via UUID + vg_name (strict)
    pin_index = -1
    for i, item in enumerate(group.pin_vertex_groups):
        if item.object_uuid != _obj_uuid:
            continue
        _, item_vg = decode_vertex_group_identifier(item.name)
        if item_vg == vg_name:
            pin_index = i
            break

    if pin_index == -1:
        raise ValidationError(
            f"Pin '{obj_name}::{vg_name}' (object_uuid={_obj_uuid}) "
            f"not found in group {group_uuid}"
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
        "object_uuid": _obj_uuid,
        "vertex_group_name": vg_name,
        "pin_count": len(group.pin_vertex_groups),
    }


@group_handler
def set_group_material_properties(group_uuid: str, properties: dict):
    """Set material properties for a dynamics group.

    Args:
        group_uuid: UUID of target group
        properties: Dict of property_name -> value mappings

    Supported properties by group type:

    - SHELL: enable_strain_limit, strain_limit, shell_density, shell_young_modulus, shell_poisson_ratio, shell_model, bend, shrink_x, shrink_y, friction, enable_inflate, inflate_pressure, stitch_stiffness
    - SOLID: solid_density, solid_young_modulus, solid_poisson_ratio, solid_model, shrink, friction, stitch_stiffness
    - ROD: rod_density, rod_young_modulus, rod_model, friction, bend, enable_strain_limit, strain_limit, stitch_stiffness
    - STATIC: friction (limited set)

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
        type_specific_props = {
            "SHELL": {
                "enable_strain_limit",
                "strain_limit",
                "shell_density",
                "shell_young_modulus",
                "shell_poisson_ratio",
                "shell_model",
                "bend",
                "shrink_x",
                "shrink_y",
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
                "friction",
                "stitch_stiffness",
                "enable_plasticity",
                "plasticity",
                "plasticity_threshold",
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
                "friction",
                "bend",
                "enable_strain_limit",
                "strain_limit",
                "stitch_stiffness",
                "enable_bend_plasticity",
                "bend_plasticity",
                "bend_plasticity_threshold",
                "bend_rest_angle_source",
                "contact_gap",
                "contact_offset",
                "contact_gap_rat",
                "contact_offset_rat",
                "use_group_bounding_box_diagonal",
            },
            "STATIC": {
                "friction",
                "contact_gap",
                "contact_offset",
                "contact_gap_rat",
                "contact_offset_rat",
                "use_group_bounding_box_diagonal",
            },
        }

        valid_props = type_specific_props.get(object_type, set())
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

    # Apply properties atomically
    updated_props = []
    for prop_name, value in props_to_set.items():
        try:
            # Get current value for logging
            old_value = getattr(group, prop_name, None)

            # Set new value (PropertyGroup validation will occur automatically)
            setattr(group, prop_name, value)

            updated_props.append(f"{prop_name}: {old_value} -> {value}")

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
