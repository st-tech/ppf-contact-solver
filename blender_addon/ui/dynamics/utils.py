# File: utils.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ...models.groups import get_addon_data
from ..state import ObjectGroup, decode_vertex_group_identifier


def get_group_from_index(scene, group_index: int) -> ObjectGroup | None:
    """Return the group at *group_index* if it exists and is active, else None."""
    prop_name = f"object_group_{group_index}"
    group: ObjectGroup | None = getattr(get_addon_data(scene), prop_name, None)
    if group is None or not group.active:
        return None
    return group


def get_assigned_by_selection_uuid(group: ObjectGroup, selection_attr: str):
    """Return the AssignedObject whose uuid matches ``group.<selection_attr>``.

    Used by the Velocity/Collision-Window sub-panels which store the active
    object as a UUID string on the group rather than as a positional index.
    Returns None if nothing is selected or the uuid is stale.
    """
    sel_uuid = getattr(group, selection_attr, "") or ""
    if not sel_uuid or sel_uuid == "NONE":
        return None
    for assigned in group.assigned_objects:
        if assigned.uuid == sel_uuid:
            return assigned
    return None


def reset_object_display(obj):
    """Reset an object's display color and wireframe overlays to defaults."""
    obj.color = (1.0, 1.0, 1.0, 1.0)
    obj.show_wire = False
    obj.show_all_edges = False


def cleanup_pin_vertex_groups_for_object(group: ObjectGroup, object_uuid: str):
    """Remove pin vertex groups that reference the specified object (by UUID)."""
    pin_indices_to_remove = []
    for pin_index in range(len(group.pin_vertex_groups)):
        pin_item = group.pin_vertex_groups[pin_index]
        if pin_item.object_uuid == object_uuid:
            pin_indices_to_remove.append(pin_index)

    for pin_index in reversed(pin_indices_to_remove):
        group.pin_vertex_groups.remove(pin_index)

    if group.pin_vertex_groups_index >= len(group.pin_vertex_groups):
        group.pin_vertex_groups_index = max(0, len(group.pin_vertex_groups) - 1)
