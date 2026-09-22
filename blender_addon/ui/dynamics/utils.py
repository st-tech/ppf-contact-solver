# File: utils.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ...models.groups import get_addon_data
from ..state import ObjectGroup


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


def cleanup_group_references_for_object(group: ObjectGroup, object_uuid: str):
    """Drop everything on *group* that names *object_uuid*.

    Called at the moment an object's membership ends, by every path that
    disowns one, so no list on the group outlives the assignment it was
    written against. A reference left behind would be shown in a panel and
    would silently start meaning a different object once the group's slot
    was reused.
    """
    cleanup_pin_vertex_groups_for_object(group, object_uuid)
    cleanup_intersection_allowances_for_object(group, object_uuid)


def cleanup_intersection_allowances_for_object(group: ObjectGroup, object_uuid: str):
    """Remove *object_uuid* from every intersection-allowance subset."""
    from ...models.intersection_allowances import (
        INTERSECTION_ALLOWANCES,
        allowance_objects,
    )
    from ...models.collection_utils import safe_update_index

    for spec in INTERSECTION_ALLOWANCES:
        collection = allowance_objects(group, spec)
        removed = False
        for index in range(len(collection) - 1, -1, -1):
            if collection[index].uuid == object_uuid:
                collection.remove(index)
                removed = True
        if removed:
            setattr(
                group,
                spec.index_prop,
                safe_update_index(getattr(group, spec.index_prop), len(collection)),
            )


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
