# File: force_field_targets.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Which object groups a force field source reaches, answered in ONE place for
# the panel, the encoder, the MCP tools and the Python API.
#
# A source is a force field OBJECT (keyed by the object's uuid) or the exact
# SCRIPT. A field object with no entry in `state.force_field_targets`
# reaches every simulated group, which is the default and the common case; an
# entry is created only when the artist narrows the object, through an
# operator, since a panel's draw may not write. The script's choice lives in
# three `force_field_script_*` properties of its own.

from __future__ import annotations

SCRIPT = "SCRIPT"


def _uuid(obj) -> str:
    from ..core.uuid_registry import get_object_uuid

    return get_object_uuid(obj)


def entry_for(state, obj):
    """The target entry of field object ``obj``, or None when it has none."""
    uid = _uuid(obj)
    if not uid:
        return None
    for entry in state.force_field_targets:
        if entry.source_uuid == uid:
            return entry
    return None


def ensure_entry(state, obj):
    """The target entry of ``obj``, created when missing. Writes state: call
    from an operator, never from a draw."""
    from ..core.uuid_registry import get_or_create_object_uuid

    uid = get_or_create_object_uuid(obj)
    for entry in state.force_field_targets:
        if entry.source_uuid == uid:
            entry.source_name = obj.name
            return entry
    entry = state.force_field_targets.add()
    entry.source_uuid = uid
    entry.source_name = obj.name
    return entry


def source_view(state, source):
    """``(apply_all, groups collection or None, index attr owner, index attr)``
    for ``source``, a field object or :data:`SCRIPT`. ``groups`` is None for a
    field object with no entry, which reaches every group."""
    if source == SCRIPT:
        return (state.force_field_script_all, state.force_field_script_groups,
                state, "force_field_script_groups_index")
    entry = entry_for(state, source)
    if entry is None:
        return True, None, None, None
    return entry.apply_all, entry.groups, entry, "groups_index"


def resolve_source(state, source):
    """The source by name: :data:`SCRIPT`, or a field object's name."""
    import bpy  # pyright: ignore

    if source == SCRIPT:
        return SCRIPT
    obj = bpy.data.objects.get(source)
    if obj is None or getattr(obj, "field", None) is None or obj.field.type == "NONE":
        raise ValueError(f"{source!r} is neither SCRIPT nor a force field object")
    return obj


def target_group_uuids(state, source):
    """None when ``source`` reaches every group, else its group uuids."""
    apply_all, groups, _, _ = source_view(state, source)
    if apply_all or groups is None:
        return None
    return [ref.uuid for ref in groups]


def ref_problem(scene, ref) -> str | None:
    """Why a group reference cannot be sent, or None when it can."""
    from .groups import get_group_by_uuid

    group = get_group_by_uuid(scene, ref.uuid)
    if group is None or not group.active:
        return f"group '{ref.name}' no longer exists"
    if str(group.object_type) == "STATIC":
        return f"group '{group.name}' is Static, and colliders ignore force fields"
    return None


def set_targets(scene, state, source, group_uuids) -> None:
    """Point ``source`` at every group (``group_uuids`` None) or at exactly
    the groups named. Refuses a group that does not exist or is Static, and
    an empty list, which would reach nothing."""
    from .groups import get_group_by_uuid

    if group_uuids is not None:
        if not group_uuids:
            raise ValueError("an empty group list reaches nothing; pass None for every group")
        for uid in group_uuids:
            group = get_group_by_uuid(scene, uid)
            if group is None or not group.active:
                raise ValueError(f"no active group has uuid {uid!r}")
            if str(group.object_type) == "STATIC":
                raise ValueError(f"group '{group.name}' is Static, and colliders ignore force fields")
    if source == SCRIPT:
        groups = state.force_field_script_groups
        if group_uuids is None:
            state.force_field_script_all = True
            return
        state.force_field_script_all = False
    else:
        entry = ensure_entry(state, source)
        groups = entry.groups
        if group_uuids is None:
            entry.apply_all = True
            return
        entry.apply_all = False
    groups.clear()
    for uid in group_uuids:
        ref = groups.add()
        ref.uuid = uid
        ref.name = get_group_by_uuid(scene, uid).name


def encode_positions(scene, state, source, position_by_uuid):
    """The payload's ``groups`` for ``source``: None for every group, else the
    positions of its groups in the PARAM payload's group list, which is the
    label the frontend decoder gives each group.

    Raises ValueError naming the source for a reference that cannot be sent
    and for a narrowed source with no group left.
    """
    uuids = target_group_uuids(state, source)
    if uuids is None:
        return None
    label = "The force field script" if source == SCRIPT else f"Force field '{source.name}'"
    _, groups, _, _ = source_view(state, source)
    for ref in groups:
        problem = ref_problem(scene, ref)
        if problem:
            raise ValueError(f"{label} targets a group that cannot be used: {problem}")
    if not uuids:
        raise ValueError(
            f"{label} has Apply to All Groups off and no group chosen, so it would "
            "reach nothing; add a group or turn Apply to All Groups back on"
        )
    positions = []
    for uid in uuids:
        if uid not in position_by_uuid:
            raise ValueError(f"{label} targets a group that is not simulated")
        positions.append(position_by_uuid[uid])
    return sorted(set(positions))


def drop_group(scene, group_uuid: str) -> None:
    """Remove every force field reference to a group that is going away.

    Called by `ObjectGroup.reset_to_defaults`, which every path that deletes a
    group (the panel, the Python API, MCP, Delete All) goes through, so no
    list keeps naming a group that no longer exists. A source left with an
    empty list keeps Apply to All Groups off, and Transfer then names it
    rather than guessing which groups it meant.
    """
    from .groups import get_addon_data

    if not group_uuid:
        return
    state = get_addon_data(scene).state

    def prune(collection, owner, index_attr):
        for i in reversed(range(len(collection))):
            if collection[i].uuid == group_uuid:
                collection.remove(i)
        setattr(owner, index_attr, min(getattr(owner, index_attr), len(collection) - 1))

    prune(state.force_field_script_groups, state, "force_field_script_groups_index")
    for entry in state.force_field_targets:
        prune(entry.groups, entry, "groups_index")


def display_name(scene, ref) -> str:
    """The group's CURRENT name, so a rename shows at once; the stored name
    only for a reference whose group is gone."""
    from .groups import get_group_by_uuid

    group = get_group_by_uuid(scene, ref.uuid)
    if group is not None and group.active:
        return group.name or "Dynamics Group"
    return ref.name
