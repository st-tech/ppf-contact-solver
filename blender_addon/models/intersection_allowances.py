# File: intersection_allowances.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The three group-level intersection allowances of issue #138, and which of a
# group's objects each one reaches.
#
# An allowance is a per-OBJECT fact all the way down: the frontend resolves
# each object's three material params into one policy byte per vertex
# (`bin/intersect_policy.bin`). A checkbox that covered the whole group would
# therefore be the WIDER of the two settings the data model supports, and the
# wrong one for the case the feature exists for: a scene usually arrives with
# ONE garment tangled, and covering the whole group buys silence about every
# other object in it as well.
#
# The inter-group allowance is the one that needs the group at all, and only
# to answer which pairs it covers: two objects in DIFFERENT groups. The
# frontend writes that membership next to the policy (`bin/group_vert.bin`),
# while WHICH objects of a group carry the allowance is still per object like
# the other two.
#
# So each allowance carries a subset: an "Apply to All Objects" switch and,
# while it is off, a list naming the objects the allowance reaches. This
# module is the one place the three allowances' property names are spelled,
# so the panel, the operators, the encoder and the MCP surface cannot drift
# into disagreeing about which list belongs to which checkbox.

from __future__ import annotations


class AllowanceSpec:
    """The property names one allowance is stored under.

    `key` is the identifier the operators and the MCP tools take, and the
    solver parameter it feeds is `param_key`. Everything else is an
    ObjectGroup property name.
    """

    __slots__ = (
        "key",
        "label",
        "enable_prop",
        "all_objects_prop",
        "objects_prop",
        "index_prop",
        "param_key",
    )

    def __init__(self, key, label, enable_prop, param_key):
        self.key = key
        self.label = label
        self.enable_prop = enable_prop
        self.all_objects_prop = f"{enable_prop}_all_objects"
        self.objects_prop = f"{enable_prop}_objects"
        self.index_prop = f"{enable_prop}_objects_index"
        self.param_key = param_key


SELF_ALLOWANCE = AllowanceSpec(
    "self",
    "Allow Self-Intersections",
    "allow_self_intersection",
    "allow-self-intersection",
)
INTER_OBJECT_ALLOWANCE = AllowanceSpec(
    "inter_object",
    "Allow Inter-Object Intersections",
    "allow_inter_object_intersection",
    "allow-inter-object-intersection",
)

INTER_GROUP_ALLOWANCE = AllowanceSpec(
    "inter_group",
    "Allow Inter-Group Intersections",
    "allow_inter_group_intersection",
    "allow-inter-group-intersection",
)

INTERSECTION_ALLOWANCES = (
    SELF_ALLOWANCE,
    INTER_OBJECT_ALLOWANCE,
    INTER_GROUP_ALLOWANCE,
)

# Operator / MCP enum items, in panel order.
INTERSECTION_ALLOWANCE_ITEMS = [
    (
        SELF_ALLOWANCE.key,
        "Self-Intersections",
        "An overlap of one object with itself",
    ),
    (
        INTER_OBJECT_ALLOWANCE.key,
        "Inter-Object Intersections",
        "An overlap between two different objects",
    ),
    (
        INTER_GROUP_ALLOWANCE.key,
        "Inter-Group Intersections",
        "An overlap between two objects assigned to different groups",
    ),
]


def allowance_by_key(key: str) -> AllowanceSpec:
    """The spec named by *key*, raising on anything else.

    A caller that mistypes the key would otherwise silently act on the other
    allowance's list, which reads as a UI that drops entries.
    """
    for spec in INTERSECTION_ALLOWANCES:
        if spec.key == key:
            return spec
    raise ValueError(
        f"unknown intersection allowance {key!r}; it is one of "
        + ", ".join(repr(s.key) for s in INTERSECTION_ALLOWANCES)
    )


def allowance_objects(group, spec: AllowanceSpec):
    """The subset collection of *group* for *spec*."""
    return getattr(group, spec.objects_prop)


def allowance_applies_to_all(group, spec: AllowanceSpec) -> bool:
    return bool(getattr(group, spec.all_objects_prop))


def allowance_enabled(group, spec: AllowanceSpec) -> bool:
    return bool(getattr(group, spec.enable_prop))


def allowed_object_uuids(group, spec: AllowanceSpec) -> set[str]:
    """The uuids of the INCLUDED assigned objects *spec* reaches on *group*.

    Empty when the allowance is off. When it applies to all, every included
    assigned object; otherwise the named subset, intersected with the group's
    membership so an entry left behind by a removed object reaches nothing.
    That intersection is what makes a stale entry harmless rather than a way
    to ship an allowance for an object this group no longer holds.
    """
    if not allowance_enabled(group, spec):
        return set()
    member_uuids = {
        assigned.uuid
        for assigned in group.assigned_objects
        if assigned.included and assigned.uuid
    }
    if allowance_applies_to_all(group, spec):
        return member_uuids
    return {
        item.uuid
        for item in allowance_objects(group, spec)
        if item.uuid
    } & member_uuids
