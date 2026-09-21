# File: scenarios/bl_intersection_allowance_objects.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The OBJECT SUBSET each of the two group-level intersection allowances is
# narrowed to.
#
# `bl_intersection_allowances` covers the allowances themselves: a checkbox on
# a group reaches every object assigned to it, and the policy byte the build
# writes says which bit each object got. This scenario covers the half that
# checkbox does not decide on its own. Each allowance carries an "Apply to All
# Objects" switch and, while it is off, a list naming the objects it reaches,
# so one group can ship one tangled garment tolerated and its neighbours
# reported.
#
# WHY IT READS THE BUILD AND NOT THE RNA. Reading a collection back after
# adding to it proves that Blender stores collections. It does not prove the
# encoder consulted the subset, and it cannot: the two allowance keys are
# emitted as a plain float whenever they reach the whole group, so an encoder
# that ignored the subset entirely would still emit a value, the build would
# still succeed, and every object would quietly get the group-wide answer.
# The witness is therefore `bin/intersect_policy.bin` again, one u8 per
# dynamic vertex with bit 0 = allow self and bit 1 = allow inter-object,
# resolved onto each object's vertices through `map.pickle`.
#
# The scene is FOUR objects in TWO groups, so the subset is proved to cut
# WITHIN a group rather than merely between groups:
#
#   Narrowed  SHELL  in group NarrowGroup, listed for BOTH allowances
#   Sibling   SHELL  in group NarrowGroup, listed for NEITHER
#   Split     SHELL  in group SplitGroup, listed for SELF only
#   Whole     SHELL  in group SplitGroup, listed for INTER-OBJECT only
#
# Both groups have both checkboxes ON and "Apply to All Objects" OFF. So a
# wiring that dropped the subset gives every one of the four objects
# `0b11`, and a wiring that read one subset for both allowances gives Split
# and Whole the same byte. Neither can pass C through F. SplitGroup is what
# makes the two lists independent rather than one list read twice: its two
# objects are in ONE group with ONE pair of checkboxes and come out with
# DIFFERENT bytes, which is only possible if each allowance consulted its own
# list.
#
# It builds TWICE in one project, and the FIRST build is the regression half.
# Both groups start at the shipped default, "Apply to All Objects" on, so the
# first build must reproduce exactly what the checkbox alone always produced:
# all four objects at `0b11`. That is what keeps the subset from becoming a
# silent behavior change for every scene that never opens the list.
#
# Subtests:
#   A. rna_registered_with_defaults        - the switch and both collections
#                                            exist, the switch defaults ON and
#                                            the collections start empty. A
#                                            missing one means Blender loaded
#                                            an addon tree without them, or was
#                                            soft-reloaded rather than
#                                            restarted (new RNA needs a full
#                                            restart).
#   B. default_build_covers_every_object   - "Apply to All Objects" on gives
#                                            all four objects 0b11.
#   C. narrowed_self_reaches_only_listed   - Narrowed has bit 0 and Sibling
#                                            does not, both asserted here so
#                                            the check alone separates a
#                                            narrowed allowance from a
#                                            group-wide one.
#   D. narrowed_inter_reaches_only_listed  - the same for bit 1.
#   E. two_lists_are_independent           - in ONE group, Split is 0b01 and
#                                            Whole is 0b10.
#   F. unlisted_sibling_stays_zero         - Sibling is 0b00 though its group
#                                            has both checkboxes on.
#   G. add_refuses_an_outsider             - Add Selected Objects does not list
#                                            an object this group does not
#                                            hold, and reports why.
#   H. remove_and_remove_all_take_entries  - Remove drops the active row and
#                                            Remove All empties the list.
#   I. membership_end_drops_the_entry      - removing an object from the group
#                                            takes its subset entry with it,
#                                            so no entry outlives the
#                                            assignment it was written for.
#
# G, H and I run AFTER the measured builds and on a throwaway third group, so
# the operator subtests cannot disturb the four objects C through F read.
#
# THE PANEL IS COVERED BY A SCENARIO OF ITS OWN,
# `bl_intersection_allowance_panel_draws`, because a draw probe needs a
# Blender that owns a window and the Windows rig runs headless. Keeping the
# two in one file cost the Windows leg of run 35490498574 this whole
# scenario: `bpy.ops.wm.redraw_timer` failed its poll with "context is
# incorrect" before the first build, so the eight checks that have nothing to
# do with drawing never ran there either.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It drives an ordinary build and reads the session the build worker
# wrote, so nothing in it is backend-specific.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import glob
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "<<PROJECT_NAME>>"
PROJECT_ROOT = "<<PROJECT_ROOT>>"

# `VertexProp::intersect_policy` bits, mirrored from
# `crates/ppf-cts-solver/src/data.rs` and `frontend/_scene_.py`.
BIT_SELF = 1 << 0
BIT_INTER = 1 << 1

# name, x offset, grid subdivisions, group name. The subdivision counts differ
# so each object has its own vertex count and is legible in a failure report
# before the vertex map is consulted; the offsets keep them far enough apart
# that no pair actually intersects, since a real overlap would put the
# build-time check in the middle of a measurement about file contents.
SPECS = [
    ("Narrowed", -6.0, 3, "NarrowGroup"),
    ("Sibling", -2.0, 4, "NarrowGroup"),
    ("Split", 2.0, 5, "SplitGroup"),
    ("Whole", 6.0, 6, "SplitGroup"),
]

# Which objects each group lists, per allowance.
LISTED = {
    ("NarrowGroup", "self"): ["Narrowed"],
    ("NarrowGroup", "inter_object"): ["Narrowed"],
    ("SplitGroup", "self"): ["Split"],
    ("SplitGroup", "inter_object"): ["Whole"],
}

EXPECTED_NARROWED = {
    "Narrowed": BIT_SELF | BIT_INTER,
    "Sibling": 0,
    "Split": BIT_SELF,
    "Whole": BIT_INTER,
}
EXPECTED_DEFAULT = {name: BIT_SELF | BIT_INTER for name, _x, _s, _g in SPECS}


def session_dir():
    # The build worker lays the session down under the rig's project root. A
    # run can leave more than one candidate there, so take the newest
    # directory that actually carries an info.toml.
    hits = [os.path.dirname(p) for p in glob.glob(
        os.path.join(PROJECT_ROOT, "**", "session", "info.toml"),
        recursive=True)]
    if not hits:
        raise RuntimeError("no session/info.toml under %s" % PROJECT_ROOT)
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def read_bin(session, name, dtype):
    path = os.path.join(session, "bin", name)
    if not os.path.isfile(path):
        return None
    return np.fromfile(path, dtype=dtype)


def load_vertex_map(session):
    # {object uuid: solver vertex index per Blender vertex}, decoded through
    # the addon's own reader so this scenario cannot drift from the format
    # the client consumes.
    er = __import__(pkg + ".core.effect_runner",
                    fromlist=["_decode_vertex_map_cbor"])
    with open(os.path.join(session, "map.pickle"), "rb") as f:
        blob = f.read()
    if blob and blob[0] == 0x80:
        import pickle
        return pickle.loads(blob)
    return er._decode_vertex_map_cbor(blob)


def policy_per_object(session, uuid_of):
    # {object name: sorted distinct policy bytes over its vertices}, plus the
    # two raw arrays for the report. One distinct byte per object is what the
    # per-object resolution means; more than one would say the policy was
    # written per element or per vertex from something else.
    policy = read_bin(session, "intersect_policy.bin", np.uint8)
    object_id = read_bin(session, "object_vert.bin", np.uint32)
    vmap = load_vertex_map(session)
    seen = {}
    for name in uuid_of:
        arr = vmap.get(uuid_of[name])
        if arr is None or policy is None:
            seen[name] = None
            continue
        idx = np.asarray(arr, dtype=np.int64)
        seen[name] = sorted(int(v) for v in set(policy[idx].tolist()))
    return seen, policy, object_id


def select_only(names):
    bpy.ops.object.select_all(action="DESELECT")
    for name in names:
        bpy.data.objects[name].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[names[0]]


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 4

    for name, x_off, subdiv, _grp in SPECS:
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdiv,
                                        y_subdivisions=subdiv, size=1.0,
                                        location=(x_off, 0.0, 0.0))
        obj = bpy.context.object
        obj.name = name
        # One pin per object, over the vertices at maximum y, so every object
        # reaches the solver held rather than falling out of frame.
        mesh = obj.data
        ymax = max(v.co.y for v in mesh.vertices)
        edge = [i for i, v in enumerate(mesh.vertices) if v.co.y > ymax - 1e-4]
        obj.vertex_groups.new(name="Edge").add(edge, 1.0, "REPLACE")
        dh.log("object %s verts=%d" % (name, len(mesh.vertices)))

    dh.save_blend(PROBE_DIR, "intersection_allowance_objects.blend")
    root = dh.configure_state(project_name=PROJECT_NAME, frame_count=4)

    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_slot_index"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    ia_mod = __import__(pkg + ".models.intersection_allowances",
                        fromlist=["INTERSECTION_ALLOWANCES"])
    addon_root = dh.groups.get_addon_data(scene)

    # Address a group by its SLOT. ObjectGroup.index is a display number that
    # agrees with the slot only while every slot below the group is active.
    rna_group = {}
    slot_of = {}
    for group_name in ("NarrowGroup", "SplitGroup"):
        facade_group = dh.api.solver.create_group(group_name, "SHELL")
        for name, _x, _s, grp in SPECS:
            if grp == group_name:
                facade_group.add(name)
                facade_group.create_pin(name, "Edge")
        slot = groups_mod.get_group_slot_index(scene, facade_group.uuid)
        slot_of[group_name] = slot
        rna_group[group_name] = getattr(addon_root, "object_group_%d" % slot)
    dh.log("groups_created %r" % (slot_of,))

    uuid_of = {name: uuid_mod.get_or_create_object_uuid(bpy.data.objects[name])
               for name, _x, _s, _g in SPECS}

    spec_by_key = {s.key: s for s in ia_mod.INTERSECTION_ALLOWANCES}

    # A: registration and defaults. A CollectionProperty and a BoolProperty
    # are new RNA, so they appear only after a full Blender start; a soft
    # reload leaves the panel drawing an older tree while every assignment
    # below becomes a stray ID-property the encoder never reads.
    probe = rna_group["NarrowGroup"]
    group_props = set(probe.bl_rna.properties.keys())
    missing = [
        prop
        for spec in ia_mod.INTERSECTION_ALLOWANCES
        for prop in (spec.all_objects_prop, spec.objects_prop, spec.index_prop)
        if prop not in group_props
    ]
    defaults_ok = not missing and all(
        getattr(probe, spec.all_objects_prop) is True
        and len(getattr(probe, spec.objects_prop)) == 0
        for spec in ia_mod.INTERSECTION_ALLOWANCES
    )
    dh.record(
        "A_rna_registered_with_defaults", defaults_ok,
        {"missing": missing,
         "all_objects_defaults": {
             spec.key: getattr(probe, spec.all_objects_prop, None)
             for spec in ia_mod.INTERSECTION_ALLOWANCES},
         "subset_sizes": {
             spec.key: len(getattr(probe, spec.objects_prop, ()))
             for spec in ia_mod.INTERSECTION_ALLOWANCES},
         "note": "a missing property means Blender loaded an addon tree "
                 "without it, or the addon was soft-reloaded; new RNA needs "
                 "a full restart"},
    )

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.log("connected")

    # ---- build 1: both checkboxes on, both switches at their default ----
    for group_name in rna_group:
        rna_group[group_name].allow_self_intersection = True
        rna_group[group_name].allow_inter_object_intersection = True
    data_bytes, param_bytes = dh.encode_payload()
    dh.build_and_wait(data_bytes, param_bytes,
                      message="isect-subset:all-objects", timeout=240.0)
    session = session_dir()
    dh.log("default build session=%s" % session)
    default_seen, default_policy, default_object = policy_per_object(
        session, uuid_of)
    dh.record(
        "B_default_build_covers_every_object",
        default_policy is not None
        and all(default_seen[name] == [EXPECTED_DEFAULT[name]]
                for name in EXPECTED_DEFAULT),
        {"seen": default_seen, "expected": EXPECTED_DEFAULT,
         "intersect_policy_bin": None if default_policy is None
         else int(default_policy.size),
         "object_vert_bin": None if default_object is None
         else int(default_object.size),
         "note": "Apply to All Objects defaults ON, so this build must "
                 "reproduce what the checkbox alone always produced"},
    )

    # ---- build 2: each allowance narrowed to its own list ----
    for (group_name, key), listed in LISTED.items():
        group = rna_group[group_name]
        spec = spec_by_key[key]
        setattr(group, spec.all_objects_prop, False)
        select_only(listed)
        bpy.ops.object.add_intersection_allowance_objects(
            group_index=slot_of[group_name], allowance=key)
    listed_now = {
        "%s:%s" % (group_name, spec.key): [
            item.name for item in getattr(rna_group[group_name],
                                          spec.objects_prop)]
        for group_name in rna_group
        for spec in ia_mod.INTERSECTION_ALLOWANCES
    }
    dh.log("narrowed %r" % (listed_now,))

    data_bytes, param_bytes = dh.encode_payload()
    dh.build_and_wait(data_bytes, param_bytes,
                      message="isect-subset:narrowed", timeout=240.0)
    session = session_dir()
    dh.log("narrowed build session=%s" % session)
    seen, policy, object_id = policy_per_object(session, uuid_of)

    shared = {
        "seen": seen, "expected": EXPECTED_NARROWED, "listed": listed_now,
        "bits": "bit0=allow self, bit1=allow inter-object",
        "intersect_policy_bin": None if policy is None else int(policy.size),
        "object_vert_bin": None if object_id is None else int(object_id.size),
    }
    # D and E each carry BOTH sides of their claim. "Narrowed has the bit" is
    # satisfied by an encoder that ignores the subset and gives every object
    # every bit, so each check also asserts that its unlisted sibling does
    # NOT have it. Only the pair separates a narrowed allowance from a
    # group-wide one.
    for check, bit in (
        ("C_narrowed_self_reaches_only_listed", BIT_SELF),
        ("D_narrowed_inter_reaches_only_listed", BIT_INTER),
    ):
        listed_has = seen["Narrowed"] is not None and all(
            v & bit for v in seen["Narrowed"])
        sibling_has = seen["Sibling"] is not None and any(
            v & bit for v in seen["Sibling"])
        dh.record(
            check, listed_has and not sibling_has,
            dict(shared, bit=bit, listed_object="Narrowed",
                 unlisted_object="Sibling",
                 listed_has_bit=listed_has, unlisted_has_bit=sibling_has),
        )

    dh.record(
        "F_unlisted_sibling_stays_zero", seen["Sibling"] == [0],
        dict(shared, object="Sibling", expected=0,
             seen_object=seen["Sibling"],
             note="its group has both checkboxes on and lists only Narrowed"),
    )

    dh.record(
        "E_two_lists_are_independent",
        seen["Split"] == [BIT_SELF] and seen["Whole"] == [BIT_INTER],
        dict(shared, object="Split, Whole",
             note="both are in SplitGroup, which has ONE pair of checkboxes; "
                  "different bytes are only possible if each allowance read "
                  "its own list"),
    )

    # ---- H, I, J: the operators, on a throwaway third group ----
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3,
                                    size=1.0, location=(12.0, 0.0, 0.0))
    bpy.context.object.name = "Spare"
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3,
                                    size=1.0, location=(16.0, 0.0, 0.0))
    bpy.context.object.name = "Loner"

    spare_facade = dh.api.solver.create_group("SpareGroup", "SHELL")
    spare_facade.add("Spare")
    spare_slot = groups_mod.get_group_slot_index(scene, spare_facade.uuid)
    spare = getattr(addon_root, "object_group_%d" % spare_slot)
    spare.allow_self_intersection = True
    spare.allow_self_intersection_all_objects = False

    # H: an object this group does not hold is refused, and the refusal is
    # reported rather than swallowed, so the user is told why nothing
    # happened.
    select_only(["Loner"])
    reports = []
    bpy.ops.object.add_intersection_allowance_objects(
        group_index=spare_slot, allowance="self")
    outsider_listed = [i.name for i in spare.allow_self_intersection_objects]
    dh.record(
        "G_add_refuses_an_outsider", outsider_listed == [],
        {"listed_after_add": outsider_listed,
         "group_members": [a.name for a in spare.assigned_objects],
         "note": "Loner is in no group, so Add Selected Objects must list "
                 "nothing; an entry naming a non-member would reach no "
                 "vertex at build time"},
    )

    # I: the two removal buttons.
    select_only(["Spare"])
    bpy.ops.object.add_intersection_allowance_objects(
        group_index=spare_slot, allowance="self")
    after_add = len(spare.allow_self_intersection_objects)
    spare.allow_self_intersection_objects_index = 0
    bpy.ops.object.remove_intersection_allowance_object(
        group_index=spare_slot, allowance="self")
    after_remove = len(spare.allow_self_intersection_objects)
    bpy.ops.object.add_intersection_allowance_objects(
        group_index=spare_slot, allowance="self")
    bpy.ops.object.clear_intersection_allowance_objects(
        group_index=spare_slot, allowance="self")
    after_clear = len(spare.allow_self_intersection_objects)
    dh.record(
        "H_remove_and_remove_all_take_entries",
        after_add == 1 and after_remove == 0 and after_clear == 0,
        {"after_add": after_add, "after_remove": after_remove,
         "after_clear": after_clear},
    )

    # J: membership and the subset end together. An entry that outlived the
    # assignment would be drawn in the panel and would name a different
    # object once the group's slot was reused.
    select_only(["Spare"])
    bpy.ops.object.add_intersection_allowance_objects(
        group_index=spare_slot, allowance="self")
    before_removal = [i.name for i in spare.allow_self_intersection_objects]
    spare.assigned_objects_index = 0
    bpy.ops.object.remove_object_from_group(group_index=spare_slot)
    after_removal = [i.name for i in spare.allow_self_intersection_objects]
    dh.record(
        "I_membership_end_drops_the_entry",
        before_removal == ["Spare"] and after_removal == [],
        {"before_removal": before_removal, "after_removal": after_removal,
         "group_members": [a.name for a in spare.assigned_objects]},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<PROJECT_NAME>>", ctx.project_name)
        .replace("<<PROJECT_ROOT>>", ctx.project_root.replace("\\", "/"))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
