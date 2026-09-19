# File: scenarios/bl_intersection_allowances.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The ADDON half of the three intersection allowances of issue #138: two
# per-group checkboxes ("Allow Self-Intersections", "Allow Inter-Object
# Intersections") and a per-pin one ("Allow Intersections Here").
#
# `rig_intersection_allowances` covers what the
# allowances MEAN, at the scene-build gate and at the solver's own scan. Both
# drive the frontend API directly and never load Blender, so neither can see
# whether a checkbox is wired to anything. A property that is declared, drawn
# and never encoded passes both of them and ships a UI that sets nothing.
# This scenario closes that gap by driving real Blender and reading what the
# build wrote.
#
# It asserts the ENCODED ARTIFACT rather than the RNA. Reading back a
# BoolProperty that was just assigned proves only that Blender stores
# booleans, and it does not even prove that much: a PropertyGroup accepts an
# assignment to an UNREGISTERED name as a stray ID-property, so a scenario
# built on read-back would pass against an addon tree that never declared the
# property at all. The witnesses here are three files the build worker writes
# into the session directory:
#
#   bin/object_vert.bin      u32 per dynamic vertex, source-object identity.
#                            This is the only thing that separates a SELF
#                            intersection from an INTER-OBJECT one, so it has
#                            to give each OBJECT a distinct id. Identity is
#                            per Blender object, not per group: two meshes in
#                            one group form an inter-object pair.
#   bin/intersect_policy.bin u8 per dynamic vertex, bit 0 = allow self,
#                            bit 1 = allow inter-object. Written only when
#                            some object asks for an allowance.
#   info.toml                the pin sections, where the per-pin flag lands as
#                            `allow_intersection = true`.
#
# The scene holds five objects, one per group, differing in which allowances
# their group carries and in the group's TYPE:
#
#   SelfSheet   SHELL  allow self only          -> policy 0b01
#   InterSheet  SHELL  allow inter-object only  -> policy 0b10
#   BothSheet   SHELL  both                     -> policy 0b11
#   PlainSheet  SHELL  neither, and its PIN carries the per-pin flag
#   RigidBody   PDRD   both                     -> policy 0b11
#
# The PDRD body is what covers the SHAPE of the encoder's per-type key
# whitelist (`blender_addon/core/encoder/params.py`), which is six
# independent lists, one per group type, each of which has to name both
# allowance keys. The value dict the encoder builds computes both keys for
# every group, so a type whose list omits one ships a checkbox that sets
# nothing, and an all-SHELL scene reports that as a pass. One non-SHELL group
# turns the claim from "the SHELL row is right" into "the rows are not
# written from one shared assumption". PDRD is the cheapest type that can
# make the claim: its group takes a plain closed mesh, so the build pays no
# tetrahedralization (SOLID does, twice, once per build), and its vertices
# are ordinary dynamic vertices, so the policy byte is written for them and
# the existing per-object assertions read it unchanged. A STATIC group is not
# used here because whether it answers depends on how it is driven: animated,
# soft-constrained, or named as a cross-stitch endpoint, it decodes to a pin
# shell in the solved namespace and does carry an object id and a policy byte;
# none of those, and it stays a disjoint contact-only collision mesh that no
# byte of `intersect_policy.bin` answers for.
#
# Every object has a pin, so the pin subtest has four controls: a run where
# exactly one of the five pin sections may carry the flag is a much narrower
# claim than one where the only pin present carries it.
#
# It builds TWICE in the same project. The first build leaves every checkbox
# at its default and is the regression half: the session must carry NO
# `bin/intersect_policy.bin` and no `allow_intersection` anywhere in
# `info.toml`, so a scene that leaves the three allowances alone carries no
# trace of them into the session. The second build sets the flags and reads
# the same three files again. Checking the default case against a real build
# rather than against the encoder alone is what makes it cover the frontend's
# "write the file only when some object asks" rule as well as the addon's.
#
# Subtests:
#   A. rna_registered_with_defaults       - all three properties exist and
#                                           default to off. A missing one
#                                           means Blender loaded an addon
#                                           tree that predates them, or was
#                                           soft-reloaded instead of
#                                           restarted (new RNA needs a full
#                                           restart).
#   B. default_build_writes_no_policy_file
#   C. default_build_pin_toml_has_no_allowance
#   D. policy_file_written_when_asked
#   E. object_vert_ids_distinct_per_object
#   F. self_allowance_sets_bit0_only
#   G. inter_allowance_sets_bit1_only
#   H. both_allowances_set_both_bits
#   I. untouched_group_stays_zero
#   J. non_shell_group_sets_both_bits
#   K. pin_flag_reaches_pin_toml
#
# F through I are the ones that separate the two checkboxes from each other.
# A wiring that sent both group booleans to the same bit, or that swapped
# them, still produces a policy file of the right length with non-zero bytes
# in it, and passes D and E. J asks the same question of a second group type,
# and it asks it of BOTH keys at once, so dropping either one from the PDRD
# whitelist row leaves a policy byte that is non-zero and still wrong.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import glob
import os
import re
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
# `crates/ppf-cts-solver/src/data.rs` and `frontend/_scene_.py`. This scenario
# reads the exported byte, so it has to name the bits the solver reads.
BIT_SELF = 1 << 0
BIT_INTER = 1 << 1

# name, x offset, group type, grid subdivisions, group name,
# (allow self, allow inter). The subdivision counts differ so the objects
# have five different vertex counts, which makes each one legible in a
# failure report before the vertex map is consulted. The offsets keep them
# well apart: an actual overlap would put the build-time intersection check
# in the middle of a measurement about file contents.
#
# The PDRD entry takes a CUBE rather than a grid. A rigid body's mass and
# inertia come from the volume its surface encloses, so an open sheet is not
# a body it can be built from.
SPECS = [
    ("SelfSheet", -6.0, "SHELL", 3, "SelfGroup", (True, False)),
    ("InterSheet", -2.0, "SHELL", 4, "InterGroup", (False, True)),
    ("BothSheet", 2.0, "SHELL", 5, "BothGroup", (True, True)),
    ("PlainSheet", 6.0, "SHELL", 6, "PlainGroup", (False, False)),
    ("RigidBody", 10.0, "PDRD", 0, "RigidGroup", (True, True)),
]
PIN_GROUP = "Edge"
EXPECTED_POLICY = {
    "SelfSheet": BIT_SELF,
    "InterSheet": BIT_INTER,
    "BothSheet": BIT_SELF | BIT_INTER,
    "PlainSheet": 0,
    "RigidBody": BIT_SELF | BIT_INTER,
}


def session_dir():
    # The build worker lays the session down under the rig's project root. A
    # run can leave more than one candidate there, so take the newest
    # directory that actually carries an info.toml rather than whichever the
    # glob orders first.
    hits = [os.path.dirname(p) for p in glob.glob(
        os.path.join(PROJECT_ROOT, "**", "session", "info.toml"),
        recursive=True)]
    if not hits:
        raise RuntimeError("no session/info.toml under %s" % PROJECT_ROOT)
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def read_bin(session, name, dtype):
    # None when the file is absent, which for intersect_policy.bin is the
    # answer the default build has to give.
    path = os.path.join(session, "bin", name)
    if not os.path.isfile(path):
        return None
    return np.fromfile(path, dtype=dtype)


def read_pin_sections(session):
    # [(pin_group_id, allows)] over every [pin-N] section, plus the raw file.
    # The `[pin-N-op-M]` sub-blocks carry neither key and drop out of the
    # section regex. pin_group_id is "<object uuid>:<vertex group>" only for a
    # pin the addon encoder emitted a cfg for; a plain pin keeps the frontend
    # default "<object name>:pin_<n>", so do not key on the uuid form.
    txt = open(os.path.join(session, "info.toml")).read()
    out = []
    for body in re.findall(r"\[pin-\d+\]\n(.*?)(?=\n\[|\Z)", txt, re.S):
        gm = re.search(r'pin_group_id\s*=\s*"([^"]*)"', body)
        out.append((gm.group(1) if gm else "",
                    bool(re.search(r"allow_intersection\s*=\s*true", body))))
    return out, txt


def load_vertex_map(session):
    # {object uuid: solver vertex index per Blender vertex}. Decoded through
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


def make_object(name, x_off, kind, subdiv):
    if kind == "PDRD":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_off, 0.0, 0.0))
    else:
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdiv,
                                        y_subdivisions=subdiv, size=1.0,
                                        location=(x_off, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    # One pin vertex group per object, over the vertices at maximum y, so
    # every object reaches the solver with a pin whatever its type.
    mesh = obj.data
    ymax = max(v.co.y for v in mesh.vertices)
    edge = [i for i, v in enumerate(mesh.vertices) if v.co.y > ymax - 1e-4]
    vg = obj.vertex_groups.new(name=PIN_GROUP)
    vg.add(edge, 1.0, "REPLACE")
    return obj


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 4

    for name, x_off, kind, subdiv, _grp, _flags in SPECS:
        obj = make_object(name, x_off, kind, subdiv)
        dh.log("object %s kind=%s verts=%d"
               % (name, kind, len(obj.data.vertices)))

    dh.save_blend(PROBE_DIR, "intersection_allowances.blend")
    root = dh.configure_state(project_name=PROJECT_NAME, frame_count=4)

    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_slot_index"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    addon_root = dh.groups.get_addon_data(scene)

    # Address a group by its SLOT. ObjectGroup.index is a display number that
    # agrees with the slot only while every slot below the group is active.
    rna_group = {}
    for name, _x, kind, _subdiv, group_name, _flags in SPECS:
        facade_group = dh.api.solver.create_group(group_name, kind)
        facade_group.add(name)
        facade_group.create_pin(name, PIN_GROUP)
        slot = groups_mod.get_group_slot_index(scene, facade_group.uuid)
        rna_group[group_name] = getattr(addon_root, "object_group_%d" % slot)
    dh.log("groups_created")

    uuid_of = {name: uuid_mod.get_or_create_object_uuid(bpy.data.objects[name])
               for name, _x, _k, _s, _g, _f in SPECS}

    # A: registration. A BoolProperty is new RNA, so it appears only after a
    # full Blender start; a soft reload leaves the panel drawing an older
    # tree while assignments below become stray ID-properties that the
    # encoder never reads. Naming the missing property here is the difference
    # between one clear line and a hunt through the artifact checks.
    group_props = set(rna_group["SelfGroup"].bl_rna.properties.keys())
    plain_pin = rna_group["PlainGroup"].pin_vertex_groups[0]
    pin_props = set(plain_pin.bl_rna.properties.keys())
    missing = [p for p in ("allow_self_intersection",
                           "allow_inter_object_intersection")
               if p not in group_props]
    if "allow_intersection" not in pin_props:
        missing.append("PinVertexGroupItem.allow_intersection")
    defaults_ok = (
        not missing
        and rna_group["SelfGroup"].allow_self_intersection is False
        and rna_group["SelfGroup"].allow_inter_object_intersection is False
        and plain_pin.allow_intersection is False
    )
    dh.record(
        "A_rna_registered_with_defaults", defaults_ok,
        {"missing": missing,
         "group_self_default": getattr(
             rna_group["SelfGroup"], "allow_self_intersection", None),
         "group_inter_default": getattr(
             rna_group["SelfGroup"], "allow_inter_object_intersection", None),
         "pin_default": getattr(plain_pin, "allow_intersection", None),
         "note": "a missing property means Blender loaded an addon tree "
                 "without it, or the addon was soft-reloaded; new RNA needs "
                 "a full restart"},
    )

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    # ---- build 1: every allowance at its default ----
    data_bytes, param_bytes = dh.encode_payload()
    dh.build_and_wait(data_bytes, param_bytes,
                      message="isect-allowance:default", timeout=240.0)
    session = session_dir()
    dh.log("default build session=%s" % session)

    default_policy = read_bin(session, "intersect_policy.bin", np.uint8)
    default_object = read_bin(session, "object_vert.bin", np.uint32)
    dh.record(
        "B_default_build_writes_no_policy_file",
        default_policy is None and default_object is not None,
        {"intersect_policy_bin": None if default_policy is None
         else int(default_policy.size),
         "object_vert_bin": None if default_object is None
         else int(default_object.size),
         "note": "object_vert.bin is written unconditionally, so its "
                 "presence is what makes the absence of intersect_policy.bin "
                 "a statement about the feature rather than about the build "
                 "having produced no bins at all"},
    )

    default_pins, default_txt = read_pin_sections(session)
    dh.record(
        "C_default_build_pin_toml_has_no_allowance",
        not any(allows for _pg, allows in default_pins)
        and "allow_intersection" not in default_txt,
        {"pin_sections": len(default_pins),
         "allowing_sections": [pg for pg, allows in default_pins if allows],
         "key_anywhere_in_file": "allow_intersection" in default_txt},
    )

    # ---- build 2: the allowances the scene actually asks for ----
    for name, _x, _kind, _subdiv, group_name, (a_self, a_inter) in SPECS:
        rna_group[group_name].allow_self_intersection = a_self
        rna_group[group_name].allow_inter_object_intersection = a_inter
    plain_pin.allow_intersection = True
    dh.log("allowances set")

    data_bytes, param_bytes = dh.encode_payload()
    dh.build_and_wait(data_bytes, param_bytes,
                      message="isect-allowance:enabled", timeout=240.0)
    session = session_dir()
    dh.log("enabled build session=%s" % session)

    policy = read_bin(session, "intersect_policy.bin", np.uint8)
    object_id = read_bin(session, "object_vert.bin", np.uint32)
    vmap = load_vertex_map(session)
    solver_index = {}
    for name, _x, _k, _s, _g, _f in SPECS:
        arr = vmap.get(uuid_of[name])
        solver_index[name] = (None if arr is None
                              else np.asarray(arr, dtype=np.int64))
    dh.log("map objects=%d indices=%r"
           % (len(vmap), {n: (None if a is None else int(a.size))
                          for n, a in solver_index.items()}))

    n_vert = None if object_id is None else int(object_id.size)
    dh.record(
        "D_policy_file_written_when_asked",
        policy is not None and object_id is not None
        and policy.size == object_id.size,
        {"intersect_policy_bin": None if policy is None else int(policy.size),
         "object_vert_bin": n_vert},
    )

    # E: object identity. Every object's vertices must carry ONE object id,
    # the five ids must differ, and between them the five objects must account
    # for every dynamic vertex. Without that last part a policy check could
    # pass while some vertices carried no object id. Such a vertex never
    # compares same-object, not even against another unmapped one, so its
    # pairs all fall to the INTER-OBJECT branch: it would lose the self
    # allowance and take the inter-object one, in both directions silently.
    # The frontend asserts full coverage at build, so this subtest is a guard
    # on that assert rather than on a reachable state.
    ids_seen = {}
    covered = []
    identity_ok = object_id is not None
    for name, _x, _k, _s, _g, _f in SPECS:
        idx = solver_index[name]
        if idx is None or object_id is None:
            identity_ok = False
            ids_seen[name] = None
            continue
        ids_seen[name] = sorted(int(v) for v in set(object_id[idx].tolist()))
        covered.append(idx)
        if len(ids_seen[name]) != 1:
            identity_ok = False
    distinct = {v[0] for v in ids_seen.values() if v}
    all_covered = (len(covered) == len(SPECS)
                   and object_id is not None
                   and int(np.unique(np.concatenate(covered)).size)
                   == int(object_id.size))
    dh.record(
        "E_object_vert_ids_distinct_per_object",
        identity_ok and len(distinct) == len(SPECS) and all_covered,
        {"ids_per_object": ids_seen, "distinct_ids": sorted(distinct),
         "dynamic_vertices": n_vert,
         "vertices_covered_by_every_object": all_covered},
    )

    # F through J: the per-group bits, one check per case so a report names
    # which allowance went astray instead of one combined verdict. J is the
    # PDRD group, so F through I answer for one row of the encoder's per-type
    # key whitelist and J answers for a second.
    policy_seen = {}
    for name, _x, _k, _s, _g, _f in SPECS:
        idx = solver_index[name]
        if idx is None or policy is None:
            policy_seen[name] = None
            continue
        policy_seen[name] = sorted(int(v) for v in set(policy[idx].tolist()))

    # The encoder picks a whitelist row by the group's OWN type, so each
    # check reads that type off the group rather than trusting the spec it
    # was built from. A `set_group_type` that did not take would leave the
    # PDRD group a SHELL, and J would then pass on the row F through I
    # already cover.
    spec_by_name = {s[0]: s for s in SPECS}
    for check, name in (
        ("F_self_allowance_sets_bit0_only", "SelfSheet"),
        ("G_inter_allowance_sets_bit1_only", "InterSheet"),
        ("H_both_allowances_set_both_bits", "BothSheet"),
        ("I_untouched_group_stays_zero", "PlainSheet"),
        ("J_non_shell_group_sets_both_bits", "RigidBody"),
    ):
        want = EXPECTED_POLICY[name]
        wanted_type = spec_by_name[name][2]
        seen_type = rna_group[spec_by_name[name][4]].object_type
        dh.record(
            check,
            policy_seen[name] == [want] and seen_type == wanted_type,
            {"object": name, "group_type": seen_type,
             "group_type_expected": wanted_type, "expected": want,
             "seen": policy_seen[name], "all_objects": policy_seen,
             "bits": "bit0=allow self, bit1=allow inter-object",
             "note": "a zero here on a group whose checkboxes are both on "
                     "means the encoder's per-type key whitelist "
                     "(core/encoder/params.py) does not name the two keys "
                     "for this group type"},
        )

    # K: the per-pin flag. Five pins reach the solver and exactly one of them
    # asked, so this is a statement about which pin carries the flag, not
    # merely that some pin does.
    pins, _pin_txt = read_pin_sections(session)
    allowing = [pg for pg, allows in pins if allows]
    dh.record(
        "K_pin_flag_reaches_pin_toml",
        len(allowing) == 1
        and allowing[0].split(":")[0] == uuid_of["PlainSheet"],
        {"pin_sections": len(pins),
         "allowing_pin_group_ids": allowing,
         "expected_object_uuid": uuid_of["PlainSheet"],
         "all_pin_group_ids": [pg for pg, _a in pins],
         "note": "pin_group_id is '<object uuid>:<vertex group>' for an "
                 "encoder-emitted pin, else '<object name>:pin_<n>'"},
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
