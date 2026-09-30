# File: scenarios/bl_contact_gap_per_object.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A relative contact gap is a fraction of EACH OBJECT'S OWN bounding-box
# diagonal, and each object is solved at its own.
#
# "Contact Gap Ratio" and "Contact Offset Ratio" size a contact distance from
# the geometry instead of asking the artist for a length. The geometry that
# counts is the object carrying the contact: a distance taken from the box
# around the whole group also measures how far apart the group's objects were
# placed, so adding objects to a group, or spreading them out, would widen
# every gap in it. Twenty-eight objects of 8.6 stacked 290 high would each be
# given the gap of a 290 object, 0.29 on a body 1 thick (public issue #153).
#
# So the group's two lengths cross the wire as a per-uuid map in relative
# mode, and the frontend sets each object's own through `obj.param.set`. In
# absolute mode the group authors one distance and sends one number.
#
# The scene is two groups of two planes each, a SHELL group and a STATIC
# group, every plane a different size and the two of each group thirty units
# apart. A group-wide diagonal would be about thirty; the largest object's is
# under nine.
#
# Subtests:
#   A. relative_gap_is_each_objects_own
#         the SHELL group's encoded `contact-gap` is a map naming both objects,
#         each value the ratio times that object's own diagonal.
#   B. relative_offset_is_each_objects_own
#         the same for `contact-offset`.
#   C. static_group_is_per_object
#         the STATIC group's two lengths take the same shape and values.
#   D. distance_between_objects_does_not_enter
#         moving one object of each group ten times further away changes no
#         encoded value.
#   E. absolute_mode_sends_one_number
#         with the switch off the group sends a plain number, the authored one.
#   F. animated_ratio_is_per_object
#         a keyframed ratio ships one series per object under `param-anim`,
#         each the sampled ratio times that object's diagonal.
#   G. session_holds_each_objects_own
#         after a real build, the session's per-triangle `contact-gap` and
#         `contact-offset` hold both objects' values and no other, which is
#         what the frontend set on each object after decoding the map. The
#         static pool is read the same way for the two colliders.
#   H. session_schedule_is_per_object
#         the per-frame table the same build wrote holds both objects' series.
#   I. hash_does_not_follow_the_playhead
#         with a collider rotating over the timeline, which changes its
#         world-space box from frame to frame, the param hash and the encoded
#         gap are the same wherever the playhead is parked: both are read at
#         the starting frame.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND. The encode checks need none, and G and H read what
# a real server and frontend wrote after decoding the payload.
BACKENDS = ("real",)


_FRAME_COUNT = 6
_GAP_RATIO = 0.01
_OFFSET_RATIO = 0.002
_GAP_RATIO_END = 0.02


_DRIVER_BODY = r"""
import glob
import math
import os
import struct
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "<<PROJECT_NAME>>"
PROJECT_ROOT = "<<PROJECT_ROOT>>"
FRAME_COUNT = <<FRAME_COUNT>>
GAP_RATIO = <<GAP_RATIO>>
OFFSET_RATIO = <<OFFSET_RATIO>>
GAP_RATIO_END = <<GAP_RATIO_END>>


def plane(name, size, location):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def own_diagonal(obj):
    # Measured here from the vertices, independently of the add-on: the
    # diagonal of the world-space box around this one object.
    pts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))


def close(a, b):
    return abs(float(a) - float(b)) <= 1e-5 * max(abs(float(b)), 1e-6)


def encoded_groups():
    # {group name: params} for the two groups, off a fresh encode.
    blob = dh.decode_addon_blob(dh.encoder_param.encode_param(bpy.context))
    out = {}
    for params, names, uuids in blob.get("group", []):
        out[tuple(sorted(names))] = (params, dict(zip(names, uuids)))
    return blob, out


def lengths_of(groups, names, key):
    # {object name: encoded length} for one group, or the raw value when it
    # was not sent per object.
    params, uuid_of = groups[tuple(sorted(names))]
    value = params.get(key)
    if not isinstance(value, dict):
        return value
    name_of = {u: n for n, u in uuid_of.items()}
    return {name_of.get(u, u): float(v) for u, v in value.items()}


def matches(seen, expected):
    return (isinstance(seen, dict)
            and sorted(seen) == sorted(expected)
            and all(close(seen[n], expected[n]) for n in expected))


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    big = plane("Big", 4.0, (0.0, 0.0, 0.0))
    small = plane("Small", 1.0, (30.0, 0.0, 0.0))
    wall_big = plane("WallBig", 6.0, (0.0, 0.0, -5.0))
    wall_small = plane("WallSmall", 2.0, (30.0, 0.0, -5.0))
    dh.save_blend(PROBE_DIR, "contact_gap_per_object.blend")
    root = dh.configure_state(project_name=PROJECT_NAME,
                              frame_count=FRAME_COUNT)
    scene = bpy.context.scene

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(big.name, small.name)
    walls = dh.api.solver.create_group("Walls", "STATIC")
    walls.add(wall_big.name, wall_small.name)
    pg_cloth = dh.groups.get_active_group_by_uuid(scene, cloth.uuid)
    pg_walls = dh.groups.get_active_group_by_uuid(scene, walls.uuid)
    for pg in (pg_cloth, pg_walls):
        pg.use_group_bounding_box_diagonal = True
        pg.contact_gap_rat = GAP_RATIO
        pg.contact_offset_rat = OFFSET_RATIO

    CLOTH = ("Big", "Small")
    WALLS = ("WallBig", "WallSmall")
    diag = {o.name: own_diagonal(o)
            for o in (big, small, wall_big, wall_small)}
    want_gap = {n: GAP_RATIO * d for n, d in diag.items()}
    want_off = {n: OFFSET_RATIO * d for n, d in diag.items()}

    _blob, groups = encoded_groups()
    gap_cloth = lengths_of(groups, CLOTH, "contact-gap")
    off_cloth = lengths_of(groups, CLOTH, "contact-offset")
    gap_walls = lengths_of(groups, WALLS, "contact-gap")
    off_walls = lengths_of(groups, WALLS, "contact-offset")
    dh.record(
        "A_relative_gap_is_each_objects_own",
        matches(gap_cloth, {n: want_gap[n] for n in CLOTH})
        and not close(gap_cloth["Big"], gap_cloth["Small"]),
        {"encoded": gap_cloth, "expected": {n: want_gap[n] for n in CLOTH},
         "diagonals": {n: diag[n] for n in CLOTH}},
    )
    dh.record(
        "B_relative_offset_is_each_objects_own",
        matches(off_cloth, {n: want_off[n] for n in CLOTH}),
        {"encoded": off_cloth, "expected": {n: want_off[n] for n in CLOTH}},
    )
    dh.record(
        "C_static_group_is_per_object",
        matches(gap_walls, {n: want_gap[n] for n in WALLS})
        and matches(off_walls, {n: want_off[n] for n in WALLS}),
        {"gap": gap_walls, "offset": off_walls,
         "expected_gap": {n: want_gap[n] for n in WALLS}},
    )

    # ---- D: ten times further apart, and nothing moves ----------------
    small.location = (300.0, 0.0, 0.0)
    wall_small.location = (300.0, 0.0, -5.0)
    bpy.context.view_layer.update()
    _blob, far = encoded_groups()
    dh.record(
        "D_distance_between_objects_does_not_enter",
        lengths_of(far, CLOTH, "contact-gap") == gap_cloth
        and lengths_of(far, CLOTH, "contact-offset") == off_cloth
        and lengths_of(far, WALLS, "contact-gap") == gap_walls
        and lengths_of(far, WALLS, "contact-offset") == off_walls,
        {"near": gap_cloth, "far": lengths_of(far, CLOTH, "contact-gap"),
         "walls_near": gap_walls,
         "walls_far": lengths_of(far, WALLS, "contact-gap")},
    )
    small.location = (30.0, 0.0, 0.0)
    wall_small.location = (30.0, 0.0, -5.0)
    bpy.context.view_layer.update()

    # ---- E: absolute mode is one number -------------------------------
    pg_cloth.use_group_bounding_box_diagonal = False
    pg_cloth.contact_gap = 0.004
    pg_cloth.contact_offset = 0.002
    authored_gap = float(pg_cloth.contact_gap)
    authored_off = float(pg_cloth.contact_offset)
    _blob, absolute = encoded_groups()
    abs_gap = lengths_of(absolute, CLOTH, "contact-gap")
    abs_off = lengths_of(absolute, CLOTH, "contact-offset")
    dh.record(
        "E_absolute_mode_sends_one_number",
        not isinstance(abs_gap, dict) and not isinstance(abs_off, dict)
        and close(abs_gap, authored_gap) and close(abs_off, authored_off),
        {"gap": abs_gap, "offset": abs_off,
         "authored": [authored_gap, authored_off]},
    )
    pg_cloth.use_group_bounding_box_diagonal = True

    # ---- F: a keyframed ratio is one series per object -----------------
    pg_cloth.contact_gap_rat = GAP_RATIO
    pg_cloth.keyframe_insert(data_path="contact_gap_rat", frame=1)
    pg_cloth.contact_gap_rat = GAP_RATIO_END
    pg_cloth.keyframe_insert(data_path="contact_gap_rat", frame=FRAME_COUNT)
    scene.frame_set(1)
    blob, animated = encoded_groups()
    times = blob.get("param_anim_times") or []
    params, uuid_of = animated[tuple(sorted(CLOTH))]
    series = (params.get("param-anim") or {}).get("contact-gap")
    per_object = {}
    if isinstance(series, dict):
        name_of = {u: n for n, u in uuid_of.items()}
        per_object = {name_of.get(u, u): [float(v) for v in track]
                      for u, track in series.items()}
    dh.record(
        "F_animated_ratio_is_per_object",
        sorted(per_object) == sorted(CLOTH) and len(times) >= 2
        and all(len(per_object[n]) == len(times) for n in CLOTH)
        and all(close(per_object[n][0], GAP_RATIO * diag[n]) for n in CLOTH)
        and all(close(per_object[n][-1], GAP_RATIO_END * diag[n])
                for n in CLOTH),
        {"times": times, "series": per_object,
         "raw_type": type(series).__name__,
         "expected_first": {n: GAP_RATIO * diag[n] for n in CLOTH},
         "expected_last": {n: GAP_RATIO_END * diag[n] for n in CLOTH}},
    )

    # ---- G, H: what a real build holds after decoding the maps --------
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="contact gap per object")
    dh.log("built")
    # The build writes to <project root>/../<project name>/session, a SIBLING
    # of the slot directory PROJECT_ROOT names, so search from the parent.
    search_root = os.path.dirname(PROJECT_ROOT.rstrip("/")) or PROJECT_ROOT
    hits = glob.glob(os.path.join(search_root, "**", "session"), recursive=True)
    if not hits:
        raise RuntimeError("no session directory under %s" % search_root)
    session = sorted(hits, key=os.path.getmtime)[-1]

    def floats(path):
        if not os.path.isfile(path):
            return []
        with open(path, "rb") as handle:
            raw = handle.read()
        return list(struct.unpack("<%df" % (len(raw) // 4), raw))

    def holds(values, wanted):
        return any(close(v, wanted) for v in values)

    static_dir = os.path.join(session, "bin", "param")
    listing = sorted(os.listdir(static_dir)) if os.path.isdir(static_dir) else []
    tri_gap = floats(os.path.join(static_dir, "tri-contact-gap.bin"))
    tri_off = floats(os.path.join(static_dir, "tri-contact-offset.bin"))
    wall_gap = floats(os.path.join(static_dir, "static-contact-gap.bin"))
    wall_off = floats(os.path.join(static_dir, "static-contact-offset.bin"))

    def holds_exactly(values, names, wanted):
        # Every object's value is present, and nothing else is.
        expected = [wanted[n] for n in names]
        return (bool(values)
                and all(holds(values, w) for w in expected)
                and all(holds(expected, v) for v in values))

    dh.record(
        "G_session_holds_each_objects_own",
        holds_exactly(tri_gap, CLOTH, want_gap)
        and holds_exactly(tri_off, CLOTH, want_off)
        and holds_exactly(wall_gap, WALLS, want_gap)
        and holds_exactly(wall_off, WALLS, want_off),
        {"tri_contact_gap": tri_gap, "tri_contact_offset": tri_off,
         "static_contact_gap": wall_gap, "static_contact_offset": wall_off,
         "expected_gap": want_gap, "expected_offset": want_off,
         "files": listing},
    )
    anim_dir = os.path.join(session, "bin", "param_anim")
    anim_listing = sorted(os.listdir(anim_dir)) if os.path.isdir(anim_dir) else []
    tri_anim = floats(os.path.join(anim_dir, "tri-contact-gap.bin"))
    dh.record(
        "H_session_schedule_is_per_object",
        holds(tri_anim, GAP_RATIO * diag["Big"])
        and holds(tri_anim, GAP_RATIO * diag["Small"])
        and holds(tri_anim, GAP_RATIO_END * diag["Big"])
        and holds(tri_anim, GAP_RATIO_END * diag["Small"]),
        {"tri_contact_gap_anim": tri_anim, "files": anim_listing,
         "expected": [GAP_RATIO * diag["Big"], GAP_RATIO * diag["Small"],
                      GAP_RATIO_END * diag["Big"],
                      GAP_RATIO_END * diag["Small"]]},
    )

    # ---- I: the playhead does not enter --------------------------------
    # A square rotated by 45 degrees about its normal has a wider world-space
    # box, so this collider's diagonal differs from frame to frame.
    wall_small.rotation_euler = (0.0, 0.0, 0.0)
    wall_small.keyframe_insert(data_path="rotation_euler", frame=1)
    wall_small.rotation_euler = (0.0, 0.0, math.radians(45.0))
    wall_small.keyframe_insert(data_path="rotation_euler", frame=FRAME_COUNT)
    scene.frame_set(1)
    start_diag = own_diagonal(wall_small)
    hash_at_start = dh.encoder_param.compute_param_hash(bpy.context)
    _blob, at_start = encoded_groups()
    scene.frame_set(FRAME_COUNT)
    end_diag = own_diagonal(wall_small)
    hash_at_end = dh.encoder_param.compute_param_hash(bpy.context)
    _blob, at_end = encoded_groups()
    parked = scene.frame_current
    gap_start = lengths_of(at_start, WALLS, "contact-gap")
    gap_end = lengths_of(at_end, WALLS, "contact-gap")
    dh.record(
        "I_hash_does_not_follow_the_playhead",
        hash_at_start == hash_at_end and gap_start == gap_end
        and isinstance(gap_start, dict)
        and close(gap_start["WallSmall"], GAP_RATIO * start_diag)
        and end_diag > 1.2 * start_diag
        and parked == FRAME_COUNT,
        {"hash_at_start": hash_at_start, "hash_at_end": hash_at_end,
         "gap_at_start": gap_start, "gap_at_end": gap_end,
         "diagonal_at_start": start_diag, "diagonal_at_end": end_diag,
         "playhead_after": parked},
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
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
        .replace("<<GAP_RATIO_END>>", repr(_GAP_RATIO_END))
        .replace("<<GAP_RATIO>>", repr(_GAP_RATIO))
        .replace("<<OFFSET_RATIO>>", repr(_OFFSET_RATIO))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
