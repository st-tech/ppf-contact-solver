# File: scenarios/bl_contact_gap_limit.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The local gap limit caps a contact gap at a share of the room between mesh
# neighbors, element by element, so a mesh finer than its gap does not start
# the solve with its own neighbors pressing on each other.
#
# A contact pair engages inside the mean of its two sides' gaps plus their
# offsets, and the solver forms a pair between a vertex and the vertices and
# edges of its own triangle or rod segment. A gap wider than the distance
# between those neighbors is therefore active at rest: the mesh swells and
# stiffens against bending, which Gomez-Nogales et al., "Unlocking Thickness
# Modeling for Codimensional Contact Simulation" (arXiv:2510.10256), name
# contact locking. "Local Gap Limit" on a SOLID, SHELL or ROD group is
# the percentage of that room the gap may take. The frontend applies it at
# build, on the mesh the solver ingests (`frontend/_contact_gap_limit_.py`).
#
# The scene is a SHELL group holding a triangulated sheet whose cells grow by
# half from one corner to the other and a two-triangle plane, a ROD group
# holding a chain with one short segment, a STATIC plane and a PDRD cube.
#
# Subtests:
#   A. default_is_a_quarter
#         a new group's limit is 25 percent, and a SOLID, a SHELL and a ROD
#         group each encode `contact-gap-limit` as 0.25.
#   B. colliders_and_rigid_bodies_send_none
#         a STATIC and a PDRD group encode no `contact-gap-limit`.
#   C. the_setting_reaches_the_wire
#         10 percent encodes as 0.1 and moves the param hash.
#   D. the_range_is_one_to_fifty
#         a value outside it is stored clamped to the nearest end, and the two
#         ends cross the wire as exactly 0.01 and 0.5, the frontend's range.
#   E. it_offers_no_keyframe
#         inserting a keyframe on it is refused, since it is read once.
#   F. profile_round_trip
#         a material profile carries it out and back in.
#   G. panel_row_follows_the_group_type
#         the contact box draws the limit for SOLID, SHELL and ROD and not
#         for STATIC and PDRD, and a ROD draws the absolute pair.
#   H. session_gaps_are_lowered_per_triangle
#         after a real build, the session's per-triangle `contact-gap` is the
#         smaller of the authored gap and a quarter of each triangle's room,
#         computed here from the Blender mesh. The fine cells are lowered and
#         the coarse ones and the two-triangle plane are not.
#   I. session_rod_gaps_are_lowered_per_segment
#         the same for the chain: the short segment and the two it touches
#         are lowered, the rest keep the authored gap.
#   J. colliders_keep_the_authored_gap
#         the STATIC plane's and the PDRD cube's faces hold their authored
#         gaps, whatever their meshes would allow.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND. The encode and panel checks need none, and H to J
# read what a real server and frontend wrote after decoding the payload.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import glob
import math
import os
import struct
import traceback
import types

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "<<PROJECT_NAME>>"
PROJECT_ROOT = "<<PROJECT_ROOT>>"

# Cells per side of the graded sheet, and how much each is wider than the
# last. The widest is about seventeen times the narrowest.
CELLS = 8
GROWTH = 1.5
SHELL_GAP = 0.01
ROD_GAP = 0.001
ROD_OFFSET = 0.005
# One segment barely longer than two offsets among ordinary ones.
ROD_SEGMENTS = (0.1, 0.012, 0.1, 0.1, 0.1)


class MockLayout:
    # Answers the layout calls the contact box makes and records the
    # properties drawn into a sink its child layouts share.

    def __init__(self, sink=None):
        self.sink = [] if sink is None else sink
        self.enabled = True

    def box(self, *a, **k):
        return MockLayout(self.sink)

    def row(self, *a, **k):
        return MockLayout(self.sink)

    def label(self, *a, **k):
        pass

    def prop(self, data, prop_name, *a, **k):
        self.sink.append(prop_name)

    def prop_decorator(self, *a, **k):
        pass


def graded_sheet(name, location):
    axis = [0.0]
    for k in range(CELLS):
        axis.append(axis[-1] + GROWTH ** k)
    axis = [a / axis[-1] for a in axis]
    n = CELLS + 1
    verts = [(axis[i], axis[j], 0.0) for i in range(n) for j in range(n)]
    faces = []
    for i in range(CELLS):
        for j in range(CELLS):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            faces += [(a, b, c), (a, c, d)]
    mesh = bpy.data.meshes.new(name + "Data")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    return obj


def two_triangle_plane(name, size, location):
    h = 0.5 * size
    mesh = bpy.data.meshes.new(name + "Data")
    mesh.from_pydata(
        [(-h, -h, 0.0), (h, -h, 0.0), (h, h, 0.0), (-h, h, 0.0)],
        [], [(0, 1, 2), (0, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    return obj


def chain(name, location):
    x = [0.0]
    for length in ROD_SEGMENTS:
        x.append(x[-1] + length)
    mesh = bpy.data.meshes.new(name + "Data")
    mesh.from_pydata([(v, 0.0, 0.0) for v in x],
                     [(i, i + 1) for i in range(len(ROD_SEGMENTS))], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    return obj


def point_segment(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / ab.dot(ab)))
    return (p - (a + t * ab)).length


def room_per_element(obj, offset):
    # Measured here from the Blender mesh, independently of the frontend:
    # per element, the smallest size among the elements at its vertices, less
    # twice the offset. A triangle's size is the closest a vertex comes to
    # the opposite edge, and a segment's is its length.
    pts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if obj.data.polygons:
        elements = [tuple(p.vertices) for p in obj.data.polygons]
        size = [min(point_segment(pts[e[k]], pts[e[(k + 1) % 3]], pts[e[(k + 2) % 3]])
                    for k in range(3))
                for e in elements]
    else:
        elements = [tuple(e.vertices) for e in obj.data.edges]
        size = [(pts[a] - pts[b]).length for a, b in elements]
    around = {}
    for e, s in zip(elements, size):
        for v in e:
            around[v] = min(around.get(v, math.inf), s)
    return [min(around[v] for v in e) - 2.0 * offset for e in elements]


def close(a, b):
    return abs(float(a) - float(b)) <= 2e-5 * max(abs(float(b)), 1e-9)


def same_values(seen, expected):
    return (len(seen) == len(expected)
            and all(close(s, e) for s, e in zip(sorted(seen), sorted(expected))))


def encoded_groups():
    blob = dh.decode_addon_blob(dh.encoder_param.encode_param(bpy.context))
    return {tuple(sorted(names)): params
            for params, names, _uuids in blob.get("group", [])}


def limit_of(params):
    value = params.get("contact-gap-limit")
    return None if value is None else float(value)


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    fine = graded_sheet("Fine", (0.0, 0.0, 0.0))
    coarse = two_triangle_plane("Coarse", 1.0, (3.0, 0.0, 0.0))
    strand = chain("Strand", (0.0, 3.0, 0.0))
    floor = two_triangle_plane("Floor", 4.0, (0.0, 0.0, -2.0))
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(6.0, 0.0, 0.0))
    body = bpy.context.active_object
    body.name = "Body"
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(9.0, 0.0, 0.0))
    block = bpy.context.active_object
    block.name = "Block"
    dh.save_blend(PROBE_DIR, "contact_gap_limit.blend")
    root = dh.configure_state(project_name=PROJECT_NAME, frame_count=4)
    scene = bpy.context.scene
    scale = float(root.state.world_scaling)

    def group(name, kind, *objects):
        api = dh.api.solver.create_group(name, kind)
        api.add(*[o.name for o in objects])
        return api, dh.groups.get_active_group_by_uuid(scene, api.uuid)

    cloth_api, cloth = group("Cloth", "SHELL", fine, coarse)
    rod_api, rod = group("Rod", "ROD", strand)
    walls_api, walls = group("Walls", "STATIC", floor)
    rigid_api, rigid = group("Rigid", "PDRD", body)
    solid_api, solid = group("Solid", "SOLID", block)
    cloth.use_group_bounding_box_diagonal = False
    cloth.contact_gap = SHELL_GAP
    cloth.contact_offset = 0.0
    rod.contact_gap = ROD_GAP
    rod.contact_offset = ROD_OFFSET
    for pg in (walls, rigid):
        pg.use_group_bounding_box_diagonal = False
        pg.contact_gap = SHELL_GAP
        pg.contact_offset = 0.0

    CLOTH, ROD, WALLS, RIGID, SOLID = (
        ("Coarse", "Fine"), ("Strand",), ("Floor",), ("Body",), ("Block",))

    # ---- A, B: the default, and which group types send it --------------
    groups = encoded_groups()
    defaults = {kind: float(pg.contact_gap_limit_percent)
                for kind, pg in (("SHELL", cloth), ("ROD", rod),
                                 ("SOLID", solid))}
    sent = {name: limit_of(groups[key])
            for name, key in (("SHELL", CLOTH), ("ROD", ROD), ("SOLID", SOLID))}
    dh.record(
        "A_default_is_a_quarter",
        all(v == 25.0 for v in defaults.values())
        and all(v is not None and close(v, 0.25) for v in sent.values()),
        {"property": defaults, "encoded": sent},
    )
    unsent = {name: limit_of(groups[key])
              for name, key in (("STATIC", WALLS), ("PDRD", RIGID))}
    dh.record(
        "B_colliders_and_rigid_bodies_send_none",
        all(v is None for v in unsent.values()),
        {"encoded": unsent},
    )

    # ---- C: the setting reaches the wire --------------------------------
    hash_before = dh.encoder_param.compute_param_hash(bpy.context)
    cloth.contact_gap_limit_percent = 10.0
    hash_after = dh.encoder_param.compute_param_hash(bpy.context)
    ten = limit_of(encoded_groups()[CLOTH])
    dh.record(
        "C_the_setting_reaches_the_wire",
        ten is not None and close(ten, 0.1) and hash_before != hash_after,
        {"encoded": ten, "hash_moved": hash_before != hash_after},
    )

    # ---- D: the range ----------------------------------------------------
    # The two ends are what the frontend accepts, 0.01 and 0.5, and they have
    # to arrive as exactly those: a float32 of 0.01 is below 0.01 and would be
    # refused at build as out of range.
    cloth.contact_gap_limit_percent = 0.2
    low = float(cloth.contact_gap_limit_percent)
    low_sent = limit_of(encoded_groups()[CLOTH])
    cloth.contact_gap_limit_percent = 80.0
    high = float(cloth.contact_gap_limit_percent)
    high_sent = limit_of(encoded_groups()[CLOTH])
    dh.record(
        "D_the_range_is_one_to_fifty",
        low == 1.0 and high == 50.0 and low_sent == 0.01 and high_sent == 0.5,
        {"stored_for_0.2": low, "stored_for_80": high,
         "sent": [repr(low_sent), repr(high_sent)]},
    )

    # ---- E: no keyframe --------------------------------------------------
    refused = ""
    try:
        cloth.keyframe_insert(data_path="contact_gap_limit_percent", frame=1)
    except (TypeError, RuntimeError) as exc:
        refused = str(exc)
    dh.record(
        "E_it_offers_no_keyframe",
        bool(refused),
        {"refusal": refused},
    )

    # ---- F: a material profile carries it --------------------------------
    profile = __import__(pkg + ".core.profile", fromlist=["read_material_profile"])
    cloth.contact_gap_limit_percent = 40.0
    saved = profile.read_material_profile(cloth)
    cloth.contact_gap_limit_percent = 25.0
    profile.apply_material_profile(saved, cloth)
    restored = float(cloth.contact_gap_limit_percent)
    # A profile written before the limit existed names no value and leaves
    # the group's own alone.
    older = dict(saved)
    del older["contact_gap_limit_percent"]
    cloth.contact_gap_limit_percent = 15.0
    profile.apply_material_profile(older, cloth)
    kept = float(cloth.contact_gap_limit_percent)
    dh.record(
        "F_profile_round_trip",
        saved.get("contact_gap_limit_percent") == 40.0
        and restored == 40.0 and kept == 15.0,
        {"saved": saved.get("contact_gap_limit_percent"),
         "restored": restored, "kept_under_an_older_profile": kept},
    )
    cloth.contact_gap_limit_percent = 25.0

    # ---- G: the panel row -------------------------------------------------
    panels = __import__(pkg + ".ui.dynamics.panels",
                        fromlist=["_draw_contact_distances"])
    drawn = {}
    for kind, pg in (("SHELL", cloth), ("ROD", rod), ("SOLID", solid),
                     ("STATIC", walls), ("PDRD", rigid)):
        layout = MockLayout()
        panels._draw_contact_distances(layout, pg)
        drawn[kind] = layout.sink
    dh.record(
        "G_panel_row_follows_the_group_type",
        all("contact_gap_limit_percent" in drawn[k]
            for k in ("SHELL", "ROD", "SOLID"))
        and all("contact_gap_limit_percent" not in drawn[k]
                for k in ("STATIC", "PDRD"))
        and "contact_gap" in drawn["ROD"]
        and "use_group_bounding_box_diagonal" not in drawn["ROD"]
        and "contact_gap" in drawn["SHELL"]
        and "use_group_bounding_box_diagonal" in drawn["SHELL"],
        {"drawn": drawn},
    )

    # ---- H to J: what a real build holds ----------------------------------
    # A SOLID is tetrahedralized at build and its surface is the mesher's, so
    # nothing here could say what its triangles should hold. It was only
    # needed for the encode and the panel.
    solid_api.delete()
    bpy.data.objects.remove(block, do_unlink=True)

    expected_tri = []
    lowered = 0
    for obj in (fine, coarse):
        for room in room_per_element(obj, 0.0):
            bound = 0.25 * room * scale
            lowered += bound < SHELL_GAP * scale
            expected_tri.append(min(SHELL_GAP * scale, bound))
    fine_count = len(fine.data.polygons)
    expected_rod = [
        min(ROD_GAP * scale, 0.25 * room * scale)
        for room in room_per_element(strand, ROD_OFFSET)
    ]
    body_faces = sum(len(p.vertices) - 2 for p in body.data.polygons)
    expected_tri += [SHELL_GAP * scale] * body_faces

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="contact gap limit")
    dh.log("built")
    # The build writes to <project root>/../<project name>/session, a SIBLING
    # of the slot directory PROJECT_ROOT names, so search from the parent.
    search_root = os.path.dirname(PROJECT_ROOT.rstrip("/")) or PROJECT_ROOT
    hits = glob.glob(os.path.join(search_root, "**", "session"), recursive=True)
    if not hits:
        raise RuntimeError("no session directory under %s" % search_root)
    session = sorted(hits, key=os.path.getmtime)[-1]

    def floats(name):
        path = os.path.join(session, "bin", "param", name)
        if not os.path.isfile(path):
            return []
        with open(path, "rb") as handle:
            raw = handle.read()
        return list(struct.unpack("<%df" % (len(raw) // 4), raw))

    listing = sorted(os.listdir(os.path.join(session, "bin", "param")))
    tri_gap = floats("tri-contact-gap.bin")
    rod_gap = floats("rod-contact-gap.bin")
    static_gap = floats("static-contact-gap.bin")
    dh.record(
        "H_session_gaps_are_lowered_per_triangle",
        same_values(tri_gap, expected_tri)
        # Some of the graded sheet is lowered and some of it is not, so the
        # comparison above reads both sides of the minimum.
        and 0 < lowered < fine_count
        and not any(n.endswith("contact-gap-limit.bin") for n in listing),
        {"triangles": len(tri_gap), "expected": len(expected_tri),
         "lowered": lowered, "fine_triangles": fine_count,
         "smallest": [min(tri_gap or [0.0]), min(expected_tri)],
         "largest": [max(tri_gap or [0.0]), max(expected_tri)],
         "files": listing},
    )
    dh.record(
        "I_session_rod_gaps_are_lowered_per_segment",
        same_values(rod_gap, expected_rod)
        and sum(g < ROD_GAP * scale for g in expected_rod) == 3,
        {"rod_contact_gap": rod_gap, "expected": expected_rod},
    )
    dh.record(
        "J_colliders_keep_the_authored_gap",
        len(static_gap) == 2
        and all(close(g, SHELL_GAP * scale) for g in static_gap)
        and sum(close(g, SHELL_GAP * scale) for g in tri_gap)
        >= body_faces + 2,
        {"static_contact_gap": static_gap, "body_faces": body_faces},
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
