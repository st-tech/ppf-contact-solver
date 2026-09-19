# File: scenarios/bl_solid_spatial_material_map.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A SOLID's painted weights have to reach its TETRAHEDRA, interior included.
#
# The artist paints on the Blender mesh they can see. The solver's object is
# the tetrahedralized one, whose vertices are different points, so the weights
# are carried across in two convex stages: each tet surface vertex takes the
# weights of the Blender triangle closest to it, and each interior vertex takes
# the Dirichlet Laplace extension of the surface values. Both stages are convex
# combinations of painted values, which is what keeps the result inside [0, 1]
# and therefore inside the interval the group's slider and the map's target
# define.
#
# Checks, on the solver so this runs on any host:
#
#   A. payload_is_blender_sized: the map ships one weight per BLENDER vertex.
#      A payload sized to the tet mesh would mean the addon guessed at
#      geometry it does not have.
#   B. tets_carry_the_map: bin/param/tet-young-mod.bin holds one value per tet
#      and more than one distinct value.
#   C. blend_endpoints_respected: every tet value lies between the group's own
#      slider and the map's target, which the convexity argument guarantees.
#   D. interior_is_extended: at least one tet whose four vertices are all
#      interior (absent from the surface) reads strictly between the endpoints,
#      so the interior was extended rather than left at a default.
#   E. surface_tris_carry_it_too: a solid ships surface triangles as well, and
#      their table varies too, so the two element kinds agree on the map.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)

_BASE_YOUNG = 1000.0
_TARGET_YOUNG = 9000.0
_SUBDIV = 3


_DRIVER_BODY = r"""
import glob
import os
import struct
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_ROOT = "<<PROJECT_ROOT>>"
BASE_YOUNG = <<BASE_YOUNG>>
TARGET_YOUNG = <<TARGET_YOUNG>>
SUBDIV = <<SUBDIV>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "MappedBox"
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=SUBDIV)
    bpy.ops.object.mode_set(mode="OBJECT")

    zmax = max(v.co.z for v in cube.data.vertices)
    anchor = [v.index for v in cube.data.vertices if abs(v.co.z - zmax) < 1e-4]
    cube.vertex_groups.new(name="Anchor").add(anchor, 1.0, "REPLACE")

    # A gradient along z, painted on the surface the artist can see.
    zs = [v.co.z for v in cube.data.vertices]
    lo, hi = min(zs), max(zs)
    stiffen = cube.vertex_groups.new(name="Stiffen")
    for i, v in enumerate(cube.data.vertices):
        stiffen.add([i], (v.co.z - lo) / (hi - lo), "REPLACE")

    root = dh.configure_state(
        project_name="solid_spatial_map",
        frame_count=2,
        frame_rate=100,
        step_size=0.01,
        gravity=(0.0, 0.0, 0.0),
    )
    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(cube.name)
    pin = solid.create_pin(cube.name, "Anchor")
    pin.move_by(delta=(0.0, 0.0, 0.02), frame_start=1, frame_end=2,
                transition="LINEAR")

    group = root.object_group_0
    group.solid_young_modulus = BASE_YOUNG
    group.young_mod_density_normalized = True
    entry = group.material_maps.add()
    entry.parameter = "young-mod"
    entry.source_type = "VERTEX_GROUP"
    entry.source_name = "Stiffen"
    entry.target_value = TARGET_YOUNG
    dh.log("solid map authored")

    data_bytes, param_bytes = dh.encode_payload()
    blob = dh.decode_addon_blob(param_bytes)
    seen = {}
    for e in blob.get("group", []):
        mm = e[0].get("material-maps")
        if mm:
            seen = mm
    row = seen.get("young-mod") or {}
    weights = list((row.get("weights") or {}).values())
    flat = weights[0] if weights else []
    dh.record(
        "A_payload_is_blender_sized",
        len(flat) == len(cube.data.vertices),
        {"n_weights": len(flat), "n_blender_verts": len(cube.data.vertices)},
    )

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="solid spatial map",
                      timeout=240.0)
    dh.log("built")

    search_root = os.path.dirname(PROJECT_ROOT.rstrip("/")) or PROJECT_ROOT
    hits = glob.glob(os.path.join(search_root, "**", "session"), recursive=True)
    if not hits:
        raise RuntimeError("no session directory under %s" % search_root)
    session = sorted(hits, key=os.path.getmtime)[-1]
    binp = os.path.join(session, "bin")

    def read_f32(path):
        raw = open(path, "rb").read()
        return list(struct.unpack("<%df" % (len(raw) // 4), raw))

    def read_u64(path):
        raw = open(path, "rb").read()
        return list(struct.unpack("<%dQ" % (len(raw) // 8), raw))

    tet_vals = read_f32(os.path.join(binp, "param", "tet-young-mod.bin"))
    tets = read_u64(os.path.join(binp, "tet.bin"))
    tris = read_u64(os.path.join(binp, "tri.bin"))
    n_tet = len(tets) // 4
    distinct = len(set(round(v, 3) for v in tet_vals))
    dh.record(
        "B_tets_carry_the_map",
        len(tet_vals) == n_tet and n_tet > 0 and distinct > 1,
        {"n_tet": n_tet, "n_values": len(tet_vals), "distinct": distinct},
    )
    lo_v = min(tet_vals) if tet_vals else None
    hi_v = max(tet_vals) if tet_vals else None
    dh.record(
        "C_blend_endpoints_respected",
        bool(tet_vals) and lo_v >= BASE_YOUNG - 1e-2
        and hi_v <= TARGET_YOUNG + 1e-2,
        {"min": lo_v, "max": hi_v, "base": BASE_YOUNG, "target": TARGET_YOUNG},
    )

    surface = set(tris)
    interior_reads = [
        tet_vals[t]
        for t in range(n_tet)
        if not any(v in surface for v in tets[4 * t:4 * t + 4])
    ]
    strictly_inside = [
        v for v in interior_reads
        if BASE_YOUNG + 1e-2 < v < TARGET_YOUNG - 1e-2
    ]
    dh.record(
        "D_interior_is_extended",
        bool(interior_reads) and bool(strictly_inside),
        {"n_fully_interior_tets": len(interior_reads),
         "n_strictly_between": len(strictly_inside),
         "note": "zero interior tets means the fixture is too coarse"},
    )

    tri_vals = read_f32(os.path.join(binp, "param", "tri-young-mod.bin"))
    tri_distinct = len(set(round(v, 3) for v in tri_vals))
    dh.record(
        "E_surface_tris_carry_it_too",
        len(tri_vals) == len(tris) // 3 and tri_distinct > 1,
        {"n_tri": len(tris) // 3, "n_values": len(tri_vals),
         "distinct": tri_distinct},
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
        .replace("<<PROJECT_ROOT>>", ctx.project_root.replace("\\", "/"))
        .replace("<<BASE_YOUNG>>", repr(_BASE_YOUNG))
        .replace("<<TARGET_YOUNG>>", repr(_TARGET_YOUNG))
        .replace("<<SUBDIV>>", repr(_SUBDIV))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 480.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
