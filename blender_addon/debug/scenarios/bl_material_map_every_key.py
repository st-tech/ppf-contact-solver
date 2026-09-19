# File: scenarios/bl_material_map_every_key.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Every parameter a spatial map is offered for has to reach the solver varying.
#
# The failure this exists for is a parameter the UI offers a map on that gets
# shipped and then silently collapses: the artist paints it, the run succeeds,
# and nothing differs. One scenario covers all of them, because the encoder's
# rule is one map per PARAMETER per group, not one map per group, so a single
# SHELL group legally carries every SHELL key at once and a single SOLID group
# every SOLID key.
#
# The windows are pairwise DISJOINT across every (key, object type). That turns
# the range check into a key-to-target pairing check: a row mix-up, a swapped
# tri pass order, or a SHELL value landing in a SOLID slot fails outright
# rather than passing on a plausible-looking gradient.
#
#   A_<key>: the shipped table holds one value per element of its channel.
#   B_<key>: it holds MORE THAN ONE distinct value over the mapping object's
#            range. A replicated scalar passes every other check and fails this.
#   C_<key>: every value lies inside [base, target], which each element's mean
#            weight guarantees.
#   D_<key>: the spread is at least 40% of the window, so a table varying only
#            in fp32 noise fails where B alone would pass.
#
# Cross-cutting, because a solid ships BOTH surface triangles and tets and the
# two are read by different kernels:
#   E: the solid's stiffness, creep and damping vary on the TET table.
#   F: no tet-friction.bin exists. A solid's friction is a surface quantity.
#   G: no tet- file exists for any SHELL-only key.
#   H: a SHELL-only key on a SOLID group is refused at encode, by name.
#
# The solver runs the full scene BUILD, so every one of these files is
# written for real. What it cannot show is the physics, which needs a real GPU
# run to measure.

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
import struct
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_ROOT = "<<PROJECT_ROOT>>"

# (key, base, target). Pairwise disjoint so a value identifies its own key.
SHELL_WINDOWS = [
    ("young-mod", 100.0, 300.0),
    ("bend", 40.0, 80.0),
    ("friction", 0.10, 0.40),
    ("deformation-damping", 0.001, 0.004),
    ("bending-damping", 0.02, 0.05),
    ("strain-limit", 5.0, 9.0),
    ("plasticity", 1.0, 3.0),
    ("bend-plasticity", 12.0, 16.0),
    ("bend-warp", 200.0, 600.0),
    ("bend-weft", 2000.0, 6000.0),
]
# A solid's elastic energy and creep live on its TETS. Its contact does not:
# friction is read from the surface triangles, and TetParam carries no friction
# field at all, so the two are asserted on different channels.
SOLID_TET_WINDOWS = [
    ("young-mod", 1000.0, 3000.0),
    ("deformation-damping", 0.006, 0.009),
    ("plasticity", 5.0, 9.0),
]
SOLID_SURFACE_WINDOWS = [
    ("friction", 0.60, 0.90),
]
SOLID_WINDOWS = SOLID_TET_WINDOWS + SOLID_SURFACE_WINDOWS
# The solver takes strain-limit as a fraction; every other key is sent as-is.
SOLVER_SCALE = {"strain-limit": 0.01}

# The addon property each key blends away from, so the fixture sets the base on
# the group rather than trusting a default.
SHELL_BASE_PROP = {
    "young-mod": "shell_young_modulus", "bend": "bend", "friction": "friction",
    "deformation-damping": "deformation_damping",
    "bending-damping": "bending_damping",
    "strain-limit": "strain_limit_percent", "plasticity": "plasticity",
    "bend-plasticity": "bend_plasticity", "bend-warp": "bend_warp",
    "bend-weft": "bend_weft",
}
SOLID_BASE_PROP = {
    "young-mod": "solid_young_modulus", "friction": "friction",
    "deformation-damping": "deformation_damping", "plasticity": "plasticity",
}

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # A grid carries a UV, which bend-warp and bend-weft require or the build
    # refuses the anisotropy rather than discarding it.
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12,
                                    size=1.0, location=(0.0, 0.0, 0.0))
    shell = bpy.context.active_object
    shell.name = "MapShell"

    bpy.ops.mesh.primitive_cube_add(size=0.6, location=(2.0, 0.0, 0.0))
    solid = bpy.context.active_object
    solid.name = "MapSolid"
    bpy.context.view_layer.objects.active = solid
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=2)
    bpy.ops.object.mode_set(mode="OBJECT")

    def paint(obj, axis):
        vals = [getattr(v.co, axis) for v in obj.data.vertices]
        lo, hi = min(vals), max(vals)
        vg = obj.vertex_groups.new(name="Grad")
        for i, v in enumerate(obj.data.vertices):
            vg.add([i], (getattr(v.co, axis) - lo) / (hi - lo), "REPLACE")
        return vg

    paint(shell, "y")
    paint(solid, "z")
    # Pin ONE edge row of the shell. A fully pinned face is skipped by the face
    # and hinge dispatches, so pinning everything would hide the very tables
    # this asserts on.
    ymax = max(v.co.y for v in shell.data.vertices)
    shell.vertex_groups.new(name="Edge").add(
        [v.index for v in shell.data.vertices if abs(v.co.y - ymax) < 1e-5],
        1.0, "REPLACE",
    )
    zmax = max(v.co.z for v in solid.data.vertices)
    solid.vertex_groups.new(name="Edge").add(
        [v.index for v in solid.data.vertices if abs(v.co.z - zmax) < 1e-5],
        1.0, "REPLACE",
    )

    root = dh.configure_state(project_name="map_every_key", frame_count=2)
    for name, obj, kind in (("Shell", shell, "SHELL"), ("Solid", solid, "SOLID")):
        grp = dh.api.solver.create_group(name, kind)
        grp.add(obj.name)
        grp.create_pin(obj.name, "Edge")

    shell_group = root.object_group_0
    solid_group = root.object_group_1
    for group, windows, props in (
        (shell_group, SHELL_WINDOWS, SHELL_BASE_PROP),
        (solid_group, SOLID_WINDOWS, SOLID_BASE_PROP),
    ):
        # A map on a switched-off parameter is refused, so the features whose
        # keys are mapped have to be on.
        group.enable_strain_limit = True
        group.enable_plasticity = True
        group.enable_bend_plasticity = True
        group.shrink_x = 1.0
        group.shrink_y = 1.0
        group.young_mod_density_normalized = True
        for key, base, target in windows:
            setattr(group, props[key], base)
            entry = group.material_maps.add()
            entry.parameter = key
            entry.source_type = "VERTEX_GROUP"
            entry.source_name = "Grad"
            entry.target_value = target
    dh.log(f"authored {len(SHELL_WINDOWS)} shell and {len(SOLID_WINDOWS)} solid maps")

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    data_bytes, param_bytes = dh.encode_payload()
    dh.build_and_wait(data_bytes, param_bytes, message="every key", timeout=300.0)
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

    shell.data.calc_loop_triangles()
    n_shell_tri = len(shell.data.loop_triangles)
    n_tri = os.path.getsize(os.path.join(binp, "tri.bin")) // 24
    n_tet = os.path.getsize(os.path.join(binp, "tet.bin")) // 32

    def check(prefix, key, base, target, lo_i, hi_i, n_expected):
        path = os.path.join(binp, "param", f"{prefix}-{key}.bin")
        if not os.path.isfile(path):
            dh.record(f"A_{prefix}_{key}", False, {"missing": path})
            return
        vals = read_f32(path)
        scale = SOLVER_SCALE.get(key, 1.0)
        lo, hi = base * scale, target * scale
        mine = vals[lo_i:hi_i]
        spread = (max(mine) - min(mine)) if mine else 0.0
        dh.record(f"A_{prefix}_{key}_length", len(vals) == n_expected,
                  {"n": len(vals), "expected": n_expected})
        dh.record(f"B_{prefix}_{key}_varies",
                  len({round(v, 6) for v in mine}) > 1,
                  {"distinct": len({round(v, 6) for v in mine}),
                   "range": [min(mine), max(mine)] if mine else None})
        dh.record(f"C_{prefix}_{key}_window",
                  bool(mine) and min(mine) >= lo - 1e-6 and max(mine) <= hi + 1e-6,
                  {"range": [min(mine), max(mine)] if mine else None,
                   "window": [lo, hi]})
        dh.record(f"D_{prefix}_{key}_amplitude",
                  spread >= 0.4 * abs(hi - lo),
                  {"spread": spread, "window_width": abs(hi - lo)})

    for key, base, target in SHELL_WINDOWS:
        check("tri", key, base, target, 0, n_shell_tri, n_tri)
    for key, base, target in SOLID_TET_WINDOWS:
        check("tet", key, base, target, 0, n_tet, n_tet)
    for key, base, target in SOLID_SURFACE_WINDOWS:
        # The solid's own triangles are the suffix of the tri channel: rods
        # first, then pure shells, then SOLID surface triangles.
        check("tri", key, base, target, n_shell_tri, n_tri, n_tri)

    # CORRESPONDENCE. Every check above is a function of the multiset of
    # shipped values alone, so a permuted, mirrored or fully inverted weight
    # field passes all of them. Tie each element's value to its own position
    # along the painted axis: the map runs 0 at the low end to 1 at the high
    # end, so the value has to rise with the centroid. Read the geometry the
    # session actually shipped rather than the Blender mesh.
    def read_u64(path):
        raw = open(path, "rb").read()
        return list(struct.unpack("<%dQ" % (len(raw) // 8), raw))

    def read_f64(path):
        raw = open(path, "rb").read()
        return list(struct.unpack("<%dd" % (len(raw) // 8), raw))

    vert = read_f64(os.path.join(binp, "vert.bin"))
    tris = read_u64(os.path.join(binp, "tri.bin"))
    tets = read_u64(os.path.join(binp, "tet.bin"))

    def centroids(indices, per_element, axis):
        out = []
        for e in range(len(indices) // per_element):
            corners = indices[e * per_element:(e + 1) * per_element]
            out.append(sum(vert[3 * c + axis] for c in corners) / per_element)
        return out

    def correlation(values, coords):
        n = len(values)
        mv = sum(values) / n
        mc = sum(coords) / n
        num = sum((v - mv) * (c - mc) for v, c in zip(values, coords))
        dv = sum((v - mv) ** 2 for v in values) ** 0.5
        dc = sum((c - mc) ** 2 for c in coords) ** 0.5
        return num / (dv * dc) if dv > 0 and dc > 0 else 0.0

    # The solver's vertex space is not Blender's, so the painted direction is
    # found rather than assumed: the axis a channel correlates with most.
    # A permuted weight field correlates with none of the three.
    def best_axis(values, indices, per_element, lo, hi):
        best = (0.0, None)
        for axis in range(3):
            coords = centroids(indices, per_element, axis)[lo:hi]
            c = correlation(values[lo:hi], coords)
            if abs(c) > abs(best[0]):
                best = (c, axis)
        return best

    correlations = {}
    for key, _b, _t in SHELL_WINDOWS:
        vals = read_f32(os.path.join(binp, "param", f"tri-{key}.bin"))
        correlations[f"tri-{key}"] = best_axis(vals, tris, 3, 0, n_shell_tri)
    for key, _b, _t in SOLID_TET_WINDOWS:
        vals = read_f32(os.path.join(binp, "param", f"tet-{key}.bin"))
        correlations[f"tet-{key}"] = best_axis(vals, tets, 4, 0, n_tet)
    strong = all(abs(c) > 0.95 for c, _a in correlations.values())
    # Every SHELL key reads one vertex group and every SOLID key another, so
    # within each object the axis and the sign have to agree. A key whose
    # weights were inverted, or landed on another key's elements, breaks that.
    shell_axes = {(round(c, 3) > 0, a) for k, (c, a) in correlations.items()
                  if k.startswith("tri-")}
    solid_axes = {(round(c, 3) > 0, a) for k, (c, a) in correlations.items()
                  if k.startswith("tet-")}
    dh.record(
        "I_values_follow_the_painted_axis",
        bool(correlations) and strong
        and len(shell_axes) == 1 and len(solid_axes) == 1,
        {"correlations": {k: (round(c, 4), a) for k, (c, a) in correlations.items()},
         "shell_axis_and_sign": sorted(str(x) for x in shell_axes),
         "solid_axis_and_sign": sorted(str(x) for x in solid_axes),
         "note": "a permuted field correlates with no axis; an inverted or "
                 "misrouted one disagrees with its siblings"},
    )

    absent = []
    for key in ("friction",):
        if os.path.isfile(os.path.join(binp, "param", f"tet-{key}.bin")):
            absent.append(f"tet-{key}.bin exists")
    dh.record(
        "F_no_tet_friction",
        not absent,
        {"note": "a solid's friction is read from its surface triangles",
         "unexpected": absent},
    )
    shell_only = [k for k, _b, _t in SHELL_WINDOWS
                  if k not in {s for s, _b2, _t2 in SOLID_WINDOWS}]
    leaked = [k for k in shell_only
              if os.path.isfile(os.path.join(binp, "param", f"tet-{k}.bin"))]
    dh.record("G_no_tet_file_for_a_shell_only_key", not leaked,
              {"leaked": leaked, "shell_only": shell_only})

    # H: the SHELL-only half stays honest as keys are added.
    stray = solid_group.material_maps.add()
    stray.parameter = "bend"
    stray.source_type = "VERTEX_GROUP"
    stray.source_name = "Grad"
    stray.target_value = 50.0
    refused = ""
    try:
        dh.encode_payload()
    except Exception as exc:
        refused = str(exc)
    solid_group.material_maps.remove(len(solid_group.material_maps) - 1)
    dh.record(
        "H_solid_refuses_a_shell_only_key",
        "not available as a map" in refused and "SOLID" in refused,
        {"message": refused[:300]},
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
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
