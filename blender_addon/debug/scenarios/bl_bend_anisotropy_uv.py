# File: scenarios/bl_bend_anisotropy_uv.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Gates UV-driven anisotropic (orthotropic) shell bending end to end on a
# no-GPU host, through the emulator's Projective-Dynamics bending term
# (crates/ppf-cts-solver/src/cpp_emul/pd_arap.hpp).
#
# `bend-warp` / `bend-weft` add directional stiffness on top of the isotropic
# `bend`, so how stiff a hinge is depends on where its shared edge sits in the
# UV frame. The measurement has to separate that from every other reason a sheet
# might droop differently, so the scene holds SEVEN cantilever strips that are
# identical in geometry, pins and material, and differ only in the UV
# orientation they carry, whether their group has anisotropy on, and their mesh
# resolution:
#
#   AnisoAlign / AnisoRot          directional ON, UV 90 degrees apart, coarse
#   AnisoAlignFine / AnisoRotFine  the same pair, refined
#   IsoAlign / IsoRot              directional OFF (both 0), same orientations
#   IsoAlignFine                   the isotropic refinement control
#
# Each is pinned along its +y edge, so it folds about hinges whose shared edges
# run along x. With UV aligned (u -> x) those edges sit at psi = 0 and take the
# WEFT stiffness; rotated (u -> y) the same edges sit at psi = 90 degrees and
# take the warp stiffness. A weft ratio well below 1 therefore has to make the
# aligned strip droop measurably FARTHER than the rotated one.
#
# Checks:
#   A. all_pc2_present         - every strip produced every output frame.
#   B. pinned_edges_held       - the pinned edge of each strip stays put.
#   C. aniso_uv_changes_droop  - under anisotropy the two UV orientations
#                                droop differently, by a wide margin.
#   D. aniso_soft_axis_correct - and the SOFT (weft) direction is the one that
#                                droops farther, so the effect has the sign the
#                                orthotropic formula predicts rather than just
#                                being some difference.
#   E. iso_uv_is_direction_free - with both directional values at 0 the same UV
#                                orientations droop identically. This is the
#                                regression half: it proves the isotropic path
#                                ignores UV entirely, so existing scenes are
#                                unaffected by the feature.
#   G. resolution_independent  - the same sheet at double the subdivision
#                                settles the same way, under BOTH materials.
#   H. anisotropy_survives_refinement - and the aligned-vs-rotated contrast is
#                                the same size on the refined pair, so the
#                                direction dependence itself does not drift
#                                with the mesh.
#   F. simulation_stable       - all positions finite and bounded.
#
# C and E together are the core: a change that made bending direction-blind
# would fail C, and one that leaked anisotropy into the default path would fail
# E. G and H then pin the coefficient, which is the half that a dimensionally
# wrong per-hinge weight silently breaks (weighting by area instead of its
# reciprocal makes bending VANISH under refinement while every other check
# still passes).

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# The emulator applies only kinematic pins unless its implicit elastic step is
# switched on, and that step is where the bending term lives.
KNOBS = {"PPF_EMULATED_ELASTIC": "1", "PPF_EMULATED_STEP_MS": "0"}


_DRIVER_BODY = r"""
import os
import math
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 24

# Warp far above weft, so the two UV orientations are far apart. These are
# absolute stiffnesses added on top of BEND_STIFFNESS, in its units.
WARP_STIFFNESS = 150.0
WEFT_STIFFNESS = 0.0
# The isotropic base is what the SOFT fold direction gets (weft is 0 here), so
# it has to sit where bending genuinely competes with gravity: far above it
# every strip behaves as a rigid plate, far below it as a rag, and in both
# limits the fold direction stops mattering, so the contrast this scenario
# measures would vanish for reasons unrelated to the code under test. The warp
# value then puts the stiff direction well clear of it.
BEND_STIFFNESS = 8.0
# The resolution pair. Halving the spacing changes hinge count, edge length and
# per-hinge area all at once, which is exactly what the |e|^2/area coefficient
# (CUDA) and the 1/(A1+A2) weight (emulator) exist to cancel.
COARSE_SUBDIV = 10
FINE_SUBDIV = 20


# A grid pinned along its +y edge, with an explicit UV map. `rotate_uv` turns
# the UV layout 90 degrees. The UV triangles stay congruent (it is a rotation,
# not a reshape), so nothing about the membrane changes; only which material
# direction each hinge edge points along does.
def _make_strip(name, x_offset, rotate_uv, subdiv=10):
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=subdiv, y_subdivisions=subdiv, size=2,
        location=(x_offset, 0, 0),
    )
    obj = bpy.context.object
    obj.name = name

    mesh = obj.data
    # Write into the ACTIVE layer, which is what the encoder reads
    # (core/numpy_mesh_utils.py uses mesh.uv_layers.active). primitive_grid_add
    # already supplies one; calling uv_layers.new() here would add a SECOND
    # layer that the encoder never looks at, leaving both strips on the grid's
    # default UV and the rotation silently ineffective.
    uv = mesh.uv_layers.active
    if uv is None:
        uv = mesh.uv_layers.new(name="UVMap")
    mesh.uv_layers.active = uv
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        if rotate_uv:
            # 90 degree rotation of the aligned layout.
            u, v = co.y, -co.x
        else:
            u, v = co.x, co.y
        # Map [-1, 1] to [0, 1]; a uniform affine map, identical for both, so
        # it cannot itself introduce a difference between the two strips.
        uv.data[loop.index].uv = (0.5 * (u + 1.0), 0.5 * (v + 1.0))
    mesh.update()

    pinned = [i for i, v in enumerate(mesh.vertices) if v.co.y > 0.99]
    free = [i for i, v in enumerate(mesh.vertices) if v.co.y <= 0.99]
    # The far edge, opposite the pin. Bending stiffness shows up most where
    # the lever arm is longest, so the tip separates the two fold directions
    # far more sharply than an average over the whole sheet does.
    tip = [i for i, v in enumerate(mesh.vertices) if v.co.y < -0.99]
    vg = obj.vertex_groups.new(name="TopEdge")
    vg.add(pinned, 1.0, "REPLACE")
    readback = tuple(round(c, 4) for c in mesh.uv_layers.active.data[0].uv)
    return obj, pinned, free, tip, len(mesh.uv_layers), uv.name, readback


def _raw_group(dh, name):
    for g in dh.groups.iterate_active_object_groups(bpy.context.scene):
        if g.name == name:
            return g
    return None


# Mean downward (-z) travel of the free vertices, as a positive number.
def _mean_droop(arr, free_idx):
    free = np.array(free_idx, dtype=int)
    return float(-np.mean((arr[-1][free] - arr[0][free])[:, 2]))


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    strips = {}
    # name, x offset, rotate UV, group, subdivisions. The *Fine entries are the
    # same 2x2 sheet, same UV layout and same material at double the
    # resolution, so comparing each against its coarse twin measures resolution
    # independence directly rather than by proxy.
    layout = [
        ("AnisoAlign", -10.0, False, "Aniso", COARSE_SUBDIV),
        ("AnisoRot", -6.0, True, "Aniso", COARSE_SUBDIV),
        ("AnisoAlignFine", -2.0, False, "Aniso", FINE_SUBDIV),
        ("AnisoRotFine", 2.0, True, "Aniso", FINE_SUBDIV),
        ("IsoAlign", 6.0, False, "Iso", COARSE_SUBDIV),
        ("IsoRot", 10.0, True, "Iso", COARSE_SUBDIV),
        ("IsoAlignFine", 14.0, False, "Iso", FINE_SUBDIV),
    ]
    for name, x_off, rot, _grp, subdiv in layout:
        obj, pinned, free, tip, n_uv, uv_name, uv0 = _make_strip(
            name, x_off, rot, subdiv
        )
        strips[name] = {"obj": obj, "pinned": pinned, "free": free, "tip": tip}
        dh.log(
            f"strip {name} rot_uv={rot} uv_layers={n_uv} "
            f"active={uv_name} uv[0]={uv0}"
        )

    # The encoder reads each object's DEPSGRAPH-EVALUATED mesh, which still
    # holds the UV the grid primitive shipped with until the graph is flushed.
    # Without this the rotated layouts stay invisible to the solver and every
    # strip is exported with the same UV, which reads as anisotropy not
    # working rather than as a stale evaluated mesh.
    bpy.context.view_layer.update()

    dh.save_blend(PROBE_DIR, "bend_anisotropy_uv.blend")
    root = dh.configure_state(
        project_name="bend_anisotropy_uv",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )

    for group_name in ("Aniso", "Iso"):
        grp_api = dh.api.solver.create_group(group_name, "SHELL")
        for name, _x, _rot, owner, _sub in layout:
            if owner == group_name:
                grp_api.add(name)
                grp_api.create_pin(name, "TopEdge")
        raw = _raw_group(dh, group_name)
        if raw is None:
            raise RuntimeError(f"group {group_name} not found after create")
        # Same bending stiffness on both groups; only the ratios differ, so
        # the isotropic pair is the exact control for the anisotropic pair.
        raw.bend = BEND_STIFFNESS
        if group_name == "Aniso":
            raw.bend_warp = WARP_STIFFNESS
            raw.bend_weft = WEFT_STIFFNESS
        else:
            raw.bend_warp = 0.0
            raw.bend_weft = 0.0
        # Report the RNA state that was actually set. A new PropertyGroup field
        # only registers on a full Blender start, and an unregistered one still
        # ACCEPTS assignment (it becomes a stray ID-property) while the encoder
        # keeps reading the default, so without this line the scenario would
        # fail as "the directional stiffness does nothing" with no hint that
        # the addon tree Blender loaded is not the one under test.
        dh.log(
            f"group {group_name}: "
            f"rna={'bend_warp' in raw.bl_rna.properties.keys()} "
            f"warp={raw.bend_warp:.4f} weft={raw.bend_weft:.4f} "
            f"bend={raw.bend}"
        )
    dh.log("groups_configured")

    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    data_bytes, param_bytes, _pre_data_hash, _pre_param_hash = (
        encoder_pkg.prepare_upload(bpy.context)
    )

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, "aniso-bend:build", timeout=180.0)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    saw_running = dh.run_and_wait(timeout=180.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=90.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"ran saw_running={saw_running} fetch={applied}/{total}")

    # --- gather per-strip PC2 ---
    frames_ok = True
    for name, info in strips.items():
        pc2_path = dh.find_pc2_for(info["obj"])
        if not pc2_path or not os.path.isfile(pc2_path):
            raise RuntimeError(f"no PC2 for {name} (path={pc2_path!r})")
        info["arr"] = dh.read_pc2(pc2_path)
        if info["arr"].shape[0] != FRAME_COUNT:
            frames_ok = False
        dh.log(f"pc2 {name} shape={info['arr'].shape}")

    dh.record(
        "A_all_pc2_present",
        frames_ok,
        {n: int(i["arr"].shape[0]) for n, i in strips.items()},
    )

    # B: every pinned edge holds.
    pin_disps = {}
    for name, info in strips.items():
        pinned = np.array(info["pinned"], dtype=int)
        arr = info["arr"]
        pin_disps[name] = round(float(
            np.max(np.linalg.norm(arr[-1][pinned] - arr[0][pinned], axis=1))
        ), 5)
    dh.record(
        "B_pinned_edges_held",
        all(d < 0.05 for d in pin_disps.values()),
        pin_disps,
    )

    # Mean droop over all free vertices, NOT the tip. A strip that is very soft
    # in its fold direction rolls up near the pin and tucks its far edge under,
    # so the tip's net vertical travel runs OPPOSITE to the sheet's overall
    # sag and reverses the comparison. The mean is the honest aggregate, and it
    # is also the one that survives a resolution change, where the two meshes
    # have different vertex counts and no tip vertex corresponds to another.
    droop = {n: _mean_droop(i["arr"], i["free"]) for n, i in strips.items()}
    tip_droop = {n: _mean_droop(i["arr"], i["tip"]) for n, i in strips.items()}
    dh.log(f"mean_droop={ {k: round(v, 5) for k, v in droop.items()} }")
    dh.log(f"tip_droop={ {k: round(v, 5) for k, v in tip_droop.items()} }")

    # C: under anisotropy the two UV orientations must part company. Scale the
    # gap by the larger droop so the criterion is relative, not a length that
    # silently depends on how far this particular cloth happens to fall.
    a_align, a_rot = droop["AnisoAlign"], droop["AnisoRot"]
    denom = max(abs(a_align), abs(a_rot), 1e-9)
    aniso_gap = abs(a_align - a_rot) / denom
    dh.record(
        "C_aniso_uv_changes_droop",
        aniso_gap > 0.04,
        {"aligned": round(a_align, 5), "rotated": round(a_rot, 5),
         "relative_gap": round(aniso_gap, 5), "threshold": 0.04},
    )

    # D: and the sign is the one the formula predicts. The aligned strip's fold
    # edges run along warp, so they pick up the WEFT stiffness, which is the
    # smaller of the two here: it must be the floppier of the two strips.
    dh.record(
        "D_aniso_soft_axis_correct",
        a_align > a_rot,
        {"aligned_droop": round(a_align, 5), "rotated_droop": round(a_rot, 5),
         "expected": "aligned droops farther (weft below warp)"},
    )

    # E: the isotropic control must not care about UV at all. Same geometry,
    # same material, same pins, only the UV layout differs, so any difference
    # here is directional stiffness leaking into the default path.
    i_align, i_rot = droop["IsoAlign"], droop["IsoRot"]
    i_denom = max(abs(i_align), abs(i_rot), 1e-9)
    iso_gap = abs(i_align - i_rot) / i_denom
    dh.record(
        "E_iso_uv_is_direction_free",
        iso_gap < 0.01,
        {"aligned": round(i_align, 5), "rotated": round(i_rot, 5),
         "relative_gap": round(iso_gap, 6), "threshold": 0.01},
    )

    # G: resolution independence. The same sheet at double the subdivision must
    # settle to the same drape. Bending is the term at risk here: the stencil
    # reports a second difference that shrinks as h^2 while the hinge count
    # grows as 1/h^2, so a per-hinge weight that does not carry the matching
    # 1/area factor makes bending drift with the mesh (weighting by area
    # instead makes it vanish outright). Checked under BOTH materials, because
    # the direction factor multiplies that coefficient and must not disturb it.
    res_gaps = {}
    for label, coarse, fine in (
        ("aniso", "AnisoAlign", "AnisoAlignFine"),
        ("iso", "IsoAlign", "IsoAlignFine"),
    ):
        c_val, f_val = droop[coarse], droop[fine]
        res_gaps[label] = round(
            abs(c_val - f_val) / max(abs(c_val), abs(f_val), 1e-9), 5
        )
    dh.record(
        "G_resolution_independent",
        all(g < 0.15 for g in res_gaps.values()),
        {"relative_gap": res_gaps, "threshold": 0.15,
         "coarse_subdiv": COARSE_SUBDIV, "fine_subdiv": FINE_SUBDIV,
         "droop": {k: round(droop[k], 5) for k in
                   ("AnisoAlign", "AnisoAlignFine", "IsoAlign", "IsoAlignFine")}},
    )

    # H: the anisotropy itself is resolution independent, not merely the
    # overall droop. The aligned-vs-rotated contrast is a property of the
    # MATERIAL, so measuring it again on the refined pair must give the same
    # answer: same sign, comparable size. A coefficient that drifted with the
    # mesh would still pass G (both strips drift together) while quietly
    # strengthening or washing out the direction dependence, which is the
    # failure this check exists to catch.
    f_align, f_rot = droop["AnisoAlignFine"], droop["AnisoRotFine"]
    fine_gap = abs(f_align - f_rot) / max(abs(f_align), abs(f_rot), 1e-9)
    same_sign = (f_align > f_rot) == (a_align > a_rot)
    gap_ratio = fine_gap / aniso_gap if aniso_gap > 0 else 0.0
    dh.record(
        "H_anisotropy_survives_refinement",
        fine_gap > 0.04 and same_sign and 0.5 < gap_ratio < 2.0,
        {"coarse_gap": round(aniso_gap, 5), "fine_gap": round(fine_gap, 5),
         "gap_ratio": round(gap_ratio, 4), "same_sign": same_sign,
         "fine_aligned": round(f_align, 5), "fine_rotated": round(f_rot, 5)},
    )

    # F: stability across every strip and frame.
    all_arr = np.concatenate([i["arr"].reshape(-1, 3) for i in strips.values()])
    finite = bool(np.all(np.isfinite(all_arr)))
    max_abs = float(np.max(np.abs(all_arr)))
    dh.record(
        "F_simulation_stable",
        finite and max_abs < 100.0,
        {"finite": finite, "max_abs": round(max_abs, 4)},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = REPO_ROOT_POSIX
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
