# File: scenarios/bl_fluid_bake_retry.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The rig's own Mantaflow bake helper, `bake_fluid_domain` in
# `_driver_lib.FLUID_BAKE_LIB`, which every scenario that bakes a gas domain
# goes through. Blender's bake writes a value that is not finite about once
# in thirty bakes, and the helper then bakes again; that retry must not call
# `bpy.ops.fluid.free_all()`, because a bake that follows a free crashes
# Blender 5.1.2 on Windows. It resets the bake through
# Blender's own update instead, by changing Cache Type away and back. The
# natural failure is too rare to wait for, so the scenario makes it: the
# helper's `after_bake` hook writes a NaN into one voxel of frame 5's velocity
# on disk after a chosen attempt.
#
# Subtests:
#   A. a_bad_bake_is_baked_again_without_a_free: a NaN after the first bake
#      only. The helper records one retry naming frame 5's file and returns;
#      every data file of 1..11 is written again, nothing in it is not
#      finite, and the domain's evaluated velocity at frames 5 and 11 is the
#      first bake's to 1e-5 of its peak (measured: bit for bit on Blender
#      5.1.2, 5.2.0 and 5.2.1). A trial in which Blender's own bake wrote a
#      value that is not finite is recorded and run again on a fresh domain.
#   B. a_second_bad_bake_fails_loudly: a NaN after both bakes. The helper
#      raises, naming frame 5's file.
#   C. a_baked_domain_is_baked_whole: a domain baked, its inflow turned, and
#      the helper called again. Every frame's file is written again and frame
#      5's velocity changes; a bare bake_all over a bake would resume at its
#      last frame and write frame 11 alone (measured).
#   D. the_helper_never_frees: no code line of FLUID_BAKE_LIB calls
#      `fluid.free_all`, checked on the host and reported even when Blender
#      exited without a result, which is what a restored free does to the
#      Windows leg.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("real",)

_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    import openvdb

    dh = DriverHelpers(pkg, result)
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene.frame_end = 11
    BAD = "fluid_data_0005.vdb"

    def gas_domain(name, y):
        # A size-2 gas domain baked over 1..11 (Cache Type All, OpenVDB) as
        # the force-field scenarios build it, its inflow jet inside it.
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, y, 0.0))
        dom = bpy.context.object
        dom.name = name
        mod = dom.modifiers.new("Fluid", "FLUID")
        mod.fluid_type = "DOMAIN"
        ds = mod.domain_settings
        ds.domain_type = "GAS"
        ds.resolution_max = 16
        ds.alpha = ds.beta = 0.0
        ds.gravity = (0.0, 0.0, 0.0)
        ds.use_adaptive_timesteps = False
        ds.cache_type = "ALL"
        ds.cache_data_format = "OPENVDB"
        ds.cache_directory = os.path.join(PROBE_DIR, name.lower() + "_cache")
        ds.cache_frame_start, ds.cache_frame_end = 1, 11
        bpy.ops.mesh.primitive_cube_add(size=0.4, location=(-0.6, y, 0.0))
        jet = bpy.context.object
        jet.name = name + "Jet"
        fm = jet.modifiers.new("Fluid", "FLUID")
        fm.fluid_type = "FLOW"
        fl = fm.flow_settings
        fl.flow_type = "SMOKE"
        fl.flow_behavior = "INFLOW"
        fl.use_initial_velocity = True
        fl.velocity_coord = (1.5, 0.3, 0.0)
        return dom, jet, ds

    def data_dir(ds):
        return os.path.join(bpy.path.abspath(ds.cache_directory), "data")

    def velocity(dom, frame):
        # The evaluated domain's velocity at `frame`, entered from the frame
        # before it.
        scene.frame_set(frame - 1)
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        live = next(m for m in dom.evaluated_get(dg).modifiers
                    if m.type == "FLUID").domain_settings
        out = np.zeros(len(live.velocity_grid), dtype=np.float32)
        live.velocity_grid.foreach_get(out)
        return out

    def poison(path):
        # One voxel of the file's velocity grid set to NaN, the grids and the
        # file's metadata written back as they were.
        grids, meta = openvdb.readAll(path)
        vel = next(g for g in grids if "Vec3" in type(g).__name__)
        ijk = next(tuple(int(v) for v in it.min) for it in vel.citerOnValues()
                   if it.count == 1)
        vel.getAccessor().setValueOn(ijk, (float("nan"), 0.0, 0.0))
        openvdb.write(path, grids=grids, metadata=meta)

    def stamps(ds):
        d = data_dir(ds)
        return {n: os.stat(os.path.join(d, n)).st_mtime_ns for n in sorted(os.listdir(d))}

    # A: a NaN after the first bake only. Blender's own bake writes a value
    # that is not finite now and then, and a trial in which it did (in the
    # first bake, or in the retry, which nothing here poisons) says nothing
    # about the retry: it is recorded and A runs again on a fresh domain.
    def retry_trial(name, y):
        dom, jet, ds = gas_domain(name, y)
        first = {}

        def bad_first(d, attempt):
            if attempt == 0:
                first["natural"] = _baked_non_finite(d)
                first["v5"], first["v11"] = velocity(d, 5), velocity(d, 11)
                poison(os.path.join(data_dir(ds), BAD))
                first["scan"] = _baked_non_finite(d)
                first["stamps"] = stamps(ds)

        notes, error = [], None
        try:
            bake_fluid_domain(scene, dom, notes, after_bake=bad_first)
        except RuntimeError as e:
            error = str(e)
        if first.get("natural"):
            return None, {"first_bake_wrote": first["natural"]}
        if error is not None and "twice" in error:
            left = _baked_non_finite(dom)
            if left != {BAD: 1}:
                # Not the injected value surviving: the retry bake went bad.
                return None, {"retry_bake_wrote": left}
        detail = {"error": error, "notes": notes, "scan_after_poison": first.get("scan")}
        if error is not None or first.get("scan") != {BAD: 1}:
            return False, detail
        after = stamps(ds)
        v5, v11 = velocity(dom, 5), velocity(dom, 11)
        peak = float(max(np.abs(first["v5"]).max(), np.abs(first["v11"]).max()))
        detail.update(
            missing=_unbaked_frames(dom), scan=_baked_non_finite(dom),
            not_rewritten=sorted(n for n in first["stamps"] if after.get(n) == first["stamps"][n]),
            frame5_max_diff=float(np.abs(v5.astype(np.float64) - first["v5"]).max()),
            frame11_max_diff=float(np.abs(v11.astype(np.float64) - first["v11"]).max()),
            peak=peak)
        # The re-bake is the first bake again: measured bit for bit in every
        # run on Blender 5.1.2, 5.2.0 and 5.2.1. A bake that resumed instead
        # keeps the NaN and moves frame 11 by about 0.9 at a peak near 3.
        tol = 1e-5 * peak
        return (len(notes) == 1 and BAD in notes[0] and detail["missing"] == []
                and detail["scan"] == {} and detail["not_rewritten"] == []
                and detail["frame5_max_diff"] <= tol and detail["frame11_max_diff"] <= tol
                and peak > 0.1), detail

    trials = []
    for name, y in (("RetryOnce", 0.0), ("RetryOnceAgain", 12.0)):
        ok, detail = retry_trial(name, y)
        trials.append(detail)
        if ok is not None:
            break
    dh.record("A_a_bad_bake_is_baked_again_without_a_free", ok is True, {"trials": trials})

    # B: a NaN after both bakes.
    dom_b, jet_b, ds_b = gas_domain("RetryTwice", 4.0)

    def bad_always(d, attempt):
        poison(os.path.join(data_dir(ds_b), BAD))

    notes_b = []
    error = None
    try:
        bake_fluid_domain(scene, dom_b, notes_b, after_bake=bad_always)
    except RuntimeError as e:
        error = str(e)
    dh.record("B_a_second_bad_bake_fails_loudly",
              error is not None and "twice" in error and BAD in error and len(notes_b) == 1,
              {"error": error, "notes": notes_b})

    # C: a baked domain baked again after its inflow turned.
    dom_c, jet_c, ds_c = gas_domain("RebakeWhole", 8.0)
    bake_fluid_domain(scene, dom_c)
    before_v5 = velocity(dom_c, 5)
    before = stamps(ds_c)
    jet_c.modifiers["Fluid"].flow_settings.velocity_coord = (0.0, 1.5, 0.0)
    bake_fluid_domain(scene, dom_c)
    after = stamps(ds_c)
    after_v5 = velocity(dom_c, 5)
    rewritten = sorted(n for n in after if n in before and after[n] != before[n])
    dh.record("C_a_baked_domain_is_baked_whole",
              len(before) == 11 and rewritten == sorted(before)
              and not np.array_equal(after_v5, before_v5),
              {"files_before": len(before), "rewritten": len(rewritten),
               "frame5_changed": not np.array_equal(after_v5, before_v5)})
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def _frees() -> list[str]:
    # The code lines of the bake helper's library that call a free.
    return [line.strip() for line in dl.FLUID_BAKE_LIB.splitlines()
            if "free_all" in line and not line.strip().startswith("#")]


def run(ctx: r.ScenarioContext) -> dict:
    frees = _frees()
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        # A free in the helper crashes Blender 5.1.2 on Windows before any
        # check is written, so the cause is named here too.
        if frees:
            err.setdefault("violations", []).insert(
                0, f"D_the_helper_never_frees: the helper calls {frees}")
        return err
    checks = dict(result.get("checks", {}))
    # A driver that stopped early recorded fewer checks; each is required.
    for name in ("A_a_bad_bake_is_baked_again_without_a_free",
                 "B_a_second_bad_bake_fails_loudly", "C_a_baked_domain_is_baked_whole"):
        checks.setdefault(name, {"ok": False, "details": "not recorded"})
    checks["D_the_helper_never_frees"] = {"ok": not frees, "details": {"lines": frees}}
    verdict = r.report_named_checks(checks)
    if result.get("errors"):
        verdict["status"] = "fail"
        verdict.setdefault("violations", []).extend(result["errors"][:2])
    return verdict
