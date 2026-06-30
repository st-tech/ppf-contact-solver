#!/usr/bin/env python3
# File: cantilever_bend.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Cantilever bending test for the bundled fabric presets (ASTM D1388 geometry).
#
# A flat fabric strip is clamped at one end and overhangs horizontally; under
# gravity the free end droops. The tip droop angle (the angle below horizontal
# of the chord from the clamp edge to the settled tip) is a direct, visual
# measure of bending stiffness relative to weight: a stiff strip (denim) barely
# droops (small angle), a limp one (silk) folds down (large angle). Like the
# Cusick drape test, this measures BENDING, not in-plane stiffness.
#
# It runs on the same frontend session API and the same wire-parameter mapping
# as the drape harness (it imports them from cusick_drape), so a strip uses
# exactly the calibrated fabric parameters. CUDA only (session.start needs a GPU).
#
# Usage (CUDA host, from worktree root):
#   PYTHONPATH=. python calibration/cantilever_bend/cantilever_bend.py --all
#   PYTHONPATH=. python calibration/cantilever_bend/cantilever_bend.py --fabric Denim

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

# Reuse the drape harness's preset loading and wire-param mapping.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cusick_drape"))
import cusick_drape as cd  # noqa: E402

# Strip geometry (meters). ASTM D1388 uses a ~2.5 cm wide strip.
#
# The clamp holds the LEFT HALF of the strip and the right half is free. A large
# fixed base is what makes the test well-behaved: with a thin clamp the bending
# concentrates at the clamp edge and snaps erratically (the droop ordering comes
# out scrambled), whereas clamping half the strip lets the free half bend over a
# distributed region from a stable root, so the tip droop angle tracks bending
# stiffness MONOTONICALLY (stiff denim droops least, limp silk/flag most).
#
# Length 0.09 with CLAMP_FRAC 0.5 gives a 4.5 cm free overhang, which sits just
# past the droop knee and gives the widest clean monotonic angle spread
# (~69 deg stiffest to ~87 deg limpest). Much shorter leaves every fabric flat;
# much longer saturates them all near 90 deg.
STRIP_LENGTH = 0.09
STRIP_WIDTH = 0.025
STRIP_RES = 60          # subdivisions along the length
CLAMP_FRAC = 0.5        # fix the left half, free the right half

DEFAULT_FRAMES = 300
DEFAULT_DT = 0.01


def simulate_bend(name: str, preset: dict, *, length: float = STRIP_LENGTH,
                  width: float = STRIP_WIDTH, res: int = STRIP_RES,
                  clamp_frac: float = CLAMP_FRAC, frames: int = DEFAULT_FRAMES,
                  dt: float = DEFAULT_DT) -> dict:
    """Cantilever a strip with the fabric's parameters; return settled mesh +
    the tip droop angle and supporting geometry."""
    from frontend import App

    app = App.create(f"bend-{name.lower()}")
    mesh = app.mesh.rectangle(res, length, width, [1, 0, 0], [0, 0, 1])
    V = np.asarray(mesh[0], dtype=np.float64)

    xmin, xmax = float(V[:, 0].min()), float(V[:, 0].max())
    span = xmax - xmin
    clamp_x = xmin + clamp_frac * span
    # Pin the left half by ASSET-order indices (the frontend maps them to the
    # solver's own ordering, which is what makes the clamp hold).
    clamp = np.nonzero(V[:, 0] <= clamp_x + 1e-9)[0].tolist()

    app.asset.add.tri("strip", mesh[0], mesh[1])
    scene = app.scene.create()
    obj = scene.add("strip").at(0, 0, 0)
    obj.pin(clamp)
    for key, value in cd.wire_params_from_preset(preset).items():
        obj.param.set(key, value)
    scene = scene.build()

    # The built scene reorders vertices (pinned vertices are grouped), so all
    # post-hoc geometry must be identified in the SOLVER OUTPUT order, not the
    # asset order. scene.vertex(True) gives the rest positions and scene._tri
    # the faces in exactly the order session.get.vertex() returns -- pairing
    # asset-order indices with the settled buffer scrambles the result.
    rest = np.asarray(scene.vertex(True), dtype=np.float64)
    F = np.asarray(scene._tri)
    xo = rest[:, 0]
    xmin_o, xmax_o = float(xo.min()), float(xo.max())
    span_o = xmax_o - xmin_o
    clamp_x = xmin_o + clamp_frac * span_o
    tip_rest = np.nonzero(xo >= xmax_o - 0.02 * span_o)[0]

    session = app.session.create(scene)
    session.param.set("frames", frames).set("dt", dt)
    session.param.set("fps", round(1.0 / dt))
    session.param.set("gravity", [0.0, -9.8, 0.0])
    session = session.build()
    session.start(blocking=True)

    got = session.get.vertex()
    if got is None:
        raise RuntimeError(f"{name}: no settled vertex data returned")
    settled = np.asarray(got[0], dtype=np.float64)

    tip = settled[tip_rest].mean(axis=0)
    run = tip[0] - clamp_x          # horizontal distance clamp-edge -> tip
    drop = -tip[1]                  # vertical droop (tip Y is negative)
    overhang = xmax_o - clamp_x     # rest overhang length
    angle = math.degrees(math.atan2(drop, run))
    xmin, xmax = xmin_o, xmax_o
    # Centerline (the single Z-row nearest the midline, in output order): a clean
    # droop profile without the width-curl clutter. Ordered by rest X.
    zvals = np.unique(np.round(rest[:, 2], 6))
    z0 = zvals[int(np.argmin(np.abs(zvals)))]
    center = np.nonzero(np.abs(rest[:, 2] - z0) < 1e-6)[0]
    center = center[np.argsort(rest[center, 0])]
    return {
        "vert": settled, "face": F, "rest": rest,
        "angle_deg": angle, "drop": drop, "run": run, "overhang": overhang,
        "clamp_x": clamp_x, "tip": tip, "xmin": xmin, "xmax": xmax,
        "centerline": center.tolist(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cantilever bending test for the "
                                 "bundled fabric presets.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fabric")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--length", type=float, default=STRIP_LENGTH,
                    help="Strip length in meters (overhang ~ length*(1-clamp)).")
    ap.add_argument("--clamp", type=float, default=CLAMP_FRAC,
                    help="Fraction of the length held by the clamp "
                    "(0.5 = fix the left half, free the right half).")
    ap.add_argument("--res", type=int, default=STRIP_RES,
                    help="Subdivisions along the strip length (resolution sweep: "
                    "the droop angle should be invariant to this).")
    ap.add_argument("--bend", type=float, default=None,
                    help="Override the preset bend (probe; requires --fabric).")
    ap.add_argument("--dump", default=None,
                    help="Save the settled strip mesh to this .npz "
                    "(requires --fabric) for offline inspection.")
    args = ap.parse_args(argv)

    if args.bend is not None and not args.fabric:
        print("--bend is a probe; use with --fabric", file=sys.stderr)
        return 2
    if args.dump and not args.fabric:
        print("--dump requires --fabric", file=sys.stderr)
        return 2

    presets = cd.load_presets()
    names = cd.shell_fabrics(presets) if args.all else [args.fabric]
    print(f"{'fabric':10s} {'droop angle':>12s}  {'overhang':>9s}  {'bend':>6s}")
    print("-" * 46)
    for name in names:
        if name not in presets:
            print(f"unknown preset: {name}", file=sys.stderr)
            return 2
        preset = dict(presets[name])
        if args.bend is not None:
            preset["bend"] = args.bend
        r = simulate_bend(name, preset, length=args.length, res=args.res,
                          clamp_frac=args.clamp, frames=args.frames, dt=args.dt)
        print(f"{name:10s} {r['angle_deg']:10.1f} deg  "
              f"{r['overhang'] * 100:7.1f}cm  {preset['bend']:6.3f}  res={args.res}")
        if args.dump:
            np.savez(args.dump, vert=r["vert"], face=r["face"], rest=r["rest"],
                     clamp_x=r["clamp_x"], angle=r["angle_deg"],
                     xmin=r["xmin"], xmax=r["xmax"],
                     centerline=np.asarray(r["centerline"]))
            print(f"dumped {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
