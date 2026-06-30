#!/usr/bin/env python3
# File: tensile.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Tensile / stretch harness for the bundled fabric presets: calibrates the
# in-plane membrane stiffness (young-mod) and Poisson ratio (poiss-rat), which
# the Cusick drape test cannot pin (drape is bending-dominated). Two tests:
#
#  - STRETCH (young-mod): a vertical strip is pinned along its top edge and hangs
#    under amplified gravity; the total elongation is measured. The solver's
#    young-mod is density-normalized, so the strain scales as g/young-mod
#    (independent of density): a more compliant fabric (lower young-mod) stretches
#    more. The elongation ranking calibrates the relative young-mod across
#    fabrics against the collected reference (targets.csv): Silk/Wool stretch more
#    than Denim/Leather.
#
#    IMPORTANT (bilinear model): the membrane is linear (slope = young-mod) only
#    up to the per-fabric STRAIN LIMIT, then it is near-inextensible -- a bilinear
#    approximation of real fabric's nonlinear stiffening. young-mod is the INITIAL
#    (small-strain) modulus, so it must be measured BELOW the strain limit; past
#    the limit the elongation reports the cap, not the modulus. The harness
#    therefore also reports the max local strain (the top segment carries the
#    whole strip's weight) so the load can be kept in the elastic regime. The
#    strain limit is the separate "max stretch" knob and is not changed here.
#
#  - POISSON (poiss-rat): a horizontal strip is pinned at both ends and stretched
#    a small amount along its length (below the strain limit, elastic regime);
#    the lateral (width) contraction at mid-span gives the measured Poisson ratio
#    nu = -lateral_strain / axial_strain, used to set poiss-rat.
#
# Like the other calibration tests it uses the frontend session API and runs
# ONLY on a real CUDA host. The built scene reorders vertices, so all geometry
# is read in solver output order via scene.vertex(True) / scene._tri (see
# cantilever_bend.py for the same pattern).
#
# Usage (CUDA host, from worktree root):
#   PYTHONPATH=. python calibration/tensile/tensile.py --test stretch --all
#   PYTHONPATH=. python calibration/tensile/tensile.py --test poisson --all
#   PYTHONPATH=. python calibration/tensile/tensile.py --test stretch --fabric Silk --young-mod 500

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cusick_drape"))
import cusick_drape as cd  # noqa: E402

# Stretch test geometry + load (meters; gravity amplified to get a measurable,
# still-elastic elongation, ~1-5% for the compliant fabrics).
STRETCH_LENGTH = 0.20
STRETCH_WIDTH = 0.03
STRETCH_RES = 80
STRETCH_G = 60.0

# Poisson test geometry: a long strip so lateral contraction develops at mid-span
# (Saint-Venant) away from the clamped ends.
POISSON_LENGTH = 0.20
POISSON_WIDTH = 0.04
POISSON_RES = 80
POISSON_CLAMP_FRAC = 0.06
POISSON_STRETCH = 0.015   # axial engineering strain applied (below strain limit)

DEFAULT_FRAMES = 250
DEFAULT_DT = 0.01


def _apply_params(obj, preset, young_mod=None, poisson=None):
    wp = cd.wire_params_from_preset(preset)
    if young_mod is not None:
        wp["young-mod"] = young_mod
    if poisson is not None:
        wp["poiss-rat"] = poisson
    for key, value in wp.items():
        obj.param.set(key, value)


def simulate_stretch(name: str, preset: dict, *, young_mod=None,
                     length=STRETCH_LENGTH, width=STRETCH_WIDTH, res=STRETCH_RES,
                     g=STRETCH_G, frames=DEFAULT_FRAMES, dt=DEFAULT_DT) -> dict:
    """Hang a vertical strip (top edge pinned) under gravity g; return the total
    axial elongation (settled length / rest length - 1)."""
    from frontend import App

    app = App.create(f"stretch-{name.lower()}")
    # length along Y (vertical), width along X.
    mesh = app.mesh.rectangle(res, length, width, [0, 1, 0], [1, 0, 0])
    V = np.asarray(mesh[0], dtype=np.float64)
    ymax = float(V[:, 1].max())
    span = ymax - float(V[:, 1].min())
    top = np.nonzero(V[:, 1] >= ymax - 0.02 * span)[0].tolist()

    app.asset.add.tri("strip", mesh[0], mesh[1])
    scene = app.scene.create()
    obj = scene.add("strip").at(0, 0, 0)
    obj.pin(top)
    _apply_params(obj, preset, young_mod=young_mod)
    scene = scene.build()

    rest = np.asarray(scene.vertex(True), dtype=np.float64)
    yo = rest[:, 1]
    ytop, ybot = float(yo.max()), float(yo.min())
    L0 = ytop - ybot
    bot = np.nonzero(yo <= ybot + 0.02 * L0)[0]
    # Top two mesh rows (by rest Y) -> the top segment carries the whole strip's
    # weight, so its strain is the max local strain (checked vs the strain limit).
    levels = np.unique(np.round(yo, 6))
    top_row = np.nonzero(np.abs(yo - levels[-1]) < 1e-6)[0]
    row2 = np.nonzero(np.abs(yo - levels[-2]) < 1e-6)[0]
    seg0 = levels[-1] - levels[-2]

    session = app.session.create(scene)
    session.param.set("frames", frames).set("dt", dt)
    session.param.set("fps", round(1.0 / dt))
    session.param.set("gravity", [0.0, -g, 0.0])
    session = session.build()
    session.start(blocking=True)

    settled = np.asarray(session.get.vertex()[0], dtype=np.float64)
    bot_y = float(settled[bot, 1].mean())
    elong = (ytop - bot_y) / L0 - 1.0   # top pinned, bottom drops
    seg1 = float(settled[top_row, 1].mean() - settled[row2, 1].mean())
    max_local_strain = seg1 / seg0 - 1.0
    sl = preset.get("strain_limit_percent", 0.0) if preset.get(
        "enable_strain_limit", False) else None
    return {"elongation_pct": 100.0 * elong,
            "max_local_strain_pct": 100.0 * max_local_strain,
            "strain_limit_pct": sl, "g": g,
            "young_mod": young_mod if young_mod is not None
            else preset["shell_young_modulus"]}


def simulate_poisson(name: str, preset: dict, *, poisson=None, young_mod=None,
                     length=POISSON_LENGTH, width=POISSON_WIDTH,
                     res=POISSON_RES, clamp_frac=POISSON_CLAMP_FRAC,
                     stretch=POISSON_STRETCH, frames=DEFAULT_FRAMES,
                     dt=DEFAULT_DT) -> dict:
    """Stretch a strip uniaxially (both ends pinned, one end moved) by a small
    elastic strain; return the measured Poisson ratio from mid-span lateral
    (width) contraction."""
    from frontend import App

    app = App.create(f"poisson-{name.lower()}")
    # length along X, width along Y.
    mesh = app.mesh.rectangle(res, length, width, [1, 0, 0], [0, 1, 0])
    V = np.asarray(mesh[0], dtype=np.float64)
    xmin, xmax = float(V[:, 0].min()), float(V[:, 0].max())
    span = xmax - xmin
    cx0 = xmin + clamp_frac * span
    cx1 = xmax - clamp_frac * span
    left = np.nonzero(V[:, 0] <= cx0)[0].tolist()
    right = np.nonzero(V[:, 0] >= cx1)[0].tolist()
    gauge = cx1 - cx0
    dx = stretch * gauge

    app.asset.add.tri("strip", mesh[0], mesh[1])
    scene = app.scene.create()
    obj = scene.add("strip").at(0, 0, 0)
    obj.pin(left)
    obj.pin(right).move_by([dx, 0.0, 0.0], 0.0, 1.0)
    _apply_params(obj, preset, young_mod=young_mod, poisson=poisson)
    scene = scene.build()

    rest = np.asarray(scene.vertex(True), dtype=np.float64)
    xo = rest[:, 0]
    midx = 0.5 * (float(xo.min()) + float(xo.max()))
    band = np.nonzero(np.abs(xo - midx) <= 0.10 * gauge)[0]
    w0 = float(rest[band, 1].max() - rest[band, 1].min())

    session = app.session.create(scene)
    session.param.set("frames", frames).set("dt", dt)
    session.param.set("fps", round(1.0 / dt))
    session.param.set("gravity", [0.0, 0.0, 0.0])
    session = session.build()
    session.start(blocking=True)

    settled = np.asarray(session.get.vertex()[0], dtype=np.float64)
    w1 = float(settled[band, 1].max() - settled[band, 1].min())
    lateral = (w1 - w0) / w0
    axial = dx / gauge
    nu = -lateral / axial if axial != 0 else float("nan")
    return {"poisson_measured": nu, "axial_pct": 100.0 * axial,
            "lateral_pct": 100.0 * lateral,
            "poiss_rat": poisson if poisson is not None
            else preset["shell_poisson_ratio"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tensile/Poisson calibration harness "
                                 "for the bundled fabric presets.")
    ap.add_argument("--test", choices=["stretch", "poisson"], required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fabric")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--g", type=float, default=STRETCH_G, help="stretch-test gravity")
    ap.add_argument("--young-mod", dest="young_mod", type=float, default=None,
                    help="override young-mod (probe; requires --fabric)")
    ap.add_argument("--poisson", type=float, default=None,
                    help="override poiss-rat (probe; requires --fabric)")
    args = ap.parse_args(argv)
    if (args.young_mod is not None or args.poisson is not None) and not args.fabric:
        print("--young-mod / --poisson are probes; use with --fabric", file=sys.stderr)
        return 2

    presets = cd.load_presets()
    names = cd.shell_fabrics(presets) if args.all else [args.fabric]
    if args.test == "stretch":
        print(f"{'fabric':10s} {'elong%':>7s} {'maxloc%':>8s} {'limit%':>7s} "
              f"{'young-mod':>9s} {'regime':>8s}  (g={args.g})")
        print("-" * 60)
        for name in names:
            if name not in presets:
                print(f"unknown preset: {name}", file=sys.stderr); return 2
            r = simulate_stretch(name, presets[name], young_mod=args.young_mod,
                                 g=args.g, frames=args.frames, dt=args.dt)
            sl = r["strain_limit_pct"]
            # elastic only if the max local strain stays under the strain limit
            regime = ("elastic" if sl is None or r["max_local_strain_pct"] < sl
                      else "CAPPED")
            slt = f"{sl:.1f}" if sl is not None else "-"
            print(f"{name:10s} {r['elongation_pct']:6.2f} "
                  f"{r['max_local_strain_pct']:7.2f} {slt:>7s} "
                  f"{r['young_mod']:9.0f} {regime:>8s}")
    else:
        print(f"{'fabric':10s} {'nu_meas':>8s}  {'poiss_rat':>9s}  {'axial%':>7s}")
        print("-" * 42)
        for name in names:
            if name not in presets:
                print(f"unknown preset: {name}", file=sys.stderr); return 2
            r = simulate_poisson(name, presets[name], poisson=args.poisson,
                                 young_mod=args.young_mod, frames=args.frames,
                                 dt=args.dt)
            print(f"{name:10s} {r['poisson_measured']:8.3f}  {r['poiss_rat']:9.2f}"
                  f"  {r['axial_pct']:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
