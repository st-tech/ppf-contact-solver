#!/usr/bin/env python3
# File: cusick_drape.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Cusick drape-test harness for the bundled fabric presets.
#
# This is the Tier-2 validation step (see this directory's README.md):
# it reproduces the standard
# Cusick fabric-drape experiment (Cusick 1965/1968; BS 5058 / ISO 9073-9) in
# the solver and measures the Drape Coefficient (DC) of each SHELL preset, so
# the shipped fabric `bend` / `young-mod` numbers are tied to published textile
# data rather than chosen by eye.
#
# Method
# ------
# A flat circular cloth specimen (radius R_SPECIMEN = 15 cm, i.e. the 30 cm
# Cusick template) is supported over its central disc of radius R_SUPPORT
# (9 cm, the 18 cm Cusick support) and allowed to drape under gravity. We model
# the support exactly as the test does: the specimen's central region (every
# vertex within R_SUPPORT of the center) is pinned flat and horizontal, holding
# up the "table top"; the overhang (R_SUPPORT < r <= R_SPECIMEN) is free and
# drapes down. No collider is needed and the configuration is unconditionally
# stable, so the measured DC depends only on the fabric parameters.
#
# After the specimen settles to quasistatic rest we measure the vertically
# projected (shadow) area A_proj of the draped mesh by rasterizing its
# horizontal projection onto a fine grid (folds and overlaps count once, which
# is what a shadow does). The Drape Coefficient is
#
#     DC% = (A_proj - A_disc) / (A_specimen - A_disc) * 100
#
# DC = 0   -> perfectly limp (the skirt hangs straight down, projecting to the
#             support disc),
# DC = 100 -> perfectly stiff (the specimen stays flat, projecting to the full
#             specimen disc).
# A_specimen and A_disc are rasterized on the SAME grid as A_proj so the
# half-cell edge bias cancels in the ratio.
#
# Coordinate convention: the solver's default gravity is (0, -9.8, 0), so "up"
# is +Y. The specimen is built horizontal in the XZ plane and the shadow is its
# projection onto XZ (the Y coordinate is dropped). This matches every cloth
# example in examples/ (e.g. bench_drape.py: sheets in XZ, sphere below in -Y).
#
# IMPORTANT: the emulated (macOS) backend has NO real physics, so the DC it
# reports is meaningless; it is useful only as a plumbing smoke test. Run this
# on a real CUDA host (see calibration/cusick_drape/README.md) for the numbers
# that calibrate the presets.
#
# Usage (from the worktree root, with the project venv):
#     PYTHONPATH=. python calibration/cusick_drape/cusick_drape.py --all
#     PYTHONPATH=. python calibration/cusick_drape/cusick_drape.py --fabric Silk

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tomllib

import numpy as np

# Cusick geometry, in meters (standard template / support).
R_SPECIMEN = 0.15  # 30 cm circular specimen
R_SUPPORT = 0.09   # 18 cm support disc (its footprint is pinned flat)

# Default specimen tessellation (concentric rings). RINGS is a multiple of 5 so
# that R_SUPPORT/R_SPECIMEN = 0.6 lands a ring exactly on R_SUPPORT (ring 18 of
# 30 = 0.6 * 0.15 = 0.09 m): the pinned/free boundary then coincides with the
# A_disc reference radius, removing a systematic DC offset.
DEFAULT_RINGS = 30
DEFAULT_SEG = 96

# Default simulation length. The harness pins fps = 1/dt (see run_drape) so one
# frame is exactly one dt step and total simulated time = frames * dt. 350
# frames at dt = 0.01 is 3.5 s, comfortably past quasistatic rest for a 15 cm
# specimen.
DEFAULT_FRAMES = 350
DEFAULT_DT = 0.01

# Rasterization grid for the projected-area measurement.
DEFAULT_GRID = 1024

_HERE = os.path.dirname(os.path.abspath(__file__))
_PRESETS_TOML = os.path.normpath(
    os.path.join(_HERE, "..", "..", "blender_addon", "presets", "materials.toml")
)
_TARGETS_CSV = os.path.join(_HERE, "targets.csv")
_RESULTS_CSV = os.path.join(_HERE, "results.csv")


# ---------------------------------------------------------------------------
# Preset loading (reads the SAME file the addon ships)
# ---------------------------------------------------------------------------

def load_presets() -> dict:
    with open(_PRESETS_TOML, "rb") as f:
        return tomllib.load(f)


def shell_fabrics(presets: dict) -> list[str]:
    """Preset names whose object_type is SHELL, in file order."""
    return [n for n, p in presets.items() if p.get("object_type") == "SHELL"]


def wire_params_from_preset(preset: dict) -> dict:
    """Map a SHELL preset's addon keys to the solver's wire param keys.

    Mirrors blender_addon/core/encoder/params.py: with
    young_mod_density_normalized == True the stored young-mod is already the
    solver's density-normalized value and is sent unchanged; otherwise it is
    divided by density. strain-limit is the percentage / 100.
    """
    if preset.get("object_type") != "SHELL":
        raise ValueError("Cusick drape harness only handles SHELL (fabric) presets")
    density = float(preset["shell_density"])
    young = float(preset["shell_young_modulus"])
    if not preset.get("young_mod_density_normalized", True):
        young = young / density
    strain = (
        float(preset.get("strain_limit_percent", 0.0)) / 100.0
        if preset.get("enable_strain_limit", False)
        else 0.0
    )
    model = "baraff-witkin" if preset.get("shell_model") == "BARAFF_WITKIN" else "arap"
    return {
        "model": model,
        "density": density,
        "young-mod": young,
        "poiss-rat": float(preset.get("shell_poisson_ratio", 0.3)),
        "bend": float(preset["bend"]),
        "friction": float(preset.get("friction", 0.0)),
        "strain-limit": strain,
    }


# ---------------------------------------------------------------------------
# Specimen geometry (concentric-ring disc in the horizontal XZ plane)
# ---------------------------------------------------------------------------

def make_disc(radius: float, rings: int, seg: int) -> tuple[np.ndarray, np.ndarray]:
    """Concentric-ring triangulated disc in the XZ plane (y = 0).

    Returns (V, F): V is (N, 3) float64, F is (M, 3) int32. Vertex 0 is the
    center; vertices are ordered ring by ring outward, which makes selecting
    the central pinned region a simple radius test.
    """
    verts = [(0.0, 0.0, 0.0)]
    for k in range(1, rings + 1):
        r = radius * k / rings
        for j in range(seg):
            a = 2.0 * math.pi * j / seg
            verts.append((r * math.cos(a), 0.0, r * math.sin(a)))
    V = np.asarray(verts, dtype=np.float64)

    faces = []
    # Center fan to the first ring.
    for j in range(seg):
        a = 1 + j
        b = 1 + (j + 1) % seg
        faces.append((0, a, b))
    # Quad strips between consecutive rings (two triangles each).
    for k in range(1, rings):
        base0 = 1 + (k - 1) * seg
        base1 = 1 + k * seg
        for j in range(seg):
            jn = (j + 1) % seg
            a = base0 + j
            b = base0 + jn
            c = base1 + j
            d = base1 + jn
            faces.append((a, c, d))
            faces.append((a, d, b))
    F = np.asarray(faces, dtype=np.int32)
    return V, F


def central_indices(V: np.ndarray, support_radius: float) -> list[int]:
    """Indices of vertices whose XZ radius is within the support disc."""
    r = np.sqrt(V[:, 0] ** 2 + V[:, 2] ** 2)
    return np.nonzero(r <= support_radius + 1e-9)[0].tolist()


def count_folds(vert: np.ndarray, rings: int, seg: int,
                mode_lo: int = 3, mode_hi: int | None = None) -> int:
    """Dominant azimuthal fold (node) count of the draped specimen.

    The overhang sheds circumferential length as it drops (hoop compression),
    which an inextensible sheet relieves by buckling out of plane into folds.
    We read the count as the dominant angular frequency of the outer rim's
    height profile (FFT over a sane mode band). It rises as the fabric gets
    limper, so it is a qualitative bending signal, NOT a calibrated metric: the
    buckling is seeded by solver asymmetry, so the exact count is somewhat
    run-variable and tends to come out finer than a physical drape-meter.
    """
    base = 1 + (rings - 1) * seg
    rim = vert[base:base + seg]
    y = rim[:, 1] - rim[:, 1].mean()
    spec = np.abs(np.fft.rfft(y))
    hi = mode_hi if mode_hi is not None else seg // 4
    band = spec[mode_lo:hi + 1]
    if len(band) == 0:
        return 0
    return int(np.argmax(band) + mode_lo)


# ---------------------------------------------------------------------------
# Projected (shadow) area via rasterization onto the XZ plane
# ---------------------------------------------------------------------------

def projected_area(P: np.ndarray, F: np.ndarray, half_extent: float,
                   grid: int) -> float:
    """Rasterize the 2D triangle soup (P, F) and return the covered area.

    P is (N, 2) projected coordinates; F is (M, 3). A square grid of side
    ``grid`` spans [-half_extent, half_extent]^2. A cell counts once even if
    many (folded) triangles cover it, exactly modeling a vertical shadow.
    """
    lo, hi = -half_extent, half_extent
    cell = (hi - lo) / grid
    cov = np.zeros((grid, grid), dtype=bool)
    gx = (P[:, 0] - lo) / cell
    gy = (P[:, 1] - lo) / cell
    for i0, i1, i2 in F:
        x0, y0 = gx[i0], gy[i0]
        x1, y1 = gx[i1], gy[i1]
        x2, y2 = gx[i2], gy[i2]
        minx = max(int(math.floor(min(x0, x1, x2))), 0)
        maxx = min(int(math.ceil(max(x0, x1, x2))), grid - 1)
        miny = max(int(math.floor(min(y0, y1, y2))), 0)
        maxy = min(int(math.ceil(max(y0, y1, y2))), grid - 1)
        if maxx < minx or maxy < miny:
            continue
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if d == 0.0:
            continue
        xs = np.arange(minx, maxx + 1) + 0.5
        ys = np.arange(miny, maxy + 1) + 0.5
        px, py = np.meshgrid(xs, ys)
        a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / d
        b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / d
        c = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if inside.any():
            cov[miny:maxy + 1, minx:maxx + 1] |= inside
    return float(cov.sum()) * cell * cell


# ---------------------------------------------------------------------------
# Drape simulation
# ---------------------------------------------------------------------------

def run_drape(name: str, preset: dict, *, frames: int, dt: float,
              rings: int, seg: int) -> np.ndarray:
    """Simulate the draped specimen and return its settled (N, 3) vertices."""
    from frontend import App

    V, F = make_disc(R_SPECIMEN, rings, seg)
    pinned = central_indices(V, R_SUPPORT)

    app = App.create(f"cusick-{name.lower()}")
    app.asset.add.tri("specimen", V, F)

    scene = app.scene.create()
    obj = scene.add("specimen").at(0, 0, 0)
    obj.pin(pinned)
    wp = wire_params_from_preset(preset)
    for key, value in wp.items():
        obj.param.set(key, value)

    scene = scene.build()

    session = app.session.create(scene)
    # Pin fps = 1/dt so one output frame is exactly one dt step; then the run
    # loop's frame budget makes total simulated time = frames * dt. Without
    # this the session keeps its default fps (60) and total time would be
    # frames/60, decoupled from dt.
    session.param.set("frames", frames).set("dt", dt)
    session.param.set("fps", round(1.0 / dt))
    session.param.set("gravity", [0.0, -9.8, 0.0])
    session = session.build()
    session.start(blocking=True)

    got = session.get.vertex()
    if got is None:
        raise RuntimeError(f"{name}: no settled vertex data returned")
    vert, _frame = got
    return np.asarray(vert, dtype=np.float64)


def simulate_drape(name: str, preset: dict, *, frames: int, dt: float,
                   rings: int, seg: int, grid: int) -> dict:
    """Run the drape and measure DC; return the settled mesh + metrics.

    Returns a dict with the settled vertices/faces (for rendering) and the area
    terms, so both the calibration CLI (drape_coefficient) and the report
    renderer share one physics path.
    """
    V0, F = make_disc(R_SPECIMEN, rings, seg)
    Vd, Fd = make_disc(R_SUPPORT, max(4, rings // 2), seg)

    # Margin past R_SPECIMEN so a strain-limit-stretched rim vertex (up to a
    # few percent for the limpest fabrics) cannot fall outside the grid.
    half = R_SPECIMEN * 1.12
    a_specimen = projected_area(V0[:, [0, 2]], F, half, grid)
    a_disc = projected_area(Vd[:, [0, 2]], Fd, half, grid)

    settled = run_drape(name, preset, frames=frames, dt=dt, rings=rings, seg=seg)
    a_proj = projected_area(settled[:, [0, 2]], F, half, grid)

    denom = a_specimen - a_disc
    dc = 100.0 * (a_proj - a_disc) / denom if denom > 0 else float("nan")
    return {
        "vert": settled, "face": F,
        "dc": dc, "a_proj": a_proj,
        "a_specimen": a_specimen, "a_disc": a_disc,
        "fold_k": count_folds(settled, rings, seg),
        "rings": rings, "seg": seg,
    }


def drape_coefficient(name: str, preset: dict, *, frames: int, dt: float,
                      rings: int, seg: int, grid: int) -> dict:
    """Run the drape, measure DC, and return a result row."""
    sim = simulate_drape(name, preset, frames=frames, dt=dt,
                         rings=rings, seg=seg, grid=grid)
    return {
        "fabric": name,
        "dc_percent": round(sim["dc"], 1),
        "a_proj_cm2": round(sim["a_proj"] * 1e4, 2),
        "a_specimen_cm2": round(sim["a_specimen"] * 1e4, 2),
        "a_disc_cm2": round(sim["a_disc"] * 1e4, 2),
        "bend": preset["bend"],
        "young_mod": preset["shell_young_modulus"],
        "density": preset["shell_density"],
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Targets + reporting
# ---------------------------------------------------------------------------

def load_targets() -> dict:
    """fabric -> (dc_min, dc_max) from targets.csv (citations are columns)."""
    out = {}
    if not os.path.exists(_TARGETS_CSV):
        return out
    with open(_TARGETS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["fabric"]] = (float(row["dc_min"]), float(row["dc_max"]))
            except (KeyError, ValueError):
                continue
    return out


def write_results(rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys()) + ["target_min", "target_max", "in_band"]
    targets = load_targets()
    with open(_RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            lo, hi = targets.get(r["fabric"], (None, None))
            out = dict(r)
            out["target_min"] = lo if lo is not None else ""
            out["target_max"] = hi if hi is not None else ""
            out["in_band"] = (
                "" if lo is None else bool(lo <= r["dc_percent"] <= hi)
            )
            w.writerow(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cusick drape-test harness for the "
                                 "bundled fabric presets.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fabric", help="Run a single SHELL preset by name (e.g. Silk).")
    g.add_argument("--all", action="store_true", help="Run every SHELL preset.")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--rings", type=int, default=DEFAULT_RINGS)
    ap.add_argument("--seg", type=int, default=DEFAULT_SEG)
    ap.add_argument("--grid", type=int, default=DEFAULT_GRID)
    ap.add_argument("--bend", type=float, default=None,
                    help="Override the preset's bend (calibration probe; "
                    "requires --fabric).")
    ap.add_argument("--young-mod", dest="young_mod", type=float, default=None,
                    help="Override the preset's shell_young_modulus "
                    "(calibration probe; requires --fabric).")
    ap.add_argument("--density", type=float, default=None,
                    help="Override the preset's shell_density "
                    "(probe; requires --fabric).")
    ap.add_argument("--strain-limit", dest="strain_limit", type=float,
                    default=None, help="Override strain_limit_percent "
                    "(probe; requires --fabric). 0 disables strain limiting.")
    ap.add_argument("--no-write", action="store_true",
                    help="Do not update results.csv (print only).")
    args = ap.parse_args(argv)

    probes = (args.bend, args.young_mod, args.density, args.strain_limit)
    if any(p is not None for p in probes) and not args.fabric:
        print("--bend / --young-mod / --density / --strain-limit are probes; "
              "use with --fabric", file=sys.stderr)
        return 2

    # The DC zero-point (A_disc) is rasterized at exactly R_SUPPORT. If no
    # tessellation ring lands there, the pinned/free boundary sits off the
    # reference radius and DC carries a small systematic offset.
    ring_at_support = (R_SUPPORT / R_SPECIMEN) * args.rings
    if abs(ring_at_support - round(ring_at_support)) > 1e-6:
        print(f"warning: --rings {args.rings} places no ring exactly on "
              f"R_SUPPORT ({R_SUPPORT} m); DC will carry a small systematic "
              f"offset. Pick --rings so {R_SUPPORT}/{R_SPECIMEN} * rings is "
              f"an integer (a multiple of 5 for the default geometry).",
              file=sys.stderr)

    presets = load_presets()
    if args.all:
        names = shell_fabrics(presets)
    else:
        if args.fabric not in presets:
            print(f"unknown preset: {args.fabric}", file=sys.stderr)
            return 2
        names = [args.fabric]

    targets = load_targets()
    rows = []
    print(f"{'fabric':10s} {'DC%':>6s}  {'target':>10s}  {'band?':>5s}")
    print("-" * 40)
    for name in names:
        preset = dict(presets[name])  # copy so probe overrides do not mutate the cache
        if args.bend is not None:
            preset["bend"] = args.bend
        if args.young_mod is not None:
            preset["shell_young_modulus"] = args.young_mod
        if args.density is not None:
            preset["shell_density"] = args.density
        if args.strain_limit is not None:
            preset["strain_limit_percent"] = args.strain_limit
            preset["enable_strain_limit"] = args.strain_limit > 0
        row = drape_coefficient(
            name, preset, frames=args.frames, dt=args.dt,
            rings=args.rings, seg=args.seg, grid=args.grid,
        )
        rows.append(row)
        lo, hi = targets.get(name, (None, None))
        band = "" if lo is None else ("PASS" if lo <= row["dc_percent"] <= hi else "FAIL")
        tstr = "" if lo is None else f"{lo:.0f}-{hi:.0f}"
        print(f"{name:10s} {row['dc_percent']:6.1f}  {tstr:>10s}  {band:>5s}")

    if not args.no_write and args.all:
        write_results(rows)
        print(f"\nwrote {_RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
