# Fabric preset calibration suite

Reproducible, physically-grounded validation for the bundled **fabric** (SHELL)
material presets in `blender_addon/presets/materials.toml`. Standard textile
tests, plus a report (Markdown + HTML) that renders the results so they can be
reviewed at a glance.

Everything here runs on a **real CUDA GPU host** (the underlying solver session
needs a GPU; the emulated backend has no physics). Run from the worktree root
with `PYTHONPATH=.` and the project Python environment.

## What is measured, and what it means

Two independent fabric properties are calibrated:

- **Bending** (drape + cantilever tests): a fabric barely stretches under its
  own weight, so its drooped shape is governed by bending vs gravity. This sets
  `bend`, validated against published Cusick drape coefficients. See Cusick 1965
  ("dependence of fabric drape on bending and shear stiffness") and Feng et al.
  2022 ("in-plane stiffness ... do not matter to drape simulation").
- **In-plane stretch** (tensile test): how easily the fabric stretches when
  pulled in its plane. This sets `young-mod` and the Poisson ratio, calibrated
  against measured fabric tensile data so Silk and Wool stretch more easily than
  Denim and Leather.

The two are coupled in this solver: a stiffer membrane also drapes a little less,
so `young-mod` and `bend` are calibrated together to keep the drape coefficient
on target.

`bend` is **resolution-independent**: the solver uses the convergent Discrete
Shells per-hinge stiffness (proportional to edge-length squared over triangle
area; Grinspun et al. 2003), so the same `bend` gives the same drape at any mesh
density. These calibrated values therefore hold across resolutions; the drape
coefficient is stable when the specimen is re-tessellated (verified across the
ring counts in `cusick_drape/`). An earlier formulation scaled with edge length
alone, which weakened bending under refinement (finer cloth drooped more); that
is fixed.

## Components

- `cusick_drape/` - the **Cusick drape test**. A 30 cm circular specimen drapes
  over an 18 cm support disc; the **drape coefficient** is the projected
  (shadow) area ratio, calibrated against published values per fabric. The
  overhang sheds circumferential length as it drops (hoop compression), so the
  specimen **buckles into folds**, exactly like a physical drape-meter. See
  `cusick_drape/README.md`.
- `cantilever_bend/` - the **cantilever bending test** (ASTM D1388 geometry). A
  fabric strip is clamped at one end and droops under gravity; the tip droop
  angle is a direct, visual bending measure. See `cantilever_bend/README.md`.
- `tensile/` - the **tensile / stretch test**. Calibrates the in-plane membrane
  stiffness (`young-mod`) and Poisson ratio, which the drape test cannot pin. A
  hanging strip measures stretch compliance (Silk/Wool stretch more than
  Denim/Leather); reference data and the bilinear-model caveat are in
  `tensile/README.md`.
- `report.py` - runs both tests for every fabric and writes `report.md` and
  `report.html` plus a `report_images/` subdirectory of PNGs (each
  palette-quantized under 32 KiB). Both link the same image files (no base64):
  `report.md` renders natively on GitHub once committed; `report.html` opens in
  any web browser. Each shows top-down + oblique drape views (drape coefficient +
  observed fold count) and side-on cantilever views (droop angle), plus a summary
  table. It also caches the sim meshes (see below) so the report can be re-styled
  with `--render-only` without a GPU.
- `viz.py` - shared software renderer (numpy orthographic z-buffer + flat
  shading) and PIL label/overlay helpers. Pure software, no GPU/display.

## Running

```sh
# the full visual report (all fabrics) -> report.md + report.html + report_images/
# (CUDA host) also caches the sim meshes to calibration/report_data/
PYTHONPATH=. python calibration/report.py

# re-render the report from the cache after a style/layout change, WITHOUT
# re-running the sims (no GPU needed; works anywhere the cache was fetched to)
PYTHONPATH=. python calibration/report.py --render-only

# just the numeric drape coefficients (calibration loop)
PYTHONPATH=. python calibration/cusick_drape/cusick_drape.py --all

# just the bending angles
PYTHONPATH=. python calibration/cantilever_bend/cantilever_bend.py --all

# the in-plane stretch ranking (young-mod) and Poisson diagnostic
PYTHONPATH=. python calibration/tensile/tensile.py --test stretch --all --g 100
PYTHONPATH=. python calibration/tensile/tensile.py --test poisson --all

# dump one settled drape mesh for offline inspection
PYTHONPATH=. python calibration/cusick_drape/dump_mesh.py --fabric Silk --out /tmp/silk.npz
```

## Notes on the fold count

The drape fold (node) count is an **emergent** quantity: the buckling is seeded
by the solver's floating-point asymmetry, so the exact count is somewhat
run-variable and tends to come out finer than a physical drape-meter (which
typically shows ~3-8 nodes). It is reported as a **qualitative bending signal**
(limper fabric -> more, sharper folds) for visual assessment, not as a
calibrated metric. The calibrated metric is the drape coefficient (area).
