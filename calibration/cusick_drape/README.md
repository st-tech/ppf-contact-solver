# Cusick drape-test harness

This directory holds the Tier-2 validation for the bundled **fabric** material
presets (`blender_addon/presets/materials.toml`). It reproduces the standard
Cusick fabric-drape experiment in the solver and measures each SHELL preset's
**Drape Coefficient (DC)**, so the shipped `bend` and `young-mod` numbers are
tied to published textile data rather than chosen by eye.

The sections below document the full methodology and its primary-source
grounding.

## The Cusick drape test

In the physical test (Cusick 1965, 1968; standardized as BS 5058 / ISO 9073-9),
a circular fabric specimen drapes under its own weight over a smaller horizontal
support disc, and the vertically projected (shadow) area of the draped specimen
is measured. The Drape Coefficient is

```
DC% = (A_proj - A_disc) / (A_specimen - A_disc) * 100
```

- `A_specimen` = area of the flat specimen disc (radius 15 cm here, the 30 cm
  template).
- `A_disc` = area of the support disc (radius 9 cm here, the 18 cm support).
- `A_proj` = projected shadow area of the draped specimen.

`DC = 0` is perfectly limp (the skirt hangs straight down, projecting to the
support disc); `DC = 100` is perfectly stiff (the specimen stays flat,
projecting to the full specimen disc). Higher DC means stiffer, less drape.

## How the harness models it

`cusick_drape.py`:

1. Builds a flat circular cloth specimen (radius 15 cm) as a concentric-ring
   triangulated disc in the horizontal XZ plane (the solver's default gravity is
   `(0, -9.8, 0)`, so up is +Y, matching every cloth example in `examples/`).
2. Models the support exactly as the test does: every specimen vertex within the
   9 cm support radius is **pinned flat and horizontal** (the "table top"); the
   overhang is free and drapes down. This needs no collider and is
   unconditionally stable, so the measured DC depends only on the fabric
   parameters.
3. Applies a fabric preset read from the SAME `materials.toml` the addon ships
   (so the harness validates exactly what users get), mapping the preset's addon
   keys to the solver's wire keys the same way `core/encoder/params.py` does
   (with `young_mod_density_normalized = true`, the stored `young-mod` is sent
   unchanged; `strain-limit` is the percentage / 100).
4. Simulates under gravity to quasistatic rest.
5. Measures `A_proj` by rasterizing the settled mesh's XZ projection onto a fine
   grid (folds and overlaps count once, exactly like a shadow), then evaluates
   DC. `A_specimen` and `A_disc` are rasterized on the same grid so the
   half-cell edge bias cancels in the ratio.

The specimen really does **buckle into folds**, exactly like a physical
drape-meter: as the overhang drops it must shed circumferential length (hoop
compression), and an inextensible sheet relieves that by buckling out of plane.
So the drape coefficient above is measured on the folded state, which is the
realistic one. The fold (node) count is an emergent, slightly run-variable
signal (the buckling is seeded by the solver's floating-point asymmetry) and is
reported only qualitatively, not calibrated; see the top-level
`calibration/README.md` and `calibration/report.py` for the rendered fold views.
To inspect a single settled drape mesh offline, use `dump_mesh.py`.

## Running it

This MUST run on a real CUDA GPU host. The emulated backend (macOS / any
non-CUDA host) has NO real physics, so the DC it reports is meaningless and is
only a plumbing smoke test.

On a CUDA host, from the worktree root, with the real CUDA backend built
(`cargo build --release`, default features) and the project Python environment:

```sh
# every fabric (writes results.csv)
PYTHONPATH=. python calibration/cusick_drape/cusick_drape.py --all

# a single fabric (prints only)
PYTHONPATH=. python calibration/cusick_drape/cusick_drape.py --fabric Silk
```

Useful flags: `--frames` (default 250), `--dt` (0.01), `--rings` / `--seg`
(specimen tessellation), `--grid` (rasterization resolution), `--no-write`.

## Files

- `cusick_drape.py` - the harness (specimen build, drape sim, DC measurement,
  fold counting, CLI).
- `dump_mesh.py` - run one fabric and save the settled mesh to an `.npz` for
  offline rendering/inspection (e.g. on a non-GPU host).
- `targets.csv` - published Drape Coefficient bands per fabric, with the source
  for each row.
- `results.csv` - the latest measured DCs from a real-CUDA run, committed for
  transparency so the "matches published drape" claim is checkable without
  re-running. The placeholder in the repo is replaced by the GPU run.

## Calibration loop

For each fabric, run the harness, then adjust `bend` (primary; drape is
bending-dominated per Feng 2022) and secondarily `young-mod` in
`materials.toml` until the simulated DC falls inside the published band in
`targets.csv`. Commit the updated preset numbers and `results.csv` together.

## Sources

- Cusick. "The Dependence of Fabric Drape on Bending and Shear Stiffness."
  J. Textile Inst. 56(11), 1965. doi:10.1080/19447026508662319.
- Cusick. "The Measurement of Fabric Drape." J. Textile Inst. 59(6), 1968.
  doi:10.1080/00405006808659985. BS 5058:1974; ISO 9073-9:2008.
- Feng, Huang, Xu, Wang. "Learning-Based Bending Stiffness Parameter Estimation
  by a Drape Tester." ACM TOG 41(6), 2022. doi:10.1145/3550454.3555464.
- Gan, Ly, Steven. "Drape Prediction by Means of Finite-element Analysis."
  J. Textile Inst. 82(1), 1991. doi:10.1080/00405009108658741. (cotton DC 68.4%)
- Memon et al. "Study on Effect of Leather Rigidity and Thickness on
  Drapability." Materials 14(16):4553, 2021. PMC8398511. (leather DC 47-70%)
- Matsudaira & Yang. "Features of Conventional Static and New Dynamic Drape
  Coefficients of Woven Silk Fabrics." Textile Research J. 73(3), 2003.
  doi:10.1177/004051750307300309.
