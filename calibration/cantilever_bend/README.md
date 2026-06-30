# Cantilever bending test

A cantilever fabric-bending test (ASTM D1388 geometry) for the bundled fabric
(SHELL) presets, as a second, more isolated check of the bending calibration
alongside the Cusick drape test.

## Method

A flat fabric strip (9 cm long, 2.5 cm wide) has its **left half clamped** and
its **right half free**; under gravity the free half droops. The harness
measures the **tip droop angle**: the angle below horizontal of the chord from
the clamp edge to the settled tip. A stiff strip (denim) droops less; a limp one
(silk/flag) droops more.

Clamping half the strip is deliberate. With a thin clamp the bending
concentrates at the clamp edge and snaps erratically, so the droop ordering
comes out scrambled (the limpest fabric can end up drooping less than a stiffer
one). Clamping a large, stable base lets the free half bend over a distributed
region, and the tip droop angle then tracks bending stiffness **monotonically**.
The 4.5 cm free overhang sits just past the droop knee, where the monotonic
angle spread is widest (~69 deg stiffest to ~87 deg limpest); a much shorter
overhang leaves every fabric flat and a much longer one saturates them all near
90 deg.

Like the drape test, this measures **bending stiffness relative to weight**, not
in-plane / tensile stiffness: the standard cantilever test is precisely how
textile bending length and flexural rigidity are obtained.

The strip uses the exact same fabric wire parameters as the drape harness (it
imports `wire_params_from_preset` from `../cusick_drape/cusick_drape.py`), so the
droop reflects the same calibrated `bend` that the drape coefficient was tuned
on. The cantilever isolates bending from the drape geometry: weight pulls a
simple beam down, with no hoop compression or self-contact, so the angle is a
cleaner read on `bend` than the drape coefficient.

## Running

```sh
# all fabrics (CUDA host, from the worktree root)
PYTHONPATH=. python calibration/cantilever_bend/cantilever_bend.py --all

# a single fabric, with an optional bend override for probing
PYTHONPATH=. python calibration/cantilever_bend/cantilever_bend.py --fabric Denim
PYTHONPATH=. python calibration/cantilever_bend/cantilever_bend.py --fabric Silk --bend 0.05

# resolution sweep (--res subdivides the strip length)
PYTHONPATH=. python calibration/cantilever_bend/cantilever_bend.py --fabric Cotton --res 30
```

`--res` subdivides the strip along its length. With the resolution-independent
bending formula the droop is stable across well-shaped meshes (res 30 vs 60 agree
closely). Very high `--res` at the fixed strip width makes sliver triangles, whose
convergent `|edge|^2/area` bending coefficient correctly spikes, so the strip goes
locally rigid and stops drooping; that is a property of this single-axis
refinement, not of the formula. For a clean resolution check prefer the drape
test, whose concentric-ring mesh stays well shaped as it refines.

The side-on images with the droop angle drawn are produced by
`calibration/report.py` (see the top-level `calibration/README.md`).

## Reference

- ASTM D1388, Standard Test Method for Stiffness of Fabrics (cantilever and
  heart-loop options).
- Cusick. "The Dependence of Fabric Drape on Bending and Shear Stiffness."
  J. Textile Inst. 56(11), 1965.
