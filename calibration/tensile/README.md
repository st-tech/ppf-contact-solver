# Tensile / Poisson calibration (membrane stiffness)

The Cusick drape test (`../cusick_drape/`) is bending-dominated, so it cannot pin
the in-plane membrane stiffness (`young-mod`) or the Poisson ratio (`poiss-rat`).
This harness calibrates those: a fabric that stretches more easily (Silk, Wool)
gets a lower `young-mod` than a stiff one (Denim, Leather), grounded in collected
reference data (`targets.csv`).

## The bilinear membrane model (important)

The solver membrane is **bilinear**: linear with slope `young-mod` up to the
per-fabric **strain limit**, then near-inextensible. This approximates real
fabric's nonlinear stiffening (low initial modulus from yarn-crimp removal, then
a steep lock-up). Consequences for calibration:

- `young-mod` is the **initial (small-strain) modulus**. It must be measured
  **below the strain limit**; past the limit the strain reports the cap, not the
  modulus. The stretch test reports the max local strain so the load can be kept
  in the elastic regime.
- The **strain limit** is the separate "max stretch" knob (set from extensibility
  data, in `materials.toml`). It is not changed here.

## Tests

- `tensile.py --test stretch`: a vertical strip is pinned along its top edge and
  hangs under amplified gravity. `young-mod` is density-normalized, so the strain
  scales as `g / young-mod` (independent of density): more compliant fabric
  stretches more. `elong%` is the elongation of the strip's FREE span, `maxloc%`
  the strain of its topmost free segment, which carries the weight of everything
  below it and is therefore the largest local strain; `maxloc%` is what the
  elastic-regime check compares against the strain limit. Measured:

  | Fabric  | young-mod | elong% | maxloc% | strain limit |
  | ------- | --------: | -----: | ------: | -----------: |
  | Silk    |       500 |   2.05 |    3.27 |         6.0% |
  | Wool    |      2000 |   1.24 |    2.13 |         8.0% |
  | Flag    |      1000 |   1.19 |    1.94 |         4.0% |
  | Cotton  |      5500 |   0.68 |    1.20 |         5.0% |
  | Denim   |     10000 |   0.46 |    0.81 |         3.0% |
  | Leather |     13000 |   0.24 |    0.44 |         2.0% |

  Every fabric stays below its own strain limit, so all six are read in the
  elastic regime, and the ordering is monotonic in `young-mod` with ONE
  exception: Wool and Flag come out swapped relative to the reference (Wool is
  the stiffer of the two by `young-mod` yet stretches marginally more, 1.24
  against 1.19). The pair is separated by 0.05 points, it reproduces, and it is
  the one place this test does not confirm the reference order.

- `tensile.py --test poisson`: a strip pinned at both ends is stretched a small
  amount; the mid-span lateral contraction would give nu = -lateral/axial.
  **CAVEAT:** under in-plane stretch a thin membrane's free lateral edges go into
  transverse compression and **wrinkle out of plane**, so the projected width is
  not a clean Poisson read (measured nu came out erratic, 0.0-0.33, for a uniform
  set poiss-rat of 0.40). This is the same reason the textile literature measures
  fabric Poisson with DIC, not calipers. Because `poiss-rat` *is* the constitutive
  Poisson ratio by construction, it is set directly from the literature
  (`targets.csv`), and this test is kept only as a qualitative diagnostic.

## Reference data (`targets.csv`)

Collected from the paper corpus + textile-engineering web sources (relative
membrane stiffness normalized to Denim = 1.0; smaller stretches more):

| Fabric  | rel. stiffness | young-mod (wire) | Poisson |
| ------- | -------------: | ---------------: | ------: |
| Silk    | 0.05           | 500              | 0.40    |
| Flag    | 0.10           | 1000             | 0.40    |
| Wool    | 0.20           | 2000             | 0.40    |
| Cotton  | 0.55           | 5500             | 0.35    |
| Denim   | 1.00           | 10000            | 0.25    |
| Leather | 1.30           | 13000            | 0.40    |

Sources: Wang, O'Brien, Ramamoorthi 2011 (denim stiff vs knits/wool compliant);
Miguel 2012 (nonlinear, hysteretic tensile); Zhang 2019 (cotton Ex/Ey=178/98 MPa,
Poisson 0.39/0.25); arXiv 2212.08790 (membrane E 0.8-32.6 MPa band); Hursa 2009 /
Penava 2014 (woven Poisson 0.2-0.5, plain > twill); FAST/SiroFAST (wool
extensibility); Efendioglu 2025 / ISO 3376 (leather). Full citations are in the
per-fabric notes of `targets.csv`.

## Running (CUDA host)

```sh
PYTHONPATH=. python calibration/tensile/tensile.py --test stretch --all
PYTHONPATH=. python calibration/tensile/tensile.py --test poisson --all
# probe a single fabric / value:
PYTHONPATH=. python calibration/tensile/tensile.py --test stretch --fabric Silk --young-mod 800
```

`young-mod` shifts the Cusick drape coefficient (a stiffer membrane drapes less),
so after changing it the shell `bend` is re-tuned on the drape test to keep the
drape coefficient in band. The two harnesses are therefore calibrated together.
