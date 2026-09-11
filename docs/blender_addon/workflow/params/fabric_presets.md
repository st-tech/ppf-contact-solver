# 🧵 Fabric Presets

The add-on includes a set of ready-to-use **fabric presets** for **Shell**
groups, each calibrated against real textile measurements so it drapes, bends,
and stretches like the named fabric:

- **Silk** and **Flag** (lightweight banner cloth): light and very drapey.
- **Cotton** and **Wool**: medium stiffness and drape.
- **Denim**: stiff and the least drapey of the set, so it holds its shape.
- **Leather**: heavy and barely stretchy, but it drapes with the medium
  fabrics and droops further than either of them.

## Solid presets

**Solid** groups get their own list in the same **Preset** dropdown — the menu
is filtered by the group's Type, so a Solid group never sees the fabrics and a
Shell group never sees these:

- **Rubber**: vulcanized rubber (real E ~3 MPa), near-incompressible and
  grippy (friction 0.85).
- **Silicone**: silicone elastomer / PDMS (~2 MPa), very soft and
  near-incompressible.
- **Foam**: flexible polyurethane foam (~50 kPa), light and springy.
- **Sponge**: dry cellulose sponge (~0.3 MPa), soft and porous.
- **Jelly**: gelatin / ballistic gel (~150 kPa), very soft, wobbly and
  slippery (friction 0.10).

Unlike the fabrics, these are not calibrated by simulation: each carries the
real material's Young's modulus, entered density-normalized (the real E
divided by the real density), which is the form the solver consumes. Their
**Density** ships at a uniform 100 kg/m³, deliberately — density only affects
inertia and contact, not the static deformed shape, so a uniform value avoids
the mass-ratio conditioning problems that real volumetric densities cause in
mixed-material scenes. Change it freely.

Applying a preset never changes the group's Type, and a row with its padlock
engaged keeps its current value.

## Calibration report

The report linked below shows, for every fabric, how it behaves in standard
fabric tests, rendered from simulations:

- a **drape** over a disc (how far it collapses into folds),
- a **cantilever bend** (how far a held strip droops under its own weight), and
- its **stretch** parameters (how easily it pulls in its plane: Silk and Wool
  stretch more easily than Denim and Leather).

:::{raw} html
<p><a class="reference external" href="../../../fabric-report/index.html"><strong>Open the fabric calibration report &raquo;</strong></a></p>
:::

:::{note}
The report opens as a standalone page. Each fabric is shown with a top-down and
an angled view of its drape, a side view of its cantilever bend, and a summary
table of its parameters. The drape values follow the standard Cusick drape test;
the bending follows the ASTM D1388 cantilever test.
:::
