# 🔄 Workflow

Once you have a running solver and a live connection (see
[Connections](../connections/index.md) if you have not set that up yet),
the day-to-day loop is:

1. Organize your scene into object groups: **Solid**, **Shell**,
   **Rod**, **PDRD**, **Sand**, or **Static**.
2. Assign material parameters per group (density, Young's modulus,
   Poisson ratio, friction, bend, shrink, strain limit, and so on).
3. Set scene parameters: **Gravity (m/s²)**, **Wind**, **Step Size**,
   **Frame Count**, **Air Density (kg/m³)**, and **Air Friction**.
4. Add pins and attach operations: **Move By**, **Spin**, **Scale**,
   or **Torque**. (Keyframing a pin by hand attaches **[Embedded] Move**
   for you; it is not offered in the Add menu.)
5. *(Optional)* Add invisible colliders: infinite walls or parametric
   spheres (including bowl and inverted-sphere variants).
6. *(Optional)* Sew two objects together with soft stitches in the
   **Snap and Merge** panel: **Snap A to B** moves one object next to the
   other and records the stitch.
7. **Transfer** geometry and parameters to the solver. The build starts on
   the solver host automatically once the upload completes.
8. **Run** the simulation, then bring the frames back with **Fetch All
   Animation**, which attaches them to your Blender objects as animation.
9. *(Optional)* Press **Bake Animation** to write the fetched animation onto
   the objects as standard Blender keyframes, so the scene plays back
   without the add-on. **Bake Single Frame** writes only the current pose.

```{figure} ../images/workflow/day_to_day_loop.svg
:alt: Nine numbered step boxes across two rows. Row one covers steps 1 through 5: object groups (Solid, Shell, Rod, PDRD, Sand, Static), material parameters, scene parameters, pins with operations (Move By, Spin, Scale, Torque), and the optional invisible colliders. Row two covers steps 6 through 9: the optional Snap and Merge, Transfer and Build, Run and Fetch, and the optional Bake. Scene-setup boxes are blue, solver boxes are orange, and the bake box is green. Optional steps have dashed borders.
:width: 900px

The same nine steps, laid out by phase. Optional steps have dashed borders.
```

Steps 1 through 6 prepare the scene in Blender, steps 7 and 8 send it to the
solver and run the solve, and step 9 writes the result as standard Blender
keyframes. You will usually repeat **Run** and the parameter edits many times
before baking.

Before step 7, scan any geometry you did not author yourself (an imported
character, a downloaded prop) with [Mesh Cleaning](scene/mesh_cleaning.md).
It reports many of the faults that make a **Transfer** fail or a solve stop.

The **Run** half of step 8 can also happen from JupyterLab with Blender
closed. **Fetch All Animation** still runs in Blender, after you reopen the
file. See [JupyterLab](sim/jupyterlab.md) for the full export → simulate →
relaunch Blender → fetch loop.

## Chapters

- [Scene](scene/index.md): object groups, static colliders and mesh cleaning.
- [Parameters](params/index.md): material, scene and dynamic parameters,
  fabric presets and force fields.
- [Constraints](constraints/index.md): pins, invisible colliders, and snap
  and merge.
- [Simulation and Output](sim/index.md): running the solve, JupyterLab,
  baking, and USD or Alembic export.

## Coordinate System

Blender is Z-up; the solver is Y-up. The add-on converts in both directions
automatically. A Blender vector `(x, y, z)` is sent to the solver as
`[x, z, -y]`, and results are converted back on fetch. You only need to
think about this when you read raw solver output. Everything visible in
Blender (pin centers, gravity arrows, overlay previews, baked animation) is
already in Z-up.

```{figure} ../images/workflow/coordinate_system.svg
:alt: Side-by-side diagram of Blender's Z-up right-handed coordinate frame and the solver's Y-up right-handed frame, linked by arrows showing the conversion to the solver (x, y, z) -> [x, z, -y] and the conversion back on fetch [X, Y, Z] -> (X, -Z, Y)
:width: 640px

Both frames are right-handed. On the way to the solver, Blender's `Y`
(into the scene) becomes the solver's `-Z`, and Blender's `Z` (up) becomes
the solver's `Y`. On fetch the conversion runs in reverse, so everything
read back on Blender meshes, pins, and overlays stays in Z-up.
```

:::{note}
Positions you pass through the Python API (pin centers, operation pivots,
collider positions) are Blender world positions. You do not need to
subtract an object's origin.
:::

## Where Each Parameter Is Set

The main sidebar panels and the page that covers each. The
[Tour of the UI](../getting_started/tour.md) lists what every panel holds.

| Panel | What it holds | Docs |
| ----- | ------------- | ---- |
| **Backend Communicator** | The connection to the solver and its status. | [Connections](../connections/index.md) |
| **Scene Configuration** | Global simulation parameters: gravity, step size, frame count, linear-solver tolerances, air density and friction, auto-save. | [Scene Parameters](params/scene.md) |
| **Scene Configuration** *(keyframed sliders)* | Gravity, wind, air density, air friction, vertex air damping, and step size are keyframed on their own sliders, so the curves sit on the timeline with every other keyframe. | [Dynamic Parameters](params/dynamic.md) |
| **Dynamics Groups** | Per-group type, **Preset**, material model, densities, moduli, contact gap, **Spatial Material Maps**, **Velocity Overwrite**, locks, overlay color. | [Object Groups](scene/object_groups.md), [Material Parameters](params/material.md), [Fabric Presets](params/fabric_presets.md) |
| **Dynamics Groups → Pins** *(**Pins & Motion** on a PDRD group)* | Pins and the operations that move them. | [Pins and Operations](constraints/pins.md) |
| **Dynamics Groups → Transform** *(Static only)* | Per-object **Move By**, **Spin**, and **Scale** operations, or the object's own Blender animation. | [Static Objects](scene/static_objects.md) |
| **Scene Configuration → Force Fields** | Blender **Force**, **Wind**, **Vortex** and **Turbulence** objects, and one Python script, that push the simulated objects. | [Force Fields](params/force_fields.md) |
| **Scene Configuration → Invisible Colliders** | Walls and spheres with keyframed position and radius. | [Invisible Colliders](constraints/colliders.md) |
| **Snap and Merge** | Snap and merge pairs with optional stitch stiffness. | [Snap and Merge](constraints/snap_merge.md) |
| **Solver** | The buttons that transfer, run, fetch and bake a simulation. | [Running the Simulation](sim/simulating.md), [Baking Animation](sim/baking.md) |
| **Mesh Repair** | **Mesh Cleaning** (scan and repair), **Symmetric Triangulate**, **Intersection Correction** (separates overlapping **Solid** and **Pdrd** objects) and **Resolve All Intersections** (also reshapes **Shell** cloth). | [Mesh Cleaning](scene/mesh_cleaning.md) |
| **Visualization** | Checkboxes that hide overlays in the viewport. | [Tour: Visualization](../getting_started/tour.md#visualization) |
| **Object Statistics** | Read-only numbers for one object on the frame under the playhead. | [Reading Per-Frame Statistics](sim/simulating.md#reading-per-frame-statistics) |

:::{note}
Everything reachable from the **Scene Configuration** panel is also
reachable from a fluent Blender Python API. See the
[Blender Python API Reference](../integrations/python_api_reference.rst) for
the full surface. Most chapters in this section end with a Python API section
that covers the calls relevant to that chapter.
:::

## Blender Python API

The same workflow is available from Python. [Import](../integrations/python_api.md#import)
`solver`, then read or write through `solver.param`,
`solver.create_group(...)`, `solver.add_wall(...)`, and similar calls:

```python
solver.param.gravity = (0, 0, -9.8)
cloth = solver.create_group("Cloth", "SHELL")
cloth.add("Shirt")
```

:::{admonition} Under the hood
:class: toggle

All add-on data hangs off `scene.zozo_contact_solver`:

| Stored on | What it holds |
| -------- | ------------- |
| `scene.zozo_contact_solver.state` | Global simulation parameters: gravity, step size, frame count, linear-solver tolerances, air density and friction, auto-save. |
| `scene.zozo_contact_solver.object_group_0…31` | Per-group type, material model, densities, moduli, contact gap, overlay color. |
| `object_group_N.pin_vertex_groups` | Pins and their list of operations (`MOVE_BY`, `SPIN`, `SCALE`, `TORQUE`, `EMBEDDED_MOVE`). |
| `scene.zozo_contact_solver.state.invisible_colliders` | Walls and spheres with keyframed position and radius. |
| `scene.zozo_contact_solver.state.merge_pairs` | Snap and merge pairs with optional stitch stiffness. |

The solver stores vertices in untranslated object space (rotation and
scale only) and tracks each object's translation separately. Pin centers
and operation pivots you give the Python API are transformed into that
space when the scene is encoded.
:::
