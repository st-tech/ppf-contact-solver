# 🔄 Workflow

Once you have a running solver and a live connection (see
[Connections](../connections/index.md) if you haven't set that up yet),
the day-to-day loop is:

1. **Organize your scene** into object groups: **Solid**, **Shell**,
   **Rod**, or **Static**.
2. **Assign material parameters** per group (density, Young's modulus,
   Poisson ratio, friction, bend, shrink, strain limit, and so on).
3. **Set scene parameters**: gravity, wind, time step, frame count, air
   density, air friction.
4. **Add pins** and attach operations: **Move By**, **Spin**, **Scale**,
   **Torque**, or keyframed **Embedded Move**.
5. *(Optional)* **Add invisible colliders**: infinite walls or parametric
   spheres (including bowl / inverted sphere variants).
6. *(Optional)* **Snap and merge** overlapping meshes to stitch them across
   group boundaries.
7. **Transfer** geometry and parameters to the solver, then **Build**.
8. **Run** the simulation and **Fetch** frames back as PC2 animation on your
   Blender objects.
9. *(Optional)* **Bake** the fetched animation onto the objects as
   standard Blender keyframes, so the scene plays back without the
   add-on.

```{figure} ../images/workflow/day_to_day_loop.svg
:alt: Nine numbered step boxes across two rows. Row one covers scene-setup steps 1 through 5: object groups (Solid/Shell/Rod/Static), material parameters (density, Young's, Poisson, bend, shrink, strain limit), scene parameters (gravity, wind, time step, frame count, air density), pins and operations (Move By/Spin/Scale/Torque/Embedded Move), and the optional invisible colliders (walls, spheres, bowls). Row two covers steps 6 through 9: the optional Snap & Merge that stitches overlapping meshes across group boundaries, Transfer and Build which encodes the scene for the solver, Run and Fetch which solves on the GPU and downloads per-frame PC2 vertex data onto each Blender mesh, and the optional Bake step that converts the fetched PC2 data into standard Blender shape keys and fcurves and removes the ContactSolverCache modifier. Scene-setup boxes are blue, solver boxes are orange, and the bake box is green. Optional steps have dashed borders.
:width: 900px

The same nine steps, laid out by phase. Steps 1 through 6 are scene
setup in Blender, 7 and 8 are the solve on the GPU, and 9 hands the
result off as self-contained Blender animation. In practice you
bounce between **Run** and the material / scene parameters many times
before baking.
```

Step 8 can also happen entirely from JupyterLab, including with
Blender closed. See [JupyterLab](sim/jupyterlab.md) for the full
export → simulate → relaunch Blender → fetch loop.

## Coordinate System

Blender is Z-up; the solver is Y-up. The add-on converts in both directions
automatically. A Blender vector `(x, y, z)` is sent to the solver as
`[x, z, -y]`, and results are flipped back on fetch. You only need to think
about this when you read raw solver output. Everything visible in Blender
(pin centers, gravity arrows, overlay previews, baked animation) is already
in Z-up.

```{figure} ../images/workflow/coordinate_system.svg
:alt: Side-by-side diagram of Blender's Z-up right-handed coordinate frame and the solver's Y-up right-handed frame, linked by arrows showing the encode conversion (x, y, z) -> [x, z, -y] and the decode conversion [X, Y, Z] -> (X, -Z, Y)
:width: 640px

Both frames are right-handed. On the encode side, Blender's `Y` (into
the scene) becomes the solver's `-Z`, and Blender's `Z` (up) becomes the
solver's `Y`. On fetch the conversion runs in reverse, so everything
read back on Blender meshes, pins, and overlays stays in Z-up.
```

:::{note}
The solver stores vertices in untranslated object space (rotation + scale
only) with object translation tracked separately. Any absolute coordinate you
supply through the Python API (pin centers, operation pivots, collider
positions) is transformed into that space at encode time. You do not need
to pre-subtract the origin.
:::

## Where Parameters Live

Each sidebar panel maps to one chapter below:

| Panel                                  | What's there                                                                                 | Docs                                           |
| -------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Scene Configuration**                | Global sim params: gravity, time step, frame count, CG tolerances, air density / friction, auto-save. | [Scene Parameters](params/scene.md)            |
| **Scene Configuration → Dynamic Parameters** | Keyframed gravity / wind / air density / air friction / vertex air damping.          | [Dynamic Parameters](params/dynamic.md)    |
| **Dynamics Groups**                    | Per-group type, material model, densities, moduli, contact gap, overlay color.               | [Object Groups](scene/object_groups.md), [Material Parameters](params/material.md) |
| **Dynamics Groups → Pin Vertex Groups** | Pins and their list of operations (**Move By** / **Spin** / **Scale** / **Torque** / **Embedded Move**). | [Pins and Operations](constraints/pins.md)  |
| **Dynamics Groups → Transform** *(Static only)* | Per-object **Move By** / **Spin** / **Scale** ops, or Blender transform keyframes.  | [Static Objects](scene/static_objects.md)            |
| **Scene Configuration → Invisible Colliders** | Walls and spheres with keyframed position / radius.                                   | [Invisible Colliders](constraints/colliders.md)  |
| **Snap and Merge**                     | Snap/merge pairs with optional stitch stiffness.                                             | [Snap and Merge](constraints/snap_merge.md)            |

:::{note}
Everything reachable from the Scene Configuration panel is also
reachable from a fluent Blender Python API. See the
[Blender Python API reference](../integrations/python_api.md) for the
full surface; each chapter below also ends with a short
`## Blender Python API` section covering the calls relevant to that
chapter.
:::

## Blender Python API

The same workflow is available from Python. Import `solver` and read or
write through `solver.param`, `solver.create_group(...)`,
`solver.add_wall(...)`, and friends:

```python
from zozo_contact_solver import solver

solver.param.gravity = (0, 0, -9.8)
cloth = solver.create_group("Cloth", "SHELL")
cloth.add("Shirt")
```

:::{admonition} Under the hood
:class: toggle

All add-on data hangs off `scene.zozo_contact_solver`:

| Lives on                                              | What's there                                                                                 |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `scene.zozo_contact_solver.state`                     | Global sim params: gravity, dt, frame count, CG tolerances, air density/friction, auto-save. |
| `scene.zozo_contact_solver.state.dyn_params`          | Keyframed gravity / wind / air density / air friction / vertex air damp.                     |
| `scene.zozo_contact_solver.object_group_0…31`         | Per-group type, material model, densities, moduli, contact gap, overlay color.              |
| `object_group_N.pin_vertex_groups`                    | Pins and their list of operations (MOVE_BY / SPIN / SCALE / TORQUE / EMBEDDED_MOVE).         |
| `scene.zozo_contact_solver.state.invisible_colliders` | Walls and spheres with keyframed position / radius.                                          |
| `scene.zozo_contact_solver.state.merge_pairs`         | Snap/merge pairs with optional stitch stiffness.                                             |
:::
