# 🎭 Scene

How the solver sees your Blender scene: as a collection of **object
groups** that carry type, material, and assigned meshes, including
**PDRD** rigid bodies, plus a **Static** variant for non-deforming
colliders and props.

:::{warning}
**All objects must not intersect and must not be self-intersecting at
the start of the simulation.** The solver rejects rest geometry where
any face penetrates another face, on the same object or on a
different one, and it cannot recover from a start state that is
already inside out. Before you transfer, confirm that every mesh is
cleanly separated from every other mesh and that no mesh folds
through itself. This rule applies to every object type: Solid, Shell,
Rod, PDRD, and Static.

Where geometry should pass through other geometry instead of colliding
with it (a mesh tangled in its pose that should stay tangled, a cloth
whose self-collision is not wanted, or layers that should pass through a
body), the group settings under
[Allow Intersections](../params/material.md#allow-intersections) let the
objects you choose pass through each other, and the start pose is
accepted for those pairs. An allowed pair has no contact at all, so the
solver does not push it apart. Set them on the moving object's group: a
**Static** collider left in its rest pose ignores the boxes on its own
group.
:::

:::{note}
**Accepted geometry.** Mesh objects may contain triangles, quads,
n-gons, or any mix of these. The add-on triangulates every polygon
automatically, matching the tessellation you already see in the
viewport, so no manual triangulation is needed and concave n-gons are
handled correctly. **Solid** groups are tetrahedralized internally, and
the tetrahedralizer backend is selectable per object:
[fTetWild](https://github.com/wildmeshing/fTetWild) (the tolerant
default) does not strictly need a closed manifold and handles small
cracks or near-duplicate vertices automatically, while
[TetGen](https://www.wias-berlin.de/software/tetgen/) preserves the
input surface exactly but requires a clean, closed, manifold mesh.
**PDRD** groups are exactly-rigid bodies driven directly from their surface
mesh, so they are not tetrahedralized. **Rod** groups accept both mesh
and Bezier curve objects: mesh edges become rod elements directly
(faces are ignored, and a Wireframe modifier is added on assignment so
the rod structure is visible). For Bezier curves each control point
becomes one rod vertex, so the edge length equals the CP spacing and
the simulation evolves CP positions directly, and you control rod
resolution by adding or removing CPs. NURBS curves are sampled per arc
at four `t` values because NURBS CPs are off-curve.
:::

```{toctree}
:maxdepth: 1

object_groups
static_objects
mesh_cleaning
```
