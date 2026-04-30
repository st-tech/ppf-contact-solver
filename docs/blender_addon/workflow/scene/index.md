# 🎭 Scene

How the solver sees your Blender scene: as a collection of **object
groups** that carry type, material, and assigned meshes, plus a
**Static** variant for non-deforming colliders and props.

:::{warning}
**All objects must not intersect and must not be self-intersecting at
the start of the simulation.** The solver rejects rest geometry where
any face penetrates another face, on the same object or on a
different one, and it cannot recover from a start state that is
already inside out. Before you transfer, confirm that every mesh is
cleanly separated from every other mesh and that no mesh folds
through itself. This rule applies to every object type: Solid, Shell,
Rod, and Static.
:::

:::{note}
**Accepted geometry.** Mesh objects may contain triangles, quads, or
any mix of the two; n-gons are not supported, so triangulate those
before assigning. **Solid** groups are tetrahedralized internally by
[fTetWild](https://github.com/wildmeshing/fTetWild), which is tolerant
of input quality: the surface does not strictly need to be a closed
manifold, and small cracks or near-duplicate vertices are handled
automatically. **Rod** groups accept both mesh and Bezier curve
objects: mesh edges become rod elements directly (faces are ignored,
and a Wireframe modifier is added on assignment so the rod structure
is visible), while Bezier curves are resampled along arc length at
transfer time.
:::

```{toctree}
:maxdepth: 1

object_groups
static_objects
```
