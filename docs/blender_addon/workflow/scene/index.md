# 🎭 Scene

A Blender scene reaches the solver as **object groups** that carry a type,
a material, and assigned objects. This chapter has four pages:
[Object Groups](object_groups.md) covers creating groups and assigning
objects, including **PDRD** rigid bodies; [Static Objects](static_objects.md)
covers colliders and props that the solver does not simulate, including ones
that move or that an armature deforms; [Hair Curves](hair_curves.md) covers
Blender's hair objects, which are simulated as strands; and
[Mesh Cleaning](mesh_cleaning.md) checks imported or edited geometry before
you transfer it.

:::{warning}
**All objects must start the simulation without intersecting each other and
without intersecting themselves.** The build refuses a scene in which any
face passes through another face, on the same object or on a different one.
It also refuses a scene in which two elements start closer together than
their contact offsets allow, or in which vertices start on the wrong side of
an invisible wall or sphere. Before you **Transfer**, separate every mesh
from every other mesh, make sure that no mesh folds through itself, and keep
every object clear of invisible walls and spheres. This rule applies to
every object type, with two exceptions: a **Static** collider may overlap
another **Static** collider or itself, and a pinned vertex is exempt from the
wall and sphere check.

To move overlapping **Solid** and **Pdrd** objects apart, or to reshape
**Shell** cloth that passes through itself, without running the solver, use
[Intersection Correction](mesh_cleaning.md#intersection-correction) in
**Mesh Repair**.

Where geometry should pass through other geometry instead of colliding
with it, use the settings in a group's
[Allow Intersections](../params/material.md#allow-intersections) box (four
settings, or three on a **Sand** group). They give up contact for whole
objects or, with
[Allow Existing Intersections](../params/material.md#allow-existing-intersections),
only for the pairs that already overlap at the start. Set them on the moving
object's group: a **Static** collider left in its rest pose ignores the boxes
on its own group.
:::

(accepted-geometry)=

:::{note}
**Accepted geometry.** Mesh objects may contain triangles, quads,
n-gons, or any mix of these. **Transfer** triangulates every polygon
automatically, matching the tessellation you see in the viewport, so you
do not triangulate by hand, and concave n-gons are handled correctly. If the
way a face is split leaves a triangle too thin to simulate, **Transfer**
refuses it and opens a dialog. Where a different split fixes every flagged
face, the dialog carries a **Triangulate Faces** button (see
[Mesh Cleaning](mesh_cleaning.md#triangulate-faces)). What else a group takes
depends on its type:

- **Solid**: tetrahedralized internally. The tetrahedralizer is chosen per
  object (see
  [Tetrahedralizer (per object)](../params/material.md#tetrahedralizer-per-object)).
  [fTetWild](https://github.com/wildmeshing/fTetWild), the tolerant default,
  does not strictly need a closed manifold and handles small cracks or
  near-duplicate vertices automatically.
  [TetGen](https://www.wias-berlin.de/software/tetgen/) preserves the input
  surface exactly but requires a clean, closed, manifold mesh.
- **PDRD**: an exactly rigid body driven directly from its surface mesh,
  so it is not tetrahedralized. The surface must be closed and wound
  consistently, because the body's mass is its density times the enclosed
  volume. It may face either way and be of any size.
- **Rod**: a mesh (its edges become rod elements), a Blender curve, or a
  Hair Curves object. See [Group Types](object_groups.md#group-types) for how
  edges and curve points become rod vertices, and
  [Hair Curves](hair_curves.md) for hair.
- **Sand**: only a particle mesh made with **Convert To Solid Particle
  Mesh**. Any other mesh is refused.
- **Static**: any mesh that passes the checks under
  [Which Errors Transfer Also Refuses](mesh_cleaning.md#which-errors-transfer-also-refuses),
  which include a refusal of points that belong to no face.
:::

```{toctree}
:maxdepth: 1
:hidden:

object_groups
static_objects
hair_curves
mesh_cleaning
```
