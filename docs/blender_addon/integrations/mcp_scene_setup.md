# 🎨 Scene Setup via MCP

The MCP surface is not just a protocol. It is also the recommended way
for agents (and automation scripts) to *build* a scene before hitting
**Transfer → Run**. Unlike a human user, an agent cannot eyeball a
mesh and decide it is "probably fine", and the solver is unforgiving
about resolution and topology with failure modes that are usually
silent. This page collects the rules that keep MCP-driven setup stable.

:::{tip}
Everything below is callable via `tools/call` on the MCP server; the
exact parameter schemas live at
[MCP Tool Reference](mcp_reference.rst). The advice here is about
*which* tools to reach for in what order, not about their JSON
schemas.
:::

## Use MCP Tools, Not Raw Python

The add-on ships a `run_python_script` tool that evaluates arbitrary
Python inside Blender. It is an escape hatch, so reserve it for things
the dedicated handlers genuinely do not cover. When a first-class tool
exists, prefer it:

- Creating, deleting, or typing a group: `create_group`,
  `delete_group`, `set_group_type`.
- Assigning objects to a group: `add_objects_to_group`.
- Setting material parameters: `set_group_material_properties`.
- Scene-level parameters: `set_scene_parameters`.
- Pins, colliders, merges: the dedicated handlers in the
  [reference](mcp_reference.rst).

The dedicated handlers all go through the same validation layer the UI
uses, so a misbehaving agent gets the same errors a user would.
`run_python_script` bypasses that layer entirely.

Two further rules for agents:

- **Do not create temporary files** to stage Python code. Pass the code
  string directly to `run_python_script`.
- **Do not shell out** to `blender --python …`, `cargo run`, or the
  debug CLI to work around a missing MCP surface. If a tool is
  missing, surface it as a gap rather than routing around it.

## Placement and Clearance

The solver rejects self-intersecting rest geometry and most
zero-clearance contact. Before creating groups:

1. Populate the scene with no overlapping meshes. Cloth must start
   **outside** the body it will eventually wrap, and colliders must not
   pierce each other.
2. Leave a small clearance at least as large as the group's contact
   gap (see [Material Parameters](../workflow/params/material.md) for
   how contact gap is specified). A good rule of thumb is a clearance
   of two to three contact-gap widths.

Running `get_object_bounding_box_diagonal(object_name)` on the target
object is a cheap way to get a sense of scale before deciding what
"small clearance" means in Blender units.

## Creating Sphere Primitives

When an agent needs a spherical mesh (a ball, a drop, a filler object),
create an **icosphere**, not a UV sphere. UV spheres concentrate
triangles at the poles and leave equatorial cells stretched; that
anisotropy shows up in the solver as direction-dependent stiffness and
can make bending look unnatural. Icospheres are near-uniform, which
the solver prefers.

In a `run_python_script` call, this looks like:

```python
import bpy
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.5)
```

For `subdivisions`, pick the level that puts the average edge length
inside the 1–3 % window described next, not a fixed default.

## Mesh Resolution: The 1–3 % Window

The most common reason MCP-driven scenes blow up is **under-resolved
meshes**. The solver needs enough vertex density to represent bending
and contact; a coarse mesh produces visible facets and, worse,
missed contacts.

:::{warning}
Aim for an **average edge length of 1–3 % of the object's
bounding-box diagonal**. Below ~1 % you pay a large simulation cost
without much visual gain; above ~3 % folds and wrinkles get blocky.
:::

Two MCP tools let an agent check this without opening Blender's UI:

- `get_average_edge_length(object_name)`: returns the mean world-space
  edge length for a mesh.
- `get_object_bounding_box_diagonal(object_name)`: returns the largest
  diagonal of the object's bounding box in world units.

Typical check before assigning an object to a group:

```text
ratio = average_edge_length / bounding_box_diagonal
if 0.01 <= ratio <= 0.03:
    assign to group
elif ratio > 0.03:
    subdivide or recreate at higher resolution
else:  # ratio < 0.01
    mesh is finer than needed; fine to assign, but slow
```

Treat the 1–3 % range as a hard requirement for dynamic (**Shell**,
**Solid**, **Rod**) groups. **Static** colliders can sit outside the
window when they are simple primitives (a flat ground plane, a single
large sphere), but anything dynamic needs to be inside it.

## Subdividing to Reach the Window

If a mesh is too coarse, prefer **subdivision** over remeshing.
Subdivision preserves the vertex order the add-on's UUID tracking
depends on, so pins and vertex groups survive.

**Always use the Simple method.** The default Catmull-Clark method
rounds corners, which changes the silhouette of engineered meshes
(plates, boxes, cut-outs) and can introduce self-intersections near
sharp features. Simple keeps the original topology and lets the
solver handle smoothing implicitly.

Example via `run_python_script`:

```python
import bpy
obj = bpy.data.objects["Cloth"]
mod = obj.modifiers.new(name="Subsurf", type="SUBSURF")
mod.subdivision_type = "SIMPLE"   # NOT the default "CATMULL_CLARK"
mod.levels = 2                    # viewport level
mod.render_levels = 2             # render level, match for determinism
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier=mod.name)
```

Re-check the edge-length ratio after applying. If a single round of
subdivision still leaves you above 3 %, apply another.

## When to Delete and Recreate

Subdivision has a ceiling: past roughly four Simple levels the vertex
count explodes and performance suffers. When a primitive is
unambiguously procedural (icosphere, cylinder, torus, grid), it is
usually better to **delete and recreate** at the target resolution
than to keep subdividing.

The recreate loop:

1. Record the object's name, transform, and whether it sits in any
   groups or pin vertex groups.
2. Delete the object. Confirm it is gone (for example, by listing
   scene objects and asserting the name is absent) before creating
   the replacement. Creating before deleting can leave two copies
   and silently double-assign them.
3. Create the new primitive with a `subdivisions` / `resolution`
   parameter high enough that the 1–3 % check passes on the first
   try.
4. Re-apply the recorded transform and re-add it to its groups.

This pattern is much cleaner for agents than iteratively poking a
mesh toward the right density.

## Material Parameters: Don't Invent Defaults

`set_group_material_properties` accepts the full material whitelist
for the group's type (see
[Material Parameters](../workflow/params/material.md) for the list).
**Set only what the user has explicitly requested.** Do not pick
"reasonable" defaults for density, Young's modulus, Poisson ratio,
friction, or contact gap on the user's behalf. The group's property
defaults already represent a calibrated starting point, and silently
overwriting them with round numbers is a recipe for sims that don't
match what the user asked for.

When the user *does* request something material-specific (e.g. "make
this cloth stiffer"), prefer a [material profile](../workflow/params/material.md#material-profiles)
over a hand-rolled set of numbers when one fits. Profiles are the
curated values.

### Strain Limit for Stiff Cloth

If the user asks for a stiff, low-stretch cloth, the common recipe is:

- `enable_strain_limit = true`
- `strain_limit = 0.05` (≈ 5 %)

The rest of the defaults can stay as-is unless the user calls them
out. See [Strain Limit](../workflow/params/material.md#strain-limit)
for the semantics.

### Contact Gap: Pick One Mode

The absolute and ratio contact-gap modes are mutually exclusive, and
`use_group_bounding_box_diagonal` switches between them. Do not try
to set both pairs of fields in a single
`set_group_material_properties` call. If you need to change modes,
issue one call that sets `use_group_bounding_box_diagonal`, then a
second that sets the matching `contact_gap` / `contact_offset` or
`contact_gap_rat` / `contact_offset_rat` pair.

## Group-Type Cheat Sheet

When assigning a type with `set_group_type`, use these rules:

| Object                                                 | Type       |
| ------------------------------------------------------ | ---------- |
| Static rigid collider (floor, mannequin, wall, table)  | `STATIC`   |
| Thin flexible surface (cloth, sheet, banner, shell)    | `SHELL`    |
| Volumetric flexible solid (rubber ball, sponge, foam)  | `SOLID`    |
| 1-D flexible line (rope, cable, hair strand)           | `ROD`      |

A group with no assigned objects is almost always a bug. Before
calling `transfer_data`, list every active group via
`get_active_groups` and verify each one has at least one object, and
that the total across groups matches the set of objects the user
intends to simulate.

## See Also

- [MCP Server](mcp.md): protocol, transport, and security.
- [MCP Tool Reference](mcp_reference.rst): full handler list and
  schemas.
- [Object Groups](../workflow/scene/object_groups.md): what each group type
  does in the solver.
- [Material Parameters](../workflow/params/material.md): the fields
  `set_group_material_properties` accepts and their defaults.
