# 📦 Object Groups

Objects reach the solver through **object groups**. A group has a *type*,
*material parameters* (a *material model* for **Shell**, **Solid**, and
**Rod**), and a list of *assigned objects*. A scene holds up to **32**
groups, and only the groups you have created are sent to the solver.

Which group an object belongs to sets the scope of one contact setting:
**Allow Inter-Group Intersections** lets an object pass through the objects
of other groups while the objects of its own group still collide with it.
See [Allow Intersections](../params/material.md#allow-intersections).

## The Dynamics Groups Panel

Everything you do with groups happens on the **Dynamics Groups** panel in
the 3D viewport sidebar (`N`). From top to bottom the panel holds:

- A top row with **Create Group** and **Delete All Groups**.
- One **group box** per group, stacked below.

Each group box contains, in order:

1. A **header row** with a collapse triangle, an icon for the group type,
   the group name, and a duplicate-group button.
2. The **Name** field with the type dropdown to its right (it shows the
   current type), and below them an **Overlay Color** row with a checkbox
   and a color swatch.
3. The object list. Each row starts with a checkbox (tooltip **Include**)
   followed by the object name. Below the list sit **Add Selected
   Objects** and **Remove Object**.
4. **Bake Animation** and **Bake Single Frame**. They act on the object
   selected in the list, are enabled once that object has fetched
   animation, and remove the baked object from the group (see
   [Baking Animation](../sim/baking.md)).
5. Controls for two types. On a **Static** group, a **Capture
   Deformation** / **Clear Deformation Cache** row with a status line
   beneath it (see [Static Objects](static_objects.md)). On a **Sand**
   group, a box holding **Convert To Solid Particle Mesh** (see
   [Sand-Specific](../params/material.md#sand-specific)). Both are always
   drawn for their type; the buttons gray out until their preconditions
   are met.
6. A full-width **Delete Group** button.
7. A collapsible **Pins** section with the pin vertex groups attached to
   this group (see [Pins and Operations](../constraints/pins.md)). The same
   section reads **Transform** on a **Static** group and **Pins & Motion**
   on a **PDRD** group.
8. A collapsible **Stats** section listing the vertex count (**#Vert**) and
   face count (**#Face**) of each included mesh.
9. A collapsible **Material Params** section (see
   [Material Parameters](../params/material.md)).

```{figure} ../../images/object_groups/panel_overview.png
:alt: Dynamics Groups panel with the Create Group and Delete All Groups buttons on the top row, then one group box with its header row, Name field and Solid type dropdown, Overlay Color row, an empty object list, the Add Selected Objects, Remove Object, Bake Animation, Bake Single Frame and Delete Group buttons, and collapsed Pins, Stats and Material Params sections
:width: 500px

The **Dynamics Groups** panel with one group. **Create Group** and
**Delete All Groups** sit on the top row, and the group box follows.
```

## Creating a Group

Click **Create Group** at the top of the panel. A new group box appears
with the header **Group 1** (then **Group 2**, and so on) over an empty
**Name** field, the default **Solid** type, an empty object list, and
default material parameters. Typing a name in **Name** replaces the numbered
header. The group takes the lowest free slot, so after a group has been
deleted the new box can appear above the existing ones instead of at the
bottom. When all 32 slots are in use, **Create Group** reports *Maximum
number of groups reached*; put objects of the same type and material into
one group instead of creating more.

## Group Types

| Type       | Use it for                           | Default model    | Available models                           |
| ---------- | ------------------------------------ | ---------------- | ------------------------------------------ |
| **Shell**  | Thin surfaces (cloth, fabric)        | Baraff-Witkin    | Baraff-Witkin, ARAP                        |
| **Solid**  | Volumetric bodies                    | ARAP             | Stable NeoHookean, ARAP                    |
| **Rod**    | 1D structures (ropes, wires, hair)   | ARAP             | ARAP only                                  |
| **PDRD**   | Rigid bodies (props, rigid shells)   | n/a (rigid)      | n/a                                        |
| **Sand**   | Granular bodies (sand, loose grains) | n/a (granular)   | n/a                                        |
| **Static** | Collision objects the solver does not simulate (they can still move) | n/a              | n/a                                        |

The type controls which material parameters appear and which material
models are on offer.

A **Static** group shows only **Friction**, an **Apply Soft Constraints**
box, the **Contact** rows, and the **Allow Intersections** box that every
type carries. Where other types have a pin region, it has a **Transform**
sub-box that holds per-object **Move By** / **Spin** / **Scale** ops (an
alternative to the object's own Blender animation). See
[Static Objects](static_objects.md) for the full surface.

**PDRD** (Painless Differentiable Rotation Dynamics) groups move a surface
mesh as a single exactly rigid transform: a translation plus a rotation,
with no stretch at all. Use one for props and panels that must hold their
shape (furniture, plates, hard parts of a rig). The mesh must be a closed
surface whose faces are all wound the same way, because the body's mass is its
density times the enclosed volume. Which way the faces point does not matter,
and neither does the size of the body. **Transfer** refuses an open surface
(*its surface is open*) and one with faces wound against each other (*its
faces are not wound consistently*, which **Recalculate Outside** in
[Mesh Cleaning](mesh_cleaning.md#recalculate-outside) repairs), and it names
the body. A closed part wound against the rest counts as a cavity, which is how
a hollow body is made, and is accepted only where it lies inside the rest. A PDRD body still collides and takes pins (in its **Pins & Motion** section),
but it does not bend or stretch like cloth or soft solids. It has no
stiffness parameters. Its material rows are **Density** and **Friction**,
and it also has the contact rows, a **Hinge** box, a **Velocity (free
launch)** box, **Lock Translation**, and the collision windows described
[below](#active-collision-windows).

**Rod** groups accept mesh objects, Blender curve objects and Hair Curves
objects.

- A **mesh** contributes one rod element per edge at transfer time. Faces
  are ignored. On assignment the add-on adds a **Wireframe** modifier at the
  top of the object's modifier stack, with **Even Thickness** off, unless the
  object already has one, so the rod structure shows in the viewport instead
  of filled faces. **Fetch All Animation** switches **Even Thickness** off
  again on every **Wireframe** modifier of a **Rod** mesh, including one you
  added yourself, because even thickness makes a folding strand jump from
  frame to frame. **Remove Object** and **Delete Group** switch the
  **Wireframe** and **All Edges** viewport overlays off but leave the
  modifier on the object. Delete it in the **Modifiers** tab to stop drawing
  the strand as a wireframe.
- A **Bezier** curve contributes one rod vertex per control point, so the
  edge length equals the control-point spacing and you set the simulation
  resolution by adding or removing control points.
- A **Poly** spline contributes one rod vertex per point.
- A **NURBS** curve is sampled per arc at four `t` values, because NURBS
  control points are off-curve. Each arc spans **Order** minus one points,
  so the spline's points have to fill whole arcs: **Transfer** refuses a
  spline that leaves points past its last whole arc, since those points
  would not be simulated. Blender's **NURBS Circle** (order 3, eight points,
  cyclic) fills whole arcs. See
  [Pins on a Curve Rod](../constraints/pins.md#pins-on-a-curve-rod) for
  which NURBS points can be pinned.
- A **Hair Curves** object (the object Blender's hair tools and **Quick
  Fur** create) contributes one rod vertex per point and one segment per pair
  of neighboring points on a curve. Joining the group pins the roots of hair
  that is attached to a surface and sets the group's **Rest Angle** and
  **Contact Offset** for a groom. See [Hair Curves](hair_curves.md).

**Sand** groups hold grains. A Sand group takes only a particle mesh, a
cloud of grain centers made from a solid mesh with **Convert To Solid
Particle Mesh**. The grain radius is fixed at that conversion. Particle
mass and friction are the Sand material numbers; see
[Sand-Specific](../params/material.md#sand-specific).

The **Material Params** box shows only the rows of the group's type.
[Rows by Group Type](../params/material.md#rows-by-group-type) lists which
type draws which row. Three details are worth stating here:

- A **Rod** group has a **Shrink** row of its own (`length_factor`, drawn
  just under **Friction**) that scales the rest length of every segment of
  the strand.
- **Rod** and **Shell** write the same `bend` property but scale it on
  their own terms, so the number is shared and the stiffness it produces is
  not. A rod draws the field in a separate **Bend** box and starts at
  `1.0`, where a shell starts at `10.0`. Both are covered in
  [Material Parameters](../params/material.md).
- A **Rod** group draws only the absolute **Contact Gap** and **Contact
  Offset** rows, with no ratio rows and no **Use Object Bounding Box
  Diagonal** checkbox (see
  [Contact Gap and Contact Offset](../params/material.md#contact-gap-and-contact-offset)).

## Assigning Objects

To put objects into a group:

1. Select one or more objects in the viewport.
2. In the target group's box, click **Add Selected Objects**. The button is
   grayed out while nothing is selected.

The button accepts mesh objects, plus curve objects and Hair Curves objects
for **Rod** groups. Anything else in your selection is skipped silently.

:::{tip}
Check the mesh resolution before the first **Transfer**. The recommended
**average edge length is 1–3 % of the object's bounding-box diagonal**: a
finer mesh costs more to simulate, and a coarser one folds in blocky
wrinkles. The add-on does not enforce a range. See
[Your First Simulation](../../getting_started/first_simulation.md) for a
worked example and {ref}`Mesh Resolution: The 1–3 % Window <mesh-resolution-window>`
for the two MCP tools that report the numbers.
:::

Five further situations put a message at the bottom of the Blender
window. In each of them that one object is skipped and the rest of the
selection is still processed:

- **Library-linked**. The object comes from a linked library file, and the
  add-on cannot store its identifier on it. A warning says so. Make the
  object local first.
- **Not a particle mesh**. A **Sand** group takes only a particle mesh made
  with **Convert To Solid Particle Mesh**, so a raw solid mesh is refused
  with an error. Run the conversion on it first.
- **Linked duplicate**. The object shares its mesh datablock with another
  object (an `Alt`-`D` copy). Refused with an error; make it single-user
  through **Object > Relations > Make Single User > Object & Data**.
- **Duplicate faces**. Two triangles of the tessellated mesh share the same
  three vertices, usually because a face is doubled. Refused with an error;
  run **Merge by Distance** or **Delete Duplicate Faces** (see
  [Mesh Cleaning](mesh_cleaning.md)) before assigning.
- **Already in another group**. The object is assigned to a different
  active group. A warning is raised; remove it from the first group before
  re-assigning. An object already in *this* group is skipped silently
  instead.

When an assignment succeeds, the add-on enables the object's
**Wireframe** and **All Edges** viewport overlays and, if the group has
**Overlay Color** ticked, tints the object's viewport color to the group's
color.

:::{important}
**The solver simulates the mesh's own vertices and faces.** A
**Subdivision Surface**, **Mirror**, **Bevel**, **Remesh**,
**Solidify**, or any other modifier that changes the vertex count adds
no geometry to the simulation. If the subdivided or beveled mesh is
what you want to simulate, **Apply** those modifiers first
(**Object > Apply > Visual Geometry to Mesh**, or **Apply** in the
modifier's drop-down menu) and then **Transfer**.

Deformers still count. **Transfer** sends each vertex where the modifier
stack puts it at the solve's **Starting Frame**, up to the first modifier
that changes the vertex count. An **Armature**, **Lattice**, or Shape Key
in front of a **Subdivision Surface** reaches the solve, while a deformer
after it applies on display only, on top of the simulated result. The
*Under the hood* block at the end of this page gives the cut rules.
:::

### The Object List

Each row of the object list has a checkbox (tooltip **Include**) and the
object's name. Unchecking the checkbox keeps the object in the list but
leaves it out of the next **Transfer** and resets its viewport color to
white. Use it to A/B-test a scene without dismantling the group. Double-click
an object's name in the list to rename the object. Its pins, merge pairs and
**Allow Intersections** list entries follow the new name.

If you delete an assigned object in Blender, the add-on takes it out of its
group on its own, together with the pins registered on it, its entries in the
group's **Allow Intersections** object lists and every
[merge pair](../constraints/snap_merge.md) that names it. A row that reads
*(Missing)* after a name is an object the add-on cannot find in the file, and
the add-on removes such rows on its own.

To remove an object from a group, select its row and click **Remove
Object**. Removing an object:

- resets its viewport color to white and switches its **Wireframe** and
  **All Edges** overlays off;
- discards its fetched animation, so fetch again to get the frames back;
- removes the pins registered on it, its entries in the group's **Allow
  Intersections** object lists, and every
  [merge pair](../constraints/snap_merge.md) that names it.

```{figure} ../../images/object_groups/assigned_objects.png
:alt: Assigned object list showing one entry
:width: 500px

The object list of a **Shell** group with one `Cloth` mesh assigned. The
left checkbox is the **Include** flag; unchecking it excludes the object
from the next transfer without removing it from the group.
```

## Active Collision Windows

For **Solid**, **Shell**, **Rod**, and **PDRD** groups, contact on each
assigned object can be limited to frame ranges. The control is the
**Collision Active Duration Windows** toggle in the **Material Params** box,
just below the **Contact Gap** rows (see also
[Material Parameters](../params/material.md)).

When the toggle is off (the default), every assigned object collides for
the full timeline. Switching it on reveals a per-object editor:

- A dropdown next to the **Active Windows** label picks which of the
  group's assigned objects to edit.
- Below it sits the list of **Active Windows** for that object: frame
  ranges during which contact is enabled. An object holds up to **8**
  windows.
- The **+** and **−** buttons add and remove entries. A new window covers
  frames **1** to **60**, or the same 60 frames beginning at the solve's
  **Starting Frame** when that is later.
- Selecting a window shows **Start Frame** and **End Frame** spinners that
  edit its bounds. **Transfer** refuses a window whose **End Frame** is not
  after its **Start Frame**, or whose **Start Frame** is before the
  **Starting Frame**, naming the object and both frames. Nothing is
  clipped: move the window or the **Starting Frame**.

Outside every window listed for an object, that object's contact is
muted. Three rules apply:

- **No windows on an object means contact is always active.** Switching
  the toggle on does not disable anything by itself; add at least one
  window on an object for the cutoff to apply to it.
- **Invisible colliders are not affected.** Walls and spheres always
  collide, even when surrounding deformables have collision windows set.
  Invisible colliders have their own per-collider **Active Duration**
  option; see [Invisible Colliders](../constraints/colliders.md).
- **Static groups have no collision windows.** Static objects collide for
  the full timeline. To make one come and go mid-shot, use its
  **Transform** ops (see [Static Objects](./static_objects.md)).

Unticking **Collision Active Duration Windows** keeps the windows you set
but stops using them, so every object collides for the full timeline until
you tick it again. Adding a ninth window to an object is refused with
*Maximum 8 collision windows per object*. The windows belong to the objects,
so a material profile and the duplicate icon carry the toggle and no windows.

Windows suit timed catch-and-release setups (a hand grabs a piece of
cloth, then lets go) and sequential drops where each object should start
interacting only after the one below it has settled.

## Overlay Colors

The **Overlay Color** row tints the objects of a group in the viewport, so
you can tell at a glance which objects belong to which group. The checkbox
turns the tint on and off. The color swatch beside it, drawn only while the
checkbox is ticked, opens Blender's standard color picker. Picking a color
tints every assigned object's viewport color, the Object color that Solid
shading displays.

:::{note}
The overlay color is **not a material**. It is only a viewport preview
tint. It is not sent to the solver and has no effect on the simulation.
:::

- Unticking **Overlay Color** resets every assigned object's viewport color
  to white. It does not restore the color the object carried before it
  joined the group. Ticking it again tints the objects with the swatch's
  current color.
- Unchecking an object's **Include** box resets that object's color to
  white.
- On a **Rod** group, **Overlay Color** also draws each segment of every
  included mesh rod as a thin line in the swatch color. The lines are a
  viewport preview only, and **Hide all overlaid colors** does not hide them.
- Changing a group's type resets its overlay color to the new type's
  default (table below).
- **Hide all overlaid colors** in the **Visualization** panel suppresses
  every group tint at once without changing any group.
- The tint shows when Solid shading's **Color** is set to **Object**.
  **Add Selected Objects** sets that for you in every Solid-shaded 3D
  viewport when **Overlay Color** is ticked.

```{figure} ../../images/object_groups/group_header_row.png
:alt: Top rows of a Shell group box: the header row with the group name and duplicate icon, the Name field with the Shell type dropdown, and the Overlay Color checkbox with a green color swatch
:width: 500px

Top rows of a **Shell** group box. The **Overlay Color** checkbox and the
color swatch beside it control the viewport tint; clicking the swatch opens
Blender's color picker. The duplicate icon sits at the right of the header
row (covered below).
```

```{figure} ../../images/object_groups/overlay_color_preview.png
:alt: Blender viewport with a cube tinted green, assigned to a Shell group whose overlay color is green. The Dynamics Groups panel is visible at the right edge of the window.
:width: 720px

The `Cloth` cube belongs to a **Shell** group whose overlay color is green,
so it reads green in the viewport. With a distinct color per group you can
tell which object belongs where without clicking each one.
```

Each group gets a default overlay color based on its type:

| Type       | Default RGB           |
| ---------- | --------------------- |
| **Solid**  | red `(0.75, 0, 0)`    |
| **Shell**  | green `(0, 0.75, 0)`  |
| **Rod**    | yellow `(0.75, 0.75, 0)` |
| **PDRD**   | magenta `(0.75, 0, 0.75)` |
| **Sand**   | tan `(0.75, 0.375, 0)` |
| **Static** | blue `(0, 0, 0.75)`   |

## Duplicating a Group

The duplicate icon at the right of a group's header row creates a new group
with the source's type, overlay color, and the core material parameters
that a material profile stores. Objects, pins, and pin operations are
**not** copied; the new group is empty. Parameters a profile does not store
start at their defaults, for example Rayleigh damping, the directional bend
rows, **Allow Intersections**, **Force Field Weight**, a **Rod**'s
**Shrink**, the PDRD and Sand fields, and the **Spatial Material Maps**
rows; see
[Profile Buttons](../params/material.md#profile-buttons-open--clear--reload--save).
The name increments: `Silk` → `Silk-1`, `Silk-1` → `Silk-2`.

```{figure} ../../images/object_groups/duplicate_icon.png
:alt: Dynamics Groups panel with the duplicate icon in the group header outlined in red
:width: 500px

The duplicate icon sits at the right of each group's header row. It
creates a new group with the same type, overlay color, and core material
parameters; the new group has no assigned objects, no pins, and no pin
operations.
```

## Deleting Groups

- **Delete Group** (the full-width button in the group's own box, below the
  Bake row) removes one group. When the group has assigned objects, Blender
  asks you to confirm first. The group's objects get a white viewport color
  and their **Wireframe** and **All Edges** overlays switch off, the
  group's pin vertex groups are removed, and so is every
  [merge pair](../constraints/snap_merge.md) naming one of the group's
  objects.
- **Delete All Groups** (top row of the panel, next to **Create Group**)
  does the same for every group after a confirmation. It is grayed out
  while no group exists.

Neither button discards the fetched animation. A mesh keeps its cache
modifier, and with it the fetched animation. The **Solver** panel then lists
the mesh as a stale cache (see
[The Solver Panel](../sim/simulating.md#the-solver-panel)), and **Remove
Stale Cache** clears it. To drop an object's animation together with its
membership, use **Remove Object** instead.

```{figure} ../../images/object_groups/delete_all_groups.png
:alt: Dynamics Groups panel with the Delete All Groups button on the top row outlined in red
:width: 500px

**Delete All Groups** (outlined in red) sits on the panel's top row, paired
with **Create Group**. Each group box has its own full-width **Delete
Group** button.
```

## Blender Python API

The same workflow is available from Python. The API mirrors the UI
one-for-one: groups are created through `solver.create_group` and objects
are assigned with `group.add`. Most operations on a group return the group
itself for chaining; `delete()` returns nothing. The examples assume
`solver` is imported as shown under
[Import](../../integrations/python_api.md#import).

```python
# Create groups of each type.
cloth = solver.create_group("Cloth", "SHELL")
body  = solver.create_group("Body",  "STATIC")

# Assign Blender objects by name.
cloth.add("Shirt")
cloth.add("Skirt")
body.add("Character")

# Override the overlay color (RGBA, 0 to 1).
cloth.set_overlay_color(0.2, 0.8, 1.0, 0.75)

# Delete one group, or wipe them all.
cloth.delete()
solver.delete_all_groups()
```

A few operations have no fluent method. Each is an MCP tool (see
[MCP Server](../../integrations/mcp.md)), and a script can call it as
`solver.<tool>(...)` through
[Raw Operator Dispatch](../../integrations/python_api.md#fallback-raw-operator-dispatch),
for example
`solver.set_object_included(group_uuid=cloth.uuid, object_name="Skirt", included=False)`.
Dispatch returns only the operator status, never the tool's own result.
`get_group_objects` and `list_collision_windows` exist to report data, so they
answer over MCP only, and so does `duplicate_group`, which reports the new
group's `group_uuid`. A script finds the new group with `solver.get_groups()`.

- `rename_group(group_uuid, name)` renames a group. An empty name falls back
  to `Group N`.
- `duplicate_group(group_uuid)` is the duplicate icon.
- `set_object_included(group_uuid, object_name, included)` is an object's
  **Include** checkbox.
- `remove_all_objects_from_group(group_uuid)` removes every object.
  `group.remove("Skirt")` removes one.
- `get_group_objects(group_uuid)` reports each assigned object's own state:
  its **Include** flag, locks, hinge, bending reference, tetrahedralizer
  settings, and how many collision windows, static operations and velocity
  keyframes it holds.

The [collision windows](#active-collision-windows) are reached the same way:

- `set_use_collision_windows(group_uuid, enable)` is the **Collision Active
  Duration Windows** toggle.
- `add_collision_window(group_uuid, object_name, frame_start, frame_end)`
  adds a window. Both frames must be 1 or later, `frame_end` must be after
  `frame_start`, and an object holds at most 8.
- `set_collision_window(group_uuid, object_name, index, frame_start, frame_end)`
  changes either bound of a window in place. The window that results must
  still end after it starts.
- `remove_collision_window(group_uuid, object_name, index)` removes one
  window by its zero-based `index`.
- `list_collision_windows(group_uuid, object_name)` and
  `clear_collision_windows(group_uuid, object_name)` read and empty an
  object's list.

:::{admonition} Under the hood
:class: toggle

**32-slot model**

The 32 slots are fixed properties on the scene (`object_group_0` through
`object_group_31`), not a dynamic collection. Creating a group takes the
first inactive slot; deleting one frees the slot. This means:

- Groups survive file save / load without reassignment.
- The underlying slot index is **not** the same as the display number:
  the UI renumbers visible groups sequentially (1, 2, 3, and so on) even if the
  underlying slots are 0, 3, 7.

**Library-linked rejection**

The add-on stamps a persistent UUID on each assigned object. Library-linked
objects are refused because the add-on cannot store its UUID on them.

**Modifier stack and the cache modifier**

The solve uses the modifier stack cut at the first modifier that changes the
vertex count. In a stack that does change the vertex count, a **Geometry
Nodes** modifier counts as a cut point even when it only moves vertices.
**Capture Deformation** on a pin reads the same cut stack.

The **ContactSolverCache** Mesh Cache modifier that the add-on installs
after **Fetch All Animation** sits at that cut. On a simulated object it
takes the first slot when the object has no deforming modifier; otherwise
it sits after the deformers and before the first modifier that changes the
vertex count (always the latter on a **Static** collider). Modifiers after
it, such as a Subdivision Surface for smooth rendering, stack on top of the
simulated result without interfering with it.
:::
