# 🗿 Static Objects

A **Static** group holds meshes that the solver does not deform: ground
planes, mannequins, props, and anything else that should push on the
simulation without being simulated. It uses the same panel, **Transfer** and
bake flow as a **Solid**, **Shell** or **Rod** group, with a short list of
material settings and three ways to move an object: Static ops (the **Move
By**, **Spin** and **Scale** entries listed in the **Transform** box), Blender
animation, and captured deformation. An object follows one of the three at a
time.

This page covers everything specific to the Static type:

- [Creating a Static group](#creating-a-static-group)
- [Moving a Static object](#moving-a-static-object): the three ways to give
  an object motion, and which one applies
- [The Transform sub-box](#the-transform-sub-box): where Static ops are
  edited
- [Armature-driven Static objects](#armature-driven-static-objects):
  **Capture Deformation** records a deforming mesh onto the collider
- [Re-capturing every deformation at once](#re-capturing-every-deformation-at-once):
  the **Re-capture All Deformations** and **Clear All Deformations** buttons
  on the **Solver** panel
- [Contact parameters](#contact-parameters): the small parameter set a
  Static group exposes
- [Baking behavior](#baking-behavior)
- [Snap and merge](#snap-and-merge)
- [Python / MCP API](#python--mcp-api)

## Creating a Static Group

1. Click **Create Group** on the **Dynamics Groups** panel.
2. Open the type dropdown next to **Name** and choose **Static**.
3. Select the meshes and click **Add Selected Objects**.

The group card differs from the other types in four ways:

- The object list accepts meshes only. Curves are rejected because they
  belong in **Rod** groups.
- The region that holds **Pins** on other types is labeled **Transform**, has
  a driver icon and starts collapsed. A Static group has no pin vertex groups;
  the box holds each object's [Static ops](#the-transform-sub-box).
- **Material Params** shrinks to **Friction**, an **Apply Soft Constraints**
  box, the contact rows, and the **Allow Intersections** box that every group
  type carries (see [Contact parameters](#contact-parameters)).
- The default overlay color is blue.

Duplicating the group, the per-object **Include** checkbox, removing objects
and deleting the group work as they do for the other types. See
[Object Groups](object_groups.md).

:::{note}
**A Static collider mesh must have no stray vertices.** A point that belongs
to no face, often left behind by an imported model, makes **Transfer** stop
with an "isolated vert" error. Click **Remove Isolated Vertices** under the
error, or select the mesh, run **Scan Selected Meshes** in **Mesh Repair >
Mesh Cleaning**, and click **Remove Loose Vertices** (see
[Mesh Cleaning](mesh_cleaning.md)). In Edit Mode, **Select > Select All by
Trait > Loose Geometry** followed by **Mesh > Clean Up > Delete Loose** does
the same and also removes edges that belong to no face. Then **Transfer**
again. The message is also listed under
[Transfer, run, fetch](../../troubleshooting.md#transfer-run-fetch).
:::

## Moving a Static Object

A Static mesh does not deform in the simulation, but it can move, turn, scale
or follow a posed rig over time. Give each object one of three kinds of
motion:

1. **Static ops**: **Move By**, **Spin** and **Scale** entries that you add
   per object in the group's [Transform sub-box](#the-transform-sub-box).
   Each op has a time range and a delta, axis or factor. Use them when the
   motion is easy to describe with a few time ranges: a floor plate that
   slides from A to B between frames 30 and 60, a turntable, a shrinking
   platform.
2. **Blender animation of the object as a whole**: insert keyframes on the
   object's Location, Rotation or Scale (for example, press <kbd>I</kbd> in
   the 3D viewport). Motion that a parent, a constraint, a driver or an NLA
   strip gives the object counts too. The add-on reads the object's world
   transform at every frame of the solve, so easing, Bezier handles and
   parent motion reach the simulation as Blender plays them. Use this when
   the motion is already keyed in Blender: a prop animated by hand in the
   Graph Editor, a collider parented to an animated object such as a Camera,
   a collider that follows a constraint.
3. **Captured deformation**: for a mesh whose shape changes over time, such
   as an **Armature** modifier driven by a posed rig, a **Lattice** or
   **Mesh Deform** cage, animated **Shape Keys** or **Geometry Nodes**. Click
   [**Capture Deformation**](#armature-driven-static-objects) to record the
   animation onto the collider. A collider that only moves, turns or scales
   as a whole needs no capture.

:::{warning}
**An object follows one source of motion.**

- If an assigned mesh has Location, Rotation or Scale keyframes of its own
  that move it during the solve, the add-on uses them and ignores the
  object's Static ops. As soon as the object has such keyframes, the
  **Transform** box shows a label that begins *"Object has Blender
  keyframes"* and says the ops will be ignored.
- If a parent, a constraint, a driver or an NLA strip moves the collider, it
  cannot also carry Static ops. **Transfer** refuses it and names the object;
  remove its ops or stop the other motion.
- A captured deformation takes priority over both. While an object has a
  recording, its Static ops and transform animation are ignored, because the
  recording already includes any rigid parent motion.
:::

:::{note}
**A moving collider cannot be sheared.** A moving collider carries a
location, a rotation and a scale. A rotated child of a parent with a
non-uniform scale is sheared, and **Transfer** refuses it when it moves,
naming the object and the first sheared frame. Give the parent a uniform
scale, or click **Capture Deformation** on the collider's row to use the
recorded shape instead.
:::

:::{note}
**Shape Keys and other mesh-level animation work only through Capture
Deformation.** A Static mesh with Shape Key animation does not move in the
solver until you click **Capture Deformation**, and **Transfer** refuses to
upload it with an error until you do.
:::

## The Transform Sub-Box

The **Transform** box edits the Static ops of the object selected in the
group's object list. The box starts collapsed: select the object in the list,
then click the disclosure triangle next to **Transform**. It has no object
picker of its own. With no row selected it shows *"Select an assigned object
above"*, and when the selected object cannot be found it shows *"Object
unresolved"*.

From top to bottom it holds:

1. **A warning row**, drawn only when the selected object has Location,
   Rotation or Scale keyframes of its own. It begins *"Object has Blender
   keyframes"* and says the ops will be ignored, which they are once those
   keyframes move the object during the solve.
2. **The ops list.** Each row shows the op, its frame range and an eye icon.
   The eye shows a preview of that op in the 3D viewport and is off by
   default: an arrow for **Move By**, a circle with a sweep arc for **Spin**,
   and arrows from each vertex to its scaled position for **Scale**. Beside
   the list, the plus button opens a menu with **Move By**, **Spin** and
   **Scale**, the minus button removes the selected op, and the up and down
   arrows move it in the list.
3. **The editor for the selected op**, with these fields:
   - **Start** / **End**: Blender frames. The op is active across the closed
     range. A new op spans frames 1 to 60, or 60 frames beginning at the
     **Starting Frame** when that is later. **Transfer** refuses an op whose
     **End** is not after its **Start**, or whose **Start** is before the
     **Starting Frame**, naming the object, the op and both frames. Move the
     op's frames or the **Starting Frame**.
   - **Transition**: **Linear** (the default) or **Smooth** (smoothstep). It
     shapes **Move By** only. A **Scale** op ramps linearly and a **Spin**
     turns at a constant rate whichever is chosen, even though the dropdown is
     drawn for all three. (A pin's **Scale** does honor **Transition**; see
     [Pins and Operations](../constraints/pins.md).)
   - **Delta (m)**: **Move By** only. The `(x, y, z)` translation.
   - **Axis** / **Angular Velocity (°/s)**: **Spin** only, with defaults
     `(0, 0, 1)` and 360. The object turns around its origin, and a negative
     angular velocity turns it the other way. **Transfer** refuses an
     **Axis** of `(0, 0, 0)` with a nonzero angular velocity, since it names
     no axis to turn about.
   - **Factor**: **Scale** only. A uniform scale about the object's origin,
     default 1.0, minimum 0.01. A value below 1 shrinks the object and a
     value above 1 grows it.

   **Spin** and **Scale** also show the label *"Center: object origin"*.

A new op is added at the top of the list and becomes the selected row. The
ops are applied in list order, from the top row down, so moving a row can
change the result. If a **Move By** and a **Spin** overlap in time, the
object moves and turns at the same time inside the overlap. Edit a row in
place: an op that is removed and added again returns at the top of the list,
which can change the result.

Adding an op to an object that has its own Location, Rotation or Scale
keyframes works, and the status bar warns that the op will be ignored at
simulate time. The op is kept, and it takes effect when the keyframes are
removed or do not move the object.

Static ops are uploaded with the object's geometry. After you add, remove,
reorder or edit any op, or record a deformation again, press **Transfer**.
**Update Params on Remote** is not enough.

Before an op's **Start** frame it has no effect. After its **End** frame the
object stays where the op left it (moved, turned or scaled) until a later op
changes it. Until its first op starts, an object keeps its own pose at the
**Starting Frame**.

## Armature-Driven Static Objects

A Static mesh can move because of something inside the mesh: an **Armature**
modifier on a body model, a **Lattice** or **Mesh Deform** cage, animated
**Shape Keys**, or a similar setup. Object keyframes cannot describe that
motion, so you record it onto the collider with the **Capture Deformation**
button.

When you assign such an object to a Static group, click its row in the
group's object list. **Capture Deformation** and **Clear Deformation Cache**
act on the selected row, so both are grayed out until a row is selected. Once
the row of a deforming object is selected, **Capture Deformation** is
enabled, **Clear Deformation Cache** stays grayed out because nothing is
recorded yet, and a hint appears under the buttons.

```{figure} ../../images/static_objects/armature_panel_pre_capture.png
:alt: Dynamics Groups panel showing a Static group "Mannequin" with the deforming Cube assigned and its row selected. The Capture Deformation button is enabled, the Clear Deformation Cache button is grayed out, and a hint reads "Deforming modifier detected; capture to encode".
:width: 500px

A **Static** group after a deforming mesh is assigned and its row is
selected. **Capture Deformation** is enabled, **Clear Deformation Cache** is
grayed out because nothing is recorded yet, and the line under the buttons
reads *Deforming modifier detected; capture to encode*.
```

The line under the two buttons depends on the selected row. With no row
selected, the first row below applies. With a row selected and no recording
yet, the line is the first of the other rows that matches the object:

| Line under the buttons | The object | What to do |
| ---------------------- | ---------- | ---------- |
| *Deformable object(s) detected; select one from the list above to enable capture* | No row is selected in the group's object list, and an assigned mesh has a deforming modifier. | Click the object's row. Both buttons stay grayed out until you do. |
| *Deforming modifier detected; capture to encode* | Its mesh changes shape. | Click **Capture Deformation**. **Transfer** refuses the object until you do. |
| *Parent or constraint motion transfers automatically; capture is optional* | It moves as a whole through a parent, a constraint, a driver or an NLA strip. | Nothing. The motion is sampled at every frame at **Transfer**. **Capture Deformation** stays available, and a recording, if you take one, is used instead of the samples. |
| *Keyframe animation transfers automatically; capture is for deformers* | It has Location, Rotation or Scale keyframes of its own. | Nothing. The keyframed motion is sampled at **Transfer**, and **Capture Deformation** stays grayed out. |

Once the selected object has a recording, the line shows the recorded frame
count instead of any of these rows.

### The Capture Deformation Button

Click **Capture Deformation** to record the animation. The add-on steps
through the animation frame by frame and stores the shape of the mesh at each
frame. The recording starts at the **Starting Frame** and runs to the last
keyframe found on the object, its Shape Keys, its parents and the objects its
modifiers point to (an armature, for instance). When none of those carry
keyframes, as with a driver, a constraint or Geometry Nodes on scene time, it
runs for the **Frame Count** instead.

While it runs, a progress bar with an **Abort** button appears below the
**Transfer** and **Run** buttons in the **Solver** panel. The timeline
playhead moves during the capture and returns to where it was.
**Abort** stops the capture without saving a partial recording.

When the capture finishes, the line under the buttons shows *Deform cache: N
frame(s)* and **Clear Deformation Cache** becomes available.

```{figure} ../../images/static_objects/armature_btn_capture_deformation.png
:alt: The Dynamics Groups panel with a Static group "Mannequin" holding the object Cube. A red box marks the Capture Deformation button, which sits left of the Clear Deformation Cache button on the row under Bake Animation and Bake Single Frame. The line below the buttons reads "Deform cache: 60 frame(s)" and Clear Deformation Cache is enabled.
:width: 500px

After a capture. The red box marks **Capture Deformation**; press it again
whenever the animation changes. The line below the buttons shows the
recorded frame count, and **Clear Deformation Cache** is available.
```

:::{warning}
**Press Capture Deformation again whenever the animation changes.** The
recording is a snapshot taken when you clicked the button, and it does not
update on its own. If you change the armature pose, edit the action's
keyframes, change the modifier stack, edit the rest mesh, or alter parent or
constraint chains, the recording is out of date and the solver keeps using
the old motion. Capture again before the next **Transfer**.

A topology-changing modifier (Subdivision Surface, Remesh, Decimate) in the
stack does not stop the capture. The recording holds the mesh as deformed by
everything above the first such modifier, which is the shape the solver
collides against. That modifier and everything below it apply on top of the
recording when the result plays back. Put the deformer above the Subdivision
Surface: a deformer below it changes only what you see, not what the cloth
hits.
:::

### The Clear Deformation Cache Button

**Clear Deformation Cache** discards the recording of the selected object.
Use it to discard a recording you do not want (for instance, when you turn the
object back into a rigid collider), or when an edit changed the mesh's vertex
count and the recording does not match the mesh. The line under the
buttons then shows the hint for that object again, and **Capture Deformation**
stays enabled while the object still deforms.

```{figure} ../../images/static_objects/armature_btn_clear_deformation_cache.png
:alt: The same Dynamics Groups panel as in the previous figure. A red box marks the Clear Deformation Cache button, which sits right of the Capture Deformation button.
:width: 500px

The red box marks **Clear Deformation Cache**.
```

## Re-capturing Every Deformation at Once

Capturing object by object is slow once a scene holds several animated
colliders, and a missed one stops the next **Transfer**. The **Deformations**
box on the **Solver** panel has two buttons that work across the whole scene:

- **Re-capture All Deformations** records every deforming Static collider and
  every animated pin in one pass. It runs the collider captures first and the
  pin captures after, because the two read the same evaluated scene and
  cannot run together. A progress bar and an **Abort** button appear below
  the **Transfer** and **Run** buttons in the **Solver** panel while it runs,
  the same as for a single **Capture Deformation**.
- **Clear All Deformations** deletes every recording in the scene: all
  Static-collider recordings, all animated-pin captures, and any recording
  left behind by an object that was deleted or taken out of its group. The
  objects keep their armatures, lattices and Shape Keys, so **Re-capture All
  Deformations** rebuilds what it removed.

Use the per-object buttons when one object's animation has changed. Use these
two after a change that touches many objects, or when you are not sure which
recordings are current.

Both buttons are disabled while a capture or a bake is running. **Re-capture
All Deformations** is also grayed out when nothing in the scene needs a
capture (no capturable Static collider and no capturable animated pin), and
**Clear All Deformations** when the scene holds no recording to clear.

## Contact Parameters

A Static group shows only the contact-related material settings. Density,
Young's modulus, Poisson ratio, bend, shrink, strain limit, inflate, stitch,
plasticity and velocity overwrite are not drawn.

| UI label                             | Default | Description                                                                  |
| ------------------------------------ | ------- | ---------------------------------------------------------------------------- |
| **Friction**                         | 0.5     | Friction coefficient of the collider surface, from 0 to 1.                   |
| **Apply Soft Constraints**           | off     | Hold the collider with springs instead of locking it to its animation.       |
| **Stiffness**                        | 10.0    | How firmly those springs hold. Drawn only while **Apply Soft Constraints** is on. The minimum is 0.000001, and the slider covers 0.001 to 1000. |
| **Use Object Bounding Box Diagonal** | on      | On: contact distances are ratios of each object's own bounding-box diagonal. Off: they are absolute distances. |
| **Contact Gap Ratio**                | 0.001   | Contact gap as a fraction of each object's own bounding-box diagonal. Drawn while the box above is on. |
| **Contact Offset Ratio**             | 0.0     | Contact offset as a fraction of each object's own bounding-box diagonal. Drawn while the box above is on. |
| **Contact Gap**                      | 0.001   | Absolute contact gap in Blender units. Drawn while the box above is off.     |
| **Contact Offset**                   | 0.0     | Absolute contact offset in Blender units. Drawn while the box above is off.  |

See [Material Parameters](../params/material.md#contact-gap-absolute-vs-ratio)
for absolute versus ratio contact distances, and the
[`Static`](../params/material.md#material-profiles) profile example there for
a minimal collider material.

The **Allow Intersections** box below these rows is the same on every group
type. See [Allow Intersections](../params/material.md#allow-intersections).
What is specific to Static is when the boxes take effect: a collider's own
boxes reach the solver only while it is animated, soft-constrained, or named
as one end of a cross-stitch. A collider that is none of those is a collision
surface only and ignores the boxes on its group, though they are still drawn.
A simulated object still passes through such a collider when its own group
has **Allow Inter-Object Intersections**, **Allow Inter-Group Intersections**
or **Allow Existing Intersections** on.

**Apply Soft Constraints** matters most for the armature-driven colliders
above. A body rig folds against itself as it moves, and where it closes onto
a garment, a collider that follows its animation exactly pushes the cloth into
a space it cannot leave, and the simulation stops. With springs, the collider
yields where the contact force exceeds the spring force and returns
afterwards. See
[Apply Soft Constraints](../params/material.md#apply-soft-constraints) for how
to choose the stiffness.

A Static group draws no **Local Gap Limit** row. That limit keeps a
deformable mesh from pressing on its own neighbors, and the faces of a
collider never come into contact with each other, so a collider's gap is
used exactly as set, however fine its mesh is. See
[Local Gap Limit](../params/material.md#local-gap-limit).

:::{note}
**Static groups have no collision windows.** The
[Collision Active Duration Windows](object_groups.md#active-collision-windows)
control, which mutes contact on dynamic objects for chosen frame ranges, is
not drawn for Static groups, and an object's **Include** checkbox cannot be
keyframed. Their meshes collide for the entire timeline. To make a Static
collider stop acting mid-shot, move it away with a **Move By** op, or use an
[Invisible Collider](../constraints/colliders.md) with **Active Duration**
instead. The scene's render and viewport visibility settings do not affect
what collides.
:::

## Baking Behavior

Two buttons carry the label **Bake Animation**. The one on the **Solver**
panel processes every assigned object in every active group, Static groups
included. The one inside a group box processes only the object selected in
that group's object list.

After **Fetch All Animation**, every Static collider that moves has fetched
animation, whether its motion comes from its own keyframes, a parent or
constraint, Static ops or a captured deformation. **Bake Animation** turns it
into Shape Keys, as for a dynamic object, and takes the object out of its
group. A collider that never moves has nothing to bake, and the scene-wide
**Bake Animation** still takes it out of its group. **Bake Animation** leaves
the object's own keyframes in place.

See [Baking Animation](../sim/baking.md) for the full bake flow.

## Snap and Merge

Static objects are valid endpoints for **Snap and Merge**. The common case is
snapping a **Shell** garment to a **Static** mannequin. The snap moves the
garment to just outside the contact band of the body and registers the pair
as a merge pair for cross-group stitching. The gap follows the same rule as
every other pair; see [Gap Rules](../constraints/snap_merge.md#gap-rules) and
[Stitching to a Static collider](../constraints/snap_merge.md#stitching-to-a-static-collider).
The operator and its options are on the
[Snap and Merge](../constraints/snap_merge.md) page.

## Python / MCP API

Create a Static group the same way as any other. The import below assumes
the add-on is in the **User Default** repository; for an add-on installed
from a remote repository, see
[Blender Python API: Import](../../integrations/python_api.md#import).

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

floor = solver.create_group("Floor", type="STATIC")
floor.add("Ground")
floor.param.friction = 0.8
```

**Property keys.** Each setting in [Contact parameters](#contact-parameters)
has a key. The table shows where each key can be set.

| UI label                             | Key                               | Where it works                                           |
| ------------------------------------ | --------------------------------- | -------------------------------------------------------- |
| **Friction**                         | `friction`                        | `group.param`, a material profile, MCP                   |
| **Use Object Bounding Box Diagonal** | `use_group_bounding_box_diagonal` | `group.param`, a material profile, MCP                   |
| **Contact Gap**                      | `contact_gap`                     | `group.param`, a material profile, MCP                   |
| **Contact Offset**                   | `contact_offset`                  | `group.param`, a material profile, MCP                   |
| **Contact Gap Ratio**                | `contact_gap_rat`                 | `group.param`, a material profile, MCP                   |
| **Contact Offset Ratio**             | `contact_offset_rat`              | `group.param`, a material profile, MCP                   |
| **Apply Soft Constraints**           | `enable_soft_constraint`          | A material profile, MCP (not a `group.param` attribute)  |
| **Stiffness**                        | `soft_constraint_stiffness`       | A material profile, MCP (not a `group.param` attribute)  |
| **Allow Self-Intersections**         | `allow_self_intersection`         | MCP only                                                 |
| **Allow Inter-Object Intersections** | `allow_inter_object_intersection` | MCP only                                                 |
| **Allow Inter-Group Intersections**  | `allow_inter_group_intersection`  | MCP only                                                 |
| **Allow Existing Intersections**     | `allow_existing_intersection`     | MCP only                                                 |

The MCP tool for all of them is `set_group_material_properties`.

**Static ops.** The fluent `solver` API has no Static-op methods. Use
Blender's operators or the MCP handlers.

*Blender operators* press the same buttons as the **Transform** box, one op
per call. `group_index` is the group's slot, from 0 to 31, which is not its
position in the panel; the group proxy returned by `solver.create_group`
reports it as `.slot`. The operators act on the object row selected in the
group's object list, and a script selects the row by setting
`assigned_objects_index` on the group (0 is the first row, in the order the
objects were added). With no row selected, `add_static_op` fails with the
error *"No assigned object selected"*.

```python
import bpy

root = bpy.context.scene.zozo_contact_solver
slot = floor.slot  # the group created above
group = getattr(root, f"object_group_{slot}")
group.assigned_objects_index = 0  # the row of "Ground"

bpy.ops.object.add_static_op(group_index=slot, op_type="MOVE_BY")
bpy.ops.object.add_static_op(group_index=slot, op_type="SPIN")  # new top row, selected
bpy.ops.object.move_static_op(group_index=slot, direction=1)  # selected op down one row
bpy.ops.object.remove_static_op(group_index=slot)  # remove the selected op
```

A new op is added at the top of the list and becomes the selected op.
`remove_static_op` and `move_static_op` act on the selected op, and
`direction` is `-1` to move it up and `1` to move it down. The operators add,
remove and reorder ops with their default fields; to set an op's fields from
a script, use the MCP handlers below.

*MCP handlers* identify the object by name and an op by its zero-based
position in the list, in the order `list_static_ops` reports. In the
signatures below, `...` stands for a value you supply and `|` separates the
arguments that belong to each op type. In `add_static_op` and
`set_static_op`, every argument after `op_type` (or `index`) is optional: an
omitted one keeps the value the op already has, or its default for a new op.

```
add_static_op(group_uuid, object_name, op_type,
              frame_start=..., frame_end=..., transition=...,
              delta=[x,y,z] | spin_axis=[x,y,z], spin_angular_velocity=deg_per_s
                                                | scale_factor=f)
set_static_op(group_uuid, object_name, index,
              frame_start=..., frame_end=..., transition=...,
              delta=... | spin_axis=..., spin_angular_velocity=... | scale_factor=...)
move_static_op(group_uuid, object_name, index, new_index)
remove_static_op(group_uuid, object_name, index)
list_static_ops(group_uuid, object_name)
clear_static_ops(group_uuid, object_name)
```

`add_static_op` puts the new op at the top of the list, as the **+** menu
does. `set_static_op` changes fields in place and keeps the list order, so
use it instead of removing and re-adding an op. `move_static_op` changes the
order, and therefore the motion. See the
[MCP Tool Reference](../../integrations/mcp_reference.rst) for the full
signatures.

*Ops reference:*

| Op        | Fields                                         | Notes                                                     |
| --------- | ---------------------------------------------- | --------------------------------------------------------- |
| `MOVE_BY` | `delta` (x, y, z)                              | Translates the object by `delta` over the range.          |
| `SPIN`    | `spin_axis` (x, y, z), `spin_angular_velocity` | Rotates about the object origin around `spin_axis`, in degrees per second. |
| `SCALE`   | `scale_factor`                                 | Uniform scale about the object origin; below 1 shrinks, above 1 grows. |

Every op also has `frame_start`, `frame_end`, `transition` (`LINEAR` or
`SMOOTH`) and `show_overlay` (the eye icon). `transition` shapes `MOVE_BY`
only; `SCALE` ramps linearly and `SPIN` turns at a constant rate.

**Blender animation and captures.** For the Blender-animation route there is no add-on API. Animate the object as
you normally would (<kbd>I</kbd> in the viewport, the Graph Editor, a parent,
a constraint, a driver, or `obj.keyframe_insert(data_path="location",
frame=...)` from Python) and the add-on reads it at **Transfer**.

A script that records deformations must finish the captures before
**Transfer**:

- MCP: `capture_static_deformation(group_uuid, object_name)`,
  `clear_static_deformation(group_uuid, object_name)`,
  `get_static_deformation_status(group_uuid, object_name)`,
  `recapture_all_deformations()` and `clear_all_deformations()`.
- Python: `solver.recapture_all_deformations()`,
  `solver.clear_all_deformations()` and `solver.is_capture_running()`.

Captures advance on Blender's event loop. A call returns once the first
capture has started, so poll the status (or `solver.is_capture_running()`
from a timer) before you transfer. Blender started with `--background` runs
no capture.

:::{admonition} Under the hood
:class: toggle

**Mutual exclusion**

The encoder checks each Static object in this order, and the first match applies:

1. The object has a populated deformation cache (the recording written by
   **Capture Deformation**). It is sent as `static_deform_animation`, and the
   per-frame transform is replaced by identity. The cache already includes any
   rigid parent motion, so sending `transform_animation` or `static_ops` as
   well would count that motion twice.
2. Its mesh changes shape (a deforming modifier stack, Shape Key animation, or
   a change in its local vertex positions across the frame range) and there is
   no cache. The upload is refused with an explicit error, since sending it as
   a rest-pose collider would hide the missing deformation.
3. Something can move the object as a whole (its own transform keyframes, or a
   parent, constraint, driver or NLA strip up its parent chain) and its world
   transform changes over the solve. The add-on samples `(translation,
   quaternion, scale)` at every frame of the solve, with a linear segment
   between consecutive samples, and sends them as `transform_animation`. All
   moving colliders are sampled in one pass over the frames. A sheared world
   matrix is refused, and so are `static_ops` on an object moved by anything
   other than its own keyframes.
4. The object's entry in the group has Static ops. They are sent as
   `static_ops`, with frames as offsets from the starting frame and axes
   converted to the solver's orientation.
5. Otherwise the object is sent with no animation: a rigid, unmoving collider.

This is why the **Transform** box warns that ops will be ignored as soon as
an object has keyframes (the ops are dropped once the keyed motion changes the
object's world transform during the solve), and why **Capture Deformation** is
required, not just helpful, for an armature-driven collider. Sampling every frame, instead of the
keys, makes the solver's motion match Blender's at every frame the solve
reaches, whatever the animation is built from. The one motion a per-frame
sample cannot tell from its opposite is a turn of more than 180 degrees within
a single frame, which reads as the shorter turn the other way.

**Time conversion**

Frame values in the UI and in MCP are Blender frames. Simulated time zero is
the resolved **Starting Frame**: the **Starting Frame** field, or the Blender
scene's start frame while **Take Starting Frame from Scene** is on. A frame
maps to solver seconds as `(frame - starting frame) / fps`, where `fps` is the
effective FPS (the scene's frame rate, or the **FPS** field while **Take FPS
from Scene** is off) multiplied by **Time Scale**; see
[Scene Parameters](../params/scene.md). Static ops travel as frame offsets
from the starting frame, so an op that starts before the **Starting Frame** is
refused at **Transfer** instead of being clipped. The solver derives seconds
from the parameter payload's FPS, so changing **Time Scale** or the effective
FPS needs only **Update Params on Remote** and no geometry transfer.
Every op field (**Start**, **End**, **Transition**, **Delta**, **Axis**,
**Angular Velocity** and **Factor**), the list of ops itself, and the
**Starting Frame** are stored with the geometry, so changing any of them needs
a full **Transfer**.

**Where ops are stored**

Static ops belong to the object's entry in its group, not to the group or the
mesh. That is why the **Transform** box shows one ops list per selected
object. Removing an object from the group, or clearing its **Include** box,
drops its ops from the next transfer and leaves every other object in the
group alone.
:::
