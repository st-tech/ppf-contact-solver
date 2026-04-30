# 🗿 Static Objects

A **Static** object group holds meshes that do not deform in the solver:
colliders, ground planes, mannequins, props, anything that should
influence the simulation without being simulated itself. Static groups
share the same panel, transfer, and bake flow as **Solid** / **Shell** /
**Rod** groups, but with a smaller material-parameter surface and two
exclusive ways to drive motion.

This page centralizes everything specific to the Static type:

- [Creating a Static group](#creating-a-static-group)
- [Moving a Static object](#moving-a-static-object): the two ways
  (Static Ops vs Blender keyframes) and the rule that picks one
- [The Transform sub-box](#the-transform-sub-box): where Static ops
  live in the UI
- [Contact parameters](#contact-parameters): the small parameter set a
  Static group exposes
- [Baking behavior](#baking-behavior)
- [Snap and merge](#snap-and-merge)
- [Python / MCP API](#python--mcp-api)

## Creating a Static Group

Click **Create Group** on the **Dynamics Groups** panel, then change
**Type** to **Static**. The group card updates to reflect the Static
surface:

- The **Assigned Objects** list accepts only meshes (curves are rejected
  because they only make sense for **Rod** groups).
- The pin region is relabeled **Transform** (with a driver icon
  replacing the pin icon). No vertex-group pins are possible on a
  Static group; instead this box holds per-object
  [Static ops](#the-transform-sub-box).
- The **Material Params** box collapses down to just **Friction** and
  the **Contact** rows (see [Contact parameters](#contact-parameters)).
- The default overlay color is blue `(0, 0, 0.75)`.

Everything else (duplicating the group, per-object **Include**
checkboxes, removing objects, deleting the group) works the same as
for Solid/Shell/Rod groups. See
[Object Groups](object_groups.md) for the shared UI surface.

## Moving a Static Object

Static meshes don't deform, but they can translate, rotate, or scale
over time. There are **two mutually exclusive** ways to drive that
motion:

1. **Static ops**. UI-assigned **Move By** / **Spin** / **Scale**
   entries edited per assigned object inside the group's
   [Transform sub-box](#the-transform-sub-box). These are declarative
   (time range + delta/axis/factor) and shipped as a list to the solver
   at transfer time.
2. **Blender transform keyframes**. The usual way you animate an
   object in Blender: select it, hit <kbd>I</kbd> on a frame, and pick
   **Location**, **Rotation**, or **Scale**. Any keyframes you set on
   the object's own transform channels get picked up automatically.
   The add-on samples the world transform at each keyframe and ships
   the track, including Bezier handles, so eases you see in the Graph
   Editor carry over to the simulation.

:::{warning}
**Only one source of motion per object at a time.** If an assigned
Static mesh has any transform fcurve, the encoder sends its keyframed
transform and **ignores** that object's Static ops list. The UI flags
this up-front with an error label above the ops list,
*"Object has Blender keyframes; these ops will be ignored"*, and the
Add-Op operator emits the same warning when you add a new op. The ops
stay in the list; they just don't take effect until you remove the
fcurves.
:::

:::{note}
**Shape keys / mesh-level animation are not supported on Static
objects.** Only object-level transform animation counts. The encoder
raises a `RuntimeError` at transfer time if a Static mesh has an action
on its mesh datablock (shape-key fcurves), rather than silently
dropping the animation.
:::

Use **Static ops** when the motion is scripted and easy to describe
with a few time ranges: a sliding floor plate that moves from A to B
between frame 30 and 60, a spinning turntable, a shrinking platform. Use
**Blender keyframes** when the motion lives in Blender's own timeline
already, typically a rigged mannequin driven by an armature or a prop
animated by hand in the Graph Editor.

## The Transform Sub-Box

On a Static group, the region that would be **Pins** on other group
types is relabeled **Transform**. Expanding it shows:

1. **A per-assigned-object list**: the same object list as the group
   card above. Which object you select here determines which object's
   static-ops list is being edited in the box below.
2. **A warning row**: visible only when the selected object has
   Blender transform fcurves; tells you its ops will be ignored.
3. **The static-ops list**: `Move By` / `Spin` / `Scale` entries with
   `+` (add menu), `−` (remove), and up/down reorder buttons.
4. **The per-op editor** with fields for the active row:
   - **Start** / **End**: Blender frames; the op is active across the
     closed range.
   - **Transition**: `Linear` or `Smooth` (smoothstep).
   - **Delta (m)**: *Move By* only; `(x, y, z)` translation in
     world units.
   - **Axis** / **Angular Velocity (°/s)**: *Spin* only; pivots
     around the object's origin.
   - **Factor**: *Scale* only; uniform scale multiplier around the
     object's origin.

Ops compose in list order: if you stack a `Move By` and a `Spin` whose
time ranges overlap, the object translates and rotates simultaneously
inside the overlap. Outside every op's time range the object rests at
its un-modified transform (the pose it was at when you assigned it to
the group).

### Static Ops Reference

| Op         | Fields                                         | Pivot            | Notes                                                   |
| ---------- | ---------------------------------------------- | ---------------- | ------------------------------------------------------- |
| `MOVE_BY`  | `delta` (x, y, z)                              | N/A              | Translate the whole object by `delta` over the range.   |
| `SPIN`     | `spin_axis` (x, y, z), `spin_angular_velocity` | Object origin    | Rotate around `spin_axis` at `°/s`.                     |
| `SCALE`    | `scale_factor`                                 | Object origin    | Uniform scale; `< 1` shrinks, `> 1` grows.              |

Common fields on every op: `frame_start`, `frame_end`, `transition`
(`LINEAR` / `SMOOTH`), and `show_overlay` (toggle the viewport
preview).

## Contact Parameters

Static groups expose only the contact-relevant subset of material
parameters. Everything deformation-related (density, Young's modulus,
Poisson ratio, bend, shrink, strain limit, inflate, stitch, plasticity,
velocity overwrite) is hidden.

| UI label                             | Python / TOML key                 | Default | Description                                                           |
| ------------------------------------ | --------------------------------- | ------- | --------------------------------------------------------------------- |
| **Friction**                         | `friction`                        | 0.0     | Coulomb friction coefficient between this mesh and other groups.      |
| **Contact Gap**                      | `contact_gap`                     | 0.001   | Absolute contact gap distance, in Blender units.                      |
| **Contact Offset**                   | `contact_offset`                  | 0.0     | Absolute contact offset, in Blender units.                            |
| **Use Group Bounding Box Diagonal**  | `use_group_bounding_box_diagonal` | `True`  | When true, contact distances are ratios of the group's bbox diagonal. |
| **Contact Gap Ratio**                | `contact_gap_rat`                 | 0.001   | Contact gap as a fraction of the group's bounding-box diagonal.       |
| **Contact Offset Ratio**             | `contact_offset_rat`              | 0.0     | Contact offset as a fraction of the group's bounding-box diagonal.    |

See [Material Parameters](../params/material.md#contact-gap-absolute-vs-ratio)
for the full story on absolute vs ratio contact gap, and
[`Static`](../params/material.md) in the shipped material-profile TOML for
a minimal example.

:::{note}
**Static groups have no collision windows.** The
[Collision Active Duration Windows](object_groups.md#active-collision-windows)
control, which mutes contact on dynamic objects for chosen frame
ranges, is not exposed for Static groups; their meshes collide for the
entire timeline. If you need a Static collider to come and go mid-shot,
animate its visibility, drive it out of the way with a Static op, or
use a per-collider **Active Duration** on an
[Invisible Collider](../constraints/colliders.md) instead.
:::

## Baking Behavior

**Bake Animation** walks through active groups in slot order
(`object_group_0` → `object_group_31`) and processes every assigned
object. Static groups are included: if a Static collider was driven by
Blender transform keyframes and therefore carried a
`ContactSolverCache` modifier and `.pc2` file after a Fetch, both are
cleaned up during bake even though the Static object itself has no
simulated deformation. Bake never touches object-level transform
fcurves, only the per-frame PC2 data.

If the Static object is driven by Static ops (no fcurves), there is
nothing to bake on it; the motion lives on the solver side, and
re-running Transfer + Run produces the same motion deterministically.
Bake is only meaningful for Static objects that have fetched per-frame
vertex data.

See [Baking Animation](../sim/baking.md) for the full bake flow.

## Snap and Merge

Static objects are valid endpoints for **Snap and Merge**. The common
case is snapping a Shell garment to a Static mannequin so the
cloth's nearest vertices touch the body before the solve begins; the
pair is then registered as a merge pair for cross-group stitching, with
the contact gap picked from the Shell ↔ Static pairing. See
[Snap and Merge](../constraints/snap_merge.md) for the operator and its options.

## Python / MCP API

Create a Static group the same way as any other:

```python
from zozo_contact_solver import solver

floor = solver.create_group("Floor", type="STATIC")
floor.add("Ground")
floor.param.friction = 0.8
```

Static ops are not yet on the fluent `solver` surface; drive them
either through Blender's raw operators or through the MCP handlers.

**Blender operators** (one op per call; the `group_index` is the slot
from 0 to 31, not the UI display number):

```python
import bpy

bpy.ops.object.add_static_op(group_index=0, op_type="MOVE_BY")
bpy.ops.object.remove_static_op(group_index=0)
bpy.ops.object.move_static_op(group_index=0, direction=-1)  # reorder up
```

The operators edit whichever assigned object is currently selected in
the group's assigned-objects list; set `group.assigned_objects_index`
first to pick a specific object.

**MCP handlers** (identify the object by name, not by list index):

```
add_static_op(group_uuid, object_name, op_type,
              frame_start=..., frame_end=..., transition="LINEAR",
              delta=[x,y,z] | spin_axis=[x,y,z], spin_angular_velocity=deg_per_s
                                                | scale_factor=f)
remove_static_op(group_uuid, object_name, index)
list_static_ops(group_uuid, object_name)
clear_static_ops(group_uuid, object_name)
```

See the
[MCP Tool Reference](../../integrations/mcp_reference.rst) for the full
signatures.

For the Blender-keyframe route there is no add-on-specific API at all.
Key the object's transform in Blender as you normally would (`I` in the
viewport, the Graph Editor, constraints baked to fcurves, or
`obj.keyframe_insert(data_path="location", frame=...)` from Python)
and the encoder picks it up at Transfer time.

:::{admonition} Under the hood
:class: toggle

**Mutual exclusion**

The encoder checks each Static object in order:

1. If `obj.data.animation_data` has any fcurve (shape-key animation),
   raise `RuntimeError`. This case is unsupported, and silently
   dropping it would mislead the user.
2. Else if `obj.animation_data.action` has any transform fcurve,
   extract sparse `(time, translation, quaternion, scale)` keyframes
   plus per-segment Bezier-handle data and send them as
   `transform_animation`.
3. Else if the matching `AssignedObject` has a non-empty
   `static_ops` collection, serialize those ops (frames converted
   to seconds using the scene FPS, axes swapped into solver
   orientation) as `static_ops`.
4. Else send the object with no animation: a rigid, unmoving
   collider.

The first match wins; the other channels are dropped. This is why the
UI can warn *"these ops will be ignored"* as soon as fcurves appear on
the object.

**Time conversion**

Frame values in the UI and MCP handlers are 1-based Blender frames.
The encoder converts to seconds as `(frame - 1) / fps`, using the
scene's effective FPS (`scene.render.fps` or the add-on's
`frame_rate` override; see
[Scene Parameters](../params/scene.md)).

**Assigned-object wiring**

Static ops live on the `AssignedObject` record, not on the group or
the mesh. That is why the **Transform** sub-box shows one ops list per
selected object; deleting an object from the group (or unchecking its
**Include** box) drops its ops from the next transfer without touching
any other object in the group.
:::
