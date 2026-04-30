# Scene and object groups

How the solver sees your Blender scene: as a collection of object groups that carry type, material, and assigned meshes, plus a Static variant for non-deforming colliders and props.

## Overview

How the solver sees your Blender scene: as a collection of object groups that carry type, material, and assigned meshes, plus a Static variant for non-deforming colliders and props. Sub-pages: object_groups, static_objects.

WARNING: All objects must not intersect and must not be self-intersecting at the start of the simulation. The solver rejects rest geometry where any face penetrates another face (on the same object or on a different one), and it cannot recover from a start state that is already inside out. Before Transfer, confirm that every mesh is cleanly separated from every other mesh and that no mesh folds through itself. This rule applies to every object type: Solid, Shell, Rod, and Static.

Accepted geometry:

- Mesh objects may contain triangles, quads, or any mix of the two; n-gons are not supported, so triangulate those before assigning.
- Solid groups are tetrahedralized internally by fTetWild (https://github.com/wildmeshing/fTetWild), which is tolerant of input quality: the surface does not strictly need to be a closed manifold, and small cracks or near-duplicate vertices are handled automatically.
- Rod groups additionally accept Bezier curve objects, resampled along arc length at transfer time.

## Object Groups

The solver treats your scene as a collection of up to 32 object groups. Each group carries a type, a material model, a set of material parameters, and a list of assigned Blender mesh objects. Only active groups (ones you've created) are sent to the solver.

### The Dynamics Groups panel

Everything you do with groups happens on the Dynamics Groups panel in the 3D viewport's N-panel. The panel is laid out top-to-bottom:

- A Create Group button at the very top.
- One group box per active group, stacked below.

Each group box contains, in order:

1. A header row with the group name, an icon representing the group type, a duplicate-group button, and a delete-group button.
2. An Assigned Objects list of the Blender objects belonging to this group, each with an Include checkbox.
3. A Pins section with the pin vertex groups attached to this group (see Pins and Operations).
4. A Material Params box (see Material Parameters).
5. A Bake row with per-group Bake Animation / Bake Single Frame controls (see Baking Animation).

Figure: The Dynamics Groups panel with one default group. The Create Group button sits at the top; below it, each group box follows the header, Assigned Objects, Pins, Material Params, and Bake layout.

### Creating a group

Click Create Group at the top of the panel. A new group box is inserted beneath the button with a default name (`Group 1`, `Group 2`, and so on), the default Solid type, an empty object list, and default material parameters. The panel auto-scrolls so the new box is visible.

Figure: The panel right after a fresh Create Group click: a single `Group 1` box with the default Solid type, an empty object list, and default material parameters.

### The four group types

| Type       | Description                       | Default model    | Available models                           |
| ---------- | --------------------------------- | ---------------- | ------------------------------------------ |
| **Shell**  | Thin surfaces (cloth, fabric)     | Baraff-Witkin    | Baraff-Witkin, Stable NeoHookean, ARAP     |
| **Solid**  | Volumetric bodies                 | ARAP             | Stable NeoHookean, ARAP                    |
| **Rod**    | 1D structures (ropes, wires)      | ARAP             | ARAP only                                  |
| **Static** | Non-deforming collision objects   | N/A              | N/A                                        |

The type controls which material parameters are relevant and which material models are available. Static groups collapse to just Friction and Contact rows and replace the pin region with a Transform sub-box that holds per-object Move By / Spin / Scale ops (an alternative to Blender transform keyframes). See Static Objects for the full surface.

Rod groups additionally accept Blender curve objects: Bezier curves are resampled along their arc length into rod vertices when transferred.

Figure: Reference matrix of the four types. Shell (green), Solid (red), Rod (yellow), Static (blue). Accepted object types: Shell/Solid/Static take meshes; Rod takes meshes plus Bezier curves. Default material model: Baraff-Witkin for Shell, ARAP for Solid and Rod, none for Static. Available models: Shell offers Baraff-Witkin / Stable NeoHookean / ARAP; Solid offers Stable NeoHookean and ARAP; Rod offers ARAP only; Static none. Density unit: kg/m² for Shell, kg/m³ for Solid, kg/m for Rod. Young's Modulus: Shell/Solid/Rod, not Static. Poisson's Ratio: Shell and Solid. Bend Stiffness: Shell (Rod inherits). Shrink: Shell anisotropic X/Y, Solid uniform, Rod/Static none. Strain Limit: Shell and Rod. Inflate: Shell only. Friction and Contact Gap: all four. Pin storage: Blender vertex groups for Shell/Solid, internal `_pin_name` custom property on curves for Rod, none for Static (uses a Transform sub-box). Default overlay colors: green, red, yellow, blue. Static is the thinnest column because the solver only uses it for collision; no material model and no parameters beyond Friction and Contact Gap. The Material Params box reshapes itself automatically to match the column.

WARNING: You can have at most 32 active groups in a scene. If you need more, fold objects with similar materials into a shared group; there's no cost to many objects sharing one group.

### Assigning objects

To put objects into a group:

1. Select one or more objects in the viewport.
2. In the target group's box, click Add Selected Objects.

The button only accepts mesh objects, plus curve objects for Rod groups. Anything else in your current selection is skipped silently.

The add-on also reports user-visible warnings in three situations:

- Wrong type. The object type is incompatible with the group's type. The object is skipped and a warning appears at the bottom of the Blender window.
- Already assigned. The object is already in this group (silent skip) or already in another active group (warning; you must remove it from the first group before re-assigning).
- Library-linked. The object is a library-linked datablock. These cannot be assigned; an explicit warning is raised. Make the object local first.

When an assignment succeeds, the add-on enables the object's Wireframe and All Edges viewport overlays and, if the group has overlay color enabled, tints its viewport color to the group's color.

IMPORTANT: The solver sees the base mesh, not the modifier-evaluated mesh. The encoder reads `obj.data.vertices` directly, so any Subdivision Surface, Bevel, Remesh, Solidify, or other modifier on the object's stack is ignored at transfer time. If the subdivided or beveled topology is what you actually want to simulate, Apply those modifiers first (`Ctrl`-`A` → Visual Geometry to Mesh, or modifier header → Apply) and then Transfer. The one exception is the ContactSolverCache MESH_CACHE modifier the add-on installs after Fetch. It sits in the first modifier slot and deforms the rest mesh before any other deformer runs, so modifiers you add after simulation (a Subdivision Surface for smooth render, for example) stack on top of the simulated result without interfering with it.

#### Assigned Objects list

Each entry in the Assigned Objects list shows the object name and a small Include checkbox column. Unchecking the checkbox keeps the object in the list for later re-use but excludes it from the next transfer to the solver. This is handy for A/B-testing scenes without dismantling the group.

To remove an object from a group, select it in the Assigned Objects list and click Remove. Removing an object resets its viewport color to white and strips out any pin vertex groups attached to it.

Figure: The Assigned Objects list for a Shell group with one `Cloth` mesh assigned. The left checkbox is the Include flag; unchecking excludes the object from the next transfer without removing it from the group.

### Active collision windows

For Solid, Shell, and Rod groups, contact detection on each assigned object can be restricted to specific frame ranges. The control lives inside the Material Params box (just below the Contact Gap rows) as the Collision Active Duration Windows toggle.

When the toggle is off (the default), every assigned object collides for the full timeline. Switching it on reveals a per-object editor:

- An Object dropdown picks which of the group's assigned objects to edit.
- Below it sits a list of Active Windows for that object: frame ranges during which contact is enabled. Up to 8 windows per object.
- The + / − buttons add and remove entries; new windows default to frames 1 to 60.
- Selecting a window exposes Start and End spinners that edit its bounds.

Outside every window listed for an object, that object's contact is muted. A few rules worth knowing:

- No windows on an object means contact is always active. Flipping the toggle on by itself does not disable anything; you must add at least one window on a given object for the cutoff to apply to it.
- Invisible colliders are not affected. Walls and spheres always collide, even when surrounding deformables have collision windows set. Invisible colliders have their own per-collider Active Duration field; see Invisible Colliders.
- Static groups have no collision windows. Static objects collide for the full timeline; if you need them to come and go mid-shot, animate their visibility or use their Transform ops (see Static Objects).

Typical uses are timed catch-and-release setups (a hand grabs a piece of cloth, then lets go without re-engaging) and sequential drops where a stack of objects should only start interacting once each lower piece has settled.

### Overlay colors

In the group header there's a small color-swatch control. Clicking it opens Blender's standard color picker; picking a color immediately tints every assigned object's viewport outline to that color, so you can see at a glance which objects belong to which group in the viewport.

A checkbox next to the swatch toggles the overlay on and off. When off, the assigned objects return to their original viewport colors; when on, they re-tint to whatever the swatch currently holds.

Figure: Top rows of a Shell group box. The Overlay Color checkbox on the left and the color swatch on the right control the viewport tint; clicking the swatch opens Blender's color picker. The duplicate icon sits in the top-right of the header row.

Figure: Overlay colors in action. The Cloth object is assigned to a Shell group whose swatch is set to green. With solid shading's color mode set to Object (which the add-on flips on when an assignment succeeds), the cube reads green in the viewport. Multiple groups picking distinct swatches lets you spot which object belongs where without clicking each one.

Each group gets a default overlay color based on its type:

| Type       | Default RGB           |
| ---------- | --------------------- |
| **Solid**  | red `(0.75, 0, 0)`    |
| **Shell**  | green `(0, 0.75, 0)`  |
| **Rod**    | yellow `(0.75, 0.75, 0)` |
| **Static** | blue `(0, 0, 0.75)`   |

### Duplicating a group

The duplicate icon in the group header spawns a new group with the same material parameters as the source. Objects, pins, and pin operations are not copied; the new group is empty. The name auto-increments: `Silk` → `Silk-1`, `Silk-1` → `Silk-2`.

Figure: The Duplicate icon lives in the top-right of each group's header row (right of the group name). Clicking it clones only the material parameters into a fresh group slot; the new group has no assigned objects, no pins, and no pin operations.

### Deleting groups

- Delete Group (the trash icon in the group's own box, below the Bake row) resets a single group slot. Assigned object colors are restored to white and any pin vertex groups are cleaned up.
- Delete All Groups (top row of the panel, next to Create Group) iterates all 32 slots and resets each active one. The UI confirms before proceeding.

Figure: Delete All Groups sits in the panel's header row, paired with Create Group. The per-group Delete Group button is visible below inside each group box (same trash icon, at full width).

### Blender Python API

The same workflow is available from Python. The API mirrors the UI one-for-one: groups are created through `solver.create_group`, objects are assigned with `group.add`, and every operation on a group returns the group itself for easy chaining.

```python
from zozo_contact_solver import solver

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

UNDER THE HOOD:

32-slot model. The 32 slots are fixed properties on the scene (`object_group_0` through `object_group_31`), not a dynamic collection. Creating a group takes the first inactive slot; deleting one frees the slot. This means:

- Groups survive file save / load without reassignment.
- The underlying slot index is not the same as the display number: the UI renumbers visible groups sequentially (1, 2, 3, and so on) even if the underlying slots are 0, 3, 7.

Library-linked rejection. Library-linked objects are rejected because the add-on stamps a persistent UUID on each assigned object, and library overrides are not writable at the level the UUID lives on.

## Static Objects

A Static object group holds meshes that do not deform in the solver: colliders, ground planes, mannequins, props, anything that should influence the simulation without being simulated itself. Static groups share the same panel, transfer, and bake flow as Solid / Shell / Rod groups, but with a smaller material-parameter surface and two exclusive ways to drive motion.

This page centralizes everything specific to the Static type:

- Creating a Static group
- Moving a Static object: the two ways (Static Ops vs Blender keyframes) and the rule that picks one
- The Transform sub-box: where Static ops live in the UI
- Contact parameters: the small parameter set a Static group exposes
- Baking behavior
- Snap and merge
- Python / MCP API

### Creating a Static group

Click Create Group on the Dynamics Groups panel, then change Type to Static. The group card updates to reflect the Static surface:

- The Assigned Objects list accepts only meshes (curves are rejected because they only make sense for Rod groups).
- The pin region is relabeled Transform (with a driver icon replacing the pin icon). No vertex-group pins are possible on a Static group; instead this box holds per-object Static ops.
- The Material Params box collapses down to just Friction and the Contact rows.
- The default overlay color is blue `(0, 0, 0.75)`.

Everything else (duplicating the group, per-object Include checkboxes, removing objects, deleting the group) works the same as for Solid/Shell/Rod groups. See Object Groups for the shared UI surface.

### Moving a Static object

Static meshes don't deform, but they can translate, rotate, or scale over time. There are two mutually exclusive ways to drive that motion:

1. Static ops. UI-assigned Move By / Spin / Scale entries edited per assigned object inside the group's Transform sub-box. These are declarative (time range + delta/axis/factor) and shipped as a list to the solver at transfer time.
2. Blender transform keyframes. The usual way you animate an object in Blender: select it, hit I on a frame, and pick Location, Rotation, or Scale. Any keyframes you set on the object's own transform channels get picked up automatically. The add-on samples the world transform at each keyframe and ships the track, including Bezier handles, so eases you see in the Graph Editor carry over to the simulation.

WARNING: Only one source of motion per object at a time. If an assigned Static mesh has any transform fcurve, the encoder sends its keyframed transform and ignores that object's Static ops list. The UI flags this up-front with an error label above the ops list, "Object has Blender keyframes; these ops will be ignored", and the Add-Op operator emits the same warning when you add a new op. The ops stay in the list; they just don't take effect until you remove the fcurves.

NOTE: Shape keys / mesh-level animation are not supported on Static objects. Only object-level transform animation counts. The encoder raises a `RuntimeError` at transfer time if a Static mesh has an action on its mesh datablock (shape-key fcurves), rather than silently dropping the animation.

Use Static ops when the motion is scripted and easy to describe with a few time ranges: a sliding floor plate that moves from A to B between frame 30 and 60, a spinning turntable, a shrinking platform. Use Blender keyframes when the motion lives in Blender's own timeline already, typically a rigged mannequin driven by an armature or a prop animated by hand in the Graph Editor.

### The Transform sub-box

On a Static group, the region that would be Pins on other group types is relabeled Transform. Expanding it shows:

1. A per-assigned-object list: the same object list as the group card above. Which object you select here determines which object's static-ops list is being edited in the box below.
2. A warning row: visible only when the selected object has Blender transform fcurves; tells you its ops will be ignored.
3. The static-ops list: `Move By` / `Spin` / `Scale` entries with `+` (add menu), `−` (remove), and up/down reorder buttons.
4. The per-op editor with fields for the active row:
   - Start / End: Blender frames; the op is active across the closed range.
   - Transition: `Linear` or `Smooth` (smoothstep).
   - Delta (m): Move By only; `(x, y, z)` translation in world units.
   - Axis / Angular Velocity (°/s): Spin only; pivots around the object's origin.
   - Factor: Scale only; uniform scale multiplier around the object's origin.

Ops compose in list order: if you stack a `Move By` and a `Spin` whose time ranges overlap, the object translates and rotates simultaneously inside the overlap. Outside every op's time range the object rests at its un-modified transform (the pose it was at when you assigned it to the group).

#### Static ops reference

| Op         | Fields                                         | Pivot            | Notes                                                   |
| ---------- | ---------------------------------------------- | ---------------- | ------------------------------------------------------- |
| `MOVE_BY`  | `delta` (x, y, z)                              | N/A              | Translate the whole object by `delta` over the range.   |
| `SPIN`     | `spin_axis` (x, y, z), `spin_angular_velocity` | Object origin    | Rotate around `spin_axis` at `°/s`.                     |
| `SCALE`    | `scale_factor`                                 | Object origin    | Uniform scale; `< 1` shrinks, `> 1` grows.              |

Common fields on every op: `frame_start`, `frame_end`, `transition` (`LINEAR` / `SMOOTH`), and `show_overlay` (toggle the viewport preview).

### Contact parameters

Static groups expose only the contact-relevant subset of material parameters. Everything deformation-related (density, Young's modulus, Poisson ratio, bend, shrink, strain limit, inflate, stitch, plasticity, velocity overwrite) is hidden.

| UI label                             | Python / TOML key                 | Default | Description                                                           |
| ------------------------------------ | --------------------------------- | ------- | --------------------------------------------------------------------- |
| **Friction**                         | `friction`                        | 0.5     | Coulomb friction coefficient between this mesh and other groups.      |
| **Contact Gap**                      | `contact_gap`                     | 0.001   | Absolute contact gap distance, in Blender units.                      |
| **Contact Offset**                   | `contact_offset`                  | 0.0     | Absolute contact offset, in Blender units.                            |
| **Use Group Bounding Box Diagonal**  | `use_group_bounding_box_diagonal` | `True`  | When true, contact distances are ratios of the group's bbox diagonal. |
| **Contact Gap Ratio**                | `contact_gap_rat`                 | 0.001   | Contact gap as a fraction of the group's bounding-box diagonal.       |
| **Contact Offset Ratio**             | `contact_offset_rat`              | 0.0     | Contact offset as a fraction of the group's bounding-box diagonal.    |

See Material Parameters for the full story on absolute vs ratio contact gap, and `Static` in the shipped material-profile TOML for a minimal example.

NOTE: Static groups have no collision windows. The Collision Active Duration Windows control, which mutes contact on dynamic objects for chosen frame ranges, is not exposed for Static groups; their meshes collide for the entire timeline. If you need a Static collider to come and go mid-shot, animate its visibility, drive it out of the way with a Static op, or use a per-collider Active Duration on an Invisible Collider instead.

### Baking behavior

Bake Animation walks through active groups in slot order (`object_group_0` → `object_group_31`) and processes every assigned object. Static groups are included: if a Static collider was driven by Blender transform keyframes and therefore carried a `ContactSolverCache` modifier and `.pc2` file after a Fetch, both are cleaned up during bake even though the Static object itself has no simulated deformation. Bake never touches object-level transform fcurves, only the per-frame mesh cache.

If the Static object is driven by Static ops (no fcurves), there is nothing to bake on it; the motion lives on the solver side, and re-running Transfer + Run produces the same motion deterministically. Bake is only meaningful for Static objects that have fetched per-frame vertex data.

See Baking Animation for the full bake flow.

### Snap and merge

Static objects are valid endpoints for Snap and Merge. The common case is snapping a Shell garment to a Static mannequin so the cloth's nearest vertices touch the body before the solve begins; the pair is then registered as a merge pair for cross-group stitching, with the contact gap picked from the Shell ↔ Static pairing. See Snap and Merge for the operator and its options.

### Python / MCP API

Create a Static group the same way as any other:

```python
from zozo_contact_solver import solver

floor = solver.create_group("Floor", type="STATIC")
floor.add("Ground")
floor.param.friction = 0.8
```

Static ops are not yet on the fluent `solver` surface; drive them either through Blender's raw operators or through the MCP handlers.

Blender operators (one op per call; the `group_index` is the slot from 0 to 31, not the UI display number):

```python
import bpy

bpy.ops.object.add_static_op(group_index=0, op_type="MOVE_BY")
bpy.ops.object.remove_static_op(group_index=0)
bpy.ops.object.move_static_op(group_index=0, direction=-1)  # reorder up
```

The operators edit whichever assigned object is currently selected in the group's assigned-objects list; set `group.assigned_objects_index` first to pick a specific object.

MCP handlers (identify the object by name, not by list index):

```
add_static_op(group_uuid, object_name, op_type,
              frame_start=..., frame_end=..., transition="LINEAR",
              delta=[x,y,z] | spin_axis=[x,y,z], spin_angular_velocity=deg_per_s
                                                | scale_factor=f)
remove_static_op(group_uuid, object_name, index)
list_static_ops(group_uuid, object_name)
clear_static_ops(group_uuid, object_name)
```

See the MCP Tool Reference for the full signatures.

For the Blender-keyframe route there is no add-on-specific API at all. Key the object's transform in Blender as you normally would (`I` in the viewport, the Graph Editor, constraints baked to fcurves, or `obj.keyframe_insert(data_path="location", frame=...)` from Python) and the encoder picks it up at Transfer time.

UNDER THE HOOD:

Mutual exclusion. The encoder checks each Static object in order:

1. If `obj.data.animation_data` has any fcurve (shape-key animation), raise `RuntimeError`. This case is unsupported, and silently dropping it would mislead the user.
2. Else if `obj.animation_data.action` has any transform fcurve, extract sparse `(time, translation, quaternion, scale)` keyframes plus per-segment Bezier-handle data and send them as `transform_animation`.
3. Else if the matching `AssignedObject` has a non-empty `static_ops` collection, serialize those ops (frames converted to seconds using the scene FPS, axes swapped into solver orientation) as `static_ops`.
4. Else send the object with no animation: a rigid, unmoving collider.

The first match wins; the other channels are dropped. This is why the UI can warn "these ops will be ignored" as soon as fcurves appear on the object.

Time conversion. Frame values in the UI and MCP handlers are 1-based Blender frames. The encoder converts to seconds as `(frame - 1) / fps`, using the scene's effective FPS (`scene.render.fps` or the add-on's `frame_rate` override; see Scene Parameters).

Assigned-object wiring. Static ops live on the `AssignedObject` record, not on the group or the mesh. That is why the Transform sub-box shows one ops list per selected object; deleting an object from the group (or unchecking its Include box) drops its ops from the next transfer without touching any other object in the group.
