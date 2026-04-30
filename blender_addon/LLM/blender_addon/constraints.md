# Constraints: pins, colliders, snap & merge

Shape the motion by pinning vertices, stitching groups together, or introducing parametric collision boundaries that never appear in the scene.

## Overview

Shape the motion by pinning vertices, stitching groups together, or introducing parametric collision boundaries that never appear in the scene. Sub-pages: pins, colliders, snap_merge.

## Pins

A pin is a named set of vertices on an object that the add-on has marked as a constraint target. By itself a pin just holds its vertices in place during the solve. On top of that you stack operations: small, keyframeable drivers that move, rotate, scale, or torque the pinned vertices while the simulation runs.

The pin tells the solver which vertices are constrained. The operations tell the solver how they should move.

Figure: Two-panel reference. Left: tree with root node "Group: Cloth (SHELL)" branching into three pins: ShoulderPins (carrying Spin and Embedded Move operations), CollarPins (carrying Move By), and WaistPins (carrying Torque). Legend lists the five operation types (Move By, Spin, Scale, Torque, Embedded Move) and notes that Torque is exclusive with Move By / Spin / Scale. Right: 2x2 grid of schematics for the four Center modes on Spin and Scale. Centroid places the pivot at the mean of the vertex cluster. Fixed places the pivot at a user-entered world-space coordinate offset from the cluster. Max Towards uses a direction arrow to place the pivot at the centroid of the vertices furthest in that direction. Vertex places the pivot on one specific mesh vertex that deforms along with the mesh. Pins nest inside a group and operations stack on each pin.

### Where pin indices are stored

How the vertex set is stored depends on the object type:

- Meshes. The pin is a Blender vertex group. Any vertex group on the mesh is eligible; the add-on simply registers it by name and reads its members at solve time. You can author the group through Blender's usual weight-paint / vertex-group UI, or let the add-on create one from an Edit-mode selection.
- Curves (Bezier, used by Rod groups). Blender curves have no native vertex groups, so the add-on creates an internal group per pin. The control-point indices are stored as a JSON array in a custom property on the curve object, keyed `_pin_<name>`. You never edit this property by hand; it is written and read through the Create Pin VG button and the Edit-mode select/deselect actions.

### The Pins section in a group box

Each group box on the Dynamics Groups panel contains a Pins section. It sits between the Assigned Objects list and the Material Params box. The section is laid out top-to-bottom as follows:

1. Vertex-group selector row. Two side-by-side dropdown menus appear at the top: the left dropdown lists the mesh objects assigned to this group (e.g. `Shirt`), and the right dropdown lists the vertex groups on the selected object (e.g. `ShoulderPins`). Together they form an `[Object][VertexGroup]` pair that identifies which vertex group you want to register as a pin. The dropdown only enumerates mesh vertex groups, since curve objects have no vertex groups and so never appear here; their pins are authored through Create Pin VG instead.

2. Action buttons row. Directly below the selector sit four buttons arranged left to right:
   - Add Pin VG: registers the vertex group currently shown in the selector dropdowns as a new pin on this group.
   - Remove: deletes the currently-selected pin from the list, including every operation attached to it. Grayed out when no pin is selected.
   - Rename: opens a small dialog prefilled with the selected pin's vertex-group name. Editing the name renames the underlying vertex group (for meshes) or the `_pin_<name>` custom property (for curves), migrates the embedded-move marker, and updates the pin's identifier and hash in one step. Grayed out when no pin is selected.
   - Create Pin VG: only available when you are in Edit Mode with vertices selected. Creates a brand-new vertex group from the selection and registers it as a pin in one step.

3. Pin UIList. Below the buttons is a scrollable list of all pins registered on this group. Each row displays:
   - The object name / vertex-group name label (e.g. `Shirt / ShoulderPins`).
   - An Include checkbox on the right side of the row. When unchecked, the pin is ignored during the solve but remains in the list for later re-enabling.

   The list supports single selection: clicking a row highlights it and opens the selected-pin details panel described below.

#### Creating a pin

Figure: The Pins section of a group box, expanded. The `[Cloth][ShoulderPins]` vertex-group selector, Add / Remove / Rename / Create buttons, the pin UIList with the registered `ShoulderPins` entry, the pin-level fields (Show Pins, Duration, Pull), and the Operations list containing the `Spin ω=360°/s` operation row are all visible.

There are two UI paths to create a pin, both driven from the action buttons above the pin UIList:

1. From an existing vertex group. Use the left dropdown to pick the mesh object, then the right dropdown to pick the vertex group. Click Add Pin VG. The vertex group appears immediately as a new row in the pin UIList with its included checkbox on. This path is mesh-only, since curves have no vertex groups to pick from.
2. From an Edit-mode selection. Enter Edit Mode on a mesh or curve that belongs to this group, select the vertices (or curve control points) you want pinned, and click Create Pin VG. The add-on creates a new group from the current selection (naming it automatically), registers it as a pin with its Include checkbox on, and the new entry appears in the UIList. For meshes this writes a regular Blender vertex group; for curves it writes the internal `_pin_<name>` custom property described above. This is the only way to create a pin on a curve object.

Remove deletes the currently-selected pin, including every operation attached to it. After removal, the selection moves to the next pin in the list, or the list becomes empty.

#### The selected-pin details panel

Selecting a pin in the UIList reveals a details panel below the list with the pin's own properties:

- Duration: a checkbox that enables an Active For frame field beside it. When on, the pin is released at that frame.
- Pull: a checkbox with a Strength field next to it. When on, the pin no longer hard-constrains the vertices; instead, it pulls them toward their target positions as a soft force of the given strength.
- Operations UIList: a list of the operations stacked on this pin, each row showing the operation type.

Pull is mutually exclusive with movement operations, since the solver would have no target to pull toward. The UI reflects this by disabling incompatible controls.

#### Edit-mode pin buttons

When the pinned object is in Edit Mode, two extra rows appear above the pin properties:

- Select / Deselect: flip the selection state of every vertex (or curve control point) in the active pin. Useful for previewing which indices are tagged before running, or for adding/removing members via Blender's regular selection tools.
- Make Keyframe / Delete All Keyframes: drive the Embedded Move operation. Make Keyframe attaches Embedded Move on the first press and records the current posed positions; Delete All Keyframes removes every keyframe and the operation in one step. Curve pins don't accept keyframes, so the buttons are inert on rods.

All four buttons are hidden outside Edit Mode because they act on the Edit-mode selection or on mesh-data animation.

#### Adding an operation

Below the Operations UIList in the selected-pin details panel are buttons arranged in two rows. The top row holds an Operations: label alongside Copy and Paste clipboard icons (copy the pin's operation list to a session-scoped clipboard; paste replaces the target pin's operations wholesale). The bottom row holds:

- Add: opens a dropdown menu listing the available operation types (see below). New operations insert at the top of the list.
- Remove: deletes the currently-selected operation. Removing an Embedded Move row is equivalent to Delete All Keyframes.
- Up / Down triangles (▲ / ▼): reorder the selected operation within the list. Order determines the sequence in which the solver applies each operation's contribution when more than one is stacked on the same pin.

The Add dropdown lists the available operation types:

- Move By: translate the pinned vertices by a delta.
- Spin: rotate the pinned vertices around an axis through a pivot.
- Scale: scale the pinned vertices uniformly from a pivot.
- Torque: apply a rotational force around a PCA-derived axis.

Embedded Move is not in the dropdown: it's attached automatically on the first Make Keyframe press. Entries that would violate compatibility rules (e.g. adding Spin when a Torque already exists) are grayed out in the dropdown so you cannot select them.

Figure: The selected-pin details panel on a Shell group. The four action buttons (Add / Remove / Rename / Create) sit above the pin list. Above the Operations list are the Copy / Paste clipboard icons; below it are Add, Remove, and the ▲ / ▼ reorder triangles. A Spin operation is selected with Center set to Fixed, which exposes the XYZ coordinate fields and the Pick from Selected eyedropper.

Picking an entry inserts a new operation row into the pin's Operations UIList. Each row in that list shows:

- The operation type label (e.g. Spin, Move By).
- A small overlay-visibility toggle (eye icon) on the right side of the row. Clicking it turns the viewport overlay for that operation on or off, useful for previewing pivots or directions without running the solver.

Clicking an operation row selects it and opens the per-operation editor directly below the operations list. The fields shown depend on the operation type.

Figure: The selected-pin details panel with the `Spin ω=360°/s` operation selected in the Operations UIList. Its per-type editor is visible below: the Center dropdown (set to Centroid), the Axis XYZ vector, Angular Velocity (°/s), Flip Direction toggle, Start / End frame range, and the Transition dropdown.

- Move By: Delta (m) (XYZ vector), Start, End, Transition dropdown (Linear / Smooth).
- Spin: Axis (XYZ vector), Angular Velocity (°/s), a Center dropdown (see below), the center-mode's companion field, Start, End, Transition.
- Scale: Factor (scalar), a Center dropdown + companion field, Start, End, Transition.
- Torque: Magnitude (N·m), Axis dropdown (PC1 / PC2 / PC3), Flip Direction checkbox, Start, End.
- Embedded Move: no editable fields; this operation is managed entirely via the Make Keyframe and Delete All Keyframes buttons.

WARNING: Torque cannot coexist with Move By, Spin, or Scale on the same pin. It can coexist with Embedded Move. The Add dropdown enforces this at creation time; incompatible entries are grayed out.

#### Make Keyframe

The Make Keyframe button appears in the edit-mode row (see above) and drives the Embedded Move operation. On the first press for a given pin:

1. It samples the current positions of the pinned vertices at the current scene frame.
2. It attaches an Embedded Move operation to the pin and stores the samples as its first keyframe.

Subsequent presses add more keyframes at the current scene frame without duplicating the operation; only one Embedded Move ever exists per pin. Visibly, each press "bakes in" the current posed shape of the pinned vertices; scrubbing the timeline then plays back the keyframes as the simulation runs.

Delete All Keyframes removes every keyframe and the Embedded Move operation in one step.

#### Center-mode dropdown and overlays

Spin and Scale both rotate or scale around something. The per-operation editor exposes that "something" as a Center dropdown with four modes, each revealing a different companion field underneath:

| Mode             | Companion field           | How the pivot is resolved                                                                       |
| ---------------- | ------------------------- | ----------------------------------------------------------------------------------------------- |
| **Centroid**     | *(none)*                  | Mean of the pinned vertex positions.                                                            |
| **Fixed**        | XYZ coordinate + **Pick from Selected** eyedropper | Fixed world-space point. With the mesh in Edit Mode and one or more vertices selected, the eyedropper writes the selection's world-space centroid into the XYZ field. |
| **Max Towards**  | Unit direction vector     | Centroid of the vertices furthest in that direction.                                            |
| **Vertex**       | Vertex index + **Pick Vertex** eyedropper | A single vertex on the mesh. The eyedropper reads the one selected vertex in Edit Mode; it reports an error if zero or more than one vertex is selected. The pivot deforms with the mesh. |

Alongside each operation are viewport-overlay toggles (Show Max Towards, Show Vertex, and siblings) that draw the computed pivot in the viewport so you can preview it before solving.

#### A Wind-Blown Banner

A minimal end-to-end scene that exercises the pin system: a vertical cloth banner with its top edge pinned, deformed by wind.

1. Build the banner. Add a flat Plane, apply a Simple subdivision at Viewport level 5 (→ a 33x33 grid of vertices, i.e. 32 quads per side), then rotate it 90° around X and apply the rotation so it stands upright in the XZ plane.
2. Create the pin vertex group. Enter Edit Mode, select the top row of vertices (the 33 with the maximum Z), and create a new vertex group named TopEdge.
3. Register with a SHELL group. Leave Edit Mode, open the Dynamics Groups panel, click Create Group, set the type to Shell, then add the plane via Add Selected Objects.
4. Pin the top edge. In the group's Pins section, pick `[ClothBanner][TopEdge]` in the vertex-group selector and click Add Pin VG. Make sure Show Pins is on (it is by default) so the pinned vertices render as white dots in the viewport.

   Figure: The Dynamics Groups panel after steps 3 and 4. `ClothBanner` is the lone assigned object under the Shell group; `[ClothBanner][TopEdge]` is selected in the Pins list; Show Pins is on, Size is 18.

5. Drive the wind. In the Scene Configuration panel, set the Wind direction to `(0, 1, 0)` and Strength to around 20 m/s, then bump Air Density to `0.01 kg/m³` (the max). The force on the cloth scales with air density, so the default `0.001 kg/m³` leaves the banner barely moving.

   Figure: Scene Configuration panel with Air Density 0.01; Wind sub-section expanded with direction `(0, 1, 0)` and strength `20.00 m/s`.

6. Transfer → Run.

Figure: Rest pose at frame 1. The row of white dots along the top edge is the Show Pins overlay drawing each vertex of TopEdge. Those vertices are the ones the solver will hold fixed.

Figure: The same scene at frame 15, a fraction of a second after Run. The pinned vertices along the top, still marked by the overlay dots, have not moved at all; the rest of the cloth has bowed cleanly out in the wind direction.

#### Rest shape and pinning every vertex

When a pin covers every vertex of an object and a movement operation (Move By, Spin, Scale, or Embedded Move) drives it, the object's rest shape is carried along by the pin: the solver treats the transformed positions as the new rest configuration rather than trying to restore the original pose. If the pin is later released via Duration / Active For (or `solver.unpin(frame=...)`), the simulation continues from the deformed shape as its rest pose; vertices do not snap back to where they started. This is how you "pose" a garment into a new resting configuration before letting it fall freely.

### Pin properties reference

| UI label                      | Python / TOML key                   | Description                                                   |
| ----------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| **Include**                   | `included`                          | Pin is active for the current solve.                          |
| **Duration** / **Active For** | `use_pin_duration` / `pin_duration` | Release the pin at the given frame (`solver.unpin(frame=...)`). |
| **Pull** / **Strength**       | `use_pull` / `pull_strength`        | Replace the hard pin with a soft pull force.                  |

### Operations reference

| UI label          | Python / TOML key | Parameters (UI labels)                                                                              | Description                                                     |
| ----------------- | ----------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Embedded Move** | `EMBEDDED_MOVE`   | N/A                                                                                                 | Plays back per-vertex keyframes. Auto-added on first Make Keyframe.     |
| **Move By**       | `MOVE_BY`         | **Delta (m)**, **Start**, **End**, **Transition**                                                   | Translate the pinned vertices by a delta over a frame range.    |
| **Spin**          | `SPIN`            | **Axis**, **Angular Velocity (°/s)**, **Center**, **Start**, **End**, **Transition**               | Rotate the pinned vertices around an axis through a pivot.      |
| **Scale**         | `SCALE`           | **Factor**, **Center**, **Start**, **End**, **Transition**                                          | Scale the pinned vertices uniformly from a pivot.               |
| **Torque**        | `TORQUE`          | **Magnitude (N·m)**, **Axis** (PC1/PC2/PC3), **Flip Direction**, **Start**, **End**                 | Apply a rotational force around a PCA-derived axis.             |

Transition is either Linear or Smooth between the operation's Start and End frames.

The Spin and Scale rows use the Center fields from the table above; under the hood their fields on each operation are `spin_center*` for spin and `scale_center*` for scale, with identical suffixes:

- `*_center_mode`: the mode enum (Centroid / Fixed / Max Towards / Vertex).
- `*_center`: the XYZ used by Fixed.
- `*_center_direction`: the direction used by Max Towards.
- `*_center_vertex`: the vertex index used by Vertex.

### Blender Python API

The Python API mirrors the UI: you create a pin by vertex-group name and chain operations onto it. All operation factories accept `frame_start`, `frame_end`, and `transition` keywords; `spin()` and `scale()` accept the center-mode inputs by keyword.

```python
from zozo_contact_solver import solver

cloth = solver.create_group("Cloth", "SHELL")
cloth.add("Shirt")

# A plain pin: vertices stay fixed.
shoulder = cloth.create_pin("Shirt", "ShoulderPins")

# Rotate the collar around world-Z for the first second.
collar = cloth.create_pin("Shirt", "CollarPins")
collar.spin(axis=[0, 0, 1], angular_velocity=360, frame_start=1, frame_end=60)

# Shrink the hem from its top edge, smoothly.
hem = cloth.create_pin("Shirt", "HemPins")
hem.scale(
    factor=0.5,
    center_direction=(0, 0, -1),   # MAX_TOWARDS: pivot = lowest verts
    frame_start=1, frame_end=60,
    transition="SMOOTH",
)

# Release a pin after 90 frames.
shoulder.unpin(frame=90)

# Keyframed move (EMBEDDED_MOVE). First .move(frame=...) auto-keys at the
# current scene frame, then keys again at the target frame.
sleeve = cloth.create_pin("Shirt", "SleevePins")
sleeve.unpin(frame=30)
sleeve.move(delta=(0, 0, 0.5), frame=20)

# Torque can coexist with EMBEDDED_MOVE but not with SPIN/SCALE/MOVE_BY.
twist = cloth.create_pin("Shirt", "TwistPins")
twist.torque(magnitude=1.0, axis_component="PC3", frame_start=1, frame_end=60)
```

#### Center-mode inference

`spin()` and `scale()` pick the center mode for you based on which kwarg you pass:

| Kwarg provided         | Mode resolved  |
| ---------------------- | -------------- |
| `center_vertex=<int>`  | `VERTEX`       |
| `center_direction=...` | `MAX_TOWARDS`  |
| `center=(x,y,z)`       | `ABSOLUTE`     |
| *(none)*               | `CENTROID`     |

Pass `center_mode=` explicitly if you want to override.

## Invisible colliders

Invisible colliders are parametric collision boundaries that constrain the simulation without being part of the Blender scene. Use them for ground planes, containment volumes, bowls, cages, and quick test rigs you do not want cluttering the viewport or render.

They are purely a solver construct: they have no mesh, no modifier, and are not affected by the scene's render or viewport visibility. They are persisted as part of the scene profile.

### Adding a collider

1. Open the sidebar (`N`) in the 3D viewport and switch to the add-on tab.
2. In the Scene Configuration panel, scroll to the Invisible Colliders sub-panel and click the disclosure triangle to expand it.
3. Click Add (the button with the plus icon). A small dropdown offers two options, Wall and Sphere. Pick one.
4. A new entry appears in the list above the Add / Remove row, named `Wall 1`, `Sphere 1`, `Wall 2`, `Sphere 2`, … by number of existing entries of that type.
5. Every new collider gets a fixed frame-1 keyframe automatically; it reads from the base properties and cannot be deleted.

Selecting a collider in the list opens a properties box just below it with inline fields for the current collider. What you see depends on the collider type:

- Wall: Name, Position, Normal, Contact Gap, Friction, Thickness, and an Active Duration toggle (expands to a duration field when on).
- Sphere: Name, Position, Radius, side-by-side Invert and Hemisphere checkboxes, then the same Contact Gap, Friction, Thickness, Active Duration rows as Wall.

To remove a collider, select it in the list and click the Remove button to the right of Add. Remove is grayed out when nothing is selected.

Figure: The Invisible Colliders sub-panel with two entries (`Wall 1` and `Sphere 1`) and the properties box open for the selected wall (Name, Position, Normal, Contact Gap, Friction, Thickness, Active Duration). Below, the keyframe list shows the auto-generated frame-1 Initial keyframe.

### Preview overlay

Because an invisible collider has no mesh, the only way to see where it sits, and crucially which side it pushes against, is the Preview overlay. Each row in the collider list carries a small eye icon on the right. The preview is off by default (closed-eye icon): the wall or sphere exists in the solver but nothing draws in the viewport. Click the icon to flip it to the open-eye state and the collider draws directly in the 3D viewport, updated live as you edit Position, Normal, Radius, or as the timeline advances through keyframes. Click the open eye again to hide the preview.

Previews are per-collider, so you can isolate one while keeping others hidden. They also respect the Active Duration cutoff: once the current frame passes the end frame the overlay disappears, mirroring what the solver does.

#### Interpreting the arrows

The overlay always draws one or more normal arrows pointing in the direction the collider pushes geometry away from. Read them as "dynamic vertices live on the arrow side".

Figure: Five side-by-side shape schematics. Wall: a flat plane patch with dashed outlines going to infinity and a single blue arrow along its Normal; dynamic geometry stays on the arrow side. Sphere (default): a circle with six outward-pointing arrows at the cardinal ±X, ±Y, ±Z points; dynamic geometry stays outside. Sphere with Invert: the same circle with six inward-pointing arrows converging at the center; dynamic geometry stays inside (containment). Sphere with Hemisphere: a bottom half-circle plus a cylinder rising to infinity from the equator, with outward arrows on both parts; dynamic geometry stays outside the bowl-plus-capsule shape. Sphere with Invert + Hemisphere: the same bowl shape with inward arrows; dynamic geometry stays inside (the bowl catches objects that fall in from above). Blue (outward) or purple (inward) arrows point toward the free side. Shaded fill marks the solid side; dashed outlines mean the surface continues to infinity.

- Wall. A single arrow sticks out of the plane along its Normal. Points on the arrow side are free; points on the opposite side get projected back across the plane. The solid grid shows a small patch of the plane at Position, and the faint dashed outlines expanding outward are a reminder that the plane is infinite; the wall extends well past the rendered patch.

  Figure: A floor wall (`Position = (0, 0, 0)`, `Normal = (0, 0, 1)`). The arrow tells you dynamic geometry stays on the +Z side.

- Sphere. Six arrows appear at the cardinal surface points (±X, ±Y, ±Z for a full sphere; no +Z for a hemisphere). Their direction encodes the two shape flags:

  - Default sphere: arrows point outward. Geometry stays outside the ball.
  - Invert ✓: arrows flip to point inward, toward the center. The sphere becomes a containment volume; geometry stays inside.
  - Hemisphere ✓: the wireframe becomes the lower half plus a cylinder extending upward from the equator (the solver treats the region above the equator as an infinite capsule). Combine with Invert for a bowl that catches falling objects from above.

  Figure: Left to right: a default sphere (objects stay outside, arrows point away from the center), an inverted sphere (containment: objects stay inside; arrows point toward the center and are mostly hidden inside the wireframe), and a hemisphere (lower half + cylindrical extension upward).

Each collider is given a distinct hue: walls start blue, spheres start green, and subsequent entries rotate through the hue wheel, so several overlays stay readable side-by-side.

### Keyframe animation

Colliders animate through their own per-collider keyframe list below the properties box, not through Blender fcurves. Each keyframe stores a frame number plus the values that change at that frame (Position for both, Radius for spheres).

To animate the selected collider:

1. With the collider selected, scrub the timeline to the frame you want.
2. Click Add Keyframe. A new entry is appended to the keyframe UIList, seeded from the collider's current values, on the current scene frame. Duplicate frames are rejected.
3. Select the new keyframe in the list. A keyframe-details box appears below with Frame, Hold, and (when Hold is off) the keyframed value rows (Position, plus Radius for spheres).
4. Adjust values inline. To delete, select the keyframe and click Remove.

The first keyframe in the list is badged Initial. It is frozen to frame 1 and shows the message "Uses base properties above" instead of value rows. It reads whatever is currently in the properties box and cannot be removed; its Remove button stays disabled.

Turning Hold on for any later keyframe makes that frame hold the previous keyframe's value, producing a step function. Useful for "stay put until frame 60, then jump to the next state" patterns: add a Hold keyframe on 60 and a value keyframe on 61.

#### Example: shrink a sphere starting at frame 60

A sphere that keeps `radius = 1.0` until frame 60, then shrinks to `0.5` over the next 60 frames (reaching `0.5` at frame 120):

1. Add a Sphere collider and set its base `Position = (0, 0, 0)` and `Radius = 1.0` in the properties box.
2. Scrub to frame 60, click Add Keyframe, select the new `Frame 60` row, and turn Hold on. No value rows are needed; the encoder will re-emit the previous keyframe's value (1.0) at this frame.
3. Scrub to frame 120, click Add Keyframe, select the new `Frame 120` row, leave Hold off, and set `Radius = 0.5`.

Figure: Five 3D sphere snapshots on a shared ground line. At Frame 1, Frame 30, and Frame 60 the sphere is radius 1.0; a yellow-bordered Hold plateau band highlights these three. At Frame 90 the sphere is radius 0.75. At Frame 120 the sphere is radius 0.5. A faint dashed outline at the Frame 120 position shows what the original radius 1.0 would have looked like.

Figure: Line graph of sphere radius vs frame. X axis Frame (ticks 1, 30, 60, 90, 120), Y axis Radius (ticks 0.5, 0.75, 1.0). Keyframe markers: Frame 1 Initial at 1.0, Frame 60 Hold at 1.0, Frame 120 at 0.5. Blue curve holds flat at 1.0 from frame 1 through frame 60 then ramps linearly down to 0.5 at frame 120. Dashed gray line shows the comparison without Hold: a single straight line from 1.0 at frame 1 down to 0.5 at frame 120, so the sphere would start shrinking at frame 1 instead of holding its initial size.

Figure: Resulting sub-panel. Sphere has three keyframes: Frame 1 (Initial), Frame 60 (Hold), and Frame 120 (Radius → 0.500, currently selected). Because frame 60 holds the previous radius (1.0), the solver keeps the sphere at 1.0 through frame 60 and then linearly interpolates from 1.0 to 0.5 across frames 60–120.

Without the Hold at frame 60 the solver would linearly interpolate radius from 1.0 (frame 1) to 0.5 (frame 120) across all 119 frames, so the sphere would start shrinking immediately at frame 1 instead of holding its initial size until frame 60.

### Types and options

| Type       | Shape                                                   | Extra options                   |
| ---------- | ------------------------------------------------------- | ------------------------------- |
| **Wall**   | Infinite plane at **Position** with outward **Normal**. | N/A                             |
| **Sphere** | Sphere at **Position** with **Radius**.                 | **Invert**, **Hemisphere**      |

A wall pushes simulation geometry onto the normal-facing side of the plane.

A sphere, by default, collides from the outside: objects stay outside the ball. Turning Invert on flips that, so objects stay inside the ball (a containment volume). Turning Hemisphere on leaves the top half open (a bowl). Both flags can be combined.

Both collider types also carry the usual contact settings:

| UI label        | Python / TOML key | Meaning                                                            |
| --------------- | ----------------- | ------------------------------------------------------------------ |
| **Contact Gap** | `contact_gap`     | Barrier gap maintained between the collider and dynamic geometry.  |
| **Friction**    | `friction`        | Tangential friction coefficient.                                   |

### Saving with a scene profile

Invisible colliders and their keyframes are written into the scene profile TOML alongside dynamic parameters. See Scene Parameters. They are not stored per material profile or per group.

The scene-profile `.toml` is generated by clicking the Save icon on the Scene Configuration profile row, not by editing TOML by hand. Add and tune colliders in this panel and then save from Scene Configuration to persist them.

### Blender Python API

The same workflow is available from Python:

```python
from zozo_contact_solver import solver

# A floor (wall facing up) with 0.5 friction.
floor = solver.add_wall([0, 0, 0], [0, 0, 1])
floor.param.friction = 0.5

# A bowl centered at origin: inside-collision hemisphere sphere.
bowl = solver.add_sphere([0, 0, 0], 0.98).invert().hemisphere()
bowl.param.friction = 0.3

# Animated sphere: hold radius=1.0 until frame 60, then jump to 0.5.
solver.add_sphere([0, 0, 0], 1.0).time(60).hold().time(61).radius(0.5)

# Animated wall: slide 1 m along +Y between frames 30 and 90.
(solver.add_wall([0, 0, 0], [0, 1, 0])
       .time(30).hold()
       .time(90).move_to([0, 1, 0]))

# Start over.
solver.clear_invisible_colliders()
```

Both builders are chainable. `time(frame)` advances the keyframe cursor (frames must strictly increase), `hold()` emits a hold keyframe, and `move_to`, `move_by`, `radius`, `transform_to` emit value keyframes. `param.contact_gap` and `param.friction` set the collider's per-contact values without keyframing them.

## Snap & merge

Two separate steps that work together:

- Snap is a one-shot alignment: it translates object A in world space so that its closest vertex lines up with the nearest vertex on object B, leaving just enough contact gap to avoid interpenetration. In the same pass it also captures stitch anchors for every A-vertex within reach of B, not just the single closest one.
- Merge is a solver-side stitch constraint between two objects. At transfer time it becomes a cross-object constraint that keeps the two meshes joined during the simulation.

You typically snap and merge in sequence (the snap operator creates the merge pair automatically), but the two concepts are independent and can be used separately.

IMPORTANT: Snap is a set-level operation, not a single-vertex operation. The feature is designed for meshes whose geometry already coincides (fully or partly) so that many A-vertices have a matching counterpart on B. A canonical example is duplicating a subdivided square patch and snapping the copy onto the original: every edge (or interior) vertex on the copy has a one-to-one counterpart on the original, so the whole set stitches in a single operation. If only one isolated vertex on A happens to be close to B while the rest of the mesh is far away, only that one pair is recorded. The solver won't magically stitch meshes whose topologies don't align. Author the meshes so the intended stitch region coincides before snapping.

### Snap and Merge panel

Open the sidebar (`N`) in the 3D viewport and switch to the add-on tab. The Snap and Merge panel sits below Scene Configuration and Dynamics Groups, and is collapsed by default. Click the header to expand it.

When expanded, the panel contains a single box titled Snap To Nearby Vertices:

- An Object A dropdown (the source; this is the object that will move) with an eyedropper button on the right for picking from the viewport.
- An Object B dropdown (the target; this stays put) with its own eyedropper.
- A Snap A to B button with a snap icon.

To snap two objects together, pick A and B with the dropdowns or eyedroppers, then press Snap A to B. The add-on finds the closest pair of vertices between A and B, translates A (in world space, parent-safe) so they line up, applies the contact-gap rules below, and records per-vertex barycentric anchor data for a later stitch. It also registers a merge pair automatically.

Figure: Snap and Merge panel with Object A set to PatchA (moves, magenta) and Object B set to PatchB (target, stays put, blue), the Snap A to B button, and an empty Merge Pairs box ready to receive the new pair.

Figure: Before Snap A to B. Two identically-subdivided shell patches assigned to Shell groups: PatchB (blue, at the origin) and PatchA (magenta, shifted along the +X axis only; same Y, same Z). Clear gap along the shared edge; the two patches are not yet joined.

Figure: After `Snap A to B`. PatchA has translated along the X axis in world space until its nearest vertex lands on PatchB. Because both patches are Shell, the gap rule is "merge exactly": PatchA's left edge now coincides with PatchB's right edge and every seam vertex is shared, so the two patches read as a single seamless 2x1 rectangle. The merge pair is registered automatically and the per-vertex stitch anchors are captured behind the scenes.

As soon as at least one merge pair exists, a second box labeled Merge Pairs appears below the snap box, containing:

- A UIList showing each pair, with both object names per row.
- A Remove Merge Pair button below the list (disabled unless a row is selected).
- A Stitch Stiffness slider, only shown when the selected pair involves a Solid. Sheet-sheet (Shell-Shell) and rod-rod pairs merge vertices exactly, so stiffness has no meaning for them.

A separate Visualization panel further down the sidebar exposes a Hide all snaps toggle that hides or shows the merge-pair / stitch overlay in the viewport.

#### Gap rules

The solver needs a small separation between the two meshes at rest, otherwise contact barriers start flagging penetration on frame 1. The snap operator picks the gap based on the group types of A and B:

| A type ↔ B type          | Applied gap                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------- |
| Shell ↔ Shell            | **No gap**. Vertices merge exactly.                                                     |
| Rod ↔ Rod                | **No gap**. Vertices merge exactly.                                                     |
| Any other pair           | The larger of the two groups' **Contact Gap** values plus both groups' **Contact Offset**. |

#### Cross-stitch anchors

For every snap, the add-on also stores per-vertex barycentric anchor data on the resulting merge pair. Every A-vertex that ends up within a small reach-threshold of B after alignment becomes its own anchor, so a single snap typically produces many stitches, one per coincident vertex pair, not just one. Conceptually each source vertex is tied to a target triangle (or a single target vertex for rod pairs) with barycentric weights, so the stitch survives later mesh edits until the topology itself changes.

The reach-threshold is derived from the applied gap (roughly `2 × gap`), which is why coinciding geometries work best: vertices that are already on top of each other are well within threshold, while distant vertices are ignored. You can see the captured set as yellow dots connected by thin yellow lines in the viewport overlay (toggle with Hide all snaps in the Visualization panel).

Figure: Yellow stitch-pair overlay between the two subdivided plane patches. PatchA (magenta) has been pulled straight down in Z after the snap, exposing vertical yellow stitch lines connecting every seam vertex of PatchA to the matching seam vertex of PatchB (blue). For this X-axis-only shift, the entire shared seam qualifies, so the overlay draws one yellow stitch per seam-vertex pair. The stitches remain attached to the original coincident seam positions.

### Merge pairs without snapping

A merge pair alone (without snapping) registers two objects as stitched during the solve. Each pair has:

| UI label              | Python / TOML key   | Description                                                                                                                               |
| --------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Object A / B**      | `object_a` / `object_b` | The two mesh objects.                                                                                                                 |
| **Stitch Stiffness**  | `stitch_stiffness`  | Per-pair stiffness. **Only shown for pairs involving a Solid.** Sheet-sheet (Shell-Shell) and rod-rod pairs merge vertices exactly, so stiffness has no meaning. |
| **Show Stitch**       | `show_stitch`       | Overlay toggle for the viewport stitch preview.                                                                                           |

Merge pairs referencing deleted or unassigned objects are cleaned up automatically on the next depsgraph update.

### Blender Python API

The same workflow is available from Python:

```python
from zozo_contact_solver import solver

# Make the shirt hug the mannequin at its closest pair of verts, with the
# right contact gap for the SHELL <-> STATIC pairing. Also registers a
# merge pair with cross-stitch anchor data.
solver.snap("Shirt", "Mannequin")

# Or register a merge pair without moving anything.
solver.add_merge_pair("Shirt", "Mannequin")

# Iterate existing pairs.
for a, b in solver.get_merge_pairs():
    print(f"{a} <-> {b}")

# Remove a specific pair, or all of them.
solver.remove_merge_pair("Shirt", "Mannequin")
solver.clear_merge_pairs()
```

NOTE: Snapping and merging both reject library-linked (non-writable) objects because the solver needs to persist UUIDs on them. Make them local first if you hit that error.

UNDER THE HOOD:

What Snap does. Snap is a one-shot alignment:

1. Finds the closest pair of vertices between objects A and B.
2. Translates A in world space (parent-safe) along the approach direction so the two vertices line up, plus the gap-rule distance and a small float32 safety margin.
3. Records per-vertex barycentric anchor data for every A-vertex close enough to B to participate in a stitch.

For the no-gap pairings (Shell-Shell, Rod-Rod) the anchor-capture threshold falls back to `max(gap_a, gap_b)` so nearby vertices still enter the stitch even though the final separation is zero.

Cross-stitch anchor data. Each merge pair carries the captured anchor payload:

- Source and target object UUIDs.
- Per source vertex: target triangle indices and barycentric weights `[1.0, α, β, γ]`.
- Target positions at snap time.
- Vertex counts at snap time, so stale entries can be detected if you edit the topology later.

When the pair is sent to the solver, the target vertex with the highest barycentric weight is picked per stitch. For rod-to-rod / rod-to-shell the target degenerates to a single vertex with weight 1.

Merge-pair encoding. Merge pairs are tracked by UUID, so renaming either object preserves the link. Pairs referencing an object that has never been snapped or merged (no UUID yet) are skipped at transfer. An empty cross-stitch payload means snap has not run or the pair is not eligible for a stitch.
