# 📌 Pins and Operations

A **pin** is a named set of vertices on an object that the add-on has
marked as a constraint target. By itself a pin just holds its vertices in
place during the solve. On top of that you stack **operations**: small,
keyframeable drivers that move, rotate, scale, or torque the pinned
vertices while the simulation runs.

The pin tells the solver *which* vertices are constrained. The operations
tell the solver *how they should move*.

```{figure} ../../images/pins/pin_hierarchy.svg
:alt: Two-panel reference. Left panel is a tree with the root node "Group: Cloth (SHELL)" branching into three pins: ShoulderPins (carrying Spin and Scale operations), CollarPins (carrying Move By), and WaistPins (carrying Torque). A legend underneath lists the five operation types (Move By, Spin, Scale, Torque, Embedded Move) and notes that Torque is exclusive with Move By / Spin / Scale. Right panel is a 2×2 grid of small schematics for the four Center modes on Spin and Scale. Centroid places the pivot at the mean of the vertex cluster. Fixed places the pivot at a user-entered world-space coordinate offset from the cluster. Max Towards uses a direction arrow to place the pivot at the centroid of the vertices furthest in that direction. Vertex places the pivot on one specific mesh vertex that deforms along with the mesh.
:width: 880px

Anatomy at a glance. On the left, **pins nest inside a group** and
**operations stack on each pin**. On the right, the four options of
the **Center** dropdown on **Spin** and **Scale**. Each picks a
different resolution for the pivot point the operation rotates or
scales around.
```

## Where Pin Indices Are Stored

How the vertex set is stored depends on the object type:

- **Meshes**. The pin *is* a Blender vertex group. Any vertex group on
  the mesh is eligible; the add-on simply registers it by name and reads
  its members at solve time. You can author the group through Blender's
  usual weight-paint / vertex-group UI, or let the add-on create one from
  an Edit-mode selection.
- **Curves** (Bezier, used by **Rod** groups). Blender
  curves have no native vertex groups, so the add-on creates an
  **internal** group per pin. The control-point indices are stored as a
  JSON array in a custom property on the curve object, keyed
  `_pin_<name>`. You never edit this property by hand; it is written and
  read through the **Create** button and the Edit-mode
  select/deselect actions.

## The Pins Section in a Group Box

Each group box on the **Dynamics Groups** panel contains a **Pins**
section. It sits between the **Assigned Objects** list and the
**Material Params** box, and is **collapsed on a newly created
group**: only the **Pins** header is drawn until you click the
disclosure triangle to its left. The header label itself is not a
button. The same triangle opens the **Transform** section on a
**Static** group and **Pins & Motion** on a **PDRD** group. Expanded,
the section is laid out top-to-bottom as follows:

1. **Vertex-group selector row.** A single dropdown sits at the top. It
   lists every vertex group on every mesh object assigned to this group
   as one combined `[Object][VertexGroup]` entry (e.g.
   `[Shirt][ShoulderPins]`), and that entry is what identifies the
   vertex group you want to register as a pin. The **Add** button shares
   the same row, to the dropdown's right. The dropdown only enumerates
   **mesh** vertex groups, since curve objects have no vertex groups and
   so never appear here; their pins are authored through **Create**
   instead.

2. **Action buttons.** **Add** sits on the selector row above and
   registers the vertex group currently shown in the selector dropdown
   as a new pin on this group. The row directly below holds three more
   buttons, left to right:
   - **Remove**: deletes the currently-selected pin from the list,
     including every operation attached to it. Grayed out when no pin is
     selected.
   - **Rename**: opens a small dialog prefilled with the selected pin's
     vertex-group name. Editing the name renames the underlying vertex
     group (for meshes) or the `_pin_<name>` custom property (for
     curves), then updates the pin's identifier and content hash. Grayed
     out when no pin is selected.
   - **Create**: only available when you are in Edit Mode with vertices
     selected. Opens a small dialog with a **Vertex Group Name** field
     (default `pin`); confirming it creates the vertex group from the
     selection and registers it as a pin.

3. **Pin UIList.** Below the buttons is a scrollable list of all pins
   registered on this group. Each row displays:
   - The **`[object][vertex-group]`** label (e.g.
     `[Shirt][ShoulderPins]`).
   - A **Show** eye toggle on the right side of the row, on by default.
     Clicking it hides that pin's overlay dots in the viewport without
     changing anything the solver does.
   - A row whose **vertex group** was deleted is drawn as
     `[object][vertex-group] (Missing)` with an error icon; a row whose
     **object** was deleted keeps its stored label and shows the same
     error icon without the `(Missing)` suffix. **Remove** the row, or
     recreate the vertex group under the same name. A plain rename needs
     no action: the list re-resolves the object by UUID and the vertex
     group by content hash, and rewrites the label.

   Below the list, a **Size** field sets the overlay dot size and the
   **▲** / **▼** buttons reorder the pins; for a vertex covered by
   more than one pin, the lower entry in the list wins.

   The list supports single selection: clicking a row highlights it and
   opens the **selected-pin details panel** described below.

### Creating a Pin

```{figure} ../../images/pins/pin_section.png
:alt: Pins section inside a Dynamics Groups group box
:width: 500px

The **Pins** section of a group box, expanded. The `[Cloth][ShoulderPins]`
vertex-group selector, **Add / Remove / Rename / Create** buttons, the
pin UIList with the registered `ShoulderPins` entry carrying its own eye
toggle, the **Size** field sharing a row with the **▲** / **▼** reorder
buttons, the pin-level fields (**Duration**, **Pull**, **Allow
Intersections Here**), and the still-empty **Operations** list are all
visible.
```

There are two UI paths to create a pin, both driven from the action
buttons above the pin UIList:

1. **From an existing vertex group.** Pick the `[Object][VertexGroup]`
   entry in the selector dropdown and click **Add**. The vertex group
   appears immediately as a new row in the pin UIList. This path is
   mesh-only, since curves have no vertex groups to pick from.
2. **From an Edit-mode selection.** Enter Edit Mode on a mesh or curve
   that belongs to this group, select the vertices (or curve control
   points) you want pinned, and click **Create**. Name the new group in
   the dialog that opens (it defaults to `pin`); on confirm the add-on
   creates the group from the current selection, registers it as a pin,
   and the new entry appears in the UIList. For meshes this writes
   a regular Blender vertex group; for curves it writes the internal
   `_pin_<name>` custom property described above. This is the **only**
   way to create a pin on a curve object.

**Remove** deletes the currently-selected pin, including every operation
attached to it. After removal, the selection moves to the next pin in
the list, or the list becomes empty.

### The Selected-Pin Details Panel

Selecting a pin in the UIList reveals a details panel below the list with
the pin's own properties:

- **Duration**: a checkbox that enables an **Active For** frame-count
  field beside it. When on, the pin is released that many frames after
  the solve's **Starting Frame**, so with the default Starting Frame of
  1 an **Active For** of 60 releases at frame 61, and with a Starting
  Frame of 100 it releases at frame 160.
- **Pull**: a checkbox with a **Strength** field next to it. When on,
  the pin no longer hard-constrains the vertices; instead, it pulls them
  toward their target positions as a soft force of the given strength.
- **Allow Intersections Here**: a checkbox. When on, the geometry this pin
  holds completely passes through other geometry, with no contact. See
  [Allow Intersections Here](#allow-intersections-here) below.
- **Operations UIList**: a list of the operations stacked on this pin,
  each row showing the operation type.

**Pull** composes with the movement operations rather than excluding
them: the operations supply the target position, and **Pull** is what
decides how the pin holds its vertices there, softly with the given
strength instead of prescribing them exactly. The one control it does
disable is **Fix Weight Threshold**, a **Solid**-only field that applies
to hard pins.

### Allow Intersections Here

By default the solver keeps every pair of surfaces apart, and it refuses a
scene whose geometry already overlaps. **Allow Intersections Here** lets the
geometry one pin holds pass through whatever it meets: the solver applies no
contact force to it, does not stop it from crossing other surfaces, and never
reports its overlaps as an error. That is narrower than the group settings,
which cover whole objects.

Turn it on when the pin drives its vertices somewhere that has to pass
through something: a cuff pulled onto a wrist that starts inside it, or a
band captured from a rig-deformed pose that arrives folded into the body
underneath. A plain pin puts its vertices exactly where it says, so contact
cannot move them aside; with the option on, the held geometry passes through
what it is driven into instead. A **Pull** pin holds its vertices only as
hard as its **Strength**, and the option also removes the contact that would
otherwise push back against the pull. The option is available on both.

**It applies only where the pin holds a whole element.** A triangle counts
when all three of its corners are pinned, a rod segment when both of its
ends are, a **Sand** grain when that one grain is, and every pin holding
those vertices has to have the option on, not just one of them. A face with
one free corner is not covered, so a band pinned along a single edge leaves
the cloth around it colliding as usual.

Coverage is asked of the pair, not of both of its sides. A pair passes
through as soon as **one** of the two elements is fully held by allowing
pins, so a face that is only partly pinned, or not pinned at all, passes
through geometry that is fully covered. Flagging the pin on a cuff therefore
lets it pass through the wrist inside it, without the wrist needing a pin of
its own. Only a pair in which neither element is fully covered keeps its
contact. While the option is on, the add-on shows "Fully pinned faces pass
through; partly pinned ones still collide" under the checkbox.

:::{important}
An element this pin covers has no contact with anything: it passes through
other objects, through **Static** colliders, and through the rest of its own
mesh alike. The option does not reach the
[invisible walls and spheres](colliders.md), which act the same with it on or
off. Every pair in which neither side is covered keeps full contact, and the
solver never lets it intersect.
:::

The checkbox sits with every pin the add-on offers. On **Solid**, **Shell**,
**Rod**, and **Sand** groups it is in the pin details panel described above;
on a **PDRD** group it is in the **Pins & Motion** section, just above
**Motion steps**, with the same note under it. That **Motion steps** list is
limited to two buttons, **Translate** and **Rotate**; **Scale**, **Torque**
and **Embedded Move** are not offered there, and a step of one of those
types imported from a scene built elsewhere shows only as
`<type>: edit in the generic panel`. A **Static** group is driven
by **Transform** operations rather than pins, so it has no pin to put the
checkbox on. For geometry that should pass through outside a pinned region,
use the group-level
[Allow Intersections](../params/material.md#allow-intersections) settings
instead. A **Static** collider left in its rest pose ignores the boxes on its
own group, so set them on the moving object's group.

### Edit-Mode Pin Buttons

When the pinned object is in Edit Mode, two extra rows appear above the
pin properties:

- **Select** / **Deselect**: flip the selection state of every vertex
  (or curve control point) in the active pin. Useful for previewing
  which indices are tagged before running, or for adding/removing
  members via Blender's regular selection tools.
- **Make Keyframe** / **Delete All Keyframes**: drive the
  **Embedded Move** operation. **Make Keyframe** attaches
  **Embedded Move** on the first press and records the current posed
  positions; **Delete All Keyframes** removes every keyframe and the
  operation in one step. Curve pins don't accept keyframes, so the
  buttons are inert on rods.

All four buttons are hidden outside Edit Mode because they act on the
Edit-mode selection or on mesh-data animation.

### Adding an Operation

Above the **Operations UIList** in the selected-pin details panel sits
an **Operations:** label alongside **Copy** and **Paste** clipboard
icons (copy the pin's operation list to a session-scoped clipboard;
paste replaces the target pin's operations wholesale), then a pin
operations **Profile** row for saving the operation list to a file and
loading it back. Directly below the list is a single row of buttons:

- **Add**: opens a dropdown menu listing the available operation types
  (see below). New operations insert at the top of the list.
- **Remove**: deletes the currently-selected operation. Removing an
  **Embedded Move** row drops the sentinel, so the solver stops reading
  the keyframes, but the per-vertex fcurves stay on the mesh; use
  **Delete All Keyframes** to remove those as well.
- Up / Down triangles (**▲** / **▼**): reorder the selected operation
  within the list. Order determines the sequence in which the solver
  applies each operation's contribution when more than one is stacked
  on the same pin.

The **Add** dropdown lists the available operation types:

- **Move By**: translate the pinned vertices by a delta.
- **Spin**: rotate the pinned vertices around an axis through a pivot.
- **Scale**: scale the pinned vertices uniformly from a pivot.
- **Torque**: apply a rotational force around a PCA-derived axis.

**Embedded Move** is not in the dropdown: it's attached automatically
on the first **Make Keyframe** press. Every other entry stays
selectable, but picking one that would violate a compatibility rule
(e.g. adding **Spin** when a **Torque** already exists) is refused with
an error in the status bar, and nothing is added.

```{figure} ../../images/pins/pin_ops_editor.png
:alt: Dynamics Groups panel on a Shell group with a pin selected and a Spin operation in the Operations list. Above the list is the Operations label with Copy and Paste clipboard icons on the right. Below the list are Add, Remove, and up/down reorder buttons. The Spin editor underneath is in Fixed center mode, showing the Pick from Selected eyedropper
:width: 500px

The selected-pin details panel on a **Shell** group. The four action
buttons (**Add / Remove / Rename / Create**) sit above the pin list.
Above the **Operations** list are the **Copy** / **Paste** clipboard
icons; below it are **Add**, **Remove**, and the **▲** / **▼**
reorder triangles. A **Spin** operation is selected with **Center**
set to **Fixed**, which exposes the **XYZ** coordinate fields and the
**Pick from Selected** eyedropper.
```

Picking an entry inserts a new operation row into the pin's **Operations
UIList**. Each row in that list shows:

- The **operation type** label (e.g. **Spin**, **Move By**).
- A small **overlay-visibility toggle** (eye icon) on the right side
  of the row. Clicking it turns the viewport overlay for that
  operation on or off, useful for previewing pivots or directions
  without running the solver.

Clicking an operation row selects it and opens the **per-operation
editor** directly below the operations list. The fields shown depend
on the operation type:

```{figure} ../../images/pins/spin_op_editor.png
:alt: Pins section with a Spin operation selected showing its editor
:width: 500px

The selected-pin details panel with the `Spin ω=360°/s` operation
selected in the Operations UIList. Its per-type editor is visible
below: the **Center** dropdown (set to **Centroid**), the **Axis** XYZ
vector, **Angular Velocity (°/s)**, **Flip Direction** toggle, **Start
/ End** frame range, and the **Transition** dropdown.
```


- **Move By**: **Delta (m)** (XYZ vector), **Start**, **End**,
  **Transition** dropdown (**Linear** / **Smooth**).
- **Spin**: **Axis** (XYZ vector), **Angular Velocity (°/s)**, a
  **Center** dropdown (see below), the center-mode's companion field,
  **Start**, **End**, **Transition**.
- **Scale**: **Factor** (scalar), a **Center** dropdown + companion
  field, **Start**, **End**, **Transition**.
- **Torque**: **Magnitude (N·m)**, **Axis** dropdown (**1st
  Component** / **2nd Component** / **3rd Component**, i.e. `PC1` /
  `PC2` / `PC3` in the Python API), **Flip Direction** checkbox,
  **Start**, **End**.
- **Embedded Move**: no editable fields; this operation is managed
  entirely via the **Make Keyframe** and **Delete All Keyframes**
  buttons (see below).

:::{warning}
**Torque** cannot coexist with **Move By**, **Spin**, or **Scale** on
the same pin, and it cannot coexist with **Embedded Move** either: a
keyframed pin refuses every operation you try to add, and **Make
Keyframe** refuses a pin that already carries operations. Picking an
incompatible entry reports an error and adds nothing.
:::

### Make Keyframe

The **Make Keyframe** button appears in the edit-mode row (see above)
and drives the **Embedded Move** operation. On the **first** press for a
given pin:

1. It samples the current positions of the pinned vertices at the current
   scene frame.
2. It attaches an **Embedded Move** operation to the pin and stores the
   samples as its first keyframe.

Subsequent presses add more keyframes at the current scene frame without
duplicating the operation; only one **Embedded Move** ever exists per
pin. Visibly, each press "bakes in" the current posed shape of the pinned
vertices; scrubbing the timeline then plays back the keyframes as the
simulation runs.

**Delete All Keyframes** removes every keyframe *and* the **Embedded
Move** operation in one step.

### Capture Deformation

When the pinned vertices already follow a deformer on the cloth mesh,
such as an **Armature** pose, a **Lattice** cage, a **Mesh Deform**
cage, animated **Shape Keys**, or a **driver**, pressing **Make
Keyframe** at every frame to mirror the motion is impractical.
**Capture Deformation** does the sampling pass for you.

```{figure} ../../images/pins/pin_capture_overlay.png
:alt: Bent cloth at frame 30 with pin overlay dots tracing the bone curve
:width: 520px

A cloth driven by a multi-bone armature partway through its pose
animation. The white pin overlay dots follow the bone-driven edge,
so the solver sees the pin's target position at every frame without
any per-frame keyframing by the artist.
```

#### Where the controls live

In the **Pins** section of the pin's group, with the pin selected in the
list, two buttons appear below the pin list's **Size** row: **Capture
Deformation** and **Clear Deformation Cache**. **Capture Deformation**
turns on only when the pin's mesh has a deforming modifier on it;
**Clear Deformation Cache** turns on only once a capture exists. Both
grey out while a capture is running. When a cache exists, a
**Pin cache: N frame(s)** label sits just below the buttons, and the
operation list shows an **`[Embedded] Move (Captured)`** entry that
labels the captured animation as the live source.

```{figure} ../../images/pins/pin_capture_panel.png
:alt: Pins section showing Capture / Clear buttons, the Pin cache label, and the captured Embedded Move row
:width: 360px

The same pin's section in the **Dynamics Groups** panel. The captured
state is visible in three places at once: the **Capture Deformation**
/ **Clear Deformation Cache** row, the **Pin cache: 180 frame(s)**
status line, and the **`[Embedded] Move (Captured)`** entry in
**Operations**.
```

#### Using it

1. Bind the cloth to its deformer the usual way (e.g. parent to the
   armature and add an **Armature** modifier; pose the bones and
   keyframe the pose).
2. Create the **Dynamics Group**, add the cloth, and register the pin
   vertex group on the edge or region you want the bones to drive.
3. Press **Capture Deformation**. The captured range is derived, not
   read from the timeline: it begins at the solve's **Starting Frame**
   and ends at the last keyframe of every action influencing the cloth,
   so shrinking the scene's frame range does not truncate the cache.
   (For procedural motion with no keyframes at all, the range instead
   runs for the Scene Configuration panel's **Frame Count**.) A
   progress label reports as it walks the range and names the range it
   derived; on completion the **Pin cache** count appears and the
   operation row updates.
4. Press **Transfer** and **Run** as usual. The pinned vertices follow
   the bones; the rest of the cloth simulates around them.

Press **Capture Deformation** again any time the underlying animation
changes, a new pose, edited keys, a different modifier; the cache does
not refresh on its own.

**Clear Deformation Cache** discards the cache and returns the pin to
its previous state. If the pin had no manual **Make Keyframe**
authoring underneath, the **`[Embedded] Move`** entry is removed too.

:::{note}
**Capture Deformation** and manual **Make Keyframe** authoring cannot
co-exist on the same pin. Capture refuses to start while manual
keyframes are present (press **Delete All Keyframes** first), and
**Make Keyframe** refuses to add new keys to a captured pin (press
**Clear Deformation Cache** first). **Torque** is also incompatible
with captured animation, matching the existing **Torque** vs
**Embedded Move** rule.
:::

:::{admonition} Under the hood
:class: dropdown

Capture writes a per-pin cache keyed by the cloth object and the
vertex group name; reopening the file on a host where the cache file
is missing automatically clears the captured marker so the pin won't
silently feed stale data into the next **Transfer**.

When the cloth has a deformer on it, the solver-output cache modifier
is installed *after* the deformer in the modifier stack. This avoids
re-applying the bone displacement on top of the solver's already
bone-aware output, which would visibly double the motion on playback.
:::

### Center-Mode Dropdown and Overlays

**Spin** and **Scale** both rotate or scale *around* something. The
per-operation editor exposes that "something" as a **Center** dropdown
with four modes, each revealing a different companion field underneath:

| Mode             | Companion field           | How the pivot is resolved                                                                       |
| ---------------- | ------------------------- | ----------------------------------------------------------------------------------------------- |
| **Centroid**     | *(none)*                  | Mean of the pinned vertex positions.                                                            |
| **Fixed**        | XYZ coordinate + **Pick from Selected** eyedropper | Fixed world-space point. With the mesh in Edit Mode and one or more vertices selected, the eyedropper writes the selection's world-space centroid into the XYZ field. |
| **Max Towards**  | Unit direction vector     | Centroid of the vertices furthest in that direction.                                            |
| **Vertex**       | Vertex index + **Pick Vertex** eyedropper | A single vertex on the mesh. The eyedropper reads the one selected vertex in Edit Mode; it reports an error if zero or more than one vertex is selected. The pivot deforms with the mesh. |

Alongside each operation are viewport-overlay toggles (**Show Max
Towards**, **Show Vertex**, and siblings) that draw the computed pivot
in the viewport so you can preview it before solving.

### A Wind-Blown Banner

A minimal end-to-end scene that exercises the pin system: a vertical
cloth banner with its top edge pinned, deformed by wind.

1. **Build the banner.** Add a flat **Plane**, apply a **Simple**
   subdivision at **Viewport level 5** (→ a 33×33 grid of vertices, i.e.
   32 quads per side), then rotate it 90° around **X** and apply the
   rotation so it stands upright in the XZ plane.
2. **Create the pin vertex group.** Enter **Edit Mode**, select the top
   row of vertices (the 33 with the maximum Z), and create a new vertex
   group named **TopEdge**.
3. **Register with a SHELL group.** Leave Edit Mode, open the **Dynamics
   Groups** panel, click **Create Group**, set the type to **Shell**,
   then add the plane via **Add Selected Objects**.
4. **Pin the top edge.** Expand the group's **Pins** section (a new
   group has it collapsed, so click the triangle beside the **Pins**
   header), then pick `[ClothBanner][TopEdge]` in the vertex-group
   selector and click **Add**. Leave the pin's eye toggle in the list
   on (it is by default) so the pinned vertices render as white dots in
   the viewport. Steps 3 and 4 leave the Dynamics Groups panel looking
   like this:

   ```{figure} ../../images/pins/pin_example_panel_groups.png
   :alt: Dynamics Groups panel with ClothBanner SHELL, ClothBanner object assigned, and TopEdge pinned with its overlay visible
   :width: 360px

   The panel after steps 3 and 4. `ClothBanner` is the lone assigned
   object under the Shell group, and `[ClothBanner][TopEdge]` is
   selected in the Pins list with its eye toggle on.
   ```

5. **Drive the wind.** In the **Scene Configuration** panel, set the
   **Wind** direction to `(0, 1, 0)` and **Strength** to around
   **20 m/s**, then bump **Air Density** to `0.01 kg/m³` (the max).
   The force on the cloth scales with air density, so the default
   `0.001 kg/m³` leaves the banner barely moving.

   ```{figure} ../../images/pins/pin_example_panel_scene.png
   :alt: Scene Configuration panel with Air Density 0.01, Wind direction (0,1,0), Strength 20 m/s
   :width: 360px

   The matching Scene Configuration panel. Air Density is bumped to
   `0.01`; the Wind sub-section is expanded with direction
   `(0, 1, 0)` and strength `20.00 m/s`.
   ```

6. **Transfer → Run.**

```{figure} ../../images/pins/pin_example_rest.png
:alt: Vertical subdivided plane with the top row of vertices marked by white pin dots
:width: 520px

Rest pose at frame 1. The row of white dots along the top edge is the
pin overlay drawing each vertex of `TopEdge`. Those vertices are the
ones the solver will hold fixed.
```

```{figure} ../../images/pins/pin_example_blown.png
:alt: Same plane after wind has bulged the body outward, top edge still straight
:width: 520px

The same scene at frame 15, a fraction of a second after **Run**. The
pinned vertices along the top, still marked by the overlay dots,
have not moved at all; the rest of the cloth has bowed cleanly out in
the wind direction.
```

### Rest Shape and Pinning Every Vertex

When a pin covers **every** vertex of an object and a movement operation
(**Move By**, **Spin**, **Scale**, or **Embedded Move**) drives it, the
object's rest shape is carried along by the pin: the solver treats the
transformed positions as the new rest configuration rather than trying
to restore the original pose. If the pin is later released via
**Duration** / **Active For** (or `pin.unpin(frame=...)`), the
simulation continues from the deformed shape as its rest pose; vertices
do not snap back to where they started. This is how you "pose" a garment
into a new resting configuration before letting it fall freely.

## Pin Properties Reference

| UI label                      | Python / TOML key                   | Description                                                   |
| ----------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| **Show**                      | `show_overlay`                      | Draw this pin's vertices as overlay dots in the viewport (no effect on the solve). |
| **Duration** / **Active For** | `use_pin_duration` / `pin_duration` | Number of frames the pin stays active, counted from the solve's **Starting Frame**; the pin is released after that many frames. `pin.unpin(frame=...)` sets this count despite the keyword's name. |
| **Pull** / **Strength**       | `use_pull` / `pull_strength`        | Replace the hard pin with a soft pull force.                  |
| **Allow Intersections Here**  | `allow_intersection`                | Let the elements this pin holds completely pass through whatever they meet, with no contact. |
| **Track Rest-Pose Deformation** | `track_rest_pose_deformation`     | **Solid** only, off by default. Drives a time-varying rest pose from the pin's captured deformation, so the body settles into the captured shape instead of straining against it. Editable only on a *full* pin (one covering every vertex of the mesh) that has a capture; the **Refresh** button beside it re-checks that coverage. It cannot coexist with plasticity, and the panel warns when both are on. |

## Operations Reference

| UI label          | Python / TOML key | Parameters (UI labels)                                                                              | Description                                                     |
| ----------------- | ----------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Embedded Move** | `EMBEDDED_MOVE`   | N/A                                                                                                 | Plays back per-vertex animation. Auto-added on first **Make Keyframe** (manual) or **Capture Deformation** (shown as **`[Embedded] Move (Captured)`**). |
| **Move By**       | `MOVE_BY`         | **Delta (m)**, **Start**, **End**, **Transition**                                                   | Translate the pinned vertices by a delta over a frame range.    |
| **Spin**          | `SPIN`            | **Axis**, **Angular Velocity (°/s)**, **Center**, **Start**, **End**, **Transition**               | Rotate the pinned vertices around an axis through a pivot.      |
| **Scale**         | `SCALE`           | **Factor**, **Center**, **Start**, **End**, **Transition**                                          | Scale the pinned vertices uniformly from a pivot.               |
| **Torque**        | `TORQUE`          | **Magnitude (N·m)**, **Axis** (**1st** / **2nd** / **3rd Component**), **Flip Direction**, **Start**, **End** | Apply a rotational force around a PCA-derived axis.             |

**Transition** is either **Linear** or **Smooth** between the
operation's **Start** and **End** frames.

The **Spin** and **Scale** rows use the **Center** fields from the table
above; under the hood their fields on each operation are `spin_center*`
for spin and `scale_center*` for scale, with identical suffixes:

- `*_center_mode`: the mode enum (**Centroid** / **Fixed** / **Max
  Towards** / **Vertex**).
- `*_center`: the XYZ used by **Fixed**.
- `*_center_direction`: the direction used by **Max Towards**.
- `*_center_vertex`: the vertex index used by **Vertex**.

## Blender Python API

The Python API mirrors the UI: you create a pin by vertex-group name and
chain operations onto it. All operation factories accept `frame_start`,
`frame_end`, and `transition` keywords; `spin()` and `scale()` accept the
center-mode inputs by keyword.

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

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

# Translate the pinned vertices by a fixed offset, ramped over a frame
# range. (The interactive keyframed move, EMBEDDED_MOVE, is UI-only; the
# Python API exposes move_by instead.)
sleeve = cloth.create_pin("Shirt", "SleevePins")
sleeve.move_by(delta=(0, 0, 0.5), frame_start=20, frame_end=30, transition="SMOOTH")
sleeve.unpin(frame=60)

# Torque cannot be mixed with MOVE_BY, SPIN, or SCALE on the same pin.
twist = cloth.create_pin("Shirt", "TwistPins")
twist.torque(magnitude=1.0, axis_component="PC3", frame_start=1, frame_end=60)
```

### Center-Mode Inference

`spin()` and `scale()` pick the center mode for you based on which kwarg
you pass:

| Kwarg provided         | Mode resolved  |
| ---------------------- | -------------- |
| `center_vertex=<int>`  | `VERTEX`       |
| `center_direction=...` | `MAX_TOWARDS`  |
| `center=(x,y,z)`       | `ABSOLUTE`     |
| *(none)*               | `CENTROID`     |

Pass `center_mode=` explicitly if you want to override.
