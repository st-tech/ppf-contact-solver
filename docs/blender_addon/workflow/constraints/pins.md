# 📌 Pins and Operations

A **pin** names a set of vertices that the solver holds in place or, with a
**Torque**, pushes with a force. A pin belongs to an object group. On top
of a pin you stack **operations**: small motions or forces, each running
between a **Start** and an **End** frame, that move, rotate, scale, or twist
the pinned vertices. A pin can also follow motion that you keyframe by hand or
capture from a deformer such as an armature.

The pin sets *which* vertices are driven. The operations set *how they
move*.

## A Wind-Blown Banner

A vertical cloth banner with its top edge pinned, blown by wind. It uses a
plain pin with no operations. The sections after it cover everything else a
pin can do.

1. **Build the banner.** Add a flat **Plane** and rename the object
   `ClothBanner`. Add a **Subdivision Surface** modifier set to **Simple**
   with **Levels Viewport** 5, and apply it. The plane becomes a 33×33 grid
   of vertices (32 quads per side). Rotate the plane 90° around **X** and
   apply the rotation so it stands upright in the XZ plane.
2. **Create the pin vertex group.** Enter **Edit Mode**, select the top row
   of vertices (the 33 with the greatest Z), and create a vertex group named
   `TopEdge` (Object Data properties, **Vertex Groups**, **+**, rename it,
   **Assign**).
3. **Register the object in a Shell group.** Leave Edit Mode, open the
   **Dynamics Groups** panel, click **Create Group**, set the type to
   **Shell**, and click **Add Selected Objects** with the plane selected.
4. **Pin the top edge.** Expand the group's **Pins** section (a new group
   has it collapsed, so click the triangle beside the **Pins** header), pick
   `[ClothBanner][TopEdge]` in the vertex-group selector, and click **Add**.
   Leave the pin's eye icon on (it is on by default), so the pinned vertices
   draw as white dots in the viewport. Steps 3 and 4 leave the **Dynamics
   Groups** panel looking like this:

   ```{figure} ../../images/pins/pin_example_panel_groups.png
   :alt: Dynamics Groups panel with a ClothBanner Shell group, the ClothBanner object assigned, and the TopEdge pin in the Pins list with its eye icon on
   :width: 360px

   The panel after steps 3 and 4. `ClothBanner` is the only assigned object
   in the Shell group, and `[ClothBanner][TopEdge]` is selected in the
   **Pins** list with its eye icon on.
   ```

5. **Set the wind.** In the **Scene Configuration** panel, set **Air
   Density** to `0.01` kg/m³ (its maximum), expand the **Wind** box, set
   **Direction** to `(0, 1, 0)`, and set **Strength** to `20` m/s. The wind
   acts through air drag, which is scaled by **Air Density** (see
   [Air Friction](../params/scene.md#air-friction)), so **Air Density** sets
   how strongly the wind pushes the cloth.
6. **Transfer and run.** Connect to a solver (see
   [Connections](../../connections/index.md)), then press **Transfer** and
   **Run** in the **Solver** panel (see
   [Running the Simulation](../sim/simulating.md)).

```{figure} ../../images/pins/pin_example_rest.png
:alt: Vertical subdivided plane with the top row of vertices marked by white pin dots
:width: 520px

Rest pose at frame 1. The white dots along the top edge are the pin overlay,
one dot per vertex of `TopEdge`. The solver holds those vertices in place.
```

```{figure} ../../images/pins/pin_example_blown.png
:alt: The same plane after wind has bulged the body outward, with the top edge still straight
:width: 520px

The same scene at frame 15. The pinned vertices along the top edge have not
moved. The rest of the cloth bulges out in the wind direction. The dots are
light blue because the fetched animation is loaded on the object (see the dot
colors under [The Pins Section in a Group Box](#the-pins-section-in-a-group-box)).
```

## What a Pin Holds

The vertices a pin holds depend on the object type:

- **Mesh.** The pin is one of the object's Blender vertex groups. Any vertex
  group on the mesh can become a pin, and every vertex in the group is held,
  whatever its weight. Build the group with Blender's vertex-group or
  weight-paint tools, or select vertices in Edit Mode and press **Create**.
- **Curve** (used by **Rod** groups). A curve has no vertex groups, so the
  add-on records the selected control points for you. Select the control
  points in Edit Mode and press **Create**. Which control points a rod can
  hold is described in [Pins on a Curve Rod](#pins-on-a-curve-rod).
- **Hair Curves** (used by **Rod** groups). Hair has no vertex groups either.
  A pin is a named attribute on its points: a Boolean attribute holds the
  points it marks, and a Float attribute holds every point with a value above
  `0` and gives each one a weight. Hair attached to a surface also has a
  **Roots** pin that needs no attribute. See
  [Pins on Hair Curves](#pins-on-hair-curves).

:::{admonition} Under the hood
:class: toggle

A curve pin is stored as a JSON array of control-point indices in a custom
property on the curve object, keyed `_pin_<name>`. **Create** writes it,
**Rename** renames it, and **Select** / **Deselect** read it.
:::

## The Pins Section in a Group Box

Each group box on the **Dynamics Groups** panel has a **Pins** section. It
sits below the **Delete Group** button and above the **Stats** and
**Material Params** boxes, and it is collapsed on a newly created group:
click the disclosure triangle to the left of the **Pins** header to open it.
The header label itself is not a button. The same triangle opens
**Transform** on a **Static** group (see
[Static Objects](../scene/static_objects.md)) and **Pins & Motion** on a
**PDRD** group (see [Pins on a PDRD Group](#pins-on-a-pdrd-group)).

On **Solid**, **Shell**, **Rod**, and **Sand** groups the opened section is
laid out top to bottom as follows:

1. **Vertex-group selector row.** A dropdown lists every vertex group on every
   mesh object assigned to this group as one `[Object][VertexGroup]` entry,
   for example `[Shirt][ShoulderPins]`. The **Add** button sits to its right.
   For a Hair Curves object it lists the point attributes that can be a pin,
   and `[Object][Roots]`. A curve object never appears in it, and it reads
   **None** when no assigned object offers an entry.

2. **Action buttons.** **Add** registers the vertex group shown in the
   dropdown as a pin on this group. The row below holds three more buttons,
   left to right:
   - **Remove**: deletes the selected pin and every operation attached to it.
     The vertex group itself stays on the mesh. The pin that was below the
     removed one is selected afterward, or the new last pin when the last one
     was removed. Grayed out when no pin is selected.
   - **Rename**: opens a dialog prefilled with the selected pin's
     vertex-group name. Confirming renames the vertex group (on a curve, the
     pin's recorded control points; on hair, the attribute), and the pin
     follows. Grayed out when no pin is selected.
   - **Create**: available only in Edit Mode with vertices (or the points of
     a curve or of hair) selected. See [Creating a Pin](#creating-a-pin).

3. **Pin list.** A scrollable list of the pins registered on this group.
   Clicking a row selects the pin and shows its settings below (see
   [Pin Settings](#pin-settings)). Each row shows:
   - The label `[Object][VertexGroup]`.
   - An eye icon at the right end, on by default. Clicking it hides that
     pin's dots in the viewport and changes nothing the solver does.

   A pin whose vertex group was deleted is drawn as
   `[Object][VertexGroup] (Missing)` with an error icon. A pin whose object
   was deleted keeps its stored label and shows the same error icon without
   the `(Missing)` suffix. Remove the row, or recreate the vertex group under
   the same name.

   Renaming the object or the vertex group in Blender needs no action: the
   list follows the change and rewrites the label. A renamed vertex group is
   found only while its members are unchanged and no other vertex group on
   the object has the same members.

   **Transfer** never skips a pin whose vertex group is gone (deleted, or
   renamed and then edited so its members do not identify it) or whose
   group holds no vertices (on a curve or on hair, no points). It stops with a
   message naming the pin, the object, and the group, instead of sending the
   object without a pin that the list shows. Pin the group you mean again,
   assign vertices to it, or remove the pin.

4. **Size and order.** The **Size** field below the list sets the dot size for
   all pins of the group (4 to 32, default 12). The **▲** / **▼** buttons
   beside it move the selected pin up or down. For a vertex covered by more
   than one pin, the lower entry in the list wins, whatever its settings: a
   plain pin listed below a pin with **Pull** on holds the vertices the two
   share exactly, and listed above it, it leaves them to the pull.

In the viewport a pin is drawn as dots, white by default. The dots turn light
blue when the pin carries a **Move By**, **Spin**, **Scale**, or **Torque**
operation. On a mesh, every pin of the object also turns light blue once
fetched animation is loaded on it. **[Embedded] Move** alone leaves the dots
white. **Hide all pins** and **Hide all pin operations** in the
**Visualization** panel hide the dots and the operation overlays for the whole
scene.

A pin with **Duration** on stops drawing its dots on the frame where it is
released, which is the **Starting Frame** plus **Active For**. On a mesh the
dots are drawn at the current positions of the pinned vertices. They follow the
fetched animation while you scrub the timeline, and a deforming modifier such
as an **Armature** before any simulation has been fetched. In Edit Mode they
follow a vertex as you drag it.

### Creating a Pin

```{figure} ../../images/pins/pin_section.png
:alt: Pins section of a Dynamics Groups group box with one pin in the list and an empty Operations list
:width: 500px

The **Pins** section of a group box, expanded: the `[Cloth][ShoulderPins]`
selector with **Add**, the **Remove** / **Rename** / **Create** row, the pin
list with the `ShoulderPins` entry and its eye icon, the **Size** field with
the **▲** / **▼** reorder buttons, the **Capture Deformation** row with its
status line, **Duration**, **Pull**, **Allow Intersections Here**, and the
still-empty **Operations** list.
```

There are two ways to create a pin from the buttons above the pin list:

1. **From an existing vertex group** (meshes only). Pick the
   `[Object][VertexGroup]` entry in the selector and click **Add**. The
   vertex group appears as a new row in the pin list. Adding a group that is
   already a pin adds nothing.
2. **From an Edit Mode selection.** Enter Edit Mode on a mesh or curve that
   belongs to this group, select the vertices (or curve control points) to
   pin, and click **Create**. Name the vertex group in the dialog (it
   defaults to `pin`) and confirm. The add-on makes the group from the
   selection and registers it as a pin. If a vertex group with that name
   already exists on the mesh, its members are replaced by the selection. In
   the panel, **Create** is the only way to make a pin on a curve object.
   For a curve, pass `indices=` to `create_pin` from Python. On Hair Curves,
   **Create** stores the selection as a point attribute, and a selection
   with partly selected points becomes a weighted pin; see
   [Pins on Hair Curves](#pins-on-hair-curves).

### Edit-Mode Pin Buttons

While the pin's own object is in Edit Mode, two extra rows appear below the
**Size** row:

- **Select** / **Deselect**: **Select** selects every vertex (or curve control
  point) in the active pin, and **Deselect** removes them from the selection.
  Use them to check which vertices a pin holds, or to add and remove members
  with Blender's selection tools.
- **Make Keyframe** / **Delete All Keyframes**: author the pin's motion by
  hand. See [Make Keyframe](#make-keyframe). Curve pins do not accept
  keyframes, so on a curve rod both buttons only report that they do not
  apply. On Hair Curves both are grayed out.

All four buttons are drawn only while the pin's own object is in Edit Mode.

### Pin Settings

Selecting a pin in the list shows its settings below the list, in this order:

1. The Edit Mode rows above, in Edit Mode only.
2. **Capture Deformation** and **Clear Deformation Cache**, with a status line
   (see [Capture Deformation](#capture-deformation)).
3. On a **Solid** group, **Track Rest-Pose Deformation** with a refresh icon
   (see [Track Rest-Pose Deformation](#track-rest-pose-deformation)).
4. **Duration** and **Active For**.
5. **Pull** and **Strength**.
6. On a **Solid** group, **Fix Weight Threshold**.
7. **Allow Intersections Here**.
8. The **Operations** area (see [Operations](#operations)).

Details for four of these settings:

- **Duration**: a checkbox that enables an **Active For** frame-count field
  beside it (default 60, minimum 1). When on, the pin is released that many
  frames after the solve's **Starting Frame**, so with the default Starting
  Frame of 1 an **Active For** of 60 releases at frame 61, and with a
  Starting Frame of 100 it releases at frame 160.
- **Pull**: a checkbox with a **Strength** field beside it (default 1.0).
  When on, the pin does not hold the vertices exactly. It pulls them toward
  their target positions with a soft force of the given strength. **Pull**
  works with operations: the operations supply the target position, and
  **Pull** sets how firmly the pin holds the vertices there. On a **Shell**,
  **Rod**, or **Sand** group, a **Strength** of 0 holds the vertices exactly,
  as if **Pull** were off.

  A vertex that a pulling pin holds does not fall or swing on its own. The
  pull stands in for the vertex's own weight and momentum, so the vertex
  leaves its target only as far as the rest of the object, or a contact,
  drags it against the pull. An object with **Pull** on every vertex
  therefore stays at its targets under gravity, at any **Strength**. Leave
  unpinned the part that should fall and swing.

  On a weighted pin of a Hair Curves object, each point is pulled with
  **Strength** times its own weight. See
  [Pins on Hair Curves](#pins-on-hair-curves).
- **Fix Weight Threshold**: on a **Solid** group, the weight at or above which
  a pinned surface vertex is held exactly (default 0.5, range 0 to 1). It is
  drawn gray while **Pull** is on. See
  [Pinning Part of a Solid](#pinning-part-of-a-solid).
- **Allow Intersections Here**: lets the faces, rod segments, or **Sand**
  grains whose vertices this pin holds entirely pass through other geometry
  with no contact. See [Allow Intersections Here](#allow-intersections-here).

## Operations

```{figure} ../../images/pins/pin_hierarchy.svg
:alt: Two panels. The left panel shows a group named Cloth holding three pins. ShoulderPins holds a Spin and a Scale operation, CollarPins holds a Move By operation, and WaistPins holds a Torque operation, marked exclusive because Torque cannot share a pin with Move By, Spin or Scale. The right panel shows the four Center settings for Spin and Scale: Centroid, Fixed, Max Towards and Vertex, each with where the pivot lands
:width: 760px

A group holds pins, and a pin holds a stack of operations (left). The
**Center** setting of **Spin** and **Scale** places the pivot (right).
```

Operations make a pin move. They sit in the **Operations** area at the bottom
of the pin settings:

- Above the list, the label **Operations:** with two clipboard icons, then
  the **Profile** row (see [Saving Operations to a Profile](#saving-operations-to-a-profile)).
- The **Operations** list.
- Below the list, a row with **Add**, **Remove**, and the **▲** / **▼**
  buttons.
- Below that, the editor of the selected operation.

The two clipboard icons are **Copy Pin Operations** and **Paste Pin
Operations** (their tooltips). The copy icon stores the selected pin's
operation list in a clipboard that lasts for the Blender session. The paste
icon replaces the selected pin's operations with the copied list, and it stays
gray until something has been copied.

### Adding an Operation

- **Add**: opens a dropdown menu with the four operation types. A new
  operation is inserted at the top of the list.
  - **Move By**: translate the pinned vertices by a delta.
  - **Spin**: rotate the pinned vertices around an axis through a pivot.
  - **Scale**: scale the pinned vertices uniformly from a pivot.
  - **Torque**: apply a rotational force about an axis derived from the
    pinned vertices (see [Torque](#torque)).
- **Remove**: deletes the selected operation. Removing an **[Embedded] Move**
  row that **Make Keyframe** created stops the solver from reading the
  keyframes. The vertex keyframes stay on the mesh, where the Dope Sheet and
  Graph Editor show them. To delete them as well, use **Delete All
  Keyframes**. To discard a captured deformation, use **Clear Deformation
  Cache**.
- **▲** / **▼**: move the selected operation up or down. The order sets the
  sequence in which the operations are applied when several are stacked on one
  pin.

**[Embedded] Move** is not in the dropdown. It appears on its own when you
press **Make Keyframe** or **Capture Deformation**. Every entry in the dropdown
stays selectable, but picking one that breaks a compatibility rule (for
example **Spin** on a pin that already has a **Torque**) is refused with an
error in the status bar, and nothing is added.

:::{warning}
**Torque** cannot share a pin with **Move By**, **Spin**, or **Scale**, and it
cannot share a pin with keyframed or captured motion either. A keyframed or
captured pin refuses every operation you try to add, **Make Keyframe**
refuses a pin that already carries operations, and **Capture Deformation**
refuses a pin that carries a **Torque**. Each refusal is reported in the
status bar, and nothing is added.
:::

### Editing an Operation

Click an operation row to select it. Its editor opens directly below the
list, and the fields depend on the type (see
[Operations Reference](#operations-reference)). Each row names the operation
and its main value, for example `Move (0.0, 0.0, 0.0)`, `Spin ω=360°/s`,
`Scale ×1.00`, or `Torque 1.0 N·m`, and ends with an eye icon that shows or
hides that operation's viewport overlay. The eye is off for a new operation,
so it draws nothing until you turn it on. The overlay previews the operation in
the viewport without running the solver, in one color per type and with a text
label:

- **Move By** (cyan): an arrow from every pinned vertex, labeled *Δ (x, y, z)*
  with the delta. A **Delta** of `(0, 0, 0)` draws nothing.
- **Spin** (orange): a circle around the pivot with an arc for the sweep,
  labeled *ω = 360°/s*.
- **Scale** (magenta): an arrow from each vertex to its scaled position,
  labeled *×0.50*.
- **Torque** (green): a circle and an arc about the chosen axis, labeled with
  the magnitude and the axis, for example *τ = 1.0 N·m (PC3)*. PC1, PC2 and PC3
  stand for the **1st**, **2nd** and **3rd Component**.

Overlays are drawn for mesh pins only. **Hide all pin operations** hides them,
but not the hinge axle of a **PDRD** body, which its **Visualize** toggle
controls (see [Hinge Joints](../params/material.md#hinge-joints)). The
**[Embedded] Move** row has no eye icon and no editor.

What each type does between its **Start** and **End** frames:

- **Move By** translates the pinned vertices by **Delta (m)**, given along the
  world axes.
- **Spin** turns the pinned vertices about **Axis** through a pivot, at a
  constant angular velocity. **Axis** is a direction in world coordinates, and
  only its direction matters, not its length. A negative **Angular Velocity**
  turns the other way, as **Flip Direction** does. An **Axis** of `(0, 0, 0)`
  with a nonzero **Angular Velocity** names no axis to turn about, and
  **Transfer** refuses it, naming the pin.
- **Scale** scales the pinned vertices uniformly about a pivot, from 1 to
  **Factor**.
- **Torque** applies a force instead of a pose (see [Torque](#torque)).

Once the **End** frame passes, **Move By**, **Spin**, and **Scale** leave the
pinned vertices at the pose they reached. **Transition** is either **Linear**
or **Smooth** (an ease in and out) and applies to **Move By** and **Scale**
only. **Spin** always turns at the constant angular velocity, so the
**Transition** dropdown in its editor has no effect.

**Start** and **End** are Blender frames. An operation added from the panel or
over MCP spans frames 1 to 60, or, when the solve's **Starting Frame** is later
than frame 1, the same 60 frames beginning at the **Starting Frame**. The
Python API defaults to 1 and 60 whatever the **Starting Frame** is (see
[Pin Surface](../../integrations/python_api.md#pin-surface)). **Transfer**
refuses an operation whose **End** is not after its **Start**, or whose
**Start** is before the **Starting Frame** (the part before it would never
run), naming the pin, the operation, and both frames. Nothing is clipped for
you: move the operation's frames, or move the **Starting Frame**.

### Center-Mode Dropdown and Overlays

**Spin** and **Scale** both rotate or scale *around* something. The editor
shows that point as a **Center** dropdown with four modes, each with its own
companion field:

| Mode            | Companion field                                    | Where the pivot is                                                                                                 |
| --------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Centroid**    | *(none)*                                           | The mean of the pinned vertex positions.                                                                           |
| **Fixed**       | **Center** XYZ and **Pick from Selected**          | A fixed position in world space. With the mesh in Edit Mode and vertices selected, **Pick from Selected** writes the world-space centroid of the selection into the XYZ fields. The button is gray outside Edit Mode. |
| **Max Towards** | **Direction** (XYZ)                                | The centroid of the pinned vertices that lie furthest along **Direction**, measured at the operation's **Start** frame. A **Direction** of `(0, 0, 0)` picks no vertex, and **Transfer** refuses it, naming the pin. |
| **Vertex**      | **Vertex Index** and an eyedropper                 | The position of that vertex when you press **Transfer** or **Update Params on Remote**, held at that position. The eyedropper next to **Vertex Index** reads the one vertex selected in Edit Mode, and shows the warning *Select exactly one vertex* when none or several are selected. With no vertex picked, **Transfer** refuses the operation, naming the pin. |

In **Max Towards** mode the editor adds a **Show Max Towards** toggle, and in
**Vertex** mode a **Show Vertex** toggle. Each draws the pivot in the viewport
so you can check it before solving. **Show Max Towards** draws the vertices
that set the pivot as yellow dots, with a yellow arrow along **Direction** from
their centroid, labeled *Max Towards (N verts)*. **Show Vertex** draws a green
dot on the picked vertex, labeled *Center (v12)* for vertex 12.

### Torque

**Torque** applies a rotational force to the pinned vertices instead of
prescribing a pose. It acts about the centroid of the pinned vertices:

- **Axis** picks one of the three principal axes of the pinned vertices.
  **1st Component** is the direction along which the vertices spread the most,
  **2nd Component** the middle direction, and **3rd Component** the direction
  along which they spread the least, often the surface normal. **3rd
  Component** is the default.
- **Magnitude (N·m)** sets the torque (default 1.0), and **Flip Direction**
  reverses which way it turns.
- The pin must hold at least three vertices. **Transfer** refuses a **Torque**
  on a pin with fewer, naming the pin.

A pin that carries only a **Torque** does not hold its vertices. They move
freely and are pushed by the torque between **Start** and **End**; outside that
window nothing acts on them.

### Saving Operations to a Profile

The **Profile** row, below the **Operations:** label, saves the selected pin's
operations to a TOML file and loads them back, so one set of operations can be
reused on other pins and in other scenes. The row has two states:

- **No file loaded.** The row shows an **Open Profile** button (folder icon)
  and a save icon. **Open Profile** picks a profile file. The save icon asks
  for a file and an **Entry Name** (default `Default`) and writes the
  operations as one named entry.
- **File loaded.** The row shows a **Profile** dropdown listing the entries of
  the file, followed by four icons: open another file, clear (unload the
  file), reload (apply the selected entry again), and save (write the selected
  pin's operations over the selected entry, without a dialog). Choosing an
  entry in the dropdown replaces the selected pin's operations with that
  entry.

A profile holds operations only, never the pin, its vertex group,
**Duration**, or **Pull**. The loaded file belongs to the group, so it stays
loaded while you select other pins.

## Pins That Follow Animation

When a pin has to move along a path that the operations cannot describe, give
it the motion itself. **Make Keyframe** records poses that you set by hand.
**Capture Deformation** records the motion of a deformer.

### Make Keyframe

Pose the pinned vertices at a few frames, and the pin follows those poses:

1. Open the pin's object in Edit Mode and select the pin in the list.
2. At the first frame, press **Make Keyframe**. The pin gains an **[Embedded]
   Move** row in its **Operations** list.
3. Go to another frame, move the pinned vertices in Edit Mode, and press
   **Make Keyframe** again.
4. Repeat for every pose.

Each press records a keyframe for every vertex of the pin at the current
frame, with linear interpolation. The keys are ordinary vertex keyframes, so
the Dope Sheet and Graph Editor show them, and **Transfer** sends them. The
rest of the cloth is simulated around the pinned vertices. Only one
**[Embedded] Move** row exists per pin.

**Delete All Keyframes** removes this pin's **[Embedded] Move** row and the
vertex keyframes of the **whole mesh**, not just those of this pin's vertices.
If two pins on one mesh are keyframed, both lose their keys, and the other
pin keeps its **[Embedded] Move** row.

The keyed vertices start the solve at their pose at the **Starting Frame**,
read from the keyframe curves at that frame, and then follow each key after
it. A key at or before the **Starting Frame**, frame 0 included, still counts:
it shapes that starting pose through the curve. A pin whose keys begin before
the **Starting Frame** therefore starts the solve where its curve is at the
**Starting Frame**, and still ends on its last key.

### Capture Deformation

When the pinned vertices already follow a deforming modifier on the mesh, such
as an **Armature**, a **Lattice**, a **Mesh Deform** cage, a **Hook**, a
**Shrinkwrap**, or a **Geometry Nodes** setup that moves vertices, pressing
**Make Keyframe** at every frame is impractical. **Capture Deformation**
records the pinned vertices at every frame for you.

**Capture Deformation** is available only while such a modifier is enabled in
the viewport, and only on mesh pins, not on curve pins. A pin on Hair Curves
can be captured too, and there the button is available whenever anything
moves the hair object, since hair is carried by the surface it is parented
to (see [Hair on a Moving Body](../scene/hair_curves.md#hair-on-a-moving-body)).
The modifiers that
count are **Armature**, **Cast**, **Cloth**, **Corrective Smooth**, **Curve**,
**Displace**, **Hook**, **Laplacian Deform**, **Laplacian Smooth**,
**Lattice**, **Mesh Deform**, **Shrinkwrap**, **Simple Deform**, **Smooth**,
**Soft Body**, **Surface Deform**, **Volume Displace**, **Warp**, **Wave**, and
a **Geometry Nodes** modifier whose node group changes vertex positions (for
example with **Set Position** or **Transform Geometry**). Shape keys or drivers
alone do not enable it. **Make Keyframe** records only poses you set by hand;
to feed evaluated positions in, use `set_animation` from the Python API.

```{figure} ../../images/pins/pin_capture_overlay.png
:alt: Bent cloth at frame 30 with pin overlay dots tracing the bone curve
:width: 520px

A cloth driven by a multi-bone armature partway through its pose animation.
The pin overlay dots follow the bone-driven edge.
```

#### Where the controls live

With the pin selected in the **Pins** section, two buttons sit below the
**Size** row: **Capture Deformation** and **Clear Deformation Cache**. A status
line below them says which state the pin is in:

| Status line                                     | State                                                       |
| ----------------------------------------------- | ----------------------------------------------------------- |
| *No deforming modifier on this pin's object*    | **Capture Deformation** is gray.                            |
| *Deforming modifier detected; capture to encode* | **Capture Deformation** is available.                       |
| *Pin cache: N frame(s)*                         | A capture exists. **Clear Deformation Cache** is available. |

Both buttons are gray while a capture is running. Once a capture exists, the
**Operations** list shows an **[Embedded] Move (Captured)** row that marks
the captured motion as the live source.

```{figure} ../../images/pins/pin_capture_panel.png
:alt: Pins section showing the Capture Deformation and Clear Deformation Cache buttons, the Pin cache label, and the captured Embedded Move row
:width: 360px

The **Pins** section of the same pin after a capture. The captured state shows
in three places: the **Capture Deformation** / **Clear Deformation Cache**
row, the **Pin cache: 180 frame(s)** status line, and the **[Embedded] Move
(Captured)** row in **Operations**.
```

#### Using it

1. Bind the cloth to its deformer the usual way (for example parent it to the
   armature and add an **Armature** modifier, then pose the bones and keyframe
   the pose).
2. Create the object group, add the cloth, and register the pin vertex
   group on the edge or region you want the deformer to drive.
3. Press **Capture Deformation**. A progress bar with an **Abort** button
   appears in the **Solver** panel and names the range being captured. The
   range is derived, not read from the timeline: it begins at the solve's
   **Starting Frame** and ends at the last keyframe of every action that
   influences the cloth, so shrinking the scene's frame range does not
   truncate the capture. For procedural motion with no keyframes at all, the
   range instead runs for the **Frame Count** of the Scene Configuration
   panel. When the capture finishes, the **Pin cache** count appears and the
   **Operations** row updates.
4. Press **Transfer** and **Run** as usual (see
   [Running the Simulation](../sim/simulating.md)). The pinned vertices follow
   the deformer, and the rest of the cloth is simulated around them.

Press **Capture Deformation** again whenever the underlying animation changes,
such as a new pose, edited keys, or a different modifier. The capture does not
refresh on its own. To redo every capture in the scene at once, use
**Re-capture All Deformations** in the **Solver** panel (see
[Static Objects](../scene/static_objects.md#re-capturing-every-deformation-at-once)).

**Clear Deformation Cache** discards the capture. If the pin has no keyframes
from **Make Keyframe**, the **[Embedded] Move** row is removed too.

Capture works on any mesh a pin sits on: a **Shell** or **Solid**, a mesh
**Rod**, or a **Sand** body, and on a Hair Curves object in a **Rod** group.
Modifiers that only draw the object are left out
of the capture, as they are left out of the pose the object is sent at: the
add-on's **ContactSolverCache** modifier, a **Rod**'s **Wireframe**, and a **Sand** body's
**Particle Mesh**. A deformer in front of a **Subdivision Surface** is
captured, and one after it is not, the same as for **Transfer** (see
[Object Groups](../scene/object_groups.md#assigning-objects)).

A **Fluid** domain, an **Ocean** modifier in **Generate** mode, or a **Mesh
Sequence Cache** of different topology on the mesh stops the capture with a
message naming the pin, the object, and the frame. **Transfer** refuses an
object that holds a captured pin while such a modifier is on it. Remove or
disable the modifier, or clear the capture.

:::{note}
**Capture Deformation** and **Make Keyframe** cannot share a pin. Capture
refuses to start while the pin has keyframes, so press **Delete All
Keyframes** first. **Make Keyframe** refuses to add keys to a captured pin, so
press **Clear Deformation Cache** first.
:::

:::{admonition} Under the hood
:class: toggle

A capture is stored as one cache file per pin, in the scene's data folder.
Opening a file on a host where that cache file is missing clears the pin's
captured marker, so the pin does not feed stale data into the next
**Transfer**.

When the cloth has a deformer on it, the add-on places its
**ContactSolverCache** modifier after the deformer in the modifier stack, so the deformer's
motion is not applied a second time on playback.

In a stack that changes the vertex count, capture reads the stack cut at the
first modifier that the add-on recognizes as changing it, such as a
**Subdivision Surface**, **Mirror**, **Bevel**, or **Solidify**. That is the
cut **Transfer** sends and the place where the cache modifier sits. A **Fluid**
domain, an **Ocean** modifier in **Generate** mode, and a **Mesh Sequence
Cache** of different topology change the vertex count without being
recognized that way, so when one sits in front of the cut, the cut stack still
has a different vertex count and the pinned vertices cannot be read.
:::

### Track Rest-Pose Deformation

On a **Solid** group, **Track Rest-Pose Deformation** lets a captured
deformation also drive the body's rest pose, so the rest shape follows the
captured shape. The checkbox sits below the
**Pin cache** line, with a refresh icon beside it, and is off by default.

It stays gray until the pin has a capture and you have clicked the refresh
icon, which checks whether the pin holds **every** vertex of the mesh. Until
then the line below reads *Click Refresh to check pin coverage*, and when the
pin does not hold every vertex it reads *Pin every vertex to enable rest-pose
tracking*. **Transfer** refuses a tracking pin that does not hold every
vertex.

It cannot be combined with the group's **Plasticity**. The panel shows
*Plasticity and rest-pose tracking cannot both be on; Transfer will refuse*,
and **Transfer** does refuse until one of the two is off.

## Pins on Solids and Curves

### Pinning Part of a Solid

A **Solid** is simulated as tetrahedra built from its surface, so its
tetrahedral vertices are not the Blender vertices you pinned. A pin on part of
a **Solid** is therefore carried over as a weight: the pinned region is spread
smoothly across the tetrahedral surface and into the interior, and each
tetrahedral vertex it reaches is held in proportion to its weight, strongest in
the middle of the region and fading toward its edge. On a hard pin, **Fix
Weight Threshold** sets the weight at or above which a surface vertex is held
exactly. Lower-weight surface vertices and every interior vertex are pulled
softly instead, which gives the body a firmly held shell and a compliant core.

Two shapes of geometry cannot carry that spread. The pin then holds only the
tetrahedral surface vertices nearest the pinned Blender vertices, and the scene
build writes a notice naming the object to `server.log` in the solver folder
(see [Troubleshooting](../../troubleshooting.md)):

- a piece of the tetrahedral surface that no Blender vertex lies on, which the
  tetrahedralizer can produce;
- a pin so small against a much finer Blender mesh that the spread reaches no
  tetrahedral vertex.

A pin that reaches no tetrahedral vertex even that way would leave the object
unpinned, so the scene build stops, naming the object: pin more of its
vertices, or tetrahedralize it more finely. Any other failure to carry the pin
over stops the build the same way, naming the object, instead of simulating it
unpinned.

:::{admonition} Under the hood
:class: toggle

The spread is a least-squares Poisson solve on the tetrahedral surface
followed by a harmonic extension into the interior, solved with SciPy on the
solver host. SciPy is a required frontend dependency that every distribution
ships. A Python that cannot import it stops the build with a message saying how
to install it, instead of building a different pin.
:::

### Pins on a Curve Rod

On a **Bezier** or **Poly** spline every control point is a vertex of the
simulated rod, and a pinned control point holds that one vertex.

A **NURBS** spline's points mostly sit off the curve, so the rod follows the
spline arc by arc, each arc spanning **Order** minus one points. Only the point
that ends an arc is a vertex of the rod: the one whose index along its spline
is a multiple of **Order** minus one (every second point at order 3, every
third at order 4). A pin on any other NURBS point, or on a spline with a single
point, has no rod vertex to hold, and **Transfer** refuses it, naming the
curve, the spline, and the point. Pin arc-end points, or convert the spline to
**Bezier** or **Poly** with **Set Spline Type**.

The same arcs set a rule for the whole spline: its points have to fill whole
arcs, since points past the last whole arc would not be simulated.
**Transfer** refuses a NURBS spline that leaves points over, naming it. Add or
remove points, or convert the spline. Blender's **NURBS Circle** (order 3,
eight points, cyclic) fills whole arcs and is accepted as it is.

### Pins on Hair Curves

On a Hair Curves object every point is a vertex of a simulated strand, and a
pin is a point attribute of the hair:

- **Roots** holds the first two points of every curve. It is worked out from
  the curves each time it is read and needs no attribute, and hair attached
  to a surface gets it on joining the group.
- A **Boolean** attribute holds the points it marks. **Create** in Edit Mode
  writes one from the selection.
- A **Float** attribute is a *weighted pin*. It holds every point whose
  weight is above `0`. With **Pull** on, each point is pulled with
  **Strength** times its weight, so the hold can fade along a strand. With
  **Pull** off, every weighted point is held exactly and alike. The panel
  shows which of the two applies in a line under the **Pull** row.

Three things decide how a pinned strand behaves:

- **An exact pin does not hold a strand's direction.** The strand turns
  freely about the last point the pin holds, whatever its **Bend Stiffness**.
  **Roots** keeps the first segment on the scalp and nothing more.
- **A pulled point does not fall or swing on its own** (see **Pull** under
  [Pin Settings](#pin-settings)). A pulling pin whose weight is above zero
  along a whole strand holds the whole strand still. Let the weight reach `0`
  where the hair should move.
- **The lower pin in the list wins a shared point.** A weighted pin that
  starts at the root covers the points **Roots** holds, so move **Roots**
  below it to keep the roots exact. The panel shows *Pins above lose shared
  points to this one: move Roots below* under the **Pull** row while the
  order stands the other way.

A hair pin takes no keyframes and follows a moving head through
**Capture Deformation**. [Hair Curves](../scene/hair_curves.md) covers all of
this in detail, with the ways to make a weighted pin.

## Allow Intersections Here

By default the solver applies contact to every pair of surfaces, and it
refuses a scene whose geometry already overlaps. **Allow Intersections Here**
lets the geometry one pin holds pass through whatever it meets: the solver
applies no contact force to it, does not stop it from crossing other surfaces,
and never reports its overlaps as an error. That is narrower than the group
settings, which cover whole objects.

Turn it on when the pin drives its vertices somewhere that has to pass through
something: a cuff pulled onto a wrist that starts inside it, or a band captured
from a rig-deformed pose that arrives folded into the body underneath. A plain
pin places its vertices exactly at their target positions, so contact cannot
move them aside. With the option on, the held geometry passes through what it
is driven into instead. A **Pull** pin holds its vertices only as hard as its
**Strength**, and the option also removes the contact that would otherwise push
back against the pull. The option is available on both.

**It applies only where the pin holds a whole element.** A triangle counts when
all three of its corners are pinned, a rod segment when both of its ends are,
a **Sand** grain when that one grain is, and every pin holding those vertices
has to have the option on, not just one of them. A face with one free corner is
not covered, so a band pinned along a single edge leaves the cloth around it
colliding as usual.

The option applies to a pair as soon as **one** of its two elements is fully
held by allowing pins, so a face that is only partly pinned, or not pinned at
all, passes through geometry that is fully covered. Flagging the pin on a cuff
therefore lets it pass through the wrist inside it, without the wrist needing
a pin of its own. Only a pair in which neither element is fully covered keeps
its contact. While the option is on, the add-on shows *Fully pinned faces pass
through; partly pinned ones still collide* under the checkbox.

:::{important}
An element this pin covers has no contact with anything: it passes through
other objects, through **Static** colliders, and through the rest of its own
mesh alike. The option does not reach the
[invisible walls and spheres](colliders.md), which act the same with it on or
off. Every pair in which neither side is covered keeps full contact, and the
solver never lets it intersect.
:::

Every pin has this checkbox. On **Solid**, **Shell**, **Rod**, and **Sand**
groups it is in the pin settings, and on a **PDRD** group it is in **Pins &
Motion**, with the same note under it. A **Static** group is driven by Static
ops in its **Transform** box rather than by pins, so it has no pin to put the
checkbox on (see
[The Transform Sub-Box](../scene/static_objects.md#the-transform-sub-box)).
For geometry that should pass through outside a pinned region, use the
group-level
[Allow Intersections](../params/material.md#allow-intersections) settings
instead. A **Static** collider left in its rest pose ignores the boxes on its
own group, so set them on the moving object's group.

## Pins on a PDRD Group

A **PDRD** group moves as one rigid body (see
[PDRD-Specific](../params/material.md#pdrd-specific)), so its pins hold the
body instead of individual vertices. Open the **Pins & Motion** section of the
group box with the disclosure triangle left of its header. It differs from the
layout in
[The Pins Section in a Group Box](#the-pins-section-in-a-group-box):

- **Add** and **Create** share the selector row. **Remove** and **Rename** are
  on the row below. The pin list, **Size** field, and **▲** / **▼** buttons
  are as on other groups, and **Select** / **Deselect** appear in Edit Mode.
- The panel shows the hint *1 vertex = pivot, edge (2) = axis, 3+ = fully
  locked*. A pin on one vertex acts as a pivot for the body, a pin on an edge
  acts as an axis, and a pin on three or more vertices locks the body fully.
- The release control is labeled **Release**, with an **After (frames)** field.
  It releases the pin that many frames after the solve's **Starting Frame**,
  as **Duration** does on other groups.
- **Allow Intersections Here** sits below the hint, with the label *Fully
  pinned faces pass through; partly pinned ones still collide* under it while
  it is on (see [Allow Intersections Here](#allow-intersections-here)).
- A **Motion steps (optional):** list holds the motion. **Translate** adds a
  **Move By** step and **Rotate** adds a **Spin** step. **Remove** and the
  **▲** / **▼** buttons work as for operations. With no steps, the label
  *No steps: the pin holds fixed* shows. The **Rotate** step takes its center
  from **Centroid** or **Vertex** only.
- Selecting a step opens its editor below the buttons. **Translate** shows
  **Delta (m)**, **Start**, **End**, and **Transition**. **Rotate** shows a
  **Centroid** / **Vertex** toggle for the center (**Vertex** adds **Vertex
  Index**, an eyedropper, and **Show Vertex**), then **Axis**, **Angular
  Velocity (°/s)**, **Flip Direction**, **Start**, **End**, and
  **Transition**. The fields mean the same as those of **Move By** and **Spin**
  in [Operations Reference](#operations-reference).
- **Scale**, **Torque**, and **[Embedded] Move** steps are not offered. A step
  of one of those types imported from a scene built elsewhere shows only as
  `<type>: edit in the generic panel`.
- **Pull**, **Fix Weight Threshold**, **Capture Deformation**, **Make
  Keyframe**, **Track Rest-Pose Deformation**, the two clipboard icons, and the
  **Profile** row are not drawn.

## Rest Shape and Pinning Every Vertex

By default an object's rest shape is the shape you modeled. When every pin on
an object has **Duration** on, and the pins together hold every vertex of the
object, the rest shape becomes the pose that the pins' movements (**Move By**,
**Spin**, **Scale**, or **[Embedded] Move**) reach at the release frame. After
the release the body keeps that pose as its rest shape instead of returning to
the shape you modeled. This is how you pose a garment into a new resting
configuration before letting it fall freely. If any pin on the object has no
**Duration**, or any vertex is unpinned, the rest shape stays as modeled.
[Track Rest-Pose Deformation](#track-rest-pose-deformation) is the other way a
pin sets the rest shape.

## Pin Properties Reference

| UI label                        | Python property                     | Default          | Description |
| ------------------------------- | ----------------------------------- | ---------------- | ----------- |
| Eye icon on the pin row (**Show**) | `show_overlay`                   | on               | Draw this pin's vertices as dots in the viewport. No effect on the solve. |
| **Size** (per group)            | `pin_overlay_size`                  | 12 (4 to 32)     | Dot size for all pins of the group. |
| **Duration** / **Active For**   | `use_pin_duration` / `pin_duration` | off / 60 (min 1) | Number of frames the pin stays active, counted from the solve's **Starting Frame**. `pin.unpin(frame=...)` sets this count despite the keyword's name. |
| **Pull** / **Strength**         | `use_pull` / `pull_strength`        | off / 1.0 (min 0) | Replace the hard pin with a soft pull force. On a **Shell**, **Rod**, or **Sand** group, a **Strength** of 0 holds the vertices exactly, as with **Pull** off. On a weighted pin of a Hair Curves object, each point is pulled with **Strength** times its weight. |
| **Fix Weight Threshold**        | `fix_weight_threshold`              | 0.5 (0 to 1)     | **Solid** only, hard pins only. See [Pinning Part of a Solid](#pinning-part-of-a-solid). |
| **Allow Intersections Here**    | `allow_intersection`                | off              | Let the elements this pin holds completely pass through whatever they meet, with no contact. |
| **Track Rest-Pose Deformation** | `track_rest_pose_deformation`       | off              | **Solid** only. See [Track Rest-Pose Deformation](#track-rest-pose-deformation). |

## Operations Reference

| UI label          | `op_type` value | Parameters (UI labels)                                                                                        | Description |
| ----------------- | --------------- | ------------------------------------------------------------------------------------------------------------- | ----------- |
| **[Embedded] Move** | `EMBEDDED_MOVE` | none                                                                                                        | Plays back per-vertex animation. Added by **Make Keyframe**, or by **Capture Deformation**, which shows it as **`[Embedded] Move (Captured)`**. |
| **Move By**       | `MOVE_BY`       | **Delta (m)**, **Start**, **End**, **Transition**                                                             | Translate the pinned vertices by a delta over a frame range. |
| **Spin**          | `SPIN`          | **Center** and its companion field, **Axis**, **Angular Velocity (°/s)**, **Flip Direction**, **Start**, **End** | Rotate the pinned vertices about an axis through a pivot at a constant angular velocity. |
| **Scale**         | `SCALE`         | **Center** and its companion field, **Factor**, **Start**, **End**, **Transition**                            | Scale the pinned vertices uniformly from a pivot. |
| **Torque**        | `TORQUE`        | **Magnitude (N·m)**, **Axis** (**1st** / **2nd** / **3rd Component**), **Flip Direction**, **Start**, **End** | Apply a rotational force about a principal axis of the pinned vertices. |

Defaults of a new operation, apart from its frame window (see
[Editing an Operation](#editing-an-operation)):

- **Move By**: **Delta (m)** is `(0, 0, 0)`.
- **Spin**: **Axis** is `(1, 0, 0)` (a **Static** op starts at `(0, 0, 1)`),
  **Angular Velocity (°/s)** is 360, **Flip Direction** is off, and **Center**
  is **Centroid**.
- **Scale**: **Factor** is 1.0, with a minimum of 0.01, and **Center** is
  **Centroid**.
- **Torque**: **Magnitude (N·m)** is 1.0, **Axis** is **3rd Component**, and
  **Flip Direction** is off.
- All centers: **Vertex Index** is -1 (no vertex picked) and the **Direction**
  of **Max Towards** is `(0, 0, -1)`.
- All operations: **Transition** is **Linear**, and **Start** and **End** have
  a minimum of 1.

The properties behind the parameters, as set from Python or a profile file:

| Parameter                       | Property                                        |
| ------------------------------- | ----------------------------------------------- |
| **Delta (m)**                   | `delta`                                         |
| **Axis** (**Spin**)             | `spin_axis`                                     |
| **Angular Velocity (°/s)**      | `spin_angular_velocity`                         |
| **Flip Direction** (**Spin**)   | `spin_flip`                                     |
| **Factor**                      | `scale_factor`                                  |
| **Magnitude (N·m)**             | `torque_magnitude`                              |
| **Axis** (**Torque**)           | `torque_axis_component`: `PC1`, `PC2`, `PC3`    |
| **Flip Direction** (**Torque**) | `torque_flip`                                   |
| **Start** / **End**             | `frame_start` / `frame_end`                     |
| **Transition**                  | `transition`: `LINEAR` or `SMOOTH`              |

The **Center** fields of **Spin** are `spin_center*` and those of **Scale** are
`scale_center*`, with identical suffixes:

- `*_center_mode`: the mode (`CENTROID`, `ABSOLUTE` for **Fixed**,
  `MAX_TOWARDS`, or `VERTEX`).
- `*_center`: the XYZ used by **Fixed**.
- `*_center_direction`: the direction used by **Max Towards**.
- `*_center_vertex`: the vertex index used by **Vertex**.

## Blender Python API

The Python API mirrors the panel: you create a pin by vertex-group name and
chain operations onto it. The import in the example below works for an add-on
in the **User Default** extension repository. For an add-on installed from a
remote repository, use the lookup under
[Blender Python API: Import](../../integrations/python_api.md#import).

```python
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

cloth = solver.create_group("Cloth", "SHELL")
cloth.add("Shirt")

# A plain pin: the vertices stay fixed.
shoulder = cloth.create_pin("Shirt", "ShoulderPins")

# Turn the collar about the world Z axis from frame 1 to frame 60.
collar = cloth.create_pin("Shirt", "CollarPins")
collar.spin(axis=[0, 0, 1], angular_velocity=360, frame_start=1, frame_end=60)

# Shrink the hem to half its size about its lowest vertices, easing in
# and out.
hem = cloth.create_pin("Shirt", "HemPins")
hem.scale(
    factor=0.5,
    center_direction=(0, 0, -1),   # MAX_TOWARDS: pivot = lowest vertices
    frame_start=1, frame_end=60,
    transition="SMOOTH",
)

# Release the shoulder pin 90 frames after the solve starts.
shoulder.unpin(frame=90)
```

The full pin surface (`pull`, `move_by`, `torque`, `set_animation`, `delete`),
the rules for choosing the center mode from keyword arguments, and the
behavior of a pin proxy whose pin has been deleted are on the
[Python API page](../../integrations/python_api.md#pins-and-operations).
For a curve or a Hair Curves object, `create_pin` takes the point `indices`
to hold, and for hair a `weights` list beside them makes a weighted pin (see
[Hair Curves](../scene/hair_curves.md#python-and-mcp)).
`set_animation(positions)` writes the same capture that **Capture
Deformation** writes, with no deformer needed; see
[`set_animation` from Per-Frame Positions](../../integrations/python_api.md#set_animation-from-per-frame-positions).

## MCP Tools

An MCP client reaches the same controls through tools. Most of them address a
group by its `group_uuid` (from `create_group`, or the `uuid` of an entry of
`get_active_groups`) and a pin by a `vertex_group_identifier` of the form
`object_name::vertex_group_name`, for example `Shirt::ShoulderPins`. `list_pins(group_uuid)` reports that identifier
for each pin, with its **Duration** and **Pull** settings, whether it is a
weighted pin (`weighted`) and its number of operations, in the order of the
list.

- **Pins list.** The vertex group of a mesh pin has to exist already.
  `list_vertex_groups(object_name)` lists the vertex groups of a mesh with
  their vertex counts, and refuses an object that is in Edit Mode.
  `create_vertex_group(object_name, name, indices, weight)` builds one from
  vertex indices and refuses a name the object already has.
  `add_pin_vertex_group` and `remove_pin_vertex_group` add a pin to the list
  and remove it. For a curve, `add_pin_vertex_group` takes `indices` for the
  control points. For a Hair Curves object it takes `indices` for the points,
  and `weights`, one per index and each above 0 and at most 1, to make a
  weighted pin; `weights` is refused for any other object. Without `indices`
  the name has to be an attribute the hair carries, or `Roots`.
  `rename_pin_vertex_group(..., new_name)` renames the vertex
  group together with the pin. `move_pin_vertex_group(..., direction)` moves a
  pin one place `UP` or `DOWN`. For a vertex covered by two pins, the lower pin
  in the list supplies the settings.
- **Pin settings.** `set_pin_settings` takes `use_pin_duration`,
  `pin_duration`, `use_pull`, `pull_strength`, `fix_weight_threshold`,
  `allow_intersection` and `track_rest_pose_deformation`, and an omitted
  argument stays as it is. `fix_weight_threshold` needs a **Solid** group.
  `track_rest_pose_deformation` needs a **Solid** group, a pin that covers
  every vertex of its mesh, and a captured deformation. A call that breaks one
  of these rules is refused and names what is missing.
- **Operations.** `add_pin_operation` takes an `op_type` (`MOVE_BY`, `SPIN`,
  `SCALE` or `TORQUE`) and the property keys of
  [Operations Reference](#operations-reference). It puts the new operation at
  the head of the list, and the list order is the order in which the
  operations are applied. `set_pin_operation` changes fields in place and keeps
  the order. `move_pin_operation` and `remove_pin_operation` address an
  operation by its zero-based `index`, in the order `list_pin_operations`
  reports, and `clear_pin_operations` empties the list. `copy_pin_operations`
  and `paste_pin_operations` use the **Copy Pin Operations** clipboard: a paste
  replaces the operations of the target pin, and the clipboard is empty after a
  Blender restart. The rules of the panel apply: a **Torque** cannot share a
  pin with other operations, and a pin takes operations or keyframes, not both.
- **Profiles.** `list_profiles`, `load_profile`, `save_profile` and
  `clear_profile_path` with kind `PIN` drive the **Profile** row (see
  [MCP Scene Setup](../../integrations/mcp_scene_setup.md#profiles)).
- **Keyframes.** `add_pin_keyframe` is **Make Keyframe**: it keys the pin's
  vertices at the scene's current frame, so set the frame and pose the mesh
  first. `delete_pin_keyframes` is **Delete All Keyframes** and removes the
  keys of the whole mesh. Both are refused on a curve pin.
- **Capture Deformation.** `capture_pin_deformation`, `clear_pin_deformation`
  (**Clear Deformation Cache**) and `get_pin_deformation_status` mirror the
  buttons. A capture runs in the background: poll `get_modal_job_status` until
  its `pin_deformation_capture` job reports `running` false, then read
  `frame_count` from `get_pin_deformation_status`.
  `abort_pin_deformation_capture` stops a running capture.
  `capture_pin_deformation` is refused for a pin whose object has no enabled
  deforming modifier, as the grayed button is. `add_pin_keyframe` records only
  poses set by hand; to send the positions of shape keys or drivers, use
  `set_animation` from Python.

See the [MCP Tool Reference](../../integrations/mcp_reference.rst) for the
signatures.
