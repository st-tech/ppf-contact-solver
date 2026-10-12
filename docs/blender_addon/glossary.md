# 📖 Glossary

Short definitions for the terms used across the add-on documentation, grouped
by subject and alphabetized within each group. **Scene and Constraints** covers
object groups and the **Dynamics Groups** panel, their types, pins and their
operations, stitches, colliders, and the **Snap and Merge**, **Mesh Repair**,
and **Visualization** panels. **Parameters and Simulation** covers the **Scene
Configuration** and **Solver** panels, the settings in **Scene Configuration**
and in a group's material parameters, and the steps from **Transfer** to
**Bake**. **Connections and Integrations** covers connection types, the server,
the MCP server, and the Blender Python API. Most entries link to the page that
explains the term in full.

## Scene and Constraints

**Apply Soft Constraints**
: A checkbox in the **Material Params** box of a **Static** group that holds
  the collider with springs instead of locking it to its animation. Where a
  body pushes harder than the springs hold, the collider yields, and it
  returns to its animation once the pinch passes. **Stiffness**, drawn only
  while the box is on, sets how firmly the springs hold. See
  [Apply Soft Constraints](workflow/params/material.md#apply-soft-constraints).

**Capture Deformation**
: A button that records how a deforming mesh moves, so a pin or a collider
  follows that motion during the solve without per-frame keyframes. It is
  offered on a pin (mesh objects only) and on a **Static** object, for a mesh
  driven by an **Armature**, a **Lattice**, or a similar deformer. On a pin it
  adds an **[Embedded] Move (Captured)** row to the **Operations** list.
  **Clear Deformation Cache** discards one recording, and **Re-capture All
  Deformations** and **Clear All Deformations** in the **Deformations** box of
  the **Solver** panel redo or delete every recording in the scene. See
  [Capture Deformation](workflow/constraints/pins.md#capture-deformation) for
  pins,
  [Armature-Driven Static Objects](workflow/scene/static_objects.md#armature-driven-static-objects)
  for colliders, and
  [Re-capturing Every Deformation at Once](workflow/scene/static_objects.md#re-capturing-every-deformation-at-once).

**Center**
: The dropdown that sets the pivot of a pin's **Spin** or **Scale** operation
  (**Spin** and **Scale** in **Static ops** always act about the object's
  origin): **Centroid** (the mean of the vertex positions), **Fixed** (a point
  in world space), **Max Towards** (the centroid of the vertices furthest
  along a **Direction**), or **Vertex** (one vertex you pick in Edit Mode).
  See
  [Center-Mode Dropdown and Overlays](workflow/constraints/pins.md#center-mode-dropdown-and-overlays).

**Collision Active Duration Windows**
: A toggle in the **Material Params** box of a **Solid**, **Shell**, **Rod**,
  or **PDRD** group that limits contact on each assigned object to frame
  ranges called **Active Windows**, up to 8 per object. Outside every window
  the object's contact is muted, and an object with no windows collides for
  the whole timeline. Invisible colliders and **Static** groups are not
  affected. See
  [Active Collision Windows](workflow/scene/object_groups.md#active-collision-windows).

**Duration**
: A pin setting: the **Duration** checkbox with an **Active For** frame count
  (default 60). The pin is released that many frames after the solve's
  **Starting Frame**, and its vertices return to free dynamics. On a **PDRD**
  group the same controls read **Release** and **After (frames)**. See
  [Pin Settings](workflow/constraints/pins.md#pin-settings).

**Dynamics Groups**
: The sidebar panel where object groups are created and edited. **Create
  Group** adds a group box and **Delete All Groups** removes every group. Each
  group box holds the group's **Name** and type, its assigned objects, its
  **Pins** section (**Transform** on a **Static** group, **Pins & Motion** on a
  **PDRD** group), and its **Material Params**. See
  [The Dynamics Groups Panel](workflow/scene/object_groups.md#the-dynamics-groups-panel).

**[Embedded] Move**
: A row the add-on adds to a pin by itself when **Make Keyframe** or
  **Capture Deformation** records per-vertex motion on it. The **Operations**
  list shows it as **[Embedded] Move**, or **[Embedded] Move (Captured)**
  after a capture. It is not offered in the **Add** dropdown, and it makes the
  pinned vertices follow the recorded motion. **Delete All Keyframes** removes
  the row after **Make Keyframe**, and **Clear Deformation Cache** removes it
  after a capture when the pin has no keyframes from **Make Keyframe**. See
  [Pins That Follow Animation](workflow/constraints/pins.md#pins-that-follow-animation).

**Hair Curves**
: Blender's hair object, which **Empty Hair**, **Fur** and **Quick Fur**
  create. It joins a **Rod** group, and every point of every curve is a
  vertex of a simulated strand. Its pins are point attributes, with a derived
  **Roots** pin, and a Float attribute makes a **weighted pin**. See
  [Hair Curves](workflow/scene/hair_curves.md).

**Include**
: The checkbox at the start of each row of a group's object list. Unchecking it
  keeps the object in the list but leaves it out of the next **Transfer** and
  resets its viewport color to white, so a scene can be compared with and
  without it. See
  [The Object List](workflow/scene/object_groups.md#the-object-list).

**Invisible collider**
: A wall or sphere that the solver treats as a collision boundary, added in
  the **Invisible Colliders** box of **Scene Configuration**. It has no mesh,
  so it never appears in a render, and a preview overlay shows it in the
  viewport. A wall is an infinite plane. A sphere supports **Invert**
  (collision on the inside surface) and **Hemisphere** (bowl shape, top half
  open). A mesh in a **Static** group is a Static collider and is a different
  thing. See [Invisible Colliders](workflow/constraints/colliders.md).

**Merge pair**
: Two objects that the solver sews together with a soft stitch. **Snap A to
  B** creates the pair and records its stitch points, and a pair without
  stitch points is refused at **Transfer**. Merge pairs are listed in the
  **Merge Pairs** box and saved with the `.blend`. See
  [Snap and Merge](workflow/constraints/snap_merge.md).

**Mesh Cleaning**
: The box in the **Mesh Repair** panel that scans the selected mesh objects
  for geometry that stops a **Transfer** or a solve, and offers a repair
  button for each finding that has a safe repair. Scanning never changes a
  mesh, and each repair is a separate button press. See
  [Mesh Cleaning](workflow/scene/mesh_cleaning.md).

**Mesh Repair**
: The panel of mesh preparation tools that belong to no single group: **Mesh
  Cleaning**, **Symmetric Triangulate**, **Intersection Correction** and
  **Resolve All Intersections**. It is collapsed by default. See
  [Mesh Cleaning](workflow/scene/mesh_cleaning.md).

**Object group**
: A named set of assigned objects that share one type (**Solid**, **Shell**,
  **Rod**, **PDRD**, **Sand**, or **Static**) and one set of material
  parameters. Groups are created in the **Dynamics Groups** panel, so some
  pages also call one a dynamics group. A scene holds at most 32 active
  groups. Groups other than **Static** also hold the pins of their objects.
  See [Object Groups](workflow/scene/object_groups.md); **Static** groups are
  covered separately in [Static Objects](workflow/scene/static_objects.md).

**Operation**
: A timed action stacked on a pin: **Move By**, **Spin**, **Scale**, or
  **Torque**. Each runs between a **Start** and an **End** frame. **Torque**
  cannot share a pin with the other three, and a pin that has an **[Embedded]
  Move** accepts no operation at all. A **PDRD** pin offers only **Translate**
  and **Rotate** steps. See
  [Pins and Operations](workflow/constraints/pins.md).

**Overlay Color**
: A checkbox on each group, with a color swatch beside it, that tints the
  group's objects in the viewport so you can see which group an object
  belongs to. The default color depends on the group type. The tint is a
  viewport preview only and is not sent to the solver. See
  [Overlay Colors](workflow/scene/object_groups.md#overlay-colors).

**PDRD**
: Group type for exactly rigid bodies. The mesh moves as one rigid body, a
  translation plus a rotation, and never stretches or bends, so it is not
  tetrahedralized. The surface must be closed, because the body's mass is its
  density times the enclosed volume. Its material rows are **Density** and
  **Friction**. PDRD stands for Painless Differentiable Rotation Dynamics. See
  [PDRD-Specific](workflow/params/material.md#pdrd-specific).

**Pin**
: A named set of vertices on an object that the solver holds in place. A pin
  can carry **operations**, recorded motion, a **Pull** strength, and a
  **Duration**. On a mesh a pin is a Blender vertex group. A curve has no
  vertex groups, so on a curve rod the add-on stores the pinned control
  points on the curve itself. On a Hair Curves object a pin is a point
  attribute, and one that holds a weight per point is a **weighted pin**. A
  **PDRD** group draws its pins in **Pins &
  Motion**, where a pin on one vertex is a pivot, a pin on an edge is an axis,
  and a pin on three or more vertices locks the body. A **PDRD** pin carries
  neither **Pull** nor recorded motion. A **Static** group has no pins and
  draws **Transform** in the same place. See
  [Pins and Operations](workflow/constraints/pins.md).

**Pull**
: A pin setting: the **Pull** checkbox with a **Strength** value (default
  1.0). It replaces the exact hold with a soft force that pulls the vertices
  toward their target positions. A pulled vertex does not fall or swing on
  its own: it leaves its target only as far as the rest of the object or a
  contact drags it. See
  [Pin Settings](workflow/constraints/pins.md#pin-settings).

**Resolve All Intersections**
: The button in the **Mesh Repair** panel that moves overlapping **Solid** and
  **Pdrd** objects apart and then reshapes **Shell** cloth until no surface is
  inside another's contact offsets or its own. It edits the cloth's vertex
  coordinates, runs in the background and can be aborted. See
  [Resolve All Intersections](workflow/scene/mesh_cleaning.md#resolve-all-intersections).

**Rod**
: Group type for 1D structures (ropes, wires, threads, hair). Accepts mesh
  objects, curve objects and Hair Curves objects. See
  [Group Types](workflow/scene/object_groups.md#group-types).

**Sand**
: Group type for granular bodies: a cloud of grains of one radius rather than
  a connected mesh. A mesh is converted to grains first with **Convert To
  Solid Particle Mesh**. See
  [Sand-Specific](workflow/params/material.md#sand-specific).

**Shell**
: Group type for thin deformable surfaces (cloth, fabric). Accepts mesh
  objects. See [Group Types](workflow/scene/object_groups.md#group-types).

**Snap A to B**
: The button that moves object A next to object B and leaves a small contact
  gap between them, records a stitch point for every vertex within reach, and
  registers the two objects as a **merge pair**. A **Static** object never
  moves, so when A is **Static**, object B moves instead. See
  [Snap and Merge](workflow/constraints/snap_merge.md).

**Snap and Merge**
: The panel that holds **Snap A to B**, the **Merge Pairs** box, and the
  **Post Snap Exactly** checkbox. It is collapsed by default. See
  [Snap and Merge](workflow/constraints/snap_merge.md).

**Solid**
: Group type for volumetric deformable bodies, which the solver
  tetrahedralizes. See
  [Group Types](workflow/scene/object_groups.md#group-types).

**Static**
: Group type for collision objects that the solver does not deform (ground
  planes, props, mannequins), also called Static colliders. Exposes friction,
  [**Apply Soft Constraints**](workflow/params/material.md#apply-soft-constraints),
  and contact settings. An object moves through **Static ops**, through its
  own Blender animation (keyframes, a parent, a constraint, a driver, or an
  NLA strip), or through a captured deformation. See
  [Static Objects](workflow/scene/static_objects.md).

**Static ops**
: **Move By**, **Spin**, and **Scale** entries that you add to a **Static**
  object in the **Transform** sub-box of its group, to give it a simple
  motion over time. See
  [The Transform Sub-Box](workflow/scene/static_objects.md#the-transform-sub-box).

**Stitch points**
: The points that **Snap A to B** records for a **merge pair**: for each
  stitched vertex, the point on the other object that it is stitched to. They
  stay valid while both meshes keep their vertex count, and **Re-snap**
  records them again. Some pages also call a merge pair's stitch a
  cross-stitch, and these points its stitch anchors. See
  [What a Snap Records](workflow/constraints/snap_merge.md#what-a-snap-records).

**Stitch Stiffness**
: Strength of the soft force that holds a stitch together (default 1.0). The
  **Merge Pairs** box sets it for each merge pair, and a **Shell** or
  **Solid** group sets it for the loose edges of its meshes. A stitch is
  always a soft force, never an exact weld. It is available for every
  stitchable pair: Shell-Shell, Shell-Solid, Rod-Shell, Rod-Solid, Rod-Rod,
  Solid-Solid, and a **Shell**, **Solid**, or **Rod** stitched to a
  **Static** collider.

**Torque**
: A pin **Operation** that applies a rotational force about a principal axis
  of the pinned vertices, chosen with **Axis** (**1st**, **2nd**, or **3rd
  Component**). It cannot share a pin with the other operations. See
  [Torque](workflow/constraints/pins.md#torque).

**UUID**
: The stable ID the add-on stores on each group and assigned object. A pin and
  a merge pair refer to their objects by that ID, so renaming an object or a
  pin's vertex group does not break the references between them. A pin's
  vertex group is matched by a hash of the vertex indices it holds. **Run UUID
  Migration**, drawn under **Debug Options**, adds any missing ID, and
  **Transfer** refuses a scene with incomplete IDs. See
  [UUID Migration](troubleshooting.md#uuid-migration).

**Visualization**
: The panel of five checkboxes that each hide one overlay category in the
  viewport: pins, directional arrows, overlaid colors, snaps, and pin
  operations. They change only what the viewport draws. See
  [Tour of the UI](getting_started/tour.md#visualization).

## Parameters and Simulation

**Allow Intersections**
: The box in a group's **Material Params**, drawn on every group type, whose
  checkboxes take chosen pairs of geometry out of contact: an allowed pair has
  no contact force, and its overlap is never reported as an error. A group has
  four (**Allow Self-Intersections**, **Allow Inter-Object Intersections**,
  **Allow Inter-Group Intersections**, **Allow Existing Intersections**; a
  **Sand** group has no **Allow Existing Intersections**), and a pin carries
  one (**Allow Intersections Here**). Invisible walls and spheres are never
  affected. Some pages also call these settings intersection allowances. See
  [Allow Intersections](workflow/params/material.md#allow-intersections).

**Bake**
: Converts the fetched animation into standard Blender data: per-frame Shape
  Keys on meshes and per-control-point keyframes on curves. **Bake
  Animation** keyframes every fetched frame, and **Bake Single Frame**
  freezes the current frame's pose. A baked object plays back without the
  add-on installed. See [Baking Animation](workflow/sim/baking.md).

**Bend Stiffness**
: Per-group resistance to bending on **Shell** and **Rod** groups. Both types
  store it as a single `bend` value and scale it on their own terms. On a rod
  it is measured against a one-centimeter reference segment, so a strand
  bends the same way however many segments it is drawn with. See
  [Bend Stiffness on a Rod](workflow/params/material.md#bend-stiffness-on-a-rod).

**Checkpoint**
: A saved state of a run that **Resume** can continue from. The solver writes
  one at each frame in the **Save Checkpoints** list, at every **Auto Save
  Interval** while **Auto Save** is on, on the last frame when **Save State on
  Finish** is on, and when you press **Save and Quit**. See
  [Save and Checkpoints](workflow/params/scene.md#save-and-checkpoints).

**Clear Local Animation**
: Removes the fetched animation from the simulated objects, forgets which
  frames were fetched, and sets the playhead to the **Starting Frame**.
  Captured pin animation is kept, and the solver host is not touched. While
  fetched animation is attached, **Run** stays grayed out until this button
  removes it. See
  [The Buttons](workflow/sim/simulating.md#the-buttons).

**Contact Gap**
: The distance at which the solver starts pushing two surfaces apart. It is
  set per group or per invisible collider, as an absolute length in
  Blender units or, on the group types that offer the choice, as a ratio of
  each object's own bounding-box diagonal. On **Solid**, **Shell** and **Rod**
  groups the **Local Gap Limit** can lower it where the mesh is fine. See
  [Contact Gap and Contact Offset](workflow/params/material.md#contact-gap-and-contact-offset).

**Contact Offset**
: Per-group padding added on top of the **Contact Gap**. Two surfaces are in
  contact once their distance is below the mean of their two gaps plus the sum
  of their two offsets. Like the gap, it can be an absolute length or a ratio
  of the bounding-box diagonal. See
  [Contact Gap and Contact Offset](workflow/params/material.md#contact-gap-and-contact-offset).

**Delete Remote Data**
: Wipes the project data on the solver host, cached results included. The next
  step is **Transfer**. See
  [The Buttons](workflow/sim/simulating.md#the-buttons).

**Density-Normalized (Pa/ρ)**
: A checkbox beside **Young's Modulus** on **Solid**, **Shell** and **Rod**
  groups. On (the default), the modulus is in Pa/ρ, the unit the solver reads,
  so changing **Density** alone does not change the motion. Off, the field is
  a true modulus in pascals and the add-on divides it by the group's density.
  See
  [Density-Normalized (Pa/ρ)](workflow/params/material.md#density-normalized-paρ).

**Dynamic parameter**
: A scene setting whose value changes over time through keyframes on its own
  slider in **Scene Configuration**. Six can be keyframed: **Gravity**,
  **Wind**, **Air Density**, **Air Friction**, **Vertex Air Damping**, and
  **Step Size**. See [Dynamic Parameters](workflow/params/dynamic.md).

**Export USD and Export Alembic (ABC)**
: Buttons in the **Export** box of the **Solver** panel that write the fetched
  animation to a point cache file another application can read, once every
  frame has been fetched. The scene is left as it is, so the export can be
  repeated, which **Bake** cannot. Rods are not carried. See
  [Exporting USD and Alembic Caches](workflow/sim/exporting.md).

**Fetch All Animation**
: The button that downloads the frames the scene does not have yet from the
  solver host and applies them to the simulated objects, so the timeline
  plays back the result. See
  [How Animation Plays Back](workflow/sim/simulating.md#how-animation-plays-back).

**Force Fields**
: The box in **Scene Configuration** that sends Blender's **Force**, **Wind**,
  **Vortex**, and **Turbulence** field objects to the solver, so they push the
  simulated objects. A short Python script can give a push that no field
  object can describe. Each group except a **Static** one also has a **Force
  Field Weight** setting. See [Force Fields](workflow/params/force_fields.md).

**Hinge**
: A per-object setting on **PDRD** bodies that pins the body and locks its
  rotation to one principal axis of its rest shape, so the body spins on that
  axis only. See [Hinge Joints](workflow/params/material.md#hinge-joints).

**JupyterLab**
: The notebook environment on the solver host. **Export** in the
  **JupyterLab** box of the **Solver** panel writes a template notebook to the
  solver host, which can run the simulation after a **Transfer** while Blender
  stays closed. See
  [JupyterLab](workflow/sim/jupyterlab.md).

**Local Gap Limit**
: A percentage on **Solid**, **Shell** and **Rod** groups that caps the
  contact gap of each triangle or strand segment at that share of the room
  its own mesh neighbors leave, so a mesh finer than its gap does not push
  itself apart. `25` by default. It follows Gomez-Nogales et al.,
  "Unlocking Thickness Modeling for Codimensional Contact Simulation"
  ([arXiv:2510.10256](https://arxiv.org/abs/2510.10256), 2025), which
  names the effect *contact locking*. See
  [Local Gap Limit](workflow/params/material.md#local-gap-limit).

**Lock Translation / Lock Rotation**
: Two per-object checkboxes in the **Lock Translation** box of every dynamic
  group type. **Lock Translation** confines an object's center of mass to a
  fixed world-space line, or to its initial point with **Lock All
  Translations**. **Lock Rotation** restricts the object's rigid rotation to
  one world-space axis, or forbids rotation with **Lock All Rotations**. With
  **Prohibit Rotation on Axis** on, rotation about that axis is forbidden and
  the perpendicular plane stays free. Deformation stays free. See
  [Lock Translation and Lock Rotation](workflow/params/material.md#lock-translation-and-lock-rotation).

**Material parameter**
: A per-group property that controls deformation, density, stiffness,
  friction, and contact. The fields a group shows depend on its type. See
  [Material Parameters](workflow/params/material.md).

**Mesh topology changed warning**
: The warning *Mesh topology changed since last transfer (groups differing:
  ...). Re-transfer to sync.* that **Run**, **Resume**, and **Fetch All
  Animation** report when a group's vertex, polygon, edge, or object count, the
  vertex count of a pin, or a captured **Static** deformation differs from what
  the last **Transfer** sent. It does not block the action. See
  [Mesh Topology Changed Warning](workflow/sim/simulating.md#mesh-topology-changed-warning).

**Model**
: The **Model** dropdown of a group, which sets how the group's material
  deforms (its material model). **Shell** offers **Baraff-Witkin** and
  **ARAP**, and **Solid** offers **Stable NeoHookean** and **ARAP**. A **Rod**
  always uses ARAP and has no **Model** row. See
  [Rows by Group Type](workflow/params/material.md#rows-by-group-type).

**Object Statistics**
: The last panel in the add-on tab, collapsed by default. It shows what the
  solver measured for one object, chosen in the **Object** menu, on the frame
  under the playhead, and **Export CSV** writes one value for every fetched
  frame. See
  [Reading Per-Frame Statistics](workflow/sim/simulating.md#reading-per-frame-statistics).

**Padlock**
: The lock icon at the right end of most numeric rows in **Material Params**.
  A closed padlock keeps the row's value when you apply a **Preset** or use
  **Paste Material Params**. It does not stop keyframes, and a material
  profile ignores it. See
  [Padlocks](workflow/params/material.md#padlocks).

**PC2**
: The file format behind the `ContactSolverCache` modifier, a Mesh Cache
  modifier in the **Modifier Properties** tab whose **Format** reads PC2.
  **Fetch All Animation** writes one file per simulated object into a `data`
  folder next to the `.blend`, and the modifier plays it back on a mesh. The
  add-on updates the control points of a curve rod on every frame change. The
  first solver frame lands on the **Starting Frame** and each later frame on
  the next Blender frame. See
  [How Animation Plays Back](workflow/sim/simulating.md#how-animation-plays-back).

**Plasticity**
: A material setting on **Shell** and **Solid** groups that adds permanent
  deformation to the elastic response: where the local stretch differs from the
  rest size by more than the **Threshold**, the rest shape drifts toward the
  current shape at a rate set by **Theta**. **Shell** and **Rod** groups also
  have **Bend Plasticity** for the bending rest angle. See
  [Plasticity](workflow/params/material.md#plasticity).

**Preset**
: The dropdown in a **Shell** or **Solid** group's **Material Params** box that
  applies a bundled material, six fabrics for a **Shell** and five soft solids
  for a **Solid**, in one click. It sets only the rows the preset names, never
  changes the group's **Type**, and leaves rows with a closed padlock alone. A
  **Profile** is a file you save yourself. See
  [Fabric Presets](workflow/params/fabric_presets.md).

**Profile**
: A named set of values saved in a TOML file. The add-on has four kinds: a
  scene profile (the scene settings and the invisible colliders; keyframes
  drawn on sliders stay in the `.blend`), a material profile (one group's
  material parameters), a pin-operations profile (one pin's **Operations**),
  and a connection profile (see **Connection profile**). Each is loaded and
  saved from the profile row of its panel. See
  [Scene Profiles](workflow/params/scene.md#scene-profiles),
  [Material Profiles](workflow/params/material.md#material-profiles), and
  [Saving Operations to a Profile](workflow/constraints/pins.md#saving-operations-to-a-profile).

**Rayleigh Damping**
: Damping proportional to a group's stiffness, set per group by
  **Deformation Damping** and **Bending Damping** (both default 0.0, which
  disables it). **Deformation Damping** applies to **Solid**, **Shell**, and
  **Rod** groups, and **Bending Damping** to **Shell** and **Rod** groups
  only. Air drag (**Air Friction**) and **Vertex Air Damping** are separate
  and act on the whole scene. See
  [Rayleigh Damping](workflow/params/material.md#rayleigh-damping).

**Resume**
: Continues a simulation that stopped or failed, from a saved checkpoint, and
  retains the frames before it. The **Resume** button opens a picker of the
  saved checkpoints, and the frames after the chosen one are overwritten. It
  is available while the status reads **Resumable** or **Simulation Failed**
  and the solver host holds at least one checkpoint. It refuses when the
  geometry or parameters differ from the last upload. See
  [Resuming a Run](workflow/sim/simulating.md#resuming-a-run).

**Run**
: Starts the simulation on the solver host. It refuses when the geometry or
  parameters differ from the last upload (use **Transfer** or **Update Params
  on Remote**), and it reports the **Mesh topology changed warning** when the
  topology differs from the last **Transfer**. The button stays grayed out
  while fetched animation is attached to the objects, until **Clear Local
  Animation** removes it. See
  [Running the Simulation](workflow/sim/simulating.md).

**Save and Quit**
: A button on **Backend Communicator**, drawn while a solve runs, that asks the
  solve to write a state and stop. The status line then reads **Resumable**,
  and **Resume** lists the new checkpoint. See
  [The Buttons](workflow/sim/simulating.md#the-buttons).

**Scene Configuration**
: The sidebar panel for the settings that apply to the whole simulation:
  **FPS**, **Time Scale**, **Starting Frame**, **Frame Count**, **Step Size**,
  **Min Newton Steps**, **Air Density**, **Air Friction**, **World Scaling**,
  **Gravity**, **Preview Direction**, and **Inactive Momentum Frames**, plus
  the collapsible **Save and Checkpoints**, **Wind**, **Force Fields**,
  **Invisible Colliders**, **Linear System Solver**, and **Advanced Params**
  boxes. See
  [Scene Configuration Panel](workflow/params/scene.md#scene-configuration-panel).

**Scene parameter**
: A setting that applies to the whole scene rather than to one group, such as
  **Frame Count**, **Step Size**, **Gravity**, wind, **Air Density**, and the
  solver settings. See [Scene Parameters](workflow/params/scene.md).

**Shrink**
: Rest-shape scale in a group's material parameters: **Shrink X** and
  **Shrink Y** on a **Shell**, a single **Shrink** on a **Solid**, and
  **Shrink** on a **Rod** (the rest length of every segment). Below 1.0 the
  rest shape is smaller than the drawn geometry, and above 1.0 it is larger.
  The solver drives the mesh toward the rest shape. See
  [Shrink X / Shrink Y](workflow/params/material.md#shrink-x--shrink-y),
  [Shrink](workflow/params/material.md#shrink), and
  [Shrink on a Rod](workflow/params/material.md#shrink-on-a-rod).

**Solver**
: The sidebar panel below **Backend Communicator** that holds the buttons that
  drive a simulation: **Transfer** and **Update Params on Remote**, **Run** and
  **Resume**, **Fetch All Animation** and **Delete Remote Data**, **Bake
  Animation** and **Bake Single Frame**, and **Clear Local Animation**. It
  also holds the **Deformations** and **Export** boxes and the collapsible
  **JupyterLab** and **MCP Server** boxes. See
  [The Solver Panel](workflow/sim/simulating.md#the-solver-panel).

**Spatial Material Map**
: A row in the **Spatial Material Maps** box of a **Shell**, **Solid** or
  **Rod** group that varies one material parameter across the surface or
  along a strand, from the group's own value where a vertex weight is 0 to
  the row's **Target** where it is 1. The weights come from a vertex group or
  an attribute. See
  [Spatial Material Maps](workflow/params/material.md#spatial-material-maps).

**Starting Frame**
: The Blender frame the first solver frame lands on, and simulated time zero
  (default 1, minimum 0). Tick **Take Starting Frame from Scene** to use the
  scene's start frame instead. Pin durations count from it, and checkpoint
  frames must come after it. See
  [Scene Parameters](workflow/params/scene.md#basic).

**Status line**
: The `Status:` line of the **Backend Communicator** panel, which shows the
  solver's current state, for example **Disconnected**, **Waiting for Server
  Start...**, **Ready to Run**, **Simulation Running...**, **Resumable**,
  **Fetching Animation...**, **Simulation Failed**, and **Protocol Version
  Mismatch**, plus the transient states in between. See
  [Status Line States](workflow/sim/simulating.md#status-line-states).

**Strain Limit**
: A material setting on **Shell** and **Rod** groups: a clamp that stops mesh
  edges from stretching beyond a percentage of their rest length (default 5,
  active when **Enable Strain Limit** is on). See
  [Strain Limit](workflow/params/material.md#strain-limit).

**Terminate**
: A button on **Backend Communicator**, drawn while a solve runs, that stops
  the solve at once and writes no state. The frames simulated so far stay on
  the solver host and can be fetched. The status line then reads
  **Resumable** if a saved state exists, otherwise **Ready to Run**. See
  [The Buttons](workflow/sim/simulating.md#the-buttons).

**Tetrahedralizer**
: The per-object setting in the **Tetrahedralizer** box of a **Solid** group
  that picks **fTetWild** (the default) or **TetGen** to turn the surface into
  tetrahedra, with override rows for each. See
  [Tetrahedralizer (per object)](workflow/params/material.md#tetrahedralizer-per-object).

**Time Scale**
: A **Scene Configuration** slider (default 1.0) that sets the playback speed
  of the Blender animation in simulated time. At `0.5` keyframed motion is
  re-interpreted at half speed, so fast motion covers the same path over twice
  the simulated time. Gravity and materials do not change, and the solve cost
  grows as 1 / Time Scale. See
  [Scene Parameters](workflow/params/scene.md#basic).

**Transfer**
: Uploads geometry, pins, colliders, and every parameter to the solver host
  and builds its scene. Required whenever topology or group membership
  changes. It starts the remote project over: simulated frames and saved
  checkpoints already on the solver host are removed, so **Resume** has
  nothing to offer afterward. See
  [The Buttons](workflow/sim/simulating.md#the-buttons).

**Update Params on Remote**
: Re-encodes the parameters, uploads them without resending geometry, and
  rebuilds. Saved checkpoints are kept, so **Resume** still works afterward.
  Use it to iterate quickly on dynamics and materials. See
  [Update Params vs Transfer](workflow/sim/simulating.md#update-params-vs-transfer).

**Velocity Overwrite**
: A per-object list of keyframes in the **Material Params** box of a
  **Solid**, **Shell**, **Rod**, or **PDRD** group (titled **Velocity (free
  launch)** on a **PDRD** group) that sets the object's translational or spin
  velocity at chosen frames instead of leaving it to the simulation. See
  [Velocity Overwrite](workflow/params/material.md#velocity-overwrite).

**World Scaling**
: A **Scene Configuration** setting (default 1.0) that sets the physical size
  a scene is simulated at. The scene is scaled by this factor before the solve
  and the result is scaled back, so it keeps its authored size: at `0.1` a
  15 m mesh simulates as a 1.5 m one under the same gravity. Lengths such as
  positions, contact gaps, and velocities scale with it. Gravity, wind, and
  material parameters do not. See
  [World Scaling](workflow/params/scene.md#world-scaling).

## Connections and Integrations

**Backend Communicator**
: The main sidebar panel, where you pick a connection **Type**, connect,
  start the server on the solver host, and read the status line. See
  [Tour of the UI](getting_started/tour.md#backend-communicator).

**Blender Python API**
: The add-on's Python API, imported in Blender's Text Editor, for example
  `from bl_ext.user_default.ppf_contact_solver.ops.api import solver` when the
  add-on is in the **User Default** extension repository. A script runs the
  same actions as the sidebar buttons, without button-only steps such as
  confirmation dialogs. It is separate from the
  [JupyterLab Python API](../jupyterlab_api/index.rst), which drives the
  solver from a notebook. See
  [Blender Python API](integrations/python_api.md) and its
  [Import](integrations/python_api.md#import) section.

**Compute Device**
: The **Connection** box row that says whether a run uses the solver host's
  accelerator (**GPU**: CUDA or ROCm on Windows and Linux, Metal on macOS) or
  the portable **CPU** build. It selects which build of the solver the server
  starts from, and a selection the host cannot serve is refused rather than
  substituted. See
  {ref}`Choosing the build <choosing-the-build>`.

**Connect**
: The **Backend Communicator** button that opens the connection to wherever
  the solver lives (a local process, an SSH session, or a Docker container)
  and checks the solver folder you set. It does not start the server, and it
  stays grayed out until the required fields are valid. See
  [What Happens When You Connect](connections/index.md#what-happens-when-you-connect).

**Connection profile**
: A named entry in a TOML file that stores the connection fields of the
  **Backend Communicator** panel for one host, to switch between hosts and to
  share connection settings across a team. **Project Name**, **Compute
  Device**, and **GPU Backend** are not stored. See
  [Connection Profiles](connections/profiles.md).

**Disconnect**
: Closes the connection and leaves the server running, on every connection
  type, so the next **Connect** reaches that server again. **Stop Server on
  Remote** is the button that ends the server. See
  [Disconnecting and Stopping the Server](connections/index.md#disconnecting-and-stopping-the-server).

**Docker**
: A connection type where the solver runs inside a Docker container on the
  Docker daemon of this machine. See
  [Docker (Local)](connections/docker.md).

**Docker over SSH**
: A connection type where the solver runs in a container on a Docker host
  reached through SSH, for when you have SSH access to the host and a
  container to use. See [Docker over SSH](connections/docker_over_ssh.md).

**Docker over SSH Command**
: The **Docker over SSH** type with the host, port, user, key, and jump host
  taken from a pasted `ssh ...` command instead of separate fields. See
  [Docker over SSH](connections/docker_over_ssh.md).

**`execute_shell_command`**
: MCP tool that runs shell commands on the connected solver host, through the
  active connection (inside the container for the Docker types), and only
  while a connection is live. With `run_python_script`, which runs inside
  Blender, it covers provisioning and maintenance tasks that have no
  dedicated tool.

**Force Terminate Process**
: A recovery button drawn under a connection error: a protocol version
  mismatch, a port already in use, or any error on a native type that is not
  connected. On a native type that is not connected, or is attached to a
  server this add-on did not start, it ends every process listening on the
  server port of this machine, which can include a program that is not the
  solver, so read the line under it first. On a remote type it ends only a
  `ppf-cts-server` that was started with `--port N`. See
  [What Force Terminate Process Ends](connections/index.md#what-force-terminate-process-ends).

**GPU Backend**
: The **Connection** box row that says which accelerator a run uses when
  **Compute Device** is **GPU** and the solver host holds more than one GPU
  build: **Automatic**, **CUDA**, or **ROCm**. See
  {ref}`Choosing the build <choosing-the-build>`.

**Linux Native**
: A connection type where the solver runs directly as a child process on
  this Linux machine, on CUDA, ROCm, or the CPU build, with no SSH or
  Docker. See [Linux Native](connections/linux.md).

**macOS Native**
: A connection type where the solver runs directly as a child process on
  an Apple-silicon Mac, on Metal or the CPU build, with no SSH or
  Docker. See {ref}`macOS Native <macos-native>`.

**MCP prompt**
: A short workflow starting point that the [MCP server](integrations/mcp.md)
  offers through `prompts/list` and `prompts/get`: `run_simulation`,
  `pin_and_constrain`, `tune_parameters`, and `diagnose_failure`. See
  [Instructions and Prompts](integrations/mcp.md#instructions-and-prompts).

**MCP resource**
: A read-only asset exposed by the [MCP server](integrations/mcp.md) via
  `resources/read`. The server exposes one, the live scene snapshot
  `blender://scene/current`.

**MCP server**
: The bundled Model Context Protocol server on `localhost`, port 9633 by
  default, that exposes add-on operations as JSON-RPC tools for external
  agents. You start it from the **MCP Server** box of the **Solver** panel. See
  [MCP Server](integrations/mcp.md).

**MCP tool**
: A JSON-RPC method exposed by the [MCP server](integrations/mcp.md) and
  dispatched with `tools/call`. A tool that starts a transfer, a solve, or a
  fetch runs the same operator as the panel button and is refused when that
  operator's availability check fails. Button-only steps such as a
  confirmation dialog do not run: a bake called this way skips the dialog and
  the unfetched-frames check. See
  [Baking Animation](workflow/sim/baking.md#blender-python-api).

**Native type**
: Collectively, the three connection types whose solver is a child
  process on the machine Blender runs on, started by the add-on:
  [Windows Native](connections/windows.md),
  {ref}`macOS Native <macos-native>`, and
  [Linux Native](connections/linux.md). Only your own platform's type is
  usable. See [Connections](connections/index.md).

**`ppf-cts-server`**
: The server program (`ppf-cts-server`, or `ppf-cts-server.exe` on Windows)
  that the add-on starts on the solver host: directly on the remote host for
  **SSH** and **SSH Command**, inside the container for the Docker types, and
  as a child process for the three native types. It listens on
  a TCP port, 9090 by default, receives the scene and parameters, and runs
  the solver. The port is the **Docker Port** setting, which the panel draws
  only for the Docker types; see
  [Port Usage at a Glance](connections/index.md#port-usage-at-a-glance).
  **Start Server on Remote** starts it.

**Project Name**
: A short name for the project on the solver host, the same field on every
  connection type (default `unnamed`). It takes letters, digits, `.`, `-`, and
  `_`, and **Connect** stays grayed out while it is empty or holds anything
  else. In a saved `.blend`, a copy icon beside it fills in the `.blend` file
  name while the project name is empty or reads `unnamed`. A connection
  profile does not store it. See
  [When Connect Is Unavailable](connections/index.md#when-connect-is-unavailable).

**Protocol version**
: The version of the exchange between the add-on and the server. Both take it
  from the same shipped file, so there is no number for you to set. When the
  server reports a different version than the add-on expects, the status
  reads **Protocol Version Mismatch** and one of the two is out of date. See
  [Protocol version mismatch](troubleshooting.md#status-protocol-version-mismatch).

**Remote type**
: Collectively, the connection types whose solver runs on another machine or in
  a container, reached over SSH or through the Docker daemon: **SSH**, **SSH
  Command**, **Docker**, **Docker over SSH**, and **Docker over SSH Command**.
  See [Connections](connections/index.md).

**`run_python_script`**
: MCP tool that runs Python inside Blender, for operations that have no
  dedicated tool. See the security note in
  [MCP Server](integrations/mcp.md).

**Session ID**
: A short identifier the add-on creates each time you connect. It is saved
  with the `.blend` and stamped on the animation each run produces. When you
  reconnect, the add-on compares the saved ID with the new one and, where they
  differ, logs a line starting with `Reconcile` in the Console; the line is
  informational and blocks nothing. See
  [Sessions and Recovery](workflow/sim/simulating.md#sessions-and-recovery).

**Solver host**
: The machine or container that runs `ppf-cts-server` and holds the project's
  scene and simulated frames: this machine for a native type, the remote host
  for **SSH** and **SSH Command**, and the container for the three Docker
  types. See [Connections](connections/index.md).

**Solver Path**
: The field of a native type that names the folder holding the solver, an
  unzipped solver distribution or a repository checkout. A *Solver path valid*
  line under it confirms the folder, and **Connect** checks that it holds the
  build chosen under **Compute Device**. See
  [What Happens When You Connect](connections/index.md#what-happens-when-you-connect).

**SSH**
: A connection type where the solver runs on a remote Linux host reached by
  SSH, with credentials entered as explicit fields. See
  [SSH (Direct)](connections/ssh.md).

**SSH Command**
: A connection type that takes the host, port, user name, key path, and jump
  host from a pasted `ssh ...` command. Other standard `ssh` options in the
  command are ignored, and `~/.ssh/config` is still consulted for the host.
  Convenient when you already have such a line from a cloud provider or a
  shared ops document. See [SSH (Direct)](connections/ssh.md).

**Start Server on Remote**
: Starts `ppf-cts-server` on the solver host and waits for it to report that it
  is ready. A native type starts the server as a child process, or attaches to
  one that already listens on the port. The button reads **Server Starting...**
  while it waits, and **Cancel** stops the launch. See
  [What Happens When You Connect](connections/index.md#what-happens-when-you-connect).

**Stop Server on Remote**
: Ends the server on the solver host. **Disconnect** leaves it running. See
  [Disconnecting and Stopping the Server](connections/index.md#disconnecting-and-stopping-the-server).

**Streamable HTTP**
: The MCP transport used by the bundled MCP server. All traffic goes
  through a single `/mcp` endpoint. See
  [Protocol](integrations/mcp.md#protocol).

**Windows Native**
: A connection type where the solver runs directly as a Windows
  subprocess, on CUDA, ROCm, or the CPU build, using a bundled Python
  interpreter and no SSH or Docker.
  See [Windows Native](connections/windows.md).
