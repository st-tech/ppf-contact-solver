# 🎨 Scene Setup via MCP

The MCP tools are the recommended way for an agent or an automation script to
build a scene before it runs **Transfer** and **Run**. An agent cannot judge a
mesh by eye, so this page names the tools to call, the order to call them in,
and the numbers to check.

The server's instructions give the order a scene is built in (see
[Instructions and Prompts](mcp.md#instructions-and-prompts)): connect, create a
group, assign objects, constrain, set parameters, build, solve, fetch. Connect
before building the scene. The sections below cover the scene steps, and
[Connect, Transfer, Solve, Fetch](#connect-transfer-solve-fetch) covers
connecting and the last three steps. Group, pin, collider and parameter tools
run without a connection. The tools that act on the solver host, such as
`start_remote_server`, `get_remote_status` and `transfer_data`, refuse until
the connection is up, and `transfer_data` also refuses until the remote server
is running.

:::{tip}
Everything below is callable with `tools/call` on the MCP server. The
parameter schemas are in the [MCP Tool Reference](mcp_reference.rst).
:::

## Reading the Scene First

An agent that did not build the scene starts with `get_scene_info`. It returns
the scene's `frame_start`, `frame_end`, `frame_current` and `blender_fps`, the
simulation's `frame_count` and `fps` as the solver will use them, every scene
object with its type, visibility, vertex and face counts (spline and point
counts for a curve) and an `in_group` flag, and every active group with its
name, `uuid`, `object_type` and object names. The `blender://scene/current`
resource returns the same JSON.

Then `get_group_objects(group_uuid)` reports each assigned object's own state:
its included flag, locks, PDRD hinge, bending reference, intersection
allowances and tetrahedralizer settings, and how many static operations,
velocity keyframes and collision windows it holds. `list_pins(group_uuid)`
lists a group's pins, and `get_scene_parameters` returns the scene settings.

`clear_solver` is `solver.clear()`: it resets the add-on state of the whole
file and deletes the animation data stored on the mesh data of every mesh
object in it, whether or not the object is in a group (see
[Reset](python_api.md#reset)). Call it only when the scene should start over.

## Use MCP Tools, Not Raw Python

`run_python_script` evaluates arbitrary Python inside Blender. Use it only for
what no dedicated tool covers. When a dedicated tool exists, use it:

- Creating, deleting or typing a group: `create_group`, `delete_group`,
  `set_group_type`.
- Assigning objects to a group: `add_objects_to_group`.
- Reading and setting material parameters: `get_group_material_properties`,
  `set_group_material_properties`, and the preset tools described in the
  Material Parameters section below.
- Scene-level parameters: `set_scene_parameters`.
- Force fields: `set_force_field_settings`, `get_force_field_settings`,
  `set_force_field_targets` and `check_force_field_script`, with every
  function a script may call listed by `get_force_field_settings` under
  `script_builtins` (see
  [Force Fields](../workflow/params/force_fields.md)); a group's
  `force_field_weight` goes through `set_group_material_properties`.
- Pins: `create_vertex_group`, `add_pin_vertex_group`, `add_pin_operation`.
- Colliders: `add_static_op` for a moving static collider,
  `add_invisible_wall`, `add_invisible_sphere`.
- Merges: `snap_to_vertices` (the **Snap A to B** button: it moves object A
  next to object B, records the stitch points and registers the pair; when A
  is in a **Static** group, B moves instead), `add_merge_pair` (registers a
  pair with no stitch points), `resnap_merge_pair` (records the stitch points
  of an existing pair), `list_merge_pairs`, `set_merge_pair_properties` and
  `remove_merge_pair`.
- Rods: `create_curve`, `add_curve_spline`, `set_curve_material` and
  `finalize_curve` (see
  [Building a Rod from Control Points](#building-a-rod-from-control-points)),
  then `add_objects_to_group` and `add_pin_vertex_group` with `indices`.
- Profiles: `list_profiles`, `load_profile`, `save_profile` and
  `clear_profile_path` (see [Profiles](#profiles)).
- Sand grains: `convert_to_particle_mesh`.

The tools that add a pin vertex group, a merge pair or an invisible collider
check their input before they change the scene: a missing object, a missing
vertex group, a merge of an object with itself and a sphere radius that is not
positive are refused. `add_pin_operation` and `add_static_op` check their own
arguments, and `set_group_material_properties` refuses a value its property
cannot hold. `run_python_script` applies none of these checks.

Two further rules for agents:

- **Do not create temporary files** to stage Python code. Pass the code string
  directly to `run_python_script`.
- **Do not work around a missing tool** by writing files or launching other
  programs. If a tool is missing, report the gap.

## Placement and Clearance

When the scene is built, a start pose in which elements cross each other, or
sit closer than their contact offsets, is refused unless an intersection
allowance covers the pair (see
[Allow Intersections](../workflow/params/material.md#allow-intersections)).
Before creating groups:

1. Place the objects so that no meshes overlap. Cloth starts **outside** the
   body it will wrap, and colliders do not pierce each other.
2. As a recommendation, leave a clearance of at least the contact gap between
   objects, so that no object starts inside the contact gap of another (see
   [Contact Gap and Contact Offset](../workflow/params/material.md#contact-gap-and-contact-offset)).
   In ratio mode the gap is a fraction of each object's own bounding-box
   diagonal.

`get_object_bounding_box_diagonal(object_name)` returns the object's size in
Blender units, which turns "small clearance" into a number. It returns two
diagonals:

- `contact_length_diagonal` is the diagonal of the box around the object's own
  vertices in world space. A group in ratio mode multiplies it by its **Contact
  Gap Ratio** and **Contact Offset Ratio**.
- `largest_diagonal_distance` is the diagonal of the box around the corners of
  the object's local bounding box. It is at least as large as
  `contact_length_diagonal`, and grows when the object is rotated.

Separating the geometry is the default. Where two pieces should pass through
each other instead of colliding, `set_group_material_properties` accepts four
intersection allowances: `allow_self_intersection`,
`allow_inter_object_intersection`, `allow_inter_group_intersection` and
`allow_existing_intersection`. They are the **Allow Self-Intersections**,
**Allow Inter-Object Intersections**, **Allow Inter-Group Intersections** and
**Allow Existing Intersections** checkboxes, and a **Sand** group does not take
`allow_existing_intersection`. An allowed pair has no contact at all, no
allowance separates an overlap, and the invisible walls and spheres always
collide. Which key covers a pair is decided per Blender object, and either side
of a pair is enough; see
[Which Pairs Each Setting Covers](../workflow/params/material.md#which-pairs-each-setting-covers)
and [Only One Side Has to Allow It](../workflow/params/material.md#only-one-side-has-to-allow-it).

Each allowance reaches every object assigned to the group while its matching
`..._all_objects` switch is on, which is the default.
`set_intersection_allowance_objects` narrows one to named objects of the group;
its `allowance` argument is `self`, `inter_object`, `inter_group` or
`existing`. See
[Allow Intersections](../workflow/params/material.md#allow-intersections) and
[Allow Existing Intersections](../workflow/params/material.md#allow-existing-intersections).

## Creating Sphere Primitives

When an agent needs a spherical mesh (a ball, a drop, a filler object), create
an ico sphere (**Add > Mesh > Ico Sphere**), not a UV sphere. The triangles of a
UV sphere shrink toward its poles, so its edge length varies across the
surface. The triangles of an ico sphere are near-uniform in size, so one
average edge length describes the whole mesh.

In a `run_python_script` call, this looks like:

```python
import bpy
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.5)
```

For `subdivisions`, pick the level that puts the average edge length inside the
1–3 % window described under Mesh Resolution below, not a fixed default.

## Building a Rod from Control Points

A **Rod** group simulates curve objects. Four tools build one without
`run_python_script`:

1. `create_curve(name, bevel_depth, bevel_resolution, resolution_u, dimensions, clear_existing)`
   starts a pending curve called `name`. With `clear_existing` true, the
   default, an existing object of that name is deleted by this call.
2. `add_curve_spline(name, points, closed)` appends one Bezier spline from a
   list of `[x, y, z]` control points (at least two) and returns its
   `spline_index`. Call it once per strand.
3. Optional: `set_curve_material(name, spline_index, material_name, create_if_missing)`
   binds a material to a spline. With `create_if_missing` true, a material that
   does not exist is created.
4. `finalize_curve(name)` links the curve object into the scene and returns its
   `object_name`.
5. Assign the object to a **Rod** group with `add_objects_to_group`. A **Rod**
   group accepts curve objects, and the other group types skip one with a
   warning.
6. Pin control points with
   `add_pin_vertex_group(group_uuid, "<object>::<name>", indices=[0, 7])`. A
   curve has no vertex groups, so `indices` names the pinned control points, and
   `indices` for a mesh is refused. See
   [Pins on a Curve Rod](../workflow/constraints/pins.md#pins-on-a-curve-rod).

`get_average_edge_length`, used in the next section, accepts mesh objects only
and refuses a curve.

A **Rod** group also takes a Hair Curves object (object type `CURVES`), the
object Blender's hair tools and **Quick Fur** create. No tool builds one;
assign an existing one with `add_objects_to_group`. Its pins are point
attributes: `add_pin_vertex_group(group_uuid, "<object>::Roots")` pins the
first two points of every curve, `indices` writes and pins a set of points,
and `weights`, one per index and each above 0 and at most 1, makes a weighted
pin whose pull fades along the strand once `set_pin_settings` turns
`use_pull` on. `list_pins` reports `weighted` for each pin. A pin that is
lower in the list wins a point two pins share, so move a pin that holds the
roots exactly below a weighted pin that covers them with
`move_pin_vertex_group`. See [Hair Curves](../workflow/scene/hair_curves.md).

(mesh-resolution-window)=
## Mesh Resolution: The 1–3 % Window

The recommended **average edge length is 1–3 % of the object's bounding-box
diagonal**. A finer mesh has more elements and costs more to simulate. Two MCP
tools report the numbers without opening Blender's UI, and neither enforces a
range:

- `get_average_edge_length(object_name)`: returns the mean world-space edge
  length of a mesh object. It accepts mesh objects only.
- `get_object_bounding_box_diagonal(object_name)`: returns the two diagonals
  described under [Placement and Clearance](#placement-and-clearance). This
  check uses `contact_length_diagonal`, the same length the group's relative
  contact ratios use.

Typical check before assigning an object to a group:

```text
ratio = average_edge_length / contact_length_diagonal
if 0.01 <= ratio <= 0.03:
    assign to group
elif ratio > 0.03:
    subdivide, or recreate at higher resolution
else:  # ratio < 0.01
    finer than the recommended range: acceptable, but it costs more to simulate
```

The recommendation is for **Shell**, **Solid** and **Rod** groups. A **Sand**
group is a faceless mesh of grain centers with no edges, so
`get_average_edge_length` returns `0.0` and the ratio is undefined.

## Subdividing to Reach the Window

If a mesh is too coarse, prefer **subdivision** over remeshing. The add-on
identifies objects by a per-object UUID and pins by vertex-group name, neither
of which subdivision touches, so pins and vertex groups survive. A merge pair's
stitch anchors refer to vertices of the two meshes. Subdividing an object that
is in a merge pair changes its vertex count, and the pair is refused until it
is re-snapped, so call `resnap_merge_pair` on the pair afterward. The re-snap
moves one object of the pair (the pair's first object, or the other one when
the first is a **Static** collider; the tool description says which) and
leaves a small gap between the two, so check placement and clearance
afterward. See [Snap and Merge](../workflow/constraints/snap_merge.md).

Use the **Simple** method. Simple subdivision leaves the outline unchanged,
while the default Catmull-Clark method rounds corners, which changes the
silhouette of engineered meshes (plates, boxes, cut-outs).

Example via `run_python_script`:

```python
import bpy
obj = bpy.data.objects["Cloth"]
mod = obj.modifiers.new(name="Subsurf", type="SUBSURF")
mod.subdivision_type = "SIMPLE"   # NOT the default "CATMULL_CLARK"
mod.levels = 2                    # viewport level
mod.render_levels = 2             # same value as the viewport level
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier=mod.name)
```

Re-check the edge-length ratio after applying. If one round of subdivision
still leaves you above 3 %, apply another.

## When to Delete and Recreate

Each Simple level multiplies the face count of a quad mesh by 4, so cost grows
quickly with the level. When a primitive is procedural (ico sphere, cylinder,
torus, grid), it is usually better to **delete and recreate** it at the target
resolution than to keep subdividing.

The recreate loop:

1. Record the object's name, its transform, the groups it is in, and
   everything that names it: its pin vertex groups and their pin operations,
   its entries in the groups' intersection-allowance object lists, and its
   merge pairs. `remove_object_from_group` removes the pins,
   intersection-allowance entries and merge pairs that name the object.
   `remove_all_objects_from_group` removes the pins and allowance entries but
   leaves the merge pairs, so call `list_merge_pairs` and `remove_merge_pair`
   afterward. Neither tool discards the object's fetched animation, which the
   panel's **Remove Object** button does; `clear_local_animation` clears the
   fetched animation of every object in every group.
2. Delete the object and confirm it is gone (for example, by listing scene
   objects) before creating the replacement. While the name is still taken,
   Blender gives the new object a numbered name such as `Cloth.001`.
3. Create the new primitive with a `subdivisions` or `resolution` parameter
   high enough that the 1–3 % check passes on the first try.
4. Re-apply the recorded transform and re-add the object to its groups with
   `add_objects_to_group`. Then re-create what step 1 recorded: the vertex
   groups (`create_vertex_group`) and pins (`add_pin_vertex_group`,
   `add_pin_operation`), the allowance object lists
   (`set_intersection_allowance_objects`) and the merge pairs
   (`add_merge_pair`, then `resnap_merge_pair`, because a pair added without a
   snap has no stitch points and is refused; `list_merge_pairs` reports each
   pair's `stitch_row_count`). See
   [Merge Pairs Without Snapping](../workflow/constraints/snap_merge.md#merge-pairs-without-snapping).

## Material Parameters: Don't Invent Defaults

1. Call `get_group_material_properties(group_uuid)`. It reports every
   parameter the group's type accepts, with its current value, the add-on
   default, its description and its limits: minimum and maximum for a number,
   accepted identifiers for an enum. The parameters are listed per type under
   [Material Parameters](../workflow/params/material.md).
2. Set only what the user has explicitly requested.
   `set_group_material_properties` refuses a number outside its limits and an
   enum identifier the picker does not offer, and writes nothing from that
   call. Do not pick values for density, Young's modulus, Poisson ratio,
   friction or contact gap on the user's behalf; leave the add-on defaults.
3. When the user requests something material-specific (for example "make this
   cloth stiffer"), prefer a bundled preset to hand-picked numbers.
   `list_material_presets` lists the **Shell** fabrics and the **Solid** soft
   solids, and `apply_material_preset(group_uuid, preset_name)` writes one onto
   a group of the same type. See
   [Fabric Presets](../workflow/params/fabric_presets.md). A material profile
   the user saved to a file loads with `list_profiles` and `load_profile`; see
   [Profiles](#profiles) below and
   [Material Profiles](../workflow/params/material.md#material-profiles).

## Profiles

A profile is a TOML file of named entries, and its panel row loads and saves
it. Four profile kinds are reachable over MCP: `SCENE` (the scene parameters,
the dynamic parameters and the invisible colliders), `MATERIAL` (one group's
material parameters), `PIN` (one pin's operations) and `CONNECTION` (the solver
host settings). See
[Scene Profiles](../workflow/params/scene.md#scene-profiles),
[Material Profiles](../workflow/params/material.md#material-profiles),
[Saving Operations to a Profile](../workflow/constraints/pins.md#saving-operations-to-a-profile)
and [Connection Profiles](../connections/profiles.md).

- `list_profiles(kind, group_uuid, path)` lists the entries of the file the
  kind is bound to, or of `path` when given. A kind with no bound file is
  refused. `MATERIAL` and `PIN` are bound to a group and need `group_uuid`;
  `SCENE` and `CONNECTION` refuse one.
- `load_profile(kind, name, group_uuid, vertex_group_identifier, path)` applies
  an entry and overwrites every setting its kind covers. A padlock does not
  protect a row from a profile. A `PIN` call also needs
  `vertex_group_identifier`, in the form `object_name::vertex_group_name`.
- `save_profile(kind, name, group_uuid, vertex_group_identifier, path)` writes
  the entry `name`, replacing an entry of that name and keeping every other
  entry. Passing `path` writes that file and binds the kind to it. The name
  `NONE` is refused.
- `clear_profile_path(kind, group_uuid)` unbinds the file, as the X icon beside
  the profile row's dropdown does (**Clear Scene Profile**, **Clear Material
  Profile**, **Clear Pin Profile** or **Clear Profile**, by kind). It does not
  touch the file or the settings already applied. No tool removes an entry from
  a file.

`path` is absolute, or starts with `//` to be relative to the saved `.blend`
file.

## Strain Limit for Stiff Cloth

If the user asks for a stiff, low-stretch cloth, set `enable_strain_limit` to
`true` on the group. A **Shell** or **Rod** group has the setting and a
**Solid** group does not. On a **Shell** group it has no effect while `shrink_x`
or `shrink_y` is not `1`. `strain_limit_percent` defaults to `5.0` (a 5 %
stretch limit). The range is 0 to 100 and `0` applies no limit, so a stiffer
result is a smaller positive value, never `0`. Lower it only when the user asks
for a stiffer result. Leave the other defaults as they are unless the user
calls them out. See
[Strain Limit](../workflow/params/material.md#strain-limit) for the semantics.

## Contact Gap: Pick One Mode

The absolute and ratio contact-gap modes are mutually exclusive, and
`use_group_bounding_box_diagonal` switches between them. A single
`set_group_material_properties` call that sends any absolute field
(`contact_gap`, `contact_offset`) together with any ratio field
(`contact_gap_rat`, `contact_offset_rat`) is refused. You do not have to set
the flag yourself: sending `contact_gap` or `contact_offset` sets it to
`false`, and sending `contact_gap_rat` or `contact_offset_rat` sets it to
`true`, in the same call.

`contact_gap_limit_percent`, on **Solid**, **Shell** and **Rod** groups,
belongs to neither mode and can be sent with either pair. It caps each
element's gap at that percentage (1 to 50, 25 by default) of the room its mesh
neighbors leave, so the gap an element is solved at can be smaller than the one
the group reports. Leave it alone unless the user asks; see
[Local Gap Limit](../workflow/params/material.md#local-gap-limit).

## Group-Type Cheat Sheet

When assigning a type with `set_group_type`, or with the `type` argument of
`create_group`, use these rules:

| Object                                                 | Type       |
| ------------------------------------------------------ | ---------- |
| Static rigid collider (floor, mannequin, wall, table)  | `STATIC`   |
| Thin flexible surface (cloth, sheet, banner, shell)    | `SHELL`    |
| Volumetric flexible solid (rubber ball, sponge, foam)  | `SOLID`    |
| 1-D flexible line (rope, cable, hair strand)           | `ROD`      |
| Exactly-rigid moving body (no tetrahedralization)      | `PDRD`     |
| Granular body (loose grain-center vertices, no faces)  | `SAND`     |

A regular mesh must be converted before a **Sand** group takes it. Call
`convert_to_particle_mesh(object_name, grain_radius)`, which discards the faces
and replaces the mesh with grain centers. The grain radius is locked after the
conversion, so choose it first. See
[Sand-Specific](../workflow/params/material.md#sand-specific).

Before calling `transfer_data`, list the groups with `get_active_groups` and
check that each one has at least one object and that the groups together hold
the objects the user intends to simulate.

## Connect, Transfer, Solve, Fetch

Each tool that starts work returns at once; see
[Tool Calls and Long-Running Tools](mcp.md#tool-calls-and-long-running-tools).
A tool whose preconditions are not met refuses with an error instead of
starting. The error names the refused action. The tools that run a panel button
(`transfer_data`, `run_simulation`, `resume_simulation`,
`resume_simulation_from`, `update_params`, `delete_remote_data`,
`fetch_animation` and `clear_local_animation`) add the current status and the
last error, as `(status=..., error=...)`. `start_remote_server` and
`stop_remote_server` do not, so read `get_remote_status` or
`get_connection_info` for the cause.

1. Connect with `connect`, which uses the panel's current connection settings,
   or with `connect_ssh`, `connect_docker`, `connect_linux_native`,
   `connect_win_native` or `connect_mac_native`. Poll `get_connection_info`
   until `connection_status.connected` is `true`.

   The connect tools do not set **Compute Device**. It keeps the value the
   panel holds, **GPU** by default. `connect_linux_native` and
   `connect_win_native` take `gpu_backend` (`AUTO`, `CUDA` or `ROCM`; an empty
   value keeps the current one), and no other connect tool sets **GPU
   Backend**. To run the CPU build, set **Compute Device** to **CPU** in the
   panel, or from a script with `solver.param.native_device = "CPU"`, for
   example through `run_python_script`. A native connection checks the choice
   when it connects, so set it first; a remote connection reads it at
   `start_remote_server`. See {ref}`Choosing the build <choosing-the-build>`.
2. If `is_remote_server_running` returns `server_running: false`, call
   `start_remote_server`, and poll `is_remote_server_running` until
   `server_running` is `true`.
3. Call `transfer_data`, and poll `get_remote_status` until `current_status`
   reads `Ready to Run`. If it does not get there, read `error`.
4. Call `run_simulation`, and poll `get_remote_status` for its status and
   progress.
5. Call `fetch_animation`, and poll `get_fetch_status` for the frames fetched
   into Blender.

## See Also

- [MCP Server](mcp.md): protocol, transport, and security.
- [MCP Tool Reference](mcp_reference.rst): full handler list and
  schemas.
- [Object Groups](../workflow/scene/object_groups.md): what each group type
  does in the solver.
- [Material Parameters](../workflow/params/material.md): the fields
  `set_group_material_properties` accepts and their defaults.
- [Fabric Presets](../workflow/params/fabric_presets.md): the bundled
  **Shell** and **Solid** presets.
