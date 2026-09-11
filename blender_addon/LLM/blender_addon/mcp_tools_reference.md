# MCP tool reference

This document tracks the bundled MCP tool surface shipped with the add-on. Every tool listed here is callable over the MCP Streamable HTTP server (`POST /mcp` with `tools/call`, as a standalone request carrying `params._meta` and the mirrored `MCP-Protocol-Version`, `Mcp-Method` and `Mcp-Name` headers) and, equivalently, via `bpy.ops.zozo_contact_solver.<tool_name>()` inside Blender. See `llm://integrations` for protocol, transport, and security notes.

If you reached this file as MCP resource `llm://mcp_tools_reference`, its sibling resources (`llm://index`, `llm://overview`, `llm://parameters`, and so on) cover the surrounding concepts. Call `resources/list` once and pick the matching URI; the full resource surface (URI scheme, list/read examples, error handling) is documented under the **Resources** section of `llm://integrations`.

## How to invoke

Equivalent ways to call a tool named `<tool>` with JSON `arguments`:

1. HTTP (JSON-RPC `tools/call` over MCP Streamable HTTP). There is no handshake and no session: send `Accept: application/json, text/event-stream`, the `MCP-Protocol-Version`, `Mcp-Method` and `Mcp-Name` headers, and the protocol fields in `params._meta`:

```text
curl -s -X POST http://127.0.0.1:9633/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/call' \
  -H 'Mcp-Name: <tool>' \
  -d '{"jsonrpc":"2.0","method":"tools/call","id":1,
       "params":{"name":"<tool>","arguments":{...},
                 "_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28",
                          "io.modelcontextprotocol/clientCapabilities":{}}}}'
```

`Mcp-Name` must repeat `params.name` exactly, or the server answers `-32020`.

See `llm://integrations` for the transport in full: the required headers, the
error codes, and how a client written against `2025-06-18` is still served.

2. Python (inside Blender):

```python
bpy.ops.zozo_contact_solver.<tool>(<kwargs>)
```

3. CLI helper:

```text
python blender_addon/debug/main.py call <tool> '{"arg":"value"}'
```

## Categories

- Connection (12)
- Group (20)
- Object operations (31)
- Mesh cleaning (8)
- Simulation (13)
- Scene (13)
- Dynamic parameters (9)
- Remote (8)
- Console (3)
- Debug (8)
- Blender (10)

## Connection

### connect_ssh(host: str, username: str, key_path: str, remote_path: str, port: int=22, container: Optional[str]=None)

Establish SSH connection to remote server for contact solver.

**Parameters:**

- **host**: SSH hostname or IP address
- **username**: SSH username
- **key_path**: Path to SSH private key
- **remote_path**: Remote working directory path
- **port**: SSH port
- **container**: Docker container name (optional)

### connect_docker(container: str, path: str)

Establish Docker connection for contact solver.

**Parameters:**

- **container**: Docker container name
- **path**: Working directory path in container

### connect_local(path: str)

Establish local connection for contact solver.

**Parameters:**

- **path**: Local working directory path

### connect_win_native(path: str, port: int=DEFAULT_SERVER_PORT)

Establish Windows native connection for contact solver.

**Parameters:**

- **path**: Path to the Windows native build or distribution directory
- **port**: Port for the solver server

### disconnect()

Disconnect from the solver host, or cancel a connection still in progress.

### connect()

Connect using current connection settings, mimicking the connect button press.

### start_remote_server()

Start the remote server process.

### stop_remote_server()

Stop the remote server process.

### is_remote_server_running()

Check if remote server is running.

### get_remote_status()

Get detailed remote server status.

### update_remote_status()

Update remote server status.

### get_connection_info()

Get detailed connection information.


### list_solver_gpus()

List the GPUs on the solver host, and which one is selected.

The list is a cache filled by `refresh_solver_gpus`, which reads it from
the solver host over the active connection. Before the first refresh the
list is empty, which reports as `probed: false` rather than as a host with
no GPUs.

A selection is stored as both an index and a stable UUID, and the UUID
wins: a .blend saved against one host and opened against another must not
silently resolve to a different physical device.

### refresh_solver_gpus()

Re-read the GPU list from the solver host.

Requires an active connection: the list is produced by a command run on
the host, so there is nowhere to read it from otherwise. The refreshed
list is available from `list_solver_gpus`.

### set_solver_gpu(uuid: str=None, index: int=None)

Choose which GPU on the solver host runs the simulation.

Pass `uuid` to name a device stably, which is what the add-on stores and
prefers. `index` alone selects by CUDA index and is only reliable while
the host's device set does not change. Passing neither clears the
selection back to automatic.

The selection is validated against the cached device list when one has
been probed; with no list there is no evidence to contradict the request,
so it is honored as given.

**Parameters:**

- **uuid**: Stable device UUID, from list_solver_gpus
- **index**: CUDA device index, used when no uuid is given
## Group

### create_group(name: str='', type: str='SOLID')

Create a new dynamics group.

**Parameters:**

- **name**: Display name for the new group (optional)
- **type**: Group type (SOLID, SHELL, ROD, STATIC, PDRD, SAND). PDRD is an exactly-rigid body type whose surface mesh moves as a single best-fit rigid transform (no tetrahedralization, no Young's/Poisson/bend/shrink/strain/inflate). SAND is a granular body of loose grain-center vertices; feed it with `convert_to_particle_mesh`, which turns a closed mesh into the grain cloud it simulates.

### delete_group(group_uuid: str)

Delete a specific group by UUID.

**Parameters:**

- **group_uuid**: UUID of group to delete

### delete_all_groups()

Delete all active groups.

### duplicate_group(group_uuid: str)

Duplicate a dynamics group (material params only, no objects or pins).

**Parameters:**

- **group_uuid**: UUID of the source group to duplicate

### rename_group(group_uuid: str, name: str)

Rename a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group to rename
- **name**: New display name (empty string falls back to 'Group N')

### bake_group_animation(group_uuid: str, object_name: str)

Bake simulated animation for one object in a group to Blender keyframes.

The object is removed from the group and keeps its baked animation.

**Parameters:**

- **group_uuid**: UUID of group containing the object
- **object_name**: Name of the object to bake

### bake_group_single_frame(group_uuid: str, object_name: str)

Bake the current frame as frame 1 for one object and drop it from the group.

**Parameters:**

- **group_uuid**: UUID of group containing the object
- **object_name**: Name of the object to bake

### set_object_included(group_uuid: str, object_name: str, included: bool)

Toggle whether an assigned object is included in the simulation.

**Parameters:**

- **group_uuid**: UUID of the group
- **object_name**: Name of the assigned object
- **included**: True to include, False to mute

### get_group(group_uuid: str)

Get one active group by UUID.

**Parameters:**

- **group_uuid**: UUID of group

### get_active_groups()

Get list of all active groups with their properties.

### add_objects_to_group(group_uuid: str, object_names: list[str])

Add objects to a dynamics group.

**Parameters:**

- **group_uuid**: UUID of target group
- **object_names**: List of object names to add

### remove_object_from_group(group_uuid: str, object_name: str)

Remove an object from a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of object to remove

### remove_all_objects_from_group(group_uuid: str)

Remove all objects from a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group to clear

### get_group_objects(group_uuid: str)

Get objects assigned to a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group

### set_group_type(group_uuid: str, type: str)

Set the type of a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group
- **type**: Group type (SOLID, SHELL, ROD, STATIC, PDRD, SAND). Retyping to SAND does not convert the assigned geometry: a SAND group simulates loose grain-center vertices, so run `convert_to_particle_mesh` on each assigned mesh as well.

### add_pin_vertex_group(group_uuid: str, vertex_group_identifier: str, indices: Optional[list[int]]=None)

Add a mesh vertex group or curve control-point pin set to the pin list of a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Identifier in format "object_name::vertex_group_name"
- **indices**: Optional curve control-point indices for CURVE objects

### remove_pin_vertex_group(group_uuid: str, vertex_group_identifier: str)

Remove a vertex group from the pin list of a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Identifier in format "object_name::vertex_group_name"

### list_pins(group_uuid: str)

List all pins in a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group

### set_group_overlay_color(group_uuid: str, r: float, g: float, b: float, a: float=1.0)

Set the viewport overlay color for a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group
- **r**: Red channel in [0, 1]
- **g**: Green channel in [0, 1]
- **b**: Blue channel in [0, 1]
- **a**: Alpha channel in [0, 1]

### set_group_material_properties(group_uuid: str, properties: dict)

Set material properties for a dynamics group.

**Parameters:**

- **group_uuid**: UUID of target group
- **properties**: Dict of property_name -> value mappings

Supported properties by group type:

- SHELL: enable_strain_limit, strain_limit_percent, shell_density, shell_young_modulus, shell_poisson_ratio, shell_model, bend, bend_warp, bend_weft, shrink_x, shrink_y, deformation_damping, bending_damping, young_mod_density_normalized, friction, enable_inflate, inflate_pressure, stitch_stiffness, bend_rest_angle_source, bend_rest_from_reference, allow_self_intersection, allow_inter_object_intersection
- SOLID: solid_density, solid_young_modulus, solid_poisson_ratio, solid_model, shrink, deformation_damping, young_mod_density_normalized, friction, stitch_stiffness, allow_self_intersection, allow_inter_object_intersection
- ROD: rod_density, rod_young_modulus, rod_model, deformation_damping, bending_damping, young_mod_density_normalized, friction, bend, length_factor, enable_strain_limit, strain_limit_percent, stitch_stiffness, bend_rest_angle_source, bend_rest_from_reference, allow_self_intersection, allow_inter_object_intersection
- PDRD: pdrd_density, friction, stitch_stiffness, allow_self_intersection, allow_inter_object_intersection (the hinge joint is per-object; use the `set_pdrd_hinge` tool)
- SAND: sand_grain_radius, sand_particle_mass, sand_friction, allow_self_intersection, allow_inter_object_intersection (faceless granular body of loose grain-center vertices)
- STATIC: friction, enable_soft_constraint, soft_constraint_stiffness, allow_self_intersection, allow_inter_object_intersection (a collider tracks its animation exactly unless soft constraints are on, which holds it with springs of that stiffness so contact can push it off its path)

Per-type property notes:

- pdrd_density: PDRD body volume density (kg/m^3), default 100. Mass is density times the enclosed volume of the surface mesh.
- PDRD hinge: the hinge joint is a per-object setting, not a group material. Set it with the `set_pdrd_hinge` tool (group_uuid, object_name, enable, pca_axis), so each body in a PDRD group can be hinged on its own axle.
- bend_rest_from_reference (SHELL, ROD): group-level master toggle for per-object reference rest angles. Settable via this tool, but the per-object reference itself (which object opts in, and which object is its reference) is not exposed over MCP: it is picked in the add-on UI (the eyedropper that runs `object.pick_bend_reference`). When enabled with a valid reference, that object's bending rest angle (shell hinge dihedral, or rod interior-vertex bend angle) is computed from the reference geometry, overriding `bend_rest_angle_source` for that object. Mesh references are modifier-evaluated (vertex count + connectivity must match); curve-rod references are sampled at the control-point level.
- deformation_damping: stiffness-proportional Rayleigh damping (seconds) for stretch/membrane/solid deformation; default 0.0, min 0.0. Applies to SOLID, SHELL, ROD. 0 disables it. PDRD groups are not Rayleigh-damped.
- bending_damping: stiffness-proportional Rayleigh damping (seconds) for bending; default 0.0, min 0.0. SHELL and ROD only (SOLID/tet has no bending term; rejected for SOLID and PDRD). 0 disables it.
- length_factor (ROD, UI label Shrink): multiplies every rod edge's rest length, so below 1.0 it tensions a pinned rod and above 1.0 it slackens it. Mass is taken from the drawn length and does not move with it. Rod bending stiffness is normalized against that same rest length and varies as its inverse square, so halving length_factor also makes the rod about four times stiffer in bending.
- sand_grain_radius (SAND): the group-level fallback only. `convert_to_particle_mesh` stamps the radius it seeded with onto the object as `ppf_grain_radius`, and both the panel and the encoder prefer that stamped value over this one, so setting it here changes nothing once an included object has been converted. The panel draws the radius read-only either way. To change the radius, convert again from an unconverted copy of the source mesh.
- sand_particle_mass (SAND): mass of one grain in GRAMS, default 1.0, min 1e-6. The add-on multiplies by 1e-3 and ships kilograms to the solver, so a value chosen as if it were SI is off by a thousand.
- SAND contact keys: the locked grain radius is sent as the group's contact OFFSET, because a grain's skin is its radius. `contact_gap` is the extra barrier distance on top of that skin and is always the absolute field: `contact_offset`, `contact_gap_rat`, `contact_offset_rat` and `use_group_bounding_box_diagonal` are accepted by the validator but the encoder ignores them for SAND.
- young_mod_density_normalized: SOLID/SHELL/ROD only. True (default) interprets the Young's modulus field as a density-normalized value (Pa/rho), the solver's native convention. False interprets it as a true Young's modulus in pascals, which the addon divides by this group's density before sending it.
- stitch_stiffness: per-object soft cross-stitch force stiffness, default 1.0. Cross-stitch is a soft 6-slot barycentric force, not a topological weld: the two sides keep their own vertices and are pulled together by a spring. Supported pairs: Shell-Shell, Shell-Solid, Rod-Shell, Rod-Solid, Rod-Rod, Solid-Solid, and any dynamic group stitched to a STATIC collider.
- allow_self_intersection, allow_inter_object_intersection: accepted on every group type, both default off. They suppress the REPORT of an intersecting pair, so a run starts and keeps going through a tangled pose; contact, CCD and the line search are unchanged. The value is applied to every object assigned to the group, and self versus inter-object is decided per Blender object, not per group: an overlap between two objects of the same group is an inter-object pair. For the inter-object key either side is enough, so setting it on a garment also covers the body it is fitted to. On a STATIC group both keys reach the solver whenever the collider is part of the solved scene, which covers an animated collider, a soft-constrained one, and one named as a cross-stitch endpoint, since each decodes to a pin shell whose vertices carry the policy; a collider that is none of them stays a contact-only collision mesh carrying no object id and an empty policy, so a pair involving it is tolerated only when the opposing dynamic side opts in.

Contact properties (mutually exclusive modes):

- Absolute mode: contact_gap, contact_offset (sets use_group_bounding_box_diagonal=False)
- Relative mode: contact_gap_rat, contact_offset_rat (sets use_group_bounding_box_diagonal=True)

**Returns:** Dict with success message and properties set


### create_vertex_group(object_name: str, name: str, indices: list[int], weight: float=1.0)

Create a vertex group on a mesh and assign the given vertices to it.

This is the membership a mesh pin names: create the group here, then pass
"object_name::name" to add_pin_vertex_group to pin it. Call
list_vertex_groups first to see which names the object already carries.

The object does not have to be the active one and Blender can be in any
mode. An object in Edit Mode is taken to Object Mode for the write and put
back, which also writes the edit session to the mesh, so the indices below
address the geometry the caller can see.

Fails before creating anything, leaving Blender as it was found, when the
object is not a MESH, when it is library-linked, when indices is empty,
when any index is outside the mesh, or when the object already carries a
group of that name. An existing group can be driven by an armature or a
modifier, so it is never overwritten.

**Parameters:**

- **object_name**: Name of the mesh object to create the vertex group on.
- **name**: Name for the new vertex group; must not already exist on the object.
- **indices**: Vertex indices to assign, each in 0 to vertex_count - 1. Repeated indices are assigned once.
- **weight**: Weight for every assigned vertex, in [0, 1]. Defaults to 1.0, which is what the panel's Create button assigns.

### list_vertex_groups(object_name: str)

List a mesh's vertex groups and how many vertices each one holds.

A mesh pin names a vertex group that already exists on the object, so this
reports the names add_pin_vertex_group accepts and create_vertex_group
will refuse as duplicates. ``vertex_count`` counts the vertices assigned
to the group at any weight; a group holding zero vertices pins nothing.

Only a MESH carries vertex groups. A curve's pinned control points live on
the curve object and are reported by list_pins once they are pinned.

Refuses an object that is in Edit Mode. That session holds the geometry
and the weights in a BMesh the mesh datablock does not receive until the
mode is left, so the counts would describe the mesh as it stood before the
session. Leave Edit Mode and call again.

**Parameters:**

- **object_name**: Name of the mesh object to inspect.

### get_group_material_properties(group_uuid: str)

Report every material parameter a group accepts, with its value.

Which parameters a group carries is decided by its object_type, and the
set reported here is exactly the set set_group_material_properties
accepts for this group, so a name absent from this report is refused by
that tool. A parameter another type carries is reachable only by
retyping the group with set_group_type first.

Each entry carries the current value, the add-on default, the
description the panel shows for it, and the limits the property enforces:
min and max for a number, the accepted identifiers for an enum. Blender
clamps a number written outside its own min and max, so read the limits
before setting one.

The values are the authored ones, not what the solver derives from them.
Contact distances in particular are stored as an absolute pair
(contact_gap, contact_offset) and a relative pair (contact_gap_rat,
contact_offset_rat), and both pairs are reported whichever one
use_group_bounding_box_diagonal currently selects.

Per-object state (inclusion, locks, hinge, bending reference,
tetrahedralizer) is reported by get_group_objects, and a parameter driven
across the surface by a weight map is reported by list_material_maps.

**Parameters:**

- **group_uuid**: UUID of the group to report.

### move_pin_vertex_group(group_uuid: str, vertex_group_identifier: str, direction: str)

Move a pin one place up or down its group's pin list.

Pin order decides what two pins of one group do where they hold the same
vertex: the scene build writes each pin's settings in list order, so for a
shared vertex the pin lower in the list is the one whose duration, pull
and operations that vertex takes. Order says nothing about pins that share
no vertex.

One place per call. list_pins reports the pins in list order, so the
position of a pin in that array is the position this moves it from. A pin
already at the top cannot move up and one already at the bottom cannot
move down; either is refused rather than reported as a move that did
nothing.

**Parameters:**

- **group_uuid**: UUID of the group holding the pin.
- **vertex_group_identifier**: The pin to move, in the format "object_name::vertex_group_name".
- **direction**: "UP" to move it one place toward the start of the list, "DOWN" to move it one place toward the end.

### rename_pin_vertex_group(group_uuid: str, vertex_group_identifier: str, new_name: str)

Rename the vertex group a pin names, on the object and in the pin list.

This renames the membership as well as the pin entry: on a mesh the
object's vertex group is renamed, and on a curve the "_pin_<name>"
property holding the pinned control points is. Anything else that names
that vertex group, an armature or a modifier for instance, refers to it by
name and stops finding it, so rename a group only the solver pin uses.

The object keeps its name; only the vertex group half of the identifier
changes. Refused before anything is renamed when the new name is empty,
when it is the name the pin already has, or when the object already
carries a vertex group (on a curve, a pin property) under that name, since
Blender would then store a name other than the one asked for.

**Parameters:**

- **group_uuid**: UUID of the group holding the pin.
- **vertex_group_identifier**: The pin to rename, in the format "object_name::vertex_group_name".
- **new_name**: New name for the vertex group half of the identifier.
## Object operations

### set_pin_settings(group_uuid: str, vertex_group_identifier: str, included: Optional[bool]=None, use_pin_duration: Optional[bool]=None, pin_duration: Optional[int]=None, use_pull: Optional[bool]=None, pull_strength: Optional[float]=None)

Set per-pin runtime settings (include/duration/pull).

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form
- **included**: Include this pin in the simulation
- **use_pin_duration**: Enable per-pin active duration
- **pin_duration**: Number of frames the pin is active
- **use_pull**: Use pull force instead of hard constraint
- **pull_strength**: Pull force strength

### add_pin_operation(group_uuid: str, vertex_group_identifier: str, op_type: str, frame_start: Optional[int]=None, frame_end: Optional[int]=None, transition: Optional[str]=None, delta: Optional[list[float]]=None, spin_axis: Optional[list[float]]=None, spin_angular_velocity: Optional[float]=None, spin_flip: Optional[bool]=None, spin_center: Optional[list[float]]=None, spin_center_mode: Optional[str]=None, spin_center_vertex: Optional[int]=None, spin_center_direction: Optional[list[float]]=None, scale_factor: Optional[float]=None, scale_center: Optional[list[float]]=None, scale_center_mode: Optional[str]=None, scale_center_vertex: Optional[int]=None, scale_center_direction: Optional[list[float]]=None, torque_axis_component: Optional[str]=None, torque_magnitude: Optional[float]=None, torque_flip: Optional[bool]=None)

Append an operation to a pin's operation list.

TORQUE cannot coexist with other op types on the same pin.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form
- **op_type**: One of MOVE_BY, SPIN, SCALE, TORQUE
- **frame_start**: First frame the op is active
- **frame_end**: Last frame the op is active
- **transition**: LINEAR or SMOOTH
- **delta**: [x, y, z] translation for MOVE_BY (meters)
- **spin_axis**: [x, y, z] rotation axis for SPIN
- **spin_angular_velocity**: Degrees per second (SPIN)
- **spin_flip**: Reverse spin direction
- **spin_center**: [x, y, z] fixed center for SPIN (ABSOLUTE mode only)
- **spin_center_mode**: CENTROID, ABSOLUTE, MAX_TOWARDS, or VERTEX
- **spin_center_vertex**: Vertex index for SPIN VERTEX mode
- **spin_center_direction**: [x, y, z] direction vector for SPIN MAX_TOWARDS mode
- **scale_factor**: Scale multiplier for SCALE
- **scale_center**: [x, y, z] fixed center for SCALE (ABSOLUTE mode only)
- **scale_center_mode**: CENTROID, ABSOLUTE, MAX_TOWARDS, or VERTEX
- **scale_center_vertex**: Vertex index for SCALE VERTEX mode
- **scale_center_direction**: [x, y, z] direction vector for SCALE MAX_TOWARDS mode
- **torque_axis_component**: PC1, PC2, or PC3 (principal axis)
- **torque_magnitude**: Torque in newton-meters
- **torque_flip**: Reverse torque direction

### remove_pin_operation(group_uuid: str, vertex_group_identifier: str, index: int)

Remove a pin operation by index.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form
- **index**: Zero-based index into the pin's operations list

### list_pin_operations(group_uuid: str, vertex_group_identifier: str)

List operations attached to a pin.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form

### clear_pin_operations(group_uuid: str, vertex_group_identifier: str)

Remove every non-embedded operation from a pin.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form

### add_static_op(group_uuid: str, object_name: str, op_type: str, frame_start: Optional[int]=None, frame_end: Optional[int]=None, transition: Optional[str]=None, delta: Optional[list[float]]=None, spin_axis: Optional[list[float]]=None, spin_angular_velocity: Optional[float]=None, scale_factor: Optional[float]=None)

Add a move/spin/scale op to a static-moving object.

Only valid on groups of type STATIC.

**Parameters:**

- **group_uuid**: UUID of STATIC group
- **object_name**: Name of the assigned object
- **op_type**: One of MOVE_BY, SPIN, SCALE
- **frame_start**: First frame the op is active
- **frame_end**: Last frame the op is active
- **transition**: LINEAR or SMOOTH
- **delta**: [x, y, z] translation (MOVE_BY)
- **spin_axis**: [x, y, z] rotation axis (SPIN)
- **spin_angular_velocity**: Degrees per second (SPIN)
- **scale_factor**: Scale multiplier (SCALE)

### remove_static_op(group_uuid: str, object_name: str, index: int)

Remove a static op by index.

**Parameters:**

- **group_uuid**: UUID of STATIC group
- **object_name**: Name of the assigned object
- **index**: Zero-based index into the object's static_ops list

### list_static_ops(group_uuid: str, object_name: str)

List static ops attached to an assigned object.

**Parameters:**

- **group_uuid**: UUID of STATIC group
- **object_name**: Name of the assigned object

### clear_static_ops(group_uuid: str, object_name: str)

Remove all static ops from an assigned object.

**Parameters:**

- **group_uuid**: UUID of STATIC group
- **object_name**: Name of the assigned object

### capture_static_deformation(group_uuid: str, object_name: str)

Record the per-frame shape of an animated STATIC mesh onto the collider. Use this for STATIC objects whose vertices move because of an Armature modifier, a Lattice or Mesh Deform cage, animated Shape Keys, or a driver that pokes vertex coordinates. The recording runs as a modal operator and continues after this call returns; poll `get_static_deformation_status` to detect completion. Press again any time the underlying animation changes (a new pose, edited action keyframes, a modifier swap). The recording does NOT update on its own.

**Parameters:**

- **group_uuid**: UUID of STATIC group containing the object
- **object_name**: Name of the assigned mesh to capture

### clear_static_deformation(group_uuid: str, object_name: str)

Discard the recorded deformation cache for one STATIC object. The object returns to the pre-capture state: Capture Deformation becomes the only enabled button on the row, and the next Transfer will refuse to upload the object until a fresh capture is taken.

**Parameters:**

- **group_uuid**: UUID of STATIC group containing the object
- **object_name**: Name of the assigned mesh

### get_static_deformation_status(group_uuid: str, object_name: str)

Report the deformation-capture state of one STATIC object. Returns three fields: `is_deforming` (True if the object's modifier stack or shape-key animation actually moves vertices over the timeline; when False, Capture Deformation is not needed and the button is grayed out), `has_cache` (True if a deformation cache exists), and `frame_count` (number of frames in the cache, or 0 when absent).

**Parameters:**

- **group_uuid**: UUID of STATIC group containing the object
- **object_name**: Name of the assigned mesh

### detect_isolated_static_vertices()

Report stray faceless vertices on active STATIC colliders that block Transfer. Scans every included, active STATIC collider mesh for vertices that belong to no triangle (no face). The solver build aborts on these, and Transfer reports a ValueError whose message contains "isolated vert", naming the object and the vertex indices. Read-only; pair with `remove_isolated_static_vertices` to delete them.

### remove_isolated_static_vertices()

Delete stray faceless vertices from active STATIC colliders so the scene transfers. Removes only vertices that belong to no triangle (with their loose edges); faces are untouched. Mirrors the **Remove Isolated Vertices** panel button and scans every included, active STATIC collider. Run `detect_isolated_static_vertices` first to preview what will be deleted.

### capture_pin_deformation(group_uuid: str, vertex_group_identifier: str)

Record the per-frame shape of a deformable pin onto the cloth mesh. Use this for pins whose vertices ride along with an Armature, Lattice, Mesh Deform cage, animated Shape Keys, or a driver. The recording runs as a modal operator and continues after this call returns; poll `get_pin_deformation_status` until `frame_count` is non-zero. Press again any time the underlying animation changes. The recording does NOT update on its own. Refuses to start if the pin already carries manual Make-Keyframe vertex-co fcurves; clear those first.

**Parameters:**

- **group_uuid**: UUID of the SHELL/SOLID/ROD group containing the pin
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form

### clear_pin_deformation(group_uuid: str, vertex_group_identifier: str)

Discard the captured deformation cache for one pin. The pin returns to whatever motion source it had before (none, or manual Make-Keyframe fcurves if any). If no manual fcurves exist the EMBEDDED_MOVE sentinel is also removed so the pin no longer appears animated.

**Parameters:**

- **group_uuid**: UUID of the group containing the pin
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form

### get_pin_deformation_status(group_uuid: str, vertex_group_identifier: str)

Report the captured-deformation state of one pin. Returns four fields: `is_deforming` (True if the pin object's modifier stack will move vertices over the timeline, e.g. Armature or Lattice), `has_cache` (True if a captured-deformation cache exists for the pin, in memory or on disk), `frame_count` (number of frames in the cache, or 0 when absent), and `has_captured_anim_flag` (the pin item's `has_captured_anim` bool; should match `has_cache` after the load_post reconciler runs).

**Parameters:**

- **group_uuid**: UUID of the group containing the pin
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form

### add_velocity_keyframe(group_uuid: str, object_name: str, frame: int, direction: list[float], speed: float, angular_axis: int | str = "PC3", angular_speed: float = 0.0, angular_axis_custom: list[float] | None = None, enable_translational: bool = True, enable_angular: bool | None = None)

Add a velocity keyframe at the given frame for an assigned object.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object
- **frame**: Blender frame number (>= 1)
- **direction**: [x, y, z] direction vector (normalized at runtime)
- **speed**: Velocity magnitude (m/s)
- **angular_axis**: Axis to spin about (solid/shell/PDRD only). `"PC1"`/`"PC2"`/`"PC3"` (principal axes resolved dynamically from the geometry), `"X"`/`"Y"`/`"Z"` (fixed world axes), or `"CUSTOM"` (the `angular_axis_custom` vector). Ints `0`/`1`/`2` map to PC1/PC2/PC3. Ignored when `angular_speed == 0`.
- **angular_speed**: Signed spin speed in degrees per second (0 = no spin).
- **angular_axis_custom**: World `[x, y, z]` axis used when `angular_axis == "CUSTOM"` (normalized before use). Defaults to `[0, 0, 1]`.
- **enable_translational**: Overwrite the translational velocity at this frame (False leaves translation alone, e.g. a pure spin).
- **enable_angular**: Overwrite the angular velocity at this frame. Defaults to True when `angular_speed` is non-zero, else False.

### remove_velocity_keyframe(group_uuid: str, object_name: str, frame: int)

Remove the velocity keyframe at the given frame.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object
- **frame**: Frame number of the keyframe to remove

### list_velocity_keyframes(group_uuid: str, object_name: str)

List velocity keyframes for an assigned object.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object

### clear_velocity_keyframes(group_uuid: str, object_name: str)

Clear all velocity keyframes on an assigned object.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object

### set_pdrd_hinge(group_uuid: str, object_name: str, enable: bool=True, pca_axis: int=2)

Pin a PDRD body as a hinge (per object): lock its position and restrict rotation to one principal (PCA) axis of its rest shape, the building block for gears. The group must be of type PDRD. Per-object, so each body in a group can be hinged on its own axle.

**Parameters:**

- **group_uuid**: UUID of the PDRD group
- **object_name**: Name of the assigned object
- **enable**: Pin the body (True) or release it so it moves freely (False)
- **pca_axis**: Free axle: 0 (largest extent), 1 (middle), 2 (thinnest, the usual axle for a flat gear or disk)

### set_use_collision_windows(group_uuid: str, enable: bool)

Toggle the per-object collision-window feature for a group.

**Parameters:**

- **group_uuid**: UUID of group
- **enable**: True to enable, False to disable

### add_collision_window(group_uuid: str, object_name: str, frame_start: int, frame_end: int)

Add a collision-active window on an assigned object.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object
- **frame_start**: First frame of the window
- **frame_end**: Last frame of the window

### remove_collision_window(group_uuid: str, object_name: str, index: int)

Remove a collision window by index.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object
- **index**: Zero-based index into the object's collision_windows list

### list_collision_windows(group_uuid: str, object_name: str)

List collision windows on an assigned object.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object

### clear_collision_windows(group_uuid: str, object_name: str)

Clear every collision window on an assigned object.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object

### set_object_tet_settings(group_uuid: str, object_name: str, tet_backend: Optional[str]=None, ftetwild_edge_length_fac: Optional[float]=None, ftetwild_epsilon: Optional[float]=None, ftetwild_stop_energy: Optional[float]=None, ftetwild_num_opt_iter: Optional[int]=None, ftetwild_optimize: Optional[bool]=None, ftetwild_simplify: Optional[bool]=None, ftetwild_coarsen: Optional[bool]=None, tetgen_min_ratio: Optional[float]=None, tetgen_max_volume: Optional[float]=None)

Set the per-object tetrahedralizer backend and overrides. SOLID meshes are tetrahedralized at build time and each object picks its backend and overrides independently. Passing any override value also enables that override; otherwise the backend default applies. Ignored for non-SOLID objects.

- **group_uuid**: UUID of the group containing the object
- **object_name**: Name of the assigned object in the group
- **tet_backend**: "FTETWILD" (tolerant remesher, default) or "TETGEN" (preserves the surface exactly, needs a clean closed manifold)
- **ftetwild_edge_length_fac / ftetwild_epsilon / ftetwild_stop_energy / ftetwild_num_opt_iter / ftetwild_optimize / ftetwild_simplify / ftetwild_coarsen**: fTetWild overrides
- **tetgen_min_ratio**: TetGen minimum radius-edge ratio
- **tetgen_max_volume**: TetGen maximum tet volume (0 = uncapped)

### convert_to_particle_mesh(object_name: str, grain_radius: float, extra_spacing: float=0.0, rng_seed: int=0)

Replace a solid mesh with a cloud of grain centers for a SAND group. Destructive: the faces are discarded and the object becomes a faceless mesh of loose vertices, plus a Particle Mesh geometry-nodes modifier that draws each grain at the given radius. The grain count is not chosen, it is whatever fills the volume at the given separation, and it comes back in the result. `grain_radius` is locked after conversion, since the non-overlapping spacing is derived from it, so pick it before converting rather than adjusting it afterward. Refused when the object is missing, is not a MESH, carries no faces, is already a particle mesh, when `grain_radius` is not positive, or when no grain fits inside the mesh.

**Parameters:**

- **object_name**: Solid mesh object with faces, not already a particle mesh
- **grain_radius**: Physical grain radius, which is also the contact skin
- **extra_spacing**: Gap added between grains beyond touching. 0 packs them as densely as non-overlap allows
- **rng_seed**: Seed for the Poisson-disk seeding, for a repeatable cloud

**Returns:** Dict with `grain_count` plus the `grain_radius` / `extra_spacing` / `rng_seed` the cloud was seeded with

### recapture_all_deformations()

Re-capture every deforming STATIC collider and every animated pin. One pass over the whole scene, instead of calling `capture_static_deformation` and `capture_pin_deformation` per object. The statics are captured first and the pins after, since the two share the depsgraph and cannot run at once. The captures run in the background after this returns; poll `get_static_deformation_status` and `get_pin_deformation_status` until they report the frame counts you expect. Refused when there is nothing to re-capture, or when a capture or bake is already running.

### clear_all_deformations()

Delete every captured deformation cache in the scene. Covers all STATIC-collider deform caches and all animated-pin captures across the active groups, plus any cache orphaned by an object that was deleted or taken out of its group. The objects keep their deformers, so `recapture_all_deformations` rebuilds what this removes. Refused when there is no captured cache to clear, or when a capture or bake is already running.


### set_object_locks(group_uuid: str, object_name: str, lock_translation_enable: bool=None, lock_translation_all: bool=None, lock_translation_axis: list[float]=None, lock_rotation_enable: bool=None, lock_rotation_all: bool=None, lock_rotation_axis: list[float]=None, lock_rotation_prohibit_axis: bool=None)

Lock an object's rigid translation, its rigid rotation, or both.

Lock Translation constrains the object's mass-weighted center of mass to a
fixed world-space line through its initial position; Lock Rotation
restricts its mass-weighted best-fit rigid rotation to a fixed world-space
axis. Deformation stays free under either, and the two are independent
booleans on the same object: either, both or neither may be enabled. Both
are exact constraints on the Newton direction rather than penalty springs,
so there is no stiffness to tune.

Per object, and available on the dynamic group types (SOLID, SHELL, ROD,
PDRD, SAND). A STATIC group is refused, since the encoder ships no lock for
one. A lock also reaches the solver only for an object that is included in
its group.

Every argument is optional and an omitted one leaves that field as it is.
The MODE carries the enable bit, not the axis: lock_translation_all and
lock_rotation_all saturate their lock to all three axes and stop the axis
being read, so a zero axis is correct under either. For the per-axis mode
the axis must be non-zero and finite, and a call that would leave an
enabled per-axis lock with a zero axis is refused. That is decided on the
state the call results in, so an axis and its mode can be set together in
one call in either order.

**Parameters:**

- **group_uuid**: UUID of the group containing the object
- **object_name**: Name of the assigned object in the group
- **lock_translation_enable**: Constrain the center of mass (True) or let it move freely (False)
- **lock_translation_all**: Pin the center of mass to its initial point instead of letting it slide along the translation axis
- **lock_translation_axis**: World-space direction [x, y, z] of the line the center of mass may move along. Direction only, normalized by the encoder
- **lock_rotation_enable**: Restrict the best-fit rigid rotation (True) or leave it free (False)
- **lock_rotation_all**: Forbid net rotation about every axis instead of about the rotation axis alone
- **lock_rotation_axis**: World-space rotation axis [x, y, z]. Direction only, normalized by the encoder
- **lock_rotation_prohibit_axis**: False: rotation about the rotation axis is the object's only rotational freedom. True: rotation about that axis is the one thing forbidden, and the perpendicular plane stays free

### add_pin_keyframe(group_uuid: str, vertex_group_identifier: str)

Key the positions of a pin's vertices at the scene's current frame.

The key records the positions the mesh holds right now, at the frame the
scene is on, so move the timeline and pose the mesh before calling; the
frame that was keyed comes back in the result. Call it once per pose to
build the track. The keys are ordinary Blender keyframes on the mesh, set
to LINEAR interpolation to match how the solver reads a sparse pin track,
and the Dope Sheet retimes or deletes them like any other key.

A pin takes its motion from EITHER the parametric operations
(add_pin_operation) OR keyframes, never both. A pin that already carries
Move/Spin/Scale/Torque operations is therefore refused here, the mirror of
add_pin_operation refusing a keyframed pin. A pin holding a captured
deformation is refused too: the capture wins at encode time, so keys
written on top of it would never be read.

**Parameters:**

- **group_uuid**: UUID of the group containing the pin
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form

### delete_pin_keyframes(group_uuid: str, vertex_group_identifier: str)

Remove the keyframed motion of a pin, at every frame it was keyed on.

This deletes the vertex position curves add_pin_keyframe wrote and drops
the marker that records the pin as keyframed, which is what frees the pin
to take parametric operations again. There is no per-frame form: the whole
track goes, so retime or delete single keys in the Dope Sheet instead when
that is what you want.

The curves are addressed by mesh, not by pin, so a second pin on the same
OBJECT loses its keys in the same call. A captured deformation is a
separate motion source and is left untouched; clear_pin_deformation is
what removes that.

**Parameters:**

- **group_uuid**: UUID of the group containing the pin
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form

### set_pin_operation(group_uuid: str, vertex_group_identifier: str, index: int, frame_start: int=None, frame_end: int=None, transition: str=None, delta: list[float]=None, spin_axis: list[float]=None, spin_angular_velocity: float=None, spin_flip: bool=None, spin_center: list[float]=None, spin_center_mode: str=None, spin_center_vertex: int=None, spin_center_direction: list[float]=None, scale_factor: float=None, scale_center: list[float]=None, scale_center_mode: str=None, scale_center_vertex: int=None, scale_center_direction: list[float]=None, torque_axis_component: str=None, torque_magnitude: float=None, torque_flip: bool=None)

Change fields on one operation a pin already carries, addressed by its
index.

Every field argument is optional and an omitted one is left as it is, so
one number can be changed without restating the rest of the entry. Editing
in place is also what preserves the LIST ORDER: the operations are shipped
to the solver in list order and compose in that order, while adding one
puts it at the head, so removing an entry and adding it back to change a
field moves it to the front and changes the motion the pin performs.

The op type is fixed when the entry is added. A field belonging to another
op type is refused rather than written where nothing reads it, so turn a
MOVE_BY into a SPIN by removing it and adding the SPIN in its place. An
entry on a keyframed pin holds no editable field and is refused as well.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form
- **index**: Zero-based index into the pin's operations list, in the order list_pin_operations reports
- **frame_start**: First frame the op is active
- **frame_end**: Last frame the op is active
- **transition**: LINEAR or SMOOTH
- **delta**: [x, y, z] translation for MOVE_BY (metres)
- **spin_axis**: [x, y, z] rotation axis for SPIN
- **spin_angular_velocity**: Degrees per second (SPIN)
- **spin_flip**: Reverse spin direction
- **spin_center**: [x, y, z] fixed center for SPIN (ABSOLUTE mode only)
- **spin_center_mode**: CENTROID, ABSOLUTE, MAX_TOWARDS, or VERTEX
- **spin_center_vertex**: Vertex index for SPIN VERTEX mode
- **spin_center_direction**: [x, y, z] direction vector for SPIN MAX_TOWARDS mode
- **scale_factor**: Scale multiplier for SCALE
- **scale_center**: [x, y, z] fixed center for SCALE (ABSOLUTE mode only)
- **scale_center_mode**: CENTROID, ABSOLUTE, MAX_TOWARDS, or VERTEX
- **scale_center_vertex**: Vertex index for SCALE VERTEX mode
- **scale_center_direction**: [x, y, z] direction vector for SCALE MAX_TOWARDS mode
- **torque_axis_component**: PC1, PC2, or PC3 (principal axis)
- **torque_magnitude**: Torque in newton-metres
- **torque_flip**: Reverse torque direction

### set_static_op(group_uuid: str, object_name: str, index: int, frame_start: int=None, frame_end: int=None, transition: str=None, delta: list[float]=None, spin_axis: list[float]=None, spin_angular_velocity: float=None, scale_factor: float=None)

Change fields on one static op an object already carries, addressed by
its index.

Every field argument is optional and an omitted one is left as it is, so
one number can be changed without restating the rest of the entry. Editing
in place is also what preserves the LIST ORDER: the ops are shipped to the
solver in list order and compose in that order, while adding one puts it
at the head, so removing an entry and adding it back to change a field
moves it to the front and changes the motion of the object.

The op type is fixed when the entry is added, and a field belonging to
another op type is refused rather than written where nothing reads it.

**Parameters:**

- **group_uuid**: UUID of STATIC group
- **object_name**: Name of the assigned object
- **index**: Zero-based index into the object's static_ops list, in the order list_static_ops reports
- **frame_start**: First frame the op is active
- **frame_end**: Last frame the op is active
- **transition**: LINEAR or SMOOTH
- **delta**: [x, y, z] translation (MOVE_BY)
- **spin_axis**: [x, y, z] rotation axis (SPIN)
- **spin_angular_velocity**: Degrees per second (SPIN)
- **scale_factor**: Scale multiplier (SCALE)

### set_velocity_keyframe(group_uuid: str, object_name: str, index: int, frame: int=None, direction: list[float]=None, speed: float=None, angular_axis: int | str=None, angular_speed: float=None, angular_axis_custom: list[float]=None, enable_translational: bool=None, enable_angular: bool=None)

Change fields on one velocity keyframe an object already carries,
addressed by its index.

Every field argument is optional and an omitted one is left as it is, so a
keyframe's speed can be changed without restating its direction and its
two enable gates.

The frame may be changed as well, which retimes the keyframe in place. The
list is held in frame order, so the entry can land at a different index,
and the index it ends up at comes back as new_index. A frame another
keyframe on the same object already occupies is refused, since a frame
carries at most one velocity keyframe.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object
- **index**: Zero-based index into the object's velocity keyframe list, in the frame order list_velocity_keyframes reports
- **frame**: Blender frame number (>= 1) to retime this keyframe to
- **direction**: [x, y, z] direction vector (normalized at runtime)
- **speed**: Velocity magnitude in m/s, zero or greater
- **angular_axis**: Axis to spin about (solid/shell/PDRD). One of "PC1"/"PC2"/"PC3" (principal axes, resolved dynamically from the geometry), "X"/"Y"/"Z" (fixed world axes), or "CUSTOM" (the angular_axis_custom vector). Ints 0/1/2 map to PC1/PC2/PC3
- **angular_speed**: Signed spin speed in degrees per second (0 = no spin)
- **angular_axis_custom**: World [x, y, z] axis used when angular_axis is "CUSTOM" (normalized before use)
- **enable_translational**: Overwrite the translational velocity at this frame (False leaves translation alone, for a pure spin)
- **enable_angular**: Overwrite the angular velocity at this frame

### set_collision_window(group_uuid: str, object_name: str, index: int, frame_start: int=None, frame_end: int=None)

Change the bounds of one collision window an object already carries,
addressed by its index.

Either bound may be given on its own and the other is left as it is. The
window that results is what gets validated, so moving frame_start past the
frame_end already stored is refused instead of being kept as an inverted
window that turns contact off for the whole run.

Editing in place also keeps the window at its index, which is how
list_collision_windows and remove_collision_window address it.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object
- **index**: Zero-based index into the object's collision_windows list
- **frame_start**: First frame of the window (>= 1)
- **frame_end**: Last frame of the window (>= frame_start)

### move_pin_operation(group_uuid: str, vertex_group_identifier: str, index: int, new_index: int)

Move one of a pin's operations to another position in its list.

The order is semantic rather than presentational: the operations are
shipped to the solver in list order and compose in that order, so moving
an entry changes the motion the pin performs. Index 0 is the head of the
list, and the entry that follows composes on top of what precedes it.

Both positions must address an entry that exists, and they must differ.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form
- **index**: Zero-based index of the operation to move
- **new_index**: Zero-based position to move it to

### move_static_op(group_uuid: str, object_name: str, index: int, new_index: int)

Move one of an object's static ops to another position in its list.

The order is semantic rather than presentational: the ops are shipped to
the solver in list order and compose in that order, so moving an entry
changes the motion the object performs. Index 0 is the head of the list,
and the entry that follows composes on top of what precedes it.

Both positions must address an entry that exists, and they must differ.

**Parameters:**

- **group_uuid**: UUID of STATIC group
- **object_name**: Name of the assigned object
- **index**: Zero-based index of the static op to move
- **new_index**: Zero-based position to move it to

### set_bend_reference(group_uuid: str, object_name: str, reference_object_name: str, enable: bool=None)

Point one assigned object's bending rest angle at a reference object.

A reference is a topological COPY of the object whose vertices were moved:
the same vertex count and the same connectivity (faces for a SHELL, edges
for a ROD), with only positions differing. Its modifiers and geometry
nodes are evaluated before the comparison, so a copy shaped by a modifier
is a valid reference. A curve rod is compared at control-point level
instead, which is how a curve rod is shipped, and a curve modifier is not
sampled there. Anything that fails the comparison is refused here, naming
the mismatch, rather than at scene build.

The group's own bend_rest_from_reference flag is what makes the group read
a reference at all, so it has to be on before a reference can be set; turn
it on with set_group_material_properties. Only SHELL and ROD groups carry
that flag.

Pass an empty reference_object_name to clear the reference, which also
stops this object reading one. Clearing is accepted whatever the group
flag holds, so a stale reference can always be taken off.

**Parameters:**

- **group_uuid**: UUID of the SHELL or ROD group
- **object_name**: Name of the assigned object whose rest angle comes from the reference
- **reference_object_name**: Name of the reference object, or "" to clear the reference this object holds
- **enable**: Whether this object reads its reference. Defaults to True when a reference is given and False when one is cleared, so it is worth naming only to record a reference without using it yet
## Mesh cleaning

Wrappers over the Utility Tools panel's Mesh Cleaning operators, for the geometry the solver rejects: near-coincident vertices, faceless vertices, zero-area faces, duplicate faces, n-gons, and inconsistent face winding.

Every tool takes an explicit `object_names` list and does not read the viewport selection. Each name must resolve to a MESH object in the active view layer, and the whole list is resolved before anything runs, so a typo in the third name cannot leave the first two already repaired. A named object that cannot be selected (hidden in the viewport, hidden by its collection, or carrying Disable Selection) is refused by name rather than skipped: these tools drive operators that read the selection, so a skipped object would be missing from a result that otherwise reads as a success. Make it visible and selectable, then call again. Blender is switched to Object Mode for the call (the operators' poll requires it) and the previous selection is restored afterward. Objects sharing one mesh datablock are repaired once, not once per user.

`scan_meshes` is read-only. The other seven modify meshes. `merge_by_distance`, `remove_loose_vertices` and `dissolve_degenerate_faces` change the vertex count and refuse until `acknowledge=true`, mirroring the panel's confirmation dialog; their refusal message names what the change would invalidate. `symmetric_triangulate` also changes the vertex count, but takes no acknowledgement and deletes nothing it invalidates.

Every repair returns `changed` (one entry per object whose measured quantities moved, each carrying a `<name>_before` / `<name>_after` pair), `changed_count`, and `operator_status` (the operator's own return set, reported under this name so it cannot be mistaken for the MCP envelope's `status`). The three count-changing repairs also return `cleared_caches`, the caches that were actually deleted. A repair that finds nothing returns `changed_count: 0` and a "Nothing to repair" message.

A vertex-count change invalidates a captured deformation and the display cache, and can shift which vertices a pin group holds. Run `scan_meshes` first: its `dependents` field reports exactly what each object would lose. After any repair, run Transfer again before the next simulation.

### scan_meshes(object_names: list[str], merge_threshold: float=1e-4, area_eps: float=0.0)

Scan meshes for geometry the solver rejects, without modifying anything.

Reports per object, split into errors (near-coincident vertices, isolated and hanging vertices, duplicate and degenerate faces, linked duplicates, inconsistent winding) and notes (boundary edges, non-manifold edges, re-splittable quads). Notes are normal for cloth: an open quad panel is not a defect.

**Parameters:**

- **object_names**: Mesh objects to scan
- **merge_threshold**: Vertices closer than this (local units) count as near-coincident. Matches Blender's Merge by Distance default
- **area_eps**: Faces at or below this area (local units squared) count as degenerate. Zero reports only exactly zero-area faces

**Returns:** Dict with `reports` (one per object: `object`, `n_verts`, `n_polys`, `defects`, `n_errors`, `n_notes`, `total`, and `dependents`), `objects_needing_attention`, `total_errors`, `total_notes`. The full per-vertex index lists are stripped from `defects`; every defect keeps its `count`, and the vertex-level ones keep a `preview` of up to eight indices.

### merge_by_distance(object_names: list[str], merge_threshold: float=1e-4, acknowledge: bool=False, clear_stale_caches: bool=True)

Weld near-coincident vertices. Changes the vertex count.

A pair of vertices separated by a tiny gap drives the contact barrier's mass/gap^2 stiffness through the conditioning of the solver's fp32 Newton matrix, so welding them is what makes such a mesh simulable.

**Parameters:**

- **object_names**: Mesh objects to repair
- **merge_threshold**: Weld vertices closer together than this, local units
- **acknowledge**: Must be true. Confirms the vertex-count change and the caches it invalidates, which scan_meshes reports as dependents
- **clear_stale_caches**: Delete the capture and display caches the change invalidates, which the result reports as cleared_caches. True by default, the same value the panel's dialog opens with. Pass false to keep them, and expect the viewport overlay to read data sized for the old vertex count until Transfer rewrites it

### remove_loose_vertices(object_names: list[str], acknowledge: bool=False, clear_stale_caches: bool=True)

Delete vertices that belong to no face, together with their loose edges. Changes the vertex count.

A faceless vertex carries no elastic energy, and the solver averages a vertex's contact parameters over its incident faces, so the build aborts when a vertex has none. Pinned vertices are exempt and are never removed. A SAND particle mesh is skipped whole, since every grain center is legitimately faceless.

**Parameters:**

- **object_names**: Mesh objects to repair
- **acknowledge**: Must be true. Confirms the vertex-count change and the caches it invalidates, which scan_meshes reports as dependents
- **clear_stale_caches**: Delete the capture and display caches the change invalidates, which the result reports as cleared_caches. True by default, the same value the panel's dialog opens with. Pass false to keep them, and expect the viewport overlay to read data sized for the old vertex count until Transfer rewrites it

### dissolve_degenerate_faces(object_names: list[str], merge_threshold: float=1e-4, acknowledge: bool=False, clear_stale_caches: bool=True)

Collapse zero-area and slivered faces. Changes the vertex count.

A face with no area has no well-defined normal, which is what the contact and bending terms are built on.

**Parameters:**

- **object_names**: Mesh objects to repair
- **merge_threshold**: Edges shorter than this (local units) are collapsed
- **acknowledge**: Must be true. Confirms the vertex-count change and the caches it invalidates, which scan_meshes reports as dependents
- **clear_stale_caches**: Delete the capture and display caches the change invalidates, which the result reports as cleared_caches. True by default, the same value the panel's dialog opens with. Pass false to keep them, and expect the viewport overlay to read data sized for the old vertex count until Transfer rewrites it

### delete_duplicate_faces(object_names: list[str])

Delete faces that repeat an existing face's vertex set.

Two faces on the same vertices contribute their contact and elastic terms twice. The vertex count is unchanged, so no cache is invalidated and no acknowledgement is needed.

**Parameters:**

- **object_names**: Mesh objects to repair

### triangulate_for_solver(object_names: list[str])

Triangulate n-gons and quads with a single diagonal per face.

The vertex count is unchanged, so no cache is invalidated. Transfer triangulates on its own at encode time; use this when the triangulation has to be visible and stable in the viewport. For a mesh whose symmetry matters under bending, prefer `symmetric_triangulate`.

**Parameters:**

- **object_names**: Mesh objects to triangulate

### recalculate_normals_outside(object_names: list[str])

Make face winding consistent and outward.

Inconsistent winding flips the normal a face contributes, which the contact and inflate terms read. The vertex count is unchanged, so no cache is invalidated, and the repair is reported as the `bad_winding_before` / `bad_winding_after` edge counts rather than as an element delta.

**Parameters:**

- **object_names**: Mesh objects to repair

### symmetric_triangulate(object_names: list[str])

Triangulate by poking each face, keeping the mesh mirror-symmetric.

A single-diagonal triangulation breaks a symmetric mesh's symmetry, which shows up as a lopsided drape under bending. Poking inserts a center vertex and fans the face into triangles instead, so it adds one vertex per face and therefore invalidates a captured deformation and the display cache, exactly as the count-changing repairs do. It is a Utility Tools operation rather than a repair, so it takes no acknowledgement and deletes nothing it invalidates: the vertex deltas come back in the result, and Transfer and Capture Deformation are what re-take the stale caches.

**Parameters:**

- **object_names**: Mesh objects to triangulate


### triangulate_degenerate_faces()

Re-split only the faces whose tessellation leaves the solver no rest shape.

This is the targeted repair the Transfer refusal names. Blender splits a
quad along one of its two diagonals, and on a quad whose corner sits on,
or very near, the straight edge between its neighbors that diagonal
produces a triangle of three nearly collinear vertices. The solver inverts
each rest triangle once at scene build and the elastic Hessian is
quadratic in that inverse. An exactly zero-area triangle aborts the build
on a degenerate-face assertion; a merely near-collinear one clears that
assertion, inverts to a finite but enormous rest matrix, and reaches the
linear solve as a non-finite Hessian that names no geometry. The test is
the conditioning of the rest matrix against sqrt(float32 eps), not an area
threshold, so a thin triangle above that ratio is legitimate geometry and
is left alone.

Only flagged faces are split, and only those whose replacement fill is
measured sound before the split. Every other face keeps its shape, which is
what separates this from triangulate_for_solver and symmetric_triangulate:
each of those rewrites every quad and n-gon of the mesh. The vertex count
does not change, so no cache is invalidated; the face count grows by one
face per split quad and by more for a wider n-gon.

There is no object argument, because the operator underneath offers no
property to narrow its scope: it scans the included objects of every
active dynamics group except SAND, which is the set Transfer refuses over,
and repairs all of them in one pass. Each mesh datablock is repaired once,
so two objects sharing a mesh are reported under a single name.

The result carries, per object, how many flagged faces the repair cleared
and how many faces the mesh gained. A face no split can rescue is left
exactly as it is and comes back under ``still_degenerate``: a face that is
already a triangle is its own only triangulation, and one with no area or
a zero-length boundary edge forces a degenerate triangle into every
triangulation. Those need the offending vertex moved, merge_by_distance to
weld coincident vertices, or dissolve_degenerate_faces. A call that finds
nothing flagged, or nothing triangulating can repair, is refused rather
than reported as a success. Run Transfer again afterward.
## Simulation

### transfer_data()

Transfer data to the solver.

### run_simulation()

Start simulation.

### resume_simulation()

Resume paused simulation.

### terminate_simulation()

Force terminate simulation.

### save_and_quit_simulation()

Save and quit simulation gracefully.

### update_params()

Update the parameters of the solver.

### delete_remote_data()

Delete data on the remote server.

### fetch_animation()

Fetch simulation results from server.

### clear_local_animation()

Clear local animation data and keyframes.

### list_checkpoint_frames()

List resumable checkpoint frames (Blender 1-based) saved on the server, read from the latest status response. Empty until a checkpoint has been saved (via Save Checkpoints, Auto Save, or Save State on Finish).

### resume_simulation_from(frame: int)

Resume the simulation from a specific saved checkpoint frame without re-uploading or rebuilding (frames before the checkpoint are kept, the rest overwritten). Refuses on geometry drift (transfer_data + run_simulation instead) or parameter drift (update_params first). `resume_simulation` continues from the latest checkpoint.

- **frame**: Saved checkpoint frame to resume from (Blender 1-based)

### export_usd(filepath: str)

Export the simulated mesh sequence as a USD cache.

A lighter alternative to baking shape keys: the deformation is sampled per frame from the solver cache into a file other DCC tools can play back, and the scene itself is left untouched. Every frame must be fetched first; call `fetch_animation` and wait for it to finish. The call is also refused while another solver activity is in progress, outside Object Mode, and when no simulated mesh sequence exists. Rod and curve objects are not carried by this format; their names come back in `excluded_curves`, and Bake Animation is the route for them.

**Parameters:**

- **filepath**: Destination path. A leading `//` is resolved against the .blend, and the parent directory must already exist. The suffix picks the USD flavor: `.usdc` (crate, the suffix the panel's file browser offers), `.usda` (ASCII), `.usd`, or `.usdz` (package)

**Returns:** Dict with the resolved `filepath`, the `format`, and `excluded_curves` when any simulated curve was skipped

### export_alembic(filepath: str)

Export the simulated mesh sequence as an Alembic (ABC) cache.

Same preconditions and exclusions as `export_usd`: every frame fetched, no other solver activity in progress, Object Mode, at least one simulated mesh sequence, and rods and curves are not carried.

**Parameters:**

- **filepath**: Destination `.abc` path. A leading `//` is resolved against the .blend, and the parent directory must already exist

**Returns:** Dict with the resolved `filepath`, the `format`, and `excluded_curves` when any simulated curve was skipped


### get_fetch_status()

Report which simulated frames have been fetched back into Blender.

Fetching is a modal operation: `fetch_animation` returns as soon as it has
started, so a caller needs a separate way to see how far it got. The export
tools refuse while any frame is still unfetched, and `bpy.ops` hands back
only a cancelled status without the reason, so this reports the export
preflight verdict alongside the frame list.

`fetched_frames` is what landed locally, which is a different question from
`list_checkpoint_frames` (frames saved on the remote) and from
`get_remote_status` (how the run itself is doing).

### abort_bake()

Stop a running keyframe bake and undo what it has written so far.

Baking runs as a modal job that keeps going after bake_group_animation,
bake_all_animation, bake_group_single_frame or bake_all_single_frame
returns. This raises the job's abort flag. The job stops on its next timer
tick and rolls back what it wrote: the shape keys and F-curves it added are
removed and the curve handle types it changed are restored, leaving the PC2
caches, the ContactSolverCache modifiers and group membership as they were
before the bake started.

Refused when no bake is running, and refused again while an abort of the
same bake is already in flight. Call get_modal_job_status for the jobs
running now, and poll it afterwards until the bake reports running false.

This stops the bake inside Blender. abort_operation stops an operation on
the solver server, which is a different job.

### abort_static_deformation_capture()

Stop a running STATIC collider deformation capture.

capture_static_deformation and recapture_all_deformations start a modal job
that steps the timeline and samples the shape of each deforming STATIC
collider. This raises the job's abort flag. The job stops on its next timer
tick, restores the frame it started from, and re-enables the
ContactSolverCache modifiers it suspended for the sampling.

A capture writes an object's result only once every frame of that object is
sampled, so the frames taken before the abort are discarded and each object
keeps the deformation cache it already had. get_static_deformation_status
reports what is on an object; run the capture again to record it.

Refused when no static capture is running, and refused again while an abort
of it is already in flight. Call get_modal_job_status for the jobs running
now, and poll it afterwards until this job reports running false.

### abort_pin_deformation_capture()

Stop a running pin deformation capture.

capture_pin_deformation and recapture_all_deformations start a modal job
that steps the timeline and samples the moving pin vertices of each
animated pin. This raises the job's abort flag. The job stops on its next
timer tick, restores the frame it started from, and re-enables the
ContactSolverCache modifiers it suspended for the sampling.

A capture writes a pin's result only once every frame of that pin is
sampled, so the frames taken before the abort are discarded and each pin
keeps the capture it already had. get_pin_deformation_status reports what is
on a pin; run the capture again to record it.

Refused when no pin capture is running, and refused again while an abort of
it is already in flight. Call get_modal_job_status for the jobs running now,
and poll it afterwards until this job reports running false.

### get_modal_job_status()

Report which long-running bake or capture job is running right now.

Three jobs run on a timer inside Blender and outlive the tool call that
started them, so a caller that starts one has no other way to tell whether
it is still going: the keyframe bake, the STATIC collider deformation
capture, and the pin deformation capture. This reports all three in one
call, each with the frames it has processed and the tool that stops it.

``jobs`` carries one entry per job. ``bake`` is started by
bake_group_animation, bake_all_animation, bake_group_single_frame or
bake_all_single_frame and stopped by abort_bake.
``static_deformation_capture`` is started by capture_static_deformation or
recapture_all_deformations and stopped by
abort_static_deformation_capture. ``pin_deformation_capture`` is started by
capture_pin_deformation or recapture_all_deformations and stopped by
abort_pin_deformation_capture.

``abort_requested`` is true once the abort tool has been called and the job
has not yet reached the tick that stops it. ``frames_done``,
``frames_total``, ``item_count`` (objects for the bake and the STATIC
capture, pins for the pin capture) and ``status_line`` are null while a job
is not running, because those counters are cleared when a job ends.

This covers the jobs running inside Blender. get_fetch_status reports how
much of a solve has been fetched back into Blender and whether an export
would be accepted, and get_remote_status reports the run on the server.
## Scene

### clear_solver()

Reset the entire solver state to defaults.

### add_invisible_wall(position: list[float], normal: list[float])

Add an invisible wall collider at a given position and normal.

**Parameters:**

- **position**: Wall origin in Blender world space [x, y, z].
- **normal**: Outward-facing normal vector [x, y, z].

### add_invisible_sphere(position: list[float], radius: float, invert: bool=False, hemisphere: bool=False)

Add an invisible sphere collider.

**Parameters:**

- **position**: Center in Blender world space [x, y, z].
- **radius**: Sphere radius.
- **invert**: If true, acts as an inverted sphere (contact from inside).
- **hemisphere**: If true, only the upper half acts as a collider.

### list_invisible_colliders()

Return a list of all invisible colliders currently in the scene.

### remove_invisible_collider(index: int)

Remove an invisible collider by its index in the scene list.

**Parameters:**

- **index**: Zero-based index as reported by list_invisible_colliders.

### clear_invisible_colliders()

Remove every invisible collider from the scene.

### add_merge_pair(object_a: str, object_b: str)

Stitch two objects together along their nearest overlapping vertices.

**Parameters:**

- **object_a**: Name of the source object.
- **object_b**: Name of the target object.

### remove_merge_pair(object_a: str, object_b: str)

Remove a merge pair by the two object names.

**Parameters:**

- **object_a**: Name of the source object.
- **object_b**: Name of the target object.

### list_merge_pairs()

Return all stored merge pairs with both display names and UUIDs.

### clear_merge_pairs()

Remove every merge pair from the scene.

### snap_to_vertices(object_a: str, object_b: str)

Move object A so its nearest vertex matches object B's nearest vertex.

**Parameters:**

- **object_a**: Name of the object that will move.
- **object_b**: Name of the target object (stays put).

### bake_all_animation()

Bake simulated animation for every dynamic group to Blender keyframes.

### bake_all_single_frame()

Bake the current frame as frame 1 for every dynamic group.


### get_scene_info()

Enumerate the current Blender scene: objects, frame range, and groups.

This is the starting point for an agent that did not create the scene: it
reports what is in the file and which objects are already assigned to a
dynamics group, so the caller can tell setup work that remains from work
already done.

Returns the scene's frame range as Blender holds it, alongside the
simulation frame count and fps the solver will actually use, which are
separate values and are resolved differently.

### set_merge_pair_properties(object_a: str, object_b: str, stitch_stiffness: float=None, show_stitch: bool=None)

Set one merge pair's own stitch stiffness and stitch visualization.

This ``stitch_stiffness`` belongs to the PAIR and is a separate solver
input from the group parameter of the same name that
set_group_material_properties writes: the solver scales this pair's
stitch gradient and Hessian by it directly, with no mass or dt
normalization, so raise it to hold this one seam harder.

The value reaches the solver only through the stitch anchors captured at
snap time, so it stays inert on a pair whose ``stitch_row_count`` (see
list_merge_pairs) is 0; call resnap_merge_pair to build the anchors. An
argument left out is not written.

**Parameters:**

- **object_a**: Name of one object in the pair.
- **object_b**: Name of the other object in the pair, in either order.
- **stitch_stiffness**: Stiffness of this pair's stitch, 0 or greater.
- **show_stitch**: Draw this pair's stitch in the viewport.

### resnap_merge_pair(object_a: str, object_b: str)

Re-run the snap on an existing merge pair to rebuild its stitch.

The two objects must already form a merge pair (add_merge_pair or
snap_to_vertices). The snap MOVES one of them: object A of the STORED
pair, unless that side is in a STATIC group, in which case the other side
moves instead. Which object moves therefore follows the stored pair, not
the argument order used here. The two are left a small gap apart, sized
from their contact offsets, rather than coincident.

This is what makes a pair's stitch anchors current after either mesh was
edited, and what gives a pair anchors at all when it was created without
a snap. A pair whose ``stitch_row_count`` stays 0 forms no stitch at
solve time.

**Parameters:**

- **object_a**: Name of one object in the pair.
- **object_b**: Name of the other object in the pair, in either order.
## Dynamic parameters

### add_dynamic_param(param_type: str)

Add a dynamic (time-varying) scene parameter.

Creates an initial keyframe at frame 1 seeded from the current static scene value.

**Parameters:**

- **param_type**: One of GRAVITY, WIND, AIR_DENSITY, AIR_FRICTION, VERTEX_AIR_DAMP

### remove_dynamic_param(param_type: str)

Remove a dynamic scene parameter entry.

**Parameters:**

- **param_type**: One of GRAVITY, WIND, AIR_DENSITY, AIR_FRICTION, VERTEX_AIR_DAMP

### list_dynamic_params()

List all dynamic scene parameters and their keyframes.

### add_dynamic_param_keyframe(param_type: str, frame: int, gravity: Optional[list[float]]=None, wind_direction: Optional[list[float]]=None, wind_strength: Optional[float]=None, value: Optional[float]=None, use_hold: Optional[bool]=None)

Add a keyframe to a dynamic scene parameter.

Supply the field matching the param_type (gravity for GRAVITY; wind_direction + wind_strength for WIND; value for the scalar params).

**Parameters:**

- **param_type**: GRAVITY, WIND, AIR_DENSITY, AIR_FRICTION, or VERTEX_AIR_DAMP
- **frame**: Blender frame (>= 1)
- **gravity**: [x, y, z] for GRAVITY param
- **wind_direction**: [x, y, z] for WIND param
- **wind_strength**: Scalar speed (m/s) for WIND param
- **value**: Scalar for AIR_DENSITY, AIR_FRICTION, or VERTEX_AIR_DAMP
- **use_hold**: Hold previous keyframe value (step function)

### remove_dynamic_param_keyframe(param_type: str, frame: int)

Remove a keyframe from a dynamic scene parameter.

The initial keyframe (frame 1) cannot be removed.

**Parameters:**

- **param_type**: GRAVITY, WIND, AIR_DENSITY, AIR_FRICTION, or VERTEX_AIR_DAMP
- **frame**: Frame number of the keyframe to remove

### set_collider_properties(index: int, name: Optional[str]=None, position: Optional[list[float]]=None, normal: Optional[list[float]]=None, radius: Optional[float]=None, contact_gap: Optional[float]=None, friction: Optional[float]=None, thickness: Optional[float]=None, invert: Optional[bool]=None, hemisphere: Optional[bool]=None, enable_active_duration: Optional[bool]=None, active_duration: Optional[int]=None)

Update properties on an invisible collider.

Pass only the fields you want to change. `normal` is wall-only; `radius`/`invert`/`hemisphere` are sphere-only.

**Parameters:**

- **index**: Zero-based collider index as reported by list_invisible_colliders
- **name**: Display name
- **position**: [x, y, z] origin
- **normal**: [x, y, z] outward normal (WALL only)
- **radius**: Sphere radius (SPHERE only)
- **contact_gap**: Contact gap tolerance
- **friction**: Friction coefficient [0, 1]
- **thickness**: Max penetration depth (> 0)
- **invert**: Flip contact direction (SPHERE only)
- **hemisphere**: Restrict to upper half (SPHERE only)
- **enable_active_duration**: Enable per-collider active-until frame
- **active_duration**: First frame the collider is no longer active

### add_collider_keyframe(index: int, frame: int, position: Optional[list[float]]=None, radius: Optional[float]=None, use_hold: Optional[bool]=None)

Add a keyframe to an invisible collider.

**Parameters:**

- **index**: Zero-based collider index
- **frame**: Blender frame (>= 1)
- **position**: [x, y, z] at this keyframe
- **radius**: Sphere radius at this keyframe (SPHERE only)
- **use_hold**: Hold the previous keyframe value (step function)

### remove_collider_keyframe(index: int, frame: int)

Remove a keyframe from an invisible collider.

**Parameters:**

- **index**: Zero-based collider index
- **frame**: Frame number of the keyframe to remove

### list_collider_keyframes(index: int)

List keyframes on an invisible collider.

**Parameters:**

- **index**: Zero-based collider index

## Remote

### abort_operation()

Abort the current in-progress operation.

### install_paramiko()

Install the Paramiko library.

### install_docker()

Install the Docker library.

### set_scene_parameters(step_size: Optional[float]=None, min_newton_steps: Optional[int]=None, frame_count: Optional[int]=None, frame_rate: Optional[int]=None, gravity: Optional[list[float]]=None, wind_direction: Optional[list[float]]=None, wind_strength: Optional[float]=None, air_density: Optional[float]=None, air_friction: Optional[float]=None, vertex_air_damp: Optional[float]=None, inactive_momentum_frames: Optional[int]=None, contact_nnz: Optional[int]=None, line_search_max_t: Optional[float]=None, constraint_ghat: Optional[float]=None, cg_max_iter: Optional[int]=None, cg_tol: Optional[float]=None, include_face_mass: Optional[bool]=None, disable_contact: Optional[bool]=None, auto_save: Optional[bool]=None, auto_save_interval: Optional[int]=None, save_state_on_finish: Optional[bool]=None, keep_states: Optional[int]=None, precond: Optional[str]=None, schwarz_levels: Optional[int]=None, use_scene_fps: Optional[bool]=None, project_name: Optional[str]=None)

Set global scene parameters for physics simulation.

**Parameters:**

- **step_size**: Simulation step size (seconds)
- **min_newton_steps**: Minimum Newton iterations per step
- **frame_count**: Number of simulation frames
- **frame_rate**: Frame rate for simulation
- **gravity**: Gravity acceleration vector [x, y, z] m/s^2
- **wind_direction**: Wind direction vector [x, y, z]
- **wind_strength**: Wind speed magnitude (m/s)
- **air_density**: Air density (kg/m^3)
- **air_friction**: Tangential/normal air friction ratio
- **vertex_air_damp**: Vertex-level air damping factor
- **inactive_momentum_frames**: Inactive momentum frame count
- **contact_nnz**: Max contact non-zero entries
- **line_search_max_t**: CCD TOI extension factor
- **constraint_ghat**: Boundary constraint gap distance
- **cg_max_iter**: PCG max iterations
- **cg_tol**: PCG relative tolerance
- **include_face_mass**: Include shell face mass for solids' surface elements
- **disable_contact**: Disable all contact detection
- **auto_save**: Enable auto-save
- **auto_save_interval**: Auto-save interval (frames)
- **save_state_on_finish**: Save a resumable state when the simulation finishes (default False)
- **keep_states**: Number of most-recent saved states to retain (0 = keep all, the default)
- **precond**: PCG preconditioner, "BLOCK_JACOBI" (default) or "SCHWARZ"
- **schwarz_levels**: Number of additive Schwarz levels, 1 (single-level smoother) or 2 (two-level coarse correction, default). Only used when precond is "SCHWARZ".
- **use_scene_fps**: Run the simulation at the Blender scene's frame rate instead of the `frame_rate` field
- **project_name**: Project name used for remote session directory

### get_scene_parameters()

Get current scene parameters.

### set_save_checkpoint_frames(frames: list[int])

Set the explicit frames at which to save a resumable checkpoint, replacing the current list (de-duplicated, clamped to >= 1, sorted ascending). These are the frames the Resume dialog offers, in addition to Auto Save and Save State on Finish.

- **frames**: Frame indices (1-based) to save checkpoints at

### clear_save_checkpoint_frames()

Clear all explicit Save Checkpoints frames.

### list_save_checkpoint_frames()

List the explicit Save Checkpoints frames configured for the next run.

## Console

### get_console_lines()

Get current console text lines.

### get_latest_error()

Get latest error from both local and remote.

### show_console()

Show console window.

## Debug

### debug_data_send(data_size_mb: int=1)

Send test data to remote server for debugging data transfer.

**Parameters:**

- **data_size_mb**: Size of test data in megabytes (default: 1MB)

### debug_data_receive()

Receive test data from remote server and verify integrity.

This function should be called after debug_data_send to test the complete round-trip data transfer functionality.

### execute_server_command(server_script: str)

Execute a server command/script.

**Parameters:**

- **server_script**: Server script command to execute

### execute_shell_command(shell_command: str, use_shell: bool=True)

Execute a shell command on remote server.

**Parameters:**

- **shell_command**: Shell command to execute
- **use_shell**: Whether to use shell execution

### git_pull_remote()

Pull the latest changes from the Git repository on remote server.

### compile_project()

Compile the project on remote server.

### delete_log_file(log_file_path: str)

Delete the specified log file.

**Parameters:**

- **log_file_path**: Path to the log file to delete

### git_pull_local()

Pull the latest changes from the local Git repository.

## Blender

### run_python_script(code: str)

Execute arbitrary Python code in Blender with access to bpy, bmesh, and mathutils modules.

**Parameters:**

- **code**: Python code to execute in Blender context

### capture_viewport_image(filepath: str, max_size: int=800)

Capture a screenshot of the current 3D viewport and save it to specified file path.

**Parameters:**

- **filepath**: File path where to save the screenshot
- **max_size**: Maximum size in pixels for the largest dimension

### create_curve(name: str, bevel_depth: float=0.0, bevel_resolution: int=2, resolution_u: int=4, dimensions: str='3D', clear_existing: bool=True)

Create a pending curve builder for ROD-scene authoring.

**Parameters:**

- **name**: Object name for the curve to be finalized later
- **bevel_depth**: Tube radius for viewport visualization
- **bevel_resolution**: Tube cross-section subdivisions
- **resolution_u**: Spline interpolation resolution
- **dimensions**: Curve dimensions ("3D" or "2D")
- **clear_existing**: Remove an existing same-name object before finalize

### add_curve_spline(name: str, points: list[list[float]], closed: bool=False)

Append a Bezier spline to a pending curve builder.

**Parameters:**

- **name**: Curve builder name passed to create_curve
- **points**: List of [x, y, z] control-point coordinates
- **closed**: Whether to make the spline cyclic

### set_curve_material(name: str, spline_index: int, material_name: str, create_if_missing: bool=False)

Bind a Blender material to a spline on a pending curve builder.

**Parameters:**

- **name**: Curve builder name passed to create_curve
- **spline_index**: Spline index returned by add_curve_spline
- **material_name**: Existing Blender material name
- **create_if_missing**: Create the material when it does not exist

### finalize_curve(name: str)

Finalize a pending curve builder, link it to the scene, and return the object.

**Parameters:**

- **name**: Curve builder name passed to create_curve

### get_ui_element_status(element_type: str='all', element_name: Optional[str]=None, category: Optional[str]=None)

Get status of Blender addon UI elements - poll results for operators, values for properties.

**Parameters:**

- **element_type**: Type of elements to check ("operator", "property", "all")
- **element_name**: Specific element name to check (optional)
- **category**: Filter by category ("solver", "dynamics", "client", "debug")

### get_average_edge_length(object_name: str)

Compute the average edge length of a mesh object.

**Parameters:**

- **object_name**: Name of the mesh object to analyze

### get_object_bounding_box_diagonal(object_name: str)

Compute the bounding box of an object and return the largest diagonal distance.

**Parameters:**

- **object_name**: Name of the object to analyze

### refresh_ui()

Refresh all UI areas in Blender to reflect recent changes.

This is useful when programmatic changes need to be reflected in the UI, such as after starting/stopping servers or updating addon state.

---

Bundled MCP reference synced to `blender_addon/mcp/handlers/*.py` and `blender_addon/mcp/blender_handlers.py`.

## Statistics

Read what the solver measured per object, frame by frame, from the statistics
cache written by the frame-fetch path. These report a past run, not what the
scene would produce if it were run now.

### list_statistics_objects()

List the objects the solver recorded statistics for, with their channels.

The statistics are whatever is on disk from the last simulation whose
frames were fetched, so this reports a past run, not what the scene would
produce if it were run now.

``object_name`` is the object's name in the scene at this moment and is
null when its UUID resolves to nothing, which happens once the object is
deleted; pass ``object_uuid`` to the other statistics tools in that case.
``recorded_name`` and ``dynamics_type`` are what the solver stored at run
time. ``channels`` holds the channel ids measured for that object, which is
the set get_object_statistics_series accepts for it, and
``channel_catalog`` gives every channel's label and unit.

``start_frame`` is the Blender frame the solve starts on, which every frame
number in these tools is expressed against.

### get_object_statistics(object_name: str, frame: int)

Read every channel the solver measured for one object at one frame.

``frame`` is a Blender timeline frame, the same number the statistics panel
shows, and it is converted to the solver frame by subtracting the start
frame reported as ``effective_start_frame`` by get_scene_parameters.

Only the channels the object supports are returned, since which quantities
exist depends on what the object is: a rod has a length, a solid has a
volume. The remaining ids are listed under ``unsupported_channels``. A
supported channel whose value the solver did not record for this frame
comes back with a null ``value``.

A frame the run never wrote is refused rather than reported as zero; call
get_object_statistics_series for the frames that are present.

**Parameters:**

- **object_name**: Blender object name, or the object_uuid from list_statistics_objects when the object is gone from the scene.
- **frame**: Blender timeline frame to read.

### get_object_statistics_series(object_name: str, channel: str, frame_start: int=None, frame_end: int=None)

Read one channel of one object across frames, as the CSV export does.

Returns one sample per recorded frame, each carrying the Blender frame, the
simulated time in seconds, and the value. The channel is a single id from
list_statistics_objects, so a vector is read one component at a time
(LOCATION_X, LOCATION_Y, LOCATION_Z), and a channel the object does not
support is refused instead of answered with nulls. A sample whose value the
solver did not record for that frame carries a null ``value``.

The window bounds are Blender frames and both ends are inclusive. Leaving
one out extends the window to the recorded frames on that side, so leaving
both out returns every frame in the cache. An empty ``samples`` list means
no frame in the window has been recorded yet.

**Parameters:**

- **object_name**: Blender object name, or the object_uuid from list_statistics_objects when the object is gone from the scene.
- **channel**: Channel id, for example SPEED or CONTACT_COUNT.
- **frame_start**: First Blender frame to include; omit for the earliest recorded frame.
- **frame_end**: Last Blender frame to include; omit for the latest recorded frame.

## Material maps

Spatial material maps drive a solver parameter from a per-vertex weight source,
reduced to one coefficient per element. A map can also be animated by adding a
sample per frame.

### add_material_map(group_uuid: str, parameter: str, source_type: str, source_name: str, target_value: float, enabled: bool=True)

Add a spatial material map, varying one parameter across the surface.

The value at a vertex is lerp(base, target, weight), where base is the
group's own slider for that parameter and weight is read per vertex from
the named source, clamped to [0, 1]. A weight of 0 therefore reproduces the
unmapped result exactly. Each element takes the mean of its own vertices'
weights.

Only SHELL and SOLID groups carry the element tables a map is reduced over,
and each object type reads a different set of parameters, so 'parameter' is
checked against this group's type. 'pressure' is never mappable. A group
takes at most one enabled map per parameter.

The source is resolved when the scene is built, so the vertex group or
attribute does not have to exist yet. A vertex group is read by name from
the object; an attribute is read from the evaluated mesh on the POINT
domain, which is where a Store Named Attribute node writes one.

**Parameters:**

- **group_uuid**: UUID of the group to add the map to.
- **parameter**: Solver key to vary. One of young-mod, bend, friction, deformation-damping, bending-damping, strain-limit, plasticity, bend-plasticity, bend-warp, bend-weft.
- **source_type**: VERTEX_GROUP to read weight paint, ATTRIBUTE to read a float attribute off the evaluated mesh.
- **source_name**: Name of the vertex group or float attribute holding the weights at the start frame.
- **target_value**: Value reached where the weight is 1, in the same units as the group's own slider for this parameter.
- **enabled**: Whether the map is included in the simulation.

### set_material_map(group_uuid: str, index: int, parameter: str=None, source_type: str=None, source_name: str=None, target_value: float=None, enabled: bool=None)

Edit fields of an existing spatial material map.

Every field left out keeps its current value. The whole resulting row is
validated before anything is written, so a refusal leaves the map exactly
as it was. That means changing 'parameter' alone can be refused because the
target already stored is below the new parameter's own minimum; pass both
in one call.

**Parameters:**

- **group_uuid**: UUID of the group that owns the map.
- **index**: Zero-based index as reported by list_material_maps.
- **parameter**: New solver key to vary, or omit to keep the current one.
- **source_type**: VERTEX_GROUP or ATTRIBUTE, or omit to keep the current one.
- **source_name**: New vertex group or attribute name, or omit to keep it.
- **target_value**: New value reached where the weight is 1, or omit to keep it.
- **enabled**: Whether the map is included in the simulation, or omit to keep the current setting.

### remove_material_map(group_uuid: str, index: int)

Remove a spatial material map and every weight source on it.

**Parameters:**

- **group_uuid**: UUID of the group that owns the map.
- **index**: Zero-based index as reported by list_material_maps. Removing a map renumbers the ones after it.

### list_material_maps(group_uuid: str)

List a group's spatial material maps and its mappable parameters.

Each map reports the slider it blends away from as 'base_property', and
'gate_closed_reason' whenever the parameter is switched off for the whole
solve, in which case the build refuses the map: a map target cannot
reintroduce a value the group turned off.

'available_parameters' is what this group's object type can map, which is
what add_material_map accepts. 'start_frame' is the frame the map's own
source describes, and every weight sample has to sit after it.

**Parameters:**

- **group_uuid**: UUID of the group to report.

### add_material_map_sample(group_uuid: str, index: int, frame: int, source_name: str, source_type: str=None)

Add a later weight source to a spatial material map.

The map's own source is the weights at the simulation start frame, and each
sample names a different source reached at its own frame. Between two
consecutive samples the weights are the linear interpolation of the two, so
a constant hold is two samples naming one source.

A frame at or before the start frame is refused, because the map's own
source already describes that frame. Only a SHELL group carries a
per-element material schedule, so a map on any other type takes a single
source and no samples.

**Parameters:**

- **group_uuid**: UUID of the group that owns the map.
- **index**: Zero-based index of the map, as reported by list_material_maps.
- **frame**: Blender frame at which the weights are exactly this source.
- **source_name**: Vertex group or float attribute holding this sample's weights.
- **source_type**: VERTEX_GROUP or ATTRIBUTE. Omit to use the map's own source type.

### remove_material_map_sample(group_uuid: str, index: int, frame: int)

Remove the weight source at a given frame from a material map.

The map's own source is not a sample and cannot be removed here; change it
with set_material_map instead.

**Parameters:**

- **group_uuid**: UUID of the group that owns the map.
- **index**: Zero-based index of the map, as reported by list_material_maps.
- **frame**: Frame of the sample to remove, as reported by list_material_maps.

## Presets and profiles

A **preset** is a bundle of physically grounded material values the add-on
ships. A **profile** is a named snapshot of settings the artist saved, and
there is one per settings group: scene, material, pin and connection.
Copy and paste move the same settings between objects without naming them.

### apply_material_preset(group_uuid: str, preset_name: str)

Write a bundled material preset's parameters onto a dynamics group.

The preset's object_type has to match the group's Type. Applying one never
changes the Type, so a fabric preset on a SOLID group would write shell
parameters that group's elements never read, and it is refused instead.
Use set_group_type first, or pick a preset for the Type the group has.

A parameter the group has locked keeps its value, which is what the
padlock beside it promises against the tools that overwrite a whole group
at once. Locked parameters are reported under 'kept_locked'.

'written' reports every parameter that now carries the preset's value,
including any that already did. A parameter the group's own RNA refuses,
because the value falls outside the range that property enforces, leaves
the group holding part of the preset and raises rather than reporting
success.

**Parameters:**

- **group_uuid**: UUID of the group to write the preset onto.
- **preset_name**: Preset name as reported by list_material_presets.

### clear_profile_path(kind: str, group_uuid: str=None)

Unbind a profile file from the scene, leaving the file untouched.

This is the panel's Clear button: the scene stops pointing at the file,
and the dropdown for that kind goes empty. Nothing on disk changes, and
the settings the last load applied stay as they are. Removing an entry
from a profile file has no path in the addon, so no tool here does it.

**Parameters:**

- **kind**: SCENE, MATERIAL, PIN or CONNECTION.
- **group_uuid**: UUID of the group the file is bound to, for MATERIAL and PIN.

### copy_material_parameters(group_uuid: str)

Copy a group's material parameters into the addon's material clipboard.

The clipboard holds one set of parameters at a time and lives on the
window manager, so it is not saved in the .blend and is empty again after
a Blender restart. Copying records the source group's Type as well, which
decides which parameters a later paste applies.

This copies parameters only. Identity, the group's Type, its overlay
color, its profile bindings, its per parameter locks and everything owned
by an assigned object stay with their own group.

**Parameters:**

- **group_uuid**: UUID of the group to copy from.

### copy_pin_operations(group_uuid: str, vertex_group_identifier: str)

Copy one pin's operations into the addon's pin operation clipboard.

The clipboard holds the operations of one pin at a time and lives on the
window manager, so it is not saved in the .blend and is empty again after
a Blender restart. The pin named here also becomes the one selected in the
panel, which is how the pin clipboard addresses a pin.

**Parameters:**

- **group_uuid**: UUID of the group that owns the pin.
- **vertex_group_identifier**: Pin in 'object_name::vertex_group_name' form, as reported by list_pins.

### list_material_presets(object_type: str=None)

List the bundled material presets and the parameters each one writes.

A preset carries an object_type that decides which groups may take it: a
SHELL group is offered the fabrics and a SOLID group the soft solids, and
apply_material_preset refuses a mismatch. A Type the library ships no
preset for gives an empty list rather than an error, so an empty result is
an answer and not a failure.

'parameters' is what applying the preset writes, keyed by group property
name. 'unknown_keys' names any key in the preset table that matches no
group property; those are written by nothing, and a non-empty list is a
defect in the bundled file rather than something a caller can act on.

**Parameters:**

- **object_type**: Group Type to filter by, one of SOLID, SHELL, ROD, STATIC, PDRD, SAND. Omit to list every preset.

### list_profiles(kind: str, group_uuid: str=None, path: str=None)

List the entries of a profile file, for one of the four profile kinds.

A profile file holds several entries, one TOML table per name, and the
scene binds one file and one selected entry per kind. With no 'path' the
bound file is read, and a kind with no file bound is refused rather than
reported as empty.

MATERIAL and PIN bind their file to a dynamics group, so both need
group_uuid; SCENE and CONNECTION refuse one.

'unrecognized_keys' names keys of an entry that this kind's apply drops,
which is what an entry saved under a different kind looks like from here.

**Parameters:**

- **kind**: SCENE, MATERIAL, PIN or CONNECTION.
- **group_uuid**: UUID of the group the file is bound to, for MATERIAL and PIN.
- **path**: Profile file to read instead of the bound one. Absolute, or '//' relative to a saved .blend.

### load_profile(kind: str, name: str, group_uuid: str=None, vertex_group_identifier: str=None, path: str=None)

Apply a named entry from a profile file onto the scene.

The entry overwrites every setting its kind covers, so a MATERIAL entry
replaces the group's material parameters, including its Type and, when the
entry embeds pins, the operations of the pins it names. A material lock
does not hold against a profile load; it guards against the presets and
the clipboard.

An entry whose keys this kind's apply understands none of is refused
rather than applied as nothing, which is what loading an entry saved under
a different kind would otherwise look like. Keys the apply does drop are
reported under 'ignored_keys', for an entry written by an older build.

The file and the entry become the selection the panel shows.

**Parameters:**

- **kind**: SCENE, MATERIAL, PIN or CONNECTION.
- **name**: Entry name, as reported by list_profiles.
- **group_uuid**: UUID of the group to write, for MATERIAL and PIN.
- **vertex_group_identifier**: Pin to write, in 'object_name::vertex_group_name' form, for PIN. The pin also becomes the one selected in the panel, which is how the pin profile picker addresses a pin.
- **path**: Profile file to read instead of the bound one. Absolute, or '//' relative to a saved .blend.

### paste_material_parameters(group_uuid: str)

Paste the material clipboard onto a group, keeping its locked values.

Call copy_material_parameters first: the clipboard lives on the window
manager, so a paste is refused after a restart, and after a session that
never copied.

A parameter the target has locked keeps its value, which is what the
padlock beside it promises; those are reported under 'kept_locked'. A
parameter only the source's Type reads is not pasted at all, so a paste
between two Types carries the shared parameters and leaves the target's
own model fields alone. The target's Type never changes.

**Parameters:**

- **group_uuid**: UUID of the group to paste onto.

### paste_pin_operations(group_uuid: str, vertex_group_identifier: str)

Paste the pin operation clipboard onto a pin, replacing its operations.

Every operation the target pin carries is discarded and replaced by the
clipboard's, so this is not an append. Call copy_pin_operations first: the
clipboard lives on the window manager, so a paste is refused after a
restart, and after a session that never copied.

The pin named here also becomes the one selected in the panel, which is
how the pin clipboard addresses a pin.

**Parameters:**

- **group_uuid**: UUID of the group that owns the pin.
- **vertex_group_identifier**: Pin in 'object_name::vertex_group_name' form, as reported by list_pins.

### save_profile(kind: str, name: str, group_uuid: str=None, vertex_group_identifier: str=None, path: str=None)

Save current settings as a named entry in a profile file.

Each kind reads a different part of the scene: SCENE the solver
parameters, the dynamic parameter schedules and the invisible colliders;
MATERIAL one group's material parameters, and no pins; PIN the operations
of one pin; CONNECTION the solver host settings. MATERIAL and PIN need
group_uuid, and PIN also needs vertex_group_identifier.

An entry that already carries this name is replaced, and the result says
so under 'replaced_existing_entry'. Every other entry in the file is kept.
With no 'path' the file already bound to the scene is written; passing one
writes that file and binds it, which is what the panel's Save button does
with a file it was just given. The file and the entry become the selection
the panel shows.

**Parameters:**

- **kind**: SCENE, MATERIAL, PIN or CONNECTION.
- **name**: Entry name to write. An entry named NONE is refused, since that identifier means "no profile" in the dropdowns.
- **group_uuid**: UUID of the group to read, for MATERIAL and PIN.
- **vertex_group_identifier**: Pin to read, in 'object_name::vertex_group_name' form, for PIN. The pin also becomes the one selected in the panel, which is how the pin profile picker addresses a pin.
- **path**: Profile file to write instead of the bound one. Absolute, or '//' relative to a saved .blend.
