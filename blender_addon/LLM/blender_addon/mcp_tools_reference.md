# MCP tool reference

This document tracks the bundled MCP tool surface shipped with the add-on. Every tool listed here is callable over the MCP Streamable HTTP server (`POST /mcp` with `tools/call`, after an `initialize` handshake that returns the `Mcp-Session-Id` to echo on every subsequent request) and, equivalently, via `bpy.ops.zozo_contact_solver.<tool_name>()` inside Blender. See `llm://integrations` for protocol, transport, and security notes.

If you reached this file as MCP resource `llm://mcp_tools_reference`, its sibling resources (`llm://index`, `llm://overview`, `llm://parameters`, and so on) cover the surrounding concepts. Call `resources/list` once and pick the matching URI; the full resource surface (URI scheme, list/read examples, error handling) is documented under the **Resources** section of `llm://integrations`.

## How to invoke

Equivalent ways to call a tool named `<tool>` with JSON `arguments`:

1. HTTP (JSON-RPC `tools/call` over MCP Streamable HTTP). After `initialize` returns an `Mcp-Session-Id` header, echo it on every subsequent request and send `Accept: application/json, text/event-stream`:

```text
curl -s -X POST http://127.0.0.1:9633/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"<tool>","arguments":{...}},"id":1}'
```

See `llm://integrations` for the full `initialize` handshake, session lifecycle, and `DELETE /mcp` termination.

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
- Object operations (28)
- Simulation (11)
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

Disconnect from remote server.

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

## Group

### create_group(name: str='', type: str='SOLID')

Create a new dynamics group.

**Parameters:**

- **name**: Display name for the new group (optional)
- **type**: Group type (SOLID, SHELL, ROD, STATIC, PDRD, SAND). PDRD is an exactly-rigid body type whose surface mesh moves as a single best-fit rigid transform (no tetrahedralization, no Young's/Poisson/bend/shrink/strain/inflate). SAND is a faceless granular body of loose grain-center vertices.

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
- **type**: Group type (SOLID, SHELL, ROD, STATIC, PDRD, SAND)

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

- SHELL: enable_strain_limit, strain_limit_percent, shell_density, shell_young_modulus, shell_poisson_ratio, shell_model, bend, shrink_x, shrink_y, deformation_damping, bending_damping, young_mod_density_normalized, friction, enable_inflate, inflate_pressure, stitch_stiffness, bend_rest_angle_source, bend_rest_from_reference
- SOLID: solid_density, solid_young_modulus, solid_poisson_ratio, solid_model, shrink, deformation_damping, young_mod_density_normalized, friction, stitch_stiffness
- ROD: rod_density, rod_young_modulus, rod_model, deformation_damping, bending_damping, young_mod_density_normalized, friction, bend, enable_strain_limit, strain_limit_percent, stitch_stiffness, bend_rest_angle_source, bend_rest_from_reference
- PDRD: pdrd_density, friction, stitch_stiffness (the hinge joint is per-object; use the `set_pdrd_hinge` tool)
- SAND: sand_grain_radius, sand_particle_mass, sand_friction (faceless granular body of loose grain-center vertices)
- STATIC: friction (limited set)

Per-type property notes:

- pdrd_density: PDRD body volume density (kg/m^3), default 100. Mass is density times the enclosed volume of the surface mesh.
- PDRD hinge: the hinge joint is a per-object setting, not a group material. Set it with the `set_pdrd_hinge` tool (group_uuid, object_name, enable, pca_axis), so each body in a PDRD group can be hinged on its own axle.
- bend_rest_from_reference (SHELL, ROD): group-level master toggle for per-object reference rest angles. Settable via this tool, but the per-object reference itself (which object opts in, and which object is its reference) is not exposed over MCP: it is picked in the add-on UI (the eyedropper that runs `object.pick_bend_reference`). When enabled with a valid reference, that object's bending rest angle (shell hinge dihedral, or rod interior-vertex bend angle) is computed from the reference geometry, overriding `bend_rest_angle_source` for that object. Mesh references are modifier-evaluated (vertex count + connectivity must match); curve-rod references are sampled at the control-point level.
- deformation_damping: stiffness-proportional Rayleigh damping (seconds) for stretch/membrane/solid deformation; default 0.0, min 0.0. Applies to SOLID, SHELL, ROD. 0 disables it. PDRD groups are not Rayleigh-damped.
- bending_damping: stiffness-proportional Rayleigh damping (seconds) for bending; default 0.0, min 0.0. SHELL and ROD only (SOLID/tet has no bending term; rejected for SOLID and PDRD). 0 disables it.
- young_mod_density_normalized: SOLID/SHELL/ROD only. True (default) interprets the Young's modulus field as a density-normalized value (Pa/rho), the solver's native convention. False interprets it as a true Young's modulus in pascals, which the addon divides by this group's density before sending it.
- stitch_stiffness: per-object soft cross-stitch force stiffness, default 1.0. Cross-stitch is a soft 6-slot barycentric force (it replaced the old DOF-fold/exact-weld). Supported pairs: Shell-Shell, Shell-Solid, Rod-Shell, Rod-Solid, Rod-Rod, Solid-Solid, and any dynamic group stitched to a STATIC collider.

Contact properties (mutually exclusive modes):

- Absolute mode: contact_gap, contact_offset (sets use_group_bounding_box_diagonal=False)
- Relative mode: contact_gap_rat, contact_offset_rat (sets use_group_bounding_box_diagonal=True)

**Returns:** Dict with success message and properties set

## Object operations

### set_pin_settings(group_uuid: str, vertex_group_identifier: str, included: Optional[bool]=None, use_pin_duration: Optional[bool]=None, pin_duration: Optional[int]=None, use_pull: Optional[bool]=None, pull_strength: Optional[float]=None, pin_stiffness: Optional[float]=None)

Set per-pin runtime settings (include/duration/pull).

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form
- **included**: Include this pin in the simulation
- **use_pin_duration**: Enable per-pin active duration
- **pin_duration**: Number of frames the pin is active
- **use_pull**: Use pull force instead of hard constraint
- **pull_strength**: Pull force strength
- **pin_stiffness**: Scale on the pin's moving (kinematic) constraint force; 1.0 default, only affects animated pins

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

### set_scene_parameters(step_size: Optional[float]=None, min_newton_steps: Optional[int]=None, frame_count: Optional[int]=None, frame_rate: Optional[int]=None, gravity: Optional[list[float]]=None, wind_direction: Optional[list[float]]=None, wind_strength: Optional[float]=None, air_density: Optional[float]=None, air_friction: Optional[float]=None, vertex_air_damp: Optional[float]=None, inactive_momentum_frames: Optional[int]=None, contact_nnz: Optional[int]=None, line_search_max_t: Optional[float]=None, constraint_ghat: Optional[float]=None, cg_max_iter: Optional[int]=None, cg_tol: Optional[float]=None, include_face_mass: Optional[bool]=None, disable_contact: Optional[bool]=None, auto_save: Optional[bool]=None, auto_save_interval: Optional[int]=None, save_state_on_finish: Optional[bool]=None, keep_states: Optional[int]=None, precond: Optional[str]=None, schwarz_levels: Optional[int]=None, use_frame_rate_in_output: Optional[bool]=None, project_name: Optional[str]=None)

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
- **use_frame_rate_in_output**: Use frame rate in output
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
