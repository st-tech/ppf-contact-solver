# MCP tool reference

This document mirrors the auto-generated `docs/blender_addon/integrations/mcp_reference.rst`. Every tool listed here is callable over the MCP Streamable HTTP server (`POST /mcp` with `tools/call`, after an `initialize` handshake that returns the `Mcp-Session-Id` to echo on every subsequent request) and, equivalently, via `bpy.ops.zozo_contact_solver.<tool_name>()` inside Blender. See `llm://integrations` for protocol, transport, and security notes.

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
- Group (17)
- Object operations (18)
- Simulation (9)
- Scene (12)
- Dynamic parameters (9)
- Remote (5)
- Console (3)
- Debug (8)
- Blender (6)

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

### create_group()

Create a new dynamics group.

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
- **type**: Group type (SOLID, SHELL, ROD, STATIC)

### add_pin_vertex_group(group_uuid: str, vertex_group_identifier: str)

Add a vertex group to the pin list of a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Identifier in format "object_name::vertex_group_name"

### remove_pin_vertex_group(group_uuid: str, vertex_group_identifier: str)

Remove a vertex group from the pin list of a dynamics group.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Identifier in format "object_name::vertex_group_name"

### set_group_material_properties(group_uuid: str, properties: dict)

Set material properties for a dynamics group.

**Parameters:**

- **group_uuid**: UUID of target group
- **properties**: Dict of property_name -> value mappings

Supported properties by group type:

- SHELL: enable_strain_limit, strain_limit, shell_density, shell_young_modulus, shell_poisson_ratio, shell_model, bend, shrink_x, shrink_y, friction, enable_inflate, inflate_pressure, stitch_stiffness
- SOLID: solid_density, solid_young_modulus, solid_poisson_ratio, solid_model, shrink, friction, stitch_stiffness
- ROD: rod_density, rod_young_modulus, rod_model, friction, bend, enable_strain_limit, strain_limit, stitch_stiffness
- STATIC: friction (limited set)

Contact properties (mutually exclusive modes):

- Absolute mode: contact_gap, contact_offset (sets use_group_bounding_box_diagonal=False)
- Relative mode: contact_gap_rat, contact_offset_rat (sets use_group_bounding_box_diagonal=True)

**Returns:** Dict with success message and properties set

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

### add_pin_operation(group_uuid: str, vertex_group_identifier: str, op_type: str, frame_start: Optional[int]=None, frame_end: Optional[int]=None, transition: Optional[str]=None, delta: Optional[list[float]]=None, spin_axis: Optional[list[float]]=None, spin_angular_velocity: Optional[float]=None, spin_flip: Optional[bool]=None, spin_center: Optional[list[float]]=None, spin_center_mode: Optional[str]=None, scale_factor: Optional[float]=None, scale_center: Optional[list[float]]=None, scale_center_mode: Optional[str]=None, torque_axis_component: Optional[str]=None, torque_magnitude: Optional[float]=None, torque_flip: Optional[bool]=None)

Append an operation to a pin's operation list.

TORQUE cannot coexist with other op types on the same pin.

**Parameters:**

- **group_uuid**: UUID of group
- **vertex_group_identifier**: Pin id in 'object::vertex_group' form
- **op_type**: One of MOVE_BY, SPIN, SCALE, TORQUE
- **frame_start**: First frame the op is active
- **frame_end**: Last frame the op is active
- **transition**: LINEAR or SMOOTH
- **delta**: [x, y, z] translation for MOVE_BY (metres)
- **spin_axis**: [x, y, z] rotation axis for SPIN
- **spin_angular_velocity**: Degrees per second (SPIN)
- **spin_flip**: Reverse spin direction
- **spin_center**: [x, y, z] fixed center for SPIN (ABSOLUTE mode only)
- **spin_center_mode**: CENTROID, ABSOLUTE, MAX_TOWARDS, or VERTEX
- **scale_factor**: Scale multiplier for SCALE
- **scale_center**: [x, y, z] fixed center for SCALE (ABSOLUTE mode only)
- **scale_center_mode**: CENTROID, ABSOLUTE, MAX_TOWARDS, or VERTEX
- **torque_axis_component**: PC1, PC2, or PC3 (principal axis)
- **torque_magnitude**: Torque in newton-metres
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

### add_velocity_keyframe(group_uuid: str, object_name: str, frame: int, direction: list[float], speed: float)

Add a velocity keyframe at the given frame for an assigned object.

**Parameters:**

- **group_uuid**: UUID of group
- **object_name**: Name of the assigned object
- **frame**: Blender frame number (>= 1)
- **direction**: [x, y, z] direction vector (normalized at runtime)
- **speed**: Velocity magnitude (m/s)

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

## Scene

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

### set_scene_parameters(step_size: Optional[float]=None, min_newton_steps: Optional[int]=None, frame_count: Optional[int]=None, frame_rate: Optional[int]=None, gravity: Optional[list[float]]=None, wind_direction: Optional[list[float]]=None, wind_strength: Optional[float]=None, air_density: Optional[float]=None, air_friction: Optional[float]=None, vertex_air_damp: Optional[float]=None, inactive_momentum_frames: Optional[int]=None, contact_nnz: Optional[int]=None, line_search_max_t: Optional[float]=None, constraint_ghat: Optional[float]=None, cg_max_iter: Optional[int]=None, cg_tol: Optional[float]=None, include_face_mass: Optional[bool]=None, disable_contact: Optional[bool]=None, auto_save: Optional[bool]=None, auto_save_interval: Optional[int]=None, use_frame_rate_in_output: Optional[bool]=None, project_name: Optional[str]=None)

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
- **use_frame_rate_in_output**: Use frame rate in output
- **project_name**: Project name used for remote session directory

### get_scene_parameters()

Get current scene parameters.

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

Reference mirror of `docs/blender_addon/integrations/mcp_reference.rst` regenerated from `blender_addon/mcp/handlers/*.py` and `blender_addon/mcp/blender_handlers.py`.
