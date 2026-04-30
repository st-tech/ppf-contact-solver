# ZOZO's Contact Solver - Blender Addon

A Blender 5.0+ addon for physics-based contact simulation. Connects to a remote (or local) solver backend, transfers scene geometry and material parameters, runs simulations, and fetches animation results back into Blender.

**Package ID:** `ppf_contact_solver`
**License:** Apache-2.0
**GitHub:** https://github.com/st-tech/ppf-contact-solver

---

## Architecture Overview

```
blender_addon/
├── __init__.py              # Entry point: register/unregister, reload server, group migration
├── blender_manifest.toml    # Blender extension metadata (schema 1.0.0)
├── debug/                   # CLI tools for MCP/debug interaction (not part of addon)
│   ├── __init__.py          # Package docstring
│   ├── client.py            # Transport + control primitives
│   ├── output.py            # Shared response-printing helpers
│   ├── main.py              # General CLI (status, tools, call, exec, reload, ...)
│   └── perf.py              # Profiler CLI (enable, disable, reset, report, sample)
├── example_profile.toml     # Connection profile presets (SSH, Docker, local, Windows)
├── example_material_profile.toml  # Material presets (Flag, Cotton, Silk, Denim, Rubber, Steel, Rope, Static)
├── example_scene_profile.toml     # Scene presets (Default, Windy, HighRes, SlowMotion, ZeroGravity)
├── core/                    # Communication, encoding, protocol, utilities
│   ├── transform.py         # Coordinate transforms (Z-up ↔ Y-up): _swap_axes, world_matrix, etc.
│   ├── encoder/             # Data serialization (mesh, params, pins, dynamics)
│   └── ...
├── ui/                      # Blender panels, operators, property groups
│   ├── state.py             # State/SSHState/SceneRoot PropertyGroups, registration
│   ├── state_types.py       # Collection-item PropertyGroups (keyframes, colliders, pins, etc.)
│   ├── object_group.py      # ObjectGroup PropertyGroup (material params, overlays)
│   └── dynamics/            # Scene config, group/pin management, overlay rendering
│       ├── overlay.py           # Draw callbacks, cache management, registration
│       ├── overlay_geometry.py  # Geometry generation, GPU batch building
│       └── overlay_labels.py    # Text/label rendering (blf)
├── ops/                     # High-level API, dynamic operator generation
├── models/                  # Data access helpers, defaults, console
│   ├── collection_utils.py  # Shared helpers: keyframe sorting, index safety, unique naming
│   └── ...
├── mesh_ops/                # Snap and merge operators
├── mcp/                     # MCP (Model Context Protocol) HTTP server + handlers
│   └── handlers/            # Categorized handler submodules
└── lib/                     # Vendored dependencies (paramiko, invoke, cryptography, etc.)
```

### Registration Order

`state` -> `main_panel` -> `solver` -> `dynamics` -> `mesh_ops` -> `console` -> `zozo_contact_solver`

On register, `__init__.py` also:
1. Starts a `ReloadServer` (TCP port 8765) for hot-reload during development
2. Schedules a 1-second timer to clean up group UUIDs and normalize group names

Unregistration is reversed. MCP server cleanup (`cleanup_mcp_server()`) happens before component unregistration.

### Addon Namespace

All addon data lives under `scene.zozo_contact_solver`, accessed via `get_addon_data()` from `models/groups.py`. The namespace constant is `_ADDON_NAMESPACE = "zozo_contact_solver"`.

---

## Key Concepts

### Object Types and Material Models

| Type | Description | Default Model | Available Models |
|------|------------|---------------|-----------------|
| **SHELL** | Thin surfaces (cloth, fabric) | Baraff-Witkin | Baraff-Witkin, Stable NeoHookean, ARAP |
| **SOLID** | Volumetric bodies | ARAP | Stable NeoHookean, ARAP |
| **ROD** | 1D structures (ropes, wires) | ARAP | ARAP only |
| **STATIC** | Non-deforming collision objects | N/A | N/A |

### Dynamics Groups

Up to 32 groups (`object_group_0` through `object_group_31`, constant `N_MAX_GROUPS = 32`). Each group has:
- Object type and material model selection
- Material parameters: density, Young's modulus, Poisson ratio, friction, bend stiffness, anisotropic shrink (X/Y) for shells, uniform shrink for solids, strain limit, inflate pressure, stitch stiffness, initial velocity
- Contact parameters: contact_gap, contact_offset (absolute or ratio of bounding box diagonal)
- Assigned Blender mesh objects (with per-object inclusion toggle)
- Pin vertex groups, each with a list of operations (MOVE_BY, SPIN, SCALE, TORQUE, EMBEDDED_MOVE)
- Overlay visualization: color, wireframe, pin spheres, operation previews

Default overlay colors by type: SOLID = red (0.75,0,0), SHELL = green (0,0.75,0), ROD = yellow (0.75,0.75,0), STATIC = blue (0,0,0.75).

### Pin Operations

Each pin vertex group can have multiple operations:

| Operation | Parameters | Description |
|-----------|-----------|-------------|
| **EMBEDDED_MOVE** | (n/a) | Keyframe-based vertex movement; auto-added on first `MakePinKeyframe`. |
| **MOVE_BY** | delta (XYZ), frame_start, frame_end, transition | Translate vertices by delta over time range |
| **SPIN** | axis (XYZ), angular_velocity (deg/s), center mode (CENTROID/ABSOLUTE/MAX_TOWARDS/VERTEX), frame_start, frame_end, transition | Rotate around axis. Center mode determines the pivot point. |
| **SCALE** | factor, center mode (CENTROID/ABSOLUTE/MAX_TOWARDS/VERTEX), frame_start, frame_end, transition | Scale from center point. Center mode determines the pivot point. |
| **TORQUE** | magnitude (N*m), axis_component (PC1/PC2/PC3), torque_flip, frame_start, frame_end | Rotational force around PCA-computed axis |

**Center Modes** (for SPIN and SCALE):

| Mode | Description | Properties |
|------|-------------|------------|
| **CENTROID** | Compute center from vertex positions at runtime | (none) |
| **ABSOLUTE** | Use fixed coordinate | `spin_center`/`scale_center` (XYZ). "Pick from Selected" button sets from edit-mode selection. |
| **MAX_TOWARDS** | Centroid of vertices furthest in a direction | `spin_center_direction`/`scale_center_direction` (XYZ). `show_max_towards_spin`/`show_max_towards_scale` toggles overlay. |
| **VERTEX** | Single vertex picked in Edit Mode | `spin_center_vertex`/`scale_center_vertex` (int index). "Pick Vertex" button. `show_vertex_spin`/`show_vertex_scale` toggles overlay. |

Constraint: TORQUE cannot mix with MOVE_BY/SPIN/SCALE (can mix with EMBEDDED_MOVE).

### Invisible Colliders

Parametric collision boundaries that constrain simulation geometry without being rendered. Two types:

| Type | Parameters | Description |
|------|-----------|-------------|
| **WALL** | position (XYZ), normal (XYZ) | Infinite plane. Objects must stay on the normal-facing side. |
| **SPHERE** | position (XYZ), radius | Sphere boundary. Options: `invert` (collision on inside), `hemisphere` (bowl shape, top half open). |

Both support: `contact_gap`, `friction`, and animation keyframes (position changes, radius changes for spheres) with hold for step functions. Saved/loaded as part of scene profiles. Encoded into `param.pickle` under `"invisible_colliders"` key. The decoder creates `Wall`/`Sphere` objects via the frontend API before scene build.

### Simulation Workflow

1. **Connect** to remote solver (SSH/Docker/Local/Windows)
2. **Start server** on remote (`server.py --port <port>`)
3. **Transfer** mesh geometry and parameters (`encode_obj` + `encode_param`, pickled)
4. **Build** solver data structures
5. **Run** simulation (validates mesh hash first)
6. **Fetch** animation frames back to Blender (downloads `map.pickle`, `surface_map.pickle`, `vert_N.bin`)
7. Optionally **resume** or update parameters mid-simulation

### Coordinate System

Blender uses Z-up; the solver uses Y-up. Conversion: `(x, y, z)` -> `[x, z, -y]`.

Functions (all in `core/transform.py`, re-exported by `core/utils.py` and `core/encoder/__init__.py`): `_swap_axes(v)` for directions only, `_to_solver(v)` for positions (axis swap: `[x, z, -y]`), `_normalize_and_scale(direction, strength)`, `zup_to_yup()` returns 4x4 matrix `[[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]`, `world_matrix(obj)` = `zup_to_yup @ obj.matrix_world`, `inv_world_matrix(obj)`.

**Solver space convention:** The solver stores vertices in untranslated space (rotation+scale only), with object translation stored separately as displacement. All encoded absolute positions (pin animation, operation centers, centroids) must be in this untranslated space. Use `world_matrix().to_3x3().to_4x4()` for vertex-derived positions, or subtract `world_matrix().to_translation()` from `_to_solver()` results for global coordinates.

### Communication Protocol

- Protocol version: `0.02` (constant `PROTOCOL_VERSION`)
- Transport: TCP socket with chunked binary transfer (default chunk size 32KB)
- Headers: `TCMD` (text command query), `BDAT` (binary data, unused), `JSON` (JSON-prefixed binary)
- Data serialization: Python pickle for mesh/param data
- Status polling: `send_query(args)` sends flattened `--key value` text, receives JSON response
- Data send: JSON header + raw binary chunks, waits for `"OK\n"` confirmation
- Data receive: JSON header, then JSON metadata `{size: int}`, then binary chunks

### Connection Types

| Type | Enum Value | Transport | Requirements |
|------|-----------|-----------|-------------|
| Local | `LOCAL` | Direct filesystem | None |
| SSH | `CUSTOM` | paramiko SSHClient | paramiko |
| SSH Command | `COMMAND` | paramiko (parsed from command string) | paramiko |
| Docker | `DOCKER` | docker-py | docker |
| Docker over SSH | `DOCKER_SSH` | paramiko + docker exec | paramiko |
| Docker over SSH Command | `DOCKER_SSH_COMMAND` | paramiko (parsed) + docker exec | paramiko |
| Windows Native | `WIN_NATIVE` | subprocess (auto-detects Python/CUDA) | None |

---

## Module Details

### `core/` - Communication & Encoding

#### `client.py` - Central Communicator

**Singleton:** `communicator = Communicator()` (module-level instance)

The `Communicator` class manages all remote operations in a background daemon thread. It uses 5 domain-specific `RLock`s (alphabetical acquire order to prevent deadlocks): `animation`, `connection`, `data`, `status`, `task`.

**Connection methods** (all use internal `_queue_connection_task()` helper for lock management):
- `connect_ssh(host, port, username, key_path, path, container, server_port=9090, keepalive_interval=30)`
- `connect_docker(container, path, server_port=9090)`
- `connect_local(path, server_port=9090)`
- `connect_win_native(path, port)` - auto-detects dev vs bundle layout for Python/CUDA paths
- `disconnect()` - dispatches to type-specific disconnect

**Solver control:**
- `build()` - queries `{"request": "build"}`
- `run(context)` - clears fetched frames, queries `{"request": "start"}`
- `resume(context)` - queries `{"request": "resume"}`
- `fetch(context)` - downloads `map.pickle`, `surface_map.pickle`, then `vert_N.bin` per frame
- `terminate()` - queries `{"request": "terminate"}`, polls until not running
- `save_and_quit()` - queries `{"request": "save_and_quit"}`
- `abort()` - sets interrupt flag, clears animation frames

**Server lifecycle:**
- `start_server()` - creates shell script activating venv, runs `nohup python3 server.py --port <port>`, monitors `progress.log` for `SERVER_READY` marker (16s timeout)
- `stop_server()` - runs `pkill -f server.py`, polls until stopped

**Data transfer** (both use `open_server_channel()` from protocol.py for SSH/socket abstraction):
- `data_send(remote_path, data, message)` - local: direct file write; remote: socket with `socket_data_send()`
- `data_receive(remote_path, message)` - local: direct file read; remote: socket with `socket_data_receive()`

**Task processing:** Background daemon thread runs `_process_task()` in a loop. Supported task names: `connect_ssh`, `connect_docker`, `connect_local`, `connect_win_native`, `disconnect_*`, `start_server`, `stop_server`, `exec`, `query`, `build`, `run`, `resume`, `fetch`, `data_send`, `data_receive`.

**Status update (`_update_status`):** Maps server response status strings to `RemoteStatus` enum. Auto-fetches latest animation frame during simulation. Detects simulation completion, failure, and protocol version mismatch.

**Animation application:** `apply_animation()` (module-level) pops one frame at a time, writes vertex/CV data to PC2 files. Frame numbering: Blender frames 1..N map to remote frames 0..N-1 (PC2 index = blender_frame - 1). MESH_CACHE modifier uses `frame_start=1.0` and is moved to first modifier position. Curves use `frame_change_post` handler with `frame_idx = current - 1`. Encoder sends `frames = N-1` so remote produces `vert_0.bin..vert_(N-1).bin`. `frame_end` is set to the latest fetched Blender frame as frames arrive. Snap overlay reads PC2 files directly (no depsgraph access).

**File change monitoring:** Disconnects automatically if the Blender file path changes.

#### `protocol.py` - Wire Protocol

**Constants:** `PROTOCOL_VERSION = "0.02"`, `HEADER_TEXT_CMD = b"TCMD"`, `HEADER_BINARY_DATA = b"BDAT"`, `HEADER_JSON_DATA = b"JSON"`, `DEFAULT_CHUNK_SIZE = 32 * 1024`

**Functions:**
- `open_server_channel(connection)` - factory that returns a connected socket (local/docker) or SSH channel depending on connection type; used by `send_query`, `_data_send`, `_data_receive`
- `format_traffic(bytes_per_second) -> str` - formats as "B/s", "KB/s", or "MB/s"
- `exec_command(command, connection, shell=False, cwd=None) -> {exit_code, stdout[], stderr[]}` - dispatches to SSH channel / Docker exec_run / subprocess based on connection type
- `send_query(args, connection, project_name, chunk_size) -> (response_dict, server_running)` - sends `TCMD` + flattened args, reads JSON response via `open_server_channel`
- `socket_data_send(sock, request_data, data, chunk_size, progress_callback, interrupt_callback, bps_calculator)` - sends `JSON` header + binary chunks, waits for "OK"
- `socket_data_receive(sock, request_data, chunk_size, progress_callback, interrupt_callback, bps_calculator) -> bytes` - sends `JSON` header, reads metadata `{size}`, then binary chunks

#### `status.py` - State Enums & Dataclasses

**RemoteStatus enum values:** DISCONNECTED, CONNECTING, WAITING_FOR_DATA, WAITING_FOR_BUILD, SERVER_NOT_RUNNING, SERVER_LAUNCHING, STOPPING_SERVER, PROTOCOL_VERSION_MISMATCH, BUILDING, SIMULATION_IN_PROGRESS, FETCHING, APPLYING_DOWNLOADED_ANIM, DATA_SENDING, DATA_RECEIVING, EXECUTING_COMMAND, SAVING_IN_PROGRESS, READY, RESUMABLE, STARTING_SOLVER, SIMULATION_FAILED, ERROR, ABORTING, UNKNOWN.

Methods: `in_progress()` (7 states), `abortable()` (4 states), `ready()` (not in 4 error states), `icon` property (maps to Blender icon names).

**Dataclasses:**
- `ConnectionInfo` - type, current_directory, remote_root, instance, server_running, container, thread, server_thread, server_port
- `CommunicatorInfo` - status, message, error, server_error, response, progress (0-1), traffic
- `CommunicatorLocks` - task, status, connection, animation, data (all RLock)
- `AnimationData` - map (object->vertex mapping), frame (list of (frame_num, vertices)), surface_map, total_frames, applied_frames
- `BytesPerSecondCalculator` - sliding window (3s) throughput calculator

#### `transform.py` - Coordinate Transforms

All Blender (Z-up) to solver (Y-up) coordinate conversion functions, consolidated in one module:
- `_swap_axes(v) -> [x, z, -y]` - direction-only conversion (no scaling). Use for gravity, wind, spin axis, normals.
- `_to_solver(v) -> [x, z, -y]` - position conversion (axis swap only). Use for absolute coordinates, deltas.
- `_normalize_and_scale(direction, strength) -> list[float]` - normalize and scale a direction vector
- `zup_to_yup() -> Matrix` - 4x4 coordinate system transform matrix
- `world_matrix(obj) -> Matrix` - combined `zup_to_yup @ obj.matrix_world`
- `inv_world_matrix(obj) -> Matrix` - inverse of world_matrix

Re-exported by `core/utils.py` (matrix functions) and `core/encoder/__init__.py` (vector functions) for backward compatibility.

#### `encoder/` - Data Serialization

**`encode_obj(context) -> bytes`** (pickle): Per active group, per included object:
- ROD: edges array only
- Others: extract to NumPy, apply world transform, triangulate (faces + UV)
- Detects stitch edges (loose edges not in any face) -> `(Ind[#,4], W[#,2])` format
- STATIC objects: extracts per-frame animation (positions at each keyframe)
- Other objects: collects pin vertex indices

**`encode_param(context) -> bytes`** (pickle): Dict with keys:
- `scene`: dt, min-newton-steps, air-density, air-friction, gravity[3], wind[3], frames, fps, csrmat-max-nnz, isotropic-air-friction, auto-save, line-search-max-t, constraint-ghat, cg-max-iter, cg-tol, include-face-mass, disable-contact, inactive-momentum, stitch-stiffness
- `group`: list of (params_dict, object_names) per group. Params: model, density, young-mod, poiss-rat, friction, contact-gap, contact-offset, bend, shrink (solid), shrink-x, shrink-y, strain-limit, pressure, length-factor, velocity
- `pin_config`: {obj_name: {vertex_idx: {unpin_time, pull_strength, operations[], embedded_move_index, pin_anim, pin_group_id}}}
- `merge_pairs`: [(object_a, object_b, stitch_stiffness), ...]
- `dyn_param` (optional): dict mapping solver key to list of `(time_seconds, value_list, is_hold)` tuples. Keys: `"gravity"`, `"wind"`, `"air-density"`, `"air-friction"`, `"isotropic-air-friction"`. Only present when dynamic parameters with >1 keyframe exist. Encoded by `_encode_dyn_params()`. Vectors are coordinate-converted (Z-up to Y-up), wind is direction*strength combined. The decoder (`frontend/_decoder_.py`) reads this key and applies via `session.param.dyn(key).time(t).hold()` or `.change(v)`.
- `invisible_colliders` (optional): dict with `"walls"` and `"spheres"` lists. Each wall: `{position, normal, contact_gap, friction, keyframes: [{time, position}]}`. Each sphere: `{position, radius, hemisphere, invert, contact_gap, friction, keyframes: [{time, position, radius}]}`. Encoded by `_encode_invisible_colliders()`. The decoder creates `Wall`/`Sphere` objects via `scene.add.invisible.wall()` / `.sphere()` before `scene.build()`.

**Other functions:**
- `compute_mesh_hash(context) -> dict` - topology hash per group for validation
- `compute_average_edge_length(group) -> float`
- `compute_group_bounding_box_diagonal(group) -> float`
- `detect_stitch_edges(mesh) -> (Ind, W) | None`

#### `connection.py` - Connection Backends

- `connect_ssh(host, port, username, key_path, path, container, keepalive_interval, server_port, exec_fn) -> ConnectionInfo` - uses paramiko SSHClient with compress=True, validates Docker container if specified
- `connect_docker(container, path, server_port) -> ConnectionInfo` - uses docker.from_env(), starts container if not running
- `connect_local(path, server_port) -> ConnectionInfo`
- `connect_win_native(root, port) -> (ConnectionInfo, subprocess.Popen)` - auto-detects dev layout (`build-win-native/python/`) vs bundle layout (`root/python/`), sets PATH/PYTHONPATH/CUDA_PATH, launches subprocess
- `validate_remote_path(path, exec_fn)` - checks `server.py` exists at path

#### `async_op.py` - Modal Operator Base

**`AsyncOperator(Operator)`**: Base for timer-based modal operators.
- Abstract: `is_complete() -> bool`
- Optional: `on_complete(context)`, `on_timeout(context)`, `timeout: float = 60.0`
- `auto_redraw: bool = False` - when True, calls `redraw_all_areas()` on each timer tick (eliminates manual modal overrides for progress display)
- Timer interval: `get_timer_wait_time()` (0.25s)

**`Pipeline`**: Multi-stage pipeline with list of `(name, start_fn, is_complete_fn)` tuples. Auto-advances between stages.

#### `utils.py` - Utility Functions

- `get_category_name() -> "ZOZO's Contact Solver"`
- `get_timer_wait_time() -> 0.25`
- `redraw_all_areas(context)` - tags all screen areas for redraw
- `parse_vertex_index(data_path) -> int | None` - extracts vertex index from fcurve data path like `"vertices[42].co"`
- `_get_fcurves(action) -> list` - compatible with Blender 4.x (`action.fcurves`) and 5.0+ (`action.layers[].strips[].channelbags[].fcurves`)
- `get_vertices_in_group(obj, vg) -> list[int]`
- `set_linear_interpolation(action)` - sets all keyframe points to LINEAR
- `get_moving_vertex_indices(obj, exclude=None) -> list[int]` - vertices with animation keyframes
- `get_pin_vertex_indices(obj, context, frame=None) -> list[int]` - pinned vertices excluding pull pins and pins with movement operations; respects pin_duration
- `get_animation(obj, context) -> (frames, verts)` - per-frame world-space vertex positions for STATIC objects
- Re-exports from `transform.py`: `zup_to_yup`, `world_matrix`, `inv_world_matrix`

#### `animation.py` - Keyframe Management

- `save_pin_keyframes(context)` - saves all fcurves for pinned vertices to `state.saved_pin_keyframes` collection for restoration
- `clear_animation_data(context, move_to_frame=True)` - removes simulation keyframes while preserving pinned vertex animation; restores from saved keyframes

#### `profile.py` - TOML Profile System

Mapping dicts: `PROFILE_TYPE_MAP` (7 connection types), `_SSH_STATE_FIELDS` (11 fields), `_SCENE_PARAM_FIELDS` (21 fields), `_MATERIAL_PARAM_FIELDS` (24 fields), `_OP_FIELDS` (20 fields).

Functions: `load_profiles(path) -> dict`, `get_profile_names(path) -> list[str]`, `apply_profile(profile, ssh_state) -> bool`, `read_connection_profile(ssh_state) -> dict`, `apply_scene_profile(profile, state) -> bool`, `read_scene_profile(state) -> dict`, `apply_material_profile(profile, object_group) -> bool`, `read_material_profile(object_group, include_pins=False) -> dict`, `read_pin_operations(pin_item) -> dict`, `apply_pin_operations(profile, pin_item)`, `save_profile_entry(path, entry_name, data)`.

#### Other Core Files

- **`reload_server.py`**: `ReloadServer` class - TCP server on port 8765. Commands: `"reload"` (disables addon, invalidates modules from sys.modules, re-enables), `"execute"` (exec arbitrary Python), `"start_mcp"` (starts MCP server). Module functions: `start_reload_server(port)`, `stop_reload_server()`, `trigger_reload_now()`.
- **`module.py`**: `import_module(name)` - imports from lib/ directory, supports dotted names. `install_module(packages)` - async pip install to lib/ in background thread.
- **`ssh_config.py`**: `resolve_ssh_config(host, default_port=22) -> SSHConfigEntry` - parses `~/.ssh/config` with Include/glob support, first-match semantics.
- **`numpy_mesh_utils.py`**: `extract_mesh_to_numpy(mesh)`, `triangulate_numpy_mesh(verts, faces)`, `triangulate_uv_data(mesh, tri_faces)`. NumPy helpers used by `encoder/mesh.py` to triangulate tri/quad/n-gon meshes and match UVs.
- **`curve_rod.py`**: `sample_curve(obj, target_len, world_matrix) -> (verts, edges, params_data)` - samples Bezier/NURBS/Poly curves into rod vertices with stored parameterization for least-squares fitting. `apply_fit(spline, sim_pos, spline_meta)` - fits simulated positions back to control points.

---

### `ui/` - User Interface

#### `ui/state.py`, `ui/state_types.py`, `ui/object_group.py` - PropertyGroup Definitions

State is split across three files for maintainability:
- **`state_types.py`**: Collection-item PropertyGroups (`FetchedFrameItem`, `SavedPinKeyframePoint`, `SavedPinFCurve`, `SavedPinGroup`, `MergePairItem`, `DynParamKeyframe`, `DynParamItem`, `InvisibleColliderKeyframe`, `InvisibleColliderItem`, `AssignedObject`, `PinOperation`, `PinVertexGroupItem`)
- **`object_group.py`**: `ObjectGroup` PropertyGroup with all material/overlay/pin properties and profile callbacks
- **`state.py`**: `SSHState`, `State`, `SceneRoot` PropertyGroups, registration, and re-exports from the other two files for backward compatibility (all `from ..state import X` imports continue to work)

**SSHState PropertyGroup:**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `profile_path` | StringProperty (FILE_PATH) | "" | TOML connection profile file |
| `profile_selection` | EnumProperty (dynamic) | - | Selected profile from file |
| `host` | StringProperty | "" | SSH host |
| `port` | IntProperty | 22 | SSH port |
| `username` | StringProperty | "" | SSH username |
| `key_path` | StringProperty (FILE_PATH) | "~/.ssh/id_ed25519" or "~/.ssh/id_rsa" | SSH key |
| `docker_path` | StringProperty | "/root/ppf-contact-solver" | Container working path |
| `local_path` | StringProperty | "~/ppf-contact-solver" | Local solver path |
| `server_type` | EnumProperty | "CUSTOM" | LOCAL/CUSTOM/COMMAND/DOCKER/DOCKER_SSH/DOCKER_SSH_COMMAND/WIN_NATIVE |
| `command` | StringProperty | "ssh -p xxx root@zzz" | SSH command string |
| `container` | StringProperty | "ppf-dev" | Docker container name |
| `ssh_remote_path` | StringProperty | "/root/ppf-contact-solver" | Remote solver path |
| `win_native_path` | StringProperty (DIR_PATH) | "" | Windows solver root |
| `docker_port` | IntProperty | 9090 | Server port (min 1024, max 65535) |

**State PropertyGroup (scene-level simulation parameters):**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `step_size` | FloatProperty | 0.001 | Simulation step size (0.001-0.01) |
| `min_newton_steps` | IntProperty | 1 | Minimum Newton iterations (1-64) |
| `air_density` | FloatProperty | 0.001 | Air density kg/m^3 (0-0.01) |
| `air_friction` | FloatProperty | 0.2 | Air friction ratio (0-1) |
| `gravity_3d` | FloatVectorProperty (XYZ) | (0, 0, -9.8) | Gravity m/s^2 |
| `wind_direction` | FloatVectorProperty (XYZ) | (0, 0, 0) | Wind direction |
| `wind_strength` | FloatProperty | 0.0 | Wind strength m/s (0-1000) |
| `frame_count` | IntProperty | 180 | Simulation frames (min 10) |
| `frame_rate` | IntProperty | 60 | FPS (min 24) |
| `inactive_momentum_frames` | IntProperty | 0 | Frames with inactive momentum (0-600) |
| `contact_nnz` | IntProperty | 100000000 | Contact matrix non-zero entries |
| `line_search_max_t` | FloatProperty | 1.25 | CCD line search factor (0.1-10) |
| `constraint_ghat` | FloatProperty | 0.001 | Barrier gap distance (0.0001-0.1) |
| `cg_max_iter` | IntProperty | 10000 | PCG max iterations (100-100000) |
| `cg_tol` | FloatProperty | 0.001 | PCG tolerance (0.0001-0.1) |
| `include_face_mass` | BoolProperty | False | Include shell mass for solids |
| `disable_contact` | BoolProperty | False | Disable contact detection |
| `vertex_air_damp` | FloatProperty | 0.0 | Vertex air damping (0-1) |
| `auto_save` | BoolProperty | False | Auto-save simulation state |
| `auto_save_interval` | IntProperty | 10 | Auto-save interval (min 1) |
| `project_name` | StringProperty | "unnamed" | Project name |
| `mcp_port` | IntProperty | 9633 | MCP server port |
| `reload_port` | IntProperty | 8765 | Reload server port |
| `mesh_hash_json` | StringProperty | "{}" | Mesh topology hash JSON |
| `mesh_hash_validated` | BoolProperty | False | Hash validated this session |
| `use_frame_rate_in_output` | BoolProperty | False | Use Blender render FPS instead of frame_rate property |

Collections: `fetched_frame` (FetchedFrameItem), `saved_pin_keyframes` (SavedPinGroup), `merge_pairs` (MergePairItem), `dyn_params` (DynParamItem), `invisible_colliders` (InvisibleColliderItem).

**ObjectGroup PropertyGroup (per-group material and config):**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `active` | BoolProperty | False | Group is active |
| `name` | StringProperty | "" | Group name (empty shows "Group N") |
| `uuid` | StringProperty | "" | Unique identifier |
| `object_type` | EnumProperty | "SOLID" | SOLID/SHELL/ROD/STATIC |
| `solid_model` | EnumProperty | "ARAP" | STABLE_NEOHOOKEAN/ARAP |
| `shell_model` | EnumProperty | "BARAFF_WITKIN" | STABLE_NEOHOOKEAN/ARAP/BARAFF_WITKIN |
| `rod_model` | EnumProperty | "ARAP" | ARAP only |
| `solid_density` | FloatProperty | 1000.0 | kg/m^3 |
| `shell_density` | FloatProperty | 1.0 | kg/m^2 |
| `rod_density` | FloatProperty | 1.0 | kg/m |
| `solid_young_modulus` | FloatProperty | 500.0 | Pa (0-10M) |
| `shell_young_modulus` | FloatProperty | 1000.0 | Pa (0-10M) |
| `rod_young_modulus` | FloatProperty | 10000.0 | Pa (0-10M) |
| `solid_poisson_ratio` | FloatProperty | 0.35 | 0-0.4999 |
| `shell_poisson_ratio` | FloatProperty | 0.35 | 0-0.4999 |
| `friction` | FloatProperty | 0.5 | 0-1 |
| `contact_gap` | FloatProperty | 0.001 | Absolute gap |
| `contact_offset` | FloatProperty | 0.0 | Absolute offset |
| `use_group_bounding_box_diagonal` | BoolProperty | True | Use ratio-based contact |
| `contact_gap_rat` | FloatProperty | 0.001 | Ratio of bbox diagonal |
| `contact_offset_rat` | FloatProperty | 0.0 | Ratio of bbox diagonal |
| `enable_strain_limit` | BoolProperty | False | Shell strain limit |
| `strain_limit` | FloatProperty | 0.05 | Shell strain limit value |
| `enable_inflate` | BoolProperty | False | Shell inflation pressure |
| `inflate_pressure` | FloatProperty | 0.0 | Pressure along face normals (Pa) |
| `bend` | FloatProperty | 100.0 | Bend stiffness (0-100) |
| `shrink_x` | FloatProperty | 1.0 | Anisotropic scale factor X (min 0.1; <1 shrinks, >1 extends) |
| `shrink_y` | FloatProperty | 1.0 | Anisotropic scale factor Y (min 0.1; <1 shrinks, >1 extends) |
| `shrink` | FloatProperty | 1.0 | Uniform scale factor for solids (min 0.1; <1 shrinks, >1 extends) |
| `stitch_stiffness` | FloatProperty | 1.0 | Stitch constraint stiffness |
| `color` | FloatVectorProperty (COLOR, 4) | per-type default | RGBA overlay color |

Collections: `assigned_objects` (AssignedObject with name + included toggle), `pin_vertex_groups` (PinVertexGroupItem with operations CollectionProperty).

**PinVertexGroupItem:** `name` (format `[ObjectName][VertexGroupName]`), `included`, `use_pin_duration`, `pin_duration` (frames), `use_pull`, `pull_strength`, `operations` (CollectionProperty of PinOperation), `operations_index`.

**PinOperation:** `op_type` (EMBEDDED_MOVE/MOVE_BY/SPIN/SCALE/TORQUE), `delta` (XYZ), `spin_axis` (XYZ), `spin_angular_velocity` (deg/s, default 360), `spin_center` (XYZ), `spin_center_mode` (CENTROID/ABSOLUTE/MAX_TOWARDS/VERTEX), `spin_center_vertex` (int, default -1), `spin_center_direction` (XYZ), `show_max_towards_spin` (bool), `show_vertex_spin` (bool), `scale_factor`, `scale_center` (XYZ), `scale_center_mode` (CENTROID/ABSOLUTE/MAX_TOWARDS/VERTEX), `scale_center_vertex` (int, default -1), `scale_center_direction` (XYZ), `show_max_towards_scale` (bool), `show_vertex_scale` (bool), `torque_axis_component` (PC1/PC2/PC3), `torque_magnitude` (N*m), `torque_flip`, `frame_start`, `frame_end`, `transition` (LINEAR/SMOOTH), `show_overlay`.

**MergePairItem:** `object_a`, `object_b`, `cross_stitch_json` (HIDDEN, stores explicit stitch anchor data), `stitch_stiffness` (default 1.0), `show_stitch` (bool, default True).

**DynParamKeyframe:** `frame` (int, min 1), `gravity_value` (XYZ, default (0,0,-9.8)), `wind_direction_value` (XYZ), `wind_strength_value` (float), `scalar_value` (float), `use_hold` (bool; when true, holds previous keyframe value, creating a step function).

**DynParamItem:** `param_type` (enum: GRAVITY/WIND/AIR_DENSITY/AIR_FRICTION/VERTEX_AIR_DAMP), `keyframes` (CollectionProperty of DynParamKeyframe), `keyframes_index`. First keyframe (frame 1) always exists and reads from global State params at encode time.

**InvisibleColliderKeyframe:** `frame` (int, min 1), `position` (XYZ), `radius` (float, sphere only), `use_hold` (bool; holds previous value).

**InvisibleColliderItem:** `collider_type` (WALL/SPHERE), `name` (auto-generated), `position` (XYZ), `normal` (XYZ, wall only), `radius` (float, sphere only), `hemisphere` (bool), `invert` (bool), `contact_gap` (float), `friction` (float), `keyframes` (CollectionProperty of InvisibleColliderKeyframe). First keyframe reads from base properties at encode time.

**AddonData (root):** `state` (State), `ssh_state` (SSHState), `object_group_0` through `object_group_31` (ObjectGroup).

#### `ui/main_panel.py` - Main Panel

Panel `MAIN_PT_RemotePanel` in VIEW_3D sidebar. Displays:
- Module installation prompts (Paramiko for SSH, Docker-Py for Docker)
- Profile management row (Open/Clear/Reload/Save)
- Server type selection with conditional property display per type
- Project name, docker_port (if applicable)
- Connect/Disconnect and StartServer/StopServer buttons
- Status display with progress bar, traffic, error messages
- UpdateStatus, ShowConsole, debug_mode toggle
- Hardware info and statistics collapsibles (when connected/running)
- MCP Settings (start/stop, port), JupyterLab (export/open/delete)
- Debug section: shell commands, data transfer tests, git operations, reload server

#### `ui/solver.py` - Solver Panel & Operators

Defines `TransferRequestMixin` (shared by `SOLVER_OT_Transfer` and `DEBUG_OT_TransferWithoutBuild`) with `request_data_send`, `request_param_send`, `request_delete` methods. Both `request_data_send` and `request_param_send` catch `ValueError` from encoding and surface it in the Blender UI via `self.report({"ERROR"}, ...)` and `com.error`.

Panel `SSH_PT_SolverPanel`. Key operators:

| Operator | bl_idname | Timeout | Description |
|----------|-----------|---------|-------------|
| Transfer | `solver.transfer` | 300s | Multi-stage: delete existing -> send mesh -> send params -> build |
| Run | `solver.run` | 86400s (24h) | Validates mesh hash, clears animation, starts simulation |
| Resume | `solver.resume` | 86400s | Resumes paused simulation |
| UpdateParams | `solver.update_params` | 120s | Re-sends parameters and rebuilds |
| ClearAnimation | `solver.clear_animation` | - | Clears simulation keyframes, preserves pin keyframes |
| FetchData | `solver.fetch_remote_data` | 600s | Downloads animation, validates mesh hash |
| DeleteRemoteData | `solver.delete_remote_data` | 60s | Queries `{"request": "delete"}` |

**Mesh hash validation:** Before Run and Fetch, `compute_mesh_hash()` is compared against stored hash to detect topology changes since last transfer.

#### `ui/connection_ops.py` - Connection Operators

| Operator | bl_idname | Description |
|----------|-----------|-------------|
| Connect | `ssh.run_command` | Modal operator, stays alive for connection lifetime. Parses COMMAND type for host/port/username/key_path from SSH command string. Timeout 60s before connection. |
| Disconnect | `ssh.disconnect` | Calls `com.disconnect()` |
| Abort | `ssh.abort` | Calls `com.abort()` |
| StartServer | `ssh.start_server` | AsyncOperator, timeout 60s |
| StopServer | `ssh.stop_server` | AsyncOperator, timeout 60s |
| OpenProfile | `ssh.open_profile` | File browser for TOML |
| ClearProfile | `ssh.clear_profile` | Clears loaded profile |
| ReloadProfile | `ssh.reload_profile` | Re-applies current profile |
| SaveProfile | `ssh.save_profile` | Saves current settings to TOML |

#### `ui/solver_control_ops.py`

- `solver.show_console` - opens console window
- `solver.terminate` - terminates simulation
- `solver.save_quit` - save and quit gracefully
- `solver.update_status` - refreshes server status

#### `ui/debug_ops.py`

- `debug.execute_server` (AsyncOperator, 60s) - runs server script
- `debug.execute_shell` - shell command on remote
- `debug.data_send` / `debug.data_receive` (AsyncOperator, 300s) - data transfer tests
- `debug.git_pull` / `debug.git_pull_local` - git operations
- `debug.compile` - `cargo build --release` on remote
- `debug.delete_log` - delete log file
- `wm.open_github_link` - opens GitHub in browser

#### `ui/install_ops.py`

- `ssh.install_paramiko` (AsyncOperator, 120s) - installs paramiko to lib/
- `ssh.install_docker` (AsyncOperator, 120s) - installs docker-py to lib/

#### `ui/jupyter_ops.py`

- `solver.jupyter_export` - creates notebook via Jupyter REST API (GET /lab for XSRF token, PUT /api/contents/)
- `solver.jupyter_open` - opens notebook URL in browser
- `solver.jupyter_delete` - deletes notebook via DELETE /api/contents/

#### `ui/mcp_ops.py`

- `mcp.start_server` - starts MCP server with automatic port fallback (tries port to port+9)
- `mcp.stop_server` - stops MCP server

#### `ui/addon_ops.py`

- `addon.start_reload_server` / `addon.stop_reload_server` / `addon.trigger_reload`

#### `ui/console.py`

Registers persistent Blender timer that calls `console.process_messages()` and `process_mcp_tasks()` at `get_timer_wait_time()` interval.

---

### `ui/dynamics/` - Scene & Group Configuration

#### `dynamics/panels.py` - Main Configuration Panels

**Panel `MAIN_PT_SceneConfiguration`** (bl_idname `SSH_PT_ObjectGroupsManager`):

All parameters shown directly (no outer collapsible box):
1. **Profile row**: Open/Clear/Reload/Save scene profile buttons
2. **Basic params**: frame_rate, frame_count, step_size, min_newton_steps, air_density, air_friction, gravity_3d (with preview toggle), inactive_momentum_frames (enabled only when SHELL groups exist)
3. **Wind** (collapsible, FORCE_WIND icon): wind_direction (with preview toggle), wind_strength
4. **Advanced Params** (collapsible, PREFERENCES icon): contact_nnz, vertex_air_damp, auto_save/interval, line_search_max_t, constraint_ghat, cg_max_iter, cg_tol, include_face_mass, disable_contact
5. **Dynamic Parameters** (collapsible, TIME icon): UIList of enabled dynamic params with Add dropdown/Remove. Keyframe UIList with type-specific editor. Hold checkbox for step functions. First keyframe reads from global values.
6. **Invisible Colliders** (collapsible, GHOST_ENABLED icon): UIList of walls/spheres with Add dropdown/Remove. Properties box (position, normal/radius, invert, hemisphere, contact_gap, friction). Keyframe UIList for animated colliders with Hold support. Saved/loaded as part of scene profile.

**Panel `DYNAMICS_PT_Groups`**: Standalone panel for dynamics group management. Create/DeleteAll buttons, then per-group boxes:
- Header: name, type icon (MESH_CUBE=SOLID, SURFACE_DATA=SHELL, CURVE_DATA=ROD, FREEZE=STATIC), duplicate button
- Group settings: name, object_type, overlay color, assigned objects list, add/remove buttons
- Bake buttons: Bake Animation, Bake Single Frame
- Pins section: vertex group dropdown, add/remove/create, pin list, overlay controls, pin item properties (duration, pull), copy/paste operations, operation list with type-specific parameter panels
- Stats: per-object vertex/face counts
- Material Params: copy/paste, profile management, type-specific parameter display

**Panel `SNAPMERGE_PT_SnapAndMerge`** (default closed): Object A/B dropdowns with snap button, merge pairs list with remove button. Stitch stiffness shown only for pairs involving a SOLID object (shell+solid, rod+solid). Sheet-sheet and rod-rod pairs merge vertices exactly without stiffness. Missing frames warning shown below Clear Animation when remote has unfetched frames (hidden during simulation).

#### `dynamics/utils.py` - Shared Dynamics Helpers

- `get_group_from_index(scene, group_index) -> ObjectGroup | None` - returns active group at slot index, or None
- `reset_object_display(obj)` - resets object color to white and disables wireframe overlays
- `cleanup_pin_vertex_groups_for_object(group, object_uuid)` - removes pin references for a specific object (by UUID)

#### `dynamics/group_ops.py` - Group Operators

| Operator | bl_idname | Description |
|----------|-----------|-------------|
| CreateGroup | `object.create_group` | Finds available slot, resets to defaults, generates UUID |
| DeleteGroup | `object.delete_group` | Resets object colors to white, calls reset_to_defaults() |
| DuplicateGroup | `object.duplicate_group` | Copies material params only (not objects/pins), increments name |
| DeleteAllGroups | `object.delete_all_groups` | Iterates all 32 slots, resets each |
| AddObjectsToGroup | `object.add_objects_to_group` | Validates MESH type, prevents duplicate assignment, sets wireframe |
| RemoveObjectFromGroup | `object.remove_object_from_group` | Resets color, cleans up pin references |

**Persistent handler:** `_cleanup_deleted_objects` runs on `depsgraph_update_post` to remove stale references from assigned_objects, pin_vertex_groups, and merge_pairs when objects are deleted.

#### `dynamics/dyn_param_ops.py` - Dynamic Parameter Operators

| Operator | bl_idname | Description |
|----------|-----------|-------------|
| AddDynParam | `scene.add_dyn_param` | Adds dynamic parameter via enum dropdown (GRAVITY/WIND/AIR_DENSITY/AIR_FRICTION/VERTEX_AIR_DAMP). Rejects duplicates. Creates initial frame-1 keyframe. |
| RemoveDynParam | `scene.remove_dyn_param` | Removes selected dynamic parameter |
| AddDynParamKeyframe | `scene.add_dyn_param_keyframe` | Adds keyframe at current scene frame. Rejects duplicate frames. Initializes values from global params. Sorts by frame. |
| RemoveDynParamKeyframe | `scene.remove_dyn_param_keyframe` | Removes selected keyframe. Cannot remove initial (index 0) keyframe. |

#### `dynamics/invisible_collider_ops.py` - Invisible Collider Operators

| Operator | bl_idname | Description |
|----------|-----------|-------------|
| AddInvisibleCollider | `scene.add_invisible_collider` | Adds wall or sphere via enum dropdown. Auto-names ("Wall 1", "Sphere 2"). Creates initial frame-1 keyframe. |
| RemoveInvisibleCollider | `scene.remove_invisible_collider` | Removes selected collider |
| AddColliderKeyframe | `scene.add_collider_keyframe` | Adds keyframe at current scene frame. Copies base position/radius. Sorts by frame. |
| RemoveColliderKeyframe | `scene.remove_collider_keyframe` | Removes selected keyframe. Cannot remove initial (index 0). |

#### `dynamics/pin_ops.py` - Pin Operators

| Operator | bl_idname | Description |
|----------|-----------|-------------|
| CreatePinVertexGroup | `object.create_pin_vertex_group` | Creates vertex group from EDIT mode selection, prompts for name |
| AddPinVertexGroup | `object.add_pin_vertex_group` | Adds existing vertex group from dropdown |
| RemovePinVertexGroup | `object.remove_pin_vertex_group` | Removes pin from list |
| SelectPinVertices | `object.select_pin_vertices` | Selects pin vertices in EDIT mode |
| DeselectPinVertices | `object.deselect_pin_vertices` | Deselects pin vertices |
| MakePinKeyframe | `object.make_pin_keyframe` | Keyframes vertex positions, auto-adds EMBEDDED_MOVE operation, saves pin keyframes |
| DeletePinKeyframes | `object.delete_pin_keyframes` | Removes matching fcurves, removes EMBEDDED_MOVE operation |
| AddPinOperation | `object.add_pin_operation` | Validates TORQUE exclusivity, inserts at head of list |
| RemovePinOperation | `object.remove_pin_operation` | Prevents removal of EMBEDDED_MOVE (use DeletePinKeyframes) |
| MovePinOperation | `object.move_pin_operation` | Reorders operations (direction: -1=up, 1=down) |
| PickCenterFromSelected | `object.pick_center_from_selected` | Sets ABSOLUTE center from centroid of selected vertices in Edit Mode (for SPIN or SCALE) |
| PickVertexCenter | `object.pick_vertex_center` | Sets VERTEX center from a single selected vertex in Edit Mode (for SPIN or SCALE) |

#### `dynamics/overlay.py`, `overlay_geometry.py`, `overlay_labels.py` - 3D Viewport Overlays

Split across three files:
- **`overlay.py`**: Cache management (`_overlay_cache`), draw callback dispatch, `apply_object_overlays()`, registration
- **`overlay_geometry.py`**: All geometry generation (`_generate_*` primitives) and GPU batch building (`_build_*_batches`), `DirectionPreviewManager`
- **`overlay_labels.py`**: Text/label rendering via `blf`

Registered as POST_VIEW and POST_PIXEL draw handlers.

**Visualizations:**
- **Rod edges**: Cross-shaped thick lines via `_line_to_tris()` for consistent thickness
- **Pin vertices**: Point rendering with per-vertex color (blue if animated/has operations, white otherwise) and configurable size
- **Direction previews**: Gravity and wind arrows with filled+wireframe sphere, scaled to 0.08 of view distance
- **Operation overlays** (per operation with `show_overlay=True`):
  - SPIN: Circle around axis + rotation arc (270 deg) with angular velocity label. Center resolved from mode (CENTROID/ABSOLUTE/MAX_TOWARDS/VERTEX).
  - MOVE_BY: Per-vertex arrows from current to displaced position
  - SCALE: Per-vertex arrows from current to scaled position. Center resolved from mode.
  - TORQUE: PCA axis circle + full rotation arc with magnitude label. Computes PCA in solver space (Y-up), uses hint vertex for axis orientation.
- **Max Towards overlay** (toggled via `show_max_towards_spin`/`show_max_towards_scale`): Highlights vertices furthest in a direction (yellow points) with direction arrow from centroid, view-distance scaled.
- **Vertex center overlay** (toggled via `show_vertex_spin`/`show_vertex_scale`): Highlights the single picked center vertex (green point) with "Center (vN)" label.

**Cache system**: Module-level `_overlay_cache` with version tracking. Rod batches rebuild on version/frame change. Pin data always rebuilds (tracks transforms). Operation batches always rebuild.

**`apply_object_overlays()`**: Resets all mesh colors to white, then applies group colors for groups with `show_overlay_color`. Ensures solid shading `color_type = "OBJECT"`.

#### `dynamics/profile_ops.py` - Profile Operators

16 operators for scene, material, and pin profile management (Open/Clear/Reload/Save for each). Plus copy/paste for material params and pin operations using module-level clipboard dicts (`_material_clipboard`, `_pin_ops_clipboard`).

#### `dynamics/ui_lists.py` - Custom UIList Renderers

- `OBJECT_UL_AssignedObjectsList` - type-aware icons (MESH_CUBE, OUTLINER_OB_SURFACE, VIEW_ORTHO, OBJECT_ORIGIN), inclusion toggle, missing object detection
- `OBJECT_UL_PinVertexGroupsList` - displays `[ObjectName][VGName]`, GROUP_VERTEX icon, missing detection
- `OBJECT_UL_PinOperationsList` - type-specific labels and icons (KEYFRAME, ORIENTATION_LOCAL, DRIVER_ROTATIONAL_DIFFERENCE, FULLSCREEN_ENTER, FORCE_MAGNETIC), overlay visibility toggle
- `OBJECT_UL_MergePairsList` - bidirectional display `A <-> B`, AUTOMERGE_ON icon
- `SCENE_UL_DynParamsList` - param name + keyframe count, type-specific icons (FORCE_FORCE, FORCE_WIND, MOD_FLUID, FORCE_DRAG, MOD_SMOOTH)
- `SCENE_UL_DynParamKeyframesList` - frame number, "(Initial)" label + DECORATE_KEYFRAME icon for first entry, KEYFRAME icon for others
- `SCENE_UL_InvisibleCollidersList` - type icon (MESH_PLANE for wall, MESH_UVSPHERE for sphere) + name + flags ("Inv", "Hemi")
- `SCENE_UL_ColliderKeyframesList` - frame number, "(Initial)" for first entry

---

### `ops/` - Operator System

Three-layer architecture:

#### `ops/zozo_contact_solver.py` - Dynamic Operator Generation

At registration time:
1. Imports `blender_handlers` and `solver_handlers` to trigger handler registration
2. Gets handler registry from `get_handler_registry()`
3. For each handler, calls `_build_operator_class(name, handler_info)`:
   - Extracts parameter info via `inspect.signature()` and `get_type_hints()`
   - Creates `bl_idname = "zozo_contact_solver.{name}"`, `bl_label`, `bl_description` from schema
   - Generates Blender properties: int->IntProperty, float->FloatProperty, bool->BoolProperty, others->StringProperty (JSON-encoded)
   - `execute()` method extracts properties, JSON-decodes complex types, calls handler wrapper
4. Registers `state_ops` operators
5. Injects `api` module into `sys.modules["zozo_contact_solver"]` so `from zozo_contact_solver import solver` works

#### `ops/api.py` - High-Level Python API

**`_Solver` class** (top-level, exposed as `solver`):
- `param` (class var) -> `_SceneProxy` for `solver.param.frame_count = 180`
- `create_group(name="", type="SOLID") -> _Group`
- `get_group(uuid) -> _Group`, `get_groups() -> list[_Group]`
- `delete_all_groups()`, `clear()` (comprehensive reset including scene defaults)
- `snap(object_a, object_b)`, `add_merge_pair(a, b)`, `remove_merge_pair(a, b)`, `get_merge_pairs()`, `clear_merge_pairs()`
- `add_wall(position, normal) -> _InvisibleWallBuilder`, `add_sphere(position, radius) -> _InvisibleSphereBuilder`
- `get_invisible_colliders()`, `clear_invisible_colliders()`
- `__getattr__(name)` -> delegates to `bpy.ops.zozo_contact_solver.<name>()`

**`_Group` class:**
- `uuid` property, `param` -> `_ParamProxy`
- `set_overlay_color(r, g, b, a=1.0)`, `add(*object_names)`, `remove(object_name)`
- `create_pin(object_name, vertex_group_name) -> _Pin`, `get_pins() -> list[_Pin]`
- `clear_keyframes()`, `delete()`

**`_Pin` class:**
- `move(delta, frame)` - auto-keyframes, adds EMBEDDED_MOVE, respects unpin_frame
- `pull(strength)`, `unpin(frame)`, `move_by(delta, start, end, transition)`
- `spin(axis, angular_velocity, center, center_mode, center_direction, center_vertex, start, end, transition)` - center_mode auto-inferred from args: CENTROID (no center), ABSOLUTE (center given), MAX_TOWARDS (center_direction given), VERTEX (center_vertex given)
- `scale(factor, center, center_mode, center_direction, center_vertex, start, end, transition)` - same center_mode inference as spin
- `torque(magnitude, axis_component, flip=False, frame_start=1, frame_end=60)` - PC1/PC2/PC3, with flip and time range
- `clear_keyframes()`, `delete()`
- All methods are chainable (return self)

**`_ParamProxy`**: Attribute proxy with whitelist of 23 properties (solid/shell density, young_modulus, poisson_ratio, friction, contact_gap/offset/rat, strain_limit, enable_inflate, inflate_pressure, bend, shrink, stitch_stiffness, etc.)

**`_SceneProxy`**: Attribute proxy for `addon_data.state` and `addon_data.ssh_state`. Alias: `gravity` -> `gravity_3d`. Method: `dyn(key) -> _DynParamBuilder` for dynamic parameter keyframing.

**`_DynParamBuilder`**: Fluent chainable builder for dynamic parameter keyframes. Mirrors the frontend `session.param.dyn()` API but uses **frames** instead of seconds. Key mapping: `"gravity"` -> GRAVITY, `"wind"` -> WIND, `"air_density"` -> AIR_DENSITY, `"air_friction"` -> AIR_FRICTION, `"vertex_air_damp"` -> VERTEX_AIR_DAMP.
- `time(frame)` - advance frame cursor (must be strictly increasing)
- `hold()` - add keyframe with `use_hold=True` (step function)
- `change(value, strength=None)` - add keyframe with value. For WIND, `strength` kwarg sets wind strength.
- `clear()` - remove the dynamic parameter entirely
- All methods return self for chaining.

**`_InvisibleWallBuilder`**: Chainable builder for invisible wall colliders. Created via `solver.add_wall(position, normal)`.
- `time(frame)`, `hold()`, `move_to(position)`, `move_by(delta)` - keyframe animation
- `param` -> `_ColliderParamProxy` for `contact_gap`, `friction`
- `delete()` - remove the collider

**`_InvisibleSphereBuilder`**: Chainable builder for invisible sphere colliders. Created via `solver.add_sphere(position, radius)`.
- `invert()`, `hemisphere()` - toggle collision modes
- `time(frame)`, `hold()`, `move_to(position)`, `move_by(delta)`, `radius(r)`, `transform_to(position, radius)` - keyframe animation
- `param` -> `_ColliderParamProxy` for `contact_gap`, `friction`
- `delete()` - remove the collider

#### `ops/state_ops.py`

- `zozo_contact_solver.set` operator: Sets any property by name with auto type coercion via `bl_rna.properties` introspection. Props: `key`, `value` (string), `group_uuid` (optional). Aliases: `{"gravity": "gravity_3d"}`.

---

### `models/` - Data Access & Defaults

#### `models/groups.py`

- `N_MAX_GROUPS = 32`
- `_ADDON_NAMESPACE = "zozo_contact_solver"`
- `get_addon_data(scene=None)` - returns `scene.zozo_contact_solver`
- `invalidate_overlays()` - bumps `overlay_version`, redraws all VIEW_3D areas
- `OBJECT_GROUP_DEFAULTS` dict (43 keys): all default values for ObjectGroup properties
- `get_object_type(type) -> RGBA tuple` (SOLID=red, SHELL=green, ROD=yellow, STATIC=blue)
- `get_vertex_group_items(self, _)` - callback for vertex group enum, format `[ObjectName][VGName]`
- `decode_vertex_group_identifier(identifier) -> (object_name, vg_name)` - regex `\[(.*)]\[(.*)]`
- `iterate_object_groups(scene)` / `iterate_active_object_groups(scene)`
- `get_group_by_uuid(scene, uuid)` / `get_active_group_by_uuid(scene, uuid)`
- `find_available_group_slot(scene) -> int | None`
- `assign_display_indices(scene)` - sequential numbering of active groups
- `pair_supports_cross_stitch(type_a, type_b) -> bool` - checks if two group types support cross-object stitching

#### `models/collection_utils.py`

Shared utilities for Blender PropertyGroup collections:
- `sort_keyframes_by_frame(keyframes) -> int` - insertion sort on collection by `.frame`, returns final index
- `validate_no_duplicate_frame(keyframes, frame)` - raises ValueError on duplicate
- `safe_update_index(current_index, new_length) -> int` - clamps index after collection resize: `min(current_index, max(0, new_length - 1))`
- `generate_unique_name(prefix, existing_names) -> str` - generates "Prefix N" avoiding collisions

#### `models/defaults.py`

```python
DEFAULT_SERVER_PORT = 9090
DEFAULT_MCP_PORT = 9633
DEFAULT_RELOAD_PORT = 8765
```

#### `models/console.py`

`Console` singleton with thread-safe message queue. Methods: `get_or_create()` (Blender text object), `show(last_lines=10)` (opens TEXT_EDITOR window), `write(message, timestamp=True)`, `process_messages()` (flushes to text object and optional log file, trims to `max_console_lines`).

#### `models/git_utils.py`

`get_git_branch() -> str` - reads `.git/branch_name.txt` first, falls back to `git branch --show-current`, returns "unknown" on failure.

---

### `mesh_ops/` - Mesh Operations

#### `mesh_ops/snap_ops.py`

**Operator `object.snap_to_vertices`**: KDTree-based vertex snapping.
1. Builds KDTree from target object B's vertices (world space)
2. For each vertex in object A, finds nearest B vertex via `kd.find()`
3. Computes translation = closest_B - closest_A
4. Applies gap along approach direction (closest_A - closest_B, normalized). Gap = `base_gap + max(base_gap, float32_margin)` where `base_gap = max(gap_a, gap_b) + offset_a + offset_b`. No gap for SHELL-SHELL and ROD-ROD (vertices merge exactly).
5. Translates object A in world space (handles parenting)
6. Builds cross-stitch pairs with threshold = `2 * h` where `h = total_gap` (gap cases) or `h = max(gap_a, gap_b)` (no-gap cases). Merge pair encoder picks target vertex with highest barycentric weight.
7. Registers merge pair in state with cross_stitch_json

Duplicate object names across active groups are detected at encode time and raise `ValueError`.

#### `mesh_ops/merge_ops.py`

**`cleanup_stale_merge_pairs(scene)`**: Removes pairs where either object is not in any active group.

**`object.remove_merge_pair`**: Removes selected merge pair from state.

---

### `mcp/` - Model Context Protocol Server

HTTP server exposing addon functionality as MCP tools for AI/automation integration.

#### `mcp/decorators.py` - Handler Registration System

**Decorators:** `@mcp_handler` (general), `@connection_handler`, `@group_handler`, `@simulation_handler`, `@debug_handler`, `@remote_handler`.

Each decorator:
1. Parses docstring for description and parameter info
2. Generates JSON Schema from type hints (`get_json_schema_type()`)
3. Creates wrapper that validates args (`validate_and_convert_args()`), calls function, formats response
4. Registers in global `_handler_registry` with `{func, schema, category}`

**Registry access:** `get_handler_registry()`, `get_handlers_by_category(category)`.

**Custom exceptions:** `MCPError`, `ValidationError(MCPError)`, `ConnectionError(MCPError)`.

#### `mcp/mcp_server.py`

`BlenderMCPServer` class: HTTP server in daemon thread. `start(port)` checks port availability, tries fallback ports (port to port+9), configures SO_REUSEADDR. `stop()` signals shutdown event, closes socket, waits for thread (5s), waits for port release (exponential backoff).

#### `mcp/http_handler.py`

`MCPRequestHandler(BaseHTTPRequestHandler)` implements the MCP Streamable HTTP transport; every request flows through a single `/mcp` endpoint (other paths return 404):
- `POST /mcp`: JSON-RPC requests, notifications, or responses (`initialize`, `tools/list`, `resources/list`, `resources/read`, `tools/call`). Requests receive a JSON reply; notification/response-only batches return 202 Accepted. `Accept` must include both `application/json` and `text/event-stream`.
- `GET /mcp` with `Accept: text/event-stream`: opens a server-to-client SSE stream scoped to `Mcp-Session-Id`. `Last-Event-ID` resumes the stream.
- `DELETE /mcp`: terminates the session.
- `tools/call`: posts task via `post_mcp_task()`, waits for result via `get_mcp_result(timeout=5s)`.
- Protocol version: `2025-06-18`. Origin is restricted to localhost / 127.0.0.1 / `::1`.
- CORS headers on all responses.

#### `mcp/task_system.py`

Thread-safe bridge: HTTP thread posts tasks via `post_mcp_task(task_type, args) -> task_id`, Blender timer calls `process_mcp_tasks()` which executes via integrated handlers, HTTP thread polls `get_mcp_result(task_id, timeout=5.0)`.

#### `mcp/blender_handlers.py`

| Handler | Parameters | Description |
|---------|-----------|-------------|
| `run_python_script` | code: str | Execute Python in Blender with captured stdout |
| `capture_viewport_image` | filepath: str, max_size: int=800 | Screenshot via `bpy.ops.render.opengl()` |
| `get_ui_element_status` | element_type, element_name, category | Introspect operators and properties |
| `get_average_edge_length` | object_name: str | Edge statistics in world space |
| `get_object_bounding_box_diagonal` | object_name: str | Bounding box info |
| `refresh_ui` | - | Tag all areas for redraw |

#### `mcp/handlers/` - Categorized Handlers

**connection.py** (12 handlers): `connect_ssh(host, username, key_path, remote_path, port=22, container=None)`, `connect_docker(container, path)`, `connect_local(path)`, `connect_win_native(path, port=9090)`, `disconnect()`, `connect()` (uses current settings), `start_remote_server()`, `stop_remote_server()`, `is_remote_server_running()`, `get_remote_status()`, `update_remote_status()`, `get_connection_info()`.

**group.py** (17 handlers): `create_group()`, `delete_group(uuid)`, `delete_all_groups()`, `duplicate_group(uuid)`, `rename_group(uuid, name)`, `bake_group_animation(uuid, object_name)`, `bake_group_single_frame(uuid, object_name)`, `set_object_included(uuid, object_name, included)`, `get_active_groups()`, `add_objects_to_group(uuid, object_names)`, `remove_object_from_group(uuid, object_name)`, `remove_all_objects_from_group(uuid)`, `get_group_objects(uuid)`, `set_group_type(uuid, type)`, `set_group_material_properties(uuid, properties)` (atomic updates with contact mode validation), `add_pin_vertex_group(uuid, identifier)` (accepts `"object::vgroup"` or `"[object][vgroup]"`), `remove_pin_vertex_group(uuid, identifier)`.

**object_ops.py** (18 handlers): pin settings (`set_pin_settings`), pin operations (`add_pin_operation`, `remove_pin_operation`, `list_pin_operations`, `clear_pin_operations`), static ops (`add_static_op`, `remove_static_op`, `list_static_ops`, `clear_static_ops`), velocity keyframes (`add_velocity_keyframe`, `remove_velocity_keyframe`, `list_velocity_keyframes`, `clear_velocity_keyframes`), and collision-window helpers (`set_use_collision_windows`, `add_collision_window`, `remove_collision_window`, `list_collision_windows`, `clear_collision_windows`).

**scene.py** (12 handlers): invisible colliders (`add_invisible_wall`, `add_invisible_sphere`, `list_invisible_colliders`, `remove_invisible_collider`, `clear_invisible_colliders`), merge pairs (`add_merge_pair`, `remove_merge_pair`, `list_merge_pairs`, `clear_merge_pairs`), `snap_to_vertices(object_a, object_b)`, `bake_all_animation()`, `bake_all_single_frame()`.

**dyn_params.py** (9 handlers): dynamic-parameter CRUD (`add_dynamic_param`, `remove_dynamic_param`, `list_dynamic_params`, `add_dynamic_param_keyframe`, `remove_dynamic_param_keyframe`) and collider keyframing (`set_collider_properties`, `add_collider_keyframe`, `remove_collider_keyframe`, `list_collider_keyframes`).

**simulation.py** (9 handlers): `transfer_data()`, `run_simulation()`, `resume_simulation()`, `terminate_simulation()`, `save_and_quit_simulation()`, `update_params()`, `delete_remote_data()`, `fetch_animation()`, `clear_local_animation()`.

**debug.py** (8 handlers): `debug_data_send(data_size_mb=1)`, `debug_data_receive()`, `execute_server_command(server_script)`, `execute_shell_command(shell_command, use_shell=True)`, `git_pull_remote()`, `compile_project()`, `delete_log_file(path)`, `git_pull_local()`.

**remote.py** (5 handlers): `abort_operation()`, `install_paramiko()`, `install_docker()`, `set_scene_parameters(step_size, frame_count, frame_rate, gravity, air_density, ...)`, `get_scene_parameters()`.

**console.py** (3 handlers): `get_console_lines()`, `get_latest_error()`, `show_console()`.

---

### `lib/` - Vendored Dependencies

Pre-bundled Python packages (not addon source code):

| Package | Version | Purpose |
|---------|---------|---------|
| paramiko | 4.0.0 | SSH protocol (SSHClient, Transport, SFTP, key management) |
| invoke | 2.2.1 | Task execution framework |
| cryptography | 46.0.5 | Cryptographic primitives (used by paramiko) |
| bcrypt | 5.0.0 | Password hashing (used by paramiko) |
| cffi | 2.0.0 | C foreign function interface (used by cryptography, bcrypt, pynacl) |
| pynacl | 1.6.2 | libsodium bindings / Ed25519 (used by paramiko) |
| pycparser | 3.0 | C language parser (cffi dependency) |

---

## Python Scripting API

```python
from zozo_contact_solver import solver

# Scene parameters
solver.param.frame_count = 180
solver.param.frame_rate = 60
solver.param.gravity = (0, 0, -9.8)
solver.param.step_size = 0.001

# Create and configure groups
cloth = solver.create_group("Cloth", "SHELL")
cloth.add("Shirt", "Pants")
cloth.param.shell_density = 0.5
cloth.param.shell_young_modulus = 50.0
cloth.param.friction = 0.3
cloth.param.bend = 0.5
cloth.set_overlay_color(0, 0.75, 0, 0.75)

body = solver.create_group("Body", "STATIC")
body.add("Mannequin")

# Pin operations
pin = cloth.create_pin("Shirt", "ShoulderPins")
pin.spin(axis=[0, 0, 1], angular_velocity=360, frame_start=1, frame_end=60)
pin.unpin(frame=90)

# Center mode options for spin/scale:
pin3 = cloth.create_pin("Shirt", "HemPins")
pin3.scale(factor=0.5, center_direction=(0, 0, -1), frame_start=1, frame_end=60)  # MAX_TOWARDS
pin3.spin(axis=[0, 1, 0], angular_velocity=180, center_vertex=42)  # VERTEX mode

# Embedded move with unpin
pin4 = cloth.create_pin("Shirt", "SleevePins")
pin4.unpin(frame=30)
pin4.move(delta=(0, 0, 0.5), frame=20)

pin2 = cloth.create_pin("Shirt", "CollarPins")
pin2.move_by(delta=[0, 0, 0.5], frame_start=30, frame_end=60, transition="SMOOTH")

# Dynamic scene parameters (keyframeable)
# Flip gravity at frame 60 (hold initial value, then change)
solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))

# Start wind at frame 30
solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)

# Dynamic air density
solver.param.dyn("air_density").time(100).change(0.005)

# Remove a dynamic param
solver.param.dyn("gravity").clear()

# Invisible colliders
solver.add_wall([0, 0, 0], [0, 0, 1]).param.friction = 0.5
solver.add_sphere([0, 0, 0], 0.98).invert().hemisphere()
solver.add_sphere([0, 0, 0], 1.0).time(60).hold().time(61).radius(0.5)
solver.clear_invisible_colliders()

# Snap and merge
solver.snap("Shirt", "Mannequin")
solver.add_merge_pair("Shirt", "Mannequin")

# Reset everything
solver.clear()

# Direct operator access (fallback)
solver.some_mcp_handler_name(arg1="value")
```

---

## Development

### Hot Reload

The addon starts a TCP reload server (port 8765) on registration. Use `blender_addon/debug/main.py`:

```bash
python blender_addon/debug/main.py status                    # Check debug and MCP server status
python blender_addon/debug/main.py reload                    # Hot-reload addon
python blender_addon/debug/main.py exec "print(bpy.app.version_string)"  # Execute Python in Blender
python blender_addon/debug/main.py start-mcp --port 9633     # Start MCP server
python blender_addon/debug/main.py tools                     # List MCP tools
python blender_addon/debug/main.py tools --json              # List MCP tools as JSON
python blender_addon/debug/main.py call run_python_script '{"code": "print(1+1)"}'  # Call MCP tool
python blender_addon/debug/main.py scene                     # Get current scene JSON
python blender_addon/debug/main.py resources                 # List MCP resources
```

Options: `--host HOST`, `--mcp-port PORT`, `--timeout SECONDS`.

Profiler CLI (`blender_addon/debug/perf.py`) exposes `enable`, `disable`, `reset`, `report`, and `sample` subcommands for controlling and inspecting the in-addon performance profiler.

### TOML Profiles

**Connection profiles** (`example_profile.toml`): Presets for SSH, Docker, local, and Windows connections with all credentials and paths.

**Material profiles** (`example_material_profile.toml`): Presets including Flag (shell, young=100, density=0.1), Cotton (shell, young=50), Silk (shell, young=30), Denim (shell+solid+rod hybrid), Rubber (solid, neohookean, density=1100), Steel (solid, young=200000), Rope (rod, young=10000), Static.

**Scene profiles** (`example_scene_profile.toml`): Default (step=0.001, frames=180, fps=60, gravity=-9.8), Windy (wind_direction=[0,1,0], strength=5), HighRes (step=0.001, frames=360, cg_max_iter=20000), SlowMotion (frames=600, fps=120), ZeroGravity (gravity=[0,0,0]). Scene profiles also save/load dynamic parameters in grouped TOML format:
```toml
[[ProfileName.dyn_params]]
param_type = "WIND"

[[ProfileName.dyn_params.keyframes]]
frame = 1

[[ProfileName.dyn_params.keyframes]]
frame = 40
use_hold = true

[[ProfileName.dyn_params.keyframes]]
frame = 60
wind_direction_value = [0.0, 1.0, 0.0]
wind_strength_value = 5.0

[[ProfileName.colliders]]
collider_type = "SPHERE"
name = "Sphere 1"
position = [0.0, 0.0, 0.0]
radius = 0.98
hemisphere = true
invert = true
contact_gap = 0.001
friction = 0.0

[[ProfileName.colliders.keyframes]]
frame = 1
```

### Thread Safety

The Communicator uses 5 domain-specific `RLock`s with alphabetical acquire order to prevent deadlocks:
1. `animation` - protects `_animation`, `_fetched`
2. `connection` - protects `_connection`
3. `data` - protects `_data`, `_exec_output`
4. `status` - protects `_com` (progress, status, errors, response)
5. `task` - protects `_task`, `_interrupt`

MCP task system uses a separate `threading.Lock` for the task queue, with tasks posted from the HTTP server thread and processed in the Blender main thread via timer callback.
