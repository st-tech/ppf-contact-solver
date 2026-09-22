# 📖 Glossary

One-line definitions for the terms that appear across the rest of the
documentation, grouped by subject and alphabetized within each group.

## Scene and Constraints

**Center mode**
: How the pivot for a **Spin** or **Scale** operation is resolved. One of
  **Centroid** (vertex centroid at runtime), **Fixed** (a user-entered
  coordinate), **Max Towards** (centroid of vertices furthest along a
  direction), or **Vertex** (a single vertex picked in Edit Mode).

**Contact gap**
: Separation distance the solver enforces between surfaces, specified per
  group or per collider as an absolute Blender-unit value or as a ratio of
  the group's bounding-box diagonal. See
  [Material Parameters](workflow/params/material.md).

**Cross-stitch anchor**
: Per-vertex barycentric data recorded during a [Snap](workflow/constraints/snap_merge.md),
  tying each source-object vertex to a target triangle so the stitch
  survives small mesh edits.

**Embedded Move**
: A marker the add-on puts on a pin by itself when **Make Keyframe**
  writes per-vertex keyframes on it; it is not offered in the pin's
  **Add** dropdown. It tells the encoder where to splice the per-vertex
  track in, so the pinned vertices follow the hand-animated or scripted
  pose sequence during the solve. Deleting the keyframes removes it
  again.

**Intersection allowance**
: An opt-in that lets chosen geometry pass through other geometry: an
  allowed pair has no contact, and its overlap is never reported as an
  error. Three are set on a group (**Allow Self-Intersections**, **Allow
  Inter-Object Intersections**, **Allow Inter-Group Intersections**), each
  reaching every object assigned to it or only the objects its list names,
  and one is set on a pin (**Allow Intersections Here**). "Self" means one
  object passing through itself, so two objects in one group make an
  inter-object pair; "inter-group" covers only objects of different groups.
  None of them affects the invisible walls and spheres. See
  [Allow Intersections](workflow/params/material.md#allow-intersections).

**Invisible collider**
: A parametric wall or sphere that acts as a collision boundary for the
  simulation without being rendered in Blender. Walls are infinite planes;
  spheres support **invert** (inside-surface collision) and **hemisphere**
  (bowl shape) flags. See [Invisible Colliders](workflow/constraints/colliders.md).

**Merge pair**
: Two snapped objects registered as a solver stitch constraint, optionally
  with explicit stitch anchors. Created by the snap operator and stored on
  the scene. See [Snap and Merge](workflow/constraints/snap_merge.md).

**Object group**
: One of up to 32 slots on a scene that holds a type (**Solid** / **Shell**
  / **Rod** / **PDRD** / **Sand** / **Static**), material parameters,
  assigned meshes, and pins.
  See [Object Groups](workflow/scene/object_groups.md); Static groups are
  covered separately in [Static Objects](workflow/scene/static_objects.md).

**Operation**
: A keyframed action stacked on a pin: **Move By**, **Spin**, **Scale**,
  or **Torque**. **Torque** is exclusive with the other three, and a pin
  that carries keyframed per-vertex animation (its **Embedded Move**
  marker) accepts none of the four at all. See
  [Pins and Operations](workflow/constraints/pins.md).

**Overlay color**
: Per-group viewport tint that shows which objects belong to which group.
  Defaults by group type and can be overridden per group.

**PDRD**
: Group type for exactly-rigid bodies. The surface mesh moves as a single
  best-fit rigid transform, so no tetrahedralization is needed. Short for
  Painless Differentiable Rotation Dynamics.

**Pin**
: A named set of vertices on an object, registered as a simulation
  constraint target. On meshes this is a regular Blender vertex group;
  on curve objects (which have no vertex groups) the add-on stores the
  control-point indices in an internal `_pin_<name>` custom property on
  the curve. Stacks **operations** on top, optionally with pull strength
  or a pin duration.

**Pin duration**
: A frame limit that releases a pin at a specified frame, handing the
  affected vertices back to free dynamics mid-simulation.

**Pull strength**
: Soft-force magnitude that replaces a hard pin constraint, pulling
  vertices toward their target positions rather than holding them rigidly.

**Rod**
: Group type for 1D structures (ropes, wires, threads). Accepts mesh and
  curve objects.

**Sand**
: Group type for granular bodies, simulated as a particle cloud rather
  than a connected mesh.

**Shell**
: Group type for thin deformable surfaces (cloth, fabric). Accepts mesh
  objects.

**Snap**
: One-shot alignment that translates object A so its closest vertex
  lands on the nearest point of object B's surface -- the nearest
  triangle for a mesh or solid target, the nearest segment for a rod.
  Typically followed by a merge pair registration.

**Solid**
: Group type for volumetric deformable bodies.

**Static**
: Group type for non-deforming collision objects (ground planes, props,
  mannequins). Exposes friction and contact settings; motion is driven by
  Blender transform keyframes rather than by the solver.

**Stitch stiffness**
: Strength of the soft force that holds a stitch together, set
  per-merge-pair (or per-group for loose edges), default 1.0. The stitch
  is always a soft force, never an exact weld, and is exposed for every
  supported pair: Shell-Shell, Shell-Solid, Rod-Shell, Rod-Solid,
  Rod-Rod, Solid-Solid, and any dynamic group stitched to a Static
  collider.

**Torque**
: A pin **Operation** that applies rotational force around an axis derived
  from the pinned vertices. Exclusive with Move By, Spin, and Scale, and
  refused outright on a pin that carries keyframed per-vertex animation
  (Embedded Move).

**UUID registry**
: Per-object UUID assignment maintained by the add-on. Keeps group
  references stable across rename operations so merge pairs, assigned
  objects, and pin vertex groups survive object renames.

## Parameters and Simulation

**Bake**
: Converts the live PC2-plus-modifier preview into standard Blender
  animation: shape keys and fcurves on meshes, or per-control-point
  keyframes on curves. The baked result renders without the add-on
  installed. See [Baking Animation](workflow/sim/baking.md).

**Bend stiffness**
: Per-group resistance to bending, carried by **Shell** and **Rod** groups
  as a single `bend` value that each type scales on its own terms. On a rod
  it is measured against a one-centimeter reference segment, so a strand
  bends the same way however many segments it is drawn with. See
  [Bend Stiffness on a Rod](workflow/params/material.md#bend-stiffness-on-a-rod).

**Checkpoint**
: A saved, resumable solver state captured at a chosen frame. Add frames
  to the **Save Checkpoints** list to have the solver write a checkpoint
  at each, or let **Auto Save** record them at a fixed interval. Saved
  checkpoints are what the **Resume** picker lists, so the simulation
  can be continued from any of them later.

**Constitutive model**
: The mathematical model that governs how a group deforms (for example
  Baraff-Witkin, Stable NeoHookean, or ARAP). Available choices depend on
  group type: Shell offers Baraff-Witkin and ARAP, Solid offers Stable
  NeoHookean and ARAP, Rod is locked to ARAP.

**Dynamic parameter**
: A scene-level parameter whose value is keyframed over time. Seven can
  be: gravity, wind, air density, air friction, vertex air damp, step
  size, and inactive momentum frames. They are keyframed on their own
  **Scene Configuration** sliders as ordinary Blender F-curves — there is
  no dynamic-parameter sub-panel — then sampled per frame, uploaded with
  the rest of the parameters at transfer time, and replayed by the solver
  during the run. See
  [Dynamic Parameters](workflow/params/dynamic.md).

**Fetch**
: Downloads per-frame vertex data from the solver and wires it up to
  each simulated object so the timeline plays back the result.

**JupyterLab integration**
: A first-class path for driving the solver from a notebook on the solver
  host, including headless runs and parameter sweeps without Blender open.
  See [JupyterLab](workflow/sim/jupyterlab.md).

**Material parameter**
: A per-group property that controls deformation, density, stiffness,
  friction, and contact. The applicable fields depend on the group's
  type. See [Material Parameters](workflow/params/material.md).

**Mesh hash**
: A topology fingerprint (vertex, polygon, edge, and object counts, the
  vertex count of every pin group, and a digest of each captured Static
  deformation; no UV data enters it) recorded at
  transfer time and compared again before Run, Resume, and Fetch. A
  mismatch means the Blender mesh no longer matches what the solver
  has.

**PC2**
: Point Cache 2. The per-object vertex cache the add-on writes locally
  on Fetch (`data/<blend name>/<object uuid>.pc2`, built from the
  solver's own `vert_N.bin` frames on the remote), replayed as the
  timeline plays — through a `MESH_CACHE` modifier on meshes, and
  through a frame handler on curve rods, which carry no such modifier.
  Blender frames `1..N` map to remote frames `0..N-1`.

**Profile**
: A named preset saved to a TOML file: either a **scene profile** (scene
  parameters, dynamic parameters, and invisible colliders) or a
  **material profile** (one group's material parameters). Loaded and
  saved from the profile dropdown next to the relevant panel.

**Rayleigh damping**
: Velocity-dependent energy loss proportional to a group's material
  stiffness, set per group by **Deformation Damping** and **Bending
  Damping** (both default 0.0, which disables it). It calms
  high-frequency jitter without slowing down the bulk motion of the
  body. Deformation Damping applies to Solid, Shell, and Rod groups;
  Bending Damping applies to Shell and Rod groups only. This is distinct
  from air damping, which acts on the whole scene.

**Resume**
: Continues a paused or partially completed simulation from a saved
  state, preserving earlier results. The panel's single **Resume** button
  opens a picker listing every saved checkpoint, so you choose which one
  to continue from; frames before it are kept and the rest are
  overwritten. It stays available after a failed run as long as the
  solver still holds at least one saved checkpoint.

**Run**
: Starts the simulation on the remote solver. Warns on a stale mesh hash
  and clears prior simulation output before beginning.

**Scene parameter**
: A whole-scene setting (frame range, step size, gravity, wind, air
  properties, solver tolerances) applied globally rather than per group.
  See [Scene Parameters](workflow/params/scene.md).

**Shrink**
: Rest-shape scale carried in a group's material parameters: anisotropic
  **Shrink X** / **Shrink Y** on **Shell**, a single uniform **Shrink** on
  **Solid**, and a per-segment rest-length scale on **Rod**. Below 1.0 the
  rest shape is smaller than the drawn geometry, so the body pulls itself
  taut; above 1.0 it is larger, so the body slackens. See
  [Material Parameters](workflow/params/material.md).

**Solver state**
: The status the Backend Communicator panel shows on its "Status: ..."
  line, one of twenty-three values: Disconnected, Waiting for Server
  Start..., Ready to Run, Simulation Running..., Resumable, Fetching
  Animation..., Simulation Failed, Protocol Version Mismatch, and the
  transient ones in between.

**Transfer**
: Uploads geometry, pins, colliders, and every parameter to the solver
  and rebuilds its scene. Required whenever topology or group membership
  changes.

**Update Params on Remote**
: Re-encodes and uploads parameters without resending geometry, for fast
  iteration on dynamics and materials.

## Connections and Integrations

**Communicator**
: The add-on's single connection manager. It owns all remote operations
  from a background thread so the UI never blocks on the network.

**Compute Device**
: The Connection-box row that says whether a run uses the solver host's
  accelerator (`GPU`: CUDA or ROCm on Windows and Linux, Metal on macOS)
  or the portable `CPU` build. It names which build *directory* the
  server comes out of, and a selection the host cannot serve is refused
  rather than substituted. See
  {ref}`Choosing the build <choosing-the-build>`.

**Connection profile**
: A saved TOML entry capturing every connection field of the Backend
  Communicator panel for one host, used to switch between hosts and
  share presets across a team. **Compute Device** and **GPU Backend**
  are not among them.
  See [Connection Profiles](connections/profiles.md).

**Docker connection**
: A connection type where the solver runs inside a Docker container on a
  local Docker daemon. See [Docker (Local)](connections/docker.md).

**Docker over SSH**
: A connection type where the solver runs in a container on a Docker
  daemon reached through SSH, for environments where the administrator
  hands you a container rather than shell access. See
  [Docker over SSH](connections/docker_over_ssh.md).

**`execute_shell_command`**
: MCP tool that runs arbitrary shell commands on the connected solver
  host, through the active connection backend - a different machine
  entirely for the SSH-based types, and inside the container for the
  Docker ones - and only while a connection is live. Paired with
  `run_python_script`, which runs inside Blender, as an escape hatch for
  provisioning and maintenance tasks not yet covered by dedicated tools.

**GPU Backend**
: The Connection-box row that says which accelerator a run uses when
  **Compute Device** is `GPU` and the solver host holds more than one
  GPU build. `Automatic`, `CUDA`, or `ROCm`. See
  {ref}`Choosing the build <choosing-the-build>`.

**Linux Native connection**
: A connection type where the solver runs directly as a child process on
  this Linux machine, on CUDA, ROCm, or the CPU build, with no SSH or
  Docker. See [Linux Native](connections/linux.md).

**macOS Native connection**
: A connection type where the solver runs directly as a child process on
  an Apple-silicon Mac, on Metal or the CPU build, with no SSH or
  Docker. See {ref}`macOS Native <macos-native>`.

**MCP resource**
: A read-only asset exposed by the [MCP server](integrations/mcp.md) via
  `resources/read`, covering live scene snapshots (`blender://scene/current`)
  and the bundled `llm://<topic>` markdown docs.

**MCP server**
: The bundled Model Context Protocol server on `localhost:9633` that
  exposes add-on operations as JSON-RPC tools for external agents. See
  [MCP Server](integrations/mcp.md).

**MCP tool**
: A JSON-RPC method exposed by the [MCP server](integrations/mcp.md) and
  dispatched with `tools/call`. Goes through the same validation layer as
  the sidebar buttons.

**Native connection**
: Collectively, the three connection types whose solver is a child
  process on the machine Blender runs on, started by the add-on:
  [Windows Native](connections/windows.md),
  {ref}`macOS Native <macos-native>`, and
  [Linux Native](connections/linux.md). Only your own platform's type is
  usable.

**Protocol version**
: The wire protocol version between the add-on and the `ppf-cts-server`
  binary. Both halves take it from the same shipped file, so the pair
  that ships together always agrees and there is no number for you to
  set. TCMD requests carry a 4-byte big-endian length prefix between the
  `b"TCMD"` header and the payload, and the server supports heartbeat
  recovery on long-running operations. CBOR envelope payloads on this
  connection use the schema from the `ppf-cts-formats` crate. The server
  advertises its version on connect; a mismatch means one of the two
  binaries is stale, surfaces as a protocol-version-mismatch status, and
  refuses to proceed. See
  [Protocol version mismatch](troubleshooting.md#status-protocol-version-mismatch).

**Python API (add-on)**
: The add-on's Python API, exposed under
  `bl_ext.user_default.ppf_contact_solver.ops.api` and imported from
  Blender's text editor or a notebook. Covers the same validation layer
  as the sidebar and the MCP server. See
  [Blender Python API](integrations/python_api.md).

**`run_python_script`**
: MCP tool that evaluates arbitrary Python inside Blender. Exists for
  operations the add-on does not yet expose as first-class tools; see the
  security note in [MCP Server](integrations/mcp.md).

**`ppf-cts-server`**
: The Rust solver binary (`ppf-cts-server` on Linux and macOS,
  `ppf-cts-server.exe` on Windows) launched on the remote side (or as a
  child process for the three native types) that listens for work over
  TCP on the configured port (default 9090). Built from the `ppf-cts-server`
  crate, which wraps the algorithmic core in `ppf-cts-core`.

**Session ID**
: Per-connection identifier the add-on mints at connect (12 hex
  characters) and stamps on the artifacts that run produces. Persisted
  with the `.blend` on save so a reopened file can detect whether the
  remote has been reset since the file was saved.

**SSH Command mode**
: A connection backend that parses host, port, username, and key path
  out of a plain `ssh …` string. Convenient when the command is pasted
  from a deployment script but brittle if your real config depends on
  `~/.ssh/config` wildcards.

**SSH connection**
: A connection type where the solver runs on a remote Linux host reached
  by SSH, with credentials entered as explicit fields. See [SSH](connections/ssh.md).

**Streamable HTTP**
: The MCP transport profile (protocol version `2025-06-18`) used by the
  bundled MCP server. All traffic goes through a single `/mcp` endpoint
  with a server-assigned `Mcp-Session-Id`.

**Windows Native connection**
: A connection type where the solver runs directly as a Windows
  subprocess, on CUDA, ROCm, or the CPU build, using a bundled Python
  interpreter and no SSH or Docker.
  See [Windows Native](connections/windows.md).
