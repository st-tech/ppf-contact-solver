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
: A pin **Operation** that plays back per-vertex keyframes captured in
  Blender, letting pinned vertices follow a hand-animated or scripted
  pose sequence during the solve.

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
  / **Rod** / **Static**), material parameters, assigned meshes, and pins.
  See [Object Groups](workflow/scene/object_groups.md); Static groups are
  covered separately in [Static Objects](workflow/scene/static_objects.md).

**Operation**
: A keyframed action stacked on a pin: **Move By**, **Spin**, **Scale**,
  **Torque**, or **Embedded Move**. **Torque** is exclusive with the first
  three; it can still coexist with **Embedded Move**. See
  [Pins and Operations](workflow/constraints/pins.md).

**Overlay color**
: Per-group viewport tint that shows which objects belong to which group.
  Defaults by group type and can be overridden per group.

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

**Shell**
: Group type for thin deformable surfaces (cloth, fabric). Accepts mesh
  objects.

**Snap**
: KDTree-based vertex alignment that translates object A so its nearest
  vertices land on object B's nearest vertices. Typically followed by a
  merge pair registration and, for shell+solid pairs, a stitch stiffness.

**Solid**
: Group type for volumetric deformable bodies.

**Static**
: Group type for non-deforming collision objects (ground planes, props,
  mannequins). Exposes friction and contact settings; motion is driven by
  Blender transform keyframes rather than by the solver.

**Stitch stiffness**
: Per-merge-pair compliance value that softens the stitch when a solid is
  involved. Pure shell-shell and rod-rod pairs merge vertices exactly and
  do not expose a stiffness slider.

**Torque**
: A pin **Operation** that applies rotational force around an axis derived
  from the pinned vertices. Exclusive with Move By, Spin, and Scale;
  coexists with Embedded Move.

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

**Constitutive model**
: The mathematical model that governs how a group deforms (for example
  Baraff-Witkin, Stable NeoHookean, or ARAP). Available choices depend on
  group type.

**Dynamic parameter**
: A scene-level parameter whose value is keyframed over time: gravity,
  wind, air density, air friction, or vertex air damp. Uploaded with
  the rest of the parameters at transfer time and replayed by the
  solver during the run. See [Dynamic Parameters](workflow/params/dynamic.md).

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
: A topology fingerprint (vertex count, face count, UV layout) recorded
  at transfer time and compared again before Run and Fetch. A mismatch
  means the Blender mesh no longer matches what the solver has.

**PC2**
: Point Cache 2. The per-frame vertex file format the add-on writes on
  Fetch (`vert_N.bin`), read back as the timeline plays. Blender frames
  `1..N` map to remote frames `0..N-1`.

**Profile**
: A named preset saved to a TOML file: either a **scene profile** (scene
  parameters, dynamic parameters, and invisible colliders) or a
  **material profile** (one group's material parameters). Loaded and
  saved from the profile dropdown next to the relevant panel.

**Resume**
: Continues a paused or partially completed simulation from the last
  completed frame, preserving earlier results.

**Run**
: Starts the simulation on the remote solver. Warns on a stale mesh hash
  and clears prior simulation output before beginning.

**Scene parameter**
: A whole-scene setting (frame range, step size, gravity, wind, air
  properties, solver tolerances) applied globally rather than per group.
  See [Scene Parameters](workflow/params/scene.md).

**Solver state**
: The status surfaced by the Solver panel: Connected, Ready, Running,
  Complete, or Fetched.

**Transfer**
: Uploads geometry, pins, colliders, and every parameter to the solver
  and rebuilds its scene. Required whenever topology or group membership
  changes.

**Update Params**
: Re-encodes and uploads parameters without resending geometry, for fast
  iteration on dynamics and materials.

## Connections and Integrations

**Communicator**
: The add-on's single connection manager. It owns all remote operations
  from a background thread so the UI never blocks on the network.

**Connection profile**
: A saved TOML entry capturing every field of the Connections panel for
  one host, used to switch between hosts and share presets across a team.
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
: MCP tool that runs arbitrary shell commands on the Blender host. Paired
  with `run_python_script` as an escape hatch for provisioning and
  maintenance tasks not yet covered by dedicated tools.

**Local connection**
: A connection type where the solver runs on the same Linux host as
  Blender, reached over a loopback socket. See [Local](connections/local.md).

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

**Protocol 0.02**
: The current wire protocol version between the add-on and `server.py`.
  The server advertises its version on connect; mismatches surface as a
  protocol-version-mismatch status and refuse to proceed.

**Python API (add-on)**
: The `zozo_contact_solver` module imported from Blender's text editor or
  a notebook. Covers the same validation layer as the sidebar and the MCP
  server. See [Blender Python API](integrations/python_api.md).

**`run_python_script`**
: MCP tool that evaluates arbitrary Python inside Blender. Exists for
  operations the add-on does not yet expose as first-class tools; see the
  security note in [MCP Server](integrations/mcp.md).

**`server.py`**
: The solver process launched on the remote side (or as a local
  subprocess for Windows Native) that listens for work over TCP on the
  configured port (default 9090).

**Session ID**
: Per-connection identifier the server assigns at start. Persisted with
  the `.blend` on save so a reopened file can detect whether the remote
  has been reset since the file was saved.

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
  subprocess, using a bundled Python interpreter and no SSH or Docker.
  See [Windows Native](connections/windows.md).
