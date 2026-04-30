# Overview

ZOZO's Contact Solver (https://github.com/st-tech/ppf-contact-solver) is a GPU-accelerated contact simulation engine developed by ZOZO, Inc (https://corp.zozo.com/en/). The Blender 5.0+ add-on documented here is one front-end that ships with the engine: you model the scene in Blender, assign material groups and pins, and the add-on streams geometry and parameters to the engine (running locally or on a remote host), drives the simulation, and fetches the animation back.

## Gallery

The add-on ships with six small example clips that cover a range of motion types (drape, compression, inflation, and contact-driven shape change): Crumple (compressed paper crumpling into a stack), Curtain (curtains waving in a breeze), Kite (a kite blown by wind and caught on tree branches), Press (a prawn pressed permanently together with a flower ball to form a cracker), Puff (an inflated cushion with monkey heads resting on top), and Zebra (a striped zebra car sweeping through a grass field). A separate MCP-driven demo shows an external agent (Codex) building a bowl-and-spheres scene end to end from a natural language prompt, with no UI clicks, by driving Blender and the solver through the add-on's bundled MCP server.

## Reaching the author

The project is maintained by ZOZO, Inc (https://corp.zozo.com/en/) and authored by Ryoichi Ando. The primary channel for bug reports, feature requests, and usage questions is GitHub issues (https://github.com/st-tech/ppf-contact-solver/issues), which keeps the discussion searchable for other users. For private contact, email ryoichi.ando@zozo.com. If you use the project in a public piece of work (a paper, a production credit, or a personal project), the author would like to feature it in the docs: send a link to the article, project page, or website (rather than images or clips themselves, since hosting them may run into licensing issues).

## LLM transparency

Documentation and media: all text, screenshots, diagrams, and the test simulations behind them were produced by an LLM under iterative human direction and then reviewed. The LLM drives Blender and the solver through the add-on's bundled MCP server. Planning, steering, and verifying claims against the running add-on still take real human effort; the LLM is used as an authoring tool, not a fully autonomous author. The broader goal is a semi-automatic pipeline for LLM-assisted documentation authoring rather than one-shot generation, and that pipeline still depends on noticeable human attention at every iteration.

Add-on code: the Blender add-on itself was first developed with GitHub Copilot in its early stages. Later, essentially all direct coding has been carried out by Claude Code and Codex under the author's direction. The author has reviewed the resulting code, but its internal algorithms have not been scrutinized to the same depth as the academic papers associated with the underlying solver engine. Readers relying on the add-on for research or production work should treat the add-on's code with that context in mind; the solver engine itself remains backed by the peer-reviewed publications it ships with.

Code quality and testing: code quality is kept in check through an automated test suite, and the coding agents themselves are a significant help in writing and maintaining those tests. Exhaustively hunting down every edge-case bug (which would be possible with more effort) is not a hard requirement of the project, so occasional rough edges should be expected. Bug reports with a reproduction path on GitHub issues are especially helpful and are the main way the add-on's rough edges get smoothed out over time.

## Glossary

One-line definitions for the terms that appear across the rest of the documentation, grouped by subject and alphabetized within each group.

### Scene and Constraints

**Center mode**
: How the pivot for a **Spin** or **Scale** operation is resolved. One of **Centroid** (vertex centroid at runtime), **Fixed** (a user-entered coordinate), **Max Towards** (centroid of vertices furthest along a direction), or **Vertex** (a single vertex picked in Edit Mode).

**Contact gap**
: Separation distance the solver enforces between surfaces, specified per group or per collider as an absolute Blender-unit value or as a ratio of the group's bounding-box diagonal.

**Cross-stitch anchor**
: Per-vertex barycentric data recorded during a Snap, tying each source-object vertex to a target triangle so the stitch survives small mesh edits.

**Embedded Move**
: A pin **Operation** that plays back per-vertex keyframes captured in Blender, letting pinned vertices follow a hand-animated or scripted pose sequence during the solve.

**Invisible collider**
: A parametric wall or sphere that acts as a collision boundary for the simulation without being rendered in Blender. Walls are infinite planes; spheres support **invert** (inside-surface collision) and **hemisphere** (bowl shape) flags.

**Merge pair**
: Two snapped objects registered as a solver stitch constraint, optionally with explicit stitch anchors. Created by the snap operator and stored on the scene.

**Object group**
: One of up to 32 slots on a scene that holds a type (**Solid** / **Shell** / **Rod** / **Static**), material parameters, assigned meshes, and pins. Static groups are covered separately under Static Objects.

**Operation**
: A keyframed action stacked on a pin: **Move By**, **Spin**, **Scale**, **Torque**, or **Embedded Move**. **Torque** is exclusive with the first three; it can still coexist with **Embedded Move**.

**Overlay color**
: Per-group viewport tint that shows which objects belong to which group. Defaults by group type and can be overridden per group.

**Pin**
: A named set of vertices on an object, registered as a simulation constraint target. On meshes this is a regular Blender vertex group; on curve objects (which have no vertex groups) the add-on stores the control-point indices in an internal `_pin_<name>` custom property on the curve. Stacks **operations** on top, optionally with pull strength or a pin duration.

**Pin duration**
: A frame limit that releases a pin at a specified frame, handing the affected vertices back to free dynamics mid-simulation.

**Pull strength**
: Soft-force magnitude that replaces a hard pin constraint, pulling vertices toward their target positions rather than holding them rigidly.

**Rod**
: Group type for 1D structures (ropes, wires, threads). Accepts mesh and curve objects.

**Shell**
: Group type for thin deformable surfaces (cloth, fabric). Accepts mesh objects.

**Snap**
: KDTree-based vertex alignment that translates object A so its nearest vertices land on object B's nearest vertices. Typically followed by a merge pair registration and, for shell+solid pairs, a stitch stiffness.

**Solid**
: Group type for volumetric deformable bodies.

**Static**
: Group type for non-deforming collision objects (ground planes, props, mannequins). Exposes friction and contact settings; motion is driven by Blender transform keyframes rather than by the solver.

**Stitch stiffness**
: Per-merge-pair compliance value that softens the stitch when a solid is involved. Pure shell-shell and rod-rod pairs merge vertices exactly and do not expose a stiffness slider.

**Torque**
: A pin **Operation** that applies rotational force around an axis derived from the pinned vertices. Exclusive with Move By, Spin, and Scale; coexists with Embedded Move.

**UUID registry**
: Per-object UUID assignment maintained by the add-on. Keeps group references stable across rename operations so merge pairs, assigned objects, and pin vertex groups survive object renames.

### Parameters and Simulation

**Bake**
: Converts the live PC2-plus-modifier preview into standard Blender animation: shape keys and fcurves on meshes, or per-control-point keyframes on curves. The baked result renders without the add-on installed.

**Constitutive model**
: The mathematical model that governs how a group deforms (for example Baraff-Witkin, Stable NeoHookean, or ARAP). Available choices depend on group type.

**Dynamic parameter**
: A scene-level parameter whose value is keyframed over time: gravity, wind, air density, air friction, or vertex air damp. Uploaded with the rest of the parameters at transfer time and replayed by the solver during the run.

**Fetch**
: Downloads per-frame vertex data from the solver and applies it as PC2 animation through a Mesh Cache modifier.

**JupyterLab integration**
: A first-class path for driving the solver from a notebook on the solver host, including headless runs and parameter sweeps without Blender open.

**Material parameter**
: A per-group property that controls deformation, density, stiffness, friction, and contact. The applicable fields depend on the group's type.

**Mesh hash**
: A topology fingerprint (vertex count, face count, UV layout) recorded at transfer time and compared again before Run and Fetch. A mismatch means the Blender mesh no longer matches what the solver has.

**PC2**
: Point Cache 2. The per-frame vertex cache format the add-on writes (`vert_N.bin`) and plays back through a Mesh Cache modifier. Blender frames `1..N` map to remote frames `0..N-1`.

**Profile**
: A named preset saved to a TOML file: either a **scene profile** (scene parameters, dynamic parameters, and invisible colliders) or a **material profile** (one group's material parameters). Loaded and saved from the profile dropdown next to the relevant panel.

**Resume**
: Continues a paused or partially completed simulation from the last completed frame, preserving earlier results.

**Run**
: Starts the simulation on the remote solver. Warns on a stale mesh hash and clears prior cache before beginning.

**Scene parameter**
: A whole-scene setting (frame range, step size, gravity, wind, air properties, solver tolerances) applied globally rather than per group.

**Solver state**
: The status surfaced by the Solver panel: Connected, Ready, Running, Complete, or Fetched.

**Transfer**
: Uploads geometry, pins, colliders, and every parameter to the solver and rebuilds its scene. Required whenever topology or group membership changes.

**Update Params**
: Re-encodes and uploads parameters without resending geometry, for fast iteration on dynamics and materials.

### Connections and Integrations

**Communicator**
: The add-on's single connection manager. It owns all remote operations from a background thread so the UI never blocks on the network.

**Connection profile**
: A saved TOML entry capturing every field of the Connections panel for one host, used to switch between hosts and share presets across a team.

**Docker connection**
: A connection type where the solver runs inside a Docker container on a local Docker daemon.

**Docker over SSH**
: A connection type where the solver runs in a container on a Docker daemon reached through SSH, for environments where the administrator hands you a container rather than shell access.

**`execute_shell_command`**
: MCP tool that runs arbitrary shell commands on the Blender host. Paired with `run_python_script` as an escape hatch for provisioning and maintenance tasks not yet covered by dedicated tools.

**Local connection**
: A connection type where the solver runs on the same Linux host as Blender, reached over a loopback socket.

**MCP resource**
: A read-only asset exposed by the MCP server via `resources/read`, covering live scene snapshots (`blender://scene/current`) and the bundled `llm://<topic>` markdown docs.

**MCP server**
: The bundled Model Context Protocol server on `localhost:9633` that exposes add-on operations as JSON-RPC tools for external agents.

**MCP tool**
: A JSON-RPC method exposed by the MCP server and dispatched with `tools/call`. Goes through the same validation layer as the sidebar buttons.

**Protocol 0.02**
: The current wire protocol version between the add-on and `server.py`. The server advertises its version on connect; mismatches surface as a protocol-version-mismatch status and refuse to proceed.

**Python API (add-on)**
: The `zozo_contact_solver` module imported from Blender's text editor or a notebook. Covers the same validation layer as the sidebar and the MCP server.

**`run_python_script`**
: MCP tool that evaluates arbitrary Python inside Blender. Exists for operations the add-on does not yet expose as first-class tools.

**`server.py`**
: The solver process launched on the remote side (or as a local subprocess for Windows Native) that listens for work over TCP on the configured port (default 9090).

**Session ID**
: Per-connection identifier the server assigns at start. Persisted with the `.blend` on save so a reopened file can detect whether the remote has been reset since the file was saved.

**SSH Command mode**
: A connection backend that parses host, port, username, and key path out of a plain `ssh …` string. Convenient when the command is pasted from a deployment script but brittle if your real config depends on `~/.ssh/config` wildcards.

**SSH connection**
: A connection type where the solver runs on a remote Linux host reached by SSH, with credentials entered as explicit fields.

**Streamable HTTP**
: The MCP transport profile (protocol version `2025-06-18`) used by the bundled MCP server. All traffic goes through a single `/mcp` endpoint with a server-assigned `Mcp-Session-Id`.

**Windows Native connection**
: A connection type where the solver runs directly as a Windows subprocess, using a bundled Python interpreter and no SSH or Docker.
