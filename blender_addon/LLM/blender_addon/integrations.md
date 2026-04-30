# Integrations: MCP and Python API

This document condenses the Blender add-on integration surfaces: the bundled MCP Streamable HTTP server (for external agents and automation), rules for MCP-driven scene setup, and the in-Blender `zozo_contact_solver` Python module. All three are sibling entry points that land on the same validation layer and share the same transport to `server.py`.

## MCP server

The add-on bundles a Model Context Protocol (https://modelcontextprotocol.io/) server that exposes nearly every operation (connecting to hosts, creating groups, running a simulation, capturing the viewport, running arbitrary Python) as MCP tools. External agents (Claude Desktop, IDE plugins, automation scripts, CI runners) drive Blender and the solver through a Streamable HTTP JSON-RPC surface instead of scripting the UI.

Layered block diagram of entry points: AI agents/automation, the human operator in the Blender sidebar, and the Python user all land on three protocol surfaces (the MCP Streamable HTTP server on localhost:9633 with a "localhost only, do not expose" warning; Blender operators by `bl_idname`; and the `zozo_contact_solver` Python module which forwards unknown attributes to the matching operator). All three funnel into a shared add-on core at `scene.zozo_contact_solver` where pins, merges, colliders, and every scene mutation run through the same validation. A single wide transport row represents the five connection types (Local, SSH, Docker, Docker over SSH, Windows Native), feeding into `server.py :PORT`.

The MCP server sits beside two sibling entry points: the Blender sidebar and the `zozo_contact_solver` Python module. All three cover the same surface, land on the same validation layer, and share the same transport to `server.py`. An agent calling `tools/call` hits the same operator a human hits by clicking the button. The "localhost only" pill on the MCP box is the single security boundary this stack relies on; see Security below.

### What MCP gives you

- A stable JSON surface that does not care how the add-on's buttons are laid out this week.
- The same validation the UI uses: pins, merges, and colliders all go through the add-on's shared mutation layer, so a misbehaving agent gets the same errors a user would.
- One `run_python_script` tool for arbitrary `bpy.*` access, and one `execute_shell_command` tool for provisioning remote hosts.

### What the tool surface covers, and what it does not

Before reaching for `run_python_script`, check whether a dedicated tool already exists. The MCP server exposes ~100 tools across ten categories; `run_python_script` is the escape hatch for the gaps, not the default. The authoritative list is `llm://mcp_tools_reference`; the short version:

| Task                                                                                                                               | Where it lives                                                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Connect / disconnect / start / stop the remote solver                                                                              | Connection tools (`connect_ssh`, `connect_docker`, `connect_local`, `connect_win_native`, `disconnect`, `connect`, `start_remote_server`, `stop_remote_server`, ...) |
| Create / delete / type / rename / duplicate groups; assign and unassign objects; bake a group                                      | Group tools (`create_group`, `delete_group`, `set_group_type`, `add_objects_to_group`, `bake_group_animation`, ...) |
| Pins (settings + operations), collision windows, velocity keyframes, static object ops                                             | Object-operations tools (`add_pin_vertex_group`, `set_pin_settings`, `add_pin_operation`, `add_static_op`, `add_velocity_keyframe`, `add_collision_window`, ...) |
| Transfer, run, resume, terminate, save-and-quit, update params, delete remote data, fetch animation, clear local animation        | Simulation tools (`transfer_data`, `run_simulation`, `resume_simulation`, `terminate_simulation`, `update_params`, `fetch_animation`, ...) |
| Invisible walls and spheres, merge pairs, snap, bake-all                                                                           | Scene tools (`add_invisible_wall`, `add_invisible_sphere`, `add_merge_pair`, `snap_to_vertices`, `bake_all_animation`, ...) |
| Keyframed gravity / wind / air, collider keyframes, scene-wide step/gravity/frame/contact parameters                               | Dynamic-parameters tools (`add_dynamic_param`, `add_dynamic_param_keyframe`, `add_collider_keyframe`, `set_scene_parameters`, ...) |
| Abort long-running operations, install paramiko or docker, run server-side commands, git pull remote, compile project             | Remote tools (`abort_operation`, `install_paramiko`, `install_docker`, `execute_server_command`, `git_pull_remote`, `compile_project`, ...) |
| Read Blender's console, grab the latest error, measure a mesh, viewport screenshot, query UI state                                 | Console / Blender tools (`get_console_lines`, `get_latest_error`, `get_average_edge_length`, `get_object_bounding_box_diagonal`, `capture_viewport_image`, `get_ui_element_status`) |
| **Anything touching Blender primitives, mesh edits, object transforms, materials, modifiers, or pure `bpy.*` scripting**           | **`run_python_script` (no dedicated tool exists)**                                                       |

Concrete examples of things that **need** `run_python_script`:

- Adding primitives: `bpy.ops.mesh.primitive_cube_add`, `primitive_uv_sphere_add`, `primitive_ico_sphere_add`, `primitive_plane_add`, `primitive_cylinder_add`, etc.
- Mesh editing: entering Edit Mode, selecting vertices/edges/faces, `bpy.ops.mesh.subdivide`, `bpy.ops.mesh.extrude_region_move`, `bpy.ops.mesh.loop_cut`, ...
- Object transforms: setting `obj.location`, `obj.rotation_euler`, `obj.scale`, parenting, `bpy.ops.object.transform_apply`.
- Modifiers: adding Subsurf / Mirror / Array / Solidify, applying them, tweaking levels.
- Materials, UVs, shaders, lights, cameras, scene collections.
- Reading or mutating `.blend` data beyond what `blender://scene/current` exposes.

Concrete examples of things that should **not** use `run_python_script`:

- Driving the solver (transfer / run / resume / terminate / update_params / fetch).
- Creating, typing, or populating groups.
- Editing pins, collision windows, velocity keyframes, or static ops.
- Setting scene or material parameters.
- Adding invisible walls / spheres / merges / snap.
- Installing paramiko or docker, or triggering a remote build.

When in doubt, list `tools/list` (or ask for a quick summary of `llm://mcp_tools_reference`) before writing `bpy.*` code. A misbehaving dedicated tool returns the same validation errors the UI does; `run_python_script` bypasses that layer and can leave the add-on in an inconsistent state.

### Starting the server

Main panel → **MCP Settings** → **Start**.

Screenshot of the MCP Server section inside the Solver panel, expanded. **Start** launches the HTTP server on the port shown to the right; while running, this button swaps to **Stop** and the port field becomes read-only.

The server binds to the **MCP Port** on `localhost` (default `9633`). If the port is busy, it falls back by trying `port+1`, `port+2`, ... up to `port+9` and prints the chosen port to the Blender console. **Stop** shuts the HTTP listener down and drains the task queue.

WARNING: The server binds to `localhost` only. Do **not** port-forward it or bind it to `0.0.0.0`. `run_python_script` and `execute_shell_command` are remote code execution by design: anyone who can reach the port has full control of Blender and the machine it runs on.

### Adding the server to an MCP client

Once the server is running, point your MCP client at `http://localhost:9633/mcp` using the Streamable HTTP transport (protocol version `2025-06-18`). If port 9633 was busy and the add-on fell back to 9634, 9635, ..., up to 9642, use the port printed to the Blender console.

For Claude Code, run:

```bash
claude mcp add --transport http zozo-contact-solver http://localhost:9633/mcp
```

For clients configured through a JSON config file (Claude Desktop, Cursor, Windsurf, and similar), add an entry like:

```json
{
  "mcpServers": {
    "zozo-contact-solver": {
      "type": "http",
      "url": "http://localhost:9633/mcp"
    }
  }
}
```

Restart the client after editing its config so it picks up the new server. If the client only supports stdio-based MCP servers, use a generic Streamable HTTP bridge (for example `mcp-remote`, https://www.npmjs.com/package/mcp-remote) to wrap the endpoint.

### Security

`run_python_script` evaluates arbitrary Python inside Blender. `execute_shell_command` runs arbitrary shell. There is no sandboxing, no allowlist, no auth. This is intentional: it is the escape hatch that lets agents do anything the add-on cannot yet express.

Rules of the road:

- **Bind localhost only.** The server already does; do not change that.
- **Do not expose the port** through `ssh -R`, `ngrok`, `gh codespaces`, or any reverse proxy unless you have decided the machine is disposable.
- **Treat prompts as untrusted.** If you pipe unsanitized LLM output into `run_python_script`, you have given the LLM shell. Audit its tool calls.

### Protocol

| Property     | Value                                                                         |
| ------------ | ----------------------------------------------------------------------------- |
| Version      | `2025-06-18`                                                                  |
| Transport    | Streamable HTTP (https://modelcontextprotocol.io/specification/2025-06-18/basic/transports) on a single `/mcp` endpoint |
| Requests     | `POST /mcp` with a JSON-RPC message (or batch)                                |
| Server push  | `GET /mcp` with `Accept: text/event-stream`, resumable via `Last-Event-ID`    |
| Termination  | `DELETE /mcp` with `Mcp-Session-Id` ends the session                          |
| CORS         | Enabled on every response                                                     |

All traffic flows through `/mcp`. The client calls `initialize` first; the server returns an `Mcp-Session-Id` header that the client echoes on subsequent requests. Every POST must send `Accept: application/json, text/event-stream`. Requests without a valid session get HTTP 404 (the spec's signal to re-initialize). The JSON-RPC surface itself is the standard MCP set: `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`.

### Exposed tools

The authoritative list, with every tool name, its parameters, and its description, is auto-generated from the handler sources at every docs build and available as `mcp_reference.rst` in the docs build (not inlined here). For a live, schema-attached enumeration against a running server, use `tools/list` (or the CLI `tools` subcommand).

### Calling a tool from the CLI

The debug CLI at `blender_addon/debug/main.py` wraps the MCP client, which is the fastest way to poke at the server:

```bash
# List every tool (names only)
python blender_addon/debug/main.py tools

# Same, with full JSON Schema
python blender_addon/debug/main.py tools --json

# Call a tool; arguments are a single JSON blob
python blender_addon/debug/main.py call run_python_script '{"code": "print(1+1)"}'
python blender_addon/debug/main.py call capture_viewport_image '{"filepath": "/tmp/shot.png"}'
python blender_addon/debug/main.py call create_group '{}'

# Dump current scene state
python blender_addon/debug/main.py scene

# List MCP resources
python blender_addon/debug/main.py resources
```

CLI options: `--host`, `--mcp-port`, `--timeout`. Run `python blender_addon/debug/main.py --help` for the full subcommand surface.

### Calling a tool over HTTP

If you are integrating from something that is not the bundled CLI, drive the Streamable HTTP transport directly. The client initializes once, reuses the returned session ID on every subsequent request, and ends the session with `DELETE`:

```bash
HDR_ACCEPT='Accept: application/json, text/event-stream'
HDR_JSON='Content-Type: application/json'

# 1. Initialize. The response's Mcp-Session-Id header is the session handle.
SID=$(curl -sD - -o /dev/null -X POST http://localhost:9633/mcp \
  -H "$HDR_JSON" -H "$HDR_ACCEPT" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
       "params":{"protocolVersion":"2025-06-18","capabilities":{},
                 "clientInfo":{"name":"example","version":"0"}}}' \
  | awk 'tolower($1)=="mcp-session-id:"{print $2}' | tr -d '\r')

# 2. (Optional) send the initialized notification.
curl -s -X POST http://localhost:9633/mcp \
  -H "$HDR_JSON" -H "$HDR_ACCEPT" -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# 3. Call a tool.
curl -s -X POST http://localhost:9633/mcp \
  -H "$HDR_JSON" -H "$HDR_ACCEPT" -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call",
       "params":{"name":"run_python_script",
                 "arguments":{"code":"import bpy; print(bpy.app.version_string)"}}}'

# 4. Terminate the session when done.
curl -s -X DELETE http://localhost:9633/mcp -H "Mcp-Session-Id: $SID"
```

Requests missing `Accept: application/json, text/event-stream` get HTTP 406; requests with an unknown or absent `Mcp-Session-Id` get HTTP 404 (re-initialize to recover); non-localhost `Origin` headers get 403.

### Resources: self-contained docs for LLM clients

Beyond the tools that *do* things, the MCP server exposes a set of **resources** that *describe* the add-on. An LLM-driven client can pull every piece of reference material it needs without visiting a website, cloning the repo, or shelling into the host: the docs travel inside the add-on package and ride the same localhost HTTP channel as the tools.

Two resource families are exposed:

| URI                          | Content                                                                     |
| ---------------------------- | --------------------------------------------------------------------------- |
| `blender://scene/current`    | Live JSON snapshot of the current Blender scene. Updates on every read.     |
| `llm://index`                | Top-level routing table over every doc topic. Always read this first.       |
| `llm://<topic>`              | One clean-text markdown file per topic (overview, parameters, simulation, constraints, integrations, MCP tool reference, Python API reference, ...). Each file is sized to fit in a single context window. |

The `llm://` URIs mirror the `LLM/` tree bundled with the add-on (`blender_addon/LLM.md` plus `blender_addon/LLM/blender_addon/*.md`). They drop Sphinx / HTML directives so the text loads cleanly into an LLM without parsing noise.

Why it helps an LLM client:

1. Call `resources/list` to discover what documentation exists.
2. Read `llm://index` to see which topic file answers the user's question (parameter tuning? simulation workflow? connection setup?).
3. Read only the matching topic file, keeping the rest of its context window free for the actual task.
4. Skip to `llm://mcp_tools_reference` or `llm://python_api_reference` when the user asks for exact call signatures instead of narrative.

No HTML scraping, no filesystem access, no separate endpoint to poll: one `resources/list` + one `resources/read` and the model has every answer it needs to route the next question.

Enumerating resources (assuming `$SID` is the session ID from `initialize`):

```bash
curl -s -X POST http://localhost:9633/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"resources/list"}'
```

The reply lists every URI with a `name`, a `description` (first line of the file body), and `mimeType` (`text/markdown` for docs, `application/json` for the live scene resource).

Reading a resource:

```bash
curl -s -X POST http://localhost:9633/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SID" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "resources/read",
    "params": {"uri": "llm://overview"}
  }'
```

The response is a JSON-RPC envelope whose `result.contents[0].text` holds the markdown body. `mimeType` is `text/markdown`. Unknown URIs return a JSON-RPC error with code `-32602`; path-traversal attempts (`llm://../../etc/passwd`) are rejected at the resolver so the URI scheme cannot escape the bundled `LLM/` tree.

URI scheme:

| URI form                     | Resolves to                                             |
| ---------------------------- | ------------------------------------------------------- |
| `llm://index`                | `blender_addon/LLM.md`                                  |
| `llm://<name>`               | `blender_addon/LLM/blender_addon/<name>.md`             |
| `llm://<section>/<name>`     | `blender_addon/LLM/<section>/<name>.md` (future use)    |

The `blender_addon/` section is the only one today, so its prefix is dropped from the URI. If more sections are added later, their prefix stays in the URI to keep namespaces apart.

### See also

- Blender Python API: the same surface, but called from Blender's text editor instead of over HTTP.

UNDER THE HOOD:

**Thread model**

The HTTP handler runs on a daemon thread. It has to, because Blender owns the main thread. Mutating `bpy.*` from the daemon would race the UI and corrupt scene state.

```
  HTTP request comes in
      → request is queued
      → Blender's main-thread tick drains the queue and runs the tool
      → HTTP side returns the result
```

Practical consequences:

- Handlers see a fully valid `bpy.context` and can call `bpy.ops.*`.
- Handler callbacks should not block for long. The default poll timeout is 5 seconds. Long-running work (simulate, transfer) runs asynchronously in the add-on; the MCP call just kicks it off and returns.
- Because the queue is drained on Blender's tick, the server is effectively paused while Blender is modal (popups, file dialogs).

## MCP scene setup

The MCP surface is not just a protocol. It is also the recommended way for agents (and automation scripts) to *build* a scene before hitting **Transfer → Run**. Unlike a human user, an agent cannot eyeball a mesh and decide it is "probably fine", and the solver is unforgiving about resolution and topology with failure modes that are usually silent. This section collects the rules that keep MCP-driven setup stable.

TIP: Everything below is callable via `tools/call` on the MCP server; the exact parameter schemas are auto-generated and available as `mcp_reference.rst` in the docs build. The advice here is about *which* tools to reach for in what order, not about their JSON schemas.

### Use MCP tools, not raw Python

The add-on ships a `run_python_script` tool that evaluates arbitrary Python inside Blender. It is an escape hatch, so reserve it for things the dedicated handlers genuinely do not cover. When a first-class tool exists, prefer it:

- Creating, deleting, or typing a group: `create_group`, `delete_group`, `set_group_type`.
- Assigning objects to a group: `add_objects_to_group`.
- Setting material parameters: `set_group_material_properties`.
- Scene-level parameters: `set_scene_parameters`.
- Pins, colliders, merges: the dedicated handlers in the reference (see `mcp_reference.rst` in the docs build).

The dedicated handlers all go through the same validation layer the UI uses, so a misbehaving agent gets the same errors a user would. `run_python_script` bypasses that layer entirely.

Two further rules for agents:

- **Do not create temporary files** to stage Python code. Pass the code string directly to `run_python_script`.
- **Do not shell out** to `blender --python ...`, `cargo run`, or the debug CLI to work around a missing MCP surface. If a tool is missing, surface it as a gap rather than routing around it.

### Placement and clearance

The solver rejects self-intersecting rest geometry and most zero-clearance contact. Before creating groups:

1. Populate the scene with no overlapping meshes. Cloth must start **outside** the body it will eventually wrap, and colliders must not pierce each other.
2. Leave a small clearance at least as large as the group's contact gap (see Material Parameters for how contact gap is specified). A good rule of thumb is a clearance of two to three contact-gap widths.

Running `get_object_bounding_box_diagonal(object_name)` on the target object is a cheap way to get a sense of scale before deciding what "small clearance" means in Blender units.

### Creating sphere primitives

When an agent needs a spherical mesh (a ball, a drop, a filler object), create an **icosphere**, not a UV sphere. UV spheres concentrate triangles at the poles and leave equatorial cells stretched; that anisotropy shows up in the solver as direction-dependent stiffness and can make bending look unnatural. Icospheres are near-uniform, which the solver prefers.

In a `run_python_script` call, this looks like:

```python
import bpy
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.5)
```

For `subdivisions`, pick the level that puts the average edge length inside the 1-3 % window described next, not a fixed default.

### Mesh resolution: the 1-3 % window

The most common reason MCP-driven scenes blow up is **under-resolved meshes**. The solver needs enough vertex density to represent bending and contact; a coarse mesh produces visible facets and, worse, missed contacts.

WARNING: Aim for an **average edge length of 1-3 % of the object's bounding-box diagonal**. Below ~1 % you pay a large simulation cost without much visual gain; above ~3 % folds and wrinkles get blocky.

Two MCP tools let an agent check this without opening Blender's UI:

- `get_average_edge_length(object_name)`: returns the mean world-space edge length for a mesh.
- `get_object_bounding_box_diagonal(object_name)`: returns the largest diagonal of the object's bounding box in world units.

Typical check before assigning an object to a group:

```text
ratio = average_edge_length / bounding_box_diagonal
if 0.01 <= ratio <= 0.03:
    assign to group
elif ratio > 0.03:
    subdivide or recreate at higher resolution
else:  # ratio < 0.01
    mesh is finer than needed; fine to assign, but slow
```

Treat the 1-3 % range as a hard requirement for dynamic (**Shell**, **Solid**, **Rod**) groups. **Static** colliders can sit outside the window when they are simple primitives (a flat ground plane, a single large sphere), but anything dynamic needs to be inside it.

### Subdividing to reach the window

If a mesh is too coarse, prefer **subdivision** over remeshing. Subdivision preserves the vertex order the add-on's UUID tracking depends on, so pins and vertex groups survive.

**Always use the Simple method.** The default Catmull-Clark method rounds corners, which changes the silhouette of engineered meshes (plates, boxes, cut-outs) and can introduce self-intersections near sharp features. Simple keeps the original topology and lets the solver handle smoothing implicitly.

Example via `run_python_script`:

```python
import bpy
obj = bpy.data.objects["Cloth"]
mod = obj.modifiers.new(name="Subsurf", type="SUBSURF")
mod.subdivision_type = "SIMPLE"   # NOT the default "CATMULL_CLARK"
mod.levels = 2                    # viewport level
mod.render_levels = 2             # render level, match for determinism
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier=mod.name)
```

Re-check the edge-length ratio after applying. If a single round of subdivision still leaves you above 3 %, apply another.

### When to delete and recreate

Subdivision has a ceiling: past roughly four Simple levels the vertex count explodes and performance suffers. When a primitive is unambiguously procedural (icosphere, cylinder, torus, grid), it is usually better to **delete and recreate** at the target resolution than to keep subdividing.

The recreate loop:

1. Record the object's name, transform, and whether it sits in any groups or pin vertex groups.
2. Delete the object. Confirm it is gone (for example, by listing scene objects and asserting the name is absent) before creating the replacement. Creating before deleting can leave two copies and silently double-assign them.
3. Create the new primitive with a `subdivisions` / `resolution` parameter high enough that the 1-3 % check passes on the first try.
4. Re-apply the recorded transform and re-add it to its groups.

This pattern is much cleaner for agents than iteratively poking a mesh toward the right density.

### Material parameters: don't invent defaults

`set_group_material_properties` accepts the full material whitelist for the group's type (see Material Parameters for the list). **Set only what the user has explicitly requested.** Do not pick "reasonable" defaults for density, Young's modulus, Poisson ratio, friction, or contact gap on the user's behalf. The group's property defaults already represent a calibrated starting point, and silently overwriting them with round numbers is a recipe for sims that don't match what the user asked for.

When the user *does* request something material-specific (e.g. "make this cloth stiffer"), prefer a material profile over a hand-rolled set of numbers when one fits. Profiles are the curated values.

#### Strain limit for stiff cloth

If the user asks for a stiff, low-stretch cloth, the common recipe is:

- `enable_strain_limit = true`
- `strain_limit = 0.05` (≈ 5 %)

The rest of the defaults can stay as-is unless the user calls them out. See Strain Limit for the semantics.

#### Contact gap: pick one mode

The absolute and ratio contact-gap modes are mutually exclusive, and `use_group_bounding_box_diagonal` switches between them. Do not try to set both pairs of fields in a single `set_group_material_properties` call. If you need to change modes, issue one call that sets `use_group_bounding_box_diagonal`, then a second that sets the matching `contact_gap` / `contact_offset` or `contact_gap_rat` / `contact_offset_rat` pair.

### Group-type cheat sheet

When assigning a type with `set_group_type`, use these rules:

| Object                                                 | Type       |
| ------------------------------------------------------ | ---------- |
| Static rigid collider (floor, mannequin, wall, table)  | `STATIC`   |
| Thin flexible surface (cloth, sheet, banner, shell)    | `SHELL`    |
| Volumetric flexible solid (rubber ball, sponge, foam)  | `SOLID`    |
| 1-D flexible line (rope, cable, hair strand)           | `ROD`      |

A group with no assigned objects is almost always a bug. Before calling `transfer_data`, list every active group via `get_active_groups` and verify each one has at least one object, and that the total across groups matches the set of objects the user intends to simulate.

### See also

- MCP Server: protocol, transport, and security.
- MCP Tool Reference: full handler list and schemas (auto-generated as `mcp_reference.rst` in the docs build).
- Object Groups: what each group type does in the solver.
- Material Parameters: the fields `set_group_material_properties` accepts and their defaults.

## Python API

Everything the add-on does from the UI (creating groups, pinning vertex groups, keyframing spins, dropping invisible colliders, snapping meshes) can be driven from Python inside Blender's scripting editor. This is the right tool for procedural scene setup, batch variant generation, regression tests, and anything you do not want to click through three hundred times.

TIP: This section is a tutorial-style walkthrough. For the full method-by-method list, generated directly from the source, see the Blender Python API Reference, which is auto-generated and available as `python_api_reference.rst` in the docs build (not inlined here).

### Import

```python
from zozo_contact_solver import solver
```

The add-on publishes the `zozo_contact_solver` package at registration time, so this import works regardless of where the add-on lives on disk. Every example below assumes it is already imported.

### Scene parameters

`solver.param` is a whitelisted proxy over the scene-level state. Set any exposed property by attribute:

```python
solver.param.project_name = "shirt_drape"
solver.param.frame_count  = 180
solver.param.frame_rate   = 60
solver.param.step_size    = 0.001
solver.param.gravity      = (0, 0, -9.8)   # alias for gravity_3d
solver.param.air_density  = 0.001225
```

`gravity` is an alias for `gravity_3d`; reads and writes go through it transparently. See Scene Parameters for the full list.

#### Dynamic parameters

Keyframe-driven scene parameters use the `dyn()` builder. The API mirrors the frontend's `session.param.dyn()` but takes **frames**, not seconds:

```python
# Flip gravity at frame 60: hold the initial value through 60,
# then snap to the new value at 61.
solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))

# Start a wind gust at frame 30.
solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)

# Scalars like air_density just take a float.
solver.param.dyn("air_density").time(100).change(0.005)

# Nuke a dynamic param entirely.
solver.param.dyn("gravity").clear()
```

Valid keys: `"gravity"`, `"wind"`, `"air_density"`, `"air_friction"`, `"vertex_air_damp"`. Frames must be strictly increasing within a chain; `time(30).time(30)` raises.

See Dynamic Parameters for the semantics of `hold()` vs. `change()`.

### Groups

```python
cloth = solver.create_group("Cloth", type="SHELL")   # SOLID | SHELL | ROD | STATIC
cloth.add("Shirt", "Pants")
cloth.param.shell_density       = 0.5
cloth.param.shell_young_modulus = 50.0
cloth.param.friction            = 0.3
cloth.param.bend                = 0.5
cloth.set_overlay_color(0.0, 0.75, 0.0, 0.75)         # rgba in [0, 1]

body = solver.create_group("Body", type="STATIC")
body.add("Mannequin")
```

`create_group` returns a group proxy. Look one up later by UUID with `solver.get_group(uuid)`, or walk every active group:

```python
for g in solver.get_groups():
    print(g.uuid, g.param.friction)
```

#### Group surface

| Method                                | Purpose                                                       |
| ------------------------------------- | ------------------------------------------------------------- |
| `add(*object_names)`                  | Add one or more mesh objects by name                          |
| `remove(object_name)`                 | Remove one object                                             |
| `set_overlay_color(r, g, b, a=1.0)`   | Set and enable the viewport overlay color                     |
| `create_pin(obj, vg)`                 | Pin a vertex group; returns a pin proxy (see below)           |
| `get_pins()`                          | List every pin in this group as pin proxies                   |
| `clear_keyframes()`                   | Shortcut for `pin.clear_keyframes()` on every pin             |
| `delete()`                            | Remove this group                                             |
| `.param.<name>`                       | Whitelisted material/contact parameter access                 |
| `.uuid`                               | UUID string, stable across renames                            |

Material parameters on `.param` are validated: assigning a name outside the whitelist raises `AttributeError`. See Material Parameters for the full list.

Bulk lifecycle operations live on the solver:

```python
solver.delete_all_groups()
solver.clear()   # full reset: groups, scene params, merge pairs,
                 #             colliders, dyn params, cached anim
```

### Pins and operations

`group.create_pin(object_name, vertex_group_name)` returns a pin proxy. Every mutating method returns `self`, so chaining works:

```python
pin = cloth.create_pin("Shirt", "ShoulderPins")
pin.spin(axis=(0, 0, 1), angular_velocity=360, frame_start=1, frame_end=60)
pin.unpin(frame=90)

# Soft pin instead of a hard constraint.
cloth.create_pin("Shirt", "HemPins").pull(strength=2.0)

# Chain a scale and a spin on the same pin.
(cloth.create_pin("Shirt", "HemPins")
      .scale(factor=0.5, center_direction=(0, 0, -1), frame_start=1, frame_end=60)
      .spin(axis=(0, 1, 0), angular_velocity=180, center_vertex=42))
```

#### Pin surface

| Method                                                          | Purpose                                                   |
| --------------------------------------------------------------- | --------------------------------------------------------- |
| `pull(strength=1.0)`                                            | Switch to a soft pull force                               |
| `move(delta, frame=None)`                                       | Translate the pinned verts; auto-keyframes when `frame` given |
| `move_by(delta, frame_start, frame_end, transition="LINEAR")`   | Ramp a translation over a frame range                     |
| `spin(axis, angular_velocity, flip, center*, frame_start, frame_end, transition)` | Rotate about a derived pivot          |
| `scale(factor, center*, frame_start, frame_end, transition)`    | Scale from a derived pivot                                |
| `torque(magnitude, axis_component="PC3", flip, frame_start, frame_end)` | PCA-axis torque                                   |
| `unpin(frame)`                                                  | Release the pin at `frame`; also blocks later `move(frame=N≥frame)` |
| `clear_keyframes()`                                             | Drop all positional keyframes for this pin's verts        |
| `delete()`                                                      | Remove this pin from its group                            |

`transition` is `"LINEAR"` or `"SMOOTH"`. `torque`'s `axis_component` is `"PC1"` / `"PC2"` / `"PC3"`.

#### `move` with auto-keyframing

The first `move(...)` call with a `frame=` argument auto-keyframes the current vertex positions at the current scene frame before applying `delta` and keyframing again at `frame`. Subsequent calls only keyframe at the target frame. Calls with `frame >= unpin_frame` are silently ignored, which is handy when you want to leave the pin released cleanly.

```python
pin = cloth.create_pin("Shirt", "SleevePins")
pin.unpin(frame=30)
pin.move(delta=(0, 0, 0.5), frame=20)   # keyframed; under the unpin frame
pin.move(delta=(0, 0, 0.5), frame=40)   # ignored; past unpin
```

#### Center-mode inference for `spin` and `scale`

Pass whichever argument names your pivot, and the API picks the matching mode for you:

| Argument you pass    | Inferred `center_mode` |
| -------------------- | ---------------------- |
| `center=(x, y, z)`   | `ABSOLUTE`             |
| `center_direction=v` | `MAX_TOWARDS`          |
| `center_vertex=idx`  | `VERTEX`               |
| none of the above    | `CENTROID`             |

Passing `center_mode="..."` explicitly overrides the inference. See Pins and Operations for what each mode actually computes.

```python
pin.spin(axis=(0, 0, 1), angular_velocity=360)                        # CENTROID
pin.spin(axis=(0, 0, 1), angular_velocity=360, center=(0, 0, 1))      # ABSOLUTE
pin.spin(axis=(0, 0, 1), angular_velocity=360, center_direction=(0, 0, -1))  # MAX_TOWARDS
pin.spin(axis=(0, 0, 1), angular_velocity=360, center_vertex=42)      # VERTEX
```

### Snap and merge

```python
solver.snap("Shirt", "Mannequin")               # translate Shirt onto nearest vertex on Mannequin

solver.add_merge_pair("Shirt", "Mannequin")
solver.remove_merge_pair("Shirt", "Mannequin")
solver.get_merge_pairs()                        # → [("Shirt", "Mannequin"), ...]
solver.clear_merge_pairs()
```

All of these share the same validation layer as the MCP interface and the UI, so you get identical errors. Bad names raise `ValueError`.

### Invisible colliders

Walls and spheres return a chainable builder. Parameters on `.param` cover `friction`, `contact_gap`, `thickness`, and `enable_active_duration` / `active_duration`.

```python
# A ground plane with extra friction.
solver.add_wall(position=(0, 0, 0), normal=(0, 0, 1)).param.friction = 0.5

# An inverted hemispherical container (keeps the cloth inside a bowl).
(solver.add_sphere(position=(0, 0, 0), radius=0.98)
       .invert()
       .hemisphere())

# A sphere that shrinks at frame 61.
(solver.add_sphere(position=(0, 0, 0), radius=1.0)
       .time(60).hold()
       .time(61).radius(0.5))

# A wall that slides to a new position.
(solver.add_wall(position=(0, 0, 0), normal=(0, 1, 0))
       .time(60).hold()
       .time(61).move_to((0, 1, 0)))

solver.get_invisible_colliders()   # → [("WALL", "Wall"), ("SPHERE", "Sphere"), ...]
solver.clear_invisible_colliders()
```

#### Builder surface

| Method             | Wall | Sphere | Purpose                                              |
| ------------------ | :--: | :----: | ---------------------------------------------------- |
| `.time(frame)`     |  yes |  yes   | Advance the keyframe cursor (must be increasing)     |
| `.hold()`          |  yes |  yes   | Hold the previous value at the cursor                |
| `.move_to(pos)`    |  yes |  yes   | Keyframe a new position                              |
| `.move_by(delta)`  |  yes |        | Keyframe a position offset from the previous         |
| `.radius(r)`       |      |  yes   | Keyframe a new radius                                |
| `.transform_to(p, r)` |   |  yes   | Keyframe position + radius together                  |
| `.invert()`        |      |  yes   | Flip inside-out (contact on the inside)              |
| `.hemisphere()`    |      |  yes   | Treat as a hemisphere                                |
| `.param.*`         |  yes |  yes   | `friction`, `contact_gap`, `thickness`, `active_duration`, `enable_active_duration` |
| `.delete()`        |  yes |  yes   | Remove this collider                                 |

See Invisible Colliders for how the keyframe timeline is evaluated.

### Reset

```python
solver.clear()
```

Wipes every active group, resets scene parameters to their property defaults, clears merge pairs, invisible colliders, dynamic parameters, cached fetched frames, and residual `MESH_CACHE` modifiers. Run it at the top of any script that needs a clean slate.

### Fallback: raw operator dispatch

Anything not yet on the fluent API is reachable by attribute lookup. Unknown attributes on `solver` fall through to `bpy.ops.zozo_contact_solver.<name>`:

```python
# Equivalent to bpy.ops.zozo_contact_solver.transfer_data()
solver.transfer_data()

# Keyword args are forwarded as the operator's properties.
solver.set(key="project_name", value="hero_shot")
```

Every MCP handler name (see MCP Server) has a matching operator, so whatever you can call over MCP you can also call here.

### See also

- Pins and Operations: the full semantics of `spin`, `scale`, `torque`, and the center modes.
- Dynamic Parameters: `.dyn()` keys, `hold()` / `change()`, and how they interpolate.
- Object Groups: what each group type actually does in the solver.
- MCP Server: the same surface over JSON-RPC for agents.

UNDER THE HOOD:

The fluent API is a thin layer of proxy objects over the add-on's operators and scene state:

- `solver.param` exposes a whitelisted attribute surface over scene-level properties. Assigning an unknown name raises `AttributeError`. `solver.param.dyn(name)` returns a dynamic-parameter builder.
- `solver.create_group(...)` returns a group handle. Its `.param` exposes that group's material/contact whitelist.
- `group.create_pin(...)` returns a pin handle; every mutating method returns `self` so calls chain.
- `solver.add_wall(...)` and `solver.add_sphere(...)` return builder handles. The `.time()` cursor is tracked on the builder itself; frames must be strictly increasing.

The underlying proxy types are not part of the public contract. Pin your scripts to the attribute and method names shown above, not to `isinstance` checks.
