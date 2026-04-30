# LLM-friendly documentation index

Plain-text condensation of the docs for LLM consumption. Each file under
`LLM/` drops Sphinx / MyST / HTML-specific directives (figures,
admonitions, cross-reference rewriting, toctree) while preserving every
substantive fact: parameter names, defaults, ranges, tables, code
examples, command lines, state-transition rules.

Each topic file is sized to fit comfortably in one context window. If
you are routing a user question, use the **"Where to look"** section
below to narrow down before loading anything; if you are debugging a
specific failure, jump straight to `troubleshooting.md`.

## How to load these files

Two equivalent delivery channels:

- **Filesystem** (when the add-on is unpacked): read `LLM.md` at the
  add-on root, then any file under `LLM/`. All paths in the tables
  below are relative to `blender_addon/` (the add-on package root).
- **MCP** (when connected to the add-on's MCP server): every file is
  exposed as an MCP resource. Call `resources/list` to enumerate, then
  `resources/read` with the `uri`. URIs:
  - `llm://index` for this file.
  - `llm://<name>` for each `LLM/blender_addon/<name>.md` (for example
    `llm://parameters`). The `blender_addon/` section prefix is dropped
    from the URI because every file currently lives under that section;
    if more sections are added, their prefix will be kept.
  The `mimeType` is `text/markdown` and the `text` field of the first
  `contents[]` entry is the file body.

## Repository at a glance

ZOZO's Contact Solver is a GPU-accelerated contact simulation engine.
The repo contains:

- `blender_addon/`: Blender 5.0+ front-end that models scenes, assigns
  material groups and pins, streams geometry to the engine, and fetches
  animation back. **This file (`LLM.md`) and the `LLM/` tree ship
  inside the add-on** so they're available to any MCP client connected
  to the add-on's server.
- `src/`: Rust + CUDA solver kernels (`cargo build --release`).
- `server/`: Python server process the add-on talks to over
  SSH / Docker / Windows-native transports.
- `frontend/`: Python asset / scripting layer, auto-documented via
  Sphinx autodoc; no hand-written prose beyond module docstrings.
- `docs/`: Sphinx site with HTML-facing versions of the same content.
  The LLM tree here is the clean-text mirror for programmatic
  consumption.

## Conventions

- American English throughout (color / behavior / center / gray /
  analyze / optimize / modeling / canceled). No em-dashes.
- Python scripts run under `~/.local/share/ppf-cts/venv`.
- Docs build under `docs/.venv` via `docs/build.sh`.
- CUDA / Rust compilation lives on `dev-host` host in the `ppf-dev`
  container; `~/ppf-contact-solver` on dev-host is shared with the container.

## Blender add-on section map

Each file is self-contained. Load only the ones that match the
question.

### `LLM/blender_addon/overview.md` (~190 lines)

What the add-on is, who maintains it (ZOZO, Inc), how to reach the
author, a gallery of example clips, the LLM-transparency disclosure,
and a glossary of every term the other docs use, grouped into Scene
and Constraints, Parameters and Simulation, and Connections and
Integrations. Load this when:

- You don't know what "Shell" / "Static" / "PC2" / "session ID" /
  "MCP" mean in this codebase.
- You need the one-paragraph elevator pitch of the add-on.
- You want the glossary entry for a term before diving deeper.
- You need the maintainer / contact / contribution pointers.

### `LLM/blender_addon/getting_started.md` (~140 lines)

Install, UI tour, and the first end-to-end simulation. Load when:

- User is brand new and asking "how do I even start".
- You need the panel-by-panel layout of the sidebar: Backend
  Communicator, Solver, Scene Configuration, Dynamics Groups, Snap
  and Merge, Visualization.
- You need the exact click sequence for a first sim: create a group,
  assign a mesh, Transfer → Run → Fetch.
- You need prerequisites (Blender 5.0+, NVIDIA GPU + CUDA, Python /
  Docker / SSH availability).

### `LLM/blender_addon/connections.md` (~585 lines, longest aside from parameters.md)

How to wire Blender to a solver backend. Four backends, plus connection
profiles. Load when:

- User's connection isn't working, and `troubleshooting.md` didn't
  resolve it.
- User asks about a specific backend: Local (solver on same Mac/Linux
  host), SSH (remote Linux), Docker (local or over SSH), Windows
  Native (Windows solver + Windows Blender).
- Questions about ports: 9090 server, 9633 MCP, paramiko keepalive
  (30 s), modal timeouts (60 s / 16 s / 45 s / 70 s).
- Questions about `~/.ssh/config` support (only the six keywords
  listed; ProxyJump / ProxyCommand / LocalForward are ignored, use
  an external `ssh -L` tunnel instead).
- SSH multi-user setups (per-user **Server Port** and **Remote Path**
  to avoid colliding on one GPU box).
- Connection profile TOML format and how it maps across backend
  types.

### `LLM/blender_addon/scene.md` (~310 lines)

Object groups and how you tell the solver which Blender objects
matter. Load when:

- User asks about the four group types (Shell / Solid / Rod / Static)
  and what each accepts (meshes only; Rod also accepts Bezier curves;
  Static is collision-only).
- Questions about the 32-slot group model and why it survives file
  save/load.
- How to assign an object to a group, remove it, toggle Include,
  handle library-linked rejection, mix types in one scene.
- **Active collision windows**: per-object frame ranges (up to 8 per
  object) that gate contact.
- **Static objects and their Transform sub-box** (Move By / Spin /
  Scale ops, alternative to Blender fcurves, interaction with baked
  output).
- Overlay colors per group type (Solid red, Shell green, Rod yellow,
  Static blue).
- **Modifier behavior**: the solver reads `obj.data.vertices`, so
  Subdivision Surface / Bevel / Remesh / Solidify modifiers are
  ignored at transfer time (apply them first if you want the
  subdivided topology).

### `LLM/blender_addon/constraints.md` (~480 lines)

Everything that constrains vertices or objects on top of the base
group. Load when:

- Questions about **pins**: how they're stored (vertex group for
  meshes, `_pin_<name>` custom property for curves), creating them
  via Add / Create / Rename / Remove buttons, selecting / deselecting
  pinned vertices in Edit Mode.
- Questions about **pin operations**: Move By / Spin / Scale / Torque
  / Embedded Move. Torque can't coexist with Move/Spin/Scale on the
  same pin; Embedded Move is auto-added by the first Make Keyframe
  press.
- **Center modes** for Spin and Scale (Centroid / Fixed / Max Towards
  / Vertex) and the two eyedroppers (Pick from Selected, Pick
  Vertex).
- **Make Keyframe** / **Delete All Keyframes** for keyframed per-pin
  motion; Copy/Paste of pin ops between groups.
- **Invisible colliders**: walls and spheres with friction, keyframed
  position/radius, hemisphere + invert modes for bowl-shape contact.
- **Snap & merge**: making two meshes meet cleanly at a seam; merge
  pairs with and without snapping; cross-stitch anchor data.

### `LLM/blender_addon/parameters.md` (~670 lines)

Three parameter surfaces: scene (whole-sim), material (per group),
dynamic (keyframed). Load when:

- **Scene parameters**: frame count, FPS, step size, gravity, wind
  direction+strength, air density/friction, PCG iteration limits,
  auto-save interval, line-search bounds, contact NNZ, inactive
  momentum frames (defaults, ranges, UI labels, Python / TOML keys).
- **Material parameters**: per-type tables for Shell / Solid / Rod /
  Static. Density units (kg/m² / kg/m³ / kg/m), Young's modulus
  (normalized by density, see the "non-conventional" note),
  Poisson, bend, anisotropic Shrink X/Y (Shell) vs uniform Shrink
  (Solid), Strain Limit, Inflate, Stitch Stiffness, Plasticity + Bend
  Plasticity (theta + threshold), Velocity Overwrite keyframes.
- **Model selectors**: `shell_model` (Baraff-Witkin default), `solid_model` (ARAP default), `rod_model` (ARAP only). Valid
  values per type.
- **Contact Gap vs Contact Offset**: physical meaning (gap = barrier
  reach; offset = per-group padding summed across participants);
  absolute vs ratio-of-bbox-diagonal via `use_group_bounding_box_diagonal`.
- **fTetWild overrides** (Solid only): Edge Length Factor, Epsilon,
  Stop Energy, Max Opt Iterations, Optimize, Simplify Input, Coarsen
  Output, each gated by its own `ftetwild_override_<field>`
  checkbox.
- **Dynamic parameters**: keyframed gravity, wind, air density, air
  friction, vertex air damp; `use_hold` step semantics vs
  interpolated behavior; keyframe construction via `.dyn(param).key(...)`.
- Scene and material profile TOML files (always written by the Save
  icon, not by hand; schema shown for inspection).

### `LLM/blender_addon/simulation.md` (~600 lines)

The Transfer → Run → Fetch day-to-day loop, caching, baking, and
JupyterLab integration. Load when:

- User is unsure whether to press **Transfer** or **Update Params**
  (decision table covers 12 common edits including topology,
  material, pin, collider, profile load).
- Questions about the solver state machine (Connected → Ready →
  Running → Complete → Fetched; Update Params loop on Ready;
  Terminate / Resume recovery transitions).
- **Crash / resume recovery**: what happens when Blender closes
  mid-run, you click Terminate, or the solver process crashes
  (distinguish auto-save-snapshot vs no-auto-save paths; session ID
  mismatch on reconnect).
- **Sessions and recovery**: 12-hex-char session IDs stamped on PC2
  files, cache modifier, remote project dir.
- **Auto Save** vs **Save & Quit** vs **Terminate**.
- **PC2 playback**: the `ContactSolverCache` `MESH_CACHE` modifier
  sits in the first slot with `frame_start = 1.0`, drives playback
  from `<blend_dir>/data/<blend_basename>/*.pc2`. Curves update
  directly without a modifier.
- **Baking**: converts PC2 + cache modifier into shape keys + fcurves
  (destructive for the cache); per-group or per-scene.
- **JupyterLab**: running the solver from a notebook on the same
  host, and mixing notebook + Blender sessions.

### `LLM/blender_addon/integrations.md` (~635 lines)


Programmatic surfaces: MCP server (for LLMs) and Python API (for
scripting). Load when:

- User wants to drive Blender from an LLM or from an external script.
- You need the **tool surface scope** up-front: which tasks have
  dedicated MCP tools (connection, groups, pins, simulation control,
  scene/material/dynamic params, invisible colliders, snap/merge,
  bake) and which only work through `run_python_script` (Blender
  primitives, mesh edits, object transforms, modifiers, materials,
  shaders, cameras). `run_python_script` is the escape hatch, not
  the default. Read this *before* writing `bpy.*` code.
- Questions about the MCP server: how to start it (the panel, or
  `python blender_addon/debug/main.py start-mcp`), security
  (localhost-only), the Streamable HTTP endpoint (`POST /mcp` +
  session headers), `tools/list`, resource discovery.
- **MCP scene setup**: preferred patterns (use MCP tools, not raw
  Python), placement with world clearance, sphere primitive
  construction, target mesh resolution (1-3 % edge length / bbox
  diagonal window), group-type decisions.
- **Python API** via `from zozo_contact_solver import solver`: the
  `solver.param`, `solver.dyn(...)`, `solver.create_group`,
  `group.create_pin(...)`, pin operation chaining, collider builders
  (`InvisibleWallBuilder`, `InvisibleSphereBuilder`), `solver.snap(...)`,
  `solver.reset()`, `solver.transfer_data() / run_simulation() / fetch_animation() / update_params()`.
- Fallback operator dispatch: any `bpy.ops.zozo_contact_solver.<tool>`
  is callable when the Python API doesn't cover a surface.
- Pointer to auto-generated per-tool and per-method references
  (`docs/blender_addon/integrations/mcp_reference.rst` and
  `python_api_reference.rst`, regenerated by `docs/build.sh`).

### `LLM/blender_addon/debug.md` (~220 lines)

Developer tooling (not end-user workflows). Load when:

- User is working on the add-on's code and wants to hot-reload
  changes without restarting Blender (TCP port 8765, `debug/main.py`
  CLI with `reload` / `full-reload` / `call` / `status` / `tools` /
  `exec` / `scene` / `start-mcp` / `resources`).
- PropertyGroup schema changes (why a reload sometimes isn't enough
  and full-reload or Blender restart is required).
- Draw-time profiling: `debug/perf.py` CLI (enable / sample / report
  with `--top N` / `--json` / `--no-panels` / `--no-uilists` /
  `--no-overlay`).
- Related ports: 8765 (reload server), 9633 (MCP).

### `LLM/blender_addon/mcp_tools_reference.md` (~870 lines)

Every MCP tool exposed by the add-on (99 tools in 10 categories:
Connection, Group, Object operations, Simulation, Scene, Dynamic
parameters, Remote, Console, Debug, Blender). Each entry has the full
typed signature, parameter list with descriptions, and return notes.
The top of the file shows three equivalent ways to invoke every tool:
MCP Streamable HTTP `POST /mcp` (JSON-RPC `tools/call`, after an
`initialize` handshake that returns an `Mcp-Session-Id` to echo on
every subsequent request), `bpy.ops.zozo_contact_solver.<tool>(...)`,
and `python blender_addon/debug/main.py call <tool> '{"arg":"value"}'`.

Load when:

- You are an LLM deciding which MCP tool to call for a task.
- You need the exact parameter names and types for a tool.
- You need to distinguish two similarly-named tools (e.g.
  `connect_ssh` vs `connect_docker_ssh`, `set_material_params` vs
  `set_scene_params`).
- The narrative explanation in `integrations.md` doesn't tell you
  enough to call the tool cold.

Mirror of the auto-generated
`docs/blender_addon/integrations/mcp_reference.rst` (regenerated from
`blender_addon/mcp/handlers/*.py` by `docs/build.sh`).

### `LLM/blender_addon/python_api_reference.md` (~745 lines)

Every class, method, and attribute on the `solver` singleton you get
from `from zozo_contact_solver import solver`. Classes: `Solver`,
`SceneParam`, `DynParam`, `Group`, `GroupParam`, `Pin`, `Wall`,
`Sphere`, `ColliderParam`. Each entry has the typed signature,
parameters, return type, and example usage. Plus the top-level
fallback: any `bpy.ops.zozo_contact_solver.<name>()` is reachable as
`solver.<name>(...)` via `__getattr__`, so every MCP tool is also a
Python method.

Load when:

- You are scripting the add-on in Python and need method signatures.
- You are chaining operations on a pin (`pin.spin(...).scale(...)`)
  and need to know which kwargs are accepted.
- You need the exact shape of `solver.param` / `group.param` / the
  `dyn()` keyframe builder.
- The narrative walkthrough in `integrations.md` doesn't give you
  enough to write correct code cold.

Mirror of the auto-generated
`docs/blender_addon/integrations/python_api_reference.rst` (regenerated
from `blender_addon/ops/api.py` by `docs/build.sh`).

### `LLM/blender_addon/troubleshooting.md` (~430 lines)

Failure modes grouped by subject (installation, each connection type,
connection profiles, server startup, scene setup, transfer, simulation,
fetch / playback, bake, MCP server, debug CLI, hot reload), with each
entry in a You see / Why / Fix shape for fast lookup. Load when:

- User reports a specific error message or symptom.
- Quick triage before diving into the relevant long doc.

## Cross-file topic index

Some topics show up in multiple files. This index points at the
primary home and the secondary homes so you don't miss anything.

| Topic                                     | Primary               | Secondary                                    |
| ----------------------------------------- | --------------------- | -------------------------------------------- |
| Pin creation & vertex groups              | constraints.md        | integrations.md (Python API), getting_started.md (example walkthrough) |
| Pin operations (Move/Spin/Scale/Torque)   | constraints.md        | integrations.md (Python API chaining)        |
| Pin Keyframes (Embedded Move)             | constraints.md        | parameters.md (Velocity Overwrite is a separate surface), integrations.md |
| Material parameters per group type        | parameters.md         | scene.md (which group type exposes which)    |
| Contact Gap / Offset semantics            | parameters.md         | scene.md (inherits), constraints.md (colliders have their own contact_gap) |
| Scene parameters (gravity/wind/air)       | parameters.md         | constraints.md (dynamic params can keyframe them) |
| Dynamic (keyframed) parameters            | parameters.md         | integrations.md (Python API: `solver.dyn(...)`) |
| Transfer vs Update Params decision        | simulation.md         | parameters.md (references it when discussing scene param changes) |
| PC2 cache & session ID                    | simulation.md         | troubleshooting.md (session mismatch error)  |
| Invisible colliders                       | constraints.md        | parameters.md (referenced from scene profiles section), integrations.md (Python API) |
| Snap & merge                              | constraints.md        | scene.md (briefly, in Static Objects), integrations.md |
| Active collision windows                  | scene.md              | parameters.md (the toggle lives in Material Params) |
| Object group types (SHELL/SOLID/ROD/STATIC) | scene.md            | parameters.md (per-type param tables), constraints.md (curve pins live on ROD only) |
| Static group Transform ops                | scene.md              | constraints.md (Move By / Spin / Scale share the operation semantics with pin ops) |
| Modifier stack behavior                   | scene.md              | simulation.md (the cache modifier is installed post-Fetch) |
| SSH + port forwarding                     | connections.md        | troubleshooting.md (auth errors)             |
| MCP server & tools (narrative)            | integrations.md       | debug.md (CLI starts / reaches it)           |
| **MCP tool invocation** (per-tool signatures) | mcp_tools_reference.md | integrations.md (narrative overview)     |
| Python API entry points (narrative)       | integrations.md       | every workflow page has end-of-file API examples |
| **Python API reference** (class / method signatures) | python_api_reference.md | integrations.md (narrative overview) |
| Hot reload + debug CLI                    | debug.md              | troubleshooting.md ("reload didn't restart MCP")   |

## Where to look: question → file

Fast lookup for common question shapes.

| Question shape                                                  | Start at                   |
| --------------------------------------------------------------- | -------------------------- |
| "What does this error mean?"                                    | troubleshooting.md         |
| "How do I connect to [backend]?"                                | connections.md             |
| "Does my SSH / Docker / Windows setup work?"                    | connections.md + troubleshooting.md |
| "What units are [parameter] in? What's its default?"            | parameters.md              |
| "Why is my cloth not behaving right?"                           | parameters.md (material), then simulation.md (Update Params vs Transfer) |
| "How do I pin vertices and move them?"                          | constraints.md             |
| "How do I add a collider / wall / sphere?"                      | constraints.md             |
| "How do I snap object A to object B at a seam?"                 | constraints.md             |
| "Should I press Transfer or Update Params?"                     | simulation.md              |
| "Simulation crashed / Blender closed, can I recover?"           | simulation.md (Recovery scenarios section) |
| "How do I bake my simulation into keyframes?"                   | simulation.md (Baking)     |
| "Drive the add-on from Python / LLM" (narrative)                | integrations.md            |
| "What's the signature of `<Python method>`?"                    | python_api_reference.md    |
| "What's the signature of `<MCP tool>`? How do I call it?"       | mcp_tools_reference.md     |
| "List every MCP tool in category X"                             | mcp_tools_reference.md     |
| "List every method on `solver` / `group` / `pin`"               | python_api_reference.md    |
| "How do I iterate on the add-on code without restarting Blender?" | debug.md                 |
| "What does term X mean in this codebase?"                       | overview.md (Glossary)     |
| "I'm brand new, where do I start?"                              | getting_started.md         |

## How to use these files

- **One file at a time.** Each is sized to fit in a single context
  window.
- **Skim the headings first.** Every file has predictable
  `## Overview` / `## <Topic>` / `### <Subtopic>` structure matching
  the source docs.
- **Full reference docs are now mirrored:** `mcp_tools_reference.md`
  and `python_api_reference.md` condense the auto-generated `.rst`
  references into clean markdown. Load one of these when you need an
  exact method or tool signature. The source `.rst` files remain the
  authoritative auto-generated version (regenerated from
  `blender_addon/` by `docs/build.sh`); the markdown mirrors may lag
  behind by one regeneration cycle.
- **When in doubt, load `overview.md`** for vocabulary, then branch
  into the topic file.
