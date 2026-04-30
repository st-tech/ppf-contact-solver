# Debug tooling

This document condenses the Blender add-on debug surfaces: the hot-reload TCP server with its `debug/main.py` CLI (for iterating on add-on source without restarting Blender), and the draw-time profiler driven by `debug/perf.py` (for diagnosing laggy sidebars and overlays). Both tools share the same transport.

## Hot reload

While you are iterating on the add-on source, you do not want to disable, re-enable, and sometimes restart Blender on every change. The add-on registers a small TCP reload server on startup that does all of that from the command line in roughly a second.

### The reload server

Every time the add-on registers, it starts a small JSON-over-TCP server on `127.0.0.1:8765`. Each connection sends one JSON object with a `command` key and reads the reply until the server half-closes.

| Command       | Effect                                                                                  |
| ------------- | --------------------------------------------------------------------------------------- |
| `ping`        | Liveness check. Used by `debug/main.py status`.                                         |
| `reload`      | Disables the add-on, purges its modules, re-enables it.                                 |
| `full_reload` | Same as `reload`, but splits disable and enable across two event-loop ticks. See below. |
| `execute`     | Runs arbitrary Python inside Blender's main thread. `print()` output is captured.       |
| `start_mcp`   | Starts the MCP server on the supplied `port` (default `9633`).                          |

All commands that touch Blender state are queued onto the main thread; the TCP handler waits for the main thread to report back, then sends the reply.

#### `reload` vs `full_reload`

`reload` runs both phases inside the same timer callback. This is fine for most changes but occasionally fails to refresh RNA for nested `CollectionProperty` schemas (for example, adding a field to a `PropertyGroup` that is embedded inside another `PropertyGroup`'s collection).

`full_reload` splits disable and enable across two ticks with a 0.3 s gap, which lets Blender run its RNA cleanup between them. It is slower but more reliable for schema-heavy edits. Use it via `python blender_addon/debug/main.py full-reload`.

WARNING: **New PropertyGroup fields still require a full Blender restart.** Neither `reload` nor `full_reload` can register new fields on a `PropertyGroup` that is already bound into a `CollectionProperty` or pointed to by existing saved `.blend` data. If you added a new property, removed one, or changed a property's type, quit Blender and start it again. This is a long-standing limitation of Blender's RNA system, not of the reload server. When the UI "looks right" but new fields are missing from operators or profile exports, this is why.

### The `debug/main.py` CLI

The CLI lives at `blender_addon/debug/main.py` and talks to both the debug reload port (TCP 8765) and the MCP HTTP port (default 9633). All commands accept the global options listed below. Run it from a host shell, not from inside Blender.

| Subcommand            | Description                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------ |
| `status`              | Check whether the debug port and MCP server are reachable.                                 |
| `reload`              | Single-tick hot reload. Timeout 45 s.                                                      |
| `full-reload`         | Two-phase reload for schema changes. Timeout 70 s.                                         |
| `exec <code>`         | Execute Python inside Blender. Pass `-` to read from stdin.                                |
| `start-mcp`           | Ask the reload server to start the MCP server. `--port` selects port.                      |
| `tools`               | List MCP tools. `--json` for raw JSON.                                                     |
| `call <tool> [json]`  | Invoke an MCP tool with the given JSON arguments.                                          |
| `scene`               | Fetch the current Blender scene via the MCP `blender://scene/current` resource.            |
| `resources`           | List MCP resources. `--json` for raw JSON.                                                 |
| `read <uri>`          | Read an MCP resource by URI. Prints the text body to stdout; `--json` prints the raw envelope. |

Global options:

| Option             | Default     | Description                                            |
| ------------------ | ----------- | ------------------------------------------------------ |
| `--host HOST`      | `localhost` | Target host for both the debug port and the MCP port.  |
| `--mcp-port PORT`  | `9633`      | MCP server port. The debug port is hardcoded to 8765.  |
| `--timeout SEC`    | `30`        | Per-request timeout (used by `call`).                  |

#### Examples

```bash
python blender_addon/debug/main.py status
python blender_addon/debug/main.py reload
python blender_addon/debug/main.py full-reload
python blender_addon/debug/main.py exec "print(bpy.app.version_string)"
python blender_addon/debug/main.py start-mcp --port 9633
python blender_addon/debug/main.py tools
python blender_addon/debug/main.py call run_python_script '{"code": "print(1+1)"}'
python blender_addon/debug/main.py scene
python blender_addon/debug/main.py resources
python blender_addon/debug/main.py read llm://overview
```

TIP: For interactive exploration, pipe into `exec -`:

```bash
echo "import bpy; print(len(bpy.data.objects))" | \
    python blender_addon/debug/main.py exec -
```

The `exec` path goes through MCP's `run_python_script` tool if available and falls back to the raw debug port otherwise, so both paths need to agree on the host.

### Related pages

- Profiling uses the same transport to enable, sample, and report draw-time statistics.

### Running shell commands on the solver host

`execute_shell_command(shell_command, use_shell=True)` runs an arbitrary shell command on whichever host the active connection points at (Local: the Blender machine; SSH / Docker / Docker over SSH: the remote host or its container; Windows Native: the Windows solver process's host). It is the generic counterpart to the dedicated `git_pull_remote`, `compile_project`, `install_paramiko`, and `install_docker` tools: reach for it when no dedicated tool covers the task (environment inspection, filesystem triage, ad-hoc one-shot commands during debugging).

`use_shell=True` evaluates the command through the remote side's default shell (pipes, glob expansion, environment variables). Set `use_shell=False` only when the command is a single executable with explicit argv and you want to skip shell interpolation for safety.

The companion `execute_server_command(server_script)` is a **narrower** tool: it runs a server-side Python/solver script the remote `server.py` already understands, rather than a free-form shell command, so it is the right entry point for anything the server itself exposes as a subcommand. Use `execute_shell_command` when you need the host shell, `execute_server_command` when the solver already has a built-in for it.

Both tools need an active, non-busy connection (see Connections: backends) and fail fast if the server is mid-transfer or mid-run.

UNDER THE HOOD:

**What "reload" actually does**

Reloading purges every add-on module from `sys.modules` (child modules first) and re-imports the package. Before disabling, the add-on remembers which companion servers (the reload server itself, and the MCP server if it was running) were active; after the fresh import finishes, those companions are restarted in place so your existing `debug/main.py` session stays connected.

The restart hand-off hides in `bpy.app.driver_namespace` because Blender rejects arbitrary attribute writes on `bpy.app` itself, but the driver namespace survives a full disable/enable cycle.

**Timers and modal operators**

Teardown is careful to avoid firing into freed module state after a reload:

- Every one-shot timer that was scheduled at registration (the group-name cleanup, the post-reload server restart, and so on) is canceled.
- The frame-pump modal operator is asked to exit on its next timer event *before* its class is unregistered. Unregistering a class whose modal is still alive crashes Blender.
- The add-on flips its own "ready" flag off as the very first step of teardown so the persistent engine tick (which runs outside the add-on's class registrations) stops touching PropertyGroup state before anything is torn down.

## Profiling

When the sidebar feels laggy or a viewport overlay is dragging redraws, the in-addon draw-time profiler tells you which draw callbacks are eating the frame. It wraps every add-on draw method so each call is timed, and leaves everything else alone.

The profiler lives at `blender_addon/ui/perf.py`. You drive it from the CLI at `blender_addon/debug/perf.py`, which ships Python snippets to Blender over the same transport as `debug/main.py`.

### What it instruments

| Category        | Target                                                                         |
| --------------- | ------------------------------------------------------------------------------ |
| Panels          | Every registered `Panel.draw` whose class lives in this add-on.                |
| UILists         | Every `UIList.draw_item`, `draw_filter`, and `filter_items` method.            |
| Viewport overlay | `POST_VIEW` and `POST_PIXEL` draw-handler callbacks in `ui/dynamics/overlay`. |

Each recorded entry stores call count, cumulative total, minimum, and maximum time. Draw methods whose module is not part of this add-on are left alone.

### CLI

```bash
python blender_addon/debug/perf.py enable           # install timing wrappers
python blender_addon/debug/perf.py sample           # enable + force redraws + report
python blender_addon/debug/perf.py report           # print current stats
python blender_addon/debug/perf.py report --json    # raw JSON for further analysis
python blender_addon/debug/perf.py reset            # clear collected stats
python blender_addon/debug/perf.py disable          # restore original draw methods
```

Options for `enable`:

| Flag            | Effect                                           |
| --------------- | ------------------------------------------------ |
| `--no-panels`   | Skip wrapping `Panel.draw` methods.              |
| `--no-uilists`  | Skip wrapping `UIList` methods.                  |
| `--no-overlay`  | Skip wrapping viewport overlay callbacks.        |

`sample` forces redraws via `bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=N)` inside the same snippet that enables the profiler and prints the report. It is the fastest way to get a first readout, but it only covers panels and overlays that are **currently visible**. Panels in a collapsed region, a closed sidebar, or a different workspace will report zero samples.

Global options (same as `debug/main.py`):

| Option             | Default     | Description                       |
| ------------------ | ----------- | --------------------------------- |
| `--host HOST`      | `localhost` | Target host.                      |
| `--mcp-port PORT`  | `9633`      | MCP server port.                  |

### Recommended workflow

1. **Enable.** Run `perf.py enable`. The profiler resets its stats and tags all areas for redraw, so the first samples land almost immediately.
2. **Reproduce.** Hover the mouse over the affected panels, resize the sidebar, scrub frames, toggle overlays, or anything else that would cause the draws you want to measure. The more you trigger the slow path, the more meaningful the numbers.
3. **Report.** Run `perf.py report`. Rows are sorted by cumulative total time.
4. **Disable.** Run `perf.py disable` when you are done. The wrappers are cheap but they do add a `time.perf_counter()` call per draw.

`reset` without `disable` is useful when you want to measure a specific interaction: enable, idle, reset, interact, report.

### Reading the output

Columns are fixed-width:

```text
name                                                         count  total(ms)  mean(ms)   max(ms)
----------------------------------------------------------------------------------------------------
Panel:DYNAMICS_PT_Groups.draw                                  42      318.14     7.575    18.402
Overlay:draw_overlay_callback                                 180       84.20     0.468     2.031
UIList:OBJECT_UL_PinOperationsList.draw_item                  612       31.44     0.051     0.330
```

Interpretation hints:

- **Sort order is cumulative total time.** A panel with a very high `count` but low `mean` is not necessarily slow, but it is being redrawn far more often than expected. This is usually a sign of over-aggressive `tag_redraw` calls somewhere (e.g. a timer that tags on every tick, or an operator that tags inside a loop).
- **Overlay entries appear by handler name**, prefixed with `Overlay:`. There are two hooks: one for geometry (POST_VIEW) and one for labels (POST_PIXEL text). If the labels hook dominates, the culprit is usually a large number of text draws per pin / operation.
- **`max` spikes** tell you about worst-case frames. A `mean` of 0.5 ms with a `max` of 20 ms means something occasionally re-builds geometry from scratch, which is worth profiling with `enable` and then looking at the cache version counters in `overlay_cache`.

TIP: `UIList.draw_item` is called once per row every draw. If a list with 500 pins shows `count = 50000`, the list has been drawn 100 times. That is expensive regardless of how fast each row is.

### Related pages

- Hot Reload: the transport used by the profiler CLI.

UNDER THE HOOD:

**Overlay handler install**

Overlay handlers need a slightly different install than panels and UILists because Blender stores a function pointer at registration time: the profiler briefly unregisters the overlay handlers, swaps in the timed versions, and re-registers them.

**Calling the profiler from your own script**

The CLI loads `ui/perf.py` dynamically because the add-on's top-level package name depends on how Blender installed it (legacy vs extension). The CLI scans loaded modules to find the add-on's top-level package name, then imports `<pkg>.ui.perf` from there.

If you want to drive the profiler from a Python snippet that is not routed through `debug/perf.py`, reuse the same pattern:

```python
import sys
import importlib

top = next(
    (n.removesuffix(".ui.dynamics.overlay") for n in sys.modules
     if n.endswith(".ui.dynamics.overlay")),
    None,
)
if top is None:
    raise RuntimeError("ppf-contact-solver addon does not appear to be loaded")

perf = importlib.import_module(top + ".ui.perf")
perf.enable()
perf.tag_redraw_all()
# ... trigger redraws ...
print(perf.report())
perf.disable()
```

This is also the right entry point if you are wiring a test or a regression check that needs deterministic draw-time numbers.
