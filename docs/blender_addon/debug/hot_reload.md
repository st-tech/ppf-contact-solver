# Hot Reload

While you are iterating on the add-on source, you do not want to disable,
re-enable, and sometimes restart Blender on every change. The add-on registers
a small TCP reload server on startup that does all of that from the command
line in roughly a second.

## The Reload Server

Every time the add-on registers, it starts a small JSON-over-TCP server
on `127.0.0.1:8765`. Each connection sends one JSON object with a
`command` key and reads the reply until the server half-closes.

| Command       | Effect                                                                                  |
| ------------- | --------------------------------------------------------------------------------------- |
| `ping`        | Liveness check. Used by `debug/main.py status`.                                         |
| `reload`      | Disables the add-on, purges its modules, re-enables it.                                 |
| `full_reload` | Same as `reload`, but splits disable and enable across two event-loop ticks. See below. |
| `execute`     | Runs arbitrary Python inside Blender's main thread. `print()` output is captured.       |
| `start_mcp`   | Starts the MCP server on the supplied `port` (default `9633`).                          |

All commands that touch Blender state are queued onto the main thread;
the TCP handler waits for the main thread to report back, then sends
the reply.

### `reload` vs `full_reload`

`reload` runs both phases inside the same timer callback. This is fine for
most changes but occasionally fails to refresh RNA for nested
`CollectionProperty` schemas (for example, adding a field to a
`PropertyGroup` that is embedded inside another `PropertyGroup`'s
collection).

`full_reload` splits disable and enable across two ticks with a 0.3 s gap,
which lets Blender run its RNA cleanup between them. It is slower but more
reliable for schema-heavy edits. Use it via
`python blender_addon/debug/main.py full-reload`.

:::{warning}
**New PropertyGroup fields still require a full Blender restart.**

Neither `reload` nor `full_reload` can register new fields on a
`PropertyGroup` that is already bound into a `CollectionProperty` or pointed
to by existing saved `.blend` data. If you added a new property, removed
one, or changed a property's type, quit Blender and start it again.

This is a long-standing limitation of Blender's RNA system, not of the
reload server. When the UI "looks right" but new fields are missing from
operators or profile exports, this is why.
:::

## The `debug/main.py` CLI

The CLI lives at `blender_addon/debug/main.py` and talks to both the debug
reload port (TCP 8765) and the MCP HTTP port (default 9633). All commands
accept the global options listed below. Run it from a host shell, not from
inside Blender.

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

### Examples

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

:::{tip}
For interactive exploration, pipe into `exec -`:

```bash
echo "import bpy; print(len(bpy.data.objects))" | \
    python blender_addon/debug/main.py exec -
```

The `exec` path goes through MCP's `run_python_script` tool if available and
falls back to the raw debug port otherwise, so both paths need to agree on
the host.
:::

## Related Pages

- [Profiling](profiling.md) uses the same transport to enable, sample, and
  report draw-time statistics.

:::{admonition} Under the hood
:class: toggle

**What "reload" actually does**

Reloading purges every add-on module from `sys.modules` (child modules
first) and re-imports the package. Before disabling, the add-on
remembers which companion servers (the reload server itself, and the
MCP server if it was running) were active; after the fresh import
finishes, those companions are restarted in place so your existing
`debug/main.py` session stays connected.

The restart hand-off hides in `bpy.app.driver_namespace` because Blender
rejects arbitrary attribute writes on `bpy.app` itself, but the driver
namespace survives a full disable/enable cycle.

**Timers and modal operators**

Teardown is careful to avoid firing into freed module state after a
reload:

- Every one-shot timer that was scheduled at registration (the
  group-name cleanup, the post-reload server restart, and so on) is
  canceled.
- The frame-pump modal operator is asked to exit on its next timer
  event *before* its class is unregistered. Unregistering a class
  whose modal is still alive crashes Blender.
- The add-on flips its own "ready" flag off as the very first step of
  teardown so the persistent engine tick (which runs outside the
  add-on's class registrations) stops touching PropertyGroup state
  before anything is torn down.
:::
