# Hot Reload

Hot reload applies edits to the add-on source without disabling,
re-enabling, or restarting Blender by hand. The add-on can run a small TCP
reload server, which stays off until you start it, and the
`blender_addon/debug/main.py` CLI sends it commands from a host shell.

## Before You Start

1. Install the add-on from the repository with `./install-blender-addon.sh`.
   It links `blender_addon/` into Blender's user extensions folder, so a
   reload loads the files you edit.
2. Start Blender in one of two ways:
   - Run `blender_addon/launch.sh`. Without `PPF_BLENDER_BIN` it looks for
     Blender on macOS and Linux only. It opens Blender with the add-on
     enabled, then starts the reload server on port 8765 and the MCP server on
     port 9633 two seconds later. If the add-on is not installed under the
     name `bl_ext.user_default.ppf_contact_solver`, it prints `addon not
     loaded, servers not started`.
   - Start Blender yourself and start the server from the sidebar, as
     described below.

`launch.sh` reads these environment variables:

| Variable              | Effect                                                              |
| --------------------- | ------------------------------------------------------------------- |
| `PPF_BLENDER_BIN`     | Path of the Blender executable to start.                            |
| `PPF_BLENDER_WINDOW`  | Window geometry, `WxH` or `X,Y,W,H`.                                |
| `DEBUG_PORT`          | Reload server port (default 8765).                                  |
| `MCP_PORT`            | MCP server port (default 9633).                                     |

On Linux, `launch.sh` also needs `DISPLAY` set and exits without it.

## The Reload Server

The server is opt-in: registering the add-on does not open the port. To start
it from the sidebar:

1. In the **Backend Communicator** panel, tick **Debug Options**.
2. Under **Add-on Local Debug Server**, set **Port** if you need another one
   (default 8765, range 1024 to 65535; the field is grayed out while the
   server runs).
3. Press **Start**. Press **Stop** to end the server.

The server listens on `localhost` only.

:::{warning}
The reload server runs any Python it receives, with no authentication, and any
process on your machine can reach it. Start it only while you are debugging
and stop it afterward. See
[Security](../security.md#debug-server).
:::

The debug section of the panel also holds **Reload Add-on Now** and **Full
Reload**. They do what the `reload` and `full_reload` commands do, without the
CLI, and both report an error until the server has been started. See
[Add-on Local Debug Server](../troubleshooting.md#add-on-local-debug-server).

A client opens one connection per command, sends one JSON object with a
`command` key, and reads the reply until the server closes the connection.

| Command       | Effect                                                                                  |
| ------------- | --------------------------------------------------------------------------------------- |
| `ping`        | Liveness check. Used by `debug/main.py status`.                                         |
| `reload`      | Disables the add-on, purges its modules, and re-enables it, all in one timer callback. The server answers `Reload timed out` after 30 s. |
| `full_reload` | Same as `reload`, but splits disable and enable across two event-loop ticks. See below. The server answers `Full reload timed out` after 60 s. |
| `execute`     | Runs the Python in the `code` field (required) on Blender's main thread, and returns the captured `print()` output. Only `bpy` is predefined. `timeout` is in seconds (default 30). |
| `start_mcp`   | Starts the MCP server on the supplied `port` (default `9633`). A server that is already running keeps its port. |

Commands that touch Blender state are queued onto the main thread, and the TCP
handler waits for the main thread to report back before it replies.

A reply is one JSON object. A successful command carries `status` set to `ok`.
A failure carries an `error` message, with `status` set to `error` in most
cases. A request that is not valid JSON, has no `command`, or names an unknown
command gets an `error` reply.

### `reload` vs `full_reload`

The JSON command is `full_reload`, and the CLI subcommand that sends it is
`full-reload`.

`reload` runs both phases inside the same timer callback. That is enough for
most changes, but in some cases it does not refresh RNA for nested
`CollectionProperty` schemas, for example when you add a field to a
`PropertyGroup` that is embedded in another `PropertyGroup`'s collection.

`full_reload` splits disable and enable across two ticks with a 0.3 s gap,
which lets Blender run its RNA cleanup between them. Use it for edits to
property definitions with `python blender_addon/debug/main.py full-reload`.

:::{warning}
**New PropertyGroup fields need `full-reload`, and sometimes a restart.**

`reload` can leave a `PropertyGroup` that is already bound into a
`CollectionProperty` without the new field. `full_reload` exists for this
case: its extra event-loop tick lets Blender drop the old RNA before the new
`register()` runs. If a property you added, removed, or retyped is still
missing after `full-reload`, quit Blender and start it again.

The symptom is a UI that looks right while new fields are missing from
operators or profile exports.
:::

## The `debug/main.py` CLI

The CLI lives at `blender_addon/debug/main.py` and talks to the reload server
(TCP 8765) and the MCP HTTP server (default 9633). Run it from a host shell,
not from inside Blender.

| Subcommand            | Description                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------ |
| `status`              | Check whether the reload port and the MCP server are reachable. For a running MCP server it also prints the server name, version, and protocol versions. |
| `reload`              | Single-tick hot reload. The CLI waits up to 45 s.                                          |
| `full-reload`         | Two-phase reload for schema changes. The CLI waits up to 70 s.                             |
| `exec <code>`         | Execute Python inside Blender. Pass `-` to read from stdin.                                |
| `start-mcp`           | Ask the reload server to start the MCP server. `--port` selects the port.                  |
| `tools`               | List MCP tools. `--json` for raw JSON.                                                     |
| `call <tool> [json]`  | Invoke an MCP tool with the given JSON arguments. `--timeout SEC` (default 30) bounds the request. |
| `scene`               | Fetch the current Blender scene via the MCP `blender://scene/current` resource.            |
| `resources`           | List MCP resources. `--json` for raw JSON.                                                 |
| `read <uri>`          | Read an MCP resource by URI. Prints the text body to stdout; `--json` prints the raw envelope. |
| `runtests --backend real [names]` | Run the headless test rig. See [Running the test rig](#running-the-test-rig). |

`reload`, `full-reload`, and `start-mcp` use the reload server only. `tools`,
`call`, `scene`, `resources`, and `read` need the MCP server. `exec` uses the
MCP server's `run_python_script` tool when the server answers and the reload
server otherwise.

Global options go before the subcommand, for example
`main.py --mcp-port 9635 tools`. `--host` applies to every subcommand except
`runtests`. `--mcp-port` applies to the subcommands that talk to the MCP
server: `status`, `tools`, `call`, `exec`, `scene`, `resources`, and `read`.

| Option             | Default     | Description                                            |
| ------------------ | ----------- | ------------------------------------------------------ |
| `--host HOST`      | `localhost` | Target host for both the reload port and the MCP port. |
| `--mcp-port PORT`  | `9633`      | MCP server port.                                       |

The CLI always connects to the reload server on port 8765, so start that
server on its default port to use the CLI. `start-mcp --port` chooses the port
the MCP server starts on. If you start it on a non-default port, pass the same
value as `--mcp-port` to the later commands.

### Running the Test Rig

`runtests` starts its own solver server for each scenario, and its own Blender
for each scenario that needs one. It does not use your live Blender, its reload
server, or the global options. `--backend` is required, and `real` is the only
value. Options:

| Option           | Effect                                                          |
| ---------------- | --------------------------------------------------------------- |
| `--list`         | Print the scenario names the run would select, and exit.        |
| `--shard I/N`    | Run every N-th scenario of the selection, starting at the I-th (0-based). |
| `--parallel N`   | Worker pool size (default 1, sequential).                       |
| `--repeat N`     | Run the scenario list N times (default 1).                      |
| `--report PATH`  | Write the aggregated report to this path.                       |
| `--timeout SEC`  | Time limit per scenario (default 60).                           |

See `blender_addon/debug/TEST_RIG.md` for the rest.

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
python blender_addon/debug/main.py read blender://scene/current
python blender_addon/debug/main.py runtests --backend real --list
```

:::{tip}
For interactive exploration, pipe into `exec -`:

```bash
echo "import bpy; print(len(bpy.data.objects))" | \
    python blender_addon/debug/main.py exec -
```

`exec` uses the MCP path when the MCP server is reachable. That path
predefines `bpy`, `bmesh`, `mathutils`, and `Vector`. Over the reload port
only `bpy` is predefined, so `import` anything else in the snippet.
:::

## Related Pages

- [Profiling](profiling.md) uses the same transport to enable, sample, and
  report draw-time statistics.
- [Troubleshooting](../troubleshooting.md#add-on-local-debug-server) covers the
  sidebar controls and recovery steps.
- [MCP Server](../integrations/mcp.md#calling-a-tool-from-the-cli) covers
  calling a tool from the CLI.

:::{admonition} Under the hood
:class: toggle

**What "reload" does**

Reloading purges every add-on module from `sys.modules` (child modules first)
and re-imports the package. Before disabling, the add-on records whether the
reload server and the MCP server were running. About one second after the
fresh import, it starts again the ones that were running, on their default
ports (8765 and 9633). A server that ran on another port is not restored, and
a CLI command sent in that interval is refused, so wait a moment before the
next `debug/main.py` command.

The record is stored in `bpy.app.driver_namespace`, because Blender rejects
arbitrary attribute writes on `bpy.app` itself and the driver namespace
is not cleared by a full disable and enable cycle.

**Timers and modal operators**

Teardown avoids running code in freed module state after a reload:

- Every one-shot timer that was scheduled at registration (the group-name
  cleanup, the post-reload server restart, and so on) is canceled.
- The frame-pump modal operator is asked to exit on its next timer event
  *before* its class is unregistered. Unregistering a class whose modal is
  still alive crashes Blender.
- The add-on sets its "ready" flag to false as the first step of teardown, so
  the persistent engine tick (which runs outside the add-on's class
  registrations) stops reading PropertyGroup state before anything is torn
  down.
:::
