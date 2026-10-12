# 🤖 MCP Server

The add-on bundles a [Model Context Protocol](https://modelcontextprotocol.io/)
(MCP) server. An external agent (Claude Desktop, an IDE plugin, an
automation script, a CI runner) connects over HTTP and calls tools that
connect to hosts, create groups, run a simulation, capture the viewport, or
run Python inside Blender. The server is off until you start it from the
**Solver** panel (see [Starting the Server](#starting-the-server)). For the
order in which an agent should build a scene, see
[Scene Setup via MCP](mcp_scene_setup.md).

```{figure} ../images/integrations/entry_points.svg
:alt: Diagram of three ways to drive the add-on. AI agents and automation use the MCP Streamable HTTP server on localhost:9633, which carries the label "localhost only (do not expose)". A person in the Blender sidebar uses Blender operators. A Python user calls the add-on's Python API. All three end in one box labeled Add-on core, then one box labeled Transport that lists the SSH, Docker, Docker over SSH and native connection types and leads to ppf-cts-server.
:width: 820px

The MCP server, the Blender sidebar and the add-on's
[Python API](./python_api.md) are three entry points to the same actions.
For the tools that start a transfer, a solve or a fetch, a `tools/call` runs
the same operator as the panel button and is refused when that operator's
availability check fails. All three entry points use the same transport to
`ppf-cts-server`. The label on the MCP box reads "localhost only (do not
expose)"; see [Security](#security).
```

## What MCP Gives You

- A JSON interface that does not change when the panel layout changes.
- Input checks on the tools that add pins, merge pairs and invisible
  colliders. A missing object, a missing vertex group, a merge of an object
  with itself or a sphere radius that is not positive is refused with an error
  that names the problem, and the add-on's [Python API](./python_api.md)
  applies the same checks.
- `run_python_script` for raw `bpy`, `bmesh` and `mathutils` access when no
  dedicated tool covers an operation, and `execute_shell_command` to run a
  shell command on the connected solver host.

## Starting the Server

1. In the **Solver** panel, expand the **MCP Server** box with its triangle.
   The box starts collapsed.
2. Optional: set **Port**. The range is 1024 to 65535 and the default is
   `9633`.
3. Click **Start MCP Server on Local**. The header changes from
   **MCP Server (Stopped)** to **MCP Server (Running :9633)**, with the live
   port in place of `9633`.

```{figure} ../images/integrations/mcp_row.png
:alt: The MCP Server box in the Solver panel, expanded, with the server stopped. The header reads MCP Server (Stopped). Below it, the Start MCP Server on Local button is outlined in red under a red label reading Start, next to a Port field that reads 59864.
:width: 500px

The expanded **MCP Server** box with the server stopped. The red outline
and the red label mark **Start MCP Server on Local**. The **Port** field
shows an example value; the default is `9633`. While the server runs, the
button reads **Stop MCP Server** and the **Port** field is grayed out.
```

The server binds to `localhost` on the **Port** value. If that port is in
use, the server waits for it to be released (five attempts with a growing
delay) and then tries `port+1` through `port+9`. A substituted port is
reported as a warning and written into the **Port** field, so the header
always shows the live port. If all ten ports are taken, the start fails with
an error; see [Troubleshooting](../troubleshooting.md#mcp-server).

Click **Stop MCP Server** to shut the HTTP listener down and close every live
MCP session. The server also stops when you disable the add-on or quit
Blender.

:::{warning}
The server binds to `localhost` only. Do **not** port-forward it.
`run_python_script` and `execute_shell_command` run code for anyone who can
reach the port; see [Security](#security).
:::

## Adding the Server to an MCP Client

Start the server, then point your MCP client at

```
http://localhost:9633/mcp
```

using the **Streamable HTTP** transport. The server answers protocol
revisions `2026-07-28` and `2025-06-18` on that one URL, so a client of
either revision connects without further setup (see
[Protocol](#protocol)). If the add-on substituted another port, use the one
in the **MCP Server (Running :port)** header.

For Claude Code, run:

```bash
claude mcp add --transport http zozo-contact-solver http://localhost:9633/mcp
```

Clients that declare servers in an `mcpServers` JSON file take an entry like
this one. Key names such as `type` vary by client, so follow the client's own
documentation for remote HTTP servers.

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

A client that supports only stdio servers needs a Streamable HTTP bridge,
for example [`mcp-remote`](https://www.npmjs.com/package/mcp-remote), to wrap
the endpoint.

## Security

`run_python_script` evaluates arbitrary Python inside Blender, as your
account. `execute_shell_command` runs an arbitrary shell command on the solver
host the add-on is connected to: on the remote machine for an SSH connection,
inside the container for a Docker or Docker over SSH connection (the container
is on your own machine for Docker and on the remote machine for Docker over
SSH), and on your own machine for a native connection. It needs a live
connection that is not busy with another operation. There is no sandbox, no
allowlist and no authentication, so anyone who can reach the port can run code
through both tools, on the solver host as well as in Blender.

These tools also act outside the scene. `git_pull_local` runs `git pull` in the
add-on folder. `git_pull_remote` and `compile_project` run `git pull` and a
release build on the connected host. `install_paramiko` and `install_docker`
run `pip` with Blender's Python. `delete_log_file` deletes whatever file its
`log_file_path` argument names. `save_profile` with a `path`, `export_usd`,
`export_alembic` and `capture_viewport_image` write to a path the caller
names, and a file already at that path is rewritten. None is marked
open-world, and only `delete_log_file` is marked destructive, by its name. All
of them work with **Debug Options** off, so keep your client's approval prompts
on for them as well as for the tools marked destructive; see the annotations
under [Exposed Tools](#exposed-tools).

The server binds to `localhost` only, and the add-on has no setting to change
that. It refuses a request whose `Origin` header is not `localhost`,
`127.0.0.1` or `::1` with HTTP 403 (see [Protocol](#protocol)), which stops a
web page you visit from driving the server through your browser. Neither
protection authenticates a caller.
[Security](../security.md#code-execution-risk-mcp) covers what to do about
this risk, including the ports not to forward and reviewing an agent's tool
calls, together with the debug server, a second port that runs Python.

## Exposed Tools

The [MCP Tool Reference](./mcp_reference.rst) lists every tool with its
parameters and description. It is generated from the handler sources at every
docs build. A running server lists the same tools through `tools/list`, which
also carries each input schema; the CLI's `tools` subcommand prints that list.

Each tool's description and input schema state what it needs, its units and
what it refuses. Do not match on a tool's description text; the wording can
change. A raw HTTP example of a `tools/call` is in the **Under the hood** box
under [Protocol](#protocol).

`tools/list` also carries behavior annotations. Tools named `get_*` and
`list_*` are marked read-only, tools named `remove_*`, `delete_*` and
`clear_*` are marked destructive, and `run_python_script` and
`execute_shell_command` are marked destructive and open-world. A client can use
them to decide when to ask before running a tool. Other tools carry no
annotation.

### Reading Errors and Panel State

`get_latest_error` returns `local_error`, `remote_error` and `has_errors`.
`get_console_lines` returns the add-on console as `console_lines` and
`line_count`, and `show_console` opens the console window. The console keeps
the last **Max Console Lines** lines (see
[Console](../troubleshooting.md#console-max-console-lines-and-console-log-export)).

`get_ui_element_status(element_type, element_name, category)` reports whether
a panel operator can run at that moment (`clickable`, the result of its
availability check) and the value of add-on properties. `element_type` is
`operator`, `property` or `all`, `element_name` narrows the answer to one
name, and `category` is `solver`, `dynamics`, `client` or `debug`. Use it to
check whether a button is available. It reports `clickable` as `true` or
`false` and gives no reason, and a `poll_error` string appears when the
availability check itself raised. `element_name` for an operator is its class
name as the result lists it, for example `SOLVER_OT_Transfer`; call without
`element_name` to list the names. `refresh_ui` redraws the areas of the
current screen after a script changes state.

## Instructions and Prompts

The server's instructions tell an agent to read each tool's description and
input schema before calling it, and give the order a scene is built in:
connect, create a group, assign objects, constrain, set parameters, build,
solve, fetch. A client gets them from `server/discover`. The `2025-06-18`
`initialize` result carries no instructions, so a session-bound client must
call `server/discover` to read them.

`prompts/list` and `prompts/get` offer four prompts. Each is a short starting
point that names the ordered steps of one workflow and leaves the detail to the
tools' own descriptions:

| Prompt              | Arguments                                                   | Workflow                                                        |
| ------------------- | ----------------------------------------------------------- | --------------------------------------------------------------- |
| `run_simulation`    | `backend`, `frames` (both optional)                         | Take a scene from unconfigured to fetched results.              |
| `pin_and_constrain` | `object_name` (required), `intent` (optional)               | Hold or drive part of a mesh with pins, colliders, or merges.   |
| `tune_parameters`   | `goal` (required)                                           | Choose parameters for a goal and read back what the solver got. |
| `diagnose_failure`  | `symptom` (required)                                        | Work from a failed or stalled solve's symptom to its cause.     |

`prompts/get` refuses a request that omits a required argument. The
`run_simulation` prompt is a workflow outline; it is not the `run_simulation`
tool, which starts a solve.

The `backend` argument of the `run_simulation` prompt is free text naming where
the solver runs. The prompt inserts it as written and does not validate it. The
tools that open a connection are `connect_ssh`, `connect_docker`,
`connect_linux_native`, `connect_win_native` and `connect_mac_native`, and
`connect` uses the panel's current settings.

## Tool Calls and Long-Running Tools

A tool that starts long work returns as soon as the work has started, with a
message such as `Data transfer initiated` or `Simulation started`, and the work
continues in Blender. Poll for the outcome:

| Tools that start work | Poll with |
| --- | --- |
| `connect` and the `connect_*` tools | `get_connection_info`, field `connection_status.connected` |
| `start_remote_server`, `stop_remote_server`, `update_remote_status`, `transfer_data`, `run_simulation`, `resume_simulation`, `resume_simulation_from`, `update_params`, `terminate_simulation`, `save_and_quit_simulation`, `delete_remote_data` | `get_remote_status`: `current_status`, `progress`, `message`, `error` |
| `refresh_solver_gpus` | `list_solver_gpus`: `probed` and the refreshed `devices` |
| `fetch_animation` | `get_fetch_status`: which frames are fetched into Blender |
| `bake_group_animation`, `bake_all_animation`, `capture_static_deformation`, `capture_pin_deformation`, `recapture_all_deformations` | `get_modal_job_status`: the running bake or capture job |

To stop work that has started, call `abort_operation` for a transfer or a
fetch, `abort_bake` for a keyframe bake, and `abort_static_deformation_capture`
or `abort_pin_deformation_capture` for a capture, then poll the same status
tool until the job stops. `terminate_simulation` ends a solve at once, and
`save_and_quit_simulation` saves the current state, then quits the solver.
`disconnect` ends a connection, or cancels one that is still connecting, and
is refused while a transfer or a fetch is running.

The result's `content` holds the tool's return value as JSON text, and
`structuredContent` holds the same value parsed when it is a JSON object. A
tool that reports a failure sets `isError`, so a failure never arrives as a
successful result. A call that
takes longer than about a second can stream its answer; see the slow-call
details under [Protocol](#protocol).

## Calling a Tool from the CLI

The debug CLI is the script `blender_addon/debug/main.py` in the source tree.
It needs a checkout of the repository and Python 3 (standard library only), and
the paths below are relative to the repository root. The global options
`--host` and `--mcp-port` go before the subcommand.

```bash
# List every tool with the first line of its description, or with full JSON Schema
python blender_addon/debug/main.py tools
python blender_addon/debug/main.py tools --json

# Call a tool; arguments are a single JSON blob
python blender_addon/debug/main.py call run_python_script '{"code": "print(1+1)"}'
python blender_addon/debug/main.py call capture_viewport_image '{"filepath": "/path/to/shot.png"}'
python blender_addon/debug/main.py call create_group '{}'

# Server on another port
python blender_addon/debug/main.py --mcp-port 9635 tools

# Check that the server is reachable, then read the scene resource
python blender_addon/debug/main.py status
python blender_addon/debug/main.py read blender://scene/current

# Dump current scene state
python blender_addon/debug/main.py scene

# List MCP resources
python blender_addon/debug/main.py resources
```

`--timeout` belongs to the subcommand: on `call` it is the request timeout
(default 30 s). The full subcommand list is printed by
`python blender_addon/debug/main.py --help`.

## Resources

The server offers one resource through `resources/list` and `resources/read`:

| URI                       | Content                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------- |
| `blender://scene/current` | Objects, frame range and solver-relevant properties of the scene open in Blender, read on demand. |

Each read returns the scene as it is at that moment. An unknown URI returns a
JSON-RPC error with code `-32602`. The CLI's `read blender://scene/current`
prints the resource body. A raw HTTP example of `resources/read` is in the
**Under the hood** box under [Protocol](#protocol).

## Protocol

The server speaks MCP over Streamable HTTP on one endpoint and answers two
protocol revisions, so a client of either connects without configuration.

| Property  | Value                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------- |
| Revisions | `2026-07-28` (stateless, no handshake) and `2025-06-18` (session-bound, opens with `initialize`)         |
| Transport | Streamable HTTP on the single endpoint `/mcp`                                                           |
| Requests  | `POST /mcp`, one JSON-RPC request or notification per POST; batches are refused                         |
| Methods   | `server/discover`, `tools/list`, `tools/call`, `resources/list`, `resources/templates/list` (always empty), `resources/read`, `prompts/list`, `prompts/get`, and `initialize` for `2025-06-18` |
| Lists     | Never paged; a request carrying a `cursor` is refused                                                   |
| Origin    | An `Origin` other than `localhost`, `127.0.0.1` or `::1` is refused with HTTP 403                       |
| CORS      | Enabled on every response                                                                               |

The server reads the revision from the request itself. A request is
`2026-07-28` when its `params._meta` carries
`io.modelcontextprotocol/protocolVersion`, or when its `MCP-Protocol-Version`
header names `2026-07-28` or later. Every other request is `2025-06-18`.

To learn what the server speaks before sending anything else, call
`server/discover`. It needs no session in either revision and answers with
`supportedVersions`, the server's `capabilities` and its `instructions`.

:::{admonition} Under the hood
:class: toggle

**A `2026-07-28` request**

There is no handshake and no session: every request is self-contained. It is a
`POST /mcp` whose body carries the protocol version and the client's
capabilities in `params._meta`, and whose headers repeat what the body says:

| Where          | What                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| `params._meta` | `io.modelcontextprotocol/protocolVersion`: `"2026-07-28"` (required)                                    |
| `params._meta` | `io.modelcontextprotocol/clientCapabilities`: an object, `{}` at least (required)                       |
| Header         | `MCP-Protocol-Version`: the same version as the body                                                    |
| Header         | `Mcp-Method`: the request's `method`                                                                    |
| Header         | `Mcp-Name`: the tool name for `tools/call`, the URI for `resources/read`, the prompt name for `prompts/get` |
| Header         | `Accept`: when sent, it must allow `application/json`; add `text/event-stream` to let a slow tool call answer on a stream |

A request missing either `_meta` field is refused with error `-32602`. A
header that is missing or disagrees with the body is refused with `-32020`. A
version this revision does not serve is refused with `-32022`, and the error's
`data.supported` lists the versions to retry with. An `Mcp-Name` that is not
plain ASCII is sent as `=?base64?<Base64 of its UTF-8>?=`.

A result carries `resultType: "complete"` and names the server under `_meta`.
The results of `server/discover`, the four list methods (`tools/list`,
`prompts/list`, `resources/list` and `resources/templates/list`) and
`resources/read` also carry the caching hints `ttlMs` and `cacheScope`.

**A `2025-06-18` session**

The client sends `initialize` first. The response names protocol version
`2025-06-18` and carries an `Mcp-Session-Id` header, and every later request
sends that id back. A request with a missing or unknown id is refused with HTTP
400, except `server/discover`, which needs no session. A `GET /mcp` with
`Accept: text/event-stream` and the session id opens a stream that carries only
keep-alive comments, since the server never pushes events, and a `DELETE /mcp`
with the id ends the session. Without a session, `GET` and `DELETE` are
answered with HTTP 405. Results in this revision carry none of the fields
listed above for `2026-07-28`, except the result of `server/discover`, which
carries them in both revisions.

**Slow tool calls**

A `tools/call` that finishes within about a second is answered with a single
JSON response. A slower one, when the request's `Accept` allows
`text/event-stream`, is answered on an event stream. The result is the last
event, preceded every two seconds by a `notifications/progress` event when
the request's `params._meta` carried a `progressToken`, or by a keep-alive
comment otherwise. A client that does not accept a stream waits on the
connection.

On an event stream, closing the connection cancels the call. A client that
does not accept a stream cannot cancel a call this way, because the server
notices the disconnect only when it sends the result. A call still running
after 900 seconds is abandoned and answered with an error result, though the
handler may still be running on Blender's main thread.

**Errors**

A protocol failure is a JSON-RPC error carrying the HTTP status that belongs
to it: `400` for a malformed request, invalid parameters (including an unknown
tool name), a header mismatch or an unsupported version, `404` for an unknown
method, and `500` for an internal error. A body that is not JSON, a JSON-RPC
batch and a body over 10 MB are refused with `400`. A failure inside a tool is
not a protocol error: the call succeeds with `isError` set.

Four refusals happen before any JSON-RPC handling and are plain-text statuses:
`403` for a disallowed `Origin`, `404` for any path other than `/mcp`, `405`
for a `GET` or `DELETE` without a session, and `406` for an `Accept` header
that does not allow `application/json`.

**Thread model**

The server is a `ThreadingHTTPServer`. Its accept loop runs on one daemon
thread and passes each accepted request to a worker thread. A worker cannot
change `bpy` data itself: only Blender's main thread may, because a change from
another thread would race the UI. A worker therefore queues the tool call, and
Blender's persistent timer callback takes the queue and runs each handler on
the main thread.

```
  HTTP request comes in
      -> ThreadingHTTPServer hands it to a per-request worker thread
      -> worker thread enqueues the tool call on the main-thread task queue
      -> Blender's timer callback drains the queue and runs the tool
      -> worker thread resumes when the result is ready, serializes it and
         returns the HTTP response
```

Practical consequences:

- A thread per HTTP request means the blocking `GET /mcp` keep-alive stream
  of a `2025-06-18` session does not stall other requests.
- Handlers run one at a time on the main thread, so two `tools/call` requests
  never change the scene together.
- A handler runs from Blender's timer callback and can call `bpy.ops.*`.
- A handler should not block for long. Long-running work (transfer, run,
  fetch, bake) runs asynchronously in the add-on, and the tool call starts it
  and returns.

**Example: calling a tool**

```bash
curl -s -X POST http://localhost:9633/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/call' \
  -H 'Mcp-Name: run_python_script' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"run_python_script",
                 "arguments":{"code":"import bpy; print(bpy.app.version_string)"},
                 "_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28",
                          "io.modelcontextprotocol/clientCapabilities":{}}}}'
```

**Example: reading the scene resource**

`resources/read` takes the same `_meta` and headers as above, with the header
`Mcp-Method: resources/read`, the header `Mcp-Name: blender://scene/current`
and `params` of `{"uri": "blender://scene/current", "_meta": {...}}`. The
response is a JSON-RPC envelope whose `result.contents[0].text` holds the scene
as JSON. The scene changes between reads, so its caching hint is `ttlMs: 0`:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "contents": [
      {
        "uri": "blender://scene/current",
        "mimeType": "application/json",
        "text": "{...}"
      }
    ],
    "resultType": "complete",
    "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "zozo_contact_solver", "version": "..."}},
    "ttlMs": 0,
    "cacheScope": "private"
  }
}
```
:::

## See Also

- [Scene Setup via MCP](mcp_scene_setup.md): the order and the checks an
  agent follows when it builds a scene.
- [MCP Tool Reference](./mcp_reference.rst): every tool, its parameters and
  its description.
- [Blender Python API](./python_api.md): the same actions, called from
  Blender's text editor instead of over HTTP.
- [Security](../security.md#code-execution-risk-mcp): the code-execution risk
  of the MCP and debug ports.
- [Troubleshooting](../troubleshooting.md#mcp-server): port conflicts and
  start failures.
