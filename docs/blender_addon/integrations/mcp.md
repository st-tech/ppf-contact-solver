# 🤖 MCP Server

The add-on bundles a [Model Context Protocol](https://modelcontextprotocol.io/)
server that exposes nearly every operation (connecting to hosts, creating
groups, running a simulation, capturing the viewport, running arbitrary
Python) as MCP tools. External agents (Claude Desktop, IDE plugins,
automation scripts, CI runners) drive Blender and the solver through a
Streamable HTTP JSON-RPC surface instead of scripting the UI.

```{figure} ../images/integrations/entry_points.svg
:alt: Layered block diagram. Top row shows three client categories: AI agents and automation, the human operator clicking in the Blender sidebar, and the Python user scripting the add-on from Blender's text editor or a notebook. Second row shows the three protocol surfaces each client lands on: the MCP Streamable HTTP server on localhost:9633 (with a pill warning "localhost only, do not expose"); Blender operators identified by their bl_idname; and the add-on's Python API (exposed under bl_ext.user_default.ppf_contact_solver.ops.api) that forwards unknown attributes to the matching operator. All three protocol boxes funnel into a single shared add-on core row labeled scene.zozo_contact_solver, described as the place where pins, merges, colliders, and every scene mutation run through the same validation. Below that, one wide transport row stands in for the six connection types (SSH, Docker, Docker over SSH, and the Windows, macOS, and Linux Native types), feeding into ppf-cts-server :PORT at the very bottom.
:width: 820px

The MCP server sits beside two sibling entry points: the Blender
sidebar and the add-on's Python API (exposed under
`bl_ext.user_default.ppf_contact_solver.ops.api`). All three cover
the same surface, land on the same validation layer, and share the
same transport to `ppf-cts-server`. An agent calling `tools/call` hits the
same operator a human hits by clicking the button. The "localhost
only" pill on the MCP box is the single security boundary this stack
relies on; see [Security](#security) below.
```

## What MCP Gives You

- A stable JSON surface that does not care how the add-on's buttons are
  laid out this week.
- The same validation the UI uses: pins, merges, and colliders all go
  through the add-on's shared mutation layer, so a misbehaving agent
  gets the same errors a user would.
- One `run_python_script` tool if you genuinely need raw `bpy.*` access,
  and one `execute_shell_command` tool for provisioning remote hosts.

## Starting the Server

Solver panel → **MCP Server** → **Start MCP Server on Local**.

```{figure} ../images/integrations/mcp_row.png
:alt: MCP Server section inside the Solver panel
:width: 500px

The MCP Server section, expanded. **Start MCP Server on Local** launches
the HTTP server on the port shown to the right; while running, this
button swaps to **Stop MCP Server** and the port field becomes
read-only. The collapsed header reads `MCP Server (Stopped)` or
`MCP Server (Running :<port>)` to reflect the current state.
```

The server binds to the **MCP Port** on `localhost` (default `9633`).
If the port is busy it first waits for it to be released, retrying five
times with a backoff, and only then walks `port+1`, `port+2`, … up to
`port+9`. A substituted port is reported as a warning and written back
into the **MCP Port** field, so the `MCP Server (Running :<port>)`
header always names the live port. **Stop MCP Server** shuts the HTTP
listener down and closes any live MCP session.

:::{warning}
The server binds to `localhost` only. Do **not** port-forward it or bind
it to `0.0.0.0`. `run_python_script` and `execute_shell_command` are
remote code execution by design: anyone who can reach the port has
full control of Blender and the machine it runs on.
:::

## Adding the Server to an MCP Client

Once the server is running, point your MCP client at

```
http://localhost:9633/mcp
```

using the **Streamable HTTP** transport. The server answers protocol
versions `2026-07-28` and `2025-06-18` on that one URL, so a client of
either revision connects without further setup (see [Protocol](#protocol)).
If the default port was busy and the add-on fell back to `9634`, `9635`,
and so on, use the port printed to the Blender console.

For Claude Code, run:

```bash
claude mcp add --transport http zozo-contact-solver http://localhost:9633/mcp
```

For clients configured through a JSON config file (Claude Desktop,
Cursor, Windsurf, and similar), add an entry like:

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

Restart the client after editing its config so it picks up the new
server. If the client only supports stdio-based MCP servers, use a
generic Streamable HTTP bridge (for example
[`mcp-remote`](https://www.npmjs.com/package/mcp-remote)) to wrap the
endpoint.

## Security

`run_python_script` evaluates arbitrary Python inside Blender.
`execute_shell_command` runs arbitrary shell. There is no sandboxing,
no allowlist, no auth. This is intentional: it is the escape hatch
that lets agents do anything the add-on cannot yet express.

Rules of the road:

- **Bind localhost only.** The server already does; do not change that.
- **Do not expose the port** through `ssh -R`, `ngrok`,
  `gh codespaces`, or any reverse proxy unless you have decided the
  machine is disposable.
- **Treat prompts as untrusted.** If you pipe unsanitized LLM output
  into `run_python_script`, you have given the LLM shell. Audit its
  tool calls.

## Protocol

| Property     | Value                                                             |
| ------------ | ----------------------------------------------------------------- |
| Versions     | `2026-07-28` (stateless) and `2025-06-18` (session-bound), both on one endpoint |
| Transport    | Streamable HTTP on a single `/mcp` endpoint                       |
| Requests     | `POST /mcp`, one JSON-RPC request or notification per POST (no batches) |
| Origin       | Requests from a browser origin other than `localhost`, `127.0.0.1` or `::1` are refused with HTTP 403 |
| CORS         | Enabled on every response                                         |

All traffic goes through `/mcp`. The server reads which revision a
request is written in from the request itself, so the two need no
configuration and one running server answers clients of both:

- **`2026-07-28`, stateless.** A request is modern when its
  `params._meta` carries `io.modelcontextprotocol/protocolVersion`, or
  its `MCP-Protocol-Version` header names `2026-07-28` or later. There
  is no handshake and no session: every request stands alone.
- **`2025-06-18`, session-bound.** Every other request. The client
  opens with `initialize`, and the response carries an `Mcp-Session-Id`
  header.

The methods are the same in both: `server/discover`, `tools/list`,
`tools/call`, `resources/list`, `resources/templates/list` (always
empty), `resources/read`, `prompts/list` and `prompts/get`, plus
`initialize` for the session-bound revision.

### A `2026-07-28` Client

Every request is a `POST /mcp` whose body carries the protocol version
and the client's capabilities in `params._meta`, and whose headers
repeat what the body says:

| Where            | What                                                                          |
| ---------------- | ----------------------------------------------------------------------------- |
| `params._meta`   | `io.modelcontextprotocol/protocolVersion`: `"2026-07-28"` (required)           |
| `params._meta`   | `io.modelcontextprotocol/clientCapabilities`: an object, `{}` at least (required) |
| Header           | `MCP-Protocol-Version`: the same version as the body                          |
| Header           | `Mcp-Method`: the request's `method`                                          |
| Header           | `Mcp-Name`: the tool name for `tools/call`, the URI for `resources/read`, the prompt name for `prompts/get` |
| Header           | `Accept`: when sent, it must allow `application/json`; add `text/event-stream` to let a slow tool call answer on a stream |

A request missing either `_meta` field is refused with error `-32602`,
and a header that is missing or disagrees with the body with `-32020`.
A version this revision does not serve is refused with
`-32022`, and the error's `data.supported` lists the versions to retry
with. An `Mcp-Name` that is not plain ASCII is sent as
`=?base64?<Base64 of its UTF-8>?=`.

To learn what the server speaks before sending anything else, call
`server/discover`. It needs no session in either revision and answers
with `supportedVersions`, the server's `capabilities` and its
`instructions`.

A modern result carries `resultType: "complete"` and names the server
under `_meta`. The results of `server/discover`, the three list methods
and `resources/read` also carry caching hints, `ttlMs` and
`cacheScope`. No list is ever paged, so a request carrying a `cursor`
is refused.

### A `2025-06-18` Client

The client sends `initialize` first. The response names protocol
version `2025-06-18` and carries an `Mcp-Session-Id` header, and every
later request sends that id back; a request with a missing or unknown
id is refused with HTTP 400. A `GET /mcp` with `Accept:
text/event-stream` and the session id opens a stream that carries only
keep-alive comments, since the server never pushes events, and a
`DELETE /mcp` with the id ends the session. Without a session, `GET`
and `DELETE` are answered with HTTP 405. Results in this revision carry
none of the modern fields above.

### Tool Calls and Long-Running Tools

A `tools/call` that finishes within about a second is answered with a
single JSON response. A slower one, when the request's `Accept` allows
`text/event-stream`, is answered on an event stream: the result arrives
as the stream's last event, preceded every two seconds by a
`notifications/progress` event when the request's `params._meta`
carried a `progressToken`, or by a keep-alive comment otherwise. A
client that does not accept a stream simply waits on the connection.
Closing the connection cancels the call. A call still running after 900
seconds is abandoned and answered with an error result, though the
handler may still be running on Blender's main thread.

The result's `content` holds the tool's return value as JSON text, and
`structuredContent` holds the same value parsed. A tool that reports a
failure sets `isError`, so a failure never arrives as a successful
result.

### Errors

Every failure is a JSON-RPC error carrying the HTTP status that belongs
to it: `400` for a malformed request, invalid parameters, a header
mismatch or an unsupported version, `404` for an unknown method, and
`500` for an internal error. A body that is not JSON, a JSON-RPC batch,
and a body over 10 MB are refused with `400`; an `Accept` header that
does not allow `application/json` with `406`; and any path other than
`/mcp` with `404`.

## Exposed Tools

The authoritative list, with every tool name, its parameters, and its
description, lives at [MCP Tool Reference](./mcp_reference.rst).
That page is regenerated from the handler sources at every docs build,
so it cannot drift. For a live, schema-attached enumeration against a
running server, use `tools/list` (or the CLI `tools` subcommand).

Tool descriptions returned by `tools/list` are the function docstrings
registered via the handler decorators, exactly as the handler declares
them. A tool's description and input schema are its whole reference:
they state what the tool needs, its units, and what it refuses, and
they come from the code that runs, so they describe that tool and no
other. Do not match on the docstring text verbatim.

## Instructions and Prompts

The server's instructions, returned by `server/discover`, tell an agent
to read each tool's description and input schema before calling it, and
give the order a scene is built in: connect, create a group, assign
objects, constrain, set parameters, build, solve, fetch.

`prompts/list` and `prompts/get` offer four prompts, each a short
starting point that names the ordered steps of one workflow and leaves
the detail to the tools' own descriptions:

| Prompt              | Arguments                                   | Workflow                                                         |
| ------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| `run_simulation`    | `backend`, `frames` (both optional)         | Take a scene from unconfigured to fetched results.               |
| `pin_and_constrain` | `object_name`, `intent` (optional)          | Hold or drive part of a mesh with pins, colliders, or merges.    |
| `tune_parameters`   | `goal`                                      | Choose parameters for a goal and read back what the solver got.  |
| `diagnose_failure`  | `symptom`                                   | Work from a failed or stalled solve's symptom to its cause.      |

## Calling a Tool from the CLI

The debug CLI at `blender_addon/debug/main.py` wraps the MCP client,
which is the fastest way to poke at the server:

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

Global options are `--host` and `--mcp-port`. `--timeout` is
per-subcommand: on `call` it is the request timeout (default 30s), on
`runtests` the per-scenario timeout (default 60s). Run
`python blender_addon/debug/main.py --help` for the full subcommand
surface.

## Calling a Tool over HTTP

If you are integrating from something that is not the bundled CLI, drive the
HTTP transport directly. A `2026-07-28` request needs no handshake: send
the version and the client's capabilities in `params._meta`, and repeat
the version, the method and the tool name in the headers.

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

## Resources

The MCP server exposes one resource via `resources/list` and
`resources/read`:

| URI                       | Content                                                                   |
| ------------------------- | ------------------------------------------------------------------------- |
| `blender://scene/current` | Live JSON snapshot of the current Blender scene. Refreshed on every read. |

It serves no documentation as resources: what a tool does is in its
description and input schema from `tools/list`, and the pages of this
guide cover the rest.

### Enumerating Resources

```bash
curl -s -X POST http://localhost:9633/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'MCP-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: resources/list' \
  -d '{"jsonrpc":"2.0","id":2,"method":"resources/list",
       "params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28",
                          "io.modelcontextprotocol/clientCapabilities":{}}}}'
```

### Reading a Resource

```bash
curl -s -X POST http://localhost:9633/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'MCP-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: resources/read' \
  -H 'Mcp-Name: blender://scene/current' \
  -d '{"jsonrpc":"2.0","id":3,"method":"resources/read",
       "params":{"uri":"blender://scene/current",
                 "_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28",
                          "io.modelcontextprotocol/clientCapabilities":{}}}}'
```

The response is a JSON-RPC envelope whose `result.contents[0].text`
holds the JSON body. The scene changes between reads, so its caching
hint is `ttlMs: 0`:

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

Unknown URIs return a JSON-RPC error with code `-32602`.

## See Also

- [Blender Python API](./python_api.md): the same surface, but called
  from Blender's text editor instead of over HTTP.

:::{admonition} Under the hood
:class: toggle

**Thread model**

The server is a `ThreadingHTTPServer` running its `handle_request()`
accept loop on one daemon thread and handing each accepted request to a
worker thread. It has to stay off Blender's main thread, because
Blender owns the main thread. Mutating `bpy.*` from the server thread
would race the UI and corrupt scene state, so tool dispatch marshals back
onto Blender's tick via a task queue.

```
  HTTP request comes in
      -> ThreadingHTTPServer hands it to a per-request worker thread
      -> worker thread enqueues the tool call on the main-thread task queue
      -> Blender's main-thread tick drains the queue and runs the tool
      -> worker thread wakes, serializes the result, returns HTTP response
```

Practical consequences:

- The server spawns a thread per HTTP request, so the blocking
  `GET /mcp` keep-alive stream does not stall other requests.
- Every tool call serializes through Blender's main thread, so two
  `tools/call` requests cannot mutate the scene at the same time. That
  is what keeps the validation layer coherent.
- Handlers see a fully valid `bpy.context` and can call `bpy.ops.*`.
- Handler callbacks should not block for long. Long-running work
  (simulate, transfer) runs asynchronously in the add-on; the MCP call
  just kicks it off and returns.
- Because the queue is drained on Blender's tick, tool dispatch is
  effectively paused while Blender is modal (popups, file dialogs).
:::
