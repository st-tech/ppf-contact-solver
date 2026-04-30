# 🤖 MCP Server

The add-on bundles a [Model Context Protocol](https://modelcontextprotocol.io/)
server that exposes nearly every operation (connecting to hosts, creating
groups, running a simulation, capturing the viewport, running arbitrary
Python) as MCP tools. External agents (Claude Desktop, IDE plugins,
automation scripts, CI runners) drive Blender and the solver through a
Streamable HTTP JSON-RPC surface instead of scripting the UI.

```{figure} ../images/integrations/entry_points.svg
:alt: Layered block diagram. Top row shows three client categories: AI agents and automation, the human operator clicking in the Blender sidebar, and the Python user scripting the add-on from Blender's text editor or a notebook. Second row shows the three protocol surfaces each client lands on: the MCP Streamable HTTP server on localhost:9633 (with a pill warning "localhost only, do not expose"); Blender operators identified by their bl_idname; and the zozo_contact_solver Python module that forwards unknown attributes to the matching operator. All three protocol boxes funnel into a single shared add-on core row labeled scene.zozo_contact_solver, described as the place where pins, merges, colliders, and every scene mutation run through the same validation. Below that, one wide transport row stands in for the five connection types (Local, SSH, Docker, Docker over SSH, Windows Native), feeding into server.py :PORT at the very bottom.
:width: 820px

The MCP server sits beside two sibling entry points: the Blender
sidebar and the `zozo_contact_solver` Python module. All three cover
the same surface, land on the same validation layer, and share the
same transport to `server.py`. An agent calling `tools/call` hits the
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

Main panel → **MCP Settings** → **Start**.

```{figure} ../images/integrations/mcp_row.png
:alt: MCP Server section inside the Solver panel
:width: 500px

The MCP Server section, expanded. **Start** launches the HTTP server on
the port shown to the right; while running, this button swaps to
**Stop** and the port field becomes read-only.
```

The server binds to the **MCP Port** on `localhost` (default `9633`).
If the port is busy, it tries `port+1`, `port+2`, … up to `port+9`
and prints the chosen port to the Blender console. **Stop** shuts the
HTTP listener down and drains the task queue.

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

using the **Streamable HTTP** transport (protocol version `2024-11-05`).
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
| Version      | `2024-11-05`                                                      |
| Transport    | Streamable HTTP on a single `/mcp` endpoint                       |
| Requests     | `POST /mcp` with a JSON-RPC message                               |
| Server push  | `GET /mcp` with `Accept: text/event-stream` (one-shot init event) |
| CORS         | Enabled on every response                                         |

All traffic goes through `/mcp`. The client calls `initialize` first; the
server replies with the negotiated `protocolVersion` and capabilities. The
JSON-RPC surface itself is the standard MCP set: `initialize`, `tools/list`,
`tools/call`, `resources/list`, `resources/read`.

## Exposed Tools

The authoritative list, with every tool name, its parameters, and its
description, lives at [MCP Tool Reference](./mcp_reference.rst).
That page is regenerated from the handler sources at every docs build,
so it cannot drift. For a live, schema-attached enumeration against a
running server, use `tools/list` (or the CLI `tools` subcommand).

Tool descriptions returned by `tools/list` are taken directly from the
function docstrings registered via the handler decorators.

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

CLI options: `--host`, `--mcp-port`, `--timeout`. Run
`python blender_addon/debug/main.py --help` for the full subcommand
surface.

## Calling a Tool over HTTP

If you are integrating from something that is not the bundled CLI, drive the
HTTP transport directly. The server is stateless — every request is
self-contained JSON-RPC; there is no session token to manage.

```bash
HDR_ACCEPT='Accept: application/json, text/event-stream'
HDR_JSON='Content-Type: application/json'

# 1. Initialize.
curl -s -X POST http://localhost:9633/mcp \
  -H "$HDR_JSON" -H "$HDR_ACCEPT" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
       "params":{"protocolVersion":"2024-11-05","capabilities":{},
                 "clientInfo":{"name":"example","version":"0"}}}'

# 2. Call a tool.
curl -s -X POST http://localhost:9633/mcp \
  -H "$HDR_JSON" -H "$HDR_ACCEPT" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call",
       "params":{"name":"run_python_script",
                 "arguments":{"code":"import bpy; print(bpy.app.version_string)"}}}'
```

POST bodies over 10 MB are rejected with HTTP 413.

## Resources

The MCP server exposes one resource via `resources/list` and
`resources/read`:

| URI                       | Content                                                         |
| ------------------------- | --------------------------------------------------------------- |
| `blender://scene/current` | Live JSON snapshot of the current Blender scene. Refreshed on every read. |

### Enumerating Resources

```bash
curl -s -X POST http://localhost:9633/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"resources/list"}'
```

### Reading a Resource

```bash
curl -s -X POST http://localhost:9633/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "resources/read",
    "params": {"uri": "blender://scene/current"}
  }'
```

The response is a JSON-RPC envelope whose `result.contents[0].text`
holds the JSON body:

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
    ]
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

The server is a plain `HTTPServer` running a `handle_request()` loop on
one daemon thread. It has to stay off Blender's main thread, because
Blender owns the main thread. Mutating `bpy.*` from the server thread
would race the UI and corrupt scene state, so tool dispatch marshals back
onto Blender's tick via a task queue.

```
  HTTP request comes in
      -> HTTPServer handles the request on its dedicated server thread
      -> server thread enqueues the tool call on the main-thread task queue
      -> Blender's main-thread tick drains the queue and runs the tool
      -> server thread wakes, serializes the result, returns HTTP response
```

Practical consequences:

- The server processes one HTTP request at a time; it does not spawn a
  new thread per request.
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
