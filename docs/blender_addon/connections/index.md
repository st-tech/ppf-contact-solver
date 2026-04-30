# 🔌 Connections

The add-on talks to a solver process over one of several transports. Pick the
one that matches where the solver runs.

| Type | Use when | See |
| ---- | -------- | --- |
| **Docker over SSH** / **Docker over SSH Command** | Solver runs inside a container on a remote Docker host | [Docker over SSH](docker_over_ssh.md) |
| **Docker** | Solver runs inside a container on the local Docker daemon | [Docker (Local)](docker.md) |
| **Windows Native** | Solver runs as a Windows subprocess using a bundled Python + CUDA | [Windows](windows.md) |
| **SSH** / **SSH Command** | Solver runs **directly** on a remote Linux host | [SSH (Direct)](ssh.md) |
| **Local** | Solver runs on the same Linux machine as Blender | [Local](local.md) |

All types share the same server-side protocol (TCP, version `0.02`) and the
same UI flow in the panel: **Connect** -> **Start Server** -> transfer data
-> **Run** -> **Fetch**.

```{figure} ../images/connections/transport_topologies.svg
:alt: Five stacked block diagrams, one per connection type. Top to bottom: Local (Linux) places the Blender add-on and server.py on a single Linux workstation with a loopback TCP socket between them; SSH / SSH Command keeps the add-on on the workstation and reaches server.py on a remote Linux host over a paramiko SSH session carrying both lifecycle commands and an SSH tunnel to server.py; Docker (local) adds a Docker daemon between the add-on and a container on the same workstation, with server traffic bypassing the daemon straight to the container's published port on localhost; Docker over SSH / over SSH Command pushes the daemon and container onto a remote Linux host reached through one SSH session; Windows Native launches server.py as a hidden Win32 subprocess on the same Windows workstation. Blue solid arrows carry lifecycle commands, purple dashed arrows carry TCP traffic to server.py.
:width: 820px

Where each piece lives, and how the add-on reaches it, for the five
connection types. Blue solid arrows carry lifecycle commands
(start / stop / exec / port check); purple dashed arrows carry the TCP
connection to `server.py`. The three Docker sub-modes of row 3 are
broken out separately in [Docker (Local)](docker.md) and
[Docker over SSH](docker_over_ssh.md).
```

## What Happens When You Connect

Starting a solver session is two button presses:

1. **Connect** opens the transport to wherever the solver lives (a local
   process, an SSH session, or a Docker container) and checks that a
   `server.py` is actually present at the path you configured. If the
   check fails, an error is shown in the panel; the transport stays open
   so you can correct the path without reconnecting.
2. **Start Server** launches `server.py` on the remote side and waits up
   to **16 seconds** for it to report that it is ready to accept work. If
   the server prints an error during startup, the panel reports the error
   immediately; if it simply never becomes ready, you get a timeout and a
   tail of its log.

The panel stays responsive while this happens -- the actual work runs on
a background thread and the UI polls it several times a second.

Windows Native is slightly different: it launches the server directly as
part of the Connect step, so **Start Server** is effectively a no-op on
that backend.

:::{tip}
**Save the connection once it works.** As soon as you have a successful
**Connect**, click the **Save** icon on the profile row to write the
current fields to a `.toml` file. Next session, **Open** the file, pick
the entry, and every field auto-fills, with no retyping host, key path,
container, or port. See [Connection Profiles](profiles.md) for the full
workflow.
:::

## Port Usage at a Glance

| Port | Role | Default |
| ---- | ---- | ------- |
| Server | Solver TCP listener (`server.py`) | `9090` |
| MCP | MCP Streamable HTTP server (for AI integration) | `9633` |

Only the server port crosses the transport boundary; MCP is local to the
machine running Blender. The server port is configurable per connection.

:::{note}
For Docker-over-SSH, the server port must be published on the container
(`-p 9090:9090`). The add-on checks this before launching the server and
refuses to continue if the port is not exposed.
:::

(connections-under-the-hood)=

:::{admonition} Under the hood
:class: toggle

**Non-blocking UI**

The panel does not freeze while Connect or Start Server is running.
Work happens on a background thread and the panel refreshes several
times a second, so any status or error reported by the background work
appears in the panel promptly.

**Connect step**

Connect opens the configured transport (SSH session, Docker client, or a
local subprocess) and verifies that `server.py` is present at the
configured path. If that verification fails, the error is reported in
the panel but the transport remains open.

**Start Server step**

On Unix-family backends (Local, SSH, Docker, Docker over SSH) the
add-on launches the server via a small script that activates
`$HOME/.local/share/ppf-cts/venv` if it exists and then runs:

```sh
nohup python3 server.py --port <port> > server.log 2>&1 &
```

The UI waits up to **16 seconds** for the server to announce it is
ready. If a line containing `ERROR` or `FAILED` appears first, the wait
aborts with that message; on plain timeout, the panel prints the last 20
lines of `server.log`.

Windows Native launches the server directly during the Connect step.
See {ref}`Windows - Under the hood <windows-under-the-hood>`.

**Docker port pre-launch check**

Before **Start Server** on Docker-over-SSH, the add-on checks that the
configured server port is published on the container. If it is not, the
operator aborts with:

> Docker port 9090 is not exposed on container 'ppf-dev'. Please expose
> the port with '-p 9090:9090' when starting the container.

The add-on cannot publish a port on an existing container; this has to
be fixed on the container side (for example by re-running `docker run
-p` or editing `compose.yaml`).
:::
