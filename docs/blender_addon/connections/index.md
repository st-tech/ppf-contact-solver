# 🔌 Connections

Pick the connection **Type** that matches where the solver runs. Each type
has its own page.

| Type | Use when | See |
| ---- | -------- | --- |
| **Windows Native** | Solver runs as a subprocess on this Windows machine, on CUDA, ROCm, or the CPU build | [Windows](windows.md) |
| **macOS Native** | Solver runs as a subprocess on this Apple-silicon Mac, on Metal or the CPU build | [macOS](macos.md) |
| **Linux Native** | Solver runs as a subprocess on this Linux machine, on CUDA, ROCm, or the CPU build | [Linux](linux.md) |
| **Docker** | Solver runs inside a container on the Docker daemon of this machine | [Docker (Local)](docker.md) |
| **Docker over SSH** / **Docker over SSH Command** | Solver runs inside a container on a remote Docker host | [Docker over SSH](docker_over_ssh.md) |
| **SSH** / **SSH Command** | Solver runs **directly** on a remote Linux host | [SSH (Direct)](ssh.md) |

The first three rows are the **native** types: the add-on starts the solver
as a child process on the machine Blender runs on, and only the type for
your own platform works. The last three rows are the **remote** types (the
**Docker** type reaches a container on this machine, the other two a remote
one): the solver runs in a container or on another machine, and the add-on
reaches it through the Docker daemon or over SSH. The **solver host** below
is the machine or container that runs `ppf-cts-server`: this machine for a
native type, the remote host for **SSH** and **SSH Command**, and the
container for the Docker types.

Every type uses the same flow in the panel: **Connect** ->
**Start Server on Remote** -> **Transfer** -> **Run** ->
**Fetch All Animation**.

Below, this page covers what happens when you connect, how the server stops,
the ports in use, [choosing the build](#choosing-the-build-compute-device-and-gpu-backend)
(**Compute Device** and **GPU Backend**), [picking a GPU](#picking-a-gpu) and
the [Remote Hardware](#remote-hardware) readout.

```{toctree}
:maxdepth: 1
:hidden:

windows
macos
linux
docker
docker_over_ssh
ssh
profiles
```

```{figure} ../images/connections/transport_topologies.svg
:alt: Five rows of block diagrams, one per connection layout, from top to bottom: Linux Native and macOS Native; SSH and SSH Command; Docker; Docker over SSH and Docker over SSH Command; Windows Native. Blue solid arrows carry lifecycle commands and purple dashed arrows carry TCP traffic to ppf-cts-server.
:width: 820px

Where each piece runs, for all eight connection types, and how the
add-on reaches it. Blue solid arrows carry lifecycle commands (start,
stop, exec, port check); purple dashed arrows carry the TCP connection to
the `ppf-cts-server` binary. The three Docker types of rows 3 and 4 are
broken out in [Docker (Local)](docker.md) and
[Docker over SSH](docker_over_ssh.md). The three native types share one
shape: a child process on the same machine, reached over a loopback
socket. Rows 1 and 5 differ only in how the process is started.
```

## What Happens When You Connect

Starting a solver session takes two button presses. The panel stays usable
while either one runs.

1. **Connect** opens the connection to wherever the solver runs (a local
   process, an SSH session, or a Docker container) and checks the solver
   folder you set:
   - On a native type, it checks that **Solver Path** holds the build
     selected under **Compute Device**. If it does not, the connection is
     refused, you stay disconnected, and the panel shows the error. Correct
     the path and press **Connect** again.
   - On the remote types, it checks that the path holds a solver in any
     layout. If it does not, the panel shows an error but the connection
     stays open. The fields from **Type** through the path are grayed out
     while you are connected, so press **Disconnect**, correct the path, and
     press **Connect** again. Whether the host holds the build chosen under
     **Compute Device** is checked later, at **Start Server on Remote**.
   - If the host, the container, or the Docker daemon cannot be reached,
     **Connect** fails and the panel shows why. See
     [Troubleshooting](../troubleshooting.md).
2. **Start Server on Remote** starts `ppf-cts-server` on the solver host
   and waits up to **16 seconds** for it to report that it is ready to
   accept work. If the server reports an error while starting, the panel
   shows it at once. If the server never becomes ready, the panel reports
   a timeout: on the remote types it includes the last 20 lines of
   `server.log`, and on the native types it points you to `server.log` in
   the solver folder. A native type starts the server as a child process,
   or attaches to one that already listens on the port.

### Project Name

**Project Name** names the project that the solver host keeps for this
scene. The uploaded scene, its simulated frames, its saved states and its
cached results are stored in a folder of that name on the solver host. Two
Blender files that use the same name share one project, and a **Transfer**
from either starts that project over, removing the simulated frames and
saved states the other produced, so give each scene its own name. Connecting
later with the same name reaches the same project again, from the `.blend`
after you reopen it or from a notebook (see
[JupyterLab](../workflow/sim/jupyterlab.md)).

The session takes the name when you press **Connect**, and the field is
grayed out while the server is starting or running. If the name differs from
the session's afterward, **Transfer**, **Run**, **Resume** and **Update
Params on Remote** refuse until you press **Disconnect** and **Connect**
again (see
[Troubleshooting](../troubleshooting.md#project-name-out-of-sync-ui-but-active-session)).
When the `.blend` is saved and **Project Name** is empty or reads
`unnamed`, a copy icon beside the field fills in the `.blend` file name
without its extension.

(connect-unavailable)=

### When Connect Is Unavailable

**Connect** is grayed out until all of these hold:

| Requirement | What the panel shows when it is not met | Fix |
| ----------- | --------------------------------------- | --- |
| **Project Name** is not empty and holds only letters, digits, `.`, `-` and `_` | *Project name should not contain spaces or special characters* for a bad character; nothing for an empty name | Edit the name |
| The required field is filled: **Host** (types **SSH** and **Docker over SSH**), the **SSH Command** field (types **SSH Command** and **Docker over SSH Command**), **Container** (the three Docker types), **Solver Path** (the native types) | The empty field | Fill it in |
| **SSH Key** (**SSH**, **Docker over SSH**), **Remote Path** (**SSH**, **SSH Command**) and **Container Path** (the three Docker types) hold no space and no shell special character | *Path should not contain spaces or special characters* | Remove the character. On Windows the default **SSH Key** path starts in your user folder, which contains a space when your account name does: point the field at a copy of the key in a path without one |
| **Solver Path** on a native type holds no shell special character (spaces are allowed) | *Path should not contain special characters* | Remove the character |
| The Python module the type needs is installed: `paramiko` for the four SSH-based types, `docker` for **Docker** | A banner above the profile row and an install button | Click **Install Paramiko to Add-on Directory** or **Install Docker-Py to Add-on Directory** and wait for the install to finish |
| No connection attempt is running and you are not connected | While connecting, the button reads **Connecting...**; once connected, it is grayed out | Wait, press **Cancel**, or press **Disconnect** |

Shell special characters are `& | ; < > ( ) $ * ? [ ] { } # ! % ^`, the
double and single quote, and the backtick. The add-on does not expand a
leading `~` in **SSH Key**, **Remote Path**, **Container Path**, or
**Solver Path**, so write the full path in each. A `~` in **Remote Path** on
**SSH** and **SSH Command** can pass **Connect** and still make **Start
Server on Remote** fail.

### Canceling

While **Connect** or **Start Server on Remote** is in flight, its button
reads **Connecting...** or **Server Starting...** and is disabled, and a
**Cancel** button appears under the status line.

- **Cancel** during **Connecting...** abandons the attempt and returns the
  panel to disconnected. This helps most with a host that is not
  answering, where the handshake would otherwise wait until it times out.
- A connect that nobody cancels is abandoned the same way after
  **60 seconds** and reports *Connection timed out*. The next **Connect**
  starts cleanly.
- **Cancel** during **Server Starting...** stops the launch and the server,
  the same as **Stop Server on Remote**, and leaves you connected.

### Disconnecting and Stopping the Server

**Disconnect** closes the connection and leaves the server running, on
every connection type. A server started on an SSH host or in a container
keeps running there, and a native server is a separate process on this
machine. **Stop Server on Remote** is the button that ends it, and a server
that nobody uses also stops itself (see [Idle Server Stop](#idle-server-stop)).
The next **Connect** reaches the server again while it is still running.

**Disconnect** and **Stop Server on Remote** are grayed out while data is
moving, that is, while the status line reads *Data Sending...*, *Data
Receiving...*, *Fetching Animation...* or *Applying Downloaded Animation...*.
Press **Abort** under the progress bar, or wait for the transfer to finish;
both buttons are available again once the status line leaves those states. A
running solve does not gray them out; see
[Disconnecting While a Simulation Runs](../workflow/sim/simulating.md#disconnecting-while-a-simulation-runs).

:::{tip}
**Save the connection once it works.** After a successful **Connect** has
confirmed the fields, press **Disconnect**, then click the **Save Profile**
icon on the profile row to write them to a `.toml` file. The whole profile
row is disabled while a connection is live. In a later session, press
**Open Profile**, pick the entry, and every field fills in: no retyping
host, key path, container, or port. See
[Connection Profiles](profiles.md) for the full workflow.
:::

### Idle Server Stop

A server that nobody is using stops itself, so a forgotten host does not keep
running. A server is idle when no Blender is connected to it and no simulation
or build is running. Once it has been idle for **Stop Idle Server After (s)**
seconds, it stops. To use the host again, press **Start Server on Remote**.

Both settings are in the **Advanced** box of the Connection box, which is
collapsed by default:

| Setting | Default | What it does |
| ------- | ------- | ------------ |
| **Stop Idle Server After (s)** | `60` | Seconds of idleness after which the server stops itself. `0` keeps it running until you press **Stop Server on Remote**. It is applied when the server starts, so set it before you press **Start Server on Remote**. |
| **Ping Interval (s)** | `10` | Seconds between the pings that Blender sends to a server it is connected to. The pings keep the server from stopping while Blender sits idle. The range is 1 to 3600, and a change takes effect at once. |

**Ping Interval** must be shorter than **Stop Idle Server After (s)**, or the
server would stop under a connected Blender. When it is not, the box shows
*Ping Interval must be shorter than the idle time*, and **Start Server on
Remote** refuses with *Ping Interval (N s) must be shorter than Stop Idle
Server After (M s), or the server stops while connected*. A value of `0` for the
idle time turns the check off.

## Port Usage at a Glance

| Port | Role | Default |
| ---- | ---- | ------- |
| Server | Solver TCP listener (`ppf-cts-server`) | `9090` |
| MCP | MCP Streamable HTTP server (for AI integration) | `9633` |
| Debug server | **Add-on Local Debug Server** under **Debug Options**, off until you press **Start** | `8765` |
| JupyterLab | The **Port** field of the JupyterLab box | `8080` |

Only the server port reaches the solver host. The MCP and debug servers stay
on the machine running Blender, and the JupyterLab **Port** is only the
port that **Open URL** opens in your browser (see
[JupyterLab](../workflow/sim/jupyterlab.md) and
[Security](../security.md#where-the-trust-boundaries-are)).

The server port is the **Docker Port** field, which the panel draws on the
three Docker types (range 1024-65535). The other types draw no field for the
server port and use the value **Docker Port** holds, `9090` by default. The
**Port** field of **SSH** and **Docker over SSH** is the SSH login port
(default `22`), not the server port. A saved
[connection profile](profiles.md) sets the server port with the `docker_port`
key.

The connection reads the server port when you press **Connect**. To change
it, press **Disconnect**, edit **Docker Port**, and press **Connect** again.

(docker-port-publishing)=

### Publishing the Port for Docker

All three Docker types need the server port published on the container,
with the same number on both sides (`-p 9090:9090`): the add-on connects to
that port number on the Docker host. The add-on checks the mapping and
refuses to continue when the port is not published.

| Type | When it checks | Message |
| ---- | -------------- | ------- |
| **Docker** | At **Connect** | *Container 'NAME' does not publish port N to this machine, so the add-on cannot reach the solver inside it. Recreate it with '-p N:N'.* |
| **Docker over SSH**, **Docker over SSH Command** | At **Start Server on Remote** | *Docker port N is not exposed on container 'NAME'. Please expose the port with '-p N:N' when starting the container.* |

The add-on cannot publish a port on an existing container. Recreate the
container with `docker run -p N:N`, or add the mapping to your compose file.
**Docker** accepts a container on `--network host` as is, because such a
container publishes nothing. The Docker over SSH check has no such
exception.

:::{note}
`-p 9090:9090` makes the port reachable on every network interface of the
Docker host. See
[Network exposure of the solver port](../security.md#network-exposure-of-the-solver-port).
:::

## Port Already in Use

If the server port is already taken, the panel shows an error that names the
port, together with a **Force Terminate Process** button: *Port N is in use*
on a native type, or *Server port N is already in use on the remote host ...*
on a remote type.

On a native type, a listener that is a `ppf-cts-server` left over from an
earlier Blender session (for example after a Blender restart) is not an
error: the add-on detects it and attaches to it. The exception is a
server running a different build from the one selected, which is refused;
see {ref}`Attaching to a running server <attach-mismatch>`. Any other
listener on the port still produces the in-use error.

### What Force Terminate Process Ends

- **On a native type** that is not connected, or is attached to a server this
  add-on did not start, it ends every process listening on that port on this
  machine (on Windows, together with the processes they started). That
  includes a program that is not the solver, so read the status line under
  the button first (the table below lists what it says): it says whether the
  listener is a solver server or another program. Connected to a server this
  add-on started, the button ends only that server.
- **On a remote type** it ends only a `ppf-cts-server` that was started with
  `--port N`, on the solver host or in the container. A server started by
  hand without `--port` is not matched.

The remote launch error *Server port N is already in use on the remote
host ...* names a "Kill Process" button. That is the
**Force Terminate Process** button.

### When the Force Terminate Process row is drawn

The row appears under the error text, outside the collapsible Connection
box, so it stays visible after a refused **Connect**. It is drawn only
while one of three failures stands, and disappears with the failure:

- a **protocol version mismatch**, which is what a server left over from
  an older binary looks like after you update the add-on;
- **any error while not connected** (native types only: a remote type has
  no connection to act through), because **Stop Server on Remote** is
  reachable only through a connection;
- **any error that names a held port**, connected or not: *Port N is in
  use*, *A solver server is already running on port N*, and *Server port N
  is already in use on the remote host*.

A build or transfer refusal (a stray isolated vertex, an unusable rest
shape) is not one of them; those carry their own repair button.

One line under the button says what it would act on:

| Line | Types | Meaning |
| ---- | ----- | ------- |
| *A solver server is listening on port N* | Native | A `ppf-cts-server` answers on that port of this machine. |
| *Port N is held by another program* | Native | Something else has the port. |
| *Nothing is listening on port N* | Native | The port is free; the error is about something else. |
| *A solver server is listening on port N of the solver host* | Remote | The add-on has a running server on this connection. |
| *Cannot check port N from here: ...* | Remote | No server is running on this connection, and the add-on cannot inspect the port of another host. The button looks for a matching `ppf-cts-server` and ends it. |

On a remote type the row is drawn only **while connected**, because the
kill runs through the live connection.

On a native type, a *Port N is in use* error hides both the error and the
button once a `ppf-cts-server` answers on that port of this machine (for
example after a second **Connect** attached to it), so a healthy server is
not ended by mistake.

### What the Button Reports

When the button finishes, Blender's status bar says what it found. *(where)*
is *this machine* on a native type, *container NAME* on a Docker type, and
*the solver host* on **SSH** and **SSH Command**.

| Report | Meaning |
| ------ | ------- |
| *Killed pid N on port P (where).* followed by *Press Connect to start a new one.* (a native type that is not connected) or *Press Start Server to launch a new one.* (connected) | The listed processes were ended, and the sentence names the next step. |
| *... Still running after the kill: pid N.* (warning) | A listed process survived the forced end. |
| *No ppf-cts-server was listening on port P (where).* (warning), followed by the same next-step sentence | Nothing matched, so nothing was ended. |
| *Could not inspect port P (where): reason* (error) | The port could not be checked, for example because `lsof` (macOS and Linux), `netstat` (Windows) or, on a remote host, `pgrep` is missing. |
| *Force Terminate Process: the connection is busy, try again* (warning) | Something else started between the click and the kill. |

(choosing-the-build)=

## Choosing the Build: Compute Device and GPU Backend

A solver build is compiled for one backend: CUDA, ROCm, Metal, or CPU.
Choosing which one runs is therefore choosing which **build directory** the
server starts from, and a machine that has several keeps them in separate
directories:

| Directory | Holds |
| --------- | ----- |
| `target/release/` | A plain `cargo build --release` on a GPU tree, a CPU build (`--features cpu`) written to `target/release/` when no GPU build occupies it, and the GPU (Metal) build of the macOS distribution. |
| `target/cuda/release/`, `target/rocm/release/` | One per accelerator in the Windows and Linux distributions. `target/metal/release/` is also accepted. |
| `target/cpu/release/` | The portable CPU build, built beside a GPU one with `CARGO_TARGET_DIR=target/cpu cargo build --release --features cpu`. |

Two controls in the Connection box say which one to use:

- The **GPU** / **CPU** toggle, named **Compute Device** in tooltips and
  refusal messages and called that below. **GPU** (the default) means the
  accelerator of the machine that runs the solver (CUDA or ROCm on Windows
  and Linux, Metal on macOS). **CPU** means the portable build, which needs
  no GPU and is substantially slower. It is one setting shared by every
  connection type.
- **GPU Backend**: **Automatic** (the default), **CUDA**, or **ROCm**. It is
  drawn only while **Compute Device** is **GPU**, and only where there is
  something to choose: a folder or solver host holding **more than one** GPU
  build (an x64 Windows or x86_64 Linux distribution carries CUDA and ROCm
  together), or one where a saved choice names a build that is not there.
  **Automatic** takes the only GPU build present. With several, a native type
  takes the first whose solver reports a usable device, and a remote type
  takes the first in order, CUDA before ROCm.

Neither is written to a [connection profile](profiles.md); they keep the
value you last set.

From a script, set them as `native_device` and `native_gpu_backend` (see
[Connection Settings](../integrations/python_api.md#connection-settings)).
The MCP tools `connect_linux_native` and `connect_win_native` also take a
`gpu_backend` argument, `"AUTO"`, `"CUDA"` or `"ROCM"`.

### When the rows are drawn, and what they are based on

| Aspect | Native types | Remote types |
| ------ | ------------ | ------------ |
| Drawn | **Compute Device** in the Connection box before Connect, beside **Solver Path**; **GPU Backend** where there is a choice | Only **while connected** |
| Based on | The folders of this machine, as you edit **Solver Path** | One listing of the solver host or container, taken at **Connect** and kept |
| Editable | Before **Connect**, like the rest of the Connection box | While connected, until a server is running or starting |
| To change it | **Disconnect**, pick, **Connect** | **Stop Server on Remote**, pick, **Start Server on Remote** |

A native type can read the folders of this machine at any time, so its rows
sit with the path field, and **Connect** refuses a selection the folder
cannot serve. A remote type has to list the builds on the solver host, so the
add-on does that once, at **Connect**, and the rows appear after that,
outside the part of the box that **Connect** grays out. The choice is read when you
press **Start Server on Remote**, so a change made after Connect always
takes effect. To list the builds again after changing the host, use the
refresh button (see [Refresh GPU List](#refresh-gpu-list)).

On a native type, **Compute Device** is always drawn, even when only one
build is there, and a line says which build is missing. When the folder has
no CPU build, the panel shows the command that adds one beside the existing
GPU build:

```sh
CARGO_TARGET_DIR=target/cpu cargo build --release --features cpu
```

When the listing of the solver host's builds fails or finds nothing, the
**Compute Device** toggle is still drawn, with one of these lines:

| Line | Meaning and fix |
| ---- | --------------- |
| *The connection names no directory on the solver host, so its solver builds could not be listed.* | **Remote Path** or **Container Path** is empty. Disconnect, fill it in, and connect again. |
| *Could not list the solver builds on the solver host: ...* | The listing command could not run. Check the connection, then press the refresh button. |
| *Listing the solver builds on the solver host failed: ...* | The host answered with an error, which follows the colon. Fix it on the host, then press the refresh button. |
| *No solver build found under `<path>`* | The listing worked but the path holds no solver. Disconnect, correct the path, and connect again. |
| *The solver host has no `<device>` build here; it has `<other>`* | The path holds only the other device's build. Set **Compute Device** to that device, or build the missing one on the host and press the refresh button. |

### What determines which device a directory counts as

A CPU build in `target/release/`, which is what a plain
`cargo build --release --features cpu` produces, counts as **CPU**, not as
GPU: the add-on reads the device from the build itself, not from the
directory name.

### Refusals, never substitutions

A selection that cannot be served is refused by name. The add-on never runs
a different build from the one you selected; the message names what is
missing.

| Situation | What you see |
| --------- | ------------ |
| Folder holds the other device's build | *`<root>` holds the CPU build of the solver and no GPU build. Set Compute Device to CPU to run it, or point Solver Path at a folder that has a GPU build.* |
| Folder holds GPU builds, but not the named one | *`<root>` holds these GPU builds: cuda, and no rocm build. Set GPU Backend to one of those or to Automatic ...* |
| Folder holds no solver at all | *ppf-cts-server not found under `<root>` ...*, which names where the add-on looks for the solver |
| **Automatic**, several GPU builds, none with a usable device | *No GPU build in this folder has a usable device (cuda: ...; rocm: ...). Set Compute Device to CPU to run without a GPU.* |

On the remote types the messages name the solver host instead of a local
folder and say **Remote Path** whichever type you use (the Docker types call
that field **Container Path**). They appear when you press **Start
Server on Remote**, except the one for no solver at all, which appears after
**Connect**.

The last row applies to native types only, and it appears at **Start Server
on Remote**: the add-on probes each candidate solver for a usable device,
and it cannot probe a build on another machine. On the remote
types, **Automatic** with several GPU builds takes the first in order
(CUDA, then ROCm), and a device problem surfaces from the solver itself.

(attach-mismatch)=

### Attaching to a running server

The native types leave their server running on disconnect, so the next
**Connect** attaches to it instead of starting a second server on the same
port. Attaching could discard a device selection without any message:
connecting with **CPU** selected to a port still held by the **GPU** server of
an earlier session would run every solve on the GPU build while the panel
says **CPU**. The add-on refuses instead:

> A solver server is already running on port 9090, and its runs use the
> build in `/home/alice/ppf/target/release` (its solver reports cuda).
> Compute Device is set to CPU, whose build is in
> `/home/alice/ppf/target/cpu`. To switch, press Force Terminate Process
> and Connect again, or set Compute Device to GPU, Connect, click Stop
> Server, then set it back to CPU and Connect again.

The comparison uses the build directory the running server reports, not a
backend name, so a server built from a different folder is refused even when
it uses the same device.

(gpu-picker)=

## Picking a GPU

The **GPU** row in the Connection box chooses which NVIDIA GPU the solver
runs on. The add-on delivers the choice as `CUDA_VISIBLE_DEVICES` in the
server's environment when it starts the server, and the
[**Remote Hardware**](#remote-hardware) block below the box names the device
the solver ended up on.

The row appears **only while connected**, and only where a CUDA device could
be named at all. Two cases draw no picker:

- **macOS Native**, because the Metal build opens the system default
  device and offers no way to name another.
- **Compute Device** set to **CPU**, on any connection type. A CPU build
  opens no device, so the dropdown would offer the host's cards for a
  server that uses none of them.

The list is read from the solver host with `nvidia-smi`, so before
**Connect** there is nothing to offer. The devices named are those of the
machine that runs the server: this machine for the native types, the
container for the Docker types, and the remote host for **SSH** and
**SSH Command**.

The picker is a CUDA control: `nvidia-smi` names NVIDIA devices only. A
ROCm, Metal, or CPU solver is unaffected by it, and on a host with no
NVIDIA driver the dropdown holds **Automatic** alone with a message
beneath it. That is not a refusal and does not stop **Start Server on
Remote**; see [Hosts with one GPU, or none](#hosts-with-one-gpu-or-none).

The dropdown and the refresh button gray out together **while the server is
running or starting**, because the selection is applied at **Start Server on
Remote**. To move a running solver to another GPU, press **Stop Server on
Remote**, pick, and press **Start Server on Remote**.

### What the entries mean

| Entry | Meaning |
| ----- | ------- |
| **Automatic** | The default. The add-on sets no `CUDA_VISIBLE_DEVICES`, so whatever the solver host's own environment already holds stands. |
| **`<index>: <name>`** | One per device `nvidia-smi` reported, for example `0: NVIDIA RTX 6000 Ada Generation`. The number is the index `nvidia-smi` gives it on the solver host. |
| **`<index>: not detected`**, with an error icon | Not a device. It is a saved selection this host cannot satisfy, kept visible and named instead of resolving to a different GPU. The label carries the saved UUID instead of a number when the selection was made from a populated list. You cannot choose it; picking any real entry replaces it. |

A pick saves two things: the index, which is what the panel shows, and the
device's UUID, which is what the launch uses. CUDA's own device ordering can
differ from the ordering of `nvidia-smi`, and the UUID identifies the same
GPU across that difference. Both are written to a connection profile; see
[Connection Profiles](profiles.md).

**Start Server on Remote refuses a device the host does not have**, instead
of launching with no visible device, which would surface much later as a
solver error that names no GPU:

> GPU 3 is not present on the solver host. Detected: 0 (NVIDIA RTX 6000
> Ada Generation), 1 (NVIDIA RTX 6000 Ada Generation).

When a UUID is saved, it follows the index in parentheses. The refusal
applies only when the devices could be listed; with no list there is nothing
to contradict the request.

If **Start Server on Remote** attached to a server that was already
listening instead of launching one, the selection reached nothing. These
lines under the picker report the cases where the server's GPU and the
selection disagree or cannot be compared:

| Line | When it appears |
| ---- | --------------- |
| *Solver is on GPU 1, not the selected GPU 0*, then *Press Stop Server, then Start Server, to move it* | The server you attached to runs on another GPU than the one picked. |
| *Server does not report which GPU it is on* | The server does not report its GPU index, so the pick cannot be compared. |
| A red line with the reason the server reports (*Server resolved no CUDA device* when it gives none) | A CUDA server started but resolved no device. See [Hosts with one GPU, or none](#hosts-with-one-gpu-or-none). |

When the server and the pick agree, no line is drawn, because **Remote
Hardware** already names the device. These lines are drawn only for a
server that reports the CUDA backend, so the add-on never queries a Metal or
CPU server for them, including one reached over SSH.

Every start also writes one line to the add-on console that records what the
launch did with the selection: the GPU it started on and the
`CUDA_VISIBLE_DEVICES` it set, or that it attached to a server whose device
was already fixed.

### Refresh GPU List

The refresh icon beside the dropdown runs `nvidia-smi` on the solver host
again and rebuilds the list. The add-on lists the GPUs once, at
**Connect**, and keeps the result for the life of the connection. A probe
that failed stays failed until you press refresh.

The same button also lists the **solver builds** of the host again, which
feeds the remote **Compute Device** and **GPU Backend** rows. Press it when
the probe failed at **Connect** and you have since fixed the cause, or when
the host's GPUs or builds changed while you stayed connected.

The refresh button is part of the GPU row, which is hidden while **Compute
Device** is **CPU**. To list the builds again after building a CPU backend on
the host, set **Compute Device** to **GPU**, press refresh, then switch back.

**Disconnect** drops both lists, because the next connection may reach a
different machine.

### From a Script

Three MCP tools mirror the picker (see the
[MCP Tool Reference](../integrations/mcp_reference.rst)):

- `list_solver_gpus` returns the devices read from the solver host (each with
  `index`, `uuid` and `name`), `probed`, `probe_error` and the current
  selection. `probed` is `false` until the list has been read, which is not
  the same as a host with no GPU.
- `refresh_solver_gpus` reads the list again, like the refresh button. It
  needs a connection, and it lists the host's solver builds again as well.
- `set_solver_gpu` picks a device by `uuid`, which is what the add-on stores
  and prefers, or by `index`, which is reliable only while the host's devices
  do not change. A `uuid` or `index` that the list read from the host does
  not contain is refused. An `index` of `-1` with no `uuid` selects
  **Automatic**. Called with neither argument, it selects the device at
  index `0`, not **Automatic**.

From Python, the selection is held by `solver.param.solver_gpu_uuid` and
`solver.param.solver_gpu_index`. To return to **Automatic**, set the index
to `-1` and the UUID to an empty string.

### Hosts with one GPU, or none

A single-GPU host still offers both **Automatic** and that one device, and
the two differ. **Automatic** leaves any `CUDA_VISIBLE_DEVICES` already in
the server's environment intact. Naming the device replaces it, so the
panel's choice wins over an inherited one.

When `nvidia-smi` reports no device, cannot be run, or answers with
something unreadable, the dropdown holds **Automatic** alone and a line
beneath it gives the reason, one of:

> nvidia-smi listed no CUDA device. An NVIDIA GPU is required to run
> the solver.

> nvidia-smi failed on the solver host: *(its stderr)*

> Could not run nvidia-smi on the solver host: *(the backend error)*

All three are about NVIDIA devices. A host with no NVIDIA driver, which is
the ordinary case for a ROCm, Metal, or CPU solver, reaches one of them and
still runs; only the GPU picker is unavailable.

The last of the three is what an unreachable host or a backend command
error looks like; the probe is given five seconds. None of them blocks
**Start Server on Remote**: the launch goes ahead, and a genuinely missing
GPU surfaces from the solver. A CUDA server that starts but resolves no
device turns the line under the picker red and gives the server's reason
(*Server resolved no CUDA device* when it gives none). Treat the probe error
as the panel saying it could not confirm that the host has a usable device,
not as a refusal.

## Remote Hardware

Once you are connected and the server is running, **Backend Communicator**
shows a **Remote Hardware** box below the **Update Stat** row. It starts
collapsed; click its triangle to open it. Each row describes the machine or
container that runs the solver, as the server reports it. A row that does not
apply to the build in use is left out.

| Row | Shown for | Meaning |
| --- | --------- | ------- |
| **Backend** | Every build | `cuda`, `rocm`, `metal` or `cpu`: the build the server runs. |
| **GPU** | CUDA, ROCm, Metal | CUDA: `<index>: <name>` of the device the solver runs on, which is how two cards of one model are told apart. When the server resolved no device, the row holds the reason instead and **VRAM** and **SM** are left out. ROCm: the device's architecture name. Metal: the device name. |
| **GPU Family** | Metal | The Apple GPU family, for example `Apple8`. |
| **VRAM** | CUDA, Metal | CUDA: the device's memory, such as `48.0 GB`. Metal: `<n> GB (unified)`, memory the GPU shares with the CPU. |
| **CUDA** | CUDA | The CUDA version that `nvidia-smi` reports. |
| **SM** | CUDA | The device's compute capability, such as `sm_89`. |
| **CPU** | Every build | The processor name. |
| **RAM** | Every build | The machine's total memory. |

**CPU** and **RAM** read *Unknown* when the server cannot read them. On a CUDA
build, the **GPU**, **VRAM** and **SM** rows are left out when `nvidia-smi`
cannot be run on the solver host, and the **CUDA** row when its version
cannot be read. A CPU build lists no GPU rows because it uses no GPU.

(connections-under-the-hood)=

:::{admonition} Under the hood
:class: toggle

**Protocol**

Every connection type uses the same TCP protocol. Payloads are CBOR
envelopes. Text commands start with the `TCMD` header followed by a 4-byte
big-endian length prefix.

**Backend marker**

A solver build writes a `.ppf-backend` file beside its server binary that
names the backend it was linked for. Every backend links the same
executable name, so the marker is the evidence the add-on prefers for what
a directory holds. A directory without a marker is judged by where it sits
under the solver folder: for example, `target/cpu/release/` counts as the CPU
build.

**Start Server step**

On the remote types the add-on writes a small script on the solver host and
runs it. The script activates `$HOME/.local/share/ppf-cts/venv` if
it exists (the build worker that the server starts needs the Python modules
installed there), exports `CARGO_TARGET_DIR` for the build directory the
chosen server came out of, and starts the server under `nohup`:

```sh
export CARGO_TARGET_DIR="<root>/target/cuda"
nohup bash -c "...; CUDA_VISIBLE_DEVICES=<gpu> <root>/target/cuda/release/ppf-cts-server --host 0.0.0.0 --port <port>" \
  > server.log 2>&1 &
```

`CUDA_VISIBLE_DEVICES=<gpu>` is present only when a GPU is picked and
**Compute Device** is **GPU**. `--host 0.0.0.0` is present only in the Docker
modes, so that the published port reaches the server inside the container;
see [Security](../security.md#network-exposure-of-the-solver-port).

The binary is the one the **Compute Device** and **GPU Backend** selection
resolves to in the listing taken from that host, not a fixed
`target/release/ppf-cts-server`: a Windows or Linux distribution has no
`target/release`.
`CARGO_TARGET_DIR` is exported so that the server, its build worker, and the
solver all come from the same build directory.

The add-on waits up to **16 seconds** for the server to report that it is
ready. If a line containing `ERROR` or `FAILED` appears in `progress.log`
first, the wait aborts with that message; on a plain timeout, the panel
prints the last 20 lines of `server.log`.

The three native types start from **Start Server on Remote** too, but as a
direct child process instead of through a shell script, and they wait for a
`TCMD` ping to be answered instead of a `progress.log` marker. They set the
same `CARGO_TARGET_DIR`, and name the build worker's interpreter in
`PPF_CTS_BUILD_PYTHON`. See
{ref}`Windows - Under the hood <windows-under-the-hood>`,
[macOS](macos.md), and [Linux](linux.md).
:::
