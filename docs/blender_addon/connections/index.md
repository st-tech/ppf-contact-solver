# 🔌 Connections

The add-on talks to a solver process over one of several transports. Pick the
one that matches where the solver runs.

| Type | Use when | See |
| ---- | -------- | --- |
| **Docker over SSH** / **Docker over SSH Command** | Solver runs inside a container on a remote Docker host | [Docker over SSH](docker_over_ssh.md) |
| **Docker** | Solver runs inside a container on the local Docker daemon | [Docker (Local)](docker.md) |
| **SSH** / **SSH Command** | Solver runs **directly** on a remote Linux host | [SSH (Direct)](ssh.md) |
| **Windows Native** | Solver runs as a subprocess on this Windows machine, on CUDA, ROCm, or the CPU build | [Windows](windows.md) |
| **macOS Native** | Solver runs as a subprocess on this Apple-silicon Mac, on Metal or the CPU build | [macOS](macos.md) |
| **Linux Native** | Solver runs as a subprocess on this Linux machine, on CUDA, ROCm, or the CPU build | [Linux](linux.md) |

The last three are the **native** types: the solver is a child process
on the machine Blender runs on, started by the add-on, and only your own
platform's type is usable. The first three are the **remote** types: the
solver lives on another machine, or in a container, and the add-on
reaches it over SSH or the Docker API.

All types share the same server-side protocol (TCP) and the same UI flow in
the panel: **Connect** -> **Start Server on Remote** -> transfer data ->
**Run** -> **Fetch**. The wire payloads are CBOR envelopes whose schema is
defined by the `ppf-cts-formats` crate. Text commands use the TCMD header
followed by a 4-byte big-endian length prefix and a heartbeat-based recovery
flow.

```{figure} ../images/connections/transport_topologies.svg
:alt: Five stacked block diagrams, one per connection type. Top to bottom: Linux Native / macOS Native places the Blender add-on and the ppf-cts-server binary on a single workstation with a loopback TCP socket between them, the add-on spawning the server as a child process; SSH / SSH Command keeps the add-on on the workstation and reaches ppf-cts-server on a remote Linux host over a paramiko SSH session carrying both lifecycle commands and an SSH tunnel to the server; Docker (local) adds a Docker daemon between the add-on and a container on the same workstation, with server traffic bypassing the daemon straight to the container's published port on localhost; Docker over SSH / over SSH Command pushes the daemon and container onto a remote Linux host reached through one SSH session; Windows Native launches ppf-cts-server.exe as a hidden Win32 subprocess on the same Windows workstation. Blue solid arrows carry lifecycle commands, purple dashed arrows carry TCP traffic to the server.
:width: 820px

Where each piece lives, and how the add-on reaches it, for the
connection types the diagram covers. Blue solid arrows carry lifecycle
commands (start / stop / exec / port check); purple dashed arrows carry
the TCP connection to the `ppf-cts-server` binary. The three Docker
sub-modes of rows 3 and 4 are broken out separately in
[Docker (Local)](docker.md) and [Docker over SSH](docker_over_ssh.md).
The three native types share one shape: a child process on the same
machine, reached over a loopback socket. Rows 1 and 5 differ only in
how the process is spawned.
```

## What Happens When You Connect

Starting a solver session is two button presses:

1. **Connect** opens the transport to wherever the solver lives (a local
   process, an SSH session, or a Docker container) and checks that a
   `ppf-cts-server` binary is actually present at the path you configured,
   for the build you selected.
   If the check fails, an error is shown in the panel; the transport stays
   open so you can correct the path without reconnecting.
2. **Start Server on Remote** launches `ppf-cts-server` on the remote
   side and waits up to
   **16 seconds** for it to report that it is ready to accept work. If the
   server prints an error during startup, the panel reports the error
   immediately; if it simply never becomes ready, you get a timeout and a
   tail of its log.

The panel stays responsive while this happens -- the actual work runs on
a background thread and the UI polls it several times a second.

The three native types follow the same two steps: **Connect** only
resolves the solver root and checks that it holds the build you asked
for, and **Start Server on Remote** spawns the server as a child process
(or attaches to one already listening on the port).

### Cancelling

While either step is in flight, the button for it reads *Connecting...*
or *Server Starting...* and is disabled, and a **Cancel** button appears
under the status line. Pressing it tears the attempt down and returns
the panel to where it was, which matters most for a connect to a host
that is not answering: the handshake would otherwise sit there.

A connect that nobody cancels is given **60 seconds** and then torn down
the same way, reporting *Connection timed out*. Both paths end the
attempt rather than merely stopping the watch, so the next **Connect**
starts cleanly.

:::{tip}
**Save the connection once it works.** After a successful **Connect**
has confirmed the fields, **Disconnect**, then click the **Save** icon on
the profile row to write them to a `.toml` file; the whole profile row
is disabled while a connection is live. Next session, **Open** the file, pick
the entry, and every field auto-fills, with no retyping host, key path,
container, or port. See [Connection Profiles](profiles.md) for the full
workflow.
:::

(choosing-the-build)=

## Choosing the Build: Compute Device and GPU Backend

The solver links **one** backend per build: a build is a CUDA build, a
ROCm build, a Metal build, or a CPU build, decided when it was compiled.
So choosing which of them runs is choosing which **build directory** the
server comes out of, and a machine that has several keeps them in
separate directories:

| Directory | Holds |
| --------- | ----- |
| `target/release/` | The historical layout: whatever was built there. A plain `cargo build --release` on a GPU tree, or a `--features cpu` build on a tree whose `target/release` was free. |
| `target/cuda/release/`, `target/rocm/release/`, `target/metal/release/` | One per accelerator, which is what the Windows and Linux distributions ship and what `build.bat` / `build-linux-native/build.sh` produce. The macOS distribution has one accelerator and keeps it in `target/release/`. |
| `target/cpu/release/` | The portable CPU build, built alongside a GPU one with `CARGO_TARGET_DIR=target/cpu cargo build --release --features cpu`. |

Two rows in the Connection box say which one to use:

- **Compute Device** -- `GPU` or `CPU`. `GPU` means the machine's
  accelerator, whichever one it has (CUDA or ROCm on Windows and Linux,
  Metal on macOS); `CPU` means the portable build, which needs no GPU
  and is substantially slower. It is one property shared by every
  connection type, because they all ask the same question.
- **GPU Backend** -- `Automatic`, `CUDA`, or `ROCm`. Drawn only where
  there is something to choose: a folder or a solver host holding **more
  than one** GPU build (an x64 Windows or x86_64 Linux distribution
  carries CUDA and ROCm together), or one where a saved choice names a
  build that is not there. `Automatic` takes the only GPU build present,
  or, where there are several, the first whose solver reports a usable
  device, CUDA before ROCm.

Neither is written to a [connection profile](profiles.md); they stay at
whatever you last set them to.

### When the rows are drawn, and what they are asked

| | Native types | Remote types |
| --- | --- | --- |
| Drawn | As soon as **Solver Path** names a root, before Connect | Only **while connected** |
| Answered from | This machine's filesystem, on every redraw | One listing command run on the solver host at Connect, then cached |
| Editable | Before **Connect**, like the rest of the Connection box | While connected, until a server is running or launching |
| To change it | **Disconnect**, pick, **Connect** | **Stop Server on Remote**, pick, **Start Server on Remote** |

The difference follows from where the builds are. A native type can look
at the disk as often as it likes, so the rows can sit in the Connection
box with the path field they belong to, and **Connect** refuses a
selection the folder cannot serve. A remote type would have to run a
command over the connection to answer at all, which a panel redrawing
several times a second cannot do -- so the listing is taken once at
**Connect**, and the rows appear only after that, outside the part of
the box that Connect greys out. They are read at **Start Server on
Remote** rather than held from Connect, which is what makes them worth
moving: a value captured at Connect would leave a control the artist can
change and that changes no run.

**Compute Device is always drawn on a native type, even when only one
build is there**, disabled with a line saying which one is missing. A
control that disappears when its precondition is unmet costs you the
ability to tell "this folder has no CPU solver" from "the add-on cannot
do that". When the folder has no CPU build, the panel names the command
that adds one beside the existing GPU build:

```sh
CARGO_TARGET_DIR=target/cpu cargo build --release --features cpu
```

### What decides which device a directory answers for

`crates/ppf-cts-solver/build.rs` writes a `.ppf-backend` marker beside
its artifacts naming the backend it just linked, and that marker is the
only real evidence of what a directory holds -- every backend links the
same executable name, so a path is a convention rather than proof. A CPU
build sitting in `target/release`, which is exactly what a plain
`cargo build --release --features cpu` produces, therefore resolves as
**CPU** and not as GPU.

A directory with no marker is taken at its layout's word. That is the
normal state of an older distributed bundle, which ships the binaries
without the marker, and it is why such a bundle resolves exactly as it
always did.

### Refusals, never substitutions

A selection that cannot be served is refused by name. The add-on never
quietly runs the other build: an explicit choice that silently became a
30x slower run, reported nowhere, is the failure this whole mechanism
exists to prevent.

| Situation | What you see |
| --------- | ------------ |
| Folder holds the other device's build | *`<root>` holds the CPU build of the solver and no GPU build. Set Compute Device to CPU to run it, or point Solver Path at a folder that has a GPU build.* |
| Folder holds GPU builds, but not the named one | *`<root>` holds these GPU builds: cuda, and no rocm build. Set GPU Backend to one of those or to Automatic ...* |
| Folder holds no solver at all | *ppf-cts-server not found under `<root>` ...*, listing the layouts that were searched |
| `Automatic`, several GPU builds, none with a usable device | *No GPU build in this folder has a usable device (cuda: ...; rocm: ...). Set Compute Device to CPU to run without a GPU.* |

The remote types phrase the first three against the solver host rather
than against a local folder, and name **Remote Path** / **Container
Path** instead of **Solver Path**.

The last row is a **native-only** refusal, and it happens at **Start
Server on Remote** rather than while drawing: deciding it means running
each candidate's `ppf-contact-solver --probe`, which is a solver
execution and has no place in a panel redraw, and which the add-on
cannot do for builds sitting on another machine. There, `Automatic` with
several GPU builds takes the first in order (CUDA, then ROCm) and a
device problem surfaces from the solver itself. Everything the rows
themselves report, on either side, is a fact about which directories
exist.

(attach-mismatch)=

### Attaching to a running server

The native types leave their server running on disconnect on purpose, so
a later Connect attaches to it rather than fighting for the port. That
attach is also where a device selection could be silently lost --
connecting with `CPU` selected to a port still held by the `GPU` server
an earlier session started would run every solve on the GPU build while
the panel said CPU. It is refused instead:

> A solver server is already running on port 9090, and its runs use the
> build in `/home/alice/ppf/target/release` (its solver reports cuda).
> Compute Device is set to CPU, whose build is in
> `/home/alice/ppf/target/cpu`. To switch, press Force Terminate Process
> and Connect again, or set Compute Device to GPU, Connect, click Stop
> Server, then set it back to CPU and Connect again.

What is compared is the **target directory** the running server reports,
not a backend name: that directory is what decides which solver a run
actually executes, and a name would also accept a server from an
unrelated tree built for the same device.

(retired-local)=

## The Retired `Local` Type

Earlier versions offered a **Local** type: a connection to a server you
had started by hand on this machine. The three native types replaced it.
They reach the same server and additionally know how to start it, which
build directory it came out of, and which device it runs on.

Nothing needs to be migrated by hand:

- A `.blend` saved with `Local` is moved onto this platform's native
  type when it is opened, and its path is carried across into the
  native **Solver Path** field, unless that field already holds one --
  a scene carrying both was connected some other way since, and that
  later answer is the one kept. The move is recorded in the add-on
  console as `[auto-migrate] server_type=Local -> LINUX_NATIVE;
  local_path -> linux_native_path`, so a scene that changes type is not
  silent about it. Saving the file makes the move permanent.
- A [connection profile](profiles.md) whose `type` is still `"Local"`
  is applied as `Windows Native`, `macOS Native`, or `Linux Native`
  depending on the platform reading it, and a `local_path` key lands on
  that platform's path field. A profile is a file you wrote and keep, so
  a name that was once documented keeps working.

`Local` no longer appears in the **Type** dropdown, and the slot it
occupied is retired permanently rather than reused, so no saved file can
be repointed at a different type by accident.

(gpu-picker)=

## Picking a GPU

The solver never calls `cudaSetDevice`, so it runs on device 0 of
whatever CUDA can see, and the add-on is what launches the server on
every backend. The **GPU** row in the Connection box is therefore the
only place to say which device a run uses: the choice is delivered as
`CUDA_VISIBLE_DEVICES` on the server's environment at launch, and the
server's own hardware probe reads the same variable, so the **Remote
Hardware** block below the box names the device the solver ended up on.

The row appears **only while connected**, and only where a CUDA device
could be named at all. Two cases draw no picker:

- **macOS Native**, because the Metal build opens the system default
  device and offers no way to name another.
- **Compute Device set to `CPU`**, on any connection type. A CPU build
  opens no device, so the dropdown would offer the host's cards for a
  server that uses none of them.

The list is read from the solver host by running `nvidia-smi` through
the connection, so before **Connect** there is nothing to offer, and the
devices named are those of the machine that will run the server -- which
for SSH and both Docker modes is not the machine Blender runs on.

The picker is a CUDA control: `nvidia-smi` names NVIDIA devices only,
and the choice is delivered as `CUDA_VISIBLE_DEVICES`. A ROCm, Metal, or
CPU solver is unaffected by it, and on a host with no NVIDIA driver the
dropdown holds **Automatic** alone with a probe message beneath it. That
is not a refusal, and it does not stop **Start Server on Remote**; see
**Hosts with one GPU, or none** below.

Dropdown and Refresh grey out together **while the server is running or
launching**, because the selection is applied at **Start Server on Remote**
and nothing later re-reads it. Moving a running solver to another device is
**Stop Server on Remote**, pick, **Start Server on Remote**.

### What the entries mean

| Entry | Meaning |
| ----- | ------- |
| **Automatic** | The default. The add-on sets no `CUDA_VISIBLE_DEVICES`, so whatever the solver host's own environment already puts there stands. |
| **`<index>: <name>`** | One per device `nvidia-smi` reported, for example `0: NVIDIA RTX 6000 Ada Generation`. The number is the index `nvidia-smi` gives it on the solver host. |
| **`<index>: not detected`**, with an error icon | Not a device. It is a saved selection this host cannot satisfy, kept visible and named rather than quietly resolving to a different GPU. The label carries the saved UUID rather than a number when the selection was made from a populated list. It cannot be chosen; picking any real entry replaces it. |

A pick saves two things: the index, which is what the panel shows, and
the device's UUID, which is what the launch actually uses. CUDA's own
device ordering can differ from `nvidia-smi`'s, and the UUID is the
identity that survives that disagreement. Both are written to a
connection profile; see [Connection Profiles](profiles.md).

**Start Server refuses a device the host does not have**, rather than
launching against an empty visible set, which would otherwise surface
much later as a solver error naming no GPU at all:

> GPU 3 is not present on the solver host. Detected: 0 (NVIDIA RTX 6000
> Ada Generation), 1 (NVIDIA RTX 6000 Ada Generation).

That refusal only fires when the devices could be enumerated; with no
list there is nothing to contradict the request with.

If **Start Server on Remote** attached to a server that was already listening
instead of launching one, the selection reached nothing, and when the
device that server is on is not the one picked, a line under the picker
says so -- `Solver is on GPU 1, not the selected GPU 0`, followed by
*Press Stop Server, then Start Server, to move it*. Agreement gets no
line of its own, since **Remote Hardware** already names the device. A
server too old to report its device at all gets *Server does not report
which GPU it is on* instead, because silence there would read as
agreement. Both lines are CUDA questions, decided by the backend the
**server** reports rather than by the connection type, so a Metal or CPU
server is never asked them -- including one reached over SSH, which is
exactly where the connection type and the server's backend disagree.

Either way, every start writes one line to the add-on console recording
what the launch did with the selection -- the GPU it started on and the
`CUDA_VISIBLE_DEVICES` it set, or that it attached to a server whose
device was already fixed. A run's GPU is otherwise only visible while
the panel is open.

### Refresh GPU List

The refresh icon beside the dropdown re-runs `nvidia-smi` on the solver
host and rebuilds the list. It exists because the list is enumerated
exactly once, at **Connect**, and cached for the life of the
connection: the dropdown asks for it on every redraw, and each answer
costs a command on the solver host. The outcome is cached whether it
succeeded or failed, so a host with no NVIDIA driver is not re-probed
on every redraw either -- which is why a probe that failed stays failed
until you press Refresh.

The same button also re-runs the **solver build** listing behind the
remote **Compute Device** and **GPU Backend** rows, for the same reason
and with the same caching. Both answer "what can this solver host run",
so building a backend there, or freeing a GPU, is one press either way.

Reach for it when the probe failed at **Connect** and you have since
fixed the cause, or when the solver host's devices or builds changed
while you stayed connected.

Disconnecting drops both lists rather than refreshing them: the next
connection may reach a different machine, where a list left over from
this one would name GPUs and build directories that are not there.

### Hosts with one GPU, or none

A single-GPU host still offers both **Automatic** and that one device,
and the two are not the same instruction. Automatic leaves any
`CUDA_VISIBLE_DEVICES` already in the server's environment intact;
naming the device replaces it, so the panel's choice wins over an
inherited one.

When `nvidia-smi` reports no device, cannot be run, or answers with
something unreadable, the dropdown holds **Automatic** alone and the
reason is spelled out on a line beneath it, one of:

> nvidia-smi listed no CUDA device. An NVIDIA GPU is required to run
> the solver.

> nvidia-smi failed on the solver host: *(its stderr)*

> Could not run nvidia-smi on the solver host: *(the backend error)*

All three are about NVIDIA devices. A host with no NVIDIA driver, which
is the ordinary case for a ROCm, Metal, or CPU solver, reaches one of
them and still runs; only the GPU picker is unavailable.

The last of the three is what an unreachable host or a backend command error
looks like; the probe is given five seconds. None of them blocks **Start
Server on Remote** -- with nothing enumerated there is no list to check a
request against -- so the launch goes ahead and a genuinely missing GPU
surfaces from the solver instead: a CUDA server that starts but resolves no
device turns the outcome line under the picker red, reading *Server resolved
no CUDA device*. That line is drawn only for a server that reports CUDA.
Treat the probe error as the panel saying it could not confirm the host
has a usable device, not as a refusal.

## Port Usage at a Glance

| Port | Role | Default |
| ---- | ---- | ------- |
| Server | Solver TCP listener (`ppf-cts-server`) | `9090` |
| MCP | MCP Streamable HTTP server (for AI integration) | `9633` |

Only the server port crosses the transport boundary; MCP is local to the
machine running Blender. The server port is configurable per connection.

:::{note}
For both Docker modes the server port must be published on the container
(`-p 9090:9090`). Local Docker checks it at **Connect**; Docker over SSH
checks it before **Start Server on Remote**. Either way the add-on refuses to
continue if the port is not exposed. Local Docker makes one exception:
a container on `--network host` publishes nothing, needs no `-p`, and is
accepted as is. The Docker-over-SSH check has no such exception.
:::

## Port Already in Use

If the port is already bound when the add-on tries to start the server,
the panel shows a `Port N is in use` error and a **Force Terminate
Process** button. Clicking it locates the listening process by port and
kills it, including child processes on Windows.

If the listener is itself a `ppf-cts-server` from a previous Blender
session (for example after a Blender restart on a native connection), the
add-on detects this with a TCMD probe and reuses the running server
instead of erroring -- unless it is running a different build from the one
selected, which is refused; see
{ref}`Attaching to a running server <attach-mismatch>`. Foreign listeners
on the port still surface the in-use error so the user can decide whether
to terminate.

### When the Force Terminate Process row is drawn

The row sits at panel level, beside the error text rather than inside
the collapsible Connection box, so a refused **Connect** can never leave
you with a server process and no visible way to end it. It appears only
while one of three failures stands, and disappears when the failure
does:

- a **protocol version mismatch**, which is what a server orphaned from
  an older binary looks like after you update the add-on;
- **any error while not connected**, since **Stop Server on Remote** is
  reachable only through a connection;
- **any error naming a held port**, connected or not -- `Port N is in
  use`, `A solver server is already running on port N`, and the remote
  launch's `Server port N is already in use on the remote host`.

A build or transfer refusal (a stray isolated vertex, an unusable rest
shape) is not one of them; those carry their own repair button instead.

Underneath, one line says what the kill would act on, which is worth
reading before pressing it:

| Line | Meaning |
| ---- | ------- |
| *A solver server is listening on port N* | A `ppf-cts-server` of ours answers there. |
| *Port N is held by another program* | Something else has it. |
| *Nothing is listening on port N* | The port is free; the error is about something else. |
| *Cannot check port N from here: ...* | A remote type, where the port belongs to the solver host and a panel redraw must not run a command on it. |

On a remote type the row is drawn only **while connected**, because the
kill runs through the live backend; disconnected, there is no transport
for it to act through. A port error that has gone stale -- our own server
now answering on the port because a second **Connect** attached to it --
hides both the error and the button rather than tempting you into
killing a healthy server.

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
local subprocess) and verifies that a `ppf-cts-server` binary for the
selected build is present at the configured path. If that verification
fails, the error is reported in the panel but the transport remains open.
On a remote type, Connect is also where the solver host's GPU list and
its solver-build listing are enumerated, once each.

**Start Server step**

On the remote backends (SSH, Docker, Docker over SSH) the add-on writes
a small script on the solver host and runs it. The script activates
`$HOME/.local/share/ppf-cts/venv` if it exists (the Rust server spawns a
Python build worker that imports the `_ppf_cts_py` PyO3 module from that
venv), exports `CARGO_TARGET_DIR` for the build directory the chosen
server came out of, and then runs the server under `nohup`:

```sh
export CARGO_TARGET_DIR="<root>/target/cpu"
nohup bash -c "...; <root>/target/cpu/release/ppf-cts-server --port <port>" \
  > server.log 2>&1 &
```

The binary is the one the **Compute Device** and **GPU Backend**
selection resolves to in the listing taken from that host, not a fixed
`target/release/ppf-cts-server`: a solver host running a distribution has
no `target/release` at all. `CARGO_TARGET_DIR` is exported because a run
takes three things out of a build directory and the server binary is only
one of them -- the build worker loads the cdylib from the first target
directory it finds, and the session's `SOLVER_PATH` is written from that
same directory. Naming only the binary would start the CPU server and run
the solve on the GPU solver, with nothing anywhere reporting the split.

The UI waits up to **16 seconds** for the server to announce it is
ready. If a line containing `ERROR` or `FAILED` appears first, the wait
aborts with that message; on plain timeout, the panel prints the last 20
lines of `server.log`.

The three native types launch from **Start Server on Remote** too, but as
a direct child process rather than through a shell script, and they wait
for a TCMD probe instead of a `progress.log` marker. They set the same
`CARGO_TARGET_DIR`, and name the build worker's interpreter explicitly in
`PPF_CTS_BUILD_PYTHON`. See {ref}`Windows - Under the hood <windows-under-the-hood>`,
[macOS](macos.md), and [Linux](linux.md).

**Docker port pre-launch check**

Before **Start Server on Remote** runs on Docker-over-SSH, the add-on checks
that the configured server port is published on the container. If it is not,
the operator aborts with:

> Docker port 9090 is not exposed on container 'ppf-contact-solver'.
> Please expose the port with '-p 9090:9090' when starting the
> container.

The add-on cannot publish a port on an existing container; this has to
be fixed on the container side (for example by re-running `docker run
-p` or editing `compose.yaml`).
:::
