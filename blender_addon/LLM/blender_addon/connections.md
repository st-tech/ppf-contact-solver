# Connections

This document condenses the Blender add-on connection docs: the Connections overview (index.md), and the per-backend pages for SSH, Docker, Windows Native, macOS Native, Linux Native, and Connection Profiles.

## Overview

The add-on talks to a solver process over one of several transports. Pick the one that matches where the solver runs.

| Type | Use when | See |
| ---- | -------- | --- |
| **SSH** | Solver runs on a remote Linux host, credentials entered as fields | SSH |
| **SSH Command** | Same as SSH, but configured by pasting an `ssh ...` shell command | SSH |
| **Docker** | Solver runs inside a container on the local Docker daemon | Docker |
| **Docker over SSH** | Solver runs inside a container on a remote Docker host | Docker |
| **Docker over SSH Command** | Docker-over-SSH configured from a shell `ssh ...` command | Docker |
| **Windows Native** | Solver runs as a Windows subprocess using a bundled Python + CUDA | Windows |
| **macOS Native** | Solver runs as a macOS subprocess against the local Metal build | macOS |
| **Linux Native** | Solver runs as a Linux subprocess on the Blender machine, from a checkout or an unpacked distribution | Linux Native |

All types share the same server-side TCP protocol (see `PROTOCOL_VERSION` in `core/protocol.py` for the current wire version) and the same UI flow in the panel: **Connect** -> **Start Server** -> transfer data -> **Run** -> **Fetch**.

Figure: Five stacked block diagrams showing where each piece lives for the five connection types, with blue solid arrows for lifecycle commands (start / stop / exec / port check) and purple dashed arrows for the TCP connection to the solver. The three Docker sub-modes are broken out separately in Docker.

### What happens when you connect

Starting a solver session is two button presses:

1. **Connect** opens the transport to wherever the solver lives (a local process, an SSH session, or a Docker container) and checks that the `ppf-cts-server` binary is actually present at the path you configured. If the check fails, the connection is dropped and an error is reported.
2. **Start Server** launches `ppf-cts-server` on the remote side and waits up to **16 seconds** for it to report that it is ready to accept work. If the server prints an error during startup, the panel reports the error immediately; if it simply never becomes ready, you get a timeout and a tail of its log.

The panel stays responsive while this happens, the actual work runs on a background thread and the UI polls it several times a second.

Every backend works this way, the three native types included: **Connect** opens the transport and does not start anything, and **Start Server** starts the solver server. If a `ppf-cts-server` from a previous session is still on the port, **Start Server** attaches to it instead of launching a second one.

### Choosing the build: Compute Device and GPU Backend

Every build of the solver links exactly one backend, and each one lives in its
own directory, so choosing GPU or CPU is choosing which build directory the
server comes from. Two rows in the Backend Communicator make that choice, and
they apply to every connection type.

**Compute Device** picks `GPU` or `CPU`. The GPU build is the accelerated one
(CUDA or ROCm on Windows and Linux, Metal on macOS); the CPU build needs no GPU
and is substantially slower. The row accepts a change where both builds are
there, and also while the selection names a build that is absent, so a folder
holding only the CPU build does not lock the row on `GPU`. It is closed when the
selection is the one build present, since there is nowhere else to move it. It
is never changed for you: a device that is not there is refused by name, never
substituted.

**GPU Backend** picks `Automatic`, `CUDA` or `ROCm`, and is drawn only where
there is something to pick: the root or the solver host holds more than one GPU
build, or a saved choice names a build that is not there. `Automatic` takes the
only GPU build present, or where there are several, CUDA before ROCm. On a
native connection it also skips a build whose solver reports no usable device,
because that solver is on this machine and is run to ask; on a remote connection
nothing is run to ask, so `Automatic` takes the first of those present and a
host holding both is worth naming explicitly.

A directory that a build wrote carries a `.ppf-backend` marker naming the
backend in it, and that marker is what the add-on reads. Every backend links the
same executable name, so the directory alone is not evidence of what it holds; a
directory with no marker, which is the shape of a distribution packaged without
one, is read by its layout instead.

**On the three native types the rows are drawn before you connect**, because the
builds are on the machine Blender runs on and the panel resolves them from the
filesystem while you edit **Solver Path**.

**On the remote types (SSH, SSH Command, Docker, Docker over SSH, Docker over
SSH Command) they appear after you connect.** The builds are on the solver host,
so the add-on asks that host once per connection for a listing of the build
directories under the configured path, and draws the rows from the listing.
Before connecting there is nothing to ask, which is why the rows are not there
yet. The refresh button beside the **GPU** dropdown re-asks for both the GPU list
and the build listing, so a backend built on the host during the session appears
after pressing it. If the listing command fails, the reason is printed under the
rows and the rows stay, because what the host holds is then unknown rather than
known to be nothing.

The selection is applied at **Start Server**, so the rows stay editable while
connected as long as the server is stopped: **Stop Server**, pick, **Start
Server** moves the solver onto another build without reconnecting. A selection
the solver host cannot serve is refused by name at **Start Server**, and the
three refusals are distinct: the folder holds no solver at all, it holds the
other device's build, or it holds GPU builds and not the accelerator that was
asked for. Only the first is a reason to change the path.

The launch names the build directory as well as the binary: the resolved
`ppf-cts-server` is started, and `CARGO_TARGET_DIR` is exported to the directory
it came out of so the build worker loads the matching `_ppf_cts_py` cdylib.
Without that, the server and the solve it drives can come from different builds,
and nothing reports the split, because each binary answers `--backend` honestly
about itself and neither is asked about the other.

### Choosing a GPU

On a solver host with more than one NVIDIA card, the **GPU** dropdown in the
Backend Communicator picks the one the solver runs on. It lists the cards by
name with the index the driver gives them, and two cards of the same model are
told apart by that index.

One mechanism serves every backend. **Connect** enumerates the solver host by
running `nvidia-smi` through the connection, so the list belongs to the machine
that will run the server rather than to the workstation. The row appears only
once connected, since before that there is no list to offer. The refresh button
beside it re-reads it, and the solver host's build listing with it. **Start
Server** applies the choice.

The row is drawn only while **Compute Device** is `GPU`. A CPU server opens no
device, so with `CPU` selected there is nothing for the dropdown to name.

The row stays editable while connected as long as the server is stopped, so
**Stop Server**, pick another card, **Start Server** moves a solver without
reconnecting. The dropdown carries no status text of its own: the selected
entry already names the card, and a card the host does not have reads
`<index>: not detected`. Only a failed enumeration gets a line, since that
leaves the dropdown holding nothing but `Automatic`.

The card the solver is on is named by the **GPU** row of the Remote Hardware
block, as `<index>: <model>`, which is the server's own answer. A line appears
under the dropdown only when that answer and the selection disagree, which
happens when Start Server attached to a server it did not launch, or when the
server is too old to report its device at all. Every start is also written to
the add-on console, so a session can be traced back to a device later.

`Automatic` leaves `CUDA_VISIBLE_DEVICES` unchanged. On a multi-GPU host that
variable must identify the chosen card by UUID; an unset or numeric value is
ambiguous because CUDA and nvidia-smi need not use the same numeric ordering.
Picking a card in the panel writes its UUID automatically.

A choice the solver host cannot satisfy is refused by **Start Server**, naming
the cards that are present, and the dropdown keeps it visible as
`<index>: not detected` rather than resolving it to a different card.

TIP: **Save the connection once it works.** As soon as you have a successful **Connect**, click the **Save** icon on the profile row to write the current fields to a `.toml` file. Next session, **Open** the file, pick the entry, and every field auto-fills, with no retyping host, key path, container, or port. See Connection Profiles for the full workflow.

### Port usage at a glance

| Port | Role | Default |
| ---- | ---- | ------- |
| Server | Solver TCP listener (`ppf-cts-server`) | `9090` |
| MCP | MCP Streamable HTTP server (for AI integration) | `9633` |

Only the server port crosses the transport boundary; MCP is local to the machine running Blender. The server port is configurable per connection.

NOTE: For Docker-over-SSH, the server port must be published on the container (`-p 9090:9090`). The add-on checks this before launching the server and refuses to continue if the port is not exposed.

UNDER THE HOOD:

**Non-blocking UI**

The panel does not freeze while Connect or Start Server is running. Work happens on a background thread and the panel refreshes several times a second, so any status or error reported by the background work appears in the panel promptly.

**Connect step**

Connect opens the configured transport (SSH session, Docker client, or a local subprocess) and verifies that the `ppf-cts-server` binary is present at the configured path. If that verification fails, the transport is torn down before the error is reported.

**Start Server step**

On the remote backends (SSH, SSH Command, Docker, Docker over SSH, Docker over SSH Command) the add-on writes a small launch script to `/tmp/start_server.sh` and runs it. The script changes to the configured directory, exports `CARGO_TARGET_DIR` when the build it resolved sits in a cargo target directory, and invokes:

```sh
nohup bash -c '[ -f $HOME/.local/share/ppf-cts/venv/bin/activate ] && source $HOME/.local/share/ppf-cts/venv/bin/activate; <resolved build dir>/ppf-cts-server --port <port>' > server.log 2>&1 &
```

The binary is the one the **Compute Device** and **GPU Backend** selection resolves against the solver host's build listing (see Choosing the build), not a fixed path.

Inside a Docker container the server is launched with `--host 0.0.0.0` so the published port mapping reaches it; on SSH and on a native connection it binds to `127.0.0.1`.

When a GPU is picked, its enumerated UUID is assigned to `CUDA_VISIBLE_DEVICES` in front of that server command. Using the UUID avoids any difference between nvidia-smi's numeric ordering and CUDA's. Windows Native and Linux Native set the same variable in the process environment they spawn the server with. The solver never calls `cudaSetDevice`, so it runs on device 0 of whatever CUDA can see, which is what makes restricting the visible set the whole mechanism.

The server writes `SERVER_STARTING` and `SERVER_READY` markers to `progress.log` in the working directory. The UI tails that file and waits up to **16 seconds** for `SERVER_READY`. If a line containing `ERROR` or `FAILED` appears first, the wait aborts with that message; on plain timeout, the panel prints the last 20 lines of `server.log`.

The three native types spawn the server as a child process instead of writing a launch script, since there is no shell on the other side of them. See Windows Native - Under the hood and Linux Native - Under the hood.

**Docker port pre-launch check**

Before **Start Server** on Docker-over-SSH, the add-on checks that the configured server port is published on the container. If it is not, the operator aborts with:

> Docker port 9090 is not exposed on container 'ppf-contact-solver'. Please expose the port with '-p 9090:9090' when starting the container.

The add-on cannot publish a port on an existing container; this has to be fixed on the container side (for example by re-running `docker run -p` or editing `compose.yaml`).

## Linux Native

The solver runs as a subprocess on the same **Linux** machine as Blender. The
add-on launches it and talks to it over a local TCP socket, with no SSH and no
Docker in between. On Windows the equivalent is the Windows Native backend, and
on macOS the macOS Native backend, which runs the local Metal build.

### When to use it

- You are running Blender on Linux with either the extracted Linux distribution or a solver checkout you built, on a local GPU or on the CPU build.
- You are iterating on solver code and want the fastest possible disconnect/reconnect turnaround.
- You do not want to pay the cost of SSH or Docker for every transfer.

### Setup

1. Set **Server Type** to `Linux Native`.
2. Set **Solver Path** to the solver root: the folder that holds `target/release/ppf-cts-server`, or `target/cuda/release`, `target/rocm/release` or `target/cpu/release`. An extracted Linux distribution and a checkout you built have the same shape, because the distribution ships each backend in the `target/<backend>/release` directory it was built in, so one rule accepts both. The distribution's `bin/` holds the backend library and ffmpeg and never a server, so it is not a root.
3. If you pick a subdirectory (`target`, `target/release`, `target/cuda/release`, or the distribution's `bin`), the add-on walks up to the real root and the panel names the root it resolved to.
4. Pick **Compute Device**, and **GPU Backend** where the root holds more than one GPU build. Both rows read the folder as you set it, so they are usable before you connect. See Choosing the build.
5. Set **Project Name** on the main panel.
6. Click **Connect**. The add-on checks that the root holds the build your selection names, and refuses the connection if it does not, rather than reporting a connection and leaving the failure to Start Server.
7. Click **Start Server**. The panel waits a few seconds for the server to report that it is ready.

Figure: Backend Communicator with **Server Type** set to `Linux Native`. **Solver Path**, the **Compute Device** row and **Project Name** show up; no SSH, Docker, or Windows-native fields. **Connect** is highlighted.

TIP: The interpreter the build worker runs under belongs to the root you selected, and the add-on names it for you: a distribution's own `python/bin/python3` when the root ships one, otherwise the developer virtual environment at `$HOME/.local/share/ppf-cts/venv`. There is nothing to activate by hand.

### Fields

| Field | Description |
| ----- | ----------- |
| Solver Path | Filesystem path to the solver root, an extracted Linux distribution or a checkout you built, for example `~/ppf-contact-solver`. Empty until you fill it in. |
| Compute Device | Whether the GPU or the CPU build under the root runs. See Choosing the build. |
| GPU Backend | Which accelerator runs when Compute Device is GPU and the root holds more than one GPU build. See Choosing the build. |
| GPU | Which CUDA device the solver runs on. See Choosing a GPU. |
| Server Port | TCP port for `ppf-cts-server`. Default `9090`; range 1024-65535. |

### Dependencies

Linux Native mode requires neither `paramiko` nor `docker-py`. The main panel's **Install Paramiko** and **Install Docker** buttons are only relevant for SSH and Docker modes.

`cbor2` is required for **all** backends: it encodes the scene on the Blender side before every Transfer, so the dependency is independent of the transport. This is unlike `paramiko` (needed only for SSH and Docker-over-SSH) and `docker-py` (needed only for Docker). `cbor2` ships as a per-ABI wheel in `blender_manifest.toml` and installs automatically when the extension is installed through Blender; if that wheel is missing (e.g. the add-on was copied in manually or carried over by a settings migration without reinstalling its wheels), the main panel shows an **Install cbor2 to Add-on Directory** button as a recovery path.

### Troubleshooting

- **"ppf-cts-server not found under \<root\>"** - the folder holds no solver in any accepted layout. Point **Solver Path** at the folder that has `target/release/ppf-cts-server` in it, or `target/cuda/release`, `target/rocm/release` or `target/cpu/release`. For a downloaded release that is the extracted distribution root, not the folder you extracted it into; the walk above goes up from your selection, never down into it.
- **"... holds the CPU build ... and no GPU build"** (or the reverse) - the path is right and the selection is not. Set **Compute Device** to the device that is there.
- **"... holds these GPU builds: cuda, and no rocm build"** (or the reverse) - the root holds GPU builds, and not the accelerator **GPU Backend** names. The message lists the ones that are there; pick one of those, or `Automatic`.
- **"No GPU build in this folder has a usable device"** - every GPU build under the root was asked at Start Server and none found a device it can run on. The message names what each one reported. Set **Compute Device** to `CPU` to run without a GPU.
- **"A solver server is already running on port ..., and its runs use the build in ..."** - a `ppf-cts-server` left over from an earlier session holds the port and runs a different build from the one selected. The message names both directories and the way out: **Force Terminate Process**, or connect with the device that matches, **Stop Server**, and connect again.
- **Server startup timed out.** - the solver launched but did not report readiness within 16 seconds. Check `server.log` inside the solver directory; the panel also prints the last 20 lines when the timeout fires.
- **Port already in use.** - something that is not a `ppf-cts-server` is bound to the port. Change the **Server Port**, or free it.

UNDER THE HOOD:

**Launch**

Linux Native spawns the resolved `ppf-cts-server` as a child process with `--port <port>`, working directory at the solver root, and its output appended to `server.log` in that root. It writes no launch script and starts no shell. Redirection goes to a real file rather than a pipe: with a pipe the add-on owns the read end and never drains it, and once the OS buffer fills the server's worker threads block in a write syscall and it appears wedged.

**The environment the server is given**

- `CARGO_TARGET_DIR` names the build directory the selection resolved to, so the build worker loads the cdylib from the same build the server came from. An inherited value is dropped for a layout that is not a cargo target directory rather than guessed at.
- `PYTHONPATH` begins with the root, so the build worker can import the bundled `frontend` package.
- `PPF_CTS_BUILD_PYTHON` names the interpreter for the build worker, `<root>/python/bin/python3` in a distribution and `$HOME/.local/share/ppf-cts/venv/bin/python` for a checkout. An inherited value wins, since someone who set it meant it.
- `VIRTUAL_ENV` is set only when that interpreter really is a virtual environment, and cleared otherwise, so a venv active in the shell that launched Blender cannot contradict the interpreter actually chosen.
- `CUDA_VISIBLE_DEVICES` carries the GPU choice, by UUID. See Choosing a GPU.

No library search path is set. Every binary this project ships on Linux finds its backend library through its own RPATH, which the loader searches before `LD_LIBRARY_PATH`, and the distribution is built that way so it does not depend on what a shell exports.

**Attaching to a server that is already there**

If a `ppf-cts-server` is already answering on the port (Blender was restarted while the previous session's server kept running), **Start Server** attaches to it instead of launching a second one. What it compares is the build directory the running server reports its runs take the solver from, so a server running a build other than the one **Compute Device** names is refused rather than adopted.

**Shared port field**

The **Server Port** field is the same underlying property for every connection type. In profile TOML files it is written as `docker_port`, whatever the type.

**File transfer fast path**

On a Linux Native connection, file transfers copy directly on disk instead of going through the solver TCP socket: no pickle overhead, and much faster than the SSH or Docker paths. The trade-off is cosmetic: the panel does not display a bandwidth figure while such a transfer is in progress. That is expected on this backend.

## SSH

The solver runs on a remote Linux host reached over SSH. Two UI modes are available: **Custom** (explicit fields) and **Command** (parsed from a raw `ssh ...` string). Both produce the same connection; once connected, the rest of the UI behaves identically.

### When to use it

- The GPU lives on a different machine from the user's workstation.
- Multiple users share a lab or cloud solver box.
- You want a persistent remote project that survives Blender restarts.

### Setup - Custom mode

1. Set **Server Type** to `SSH`.
2. Fill in the fields below.
3. Click **Connect** -> **Start Server**.

Figure: Backend Communicator with **Server Type** set to `SSH`. **Host**, **Port**, **User**, **SSH Key**, and **Remote Path** are exposed, plus the shared **Project Name** field. **Connect** is highlighted.

| Field | Default | Description |
| ----- | ------- | ----------- |
| Host | `""` | Hostname or SSH alias from `~/.ssh/config`. |
| Port | `22` | SSH port. |
| Username | `""` | Remote user. Leave empty to use SSH config's `User`. |
| Key Path | `~/.ssh/id_ed25519` or `~/.ssh/id_rsa` | Private key file. |
| Proxy Jump | `""` | Jump host to tunnel through, in `ssh -J` form: `[user@]host[:port]`, comma separated for a chain. Empty uses the alias's `ProxyJump` from `~/.ssh/config`. See Jump hosts. |
| Remote Path | `""` | Remote solver directory, e.g. `/root/ppf-contact-solver` (must contain the built `ppf-cts-server` binary). |
| Compute Device | `GPU` | Which build on the REMOTE host runs, the GPU one or the CPU one. The build listing is read from that host, so the row is offered once connected. See Choosing the build. |
| GPU Backend | `Automatic` | Which accelerator runs when Compute Device is GPU. Offered once connected, and only where the remote root holds more than one GPU build. See Choosing the build. |
| GPU | `Automatic` | Which CUDA device on the REMOTE host the solver runs on. The list is read from that host, so it is offered once connected. See Choosing a GPU. |
| Server Port | `9090` | Port on the remote host where `ppf-cts-server` listens. |

Aliases from your `~/.ssh/config` are resolved automatically, including entries pulled in via `Include` directives. If the alias's config supplies a hostname, port, user, or identity file, you can leave those fields blank in the panel and they will be filled in at connect time.

### Supported `ssh_config` options

The add-on ships its own minimal parser, it does **not** shell out to the system `ssh` binary. Only the following keywords are honored:

| Keyword | Supported | Notes |
| ------- | --------- | ----- |
| `Host` | yes | Wildcards `*` and `?` match via `fnmatch`. Multiple patterns per line are allowed. |
| `HostName` | yes | |
| `Port` | yes | |
| `User` | yes | |
| `IdentityFile` | yes | `~` is expanded. Only the first match per host wins; multiple `IdentityFile` lines are not tried in sequence. |
| `ProxyJump` | yes | Each jump host is resolved through the config in turn, so an alias brings its own `HostName` / `Port` / `User` / `IdentityFile`, and a jump host that has a `ProxyJump` of its own extends the chain. `none` connects directly. See Jump hosts. |
| `Include` | yes | Relative paths resolve against `~/.ssh/`. Globs (`*`, `?`) expand. |

Everything else, including `ProxyCommand`, `Match`, `ForwardAgent`, `LocalForward`/`RemoteForward`, `StrictHostKeyChecking`, `UserKnownHostsFile`, `PreferredAuthentications`, `IdentitiesOnly`, `CertificateFile`, `ControlMaster`/`ControlPath`/`ControlPersist`, `ServerAliveInterval`/`ServerAliveCountMax`, `ConnectTimeout`, `AddressFamily`, `BindAddress`, `LogLevel`, `PubkeyAcceptedAlgorithms`, and `SetEnv`, is silently ignored. Host-key checking is always `AutoAddPolicy` regardless of what your config says, and the keepalive interval is hard-coded to 30 s (see Keepalive and timeouts).

### Setup - Command mode

Paste a shell-style SSH command and the add-on extracts host, port, username, and key path from it. This is convenient when you already copy such a line from a cloud provider or a shared ops doc.

1. Set **Server Type** to `SSH Command`.
2. Paste into **Command**, for example:

   ```text
   ssh -p 2222 -i ~/.ssh/gpu_key alice@gpu01.example.com
   ```

3. Set **Remote Path** and **Server Port** as above. Click **Connect**.

The parser reads the destination (`[user@]host`, or an `ssh://user@host:port` URI) plus `-p` for port, `-i` for key path, `-l` for login name, `-J` for jump hosts, and the same four settings written as `-o Port=`, `-o IdentityFile=`, `-o User=`, `-o ProxyJump=`. Every other ssh option is accepted and ignored, including `-F`: the add-on always reads `~/.ssh/config`. A setting given twice keeps the first value, so `-p` outranks a later `-o Port=`.

Options are matched against the real ssh option list rather than by a leading dash, so an option consumes its own argument. `ssh -p 2222 gpu-alias` connects to `gpu-alias` on port 2222, and `ssh -J me@bastion gpu01` reads `me@bastion` as the jump host rather than as the destination. Options written after the destination count too (`ssh gpu01 -p 2222`), matching ssh itself; anything past a second bare word is the remote command and is ignored.

If the command cannot be parsed, the operator reports why and aborts: an unknown option, an option missing its argument, an unbalanced quote, or no host token at all.

### SSH keys

- Ed25519 and RSA keys work out of the box.
- Encrypted keys prompt for a passphrase at the **terminal that launched Blender**, not in the Blender UI. If your key is passphrase-protected, either use `ssh-agent` or decrypt the key file.
- PuTTY `.ppk` keys are not supported; convert them to OpenSSH or PKCS#8 format first.

### Keepalive and timeouts

| Knob | Value |
| ---- | ----- |
| SSH keepalive | 30 s |
| Connect modal timeout | 60 s |
| Server startup timeout | 16 s |

The keepalive pings the remote every 30 seconds to prevent idle disconnects on NATed links. The UI modal gives up after 60 seconds if the connect has not completed.

### Jump hosts

A solver host that is only routable from a bastion is reached by naming the bastion, and the add-on opens the hops itself. Every SSH-backed type supports it:

- **SSH** and **Docker over SSH**: type the jump host into **Proxy Jump**, in the form `ssh -J` takes it (`[user@]host[:port]`, comma separated for a chain, hops ordered outward from your machine).
- **SSH Command** and **Docker over SSH Command**: put `-J` (or `-o ProxyJump=`) in the command, exactly as you would run it in a shell.
- Either way, leaving it empty falls back to the `ProxyJump` entry `~/.ssh/config` gives for the host, so a host already configured for the terminal needs nothing typed in the panel.

```text
ssh -J bastion.example.com gpu01.internal        # Command mode
bastion.example.com                              # Proxy Jump field
alice@bastion.example.com:2222,inner.internal    # two hops, first one reached directly
```

Each hop is resolved through `~/.ssh/config` the same way the destination is, so a jump host written as an alias brings its own `HostName`, `Port`, `User`, and `IdentityFile`, and a jump host whose own config carries a `ProxyJump` extends the chain in front of itself. A user or port written into the spec overrides what the config says for that alias. A hop with no `IdentityFile` authenticates with your agent and default keys, the same way `ssh` would.

The hops are opened in order, each one tunneled through the one before it, and the connection to the solver host rides the last one. They are torn down with the connection, and on **Disconnect** they close from the far end inward. A hop that refuses the connection reports which one it was (`Jump host bastion.example.com:22 failed: ...`) and closes the hops already opened. A spec that names no host, or one whose jump hosts point back at each other, is refused before anything is dialed.

### Port forwarding and tunnels

The bundled parser ignores `LocalForward`, `RemoteForward`, and `ProxyCommand` (see Supported ssh_config options), so the panel cannot stand up a forwarded port on its own. For a bastion, use Jump hosts above. If the solver host is reachable only over a port you forward yourself, set up the tunnel from a separate terminal first and point the add-on at the local end:

```bash
# Example: reach a solver host behind a bastion via local forward.
ssh -N -L 2222:gpu01.internal:22 bastion.example.com
```

Then in the Backend Communicator, set **Host** to `localhost` and **Port** to `2222`. The add-on talks to `localhost:2222`, the forward carries it through the bastion, and paramiko never needs to know the bastion exists. The same trick works for cloud providers that expose GPU hosts only through a jump host.

Keep the terminal that holds the tunnel open for as long as you want the connection to work. If the tunnel dies, the next add-on request fails with a connection error and you re-run the `ssh -L` command.

### Multiple users on one solver host

Two users on the same solver box collide on the **Server Port** (default `9090`): each Blender client expects the server it starts to be the one listening on that port. To share a box safely:

- Give each user (or each concurrent project) a different **Server Port** in their connection profile (e.g. `9090`, `9091`, `9092`, ...). The server binds whatever the client asks for, so non-overlapping ports let independent simulations run side by side on one GPU.
- Pick a distinct **Remote Path** per user as well. The remote path is where `ppf-cts-server` lives *and* where the solver writes per-run data; two clients sharing a path will stomp each other's checkpoints and PC2 files.
- Remember the box's GPU is shared. Concurrent sims contend for VRAM and CUDA streams, so two heavy scenes on one GPU each run slower than they would alone. Stagger large runs when throughput matters.

The session ID stamped on PC2 files and the remote project directory (see Sessions and recovery) keeps each client from accidentally fetching another user's frames as long as the **Remote Path** is distinct.

### Installing paramiko

The SSH backend requires the `paramiko` Python package. If it is not present, the main panel shows an **Install Paramiko** button that installs it into Blender's `scripts/addons/modules` directory (the same target as the `cbor2` recovery install); click it and wait for the background installer to finish.

UNDER THE HOOD:

**Command-mode parser**

The Command-mode parser splits the pasted string with `shlex` and picks out only these tokens:

- `-p N` -> port
- `-i <path>` -> key path
- `user@host` -> username + host
- the first bare token after `ssh` that is not an option -> host

Every other flag is silently ignored. If no host can be extracted the operator reports `Failed to parse command. Ensure it includes host.` and aborts. The parser never invokes the system `ssh` binary; the parsed fields go straight into paramiko, which is why `-o`, `-J`, and `ProxyCommand` do not work in Command mode.

**`~/.ssh/config` resolution**

When **Host** looks like an alias instead of a DNS name, the add-on parses `~/.ssh/config` (first-match, with `Include` directives resolved) and fills in `HostName`, `Port`, `User`, and `IdentityFile` for the alias. Later matching entries fill in fields earlier entries left blank, so a trailing wildcard `Host *` block provides sensible defaults without overriding explicit blocks. If the config file is missing or the alias is not found, the alias text is used as the hostname verbatim. Only the six keywords listed in Supported ssh_config options are read; the parser tokenizes each non-comment line on whitespace or `=`, matches the first word case-insensitively, and drops the line if the keyword isn't one it recognizes.

**Host-key policy**

Unknown host keys are accepted silently (paramiko `AutoAddPolicy`). This is not hardened against MITM attacks and should not be relied on for untrusted networks.

**Key-loading errors**

`SSHException: not a valid ... key` means the key file is in a format paramiko cannot read. Typical causes: a PuTTY `.ppk` file (convert first), or a modern OpenSSH key written with a cipher paramiko was built without.

**paramiko install path**

The **Install Paramiko** button runs `pip install --target` into Blender's `scripts/addons/modules` directory on a background thread. If paramiko is already installed system-wide, the add-on uses that copy instead; both paths work.

## Docker

The solver runs inside a Docker container. Three UI modes cover the possible locations of the Docker daemon:

| Mode | Daemon | Where it runs |
| ---- | ------ | ------------- |
| **Docker** | Local | The Docker daemon on the Blender machine |
| **Docker over SSH** | Remote | A Docker daemon on an SSH-reachable host |
| **Docker over SSH Command** | Remote | Same as above, SSH fields parsed from an `ssh ...` string |

Figure: Two stacked block diagrams showing the Docker local topology (add-on, daemon, and container all on one workstation) and the Docker over SSH topology (add-on on the workstation, daemon and container on a remote Linux host reached through one SSH session). Blue solid arrows carry lifecycle commands; purple dashed arrows carry the TCP connection to the solver. The container must publish the server port in both rows.

### When to use it

- The solver depends on CUDA/driver versions you do not want to install on the host.
- Your cluster administrator hands you a container instead of shell access.
- Multiple solvers share one GPU host and you want each project isolated.

### Setup - local Docker

1. Set **Server Type** to `Docker`.
2. Leave **Container** at its default, `ppf-contact-solver`, which is the name the `docker run --name` in the project README creates. Change it only if your container carries another name.
3. Leave **Container Path** at its default, `/root/ppf-contact-solver`, which is where the published image puts the built `ppf-cts-server`. Change it only if the solver lives elsewhere inside the container.
4. Set **Server Port** to the TCP port `ppf-cts-server` listens on inside the container.
5. Click **Connect**. If the container exists but is stopped, the add-on starts it for you. A missing container is reported as an error.
6. Once connected, pick **Compute Device**. The published image carries both builds, the CUDA one in `target/release` and the CPU one in `target/cpu/release`, so both rows are offered: GPU needs the container to have been started with `--gpus all`, and CPU runs without a GPU at all and is substantially slower. The choice is applied at Start Server.

Figure: Backend Communicator with **Server Type** set to `Docker`. **Container**, **Container Path**, and **Docker Port** replace the SSH fields. The **Install Docker-Py** banner appears when the vendored module is missing. **Connect** is highlighted.

#### Fields

| Field | Description |
| ----- | ----------- |
| Container | Docker container name. Must already exist. Defaults to `ppf-contact-solver`, the name the README `docker run` creates. |
| Container Path | Working directory inside the container (contains the built `ppf-cts-server` binary). Defaults to `/root/ppf-contact-solver`, where the published image puts it. |
| Compute Device | Whether the GPU or the CPU build inside the container runs. The build listing is read inside the container, so the row is offered once connected. See Choosing the build. |
| GPU Backend | Which accelerator runs when Compute Device is GPU. Offered once connected, and only where the container holds more than one GPU build. See Choosing the build. |
| GPU | Which CUDA device the solver runs on, as the CONTAINER sees them. A container started without `--gpus all` sees a subset of its host's cards, and the list is read inside it. See Choosing a GPU. |
| Server Port | Port inside the container where `ppf-cts-server` listens. |

### Setup - Docker over SSH

The SSH fields from the SSH page are combined with the Docker fields: the add-on opens an SSH session to the remote host and runs every Docker command there.

1. Set **Server Type** to `Docker over SSH`.
2. Fill Host / Port / Username / Key Path as in SSH Custom mode.
3. Check Container and Container Path. They carry the same defaults as local Docker, so a container created by the README `docker run` needs neither changed.
4. Click **Connect**. The add-on verifies that the container exists on the remote host and starts it if it is stopped.

WARNING: The server port must be published on the container (`-p 9090:9090` or equivalent in your compose file). Before **Start Server**, the add-on checks the port mapping on the remote host and refuses to continue if the port is not exposed, the error text tells you exactly which port and container failed. You must fix this on the container side; the add-on cannot publish ports on a container that is already created.

### Setup - Docker over SSH Command

Identical to Docker over SSH, but the SSH parameters come from a pasted command. Set **Server Type** to `Docker over SSH Command` and put the string in **Command** (see the SSH Command section for the parser rules). Container and Container Path are still fields.

### Installing docker-py

The Docker backend requires the `docker` Python package (sometimes called `docker-py`). When it is missing the main panel shows an **Install Docker** button that installs the package into Blender's `scripts/addons/modules` directory on a background thread.

Docker-over-SSH modes also require paramiko; install both if the remote-container path is what you need.

### Troubleshooting

- **`Container 'X' does not exist.`** - the name is wrong or the container was removed on the remote. Run `docker ps -a` on the remote to see what is actually there.
- **`Error starting container 'X'`** - the daemon returned a non-zero exit or the user lacks `docker` group membership on the remote.
- **Server startup timed out.** - the container started but `ppf-cts-server` did not become ready within 16 seconds. Check `server.log` inside the directory set in **Container Path**; the panel prints the last 20 lines automatically.
- **`docker-py` not found** - click **Install Docker** on the main panel and wait for the background installer to finish.

UNDER THE HOOD:

**Local Docker transport**

The local backend talks to the Docker daemon through the standard `docker` Python client and looks up the container by name. A stopped container is started automatically; a missing container aborts connect with `Container 'X' does not exist.`

**Docker over SSH transport**

An SSH session is opened to the remote host and every Docker command (including data transfer) is wrapped in `docker exec -i <container> ...` on that session. During connect the add-on checks that the container exists and starts it if it is stopped; a missing container aborts the connect.

**Port publication check**

Before **Start Server** the add-on runs

```sh
docker port <container> <port>
```

on the Docker-serving host. An empty result aborts with:

> Docker port 9090 is not exposed on container 'ppf-contact-solver'. Please expose the port with '-p 9090:9090' when starting the container.

Fix this on the container side by re-running `docker run -p 9090:9090` (or editing your `compose.yaml`); the add-on cannot publish ports on an existing container.

**Server startup path**

Both Docker modes use the same Unix server-launch path as the SSH backends (see Connections - Under the hood): a small script inside the container launches the resolved `ppf-cts-server` on the configured port and the UI waits up to 16 s for readiness.

## Windows Native

The solver runs directly as a Windows subprocess using a bundled Python interpreter and (optionally) a bundled CUDA runtime. No SSH or Docker is involved, the add-on launches the solver alongside Blender and talks to it over a local TCP socket.

### When to use it

- User workstations running Blender on Windows with a local NVIDIA GPU.
- Bundled deployments that ship the solver next to the add-on.
- Reproducible test rigs where you want the exact shipped Python + CUDA, not whatever the system has.

### Setup

1. Set **Server Type** to `Win Native`.
2. Set **Win Native Path** to the root of your solver install. This is the directory that contains the built `ppf-cts-server.exe` binary (under `target\release\` for a developer build, or under `bin\` in the redistributable bundle layout) plus either a shipped `python\` subfolder (redistributable bundle) or a `build-win-native\python\` subfolder (developer build) for the build worker.
3. Set **Server Port** (default `9090`).
4. Click **Connect**. The add-on verifies `ppf-cts-server.exe` is where it should be, picks up the right Python runtime for the build worker, and launches the solver as a hidden subprocess.
5. The server is launched as part of the connect step, so once **Connect** reports success the server is already running. Pressing **Start Server** is a no-op while it is alive; it re-spawns the subprocess only after a **Stop Server**.

Figure: Backend Communicator with **Server Type** set to `Windows Native`. Only **Solver Path** and **Project Name** appear, with no SSH or Docker fields. **Connect** is highlighted.

### Fields

| Field | Description |
| ----- | ----------- |
| GPU | Which CUDA device the solver runs on. See Choosing a GPU. |
| Win Native Path | Root directory containing `ppf-cts-server.exe` (under `target\release\` for dev builds, under `bin\` for bundles) plus either `python\` (bundle) or `build-win-native\python\` (dev) for the build worker. |
| Server Port | TCP port for `ppf-cts-server.exe`. Default `9090`. |

### Troubleshooting

- **`ppf-cts-server.exe not found in <root>`** - the root points at the wrong directory. It must be the solver checkout root (so that `target\release\ppf-cts-server.exe` resolves) or the bundle root that ships the binary under `bin\`.
- **`Embedded Python not found ...`** - the add-on could not find a Python runtime under the root. Either rebuild the dev tree, or download and unpack the bundle zip.
- **CUDA DLL load errors** - on the shipped bundle, the solver relies on the system CUDA runtime. Install a matching CUDA version, or switch to the developer build which ships its own CUDA.

UNDER THE HOOD:

**Layout auto-detection**

Connect picks one of two layouts by looking for `python.exe`:

#### Dev layout

```text
<root>/
  build-win-native/
    python/python.exe
    cuda/bin/*.dll
  target/release/
    ppf-cts-server.exe       # Rust server binary
  src/kernels/build/lib/
```

Used when you built the server from source. The Python interpreter for the build worker is `build-win-native\python\python.exe`, `CUDA_PATH` is set to `build-win-native\cuda`, and the launcher prepends, in order, `build-win-native\python`, `target\release`, `src\cpp\build\lib`, and `build-win-native\cuda\bin` to `PATH`.

#### Bundle layout

```text
<root>/
  bin/
    ppf-cts-server.exe       # Rust server binary
  python/python.exe
```

Used by a shipped redistributable. The Python interpreter for the build worker is `root\python\python.exe`, `CUDA_PATH` is not set (CUDA is expected on the system `PATH`), and the launcher prepends `root\python`, `root\bin`, and `root\target\release` to `PATH`.

If neither interpreter is present, connect fails with:

> Embedded Python not found in \<build\_dir\> or \<root\>

**Subprocess environment**

The subprocess inherits your current environment with a few additions:

- `PATH` is prepended with the layout-specific directories above; the existing `PATH` is appended so system tools still work.
- `PYTHONPATH` begins with `<root>` so the build worker can import the bundled `frontend` package.
- `CUDA_PATH` is added on the dev layout only.

**Launch flags**

The solver is launched as `ppf-cts-server.exe --port <port>` with no visible console window, so nothing appears behind Blender.

**Shutdown**

On disconnect (or **Stop Server**), the add-on asks the subprocess to terminate and waits up to 5 seconds; if it is still alive, it is killed. The Unix `pkill -f ppf-cts-server` path is not used; the backend holds the Windows process handle directly.

**Why Start Server is usually a no-op**

The subprocess is started as part of the Connect step, so by the time Connect reports success the server is already running and pressing **Start Server** has nothing to do. After a **Stop Server**, however, the process handle is cleared and **Start Server** re-spawns `ppf-cts-server.exe` via the same launcher used at Connect.

## Connection profiles

A connection profile is a TOML entry that captures every field of the **Connections** panel for one connection. Profiles let you switch between hosts without re-typing credentials and share presets across a team.

Profiles live in a plain `.toml` file; one file can contain many profiles as top-level tables.

IMPORTANT: **You do not write these TOML files by hand.** Fill in the connection fields in the panel, then click the **Save** icon (floppy disk) at the top-right of the profile row. The add-on creates the `.toml` file for you on first save and appends or overwrites the current entry on subsequent saves. The file format documented below is shown only so you can inspect or share the output; the intended authoring path is always through the UI.

Figure: The **Save** icon (floppy disk, highlighted in red) at the top-right of the profile row. Click it to write the on-screen connection fields to a `.toml` file, creating the file the first time and overwriting the currently selected entry after a profile is loaded.

### The profile row

At the top of the **Backend Communicator** panel, four buttons manage the active file:

| Button | Effect |
| ------ | ------ |
| **Open** | Pick a `.toml` file; the **Profile** dropdown fills with the entries found in that file. |
| **Clear** | Forget the loaded file; connection fields stay as they are. |
| **Reload** | Re-apply the currently selected entry, discarding any edits you made since load. |
| **Save** | Write the current field values back into the file under the currently selected entry (or a new one if none is selected). |

**Save** takes whatever is on screen and writes it to disk. Entries in the file that you are not currently editing are preserved.

Figure: The profile row. **Open Profile** picks a `.toml` file; once a file is loaded, this row changes to a profile dropdown plus Open / Clear / Reload / Save icon buttons. The save icon on the right writes the current on-screen fields back into the file.

NOTE: The Save button does not preserve comments or formatting in the TOML file. If you keep comments in your profile file, re-save from a backup or edit the file by hand rather than round-tripping through the button.

### File format

The sections below describe the on-disk layout for reference. Remember that this file is **generated by the Save icon**, not authored by hand. Open it in an editor only to inspect, diff, or share entries; round-tripping through the Save button is the supported edit path.

Each profile is a top-level table. The table name is free-form (quote it if it contains spaces or other non-bare characters). Inside the table, one required discriminator and the connection fields below:

| TOML key | Notes |
| -------- | ----- |
| `type` | Required. One of `SSH`, `SSH Command`, `Docker`, `Docker over SSH`, `Docker over SSH Command`, `Windows Native`, `macOS Native`, `Linux Native`. |
| `host` | SSH host / alias. |
| `port` | SSH port. |
| `username` | SSH user. |
| `key_path` | Private key path. `~` is expanded when used. |
| `command` | Raw `ssh ...` string for Command modes. |
| `container` | Docker container name. |
| `remote_path` | Remote solver directory for SSH. |
| `docker_path` | Solver directory inside a Docker container. |
| `win_native_path` | Windows solver root. |
| `mac_native_path` | macOS solver root. |
| `linux_native_path` | Linux solver root. |
| `solver_gpu` | CUDA device index for the solver server, or `-1` to set no `CUDA_VISIBLE_DEVICES`. |
| `solver_gpu_uuid` | Stable UUID of the selected GPU. Written with `solver_gpu`; preferred when the host reorders indices. |
| `docker_port` | Server TCP port (1024-65535). |

A profile that names `type = "Local"` still loads. It is applied as the native type of the platform reading it (`Linux Native` on Linux, `Windows Native` on Windows, `macOS Native` on macOS), and a `local_path` beside it lands on that type's path key, unless the profile already carries a value for that key. A profile file is something you wrote and keep, so a name the panel no longer offers is mapped rather than rejected.

Unknown keys are silently ignored, so it is safe to sprinkle comments or future additions in the file.

### Example

```toml
# connections.toml -- one entry per environment
[Workstation Linux]
type = "Linux Native"
linux_native_path = "~/ppf-contact-solver"
docker_port = 9090

[LocalDocker]
type = "Docker"
container = "ppf-contact-solver"
docker_path = "/root/ppf-contact-solver"
docker_port = 9090

[GPU01]
type = "SSH"
host = "gpu01.example.com"
port = 22
username = "alice"
key_path = "~/.ssh/id_ed25519"
remote_path = "/home/alice/ppf-contact-solver"
docker_port = 9090

[GPU01 via command]
type = "SSH Command"
command = "ssh -p 22 -i ~/.ssh/id_ed25519 alice@gpu01.example.com"
remote_path = "/home/alice/ppf-contact-solver"
docker_port = 9090

[GPU01 Docker]
type = "Docker over SSH"
host = "gpu01.example.com"
port = 22
username = "alice"
key_path = "~/.ssh/id_ed25519"
container = "ppf-contact-solver"
docker_path = "/root/ppf-contact-solver"
docker_port = 9090

[Workstation Windows]
type = "Windows Native"
win_native_path = "C:\\Users\\alice\\ppf-win"
docker_port = 9090
```

Remember to escape backslashes in Windows paths (`\\`) or use forward slashes.

### Loading a profile

1. Click **Open** and pick the `.toml` file.
2. The **Profile** dropdown now lists every top-level table, sorted alphabetically.
3. Pick an entry; the connection fields fill in from the table.
4. Edit anything you like, then **Connect** as usual. Your edits stay in the UI; **Reload** reverts them to the file's version, **Save** writes them back.

### Related profile types

The same TOML machinery drives **scene profiles** and **material profiles** elsewhere in the add-on. They live in separate files and are managed from their own panels; only the connection profile is documented here, but the Save/Reload semantics are identical.

## Running commands on the remote

Three MCP tools run commands against whichever host the active connection points at:

- `execute_shell_command(shell_command, use_shell=True)`: free-form shell command on the solver host (Windows Native, macOS Native and Linux Native: the machine Blender runs on; SSH and Docker: the remote host or its container). Use this when no dedicated tool covers the task.
- `execute_server_command(server_script)`: a `--key value` argument string sent as a TCMD query to the running `ppf-cts-server`. Narrower than the shell tool; reach for it when the solver exposes the subcommand on its TCMD surface.
- `git_pull_remote()`, `compile_project()`, `install_paramiko()`, `install_docker()`: dedicated wrappers for the most common remote operations. Prefer these over re-typing the shell command.

All of these require an active, non-busy connection. They fail fast while a transfer or run is in progress. See Debug tooling for the full shell-command semantics and `use_shell` flag.

UNDER THE HOOD:

**Operators**

| Button | Operator `bl_idname` |
| ------ | -------------------- |
| Open   | `ssh.open_profile`   |
| Clear  | `ssh.clear_profile`  |
| Reload | `ssh.reload_profile` |
| Save   | `ssh.save_profile`   |

**Save behavior**

**Save** loads the file on disk, replaces only the currently selected entry with the on-screen values, and writes the whole file back. Other entries are preserved, but the rewrite only handles scalars, lists, and arrays of tables, so comments and original formatting in the input file are lost. This is why the user-facing note above warns about comment loss.

**`type` validation**

The `type` value in each TOML entry must exactly match one of the server-type strings listed in the **File format** table, or the retired `Local`, which loads as this platform's native type. Any other value is rejected at load time.
