# Connections

This document condenses the Blender add-on connection docs: the Connections overview (index.md), and the per-backend pages for Local, SSH, Docker, Windows Native, and Connection Profiles.

## Overview

The add-on talks to a solver process over one of several transports. Pick the one that matches where the solver runs.

| Type | Use when | See |
| ---- | -------- | --- |
| **Local** | Solver runs on the same machine as Blender | Local |
| **SSH** | Solver runs on a remote Linux host, credentials entered as fields | SSH |
| **SSH Command** | Same as SSH, but configured by pasting an `ssh ...` shell command | SSH |
| **Docker** | Solver runs inside a container on the local Docker daemon | Docker |
| **Docker over SSH** | Solver runs inside a container on a remote Docker host | Docker |
| **Docker over SSH Command** | Docker-over-SSH configured from a shell `ssh ...` command | Docker |
| **Windows Native** | Solver runs as a Windows subprocess using a bundled Python + CUDA | Windows |

All types share the same server-side protocol (TCP, version `0.02`) and the same UI flow in the panel: **Connect** -> **Start Server** -> transfer data -> **Run** -> **Fetch**.

Figure: Five stacked block diagrams showing where each piece lives for the five connection types, with blue solid arrows for lifecycle commands (start / stop / exec / port check) and purple dashed arrows for the TCP connection to `server.py`. The three Docker sub-modes are broken out separately in Docker.

### What happens when you connect

Starting a solver session is two button presses:

1. **Connect** opens the transport to wherever the solver lives (a local process, an SSH session, or a Docker container) and checks that a `server.py` is actually present at the path you configured. If the check fails, the connection is dropped and an error is reported.
2. **Start Server** launches `server.py` on the remote side and waits up to **16 seconds** for it to report that it is ready to accept work. If the server prints an error during startup, the panel reports the error immediately; if it simply never becomes ready, you get a timeout and a tail of its log.

The panel stays responsive while this happens, the actual work runs on a background thread and the UI polls it several times a second.

Windows Native is slightly different: it launches the server directly as part of the Connect step, so **Start Server** is effectively a no-op on that backend.

TIP: **Save the connection once it works.** As soon as you have a successful **Connect**, click the **Save** icon on the profile row to write the current fields to a `.toml` file. Next session, **Open** the file, pick the entry, and every field auto-fills, with no retyping host, key path, container, or port. See Connection Profiles for the full workflow.

### Port usage at a glance

| Port | Role | Default |
| ---- | ---- | ------- |
| Server | Solver TCP listener (`server.py`) | `9090` |
| MCP | MCP Streamable HTTP server (for AI integration) | `9633` |

Only the server port crosses the transport boundary; MCP is local to the machine running Blender. The server port is configurable per connection.

NOTE: For Docker-over-SSH, the server port must be published on the container (`-p 9090:9090`). The add-on checks this before launching the server and refuses to continue if the port is not exposed.

UNDER THE HOOD:

**Non-blocking UI**

The panel does not freeze while Connect or Start Server is running. Work happens on a background thread and the panel refreshes several times a second, so any status or error reported by the background work appears in the panel promptly.

**Connect step**

Connect opens the configured transport (SSH session, Docker client, or a local subprocess) and verifies that `server.py` is present at the configured path. If that verification fails, the transport is torn down before the error is reported.

**Start Server step**

On Unix-family backends (Local, SSH, Docker, Docker over SSH) the add-on launches the server via a small script that activates `$HOME/.local/share/ppf-cts/venv` if it exists and then runs:

```sh
nohup python3 server.py --port <port> > server.log 2>&1 &
```

The UI waits up to **16 seconds** for the server to announce it is ready. If a line containing `ERROR` or `FAILED` appears first, the wait aborts with that message; on plain timeout, the panel prints the last 20 lines of `server.log`.

Windows Native launches the server directly during the Connect step. See Windows - Under the hood.

**Docker port pre-launch check**

Before **Start Server** on Docker-over-SSH, the add-on checks that the configured server port is published on the container. If it is not, the operator aborts with:

> Docker port 9090 is not exposed on container 'ppf-dev'. Please expose the port with '-p 9090:9090' when starting the container.

The add-on cannot publish a port on an existing container; this has to be fixed on the container side (for example by re-running `docker run -p` or editing `compose.yaml`).

## Local

The solver runs on the same **Linux** machine as Blender. This is the simplest backend and the right default for single-workstation development. On Windows, use the Windows Native backend instead; macOS is not supported as a solver backend (the solver requires CUDA).

### When to use it

- You are running Blender on Linux with a local NVIDIA GPU and a working solver checkout.
- You are iterating on solver code and want the fastest possible disconnect/reconnect turnaround.
- You do not want to pay the cost of SSH or Docker for every transfer.

### Setup

1. Set **Server Type** to `Local`.
2. Fill **Local Path** with the solver checkout (the directory that contains `server.py`).
3. Set **Project Name** on the main panel.
4. Click **Connect**. The add-on checks that `server.py` exists at the path you gave it.
5. Click **Start Server**. The panel waits a few seconds for the server to report that it is ready.

Figure: Backend Communicator with **Server Type** set to `Local`. Only **Path** and **Project Name** show up; no SSH, Docker, or Windows-native fields. **Connect** is highlighted.

TIP: If you ran the solver's installer, it may have created a Python virtual environment at `$HOME/.local/share/ppf-cts/venv`. When the add-on finds that venv it activates it automatically before launching the server, you do not need to do anything.

### Fields

| Field | Description |
| ----- | ----------- |
| Local Path | Filesystem path to the solver checkout. Default `~/ppf-contact-solver`. |
| Server Port | TCP port for `server.py`. Default `9090`; range 1024-65535. |

### Dependencies

Local mode requires neither `paramiko` nor `docker-py`. The main panel's **Install Paramiko** and **Install Docker** buttons are only relevant for SSH and Docker modes.

### Troubleshooting

- **"Remote path not found (.../server.py)"** - the path you entered does not contain `server.py`. Point it at the checkout root, not the `build/` or `src/` subdirectory.
- **Server startup timed out.** - the solver launched but did not report readiness within 16 seconds. Check `server.log` inside the solver directory; the panel also prints the last 20 lines when the timeout fires.
- **Port already in use.** - another solver (or a stale `server.py`) is already bound to the port. Click **Stop Server** first, or change the Server Port.

UNDER THE HOOD:

**Launch script**

Local mode launches the server with a bash script (`nohup`, `source .../bin/activate`, `python3`). That is the same launch path the SSH and Docker backends use. The script is `bash`-only, which is why Local mode is Linux-only; Windows goes through the Windows Native backend and macOS is not supported (the solver requires CUDA).

**Shared port field**

The **Server Port** field is the same underlying property for every connection type. The label in profile TOML files is `docker_port` for historical reasons, even on Local connections.

**Virtual environment activation**

Local mode reuses the same launch script as the SSH and Docker backends: it sources `$HOME/.local/share/ppf-cts/venv/bin/activate` if that file exists, otherwise it falls back to system `python3`. The solver's own install scripts are responsible for creating the venv; the add-on never creates or modifies it.

**File transfer fast path**

On local connections, file transfers copy directly on disk instead of going through the solver TCP socket: no pickle overhead, and much faster than the SSH or Docker paths. The trade-off is cosmetic: the panel does not display a bandwidth figure while a local transfer is in progress. That is expected on this backend.

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
| Remote Path | `/root/ppf-contact-solver` | Remote solver directory (must contain `server.py`). |
| Server Port | `9090` | Port on the remote host where `server.py` listens. |

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
| `Include` | yes | Relative paths resolve against `~/.ssh/`. Globs (`*`, `?`) expand. |

Everything else, including `ProxyJump`/`ProxyCommand`, `Match`, `ForwardAgent`, `LocalForward`/`RemoteForward`, `StrictHostKeyChecking`, `UserKnownHostsFile`, `PreferredAuthentications`, `IdentitiesOnly`, `CertificateFile`, `ControlMaster`/`ControlPath`/`ControlPersist`, `ServerAliveInterval`/`ServerAliveCountMax`, `ConnectTimeout`, `AddressFamily`, `BindAddress`, `LogLevel`, `PubkeyAcceptedAlgorithms`, and `SetEnv`, is silently ignored. If your workflow depends on any of them (for example, reaching a host only through a bastion with `ProxyJump`), the add-on cannot connect and there is no workaround inside the panel. Host-key checking is always `AutoAddPolicy` regardless of what your config says, and the keepalive interval is hard-coded to 30 s (see Keepalive and timeouts).

### Setup - Command mode

Paste a shell-style SSH command and the add-on extracts host, port, username, and key path from it. This is convenient when you already copy such a line from a cloud provider or a shared ops doc.

1. Set **Server Type** to `SSH Command`.
2. Paste into **Command**, for example:

   ```text
   ssh -p 2222 -i ~/.ssh/gpu_key alice@gpu01.example.com
   ```

3. Set **Remote Path** and **Server Port** as above. Click **Connect**.

The parser understands the most common flags: `-p` for port, `-i` for key path, and `user@host`. Advanced flags like `-o`, `-J`, or `ProxyCommand` are **not** honored, if you need those, use Custom mode and let `~/.ssh/config` provide them.

If the command cannot be parsed (no host token found), the operator reports an error and aborts.

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

### Port forwarding and tunnels

The bundled parser ignores `LocalForward`, `RemoteForward`, `ProxyJump`, and `ProxyCommand` (see Supported ssh_config options), so the panel cannot stand up an SSH tunnel on its own. If the solver host is only reachable through a bastion or over a forwarded port, set up the tunnel from a separate terminal first and point the add-on at the local end:

```bash
# Example: reach a solver host behind a bastion via local forward.
ssh -N -L 2222:gpu01.internal:22 bastion.example.com
```

Then in the Backend Communicator, set **Host** to `localhost` and **Port** to `2222`. The add-on talks to `localhost:2222`, the forward carries it through the bastion, and paramiko never needs to know the bastion exists. The same trick works for cloud providers that expose GPU hosts only through a jump host.

Keep the terminal that holds the tunnel open for as long as you want the connection to work. If the tunnel dies, the next add-on request fails with a connection error and you re-run the `ssh -L` command.

### Multiple users on one solver host

Two users on the same solver box collide on the **Server Port** (default `9090`): each Blender client expects the server it starts to be the one listening on that port. To share a box safely:

- Give each user (or each concurrent project) a different **Server Port** in their connection profile (e.g. `9090`, `9091`, `9092`, ...). The server binds whatever the client asks for, so non-overlapping ports let independent simulations run side by side on one GPU.
- Pick a distinct **Remote Path** per user as well. The remote path is where `server.py` lives *and* where the solver writes per-run data; two clients sharing a path will stomp each other's checkpoints and PC2 files.
- Remember the box's GPU is shared. Concurrent sims contend for VRAM and CUDA streams, so two heavy scenes on one GPU each run slower than they would alone. Stagger large runs when throughput matters.

The session ID stamped on PC2 files and the remote project directory (see Sessions and recovery) keeps each client from accidentally fetching another user's frames as long as the **Remote Path** is distinct.

### Installing paramiko

The SSH backend requires the `paramiko` Python package. If it is not present, the main panel shows an **Install Paramiko** button that installs it into the add-on's private library directory; click it and wait for the background installer to finish.

UNDER THE HOOD:

**Command-mode parser**

The Command-mode parser splits the pasted string with `shlex` and picks out only these tokens:

- `-p N` -> port
- `-i <path>` -> key path
- `user@host` -> username + host
- the first bare token after `ssh` that is not an option -> host

Every other flag is silently ignored. If no host can be extracted the operator reports `Failed to parse command. Ensure it includes host.` and aborts. The parser never invokes the system `ssh` binary; the parsed fields go straight into paramiko, which is why `-o`, `-J`, and `ProxyCommand` do not work in Command mode.

**`~/.ssh/config` resolution**

When **Host** looks like an alias instead of a DNS name, the add-on parses `~/.ssh/config` (first-match, with `Include` directives resolved) and fills in `HostName`, `Port`, `User`, and `IdentityFile` for the alias. Later matching entries fill in fields earlier entries left blank, so a trailing wildcard `Host *` block provides sensible defaults without overriding explicit blocks. If the config file is missing or the alias is not found, the alias text is used as the hostname verbatim. Only the six keywords listed in Supported ssh_config options are read; the parser tokenises each non-comment line on whitespace or `=`, matches the first word case-insensitively, and drops the line if the keyword isn't one it recognises.

**Host-key policy**

Unknown host keys are accepted silently (paramiko `AutoAddPolicy`). This is not hardened against MITM attacks and should not be relied on for untrusted networks.

**Key-loading errors**

`SSHException: not a valid ... key` means the key file is in a format paramiko cannot read. Typical causes: a PuTTY `.ppk` file (convert first), or a modern OpenSSH key written with a cipher paramiko was built without.

**paramiko install path**

The **Install Paramiko** button runs `pip install` into the add-on's bundled `lib/` directory on a background thread. If paramiko is already installed system-wide, the add-on uses that copy instead; both paths work.

## Docker

The solver runs inside a Docker container. Three UI modes cover the possible locations of the Docker daemon:

| Mode | Daemon | Where it runs |
| ---- | ------ | ------------- |
| **Docker** | Local | The Docker daemon on the Blender machine |
| **Docker over SSH** | Remote | A Docker daemon on an SSH-reachable host |
| **Docker over SSH Command** | Remote | Same as above, SSH fields parsed from an `ssh ...` string |

Figure: Two stacked block diagrams showing the Docker local topology (add-on, daemon, and container all on one workstation) and the Docker over SSH topology (add-on on the workstation, daemon and container on a remote Linux host reached through one SSH session). Blue solid arrows carry lifecycle commands; purple dashed arrows carry the TCP connection to `server.py`. The container must publish the server port in both rows.

### When to use it

- The solver depends on CUDA/driver versions you do not want to install on the host.
- Your cluster administrator hands you a container instead of shell access.
- Multiple solvers share one GPU host and you want each project isolated.

### Setup - local Docker

1. Set **Server Type** to `Docker`.
2. Fill **Container** with the container name (default `ppf-dev`).
3. Fill **Docker Path** with the working directory inside the container (default `/root/ppf-contact-solver`, containing `server.py`).
4. Set **Server Port** to the TCP port `server.py` listens on inside the container.
5. Click **Connect**. If the container exists but is stopped, the add-on starts it for you. A missing container is reported as an error.

Figure: Backend Communicator with **Server Type** set to `Docker`. **Container**, **Container Path**, and **Docker Port** replace the SSH fields. The **Install Docker-Py** banner appears when the vendored module is missing. **Connect** is highlighted.

#### Fields

| Field | Description |
| ----- | ----------- |
| Container | Docker container name. Must already exist. |
| Docker Path | Working directory inside the container (contains `server.py`). |
| Server Port | Port inside the container where `server.py` listens. |

### Setup - Docker over SSH

The SSH fields from the SSH page are combined with the Docker fields: the add-on opens an SSH session to the remote host and runs every Docker command there.

1. Set **Server Type** to `Docker over SSH`.
2. Fill Host / Port / Username / Key Path as in SSH Custom mode.
3. Fill Container and Docker Path.
4. Click **Connect**. The add-on verifies that the container exists on the remote host and starts it if it is stopped.

WARNING: The server port must be published on the container (`-p 9090:9090` or equivalent in your compose file). Before **Start Server**, the add-on checks the port mapping on the remote host and refuses to continue if the port is not exposed, the error text tells you exactly which port and container failed. You must fix this on the container side; the add-on cannot publish ports on a container that is already created.

### Setup - Docker over SSH Command

Identical to Docker over SSH, but the SSH parameters come from a pasted command. Set **Server Type** to `Docker over SSH Command` and put the string in **Command** (see the SSH Command section for the parser rules). Container and Docker Path are still fields.

### Installing docker-py

The Docker backend requires the `docker` Python package (sometimes called `docker-py`). When it is missing the main panel shows an **Install Docker** button that installs the package into the add-on's private library on a background thread.

Docker-over-SSH modes also require paramiko; install both if the remote-container path is what you need.

### Troubleshooting

- **`Container 'X' does not exist.`** - the name is wrong or the container was removed on the remote. Run `docker ps -a` on the remote to see what is actually there.
- **`Error starting container 'X'`** - the daemon returned a non-zero exit or the user lacks `docker` group membership on the remote.
- **Server startup timed out.** - the container started but `server.py` did not become ready within 16 seconds. Check `server.log` inside the directory set in **Container Path**; the panel prints the last 20 lines automatically.
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

> Docker port 9090 is not exposed on container 'ppf-dev'. Please expose the port with '-p 9090:9090' when starting the container.

Fix this on the container side by re-running `docker run -p 9090:9090` (or editing your `compose.yaml`); the add-on cannot publish ports on an existing container.

**Server startup path**

Both Docker modes use the same Unix server-launch path as the SSH and Local backends (see Connections - Under the hood): a small script inside the container launches `server.py` on the configured port and the UI waits up to 16 s for readiness.

## Windows Native

The solver runs directly as a Windows subprocess using a bundled Python interpreter and (optionally) a bundled CUDA runtime. No SSH or Docker is involved, the add-on launches the solver alongside Blender and talks to it over a local TCP socket.

### When to use it

- User workstations running Blender on Windows with a local NVIDIA GPU.
- Bundled deployments that ship the solver next to the add-on.
- Reproducible test rigs where you want the exact shipped Python + CUDA, not whatever the system has.

### Setup

1. Set **Server Type** to `Win Native`.
2. Set **Win Native Path** to the root of your solver install. This is the directory that contains `server.py` plus either a shipped `python\` subfolder (redistributable bundle) or a `build-win-native\python\` subfolder (developer build).
3. Set **Server Port** (default `9090`).
4. Click **Connect**. The add-on verifies `server.py` is where it should be, picks up the right Python runtime, and launches the solver as a hidden subprocess.
5. The server is launched as part of the connect step, so once **Connect** reports success the server is already running. Pressing **Start Server** afterwards is a no-op on this backend.

Figure: Backend Communicator with **Server Type** set to `Windows Native`. Only **Solver Path** and **Project Name** appear, with no SSH or Docker fields. **Connect** is highlighted.

### Fields

| Field | Description |
| ----- | ----------- |
| Win Native Path | Root directory containing `server.py` plus either `python\` (bundle) or `build-win-native\python\` (dev). |
| Server Port | TCP port for `server.py`. Default `9090`. |

### Troubleshooting

- **`server.py not found: <root>\server.py`** - the root points at the wrong directory. It must be the solver checkout root, not the `python\` subdirectory.
- **`Embedded Python not found ...`** - the add-on could not find a Python runtime under the root. Either rebuild the dev tree, or download and unpack the bundle zip.
- **CUDA DLL load errors** - on the shipped bundle, the solver relies on the system CUDA runtime. Install a matching CUDA version, or switch to the developer build which ships its own CUDA.

UNDER THE HOOD:

**Layout auto-detection**

Connect picks one of two layouts by looking for `python.exe`:

#### Dev layout

```text
<root>/
  server.py
  build-win-native/
    python/python.exe
    cuda/bin/*.dll
  target/release/          # Rust solver binaries
  src/cpp/build/lib/
```

Used when you built the solver from source. The Python interpreter is `build-win-native\python\python.exe`, `CUDA_PATH` is set to `build-win-native\cuda`, and the launcher prepends, in order, `build-win-native\python`, `build-win-native\cuda\bin`, `target\release`, and `src\cpp\build\lib` to `PATH`.

#### Bundle layout

```text
<root>/
  server.py
  python/python.exe
  bin/                     # native shared libraries
  target/release/
```

Used by a shipped redistributable. The Python interpreter is `root\python\python.exe`, `CUDA_PATH` is not set (CUDA is expected on the system `PATH`), and the launcher prepends `root\python`, `root\bin`, and `root\target\release`.

If neither interpreter is present, connect fails with:

> Embedded Python not found in \<build\_dir\> or \<root\>

**Subprocess environment**

The subprocess inherits your current environment with a few additions:

- `PATH` is prepended with the layout-specific directories above; the existing `PATH` is appended so system tools still work.
- `PYTHONPATH` begins with `<root>` so `server.py` can import its own modules.
- `CUDA_PATH` is added on the dev layout only.

**Launch flags**

The solver is launched as `python.exe server.py --port <port>` with no visible console window, so nothing appears behind Blender.

**Shutdown**

On disconnect (or **Stop Server**), the add-on asks the subprocess to terminate and waits up to 5 seconds; if it is still alive, it is killed. The Unix `pkill -f server.py` path is not used; the backend holds the Windows process handle directly.

**Why Start Server is a no-op**

The subprocess is started as part of the Connect step, so by the time Connect reports success the server is already running. Pressing **Start Server** afterwards has nothing to do.

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

Each profile is a top-level table. The table name is free-form (quote it if it contains spaces or other non-bare characters). Inside the table, one required discriminator and up to eleven connection fields:

| TOML key | Notes |
| -------- | ----- |
| `type` | Required. One of `Local`, `SSH`, `SSH Command`, `Docker`, `Docker over SSH`, `Docker over SSH Command`, `Windows Native`. |
| `host` | SSH host / alias. |
| `port` | SSH port. |
| `username` | SSH user. |
| `key_path` | Private key path. `~` is expanded when used. |
| `command` | Raw `ssh ...` string for Command modes. |
| `container` | Docker container name. |
| `remote_path` | Remote solver directory for SSH. |
| `docker_path` | Solver directory inside a Docker container. |
| `local_path` | Local solver directory. |
| `win_native_path` | Windows solver root. |
| `docker_port` | Server TCP port (1024-65535). |

Unknown keys are silently ignored, so it is safe to sprinkle comments or future additions in the file.

### Example

```toml
# connections.toml -- one entry per environment
[Local]
type = "Local"
local_path = "~/ppf-contact-solver"
docker_port = 9090

[LocalDocker]
type = "Docker"
container = "ppf-dev"
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
container = "ppf-dev"
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

- `execute_shell_command(shell_command, use_shell=True)`: free-form shell command on the solver host (Local: the Blender machine; SSH and Docker: the remote host or its container; Windows Native: the Windows solver host). Use this when no dedicated tool covers the task.
- `execute_server_command(server_script)`: a server-side script the remote `server.py` already understands. Narrower than the shell tool; reach for it when the solver exposes the subcommand.
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

The `type` value in each TOML entry must exactly match one of the server-type strings listed in the **File format** table. Any other value is rejected at load time.
