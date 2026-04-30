# 🩹 Troubleshooting

Problems users hit in practice, grouped by where they show up. Quoted
strings are exact panel/log text. For full tracebacks read `server.log`,
`progress.log`, or the Blender system console.

## Installation

### "No module named paramiko" or "No module named docker"

Vendored copies are populated on demand. Click **Install Paramiko** (SSH
backends) or **Install Docker** (Docker backends).

### Install operator hangs or fails

`pip` exited non-zero or hit the 120 s internal timeout. Read its stderr
in the Blender system console. The add-on directory must be writable; as
a fallback, install manually with `python -m pip install --target
<addon>/lib paramiko docker`.

## Connection: SSH

### SSH authentication failed

Verify **SSH Key** points at the private key (not the `.pub`), it has
no passphrase or your agent has it loaded, and the matching public key
is in `~/.ssh/authorized_keys` on the remote.

### "Failed to parse command. Ensure it includes host."

Only `ssh -p <port> -i <key> user@host` parses. For `-o`, `-J`, or
`ProxyCommand` setups, switch to **SSH Custom** mode and let
`~/.ssh/config` fill in the blanks.

### `~/.ssh/config` host works in terminal but not in the add-on

The parser understands only `Host`, `HostName`, `Port`, `User`,
`IdentityFile`, and `Include`. `ProxyJump`, `ProxyCommand`, `Match`,
and certificates are ignored. Bring the tunnel up from a terminal
(`ssh -L 2222:gpu01.example.com:22 bastion.example.com`) and point the
add-on at `localhost:2222`. See {ref}`Supported ssh_config options
<supported-ssh-config-options>`.

## Connection: Docker

### "Container '...' does not exist"

Run `docker ps -a` on the daemon host and correct the **Container**
field, or create/start the container.

### "Error starting container '...'"

The container exists but `docker start` failed. Run it manually on the
host and read the error; common causes are publish-time port conflicts
or the user not being in the `docker` group.

### "Docker port N is not exposed on container '...'"

The add-on cannot publish ports on an existing container. Recreate it
with `docker run -p <port>:<port>` (or update `compose.yaml`).

### Docker daemon or permission errors

`docker version` should work as the same user. On Linux: `sudo usermod
-aG docker $USER`, then log out and back in.

## Connection: Windows Native

### "Solver path is not set" / "server.py not found under the solver root"

Set **Solver Path** to the directory that contains `server.py`.

### "Embedded Python not found"

Neither the dev layout (`build-win-native\python\python.exe`) nor the
bundle layout (`python\python.exe`) resolved. Rebuild the dev tree, or
unpack the shipped bundle zip next to `server.py`.

### CUDA DLL load errors

`server.log` shows a missing CUDA runtime DLL. The bundle layout
expects CUDA on the system `PATH`; install a matching CUDA runtime, or
switch to the developer build (which ships its own).

### "Remote path not found (.../server.py)"

The post-connect path check failed. Point **Remote Path** / **Path** /
**Container Path** / **Solver Path** at the directory containing
`server.py`, not its parent or a `build/` subdirectory.

## Connection profiles

### Profile dropdown empty after Open

The TOML is malformed, unreadable, or empty. `load_profiles` swallows
all exceptions. Run the file through any TOML validator; common causes
are unclosed quotes or unescaped backslashes in Windows paths (use
`\\` or forward slashes).

### Profile loads but fields stay blank

The `type` value does not match one of `Local`, `SSH`, `SSH Command`,
`Docker`, `Docker over SSH`, `Docker over SSH Command`, `Windows
Native`. Case matters.

:::{note}
**Save** rewrites the whole TOML; comments and original formatting are
lost on round-trip. Keep a backup if comments matter.
:::

## Server startup

### Status stuck on "Waiting for server start..."

You connected but did not click **Start Server on Remote**, or `server.py` exited
before booting. Click **Start Server on Remote**; if it then times out, see the
next entry.

### "Server startup timed out"

Sixteen seconds passed without a ready marker. The panel pastes the
last 20 lines of `server.log`. Usual causes:

- venv missing at `$HOME/.local/share/ppf-cts/venv`
- CUDA driver missing or mismatched
- the bound port is already in use (change **Server Port**)

### "Server startup failed" with a log line

`progress.log` emitted `ERROR` or `FAILED` during startup. Open the
full log on the remote; the real error (usually a missing module or
failed import) is higher up.

### "Failed to launch server"

The launch script never started: permission denied, read-only working
directory, or no `python3` on `$PATH`. Make the remote path writable
and confirm `python3` resolves via the venv or `$PATH`.

### Status: "Protocol version mismatch"

The server's wire version does not match the add-on (`0.02`). Rebuild
the solver from a matching revision, or update the add-on.

## Object groups and pins

### "Maximum number of groups reached"

The cap is 32 (`N_MAX_GROUPS`). Delete or merge unused groups.

### "Object '...' is already in another group"

Each object's UUID can live in exactly one active group (the encoder
uses it as the routing key). Remove it from the other group first.

### "Object '...' is library-linked and cannot be assigned"

Linked data blocks are read-only and cannot carry the add-on's UUID
property. `Object > Make Local...` first.

### Pin operator errors in Edit Mode

- **"No active edit object"** - enter Edit Mode on a mesh/curve in the
  active group.
- **"Name cannot be empty"** - type a pin name.
- **"No vertices selected"** - select at least one vertex (or curve
  control point).
- **"'...' is not in this group"** - add the object via **Add Selected
  Objects** first.

## Transfer, run, fetch

### "Mesh topology changed since last transfer"

Topology (vertex count, face count, UVs, or pin-group membership)
changed after **Transfer**. Click **Transfer** again. Fetching across
a mismatch would bind a PC2 to a mesh of a different vertex count.

### "Objects missing UUID" / "Stale UUID references"

Usually after loading an old file or renaming objects. Run **UUID
Migration** from the Tools panel.

### `ValueError: Object '...' is assigned to both '...' and '...' groups`

The same object (identified by its UUID) appears in more than one active
group. Remove it from all but one group via **Add Selected Objects** / the
group's object list.

### Run button is disabled

A bake is still running. Let it finish or click **Abort**.

### Status: "Connection lost" during a run

Network hiccup, server crash (OOM, driver fault), or host reboot.
Reconnect; if frames were produced, **Fetch All Animation** pulls them. The
remote `server.log` has the cause.

### "Server error" in the status line

The server returned a JSON `error`, usually from a solver exception.
Read `server.log` for the traceback. Common causes: out of disk on the
remote, permission denied on the project directory, CUDA OOM.

## Fetch and playback

### "Missing frames" warning

The remote has frames the local Blender does not. Click **Fetch All Animation**.

### Render with unfetched frames

A popup warns "N frames unfetched". Fetch first, then render. The
popup fires once per render and does not block.

### "Data path: ... does not exist"

The `data/<session>/` folder referenced by a `MESH_CACHE` modifier is
missing (deleted, renamed, or not copied across machines). Restore
from backup, or click **Migrate data/...** to rebind.

## Bake

### "Remove all shape keys except Basis before baking"

Baking writes fcurves that conflict with shape keys. In Object Data
Properties, delete every shape key except `Basis` on the listed
objects.

## MCP server

### Port already in use

The add-on silently walks `9633`-`9642` and binds the first free port.
Check the actual port in the MCP panel if an external client points at
the base.

### "Could not find available port in range 9633-9642"

All ten slots are taken. Kill the holder (`lsof -i :9633-9642` on
macOS/Linux, `netstat -ano | findstr 963` on Windows), or change the
base port.

### "Failed to start MCP server"

The server thread raised during startup (socket permission, port
collision, import error). The Blender system console prints the
exception as `MCP Server error: ...`.

### `run_python_script` returns `"success": false`

The snippet you sent raised. Read the `error` field; the full
traceback is in the Blender system console.

### `capture_viewport_image`: "No 3D viewport found"

The current Blender screen layout has no `VIEW_3D` area. Switch to
`Layout`, `Modeling`, or `Sculpting`, then retry.

## Debug CLI

### "MCP server not reachable on localhost:9633"

The CLI got no answer in 2 s. Start the MCP server from the Blender
panel or via `python blender_addon/debug/main.py start-mcp`. If it is
running on a fallback port, pass `--mcp-port <port>`.

### "Debug/reload port (TCP 8765): unreachable"

The add-on is not loaded, or the reload server never bound. Enable the
add-on in `Edit > Preferences > Add-ons`; if Blender is running and the
port stays down, restart Blender.

## Hot reload

### Change didn't show up after reload

Almost always a `PropertyGroup` schema change; plain reload swaps code
but cannot rebind Blender's RNA. Run

```
python blender_addon/debug/main.py full-reload
```

If even that fails, restart Blender. This is a Blender RNA limitation.

### "Reload timed out"

Plain reload times out at 30 s, full reload at 60 s. Something in
top-level module code, `register`, or `unregister` is blocking the
main thread (network call, large mesh op, missing
`bpy.app.timers.unregister`). The Blender system console shows what
was still running.
