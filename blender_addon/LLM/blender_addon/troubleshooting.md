# Troubleshooting

This file condenses `docs/blender_addon/troubleshooting.md` into a self-contained lookup of common errors, grouped by subject. Each entry is a short `You see / Why / Fix` triple. Status strings in quotes are exact copies of what the panel shows; backticks mark log lines from the Blender system console or remote `server.log` / `progress.log`.

## Installation and Dependencies

### "No module named paramiko" (or docker)

- You see: error popup or status mentioning `paramiko` or `docker` when you first pick an SSH or Docker backend.
- Why: the add-on vendors both packages but only populates the vendored copies on demand.
- Fix: click **Install Paramiko** (or **Install Docker**) on the main panel; it runs `pip install` in a background thread.

### Install operator hangs or fails

- You see: install modal never returns, or closes with `Failed to install paramiko` / `Failed to install docker`.
- Why: `pip` subprocess returned non-zero (network error, no compiler for a C extension, write-denied target), or it exceeded the 120 s internal timeout.
- Fix: check Blender system console for full `pip` stderr. Verify the add-on directory is writable; as a last resort run `python -m pip install --target <addon>/lib paramiko docker`.

## Connection: Local

### "Remote path not found (.../server.py)"

- You see: this in the status line after **Connect**.
- Why: the path does not contain `server.py`.
- Fix: point **Local Path** at the checkout root, not a `build/` or `src/` subdirectory.

### Port already in use

- You see: **Start Server** fails, log shows the port is taken.
- Why: a stale `server.py` from an earlier session, or another solver, is still bound.
- Fix: click **Stop Server** first, or change **Server Port**. On the host, `ss -tlnp | grep <port>` names the process.

## Connection: SSH / SSH Command

### SSH authentication failed

- You see: a paramiko auth failure at connect time.
- Why: the key the add-on uses does not match a key on the remote.
- Fix: verify that the key path points at the actual private key (not `.pub`); the key has no passphrase, or your agent has it loaded; `~/.ssh/authorized_keys` on the remote contains the matching public key.

### "Failed to parse command. Ensure it includes host."

- You see: this after clicking **Connect** in SSH Command mode.
- Why: the pasted string has no recognizable host token. The bundled parser only understands `-p`, `-i`, `user@host`, and the first bare host token.
- Fix: use `ssh -p <port> -i <key> user@host`. For any `-o`, `-J`, or `ProxyCommand` setup, switch to SSH Custom mode and let `~/.ssh/config` fill in the blanks.

### `~/.ssh/config` options not honored

- You see: a host that works from your terminal fails from the add-on, or connects to the wrong place.
- Why: the add-on's `ssh_config` parser only understands `Host`, `HostName`, `Port`, `User`, `IdentityFile`, `Include`. Everything else (`ProxyJump`, `ProxyCommand`, `Match`, `StrictHostKeyChecking`, `IdentitiesOnly`, certificates) is ignored. See connections.md for the supported subset.
- Fix: bring up the tunnel or bastion from a terminal (e.g. `ssh -L 2222:gpu01.internal:22 bastion`) and point the add-on at `localhost:2222`.

Note: the add-on accepts unknown host keys silently (paramiko `AutoAddPolicy`). Host-key verification is not surfaced in the UI; see security.md for the trust-on-first-use caveat.

## Connection: Docker and Docker over SSH

### "Container '...' does not exist."

- You see: this exact string after clicking **Connect**.
- Why: the container name is validated at connect time and remote `docker ps -a` returns no match.
- Fix: run `docker ps -a` on the daemon host; correct the **Container** field, or start/create the container.

### "Error starting container '...'"

- You see: connect fails right after the existence check.
- Why: the container exists but is stopped, and `docker start` returned non-zero. Common causes: port conflicts on publish time, remote user not in the `docker` group.
- Fix: on the daemon host, run `docker start <name>` manually and read the error, or `docker logs <name>` for the previous failure.

### "Docker port N is not exposed on container '...'"

- You see: e.g. `Docker port 9090 is not exposed on container 'ppf-dev'. Please expose the port with '-p 9090:9090' when starting the container.`
- Why: before **Start Server**, the add-on runs `docker port` against the container. An empty result aborts the operator.
- Fix: re-run `docker run -p <port>:<port>` (or edit `compose.yaml` and recreate the container). The add-on cannot publish ports on an existing container.

### Docker daemon / permission errors

- You see: `Error: <docker stderr>` from a `docker ps` or `docker inspect` call.
- Why: the daemon is not running, or the user is not in the `docker` group.
- Fix: confirm `docker version` works as the same user. On Linux, `sudo usermod -aG docker $USER` and log out / in.

## Connection: Windows Native

### "Solver path is not set"

- You see: this error at connect time on the Windows Native backend.
- Why: the path field is blank.
- Fix: set **Win Native Path** to the solver root (the directory that contains `server.py`, not its parent).

### server.py not found under the solver root

- You see: connect fails immediately with this path in the message.
- Why: the root points one level too high or too low.
- Fix: verify the directory contains `server.py`. On dev host `win-build` that is `C:\ppf-contact-solver`.

### "Embedded Python not found"

- You see: connect fails after `server.py` is located.
- Why: neither the dev layout (`build-win-native\python\python.exe`) nor the bundle layout (`python\python.exe`) resolved.
- Fix: rebuild the dev tree, or download and unpack the shipped bundle zip next to `server.py`.

### CUDA DLL load errors

- You see: `server.log` on the Windows host reports a missing CUDA runtime DLL.
- Why: on the bundle layout the solver expects CUDA on the system `PATH`; only the dev layout ships its own CUDA.
- Fix: install a matching CUDA runtime, or switch to the developer build.

## Connection Profiles

### Profile dropdown is empty after Open

- You see: **Open** succeeds but the profile picker has no entries.
- Why: the TOML file is malformed, unreadable, or genuinely empty; `load_profiles` catches all exceptions and returns an empty dict.
- Fix: run the file through any TOML validator. Common causes: unclosed quote, unescaped backslashes in a Windows path (use `\\` or forward slashes), missing `type` key.

### Profile loads but fields are blank

- You see: after picking a profile, connection fields do not populate.
- Why: the `type` value does not match one of `Local`, `SSH`, `SSH Command`, `Docker`, `Docker over SSH`, `Docker over SSH Command`, `Windows Native`, so the loader rejects the entry.
- Fix: fix the `type` value; the set is exact and case-sensitive.

Note: **Save** overwrites the currently selected entry and rewrites the whole file. Comments and original formatting are lost on round-trip; keep a backup if comments matter.

## Server Startup

### Status stuck on "Waiting for server start..."

- You see: status never advances past this string.
- Why: you connected but have not clicked **Start Server**, or `server.py` exited before finishing boot.
- Fix: click **Start Server**. If it then times out, see next entry.

### "Server startup timed out."

- You see: this plus the last 20 lines of `server.log` pasted into the panel.
- Why: 16 seconds elapsed without the server writing a ready-marker, and no `ERROR` or `FAILED` line appeared.
- Fix: tail `progress.log` and `server.log` on the remote. Usual causes: missing venv at `$HOME/.local/share/ppf-cts/venv`, missing CUDA driver, port collision on the bound port, corrupted checkout.

### "Server startup failed" with a log line

- You see: add-on aborts the wait early and shows one log line.
- Why: `progress.log` emitted a line containing `ERROR` or `FAILED` during startup.
- Fix: open full `progress.log` on the remote; the error is usually Python-level (missing module, failed import) and visible higher up.

### "Failed to launch server"

- You see: launch never reaches the wait phase.
- Why: the generated shell script did not start (permission denied, read-only working directory, missing `python3`).
- Fix: check the remote path is writable (the script writes `server.log` and a PID file there) and `python3` resolves via the venv or on `$PATH`.

### Status: "Protocol version mismatch"

- You see: this exact status.
- Why: the server reports a wire version other than `0.02`.
- Fix: rebuild the solver from a revision that matches the add-on, or update the add-on.

### "Remote path not found (.../server.py)."

- You see: this message the instant **Connect** returns.
- Why: post-connect path check found no `server.py` at the configured directory.
- Fix: fix **Remote Path** / **Local Path** / **Docker Path** / **Win Native Path** to point at the directory that actually contains `server.py`.

## Scene Setup: Object Groups

### "Maximum number of groups reached"

- You see: this error when adding a new group.
- Why: the add-on caps active groups at 32 (`N_MAX_GROUPS`).
- Fix: delete an unused group, or merge two groups that share a material.

### "Object '...' is already in another group"

- You see: this error when adding an object tracked elsewhere.
- Why: each object's UUID can live in exactly one active group; the encoder uses the UUID as the routing key.
- Fix: remove the object from the other group first, or assign it to the intended group only.

### "Object '...' is library-linked and cannot be assigned"

- You see: this error when adding a linked object.
- Why: library-linked data blocks are read-only; the add-on cannot attach its custom properties (UUID, per-object overrides).
- Fix: make the object local (`Object > Make Local...`) before assigning.

## Scene Setup: Pins and Constraints

### "More than one EMBEDDED_MOVE operation is not supported"

- You see: encoder rejects a pin with two move operations stacked.
- Why: each pin may track at most one move animation; two would not compose meaningfully.
- Fix: delete the duplicate operation, keeping the first.

### "Torque cannot be mixed with Move/Spin/Scale operations"

- You see: encoder rejects the pin.
- Why: TORQUE runs a PCA over the whole pin group and is mutually exclusive with the kinematic operations.
- Fix: split intent across two pins: one TORQUE, one with the kinematic operation.

### "SPIN/SCALE with CENTROID center cannot be combined with EMBEDDED_MOVE"

- You see: message ends with "(centroid is baked at frame 1 and would drift from the moving pin)".
- Why: CENTROID mode bakes pivot at frame 1; once the pin moves the pivot drifts and rotation/scale becomes nonsense.
- Fix: switch center to `MAX_TOWARDS` or `VERTEX`, or drop the EMBEDDED_MOVE.

### "Spin/Scale center vertex not set"

- You see: message ends with "pick a vertex in Edit Mode".
- Why: center mode is `VERTEX` but no vertex was registered as the pivot.
- Fix: in Edit Mode select a single vertex and register it as the spin/scale center from the pin operation UI.

### "TORQUE requires at least 3 vertices for PCA axis (got N)"

- You see: encoder rejects a thin TORQUE pin.
- Why: PCA needs at least three non-collinear points to define an axis.
- Fix: add more vertices to the pin group.

### "TORQUE PCA produced non-finite axis (are pins collinear?)"

- You see: encoder rejects the pin even with enough vertices.
- Why: vertices are collinear or coplanar; PCA degenerates and emits NaN/inf.
- Fix: spread the pin group across three non-degenerate dimensions.

### "Maximum 8 collision windows per object"

- You see: operator refuses to add a ninth collision window.
- Why: each object caps at 8 active/inactive windows for runtime cost reasons.
- Fix: consolidate adjacent or overlapping windows.

### Pin operator errors in Edit Mode

- "No active edit object": not in Edit Mode on the right object. Enter Edit Mode on a mesh or curve assigned to the active group.
- "Name cannot be empty": pin name field is blank; type one.
- "No vertices selected": select at least one vertex (or curve control point) before creating the pin.
- "'...' is not in this group": the edited object is not a member of the active group; add it via **Add Selected Objects** first.

## Transfer and Encoding

### "Mesh hash mismatch" before Run or Fetch

- You see: warning in the panel; operators refuse to run.
- Why: the add-on records a topology fingerprint (vertex count, face count, UV channels) on transfer and rechecks before Run and Fetch. Topology has changed.
- Fix: click **Transfer** to re-upload. Fetching across a mismatch would bind a PC2 with one vertex count to a mesh with another.

### "Mesh topology changed since last transfer..."

- You see: this with a list of differing groups.
- Why: finer-grained form of the mismatch above; vertex or triangle counts, or pin-group membership, changed since transfer.
- Fix: **Transfer** again.

### `ValueError: duplicate object name across groups`

- You see: encoder aborts transfer with this ValueError.
- Why: two active groups reference mesh objects that share a name. The encoder uses the object name as routing key and cannot disambiguate duplicates.
- Fix: rename one of the two objects.

### "Objects missing UUID" / "Stale UUID references"

- You see: these strings followed by a list, ending in "Run UUID Migration first."
- Why: an object has no stored UUID (usually after loading an old file), or its stored UUID no longer matches the object's live UUID (usually after a rename round-trip).
- Fix: run **UUID Migration** from the Tools panel; it assigns fresh UUIDs and reconciles references.

### "Project name out of sync: UI='...' but active session='...'."

- You see: this error before Run / Transfer / Fetch.
- Why: **Project Name** in the UI was changed after the server session started; the remote run still carries the old name and Blender would write output into a new folder.
- Fix: disconnect and reconnect; the session picks up the current UI value.

## Simulation Runtime

### Run button is disabled

- You see: button greys out.
- Why: a bake is still running.
- Fix: let it finish, or click **Abort Bake** first.

### Status: "Connection lost" (during a run)

- You see: status flips mid-simulation; console shows `Connection lost.`
- Why: SSH session, Docker daemon, or local server dropped the socket. Causes: network hiccup, server crash (OOM, driver fault), host reboot.
- Fix: reconnect. If the solver survived and produced frames on the remote, click **Fetch Data** to pull what is there; otherwise re-transfer and re-run. Remote `server.log` usually shows why.

### "Server error" in the status line

- You see: this during a Run, Transfer, or Fetch.
- Why: the server returned a JSON response with an `error` key, typically from an internal solver exception.
- Fix: read `server.log` on the remote for the full traceback. Common causes: out of disk space, permission denied on the project directory, CUDA out of memory.

## Fetch and Playback

### "Missing frames" warning below Clear Animation

- You see: warning with an **ERROR** icon: "N frames unfetched. Press 'Fetch All Animation'."
- Why: remote produced frames that have not been fetched into Blender. Hidden while a simulation is running.
- Fix: click **Fetch Data** or **Fetch All Animation**; the toolbar shows a progress bar during download.

### Render with unfetched frames

- You see: at render time, popup "N frames unfetched; rendered animation may be incomplete."
- Why: render kicked off while the local PC2 cache is behind the remote.
- Fix: fetch first, then render. The popup fires once per render job and does not block.

### Mesh hash drift detected on fetch

- You see: fetch aborts with a hash-drift message.
- Why: same cause as the transfer-time mismatch; Blender mesh has changed since transfer, so incoming PC2 vertex count would not match the live mesh.
- Fix: re-transfer and re-run if you need the output to line up.

### "Missing animation mapping for curve '...'"

- You see: fetch raises this ValueError.
- Why: the curve is assigned to a group but the encoder produced no vertex indices for it (usually an empty curve, or a curve that was just converted and lost its points).
- Fix: open the curve in Edit Mode; confirm at least one control point; re-transfer.

### "Mesh mapping out of range for '...'"

- You see: fetch raises this ValueError.
- Why: a vertex index in the server's response exceeds the live mesh's vertex count. The mesh was changed between transfer and fetch, or two meshes swapped names.
- Fix: re-transfer and re-run.

### "Data path: ... does not exist."

- You see: panel flags a missing PC2 file.
- Why: the `data/<session>/` folder referenced by the MESH_CACHE modifier is missing (manually deleted, renamed, or not copied across machines).
- Fix: restore the folder from backup, or click **Migrate data/...** to rebind the modifier to an existing folder under a new name.

### Silent PC2 heal on playback

- You see: no error; an object's MESH_CACHE modifier was automatically reattached during playback. Console logs `[heal_mesh_caches] skipping <obj>: could not read <pc2>` once if a file is corrupt.
- Why: the add-on repairs broken modifier bindings (cleared filepath, wrong cache format, missing format tag) each pump.
- Fix: usually none; it just works. If a file is corrupted, delete it and re-fetch.

## Bake

### "Remove all shape keys except Basis before baking"

- You see: error lists conflicting objects.
- Why: baking writes fcurves that would fight with existing shape keys.
- Fix: in Object Data Properties, delete every shape key except `Basis` on the listed objects, then bake.

### Bake aborted

- You see: "Bake aborted" and the button re-enables.
- Why: you clicked **Abort Bake**, or Blender closed mid-bake.
- Fix: none expected; rerun if you still need the cache.

## MCP Server

### "Port already in use"

- You see: when the configured MCP port is busy, the add-on silently walks the next nine (`9633` through `9642`) and binds the first free one.
- Why: another MCP server, another Blender, or a stale process is bound to the base port.
- Fix: nothing required, but check the actual bound port in the MCP panel if an external client is pointing at the base. Confirm with `python blender_addon/debug/main.py --mcp-port <port> status`.

### "MCP server: could not find available port in range 9633-9642"

- You see: this error when starting the MCP server.
- Why: all ten slots are taken.
- Fix: kill the holding process (`lsof -i :9633-9642` on macOS/Linux, `netstat -ano | findstr 963` on Windows), or set a different base port in the MCP panel.

### "Failed to start MCP server"

- You see: this error from the MCP operator.
- Why: server thread raised during startup (socket permissions, unexpected port collision, Python import error).
- Fix: open Blender system console; the underlying exception prints as `MCP Server error: ...`.

### "MCP server may still be running"

- You see: this warning after a stop.
- Why: server thread did not terminate within 5 seconds (stuck in a long-running handler).
- Fix: check the console; if it never clears, restart Blender or kill the bound PID directly.

### HTTP 404: "Unknown or missing Mcp-Session-Id"

- You see: this response when an external client opens an SSE stream.
- Why: the client opened `GET /mcp` (streamable HTTP) without a session, or with a session that has been torn down.
- Fix: call `initialize` first; every subsequent request must echo the returned `Mcp-Session-Id`.

### HTTP 404 on wrong path

- You see: a POST returns 404 with no body.
- Why: the request targeted a path other than `/mcp` or `/`.
- Fix: point the client at `http://localhost:<mcp-port>/mcp`.

### `run_python_script` tool error

- You see: MCP response contains `"success": false`, `"error": "..."`, and any partial stdout.
- Why: the Python snippet you asked the tool to run raised.
- Fix: read the `error` field; fix the snippet. Full traceback is in the Blender system console.

### `capture_viewport_image`: "No 3D viewport found"

- You see: tool fails with this message.
- Why: the current Blender screen layout has no `VIEW_3D` area.
- Fix: switch to a layout that has a 3D viewport (`Layout`, `Modeling`, `Sculpting`) and retry.

## Debug CLI

### "MCP server not reachable on localhost:9633"

- You see: full message points at a host and port and suggests starting it from Blender or via `start-mcp`.
- Why: CLI opened a socket, got no answer within 2 seconds.
- Fix: start MCP server from the Blender panel, or run `python blender_addon/debug/main.py start-mcp`. If it is running on a fallback port, pass `--mcp-port <port>` to the CLI.

### "Debug/reload port (TCP 8765): unreachable"

- You see: `status` reports the reload server is down.
- Why: add-on is not loaded in Blender, or the reload server never bound its port.
- Fix: enable the add-on (`Edit > Preferences > Add-ons`); confirm Blender is running; restart Blender if the reload server refuses to rebind.

### Invalid JSON in tool arguments

- You see: `json.decoder.JSONDecodeError` from the `call` subcommand.
- Why: the `arguments` string is not valid JSON. Shell quoting is a common trap.
- Fix: wrap the blob in single quotes on Unix shells, double up on Windows: `call run_python_script '{"code": "print(1+1)"}'`.

## Hot Reload and Development

### "My change didn't show up after reload"

- You see: add-on reloads cleanly but the new field, new property, or renamed class is not visible.
- Why: almost always a PropertyGroup schema change. Plain reload swaps code but cannot rebind Blender's RNA.
- Fix: run `python blender_addon/debug/main.py full-reload`. If even that does not take, restart Blender; this is a limitation of Blender's RNA system, not the reload server.

### "Reload timed out"

- You see: `reload` or `full-reload` exits with this error after 30 s (plain) or 60 s (full).
- Why: something in the module's top-level code, `register`, or `unregister` is blocking the main thread (network call, large mesh operation, missing `bpy.app.timers.unregister`).
- Fix: move expensive work to background threads; check the Blender system console for whatever was still printing when the timer fired.

### "Reload didn't restart the MCP server"

- You see: after a reload, MCP panel shows the server as stopped.
- Why: on reload the add-on remembers which companion servers (MCP, debug reload) were running and restarts them; if the add-on crashed mid-reload that hand-off is lost.
- Fix: restart the MCP server from the panel or via `python blender_addon/debug/main.py start-mcp`.
