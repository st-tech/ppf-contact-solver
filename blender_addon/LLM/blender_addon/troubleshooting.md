# Troubleshooting

This file condenses `docs/blender_addon/troubleshooting.md` into a self-contained lookup of common errors, grouped by subject. Each entry is a short `You see / Why / Fix` triple. Status strings in quotes are exact copies of what the panel shows; backticks mark log lines from the Blender system console or the remote `server.log` / `progress.log` files written by `ppf-cts-server`.

## Installation and Dependencies

### "No module named paramiko" (or docker)

- You see: error popup or status mentioning `paramiko` or `docker` when you first pick an SSH or Docker backend.
- Why: the add-on vendors both packages but only populates the vendored copies on demand.
- Fix: click **Install Paramiko** (or **Install Docker**) on the main panel; it runs `pip install` in a background thread.

### "No module named cbor2"

- You see: a popup or raised `ModuleNotFoundError` / `Cbor2NotInstalledError` mentioning `cbor2` at Transfer time, or a `cbor2` install prompt on the main panel before connecting. cbor2 encodes every Transfer regardless of backend.
- Why: cbor2 ships as a per-ABI wheel (`cbor2==6.0.1`) in `blender_manifest.toml` and is installed into Blender's extension site-packages when the packaged extension is installed through Blender's installer. It is not present if the add-on was copied in manually, or a 5.0->5.1 settings migration carried it over without reinstalling its wheels.
- Fix: reinstall the extension through Blender (**Get Extensions** or **Install from Disk**) so its bundled wheel is installed, or click **Install cbor2** on the main panel. The button pip-installs the matching `cbor2==6.0.1` wheel for the running interpreter into Blender's user `scripts/addons/modules/` dir (on `sys.path`); same paths and last-resort `pip install --target` recipe as the next entry.

### Install operator hangs or fails

- You see: install modal never returns, or closes with `Failed to install paramiko` / `Failed to install docker`.
- Why: `pip` subprocess returned non-zero (network error, no compiler for a C extension, write-denied target), or it exceeded the 120 s internal timeout.
- Fix: check Blender system console for full `pip` stderr. Verify the install target is writable; the target is Blender's user `scripts/addons/modules/` dir (macOS: `~/Library/Application Support/Blender/<ver>/scripts/addons/modules/`, Linux: `~/.config/blender/<ver>/scripts/addons/modules/`, Windows: `%APPDATA%\Blender Foundation\Blender\<ver>\scripts\addons\modules\`). As a last resort run `python -m pip install --target <that path> paramiko docker`.

## Connection: SSH / SSH Command

### SSH authentication failed

- You see: a paramiko auth failure at connect time.
- Why: the key the add-on uses does not match a key on the remote.
- Fix: verify that the key path points at the actual private key (not `.pub`); the key has no passphrase, or your agent has it loaded; `~/.ssh/authorized_keys` on the remote contains the matching public key.

### "Failed to parse command. Ensure it includes host."

- You see: this after clicking **Connect** in SSH Command mode.
- Why: the pasted string has no recognizable host token, or it carries an option the parser refuses (an unknown one, or one missing its argument). The parser reads the destination plus `-p`, `-i`, `-l`, `-J`, and `-o Port=/IdentityFile=/User=/ProxyJump=`; other ssh options are accepted and ignored.
- Fix: use `ssh -p <port> -i <key> user@host`, adding `-J user@bastion` when the host is behind a jump host. The reported message names the specific problem. For `ProxyCommand` or any other setup the parser ignores, switch to SSH Custom mode and let `~/.ssh/config` fill in the blanks.

### `~/.ssh/config` options not honored

- You see: a host that works from your terminal fails from the add-on, or connects to the wrong place.
- Why: the add-on's `ssh_config` parser only understands `Host`, `HostName`, `Port`, `User`, `IdentityFile`, `ProxyJump`, `Include`. Everything else (`ProxyCommand`, `Match`, `StrictHostKeyChecking`, `IdentitiesOnly`, certificates) is ignored. See connections.md for the supported subset.
- Fix: a bastion needs nothing done by hand, `ProxyJump` is honored (the panel's **Proxy Jump** field and `ssh -J` set it too). For the rest, bring up the tunnel from a terminal (e.g. `ssh -L 2222:gpu01.internal:22 bastion`) and point the add-on at `localhost:2222`.

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

- You see: e.g. `Docker port 9090 is not exposed on container 'ppf-contact-solver'. Please expose the port with '-p 9090:9090' when starting the container.`
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
- Fix: set **Solver Path** to the solver root (the directory that contains the `ppf-cts-server.exe` binary, either at the root for bundle layouts or under `target\release\` for dev builds).

### ppf-cts-server.exe not found under the solver root

- You see: connect fails immediately with this path in the message.
- Why: the root points one level too high or too low.
- Fix: verify the directory contains `ppf-cts-server.exe` (under `target\release\` for dev builds, at the root for bundle layouts).

### "Embedded Python not found"

- You see: connect fails after `ppf-cts-server.exe` is located.
- Why: neither the dev layout (`build-win-native\python\python.exe`) nor the bundle layout (`python\python.exe`) resolved.
- Fix: rebuild the dev tree, or download and unpack the shipped bundle zip next to `ppf-cts-server.exe`.

### CUDA DLL load errors

- You see: `server.log` on the Windows host reports a missing CUDA runtime DLL.
- Why: on the bundle layout the solver expects CUDA on the system `PATH`; only the dev layout ships its own CUDA.
- Fix: install a matching CUDA runtime, or switch to the developer build.

## Connection: macOS Native

### "Solver path is not set"

- You see: this error at connect time on the macOS Native backend.
- Why: the path field is blank.
- Fix: set **Solver Path** to the solver root, the directory holding `target/release/ppf-cts-server`. Both layouts keep it there: a repo checkout you built, and the extracted bundle.

### ppf-cts-server not found under the solver root

- You see: connect fails immediately with this path in the message.
- Why: the root points somewhere that holds no solver. Picking a subdirectory is fine, the resolver walks up to the real root, but picking an unrelated folder is not.
- Fix: verify the directory contains `target/release/ppf-cts-server`. There is no `bin/` fallback for the server on macOS; `bin/` holds the Metal backend dylib and its shader libraries.

### "This bundle ships no Python"

- You see: `start.sh` refuses before launching JupyterLab.
- Why: unlike the Windows bundle, the macOS bundle carries no interpreter, so it needs one on the machine.
- Fix: point `PPF_CTS_VENV` at an environment with the frontend dependencies and Python 3.10 or newer, either by editing `config.sh` beside the script or by exporting it. macOS ships 3.9, which the frontend does not parse.

### The app is blocked on first launch

- You see: macOS refuses to open the binaries after you downloaded the bundle in a browser.
- Why: the download attached `com.apple.quarantine`, and the bundle is signed but not notarized.
- Fix: nothing, normally. The bundle's launcher clears that mark on its own folder the first time you run it and prints one line saying so, and connecting the add-on to a downloaded bundle clears it the same way before the server is spawned. Fetching with `curl`, `scp` or `git` sets no mark and prints nothing.
- When it is not automatic: a bundle owned by another user, or on a read-only volume, cannot be cleared. The launcher says so and the add-on leaves the spawn to fail with its own message. Move the folder somewhere you own, or clear it by hand with `xattr -s -d -r com.apple.quarantine <extracted bundle>`. Use `-s`: without it `xattr` follows symbolic links, which leaves their own marks in place and exits non-zero on any link pointing nowhere.

### "does not support the Metal GPU family this solver build requires"

- You see: this at `Run`, naming the GPU.
- Why: the solver requires the Apple7 family (M1 / A14 or newer). An Intel Mac, or a virtualized Mac whose GPU appears as `Apple Paravirtual device`, does not report it.
- Fix: run on Apple Silicon hardware. There is no fallback and no override, because a device that cannot support the family cannot be trusted to compute the right answer.

## Connection: Linux Native

### "Solver path is not set"

- You see: this error at connect time on the Linux Native backend.
- Why: the path field is blank.
- Fix: set **Solver Path** to the solver root. A repo checkout you built holds the server at `target/release/ppf-cts-server`; an unpacked Linux distribution holds one directory per backend it ships, `target/cuda/release`, `target/rocm/release` and `target/cpu/release`, each beside a `.ppf-backend` marker naming the backend it carries.

### "ppf-cts-server not found under ..."

- You see: connect fails immediately, and the message names the directory it examined and the layouts it accepts.
- Why: the folder holds no server in any accepted layout. The usual cause is picking the folder the distribution was extracted into rather than the distribution root: the resolver walks up from a subdirectory to a root, never down into one.
- Fix: point **Solver Path** at the folder that has `target/release`, `target/cuda/release`, `target/rocm/release` or `target/cpu/release` under it. Building from source, run `cargo build --release -p ppf-cts-server` first. The distribution's `bin/` holds the backend library and ffmpeg and never a server, so it is not a root.

### "... holds the CPU build of the solver and no GPU build"

- You see: connect refuses with this, naming the directory, instead of the not-found message. The two device names swap when the folder holds only the GPU build and **Compute Device** says CPU.
- Why: the folder is right and **Compute Device** is not. Every backend links the same executable name, so the `.ppf-backend` marker in each build directory is what says which device a directory answers to.
- Fix: set **Compute Device** to the device the message names, or point **Solver Path** at a folder that has the build you want. The add-on refuses rather than running the other build: a CPU run is substantially slower than a GPU one, and nothing later in the run would report the substitution.

### "... holds these GPU builds: cuda, and no rocm build"

- You see: connect refuses with this after naming an accelerator in **GPU Backend**.
- Why: **Compute Device** is GPU and **GPU Backend** names a backend this root does not hold. A named choice resolves to that build or to nothing; it never falls through to another accelerator.
- Fix: set **GPU Backend** to one the message lists or to Automatic, or point **Solver Path** at a folder that has the backend you asked for.

### "No GPU build in this folder has a usable device"

- You see: this at **Start Server on Remote**, listing what each GPU build reported.
- Why: the root holds one or more GPU builds, and none of their solvers found a device it can run on. Each is asked with `ppf-contact-solver --probe` when the server is spawned, which is why connect succeeded and the launch is what refused.
- Fix: install or repair the driver or runtime the reasons name, or set **Compute Device** to CPU to run without a GPU. An explicit GPU choice never falls back on its own.

## Connection Profiles

### Profile dropdown is empty after Open

- You see: **Open** succeeds but the profile picker has no entries.
- Why: the TOML file is malformed, unreadable, or genuinely empty; `load_profiles` catches all exceptions and returns an empty dict.
- Fix: run the file through any TOML validator. Common causes: unclosed quote, unescaped backslashes in a Windows path (use `\\` or forward slashes), missing `type` key.

### Profile loads but fields are blank

- You see: after picking a profile, connection fields do not populate.
- Why: the `type` value does not match one of `SSH`, `SSH Command`, `Docker`, `Docker over SSH`, `Docker over SSH Command`, `Windows Native`, `macOS Native`, `Linux Native`, so the loader rejects the entry.
- Fix: fix the `type` value; the set is exact and case-sensitive.
- Exception: a profile written with the retired `Local` type still loads. It is applied as this platform's native type (`Linux Native`, `macOS Native` or `Windows Native`), and its `local_path` key lands on that type's path field. A profile carrying both `local_path` and the current key keeps the current one.

Note: **Save** overwrites the currently selected entry and rewrites the whole file. Comments and original formatting are lost on round-trip; keep a backup if comments matter.

## Server Startup

### Status stuck on "Waiting for server start..."

- You see: status never advances past this string.
- Why: you connected but have not clicked **Start Server**, or `ppf-cts-server` exited before finishing boot.
- Fix: click **Start Server**. If it then times out, see next entry.

### "Server startup timed out."

- You see: this plus the last 20 lines of `server.log` pasted into the panel.
- Why: 16 seconds elapsed without the server writing a ready-marker, and no `ERROR` or `FAILED` line appeared.
- Fix: tail `progress.log` and `server.log` on the remote. Usual causes: missing CUDA driver, port collision on the bound port, corrupted checkout, or a missing build worker Python at `$HOME/.local/share/ppf-cts/venv`.

### "Server startup failed" with a log line

- You see: add-on aborts the wait early and shows one log line.
- Why: `progress.log` emitted a line containing `ERROR` or `FAILED` during startup.
- Fix: open full `progress.log` on the remote; the error is usually Python-level (missing module, failed import) and visible higher up.

### "Failed to launch server"

- You see: launch never reaches the wait phase.
- Why: the generated shell script did not start (permission denied, read-only working directory, missing or unbuilt `ppf-cts-server` binary).
- Fix: check the remote path is writable (the script writes `server.log` and a PID file there) and confirm the solver the selected **Compute Device** resolves to (`target/release/ppf-cts-server`, or that backend's own `target/<backend>/release/ppf-cts-server`) exists and is executable on the remote.

### Status: "Protocol version mismatch"

- You see: this exact status.
- Why: the server reports a wire version other than the one the add-on carries (`blender_addon/protocol_version.toml`, the single source both halves read).
- Fix: rebuild the solver from a revision that matches the add-on, or update the add-on.

### "ppf-cts-server not found under ... on the solver host, in any layout"

- You see: this message the instant **Connect** returns on an SSH or Docker connection. A Transfer started in the same breath as the connection is canceled by it, which leaves the solver at `NO_BUILD` with no build pending.
- Why: the post-connect check asks the solver host which directories under the configured root hold a server (`target/release` and the per-backend `target/<backend>/release` directories) and it reported none. It asks only whether the root holds a server at all, never whether it holds the build **Compute Device** names, so a device that host cannot serve is refused later, at Start Server, where it can still be changed.
- Fix: fix **Remote Path** or **Docker Path** to point at a directory on that host which holds a built solver, and build one there with `cargo build --release -p ppf-cts-server` if there is none. On the three native connection types the same check runs against this machine and the message names **Solver Path** instead.

### Port already in use

- You see: **Start Server on Remote** fails, and the log shows the port is taken.
- Why: a stale `ppf-cts-server` process from an earlier session, or another solver, is still bound to it.
- Fix: click **Stop Server** first, or end the process holding the port (`ss -tlnp | grep <port>` names it on Linux). On a Docker connection the **Docker Port** field selects another port; the MCP connection tools take the port as an argument.

## Compute Device and GPU Backend

**Compute Device** (GPU / CPU) and **GPU Backend** (Automatic / CUDA / ROCm) choose which build of the solver the next **Start Server on Remote** launches, and they apply to every connection type. On the three native types the add-on reads the build directories off this machine while you set the path; on SSH and Docker it asks the solver host once per connection for that listing, which is why those two rows are drawn only after **Connect**. The launch exports `CARGO_TARGET_DIR` for the build it resolved, so the build worker loads the matching cdylib out of the same directory.

### "The solver host holds the CPU build of the solver under ... and no GPU build"

- You see: this at **Start Server on Remote** on an SSH or Docker connection. The two device names swap when the host holds only the GPU build and **Compute Device** says CPU.
- Why: the listing that host returned holds no build for the selected device under **Remote Path** / **Docker Path**. The panel draws the same finding above the button, as "The solver host has no GPU build here; it has CPU".
- Fix: set **Compute Device** to the device the message names, or point the path at a directory on that host which holds the build you want. The `.ppf-backend` marker in a build directory, not the directory's name, is what says which backend it holds.

### "The solver host holds these GPU builds under ...: cuda, and no rocm build"

- You see: this at **Start Server on Remote** after naming an accelerator in **GPU Backend**.
- Why: **Compute Device** is GPU and **GPU Backend** names a backend that host does not have under this root. A named choice resolves to that build or to nothing, never to another accelerator.
- Fix: set **GPU Backend** to one the message lists, or to Automatic, which takes the first GPU build present in CUDA, ROCm order. On a remote connection Automatic does not probe the builds, since probing means running a solver on the far machine, so name the backend where that host holds two GPU builds and you want the second.

### "Could not list the solver builds on the solver host: ..."

- You see: this line with an error icon under the **Compute Device** row, and the same text as a connection error, which cancels a Transfer started in the same breath as the connection.
- Why: the listing command the add-on runs once at connect time failed, timed out (it is bounded at 10 seconds), or the connection names no directory on the host. The device rows stay drawn, because which builds that host holds is unknown rather than known to be none.
- Fix: read the reason in the line, which is the backend's own error, then fix the path or the host and press the refresh button beside the GPU dropdown. That one button re-asks for both the GPU list and the build listing. **Start Server on Remote** refuses with this same reason until the listing succeeds.

### Notebook prints "No usable GPU was found on this machine ..., so the CPU backend is selected"

- You see: one line in a notebook or JupyterLab session on the solver host, naming what each GPU backend reported, followed by a run that is substantially slower than expected.
- Why: no backend was named, so the frontend asked every GPU build present through its own `ppf-contact-solver --probe`, none reported a usable device, and a CPU build was there to answer with. This is the automatic rule's only fallback, and it prints the line so the substitution is never silent. The line is printed once per process.
- Fix: nothing, when the CPU backend is what you want. Otherwise repair the driver or runtime the reasons name, or pin the backend with `frontend.set_backend("cuda")` (also `"rocm"`, `"metal"`, `"cpu"`). An explicit choice never falls back: it runs that backend or raises, on a machine with a GPU and on one without.

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

### `ValueError: Object '...' has N isolated vertex(es)` (substring "isolated vert")

- You see: transfer aborts with this ValueError naming the object, the count, and up to eight vertex indices.
- Why: a Static collider mesh has vertices that belong to no triangle (no face). The solver averages each collider vertex's contact parameters over its incident faces and aborts when a vertex has none. Common in imported models with stray points.
- Fix: click **Remove Isolated Vertices** under the error (operator `ssh.remove_isolated_vertices`). When the error names a Static collider, the MCP pair `detect_isolated_static_vertices` (preview) and `remove_isolated_static_vertices` (fix) targets exactly that case. Across a mixed selection, scan in **Sidebar -> Utility Tools -> Mesh Cleaning** and press the **Remove Loose Vertices** button the report offers (MCP `remove_loose_vertices`, Python `solver.remove_loose_vertices`). It acts on every selected mesh at once, keeps pinned vertices, takes the loose edges with them, skips Sand particle meshes, and clears the caches the vertex-count change invalidates. Transfer again either way.

### Mesh will not build (scan it before Transfer)

- You see: Transfer or the remote build aborts on geometry, or a run stops on the solver's SPD guard without naming any geometry.
- Why: the encoder and the solver reject several classes of input geometry, and most of them are invisible in the viewport.
- Fix: select the meshes and press **Scan Selected Meshes** in **Sidebar -> Utility Tools -> Mesh Cleaning** (MCP `scan_meshes`, Python `solver.scan_meshes`). The report lists each class with a one-click repair beside it. This is cheaper than diagnosing a rejected build, so run it on any generated or imported mesh before the first Transfer.

The lines the scan reports as errors, and the repair for each:

| Scan line                                | Repair button          | MCP / Python name              | Vertex count |
| ---------------------------------------- | ---------------------- | ------------------------------ | ------------ |
| `Linked Duplicate of <name>`             | none (see below)       | none                           | unchanged    |
| `N near-coincident vertex pair(s)`       | Merge by Distance      | `merge_by_distance`            | **changes**  |
| `N isolated vertex(es), in no face`      | Remove Loose Vertices  | `remove_loose_vertices`        | **changes**  |
| `N hanging seam vertex(es)`              | Remove Loose Vertices  | `remove_loose_vertices`        | **changes**  |
| `N duplicate face(s)`                    | Delete Duplicate Faces | `delete_duplicate_faces`       | unchanged    |
| `N degenerate (zero-area) face(s)`       | Dissolve Degenerate    | `dissolve_degenerate_faces`    | **changes**  |
| `N inconsistently wound edge(s)`         | Recalculate Outside    | `recalculate_normals_outside`  | unchanged    |

A linked duplicate has no repair button because it is not a defect in the mesh: two objects share one mesh datablock, which Transfer refuses because it routes per object. Fix it with **Object > Relations > Make Single User > Object & Data**. The scan itself, and every repair, treats a shared datablock once rather than once per user.

The three remaining lines are **notes, not errors**, and are normal on cloth:

- `N boundary edge(s), surface is open`: fine for a Shell; a Solid needs a closed surface for tetrahedralization. Closing a boundary is modeling, so nothing repairs it automatically.
- `N non-manifold edge(s)`: same, reported for the same reason.
- `N face(s) with more than 3 corners`: quads and N-gons are accepted, and Transfer triangulates them itself. Press **Triangulate** (`triangulate_for_solver`) only when the viewport's own choice of diagonal drifts from the simulated one, or **Symmetric Triangulate** (`symmetric_triangulate`) when the mesh's mirror symmetry has to survive.

The three repairs marked **changes** invalidate the object's PC2 display cache and any captured deformation, and can shift which vertices a pin vertex group holds. The panel names exactly what is affected before it runs and asks for confirmation; the MCP tools require `acknowledge=true` and clear those caches by default (`clear_stale_caches`, which matches the pre-ticked checkbox in the panel's dialog). **Symmetric Triangulate** adds one vertex per face, so it has the same consequence, but it neither asks for confirmation nor clears anything: re-run Transfer for the display cache and Capture Deformation for the captures.

Run Transfer again after any repair, including the ones that leave the count alone: the encoder captures winding and topology at Transfer time.

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

- You see: button is grayed out.
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

### "Newton solve made no progress (over-constrained configuration)"

- You see: the run stops mid-simulation and the solver log ends with `### newton stalled: no acceptable step after N iterations`, followed by `an over-constrained configuration cannot be advanced`.
- Why: something prescribed is being driven into geometry that cannot yield, so no step exists that avoids penetration and the solver refuses to fake one. Two shapes are common: a pin whose path runs into a collider, and cloth caught in a closing crevice of a Static collider (a character's armpit shutting, a hand pressing into a thigh). The reported frame is where it gave up; the geometry usually starts closing several frames earlier.
- Fix: for a pin, re-author its path or make it a soft pull pin. For a collider crevice, turn on **Apply Soft Constraints** on the Static group so the collider is held by springs and gives way where the cloth pushes back, and lower its **Stiffness** if the cloth is still trapped. Widening the crevice or lowering the group's **Contact Gap** also buys room, but neither is reliable once the collider closes all the way.

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

- You see: when the configured MCP port is busy, the add-on silently walks the next nine (`9634` through `9642`) and binds the first free one.
- Why: another MCP server, another Blender, or a stale process is bound to the base port.
- Fix: nothing required, but check the actual bound port in the MCP panel if an external client is pointing at the base. Confirm with `python blender_addon/debug/main.py --mcp-port <port> status`.

### "MCP Server: Could not find available port in range 9633-9642"

- You see: this error when starting the MCP server.
- Why: all ten slots in the search range are taken.
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
- Why: almost always a PropertyGroup schema change. Plain reload swaps code but cannot rebind Blender's RNA. New PropertyGroup fields, removed fields, or changed property types cannot be picked up by either reload command.
- Fix: for added/removed/retyped properties, restart Blender. For other schema changes, try `python blender_addon/debug/main.py full-reload`; if it still does not take, restart Blender. This is a limitation of Blender's RNA system, not the reload server.

### "Reload timed out"

- You see: `reload` or `full-reload` exits with this error after 45 s (plain) or 70 s (full).
- Why: something in the module's top-level code, `register`, or `unregister` is blocking the main thread (network call, large mesh operation, missing `bpy.app.timers.unregister`).
- Fix: move expensive work to background threads; check the Blender system console for whatever was still printing when the timer fired.

### "Reload didn't restart the MCP server"

- You see: after a reload, MCP panel shows the server as stopped.
- Why: on reload the add-on remembers which companion servers (MCP, debug reload) were running and restarts them; if the add-on crashed mid-reload that hand-off is lost.
- Fix: restart the MCP server from the panel or via `python blender_addon/debug/main.py start-mcp`.
