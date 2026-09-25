# 🩹 Troubleshooting

Problems users hit in practice, grouped by where they show up. Quoted
strings are exact panel/log text. For full tracebacks read `server.log`,
`progress.log`, or the Blender system console.

## Installation

### "No module named paramiko" or "No module named docker"

Vendored copies are populated on demand. Click **Install Paramiko to
Add-on Directory** (SSH backends) or **Install Docker-Py to Add-on
Directory** (Docker backends).

### "No module named cbor2"

The `cbor2` module is needed to encode the scene before each transfer.
It is normally installed for you when the extension is installed, so a
missing copy usually means the install did not complete. Reinstall the
extension through Blender (**Get Extensions** or **Install from Disk**),
or click **Install cbor2 to Add-on Directory** in the Backend
Communicator panel. The button installs `cbor2==6.0.1` into Blender's
`scripts/addons/modules` directory.

### Install operator hangs or fails

`pip` exited non-zero or hit the 120 s internal timeout. Read its stderr
in the Blender system console. The install target - Blender's user
`scripts/addons/modules` directory, which is both what the buttons write
to and where the add-on looks - must be writable; as a fallback, install
into it by hand with `python -m pip install --target <blender user
scripts>/addons/modules paramiko docker`.

## Connection: SSH

### SSH authentication failed

Verify **SSH Key** points at the private key (not the `.pub`), it has
no passphrase or your agent has it loaded, and the matching public key
is in `~/.ssh/authorized_keys` on the remote.

### "Failed to parse command. Ensure it includes host."

The command names no destination. The parser reads `[user@]host` plus
`-p`, `-i`, `-l`, `-J`, and `-o Port=` / `-o IdentityFile=` /
`-o User=` / `-o ProxyJump=`; other ssh options are accepted and
ignored. An unknown option, an option missing its argument, and an
unbalanced quote are reported in their own words rather than under this
message. For a `ProxyCommand` setup no mode helps: the directive is read
nowhere in the add-on. Bring the tunnel up from a terminal and point the
add-on at the forwarded local port instead, as the next entry
describes.

### `~/.ssh/config` host works in terminal but not in the add-on

The parser understands only `Host`, `HostName`, `Port`, `User`,
`IdentityFile`, `ProxyJump`, and `Include`. `ProxyCommand`, `Match`,
and certificates are ignored. A bastion needs nothing done by hand:
`ProxyJump` is honored, and the **Proxy Jump** field and `ssh -J` set
the same thing. For the directives that are ignored, bring the tunnel up
from a terminal (`ssh -L 2222:gpu01.example.com:22
bastion.example.com`) and point the add-on at `localhost:2222`. See
{ref}`Supported ssh_config options <supported-ssh-config-options>`.

### "Jump host ... failed"

The hop named in the message refused the connection or was unreachable;
the hops opened before it were closed. Test the same chain from a
terminal (`ssh -J <jump> <host>`), which uses the same hops. A message
naming a loop instead means two `ProxyJump` entries point at each other,
and the trail it followed is printed with it.

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

## Connection: the solver path

### "Remote path not found (.../ppf-cts-server)"

The post-connect check looks for `<path>/target/release/ppf-cts-server`
and did not find it. Point **Remote Path** / **Container Path** at the
solver root - the directory that has `target/release/` under it, for
example `/root/ppf-contact-solver` - not at `target/release` itself.
Only the SSH and Docker types raise this message, and they raise it
naming every layout they searched: `target/release`,
`target/cuda/release`, `target/rocm/release`, and `target/cpu/release`.
The three native types each name the layouts they accept instead:
**Windows Native** reports "ppf-cts-server.exe not found under ..." for
a **Solver Path** with no binary under `target\release\`, a per-backend
`target\<backend>\release\`, `target\cpu\release\`, or `bin\`;
**macOS Native** and **Linux Native** report "ppf-cts-server not found
under ..." for one with none under the POSIX equivalents (`bin/` is not
a server directory on either).

A folder that does hold a solver, but not for the **Compute Device** or
**GPU Backend** you selected, is refused with a different message that
names what is there; see
{ref}`Choosing the build <choosing-the-build>`.

## Connection: the native types

These apply to **Windows Native**, **macOS Native**, and **Linux
Native** -- the three types whose solver is a child process on the
machine Blender runs on. See [Windows](connections/windows.md),
[macOS](connections/macos.md), and [Linux](connections/linux.md).

### "Solver path is not set" / "ppf-cts-server not found under ..."

Set **Solver Path** to the root directory that holds the server, not to
the `target/release` inside it and not to the folder you extracted the
archive into. A subfolder is resolved upward automatically, so this
message means no ancestor within six levels held a solver in any
accepted layout.

### "... holds the CPU build of the solver and no GPU build"

The folder is right and **Compute Device** is not. Set it to the build
that is actually there, or point **Solver Path** at a folder that has
the one you want. The reverse message appears for a GPU-only folder with
`CPU` selected. The add-on never runs the other build silently; see
{ref}`Choosing the build <choosing-the-build>`.

### "... holds these GPU builds: cuda, and no rocm build"

Same situation one level down: **GPU Backend** names an accelerator this
folder does not carry. Set it to one of the builds named, or to
`Automatic`.

### "No GPU build in this folder has a usable device"

Raised at **Start Server on Remote** when **GPU Backend** is
`Automatic`, the folder holds more than one GPU build, and none of them
reports a device it can run on. The message carries what each backend
said. Fix the driver, or set **Compute Device** to `CPU`.

### "A solver server is already running on port N, and its runs use ..."

A `ppf-cts-server` the add-on did not launch is on the port, running a
different build from the one **Compute Device** names. Attaching to it
would run every solve on that other build, so it is refused; the message
names both build directories and the way out. See
{ref}`Attaching to a running server <attach-mismatch>`.

### "Embedded Python not found" (Windows)

Neither the dev layout (`build-win-native\python\python.exe`) nor the
bundle layout (`python\python.exe`) resolved. Rebuild the dev tree, or
unpack the shipped bundle zip next to `ppf-cts-server.exe`.

On macOS and Linux the equivalent is silent: the add-on names
`<root>/python/bin/python3` for a distribution and
`~/.local/share/ppf-cts/venv/bin/python` for a checkout, and if neither
exists the build worker reports the missing module itself. A
distribution complaining about a module it ships (for example
`No module named 'pytetwild'`) means the wrong interpreter was picked --
check whether `PPF_CTS_BUILD_PYTHON` is set in the environment Blender
inherited, since an inherited value wins.

### CUDA DLL load errors (Windows)

`server.log` shows a missing CUDA runtime DLL. The shipped distribution
carries the runtime in `bin\`, which the launcher puts on `PATH`, so
this normally means the folder is incomplete or the NVIDIA driver is
missing or too old. Update the driver, re-extract the distribution, or
set **Compute Device** to `CPU`.

### "... entries are still marked com.apple.quarantine" (macOS)

The add-on clears the download mark from a packaged distribution before
spawning, but could not write to this folder -- which is what a folder
owned by another user or on a read-only volume looks like. Move it
somewhere you own, or clear it yourself:

```bash
xattr -s -d -r com.apple.quarantine "<root>"
```

## Connection profiles

### Profile dropdown empty after Open

The TOML is malformed, unreadable, or empty. `load_profiles` swallows
all exceptions. Run the file through any TOML validator; common causes
are unclosed quotes or unescaped backslashes in Windows paths (use
`\\` or forward slashes).

### Profile loads but fields stay blank

The `type` value does not match one of `SSH`, `SSH Command`,
`Docker`, `Docker over SSH`, `Docker over SSH Command`, `Windows
Native`, `macOS Native`, or `Linux Native`. Case matters.

:::{note}
**Save** rewrites the whole TOML; comments and original formatting are
lost on round-trip. Keep a backup if comments matter.
:::

## Server startup

### Status stuck on "Waiting for Server Start..."

You connected but did not click **Start Server on Remote**, or
`ppf-cts-server` exited before booting. Click **Start Server on
Remote**; if it then times out, see the next entry.

### "Connection timed out" / Cancel

A connect that has not completed in 60 seconds is torn down and reported
as *Connection timed out*; pressing **Cancel** under the status line
does the same thing sooner. Either way the attempt is ended rather than
abandoned, so the next **Connect** starts from a clean state. A connect
that hangs this way is usually a host that is not up, a firewall
swallowing the SSH port, or a jump host that cannot be reached.

### "Server startup timed out"

Sixteen seconds passed without a ready marker. The panel pastes the
last 20 lines of `server.log`, and **Cancel** under the status line
stops waiting and stops the server. Usual causes:

- the build worker's interpreter is missing: for the remote types that
  is the venv at `$HOME/.local/share/ppf-cts/venv`; for the native
  types it is `<root>/python/bin/python3` in a distribution or the same
  venv in a checkout
- GPU driver missing or mismatched
- the bound port is already in use (every backend takes the port from
  **Docker Port**, which the panel draws only for the Docker types)

### "Port N is in use"

Something is already bound to the port the server was told to use.
The three native types -- Windows Native, macOS Native, and Linux
Native -- raise this.
Each first probes the port: if it answers a ppf-cts-server protocol
ping, the add-on attaches to that running server instead of erroring
out (this is what lets you restart Blender without losing the
server). If the holder is not a
ppf-cts-server, the panel surfaces the error and shows a **Force
Terminate Process** button beside it, at panel level rather than inside
the collapsible Connection box, with one line under it saying what is
actually on the port. Clicking it walks the process tree and
force-kills the listener on that port. If the squatter is not yours,
stop it by hand, or move the solver off that port: the field that
carries the port for every backend (**Docker Port**) is one shared
property drawn only for the Docker connection types, so change it from
a Docker mode and switch back, or set `docker_port` on a
[connection profile](connections/profiles.md) entry.

### "Server startup failed" with a log line

`progress.log` emitted `ERROR` or `FAILED` during startup. Open the
full log on the remote; the real error (usually a missing module or
failed import) is higher up.

### "Failed to launch server"

The launch script itself never ran: permission denied, a `/tmp` that is
not writable or is mounted `noexec`, or a shell that could not execute
it. The script backgrounds the server with `nohup`, so it reports
success whatever happens after that - a missing venv or `python3`
surfaces later as a build-worker `ModuleNotFoundError`, not here.

### Status: "Protocol version mismatch"

The solver binary and the add-on were built from revisions that speak
different wire protocols. The add-on stops the server on this status, so
a restart picks up the on-disk binary; if the status comes back after
that, the two halves are genuinely out of step. Rebuild the solver from a
matching revision, or update the add-on. The add-on's log names both
sides ("server reports X, addon expects Y"), which tells you which one is
older.

The add-on and the solver each take this number from the same shipped
file, so there is nothing to configure and nothing to match up by hand: a
mismatch always means one of the two binaries is stale.

:::{admonition} Under the hood
:class: toggle

The version is single-sourced in `blender_addon/protocol_version.toml`.
The add-on reads that file when it starts, and the server binary carries
the value that was compiled into it, so an add-on and a server built from
one revision always hold the same number. That file also carries the
changelog for every bump, which is the fastest way to tell whether a
stale server would merely behave differently or would reject the payload
outright.
:::

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

Topology (vertex count, polygon count, edge count, object count, or a
pin group's vertex count) changed after **Transfer**, or a Static
object's captured deformation was re-recorded; UVs are not part
of the fingerprint and a UV-only edit never raises this. Click
**Transfer** again. Fetching across a mismatch would bind a PC2 to a
mesh of a different vertex count.

### "Objects missing UUID" / "Stale UUID references"

Usually after loading an old file or renaming objects. Tick **Debug
Options** in the Backend Communicator panel, then click **Run UUID
Migration**.

### `ValueError: Object '...' is assigned to both '...' and '...' groups`

The same object (identified by its UUID) appears in more than one active
group. Select it in a group's object list and click **Remove Object**
until it is left in exactly one group.

### "Object '...' has N isolated vertex(es)" (the message says "isolated vert")

A Static collider mesh has stray points that belong to no face, common in
imported models. Click **Remove Isolated Vertices** under the error to delete
them, then **Transfer** again. That button and the matching MCP pair
`remove_isolated_static_vertices` / `detect_isolated_static_vertices` work on
active Static colliders only, which is what this error names.

For the same points on any mesh, select the objects and run
**Scan Selected Meshes** in the **Utility Tools > Mesh Cleaning** panel,
then **Remove Loose Vertices** from the report. That route works on the
whole selection whatever the group type, keeps pinned vertices, skips a
Sand particle mesh (whose grain centers are legitimately in no face), and
names the caches the vertex-count change invalidates instead of leaving
them stale. See [Mesh Cleaning](workflow/scene/mesh_cleaning.md).

### Transfer or build rejects the mesh

Any rejection that is shaped like "this geometry is wrong" is worth
scanning before you debug it. **Scan Selected Meshes** in the
**Utility Tools > Mesh Cleaning** panel covers every geometry check
**Transfer** performs and adds several that nothing else performs, so
anything **Transfer** would reject on geometry grounds shows up in the
scan first, with its fix button attached:

| Scan reports                              | Fix                                                                             |
| ----------------------------------------- | ------------------------------------------------------------------------------- |
| Near-coincident vertex pairs              | **Merge by Distance**                                                            |
| Isolated vertices, hanging seam vertices  | **Remove Loose Vertices**                                                        |
| Degenerate (zero-area) faces              | **Dissolve Degenerate**                                                          |
| Duplicate faces                           | **Delete Duplicate Faces**                                                       |
| Inconsistently wound edges                | **Recalculate Outside**                                                          |
| Linked duplicate of another object        | `Object > Relations > Make Single User > Object & Data` (no button; Transfer refuses a shared mesh datablock) |

The first three change the vertex count and ask for confirmation first.

It does not work the other way around: a clean scan is not a promise that
**Transfer** will pass. **Transfer** also checks things that are not
geometry, such as an object assigned to two active groups at once, an
assigned object that no longer resolves, a **Reference Rest Angle**
object that has gone missing, a captured deformation whose vertex
count no longer matches its mesh, a pin whose vertex group is gone or
empty, and
a merge pair that cannot stitch (see
[Snap and Merge](workflow/constraints/snap_merge.md#merge-pairs-without-snapping)).
The solver then runs its own
intersection test when it builds the scene. None of those are visible to
the scan, and each is reported by name in the panel's error line.

The scan is also stricter than **Transfer** on two rows, on purpose.
Isolated vertices stop a **Transfer** only on a **Static** collider, and
hanging seam vertices only on a **Shell** or **Solid**, while the scan
reports both on any mesh you select. Where **Transfer** tolerates them
they are still worth removing: a point that belongs to no face has no
surface area and no mass, so it does nothing in the solve.

Boundary edges, non-manifold edges, and faces with more than three corners
are reported as notes rather than errors: an open, quad-built cloth mesh is
normal, and only a **Solid** group needs a closed surface. The
**Triangulate** button under the quad note leaves an explicit triangle
with no diagonal left for Blender to re-pick as the mesh deforms, which
is what keeps the displayed surface on the simulated one.

### "N self-intersections (... tri-tri, ... rod-tri)"

Two pieces of geometry already overlap, so the scene is refused before the
first frame is solved: a mesh folding through itself, a mesh through another
mesh, or a mesh through a collider. The count is of overlapping element
pairs, split by whether the pair is two triangles or a rod segment against a
triangle. The solver runs its own intersection test as it steps; an overlap
found there ends the run, and the log line reads `### intersection
detected`.

The fix is to separate the geometry: move the collider or the character to a
pose the garment sits outside of, or edit the mesh so the fold is gone.

When the overlap comes from the pose and cannot easily be fixed, such as a
garment whose armpit a rig folds through itself, turn on
[Allow Existing Intersections](workflow/params/material.md#allow-existing-intersections)
on the garment's group. Only the places that overlap at the start pass
through each other, and the rest of the garment keeps colliding. After the
build the viewport tints those places on the starting frame, so you can see
what was let through. The same setting covers a start pose refused with
*N element pairs too close*.

When the overlapping pieces should pass through each other rather than
collide (a mesh tangled in its pose that should stay tangled, or a cloth
whose self-collision is not wanted), the group settings under
[Allow Intersections](workflow/params/material.md#allow-intersections)
let the pairs you name pass through instead of stopping. An allowed pair
has no contact at all, so the solver does not push it apart; where the
geometry should end up separated, separate it before the run instead. Set
the settings on the moving object's group: a **Static** collider left in its
rest pose ignores the boxes on its own group. For geometry confined to a
pinned region, the narrower per-pin
[Allow Intersections Here](workflow/constraints/pins.md#allow-intersections-here)
does the same for the elements that pin holds completely.

### "This scene uses Allow Existing Intersections, which exempts only the pairs the scene-build check found"

The build accepted the start pose, but the solver's own check, run just
before the first frame, found an overlapping pair that the build did not.
The two checks measure positions at slightly different precision, so a pair
that barely touches can pass one and fail the other. The message names the
first few pairs.

Move the named pieces a little apart, or a little further into each other,
so both checks see the same thing, and build again. The solver does not add
such a pair to the allowed places by itself, so that only what the build
found and showed you is ever let through.

### Run button is disabled

Most often the previous simulation output is still attached to the
objects, and the panel shows *Clear local animation before running*:
click **Clear Local Animation**. **Run** is also greyed out while a
bake is in progress — let it finish or click **Abort**.

### Status: "Connection lost" during a run

Network hiccup, server crash (OOM, driver fault), or host reboot.
Reconnect; if frames were produced, **Fetch All Animation** pulls them. The
remote `server.log` has the cause.

### "Server error" in the status line

The server returned a JSON `error`, usually from a solver exception.
Read `server.log` for the traceback. Common causes: out of disk on the
remote, permission denied on the project directory, CUDA OOM.

### Status: "Simulation Failed"

The solver crashed instead of finishing. Backend Communicator prints
*Solver failed:* and a one-line cause — an intersection, a failed
continuous collision detection, a linear solver that would not
converge, a Newton solve that made no progress, out of GPU memory, an
unrecoverable CUDA fault, a kernel past the OS watchdog timeout, a GPU
architecture missing from the solver's device image, or a process
killed before it could report — with an untranslated detail line under
it carrying the machine data (a CUDA error name, a signal, a file and
line). The full report, solver stdout and stderr tails included, goes
to the add-on console: press **Show Console** on the row below. Under
that row, **Open Session Folder** opens the run's `session` directory,
where the solver's logs and status record are written. It is grayed
out with *The session folder is not known for this run* when the
remote root cannot be resolved, and the path it opens is a path on the
machine the **server** runs on, so it reaches real files only when
that machine is this one.

## Fetch and playback

### "N frames unfetched" in the Solver panel

The remote has frames the local Blender does not; the panel's line reads
`N frames unfetched. Press "Fetch All Animation".` Click **Fetch All
Animation**.

### Render with unfetched frames

A popup warns "N frames unfetched". Fetch first, then render. The
popup fires once per render and does not block.

### "Data path: ... does not exist"

The `data/<blend-file-name>/` folder referenced by a `MESH_CACHE`
modifier is missing (deleted, renamed, or not copied across machines).
Restore from backup, or click **Migrate data/...** to rebind.

## Bake

### "Remove all shape keys except Basis before baking"

Baking adds one shape key per frame and keyframes their values, which
would double-blend with any shape keys the mesh already carries. In
Object Data Properties, delete every shape key except `Basis` on the
listed objects.

## MCP server

### Port already in use

The add-on retries the base port a few times, then walks `9634`-`9642`
and binds the first free one. Started from **Start MCP Server on
Local** it does not do this silently: the substitution is reported as a
warning and written back into the panel's **Port** field, so the
**MCP Server (Running :port)** header names the live port. Check it
there if an external client points at the base.

### "Could not find available port in range 9633-9642"

All ten slots are taken. Kill the holder (`lsof -i :9633-9642` on
macOS/Linux, `netstat -ano | findstr 963` on Windows), or change the
base port.

### "Failed to start MCP server"

The server thread raised during startup (socket permission, port
collision, import error). The Blender system console prints the
exception as `MCP Server error: ...`.

### `run_python_script` returns `"status": "error"`

The snippet you sent raised. Read the `error` field; the full
traceback is in the Blender system console.

### `capture_viewport_image`: "No 3D viewport found"

The current Blender screen layout has no `VIEW_3D` area. Switch to
`Layout`, `Modeling`, or `Sculpting`, then retry.

## Debug Options

**Debug Options** is the checkbox on the same row as **Update Stat**
and **Show Console** in the Backend Communicator panel, and it is off by
default. Ticking it reveals nine labelled blocks below the panel's usual
contents. They are development and diagnosis tools rather than parts of
the simulation workflow: they address the connected machine directly,
exercise the transport on its own, or reach into the add-on itself.
What they run elsewhere - a shell command, a server query, a render -
reports back into the add-on console, which **Show Console** opens.
Controls that need a live connection are greyed out rather than hidden,
so the block always shows what would become available once you connect.

### Shell Calls

Two tools that both address the machine at the other end of the
connection, at two different levels.

**Exec Command via Server** reads the **Args** field as a run of
`--key value` pairs and sends them to the running solver server as a
single query, over the same socket the panel's status poll uses; the
JSON reply is printed to the console between two `------` rules. It
speaks the server's own text-command protocol rather than a shell, so it
needs both a connection and a started server, and it is the way to ask
the server something the panel does not put on screen.

**Execute Shell Command on Remote** runs the **Command** field on the
machine the backend runs on, in the directory the connection points at
(the solver root), and prints its stdout and stderr to the console. It
needs a connection but not a running server, which is what makes it the
tool for the case where the server will not start: list the venv, run
`nvidia-smi`, tail `server.log`. On a Docker backend the command runs
inside the container, not on the daemon host. **Run as Shell**, on by
default, is what makes pipes, redirection and `&&` work - it wraps the
command in `/bin/sh -c` on the SSH and Docker backends and hands it to a
shell on the three native ones (Windows Native, macOS Native, Linux
Native) - and there is rarely a reason to untick it.

### Data Transfer Tests

**Data Send** generates as many megabytes of random bytes as **Data Size
(MB)** asks for (1 to 256) and uploads them to `dummy_data.pickle` in
the remote root. **Data Receive** downloads that same file and compares
it byte for byte against the copy still held in memory, reporting either
"Data received matches test data." or an error.

The pair exists to take the scene out of the picture. When a
**Transfer** stalls, a fetch never finishes, or a payload arrives
damaged, the cause can be the link, the encoder, or the solver, and
nothing in the panel separates them. A block of random bytes has no
mesh, no encoder and no solver behind it, so a round trip that fails
here is the connection, and one that succeeds at a size comparable to
your scene moves the suspicion upstream. The progress bar and throughput
readout that run alongside it also give you the link's real speed, which
is the number to compare against when a transfer merely feels slow.

**Data Receive** stays unavailable until a **Data Send** has run in the
same Blender session: the reference copy it compares against is held in
memory, so before then there is nothing to check a download against.
Both need the server running, not just a connection. Two things worth
knowing before you read the result: on the three native backends
(Windows Native, macOS Native, Linux Native) the payload is written straight
to the filesystem instead of through the socket (unless `PPF_FORCE_TCP_TRANSFER=1` is set in the
environment), so there the round trip measures a file copy rather than a
network; and the test file is left behind in the remote root - nothing
in the add-on deletes it - so clean it up yourself after a large test.

### Options

A single field, **Max Console Lines** (default 60, from 8 to 10000).
Each time the add-on flushes queued messages into its console text
block, it trims the block back to that length. The cap is what keeps a
long run from growing the text block without bound; raise it when a
traceback or a **Simulation Failed** report scrolls past before you can
read it. It bounds only what Blender keeps in memory - a log file
written by the block below still receives every line.

### Console Log Export

The path field is the switch: while it holds a path, every line the
add-on writes to its console is also appended to that file, with the
parent directory created if it does not exist, and clearing the path
turns file logging off again. The folder button opens a file browser
(pre-filled with `log.txt` in your home directory when nothing is set
yet), the **X** clears the path, and **Delete Log** deletes the file
itself.

Reach for it when what you need to read outlives the console. The
console is trimmed to **Max Console Lines** and lives only as long as
the Blender session, so a crash, a driver fault, or an unattended
overnight run leaves nothing to inspect afterwards, while a file
survives all three. It is appended to and never truncated, so it
accumulates across sessions, which is what **Delete Log** is for.

### GitHub Repo on Remote

**Git Pull** and **Compile** run `git pull` and
`/root/.cargo/bin/cargo build --release` in the connection's directory,
by the same route as **Execute Shell Command on Remote** and with their
output in the console. Both need a connection and nothing else in
flight. Together they are the update loop for a remote you develop
against: pull, rebuild, then stop and start the server so the new binary
is the one running. The `cargo` path is hardcoded to
`/root/.cargo/bin/cargo`, which is where the project's container image
installs it; on a host that keeps cargo elsewhere the button fails, and
the shell field above is the way to run the build by hand.

**Open GitHub Link** sits under the same label but touches no connection
at all: it opens the project's public repository page in the browser on
your own machine.

### GitHub Repo on Local

**Git Pull (Local)** runs `git pull` as a subprocess whose working
directory is inside the add-on's own source tree, so it updates the copy
of the add-on that Blender is running - not the solver on the remote.
That only does something when the add-on is installed as a git checkout,
which is the developer layout; an extension unpacked from a zip is not a
repository, and the pull reports git's own complaint instead. It gives
up after 60 seconds, killing the process and reporting "Local git pull
timed out". Follow it with **Reload Add-on Now** below to run what you
just pulled without restarting Blender.

### UUID Migration

**Run UUID Migration** walks the scene and fills in the stable
identifiers the add-on uses to follow things across renames: a group's
own UUID, the UUID on every assigned object, the object UUID and
vertex-group content hash on every pin, and the object UUIDs on every
snap/stitch pair, including the ones stored inside a pair's stitch data.

The problem it solves is historical. Earlier versions referenced objects
and vertex groups by NAME, so renaming an object - or letting Blender
append `.001` to a duplicate - silently broke the reference. An object
now carries a UUID as a custom property and a vertex group is matched by
a hash of the vertex indices it holds, so both survive a rename. A
`.blend` saved before that change still carries name-only references,
and this pass is what converts them.

You rarely have to press it. The same pass runs automatically after a
file is loaded whenever anything is found missing, writing its result to
the console prefixed with `[auto-migrate]`, and **Transfer** refuses a
scene whose identifiers are incomplete with a message that ends "Run
UUID Migration first." rather than uploading it. The button matters for
what the automatic pass does not cover: a record inserted
programmatically rather than through the panel, and the case where the
automatic attempt raised and was swallowed, which leaves no message at
all. Pressing it always reports what it did.

Two details about the result. It is printed under the button and stays
there until the next run, so you can read it after the panel redraws;
and it ends with "Save to persist", which is literal - the identifiers
are scene data, and closing the `.blend` without saving loses them.
Every identifier it writes is one that was empty, so a second run on an
already-migrated scene changes nothing and says so.

### Render

**Render Animation** renders the scene's frame range one frame at a
time, driven by a timer, with a progress bar naming the frame and the
percentage done; **Stop** halts it.

It exists because of one specific interaction with curve playback. The
built-in **Render Animation** has been observed to evaluate the
depsgraph between `render_pre` and `frame_change_pre`, which leaves the
add-on's curve cache half-applied - about half the splines render at
their rest pose while the viewport looks correct. This loop sets the
frame first, which runs the add-on's playback handler to completion, and
only then renders that single frame. So reach for it when a rendered
animation of simulated curves or rods shows strands frozen in the rest
pose that the viewport shows moving.

Output goes to the paths the built-in render would use, one file per
frame from the scene's output settings, so it is a drop-in replacement
rather than a separate export. If files matching those paths already
exist, a dialog says how many, and they are deleted before the render
starts - cancel there if they matter. **Stop** sets a flag that is read
between frames, so the frame in flight finishes first; Blender exposes
no mid-frame cancel to Python. The playhead and the output path are
restored when the run ends or is cancelled.

### Add-on Local Debug Server

**Start** binds a TCP socket on `localhost` at **Port** (8765 by
default; the field is greyed out while the server runs), and it is what
the `blender_addon/debug/main.py` CLI talks to - see
[Debug CLI](troubleshooting.md#debug-cli) below. It is also what makes
reloading possible at all: **Reload Add-on Now** and **Full Reload** are
drawn whether or not the server is running, but both fail with an error
naming the reload server as not running until you press **Start**.

**Reload Add-on Now** deletes the add-on's modules from `sys.modules`,
invalidates the import caches, and disables and re-enables the add-on
within one event-loop tick, so what runs afterwards is the source
currently on disk.
**Full Reload** does the same work but splits the disable and the enable
across two ticks, which is what lets Blender rebuild RNA for a changed
`PropertyGroup`; reach for it when an edit to a property definition does
not show up after a plain reload (see
[Hot reload](troubleshooting.md#hot-reload)). Both are scheduled through
a timer rather than run inline, so the button returns immediately and
the reload happens a tick later; running the reload inline would free
the operator while its own Python frame is still on the stack, which
crashes Blender.

The server is restarted for you after a reload if it was running before
it, but it comes back on the default port rather than the one you typed,
and the CLI only ever talks to 8765. A custom **Port** is therefore only
useful for a session you drive by hand.

:::{warning}
While this server runs, any process on your machine can send it a JSON
packet that executes arbitrary Python inside Blender. There is no
authentication and no `Origin` check. Start it when you are debugging
and stop it when you are done; see
[Code Execution Risk](security.md#code-execution-risk-mcp).
:::

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

Plain reload times out at 45 s, full reload at 70 s. Something in
top-level module code, `register`, or `unregister` is blocking the
main thread (network call, large mesh op, missing
`bpy.app.timers.unregister`). The Blender system console shows what
was still running.
