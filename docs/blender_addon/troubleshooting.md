# 🩹 Troubleshooting

Find the message or symptom you see. Entries are grouped by where it shows
up. A heading in quotation marks is the text the add-on shows, and `...`
stands for a part that varies. For a longer report, press **Show Console** to
read the add-on console, or open `server.log` in the solver folder. Blender's
system console shows what the add-on prints outside its own console: on
Windows, open **Window > Toggle System Console**; on macOS and Linux, start
Blender from a terminal.

## Installation

### "Paramiko needs to be installed" or "Docker-Py needs to be installed"

The Backend Communicator panel shows an install button with this line under
it, and **Connect** stays grayed out until the install finishes. Click the
button that matches the **Type** you chose:

| Connection type | Button |
| --------------- | ------ |
| **SSH**, **SSH Command**, **Docker over SSH**, **Docker over SSH Command** | **Install Paramiko to Add-on Directory** |
| **Docker** | **Install Docker-Py to Add-on Directory** |

The button runs `pip` with Blender's own Python and installs the package into
the `scripts/addons/modules` folder of Blender's user folder. Blender needs
internet access for the install. The panel shows *Installing...* while it
runs.

### "The cbor2 module is required to encode the scene but is not installed"

`cbor2` encodes the scene for every **Transfer**. Without it, the Solver panel
shows *cbor2 is missing; Transfer and Run cannot encode*. The extension
installs `cbor2` together with the add-on through Blender's installer. If it
is missing, do one of the following:

1. Reinstall the extension through Blender (**Get Extensions** or **Install
   from Disk**).
2. Click **Install cbor2 to Add-on Directory** in the Backend Communicator
   panel. The button installs `cbor2==6.0.1` into `scripts/addons/modules`.

### "Failed to install ..." from an install button

The first two messages below appear under the button, and the add-on console
holds the same text with `pip`'s full output (press **Show Console**).

- *Failed to install `<package>`: `<pip's error>`*: `pip` ran and failed. Its
  own message names the cause.
- *Error during installation: Could not bootstrap pip (ensurepip exited N):
  ...*: Blender's Python could not set up `pip`.
- *Operation timed out*: Blender's status bar shows this when the button has
  stopped waiting after 2 minutes, but the install can still be running in the
  background. Wait until the *Installing...* line clears.

The folder the buttons install into, `scripts/addons/modules` in Blender's
user folder, must be writable.

To install by hand, run `pip` with Blender's own Python (its path is
`sys.executable` in Blender's Python console), not the system Python:

```bash
<blender python> -m pip install --target <blender user scripts>/addons/modules paramiko docker
```

## Connection: before you connect

### Connect is grayed out

**Connect** stays unavailable until the connection type's fields are valid.
Check these, and see {ref}`When Connect is unavailable <connect-unavailable>`
for the full table:

- *Project name should not contain spaces or special characters*: **Project
  Name** may hold only letters, digits, `.`, `-` and `_`. An empty **Project
  Name** also disables **Connect**, and the panel draws no message for it.
- An empty required field (**Host**, **SSH Command**, **Container** or
  **Solver Path**, depending on the type): fill it in.
- *Path should not contain spaces or special characters* under **SSH Key**,
  **Remote Path** or **Container Path**, and *Path should not contain special
  characters* under **Solver Path**: remove the character.
- An install button for Paramiko or Docker-Py: see
  [Installation](#installation).
- **Connecting...** on the button, or an open connection: wait, press
  **Cancel**, or press **Disconnect**.

### "Connection timed out" and Cancel

A connect that has not finished in 60 seconds is canceled and reported as
*Connection timed out*. Pressing **Cancel** under the status line ends it
sooner. Either way the attempt is closed, and the next **Connect** starts from
a clean state. If it keeps timing out, run `ssh <user>@<host>` (or `ssh -J
<jump> <host>`) in a terminal. If that stalls too, the host, a firewall or a
jump host is not answering.

## Connection: SSH

These apply to **SSH**, **SSH Command**, **Docker over SSH** and **Docker
over SSH Command**. See [SSH (Direct)](connections/ssh.md) and
[Docker over SSH](connections/docker_over_ssh.md).

### "Authentication failed."

This is the SSH library's own text, shown as the connection error. Check
that:

1. **SSH Key** points at the private key, not the `.pub` file.
2. The key has no passphrase, or your `ssh-agent` has it loaded.
3. The matching public key is in `~/.ssh/authorized_keys` on the remote host.

See [SSH Keys](connections/ssh.md#ssh-keys).

### "Failed to parse command. Ensure it includes host."

The **SSH Command** (or **Docker over SSH Command**) text names no
destination. The add-on reads `[user@]host` or `ssh://[user@]host[:port]`,
plus `-p`, `-i`, `-l`, `-J` and `-o Port=`, `-o IdentityFile=`, `-o User=`,
`-o ProxyJump=`. Other known ssh options are accepted and ignored. Other
messages report a different fault in the command:

- *Cannot split the command into arguments: ...*: an unbalanced quote.
- *Unrecognized ssh option '-`<letter>`'*: an option letter that ssh does not
  have.
- *Option '-p' is missing its argument*: an option that needs a value ends the
  command.
- *'`<value>`' is not a port number* or *Port N is outside 1-65535*: the
  port given to `-p` is not valid.

A `ProxyCommand` is not read. See the next entry.

### `~/.ssh/config` host works in a terminal but not in the add-on

The add-on reads only `Host`, `HostName`, `Port`, `User`, `IdentityFile`,
`ProxyJump` and `Include` from `~/.ssh/config`. It ignores `ProxyCommand`,
`Match` and certificates. A bastion needs nothing done by hand: `ProxyJump` is
honored, and the **Proxy Jump** field and `ssh -J` set the same thing. For an
ignored directive, bring the tunnel up from a terminal (`ssh -L
2222:gpu01.example.com:22 bastion.example.com`) and point the add-on at
`localhost:2222`. See
{ref}`Supported ssh_config options <supported-ssh-config-options>`.

### "Jump host ... failed"

The message reads *Jump host `<host>:<port>` failed: `<reason>`*. The hop it
names is the one that failed, and the reason after the colon can be a refused
connection, an unreachable host, a handshake that took longer than 20 seconds,
or a rejected key. The hops opened before it are closed. Test the same chain
from a terminal (`ssh -J <jump> <host>`).

A message that reads *jump hosts loop back on themselves: a -> b -> a* means
the `ProxyJump` entries lead back to a host already in the chain, whatever the
chain length. The trail it followed is printed with it. See
[Jump Hosts](connections/ssh.md#jump-hosts).

## Connection: Docker

These apply to **Docker**, **Docker over SSH** and **Docker over SSH
Command**. See [Docker (Local)](connections/docker.md) and
[Docker over SSH](connections/docker_over_ssh.md).

### "Container '...' does not exist"

Run `docker ps -a` on the Docker host, correct **Container**, or create the
container.

### "Error starting container '...'"

The container exists but `docker start` failed, and the text after the colon
is the daemon's reason. Run `docker start <container>` on the Docker host to
see the same error.

### "Cannot reach the Docker daemon (...)"

Only the **Docker** type shows this. Start Docker (for example Docker
Desktop), and check that `docker version` works in a terminal as the same
user. On Linux, add yourself to the `docker` group with `sudo usermod -aG
docker $USER`, then log out and back in.

### "Error: ..." after Connect (Docker over SSH)

The line carries the stderr of a `docker` command that ran on the remote host.
Run the same command in a terminal on that host.

### "Container '...' does not publish port N to this machine"

Raised by **Connect** on the **Docker** type. The container was created
without the server port published. Recreate it with `docker run -p N:N`, using
the same number on both sides. The add-on cannot publish a port on an existing
container. A container on host networking is accepted as it is. See
{ref}`Publishing the port for Docker <docker-port-publishing>`.

### "Docker port N is not exposed on container '...'"

Raised at **Start Server on Remote** on **Docker over SSH** and **Docker over
SSH Command**, from `docker port <container> N` on the remote host. Recreate
the container with `docker run -p N:N`, or add the equivalent `ports` entry to
your compose file. See
{ref}`Publishing the port for Docker <docker-port-publishing>`.

## Connection: the solver path

### "ppf-cts-server not found under ... on the solver host, in any layout"

This message is for the SSH and Docker types. Point **Remote Path** (SSH
types) or **Container Path** (Docker types) at the solver root: the directory
that has `target/release/`, `target/cuda/release/`, `target/rocm/release/` or
`target/cpu/release/` under it, for example `/root/ppf-contact-solver`. Do not
point at `target/release` itself. The message names **Remote Path** for the
Docker types as well. If the host has no solver, build one there with `cargo
build --release`, which builds the server, the solver and the Python
extension together. The message itself may suggest `cargo build --release -p
ppf-cts-server`, which builds only the server, so use the plain command. See
[SSH (Direct)](connections/ssh.md).

Two related messages come from the same check:

- *The connection names no directory on the solver host, so its solver builds
  could not be listed.*: the path field is empty.
- *Could not list the solver builds on the solver host: ...*: the command that
  lists the builds failed, and the text after the colon says why.

A folder that holds a solver, but not the build that the **GPU** / **CPU**
buttons (the panel does not label them, and tooltips and these messages call
the choice **Compute Device**) or the **GPU Backend** selection asks for, is
refused with a different message that names what is there. See {ref}`Choosing the build <choosing-the-build>`.

## Connection: the native types

These apply to **Windows Native**, **macOS Native** and **Linux Native**, the
connection types whose solver runs as a child process on the machine Blender
runs on. See [Windows](connections/windows.md), [macOS](connections/macos.md)
and [Linux](connections/linux.md).

### "ppf-cts-server not found under ..." and "Solver path is not set"

**Solver Path** must be the root of the solver, not `target/release` inside it
and not the folder you extracted the archive into. A subfolder is resolved
upward automatically, so the message means that no folder within six levels
above your choice holds a solver in an accepted layout:

| Type | Message starts | Accepted layouts under the root |
| ---- | -------------- | ------------------------------- |
| **Windows Native** | *ppf-cts-server.exe not found under ...* | `target\release\`, `target\cuda\release\`, `target\rocm\release\`, `target\cpu\release\`, `bin\` or `bin-cpu\` |
| **macOS Native** | *ppf-cts-server not found under ...* | `target/release/` or `target/cpu/release/` |
| **Linux Native** | *ppf-cts-server not found under ...* | `target/release/`, `target/cuda/release/`, `target/rocm/release/` or `target/cpu/release/` |

`target/metal/release/` is also accepted on every native type. `bin/` is not a
server folder on macOS or Linux. To build from source, run `cargo build
--release` in the checkout (or `build-win-native\build.bat`,
`build-mac-native/build.sh` or `build-linux-native/build.sh`). It builds the
server, the solver and the Python extension together, and a run takes all
three from the same build folder. The message itself may suggest `cargo build
--release -p ppf-cts-server`, which builds only the server, so use the plain
command.

*Solver path is not set* means **Solver Path** is empty. Fill it in. The
**Connect** button is grayed out while the field is empty.

### "... holds the CPU build of the solver and no GPU build"

The folder is right and **Compute Device** is not. Set it to the build that is
there, or point **Solver Path** at a folder that has the one you want. The
reverse message appears for a GPU-only folder with **CPU** selected. The
add-on never runs the other build in its place. See
{ref}`Choosing the build <choosing-the-build>`.

### "... holds these GPU builds: cuda, and no rocm build"

**GPU Backend** names an accelerator this folder does not carry. Set it to one
of the builds named, or to **Automatic**.

### "No GPU build in this folder has a usable device"

Raised at **Start Server on Remote** when **GPU Backend** is **Automatic**,
the folder holds more than one GPU build, and none of them reports a device it
can run on. The message carries what each backend said. Fix the GPU driver, or
set **Compute Device** to **CPU**.

### "A solver server is already running on port N, and its runs use ..."

A `ppf-cts-server` is already running on the port, left over from an earlier
session or started from another build folder, and it runs a different build
from the one **Compute Device** names. Attaching to it would run every solve
on that other build, so **Connect** refuses. Press **Force Terminate Process**,
or change **Compute Device** to the build the server runs. The message names
both build folders.
See {ref}`Attaching to a running server <attach-mismatch>`.

### "Embedded Python not found in ... or ..." (Windows)

**Solver Path** must hold a `python\` folder (with `python\python.exe`) next to
`bin\`. Extract the whole Windows distribution again. For a build from source,
run `build-win-native\warmup.bat`, which creates `build-win-native\python\`.

On macOS and Linux there is no such message. A distribution that reports a
module it ships (for example *No module named 'pytetwild'*) is running under a
different Python than its own.

:::{admonition} Under the hood
:class: toggle

On macOS and Linux the add-on names the build worker's interpreter itself:
`<root>/python/bin/python3` for a distribution and
`~/.local/share/ppf-cts/venv/bin/python` for a checkout. If neither exists,
the build worker reports the missing module itself. If `PPF_CTS_BUILD_PYTHON`
is set in the environment Blender inherited, that value takes precedence over
both, so unset it to use the distribution's own Python.
:::

### CUDA DLL load errors (Windows)

`server.log` names a missing CUDA runtime DLL. The distribution carries the
runtime in `bin\`, so the folder is incomplete or the NVIDIA driver is
missing or too old. Update the driver, extract the distribution again, or set
**Compute Device** to **CPU**.

### "... entries are still marked com.apple.quarantine" (macOS)

The add-on prints this line to Blender's system console (start Blender from
Terminal to see it), not to the panel. If macOS then refuses to load a file,
the panel shows the failure to start the server. The add-on clears the
download mark from a packaged distribution before it starts the server, and
this line means it could not write to the folder, which is what a folder owned
by another user or on a read-only volume looks like. Move the folder somewhere
you own, or clear the mark yourself:

```bash
xattr -s -d -r com.apple.quarantine "<root>"
```

## Connection profiles

### The profile dropdown shows "(No Profile)" after Open Profile

The file is missing, cannot be parsed as TOML, or holds no entries. The
add-on does not report the parse error. Run the file through a TOML
validator. Common mistakes are unclosed quotes and unescaped backslashes in
Windows paths (write `\\` or use forward slashes).

### A profile loads but the fields do not change

The entry's `type` value is missing or is not spelled exactly like one of
`SSH`, `SSH Command`, `Docker`, `Docker over SSH`, `Docker over SSH Command`,
`Windows Native`, `macOS Native` or `Linux Native`. Case matters. The fields
keep their previous values.

:::{note}
**Save Profile** rewrites the whole TOML file. Comments and the original
formatting are lost. Keep a backup if comments matter. See
[Connection Profiles](connections/profiles.md).
:::

## Server startup

### Status stuck on "Waiting for Server Start..."

You connected but did not press **Start Server on Remote**, or
`ppf-cts-server` exited before it was ready. Press **Start Server on Remote**.
If it then times out, see the next entries. A connect that hangs is covered
under [Connection timed out](#connection-timed-out-and-cancel). **Cancel**
under the status line also ends *Server Starting...*: it stops the launch and
the server and leaves you connected.

### "Server startup timed out." (SSH and Docker types)

The server did not report ready within 16 seconds. The panel pastes the last
20 lines of `server.log` under the message.

1. Read the pasted lines.
2. For more, open `server.log` in the solver folder on the solver host, or
   run `tail -50 server.log` with **Execute Shell Command on Remote** (see
   [Shell Calls](#shell-calls)).
3. To check the GPU driver from Blender, run `nvidia-smi` the same way.

### "Failed to start server: ..." (native types)

The three native types prefix every start-up failure with *Failed to start
server:*, and none of them pastes the log. The text after the prefix is the
cause, such as the missing embedded Python or the build with no usable GPU
device described above. Two causes come from the wait for the server to
answer:

- *Failed to start server: Windows Native server did not become ready within
  16 seconds. Check server.log.* (*macOS Native* and *Linux Native* read the
  same): the process started but did not answer in time.
- *Failed to start server: ... server exited with code N before becoming
  ready. Check server.log.*: the process ended during startup.

Open `server.log` in the folder set under **Solver Path**. The panel does not
show the log.

### "Server startup failed: ..." (SSH and Docker types)

`progress.log` on the solver host contained a line with `ERROR` or `FAILED`
during startup, and the panel quotes it. Open `server.log` in the solver
folder on the solver host for the full output.

### "Failed to launch server" (SSH and Docker types)

The add-on writes a start-up script to `/tmp` on the solver host and runs it,
and that script exited with an error. Check that `/tmp` on the host is
writable and is not mounted `noexec`.

:::{admonition} Under the hood
:class: toggle

The script is `/tmp/start_server.sh`. It starts the server in the background
with `nohup`, so a missing virtual environment or `python3` does not show
here. It appears later, as a `ModuleNotFoundError` from the build worker, the
process the server starts to build the scene. On the SSH and Docker types the
script activates the virtual environment at `$HOME/.local/share/ppf-cts/venv`
on the solver host when that folder exists, and starts the server without it
otherwise.
:::

### "Server port N is already in use on the remote host ..." (SSH and Docker types)

The server could not bind its port on the solver host or in the container. The
holder can be a leftover `ppf-cts-server` or a Docker container that publishes
the same port. Despite the wording, **remote host** means the container on the
**Docker** type. The message tells you to press "Kill Process", which is the
**Force Terminate Process** button.

1. While connected, press **Force Terminate Process**. It ends a
   `ppf-cts-server` that was started with `--port N` on the host or in the
   container.
2. Press **Start Server on Remote** again.
3. If another program holds the port, stop it on the host, or choose another
   server port. The port is the **Docker Port** field, which the panel draws
   only for the Docker types. For an SSH type, set `docker_port` in a
   [connection profile](connections/profiles.md).

### "Server reached SERVER_READY but the client cannot reach it" (SSH and Docker types)

The server is running, but this machine cannot reach it on the server port.
Check that the port is forwarded to this machine. For a container, that means
it was created with `-p N:N`, with the same number on both sides. See
{ref}`Publishing the port for Docker <docker-port-publishing>`.

### "Port N is in use"

Another program on this machine holds the server port. The three native types
raise this. A native type first asks the port whether a `ppf-cts-server`
answers there, and attaches to one that does, which lets you restart Blender
without losing the server. This error means the holder is something else.

1. Read the line under the **Force Terminate Process** button. *Port N is held
   by another program* means the listener is not a solver server. *Nothing is
   listening on port N* means the port is free, so try again.
2. Press **Force Terminate Process** to end the listener. It ends whatever
   process listens on that port, on Windows together with its child processes,
   and it does not check what that program is.
3. If the program is not yours to end, stop it another way, or move the solver
   to another port. The port is the **Docker Port** field, which the panel
   draws only for the Docker types. Select a Docker type, change the value,
   and switch back, or set `docker_port` in a
   [connection profile](connections/profiles.md).

### **Force Terminate Process** does not appear

The button is drawn under the error line, outside the collapsible Connection
box, only while one of three failures stands: a protocol version mismatch, an
error that names a held port, or, on the native types, any other error while
disconnected. On the SSH and Docker types it is drawn only while connected.
See
[When the Force Terminate Process row is drawn](connections/index.md#when-the-force-terminate-process-row-is-drawn)
for the exact conditions and
[Port Already in Use](connections/index.md#port-already-in-use) for what the
button ends on each type.

### Status: "Protocol Version Mismatch"

The status line reads *Protocol Version Mismatch*, and the error line reads
*Protocol version mismatch: server reports X, add-on expects Y.* The same text
goes to the add-on console. The solver server and the add-on speak different
protocol versions. The add-on stops the server when it detects this, so press
**Start Server on Remote** to restart from the on-disk binary. If the message
comes back, the two halves are different releases: update the add-on, or take
the solver image or the Windows, macOS or Linux distribution that matches it.
**Force Terminate Process** is drawn in this state for a server that is still
running from an older binary.

:::{admonition} Under the hood
:class: toggle

Both versions come from one file, `blender_addon/protocol_version.toml`. The
add-on reads it at start, and the server binary carries the value compiled
into it. The add-on compares the two numbers for equality. The file also lists
what changed with each version, which tells you whether a stale server would
merely behave differently or reject the data it is sent.
:::

## Object groups and pins

### "Maximum number of groups reached"

A scene holds up to 32 object groups. Delete groups you do not use, or assign
more objects to an existing group of the same type.

### "Object '...' is already in another group"

Each object can belong to one active group. **Add Selected Objects** reports
this warning and skips the object. The same fault at **Transfer** reads
*Object '...' is assigned to both '...' and '...' groups*. In either case,
select the object in the group that should not hold it and click **Remove
Object**, until it is left in exactly one group.

### "Object '...' is library-linked and cannot be assigned"

A library-linked object is read-only, and the add-on stores an ID on each
assigned object. Use **Object > Relations > Make Local > Selected Objects and
Data**, then add the object again.

### **Add Selected Objects** refuses an object

Besides the two messages above, the button skips an object and reports one of
these:

- *Object '...' is a Linked Duplicate (shares mesh data with '...')*: make it
  single-user with **Object > Relations > Make Single User > Object & Data**.
- *Object '...' has N duplicate face(s) ...*: press **Delete Duplicate Faces**
  in [Mesh Cleaning](workflow/scene/mesh_cleaning.md), or use **Mesh > Merge >
  By Distance** in Edit Mode.
- *Object '...' is not a particle mesh. Convert to particle mesh first.*: a
  **Sand** group accepts only a particle mesh. Use **Convert To Solid Particle
  Mesh**, described under
  [Creating a Sand Body](workflow/params/material.md#creating-a-sand-body).

### Pin button errors

Some of these come from creating a pin and from the buttons drawn in Edit Mode
(**Select**, **Deselect**, **Make Keyframe** and **Delete All Keyframes**).
Others come from **Add Operation** and **Capture Deformation**, which work in
any mode.

- **"No active edit object"**: enter Edit Mode on a mesh, a curve or a Hair
  Curves object in the active group.
- **"Name cannot be empty"**: type a pin name.
- **"No vertices selected"**: select at least one vertex (or curve control
  point).
- **"'...' is not in this group"**: add the object with **Add Selected
  Objects** first.
- **"'...' is not writable (library-linked)"**: make the object local, as
  described above.
- **"No pin vertex group selected"**: click a row in the pin list first.
- **"Edit object must be '...'"**: **Select** and **Deselect** act on the
  pin's own object, which must be the one in Edit Mode.
- **"Vertex group '...' not found"** and **"No pin vertices found"**: the
  pin's vertex group was deleted, or holds no vertices (on a curve, no
  control points). Restore it and assign vertices to it, or remove the pin.
- **"Keyframing not supported for curve pins"** and **"Keyframe deletion not
  applicable for curve pins"**: information, not a failure. **Make Keyframe**
  and **Delete All Keyframes** do nothing on a pin on a curve rod.
- **"Pin has Move/Spin/Scale/Torque operations; keyframed animation cannot be
  combined with them"** and **"Pin is keyframed; remove its keyframed
  animation before adding ..."**: a pin follows keyframes or operations, not
  both. Remove the pin's operations before **Make Keyframe**, or press
  **Delete All Keyframes** before adding an operation.
- **"Pin is captured ..."**: press **Clear Deformation Cache** before **Make
  Keyframe**.
- **"Pin has manual keyframes ..."**: press **Delete All Keyframes** before
  **Capture Deformation**.
- **"Torque cannot be mixed with Move/Spin/Scale operations"** and **"Cannot
  add Move/Spin/Scale to a pin that has Torque"**: give the torque a pin of
  its own.
- **"Pin object has no deforming modifier stack ..."**: **Capture
  Deformation** needs an enabled deforming modifier on the pin's mesh. The
  panel grays the button out when there is none, so this message reaches you
  from an MCP or Python call. The message's list also names Shape Keys, but
  shape keys or drivers alone do not count; record those with **Make
  Keyframe**. See
  [Capture Deformation](workflow/constraints/pins.md#capture-deformation).

## Hair Curves

These are specific to a Hair Curves object in a **Rod** group. See
[Hair Curves](workflow/scene/hair_curves.md).

### Hair does not move at all

A pin with **Pull** on holds every point it covers in place: a pulled point
does not fall or swing on its own. A weighted pin whose weight is above zero
along a whole strand therefore holds the whole strand still, at any
**Strength**. Let the weight reach `0` where the hair should move. See
[Let the Weight Reach Zero Where Hair Should Move](workflow/scene/hair_curves.md#let-the-weight-reach-zero-where-hair-should-move).

### The roots of hair slide on the scalp

A pin with **Pull** on that covers the root points is listed below **Roots**,
so it wins those points and holds them softly. Move **Roots** below it with
the **▼** button. The panel shows *Pins above lose shared points to this one:
move Roots below* under the **Pull** row of the pulling pin while the order
stands that way. See
[Put Roots Below the Weighted Pin](workflow/scene/hair_curves.md#put-roots-below-the-weighted-pin).

### Hair hangs from its roots whatever the Bend Stiffness

A strand turns freely about the last point an exact pin holds, so **Roots**
alone does not keep hair pointing the way it was combed, and a higher **Bend
Stiffness** only keeps each strand straight. Hold the points after the roots
with a weighted pin that pulls. See
[What an Exact Pin Holds on a Strand](workflow/scene/hair_curves.md#what-an-exact-pin-holds-on-a-strand).

### Transfer refuses the hair

**Transfer** names the hair object and says which of these it found:

- *holds no curves*, *has 1 point(s)* or *is a Bezier curve* (or *NURBS*): see
  [What a Strand Has to Be](workflow/scene/hair_curves.md#what-a-strand-has-to-be).
- *is in group '...', a SHELL group, and is simulated as rods only*: move the
  hair to a **Rod** group.
- *its captured deformation cache is missing*: press **Capture Deformation**
  on that pin again, or **Clear Deformation Cache**.
- *its modifier 'Hair Dynamics' is pinned to the end of the stack*: the body
  under the hair changes shape. Turn off **Pin to Last** on that modifier and
  move it above the first modifier that generates curves, or disable it.
- *STATIC object '...' ... has a deforming modifier stack (or shape-key
  animation) but no deformation cache*, naming the head: **Quick Fur** puts a
  Geometry Nodes modifier on the mesh it grows on. Capture the head. See
  [Hair on a Moving Body](workflow/scene/hair_curves.md#hair-on-a-moving-body).
- *has no vertex groups, so its material map cannot read vertex group '...'*:
  set the map's **Source** to **Attribute**.

## Transfer, run, fetch

### "Mesh topology changed since last transfer"

The full text is *Mesh topology changed since last transfer (groups differing:
`SHELL_<group ID>`). Re-transfer to sync.* It appears as a warning when you
press **Run**, **Resume** or **Fetch All Animation**, and it does not stop the
action. The names in parentheses are group identifiers (the group type
followed by an ID), not the names shown in **Dynamics Groups**, and each one
is a group that differs.

A group differs when its vertex count, polygon count, edge count or object
count changed after **Transfer**, when a pin vertex group changed size, or
when a Static object's captured deformation was recorded again. A UV-only edit
never raises it. Click **Transfer** again. Fetching animation for a mesh whose
vertex count changed would put frames on the wrong vertices. See
[Mesh Topology Changed Warning](workflow/sim/simulating.md#mesh-topology-changed-warning).

### "UUID migration needed", "Objects missing UUID" and "Stale UUID references"

The full text is *UUID migration needed (...). Run UUID Migration first.*, and
the parentheses count the groups, objects, pins or merge pairs that carry no ID
yet. *Objects missing UUID: Cloth. Run UUID Migration first.* asks for the same
repair for objects. These appear when the file holds references with no stored
ID, for example records that a script added directly. Tick **Debug Options** in
the Backend Communicator panel, click **Run UUID Migration**, then **Transfer**
again. See [UUID Migration](#uuid-migration).

*Stale UUID references: Cloth. Run UUID Migration first.* means the ID stored
with an assigned object differs from the ID the object itself carries.
**Run UUID Migration** writes only IDs that are empty, so it leaves a stale one
as it is. Select the object in its group, click **Remove Object**, and add it
again with **Add Selected Objects**, which stores the object's current ID.

### "Project name out of sync: UI='...' but active session='...'"

The session takes the project name when you press **Connect**. The name
shown in **Project Name** was changed afterward, for example by an MCP call.
**Transfer**, **Run**, **Resume** and **Update Params on Remote** refuse until
the two match. Press **Disconnect**, then **Connect**.

### "Object '...' has N isolated vertex(es)" (the message says "isolated vert")

The error line reads *Object '...' in group '...' has N isolated vertex(es)
(present in the mesh but in no triangle/face) ...*. A Static collider mesh has
stray points that belong to no face, for example points left behind in an
imported model. Click **Remove Isolated Vertices** under the error, then
**Transfer** again. That button and the MCP tools
`remove_isolated_static_vertices` and `detect_isolated_static_vertices` work
on active Static colliders only, which is what this error names.

For the same points on any mesh, select the objects and run **Scan Selected
Meshes** in the **Mesh Repair > Mesh Cleaning** panel, then **Remove Loose
Vertices** from the report. That route covers the whole selection, keeps
pinned vertices, and skips a Sand particle mesh. Do not use it on a Rod mesh:
a Rod has no faces, so the scan reports its unpinned vertices as stray and the
repair deletes them. See [Mesh Cleaning](workflow/scene/mesh_cleaning.md).

### Static collider refused: deformation cache and motion

**Transfer** names the object and its group in each of these refusals for a
Static collider. **Capture Deformation** and **Clear Deformation Cache** sit
below the group's object list and act on the selected row, so select the
object's row in the Static group before you click either:

- *has a deforming modifier stack (or shape-key animation) but no deformation
  cache*: select the object's row and click **Capture Deformation**, or remove
  the deformer if its motion should not reach the solver.
- *has an empty deformation cache*: select the object's row and click
  **Capture Deformation** again.
- *deformation cache has N vertices but the mesh now has M*: the mesh was
  edited after the capture and its vertex count changed. Select the object's
  row, click **Clear Deformation Cache**, then **Capture Deformation**. The
  message spells the first button *Clear Cache*.
- *has Static ops and is also moved by its parent, a constraint, a driver or
  an NLA strip*: a collider takes one source of motion. Remove its Static ops,
  or stop the other motion.

See [Armature-Driven Static
Objects](workflow/scene/static_objects.md#armature-driven-static-objects) and
[Moving a Static Object](workflow/scene/static_objects.md#moving-a-static-object).

### Transfer or build rejects the mesh

A rejection that says the geometry is wrong is worth scanning before you debug
it. **Scan Selected Meshes** in the **Mesh Repair > Mesh Cleaning** panel
reports the geometry faults **Transfer** refuses and several it does not, and
attaches a fix button to most. The table pairs each **Transfer** message with
its fix:

| Transfer message, or scan row | Fix |
| ----------------------------- | --- |
| *... is a Linked Duplicate (shares mesh data with ...)* | **Object > Relations > Make Single User > Object & Data**. The scan has no button for it, because repairing shared data would edit every object that uses it. |
| *... has N duplicate face(s) ...* | **Delete Duplicate Faces** |
| *... has N vertex(es) sitting midway along a sewing (loose) edge ...* (scan row: Hanging seam vertices) | **Mesh > Dissolve > Dissolve Vertices** in Edit Mode keeps the seam. **Remove Loose Vertices** cuts it. |
| *... has N isolated vertex(es) ...* (Static collider only) | **Remove Isolated Vertices**, or **Remove Loose Vertices** |
| *... tessellates into N triangle(s) with no usable rest shape ...* (the **Cannot Transfer This Mesh** dialog) | **Triangulate Faces**, offered in the dialog and under the panel error when a different split fixes every flagged face. A face that is already a triangle needs its vertex moved or welded. A face with no area, or with a zero-length edge, has no split at all: use **Merge by Distance** or **Dissolve Degenerate**. |
| Near-coincident vertex pairs (scan only) | **Merge by Distance** |
| Degenerate (zero-area) faces (the *no usable rest shape* message, and the scan) | **Dissolve Degenerate** |
| Inconsistently wound edges (scan only) | **Recalculate Outside** |

**Merge by Distance**, **Remove Loose Vertices** and **Dissolve Degenerate**
change the vertex count and ask for confirmation first. See
[The Repairs](workflow/scene/mesh_cleaning.md#the-repairs) for what each one
does, and
[Which Errors Transfer Also Refuses](workflow/scene/mesh_cleaning.md#which-errors-transfer-also-refuses)
for the group types each refusal covers.

A clean scan does not promise that **Transfer** passes. **Transfer** also
refuses things that are not mesh geometry, and the panel's error line names
each one. Examples are an object assigned to two active groups, a **Reference
Rest Angle** object that has gone missing, a captured deformation whose
vertex count differs from its mesh, a pin whose vertex group is gone or
empty, and a merge pair that cannot stitch (see
[Snap and Merge](workflow/constraints/snap_merge.md#merge-pairs-without-snapping)).
The solver also runs its own intersection test when it builds the scene,
which the scan does not do.

### "N self-intersections (... tri-tri, ... rod-tri)"

Two pieces of geometry already overlap, so the scene is refused before the
first frame is solved: a mesh folding through itself, a mesh through another
mesh, or a mesh through a collider. The count is of overlapping element pairs,
split by whether the pair is two triangles or a rod segment against a
triangle. The solver runs its own intersection test as it steps. An overlap
found there ends the run, and the log line reads `### intersection detected`.

The fix is to separate the geometry: move the collider or the character to a
pose that the garment sits outside of, or edit the mesh so the fold is gone.

When the overlap comes from the pose and cannot easily be fixed, such as a
garment whose armpit a rig folds through itself, turn on
[Allow Existing Intersections](workflow/params/material.md#allow-existing-intersections)
on the garment's group. Only the places that overlap at the start pass through
each other, and the rest of the garment keeps colliding. After the build, the
viewport tints those places on the starting frame. The same setting covers a
start pose refused with *N element pairs too close*.

When the overlapping pieces should pass through each other rather than collide
(a mesh tangled in its pose that should stay tangled, or a cloth whose
self-collision is not wanted), use the group settings under
[Allow Intersections](workflow/params/material.md#allow-intersections). An
allowed pair has no contact at all, so the solver does not push it apart. If
the geometry should end up separated, separate it before the run. Set the
options on the moving object's group, because a **Static** collider left in
its rest pose ignores the boxes on its own group. For geometry confined to a
pinned region, the per-pin
[Allow Intersections Here](workflow/constraints/pins.md#allow-intersections-here)
does the same for the elements that pin holds completely.

### The viewport marks the geometry at fault

When the scene build refuses a scene, the 3D viewport marks the geometry at
fault and prints a count label above the first mark. The marks stay until the
next **Transfer**, which clears them and marks whatever that build refuses, or
until **Resolve Intersections** or **Resolve All Intersections** moves an object (see
[Intersection Correction](workflow/scene/mesh_cleaning.md#intersection-correction)),
since the marks then point at where the surfaces used to be. The **Visualization**
checkboxes do not hide them. At most the first 100 items of each kind are
marked, while the number in the label is the total.

| Label in the viewport | Mark | What it points at |
| --------------------- | ---- | ----------------- |
| *N Self-Intersections* | Red tint on triangles | Triangles that overlap at the start. For a rod segment against a triangle, the triangle is marked. |
| *N Contact-Offset Violations* | Orange tint on triangles and edges | Pairs that start closer together than their contact offsets allow. |
| *N Wall Violations* | Red dots | Vertices that start on the wrong side of a wall in **Invisible Colliders**. |
| *N Sphere Violations* | Purple dots | Vertices that start on the wrong side of a sphere in **Invisible Colliders**. |

Move or edit the marked geometry, then press **Transfer** again. For
overlapping **Solid** or **Pdrd** objects, **Resolve Intersections** in
**Mesh Repair** moves them apart for you, and **Resolve All Intersections**
there also reshapes **Shell** cloth that passes through itself or through
another surface. When the
label counts more than the viewport marks, the next ones are marked after that
build. The light blue *Existing Intersections Allowed* tint is not an error; see
[Allow Existing Intersections](workflow/params/material.md#allow-existing-intersections).

### "This scene uses Allow Existing Intersections, which exempts only the pairs the scene-build check found"

The build accepted the start pose, but the solver's own check, run just before
the first frame, found an overlapping pair that the build did not. The two
checks measure positions at slightly different precision, so a pair that
barely touches can pass one and fail the other. The message names the first
few pairs.

Move the named pieces a little apart, or a little further into each other, so
both checks agree, and build again. The solver does not add such a pair to the
allowed places by itself, so only what the build found and showed you is let
through.

### **Run** is grayed out

**Run** is available when all of these hold:

1. You are connected and the server is started (**Connect**, then **Start
   Server on Remote**).
2. The status line reads *Ready to Run*, *Resumable* or *Simulation Failed*.
   Another status, such as *Waiting for Data*, means **Transfer** is still
   needed.
3. No simulation output is attached to the objects. Otherwise the panel shows
   *Clear local animation before running*: click **Clear Local Animation**.
4. No bake is running. Let it finish, or click **Abort**.
5. No solve, transfer or encode is in progress.

When you press **Run** with the button enabled, two refusals name the fix:

- *Geometry has changed since the last transfer. Click "Transfer" to
  re-upload before running.*: press **Transfer**.
- *Parameters have changed since the last transfer. Click "Update Params"
  before running.*: press **Update Params on Remote**.

### "Connection lost" during a run

The error line under the status reads *Connection lost: `<cause>`* (or
*Connection lost.*). The cause is whatever the connection reported. A lost
connection clears the animation loaded in the scene. Reconnect, and if the
solver produced frames, **Fetch All Animation** pulls them. The remote
`server.log` has the server's side.

### "Remote: ..." under the status line

The server returned an error. The panel shows the first line after *Remote:*
with an error icon. The rest of the report, such as a build failure's
traceback, is in the add-on console (**Show Console**) and in `server.log`.

### "Server error: ..."

A fetch, or **Data Receive** in the data transfer test, asked the server for
data and the server answered with an error instead. The text after the colon
is the server's reason. `server.log` in the solver folder on the solver host
has the server's side.

### Status: "Simulation Failed"

The solver stopped instead of finishing. The panel shows *Solver failed:*
followed by a one-line cause, and under it an untranslated detail line with
machine data, such as a CUDA error name, a signal, or a file and line. If
saved states exist, the status reads *Simulation Failed (Resumable)* and
**Resume** can continue from one (see
[Resuming a Run](workflow/sim/simulating.md#resuming-a-run)).

| Cause line | What it means and what to do |
| ---------- | ---------------------------- |
| Intersection detected | The solver's own check found overlapping geometry while stepping. See [N self-intersections](#n-self-intersections--tri-tri--rod-tri). |
| Intersection in the initial configuration | The scene starts with overlapping geometry. See [N self-intersections](#n-self-intersections--tri-tri--rod-tri). |
| Two surfaces start the step already touching or overlapping | Two surfaces begin a step inside their contact offset. Start the scene with a small clearance between them. |
| A pinned vertex is driven into a collider it cannot yield to | A fixed pin's path crosses a floor, wall or sphere, so the pin cannot be satisfied. Change the pin's path, or turn on **Pull** for that pin so it holds softly. |
| Newton solve made no progress (over-constrained configuration) | A typical cause is a prescribed pin driven into geometry that cannot yield. Check the pins and colliders near the failing frame. |
| Continuous collision detection failed | The solver's collision check reported a failure (log line `### ccd failed`). The console report has the details. |
| Linear solver failed to converge | The solver's linear solve did not converge (log line `### cg failed`). The console report has the details. |
| Out of GPU memory | The GPU ran out of memory for this scene. Use a smaller scene or a GPU with more memory. |
| Unrecoverable CUDA runtime or driver error | The detail line names the CUDA error. |
| A GPU kernel ran past the operating system's watchdog timeout | The operating system reset a GPU that also drives a display. Run on a GPU with no display attached, or raise the operating system's timeout. A driver reinstall does not help. |
| This GPU's architecture is not in the solver's device image | The solver was not built for this GPU. Use a solver build that includes it. The detail line names the GPU's compute capability and the architectures the build carries. |
| Solver host panicked | The solver's host code panicked. The console report carries the panic message. |
| Solver stopped on a failed internal check | A solver check over its own state failed, and the detail line carries the check's report. The run cannot be resumed past it without changing the scene. |
| A solver invariant failed on the GPU | A check that runs on the GPU failed. The stdout tail in the console report is the only place near the failed check. |
| The solver process was killed before it could report | The process ended on a fatal signal. The detail line names the signal when it is known. |
| A required library could not be loaded | The system could not load a library the solver needs. Check the library search path, not the scene. |
| The solver exited before it started | The process ended before the solver wrote any status. The stderr tail in the console report is the evidence. |
| Solver exited abnormally without reporting a cause | No cause was recorded. The console report lists the raw facts. |

To read more:

1. Press **Show Console**, on the row with **Update Stat**. The add-on console
   holds the full report, including the tails of the solver's stdout and
   stderr.
2. Press **Open Session Folder**, under that row. It opens the run's `session`
   folder, where the solver's logs and status record are written. It is
   grayed out, with *The session folder is not known for this run*, when the
   solver root cannot be resolved. The path it opens is a path on the machine
   the server runs on, so it reaches real files only when that machine is this
   one.

## Fetch and playback

### "N frames unfetched" in the Solver panel

The solver host has frames that this Blender does not. The panel line reads
*N frames unfetched. Press "Fetch All Animation".* Click **Fetch All
Animation**.

### Render with unfetched frames

A popup titled **Unfetched Frames** warns that the rendered animation may be
incomplete. Fetch first, then render. The popup appears once per render and
does not block it.

### "Unfetched animation frames exist. Fetch all animation frames first."

**Bake Animation** and the export buttons (**Export USD**, **Export Alembic
(ABC)**) refuse while frames are unfetched. Click **Fetch All Animation**,
then try again.

### "Data path: ... does not exist"

The simulated animation files that belong to this `.blend` are missing from
`data/<blend-file-name>/` next to it. The panel also lists the missing files.

- If you renamed or copied the `.blend`, a **Migrate data/`<old>`/ →
  data/`<new>`/** button appears. It copies the old folder to the new name, or
  moves it when you turn off **Keep Copy** in the dialog.
- Otherwise, restore the folder from a backup, or click **Clear Local
  Animation** and run again.

### A rendered animation shows rods or curves in their rest pose

Some strands render in their rest pose while the viewport shows them moving.
Use **Render Animation** under **Debug Options > Render**. It sets the frame
before it renders each one, so the simulated pose is applied first. See
[Render](#render).

## Bake

### "Remove all shape keys except Basis before baking"

Baking adds one shape key per frame and keyframes their values, which would
double-blend with any shape keys the mesh already carries. In **Object Data
Properties**, delete every shape key except `Basis` on the objects the message
lists.

### "'...' is a Hair Curves object and cannot be baked"

Blender keeps no keyframes for hair points, so a solve cannot be turned into
Blender animation on a Hair Curves object. The hair keeps playing from the
add-on's cache. A bake of everything bakes the other objects and lists the
hair it left. See
[What Hair Does Not Do](workflow/scene/hair_curves.md#what-hair-does-not-do).

## MCP server

The **MCP Server** box in the Solver panel holds **Start MCP Server on Local**
and the **Port** field.

### Port already in use

The add-on retries the base port a few times, then tries the nine ports after
it (`9634` to `9642` with the default `9633`) and binds the first free one.
**Start MCP Server on Local** reports the substitution as a warning and writes
the new number into **Port**, so the **MCP Server (Running :port)** header
names the live port. Check it there if an external client points at the base
port.

### "Port conflict: MCP Server: Could not find available port in range 9633-9642 ..."

All ten ports in the range are taken. Find the holder (`lsof -i :9633-9642` on
macOS and Linux, `netstat -ano | findstr ":963 :964"` on Windows) and end it,
or change **Port** to a base port with free ports after it. Use the range the
message names.

### "Port conflict: ..." and "Failed to start MCP server: ..."

Any start error whose text mentions a port is shown as *Port conflict:
`<error>`. Try reloading the addon or using a different port.* Examples are
*MCP Server: Failed to bind to port N: ...* and *MCP Server: Timed out waiting
for bind on port N*. A bind failure also prints `MCP Server error: ...` to
Blender's system console. Other errors, such as an import error, are shown as
*Failed to start MCP server: `<error>`*. *Failed to start MCP server - unknown
error* means the server was not running after the start.

### `run_python_script` returns `"status": "error"`

The snippet you sent raised. Read the `error` field, which holds the exception
text, and the `output` field, which holds what the script printed before it
failed. The reply carries no traceback. To get one, wrap the snippet in
`try`/`except`, add `import traceback` at the top, and call
`print(traceback.format_exc())` so it comes back in `output`.

### `capture_viewport_image`: "No 3D viewport found"

The current Blender screen layout has no 3D Viewport area. Switch to a
workspace that has one, such as **Layout**, **Modeling** or **Sculpting**, and
try again.

## Debug Options

```{figure} images/tour/debug_options.png
:alt: Backend Communicator panel with the Debug Options checkbox highlighted and nine debug blocks expanded below it: Shell Calls, Data Transfer Tests, Options, Console Log Export, GitHub Repo on Remote, GitHub Repo on Local, UUID Migration, Render, and Add-on Local Debug Server
:width: 500px

The **Debug Options** checkbox (highlighted) and the nine blocks it shows.
```

**Debug Options** is the checkbox on the row with **Update Stat** and **Show
Console** in the Backend Communicator panel. It is off by default. When it is
on, nine blocks appear below the panel's usual contents. They are diagnosis
tools, not part of the simulation workflow. A shell command or a server query
reports into the add-on console, which **Show Console** opens. Controls that
need a live connection are grayed out until you connect.

| Block | What it does | Needs |
| ----- | ------------ | ----- |
| [Shell Calls](#shell-calls) | Sends a query to the server, or runs a shell command on the solver host. | A connection; a started server for the query |
| [Data Transfer Tests](#data-transfer-tests) | Uploads random data to the solver folder and checks it. | A started server |
| [Options](#console-max-console-lines-and-console-log-export) | **Max Console Lines**. | Nothing |
| [Console Log Export](#console-max-console-lines-and-console-log-export) | Writes the add-on console to a file. | Nothing |
| [GitHub Repo on Remote](#github-repo-on-remote-and-local) | Runs `git pull` and a build on the solver host; **Open GitHub Link** opens the project page. A development tool. | A connection (not for **Open GitHub Link**) |
| [GitHub Repo on Local](#github-repo-on-remote-and-local) | Runs `git pull` in the add-on's own folder. A development tool. | Nothing |
| [UUID Migration](#uuid-migration) | **Run UUID Migration**. | Nothing |
| [Render](#render) | **Render Animation** and **Stop**. | Nothing |
| [Add-on Local Debug Server](#add-on-local-debug-server) | Starts the add-on's reload and debug port. A development tool. | Nothing |

The three development tools are described at the end of this section.

The buttons in **Shell Calls**, **Data Transfer Tests**, **Console Log
Export** and **GitHub Repo** have MCP tools, which work with **Debug
Options** off. `execute_server_command` is **Exec Command via Server**, with
the `--key value` text in `server_script`. `execute_shell_command` is
**Execute Shell Command on Remote**, with `shell_command` and `use_shell`
(**Run as Shell**). `debug_data_send` is **Data Send**, with `data_size_mb` as
**Data Size (MB)**, and `debug_data_receive` is **Data Receive**.
`git_pull_remote` is **Git Pull**, `compile_project` is **Compile**, and
`git_pull_local` is **Git Pull (Local)**. `delete_log_file` is **Delete Log**:
it takes the path in `log_file_path` and also writes that path into the
**Console Log Export** field, so the console keeps appending to it until you
clear the field with the **X** button. These tools ask for no confirmation;
see [Code Execution Risk: MCP](security.md#code-execution-risk-mcp).

### Shell Calls

Two tools that address the other end of the connection.

**Exec Command via Server** reads the **Args** field as `--key value` pairs
and sends them to the running solver server as one query. The reply is
written to the add-on console between two `------` lines, one `key: value`
per line. It needs a connection and a started server. Use it to ask the server
for something the panel does not show.

**Execute Shell Command on Remote** runs the **Command** field in the solver
folder and prints its stdout and stderr to the console. It needs a connection
but not a running server, so it is the tool to use when the server will not
start: list the virtual environment, run `nvidia-smi`, or read `server.log`
with `tail`. On a Docker type, the command runs inside the container, not on
the Docker host. **Run as Shell** is on by default. It makes pipes,
redirection and `&&` work: the command is wrapped in `/bin/sh -c` on the SSH
and Docker types, and handed to the system shell on the native types.

### Data Transfer Tests

Use these when a **Transfer** stalls, a fetch never finishes, or data arrives
damaged, to find out whether the connection is the cause. A block of random
bytes involves no scene data and no solver, so a round trip that fails here
points at the connection, and one that succeeds at a size close to your scene
points elsewhere.

1. Set **Data Size (MB)** (1 to 256, default 1).
2. Click **Data Send**. It uploads that many megabytes of random bytes to a
   test file in the solver folder.
3. Click **Data Receive**. It downloads the file and compares it byte for
   byte with the copy still in memory. The result is *Data received matches
   test data.* or *Data received does not match test data.*

The progress bar and the throughput readout also give the link's speed.
**Data Receive** stays unavailable until a **Data Send** has run in the same
Blender session. Both need the server running. The test file stays in the
solver folder until you delete it, so remove it after a large test.

:::{admonition} Under the hood
:class: toggle

The test file is named `dummy_data.pickle`.

On the three native types the data is written straight to the file system
instead of through the socket, unless `PPF_FORCE_TCP_TRANSFER=1` is set in the
environment Blender inherited. There the round trip measures a file copy, not
a network.
:::

### Console: Max Console Lines and Console Log Export

The add-on console keeps the last **Max Console Lines** lines (default 60,
from 8 to 10000). Raise it when a **Simulation Failed** report or another
long message scrolls past before you can read it. It limits only what the
console keeps in Blender. A log file receives every line.

**Console Log Export** is a path field. While it holds a path, every line the
add-on writes to its console is also appended to that file, and the parent
folder is created if it does not exist. The folder button opens a file browser
(pre-filled with `log.txt` in your home folder when the field is empty), the
**X** button clears the path and turns file logging off, and **Delete Log**
deletes the file. Use the file when you need the lines after the console has
trimmed them, for example during an unattended run. The file is appended to
and never truncated, so it grows across sessions until you delete it.

### UUID Migration

**Run UUID Migration** fills in the IDs the add-on uses to follow objects
across renames: one for each group, one on every assigned object, the object
and vertex group that every pin refers to, and the objects that every snap or
stitch pair refers to. A rename or a `.001` suffix then does not break the
reference. A group, assigned object, pin or merge pair that carries no ID, for
example one that a script added directly, cannot be followed that way, and this
pass adds the missing IDs.

You rarely need to press it:

- The same pass runs automatically after a file loads whenever something is
  missing. It writes its result to the add-on console, prefixed with
  `[auto-migrate]`.
- **Transfer** refuses a scene with incomplete IDs, with a message that ends
  *Run UUID Migration first.*
- Press the button yourself when the automatic pass did not run or did not
  cover a record, for example one added by a script. It always reports what it
  did.

The result stays under the button until the next run. When something was
migrated, it ends with *Save to persist*: the IDs are scene data, so closing
the `.blend` without saving loses them. The pass writes only IDs that were
empty, so a second run on a migrated scene changes nothing and reads *No
migration needed*. A failure reads *Migration error: ...*.

:::{admonition} Under the hood
:class: toggle

An object carries its ID as a custom property. A pin refers to its vertex group
by a hash of the vertex indices the group holds, so renaming the group does not
break the pin. The IDs the pass adds to a pair include those stored in the
pair's stitch data.
:::

### Render

**Render Animation** renders the scene's frame range one frame at a time, with
a progress bar that names the frame and the percentage done. **Stop** halts it
after the frame in flight finishes, because Blender offers no way to cancel in
the middle of a frame. Use it when a rendered animation of simulated curves or
rods shows some strands in their rest pose while the viewport shows them
moving. It sets each frame before rendering it, so the simulated pose is
applied first.

The files go to the paths Blender's own animation render would use, one file
per frame, from the scene's output settings. If files that match those paths
already exist, a dialog states how many, and they are deleted before the
render starts: cancel the dialog if they matter. When the run ends or is
stopped, the current frame and the output path return to what they were.

### GitHub Repo on Remote and Local

These buttons update and rebuild a solver or the add-on from git. They are
development tools and are not needed to run a simulation.

:::{admonition} Under the hood
:class: toggle

**GitHub Repo on Remote.** **Git Pull** runs `git pull`, and **Compile** runs
`/root/.cargo/bin/cargo build --release`, in the solver folder, by the same
route as **Execute Shell Command on Remote**. Both need a connection and no
other job running. After a pull and a compile, stop and start the server so
that the new binary runs. The `cargo` path is fixed, the place where the
project's container image installs it. On a host that keeps `cargo` elsewhere,
**Compile** fails, so run the build with **Execute Shell Command on Remote**.
**Open GitHub Link** opens the project's public repository page in your
browser and needs no connection.

**GitHub Repo on Local.** **Git Pull (Local)** runs `git pull` inside the
add-on's own folder, so it updates the copy of the add-on that Blender runs,
not the solver. It does something only when the add-on is installed as a git
checkout. For an extension unpacked from a zip, it reports *Local git pull
failed:* followed by git's own error message. After 60 seconds it stops and
reports *Local git pull timed out*.
Follow it with **Reload Add-on Now** to run what you pulled without
restarting Blender.
:::

### Add-on Local Debug Server

**Start** opens a local port (8765 by default) that the **Reload Add-on Now**
and **Full Reload** buttons and the `blender_addon/debug/main.py`
command-line tool use. It is a development tool, and it stays off until you
press **Start** or launch Blender with `blender_addon/launch.sh`. Until it
runs, the reload buttons report *Failed to trigger reload: Reload server is
not running* (*Failed to trigger full reload: ...* for **Full Reload**).

:::{warning}
While this server runs, any process on your machine can send it a JSON packet
that executes arbitrary Python inside Blender. There is no authentication and
no `Origin` check. Start it when you are debugging and stop it when you are
done. See [Debug Server](security.md#debug-server).
:::

:::{admonition} Under the hood
:class: toggle

**Start** binds a TCP socket on `localhost` at **Port**, and the field is
grayed out while the server runs.

- **Reload Add-on Now** reloads the add-on's code from disk without
  restarting Blender.
- **Full Reload** does the same in two steps. Use it when a change to a
  property definition does not show up after a plain reload. If it still does
  not show up, restart Blender, because a reload cannot rebind Blender's
  property definitions in every case.
- After a reload, the server restarts if it was running, on the default port
  8765 and not the one you typed. The command-line tool always uses 8765.

**Command-line tool messages.**

- *Error: MCP server not reachable on localhost:9633.*: the tool got no answer
  in 2 seconds. Start the MCP server from the panel, or run `python
  blender_addon/debug/main.py start-mcp`, which works only while the
  **Add-on Local Debug Server** runs, because it asks that server to start the
  MCP server. If the MCP server runs on a fallback port, pass
  `--mcp-port <port>`.
- *Debug/reload port (TCP 8765): unreachable*, printed by `status`: the debug
  server is not running. Registering the add-on does not start it. Tick
  **Debug Options** and press **Start** under **Add-on Local Debug Server**, or
  launch Blender with `blender_addon/launch.sh`.
- *Reload timed out*: the add-on reports this after 30 seconds without a
  finished reload on Blender's main thread, and *Full reload timed out* after
  60 seconds. The tool itself stops waiting at 45 and 70 seconds. Reload
  errors are printed to Blender's system console.
- A change that does not show up after a reload: run `python
  blender_addon/debug/main.py full-reload`. If that fails too, restart
  Blender.
:::
