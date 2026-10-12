# 🪟 Windows Native

Use **Windows Native** when Blender and the solver run on the same
Windows machine. The add-on starts `ppf-cts-server.exe` from the solver
folder you choose and talks to it over a local TCP socket. No SSH or
Docker is involved.

The solver folder is the extracted Windows distribution from
[GitHub Releases](https://github.com/st-tech/ppf-contact-solver/releases),
or a checkout you built. The x64 distribution carries CUDA, ROCm, and CPU
builds; the Windows on ARM distribution carries the CPU build only. A GPU
build needs the NVIDIA or AMD driver, and the CPU build needs none. The
ROCm build has not been run on AMD hardware, so the project relies on bug
reports from users for it.

```{figure} ../images/connections/windows_topology.svg
:alt: Block diagram with a single Blender workstation (Windows) box that contains the add-on on the left and ppf-cts-server.exe on the right. A blue solid arrow for starting the subprocess with no console window points from the add-on to ppf-cts-server.exe; a purple dashed arrow for a socket to localhost carries server traffic between them.
:width: 760px

Where each piece lives, and how the add-on reaches it. The add-on starts
`ppf-cts-server.exe` as a hidden subprocess with no console window (blue
solid arrow) and talks to it over a loopback TCP socket (purple dashed
arrow). The figure shows the default port, 9090. Everything stays on the
same Windows machine.
```

:::{note}
The Windows distribution is a self-contained folder. Its Python
interpreter, GPU runtimes, solver binaries, and libraries all live inside
it, and nothing is installed into the system. Deleting the folder removes
it. A GPU build still needs the GPU driver.
:::

## When to Use It

- Blender on Windows with a local NVIDIA or AMD GPU: click **GPU** and the
  CUDA or ROCm build runs.
- A Windows machine with no usable GPU, or a Windows on ARM machine: click
  **CPU**. The CPU build is substantially slower.

## Setup

1. Set **Type** to **Windows Native**.
2. Set **Solver Path** to the solver folder. The Fields table below lists
   the layouts the panel accepts. Picking a subfolder such as
   `target\release`, `bin`, or `python` also works: the add-on searches the
   parent folders for the solver root and shows the one it used. Under the
   field the panel shows **Solver path valid** (with **Using solver root**
   when it resolved a subfolder), or **ppf-cts-server.exe not found**.
3. Click **GPU** or **CPU** in the toggle under **Solver Path**. When the
   folder holds more than one GPU build, a **GPU Backend** dropdown also
   appears while the toggle is on **GPU**; the x64 distribution holds CUDA
   and ROCm. See {ref}`Choosing the build <choosing-the-build>`.
4. Set **Project Name** in the Connection box. It takes letters, digits,
   `.`, `-`, and `_`.
5. Click **Connect**. The add-on resolves the solver root and refuses a
   folder with no `ppf-cts-server.exe`, or none for the build you picked.
   Connecting does not start the server.
6. If the machine has several NVIDIA cards, choose one in the **GPU**
   dropdown that appears after **Connect**.
7. Click **Start Server on Remote**. The add-on launches the solver as a
   hidden subprocess on the server port and waits up to 16 seconds for it
   to answer. If a `ppf-cts-server` from an earlier Blender session is
   still listening on that port, the add-on attaches to it instead of
   starting a second one, and an attached server keeps the GPU it started
   on. The choice in the **GPU** dropdown is applied here.

The server port is the **Docker Port** setting, which the panel draws only
for the three Docker types. To change it, set **Type** to a Docker type,
edit **Docker Port**, then select **Windows Native** again, or set
`docker_port` in a [connection profile](profiles.md). See
[Port Usage at a Glance](index.md#port-usage-at-a-glance) for the default
and the range.

```{figure} ../images/connections/windows.png
:alt: Backend Communicator panel with Type set to Windows Native, an empty Solver Path field, a Project Name field, and the Connect button highlighted
:width: 500px

Backend Communicator with **Type** set to **Windows Native** and
**Solver Path** still empty. The native types have no SSH or Docker
fields. **Connect** is highlighted. The **GPU** / **CPU** toggle and the
path status line that the panel draws under **Solver Path** are not part
of this capture.
```

### When Connect Is Unavailable

**Connect** stays unavailable while **Project Name** is empty or holds
anything other than letters, digits, `.`, `-`, and `_`, and while **Solver
Path** is empty or contains a shell special character. Spaces are allowed,
but parentheses are not, so a folder such as `C:\Program Files (x86)\ppf`
is refused. Move or rename the folder.
{ref}`When Connect is unavailable <connect-unavailable>` lists the message
the panel shows for each case and the characters that are refused.

## Fields

| Field | Description |
| ----- | ----------- |
| **Solver Path** | The folder that holds `ppf-cts-server.exe` under `target\release\`, `target\cuda\release\`, `target\rocm\release\`, or `target\cpu\release\` (`bin\` and `bin-cpu\` are also accepted). To start the server it also needs `python\` (distribution) or `build-win-native\python\` (developer build). A subfolder is accepted and resolved upward to the root. |
| **GPU** / **CPU** toggle | Which build of the solver runs. It is named **Compute Device** in tooltips and refusals, and the rows below use that name. **GPU** is the CUDA or ROCm build. **CPU** is the portable build, which needs no GPU and is substantially slower. Default: **GPU**. Drawn whenever **Solver Path** holds no forbidden character, including while it is empty, so you can see which builds are present; disabled only when the folder holds just the build that is already selected. |
| **GPU Backend** | **Automatic**, **CUDA**, or **ROCm**. Drawn only while **Compute Device** is **GPU**, and only when the folder holds more than one GPU build or a saved choice names a build the folder lacks. **Automatic** takes the only GPU build, or, with several, the first whose solver reports a usable device, CUDA before ROCm. |
| **GPU** dropdown | Drawn only after **Connect**, and only while **Compute Device** is **GPU**. Lists the NVIDIA devices `nvidia-smi` reports and applies the choice at **Start Server on Remote**. See {ref}`Picking a GPU <gpu-picker>`. |

The toggle and **GPU Backend** behave the same on every connection type,
and **Connect** refuses a build the folder lacks by name instead of
running another one. See {ref}`Choosing the build <choosing-the-build>`.

## Troubleshooting

Errors raised by **Start Server on Remote** begin with **Failed to start
server:**. A missing build, a build that does not match the selection, and a
running server of another build are also refused at **Connect**, where the
message appears without that prefix.

- **`ppf-cts-server.exe not found under <root>`**: the folder you picked
  is not a solver root, usually because it is the folder you extracted the
  distribution *into*, or the `.zip` itself. Pick the folder that holds
  `target\`, `python\`, and `bin\`. When building from source, run
  `cargo build --release` (or `build-win-native\build.bat`) so the
  solver and the Python extension are built next to the server; a run
  takes all three from the same build folder.
- **`Embedded Python not found in <build dir> or <root>`**: the folder has
  neither `python\python.exe` (distribution) nor
  `build-win-native\python\python.exe` (developer build). Extract the
  whole distribution, or set up the developer tree with
  `build-win-native\warmup.bat`.
- **CUDA DLL load errors**: the distribution carries the CUDA runtime in
  `bin\`, so the CUDA Toolkit does not have to be installed. Check that
  the NVIDIA driver is present and up to date, or click **CPU** to run
  without a GPU.
- **`<root> holds the CPU build of the solver and no GPU build`** (or the
  reverse): the folder is right and the toggle is not. Click the build
  that is there, or point **Solver Path** at a folder that has the one you
  want. The Windows on ARM distribution carries only the CPU build and the
  toggle starts on **GPU**, so the first **Connect** there is refused
  until you click **CPU**.
- **`<root> holds these GPU builds: cuda, and no rocm build`**: **GPU
  Backend** names an accelerator the folder does not carry. Pick one of the
  builds named, or **Automatic**.
- **`No GPU build in this folder has a usable device`**: raised at **Start
  Server on Remote** when **GPU Backend** is **Automatic**, the folder
  holds both the CUDA and the ROCm build, and neither reports a device it
  can run on. The message carries what each backend said. Check the GPU
  driver, or click **CPU**. The add-on never falls back to the CPU build
  on its own.
- **`A solver server is already running on port N, and its runs use the
  build in ...`**: **Connect** (or **Start Server on Remote**) found a
  server it did not launch, running a different build from the one
  selected. See {ref}`Attaching to a running server <attach-mismatch>`.
- **`Port N is in use`**: another program is listening on the server port
  and does not answer as a `ppf-cts-server`. Click **Force Terminate
  Process**, shown next to the error, to end the listener, or end that
  program yourself. A `ppf-cts-server` left from an earlier session is not
  this error: the add-on attaches to it.
- **`Windows Native server did not become ready within 16 seconds. Check
  server.log.`**: the process started but did not answer within 16
  seconds. The panel does not show the log; open `server.log` in the solver
  root.
- **`Windows Native server exited with code N before becoming ready.
  Check server.log.`**: the process ended during startup. Its output goes
  to `server.log` in the solver root, not to a console.

The panel also shows one warning, which is not an error:
**`Path too long: cache files reach N chars (Windows limit 260)`**. It
appears under **Solver Path**, below the toggle, when the longest cache
file path under the folder would reach Windows' 260-character limit and
Windows long paths are off. The length includes **Project Name**. The
warning does not block **Connect** or **Start Server on Remote**, but the
failure would otherwise surface as a cache write error during a later
Transfer. Move the solver to a shorter path or enable Windows long paths.

(windows-under-the-hood)=

:::{admonition} Under the hood
:class: toggle

**Layout auto-detection**

Start Server checks `build-win-native\python\python.exe` first and
`python\python.exe` second, and the one it finds selects the layout.

**Dev layout**

```text
<root>/
  build-win-native/
    python/python.exe
    cuda/bin/*.dll
  target/release/          # ppf-cts-server.exe and other Rust binaries
  target/cuda/release/     # one directory per backend build.bat builds
  crates/ppf-cts-compute/cuda/build/lib/
```

Used when you built the solver from source. `build-win-native\build.bat`
builds each backend into its own `target\<backend>` directory, and a
plain `cargo build --release` builds into `target\release`. The Python
interpreter is `build-win-native\python\python.exe`, `CUDA_PATH` is set
to `build-win-native\cuda`, and the launcher prepends, in order, the
directory the resolved `ppf-cts-server.exe` came out of,
`build-win-native\python`, `target\release`,
`crates\ppf-cts-compute\cuda\build\lib`, and
`build-win-native\cuda\bin` to `PATH`.

**Bundle layout**

```text
<root>/
  bin/                     # each GPU backend's library, the runtime it loads, ffmpeg
  python/python.exe
  target/cuda/release/     # ppf-cts-server.exe, one directory per backend shipped
  target/rocm/release/
  target/cpu/release/
```

Used by a distribution. The Python interpreter is
`root\python\python.exe`, the launcher does not add `CUDA_PATH`, and it
prepends the directory the resolved `ppf-cts-server.exe` came out of,
then `root\python`, `root\bin`, and `root\target\release`. The backend
library each solver loads is in `root\bin`, which is why that
directory is on the search path.

If neither interpreter is present, Start Server fails with:

> Embedded Python not found in \<build\_dir\> or \<root\>

**Subprocess environment**

The subprocess inherits your current environment with a few additions:

- `PATH` is prepended with the layout-specific directories above; the
  existing `PATH` is appended so system tools still work.
- `PYTHONPATH` begins with `<root>` so the build worker spawned by
  `ppf-cts-server.exe` can load `_ppf_cts_py` and the other Python
  modules.
- `CUDA_PATH` is added on the dev layout only.
- `CUDA_VISIBLE_DEVICES` carries the **GPU** dropdown's choice when one is
  set.
- `CARGO_TARGET_DIR` names the build directory, as described below.

**Launch flags and log file**

The solver is launched as `ppf-cts-server.exe --port <port>` from the
solver root, with no visible console window, so nothing appears behind
Blender. Its stdout and stderr are redirected to `server.log` in the
solver root, opened in append mode, and not to a pipe, which nothing in
the add-on would drain.

**Attach to an already-running server**

If a `ppf-cts-server.exe` from a previous Blender session is still
listening on the port, **Connect** and **Start Server on Remote** send a
TCMD probe and reuse that server instead of failing with `Port N is in
use`. The probe checks that the response is valid JSON containing
`protocol_version`, so a listener that is not a solver server (for
example, a Jupyter server on the same port) still surfaces as an error. An
adopted server running a different build from the one the toggle names is
refused rather than adopted, since the selection would otherwise reach
nothing while every solve ran on the other build.

**Stop and disconnect**

**Disconnect** leaves the server running, so the next **Connect**
attaches to it. This is deliberate: the solver subprocess the server
spawns inherits the server's listening socket, so ending only the server
process can leave the port held by that subprocess.

**Stop Server on Remote** ends the process the add-on launched: it
terminates it and kills it if it is still alive after 5 seconds. In
attach mode the add-on holds no process handle, so it finds the process
listening on this connection's port and ends it together with its child
processes. A server on another port is left alone. A listener that is not
a `ppf-cts-server` is cleared with the **Force Terminate Process** button
instead.

**What Start Server does**

Start Server is what spawns `ppf-cts-server.exe` (or attaches to one
already listening on the port), then waits up to 16 seconds for it to
answer a TCMD probe. It is also where the GPU selection is applied, as
`CUDA_VISIBLE_DEVICES` on the child's environment, so moving a running
solver to another device is Stop Server, pick, Start Server. It is
likewise where **GPU Backend**'s **Automatic** runs each candidate
solver's `--probe` when the folder holds more than one GPU build: a spawn
is the one moment a solver is about to run, so it is the moment to ask
which accelerator can run it.

**File transfer fast path**

Transfers take the direct disk-copy path, which is the same on every
native type and is described under
{ref}`Linux Native, Under the hood <linux-under-the-hood>`.

**`CARGO_TARGET_DIR`**

The child is told which build directory its server came out of. A run
takes three things out of one, and the server binary is only the first:
the build worker loads the `_ppf_cts_py` library from the first target
directory it finds, and the session's `SOLVER_PATH` is written from that
same directory. Without naming it, selecting **CPU** would start the CPU
server and run the solve on the GPU solver in `target\release`, with
nothing reporting the split. A server located in `bin\` is not in a
cargo target directory, so an inherited `CARGO_TARGET_DIR` is removed
rather than replaced.
:::
