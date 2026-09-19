# 🪟 Windows Native

The solver runs directly as a Windows subprocess. The add-on launches
the `ppf-cts-server.exe` binary alongside Blender and talks to it over a
local TCP socket. A bundled Python interpreter and (optionally) a
bundled CUDA runtime ship next to the binary, since the Rust server
spawns a Python build worker that loads the `_ppf_cts_py` PyO3 module.
No SSH or Docker is involved.

```{figure} ../images/connections/windows_topology.svg
:alt: Block diagram with a single Blender workstation (Windows) box that contains the add-on on the left and ppf-cts-server.exe on the right. A blue solid arrow labeled CreateProcess (no console window) points from the add-on to ppf-cts-server.exe; a purple dashed arrow labeled socket to localhost:9090 carries server traffic between them.
:width: 760px

Where each piece lives, and how the add-on reaches it. The add-on
launches `ppf-cts-server.exe` as a hidden subprocess via `CreateProcess`
(blue solid arrow) and then talks to it over a loopback TCP socket
(purple dashed arrow). Everything stays on the same Windows machine: no
SSH and no Docker.
```

:::{note}
Unlike the [Local](local.md) (Linux) backend, the Windows Native build ships
as a self-contained tree: the embedded Python interpreter, CUDA runtime, Rust
solver binaries, and all third-party shared libraries live entirely under the
install root. Nothing is installed into the system, no `apt` packages, no
shell rc edits, no global services. Uninstalling is just deleting the
bundled directory; there is no contamination of the host to clean up.
:::

## When to Use It

- User workstations running Blender on Windows with a local NVIDIA or
  AMD GPU, or with no usable GPU at all, where the CPU build runs
  instead.
- Bundled deployments that ship the solver next to the add-on.
- Reproducible test rigs where you want the exact shipped Python +
  CUDA, not whatever the system has.

## Setup

1. Set **Type** to `Windows Native`.
2. Set **Solver Path** to the root of your solver install. This is
   the directory that contains `ppf-cts-server.exe`, in any of the
   layouts a build produces: `target\release\`, a per-backend
   `target\cuda\release\` / `target\rocm\release\`,
   `target\cpu\release\`, or `bin\`. It also needs either a
   `python\` subfolder (shipped distribution) or a
   `build-win-native\python\` subfolder (developer build). Picking a
   subfolder such as `target\release`, `bin`, or `python` is fine: the
   add-on walks up to the real root and names the one it used.
3. Click **Connect**. The add-on resolves the solver root and refuses
   one with no `ppf-cts-server.exe` under it. Connecting does not start
   the server.
4. Click **Start Server on Remote**. The add-on picks up the right Python
   runtime and launches the solver as a hidden subprocess on the
   configured port (`9090` by default). If a `ppf-cts-server` from a
   previous Blender session is still listening on that port, the add-on
   attaches to it instead of launching a second one. The GPU picker is
   filled in over the connection, so this step is also where the
   selected device is applied.
   If the path is deep enough that the cache files the pipeline writes
   would exceed Windows' 260-character limit, the panel warns under
   **Solver Path** with *Path too long: cache files reach N chars
   (Windows limit 260)*. The warning does not block **Connect** or
   **Start Server on Remote**; the failure would otherwise surface as a
   cache write error during a later Transfer. Move the solver to a
   shorter path or enable Windows long paths.

```{figure} ../images/connections/windows.png
:alt: Backend Communicator panel in Windows Native mode
:width: 500px

Backend Communicator with **Type** set to `Windows Native`.
Only **Solver Path** and **Project Name** appear, with no SSH or Docker
fields. **Connect** is highlighted.
```

## Fields

| Field | Description |
| ----- | ----------- |
| Solver Path | Root directory containing `ppf-cts-server.exe` (under `target\release\`, a per-backend `target\cuda\release\` or `target\rocm\release\`, `target\cpu\release\`, or `bin\`) plus either `python\` (shipped distribution) or `build-win-native\python\` (dev). A subfolder of it is accepted and resolved upward to the real root. |
| Compute Device | `GPU` or `CPU`, which build of the solver to run. Always drawn, so you can see which builds are there, and disabled only when the one build present is the one already selected. `CPU` needs no GPU and is substantially slower. **Connect** refuses a device the folder has no build for by name rather than running the other one. |
| GPU Backend | `Automatic`, `CUDA`, or `ROCm`, drawn only when the folder holds more than one GPU build (a Windows x64 distribution carries CUDA and ROCm together) or when a saved choice names a build the folder does not hold. `Automatic` takes the only GPU build present, or, where there are several, the first whose solver reports a usable device, CUDA before ROCm. |

## Troubleshooting

- **`ppf-cts-server.exe not found under <root>`** - the selection is not
  inside a solver root at all. A subfolder (`target\release`, `bin`,
  `python`) is resolved upward automatically, so this means the folder
  you picked is the one you extracted the bundle *into* rather than the
  bundle root, or you picked the `.zip`. If you are building from
  source, run `cargo build --release -p ppf-cts-server` first.
- **`Embedded Python not found ...`** - the add-on could not find a
  Python runtime under the root. Either rebuild the dev tree, or
  download and unpack the bundle zip.
- **CUDA DLL load errors** - the shipped distribution carries the CUDA
  runtime in `bin\`, so the CUDA toolkit does not have to be
  installed. Check that the NVIDIA driver is present and up to date,
  or set **Compute Device** to `CPU` to run without a GPU.
- **`Port N is in use`** - something is already bound to the configured
  server port and it is not a `ppf-cts-server` the add-on recognizes.
  Use the **Force Terminate Process** button shown next to the error to
  kill the listener. The port is the shared port property, which this
  mode does not draw, so changing it instead means setting it from a
  Docker mode or through a profile's `docker_port` key. The add-on
  auto-attaches to an already-running `ppf-cts-server`, so this error
  means the listener is a different process.
- **Server output not visible** - stdout and stderr go to `server.log`
  in the solver root, not to a console. Open that file when diagnosing
  startup failures.

(windows-under-the-hood)=

:::{admonition} Under the hood
:class: toggle

**Layout auto-detection**

Start Server picks one of two layouts by looking for `python.exe`:

#### Dev layout

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
plain `cargo build --release -p ppf-cts-server` puts one in
`target\release`. The Python interpreter is
`build-win-native\python\python.exe`, `CUDA_PATH` is set to
`build-win-native\cuda`, and the launcher prepends, in order, the
directory the resolved `ppf-cts-server.exe` came out of,
`build-win-native\python`, `target\release`,
`crates\ppf-cts-compute\cuda\build\lib`, and
`build-win-native\cuda\bin` to `PATH`.

#### Bundle layout

```text
<root>/
  bin/                     # each GPU backend's library, the runtime it loads, ffmpeg
  python/python.exe
  target/cuda/release/     # ppf-cts-server.exe, one directory per backend shipped
  target/cpu/release/
```

Used by a shipped distribution. The Python interpreter is
`root\python\python.exe`, `CUDA_PATH` is not set, and the launcher
prepends the directory the resolved `ppf-cts-server.exe` came out of,
then `root\python`, `root\bin`, and `root\target\release`. The backend
library each solver loads sits in `root\bin`, which is why that
directory is on the search path.

If neither interpreter is present, Start Server fails with:

> Embedded Python not found in \<build\_dir\> or \<root\>

**Subprocess environment**

The subprocess inherits your current environment with a few additions:

- `PATH` is prepended with the layout-specific directories above; the
  existing `PATH` is appended so system tools still work.
- `PYTHONPATH` begins with `<root>` so the build worker spawned by
  `ppf-cts-server.exe` can import `_ppf_cts_py` and other Python modules.
- `CUDA_PATH` is added on the dev layout only.

**Launch flags**

The solver is launched as `ppf-cts-server.exe --port <port>` with no
visible console window, so nothing appears behind Blender.

**Server log file**

The subprocess's stdout and stderr are redirected to `server.log` in the
solver root, opened in append-binary mode. Writing to a real file rather
than `subprocess.PIPE` avoids a Windows-specific wedge: the OS pipe
buffer is small, and once the server's log4rs console appender fills
it, every subsequent write blocks the tokio worker that emitted it,
which eventually freezes the runtime. Redirecting to a file removes the
back-pressure path entirely. The same log is what the panel tails when
the readiness wait times out.

**Attach to an already-running server**

If a `ppf-cts-server.exe` from a previous Blender session is still listening
on the port, **Start Server on Remote** sends a TCMD probe and reuses that
server instead of failing with `Port N is in use`. The probe checks that the
response is valid JSON containing `protocol_version`, so a non-server
listener (for example a notebook server parked on the port) still surfaces
as an error. In attach mode the add-on holds no process handle, so **Stop
Server on Remote** falls back to
`taskkill /F /IM ppf-cts-server.exe`, which stops the adopted server
(along with any other instance). A foreign listener is cleared with the
**Force Terminate Process** button instead.

**Shutdown**

On disconnect (or **Stop Server on Remote**), the add-on asks the subprocess
to terminate and waits up to 5 seconds; if it is still alive, it is killed.
The Unix `pkill -f ppf-cts-server` path is not used; the backend holds the
Windows process handle directly.

**What Start Server does**

Start Server is what spawns `ppf-cts-server.exe` (or attaches to one
already listening on the port), then waits up to 16 seconds for it to
answer a TCMD probe. It is also where the GPU selection is applied, as
`CUDA_VISIBLE_DEVICES` on the child's environment, so moving a running
solver to another device is Stop Server, pick, Start Server.
:::
