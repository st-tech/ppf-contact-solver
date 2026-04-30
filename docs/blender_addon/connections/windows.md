# 🪟 Windows Native

The solver runs directly as a Windows subprocess using a bundled Python
interpreter and (optionally) a bundled CUDA runtime. No SSH or Docker is
involved -- the add-on launches the solver alongside Blender and talks to
it over a local TCP socket.

```{figure} ../images/connections/windows_topology.svg
:alt: Block diagram with a single Blender workstation (Windows) box that contains the add-on on the left and server.py running under python.exe on the right. A blue solid arrow labelled CreateProcess (no console window) points from the add-on to server.py; a purple dashed arrow labelled socket to localhost:9090 carries server traffic between them.
:width: 760px

Where each piece lives, and how the add-on reaches it. The add-on
launches `server.py` as a hidden subprocess via `CreateProcess` (blue
solid arrow) and then talks to it over a loopback TCP socket (purple
dashed arrow). Everything stays on the same Windows machine -- no SSH
and no Docker.
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

- User workstations running Blender on Windows with a local NVIDIA GPU.
- Bundled deployments that ship the solver next to the add-on.
- Reproducible test rigs where you want the exact shipped Python +
  CUDA, not whatever the system has.

## Setup

1. Set **Server Type** to `Windows Native`.
2. Set **Solver Path** to the root of your solver install. This is
   the directory that contains `server.py` plus either a shipped
   `python\` subfolder (redistributable bundle) or a
   `build-win-native\python\` subfolder (developer build).
3. Click **Connect**. The add-on verifies `server.py` is where it
   should be, picks up the right Python runtime, and launches the
   solver as a hidden subprocess on port `9090`.
4. The server is launched as part of the connect step, so once
   **Connect** reports success the server is already running. Pressing
   **Start Server** afterwards is a no-op on this backend.

```{figure} ../images/connections/windows.png
:alt: Backend Communicator panel in Windows Native mode
:width: 500px

Backend Communicator with **Server Type** set to `Windows Native`.
Only **Solver Path** and **Project Name** appear, with no SSH or Docker
fields. **Connect** is highlighted.
```

## Fields

| Field | Description |
| ----- | ----------- |
| Solver Path | Root directory containing `server.py` plus either `python\` (bundle) or `build-win-native\python\` (dev). |

## Troubleshooting

- **`server.py not found: <root>\server.py`** - the root points at the
  wrong directory. It must be the solver checkout root, not the
  `python\` subdirectory.
- **`Embedded Python not found ...`** - the add-on could not find a
  Python runtime under the root. Either rebuild the dev tree, or
  download and unpack the bundle zip.
- **CUDA DLL load errors** - on the shipped bundle, the solver relies
  on the system CUDA runtime. Install a matching CUDA version, or
  switch to the developer build which ships its own CUDA.

(windows-under-the-hood)=

:::{admonition} Under the hood
:class: toggle

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

Used when you built the solver from source. The Python interpreter is
`build-win-native\python\python.exe`, `CUDA_PATH` is set to
`build-win-native\cuda`, and the launcher prepends, in order,
`build-win-native\python`, `target\release`,
`src\cpp\build\lib`, and `build-win-native\cuda\bin` to `PATH`.

#### Bundle layout

```text
<root>/
  server.py
  python/python.exe
  bin/                     # native shared libraries
  target/release/
```

Used by a shipped redistributable. The Python interpreter is
`root\python\python.exe`, `CUDA_PATH` is not set (CUDA is expected on
the system `PATH`), and the launcher prepends `root\python`,
`root\bin`, and `root\target\release`.

If neither interpreter is present, connect fails with:

> Embedded Python not found in \<build\_dir\> or \<root\>

**Subprocess environment**

The subprocess inherits your current environment with a few additions:

- `PATH` is prepended with the layout-specific directories above; the
  existing `PATH` is appended so system tools still work.
- `PYTHONPATH` begins with `<root>` so `server.py` can import its own
  modules.
- `CUDA_PATH` is added on the dev layout only.

**Launch flags**

The solver is launched as `python.exe server.py --port <port>` with no
visible console window, so nothing appears behind Blender.

**Shutdown**

On disconnect (or **Stop Server**), the add-on asks the subprocess to
terminate and waits up to 5 seconds; if it is still alive, it is
killed. The Unix `pkill -f server.py` path is not used; the backend
holds the Windows process handle directly.

**Why Start Server is a no-op**

The subprocess is started as part of the Connect step, so by the time
Connect reports success the server is already running. Pressing
**Start Server** afterwards has nothing to do.
:::
