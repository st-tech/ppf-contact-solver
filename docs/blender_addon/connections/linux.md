# 🐧 Linux Native

Use **Linux Native** when Blender and the solver run on the same Linux
machine. The add-on starts `ppf-cts-server` as a child process and
reaches it over a loopback TCP socket. No SSH, no Docker, and no shell
script in between.

The solver folder is the extracted **Linux Native** distribution from
[GitHub Releases](https://github.com/st-tech/ppf-contact-solver/releases),
or a checkout you built. The x86_64 archive carries the CUDA, ROCm, and
CPU backends together, and the aarch64 one carries CUDA and CPU, so one
folder can hold several builds and you choose which of them runs in the
panel; see
{ref}`Choosing the build <choosing-the-build>`. The distribution needs
glibc 2.28 or newer and no Python, CUDA Toolkit, or ROCm installation. A
GPU build needs the NVIDIA or AMD driver. The ROCm build has not been run
on AMD hardware, so the project relies on bug reports from users for it.

```{figure} ../images/connections/transport_topologies.svg
:alt: Five stacked block diagrams, one per connection type. The first row, Linux Native / macOS Native, places the Blender add-on and the ppf-cts-server binary on a single workstation with a loopback TCP socket between them, the add-on spawning the server as a child process. The remaining rows cover SSH, Docker, Docker over SSH, and Windows Native.
:width: 820px

Linux Native is the first row: everything on one machine, the add-on
spawning `ppf-cts-server` as a child process (blue solid arrow) and
talking to it over a loopback socket (purple dashed arrow). macOS
Native has the same shape; see {ref}`macOS Native <macos-native>`.
```

## When to Use It

- A Linux workstation that has the GPU, so there is no remote host to
  reach and no reason to start a container.
- A machine with no usable GPU: click **CPU**. The CPU build is
  substantially slower.

:::{tip}
To build the solver on your own workstation, use the scripts in
`build-linux-native/`: `warmup.sh` needs no root and writes only into
that folder and the developer environment, and `build.sh` compiles each
backend. Avoid
[`warmup.py`](https://github.com/st-tech/ppf-contact-solver/blob/main/warmup.py)
on a machine you care about. It changes the whole machine (system
packages through `apt`, the CUDA toolkit, shell startup files, a
user-level Python environment, Node.js through nvm, Rust through rustup,
NTP) and does not undo any of it. If you only want to run the solver,
download the distribution or use the [Docker](docker.md) backend.
:::

## Setup

1. Set **Type** to **Linux Native**.
2. Set **Solver Path** to the root of the distribution or checkout: the
   folder that has `target/release/ppf-cts-server` under it, or a
   per-backend `target/cuda/release/`, `target/rocm/release/`, or
   `target/cpu/release/`. Picking a subfolder also works: the add-on
   searches the parent folders for the solver root and shows the one it
   used. Under the field the panel shows **Solver path valid** (with
   **Using solver root** when it resolved a subfolder), or
   **ppf-cts-server not found**.
3. Click **GPU** or **CPU** in the toggle under **Solver Path**. When the
   folder holds more than one GPU build, a **GPU Backend** dropdown also
   appears while the toggle is on **GPU**. See
   {ref}`Choosing the build <choosing-the-build>`.
4. Set **Project Name** in the Connection box. It takes letters, digits,
   `.`, `-`, and `_`.
5. Click **Connect**. The add-on resolves the root and checks that it
   holds a build for the toggle you picked. Connecting does not start the
   server.
6. If the machine has several NVIDIA cards, choose one in the **GPU**
   dropdown that appears after **Connect**.
7. Click **Start Server on Remote**. The add-on starts `ppf-cts-server` on
   the server port and waits up to 16 seconds for it to answer. If a
   `ppf-cts-server` from an earlier Blender session is still listening on
   that port, the add-on attaches to it instead of starting a second one,
   and an attached server keeps the GPU it started on. The choice in the
   **GPU** dropdown is applied here.

The server port is the **Docker Port** setting, which the panel draws only
for the three Docker types. To change it, set **Type** to a Docker type,
edit **Docker Port**, then select **Linux Native** again, or set
`docker_port` in a [connection profile](profiles.md). See
[Port Usage at a Glance](index.md#port-usage-at-a-glance) for the default
and the range.

### When Connect Is Unavailable

**Connect** stays unavailable while **Project Name** is empty or holds
anything other than letters, digits, `.`, `-`, and `_`, and while **Solver
Path** is empty or contains a shell special character. Spaces are allowed,
but parentheses are not, so a folder named `ppf-contact-solver (1)` is
refused. Rename it.
{ref}`When Connect is unavailable <connect-unavailable>` lists the message
the panel shows for each case and the characters that are refused.

## Fields

| Field | Description |
| ----- | ----------- |
| **Solver Path** | The folder that holds `ppf-cts-server` under `target/release/`, `target/cuda/release/`, `target/rocm/release/`, or `target/cpu/release/`: the extracted Linux distribution, or a checkout you built. A subfolder is accepted and resolved upward to the root. The distribution's `bin/` is **not** a server directory (it holds the backend libraries and ffmpeg), so pointing at it resolves upward like any other subfolder. |
| **GPU** / **CPU** toggle | Which build of the solver runs. It is named **Compute Device** in tooltips and refusals, and the rows below use that name. **GPU** is the CUDA or ROCm build. **CPU** is the portable build, which needs no GPU and is substantially slower. Default: **GPU**. Drawn whenever **Solver Path** holds no forbidden character, including while it is empty, so you can see which builds are present; disabled only when the folder holds just the build that is already selected. |
| **GPU Backend** | **Automatic**, **CUDA**, or **ROCm**. Drawn only while **Compute Device** is **GPU**, and only when the folder holds more than one GPU build (the x86_64 distribution holds CUDA and ROCm) or a saved choice names a build the folder lacks. **Automatic** takes the only GPU build, or, with several, the first whose solver reports a usable device, CUDA before ROCm. |
| **GPU** dropdown | Drawn only after **Connect**, and only while **Compute Device** is **GPU**. Lists the NVIDIA devices `nvidia-smi` reports and applies the choice at **Start Server on Remote**. See {ref}`Picking a GPU <gpu-picker>`. |

The toggle and **GPU Backend** behave the same on every connection type,
and **Connect** refuses a build the folder lacks by name instead of
running another one. See {ref}`Choosing the build <choosing-the-build>`.

## Troubleshooting

Errors raised by **Start Server on Remote** begin with **Failed to start
server:**. A missing build, a build that does not match the selection, and a
running server of another build are also refused at **Connect**, where the
message appears without that prefix.

- **`ppf-cts-server not found under <root>`**: the folder you picked is
  not a solver root, usually because it is the folder you extracted the
  archive *into* rather than the archive root. The message names the four
  layouts it accepts. When building from source, run
  `cargo build --release` (or `build-linux-native/build.sh`) so the
  solver and the Python extension are built next to the server; a run
  takes all three from the same build folder.
- **`<root> holds the CPU build of the solver and no GPU build`** (or the
  reverse): the folder is right and the toggle is not. Click the build
  that is there, or point **Solver Path** at a folder that has the one you
  want.
- **`<root> holds these GPU builds: cuda, and no rocm build`**: **GPU
  Backend** names an accelerator the folder does not carry. Pick one of the
  builds named, or **Automatic**.
- **`No GPU build in this folder has a usable device`**: raised at **Start
  Server on Remote** when **GPU Backend** is **Automatic**, the folder
  holds several GPU builds, and none of them reports a device it can run
  on. The message carries what each backend said. Check the GPU driver, or
  click **CPU**. The add-on never falls back to the CPU build on its own.
- **`A solver server is already running on port N, and its runs use the
  build in ...`**: **Connect** (or **Start Server on Remote**) found a
  server it did not launch, running a different build from the one
  selected. See {ref}`Attaching to a running server <attach-mismatch>`.
- **`Linux Native server did not become ready within 16 seconds. Check
  server.log.`**: the process started but did not answer within 16
  seconds. The panel does not show the log; open `server.log` in the solver
  root, where the server's stdout and stderr go instead of a console.
- **`Linux Native server exited with code N before becoming ready. Check
  server.log.`**: the process ended during startup. Read `server.log` in
  the solver root.
- **`Port N is in use`**: another program is listening on the server port
  and does not answer as a `ppf-cts-server`. Click **Force Terminate
  Process**, shown next to the error, to end the listener, or end that
  program yourself. A `ppf-cts-server` left from an earlier session is not
  this error: the add-on attaches to it.

(linux-under-the-hood)=

:::{admonition} Under the hood
:class: toggle

The server log, the stop sequence, and the build worker's interpreter
below also describe {ref}`macOS Native <macos-native>`, which differs only
where its own Under the hood section says so. The file-transfer path is
the same on every native type.

**Layout**

A distribution has the same layout as a checkout: each backend is in
the `target/<backend>/release` directory it was built in, with a
`.ppf-backend` marker beside it naming the backend. `bin/` holds no
server, so it is not an accepted server location.

```text
<root>/
  bin/                     # backend libraries and ffmpeg (no server here)
  python/bin/python3       # the distribution's own interpreter
  target/cuda/release/     # ppf-cts-server, one directory per backend shipped
  target/rocm/release/
  target/cpu/release/
```

**Which Python the build worker runs under**

The interpreter belongs to the folder, not to the machine. A
distribution ships its own at `<root>/python/bin/python3` with the
frontend dependencies already installed into it; a checkout ships none
and uses the developer environment at
`~/.local/share/ppf-cts/venv/bin/python`. The choice is passed to the
server as `PPF_CTS_BUILD_PYTHON` and its folder is put first on `PATH`.
An inherited value of that variable wins, so launching Blender from
inside a distribution's own environment does not produce two answers.
`VIRTUAL_ENV` is set alongside it when the interpreter really is a venv
(decided by `pyvenv.cfg`, not by the path) and cleared when it is not, so
a venv that happened to be active in the shell that started Blender
cannot contradict the interpreter actually chosen.

**No library search path is set**

Every binary shipped here finds its backend library through its own
`RPATH`, which the loader searches before `LD_LIBRARY_PATH`. The
distribution is built with that `RPATH` so it does not depend on what a
developer's shell exports, and the add-on adds no search path at launch.

**Subprocess environment**

- `PYTHONPATH` begins with `<root>` so the build worker can import the
  project's Python modules.
- `CARGO_TARGET_DIR` names the target directory the resolved server
  came out of, so the build worker's library and the session's
  `SOLVER_PATH` come from the same build as the server. Without it, a
  CPU server could drive a solve that ran the GPU solver out of
  `target/release`, with nothing reporting the split.
- `CUDA_VISIBLE_DEVICES` carries the {ref}`GPU dropdown <gpu-picker>`'s
  choice when one is set. **Automatic** sets none, so a value inherited
  from Blender's own environment passes through.

**Server log file**

The child's stdout and stderr are redirected to `server.log` in the
solver root, opened in append mode. The log is a real file rather than a
pipe because nothing in the add-on drains a pipe: once its buffer is full,
every further write blocks and the server accepts connections but never
answers.

**Stop and disconnect**

**Disconnect** leaves the server running, so the next **Connect**
attaches to it instead of starting a second server on the same port.
**Stop Server on Remote** ends it: the process the add-on started is
terminated, and killed if it is still alive after 5 seconds; an adopted
one is found by this connection's port, sent `SIGTERM`, and sent
`SIGKILL` if it survives. The kill is limited to this connection's port,
so a second `ppf-cts-server` on another port is left alone.

**File transfer fast path**

On every native type, transfers copy directly on disk instead of going
through the solver TCP socket. The panel shows no bandwidth figure while
such a transfer runs. Set `PPF_FORCE_TCP_TRANSFER=1` in the environment
to force the socket path instead.
:::
