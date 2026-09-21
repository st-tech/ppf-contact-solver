(macos-native)=

# 🍎 macOS Native

Blender and the solver both run on the same Apple-silicon Mac, and the
add-on starts the solver itself. `ppf-cts-server` is launched as a child
process and reached over a loopback TCP socket; no SSH and no Docker.

The root you point it at is either the extracted **macOS Native**
distribution from
[GitHub Releases](https://github.com/st-tech/ppf-contact-solver/releases),
which carries the Metal and CPU backends and needs no Python, Homebrew,
or Xcode installation, or a checkout you built.

```{figure} ../images/connections/transport_topologies.svg
:alt: Five stacked block diagrams, one per connection type. The first row, Linux Native / macOS Native, places the Blender add-on and the ppf-cts-server binary on a single workstation with a loopback TCP socket between them, the add-on spawning the server as a child process. The remaining rows cover SSH, Docker, Docker over SSH, and Windows Native.
:width: 820px

macOS Native is the first row: everything on one machine, the add-on
spawning `ppf-cts-server` as a child process (blue solid arrow) and
talking to it over a loopback socket (purple dashed arrow). Linux
Native has the same shape; see [Linux Native](linux.md).
```

:::{note}
This type replaces the retired **Local** connection on macOS. A
`.blend` or a connection profile that still names `Local` is moved onto
it automatically, carrying its path across; see
{ref}`The retired Local type <retired-local>`.
:::

## When to Use It

- An Apple-silicon Mac running Blender, where the Metal build can run
  the solve on the same machine.
- Evaluation, learning, and small examples. Metal trails a modern
  discrete NVIDIA or AMD GPU by a wide margin; mid to large scenes
  still want one of those, reached over [SSH](ssh.md) or
  [Docker over SSH](docker_over_ssh.md).
- A Mac with no Metal-capable device available, where the CPU build
  runs instead.

## Setup

1. Set **Type** to `macOS Native`.
2. Set **Solver Path** to the root of the distribution or checkout --
   the directory that has `target/release/ppf-cts-server` in it.
   Picking a subfolder such as `target` or `target/release` is fine:
   the add-on walks up to the real root and names the one it used on
   the line below the field.
3. Set **Compute Device**: `GPU` runs the Metal build and `CPU` the
   portable one. See {ref}`Choosing the build <choosing-the-build>`.
4. Set **Project Name** on the main panel.
5. Click **Connect**. This resolves the root and refuses one that holds
   no build for the device you picked, naming the folder it looked in.
   It does not start the server.
6. Click **Start Server on Remote**. The add-on clears the Gatekeeper
   quarantine mark if the folder needs it, spawns `ppf-cts-server`, and
   waits up to **16 seconds** for it to answer the solver protocol. If
   a `ppf-cts-server` from a previous Blender session is still
   listening on the port, the add-on attaches to it instead of
   launching a second one.

## Fields

| Field | Description |
| ----- | ----------- |
| Solver Path | Root directory holding `target/release/ppf-cts-server`: the extracted macOS distribution, or a repo checkout you built. A subfolder of it is accepted and resolved upward to the real root. Unlike Windows Native, a `bin/`-only folder is **not** a solver root here. |
| Compute Device | `GPU` (the Metal build) or `CPU` (the portable one, which needs no GPU and is substantially slower). Always drawn, so you can see which builds are there, and disabled only when the one build present is the one already selected. **Connect** refuses a device the folder has no build for by name rather than running the other one. |

No **GPU Backend** row is drawn on this type: an Apple-silicon Mac has
one accelerator, so a CUDA-or-ROCm choice could never mean anything
here. No **GPU** picker is drawn either -- the Metal build opens the
system default device and offers no way to name another, so there is
nothing to pick. See {ref}`Picking a GPU <gpu-picker>`.

The panel does not draw a server port field on this type -- the port
field is drawn only for the Docker-family types -- so the port used
here is whatever the shared port property currently holds, `9090` by
default. Set it from a Docker mode and switch back, or give an entry
its own with a [connection profile](profiles.md)'s `docker_port` key.

## Troubleshooting

- **`ppf-cts-server not found under <root>`** -- the selection is not
  inside a solver root. A subfolder is resolved upward automatically,
  so this usually means the folder you picked is the one you extracted
  the archive *into* rather than the archive root. If you are building
  from source, run `cargo build --release -p ppf-cts-server` first.
- **`<root> holds the CPU build of the solver and no GPU build`** (or
  the reverse) -- the folder is right and **Compute Device** is not.
  Set it to the build that is there, or point **Solver Path** at a
  folder that has the one you want.
- **`N entries under <root> are still marked com.apple.quarantine`** --
  printed as a warning when the add-on could not clear the download
  mark, which is what a folder owned by another user or on a read-only
  volume looks like. Move the distribution somewhere you own, or clear
  it yourself with
  `xattr -s -d -r com.apple.quarantine "<root>"`.
- **`A solver server is already running on port N, and its runs use the
  build in ...`** -- **Start Server on Remote** found a server it did
  not launch, running a different build from the one selected. See
  {ref}`Attaching to a running server <attach-mismatch>`.
- **Server startup timed out.** -- the solver launched but did not
  answer within 16 seconds. Check `server.log` in the solver root; the
  panel prints its last 20 lines when the timeout fires.
- **`Port N is in use`** -- something other than a `ppf-cts-server` the
  add-on recognizes is bound to the port. Use the **Force Terminate
  Process** button shown next to the error.

:::{admonition} Under the hood
:class: toggle

**Gatekeeper quarantine**

macOS marks every file extracted from a download, and a marked binary
is stopped when it is *loaded* rather than when it is started, so
clearing the mark on the server alone would move the failure into the
backend dylib or the build worker's first import. Before spawning, the
add-on clears `com.apple.quarantine` from the whole selected folder,
and only for a packaged distribution, which it recognizes by the
`.ppf-selfcontained` marker the bundler writes. A developer checkout is
never downloaded as a unit and is never marked.

The clearing never blocks the connection and never claims more than it
did: the marked entries are counted again afterwards, so a folder that
could not be written reports what is still marked rather than a success
it did not have. A distribution's own launcher does the same thing at
startup, but this backend never runs that launcher -- it spawns
`ppf-cts-server` out of the folder directly -- which is why the add-on
has to do it too.

**Which Python the build worker runs under**

The interpreter belongs to the folder, not to the machine. A
distribution ships its own at `<root>/python/bin/python3` with the
frontend dependencies already installed into it; a checkout ships none
and uses the developer environment at
`~/.local/share/ppf-cts/venv/bin/python`. It is named explicitly in
`PPF_CTS_BUILD_PYTHON` rather than left to `PATH`, because macOS ships
Python 3.9 as its system `python3` and the frontend does not parse
under it. An inherited `PPF_CTS_BUILD_PYTHON` wins, and `VIRTUAL_ENV`
is set alongside it only when the interpreter really is a venv (decided
by `pyvenv.cfg`, not by the path), so a venv that happened to be active
in the shell that started Blender cannot contradict the interpreter
actually chosen.

**No library search path is set**

The solver binary carries an `LC_RPATH` and the Metal backend dylib an
`@rpath` install name, so each resolves the other on its own.

**Subprocess environment**

- `PYTHONPATH` begins with `<root>` so the build worker can import the
  project's Python modules.
- `CARGO_TARGET_DIR` names the target directory the resolved server
  came out of, so the build worker's cdylib and the session's
  `SOLVER_PATH` come from the same build as the server.
- No `CUDA_VISIBLE_DEVICES`: there is no CUDA here and no device
  selection to carry.

**Server log file**

The child's stdout and stderr are redirected to `server.log` in the
solver root, opened in append-binary mode, for the same reason the
other natives do it: nothing in the add-on drains a pipe, and a full
pipe buffer wedges the server's async runtime.

**Stop and disconnect**

Disconnect leaves the server running on purpose, so the next Connect
attaches to it rather than fighting for the port. **Stop Server on
Remote** ends it, scoped to this connection's port rather than to the
binary's name.

**File transfer fast path**

Transfers copy directly on disk instead of going through the solver TCP
socket, so the panel shows no bandwidth figure while one runs. Set
`PPF_FORCE_TCP_TRANSFER=1` in the environment to force the socket path.
:::
