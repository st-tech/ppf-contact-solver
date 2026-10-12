(macos-native)=

# 🍎 macOS Native

Use **macOS Native** when Blender and the solver run on the same
Apple-silicon Mac. The add-on starts `ppf-cts-server` as a child process
and reaches it over a loopback TCP socket. No SSH or Docker is involved.

The solver folder is the extracted **macOS Native** distribution from
[GitHub Releases](https://github.com/st-tech/ppf-contact-solver/releases),
or a checkout you built. The distribution carries the Metal and CPU
backends and needs no Python, Homebrew, or Xcode installation.

```{figure} ../images/connections/transport_topologies.svg
:alt: Five stacked block diagrams, one per connection type. The first row, Linux Native / macOS Native, places the Blender add-on and the ppf-cts-server binary on a single workstation with a loopback TCP socket between them, the add-on spawning the server as a child process. The remaining rows cover SSH, Docker, Docker over SSH, and Windows Native.
:width: 820px

macOS Native is the first row: everything on one machine, the add-on
spawning `ppf-cts-server` as a child process (blue solid arrow) and
talking to it over a loopback socket (purple dashed arrow). Linux
Native has the same shape; see [Linux Native](linux.md).
```

## When to Use It

- An Apple-silicon Mac running Blender: click **GPU** and the Metal build
  runs the solve on the same machine.
- Evaluation, learning, and small examples: the Metal and CPU backends are
  slow, and larger scenes need a powerful GPU.
- A Mac with no Metal-capable device: click **CPU**.

To run larger scenes on a remote GPU instead, use [SSH](ssh.md) or
[Docker over SSH](docker_over_ssh.md).

## Setup

1. Set **Type** to **macOS Native**.
2. Set **Solver Path** to the root of the distribution or checkout: the
   folder that has `target/release/ppf-cts-server` in it. Picking a
   subfolder such as `target` or `target/release` also works: the add-on
   searches the parent folders for the solver root and shows the one it
   used. Under the field the panel shows **Solver path valid** (with
   **Using solver root** when it resolved a subfolder), or
   **ppf-cts-server not found**.
3. Click **GPU** (the Metal build) or **CPU** (the portable build) in the
   toggle under **Solver Path**. See
   {ref}`Choosing the build <choosing-the-build>`.
4. Set **Project Name** in the Connection box. It takes letters, digits,
   `.`, `-`, and `_`.
5. Click **Connect**. The add-on resolves the root and refuses one that
   holds no build for the toggle you picked, naming the folder it looked
   in. Connecting does not start the server.
6. Click **Start Server on Remote**. The add-on clears the Gatekeeper
   quarantine mark if the folder needs it, starts `ppf-cts-server`, and
   waits up to 16 seconds for it to answer. If a `ppf-cts-server` from an
   earlier Blender session is still listening on the server port, the
   add-on attaches to it instead of starting a second one.

The server port is the **Docker Port** setting, which the panel draws only
for the three Docker types. To change it, set **Type** to a Docker type,
edit **Docker Port**, then select **macOS Native** again, or set
`docker_port` in a [connection profile](profiles.md). See
[Port Usage at a Glance](index.md#port-usage-at-a-glance) for the default
and the range.

### When Connect Is Unavailable

**Connect** stays unavailable while **Project Name** is empty or holds
anything other than letters, digits, `.`, `-`, and `_`, and while **Solver
Path** is empty or contains a shell special character. Spaces are allowed,
but parentheses are not, so a download folder named
`ppf-contact-solver (1)` is refused. Rename it.
{ref}`When Connect is unavailable <connect-unavailable>` lists the message
the panel shows for each case and the characters that are refused.

## Fields

| Field | Description |
| ----- | ----------- |
| **Solver Path** | The folder that holds `target/release/ppf-cts-server` (the distribution or a checkout you built); the CPU build is under `target/cpu/release/`. A subfolder is accepted and resolved upward to the root. Unlike Windows Native, a `bin/`-only folder is **not** a solver root here. |
| **GPU** / **CPU** toggle | Which build of the solver runs. It is named **Compute Device** in tooltips and refusals. **GPU** is the Metal build. **CPU** is the portable build, which needs no GPU and is substantially slower. Default: **GPU**. Drawn whenever **Solver Path** holds no forbidden character, including while it is empty, so you can see which builds are present; disabled only when the folder holds just the build that is already selected. **Connect** refuses a build the folder lacks by name instead of running the other one. |

No **GPU** dropdown is drawn on this type: the Metal build opens the
system default device and offers no way to name another. See
{ref}`Picking a GPU <gpu-picker>`.

**GPU Backend** is normally not drawn, because a Mac has one accelerator.
If the .blend file carries a saved CUDA or ROCm choice and **Compute Device**
is **GPU**, the row appears with a note such as *This folder has no CUDA
build*, and the choice has no effect on a Mac.

## Troubleshooting

Errors raised by **Start Server on Remote** begin with **Failed to start
server:**. A missing build, a build that does not match the selection, and a
running server of another build are also refused at **Connect**, where the
message appears without that prefix.

- **`ppf-cts-server not found under <root>`**: the folder you picked is
  not a solver root, usually because it is the folder you extracted the
  archive *into* rather than the archive root. When building from source,
  run `cargo build --release` (or `build-mac-native/build.sh`) so the
  solver and the Python extension are built next to the server; a run
  takes all three from the same build folder.
- **`<root> holds the CPU build of the solver and no GPU build`** (or the
  reverse): the folder is right and the toggle is not. Click the build
  that is there, or point **Solver Path** at a folder that has the one you
  want.
- **`N entries under <root> are still marked com.apple.quarantine`**:
  printed to Blender's system console (start Blender from Terminal to see
  it) when the add-on could not clear the download mark, which is what a
  folder owned by another user or on a read-only volume looks like. Move
  the distribution somewhere you own, or clear the mark yourself with
  `xattr -s -d -r com.apple.quarantine "<root>"`.
- **`A solver server is already running on port N, and its runs use the
  build in ...`**: **Connect** (or **Start Server on Remote**) found a
  server it did not launch, running a different build from the one
  selected. See {ref}`Attaching to a running server <attach-mismatch>`.
- **`macOS Native server did not become ready within 16 seconds. Check
  server.log.`**: the process started but did not answer within 16
  seconds. The panel does not show the log; open `server.log` in the solver
  root.
- **`macOS Native server exited with code N before becoming ready. Check
  server.log.`**: the process ended during startup. Its output goes to
  `server.log` in the solver root, not to a console.
- **`Port N is in use`**: another program is listening on the server port
  and does not answer as a `ppf-cts-server`. Click **Force Terminate
  Process**, shown next to the error, to end the listener, or end that
  program yourself. A `ppf-cts-server` left from an earlier session is not
  this error: the add-on attaches to it.

:::{admonition} Under the hood
:class: toggle

**Gatekeeper quarantine**

macOS marks every file extracted from a download, and a marked binary
is stopped when it is *loaded* rather than when it is started, so
clearing the mark on the server alone would move the failure into the
backend dylib or the build worker's first import. Before starting the
server, the add-on clears `com.apple.quarantine` from the whole selected
folder, and only for a packaged distribution, which it recognizes by the
`.ppf-selfcontained` marker the bundler writes. A developer checkout is
never downloaded as a unit and is never marked.

The clearing never blocks the connection and never claims more than it
did: the marked entries are counted again afterwards, so a folder that
could not be written reports what is still marked rather than a success
it did not have. A distribution's own launcher does the same thing at
startup, but this backend never runs that launcher: it starts
`ppf-cts-server` out of the folder directly, which is why the add-on
has to clear the mark too.

**Which Python the build worker runs under**

The interpreter is chosen as described under
{ref}`Linux Native, Under the hood <linux-under-the-hood>`: the
distribution's own `<root>/python/bin/python3`, otherwise the developer
environment, and named in `PPF_CTS_BUILD_PYTHON` with its folder first on
`PATH`. On a Mac the choice cannot be left to `PATH` alone, because macOS
ships Python 3.9 as its system `python3` and the frontend does not parse
under it.

**No library search path is set**

The solver binary carries an `LC_RPATH` and the Metal backend dylib an
`@rpath` install name, so each resolves the other on its own.

**Subprocess environment**

`PYTHONPATH` and `CARGO_TARGET_DIR` are set as described under
{ref}`Linux Native, Under the hood <linux-under-the-hood>`. No
`CUDA_VISIBLE_DEVICES` is set: there is no CUDA here and no device
selection to carry.

**Server log, stop, and file transfer**

These match {ref}`Linux Native <linux-under-the-hood>`: the child's output
goes to `server.log` in the solver root, **Stop Server on Remote** ends the
server with `SIGTERM` and then `SIGKILL` on this connection's port only,
and transfers copy directly on disk.
:::
