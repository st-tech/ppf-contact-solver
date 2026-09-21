# Debug test rig

Automated, parallelizable scenarios that exercise the addon's
production state machine end-to-end against the **real** production
pipeline:

- Real ``frontend`` Python module loaded in-process by the server
  (``populate``, ``make``, session export). ``frontend`` imports the
  ``_ppf_cts_py`` PyO3 module from ``crates/ppf-cts-py``; the addon
  itself does not import it.
- Real Rust solver binary at ``target/release/ppf-cts-server`` (built
  from ``crates/ppf-cts-server``), going through the real scene loader,
  the real ``Constraint`` build, the real per-frame loop, and writing
  ``vert_*.bin`` in the real wire format from ``crates/ppf-cts-formats``.

## EVERY SCENARIO NAMES THE BACKEND IT WAS RUN AGAINST

There is no default backend and no scenario inherits one. ``--backend`` is
REQUIRED, every scenario declares ``BACKENDS = ("real",)``, and a scenario that
declares nothing is refused rather than defaulted: a default is how a scenario
acquires a claim its author never made.

``real`` means a backend that computes real physics, which is CUDA on a CUDA
host, Metal on macOS, or the Rust CPU backend
(``cargo build --release --features cpu``, about 30x the wall clock). The rig
drives whichever one the tree was built for.

Two refusals follow from that, both deliberate:

- ``--backend`` is REQUIRED and has no default, because a default would label
  every invocation that omitted it as having targeted something it did not.
- Naming an unrunnable scenario on the command line refuses the run.
  Explicit names bypass the selection filter, so this is the only place
  that case can be caught.

## Quick start

```sh
# Build the server the rig spawns, plus a solver for it to launch.
# On a CUDA host, plain `cargo build --release` gives the real backend
# and is what `--backend real` needs.
cargo build --release
cargo build --release -p ppf-cts-server

# Install Blender-side deps (one-time).
./install-blender-addon.sh
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy scipy tqdm psutil tomli ipython pillow pythreejs pytetwild tetgen

# All scenarios this backend can run (and a named report of those it cannot).
python3.12 blender_addon/debug/main.py runtests --backend real

# One scenario.
python3.12 blender_addon/debug/main.py runtests bl_connect_linux_native \
    --backend real

# Stress: 3 repeats of everything at parallel=4.
python3.12 blender_addon/debug/main.py runtests --backend real \
    --parallel 4 --repeat 3 --report run.json
```

The rig is all-green on macOS, Linux, and Windows. Each host runs the
subset its ``PLATFORMS`` and ``BACKENDS`` gates admit. ``runtests
--list`` prints that set for the host you are on and is the only source
of truth for its size, so count its output rather than looking for a
figure here. Chain
scenarios can flake at ``--parallel 4`` (see "Parallel mode caveats"
below); drop to ``--parallel 2`` if you need stable chain results.

### Displays on Linux

Blender is launched WITHOUT ``--background`` (the bootstrap needs the
event loop to tick its timers), so it needs an X display. The rig
starts and owns one: ``ensure_display()`` in ``blender_harness.py``
brings up an Xvfb on the first free display from ``:100``, publishes it
through ``DISPLAY`` before the worker pool starts, and tears it down at
exit. Nothing has to be wrapped in ``xvfb-run``, and a host with no X
at all runs the suite unchanged.

That happens even when a desktop session is available, which is
deliberate: rendering onto the developer's desktop puts a window per
worker in front of whatever they are doing, and it makes the display
size a property of the machine, so the same run behaves differently on
a laptop and a CI runner.

- ``PPF_BLENDER_DISPLAY=inherit`` uses the ambient ``DISPLAY`` instead.
  Reach for it when you want to WATCH the scenarios drive the UI.
- ``PPF_BLENDER_WINDOW`` sets the window size, as ``WxH``, ``X,Y,W,H``,
  or ``off`` to let Blender size its own. The default is 800x600, which
  keeps the per-worker framebuffer small while still laying out the
  viewport and its sidebar.
- ``PPF_BLENDER_BIN`` picks the binary. Otherwise the search is per-OS:
  the ``.app`` bundle first on macOS, ``blender`` on PATH first on
  Linux, with ``/opt/blender-*`` as the fallback.

## What runs where

| Layer                          | What's faked                          | What's real                                  |
| ------------------------------ | ------------------------------------- | -------------------------------------------- |
| ``blender_addon/debug/probe.py`` | nothing (it's the observer)         | Hooks Blender's real handler tables, samples real ``engine.state`` |
| ``blender_addon/debug/orchestrator.py`` | nothing (host-side)            | Spawns the real ``ppf-cts-server`` binary and real Blender |

NOTHING IS FAKED. The server is the real ``ppf-cts-server`` binary and the
solver it launches is a real one; a GPU-less host is served by building the CPU
backend, whose ``check_gpu`` answers for itself rather than being patched out
from Python.

## Backend selection (``--backend``)

``runtests --backend <name>`` gates the scenario set on a per-scenario
``BACKENDS`` tag, alongside the ``PLATFORMS`` gate. The flag is required:
there is no default backend.

- ``BACKENDS`` is unset => REFUSED. Every scenario must name the backend it
  was RUN against; there is no default to inherit.
- ``BACKENDS = ("real",)`` => runs against a backend that computes real
  physics. That is every scenario in the registry.

Every job in ``.github/workflows/blender.yml`` runs
``runtests --backend real``.
When you add a real-capable scenario, remember the connection path:
``dh.connect(...)`` picks the NATIVE connection of the machine it runs on,
LINUX_NATIVE on Linux, MAC_NATIVE on macOS, WIN_NATIVE on Windows, so a
cross-platform real scenario uses it and carries no platform branch of its own.
See ``bl_real_solid_smoke`` for the pattern. A scenario that exists to exercise
ONE platform's native connection names that one instead
(``dh.connect_win_native``, ``dh.connect_mac_native``,
``dh.connect_linux_native``) and gates itself with ``PLATFORMS``;
``bl_mac_native_real_solve`` is that pattern.

### The device is asked of the tree, and the rig owns the server

``dh.connect_native`` resolves the Compute Device from the tree it is pointed
at rather than assuming one: ``resolve_native_device`` calls the same
``core.connection.native_resolvers`` entry the panel's own device row calls, so
a leg that built only the CPU backend connects as CPU and a CUDA leg connects
as GPU. Do not hard-code a device. ``native_device`` defaults to GPU and a
native connection REFUSES a root holding only the other device's build, by
name, so a scenario pinned to GPU is refused on every connection on a CPU-only
leg, and one pinned to CPU is refused on the CUDA legs. Pass ``device=`` only
where the refusal itself is what the scenario asserts. A driver that builds no
``DriverHelpers`` gets the same three answers from ``platform_native()``,
``select_platform_native()`` and ``connect_platform_native()`` in
``scenarios/_driver_lib.py``.

**WHERE EACH NATIVE IS ACTUALLY EXERCISED IN CI, which is not one leg per
platform.** The Linux leg runs the whole set on a disposable AWS GPU instance,
so LINUX_NATIVE is covered there. The Windows leg does the same for WIN_NATIVE.
The macOS leg is a GitHub-hosted runner, and that runner is VIRTUALIZED WITH NO
METAL ACCESS, so it cannot run an accelerated solver of its own at all; no macOS
instance can be launched on AWS either. That is why the leg runs only
``bl_ssh_remote_solve`` and ``bl_ssh_remote_solid``, driving a remote Linux GPU
box over SSH, and why MAC_NATIVE is NOT exercised by CI. It is not an omission
to be fixed by adding scenarios there: a Metal run has nowhere to happen in CI.
A change to the macOS native path is verified by running the rig on one of the
Mac dev boxes by hand, which are reachable from dev-head and from nowhere in
GitHub Actions.

The server a native connection attaches to belongs to the rig, not to the
addon. ``blender_harness.py`` sets ``PPF_WIN_NATIVE_NO_SPAWN``,
``PPF_MAC_NATIVE_NO_SPAWN`` and ``PPF_LINUX_NATIVE_NO_SPAWN`` in every worker's
Blender environment, so the native backend attaches to the port the
orchestrator already started a server on. Without them the addon spawns a
second ``ppf-cts-server`` against the port the rig's own one still holds, and
the scenario drives whichever of the two wins the race. Scenarios do not opt
in.

### CI: three jobs, one backend

``.github/workflows/blender.yml`` runs the rig on:

- **macOS (SSH)** - macOS Blender drives a REAL CUDA
  solver on a disposable AWS GPU box over the addon's paramiko SSH backend
  (``server_type`` CUSTOM). Runs the real-only ``bl_ssh_remote_solve``
  (SHELL) and ``bl_ssh_remote_solid`` (SOLID) smokes, so both encode
  paths cross the tunnel. This is the only job exercising the SSH
  backend and the "local Blender + remote GPU" workflow.
- **Linux** - disposable AWS GPU instance (an L40S on the default
  ``g6e.2xlarge``, an L4 on the ``g6`` types), real CUDA build, the
  ``--backend real`` subset. Blender's window runs under Xvfb (software
  GL) while the solver uses the real GPU.
- **Windows** - disposable AWS GPU instance of the same types, real
  Windows-native build. The rig runs Blender headless (``--background``
  via ``PPF_BLENDER_HEADLESS=1``), which needs no OpenGL/desktop, so it
  runs directly over SSH whatever driver model the GPU comes up in. An L4
  comes up in TCC (compute-only) mode with no WGL/OpenGL, which is why a
  GUI launch is avoided entirely. The driver holds the
  main thread and drains its own PC2 frames, so scenarios complete in a
  single ``--python`` run with no event loop (see
  ``.github/workflows/scripts/win/run-blender-rig.ps1``).

## Scenarios

All registered scenarios live in ``blender_addon/debug/scenarios/`` and
are wired into the ``REGISTRY`` dict in ``scenarios/__init__.py``. They
split into two families:

Server-only (no Blender needed; do not require a build, just exercise
the wire protocol or the rig's own helpers):

- ``server_smoke``: first-ping NO_DATA contract.
- ``upload_id_changes``: two atomic uploads back-to-back must mint
  distinct ids; status returns to NO_BUILD because the prior build (if
  any) is invalid for the new upload.
- ``rig_launch_config``: how the rig launches Blender, rather than what
  the addon does once it is up. Pins the window default small, requires
  an unparseable ``PPF_BLENDER_WINDOW`` to raise instead of falling back
  to a size the caller will then misreport, and requires a dead display
  to read as dead (a false positive there sends Blender at a display
  that is not running, which fails inside GHOST naming nothing).

Blender-driven (opt-in, requires Blender; see the per-OS search order
under "Displays on Linux", or set ``PPF_BLENDER_BIN``): every other
entry. They cover connect paths (one scenario per native type:
``bl_connect_linux_native`` is Linux-only, ``bl_connect_win_native`` is
Windows-only, ``bl_mac_native_real_solve`` is macOS-only), the build and the
device a REMOTE launch names (``bl_remote_device_select``), the pin-fidelity
matrix, UI / state-machine integration, chain lifecycle, copy/paste
clipboards, fetch/transfer regressions, progress UX, and the
intersection-feedback round-trip. Use ``main.py runtests --list`` to
enumerate the full set on the current platform.

``bl_remote_device_select`` is the Compute Device and GPU Backend choices
reaching a server on ANOTHER machine, over SSH and over Docker, and it runs on
every CI leg because it needs no second machine, no sshd, no daemon and no GPU.
A remote backend is reached through exactly one abstraction, ``exec_command``,
so a stand-in for it records the script ``effect_runner._do_launch_server``
composes, and the scenario reads the resolved build directory and the exported
``CARGO_TARGET_DIR`` back out of that script. It asserts the launch the addon
sends, not the far side's response to it.

``bl_retired_connection_migration`` is what a saved artifact naming the retired
Local connection opens as, and it has to run inside Blender: an EnumProperty is
stored as its ITEM NUMBER, so a file holding a retired one reads the field's
DEFAULT with nothing reported, and only a real PropertyGroup carries the raw
ID-property the migration reads to see it. It covers the .blend path and the
profile path together, since a profile is the same question in a file the user
wrote, and it reads the platform mapping out of the module it checks rather than
restating it, so it asserts this platform's native connection wherever it runs.

Scenarios that need a *real build + run* (cancel-build, terminate-
run, save-and-quit-resume, solver-crash, fidelity tests of pin
animation against frontend Python) drive the addon to encode a real
scene because the real frontend rejects synthetic ``data.pickle``;
they live in the Blender-driven family.

## Per-worker isolation

Each worker gets its own subdirectory under
``$TMPDIR/ppf-debug/<run-id>/worker-NN/``:

- ``server/``  : ``ppf-cts-server`` CWD; ``progress.log``, ``server.log``, ``stdout.log``, ``stderr.log``.
- ``project/`` : ``PPF_CTS_DATA_ROOT`` shadow (per-worker isolation
  patches ``frontend.BlenderApp.__init__`` to honor it). Holds
  ``data.pickle``, ``param.pickle``, ``upload_id.txt``, ``app_state.pickle``,
  ``output/vert_*.bin``, ``save_*.bin``, ... Per-project filenames are
  defined in ``crates/ppf-cts-formats::files``, the single source of
  truth shared with the Rust server.
- ``probe/``   : ``probe_events.jsonl``, ``probe_assertions.jsonl``, ``probe_summary.json``.
- ``scenario.log`` : timestamped scenario log.

Passing workers' dirs are removed automatically. Failing workers are
kept verbatim for inspection. Use ``--keep-all`` to retain everything.

Ports are bind-to-zero allocated in the orchestrator before fork, so
parallel runs cannot collide.

## Parallel mode caveats

Two gotchas surface only at ``--parallel >= 2``:

- ``Utils.busy()`` is module-global state; under multiprocessing it
  can leak across workers if a scenario forgets to clear it. Chain
  scenarios are most exposed because they reuse the same engine
  across multiple build/run cycles.
- The live-fetch path races the Blender modal: when the modal cannot
  drain mid-driver, frames can be dropped. Scenarios that observe
  fetched frame counts must allow for this or pin ``--parallel 1`` /
  ``--parallel 2``.

## Knobs

Pass via ``--knob KEY=value``. The orchestrator forwards them to the
spawned server's environment.

| Env                              | Effect                                       |
| -------------------------------- | -------------------------------------------- |
| ``PPF_CTS_DATA_ROOT``            | Set automatically per worker; do not override |

Probe knobs (Blender-side scenarios only):

| Env                                  | Effect                                       |
| ------------------------------------ | -------------------------------------------- |
| ``PROBE_SAMPLE_HZ``              | State-sample rate (default 10)               |
| ``PROBE_BUDGET_CONNECTING_S``    | Stuck-CONNECTING budget in seconds (default 30) |
| ``PROBE_BUDGET_LAUNCHING_S``     | Stuck-LAUNCHING budget (default 20)          |
| ``PROBE_BUDGET_BUILDING_UNCHANGED_S`` | Build-progress-unchanged budget (default 10) |
| ``PROBE_BUDGET_RUNNING_UNCHANGED_S``  | Run-frame-unchanged budget (default 8)    |
| ``PROBE_MAX_DEPSGRAPH_PER_S``    | Runaway-depsgraph threshold (default 100)    |

## Adding a new scenario

1. Drop a module into ``blender_addon/debug/scenarios/``. Export
   ``run(ctx) -> dict`` returning ``{"status": "pass"|"fail",
   "violations": [...], "notes": [...]}``.
2. For Blender-driven scenarios, also set ``NEEDS_BLENDER = True`` at
   module level **and** export ``build_driver(ctx) -> str`` returning
   the Python source the bootstrap will exec inside Blender. The
   bootstrap exposes ``pkg`` (resolved addon module name), ``bpy``, and
   ``result`` (a dict to populate). When the driver returns, the
   bootstrap writes ``result`` to disk and quits Blender. The
   scenario's ``run(ctx)`` collects
   ``ctx.artifacts["blender_spec"].result_path``.
3. Add an entry to ``REGISTRY`` in ``scenarios/__init__.py``.

If your driver needs the wire protocol version, import it from the
canonical source per layer (the Python addon and ``ppf-cts-server``
each pin a single ``PROTOCOL_VERSION`` constant). Do not redefine it
inside the scenario. ``bl_upload_id_desync_recovery``,
``bl_violation_overlay_classification``, and ``server_smoke`` are
examples of this pattern.

If your driver mutates Blender data directly, use the helpers in
``blender_addon/core/mutation`` (``_raw_*``). Examples that call
them: ``bl_pin_rod_curve`` and ``bl_bake_animation``.

### Why the bootstrap pattern, not exec-over-TCP?

Blender's headless UI mode (no ``--background``, no display) only
ticks its event loop reliably for the first ``bpy.app.timers``
callback after launch. Subsequent timers, including those scheduled
by the addon's reload server's ``execute`` command, may never fire.
The bootstrap pattern packs the entire scenario into the first tick
and writes its result to disk before quitting.

## Building the Rust binaries the rig needs

```sh
# The real backend for the host (CUDA where a toolkit is present, Metal
# on macOS), which is what --backend real requires.
cargo build --release
cargo build --release -p ppf-cts-server
```

A CUDA-free host runs a scene through this rig by building the CPU backend:
``cargo build --release --features cpu`` produces a solver that computes real
physics from the same neutral kernels, at about 30x the wall clock; it is the right thing to build when a server
binary has to EXIST locally while the solving happens elsewhere
(``.github/workflows/blender.yml``'s ``macos-ssh`` job does exactly this
for the local worker slots), and unlike the deleted stub it cannot hand
back fake kinematics.

The workspace produces two release binaries that matter here:
``target/release/ppf-cts-server`` (from ``crates/ppf-cts-server``,
spawned by the orchestrator) and ``target/release/ppf-contact-solver``
(from ``crates/ppf-cts-solver``, the per-session binary that
``frontend.session.shell_command()`` invokes from inside the
server). The crate ``ppf-cts-solver`` keeps the historical binary
name ``ppf-contact-solver`` via its ``[[bin]]`` stanza, so launcher
scripts that hardcode that filename keep working. Both are built from
the same feature selection, so the server and the per-session solver it
launches always agree on which backend is in play.

## Solver math unit tests (host, no CUDA)

Some solver math is pure float code that runs identically on host and
device, so it can be unit tested with a plain C++ compiler with no nvcc
(macOS included). These are standalone from the Blender ``runtests``
rig, which drives the pipeline around the solver rather than the solver
math itself.

PDRD exact-rigid polar fit (``rigid_polar_quat`` in
``crates/ppf-cts-solver/src/kernels/energy/model/pdrd_polar.hpp``):

```sh
make -C crates/ppf-cts-solver/src/kernels/energy/model/tests test
```

This guards the rigid-body collapse where a PDRD body settled exactly
180 degrees from its rest pose lost half its volume in one frame. At
the 180-degree antipodal singularity of SO(3) an identity-seeded polar
fit returns a wrong rotation, and the rigidify partial-snap then lerps
the body linearly across it. The test asserts the fit recovers the true
rotation at the antipode (and over random rotations) and that a partial
rigidify snap keeps ``det(F) ~ 1`` (no collapse); it also confirms the
old identity-seed path collapses, so the test self-verifies it is in
the bug regime. Exit code is nonzero on any failure.
