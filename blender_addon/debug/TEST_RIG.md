# Debug test rig

Automated, parallelizable scenarios that exercise the addon's
production state machine end-to-end against the **real** production
pipeline:

- Real ``frontend`` Python module loaded in-process by the server
  (``populate``, ``make``, session export). ``frontend`` imports the
  ``_ppf_cts_py`` PyO3 module from ``crates/ppf-cts-py``; the addon
  itself does not import it.
- Real Rust solver binary at ``target/release/ppf-cts-server`` (built
  from ``crates/ppf-cts-server``), with the workspace ``emulated``
  feature enabled (``cargo build --release -p ppf-cts-server
  --features emulated``) so it skips CUDA but still goes through the
  real scene loader, the real ``Constraint`` build, the real per-frame
  loop, and writes ``vert_*.bin`` in the real wire format from
  ``crates/ppf-cts-formats``.

The ``--features emulated`` build:

- Stubs the ``extern "C"`` CUDA kernel calls (``advance``, ``fetch``,
  ``initialize``, ``update_constraint``, ...) with Rust no-ops.
- After every ``scene.make_constraint(time)``, applies the kinematic
  ``FixPair`` positions directly to ``state.curr_vertex``. Pinned
  vertices land exactly where the addon's encoder said they should at
  that frame, which is what the production CUDA solver does too for
  kinematic pins.
- Sleeps ``PPF_EMULATED_STEP_MS`` milliseconds per solver step
  (default 1000ms) so the run paces like a real simulation. Set
  ``PPF_EMULATED_STEP_MS=0`` for unit tests.

## Quick start

```sh
# Build the emulated Rust binary (one-time, on any host).
cargo build --release -p ppf-cts-server --features emulated

# Install Blender-side deps (one-time).
./install-blender-addon.sh
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy scipy tqdm psutil tomli ipython pillow pythreejs pytetwild

# All scenarios.
python3.12 blender_addon/debug/main.py runtests

# One scenario.
python3.12 blender_addon/debug/main.py runtests bl_connect_local

# Stress: 3 repeats of everything at parallel=4.
python3.12 blender_addon/debug/main.py runtests --parallel 4 --repeat 3 --report run.json
```

The rig is all-green on macOS, Linux, and Windows (host scenario
counts vary by platform gate: 65 on macOS, 67 on Linux, 66 on
Windows). Chain scenarios can flake at ``--parallel 4`` (see
"Parallel mode caveats" below); drop to ``--parallel 2`` if you need
stable chain results.

## What runs where

| Layer                          | What's faked                          | What's real                                  |
| ------------------------------ | ------------------------------------- | -------------------------------------------- |
| ``server/emulator.py``         | ``Utils.check_gpu`` (no-op stub)      | All of ``frontend``, all of the addon, all transitions, atomic upload, monitor |
| ``crates/ppf-cts-solver/src/backend.rs`` (emulated) | CUDA kernels (no-op stubs)            | Scene loading, constraint build, vert_*.bin writer, frame loop |
| ``blender_addon/debug/probe.py`` | nothing (it's the observer)         | Hooks Blender's real handler tables, samples real ``engine.state`` |
| ``blender_addon/debug/orchestrator.py`` | nothing (host-side)            | Spawns the real ``ppf-cts-server`` binary (``--features emulated`` build) and real Blender |

Two artifacts cover the GPU absence:

- ``frontend.Utils.check_gpu`` is patched to a no-op (otherwise it
  raises ``RuntimeError: nvidia-smi not found``).
- The Rust binary is built with ``--features emulated`` so it never
  links against ``simbackend_cuda`` and never makes a CUDA call.

## Scenarios

All registered scenarios live in ``blender_addon/debug/scenarios/`` and
are wired into the ``REGISTRY`` dict in ``scenarios/__init__.py`` (71
entries at last count). They split into two families:

Server-only (no Blender needed; do not require a build, just exercise
the wire protocol):

- ``server_smoke``: first-ping NO_DATA contract.
- ``upload_id_changes``: two atomic uploads back-to-back must mint
  distinct ids; status returns to NO_BUILD because the prior build (if
  any) is invalid for the new upload.

Blender-driven (opt-in, requires Blender installed at
``/Applications/Blender.app`` or ``PPF_BLENDER_BIN``): the remaining
69 entries. They cover connect paths (``bl_connect_local`` is Linux-
only, ``bl_connect_win_native`` is Windows-only), the pin-fidelity
matrix, UI / state-machine integration, chain lifecycle, copy/paste
clipboards, fetch/transfer regressions, progress UX, and the
intersection-feedback round-trip. Use ``main.py runtests --list`` to
enumerate the full set on the current platform.

Scenarios that need a *real build + run* (cancel-build, terminate-
run, save-and-quit-resume, solver-crash, fidelity tests of pin
animation against frontend Python) drive the addon to encode a real
scene because the real frontend rejects synthetic ``data.pickle``;
they live in the Blender-driven family.

## Per-worker isolation

Each worker gets its own subdirectory under
``$TMPDIR/ppf-debug/<run-id>/worker-NN/``:

- ``server/``  : ``ppf-cts-server`` CWD; ``progress.log``, ``server.log``, ``stdout.log``, ``stderr.log``.
- ``project/`` : ``PPF_CTS_DATA_ROOT`` shadow (the rust-mode emulator
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
| ``PPF_EMULATED_STEP_MS``         | Wall-clock ms per solver step (default 1000) |
| ``PPF_CTS_DATA_ROOT``            | Set automatically per worker; do not override |

Probe knobs (Blender-side scenarios only):

| Env                                  | Effect                                       |
| ------------------------------------ | -------------------------------------------- |
| ``PPF_PROBE_SAMPLE_HZ``              | State-sample rate (default 10)               |
| ``PPF_PROBE_BUDGET_CONNECTING_S``    | Stuck-CONNECTING budget in seconds (default 30) |
| ``PPF_PROBE_BUDGET_LAUNCHING_S``     | Stuck-LAUNCHING budget (default 20)          |
| ``PPF_PROBE_BUDGET_BUILDING_UNCHANGED_S`` | Build-progress-unchanged budget (default 10) |
| ``PPF_PROBE_BUDGET_RUNNING_UNCHANGED_S``  | Run-frame-unchanged budget (default 8)    |
| ``PPF_PROBE_MAX_DEPSGRAPH_PER_S``    | Runaway-depsgraph threshold (default 100)    |

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

## Building the emulated Rust binary

The default build links ``simbackend_cuda`` and requires the CUDA
toolkit. To build a CUDA-free binary for the test rig:

```sh
cargo build --release -p ppf-cts-server --features emulated
```

The workspace produces two release binaries that matter here:
``target/release/ppf-cts-server`` (from ``crates/ppf-cts-server``,
spawned by the orchestrator) and ``target/release/ppf-contact-solver``
(from ``crates/ppf-cts-solver``, the per-session binary that
``frontend.session.shell_command()`` invokes from inside the
server). The crate ``ppf-cts-solver`` keeps the historical binary
name ``ppf-contact-solver`` via its ``[[bin]]`` stanza, so launcher
scripts that hardcode that filename keep working. Both binaries pick
up the workspace ``emulated`` feature: CUDA calls are stubbed and
per-frame kinematics are applied directly to vertex positions in
``Backend::apply_kinematic_constraint`` (in
``crates/ppf-cts-solver/src/backend.rs``).
