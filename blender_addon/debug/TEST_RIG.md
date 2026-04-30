# Debug test rig

Automated, parallelizable scenarios that exercise the addon's
production state machine end-to-end against the **real** production
pipeline:

- Real ``frontend`` Python module on the server (``populate``, ``make``,
  session export).
- Real Rust solver binary at ``target/release/ppf-contact-solver``,
  built with ``--features emulated --no-default-features`` so it skips
  CUDA but still goes through the real scene loader, the real
  ``Constraint`` build, the real per-frame loop, and writes
  ``vert_*.bin`` in the real wire format.

The ``--features emulated`` build:

- Stubs the ``extern "C"`` CUDA kernel calls (``advance``, ``fetch``,
  ``initialize``, ``update_constraint`` ...) with Rust no-ops.
- After every ``scene.make_constraint(time)``, applies the kinematic
  ``FixPair`` positions directly to ``state.curr_vertex``. Pinned
  vertices land exactly where the addon's encoder said they should at
  that frame -- which is what the production CUDA solver does too for
  kinematic pins.
- Sleeps ``PPF_EMULATED_STEP_MS`` milliseconds per solver step
  (default 1000ms) so the run paces like a real simulation. Set
  ``PPF_EMULATED_STEP_MS=0`` for unit tests.

## Quick start

```sh
# Build the emulated Rust binary (one-time, on any host).
cargo build --release --features emulated --no-default-features

# Install Blender-side deps (one-time).
./install-blender-addon.sh
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy scipy numba tqdm psutil tomli ipython pillow pythreejs pytetwild

# All scenarios.
python3.12 blender_addon/debug/main.py runtests

# One scenario.
python3.12 blender_addon/debug/main.py runtests bl_connect_local

# Stress: 3 repeats of everything at parallel=4.
python3.12 blender_addon/debug/main.py runtests --parallel 4 --repeat 3 --report run.json
```

## What runs where

| Layer                          | What's faked                          | What's real                                  |
| ------------------------------ | ------------------------------------- | -------------------------------------------- |
| ``server/emulator.py``         | ``Utils.check_gpu`` (no-op stub)      | All of ``frontend``, all of the addon, all transitions, atomic upload, monitor |
| ``src/backend.rs`` (emulated)  | CUDA kernels (no-op stubs)            | Scene loading, constraint build, vert_*.bin writer, frame loop |
| ``blender_addon/debug/probe.py`` | nothing (it's the observer)         | Hooks Blender's real handler tables, samples real ``engine.state`` |
| ``blender_addon/debug/orchestrator.py`` | nothing (host-side)            | Spawns real ``server.py --debug``, real Blender |

Two artifacts cover the GPU absence:

- ``frontend.Utils.check_gpu`` is patched to a no-op (otherwise it
  raises ``RuntimeError: nvidia-smi not found``).
- The Rust binary is built with ``--features emulated`` so it never
  links against ``simbackend_cuda`` and never makes a CUDA call.

## Scenarios

Server-only (no Blender needed; do not require a build, just exercise
wire protocol):

- ``server_smoke`` -- first-ping NO_DATA contract.
- ``upload_id_changes`` -- two atomic uploads back-to-back must mint
  distinct ids; status returns to NO_BUILD because the prior build (if
  any) is invalid for the new upload.

Blender-driven (opt-in, requires Blender installed at
``/Applications/Blender.app`` or ``PPF_BLENDER_BIN``):

- ``bl_connect_local`` -- addon's Local backend reaches ONLINE; probe
  is active and captures the connect lifecycle.

Scenarios that need a *real build + run* (e.g. cancel-build,
terminate-run, save-and-quit-resume, solver-crash, fidelity tests of
pin animation against frontend Python) must drive the addon to encode
a real scene because the real frontend rejects synthetic
``data.pickle``. These belong under the Blender-driven family; the
older synthetic ``server_build`` / ``server_run`` etc. are gone.

## Per-worker isolation

Each worker gets its own subdirectory under
``$TMPDIR/ppf-debug/<run-id>/worker-NN/``:

- ``server/``  : server.py CWD; ``progress.log``, ``server.log``, ``stdout.log``, ``stderr.log``.
- ``project/`` : ``PPF_CTS_DATA_ROOT`` shadow (the rust-mode emulator
  patches ``frontend.BlenderApp.__init__`` to honor it). Holds
  ``data.pickle``, ``param.pickle``, ``upload_id.txt``, ``app_state.pickle``,
  ``output/vert_*.bin``, ``save_*.bin`` ...
- ``probe/``   : ``probe_events.jsonl``, ``probe_assertions.jsonl``, ``probe_summary.json``.
- ``scenario.log`` : timestamped scenario log.

Passing workers' dirs are removed automatically. Failing workers are
kept verbatim for inspection. Use ``--keep-all`` to retain everything.

Ports are bind-to-zero allocated in the orchestrator before fork, so
parallel runs cannot collide.

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

### Why the bootstrap pattern, not exec-over-TCP?

Blender's headless UI mode (no ``--background``, no display) only
ticks its event loop reliably for the first ``bpy.app.timers``
callback after launch. Subsequent timers -- including those scheduled
by the addon's reload server's ``execute`` command -- may never fire.
The bootstrap pattern packs the entire scenario into the first tick
and writes its result to disk before quitting.

## Building the emulated Rust binary

The default build links ``simbackend_cuda`` and requires the CUDA
toolkit. To build a CUDA-free binary for the test rig:

```sh
cargo build --release --features emulated --no-default-features
```

The resulting binary at ``target/release/ppf-contact-solver`` is what
``frontend.session.shell_command()`` invokes. The CUDA calls are
stubbed; per-frame kinematics are applied directly to vertex positions
in ``Backend::apply_kinematic_constraint``.
