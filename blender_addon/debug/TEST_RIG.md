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
- Wire protocol 0.04: ``TCMD`` text commands carry a big-endian u32
  length prefix in front of the payload (matches
  ``blender_addon/core/protocol.py::PROTOCOL_VERSION``).
  ``scenarios/_runner.py::ProtoClient`` mirrors it so server-only
  scenarios stay compatible with the production communicator.

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
cargo build --release -p ppf-cts-server --features emulated

# Install Blender-side deps (one-time).
./install-blender-addon.sh
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy scipy tqdm psutil tomli ipython pillow pythreejs pytetwild

# All scenarios.
.venv/bin/python blender_addon/debug/main.py runtests

# One scenario.
.venv/bin/python blender_addon/debug/main.py runtests bl_connect_local

# Stress: 3 repeats of everything at parallel=4.
.venv/bin/python blender_addon/debug/main.py runtests --parallel 4 --repeat 3 --report run.json

# Underlying orchestrator can be invoked directly too.
.venv/bin/python blender_addon/debug/orchestrator.py --parallel 4
```

## What runs where

| Layer                          | What's faked                          | What's real                                  |
| ------------------------------ | ------------------------------------- | -------------------------------------------- |
| ``blender_addon/debug/emulator.py`` | ``Utils.check_gpu`` (no-op stub), ``Utils.get_driver_version``, ``Utils.busy`` | All of ``frontend``, all of the addon, all transitions, atomic upload, monitor |
| ``crates/ppf-cts-solver/src/backend.rs`` (emulated) | CUDA kernels (no-op stubs)            | Scene loading, constraint build, vert_*.bin writer, frame loop |
| ``blender_addon/debug/probe.py`` | nothing (it's the observer)         | Hooks Blender's real handler tables, samples real ``engine.state`` |
| ``blender_addon/debug/orchestrator.py`` | nothing (host-side)            | Spawns the real ``ppf-cts-server`` binary (``--features emulated`` build) and real Blender |

Two artifacts cover the GPU absence:

- ``frontend.Utils.check_gpu`` is patched to a no-op (otherwise it
  raises ``RuntimeError: nvidia-smi not found``).
- The Rust binary is built with ``--features emulated`` so it never
  links against ``simbackend_cuda`` and never makes a CUDA call.

## Scenarios

The full catalog is registered in ``scenarios/__init__.py``; run
``main.py runtests --list`` to print every name. The two families are:

**Server-only (no Blender needed; just exercise wire protocol).**
These don't require a build because they only touch the upload /
status path:

- ``server_smoke`` -- first-ping NO_DATA contract.
- ``upload_id_changes`` -- two atomic uploads back-to-back must mint
  distinct ids; status returns to NO_BUILD because the prior build
  (if any) is invalid for the new upload.

**Blender-driven (opt-in, requires Blender installed at
``/Applications/Blender.app`` or ``PPF_BLENDER_BIN``).** Each module
sets ``NEEDS_BLENDER = True`` and exports ``build_driver(ctx) -> str``;
the harness spawns Blender, the bootstrap execs the driver on the
first event-loop tick, and the result lands in
``scenario_result.json``. Coverage spans:

- Connect / lifecycle: ``bl_connect_local``, ``bl_connect_win_native``,
  ``bl_rust_binary_protocol``, ``bl_addon_reload_handoff``.
- Pin-op fidelity matrix (cross-checks the Rust solver's per-frame
  pin trajectory against ``frontend.FixedScene.time(t)``):
  ``bl_pin_animation_fidelity``, ``bl_pin_spin_*``, ``bl_pin_scale_*``,
  ``bl_pin_torque``, ``bl_pin_compose_*``, ``bl_pin_rod_curve``,
  ``bl_static_op_anim``, ``bl_velocity_keyframes``,
  ``bl_collider_keyframes``, ``bl_bake_animation``.
- State-machine / overlay: ``bl_overlay_invalidation``,
  ``bl_race_state_machine``, ``bl_fetch_clear_refetch``,
  ``bl_geometry_hash``, ``bl_param_change``, ``bl_param_dirty``,
  ``bl_friction_mode``, ``bl_save_resume``, ``bl_load_disconnect``,
  ``bl_open_mainfile_disconnect``, ``bl_run_consistency``,
  ``bl_drape_ready_to_run``, ``bl_shallow_copy``,
  ``bl_shared_object_data``, ``bl_pc2_migration``,
  ``bl_ngon_rejection``, ``bl_multi_group``, ``bl_stitch_merge``.
- Transition chains: ``bl_transition_chains``, ``bl_chain_*``.
- Tier 1 bug-fix coverage: ``bl_upload_id_desync_recovery``,
  ``bl_mesh_cache_self_heal``, ``bl_live_frame_end_tracking``,
  ``bl_fetch_failed_watchdog``, ``bl_server_unknown_recovery``,
  ``bl_profile_load_batch``.
- Solver intersection feedback round-trip:
  ``bl_intersection_records_roundtrip``,
  ``bl_violation_overlay_classification``,
  ``bl_self_intersection_build_reject``.
- Copy/paste clipboards: ``bl_copy_paste_material_params``,
  ``bl_copy_paste_pin_ops``, ``bl_copy_paste_cross_type_material``.
- Operator-poll regressions: ``bl_transfer_disabled_during_run``,
  ``bl_transfer_skip_delete_when_no_data``.
- UX progress + stats: ``bl_progress_simulating``,
  ``bl_progress_fetching``, ``bl_realtime_stats_shown``.
- MCP and project-rename plumbing: ``bl_mcp_roundtrip``,
  ``bl_ftetwild_overrides``, ``bl_project_rename_resync``.

Scenarios that need a real build + run (cancel-build, terminate-run,
save-and-quit-resume, solver-crash, pin-animation fidelity ...) all
live in the Blender-driven family because the real frontend rejects
synthetic ``data.pickle``; the older protocol-only ``server_build`` /
``server_run`` synthetic-pickle scenarios were removed.

## Per-worker isolation

Each worker gets its own subdirectory under the per-run dir
(``$PPF_DEBUG_ROOT/<run-id>/worker-NN/`` if set; otherwise
``<tempdir>/ppf-debug/<run-id>/worker-NN/``, where ``<tempdir>`` is
``$TMPDIR`` on POSIX and ``%TEMP%`` on Windows):

- ``server/``  : ``ppf-cts-server`` CWD; ``progress.log``, ``server.log``, ``stdout.log``, ``stderr.log``.
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
| ``PPF_EMULATED_STEP_MS``         | Wall-clock ms per solver step (default 1000; ``0`` for unit tests) |
| ``PPF_EMULATED_FAIL_AT_FRAME``   | Trip a synthetic solver failure at frame N (intersection-records / recovery scenarios)  |
| ``PPF_DEBUG_ROOT``               | Override per-run dir (default: platform tempdir + ``/ppf-debug``) |
| ``PPF_BLENDER_BIN``              | Override Blender binary discovery             |
| ``PPF_BLENDER_HEADLESS``         | Pass ``--background`` to Blender; only safe for scenarios that complete inside one ``--python`` script run |
| ``PPF_CTS_DATA_ROOT``            | Set automatically per worker; do not override |
| ``PPF_CTS_BUILD_PYTHON``         | Set automatically per worker (points at the project ``.venv`` so the build worker can ``import frontend``); do not override |
| ``PPF_WIN_NATIVE_NO_SPAWN``      | Set automatically by the harness so the addon's WIN_NATIVE backend skips its own server spawn |

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
cargo build --release -p ppf-cts-server --features emulated
```

The workspace produces two release binaries that matter here:
``target/release/ppf-cts-server`` (from ``crates/ppf-cts-server``,
spawned by the orchestrator) and ``target/release/ppf-contact-solver``
(the binary name preserved via ``[[bin]]`` in
``crates/ppf-cts-solver``, the per-session binary that
``frontend.session.shell_command()`` invokes from inside the
server). Both pick up the workspace ``emulated`` feature: CUDA
calls are stubbed and per-frame kinematics are applied directly to
vertex positions in ``Backend::apply_kinematic_constraint`` (in
``crates/ppf-cts-solver/src/backend.rs``).
