# Debug tooling

Two unrelated tool families share this directory:

1. **Live-Blender debug client** (`main.py`, `perf.py`, `client.py`,
   `output.py`). Talks over TCP to a Blender instance that has the
   addon loaded with the debug-reload server enabled. Useful while
   developing: inspect state, exec arbitrary Python, reload the
   addon, drive MCP tools, profile draw time.

2. **Headless test rig** (`orchestrator.py`, `blender_harness.py`,
   `probe.py`, `scenarios/`, driven via `main.py runtests`). Spawns
   its own Blender + `ppf-cts-server` processes per worker, runs
   end-to-end integration scenarios against the real production
   pipeline (real `frontend` module, real Rust solver built with
   `--features emulated`), and writes pass/fail reports. See
   [`TEST_RIG.md`](TEST_RIG.md) for details.

The two families share nothing at runtime; the rig does not connect to
your live Blender, and the live-debug client does not invoke the rig.

## Live-Blender debug client

Prerequisite: launch Blender with the addon enabled and the
debug-reload server listening on TCP `8765` (the standard launch
script does this automatically).

```sh
# Status of debug + MCP ports.
python blender_addon/debug/main.py status

# List MCP tools the live addon exposes.
python blender_addon/debug/main.py tools

# Call a specific MCP tool with JSON arguments.
python blender_addon/debug/main.py call <tool> '{"arg": "value"}'

# Run a Python snippet inside the live Blender process.
python blender_addon/debug/main.py exec 'print(bpy.data.objects.keys())'

# Reload the addon (Python module + handlers).
python blender_addon/debug/main.py reload

# Two-phase reload (use when a PropertyGroup schema change doesn't
# show up after `reload`).
python blender_addon/debug/main.py full-reload

# Boot the MCP server inside Blender if it's not already up.
python blender_addon/debug/main.py start-mcp

# Run server/restart.sh on the addon's currently-connected remote.
python blender_addon/debug/main.py restart-server

# Inspect the current scene snapshot.
python blender_addon/debug/main.py scene

# List MCP resources, read one by URI.
python blender_addon/debug/main.py resources
python blender_addon/debug/main.py read <uri>
```

The MCP port defaults to `9633`; override with `--mcp-port`. The
debug/reload port (`8765`) is fixed.

`perf.py` is a focused front-end for `blender_addon/ui/perf.py`,
the in-addon draw-time profiler:

```sh
python blender_addon/debug/perf.py enable    # install timing wrappers
python blender_addon/debug/perf.py sample    # enable + force redraws + report
python blender_addon/debug/perf.py report    # print current stats
python blender_addon/debug/perf.py reset     # clear collected stats
python blender_addon/debug/perf.py disable
```

## Headless test rig

The rig runs without a live Blender. It spawns its own Blender and
`ppf-cts-server` processes in per-worker temp dirs, drives them
through the addon's public API, and validates end-to-end behavior.
It does not require a CUDA GPU: the Rust binary built with
`--features emulated` stubs the CUDA FFI and applies kinematic
constraints in pure Rust, producing a `vert_*.bin` stream that
matches a real run bit-for-bit for the pin-driven motion the
scenarios exercise.

The canonical entry point is the `runtests` subcommand of `main.py`,
which dispatches into `orchestrator.py`:

```sh
# All scenarios, sequential.
python blender_addon/debug/main.py runtests

# Parallel, four workers.
python blender_addon/debug/main.py runtests --parallel 4

# A specific subset.
python blender_addon/debug/main.py runtests \
    bl_overlay_invalidation bl_save_resume

# List every registered scenario.
python blender_addon/debug/main.py runtests --list

# Stability shake-out.
python blender_addon/debug/main.py runtests --parallel 4 --repeat 3

# Write the aggregated report to a specific path.
python blender_addon/debug/main.py runtests --report run.json
```

`orchestrator.py` is also runnable directly with the same flags
(`.venv/bin/python blender_addon/debug/orchestrator.py ...`); the
`runtests` subcommand is a thin wrapper that lazy-imports the
scenarios package so a syntax error in one scenario doesn't break
the rest of `main.py`.

Useful environment knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PPF_EMULATED_STEP_MS` | `1000` | Wall-clock ms per solver step. Set to `0` for unit tests, `100` for fast scenarios. |
| `PPF_EMULATED_FAIL_AT_FRAME` | (off) | Trip a synthetic solver failure at frame N (used by intersection-records and recovery scenarios). |
| `PPF_DEBUG_ROOT` | platform tempdir + `/ppf-debug` | Where per-run / per-worker dirs are created. |
| `PPF_BLENDER_BIN` | (auto) | Override Blender binary path. |
| `PPF_BLENDER_HEADLESS` | (off) | Pass `--background` to Blender. Only safe for scenarios that complete inside one `--python` script run. |
| `PPF_CTS_DATA_ROOT` | (auto) | Set per worker by the orchestrator; do not override. |

Each run drops a `report.json` plus per-worker artifacts (server logs,
Blender stdout/stderr, `scenario_result.json`, `probe_*.jsonl`) under
`$PPF_DEBUG_ROOT/<run-id>/`. On failure the worker dir is preserved so
you can inspect it; on success it is removed (use `--keep-all` to
retain everything).

For the full reference (scenario catalog, knobs, troubleshooting, the
emulated-Rust contract), see [`TEST_RIG.md`](TEST_RIG.md).
