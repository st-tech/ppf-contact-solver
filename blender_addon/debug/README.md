# Debug tooling

Two unrelated tool families share this directory:

1. **Live-Blender debug client** (`main.py`, `perf.py`, `client.py`,
   `output.py`). Talks over TCP to a Blender instance that has the
   addon loaded with the debug-reload server enabled. Useful while
   developing: inspect state, exec arbitrary Python, reload the
   addon, drive MCP tools, profile draw time.

2. **Headless test rig** (`orchestrator.py`, `blender_harness.py`,
   `probe.py`, `scenarios/`). Spawns its own Blender + server
   processes per worker, runs end-to-end integration scenarios
   against the real production pipeline (real `frontend` module,
   real Rust solver built with `--features emulated`), and writes
   pass/fail reports. See [`TEST_RIG.md`](TEST_RIG.md) for details.

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

# Boot the MCP server inside Blender if it's not already up.
python blender_addon/debug/main.py start-mcp

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
python blender_addon/debug/perf.py enable    # start collecting timings
python blender_addon/debug/perf.py report    # dump aggregated frame times
python blender_addon/debug/perf.py disable
```

## Headless test rig

The rig runs without a live Blender. It spawns its own Blender and
server processes in per-worker temp dirs, drives them through the
addon's public API, and validates end-to-end behavior. It does not
require a CUDA GPU: the Rust binary built with `--features emulated`
stubs the CUDA FFI and applies kinematic constraints in pure Rust,
producing a `vert_*.bin` stream that's bit-for-bit comparable to a
real run for the pin-driven motion the scenarios exercise.

```sh
# All scenarios, sequential.
.venv/bin/python blender_addon/debug/orchestrator.py

# Parallel, four workers.
.venv/bin/python blender_addon/debug/orchestrator.py --parallel 4

# A specific subset.
.venv/bin/python blender_addon/debug/orchestrator.py \
    bl_overlay_invalidation bl_save_resume

# List every registered scenario.
.venv/bin/python blender_addon/debug/orchestrator.py --list

# Stability shake-out.
.venv/bin/python blender_addon/debug/orchestrator.py \
    --parallel 4 --repeat 3
```

Useful environment knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PPF_EMULATED_STEP_MS` | `1000` | Wall-clock ms per solver step. Tests use `100`. |
| `PPF_DEBUG_ROOT` | `$TMPDIR/ppf-debug` | Where worker dirs are created. |
| `PPF_BLENDER_BIN` | (auto) | Override Blender binary path. |
| `PPF_DEBUG_PROBE` | (off) | Enable in-Blender event probe. |

Each run drops a `report.json` plus per-worker artifacts (server logs,
Blender stdout/stderr, `scenario_result.json`) under
`$PPF_DEBUG_ROOT/<run-id>/`. On failure the worker dir is preserved so
you can dig in; on success it is removed.

For the full reference (scenario catalog, knobs, troubleshooting, the
emulated-Rust contract), see [`TEST_RIG.md`](TEST_RIG.md).
