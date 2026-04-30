# Windows-native backend test harnesses

Two scripts that verify the Blender addon's `WinNativeBackend` end-to-end
against a freshly built Windows solver. Both test the same production
code paths; one runs standalone, the other drives the full addon inside
a headless Blender session.

They were written to verify commit **24a9372d** ("Fix Windows-native
backend, compile, and bundle") and are the regression guard if anyone
touches `WinNativeBackend`, the three `win_native` branches in
`blender_addon/core/effect_runner.py`, or anything in
`build-win-native/bundle.bat` that changes what lands in the dist tree.

## Layouts

Both scripts accept either project layout. Past bugs have lived in each.

| Layout | Root path | Python |
|---|---|---|
| **Dev** | `C:\ppf-contact-solver` (repo root) | `C:\ppf-contact-solver\build-win-native\python\python.exe` |
| **Bundle** | `C:\ppf-contact-solver\build-win-native\dist` (from `bundle.bat`) | `<root>\python\python.exe` |

`connect_win_native` in `blender_addon/core/connection.py` auto-detects
the layout based on where it finds `python.exe`; that branching is
exactly what these tests exercise.

## Prerequisites

All of these run on the Windows box (`ssh win-build`). The host side
needs none of them — the scripts are SSH-ed in.

1. **Build done.** `build.bat` has run and produced
   `target/release/ppf-contact-solver.exe` +
   `src/cpp/build/lib/libsimbackend_cuda.dll`.
2. **(Bundle tests)** `bundle.bat` has run and populated
   `build-win-native/dist/`.
3. **(E2E tests)** Blender 5.0 at
   `C:\Program Files\Blender Foundation\Blender 5.0\blender.exe`, with
   the addon junction `%APPDATA%\Blender Foundation\Blender\5.0\scripts\addons\ppf-contact-solver → C:\ppf-contact-solver\blender_addon`.

## Running

### `test_backend_unit.py` — standalone, no Blender

Imports `blender_addon.core.backends` directly (side-loaded to dodge
the addon's bpy-dependent top-level `__init__.py`) and drives
`create_backend("win_native", …)` through: subprocess launch → port
bind → query round-trip → frame-file glob → `stop_server()`.

```bash
# Dev layout
ssh win-build 'C:\ppf-contact-solver\build-win-native\python\python.exe \
    C:\ppf-contact-solver\build-win-native\scripts\test_backend_unit.py  C:\ppf-contact-solver  9091'

# Bundle layout (use the bundled Python so the dist is really self-contained)
ssh win-build '"C:\ppf-contact-solver\build-win-native\dist\python\python.exe" \
    C:\ppf-contact-solver\build-win-native\scripts\test_backend_unit.py \
    "C:\ppf-contact-solver\build-win-native\dist"  9092'
```

Argv: `<root>  [port]`. Success prints an `OK [...]` line per step and
ends with `PASS test_backend_unit`. Failure dumps the server
subprocess's stderr (because a silent "port never bound" is useless for
debugging — ask me how I know).

### `test_backend_e2e.py` — headless Blender

Enables the addon, resolves its (hyphenated) module name dynamically,
then pumps `facade.tick()` through the full communicator lifecycle.

```bash
# Dev layout
ssh win-build '"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" \
    -b --python C:\ppf-contact-solver\build-win-native\scripts\test_backend_e2e.py'

# Bundle layout
ssh win-build 'set T2_ROOT=C:\ppf-contact-solver\build-win-native\dist && \
    set T2_PORT=9092 && \
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" \
    -b --python C:\ppf-contact-solver\build-win-native\scripts\test_backend_e2e.py'
```

Env: `T2_ROOT` (default `C:\ppf-contact-solver`), `T2_PORT` (default `9091`). Blender
eats positional argv before `--`, so env vars are the clean channel.
Success ends with `PASS test_backend_e2e`.

### Typical verification run — all four combinations

```bash
# 1/4 unit × dev
ssh win-build 'C:\ppf-contact-solver\build-win-native\python\python.exe \
    C:\ppf-contact-solver\build-win-native\scripts\test_backend_unit.py  C:\ppf-contact-solver  9091'

# 2/4 unit × bundle
ssh win-build '"C:\ppf-contact-solver\build-win-native\dist\python\python.exe" \
    C:\ppf-contact-solver\build-win-native\scripts\test_backend_unit.py \
    "C:\ppf-contact-solver\build-win-native\dist"  9092'

# 3/4 e2e × dev
ssh win-build '"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" \
    -b --python C:\ppf-contact-solver\build-win-native\scripts\test_backend_e2e.py'

# 4/4 e2e × bundle
ssh win-build 'set T2_ROOT=C:\ppf-contact-solver\build-win-native\dist && set T2_PORT=9092 && \
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" \
    -b --python C:\ppf-contact-solver\build-win-native\scripts\test_backend_e2e.py'
```

## Known quirks you'll hit

- **Addon module name has hyphens.** The install junction is
  `ppf-contact-solver`; the manifest id `ppf_contact_solver` is ignored
  by Blender's extension loader when the folder is a junction. You
  can't write `from ppf-contact-solver.core.x import y` — hyphens
  aren't identifier-legal. Both scripts use
  `importlib.import_module("ppf-contact-solver.core.x")`.

- **Addon disable stops the effect-runner worker** (expected — cleanup
  releases threads). Re-enable is responsible for reviving it. The E2E
  harness asserts the worker is alive after `addon_enable` and fails
  if `register()` → `ensure_engine_timer` → `runner.restart()` didn't
  fire. If you see that assertion trip, the reload path in
  `core/facade.py` is broken.

- **Unit test can't `import blender_addon`.** `blender_addon/__init__.py`
  pulls in Blender-only UI modules. The unit harness stubs `bpy` plus
  `blender_addon.models.console`, then side-loads `backends.py`,
  `protocol.py`, and `status.py` via
  `importlib.util.spec_from_file_location`.

- **Blender `-b` mode doesn't pump `bpy.app.timers` the way GUI mode
  does.** The persistent engine timer in `facade.ensure_engine_timer()`
  won't run on its own in background mode, so the E2E harness calls
  `facade.tick()` manually inside a loop.

- **server.py requires the `server/` package.** `server.py` at the
  repo root is a thin launcher — the state machine lives in `server/`.
  `bundle.bat` was missing a robocopy step for this; fixed in
  24a9372d.

## Cleanup

The unit test creates a disposable subdirectory at
`<ROOT>\session\output_backend_unit_fake\` and removes it on exit. If
the script is interrupted, clean up manually:

```bash
ssh win-build 'rmdir /s /q C:\ppf-contact-solver\session 2>NUL'
ssh win-build 'rmdir /s /q C:\ppf-contact-solver\build-win-native\dist\session 2>NUL'
```

## What these tests actually protect

Regression coverage for the fixes landed in **24a9372d**:

| Production change | Unit test covers | E2E test covers |
|---|:---:|:---:|
| `core/effect_runner.py:_do_validate_path` win_native branch (os.path.isfile) | — | ✓ |
| `core/effect_runner.py:_do_stop_server` win_native branch (backend.stop_server()) | ✓ | ✓ |
| `core/effect_runner.py:_count_remote_frames` win_native branch (glob.glob) | ✓ | — |
| `core/backends.py:WinNativeBackend.stop_server` new method | ✓ | ✓ |
| `core/connection.py:connect_win_native` layout autodetect (dev vs bundle) | ✓ | ✓ |
| `ui/connection_ops.py` port threading | — | ✓ (via T2_PORT) |
| `src/cpp/main/main.cu` invalidate_inactive_aabbs linkage fix | — (just lets `build.bat` succeed) | — |
| `build-win-native/bundle.bat` `server/` package copy | indirectly (bundle run fails without) | indirectly |

If either test starts failing, check the table above to scope the
regression before blaming the tests.
