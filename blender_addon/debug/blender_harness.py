# File: blender_harness.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Per-worker Blender process management. Used by Phase-2 scenarios that
# need to drive the actual addon UI through its operators rather than
# just talking to the server on the wire.
#
# Each worker that opts into Blender gets its own:
#   - Blender process (headless: --background, no audio)
#   - debug/reload TCP port (ephemeral, allocated by the orchestrator)
#   - MCP HTTP port (optional; only spawned when the scenario asks for it)
#   - factory-startup .blend (no user prefs leak across workers)
#
# The harness exposes a thin interface on top of the existing debug
# transport (debug/client.py) so scenarios can:
#   - exec Python inside Blender
#   - install + start the probe
#   - retrieve the probe's summary

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class BlenderSpec:
    """Where to find Blender, where its companion ports + paths live."""

    blender_bin: str
    workspace: str
    probe_dir: str
    blend_file: str
    driver_source: str  # Python source to exec inside Blender
    addon_name: str = "bl_ext.user_default.ppf_contact_solver"

    stdout_path: str = ""
    stderr_path: str = ""
    result_path: str = ""
    driver_path: str = ""

    def __post_init__(self) -> None:
        if not self.stdout_path:
            self.stdout_path = os.path.join(self.workspace, "blender_stdout.log")
        if not self.stderr_path:
            self.stderr_path = os.path.join(self.workspace, "blender_stderr.log")
        if not self.result_path:
            self.result_path = os.path.join(self.workspace, "scenario_result.json")
        if not self.driver_path:
            self.driver_path = os.path.join(self.workspace, "scenario_driver.py")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_blender(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve a Blender binary. Order of precedence:

      1. ``explicit`` argument
      2. PPF_BLENDER_BIN env
      3. /Applications/Blender.app/Contents/MacOS/Blender   (macOS default)
      4. ``blender`` on PATH
    """
    if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
        return explicit
    env = os.environ.get("PPF_BLENDER_BIN")
    if env and os.path.isfile(env):
        return env
    macos = "/Applications/Blender.app/Contents/MacOS/Blender"
    if os.path.isfile(macos):
        return macos
    return shutil.which("blender")


# ---------------------------------------------------------------------------
# Bootstrap script
# ---------------------------------------------------------------------------

# Bootstrap mirrors blender_addon/launch.sh: addon enable happens via
# the ``--addons`` CLI flag (which Blender resolves before any --python
# runs), and the package name is discovered via sys.modules so the
# Blender 5 ``bl_ext.user_default.<id>`` rename is handled transparently.
# Bootstrap pattern: the orchestrator embeds the scenario's driver code
# directly into the bootstrap, runs it on the first event-loop tick,
# writes the result JSON to ``RESULT_PATH``, then quits Blender. The
# orchestrator collects the artifact after Blender exits. This avoids
# the Blender headless event-loop pitfall where only the first
# bpy.app.timers callback after launch reliably fires.
_BOOTSTRAP_TEMPLATE = """\
import bpy, sys, json, traceback

PROBE_DIR = {probe_dir!r}
RESULT_PATH = {result_path!r}
DRIVER_PATH = {driver_path!r}


def _write_result(result):
    try:
        with open(RESULT_PATH, "w") as f:
            json.dump(result, f, default=str)
    except OSError as exc:
        print(f"bootstrap: result write failed: {{exc}}",
              file=sys.stderr, flush=True)


def _quit():
    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        # quit_blender requires a window context that may be missing
        # in headless UI mode -- fall back to abort.
        import os
        os._exit(0)
    return None


def _hide_window():
    # macOS: hide Blender's UI immediately. Many test workers visible
    # at once on the desktop is noisy and steals focus. ``--background``
    # would skip the event loop (we need it for timers), but
    # ``System Events`` -> "set visible to false" hides the window
    # while Blender keeps ticking.
    if sys.platform == "darwin":
        try:
            import subprocess
            subprocess.Popen(
                ["osascript", "-e",
                 'tell application "System Events" to set visible '
                 'of process "Blender" to false'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def _start():
    result = {{
        "phase": "?", "server": "?", "solver": "?",
        "connected": False, "errors": [], "scenario_done": False,
    }}
    try:
        _hide_window()
        # Strip a known leaf off whatever sys.modules entry holds the
        # addon to recover its root package name. Works for both the
        # extension layout (bl_ext.user_default.<id>.ui.solver) and any
        # legacy single-segment layout.
        pkg = next(n.removesuffix(".ui.solver") for n in sys.modules
                   if n.endswith(".ui.solver"))
        print(f"bootstrap: addon package = {{pkg}}", flush=True)

        if PROBE_DIR:
            try:
                probe = __import__(pkg + ".debug.probe", fromlist=["start"])
                probe.start(PROBE_DIR)
            except Exception as exc:
                result["errors"].append(f"probe.start: {{exc}}")

        # Driver runs on the same main-thread tick. exec_globals exposes
        # the resolved package and a result dict the driver can populate.
        with open(DRIVER_PATH) as f:
            driver_src = f.read()
        exec_globals = {{
            "pkg": pkg,
            "result": result,
            "bpy": bpy,
            "PROBE_DIR": PROBE_DIR,
            "RESULT_PATH": RESULT_PATH,
        }}
        exec(driver_src, exec_globals)
        result["scenario_done"] = True

        if PROBE_DIR:
            try:
                probe = __import__(pkg + ".debug.probe", fromlist=["stop"])
                summary = probe.stop()
                result["probe_summary"] = summary
            except Exception as exc:
                result["errors"].append(f"probe.stop: {{exc}}")

    except Exception as exc:
        result["errors"].append(f"{{type(exc).__name__}}: {{exc}}")
        result["errors"].append(traceback.format_exc())

    _write_result(result)

    # Drain any queued animation frames before quitting. The addon's
    # frame_pump modal writes PC2 from _anim_frames -- but a modal
    # operator's TIMER cannot fire while we're inside the driver exec
    # (we hold the main thread). Returning a float from a timer
    # callback yields control to the event loop, which lets the
    # modal's 0.1s timer fire and drain. We re-queue ourselves until
    # the queue is empty (or a 30s timeout hits, in case the scenario
    # never fetched any frames).
    try:
        runner_mod = __import__(
            (next(n.removesuffix(".ui.solver") for n in sys.modules
                  if n.endswith(".ui.solver")) + ".core.facade"),
            fromlist=["runner"],
        )
        runner = runner_mod.runner
    except Exception:
        runner = None

    deadline = [__import__("time").monotonic() + 30.0]
    quiet_polls = [0]

    def _drain_then_quit():
        import time as _t
        if runner is None:
            bpy.app.timers.register(_quit, first_interval=0.05)
            return None
        with runner._anim_lock:
            queued = len(runner._anim_frames)
            total = runner._anim_total
            applied = runner._anim_applied
        # "done" = queue empty AND (we never had work, OR we applied
        # everything we expected). Need a few quiet polls in a row to
        # avoid quitting between two TIMER firings of the modal.
        if queued == 0 and (total == 0 or applied >= total):
            quiet_polls[0] += 1
        else:
            quiet_polls[0] = 0
        if quiet_polls[0] >= 3 or _t.monotonic() > deadline[0]:
            print(f"bootstrap: drain done queued={{queued}} "
                  f"applied={{applied}}/{{total}}", flush=True)
            bpy.app.timers.register(_quit, first_interval=0.05)
            return None
        return 0.2  # re-queue, yields the event loop to the modal

    bpy.app.timers.register(_drain_then_quit, first_interval=0.2)
    return None


# 2s delay so addon registration and any post-load handlers settle.
bpy.app.timers.register(_start, first_interval=2.0)
"""


def _bootstrap_source(spec: BlenderSpec) -> str:
    return _BOOTSTRAP_TEMPLATE.format(
        probe_dir=spec.probe_dir,
        result_path=spec.result_path,
        driver_path=spec.driver_path,
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def alloc_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def spawn(spec: BlenderSpec) -> subprocess.Popen:
    """Launch Blender with the addon enabled and the bootstrap script
    queued. Returns the Popen so the caller can wait/kill it.

    The bootstrap reads the scenario driver from disk, runs it on the
    first event-loop tick, writes ``spec.result_path``, and quits."""
    os.makedirs(spec.workspace, exist_ok=True)
    os.makedirs(spec.probe_dir, exist_ok=True)

    # Driver source landed on disk so the bootstrap can exec() it. We
    # don't inline driver code into the bootstrap because it can be
    # large and contain literals that escape format-string handling.
    with open(spec.driver_path, "w") as f:
        f.write(spec.driver_source)

    # Drop the bootstrap onto disk so Blender's --python flag can find it.
    bootstrap_path = os.path.join(spec.workspace, "bootstrap.py")
    with open(bootstrap_path, "w") as f:
        f.write(_bootstrap_source(spec))

    # We deliberately do NOT override BLENDER_USER_RESOURCES: that env
    # var redirects scripts/addons discovery, and the addon under test
    # lives under the user's normal prefs dir via the install script.
    # ``--factory-startup`` already gives us a clean per-session state
    # (no recent files / window layout / user prefs); per-worker dir
    # isolation comes from the workspace + project paths instead.
    env = os.environ.copy()
    env["PPF_DEBUG_PROBE"] = "1"
    env["PPF_DEBUG_PROBE_DIR"] = spec.probe_dir

    # Blender's ``--background`` mode skips the event loop, so any code
    # we register via ``bpy.app.timers.register`` never runs. The
    # bootstrap relies on a deferred timer to start the reload server,
    # so we launch with the UI by default. On macOS this opens a small
    # window per worker; the orchestrator tears it down at scenario
    # end. Set ``PPF_BLENDER_HEADLESS=1`` to opt into background mode
    # for scenarios that complete entirely within a single --python
    # script run (no event-loop required).
    #
    # ``--addons <id>`` is what enables the addon (NOT bpy.ops.preferences
    # .addon_enable from inside the bootstrap); this matches launch.sh.
    args = [
        spec.blender_bin,
        "--factory-startup",
        "--addons", spec.addon_name,
        "--python", bootstrap_path,
    ]
    if os.environ.get("PPF_BLENDER_HEADLESS") == "1":
        args.insert(1, "--background")
    if spec.blend_file:
        args.insert(1, spec.blend_file)

    stdout = open(spec.stdout_path, "wb")
    stderr = open(spec.stderr_path, "wb")
    proc = subprocess.Popen(
        args, env=env, stdout=stdout, stderr=stderr,
        cwd=spec.workspace, start_new_session=True,
    )
    proc._ppf_stdout = stdout  # type: ignore[attr-defined]
    proc._ppf_stderr = stderr  # type: ignore[attr-defined]
    return proc


def wait_for_result(spec: BlenderSpec, proc: subprocess.Popen, *,
                    timeout: float = 120.0) -> dict:
    """Block until ``spec.result_path`` is written or Blender exits.

    Returns the parsed result dict. Raises ``TimeoutError`` if neither
    happens within ``timeout``."""
    import json
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(spec.result_path):
            # Wait a beat for the writer to finish flushing.
            for _ in range(20):
                try:
                    with open(spec.result_path) as f:
                        return json.load(f)
                except (OSError, ValueError):
                    time.sleep(0.05)
            return {"errors": ["result file unreadable"]}
        if proc.poll() is not None:
            # Blender exited without writing -- collect diagnostics.
            return {
                "errors": [
                    f"Blender exited (rc={proc.returncode}) without writing "
                    f"{spec.result_path}",
                ],
            }
        time.sleep(0.2)
    raise TimeoutError(
        f"Blender did not write {spec.result_path} within {timeout}s"
    )


def _kill_tree(proc: subprocess.Popen, *, timeout: float) -> None:
    """SIGTERM the spawned process plus any descendants, fall back to
    SIGKILL after ``timeout``. POSIX uses killpg against the new session
    we created with start_new_session=True; Windows uses taskkill /T to
    walk the process tree."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, timeout=2.0,
                )
            except (subprocess.TimeoutExpired, OSError):
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        return
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=2.0)


def shutdown(proc: subprocess.Popen, *, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        _kill_tree(proc, timeout=timeout)
    finally:
        for h in (
            getattr(proc, "_ppf_stdout", None),
            getattr(proc, "_ppf_stderr", None),
        ):
            if h:
                try:
                    h.close()
                except OSError:
                    pass


