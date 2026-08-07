# File: blender_harness.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Per-worker Blender process management. Used by scenarios that drive
# the actual addon UI through its operators rather than just talking to
# the server on the wire.
#
# Each worker that opts into Blender gets its own:
#   - Blender process (UI mode, small window; --background only under
#     PPF_BLENDER_HEADLESS=1, since the bootstrap needs the event loop)
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

import atexit
import os
import select
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

    # Extra environment variables to inject into the Blender process.
    # The orchestrator passes the scenario's effective knobs here so
    # addon-side knobs (e.g. PPF_FORCE_TCP_TRANSFER, which selects the
    # co-located transport) reach the addon, not just the server.
    env_extra: dict = field(default_factory=dict)

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

def _default_install_paths() -> list[str]:
    """Well-known Blender locations for the running OS, newest first.

    ``blender`` is often absent from PATH on a desktop install (macOS
    puts the binary inside the .app bundle, and a Linux tarball install
    is usually unpacked under /opt without a PATH entry), so PATH alone
    is not a sufficient fallback. Linux entries are globbed because the
    tarball unpacks to a version-stamped directory.
    """
    if sys.platform == "darwin":
        return ["/Applications/Blender.app/Contents/MacOS/Blender"]
    if sys.platform.startswith("linux"):
        import glob
        found: list[str] = []
        for pattern in (
            "/opt/blender-*/blender",
            "/usr/local/blender-*/blender",
            "/opt/blender/blender",
            "/snap/bin/blender",
        ):
            found.extend(sorted(glob.glob(pattern), reverse=True))
        return found
    return []


def find_blender(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve a Blender binary. Order of precedence:

      1. ``explicit`` argument
      2. PPF_BLENDER_BIN env
      3. the running OS's well-known install locations
      4. ``blender`` on PATH
    """
    if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
        return explicit
    env = os.environ.get("PPF_BLENDER_BIN")
    if env and os.path.isfile(env):
        return env
    for candidate in _default_install_paths():
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("blender")


# ---------------------------------------------------------------------------
# Window sizing
# ---------------------------------------------------------------------------

# A worker's Blender exists to tick its event loop and run operators;
# nothing reads pixels back off the screen. Left to itself Blender opens
# near-fullscreen, and both the window surface and the GPU-side buffers
# it drives scale with that area, so N parallel workers each hold a
# full-screen surface for no benefit. 800x600 is small enough to make
# that cost negligible while still laying out the default 3D viewport
# with its sidebar, which the panel-draw scenarios rely on.
_DEFAULT_WINDOW = (0, 0, 800, 600)


def window_geometry() -> Optional[tuple[int, int, int, int]]:
    """Return ``(x, y, w, h)`` for the rig's Blender window.

    ``PPF_BLENDER_WINDOW`` overrides it and accepts ``WxH``, ``X,Y,W,H``,
    or one of ``0`` / ``off`` / ``default`` to pass no geometry flags at
    all and let Blender size its own window. An unparseable value raises:
    a silently ignored override would be reported as a size that is not
    in effect.
    """
    raw = (os.environ.get("PPF_BLENDER_WINDOW") or "").strip().lower()
    if not raw:
        return _DEFAULT_WINDOW
    if raw in ("0", "off", "default", "none"):
        return None
    try:
        if "x" in raw and "," not in raw:
            w, h = (int(v) for v in raw.split("x", 1))
            return (0, 0, w, h)
        parts = [int(v) for v in raw.split(",")]
    except ValueError:
        raise ValueError(
            f"PPF_BLENDER_WINDOW={raw!r} is not WxH, X,Y,W,H, or off"
        ) from None
    if len(parts) == 2:
        return (0, 0, parts[0], parts[1])
    if len(parts) == 4:
        return (parts[0], parts[1], parts[2], parts[3])
    raise ValueError(
        f"PPF_BLENDER_WINDOW={raw!r} is not WxH, X,Y,W,H, or off"
    )


def window_args() -> list[str]:
    """Blender CLI flags that keep a worker's window small and passive.

    ``--no-window-focus`` matters beyond politeness on a desktop: a
    worker that steals focus on spawn pulls the keyboard away from
    whatever the developer is doing, and several workers starting at
    once fight over it.
    """
    geom = window_geometry()
    if geom is None:
        return []
    x, y, w, h = geom
    return [
        "--window-geometry", str(x), str(y), str(w), str(h),
        "--no-window-focus",
    ]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

# Xvfb owned by this interpreter, if we started one. The rig launches
# Blender WITHOUT --background because the bootstrap needs the event loop
# to tick its timers, and a ticking event loop needs a window, which on
# Linux needs an X display.
_xvfb: Optional[subprocess.Popen] = None
_xvfb_display: str = ""

# Display numbers the rig's own server may occupy. High enough to stay
# clear of a desktop session (:0) and of the :99 that CI and
# install-blender.sh use, so a rig run can coexist with both.
_XVFB_DISPLAY_BASE = 100
_XVFB_DISPLAY_TRIES = 30


def _terminate(proc: subprocess.Popen) -> None:
    """Stop *proc*, escalating to SIGKILL, and close its stderr pipe."""
    if proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if proc.stderr:
        try:
            proc.stderr.close()
        except OSError:
            pass


def _display_is_live(display: str) -> bool:
    """True if *display* accepts a connection right now.

    A DISPLAY left over from a closed ssh -X forward or an ended RDP
    session still looks set, and Blender meeting one fails late and
    opaquely inside GHOST. Connecting first turns that into something
    the rig can decide on.
    """
    host, _, tail = display.rpartition(":")
    if not tail:
        return False
    try:
        index = int(tail.split(".", 1)[0])
    except ValueError:
        return False
    if host in ("", "unix"):
        if not hasattr(socket, "AF_UNIX"):
            # A bare ":N" names a local X11 socket, which is reachable only
            # through AF_UNIX, and Windows CPython exposes no such family.
            # So the display cannot be live here, and saying so is the whole
            # answer rather than a fallback. The remote "host:N" form below
            # is AF_INET and stays available on every platform.
            return False
        path = f"/tmp/.X11-unix/X{index}"
        # A Linux server may listen on the abstract namespace, the
        # filesystem socket, or both, and a container can expose one
        # without the other.
        for addr in (path, "\0" + path):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            try:
                sock.connect(addr)
                return True
            except OSError:
                continue
            finally:
                sock.close()
        return False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        sock.connect((host, 6000 + index))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _start_xvfb() -> Optional[str]:
    """Start an Xvfb sized to hold the rig's window.

    Returns its DISPLAY, or None when Xvfb is not installed, leaving the
    caller to decide whether a real display can stand in.
    """
    global _xvfb, _xvfb_display
    # One server per process. Without this a second ensure_display()
    # would start another and drop the reference to the first, leaving
    # it running until the interpreter exits.
    if _xvfb is not None and _xvfb.poll() is None:
        return _xvfb_display
    exe = shutil.which("Xvfb")
    if not exe:
        return None
    x, y, w, h = window_geometry() or (0, 0, 1280, 1024)
    screen_w, screen_h = max(1280, x + w), max(1024, y + h)

    # Search upward from :100 rather than letting the server choose,
    # which starts at :0. A throwaway framebuffer sitting on the
    # conventional primary display would silently collect any client
    # that falls back to DISPLAY=:0. Each attempt is atomic (the server
    # binds that display or exits), so two rig runs racing for the same
    # number costs one retry rather than a shared display. -displayfd
    # still carries the readiness signal: the number arrives only once
    # the server is accepting connections.
    last_err = ""
    for number in range(_XVFB_DISPLAY_BASE,
                        _XVFB_DISPLAY_BASE + _XVFB_DISPLAY_TRIES):
        proc, err = _spawn_xvfb(exe, number, screen_w, screen_h)
        if proc is not None:
            _xvfb = proc
            display = f":{number}"
            if not _display_is_live(display):
                shutdown_display()
                last_err = f"{display} was reported ready but refuses connections"
                continue
            atexit.register(shutdown_display)
            _xvfb_display = display
            return display
        last_err = err
    raise RuntimeError(
        f"could not start Xvfb on any display in "
        f":{_XVFB_DISPLAY_BASE}..:{_XVFB_DISPLAY_BASE + _XVFB_DISPLAY_TRIES - 1}"
        f"; last error: {last_err}"
    )


def _spawn_xvfb(exe: str, number: int, screen_w: int,
                screen_h: int) -> tuple[Optional[subprocess.Popen], str]:
    """Try to start Xvfb on display *number*.

    Returns ``(proc, "")`` once the server reports itself ready, or
    ``(None, reason)`` if it exited or never reported.
    """
    read_fd, write_fd = os.pipe()
    os.set_inheritable(write_fd, True)
    try:
        proc = subprocess.Popen(
            [exe, f":{number}", "-displayfd", str(write_fd),
             "-screen", "0", f"{screen_w}x{screen_h}x24",
             "-nolisten", "tcp"],
            pass_fds=(write_fd,),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            start_new_session=True,
        )
    finally:
        os.close(write_fd)

    # select() rather than a blocking read: an Xvfb that starts but never
    # reports would otherwise hang the run instead of hitting the deadline.
    buf = b""
    deadline = time.monotonic() + 20.0
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                err = (proc.stderr.read() or b"").decode("utf-8", "replace")
                return None, (f"Xvfb :{number} exited with "
                              f"{proc.returncode}: {err.strip()[-300:]}")
            if not select.select([read_fd], [], [], 0.25)[0]:
                continue
            chunk = os.read(read_fd, 64)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                return proc, ""
    finally:
        os.close(read_fd)

    _terminate(proc)
    return None, f"Xvfb :{number} did not report readiness within 20s"


def ensure_display() -> Optional[str]:
    """Give this process an X display for the rig's Blender to open into.

    Returns the DISPLAY now in effect, or None where the question does
    not arise: macOS and Windows always have a window server, and
    background mode opens no window at all.

    On Linux the rig gets its OWN Xvfb by default, even when a desktop
    session is available. Rendering onto the developer's desktop would
    put a window per worker in front of whatever they are doing, and it
    makes the display size a property of the machine, so the same run
    behaves differently on a laptop and a CI runner. A private server
    fixes both. Set ``PPF_BLENDER_DISPLAY=inherit`` to use the ambient
    DISPLAY instead, which is what to reach for when you want to WATCH
    the scenarios drive the UI.
    """
    if os.environ.get("PPF_BLENDER_HEADLESS") == "1":
        return None
    if not sys.platform.startswith("linux"):
        return os.environ.get("DISPLAY")

    ambient = os.environ.get("DISPLAY", "")
    if (os.environ.get("PPF_BLENDER_DISPLAY") or "").lower() == "inherit":
        if not ambient or not _display_is_live(ambient):
            raise RuntimeError(
                "PPF_BLENDER_DISPLAY=inherit but DISPLAY is "
                f"{'unset' if not ambient else ambient + ' and unreachable'}"
            )
        print(f"[blender] using the ambient display {ambient}", flush=True)
        return ambient

    display = _start_xvfb()
    if display:
        os.environ["DISPLAY"] = display
        print(f"[blender] rig display: Xvfb on {display}", flush=True)
        return display

    # No Xvfb installed. A live desktop display still runs the suite, so
    # say plainly what that costs rather than refusing outright.
    if ambient and _display_is_live(ambient):
        print(f"[blender] Xvfb not installed; using {ambient}, so a window "
              f"will appear per worker. Install it (apt-get install xvfb) "
              f"to keep the rig off the desktop.", flush=True)
        return ambient
    raise RuntimeError(
        "no X display for Blender: Xvfb is not installed (apt-get install "
        "xvfb) and DISPLAY is "
        f"{'unset' if not ambient else ambient + ', which refuses connections'}"
    )


def shutdown_display() -> None:
    """Stop an Xvfb this process started. Idempotent."""
    global _xvfb, _xvfb_display
    proc, _xvfb, _xvfb_display = _xvfb, None, ""
    if proc is not None:
        _terminate(proc)


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
        with open(DRIVER_PATH, encoding="utf-8") as f:
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


# In --background (headless) mode there is no event loop, so bpy.app.timers
# callbacks never fire. Run _start synchronously: the rig driver holds the
# main thread and drains its own PC2 frames via fetch_and_drain (direct
# apply_animation calls), so no modal / event-loop tick is required. This is
# what lets the real-GPU jobs run the rig headless (no OpenGL/desktop needed,
# sidestepping the GPU's TCC-mode lack of WGL) via PPF_BLENDER_HEADLESS=1.
# In UI mode keep the 2s deferral so addon registration + post-load handlers
# settle before the driver runs.
if bpy.app.background:
    _start()
else:
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
    # ``encoding="utf-8"`` is mandatory: scenarios contain em-dashes
    # and other non-ASCII glyphs in comments, and the orchestrator's
    # Python on Windows defaults to cp1252; without this the bootstrap
    # (which reads back as utf-8) chokes on byte 0x97.
    with open(spec.driver_path, "w", encoding="utf-8") as f:
        f.write(spec.driver_source)

    # Drop the bootstrap onto disk so Blender's --python flag can find it.
    bootstrap_path = os.path.join(spec.workspace, "bootstrap.py")
    with open(bootstrap_path, "w", encoding="utf-8") as f:
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
    # Tell the addon's WIN_NATIVE backend to skip its own
    # ``ppf-cts-server.exe`` spawn: the orchestrator already started a
    # rig-owned server for this worker, and the addon would otherwise
    # try to relaunch one (and likely race the existing port binding).
    # The bl_connect_win_native scenario documents this as a hard
    # requirement; setting it here so individual scenarios don't have
    # to opt in.
    env.setdefault("PPF_WIN_NATIVE_NO_SPAWN", "1")

    # Scenario knobs that the addon (not just the server) reads, e.g.
    # PPF_FORCE_TCP_TRANSFER selecting the co-located transport. These
    # win over the inherited environment so a scenario's KNOBS override
    # the rig-wide default.
    for key, value in spec.env_extra.items():
        env[key] = str(value)

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
    args = [spec.blender_bin]
    if spec.blend_file:
        args.append(spec.blend_file)
    if os.environ.get("PPF_BLENDER_HEADLESS") == "1":
        args.append("--background")
    else:
        # Window flags are meaningless in background mode (no window is
        # created) and Blender warns about them, so they belong only on
        # the UI branch.
        args.extend(window_args())
    args.extend([
        "--factory-startup",
        "--addons", spec.addon_name,
        "--python", bootstrap_path,
    ])

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
                    timeout: float = 120.0,
                    post_write_drain: float = 35.0) -> dict:
    """Block until ``spec.result_path`` is written and Blender exits.

    Returns the parsed result dict. Raises ``TimeoutError`` if the
    result file does not appear within ``timeout``.

    Once the result file appears, waits up to ``post_write_drain``
    seconds for the Blender process to exit. The bootstrap writes the
    result file BEFORE registering its drain-then-quit timer that lets
    the FramePump modal flush queued PC2 frames to disk; if the host
    side proceeded the moment the file appeared (the prior behavior),
    its diff subprocess could ``os.listdir`` the addon data dir before
    any PC2 had been written. Waiting for proc exit guarantees the
    modal has finished. The bound (35s) is one second above the
    bootstrap's 30s drain deadline; if Blender hangs past it we still
    return the parsed result and let the orchestrator's tree-kill
    cleanup handle the stuck process.
    """
    import json
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(spec.result_path):
            # Wait a beat for the writer to finish flushing.
            parsed: dict | None = None
            for _ in range(20):
                try:
                    with open(spec.result_path) as f:
                        parsed = json.load(f)
                    break
                except (OSError, ValueError):
                    time.sleep(0.05)
            if parsed is None:
                return {"errors": ["result file unreadable"]}
            # Now wait for the process to exit so the modal-driven PC2
            # writes are guaranteed flushed before the caller proceeds.
            exit_deadline = time.monotonic() + post_write_drain
            while time.monotonic() < exit_deadline:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            return parsed
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
    # ``killpg`` is the fast path: signal every descendant of the
    # session we created with ``start_new_session=True``. On macOS we
    # occasionally see ``PermissionError(EPERM)`` here mid-run, even
    # for our own child, after Blender's children have rebound their
    # session. ``ESRCH`` (ProcessLookupError) means the leader already
    # exited; both cases fall back to a per-process ``proc.terminate``
    # + ``proc.kill`` so a single weird shutdown can't crash the
    # whole multiprocessing pool.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


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


