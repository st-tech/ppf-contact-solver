"""Headless-Blender end-to-end test for the Windows-native backend.

Drives the addon's ``CommunicatorFacade`` through a full win_native
lifecycle inside a background (``blender.exe -b``) session:

    connect  →  server detected running  →  stop_server  →  disconnect

Passes only if no error state is produced at any step. This is the
regression guard for the three bash-syntax bugs in
``blender_addon/core/effect_runner.py`` (``_do_validate_path``,
``_do_stop_server``, ``_count_remote_frames``) plus the port-threading
fix in ``ui/connection_ops.py``, all fixed in commit 24a9372d.

Run:

    # Dev layout
    "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" \\
        -b --python C:\\dev\\build-win-native\\scripts\\test_backend_e2e.py

    # Bundle layout (override via env)
    set T2_ROOT=C:\\dev\\build-win-native\\dist && set T2_PORT=9092 && \\
        "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" \\
        -b --python C:\\dev\\build-win-native\\scripts\\test_backend_e2e.py

Env:
    T2_ROOT — solver root directory. Default ``C:\\dev``.
    T2_PORT — socket port. Default ``9091``.

Sub-argv is not used: Blender consumes script args before the ``--``
separator. Env vars are the clean channel.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import traceback

import bpy  # pyright: ignore
import addon_utils  # pyright: ignore


ROOT = os.environ.get("T2_ROOT", r"C:\ppf-contact-solver")
PORT = int(os.environ.get("T2_PORT", "9091"))
PROJECT_NAME = "backend_e2e"

CONNECT_TIMEOUT_S = 30.0
SERVER_DETECT_TIMEOUT_S = 15.0
STOP_TIMEOUT_S = 15.0
DISCONNECT_TIMEOUT_S = 5.0


# Filled in by main() once the addon's module name is known.
_facade = None


def _die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def _dump_state(label: str) -> None:
    s = _facade.engine.state
    print(
        f"  state@{label}: phase={s.phase} activity={s.activity} "
        f"server={s.server} solver={s.solver} error={s.error!r} "
        f"server_error={s.server_error!r}",
        file=sys.stderr, flush=True,
    )


def _pump(predicate, timeout_s: float, label: str) -> None:
    """Call ``facade.tick()`` until ``predicate()`` holds, else fail."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _facade.tick()
        if predicate():
            return
        time.sleep(0.1)
    _dump_state(label)
    _die(f"timeout waiting for {label} (>{timeout_s:.1f}s)")


def _resolve_addon_module() -> str:
    """Return the sys.modules name Blender assigned to the ppf addon.

    The install-blender-addon.ps1 junction is ``ppf-contact-solver``
    (hyphens), so Blender registers the module under that exact name —
    NOT the manifest's ``id = ppf_contact_solver`` (underscores). Since
    hyphens aren't valid identifiers, callers must use
    ``importlib.import_module`` rather than ``from X import Y``.
    """
    for mod in addon_utils.modules():
        name = mod.__name__
        if "ppf" in name.lower() and (
            "contact" in name.lower() or "solver" in name.lower()
        ):
            return name
    _die("could not find ppf addon via addon_utils.modules()")
    return ""  # unreachable — _die exits


def main() -> None:
    global _facade

    addon_name = _resolve_addon_module()
    print(f"addon module: {addon_name}  (root={ROOT}, port={PORT})")

    # Blender auto-enables the addon at startup; this call cycles it off
    # then on. `ensure_engine_timer()` in register() now revives the
    # effect-runner worker (stopped by unregister's cleanup()), so no
    # manual restart is needed here — if the worker is dead after this
    # call, register() itself is broken.
    try:
        bpy.ops.preferences.addon_enable(module=addon_name)
    except Exception:
        pass

    client = importlib.import_module(f"{addon_name}.core.client")
    _facade = importlib.import_module(f"{addon_name}.core.facade")
    com = client.communicator

    if not _facade.runner._worker.is_alive():
        _die("runner worker dead after addon_enable — the register() path "
             "should have called runner.restart() via ensure_engine_timer")

    com.set_project_name(PROJECT_NAME)

    # --- Step 1: Connect ---
    com.connect_win_native(ROOT, PORT)
    _pump(com.is_connected, CONNECT_TIMEOUT_S, "is_connected")
    _facade.tick()  # absorb DoValidateRemotePath + DoQuery after Connected
    if com.info.error:
        _die(f"spurious error after connect: {com.info.error!r}")
    print(f"OK [connect]: session={com.session_id}")

    # --- Step 2: Server detected running ---
    # Connected dispatches DoQuery; if the server is up it transitions
    # Server.UNKNOWN → Server.RUNNING on the next tick.
    _pump(com.is_server_running, SERVER_DETECT_TIMEOUT_S, "is_server_running")
    print("OK [server-up]: Server.RUNNING observed")

    # --- Step 3: Stop server ---
    com.stop_server()
    _pump(lambda: not com.is_server_running(), STOP_TIMEOUT_S, "server-stop")
    if com.info.error:
        _die(f"stop_server produced error: {com.info.error!r}")
    print("OK [stop-server]: Server state cleared")

    # --- Step 4: Disconnect ---
    com.disconnect()
    _pump(lambda: not com.is_connected(), DISCONNECT_TIMEOUT_S, "disconnect")
    print("OK [disconnect]: phase OFFLINE")

    print("PASS test_backend_e2e")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"EXCEPTION: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
