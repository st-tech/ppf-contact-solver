# File: server/monitor.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Background solver monitor thread.
#
# Replaces the per-request filesystem checks that the old ``_status()``,
# ``_frame()``, and ``_check_solver_error()`` methods performed on every
# client query.  Instead, this thread runs continuously and dispatches
# events to the engine when it detects state changes.

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

from .state import Solver
from .events import (
    SolverCrashed,
    SolverFinished,
    SolverFrameUpdated,
    SolverSaving,
)

if TYPE_CHECKING:
    from .engine import ServerEngine


SOLVER_STARTUP_GRACE = 3.0  # seconds to wait before declaring solver dead


def _adopt_external_solver(engine: ServerEngine) -> None:
    """Promote engine state to reflect an externally-started solver.

    Loads ``app_state.pickle`` from disk if the engine doesn't hold one
    yet (Jupyter's ``BlenderApp._persist_app_state`` writes it after
    ``make()``), then directly mutates ``engine._state.solver`` to
    RUNNING. Subsequent monitor ticks handle frame updates, save-flag
    detection and the SolverFinished/SolverCrashed transition normally.
    """
    from dataclasses import replace
    from .engine import AppState

    s = engine.state
    if engine.app is None and s.name and s.root:
        loaded = AppState.load(s.name, s.root)
        if loaded is not None:
            engine._app = loaded
    with engine._lock:
        current = engine._state
        if current.solver == Solver.IDLE:
            from .state import Build, Data
            # Promote build to BUILT and data to UPLOADED if we now have
            # an app loaded; otherwise status_string short-circuits to
            # NO_DATA/NO_BUILD even while the external sim runs.
            if engine.app is not None:
                build = Build.BUILT
                data = Data.UPLOADED
            else:
                build = current.build
                data = current.data
            engine._state = replace(
                current, solver=Solver.RUNNING, build=build, data=data,
            )


def start_solver_monitor(engine: ServerEngine, interval: float = 0.25) -> threading.Thread:
    """Start the background solver monitor thread.

    The monitor watches for:
    - Frame count changes (new vert_*.bin files)
    - Solver exit (Utils.busy() goes False)
    - Solver crash (error.log analysis)
    - Save-in-progress flag file

    Returns the daemon thread (already started).
    """
    ctx = _MonitorContext()
    t = threading.Thread(target=_monitor_loop, args=(engine, interval, ctx), daemon=True)
    t.start()
    return t


class _MonitorContext:
    """Mutable context for the monitor thread (not shared with engine)."""
    def __init__(self):
        self.last_solver_state = Solver.IDLE
        self.solver_started_at: float = 0.0


def _monitor_loop(engine: ServerEngine, interval: float, ctx: _MonitorContext) -> None:
    """Main monitor loop. Runs until process exits."""
    while True:
        try:
            _tick(engine, ctx)
        except Exception as e:
            logging.error(f"Monitor error: {e}")
        time.sleep(interval)


def _tick(engine: ServerEngine, ctx: _MonitorContext) -> None:
    """One monitor iteration. Check solver state and dispatch events."""
    s = engine.state

    from frontend import Utils
    # Detect a solver process started outside this server
    # (e.g. by JupyterLab). Promote state to RUNNING so the rest of
    # this function drives frame/finish/crash transitions just as if
    # the server launched it.
    if (
        s.solver == Solver.IDLE
        and not engine.solver_launching
        and Utils.busy()
        and s.root
    ):
        _adopt_external_solver(engine)
        s = engine.state  # re-read after mutation
        # Past-date the start time so the grace-period check below
        # never treats a long-running external solver as "just died".
        ctx.solver_started_at = 0.0

    # Track when solver transitions to RUNNING for grace period
    if s.solver in (Solver.RUNNING, Solver.SAVING) and ctx.last_solver_state == Solver.IDLE:
        ctx.solver_started_at = time.time()
    ctx.last_solver_state = s.solver

    if s.solver not in (Solver.RUNNING, Solver.SAVING):
        return

    # Don't check while the launch effect is still executing
    if engine.solver_launching:
        return

    if not Utils.busy():
        # Grace period: subprocess takes time to start.
        # Don't declare it dead until the grace period has elapsed.
        elapsed = time.time() - ctx.solver_started_at
        if elapsed < SOLVER_STARTUP_GRACE:
            return

        # Final frame-count refresh before declaring the run terminal.
        # The solver writes its last vert_N.bin and then exits; the
        # previous monitor pass saw N-1 vert files and dispatched
        # SolverFrameUpdated(N-1). Without this re-count the very last
        # frame is invisible to clients (state.frame stays one short),
        # which surfaces in bl_live_frame_end_tracking as
        # final_frame_end == FRAME_COUNT-2 instead of FRAME_COUNT-1.
        final_frame = _count_frames(s.root)
        if final_frame != s.frame:
            engine.dispatch(SolverFrameUpdated(frame=final_frame))

        # Solver exited -- determine why
        if s.solver == Solver.SAVING:
            engine.dispatch(SolverFinished(resumable=True))
        else:
            error = _check_solver_error(s.root)
            if error:
                violations = _read_intersection_violations(s.root)
                engine.dispatch(SolverCrashed(error=error, violations=violations))
            else:
                has_saved = _has_saved_frames(engine)
                engine.dispatch(SolverFinished(resumable=has_saved))
    else:
        # Still running -- check for updates
        frame = _count_frames(s.root)
        if frame != s.frame:
            engine.dispatch(SolverFrameUpdated(frame=frame))

        # Detect save-in-progress
        if s.solver != Solver.SAVING and _is_saving(engine):
            engine.dispatch(SolverSaving())


# ---------------------------------------------------------------------------
# Filesystem helpers (only called from monitor thread, not request path)
# ---------------------------------------------------------------------------

def _count_frames(root: str) -> int:
    """Count vert_*.bin files in the output directory."""
    if not root:
        return 0
    vert_dir = os.path.join(root, "session", "output")
    if not os.path.exists(vert_dir):
        return 0
    try:
        max_num = 0
        for fname in os.listdir(vert_dir):
            if fname.startswith("vert_") and fname.endswith(".bin"):
                try:
                    num = int(fname[5:-4])
                    if num > max_num:
                        max_num = num
                except ValueError:
                    continue
        return max_num
    except Exception as e:
        logging.error(f"Failed to count frames: {e}")
        return 0


def _has_saved_frames(engine: ServerEngine) -> bool:
    """Check if the app has saved checkpoint frames."""
    app = engine.app
    if app is None:
        return False
    try:
        return app.resumable()
    except Exception:
        return False


def _is_saving(engine: ServerEngine) -> bool:
    """Check if the save_and_quit flag file exists."""
    app = engine.app
    if app is None:
        return False
    try:
        return app.is_saving_in_progress()
    except Exception:
        return False


def _check_solver_error(root: str) -> str:
    """Analyze error.log and stdout.log for solver crashes.

    Returns an error string if a crash is detected, empty string otherwise.
    """
    if not root:
        return ""
    try:
        session_path = os.path.join(root, "session")
        output_dir = os.path.join(session_path, "output")
        finished_path = os.path.join(output_dir, "finished.txt")

        if os.path.exists(finished_path):
            return ""  # Normal completion

        err_path = os.path.join(session_path, "error.log")
        log_path = os.path.join(session_path, "stdout.log")

        err_lines = []
        log_lines = []
        if os.path.exists(err_path):
            with open(err_path) as f:
                err_lines = f.readlines()
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_lines = f.readlines()[-200:]

        # Empty stderr or just "Terminated" = user terminated, not a crash
        err_content = "".join(err_lines).strip()
        if not err_content or err_content.lower() == "terminated":
            return ""

        # Pattern matching for known failure types
        patterns = [
            ("### intersection detected", "Intersection detected"),
            ("### ccd failed", "CCD failed"),
            ("### cg failed", "Linear solver failed"),
            ("failed to advance", "Solver crashed"),
            ("panic", "Solver panic"),
            ("assert", "Assertion failed"),
        ]
        reason = "Solver crashed unexpectedly"
        for line in log_lines + err_lines:
            low = line.lower().strip()
            for pattern, msg in patterns:
                if pattern in low:
                    reason = msg
                    break

        tail = "".join(log_lines[-32:])
        return (
            f"{reason}\n"
            f"--- Solver Log (last 32 lines) ---\n{tail}"
            f"--- Solver Error Log ---\n{err_content}"
        )
    except Exception as e:
        logging.error(f"_check_solver_error failed: {e}")
        return f"Solver error: {e}"


def _read_intersection_violations(root: str) -> list:
    """Read intersection_records.json written by the solver and convert to violation format."""
    if not root:
        return []
    json_path = os.path.join(root, "session", "output", "intersection_records.json")
    if not os.path.exists(json_path):
        return []
    try:
        import json
        with open(json_path) as f:
            data = json.load(f)
        records = data.get("records", [])
        if not records:
            return []
        entries = []
        for rec in records:
            entries.append({
                "itype": rec.get("type", "unknown"),
                "elem0": rec.get("elem0", 0),
                "elem1": rec.get("elem1", 0),
                "positions0": rec.get("positions0", []),
                "positions1": rec.get("positions1", []),
            })
        return [{
            "type": "runtime_intersection",
            "count": data.get("count", len(entries)),
            "entries": entries,
        }]
    except Exception as e:
        logging.error(f"Failed to read intersection records: {e}")
        return []
