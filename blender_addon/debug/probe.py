# File: probe.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Blender-side monitor for the debug test rig. Hooks event handlers,
# wraps timer registration, samples addon state at 10 Hz, and writes a
# JSONL event stream + a final summary. Imported only inside Blender; no
# Blender import at module scope so the file stays linter-clean from the
# host orchestrator.
#
# Usage from inside Blender (e.g. via debug exec):
#
#     from ppf_contact_solver.debug import probe
#     probe.start(workspace_dir="/path/to/worker-NN/probe")
#     ...drive the addon...
#     summary = probe.stop()
#
# Output files under ``<workspace_dir>``:
#
#   probe_events.jsonl       streamed timestamped events (state samples,
#                            handler fires, modal lifecycles, timer churn)
#   probe_assertions.jsonl   one line per detected violation
#   probe_summary.json       final aggregate written on stop()

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from collections import defaultdict
from typing import Any, Optional


# Tunables. Overridable via env at start time so different scenarios can
# tighten / loosen budgets without editing this file.
_DEFAULTS = {
    "sample_hz": 10.0,
    # Stuck-state budgets (seconds). Beyond these dwell times we record
    # an assertion. Names match the addon's Phase / Server / Solver enums.
    "budget_CONNECTING_s": 30.0,
    "budget_LAUNCHING_s": 20.0,
    "budget_BUILDING_unchanged_s": 10.0,
    "budget_RUNNING_unchanged_s": 8.0,
    # Maximum depsgraph_update_post fires per second before we flag a
    # runaway redraw loop.
    "max_depsgraph_per_s": 100.0,
}


class _Probe:
    """Singleton-ish monitor; one per Blender session is enough."""

    def __init__(self) -> None:
        self.workspace: str = ""
        self.events_path: str = ""
        self.assertions_path: str = ""
        self.summary_path: str = ""
        self.start_time: float = 0.0
        self.running: bool = False
        self.lock = threading.Lock()
        self.cfg: dict[str, Any] = {}

        # State trackers
        self.depsgraph_count: int = 0
        self.depsgraph_window: list[float] = []   # times within last 1s
        self.frame_change_count: int = 0
        self.modal_active: dict[str, int] = defaultdict(int)  # bl_idname -> live count
        self.modal_seen: set[str] = set()
        self.timer_registry: set[Any] = set()     # timer fns we observed
        self.last_phase: tuple[str, str, str] = ("?", "?", "?")
        self.phase_changed_at: float = 0.0
        self.last_progress: float = 0.0
        self.last_progress_at: float = 0.0
        self.last_frame_field: int = -1
        self.last_frame_field_at: float = 0.0

        # Hook handles, kept so stop() can unhook cleanly.
        self._depsgraph_handler = None
        self._frame_pre_handler = None
        self._frame_post_handler = None
        self._wrapped_timer_register = None
        self._original_timer_register = None
        self._sample_timer = None

        # Counters
        self.assertions: list[dict] = []
        self.event_count: int = 0
        self.errors: list[str] = []

    # -- I/O ---------------------------------------------------------------

    def _write_event(self, etype: str, data: dict) -> None:
        line = {
            "ts": round(time.time() - self.start_time, 4),
            "type": etype,
            **data,
        }
        try:
            with open(self.events_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            with self.lock:
                self.event_count += 1
        except OSError as e:
            self.errors.append(f"event write: {e}")

    def _write_assertion(self, kind: str, message: str, **extra: Any) -> None:
        record = {
            "ts": round(time.time() - self.start_time, 4),
            "kind": kind,
            "message": message,
            **extra,
        }
        with self.lock:
            self.assertions.append(record)
        try:
            with open(self.assertions_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            self.errors.append(f"assertion write: {e}")

    # -- Handlers ---------------------------------------------------------

    def _on_depsgraph(self, scene, depsgraph) -> None:
        del scene, depsgraph
        now = time.time()
        with self.lock:
            self.depsgraph_count += 1
            self.depsgraph_window.append(now)
            cutoff = now - 1.0
            while self.depsgraph_window and self.depsgraph_window[0] < cutoff:
                self.depsgraph_window.pop(0)
            rate = len(self.depsgraph_window)
        if rate > self.cfg["max_depsgraph_per_s"]:
            self._write_assertion(
                "runaway_depsgraph",
                f"depsgraph_update_post fired {rate} times in last 1s "
                f"(budget={int(self.cfg['max_depsgraph_per_s'])})",
                rate=rate,
            )

    def _on_frame_change_pre(self, scene, depsgraph) -> None:
        del depsgraph
        try:
            frame = int(scene.frame_current)
        except (AttributeError, RuntimeError, ReferenceError):
            # scene may be a stale reference if the .blend was reloaded
            # mid-handler; default to -1 so the event still records.
            frame = -1
        self._write_event("frame_change_pre", {"frame": frame})

    def _on_frame_change_post(self, scene, depsgraph) -> None:
        del depsgraph
        with self.lock:
            self.frame_change_count += 1
        try:
            frame = int(scene.frame_current)
        except (AttributeError, RuntimeError, ReferenceError):
            # scene may be a stale reference if the .blend was reloaded
            # mid-handler; default to -1 so the event still records.
            frame = -1
        self._write_event("frame_change_post", {"frame": frame})

    # -- Timer wrapping ---------------------------------------------------

    def _wrap_timer_register(self) -> None:
        """Replace ``bpy.app.timers.register`` so we can audit registrations
        the addon makes. Restored in stop()."""
        import bpy  # pyright: ignore

        original = bpy.app.timers.register
        self._original_timer_register = original
        probe = self

        def _wrapped(func, *, first_interval: float = 0.0, persistent: bool = False):
            with probe.lock:
                probe.timer_registry.add(func)
            probe._write_event("timer_register", {
                "name": getattr(func, "__name__", repr(func)),
                "first_interval": first_interval,
                "persistent": persistent,
            })
            return original(func, first_interval=first_interval, persistent=persistent)

        self._wrapped_timer_register = _wrapped
        bpy.app.timers.register = _wrapped  # type: ignore[assignment]

    # -- Sampling loop ----------------------------------------------------

    def _sample(self) -> Optional[float]:
        if not self.running:
            return None
        try:
            self._sample_state()
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"sample: {e}\n{traceback.format_exc()}")
        return 1.0 / float(self.cfg["sample_hz"])

    def _sample_state(self) -> None:
        # Read engine state and communicator info via the existing facade.
        # The probe has no addon-internal knowledge beyond what facade
        # exposes; that keeps it loosely coupled. ImportError is caught
        # because the probe is allowed to run before / without the
        # addon being importable; any other exception is a real bug.
        phase = server = solver = "?"
        progress = 0.0
        message = ""
        frame_field = -1
        traffic = ""
        # Resolve the addon root so this works under both the extension
        # layout (bl_ext.user_default.ppf_contact_solver) and any legacy
        # single-segment layout. We're inside Blender here; if neither
        # is loaded, the ImportError fallback applies.
        try:
            import sys as _sys
            _root = next(
                n.removesuffix(".debug.probe") for n in _sys.modules
                if n.endswith(".debug.probe") and n != "debug.probe"
            )
            facade = __import__(_root + ".core.facade", fromlist=["engine"])
            client_mod = __import__(_root + ".core.client",
                                    fromlist=["communicator"])
            engine = facade.engine
            com = client_mod.communicator
        except (StopIteration, ImportError):
            engine = com = None
        if engine is not None:
            state = engine.state
            phase = str(state.phase)
            server = str(state.server)
            solver = str(state.solver)
            progress = float(state.progress)
            message = str(state.message)
            frame_field = int(state.frame)
        if com is not None:
            traffic = com.info.traffic

        now = time.time()
        sample = {
            "phase": phase, "server": server, "solver": solver,
            "progress": progress, "message": message,
            "frame": frame_field, "traffic": traffic,
        }
        self._write_event("state_sample", sample)

        triple = (phase, server, solver)
        with self.lock:
            if triple != self.last_phase:
                self.last_phase = triple
                self.phase_changed_at = now
                self._write_event("phase_change", {
                    "phase": phase, "server": server, "solver": solver,
                })
            if progress != self.last_progress:
                self.last_progress = progress
                self.last_progress_at = now
            if frame_field != self.last_frame_field:
                self.last_frame_field = frame_field
                self.last_frame_field_at = now
            phase_dwell = now - self.phase_changed_at
            progress_dwell = now - self.last_progress_at
            frame_dwell = now - self.last_frame_field_at

        # Stuck-state budgets.
        budgets = {
            "CONNECTING": self.cfg["budget_CONNECTING_s"],
            "LAUNCHING": self.cfg["budget_LAUNCHING_s"],
        }
        for needle, budget in budgets.items():
            if needle in phase or needle in server or needle in solver:
                if phase_dwell > budget:
                    self._write_assertion(
                        "phase_stuck",
                        f"state contains {needle} for {phase_dwell:.1f}s "
                        f"(budget={budget}s)",
                        triple=triple, dwell=phase_dwell,
                    )
        if "BUILDING" in solver:
            if progress_dwell > self.cfg["budget_BUILDING_unchanged_s"]:
                self._write_assertion(
                    "build_progress_stalled",
                    f"build progress unchanged for {progress_dwell:.1f}s "
                    f"(budget={self.cfg['budget_BUILDING_unchanged_s']}s)",
                    progress=progress,
                )
        if "RUNNING" in solver:
            if frame_dwell > self.cfg["budget_RUNNING_unchanged_s"]:
                self._write_assertion(
                    "run_frame_stalled",
                    f"frame counter unchanged for {frame_dwell:.1f}s "
                    f"(budget={self.cfg['budget_RUNNING_unchanged_s']}s)",
                    frame=frame_field,
                )

    # -- Operator wrapping (modal concurrency) ----------------------------

    def wrap_operator(self, op_class) -> None:
        """Wrap ``op_class.invoke`` and ``op_class.modal`` so the probe
        can count concurrent live modals per ``bl_idname`` and emit a
        violation when two of the same kind run at once.

        Caller passes the operator class explicitly; we don't enumerate
        all operators because some have side effects in their invoke
        that we don't want triggered."""
        bl_idname = getattr(op_class, "bl_idname", op_class.__name__)
        original_invoke = getattr(op_class, "invoke", None)
        original_modal = getattr(op_class, "modal", None)
        probe = self

        if original_invoke is not None:
            def invoke(self, context, event):
                probe._on_modal_enter(bl_idname)
                try:
                    return original_invoke(self, context, event)
                except Exception as exc:
                    probe._on_modal_exit(bl_idname, "raise")
                    raise
            op_class.invoke = invoke  # type: ignore[attr-defined]

        if original_modal is not None:
            def modal(self, context, event):
                result = original_modal(self, context, event)
                if isinstance(result, set) and (
                    {"FINISHED", "CANCELLED"} & result
                ):
                    probe._on_modal_exit(bl_idname, str(result))
                return result
            op_class.modal = modal  # type: ignore[attr-defined]

    def _on_modal_enter(self, bl_idname: str) -> None:
        with self.lock:
            self.modal_active[bl_idname] += 1
            self.modal_seen.add(bl_idname)
            count = self.modal_active[bl_idname]
        self._write_event("modal_enter", {"op": bl_idname, "active": count})
        if count > 1:
            self._write_assertion(
                "modal_concurrent",
                f"two instances of {bl_idname} active at once",
                op=bl_idname, count=count,
            )

    def _on_modal_exit(self, bl_idname: str, result: str) -> None:
        with self.lock:
            if self.modal_active[bl_idname] > 0:
                self.modal_active[bl_idname] -= 1
            count = self.modal_active[bl_idname]
        self._write_event("modal_exit", {
            "op": bl_idname, "active": count, "result": result,
        })

    # -- Lifecycle --------------------------------------------------------

    def start(self, workspace_dir: str, **overrides: Any) -> None:
        if self.running:
            return
        os.makedirs(workspace_dir, exist_ok=True)
        self.workspace = workspace_dir
        self.events_path = os.path.join(workspace_dir, "probe_events.jsonl")
        self.assertions_path = os.path.join(workspace_dir, "probe_assertions.jsonl")
        self.summary_path = os.path.join(workspace_dir, "probe_summary.json")
        # Truncate previous artifacts so a re-run starts clean.
        for p in (self.events_path, self.assertions_path):
            try:
                open(p, "w").close()
            except OSError:
                pass

        self.cfg = dict(_DEFAULTS)
        # Env knobs override defaults.
        for k in self.cfg:
            envname = f"PPF_PROBE_{k.upper()}"
            v = os.environ.get(envname)
            if v is not None:
                try:
                    self.cfg[k] = float(v)
                except ValueError:
                    pass
        self.cfg.update(overrides)

        self.start_time = time.time()
        self.running = True

        import bpy  # pyright: ignore
        self._depsgraph_handler = self._on_depsgraph
        self._frame_pre_handler = self._on_frame_change_pre
        self._frame_post_handler = self._on_frame_change_post
        bpy.app.handlers.depsgraph_update_post.append(self._depsgraph_handler)
        bpy.app.handlers.frame_change_pre.append(self._frame_pre_handler)
        bpy.app.handlers.frame_change_post.append(self._frame_post_handler)

        self._wrap_timer_register()

        bpy.app.timers.register(self._sample, first_interval=0.1)

        self._write_event("probe_started", {"workspace": workspace_dir})

    def stop(self) -> dict:
        if not self.running:
            return {"status": "not_running"}
        self.running = False

        import bpy  # pyright: ignore
        for store, handler in (
            (bpy.app.handlers.depsgraph_update_post, self._depsgraph_handler),
            (bpy.app.handlers.frame_change_pre, self._frame_pre_handler),
            (bpy.app.handlers.frame_change_post, self._frame_post_handler),
        ):
            if handler is not None:
                try:
                    store.remove(handler)
                except ValueError:
                    pass

        if self._original_timer_register is not None:
            bpy.app.timers.register = self._original_timer_register  # type: ignore[assignment]

        # Wait one sample tick so the in-flight tick observes running=False.
        time.sleep(0.2)

        summary = {
            "duration_s": round(time.time() - self.start_time, 3),
            "event_count": self.event_count,
            "depsgraph_total": self.depsgraph_count,
            "frame_change_total": self.frame_change_count,
            "modal_seen": sorted(self.modal_seen),
            "modal_active_at_stop": {
                k: v for k, v in self.modal_active.items() if v
            },
            "assertion_count": len(self.assertions),
            "assertions": self.assertions,
            "errors": self.errors,
            "events_path": self.events_path,
            "assertions_path": self.assertions_path,
        }
        try:
            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except OSError as e:
            summary.setdefault("errors", []).append(f"summary write: {e}")
        self._write_event("probe_stopped", {"assertions": len(self.assertions)})
        return summary


_singleton = _Probe()


# ---------------------------------------------------------------------------
# Module-level shims
# ---------------------------------------------------------------------------

def start(workspace_dir: str, **overrides: Any) -> None:
    """Start probing. Idempotent: a second call is a no-op."""
    _singleton.start(workspace_dir, **overrides)


def stop() -> dict:
    """Stop probing and return the summary dict (also written to disk)."""
    return _singleton.stop()


def emit(event_type: str, **data: Any) -> None:
    """Manually emit an event from a scenario script. Useful for marking
    boundaries (e.g. ``probe.emit("scenario_phase", name="post-transfer")``)."""
    if _singleton.running:
        _singleton._write_event(event_type, data)


def assert_violation(kind: str, message: str, **extra: Any) -> None:
    """Manually record an assertion. Use sparingly: stuck/race detectors
    fire their own; this is for scenario-specific checks."""
    if _singleton.running:
        _singleton._write_assertion(kind, message, **extra)


def wrap_operator(op_class) -> None:
    """Hook ``invoke`` / ``modal`` of an operator class so the probe
    counts concurrent instances. Returns silently when probing is off
    so scenarios can call this unconditionally."""
    if not _singleton.running:
        return
    _singleton.wrap_operator(op_class)


def is_running() -> bool:
    return _singleton.running


def summary_path() -> str:
    return _singleton.summary_path
