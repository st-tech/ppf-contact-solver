# File: facade.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Backward-compatible facade that wraps the new event-driven Engine.
#
# The existing codebase accesses the communicator singleton via::
#
#     from ..core.client import communicator as com
#     com.connect_ssh(...)
#     com.info.status
#     com.is_connected()
#
# This module provides ``engine`` and ``runner`` singletons plus a
# ``CommunicatorFacade`` that exposes the same public API as the old
# ``Communicator`` but delegates to the Engine internally.
#
# Migration strategy:
# 1. New code imports ``engine``/``runner`` directly and dispatches events.
# 2. Old code continues using ``communicator`` via the facade -- no changes needed.
# 3. Once all callers are migrated, the facade can be removed.

from __future__ import annotations

from typing import Any

from ..models.defaults import DEFAULT_SERVER_PORT
from .effect_runner import EffectRunner
from .engine import Engine
from .events import (
    AbortRequested,
    BuildRequested,
    ConnectRequested,
    DisconnectRequested,
    ExecRequested,
    FetchRequested,
    QueryRequested,
    ReceiveDataRequested,
    ResumeRequested,
    RunRequested,
    SaveAndQuitRequested,
    SendDataRequested,
    StartServerRequested,
    StopServerRequested,
    TerminateRequested,
)
from .state import Activity, AppState, Phase, Server
from .status import CommunicatorInfo, ConnectionInfo, RemoteStatus

# ---------------------------------------------------------------------------
# Module-level singletons (survive reloads — created once, reused)
# ---------------------------------------------------------------------------

# Use a hidden attribute on the module to detect reloads.
# On first import, create singletons. On reload, stop old worker
# and create fresh ones so no zombie threads accumulate.
import sys as _sys
_this = _sys.modules[__name__]

if hasattr(_this, '_engine_instance'):
    # Reload: stop old worker before creating new one
    try:
        _this._runner_instance.stop()
    except Exception:
        pass

_this._engine_instance = Engine()
_this._runner_instance = EffectRunner(_this._engine_instance)

engine = _this._engine_instance
runner = _this._runner_instance


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class CommunicatorFacade:
    """Drop-in replacement for the old ``Communicator`` class.

    Every public method translates the call into an ``Event`` dispatched to
    the ``Engine``.  Property accessors read from ``engine.state`` and map
    back to the legacy types (``CommunicatorInfo``, ``RemoteStatus``, etc.).
    """

    def __init__(self, eng: Engine, rnr: EffectRunner) -> None:
        self._engine = eng
        self._runner = rnr

    # -- project name --

    def set_project_name(self, name: str) -> None:
        self._runner.project_name = name

    @property
    def project_name(self) -> str | None:
        return self._runner.project_name

    # -- connection --

    def connect_ssh(
        self,
        host,
        port,
        username,
        key_path,
        path,
        container=None,
        server_port=DEFAULT_SERVER_PORT,
        keepalive_interval=30,
    ):
        from .ssh_config import resolve_ssh_config

        config = resolve_ssh_config(host)
        resolved_host = config.hostname
        resolved_port = port if port != 22 else config.port
        resolved_username = username if username else config.user
        resolved_key_path = key_path if key_path else config.identity_file

        self._dispatch_and_tick(ConnectRequested(
            backend_type="ssh",
            config={
                "host": resolved_host,
                "port": resolved_port,
                "username": resolved_username,
                "key_path": resolved_key_path,
                "path": path,
                "container": container or "",
                "keepalive_interval": keepalive_interval,
            },
            server_port=server_port,
        ))

    def connect_docker(self, container, path, server_port=DEFAULT_SERVER_PORT):
        self._dispatch_and_tick(ConnectRequested(
            backend_type="docker",
            config={"container": container, "path": path},
            server_port=server_port,
        ))

    def connect_local(self, path, server_port=DEFAULT_SERVER_PORT):
        self._dispatch_and_tick(ConnectRequested(
            backend_type="local",
            config={"path": path},
            server_port=server_port,
        ))

    def connect_win_native(self, path, port):
        self._dispatch_and_tick(ConnectRequested(
            backend_type="win_native",
            config={"path": path},
            server_port=port,
        ))

    def disconnect(self):
        self._dispatch_and_tick(DisconnectRequested())

    def is_connected(self) -> bool:
        return self._engine.state.phase == Phase.ONLINE

    def is_connecting(self) -> bool:
        return self._engine.state.phase == Phase.CONNECTING

    def is_server_running(self) -> bool:
        return self._engine.state.server == Server.RUNNING

    def is_server_launching(self) -> bool:
        return self._engine.state.server == Server.LAUNCHING

    def is_aborting(self) -> bool:
        return self._engine.state.activity == Activity.ABORTING

    # -- server lifecycle --

    def start_server(self):
        self._dispatch_and_tick(StartServerRequested())

    def stop_server(self):
        self._dispatch_and_tick(StopServerRequested())

    # -- solver operations --

    def _dispatch_and_tick(self, event):
        """Dispatch an event and immediately process the queue.

        This ensures the state is updated before the calling operator
        checks ``busy()`` or ``is_complete()`` in the same frame.
        """
        self._engine.dispatch(event)
        tick()

    def build(self):
        self._dispatch_and_tick(BuildRequested())

    def run(self, context=None):
        if context:
            from ..models.groups import get_addon_data
            get_addon_data(context.scene).state.clear_fetched_frames()
            self._runner.clear_fetched_frames()
        self._dispatch_and_tick(RunRequested())

    def resume(self, context=None):
        if context:
            from ..models.groups import get_addon_data
            fetched = get_addon_data(context.scene).state.convert_fetched_frames_to_list()
            self._runner.set_fetched_frames(fetched)
        self._dispatch_and_tick(ResumeRequested())

    def fetch(self, context=None):
        if context:
            from ..models.groups import get_addon_data
            fetched = get_addon_data(context.scene).state.convert_fetched_frames_to_list()
            self._runner.set_fetched_frames(fetched)
        # Animation buffer reset runs via DoResetAnimationBuffer emitted by
        # the FetchRequested transition — no direct mutation here.
        self._dispatch_and_tick(FetchRequested())

    def abort(self):
        if self._engine.state.activity == Activity.BUILDING:
            from .effects import DoQuery
            self._runner.execute(DoQuery(request={"request": "cancel_build"}))
        self._dispatch_and_tick(AbortRequested())

    def terminate(self):
        self._dispatch_and_tick(TerminateRequested())

    def save_and_quit(self):
        self._dispatch_and_tick(SaveAndQuitRequested())

    # -- data transfer --

    def data_send(self, remote_path, data, message=""):
        self._dispatch_and_tick(SendDataRequested(
            remote_path=remote_path, data=data, message=message,
        ))

    def build_pipeline(self, data=b"", param=b"",
                       data_hash="", param_hash="", message=""):
        """Atomic upload + build in a single engine-driven pipeline.

        Replaces the old modal-orchestrated data_send → param_send →
        build chain. Either payload may be empty (params-only update),
        but at least one must be non-empty. The two ``*_hash`` fields
        are the client's quick fingerprints (see
        ``encoder.mesh.compute_data_hash`` and
        ``encoder.params.compute_param_hash``); pass empty for whichever
        payload is itself empty.
        """
        from .events import BuildPipelineRequested
        self._dispatch_and_tick(BuildPipelineRequested(
            data=data, param=param,
            data_hash=data_hash, param_hash=param_hash, message=message,
        ))

    def upload_only(self, data=b"", param=b"",
                    data_hash="", param_hash="", message=""):
        """Atomic upload of data/param without chaining a build.

        Used by the debug "Transfer without Build" operator that exercises
        the upload path in isolation. Either payload may be empty but at
        least one must be non-empty.
        """
        from .events import UploadOnlyRequested
        self._dispatch_and_tick(UploadOnlyRequested(
            data=data, param=param,
            data_hash=data_hash, param_hash=param_hash, message=message,
        ))

    def data_receive(self, remote_path, message=""):
        self._dispatch_and_tick(ReceiveDataRequested(
            remote_path=remote_path, message=message,
        ))

    # -- exec --

    def exec(self, command, shell=False):
        self._dispatch_and_tick(ExecRequested(command=command, shell=shell))

    # -- query --

    def query(self, args=None, message=""):
        self._dispatch_and_tick(QueryRequested(
            request=args or {}, message=message,
        ))

    # -- status accessors (backward-compatible) --

    @property
    def info(self) -> CommunicatorInfo:
        s = self._engine.state
        return CommunicatorInfo(
            status=s.to_remote_status(),
            message=s.message,
            error=s.error,
            server_error=s.server_error,
            violations=list(s.violations),
            response=dict(self._runner._response_cache.last_response),
            progress=s.progress,
            traffic=s.traffic,
        )

    @property
    def connection(self) -> ConnectionInfo:
        """Legacy connection info.  Returns a partially-populated ConnectionInfo."""
        s = self._engine.state
        info = ConnectionInfo()
        if self._runner.backend:
            info.type = self._runner.backend.backend_type
            info.current_directory = self._runner.backend.current_directory
            info.server_port = self._runner.backend.server_port
            info.server_running = s.server == Server.RUNNING
            info.remote_root = s.remote_root
            info.instance = self._runner.backend  # For SSH alive check
        return info

    @property
    def response(self) -> dict[str, Any]:
        """Most recent raw server response (for UI display only).

        Reads from the runner's ResponseCache, which is treated as
        last-seen with no freshness guarantee.  The authoritative
        interpreted fields live on ``AppState`` (server, solver,
        activity, frame, violations, message, progress) — prefer
        those for logic.
        """
        return dict(self._runner._response_cache.last_response)

    @property
    def message(self) -> str:
        return self._engine.state.message

    @property
    def error(self) -> str:
        return self._engine.state.error

    @property
    def server_error(self) -> str:
        return self._engine.state.server_error

    @property
    def session_id(self) -> str:
        """Identifier stamped on artifacts produced by this connected run.

        Empty when offline.  Regenerated on every successful connect.
        """
        return self._engine.state.session_id

    def last_saved_session_id(self) -> str:
        """Session id stored in the scene at last save, or empty string.

        Returns '' when no addon data is attached to the active scene.
        Used by reconnect logic to detect orphaned remote sims.
        """
        try:
            import bpy  # pyright: ignore
            from ..models.groups import get_addon_data, has_addon_data
            scene = bpy.context.scene
            if not has_addon_data(scene):
                return ""
            return get_addon_data(scene).state.last_session_id or ""
        except Exception:
            return ""

    def set_error(self, error_msg: str):
        from .events import ErrorOccurred
        self._engine.dispatch(ErrorOccurred(error=error_msg))

    def busy(self) -> bool:
        return self._engine.state.busy

    def busy_guard(self):
        if self.busy():
            raise RuntimeError("Communicator is busy.")

    # -- animation --

    def take_one_animation_frame(self):
        return self._runner.take_one_animation_frame()

    @property
    def animation(self):
        """Legacy animation data accessor."""
        from .status import AnimationData
        with self._runner._anim_lock:
            return AnimationData(
                map=self._runner._anim_map,
                frame=list(self._runner._anim_frames),
                surface_map=self._runner._anim_surface_map,
                total_frames=self._runner._anim_total,
                applied_frames=self._runner._anim_applied,
            )

    @property
    def exec_output(self):
        return self._runner.exec_output

    @property
    def data(self):
        return self._runner.received_data


# ---------------------------------------------------------------------------
# Engine tick — driven by Blender's main-thread timer
# ---------------------------------------------------------------------------

def tick() -> None:
    """Process all pending events.  Runs on Blender's main thread."""
    engine.tick(runner)


_persistent_timer_registered = False

# Gate for the persistent tick body. False until register() completes and
# after unregister() starts — protects against Blender firing the timer
# while the addon is mid-unregister (classes partially torn down,
# PropertyGroup RNA invalidated) or during reload, which has caused
# Blender to segfault when the heal pass reads half-freed state.
_addon_ready = False


def mark_addon_ready(ready: bool) -> None:
    global _addon_ready
    _addon_ready = bool(ready)


_last_tick_status = [None]

# Liveness watchdog: tracks (activity, rounded-progress) on each tick so we
# can detect a fetch/apply that has stalled. ``_stuck_since`` is the
# monotonic timestamp when the snapshot last advanced; if it sits longer
# than _WATCHDOG_TIMEOUT_S we dispatch FetchFailed.
_WATCHDOG_TIMEOUT_S = 30.0
_last_progress_key: tuple | None = None
_stuck_since: float = 0.0


def _engine_is_idle() -> bool:
    """True when no work is expected: no queued events, state is idle,
    and the solver is not actively producing frames."""
    if engine.has_pending():
        return False
    s = engine.state
    from .state import Activity, Solver
    if s.activity != Activity.IDLE:
        return False
    if s.solver in (Solver.RUNNING, Solver.STARTING, Solver.SAVING, Solver.BUILDING):
        return False
    return True


def _watchdog_reset() -> None:
    global _last_progress_key, _stuck_since
    _last_progress_key = None
    _stuck_since = 0.0


def _watchdog_check() -> None:
    """If activity has been FETCHING or APPLYING at the same progress
    value for longer than _WATCHDOG_TIMEOUT_S, dispatch a FetchFailed so
    the state machine can recover instead of hanging indefinitely."""
    import time as _time
    from ..models.console import console
    from .state import Activity
    from .events import FetchFailed
    global _last_progress_key, _stuck_since

    s = engine.state
    if s.activity not in (Activity.FETCHING, Activity.APPLYING):
        _watchdog_reset()
        return

    key = (s.activity, round(s.progress, 3))
    now = _time.monotonic()
    if key != _last_progress_key:
        _last_progress_key = key
        _stuck_since = now
        return
    if now - _stuck_since <= _WATCHDOG_TIMEOUT_S:
        return
    console.write(
        f"[watchdog] fetch stalled: {s.activity.name} "
        f"progress={s.progress:.2f} for {_WATCHDOG_TIMEOUT_S:.0f}s"
    )
    engine.dispatch(FetchFailed(reason="watchdog timeout"))
    # Reset so we don't spam the event queue if the timeout fires again
    # before FetchFailed is processed by the next tick.
    _stuck_since = now


def _persistent_tick() -> float:
    """Blender persistent timer callback. Returns interval for next call."""
    # If the addon isn't fully registered (startup, unregister, reload
    # crossover), do nothing. Touching PropertyGroup state during an
    # active reload can segfault Blender.
    if not _addon_ready:
        return 0.25
    try:
        # Keep the frame-pump modal alive. It can die on file-open or
        # reload teardown; this re-invokes it so apply_animation +
        # MESH_CACHE heal keep running without user intervention.
        try:
            from . import frame_pump
            frame_pump.ensure_modal_running()
        except Exception:
            pass
        # Note: apply_animation() and heal_mesh_caches_if_stale() are NOT
        # called here. Blender 5.x denies ID writes (State PropertyGroup,
        # modifier.cache_format, scene.frame_start) from timer callbacks.
        # Those are driven from PPF_OT_FramePump.modal() instead, whose
        # modal-operator timer events run in a permissive context. This
        # tick only does Python-side engine polling and event dispatch.
        if _engine_is_idle():
            _watchdog_reset()
            return 0.25
        from .events import PollTick
        engine.dispatch(PollTick())
        tick()
        _watchdog_check()
        # Auto-redraw when status changes
        current = engine.state.to_remote_status()
        if current != _last_tick_status[0]:
            _last_tick_status[0] = current
            from .utils import redraw_all_areas
            import bpy  # pyright: ignore
            redraw_all_areas(bpy.context)
    except Exception as e:
        import logging
        logging.error(f"Engine tick error: {e}")
    return 0.25  # Re-run every 0.25s


def ensure_engine_timer() -> None:
    """Register the persistent Blender timer and revive the I/O worker.

    Called on every addon register() (startup and after disable→enable).

    Also calls ``runner.restart()`` because ``cleanup()`` stops the
    worker thread but the module-level singleton init only re-runs on
    a full sys.modules reload — a plain Blender enable cycle leaves the
    singleton cached with a dead worker, so commands queue forever.
    """
    global _persistent_timer_registered
    runner.restart()
    if not _persistent_timer_registered:
        import bpy  # pyright: ignore
        bpy.app.timers.register(_persistent_tick, first_interval=0.25, persistent=True)
        _persistent_timer_registered = True


def cleanup() -> None:
    """Stop worker threads and deregister timers. Called on addon unregister/reload."""
    global _persistent_timer_registered
    import bpy  # pyright: ignore
    # Flag the body as unsafe FIRST so any in-flight timer tick that
    # sneaks in during teardown early-returns, and deregister the timer
    # so no new ticks fire against the half-torn-down state.
    mark_addon_ready(False)
    if bpy.app.timers.is_registered(_persistent_tick):
        try:
            bpy.app.timers.unregister(_persistent_tick)
        except ValueError:
            pass
    _persistent_timer_registered = False
    runner.stop()


# ---------------------------------------------------------------------------
# Module-level facade singleton
# ---------------------------------------------------------------------------

communicator = CommunicatorFacade(engine, runner)
