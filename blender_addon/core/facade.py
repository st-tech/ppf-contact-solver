# File: facade.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Module-level singletons (``engine``, ``runner``) plus the
# ``CommunicatorFacade`` exposed as ``communicator``. Operator and UI code
# call methods on the facade; each method translates the call into an event
# dispatched to the Engine, and property accessors read back from
# ``engine.state``.

from __future__ import annotations

from typing import Any

from ..models.defaults import DEFAULT_SERVER_PORT, DEFAULT_SSH_KEEPALIVE_INTERVAL
from .effect_runner import EffectRunner
from .engine import Engine
from .gpu_devices import AUTOMATIC
from .events import (
    AbortRequested,
    BuildRequested,
    ConnectRequested,
    DisconnectRequested,
    ExecRequested,
    FetchRequested,
    KillServerRequested,
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
    """Public API used by operators and UI code.

    Every public method translates the call into an ``Event`` dispatched to
    the ``Engine``. Property accessors read from ``engine.state`` and map
    back to ``CommunicatorInfo`` / ``RemoteStatus`` / etc.
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
        keepalive_interval=DEFAULT_SSH_KEEPALIVE_INTERVAL,
        proxy_jump=None,
        device="GPU",
        gpu_backend="AUTO",
    ):
        """Open an SSH connection, optionally through one or more jump hosts.

        *proxy_jump* is a jump spec in the form ``ssh -J`` takes
        (``[user@]host[:port]``, comma separated for a chain). When it is
        empty the ``ProxyJump`` entry the ssh config gives for *host* is used,
        so an alias that is only reachable from a bastion connects with
        nothing typed into the panel. Raises ValueError on a malformed spec.

        *device* and *gpu_backend* are the same two answers the native connects
        carry, and they ride here for the same reason: they name WHICH BUILD on
        the solver host a run uses, and the launch that applies them happens
        after this call.
        """
        from .ssh_config import resolve_jump_chain, resolve_ssh_config

        config = resolve_ssh_config(host)
        resolved_host = config.hostname
        resolved_port = port if port != 22 else config.port
        resolved_username = username if username else config.user
        resolved_key_path = key_path if key_path else config.identity_file

        jump_spec = proxy_jump if proxy_jump else config.proxy_jump
        jumps = resolve_jump_chain(jump_spec) if jump_spec else []

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
                "jumps": [
                    {
                        "host": hop.hostname,
                        "port": hop.port,
                        "username": hop.user,
                        "key_path": hop.identity_file,
                    }
                    for hop in jumps
                ],
                "device": device,
                "gpu_backend": gpu_backend,
            },
            server_port=server_port,
        ))

    def connect_docker(
        self,
        container,
        path,
        server_port=DEFAULT_SERVER_PORT,
        device="GPU",
        gpu_backend="AUTO",
    ):
        # Same contract as `connect_ssh`; see the note there.
        self._dispatch_and_tick(ConnectRequested(
            backend_type="docker",
            config={
                "container": container,
                "path": path,
                "device": device,
                "gpu_backend": gpu_backend,
            },
            server_port=server_port,
        ))

    def connect_win_native(
        self, path, port=DEFAULT_SERVER_PORT, device="GPU", gpu_backend="AUTO"
    ):
        # `device` rides in `config` beside the path because both are answers
        # about WHICH SERVER to start, and the effect runner reads them
        # together. It defaults to GPU so every existing caller, and every
        # `.blend` saved before the property existed, keeps its behavior.
        # `gpu_backend` is the second half of that answer where a root holds
        # more than one GPU build, and defaults to the automatic rule.
        self._dispatch_and_tick(ConnectRequested(
            backend_type="win_native",
            config={"path": path, "device": device, "gpu_backend": gpu_backend},
            server_port=port,
        ))

    def connect_mac_native(self, path, port=DEFAULT_SERVER_PORT, device="GPU"):
        # Same contract as `connect_win_native`; see the note there.
        self._dispatch_and_tick(ConnectRequested(
            backend_type="mac_native",
            config={"path": path, "device": device},
            server_port=port,
        ))

    def connect_linux_native(
        self, path, port=DEFAULT_SERVER_PORT, device="GPU", gpu_backend="AUTO"
    ):
        # Same contract as `connect_win_native`, and it carries the same two
        # answers: a Linux x86_64 distribution ships CUDA and ROCm together.
        self._dispatch_and_tick(ConnectRequested(
            backend_type="linux_native",
            config={"path": path, "device": device, "gpu_backend": gpu_backend},
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

    def is_server_stopping(self) -> bool:
        return self._engine.state.server == Server.STOPPING

    def is_aborting(self) -> bool:
        return self._engine.state.activity == Activity.ABORTING

    # -- server lifecycle --

    def start_server(
        self, cuda_device=AUTOMATIC, cuda_device_uuid="", device="", gpu_backend=""
    ):
        # `device` and `gpu_backend` name which BUILD a remote server comes out
        # of. They are read at Start Server rather than held from connect,
        # because that is when the panel's rows have been drawn against the
        # solver host's own listing and the artist has had a chance to move
        # them. Empty keeps whatever the backend holds.
        self._dispatch_and_tick(StartServerRequested(
            cuda_device=cuda_device,
            cuda_device_uuid=cuda_device_uuid,
            device=device,
            gpu_backend=gpu_backend,
        ))

    def refresh_solver_host_gpus(self):
        """Re-enumerate the connected solver host's GPUs and solver builds.

        Not an event: it changes no application state, it refills the two
        caches the panel reads, and it has to run on the worker thread that
        owns the connection rather than in the operator that asked for it.

        BOTH ARE REFRESHED BY THE ONE BUTTON, because both answer "what can
        this solver host run" and an artist who just built a backend there, or
        just freed a GPU, means the same thing by pressing it.
        """
        self._runner.probe_solver_host_gpus()
        self._runner.probe_solver_host_builds()

    def stop_server(self):
        self._dispatch_and_tick(StopServerRequested())

    def kill_server(self):
        """End the server through the live backend, whatever the engine
        believes about it. Leaves the state the way Stop Server does: connected
        to the host, server UNKNOWN, so Start Server is the next step."""
        self._dispatch_and_tick(KillServerRequested())

    @property
    def last_kill_report(self):
        """What the most recent stop or kill found and did, or ``None``."""
        return self._runner.last_kill_report

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

    def resume(self, context=None, from_frame=None):
        if context:
            from ..models.groups import get_addon_data
            fetched = get_addon_data(context.scene).state.convert_fetched_frames_to_list()
            if from_frame is not None:
                # Drop locally-cached frames past the resume point: the
                # solver overwrites the tail from from_frame onward, so
                # the post-resume fetch must re-pull those frames.
                fetched = [f for f in fetched if f <= from_frame]
            self._runner.set_fetched_frames(fetched)
        self._dispatch_and_tick(ResumeRequested(from_frame=from_frame))

    def saved_state_frames(self) -> list[int]:
        """Resumable checkpoint frames from the latest server response.

        Reads the ``saved_states`` array the server attaches to every
        status response when a root is set (see
        ``response::build_response``). Returns a sorted, de-duplicated
        list of frame indices, skipping any malformed entry. Empty when
        no checkpoint has been saved yet (or no response cached).
        """
        raw = self._runner._response_cache.get("saved_states", []) or []
        frames = []
        for n in raw:
            try:
                frames.append(int(n))
            except (TypeError, ValueError):
                continue
        return sorted(set(frames))

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
                       data_hash="", param_hash="", message="",
                       preserve_output=False):
        """Atomic upload + build in a single engine-driven pipeline.

        Replaces the old modal-orchestrated data_send → param_send →
        build chain. Either payload may be empty (params-only update),
        but at least one must be non-empty. The two ``*_hash`` fields
        are the client's quick fingerprints (see
        ``encoder.mesh.compute_data_hash`` and
        ``encoder.params.compute_param_hash``); pass empty for whichever
        payload is itself empty.

        ``preserve_output=True`` requests a resume-rebuild that keeps the
        ``session/output/`` checkpoints in place so a resume can re-decode
        edited scene input without discarding already-simulated frames;
        the default ``False`` is a fresh build that wipes the output dir.
        """
        from .events import BuildPipelineRequested
        self._dispatch_and_tick(BuildPipelineRequested(
            data=data, param=param,
            data_hash=data_hash, param_hash=param_hash, message=message,
            preserve_output=preserve_output,
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
            # getattr keeps this safe if a backend lacks the property; it is
            # the authoritative source for docker-over-ssh detection.
            info.container = getattr(self._runner.backend, "container", "")
        return info

    def normalized_remote_root(self) -> str:
        """Remote root with any trailing slash stripped, ``''`` when unset.

        Only the native backends, win_native, mac_native and linux_native,
        normalize the root at connect time; the ssh and docker backends pass
        the raw user-typed path through, so a trailing slash would yield a
        double slash when joined into an f-string. Callers also treat
        ``''`` as "not connected".
        """
        return self.connection.remote_root.rstrip("/")

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
    def crash_kind(self) -> str:
        """Stable snake_case name of the cause behind ``server_error``.

        Empty when the error is not a solver crash, or when the connected
        server predates the field. The panel uses it to pick a localized
        one-line headline; the full report stays in ``server_error``.
        """
        return self._engine.state.crash_kind

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

    def has_pending_animation_frames(self) -> bool:
        return self._runner.has_pending_animation_frames()

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
_TICK_INTERVAL_S = 0.25  # persistent-timer poll cadence
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
        return _TICK_INTERVAL_S
    try:
        # Start the frame-pump modal whenever there is work for it, so
        # apply_animation + MESH_CACHE heal run without user
        # intervention after a file-open or reload teardown cancels it.
        # ensure_modal_running is a no-op while the pump has nothing to
        # do, which is what leaves Blender free to auto-save: Blender
        # skips auto-save for as long as any modal handler is attached.
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
        #
        # Keep the SSH panel current. REMOTE_OT_Connect's modal finishes
        # at the end of the handshake rather than holding a modal handler
        # open for the connection's lifetime, which would stop Blender
        # autosaving for the whole session, so this tick owns the refresh
        # afterwards. It only tags a redraw when the watched status
        # actually changed, and runs at the same 0.25s cadence the modal
        # used, so it must sit ahead of the idle early-return below: a
        # connection can come up and go down with the engine idle.
        try:
            from ..ui.main_panel import refresh_ssh_panel
            refresh_ssh_panel()
        except Exception:
            pass
        if _engine_is_idle():
            _watchdog_reset()
            return _TICK_INTERVAL_S
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
    return _TICK_INTERVAL_S  # Re-run every 0.25s


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
        bpy.app.timers.register(_persistent_tick, first_interval=_TICK_INTERVAL_S, persistent=True)
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
