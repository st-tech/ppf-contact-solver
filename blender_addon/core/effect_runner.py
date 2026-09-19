# File: effect_runner.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The impure shell that executes ``Effect`` objects produced by the pure
# ``transition()`` function.
#
# This is the ONLY module that performs real I/O (network, file system,
# ``bpy`` calls).  Background I/O runs on daemon threads; when it completes,
# the thread dispatches a result ``Event`` back into the ``Engine`` queue.
#
# The separation means:
#   - ``transitions.py`` decides what to do (pure, testable)
#   - ``effect_runner.py`` does it (impure, but mechanical)

from __future__ import annotations

import os
import pickle
import posixpath
import threading
import time
from typing import TYPE_CHECKING

import numpy

from ..models.console import console
from .backends import ConnectionBackend, create_backend
from .connection import NATIVE_BACKENDS, remote_target_dir
from .derived import is_sim_running_from_response
from .gpu_devices import AUTOMATIC, describe_launch, shell_prefix
from .effects import (
    DoClearAnimation,
    DoClearInterrupt,
    DoConnect,
    DoDisconnect,
    DoExec,
    DoFetchFrames,
    DoFetchMap,
    DoLaunchServer,
    DoLog,
    DoQuery,
    DoReceiveData,
    DoRedrawUI,
    DoResetAnimationBuffer,
    DoSaveAndQuit,
    DoSendData,
    DoSetInterrupt,
    DoStopServer,
    DoTerminate,
    DoUploadAtomic,
    DoValidateRemotePath,
    Effect,
)
from .events import (
    Connected,
    ConnectionFailed,
    ErrorOccurred,
    ExecComplete,
    FetchComplete,
    FetchFailed,
    FetchMapComplete,
    ProgressUpdated,
    ReceiveDataComplete,
    SendDataComplete,
    ServerLaunched,
    ServerLost,
    ServerPolled,
    ServerStopped,
    UploadPipelineComplete,
)
from .protocol import DEFAULT_CHUNK_SIZE
from .server_kill import KillReport, kill_remote_server
from .session import new_session_id

if TYPE_CHECKING:
    from .engine import Engine


def _server_join(backend, *parts: str) -> str:
    # win_native talks to a Windows server (Windows-style paths).
    # Every other backend talks to a POSIX server, so the path must use
    # forward slashes regardless of the client OS, otherwise a Windows
    # client mixes in backslashes that break both shell quoting on the
    # remote and the server's open().
    if backend.backend_type == "win_native":
        return os.path.join(*parts)
    return posixpath.join(*parts)


def _decode_vertex_map_cbor(blob: bytes) -> dict:
    """Decode a ``map.pickle`` CBOR envelope into ``dict[str, ndarray]``.

    Producer is ``frontend/_cbor_bridge_.dumps_envelope("VertexMap", ...)``.
    Numpy arrays cross the wire as nested Python lists; we cast back to int64.
    """
    from .module import get_cbor2

    cbor2 = get_cbor2()
    env = cbor2.loads(blob)
    if not isinstance(env, dict) or env.get("kind") != "VertexMap":
        raise ValueError(
            f"map.pickle envelope kind mismatch: got {env.get('kind') if isinstance(env, dict) else type(env)!r}"
        )
    payload = env.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("VertexMap payload must be a map")
    return {k: numpy.asarray(v, dtype=numpy.int64) for k, v in payload.items()}


def _decode_surface_map_cbor(blob: bytes) -> dict:
    """Decode a ``surface_map.pickle`` CBOR envelope.

    Producer is ``frontend/_cbor_bridge_.dumps_envelope("SurfaceMap", ...)``.
    The inner shape stays ``{"version": 2, "maps": {uuid: (tri, coefs, surf_tri)}}``;
    we just rehydrate the inner numpy arrays.
    """
    from .module import get_cbor2

    cbor2 = get_cbor2()
    env = cbor2.loads(blob)
    if not isinstance(env, dict) or env.get("kind") != "SurfaceMap":
        raise ValueError(
            f"surface_map.pickle envelope kind mismatch: "
            f"got {env.get('kind') if isinstance(env, dict) else type(env)!r}"
        )
    payload = env.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("SurfaceMap payload must be a map")
    maps_in = payload.get("maps", {}) or {}
    rehydrated: dict = {}
    for name, entry in maps_in.items():
        tri_indices, coefs, surf_tri = entry
        rehydrated[name] = (
            numpy.asarray(tri_indices, dtype=numpy.int64),
            numpy.asarray(coefs, dtype=numpy.float64),
            numpy.asarray(surf_tri, dtype=numpy.int64),
        )
    return {"version": payload.get("version"), "maps": rehydrated}


def _decode_display_pin_map_cbor(blob: bytes) -> dict:
    """Decode a ``display_pin_map.pickle`` CBOR envelope.

    Producer is ``frontend/_scene_.py:export_fixed``. The payload is
    ``{"version": 1, "n_total": int, "blocks": [{"uuid", "blender_index",
    "offset"}, ...]}``, with blocks in the order the solver writes them to
    ``display_pin_<N>.bin``. Returns ``{"n_total": int, "blocks": [(uuid,
    blender_index, offset), ...]}``. Anything malformed is refused: placing
    vertices from a misread map would move the wrong ones.
    """
    from .module import get_cbor2

    cbor2 = get_cbor2()
    env = cbor2.loads(blob)
    if not isinstance(env, dict) or env.get("kind") != "DisplayPinMap":
        raise ValueError(
            f"display_pin_map.pickle envelope kind mismatch: "
            f"got {env.get('kind') if isinstance(env, dict) else type(env)!r}"
        )
    payload = env.get("payload")
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(
            "display_pin_map.pickle has an unsupported payload (expected "
            "version 1); the session needs to be re-baked."
        )
    n_total = int(payload["n_total"])
    blocks = []
    offset = 0
    for block in payload["blocks"]:
        index = numpy.asarray(block["blender_index"], dtype=numpy.int64)
        if int(block["offset"]) != offset:
            raise ValueError(
                f"display_pin_map.pickle block for {block['uuid']!r} starts "
                f"at {block['offset']}, expected {offset}"
            )
        blocks.append((str(block["uuid"]), index, offset))
        offset += int(index.size)
    if offset != n_total:
        raise ValueError(
            f"display_pin_map.pickle blocks cover {offset} positions, "
            f"but n_total is {n_total}"
        )
    return {"n_total": n_total, "blocks": blocks}


class EffectRunner:
    """Execute ``Effect`` objects, dispatching result events to the engine.

    Instantiate one ``EffectRunner`` per ``Engine`` and pass it to
    ``engine.tick(runner)`` on every main-thread timer tick.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._backend: ConnectionBackend | None = None
        self._interrupt = threading.Event()
        self._project_name: str | None = None
        self._chunk_size: int = DEFAULT_CHUNK_SIZE
        # What the most recent ``_do_stop_server`` found and killed. Written
        # on the worker thread, read on the main thread after the stop has
        # settled to ``server=UNKNOWN``, so the two never overlap.
        self.last_kill_report: KillReport | None = None

        # Raw last-seen server response for UI display.  Kept here —
        # NOT in AppState — because it's a cache, not state.  See
        # core/cache.py for the rationale.
        from .cache import ResponseCache
        self._response_cache = ResponseCache()

        # Stacked task I/O: one worker thread, two slots.
        # Commands (data send, build, exec, etc.) queue in order.
        # Polls (status queries) stack — only the latest matters.
        self._cmd_queue: list[tuple] = []      # [(fn, args), ...]
        self._poll_slot: tuple | None = None   # (fn, args) or None
        self._io_lock = threading.Lock()       # Protects _cmd_queue + _poll_slot
        self._work_event = threading.Event()   # Wakes the worker
        self._stop_event = threading.Event()   # Stops the worker on cleanup
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # Connection attempts: a counter, and the lock that makes taking a
        # ticket and adopting a backend one step. A connect runs on a thread
        # of its own rather than on the worker above (see ``_do_connect``), so
        # more than one attempt can be in flight and exactly one of them may
        # land. The holder of the current ticket is that one.
        self._connect_lock = threading.Lock()
        self._connect_epoch: int = 0

        # Animation state (thread-safe buffer)
        self._anim_lock = threading.Lock()
        self._anim_map: dict[str, numpy.ndarray] = {}
        self._anim_surface_map: dict = {}
        # ``display_pin_map.pickle`` decoded by _decode_display_pin_map_cbor,
        # or empty when the session writes no display pins.
        self._anim_display_pin_map: dict = {}
        self._anim_statistics_manifest: bytes | None = None
        self._anim_statistics_zero_fetched = False
        # The upload the three session artifacts above were downloaded
        # for. They are fetched only while unset, so without an identity
        # they outlive the dataset they describe; see
        # ``_drop_stale_session_artifacts``.
        self._anim_upload_id: str | None = None
        # (solver frame, vertices, statistics blobs, display pins). The last
        # is ``(active, positions)`` parsed from ``display_pin_<N>.bin``, or
        # None when the session writes no display pins.
        self._anim_frames: list[
            tuple[int, numpy.ndarray, list[bytes], tuple | None]
        ] = []
        self._anim_total: int = 0
        self._anim_applied: int = 0
        self._fetched: list[int] = []

        # Data result store
        self._data_lock = threading.Lock()
        self._received_data: bytes | None = None
        self._exec_output: dict | None = None

    # -- public API --

    def stop(self) -> None:
        """Stop the worker thread. Called during addon unregister/reload."""
        self._stop_event.set()
        self._work_event.set()  # Wake it so it exits
        # A connection attempt runs on its own thread, which this cannot join:
        # taking the ticket away is what stops its result from landing in a
        # runner the add-on has already torn down.
        self._take_connect_ticket()
        # Drop queued I/O so a later restart() doesn't resurrect jobs
        # that reference a backend the user already walked away from.
        with self._io_lock:
            self._cmd_queue.clear()
            self._poll_slot = None

    def restart(self) -> None:
        """Resume the worker thread after a prior stop().

        Idempotent: no-op if the worker is already alive. Called from
        ``facade.ensure_engine_timer`` on every addon register() so that
        an addon disable→enable cycle (whose unregister ran ``stop()``)
        still ends up with a live worker. The module-level singleton
        init block in ``facade.py`` only re-runs on a full
        ``sys.modules`` reload, not on a plain Blender enable cycle,
        so without this the runner stays dead forever.
        """
        if self._worker.is_alive():
            return
        self._stop_event.clear()
        self._work_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    @property
    def backend(self) -> ConnectionBackend | None:
        return self._backend

    @property
    def project_name(self) -> str | None:
        return self._project_name

    @project_name.setter
    def project_name(self, value: str) -> None:
        self._project_name = value

    def execute(self, effect: Effect) -> None:
        """Execute one effect.  Called from ``Engine.tick()`` on the main thread.

        I/O effects go to the worker thread via cmd queue or poll slot.
        UI effects (log, redraw, interrupt, animation) run immediately.
        """

        match effect:
            # -- Connection (commands) --
            case DoConnect(backend_type=bt, config=cfg, server_port=sp):
                cfg_copy = dict(cfg)
                if sp:
                    cfg_copy["server_port"] = sp
                # Mint the fresh session id and read the last-saved id here,
                # on the main thread, so the bpy read happens where it is
                # safe (execute() runs inside Engine.tick() on the main
                # thread) and the Connected transition arm stays pure. The
                # values ride on the Connected event the attempt dispatches.
                sid = new_session_id()
                saved = self._last_saved_session_id()
                # ON ITS OWN THREAD, NOT ON THE I/O WORKER. See ``_do_connect``
                # for why: a handshake against a host that does not answer
                # would hold the one thread every later operation queues onto.
                epoch = self._take_connect_ticket()
                threading.Thread(
                    target=self._do_connect,
                    args=(bt, cfg_copy, sid, saved, epoch),
                    name="ppf-connect-attempt",
                    daemon=True,
                ).start()

            case DoDisconnect():
                self._do_disconnect()

            case DoValidateRemotePath():
                if self._backend:
                    self._submit_cmd(self._do_validate_path)

            # -- Server lifecycle (commands) --
            case DoLaunchServer(
                cuda_device=device,
                cuda_device_uuid=device_uuid,
                device=compute_device,
                gpu_backend=gpu_backend,
            ):
                self._submit_cmd(
                    self._do_launch_server,
                    device,
                    device_uuid,
                    compute_device,
                    gpu_backend,
                )

            case DoStopServer():
                self._submit_cmd(self._do_stop_server)

            # -- Queries --
            case DoQuery(request=req):
                if req:
                    # Command query (build, start, etc.) — never drop
                    self._submit_cmd(self._do_query, req)
                else:
                    # Status poll — latest wins
                    self._submit_poll(self._do_query, req)

            # -- Data transfer (commands) --
            case DoSendData(remote_path=p, data=d):
                self._submit_cmd(self._do_send_data, p, d)

            case DoUploadAtomic(project_root=pr, data=d, param=pm,
                                 data_hash=dh, param_hash=ph):
                self._submit_cmd(self._do_upload_atomic, pr, d, pm, dh, ph)

            case DoReceiveData(remote_path=p):
                self._submit_cmd(self._do_receive_data, p)

            # -- Fetch (commands) --
            case DoFetchMap(root=r):
                self._submit_cmd(self._do_fetch_map, r)

            case DoFetchFrames(root=r, frame_count=fc, already_fetched=_af, only_latest=ol):
                # Always use self._fetched — set by facade.fetch() from
                # Blender-side applied frames. This matches the old
                # Communicator._fetch() which used self._fetched directly.
                self._submit_cmd(self._do_fetch_frames, r, fc, self._fetched, ol)

            # -- Exec (command) --
            case DoExec(command=c, shell=s):
                self._submit_cmd(self._do_exec, c, s)

            # -- Immediate (main thread) --
            case DoSetInterrupt():
                self._interrupt.set()

            case DoClearInterrupt():
                self._interrupt.clear()

            case DoClearAnimation():
                with self._anim_lock:
                    self._anim_frames.clear()
                    self._anim_total = 0
                    self._anim_applied = 0

            case DoResetAnimationBuffer():
                with self._anim_lock:
                    self._anim_frames.clear()
                    self._anim_total = 0
                    self._anim_applied = 0
                    self._anim_map = {}
                    self._anim_surface_map = {}
                    self._anim_display_pin_map = {}
                    self._anim_statistics_manifest = None
                    self._anim_statistics_zero_fetched = False
                    self._anim_upload_id = None

            # -- Terminate / Save-and-quit (commands) --
            case DoTerminate():
                self._submit_cmd(self._do_terminate)

            case DoSaveAndQuit():
                self._submit_cmd(self._do_save_and_quit)

            # -- UI (immediate) --
            case DoLog(message=m):
                console.write(m)

            case DoRedrawUI():
                self._redraw()

    # -- animation buffer access (called from main thread) --

    def has_pending_animation_frames(self) -> bool:
        """True while frames wait to be applied (main thread).

        The frame-pump modal reads this to decide whether it still has
        work: a solve can leave its last frames queued after the state
        machine has already returned to idle, and those frames need a
        modal-operator context to be applied.
        """
        with self._anim_lock:
            return bool(self._anim_frames)

    def take_one_animation_frame(
        self,
    ) -> tuple[tuple | None, dict, dict, dict, bytes | None, int, int]:
        """Pop one pending frame (main thread)."""
        with self._anim_lock:
            frame = self._anim_frames.pop(0) if self._anim_frames else None
            if frame is not None:
                self._anim_applied += 1
            return (
                frame,
                self._anim_map,
                self._anim_surface_map,
                self._anim_display_pin_map,
                self._anim_statistics_manifest,
                self._anim_applied,
                self._anim_total,
            )

    @property
    def received_data(self) -> bytes | None:
        with self._data_lock:
            return self._received_data

    @property
    def exec_output(self) -> dict | None:
        with self._data_lock:
            return self._exec_output

    def set_fetched_frames(self, frames: list[int]) -> None:
        """Set the list of already-fetched frame numbers."""
        # Hold ``_anim_lock`` so a concurrent ``_do_fetch_frames`` running
        # on the I/O worker can detect the reassignment via its
        # ``self._fetched is not fetched`` guard and discard a stale
        # live-fetch instead of leaking frames into the new context.
        with self._anim_lock:
            self._fetched = list(frames)

    def clear_fetched_frames(self) -> None:
        """Clear the fetched frame tracking."""
        with self._anim_lock:
            self._fetched = []

    # -- background thread helpers --

    def _submit_cmd(self, fn, *args) -> None:
        """Submit a command (never dropped) to the I/O worker."""
        with self._io_lock:
            self._cmd_queue.append((fn, args))
        self._work_event.set()

    def _submit_poll(self, fn, *args) -> None:
        """Submit a poll query (latest wins, stale ones dropped)."""
        with self._io_lock:
            self._poll_slot = (fn, args)
        self._work_event.set()

    def _pick_job(self) -> tuple | None:
        """Pick next job: commands first, then poll."""
        with self._io_lock:
            if self._cmd_queue:
                return self._cmd_queue.pop(0)
            if self._poll_slot is not None:
                job = self._poll_slot
                self._poll_slot = None
                return job
            return None

    def _worker_loop(self) -> None:
        """Single I/O worker thread. One SSH operation at a time.

        Commands (data send, build, etc.) execute in order.
        Polls (status queries) stack — only the latest runs.
        After each command, a status poll runs automatically (like the
        old Communicator's _update_status after every task).
        """
        while not self._stop_event.is_set():
            job = self._pick_job()
            if job is None:
                self._work_event.wait(timeout=0.25)
                self._work_event.clear()
                continue
            fn, args = job
            is_poll = (fn == self._do_query and (not args or not args[0]))
            is_stop = fn == self._do_stop_server
            try:
                fn(*args)
                # After each command (not poll, not stop), auto-query
                # for fresh status — unless another command was queued
                # during this one. In that case, the pending command
                # will auto-query itself; slipping a status poll in
                # between lets the server's pre-command state race
                # ahead of the command reply and (e.g. for upload →
                # build pipelines) trip the solver-terminal check in
                # the outer ServerPolled handler, snapping activity
                # back to IDLE before the queued command ever runs.
                # Stop is special: ``_do_stop_server`` already
                # dispatched ``ServerStopped`` (server=UNKNOWN), and
                # an auto-poll right after would re-contact the
                # process we just told to die: on a co-located
                # backend the server is often still up (orchestrator
                # owned, or kill not yet propagated), so the poll
                # succeeds and flips server back to RUNNING. Skip the
                # auto-poll on stop.
                if not is_poll and not is_stop and self._backend and self._project_name:
                    with self._io_lock:
                        has_pending_cmd = bool(self._cmd_queue)
                    if not has_pending_cmd:
                        self._do_query({})
            except Exception as e:
                if self._backend and not self._backend.is_alive():
                    from .events import ConnectionLost
                    console.write(f"Connection lost: {e}")
                    self._engine.dispatch(ConnectionLost(cause=str(e)))
                    with self._io_lock:
                        self._cmd_queue.clear()
                        self._poll_slot = None
                else:
                    self._engine.dispatch(ErrorOccurred(
                        error=str(e), source=fn.__name__,
                    ))

    # -- I/O implementations --

    def _last_saved_session_id(self) -> str:
        """Session id stored in the active scene at last save, or "".

        Reads ``bpy`` state, so it must run on the main thread (it is
        called from ``execute()`` inside ``Engine.tick()``). Mirrors
        ``facade.last_saved_session_id`` but logs a genuine read failure to
        the console instead of swallowing it silently, so the reconcile
        branch is not skipped without a trace.
        """
        try:
            import bpy  # pyright: ignore
            from ..models.groups import get_addon_data, has_addon_data
            scene = bpy.context.scene
            if scene is None or not has_addon_data(scene):
                return ""
            return get_addon_data(scene).state.last_session_id or ""
        except Exception as e:
            console.write(f"last-saved session id read failed: {e}")
            return ""

    # -- connection attempts --

    def _take_connect_ticket(self) -> int:
        """Invalidate every attempt in flight and return the new ticket.

        Every path that starts, cancels or supersedes a connection attempt
        comes through here, so the newest caller always holds the only current
        ticket and an older attempt can tell that it is no longer wanted.
        """
        with self._connect_lock:
            self._connect_epoch += 1
            return self._connect_epoch

    def _connect_attempt_is_current(self, epoch: int) -> bool:
        """True while *epoch* is still the attempt the add-on is waiting for."""
        with self._connect_lock:
            return epoch == self._connect_epoch

    def _adopt_backend(self, backend, epoch: int) -> bool:
        """Publish *backend* as the connection, unless the attempt was dropped.

        The test and the assignment are ONE step under the lock, because the
        two orderings against a disconnect have to differ. A disconnect that
        takes the ticket first must find nothing published, so this attempt
        closes what it opened; one that arrives after must find the reference,
        so it closes it. Either way the backend has exactly one owner, and no
        live transport is ever left in a runner the state machine believes is
        offline.
        """
        with self._connect_lock:
            if epoch != self._connect_epoch:
                return False
            self._backend = backend
            return True

    def _close_abandoned(self, backend) -> None:
        """Close a connection that landed after the user gave up on it."""
        try:
            backend.disconnect()
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            console.write(f"Canceled connection attempt: close failed: {e}")
        console.write("Closed a connection attempt that was canceled.")

    def _do_connect(
        self,
        backend_type: str,
        config: dict,
        session_id: str,
        saved_session_id: str,
        epoch: int,
    ) -> None:
        """Open a connection, on a thread this attempt owns.

        RUNS OFF THE I/O WORKER, unlike every other operation here, and that
        is the whole point. A handshake against a host that is not up blocks
        for as long as its transport takes, and the worker is the one thread
        every later operation queues onto, so a connect submitted there makes
        Cancel followed by a connect somewhere ELSE wait for the host the user
        already gave up on: measured, an SSH attempt to a stopped box held the
        worker while a native connect that needed nothing but a local socket
        sat at "Connecting..." behind it, and the panel looked wedged until
        Blender was restarted. Nothing about the worker's serial discipline
        rests on the connect being queued: while the add-on is offline there
        is no backend for another job to act on, and at most one attempt ever
        publishes one.

        *epoch* is this attempt's ticket, taken when the effect ran. A cancel,
        a disconnect, an add-on teardown or a later attempt takes a new one, so
        an attempt holding a stale ticket is ABANDONED: it closes whatever it
        opened and dispatches nothing. Both halves of that matter. A
        ``Connected`` from an abandoned attempt would put the add-on ONLINE
        against a host the user has walked away from, and a ``ConnectionFailed``
        from one would replace the state of the attempt they started instead,
        reporting the dead host's error over a connection that is already
        succeeding.
        """
        try:
            # The native backends query a server already on the port before
            # attaching to it, and a query names a project the server then
            # selects; naming the add-on's own keeps that query identical to
            # the first status poll.
            config = dict(config, project_name=self._project_name or "")
            backend = create_backend(backend_type, config)
            remote_root = ""
            if hasattr(backend, 'current_directory'):
                remote_root = backend.current_directory
        except Exception as e:
            # Nothing is published until the adoption below, so a failure here
            # leaves no partial backend to reset: what it can leave is a report
            # that does not belong to anyone, which the ticket answers.
            if not self._connect_attempt_is_current(epoch):
                console.write(f"Canceled connection attempt failed: {e}")
                return
            self._engine.dispatch(ConnectionFailed(error=str(e)))
            return
        if not self._adopt_backend(backend, epoch):
            self._close_abandoned(backend)
            return
        self._probe_solver_host_gpus(backend)
        self._probe_solver_host_builds(backend)
        if not self._connect_attempt_is_current(epoch):
            # A cancel landed while the solver host was being probed, which is
            # two command round trips long. Reporting a connection now would
            # put the add-on online over a transport ``_do_disconnect`` has
            # already closed, which is the one state nothing recovers from.
            if self._backend is not backend:
                # A later attempt has taken the reference over, so the close
                # ``_do_disconnect`` performs on what it finds published did
                # not reach this one.
                self._close_abandoned(backend)
            console.write("Connection canceled while the solver host was probed.")
            return
        self._engine.dispatch(Connected(
            remote_root=remote_root,
            session_id=session_id,
            saved_session_id=saved_session_id,
        ))

    def probe_solver_host_gpus(self) -> None:
        """Re-enumerate the connected solver host's GPUs on the worker thread.

        The Refresh button calls this. It goes straight to the command queue
        rather than through an event, because it changes no application state:
        it refills a cache the panel reads.
        """
        backend = self._backend
        if backend is not None:
            self._submit_cmd(self._probe_solver_host_gpus, backend)

    def probe_solver_host_builds(self) -> None:
        """Re-list the connected solver host's solver builds on the worker thread.

        The same Refresh button, for the same reason as the GPU list: it refills
        a cache the panel reads and changes no application state.
        """
        backend = self._backend
        if backend is not None:
            self._submit_cmd(self._probe_solver_host_builds, backend)

    def _probe_solver_host_builds(self, backend) -> None:
        """List the solver builds on the machine *backend* will run the server on.

        ONLY A REMOTE BACKEND IS ASKED. A native connection's builds are on this
        machine, where the panel resolves them directly from the filesystem on
        every redraw; asking over the backend would be a slower answer to a
        question already answered.

        A FAILURE IS RECORDED, NEVER RAISED. This runs inside connect, and a
        host that cannot list its builds is still a host worth connecting to:
        the device rows then say why they have nothing to offer, and the launch
        refuses by name if the selection cannot be resolved.
        """
        from . import remote_builds

        remote_builds.forget_builds()
        if backend.backend_type in NATIVE_BACKENDS:
            return
        root = remote_builds.normalize_root(backend.current_directory)
        if not root:
            remote_builds.record_probe_failure(
                "The connection names no directory on the solver host, so its "
                "solver builds could not be listed."
            )
            return
        try:
            result = backend.exec_command(
                remote_builds.probe_command(root),
                shell=True,
                timeout=remote_builds.PROBE_TIMEOUT_SECONDS,
            )
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            remote_builds.record_probe_failure(
                f"Could not list the solver builds on the solver host: {e}"
            )
            return
        if result.get("exit_code") != 0:
            detail = " ".join(result.get("stderr") or []).strip() or "no output"
            remote_builds.record_probe_failure(
                f"Listing the solver builds on the solver host failed: {detail}"
            )
            return
        # The root goes in WITH the listing: its keys are absolute paths under
        # this root, and a reader that derived its own could resolve against a
        # different one. One probe, one root, one answer.
        remote_builds.load_builds(result.get("stdout") or [], root)

    def _probe_solver_host_gpus(self, backend) -> None:
        """Enumerate the GPUs of the machine *backend* will run the server on.

        One path for every backend: ``exec_command`` reaches the solver host
        whether that is this machine, another one over SSH, or a container, so
        nothing here has to know which. It runs on the worker thread that owns
        the connection, because a panel draw must never issue a backend
        command. The result only fills a cache, so a failure is recorded for
        the panel to show rather than raised: it costs the dropdown its device
        names, not the connection.
        """
        from . import gpu_devices

        gpu_devices.forget_devices()
        # The macOS native solver runs on the system default Metal device and
        # offers no way to select another, and macOS carries no nvidia-smi to
        # enumerate with. An empty cache is what the panel reads as "no
        # picker"; recording a probe failure would put a red line under a
        # dropdown that has nothing to offer.
        if backend.backend_type == "mac_native":
            return
        direct = backend.backend_type in ("win_native", "linux_native")
        command = (
            gpu_devices.NVIDIA_SMI_ARGS
            if direct
            else gpu_devices.NVIDIA_SMI_COMMAND
        )
        try:
            result = backend.exec_command(
                command,
                shell=not direct,
                timeout=gpu_devices.PROBE_TIMEOUT_SECONDS,
            )
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            gpu_devices.record_probe_failure(
                f"Could not run nvidia-smi on the solver host: {e}"
            )
            return
        if result.get("exit_code") != 0:
            detail = " ".join(result.get("stderr") or []).strip() or "no output"
            gpu_devices.record_probe_failure(
                f"nvidia-smi failed on the solver host: {detail}"
            )
            return
        gpu_devices.load_devices("\n".join(result.get("stdout") or []))

    def _do_disconnect(self) -> None:
        from . import gpu_devices, remote_builds

        # ABANDON ANY ATTEMPT STILL HANDSHAKING, which is what Cancel is. The
        # attempt runs on its own thread and cannot be interrupted, so taking
        # the ticket is the whole of the cancellation: whatever that thread
        # opens it will close itself, and it reports nothing. This comes FIRST
        # so an attempt that adopts a backend a moment later finds the ticket
        # already gone; see ``_adopt_backend``.
        self._take_connect_ticket()
        # The next connection may reach a different machine, where a list left
        # over from this one would name GPUs, or build directories, that are
        # not there.
        gpu_devices.forget_devices()
        remote_builds.forget_builds()
        if self._backend:
            self._backend.disconnect()
            self._backend = None
        with self._anim_lock:
            self._anim_map = {}
            self._anim_surface_map = {}
            self._anim_display_pin_map = {}
            self._anim_statistics_manifest = None
            self._anim_statistics_zero_fetched = False
            self._anim_upload_id = None
            self._anim_frames.clear()
            self._anim_total = 0
            self._anim_applied = 0
            # Inside the lock so a live-fetch in flight on the I/O worker
            # observes the reassigned reference via its stale-context
            # guard and bails instead of appending to the new list.
            self._fetched = []
        self._response_cache.clear()
        # The project_name is bound to the connection, not to the
        # Blender scene; clearing it here makes load-time disconnect
        # truly idempotent so a new scene's first ``set_project_name``
        # is always the one that sticks.
        self._project_name = None

    def _do_validate_path(self) -> None:
        if not self._backend:
            return
        directory = self._backend.current_directory
        # For the three native backends the local filesystem is probed
        # directly, each with the resolver its own connect path uses; for SSH /
        # Docker the check goes through ``exec_command``.
        # SSH/Docker targets are always Linux, so use POSIX joins and no .exe
        # regardless of the client OS.
        #
        # THE REFUSAL HERE IS NOT A MESSAGE BUT A CANCELLATION, which is why
        # each branch probes exactly what its spawn path probes. The
        # `Connected` transition schedules this check, and a caller that starts
        # an upload right after connecting (every `bl_*` scenario does) has its
        # pipeline in flight when an `ErrorOccurred` lands: that transition
        # resets the activity and clears the pending build, the upload then
        # completes into nothing, and the solver sits at NO_BUILD until the
        # caller's wait expires. Measured on the Windows leg of Blender CI,
        # where `build.bat` puts the CUDA server under target/cuda/release and
        # the check looked only under target/release: 23 scenarios failed by
        # timeout with the upload landed and no build request ever sent (runs
        # 35159051178 to 35193244970, and a Windows reproduction whose event
        # trail named `validate_path`).
        if self._backend.backend_type in NATIVE_BACKENDS:
            from .connection import native_path_check
            device = getattr(self._backend, "_device", "GPU")
            gpu_backend = getattr(self._backend, "_gpu_backend", "AUTO")
            error = native_path_check(
                self._backend.backend_type, directory, device, gpu_backend
            )
            if error is not None:
                # Same sentence the launch path raises, so the user is not told
                # two different things about one missing binary.
                self._engine.dispatch(ErrorOccurred(
                    error=error,
                    source="validate_path",
                ))
            return
        # THE REMOTE ROOT IS JUDGED BY THE LISTING CONNECT ALREADY TOOK, not by
        # a second command naming one path. Asking about
        # `target/release/ppf-cts-server` alone refuses a solver host holding a
        # DISTRIBUTION, which ships `target/<backend>/release` and no
        # `target/release`: the message is "Remote path not found" naming a
        # path that host is right not to have. That refusal is a cancellation
        # rather than a message (see the note above), so the artist's first
        # upload lands into nothing.
        #
        # ASKED ONLY WHETHER THE ROOT HOLDS A SOLVER AT ALL, never whether it
        # holds the device currently selected. The device is applied at Start
        # Server, which is where a selection the host cannot serve is refused
        # by name; refusing it here would cancel a pipeline over a choice the
        # artist can still change.
        from . import remote_builds
        from .connection import remote_holds_any_server

        if remote_builds.probe_error():
            self._engine.dispatch(ErrorOccurred(
                error=remote_builds.probe_error(),
                source="validate_path",
            ))
            return
        probed = remote_builds.normalize_root(directory)
        if not remote_holds_any_server(probed, remote_builds.cached_builds()):
            from .connection import remote_not_found_message
            self._engine.dispatch(ErrorOccurred(
                error=remote_not_found_message(
                    probed, remote_builds.cached_builds()
                ),
                source="validate_path",
            ))

    def _do_query(self, request: dict | None = None) -> None:
        if not self._backend or not self._project_name:
            return
        response, alive = self._backend.query(
            request or {}, self._project_name, self._chunk_size
        )
        if alive:
            # Mirror into cache before the transition runs so any parallel
            # UI read sees the latest.
            self._response_cache.record(response)
            self._engine.dispatch(ServerPolled(response=response))
        elif request:
            # User-initiated query (a non-empty request payload) couldn't
            # reach the server. Without this branch ``Activity.EXECUTING``
            # never clears and the modal hangs on its status message
            # ("Deleting Remote Data...", etc.) until its timeout. Empty
            # background polls stay silent so a server that hasn't booted
            # yet doesn't trip a spurious server-lost reset.
            self._engine.dispatch(ServerLost())

    def _launch_device(self, device: str = "") -> str:
        """The Compute Device this launch runs on.

        What Start Server was given, else what the connection was made with.
        One reader, so the binary a launch resolves and the GPU token it writes
        cannot disagree about which device is in play.
        """
        from .connection import DEVICE_GPU

        return device or getattr(self._backend, "_device", DEVICE_GPU)

    def _remote_server_binary(
        self, directory: str, device: str = "", gpu_backend: str = ""
    ) -> str:
        """The server on the solver host this connection's selection names.

        Raises rather than falling back, which is the whole point: a selection
        that cannot be satisfied has to say so at Start Server, naming what the
        host holds, instead of launching another build and reporting success.
        The three refusals `remote_not_found_message` distinguishes are each
        something the artist can act on without changing the path.

        A HOST THAT COULD NOT BE LISTED IS NOT A HOST WITH NO BUILDS. Where the
        probe failed, its reason is what the artist needs, so it is raised
        instead of a refusal derived from an empty listing.
        """
        from . import remote_builds
        from .connection import (
            DEVICE_GPU,
            GPU_BACKEND_AUTO,
            remote_not_found_message,
            remote_server_binary,
        )

        # THE SELECTION THE LAUNCH WAS GIVEN, falling back to the one the
        # connection was made with. The panel's remote rows are drawn only once
        # a connection is up, so the artist's answer arrives with Start Server;
        # reading the backend's connect-time copy alone would leave those rows
        # movable and inert, which is the silent substitution the whole device
        # mechanism exists to prevent.
        device = self._launch_device(device)
        gpu_backend = gpu_backend or getattr(
            self._backend, "_gpu_backend", GPU_BACKEND_AUTO
        )
        # THE ROOT THE LISTING WAS TAKEN UNDER, which is what its keys are
        # joined from. See `remote_builds.normalize_root`.
        directory = remote_builds.normalize_root(directory)
        listing = remote_builds.cached_builds()
        if not listing and remote_builds.probe_error():
            raise FileNotFoundError(remote_builds.probe_error())
        found = remote_server_binary(directory, listing, device, gpu_backend)
        if found is None:
            raise FileNotFoundError(
                remote_not_found_message(directory, listing, device, gpu_backend)
            )
        return found

    def _do_launch_server(
        self,
        cuda_device: int = -1,
        cuda_device_uuid: str = "",
        device: str = "",
        gpu_backend: str = "",
    ) -> None:
        # BOUND ONCE, for the reason `_do_stop_server` states: a disconnect
        # runs on the main thread and can clear `self._backend` while this
        # method is inside its startup wait loop.
        backend = self._backend
        if not backend:
            return

        # THE THREE NATIVES SPAWN A CHILD PROCESS instead of writing a launch
        # script, because there is no shell on the other side of them. Their
        # launch is one path rather than three: the differences are the name in
        # the message and whether a GPU can be named at all.
        if backend.backend_type in NATIVE_BACKENDS:
            # A NATIVE LAUNCH TAKES NO DEVICE FROM HERE. Its device was settled
            # when the connection was made, because connecting itself refuses a
            # root that holds no build for it, and the backend keeps that answer
            # so a Stop/Start cycle makes the same choice.
            self._launch_native_server(backend, cuda_device, cuda_device_uuid)
            return

        port = backend.server_port
        directory = backend.current_directory

        # Check Docker port exposure on the SSH host (not inside container).
        # Only meaningful for SSH+container backends, where Blender runs on
        # the user's machine and reaches the container through an SSH tunnel
        # to the docker host. For plain DOCKER backend, Blender talks to the
        # local docker daemon directly via the Container API and the
        # _instance object has no exec_command.
        if (backend.backend_type == "ssh"
                and hasattr(backend, '_container')
                and backend._container):
            container = backend._container
            if hasattr(backend, '_instance') and backend._instance:
                # Run docker port directly on the SSH host
                _stdin, stdout, stderr = backend._instance.exec_command(
                    f"docker port {container} {port}"
                )
                exit_code = stdout.channel.recv_exit_status()
                if exit_code != 0:
                    raise ConnectionError(
                        f"Docker port {port} is not exposed on container '{container}'. "
                        f"Please expose the port with '-p {port}:{port}' when starting the container."
                    )

        # Paths constructed here are sent to a Linux remote over SSH/Docker
        # (this branch never runs for a native backend, see the early
        # return above).
        # Use posixpath so a Windows client doesn't emit backslashes into
        # the shell commands.
        server_log = posixpath.join(directory, "server.log")
        progress_file = posixpath.join(directory, "progress.log")
        script_path = "/tmp/start_server.sh"

        # Clear progress.log
        backend.exec_command(f"rm -f {progress_file}", shell=True)

        # ppf-cts-server defaults to binding 127.0.0.1. That is what we want
        # for SSH (Direct): paramiko's direct-tcpip channel terminates at the
        # remote's loopback. Inside a container, however,
        # docker -p HOST:CONTAINER forwards traffic to the container's
        # external interface (eth0), not loopback, so we must bind
        # 0.0.0.0 there.
        in_container = bool(getattr(backend, "_container", ""))
        host_flag = "--host 0.0.0.0 " if in_container else ""

        # The Rust ppf-cts-server binary writes ``progress.log`` markers
        # (SERVER_STARTING / SERVER_READY) and serves the same wire
        # protocol the addon speaks; build it with
        # ``cargo build --release -p ppf-cts-server``.
        # An environment assignment in front of the binary is how the GPU
        # choice is delivered to a server started through a shell. It is
        # applied on the solver host, so the index is that host's, which is
        # also where the panel's device list was enumerated.
        #
        # WHICH BUILD DIRECTORY, ASKED OF THE LISTING THE SOLVER HOST GAVE,
        # rather than spelled here. Naming one path would be two defects at
        # once: the artist's Compute Device would reach nothing on a remote
        # connection, and a solver host holding a DISTRIBUTION could not be
        # launched at all, since `build-linux-native/bundle.sh` ships
        # `target/<backend>/release` and no `target/release`.
        rust_bin = self._remote_server_binary(directory, device, gpu_backend)
        # A CPU RUN NAMES NO GPU, and dropping the selection here is not tidiness.
        # `CUDA_VISIBLE_DEVICES` in front of a binary with no CUDA in it changes
        # nothing about the run, and `describe_launch` would then write a console
        # line naming a GPU the solve is not on, which is a record of something
        # that did not happen. The panel already hides the picker for a CPU
        # device; this is the same answer where it is applied.
        from .connection import DEVICE_CPU

        on_cpu = self._launch_device(device) == DEVICE_CPU
        gpu_index = AUTOMATIC if on_cpu else cuda_device
        gpu_uuid = "" if on_cpu else cuda_device_uuid
        server_cmd = (
            f"{shell_prefix(gpu_index, gpu_uuid)}"
            f"{rust_bin} {host_flag}--port {port}"
        )

        # Activate the project venv so the build worker subprocess
        # (spawned by ppf-cts-server) resolves `python3` to one with the
        # runtime deps (cbor2 / psutil). Without this, a non-login SSH
        # shell falls through to the system python, which lacks them and
        # the build fails with a ModuleNotFoundError. The `_ppf_cts_py`
        # cdylib itself is not installed into the venv; `frontend` loads
        # it directly from `target/<profile>/`.
        # Convention: $HOME/.local/share/ppf-cts/venv.
        # Falls through silently when the venv is absent (Docker /
        # win-native paths use embedded Python set up differently).
        venv_activate = '$HOME/.local/share/ppf-cts/venv/bin/activate'
        activate_clause = (
            f'[ -f {venv_activate} ] && source {venv_activate}; '
        )
        # THE BUILD DIRECTORY IS NAMED, NOT ONLY THE BINARY, and leaving it out
        # is the split `_apply_target_dir` prevents on the native path. A run
        # takes three things out of a build directory, and the server binary is
        # one: the build worker's Python loads the cdylib from whichever target
        # directory `frontend._target_dirs` finds first, and `frontend`
        # then writes THAT directory into the session's `command.sh` as
        # `SOLVER_PATH`. Launching `target/cpu/release/ppf-cts-server` without
        # naming its target directory therefore starts the CPU server and runs
        # the solve on whatever `<root>/target` holds, with nothing anywhere
        # reporting the split, because each binary answers `--backend` honestly
        # about itself and neither is asked about the other.
        #
        # It is exported on its own line rather than inlined into the
        # `bash -c "..."` below, where a `$` or a backtick would be expanded by
        # the remote shell before the server ever saw it.
        target_dir = remote_target_dir(rust_bin)
        target_clause = (
            f'export CARGO_TARGET_DIR="{target_dir}"\n' if target_dir else ""
        )
        script = (
            f'#!/bin/bash\ncd "{directory}"\n'
            f'{target_clause}'
            f'nohup bash -c "{activate_clause}{server_cmd}" > "{server_log}" 2>&1 &\n'
        )
        backend.exec_command(f"cat <<'EOF' > {script_path}\n{script}EOF\n", shell=True)
        backend.exec_command(f"chmod +x {script_path}", shell=True)
        result = backend.exec_command(script_path, shell=True)
        if result["exit_code"] != 0:
            raise FileNotFoundError("Failed to launch server")
        console.write(describe_launch(gpu_index, True, gpu_uuid))

        # Monitor startup. We keep looping until the client can actually
        # reach the server through whatever transport the backend uses
        # (direct-tcpip over SSH, docker-proxy, etc.). ``SERVER_READY`` in
        # progress.log only confirms that the server's own ``bind()`` call
        # returned; it says nothing about host-side port forwarding. In
        # setups where the docker port binding is misconfigured, the old
        # break-on-SERVER_READY path would dispatch ServerLaunched even
        # though no subsequent client query could reach the server — the
        # Start Server button would silently flip from "Server Starting..."
        # to a greyed-out "Start Server on Remote" (because is_server_running
        # is True) while the connection was actually broken.
        max_wait = 16
        start = time.time()
        last_lines = 0
        ready_marker_seen = False
        while True:
            elapsed = time.time() - start
            result = backend.exec_command(
                f"cat {progress_file} 2>/dev/null", shell=True,
            )
            lines = result.get("stdout", [])
            if len(lines) > last_lines:
                last_lines = len(lines)
                for line in lines:
                    if "ERROR" in line.upper() or "FAILED" in line.upper():
                        raise RuntimeError(f"Server startup failed: {line}")
                if lines and "SERVER_READY" in lines[-1]:
                    ready_marker_seen = True

            # A failed bind (port already taken) is written to server.log,
            # not progress.log, so SERVER_READY never appears and the loop
            # would otherwise burn the full max_wait and report a generic
            # timeout that leaves the user guessing. Detect the "port in
            # use" case and fail fast, naming the port: on a shared host the
            # holder is usually a stale ppf-cts-server or a Docker container
            # publishing the same port (`-p PORT:PORT`, whose docker-proxy
            # binds the host port even with nothing live inside).
            if not ready_marker_seen:
                log_tail = backend.exec_command(
                    f"tail -20 {server_log} 2>/dev/null", shell=True,
                )
                tail_text = "\n".join(log_tail.get("stdout", []))
                if "Address already in use" in tail_text or "os error 98" in tail_text:
                    raise ConnectionError(
                        f"Server port {port} is already in use on the remote "
                        "host, so ppf-cts-server could not bind it. Press Kill "
                        "Process to end a stale ppf-cts-server there, stop "
                        "whatever else holds the port (a Docker container "
                        f"publishing it with `-p {port}:{port}`), or choose a "
                        f"different server port.\n{tail_text}"
                    )

            if elapsed > max_wait:
                log_result = backend.exec_command(
                    f"tail -20 {server_log}", shell=True,
                )
                details = "\n".join(log_result.get("stdout", []))
                if ready_marker_seen:
                    raise ConnectionError(
                        "Server reached SERVER_READY but the client cannot "
                        "reach it. Check that the server port is forwarded "
                        f"to the client (e.g. `docker run -p {port}:{port}`).\n"
                        f"{details}"
                    )
                raise TimeoutError(f"Server startup timed out.\n{details}")

            # Only try querying after SERVER_READY — before that the
            # listener isn't up yet and every query is guaranteed to fail.
            if ready_marker_seen and self._project_name:
                response, alive = backend.query(
                    {}, self._project_name, self._chunk_size,
                )
                if alive:
                    break

            time.sleep(1)

        self._engine.dispatch(ServerLaunched())

    def _launch_native_server(
        self,
        # TAKEN FROM THE CALLER, never re-read. `_do_launch_server` bound it
        # once for the whole launch, and re-reading `self._backend` here would
        # reopen the window that binding closed: a disconnect runs on the main
        # thread and can clear the attribute while this helper is inside its
        # readiness wait.
        backend,
        cuda_device: int = -1,
        cuda_device_uuid: str = "",
    ) -> None:
        """Start the server a native backend owns, and wait for it to answer.

        A FAILURE AFTER A SUCCESSFUL SPAWN STOPS WHAT IT STARTED. Otherwise a
        server that came up and then failed its readiness wait would be left
        holding the port, and the next Connect would meet a squatter it did not
        start rather than the error that actually happened.
        """
        kind = backend.backend_type
        label = NATIVE_BACKENDS[kind]
        launched = False
        try:
            if kind == "mac_native":
                # No device selection accompanies it: the Metal backend opens
                # the system default device and offers no way to name another,
                # so there is nothing for a device argument to carry.
                launched = backend.start_server()
            else:
                launched = backend.start_server(cuda_device, cuda_device_uuid)
            self._wait_for_native_server(backend, label)
        except Exception as e:
            cleanup_error = None
            if launched:
                try:
                    backend.stop_server()
                except Exception as stop_error:
                    cleanup_error = stop_error
            detail = str(e)
            if cleanup_error is not None:
                detail += f" Cleanup also failed: {cleanup_error}"
            self._engine.dispatch(ErrorOccurred(
                error=f"Failed to start server: {detail}",
                source="launch_server",
            ))
            return
        if kind == "mac_native":
            console.write(
                "Solver server started on the system default Metal device."
                if launched else
                "Solver server: attached to one that was already running, so "
                "it keeps the Metal device it started with."
            )
        else:
            console.write(describe_launch(cuda_device, launched, cuda_device_uuid))
        self._engine.dispatch(ServerLaunched())

    def _wait_for_native_server(
        self,
        # TAKEN FROM THE CALLER, for the reason the launch path states: this
        # is the LONGEST window of the three, a readiness loop that polls
        # until a deadline, and it re-read `self._backend` on every pass.
        backend,
        label: str,
        timeout: float = 16.0,
    ) -> None:
        """Wait until a native child on this machine answers the solver protocol.

        ONE WAITER FOR THE THREE NATIVES, because what is waited for is
        identical: a child process the add-on spawned, on a loopback port, that
        answers the protocol once it is ready and whose exit is the one thing
        that can end the wait early. Only *label* differs, and it appears in the
        two messages the artist reads.

        A CHILD THAT EXITS IS REPORTED AS AN EXIT rather than waited out, so a
        server that cannot start says so in a second instead of at the timeout,
        with the code it exited with and where to read why.
        """
        from .connection import _probe_ppf_cts_server

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if _probe_ppf_cts_server(
                backend.server_port, timeout=min(0.5, remaining)
            ):
                return
            process = getattr(backend, "_process", None)
            if process is not None:
                returncode = process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"{label} server exited with code {returncode} before "
                        "becoming ready. Check server.log."
                    )
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(
            f"{label} server did not become ready within {timeout:g} seconds. "
            "Check server.log."
        )

    def _do_stop_server(self) -> None:
        # BOUND ONCE, THEN USED, because `self._backend` can be cleared by
        # ANOTHER THREAD while this runs. `DoDisconnect` calls
        # `_do_disconnect` directly on Blender's main thread rather than
        # queueing it on the I/O worker, so a disconnect lands mid-stop: the
        # guard above passes, the wait loop below then re-reads the attribute
        # and raises `'NoneType' object has no attribute 'query'`, which the
        # worker turns into a panel error naming this method. Measured:
        # "Disconnected." and "[_do_stop_server] 'NoneType' object has no
        # attribute 'query'" one second apart.
        #
        # Acting on the backend this stop was DISPATCHED for is also the right
        # answer rather than merely a safe one: it is the server the user asked
        # to stop, and a transport that has since closed reports a transport
        # failure, which is a true statement, instead of an AttributeError.
        backend = self._backend
        if not backend:
            return
        # The cache is keyed off "what the server last said." Once
        # the server is gone, every cached field (data="READY",
        # frame=N, scene_info, ...) is stale. Leaving them around
        # makes the UI display a Scene Info collapsible and a frame
        # count for a server that no longer holds that state — and,
        # if the user clicks Start again, those stale fields keep
        # showing until a fresh ServerPolled lands.
        # The native backends own the server subprocess directly; their
        # stop_server() handles termination and waiting. SSH/Docker backends
        # drive the same shape via exec_command in their own stop_server
        # overrides.
        #
        # Either branch leaves what it did in ``last_kill_report``, which the
        # Force Terminate Process operator reads once the stop settles; Stop Server on
        # Remote ignores it.
        if backend.backend_type in NATIVE_BACKENDS:
            self.last_kill_report = backend.stop_server()
            self._response_cache.clear()
            self._engine.dispatch(ServerStopped())
            return
        # SSH / Docker remote: ``kill_remote_server`` ends the Rust server
        # serving THIS backend's port, through the backend's exec_command,
        # which reaches the host or the container. It is scoped to the port
        # rather than to the process name because a solver host can be
        # shared, and a name-wide sweep would end other people's servers.
        container = getattr(backend, "container", "") or ""
        self.last_kill_report = kill_remote_server(
            backend.exec_command,
            where=f"container {container}" if container else "the solver host",
            port=backend.server_port,
        )
        # Wait for the server to actually stop.
        alive = True
        for _ in range(5):
            _response, alive = backend.query(
                {}, self._project_name or "", self._chunk_size
            )
            if not alive:
                break
            time.sleep(0.25)
        self._response_cache.clear()
        if alive:
            # Still answering after the grace period. ServerStopped is
            # dispatched anyway so the panel is not left wedged mid-stop, and
            # the next background poll will find the server and correct the
            # state; what must not happen is that the user is told the stop
            # worked when the only evidence available says it did not.
            self._engine.dispatch(ErrorOccurred(
                error=(
                    "Stop Server: the solver is still answering on port "
                    f"{backend.server_port}. Check that the process is "
                    f"reachable from the container or host the add-on is "
                    f"driving."
                ),
                source="stop_server",
            ))
        self._engine.dispatch(ServerStopped())

    def _do_send_data(self, remote_path: str, data: bytes) -> None:
        if not self._backend or not self._project_name:
            return

        def progress_cb(p, t):
            self._engine.dispatch(ProgressUpdated(progress=p, traffic=t))

        def interrupt_cb():
            return self._interrupt.is_set()

        self._backend.send_data(
            remote_path, data, self._project_name,
            chunk_size=self._chunk_size,
            progress_cb=progress_cb, interrupt_cb=interrupt_cb,
        )
        self._engine.dispatch(SendDataComplete())

    def _do_upload_atomic(self, project_root: str, data: bytes, param: bytes,
                          data_hash: str, param_hash: str) -> None:
        """Atomic combined upload of (data.pickle, param.pickle).

        On success dispatches ``UploadPipelineComplete`` so the transition
        can move from SENDING straight into BUILDING. Errors are routed
        via ``ErrorOccurred`` the same way ``_safe_run`` handles
        transport-level failures. ``param_hash`` rides along on the
        wire so the server can store it next to the pickles and echo
        it on every status response.
        """
        if not self._backend or not self._project_name:
            return

        def progress_cb(p, t):
            self._engine.dispatch(ProgressUpdated(progress=p, traffic=t))

        def interrupt_cb():
            return self._interrupt.is_set()

        self._backend.upload_atomic(
            project_root, data, param, self._project_name,
            data_hash=data_hash, param_hash=param_hash,
            chunk_size=self._chunk_size,
            progress_cb=progress_cb, interrupt_cb=interrupt_cb,
        )
        self._engine.dispatch(UploadPipelineComplete())

    def _do_receive_data(self, remote_path: str) -> None:
        if not self._backend or not self._project_name:
            return

        def progress_cb(p, t):
            self._engine.dispatch(ProgressUpdated(progress=p, traffic=t))

        def interrupt_cb():
            return self._interrupt.is_set()

        data = self._backend.receive_data(
            remote_path, self._project_name,
            chunk_size=self._chunk_size,
            progress_cb=progress_cb, interrupt_cb=interrupt_cb,
        )
        with self._data_lock:
            self._received_data = data
        self._engine.dispatch(ReceiveDataComplete(data=data))

    def _count_remote_frames(self, root: str) -> int:
        """Discover the max frame index from vert_*.bin files on the remote.

        Returns the highest N found in vert_N.bin, which matches the
        server's ``frame`` field semantics (max frame index, not file count).
        vert_0.bin is the rest pose and is excluded from the count.
        """
        if not self._backend or not self._project_name:
            return 0
        try:
            # The native backends read the output directory off THIS machine,
            # so glob it rather than shelling out. On Windows the shell-out
            # below is not merely slower but wrong: cmd.exe has no `ls`, so it
            # exits non-zero and this returns 0 frames, which the caller cannot
            # tell apart from a solve that produced none. The fetch then
            # applies nothing, no mesh cache is attached, and the failure
            # surfaces far from its cause. Globbing is also the cheaper of the
            # two on POSIX, where it is what the shell would have done anyway,
            # which is why the other two natives take this branch as well.
            if self._backend.backend_type in NATIVE_BACKENDS:
                import glob
                output_dir = os.path.join(root, "session", "output")
                names = [
                    os.path.basename(p)
                    for p in glob.glob(os.path.join(output_dir, "vert_*.bin"))
                ]
            else:
                output_dir = posixpath.join(root, "session", "output")
                result = self._backend.exec_command(
                    f"ls -1 {output_dir}/vert_*.bin 2>/dev/null",
                    shell=True,
                )
                if result.get("exit_code", 1) != 0 or not result.get("stdout"):
                    return 0
                names = [line.strip().rsplit("/", 1)[-1] for line in result["stdout"]]
            max_frame = 0
            for name in names:
                if name.startswith("vert_") and name.endswith(".bin"):
                    try:
                        idx = int(name[5:-4])
                        if idx > max_frame:
                            max_frame = idx
                    except ValueError:
                        continue
            return max_frame
        except Exception as e:
            console.write(f"[count_remote_frames] {e}")
        return 0

    def _drop_stale_session_artifacts(self) -> None:
        """Drop the cached session artifacts when they describe an earlier upload.

        ``_anim_map``, ``_anim_surface_map`` and
        ``_anim_statistics_manifest`` are downloaded once and reused for
        every frame of a fetch, which is what keeps a live run from
        re-reading them on each poll. They describe ONE uploaded dataset,
        though: a scene whose group set changed produces frames with a
        different object set, and ``decode_frame`` rejects those against
        the previous manifest ("statistics frame object count does not
        match manifest"), so every frame of the new run is dropped.

        Clearing them at the transitions that rebuild a session is what
        the fetch path already does (``DoResetAnimationBuffer``), but
        Transfer and Run emit ``DoClearAnimation``, which drops the queued
        frames and the counters and leaves these three in place. Rather
        than adding this to the list of places that must remember, bind
        the artifacts to the upload they came from: the server mints a
        fresh ``upload_id`` for each upload and echoes it on every status
        response, so any path that lands a new dataset invalidates them,
        including one another client initiated.

        Called at both fetch entry points, before anything reads them.

        The queued frames go with them, under the one lock, because a frame
        was fetched against these artifacts and cannot be applied against
        the next set: ``take_one_animation_frame`` hands the map to the
        main thread, and a frame drawn while the map is empty applies to
        no object at all and writes no PC2, silently. Every other place
        that drops the artifacts (``DoResetAnimationBuffer``,
        ``_do_disconnect``) clears the queue in the same breath for the
        same reason. Both callers set the counters after this returns, so
        zeroing them here cannot race a fetch already in progress.
        """
        upload_id = str(self._response_cache.get("upload_id", "") or "")
        with self._anim_lock:
            if self._anim_upload_id == upload_id:
                return
            self._anim_upload_id = upload_id
            self._anim_map = {}
            self._anim_surface_map = {}
            self._anim_display_pin_map = {}
            self._anim_statistics_manifest = None
            self._anim_statistics_zero_fetched = False
            self._anim_frames.clear()
            self._anim_total = 0
            self._anim_applied = 0

    def _ensure_anim_map(self, root: str) -> None:
        """Download animation map if not already loaded. No event dispatched."""
        with self._anim_lock:
            if self._anim_map:
                return
        if not self._backend or not self._project_name:
            return
        map_path = _server_join(self._backend, root, "session", "map.pickle")
        map_data = self._backend.receive_data(map_path, self._project_name, chunk_size=self._chunk_size)
        # Format-sniff between pickle (first byte 0x80) and CBOR map
        # (first byte 0xa0-0xb7). Producer: ``frontend/_scene_.py:export_fixed``.
        if map_data and map_data[0] != 0x80:
            anim_map = _decode_vertex_map_cbor(map_data)
        else:
            anim_map = pickle.loads(map_data)

        surface_map = {}
        try:
            smap_path = _server_join(self._backend, root, "session", "surface_map.pickle")
            smap_data = self._backend.receive_data(smap_path, self._project_name, chunk_size=self._chunk_size)
            if smap_data and smap_data[0] != 0x80:
                payload = _decode_surface_map_cbor(smap_data)
            else:
                payload = pickle.loads(smap_data)
            # Require wire format v2: {"version": 2, "maps": {uuid: (tri_indices, coefs, surf_tri)}}.
            # Anything else is rejected so the client cannot apply the wrong reconstruction math.
            if (
                isinstance(payload, dict)
                and payload.get("version") == 2
                and isinstance(payload.get("maps"), dict)
            ):
                surface_map = payload["maps"]
            else:
                console.write(
                    "surface_map.pickle has unsupported format (expected v2 "
                    "frame-embedding envelope); the session needs to be re-baked."
                )
        except Exception:
            pass

        # Every session this frontend exports carries a display-pin map, empty
        # when it has no exact SOLID pin, so a map that cannot be read is an
        # error to report rather than a feature the session lacks.
        dpin_path = _server_join(
            self._backend, root, "session", "display_pin_map.pickle",
        )
        try:
            dpin_data = self._backend.receive_data(
                dpin_path, self._project_name, chunk_size=self._chunk_size,
            )
        except Exception as e:
            raise RuntimeError(
                "display_pin_map.pickle could not be read from the session; a "
                f"session built before display pins must be transferred again ({e})"
            ) from e
        display_pins = _decode_display_pin_map_cbor(dpin_data)
        display_pin_map = display_pins if display_pins["blocks"] else {}

        with self._anim_lock:
            self._anim_map = anim_map
            self._anim_surface_map = surface_map
            self._anim_display_pin_map = display_pin_map

    def _do_fetch_map(self, root: str) -> None:
        """Download map and dispatch FetchMapComplete. Guarantees a
        terminal event (FetchMapComplete on success, FetchFailed on
        precondition miss or exception) so state never hangs in
        FETCHING."""
        try:
            if not self._backend:
                self._engine.dispatch(FetchFailed(reason="fetch map: no backend"))
                return
            if not self._project_name:
                self._engine.dispatch(FetchFailed(reason="fetch map: no project"))
                return
            self._drop_stale_session_artifacts()
            self._ensure_anim_map(root)
            with self._anim_lock:
                anim_map = self._anim_map
                surface_map = self._anim_surface_map
            self._engine.dispatch(FetchMapComplete(
                map_data=anim_map, surface_map=surface_map,
            ))
        except Exception as e:
            console.write(f"_do_fetch_map failed: {e}")
            self._engine.dispatch(FetchFailed(reason=f"fetch map: {e}"))

    def _do_fetch_frames(
        self, root: str, frame_count: int, fetched: list[int], only_latest: bool
    ) -> None:
        """Download the frame range. Guarantees a terminal event on every
        path (FetchComplete on success, FetchFailed on precondition miss
        or exception). Counter reset (_anim_total/_applied/_frames) now
        happens via DoResetAnimationBuffer, not here."""
        try:
            if not self._backend:
                if not only_latest:
                    self._engine.dispatch(FetchFailed(reason="fetch frames: no backend"))
                return
            if not self._project_name:
                if not only_latest:
                    self._engine.dispatch(FetchFailed(reason="fetch frames: no project"))
                return
            if not only_latest:
                # Always discover actual file count for Fetch All — the
                # status response "frame" field may lag behind the
                # actual output files.
                discovered = self._count_remote_frames(root)
                if discovered > 0:
                    frame_count = discovered
            elif frame_count < 1:
                frame_count = self._count_remote_frames(root)
            if frame_count < 1:
                if not only_latest:
                    self._engine.dispatch(FetchComplete(total_frames=0))
                return

            self._drop_stale_session_artifacts()
            self._ensure_anim_map(root)
            if self._anim_statistics_manifest is None:
                manifest_path = _server_join(
                    self._backend,
                    root,
                    "session",
                    "output",
                    "statistics_manifest.cbor",
                )
                try:
                    statistics_manifest = self._backend.receive_data(
                        manifest_path,
                        self._project_name,
                        chunk_size=self._chunk_size,
                    )
                except Exception as exc:
                    # Sessions produced before timeline statistics remain
                    # fetchable. Their panel reports that a rerun is required.
                    console.write(
                        "statistics manifest unavailable; this session must be "
                        f"rerun for timeline statistics ({exc})"
                    )
                    statistics_manifest = b""
                with self._anim_lock:
                    if self._fetched is not fetched:
                        if not only_latest:
                            self._engine.dispatch(FetchFailed(
                                reason="fetch frames: context reset while "
                                "fetching statistics manifest"
                            ))
                        return
                    self._anim_statistics_manifest = statistics_manifest

            if only_latest:
                start_frame = max(1, frame_count)
                frames = [start_frame]
            else:
                frames = list(range(1, frame_count + 1))

            def interrupt_cb():
                return self._interrupt.is_set()

            # Stale-context guard. ``self._fetched`` is reassigned (not
            # mutated) by ``clear_fetched_frames``, ``set_fetched_frames``,
            # ``_do_disconnect``, and ``DoResetAnimationBuffer``-adjacent
            # flows (the FetchRequested transition runs ``clear_fetched_frames``
            # on the caller side just before dispatching). Live-fetches
            # queued under the previous reference must not append into the
            # new context — otherwise frames they fetched leak into the
            # newly-cleared ``_anim_frames`` and into the new ``_fetched``,
            # making a subsequent full fetch under-count ``_anim_total``
            # while ``_anim_applied`` over-counts (the bl_chain_reconnect
            # macOS regression: ``(applied=9, total=7)``).
            if self._fetched is not fetched:
                if not only_latest:
                    self._engine.dispatch(FetchFailed(
                        reason="fetch frames: context reset before start"))
                return

            to_fetch = [i for i in frames if i not in fetched]
            if not only_latest:
                with self._anim_lock:
                    self._anim_total = len(to_fetch)
                    self._anim_applied = 0

            # The fetch pipeline has two phases, each shown as its own
            # 0→100% bar in the UI (phase is disambiguated by the status
            # label): the download half runs while activity=FETCHING and
            # the apply half while activity=APPLYING. Progress is reset
            # to 0 at the FETCHING→APPLYING transition.
            to_fetch_total = max(1, len(to_fetch))

            fetched_count = 0
            for idx, i in enumerate(to_fetch):
                if interrupt_cb():
                    break

                def inner_progress(p, t, idx=idx):
                    frac = min(1.0, (idx + p) / to_fetch_total)
                    self._engine.dispatch(
                        ProgressUpdated(progress=frac, traffic=t)
                    )

                filename = f"vert_{i}.bin"
                path = _server_join(self._backend, root, "session", "output", filename)
                data = self._backend.receive_data(
                    path, self._project_name, chunk_size=self._chunk_size,
                    progress_cb=inner_progress if not only_latest else None,
                    interrupt_cb=interrupt_cb,
                )
                vert = numpy.frombuffer(data, dtype=numpy.float32).reshape(-1, 3)
                # The frame's display pins (see _decode_display_pin_map_cbor):
                # one byte per block, set while the block is still pinned,
                # then every scripted position as three float32s.
                with self._anim_lock:
                    display_pin_map = self._anim_display_pin_map
                display_pin_frame = None
                if display_pin_map:
                    n_blocks = len(display_pin_map["blocks"])
                    n_total = display_pin_map["n_total"]
                    dpin_path = _server_join(
                        self._backend, root, "session", "output",
                        f"display_pin_{i}.bin",
                    )
                    dpin_data = self._backend.receive_data(
                        dpin_path, self._project_name,
                        chunk_size=self._chunk_size,
                        interrupt_cb=interrupt_cb,
                    )
                    if len(dpin_data) != n_blocks + 12 * n_total:
                        raise ValueError(
                            f"display_pin_{i}.bin holds {len(dpin_data)} bytes, "
                            f"expected {n_blocks + 12 * n_total} for {n_blocks} "
                            f"blocks and {n_total} positions"
                        )
                    active = numpy.frombuffer(
                        dpin_data, dtype=numpy.uint8, count=n_blocks,
                    ).astype(bool)
                    positions = numpy.frombuffer(
                        dpin_data, dtype="<f4", offset=n_blocks,
                    ).reshape(-1, 3)
                    display_pin_frame = (active, positions)
                statistics_data = []
                includes_statistics_zero = False
                if self._anim_statistics_manifest:
                    if not self._anim_statistics_zero_fetched:
                        zero_path = _server_join(
                            self._backend,
                            root,
                            "session",
                            "output",
                            "statistics_0.cbor",
                        )
                        statistics_data.append(self._backend.receive_data(
                            zero_path,
                            self._project_name,
                            chunk_size=self._chunk_size,
                            interrupt_cb=interrupt_cb,
                        ))
                        includes_statistics_zero = True
                    statistics_path = _server_join(
                        self._backend,
                        root,
                        "session",
                        "output",
                        f"statistics_{i}.cbor",
                    )
                    statistics_data.append(self._backend.receive_data(
                        statistics_path,
                        self._project_name,
                        chunk_size=self._chunk_size,
                        interrupt_cb=interrupt_cb,
                    ))
                # Re-check the context inside the lock that gates every
                # ``_anim_frames`` mutation. ``clear_fetched_frames`` /
                # ``set_fetched_frames`` take the same lock, so once we
                # observe the reference still matches we know no reset
                # can land before the append completes.
                with self._anim_lock:
                    if self._fetched is not fetched:
                        if only_latest:
                            return
                        self._engine.dispatch(FetchFailed(
                            reason="fetch frames: context reset mid-fetch"))
                        return
                    self._anim_frames.append(
                        (i, vert, statistics_data, display_pin_frame)
                    )
                    if includes_statistics_zero:
                        self._anim_statistics_zero_fetched = True
                    self._fetched.append(i)
                fetched_count += 1
                if not only_latest:
                    self._engine.dispatch(ProgressUpdated(
                        progress=(idx + 1) / to_fetch_total, traffic="",
                    ))

            if not only_latest:
                self._engine.dispatch(FetchComplete(total_frames=fetched_count))
        except Exception as e:
            console.write(f"_do_fetch_frames failed: {e}")
            if not only_latest:
                self._engine.dispatch(FetchFailed(reason=f"fetch frames: {e}"))
            else:
                # only_latest is a background poll during live sim — don't
                # tear down state, just log. A stuck state here isn't
                # possible because live sim doesn't transition to FETCHING.
                pass

    def _do_exec(self, command: str, shell: bool) -> None:
        if not self._backend:
            return
        output = self._backend.exec_command(command, shell=shell)
        with self._data_lock:
            self._exec_output = output
        if output:
            for line in output.get("stdout", []):
                console.write(line)
            for line in output.get("stderr", []):
                console.write(line)
            if not output.get("stdout") and not output.get("stderr"):
                console.write(f"Command completed with exit code: {output['exit_code']}")
        self._engine.dispatch(ExecComplete(output=output))

    def _do_terminate(self) -> None:
        if not self._backend or not self._project_name:
            return
        self._backend.query({"request": "terminate"}, self._project_name, self._chunk_size)
        # Wait for simulation to stop
        for _ in range(20):
            response, alive = self._backend.query({}, self._project_name, self._chunk_size)
            if not is_sim_running_from_response(response):
                break
            time.sleep(0.25)
        self._response_cache.record(response)
        self._engine.dispatch(ServerPolled(response=response))

    def _do_save_and_quit(self) -> None:
        if not self._backend or not self._project_name:
            return
        self._backend.query({"request": "save_and_quit"}, self._project_name, self._chunk_size)

    # -- UI helpers --

    @staticmethod
    def _redraw() -> None:
        """Tag Blender viewport areas for redraw."""
        try:
            import bpy  # pyright: ignore
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
        except Exception:
            pass
