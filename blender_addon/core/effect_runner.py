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
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy

from ..models.console import console
from .backends import ConnectionBackend, create_backend
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
    DoPollAfter,
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
    FrameFetched,
    PollTick,
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

if TYPE_CHECKING:
    from .engine import Engine


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
        self._pending_timers: list = []        # Unfired bpy.app.timers callbacks
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # Animation state (thread-safe buffer)
        self._anim_lock = threading.Lock()
        self._anim_map: dict[str, numpy.ndarray] = {}
        self._anim_surface_map: dict = {}
        self._anim_frames: list[tuple[int, numpy.ndarray]] = []
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
        # Cancel any unfired poll-after timers; the stale module state
        # they'd dispatch into would segfault after reload.
        try:
            import bpy  # pyright: ignore
            for fn in list(self._pending_timers):
                if bpy.app.timers.is_registered(fn):
                    bpy.app.timers.unregister(fn)
        except Exception:
            pass
        self._pending_timers.clear()
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
                self._submit_cmd(self._do_connect, bt, cfg_copy)

            case DoDisconnect():
                self._do_disconnect()

            case DoValidateRemotePath():
                if self._backend:
                    self._submit_cmd(self._do_validate_path)

            # -- Server lifecycle (commands) --
            case DoLaunchServer():
                self._submit_cmd(self._do_launch_server)

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

            case DoPollAfter(delay=d):
                import bpy  # pyright: ignore
                bpy.app.timers.register(
                    self._dispatch_poll_timer, first_interval=d
                )
                self._pending_timers.append(self._dispatch_poll_timer)

            # -- Data transfer (commands) --
            case DoSendData(remote_path=p, data=d):
                self._submit_cmd(self._do_send_data, p, d)

            case DoUploadAtomic(project_root=pr, data=d, param=pm,
                                 data_hash=dh, param_hash=ph):
                self._submit_cmd(self._do_upload_atomic, pr, d, pm, dh, ph)

            case DoReceiveData(remote_path=p, tag=_tag):
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

    def take_one_animation_frame(self) -> tuple[tuple | None, dict, dict, int, int]:
        """Pop one pending frame (main thread)."""
        with self._anim_lock:
            frame = self._anim_frames.pop(0) if self._anim_frames else None
            if frame is not None:
                self._anim_applied += 1
            return (
                frame,
                self._anim_map,
                self._anim_surface_map,
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
            try:
                fn(*args)
                # After each command (not poll), auto-query for fresh
                # status — unless another command was queued during this
                # one. In that case, the pending command will auto-query
                # itself; slipping a status poll in between lets the
                # server's pre-command state race ahead of the command
                # reply and (e.g. for upload → build pipelines) trip
                # the solver-terminal check in the outer ServerPolled
                # handler, snapping activity back to IDLE before the
                # queued command ever runs.
                if not is_poll and self._backend and self._project_name:
                    with self._io_lock:
                        has_pending_cmd = bool(self._cmd_queue)
                    if not has_pending_cmd:
                        self._do_query({})
            except Exception as e:
                if self._backend and not self._backend.is_alive():
                    from .events import ConnectionLost
                    console.write(f"Connection lost: {e}")
                    self._engine.dispatch(ConnectionLost())
                    with self._io_lock:
                        self._cmd_queue.clear()
                        self._poll_slot = None
                else:
                    self._engine.dispatch(ErrorOccurred(
                        error=str(e), source=fn.__name__,
                    ))

    def _dispatch_poll(self) -> None:
        self._engine.dispatch(PollTick())

    def _dispatch_poll_timer(self) -> None:
        """Blender timer callback: dispatch PollTick into the queue.

        The persistent timer (_persistent_tick) handles processing.
        """
        try:
            self._pending_timers.remove(self._dispatch_poll_timer)
        except ValueError:
            pass
        try:
            self._engine.dispatch(PollTick())
        except Exception:
            pass

    # -- I/O implementations --

    def _do_connect(self, backend_type: str, config: dict) -> None:
        try:
            backend = create_backend(backend_type, config)
            self._backend = backend
            remote_root = ""
            if hasattr(backend, 'current_directory'):
                remote_root = backend.current_directory
        except Exception as e:
            # Reset any partial backend so the next attempt starts clean.
            if self._backend is not None:
                try:
                    self._backend.disconnect()
                except Exception:
                    pass
                self._backend = None
            self._engine.dispatch(ConnectionFailed(error=str(e)))
            return
        self._engine.dispatch(Connected(remote_root=remote_root))

    def _do_disconnect(self) -> None:
        if self._backend:
            self._backend.disconnect()
            self._backend = None
        with self._anim_lock:
            self._anim_map.clear()
            self._anim_surface_map.clear()
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
        server_py = os.path.join(self._backend.current_directory, "server.py")
        # win_native runs the Blender addon on Windows, where the backend's
        # exec_command goes through cmd.exe — bash `if [ -f ]` fails there.
        # Hit the local filesystem directly instead.
        if self._backend.backend_type == "win_native":
            if not os.path.isfile(server_py):
                self._engine.dispatch(ErrorOccurred(
                    error=f"Remote path not found ({server_py}).",
                    source="validate_path",
                ))
            return
        result = self._backend.exec_command(
            f"if [ -f {server_py} ]; then echo FOUND; else echo NOT_FOUND; fi",
            shell=True, cwd="/",
        )
        output = "\n".join(result.get("stdout", []))
        if result["exit_code"] != 0 or "NOT_FOUND" in output:
            self._engine.dispatch(ErrorOccurred(
                error=f"Remote path not found ({server_py}).",
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

    def _do_launch_server(self) -> None:
        if not self._backend:
            return

        # Win_native: connect spawns the initial server.py, then Stop/Start
        # cycles re-spawn it here. Without this re-spawn, ServerStopped
        # leaves _process=None and the next ServerLaunched dispatch is a
        # lie — _do_query silently fails (channel refuses), no ServerPolled
        # follows, solver stays NO_DATA, and the UI shows "Waiting for
        # Data" with stale Scene Info because the response cache was last
        # written before the stop.
        if self._backend.backend_type == "win_native":
            try:
                self._backend.start_server()
            except Exception as e:
                self._engine.dispatch(ErrorOccurred(
                    error=f"Failed to start server: {e}",
                    source="launch_server",
                ))
                return
            self._engine.dispatch(ServerLaunched())
            return

        port = self._backend.server_port
        directory = self._backend.current_directory

        # Check Docker port exposure on the SSH host (not inside container).
        # Only meaningful for SSH+container backends, where Blender runs on
        # the user's machine and reaches the container through an SSH tunnel
        # to the docker host. For plain DOCKER backend, Blender talks to the
        # local docker daemon directly via the Container API and the
        # _instance object has no exec_command.
        if (self._backend.backend_type == "ssh"
                and hasattr(self._backend, '_container')
                and self._backend._container):
            container = self._backend._container
            if hasattr(self._backend, '_instance') and self._backend._instance:
                # Run docker port directly on the SSH host
                _stdin, stdout, stderr = self._backend._instance.exec_command(
                    f"docker port {container} {port}"
                )
                exit_code = stdout.channel.recv_exit_status()
                if exit_code != 0:
                    raise ConnectionError(
                        f"Docker port {port} is not exposed on container '{container}'. "
                        f"Please expose the port with '-p {port}:{port}' when starting the container."
                    )

        venv_path = "$HOME/.local/share/ppf-cts/venv"
        server_log = os.path.join(directory, "server.log")
        progress_file = os.path.join(directory, "progress.log")
        script_path = "/tmp/start_server.sh"

        # Clear progress.log
        self._backend.exec_command(f"rm -f {progress_file}", shell=True)

        # server.py defaults to binding 127.0.0.1. That is what we want
        # for SSH (Direct) and Local: paramiko's direct-tcpip channel
        # terminates at the remote's loopback, and Local connects to
        # localhost on the same kernel. Inside a container, however,
        # docker -p HOST:CONTAINER forwards traffic to the container's
        # external interface (eth0), not loopback, so we must bind
        # 0.0.0.0 there.
        in_container = bool(getattr(self._backend, "_container", ""))
        host_flag = "--host 0.0.0.0 " if in_container else ""

        # Detect venv
        result = self._backend.exec_command(f"test -d {venv_path}", shell=True)
        if result["exit_code"] == 0:
            activate_cmd = f"source {venv_path}/bin/activate && python3 server.py {host_flag}--port {port}"
        else:
            activate_cmd = f"python3 server.py {host_flag}--port {port}"

        script = f'#!/bin/bash\ncd "{directory}"\nnohup bash -c "{activate_cmd}" > "{server_log}" 2>&1 &\n'
        self._backend.exec_command(f"cat <<'EOF' > {script_path}\n{script}EOF\n", shell=True)
        self._backend.exec_command(f"chmod +x {script_path}", shell=True)
        result = self._backend.exec_command(script_path, shell=True)
        if result["exit_code"] != 0:
            raise FileNotFoundError("Failed to launch server")

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
            result = self._backend.exec_command(
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

            if elapsed > max_wait:
                log_result = self._backend.exec_command(
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
                response, alive = self._backend.query(
                    {}, self._project_name, self._chunk_size,
                )
                if alive:
                    break

            time.sleep(1)

        self._engine.dispatch(ServerLaunched())

    def _do_stop_server(self) -> None:
        if not self._backend:
            return
        # win_native owns the server subprocess directly; pkill isn't on
        # cmd.exe and the backend has the handle already.
        if self._backend.backend_type == "win_native":
            self._backend.stop_server()
            # The cache is keyed off "what the server last said." Once
            # the server is gone, every cached field (data="READY",
            # frame=N, scene_info, ...) is stale. Leaving them around
            # makes the UI display a Scene Info collapsible and a frame
            # count for a server that no longer holds that state — and,
            # if the user clicks Start again, those stale fields keep
            # showing until a fresh ServerPolled lands.
            self._response_cache.clear()
            self._engine.dispatch(ServerStopped())
            return
        self._backend.exec_command("pkill -f server.py", shell=True)
        # Wait for server to actually stop
        for _ in range(5):
            response, alive = self._backend.query({}, self._project_name or "", self._chunk_size)
            if not alive:
                break
            time.sleep(0.25)
        self._response_cache.clear()
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
        output_dir = os.path.join(root, "session", "output")
        try:
            # win_native: the output dir is on this machine's local disk.
            # cmd.exe has no `ls`, so glob directly.
            if self._backend.backend_type == "win_native":
                import glob
                names = [
                    os.path.basename(p)
                    for p in glob.glob(os.path.join(output_dir, "vert_*.bin"))
                ]
            else:
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

    def _ensure_anim_map(self, root: str) -> None:
        """Download animation map if not already loaded. No event dispatched."""
        with self._anim_lock:
            if self._anim_map:
                return
        if not self._backend or not self._project_name:
            return
        map_path = os.path.join(root, "session", "map.pickle")
        map_data = self._backend.receive_data(map_path, self._project_name, chunk_size=self._chunk_size)
        anim_map = pickle.loads(map_data)

        surface_map = {}
        try:
            smap_path = os.path.join(root, "session", "surface_map.pickle")
            smap_data = self._backend.receive_data(smap_path, self._project_name, chunk_size=self._chunk_size)
            payload = pickle.loads(smap_data)
            # Wire format v2: {"version": 2, "maps": {uuid: (tri_indices, coefs, surf_tri)}}.
            # Legacy format was a bare dict of bary tuples; reject it so the
            # client doesn't silently apply the wrong reconstruction math.
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

        with self._anim_lock:
            self._anim_map = anim_map
            self._anim_surface_map = surface_map

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

            self._ensure_anim_map(root)

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
                path = os.path.join(root, "session", "output", filename)
                data = self._backend.receive_data(
                    path, self._project_name, chunk_size=self._chunk_size,
                    progress_cb=inner_progress if not only_latest else None,
                    interrupt_cb=interrupt_cb,
                )
                vert = numpy.frombuffer(data, dtype=numpy.float32).reshape(-1, 3)
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
                    self._anim_frames.append((i, vert))
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
            status = response.get("status", "")
            if status not in ("BUSY", "SAVE_AND_QUIT"):
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
