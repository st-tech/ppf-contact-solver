# File: server/engine.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# ServerEngine: event queue + dispatch + response generation.
# EffectExecutor: the only place that performs real I/O.
#
# The engine processes events synchronously under a lock (transitions are
# pure and fast), then hands effects to the executor.  Response generation
# reads the frozen state -- zero filesystem I/O.

from __future__ import annotations

import logging
import os
import pickle
import shutil
import threading
import traceback
from typing import TYPE_CHECKING, Any, Optional

from .state import Build, Data, ServerState, Solver
from .events import (
    BuildCancelledEvent,
    BuildCompleted,
    BuildFailed,
    BuildProgress,
    ErrorOccurred,
    Event,
    GPUCheckFailed,
    ProjectSelected,
    SolverCrashed,
    SolverFinished,
)
from .effects import (
    DoCancelBuild,
    DoCheckGPU,
    DoDeleteProjectData,
    DoKillSolver,
    DoLaunchSolver,
    DoLoadApp,
    DoLog,
    DoRequestSaveAndQuit,
    DoSpawnBuild,
    Effect,
)
from .transitions import server_transition

# These are imported from the main server module at init time.
PROTOCOL_VERSION = "0.03"
DATA_NAME = "data.pickle"
PARAM_NAME = "param.pickle"
APP_STATE = "app_state.pickle"
UPLOAD_ID_FILE = "upload_id.txt"
DATA_HASH_FILE = "data_hash.txt"
PARAM_HASH_FILE = "param_hash.txt"


# ---------------------------------------------------------------------------
# AppState helper (kept from original server.py, manages BlenderApp instance)
# ---------------------------------------------------------------------------

class AppState:
    """Wrapper around BlenderApp for persistence and lifecycle."""

    def __init__(self, name: str, app: Any):
        self.name = name
        self.app = app

    @staticmethod
    def create(name: str, verbose: bool = False, progress_callback=None) -> "AppState":
        from frontend import BlenderApp
        app = BlenderApp(name, verbose, progress_callback=progress_callback).populate().make()
        return AppState(name, app)

    @staticmethod
    def load(name: str, dirpth: str) -> Optional["AppState"]:
        """Load a persisted AppState from ``{dirpth}/app_state.pickle``.

        Returns None only when the project dir or the pickle file does
        not exist (a legitimate "not built yet" state). If the pickle
        exists but is corrupt, the unpickling error is propagated: we
        don't silently fall through to a rebuild.
        """
        if not os.path.exists(dirpth):
            return None
        app_path = os.path.join(dirpth, APP_STATE)
        if not os.path.exists(app_path):
            return None
        with open(app_path, "rb") as f:
            app = pickle.load(f)
        return AppState(name, app)

    def resumable(self) -> bool:
        return len(self.app.session.get.saved()) > 0

    def is_saving_in_progress(self) -> bool:
        return bool(os.path.exists(self.app.session.save_and_quit_file_path()))


# ---------------------------------------------------------------------------
# EffectExecutor
# ---------------------------------------------------------------------------

class BuildCancelled(Exception):
    """Raised inside the build thread when cancellation is requested."""


class EffectExecutor:
    """Execute ``Effect`` objects. The only place with real I/O."""

    _gpu_checked: bool = False  # Class-level: GPU only needs checking once

    def __init__(self, engine: "ServerEngine"):
        self._engine = engine
        self._build_cancel = threading.Event()
        self._build_thread: threading.Thread | None = None

    def execute(self, effect: Effect) -> None:
        try:
            self._execute(effect)
        except Exception as e:
            logging.error(f"Effect execution error ({type(effect).__name__}): {e}")
            self._engine.dispatch(ErrorOccurred(error=str(e)))

    def _execute(self, effect: Effect) -> None:
        match effect:
            case DoCheckGPU():
                self._check_gpu()
            case DoSpawnBuild():
                self._spawn_build()
            case DoCancelBuild():
                self._cancel_build()
            case DoLaunchSolver(resume_from=rf):
                self._launch_solver(rf)
            case DoKillSolver():
                self._kill_solver()
            case DoRequestSaveAndQuit():
                self._request_save_and_quit()
            case DoLoadApp(name=n, root=r):
                self._load_app(n, r)
            case DoDeleteProjectData(root=r):
                self._delete_project(r)
            case DoLog(message=m):
                logging.info(m)

    def _check_gpu(self) -> None:
        try:
            from frontend import Utils
            Utils.check_gpu()
        except Exception as e:
            self._engine.dispatch(GPUCheckFailed(error=str(e)))

    def _spawn_build(self) -> None:
        engine = self._engine
        state = engine.state
        self._build_cancel.clear()
        cancel_flag = self._build_cancel

        def run_build():
            try:
                # GPU check: cached after first success (GPU doesn't change)
                if not EffectExecutor._gpu_checked:
                    try:
                        from frontend import Utils
                        Utils.check_gpu()
                        EffectExecutor._gpu_checked = True
                    except Exception as e:
                        engine.dispatch(GPUCheckFailed(error=str(e)))
                        return

                def progress_callback(progress: float, info: str):
                    if cancel_flag.is_set():
                        raise BuildCancelled("Build cancelled by user")
                    engine.dispatch(BuildProgress(progress=progress, info=info))
                    logging.info(f"[BUILD] {progress:.0%} {info}")

                new_app = AppState.create(
                    state.name,
                    verbose=True,
                    progress_callback=progress_callback,
                )
                engine._app = new_app
                engine.dispatch(BuildCompleted())
            except BuildCancelled:
                logging.info("[BUILD] cancelled by user")
                engine._app = None
                engine.dispatch(BuildCancelledEvent())
            except Exception as e:
                traceback.print_exc()
                engine._app = None
                violations = getattr(e, "violations", [])
                tb = traceback.format_exc()
                # Extract last file/line from traceback
                tb_lines = [l.strip() for l in tb.strip().split("\n") if l.strip().startswith("File ")]
                location = tb_lines[-1] if tb_lines else ""
                engine.dispatch(BuildFailed(
                    error=f"decoding failed: {str(e)}\n{location}",
                    violations=violations,
                ))

        self._build_thread = threading.Thread(target=run_build, daemon=True)
        self._build_thread.start()

    def _cancel_build(self) -> None:
        """Cancel the build: set flag for cooperative check, then force-kill."""
        self._build_cancel.set()
        t = self._build_thread
        if t is not None and t.is_alive():
            import ctypes
            tid = t.ident
            if tid is not None:
                logging.info(f"[BUILD] force-killing build thread {tid}")
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(tid), ctypes.py_object(BuildCancelled)
                )
                if res == 0:
                    logging.warning("[BUILD] thread id not found")
                elif res > 1:
                    # Revert if multiple threads affected (shouldn't happen)
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(tid), None
                    )
                    logging.error("[BUILD] failed to kill build thread cleanly")

    def _launch_solver(self, resume_from: int | None) -> None:
        app = self._engine._app
        if app is None:
            self._engine.dispatch(ErrorOccurred(error="app is not built."))
            return
        self._engine.solver_launching = True
        try:
            from frontend import Utils
            if Utils.busy():
                Utils.terminate()
            logging.info(f"_launch_solver: calling session.start (resume_from={resume_from})")
            if resume_from is not None:
                app.app.session.resume(blocking=False)
            else:
                # ``session.start`` without force=True auto-resumes when a
                # saved checkpoint is on disk (a notebook UX convenience).
                # That breaks the protocol contract here: a server-side
                # ``StartRequested`` came from a client RunRequested that
                # explicitly wants a fresh run, not an implicit resume
                # (the client has its own Resume path -> resume_from=-1).
                # Without force=True the rig's chain scenarios that do
                # save_and_quit -> resume -> ... -> run end up with the
                # second ``run`` silently turning into another resume from
                # the old checkpoint, frame counter never advancing past
                # the saved value, and the test asserting state.frame=0.
                app.app.session.start(blocking=False, force=True)
            proc = getattr(app.app.session, "_process", None)
            logging.info(f"_launch_solver: session.start returned, _process={proc}")
        except Exception as e:
            import traceback
            logging.error(f"_launch_solver raised: {e}\n{traceback.format_exc()}")
            self._engine.dispatch(ErrorOccurred(error=str(e)))
        finally:
            self._engine.solver_launching = False

    def _kill_solver(self) -> None:
        try:
            from frontend import Utils
            if Utils.busy():
                Utils.terminate()
        except Exception as e:
            logging.error(f"Kill solver error: {e}")

    def _request_save_and_quit(self) -> None:
        app = self._engine._app
        if app is not None:
            app.app.session.save_and_quit()
        else:
            self._engine.dispatch(ErrorOccurred(error="no session found"))

    def _load_app(self, name: str, root: str) -> None:
        """Load ``app_state.pickle`` into engine._app for *name*.

        Pure load, no rebuild. When the pickle is absent we leave
        engine._app = None and state.build untouched (typically NONE):
        the user has not built yet and must click Build explicitly.
        Corrupt pickles propagate the unpickling error upward rather
        than silently falling through to a rebuild.

        Also rehydrates ``state.frame`` from the on-disk ``vert_*.bin``
        count. ``state.frame`` is the live solver counter, only ever
        advanced by ``SolverFrameUpdated`` events from the monitor; on
        a fresh server process it starts at 0. Without this rehydrate,
        a client that reconnects to a project with completed frames
        sees ``frame=0`` in the response — the Blender addon's Fetch
        button stays disabled because its poll requires ``frame > 0``,
        even though the same response carries a ``scene_info`` with
        "Simulated Frames: 180" (read directly from disk by
        ``_get_scene_info``).
        """
        from .state import Build
        from dataclasses import replace
        from .monitor import _count_frames

        new_app = AppState.load(name, root)
        if new_app is None:
            self._engine._app = None
            return
        logging.info(f"App loaded from pickle for name: {name}")

        self._engine._app = new_app
        disk_frame = _count_frames(root)
        with self._engine._lock:
            s = self._engine._state
            if s.name == name:
                is_resumable = new_app.resumable()
                updates = {"resumable": is_resumable, "frame": disk_frame}
                if s.build != Build.BUILT:
                    updates["build"] = Build.BUILT
                self._engine._state = replace(s, **updates)

    def _delete_project(self, root: str) -> None:
        if root and os.path.exists(root):
            shutil.rmtree(root, ignore_errors=True)
        self._engine._app = None


# ---------------------------------------------------------------------------
# ServerEngine
# ---------------------------------------------------------------------------

class ServerEngine:
    """Event-driven server state machine.

    Thread-safe: ``dispatch()`` can be called from any thread (build thread,
    monitor thread, connection handler threads).

    ``make_response()`` generates the client-facing JSON dict by reading
    the frozen state -- zero filesystem I/O.
    """

    def __init__(self, hardware_info: dict = None, git_branch: str = ""):
        self._state = ServerState()
        self._lock = threading.RLock()
        self._app: AppState | None = None
        self._executor = EffectExecutor(self)
        self._hardware = hardware_info or {}
        self._git_branch = git_branch
        self.solver_launching = False  # Set True while DoLaunchSolver runs

    # -- dispatch --

    def dispatch(self, event: Event) -> ServerState:
        """Process an event. Thread-safe."""
        with self._lock:
            new_state, effects = server_transition(self._state, event)
            self._state = new_state
        for effect in effects:
            self._executor.execute(effect)
        return new_state

    # -- state access --

    @property
    def state(self) -> ServerState:
        with self._lock:
            return self._state

    @property
    def app(self) -> AppState | None:
        return self._app

    # -- response generation (zero I/O) --

    def make_response(self) -> dict:
        """Generate the client-facing JSON response from cached state.

        Pure function of the frozen ``ServerState``: zero filesystem I/O.
        External state changes (Jupyter-built app pickles, externally
        launched solvers, newly written frame bins) are reconciled by
        ``select_project`` (on client connect) and the background solver
        monitor, both of which dispatch events that move ``ServerState``
        to the correct value *before* this method is called.
        """
        s = self.state
        response: dict[str, Any] = {
            "status": s.status_string,
            "data": s.data_string,
            "frame": s.frame,
            "error": s.error,
            "violations": s.violations,
            "root": s.root,
            "upload_id": s.upload_id,
            "data_hash": s.data_hash,
            "param_hash": s.param_hash,
            "protocol_version": PROTOCOL_VERSION,
            "hardware": self._hardware,
            "git_branch": self._git_branch,
        }

        sim_running = s.solver == Solver.RUNNING
        live_frame = s.frame

        # Build progress
        if s.build == Build.BUILDING:
            response["progress"] = s.build_progress
            response["info"] = s.build_info

        # Simulation progress + live stats
        elif sim_running and self._app:
            try:
                try:
                    total_frames = self._app.app.session._param.get("frames")
                except Exception:
                    total_frames = 0
                if total_frames > 0:
                    response["progress"] = live_frame / total_frames
                summary = {"frame": str(live_frame)}
                try:
                    summary.update(self._app.app.session.get.log.summary())
                except Exception as e:
                    logging.error(f"summary() error: {e}")
                try:
                    from server import _get_runtime_usage
                    summary.update(_get_runtime_usage())
                except Exception:
                    pass
                response["summary"] = summary
            except Exception as e:
                logging.error(f"make_response solver stats error: {e}")

        # Post-simulation average stats
        if not sim_running and self._app:
            try:
                avg = self._app.app.session.get.log.average_summary()
                if avg:
                    response["average_summary"] = avg
            except Exception:
                pass

        # Scene info (whenever we hold an app, even one lazy-loaded
        # from app_state.pickle because Jupyter built it). self._app is
        # the authoritative signal; s.build may lag the on-disk truth.
        if self._app is not None:
            try:
                response["scene_info"] = self._get_scene_info()
            except Exception as e:
                logging.error(f"scene_info error: {e}")

        return response

    def _get_scene_info(self) -> dict:
        """Collect scene statistics from the built app. No filesystem I/O."""
        fs = self._app.app.scene
        info = {}
        def fmt(n):
            return f"{n:,}"

        if hasattr(fs, '_vert') and fs._vert is not None:
            info["Vertices"] = fmt(len(fs._vert[0]))
        if hasattr(fs, '_tri') and fs._tri is not None:
            info["Triangles"] = fmt(len(fs._tri))
        if hasattr(fs, '_tet') and fs._tet is not None:
            info["Tetrahedra"] = fmt(len(fs._tet))
        if hasattr(fs, '_rod') and fs._rod is not None and len(fs._rod) > 0:
            info["Rod Edges"] = fmt(len(fs._rod))
        if hasattr(fs, '_map_by_name'):
            # Count dynamic vs static from pickle data
            n_dynamic = 0
            n_static = 0
            if hasattr(self._app.app, '_scene_decoder') and self._app.app._scene_decoder:
                sd = self._app.app._scene_decoder
                if hasattr(sd, '_data'):
                    for group in sd._data:
                        gt = group.get("type", "")
                        count = len(group.get("object", []))
                        if gt == "STATIC":
                            n_static += count
                        else:
                            n_dynamic += count
            info["Dynamic Objects"] = fmt(n_dynamic)
            info["Static Objects"] = fmt(n_static)
        # Mesh dedup: group shared mesh instances by type
        if hasattr(self._app.app, '_scene_decoder') and self._app.app._scene_decoder:
            sd = self._app.app._scene_decoder
            if hasattr(sd, '_data'):
                # ref_name -> {type, count} (count includes the canonical + all refs)
                ref_groups = {}  # mesh_ref name -> group_type
                canonical_refs = {}  # mesh_ref name -> set of object names sharing it
                for group in sd._data:
                    group_type = group.get("type", "")
                    for obj in group.get("object", []):
                        mesh_ref = obj.get("mesh_ref")
                        if mesh_ref:
                            ref_groups.setdefault(mesh_ref, group_type)
                            canonical_refs.setdefault(mesh_ref, set()).add(obj.get("name", ""))
                            canonical_refs[mesh_ref].add(mesh_ref)  # Include canonical itself
                if canonical_refs:
                    # Group by type: {"SOLID": [4, 2], "SHELL": [3]}
                    by_type = {}
                    for ref_name, names in canonical_refs.items():
                        gt = ref_groups.get(ref_name, "")
                        by_type.setdefault(gt, []).append(len(names))
                    for gt, counts in sorted(by_type.items()):
                        counts.sort(reverse=True)
                        label = f"Shared {gt.capitalize()}s"
                        info[label] = "(" + ",".join(str(c) for c in counts) + ")"
        # Session parameters: frames and FPS
        fss = self._app.app.session
        if fss and hasattr(fss, '_param') and fss._param is not None:
            frames = fss._param.get("frames")
            if frames is not None:
                info["Total Frames"] = fmt(int(frames))
            fps = fss._param.get("fps")
            if fps is not None:
                info["FPS"] = str(int(fps))
            # Count simulated frames and saved checkpoints
            try:
                import glob
                output_dir = os.path.join(fss.info.path, "output")
                verts = glob.glob(os.path.join(output_dir, "vert_*.bin"))
                info["Simulated Frames"] = fmt(len(verts))
                saves = glob.glob(os.path.join(output_dir, "save_*.bin"))
                if saves:
                    max_save = 0
                    for sv in saves:
                        try:
                            idx = int(os.path.basename(sv).replace("save_", "").replace(".bin", ""))
                            max_save = max(max_save, idx)
                        except ValueError:
                            pass
                    info["Last Saved"] = str(max_save)
                else:
                    info["Last Saved"] = "None"
            except Exception:
                pass
        return info

    # -- project context helper --

    def select_project(self, name: str, root: str) -> None:
        """Convenience: dispatch ProjectSelected with filesystem checks.

        Reconciles the on-disk upload_id: if data and param files exist,
        load upload_id.txt (or mint one for projects that predate id
        tracking) and pass it to the transition so state mirrors disk.
        """
        has_data = os.path.exists(os.path.join(root, DATA_NAME))
        has_param = os.path.exists(os.path.join(root, PARAM_NAME))
        # upload_id.txt is the authoritative source when data is on disk.
        # If the data files exist but upload_id.txt is missing, that's an
        # invariant violation (project dir predates upload_id tracking or
        # was written out-of-band). Raise so the user sees it, rather than
        # silently minting a new id that pretends nothing is wrong.
        id_path = os.path.join(root, UPLOAD_ID_FILE)
        if has_data and has_param:
            if not os.path.exists(id_path):
                raise RuntimeError(
                    f"Project {root} has data.pickle + param.pickle but no "
                    f"upload_id.txt. This project predates upload_id tracking; "
                    f"delete the project dir and re-upload from Blender."
                )
            uid = read_upload_id(root)
        else:
            uid = ""
        # Hashes are best-effort: legacy projects and out-of-band
        # uploads have no data_hash.txt / param_hash.txt, in which
        # case the server echoes "" and the client treats every edit
        # as divergent (the safe default).
        dh = read_data_hash(root)
        ph = read_param_hash(root)
        self.dispatch(ProjectSelected(
            name=name,
            root=root,
            has_data=has_data,
            has_param=has_param,
            has_app=self._app is not None and self._app.name == name,
            is_resumable=self._app.resumable() if self._app and self._app.name == name else False,
            upload_id=uid,
            data_hash=dh,
            param_hash=ph,
        ))


def _new_upload_id() -> str:
    """Return a fresh upload_id (12 hex chars, same format as client session_id)."""
    import uuid as _uuid
    return _uuid.uuid4().hex[:12]


def read_upload_id(root: str) -> str:
    """Read upload_id.txt from *root*. Raises if absent.

    Does not mint. Minting belongs in handle_data_transfer, where a real
    write is happening. Callers must guard with ``os.path.exists`` before
    calling if the file may legitimately be missing.
    """
    with open(os.path.join(root, UPLOAD_ID_FILE)) as f:
        uid = f.read().strip()
    if not uid:
        raise ValueError(f"upload_id.txt at {root} is empty")
    return uid


def read_data_hash(root: str) -> str:
    """Read data_hash.txt from *root*. Returns "" when absent."""
    path = os.path.join(root, DATA_HASH_FILE)
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read().strip()


def write_data_hash(root: str, data_hash: str) -> None:
    """Persist *data_hash* to ``data_hash.txt`` under *root* atomically.
    Empty hash deletes the file."""
    import time as _time
    path = os.path.join(root, DATA_HASH_FILE)
    if not data_hash:
        if os.path.exists(path):
            os.remove(path)
        return
    tmp = f"{path}.tmp.{os.getpid()}.{_time.time_ns()}"
    try:
        with open(tmp, "w") as f:
            f.write(data_hash)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_param_hash(root: str) -> str:
    """Read param_hash.txt from *root*. Returns "" when the file is
    absent (legacy projects, or data-only uploads where the previous
    hash is preserved by the caller)."""
    path = os.path.join(root, PARAM_HASH_FILE)
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read().strip()


def write_param_hash(root: str, param_hash: str) -> None:
    """Persist *param_hash* to ``param_hash.txt`` under *root* atomically.
    Empty hash deletes the file (so legacy / data-only uploads don't
    inherit a stale value)."""
    import time as _time
    path = os.path.join(root, PARAM_HASH_FILE)
    if not param_hash:
        if os.path.exists(path):
            os.remove(path)
        return
    tmp = f"{path}.tmp.{os.getpid()}.{_time.time_ns()}"
    try:
        with open(tmp, "w") as f:
            f.write(param_hash)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write_upload_id(root: str, upload_id: str) -> None:
    """Persist *upload_id* to upload_id.txt under *root* atomically.

    Writes to a temp file and renames into place so a concurrent reader
    never sees an empty or half-written id. Raises on I/O error.
    """
    import time as _time
    path = os.path.join(root, UPLOAD_ID_FILE)
    tmp = f"{path}.tmp.{os.getpid()}.{_time.time_ns()}"
    try:
        with open(tmp, "w") as f:
            f.write(upload_id)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
