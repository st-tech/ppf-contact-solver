# File: server/test_transitions.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Unit tests for the server transition function.
# Run with: python server/test_transitions.py
#
# No Blender, no solver, no network required. Pure functions and asserts.

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.state import ServerState, Data, Build, Solver
from server.events import (
    ProjectSelected,
    BuildRequested,
    BuildProgress,
    BuildCompleted,
    BuildFailed,
    GPUCheckFailed,
    StartRequested,
    ResumeRequested,
    TerminateRequested,
    SaveAndQuitRequested,
    DeleteRequested,
    SolverFrameUpdated,
    SolverFinished,
    SolverCrashed,
    SolverSaving,
    ErrorOccurred,
    UploadLanded,
)
from server.effects import (
    DoCheckGPU,
    DoSpawnBuild,
    DoLaunchSolver,
    DoKillSolver,
    DoRequestSaveAndQuit,
    DoLoadApp,
    DoDeleteProjectData,
    DoLog,
    Effect,
)
from server.transitions import server_transition


def _has(effects, cls):
    return any(isinstance(e, cls) for e in effects)


def _get(effects, cls):
    for e in effects:
        if isinstance(e, cls):
            return e
    return None


# ---------------------------------------------------------------------------
# Project selection
# ---------------------------------------------------------------------------

class TestProjectSelection:
    def test_new_project_no_data(self):
        s = ServerState()
        s2, fx = server_transition(s, ProjectSelected(
            name="test", root="/tmp/test",
            has_data=False, has_param=False,
        ))
        assert s2.name == "test"
        assert s2.root == "/tmp/test"
        assert s2.data == Data.EMPTY
        assert s2.build == Build.NONE

    def test_new_project_with_data(self):
        s = ServerState()
        s2, fx = server_transition(s, ProjectSelected(
            name="test", root="/tmp/test",
            has_data=True, has_param=True,
        ))
        assert s2.data == Data.UPLOADED
        assert s2.build == Build.NONE
        assert _has(fx, DoLoadApp)

    def test_new_project_with_existing_app(self):
        s = ServerState()
        s2, fx = server_transition(s, ProjectSelected(
            name="test", root="/tmp/test",
            has_data=True, has_param=True,
            has_app=True, is_resumable=True,
        ))
        assert s2.build == Build.BUILT
        assert s2.resumable is True
        assert not _has(fx, DoLoadApp)  # Already loaded

    def test_same_project_refreshes_data(self):
        s = ServerState(name="test", root="/tmp/test", data=Data.EMPTY,
                        build=Build.BUILT, solver=Solver.RUNNING, frame=10)
        s2, fx = server_transition(s, ProjectSelected(
            name="test", root="/tmp/test",
            has_data=True, has_param=True,
        ))
        assert s2.data == Data.UPLOADED
        assert s2.solver == Solver.RUNNING  # Preserved
        assert s2.frame == 10  # Preserved

    def test_project_selected_stamps_upload_id(self):
        s = ServerState()
        s2, _ = server_transition(s, ProjectSelected(
            name="p", root="/tmp/p",
            has_data=True, has_param=True, upload_id="abc123",
        ))
        assert s2.upload_id == "abc123"

    def test_same_project_refreshes_upload_id(self):
        # Server restarts, re-reads upload_id.txt, and passes the disk
        # id via ProjectSelected. State mirrors disk.
        s = ServerState(name="p", root="/tmp/p", data=Data.UPLOADED,
                        upload_id="old")
        s2, _ = server_transition(s, ProjectSelected(
            name="p", root="/tmp/p",
            has_data=True, has_param=True, upload_id="new",
        ))
        assert s2.upload_id == "new"


# ---------------------------------------------------------------------------
# Upload landing
# ---------------------------------------------------------------------------

class TestUploadLanded:
    def test_upload_stamps_id_and_marks_uploaded(self):
        s = ServerState(name="p", root="/tmp/p")
        s2, fx = server_transition(s, UploadLanded(
            upload_id="xyz", has_data=True, has_param=True,
        ))
        assert s2.upload_id == "xyz"
        assert s2.data == Data.UPLOADED
        assert _has(fx, DoLog)

    def test_partial_upload_keeps_data_empty(self):
        s = ServerState(name="p", root="/tmp/p")
        s2, _ = server_transition(s, UploadLanded(
            upload_id="xyz", has_data=True, has_param=False,
        ))
        # Partial upload: stamp id (protects against overwrite) but
        # data stays EMPTY until both files exist.
        assert s2.upload_id == "xyz"
        assert s2.data == Data.EMPTY

    def test_upload_invalidates_built_state(self):
        # A fresh upload means the existing app artifact is for stale
        # data. Reset build to NONE so the next build request is accepted.
        s = ServerState(name="p", root="/tmp/p", data=Data.UPLOADED,
                        build=Build.BUILT, upload_id="old", resumable=True)
        s2, _ = server_transition(s, UploadLanded(
            upload_id="new", has_data=True, has_param=True,
        ))
        assert s2.upload_id == "new"
        assert s2.build == Build.NONE
        assert s2.resumable is False

    def test_upload_during_building_preserves_build_state(self):
        # Don't cancel an in-flight build via state change; the client
        # will detect upload_id mismatch and abort on its side.
        s = ServerState(name="p", root="/tmp/p", data=Data.UPLOADED,
                        build=Build.BUILDING, upload_id="old")
        s2, _ = server_transition(s, UploadLanded(
            upload_id="new", has_data=True, has_param=True,
        ))
        assert s2.upload_id == "new"
        assert s2.build == Build.BUILDING  # Unchanged


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

class TestBuild:
    def test_build_from_uploaded(self):
        s = ServerState(data=Data.UPLOADED, build=Build.NONE)
        s2, fx = server_transition(s, BuildRequested())
        assert s2.build == Build.BUILDING
        assert s2.build_progress == 0.0
        assert s2.error == ""
        assert _has(fx, DoSpawnBuild)

    def test_build_rejected_no_data(self):
        s = ServerState(data=Data.EMPTY)
        s2, fx = server_transition(s, BuildRequested())
        assert s2 == s
        assert fx == []

    def test_build_rejected_already_building(self):
        s = ServerState(data=Data.UPLOADED, build=Build.BUILDING)
        s2, fx = server_transition(s, BuildRequested())
        assert s2 == s

    def test_build_progress(self):
        s = ServerState(build=Build.BUILDING, build_progress=0.1)
        s2, fx = server_transition(s, BuildProgress(0.5, "Encoding meshes..."))
        assert s2.build_progress == 0.5
        assert s2.build_info == "Encoding meshes..."

    def test_build_completed(self):
        s = ServerState(build=Build.BUILDING, build_progress=0.9,
                        root="/tmp/test")
        s2, fx = server_transition(s, BuildCompleted())
        assert s2.build == Build.BUILT
        assert s2.build_progress == 1.0
        assert s2.resumable is False

    def test_build_failed(self):
        s = ServerState(build=Build.BUILDING)
        s2, fx = server_transition(s, BuildFailed("decode error"))
        assert s2.build == Build.FAILED
        assert s2.error == "decode error"

    def test_gpu_check_failed(self):
        s = ServerState(data=Data.UPLOADED, build=Build.BUILDING)
        s2, fx = server_transition(s, GPUCheckFailed("No CUDA device"))
        assert s2.build == Build.FAILED
        assert s2.error == "No CUDA device"

    def test_rebuild_after_built(self):
        s = ServerState(data=Data.UPLOADED, build=Build.BUILT)
        s2, fx = server_transition(s, BuildRequested())
        assert s2.build == Build.BUILDING
        assert _has(fx, DoSpawnBuild)


# ---------------------------------------------------------------------------
# Solver operations
# ---------------------------------------------------------------------------

class TestSolver:
    def test_start(self):
        s = ServerState(build=Build.BUILT, solver=Solver.IDLE)
        s2, fx = server_transition(s, StartRequested())
        assert s2.solver == Solver.RUNNING
        assert s2.frame == 0
        assert s2.error == ""
        assert _has(fx, DoLaunchSolver)

    def test_start_rejected_not_built(self):
        s = ServerState(build=Build.NONE, solver=Solver.IDLE)
        s2, fx = server_transition(s, StartRequested())
        assert s2 == s

    def test_start_rejected_already_running(self):
        s = ServerState(build=Build.BUILT, solver=Solver.RUNNING)
        s2, fx = server_transition(s, StartRequested())
        assert s2 == s

    def test_resume(self):
        s = ServerState(build=Build.BUILT, solver=Solver.IDLE, resumable=True)
        s2, fx = server_transition(s, ResumeRequested())
        assert s2.solver == Solver.RUNNING
        launch = _get(fx, DoLaunchSolver)
        assert launch is not None
        assert launch.resume_from == -1

    def test_resume_rejected_not_resumable(self):
        s = ServerState(build=Build.BUILT, solver=Solver.IDLE, resumable=False)
        s2, fx = server_transition(s, ResumeRequested())
        assert s2 == s

    def test_terminate(self):
        s = ServerState(solver=Solver.RUNNING)
        s2, fx = server_transition(s, TerminateRequested())
        assert s2.solver == Solver.IDLE
        assert _has(fx, DoKillSolver)

    def test_terminate_while_saving(self):
        s = ServerState(solver=Solver.SAVING)
        s2, fx = server_transition(s, TerminateRequested())
        assert s2.solver == Solver.IDLE
        assert _has(fx, DoKillSolver)

    def test_terminate_idle_still_fires(self):
        # TerminateRequested always emits DoKillSolver — it must also
        # work when state.solver is IDLE because an external process
        # (e.g. JupyterLab) may have started a solver the server never
        # tracked. _kill_solver's Utils.busy() check makes it a no-op
        # when nothing is running.
        s = ServerState(solver=Solver.IDLE)
        s2, fx = server_transition(s, TerminateRequested())
        assert s2.solver == Solver.IDLE
        assert _has(fx, DoKillSolver)

    def test_save_and_quit(self):
        s = ServerState(solver=Solver.RUNNING)
        s2, fx = server_transition(s, SaveAndQuitRequested())
        assert s2.solver == Solver.SAVING
        assert _has(fx, DoRequestSaveAndQuit)

    def test_save_and_quit_idle_still_fires(self):
        # Save-and-quit must also work against an externally-started
        # solver (e.g. from JupyterLab) where state.solver is still
        # IDLE. _request_save_and_quit writes the flag file through
        # the loaded app; dispatching ErrorOccurred is its own guard
        # when no app is loaded.
        s = ServerState(solver=Solver.IDLE)
        s2, fx = server_transition(s, SaveAndQuitRequested())
        assert s2.solver == Solver.SAVING
        assert _has(fx, DoRequestSaveAndQuit)


# ---------------------------------------------------------------------------
# Solver monitor events
# ---------------------------------------------------------------------------

class TestMonitorEvents:
    def test_frame_update(self):
        s = ServerState(solver=Solver.RUNNING, frame=5)
        s2, fx = server_transition(s, SolverFrameUpdated(frame=6))
        assert s2.frame == 6
        assert fx == []

    def test_solver_finished(self):
        s = ServerState(solver=Solver.RUNNING, frame=100)
        s2, fx = server_transition(s, SolverFinished(resumable=True))
        assert s2.solver == Solver.IDLE
        assert s2.resumable is True

    def test_solver_finished_not_resumable(self):
        s = ServerState(solver=Solver.RUNNING)
        s2, fx = server_transition(s, SolverFinished(resumable=False))
        assert s2.solver == Solver.IDLE
        assert s2.resumable is False

    def test_solver_crashed(self):
        s = ServerState(solver=Solver.RUNNING, frame=50)
        s2, fx = server_transition(s, SolverCrashed("Segfault"))
        assert s2.solver == Solver.FAILED
        assert s2.error == "Segfault"

    def test_solver_saving(self):
        s = ServerState(solver=Solver.RUNNING)
        s2, fx = server_transition(s, SolverSaving())
        assert s2.solver == Solver.SAVING

    def test_save_finished(self):
        s = ServerState(solver=Solver.SAVING)
        s2, fx = server_transition(s, SolverFinished(resumable=True))
        assert s2.solver == Solver.IDLE
        assert s2.resumable is True


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete(self):
        s = ServerState(name="test", root="/tmp/test", data=Data.UPLOADED,
                        build=Build.BUILT, solver=Solver.IDLE, frame=50)
        s2, fx = server_transition(s, DeleteRequested())
        assert s2.data == Data.EMPTY
        assert s2.build == Build.NONE
        assert s2.solver == Solver.IDLE
        assert s2.frame == 0
        assert s2.name == "test"  # Name preserved
        assert _has(fx, DoKillSolver)
        assert _has(fx, DoDeleteProjectData)

    def test_delete_no_root(self):
        s = ServerState(name="test", root="")
        s2, fx = server_transition(s, DeleteRequested())
        assert s2 == s


# ---------------------------------------------------------------------------
# Status string
# ---------------------------------------------------------------------------

class TestStatusString:
    def test_no_data(self):
        assert ServerState().status_string == "NO_DATA"

    def test_no_build(self):
        assert ServerState(data=Data.UPLOADED).status_string == "NO_BUILD"

    def test_building(self):
        assert ServerState(data=Data.UPLOADED, build=Build.BUILDING).status_string == "BUILDING"

    def test_ready(self):
        assert ServerState(data=Data.UPLOADED, build=Build.BUILT).status_string == "READY"

    def test_resumable(self):
        assert ServerState(data=Data.UPLOADED, build=Build.BUILT, resumable=True).status_string == "RESUMABLE"

    def test_busy(self):
        assert ServerState(data=Data.UPLOADED, build=Build.BUILT, solver=Solver.RUNNING).status_string == "BUSY"

    def test_saving(self):
        assert ServerState(solver=Solver.SAVING).status_string == "SAVE_AND_QUIT"

    def test_failed_solver(self):
        assert ServerState(data=Data.UPLOADED, build=Build.BUILT, solver=Solver.FAILED).status_string == "FAILED"

    def test_failed_build(self):
        assert ServerState(data=Data.UPLOADED, build=Build.FAILED, error="decoding failed").status_string == "NO_BUILD"

    def test_failed_error(self):
        assert ServerState(data=Data.UPLOADED, build=Build.BUILT, error="oops").status_string == "FAILED"

    def test_build_failed(self):
        assert ServerState(data=Data.UPLOADED, build=Build.FAILED).status_string == "NO_BUILD"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class TestError:
    def test_error(self):
        s = ServerState()
        s2, fx = server_transition(s, ErrorOccurred("something broke"))
        assert s2.error == "something broke"
        assert _has(fx, DoLog)


if __name__ == "__main__":
    import traceback as tb
    passed = 0
    failed = 0
    errors = []
    for name, obj in list(globals().items()):
        if isinstance(obj, type) and name.startswith("Test"):
            inst = obj()
            for method_name in sorted(dir(inst)):
                if method_name.startswith("test_"):
                    full_name = f"{name}.{method_name}"
                    try:
                        getattr(inst, method_name)()
                        passed += 1
                        print(f"  PASS  {full_name}")
                    except Exception as e:
                        failed += 1
                        errors.append((full_name, e))
                        print(f"  FAIL  {full_name}: {e}")
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        for name, e in errors:
            print(f"\n  {name}:")
            tb.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)
