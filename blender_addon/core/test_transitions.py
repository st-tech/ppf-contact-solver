# File: test_transitions.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Unit tests for the pure transition function.
# Run with: python -m pytest blender_addon/core/test_transitions.py -v
#
# These tests demonstrate the key benefit of the event-driven architecture:
# the entire state machine is testable with NO mocks, NO Blender, NO network.
# Just pure functions and plain asserts.

import sys
import os

# Add parent directory so imports work outside Blender
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Stub out modules that require Blender before any blender_addon imports.
import types
import importlib

# Create stubs for bpy and friends
bpy_stub = types.ModuleType("bpy")
bpy_stub.types = types.ModuleType("bpy.types")
bpy_stub.props = types.ModuleType("bpy.props")
bpy_stub.app = types.ModuleType("bpy.app")
bpy_stub.app.timers = types.ModuleType("bpy.app.timers")
sys.modules["bpy"] = bpy_stub
sys.modules["bpy.types"] = bpy_stub.types
sys.modules["bpy.props"] = bpy_stub.props
sys.modules["bpy.app"] = bpy_stub.app
sys.modules["bpy.app.timers"] = bpy_stub.app.timers

mathutils_stub = types.ModuleType("mathutils")
mathutils_stub.Matrix = type("Matrix", (), {})
mathutils_stub.Vector = type("Vector", (), {})
sys.modules["mathutils"] = mathutils_stub

# Prevent blender_addon/__init__.py from loading UI modules.
# We only need the core sub-package, so register a dummy package.
ba_pkg = types.ModuleType("blender_addon")
ba_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..")]
ba_pkg.__package__ = "blender_addon"
sys.modules["blender_addon"] = ba_pkg

# Now import the modules we actually need (they are self-contained core code).
import blender_addon.models as _models_pkg
import blender_addon.models.console as _console_mod
import blender_addon.models.defaults as _defaults_mod

from blender_addon.core.state import AppState, Phase, Server, Solver, Activity
from blender_addon.core.events import (
    ConnectRequested,
    Connected,
    ConnectionFailed,
    DisconnectRequested,
    StartServerRequested,
    ServerLaunched,
    StopServerRequested,
    ServerStopped,
    ServerLost,
    ServerPolled,
    BuildRequested,
    RunRequested,
    ResumeRequested,
    AbortRequested,
    BuildPipelineRequested,
    UploadOnlyRequested,
    UploadPipelineComplete,
    SendDataRequested,
    SendDataComplete,
    ReceiveDataRequested,
    ReceiveDataComplete,
    FetchRequested,
    FetchComplete,
    FetchFailed,
    AllFramesApplied,
    ProgressUpdated,
    ErrorOccurred,
    ExecRequested,
    ExecComplete,
    PollTick,
)
from blender_addon.core.effects import (
    DoConnect,
    DoDisconnect,
    DoQuery,
    DoLaunchServer,
    DoStopServer,
    DoLog,
    DoClearAnimation,
    DoClearInterrupt,
    DoSetInterrupt,
    DoSendData,
    DoUploadAtomic,
    DoReceiveData,
    DoFetchMap,
    DoFetchFrames,
    DoExec,
    DoRedrawUI,
    DoValidateRemotePath,
    DoTerminate,
    Effect,
)
from blender_addon.core.transitions import transition
from blender_addon.core.protocol import PROTOCOL_VERSION


def _has_effect(effects: list[Effect], cls: type) -> bool:
    return any(isinstance(e, cls) for e in effects)


def _get_effect(effects: list[Effect], cls: type):
    for e in effects:
        if isinstance(e, cls):
            return e
    return None


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------

class TestConnection:
    def test_connect_from_offline(self):
        s = AppState()
        s2, fx = transition(s, ConnectRequested("ssh", {"host": "x"}, 9090))
        assert s2.phase == Phase.CONNECTING
        assert s2.error == ""
        assert _has_effect(fx, DoConnect)

    def test_connect_ignored_when_already_online(self):
        s = AppState(phase=Phase.ONLINE)
        s2, fx = transition(s, ConnectRequested("ssh", {"host": "x"}, 9090))
        assert s2 == s  # No change
        assert fx == []

    def test_connected_event(self):
        s = AppState(phase=Phase.CONNECTING)
        s2, fx = transition(s, Connected(remote_root="/home/user"))
        assert s2.phase == Phase.ONLINE
        assert s2.remote_root == "/home/user"
        assert _has_effect(fx, DoQuery)
        assert _has_effect(fx, DoValidateRemotePath)

    def test_connection_failed(self):
        s = AppState(phase=Phase.CONNECTING)
        s2, fx = transition(s, ConnectionFailed("timeout"))
        assert s2.phase == Phase.OFFLINE
        assert s2.error == "timeout"
        assert _has_effect(fx, DoLog)

    def test_disconnect(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY)
        s2, fx = transition(s, DisconnectRequested())
        assert s2.phase == Phase.OFFLINE
        assert s2.server == Server.UNKNOWN
        assert s2.solver == Solver.NO_DATA
        assert _has_effect(fx, DoDisconnect)
        assert _has_effect(fx, DoClearAnimation)


# ---------------------------------------------------------------------------
# Server lifecycle tests
# ---------------------------------------------------------------------------

class TestServer:
    def test_start_server(self):
        s = AppState(phase=Phase.ONLINE, server=Server.UNKNOWN)
        s2, fx = transition(s, StartServerRequested())
        assert s2.server == Server.LAUNCHING
        assert _has_effect(fx, DoLaunchServer)

    def test_start_server_ignored_when_busy(self):
        s = AppState(phase=Phase.ONLINE, activity=Activity.SENDING)
        s2, fx = transition(s, StartServerRequested())
        assert s2 == s
        assert fx == []

    def test_server_launched(self):
        s = AppState(phase=Phase.ONLINE, server=Server.LAUNCHING)
        s2, fx = transition(s, ServerLaunched())
        assert s2.server == Server.RUNNING
        assert _has_effect(fx, DoQuery)

    def test_stop_server(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, StopServerRequested())
        assert s2.server == Server.STOPPING
        assert _has_effect(fx, DoStopServer)

    def test_cancel_server_launch(self):
        s = AppState(phase=Phase.ONLINE, server=Server.LAUNCHING)
        s2, fx = transition(s, StopServerRequested())
        assert s2.server == Server.STOPPING
        assert _has_effect(fx, DoStopServer)

    def test_server_stopped(self):
        s = AppState(phase=Phase.ONLINE, server=Server.STOPPING, solver=Solver.READY)
        s2, fx = transition(s, ServerStopped())
        assert s2.server == Server.UNKNOWN
        assert s2.solver == Solver.NO_DATA

    def test_server_lost(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.RUNNING)
        s2, fx = transition(s, ServerLost())
        assert s2.server == Server.UNKNOWN
        assert s2.activity == Activity.IDLE


# ---------------------------------------------------------------------------
# Server response interpretation tests
# ---------------------------------------------------------------------------

class TestServerResponse:
    def test_ready_response(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ServerPolled({
            "status": "READY",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.solver == Solver.READY
        assert s2.server == Server.RUNNING

    def test_resumable_response(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ServerPolled({
            "status": "RESUMABLE",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.solver == Solver.RESUMABLE

    def test_busy_response_with_frames(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ServerPolled({
            "status": "BUSY",
            "frame": 5,
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.solver == Solver.RUNNING
        assert s2.frame == 5
        assert _has_effect(fx, DoFetchFrames)  # Auto-fetch

    def test_busy_response_frame_zero(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ServerPolled({
            "status": "BUSY",
            "frame": 0,
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.solver == Solver.STARTING

    def test_failed_response(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ServerPolled({
            "status": "FAILED",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.solver == Solver.FAILED

    def test_error_overrides_ready_to_failed(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ServerPolled({
            "status": "READY",
            "error": "Segmentation fault",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.solver == Solver.FAILED
        assert s2.server_error == "Segmentation fault"

    def test_empty_response_marks_server_unknown(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ServerPolled({}))
        assert s2.server == Server.UNKNOWN

    def test_protocol_mismatch(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ServerPolled({
            "status": "READY",
            "protocol_version": "99.99",
        }))
        assert s2.version_ok is False

    def test_building_clears_activity(self):
        s = AppState(
            phase=Phase.ONLINE,
            server=Server.RUNNING,
            activity=Activity.BUILDING,
        )
        s2, fx = transition(s, ServerPolled({
            "status": "READY",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.activity == Activity.IDLE
        assert s2.solver == Solver.READY

    def test_no_data_response(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ServerPolled({
            "status": "NO_DATA",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.solver == Solver.NO_DATA

    def test_save_and_quit_response(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.RUNNING)
        s2, fx = transition(s, ServerPolled({
            "status": "SAVE_AND_QUIT",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.solver == Solver.SAVING


# ---------------------------------------------------------------------------
# Solver operation tests
# ---------------------------------------------------------------------------

class TestSolverOps:
    def test_build(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY)
        s2, fx = transition(s, BuildRequested())
        assert s2.activity == Activity.BUILDING
        q = _get_effect(fx, DoQuery)
        assert q is not None
        assert q.request == {"request": "build"}

    def test_build_resets_active_upload_id(self):
        # BuildRequested must clear any stale pinned upload_id from a
        # previous build so the next poll re-pins from the current
        # server state.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY,
            active_upload_id="stale-id",
        )
        s2, _ = transition(s, BuildRequested())
        assert s2.active_upload_id == ""

    def test_build_clears_progress_and_error(self):
        # Stale progress/error from a prior run or build must not bleed
        # into a fresh build's UI state.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY,
            progress=0.7, error="prior failure",
        )
        s2, _ = transition(s, BuildRequested())
        assert s2.progress == 0.0
        assert s2.error == ""

    def test_build_rejected_when_offline(self):
        s = AppState(phase=Phase.OFFLINE)
        s2, fx = transition(s, BuildRequested())
        assert s2 == s
        assert fx == []

    def test_run(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY)
        s2, fx = transition(s, RunRequested())
        assert s2.solver == Solver.STARTING
        assert s2.progress == 0.0
        assert _has_effect(fx, DoClearAnimation)
        assert _has_effect(fx, DoClearInterrupt)
        q = _get_effect(fx, DoQuery)
        assert q.request == {"request": "start"}

    def test_run_clears_frame_and_error(self):
        # A new run's transition must wipe the prior run's frame/error
        # tail. Without this, a poll that lands between RunRequested
        # and the start-command response can surface stale frame as
        # if it belonged to the new run, false-positive saw_running.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY,
            frame=9, error="prior run failure",
        )
        s2, _ = transition(s, RunRequested())
        assert s2.frame == 0
        assert s2.error == ""

    def test_run_rejected_when_busy(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.READY, activity=Activity.SENDING,
        )
        s2, fx = transition(s, RunRequested())
        assert s2 == s

    def test_resume(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.RESUMABLE)
        s2, fx = transition(s, ResumeRequested())
        assert s2.solver == Solver.STARTING
        q = _get_effect(fx, DoQuery)
        assert q.request == {"request": "resume"}

    def test_resume_clears_frame_and_error(self):
        # Same staleness contract as RunRequested. The server's first
        # poll after resume repopulates frame with the saved index.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.RESUMABLE,
            frame=5, error="stale",
        )
        s2, _ = transition(s, ResumeRequested())
        assert s2.frame == 0
        assert s2.error == ""

    def test_resume_rejected_when_not_resumable(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY)
        s2, fx = transition(s, ResumeRequested())
        assert s2 == s

    def test_abort(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.RUNNING, activity=Activity.IDLE,
        )
        s2, fx = transition(s, AbortRequested())
        assert s2.activity == Activity.ABORTING
        assert _has_effect(fx, DoSetInterrupt)
        assert _has_effect(fx, DoClearAnimation)


# ---------------------------------------------------------------------------
# Data transfer tests
# ---------------------------------------------------------------------------

class TestDataTransfer:
    def test_send(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, SendDataRequested("/tmp/data", b"hello", "Sending..."))
        assert s2.activity == Activity.SENDING
        assert s2.message == "Sending..."
        assert _has_effect(fx, DoSendData)

    def test_send_complete(self):
        s = AppState(activity=Activity.SENDING, progress=0.5)
        s2, fx = transition(s, SendDataComplete())
        assert s2.activity == Activity.IDLE
        assert s2.progress == 0.0
        assert _has_effect(fx, DoRedrawUI)

    def test_receive(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ReceiveDataRequested("/tmp/data", "Receiving..."))
        assert s2.activity == Activity.RECEIVING
        assert _has_effect(fx, DoReceiveData)

    def test_send_rejected_when_busy(self):
        s = AppState(activity=Activity.FETCHING)
        s2, fx = transition(s, SendDataRequested("/tmp/x", b"data"))
        assert s2 == s


# ---------------------------------------------------------------------------
# Pipeline (upload + build) tests
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_build_pipeline_starts_upload(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY,
            remote_root="/home/sim",
        )
        s2, fx = transition(s, BuildPipelineRequested(
            data=b"scene", param=b"params", message="Uploading...",
        ))
        assert s2.activity == Activity.SENDING
        assert s2.pending_build is True
        eff = _get_effect(fx, DoUploadAtomic)
        assert eff is not None
        assert eff.project_root == "/home/sim"
        assert eff.data == b"scene"
        assert eff.param == b"params"

    def test_build_pipeline_rejected_without_payload(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY,
            remote_root="/home/sim",
        )
        s2, fx = transition(s, BuildPipelineRequested(data=b"", param=b""))
        assert s2 == s
        assert not _has_effect(fx, DoUploadAtomic)

    def test_build_pipeline_rejected_without_remote_root(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY,
        )
        s2, fx = transition(s, BuildPipelineRequested(data=b"scene"))
        assert s2 == s

    def test_upload_complete_transitions_to_building(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            activity=Activity.SENDING, pending_build=True,
        )
        s2, fx = transition(s, UploadPipelineComplete())
        assert s2.activity == Activity.BUILDING
        assert s2.solver == Solver.BUILDING
        assert s2.active_upload_id == ""
        assert s2.pending_build is False
        q = _get_effect(fx, DoQuery)
        assert q is not None and q.request == {"request": "build"}

    def test_upload_only_stays_idle_after_complete(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY,
            remote_root="/home/sim",
        )
        s2, fx = transition(s, UploadOnlyRequested(data=b"scene"))
        assert s2.activity == Activity.SENDING
        assert s2.pending_build is False
        assert _has_effect(fx, DoUploadAtomic)

        s3, _ = transition(s2, UploadPipelineComplete())
        assert s3.activity == Activity.IDLE
        assert s3.solver == Solver.READY
        assert s3.pending_build is False

    def test_error_clears_pending_build(self):
        s = AppState(
            activity=Activity.SENDING, pending_build=True,
        )
        s2, _ = transition(s, ErrorOccurred(error="boom"))
        assert s2.activity == Activity.IDLE
        assert s2.pending_build is False

    def test_abort_clears_pending_build(self):
        s = AppState(
            activity=Activity.SENDING, pending_build=True,
        )
        s2, _ = transition(s, AbortRequested())
        assert s2.activity == Activity.ABORTING
        assert s2.pending_build is False


# ---------------------------------------------------------------------------
# Fetch tests
# ---------------------------------------------------------------------------

class TestFetch:
    def test_fetch_starts_with_map_download(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            remote_root="/home/sim",
        )
        s2, fx = transition(s, FetchRequested())
        assert s2.activity == Activity.FETCHING
        assert _has_effect(fx, DoFetchMap)

    def test_fetch_complete_transitions_to_applying(self):
        # Each phase owns its own 0→1 bar; apply restarts at 0.
        s = AppState(activity=Activity.FETCHING, progress=1.0)
        s2, fx = transition(s, FetchComplete(total_frames=10))
        assert s2.activity == Activity.APPLYING
        assert s2.progress == 0.0

    def test_fetch_complete_with_zero_frames(self):
        s = AppState(activity=Activity.FETCHING)
        s2, fx = transition(s, FetchComplete(total_frames=0))
        assert s2.activity == Activity.IDLE

    def test_all_frames_applied(self):
        s = AppState(activity=Activity.APPLYING)
        s2, fx = transition(s, AllFramesApplied())
        assert s2.activity == Activity.IDLE
        assert s2.progress == 0.0

    def test_all_frames_applied_ignored_when_not_applying(self):
        # Stale AllFramesApplied from a prior run must not bounce us out
        # of IDLE or a fresh FETCHING.
        for activity in (Activity.IDLE, Activity.FETCHING):
            s = AppState(activity=activity, progress=0.25)
            s2, fx = transition(s, AllFramesApplied())
            assert s2 == s
            assert fx == []

    def test_fetch_failed_during_fetching(self):
        s = AppState(activity=Activity.FETCHING, progress=0.3)
        s2, fx = transition(s, FetchFailed(reason="no backend"))
        assert s2.activity == Activity.IDLE
        assert s2.progress == 0.0
        assert s2.error == "no backend"

    def test_fetch_failed_during_applying(self):
        s = AppState(activity=Activity.APPLYING, progress=0.6)
        s2, fx = transition(s, FetchFailed(reason="apply: oom"))
        assert s2.activity == Activity.IDLE
        assert s2.progress == 0.0
        assert s2.error == "apply: oom"

    def test_fetch_failed_ignored_when_idle(self):
        s = AppState(activity=Activity.IDLE)
        s2, fx = transition(s, FetchFailed(reason="stale"))
        assert s2 == s
        assert fx == []


# ---------------------------------------------------------------------------
# Progress and error tests
# ---------------------------------------------------------------------------

class TestProgressAndError:
    def test_progress_update(self):
        s = AppState()
        s2, fx = transition(s, ProgressUpdated(0.5, "10 MB/s"))
        assert s2.progress == 0.5
        assert s2.traffic == "10 MB/s"
        assert fx == []

    def test_error_resets_activity(self):
        s = AppState(activity=Activity.SENDING, progress=0.5)
        s2, fx = transition(s, ErrorOccurred("timeout", "send"))
        assert s2.activity == Activity.IDLE
        assert s2.error == "timeout"
        assert s2.progress == 0.0
        assert _has_effect(fx, DoLog)


# ---------------------------------------------------------------------------
# Exec tests
# ---------------------------------------------------------------------------

class TestExec:
    def test_exec(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, fx = transition(s, ExecRequested("ls", True))
        assert s2.activity == Activity.EXECUTING
        assert _has_effect(fx, DoExec)

    def test_exec_complete(self):
        s = AppState(activity=Activity.EXECUTING)
        s2, fx = transition(s, ExecComplete({"exit_code": 0, "stdout": ["ok"]}))
        assert s2.activity == Activity.IDLE
        assert _has_effect(fx, DoRedrawUI)


# ---------------------------------------------------------------------------
# Poll tick tests
# ---------------------------------------------------------------------------

class TestPollTick:
    def test_poll_tick_during_simulation(self):
        s = AppState(solver=Solver.RUNNING)
        s2, fx = transition(s, PollTick())
        assert _has_effect(fx, DoQuery)

    def test_poll_tick_when_idle(self):
        s = AppState(solver=Solver.READY)
        s2, fx = transition(s, PollTick())
        assert fx == []


# ---------------------------------------------------------------------------
# State property tests
# ---------------------------------------------------------------------------

class TestStateProperties:
    def test_busy(self):
        assert not AppState().busy
        assert AppState(activity=Activity.SENDING).busy

    def test_can_operate(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        assert s.can_operate
        assert not AppState().can_operate
        assert not AppState(phase=Phase.ONLINE, server=Server.RUNNING, activity=Activity.SENDING).can_operate
        assert not AppState(phase=Phase.ONLINE, server=Server.RUNNING, version_ok=False).can_operate


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestSolverDrivenPolling:
    """Verify polling and activity clearing are purely solver-driven."""

    def test_poll_tick_during_building(self):
        s = AppState(solver=Solver.BUILDING, activity=Activity.IDLE)
        s2, fx = transition(s, PollTick())
        assert _has_effect(fx, DoQuery)

    def test_poll_tick_building_regardless_of_activity(self):
        s = AppState(solver=Solver.BUILDING, activity=Activity.BUILDING)
        s2, fx = transition(s, PollTick())
        assert _has_effect(fx, DoQuery)

    def test_poll_tick_stops_on_ready(self):
        s = AppState(solver=Solver.READY)
        s2, fx = transition(s, PollTick())
        assert fx == []

    def test_building_activity_cleared_on_solver_ready(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.BUILDING, activity=Activity.BUILDING,
        )
        s2, fx = transition(s, ServerPolled({
            "status": "READY",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.solver == Solver.READY
        assert s2.activity == Activity.IDLE

    def test_building_with_empty_server_upload_id_is_desync(self):
        # Client is BUILDING but the server reports no upload_id — that
        # means the server has no data at all, which can't be the build
        # target. Treat as desync (the stuck-state bug we're fixing).
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.BUILDING, activity=Activity.BUILDING,
        )
        s2, fx = transition(s, ServerPolled({
            "status": "NO_DATA",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.activity == Activity.IDLE
        assert s2.solver == Solver.NO_DATA
        assert s2.error != ""
        assert s2.active_upload_id == ""

    def test_building_pins_upload_id_on_first_poll(self):
        # First poll after BuildRequested: active_upload_id is empty,
        # server has a real upload_id, so we pin it.  No desync.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.BUILDING, activity=Activity.BUILDING,
        )
        s2, fx = transition(s, ServerPolled({
            "status": "BUILDING",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "abc123",
        }))
        assert s2.active_upload_id == "abc123"
        assert s2.activity == Activity.BUILDING
        assert s2.solver == Solver.BUILDING

    def test_building_no_build_with_matching_upload_id_is_race_not_terminal(self):
        # Regression for the upload+build pipeline race:
        #   engine.dispatch is async (main-thread tick), so the worker's
        #   inline auto-query-after-upload can land a status poll at the
        #   server *before* the queued DoQuery("build") runs. That poll
        #   returns NO_BUILD (data uploaded, build not yet requested).
        #   If we treat that as terminal and clear Activity.BUILDING, the
        #   modal reports "complete" before the build even starts. We
        #   must keep Activity.BUILDING pinned until the server shows
        #   READY / RESUMABLE / FAILED.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.BUILDING, activity=Activity.BUILDING,
            active_upload_id="abc123",
        )
        s2, fx = transition(s, ServerPolled({
            "status": "NO_BUILD",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "abc123",
        }))
        assert s2.activity == Activity.BUILDING  # preserved across the race
        assert s2.solver == Solver.NO_BUILD
        assert s2.active_upload_id == "abc123"  # still pinned
        assert s2.error == ""  # not a desync

    def test_building_ready_response_clears_activity(self):
        # READY is an unambiguous terminal for BUILDING — build succeeded.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.BUILDING, activity=Activity.BUILDING,
            active_upload_id="abc123",
        )
        s2, fx = transition(s, ServerPolled({
            "status": "READY",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "abc123",
        }))
        assert s2.activity == Activity.IDLE
        assert s2.solver == Solver.READY
        assert s2.active_upload_id == ""

    def test_building_no_build_with_error_is_terminal_failure(self):
        # The server's status_string maps both Build.NONE (never
        # attempted) and Build.FAILED (attempted, decoding failed) to
        # "NO_BUILD" because neither leaves a usable build artifact.
        # The error_msg is what tells them apart: a NO_BUILD response
        # carrying an error during BUILDING activity is a real build
        # failure and must clear activity (so the user can fix the
        # scene and re-upload), not a race-window ping.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.BUILDING, activity=Activity.BUILDING,
            active_upload_id="abc123",
        )
        s2, fx = transition(s, ServerPolled({
            "status": "NO_BUILD",
            "error": "decoding failed: 1 self-intersections (1 tri-tri).",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "abc123",
        }))
        assert s2.activity == Activity.IDLE
        assert s2.solver == Solver.FAILED
        assert s2.active_upload_id == ""
        assert "self-intersect" in (s2.server_error or "")

    def test_aborting_no_build_still_clears_activity(self):
        # ABORTING (e.g. user cancelled a build) DOES treat NO_BUILD as
        # terminal — the cancel has landed and the build is gone. This
        # preserves the old abort semantics while tightening BUILDING.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.BUILDING, activity=Activity.ABORTING,
        )
        s2, fx = transition(s, ServerPolled({
            "status": "NO_BUILD",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "abc123",
        }))
        assert s2.activity == Activity.IDLE
        assert s2.solver == Solver.NO_BUILD

    def test_server_missing_protocol_version_flags_mismatch(self):
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, _ = transition(s, ServerPolled({"status": "READY", "upload_id": ""}))
        assert s2.version_ok is False

    def test_server_missing_upload_id_flags_mismatch(self):
        # 0.03 contract: every status response carries an upload_id field.
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, _ = transition(s, ServerPolled({
            "status": "READY",
            "protocol_version": PROTOCOL_VERSION,
        }))
        assert s2.version_ok is False

    def test_error_only_response_preserves_solver_and_activity(self):
        # Error-only response from server.py's handle_text_command
        # exception path: empty status + non-empty error. The client
        # must surface the error without touching solver/activity, so
        # an in-flight BUILDING or SENDING survives a transient server
        # hiccup instead of snapping back to IDLE.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.BUILDING, activity=Activity.BUILDING,
            active_upload_id="pinned-id",
        )
        s2, fx = transition(s, ServerPolled({
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
            "status": "",
            "error": "select_project: upload_id.txt missing",
        }))
        assert s2.solver == Solver.BUILDING  # preserved
        assert s2.activity == Activity.BUILDING  # preserved
        assert s2.active_upload_id == "pinned-id"  # preserved
        assert s2.server_error == "select_project: upload_id.txt missing"
        assert s2.version_ok is True  # not a protocol error

    def test_empty_status_without_error_flags_malformed(self):
        # Status and error both empty is a malformed protocol 0.03 reply;
        # flag as version mismatch rather than silently preserving state.
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING)
        s2, _ = transition(s, ServerPolled({
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
            "status": "",
        }))
        assert s2.version_ok is False

    def test_unknown_status_string_is_logged(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.READY,
        )
        s2, fx = transition(s, ServerPolled({
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
            "status": "TOTALLY_BOGUS",
        }))
        # Solver is preserved, an error log is emitted so the mismatch
        # doesn't go silent.
        assert s2.solver == Solver.READY
        assert any(
            isinstance(e, DoLog) and "Unknown server status" in e.message
            for e in fx
        )

    def test_building_upload_id_changed_is_desync(self):
        # Server's upload_id differs from our pinned one: another client
        # uploaded new data under us, or the server was reset and new
        # data was uploaded.
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.BUILDING, activity=Activity.BUILDING,
            active_upload_id="abc123",
        )
        s2, fx = transition(s, ServerPolled({
            "status": "BUILDING",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "xyz999",
        }))
        assert s2.activity == Activity.IDLE
        assert s2.error != ""
        assert s2.active_upload_id == ""

    def test_executing_cleared_on_terminal_solver(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.NO_DATA, activity=Activity.EXECUTING,
        )
        s2, fx = transition(s, ServerPolled({
            "status": "NO_DATA",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.activity == Activity.IDLE

    def test_aborting_cleared_on_ready(self):
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.RUNNING, activity=Activity.ABORTING,
        )
        s2, fx = transition(s, ServerPolled({
            "status": "READY",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.activity == Activity.IDLE

    def test_sending_not_cleared_by_server_poll(self):
        """SENDING is cleared by SendDataComplete, not by server poll."""
        s = AppState(
            phase=Phase.ONLINE, server=Server.RUNNING,
            solver=Solver.NO_DATA, activity=Activity.SENDING,
        )
        s2, fx = transition(s, ServerPolled({
            "status": "NO_DATA",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }))
        assert s2.activity == Activity.SENDING  # Not cleared


class TestBackwardCompat:
    def test_to_remote_status_disconnected(self):
        from blender_addon.core.status import RemoteStatus
        s = AppState()
        assert s.to_remote_status() == RemoteStatus.DISCONNECTED

    def test_to_remote_status_ready(self):
        from blender_addon.core.status import RemoteStatus
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY)
        assert s.to_remote_status() == RemoteStatus.READY

    def test_to_remote_status_simulation(self):
        from blender_addon.core.status import RemoteStatus
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.RUNNING)
        assert s.to_remote_status() == RemoteStatus.SIMULATION_IN_PROGRESS

    def test_to_remote_status_sending_data(self):
        from blender_addon.core.status import RemoteStatus
        s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, activity=Activity.SENDING)
        assert s.to_remote_status() == RemoteStatus.DATA_SENDING


if __name__ == "__main__":
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        # Fallback: run all test classes manually
        import traceback
        passed = 0
        failed = 0
        errors = []
        for name, obj in list(globals().items()):
            if isinstance(obj, type) and name.startswith("Test"):
                inst = obj()
                for method_name in dir(inst):
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
                traceback.print_exception(type(e), e, e.__traceback__)
            sys.exit(1)
