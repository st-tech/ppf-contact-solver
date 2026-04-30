# File: test_redraw.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression tests for UI-redraw / race-condition fixes. Runs inside
# Blender via the debug port; no stubs or host pytest needed.
#
# Usage from a host shell:
#
#     echo 'import ppf_contact_solver.tests.test_redraw as t; print(t.run_all())' | \
#         python blender_addon/debug/main.py exec -
#
# Each test function begins with `test_`. `run_all()` discovers them,
# runs each, captures exceptions, and returns a summary dict.

import time
import traceback
import types
from unittest import mock

import ppf_contact_solver.core.async_op as async_op_mod
import ppf_contact_solver.core.utils as utils_mod
import ppf_contact_solver.ui.connection_ops as connection_ops_mod
import ppf_contact_solver.ui.main_panel as main_panel_mod
import ppf_contact_solver.ui.object_group as object_group_mod
import ppf_contact_solver.ui.state as state_mod
from ppf_contact_solver.core.client import communicator as com
from ppf_contact_solver.ui.main_panel import GlobalStateWatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_context():
    """Build a context stub sufficient for modal()/timer cleanup paths."""
    ctx = types.SimpleNamespace()
    ctx.window_manager = types.SimpleNamespace(
        event_timer_remove=lambda t: None,
        event_timer_add=lambda **kw: None,
        modal_handler_add=lambda op: None,
    )
    ctx.screen = types.SimpleNamespace(areas=[])
    ctx.window = None
    return ctx


def _mock_event(event_type="TIMER"):
    return types.SimpleNamespace(type=event_type)


def _fake_connect_op(**overrides):
    """Build a fake REMOTE_OT_Connect-shaped self. modal() only reads
    `_connection_established`, `_timer`, `_start_time`, `timeout`, and
    calls `self.report(...)`."""
    base = dict(
        _connection_established=False,
        _timer=None,
        _start_time=time.time(),
        timeout=60.0,
        report=lambda *a, **kw: None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _fake_async_op(**overrides):
    """Build an AsyncOperator-shaped self with all hooks defaulted."""
    base = dict(
        _timer=None,
        _start_time=time.time(),
        timeout=60.0,
        auto_redraw=False,
        _is_stale_class=lambda: False,
        cleanup_modal=lambda c: None,
        is_cancelled=lambda: False,
        is_complete=lambda: False,
        on_complete=lambda c: None,
        on_timeout=lambda c: None,
        report=lambda *a, **kw: None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# REMOTE_OT_Connect.modal() ordering race
# ---------------------------------------------------------------------------

def test_connect_modal_fast_success_before_cancel_check():
    """Regression: fast connect (phase -> ONLINE before first tick)
    must be detected as success, not misread as cancellation."""
    op = _fake_connect_op()
    with mock.patch.object(com, "is_connected", return_value=True), \
         mock.patch.object(com, "is_connecting", return_value=False), \
         mock.patch.object(connection_ops_mod, "redraw_all_areas") as redraw:
        result = connection_ops_mod.REMOTE_OT_Connect.modal(op, _mock_context(), _mock_event())
    assert result == {"PASS_THROUGH"}, f"expected PASS_THROUGH, got {result!r}"
    assert op._connection_established is True, "fast-success transition missed"
    assert redraw.called, "redraw_all_areas was not called on success transition"


def test_connect_modal_cancels_when_neither_pending_nor_connected():
    """Genuine cancellation path still works: not connecting, not connected."""
    op = _fake_connect_op()
    with mock.patch.object(com, "is_connected", return_value=False), \
         mock.patch.object(com, "is_connecting", return_value=False):
        result = connection_ops_mod.REMOTE_OT_Connect.modal(op, _mock_context(), _mock_event())
    assert result == {"CANCELLED"}, f"expected CANCELLED, got {result!r}"


def test_connect_modal_stays_running_while_connecting():
    op = _fake_connect_op()
    with mock.patch.object(com, "is_connected", return_value=False), \
         mock.patch.object(com, "is_connecting", return_value=True), \
         mock.patch.object(connection_ops_mod, "refresh_ssh_panel"):
        result = connection_ops_mod.REMOTE_OT_Connect.modal(op, _mock_context(), _mock_event())
    assert result == {"PASS_THROUGH"}, f"expected PASS_THROUGH, got {result!r}"
    assert op._connection_established is False


def test_connect_modal_non_timer_passes_through():
    op = _fake_connect_op()
    result = connection_ops_mod.REMOTE_OT_Connect.modal(op, _mock_context(), _mock_event("MOUSEMOVE"))
    assert result == {"PASS_THROUGH"}


# ---------------------------------------------------------------------------
# AsyncOperator.modal() CANCELLED paths must tag redraw
# ---------------------------------------------------------------------------

def test_async_op_stale_class_cancel_redraws():
    op = _fake_async_op(_is_stale_class=lambda: True)
    with mock.patch.object(async_op_mod, "redraw_all_areas") as redraw:
        result = async_op_mod.AsyncOperator.modal(op, _mock_context(), _mock_event())
    assert result == {"CANCELLED"}
    assert redraw.called, "stale-class CANCELLED path did not tag redraw"


def test_async_op_is_cancelled_redraws():
    op = _fake_async_op(is_cancelled=lambda: True)
    with mock.patch.object(async_op_mod, "redraw_all_areas") as redraw:
        result = async_op_mod.AsyncOperator.modal(op, _mock_context(), _mock_event())
    assert result == {"CANCELLED"}
    assert redraw.called, "is_cancelled() CANCELLED path did not tag redraw"


def test_async_op_timeout_redraws():
    op = _fake_async_op(_start_time=time.time() - 999.0, timeout=1.0)
    with mock.patch.object(async_op_mod, "redraw_all_areas") as redraw:
        result = async_op_mod.AsyncOperator.modal(op, _mock_context(), _mock_event())
    assert result == {"CANCELLED"}
    assert redraw.called, "timeout CANCELLED path did not tag redraw"


def test_async_op_complete_still_fires_on_complete():
    called = {"n": 0}
    def _on_complete(ctx):
        called["n"] += 1
    op = _fake_async_op(is_complete=lambda: True, on_complete=_on_complete)
    result = async_op_mod.AsyncOperator.modal(op, _mock_context(), _mock_event())
    assert result == {"FINISHED"}
    assert called["n"] == 1


def test_async_op_auto_redraw_true_calls_redraw_each_tick():
    op = _fake_async_op(auto_redraw=True)
    with mock.patch.object(async_op_mod, "redraw_all_areas") as redraw:
        result = async_op_mod.AsyncOperator.modal(op, _mock_context(), _mock_event())
    assert result == {"PASS_THROUGH"}
    assert redraw.called, "auto_redraw=True did not tag redraw on tick"


def test_async_op_auto_redraw_false_skips_tick_redraw():
    op = _fake_async_op(auto_redraw=False)
    with mock.patch.object(async_op_mod, "redraw_all_areas") as redraw:
        result = async_op_mod.AsyncOperator.modal(op, _mock_context(), _mock_event())
    assert result == {"PASS_THROUGH"}
    assert not redraw.called, "auto_redraw=False should not redraw on a normal tick"


# ---------------------------------------------------------------------------
# auto_redraw flag presence on subclasses that need per-tick UI sync
# ---------------------------------------------------------------------------

def test_auto_redraw_flags_on_expected_subclasses():
    from ppf_contact_solver.ui.connection_ops import (
        REMOTE_OT_StartServer, REMOTE_OT_StopServer,
    )
    from ppf_contact_solver.ui.install_ops import (
        REMOTE_OT_InstallParamiko, REMOTE_OT_InstallDocker,
    )
    from ppf_contact_solver.ui.debug_ops import (
        DEBUG_OT_ExecuteServer, DEBUG_OT_TransferWithoutBuild, DEBUG_OT_GitPullLocal,
    )
    from ppf_contact_solver.ui.solver import (
        SOLVER_OT_Transfer, SOLVER_OT_UpdateParams, SOLVER_OT_DeleteRemoteData,
        SOLVER_OT_Run, SOLVER_OT_Resume, SOLVER_OT_FetchData,
    )
    expected = [
        REMOTE_OT_StartServer, REMOTE_OT_StopServer,
        REMOTE_OT_InstallParamiko, REMOTE_OT_InstallDocker,
        DEBUG_OT_ExecuteServer, DEBUG_OT_TransferWithoutBuild, DEBUG_OT_GitPullLocal,
        SOLVER_OT_Transfer, SOLVER_OT_UpdateParams, SOLVER_OT_DeleteRemoteData,
        SOLVER_OT_Run, SOLVER_OT_Resume, SOLVER_OT_FetchData,
    ]
    missing = [c.__name__ for c in expected if getattr(c, "auto_redraw", False) is not True]
    assert not missing, f"classes missing auto_redraw=True: {missing}"


# ---------------------------------------------------------------------------
# GlobalStateWatcher widened signal coverage
# ---------------------------------------------------------------------------

class _FakeInfo:
    def __init__(self, **kw):
        self.status = kw.get("status", "DISCONNECTED")
        self.progress = kw.get("progress", 0.0)
        self.traffic = kw.get("traffic", "")


class _FakeCom:
    def __init__(self):
        self.info = _FakeInfo()
        self.message = ""
        self._is_connected = False
        self._is_connecting = False
        self._is_server_running = False
        self._is_server_launching = False
    def is_connected(self): return self._is_connected
    def is_connecting(self): return self._is_connecting
    def is_server_running(self): return self._is_server_running
    def is_server_launching(self): return self._is_server_launching


def _with_fake_com(fake):
    return mock.patch.multiple(
        main_panel_mod,
        com=fake,
        get_installing_status=mock.MagicMock(return_value=False),
        get_install_result=mock.MagicMock(return_value=None),
    )


def test_watcher_detects_phase_change():
    fake = _FakeCom()
    with _with_fake_com(fake):
        w = GlobalStateWatcher()
        w.reset()
        fake._is_connected = True
        assert w.has_changed(), "widened watcher missed is_connected flip"


def test_watcher_detects_connecting_change():
    fake = _FakeCom()
    with _with_fake_com(fake):
        w = GlobalStateWatcher()
        w.reset()
        fake._is_connecting = True
        assert w.has_changed()


def test_watcher_detects_server_running_change():
    fake = _FakeCom()
    with _with_fake_com(fake):
        w = GlobalStateWatcher()
        w.reset()
        fake._is_server_running = True
        assert w.has_changed()


def test_watcher_detects_message_change():
    fake = _FakeCom()
    with _with_fake_com(fake):
        w = GlobalStateWatcher()
        w.reset()
        fake.message = "Building..."
        assert w.has_changed()


def test_watcher_detects_traffic_change():
    fake = _FakeCom()
    with _with_fake_com(fake):
        w = GlobalStateWatcher()
        w.reset()
        fake.info.traffic = "2.4 MB/s"
        assert w.has_changed()


def test_watcher_steady_state_returns_false():
    fake = _FakeCom()
    with _with_fake_com(fake):
        w = GlobalStateWatcher()
        w.reset()
        assert not w.has_changed(), "has_changed() returned True without any state change"


# ---------------------------------------------------------------------------
# Profile-selected update callbacks
# ---------------------------------------------------------------------------

def test_on_profile_selected_calls_redraw():
    self_ = types.SimpleNamespace(
        profile_selection="my_profile",
        profile_path="/tmp/fake.toml",
    )
    with mock.patch("ppf_contact_solver.core.profile.apply_profile"), \
         mock.patch("ppf_contact_solver.core.profile.load_profiles",
                    return_value={"my_profile": object()}), \
         mock.patch.object(utils_mod, "redraw_all_areas") as redraw:
        state_mod._on_profile_selected(self_, _mock_context())
    assert redraw.called


def test_on_profile_selected_noop_for_NONE():
    self_ = types.SimpleNamespace(profile_selection="NONE", profile_path="")
    with mock.patch.object(utils_mod, "redraw_all_areas") as redraw:
        state_mod._on_profile_selected(self_, _mock_context())
    assert not redraw.called


def test_on_scene_profile_selected_calls_redraw_and_invalidate():
    import ppf_contact_solver.models.groups as groups_mod
    self_ = types.SimpleNamespace(
        scene_profile_selection="my_profile",
        scene_profile_path="/tmp/fake.toml",
    )
    with mock.patch("ppf_contact_solver.core.profile.apply_scene_profile"), \
         mock.patch("ppf_contact_solver.core.profile.load_profiles",
                    return_value={"my_profile": object()}), \
         mock.patch.object(utils_mod, "redraw_all_areas") as redraw, \
         mock.patch.object(groups_mod, "invalidate_overlays") as invalidate:
        state_mod._on_scene_profile_selected(self_, _mock_context())
    assert redraw.called
    assert invalidate.called


def test_on_material_profile_selected_calls_redraw():
    self_ = types.SimpleNamespace(
        material_profile_selection="my_profile",
        material_profile_path="/tmp/fake.toml",
    )
    with mock.patch("ppf_contact_solver.core.profile.apply_material_profile"), \
         mock.patch("ppf_contact_solver.core.profile.load_profiles",
                    return_value={"my_profile": object()}), \
         mock.patch.object(utils_mod, "redraw_all_areas") as redraw:
        object_group_mod._on_material_profile_selected(self_, _mock_context())
    assert redraw.called


def test_on_pin_profile_selected_calls_redraw_and_invalidate():
    import ppf_contact_solver.models.groups as groups_mod
    # pin_vertex_groups needs __len__ and __getitem__; use a list-like
    pin_item = types.SimpleNamespace()
    self_ = types.SimpleNamespace(
        pin_profile_selection="my_profile",
        pin_profile_path="/tmp/fake.toml",
        pin_vertex_groups=[pin_item],
        pin_vertex_groups_index=0,
    )
    with mock.patch("ppf_contact_solver.core.profile.apply_pin_operations"), \
         mock.patch("ppf_contact_solver.core.profile.load_profiles",
                    return_value={"my_profile": object()}), \
         mock.patch.object(utils_mod, "redraw_all_areas") as redraw, \
         mock.patch.object(groups_mod, "invalidate_overlays") as invalidate:
        object_group_mod._on_pin_profile_selected(self_, _mock_context())
    assert redraw.called
    assert invalidate.called


# ---------------------------------------------------------------------------
# State-machine scenarios (chained transitions via the pure transition() fn)
# ---------------------------------------------------------------------------
#
# core/test_transitions.py already covers single events. These tests chain
# multiple events to exercise the flows UI code actually observes — and
# anchor the race/redraw fixes against the state-machine layer that drives
# them.

from ppf_contact_solver.core.state import (
    AppState, Phase, Server, Solver, Activity,
)
from ppf_contact_solver.core.events import (
    ConnectRequested, Connected, ConnectionFailed, DisconnectRequested,
    StartServerRequested, ServerLaunched, StopServerRequested, ServerStopped,
    ServerLost, ServerPolled, AbortRequested, BuildRequested, PollTick,
)
from ppf_contact_solver.core.transitions import transition
from ppf_contact_solver.core.protocol import PROTOCOL_VERSION


def _apply_events(initial, events):
    """Chain transition() across a sequence, returning the final state
    and the flat list of all effects produced along the way."""
    s = initial
    all_fx = []
    for ev in events:
        s, fx = transition(s, ev)
        all_fx.extend(fx)
    return s, all_fx


def test_transition_full_connect_lifecycle():
    """OFFLINE -> CONNECTING -> ONLINE -> OFFLINE."""
    s, _ = _apply_events(AppState(), [
        ConnectRequested("ssh", {"host": "x"}, 9090),
        Connected(remote_root="/remote"),
        DisconnectRequested(),
    ])
    assert s.phase == Phase.OFFLINE
    assert s.remote_root == ""
    assert s.server == Server.UNKNOWN


def test_transition_full_server_lifecycle():
    """After connect: UNKNOWN -> LAUNCHING -> RUNNING -> STOPPING -> UNKNOWN."""
    online = AppState(phase=Phase.ONLINE)
    s, _ = _apply_events(online, [
        StartServerRequested(),
        ServerLaunched(),
        StopServerRequested(),
        ServerStopped(),
    ])
    assert s.phase == Phase.ONLINE
    assert s.server == Server.UNKNOWN
    assert s.solver == Solver.NO_DATA


def test_transition_fast_connect_no_intervening_events():
    """ConnectRequested immediately followed by Connected (no tick between).
    Mirrors the race the Connect modal now guards against — the state
    machine itself must land at ONLINE regardless of intervening ticks."""
    s, _ = _apply_events(AppState(), [
        ConnectRequested("local", {}, 9090),
        Connected(remote_root="/local"),
    ])
    assert s.phase == Phase.ONLINE
    assert s.remote_root == "/local"


def test_transition_connection_failed_during_connecting():
    s, fx = _apply_events(AppState(), [
        ConnectRequested("ssh", {"host": "x"}, 9090),
        ConnectionFailed("auth denied"),
    ])
    assert s.phase == Phase.OFFLINE
    assert s.error == "auth denied"


def test_transition_disconnect_while_server_running_clears_state():
    """Disconnect at any point must flatten phase AND server AND solver."""
    s0 = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY)
    s, _ = _apply_events(s0, [DisconnectRequested()])
    assert s.phase == Phase.OFFLINE
    assert s.server == Server.UNKNOWN
    assert s.solver == Solver.NO_DATA


def test_transition_server_lost_while_simulating():
    """ServerLost while Solver.RUNNING clears simulation activity back to IDLE."""
    s0 = AppState(
        phase=Phase.ONLINE, server=Server.RUNNING,
        solver=Solver.RUNNING, activity=Activity.FETCHING,
    )
    s, _ = _apply_events(s0, [ServerLost()])
    assert s.server == Server.UNKNOWN
    assert s.activity == Activity.IDLE


def test_transition_poll_tick_during_connecting_does_not_clobber_phase():
    """A spurious PollTick while CONNECTING must not reset phase."""
    s0 = AppState(phase=Phase.CONNECTING)
    s, _ = _apply_events(s0, [PollTick()])
    assert s.phase == Phase.CONNECTING


def test_transition_connect_then_server_start_full_path():
    s, _ = _apply_events(AppState(), [
        ConnectRequested("docker", {"container": "c"}, 9090),
        Connected(remote_root="/r"),
        StartServerRequested(),
        ServerLaunched(),
        ServerPolled({
            "status": "READY",
            "protocol_version": PROTOCOL_VERSION,
            "upload_id": "",
        }),
    ])
    assert s.phase == Phase.ONLINE
    assert s.server == Server.RUNNING
    assert s.solver == Solver.READY
    assert s.can_operate is True


def test_transition_abort_during_build_clears_build_activity():
    s0 = AppState(
        phase=Phase.ONLINE, server=Server.RUNNING,
        solver=Solver.BUILDING, activity=Activity.BUILDING,
    )
    s, _ = _apply_events(s0, [AbortRequested()])
    assert s.activity == Activity.ABORTING


# ---------------------------------------------------------------------------
# to_remote_status() mapping — what the UI reads from engine state
# ---------------------------------------------------------------------------

def test_to_remote_status_offline_maps_to_disconnected():
    from ppf_contact_solver.core.status import RemoteStatus
    assert AppState(phase=Phase.OFFLINE).to_remote_status() == RemoteStatus.DISCONNECTED


def test_to_remote_status_connecting_maps_to_connecting():
    from ppf_contact_solver.core.status import RemoteStatus
    assert AppState(phase=Phase.CONNECTING).to_remote_status() == RemoteStatus.CONNECTING


def test_to_remote_status_online_server_running_ready_maps_to_ready():
    from ppf_contact_solver.core.status import RemoteStatus
    s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, solver=Solver.READY)
    # Accepted values: READY or WAITING_FOR_DATA depending on mapping policy.
    mapped = s.to_remote_status()
    assert mapped in (RemoteStatus.READY, RemoteStatus.WAITING_FOR_DATA), mapped


def test_to_remote_status_protocol_mismatch():
    from ppf_contact_solver.core.status import RemoteStatus
    s = AppState(phase=Phase.ONLINE, server=Server.RUNNING, version_ok=False)
    assert s.to_remote_status() == RemoteStatus.PROTOCOL_VERSION_MISMATCH


# ---------------------------------------------------------------------------
# Integration: state-machine transitions observed by the watcher and modal
# ---------------------------------------------------------------------------

def _fake_com_from_state(state_obj):
    """Build a _FakeCom whose predicates reflect `state_obj`."""
    fake = _FakeCom()
    fake._is_connected = state_obj.phase == Phase.ONLINE
    fake._is_connecting = state_obj.phase == Phase.CONNECTING
    fake._is_server_running = state_obj.server == Server.RUNNING
    fake._is_server_launching = state_obj.server == Server.LAUNCHING
    fake.info.status = state_obj.to_remote_status()
    fake.info.progress = state_obj.progress
    fake.info.traffic = state_obj.traffic
    fake.message = state_obj.message
    return fake


def test_watcher_observes_connect_transition():
    """Drive AppState through ConnectRequested → Connected and assert the
    watcher flags the phase change via its widened signal coverage."""
    before = AppState()
    after_reqs, _ = transition(before, ConnectRequested("ssh", {"host": "x"}, 9090))
    after_connected, _ = transition(after_reqs, Connected(remote_root="/r"))

    fake = _fake_com_from_state(before)
    with _with_fake_com(fake):
        w = GlobalStateWatcher()
        w.reset()
        # Advance the fake com to the post-Connected state
        new = _fake_com_from_state(after_connected)
        fake._is_connected = new._is_connected
        fake._is_connecting = new._is_connecting
        fake.info.status = new.info.status
        assert w.has_changed()


def test_watcher_observes_server_launch_transition():
    connected = AppState(phase=Phase.ONLINE)
    launching, _ = transition(connected, StartServerRequested())
    running, _ = transition(launching, ServerLaunched())

    fake = _fake_com_from_state(connected)
    with _with_fake_com(fake):
        w = GlobalStateWatcher()
        w.reset()
        new = _fake_com_from_state(running)
        fake._is_server_running = new._is_server_running
        fake._is_server_launching = new._is_server_launching
        fake.info.status = new.info.status
        assert w.has_changed()


def test_connect_modal_reaches_established_after_connected_transition():
    """End-to-end on the Connect flow: simulate phase transition through
    the state machine, wire is_connected() to the resulting phase, invoke
    the modal — the fast-success fix must mark _connection_established."""
    connecting = AppState(phase=Phase.CONNECTING)
    online, _ = transition(connecting, Connected(remote_root="/r"))
    assert online.phase == Phase.ONLINE  # precondition

    op = _fake_connect_op()
    # After the Connected event, com.is_connected() == True and
    # com.is_connecting() == False.
    with mock.patch.object(com, "is_connected", return_value=True), \
         mock.patch.object(com, "is_connecting", return_value=False), \
         mock.patch.object(connection_ops_mod, "redraw_all_areas") as redraw:
        result = connection_ops_mod.REMOTE_OT_Connect.modal(op, _mock_context(), _mock_event())
    assert result == {"PASS_THROUGH"}
    assert op._connection_established is True
    assert redraw.called


def test_start_server_async_reaches_finished_when_server_running():
    """Simulate UNKNOWN -> LAUNCHING -> RUNNING and verify the
    REMOTE_OT_StartServer is_complete()/is_cancelled() hooks agree with
    the post-transition state."""
    s0 = AppState(phase=Phase.ONLINE, server=Server.UNKNOWN)
    launching, _ = transition(s0, StartServerRequested())
    running, _ = transition(launching, ServerLaunched())

    from ppf_contact_solver.ui.connection_ops import REMOTE_OT_StartServer

    # Hooks ignore self — call as unbound methods so we don't have to
    # instantiate a bpy_struct subclass (which Blender forbids).
    with mock.patch.object(com, "is_server_running", return_value=running.server == Server.RUNNING), \
         mock.patch.object(com, "is_server_launching", return_value=running.server == Server.LAUNCHING):
        assert REMOTE_OT_StartServer.is_complete(None) is True
        assert REMOTE_OT_StartServer.is_cancelled(None) is False


def test_start_server_async_is_cancelled_when_neither_launching_nor_running():
    """Failure mid-launch: both LAUNCHING and RUNNING false -> is_cancelled True."""
    from ppf_contact_solver.ui.connection_ops import REMOTE_OT_StartServer
    with mock.patch.object(com, "is_server_running", return_value=False), \
         mock.patch.object(com, "is_server_launching", return_value=False):
        assert REMOTE_OT_StartServer.is_cancelled(None) is True


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_all():
    tests = sorted(
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    passed = []
    failed = []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
        except Exception as exc:
            failed.append({
                "test": name,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
            })
    return {
        "total": len(tests),
        "passed": len(passed),
        "failed_count": len(failed),
        "failed": failed,
    }
