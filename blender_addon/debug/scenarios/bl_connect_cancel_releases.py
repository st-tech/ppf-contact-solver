# File: scenarios/bl_connect_cancel_releases.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# CANCEL MUST FREE THE ADD-ON, AND AN ABANDONED ATTEMPT MUST NOT LAND
# (``blender_addon/core/effect_runner.py``, ``blender_addon/ui/connection_ops.py``).
#
# WHAT WAS REPORTED. Connect to a remote host that is not up; it does not
# connect, so click Cancel; then connect to the local native solver, and that
# sits at "Connecting..." for good. Restarting Blender connects on the first
# try. "It seems like the state machine is broken."
#
# WHAT IT WAS. Three separate holes, each of which on its own reads as a wedged
# panel:
#
#   * A connect was submitted to the I/O WORKER, the single thread every other
#     operation queues onto. A paramiko connect carried no timeout, so an
#     attempt against a powered-off box sat in the transport for the OS TCP
#     retry schedule. Cancel returns the state machine to OFFLINE but cannot
#     interrupt that thread, so the next connect -- a native one, needing
#     nothing but a local socket -- waited behind a host the user had already
#     given up on.
#   * The abandoned attempt still reported. ``ConnectionFailed`` from the dead
#     host replaced the state of the attempt the user started instead, and a
#     ``Connected`` from one would have put the add-on ONLINE against a
#     connection it had walked away from.
#   * The connect operator's own 60 s timeout returned CANCELLED without
#     tearing the attempt down, leaving phase=CONNECTING with nothing watching
#     it. The reducer accepts a connect request only from OFFLINE, so every
#     later click was dropped in silence: the restart was the only way out.
#
# WHAT THE FIX IS. A connection attempt runs on a thread of its own and carries
# a TICKET (``EffectRunner._take_connect_ticket``). Every path that starts,
# cancels or supersedes an attempt takes a new ticket, so an attempt holding a
# stale one closes whatever it opened and dispatches nothing; adopting the
# backend and testing the ticket are one step under a lock, so the backend has
# exactly one owner whichever side wins. The handshake is bounded
# (``backends.SSH_HANDSHAKE_TIMEOUT_S``), and the operator's timeout
# disconnects.
#
# HOW IT IS MEASURED. Inside Blender, but against an Engine and EffectRunner
# the scenario builds itself, so it neither needs nor disturbs a live
# connection. ``create_backend`` is replaced by a stand-in whose blocking
# attempt is held on an Event: the test decides when a dead host answers, which
# a sleep could not. The verdict for the reported symptom is therefore a
# BOOLEAN, not a duration -- the second connect must land WHILE the first is
# still in flight.
#
# Subtests:
#   A. cancel_returns_the_phase_to_offline
#   B. the_next_connect_does_not_wait_for_the_abandoned_one
#   C. the_connection_is_the_one_just_asked_for
#   D. an_abandoned_failure_is_not_reported_over_the_live_connection
#   E. an_abandoned_attempt_that_succeeds_closes_itself
#   F. a_connect_while_connecting_is_refused_out_loud
#   G. a_cancel_during_the_probe_keeps_the_add_on_offline
#   H. an_attempt_outliving_the_runner_closes_itself
#   I. every_paramiko_connect_is_bounded
#   J. the_connect_timeout_tears_the_attempt_down
#   K. the_modal_does_not_repeat_a_teardown_it_did_not_start
#   L. the_connect_button_is_refused_while_an_attempt_is_in_flight

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it. Nothing here contacts a
# server or a solver: the subject is the runner's own ownership rules, which
# are the same under every backend.
BACKENDS = ("real",)


_DRIVER_TEMPLATE = r"""
import ast, bpy, os, sys, threading, time, traceback
result.setdefault("checks", {})
result.setdefault("errors", [])


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


class _StandIn:
    '''A connection that answers everything, and remembers being closed.'''

    backend_type = "linux_native"
    server_port = 9090

    def __init__(self, label):
        self.label = label
        self.closed = False

    @property
    def current_directory(self):
        return "/tmp/ppf-cancel-releases"

    def exec_command(self, command, shell=False, cwd=None, timeout=None):
        return {"exit_code": 0, "stdout": [], "stderr": []}

    def query(self, request, project, chunk):
        return ({}, True)

    def is_alive(self):
        return True

    def disconnect(self):
        self.closed = True


try:
    effect_runner = __import__(pkg + ".core.effect_runner",
                               fromlist=["EffectRunner"])
    engine_mod = __import__(pkg + ".core.engine", fromlist=["Engine"])
    events = __import__(pkg + ".core.events",
                        fromlist=["ConnectRequested", "DisconnectRequested"])
    state_mod = __import__(pkg + ".core.state",
                           fromlist=["AppState", "Phase"])
    transitions = __import__(pkg + ".core.transitions", fromlist=["transition"])
    backends = __import__(pkg + ".core.backends",
                          fromlist=["SSH_HANDSHAKE_TIMEOUT_S"])
    connection_ops = __import__(pkg + ".ui.connection_ops",
                                fromlist=["REMOTE_OT_Connect"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    Phase = state_mod.Phase

    opened = []

    def make_runner(probe_gate=None):
        '''An Engine and EffectRunner of this scenario's own.

        `create_backend` is the only part of an attempt that can block, so it
        is the only part replaced: `config["hold"]` waits on an Event the way a
        paramiko connect to a powered-off box waits on its transport, and
        `config["fails"]` then raises what that eventually raises.

        The probes and the remote-path check are stubbed because they are other
        scenarios' subjects (`bl_remote_device_select`,
        `bl_connection_path_validation`) and they only refill panel caches;
        what is under test here is who owns the connection. *probe_gate* holds
        the first probe open instead, which is how a cancel is made to land in
        the window AFTER the backend is adopted.
        '''
        def probe(backend):
            if probe_gate is not None:
                probe_gate.wait(timeout=60.0)

        engine = engine_mod.Engine()
        runner = effect_runner.EffectRunner(engine)
        runner.project_name = "cancel_releases"
        runner._probe_solver_host_gpus = probe
        runner._probe_solver_host_builds = lambda backend: None
        runner._do_validate_path = lambda: None
        return engine, runner

    def fake_create_backend(backend_type, config):
        if config.get("hold") is not None:
            if not config["hold"].wait(timeout=60.0):
                raise AssertionError("the held attempt was never released")
        if config.get("fails"):
            raise Exception(
                "ssh: connect to host rig-dead-host port 22: timed out")
        backend = _StandIn(config.get("label", "?"))
        opened.append(backend)
        return backend

    def settle(engine, runner, predicate, seconds=10.0):
        '''Tick the engine until *predicate* holds, or the budget runs out.'''
        deadline = time.time() + seconds
        while True:
            engine.tick(runner)
            if predicate():
                return True
            if time.time() >= deadline:
                return False
            time.sleep(0.02)

    saved_create = effect_runner.create_backend
    effect_runner.create_backend = fake_create_backend
    engine = runner = None
    engine2 = runner2 = None
    engine3 = runner3 = None
    engine4 = runner4 = None
    try:
        # ---- A to D: the reported sequence -----------------------------
        held = threading.Event()
        engine, runner = make_runner()
        engine.dispatch(events.ConnectRequested(
            backend_type="ssh",
            config={"hold": held, "fails": True, "label": "dead"},
            server_port=9090,
        ))
        engine.tick(runner)
        connecting = engine.state.phase is Phase.CONNECTING

        # Cancel, which is what the panel's button dispatches.
        engine.dispatch(events.DisconnectRequested())
        engine.tick(runner)
        record("cancel_returns_the_phase_to_offline",
               connecting and engine.state.phase is Phase.OFFLINE,
               {"during_attempt": connecting.__str__(),
                "after_cancel": engine.state.phase.name})

        # The native connect, with the dead attempt STILL in flight. Landing
        # at all is the verdict: `held` is never set below, so a queued
        # connect could not have run.
        started = time.time()
        engine.dispatch(events.ConnectRequested(
            backend_type="linux_native",
            config={"label": "native"},
            server_port=9090,
        ))
        landed = settle(engine, runner, lambda: engine.state.phase is Phase.ONLINE)
        elapsed = round(time.time() - started, 3)
        record("the_next_connect_does_not_wait_for_the_abandoned_one",
               landed and not held.is_set(),
               {"landed": landed, "seconds": elapsed,
                "first_attempt_still_held": not held.is_set(),
                "phase": engine.state.phase.name})
        record("the_connection_is_the_one_just_asked_for",
               getattr(runner.backend, "label", None) == "native",
               {"backend": getattr(runner.backend, "label", None),
                "opened": [b.label for b in opened]})

        # Now let the dead host answer. Its failure belongs to nobody: the
        # attempt that asked for it was canceled, and reporting it would take
        # the live connection offline and name a host the user is not using.
        held.set()
        settled = settle(
            engine, runner,
            lambda: engine.state.phase is not Phase.ONLINE
            or "rig-dead-host" in engine.state.error,
            seconds=3.0,
        )
        record("an_abandoned_failure_is_not_reported_over_the_live_connection",
               not settled and engine.state.phase is Phase.ONLINE,
               {"phase": engine.state.phase.name, "error": engine.state.error})

        # ---- E: an abandoned attempt that SUCCEEDS ---------------------
        held2 = threading.Event()
        engine2, runner2 = make_runner()
        engine2.dispatch(events.ConnectRequested(
            backend_type="linux_native",
            config={"hold": held2, "label": "late"},
            server_port=9090,
        ))
        engine2.tick(runner2)
        engine2.dispatch(events.DisconnectRequested())
        engine2.tick(runner2)
        held2.set()
        closed = settle(
            engine2, runner2,
            lambda: any(b.label == "late" and b.closed for b in opened),
            seconds=5.0,
        )
        record("an_abandoned_attempt_that_succeeds_closes_itself",
               closed
               and engine2.state.phase is Phase.OFFLINE
               and runner2.backend is None,
               {"closed": closed, "phase": engine2.state.phase.name,
                "backend": getattr(runner2.backend, "label", None),
                "opened": [(b.label, b.closed) for b in opened]})

        # ---- I: a cancel INSIDE the probe window -----------------------
        # The window after the backend is adopted and before `Connected` is
        # dispatched: the solver host is being asked for its GPUs and its
        # builds, which takes a command round trip each. A cancel here must
        # not leave the add-on ONLINE over a transport `_do_disconnect` has
        # already closed, which is the one state nothing recovers from.
        gate = threading.Event()
        engine3, runner3 = make_runner(probe_gate=gate)
        engine3.dispatch(events.ConnectRequested(
            backend_type="linux_native",
            config={"label": "probing"},
            server_port=9090,
        ))
        adopted = settle(engine3, runner3,
                         lambda: runner3.backend is not None, seconds=5.0)
        engine3.dispatch(events.DisconnectRequested())
        engine3.tick(runner3)
        gate.set()
        probing = [b for b in opened if b.label == "probing"]
        quiet = settle(
            engine3, runner3,
            lambda: engine3.state.phase is not Phase.OFFLINE,
            seconds=3.0,
        )
        record("a_cancel_during_the_probe_keeps_the_add_on_offline",
               adopted and not quiet
               and runner3.backend is None
               and all(b.closed for b in probing),
               {"adopted": adopted, "phase": engine3.state.phase.name,
                "backend": getattr(runner3.backend, "label", None),
                "closed": [(b.label, b.closed) for b in probing]})

        # ---- L: an attempt that outlives the runner --------------------
        # Disabling or reloading the add-on calls `stop()`, which cannot join
        # the attempt thread. Taking the ticket is what keeps its result out of
        # a runner the add-on has torn down.
        held4 = threading.Event()
        engine4, runner4 = make_runner()
        engine4.dispatch(events.ConnectRequested(
            backend_type="linux_native",
            config={"hold": held4, "label": "orphan"},
            server_port=9090,
        ))
        engine4.tick(runner4)
        runner4.stop()
        held4.set()
        orphan_closed = settle(
            engine4, runner4,
            lambda: any(b.label == "orphan" and b.closed for b in opened),
            seconds=5.0,
        )
        record("an_attempt_outliving_the_runner_closes_itself",
               orphan_closed
               and runner4.backend is None
               and engine4.state.phase is Phase.CONNECTING,
               {"closed": orphan_closed,
                "backend": getattr(runner4.backend, "label", None),
                "phase": engine4.state.phase.name})
    finally:
        effect_runner.create_backend = saved_create
        for live in (runner, runner2, runner3, runner4):
            if live is not None:
                live.stop()

    # ---- F: the refusal the reducer must not swallow --------------------
    handshaking = state_mod.AppState(phase=Phase.CONNECTING)
    same, effects = transitions.transition(
        handshaking, events.ConnectRequested(backend_type="ssh", config={}))
    logs = [getattr(e, "message", "") for e in effects
            if type(e).__name__ == "DoLog"]
    record("a_connect_while_connecting_is_refused_out_loud",
           same == handshaking
           and not any(type(e).__name__ == "DoConnect" for e in effects)
           and any("ignored" in m for m in logs),
           {"effects": [type(e).__name__ for e in effects], "logs": logs})

    # ---- G: every paramiko connect carries the three caps ---------------
    # Read off the source rather than driven: paramiko is not importable on
    # every rig host, and a connect to a host that is not up is exactly what
    # cannot be run here. An unbounded handshake is the hang the report opens
    # with, so the gate is that no `.connect(hostname=...)` lacks a cap.
    with open(backends.__file__, encoding="utf-8") as handle:
        backends_src = handle.read()
    unbounded = []
    for node in ast.walk(ast.parse(backends_src)):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "connect":
            continue
        keywords = {kw.arg for kw in node.keywords}
        if "hostname" not in keywords:
            continue
        missing = sorted(
            {"timeout", "banner_timeout", "auth_timeout"} - keywords)
        if missing:
            unbounded.append({"line": node.lineno, "missing": missing})
    # Read with a default, so a constant that has GONE fails this check
    # instead of aborting the driver and taking the other seven with it.
    cap = getattr(backends, "SSH_HANDSHAKE_TIMEOUT_S", 0)
    record("every_paramiko_connect_is_bounded",
           not unbounded and cap > 0,
           {"unbounded": unbounded, "cap_seconds": cap})

    # ---- H to J: the operator itself, driven ----------------------------
    # The timeout branch fires 60 s into a handshake, which no rig run waits
    # for, so the CLOCK is moved rather than the wait: ``modal`` is an ordinary
    # method, and a stand-in whose start time is an hour ago reaches the branch
    # on its first tick. The communicator is replaced for the duration, so what
    # is observed is the operator's own teardown and not a real connection's,
    # and the context is a stand-in because the only thing the branch asks of
    # one is a screen to tag for redraw.
    Connect = connection_ops.REMOTE_OT_Connect
    Cancel = connection_ops.REMOTE_OT_CancelConnect

    class _Com:
        def __init__(self, connecting):
            self.connecting = connecting
            self.disconnects = 0

        def is_connected(self):
            return False

        def is_connecting(self):
            return self.connecting

        def disconnect(self):
            self.disconnects += 1

    class _Screen:
        areas = ()

    class _Ctx:
        screen = _Screen()

    class _Tick:
        type = "TIMER"

    class _Op:
        timeout = 60.0
        modal = Connect.modal

        def __init__(self):
            self._timer = None
            self._connection_established = False
            self._start_time = time.time() - 3600.0
            self.reports = []

        def _detach_timer(self, context):
            pass

        def report(self, level, message):
            self.reports.append((sorted(level), message))

    saved_com = connection_ops.com
    try:
        # H: a handshake that never lands is torn down, not merely left.
        connection_ops.com = _Com(connecting=True)
        timed_out = _Op()
        verdict = timed_out.modal(_Ctx(), _Tick())
        record("the_connect_timeout_tears_the_attempt_down",
               verdict == {"CANCELLED"}
               and connection_ops.com.disconnects == 1
               and [level for level, _ in timed_out.reports] == [["ERROR"]],
               {"verdict": sorted(verdict), "reports": timed_out.reports,
                "disconnects": connection_ops.com.disconnects})

        # I: and a modal finishing because SOMEONE ELSE tore the attempt down
        # (the Cancel button) must not tear it down a second time, which would
        # reach past the attempt into a connection already being made.
        connection_ops.com = _Com(connecting=False)
        already = _Op()
        verdict = already.modal(_Ctx(), _Tick())
        record("the_modal_does_not_repeat_a_teardown_it_did_not_start",
               verdict == {"CANCELLED"}
               and connection_ops.com.disconnects == 0
               and already.reports == [],
               {"verdict": sorted(verdict),
                "disconnects": connection_ops.com.disconnects,
                "reports": already.reports})

        # J: and the button refuses a second request while one is in flight,
        # because the reducer would drop it (subtest F). The idle poll is
        # taken too: without it a False could be any other refusal.
        root = groups.get_addon_data(bpy.context.scene)
        props = root.ssh_state
        root.state.project_name = "cancel_releases"
        native = {"win32": "WIN_NATIVE", "darwin": "MAC_NATIVE"}.get(
            sys.platform, "LINUX_NATIVE")
        props.server_type = native
        setattr(props, connection_ops.NATIVE_PATH_FIELDS[native],
                "/tmp/ppf-cancel-releases")
        connection_ops.com = _Com(connecting=False)
        poll_idle = Connect.poll(bpy.context)
        connection_ops.com = _Com(connecting=True)
        poll_busy = Connect.poll(bpy.context)
        poll_cancel = Cancel.poll(bpy.context)
        record("the_connect_button_is_refused_while_an_attempt_is_in_flight",
               poll_idle is True and poll_busy is False and poll_cancel is True,
               {"server_type": native, "idle": poll_idle, "connecting": poll_busy,
                "cancel_offered": poll_cancel})
    finally:
        connection_ops.com = saved_com
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the driver source. No substitutions: nothing here is host-specific."""
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
