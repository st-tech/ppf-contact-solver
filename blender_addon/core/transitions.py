# File: transitions.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Pure state-transition function for the event-driven state machine.
#
# ``transition(state, event) -> (new_state, effects)``
#
# This module is the **single source of truth** for every state change in the
# addon.  It contains ZERO side-effects: no I/O, no locks, no ``bpy`` calls,
# no network access.  It is fully unit-testable with plain ``assert``
# statements.
#
# The function returns a list of ``Effect`` objects that describe what the
# system should *do* in response to the transition.  The ``EffectRunner``
# (effect_runner.py) is responsible for actually executing them.
#
# This is the ADDON-side state machine operating on ``AppState`` (Phase /
# Activity / Server / Solver as observed from the Blender side). It is
# distinct from the SERVER-side state machine in
# ``crates/ppf-cts-core/src/transitions/``, which operates on ``ServerState``
# (Build / Data / Solver as tracked by the solver host). The two state
# machines run on different processes, observe different events, and emit
# different effect sets; they share no types and must not be merged.
# If you change addon-visible state semantics here, also review the server
# transitions for any cross-side invariants (e.g. handshake order).

from __future__ import annotations

from dataclasses import replace

from .protocol import (
    PROTOCOL_VERSION,
    STATUS_BUILDING,
    STATUS_BUSY,
    STATUS_FAILED,
    STATUS_NO_BUILD,
    STATUS_NO_DATA,
    STATUS_READY,
    STATUS_RESUMABLE,
    STATUS_SAVE_AND_QUIT,
)
from .state import Activity, AppState, Phase, Server, Solver
from .events import (
    AbortRequested,
    AllFramesApplied,
    BuildPipelineRequested,
    BuildRequested,
    UploadOnlyRequested,
    Connected,
    ConnectionFailed,
    ConnectionLost,
    DisconnectRequested,
    Disconnected,
    ConnectRequested,
    ErrorOccurred,
    Event,
    ExecComplete,
    ExecRequested,
    FetchComplete,
    FetchFailed,
    FetchMapComplete,
    FetchRequested,
    PollTick,
    ProgressUpdated,
    QueryRequested,
    ReceiveDataComplete,
    ReceiveDataRequested,
    ResumeRequested,
    RunRequested,
    SaveAndQuitRequested,
    SendDataComplete,
    SendDataRequested,
    ServerLaunched,
    ServerLost,
    ServerPolled,
    ServerStopped,
    StartServerRequested,
    StopServerRequested,
    TerminateRequested,
    UploadPipelineComplete,
)
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


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# In-flight stale polls to drain during STARTING. Seeded into
# starting_poll_guard by RunRequested / ResumeRequested and decremented
# per poll by the post-poll guard in _interpret_response.
_STARTING_POLL_GUARD_DEPTH = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transition(state: AppState, event: Event) -> tuple[AppState, list[Effect]]:
    """Pure function: (AppState, Event) -> (new AppState, [Effect ...]).

    This is the ONLY place where ``AppState`` is replaced.  Every ``match``
    arm returns ``(replace(state, ...), [effects])`` or ``(state, [])`` for
    no-op / guard-rejected events.
    """

    match event:

        # ── Connection ─────────────────────────────────────
        case ConnectRequested(backend_type=bt, config=cfg, server_port=sp) \
                if state.phase == Phase.OFFLINE:
            return (
                replace(state, phase=Phase.CONNECTING, error="", server_error=""),
                [DoConnect(bt, cfg, sp)],
            )

        case Connected(remote_root=root, session_id=sid, saved_session_id=saved):
            # A fresh session id is minted on every successful connect so
            # downstream artifacts (PC2 headers, modifier binds, remote
            # directories) can be correlated with THIS run. The id and the
            # last-saved id both arrive on the event: the EffectRunner owns
            # the uuid mint and the bpy read, so this arm stays pure and
            # deterministic. ``sid`` may be empty if the runner could not
            # mint one; treat it as "" rather than substituting a value.
            effects = [DoValidateRemotePath(), DoQuery(), DoLog(f"Connected (session {sid}).")]
            # Reconcile log: warn the user when the .blend was last
            # saved under a different session so they know the on-disk
            # PC2 files may not correspond to anything on this server.
            if saved and saved != sid:
                effects.append(DoLog(
                    f"Reconcile: previous session {saved} differs "
                    f"from new session {sid}; cached PC2 files may "
                    f"not match the remote."
                ))
            return (
                replace(
                    state,
                    phase=Phase.ONLINE,
                    remote_root=root or state.remote_root,
                    session_id=sid,
                    activity=Activity.IDLE,
                    error="",
                    server_error="",
                ),
                effects,
            )

        case ConnectionFailed(error=e):
            return (
                replace(state, phase=Phase.OFFLINE, error=e),
                [DoLog(f"Connection failed: {e}")],
            )

        case DisconnectRequested():
            return (
                _reset_state(state),
                [DoDisconnect(), DoClearAnimation(), DoLog("Disconnected.")],
            )

        case Disconnected():
            return _reset_state(state), []

        case ConnectionLost():
            return (
                _reset_state(state, error="Connection lost."),
                [DoClearAnimation(), DoLog("Connection lost.")],
            )

        # ── Server lifecycle ───────────────────────────────
        case StartServerRequested() \
                if state.phase == Phase.ONLINE and not state.busy:
            return (
                replace(state, server=Server.LAUNCHING),
                [DoLaunchServer()],
            )

        case ServerLaunched():
            return (
                replace(state, server=Server.RUNNING),
                [DoQuery(), DoLog("Server ready.")],
            )

        case StopServerRequested() if state.server in (Server.RUNNING, Server.LAUNCHING):
            return (
                replace(state, server=Server.STOPPING),
                [DoStopServer()],
            )

        case ServerStopped():
            # Deliberate stop: same per-operation status reset as the
            # transient-loss paths, plus a clear of the last server error
            # since this was an intentional shutdown.
            return (
                replace(_server_gone(state), server_error=""),
                [DoClearInterrupt(), DoLog("Server stopped.")],
            )

        case ServerLost():
            return (
                _server_gone(state),
                [DoLog("Server not responding.")],
            )

        # ── Server response interpretation ─────────────────
        case ServerPolled(response=r):
            new_state, effects = _interpret_response(state, r)
            # A user-initiated query (EXECUTING) completes as soon as the
            # response arrives — independent of what solver state the
            # server reports. BUILDING/ABORTING are long-running client
            # activities that wait for the solver to reach a terminal
            # state before they're considered done.
            if state.activity == Activity.EXECUTING:
                new_state = replace(new_state, activity=Activity.IDLE, message="")
            elif new_state.activity == Activity.IDLE:
                # Activity already cleared by _interpret_response (desync
                # path). Nothing further to do.
                pass
            else:
                # Truly terminal solver states — the build/abort definitely
                # resolved. NO_DATA / NO_BUILD are NOT in this set when
                # state.activity is BUILDING, because the upload+build
                # pipeline has a race: ``engine.dispatch`` is asynchronous
                # (queues to main-thread tick), so the worker's inline
                # auto-query-after-upload can land a ``ping`` at the server
                # *before* the queued ``DoQuery("build")`` runs. That ping
                # returns NO_BUILD (data uploaded, build not yet requested),
                # and if we treated it as terminal we'd clear BUILDING
                # prematurely and the modal would report completion before
                # the build even started. Desync-detection inside
                # _interpret_response still catches the true "server state
                # was reset" case via upload_id mismatch.
                if state.activity == Activity.BUILDING:
                    solver_terminal = new_state.solver in (
                        Solver.READY, Solver.RESUMABLE, Solver.FAILED,
                    )
                else:
                    # ABORTING cares about NO_DATA / NO_BUILD too — those
                    # are expected after delete/cancel.
                    solver_terminal = new_state.solver in (
                        Solver.READY, Solver.RESUMABLE, Solver.FAILED,
                        Solver.NO_DATA, Solver.NO_BUILD,
                    )
                if solver_terminal and state.activity in (
                    Activity.BUILDING, Activity.ABORTING,
                ):
                    new_state = replace(
                        new_state,
                        activity=Activity.IDLE,
                        message="",
                        active_upload_id="",
                    )
            return new_state, effects

        case PollTick():
            # Re-query during simulation or build
            if state.solver in (Solver.RUNNING, Solver.STARTING, Solver.SAVING,
                                Solver.BUILDING):
                return state, [DoQuery()]
            # A pending abort must keep polling until a status response
            # confirms the solver reached a terminal state; the ServerPolled
            # handler above turns that into ``activity=IDLE``. The
            # solver-state branch is not enough: an abort issued while the
            # solver is already terminal (cancelling a fetch/send/apply, where
            # the solver stays READY/RESUMABLE the whole time, or a build that
            # finished just before the abort) leaves ABORTING with no reason
            # to re-query, so the engine would sit in ABORTING forever and
            # ``com.busy()`` gate every button (Fetch, Clear, ...) off.
            if state.activity == Activity.ABORTING:
                return state, [DoQuery()]
            # Heartbeat probe: when we're connected but the engine
            # thinks the server is UNKNOWN (a transient query failure
            # during Run/Build dispatched ServerLost), let an empty
            # background poll re-confirm liveness. _do_query only
            # dispatches ServerLost for *user-initiated* requests, so
            # an empty heartbeat that fails stays silent and an
            # empty heartbeat that succeeds flips server back to
            # RUNNING via _interpret_response. Without this branch the
            # addon stays stuck at "Waiting for Server Start..." with
            # a live server until the user manually disconnects and
            # reconnects.
            if (
                state.phase == Phase.ONLINE
                and state.server == Server.UNKNOWN
                and state.activity == Activity.IDLE
            ):
                return state, [DoQuery()]
            return state, []

        # ── Build ──────────────────────────────────────────
        case BuildRequested() if state.can_operate:
            # Reset active_upload_id; the first poll after this request
            # pins it from the server's response.  Any subsequent change
            # in the server's upload_id is then an explicit desync.
            # Reset progress and error so the build UI doesn't carry
            # residue from a prior run/build into this one.
            return (
                replace(
                    state,
                    activity=Activity.BUILDING,
                    solver=Solver.BUILDING,
                    active_upload_id="",
                    progress=0.0,
                    error="",
                ),
                [DoQuery({"request": "build"}), DoLog("Start build...")],
            )

        # ── Run / Resume ───────────────────────────────────
        case RunRequested() \
                if state.can_operate \
                and state.solver in (Solver.READY, Solver.RESUMABLE, Solver.FAILED):
            # Reset frame/error so a poll that lands between this
            # transition and the server's start-command response can't
            # surface the prior run's tail (e.g. frame=9 from run N) as
            # if it belonged to run N+1. Without this, waiters that
            # treat ``state.frame > 0`` as "the run advanced" can short-
            # circuit before run N+1 has actually started — the
            # bl_chain_reconnect (applied=9, total=0) regression.
            return (
                replace(
                    state,
                    solver=Solver.STARTING,
                    progress=0.0,
                    frame=0,
                    error="",
                    # Drain in-flight stale polls before accepting a
                    # non-BUSY/SAVING response as a real finish. See the
                    # post-poll guard for the rationale.
                    starting_poll_guard=_STARTING_POLL_GUARD_DEPTH,
                ),
                [DoClearAnimation(), DoClearInterrupt(),
                 DoQuery({"request": "start"})],
            )

        case ResumeRequested(from_frame=ff) \
                if state.can_operate \
                and state.solver in (Solver.RESUMABLE, Solver.FAILED):
            # Accept Solver.FAILED as well as RESUMABLE: after a failed
            # simulation the server still holds the saved checkpoints, so
            # resuming from one is valid (mirrors RunRequested, which also
            # accepts FAILED). Without FAILED here the event falls through
            # and is silently dropped — the dialog's OK does nothing.
            # Same staleness reset as RunRequested. The server's first
            # response after resume will repopulate ``frame`` with the
            # saved frame index; until that lands, ``frame=0`` keeps
            # waiters from false-positive on the pre-save tail.
            resume_request = {"request": "resume"}
            if ff is not None:
                resume_request["resume_from"] = int(ff)
            return (
                replace(state, solver=Solver.STARTING, progress=0.0,
                        frame=0, error="",
                        starting_poll_guard=_STARTING_POLL_GUARD_DEPTH),
                [DoClearInterrupt(), DoQuery(resume_request)],
            )

        # ── Terminate / Save-and-quit ──────────────────────
        case TerminateRequested():
            return (
                replace(state, activity=Activity.ABORTING),
                [DoSetInterrupt(), DoClearAnimation(), DoTerminate()],
            )

        case SaveAndQuitRequested():
            return state, [DoSaveAndQuit()]

        # ── Abort ──────────────────────────────────────────
        case AbortRequested():
            return (
                replace(state, activity=Activity.ABORTING, pending_build=False),
                [DoSetInterrupt(), DoClearAnimation()],
            )

        # ── Data transfer ──────────────────────────────────
        case SendDataRequested(remote_path=p, data=d, message=m) \
                if not state.busy:
            return (
                replace(
                    state,
                    activity=Activity.SENDING,
                    progress=0.0,
                    traffic="",
                    message=m,
                ),
                [DoClearInterrupt(), DoSendData(p, d)],
            )

        case SendDataComplete():
            return (
                replace(
                    state,
                    activity=Activity.IDLE,
                    progress=0.0,
                    traffic="",
                    message="",
                ),
                [DoRedrawUI()],
            )

        # ── Atomic upload + build pipeline ─────────────────
        # Transfer button dispatches one event; the engine sequences
        # upload then build. Replaces the old modal operator that
        # watched ``com.busy()`` and chained three separate operations
        # (data_send → param_send → build).
        case BuildPipelineRequested(data=d, param=p, data_hash=dh,
                                    param_hash=ph, message=m,
                                    preserve_output=po) \
                if state.can_operate and state.remote_root:
            if not d and not p:
                return state, [DoLog("BuildPipelineRequested: no payload.")]
            return (
                replace(
                    state,
                    activity=Activity.SENDING,
                    pending_build=True,
                    pending_build_preserve_output=po,
                    progress=0.0,
                    traffic="",
                    message=m or "Uploading scene...",
                ),
                [
                    DoClearInterrupt(),
                    DoClearAnimation(),
                    DoUploadAtomic(
                        project_root=state.remote_root, data=d, param=p,
                        data_hash=dh, param_hash=ph,
                    ),
                ],
            )

        case UploadOnlyRequested(data=d, param=p, data_hash=dh,
                                 param_hash=ph, message=m) \
                if state.can_operate and state.remote_root:
            if not d and not p:
                return state, [DoLog("UploadOnlyRequested: no payload.")]
            return (
                replace(
                    state,
                    activity=Activity.SENDING,
                    pending_build=False,
                    progress=0.0,
                    traffic="",
                    message=m or "Uploading scene...",
                ),
                [
                    DoClearInterrupt(),
                    DoUploadAtomic(
                        project_root=state.remote_root, data=d, param=p,
                        data_hash=dh, param_hash=ph,
                    ),
                ],
            )

        case UploadPipelineComplete() if state.activity == Activity.SENDING:
            # Upload finished. If the upload was part of a pipeline
            # (BuildPipelineRequested), kick off the build now. Otherwise
            # the caller just wanted to ship files — return to IDLE.
            if state.pending_build:
                build_request = {"request": "build"}
                if state.pending_build_preserve_output:
                    build_request["preserve_output"] = 1
                return (
                    replace(
                        state,
                        activity=Activity.BUILDING,
                        solver=Solver.BUILDING,
                        active_upload_id="",
                        pending_build=False,
                        pending_build_preserve_output=False,
                        progress=0.0,
                        traffic="",
                        message="",
                    ),
                    [
                        DoQuery(build_request),
                        DoLog("Upload complete; starting build..."),
                    ],
                )
            return (
                replace(
                    state,
                    activity=Activity.IDLE,
                    pending_build=False,
                    pending_build_preserve_output=False,
                    progress=0.0,
                    traffic="",
                    message="",
                ),
                [DoRedrawUI()],
            )

        case ReceiveDataRequested(remote_path=p, message=m) if not state.busy:
            return (
                replace(
                    state,
                    activity=Activity.RECEIVING,
                    progress=0.0,
                    traffic="",
                    message=m,
                ),
                [DoClearInterrupt(), DoReceiveData(p)],
            )

        case ReceiveDataComplete():
            return (
                replace(
                    state,
                    activity=Activity.IDLE,
                    progress=0.0,
                    traffic="",
                    message="",
                ),
                [DoRedrawUI()],
            )

        # ── Fetch animation ────────────────────────────────
        case FetchRequested() if not state.busy:
            return (
                replace(
                    state,
                    activity=Activity.FETCHING,
                    progress=0.0,
                    traffic="",
                ),
                [
                    DoResetAnimationBuffer(),
                    DoClearInterrupt(),
                    DoFetchMap(state.remote_root),
                ],
            )

        case FetchMapComplete():
            # Map downloaded, now fetch the actual frames. ``state.frame``
            # mirrors the server's last-reported frame index — that's
            # sufficient; no need to reach into the raw response dict.
            return (
                state,
                [DoFetchFrames(
                    root=state.remote_root,
                    frame_count=int(state.frame),
                )],
            )

        case FetchComplete(total_frames=total):
            if total > 0:
                # Download phase filled 0→1; apply phase restarts its
                # own 0→1 bar. Status label disambiguates the phase.
                return (
                    replace(
                        state,
                        activity=Activity.APPLYING,
                        progress=0.0,
                        message="",
                    ),
                    [],
                )
            return (
                replace(state, activity=Activity.IDLE, progress=0.0, message=""),
                [DoRedrawUI()],
            )

        case AllFramesApplied() if state.activity == Activity.APPLYING:
            return (
                replace(
                    state,
                    activity=Activity.IDLE,
                    progress=0.0,
                    message="",
                ),
                [DoRedrawUI()],
            )

        case FetchFailed(reason=r) \
                if state.activity in (Activity.FETCHING, Activity.APPLYING):
            return (
                replace(
                    state,
                    activity=Activity.IDLE,
                    progress=0.0,
                    error=r,
                    message="",
                ),
                [DoClearAnimation(), DoLog(f"Fetch failed: {r}"), DoRedrawUI()],
            )

        # ── Exec ───────────────────────────────────────────
        case ExecRequested(command=c, shell=s) if not state.busy:
            return (
                replace(state, activity=Activity.EXECUTING),
                [DoExec(c, s)],
            )

        case ExecComplete():
            return (
                replace(state, activity=Activity.IDLE),
                [DoRedrawUI()],
            )

        # ── Query (user-initiated) ─────────────────────────
        case QueryRequested(request=req, message=m) if not state.busy:
            return (
                replace(state, activity=Activity.EXECUTING, message=m),
                [DoQuery(req)],
            )

        # ── Progress from background threads ───────────────
        case ProgressUpdated(progress=p, traffic=t):
            return replace(state, progress=p, traffic=t), []

        # ── Errors ─────────────────────────────────────────
        case ErrorOccurred(error=e, source=src):
            msg = f"[{src}] {e}" if src else e
            return (
                replace(
                    state,
                    error=e,
                    activity=Activity.IDLE,
                    progress=0.0,
                    pending_build=False,
                ),
                [DoLog(msg)],
            )

        # ── Guard-rejected or unknown events ───────────────
        case _:
            return state, []


# ---------------------------------------------------------------------------
# Server response interpretation
# ---------------------------------------------------------------------------

# Maps server status strings to Solver phases.  This replaces the 11-branch
# elif chain in the old ``_update_status()``.
_STATUS_MAP: dict[str, Solver] = {
    STATUS_NO_DATA: Solver.NO_DATA,
    STATUS_NO_BUILD: Solver.NO_BUILD,
    STATUS_BUILDING: Solver.BUILDING,
    STATUS_READY: Solver.READY,
    STATUS_RESUMABLE: Solver.RESUMABLE,
    STATUS_FAILED: Solver.FAILED,
}


def _remote_log_lines(error_msg: str, prefix: str = "Remote: ") -> list[Effect]:
    """Split a multi-line server error into one DoLog per line.

    The default ``Remote: `` prefix tags out-of-band server errors in the
    Console panel; pass ``prefix=""`` to log the raw lines (used for the
    terminal / build-failure path, where the bare lines are deliberate).
    """
    return [DoLog(f"{prefix}{line}") for line in error_msg.split("\n")]


def _interpret_response(
    state: AppState,
    r: dict,
) -> tuple[AppState, list[Effect]]:
    """Interpret a server query response and derive new state + effects.

    This is the pure equivalent of the old ``Communicator._update_status()``.
    """
    effects: list[Effect] = []

    # Empty response means we lost contact with the server. Treat the
    # same as ServerLost: clear solver and any in-flight activity so
    # the UI doesn't sit in ABORTING / BUILDING / SENDING forever after
    # the server stops responding mid-operation. The previous form set
    # only ``server=Server.UNKNOWN`` and left activity alone, which
    # surfaced as "Status: Aborting…" stuck indefinitely after a
    # mid-simulation server crash or kill.
    if not r:
        return (
            _server_gone(state),
            [DoLog("Empty server response.")],
        )

    # Protocol version check. Server MUST supply protocol_version on every
    # response; a missing field means we're talking to something other than
    # a protocol-compliant server and we bail loudly rather than guessing.
    version = r.get("protocol_version")
    if version is None:
        return (
            replace(state, version_ok=False),
            [DoLog("Server response missing protocol_version field.")],
        )
    if str(version) != PROTOCOL_VERSION:
        # Take the mismatched server down on detection. Reason: a stale
        # server orphaned from a previous binary version can keep its
        # in-memory image alive across the user's update step (they
        # download a new addon or rebuild the server on disk but the
        # already-running process keeps serving its old PROTOCOL_VERSION).
        # If we just log and stay connected, the next Transfer / Run
        # round-trip hits the same mismatch and the user keeps wondering
        # why the bump did nothing. Stopping the server forces a clean
        # restart against the on-disk binary, which is the version the
        # user actually intended to run. DoStopServer routes through the
        # current backend (local/win_native: subprocess.terminate;
        # ssh/docker: pkill -f ppf-cts-server on the remote).
        return (
            replace(state, version_ok=False),
            [
                DoLog(
                    f"Protocol version mismatch: server reports {version}, "
                    f"addon expects {PROTOCOL_VERSION}. Stopping the server "
                    f"so a fresh restart picks up the on-disk binary; "
                    f"update the older side if the mismatch persists."
                ),
                DoStopServer(),
            ],
        )

    # upload_id is mandatory in the current wire protocol; absence
    # indicates a malformed response, treat as version mismatch
    # rather than silently coping.
    if "upload_id" not in r:
        return (
            replace(state, version_ok=False),
            [DoLog("Server response missing upload_id field.")],
        )
    server_upload_id = r["upload_id"]
    # data_hash / param_hash are best-effort: a server that predates
    # the fields (or an error-only response) returns "" and the UI
    # treats every client edit as divergent, which is the safe default.
    server_data_hash = str(r.get("data_hash", "") or "")
    server_param_hash = str(r.get("param_hash", "") or "")
    # Emulated-build flag rides on ``hardware.emulated`` in normal
    # responses. The minimal error-only reply omits the hardware block,
    # so fall back to the last-known value rather than resetting it.
    _hw = r.get("hardware")
    server_emulated = (
        bool(_hw.get("emulated", False)) if isinstance(_hw, dict) else state.emulated
    )

    status_str = r.get("status", "")
    error_msg = r.get("error", "")
    violations = r.get("violations", [])
    info_msg = r.get("info", "")
    root = r.get("root", state.remote_root)
    frame = int(r.get("frame", 0))
    # ``initialized`` reflects whether the running solver has finished
    # its in-process initialize() call (the server flips it on observing
    # ``initialize_finish.txt``). When true we promote the local solver
    # state to RUNNING immediately rather than waiting for the first
    # advance to complete, so the status banner does not sit at
    # "Initializing" for the entire first frame under heavy contact load.
    # Defaults to False so a server that does not report the field falls
    # back to the frame-based check below.
    initialized = bool(r.get("initialized", False))
    # Only accept progress from the response when the solver is active;
    # stale/zero values from idle responses would flicker the progress bar.
    raw_progress = r.get("progress")
    progress = float(raw_progress) if raw_progress and raw_progress > 0 else None

    # Error-only response: the server's error-only response path
    # sends ``{error, protocol_version, upload_id: "", status: ""}``.
    # This is NOT a status update; the server hit an unexpected failure
    # (e.g. select_project raised on a malformed project dir) and is
    # reporting it out-of-band. Surface the error and preserve current
    # solver/activity so the client doesn't misinterpret an empty status
    # as a phase transition. Handled explicitly here so a future change
    # to the status-mapping else-branch can't silently break the invariant.
    if not status_str and error_msg:
        log_effects = _remote_log_lines(error_msg)
        return (
            replace(
                state,
                server=Server.RUNNING,
                server_error=error_msg,
                version_ok=True,
                server_upload_id=server_upload_id,
                server_data_hash=server_data_hash,
                server_param_hash=server_param_hash,
                emulated=server_emulated,
            ),
            log_effects,
        )

    # Determine solver phase from server status string
    if status_str == STATUS_BUSY:
        solver = (
            Solver.RUNNING
            if initialized or frame >= 1
            else Solver.STARTING
        )
    elif status_str == STATUS_SAVE_AND_QUIT:
        solver = Solver.SAVING
    elif status_str in _STATUS_MAP:
        solver = _STATUS_MAP[status_str]
    elif status_str:
        # Unknown non-empty status: keep current solver and log loudly so
        # a protocol mismatch is visible rather than silently ignored.
        solver = state.solver
        effects.append(DoLog(f"Unknown server status: {status_str}"))
    else:
        # status_str == "" reaches here only when error_msg is also empty
        # (the error-only path above handled the error+empty-status case).
        # That combination shouldn't happen against a conforming server;
        # treat it as a malformed response.
        return (
            replace(state, version_ok=False),
            [DoLog(
                "Server response has empty status and no error: malformed "
                "protocol reply."
            )],
        )

    # Stale-poll guard during STARTING. After RunRequested or
    # ResumeRequested transitions us into Solver.STARTING, a status
    # poll that was already in flight on the I/O worker may return
    # the server's pre-start state (e.g. ``READY`` with the prior
    # run's frame). If we accept that response, the engine state is
    # dragged back to READY+frame=N before the start command's own
    # response arrives, and any waiter that treats ``frame > 0``
    # plus ``solver in (READY, RESUMABLE)`` as "run done" fires too
    # early — the bl_chain_reconnect (applied=9, total=0) regression.
    #
    # We discard such polls only while ``starting_poll_guard > 0``
    # (decremented per poll, seeded by RunRequested /
    # ResumeRequested). After the guard drains, accept the response
    # so a fast-finish run — solver advances BUSY → READY between two
    # addon polls — surfaces as READY/RESUMABLE instead of getting
    # locked at STARTING+frame=0 forever (the bl_pin_* regressions).
    #
    # Legitimate non-stale exits out of STARTING:
    #   - ``status="BUSY"`` (server is working) → solver advances.
    #   - SAVE_AND_QUIT request mid-start → solver=SAVING.
    #   - explicit ``error`` (start rejected) → honor the downgrade.
    # The guard fires only when the response is none of those AND we
    # still have stale polls to drain.
    if state.solver == Solver.STARTING \
            and solver not in (Solver.STARTING, Solver.RUNNING, Solver.SAVING) \
            and not error_msg \
            and state.starting_poll_guard > 0:
        solver = Solver.STARTING
        frame = 0
        # Decrement so the next stale poll inches the counter down to
        # zero; once drained, the next non-BUSY/SAVING poll exits
        # STARTING normally.
        starting_poll_guard_after = state.starting_poll_guard - 1
    else:
        # Any other path out of STARTING (or any transition while not
        # in STARTING) clears the counter so a future RunRequested
        # starts with a fresh drain budget.
        starting_poll_guard_after = (
            0 if state.solver != Solver.STARTING or solver != Solver.STARTING
            else state.starting_poll_guard
        )

    # Log error and mark as failed when server reports an error.
    #
    # Two paths converge on Solver.FAILED:
    #
    #   1. status was already a "would-be terminal" (READY / RESUMABLE
    #      / FAILED) and the server attached an error -- promote to
    #      FAILED.
    #   2. status was NO_BUILD with a non-empty error while the addon
    #      was BUILDING. The server's status_string maps both
    #      ``Build.NONE`` (never attempted) and ``Build.FAILED``
    #      (attempted, decoding failed) to "NO_BUILD" because in
    #      either case there is no usable scene artifact; the error
    #      field is what distinguishes "build failed" from the inflight
    #      race-window ping (status="NO_BUILD" before DoQuery("build")
    #      runs). Promote to FAILED so the BUILDING-terminal-set in
    #      ``ServerPolled`` clears activity and the user can fix the
    #      scene and re-upload.
    is_terminal_with_error = solver in (
        Solver.READY, Solver.RESUMABLE, Solver.FAILED,
    )
    is_build_failure = (
        solver == Solver.NO_BUILD and state.activity == Activity.BUILDING
    )
    if error_msg and (is_terminal_with_error or is_build_failure):
        solver = Solver.FAILED
        # Raw lines (no prefix) for the terminal / build-failure path; the
        # bare-line output here is deliberate, unlike the "Remote: "-tagged
        # out-of-band errors elsewhere.
        effects.extend(_remote_log_lines(error_msg, prefix=""))
    # Log newly-raised server errors even in non-terminal solver states
    # (e.g. NO_BUILD without an active build, ping race window) so they
    # surface in the Console panel, not only on the top-of-panel ERROR
    # label.
    elif error_msg and error_msg != state.server_error:
        effects.extend(_remote_log_lines(error_msg))

    # --- Solver-driven polling and side-effects ---
    # Poll and fetch decisions are based purely on solver state,
    # never on activity (which tracks client-side operations).

    is_sim_running = solver in (Solver.RUNNING, Solver.STARTING, Solver.SAVING)
    is_building = solver == Solver.BUILDING
    # The very first ServerPolled response that shows the run has finished
    # (RUNNING -> READY/RESUMABLE) is also the response that carries the
    # final frame count. Without an auto-fetch on this transition the
    # trailing frame's vertex data never makes it to the client: the
    # previous in-RUNNING fetch ran with the prior frame count, and after
    # this poll is_sim_running is False so no further auto-fetch fires.
    # Surfaces in bl_live_frame_end_tracking as final_frame_end == 10
    # when FRAME_COUNT is 12.
    just_finished_sim = (
        state.solver in (Solver.RUNNING, Solver.STARTING, Solver.SAVING)
        and solver in (Solver.READY, Solver.RESUMABLE)
    )

    # Build-target desync detection.  When Activity.BUILDING is in flight,
    # we pin state.active_upload_id to the server's reported upload_id on
    # the first poll after BuildRequested.  Any subsequent poll where the
    # server's upload_id differs means the server state was reset under
    # us: restart, external project wipe, or another client uploaded new
    # data.  We then clear BUILDING with an explicit error rather than
    # pinning the client to BUILDING forever (the bug that motivated this
    # design).  No anti-regression guard any more — the upload_id
    # comparison replaces it.
    new_active_upload_id = state.active_upload_id
    desync = False
    if state.activity == Activity.BUILDING:
        if state.active_upload_id == "":
            # First poll after BuildRequested -- pin whatever the server
            # currently reports.  An empty upload_id here means the
            # server has no data at all; that's a desync too.
            if server_upload_id == "":
                desync = True
            else:
                new_active_upload_id = server_upload_id
        elif server_upload_id != state.active_upload_id:
            desync = True

    if desync:
        effects.append(DoLog(
            f"Build desync: expected upload_id '{state.active_upload_id}', "
            f"server reports '{server_upload_id}'. Server state was reset; "
            f"re-upload the scene and rebuild."
        ))

    # Auto-fetch latest frame during simulation, plus one more on the
    # transition out so the final frame doesn't get stranded server-side.
    if is_sim_running or just_finished_sim:
        effects.append(DoFetchFrames(
            root=root,
            frame_count=frame,
            only_latest=True,
        ))

    # Polling is handled by the persistent Blender timer in facade.py
    # which dispatches PollTick every 0.25s. The PollTick transition
    # decides whether to emit DoQuery based on solver state.

    # Log build progress to console
    if is_building and info_msg:
        progress_pct = progress if progress is not None else 0.0
        effects.append(DoLog(f"[BUILD] {progress_pct:.0%} {info_msg}"))

    # Log build completion
    if state.solver == Solver.BUILDING and solver in (
        Solver.READY, Solver.RESUMABLE, Solver.FAILED,
    ):
        effects.append(DoLog("Build done."))

    # On desync, snap activity back to IDLE, clear pinned upload_id, and
    # surface a user-visible error.  Leave solver as whatever the server
    # now reports (usually NO_DATA) so the UI reflects the server's
    # actual state instead of an illusion.
    if desync:
        new_state = replace(
            state,
            solver=solver,
            server=Server.RUNNING,
            remote_root=root,
            activity=Activity.IDLE,
            active_upload_id="",
            server_upload_id=server_upload_id,
            server_data_hash=server_data_hash,
            server_param_hash=server_param_hash,
            error="Server state lost during build; re-upload and rebuild.",
            server_error=error_msg,
            violations=violations,
            message="",
            progress=0.0,
            frame=frame,
            version_ok=True,
            starting_poll_guard=starting_poll_guard_after,
            emulated=server_emulated,
        )
        return new_state, effects

    # Build new state — clear client error on successful server response
    new_state = replace(
        state,
        solver=solver,
        server=Server.RUNNING,
        remote_root=root,
        error="",
        server_error=error_msg,
        violations=violations if violations else ([] if not error_msg else state.violations),
        message=info_msg if info_msg else (
            state.message if (is_sim_running or is_building) else ""
        ),
        progress=progress if progress is not None else (
            state.progress if (is_sim_running or is_building) else 0.0
        ),
        frame=frame,
        version_ok=True,
        server_upload_id=server_upload_id,
        server_data_hash=server_data_hash,
        server_param_hash=server_param_hash,
        active_upload_id=new_active_upload_id,
        starting_poll_guard=starting_poll_guard_after,
        emulated=server_emulated,
    )

    return new_state, effects


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_state(state: AppState, error: str = "") -> AppState:
    """Return a clean offline state, preserving nothing."""
    return AppState(error=error)


def _server_gone(state: AppState) -> AppState:
    """Scrub the per-operation status fields after the server stops
    responding, preserving connection-level fields (remote_root,
    session_id, backend identity, hashes) for recovery when it returns.

    This is the shared field reset for the "server gone" paths
    (ServerLost, ServerStopped, empty-response). Do NOT route through
    _reset_state(): that returns a fresh AppState and would drop the
    fields needed for reconnection.
    """
    return replace(
        state,
        server=Server.UNKNOWN,
        solver=Solver.NO_DATA,
        activity=Activity.IDLE,
        progress=0.0,
        traffic="",
        message="",
        active_upload_id="",
    )
