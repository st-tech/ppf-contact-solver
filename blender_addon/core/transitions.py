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

from __future__ import annotations

from dataclasses import replace

from .protocol import PROTOCOL_VERSION
from .session import new_session_id
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
    FrameFetched,
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

        case Connected(remote_root=root):
            # Mint a fresh session id on every successful connect so
            # downstream artifacts (PC2 headers, modifier binds, remote
            # directories) can be correlated with THIS run.
            sid = new_session_id()
            effects = [DoValidateRemotePath(), DoQuery(), DoLog(f"Connected (session {sid}).")]
            # Reconcile log: warn the user when the .blend was last
            # saved under a different session so they know the on-disk
            # PC2 files may not correspond to anything on this server.
            try:
                import bpy  # pyright: ignore
                from ..models.groups import get_addon_data, has_addon_data
                scene = bpy.context.scene
                if scene is not None and has_addon_data(scene):
                    saved = get_addon_data(scene).state.last_session_id or ""
                    if saved and saved != sid:
                        effects.append(DoLog(
                            f"Reconcile: previous session {saved} differs "
                            f"from new session {sid}; cached PC2 files may "
                            f"not match the remote."
                        ))
            except Exception:
                pass
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
            return (
                replace(
                    state,
                    server=Server.UNKNOWN,
                    solver=Solver.NO_DATA,
                    activity=Activity.IDLE,
                    progress=0.0,
                    message="",
                    server_error="",
                ),
                [DoClearInterrupt(), DoLog("Server stopped.")],
            )

        case ServerLost():
            return (
                replace(
                    state,
                    server=Server.UNKNOWN,
                    solver=Solver.NO_DATA,
                    activity=Activity.IDLE,
                    progress=0.0,
                    message="",
                ),
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
                    activity=Activity.IDLE,
                    frame=0,
                    error="",
                ),
                [DoClearAnimation(), DoClearInterrupt(),
                 DoQuery({"request": "start"})],
            )

        case ResumeRequested() \
                if state.can_operate and state.solver == Solver.RESUMABLE:
            # Same staleness reset as RunRequested. The server's first
            # response after resume will repopulate ``frame`` with the
            # saved frame index; until that lands, ``frame=0`` keeps
            # waiters from false-positive on the pre-save tail.
            return (
                replace(state, solver=Solver.STARTING, progress=0.0,
                        frame=0, error=""),
                [DoClearInterrupt(), DoQuery({"request": "resume"})],
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
                                    param_hash=ph, message=m) \
                if state.can_operate and state.remote_root:
            if not d and not p:
                return state, [DoLog("BuildPipelineRequested: no payload.")]
            return (
                replace(
                    state,
                    activity=Activity.SENDING,
                    pending_build=True,
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
                return (
                    replace(
                        state,
                        activity=Activity.BUILDING,
                        solver=Solver.BUILDING,
                        active_upload_id="",
                        pending_build=False,
                        progress=0.0,
                        traffic="",
                        message="",
                    ),
                    [
                        DoQuery({"request": "build"}),
                        DoLog("Upload complete; starting build..."),
                    ],
                )
            return (
                replace(
                    state,
                    activity=Activity.IDLE,
                    pending_build=False,
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
    "NO_DATA": Solver.NO_DATA,
    "NO_BUILD": Solver.NO_BUILD,
    "BUILDING": Solver.BUILDING,
    "READY": Solver.READY,
    "RESUMABLE": Solver.RESUMABLE,
    "FAILED": Solver.FAILED,
}


def _interpret_response(
    state: AppState,
    r: dict,
) -> tuple[AppState, list[Effect]]:
    """Interpret a server query response and derive new state + effects.

    This is the pure equivalent of the old ``Communicator._update_status()``.
    """
    effects: list[Effect] = []

    # Empty response means server is not running
    if not r:
        return (
            replace(state, server=Server.UNKNOWN),
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
        return (
            replace(state, version_ok=False),
            [DoLog(f"Protocol version mismatch: {version} != {PROTOCOL_VERSION}")],
        )

    # upload_id is part of protocol 0.03; absence indicates a malformed
    # response, treat as version mismatch rather than silently coping.
    if "upload_id" not in r:
        return (
            replace(state, version_ok=False),
            [DoLog("Server response missing upload_id field (protocol 0.03).")],
        )
    server_upload_id = r["upload_id"]
    # data_hash / param_hash are best-effort: a server that predates
    # the fields (or an error-only response) returns "" and the UI
    # treats every client edit as divergent, which is the safe default.
    server_data_hash = str(r.get("data_hash", "") or "")
    server_param_hash = str(r.get("param_hash", "") or "")

    status_str = r.get("status", "")
    error_msg = r.get("error", "")
    violations = r.get("violations", [])
    info_msg = r.get("info", "")
    root = r.get("root", state.remote_root)
    frame = int(r.get("frame", 0))
    # Only accept progress from the response when the solver is active;
    # stale/zero values from idle responses would flicker the progress bar.
    raw_progress = r.get("progress")
    progress = float(raw_progress) if raw_progress and raw_progress > 0 else None

    # Error-only response: server.py's handle_text_command exception path
    # sends ``{error, protocol_version, upload_id: "", status: ""}``.
    # This is NOT a status update — the server hit an unexpected failure
    # (e.g. select_project raised on a malformed project dir) and is
    # reporting it out-of-band. Surface the error and preserve current
    # solver/activity so the client doesn't misinterpret an empty status
    # as a phase transition. Handled explicitly here so a future change
    # to the status-mapping else-branch can't silently break the invariant.
    if not status_str and error_msg:
        log_effects = [DoLog(f"Remote: {line}") for line in error_msg.split("\n")]
        return (
            replace(
                state,
                server=Server.RUNNING,
                server_error=error_msg,
                version_ok=True,
                server_upload_id=server_upload_id,
                server_data_hash=server_data_hash,
                server_param_hash=server_param_hash,
            ),
            log_effects,
        )

    # Determine solver phase from server status string
    if status_str == "BUSY":
        solver = Solver.RUNNING if frame >= 1 else Solver.STARTING
    elif status_str == "SAVE_AND_QUIT":
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
        # That combination shouldn't happen against a conforming 0.03
        # server; treat it as a malformed response.
        return (
            replace(state, version_ok=False),
            [DoLog(
                "Server response has empty status and no error — malformed "
                "protocol 0.03 reply."
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
    # We discard such polls. The legitimate ways out of STARTING are:
    #   - ``status="BUSY"`` (server is working) → solver advances.
    #   - SAVE_AND_QUIT request mid-start → solver=SAVING.
    #   - explicit ``error`` (start rejected) → honor the downgrade.
    # Anything else while STARTING is the stale-poll race; preserve
    # STARTING and frame=0 until the next poll/response advances us.
    if state.solver == Solver.STARTING \
            and solver not in (Solver.STARTING, Solver.RUNNING, Solver.SAVING) \
            and not error_msg:
        solver = Solver.STARTING
        frame = 0

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
        for line in error_msg.split("\n"):
            effects.append(DoLog(line))
    # Log newly-raised server errors even in non-terminal solver states
    # (e.g. NO_BUILD without an active build, ping race window) so they
    # surface in the Console panel, not only on the top-of-panel ERROR
    # label.
    elif error_msg and error_msg != state.server_error:
        for line in error_msg.split("\n"):
            effects.append(DoLog(f"Remote: {line}"))

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
    )

    return new_state, effects


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_state(state: AppState, error: str = "") -> AppState:
    """Return a clean offline state, preserving nothing."""
    return AppState(error=error)
