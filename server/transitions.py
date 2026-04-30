# File: server/transitions.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Pure state-transition function for the server state machine.
#
# ``server_transition(state, event) -> (new_state, effects)``
#
# This module is the SINGLE SOURCE OF TRUTH for every server state change.
# It contains ZERO side-effects: no I/O, no threads, no subprocess calls.
# Fully testable with plain assert statements.

from __future__ import annotations

from dataclasses import replace

from .state import Build, Data, ServerState, Solver
from .events import (
    BuildCancelledEvent,
    BuildCompleted,
    BuildFailed,
    BuildProgress,
    BuildRequested,
    CancelBuildRequested,
    DeleteRequested,
    ErrorOccurred,
    Event,
    GPUCheckFailed,
    ProjectSelected,
    ResumeRequested,
    SaveAndQuitRequested,
    SolverCrashed,
    SolverFinished,
    SolverFrameUpdated,
    SolverSaving,
    StartRequested,
    TerminateRequested,
    UploadLanded,
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


def server_transition(
    state: ServerState, event: Event
) -> tuple[ServerState, list[Effect]]:
    """Pure function: (ServerState, Event) -> (new ServerState, [Effect ...]).

    The ONLY place where ``ServerState`` is replaced.
    """

    match event:

        # ── Project context ────────────────────────────────
        case ProjectSelected(
            name=n, root=r, has_data=hd, has_param=hp,
            has_app=ha, is_resumable=ir, upload_id=uid,
            data_hash=dh, param_hash=ph,
        ):
            data = Data.UPLOADED if (hd and hp) else Data.EMPTY

            # Same project -- just refresh data availability and upload_id.
            # (Caller passes the disk-resident id or our current state.upload_id.)
            if n == state.name:
                new = replace(state, root=r, data=data,
                              upload_id=uid, data_hash=dh, param_hash=ph)
                # Update resumable from live check
                if state.build == Build.BUILT:
                    new = replace(new, resumable=ir)
                return new, []

            # Different project -- reset state, optionally load app
            build = Build.BUILT if ha else Build.NONE
            effects: list[Effect] = []
            if not ha:
                # Try to load app from disk
                effects.append(DoLoadApp(n, r))
            return (
                replace(
                    state,
                    name=n,
                    root=r,
                    data=data,
                    build=build,
                    solver=Solver.IDLE,
                    resumable=ir,
                    frame=0,
                    error="",
                    build_progress=0.0,
                    build_info="",
                    upload_id=uid,
                    data_hash=dh,
                    param_hash=ph,
                ),
                effects,
            )

        # ── Upload landing ─────────────────────────────────
        case UploadLanded(upload_id=uid, data_hash=dh, param_hash=ph,
                          has_data=hd, has_param=hp):
            # Both files present: full upload, mark data UPLOADED and stamp
            # upload_id. Partial upload (only data.pickle or only param.pickle)
            # still stamps the id but leaves data EMPTY until both exist.
            data = Data.UPLOADED if (hd and hp) else Data.EMPTY
            # Replace each hash only when the corresponding payload was
            # actually uploaded; the other side's hash is preserved so
            # the client can still detect drift against whatever is
            # currently on disk.
            new_data_hash = dh if hd else state.data_hash
            new_param_hash = ph if hp else state.param_hash
            # A fresh upload invalidates any prior build: the app artifact
            # was for an earlier (data, param) tuple. Reset to NONE so the
            # next build request is accepted; any in-flight build will be
            # rejected via upload_id mismatch on the client side.
            return (
                replace(
                    state,
                    data=data,
                    upload_id=uid,
                    data_hash=new_data_hash,
                    param_hash=new_param_hash,
                    build=Build.NONE if state.build != Build.BUILDING else state.build,
                    resumable=False,
                ),
                [DoLog(f"Upload landed (id={uid}, data={hd}, param={hp}).")],
            )

        # ── Build ──────────────────────────────────────────
        case BuildRequested() if (
            state.data == Data.UPLOADED
            and state.build != Build.BUILDING
        ):
            return (
                replace(
                    state,
                    build=Build.BUILDING,
                    build_progress=0.0,
                    build_info="Preparing build...",
                    error="",
                ),
                [DoSpawnBuild()],
            )

        case CancelBuildRequested() if state.build == Build.BUILDING:
            return state, [DoCancelBuild(), DoLog("Build cancel requested.")]

        case BuildProgress(progress=p, info=i):
            return replace(state, build_progress=p, build_info=i), []

        case BuildCompleted():
            return (
                replace(
                    state,
                    build=Build.BUILT,
                    build_progress=1.0,
                    build_info="Build complete.",
                    resumable=False,
                    violations=[],
                ),
                [DoLog("Build complete.")],
            )

        case BuildFailed(error=e, violations=v):
            return (
                replace(
                    state,
                    build=Build.FAILED,
                    error=e,
                    violations=v,
                    build_info="Build failed.",
                ),
                [DoLog(f"Build failed: {e}")],
            )

        case BuildCancelledEvent():
            return (
                replace(
                    state,
                    build=Build.NONE,
                    build_progress=0.0,
                    build_info="",
                    error="",
                ),
                [DoLog("Build cancelled by user.")],
            )

        case GPUCheckFailed(error=e):
            return (
                replace(
                    state,
                    build=Build.FAILED,
                    error=e,
                ),
                [DoLog(f"GPU check failed: {e}")],
            )

        # ── Solver operations ──────────────────────────────
        case StartRequested() if (
            state.build == Build.BUILT
            and state.solver in (Solver.IDLE, Solver.FAILED)
        ):
            return (
                replace(state, solver=Solver.RUNNING, frame=0, error=""),
                [DoLaunchSolver(), DoLog("Solver starting.")],
            )

        case ResumeRequested() if (
            state.build == Build.BUILT
            and state.solver == Solver.IDLE
            and state.resumable
        ):
            return (
                replace(state, solver=Solver.RUNNING, error=""),
                [
                    DoLaunchSolver(resume_from=-1),
                    DoLog("Solver resuming."),
                ],
            )

        case TerminateRequested():
            # Always emit DoKillSolver so we can also terminate a solver
            # process started outside this server (e.g. by JupyterLab),
            # where state.solver was never promoted to RUNNING.
            # _kill_solver checks Utils.busy() internally, so this is a
            # no-op when nothing is running.
            return (
                replace(state, solver=Solver.IDLE),
                [DoKillSolver(), DoLog("Solver terminated.")],
            )

        case SaveAndQuitRequested():
            # Always emit DoRequestSaveAndQuit so we can also save a
            # solver started outside this server (e.g. by JupyterLab)
            # where state.solver was never promoted to RUNNING.
            # _request_save_and_quit is a no-op when the server has no
            # loaded app; the caller will see the ErrorOccurred event
            # it dispatches in that case.
            return (
                replace(state, solver=Solver.SAVING),
                [DoRequestSaveAndQuit(), DoLog("Save and quit requested.")],
            )

        # ── Solver monitor events ──────────────────────────
        case SolverFrameUpdated(frame=f):
            return replace(state, frame=f), []

        case SolverFinished(resumable=r):
            return (
                replace(state, solver=Solver.IDLE, resumable=r),
                [DoLog(f"Solver finished. Resumable={r}")],
            )

        case SolverCrashed(error=e, violations=v):
            return (
                replace(state, solver=Solver.FAILED, error=e, violations=v),
                [DoLog(f"Solver crashed: {e}")],
            )

        case SolverSaving():
            return replace(state, solver=Solver.SAVING), []

        # ── Delete ─────────────────────────────────────────
        case DeleteRequested() if state.root:
            # Clear upload_id: fresh project, no in-flight build from any
            # client can match after this.
            return (
                ServerState(name=state.name),
                [DoKillSolver(), DoDeleteProjectData(state.root),
                 DoLog("Project data deleted.")],
            )

        # ── Error ──────────────────────────────────────────
        case ErrorOccurred(error=e):
            return replace(state, error=e), [DoLog(e)]

        # ── Guard-rejected or unknown ──────────────────────
        case _:
            return state, []
