// File: crates/ppf-cts-core/src/transitions/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Pure (state, event) -> (new_state, [effects]). Single source of
// truth for every server state change. ZERO side-effects: no I/O, no
// spawn, no log (the `DoLog` effect is *recorded*, not executed).
//
// This is the SERVER-side state machine, operating on `ServerState`
// (Build / Data / Solver as tracked by the solver host). It is
// intentionally distinct from the ADDON-side state machine in
// `blender_addon/core/transitions.py`, which operates on `AppState`
// (Phase / Activity / Server / Solver as observed from the Blender
// side). The two run on different processes, observe different
// events, and emit different effect sets; they are NOT duplicates
// and must not be merged. They share zero types. Any cross-side
// invariant (e.g. handshake order) has to be reviewed in BOTH files
// when changed.

use crate::effects::Effect;
use crate::events::Event;
use crate::state::{Build, Data, ServerState, Solver};

/// Single entry point. Replaces `server.transitions.server_transition`.
///
/// Returns `(new_state, effects)`. The caller (engine) is responsible
/// for replacing its held state and dispatching the effects.
pub fn transition(state: ServerState, event: Event) -> (ServerState, Vec<Effect>) {
    match event {
        // ── Project context ─────────────────────────────────────────
        Event::ProjectSelected {
            name,
            root,
            has_data,
            has_param,
            has_app,
            is_resumable,
            upload_id,
            data_hash,
            param_hash,
            total_frames,
        } => {
            let data = if has_data && has_param {
                Data::Uploaded
            } else {
                Data::Empty
            };

            // Same project: refresh data availability and upload_id;
            // preserve solver/build/frame state. Rehydrate
            // total_frames from disk only when we currently lack it
            // (e.g. the connect probe just blew it away by routing
            // through __probe__ then back). Don't clobber a fresh
            // value from an in-flight BuildMetadata with a stale
            // scene_info.json read.
            if name == state.name {
                let preserved_total = if state.total_frames > 0 {
                    state.total_frames
                } else {
                    total_frames
                };
                let mut new = ServerState {
                    root,
                    data,
                    upload_id,
                    data_hash,
                    param_hash,
                    total_frames: preserved_total,
                    ..state
                };
                if new.build == Build::Built {
                    new.resumable = is_resumable;
                }
                return (new, vec![]);
            }

            // Different project: reset state, optionally load app.
            // Carry the disk-derived total_frames straight through so
            // a re-select of a previously-built project (e.g. probe
            // -> real reconnect) immediately knows how many frames
            // the build set up, without waiting for another build.
            let build = if has_app {
                Build::Built
            } else {
                Build::None
            };
            let mut effects: Vec<Effect> = Vec::new();
            if !has_app {
                effects.push(Effect::DoLoadApp {
                    name: name.clone(),
                    root: root.clone(),
                });
            }
            (
                ServerState {
                    name,
                    root,
                    data,
                    build,
                    solver: Solver::Idle,
                    resumable: is_resumable,
                    frame: 0,
                    initialized: false,
                    error: String::new(),
                    build_progress: 0.0,
                    build_info: String::new(),
                    total_frames,
                    upload_id,
                    data_hash,
                    param_hash,
                    violations: state.violations,
                },
                effects,
            )
        }

        // ── Upload landing ──────────────────────────────────────────
        Event::UploadLanded {
            upload_id,
            data_hash,
            param_hash,
            has_data,
            has_param,
        } => {
            let data = if has_data && has_param {
                Data::Uploaded
            } else {
                Data::Empty
            };
            // Replace each hash only when *this upload* carried it;
            // a param-only update sends ``data_hash=""`` while the
            // pre-existing data file (and its hash on disk) still
            // exist, so we can't key off ``has_data`` (which is just
            // a file-exists probe). The empty-hash sniff matches
            // what the addon actually emits and preserves the prior
            // hash so the next Run-button click correctly detects
            // data drift instead of seeing server_data="".
            let new_data_hash = if !data_hash.is_empty() {
                data_hash
            } else {
                state.data_hash.clone()
            };
            let new_param_hash = if !param_hash.is_empty() {
                param_hash
            } else {
                state.param_hash.clone()
            };
            // A fresh upload invalidates any prior build: the app
            // artifact was for an earlier (data, param) tuple. Reset
            // to NONE so the next BuildRequested is accepted; an
            // in-flight build (BUILDING) is preserved; the client
            // detects upload_id mismatch and aborts on its side.
            let new_build = if state.build != Build::Building {
                Build::None
            } else {
                state.build
            };
            let log = format!(
                "Upload landed (id={upload_id}, data={has_data}, param={has_param})."
            );
            (
                ServerState {
                    data,
                    upload_id,
                    data_hash: new_data_hash,
                    param_hash: new_param_hash,
                    build: new_build,
                    resumable: false,
                    ..state
                },
                vec![Effect::DoLog { message: log }],
            )
        }

        // ── Build ───────────────────────────────────────────────────
        Event::BuildRequested
            if state.data == Data::Uploaded && state.build != Build::Building =>
        {
            (
                ServerState {
                    build: Build::Building,
                    build_progress: 0.0,
                    build_info: "Preparing build...".to_string(),
                    total_frames: 0,
                    error: String::new(),
                    ..state
                },
                vec![Effect::DoSpawnBuild],
            )
        }

        Event::CancelBuildRequested if state.build == Build::Building => (
            state,
            vec![
                Effect::DoCancelBuild,
                Effect::DoLog {
                    message: "Build cancel requested.".to_string(),
                },
            ],
        ),

        Event::BuildProgress { progress, info } => (
            ServerState {
                build_progress: progress,
                build_info: info,
                ..state
            },
            vec![],
        ),

        Event::BuildMetadata { total_frames } => (
            ServerState {
                total_frames,
                ..state
            },
            vec![],
        ),

        Event::BuildCompleted => (
            ServerState {
                build: Build::Built,
                build_progress: 1.0,
                build_info: "Build complete.".to_string(),
                resumable: false,
                violations: vec![],
                ..state
            },
            vec![Effect::DoLog {
                message: "Build complete.".to_string(),
            }],
        ),

        Event::BuildFailed { error, violations } => {
            let log = format!("Build failed: {error}");
            (
                ServerState {
                    build: Build::Failed,
                    error,
                    violations,
                    build_info: "Build failed.".to_string(),
                    ..state
                },
                vec![Effect::DoLog { message: log }],
            )
        }

        Event::BuildCancelledEvent => (
            ServerState {
                build: Build::None,
                build_progress: 0.0,
                build_info: String::new(),
                error: String::new(),
                ..state
            },
            vec![Effect::DoLog {
                message: "Build cancelled by user.".to_string(),
            }],
        ),

        Event::GpuCheckFailed { error } => {
            let log = format!("GPU check failed: {error}");
            (
                ServerState {
                    build: Build::Failed,
                    error,
                    ..state
                },
                vec![Effect::DoLog { message: log }],
            )
        }

        // ── Solver operations ──────────────────────────────────────
        Event::StartRequested
            if state.build == Build::Built
                && matches!(state.solver, Solver::Idle | Solver::Failed) =>
        {
            (
                ServerState {
                    solver: Solver::Running,
                    frame: 0,
                    initialized: false,
                    error: String::new(),
                    ..state
                },
                vec![
                    Effect::DoLaunchSolver { resume_from: None },
                    Effect::DoLog {
                        message: "Solver starting.".to_string(),
                    },
                ],
            )
        }

        Event::ResumeRequested
            if state.build == Build::Built
                && state.solver == Solver::Idle
                && state.resumable =>
        {
            (
                ServerState {
                    solver: Solver::Running,
                    initialized: false,
                    error: String::new(),
                    ..state
                },
                vec![
                    Effect::DoLaunchSolver {
                        resume_from: Some(-1),
                    },
                    Effect::DoLog {
                        message: "Solver resuming.".to_string(),
                    },
                ],
            )
        }

        // Always emits DoKillSolver: handles externally-started
        // solvers (e.g. from JupyterLab) where state.solver was never
        // promoted to Running. The executor's Utils.busy() check makes
        // it a no-op when nothing is running.
        Event::TerminateRequested => (
            ServerState {
                solver: Solver::Idle,
                initialized: false,
                ..state
            },
            vec![
                Effect::DoKillSolver,
                Effect::DoLog {
                    message: "Solver terminated.".to_string(),
                },
            ],
        ),

        // Same rationale as TerminateRequested: must work against
        // externally-started solvers.
        Event::SaveAndQuitRequested => (
            ServerState {
                solver: Solver::Saving,
                ..state
            },
            vec![
                Effect::DoRequestSaveAndQuit,
                Effect::DoLog {
                    message: "Save and quit requested.".to_string(),
                },
            ],
        ),

        // ── Solver monitor events ──────────────────────────────────
        Event::SolverFrameUpdated { frame } => (
            ServerState { frame, ..state },
            vec![],
        ),

        Event::SolverInitialized => (
            ServerState {
                initialized: true,
                ..state
            },
            vec![],
        ),

        Event::SolverFinished { resumable } => {
            let log = format!("Solver finished. Resumable={}", title_bool(resumable));
            (
                ServerState {
                    solver: Solver::Idle,
                    initialized: false,
                    resumable,
                    ..state
                },
                vec![Effect::DoLog { message: log }],
            )
        }

        Event::SolverCrashed { error, violations } => {
            let log = format!("Solver crashed: {error}");
            (
                ServerState {
                    solver: Solver::Failed,
                    initialized: false,
                    error,
                    violations,
                    ..state
                },
                vec![Effect::DoLog { message: log }],
            )
        }

        Event::SolverSaving => (
            ServerState {
                solver: Solver::Saving,
                ..state
            },
            vec![],
        ),

        // Engine thinks the solver is idle but the monitor saw a live
        // ppf-contact-solver process under our project root: Jupyter
        // started a run that didn't go through our effect pipeline.
        // Promote `solver` to Running and, when the data files are
        // present, advance `data` / `build` accordingly so the addon's
        // status string reads BUSY instead of NO_DATA / NO_BUILD.
        // Anything other than `Solver::Idle` means the engine is
        // already tracking a launch (or a crash, or saving), and we
        // leave it alone.
        Event::ExternalSolverAdopted { has_data_on_disk }
            if state.solver == Solver::Idle && state.build != Build::Building =>
        {
            let data = if has_data_on_disk {
                Data::Uploaded
            } else {
                state.data
            };
            let build = if has_data_on_disk && state.build == Build::None {
                Build::Built
            } else {
                state.build
            };
            (
                ServerState {
                    solver: Solver::Running,
                    data,
                    build,
                    ..state
                },
                vec![Effect::DoLog {
                    message: "Adopting externally-launched solver".to_string(),
                }],
            )
        }

        // ── Delete ─────────────────────────────────────────────────
        Event::DeleteRequested if !state.root.is_empty() => (
            // Fresh state with only the project name preserved. Wipes
            // upload_id so no in-flight build from any client matches
            // after this.
            ServerState {
                name: state.name.clone(),
                ..ServerState::default()
            },
            vec![
                Effect::DoKillSolver,
                Effect::DoDeleteProjectData {
                    root: state.root,
                },
                Effect::DoLog {
                    message: "Project data deleted.".to_string(),
                },
            ],
        ),

        // ── Generic error ─────────────────────────────────────────
        Event::ErrorOccurred { error } => (
            ServerState {
                error: error.clone(),
                ..state
            },
            vec![Effect::DoLog { message: error }],
        ),

        // ── Guard-rejected or unknown ─────────────────────────────
        _ => (state, vec![]),
    }
}

/// Python's `bool.__str__` is "True"/"False" (capitalized). The
/// `SolverFinished` log message format is locked in by the Python
/// reference (`f"Solver finished. Resumable={r}"`). Reproduce that
/// titlecase exactly so log parsers don't drift.
fn title_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

#[cfg(test)]
mod tests;
