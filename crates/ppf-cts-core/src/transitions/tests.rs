// File: crates/ppf-cts-core/src/transitions/tests.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// 1:1 port of server/test_transitions.py. Each Python test function
// becomes one Rust `#[test]`. Same fixtures, same assertions. The
// Python tests are still being run on every commit (see
// server/test_transitions.py); these Rust tests are the parity
// guarantee that `transition()` here behaves identically.

use super::transition;
use crate::effects::Effect;
use crate::events::Event;
use crate::state::{Build, Data, ServerState, Solver};

fn has_effect(effects: &[Effect], pred: impl Fn(&Effect) -> bool) -> bool {
    effects.iter().any(pred)
}

// ---------------------------------------------------------------------------
// Project selection

#[test]
fn new_project_no_data() {
    let s = ServerState::default();
    let (s2, _fx) = transition(
        s,
        Event::ProjectSelected {
            name: "test".into(),
            root: "/tmp/test".into(),
            has_data: false,
            has_param: false,
            has_app: false,
            is_resumable: false,
            upload_id: String::new(),
            data_hash: String::new(),
            param_hash: String::new(),
        },
    );
    assert_eq!(s2.name, "test");
    assert_eq!(s2.root, "/tmp/test");
    assert_eq!(s2.data, Data::Empty);
    assert_eq!(s2.build, Build::None);
}

#[test]
fn new_project_with_data() {
    let s = ServerState::default();
    let (s2, fx) = transition(
        s,
        Event::ProjectSelected {
            name: "test".into(),
            root: "/tmp/test".into(),
            has_data: true,
            has_param: true,
            has_app: false,
            is_resumable: false,
            upload_id: String::new(),
            data_hash: String::new(),
            param_hash: String::new(),
        },
    );
    assert_eq!(s2.data, Data::Uploaded);
    assert_eq!(s2.build, Build::None);
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoLoadApp { .. })));
}

#[test]
fn new_project_with_existing_app() {
    let s = ServerState::default();
    let (s2, fx) = transition(
        s,
        Event::ProjectSelected {
            name: "test".into(),
            root: "/tmp/test".into(),
            has_data: true,
            has_param: true,
            has_app: true,
            is_resumable: true,
            upload_id: String::new(),
            data_hash: String::new(),
            param_hash: String::new(),
        },
    );
    assert_eq!(s2.build, Build::Built);
    assert!(s2.resumable);
    assert!(!has_effect(&fx, |e| matches!(e, Effect::DoLoadApp { .. })));
}

#[test]
fn same_project_refreshes_data() {
    let s = ServerState {
        name: "test".into(),
        root: "/tmp/test".into(),
        data: Data::Empty,
        build: Build::Built,
        solver: Solver::Running,
        frame: 10,
        ..Default::default()
    };
    let (s2, _fx) = transition(
        s,
        Event::ProjectSelected {
            name: "test".into(),
            root: "/tmp/test".into(),
            has_data: true,
            has_param: true,
            has_app: false,
            is_resumable: false,
            upload_id: String::new(),
            data_hash: String::new(),
            param_hash: String::new(),
        },
    );
    assert_eq!(s2.data, Data::Uploaded);
    assert_eq!(s2.solver, Solver::Running);
    assert_eq!(s2.frame, 10);
}

#[test]
fn project_selected_stamps_upload_id() {
    let (s2, _) = transition(
        ServerState::default(),
        Event::ProjectSelected {
            name: "p".into(),
            root: "/tmp/p".into(),
            has_data: true,
            has_param: true,
            has_app: false,
            is_resumable: false,
            upload_id: "abc123".into(),
            data_hash: String::new(),
            param_hash: String::new(),
        },
    );
    assert_eq!(s2.upload_id, "abc123");
}

#[test]
fn same_project_refreshes_upload_id() {
    let s = ServerState {
        name: "p".into(),
        root: "/tmp/p".into(),
        data: Data::Uploaded,
        upload_id: "old".into(),
        ..Default::default()
    };
    let (s2, _) = transition(
        s,
        Event::ProjectSelected {
            name: "p".into(),
            root: "/tmp/p".into(),
            has_data: true,
            has_param: true,
            has_app: false,
            is_resumable: false,
            upload_id: "new".into(),
            data_hash: String::new(),
            param_hash: String::new(),
        },
    );
    assert_eq!(s2.upload_id, "new");
}

// ---------------------------------------------------------------------------
// Upload landing

#[test]
fn upload_stamps_id_and_marks_uploaded() {
    let s = ServerState {
        name: "p".into(),
        root: "/tmp/p".into(),
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::upload_landed("xyz"));
    assert_eq!(s2.upload_id, "xyz");
    assert_eq!(s2.data, Data::Uploaded);
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoLog { .. })));
}

#[test]
fn partial_upload_keeps_data_empty() {
    let s = ServerState {
        name: "p".into(),
        root: "/tmp/p".into(),
        ..Default::default()
    };
    let (s2, _) = transition(
        s,
        Event::UploadLanded {
            upload_id: "xyz".into(),
            data_hash: String::new(),
            param_hash: String::new(),
            has_data: true,
            has_param: false,
        },
    );
    assert_eq!(s2.upload_id, "xyz");
    assert_eq!(s2.data, Data::Empty);
}

#[test]
fn upload_invalidates_built_state() {
    let s = ServerState {
        name: "p".into(),
        root: "/tmp/p".into(),
        data: Data::Uploaded,
        build: Build::Built,
        upload_id: "old".into(),
        resumable: true,
        ..Default::default()
    };
    let (s2, _) = transition(s, Event::upload_landed("new"));
    assert_eq!(s2.upload_id, "new");
    assert_eq!(s2.build, Build::None);
    assert!(!s2.resumable);
}

#[test]
fn upload_during_building_preserves_build_state() {
    let s = ServerState {
        name: "p".into(),
        root: "/tmp/p".into(),
        data: Data::Uploaded,
        build: Build::Building,
        upload_id: "old".into(),
        ..Default::default()
    };
    let (s2, _) = transition(s, Event::upload_landed("new"));
    assert_eq!(s2.upload_id, "new");
    assert_eq!(s2.build, Build::Building);
}

// ---------------------------------------------------------------------------
// Build

#[test]
fn build_from_uploaded() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::None,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::BuildRequested);
    assert_eq!(s2.build, Build::Building);
    assert_eq!(s2.build_progress, 0.0);
    assert_eq!(s2.error, "");
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoSpawnBuild)));
}

#[test]
fn build_rejected_no_data() {
    let s = ServerState {
        data: Data::Empty,
        ..Default::default()
    };
    let (s2, fx) = transition(s.clone(), Event::BuildRequested);
    assert_eq!(s2, s);
    assert!(fx.is_empty());
}

#[test]
fn build_rejected_already_building() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Building,
        ..Default::default()
    };
    let (s2, fx) = transition(s.clone(), Event::BuildRequested);
    assert_eq!(s2, s);
    assert!(fx.is_empty());
}

#[test]
fn build_progress() {
    let s = ServerState {
        build: Build::Building,
        build_progress: 0.1,
        ..Default::default()
    };
    let (s2, _) = transition(
        s,
        Event::BuildProgress {
            progress: 0.5,
            info: "Encoding meshes...".into(),
        },
    );
    assert_eq!(s2.build_progress, 0.5);
    assert_eq!(s2.build_info, "Encoding meshes...");
}

#[test]
fn build_completed() {
    let s = ServerState {
        build: Build::Building,
        build_progress: 0.9,
        root: "/tmp/test".into(),
        ..Default::default()
    };
    let (s2, _) = transition(s, Event::BuildCompleted);
    assert_eq!(s2.build, Build::Built);
    assert_eq!(s2.build_progress, 1.0);
    assert!(!s2.resumable);
}

#[test]
fn build_failed() {
    let s = ServerState {
        build: Build::Building,
        ..Default::default()
    };
    let (s2, _) = transition(
        s,
        Event::BuildFailed {
            error: "decode error".into(),
            violations: vec![],
        },
    );
    assert_eq!(s2.build, Build::Failed);
    assert_eq!(s2.error, "decode error");
}

#[test]
fn gpu_check_failed() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Building,
        ..Default::default()
    };
    let (s2, _) = transition(
        s,
        Event::GpuCheckFailed {
            error: "No CUDA device".into(),
        },
    );
    assert_eq!(s2.build, Build::Failed);
    assert_eq!(s2.error, "No CUDA device");
}

#[test]
fn rebuild_after_built() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Built,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::BuildRequested);
    assert_eq!(s2.build, Build::Building);
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoSpawnBuild)));
}

// ---------------------------------------------------------------------------
// Solver operations

#[test]
fn start() {
    let s = ServerState {
        build: Build::Built,
        solver: Solver::Idle,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::StartRequested);
    assert_eq!(s2.solver, Solver::Running);
    assert_eq!(s2.frame, 0);
    assert_eq!(s2.error, "");
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoLaunchSolver { .. })));
}

#[test]
fn start_rejected_not_built() {
    let s = ServerState {
        build: Build::None,
        solver: Solver::Idle,
        ..Default::default()
    };
    let (s2, _) = transition(s.clone(), Event::StartRequested);
    assert_eq!(s2, s);
}

#[test]
fn start_rejected_already_running() {
    let s = ServerState {
        build: Build::Built,
        solver: Solver::Running,
        ..Default::default()
    };
    let (s2, _) = transition(s.clone(), Event::StartRequested);
    assert_eq!(s2, s);
}

#[test]
fn resume() {
    let s = ServerState {
        build: Build::Built,
        solver: Solver::Idle,
        resumable: true,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::ResumeRequested);
    assert_eq!(s2.solver, Solver::Running);
    let launch = fx.iter().find_map(|e| match e {
        Effect::DoLaunchSolver { resume_from } => Some(*resume_from),
        _ => None,
    });
    assert_eq!(launch, Some(Some(-1)));
}

#[test]
fn resume_rejected_not_resumable() {
    let s = ServerState {
        build: Build::Built,
        solver: Solver::Idle,
        resumable: false,
        ..Default::default()
    };
    let (s2, _) = transition(s.clone(), Event::ResumeRequested);
    assert_eq!(s2, s);
}

#[test]
fn terminate() {
    let s = ServerState {
        solver: Solver::Running,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::TerminateRequested);
    assert_eq!(s2.solver, Solver::Idle);
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoKillSolver)));
}

#[test]
fn terminate_while_saving() {
    let s = ServerState {
        solver: Solver::Saving,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::TerminateRequested);
    assert_eq!(s2.solver, Solver::Idle);
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoKillSolver)));
}

#[test]
fn terminate_idle_still_fires() {
    let s = ServerState {
        solver: Solver::Idle,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::TerminateRequested);
    assert_eq!(s2.solver, Solver::Idle);
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoKillSolver)));
}

#[test]
fn save_and_quit() {
    let s = ServerState {
        solver: Solver::Running,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::SaveAndQuitRequested);
    assert_eq!(s2.solver, Solver::Saving);
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoRequestSaveAndQuit)));
}

#[test]
fn save_and_quit_idle_still_fires() {
    let s = ServerState {
        solver: Solver::Idle,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::SaveAndQuitRequested);
    assert_eq!(s2.solver, Solver::Saving);
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoRequestSaveAndQuit)));
}

// ---------------------------------------------------------------------------
// Solver monitor events

#[test]
fn frame_update() {
    let s = ServerState {
        solver: Solver::Running,
        frame: 5,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::SolverFrameUpdated { frame: 6 });
    assert_eq!(s2.frame, 6);
    assert!(fx.is_empty());
}

#[test]
fn solver_finished() {
    let s = ServerState {
        solver: Solver::Running,
        frame: 100,
        initialized: true,
        ..Default::default()
    };
    let (s2, _) = transition(s, Event::SolverFinished { resumable: true });
    assert_eq!(s2.solver, Solver::Idle);
    assert!(s2.resumable);
    assert!(!s2.initialized, "finished resets initialized");
}

#[test]
fn solver_initialized_flips_flag() {
    let s = ServerState {
        solver: Solver::Running,
        frame: 0,
        initialized: false,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::SolverInitialized);
    assert!(s2.initialized, "SolverInitialized must set the flag");
    assert_eq!(s2.solver, Solver::Running, "solver state stays Running");
    assert_eq!(s2.frame, 0, "frame counter is independent of initialized");
    assert!(fx.is_empty(), "no side effects from the init signal");
}

#[test]
fn start_resets_initialized() {
    let s = ServerState {
        build: Build::Built,
        solver: Solver::Idle,
        initialized: true,
        frame: 9,
        ..Default::default()
    };
    let (s2, _) = transition(s, Event::StartRequested);
    assert_eq!(s2.solver, Solver::Running);
    assert_eq!(s2.frame, 0);
    assert!(!s2.initialized, "start clears the prior run's initialized flag");
}

#[test]
fn resume_resets_initialized() {
    let s = ServerState {
        build: Build::Built,
        solver: Solver::Idle,
        resumable: true,
        initialized: true,
        ..Default::default()
    };
    let (s2, _) = transition(s, Event::ResumeRequested);
    assert_eq!(s2.solver, Solver::Running);
    assert!(!s2.initialized, "resume clears the prior run's initialized flag");
}

#[test]
fn terminate_resets_initialized() {
    let s = ServerState {
        solver: Solver::Running,
        initialized: true,
        ..Default::default()
    };
    let (s2, _) = transition(s, Event::TerminateRequested);
    assert_eq!(s2.solver, Solver::Idle);
    assert!(!s2.initialized);
}

#[test]
fn solver_finished_not_resumable() {
    let s = ServerState {
        solver: Solver::Running,
        ..Default::default()
    };
    let (s2, _) = transition(s, Event::SolverFinished { resumable: false });
    assert_eq!(s2.solver, Solver::Idle);
    assert!(!s2.resumable);
}

#[test]
fn solver_crashed() {
    let s = ServerState {
        solver: Solver::Running,
        frame: 50,
        ..Default::default()
    };
    let (s2, _) = transition(
        s,
        Event::SolverCrashed {
            error: "Segfault".into(),
            violations: vec![],
        },
    );
    assert_eq!(s2.solver, Solver::Failed);
    assert_eq!(s2.error, "Segfault");
}

#[test]
fn solver_saving_event() {
    let s = ServerState {
        solver: Solver::Running,
        ..Default::default()
    };
    let (s2, _) = transition(s, Event::SolverSaving);
    assert_eq!(s2.solver, Solver::Saving);
}

#[test]
fn save_finished() {
    let s = ServerState {
        solver: Solver::Saving,
        ..Default::default()
    };
    let (s2, _) = transition(s, Event::SolverFinished { resumable: true });
    assert_eq!(s2.solver, Solver::Idle);
    assert!(s2.resumable);
}

// ---------------------------------------------------------------------------
// Delete

#[test]
fn delete() {
    let s = ServerState {
        name: "test".into(),
        root: "/tmp/test".into(),
        data: Data::Uploaded,
        build: Build::Built,
        solver: Solver::Idle,
        frame: 50,
        ..Default::default()
    };
    let (s2, fx) = transition(s, Event::DeleteRequested);
    assert_eq!(s2.data, Data::Empty);
    assert_eq!(s2.build, Build::None);
    assert_eq!(s2.solver, Solver::Idle);
    assert_eq!(s2.frame, 0);
    assert_eq!(s2.name, "test");
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoKillSolver)));
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoDeleteProjectData { .. })));
}

#[test]
fn delete_no_root() {
    let s = ServerState {
        name: "test".into(),
        root: String::new(),
        ..Default::default()
    };
    let (s2, _) = transition(s.clone(), Event::DeleteRequested);
    assert_eq!(s2, s);
}

// ---------------------------------------------------------------------------
// Status string

#[test]
fn status_no_data() {
    assert_eq!(ServerState::default().status_string(), "NO_DATA");
}

#[test]
fn status_no_build() {
    let s = ServerState {
        data: Data::Uploaded,
        ..Default::default()
    };
    assert_eq!(s.status_string(), "NO_BUILD");
}

#[test]
fn status_building() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Building,
        ..Default::default()
    };
    assert_eq!(s.status_string(), "BUILDING");
}

#[test]
fn status_ready() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Built,
        ..Default::default()
    };
    assert_eq!(s.status_string(), "READY");
}

#[test]
fn status_resumable() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Built,
        resumable: true,
        ..Default::default()
    };
    assert_eq!(s.status_string(), "RESUMABLE");
}

#[test]
fn status_busy() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Built,
        solver: Solver::Running,
        ..Default::default()
    };
    assert_eq!(s.status_string(), "BUSY");
}

#[test]
fn status_saving() {
    let s = ServerState {
        solver: Solver::Saving,
        ..Default::default()
    };
    assert_eq!(s.status_string(), "SAVE_AND_QUIT");
}

#[test]
fn status_failed_solver() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Built,
        solver: Solver::Failed,
        ..Default::default()
    };
    assert_eq!(s.status_string(), "FAILED");
}

#[test]
fn status_failed_build_with_error_routes_no_build() {
    // Mirrors `test_failed_build` from the Python suite. Even with an
    // error stamped, Build::Failed maps to NO_BUILD because the
    // status check considers Build::Failed equivalent to "no
    // successful build" before the error string.
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Failed,
        error: "decoding failed".into(),
        ..Default::default()
    };
    assert_eq!(s.status_string(), "NO_BUILD");
}

#[test]
fn status_built_with_error_returns_failed() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Built,
        error: "oops".into(),
        ..Default::default()
    };
    assert_eq!(s.status_string(), "FAILED");
}

#[test]
fn status_build_failed_alone() {
    let s = ServerState {
        data: Data::Uploaded,
        build: Build::Failed,
        ..Default::default()
    };
    assert_eq!(s.status_string(), "NO_BUILD");
}

// ---------------------------------------------------------------------------
// Generic error

#[test]
fn error_event() {
    let s = ServerState::default();
    let (s2, fx) = transition(
        s,
        Event::ErrorOccurred {
            error: "something broke".into(),
        },
    );
    assert_eq!(s2.error, "something broke");
    assert!(has_effect(&fx, |e| matches!(e, Effect::DoLog { .. })));
}

// ---------------------------------------------------------------------------
// External solver adoption (monitor-driven)

#[test]
fn external_solver_adopt_promotes_to_running() {
    // Engine is otherwise idle but the monitor saw a live ppf-contact
    // process under our project root (e.g. JupyterLab launched it
    // outside the effect pipeline). With pickles on disk, data + build
    // also advance so the response builder reads BUSY rather than
    // NO_DATA / NO_BUILD.
    let s = ServerState {
        name: "p".into(),
        root: "/tmp/p".into(),
        ..ServerState::default()
    };
    let (s2, _) = transition(
        s,
        Event::ExternalSolverAdopted {
            has_data_on_disk: true,
        },
    );
    assert_eq!(s2.solver, Solver::Running);
    assert_eq!(s2.data, Data::Uploaded);
    assert_eq!(s2.build, Build::Built);
}

#[test]
fn external_solver_adopt_without_pickles_keeps_data_state() {
    // Pickles missing on disk: still promote solver to Running so the
    // monitor's frame / finish / crash branches run, but leave data /
    // build alone.
    let s = ServerState {
        name: "p".into(),
        root: "/tmp/p".into(),
        ..ServerState::default()
    };
    let (s2, _) = transition(
        s,
        Event::ExternalSolverAdopted {
            has_data_on_disk: false,
        },
    );
    assert_eq!(s2.solver, Solver::Running);
    assert_eq!(s2.data, Data::Empty);
    assert_eq!(s2.build, Build::None);
}

#[test]
fn external_solver_adopt_skipped_while_building() {
    // Don't ride over a live build. The build worker subprocess can
    // briefly look like a busy ppf-contact descendant on some
    // platforms; the guard keeps Build::Building intact.
    let s = ServerState {
        name: "p".into(),
        root: "/tmp/p".into(),
        build: Build::Building,
        ..ServerState::default()
    };
    let (s2, fx) = transition(
        s,
        Event::ExternalSolverAdopted {
            has_data_on_disk: true,
        },
    );
    assert_eq!(s2.solver, Solver::Idle);
    assert_eq!(s2.build, Build::Building);
    assert!(fx.is_empty(), "guarded transition emits no effects");
}

#[test]
fn external_solver_adopt_skipped_when_solver_already_running() {
    // The engine already tracks a launch (our own DoLaunchSolver, or a
    // prior adoption). Re-adopting would double-fire the DoLog effect.
    let s = ServerState {
        name: "p".into(),
        root: "/tmp/p".into(),
        solver: Solver::Running,
        ..ServerState::default()
    };
    let (s2, fx) = transition(
        s,
        Event::ExternalSolverAdopted {
            has_data_on_disk: true,
        },
    );
    assert_eq!(s2.solver, Solver::Running);
    assert!(fx.is_empty());
}
