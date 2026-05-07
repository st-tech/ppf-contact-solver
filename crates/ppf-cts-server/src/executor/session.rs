// File: crates/ppf-cts-server/src/executor/session.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Lifecycle plumbing for `DoRequestSaveAndQuit` and `DoLoadApp`:
// touch the save sentinel the solver polls for, and reconcile the
// engine's project context with on-disk state on app load.

use std::path::{Path, PathBuf};

use ppf_cts_core::events::Event;

use super::dispatch_re_entrant;
use crate::engine::ServerEngine;

/// Touch the `save_and_quit` sentinel the solver polls for.
pub(super) async fn request_save_and_quit(engine: &ServerEngine) {
    let state = engine.state();
    if state.root.is_empty() {
        let msg = "no session found".to_string();
        log::error!(target: "ppf::session", "DoRequestSaveAndQuit: {msg}");
        dispatch_re_entrant(engine, Event::ErrorOccurred { error: msg }).await;
        return;
    }
    let path = PathBuf::from(&state.root)
        .join("session")
        .join("output")
        .join("save_and_quit");
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            log::error!(target: "ppf::session", "DoRequestSaveAndQuit: mkdir {parent:?} failed: {e}");
            dispatch_re_entrant(
                engine,
                Event::ErrorOccurred {
                    error: format!("mkdir: {e}"),
                },
            )
            .await;
            return;
        }
    }
    match std::fs::File::create(&path) {
        Ok(_) => log::info!(target: "ppf::session", "DoRequestSaveAndQuit: wrote {}", path.display()),
        Err(e) => {
            log::error!(target: "ppf::session", "DoRequestSaveAndQuit: touch failed: {e}");
            dispatch_re_entrant(
                engine,
                Event::ErrorOccurred {
                    error: format!("save_and_quit touch: {e}"),
                },
            )
            .await;
        }
    }
}

/// Set the engine's project context and ensure the project dir
/// exists. `app_state.pickle` is owned by the frontend Python addon
/// (it carries a pickled `(asset, scene, mesh, session, plot)` tuple
/// that only the addon's interpreter can rehydrate); the Rust binary
/// has no Python and never opens it.
///
/// The Rust binary's responsibility for `DoLoadApp` is therefore
/// limited to project-context bookkeeping: store `(name, root)` on
/// the engine and make sure the directory exists so subsequent
/// effects (`DoLaunchSolver`, `DoRequestSaveAndQuit`, monitor I/O)
/// can write to it. The frame counter, the `resumable` flag, and
/// the save-in-progress flag are reconciled by the monitor task
/// from on-disk artifacts (`vert_*.bin`, `state_*.bin.gz`,
/// `save_and_quit`), not from the pickle, so no additional state
/// needs to be hydrated here.
pub(super) async fn load_app(engine: &ServerEngine, name: &str, root: &str) {
    engine.set_project_context(name, root);
    if root.is_empty() {
        log::warn!(target: "ppf::session", "DoLoadApp: empty root for project {name:?}");
        return;
    }
    let path = Path::new(root);
    if !path.exists() {
        if let Err(e) = std::fs::create_dir_all(path) {
            log::error!(target: "ppf::session", "DoLoadApp: mkdir {root} failed: {e}");
            dispatch_re_entrant(
                engine,
                Event::ErrorOccurred {
                    error: format!("project dir create: {e}"),
                },
            )
            .await;
            return;
        }
        log::info!(target: "ppf::session", "DoLoadApp: created project dir {root}");
    } else {
        log::debug!(target: "ppf::session", "DoLoadApp: project dir already present {root}");
    }
}
