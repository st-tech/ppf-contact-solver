// File: crates/ppf-cts-core/src/state.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Frozen `ServerState` + orthogonal phase enums. The struct is replaced
// atomically inside `crate::transitions::transition`; nothing else
// constructs or mutates it.
//
// Equality and Clone are derived so engine code can compare states in
// tests and snapshot states under a lock without duplicating logic.
// PartialEq (not Eq) because `build_progress: f64` carries NaN risk.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum Data {
    /// Either data.pickle or param.pickle is missing.
    #[default]
    Empty,
    /// Both files present on disk; data is "uploaded".
    Uploaded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum Build {
    /// Never built or just deleted.
    #[default]
    None,
    /// Build thread is running.
    Building,
    /// App ready to launch the solver.
    Built,
    /// Build error.
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum Solver {
    /// Not running.
    #[default]
    Idle,
    /// Subprocess active.
    Running,
    /// `save_and_quit` flag file written, waiting for solver to drain.
    Saving,
    /// Crashed.
    Failed,
}


/// Complete, immutable snapshot of the server state.
///
/// New states are produced from old ones via Rust's struct update
/// syntax inside the transition function:
/// ```ignore
/// ServerState { build: Build::Building, ..state }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ServerState {
    /// Project identity.
    pub name: String,
    pub root: String,

    /// Orthogonal phases.
    pub data: Data,
    pub build: Build,
    pub solver: Solver,

    /// 12-char hex upload identity. Generated when data.pickle /
    /// param.pickle land; cleared when the project is deleted.
    /// Persisted to upload_id.txt alongside the data files so it
    /// survives server restarts as long as the files do.
    pub upload_id: String,

    /// Quick fingerprints of the last uploaded `data.pickle` and
    /// `param.pickle`. Producer-side (addon) computes them; the server
    /// echoes them on every status response so the client can detect
    /// drift between live scene and what's on the server.
    pub data_hash: String,
    pub param_hash: String,

    /// Extra context.
    pub resumable: bool,
    pub frame: i32,
    /// True once the running solver has finished its in-process
    /// `initialize()` call and written `<output>/initialize_finish.txt`.
    /// The legacy `solver = RUNNING if frame >= 1 else STARTING`
    /// heuristic kept the addon's status pinned at "Initializing"
    /// through the entire first step (which can run for several seconds
    /// under heavy contact load). Once `initialized` is true, the addon
    /// flips to "Running" immediately even with `frame == 0`. Reset on
    /// every Start/Resume/Terminate/SolverFinished/SolverCrashed so a
    /// re-run starts uninitialized again.
    pub initialized: bool,
    pub build_progress: f64,
    pub build_info: String,
    pub error: String,

    /// Total frames in the current build's param set, captured by the
    /// build worker and reported back via `Event::BuildMetadata`. The
    /// response builder uses this to publish solver progress
    /// (`frame / total_frames`) while the solver is `Running`.
    /// Zero when not yet known (pre-build or build worker too old).
    pub total_frames: i32,

    /// Opaque per-event violation payload, inspected by the response
    /// generator only, not by transitions.
    pub violations: Vec<String>,
}

impl ServerState {
    /// Status string sent to clients on every poll. Branch order
    /// is significant: preserve it.
    pub fn status_string(&self) -> &'static str {
        if self.build == Build::Building {
            return "BUILDING";
        }
        if self.solver == Solver::Running {
            return "BUSY";
        }
        if self.solver == Solver::Saving {
            return "SAVE_AND_QUIT";
        }
        if self.data == Data::Empty {
            return "NO_DATA";
        }
        if matches!(self.build, Build::None | Build::Failed) {
            return "NO_BUILD";
        }
        if self.solver == Solver::Failed {
            return "FAILED";
        }
        if !self.error.is_empty() {
            return "FAILED";
        }
        if self.resumable {
            return "RESUMABLE";
        }
        "READY"
    }

    /// Protocol-level data availability string.
    pub fn data_string(&self) -> &'static str {
        if self.data == Data::Uploaded {
            "READY"
        } else {
            "NO_DATA"
        }
    }
}
