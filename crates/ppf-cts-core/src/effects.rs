// File: crates/ppf-cts-core/src/effects.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Typed effect enum for the server state machine. Direct 1:1 port of
// server/effects.py. Effects describe side-effects to perform; they
// carry no behavior. The executor (in ppf-cts-server) is the only
// place they're actually run.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    // ---- Build effects ----
    /// Spawn the background build task.
    DoSpawnBuild,
    /// Signal the build task to stop (via CancellationToken).
    DoCancelBuild,

    // ---- Solver effects ----
    /// Start the solver subprocess.
    /// `resume_from = None` for fresh start, `Some(-1)` for latest
    /// checkpoint (matching the Python `int | None` semantics).
    DoLaunchSolver { resume_from: Option<i32> },
    /// Terminate the solver subprocess.
    DoKillSolver,
    /// Create the `save_and_quit` flag file the solver polls for.
    DoRequestSaveAndQuit,

    // ---- Disk I/O effects ----
    /// Deserialize app_state from disk.
    DoLoadApp { name: String, root: String },
    /// Delete the entire project directory.
    DoDeleteProjectData { root: String },

    // ---- Logging ----
    /// Write a message to the server log.
    DoLog { message: String },
}
