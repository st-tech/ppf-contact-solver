// File: crates/ppf-cts-core/src/events.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Typed event enum for the server state machine. Direct 1:1 port of
// server/events.py: every Python event class becomes a Rust enum
// variant carrying the same fields.
//
// Events come from three sources:
//   1. Client requests parsed from the socket protocol.
//   2. Build task callbacks (progress, completion, failure).
//   3. Solver monitor task (frame ticks, exit detection, crash).

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    // ---- Client request events ----

    /// Client connected with a project name. Sets up project context.
    /// `upload_id` reconciles disk identity: when the server restarts
    /// or a different process wrote data to the project dir, the id
    /// from upload_id.txt is stamped here without disturbing the
    /// build state.
    ProjectSelected {
        name: String,
        root: String,
        has_data: bool,
        has_param: bool,
        has_app: bool,
        is_resumable: bool,
        upload_id: String,
        data_hash: String,
        param_hash: String,
        /// `Total Frames` from scene_info.json when readable; 0 when
        /// the project has never been built (no scene_info.json yet)
        /// or the field is missing/unparseable. Lets a reconnect (or
        /// any other re-select) rehydrate state.total_frames from
        /// disk so the progress bar keeps working after the connect
        /// probe's `--name __probe__` transition cleared the field.
        total_frames: i32,
    },

    /// Client requested a scene build.
    BuildRequested,

    /// Client requested cancellation of an in-progress build.
    CancelBuildRequested,

    /// Client requested simulation start.
    StartRequested,

    /// Client requested simulation resume from the latest checkpoint.
    ResumeRequested,

    /// Client requested solver termination.
    TerminateRequested,

    /// Client requested save-and-quit.
    SaveAndQuitRequested,

    /// Client requested project data deletion.
    DeleteRequested,

    /// A `data.pickle` / `param.pickle` write completed. Fresh
    /// `upload_id` is stamped onto the state so any in-flight builds
    /// from other clients know their target was replaced.
    UploadLanded {
        upload_id: String,
        data_hash: String,
        param_hash: String,
        has_data: bool,
        has_param: bool,
    },

    // ---- Build task events ----
    /// Build thread reports a progress update.
    BuildProgress { progress: f64, info: String },

    /// Build worker forwards the per-build static metadata it learned
    /// from `param.pickle` (currently `frames`). Dispatched once per
    /// build, after `make()` finishes parsing the param set, so the
    /// solver-progress field on status responses can publish
    /// `frame / total_frames` without re-reading the param file.
    BuildMetadata { total_frames: i32 },

    /// Build task finished successfully.
    BuildCompleted,

    /// Build task encountered an error. `violations` is opaque.
    BuildFailed {
        error: String,
        violations: Vec<String>,
    },

    /// Build task was cancelled cooperatively (via the
    /// CancellationToken from `crate::cancel`).
    BuildCancelledEvent,

    /// GPU availability check failed before build.
    GpuCheckFailed { error: String },

    // ---- Solver monitor events ----
    /// Monitor detected a new frame in the output directory.
    SolverFrameUpdated { frame: i32 },

    /// Monitor observed `<output>/initialize_finish.txt`, the file the
    /// solver writes immediately after its in-process `initialize()`
    /// call returns. Flips `state.initialized` to true so the addon's
    /// status text leaves "Initializing" before the first frame's
    /// advance completes.
    SolverInitialized,

    /// Monitor detected the solver subprocess exited cleanly.
    SolverFinished { resumable: bool },

    /// Monitor detected solver crash via stderr / log analysis.
    SolverCrashed {
        error: String,
        violations: Vec<String>,
    },

    /// Monitor detected the `save_and_quit` flag file.
    SolverSaving,

    /// Monitor detected a `ppf-contact-solver` process running outside
    /// this server (e.g. launched from a JupyterLab notebook against
    /// the same project root). Promotes `solver` to `Running` so the
    /// rest of the monitor loop drives frame / finish / crash
    /// transitions normally; without this the addon's status panel
    /// would stay frozen at "Ready to Run" for the duration of an
    /// externally-launched simulation. `has_data_on_disk` reflects
    /// whether `data.pickle` + `param.pickle` exist under the project
    /// root: when true, `data` and `build` are advanced to mirror what
    /// the running solver implies.
    ExternalSolverAdopted { has_data_on_disk: bool },

    // ---- Generic error ----
    /// A generic error from any subsystem.
    ErrorOccurred { error: String },
}

impl Event {
    /// Field-default constructors handy for tests and dispatch sites
    /// that don't need every field.
    pub fn upload_landed(upload_id: impl Into<String>) -> Self {
        Event::UploadLanded {
            upload_id: upload_id.into(),
            data_hash: String::new(),
            param_hash: String::new(),
            has_data: true,
            has_param: true,
        }
    }
}
