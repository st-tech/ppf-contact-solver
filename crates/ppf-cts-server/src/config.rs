// File: crates/ppf-cts-server/src/config.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Static engine configuration. Values are gathered once at startup
// (hardware probe, git branch read) and shipped on every status
// response.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// What the add-on's "Remote Hardware" block renders, one row per field that is
/// present.
///
/// A FIELD THAT DOES NOT APPLY TO THIS BACKEND IS ABSENT, NOT "Unknown". Every
/// GPU field here except the backend name describes an NVIDIA device, so on a
/// Metal or CPU build they used to reach the panel as four rows reading
/// `Unknown` and one headed `CUDA`, which says the server could not find its
/// hardware when the truth is that the question does not arise. `Option` plus
/// `skip_serializing_if` is what removes the row rather than filling it with a
/// word the reader has to discount, and it leaves "Unknown" meaning what it
/// should: a probe that applied and failed.
///
/// The add-on renders whatever keys arrive, in this order, so field order here
/// is row order there.
pub struct HardwareInfo {
    /// Which backend this server's build targets: `cuda`, `metal`, `rocm` or
    /// `cpu`.
    ///
    /// FIRST, AND ALWAYS PRESENT, because it is what tells a reader why the
    /// rows below it are the ones they are. It also lets the add-on decide
    /// whether a CUDA-shaped question applies at all, instead of inferring that
    /// from its own connection type, which is right only for a locally
    /// launched server.
    #[serde(rename = "Backend")]
    pub backend: String,
    #[serde(rename = "GPU", default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// The Metal GPU family, for example `Apple8`. The analogue of the `SM`
    /// row below, and absent for the same reason that one is: neither concept
    /// exists on the other's backend.
    #[serde(rename = "GPU Family", default, skip_serializing_if = "Option::is_none")]
    pub gpu_family: Option<String>,
    /// CUDA index of the device the solver will run on, or -1 when no device
    /// could be resolved. The add-on compares it against the GPU it picked, so
    /// what it shows is the server's own answer rather than its own intent,
    /// which is the only thing that holds when the add-on attached to a server
    /// it did not launch. Absent on backends that have no such index.
    #[serde(rename = "GPU Index", default, skip_serializing_if = "Option::is_none")]
    pub gpu_index: Option<i64>,
    #[serde(rename = "VRAM", default, skip_serializing_if = "Option::is_none")]
    pub vram: Option<String>,
    #[serde(rename = "CUDA", default, skip_serializing_if = "Option::is_none")]
    pub cuda: Option<String>,
    #[serde(rename = "SM", default, skip_serializing_if = "Option::is_none")]
    pub sm: Option<String>,
    #[serde(rename = "CPU")]
    pub cpu: String,
    #[serde(rename = "RAM")]
    pub ram: String,
}

impl Default for HardwareInfo {
    /// The backend is filled from THIS build rather than left blank, so a
    /// server that never ran a probe still names it correctly. CPU and RAM stay
    /// "Unknown" because their probe applies on every backend and may fail; the
    /// GPU fields start absent because on two of the three backends they never
    /// become anything else.
    fn default() -> Self {
        Self {
            backend: ppf_cts_core::utils::backend().name().into(),
            gpu: None,
            gpu_family: None,
            gpu_index: None,
            vram: None,
            cuda: None,
            sm: None,
            cpu: "Unknown".into(),
            ram: "Unknown".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub hardware: HardwareInfo,
    pub git_branch: String,
    /// Monitor poll interval in milliseconds. Defaults to 250 (matches
    /// server/monitor.py's `0.25` second tick).
    pub monitor_interval_ms: u64,
    /// Solver startup grace period in milliseconds. Defaults to 3000
    /// (matches server/monitor.py's `SOLVER_STARTUP_GRACE = 3.0`
    /// seconds). Tests can lower this to keep wall-clock short.
    pub solver_startup_grace_ms: u64,
    /// Backoff in milliseconds applied after a failed `accept()` so the
    /// accept loop does not busy-spin on hard failures (e.g. fd
    /// exhaustion). Defaults to 50.
    pub accept_backoff_ms: u64,
    /// Log channel `(name, filename)` pairs harvested at startup from
    /// the project's `src/` tree via
    /// `ppf_cts_core::parsers::get_logging_docstrings`. Used by
    /// `response::build_response` to populate the live `summary` and
    /// post-sim `average_summary` fields. Empty when the source tree
    /// can't be located (the response simply omits those fields, and
    /// the addon's stats panel falls back to the base "Average
    /// Statistics" path).
    pub log_filenames: Vec<(String, String)>,
    /// Override the project data root. When `Some`, every project's
    /// disk path is `<override>/git-<branch>/<name>`; when `None`, the
    /// resolution falls back to `PPF_CTS_DATA_ROOT` env var, then to
    /// the canonical `~/.local/share/ppf-cts/git-<branch>/<name>`.
    /// Tests use this override so each `#[tokio::test]` can plant
    /// uploads in its own tempdir without racing on a process-global
    /// env var.
    pub data_root: Option<PathBuf>,
    /// The cargo target directory this server's runs take the solver from,
    /// absolute, or empty when the build worker cannot be located. See
    /// `executor::solver_build`, which fills it at startup.
    pub solver_target_dir: String,
    /// What the solver in `solver_target_dir` printed for `--backend`, or
    /// empty when there is no solver there or it could not be asked.
    pub solver_backend: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            hardware: HardwareInfo::default(),
            git_branch: "unknown".into(),
            monitor_interval_ms: 250,
            solver_startup_grace_ms: 3000,
            accept_backoff_ms: 50,
            log_filenames: Vec::new(),
            data_root: None,
            solver_target_dir: String::new(),
            solver_backend: String::new(),
        }
    }
}
