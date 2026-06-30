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
pub struct HardwareInfo {
    #[serde(rename = "GPU")]
    pub gpu: String,
    #[serde(rename = "VRAM")]
    pub vram: String,
    #[serde(rename = "CUDA")]
    pub cuda: String,
    #[serde(rename = "SM")]
    pub sm: String,
    #[serde(rename = "CPU")]
    pub cpu: String,
    #[serde(rename = "RAM")]
    pub ram: String,
    /// True when the server was compiled with the `emulated` feature
    /// (the CPU stub backend used by the test rig, no CUDA). The addon
    /// reads this off every status response to warn before running a
    /// simulation that would not produce real physics.
    #[serde(rename = "emulated", default)]
    pub emulated: bool,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        Self {
            gpu: "Unknown".into(),
            vram: "Unknown".into(),
            cuda: "Unknown".into(),
            sm: "Unknown".into(),
            cpu: "Unknown".into(),
            ram: "Unknown".into(),
            emulated: false,
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
        }
    }
}
