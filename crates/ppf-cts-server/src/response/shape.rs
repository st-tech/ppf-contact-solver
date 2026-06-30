// File: crates/ppf-cts-server/src/response/shape.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Pure JSON layout for the status response. No I/O: every value is
// derived from `ServerState` + `EngineConfig` directly.

use serde_json::{json, Map, Value};

use crate::config::EngineConfig;
use crate::PROTOCOL_VERSION;
use ppf_cts_core::state::{Build, ServerState, Solver};

/// Build the always-present base fields: status, data, frame,
/// initialized, error, violations, root, upload_id, hashes,
/// protocol_version, hardware, git_branch.
pub fn base_map(state: &ServerState, config: &EngineConfig) -> Map<String, Value> {
    let mut m: Map<String, Value> = Map::new();
    m.insert("status".into(), Value::String(state.status_string().into()));
    m.insert("data".into(), Value::String(state.data_string().into()));
    m.insert("frame".into(), json!(state.frame));
    m.insert("initialized".into(), json!(state.initialized));
    m.insert("error".into(), Value::String(state.error.clone()));
    m.insert(
        "violations".into(),
        Value::Array(violations_to_json(&state.violations)),
    );
    m.insert("root".into(), Value::String(state.root.clone()));
    m.insert("upload_id".into(), Value::String(state.upload_id.clone()));
    m.insert("data_hash".into(), Value::String(state.data_hash.clone()));
    m.insert("param_hash".into(), Value::String(state.param_hash.clone()));
    m.insert(
        "protocol_version".into(),
        Value::String(PROTOCOL_VERSION.into()),
    );
    m.insert(
        "hardware".into(),
        serde_json::to_value(&config.hardware).unwrap_or(Value::Null),
    );
    m.insert(
        "git_branch".into(),
        Value::String(config.git_branch.clone()),
    );
    m
}

/// Insert `progress` + `info` when the build is running, or when the
/// solver is producing frames with a known `total_frames`. The build
/// worker forwards `total_frames` once after `make()`; until then we
/// leave both fields out so the addon falls back to its default
/// "Simulation Running" label without a stale 0% bar.
pub fn insert_progress(m: &mut Map<String, Value>, state: &ServerState) {
    if state.build == Build::Building {
        m.insert("progress".into(), json!(state.build_progress));
        m.insert("info".into(), Value::String(state.build_info.clone()));
    } else if matches!(state.solver, Solver::Running | Solver::Saving) && state.total_frames > 0 {
        let frame = state.frame.max(0).min(state.total_frames);
        let progress = (frame as f64) / (state.total_frames as f64);
        m.insert("progress".into(), json!(progress));
        m.insert(
            "info".into(),
            Value::String(format!(
                "Simulation Running, frame {}/{}",
                frame, state.total_frames
            )),
        );
    }
}

/// `state.violations` is `Vec<String>` by design (opaque payload).
/// When entries are valid JSON strings (the monitor encodes them that
/// way), parse them so the response re-serializes as nested values.
/// Plain strings pass through as-is.
fn violations_to_json(items: &[String]) -> Vec<Value> {
    items
        .iter()
        .map(|s| serde_json::from_str::<Value>(s).unwrap_or_else(|_| Value::String(s.clone())))
        .collect()
}
