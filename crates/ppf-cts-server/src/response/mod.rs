// File: crates/ppf-cts-server/src/response/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Status-response JSON builder. Every status poll from a client funnels
// through here, so the field set is the protocol contract.

use std::path::PathBuf;

use serde_json::Value;

use crate::config::EngineConfig;
use ppf_cts_core::state::{ServerState, Solver};

mod scene_info;
mod shape;
mod summary;

/// Build the full status response value.
///
/// The base map (status / data / frame / initialized / error /
/// violations / root / upload_id / hashes / protocol_version /
/// hardware / git_branch) is always populated. Optional fields:
///   * `progress` + `info` while building, or while the solver is
///     running with a known `total_frames`.
///   * `scene_info` when `<root>/scene_info.json` exists.
///   * `summary` while the solver is `Running` / `Saving`.
///   * `average_summary` when the solver is idle and log files exist.
pub fn build_response(state: &ServerState, config: &EngineConfig) -> Value {
    let mut m = shape::base_map(state, config);
    shape::insert_progress(&mut m, state);

    if !state.root.is_empty() {
        if let Some(v) = scene_info::load_scene_info(&state.root) {
            m.insert("scene_info".into(), v);
        }

        let info_path = PathBuf::from(&state.root).join("session");
        match state.solver {
            Solver::Running | Solver::Saving => {
                let live = summary::build_live_summary(
                    state.frame,
                    &info_path,
                    &summary::LIVE_SUMMARY_CHANNELS,
                );
                m.insert("summary".into(), Value::Object(live));
            }
            _ => {
                let pairs = summary::average_summary_pairs(&config.log_filenames);
                let avg =
                    summary::build_average_summary(&info_path.join("output").join("data"), &pairs);
                if !avg.is_empty() {
                    m.insert("average_summary".into(), Value::Object(avg));
                }
            }
        }
    }

    Value::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use ppf_cts_core::state::{Build, Data, ServerState, Solver};

    #[test]
    fn ready_state_response_has_required_fields() {
        let s = ServerState {
            name: "demo".into(),
            root: "/tmp/demo".into(),
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Idle,
            upload_id: "abc123abc123".into(),
            data_hash: "deadbeef".into(),
            ..Default::default()
        };
        let cfg = EngineConfig::default();
        let r = build_response(&s, &cfg);
        let m = r.as_object().unwrap();
        assert_eq!(m["status"], "READY");
        assert_eq!(m["data"], "READY");
        assert_eq!(m["upload_id"], "abc123abc123");
        assert_eq!(m["protocol_version"], "0.03");
        assert!(m.contains_key("hardware"));
        // No build progress when not building.
        assert!(!m.contains_key("progress"));
    }

    #[test]
    fn building_state_includes_progress() {
        let s = ServerState {
            data: Data::Uploaded,
            build: Build::Building,
            build_progress: 0.42,
            build_info: "Encoding meshes...".into(),
            ..Default::default()
        };
        let r = build_response(&s, &EngineConfig::default());
        let m = r.as_object().unwrap();
        assert_eq!(m["status"], "BUILDING");
        assert_eq!(m["progress"], 0.42);
        assert_eq!(m["info"], "Encoding meshes...");
    }

    #[test]
    fn running_state_includes_solver_progress_when_total_frames_known() {
        let s = ServerState {
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Running,
            frame: 30,
            total_frames: 120,
            ..Default::default()
        };
        let r = build_response(&s, &EngineConfig::default());
        let m = r.as_object().unwrap();
        assert_eq!(m["status"], "BUSY");
        assert_eq!(m["progress"], 0.25);
        assert_eq!(m["info"], "Simulation Running, frame 30/120");
    }

    #[test]
    fn response_carries_initialized_flag() {
        let s = ServerState {
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Running,
            frame: 0,
            initialized: true,
            ..Default::default()
        };
        let r = build_response(&s, &EngineConfig::default());
        let m = r.as_object().unwrap();
        assert_eq!(m["status"], "BUSY");
        assert_eq!(m["frame"], 0);
        assert_eq!(
            m["initialized"], true,
            "addon needs this flag to leave 'Initializing' before frame 1 lands",
        );

        let s_pre = ServerState {
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Running,
            frame: 0,
            initialized: false,
            ..Default::default()
        };
        let r_pre = build_response(&s_pre, &EngineConfig::default());
        assert_eq!(r_pre["initialized"], false);
    }

    #[test]
    fn running_state_omits_progress_when_total_frames_unknown() {
        let s = ServerState {
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Running,
            frame: 30,
            total_frames: 0,
            ..Default::default()
        };
        let r = build_response(&s, &EngineConfig::default());
        let m = r.as_object().unwrap();
        assert_eq!(m["status"], "BUSY");
        assert!(!m.contains_key("progress"));
        assert!(!m.contains_key("info"));
    }

    #[test]
    fn violations_parse_back_into_nested_json() {
        let s = ServerState {
            violations: vec![r#"{"type":"tri-tri","elem0":1,"elem1":2}"#.into()],
            ..Default::default()
        };
        let r = build_response(&s, &EngineConfig::default());
        let arr = r["violations"].as_array().unwrap();
        assert_eq!(arr[0]["type"], "tri-tri");
        assert_eq!(arr[0]["elem0"], 1);
    }

    #[test]
    fn running_state_emits_live_summary_with_frame_seed() {
        let dir = tempfile::tempdir().unwrap();
        let s = ServerState {
            root: dir.path().to_string_lossy().into_owned(),
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Running,
            frame: 7,
            total_frames: 120,
            ..Default::default()
        };
        // Empty data dir, but the summary is still emitted with the
        // `frame` seed and the formatted (N/A) log fields, so the
        // addon's "Realtime Statistics" panel has rows to render.
        // `cfg.log_filenames` is intentionally bogus here: the live
        // summary uses the pinned `LIVE_SUMMARY_CHANNELS` table and
        // ignores the harvested map for its six channels.
        let cfg = EngineConfig {
            log_filenames: vec![("pcg-iter".into(), "wrong.out".into())],
            ..Default::default()
        };
        let r = build_response(&s, &cfg);
        let summary = r["summary"].as_object().expect("summary present");
        assert_eq!(summary["frame"], "7");
        assert!(summary.contains_key("time-per-frame"));
        assert!(summary.contains_key("num-contact"));
        assert!(summary.contains_key("pcg-iter"));
        // Channels with no `.out` file backing render as "N/A".
        assert_eq!(summary["time-per-frame"], "N/A");
        assert_eq!(summary["num-contact"], "N/A");
        assert_eq!(summary["pcg-iter"], "N/A");
        // No `average_summary` while running.
        assert!(!r.as_object().unwrap().contains_key("average_summary"));
    }

    #[test]
    fn live_summary_reads_pinned_files_regardless_of_harvest() {
        // Smoke test for the regression that motivated
        // `LIVE_SUMMARY_CHANNELS`: even when the harvester points
        // `pcg-iter` at a non-existent path, the live summary should
        // still resolve the value from the canonical
        // `advance.iter.out` file the CUDA solver actually writes.
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("session").join("output").join("data");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("advance.iter.out"), "0 7\n1 9\n").unwrap();
        let s = ServerState {
            root: dir.path().to_string_lossy().into_owned(),
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Running,
            frame: 1,
            ..Default::default()
        };
        let cfg = EngineConfig {
            log_filenames: vec![("pcg-iter".into(), "solver.pcg_iter.out".into())],
            ..Default::default()
        };
        let r = build_response(&s, &cfg);
        assert_eq!(r["summary"]["pcg-iter"], "9");
    }

    #[test]
    fn running_state_emits_live_summary_even_when_log_filenames_empty() {
        // The `summary` field appears unconditionally during a sim,
        // so the addon's Realtime Statistics panel doesn't disappear
        // when the src-tree harvest at startup found no log channels
        // (e.g., the binary is running from a deployed bundle with
        // no src/ tree).
        let dir = tempfile::tempdir().unwrap();
        let s = ServerState {
            root: dir.path().to_string_lossy().into_owned(),
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Running,
            frame: 3,
            ..Default::default()
        };
        let cfg = EngineConfig::default(); // log_filenames: vec![]
        let r = build_response(&s, &cfg);
        let summary = r["summary"].as_object().expect("summary present");
        assert_eq!(summary["frame"], "3");
        assert_eq!(summary["time-per-frame"], "N/A");
        assert_eq!(summary["time-per-step"], "N/A");
        assert_eq!(summary["num-contact"], "N/A");
        assert_eq!(summary["newton-steps"], "N/A");
        assert_eq!(summary["pcg-iter"], "N/A");
        // `stretch` is omitted (max-sigma not readable). Same for the
        // GPU-utilization rows when no nvidia-smi: those keys are
        // simply absent.
        assert!(!summary.contains_key("stretch"));
    }

    #[test]
    fn idle_state_emits_average_summary_when_log_files_exist() {
        let dir = tempfile::tempdir().unwrap();
        // executor::solver writes logs at `<root>/session/output/data/`.
        let data = dir.path().join("session").join("output").join("data");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("tpf.out"), "0 1000\n1 2000\n").unwrap();
        std::fs::write(data.join("nc.out"), "0 100\n1 500\n").unwrap();

        let s = ServerState {
            root: dir.path().to_string_lossy().into_owned(),
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Idle,
            ..Default::default()
        };
        let cfg = EngineConfig {
            log_filenames: vec![
                ("time-per-frame".into(), "tpf.out".into()),
                ("num-contact".into(), "nc.out".into()),
            ],
            ..Default::default()
        };
        let r = build_response(&s, &cfg);
        let avg = r["average_summary"].as_object().expect("average_summary present");
        assert!(avg.contains_key("time-per-frame"));
        // No live `summary` when idle.
        assert!(!r.as_object().unwrap().contains_key("summary"));
    }

    #[test]
    fn idle_state_omits_average_summary_when_no_log_files() {
        let dir = tempfile::tempdir().unwrap();
        let s = ServerState {
            root: dir.path().to_string_lossy().into_owned(),
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Idle,
            ..Default::default()
        };
        let cfg = EngineConfig {
            log_filenames: vec![("time-per-frame".into(), "tpf.out".into())],
            ..Default::default()
        };
        let r = build_response(&s, &cfg);
        let m = r.as_object().unwrap();
        assert!(!m.contains_key("summary"));
        assert!(!m.contains_key("average_summary"));
    }

    // The error-only response invariants are exercised in
    // `error::tests::text_decode_renders_full_tcmd_shape`.

    #[test]
    fn scene_info_merges_simulated_frames_and_last_saved() {
        let dir = tempfile::tempdir().unwrap();
        // build_worker writes `<root>/scene_info.json`.
        let static_info = serde_json::json!({
            "Vertices": "100",
            "Total Frames": "120",
            "FPS": "60",
        });
        std::fs::write(
            dir.path().join("scene_info.json"),
            serde_json::to_string(&static_info).unwrap(),
        )
        .unwrap();
        // The solver writes `vert_*.bin` / `save_*.bin` under
        // `<root>/session/output/`.
        let output = dir.path().join("session").join("output");
        std::fs::create_dir_all(&output).unwrap();
        std::fs::write(output.join("vert_0.bin"), b"").unwrap();
        std::fs::write(output.join("vert_1.bin"), b"").unwrap();
        std::fs::write(output.join("vert_2.bin"), b"").unwrap();
        std::fs::write(output.join("save_5.bin"), b"").unwrap();
        std::fs::write(output.join("save_12.bin"), b"").unwrap();

        let s = ServerState {
            root: dir.path().to_string_lossy().into_owned(),
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Idle,
            ..Default::default()
        };
        let r = build_response(&s, &EngineConfig::default());
        let info = r["scene_info"].as_object().expect("scene_info present");
        assert_eq!(info["Vertices"], "100");
        assert_eq!(info["Total Frames"], "120");
        assert_eq!(info["FPS"], "60");
        assert_eq!(info["Simulated Frames"], "3");
        assert_eq!(info["Last Saved"], "12");
    }

    #[test]
    fn scene_info_last_saved_none_when_no_save_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("scene_info.json"), r#"{"Vertices":"5"}"#).unwrap();
        std::fs::create_dir_all(dir.path().join("session").join("output")).unwrap();
        let s = ServerState {
            root: dir.path().to_string_lossy().into_owned(),
            data: Data::Uploaded,
            build: Build::Built,
            solver: Solver::Idle,
            ..Default::default()
        };
        let r = build_response(&s, &EngineConfig::default());
        let info = r["scene_info"].as_object().expect("scene_info present");
        assert_eq!(info["Simulated Frames"], "0");
        assert_eq!(info["Last Saved"], "None");
    }
}
