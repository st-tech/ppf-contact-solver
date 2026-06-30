// File: crates/ppf-cts-core/src/datamodel/session/tests.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Inline tests for the session sub-modules. These were moved from the
// monolithic `session.rs` and continue to exercise the same functions
// via the re-exports in `super`.

use super::*;
use std::path::{Path, PathBuf};

use crate::datamodel::param_manager::ParamManager;

fn write_text(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, body).unwrap();
}

#[test]
fn session_info_holds_name_and_path() {
    let info = SessionInfo::new("demo", "/tmp/demo");
    assert_eq!(info.name, "demo");
    assert_eq!(info.path, PathBuf::from("/tmp/demo"));
}

#[test]
fn session_output_appends_output_subdir() {
    let out = SessionOutput::new("/tmp/demo");
    assert_eq!(out.path, PathBuf::from("/tmp/demo/output"));
}

#[test]
fn session_carries_paths_and_default_param() {
    let s = Session::new("app", "demo", "/tmp/app", "/tmp/proj", "/tmp/data");
    assert_eq!(s.app_name, "app");
    assert_eq!(s.name, "demo");
    assert_eq!(s.proj_root, PathBuf::from("/tmp/proj"));
    // Default ParamManager has frames=300.
    assert!(matches!(
        s.param.get("frames").unwrap(),
        crate::datamodel::params::ParamValue::Int(300)
    ));
}

#[test]
fn fixed_session_path_joins_app_root_and_name() {
    let s = Session::new("app", "demo", "/tmp/app", "/tmp/proj", "/tmp/data");
    let fs = FixedSession::from_session(&s);
    assert_eq!(fs.info.name, "demo");
    assert_eq!(fs.info.path, PathBuf::from("/tmp/app/demo"));
    assert_eq!(fs.output.path, PathBuf::from("/tmp/app/demo/output"));
}

#[test]
fn fixed_session_param_snapshot_is_independent() {
    let mut s = Session::new("app", "demo", "/tmp/app", "/tmp/proj", "/tmp/data");
    s.param
        .set("dt", Some(crate::datamodel::params::ParamValue::Float(0.005)))
        .unwrap();

    let fs = FixedSession::from_session(&s);
    // Mutating the originating session shouldn't change the snapshot.
    s.param
        .set("dt", Some(crate::datamodel::params::ParamValue::Float(0.01)))
        .unwrap();
    assert!(matches!(
        fs.param.get("dt").unwrap(),
        crate::datamodel::params::ParamValue::Float(v) if (*v - 0.005).abs() < 1e-15
    ));
}

// PathBuf::join uses the host OS separator (``/`` on POSIX, ``\`` on
// Windows). The shell-script templates pass paths through Path::display
// which preserves whatever separator PathBuf produced, so any test
// asserting the exact path content is implicitly host-OS-specific.
// Gate the Unix-host assertions with #[cfg(unix)] and the Windows-host
// assertions with #[cfg(windows)] so ``cargo test`` works on every host;
// each branch's template is still exercised on its native CI runner.

#[cfg(unix)]
#[test]
fn unix_script_has_shebang_and_path_args() {
    let s = Session::new("app", "demo", "/tmp/app", "/tmp/proj", "/tmp/data");
    let fs = FixedSession::from_session(&s);
    let script = shell_command_script(&s, &fs, Platform::Unix);
    assert!(script.starts_with("#!/bin/bash"));
    assert!(script.contains("/tmp/proj/target/release/ppf-contact-solver"));
    // Path values are double-quoted so spaces in project root don't
    // get word-split by the shell before reaching clap.
    assert!(script.contains(r#"--path "/tmp/app/demo""#));
    assert!(script.contains(r#"--output "/tmp/app/demo/output""#));
    // Variadic forwarding for extra solver args.
    assert!(script.contains("\"$@\""));
}

#[cfg(unix)]
#[test]
fn unix_script_quotes_paths_with_spaces() {
    let s = Session::new("My App", "demo", "/tmp/has space", "/tmp/has space/proj", "/tmp/has space/data");
    let fs = FixedSession::from_session(&s);
    let script = shell_command_script(&s, &fs, Platform::Unix);
    // Both path args must reach the solver as a single argv entry.
    assert!(script.contains(r#"--path "/tmp/has space/demo""#));
    assert!(script.contains(r#"--output "/tmp/has space/demo/output""#));
}

#[cfg(windows)]
#[test]
fn windows_script_quotes_paths_with_spaces() {
    let s = Session::new(
        "My App", "demo",
        "C:\\New Folder\\app",
        "C:\\New Folder\\proj",
        "C:\\New Folder\\data",
    );
    let fs = FixedSession::from_session(&s);
    let script = shell_command_script(&s, &fs, Platform::Windows);
    // Path values are wrapped in double quotes so cmd doesn't split
    // "C:\New Folder\app\demo" on the embedded space.
    assert!(script.contains(r#"--path "C:\New Folder\app\demo""#));
    assert!(script.contains(r#"--output "C:\New Folder\app\demo\output""#));
}

#[test]
fn windows_script_has_batch_header_and_dll_path() {
    let s = Session::new(
        "app",
        "demo",
        "C:\\app",
        "C:\\proj",
        "C:\\data",
    );
    let fs = FixedSession::from_session(&s);
    let script = shell_command_script(&s, &fs, Platform::Windows);
    assert!(script.starts_with("@echo off"));
    assert!(script.contains("ppf-contact-solver.exe"));
    assert!(script.contains("LIB_PATH"));
    assert!(script.contains("CUDA_PATH"));
    // Variadic forwarding for batch.
    assert!(script.contains("%*"));
}

#[test]
fn read_log_tail_returns_empty_for_missing_file() {
    let missing = std::path::Path::new("/tmp/no-such-log-{}");
    assert!(read_log_tail(missing, None).is_empty());
    assert!(read_log_tail(missing, Some(10)).is_empty());
}

#[test]
fn read_log_tail_returns_last_n_lines() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("stdout.log");
    write_text(&p, "a\nb\nc\nd\ne\n");
    let all = read_log_tail(&p, None);
    assert_eq!(all, vec!["a", "b", "c", "d", "e"]);
    let tail3 = read_log_tail(&p, Some(3));
    assert_eq!(tail3, vec!["c", "d", "e"]);
    // n > line count returns all.
    let tail99 = read_log_tail(&p, Some(99));
    assert_eq!(tail99, vec!["a", "b", "c", "d", "e"]);
    // n == 0 mirrors Python `lines[-0:]` (== whole list).
    let tail0 = read_log_tail(&p, Some(0));
    assert_eq!(tail0, vec!["a", "b", "c", "d", "e"]);
}

#[test]
fn read_log_numbers_parses_pairs() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("data/time-per-frame.out");
    write_text(&p, "0 0.1\n1 0.15\n2 0.13\n");
    let pairs = read_log_numbers(&p);
    assert_eq!(pairs.len(), 3);
    assert_eq!(pairs[0], (0.0, 0.1));
    assert_eq!(pairs[1], (1.0, 0.15));
    assert_eq!(pairs[2], (2.0, 0.13));
}

#[test]
fn read_log_numbers_skips_malformed_lines() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("data.out");
    write_text(&p, "0 0.1\nbroken\n2 0.2\n# comment\n");
    let pairs = read_log_numbers(&p);
    // The two well-formed lines survive; "broken" + "# comment"
    // are dropped.
    assert_eq!(pairs.len(), 2);
}

#[test]
fn latest_log_value_returns_last_y() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("metric.out");
    write_text(&p, "0 1.0\n1 2.0\n2 5.0\n");
    assert_eq!(latest_log_value(&p), Some(5.0));
}

#[test]
fn latest_log_value_none_for_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("empty.out");
    write_text(&p, "");
    assert_eq!(latest_log_value(&p), None);
}

#[test]
fn latest_step_average_averages_trailing_same_timestamp_rows() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("metric.out");
    // Step t=0 had one iteration; step t=1 had three. The latest step is
    // t=1, so the average is over (2, 4, 6) = 4.0, ignoring the t=0 row.
    write_text(&p, "0 99\n1 2\n1 4\n1 6\n");
    assert_eq!(latest_step_average(&p), Some(4.0));
}

#[test]
fn latest_step_average_single_row_step() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("metric.out");
    // A step with one iteration averages to that one value.
    write_text(&p, "0 7\n1 9\n");
    assert_eq!(latest_step_average(&p), Some(9.0));
}

#[test]
fn latest_step_average_none_for_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("empty.out");
    write_text(&p, "");
    assert_eq!(latest_step_average(&p), None);
}

#[test]
fn current_platform_dispatches_correctly() {
    // Only check that filenames are right; the actual current()
    // value depends on cfg.
    let p = Platform::current();
    match p {
        Platform::Windows => assert_eq!(p.script_filename(), "command.bat"),
        Platform::Unix => assert_eq!(p.script_filename(), "command.sh"),
    }
}

#[test]
fn list_vertex_frames_sees_all_indices() {
    let dir = tempfile::tempdir().unwrap();
    write_text(&dir.path().join("vert_0.bin"), "");
    write_text(&dir.path().join("vert_2.bin"), "");
    write_text(&dir.path().join("vert_10.bin"), "");
    // Decoy files that must be skipped.
    write_text(&dir.path().join("state_3.bin.gz"), "");
    write_text(&dir.path().join("vert_oops.bin"), "");
    write_text(&dir.path().join("vert_4.dat"), "");
    let mut frames = list_vertex_frames(dir.path());
    frames.sort();
    assert_eq!(frames, vec![0, 2, 10]);
}

#[test]
fn latest_vertex_frame_picks_max() {
    let dir = tempfile::tempdir().unwrap();
    write_text(&dir.path().join("vert_0.bin"), "");
    write_text(&dir.path().join("vert_2.bin"), "");
    write_text(&dir.path().join("vert_10.bin"), "");
    assert_eq!(latest_vertex_frame(dir.path()), 10);
}

#[test]
fn latest_vertex_frame_zero_when_dir_missing() {
    assert_eq!(latest_vertex_frame(Path::new("/no/such/dir-cts-test")), 0);
}

#[test]
fn list_saved_states_pulls_state_files_only() {
    let dir = tempfile::tempdir().unwrap();
    write_text(&dir.path().join("state_5.bin.gz"), "");
    write_text(&dir.path().join("state_120.bin.gz"), "");
    write_text(&dir.path().join("vert_5.bin"), "");
    let mut got = list_saved_states(dir.path());
    got.sort();
    assert_eq!(got, vec![5, 120]);
}

#[test]
fn read_vertex_bin_roundtrips_floats() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vert_7.bin");
    let verts: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut bytes = Vec::with_capacity(24);
    for v in verts {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&path, &bytes).unwrap();
    let got = read_vertex_bin(dir.path(), 7).unwrap();
    assert_eq!(got, verts);
}

#[test]
fn read_vertex_bin_rejects_misaligned_buffers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vert_3.bin");
    std::fs::write(&path, [0u8; 7]).unwrap();
    assert!(read_vertex_bin(dir.path(), 3).is_none());
}

#[test]
fn read_vertex_bin_returns_none_when_missing() {
    let dir = tempfile::tempdir().unwrap();
    assert!(read_vertex_bin(dir.path(), 99).is_none());
}

#[test]
fn analyze_solver_error_matches_known_patterns() {
    let log = vec!["clean".to_string()];
    let err = vec!["### CCD failed at frame 7".to_string()];
    assert_eq!(
        analyze_solver_error(&log, &err).as_deref(),
        Some("Continuous Collision Detection failed"),
    );
}

#[test]
fn analyze_solver_error_returns_panic_context() {
    let log: Vec<String> = (0..10).map(|i| format!("line {i}")).collect();
    let mut err = vec![];
    err.push("thread 'main' panicked at solver.rs:42".to_string());
    err.push("note: trailing".to_string());
    let got = analyze_solver_error(&log, &err).unwrap();
    // Panic line + 3 before + up to 3 after, joined by newlines.
    assert!(got.contains("panicked at"));
    assert!(got.contains("line 9"));
    assert!(got.contains("note: trailing"));
}

#[test]
fn analyze_solver_error_returns_none_when_clean() {
    let log = vec!["progress".to_string(), "done".to_string()];
    let err = vec![];
    assert!(analyze_solver_error(&log, &err).is_none());
}

#[test]
fn convert_time_buckets() {
    assert_eq!(convert_time(500.0), "500ms");
    assert_eq!(convert_time(1500.0), "1.50s");
    assert_eq!(convert_time(120_000.0), "2.00m");
}

#[test]
fn convert_integer_buckets() {
    assert_eq!(convert_integer(42.0), "42");
    assert_eq!(convert_integer(1_500.0), "1.50k");
    assert_eq!(convert_integer(2_500_000.0), "2.50M");
    assert_eq!(convert_integer(3_000_000_000.0), "3.00B");
}

#[test]
fn format_log_summary_includes_stretch_only_when_positive() {
    let r = format_log_summary(500.0, 50.0, 1500.0, 10.0, 250.0, Some(1.20));
    assert_eq!(r[0], ("time-per-frame".to_string(), "500ms".to_string()));
    assert_eq!(r[1], ("time-per-step".to_string(), "50ms".to_string()));
    assert_eq!(r[2], ("num-contact".to_string(), "1.50k".to_string()));
    assert_eq!(r[3], ("newton-steps".to_string(), "10".to_string()));
    assert_eq!(r[4], ("pcg-iter".to_string(), "250".to_string()));
    assert_eq!(r[5], ("stretch".to_string(), "20.00%".to_string()));

    let r2 = format_log_summary(500.0, 50.0, 1500.0, 10.0, 250.0, None);
    assert_eq!(r2.len(), 5);
    let r3 = format_log_summary(500.0, 50.0, 1500.0, 10.0, 250.0, Some(0.0));
    assert_eq!(r3.len(), 5);
}

#[test]
fn format_log_average_summary_skips_none_metrics() {
    let r = format_log_average_summary(
        Some(1500.0), None, Some(2000.0), Some(5.5), None, Some(1.10),
        Some(12.0), None, Some(0.9), Some(0.42), Some(7.0), Some(0.85),
    );
    let map: std::collections::HashMap<_, _> = r.into_iter().collect();
    assert_eq!(map["time-per-frame"], "1.50s");
    assert_eq!(map["num-contact (max)"], "2.00k");
    assert_eq!(map["newton-steps"], "5.50");
    assert_eq!(map["stretch"], "10.00%");
    assert_eq!(map["matrix-assembly"], "12ms");
    assert_eq!(map["toi-advanced"], "90.00%");
    assert_eq!(map["dyn-consumed (max)"], "42.00%");
    assert_eq!(map["line-search"], "7ms");
    assert_eq!(map["toi"], "85.00%");
    assert!(!map.contains_key("time-per-step"));
    assert!(!map.contains_key("pcg-iter"));
    // pcg_linsolve_ms_avg was None, so the row is omitted.
    assert!(!map.contains_key("pcg-linsolve"));
}

#[test]
fn average_summary_includes_matrix_assembly_and_pcg_time() {
    let dir = tempfile::tempdir().unwrap();
    write_text(&dir.path().join("advance.matrix_assembly.out"), "0 8\n0 12\n");
    write_text(&dir.path().join("advance.linsolve.out"), "0 100\n1 140\n");
    let pairs = [
        ("matrix-assembly", "advance.matrix_assembly.out"),
        ("pcg-linsolve", "advance.linsolve.out"),
    ];
    let got: std::collections::HashMap<_, _> =
        average_summary_from_disk(dir.path(), &pairs).into_iter().collect();
    assert_eq!(got["matrix-assembly"], "10ms"); // avg(8,12)
    assert_eq!(got["pcg-linsolve"], "120ms"); // avg(100,140)
}

#[test]
fn average_summary_reports_dyn_consumed_as_max() {
    // dyn-consumed is shown as the run maximum (like num-contact),
    // formatted as a percentage.
    let dir = tempfile::tempdir().unwrap();
    write_text(&dir.path().join("advance.dyn_consumed.out"), "0 0.1\n1 0.73\n2 0.5\n");
    let pairs = [("dyn-consumed", "advance.dyn_consumed.out")];
    let got: std::collections::HashMap<_, _> =
        average_summary_from_disk(dir.path(), &pairs).into_iter().collect();
    assert_eq!(got["dyn-consumed (max)"], "73.00%"); // max(0.1,0.73,0.5)
}

// -----------------------------------------------------------------
// Disk-side composite helpers added for the
// `read_average_summary_from_disk` port.

#[test]
fn average_summary_from_disk_returns_empty_when_dir_missing() {
    let got = average_summary_from_disk(Path::new("/no/such/cts-dir"), &[]);
    assert!(got.is_empty());
}

#[test]
fn average_summary_from_disk_formats_present_metrics() {
    let dir = tempfile::tempdir().unwrap();
    write_text(&dir.path().join("tpf.out"), "0 1000\n1 2000\n");
    write_text(&dir.path().join("nc.out"), "0 100\n1 500\n");
    write_text(&dir.path().join("ns.out"), "0 4\n1 6\n");

    let log_filenames: &[(&str, &str)] = &[
        ("time-per-frame", "tpf.out"),
        ("num-contact", "nc.out"),
        ("newton-steps", "ns.out"),
        // intentionally missing: time-per-step, pcg-iter, max-sigma
    ];
    let r = average_summary_from_disk(dir.path(), log_filenames);
    let map: std::collections::HashMap<_, _> = r.into_iter().collect();
    // average((1000, 2000)) = 1500ms -> "1.50s"
    assert_eq!(map["time-per-frame"], "1.50s");
    // max(num-contact) = 500 -> "500"
    assert_eq!(map["num-contact (max)"], "500");
    // average((4, 6)) = 5 -> "5.00"
    assert_eq!(map["newton-steps"], "5.00");
    assert!(!map.contains_key("time-per-step"));
    assert!(!map.contains_key("pcg-iter"));
    assert!(!map.contains_key("stretch"));
}

#[test]
fn float_or_int_pair_classifies_components() {
    assert_eq!(float_or_int_pair(0.0, 0.5), (0.0, 0.5, true, false));
    assert_eq!(float_or_int_pair(2.0, 5.0), (2.0, 5.0, true, true));
    assert_eq!(float_or_int_pair(2.5, 5.5), (2.5, 5.5, false, false));
}

#[test]
fn param_summary_lines_returns_empty_when_missing() {
    let dir = tempfile::tempdir().unwrap();
    assert!(param_summary_lines(dir.path()).is_empty());
}

#[test]
fn param_summary_lines_strips_newlines() {
    let dir = tempfile::tempdir().unwrap();
    write_text(&dir.path().join("param_summary.txt"), "key1 = 1\nkey2 = 2\n");
    let lines = param_summary_lines(dir.path());
    assert_eq!(lines, vec!["key1 = 1", "key2 = 2"]);
}

#[test]
fn nvidia_smi_text_handles_missing_files() {
    let dir = tempfile::tempdir().unwrap();
    let text = nvidia_smi_text(dir.path());
    assert!(text.contains("nvidia-smi.txt not found"));
    assert!(text.contains("nvidia-smi-q.txt not found"));
}

#[test]
fn nvidia_smi_text_concatenates_with_separator() {
    let dir = tempfile::tempdir().unwrap();
    write_text(&dir.path().join("nvidia-smi/nvidia-smi.txt"), "main body");
    write_text(&dir.path().join("nvidia-smi/nvidia-smi-q.txt"), "q body");
    let text = nvidia_smi_text(dir.path());
    assert!(text.contains("main body"));
    assert!(text.contains(&"=".repeat(80)));
    assert!(text.contains("q body"));
}

#[test]
fn command_path_returns_some_when_file_exists() {
    let dir = tempfile::tempdir().unwrap();
    write_text(&dir.path().join("command.sh"), "#!/bin/bash");
    let p = command_path(dir.path(), Platform::Unix).unwrap();
    assert!(p.ends_with("command.sh"));
    assert!(command_path(dir.path(), Platform::Windows).is_none());
}

#[test]
fn marker_exists_basic() {
    let dir = tempfile::tempdir().unwrap();
    assert!(!marker_exists(dir.path(), "finished.txt"));
    write_text(&dir.path().join("finished.txt"), "");
    assert!(marker_exists(dir.path(), "finished.txt"));
}

#[test]
fn touch_save_and_quit_creates_sentinel() {
    let dir = tempfile::tempdir().unwrap();
    let p = touch_save_and_quit(dir.path()).unwrap();
    assert_eq!(p, save_and_quit_file_path(dir.path()));
    assert!(p.exists());
}

#[test]
fn is_saving_in_progress_tracks_sentinel_under_project_root() {
    let dir = tempfile::tempdir().unwrap();
    let project_root = dir.path();
    // No sentinel: not saving.
    assert!(!is_saving_in_progress(project_root));
    // Touch the sentinel via the session-relative helper to make
    // sure the wrappers agree on the layout.
    let session_path = project_root.join("session");
    touch_save_and_quit(&session_path).unwrap();
    assert!(is_saving_in_progress(project_root));
}

#[test]
fn project_resumable_when_state_file_exists() {
    let dir = tempfile::tempdir().unwrap();
    let project_root = dir.path();
    let output = project_root.join("session").join("output");
    std::fs::create_dir_all(&output).unwrap();
    // No checkpoint: not resumable.
    assert!(!project_resumable(project_root));
    // Distractors should not flip the flag.
    write_text(&output.join("vert_3.bin"), "");
    write_text(&output.join("state_garbage.bin.gz"), "");
    assert!(!project_resumable(project_root));
    // A real checkpoint flips it.
    write_text(&output.join("state_4.bin.gz"), "");
    assert!(project_resumable(project_root));
}

#[test]
fn delete_session_dir_removes_tree() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("session-x");
    write_text(&target.join("foo/bar.bin"), "data");
    delete_session_dir(&target).unwrap();
    assert!(!target.exists());
    // No-op when path doesn't exist.
    delete_session_dir(&target).unwrap();
}

#[test]
fn delete_session_dir_keep_output_preserves_output_clears_input() {
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("session");
    // Scene input that must be cleared.
    write_text(&session.join("bin/x.bin"), "input");
    write_text(&session.join("param.toml"), "frames = 60");
    // Solver output that must survive.
    write_text(&session.join("output/state_5.bin.gz"), "checkpoint");
    write_text(&session.join("output/save_and_quit"), "");

    delete_session_dir_keep_output(&session).unwrap();

    // The session directory itself remains.
    assert!(session.exists());
    // Scene input is gone.
    assert!(!session.join("bin").exists());
    assert!(!session.join("param.toml").exists());
    // Output subtree (checkpoints + sentinel) survives.
    assert!(session.join("output/state_5.bin.gz").exists());
    assert!(session.join("output/save_and_quit").exists());
}

#[test]
fn delete_session_dir_keep_output_missing_dir_is_noop() {
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("does-not-exist");
    delete_session_dir_keep_output(&session).unwrap();
    assert!(!session.exists());
}

#[test]
fn delete_session_dir_keep_output_no_output_child_clears_all() {
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("session");
    write_text(&session.join("foo/bar.bin"), "data");

    delete_session_dir_keep_output(&session).unwrap();

    // The session directory itself remains, but all children are gone.
    assert!(session.exists());
    assert!(!session.join("foo").exists());
}

#[test]
fn param_export_to_disk_writes_toml_and_dyn() {
    use crate::datamodel::params::ParamValue;
    let dir = tempfile::tempdir().unwrap();
    let mut p = ParamManager::new();
    p.set("frames", Some(ParamValue::Int(60))).unwrap();
    p.inject_dyn_entries(
        "playback",
        vec![(0.0, ParamValue::Float(1.0)), (1.0, ParamValue::Float(0.5))],
    )
    .unwrap();

    param_export_to_disk(dir.path(), &p, false).unwrap();
    let toml = std::fs::read_to_string(dir.path().join("param.toml")).unwrap();
    assert!(toml.contains("frames = 60"));
    let dyn_body = std::fs::read_to_string(dir.path().join("dyn_param.txt")).unwrap();
    assert!(dyn_body.contains("[playback]"));
}

#[test]
fn param_export_to_disk_skips_dyn_file_when_no_overrides() {
    let dir = tempfile::tempdir().unwrap();
    let p = ParamManager::new();
    param_export_to_disk(dir.path(), &p, false).unwrap();
    assert!(dir.path().join("param.toml").exists());
    assert!(!dir.path().join("dyn_param.txt").exists());
}

#[test]
fn autogenerate_session_name_basic() {
    let existing: Vec<String> = vec![];
    let (name, counter) = autogenerate_session_name(&existing, "session");
    assert_eq!(name, "session");
    assert_eq!(counter, 0);

    let existing = vec!["session".to_string()];
    let (name, counter) = autogenerate_session_name(&existing, "session");
    assert_eq!(name, "session-1");
    assert_eq!(counter, 1);

    let existing: Vec<String> = vec!["session".into(), "session-1".into(), "session-2".into()];
    let (name, counter) = autogenerate_session_name(&existing, "session");
    assert_eq!(name, "session-3");
    assert_eq!(counter, 3);
}

#[test]
fn build_symlink_name_uses_app_or_counter() {
    // First (counter=0) gets the bare app name.
    assert_eq!(build_symlink_name("drape", "session", Some(0)), "drape");
    // Subsequent collisions tack on the counter.
    assert_eq!(build_symlink_name("drape", "session-2", Some(2)), "drape-2");
    // Manual name (autogenerated=None) uses the explicit session name.
    assert_eq!(build_symlink_name("drape", "run-A", None), "run-A");
}

#[test]
fn log_filename_path_resolves() {
    let dir = tempfile::tempdir().unwrap();
    let info = dir.path().join("session-x");
    let log_filenames: &[(&str, &str)] = &[("time-per-frame", "tpf.out")];
    let p = log_filename_path(&info, "time-per-frame", log_filenames).unwrap();
    assert_eq!(p, info.join("output").join("data").join("tpf.out"));
    assert!(log_filename_path(&info, "missing", log_filenames).is_none());
}

#[test]
fn log_tail_path_chooses_correct_log() {
    let dir = tempfile::tempdir().unwrap();
    let info = dir.path().join("session-x");
    assert_eq!(log_tail_path(&info, LogStream::Stdout), info.join("stdout.log"));
    assert_eq!(log_tail_path(&info, LogStream::Stderr), info.join("error.log"));
}

#[test]
fn read_log_numbers_squashed_dedup_int_float() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("data.out");
    write_text(&p, "0 1.5\n1 2\n2 3.14\n");
    let got = super::log::read_log_numbers_squashed(&p);
    assert_eq!(got.len(), 3);
    assert_eq!(got[0], (0.0, 1.5, true, false));
    assert_eq!(got[1], (1.0, 2.0, true, true));
    assert_eq!(got[2], (2.0, 3.14, true, false));
}

#[test]
fn fixed_session_dir_layout_paths() {
    let layout = fixed_session_dir_layout("/tmp/app", "demo");
    assert_eq!(layout.session_dir, PathBuf::from("/tmp/app/demo"));
    assert_eq!(
        layout.recoverable_pickle,
        PathBuf::from("/tmp/app/demo/fixed_session.pickle")
    );
}

#[test]
fn symlink_target_path_for_name() {
    let p = symlink_target_path("/data", "drape");
    assert_eq!(p, PathBuf::from("/data/symlinks/drape"));
}

#[test]
fn param_export_to_disk_fast_check_forces_one_frame() {
    use crate::datamodel::params::ParamValue;
    let dir = tempfile::tempdir().unwrap();
    let mut p = ParamManager::new();
    p.set("frames", Some(ParamValue::Int(300))).unwrap();
    param_export_to_disk(dir.path(), &p, true).unwrap();
    let toml = std::fs::read_to_string(dir.path().join("param.toml")).unwrap();
    assert!(toml.contains("frames = 1"));
}

// -----------------------------------------------------------------
// Frontend port: third-wave helpers.

#[test]
fn convert_time_optional_handles_none() {
    assert_eq!(convert_time_optional(None), "N/A");
    assert_eq!(convert_time_optional(Some(500.0)), "500ms");
}

#[test]
fn convert_integer_optional_preserves_small_floats() {
    assert_eq!(convert_integer_optional(None), "N/A");
    // Whole number under 1000.
    assert_eq!(convert_integer_optional(Some(42.0)), "42");
    // Bucket boundary still uses the convert_integer suffix path.
    assert_eq!(convert_integer_optional(Some(2_500.0)), "2.50k");
    // Non-integer under 1000 stays as the decimal form (Python str()).
    assert_eq!(convert_integer_optional(Some(42.5)), "42.5");
}

#[test]
fn convert_average_count_two_decimals_under_thousand() {
    assert_eq!(convert_average_count_optional(None), "N/A");
    // Whole values still show two decimals (no bare "4").
    assert_eq!(convert_average_count_optional(Some(4.0)), "4.00");
    // Long fractional tails are clamped to two decimals.
    assert_eq!(convert_average_count_optional(Some(4.333_333_333)), "4.33");
    assert_eq!(convert_average_count_optional(Some(40.0)), "40.00");
}

#[test]
fn convert_average_count_abbreviates_large_values() {
    // At or above 1000 the count gets a k/M/B suffix instead of a long
    // digit run (12439.3 -> "12.44k" rather than "12439.30").
    assert_eq!(convert_average_count_optional(Some(12_439.3)), "12.44k");
    assert_eq!(convert_average_count(1_000.0), "1.00k");
    assert_eq!(convert_average_count(2_500_000.0), "2.50M");
    // Just under the threshold stays in two-decimal form.
    assert_eq!(convert_average_count(999.5), "999.50");
}

#[test]
fn write_shell_command_script_produces_persisted_file() {
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("session-x");
    let output_path = session_path.join("output");
    std::fs::create_dir_all(&output_path).unwrap();
    let proj_root = dir.path().join("proj");
    let p = write_shell_command_script(
        &session_path,
        &output_path,
        &proj_root,
        Platform::Unix,
    )
    .unwrap();
    assert!(p.ends_with("command.sh"));
    let body = std::fs::read_to_string(&p).unwrap();
    assert!(body.starts_with("#!/bin/bash"));
    assert!(body.contains(session_path.to_str().unwrap()));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755);
    }
}

#[test]
fn solver_subprocess_command_per_platform() {
    let p = std::path::Path::new("/tmp/session/command.sh");
    assert_eq!(
        solver_subprocess_command(p, 0, Platform::Unix),
        r#"bash "/tmp/session/command.sh" --load 0"#
    );
    let p = std::path::Path::new("C:\\session\\command.bat");
    assert_eq!(
        solver_subprocess_command(p, 5, Platform::Windows),
        "\"C:\\session\\command.bat\" --load 5"
    );
}

#[test]
fn solver_subprocess_command_handles_spaces_in_path() {
    let p = std::path::Path::new("/home/user/My Project/session/command.sh");
    assert_eq!(
        solver_subprocess_command(p, 0, Platform::Unix),
        r#"bash "/home/user/My Project/session/command.sh" --load 0"#
    );
    let p = std::path::Path::new("C:\\New Folder\\session\\command.bat");
    assert_eq!(
        solver_subprocess_command(p, 7, Platform::Windows),
        "\"C:\\New Folder\\session\\command.bat\" --load 7"
    );
}

#[test]
fn validate_driver_version_branches() {
    assert!(validate_driver_version(Some(525), 520).is_ok());
    let err = validate_driver_version(Some(515), 520).unwrap_err();
    assert!(err.contains("Driver version is 515"));
    let err = validate_driver_version(None, 520).unwrap_err();
    assert!(err.contains("could not be detected"));
}

#[test]
fn export_base_path_for_joins_segments() {
    let p = export_base_path_for(
        std::path::Path::new("/exports"),
        "drape",
        "session-1",
    );
    assert_eq!(p, PathBuf::from("/exports/drape/session-1"));
}

#[test]
fn select_resume_frame_picks_latest_or_explicit() {
    assert_eq!(select_resume_frame(&[5, 12, 7], -1), Some(12));
    assert_eq!(select_resume_frame(&[], -1), None);
    // Explicit positive is honored even if missing from the saved list.
    assert_eq!(select_resume_frame(&[], 30), Some(30));
    // Zero or negative (other than -1) is invalid.
    assert_eq!(select_resume_frame(&[5], 0), None);
    assert_eq!(select_resume_frame(&[5], -2), None);
}

#[test]
fn max_strain_limit_default_zero_filters_finite() {
    assert_eq!(max_strain_limit_default_zero(&[]), 0.0);
    assert_eq!(max_strain_limit_default_zero(&[0.05, 0.1, 0.02]), 0.1);
    // NaN + inf are dropped by the `is_finite` filter; only 0.03
    // survives.
    assert_eq!(
        max_strain_limit_default_zero(&[f64::NAN, 0.03, f64::INFINITY]),
        0.03,
    );
}

#[test]
fn read_log_tail_joined_strips_trailing_blanks() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("stdout.log");
    write_text(&p, "a\nb\nc\nd\n");
    let s = read_log_tail_joined(&p, 2);
    assert_eq!(s, "c\nd");
    // Missing file -> empty string.
    let s = read_log_tail_joined(&dir.path().join("missing.log"), 5);
    assert_eq!(s, "");
}

#[test]
fn zip_target_path_appends_zip_suffix() {
    let p = zip_target_path(std::path::Path::new("/tmp/export-x"));
    assert_eq!(p, PathBuf::from("/tmp/export-x.zip"));
}

#[test]
fn prepare_zip_target_removes_existing_zip() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("anim");
    let zip = zip_target_path(&target);
    write_text(&zip, "old");
    let got = prepare_zip_target(&target).unwrap();
    assert_eq!(got, zip);
    assert!(!zip.exists());
}

#[test]
fn session_violations_message_joins_with_semicolon() {
    let m = session_violations_message(&[
        "self-intersection at frame 0".to_string(),
        "contact-offset too large".to_string(),
    ]);
    assert_eq!(
        m,
        "Cannot create session: self-intersection at frame 0; contact-offset too large. "
    );
    // Single message: no trailing separator.
    let m = session_violations_message(&["one".to_string()]);
    assert_eq!(m, "Cannot create session: one. ");
}

#[test]
fn rstrip_newlines_drops_trailing_newline_only() {
    let got = rstrip_newlines(&[
        "abc\n".to_string(),
        "def".to_string(),
        "ghi\n\n".to_string(),
        "  trailing space\n".to_string(),
    ]);
    assert_eq!(got, vec!["abc", "def", "ghi", "  trailing space"]);
}

#[test]
fn locate_bundled_ffmpeg_returns_first_match() {
    let dir = tempfile::tempdir().unwrap();
    // No candidates yet.
    assert!(locate_bundled_ffmpeg(dir.path()).is_none());
    // Drop a bundled ffmpeg.exe; should find it.
    let exe = dir.path().join("bin").join("ffmpeg.exe");
    write_text(&exe, "");
    assert_eq!(locate_bundled_ffmpeg(dir.path()), Some(exe));
}

#[test]
fn ffmpeg_video_command_matches_python_template() {
    let cmd = ffmpeg_video_command(
        std::path::Path::new("/usr/bin/ffmpeg"),
        "ply",
        "frame.mp4",
    );
    assert!(cmd.contains("\"/usr/bin/ffmpeg\""));
    assert!(cmd.contains("frame_%d.ply.png"));
    assert!(cmd.contains("-c:v libx264"));
    // vid_name is double-quoted, not bare, so spaces survive the shell.
    assert!(cmd.contains("\"frame.mp4\""));
}

#[test]
fn ffmpeg_video_command_handles_spaces_in_paths() {
    let cmd = ffmpeg_video_command(
        std::path::Path::new("/My Apps/bin/ffmpeg"),
        "ply",
        "out frame.mp4",
    );
    assert!(cmd.contains("\"/My Apps/bin/ffmpeg\""));
    assert!(cmd.contains("\"out frame.mp4\""));
}

#[test]
fn project_root_from_frontend_file_strips_two_levels() {
    let p = project_root_from_frontend_file(std::path::Path::new(
        "/home/user/ppf-contact-solver/frontend/_session_.py",
    ));
    assert_eq!(p, PathBuf::from("/home/user/ppf-contact-solver"));
}

#[test]
fn read_lines_with_newlines_preserves_trailing_newlines() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("log.txt");
    write_text(&p, "a\nb\nc");
    let got = read_lines_with_newlines(&p);
    // Last line lacks a trailing newline: Python's `readlines`
    // returns ["a\n", "b\n", "c"]; we mirror.
    assert_eq!(got, vec!["a\n", "b\n", "c"]);
    // Missing file -> empty.
    let got = read_lines_with_newlines(&dir.path().join("missing.log"));
    assert!(got.is_empty());
}

#[test]
fn solver_failed_short_message_uses_first_five_lines() {
    let lines: Vec<String> = (0..10).map(|i| format!("err{i}\n")).collect();
    let m = solver_failed_short_message(&lines);
    assert_eq!(
        m,
        "Solver failed: err0\nerr1\nerr2\nerr3\nerr4\n",
    );
    // Empty input still produces the prefix.
    assert_eq!(solver_failed_short_message(&[]), "Solver failed: ");
}

#[test]
fn solver_failed_to_start_message_handles_rc() {
    assert_eq!(
        solver_failed_to_start_message(Some(2)),
        "Solver failed to start (rc=2)"
    );
    assert_eq!(
        solver_failed_to_start_message(None),
        "Solver failed to start (rc=None)"
    );
}
