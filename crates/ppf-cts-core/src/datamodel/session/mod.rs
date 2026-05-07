// File: crates/ppf-cts-core/src/datamodel/session/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Session-side data layer + I/O helpers, split into focused
// sub-modules:
//
//   * `types`:   `Session` / `SessionInfo` / `SessionOutput` /
//                `FixedSession` / `Platform`.
//   * `scripts`: solver-launcher script body, ffmpeg invocation,
//                `subprocess.Popen` formatters.
//   * `log`:     log-file readers, `summary()` / `average_summary()`
//                formatters, `analyze_solver_error`.
//   * `format`:  `convert_time` / `convert_integer` and friends.
//   * `frames`:  output-directory frame discovery + resume helpers.
//   * `paths`:   path arithmetic, sentinel files, project-root
//                helpers, `param_export_to_disk`.
//
// The public API lives at `crate::datamodel::session::FOO` exactly
// as before; sub-modules are an organizational detail.

pub mod format;
pub mod frames;
pub mod log;
pub mod paths;
pub mod scripts;
pub mod types;

pub use format::{
    convert_integer, convert_integer_optional, convert_time, convert_time_optional,
    rstrip_newlines, session_violations_message,
};
pub use frames::{
    latest_vertex_frame, list_saved_states, list_vertex_frames, read_vertex_bin,
    select_resume_frame,
};
pub use log::{
    analyze_solver_error, average_summary_from_disk, float_or_int_pair, format_log_average_summary,
    format_log_summary, latest_log_value, log_filename_path, log_tail_path, read_lines_with_newlines,
    read_log_numbers, read_log_tail, read_log_tail_joined, solver_failed_short_message,
    solver_failed_to_start_message, LogStream,
};
// `read_log_average_metrics` and `read_log_numbers_squashed` are
// module-local: the former private to `log` (called by
// `average_summary_from_disk`), the latter test-only.
pub use paths::{
    autogenerate_session_name, build_symlink_name, command_path, delete_session_dir,
    export_base_path_for, fixed_session_dir_layout, is_saving_in_progress, marker_exists,
    max_strain_limit_default_zero, nvidia_smi_text, param_export_to_disk, param_summary_lines,
    prepare_zip_target, project_resumable, project_root_from_frontend_file,
    save_and_quit_file_path, stdout_error_log_paths, symlink_target_path, touch_save_and_quit,
    validate_driver_version, zip_target_path,
};
// `cbor_scalar_keep_indices` and `FixedSessionDirLayout` have no
// callers outside `paths.rs`; reach them via
// `crate::datamodel::session::paths::FOO` if a future caller needs
// them.
pub use scripts::{
    ffmpeg_video_command, locate_bundled_ffmpeg, shell_command_script, solver_subprocess_command,
    write_shell_command_script,
};
pub use types::{FixedSession, Platform, Session, SessionInfo, SessionOutput};

#[cfg(test)]
mod tests;
