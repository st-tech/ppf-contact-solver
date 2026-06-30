// File: crates/ppf-cts-formats/src/files.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! On-disk filename constants for the per-project layout.
//!
//! The Rust server, the Rust solver, the PyO3 frontend, and the
//! Blender addon all read and write the same set of files under a
//! project root. Centralizing the names here keeps the four consumers
//! from drifting; a typo in any one consumer would otherwise surface
//! only as a silent "file not found" at the next runtime poll.
//!
//! Python consumers (the Blender addon, the frontend `_decoder_`)
//! still carry their own copies; the Python literals are deliberately
//! kept in sync by hand because there is no PyO3 binding for raw
//! string constants and a runtime fetch would be more brittle than a
//! cross-reference comment.

use std::path::{Path, PathBuf};

/// Per-project session directory under the project root.
pub const SESSION_DIR: &str = "session";

/// Solver output directory under the session dir; holds the per-frame
/// vertex bins, state checkpoints, finish/sentinel markers, and the
/// intersection records.
pub const OUTPUT_DIR: &str = "output";

/// SimpleLog data directory under the output dir; holds the per-channel
/// `*.out` numeric logs the average-summary builder reads.
pub const DATA_DIR: &str = "data";

/// Solver stdout capture inside the session dir.
pub const STDOUT_LOG: &str = "stdout.log";

/// Solver stderr capture inside the session dir.
pub const ERROR_LOG: &str = "error.log";

/// Per-project run script on Unix, handed to bash by the server.
pub const COMMAND_SH: &str = "command.sh";

/// Per-project run script on Windows, invoked via cmd /C.
pub const COMMAND_BAT: &str = "command.bat";

/// Solver-written record of self-intersections, read by the monitor
/// when a run crashes.
pub const INTERSECTION_RECORDS_JSON: &str = "intersection_records.json";

/// `<root>/session`, the per-project session directory.
pub fn session_dir(root: &Path) -> PathBuf {
    root.join(SESSION_DIR)
}

/// `<root>/session/output`, the canonical solver output directory
/// (per-frame vertex bins, state checkpoints, finish/sentinel markers,
/// intersection records). Dedupes the very common two-join chain.
pub fn session_output_dir(root: &Path) -> PathBuf {
    session_dir(root).join(OUTPUT_DIR)
}

/// `<root>/session/output/data`, where the solver writes its
/// per-channel SimpleLog `*.out` files. Dedupes the three-join chain
/// the average-summary path uses.
pub fn session_data_dir(root: &Path) -> PathBuf {
    session_output_dir(root).join(DATA_DIR)
}

/// CBOR-enveloped scene payload uploaded by the addon (filename kept as
/// data.pickle for back-compat with existing project layouts).
pub const DATA_PICKLE: &str = "data.pickle";

/// CBOR-enveloped parameter set uploaded alongside the scene (filename
/// kept as param.pickle for back-compat).
pub const PARAM_PICKLE: &str = "param.pickle";

/// 12-hex upload identity stamped at upload time.
pub const UPLOAD_ID_FILE: &str = "upload_id.txt";

/// Quick fingerprint of the last data.pickle write.
pub const DATA_HASH_FILE: &str = "data_hash.txt";

/// Quick fingerprint of the last param.pickle write.
pub const PARAM_HASH_FILE: &str = "param_hash.txt";

/// Per-object vertex-index map produced by the build worker.
pub const MAP_PICKLE: &str = "map.pickle";

/// Frame-embedding surface map for tetrahedralized SOLID groups.
pub const SURFACE_MAP_PICKLE: &str = "surface_map.pickle";

/// Static scene metadata (Vertices / Triangles / Tetrahedra / FPS / ...).
pub const SCENE_INFO_JSON: &str = "scene_info.json";

/// Touched by the solver immediately after `initialize()` returns true.
/// The server monitor watches for it to flip the `initialized` flag.
pub const INITIALIZE_FINISH: &str = "initialize_finish.txt";

/// Touched by the solver after the per-frame loop exits cleanly.
pub const FINISHED: &str = "finished.txt";

/// Sentinel the addon writes to request `save_and_quit` mid-run.
pub const SAVE_AND_QUIT: &str = "save_and_quit";

/// Structured run-status record written by the solver host (CBOR,
/// `Envelope<RunStatus>`; see [`crate::status`]). The single source of
/// truth for solver lifecycle and outcome, replacing the free-form log
/// scraping and the overlapping `finished.txt`/`crashed.txt` markers.
pub const STATUS_RECORD: &str = "status.cbor";

/// Liveness lock the solver host advisory-locks for its whole lifetime.
/// The OS releases it on ANY process death (including SIGKILL), so a
/// free lock plus a dead owning PID plus no terminal outcome is a crash
/// by construction. See [`crate::status::lock`].
pub const STATUS_LOCK: &str = "status.lock";

/// Written by the server before it kills the solver, and interpreted by
/// the monitor (not the dying host, which on Windows cannot stamp
/// anything) so an intentional Terminate is never misread as a crash and
/// survives a server restart in the kill-then-observe window.
pub const TERMINATE_REQUEST: &str = "terminate_request";

/// Leading fragment of a per-frame vertex bin name (`vert_<N>.bin`).
/// Used directly by the prefix-based sites (`strip_prefix`,
/// `remove_old_files`) that need the fragment rather than a finished
/// filename.
pub const VERT_PREFIX: &str = "vert_";

/// Trailing fragment of a per-frame vertex bin name.
pub const VERT_SUFFIX: &str = ".bin";

/// Leading fragment of a per-frame state checkpoint name
/// (`state_<N>.bin.gz`).
pub const STATE_PREFIX: &str = "state_";

/// Trailing fragment of a per-frame state checkpoint name.
pub const STATE_SUFFIX: &str = ".bin.gz";

/// Per-frame vertex bin written by the solver as `vert_<N>.bin`.
/// Defined in terms of `VERT_PREFIX`/`VERT_SUFFIX` so the fragment and
/// full-name forms cannot drift.
pub fn vert_filename(frame: i32) -> String {
    format!("{VERT_PREFIX}{frame}{VERT_SUFFIX}")
}

/// Per-frame state checkpoint written by the solver as `state_<N>.bin.gz`.
/// Defined in terms of `STATE_PREFIX`/`STATE_SUFFIX` so the fragment and
/// full-name forms cannot drift.
pub fn state_filename(frame: i32) -> String {
    format!("{STATE_PREFIX}{frame}{STATE_SUFFIX}")
}
