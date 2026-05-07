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

/// Pickled scene payload uploaded by the addon.
pub const DATA_PICKLE: &str = "data.pickle";

/// Pickled parameter set uploaded alongside the scene.
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

/// Per-frame vertex bin written by the solver as `vert_<N>.bin`.
pub fn vert_filename(frame: i32) -> String {
    format!("vert_{frame}.bin")
}

/// Per-frame state checkpoint written by the solver as `state_<N>.bin.gz`.
pub fn state_filename(frame: i32) -> String {
    format!("state_{frame}.bin.gz")
}
