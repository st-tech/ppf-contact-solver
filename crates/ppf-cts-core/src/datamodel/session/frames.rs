// File: crates/ppf-cts-core/src/datamodel/session/frames.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Output-directory frame discovery. Mirrors `SessionGet.latest_frame`,
// `SessionGet.vertex_frame_count`, `SessionGet.saved`. These are
// polled every ~0.1s during a live preview / progress bar update,
// so a Rust scan beats Python `os.listdir` + per-name str.split.

use std::path::Path;

fn parse_frame_index(name: &str, prefix: &str, suffix: &str) -> Option<u64> {
    let stem = name.strip_prefix(prefix)?;
    let body = stem.strip_suffix(suffix)?;
    body.parse::<u64>().ok()
}

/// Return every frame index N for which `<output_dir>/vert_<N>.bin`
/// exists. Missing dir returns empty.
pub fn list_vertex_frames(output_dir: &Path) -> Vec<u64> {
    let entries = match std::fs::read_dir(output_dir) {
        Ok(it) => it,
        Err(_) => return Vec::new(),
    };
    let mut frames: Vec<u64> = Vec::new();
    for ent in entries.flatten() {
        let fname = ent.file_name();
        let s = match fname.to_str() {
            Some(s) => s,
            None => continue,
        };
        if !s.starts_with("vert") || !s.ends_with(".bin") {
            continue;
        }
        // Strict `vert_<N>.bin` parse so non-conforming names
        // (e.g. `vertex_X.bin`) are not treated as frames.
        if let Some(n) = parse_frame_index(s, "vert_", ".bin") {
            frames.push(n);
        }
    }
    frames
}

/// Highest exported vertex frame index in `output_dir`, or 0 when none.
pub fn latest_vertex_frame(output_dir: &Path) -> u64 {
    list_vertex_frames(output_dir).into_iter().max().unwrap_or(0)
}

/// Return saved-state frame indices: every N for which
/// `<output_dir>/state_<N>.bin.gz` exists.
pub fn list_saved_states(output_dir: &Path) -> Vec<u64> {
    let entries = match std::fs::read_dir(output_dir) {
        Ok(it) => it,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for ent in entries.flatten() {
        let fname = ent.file_name();
        let s = match fname.to_str() {
            Some(s) => s,
            None => continue,
        };
        if !s.starts_with("state_") || !s.ends_with(".bin.gz") {
            continue;
        }
        if let Some(n) = parse_frame_index(s, "state_", ".bin.gz") {
            out.push(n);
        }
    }
    out
}

/// Read `<output_dir>/vert_<frame>.bin` and return the raw float32
/// vertex bytes. None when the file is missing or the byte count is
/// not a multiple of 12 (3 floats per vertex).
pub fn read_vertex_bin(output_dir: &Path, frame: u64) -> Option<Vec<f32>> {
    let path = output_dir.join(format!("vert_{frame}.bin"));
    let raw = std::fs::read(&path).ok()?;
    if raw.len() % 12 != 0 {
        return None;
    }
    let n = raw.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in raw.chunks_exact(4) {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(f);
    }
    Some(out)
}

/// Decide which saved frame to resume from. `requested` is the user
/// argument: `-1` means "pick the latest"; any other value is honored
/// when `> 0`.
///
/// Returns `Some(frame)` when a valid resume target exists, `None`
/// otherwise (caller falls back to "no saved state found" logging).
pub fn select_resume_frame(saved: &[u64], requested: i64) -> Option<u64> {
    if requested == -1 {
        saved.iter().copied().max()
    } else if requested > 0 {
        Some(requested as u64)
    } else {
        None
    }
}
