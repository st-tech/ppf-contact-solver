// File: crates/ppf-cts-server/src/response/scene_info.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Read `<root>/scene_info.json` (written by the build worker) and
// merge dynamic rows derived from `<root>/session/output/`.

use std::path::{Path, PathBuf};

use serde_json::Value;

use ppf_cts_core::datamodel::scene::fmt_thousands;
use ppf_cts_core::datamodel::{list_saved_states, list_vertex_frames};
use ppf_cts_formats::files::SCENE_INFO_JSON;

/// Read `<root>/scene_info.json` and merge two dynamic rows:
///   * "Simulated Frames" = count(<output>/vert_*.bin)
///   * "Last Saved"       = max index in state_<N>.bin.gz, "None" if absent
///
/// Returns `None` when the file is missing, malformed, or not an
/// object: the caller omits the `scene_info` field and the addon
/// panel hides the box.
pub fn load_scene_info(root: &str) -> Option<Value> {
    let scene_info_path = PathBuf::from(root).join(SCENE_INFO_JSON);
    let text = std::fs::read_to_string(&scene_info_path).ok()?;
    let mut v: Value = serde_json::from_str(&text).ok()?;
    {
        let obj = v.as_object_mut()?;
        let output_dir = PathBuf::from(root).join("session").join("output");
        let (simulated, last_saved) = scan_output_progress(&output_dir);
        obj.insert(
            "Simulated Frames".into(),
            Value::String(fmt_thousands(simulated)),
        );
        obj.insert(
            "Last Saved".into(),
            Value::String(match last_saved {
                Some(n) => n.to_string(),
                None => "None".to_string(),
            }),
        );
    }
    Some(v)
}

/// Return `(simulated_frames, last_saved)` for an `<output>` dir.
///
/// `simulated_frames` is the count of `vert_<N>.bin` files;
/// `last_saved` is the max N across `state_<N>.bin.gz` checkpoints,
/// or `None` when no checkpoint exists yet. The solver writes
/// `state_<N>.bin.gz` (gzipped resume points) at every auto-save and
/// on save-and-quit, so that's what "Last Saved" must reflect.
pub fn scan_output_progress(output_dir: &Path) -> (usize, Option<u64>) {
    let verts = list_vertex_frames(output_dir).len();
    let last_saved = list_saved_states(output_dir).into_iter().max();
    (verts, last_saved)
}
