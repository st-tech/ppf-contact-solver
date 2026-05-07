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
use ppf_cts_formats::files::SCENE_INFO_JSON;

/// Read `<root>/scene_info.json` and merge two dynamic rows:
///   * "Simulated Frames" = count(<output>/vert_*.bin)
///   * "Last Saved"       = max index in save_*.bin, "None" if absent
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

/// Walk `<output>` once and return `(simulated_frames, last_saved)`.
/// `simulated_frames` is the count of `vert_*.bin` files;
/// `last_saved` is the max index across `save_*.bin`, or `None` when
/// no checkpoint exists yet.
pub fn scan_output_progress(output_dir: &Path) -> (usize, Option<u64>) {
    let mut verts = 0usize;
    let mut last_save: Option<u64> = None;
    let entries = match std::fs::read_dir(output_dir) {
        Ok(e) => e,
        Err(_) => return (0, None),
    };
    for entry in entries.flatten() {
        let name = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue,
        };
        if name.starts_with("vert_") && name.ends_with(".bin") {
            verts += 1;
        } else if let Some(rest) = name
            .strip_prefix("save_")
            .and_then(|s| s.strip_suffix(".bin"))
        {
            if let Ok(idx) = rest.parse::<u64>() {
                last_save = Some(last_save.map_or(idx, |m| m.max(idx)));
            }
        }
    }
    (verts, last_save)
}
