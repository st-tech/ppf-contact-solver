// File: crates/ppf-cts-core/src/datamodel/decoder/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Decoder support that lives next to the in-memory scene/session
// types: the `BlenderApp` path resolution and the cluster of
// validation helpers the Python decoder calls before mutating its
// caller's scene/session. The pure-compute math the decoder runs
// (4x4 transforms, the barycentric anchor projection, the surface
// frame "orig_to_sim" lookup, etc.) lives in `kernels::decoder` since
// each is a self-contained slice-in / slice-out routine.
//
// The decoder itself stays in Python because the on-disk envelope is
// pickle / CBOR dispatch and the consumers are Python objects (Scene,
// Session, PinHolder, ParamHolder).

use std::path::{Path, PathBuf};

use crate::datamodel::app as app_paths;

pub mod validate;

// Re-export the math kernels at the historical path
// (`crate::datamodel::decoder::FOO`) so existing PyO3 bindings and
// downstream callers don't need updating.
pub use crate::kernels::decoder::{
    apply_transform_4x4, barycentric_project_anchors, closest_vertex_index,
    keyframe_translation_segments, solid_orig_to_sim, summarize_tetra_jobs, StitchRows, TetraJob,
};

pub use validate::{
    validate_cross_stitch_endpoints, validate_group_type, validate_invisible_collider_thickness,
    validate_mesh_ref_known, validate_object_has_mesh, validate_param_group_has_uuids,
    validate_param_object_uuid, validate_param_top_keys, validate_pickle_extension,
    validate_pin_op_types, validate_rod_has_edges, validate_scene_object_identity,
    validate_static_anim_xor_ops, validate_static_op_type, DecoderValidationError,
};

/// Bundle of directories `BlenderApp.__init__` needs up front:
/// `data_dirpath` (`~/.local/share/ppf-cts/git-{branch}` or the
/// Windows base-dir variant), the per-project `root`, and the
/// `cache_root` (`{root}/.cash`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlenderAppPaths {
    pub data_dirpath: PathBuf,
    pub root: PathBuf,
    pub cache_root: PathBuf,
}

/// Mirror the path math at the top of `BlenderApp.__init__`. Reuses
/// the same `data_dirpath` resolver `App.get_data_dirpath` uses (read
/// `branch_name.txt`, fall back to `git branch --show-current`, then
/// to `"unknown"`).
pub fn blender_app_paths(
    frontend_file: &Path,
    name: &str,
    home_dir: Option<&Path>,
) -> BlenderAppPaths {
    let base_dir = app_paths::frontend_base_dir_from_file(frontend_file);
    let data_dirpath = app_paths::data_dirpath_for(&base_dir, home_dir);
    let root = data_dirpath.join(name);
    let cache_root = root.join(".cash");
    BlenderAppPaths {
        data_dirpath,
        root,
        cache_root,
    }
}

/// Filename used for the cached tetrahedralization result:
/// `f"{hash}_tetrahedralize_.npz"`.
pub fn tetra_cache_filename(tri_mesh_hash: &str) -> String {
    format!("{tri_mesh_hash}_tetrahedralize_.npz")
}

/// Directory layout the `BlenderApp._persist_app_state` write uses:
/// the `app_state.pickle` final path plus the `.tmp` sibling we write
/// to first, then `os.replace` over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppStatePersistPaths {
    pub final_path: PathBuf,
    pub tmp_path: PathBuf,
}

/// Mirror the path arithmetic in `BlenderApp._persist_app_state`.
/// The caller still does the actual atomic write (the bytes are
/// pickle bytes, which is allowlist territory).
pub fn app_state_persist_paths(root: &Path) -> AppStatePersistPaths {
    let final_path = root.join("app_state.pickle");
    let tmp_path = {
        let mut p = final_path.clone().into_os_string();
        p.push(".tmp");
        PathBuf::from(p)
    };
    AppStatePersistPaths { final_path, tmp_path }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetra_cache_filename_format() {
        assert_eq!(
            tetra_cache_filename("deadbeef"),
            "deadbeef_tetrahedralize_.npz"
        );
    }

    #[test]
    fn app_state_persist_paths_appends_tmp() {
        let p = app_state_persist_paths(Path::new("/tmp/proj"));
        assert_eq!(p.final_path, PathBuf::from("/tmp/proj/app_state.pickle"));
        assert_eq!(p.tmp_path, PathBuf::from("/tmp/proj/app_state.pickle.tmp"));
    }

    #[test]
    fn blender_app_paths_unix_layout() {
        if cfg!(target_os = "windows") {
            return;
        }
        // Pin the branch via `.git/branch_name.txt` so we don't depend
        // on the host's git checkout.
        let tmp = tempfile::tempdir().unwrap();
        let git_dir = tmp.path().join(".git");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::write(git_dir.join("branch_name.txt"), "feature\n").unwrap();
        let frontend = tmp.path().join("frontend").join("_decoder_.py");
        std::fs::create_dir_all(frontend.parent().unwrap()).unwrap();
        let paths = blender_app_paths(&frontend, "myproj", Some(Path::new("/home/u")));
        assert_eq!(
            paths.data_dirpath,
            PathBuf::from("/home/u/.local/share/ppf-cts/git-feature")
        );
        assert!(paths.root.ends_with("myproj"));
        assert!(paths.cache_root.ends_with(".cash"));
    }
}
