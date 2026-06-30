// File: crates/ppf-cts-core/src/datamodel/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// In-memory data model for the migration. Mirrors the *user-facing*
// shape of the original `frontend/_param_.py` and `frontend/_asset_.py`,
// what a notebook caller sees through `app.session.param.set(...)` and
// `app.asset.add.tri(...)`.
//
// These types are the foundation the decoder, scene, and session
// modules build on. Wire-format types live in `ppf-cts-formats`;
// this module is in-memory only.

pub mod animation;
pub mod app;
pub mod asset;
pub mod collider;
pub mod decoder;
pub mod easing;
pub mod elastic_model;
pub mod interp_consts;
pub mod mesh;
pub mod object;
pub mod param_manager;
pub mod params;
pub mod pin;
pub mod pin_apply;
pub mod quat;
pub mod scene;
pub mod session;
pub mod validators;

pub use animation::TransformAnimation;
pub use app::{
    app_pickle_path, clear_cache_dir, compose_data_dir, data_dirpath_for, default_cache_dir,
    recover_session_path, AppPathError, RecoverablePath, RECOVERABLE_FIXED_SESSION_NAME,
};
pub use asset::{Asset, AssetError, AssetKind, AssetRegistry};
pub use collider::{ColliderError, ColliderParam, Sphere, SphereEntry, Wall, WallEntry};
pub use decoder::{
    apply_transform_4x4, barycentric_project_anchors, solid_orig_to_sim, summarize_tetra_jobs,
    StitchRows, TetraJob,
};
pub use easing::{bezier_progress, eased_progress, TransitionKind};
pub use elastic_model::{model_id_to_name, model_name_to_id, MODEL_NAMES};
pub use mesh::{
    bbox, box_mesh, cone_mesh, fix_skinny_triangles, generate_cylinder_faces,
    generate_cylinder_verts, generate_grid_faces, generate_rect_faces, icosphere, line_mesh,
    mobius, normalize, polygon_area_2d, scale_per_axis, scale_to, tet_box_mesh,
    tet_extract_surface, transform_verts_2d, ScaleAxis, TetSurfaceExtract,
};
pub use object::{DynamicColor, Object, ObjectColor};
pub use param_manager::{ParamManager, ParamManagerError};
pub use params::{
    app_param, object_param, ObjectKind, ParamEntry, ParamError, ParamHolder, ParamValue,
};
pub use pin::{
    CenterMode, InterpMode, KeyframeSegment, MoveByDelta, PinData, PinOperation,
};
pub use pin_apply::{
    move_by_step, move_to_step, progress_at, scale_factor_at, scale_step, spin_angle_rad,
    spin_step,
};
pub use quat::{
    apply_transform_to_verts, axis_angle_to_quat, mat3_to_quat, quat_multiply, quat_to_mat3, slerp,
    transform_keyframes_step, Mat3, Quat, Vec3,
};
pub use scene::{CrossStitch, Scene, SceneError, SurfaceMap};
pub use session::{
    analyze_solver_error, convert_integer, convert_time, is_saving_in_progress, latest_log_value,
    latest_vertex_frame, list_saved_states, list_vertex_frames, project_resumable,
    read_log_numbers, read_log_tail, read_vertex_bin, shell_command_script, FixedSession, Platform,
    Session, SessionInfo, SessionOutput,
};
