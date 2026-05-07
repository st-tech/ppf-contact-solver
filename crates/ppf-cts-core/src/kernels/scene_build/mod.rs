// File: crates/ppf-cts-core/src/kernels/scene_build/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Scene-build inner-loop kernels.
//
// These are the Python-level computational hotspots that dominate
// `Scene.build()` and `FixedScene.color() / center()` runtime on
// nontrivial scenes. They take primitive slices and return primitive
// results so they're testable in pure Rust and bind cleanly through
// PyO3.
//
// Originally a single 3000+ line module, split into sub-files by
// section banner with the public API re-exported here so existing call
// sites at `crate::kernels::scene_build::FUNC_NAME` keep working.

mod assembly;
mod bbox;
mod color_uv;
mod index_map;
mod mesh_metrics;
mod pin_kernel;
mod quaternion;
mod transform;

#[cfg(test)]
mod tests;

// Re-export the public API. Items currently `pub fn` upstream stay
// `pub` here; private helpers (`hsv_to_rgb`, `mat4_mul`, `add_entry`,
// `resolve`) stay sub-file-private.

pub use assembly::{
    assemble_dyn_scene, assemble_static_scene, check_shell_shrink_strain_limit_conflict,
    group_vertex_alias, rod_tri_contact_offset_check, AssembleObject, AssembleObjectStats,
    AssembleResult, AssembleStaticObject, AssembleStaticResult, CrossStitch,
    RodTriOffsetViolation, SceneAssemblyError,
};
pub use bbox::{axis_min_max, bbox, bbox_displaced, object_bbox_no_translate};
pub use color_uv::{
    all_vertices_pinned, dynamic_color, uv_from_directions, cylinder_color,
    direction_color, DirectionError,
};
pub use index_map::{build_index_map, IndexMapObject, IndexMapResult, SceneBuildError};
pub use mesh_metrics::{area_weighted_center, average_tri_area, face_to_vert_weights, triangle_areas};
pub use pin_kernel::{
    bezier_progress, eased_progress, move_by_apply, move_to_apply, scale_apply, spin_apply,
    transform_animation_evaluate, transform_keyframe_apply, SegInterp,
};
pub use quaternion::{
    apply_trs_to_verts, axis_angle_to_quat, decompose_trs, mat3_to_quat, quat_multiply,
    quat_slerp, quat_to_mat3,
};
pub use transform::{
    apply_transform_batch, grab_indices, mat4_apply_axis_rotation_keep_translation,
    mat4_apply_uniform_scale, violation_messages, Normalize, TransformError,
};
