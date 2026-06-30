// File: crates/ppf-cts-py/src/scene_py/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 bindings for `core::kernels::scene_build`. The Python frontend
// (`frontend/_scene_.py`) dispatches the per-vertex / per-face inner
// loops in `Scene.build` and `FixedScene.color/center` through these.
//
// Sub-module map:
//   * `helpers`:     shared topology readers (faces / edges / tets)
//   * `index_map`:   `scene_build_index_map`
//   * `transform`:   apply_transform_batch + bbox + axis + grab + mat4
//                    scale/rotate + violation messages + quaternion +
//                    TRS apply / decompose + area_weighted_center +
//                    dynamic_color
//   * `easing_geom`: bezier / eased progress + triangle areas +
//                    face-to-vertex weights + direction / cylinder
//                    color + all-vertices-pinned + UV-from-directions
//   * `validators`:  cold-tail validators / arithmetic helpers, mesh
//                    literals
//   * `pin_ops`:     MoveBy / MoveTo / Spin / Scale /
//                    TransformKeyframe / TransformAnimation apply
//   * `build`:       `Scene.build` full-body assembly +
//                    shell shrink/strain conflict
//   * `fixed_init`:  `FixedScene.__init__` validation pipeline
//   * `loops`:       scene_loops bindings (TOML formatters, stitch
//                    preview, segment packing, etc.)

use pyo3::prelude::*;

mod helpers;
mod index_map;
mod transform;
mod easing_geom;
mod validators;
mod pin_ops;
mod build;
mod fixed_init;
mod loops;

/// Register every scene-side PyO3 function.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // index_map
    m.add_function(wrap_pyfunction!(index_map::scene_build_index_map, m)?)?;
    // transform
    m.add_function(wrap_pyfunction!(transform::scene_apply_transform_batch, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_area_weighted_center, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_dynamic_color, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_bbox_displaced, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_axis_min_max, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_object_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_grab_indices, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_mat4_apply_scale, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_mat4_apply_rotate, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_violation_messages, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_quat_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_quat_to_mat3, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_axis_angle_to_quat, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_mat3_to_quat, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_quat_slerp, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_apply_trs_to_verts, m)?)?;
    m.add_function(wrap_pyfunction!(transform::scene_decompose_trs, m)?)?;
    // easing_geom
    m.add_function(wrap_pyfunction!(easing_geom::scene_bezier_progress, m)?)?;
    m.add_function(wrap_pyfunction!(easing_geom::scene_eased_progress, m)?)?;
    m.add_function(wrap_pyfunction!(easing_geom::scene_triangle_areas, m)?)?;
    m.add_function(wrap_pyfunction!(easing_geom::scene_face_to_vert_weights, m)?)?;
    m.add_function(wrap_pyfunction!(easing_geom::scene_average_tri_area, m)?)?;
    m.add_function(wrap_pyfunction!(easing_geom::scene_direction_color, m)?)?;
    m.add_function(wrap_pyfunction!(easing_geom::scene_cylinder_color, m)?)?;
    m.add_function(wrap_pyfunction!(easing_geom::scene_all_vertices_pinned, m)?)?;
    m.add_function(wrap_pyfunction!(easing_geom::scene_uv_from_directions, m)?)?;
    // validators
    m.add_function(wrap_pyfunction!(validators::scene_validate_surface_map_key, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_param_key_no_underscore, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_axis_letter_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_model_name_to_id, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_reduce_axis_bound, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_time_window, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_is_supported_dyn_color, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_wall_move_by_position, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_sphere_move_by_entry, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_collider_time, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_fixed_scene_report_entries, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_known_param_name, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_collider_not_already_added, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_object_rotate_axis, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_stitch_attach, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_set_uv_obj_type, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_object_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_object_not_static, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_classify_velocity_entry, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_scene_select, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_line_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_box_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_tet_box_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_cone_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_project_root_from_frontend_file, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_param_key_exists, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_param_time_strictly_increasing, m)?)?;
    m.add_function(wrap_pyfunction!(validators::scene_validate_session_exists, m)?)?;
    // pin_ops
    m.add_function(wrap_pyfunction!(pin_ops::scene_move_by_apply, m)?)?;
    m.add_function(wrap_pyfunction!(pin_ops::scene_move_to_apply, m)?)?;
    m.add_function(wrap_pyfunction!(pin_ops::scene_spin_apply, m)?)?;
    m.add_function(wrap_pyfunction!(pin_ops::scene_scale_apply, m)?)?;
    m.add_function(wrap_pyfunction!(pin_ops::scene_transform_keyframe_apply, m)?)?;
    m.add_function(wrap_pyfunction!(pin_ops::scene_transform_animation_evaluate, m)?)?;
    // build
    m.add_function(wrap_pyfunction!(build::scene_build_fixed, m)?)?;
    m.add_function(wrap_pyfunction!(
        build::scene_shell_shrink_strain_limit_conflict,
        m
    )?)?;
    // fixed_init
    m.add_function(wrap_pyfunction!(fixed_init::scene_fixed_scene_assemble, m)?)?;
    // loops
    m.add_function(wrap_pyfunction!(loops::scene_stitch_preview_lines, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_face_uv_expand, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_collect_pin_marker_indices, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_concat_static_transform_anims, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_pack_transform_keyframe_segments, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_format_pin_toml, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_format_wall_toml, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_format_sphere_toml, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_format_dyn_param_toml, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_format_static_transform_header, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_per_object_axis_bound, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_concat_i64_lists, m)?)?;
    m.add_function(wrap_pyfunction!(loops::scene_validate_cross_stitch_names, m)?)?;
    // Single source of truth for the collision-window cap, exported so the
    // Python builder imports it instead of re-declaring the literal. Shared
    // with the solver's collision-window table builder and the GPU-side
    // `#define MAX_COLLISION_WINDOWS` in cpp/main/main.cu.
    m.add(
        "MAX_COLLISION_WINDOWS",
        ppf_cts_core::datamodel::object::MAX_COLLISION_WINDOWS,
    )?;
    Ok(())
}
