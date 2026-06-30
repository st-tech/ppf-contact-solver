// File: crates/ppf-cts-core/src/kernels/scene_build/tests.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Pure-Rust tests for the scene_build kernels. Kept together (rather
// than per-sub-file) to avoid splitting risk; references items via the
// flattened re-exports in `scene_build::mod`.

use super::*;

fn make_obj<'a>(
    name: &'a str,
    n_verts: usize,
    edges: Option<&'a [[u32; 2]]>,
    faces: Option<&'a [[u32; 3]]>,
    tets: Option<&'a [[u32; 4]]>,
) -> IndexMapObject<'a> {
    IndexMapObject {
        name,
        n_verts,
        edges,
        faces,
        tets,
    }
}

#[test]
fn single_tri_assigns_sequential_globals() {
    let faces: [[u32; 3]; 1] = [[0, 1, 2]];
    let o = make_obj("sheet", 3, None, Some(&faces), None);
    let r = build_index_map(&[o]).unwrap();
    assert_eq!(r.rod_vert_range, (0, 0));
    assert_eq!(r.shell_vert_range, (0, 3));
    assert_eq!(r.concat_count, 3);
    assert_eq!(r.map_by_name["sheet"], vec![0i64, 1, 2]);
}

#[test]
fn rod_verts_precede_shell_verts() {
    let edges: [[u32; 2]; 2] = [[0, 1], [1, 2]];
    let rod = make_obj("rope", 3, Some(&edges), None, None);
    let faces: [[u32; 3]; 2] = [[0, 1, 2], [0, 2, 3]];
    let shell = make_obj("sheet", 4, None, Some(&faces), None);
    let r = build_index_map(&[rod, shell]).unwrap();
    assert_eq!(r.rod_vert_range, (0, 3));
    assert_eq!(r.shell_vert_range, (3, 7));
    assert_eq!(r.concat_count, 7);
}

#[test]
fn unused_vertex_finalized() {
    let f: [[u32; 3]; 1] = [[0, 1, 2]];
    let o = make_obj("sheet", 4, None, Some(&f), None);
    let r = build_index_map(&[o]).unwrap();
    let m = &r.map_by_name["sheet"];
    assert_eq!(m[0], 0);
    assert_eq!(m[1], 1);
    assert_eq!(m[2], 2);
    assert_eq!(m[3], 3);
    assert_eq!(r.concat_count, 4);
}

#[test]
fn apply_transform_translation_only() {
    let m = [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    apply_transform_batch(&m, &mut v, true, None);
    assert_eq!(v, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn apply_transform_no_translate() {
    let m = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 3.0, 0.0, 20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut v = vec![1.0, 2.0, 3.0];
    apply_transform_batch(&m, &mut v, false, None);
    assert_eq!(v, vec![2.0, 6.0, 12.0]);
}

#[test]
fn apply_transform_normalize_identity_centers_and_scales() {
    // Identity matrix + normalize pre-step: v' = (v - center) / bbox_max.
    let m = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    // Two vertices, axis-aligned bbox center (0.5, 0.5, 0.5),
    // bbox extent (1, 1, 1) so max = 1.
    let mut v = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let nrm = Normalize { bbox_max: 1.0, center: [0.5, 0.5, 0.5] };
    apply_transform_batch(&m, &mut v, false, Some(nrm));
    assert_eq!(v, vec![-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]);
}

#[test]
fn apply_transform_normalize_with_scale_and_translate() {
    // (v - c) / b -> 3x3 scale -> + translation (when translate=true).
    // Hand-computed expected:
    //   c = (1, 2, 3), b = 2,
    //   scale 3x3 = diag(2, 3, 4), translate = (10, 20, 30).
    // Vertex (3, 5, 7) -> (1, 1.5, 2) -> (2, 4.5, 8) -> (12, 24.5, 38).
    let m = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 3.0, 0.0, 20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut v = vec![3.0, 5.0, 7.0];
    let nrm = Normalize { bbox_max: 2.0, center: [1.0, 2.0, 3.0] };
    apply_transform_batch(&m, &mut v, true, Some(nrm));
    assert!((v[0] - 12.0).abs() < 1e-12);
    assert!((v[1] - 24.5).abs() < 1e-12);
    assert!((v[2] - 38.0).abs() < 1e-12);
}

#[test]
fn area_weighted_center_unit_tri() {
    // Single right triangle in xy plane: (0,0,0) (1,0,0) (0,1,0).
    // Area = 0.5; centroid = (1/3, 1/3, 0).
    let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let t: [[u32; 3]; 1] = [[0, 1, 2]];
    let (c, total) = area_weighted_center(&v, &t);
    assert!((c[0] - 1.0 / 3.0).abs() < 1e-12);
    assert!((c[1] - 1.0 / 3.0).abs() < 1e-12);
    assert!(c[2].abs() < 1e-12);
    assert!((total - 0.5).abs() < 1e-12);
}

#[test]
fn area_weighted_center_two_tris() {
    // Two unit-area triangles, one centroid at (0.5, 0.5, 0),
    // one at (3.5, 0.5, 0). Equal weight → midpoint.
    let v = vec![
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 1.0, 0.0, 3.0, 1.0, 0.0,
    ];
    let t: [[u32; 3]; 4] =
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]];
    let (c, total) = area_weighted_center(&v, &t);
    assert!((total - 2.0).abs() < 1e-12);
    // Both squares have unit area → weighted midpoint is x = (0.5 + 3.5)/2 = 2.0.
    assert!((c[0] - 2.0).abs() < 1e-12);
    assert!((c[1] - 0.5).abs() < 1e-12);
}

#[test]
fn area_weighted_center_empty_returns_zero() {
    let v: Vec<f64> = vec![];
    let t: [[u32; 3]; 0] = [];
    let (c, total) = area_weighted_center(&v, &t);
    assert_eq!(total, 0.0);
    assert_eq!(c, [0.0, 0.0, 0.0]);
}

#[test]
fn dynamic_color_inactive_face_returns_base_color() {
    // dyn_face_color[0] = 0 (NONE) → color stays at base.
    let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let t: [[u32; 3]; 1] = [[0, 1, 2]];
    let init = vec![0.5];
    let weights = vec![1.0, 1.0, 1.0];
    let dfc = vec![0u8];
    let dfi = vec![1.0];
    let base = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let out = dynamic_color(
        &v, &t, &init, &weights, &dfc, &dfi, &base, 2.0,
    );
    // Inactive → vert_intensity=0 → out == base.
    for i in 0..9 {
        assert!((out[i] - base[i]).abs() < 1e-12);
    }
}

#[test]
fn dynamic_color_unstretched_face_is_blue() {
    // rat = 1 (no stretch) → val = 0 → hue = 240/360 (blue).
    let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let t: [[u32; 3]; 1] = [[0, 1, 2]];
    let init = vec![0.5];
    let weights = vec![1.0, 1.0, 1.0];
    let dfc = vec![1u8];
    let dfi = vec![1.0]; // intensity 1 → out is fully face color.
    let base = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let out = dynamic_color(
        &v, &t, &init, &weights, &dfc, &dfi, &base, 2.0,
    );
    // Each vert's blended color should equal the face's blue-ish
    // RGB. hue=240/360, s=0.75, v=1.0 → R=0.25, G=0.25, B=1.0.
    for vi in 0..3 {
        assert!((out[3 * vi] - 0.25).abs() < 1e-9);
        assert!((out[3 * vi + 1] - 0.25).abs() < 1e-9);
        assert!((out[3 * vi + 2] - 1.0).abs() < 1e-9);
    }
}

#[test]
fn dynamic_color_stretched_face_is_red() {
    // rat = 2 (max stretch) → val = 1 → hue = 0 (red).
    let v0 = [0.0, 0.0, 0.0];
    let v1 = [2.0, 0.0, 0.0]; // doubled in x
    let v2 = [0.0, 1.0, 0.0];
    let v: Vec<f64> = [v0, v1, v2].iter().flatten().copied().collect();
    let t: [[u32; 3]; 1] = [[0, 1, 2]];
    let init = vec![0.5]; // initial area 0.5; new area = 1.0; rat=2.
    let weights = vec![1.0, 1.0, 1.0];
    let dfc = vec![1u8];
    let dfi = vec![1.0];
    let base = vec![1.0; 9];
    let out = dynamic_color(
        &v, &t, &init, &weights, &dfc, &dfi, &base, 2.0,
    );
    // hue=0, s=0.75, v=1.0 → R=1, G=0.25, B=0.25.
    for vi in 0..3 {
        assert!((out[3 * vi] - 1.0).abs() < 1e-9);
        assert!((out[3 * vi + 1] - 0.25).abs() < 1e-9);
        assert!((out[3 * vi + 2] - 0.25).abs() < 1e-9);
    }
}

#[test]
fn bbox_world_simple() {
    let v = vec![1.0, 2.0, 3.0, -1.0, 4.0, 0.0, 5.0, -2.0, 6.0];
    let (hi, lo) = bbox(&v);
    assert_eq!(hi, [5.0, 4.0, 6.0]);
    assert_eq!(lo, [-1.0, -2.0, 0.0]);
}

#[test]
fn bbox_displaced_uses_offsets() {
    let local = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let idx = vec![0i64, 1, 1];
    let disp = vec![10.0, 0.0, 0.0,  0.0, 100.0, 0.0];
    let (hi, lo) = bbox_displaced(&local, &idx, &disp);
    assert_eq!(hi, [10.0, 101.0, 0.0]);
    assert_eq!(lo, [0.0, 0.0, 0.0]);
}

#[test]
fn axis_min_max_y() {
    let v = vec![0.0, 5.0, 0.0, 1.0, -3.0, 2.0, -1.0, 9.0, 4.0];
    let (lo, hi) = axis_min_max(&v, 1);
    assert_eq!(lo, -3.0);
    assert_eq!(hi, 9.0);
}

#[test]
fn object_bbox_no_translate_identity() {
    let v = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0];
    let mut m = [[0.0f64; 4]; 4];
    m[0][0] = 1.0; m[1][1] = 1.0; m[2][2] = 1.0; m[3][3] = 1.0;
    m[0][3] = 100.0; // ignored
    let (size, center) = object_bbox_no_translate(&v, &m);
    assert_eq!(size, [2.0, 4.0, 6.0]);
    assert_eq!(center, [1.0, 2.0, 3.0]);
}

#[test]
fn grab_max_x_picks_rightmost() {
    let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let r = grab_indices(&v, [1.0, 0.0, 0.0], 1e-3);
    assert_eq!(r, vec![1, 3]);
}

#[test]
fn mat4_apply_uniform_scale_doubles_linear_block() {
    let mut m = [[0.0f64; 4]; 4];
    m[0][0] = 1.0; m[1][1] = 1.0; m[2][2] = 1.0; m[3][3] = 1.0;
    m[0][3] = 5.0; m[1][3] = 6.0; m[2][3] = 7.0;
    let r = mat4_apply_uniform_scale(&m, 2.0);
    assert_eq!(r[0][0], 2.0);
    assert_eq!(r[1][1], 2.0);
    assert_eq!(r[2][2], 2.0);
    // diag(s,s,s,1) keeps translation column unchanged.
    assert_eq!(r[0][3], 5.0);
    assert_eq!(r[1][3], 6.0);
    assert_eq!(r[2][3], 7.0);
}

#[test]
fn violation_messages_order_and_filtering() {
    let m = violation_messages(true, false, true, false);
    assert_eq!(m.len(), 2);
    assert_eq!(m[0], "Scene has self-intersections");
    assert_eq!(m[1], "Scene has wall constraint violations");
    let none = violation_messages(false, false, false, false);
    assert!(none.is_empty());
    let all = violation_messages(true, true, true, true);
    assert_eq!(all.len(), 4);
}

#[test]
fn quat_multiply_identity() {
    let id = [1.0, 0.0, 0.0, 0.0];
    let q = [0.5_f64.sqrt(), 0.0, 0.5_f64.sqrt(), 0.0];
    let r = quat_multiply(id, q);
    for i in 0..4 {
        assert!((r[i] - q[i]).abs() < 1e-12);
    }
}

#[test]
fn quat_to_mat3_then_back_roundtrip() {
    let q = axis_angle_to_quat([0.0, 1.0, 0.0], 60.0);
    let m = quat_to_mat3(q);
    let qb = mat3_to_quat(m);
    // Sign of quaternion is ambiguous; compare with sign-fix.
    let dot = q[0] * qb[0] + q[1] * qb[1] + q[2] * qb[2] + q[3] * qb[3];
    let qb = if dot < 0.0 {
        [-qb[0], -qb[1], -qb[2], -qb[3]]
    } else {
        qb
    };
    for i in 0..4 {
        assert!((q[i] - qb[i]).abs() < 1e-9);
    }
}

#[test]
fn axis_angle_zero_returns_identity_quat() {
    let q = axis_angle_to_quat([0.0, 1.0, 0.0], 0.0);
    assert!((q[0] - 1.0).abs() < 1e-12);
    assert!(q[1].abs() < 1e-12);
    assert!(q[2].abs() < 1e-12);
    assert!(q[3].abs() < 1e-12);
}

#[test]
fn quat_slerp_endpoints() {
    let q0 = [1.0, 0.0, 0.0, 0.0];
    let q1 = axis_angle_to_quat([0.0, 0.0, 1.0], 90.0);
    let r0 = quat_slerp(q0, q1, 0.0);
    let r1 = quat_slerp(q0, q1, 1.0);
    for i in 0..4 {
        assert!((r0[i] - q0[i]).abs() < 1e-9);
    }
    // q1 may flip sign under shortest-arc; allow either.
    let dot1 = q1[0] * r1[0] + q1[1] * r1[1] + q1[2] * r1[2] + q1[3] * r1[3];
    let r1 = if dot1 < 0.0 {
        [-r1[0], -r1[1], -r1[2], -r1[3]]
    } else {
        r1
    };
    for i in 0..4 {
        assert!((r1[i] - q1[i]).abs() < 1e-9);
    }
}

#[test]
fn apply_trs_translates_and_scales() {
    let local = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let id = [1.0, 0.0, 0.0, 0.0];
    let out = apply_trs_to_verts(&local, [10.0, 20.0, 30.0], id, [2.0, 3.0, 4.0]);
    assert_eq!(out, vec![12.0, 20.0, 30.0, 10.0, 23.0, 30.0]);
}

#[test]
fn decompose_trs_recovers_translation_and_scale() {
    let mut m = [[0.0f64; 4]; 4];
    m[0][0] = 2.0;
    m[1][1] = 3.0;
    m[2][2] = 4.0;
    m[0][3] = 5.0;
    m[1][3] = 6.0;
    m[2][3] = 7.0;
    m[3][3] = 1.0;
    let (t, q, s) = decompose_trs(&m);
    assert_eq!(t, [5.0, 6.0, 7.0]);
    assert!((s[0] - 2.0).abs() < 1e-12);
    assert!((s[1] - 3.0).abs() < 1e-12);
    assert!((s[2] - 4.0).abs() < 1e-12);
    assert!((q[0] - 1.0).abs() < 1e-12);
}

#[test]
fn bezier_progress_endpoints_and_linear() {
    // Linear-equivalent handles: (1/3, 1/3) and (2/3, 2/3) -> y == t.
    let h = ([1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]);
    for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let y = bezier_progress(t, h);
        assert!((y - t).abs() < 1e-3, "t={t}, y={y}");
    }
}

#[test]
fn eased_progress_smooth_matches_smoothstep() {
    let p = eased_progress(0.5, 0.0, 1.0, "smooth", None);
    // smoothstep(0.5) = 0.5
    assert!((p - 0.5).abs() < 1e-12);
    let p2 = eased_progress(0.25, 0.0, 1.0, "smooth", None);
    assert!((p2 - 0.15625).abs() < 1e-9);
}

#[test]
fn triangle_areas_unit_and_zero() {
    let v = vec![
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        // degenerate
        0.0, 0.0, 0.0,
    ];
    let tris = vec![[0u32, 1, 2], [0, 1, 3]];
    let a = triangle_areas(&v, &tris);
    assert!((a[0] - 0.5).abs() < 1e-12);
    assert!(a[1] < 1e-12);
}

#[test]
fn face_to_vert_weights_inverse_count() {
    let tris = vec![[0u32, 1, 2], [0, 1, 3]];
    let w = face_to_vert_weights(4, &tris, FACE_TO_VERT_WEIGHT_EPS);
    // verts 0,1 are in 2 faces -> 1/(2+eps); verts 2,3 in 1 -> 1/(1+eps).
    assert!((w[0] - 1.0 / (2.0 + FACE_TO_VERT_WEIGHT_EPS)).abs() < 1e-12);
    assert!((w[3] - 1.0 / (1.0 + FACE_TO_VERT_WEIGHT_EPS)).abs() < 1e-12);
}

#[test]
fn average_tri_area_skips_empty() {
    let empty: Vec<f64> = Vec::new();
    assert_eq!(average_tri_area(&empty), 0.0);
    let some = vec![1.0, 2.0, 3.0];
    assert!((average_tri_area(&some) - 2.0).abs() < 1e-12);
}

#[test]
fn direction_color_min_max_endpoints() {
    // Two verts: one at min projection, one at max.
    // dot with x-axis -> min=-1, max=2.
    let v = vec![-1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let c = direction_color(&v, [1.0, 0.0, 0.0]);
    // y=0 -> hue=240/360, y=1 -> hue=0 -> red.
    // Match Python: hsv(0,.75,1) = (1.0, 0.25, 0.25).
    assert!((c[3] - 1.0).abs() < 1e-9);
    assert!((c[4] - 0.25).abs() < 1e-9);
    assert!((c[5] - 0.25).abs() < 1e-9);
}

#[test]
fn cylinder_color_outputs_unit_rgb() {
    let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    // ey=up=(0,1,0); ex = direction × ey = x_dir × y = z_dir, etc.
    let c = cylinder_color(&v, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    // Each color should be in [0,1].
    for v in c {
        assert!((0.0..=1.0).contains(&v));
    }
}

#[test]
fn all_vertices_pinned_basic() {
    let a = [0i64, 1];
    let b = [2i64];
    let pinned = all_vertices_pinned(3, &[&a, &b]);
    assert!(pinned);
    let only = [0i64];
    let not_pinned = all_vertices_pinned(3, &[&only]);
    assert!(!not_pinned);
}

#[test]
fn uv_from_directions_orthogonal_axes() {
    let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let tris = vec![[0u32, 1, 2]];
    // ex=x, ey=z; normal of triangle (xz plane) is +/-y, orthogonal to both.
    let uv = uv_from_directions(
        &v, &tris, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1e-3,
    )
    .unwrap();
    // a=(0,0): u=0,v=0; b=(1,0,0): u=1,v=0; c=(0,0,1): u=0,v=1.
    assert!((uv[0] - 0.0).abs() < 1e-12);
    assert!((uv[1] - 0.0).abs() < 1e-12);
    assert!((uv[2] - 1.0).abs() < 1e-12);
    assert!((uv[3] - 0.0).abs() < 1e-12);
    assert!((uv[4] - 0.0).abs() < 1e-12);
    assert!((uv[5] - 1.0).abs() < 1e-12);
}

#[test]
fn compute_uv_rejects_non_orthogonal_inputs() {
    let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let tris = vec![[0u32, 1, 2]];
    let err = uv_from_directions(
        &v, &tris, [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 1e-3,
    );
    assert!(matches!(err, Err(DirectionError::NotOrthogonal { .. })));
}

#[test]
fn mat4_apply_axis_rotation_keeps_translation() {
    let mut m = [[0.0f64; 4]; 4];
    m[0][0] = 1.0; m[1][1] = 1.0; m[2][2] = 1.0; m[3][3] = 1.0;
    m[0][3] = 5.0; m[1][3] = 6.0; m[2][3] = 7.0;
    let r = mat4_apply_axis_rotation_keep_translation(&m, 'z', 90.0).unwrap();
    // 90° z: x->y, y->-x.
    assert!((r[0][0] - 0.0).abs() < 1e-12);
    assert!((r[0][1] - (-1.0)).abs() < 1e-12);
    assert!((r[1][0] - 1.0).abs() < 1e-12);
    // Translation preserved.
    assert_eq!(r[0][3], 5.0);
    assert_eq!(r[1][3], 6.0);
    assert_eq!(r[2][3], 7.0);
}

#[test]
fn move_by_apply_progress_endpoints() {
    // Two vertices, delta (1, 0, 0) for vertex 0 only.
    let v = vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0];
    let d = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Before t_start: returns input.
    let r = move_by_apply(&v, &d, -1.0, 0.0, 1.0, "linear", None);
    assert_eq!(r, v);
    // After t_end: full delta.
    let r = move_by_apply(&v, &d, 2.0, 0.0, 1.0, "linear", None);
    assert_eq!(r, vec![1.0, 0.0, 0.0, 5.0, 5.0, 5.0]);
    // Mid: half delta with linear.
    let r = move_by_apply(&v, &d, 0.5, 0.0, 1.0, "linear", None);
    assert!((r[0] - 0.5).abs() < 1e-12);
    assert!((r[3] - 5.0).abs() < 1e-12);
}

#[test]
fn move_to_apply_lerp_then_target() {
    let v = vec![0.0, 0.0, 0.0];
    let t = vec![10.0, 0.0, 0.0];
    let r = move_to_apply(&v, &t, -1.0, 0.0, 1.0, "linear", None);
    assert_eq!(r, v);
    let r = move_to_apply(&v, &t, 2.0, 0.0, 1.0, "linear", None);
    assert_eq!(r, t);
    let r = move_to_apply(&v, &t, 0.5, 0.0, 1.0, "linear", None);
    assert!((r[0] - 5.0).abs() < 1e-12);
}

#[test]
fn spin_apply_no_op_before_start() {
    let v = vec![1.0, 0.0, 0.0];
    let r = spin_apply(&v, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 90.0, 1.0, 2.0, 0.5);
    assert_eq!(r, v);
}

#[test]
fn spin_apply_quarter_rotation_z() {
    // Rotate point (1,0,0) by 90° around z, t=1s, ω=90°/s.
    let v = vec![1.0, 0.0, 0.0];
    let r = spin_apply(&v, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 90.0, 0.0, 10.0, 1.0);
    assert!(r[0].abs() < 1e-9);
    assert!((r[1] - 1.0).abs() < 1e-9);
    assert!(r[2].abs() < 1e-9);
}

#[test]
fn scale_apply_endpoints() {
    let v = vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    // Before t_start
    let r = scale_apply(&v, [0.0, 0.0, 0.0], 2.0, 0.0, 1.0, "linear", -1.0);
    assert_eq!(r, v);
    // After t_end: full scale.
    let r = scale_apply(&v, [0.0, 0.0, 0.0], 2.0, 0.0, 1.0, "linear", 2.0);
    assert!((r[0] - 2.0).abs() < 1e-9);
    assert!((r[3] - 4.0).abs() < 1e-9);
    // Mid: factor = 1.5.
    let r = scale_apply(&v, [0.0, 0.0, 0.0], 2.0, 0.0, 1.0, "linear", 0.5);
    assert!((r[0] - 1.5).abs() < 1e-9);
}

#[test]
fn rod_tri_offset_clean_geometry_passes() {
    // Triangle in xy plane; rod high above.
    let v = vec![
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        // rod
        0.0, 0.0, 5.0,
        0.0, 0.0, 6.0,
    ];
    let tris = vec![[0u32, 1, 2]];
    let rods = vec![[3u32, 4]];
    let tri_off = vec![0.1];
    let rod_off = vec![0.1];
    rod_tri_contact_offset_check(&v, &rods, &tris, &tri_off, &rod_off).unwrap();
}

#[test]
fn rod_tri_offset_close_rod_violates() {
    let v = vec![
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        // rod sitting just above triangle (z=0.02), longer than offset
        0.1, 0.1, 0.02,
        0.5, 0.1, 0.02,
    ];
    let tris = vec![[0u32, 1, 2]];
    let rods = vec![[3u32, 4]];
    let tri_off = vec![0.05];
    let rod_off = vec![0.05];
    let r = rod_tri_contact_offset_check(&v, &rods, &tris, &tri_off, &rod_off);
    assert!(matches!(r, Err(RodTriOffsetViolation::VertexInsideOffset { .. })));
}

#[test]
fn rod_tri_offset_short_rod_flagged() {
    // edge length 0.05, offset 0.1 → EdgeShorterThanOffset.
    let v = vec![
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.1, 0.1, 5.0,
        0.15, 0.1, 5.0,
    ];
    let tris = vec![[0u32, 1, 2]];
    let rods = vec![[3u32, 4]];
    let r = rod_tri_contact_offset_check(&v, &rods, &tris, &[], &vec![0.1]);
    assert!(matches!(r, Err(RodTriOffsetViolation::EdgeShorterThanOffset { .. })));
}

#[test]
fn rod_tri_offset_short_rod_flagged_no_tris() {
    // Rod-only scene (no triangles): the per-rod edge-length check is
    // triangle-independent, so an over-large offset must still flag.
    // edge length 0.05, offset 0.1 → EdgeShorterThanOffset.
    let v = vec![
        0.1, 0.1, 5.0,
        0.15, 0.1, 5.0,
    ];
    let rods = vec![[0u32, 1]];
    let tris: Vec<[u32; 3]> = vec![];
    let r = rod_tri_contact_offset_check(&v, &rods, &tris, &[], &vec![0.1]);
    assert!(matches!(r, Err(RodTriOffsetViolation::EdgeShorterThanOffset { .. })));
}

#[test]
fn transform_animation_evaluate_endpoints() {
    let local = vec![1.0, 0.0, 0.0];
    let times = vec![0.0, 1.0];
    let trans = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
    let quats = vec![[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
    let scales = vec![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
    // Before/at start
    let r = transform_animation_evaluate(&local, &times, &trans, &quats, &scales, -1.0);
    assert!((r[0] - 1.0).abs() < 1e-9);
    // At/after end
    let r = transform_animation_evaluate(&local, &times, &trans, &quats, &scales, 2.0);
    assert!((r[0] - 11.0).abs() < 1e-9);
    // Mid linear
    let r = transform_animation_evaluate(&local, &times, &trans, &quats, &scales, 0.5);
    assert!((r[0] - 6.0).abs() < 1e-9);
}

// ----- assemble_dyn_scene -----

fn make_assemble_obj<'a>(
    name: &'a str,
    obj_type: &'a str,
    verts: &'a [f64],
    color: &'a [f64],
    faces: Option<&'a [[u32; 3]]>,
    edges: Option<&'a [[u32; 2]]>,
    tets: Option<&'a [[u32; 4]]>,
) -> AssembleObject<'a> {
    AssembleObject {
        name,
        obj_type,
        vertex: verts,
        color,
        velocity: [0.0, 0.0, 0.0],
        edges,
        faces,
        tets,
        uv: None,
        dynamic_color: 0,
        dynamic_intensity: 1.0,
        pinned_indices: &[],
        stitch_ind: None,
        stitch_ind_cols: 0,
        stitch_w: None,
        stitch_w_cols: 0,
        stitch_stiffness: 1.0,
        position: [0.0, 0.0, 0.0],
    }
}

#[test]
fn assemble_empty_scene_returns_empty() {
    let map: std::collections::HashMap<String, Vec<i64>> = std::collections::HashMap::new();
    let r = assemble_dyn_scene(&[], &map, 0, &[], &[]).unwrap();
    assert_eq!(r.concat_count, 0);
    assert!(r.concat_vert.is_empty());
    assert!(r.concat_rod.is_empty());
    assert!(r.concat_tri.is_empty());
    assert_eq!(r.rod_count, 0);
    assert_eq!(r.shell_count, 0);
}

#[test]
fn assemble_single_tri_object() {
    let verts = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let color = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    let faces: [[u32; 3]; 1] = [[0, 1, 2]];
    let mut obj = make_assemble_obj(
        "sheet", "tri", &verts, &color, Some(&faces), None, None,
    );
    obj.velocity = [1.0, 0.0, 0.0];
    let mut map = std::collections::HashMap::new();
    map.insert("sheet".to_string(), vec![0i64, 1, 2]);
    let disp_idx = vec![("sheet".to_string(), [10.0, 20.0, 30.0])];
    let r = assemble_dyn_scene(&[obj], &map, 3, &disp_idx, &[]).unwrap();
    assert_eq!(r.concat_count, 3);
    assert_eq!(r.concat_vert.len(), 9);
    assert_eq!(r.concat_tri, vec![0u32, 1, 2]);
    assert_eq!(r.shell_count, 1);
    assert_eq!(r.rod_count, 0);
    // Velocity scattered.
    assert_eq!(r.concat_vel[0], 1.0);
    assert_eq!(r.concat_vel[3], 1.0);
    assert_eq!(r.concat_vel[6], 1.0);
    // Displacement.
    assert_eq!(r.concat_displacement, vec![10.0, 20.0, 30.0]);
    // dmap: every vert points at object idx 0.
    assert_eq!(r.concat_vert_dmap, vec![0u32, 0, 0]);
}

#[test]
fn assemble_rod_object() {
    let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let c = vec![0.0; 9];
    let edges: [[u32; 2]; 2] = [[0, 1], [1, 2]];
    let obj = make_assemble_obj("rope", "rod", &v, &c, None, Some(&edges), None);
    let mut map = std::collections::HashMap::new();
    map.insert("rope".to_string(), vec![0i64, 1, 2]);
    let disp = vec![("rope".to_string(), [0.0, 0.0, 0.0])];
    let r = assemble_dyn_scene(&[obj], &map, 3, &disp, &[]).unwrap();
    assert_eq!(r.rod_count, 2);
    assert_eq!(r.concat_rod, vec![0u32, 1, 1, 2]);
}

#[test]
fn assemble_tet_object() {
    let v = vec![
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let c = vec![0.0; 12];
    let faces: [[u32; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    let tets: [[u32; 4]; 1] = [[0, 1, 2, 3]];
    let obj = make_assemble_obj(
        "block", "tet", &v, &c, Some(&faces), None, Some(&tets),
    );
    let mut map = std::collections::HashMap::new();
    map.insert("block".to_string(), vec![0i64, 1, 2, 3]);
    let disp = vec![("block".to_string(), [0.0, 0.0, 0.0])];
    let r = assemble_dyn_scene(&[obj], &map, 4, &disp, &[]).unwrap();
    // shell_count counts pure-shell triangles only; tet surfaces
    // come in phase 4 and are not part of shell_count.
    assert_eq!(r.shell_count, 0);
    assert_eq!(r.concat_tri.len() / 3, 4);
    assert_eq!(r.concat_tet, vec![0u32, 1, 2, 3]);
}

#[test]
fn assemble_collider_flag_when_pinned_endpoints() {
    let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let c = vec![0.0; 9];
    let faces: [[u32; 3]; 1] = [[0, 1, 2]];
    let pinned = vec![0i64, 1, 2];
    let mut obj =
        make_assemble_obj("sheet", "tri", &v, &c, Some(&faces), None, None);
    obj.pinned_indices = &pinned;
    let mut map = std::collections::HashMap::new();
    map.insert("sheet".to_string(), vec![0i64, 1, 2]);
    let disp = vec![("sheet".to_string(), [0.0, 0.0, 0.0])];
    let r = assemble_dyn_scene(&[obj], &map, 3, &disp, &[]).unwrap();
    assert_eq!(r.concat_tri_is_collider, vec![1u8]);
}

#[test]
fn assemble_uv_passthrough_for_shell() {
    let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let c = vec![0.0; 9];
    let faces: [[u32; 3]; 1] = [[0, 1, 2]];
    let uv: Vec<f64> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let mut obj = make_assemble_obj("s", "tri", &v, &c, Some(&faces), None, None);
    obj.uv = Some(&uv);
    let mut map = std::collections::HashMap::new();
    map.insert("s".to_string(), vec![0i64, 1, 2]);
    let disp = vec![("s".to_string(), [0.0, 0.0, 0.0])];
    let r = assemble_dyn_scene(&[obj], &map, 3, &disp, &[]).unwrap();
    assert_eq!(r.concat_uv, uv);
}

#[test]
fn assemble_cross_stitch_remaps() {
    let va = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let vb = vec![5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 5.0, 1.0, 0.0];
    let c = vec![0.0; 9];
    let fa: [[u32; 3]; 1] = [[0, 1, 2]];
    let fb: [[u32; 3]; 1] = [[0, 1, 2]];
    let oa = make_assemble_obj("a", "tri", &va, &c, Some(&fa), None, None);
    let ob = make_assemble_obj("b", "tri", &vb, &c, Some(&fb), None, None);
    let mut map = std::collections::HashMap::new();
    map.insert("a".to_string(), vec![0i64, 1, 2]);
    map.insert("b".to_string(), vec![3i64, 4, 5]);
    let disp = vec![
        ("a".to_string(), [0.0, 0.0, 0.0]),
        ("b".to_string(), [0.0, 0.0, 0.0]),
    ];
    // 6-wide: source bary over a's verts (slots 0..2 -> src_map),
    // target bary over b's verts (slots 3..5 -> tgt_map).
    let cs_ind = vec![0i64, 1, 2, 0, 1, 2];
    let cs_w = vec![0.2, 0.3, 0.5, 0.5, 0.25, 0.25];
    let cs = CrossStitch {
        source_name: "a",
        target_name: "b",
        ind: &cs_ind,
        weights: &cs_w,
        k: 1,
        stitch_stiffness: 2.5,
    };
    let r = assemble_dyn_scene(&[oa, ob], &map, 6, &disp, &[cs]).unwrap();
    // source a[0,1,2] -> 0,1,2; target b[0,1,2] -> 3,4,5.
    assert_eq!(r.concat_stitch_ind, vec![0i64, 1, 2, 3, 4, 5]);
    assert_eq!(r.concat_stitch_w, cs_w);
    assert_eq!(r.concat_stitch_stiffness, vec![2.5f32]);
}

#[test]
fn assemble_static_scene_offsets_faces() {
    let v1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let v2 = vec![5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 5.0, 1.0, 0.0];
    let f1: [[u32; 3]; 1] = [[0, 1, 2]];
    let f2: [[u32; 3]; 1] = [[0, 1, 2]];
    let so1 = AssembleStaticObject {
        name: "g1",
        vertex_world: &v1,
        color: [0.5, 0.5, 0.5],
        faces: Some(&f1),
        n_faces: 1,
        position: [0.0, 0.0, 0.0],
    };
    let so2 = AssembleStaticObject {
        name: "g2",
        vertex_world: &v2,
        color: [0.5, 0.5, 0.5],
        faces: Some(&f2),
        n_faces: 1,
        position: [0.0, 0.0, 0.0],
    };
    let disp = vec![
        ("g1".to_string(), [0.0, 0.0, 0.0]),
        ("g2".to_string(), [0.0, 0.0, 0.0]),
    ];
    let r = assemble_static_scene(&[so1, so2], &disp).unwrap();
    // Six vertices, two triangles with offsets (0,1,2) and (3,4,5).
    assert_eq!(r.static_vert.len(), 18);
    assert_eq!(r.static_tri, vec![0u32, 1, 2, 3, 4, 5]);
    assert_eq!(r.static_vert_dmap, vec![0u32, 0, 0, 1, 1, 1]);
}

#[test]
fn shrink_strain_limit_conflict_detected() {
    let sx = vec![1.0, 0.5];
    let sy = vec![1.0, 1.0];
    let sl = vec![0.0, 0.1];
    let r = check_shell_shrink_strain_limit_conflict(&sx, &sy, &sl);
    assert_eq!(r, Some((1, 0.5, 1.0, 0.1)));
}

#[test]
fn shrink_strain_limit_no_conflict_when_strain_zero() {
    let sx = vec![0.5, 0.5];
    let sy = vec![1.0, 1.0];
    let sl = vec![0.0, 0.0];
    let r = check_shell_shrink_strain_limit_conflict(&sx, &sy, &sl);
    assert_eq!(r, None);
}
