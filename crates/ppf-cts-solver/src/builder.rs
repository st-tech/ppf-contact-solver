// File: builder.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::cvec::*;
use super::cvecvec::*;
use super::data::{self, *};
use super::{mesh::Mesh, MeshSet, SimArgs};
use more_asserts::*;
use na::{vector, Matrix2, Matrix2xX, Matrix3, Matrix3x2, Matrix3xX};
use rayon::prelude::*;
use std::collections::HashMap;

pub struct Props {
    pub edge: Vec<EdgeProp>,
    pub face: Vec<FaceProp>,
    pub tet: Vec<TetProp>,
    pub edge_params: Vec<EdgeParam>,
    pub face_params: Vec<FaceParam>,
    pub tet_params: Vec<TetParam>,
    /// Granular (SAND) scalar material knobs, present only when the scene
    /// contains a faceless particle cloud. A grain is a loose vertex with no
    /// incident element; this supplies its mass and contact param (the
    /// existing per-element param paths produce nothing for it). `None` for a
    /// scene with no SAND object, leaving every non-SAND path byte-identical.
    pub sand: Option<SandParams>,
    /// This scene animates at least one material, so every param table is
    /// built one entry per element with `param_index` fixed to the identity.
    ///
    /// Deduplication collapses by VALUE, and the per-frame path rewrites those
    /// values, so a deduplicated table's size and order would move between
    /// frames while each prop's `param_index` is written once here. The cost
    /// is one table entry per element instead of one per distinct material,
    /// and it is paid only by a scene that actually animates something.
    pub animated_materials: bool,
}

/// Scalar material parameters shared by every grain in a SAND cloud. Read by
/// `build` to inject grain mass and a per-grain contact `VertexParam`.
#[derive(Clone, Copy)]
pub struct SandParams {
    /// Mass of a single grain, kilograms. Must be positive (a zero-mass grain
    /// has zero inertia force/Hessian and singularizes the solve).
    pub particle_mass: f32,
    /// Grain contact radius (= the `contact-offset` knob): the per-grain
    /// barrier offset, so two grains repel below center distance
    /// `offset_i + offset_j`.
    pub grain_radius: f32,
    /// Barrier activation gap (`contact-gap`).
    pub contact_gap: f32,
    /// Inter-grain Coulomb friction coefficient.
    pub friction: f32,
}

/// `HingeProp::uv_edge_sin2` for a hinge with no usable UV direction, either
/// because the mesh carries no UV at all or because both incident faces have a
/// degenerate UV edge. Negative so it cannot collide with a real `sin^2`, which
/// lies in `[0, 1]`; the kernel reads it as "isotropic".
/// The build refuses a scene that pairs it with non-unit bending ratios, so it
/// never silently discards an anisotropy the user asked for.
pub(crate) const NO_UV_EDGE_DIRECTION: f32 = -1.0;

/// Signed dihedral angle between the two faces sharing the edge
/// `v0-v1` with opposite vertices `v2` and `v3`. Matches the device-side
/// `face_dihedral_angle` after `remap(hinge)` in `dihedral_angle.hpp`:
/// that kernel reorders the hinge to (v[2], v[1], v[0], v[3]) and computes
/// the angle, so here the hinge column `(h0, h1, h2, h3)` maps to
/// `v0 = h2, v1 = h1, v2 = h0, v3 = h3`. Callers pass the reordered verts.
fn signed_dihedral_angle(
    v0: &na::Vector3<f32>,
    v1: &na::Vector3<f32>,
    v2: &na::Vector3<f32>,
    v3: &na::Vector3<f32>,
) -> f32 {
    let n1 = (v1 - v0).cross(&(v2 - v0));
    let n2 = (v2 - v3).cross(&(v1 - v3));
    let n1_sq = n1.norm_squared();
    let n2_sq = n2.norm_squared();
    if n1_sq <= 0.0 || n2_sq <= 0.0 {
        return 0.0;
    }
    let dot = n1.dot(&n2) / (n1_sq * n2_sq).sqrt();
    let angle = dot.clamp(-1.0, 1.0).acos();
    if n2.cross(&n1).dot(&(v1 - v2)) < 0.0 {
        -angle
    } else {
        angle
    }
}

/// A streamed rest element whose smallest-to-largest singular-value ratio
/// drops below this is *excluded* from the energy rather than used.
///
/// A captured deformation (especially a rotation) can fold an element's *rest*
/// shape through a flat, near-singular configuration. Inversion itself is
/// harmless: the elastic model is SVD-based and handles `det < 0`. The problem
/// is singularity: as a singular value of the rest edge matrix goes to zero,
/// `inv_rest = (rest edge matrix)^-1` blows up and diverges the linear solve.
///
/// Clamping the singular values up would keep the element but fatten it,
/// visibly suppressing the legitimate stretch at a bend (a tet thinning at a
/// bend can reach a high aspect ratio; fTetWild already emits tets near
/// condition ~20). Instead, when an element crosses this ratio it is dropped
/// from the elastic/strain energy entirely (the caller flags it in the
/// returned `exclude_face`/`exclude_tet` mask, which `update_rest_shape`
/// writes into the dedicated per-element `rest_excluded` flag that the elastic
/// and strain-limiting kernels gate on, independently of and never
/// aliasing the kinematic `fixed` flag) and a benign
/// identity `inv_rest` is stored. Its vertices are still governed by inertia,
/// the pull constraint, and their other (non-singular) elements, so every
/// element that is *not* singular keeps its exact rest and full stretch.
const REST_SHAPE_EXCLUDE_RATIO: f32 = 0.01;

/// A BUILD-TIME rest element whose smallest-to-largest singular-value ratio
/// drops below this is a hard error, `sqrt(f32::EPSILON)`.
///
/// The elastic Hessian is quadratic in `inv_rest`, so a rest matrix conditioned
/// at ratio `r` carries its entries with a relative precision of about
/// `f32::EPSILON / r^2`. At `r = sqrt(f32::EPSILON)` that reaches 1, meaning no
/// correct digit is left, and below it the Hessian is rounding noise: it
/// overflows to a non-finite block, the block-Jacobi preconditioner inverts
/// garbage, and the linear solve reports a non-finite `p^T A p` at its first
/// iteration naming no geometry at all (issue #144, whose reporter's mesh
/// carries sixteen such faces at ratios 1.5e-07 to 2.9e-06 while every other
/// face of it sits at 1.3e-03 or above).
///
/// A test for exact singularity cannot see any of them: none of those sixteen
/// faces is singular in fp32, they all invert, and the inverse is finite. The
/// square root is what makes the ratio test see them, and it is a property of
/// the arithmetic rather than a tolerance: it is where the squaring consumes
/// every digit fp32 has.
///
/// It admits far more than [`REST_SHAPE_EXCLUDE_RATIO`] does, because the two
/// govern different decisions. That one drops a STREAMED element a captured
/// deformation folded flat, where dropping costs only that element's stretch
/// and a false positive is cheap. This one refuses to build at all, so it fires
/// only where the arithmetic genuinely has nothing left: on the same mesh the
/// 0.01 ratio would also reject four faces that run correctly.
const REST_SHAPE_MIN_CONDITION: f32 = 3.452_669_8e-4;

/// Exact inverse of a 3x3 rest edge matrix, or `(identity, true)` when the
/// matrix is near-singular (min/max singular-value ratio below
/// [`REST_SHAPE_EXCLUDE_RATIO`]) so the caller excludes the element from the
/// energy. Only the singular values are needed for the test (no `U`/`V`).
fn invert_or_exclude3(mat: &Matrix3<f32>) -> (Matrix3<f32>, bool) {
    let sv = mat.singular_values(); // descending
    let (smax, smin) = (sv[0], sv[2]);
    // A NaN/Inf singular value (e.g. a `normalize()` of a zero/collinear edge
    // upstream) must exclude: every comparison with NaN is false, so without
    // this guard a non-finite rest matrix would fall through to `try_inverse`
    // and leave a NaN `inv_rest` active in the energy.
    if !smax.is_finite()
        || !smin.is_finite()
        || smax <= 0.0
        || smin < REST_SHAPE_EXCLUDE_RATIO * smax
    {
        return (Matrix3::identity(), true);
    }
    (mat.try_inverse().unwrap_or_else(Matrix3::identity), false)
}

/// 2x2 analog of [`invert_or_exclude3`] for shell-face rest matrices.
fn invert_or_exclude2(mat: &Matrix2<f32>) -> (Matrix2<f32>, bool) {
    let sv = mat.singular_values(); // descending
    let (smax, smin) = (sv[0], sv[1]);
    // See `invert_or_exclude3`: a NaN/Inf singular value must exclude, since
    // every comparison with NaN is false and would otherwise pass through.
    if !smax.is_finite()
        || !smin.is_finite()
        || smax <= 0.0
        || smin < REST_SHAPE_EXCLUDE_RATIO * smax
    {
        return (Matrix2::identity(), true);
    }
    (mat.try_inverse().unwrap_or_else(Matrix2::identity), false)
}

/// Tangent-plane rest edge matrix for one shell face, built from its two rest
/// edge vectors `x1 - x0` and `x2 - x0`.
///
/// The second row is non-finite exactly when the triangle has no tangent
/// frame: the face normal `dx0 x dx1` is the zero vector for a zero-area
/// (collinear or coincident) rest triangle, so the second tangent axis
/// normalizes a zero vector, and `normalize()` of a zero vector is NaN rather
/// than an error. The first row still normalizes the first edge and stays
/// finite, which is why a caller has to test the whole matrix rather than its
/// determinant; [`invert_rest_or_panic2`] and [`invert_or_exclude2`] are the
/// two that do.
fn shell_rest_tangent_matrix(dx: &Matrix3x2<f32>) -> Matrix2<f32> {
    let n = dx.column(0).cross(&dx.column(1));
    let e2 = n.cross(&dx.column(0)).normalize();
    let proj_mat = Matrix3x2::<f32>::from_columns(&[dx.column(0).normalize(), e2]).transpose();
    proj_mat * dx
}

/// Exact inverse of a build-time shell-face rest matrix, or a panic naming the
/// face, the three vertices, and the conditioning that made it unusable.
///
/// The test is the singular-value ratio against [`REST_SHAPE_MIN_CONDITION`],
/// and it has to be, because the two ways a rest triangle goes bad look
/// completely different here and only one of them is singular:
///
/// * A ZERO-AREA triangle arrives with a NaN row in `d_mat` (see
///   [`shell_rest_tangent_matrix`]): the face normal is the zero vector and
///   `normalize()` of that is NaN. Its determinant is NaN too, and every
///   comparison `try_inverse` makes against zero is false for NaN, so it hands
///   back a NaN inverse rather than `None`. A finiteness test catches this one;
///   `try_inverse` alone does not.
/// * A NEAR-collinear triangle arrives entirely finite and inverts cleanly. No
///   finiteness test and no `None` from `try_inverse` can see it. What it
///   carries instead is magnitude: `inv_rest` scales as `1 / smin`, so a face
///   whose ratio is 2e-07 has entries around 8.9e+07 where a healthy face of
///   the same mesh has 20, and the Hessian squares that.
///
/// The ratio test covers both: a NaN singular value fails every comparison, and
/// a finite ill-conditioned one fails the threshold. Both reach the device as a
/// non-finite elastic Hessian otherwise, which surfaces only once the linear
/// solve reports a non-finite quantity, naming neither the object nor the face.
///
/// This runs on the FINAL `d_mat`, after the UV rotation and the per-axis
/// shrink, because that is the matrix being inverted. A rotation leaves the
/// singular values alone; a shrink scales them, so an extreme `shrink-x` /
/// `shrink-y` ratio is refused here on the same grounds as bad geometry.
///
/// The case the zero-area half is live for is a degenerate REST shape under a
/// sound current one. `main.rs` asserts `triutils::face_areas(&mesh.vertex, ..)`
/// before the build, so a zero-area face in the CURRENT positions aborts earlier
/// with its own face index; `compute_inv_rest` reads
/// `mesh.rest_vertex.unwrap_or(&mesh.vertex)`, which that assertion never looks
/// at. The near-collinear half is live for the shipped geometry too, which is
/// what `face_areas` cannot see.
fn invert_rest_or_panic2(d_mat: &Matrix2<f32>, face: usize, f: [usize; 3]) -> Matrix2<f32> {
    let sv = d_mat.singular_values(); // descending
    let (smax, smin) = (sv[0], sv[1]);
    // NaN fails all four comparisons, which is the point: this single test
    // rejects a non-finite matrix and a merely ill-conditioned one alike.
    if smax.is_finite()
        && smin.is_finite()
        && smax > 0.0
        && smin >= REST_SHAPE_MIN_CONDITION * smax
    {
        if let Some(inv) = d_mat.try_inverse() {
            return inv;
        }
    }
    let conditioning = describe_conditioning(smax, smin);
    let matrix = ascii_matrix2(d_mat);
    panic!(
        "Degenerate shell face {face} on vertices {}, {}, {}: the rest \
         triangle's tangent matrix {conditioning}, so the face has no usable \
         rest shape for an fp32 elastic Hessian. The three rest positions are \
         collinear, coincident, or close enough to collinear that the inverse \
         rest shape is rounding noise. Triangulate the source polygon along its \
         other diagonal, weld its coincident vertices, or remove it. A shell \
         group's `shrink-x` / `shrink-y` scale this matrix too, so check them \
         when the geometry itself looks sound.\nrest tangent matrix \
         (row-major): {matrix}",
        f[0], f[1], f[2]
    )
}

/// A small rest matrix rendered on ONE line in ASCII, row-major.
///
/// nalgebra's `Display` draws the matrix inside box-drawing characters
/// (U+250C and its family). Those have no encoding in the ANSI codepage a
/// Windows console and a default-encoded Python reader still use, so a
/// consumer there fails ON the panic text rather than reporting it: the rig
/// captures `'charmap' codec can't encode character '\u250c'` and the
/// geometry that caused the refusal is never named. Losing the diagnostic is
/// worst exactly where it is hardest to reproduce, so the matrix is spelled
/// out here instead. Every entry the box form carried is still printed.
fn ascii_matrix2(m: &Matrix2<f32>) -> String {
    format!(
        "[{:.6e}, {:.6e}; {:.6e}, {:.6e}]",
        m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]
    )
}

/// 3x3 analog of [`ascii_matrix2`], for the tet rest matrix.
fn ascii_matrix3(m: &Matrix3<f32>) -> String {
    let mut out = String::from("[");
    for i in 0..3 {
        if i > 0 {
            out.push_str("; ");
        }
        for j in 0..3 {
            if j > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("{:.6e}", m[(i, j)]));
        }
    }
    out.push(']');
    out
}

/// How a rest matrix failed [`REST_SHAPE_MIN_CONDITION`], as a phrase.
///
/// A zero-area or otherwise non-finite matrix has no ratio: `smin / smax` is
/// NaN there, and printing it alongside "below the threshold" states a
/// comparison that never happened, since NaN fails every one.
fn describe_conditioning(smax: f32, smin: f32) -> String {
    if smax.is_finite() && smin.is_finite() && smax > 0.0 {
        format!(
            "is conditioned at {:.3e} (singular values {smax:.6e} and \
             {smin:.6e}), below the {REST_SHAPE_MIN_CONDITION:.3e} \
             an fp32 elastic Hessian can carry",
            smin / smax
        )
    } else {
        format!(
            "is degenerate: its singular values are {smax:.6e} and \
             {smin:.6e}, so it has no conditioning to compare against the \
             {REST_SHAPE_MIN_CONDITION:.3e} floor"
        )
    }
}

/// 3x3 analog of [`invert_rest_or_panic2`] for build-time tet rest matrices.
///
/// The hole is the same shape here: a needle, sliver or cap tet is finite and
/// `try_inverse` succeeds on it, so a finiteness test passes it and its
/// `inv_rest3x3` reaches the device to be squared into the elastic Hessian.
/// Only the shell path has been reported against, but the two arms build the
/// same quantity for the same energy out of the same arithmetic, so both carry
/// the test.
///
/// The panic names the tet and its four vertices, because the linear solve's
/// own report names no geometry at all and a matrix on its own locates nothing.
fn invert_rest_or_panic3(mat: &Matrix3<f32>, tet: usize, t: [usize; 4]) -> Matrix3<f32> {
    let sv = mat.singular_values(); // descending
    let (smax, smin) = (sv[0], sv[2]);
    // NaN fails all four comparisons; see `invert_rest_or_panic2`.
    if smax.is_finite()
        && smin.is_finite()
        && smax > 0.0
        && smin >= REST_SHAPE_MIN_CONDITION * smax
    {
        if let Some(inv) = mat.try_inverse() {
            return inv;
        }
    }
    let conditioning = describe_conditioning(smax, smin);
    let matrix = ascii_matrix3(mat);
    panic!(
        "Degenerate tetrahedron {tet} on vertices {}, {}, {}, {}: the rest \
         edge matrix {conditioning}. The four rest positions are \
         coplanar, coincident, or close enough to coplanar that the inverse \
         rest shape is rounding noise, so the element has no usable rest \
         shape.\nrest edge matrix (row-major): {matrix}",
        t[0], t[1], t[2], t[3]
    )
}

/// Compute per-element inverse rest matrices from a rest-pose vertex set.
///
/// `inv_rest2x2` (per shell face) embeds the tangent-plane projection plus the
/// optional UV-aligned rotation and per-axis shrink; `inv_rest3x3` (per tet)
/// embeds the per-tet shrink. Extracted so the same formula runs both at build
/// time and per frame when a time-varying rest shape is streamed (the
/// `rest_vert_schedule` path), keeping init and animated rest shapes identical.
///
/// When `exclude_singular` is set (the streamed-rest-shape path), near-singular
/// elements are flagged in the returned per-element `exclude` masks (1 = drop
/// from the energy; see [`REST_SHAPE_EXCLUDE_RATIO`]) and given an identity
/// `inv_rest`. At build time pass `false`: the masks come back all-zero, and a
/// degenerate element is a hard error rather than an exclusion, because build
/// geometry the caller supplied degenerate is a defect in that geometry, not a
/// pose a captured deformation happened to pass through.
pub(crate) fn compute_inv_rest(
    rest_v: &Matrix3xX<f32>,
    mesh: &MeshSet,
    face_props: &[FaceProp],
    tet_props: &[TetProp],
    face_params: &[FaceParam],
    tet_params: &[TetParam],
    exclude_singular: bool,
) -> (Vec<Matrix2<f32>>, Vec<Matrix3<f32>>, Vec<u8>, Vec<u8>) {
    let uv = &mesh.uv;
    let (inv_rest2x2, exclude_face): (Vec<Matrix2<f32>>, Vec<u8>) =
        (0..mesh.mesh.mesh.shell_face_count)
            .into_par_iter()
            .map(|i| {
                let f = mesh.mesh.mesh.face.column(i);
                // Compute rest-pose from 3D geometry projected into tangent plane
                let (x0, x1, x2) = (
                    rest_v.column(f[0]),
                    rest_v.column(f[1]),
                    rest_v.column(f[2]),
                );
                let dx = Matrix3x2::<f32>::from_columns(&[
                    (x1 - x0).map(f32::from),
                    (x2 - x0).map(f32::from),
                ]);
                let d_mat = shell_rest_tangent_matrix(&dx);
                // When UV data exists, rotate d_mat to align with the UV first-edge
                // direction and apply shrink. This preserves UV orientation for
                // Baraff-Witkin anisotropy while keeping F ≈ I at rest.
                let d_mat = if let Some(uv) = uv.as_ref() {
                    let uv_e0 = (uv[i].column(1) - uv[i].column(0)).map(f32::from);
                    let lu0 = uv_e0.norm();
                    if lu0 > 0.0 {
                        let uv_dir = uv_e0 / lu0;
                        // Rotation from (1,0) to uv_dir
                        let rot = Matrix2::<f32>::new(uv_dir[0], -uv_dir[1], uv_dir[1], uv_dir[0]);
                        let face_param = &face_params[face_props[i].param_index as usize];
                        debug_assert!(
                            face_param.shrink_x > 0.0 && face_param.shrink_y > 0.0,
                            "FaceParam::shrink_x/shrink_y uninitialized"
                        );
                        let shrink =
                            Matrix2::<f32>::new(face_param.shrink_x, 0.0, 0.0, face_param.shrink_y);
                        shrink * rot * d_mat
                    } else {
                        d_mat
                    }
                } else {
                    d_mat
                };
                if exclude_singular {
                    let (inv, ex) = invert_or_exclude2(&d_mat);
                    (inv, ex as u8)
                } else {
                    (invert_rest_or_panic2(&d_mat, i, [f[0], f[1], f[2]]), 0u8)
                }
            })
            .unzip();

    let tet_columns: Vec<_> = mesh.mesh.mesh.tet.column_iter().collect();
    let (inv_rest3x3, exclude_tet): (Vec<Matrix3<f32>>, Vec<u8>) = tet_columns
        .into_par_iter()
        .enumerate()
        .map(|(i, tet)| {
            let (x0, x1, x2, x3) = (
                rest_v.column(tet[0]),
                rest_v.column(tet[1]),
                rest_v.column(tet[2]),
                rest_v.column(tet[3]),
            );
            let tet_param = &tet_params[tet_props[i].param_index as usize];
            let s = tet_param.shrink;
            debug_assert!(s > 0.0, "TetParam::shrink uninitialized");
            let mat = s * Matrix3::<f32>::from_columns(&[
                (x1 - x0).map(f32::from),
                (x2 - x0).map(f32::from),
                (x3 - x0).map(f32::from),
            ]);
            if exclude_singular {
                // A streamed rest frame can fold a tet through a near-singular
                // shape. Rather than clamp (which fattens it and suppresses the
                // bend's stretch), drop it from the energy: the caller flags it
                // in the `exclude_tet` mask (decoded into the dedicated
                // `rest_excluded` flag) and its verts are governed by inertia,
                // the pull, and their other non-singular tets.
                let (inv, ex) = invert_or_exclude3(&mat);
                (inv, ex as u8)
            } else {
                (
                    invert_rest_or_panic3(&mat, i, [tet[0], tet[1], tet[2], tet[3]]),
                    0u8,
                )
            }
        })
        .unzip();

    (inv_rest2x2, inv_rest3x3, exclude_face, exclude_tet)
}

/// Largest fraction of a frame a single substep may span; keeps dt strictly
/// below one full frame (1/fps) so prev-vertex extrapolation never lands on
/// or past the next frame boundary.
const MAX_SUBSTEP_FRAME_FRACTION: f32 = 0.9999;

/// Clamp a requested substep dt to stay strictly under one frame.
fn clamp_substep_dt(dt: f32, fps: f64) -> f32 {
    dt.min(MAX_SUBSTEP_FRAME_FRACTION / fps as f32)
}

/// Per-body and per-vertex PDRD inputs threaded from the scene.
///
/// `body_rows` is `PDRD_BODY_ROW_LEN` (23) f32 per body. The first two
/// slots are `f32` casts of `(vertex_start, vertex_count)`, which index
/// into the flat `pdrd_vert_list` (NOT the global vertex array). Then
/// come `volume`, `centroid[3]`, `rest_gram_inv[9 row-major]`,
/// `mass_per_vertex`, then the joint block `joint_mode`, `joint_axis[3]`
/// (world axle), `joint_pin[3]` (world pivot). Empty when the scene
/// contains no PDRD bodies.
///
/// `vert_index` is per global vertex; 1-based body id, 0 = not PDRD.
///
/// `vert_list` is the flat list of global vertex indices that
/// belong to PDRD bodies, body-major order. Indexed by
/// `[body.vertex_start, body.vertex_start + body.vertex_count)`.
///
/// `rest_centered` is 3 f32 per entry of `vert_list`, same length
/// (3 * vert_list.len()): the PDRD vertex's centered rest position
/// `ȳₘ = x̄ₘ − c̄_body`.
pub struct PdrdSceneData<'a> {
    pub body_rows: &'a [f32],
    pub vert_index: &'a [u32],
    pub vert_list: &'a [u32],
    pub rest_centered: &'a [f32],
}

/// Static aggregate-lock inputs threaded from `Scene`.
///
/// Each axis table has one normalized solver-space axis per displacement
/// group, paired with a mode table. THE MODE CARRIES THE ENABLE BIT: in an
/// axis mode a zero vector disables that component, and in an all-axes mode
/// the axis has no meaning and is required to be exactly zero. `vert_dmap`
/// assigns every dynamic vertex to its displacement group, so the builder can
/// retain only enabled physical-mass groups in the CUDA dataset.
pub struct LockSceneData<'a> {
    pub translation_axes: &'a [Vec3f],
    pub translation_modes: &'a [u32],
    pub rotation_axes: &'a [Vec3f],
    pub rotation_modes: &'a [u32],
    pub vert_dmap: &'a [u32],
}

fn normalized_or_disabled_lock_axis(axis: Vec3f, dmap_index: usize, component: &str) -> Vec3f {
    if axis == Vec3f::zeros() {
        return axis;
    }
    assert!(
        axis.iter().all(|value| value.is_finite()),
        "{component} lock displacement group {dmap_index} has a non-finite axis"
    );
    let norm = axis.norm();
    assert!(
        norm.is_finite() && (norm - 1.0).abs() <= 64.0 * f32::EPSILON,
        "{component} lock displacement group {dmap_index} must have a normalized nonzero axis"
    );
    axis / norm
}

fn validate_translation_lock_mode(mode: u32, dmap_index: usize) {
    assert!(
        mode == TRANSLATION_LOCK_AXIS || mode == TRANSLATION_LOCK_ALL,
        "translation lock displacement group {dmap_index} has invalid mode {mode}"
    );
}

fn validate_rotation_lock_mode(mode: u32, dmap_index: usize) {
    assert!(
        mode == ROTATION_LOCK_ALLOW_ONLY
            || mode == ROTATION_LOCK_PROHIBIT_AXIS
            || mode == ROTATION_LOCK_ALL,
        "rotation lock displacement group {dmap_index} has invalid mode {mode}"
    );
}

/// Is this half of the lock on, and is its record spelled canonically?
///
/// An all-axes mode is on regardless of its axis, which must be exactly zero
/// because the axis has no meaning there. An axis mode is on exactly when its
/// axis is non-zero. Reading enablement off the axis alone would silently
/// disable every all-axes lock while the UI still reported it as set, so this
/// is the ONLY enable test the builder uses.
fn lock_component_enabled(
    axis: Vec3f,
    mode: u32,
    all_mode: u32,
    dmap_index: usize,
    component: &str,
) -> bool {
    if mode == all_mode {
        assert!(
            axis == Vec3f::zeros(),
            "{component} lock displacement group {dmap_index} selects all axes, so its axis \
             must be exactly zero (got {axis:?}); the axis carries no meaning in an all-axes mode"
        );
        return true;
    }
    axis != Vec3f::zeros()
}

pub fn build(
    sim_args: &SimArgs,
    mesh: &MeshSet,
    velocity: &Matrix3xX<f32>,
    props: &mut Props,
    constraint: Constraint,
    pdrd: PdrdSceneData<'_>,
    lock_data: LockSceneData<'_>,
) -> data::DataSet {
    let dt = clamp_substep_dt(sim_args.dt, sim_args.fps);
    // Every param table this function builds follows the same rule as the face
    // table: identity indices when the scene animates a material, so a
    // per-frame rewrite cannot invalidate an index written here.
    let animated_materials = props.animated_materials;
    let vertex = &mesh.vertex;
    let n_vert = vertex.ncols();
    assert_eq!(
        lock_data.vert_dmap.len(),
        n_vert,
        "aggregate-lock vert_dmap size mismatch"
    );
    assert_eq!(
        lock_data.translation_axes.len(),
        lock_data.rotation_axes.len(),
        "translation-lock and rotation-lock axis table size mismatch"
    );
    assert_eq!(
        lock_data.translation_axes.len(),
        lock_data.rotation_modes.len(),
        "rotation-lock axis and mode table size mismatch"
    );
    for (vertex, &group) in lock_data.vert_dmap.iter().enumerate() {
        assert!(
            (group as usize) < lock_data.translation_axes.len(),
            "aggregate-lock vert_dmap[{vertex}] = {group} has no axis entry"
        );
    }
    let shell_face_count = mesh.mesh.mesh.shell_face_count;
    let rod_count = mesh.mesh.mesh.rod_count;
    let neighbor = marshal_neighbor(&mesh.mesh.neighbor);

    // A shell or rod object can opt into a "reference rest angle": its bending
    // rest angles (shell hinge dihedral, or rod interior-vertex bend angle)
    // are computed from a reference shape (a topological copy whose vertices
    // were moved) instead of its own initial pose. `bend_rest_v` holds those
    // reference positions for the masked vertices and the initial vert
    // everywhere else, so a non-reference element reads the same positions it
    // always did. The mask both selects the positions and forces the
    // from-geometry path on for the reference object, overriding the group's
    // Rest Angle source for that object.
    let bend_rest_v = mesh.bend_rest_vertex.as_ref().unwrap_or(vertex);
    let bend_mask = &mesh.bend_rest_vertex_mask;

    // Update fixed flags based on vertex fix_index (parallelized)
    // Pre-collect fixed vertex indices into a HashSet for thread-safe access
    use std::collections::HashSet;
    let fixed_vertices: HashSet<usize> = constraint
        .fix
        .iter()
        .map(|pair| pair.index as usize)
        .collect();

    // Which vertices are pinned by a pin that asked for its intersections to
    // be tolerated. FIX and PULL both count: a pull pin holds its vertex only
    // to the extent of its own force, which is exactly the case issue #138
    // singles out as one an intersection should not stop.
    //
    // A vertex can be covered by more than one pin, and the answer must hold
    // for EVERY pin covering it. An allowance says this vertex's placement is
    // not the solver's to resolve; a second pin that made no such claim
    // contradicts it, so unanimity is required. Starting from "no pin has
    // spoken" also gives an unpinned vertex the right answer, false, which is
    // what makes the element rule below mean "all N vertices are pinned AND
    // every one of those pins allows it".
    let pin_allow_vertices: Vec<bool> = {
        let mut allow = vec![false; n_vert];
        let mut seen = vec![false; n_vert];
        let mut note = |v: usize, pin_allows: bool| {
            allow[v] = if seen[v] {
                allow[v] && pin_allows
            } else {
                seen[v] = true;
                pin_allows
            };
        };
        for pair in constraint.fix.iter() {
            note(pair.index as usize, pair.allow_intersection);
        }
        for pair in constraint.pull.iter() {
            note(pair.index as usize, pair.allow_intersection);
        }
        allow
    };
    props.face.par_iter_mut().enumerate().for_each(|(i, prop)| {
        prop.pin_allow_intersection = mesh
            .mesh
            .mesh
            .face
            .column(i)
            .iter()
            .all(|&j| pin_allow_vertices[j]);
    });
    props.edge.par_iter_mut().enumerate().for_each(|(i, prop)| {
        prop.pin_allow_intersection = mesh
            .mesh
            .mesh
            .edge
            .column(i)
            .iter()
            .all(|&j| pin_allow_vertices[j]);
    });

    props.face.par_iter_mut().enumerate().for_each(|(i, prop)| {
        if mesh
            .mesh
            .mesh
            .face
            .column(i)
            .iter()
            .all(|&j| fixed_vertices.contains(&j))
        {
            prop.fixed = true;
        }
    });
    props.edge.par_iter_mut().enumerate().for_each(|(i, prop)| {
        if mesh
            .mesh
            .mesh
            .edge
            .column(i)
            .iter()
            .all(|&j| fixed_vertices.contains(&j))
        {
            prop.fixed = true;
        }
    });
    props.tet.par_iter_mut().enumerate().for_each(|(i, prop)| {
        if mesh
            .mesh
            .mesh
            .tet
            .column(i)
            .iter()
            .all(|&j| fixed_vertices.contains(&j))
        {
            prop.fixed = true;
        }
    });

    // A face belongs to a STATIC collider when all of its vertices do, mirroring
    // the `fixed` derivation above: a face shared with anything else keeps its
    // elastic energy. Faces are the only element kind a collider can reach (the
    // decoder builds its pin shell with `add.tri`, so no rod edges and no tets),
    // and hinges inherit it from their two incident faces further down, exactly
    // as `all_fixed` does.
    if !mesh.collider_vertex_mask.is_empty() {
        assert_eq!(
            mesh.collider_vertex_mask.len(),
            n_vert,
            "collider_vertex_mask size mismatch"
        );
        let collider_vertices: HashSet<usize> = mesh
            .collider_vertex_mask
            .iter()
            .enumerate()
            .filter(|(_, &c)| c != 0)
            .map(|(i, _)| i)
            .collect();
        props.face.par_iter_mut().enumerate().for_each(|(i, prop)| {
            prop.collider = mesh
                .mesh
                .mesh
                .face
                .column(i)
                .iter()
                .all(|&j| collider_vertices.contains(&j));
        });
    }

    // Props now contains final props and params directly
    let edge_props = &props.edge;
    let face_props = &props.face;
    let tet_props = &props.tet;
    let edge_params = &props.edge_params;
    let face_params = &props.face_params;
    let tet_params = &props.tet_params;
    // SandParams is Copy; bind a local so the parallel VertexParam closure
    // below reads it without borrowing the `&mut props`. `None` for non-SAND
    // scenes, keeping every grain path inert there.
    let sand = props.sand;

    // Build vertex props and params
    let mut vertex_prop = vec![VertexProp::default(); n_vert];
    // rod-bend rest angle defaults to π (straight); mutated below for interior
    // rod vertices whose adjacent edge params request rest-from-geometry.
    for vp in vertex_prop.iter_mut() {
        vp.rest_bend_angle = std::f32::consts::PI;
        // Default::default() would leave this 0, a real object's index. An
        // unknown object must not compare equal to another unknown one.
        vp.object_index = crate::data::NO_OBJECT_INDEX;
    }
    let mut temp_vertex_params = Vec::new();
    for (i, pair) in constraint.fix.iter().enumerate() {
        vertex_prop[pair.index as usize].fix_index = (i + 1) as u32;
    }
    for (i, pair) in constraint.pull.iter().enumerate() {
        vertex_prop[pair.index as usize].pull_index = (i + 1) as u32;
    }

    // Same latch the face and edge props took above, now per vertex: it is
    // what the point-point tester reads, where the element IS the vertex.
    for (i, &allow) in pin_allow_vertices.iter().enumerate() {
        vertex_prop[i].pin_allow_intersection = allow;
    }

    // Stamp the source-object identity and the per-object intersection
    // tolerances. Both arrays are optional: an older session directory has
    // neither, which leaves every vertex at NO_OBJECT_INDEX with an empty
    // policy, so no allowance can be granted and reporting is unchanged.
    if !mesh.object_vertex_index.is_empty() {
        assert_eq!(
            mesh.object_vertex_index.len(),
            n_vert,
            "object_vertex_index size mismatch"
        );
        for (i, &oi) in mesh.object_vertex_index.iter().enumerate() {
            vertex_prop[i].object_index = oi;
        }
    }
    // Stamp the source-group identity. Absent, every vertex keeps the default
    // group 0, which is exactly right for a scene no allowance asks about
    // groups; one that does is required to carry the file below.
    if !mesh.group_vertex_index.is_empty() {
        assert_eq!(
            mesh.group_vertex_index.len(),
            n_vert,
            "group_vertex_index size mismatch"
        );
        for (i, &gi) in mesh.group_vertex_index.iter().enumerate() {
            vertex_prop[i].group_index = gi;
        }
    }
    if !mesh.intersect_policy.is_empty() {
        assert_eq!(
            mesh.intersect_policy.len(),
            n_vert,
            "intersect_policy size mismatch"
        );
        // An allowance the solver cannot evaluate must not be silently
        // ignored: without object identity, "same object" is unanswerable and
        // both material flags would quietly do nothing on a scene that asked
        // for them.
        assert!(
            !mesh.object_vertex_index.is_empty(),
            "a scene requests an intersection allowance but carries no \
             object_vert.bin, so self- and inter-object pairs cannot be told \
             apart; rebuild the scene"
        );
        // A bit outside the two the solver knows how to act on would be read
        // as "allows nothing" and change no verdict, so a frontend that grew a
        // third allowance without the solver growing its rule would silently
        // do nothing. Trap it here instead, where the offending vertex can be
        // named.
        let known = crate::data::INTERSECT_ALLOW_SELF
            | crate::data::INTERSECT_ALLOW_INTER_OBJECT
            | crate::data::INTERSECT_ALLOW_INTER_GROUP;
        for (i, &policy) in mesh.intersect_policy.iter().enumerate() {
            assert_eq!(
                policy & !known,
                0,
                "intersect_policy[{i}] = {policy:#04x} sets bits outside the \
                 {known:#04x} this solver evaluates; the frontend and \
                 data.rs disagree about the allowance bits"
            );
            vertex_prop[i].intersect_policy = policy;
        }
        // The inter-group allowance compares groups, so a scene that asks for
        // it without saying which group each vertex is in would compare every
        // vertex's default group with every other's and allow nothing.
        let asks_for_groups = mesh
            .intersect_policy
            .iter()
            .any(|&p| p & crate::data::INTERSECT_ALLOW_INTER_GROUP != 0);
        assert!(
            !asks_for_groups || !mesh.group_vertex_index.is_empty(),
            "a scene requests the inter-group intersection allowance but \
             carries no group_vert.bin, so two groups cannot be told apart; \
             rebuild the scene"
        );
    }

    // Stamp the STATIC-collider flag on each vertex. Contact and intersection
    // reporting skip a pair only when BOTH sides carry it, so a collider still
    // collides with every dynamic object; it just does not collide with itself
    // or with another collider. An empty mask (an older session directory, or
    // a scene with no colliders) leaves every vertex false.
    if !mesh.collider_vertex_mask.is_empty() {
        assert_eq!(
            mesh.collider_vertex_mask.len(),
            n_vert,
            "collider_vertex_mask size mismatch"
        );
        for (i, &is_collider) in mesh.collider_vertex_mask.iter().enumerate() {
            vertex_prop[i].collider = is_collider != 0;
        }
    }

    // Stamp the PDRD body id on each vertex. The scene-side
    // vertex-index slice is 1-based already; 0 means "not in any
    // PDRD body".
    if !pdrd.vert_index.is_empty() {
        assert_eq!(
            pdrd.vert_index.len(),
            n_vert,
            "PDRD vert_index size mismatch"
        );
        for (i, &bid) in pdrd.vert_index.iter().enumerate() {
            vertex_prop[i].pdrd_body_index = bid;
        }
    }

    // Populate rod-bend rest angle for interior rod vertices whose adjacent
    // rod-edges request rest-from-geometry. Matches the device-side gate the
    // rod-bend site is dispatched under: a vertex with exactly 2 rod edges and
    // 0 faces.
    for j in 0..n_vert {
        let adj_edges: Vec<usize> = mesh.mesh.neighbor.vertex.edge[j]
            .iter()
            .copied()
            .filter(|&ei| ei < rod_count)
            .collect();
        if adj_edges.len() != 2 {
            continue;
        }
        if !mesh.mesh.neighbor.vertex.face[j].is_empty() {
            continue;
        }
        // A masked interior vertex belongs to a rod that opted into a
        // reference rest angle, so it computes its rest bend angle from the
        // reference shape even when the group's Rest Angle source is Flat.
        let from_reference = !bend_mask.is_empty() && bend_mask[j];
        let from_geometry = from_reference
            || adj_edges.iter().any(|&ei| {
                let edge_prop = &edge_props[ei];
                edge_params[edge_prop.param_index as usize].bend_rest_from_geometry
            });
        if !from_geometry {
            continue;
        }
        let edge_0 = mesh.mesh.mesh.edge.column(adj_edges[0]);
        let edge_1 = mesh.mesh.mesh.edge.column(adj_edges[1]);
        let other_0 = if edge_0[0] == j { edge_0[1] } else { edge_0[0] };
        let other_1 = if edge_1[0] == j { edge_1[1] } else { edge_1[0] };
        // Subtract the stored vertex columns directly, which is the order the
        // runtime rod bend energy uses: `rod_bend_angle` in
        // `kernels/energy/model/rod_bend.kernel.cpp` forms both edge vectors as
        // differences of the positions it was handed. Reproducing that order
        // here is what makes the stored rest angle match the runtime theta, so
        // `theta - rest_angle` is zero at frame 0 and the rod does not twitch
        // with `bend_rest_from_geometry` on. `bend_rest_v` is the reference
        // rest shape for a reference rod (and the initial vert everywhere
        // else).
        let v_other_0 = bend_rest_v.column(other_0);
        let v_j = bend_rest_v.column(j);
        let v_other_1 = bend_rest_v.column(other_1);
        let e0 = (v_other_0 - v_j).map(f32::from);
        let e1 = (v_other_1 - v_j).map(f32::from);
        let n0 = e0.norm();
        let n1 = e1.norm();
        if n0 <= 0.0 || n1 <= 0.0 {
            continue;
        }
        let cos_theta = (e0.dot(&e1) / (n0 * n1)).clamp(-1.0, 1.0);
        vertex_prop[j].rest_bend_angle = cos_theta.acos();
    }

    // Aggregate vertex area from all faces (needed for wind/air forces on all surfaces)
    let area_contributions: Vec<(usize, f32)> = face_props
        .par_iter()
        .enumerate()
        .flat_map(|(i, face_prop)| {
            mesh.mesh
                .mesh
                .face
                .column(i)
                .iter()
                .map(|&j| (j, face_prop.area / 3.0))
                .collect::<Vec<_>>()
        })
        .collect();
    for (j, area) in area_contributions {
        vertex_prop[j].area += area;
    }

    // Aggregate vertex mass from faces (shell faces always, solid faces only if include_face_mass)
    let mass_contributions: Vec<(usize, f32)> = face_props
        .par_iter()
        .enumerate()
        .filter(|(i, _)| *i < shell_face_count || sim_args.include_face_mass)
        .flat_map(|(i, face_prop)| {
            mesh.mesh
                .mesh
                .face
                .column(i)
                .iter()
                .map(|&j| (j, face_prop.mass / 3.0))
                .collect::<Vec<_>>()
        })
        .collect();
    for (j, mass) in mass_contributions {
        vertex_prop[j].mass += mass;
    }

    // Aggregate vertex mass from edges (parallelized collection, sequential merge)
    let edge_contributions: Vec<(usize, f32)> = edge_props
        .par_iter()
        .enumerate()
        .filter(|(i, _)| *i < rod_count)
        .flat_map(|(i, edge_prop)| {
            mesh.mesh
                .mesh
                .edge
                .column(i)
                .iter()
                .map(|&j| (j, edge_prop.mass / 2.0))
                .collect::<Vec<_>>()
        })
        .collect();
    for (j, mass) in edge_contributions {
        vertex_prop[j].mass += mass;
    }

    // Aggregate vertex mass/volume from tets (parallelized collection, sequential merge)
    let tet_contributions: Vec<(usize, f32, f32)> = tet_props
        .par_iter()
        .enumerate()
        .flat_map(|(i, tet_prop)| {
            mesh.mesh
                .mesh
                .tet
                .column(i)
                .iter()
                .map(|&j| (j, tet_prop.mass / 4.0, tet_prop.volume / 4.0))
                .collect::<Vec<_>>()
        })
        .collect();
    for (j, mass, volume) in tet_contributions {
        vertex_prop[j].mass += mass;
        vertex_prop[j].volume += volume;
    }

    // PDRD bodies use a uniform per-vertex mass `mass_per_vertex`
    // chosen so the body's effective rotational inertia matches the
    // volumetric I_solid (trace-ratio scaling computed at scene
    // build). Scene.rs already zeros face_prop.mass on PDRD faces so
    // the face-mass aggregation above contributes nothing for these
    // vertices; this loop adds the correct mass on top.
    if !pdrd.body_rows.is_empty() {
        assert!(
            pdrd.body_rows.len() % PDRD_BODY_ROW_LEN == 0,
            "PDRD body_rows length {} not a multiple of {} in mass pass",
            pdrd.body_rows.len(),
            PDRD_BODY_ROW_LEN,
        );
        let n_bodies = pdrd.body_rows.len() / PDRD_BODY_ROW_LEN;
        for b in 0..n_bodies {
            let row = &pdrd.body_rows[PDRD_BODY_ROW_LEN * b..PDRD_BODY_ROW_LEN * (b + 1)];
            let vertex_start = row[0] as usize;
            let vertex_count = row[1] as usize;
            let mass_per_vertex = row[15];
            for k in vertex_start..(vertex_start + vertex_count) {
                let vidx = pdrd.vert_list[k] as usize;
                vertex_prop[vidx].mass += mass_per_vertex;
            }
        }
    }

    // A grain is a loose vertex with no incident element (no face, no
    // rod-edge, no tet); the face/rod/tet/PDRD mass aggregation above
    // contributes nothing to it, so without this pass its mass stays 0.
    // A zero-mass vertex has zero inertia force/Hessian, since
    // `kernels/energy/vertex_force.kernel.cpp` scales both by `mass`, so
    // gravity is inert and the diagonal block is singular. The same
    // predicate gates the per-grain VertexParam below. Matches the device
    // gate for an isolated vertex: no faces and no rod edges (the rod-bend
    // path also checks tet-free). `neighbor.vertex.edge` ranges over the
    // full rods+face-edges matrix, so `ei < rod_count` is the load-bearing
    // rod filter (mirrors the VertexParam aggregation below).
    let is_grain = |j: usize| -> bool {
        mesh.mesh.neighbor.vertex.face[j].is_empty()
            && mesh.mesh.neighbor.vertex.tet[j].is_empty()
            && mesh.mesh.neighbor.vertex.edge[j]
                .iter()
                .all(|&ei| ei >= rod_count)
    };
    if let Some(sand) = sand {
        for j in 0..n_vert {
            if is_grain(j) {
                vertex_prop[j].mass += sand.particle_mass;
                assert_gt!(
                    vertex_prop[j].mass,
                    0.0,
                    "grain vertex {j} has non-positive mass after the SAND pass"
                );
            }
        }
    }

    // A loose vertex sewn into the cloth (a pinned "hook", or any faceless
    // stitch endpoint) gets no face/edge/tet mass, so both its inertia term
    // and the pin-barrier mass/gap^2 term vanish, leaving a singular diagonal:
    // a stitch, a contact, or even global-solve coupling then pushes it to the
    // pin's ghat boundary, where the line search clamps every Newton step's toi
    // to ~0 (a silent Zeno hang). Give each massless stitch vertex the mass of
    // the element it is sewn to (the largest mass in its seam) - the build-time
    // analog of the contact barrier's static-side max-mass substitution
    // (`kernels/barrier/contact_stiffness.kernel.cpp`, which substitutes the
    // pair's largest mass for a zero one).
    for seam in constraint.stitch.iter() {
        let seam_mass = seam
            .index
            .iter()
            .map(|&i| vertex_prop[i as usize].mass)
            .fold(0.0f32, f32::max);
        if seam_mass > 0.0 {
            for &i in seam.index.iter() {
                let m = &mut vertex_prop[i as usize].mass;
                if *m == 0.0 {
                    *m = seam_mass;
                }
            }
        }
    }

    // Per-grain spin state for SAND rolling (see sand_rigid.hpp): angular
    // velocity starts at zero, and the stored inverse inertia is the ROLLING
    // (contact-point) generalized inverse inertia 1/(I_center + m r^2), not the
    // bare center inertia 1/((2/5) m r^2).
    //
    // Why the parallel-axis term m r^2: in the staggered (post-solve) rolling
    // path the spin update lags (the translational contact solve uses last
    // step's omega, then the converged friction torque spins omega post-solve).
    // When friction sticks, the grain
    // center is slaved to omega by the no-slip relation v = r (omega x n), so
    // the friction reaction that drives omega depends on omega's own change one
    // step later. That inter-step feedback has gain m r^2 / I_center = 2.5 for a
    // solid sphere (I_center = (2/5) m r^2), so integrating omega with the bare
    // center inertia is unconditionally unstable (omega and travel blow up,
    // independent of dt). Using the inertia about the contact point
    // I_eff = I_center + m r^2 = (7/5) m r^2 is the constrained rolling DOF's
    // true generalized inertia and drops the staggered gain to m r^2 / I_eff
    // = 5/7 < 1, so the spin update is unconditionally stable and omega
    // converges to a bounded rolling rate. Non-grain vertices keep inv_inertia
    // = 0 so the post-solve integrate skips them. Sized over every vertex so a
    // grain at global index j indexes directly.
    let grain_omega_vec = vec![Vec3f::new(0.0, 0.0, 0.0); n_vert];
    let grain_torque_vec = vec![Vec3f::new(0.0, 0.0, 0.0); n_vert];
    let grain_ang_stiff_vec = vec![0.0f32; n_vert];
    let grain_contact_normal_vec = vec![Vec3f::new(0.0, 0.0, 0.0); n_vert];
    // Buffers for the implicit (Schur-condensed) rolling path (see data.hpp).
    // grain_inv_inertia_center uses the BARE center inertia I_center = (2/5) m r^2
    // (the parallel-axis m r^2 re-emerges from the Schur condensation);
    // grain_inv_inertia keeps I_eff for the grain-grain staggered (post-solve)
    // rolling integrate. A/B/grot/omega_prev start at zero.
    let grain_omega_prev_vec = vec![Vec3f::new(0.0, 0.0, 0.0); n_vert];
    let grain_a_vec = vec![Matrix3::<f32>::zeros(); n_vert];
    let grain_b_vec = vec![Matrix3::<f32>::zeros(); n_vert];
    let grain_grot_vec = vec![Vec3f::new(0.0, 0.0, 0.0); n_vert];
    let mut grain_inv_inertia_vec = vec![0.0f32; n_vert];
    let mut grain_inv_inertia_center_vec = vec![0.0f32; n_vert];
    if let Some(sand) = sand {
        let r = sand.grain_radius;
        for j in 0..n_vert {
            if is_grain(j) {
                // I_eff = I_center + m r^2 = (2/5 + 1) m r^2 = (7/5) m r^2.
                let inertia = 1.4 * vertex_prop[j].mass * r * r;
                if inertia > 0.0 {
                    grain_inv_inertia_vec[j] = 1.0 / inertia;
                }
                // I_center = (2/5) m r^2, bare solid-sphere center inertia.
                let inertia_center = 0.4 * vertex_prop[j].mass * r * r;
                if inertia_center > 0.0 {
                    grain_inv_inertia_center_vec[j] = 1.0 / inertia_center;
                }
            }
        }
    }

    // Build vertex params by aggregating from face/edge params (parallelized)
    // Step 1: Parallel computation of vertex params
    let vertex_params_data: Vec<Option<VertexParam>> = (0..n_vert)
        .into_par_iter()
        .map(|j| {
            // Weighted average of ghat/offset/friction over neighbor faces and
            // rod-edges, matching make_collision_mesh()'s vertex path and the
            // in-file hinge path. Weighting is order-independent (unlike a
            // last-writer overwrite): faces contribute by area, rod-edges by
            // length (their 1D analog of area). A vertex with no qualifying
            // neighbor keeps weight_sum == 0 and yields None below.
            // One definition, shared with the per-frame animated path.
            let derived = averaged_vertex_param(
                &mesh.mesh.neighbor.vertex.face[j],
                &mesh.mesh.neighbor.vertex.edge[j],
                face_props,
                face_params,
                edge_props,
                edge_params,
                rod_count,
            );

            if let Some(vparam) = derived {
                Some(vparam)
            } else if let Some(sand) = sand.filter(|_| is_grain(j)) {
                // A SAND grain has no incident element so weight_sum == 0, but
                // it still needs a contact VertexParam: the dedup below then
                // assigns it a valid param_index (without this it stays 0,
                // an out-of-bounds read of vertex_params[0] in a pure cloud
                // where temp_vertex_params is otherwise empty), and feeds
                // offset = grain radius to the point-point barrier, which
                // reads it as `VertexParam::offset`. Friction is the
                // inter-grain coefficient.
                Some(VertexParam {
                    ghat: sand.contact_gap,
                    offset: sand.grain_radius,
                    friction: sand.friction,
                })
            } else {
                // No SandParams (or not a grain): a stray detached vertex in a
                // normal mesh keeps the original None, unchanged.
                None
            }
        })
        .collect();

    // Step 2: Sequential deduplication
    let mut vertex_param_map: HashMap<VertexParam, u32> = HashMap::new();
    for (j, vparam_opt) in vertex_params_data.into_iter().enumerate() {
        if let Some(vparam) = vparam_opt {
            let param_idx = if animated_materials {
                temp_vertex_params.push(vparam);
                (temp_vertex_params.len() - 1) as u32
            } else {
                dedup_param(&mut vertex_param_map, &mut temp_vertex_params, vparam)
            };
            vertex_prop[j].param_index = param_idx;
        }
    }

    // Build hinge props and params (parallelized)
    // Step 1: Parallel computation of hinge data. `bend_rest_v` / `bend_mask`
    // (the reference rest shape) are defined at the top of `build`.
    let hinge_columns: Vec<_> = mesh.mesh.mesh.hinge.column_iter().collect();
    let hinge_uv = mesh.uv.as_ref();
    let hinge_data: Vec<(f32, f32, f32, f32, bool, bool, HingeParam)> = hinge_columns
        .into_par_iter()
        .enumerate()
        .map(|(i, hinge)| {
            let x = (vertex.column(hinge[0]) - vertex.column(hinge[1])).map(f32::from);
            let length = x.norm();
            // One definition of the hinge average, shared with the
            // per-frame animated path so the two cannot diverge.
            let (hparam, area_sum, all_fixed, all_collider, from_geometry) =
                averaged_hinge_param(
                    &mesh.mesh.neighbor.hinge.face[i],
                    face_props,
                    face_params,
                );
            // Direction of the hinge's shared edge in the UV material frame,
            // as sin^2 of its angle from the UV X (warp) axis. Both incident
            // faces carry their own UV triangle and a seam or a mirrored
            // island can orient them differently, so combine them by area the
            // way every other quantity in this loop is combined. sin^2 has
            // period 180 degrees, so which way round the edge is traversed
            // does not matter and no orientation convention is needed.
            let uv_edge_sin2 = match hinge_uv {
                None => NO_UV_EDGE_DIRECTION,
                Some(uv) => {
                    let mut sin2_sum = 0.0f32;
                    let mut weight_sum = 0.0f32;
                    for &j in mesh.mesh.neighbor.hinge.face[i].iter() {
                        // `uv` is indexed over shell faces only; a hinge is
                        // always between two of them, but stay in bounds
                        // rather than trusting that from a distance.
                        if j >= uv.len() {
                            continue;
                        }
                        let f = mesh.mesh.mesh.face.column(j);
                        let corner = |v: usize| (0..3).find(|&k| f[k] == v);
                        let (Some(c0), Some(c1)) = (corner(hinge[0]), corner(hinge[1])) else {
                            continue;
                        };
                        let e = uv[j].column(c1) - uv[j].column(c0);
                        let len2 = e.norm_squared();
                        if len2 > 0.0 {
                            sin2_sum += face_props[j].area * (e[1] * e[1] / len2);
                            weight_sum += face_props[j].area;
                        }
                    }
                    if weight_sum > 0.0 {
                        sin2_sum / weight_sum
                    } else {
                        // Both incident faces have a degenerate UV edge, so
                        // this hinge has no material direction at all.
                        NO_UV_EDGE_DIRECTION
                    }
                }
            };
            // Anisotropic bending is meaningless without a direction, so a
            // scene that asks for it and cannot supply one is an authoring
            // error rather than something to quietly ignore. Checked only
            // when the ratios actually differ from isotropic, so a mesh with
            // no UV (or one broken UV triangle) stays perfectly usable as
            // long as it is not asking for anisotropy.
            if directional_bending_needs_uv(
                uv_edge_sin2,
                hparam.bend_warp,
                hparam.bend_weft,
            ) {
                panic!(
                    "Hinge {i} (edge {}-{}) requests directional bending \
                     (bend-warp={}, bend-weft={}) but has no usable UV direction. \
                     Give the mesh a UV map, or leave both at 0.0.",
                    hinge[0], hinge[1], hparam.bend_warp, hparam.bend_weft
                );
            }
            // A hinge belonging to a reference object (its shared edge is
            // masked) computes its rest angle from the reference shape even
            // when the group's Rest Angle source is Flat.
            let from_reference = !bend_mask.is_empty() && bend_mask[hinge[0]];
            let rest_angle = if from_geometry || from_reference {
                // Mirror the device-side `shell_bend_remap`
                // (`kernels/energy/model/shell_bend.kernel.cpp`) before taking
                // the dihedral angle: (h2, h1, h0, h3).
                let v0 = bend_rest_v.column(hinge[2]).map(f32::from);
                let v1 = bend_rest_v.column(hinge[1]).map(f32::from);
                let v2 = bend_rest_v.column(hinge[0]).map(f32::from);
                let v3 = bend_rest_v.column(hinge[3]).map(f32::from);
                signed_dihedral_angle(&v0, &v1, &v2, &v3)
            } else {
                0.0
            };
            (
                length,
                area_sum,
                rest_angle,
                uv_edge_sin2,
                all_fixed,
                all_collider,
                hparam,
            )
        })
        .collect();

    // Step 2: Sequential deduplication
    let mut temp_hinge_props = Vec::with_capacity(hinge_data.len());
    let mut temp_hinge_params = Vec::new();
    let mut hinge_param_map: HashMap<HingeParam, u32> = HashMap::new();
    for (length, area, rest_angle, uv_edge_sin2, all_fixed, all_collider, hparam) in hinge_data {
        let param_idx = if animated_materials {
            temp_hinge_params.push(hparam);
            (temp_hinge_params.len() - 1) as u32
        } else {
            dedup_param(&mut hinge_param_map, &mut temp_hinge_params, hparam)
        };
        temp_hinge_props.push(HingeProp {
            fixed: all_fixed,
            collider: all_collider,
            length,
            area,
            rest_angle,
            uv_edge_sin2,
            param_index: param_idx,
        });
    }

    // Unpack the per-body rows. The Python frontend packs each body
    // as `PDRD_BODY_ROW_LEN` floats (see `PdrdSceneData` doc).
    let mut pdrd_body_props: Vec<PdrdBodyProp> = Vec::new();
    if !pdrd.body_rows.is_empty() {
        assert!(
            pdrd.body_rows.len() % PDRD_BODY_ROW_LEN == 0,
            "PDRD body_rows length {} not a multiple of {}",
            pdrd.body_rows.len(),
            PDRD_BODY_ROW_LEN,
        );
        let n_bodies = pdrd.body_rows.len() / PDRD_BODY_ROW_LEN;
        for b in 0..n_bodies {
            let row = &pdrd.body_rows[PDRD_BODY_ROW_LEN * b..PDRD_BODY_ROW_LEN * (b + 1)];
            let vertex_start = row[0] as u32;
            let vertex_count = row[1] as u32;
            let volume = row[2];
            let rest_centroid = Vec3f::new(row[3], row[4], row[5]);
            // 9 floats laid out row-major.
            let rest_gram_inv = Mat3x3f::new(
                row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14],
            );
            let mass_per_vertex = row[15];
            // Joint block: mode, axle[3], pivot[3] (see PdrdBodyProp).
            let joint_mode = row[16] as u32;
            let joint_axis = Vec3f::new(row[17], row[18], row[19]);
            let joint_pin = Vec3f::new(row[20], row[21], row[22]);
            assert!(vertex_count > 0, "PDRD body {b} has empty vertex range");
            assert!(
                volume > 0.0,
                "PDRD body {b} has non-positive volume {volume}"
            );
            assert!(
                mass_per_vertex > 0.0,
                "PDRD body {b} has non-positive mass_per_vertex {mass_per_vertex}"
            );
            pdrd_body_props.push(PdrdBodyProp {
                rest_centroid,
                rest_gram_inv,
                volume,
                vertex_start,
                vertex_count,
                mass_per_vertex,
                joint_mode,
                joint_axis,
                joint_pin,
            });
        }
    }

    // Compact enabled displacement-group locks into physical-mass groups. A
    // massless vertex has no contribution to either the center of mass or the
    // best-fit inertia, so it deliberately stays outside this map. Initial
    // positions and each record's `anchor` let the device form every relative
    // coordinate as a difference against a nearby reference point rather than
    // from two absolute coordinates.
    const TRANSLATION_LOCK_UNSET: u32 = u32::MAX;
    let mut translation_locks = Vec::<TranslationLock>::new();
    let mut translation_lock_index = vec![TRANSLATION_LOCK_UNSET; n_vert];
    let mut pdrd_lock_owner = vec![TRANSLATION_LOCK_UNSET; pdrd_body_props.len()];
    for dmap_index in 0..lock_data.translation_axes.len() {
        let translation_axis = lock_data.translation_axes[dmap_index];
        let translation_mode = lock_data.translation_modes[dmap_index];
        let rotation_axis = lock_data.rotation_axes[dmap_index];
        let rotation_mode = lock_data.rotation_modes[dmap_index];
        validate_translation_lock_mode(translation_mode, dmap_index);
        validate_rotation_lock_mode(rotation_mode, dmap_index);
        let translation_on = lock_component_enabled(
            translation_axis,
            translation_mode,
            TRANSLATION_LOCK_ALL,
            dmap_index,
            "translation",
        );
        let rotation_on = lock_component_enabled(
            rotation_axis,
            rotation_mode,
            ROTATION_LOCK_ALL,
            dmap_index,
            "rotation",
        );
        if !translation_on && !rotation_on {
            continue;
        }
        let translation_axis =
            normalized_or_disabled_lock_axis(translation_axis, dmap_index, "translation");
        let rotation_axis = normalized_or_disabled_lock_axis(rotation_axis, dmap_index, "rotation");

        let mut total_mass = 0.0f64;
        let mut pdrd_body_index = None;
        let mut physical_vertex_count = 0usize;
        let mut anchor = None;
        for (vertex_index, &group) in lock_data.vert_dmap.iter().enumerate() {
            if group as usize != dmap_index {
                continue;
            }
            let mass = vertex_prop[vertex_index].mass;
            assert!(
                mass.is_finite() && mass >= 0.0,
                "aggregate lock displacement group {dmap_index} has invalid mass {mass} at vertex {vertex_index}"
            );
            if mass == 0.0 {
                continue;
            }
            physical_vertex_count += 1;
            total_mass += mass as f64;
            anchor.get_or_insert_with(|| vertex.column(vertex_index).into());
            let body = vertex_prop[vertex_index].pdrd_body_index;
            match pdrd_body_index {
                Some(existing) => assert_eq!(
                    existing, body,
                    "aggregate lock displacement group {dmap_index} mixes PDRD and non-PDRD vertices, or multiple PDRD bodies; lock each physical object separately"
                ),
                None => pdrd_body_index = Some(body),
            }
        }
        assert!(
            physical_vertex_count > 0 && total_mass.is_finite() && total_mass > 0.0,
            "aggregate lock displacement group {dmap_index} has no positive physical mass"
        );
        let total_mass = total_mass as f32;
        assert!(
            total_mass.is_finite() && total_mass > 0.0,
            "aggregate lock displacement group {dmap_index} total mass is not representable in float32"
        );

        let body = pdrd_body_index.unwrap_or(0);
        if body != 0 {
            assert!(
                (body as usize) <= pdrd_body_props.len(),
                "aggregate lock displacement group {dmap_index} refers to unknown PDRD body {body}"
            );
        }
        let compact_index = translation_locks.len() as u32;
        for (vertex_index, &group) in lock_data.vert_dmap.iter().enumerate() {
            if group as usize == dmap_index && vertex_prop[vertex_index].mass > 0.0 {
                translation_lock_index[vertex_index] = compact_index;
            }
        }
        if body != 0 {
            let owner = &mut pdrd_lock_owner[body as usize - 1];
            assert_eq!(
                *owner, TRANSLATION_LOCK_UNSET,
                "PDRD body {body} is assigned to multiple aggregate-lock displacement groups"
            );
            *owner = compact_index;
            for (vertex_index, prop) in vertex_prop.iter().enumerate() {
                if prop.pdrd_body_index == body {
                    assert_eq!(
                        translation_lock_index[vertex_index],
                        compact_index,
                        "aggregate lock displacement group {dmap_index} contains only part of PDRD body {body}; a PDRD lock must cover its complete body"
                    );
                }
            }
        }
        translation_locks.push(TranslationLock {
            axis: translation_axis,
            translation_mode,
            total_mass,
            pdrd_body_index: body,
            dmap_index: dmap_index as u32,
            rotation_axis,
            rotation_mode,
            anchor: anchor.expect("positive-mass aggregate lock has no anchor"),
        });
    }
    let translation_lock_initial = if translation_locks.is_empty() {
        Vec::new()
    } else {
        vertex
            .column_iter()
            .map(|position| position.into())
            .collect::<Vec<Vec3f>>()
    };

    // Flat parallel arrays for the PDRD kernel: vertex indices and
    // centered rest positions, body-major. `body.vertex_start /
    // vertex_count` index into these (NOT into the global vertex
    // array, which may interleave non-PDRD verts).
    let pdrd_vert_list_vec: Vec<u32> = pdrd.vert_list.to_vec();
    let pdrd_rest_centered_vec: Vec<Vec3f> = if pdrd.rest_centered.is_empty() {
        Vec::new()
    } else {
        assert_eq!(
            pdrd.rest_centered.len(),
            3 * pdrd.vert_list.len(),
            "pdrd_rest_centered length mismatch (expected 3 * vert_list = {}, got {})",
            3 * pdrd.vert_list.len(),
            pdrd.rest_centered.len(),
        );
        (0..pdrd.vert_list.len())
            .map(|i| {
                Vec3f::new(
                    pdrd.rest_centered[3 * i],
                    pdrd.rest_centered[3 * i + 1],
                    pdrd.rest_centered[3 * i + 2],
                )
            })
            .collect()
    };

    let prop_set = PropSet {
        vertex: CVec::from(vertex_prop.as_ref()),
        edge: CVec::from(edge_props.as_ref()),
        face: CVec::from(face_props.as_ref()),
        hinge: CVec::from(temp_hinge_props.as_ref()),
        tet: CVec::from(tet_props.as_ref()),
        pdrd_body: CVec::from(pdrd_body_props.as_slice()),
    };

    let param_arrays = ParamArrays {
        vertex: CVec::from(temp_vertex_params.as_slice()),
        edge: CVec::from(edge_params.as_slice()),
        face: CVec::from(face_params.as_slice()),
        hinge: CVec::from(temp_hinge_params.as_slice()),
        tet: CVec::from(tet_params.as_slice()),
    };

    // Parallel computation of inverse rest matrices from the rest pose.
    // exclude_singular=false at build: a degenerate element panics there
    // instead of being masked out, so the masks are always all-zero and are
    // intentionally unused here. rest_excluded stays default-false; only the
    // streamed-rest-shape path (backend.rs) sets it.
    let rest_v = mesh.rest_vertex.as_ref().unwrap_or(&mesh.vertex);
    let (inv_rest2x2, inv_rest3x3, _, _) = compute_inv_rest(
        rest_v,
        mesh,
        face_props,
        tet_props,
        face_params,
        tet_params,
        false,
    );

    let inv_rest2x2 = CVec::from(&inv_rest2x2[..]);
    let inv_rest3x3 = CVec::from(&inv_rest3x3[..]);
    let vertex_count = mesh.mesh.mesh.vertex_count as u32;
    let surface_vert_count = mesh.mesh.mesh.surface_vert_count as u32;

    let mut fixed_index_table = vec![Vec::new(); vertex_count as usize];
    let mut insert = |i: usize, j: usize| {
        if i <= j {
            let mut index = 0;
            while index < fixed_index_table[i].len() && fixed_index_table[i][index] < j as u32 {
                index += 1;
            }
            if index == fixed_index_table[i].len() {
                fixed_index_table[i].push(j as u32);
            } else if fixed_index_table[i][index] != j as u32 {
                fixed_index_table[i].insert(index, j as u32);
            }
        }
    };
    for i in 0..vertex_count {
        insert(i as usize, i as usize);
    }
    for f in mesh.mesh.mesh.edge.column_iter() {
        for k1 in 0..2 {
            for k2 in 0..2 {
                insert(f[k1], f[k2]);
            }
        }
    }
    for f in mesh.mesh.mesh.face.column_iter() {
        for k1 in 0..3 {
            for k2 in 0..3 {
                insert(f[k1], f[k2]);
            }
        }
    }
    for hinge in mesh.mesh.mesh.hinge.column_iter() {
        for k1 in 0..4 {
            for k2 in 0..4 {
                insert(hinge[k1], hinge[k2]);
            }
        }
    }
    for tet in mesh.mesh.mesh.tet.column_iter() {
        for k1 in 0..4 {
            for k2 in 0..4 {
                insert(tet[k1], tet[k2]);
            }
        }
    }
    // Rod-bending (j-i-k) stencil. An interior rod vertex i (exactly two rod
    // edges and no incident faces, matching the gate in
    // embed_rod_bend_force_hessian) bends about its two edge-neighbors j and k,
    // assembling a 9x9 Hessian over (j, i, k). The (i,j) and (i,k) pairs are
    // already covered by the edge loop, but (j,k) is neither an edge nor a shell
    // hinge, so unless the stencil is registered here the CSR push() silently
    // drops the (j,k)/(k,j) blocks -- which turns the rank-1 PSD bending Hessian
    // indefinite and breaks the PCG solve (pAp < 0) whenever the inertia term
    // (mass / dt^2) is too small to mask it (large dt / light rods).
    for i in 0..vertex_count as usize {
        let edges = &mesh.mesh.neighbor.vertex.edge[i];
        if edges.len() == 2 && mesh.mesh.neighbor.vertex.face[i].is_empty() {
            let e0 = edges[0] as usize;
            let e1 = edges[1] as usize;
            let c0 = mesh.mesh.mesh.edge.column(e0);
            let c1 = mesh.mesh.mesh.edge.column(e1);
            let j = if c0[0] == i { c0[1] } else { c0[0] };
            let k = if c1[0] == i { c1[1] } else { c1[0] };
            let stencil = [j, i, k];
            for &a in stencil.iter() {
                for &b in stencil.iter() {
                    insert(a, b);
                }
            }
        }
    }
    for seam in constraint.stitch.iter() {
        for &i in seam.index.iter() {
            for &j in seam.index.iter() {
                insert(i as usize, j as usize);
            }
        }
    }

    // PDRD bodies use the rigid-fit path, which only needs each vertex's
    // own DIAGONAL block in the fixed sparsity used by block-Jacobi and
    // the assembled matvec diagonal.
    if !pdrd.body_rows.is_empty() {
        let mut touched: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let n_bodies = pdrd.body_rows.len() / PDRD_BODY_ROW_LEN;
        for b in 0..n_bodies {
            let row = &pdrd.body_rows[PDRD_BODY_ROW_LEN * b..PDRD_BODY_ROW_LEN * (b + 1)];
            let v_start = row[0] as usize;
            let v_count = row[1] as usize;
            let verts: &[u32] = &pdrd.vert_list[v_start..v_start + v_count];
            for &i in verts {
                fixed_index_table[i as usize].push(i);
                touched.insert(i);
            }
        }
        for i in touched {
            let row = &mut fixed_index_table[i as usize];
            row.sort_unstable();
            row.dedup();
        }
    }

    let mut transpose_table = vec![Vec::new(); vertex_count as usize];
    let mut index_sum = 0;
    for (i, row) in fixed_index_table.iter().enumerate() {
        for (k, &j) in row.iter().enumerate() {
            if i as u32 != j {
                transpose_table[j as usize].push(vector![i as u32, (index_sum + k) as u32]);
            }
        }
        index_sum += row.len();
    }

    // ---- Slot-replay assembly tables (item 3). Precompute the FixedCSRMat
    // value-slot of every 3x3 block each topology-fixed element writes, so the
    // assembly kernels can replace FixedCSRMat::push's per-block row search with
    // a direct push_at(slot). Built AFTER the fixed sparsity (including the
    // post-PDRD dedup above) is final, so row contents and their ascending sort
    // order match the device FixedCSRMat's value layout exactly. A block whose
    // (row,col) is lower-triangle (row > col) is a push() no-op, encoded as the
    // sentinel; the kernel's push_at wrapper skips sentinels, so the folded
    // atomic sums stay bitwise identical. PPF_SLOT_REPLAY=0 ships empty tables
    // and every switched kernel falls back to push() (the device push-vs-push_at
    // A/B). Emitted here (before `mesh` is rebound to FfiMesh below).
    let slot_replay = std::env::var("PPF_SLOT_REPLAY")
        .map(|v| v != "0")
        .unwrap_or(true);
    let (
        tet_hess_slots,
        face_hess_slots,
        edge_hess_slots,
        hinge_hess_slots,
        rod_bend_hess_slots,
        stitch_hess_slots,
    ) = if slot_replay {
        // Sentinel MUST match the 0xFFFFFFFFu literal every Hessian scatter
        // body in `kernels/utility/` tests a slot against.
        const FIXED_SLOT_SENTINEL: u32 = 0xFFFF_FFFF;
        // Row-major value base offset of each CSR row, mirroring the flat
        // FixedCSRMat `value` layout: row i occupies [row_offset[i],
        // row_offset[i+1]); the position of j within row i is added to it.
        let mut row_offset = vec![0u32; vertex_count as usize + 1];
        for i in 0..vertex_count as usize {
            row_offset[i + 1] = row_offset[i] + fixed_index_table[i].len() as u32;
        }
        // Mirror FixedCSRMat::push: only i <= j writes; the lower triangle is a
        // no-op, encoded as the sentinel. Every upper-triangle element block is
        // registered in the fixed sparsity by the insert loops above, so a miss
        // is a builder bug and panics (this is the M2 host assert that
        // fixed_index_table[i][pos] == j for every emitted slot).
        let slot_of = |i: usize, j: usize| -> u32 {
            if i > j {
                return FIXED_SLOT_SENTINEL;
            }
            let pos = fixed_index_table[i]
                .binary_search(&(j as u32))
                .unwrap_or_else(|_| {
                    panic!("slot precompute: block ({i},{j}) missing from fixed sparsity")
                });
            row_offset[i] + pos as u32
        };
        // tet: 16 blocks/element, index order data.mesh.mesh.tet[i].
        let mut tet_slots = Vec::with_capacity(16 * mesh.mesh.mesh.tet.ncols());
        for tet in mesh.mesh.mesh.tet.column_iter() {
            for k1 in 0..4 {
                for k2 in 0..4 {
                    tet_slots.push(slot_of(tet[k1], tet[k2]));
                }
            }
        }
        // face: 9 blocks/element; shared by the membrane, inflate, and
        // strain-limit face writes (all use data.mesh.mesh.face[i], same order).
        let mut face_slots = Vec::with_capacity(9 * mesh.mesh.mesh.face.ncols());
        for f in mesh.mesh.mesh.face.column_iter() {
            for k1 in 0..3 {
                for k2 in 0..3 {
                    face_slots.push(slot_of(f[k1], f[k2]));
                }
            }
        }
        // edge: 4 blocks/element; shared by the rod stretch and rod strain-limit
        // writes (both use data.mesh.mesh.edge[i], i in [0, rod_count)).
        let mut edge_slots = Vec::with_capacity(4 * mesh.mesh.mesh.edge.ncols());
        for e in mesh.mesh.mesh.edge.column_iter() {
            for k1 in 0..2 {
                for k2 in 0..2 {
                    edge_slots.push(slot_of(e[k1], e[k2]));
                }
            }
        }
        // hinge: 16 blocks/element in the REMAPPED (2,1,0,3) order, matching
        // dihedral_angle::face_compute_force_hessian, which permutes the hinge
        // in place before embed_hinge_force_hessian calls atomic_embed_hessian.
        let mut hinge_slots = Vec::with_capacity(16 * mesh.mesh.mesh.hinge.ncols());
        for hinge in mesh.mesh.mesh.hinge.column_iter() {
            let remapped = [hinge[2], hinge[1], hinge[0], hinge[3]];
            for k1 in 0..4 {
                for k2 in 0..4 {
                    hinge_slots.push(slot_of(remapped[k1], remapped[k2]));
                }
            }
        }
        // rod-bend: 9 blocks per interior rod vertex i, stencil (j, i, k), keyed
        // by vertex index (surface_vert_count rows, the embed's dispatch domain).
        // Non-interior verts stay all-sentinel. j, k derived exactly as
        // embed_rod_bend_force_hessian does.
        let mut rod_bend_slots = vec![FIXED_SLOT_SENTINEL; 9 * surface_vert_count as usize];
        for i in 0..surface_vert_count as usize {
            let edges = &mesh.mesh.neighbor.vertex.edge[i];
            if edges.len() == 2 && mesh.mesh.neighbor.vertex.face[i].is_empty() {
                let e0 = edges[0] as usize;
                let e1 = edges[1] as usize;
                let c0 = mesh.mesh.mesh.edge.column(e0);
                let c1 = mesh.mesh.mesh.edge.column(e1);
                let j = if c0[0] == i { c0[1] } else { c0[0] };
                let k = if c1[0] == i { c1[1] } else { c1[0] };
                let element = [j, i, k];
                for a in 0..3 {
                    for b in 0..3 {
                        rod_bend_slots[9 * i + a * 3 + b] = slot_of(element[a], element[b]);
                    }
                }
            }
        }
        // stitch: 36 blocks/seam, index order stitch.index. Degenerate repeats
        // (non-SOLID source = {s,s,s}) fold onto shared slots, exactly as the
        // repeated push() calls atomicAdd onto the same entry.
        let mut stitch_slots = Vec::with_capacity(36 * constraint.stitch.size as usize);
        for seam in constraint.stitch.iter() {
            for a in 0..6 {
                for b in 0..6 {
                    stitch_slots.push(slot_of(seam.index[a] as usize, seam.index[b] as usize));
                }
            }
        }
        (
            tet_slots,
            face_slots,
            edge_slots,
            hinge_slots,
            rod_bend_slots,
            stitch_slots,
        )
    } else {
        (
            Vec::<u32>::new(),
            Vec::<u32>::new(),
            Vec::<u32>::new(),
            Vec::<u32>::new(),
            Vec::<u32>::new(),
            Vec::<u32>::new(),
        )
    };

    let num_face = mesh.mesh.mesh.face.ncols();
    let mut face_type = vec![0_u8; num_face];
    let mut vertex_type = vec![0_u8; vertex_count as usize];
    let mut hinge_type = vec![0_u8; mesh.mesh.mesh.hinge.ncols()];
    for (i, x) in face_type.iter_mut().enumerate() {
        if i >= shell_face_count {
            *x |= 1;
        }
    }
    for &i in mesh.mesh.mesh.tet.iter() {
        vertex_type[i] |= 1;
    }
    for (i, face_neighbors) in mesh.mesh.neighbor.hinge.face.iter().enumerate() {
        for &f in face_neighbors {
            if face_type[f] & 1 == 1 {
                hinge_type[i] |= 1;
                break;
            }
        }
    }
    let ttype = data::Type {
        face: CVec::from(&face_type[..]),
        vertex: CVec::from(&vertex_type[..]),
        hinge: CVec::from(&hinge_type[..]),
    };

    let mesh = data::FfiMesh {
        face: CVec::from(
            mesh.mesh
                .mesh
                .face
                .map(|x| x as u32)
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        edge: CVec::from(
            mesh.mesh
                .mesh
                .edge
                .map(|x| x as u32)
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        hinge: CVec::from(
            mesh.mesh
                .mesh
                .hinge
                .map(|x| x as u32)
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        tet: CVec::from(
            mesh.mesh
                .mesh
                .tet
                .map(|x| x as u32)
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
    };
    let mesh_info = data::FfiMeshInfo {
        mesh,
        neighbor,
        ttype,
    };
    let vertex = VertexSet {
        prev: CVec::from(
            vertex
                .column_iter()
                .zip(velocity.column_iter())
                .map(|(x, y)| x - dt * y)
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        curr: CVec::from(
            vertex
                .column_iter()
                .map(|x| x.into_owned())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
    };

    data::DataSet {
        vertex,
        mesh: mesh_info,
        prop: prop_set,
        param_arrays,
        inv_rest2x2,
        inv_rest3x3,
        constraint,
        fixed_index_table: CVecVec::from(&fixed_index_table[..]),
        transpose_table: CVecVec::from(&transpose_table[..]),
        rod_count: rod_count as u32,
        shell_face_count: shell_face_count as u32,
        surface_vert_count,
        pdrd_vert_list: CVec::from(pdrd_vert_list_vec.as_slice()),
        pdrd_rest_centered: CVec::from(pdrd_rest_centered_vec.as_slice()),
        grain_omega: CVec::from(grain_omega_vec.as_slice()),
        grain_inv_inertia: CVec::from(grain_inv_inertia_vec.as_slice()),
        grain_torque: CVec::from(grain_torque_vec.as_slice()),
        grain_ang_stiff: CVec::from(grain_ang_stiff_vec.as_slice()),
        grain_contact_normal: CVec::from(grain_contact_normal_vec.as_slice()),
        grain_inv_inertia_center: CVec::from(grain_inv_inertia_center_vec.as_slice()),
        grain_omega_prev: CVec::from(grain_omega_prev_vec.as_slice()),
        grain_a: CVec::from(grain_a_vec.as_slice()),
        grain_b: CVec::from(grain_b_vec.as_slice()),
        grain_grot: CVec::from(grain_grot_vec.as_slice()),
        tet_hess_slots: CVec::from(tet_hess_slots.as_slice()),
        face_hess_slots: CVec::from(face_hess_slots.as_slice()),
        edge_hess_slots: CVec::from(edge_hess_slots.as_slice()),
        hinge_hess_slots: CVec::from(hinge_hess_slots.as_slice()),
        rod_bend_hess_slots: CVec::from(rod_bend_hess_slots.as_slice()),
        stitch_hess_slots: CVec::from(stitch_hess_slots.as_slice()),
        translation_lock: CVec::from(translation_locks.as_slice()),
        translation_lock_index: CVec::from(translation_lock_index.as_slice()),
        translation_lock_initial: CVec::from(translation_lock_initial.as_slice()),
        statistics_object_index: CVec::new(),
        statistics_static_object_index: CVec::new(),
        statistics_contact_count: CVec::new(),
    }
}

pub fn make_param(args: &SimArgs) -> data::ParamSet {
    let dt = clamp_substep_dt(args.dt, args.fps);
    let wind = Vec3f::new(args.wind[0], args.wind[1], args.wind[2]);
    data::ParamSet {
        time: 0.0,
        time_f32: 0.0,
        disable_contact: args.disable_contact,
        inactive_momentum: args.inactive_momentum,
        air_friction: args.air_friction,
        air_density: args.air_density,
        constraint_tol: args.constraint_tol,
        prev_dt: dt,
        dt,
        playback: args.playback,
        min_newton_steps: args.min_newton_steps,
        target_toi: args.target_toi,
        stitch_length_factor: args.stitch_length_factor,
        cg_max_iter: args.cg_max_iter,
        cg_tol: args.cg_tol,
        line_search_max_t: args.line_search_max_t,
        ccd_eps: args.ccd_eps,
        max_dx: args.max_dx,
        eiganalysis_eps: args.eiganalysis_eps,
        friction_eps: args.friction_eps,
        isotropic_air_friction: args.isotropic_air_friction,
        gravity: Vec3f::new(args.gravity[0], args.gravity[1], args.gravity[2]),
        wind,
        barrier: args.barrier.parse().unwrap_or_else(|e| panic!("{e}")),
        friction_mode: args.friction_mode.parse().unwrap_or_else(|e| panic!("{e}")),
        csrmat_max_nnz: args.csrmat_max_nnz,
        // fix_xz is a world-space Y threshold compared against the (scaled)
        // vertices in the kernel, so scale it into sim space too. 0.0 (disabled)
        // stays 0.0.
        fix_xz: args.fix_xz * args.world_scaling,
        // Linear-solve preconditioner. PPF_PRECOND env (block-jacobi|schwarz)
        // overrides the param when set; otherwise the parsed args.precond
        // (default block-jacobi) wins. Unknown values fall back to block-jacobi.
        precond: std::env::var("PPF_PRECOND")
            .ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| args.precond.parse().ok())
            .unwrap_or(data::PrecondMode::BlockJacobi),
        // Number of additive Schwarz levels (1 = single-level, 2 = two-level
        // coarse correction). schwarz::build clamps to its internal cap and
        // still honors the PPF_SCHWARZ_LEVELS env override. 0 (e.g. from an
        // older param.toml) falls back to the two-level default inside build.
        schwarz_levels: args.schwarz_levels,
        // Upper bound on Newton iterations per substep; 0 disables it. Without
        // a bound an over-constrained configuration hangs instead of failing.
        max_newton_steps: args.max_newton_steps,
        // Overwritten host-side in advance() from PPF_DISABLE_PIN_DOF_REMOVAL.
        disable_pin_dof_removal: false,
    }
}

pub fn copy_to_dataset(
    curr_vertex: &Matrix3xX<f32>,
    prev_vertex: &Matrix3xX<f32>,
    dataset: &mut data::DataSet,
) {
    let vertex = VertexSet {
        prev: CVec::from(
            prev_vertex
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        curr: CVec::from(
            curr_vertex
                .column_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
    };
    dataset.vertex = vertex;
}

trait ConvertToU32 {
    fn to_u32(&self) -> Vec<Vec<u32>>;
}

impl ConvertToU32 for Vec<Vec<usize>> {
    fn to_u32(&self) -> Vec<Vec<u32>> {
        self.iter()
            .map(|inner_vec| inner_vec.iter().map(|&x| x as u32).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    }
}

/// Marshal a mesh-side `Neighbor` into the FFI `data::FfiNeighbor`. The mesh-side
/// `VertexNeighbor.tet` list is intentionally not forwarded: `data::FfiVertexNeighbor`
/// has only face/hinge/edge/rod.
fn marshal_neighbor(src: &crate::mesh::Neighbor) -> data::FfiNeighbor {
    data::FfiNeighbor {
        vertex: FfiVertexNeighbor {
            face: CVecVec::from(&src.vertex.face.to_u32()[..]),
            hinge: CVecVec::from(&src.vertex.hinge.to_u32()[..]),
            edge: CVecVec::from(&src.vertex.edge.to_u32()[..]),
            rod: CVecVec::from(&src.vertex.rod.to_u32()[..]),
        },
        hinge: FfiHingeNeighbor {
            face: CVecVec::from(&src.hinge.face.to_u32()[..]),
        },
        edge: FfiEdgeNeighbor {
            face: CVecVec::from(&src.edge.face.to_u32()[..]),
        },
    }
}

pub fn convert_prop(young_mod: f32, poiss_rat: f32) -> (f32, f32) {
    let mu = young_mod / (2.0 * (1.0 + poiss_rat));
    let lambda = young_mod * poiss_rat / ((1.0 + poiss_rat) * (1.0 - 2.0 * poiss_rat));
    (mu, lambda)
}

/// Deduplicate a param against a (map, store) pair and return its index.
/// The map caches `param -> index`; on a miss the param is appended to `store`
/// and the fresh index is recorded. Used by every per-element param table so
/// identical params collapse to a single entry.
pub(crate) fn dedup_param<P: Eq + std::hash::Hash + Copy>(
    map: &mut HashMap<P, u32>,
    store: &mut Vec<P>,
    param: P,
) -> u32 {
    *map.entry(param).or_insert_with(|| {
        let new_idx = store.len() as u32;
        store.push(param);
        new_idx
    })
}

/// Area-weighted average of a vertex/edge's contact properties from its
/// neighboring faces, returning `(ghat, offset, friction)`.
///
/// `area_of` resolves a face's weighting area; callers pass either
/// `face_prop.area` (the FaceProp-stored area) or an external `face_area`
/// table indexed by the global face index. Those are the same value, so the
/// numeric result does not depend on which source is used. This is a pure
/// function with no captured mutable state so it stays Send/Sync-safe inside a
/// rayon `par_iter`.
pub(crate) fn area_weighted_face_param(
    neighbor_faces: &[usize],
    face_props: &[FaceProp],
    face_params: &[FaceParam],
    area_of: impl Fn(usize, &FaceProp) -> f32,
) -> (f32, f32, f32) {
    let mut ghat_sum = 0.0;
    let mut offset_sum = 0.0;
    let mut friction_sum = 0.0;
    let mut area_sum = 0.0;
    for &j in neighbor_faces.iter() {
        let face_prop = &face_props[j];
        let face_param = &face_params[face_prop.param_index as usize];
        let area = area_of(j, face_prop);
        ghat_sum += area * face_param.ghat;
        offset_sum += area * face_param.offset;
        friction_sum += area * face_param.friction;
        area_sum += area;
    }
    assert_gt!(area_sum, 0.0);
    (
        ghat_sum / area_sum,
        offset_sum / area_sum,
        friction_sum / area_sum,
    )
}

/// Average a non-rod edge's contact properties (ghat/offset/friction) from its
/// neighboring faces, weighted by face area, and return the resulting
/// `EdgeParam` with all stiffness/bend/strainlimit/plasticity fields zeroed.
/// Area-weighted average of a hinge's material from its incident faces,
/// returning the `HingeParam`, the summed incident area, and the three
/// aggregate flags the caller needs.
///
/// A hinge has no material of its own: every value it carries is this average.
/// The per-frame animated path re-runs it against that frame's face table, so
/// it lives here as one function rather than being inlined at the build site.
/// Two copies of an area-weighted average that must agree exactly is the kind
/// of duplicate that drifts without any test noticing, because both produce a
/// plausible stiffness.
///
/// `all_fixed` zeroes the bending terms, matching a fully prescribed hinge
/// having no bending DOF to resist with; `all_collider` marks a hinge whose
/// two faces both belong to a static collider; `from_geometry` is an OR, since
/// one face asking for a geometry-derived rest angle is enough.
/// Weighted average of a vertex's contact material from its incident faces and
/// rod edges, or `None` when it has neither.
///
/// Faces contribute by area, rod edges by length, which is the 1D analog. The
/// weighting is order-independent, unlike a last-writer overwrite, so a vertex
/// shared by two materials lands between them rather than on whichever
/// neighbor happened to be visited last. `None` means the caller decides: at
/// build a SAND grain falls back to its scalar params, and the per-frame path
/// leaves the entry as built.
///
/// Extracted so the build and the per-frame rewrite run the same average.
#[allow(clippy::too_many_arguments)]
/// Re-derive the vertex, edge and hinge material tables from a frame's face
/// table.
///
/// Only the faces carry an authored material; a hinge averages its two, and an
/// edge or vertex averages its neighbors. So a frame that changes the faces
/// changes all three, and updating the faces alone would animate the membrane
/// while the hinges held their build-time stiffness. The averaging functions
/// are the same ones `build` uses, so the two cannot diverge.
///
/// The averaging is LINEAR in the face params, which is why the caller may
/// interpolate these derived tables between keyframes instead of re-deriving
/// at every step: averaging interpolated faces and interpolating averaged
/// hinges give the same answer.
///
/// Entries with no incident face keep their build-time value, passed in as
/// `vertex_base` / `edge_base`. That covers a SAND grain, whose contact params
/// are scalar rather than derived, and a rod edge, which carries its own
/// authored material. Both are recognized by having no incident face rather
/// than by a flag, which is the same rule the build uses.
#[allow(clippy::too_many_arguments)]
pub fn rederive_animated_tables(
    mesh: &MeshSet,
    face_props: &[FaceProp],
    face_params: &[FaceParam],
    edge_props: &[EdgeProp],
    hinge_props: &[HingeProp],
    vertex_props: &[VertexProp],
    vertex_base: &[VertexParam],
    edge_base: &[EdgeParam],
    face_area: &[f32],
) -> (Vec<VertexParam>, Vec<EdgeParam>, Vec<HingeParam>) {
    let rod_count = mesh.mesh.mesh.rod_count;

    // Edges first: the vertex average reads the rod edges' params, so they
    // must already hold this frame's values (they are authored, not derived,
    // so they simply carry over).
    let mut edge_out = edge_base.to_vec();
    for i in rod_count..edge_props.len().min(mesh.mesh.neighbor.edge.face.len()) {
        let idx = edge_props[i].param_index as usize;
        if idx < edge_out.len() {
            edge_out[idx] = averaged_edge_param(
                &mesh.mesh.neighbor.edge.face[i],
                face_props,
                face_params,
                |j, _| face_area[j],
            );
        }
    }

    let mut vertex_out = vertex_base.to_vec();
    // A vertex's slot is read off its prop rather than assumed to be its own
    // index. `build` assigns a slot only to a vertex that HAS a param, so a
    // mesh carrying grains or stray vertices leaves gaps and the identity does
    // not hold here the way it does for faces and hinges.
    for (j, vprop) in vertex_props.iter().enumerate() {
        if j >= mesh.mesh.neighbor.vertex.face.len()
            || mesh.mesh.neighbor.vertex.face[j].is_empty()
        {
            continue;
        }
        let slot = vprop.param_index as usize;
        if slot >= vertex_out.len() {
            continue;
        }
        if let Some(vparam) = averaged_vertex_param(
            &mesh.mesh.neighbor.vertex.face[j],
            &mesh.mesh.neighbor.vertex.edge[j],
            face_props,
            face_params,
            edge_props,
            &edge_out,
            rod_count,
        ) {
            vertex_out[slot] = vparam;
        }
    }

    let mut hinge_out = Vec::with_capacity(hinge_props.len());
    for i in 0..hinge_props.len() {
        let (hparam, _area, _fixed, _collider, _geom) = averaged_hinge_param(
            &mesh.mesh.neighbor.hinge.face[i],
            face_props,
            face_params,
        );
        // The same condition the build-time table is checked against, applied
        // to every keyframe. The build sees only this hinge's value at time
        // zero, so a schedule or a spatial map that starts at zero and rises
        // later would otherwise pass there and be discarded here: the device
        // drops warp and weft outright when the hinge has no UV direction
        // (`shell_bend_directional` in
        // `kernels/energy/model/shell_bend_stiffness.kernel.cpp` returns `bend`
        // alone for a negative `uv_edge_sin2`).
        assert!(
            !directional_bending_needs_uv(
                hinge_props[i].uv_edge_sin2,
                hparam.bend_warp,
                hparam.bend_weft,
            ),
            "Hinge {i} requests directional bending (bend-warp={}, \
             bend-weft={}) at an animated keyframe but has no usable UV \
             direction. Give the mesh a UV map, or keep both at 0.0 for the \
             whole schedule.",
            hparam.bend_warp,
            hparam.bend_weft
        );
        hinge_out.push(hparam);
    }

    (vertex_out, edge_out, hinge_out)
}

pub(crate) fn averaged_vertex_param(
    neighbor_faces: &[usize],
    neighbor_edges: &[usize],
    face_props: &[FaceProp],
    face_params: &[FaceParam],
    edge_props: &[EdgeProp],
    edge_params: &[EdgeParam],
    rod_count: usize,
) -> Option<VertexParam> {
    let mut ghat_sum = 0.0f32;
    let mut offset_sum = 0.0f32;
    let mut friction_sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for &fi in neighbor_faces.iter() {
        let face_prop = &face_props[fi];
        let face_param = &face_params[face_prop.param_index as usize];
        let weight = face_prop.area;
        ghat_sum += weight * face_param.ghat;
        offset_sum += weight * face_param.offset;
        friction_sum += weight * face_param.friction;
        weight_sum += weight;
    }

    // `neighbor.vertex.edge` ranges over the full rods-plus-face-edges matrix,
    // so `ei < rod_count` is the load-bearing filter: a face edge has no
    // material of its own and would double-count its faces' contribution.
    for &ei in neighbor_edges.iter() {
        if ei < rod_count {
            let edge_prop = &edge_props[ei];
            let edge_param = &edge_params[edge_prop.param_index as usize];
            let weight = edge_prop.length;
            ghat_sum += weight * edge_param.ghat;
            offset_sum += weight * edge_param.offset;
            friction_sum += weight * edge_param.friction;
            weight_sum += weight;
        }
    }

    (weight_sum > 0.0).then(|| VertexParam {
        ghat: ghat_sum / weight_sum,
        offset: offset_sum / weight_sum,
        friction: friction_sum / weight_sum,
    })
}

/// Whether a hinge asks for directional bending it has no direction for.
///
/// The device drops warp and weft outright when the hinge carries the no-UV
/// sentinel (`shell_bend_directional` in
/// `kernels/energy/model/shell_bend_stiffness.kernel.cpp` returns `bend` alone
/// for a negative `uv_edge_sin2`), so a scene that
/// asks for anisotropy and cannot supply a direction is an authoring error
/// rather than something to discard in silence. One definition, checked at
/// build AND at every animated keyframe: the build sees only the value at time
/// zero, so a keyframed slider or a spatial map that starts at zero and rises
/// later clears the build-time check and would otherwise vanish afterwards.
pub(crate) fn directional_bending_needs_uv(
    uv_edge_sin2: f32,
    bend_warp: f32,
    bend_weft: f32,
) -> bool {
    uv_edge_sin2 < 0.0 && (bend_warp != 0.0 || bend_weft != 0.0)
}

pub(crate) fn averaged_hinge_param(
    neighbor_faces: &[usize],
    face_props: &[FaceProp],
    face_params: &[FaceParam],
) -> (HingeParam, f32, bool, bool, bool) {
    let mut offset_sum = 0.0;
    let mut ghat_sum = 0.0;
    let mut bend_sum = 0.0;
    let mut plasticity_sum = 0.0;
    let mut plasticity_threshold_sum = 0.0;
    let mut bend_damping_sum = 0.0;
    let mut bend_warp_sum = 0.0;
    let mut bend_weft_sum = 0.0;
    let mut area_sum = 0.0;
    let mut all_fixed = true;
    let mut all_collider = true;
    let mut from_geometry = false;
    for &j in neighbor_faces.iter() {
        let face_prop = &face_props[j];
        let face_param = &face_params[face_prop.param_index as usize];
        all_fixed = all_fixed && face_prop.fixed;
        all_collider = all_collider && face_prop.collider;
        offset_sum += face_prop.area * face_param.offset;
        ghat_sum += face_prop.area * face_param.ghat;
        bend_sum += face_prop.area * face_param.bend;
        plasticity_sum += face_prop.area * face_param.bend_plasticity;
        plasticity_threshold_sum += face_prop.area * face_param.bend_plasticity_threshold;
        bend_damping_sum += face_prop.area * face_param.bend_damping;
        bend_warp_sum += face_prop.area * face_param.bend_warp;
        bend_weft_sum += face_prop.area * face_param.bend_weft;
        area_sum += face_prop.area;
        if face_param.bend_rest_from_geometry {
            from_geometry = true;
        }
    }
    assert_gt!(area_sum, 0.0);
    let param = HingeParam {
        bend: if all_fixed { 0.0 } else { bend_sum / area_sum },
        ghat: ghat_sum / area_sum,
        offset: offset_sum / area_sum,
        plasticity: plasticity_sum / area_sum,
        plasticity_threshold: plasticity_threshold_sum / area_sum,
        bend_damping: if all_fixed {
            0.0
        } else {
            bend_damping_sum / area_sum
        },
        bend_warp: bend_warp_sum / area_sum,
        bend_weft: bend_weft_sum / area_sum,
    };
    (param, area_sum, all_fixed, all_collider, from_geometry)
}

pub(crate) fn averaged_edge_param(
    neighbor_faces: &[usize],
    face_props: &[FaceProp],
    face_params: &[FaceParam],
    area_of: impl Fn(usize, &FaceProp) -> f32,
) -> EdgeParam {
    let (ghat, offset, friction) =
        area_weighted_face_param(neighbor_faces, face_props, face_params, area_of);
    EdgeParam {
        stiffness: 0.0,
        bend: 0.0,
        ghat,
        offset,
        friction,
        strainlimit: 0.0,
        plasticity: 0.0,
        plasticity_threshold: 0.0,
        bend_rest_from_geometry: false,
        deform_damping: 0.0,
        bend_damping: 0.0,
    }
}

pub fn make_collision_mesh(
    vertex: &Matrix3xX<f32>,
    face: &Matrix3xX<usize>,
    face_props: &[FaceProp],
    face_params: &[FaceParam],
) -> CollisionMesh {
    let mesh = Mesh::new(
        Matrix2xX::<usize>::zeros(0),
        face.clone(),
        na::Matrix4xX::zeros(0),
        face.ncols(),
        vertex.ncols(),
    );
    let neighbor = marshal_neighbor(&mesh.neighbor);
    let n_vert = vertex.ncols();
    let n_edge = mesh.mesh.edge.ncols();
    let n_face = face.ncols();
    assert_eq!(n_face, face_props.len());

    // Build edge props and params
    let mut edge_param_map: HashMap<EdgeParam, u32> = HashMap::new();
    let mut unique_edge_params = Vec::new();
    let mut collision_edge_props = Vec::new();

    for i in 0..n_edge {
        let param = averaged_edge_param(
            &mesh.neighbor.edge.face[i],
            face_props,
            face_params,
            |_, face_prop| face_prop.area,
        );
        let param_idx = dedup_param(&mut edge_param_map, &mut unique_edge_params, param);
        collision_edge_props.push(EdgeProp {
            length: 0.0,
            initial_length: 0.0,
            mass: 0.0,
            fixed: false,
            param_index: param_idx,
            // The collision mesh carries no pins of its own (its shape is
            // driven, not solved), so no pin can grant it an allowance.
            pin_allow_intersection: false,
        });
    }

    // Build vertex props and params
    let mut vertex_param_map: HashMap<VertexParam, u32> = HashMap::new();
    let mut unique_vertex_params = Vec::new();
    let mut collision_vertex_props = Vec::new();

    for i in 0..n_vert {
        let (ghat, offset, friction) = area_weighted_face_param(
            &mesh.neighbor.vertex.face[i],
            face_props,
            face_params,
            |_, face_prop| face_prop.area,
        );
        let param = VertexParam {
            ghat,
            offset,
            friction,
        };
        let param_idx = dedup_param(&mut vertex_param_map, &mut unique_vertex_params, param);
        collision_vertex_props.push(VertexProp {
            area: 0.0,
            volume: 0.0,
            mass: 0.0,
            rest_bend_angle: std::f32::consts::PI,
            fix_index: 0,
            pull_index: 0,
            param_index: param_idx,
            pdrd_body_index: 0,
            // The collision mesh is the disjoint contact-only pool: it never
            // shares a pair with itself (it is not in the solved namespace),
            // so the flag is inert here.
            collider: false,
            // Nothing in this pool is ever the object whose allowance is
            // being asked about: a collision-mesh face is only ever tested
            // against a DYNAMIC edge, so the pair is inter-object by
            // construction and it is the dynamic side's policy that decides
            // (either side opting in is enough). Leaving these at "unknown"
            // and "nothing allowed" keeps that asymmetry explicit.
            object_index: crate::data::NO_OBJECT_INDEX,
            // Nor does it belong to any group, so it is another group from
            // every object, as the add-on builds it: a collider always sits
            // in a STATIC group of its own.
            group_index: crate::data::NO_GROUP_INDEX,
            intersect_policy: 0,
            pin_allow_intersection: false,
        });
    }

    CollisionMesh {
        vertex: CVec::from(
            vertex
                .column_iter()
                .map(|x| x.into_owned())
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        face: CVec::from(
            mesh.mesh
                .face
                .column_iter()
                .map(|x| x.map(|x| x as u32))
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        edge: CVec::from(
            mesh.mesh
                .edge
                .column_iter()
                .map(|x| x.map(|x| x as u32))
                .collect::<Vec<_>>()
                .as_slice(),
        ),
        prop: CollisionMeshPropSet {
            vertex: CVec::from(collision_vertex_props.as_slice()),
            edge: CVec::from(collision_edge_props.as_slice()),
            face: CVec::from(face_props),
        },
        param_arrays: CollisionMeshParamArrays {
            vertex: CVec::from(unique_vertex_params.as_slice()),
            face: CVec::from(face_params),
            edge: CVec::from(unique_edge_params.as_slice()),
        },
        neighbor,
    }
}

#[cfg(test)]
mod rest_shape_tests {
    use super::*;
    use na::Vector3;

    // I459: a NaN/Inf singular value (e.g. from a normalize() of a zero or
    // collinear edge upstream) must exclude the element. Every comparison with
    // NaN is false, so without the explicit is_finite() guard the matrix would
    // fall through to try_inverse and leave a NaN inv_rest active in the energy.
    #[test]
    fn invert_or_exclude3_excludes_non_finite_and_degenerate() {
        assert!(invert_or_exclude3(&Matrix3::from_element(f32::NAN)).1);
        assert!(invert_or_exclude3(&Matrix3::from_element(f32::INFINITY)).1);
        assert!(invert_or_exclude3(&Matrix3::zeros()).1); // degenerate, smax == 0
        assert!(!invert_or_exclude3(&Matrix3::<f32>::identity()).1); // well-conditioned
    }

    #[test]
    fn invert_or_exclude2_excludes_non_finite_and_degenerate() {
        assert!(invert_or_exclude2(&Matrix2::from_element(f32::NAN)).1);
        assert!(invert_or_exclude2(&Matrix2::zeros()).1);
        assert!(!invert_or_exclude2(&Matrix2::<f32>::identity()).1);
    }

    // A zero-area rest triangle has no tangent frame. The cross product that
    // builds it is the zero vector, and `normalize()` of a zero vector is NaN
    // rather than an error, so the rest matrix arrives carrying a NaN row.
    #[test]
    fn collinear_rest_triangle_has_a_non_finite_tangent_matrix() {
        // (0,0,0) - (1,0,0) - (2,0,0): the middle vertex sits exactly on the
        // edge, which is what Blender's quad tessellation emits for a quad
        // carrying one inline vertex.
        let dx = Matrix3x2::<f32>::from_columns(&[
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
        ]);
        let d_mat = shell_rest_tangent_matrix(&dx);
        // The first edge still normalizes, so only the second tangent axis is
        // NaN: the matrix is partly finite, which is why a whole-matrix
        // finiteness test is the guard and a determinant test is not.
        assert!(d_mat.iter().any(|v| !v.is_finite()), "{d_mat}");
        // And this is why `try_inverse` alone cannot be the guard: NaN compares
        // false against zero, so it reports success on this matrix.
        assert!(d_mat.try_inverse().is_some());
        // A well-formed triangle keeps its exact edge lengths in the frame.
        let dx = Matrix3x2::<f32>::from_columns(&[
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
        ]);
        let d_mat = shell_rest_tangent_matrix(&dx);
        assert!(d_mat.iter().all(|v| v.is_finite()));
        assert!((d_mat.determinant() - 6.0).abs() < 1e-5);
    }

    // The build-time inversion must REJECT that NaN matrix. `try_inverse` alone
    // does not: every comparison it makes against zero is false for NaN, so it
    // returns a NaN inverse that would be uploaded as `inv_rest2x2` and surface
    // only once the linear solve reports a non-finite quantity, naming neither
    // the object nor the face.
    #[test]
    fn build_time_inversion_rejects_a_degenerate_rest_face() {
        let nan_inverse = Matrix2::from_element(f32::NAN).try_inverse();
        assert!(
            nan_inverse.is_some_and(|inv| inv.iter().any(|v| v.is_nan())),
            "try_inverse is expected to pass NaN through; if it now rejects \
             it, this guard's premise changed"
        );
        for bad in [Matrix2::from_element(f32::NAN), Matrix2::zeros()] {
            assert!(std::panic::catch_unwind(move || {
                invert_rest_or_panic2(&bad, 7, [3, 4, 5])
            })
            .is_err());
        }
        let good = Matrix2::<f32>::new(2.0, 0.0, 0.0, 4.0);
        let inv = invert_rest_or_panic2(&good, 0, [0, 1, 2]);
        assert!((inv[(0, 0)] - 0.5).abs() < 1e-6 && (inv[(1, 1)] - 0.25).abs() < 1e-6);
    }

    // End to end through the two helpers the build path composes: a collinear
    // rest triangle aborts the build instead of reaching the device.
    #[test]
    fn collinear_rest_triangle_aborts_the_build() {
        let dx = Matrix3x2::<f32>::from_columns(&[
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
        ]);
        let d_mat = shell_rest_tangent_matrix(&dx);
        let err = std::panic::catch_unwind(move || invert_rest_or_panic2(&d_mat, 12, [8, 9, 10]))
            .unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .expect("panic payload is a formatted String");
        assert!(msg.contains("Degenerate shell face 12"), "{msg}");
        assert!(msg.contains("8, 9, 10"), "{msg}");
    }

    // Anisotropic bending without a UV direction is refused, and the refusal
    // has to hold at every animated keyframe rather than only at time zero:
    // the natural way to paint anisotropy into a region is a zero slider with
    // a positive map target, which passes a build-time-only check and is then
    // discarded by the device on every frame.
    #[test]
    fn directional_bending_without_a_uv_direction_is_refused() {
        const NO_UV: f32 = -1.0;
        let with_uv = 0.5;
        // The case a build-time-only check misses: zero now, positive later.
        assert!(!directional_bending_needs_uv(NO_UV, 0.0, 0.0));
        assert!(directional_bending_needs_uv(NO_UV, 600.0, 0.0));
        assert!(directional_bending_needs_uv(NO_UV, 0.0, 6000.0));
        // A mesh that has a direction is free to ask for anisotropy.
        assert!(!directional_bending_needs_uv(with_uv, 600.0, 6000.0));
        // And a mesh with no UV stays perfectly usable while it asks for none.
        assert!(!directional_bending_needs_uv(NO_UV, 0.0, 0.0));
    }

    // The tet arm needs a real value assertion: with an identity rest matrix
    // the inverse IS the input, so returning the input unchanged would pass.
    // A sheared, anisotropic but well-conditioned matrix distinguishes them,
    // and doubles as the "the gate is not too tight" case the tet arm lacked.
    #[test]
    fn a_sheared_well_conditioned_rest_tet_inverts_correctly() {
        let rest = Matrix3::<f32>::from_columns(&[
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.5, 4.0, 0.0),
            Vector3::new(0.0, 1.0, 8.0),
        ]);
        let sv = rest.singular_values();
        assert!(
            sv[2] >= REST_SHAPE_MIN_CONDITION * sv[0],
            "the fixture has to clear the gate: {sv:?}"
        );
        let inv = invert_rest_or_panic3(&rest, 0, [0, 1, 2, 3]);
        // Not the input, and a genuine inverse.
        assert!((inv - rest).amax() > 1.0, "returning the input would pass");
        let identity = rest * inv;
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (identity[(r, c)] - want).abs() < 1.0e-5,
                    "rest * inv is not the identity: {identity}"
                );
            }
        }
    }

    // A zero-area rest triangle arrives with a non-finite tangent matrix, so
    // `smin / smax` is NaN. Printing that as a conditioning "below the
    // threshold" states a comparison that never happened: NaN fails every one.
    #[test]
    fn a_non_finite_rest_matrix_is_named_rather_than_given_a_ratio() {
        let degenerate = describe_conditioning(f32::NAN, f32::NAN);
        assert!(degenerate.contains("is degenerate"), "{degenerate}");
        assert!(!degenerate.contains("conditioned at"), "{degenerate}");

        // A finite but ill-conditioned one still reports its ratio, which is
        // what tells a hair's-breadth sliver from an exactly collinear triple.
        let ill = describe_conditioning(1.0, 1.0e-9);
        assert!(ill.contains("conditioned at"), "{ill}");
        assert!(ill.contains("1.000e-9"), "{ill}");
    }

    // Every bending key a spatial map can drive reaches the device through the
    // HINGE table, never the face table, so a per-face map varies the physics
    // only if this average preserves the variation. It is AREA-weighted, which
    // is a convex combination of the two incident faces: the result lies
    // strictly between two different inputs and equals them when they agree.
    // Nothing else tests this function, and five parameters rest on it.
    #[test]
    fn the_hinge_average_is_area_weighted_and_preserves_variation() {
        let mut soft = FaceParam::default();
        soft.bend = 40.0;
        soft.bend_damping = 0.02;
        soft.bend_warp = 200.0;
        soft.bend_weft = 2000.0;
        soft.bend_plasticity = 12.0;
        let mut stiff = soft;
        stiff.bend = 80.0;
        stiff.bend_damping = 0.05;
        stiff.bend_warp = 600.0;
        stiff.bend_weft = 6000.0;
        stiff.bend_plasticity = 16.0;

        // Areas 1 and 3, so the area-weighted mean sits at three quarters of
        // the way to the stiff face and the arithmetic mean does not.
        let mut small = FaceProp::default();
        small.area = 1.0;
        small.param_index = 0;
        let mut large = FaceProp::default();
        large.area = 3.0;
        large.param_index = 1;

        let (hinge, area, _fixed, _collider, _geom) =
            averaged_hinge_param(&[0, 1], &[small, large], &[soft, stiff]);
        assert_eq!(area, 4.0);
        assert_eq!(hinge.bend, 70.0, "area-weighted, not the 60.0 arithmetic mean");
        assert_eq!(hinge.bend_warp, 500.0);
        assert_eq!(hinge.bend_weft, 5000.0);
        assert_eq!(hinge.plasticity, 15.0, "bend_plasticity lands on `plasticity`");
        assert!((hinge.bend_damping - 0.0425).abs() < 1e-6);

        // A convex combination: strictly inside the two inputs when they
        // differ, so a map's variation reaches the hinge rather than flattening.
        assert!(hinge.bend > soft.bend && hinge.bend < stiff.bend);
        assert!(hinge.bend_warp > soft.bend_warp && hinge.bend_warp < stiff.bend_warp);

        // And equal to the common value when the two faces agree, so an
        // unmapped scene is untouched by any of this.
        let (uniform, _a, _f, _c, _g) =
            averaged_hinge_param(&[0, 1], &[small, large], &[soft, soft]);
        assert_eq!(uniform.bend, soft.bend);
        assert_eq!(uniform.bend_warp, soft.bend_warp);
    }

    // The threshold is `sqrt(f32::EPSILON)`, which is not const-evaluable, so
    // the constant is written out and this test verifies it against the
    // computed value. The addon gate computes the same number rather than
    // repeating the literal (see `blender_addon/core/utils.min_rest_condition`),
    // and both have to agree, or the addon refuses a set the solver would have
    // accepted. The solver's gate is the authoritative one: it tests the matrix
    // it actually inverts, after the UV rotation and the per-axis shrink, which
    // the addon cannot see.
    #[test]
    fn min_condition_is_the_square_root_of_f32_epsilon() {
        assert_eq!(REST_SHAPE_MIN_CONDITION, f32::EPSILON.sqrt());
        // Loose enough to admit a genuinely thin triangle, which the exclusion
        // ratio for streamed rest shapes is not.
        assert!(REST_SHAPE_MIN_CONDITION < REST_SHAPE_EXCLUDE_RATIO);
    }

    // The case issue #144 was reported on, and the one a finiteness test cannot
    // see. This rest triangle is near-collinear rather than collinear: it is
    // entirely finite, `try_inverse` succeeds on it, and the inverse it returns
    // is finite too. Only the conditioning gives it away.
    #[test]
    fn near_collinear_rest_triangle_aborts_the_build() {
        // The worst face of the reporter's mesh, to scale: a longest edge of
        // 5.678e-02 with the third vertex 1.173e-08 off it.
        let dx = Matrix3x2::<f32>::from_columns(&[
            Vector3::new(0.056_781_4, 0.0, 0.0),
            Vector3::new(0.032_217_3, 1.172_5e-8, 0.0),
        ]);
        let d_mat = shell_rest_tangent_matrix(&dx);

        // Finiteness and invertibility both pass on it, which is why a test
        // built out of either one ships it.
        assert!(d_mat.iter().all(|v| v.is_finite()), "{d_mat}");
        let inv = d_mat.try_inverse().expect("a near-collinear matrix inverts");
        assert!(inv.iter().all(|v| v.is_finite()), "{inv}");
        // And it is enormous: the entries the elastic Hessian then squares.
        assert!(inv.amax() > 1.0e7, "{inv}");

        let sv = d_mat.singular_values();
        let ratio = sv[1] / sv[0];
        assert!(ratio < REST_SHAPE_MIN_CONDITION, "ratio {ratio:e}");

        let err = std::panic::catch_unwind(move || invert_rest_or_panic2(&d_mat, 671, [406, 404, 402]))
            .unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .expect("panic payload is a formatted String");
        assert!(msg.contains("Degenerate shell face 671"), "{msg}");
        assert!(msg.contains("406, 404, 402"), "{msg}");
        // The measured ratio is in the message: without it the artist cannot
        // tell a hair's-breadth sliver from an exactly collinear triple.
        assert!(msg.contains("conditioned at"), "{msg}");
    }

    // The tet arm carries the same test for the same reason: a bare
    // `try_inverse` accepts a needle or cap tet exactly as it accepts a
    // near-collinear triangle.
    #[test]
    fn near_degenerate_rest_tet_aborts_the_build() {
        // Unit base, height 1e-6: coplanar to within a millionth of its extent.
        let flat = Matrix3::<f32>::from_columns(&[
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0e-6),
        ]);
        assert!(flat.iter().all(|v| v.is_finite()));
        assert!(flat.try_inverse().is_some(), "invertible, so `try_inverse` accepts it");

        let err = std::panic::catch_unwind(move || invert_rest_or_panic3(&flat, 42, [1, 2, 3, 4]))
            .unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .expect("panic payload is a formatted String");
        assert!(msg.contains("Degenerate tetrahedron 42"), "{msg}");
        assert!(msg.contains("1, 2, 3, 4"), "{msg}");

        // And a well-shaped tet is untouched.
        let good = Matrix3::<f32>::identity();
        assert_eq!(invert_rest_or_panic3(&good, 0, [0, 1, 2, 3]), good);
    }

    // The other half of the threshold: a thin but sound triangle is legitimate
    // geometry and must still build. Rejecting these is the cost of setting the
    // gate too tight, so it is asserted rather than assumed.
    #[test]
    fn thin_but_sound_rest_triangle_still_builds() {
        // Aspect ratio 1000:1, an order above the threshold.
        let dx = Matrix3x2::<f32>::from_columns(&[
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0e-3, 0.0),
        ]);
        let d_mat = shell_rest_tangent_matrix(&dx);
        let sv = d_mat.singular_values();
        assert!(sv[1] / sv[0] > REST_SHAPE_MIN_CONDITION);
        let inv = invert_rest_or_panic2(&d_mat, 0, [0, 1, 2]);
        assert!((inv[(1, 1)] - 1.0e3).abs() < 1.0, "{inv}");
    }

    #[test]
    fn lock_components_are_independently_enabled() {
        let translation = normalized_or_disabled_lock_axis(Vec3f::zeros(), 3, "translation");
        let rotation = normalized_or_disabled_lock_axis(Vec3f::new(0.0, 0.0, 1.0), 3, "rotation");
        assert_eq!(translation, Vec3f::zeros());
        assert_eq!(rotation, Vec3f::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn lock_component_rejects_non_unit_axis() {
        assert!(std::panic::catch_unwind(|| {
            normalized_or_disabled_lock_axis(Vec3f::new(0.0, 2.0, 0.0), 0, "rotation")
        })
        .is_err());
    }

    #[test]
    fn rotation_lock_modes_accept_both_contract_values() {
        validate_rotation_lock_mode(ROTATION_LOCK_ALLOW_ONLY, 0);
        validate_rotation_lock_mode(ROTATION_LOCK_PROHIBIT_AXIS, 1);
    }

    #[test]
    fn rotation_lock_mode_rejects_unknown_value() {
        // 2 is ROTATION_LOCK_ALL, so the first value past the admissible set
        // is 3. Keeping the rejection case at 2 would assert the opposite of
        // the contract while still reading like a guard.
        assert!(std::panic::catch_unwind(|| { validate_rotation_lock_mode(3, 0) }).is_err());
    }

    #[test]
    fn rotation_lock_mode_accepts_every_admissible_value() {
        validate_rotation_lock_mode(ROTATION_LOCK_ALLOW_ONLY, 0);
        validate_rotation_lock_mode(ROTATION_LOCK_PROHIBIT_AXIS, 0);
        validate_rotation_lock_mode(ROTATION_LOCK_ALL, 0);
    }

    #[test]
    fn translation_lock_mode_admits_axis_and_all_only() {
        validate_translation_lock_mode(TRANSLATION_LOCK_AXIS, 0);
        validate_translation_lock_mode(TRANSLATION_LOCK_ALL, 0);
        assert!(std::panic::catch_unwind(|| { validate_translation_lock_mode(2, 0) }).is_err());
    }

    #[test]
    fn lock_component_enable_reads_the_mode_not_the_axis() {
        // The whole point of the mode field: an all-axes lock is ON despite a
        // zero axis, and an axis-mode lock is OFF because of one.
        assert!(lock_component_enabled(
            Vec3f::zeros(),
            TRANSLATION_LOCK_ALL,
            TRANSLATION_LOCK_ALL,
            0,
            "translation"
        ));
        assert!(!lock_component_enabled(
            Vec3f::zeros(),
            TRANSLATION_LOCK_AXIS,
            TRANSLATION_LOCK_ALL,
            0,
            "translation"
        ));
        assert!(lock_component_enabled(
            Vec3f::new(1.0, 0.0, 0.0),
            TRANSLATION_LOCK_AXIS,
            TRANSLATION_LOCK_ALL,
            0,
            "translation"
        ));
        // A stale axis beside an all-axes mode is the non-canonical record
        // the biconditional exists to refuse.
        assert!(std::panic::catch_unwind(|| {
            lock_component_enabled(
                Vec3f::new(1.0, 0.0, 0.0),
                TRANSLATION_LOCK_ALL,
                TRANSLATION_LOCK_ALL,
                0,
                "translation",
            )
        })
        .is_err());
    }
}
