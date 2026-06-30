// File: crates/ppf-cts-core/src/kernels/scene_loops.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Loop-elimination kernels for the scene module. Each function here
// replaces a `for ...: list.append(...)` (or similar dynamic
// list-growth) hotspot. Inputs are primitive slices / Vec; outputs are
// fully-shaped primitive buffers (or whole strings, for TOML formatters).
// The PyO3 layer wraps the buffers as numpy arrays.

use std::fmt::Write as _;

/// Typed errors returned by the scene-loops kernels. Replaces the
/// old `Result<_, String>` return shape so the PyO3 layer can map
/// each variant to the right `PyErr` class via `errors.rs`.
#[derive(Debug, thiserror::Error)]
pub enum SceneLoopsError {
    #[error("face {fi} corner {k} references vertex {vertex} outside vertex_uv (len={n_vert})")]
    FaceUvVertexOob {
        fi: usize,
        k: usize,
        vertex: i64,
        n_vert: usize,
    },
    #[error("times len {actual} != total kf {total_kf}")]
    TransformAnimTimesLen { actual: usize, total_kf: usize },
    #[error("trans len {actual} != 3 * total kf {total_kf}")]
    TransformAnimTransLen { actual: usize, total_kf: usize },
    #[error("quats len {actual} != 4 * total kf {total_kf}")]
    TransformAnimQuatsLen { actual: usize, total_kf: usize },
    #[error("scales len {actual} != 3 * total kf {total_kf}")]
    TransformAnimScalesLen { actual: usize, total_kf: usize },
    #[error("handles_right len {actual} != 2 * n_seg {n_seg}")]
    HandlesRightLen { actual: usize, n_seg: usize },
    #[error("handles_left len {actual} != 2 * n_seg {n_seg}")]
    HandlesLeftLen { actual: usize, n_seg: usize },
    #[error("unknown segment interpolation {got:?} (expected LINEAR|BEZIER|CONSTANT)")]
    UnknownSegmentInterp { got: String },
    #[error("ops_offsets len {ops_len} != headers len {headers_len} + 1")]
    PinOpsOffsetsMismatch { ops_len: usize, headers_len: usize },
    #[error("axis must be 0, 1, or 2; got {axis}")]
    InvalidAxis { axis: usize },
}

// ---------------------------------------------------------------------------
// Stitch preview build.
//
// Replaces (frontend/_scene_.py ~2568-2579):
//
//     stitch_vert, stitch_edge = [], []
//     for ind, w in zip(self._stitch_ind, self._stitch_w):
//         src = vert[ind[0]]
//         target = w[1]*vert[ind[1]] + w[2]*vert[ind[2]] + w[3]*vert[ind[3]]
//         idx0 = len(stitch_vert) + len(vert)
//         idx1 = idx0 + 1
//         stitch_vert.append(src); stitch_vert.append(target)
//         stitch_edge.append([idx0, idx1])
//
// `vert` is (n_vert, 3) f64. `stitch_ind` is (n_stitch, 6) i64.
// `stitch_w` is (n_stitch, 6) f64. Each row is laid out
// `[s0, s1, s2, t0, t1, t2]` / `[ws0, ws1, ws2, wt0, wt1, wt2]`: the
// source endpoint is the barycentric point over (s0, s1, s2) with
// weights (ws0..ws2), the target endpoint is the barycentric point over
// (t0, t1, t2) with weights (wt0..wt2). A non-SOLID source degenerates
// to s0=s1=s2 with ws=[1, 0, 0], recovering the single source vertex.
//
// Returns (stitch_vert_flat (2*n_stitch * 3), stitch_edge_flat (n_stitch * 2)).
// `stitch_edge` indices are absolute, i.e., `len(vert) + 2*i` and
// `len(vert) + 2*i + 1`.
pub fn stitch_preview_lines(
    vert: &[f64],
    n_vert: usize,
    stitch_ind: &[i64],
    stitch_w: &[f64],
    n_stitch: usize,
) -> (Vec<f64>, Vec<u32>) {
    debug_assert!(vert.len() == n_vert * 3);
    debug_assert!(stitch_ind.len() == n_stitch * 6);
    debug_assert!(stitch_w.len() == n_stitch * 6);

    let mut out_vert = vec![0.0_f64; 2 * n_stitch * 3];
    let mut out_edge = vec![0_u32; n_stitch * 2];

    for i in 0..n_stitch {
        let s = i * 6;
        let s0 = stitch_ind[s] as usize;
        let s1 = stitch_ind[s + 1] as usize;
        let s2 = stitch_ind[s + 2] as usize;
        let t0 = stitch_ind[s + 3] as usize;
        let t1 = stitch_ind[s + 4] as usize;
        let t2 = stitch_ind[s + 5] as usize;
        let ws0 = stitch_w[s];
        let ws1 = stitch_w[s + 1];
        let ws2 = stitch_w[s + 2];
        let wt0 = stitch_w[s + 3];
        let wt1 = stitch_w[s + 4];
        let wt2 = stitch_w[s + 5];

        for d in 0..3 {
            out_vert[6 * i + d] =
                ws0 * vert[3 * s0 + d] + ws1 * vert[3 * s1 + d] + ws2 * vert[3 * s2 + d];
            out_vert[6 * i + 3 + d] =
                wt0 * vert[3 * t0 + d] + wt1 * vert[3 * t1 + d] + wt2 * vert[3 * t2 + d];
        }
        let idx0 = (n_vert + 2 * i) as u32;
        out_edge[2 * i] = idx0;
        out_edge[2 * i + 1] = idx0 + 1;
    }
    (out_vert, out_edge)
}

// ---------------------------------------------------------------------------
// Per-face UV expansion.
//
// Replaces (frontend/_scene_.py ~2773-2782):
//
//     face_uv = []
//     for f in faces:
//         uv_per_face = np.array([vertex_uv[f[0]], vertex_uv[f[1]], vertex_uv[f[2]]])
//         face_uv.append(uv_per_face)
//
// `vertex_uv` is (n_vert, 2) f64. `faces` is (n_face, 3) i64.
// Returns a flat buffer of shape (n_face, 3, 2) = n_face * 6.
pub fn face_uv_expand(
    vertex_uv: &[f64],
    n_vert: usize,
    faces: &[i64],
    n_face: usize,
) -> Result<Vec<f64>, SceneLoopsError> {
    debug_assert!(vertex_uv.len() == n_vert * 2);
    debug_assert!(faces.len() == n_face * 3);

    let mut out = vec![0.0_f64; n_face * 6];
    for fi in 0..n_face {
        for k in 0..3 {
            let v = faces[3 * fi + k];
            if v < 0 || (v as usize) >= n_vert {
                return Err(SceneLoopsError::FaceUvVertexOob {
                    fi,
                    k,
                    vertex: v,
                    n_vert,
                });
            }
            let vi = v as usize;
            out[6 * fi + 2 * k] = vertex_uv[2 * vi];
            out[6 * fi + 2 * k + 1] = vertex_uv[2 * vi + 1];
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Pin-marker index collection.
//
// Replaces (frontend/_scene_.py ~2597-2604):
//
//     pts = []
//     for pin in self._pin:
//         if pin.hide_in_preview:
//             continue
//         pts.extend(pin.index)
//     pts = np.array(pts)
//
// Input: a flat (`hide_flags`, `index_offsets`, `index_data`) representation
// where `index_offsets` has length `n_pins + 1` and slices `index_data`.
pub fn collect_pin_marker_indices(
    hide_flags: &[bool],
    index_offsets: &[usize],
    index_data: &[i64],
) -> Vec<i64> {
    debug_assert_eq!(hide_flags.len() + 1, index_offsets.len());
    let mut out: Vec<i64> = Vec::new();
    for (i, &hide) in hide_flags.iter().enumerate() {
        if hide {
            continue;
        }
        let s = index_offsets[i];
        let e = index_offsets[i + 1];
        out.extend_from_slice(&index_data[s..e]);
    }
    out
}

// ---------------------------------------------------------------------------
// Static transform animation keyframe concat.
//
// Replaces (frontend/_scene_.py ~2207-2215):
//
//     all_times, all_trans, all_quats, all_scales = [], [], [], []
//     for _, anim in self._static_transform_animations:
//         all_times.extend(anim.times)
//         all_trans.extend(anim.translations)
//         all_quats.extend(anim.quaternions)
//         all_scales.extend(anim.scales)
//
// Input: each anim contributes `n_kf_i` keyframes. `times` is flat
// (sum_kf,), `trans` is flat (sum_kf * 3,), `quats` is flat (sum_kf * 4,),
// `scales` is flat (sum_kf * 3,). The function is pure validation +
// pre-sized memcpy; the caller passes already-flattened per-anim buffers
// stitched in iteration order. It exists so we keep the explicit
// per-anim stride check on the Rust side and the wrapper drops the
// Python list-extends. Returns `total_kf` for assertion.
pub fn concat_static_transform_anims(
    kf_counts: &[usize],
    times: &[f64],
    trans: &[f64],
    quats: &[f64],
    scales: &[f64],
) -> Result<usize, SceneLoopsError> {
    let total_kf: usize = kf_counts.iter().sum();
    if times.len() != total_kf {
        return Err(SceneLoopsError::TransformAnimTimesLen {
            actual: times.len(),
            total_kf,
        });
    }
    if trans.len() != total_kf * 3 {
        return Err(SceneLoopsError::TransformAnimTransLen {
            actual: trans.len(),
            total_kf,
        });
    }
    if quats.len() != total_kf * 4 {
        return Err(SceneLoopsError::TransformAnimQuatsLen {
            actual: quats.len(),
            total_kf,
        });
    }
    if scales.len() != total_kf * 3 {
        return Err(SceneLoopsError::TransformAnimScalesLen {
            actual: scales.len(),
            total_kf,
        });
    }
    Ok(total_kf)
}

// ---------------------------------------------------------------------------
// Transform-keyframe segment interpolation packing.
//
// Replaces (frontend/_scene_.py ~2277-2293): the list comprehensions
// that build (n_seg,) interp_codes and (n_seg, 4) handles arrays.
//
// `interp_strs` is per-segment interpolation name ("LINEAR", "BEZIER",
// "CONSTANT"). `handles_right` and `handles_left` are flat (n_seg * 2)
// f64 buffers; if the Python `op.segments[i]` lacks a key, the wrapper
// substitutes the same defaults: handle_right -> [1/3, 0.0],
// handle_left -> [2/3, 1.0]. Defaulting stays in the wrapper because
// the Python dict has heterogeneous keys; this kernel just packs the
// fully-resolved values.
//
// Returns (interp_codes (n_seg,), handles_flat (n_seg*4,)).
pub fn pack_transform_keyframe_segments(
    interp_strs: &[&str],
    handles_right: &[f64],
    handles_left: &[f64],
) -> Result<(Vec<u8>, Vec<f64>), SceneLoopsError> {
    let n_seg = interp_strs.len();
    if handles_right.len() != n_seg * 2 {
        return Err(SceneLoopsError::HandlesRightLen {
            actual: handles_right.len(),
            n_seg,
        });
    }
    if handles_left.len() != n_seg * 2 {
        return Err(SceneLoopsError::HandlesLeftLen {
            actual: handles_left.len(),
            n_seg,
        });
    }
    let mut codes = vec![0_u8; n_seg];
    let mut handles = vec![0.0_f64; n_seg * 4];
    for i in 0..n_seg {
        codes[i] = match interp_strs[i] {
            "LINEAR" => 0,
            "BEZIER" => 1,
            "CONSTANT" => 2,
            other => {
                return Err(SceneLoopsError::UnknownSegmentInterp {
                    got: other.to_string(),
                });
            }
        };
        handles[4 * i] = handles_right[2 * i];
        handles[4 * i + 1] = handles_right[2 * i + 1];
        handles[4 * i + 2] = handles_left[2 * i];
        handles[4 * i + 3] = handles_left[2 * i + 1];
    }
    Ok((codes, handles))
}

// ---------------------------------------------------------------------------
// Pin TOML formatting.
//
// Replaces (frontend/_scene_.py ~2014-2076): the per-pin and per-op
// `f.write(...)` blocks. Builds the entire pin section (including
// op-subsections) as a single string. The wrapper writes the result.

#[derive(Debug, Clone)]
pub struct PinHeader {
    pub operation_count: usize,
    pub pin_count: usize,
    pub pull_strength: f64,
    /// Per-pin scale on the moving (kinematic) constraint force.
    /// 1.0 leaves the force unchanged; the solver applies it only to
    /// kinematic pins.
    pub pin_stiffness: f64,
    pub unpin_time: Option<f64>,
    pub pin_group_id: Option<String>,
}

#[derive(Debug, Clone)]
pub enum PinOpToml {
    MoveBy {
        t_start: f64,
        t_end: f64,
        transition: String,
        /// Cubic-Bezier control points `[hr_x, hr_y, hl_x, hl_y]` when
        /// `transition == "bezier"`. None means linear fallback.
        bezier_handles: Option<[f64; 4]>,
    },
    MoveTo {
        t_start: f64,
        t_end: f64,
        transition: String,
        bezier_handles: Option<[f64; 4]>,
    },
    Spin {
        center_mode: String,
        center: [f64; 3],
        axis: [f64; 3],
        angular_velocity: f64,
        t_start: f64,
        t_end: f64,
    },
    Scale {
        center_mode: String,
        center: [f64; 3],
        factor: f64,
        t_start: f64,
        t_end: f64,
        transition: String,
        bezier_handles: Option<[f64; 4]>,
    },
    Torque {
        axis_component: i64,
        magnitude: f64,
        hint_vertex: i64,
        t_start: f64,
        t_end: f64,
    },
    TransformKeyframes {
        keyframe_count: usize,
        t_start: f64,
        t_end: f64,
        rest_translation: [f64; 3],
    },
}

/// Emit `bezier_h_rx / ry / lx / ly` lines when handles are present.
/// Output is keyed at the operation's level so the solver-side reader
/// can pick them up without a separate sub-table.
fn write_bezier_handles(s: &mut String, handles: Option<[f64; 4]>) {
    if let Some(h) = handles {
        let _ = writeln!(s, "bezier_h_rx = {}", format_f64(h[0]));
        let _ = writeln!(s, "bezier_h_ry = {}", format_f64(h[1]));
        let _ = writeln!(s, "bezier_h_lx = {}", format_f64(h[2]));
        let _ = writeln!(s, "bezier_h_ly = {}", format_f64(h[3]));
    }
}

/// Format a single pin-section TOML block (header plus ops).
fn format_pin_section(pin_index: usize, header: &PinHeader, ops: &[PinOpToml]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "[pin-{pin_index}]");
    let _ = writeln!(s, "operation_count = {}", header.operation_count);
    let _ = writeln!(s, "pin = {}", header.pin_count);
    let _ = writeln!(s, "pull = {}", format_f64(header.pull_strength));
    let _ = writeln!(s, "stiffness = {}", format_f64(header.pin_stiffness));
    if let Some(ut) = header.unpin_time {
        let _ = writeln!(s, "unpin_time = {}", format_f64(ut));
    }
    if let Some(pg) = &header.pin_group_id {
        if !pg.is_empty() {
            let _ = writeln!(s, "pin_group_id = \"{pg}\"");
        }
    }
    s.push('\n');

    for (j, op) in ops.iter().enumerate() {
        let _ = writeln!(s, "[pin-{pin_index}-op-{j}]");
        match op {
            PinOpToml::MoveBy {
                t_start,
                t_end,
                transition,
                bezier_handles,
            } => {
                let _ = writeln!(s, "type = \"move_by\"");
                let _ = writeln!(s, "t_start = {}", format_f64(*t_start));
                let _ = writeln!(s, "t_end = {}", format_f64(*t_end));
                let _ = writeln!(s, "transition = \"{transition}\"");
                write_bezier_handles(&mut s, *bezier_handles);
            }
            PinOpToml::MoveTo {
                t_start,
                t_end,
                transition,
                bezier_handles,
            } => {
                let _ = writeln!(s, "type = \"move_to\"");
                let _ = writeln!(s, "t_start = {}", format_f64(*t_start));
                let _ = writeln!(s, "t_end = {}", format_f64(*t_end));
                let _ = writeln!(s, "transition = \"{transition}\"");
                write_bezier_handles(&mut s, *bezier_handles);
            }
            PinOpToml::Spin {
                center_mode,
                center,
                axis,
                angular_velocity,
                t_start,
                t_end,
            } => {
                let _ = writeln!(s, "type = \"spin\"");
                let _ = writeln!(s, "center_mode = \"{center_mode}\"");
                let _ = writeln!(s, "center_x = {}", format_f64(center[0]));
                let _ = writeln!(s, "center_y = {}", format_f64(center[1]));
                let _ = writeln!(s, "center_z = {}", format_f64(center[2]));
                let _ = writeln!(s, "axis_x = {}", format_f64(axis[0]));
                let _ = writeln!(s, "axis_y = {}", format_f64(axis[1]));
                let _ = writeln!(s, "axis_z = {}", format_f64(axis[2]));
                let _ = writeln!(s, "angular_velocity = {}", format_f64(*angular_velocity));
                let _ = writeln!(s, "t_start = {}", format_f64(*t_start));
                let _ = writeln!(s, "t_end = {}", format_f64(*t_end));
            }
            PinOpToml::Scale {
                center_mode,
                center,
                factor,
                t_start,
                t_end,
                transition,
                bezier_handles,
            } => {
                let _ = writeln!(s, "type = \"scale\"");
                let _ = writeln!(s, "center_mode = \"{center_mode}\"");
                let _ = writeln!(s, "center_x = {}", format_f64(center[0]));
                let _ = writeln!(s, "center_y = {}", format_f64(center[1]));
                let _ = writeln!(s, "center_z = {}", format_f64(center[2]));
                let _ = writeln!(s, "factor = {}", format_f64(*factor));
                let _ = writeln!(s, "t_start = {}", format_f64(*t_start));
                let _ = writeln!(s, "t_end = {}", format_f64(*t_end));
                let _ = writeln!(s, "transition = \"{transition}\"");
                write_bezier_handles(&mut s, *bezier_handles);
            }
            PinOpToml::Torque {
                axis_component,
                magnitude,
                hint_vertex,
                t_start,
                t_end,
            } => {
                let _ = writeln!(s, "type = \"torque\"");
                let _ = writeln!(s, "axis_component = {axis_component}");
                let _ = writeln!(s, "magnitude = {}", format_f64(*magnitude));
                let _ = writeln!(s, "hint_vertex = {hint_vertex}");
                let _ = writeln!(s, "t_start = {}", format_f64(*t_start));
                let _ = writeln!(s, "t_end = {}", format_f64(*t_end));
            }
            PinOpToml::TransformKeyframes {
                keyframe_count,
                t_start,
                t_end,
                rest_translation,
            } => {
                let _ = writeln!(s, "type = \"transform_keyframes\"");
                let _ = writeln!(s, "keyframe_count = {keyframe_count}");
                let _ = writeln!(s, "t_start = {}", format_f64(*t_start));
                let _ = writeln!(s, "t_end = {}", format_f64(*t_end));
                let _ = writeln!(s, "rest_tx = {}", format_f64(rest_translation[0]));
                let _ = writeln!(s, "rest_ty = {}", format_f64(rest_translation[1]));
                let _ = writeln!(s, "rest_tz = {}", format_f64(rest_translation[2]));
            }
        }
        s.push('\n');
    }
    s
}

/// Format every pin block into one TOML string. `ops_offsets` slices
/// `ops_flat` into per-pin op runs.
pub fn format_all_pin_sections(
    headers: &[PinHeader],
    ops_offsets: &[usize],
    ops_flat: &[PinOpToml],
) -> Result<String, SceneLoopsError> {
    if ops_offsets.len() != headers.len() + 1 {
        return Err(SceneLoopsError::PinOpsOffsetsMismatch {
            ops_len: ops_offsets.len(),
            headers_len: headers.len(),
        });
    }
    let mut out = String::new();
    for (i, header) in headers.iter().enumerate() {
        let s = ops_offsets[i];
        let e = ops_offsets[i + 1];
        let block = format_pin_section(i, header, &ops_flat[s..e]);
        out.push_str(&block);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Wall + sphere TOML formatting.
//
// Replaces (frontend/_scene_.py ~2078-2098).

#[derive(Debug, Clone)]
pub struct WallToml {
    pub keyframe: usize,
    pub normal: [f64; 3],
    pub transition: String,
    /// `(key, formatted_value_str)` pairs. The value is already
    /// pre-rendered by the caller because Python's `param.list()` emits
    /// heterogeneous types (floats, strings, ints) and we keep that
    /// rendering Python-side; this kernel just stitches them in.
    pub params: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct SphereToml {
    pub keyframe: usize,
    pub is_hemisphere: bool,
    pub is_inverted: bool,
    pub transition: String,
    pub params: Vec<(String, String)>,
}

pub fn format_wall_sections(walls: &[WallToml]) -> String {
    let mut out = String::new();
    for (i, w) in walls.iter().enumerate() {
        let _ = writeln!(out, "[wall-{i}]");
        let _ = writeln!(out, "keyframe = {}", w.keyframe);
        let _ = writeln!(out, "nx = {}", format_f64(w.normal[0]));
        let _ = writeln!(out, "ny = {}", format_f64(w.normal[1]));
        let _ = writeln!(out, "nz = {}", format_f64(w.normal[2]));
        let _ = writeln!(out, "transition = \"{}\"", w.transition);
        for (k, v) in &w.params {
            let _ = writeln!(out, "{k} = {v}");
        }
        out.push('\n');
    }
    out
}

pub fn format_sphere_sections(spheres: &[SphereToml]) -> String {
    let mut out = String::new();
    for (i, sp) in spheres.iter().enumerate() {
        let _ = writeln!(out, "[sphere-{i}]");
        let _ = writeln!(out, "keyframe = {}", sp.keyframe);
        let _ = writeln!(
            out,
            "hemisphere = {}",
            if sp.is_hemisphere { "true" } else { "false" }
        );
        let _ = writeln!(
            out,
            "invert = {}",
            if sp.is_inverted { "true" } else { "false" }
        );
        let _ = writeln!(out, "transition = \"{}\"", sp.transition);
        for (k, v) in &sp.params {
            let _ = writeln!(out, "{k} = {v}");
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// dyn_param.txt formatting.
//
// Replaces (frontend/_scene_.py ~2152-2163). Each entry is either a
// (time, [vx, vy, vz]) velocity tuple or a (t_start, t_end) collision
// window. Inputs are pre-classified by the caller; this just emits the
// flat text.

#[derive(Debug, Clone)]
pub enum DynParamEntry {
    /// `(time, [vx, vy, vz])`
    Velocity(f64, [f64; 3]),
    /// `(t_start, t_end)`
    CollisionWindow(f64, f64),
}

pub fn format_dyn_param_sections(blocks: &[(String, Vec<DynParamEntry>)]) -> String {
    let mut out = String::new();
    for (key, entries) in blocks {
        let _ = writeln!(out, "[{key}]");
        for entry in entries {
            match entry {
                DynParamEntry::Velocity(t, v) => {
                    let _ = writeln!(
                        out,
                        "{} {} {} {}",
                        format_f64(*t),
                        format_f64(v[0]),
                        format_f64(v[1]),
                        format_f64(v[2])
                    );
                }
                DynParamEntry::CollisionWindow(s, e) => {
                    let _ = writeln!(out, "{} {}", format_f64(*s), format_f64(*e));
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// static_transform header section.
//
// Replaces (frontend/_scene_.py ~2001-2012).
pub fn format_static_transform_header(
    object_count: usize,
    total_keyframes: usize,
    keyframe_counts: &[usize],
    vert_counts: &[usize],
    vert_offsets: &[usize],
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "[static_transform]");
    let _ = writeln!(out, "object_count = {object_count}");
    let _ = writeln!(out, "total_keyframes = {total_keyframes}");
    let _ = writeln!(out, "keyframe_counts = {}", format_int_list(keyframe_counts));
    let _ = writeln!(out, "vert_counts = {}", format_int_list(vert_counts));
    let _ = writeln!(out, "vert_offsets = {}", format_int_list(vert_offsets));
    out.push('\n');
    out
}

fn format_int_list(xs: &[usize]) -> String {
    let mut s = String::from("[");
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let _ = write!(s, "{x}");
    }
    s.push(']');
    s
}

// ---------------------------------------------------------------------------
// Per-object axis bound min/max.
//
// Replaces (frontend/_scene_.py ~2913-2920 and ~2937-2945): the loop
// that calls `scene_axis_min_max` per object then aggregates. Inputs:
// flat per-object vertex arrays (one big concat) plus offsets so the
// kernel can splice each object out. axis: 0|1|2.
pub fn per_object_axis_bound(
    flat_verts: &[f64],
    object_offsets: &[usize],
    axis: usize,
    is_min: bool,
) -> Result<f64, SceneLoopsError> {
    if axis > 2 {
        return Err(SceneLoopsError::InvalidAxis { axis });
    }
    if object_offsets.is_empty() {
        return Ok(if is_min {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        });
    }
    let mut acc = if is_min {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    for w in object_offsets.windows(2) {
        let s = w[0];
        let e = w[1];
        if s == e {
            continue;
        }
        let block = &flat_verts[s * 3..e * 3];
        for r in 0..(e - s) {
            let v = block[3 * r + axis];
            if is_min {
                if v < acc {
                    acc = v;
                }
            } else if v > acc {
                acc = v;
            }
        }
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// Concat lists of i64 indices.
//
// Replaces (frontend/_scene_.py ~3122-3125): the per-object pin index
// flatten loop. Input: list of i64 slices.
pub fn concat_i64_lists(parts: &[&[i64]]) -> Vec<i64> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

// ---------------------------------------------------------------------------
// f64 formatter that matches Python `f"{float(x)}"` (str(float)) for
// the common values used in TOML export. The TOML parser tolerates any
// roundtrippable form, so we use Rust's default `{}` which emits a
// minimal decimal representation. For integer-valued floats Python
// emits e.g. "0.0" (with the ".0"); Rust's default emits "0" so we add
// the trailing `.0`. This keeps file diffs stable.
fn format_f64(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    let s = format!("{x}");
    if !s.contains('.') && !s.contains('e') && !s.contains('E') && !s.contains("inf") {
        format!("{s}.0")
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stitch_preview_two_lines() {
        // 4 dynamic verts.
        #[rustfmt::skip]
        let vert: Vec<f64> = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        // 6-wide: degenerate source [0,0,0], target tri (1,2,3).
        let stitch_ind: Vec<i64> = vec![0, 0, 0, 1, 2, 3];
        // source ws=[1,0,0], target wt=[0.5,0.25,0.25]
        let stitch_w: Vec<f64> = vec![1.0, 0.0, 0.0, 0.5, 0.25, 0.25];

        let (sv, se) = stitch_preview_lines(&vert, 4, &stitch_ind, &stitch_w, 1);
        // src is vert[0] = (0,0,0).
        assert!((sv[0] - 0.0).abs() < 1e-15);
        assert!((sv[1] - 0.0).abs() < 1e-15);
        assert!((sv[2] - 0.0).abs() < 1e-15);
        // target = 0.5*(1,0,0) + 0.25*(0,1,0) + 0.25*(0,0,1) = (0.5,0.25,0.25)
        assert!((sv[3] - 0.5).abs() < 1e-15);
        assert!((sv[4] - 0.25).abs() < 1e-15);
        assert!((sv[5] - 0.25).abs() < 1e-15);
        // Edge points to absolute indices 4 and 5 (n_vert = 4).
        assert_eq!(se, vec![4, 5]);
    }

    #[test]
    fn face_uv_expand_basic() {
        // 4 verts in UV.
        let uv = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let faces = vec![0_i64, 1, 2, 0, 2, 3];
        let out = face_uv_expand(&uv, 4, &faces, 2).unwrap();
        assert_eq!(
            out,
            vec![
                0.0, 0.0, 1.0, 0.0, 1.0, 1.0, // tri 0
                0.0, 0.0, 1.0, 1.0, 0.0, 1.0, // tri 1
            ]
        );
    }

    #[test]
    fn face_uv_expand_rejects_oob() {
        let uv = vec![0.0, 0.0, 1.0, 0.0];
        let faces = vec![0_i64, 1, 5];
        assert!(face_uv_expand(&uv, 2, &faces, 1).is_err());
    }

    #[test]
    fn collect_pin_marker_skips_hidden() {
        let hide = vec![false, true, false];
        let offsets = vec![0, 2, 5, 7];
        let data = vec![10_i64, 11, 20, 21, 22, 30, 31];
        let out = collect_pin_marker_indices(&hide, &offsets, &data);
        assert_eq!(out, vec![10, 11, 30, 31]);
    }

    #[test]
    fn concat_static_transform_anims_validates_lengths() {
        let kf = vec![2_usize, 3];
        let times = vec![0.0; 5];
        let trans = vec![0.0; 15];
        let quats = vec![0.0; 20];
        let scales = vec![0.0; 15];
        assert_eq!(
            concat_static_transform_anims(&kf, &times, &trans, &quats, &scales).unwrap(),
            5
        );
        assert!(
            concat_static_transform_anims(&kf, &vec![0.0; 4], &trans, &quats, &scales).is_err()
        );
    }

    #[test]
    fn pack_segments_round_trip() {
        let interp = vec!["LINEAR", "BEZIER", "CONSTANT"];
        let hr = vec![1.0 / 3.0, 0.0, 0.5, 0.5, 1.0 / 3.0, 0.0];
        let hl = vec![2.0 / 3.0, 1.0, 0.5, 0.5, 2.0 / 3.0, 1.0];
        let (codes, handles) = pack_transform_keyframe_segments(&interp, &hr, &hl).unwrap();
        assert_eq!(codes, vec![0, 1, 2]);
        // Row 1 (BEZIER): r=(0.5,0.5), l=(0.5,0.5).
        assert_eq!(&handles[4..8], &[0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn pack_segments_rejects_unknown() {
        let interp = vec!["LINEAR", "BANANA"];
        let hr = vec![0.0; 4];
        let hl = vec![0.0; 4];
        assert!(pack_transform_keyframe_segments(&interp, &hr, &hl).is_err());
    }

    #[test]
    fn pin_section_move_by() {
        let header = PinHeader {
            operation_count: 1,
            pin_count: 4,
            pull_strength: 0.0,
            pin_stiffness: 1.0,
            unpin_time: None,
            pin_group_id: None,
        };
        let ops = vec![PinOpToml::MoveBy {
            t_start: 0.0,
            t_end: 1.0,
            transition: "LINEAR".to_string(),
            bezier_handles: None,
        }];
        let s = format_pin_section(0, &header, &ops);
        assert!(s.contains("[pin-0]"));
        assert!(s.contains("operation_count = 1"));
        assert!(s.contains("pin = 4"));
        assert!(s.contains("[pin-0-op-0]"));
        assert!(s.contains("type = \"move_by\""));
        assert!(s.contains("t_start = 0.0"));
        assert!(s.contains("t_end = 1.0"));
        assert!(!s.contains("bezier_h_"));
    }

    #[test]
    fn pin_section_emits_bezier_handles_when_present() {
        let header = PinHeader {
            operation_count: 1,
            pin_count: 1,
            pull_strength: 0.0,
            pin_stiffness: 1.0,
            unpin_time: None,
            pin_group_id: None,
        };
        let ops = vec![PinOpToml::MoveBy {
            t_start: 0.0,
            t_end: 1.0,
            transition: "bezier".to_string(),
            bezier_handles: Some([0.42, 0.0, 0.58, 1.0]),
        }];
        let s = format_pin_section(0, &header, &ops);
        assert!(s.contains("transition = \"bezier\""));
        assert!(s.contains("bezier_h_rx = 0.42"));
        assert!(s.contains("bezier_h_ry = 0.0"));
        assert!(s.contains("bezier_h_lx = 0.58"));
        assert!(s.contains("bezier_h_ly = 1.0"));
    }

    #[test]
    fn pin_section_with_unpin_time_and_group() {
        let header = PinHeader {
            operation_count: 0,
            pin_count: 8,
            pull_strength: 0.5,
            pin_stiffness: 2.0,
            unpin_time: Some(2.5),
            pin_group_id: Some("g0".to_string()),
        };
        let s = format_pin_section(3, &header, &[]);
        assert!(s.contains("unpin_time = 2.5"));
        assert!(s.contains("pin_group_id = \"g0\""));
        assert!(s.contains("stiffness = 2"));
    }

    #[test]
    fn pin_section_torque() {
        let header = PinHeader {
            operation_count: 1,
            pin_count: 1,
            pull_strength: 0.0,
            pin_stiffness: 1.0,
            unpin_time: None,
            pin_group_id: None,
        };
        let ops = vec![PinOpToml::Torque {
            axis_component: 1,
            magnitude: 7.5,
            hint_vertex: 42,
            t_start: 0.0,
            t_end: 5.0,
        }];
        let s = format_pin_section(0, &header, &ops);
        assert!(s.contains("type = \"torque\""));
        assert!(s.contains("axis_component = 1"));
        assert!(s.contains("magnitude = 7.5"));
        assert!(s.contains("hint_vertex = 42"));
    }

    #[test]
    fn wall_section_emits_normal_and_params() {
        let walls = vec![WallToml {
            keyframe: 1,
            normal: [0.0, 1.0, 0.0],
            transition: "LINEAR".to_string(),
            params: vec![("friction".into(), "0.4".into())],
        }];
        let s = format_wall_sections(&walls);
        assert!(s.contains("[wall-0]"));
        assert!(s.contains("keyframe = 1"));
        assert!(s.contains("nx = 0.0"));
        assert!(s.contains("ny = 1.0"));
        assert!(s.contains("transition = \"LINEAR\""));
        assert!(s.contains("friction = 0.4"));
    }

    #[test]
    fn sphere_section_emits_flags() {
        let spheres = vec![SphereToml {
            keyframe: 1,
            is_hemisphere: true,
            is_inverted: false,
            transition: "LINEAR".into(),
            params: vec![],
        }];
        let s = format_sphere_sections(&spheres);
        assert!(s.contains("hemisphere = true"));
        assert!(s.contains("invert = false"));
    }

    #[test]
    fn dyn_param_velocity_then_window() {
        let blocks = vec![
            (
                "vel".to_string(),
                vec![DynParamEntry::Velocity(0.0, [0.0, -5.0, 0.0])],
            ),
            (
                "win".to_string(),
                vec![DynParamEntry::CollisionWindow(0.2, 1.0)],
            ),
        ];
        let s = format_dyn_param_sections(&blocks);
        assert!(s.contains("[vel]"));
        assert!(s.contains("0.0 0.0 -5.0 0.0"));
        assert!(s.contains("[win]"));
        assert!(s.contains("0.2 1.0"));
    }

    #[test]
    fn static_transform_header_format() {
        let s = format_static_transform_header(2, 5, &[2, 3], &[10, 12], &[0, 10]);
        assert!(s.contains("[static_transform]"));
        assert!(s.contains("object_count = 2"));
        assert!(s.contains("total_keyframes = 5"));
        assert!(s.contains("keyframe_counts = [2, 3]"));
        assert!(s.contains("vert_counts = [10, 12]"));
        assert!(s.contains("vert_offsets = [0, 10]"));
    }

    #[test]
    fn per_object_axis_bound_min_max() {
        // 2 objects: object 0 has 2 verts, object 1 has 3 verts.
        let v: Vec<f64> = vec![
            -1.0, 2.0, 0.0, 0.5, 5.0, 0.0, // obj 0
            0.0, -3.0, 0.0, 1.0, 0.0, 0.0, 2.0, 4.0, 0.0, // obj 1
        ];
        let offsets = vec![0_usize, 2, 5];
        let lo_y = per_object_axis_bound(&v, &offsets, 1, true).unwrap();
        let hi_y = per_object_axis_bound(&v, &offsets, 1, false).unwrap();
        assert_eq!(lo_y, -3.0);
        assert_eq!(hi_y, 5.0);
    }

    #[test]
    fn per_object_axis_bound_empty_returns_inf() {
        let v: Vec<f64> = vec![];
        let lo = per_object_axis_bound(&v, &[], 0, true).unwrap();
        assert!(lo.is_infinite() && lo > 0.0);
    }

    #[test]
    fn concat_i64_lists_basic() {
        let a: Vec<i64> = vec![1, 2, 3];
        let b: Vec<i64> = vec![];
        let c: Vec<i64> = vec![10];
        let out = concat_i64_lists(&[a.as_slice(), b.as_slice(), c.as_slice()]);
        assert_eq!(out, vec![1, 2, 3, 10]);
    }

    #[test]
    fn format_f64_integer_appends_dot_zero() {
        assert_eq!(format_f64(0.0), "0.0");
        assert_eq!(format_f64(1.0), "1.0");
        assert_eq!(format_f64(-2.0), "-2.0");
    }

    #[test]
    fn format_f64_decimal_unchanged() {
        assert_eq!(format_f64(0.5), "0.5");
        assert_eq!(format_f64(-0.25), "-0.25");
    }

    #[test]
    fn format_f64_handles_nonfinite() {
        assert_eq!(format_f64(f64::NAN), "nan");
        assert_eq!(format_f64(f64::INFINITY), "inf");
        assert_eq!(format_f64(f64::NEG_INFINITY), "-inf");
    }
}
