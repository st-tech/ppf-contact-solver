// File: scene.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::builder::{convert_prop, dedup_param, make_collision_mesh, SandParams};
use super::data::*;
use super::{CVec, MeshSet, ParamSet, ProgramArgs, Props, SimArgs, SimMesh};
use bytemuck::{cast_slice, Pod};
use log::{error, warn};
use more_asserts::*;
use na::{
    Const, Matrix, Matrix2x3, Matrix2xX, Matrix3xX, Matrix4xX, Matrix6xX, VecStorage, Vector3,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, Read, Write};
use toml::Value;

pub struct CollisionWindowTable {
    pub vert_dmap: Vec<u32>,
    pub windows: Vec<f32>,
    pub window_counts: Vec<u32>,
    pub n_groups: u32,
    pub has_windows: bool,
}

impl CollisionWindowTable {
    pub fn empty() -> Self {
        Self {
            vert_dmap: Vec::new(),
            windows: Vec::new(),
            window_counts: Vec::new(),
            n_groups: 0,
            has_windows: false,
        }
    }
}

/// A streamed time-varying rest shape: per-keyframe full rest-vertex sets
/// (each `n_vert` columns, same coordinate space as the static `rest_vert`)
/// and their keyframe times in seconds. Built frontend-side by blending the
/// captured pull-pin deformation into the original rest pose per vertex; the
/// solver recomputes `inv_rest` from each frame and interpolates between them.
struct RestVertSchedule {
    times: Vec<f64>,
    frames: Vec<Matrix3xX<f32>>,
}

pub struct Scene {
    args: SimArgs,
    dyn_args: DynParamTable,
    displacement: Matrix3xX<f32>,
    vert_dmap: Vec<u32>,
    vert: Matrix3xX<f32>,
    vel: Matrix3xX<f32>,
    uv: Option<Vec<Matrix2x3<f32>>>,
    rod: Matrix2xX<usize>,
    tri: Matrix3xX<usize>,
    tet: Matrix4xX<usize>,
    static_vert_dmap: Vec<u32>,
    static_vert: Matrix3xX<f32>,
    static_tri: Matrix3xX<usize>,
    stitch_ind: Matrix6xX<usize>,
    stitch_w: Matrix6xX<f32>,
    stitch_stiffness: Vec<f32>,
    pin: Vec<Pin>,
    wall: Vec<InvisibleWall>,
    sphere: Vec<InvisibleSphere>,
    rest_vert: Option<Matrix3xX<f32>>,
    rest_vert_mask: Vec<bool>,
    rest_vert_schedule: Option<RestVertSchedule>,
    /// Per-object bending reference rest shape: full per-vertex positions
    /// (unmasked entries equal the initial vert) used to compute hinge rest
    /// angles for objects that opted into a reference rest angle. `None` when
    /// no object in the scene uses one.
    bend_rest_vert: Option<Matrix3xX<f32>>,
    /// Per-vertex mask marking which vertices belong to an object with a
    /// reference rest angle. Empty when `bend_rest_vert` is `None`.
    bend_rest_vert_mask: Vec<bool>,
    shell_count: usize,
    rod_param: Vec<(String, ParamValueList)>,
    tri_param: Vec<(String, ParamValueList)>,
    tet_param: Vec<(String, ParamValueList)>,
    static_param: Vec<(String, ParamValueList)>,
    /// Granular (SAND) scalar material params, one entry per `sand-*.bin`
    /// param file (each a single float, len 1). A faceless particle cloud
    /// has no elements, so these carry the grain mass / contact-offset /
    /// contact-gap / friction outside the per-element assert. Empty when the
    /// scene has no SAND object.
    sand_param: Vec<(String, ParamValueList)>,
    /// One row per PDRD body, packed as f32: vertex_start, vertex_count,
    /// volume, centroid (3), rest_gram_inv (9), mass_per_vertex. 16
    /// floats per row. Written by the Python frontend.
    pdrd_body_rows: Vec<f32>,
    /// Per-vertex 1-based PDRD body id (0 = not in an PDRD body), one
    /// u32 per global vertex.
    pdrd_vert_index: Vec<u32>,
    /// Flat list of global vertex indices participating in PDRD
    /// bodies, body-major. Body slices are contiguous in this list
    /// per `PdrdBodyProp.vertex_start/vertex_count`.
    pdrd_vert_list: Vec<u32>,
    /// Centered rest position per entry of `pdrd_vert_list`, packed
    /// (x, y, z). Length 3 * pdrd_vert_list.len().
    pdrd_rest_centered: Vec<f32>,
}

enum ParamValueList {
    Model(Vec<Model>),
    Value(Vec<f32>),
}

// Extract the i-th scalar from a value list, panicking with the offending
// key in the message when the list is the wrong variant. `key` is the literal
// parameter key (e.g. "young-mod") so a panic maps straight back to the
// param file.
fn as_value(value: &ParamValueList, i: usize, key: &str) -> f32 {
    match value {
        ParamValueList::Value(v) => v[i],
        _ => panic!("Expected parameter '{key}' to be a value list"),
    }
}

// Extract the i-th model from a name list, panicking with the offending key
// in the message when the list is the wrong variant.
fn as_model(value: &ParamValueList, i: usize, key: &str) -> Model {
    match value {
        ParamValueList::Model(v) => v[i],
        _ => panic!("Expected parameter '{key}' to be a name list"),
    }
}

// Apply per-vertex instancing displacement in place: each column of `base`
// gains `displacement[dmap[i]]`. Shared by the dynamic and static vertex
// paths so they cannot drift in their displacement indexing.
fn apply_displacement(
    base: &mut Matrix3xX<f32>,
    displacement: &Matrix3xX<f32>,
    dmap: &[u32],
) {
    for (i, mut x) in base.column_iter_mut().enumerate() {
        x += displacement.column(dmap[i] as usize);
    }
}

enum PinOperation {
    MoveBy {
        delta: Matrix3xX<f32>,
        t_start: f64,
        t_end: f64,
        transition: String,
        /// Cubic-Bezier control points `[hr_x, hr_y, hl_x, hl_y]` when
        /// `transition == "bezier"`. None means linear fallback,
        /// matching the behavior of older saves that don't carry the
        /// handles in their TOML.
        bezier_handles: Option<[f64; 4]>,
    },
    MoveTo {
        target: Matrix3xX<f32>,
        t_start: f64,
        t_end: f64,
        transition: String,
        bezier_handles: Option<[f64; 4]>,
    },
    Spin {
        center: Vector3<f32>,
        axis: Vector3<f32>,
        angular_velocity: f32,
        t_start: f64,
        t_end: f64,
    },
    Scale {
        center: Vector3<f32>,
        factor: f32,
        t_start: f64,
        t_end: f64,
        transition: String,
        bezier_handles: Option<[f64; 4]>,
    },
    Torque {
        axis_component: u32,
        magnitude: f32,
        hint_vertex: usize,
        t_start: f64,
        t_end: f64,
    },
    /// Sparse TRS keyframes evaluated per-vertex as R(t)*S(t)*local + T(t),
    /// then offset by -rest_t so that `position + displacement` yields the
    /// interpolated world pose.  Rotation uses slerp; segments may carry
    /// Bezier easing.
    TransformKeyframes {
        local: Matrix3xX<f32>,
        times: Vec<f64>,
        translations: Vec<[f64; 3]>,
        quaternions: Vec<[f64; 4]>,
        scales: Vec<[f64; 3]>,
        interps: Vec<u8>,
        handles: Vec<[f64; 4]>,
        rest_t: [f64; 3],
    },
}

struct Pin {
    index: Vec<usize>,
    operations: Vec<PinOperation>,
    unpin_time: Option<f64>,
    pull_w: f32,
    /// Optional per-vertex pull weight (aligned to `index`). When present,
    /// overrides the scalar `pull_w` per vertex, so a single holder can
    /// pull different vertices with different (graceful, diffused)
    /// strengths. Absent for hard pins and scalar pull pins.
    pull_weights: Option<Vec<f32>>,
    /// Per-pin scale on the moving (kinematic) constraint force.
    /// 1.0 leaves it unchanged; applied only when the pin is kinematic.
    stiffness: f32,
    pin_group_id: String,
}

struct InvisibleSphere {
    center: Matrix3xX<f32>,
    radius: Vec<f32>,
    timing: Vec<f64>,
    inverted: bool,
    hemisphere: bool,
    transition: String,
    contact_gap: f32,
    friction: f32,
    /// Simulation time in seconds after which this collider stops acting.
    /// -1.0 means no limit (default).
    active_duration: f32,
    /// Maximum penetration depth (m) beyond which a vertex is considered
    /// pass-through: its contact and CCD contribution from this sphere
    /// are ignored. Must be > 0.
    thickness: f32,
}

struct InvisibleWall {
    normal: Vector3<f32>,
    position: Matrix3xX<f32>,
    timing: Vec<f64>,
    transition: String,
    contact_gap: f32,
    friction: f32,
    /// Simulation time in seconds after which this collider stops acting.
    /// -1.0 means no limit (default).
    active_duration: f32,
    /// Maximum penetration depth (m) below the wall beyond which a vertex
    /// is considered pass-through. Must be > 0.
    thickness: f32,
}

#[derive(Debug, Deserialize)]
struct Config {
    param: SimArgs,
}

type MatReadResult<T, const C: usize> =
    io::Result<Matrix<T, Const<C>, na::Dyn, VecStorage<T, Const<C>, na::Dyn>>>;
#[derive(Clone, Copy)]
enum DynParamValue {
    Scalar(f64),
    Vec3([f64; 3]),
}

impl DynParamValue {
    fn lerp(self, other: Self, w: f64) -> Self {
        match (self, other) {
            (DynParamValue::Scalar(a), DynParamValue::Scalar(b)) => {
                DynParamValue::Scalar(a * (1.0 - w) + b * w)
            }
            (DynParamValue::Vec3(a), DynParamValue::Vec3(b)) => DynParamValue::Vec3([
                a[0] * (1.0 - w) + b[0] * w,
                a[1] * (1.0 - w) + b[1] * w,
                a[2] * (1.0 - w) + b[2] * w,
            ]),
            _ => self,
        }
    }
}

type DynParamTable = Vec<(String, Vec<(f64, DynParamValue)>)>;

fn read_mat_from_file<T, const C: usize>(path: &str) -> MatReadResult<T, C>
where
    T: Pod + std::cmp::PartialEq + std::fmt::Debug,
{
    let mut file = File::open(path)?;
    let mut buff = Vec::new();
    file.read_to_end(&mut buff)?;
    if !buff.len().is_multiple_of(std::mem::size_of::<T>()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Data length is not a multiple of the element size",
        ));
    }
    let data: &[T] = cast_slice(&buff);
    if !data.len().is_multiple_of(C) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Data length is not a multiple of {C}"),
        ));
    }
    let n_column = na::Dyn(data.len() / C);
    let n_row = na::Const::<C>;
    Ok(unsafe {
        Matrix::<T, na::Const<C>, na::Dyn, VecStorage<T, na::Const<C>,na::Dyn >>::from_data_statically_unchecked(
            VecStorage::new(n_row, n_column, data.to_vec()),
        )
    })
}

fn read_vec<T>(path: &str) -> io::Result<Vec<T>>
where
    T: bytemuck::AnyBitPattern,
{
    let mut buff = Vec::new();
    let mut file = File::open(path)?;
    file.read_to_end(&mut buff)?;
    Ok(cast_slice(&buff).to_vec())
}

fn read_dyn_param(path: &str) -> io::Result<DynParamTable> {
    let mut result = Vec::new();
    let mut curr_entry_name = String::new();
    let mut curr_entry = Vec::new();
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.starts_with('[') && line.ends_with(']') {
            if !curr_entry_name.is_empty() {
                result.push((curr_entry_name, curr_entry));
                curr_entry = Vec::new();
            }
            curr_entry_name = line[1..line.len() - 1].to_string();
        } else if !line.is_empty() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let parse_f64 = |field: &str, token: &str| -> io::Result<f64> {
                token.parse::<f64>().map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{path}: bad {field} in line '{line}': {e}"),
                    )
                })
            };
            let time = parse_f64("time", parts[0])?;
            if parts.len() == 4 {
                let x = parse_f64("x", parts[1])?;
                let y = parse_f64("y", parts[2])?;
                let z = parse_f64("z", parts[3])?;
                curr_entry.push((time, DynParamValue::Vec3([x, y, z])));
            } else if parts.len() == 2 {
                let value = parse_f64("value", parts[1])?;
                curr_entry.push((time, DynParamValue::Scalar(value)));
            } else {
                warn!("{path}: skipping line with {} tokens: '{line}'", parts.len());
            }
        }
    }
    if !curr_entry_name.is_empty() {
        result.push((curr_entry_name, curr_entry));
    }
    Ok(result)
}

impl Scene {
    pub fn new(args: &ProgramArgs) -> Self {
        assert!(std::path::Path::new(&args.path).exists());

        let toml_path = format!("{}/info.toml", args.path);
        let content = fs::read_to_string(toml_path).expect("Failed to read the TOML file");
        let parsed: Value = content.parse::<Value>().expect("Failed to parse TOML");
        let read_usize = |count: &Value, key: &str| {
            count
                .get(key)
                .and_then(|v| v.as_integer())
                .unwrap_or_else(|| panic!("Failed to read {key}")) as usize
        };
        let read_bool = |count: &Value, key: &str| {
            count
                .get(key)
                .and_then(|v| v.as_bool())
                .unwrap_or_else(|| panic!("Failed to read {key}"))
        };
        let read_f32 = |count: &Value, key: &str| {
            count
                .get(key)
                .and_then(|v| v.as_float())
                .unwrap_or_else(|| panic!("Failed to read {key}")) as f32
        };
        let read_f64 = |count: &Value, key: &str| {
            count
                .get(key)
                .and_then(|v| v.as_float())
                .unwrap_or_else(|| panic!("Failed to read {key}"))
        };
        // Optional bezier handles (4 floats) at keys
        // `bezier_h_rx`, `bezier_h_ry`, `bezier_h_lx`, `bezier_h_ly`.
        // Returns None when any of the keys is absent (older saves).
        let read_bezier_handles = |count: &Value| -> Option<[f64; 4]> {
            let rx = count.get("bezier_h_rx").and_then(|v| v.as_float())?;
            let ry = count.get("bezier_h_ry").and_then(|v| v.as_float())?;
            let lx = count.get("bezier_h_lx").and_then(|v| v.as_float())?;
            let ly = count.get("bezier_h_ly").and_then(|v| v.as_float())?;
            Some([rx, ry, lx, ly])
        };
        let read_string = |count: &Value, key: &str| {
            count
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("Failed to read {key}"))
                .to_string()
        };
        let read_optional_f32 = |count: &Value, key: &str| -> Option<f32> {
            count.get(key).and_then(|v| v.as_float()).map(|x| x as f32)
        };

        let count = parsed.get("count").expect("Failed to read count");
        let n_vert = read_usize(count, "vert");
        let n_rod = read_usize(count, "rod");
        let n_tri = read_usize(count, "tri");
        let n_tet = read_usize(count, "tet");
        let n_static_vert = read_usize(count, "static_vert");
        let n_static_tri = read_usize(count, "static_tri");
        let n_pin_block = read_usize(count, "pin_block");
        let n_wall = read_usize(count, "wall");
        let n_sphere = read_usize(count, "sphere");
        let n_stitch = read_usize(count, "stitch");
        let shell_count = read_usize(count, "shell_count");

        let displacement_path = format!("{}/bin/displacement.bin", args.path);
        let vert_dmap_path = format!("{}/bin/vert_dmap.bin", args.path);
        let vert_path = format!("{}/bin/vert.bin", args.path);
        let vel_path = format!("{}/bin/vel.bin", args.path);
        let uv_path = format!("{}/bin/uv.bin", args.path);
        let rod_path = format!("{}/bin/rod.bin", args.path);
        let tri_path = format!("{}/bin/tri.bin", args.path);
        let tet_path = format!("{}/bin/tet.bin", args.path);
        let static_vert_dmap_path = format!("{}/bin/static_vert_dmap.bin", args.path);
        let static_vert_path = format!("{}/bin/static_vert.bin", args.path);
        let static_tri_path = format!("{}/bin/static_tri.bin", args.path);
        let stitch_ind_path = format!("{}/bin/stitch_ind.bin", args.path);
        let stitch_w_path = format!("{}/bin/stitch_w.bin", args.path);
        let stitch_stiffness_path = format!("{}/bin/stitch_stiffness.bin", args.path);

        // Uniform world scaling: every input geometric position/length is scaled
        // by `ws` on ingest (and the per-frame output is divided back by it in
        // backend.rs). Applied to the f64 value before the f32 cast so no
        // precision is lost. `ws == 1.0` (the default) is a no-op. All the
        // `.map(|x| (x as f64 * ws) as f32)` reads below are world-space positions,
        // centers, or deltas; the velocity, keyframe translations, rest offset, and
        // sphere radius are scaled explicitly. Directions, rotations, indices,
        // weights, UVs, and physical params (gravity, absolute gaps) are NOT scaled.
        // `args` is ProgramArgs (no sim params), so read world_scaling from
        // param.toml now (the full Config loads again at its original site below;
        // param.toml is tiny, so the extra parse is negligible).
        let ws = {
            let p = format!("{}/param.toml", args.path);
            let c: Config = toml::from_str(
                &fs::read_to_string(&p).expect("Failed to read param.toml"),
            )
            .expect("Failed to parse param.toml");
            c.param.world_scaling as f64
        };
        // ws must be strictly positive: 0 collapses the scene to the origin and a
        // negative value mirror-flips all geometry (inverting winding/normals).
        assert!(
            ws > 0.0,
            "world-scaling must be > 0 (got {ws}); 1.0 disables scaling"
        );

        let displacement_mat = read_mat_from_file::<f64, 3>(&displacement_path)
            .expect("Failed to read displacement")
            .map(|x| (x as f64 * ws) as f32);
        let vert_dmap_mat = read_vec::<u32>(&vert_dmap_path).expect("Failed to read vert_dmap");
        assert_eq!(vert_dmap_mat.len(), n_vert, "vert_dmap size mismatch");
        let vert_mat = read_mat_from_file::<f64, 3>(&vert_path)
            .expect("Failed to read vert")
            .map(|x| (x as f64 * ws) as f32);
        // Velocity has units length/time; scale it with the geometry so seeded
        // motion stays consistent after the output is divided back by `ws`.
        let vel_mat = read_mat_from_file::<f32, 3>(&vel_path)
            .expect("Failed to read velocity")
            .map(|x| (x as f64 * ws) as f32);
        let has_rest_vert = count
            .get("has_rest_vert")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Time-varying rest shape (the captured pull-pin deformation). Optional
        // and absent on legacy payloads, so the flag read must not panic.
        let has_rest_vert_anim = count
            .get("has_rest_vert_anim")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // The mask is shared between the static rest_vert and the time-varying
        // schedule (make_mesh / rest_shape_schedule both consult it), so load
        // it whenever either source is present.
        let rest_vert_mask = if has_rest_vert || has_rest_vert_anim {
            let rest_vert_mask_path = format!("{}/bin/rest_vert_mask.bin", args.path);
            let mask_bytes = read_vec::<u8>(&rest_vert_mask_path)
                .expect("Failed to read rest_vert_mask");
            let mask: Vec<bool> = mask_bytes.iter().map(|&b| b != 0).collect();
            assert_eq!(mask.len(), n_vert, "rest_vert_mask size mismatch");
            mask
        } else {
            vec![false; n_vert]
        };
        let rest_vert_mat = if has_rest_vert {
            let rest_vert_path = format!("{}/bin/rest_vert.bin", args.path);
            let mat = read_mat_from_file::<f64, 3>(&rest_vert_path)
                .expect("Failed to read rest_vert")
                .map(|x| (x as f64 * ws) as f32);
            assert_eq!(mat.ncols(), n_vert, "rest_vert size mismatch");
            Some(mat)
        } else {
            None
        };
        // Stored frame-major in rest_vert_anim.bin as (n_frames * n_vert)
        // columns of 3 rows, split here into one Matrix3xX per keyframe.
        let rest_vert_schedule = if has_rest_vert_anim {
            let n_frames = read_usize(count, "rest_vert_anim_frames");
            let anim_path = format!("{}/bin/rest_vert_anim.bin", args.path);
            let times_path = format!("{}/bin/rest_vert_times.bin", args.path);
            let flat = read_mat_from_file::<f64, 3>(&anim_path)
                .expect("Failed to read rest_vert_anim")
                .map(|x| (x as f64 * ws) as f32);
            assert_eq!(
                flat.ncols(),
                n_frames * n_vert,
                "rest_vert_anim size mismatch"
            );
            let times = read_vec::<f64>(&times_path).expect("Failed to read rest_vert_times");
            assert_eq!(times.len(), n_frames, "rest_vert_times size mismatch");
            let frames: Vec<Matrix3xX<f32>> = (0..n_frames)
                .map(|k| flat.columns(k * n_vert, n_vert).into_owned())
                .collect();
            Some(RestVertSchedule { times, frames })
        } else {
            None
        };
        // Bending reference rest shape (optional, absent on legacy payloads).
        let has_bend_rest_vert = count
            .get("has_bend_rest_vert")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let (bend_rest_vert, bend_rest_vert_mask) = if has_bend_rest_vert {
            let vert_path = format!("{}/bin/bend_rest_vert.bin", args.path);
            let mat = read_mat_from_file::<f64, 3>(&vert_path)
                .expect("Failed to read bend_rest_vert")
                .map(|x| (x as f64 * ws) as f32);
            assert_eq!(mat.ncols(), n_vert, "bend_rest_vert size mismatch");
            let mask_path = format!("{}/bin/bend_rest_vert_mask.bin", args.path);
            let mask_bytes =
                read_vec::<u8>(&mask_path).expect("Failed to read bend_rest_vert_mask");
            let mask: Vec<bool> = mask_bytes.iter().map(|&b| b != 0).collect();
            assert_eq!(mask.len(), n_vert, "bend_rest_vert_mask size mismatch");
            (Some(mat), mask)
        } else {
            (None, Vec::new())
        };
        let uv_mat = if std::path::Path::new(&uv_path).exists() {
            let data = read_vec::<f32>(&uv_path).expect("Failed to read uv");
            assert_eq!(data.len(), shell_count * 6, "UV data length mismatch");
            let mat = (0..shell_count)
                .map(|i| {
                    let start = i * 6;
                    let end = start + 6;
                    Matrix2x3::<f32>::from_vec(data[start..end].to_vec())
                })
                .collect::<Vec<_>>();
            assert_eq!(mat.len(), shell_count, "UV matrix length mismatch");
            Some(mat)
        } else {
            None
        };
        let rod_mat = if n_rod > 0 {
            read_mat_from_file::<usize, 2>(&rod_path).expect("Failed to read rod")
        } else {
            Matrix2xX::<usize>::zeros(0)
        };
        let tri_mat = if n_tri > 0 {
            read_mat_from_file::<usize, 3>(&tri_path).expect("Failed to read tri")
        } else {
            Matrix3xX::<usize>::zeros(0)
        };
        let tet_mat = if n_tet > 0 {
            read_mat_from_file::<usize, 4>(&tet_path).expect("Failed to read tet")
        } else {
            Matrix4xX::<usize>::zeros(0)
        };
        let (static_vert_dmap_mat, static_vert_mat) = if n_static_vert > 0 {
            (
                read_vec::<u32>(&static_vert_dmap_path).expect("Failed to read static_vert_dmap"),
                read_mat_from_file::<f64, 3>(&static_vert_path)
                    .expect("Failed to read static_vert")
                    .map(|x| (x as f64 * ws) as f32),
            )
        } else {
            (Vec::new(), Matrix3xX::<f32>::zeros(0))
        };
        let static_tri_mat = if n_static_tri > 0 {
            read_mat_from_file::<usize, 3>(&static_tri_path).expect("Failed to read static_tri")
        } else {
            Matrix3xX::<usize>::zeros(0)
        };
        let (stitch_ind_mat, stitch_w_mat, stitch_stiffness_vec) = if n_stitch > 0 {
            (
                read_mat_from_file::<usize, 6>(&stitch_ind_path)
                    .expect("Failed to read stitch_ind"),
                read_mat_from_file::<f32, 6>(&stitch_w_path).expect("Failed to read stitch_w"),
                // Per-stitch stiffness (M,). Legacy scenes built before
                // per-object stitch stiffness lack this file; default to 1.0.
                read_vec::<f32>(&stitch_stiffness_path)
                    .unwrap_or_else(|_| vec![1.0f32; n_stitch]),
            )
        } else {
            (
                Matrix6xX::<usize>::zeros(0),
                Matrix6xX::<f32>::zeros(0),
                Vec::new(),
            )
        };

        let mut pin = Vec::new();
        for i in 0..n_pin_block {
            let title = format!("pin-{i}");
            let count = parsed
                .get(&title)
                .unwrap_or_else(|| panic!("Failed to read pin {i}"));
            let n_pin = read_usize(count, "pin");
            let operation_count = read_usize(count, "operation_count");
            let unpin_time = count.get("unpin_time").and_then(|v| v.as_float());
            let pull_w = read_f32(count, "pull");
            // Defaults to 1.0 (no scaling) when absent so older saved
            // states without the key keep their original pin force.
            let stiffness = count
                .get("stiffness")
                .and_then(|v| v.as_float())
                .unwrap_or(1.0) as f32;
            let pin_group_id = count
                .get("pin_group_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let pin_ind_path = format!("{}/bin/pin-ind-{}.bin", args.path, i);

            let pin_ind = read_vec::<usize>(&pin_ind_path).expect("Failed to read pin index");
            assert_eq!(pin_ind.len(), n_pin);

            // Optional per-vertex pull weights (aligned to pin_ind). Present
            // only for partially-pinned pull holders; probed by file
            // existence so older payloads (scalar `pull` only) keep working.
            let pullw_path = format!("{}/bin/pin-pullw-{}.bin", args.path, i);
            let pull_weights = std::path::Path::new(&pullw_path).exists().then(|| {
                let v = read_vec::<f32>(&pullw_path)
                    .expect("Failed to read pin pull weights");
                assert_eq!(v.len(), n_pin, "pin-pullw length must match pin index count");
                v
            });

            // Read operations in order
            let mut operations = Vec::new();
            for j in 0..operation_count {
                let op_title = format!("pin-{i}-op-{j}");
                let op_entry = parsed
                    .get(&op_title)
                    .unwrap_or_else(|| panic!("Failed to read operation {j} for pin {i}"));

                let op_type = read_string(op_entry, "type");

                match op_type.as_str() {
                    "move_by" => {
                        let t_start = read_f64(op_entry, "t_start");
                        let t_end = read_f64(op_entry, "t_end");
                        let transition = read_string(op_entry, "transition");
                        let bezier_handles = read_bezier_handles(op_entry);
                        let delta_path = format!("{}/bin/pin-{}-op-{}.bin", args.path, i, j);
                        let delta = read_mat_from_file::<f64, 3>(&delta_path)
                            .expect("Failed to read move_by delta")
                            .map(|x| (x as f64 * ws) as f32);
                        operations.push(PinOperation::MoveBy {
                            delta,
                            t_start,
                            t_end,
                            transition,
                            bezier_handles,
                        });
                    }
                    "move_to" => {
                        let t_start = read_f64(op_entry, "t_start");
                        let t_end = read_f64(op_entry, "t_end");
                        let transition = read_string(op_entry, "transition");
                        let bezier_handles = read_bezier_handles(op_entry);
                        let target_path = format!("{}/bin/pin-{}-op-{}.bin", args.path, i, j);
                        let target = read_mat_from_file::<f64, 3>(&target_path)
                            .expect("Failed to read move_to target")
                            .map(|x| (x as f64 * ws) as f32);
                        operations.push(PinOperation::MoveTo {
                            target,
                            t_start,
                            t_end,
                            transition,
                            bezier_handles,
                        });
                    }
                    "spin" => {
                        let center_x = read_f32(op_entry, "center_x");
                        let center_y = read_f32(op_entry, "center_y");
                        let center_z = read_f32(op_entry, "center_z");
                        let center = Vector3::new(center_x, center_y, center_z)
                            .map(|x| (x as f64 * ws) as f32);
                        let axis_x = read_f32(op_entry, "axis_x");
                        let axis_y = read_f32(op_entry, "axis_y");
                        let axis_z = read_f32(op_entry, "axis_z");
                        let angular_velocity = read_f32(op_entry, "angular_velocity");
                        let t_start = read_f64(op_entry, "t_start");
                        let t_end = read_f64(op_entry, "t_end");
                        let axis = Vector3::new(axis_x, axis_y, axis_z);
                        operations.push(PinOperation::Spin {
                            center,
                            axis,
                            angular_velocity,
                            t_start,
                            t_end,
                        });
                    }
                    "scale" => {
                        let center_x = read_f32(op_entry, "center_x");
                        let center_y = read_f32(op_entry, "center_y");
                        let center_z = read_f32(op_entry, "center_z");
                        let center = Vector3::new(center_x, center_y, center_z)
                            .map(|x| (x as f64 * ws) as f32);
                        let factor = read_f32(op_entry, "factor");
                        let t_start = read_f64(op_entry, "t_start");
                        let t_end = read_f64(op_entry, "t_end");
                        let transition = read_string(op_entry, "transition");
                        let bezier_handles = read_bezier_handles(op_entry);
                        operations.push(PinOperation::Scale {
                            center,
                            factor,
                            t_start,
                            t_end,
                            transition,
                            bezier_handles,
                        });
                    }
                    "torque" => {
                        let axis_component = read_usize(op_entry, "axis_component") as u32;
                        let magnitude = read_f32(op_entry, "magnitude");
                        let hint_vertex = read_usize(op_entry, "hint_vertex");
                        let t_start = read_f64(op_entry, "t_start");
                        let t_end = read_f64(op_entry, "t_end");
                        operations.push(PinOperation::Torque {
                            axis_component,
                            magnitude,
                            hint_vertex,
                            t_start,
                            t_end,
                        });
                    }
                    "transform_keyframes" => {
                        let keyframe_count = read_usize(op_entry, "keyframe_count");
                        let _ = read_f64(op_entry, "t_start");
                        let _ = read_f64(op_entry, "t_end");
                        let rest_t = [
                            read_f64(op_entry, "rest_tx") * ws,
                            read_f64(op_entry, "rest_ty") * ws,
                            read_f64(op_entry, "rest_tz") * ws,
                        ];
                        let base = format!("{}/bin/pin-{}-op-{}", args.path, i, j);
                        let local = read_mat_from_file::<f64, 3>(&format!("{base}.bin"))
                            .expect("Failed to read transform_keyframes local verts")
                            .map(|x| (x as f64 * ws) as f32);
                        let times = read_vec::<f64>(&format!("{base}-time.bin"))
                            .expect("Failed to read keyframe times");
                        assert_eq!(times.len(), keyframe_count);
                        let t_flat = read_vec::<f64>(&format!("{base}-translation.bin"))
                            .expect("Failed to read keyframe translations");
                        assert_eq!(t_flat.len(), 3 * keyframe_count);
                        let translations: Vec<[f64; 3]> = t_flat
                            .chunks_exact(3)
                            .map(|c| [c[0] * ws, c[1] * ws, c[2] * ws])
                            .collect();
                        let q_flat = read_vec::<f64>(&format!("{base}-quaternion.bin"))
                            .expect("Failed to read keyframe quaternions");
                        assert_eq!(q_flat.len(), 4 * keyframe_count);
                        let quaternions: Vec<[f64; 4]> = q_flat
                            .chunks_exact(4)
                            .map(|c| [c[0], c[1], c[2], c[3]])
                            .collect();
                        let s_flat = read_vec::<f64>(&format!("{base}-scale.bin"))
                            .expect("Failed to read keyframe scales");
                        assert_eq!(s_flat.len(), 3 * keyframe_count);
                        let scales: Vec<[f64; 3]> = s_flat
                            .chunks_exact(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect();
                        let interps = if keyframe_count > 1 {
                            let v = read_vec::<u8>(&format!("{base}-interp.bin"))
                                .expect("Failed to read segment interp codes");
                            assert_eq!(v.len(), keyframe_count - 1);
                            v
                        } else {
                            Vec::new()
                        };
                        let handles = if keyframe_count > 1 {
                            let h_flat = read_vec::<f64>(&format!("{base}-handles.bin"))
                                .expect("Failed to read segment handles");
                            assert_eq!(h_flat.len(), 4 * (keyframe_count - 1));
                            h_flat
                                .chunks_exact(4)
                                .map(|c| [c[0], c[1], c[2], c[3]])
                                .collect()
                        } else {
                            Vec::new()
                        };
                        operations.push(PinOperation::TransformKeyframes {
                            local,
                            times,
                            translations,
                            quaternions,
                            scales,
                            interps,
                            handles,
                            rest_t,
                        });
                    }
                    _ => panic!("Unknown operation type: {op_type}"),
                }
            }

            pin.push(Pin {
                index: pin_ind,
                operations,
                unpin_time,
                pull_w,
                pull_weights,
                stiffness,
                pin_group_id,
            });
        }

        let mut wall = Vec::new();
        for i in 0..n_wall {
            let title = format!("wall-{i}");
            let count = parsed
                .get(&title)
                .unwrap_or_else(|| panic!("Failed to read wall {i}"));
            let n_keyframe = read_usize(count, "keyframe");
            if n_keyframe > 0 {
                let nx = read_f32(count, "nx");
                let ny = read_f32(count, "ny");
                let nz = read_f32(count, "nz");
                let transition = read_string(count, "transition");
                let mut normal = Vector3::new(nx, ny, nz);
                normal.normalize_mut();
                let position =
                    read_mat_from_file::<f64, 3>(&format!("{}/bin/wall-pos-{}.bin", args.path, i))
                        .expect("Failed to read pos_path")
                        .map(|x| (x as f64 * ws) as f32);
                let wall_timing =
                    read_vec::<f64>(&format!("{}/bin/wall-timing-{}.bin", args.path, i))
                        .expect("Failed to read wall timing");
                let contact_gap = read_f32(count, "contact-gap");
                let friction = read_f32(count, "friction");
                let active_duration =
                    read_optional_f32(count, "active-duration").unwrap_or(-1.0);
                let thickness =
                    read_optional_f32(count, "thickness").unwrap_or(1.0) * ws as f32;
                assert_gt!(
                    thickness,
                    0.0,
                    "invisible wall thickness must be > 0 (got {})",
                    thickness
                );
                assert_eq!(position.ncols(), n_keyframe);
                assert_eq!(wall_timing.len(), n_keyframe);
                wall.push(InvisibleWall {
                    normal,
                    position,
                    timing: wall_timing,
                    transition,
                    contact_gap,
                    friction,
                    active_duration,
                    thickness,
                });
            }
        }

        let mut sphere = Vec::new();
        for i in 0..n_sphere {
            let title = format!("sphere-{i}");
            let count = parsed
                .get(&title)
                .unwrap_or_else(|| panic!("Failed to read sphere {i}"));
            let inverted = read_bool(count, "invert");
            let hemisphere = read_bool(count, "hemisphere");
            let transition = read_string(count, "transition");
            let n_keyframe = read_usize(count, "keyframe");
            if n_keyframe > 0 {
                let center = read_mat_from_file::<f64, 3>(&format!(
                    "{}/bin/sphere-pos-{}.bin",
                    args.path, i
                ))
                .expect("Failed to read sphere pos_path")
                .map(|x| (x as f64 * ws) as f32);
                // Sphere collider radius is a world-space length: scale it so the
                // collider keeps its size relative to the scaled mesh.
                let radius: Vec<f32> =
                    read_vec::<f32>(&format!("{}/bin/sphere-radius-{}.bin", args.path, i))
                        .expect("Failed to read sphere radius")
                        .iter()
                        .map(|r| (*r as f64 * ws) as f32)
                        .collect();
                let timing = read_vec::<f64>(&format!("{}/bin/sphere-timing-{}.bin", args.path, i))
                    .expect("Failed to read sphere timing");
                let contact_gap = read_f32(count, "contact-gap");
                let friction = read_f32(count, "friction");
                let active_duration =
                    read_optional_f32(count, "active-duration").unwrap_or(-1.0);
                let thickness =
                    read_optional_f32(count, "thickness").unwrap_or(1.0) * ws as f32;
                assert_gt!(
                    thickness,
                    0.0,
                    "invisible sphere thickness must be > 0 (got {})",
                    thickness
                );
                assert_eq!(center.ncols(), n_keyframe);
                assert_eq!(radius.len(), n_keyframe);
                assert_eq!(timing.len(), n_keyframe);
                sphere.push(InvisibleSphere {
                    center,
                    radius,
                    timing,
                    inverted,
                    hemisphere,
                    transition,
                    contact_gap,
                    friction,
                    active_duration,
                    thickness,
                });
            }
        }

        assert_eq!(vert_mat.ncols(), n_vert);
        assert_eq!(vel_mat.ncols(), n_vert);
        assert_eq!(rod_mat.ncols(), n_rod);
        assert_eq!(tri_mat.ncols(), n_tri);
        assert_eq!(tet_mat.ncols(), n_tet);
        assert_eq!(static_vert_mat.ncols(), n_static_vert);
        assert_eq!(
            static_vert_dmap_mat.len(),
            n_static_vert,
            "static_vert_dmap size mismatch"
        );
        assert_eq!(static_tri_mat.ncols(), n_static_tri);
        assert_eq!(stitch_ind_mat.ncols(), n_stitch);
        assert_eq!(stitch_w_mat.ncols(), n_stitch);
        assert_eq!(stitch_stiffness_vec.len(), n_stitch);

        let args_path = format!("{}/param.toml", args.path);
        let file_content = fs::read_to_string(args_path).unwrap();
        let config: Config = toml::from_str(&file_content).unwrap();

        let dyn_args_path = format!("{}/dyn_param.txt", args.path);
        let dyn_args = if std::path::Path::new(&dyn_args_path).exists() {
            read_dyn_param(&dyn_args_path).unwrap()
        } else {
            Vec::new()
        };

        let param_dir = format!("{}/bin/param", args.path);
        let mut rod_param = Vec::new();
        let mut tri_param = Vec::new();
        let mut tet_param = Vec::new();
        let mut static_param = Vec::new();
        let mut sand_param = Vec::new();

        for entry in fs::read_dir(&param_dir).expect("Failed to read param directory") {
            let entry = entry.expect("Failed to read entry");
            let path = entry.path();
            if path.is_file() {
                let file_name = path.file_name().unwrap().to_str().unwrap();

                // SAND params are scalar (len 1), not per-element, so they are
                // read on a separate channel that skips the `len == n_element`
                // assert below. Every grain in the cloud shares one mass /
                // radius / gap / friction.
                if file_name.starts_with("sand-") && file_name.ends_with(".bin") {
                    let name = file_name["sand-".len()..file_name.len() - 4].to_string();
                    let values = read_vec::<f32>(&path.to_string_lossy())
                        .expect("Failed to read sand values");
                    sand_param.push((name, ParamValueList::Value(values)));
                    continue;
                }

                let (target_param, prefix, n_element) = if file_name.starts_with("rod-") {
                    (&mut rod_param, "rod-", n_rod)
                } else if file_name.starts_with("tri-") {
                    (&mut tri_param, "tri-", n_tri)
                } else if file_name.starts_with("tet-") {
                    (&mut tet_param, "tet-", n_tet)
                } else if file_name.starts_with("static-") {
                    (&mut static_param, "static-", n_static_tri)
                } else {
                    continue;
                };

                if file_name.ends_with(".bin") {
                    let name = file_name[prefix.len()..file_name.len() - 4].to_string();
                    if name == "model" {
                        let values = read_vec::<u8>(&path.to_string_lossy())
                            .expect("Failed to read model values");
                        let values = values
                            .iter()
                            .map(|&k| {
                                Model::from_id(k)
                                    .unwrap_or_else(|| panic!("Unknown model type: {k}"))
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(values.len(), n_element, "path: {}", path.display());
                        target_param.push((name, ParamValueList::Model(values)));
                    } else {
                        let values = read_vec::<f32>(&path.to_string_lossy())
                            .expect("Failed to read values");
                        assert_eq!(values.len(), n_element, "path: {}", path.display());
                        target_param.push((name, ParamValueList::Value(values)));
                    }
                }
            }
        }

        // Optional PDRD inputs. Absent files are equivalent to "no
        // PDRD bodies in this scene" and the solver falls through.
        let pdrd_body_path = format!("{}/bin/pdrd_body.bin", args.path);
        let pdrd_body_rows = if std::path::Path::new(&pdrd_body_path).exists() {
            let mut raw =
                read_vec::<f32>(&pdrd_body_path).expect("Failed to read pdrd_body.bin");
            // Row layout: PDRD_BODY_ROW_LEN floats per body (vertex_start,
            // vertex_count, volume, centroid[3], rest_gram_inv[9],
            // mass_per_vertex, joint_mode, joint_axis[3], joint_pin[3]).
            assert!(
                raw.len() % PDRD_BODY_ROW_LEN == 0,
                "pdrd_body.bin length {} not a multiple of {}",
                raw.len(),
                PDRD_BODY_ROW_LEN,
            );
            // Uniform world scaling. pdrd_body.bin is precomputed in the addon
            // from the UNSCALED mesh, but the live vertices it is matched
            // against were multiplied by `ws` on ingest. Each geometric field
            // carries a power of length and must be rescaled by that power of
            // `ws` so the shape-match dynamics stay consistent:
            //   volume          (length^3) -> * ws^3
            //   rest_centroid   (length^1) -> * ws       world position
            //   rest_gram_inv   (length^-2)-> * 1/ws^2   see derivation below
            //   mass_per_vertex (prop. volume) -> * ws^3
            //   joint_pin       (length^1) -> * ws       world pivot
            // vertex_start/count (indices/counts), joint_mode (enum) and
            // joint_axis (a UNIT vector) are dimensionless and left untouched.
            //
            // rest_gram_inv derivation: it is stored as the inverse of the rest
            // Gram `Sbar = sum_k ybar_k ybar_k^T` (length^2), where ybar_k are
            // the centered rest positions (pdrd_rest_centered, scaled by ws
            // below). The mass is factored out separately into mass_per_vertex,
            // so Sbar is a pure geometric second moment (no mass weighting).
            // Scaling ybar by ws sends Sbar -> ws^2 Sbar, hence the stored
            // inverse -> ws^-2. This keeps the recovered reference inertia
            // Iref = m (tr(Sbar) I - Sbar) ~ ws^3 * ws^2 = ws^5 = mass*length^2,
            // the physically correct moment-of-inertia scaling. The best-fit
            // rotation R is the polar factor of the cross-covariance
            // M = sum_k y_k ybar_k^T and does NOT use rest_gram_inv; with both
            // y_k and ybar_k scaled by ws, M -> ws^2 M leaves its polar factor
            // (the rotation) invariant, so R is scale-invariant as required.
            //
            // ws == 1.0 (the default) is a no-op; skip the round-trip so the
            // unscaled path stays byte-for-byte identical.
            if (ws - 1.0).abs() > 1e-12 {
                let ws3 = ws * ws * ws;
                let inv_ws2 = 1.0 / (ws * ws);
                let n_bodies = raw.len() / PDRD_BODY_ROW_LEN;
                for b in 0..n_bodies {
                    let row =
                        &mut raw[PDRD_BODY_ROW_LEN * b..PDRD_BODY_ROW_LEN * (b + 1)];
                    row[2] = (row[2] as f64 * ws3) as f32; // volume (length^3)
                    for c in &mut row[3..6] {
                        *c = (*c as f64 * ws) as f32; // rest_centroid (length)
                    }
                    for g in &mut row[6..15] {
                        *g = (*g as f64 * inv_ws2) as f32; // rest_gram_inv (length^-2)
                    }
                    row[15] = (row[15] as f64 * ws3) as f32; // mass_per_vertex (~ volume)
                    for p in &mut row[20..23] {
                        *p = (*p as f64 * ws) as f32; // joint_pin (length)
                    }
                }
            }
            raw
        } else {
            Vec::new()
        };
        let pdrd_vert_index_path = format!("{}/bin/pdrd_vert_index.bin", args.path);
        let pdrd_vert_index = if std::path::Path::new(&pdrd_vert_index_path).exists() {
            let v = read_vec::<u32>(&pdrd_vert_index_path)
                .expect("Failed to read pdrd_vert_index.bin");
            assert_eq!(v.len(), n_vert, "pdrd_vert_index size mismatch");
            v
        } else {
            vec![0u32; n_vert]
        };
        let pdrd_vert_list_path = format!("{}/bin/pdrd_vert_list.bin", args.path);
        let pdrd_vert_list = if std::path::Path::new(&pdrd_vert_list_path).exists() {
            read_vec::<u32>(&pdrd_vert_list_path).expect("Failed to read pdrd_vert_list.bin")
        } else {
            Vec::new()
        };
        let pdrd_rest_centered_path = format!("{}/bin/pdrd_rest_centered.bin", args.path);
        let pdrd_rest_centered = if std::path::Path::new(&pdrd_rest_centered_path).exists() {
            let mut v = read_vec::<f32>(&pdrd_rest_centered_path)
                .expect("Failed to read pdrd_rest_centered.bin");
            assert_eq!(
                v.len(),
                3 * pdrd_vert_list.len(),
                "pdrd_rest_centered length mismatch (expected 3 * vert_list = {}, got {})",
                3 * pdrd_vert_list.len(),
                v.len(),
            );
            // Centered rest positions ybar_k are a world length (length^1), so
            // scale by ws to match the live (ws-scaled) vertices they are fit
            // against. The reconstruct x_v = centroid + R * ybar_k would
            // otherwise rebuild the body at its unscaled size, corrupting the
            // shape match. ws == 1.0 is a no-op (skip the round-trip).
            if (ws - 1.0).abs() > 1e-12 {
                for y in &mut v {
                    *y = (*y as f64 * ws) as f32;
                }
            }
            v
        } else {
            Vec::new()
        };

        Self {
            args: config.param,
            dyn_args,
            displacement: displacement_mat,
            vert_dmap: vert_dmap_mat,
            vert: vert_mat,
            vel: vel_mat,
            uv: uv_mat,
            rod: rod_mat,
            tri: tri_mat,
            tet: tet_mat,
            static_vert_dmap: static_vert_dmap_mat,
            static_vert: static_vert_mat,
            static_tri: static_tri_mat,
            stitch_ind: stitch_ind_mat,
            stitch_w: stitch_w_mat,
            stitch_stiffness: stitch_stiffness_vec,
            pin,
            wall,
            sphere,
            rest_vert: rest_vert_mat,
            rest_vert_mask,
            rest_vert_schedule,
            bend_rest_vert,
            bend_rest_vert_mask,
            shell_count,
            rod_param,
            tri_param,
            tet_param,
            static_param,
            sand_param,
            pdrd_body_rows,
            pdrd_vert_index,
            pdrd_vert_list,
            pdrd_rest_centered,
        }
    }

    pub fn export_param_summary(
        &self,
        args: &ProgramArgs,
        props: &Props,
        face_area: &[f32],
        tet_volume: &[f32],
    ) {
        // Write the summary next to param.toml in the session directory
        let summary_path = format!("{}/param_summary.txt", args.path);
        let mut file = match File::create(&summary_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create parameter summary file: {e}");
                return;
            }
        };

        let mut content = String::new();

        // Write Shell/Triangle parameters summary
        if self.tri.ncols() > 0 {
            content.push_str(&format!(
                "=== Shells ({} elements) ===\n\n",
                self.tri.ncols()
            ));
            self.write_param_stats(&mut content, &self.tri_param);

            // Add mass and area statistics for shells
            if !props.face.is_empty() && !face_area.is_empty() {
                let masses: Vec<f32> = props.face.iter().map(|p| p.mass).collect();
                if !masses.is_empty() {
                    let min_mass = masses.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_mass = masses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum_mass: f64 = masses.iter().map(|&v| v as f64).sum();
                    let mean_mass = (sum_mass / masses.len() as f64) as f32;

                    content.push_str(&format!(
                        "mass: (max: {max_mass:.4e}, min: {min_mass:.4e}, mean: {mean_mass:.4e})\n"
                    ));
                }

                let areas: Vec<f32> = face_area.to_vec();
                if !areas.is_empty() {
                    let min_area = areas.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_area = areas.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum_area: f64 = areas.iter().map(|&v| v as f64).sum();
                    let mean_area = (sum_area / areas.len() as f64) as f32;

                    content.push_str(&format!(
                        "area: (max: {max_area:.4e}, min: {min_area:.4e}, mean: {mean_area:.4e})\n"
                    ));
                }
            }
            content.push('\n');
        }

        // Write Solid/Tetrahedral parameters summary
        if self.tet.ncols() > 0 {
            content.push_str(&format!(
                "=== Solids ({} elements) ===\n\n",
                self.tet.ncols()
            ));
            self.write_param_stats(&mut content, &self.tet_param);

            // Add mass and volume statistics for solids
            if !props.tet.is_empty() {
                let masses: Vec<f32> = props.tet.iter().map(|p| p.mass).collect();
                let min_mass = masses.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_mass = masses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_mass: f64 = masses.iter().map(|&v| v as f64).sum();
                let mean_mass = (sum_mass / masses.len() as f64) as f32;

                let min_volume = tet_volume.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_volume = tet_volume.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_volume: f64 = tet_volume.iter().map(|&v| v as f64).sum();
                let mean_volume = (sum_volume / tet_volume.len() as f64) as f32;

                content.push_str(&format!(
                    "mass: (max: {max_mass:.4e}, min: {min_mass:.4e}, mean: {mean_mass:.4e})\n"
                ));
                content.push_str(&format!(
                    "volume: (max: {max_volume:.4e}, min: {min_volume:.4e}, mean: {mean_volume:.4e})\n"
                ));
            }
            content.push('\n');
        }

        // Write Rod parameters summary
        if self.rod.ncols() > 0 {
            content.push_str(&format!("=== Rods ({} elements) ===\n\n", self.rod.ncols()));
            self.write_param_stats(&mut content, &self.rod_param);

            // Add mass statistics for rods. At summary time props.edge holds
            // only rod entries (non-rod edges are appended later), so its
            // length matches the rod count exactly.
            debug_assert_eq!(
                props.edge.len(),
                self.rod.ncols(),
                "param summary runs before non-rod edges are appended to props.edge"
            );
            let rod_count = self.rod.ncols();
            if rod_count > 0 {
                let rod_props = &props.edge[0..rod_count];
                let masses: Vec<f32> = rod_props.iter().map(|p| p.mass).collect();
                let min_mass = masses.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_mass = masses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_mass: f64 = masses.iter().map(|&v| v as f64).sum();
                let mean_mass = (sum_mass / masses.len() as f64) as f32;

                content.push_str(&format!(
                    "mass: (max: {max_mass:.4e}, min: {min_mass:.4e}, mean: {mean_mass:.4e})\n"
                ));
            }
            content.push('\n');
        }

        // Write Static object parameters summary
        if self.static_tri.ncols() > 0 {
            content.push_str(&format!(
                "=== Static Objects ({} elements) ===\n\n",
                self.static_tri.ncols()
            ));
            self.write_param_stats(&mut content, &self.static_param);
            content.push('\n');
        }

        if let Err(e) = file.write_all(content.as_bytes()) {
            eprintln!("Failed to write parameter summary: {e}");
        } else {
            println!("Parameter summary written to: {summary_path}");
        }
    }

    fn write_param_stats(
        &self,
        content: &mut String,
        params: &[(String, ParamValueList)],
    ) {
        for (name, values) in params {
            match values {
                ParamValueList::Model(models) => {
                    // Count occurrences of each model type
                    let mut model_counts = HashMap::new();
                    for model in models {
                        *model_counts.entry(format!("{model:?}")).or_insert(0) += 1;
                    }
                    // Sort by name so the breakdown is stable across runs
                    // (std HashMap iteration order is randomized).
                    let mut model_counts: Vec<(String, i32)> =
                        model_counts.into_iter().collect();
                    model_counts.sort_by(|a, b| a.0.cmp(&b.0));

                    content.push_str(&format!("{name}: ("));
                    let mut first = true;
                    for (model_name, count) in model_counts {
                        if !first {
                            content.push_str(", ");
                        }
                        content.push_str(&format!(
                            "{}: {} elements",
                            model_name.to_lowercase(),
                            count
                        ));
                        first = false;
                    }
                    content.push_str(")\n");
                }
                ParamValueList::Value(vals) => {
                    if vals.is_empty() {
                        continue;
                    }

                    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    // Use double precision for accurate mean calculation
                    let sum: f64 = vals.iter().map(|&v| v as f64).sum();
                    let mean = (sum / vals.len() as f64) as f32;

                    content.push_str(&format!(
                        "{name}: (max: {max:.4e}, min: {min:.4e}, mean: {mean:.4e})\n"
                    ));
                }
            }
        }
    }

    pub fn args(&self) -> SimArgs {
        self.args.clone()
    }

    /// Resolve the scalar SAND material params (`sand-*.bin`) into a
    /// `SandParams`, or `None` when the scene has no SAND object. The grain
    /// radius / contact gap are sent on the standard `contact-offset` /
    /// `contact-gap` keys (a grain's physical contact skin), while mass and
    /// friction come from the `sand-*` keys. Every grain in the cloud shares
    /// these scalars.
    fn sand_params(&self) -> Option<SandParams> {
        if self.sand_param.is_empty() {
            return None;
        }
        let get = |key: &str| -> Option<f32> {
            self.sand_param
                .iter()
                .find(|(name, _)| name.as_str() == key)
                .map(|(_, value)| as_value(value, 0, key))
        };
        let particle_mass = get("particle-mass")
            .expect("SAND object missing 'sand-particle-mass' param");
        let grain_radius = get("grain-radius")
            .expect("SAND object missing 'sand-grain-radius' param");
        let contact_gap = get("contact-gap")
            .expect("SAND object missing 'sand-contact-gap' param");
        let friction = get("friction")
            .expect("SAND object missing 'sand-friction' param");
        assert_gt!(particle_mass, 0.0, "Sand particle mass must be positive");
        assert_ge!(grain_radius, 0.0, "Sand grain radius must be non-negative");
        assert_gt!(contact_gap, 0.0, "Sand contact gap must be positive");
        assert_ge!(friction, 0.0, "Sand friction must be non-negative");
        Some(SandParams {
            particle_mass,
            grain_radius,
            contact_gap,
            friction,
        })
    }

    pub fn make_props(&self, mesh: &MeshSet, face_area: &[f32], tet_volume: &[f32]) -> Props {
        // Build edge props and params with deduplication
        let mut edge_param_map: HashMap<EdgeParam, u32> = HashMap::new();
        let mut edge_params = Vec::new();
        let edge = (0..self.rod.ncols())
            .map(|i| {
                let mut ghat = None;
                let mut offset = None;
                let mut friction = None;
                let mut stiffness = None;
                let mut bend = None;
                let mut length_factor = None;
                let mut strainlimit = None;
                let mut bend_plasticity = None;
                let mut bend_plasticity_threshold = None;
                let mut bend_rest_from_geometry = None;
                let mut deform_damping = None;
                let mut bend_damping = None;
                let rod = mesh.mesh.mesh.edge.column(i);
                let x0 = mesh.vertex.column(rod[0]);
                let x1 = mesh.vertex.column(rod[1]);
                let mut length = (x1 - x0).map(f32::from).norm();
                let initial_length = length;
                let mut density = None;
                for (name, value) in &self.rod_param {
                    if name == "contact-gap" {
                        ghat = Some(as_value(value, i, name));
                    } else if name == "contact-offset" {
                        offset = Some(as_value(value, i, name));
                    } else if name == "friction" {
                        friction = Some(as_value(value, i, name));
                    } else if name == "young-mod" {
                        stiffness = Some(as_value(value, i, name));
                    } else if name == "density" {
                        density = Some(as_value(value, i, name));
                    } else if name == "bend" {
                        bend = Some(as_value(value, i, name));
                    } else if name == "length-factor" {
                        length_factor = Some(as_value(value, i, name));
                    } else if name == "strain-limit" {
                        strainlimit = Some(as_value(value, i, name));
                    } else if name == "bend-plasticity" {
                        bend_plasticity = Some(as_value(value, i, name));
                    } else if name == "bend-plasticity-threshold" {
                        bend_plasticity_threshold = Some(as_value(value, i, name));
                    } else if name == "bend-rest-from-geometry" {
                        bend_rest_from_geometry = Some(as_value(value, i, name) != 0.0);
                    } else if name == "deformation-damping" {
                        deform_damping = Some(as_value(value, i, name));
                    } else if name == "bending-damping" {
                        bend_damping = Some(as_value(value, i, name));
                    } else if name == "pressure" {
                        // Pressure is a tri-only parameter; ignore for rods.
                    } else {
                        panic!("Unknown rod parameter: {name}");
                    }
                }
                let ghat = ghat.unwrap();
                let offset = offset.unwrap();
                let friction = friction.unwrap();
                let stiffness = stiffness.unwrap();
                let bend = bend.unwrap();
                let length_factor = length_factor.unwrap();
                let density = density.unwrap();
                let strainlimit = strainlimit.unwrap_or(0.0);
                let bend_plasticity = bend_plasticity.unwrap_or(0.0);
                let bend_plasticity_threshold = bend_plasticity_threshold.unwrap_or(0.0);
                let bend_rest_from_geometry = bend_rest_from_geometry.unwrap_or(false);
                let deform_damping = deform_damping.unwrap_or(0.0);
                let bend_damping = bend_damping.unwrap_or(0.0);
                assert_ge!(deform_damping, 0.0, "Deformation damping must be non-negative");
                assert_ge!(bend_damping, 0.0, "Bending damping must be non-negative");
                assert_gt!(density, 0.0, "Density must be positive");
                assert_gt!(stiffness, 0.0, "Stiffness must be positive");
                assert_gt!(length, 0.0, "Length must be positive");
                assert_ge!(friction, 0.0, "Friction must be non-negative");
                assert_ge!(bend, 0.0, "Bend modulus must be non-negative");
                assert_ge!(strainlimit, 0.0, "Strain limit must be non-negative");
                assert_ge!(bend_plasticity, 0.0, "Bend plasticity must be non-negative");
                assert_ge!(
                    bend_plasticity_threshold,
                    0.0,
                    "Bend plasticity threshold must be non-negative"
                );
                assert_gt!(ghat, 0.0, "Contact gap must be positive");
                let mass = density * length;
                length *= length_factor;

                // Create param and deduplicate
                let param = EdgeParam {
                    stiffness,
                    bend,
                    ghat,
                    offset,
                    friction,
                    strainlimit,
                    plasticity: bend_plasticity,
                    plasticity_threshold: bend_plasticity_threshold,
                    bend_rest_from_geometry,
                    deform_damping,
                    bend_damping,
                };
                let param_idx = dedup_param(&mut edge_param_map, &mut edge_params, param);

                EdgeProp {
                    length,
                    initial_length,
                    mass,
                    fixed: false,
                    param_index: param_idx,
                }
            })
            .collect::<Vec<_>>();

        // Build face props and params with deduplication
        let mut face_param_map: HashMap<FaceParam, u32> = HashMap::new();
        let mut face_params = Vec::new();
        let face = (0..mesh.mesh.mesh.face.ncols())
            .map(|i| {
                let area = face_area[i];
                let mut model = None;
                let mut young_mod = None;
                let mut poiss_rat = None;
                let mut bend = None;
                let mut shrink_x = None;
                let mut shrink_y = None;
                let mut strainlimit = None;
                let mut ghat = None;
                let mut offset = None;
                let mut friction = None;
                let mut density = None;
                let mut pressure = None;
                let mut plasticity = None;
                let mut plasticity_threshold = None;
                let mut bend_plasticity = None;
                let mut bend_plasticity_threshold = None;
                let mut bend_rest_from_geometry = None;
                let mut deform_damping = None;
                let mut bend_damping = None;
                for (name, value) in &self.tri_param {
                    if name == "contact-gap" {
                        ghat = Some(as_value(value, i, name));
                    } else if name == "contact-offset" {
                        offset = Some(as_value(value, i, name));
                    } else if name == "friction" {
                        friction = Some(as_value(value, i, name));
                    } else if name == "strain-limit" {
                        strainlimit = Some(as_value(value, i, name));
                    } else if name == "bend" {
                        bend = Some(as_value(value, i, name));
                    } else if name == "shrink-x" {
                        shrink_x = Some(as_value(value, i, name));
                    } else if name == "shrink-y" {
                        shrink_y = Some(as_value(value, i, name));
                    } else if name == "model" {
                        model = Some(as_model(value, i, name));
                    } else if name == "density" {
                        density = Some(as_value(value, i, name));
                    } else if name == "young-mod" {
                        young_mod = Some(as_value(value, i, name));
                    } else if name == "poiss-rat" {
                        poiss_rat = Some(as_value(value, i, name));
                    } else if name == "pressure" {
                        pressure = Some(as_value(value, i, name));
                    } else if name == "plasticity" {
                        plasticity = Some(as_value(value, i, name));
                    } else if name == "plasticity-threshold" {
                        plasticity_threshold = Some(as_value(value, i, name));
                    } else if name == "bend-plasticity" {
                        bend_plasticity = Some(as_value(value, i, name));
                    } else if name == "bend-plasticity-threshold" {
                        bend_plasticity_threshold = Some(as_value(value, i, name));
                    } else if name == "bend-rest-from-geometry" {
                        bend_rest_from_geometry = Some(as_value(value, i, name) != 0.0);
                    } else if name == "deformation-damping" {
                        deform_damping = Some(as_value(value, i, name));
                    } else if name == "bending-damping" {
                        bend_damping = Some(as_value(value, i, name));
                    } else {
                        panic!("Unknown face parameter: {name}");
                    }
                }
                let model = model.unwrap();
                let young_mod = young_mod.unwrap();
                let poiss_rat = poiss_rat.unwrap();
                let bend = bend.unwrap();
                let shrink_x = shrink_x.unwrap();
                let shrink_y = shrink_y.unwrap();
                let strainlimit = strainlimit.unwrap();
                let ghat = ghat.unwrap();
                let offset = offset.unwrap();
                let friction = friction.unwrap();
                let density = density.unwrap();
                let pressure = pressure.unwrap_or(0.0);
                let plasticity = plasticity.unwrap_or(0.0);
                let plasticity_threshold = plasticity_threshold.unwrap_or(0.0);
                let bend_plasticity = bend_plasticity.unwrap_or(0.0);
                let bend_plasticity_threshold = bend_plasticity_threshold.unwrap_or(0.0);
                let bend_rest_from_geometry = bend_rest_from_geometry.unwrap_or(false);
                let deform_damping = deform_damping.unwrap_or(0.0);
                let bend_damping = bend_damping.unwrap_or(0.0);
                assert_ge!(deform_damping, 0.0, "Deformation damping must be non-negative");
                assert_ge!(bend_damping, 0.0, "Bending damping must be non-negative");
                assert_gt!(density, 0.0, "Density must be positive");
                assert_gt!(area, 0.0, "Area must be positive");
                assert_ge!(pressure, 0.0, "Pressure must be non-negative");
                assert_ge!(friction, 0.0, "Friction must be non-negative");
                assert_gt!(ghat, 0.0, "Contact gap must be positive");
                // PDRD faces carry zero placeholders for elastic terms
                // (young_mod, poiss_rat, bend, shrink) so the per-face
                // expansion remains key-compatible with shells; their
                // values are ignored by the elastic dispatch (gated on
                // Model::Pdrd). Only validate the elastic terms on
                // non-PDRD faces.
                if model != Model::Pdrd {
                    assert_gt!(young_mod, 0.0, "Young's modulus must be positive");
                    assert_ge!(bend, 0.0, "Bend modulus must be non-negative");
                    assert_gt!(shrink_x, 0.0, "Shrink X factor must be positive");
                    assert_gt!(shrink_y, 0.0, "Shrink Y factor must be positive");
                    // Shrink/extend and strain-limit cannot be combined on the
                    // same face: each rewrites the rest shape independently, so
                    // the strain bound becomes ill-defined when both are active.
                    // Share the predicate with ppf-cts-core so frontend preview,
                    // PyO3, and the solver all gate on the same rule.
                    assert!(
                        !ppf_cts_core::kernels::scene_build::is_shell_shrink_strain_limit_conflict(
                            shrink_x as f64,
                            shrink_y as f64,
                            strainlimit as f64
                        ),
                        "Face {i}: shrink (x={shrink_x}, y={shrink_y}) conflicts with non-zero strain-limit ({strainlimit})"
                    );
                    assert_gt!(poiss_rat, 0.0, "Poisson's ratio must be positive");
                    assert_lt!(poiss_rat, 0.5, "Poisson's ratio must be less than 0.5");
                }
                let (mu, lambda) = if model == Model::Pdrd {
                    // PDRD faces contribute no elastic energy; the
                    // existing dispatch in energy.cu trips on
                    // `mu > 0.0f` so zero here is the natural
                    // skip-everything path.
                    (0.0f32, 0.0f32)
                } else {
                    convert_prop(young_mod, poiss_rat)
                };
                // PDRD bodies use VOLUMETRIC mass (ρ·V) distributed
                // over per-body vertices in the PDRD builder pass.
                // Zero out the per-face contribution here so the
                // face-mass aggregation in builder.rs doesn't
                // double-count.
                let mass = if model == Model::Pdrd {
                    0.0f32
                } else {
                    density * area
                };

                // Create param and deduplicate
                let param = FaceParam {
                    model,
                    mu,
                    lambda,
                    friction,
                    ghat,
                    offset,
                    bend,
                    strainlimit,
                    shrink_x,
                    shrink_y,
                    pressure,
                    plasticity,
                    plasticity_threshold,
                    bend_plasticity,
                    bend_plasticity_threshold,
                    bend_rest_from_geometry,
                    deform_damping,
                    bend_damping,
                };
                let param_idx = dedup_param(&mut face_param_map, &mut face_params, param);

                FaceProp {
                    area,
                    mass,
                    fixed: false,
                    rest_excluded: false,
                    param_index: param_idx,
                }
            })
            .collect::<Vec<_>>();

        // Build tet props and params with deduplication
        let mut tet_param_map: HashMap<TetParam, u32> = HashMap::new();
        let mut tet_params = Vec::new();
        let tet = (0..self.tet.ncols())
            .map(|i| {
                let mut model = None;
                let mut density = None;
                let mut young_mod = None;
                let mut poiss_rat = None;
                let mut shrink = None;
                let mut plasticity = None;
                let mut plasticity_threshold = None;
                let mut deform_damping = None;
                for (name, value) in &self.tet_param {
                    if name == "model" {
                        model = Some(as_model(value, i, name));
                    } else if name == "density" {
                        density = Some(as_value(value, i, name));
                    } else if name == "young-mod" {
                        young_mod = Some(as_value(value, i, name));
                    } else if name == "poiss-rat" {
                        poiss_rat = Some(as_value(value, i, name));
                    } else if name == "shrink" {
                        shrink = Some(as_value(value, i, name));
                    } else if name == "pressure" {
                        // Pressure is a tri-only parameter; ignore for tets.
                    } else if name == "plasticity" {
                        plasticity = Some(as_value(value, i, name));
                    } else if name == "plasticity-threshold" {
                        plasticity_threshold = Some(as_value(value, i, name));
                    } else if name == "deformation-damping" {
                        deform_damping = Some(as_value(value, i, name));
                    } else if name == "bending-damping" {
                        // Bending damping is a shell/rod-only key; tets have no
                        // bending energy. The frontend clears it from the tet
                        // param set, so this is a defensive ignore (mirrors the
                        // tri-only "pressure" branch above).
                    } else {
                        panic!("Unknown tet parameter: {name}");
                    }
                }
                let model = model.unwrap();
                let density = density.unwrap();
                let young_mod = young_mod.unwrap();
                let poiss_rat = poiss_rat.unwrap();
                let shrink = shrink.unwrap();
                let plasticity = plasticity.unwrap_or(0.0);
                let plasticity_threshold = plasticity_threshold.unwrap_or(0.0);
                let deform_damping = deform_damping.unwrap_or(0.0);
                assert_ge!(deform_damping, 0.0, "Deformation damping must be non-negative");
                let volume = tet_volume[i];
                assert_gt!(density, 0.0, "Density must be positive");
                assert_gt!(young_mod, 0.0, "Young's modulus must be positive");
                assert_gt!(poiss_rat, 0.0, "Poisson's ratio must be positive");
                assert_lt!(poiss_rat, 0.5, "Poisson's ratio must be less than 0.5");
                assert_gt!(shrink, 0.0, "Shrink factor must be positive");
                assert_gt!(volume, 0.0, "Volume must be positive");
                let (mu, lambda) = convert_prop(young_mod, poiss_rat);
                let mass = density * volume;

                // Create param and deduplicate
                let param = TetParam {
                    model,
                    mu,
                    lambda,
                    shrink,
                    plasticity,
                    plasticity_threshold,
                    deform_damping,
                };
                let param_idx = dedup_param(&mut tet_param_map, &mut tet_params, param);

                TetProp {
                    mass,
                    volume,
                    fixed: false,
                    rest_excluded: false,
                    param_index: param_idx,
                }
            })
            .collect::<Vec<_>>();

        Props {
            edge,
            face,
            tet,
            edge_params,
            face_params,
            tet_params,
            sand: self.sand_params(),
        }
    }

    pub fn get_initial_velocity(&self) -> Matrix3xX<f32> {
        self.vel.clone()
    }

    pub fn make_constraint(&self, time: f64) -> Constraint {
        let collision_mesh = if self.static_vert.ncols() > 0 {
            let mut vert = self.static_vert.clone();
            apply_displacement(&mut vert, &self.displacement, &self.static_vert_dmap);

            // Build face props and params with deduplication
            let mut face_param_map: HashMap<FaceParam, u32> = HashMap::new();
            let mut face_params = Vec::new();
            let face_props = (0..self.static_tri.ncols())
                .map(|i| {
                    let mut contact_gap = None;
                    let mut contact_offset = None;
                    let mut friction = None;
                    for (name, value) in self.static_param.iter() {
                        if name == "contact-gap" {
                            contact_gap = Some(as_value(value, i, name));
                        } else if name == "contact-offset" {
                            contact_offset = Some(as_value(value, i, name));
                        } else if name == "friction" {
                            friction = Some(as_value(value, i, name));
                        }
                    }
                    let area = super::triutils::area(&vert, &self.static_tri, i);
                    let ghat = contact_gap.unwrap();
                    let offset = contact_offset.unwrap();
                    let friction = friction.unwrap();
                    assert_gt!(area, 0.0, "Area of static triangle {} is zero", i);
                    assert_gt!(ghat, 0.0, "Contact gap must be positive");

                    // Create param and deduplicate
                    let param = FaceParam {
                        model: Model::default(),
                        mu: 0.0,
                        lambda: 0.0,
                        friction,
                        ghat,
                        offset,
                        bend: 0.0,
                        strainlimit: 0.0,
                        shrink_x: 1.0,
                        shrink_y: 1.0,
                        pressure: 0.0,
                        plasticity: 0.0,
                        plasticity_threshold: 0.0,
                        bend_plasticity: 0.0,
                        bend_plasticity_threshold: 0.0,
                        bend_rest_from_geometry: false,
                        deform_damping: 0.0,
                        bend_damping: 0.0,
                    };
                    let param_idx = dedup_param(&mut face_param_map, &mut face_params, param);

                    FaceProp {
                        area,
                        mass: 0.0,
                        fixed: false,
                        rest_excluded: false,
                        param_index: param_idx,
                    }
                })
                .collect::<Vec<_>>();
            make_collision_mesh(&vert, &self.static_tri, &face_props, &face_params)
        } else {
            CollisionMesh::new()
        };
        let calc_coefficient =
            |time: f64, timings: &[f64], transition: &str| -> ([usize; 2], f32) {
                if timings.is_empty() {
                    ([0, 0], 1.0)
                } else {
                    let last_time = timings[timings.len() - 1];
                    if time >= last_time {
                        ([timings.len() - 1, timings.len() - 1], 1.0)
                    } else {
                        for i in 0..timings.len() - 1 {
                            let t0 = timings[i];
                            let t1 = timings[i + 1];
                            if time >= t0 && time < t1 {
                                let mut w = (time - t0) / (t1 - t0);
                                if transition == "smooth" {
                                    w = w * w * (3.0 - 2.0 * w);
                                }
                                return ([i, i + 1], w as f32);
                            }
                        }
                        panic!("Failed to calculate coefficient")
                    }
                }
            };
        // Shared blend decision for the wall and sphere keyframe loops: a
        // single keyframe holds at column 0 (non-kinematic), otherwise the
        // pair (j, k) and weight w come from calc_coefficient (kinematic).
        let blend_coeff = |timing: &[f64], transition: &str| -> (usize, usize, f32, bool) {
            if timing.len() <= 1 {
                assert_eq!(timing[0], 0.0);
                (0, 0, 0.0, false)
            } else {
                let coeff = calc_coefficient(time, timing, transition);
                (coeff.0[0], coeff.0[1], coeff.1, true)
            }
        };
        let mut fix = Vec::new();
        let mut pull = Vec::new();
        // Bridge into the centralized primitive-typed helpers in
        // `ppf_cts_core::datamodel::pin_apply`. Both this solver crate
        // and the migration's frontend preview kernels call into the
        // same module so preview and simulation produce identical
        // positions for the same input.
        use ppf_cts_core::datamodel::pin_apply;
        let to_arr = |v: Vector3<f32>| -> [f64; 3] {
            [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
        };
        let from_arr = |a: [f64; 3]| -> Vector3<f32> {
            Vector3::new(
                a[0] as f32,
                a[1] as f32,
                a[2] as f32,
            )
        };
        let apply_op = |op: &PinOperation,
                        position: Vector3<f32>,
                        vert_idx: usize,
                        time: f64|
         -> (Vector3<f32>, bool) {
            match op {
                PinOperation::MoveBy {
                    delta,
                    t_start,
                    t_end,
                    transition,
                    bezier_handles,
                } => {
                    if time < *t_start {
                        return (position, false);
                    }
                    let progress = if time >= *t_end {
                        1.0
                    } else {
                        pin_apply::progress_at(
                            time,
                            *t_start,
                            *t_end,
                            transition,
                            *bezier_handles,
                        )
                    };
                    let d_col = delta.column(vert_idx);
                    let d = [f64::from(d_col[0]), f64::from(d_col[1]), f64::from(d_col[2])];
                    let r = pin_apply::move_by_step(to_arr(position), d, progress);
                    (from_arr(r), true)
                }
                PinOperation::MoveTo {
                    target,
                    t_start,
                    t_end,
                    transition,
                    bezier_handles,
                } => {
                    if time < *t_start {
                        return (position, false);
                    }
                    if time >= *t_end {
                        return (target.column(vert_idx).into(), true);
                    }
                    let progress = pin_apply::progress_at(
                        time,
                        *t_start,
                        *t_end,
                        transition,
                        *bezier_handles,
                    );
                    let t_col = target.column(vert_idx);
                    let t = [f64::from(t_col[0]), f64::from(t_col[1]), f64::from(t_col[2])];
                    let r = pin_apply::move_to_step(to_arr(position), t, progress);
                    (from_arr(r), true)
                }
                PinOperation::Spin {
                    center,
                    axis,
                    angular_velocity,
                    t_start,
                    t_end,
                } => {
                    let angle =
                        pin_apply::spin_angle_rad(*angular_velocity as f64, *t_start, *t_end, time);
                    if angle <= 0.0 {
                        return (position, false);
                    }
                    let c = to_arr(*center);
                    let ax = [axis[0] as f64, axis[1] as f64, axis[2] as f64];
                    let r = pin_apply::spin_step(to_arr(position), c, ax, angle);
                    (from_arr(r), true)
                }
                PinOperation::Scale {
                    center,
                    factor,
                    t_start,
                    t_end,
                    transition,
                    bezier_handles,
                } => {
                    if time < *t_start {
                        return (position, false);
                    }
                    let cur = pin_apply::scale_factor_at(
                        time,
                        *t_start,
                        *t_end,
                        *factor as f64,
                        transition,
                        *bezier_handles,
                    );
                    let r = pin_apply::scale_step(to_arr(position), to_arr(*center), cur);
                    (from_arr(r), true)
                }
                PinOperation::Torque { .. } => {
                    // Torque is a force, not kinematic. Handled separately.
                    (position, false)
                }
                PinOperation::TransformKeyframes {
                    local,
                    times,
                    translations,
                    quaternions,
                    scales,
                    interps,
                    handles,
                    rest_t,
                } => {
                    if times.is_empty() {
                        return (position, false);
                    }
                    // Evaluate the sparse TRS keyframe timeline through the
                    // shared core helper so this branch stays bit-identical
                    // with the frontend preview path (slerp on Q, optional
                    // Bezier easing, R*S*local + T - rest_t).
                    let lv = local.column(vert_idx);
                    let l = [f64::from(lv[0]), f64::from(lv[1]), f64::from(lv[2])];
                    let out = ppf_cts_core::datamodel::quat::transform_keyframes_step(
                        l,
                        times,
                        translations,
                        quaternions,
                        scales,
                        interps,
                        handles,
                        *rest_t,
                        time,
                    );
                    (from_arr(out), true)
                }
            }
        };

        for pin in self.pin.iter() {
            // Check if this pin should be active at current time
            if let Some(unpin_t) = pin.unpin_time {
                if time >= unpin_t {
                    continue; // Skip this pin, it's been unpinned
                }
            }

            // If pin has ONLY torque operations, skip fix/pull; vertices are
            // free to move and receive force from the torque kernel instead.
            let torque_only = !pin.operations.is_empty()
                && pin.operations.iter().all(|op| matches!(op, PinOperation::Torque { .. }));
            if torque_only {
                continue;
            }

            for (i, &ind) in pin.index.iter().enumerate() {
                let dx = self.displacement.column(self.vert_dmap[ind] as usize);

                let mut kinematic = false;
                let mut position: Vector3<f32> = self.vert.column(ind).into();

                for op in pin.operations.iter() {
                    let (new_pos, did_move) =
                        apply_op(op, position, i, time);
                    position = new_pos;
                    kinematic = kinematic || did_move;
                }

                // Per-vertex pull weight overrides the scalar when present.
                // The decoder drops weight-~0 verts from `index`, so every
                // member of a per-vertex-weighted holder has w > 0 and takes
                // the pull branch (a zero weight here would otherwise route a
                // vertex to the hard `fix` branch and lock it).
                let w = pin
                    .pull_weights
                    .as_ref()
                    .map(|v| v[i])
                    .unwrap_or(pin.pull_w);

                if w > 0.0 {
                    pull.push(PullPair {
                        position: position + dx,
                        index: ind as u32,
                        weight: w,
                    });
                } else {
                    fix.push(FixPair {
                        position: position + dx,
                        ghat: self.args.constraint_ghat,
                        index: ind as u32,
                        kinematic,
                        stiffness: pin.stiffness,
                    });
                }
            }
        }

        // Build torque groups and vertex arrays (PCA + force computed on GPU)
        // Group by pin_group_id (Blender vertex group) so all vertices from
        // the same vertex group share one centroid and PCA axis.
        let mut torque_groups: Vec<TorqueGroup> = Vec::new();
        let mut torque_vertices: Vec<TorqueVertex> = Vec::new();
        {
            use std::collections::HashMap;
            // Key: pin_group_id string → (group index, axis_component, magnitude, hint_vertex)
            let mut group_map: HashMap<String, (u32, u32, f32, usize)> = HashMap::new();
            let mut group_verts: Vec<Vec<usize>> = Vec::new();

            for pin in self.pin.iter() {
                if let Some(unpin_t) = pin.unpin_time {
                    if time >= unpin_t {
                        continue;
                    }
                }
                for op in pin.operations.iter() {
                    if let PinOperation::Torque {
                        axis_component,
                        magnitude,
                        hint_vertex,
                        t_start,
                        t_end,
                    } = op
                    {
                        // Torque is full power inside window, zero outside
                        if time < *t_start || time > *t_end {
                            continue;
                        }
                        let key = pin.pin_group_id.clone();
                        let gid = if let Some(&(existing, _, _, _)) = group_map.get(&key) {
                            existing
                        } else {
                            let new_id = group_verts.len() as u32;
                            group_map.insert(key, (new_id, *axis_component, *magnitude, *hint_vertex));
                            group_verts.push(Vec::new());
                            new_id
                        };
                        for &ind in pin.index.iter() {
                            group_verts[gid as usize].push(ind);
                        }
                    }
                }
            }

            // Build final arrays
            for (_key, &(gid, axis_component, magnitude, hint_vertex)) in group_map.iter() {
                let verts = &group_verts[gid as usize];
                let group_id = torque_groups.len() as u32;
                let vertex_start = torque_vertices.len() as u32;
                for &ind in verts.iter() {
                    torque_vertices.push(TorqueVertex {
                        magnitude,
                        index: ind as u32,
                        group_id,
                    });
                }
                torque_groups.push(TorqueGroup {
                    axis_component,
                    vertex_start,
                    vertex_count: verts.len() as u32,
                    hint_vertex: hint_vertex as u32,
                });
            }
        }

        let stitch = {
            let mut stitch = Vec::new();
            for i in 0..self.stitch_ind.ncols() {
                stitch.push(Stitch {
                    index: Vec6u::from_iterator(
                        self.stitch_ind.column(i).iter().map(|&x| x as u32),
                    ),
                    weight: Vec6f::from_iterator(
                        self.stitch_w.column(i).iter().copied(),
                    ),
                    stiffness: self.stitch_stiffness[i],
                });
            }
            stitch
        };
        let mut floor = Vec::new();
        let mut sphere = Vec::new();
        for wall in self.wall.iter() {
            // Drop the wall from the contact set once it passes its active
            // duration (opt-in via -1.0 sentinel).
            if wall.active_duration >= 0.0 && (time as f32) >= wall.active_duration {
                continue;
            }
            let normal = wall.normal;
            let (j, k, w, kinematic) = blend_coeff(&wall.timing, &wall.transition);
            let position = wall.position.column(j) * (1.0 - w)
                + wall.position.column(k) * w;
            floor.push(Floor {
                ground: position,
                ghat: wall.contact_gap,
                friction: wall.friction,
                thickness: wall.thickness,
                up: normal,
                kinematic,
            });
        }
        for s in self.sphere.iter() {
            if s.active_duration >= 0.0 && (time as f32) >= s.active_duration {
                continue;
            }
            let reverse = s.inverted;
            let bowl = s.hemisphere;
            let (j, k, w, kinematic) = blend_coeff(&s.timing, &s.transition);
            let center = s.center.column(j) * (1.0 - w)
                + s.center.column(k) * w;
            let radius = s.radius[j] * (1.0 - w) + s.radius[k] * w;
            sphere.push(Sphere {
                center,
                ghat: s.contact_gap,
                friction: s.friction,
                radius,
                thickness: s.thickness,
                bowl,
                reverse,
                kinematic,
            });
        }
        Constraint {
            fix: CVec::from(&fix[..]),
            pull: CVec::from(&pull[..]),
            torque_groups: CVec::from(&torque_groups[..]),
            torque_vertices: CVec::from(&torque_vertices[..]),
            sphere: CVec::from(&sphere[..]),
            floor: CVec::from(&floor[..]),
            stitch: CVec::from(&stitch[..]),
            mesh: collision_mesh,
        }
    }

    pub fn update_param(&self, _: &SimArgs, time: f64, param: &mut ParamSet) {
        for (title, entries) in self.dyn_args.iter() {
            // Clamp to this title's final keyframe independently. Use a
            // per-title local so the clamp never carries over to titles
            // processed later (each title has its own schedule).
            let t = time.min(
                entries
                    .iter()
                    .fold(0.0_f64, |max_time, (kt, _)| max_time.max(*kt)),
            );
            let mut apply = |val: DynParamValue| match (title.as_str(), val) {
                ("gravity", DynParamValue::Vec3(v)) => {
                    param.gravity = Vec3f::new(v[0] as f32, v[1] as f32, v[2] as f32);
                }
                ("gravity", DynParamValue::Scalar(v)) => {
                    param.gravity = Vec3f::new(0.0, v as f32, 0.0);
                }
                ("wind", DynParamValue::Vec3(v)) => {
                    param.wind = Vec3f::new(v[0] as f32, v[1] as f32, v[2] as f32);
                }
                ("air-density", DynParamValue::Scalar(v)) => param.air_density = v as f32,
                ("air-friction", DynParamValue::Scalar(v)) => param.air_friction = v as f32,
                ("isotropic-air-friction", DynParamValue::Scalar(v)) => param.isotropic_air_friction = v as f32,
                ("dt", DynParamValue::Scalar(v)) => param.dt = v as f32,
                ("playback", DynParamValue::Scalar(v)) => param.playback = v as f32,
                ("inactive-momentum", DynParamValue::Scalar(v)) => param.inactive_momentum = v > 0.0,
                _ => (),
            };
            // windows(2) yields no pairs for 0- or 1-element schedules, so
            // there is no usize underflow on a bare/empty section. A single
            // keyframe holds its value constant (mirrors the collider loops).
            match entries.as_slice() {
                [] => continue,
                [(_, v)] => apply(*v),
                many => {
                    for w in many.windows(2) {
                        let (t0, v0) = w[0];
                        let (t1, v1) = w[1];
                        if t >= t0 && t <= t1 {
                            let delta_t = t1 - t0;
                            let weight = if delta_t > 0.0 {
                                (t - t0) / delta_t
                            } else {
                                1.0
                            };
                            apply(v0.lerp(v1, weight));
                        }
                    }
                }
            }
        }
    }

    /// Vertices that are strict kinematic `fix` constraints at time `t0` and
    /// therefore excluded from velocity overrides. Mirrors the
    /// fix/pull/torque_only classification in the pin-building loop: a pin
    /// becomes a strict `fix` constraint iff it's not torque-only and
    /// `pull_w == 0`. Weak pins (pull) and torque-only pins stay dynamic.
    fn hard_pinned_at(&self, t0: f64) -> std::collections::HashSet<usize> {
        self.pin
            .iter()
            .filter(|pin| pin.pull_w == 0.0)
            .filter(|pin| pin.unpin_time.is_none_or(|ut| t0 < ut))
            .filter(|pin| {
                let torque_only = !pin.operations.is_empty()
                    && pin
                        .operations
                        .iter()
                        .all(|op| matches!(op, PinOperation::Torque { .. }));
                !torque_only
            })
            .flat_map(|pin| pin.index.iter().copied())
            .collect()
    }

    /// Dynamic (non-hard-pinned) vertices of displacement group `dmap_idx` at
    /// time `t0`. Shared by the linear and angular velocity overrides.
    fn override_indices(&self, dmap_idx: u32, t0: f64) -> Vec<u32> {
        let hard_pinned = self.hard_pinned_at(t0);
        self.vert_dmap
            .iter()
            .enumerate()
            .filter(|(j, &dm)| dm == dmap_idx && !hard_pinned.contains(j))
            .map(|(j, _)| j as u32)
            .collect()
    }

    /// Compute velocity overrides that fire during this time step.
    /// Only returns overrides for keyframes crossed in [time, time + dt).
    /// Hard-pinned vertices (strict kinematic `fix` constraints) are excluded;
    /// weak pins (pull) and torque-only pins still receive the override since
    /// those vertices remain dynamic.
    pub fn get_velocity_overrides(&self, time: f64, dt: f32) -> Vec<(Vec<u32>, f32, f32, f32)> {
        let mut result = Vec::new();
        let time_end = time + dt as f64;
        for (title, entries) in self.dyn_args.iter() {
            if !title.starts_with("velocity:") {
                continue;
            }
            let dmap_idx: u32 = match title["velocity:".len()..].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            for (t0, v0) in entries.iter() {
                // Fire only when this step crosses the keyframe time
                if *t0 >= time && *t0 < time_end {
                    if let DynParamValue::Vec3(vel) = v0 {
                        let indices = self.override_indices(dmap_idx, *t0);
                        if !indices.is_empty() {
                            result.push((indices, vel[0] as f32, vel[1] as f32, vel[2] as f32));
                        }
                    }
                }
            }
        }
        result
    }

    /// Compute angular velocity overrides that fire during this time step.
    /// Mirrors [`get_velocity_overrides`] but for `angular_velocity:` keys.
    /// Each firing entry carries the chosen principal-axis index (0/1/2) and a
    /// signed angular speed (rad/s) packed as `[pca_index, speed, _]`; the
    /// actual world-space spin axis is resolved by the caller from the body's
    /// live geometry so it tracks the simulated (rotated / deformed) pose.
    /// Returns `(vertex indices, pca_index, speed_rad_per_s)`.
    pub fn get_angular_velocity_overrides(&self, time: f64, dt: f32) -> Vec<(Vec<u32>, u32, f32)> {
        let mut result = Vec::new();
        let time_end = time + dt as f64;
        for (title, entries) in self.dyn_args.iter() {
            if !title.starts_with("angular_velocity:") {
                continue;
            }
            let dmap_idx: u32 = match title["angular_velocity:".len()..].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            for (t0, v0) in entries.iter() {
                if *t0 >= time && *t0 < time_end {
                    if let DynParamValue::Vec3(vel) = v0 {
                        let indices = self.override_indices(dmap_idx, *t0);
                        if !indices.is_empty() {
                            let pca_index = vel[0].round().clamp(0.0, 2.0) as u32;
                            let speed = vel[1] as f32;
                            result.push((indices, pca_index, speed));
                        }
                    }
                }
            }
        }
        result
    }

    /// Compute fixed world-axis angular velocity overrides that fire during
    /// this time step. Counterpart to [`get_angular_velocity_overrides`] for
    /// the World X/Y/Z and Custom axis modes: each entry already carries the
    /// full world-space angular velocity vector `ω` (rad/s), so no per-firing
    /// principal-axis resolution is needed. Returns `(vertex indices, ω)`.
    pub fn get_angular_velocity_world_overrides(
        &self,
        time: f64,
        dt: f32,
    ) -> Vec<(Vec<u32>, [f32; 3])> {
        let mut result = Vec::new();
        let time_end = time + dt as f64;
        for (title, entries) in self.dyn_args.iter() {
            if !title.starts_with("angular_velocity_world:") {
                continue;
            }
            let dmap_idx: u32 = match title["angular_velocity_world:".len()..].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            for (t0, v0) in entries.iter() {
                if *t0 >= time && *t0 < time_end {
                    if let DynParamValue::Vec3(w) = v0 {
                        let indices = self.override_indices(dmap_idx, *t0);
                        if !indices.is_empty() {
                            result.push((indices, [w[0] as f32, w[1] as f32, w[2] as f32]));
                        }
                    }
                }
            }
        }
        result
    }

    pub fn build_collision_window_table(&self) -> CollisionWindowTable {
        // Single source of truth for the cap (also enforced GPU-side via
        // the `MAX_COLLISION_WINDOWS` #define in cpp/main/main.cu).
        const MAX_WINDOWS: usize = ppf_cts_core::datamodel::object::MAX_COLLISION_WINDOWS;

        // Find all collision_window entries
        let mut windows_by_dmap: HashMap<u32, Vec<(f64, f64)>> = HashMap::new();
        for (title, entries) in self.dyn_args.iter() {
            if !title.starts_with("collision_window:") {
                continue;
            }
            let dmap_idx: u32 = match title["collision_window:".len()..].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let all_wins: Vec<(f64, f64)> = entries.iter()
                .filter_map(|(t_start, v)| {
                    if let DynParamValue::Scalar(t_end) = v {
                        Some((*t_start, *t_end))
                    } else {
                        None
                    }
                })
                .collect();
            if all_wins.len() > MAX_WINDOWS {
                // Loud: a FixedScene built outside the Python builder's
                // ValueError guard reaches here, and silently dropping
                // windows changes simulation results invisibly.
                error!("collision_window:{} has {} windows, exceeding the cap of {}; \
                        extra windows are dropped",
                       dmap_idx, all_wins.len(), MAX_WINDOWS);
            }
            let wins: Vec<(f64, f64)> = all_wins.into_iter().take(MAX_WINDOWS).collect();
            if !wins.is_empty() {
                windows_by_dmap.insert(dmap_idx, wins);
            }
        }

        if windows_by_dmap.is_empty() {
            return CollisionWindowTable::empty();
        }

        let n_groups = self.displacement.ncols() as u32;

        // Build flat window table: n_groups * MAX_WINDOWS * 2
        let mut windows = vec![0.0f32; n_groups as usize * MAX_WINDOWS * 2];
        let mut window_counts = vec![0u32; n_groups as usize];

        for (&dm, wins) in &windows_by_dmap {
            if (dm as usize) < n_groups as usize {
                let count = wins.len().min(MAX_WINDOWS);
                window_counts[dm as usize] = count as u32;
                for (w, (t_start, t_end)) in wins.iter().enumerate().take(count) {
                    windows[dm as usize * MAX_WINDOWS * 2 + w * 2] = *t_start as f32;
                    windows[dm as usize * MAX_WINDOWS * 2 + w * 2 + 1] = *t_end as f32;
                }
            }
        }

        CollisionWindowTable {
            vert_dmap: self.vert_dmap.clone(),
            windows,
            window_counts,
            n_groups,
            has_windows: true,
        }
    }

    /// The initial vertices with per-vertex instancing displacement applied
    /// (`self.vert + displacement[dmap]`), the base the simulation starts from.
    fn displaced_vert(&self) -> Matrix3xX<f32> {
        let mut vert = self.vert.clone();
        apply_displacement(&mut vert, &self.displacement, &self.vert_dmap);
        vert
    }

    /// Apply the same per-vertex instancing displacement to a rest-vertex set
    /// that `make_mesh` applies to the static rest pose: masked entries take
    /// `base + displacement`, unmasked entries fall back to the displaced
    /// initial vert (`vert`). Shared by the static `rest_vert` and the
    /// per-frame `rest_vert_schedule` so both stay in the same space.
    fn displaced_rest(
        &self,
        base: &Matrix3xX<f32>,
        vert: &Matrix3xX<f32>,
        mask: &[bool],
    ) -> Matrix3xX<f32> {
        let mut rest = base.clone();
        for (i, mut x) in rest.column_iter_mut().enumerate() {
            if mask[i] {
                x += self.displacement.column(self.vert_dmap[i] as usize);
            } else {
                x.copy_from(&vert.column(i));
            }
        }
        rest
    }

    /// Per-body PDRD rows as written by the Python frontend. Each row
    /// is 16 f32s: `[vertex_start, vertex_count, volume, centroid.x,
    /// centroid.y, centroid.z, gram_inv[0..9 row-major],
    /// mass_per_vertex]`. Empty Vec means "no PDRD bodies."
    pub fn pdrd_body_rows(&self) -> &[f32] {
        &self.pdrd_body_rows
    }

    /// Per-vertex 1-based PDRD body id (0 = not in any PDRD body).
    pub fn pdrd_vert_index(&self) -> &[u32] {
        &self.pdrd_vert_index
    }

    /// Flat list of global vertex indices participating in PDRD
    /// bodies (body-major).
    pub fn pdrd_vert_list(&self) -> &[u32] {
        &self.pdrd_vert_list
    }

    /// Centered rest position per entry of `pdrd_vert_list`, packed
    /// (x, y, z). Length 3 * pdrd_vert_list.len().
    pub fn pdrd_rest_centered(&self) -> &[f32] {
        &self.pdrd_rest_centered
    }

    pub fn make_mesh(&mut self) -> MeshSet {
        let vert = self.displaced_vert();
        // Capture the vertex count before `vert` is moved into the MeshSet so
        // a faceless particle cloud sizes its neighbor tables to the real
        // vertex buffer (see Mesh::new).
        let n_vert = vert.ncols();
        let rest_vertex = self
            .rest_vert
            .as_ref()
            .map(|rv| self.displaced_rest(rv, &vert, &self.rest_vert_mask));
        // Bending reference rest shape: same instancing-displacement
        // treatment as `rest_vertex`, but gated on its own per-vertex mask
        // (the objects that opted into a reference rest angle). Unmasked
        // entries fall back to the displaced initial vert, so a hinge on a
        // non-reference object computes the same angle whether or not any
        // reference exists in the scene.
        let bend_rest_vertex = self
            .bend_rest_vert
            .as_ref()
            .map(|bv| self.displaced_rest(bv, &vert, &self.bend_rest_vert_mask));
        MeshSet {
            vertex: vert,
            uv: self.uv.clone(),
            mesh: SimMesh::new(
                self.rod.clone(),
                self.tri.clone(),
                self.tet.clone(),
                self.shell_count,
                n_vert,
            ),
            rest_vertex,
            bend_rest_vertex,
            bend_rest_vertex_mask: self.bend_rest_vert_mask.clone(),
        }
    }

    /// The streamed time-varying rest shape, if any: keyframe times (seconds)
    /// and per-keyframe rest-vertex sets in the same coordinate space as
    /// `make_mesh`'s `rest_vertex` (instancing displacement applied). The
    /// backend recomputes `inv_rest` from each frame via
    /// `builder::compute_inv_rest` and interpolates between them per step.
    pub fn rest_shape_schedule(&self) -> Option<(Vec<f64>, Vec<Matrix3xX<f32>>)> {
        let sched = self.rest_vert_schedule.as_ref()?;
        let vert = self.displaced_vert();
        let frames = sched
            .frames
            .iter()
            .map(|f| self.displaced_rest(f, &vert, &self.rest_vert_mask))
            .collect();
        Some((sched.times.clone(), frames))
    }
}
