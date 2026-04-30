// File: scene.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::builder::{convert_prop, make_collision_mesh};
use super::data::*;
use super::{CVec, MeshSet, ParamSet, ProgramArgs, Props, SimArgs, SimMesh};
use bytemuck::{cast_slice, Pod};
use log::warn;
use more_asserts::*;
use na::{Const, Matrix, Matrix2x3, Matrix2xX, Matrix3xX, Matrix4xX, VecStorage, Vector3};
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
    stitch_ind: Matrix4xX<usize>,
    stitch_w: Matrix4xX<f32>,
    pin: Vec<Pin>,
    wall: Vec<InvisibleWall>,
    sphere: Vec<InvisibleSphere>,
    rest_vert: Option<Matrix3xX<f32>>,
    rest_vert_mask: Vec<bool>,
    shell_count: usize,
    rod_param: Vec<(String, ParamValueList)>,
    tri_param: Vec<(String, ParamValueList)>,
    tet_param: Vec<(String, ParamValueList)>,
    static_param: Vec<(String, ParamValueList)>,
}

enum ParamValueList {
    Model(Vec<Model>),
    Value(Vec<f32>),
}

enum PinOperation {
    MoveBy {
        delta: Matrix3xX<f32>,
        t_start: f64,
        t_end: f64,
        transition: String,
    },
    MoveTo {
        target: Matrix3xX<f32>,
        t_start: f64,
        t_end: f64,
        transition: String,
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
            format!("Data length is not a multiple of {}", C),
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

/// Spherical linear interpolation of quaternions in (w,x,y,z) order.
fn quat_slerp(q0: [f64; 4], q1: [f64; 4], t: f64) -> [f64; 4] {
    let mut dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    let mut q1 = q1;
    if dot < 0.0 {
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
        dot = -dot;
    }
    if dot > 0.9995 {
        let r = [
            q0[0] + t * (q1[0] - q0[0]),
            q0[1] + t * (q1[1] - q0[1]),
            q0[2] + t * (q1[2] - q0[2]),
            q0[3] + t * (q1[3] - q0[3]),
        ];
        let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
        return [r[0] / n, r[1] / n, r[2] / n, r[3] / n];
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let a = ((1.0 - t) * theta).sin() / sin_theta;
    let b = (t * theta).sin() / sin_theta;
    [
        a * q0[0] + b * q1[0],
        a * q0[1] + b * q1[1],
        a * q0[2] + b * q1[2],
        a * q0[3] + b * q1[3],
    ]
}

/// Cubic Bezier easing: given normalized t in [0,1] and normalized
/// control-point handles (hr_x, hr_y, hl_x, hl_y), solve B_x(u) = t via
/// Newton and return B_y(u).  Mirrors `frontend._scene_._bezier_progress`.
fn bezier_progress(t: f64, h: [f64; 4]) -> f64 {
    let (p1x, p1y, p2x, p2y) = (h[0], h[1], h[2], h[3]);
    let mut u = t;
    for _ in 0..8 {
        let omu = 1.0 - u;
        let omu2 = omu * omu;
        let u2 = u * u;
        let u3 = u2 * u;
        let bx = 3.0 * omu2 * u * p1x + 3.0 * omu * u2 * p2x + u3;
        let dbx =
            3.0 * omu2 * p1x + 6.0 * omu * u * (p2x - p1x) + 3.0 * u2 * (1.0 - p2x);
        if dbx.abs() < 1e-10 {
            break;
        }
        u -= (bx - t) / dbx;
        u = u.clamp(0.0, 1.0);
    }
    let omu = 1.0 - u;
    let omu2 = omu * omu;
    let u2 = u * u;
    let u3 = u2 * u;
    (3.0 * omu2 * u * p1y + 3.0 * omu * u2 * p2y + u3).clamp(0.0, 1.0)
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
            let time: f64 = parts[0].parse().expect("Failed to parse time");
            if parts.len() == 4 {
                let x: f64 = parts[1].parse().expect("Failed to parse x");
                let y: f64 = parts[2].parse().expect("Failed to parse y");
                let z: f64 = parts[3].parse().expect("Failed to parse z");
                curr_entry.push((time, DynParamValue::Vec3([x, y, z])));
            } else if parts.len() == 2 {
                let value: f64 = parts[1].parse().expect("Failed to parse value");
                curr_entry.push((time, DynParamValue::Scalar(value)));
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
                .unwrap_or_else(|| panic!("Failed to read {}", key)) as usize
        };
        let read_bool = |count: &Value, key: &str| {
            count
                .get(key)
                .and_then(|v| v.as_bool())
                .unwrap_or_else(|| panic!("Failed to read {}", key))
        };
        let read_f32 = |count: &Value, key: &str| {
            count
                .get(key)
                .and_then(|v| v.as_float())
                .unwrap_or_else(|| panic!("Failed to read {}", key)) as f32
        };
        let read_f64 = |count: &Value, key: &str| {
            count
                .get(key)
                .and_then(|v| v.as_float())
                .unwrap_or_else(|| panic!("Failed to read {}", key))
        };
        let read_string = |count: &Value, key: &str| {
            count
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("Failed to read {}", key))
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
        let _rod_vert_start = read_usize(count, "rod_vert_start");
        let _rod_vert_end = read_usize(count, "rod_vert_end");
        let _shell_vert_start = read_usize(count, "shell_vert_start");
        let _shell_vert_end = read_usize(count, "shell_vert_end");
        let _rod_count = read_usize(count, "rod_count");
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

        let displacement_mat = read_mat_from_file::<f64, 3>(&displacement_path)
            .expect("Failed to read displacement")
            .map(|v| v as f32);
        let vert_dmap_mat = read_vec::<u32>(&vert_dmap_path).expect("Failed to read vert_dmap");
        let vert_mat = read_mat_from_file::<f64, 3>(&vert_path)
            .expect("Failed to read vert")
            .map(|v| v as f32);
        let vel_mat = read_mat_from_file::<f32, 3>(&vel_path).expect("Failed to read velocity");
        let has_rest_vert = count
            .get("has_rest_vert")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let (rest_vert_mat, rest_vert_mask) = if has_rest_vert {
            let rest_vert_path = format!("{}/bin/rest_vert.bin", args.path);
            let rest_vert_mask_path = format!("{}/bin/rest_vert_mask.bin", args.path);
            let mat = read_mat_from_file::<f64, 3>(&rest_vert_path)
                .expect("Failed to read rest_vert")
                .map(|v| v as f32);
            let mask_bytes = read_vec::<u8>(&rest_vert_mask_path)
                .expect("Failed to read rest_vert_mask");
            let mask: Vec<bool> = mask_bytes.iter().map(|&b| b != 0).collect();
            assert_eq!(mat.ncols(), n_vert, "rest_vert size mismatch");
            assert_eq!(mask.len(), n_vert, "rest_vert_mask size mismatch");
            (Some(mat), mask)
        } else {
            (None, vec![false; n_vert])
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
                    .map(|v| v as f32),
            )
        } else {
            (Vec::new(), Matrix3xX::<f32>::zeros(0))
        };
        let static_tri_mat = if n_static_tri > 0 {
            read_mat_from_file::<usize, 3>(&static_tri_path).expect("Failed to read static_tri")
        } else {
            Matrix3xX::<usize>::zeros(0)
        };
        let (stitch_ind_mat, stitch_w_mat) = if n_stitch > 0 {
            (
                read_mat_from_file::<usize, 4>(&stitch_ind_path)
                    .expect("Failed to read stitch_ind"),
                read_mat_from_file::<f32, 4>(&stitch_w_path).expect("Failed to read stitch_w"),
            )
        } else {
            (Matrix4xX::<usize>::zeros(0), Matrix4xX::<f32>::zeros(0))
        };

        let mut pin = Vec::new();
        for i in 0..n_pin_block {
            let title = format!("pin-{}", i);
            let count = parsed
                .get(&title)
                .unwrap_or_else(|| panic!("Failed to read pin {}", i));
            let n_pin = read_usize(count, "pin");
            let operation_count = read_usize(count, "operation_count");
            let unpin_time = count.get("unpin_time").and_then(|v| v.as_float());
            let pull_w = read_f32(count, "pull");
            let pin_group_id = count
                .get("pin_group_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let pin_ind_path = format!("{}/bin/pin-ind-{}.bin", args.path, i);

            let pin_ind = read_vec::<usize>(&pin_ind_path).expect("Failed to read pin index");
            assert_eq!(pin_ind.len(), n_pin as usize);

            // Read operations in order
            let mut operations = Vec::new();
            for j in 0..operation_count {
                let op_title = format!("pin-{}-op-{}", i, j);
                let op_entry = parsed
                    .get(&op_title)
                    .unwrap_or_else(|| panic!("Failed to read operation {} for pin {}", j, i));

                let op_type = read_string(op_entry, "type");

                match op_type.as_str() {
                    "move_by" => {
                        let t_start = read_f64(op_entry, "t_start");
                        let t_end = read_f64(op_entry, "t_end");
                        let transition = read_string(op_entry, "transition");
                        let delta_path = format!("{}/bin/pin-{}-op-{}.bin", args.path, i, j);
                        let delta = read_mat_from_file::<f64, 3>(&delta_path)
                            .expect("Failed to read move_by delta")
                            .map(|v| v as f32);
                        operations.push(PinOperation::MoveBy {
                            delta,
                            t_start,
                            t_end,
                            transition,
                        });
                    }
                    "move_to" => {
                        let t_start = read_f64(op_entry, "t_start");
                        let t_end = read_f64(op_entry, "t_end");
                        let transition = read_string(op_entry, "transition");
                        let target_path = format!("{}/bin/pin-{}-op-{}.bin", args.path, i, j);
                        let target = read_mat_from_file::<f64, 3>(&target_path)
                            .expect("Failed to read move_to target")
                            .map(|v| v as f32);
                        operations.push(PinOperation::MoveTo {
                            target,
                            t_start,
                            t_end,
                            transition,
                        });
                    }
                    "spin" => {
                        let center_x = read_f32(op_entry, "center_x");
                        let center_y = read_f32(op_entry, "center_y");
                        let center_z = read_f32(op_entry, "center_z");
                        let center = Vector3::new(center_x, center_y, center_z)
                            .map(|v| v as f32);
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
                            .map(|v| v as f32);
                        let factor = read_f32(op_entry, "factor");
                        let t_start = read_f64(op_entry, "t_start");
                        let t_end = read_f64(op_entry, "t_end");
                        let transition = read_string(op_entry, "transition");
                        operations.push(PinOperation::Scale {
                            center,
                            factor,
                            t_start,
                            t_end,
                            transition,
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
                            read_f64(op_entry, "rest_tx"),
                            read_f64(op_entry, "rest_ty"),
                            read_f64(op_entry, "rest_tz"),
                        ];
                        let base = format!("{}/bin/pin-{}-op-{}", args.path, i, j);
                        let local = read_mat_from_file::<f64, 3>(&format!("{}.bin", base))
                            .expect("Failed to read transform_keyframes local verts")
                            .map(|v| v as f32);
                        let times = read_vec::<f64>(&format!("{}-time.bin", base))
                            .expect("Failed to read keyframe times");
                        assert_eq!(times.len(), keyframe_count);
                        let t_flat = read_vec::<f64>(&format!("{}-translation.bin", base))
                            .expect("Failed to read keyframe translations");
                        assert_eq!(t_flat.len(), 3 * keyframe_count);
                        let translations: Vec<[f64; 3]> = t_flat
                            .chunks_exact(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect();
                        let q_flat = read_vec::<f64>(&format!("{}-quaternion.bin", base))
                            .expect("Failed to read keyframe quaternions");
                        assert_eq!(q_flat.len(), 4 * keyframe_count);
                        let quaternions: Vec<[f64; 4]> = q_flat
                            .chunks_exact(4)
                            .map(|c| [c[0], c[1], c[2], c[3]])
                            .collect();
                        let s_flat = read_vec::<f64>(&format!("{}-scale.bin", base))
                            .expect("Failed to read keyframe scales");
                        assert_eq!(s_flat.len(), 3 * keyframe_count);
                        let scales: Vec<[f64; 3]> = s_flat
                            .chunks_exact(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect();
                        let interps = if keyframe_count > 1 {
                            let v = read_vec::<u8>(&format!("{}-interp.bin", base))
                                .expect("Failed to read segment interp codes");
                            assert_eq!(v.len(), keyframe_count - 1);
                            v
                        } else {
                            Vec::new()
                        };
                        let handles = if keyframe_count > 1 {
                            let h_flat = read_vec::<f64>(&format!("{}-handles.bin", base))
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
                    _ => panic!("Unknown operation type: {}", op_type),
                }
            }

            pin.push(Pin {
                index: pin_ind,
                operations,
                unpin_time,
                pull_w,
                pin_group_id,
            });
        }

        let mut wall = Vec::new();
        for i in 0..n_wall {
            let title = format!("wall-{}", i);
            let count = parsed
                .get(&title)
                .unwrap_or_else(|| panic!("Failed to read wall {}", i));
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
                        .map(|v| v as f32);
                let wall_timing =
                    read_vec::<f64>(&format!("{}/bin/wall-timing-{}.bin", args.path, i))
                        .expect("Failed to read wall timing");
                let contact_gap = read_f32(count, "contact-gap");
                let friction = read_f32(count, "friction");
                let active_duration =
                    read_optional_f32(count, "active-duration").unwrap_or(-1.0);
                let thickness =
                    read_optional_f32(count, "thickness").unwrap_or(1.0);
                assert_gt!(
                    thickness,
                    0.0,
                    "invisible wall thickness must be > 0 (got {})",
                    thickness
                );
                assert_eq!(position.ncols(), n_keyframe as usize);
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
            let title = format!("sphere-{}", i);
            let count = parsed
                .get(&title)
                .unwrap_or_else(|| panic!("Failed to read sphere {}", i));
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
                .map(|v| v as f32);
                let radius = read_vec::<f32>(&format!("{}/bin/sphere-radius-{}.bin", args.path, i))
                    .expect("Failed to read sphere radius");
                let timing = read_vec::<f64>(&format!("{}/bin/sphere-timing-{}.bin", args.path, i))
                    .expect("Failed to read sphere timing");
                let contact_gap = read_f32(count, "contact-gap");
                let friction = read_f32(count, "friction");
                let active_duration =
                    read_optional_f32(count, "active-duration").unwrap_or(-1.0);
                let thickness =
                    read_optional_f32(count, "thickness").unwrap_or(1.0);
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

        assert_eq!(vert_mat.ncols(), n_vert as usize);
        assert_eq!(vel_mat.ncols(), n_vert as usize);
        assert_eq!(rod_mat.ncols(), n_rod as usize);
        assert_eq!(tri_mat.ncols(), n_tri as usize);
        assert_eq!(tet_mat.ncols(), n_tet as usize);
        assert_eq!(static_vert_mat.ncols(), n_static_vert as usize);
        assert_eq!(static_tri_mat.ncols(), n_static_tri as usize);
        assert_eq!(stitch_ind_mat.ncols(), n_stitch as usize);
        assert_eq!(stitch_w_mat.ncols(), n_stitch as usize);

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

        for entry in fs::read_dir(&param_dir).expect("Failed to read param directory") {
            let entry = entry.expect("Failed to read entry");
            let path = entry.path();
            if path.is_file() {
                let file_name = path.file_name().unwrap().to_str().unwrap();

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
                                if k == 0 {
                                    Model::Arap
                                } else if k == 1 {
                                    Model::StVK
                                } else if k == 2 {
                                    Model::BaraffWitkin
                                } else if k == 3 {
                                    Model::SNHk
                                } else {
                                    panic!("Unknown model type: {}", k);
                                }
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
            pin,
            wall,
            sphere,
            rest_vert: rest_vert_mat,
            rest_vert_mask,
            shell_count,
            rod_param,
            tri_param,
            tet_param,
            static_param,
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
                eprintln!("Failed to create parameter summary file: {}", e);
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
            self.write_param_stats(&mut content, &self.tri_param, self.tri.ncols());

            // Add mass and area statistics for shells
            if !props.face.is_empty() && !face_area.is_empty() {
                let masses: Vec<f32> = props.face.iter().map(|p| p.mass).collect();
                if !masses.is_empty() {
                    let min_mass = masses.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_mass = masses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum_mass: f64 = masses.iter().map(|&v| v as f64).sum();
                    let mean_mass = (sum_mass / masses.len() as f64) as f32;

                    content.push_str(&format!(
                        "mass: (max: {:.4e}, min: {:.4e}, mean: {:.4e})\n",
                        max_mass, min_mass, mean_mass
                    ));
                }

                let areas: Vec<f32> = face_area.to_vec();
                if !areas.is_empty() {
                    let min_area = areas.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_area = areas.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum_area: f64 = areas.iter().map(|&v| v as f64).sum();
                    let mean_area = (sum_area / areas.len() as f64) as f32;

                    content.push_str(&format!(
                        "area: (max: {:.4e}, min: {:.4e}, mean: {:.4e})\n",
                        max_area, min_area, mean_area
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
            self.write_param_stats(&mut content, &self.tet_param, self.tet.ncols());

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
                    "mass: (max: {:.4e}, min: {:.4e}, mean: {:.4e})\n",
                    max_mass, min_mass, mean_mass
                ));
                content.push_str(&format!(
                    "volume: (max: {:.4e}, min: {:.4e}, mean: {:.4e})\n",
                    max_volume, min_volume, mean_volume
                ));
            }
            content.push('\n');
        }

        // Write Rod parameters summary
        if self.rod.ncols() > 0 {
            content.push_str(&format!("=== Rods ({} elements) ===\n\n", self.rod.ncols()));
            self.write_param_stats(&mut content, &self.rod_param, self.rod.ncols());

            // Add mass statistics for rods (only for actual rod elements, not shell edges)
            let rod_count = self.rod.ncols().min(props.edge.len());
            if rod_count > 0 {
                let rod_props = &props.edge[0..rod_count];
                let masses: Vec<f32> = rod_props.iter().map(|p| p.mass).collect();
                if !masses.is_empty() {
                    let min_mass = masses.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_mass = masses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum_mass: f64 = masses.iter().map(|&v| v as f64).sum();
                    let mean_mass = (sum_mass / masses.len() as f64) as f32;

                    content.push_str(&format!(
                        "mass: (max: {:.4e}, min: {:.4e}, mean: {:.4e})\n",
                        max_mass, min_mass, mean_mass
                    ));
                }
            }
            content.push('\n');
        }

        // Write Static object parameters summary
        if self.static_tri.ncols() > 0 {
            content.push_str(&format!(
                "=== Static Objects ({} elements) ===\n\n",
                self.static_tri.ncols()
            ));
            self.write_param_stats(&mut content, &self.static_param, self.static_tri.ncols());
            content.push('\n');
        }

        if let Err(e) = file.write_all(content.as_bytes()) {
            eprintln!("Failed to write parameter summary: {}", e);
        } else {
            println!("Parameter summary written to: {}", summary_path);
        }
    }

    fn write_param_stats(
        &self,
        content: &mut String,
        params: &[(String, ParamValueList)],
        _n_elements: usize,
    ) {
        for (name, values) in params {
            match values {
                ParamValueList::Model(models) => {
                    // Count occurrences of each model type
                    let mut model_counts = HashMap::new();
                    for model in models {
                        *model_counts.entry(format!("{:?}", model)).or_insert(0) += 1;
                    }

                    content.push_str(&format!("{}: (", name));
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
                        "{}: (max: {:.4e}, min: {:.4e}, mean: {:.4e})\n",
                        name, max, min, mean
                    ));
                }
            }
        }
    }

    pub fn args(&self) -> SimArgs {
        self.args.clone()
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
                let rod = mesh.mesh.mesh.edge.column(i);
                let x0 = mesh.vertex.column(rod[0]);
                let x1 = mesh.vertex.column(rod[1]);
                let mut length = (x1 - x0).norm();
                let initial_length = length;
                let mut density = None;
                for (name, value) in &self.rod_param {
                    if name == "contact-gap" {
                        ghat = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected contact-gap parameter to be a value list"),
                        };
                    } else if name == "contact-offset" {
                        offset = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected contact-offset parameter to be a value list"),
                        };
                    } else if name == "friction" {
                        friction = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected friction parameter to be a value list"),
                        };
                    } else if name == "young-mod" {
                        stiffness = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected young-modulus parameter to be a value list"),
                        };
                    } else if name == "density" {
                        density = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected density parameter to be a value list"),
                        };
                    } else if name == "bend" {
                        bend = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected bend-modulus parameter to be a value list"),
                        };
                    } else if name == "length-factor" {
                        length_factor = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected rod-length-factor parameter to be a value list"),
                        };
                    } else if name == "strain-limit" {
                        strainlimit = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected strain-limit parameter to be a value list"),
                        };
                    } else if name == "bend-plasticity" {
                        bend_plasticity = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected bend-plasticity parameter to be a value list"),
                        };
                    } else if name == "bend-plasticity-threshold" {
                        bend_plasticity_threshold = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected bend-plasticity-threshold to be a value list"),
                        };
                    } else if name == "bend-rest-from-geometry" {
                        bend_rest_from_geometry = match value {
                            ParamValueList::Value(v) => Some(v[i] != 0.0),
                            _ => panic!("Expected bend-rest-from-geometry to be a value list"),
                        };
                    } else if name == "pressure" {
                        // Pressure is a tri-only parameter; ignore for rods.
                    } else {
                        panic!("Unknown rod parameter: {}", name);
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
                assert_gt!(ghat, 0.0, "Contact gap must be non-negative");
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
                };
                let param_idx = *edge_param_map.entry(param).or_insert_with(|| {
                    let new_idx = edge_params.len() as u32;
                    edge_params.push(param);
                    new_idx
                });

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
                for (name, value) in &self.tri_param {
                    if name == "contact-gap" {
                        ghat = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected contact-gap parameter to be a value list"),
                        };
                    } else if name == "contact-offset" {
                        offset = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected contact-offset parameter to be a value list"),
                        };
                    } else if name == "friction" {
                        friction = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected friction parameter to be a value list"),
                        };
                    } else if name == "strain-limit" {
                        strainlimit = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected strainlimit parameter to be a value list"),
                        };
                    } else if name == "bend" {
                        bend = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected bend-modulus parameter to be a value list"),
                        };
                    } else if name == "shrink-x" {
                        shrink_x = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected shrink-x parameter to be a value list"),
                        };
                    } else if name == "shrink-y" {
                        shrink_y = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected shrink-y parameter to be a value list"),
                        };
                    } else if name == "model" {
                        model = match value {
                            ParamValueList::Model(v) => Some(v[i]),
                            _ => panic!("Expected model parameter to be a name list"),
                        };
                    } else if name == "density" {
                        density = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected density parameter to be a value list"),
                        };
                    } else if name == "young-mod" {
                        young_mod = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected young-modulus parameter to be a value list"),
                        };
                    } else if name == "poiss-rat" {
                        poiss_rat = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected poisson's ratio parameter to be a value list"),
                        };
                    } else if name == "pressure" {
                        pressure = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected pressure parameter to be a value list"),
                        };
                    } else if name == "plasticity" {
                        plasticity = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected plasticity parameter to be a value list"),
                        };
                    } else if name == "plasticity-threshold" {
                        plasticity_threshold = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected plasticity-threshold parameter to be a value list"),
                        };
                    } else if name == "bend-plasticity" {
                        bend_plasticity = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected bend-plasticity parameter to be a value list"),
                        };
                    } else if name == "bend-plasticity-threshold" {
                        bend_plasticity_threshold = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected bend-plasticity-threshold to be a value list"),
                        };
                    } else if name == "bend-rest-from-geometry" {
                        bend_rest_from_geometry = match value {
                            ParamValueList::Value(v) => Some(v[i] != 0.0),
                            _ => panic!("Expected bend-rest-from-geometry to be a value list"),
                        };
                    } else {
                        panic!("Unknown face parameter: {}", name);
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
                assert_gt!(density, 0.0, "Density must be positive");
                assert_gt!(young_mod, 0.0, "Young's modulus must be positive");
                assert_gt!(area, 0.0, "Area must be positive");
                assert_ge!(pressure, 0.0, "Pressure must be non-negative");
                assert_ge!(friction, 0.0, "Friction must be non-negative");
                assert_ge!(bend, 0.0, "Bend modulus must be non-negative");
                assert_gt!(shrink_x, 0.0, "Shrink X factor must be positive");
                assert_gt!(shrink_y, 0.0, "Shrink Y factor must be positive");
                // Shrink/extend and strain-limit cannot be combined on the
                // same face: each rewrites the rest shape independently, so
                // the strain bound becomes ill-defined when both are active.
                assert!(
                    !((shrink_x != 1.0 || shrink_y != 1.0) && strainlimit > 0.0),
                    "Face {}: shrink (x={}, y={}) conflicts with non-zero strain-limit ({})",
                    i, shrink_x, shrink_y, strainlimit
                );
                assert_gt!(ghat, 0.0, "Contact gap must be non-negative");
                assert_gt!(poiss_rat, 0.0, "Poisson's ratio must be positive");
                assert_lt!(poiss_rat, 0.5, "Poisson's ratio must be less than 0.5");
                let (mu, lambda) = convert_prop(young_mod, poiss_rat);
                let mass = density * area;

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
                };
                let param_idx = *face_param_map.entry(param).or_insert_with(|| {
                    let new_idx = face_params.len() as u32;
                    face_params.push(param);
                    new_idx
                });

                FaceProp {
                    area,
                    mass,
                    fixed: false,
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
                for (name, value) in &self.tet_param {
                    if name == "model" {
                        model = match value {
                            ParamValueList::Model(v) => Some(v[i]),
                            _ => panic!("Expected model parameter to be a name list"),
                        };
                    } else if name == "density" {
                        density = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected density parameter to be a value list"),
                        };
                    } else if name == "young-mod" {
                        young_mod = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected young-modulus parameter to be a value list"),
                        };
                    } else if name == "poiss-rat" {
                        poiss_rat = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected poisson's ratio parameter to be a value list"),
                        };
                    } else if name == "shrink" {
                        shrink = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected shrink parameter to be a value list"),
                        };
                    } else if name == "pressure" {
                        // Pressure is a tri-only parameter; ignore for tets.
                    } else if name == "plasticity" {
                        plasticity = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected plasticity parameter to be a value list"),
                        };
                    } else if name == "plasticity-threshold" {
                        plasticity_threshold = match value {
                            ParamValueList::Value(v) => Some(v[i]),
                            _ => panic!("Expected plasticity-threshold parameter to be a value list"),
                        };
                    } else {
                        panic!("Unknown tet parameter: {}", name);
                    }
                }
                let model = model.unwrap();
                let density = density.unwrap();
                let young_mod = young_mod.unwrap();
                let poiss_rat = poiss_rat.unwrap();
                let shrink = shrink.unwrap();
                let plasticity = plasticity.unwrap_or(0.0);
                let plasticity_threshold = plasticity_threshold.unwrap_or(0.0);
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
                };
                let param_idx = *tet_param_map.entry(param).or_insert_with(|| {
                    let new_idx = tet_params.len() as u32;
                    tet_params.push(param);
                    new_idx
                });

                TetProp {
                    mass,
                    volume,
                    fixed: false,
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
        }
    }

    pub fn get_initial_velocity(&self) -> Matrix3xX<f32> {
        self.vel.clone()
    }

    pub fn make_constraint(&self, time: f64) -> Constraint {
        let collision_mesh = if self.static_vert.ncols() > 0 {
            let mut vert = self.static_vert.clone();
            for (i, mut x) in vert.column_iter_mut().enumerate() {
                x += self.displacement.column(self.static_vert_dmap[i] as usize);
            }

            // Build face props and params with deduplication
            let mut face_param_map: HashMap<FaceParam, u32> = HashMap::new();
            let mut face_params = Vec::new();
            let face_props = (0..self.static_tri.ncols())
                .map(|i| {
                    let mut contact_gap = None;
                    let mut contact_offset = None;
                    let mut friction = None;
                    for entry in self.static_param.iter() {
                        if entry.0 == "contact-gap" {
                            contact_gap = match entry.1 {
                                ParamValueList::Value(ref v) => Some(v[i]),
                                _ => panic!("Expected contact-gap parameter to be a value list"),
                            };
                        } else if entry.0 == "contact-offset" {
                            contact_offset = match entry.1 {
                                ParamValueList::Value(ref v) => Some(v[i]),
                                _ => panic!("Expected contact-offset parameter to be a value list"),
                            };
                        } else if entry.0 == "friction" {
                            friction = match entry.1 {
                                ParamValueList::Value(ref v) => Some(v[i]),
                                _ => panic!("Expected friction parameter to be a value list"),
                            };
                        }
                    }
                    let x0 = vert.column(self.static_tri.column(i)[0]);
                    let x1 = vert.column(self.static_tri.column(i)[1]);
                    let x2 = vert.column(self.static_tri.column(i)[2]);
                    let r0 = x1 - x0;
                    let r1 = x2 - x0;
                    let area = 0.5 * r0.cross(&r1).norm();
                    let ghat = contact_gap.unwrap();
                    let offset = contact_offset.unwrap();
                    let friction = friction.unwrap();
                    assert_gt!(area, 0.0, "Area of static triangle {} is zero", i);
                    assert_gt!(ghat, 0.0, "Contact gap must be non-negative");

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
                    };
                    let param_idx = *face_param_map.entry(param).or_insert_with(|| {
                        let new_idx = face_params.len() as u32;
                        face_params.push(param);
                        new_idx
                    });

                    FaceProp {
                        area,
                        mass: 0.0,
                        fixed: false,
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
                    if time > last_time {
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
        let mut fix = Vec::new();
        let mut pull = Vec::new();
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
                } => {
                    if time < *t_start {
                        (position, false)
                    } else if time >= *t_end {
                        let delta_vec: Vector3<f32> = delta.column(vert_idx).into();
                        (position + delta_vec, true)
                    } else {
                        let mut progress = (time - t_start) / (t_end - t_start);
                        if transition == "smooth" {
                            progress = progress * progress * (3.0 - 2.0 * progress);
                        }
                        let delta_vec: Vector3<f32> = delta.column(vert_idx).into();
                        (position + delta_vec * progress as f32, true)
                    }
                }
                PinOperation::MoveTo {
                    target,
                    t_start,
                    t_end,
                    transition,
                } => {
                    if time < *t_start {
                        (position, false)
                    } else if time >= *t_end {
                        (target.column(vert_idx).into(), true)
                    } else {
                        let mut progress = (time - t_start) / (t_end - t_start);
                        if transition == "smooth" {
                            progress = progress * progress * (3.0 - 2.0 * progress);
                        }
                        let target_pos: Vector3<f32> = target.column(vert_idx).into();
                        (
                            position * (1.0 - progress) as f32
                                + target_pos * progress as f32,
                            true,
                        )
                    }
                }
                PinOperation::Spin {
                    center,
                    axis,
                    angular_velocity,
                    t_start,
                    t_end,
                } => {
                    let t = time.min(*t_end);
                    if t > *t_start {
                        let angle = *angular_velocity as f64 / 180.0
                            * std::f64::consts::PI
                            * (t - t_start);
                        let axis_norm = axis / axis.norm();
                        let cos_theta = angle.cos();
                        let sin_theta = angle.sin();
                        let p = position - center;
                        let rotated = p * cos_theta as f32
                            + axis_norm.cross(&p) * sin_theta as f32
                            + axis_norm
                                * axis_norm.dot(&p)
                                * (1.0 - cos_theta) as f32;
                        (rotated + center, true)
                    } else {
                        (position, false)
                    }
                }
                PinOperation::Scale {
                    center,
                    factor,
                    t_start,
                    t_end,
                    transition,
                } => {
                    let t = time.min(*t_end);
                    if t > *t_start {
                        let mut progress = (t - t_start) / (t_end - t_start);
                        if transition == "smooth" {
                            progress = progress * progress * (3.0 - 2.0 * progress);
                        }
                        let current_factor = 1.0 + (*factor - 1.0) * progress as f32;
                        let p = position - center;
                        (p * current_factor + center, true)
                    } else {
                        (position, false)
                    }
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
                    // Pick TRS at the current time with slerp on Q and
                    // optional Bezier easing on the segment's progress.
                    let n = times.len();
                    let (t_arr, q_arr, s_arr) = if time <= times[0] {
                        (translations[0], quaternions[0], scales[0])
                    } else if time >= times[n - 1] {
                        (
                            translations[n - 1],
                            quaternions[n - 1],
                            scales[n - 1],
                        )
                    } else {
                        let mut idx = n - 2;
                        for k in 0..n - 1 {
                            if time >= times[k] && time < times[k + 1] {
                                idx = k;
                                break;
                            }
                        }
                        let raw = (time - times[idx]) / (times[idx + 1] - times[idx]);
                        let code = interps.get(idx).copied().unwrap_or(0);
                        let progress = match code {
                            0 => raw,                               // LINEAR
                            1 => bezier_progress(raw, handles[idx]), // BEZIER
                            2 => 0.0,                               // CONSTANT
                            other => panic!(
                                "Unknown transform_keyframes interp code {}",
                                other,
                            ),
                        };
                        let t_a = translations[idx];
                        let t_b = translations[idx + 1];
                        let s_a = scales[idx];
                        let s_b = scales[idx + 1];
                        let t_i = [
                            (1.0 - progress) * t_a[0] + progress * t_b[0],
                            (1.0 - progress) * t_a[1] + progress * t_b[1],
                            (1.0 - progress) * t_a[2] + progress * t_b[2],
                        ];
                        let s_i = [
                            (1.0 - progress) * s_a[0] + progress * s_b[0],
                            (1.0 - progress) * s_a[1] + progress * s_b[1],
                            (1.0 - progress) * s_a[2] + progress * s_b[2],
                        ];
                        let q_i = quat_slerp(
                            quaternions[idx],
                            quaternions[idx + 1],
                            progress,
                        );
                        (t_i, q_i, s_i)
                    };
                    // Rotation matrix from quaternion (w,x,y,z).
                    let (w, x, y, z) = (q_arr[0], q_arr[1], q_arr[2], q_arr[3]);
                    let r = [
                        [
                            1.0 - 2.0 * (y * y + z * z),
                            2.0 * (x * y - w * z),
                            2.0 * (x * z + w * y),
                        ],
                        [
                            2.0 * (x * y + w * z),
                            1.0 - 2.0 * (x * x + z * z),
                            2.0 * (y * z - w * x),
                        ],
                        [
                            2.0 * (x * z - w * y),
                            2.0 * (y * z + w * x),
                            1.0 - 2.0 * (x * x + y * y),
                        ],
                    ];
                    let lv = local.column(vert_idx);
                    let l = [f64::from(lv[0]), f64::from(lv[1]), f64::from(lv[2])];
                    let sl = [s_arr[0] * l[0], s_arr[1] * l[1], s_arr[2] * l[2]];
                    let out = [
                        r[0][0] * sl[0] + r[0][1] * sl[1] + r[0][2] * sl[2]
                            + t_arr[0]
                            - rest_t[0],
                        r[1][0] * sl[0] + r[1][1] * sl[1] + r[1][2] * sl[2]
                            + t_arr[1]
                            - rest_t[1],
                        r[2][0] * sl[0] + r[2][1] * sl[1] + r[2][2] * sl[2]
                            + t_arr[2]
                            - rest_t[2],
                    ];
                    (
                        Vector3::new(
                            out[0] as f32,
                            out[1] as f32,
                            out[2] as f32,
                        ),
                        true,
                    )
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

            // If pin has ONLY torque operations, skip fix/pull — vertices are
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

                if pin.pull_w > 0.0 {
                    pull.push(PullPair {
                        position: position + dx,
                        index: ind as u32,
                        weight: pin.pull_w,
                    });
                } else {
                    fix.push(FixPair {
                        position: position + dx,
                        ghat: self.args.constraint_ghat,
                        index: ind as u32,
                        kinematic,
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
                    index: Vec4u::from_iterator(
                        self.stitch_ind.column(i).iter().map(|&x| x as u32),
                    ),
                    weight: Vec4f::from_iterator(
                        self.stitch_w.column(i).iter().copied(),
                    ),
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
            if wall.timing.len() <= 1 {
                assert_eq!(wall.timing[0], 0.0);
                let position = wall.position.column(0).into();
                floor.push(Floor {
                    ground: position,
                    ghat: wall.contact_gap,
                    friction: wall.friction,
                    thickness: wall.thickness,
                    up: normal,
                    kinematic: false,
                });
            } else {
                let coeff = calc_coefficient(time, &wall.timing, &wall.transition);
                let (j, k) = (coeff.0[0], coeff.0[1]);
                let w = coeff.1;
                let position = wall.position.column(j) * (1.0 - w)
                    + wall.position.column(k) * w;
                floor.push(Floor {
                    ground: position,
                    ghat: wall.contact_gap,
                    friction: wall.friction,
                    thickness: wall.thickness,
                    up: normal,
                    kinematic: true,
                });
            }
        }
        for s in self.sphere.iter() {
            if s.active_duration >= 0.0 && (time as f32) >= s.active_duration {
                continue;
            }
            let reverse = s.inverted;
            let bowl = s.hemisphere;
            if s.timing.len() <= 1 {
                assert_eq!(s.timing[0], 0.0);
                let center = s.center.column(0).into();
                let radius = s.radius[0];
                sphere.push(Sphere {
                    center,
                    ghat: s.contact_gap,
                    friction: s.friction,
                    radius,
                    thickness: s.thickness,
                    bowl,
                    reverse,
                    kinematic: false,
                });
            } else {
                let coeff = calc_coefficient(time, &s.timing, &s.transition);
                let (j, k) = (coeff.0[0], coeff.0[1]);
                let w = coeff.1;
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
                    kinematic: true,
                });
            }
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

    pub fn update_param(&self, _: &SimArgs, mut time: f64, param: &mut ParamSet) {
        for (title, entries) in self.dyn_args.iter() {
            time = time.min(
                entries
                    .iter()
                    .fold(0.0_f64, |max_time, (t, _)| max_time.max(*t)),
            );
            for i in 0..entries.len() - 1 {
                let (t0, v0) = entries[i];
                let (t1, v1) = entries[i + 1];
                if time >= t0 && time <= t1 {
                    let delta_t = t1 - t0;
                    let w = if delta_t > 0.0 {
                        (time - t0) / (t1 - t0)
                    } else {
                        1.0
                    };
                    let val = v0.lerp(v1, w);
                    match (title.as_str(), val) {
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
                    }
                }
            }
        }
    }

    /// Compute velocity overrides that fire during this time step.
    /// Only returns overrides for keyframes crossed in [time, time + dt).
    /// Hard-pinned vertices (strict kinematic `fix` constraints) are excluded;
    /// weak pins (pull) and torque-only pins still receive the override since
    /// those vertices remain dynamic.
    pub fn get_velocity_overrides(&self, time: f64, dt: f32) -> Vec<(Vec<u32>, f32, f32, f32)> {
        use std::collections::HashSet;
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
                        // Mirror the fix/pull/torque_only classification in
                        // the pin-building loop: a pin becomes a strict `fix`
                        // constraint iff it's not torque-only and pull_w == 0.
                        let hard_pinned: HashSet<usize> = self
                            .pin
                            .iter()
                            .filter(|pin| pin.pull_w == 0.0)
                            .filter(|pin| pin.unpin_time.map_or(true, |ut| *t0 < ut))
                            .filter(|pin| {
                                let torque_only = !pin.operations.is_empty()
                                    && pin.operations.iter().all(|op| {
                                        matches!(op, PinOperation::Torque { .. })
                                    });
                                !torque_only
                            })
                            .flat_map(|pin| pin.index.iter().copied())
                            .collect();
                        let indices: Vec<u32> = self.vert_dmap.iter()
                            .enumerate()
                            .filter(|(j, &dm)| dm == dmap_idx && !hard_pinned.contains(j))
                            .map(|(j, _)| j as u32)
                            .collect();
                        if !indices.is_empty() {
                            result.push((indices, vel[0] as f32, vel[1] as f32, vel[2] as f32));
                        }
                    }
                }
            }
        }
        result
    }

    pub fn build_collision_window_table(&self) -> CollisionWindowTable {
        const MAX_WINDOWS: usize = 8;

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
                warn!("collision_window:{} has {} windows, truncating to {}",
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

    pub fn make_mesh(&mut self) -> MeshSet {
        let mut vert = self.vert.clone();
        for (i, mut x) in vert.column_iter_mut().enumerate() {
            x += self.displacement.column(self.vert_dmap[i] as usize);
        }
        let rest_vertex = self.rest_vert.as_ref().map(|rv| {
            let mut rest = rv.clone();
            for (i, mut x) in rest.column_iter_mut().enumerate() {
                if self.rest_vert_mask[i] {
                    x += self.displacement.column(self.vert_dmap[i] as usize);
                } else {
                    x.copy_from(&vert.column(i));
                }
            }
            rest
        });
        MeshSet {
            vertex: vert,
            uv: self.uv.clone(),
            mesh: SimMesh::new(
                self.rod.clone(),
                self.tri.clone(),
                self.tet.clone(),
                self.shell_count,
            ),
            rest_vertex,
        }
    }
}
