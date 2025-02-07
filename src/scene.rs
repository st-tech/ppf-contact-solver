// File: scene.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::data::*;
use super::{builder, Args, CVec, MeshSet, ParamSet, SimMesh};
use bytemuck::{cast_slice, Pod};
use na::{Const, Matrix, Matrix2xX, Matrix3xX, Matrix4xX, VecStorage, Vector3};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{self, BufRead, Read};
use toml::Value;

pub struct Scene {
    args: Args,
    dyn_args: Vec<(String, Vec<(f32, f32)>)>,
    vert: Matrix3xX<f32>,
    vel: Matrix3xX<f32>,
    uv: Option<Matrix2xX<f32>>,
    rod: Matrix2xX<usize>,
    tri: Matrix3xX<usize>,
    tet: Matrix4xX<usize>,
    static_vert: Matrix3xX<f32>,
    static_tri: Matrix3xX<usize>,
    stitch_ind: Matrix3xX<usize>,
    stitch_w: Matrix2xX<f32>,
    pin: Vec<Pin>,
    wall: Vec<InvisibleWall>,
    sphere: Vec<InvisibleSphere>,
    shell_count: usize,
}

struct Spin {
    center: Vector3<f32>,
    axis: Vector3<f32>,
    angular_velocity: f32,
    t_start: f32,
    t_end: f32,
}

struct Pin {
    index: Vec<usize>,
    timing: Vec<f32>,
    target: Vec<Matrix3xX<f32>>,
    spin: Vec<Spin>,
    unpin: bool,
    transition: String,
    pull_w: f32,
}

struct InvisibleSphere {
    center: Matrix3xX<f32>,
    radius: Vec<f32>,
    timing: Vec<f32>,
    inverted: bool,
    hemisphere: bool,
    transition: String,
}

struct InvisibleWall {
    normal: Vector3<f32>,
    position: Matrix3xX<f32>,
    timing: Vec<f32>,
    transition: String,
}

#[derive(Debug, Deserialize)]
struct Config {
    param: Args,
}

type MatReadResult<T, const C: usize> =
    io::Result<Matrix<T, Const<C>, na::Dyn, VecStorage<T, Const<C>, na::Dyn>>>;

type DynParamTable = Vec<(String, Vec<(f32, f32)>)>;

fn read_mat_from_file<T, const C: usize>(path: &str) -> MatReadResult<T, C>
where
    T: Pod + std::cmp::PartialEq + std::fmt::Debug,
{
    let mut file = File::open(path)?;
    let mut buff = Vec::new();
    file.read_to_end(&mut buff)?;
    if buff.len() % std::mem::size_of::<T>() != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Data length is not a multiple of the element size",
        ));
    }
    let data: &[T] = cast_slice(&buff);
    if data.len() % C != 0 {
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

fn read_vec<T>(path: &str) -> Vec<T>
where
    T: bytemuck::AnyBitPattern,
{
    if !std::path::Path::new(path).exists() {
        Vec::new()
    } else {
        let mut buff = Vec::new();
        let mut file = File::open(path).unwrap();
        file.read_to_end(&mut buff)
            .unwrap_or_else(|_| panic!("Failed to read {}", path));
        cast_slice(&buff).to_vec()
    }
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
            let pair: Vec<&str> = line.split_whitespace().collect();
            if pair.len() == 2 {
                let time: f32 = pair[0].parse().expect("Failed to parse time");
                let value: f32 = pair[1].parse().expect("Failed to parse value");
                curr_entry.push((time, value));
            }
        }
    }
    if !curr_entry_name.is_empty() {
        result.push((curr_entry_name, curr_entry));
    }
    Ok(result)
}

impl Scene {
    pub fn new(args: &Args) -> Self {
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
        let read_string = |count: &Value, key: &str| {
            count
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("Failed to read {}", key))
                .to_string()
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

        let vert_path = format!("{}/bin/vert.bin", args.path);
        let vel_path = format!("{}/bin/vel.bin", args.path);
        let uv_path = format!("{}/bin/uv.bin", args.path);
        let rod_path = format!("{}/bin/rod.bin", args.path);
        let tri_path = format!("{}/bin/tri.bin", args.path);
        let tet_path = format!("{}/bin/tet.bin", args.path);
        let static_vert_path = format!("{}/bin/static_vert.bin", args.path);
        let static_tri_path = format!("{}/bin/static_tri.bin", args.path);
        let stitch_ind_path = format!("{}/bin/stitch_ind.bin", args.path);
        let stitch_w_path = format!("{}/bin/stitch_w.bin", args.path);

        let vert_mat = read_mat_from_file::<f32, 3>(&vert_path).expect("Failed to read vert");
        let vel_mat = read_mat_from_file::<f32, 3>(&vel_path).expect("Failed to read velocity");
        let uv_mat = if std::path::Path::new(&uv_path).exists() {
            let mat = read_mat_from_file::<f32, 2>(&uv_path).expect("Failed to read uv");
            assert_eq!(mat.ncols(), n_vert as usize);
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
        let static_vert_mat = if n_static_vert > 0 {
            read_mat_from_file::<f32, 3>(&static_vert_path).expect("Failed to read static_vert")
        } else {
            Matrix3xX::<f32>::zeros(0)
        };
        let static_tri_mat = if n_static_tri > 0 {
            read_mat_from_file::<usize, 3>(&static_tri_path).expect("Failed to read static_tri")
        } else {
            Matrix3xX::<usize>::zeros(0)
        };
        let (stitch_ind_mat, stitch_w_mat) = if n_stitch > 0 {
            (
                read_mat_from_file::<usize, 3>(&stitch_ind_path)
                    .expect("Failed to read stitch_ind"),
                read_mat_from_file::<f32, 2>(&stitch_w_path).expect("Failed to read stitch_w"),
            )
        } else {
            (Matrix3xX::<usize>::zeros(0), Matrix2xX::<f32>::zeros(0))
        };

        let mut pin = Vec::new();
        for i in 0..n_pin_block {
            let title = format!("pin-{}", i);
            let count = parsed
                .get(&title)
                .unwrap_or_else(|| panic!("Failed to read pin {}", i));
            let n_pin = read_usize(count, "pin");
            let n_keyframe = read_usize(count, "keyframe");
            let unpin = read_bool(count, "unpin");
            let pull_w = read_f32(count, "pull");
            let transition = read_string(count, "transition");
            let pin_ind_path = format!("{}/bin/pin-ind-{}.bin", args.path, i);
            let pin_dir = format!("{}/bin/pin-{}", args.path, i);

            let mut target = Vec::new();
            if n_keyframe > 0 {
                assert!(std::path::Path::new(&pin_dir).exists());
                for j in 0..n_keyframe {
                    let target_path = format!("{}/{}.bin", pin_dir, j);
                    target.push(
                        read_mat_from_file::<f32, 3>(&target_path).expect("Failed to read target"),
                    );
                }
            }
            let pin_ind = read_vec::<usize>(&pin_ind_path);
            let pin_timing =
                read_vec::<f32>(format!("{}/bin/pin-timing-{}.bin", args.path, i).as_str());
            assert_eq!(pin_ind.len(), n_pin as usize);
            assert_eq!(pin_timing.len(), target.len());

            let n_spin = read_usize(count, "spin");
            let mut spin = Vec::new();
            if n_spin > 0 {
                let spin_path = format!("{}/spin/spin-{}.toml", args.path, i);
                let content = fs::read_to_string(spin_path).expect("Failed to read spin");
                let parsed: Value = content.parse::<Value>().expect("Failed to parse spin");
                for j in 0..n_spin {
                    let spin_title = format!("spin-{}", j);
                    let entry = parsed.get(&spin_title).expect("Failed to read spin");
                    let center_x = read_f32(entry, "center_x");
                    let center_y = read_f32(entry, "center_y");
                    let center_z = read_f32(entry, "center_z");
                    let axis_x = read_f32(entry, "axis_x");
                    let axis_y = read_f32(entry, "axis_y");
                    let axis_z = read_f32(entry, "axis_z");
                    let angular_velocity = read_f32(entry, "angular_velocity");
                    let t_start = read_f32(entry, "t_start");
                    let t_end = read_f32(entry, "t_end");
                    let center = Vector3::new(center_x, center_y, center_z);
                    let axis = Vector3::new(axis_x, axis_y, axis_z);
                    spin.push(Spin {
                        center,
                        axis,
                        angular_velocity,
                        t_start,
                        t_end,
                    });
                }
            }
            pin.push(Pin {
                index: pin_ind,
                timing: pin_timing,
                target,
                unpin,
                transition,
                pull_w,
                spin,
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
                    read_mat_from_file::<f32, 3>(&format!("{}/bin/wall-pos-{}.bin", args.path, i))
                        .expect("Failed to read pos_path");
                let wall_timing =
                    read_vec::<f32>(&format!("{}/bin/wall-timing-{}.bin", args.path, i));
                assert_eq!(position.ncols(), n_keyframe as usize);
                assert_eq!(wall_timing.len(), n_keyframe);
                wall.push(InvisibleWall {
                    normal,
                    position,
                    timing: wall_timing,
                    transition,
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
            let center =
                read_mat_from_file::<f32, 3>(&format!("{}/bin/sphere-pos-{}.bin", args.path, i))
                    .expect("Failed to read sphere pos_path");
            let radius = read_vec::<f32>(&format!("{}/bin/sphere-radius-{}.bin", args.path, i));
            let timing = read_vec::<f32>(&format!("{}/bin/sphere-timing-{}.bin", args.path, i));
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
            });
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
        let mut config: Config = toml::from_str(&file_content).unwrap();

        config.param.path = args.path.clone();
        config.param.output = args.output.clone();

        let dyn_args_path = format!("{}/dyn_param.txt", args.path);
        let dyn_args = if std::path::Path::new(&dyn_args_path).exists() {
            read_dyn_param(&dyn_args_path).unwrap()
        } else {
            Vec::new()
        };

        Self {
            args: config.param,
            dyn_args,
            vert: vert_mat,
            vel: vel_mat,
            uv: uv_mat,
            rod: rod_mat,
            tri: tri_mat,
            tet: tet_mat,
            static_vert: static_vert_mat,
            static_tri: static_tri_mat,
            stitch_ind: stitch_ind_mat,
            stitch_w: stitch_w_mat,
            pin,
            wall,
            sphere,
            shell_count,
        }
    }

    pub fn override_args(&self, args: &mut Args) {
        *args = self.args.clone();
    }

    pub fn get_initial_velocity(&self, _: &Args, _: usize) -> Matrix3xX<f32> {
        self.vel.clone()
    }

    pub fn make_constraint(&self, _: &Args, time: f64, _: &MeshSet) -> Constraint {
        let collision_mesh = if self.static_vert.ncols() > 0 {
            builder::make_collision_mesh(&self.static_vert, &self.static_tri)
        } else {
            CollisionMesh::new()
        };
        let calc_coefficient =
            |time: f64, timings: &[f32], transition: &str| -> ([usize; 2], f32) {
                if timings.is_empty() {
                    ([0, 0], 1.0)
                } else {
                    let last_time = timings[timings.len() - 1] as f64;
                    if time > last_time {
                        ([timings.len() - 1, timings.len() - 1], 1.0)
                    } else {
                        for i in 0..timings.len() - 1 {
                            let t0 = timings[i] as f64;
                            let t1 = timings[i + 1] as f64;
                            if time >= t0 && time < t1 {
                                let mut w: f32 = (time - t0) as f32 / (t1 - t0) as f32;
                                if transition == "smooth" {
                                    w = w * w * (3.0 - 2.0 * w);
                                }
                                return ([i, i + 1], w);
                            }
                        }
                        panic!("Failed to calculate coefficient")
                    }
                }
            };
        let mut fix = Vec::new();
        let mut pull = Vec::new();
        for pin in self.pin.iter() {
            for (i, &ind) in pin.index.iter().enumerate() {
                let (mut kinematic, mut position) = {
                    if pin.timing.len() <= 1 {
                        let position = Some(self.vert.column(ind).into());
                        (false, position)
                    } else {
                        assert_eq!(pin.timing.len(), pin.target.len());
                        let last_time = pin.timing[pin.timing.len() - 1] as f64;
                        if time > last_time && pin.unpin {
                            (true, None)
                        } else {
                            let coeff = calc_coefficient(time, &pin.timing, &pin.transition);
                            let (j, k) = (coeff.0[0], coeff.0[1]);
                            let w = coeff.1;
                            let p0: Vector3<f32> = pin.target[j].column(i).into();
                            let p1: Vector3<f32> = pin.target[k].column(i).into();
                            (true, Some(p0 * (1.0 - w) + p1 * w))
                        }
                    }
                };
                let time = time as f32;
                for spin in pin.spin.iter() {
                    let time = time.min(spin.t_end);
                    if time > spin.t_start {
                        let angle = spin.angular_velocity / 180.0
                            * std::f32::consts::PI
                            * (time - spin.t_start);
                        let axis = spin.axis / spin.axis.norm();
                        let cos_theta = angle.cos();
                        let sin_theta = angle.sin();
                        let p = position.unwrap_or_else(|| self.vert.column(ind).into());
                        let p = p - spin.center;
                        let rotated = p * cos_theta
                            + axis.cross(&p) * sin_theta
                            + axis * axis.dot(&p) * (1.0 - cos_theta);
                        position = Some(rotated + spin.center);
                        kinematic = true;
                    }
                }
                if let Some(position) = position {
                    if pin.pull_w > 0.0 {
                        pull.push(PullPair {
                            position,
                            index: ind as u32,
                            weight: pin.pull_w,
                        });
                    } else {
                        fix.push(FixPair {
                            position,
                            index: ind as u32,
                            kinematic,
                        });
                    }
                }
            }
        }
        let stitch = {
            let mut stitch = Vec::new();
            for i in 0..self.stitch_ind.ncols() {
                stitch.push(Stitch {
                    index: Vec3u::from_iterator(
                        self.stitch_ind.column(i).iter().map(|&x| x as u32),
                    ),
                    weight: self.stitch_w.column(i)[1],
                    active: true,
                });
            }
            stitch
        };
        let mut floor = Vec::new();
        let mut sphere = Vec::new();
        for wall in self.wall.iter() {
            let normal = wall.normal;
            if wall.timing.len() <= 1 {
                assert_eq!(wall.timing[0], 0.0);
                let position = wall.position.column(0).into();
                floor.push(Floor {
                    ground: position,
                    up: normal,
                    kinematic: false,
                });
            } else {
                let coeff = calc_coefficient(time, &wall.timing, &wall.transition);
                let (j, k) = (coeff.0[0], coeff.0[1]);
                let w = coeff.1;
                let position = wall.position.column(j) * (1.0 - w) + wall.position.column(k) * w;
                floor.push(Floor {
                    ground: position,
                    up: normal,
                    kinematic: true,
                });
            }
        }
        for s in self.sphere.iter() {
            let reverse = s.inverted;
            let bowl = s.hemisphere;
            if s.timing.len() <= 1 {
                assert_eq!(s.timing[0], 0.0);
                let center = s.center.column(0).into();
                let radius = s.radius[0];
                sphere.push(Sphere {
                    center,
                    radius,
                    bowl,
                    reverse,
                    kinematic: false,
                });
            } else {
                let coeff = calc_coefficient(time, &s.timing, &s.transition);
                let (j, k) = (coeff.0[0], coeff.0[1]);
                let w = coeff.1;
                let center = s.center.column(j) * (1.0 - w) + s.center.column(k) * w;
                let radius = s.radius[j] * (1.0 - w) + s.radius[k] * w;
                sphere.push(Sphere {
                    center,
                    radius,
                    bowl,
                    reverse,
                    kinematic: true,
                });
            }
        }
        Constraint {
            fix: CVec::from(&fix[..]),
            pull: CVec::from(&pull[..]),
            sphere: CVec::from(&sphere[..]),
            floor: CVec::from(&floor[..]),
            stitch: CVec::from(&stitch[..]),
            mesh: collision_mesh,
        }
    }

    pub fn update_param(&self, _: &Args, time: f64, param: &mut ParamSet) {
        let mut time = time as f32;
        for (title, entries) in self.dyn_args.iter() {
            let mut max_time = 0.0_f32;
            for (t, _) in entries.iter() {
                max_time = max_time.max(*t);
            }
            if time > max_time {
                time = max_time;
            }
            for i in 0..entries.len() - 1 {
                let (t0, v0) = entries[i];
                let (t1, v1) = entries[i + 1];
                if time >= t0 && time <= t1 {
                    let w = (time - t0) / (t1 - t0);
                    let val = v0 * (1.0 - w) + v1 * w;
                    match title.as_str() {
                        "gravity" => param.gravity = Vec3f::new(0.0, val, 0.0),
                        "dt" => param.dt = val,
                        "playback" => param.playback = val,
                        "friction" => param.friction = val,
                        _ => (),
                    }
                }
            }
        }
    }

    pub fn make_mesh(&mut self, _: &Args) -> MeshSet {
        MeshSet {
            vertex: self.vert.clone(),
            uv: self.uv.clone(),
            mesh: SimMesh::new(
                self.rod.clone(),
                self.tri.clone(),
                self.tet.clone(),
                self.shell_count,
            ),
        }
    }
}
