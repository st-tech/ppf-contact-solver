// File: backend.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::data::Constraint;
use super::data::StepResult;
use super::{builder, mesh::Mesh, Args, BvhSet, DataSet, ParamSet, Scene};
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

extern crate nalgebra as na;

use log::*;
use na::{Matrix2xX, Matrix3xX};

extern "C" {
    fn advance() -> StepResult;
    fn fetch();
    fn update_bvh(bvhset: *const BvhSet);
    fn fetch_dyn_counts(n_value: *mut u32, n_offset: *mut u32);
    fn fetch_dyn(index: *mut u32, value: *mut f32, offset: *mut u32);
    fn update_constraint(constraint: *const Constraint);
    fn initialize(data: *const DataSet, param: *const ParamSet);
}

#[derive(Serialize, Deserialize)]
pub struct Backend {
    pub mesh: MeshSet,
    pub state: State,
    pub bvh: Box<Option<BvhSet>>,
}

#[derive(Serialize, Deserialize)]
pub struct MeshSet {
    pub mesh: Mesh,
    pub uv: Option<Matrix2xX<f32>>,
    pub vertex: Matrix3xX<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct State {
    pub curr_vertex: Matrix3xX<f32>,
    pub prev_vertex: Matrix3xX<f32>,
    pub dyn_index: Vec<u32>,
    pub dyn_offset: Vec<u32>,
    pub time: f64,
    pub prev_dt: f32,
    pub curr_frame: i32,
}

impl Backend {
    pub fn new(_: &Args, mesh: MeshSet) -> Self {
        info!("initializing backend");
        let state = State {
            curr_vertex: mesh.vertex.clone(),
            prev_vertex: mesh.vertex.clone(),
            dyn_index: Vec::new(),
            dyn_offset: Vec::new(),
            time: f64::EPSILON,
            prev_dt: 1.0,
            curr_frame: -1,
        };
        let num_face = mesh.mesh.mesh.face.ncols();
        let num_tet = mesh.mesh.mesh.tet.ncols();
        info!(
            "#v = {}, #f = {}, #tet = {}",
            mesh.mesh.mesh.vertex_count, num_face, num_tet
        );
        Self {
            state,
            mesh,
            bvh: Box::new(None),
        }
    }

    fn fetch_state(&mut self, dataset: &DataSet, param: &ParamSet) {
        unsafe {
            fetch();
        }
        let prev_vertex = unsafe {
            Matrix3xX::from_column_slice(std::slice::from_raw_parts(
                dataset.vertex.prev.data as *const f32,
                3 * dataset.vertex.prev.size as usize,
            ))
        };
        let curr_vertex = unsafe {
            Matrix3xX::from_column_slice(std::slice::from_raw_parts(
                dataset.vertex.curr.data as *const f32,
                3 * dataset.vertex.curr.size as usize,
            ))
        };
        let prev_dt = param.prev_dt;
        let mut n_value = 0;
        let mut n_offset = 0;
        unsafe {
            fetch_dyn_counts(&mut n_value, &mut n_offset);
        }
        self.state.dyn_index.resize(n_value as usize, 0);
        self.state.dyn_offset.resize(n_offset as usize, 0);
        unsafe {
            fetch_dyn(
                self.state.dyn_index.as_mut_ptr(),
                std::ptr::null_mut(),
                self.state.dyn_offset.as_mut_ptr(),
            );
        }
        self.state.prev_vertex = prev_vertex;
        self.state.curr_vertex = curr_vertex;
        self.state.prev_dt = prev_dt;
    }

    pub fn run(&mut self, args: &Args, dataset: DataSet, mut param: ParamSet, scene: Scene) {
        let finished_path = std::path::Path::new(args.output.as_str()).join("finished.txt");
        if finished_path.exists() {
            std::fs::remove_file(finished_path.clone()).unwrap();
        }
        unsafe {
            initialize(&dataset, &param);
        }
        let mut last_time = Instant::now();
        let mut constraint;

        let (task_sender, task_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();

        std::thread::spawn(move || {
            while let Ok((vertex, face, edge)) = task_receiver.recv() {
                let face = builder::build_bvh(&vertex, Some(&face));
                let edge = builder::build_bvh(&vertex, Some(&edge));
                let vertex = builder::build_bvh::<1>(&vertex, None);
                result_sender.send(BvhSet { face, edge, vertex }).unwrap();
            }
        });

        let mut first_step = true;
        loop {
            constraint = scene.make_constraint(args, self.state.time, &self.mesh);
            unsafe { update_constraint(&constraint) };
            if !first_step {
                match result_receiver.try_recv() {
                    Ok(bvh) => {
                        log::info!("bvh update");
                        self.bvh = Box::new(Some(bvh));
                        unsafe {
                            update_bvh(self.bvh.as_ref().as_ref().unwrap());
                        }
                        let data = (
                            self.state.curr_vertex.clone(),
                            self.mesh.mesh.mesh.face.clone(),
                            self.mesh.mesh.mesh.edge.clone(),
                        );
                        task_sender.send(data).unwrap();
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        log::info!("bvh still building...");
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        panic!("bvh thread disconnected");
                    }
                }
            }
            let new_frame = (self.state.time * args.fps).floor() as i32;
            if new_frame != self.state.curr_frame {
                let mut per_video_frame = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(format!("{}/data/per_video_frame.out", args.output).as_str())
                    .unwrap();
                let mut per_video_time = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(format!("{}/data/per_video_time.out", args.output).as_str())
                    .unwrap();
                let curr_time = Instant::now();
                let elapsed_time = curr_time - last_time;
                self.fetch_state(&dataset, &param);
                self.state.curr_frame = new_frame;
                writeln!(
                    per_video_frame,
                    "{} {}",
                    new_frame,
                    elapsed_time.as_millis()
                )
                .unwrap();
                writeln!(
                    per_video_time,
                    "{} {}",
                    self.state.curr_frame, self.state.time
                )
                .unwrap();
                let path = format!("{}/vert_{}.bin.tmp", args.output, self.state.curr_frame);
                let mut file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path.clone())
                    .unwrap();
                let data = self.state.curr_vertex.as_slice();
                let buff = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        std::mem::size_of_val(data),
                    )
                };
                file.write_all(buff).unwrap();
                file.flush().unwrap();
                std::fs::rename(path.clone(), path.replace(".tmp", "")).unwrap();
                last_time = Instant::now();
            }
            if self.state.curr_frame >= args.frames {
                break;
            }
            scene.update_param(args, self.state.time, &mut param);
            let result = unsafe { advance() };
            if !result.success() {
                panic!("failed to advance");
            }
            self.state.time = result.time;
            info!(
                "frame = {} time = {:.4} seconds",
                self.state.curr_frame, self.state.time
            );
            if first_step {
                let data = (
                    self.state.curr_vertex.clone(),
                    self.mesh.mesh.mesh.face.clone(),
                    self.mesh.mesh.mesh.edge.clone(),
                );
                task_sender.send(data).unwrap();
                first_step = false;
            }
        }
        let _ = result_receiver.try_recv();
        write_current_time_to_file(finished_path.to_str().unwrap()).unwrap();
    }
}

fn write_current_time_to_file(file_path: &str) -> std::io::Result<()> {
    let now = Local::now();
    let time_str = now.to_rfc3339(); // For example, "2024-12-23T12:34:56+00:00"
    let path = Path::new(file_path);
    let mut file = OpenOptions::new()
        .create(true) // Create the file if it doesn't exist
        .append(true) // Append to the file if it exists
        .open(path)?;
    writeln!(file, "{}", time_str)?;
    Ok(())
}
