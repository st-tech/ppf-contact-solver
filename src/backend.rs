// File: backend.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::data::Constraint;
use super::data::StepResult;
use super::{builder, mesh::Mesh, Args, BvhSet, DataSet, ParamSet, Scene};
use chrono::Local;
use log::*;
use na::{Matrix2xX, Matrix3xX};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

extern "C" {
    fn advance(result: *mut StepResult);
    fn fetch();
    fn update_bvh(bvhset: *const BvhSet);
    fn fetch_dyn_counts(n_value: *mut u32, n_offset: *mut u32);
    fn fetch_dyn(index: *mut u32, value: *mut f32, offset: *mut u32);
    fn update_dyn(index: *const u32, offset: *const u32);
    fn update_constraint(constraint: *const Constraint);
    fn initialize(data: *const DataSet, param: *const ParamSet, use_thrust: bool);
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
        let state = State {
            curr_vertex: mesh.vertex.clone(),
            prev_vertex: mesh.vertex.clone(),
            dyn_index: Vec::new(),
            dyn_offset: Vec::new(),
            time: 0.0,
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

    pub fn load_state(_: &Args, frame: i32, dirpath: &str) -> Self {
        let (mesh, state) = {
            let path_mesh = format!("{}/meshset.bin.gz", dirpath);
            let path_state = format!("{}/state_{}.bin.gz", dirpath, frame);
            let mesh = super::read(&super::read_gz(path_mesh.as_str()));
            let state = super::read(&super::read_gz(path_state.as_str()));
            (mesh, state)
        };
        Self {
            state,
            mesh,
            bvh: Box::new(None),
        }
    }

    fn save_state(&self, args: &Args, _scene: &Scene, dataset: &DataSet) {
        let path_mesh = format!("{}/meshset.bin.gz", args.output);
        let path_dataset = format!("{}/dataset.bin.gz", args.output);
        info!(">>> saving state started...");
        if !std::path::Path::new(&path_dataset).exists() {
            info!("saving dataset to {}", path_dataset);
            super::save(dataset, path_dataset.as_str());
        }
        if !std::path::Path::new(&path_mesh).exists() {
            info!("saving meshset to {}", path_mesh);
            super::save(&self.mesh, path_mesh.as_str());
        }
        let path_state = format!("{}/state_{}.bin.gz", args.output, self.state.curr_frame);
        info!("saving state to {}...", path_state);
        super::save(&self.state, path_state.as_str());
        super::remove_old_states(args, self.state.curr_frame);
        info!("<<< save state done.");
    }

    fn update_bvh(&mut self) {
        log::info!("building bvh...");
        let n_surface_vert = self.mesh.mesh.mesh.surface_vert_count;
        let vert: Matrix3xX<f32> = self
            .state
            .curr_vertex
            .columns(0, n_surface_vert)
            .into_owned();
        self.bvh = Box::new(Some(BvhSet {
            face: builder::build_bvh(&vert, Some(&self.mesh.mesh.mesh.face)),
            edge: builder::build_bvh(&vert, Some(&self.mesh.mesh.mesh.edge)),
            vertex: builder::build_bvh::<1>(&vert, None),
        }));
        unsafe {
            update_bvh(self.bvh.as_ref().as_ref().unwrap());
        }
    }

    pub fn run(&mut self, args: &Args, dataset: DataSet, mut param: ParamSet, scene: Scene) {
        let finished_path = std::path::Path::new(args.output.as_str()).join("finished.txt");
        if finished_path.exists() {
            std::fs::remove_file(finished_path.clone()).unwrap();
        }
        unsafe {
            initialize(&dataset, &param, args.use_thrust);
        }
        if args.load > 0 {
            self.update_bvh();
            unsafe {
                update_dyn(
                    self.state.dyn_index.as_ptr(),
                    self.state.dyn_offset.as_ptr(),
                );
            }
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
                let _ = result_sender.send(BvhSet { face, edge, vertex });
            }
        });

        let mut first_step = true;
        loop {
            constraint = scene.make_constraint(args, self.state.time, &self.mesh);
            let mut state_saved = false;
            unsafe { update_constraint(&constraint) };
            if !first_step {
                match result_receiver.try_recv() {
                    Ok(bvh) => {
                        info!("bvh update...");
                        let n_surface_vert = self.mesh.mesh.mesh.surface_vert_count;
                        let vert: Matrix3xX<f32> = self
                            .state
                            .curr_vertex
                            .columns(0, n_surface_vert)
                            .into_owned();
                        self.bvh = Box::new(Some(bvh));
                        unsafe {
                            update_bvh(self.bvh.as_ref().as_ref().unwrap());
                        }
                        let data = (
                            vert,
                            self.mesh.mesh.mesh.face.clone(),
                            self.mesh.mesh.mesh.edge.clone(),
                        );
                        task_sender.send(data).unwrap();
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        panic!("bvh thread disconnected");
                    }
                }
            }
            let new_frame = (self.state.time * args.fps).floor() as i32;
            if new_frame != self.state.curr_frame {
                // Name: Time Per Video Frame
                // Format: list[(vid_time,ms)]
                // Description:
                // Time consumed to compute a single video frame.
                /*== push "time_per_frame" ==*/
                let mut time_per_frame = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(format!("{}/data/time_per_frame.out", args.output).as_str())
                    .unwrap();
                // Name: Mapping of Video Frame to Simulation Time
                // Format: list[(int,ms)]
                // Description:
                // This file contains a list of pairs encoding the mapping of video frame to the simulation time.
                // The format is (frame) -> (ms), where frame is the video frame number and ms is the time of the simulation in milliseconds.
                /*== push "frame_to_time" ==*/
                let mut frame_to_time = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(format!("{}/data/frame_to_time.out", args.output).as_str())
                    .unwrap();
                let curr_time = Instant::now();
                let elapsed_time = curr_time - last_time;
                self.fetch_state(&dataset, &param);
                self.state.curr_frame = new_frame;
                writeln!(time_per_frame, "{} {}", new_frame, elapsed_time.as_millis()).unwrap();
                writeln!(
                    frame_to_time,
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
                let surface_vert_count = self.mesh.mesh.mesh.vertex_count;
                let data: Vec<f32> = self
                    .state
                    .curr_vertex
                    .columns(0, surface_vert_count)
                    .iter()
                    .copied()
                    .collect();
                let buff = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<f32>(),
                    )
                };
                file.write_all(buff).unwrap();
                file.flush().unwrap();
                std::fs::rename(path.clone(), path.replace(".tmp", "")).unwrap();
                if args.auto_save > 0 && new_frame > 0 && new_frame % args.auto_save == 0 {
                    info!("auto save state...");
                    self.save_state(args, &scene, &dataset);
                    state_saved = true;
                }
                last_time = Instant::now();
                if self.state.curr_frame == args.fake_crash_frame {
                    panic!("fake crash!");
                }
            }

            let save_and_quit_path =
                std::path::Path::new(args.output.as_str()).join("save_and_quit");
            if save_and_quit_path.exists() {
                if !state_saved {
                    info!("save_and_quit file found, saving state...");
                    self.save_state(args, &scene, &dataset);
                }
                std::fs::remove_file(save_and_quit_path).unwrap_or_else(|err| {
                    println!("Failed to delete 'save_and_quit' file: {}", err);
                });
                break;
            }

            if self.state.curr_frame >= args.frames {
                if !state_saved && args.auto_save > 0 {
                    info!("simulation finished, saving state...");
                    self.save_state(args, &scene, &dataset);
                } else {
                    info!("simulation finished, not saving state...");
                }
                break;
            }

            scene.update_param(args, self.state.time, &mut param);
            let mut result = StepResult::default();
            unsafe { advance(&mut result) };
            if !result.success() {
                panic!("failed to advance");
            }
            self.state.time = result.time;
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
    let time_str = now.to_rfc3339();
    let path = Path::new(file_path);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", time_str)?;
    Ok(())
}
