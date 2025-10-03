// File: backend.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::data::{Constraint, StepResult};

use super::{builder, mesh::Mesh, BvhSet, DataSet, ParamSet, ProgramArgs, Scene, SimArgs};
use chrono::Local;
use log::*;
use na::{Matrix2x3, Matrix3xX};
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
    fn initialize(data: *const DataSet, param: *const ParamSet) -> bool;
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
    pub uv: Option<Vec<Matrix2x3<f32>>>,
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
    pub fn new(mesh: MeshSet) -> Self {
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

    pub fn load_state(frame: i32, dirpath: &str) -> Self {
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

    fn save_state(
        &self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        _scene: &Scene,
        dataset: &DataSet,
    ) {
        let path_mesh = format!("{}/meshset.bin.gz", program_args.output);
        let path_dataset = format!("{}/dataset.bin.gz", program_args.output);
        info!(">>> saving state started...");
        if !std::path::Path::new(&path_dataset).exists() {
            info!("saving dataset to {}", path_dataset);
            super::save(dataset, path_dataset.as_str());
        }
        if !std::path::Path::new(&path_mesh).exists() {
            info!("saving meshset to {}", path_mesh);
            super::save(&self.mesh, path_mesh.as_str());
        }
        let path_state = format!(
            "{}/state_{}.bin.gz",
            program_args.output, self.state.curr_frame
        );
        info!("saving state to {}...", path_state);
        super::save(&self.state, path_state.as_str());
        super::remove_old_states(program_args, sim_args.keep_states, self.state.curr_frame);
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

    pub fn run(
        &mut self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        dataset: DataSet,
        mut param: ParamSet,
        scene: Scene,
    ) {
        let initialize_finish_path =
            std::path::Path::new(program_args.output.as_str()).join("initialize_finish.txt");
        let finished_path = std::path::Path::new(program_args.output.as_str()).join("finished.txt");
        if finished_path.exists() {
            std::fs::remove_file(finished_path.clone()).unwrap();
        }
        if initialize_finish_path.exists() {
            std::fs::remove_file(initialize_finish_path.clone()).unwrap();
        }
        if unsafe { initialize(&dataset, &param) } {
            write_current_time_to_file(initialize_finish_path.to_str().unwrap()).unwrap();
        } else {
            panic!("failed to initialize backend");
        }
        if program_args.load > 0 && !sim_args.disable_contact {
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

        if !sim_args.disable_contact {
            std::thread::spawn(move || {
                while let Ok((vertex, face, edge)) = task_receiver.recv() {
                    let face = builder::build_bvh(&vertex, Some(&face));
                    let edge = builder::build_bvh(&vertex, Some(&edge));
                    let vertex = builder::build_bvh::<1>(&vertex, None);
                    let _ = result_sender.send(BvhSet { face, edge, vertex });
                }
            });
        }

        let mut task_sent = false;
        loop {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=clocks.current.sm")
                .arg("--format=csv,noheader,nounits")
                .output()
            {
                if let Ok(clock_str) = String::from_utf8(output.stdout) {
                    let clock = clock_str.trim();
                    info!("GPU SM Clock: {} MHz", clock);

                    // Name: SM Clock Speed
                    // Format: list[(vid_time,clock)]
                    // Description:
                    // GPU SM clock speed sampled at each simulation time step.
                    // The format is (time) -> (clock), where time is the simulation time and clock is the SM clock speed.
                    /*== push "clock" ==*/
                    let mut clock_log = OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(format!("{}/data/clock.out", program_args.output).as_str())
                        .unwrap();
                    writeln!(clock_log, "{} {}", self.state.time, clock).unwrap();
                }
            }

            constraint = scene.make_constraint(self.state.time);
            let mut state_saved = false;
            unsafe { update_constraint(&constraint) };
            let new_frame = (self.state.time * sim_args.fps).floor() as i32;
            if new_frame != self.state.curr_frame {
                // Name: Time Per Video Frame
                // Format: list[(vid_time,ms)]
                // Description:
                // Time consumed to compute a single video frame.
                /*== push "time_per_frame" ==*/
                let mut time_per_frame = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(format!("{}/data/time_per_frame.out", program_args.output).as_str())
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
                    .open(format!("{}/data/frame_to_time.out", program_args.output).as_str())
                    .unwrap();
                let curr_time = Instant::now();
                let elapsed_time = curr_time - last_time;
                self.fetch_state(&dataset, &param);
                if !sim_args.disable_contact {
                    if task_sent {
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
                    } else {
                        let n_surface_vert = self.mesh.mesh.mesh.surface_vert_count;
                        let vert: Matrix3xX<f32> = self
                            .state
                            .curr_vertex
                            .columns(0, n_surface_vert)
                            .into_owned();
                        let data = (
                            vert,
                            self.mesh.mesh.mesh.face.clone(),
                            self.mesh.mesh.mesh.edge.clone(),
                        );
                        task_sender.send(data).unwrap();
                        task_sent = true;
                    }
                }
                self.state.curr_frame = new_frame;
                writeln!(time_per_frame, "{} {}", new_frame, elapsed_time.as_millis()).unwrap();
                writeln!(
                    frame_to_time,
                    "{} {}",
                    self.state.curr_frame, self.state.time
                )
                .unwrap();
                let path = format!(
                    "{}/vert_{}.bin.tmp",
                    program_args.output, self.state.curr_frame
                );
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
                super::remove_old_files(
                    &program_args.output,
                    "vert_",
                    ".bin",
                    sim_args.keep_verts,
                    self.state.curr_frame,
                );
                if sim_args.auto_save > 0 && new_frame > 0 && new_frame % sim_args.auto_save == 0 {
                    info!("auto save state...");
                    self.save_state(program_args, sim_args, &scene, &dataset);
                    state_saved = true;
                }
                last_time = Instant::now();
                if self.state.curr_frame == sim_args.fake_crash_frame {
                    panic!("fake crash!");
                }
            }

            let save_and_quit_path =
                std::path::Path::new(program_args.output.as_str()).join("save_and_quit");
            if save_and_quit_path.exists() {
                if !state_saved {
                    info!("save_and_quit file found, saving state...");
                    self.save_state(program_args, sim_args, &scene, &dataset);
                }
                std::fs::remove_file(save_and_quit_path).unwrap_or_else(|err| {
                    println!("Failed to delete 'save_and_quit' file: {}", err);
                });
                break;
            }

            if self.state.curr_frame >= sim_args.frames {
                if !state_saved && sim_args.auto_save > 0 {
                    info!("simulation finished, saving state...");
                    self.save_state(program_args, sim_args, &scene, &dataset);
                } else {
                    info!("simulation finished, not saving state...");
                }
                break;
            }

            scene.update_param(sim_args, self.state.time, &mut param);
            let mut result = StepResult::default();
            unsafe { advance(&mut result) };
            if !result.success() {
                panic!("failed to advance");
            }
            self.state.time = result.time;
        }
        if !sim_args.disable_contact {
            let _ = result_receiver.try_recv();
        }
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
