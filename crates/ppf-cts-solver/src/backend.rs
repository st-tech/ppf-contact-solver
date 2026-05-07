// File: backend.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::data::{Constraint, IntersectionRecord, StepResult, MAX_INTERSECTION_RECORDS};
use super::{mesh::Mesh, DataSet, ParamSet, ProgramArgs, Scene, SimArgs};
use chrono::Local;
use log::*;
use na::{Matrix2x3, Matrix3xX};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

// The C++ side picks the implementation: libsimbackend_cuda for the
// CUDA path, libsimbackend_cpu (the emulator) when --features emulated
// flips the link in build.rs. Both libraries expose the identical
// extern "C" surface, so this declaration is unconditional.
extern "C" {
    fn advance(result: *mut StepResult);
    fn fetch();
    fn fetch_inv_rest();
    fn fetch_rest_angles();
    fn fetch_dyn_counts(n_value: *mut u32, n_offset: *mut u32);
    fn fetch_dyn(index: *mut u32, value: *mut f32, offset: *mut u32);
    fn update_dyn(index: *const u32, offset: *const u32);
    fn update_constraint(constraint: *const Constraint);
    fn initialize(data: *const DataSet, param: *const ParamSet) -> bool;
    fn override_velocity(indices: *const u32, count: u32, vx: f32, vy: f32, vz: f32, dt: f32);
    fn init_collision_windows(
        vert_dmap: *const u32, vert_count: u32,
        windows: *const f32, window_counts: *const u32,
        n_groups: u32,
    );
    fn refresh_collision_active(time: f32);
    fn fetch_intersection_records(out: *mut IntersectionRecord, max_count: u32) -> u32;
}

#[derive(Serialize, Deserialize)]
pub struct Backend {
    pub mesh: MeshSet,
    pub state: State,
}

#[derive(Serialize, Deserialize)]
pub struct MeshSet {
    pub mesh: Mesh,
    pub uv: Option<Vec<Matrix2x3<f32>>>,
    pub vertex: Matrix3xX<f32>,
    pub rest_vertex: Option<Matrix3xX<f32>>,
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
        Self { state, mesh }
    }

    fn fetch_state(&mut self, dataset: &DataSet, param: &ParamSet) {
        unsafe {
            fetch();
            fetch_inv_rest();
            fetch_rest_angles();
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
            let path_mesh = format!("{dirpath}/meshset.bin.gz");
            let path_state = format!("{dirpath}/state_{frame}.bin.gz");
            let mesh = super::read(&super::read_gz(path_mesh.as_str()));
            let state = super::read(&super::read_gz(path_state.as_str()));
            (mesh, state)
        };
        Self { state, mesh }
    }

    /// Write `vert_<n>.bin`, `time_per_frame.out`, and
    /// `frame_to_time.out` for one finalized frame and run the optional
    /// auto-save side-effects. Extracted from the per-iteration loop in
    /// `run` so the rest pose (frame 0) and post-advance frames share
    /// the same write path.
    ///
    /// `time` is the sim time recorded for this frame; on the
    /// post-advance path it equals `self.state.time` after the step
    /// that produced the frame. `state_already_saved` is true when the
    /// caller (a save_and_quit branch) has already emitted a state
    /// checkpoint and the auto-save block here should skip the
    /// duplicate write.
    fn write_frame_outputs(
        &mut self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        scene: &Scene,
        dataset: &DataSet,
        param: &ParamSet,
        last_time: &mut Instant,
        frame: i32,
        time: f64,
        state_already_saved: bool,
    ) {
        // Name: Time Per Video Frame
        // Format: list[(frame, ms)]
        // Description:
        // Wall-clock time in milliseconds elapsed between producing
        // the previous output video frame and this one. Because each
        // simulation step can advance less than one frame's worth of
        // time, this aggregates however many solver steps were needed
        // to cross the frame boundary.
        /*== push "time_per_frame" ==*/
        let mut time_per_frame = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}/data/time_per_frame.out", program_args.output).as_str())
            .unwrap();
        // Name: Video Frame to Simulation Time Map
        // Format: list[(frame, seconds)]
        // Description:
        // Pairs of (video frame index, simulation time in seconds)
        // emitted once per output frame. Lets you look up which
        // simulation time each output frame corresponds to, which is
        // useful when the playback rate varies over time.
        /*== push "frame_to_time" ==*/
        let mut frame_to_time = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}/data/frame_to_time.out", program_args.output).as_str())
            .unwrap();
        let curr_time = Instant::now();
        let elapsed = curr_time - *last_time;
        self.fetch_state(dataset, param);
        writeln!(time_per_frame, "{} {}", frame, elapsed.as_millis()).unwrap();
        writeln!(frame_to_time, "{} {}", frame, time).unwrap();
        let path = format!("{}/vert_{}.bin.tmp", program_args.output, frame);
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.clone())
            .unwrap();
        let mut writer = std::io::BufWriter::with_capacity(64 * 1024, file);
        let surface_vert_count = self.mesh.mesh.mesh.vertex_count;
        // Write vertices in chunks to avoid large RAM allocation.
        const CHUNK_SIZE: usize = 4096;
        let mut chunk_buf: Vec<f32> = Vec::with_capacity(CHUNK_SIZE);
        for v in self
            .state
            .curr_vertex
            .columns(0, surface_vert_count)
            .iter()
        {
            chunk_buf.push(*v);
            if chunk_buf.len() >= CHUNK_SIZE {
                let buff = unsafe {
                    std::slice::from_raw_parts(
                        chunk_buf.as_ptr() as *const u8,
                        chunk_buf.len() * std::mem::size_of::<f32>(),
                    )
                };
                writer.write_all(buff).unwrap();
                chunk_buf.clear();
            }
        }
        if !chunk_buf.is_empty() {
            let buff = unsafe {
                std::slice::from_raw_parts(
                    chunk_buf.as_ptr() as *const u8,
                    chunk_buf.len() * std::mem::size_of::<f32>(),
                )
            };
            writer.write_all(buff).unwrap();
        }
        writer.flush().unwrap();
        std::fs::rename(path.clone(), path.replace(".tmp", "")).unwrap();
        super::remove_old_files(
            &program_args.output,
            "vert_",
            ".bin",
            sim_args.keep_verts,
            frame,
        );
        if !state_already_saved
            && sim_args.auto_save > 0
            && frame > 0
            && frame % sim_args.auto_save == 0
        {
            info!("auto save state...");
            self.save_state(program_args, sim_args, scene, dataset);
        }
        *last_time = Instant::now();
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
        info!("saving dataset to {path_dataset}");
        super::save(dataset, path_dataset.as_str());
        if !std::path::Path::new(&path_mesh).exists() {
            info!("saving meshset to {path_mesh}");
            super::save(&self.mesh, path_mesh.as_str());
        }
        let path_state = format!(
            "{}/state_{}.bin.gz",
            program_args.output, self.state.curr_frame
        );
        info!("saving state to {path_state}...");
        super::save(&self.state, path_state.as_str());
        super::remove_old_states(program_args, sim_args.keep_states, self.state.curr_frame);
        info!("<<< save state done.");
    }

    pub fn run(
        &mut self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        dataset: DataSet,
        mut param: ParamSet,
        scene: Scene,
    ) {
        let initialize_finish_path = std::path::Path::new(program_args.output.as_str())
            .join(ppf_cts_formats::files::INITIALIZE_FINISH);
        let finished_path = std::path::Path::new(program_args.output.as_str())
            .join(ppf_cts_formats::files::FINISHED);
        if finished_path.exists() {
            std::fs::remove_file(finished_path.clone()).unwrap();
        }
        if initialize_finish_path.exists() {
            std::fs::remove_file(initialize_finish_path.clone()).unwrap();
        }
        if unsafe { initialize(&dataset, &param) } {
            // Initialize collision windows on GPU
            let cw = scene.build_collision_window_table();
            info!("collision windows: has_windows={}, n_groups={}, vert_dmap_len={}, window_counts={:?}",
                  cw.has_windows, cw.n_groups, cw.vert_dmap.len(),
                  if cw.has_windows { &cw.window_counts[..cw.n_groups.min(20) as usize] } else { &[] });
            if cw.has_windows {
                unsafe {
                    init_collision_windows(
                        cw.vert_dmap.as_ptr(), cw.vert_dmap.len() as u32,
                        cw.windows.as_ptr(), cw.window_counts.as_ptr(),
                        cw.n_groups,
                    );
                }
            }
            write_current_time_to_file(initialize_finish_path.to_str().unwrap()).unwrap();
        } else {
            panic!("failed to initialize backend");
        }
        if program_args.load > 0 && !sim_args.disable_contact {
            unsafe {
                update_dyn(
                    self.state.dyn_index.as_ptr(),
                    self.state.dyn_offset.as_ptr(),
                );
            }
        }
        let mut last_time = Instant::now();
        let mut constraint;
        // Frame 0 is the rest pose at sim time 0. We write it once
        // before the loop so the loop body has a single concern: each
        // iteration prepares the *target* pin positions for the next
        // advance, runs the step, and writes vert_n.bin only after the
        // step completes — meaning pinned and free vertices in the
        // same file always reflect the same sim time.
        //
        // The previous flow set the pin constraint to pin_at(state.time)
        // (start of step) and then wrote vert_n.bin BEFORE the next
        // advance. Step n's pinned positions therefore came from
        // make_constraint at the previous iteration's start time, while
        // the free vertices in the same file had been integrated to
        // state.time. The two halves of the file disagreed by one step,
        // surfacing as the bl_pin_* fidelity divergence.
        if self.state.curr_frame < 0 && self.state.time == 0.0 {
            self.write_frame_outputs(
                program_args, sim_args, &scene, &dataset, &param,
                &mut last_time, 0, 0.0, false,
            );
            self.state.curr_frame = 0;
        }
        loop {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=clocks.current.sm")
                .arg("--format=csv,noheader,nounits")
                .output()
            {
                if let Ok(clock_str) = String::from_utf8(output.stdout) {
                    let clock = clock_str.trim();
                    info!("GPU SM Clock: {clock} MHz");

                    // Name: GPU SM Clock Speed
                    // Format: list[(time, MHz)]
                    // Description:
                    // GPU streaming-multiprocessor clock speed in MHz, sampled
                    // from nvidia-smi once per simulation step. Useful for
                    // correlating slow steps with thermal or power-related
                    // clock throttling on the device.
                    /*== push "clock" ==*/
                    let mut clock_log = OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(format!("{}/data/clock.out", program_args.output).as_str())
                        .unwrap();
                    writeln!(clock_log, "{} {}", self.state.time, clock).unwrap();
                }
            }

            // save_and_quit / frames-done checks happen at the top now;
            // the previous flow placed them after the frame-write
            // block, but with the write moved post-advance these need
            // to gate the *upcoming* step instead.
            let save_and_quit_path =
                std::path::Path::new(program_args.output.as_str()).join("save_and_quit");
            if save_and_quit_path.exists() {
                info!("save_and_quit file found, saving state...");
                self.save_state(program_args, sim_args, &scene, &dataset);
                std::fs::remove_file(save_and_quit_path).unwrap_or_else(|err| {
                    println!("Failed to delete 'save_and_quit' file: {err}");
                });
                break;
            }

            if self.state.curr_frame >= sim_args.frames {
                if sim_args.auto_save > 0 {
                    info!("simulation finished, saving state...");
                    self.save_state(program_args, sim_args, &scene, &dataset);
                } else {
                    info!("simulation finished, not saving state...");
                }
                break;
            }

            // Pin target = end-of-step time. The implicit integrator
            // pulls pinned vertices toward this constraint during the
            // step, so picking pin_at(state.time + dt) means the
            // post-advance positions match what frontend.FixedScene.
            // time(state.time + dt) computes, vertex for vertex. The
            // previous wiring used state.time (start of step) as the
            // target which left every pin one step behind by the end
            // of the step.
            let target_time = self.state.time + param.dt as f64;
            constraint = scene.make_constraint(target_time);
            unsafe { update_constraint(&constraint) };

            scene.update_param(sim_args, self.state.time, &mut param);
            for (indices, vx, vy, vz) in scene.get_velocity_overrides(self.state.time, param.dt) {
                unsafe {
                    override_velocity(indices.as_ptr(), indices.len() as u32, vx, vy, vz, param.dt);
                }
            }
            unsafe { refresh_collision_active(self.state.time as f32) };
            let mut result = StepResult::default();
            unsafe { advance(&mut result) };
            if !result.success() {
                write_intersection_records(&program_args.output);
                panic!("failed to advance");
            }
            self.state.time = result.time;

            // Frame detection now happens AFTER advance: the post-step
            // state is what gets recorded, with pin and free verts
            // both at sim time = self.state.time.
            let new_frame = (self.state.time * sim_args.fps).floor() as i32;
            if new_frame != self.state.curr_frame {
                self.write_frame_outputs(
                    program_args, sim_args, &scene, &dataset, &param,
                    &mut last_time, new_frame, self.state.time,
                    false,
                );
                self.state.curr_frame = new_frame;
                if self.state.curr_frame == sim_args.fake_crash_frame {
                    panic!("fake crash!");
                }
            }
        }
        write_current_time_to_file(finished_path.to_str().unwrap()).unwrap();
    }
}

fn write_current_time_to_file(file_path: &str) -> std::io::Result<()> {
    let now = Local::now();
    let time_str = now.to_rfc3339();
    let path = Path::new(file_path);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{time_str}")?;
    Ok(())
}

fn write_intersection_records(output_dir: &str) {
    let mut records = vec![IntersectionRecord::default(); MAX_INTERSECTION_RECORDS];
    let count =
        unsafe { fetch_intersection_records(records.as_mut_ptr(), MAX_INTERSECTION_RECORDS as u32) };
    if count == 0 {
        return;
    }
    records.truncate(count as usize);
    let type_name = |t: u32| match t {
        0 => "face_edge",
        1 => "edge_edge",
        2 => "collision_mesh",
        _ => "unknown",
    };
    let json_records: Vec<serde_json::Value> = records
        .iter()
        .map(|r| {
            let n0 = r.num_verts0 as usize;
            let n1 = r.num_verts1 as usize;
            let mut pos0 = Vec::new();
            for i in 0..n0 {
                pos0.push(vec![
                    r.positions[i * 3],
                    r.positions[i * 3 + 1],
                    r.positions[i * 3 + 2],
                ]);
            }
            let mut pos1 = Vec::new();
            for i in 0..n1 {
                let base = n0 * 3 + i * 3;
                pos1.push(vec![
                    r.positions[base],
                    r.positions[base + 1],
                    r.positions[base + 2],
                ]);
            }
            serde_json::json!({
                "type": type_name(r.itype),
                "elem0": r.elem0,
                "elem1": r.elem1,
                "positions0": pos0,
                "positions1": pos1,
            })
        })
        .collect();
    let json = serde_json::json!({
        "count": count,
        "records": json_records,
    });
    let path = format!("{output_dir}/intersection_records.json");
    if let Ok(file) = std::fs::File::create(&path) {
        let _ = serde_json::to_writer_pretty(file, &json);
        info!("wrote {count} intersection records to {path}");
    } else {
        error!("failed to create {path}");
    }
}
