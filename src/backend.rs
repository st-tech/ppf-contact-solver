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

#[cfg(not(feature = "emulated"))]
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

// Emulated build: every CUDA-bound call becomes a no-op. The real
// per-frame work happens in Backend::apply_kinematic_constraint, which
// directly reads the Constraint and writes vert_*.bin without going
// through these. ``allow(dead_code)`` silences the dead-code warning
// for the stubs that aren't reached on every code path.
#[cfg(feature = "emulated")]
#[allow(dead_code)]
mod cuda_stubs {
    use super::{Constraint, DataSet, IntersectionRecord, ParamSet, StepResult};
    pub unsafe fn advance(result: *mut StepResult) { (*result).ccd_success = true; (*result).pcg_success = true; (*result).intersection_free = true; }
    pub unsafe fn fetch() {}
    pub unsafe fn fetch_inv_rest() {}
    pub unsafe fn fetch_rest_angles() {}
    pub unsafe fn fetch_dyn_counts(n_value: *mut u32, n_offset: *mut u32) { *n_value = 0; *n_offset = 0; }
    pub unsafe fn fetch_dyn(_index: *mut u32, _value: *mut f32, _offset: *mut u32) {}
    pub unsafe fn update_dyn(_index: *const u32, _offset: *const u32) {}
    pub unsafe fn update_constraint(_constraint: *const Constraint) {}
    pub unsafe fn initialize(_data: *const DataSet, _param: *const ParamSet) -> bool { true }
    pub unsafe fn override_velocity(_i: *const u32, _c: u32, _vx: f32, _vy: f32, _vz: f32, _dt: f32) {}
    pub unsafe fn init_collision_windows(_d: *const u32, _c: u32, _w: *const f32, _wc: *const u32, _n: u32) {}
    pub unsafe fn refresh_collision_active(_t: f32) {}
    // Drains the synthetic-record buffer populated by the
    // ``PPF_EMULATED_FAIL_AT_FRAME`` knob. Returns 0 unless the knob
    // fired this run.
    pub unsafe fn fetch_intersection_records(out: *mut IntersectionRecord, max_count: u32) -> u32 {
        super::emulated_intersection::drain(out, max_count)
    }
}

#[cfg(feature = "emulated")]
use cuda_stubs::*;

// Emulator-only knob plumbing for ``PPF_EMULATED_FAIL_AT_FRAME``.
// Caches the parsed env var on first read and exposes a synthetic-
// record buffer that the ``fetch_intersection_records`` stub drains.
// Production builds skip this whole module.
#[cfg(feature = "emulated")]
mod emulated_intersection {
    use super::{IntersectionRecord, MAX_INTERSECTION_RECORDS};
    use std::sync::{Mutex, OnceLock};

    static FAIL_AT_FRAME: OnceLock<Option<i32>> = OnceLock::new();
    static SYNTHETIC: OnceLock<Mutex<Vec<IntersectionRecord>>> = OnceLock::new();

    fn buffer() -> &'static Mutex<Vec<IntersectionRecord>> {
        SYNTHETIC.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Parse ``PPF_EMULATED_FAIL_AT_FRAME`` once. Returns ``Some(n)``
    /// when the env var is set to a non-negative integer.
    pub fn fail_at_frame() -> Option<i32> {
        *FAIL_AT_FRAME.get_or_init(|| {
            std::env::var("PPF_EMULATED_FAIL_AT_FRAME")
                .ok()
                .and_then(|v| v.trim().parse::<i32>().ok())
                .filter(|n| *n >= 0)
        })
    }

    /// Build one face-edge, one edge-edge, and one collision-mesh
    /// record with deterministic positions. The Rust-side
    /// ``write_intersection_records`` reads ``itype``, ``elem0``,
    /// ``elem1``, ``num_verts0``, ``num_verts1``, and the packed
    /// ``positions`` array, so each record fills all of those.
    pub fn seed_synthetic_records() {
        let mut buf = buffer().lock().unwrap();
        if !buf.is_empty() {
            return;
        }
        // itype 0: face-edge (3-vert face, 2-vert edge).
        let mut face_edge = IntersectionRecord::default();
        face_edge.itype = 0;
        face_edge.elem0 = 11;
        face_edge.elem1 = 22;
        face_edge.num_verts0 = 3;
        face_edge.num_verts1 = 2;
        face_edge.positions = [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.5, 0.5, -0.1,
            0.5, 0.5, 0.1,
        ];
        buf.push(face_edge);

        // itype 1: edge-edge (2-vert edge, 2-vert edge).
        let mut edge_edge = IntersectionRecord::default();
        edge_edge.itype = 1;
        edge_edge.elem0 = 33;
        edge_edge.elem1 = 44;
        edge_edge.num_verts0 = 2;
        edge_edge.num_verts1 = 2;
        edge_edge.positions = [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.5, -0.5, 0.0,
            0.5, 0.5, 0.0,
            0.0, 0.0, 0.0,
        ];
        buf.push(edge_edge);

        // itype 2: collision-mesh (3-vert face, 2-vert edge). The
        // production recorder packs at most five vec3 positions
        // (face vs edge), so collision-mesh shares the (3, 2) shape
        // with face-edge.
        let mut collision_mesh = IntersectionRecord::default();
        collision_mesh.itype = 2;
        collision_mesh.elem0 = 55;
        collision_mesh.elem1 = 66;
        collision_mesh.num_verts0 = 3;
        collision_mesh.num_verts1 = 2;
        collision_mesh.positions = [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.5, 0.5, -0.2,
            0.5, 0.5, 0.2,
        ];
        buf.push(collision_mesh);
    }

    /// Copies up to ``max_count`` records into ``out``, draining the
    /// buffer so a follow-up ``fetch_intersection_records`` call
    /// returns 0. Mirrors the production CUDA path's "read once"
    /// semantics.
    pub unsafe fn drain(out: *mut IntersectionRecord, max_count: u32) -> u32 {
        let mut buf = buffer().lock().unwrap();
        let cap = (max_count as usize).min(MAX_INTERSECTION_RECORDS);
        let n = buf.len().min(cap);
        for (i, rec) in buf.drain(..n).enumerate() {
            *out.add(i) = rec;
        }
        n as u32
    }
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
        // In emulated mode the FFI does nothing; ``state.curr_vertex``
        // is already the source of truth, populated by
        // ``apply_kinematic_constraint`` each step. Skip the unsafe
        // copy from a never-updated ``dataset.vertex.curr``.
        #[cfg(feature = "emulated")]
        {
            self.state.prev_dt = param.prev_dt;
            return;
        }
        #[cfg(not(feature = "emulated"))]
        {
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
    }

    /// Emulated-mode kinematic step. Pinned (kinematic) vertices land
    /// exactly where the constraint says; everyone else stays put.
    /// This is the load-bearing piece for the pin-animation fidelity
    /// test: the production flow's CUDA solver also drives kinematic
    /// pins to the same target, so a Rust binary built with
    /// ``--features emulated`` produces a vert_*.bin trajectory that
    /// matches the addon's encoder math vertex-for-vertex.
    #[cfg(feature = "emulated")]
    pub fn apply_kinematic_constraint(&mut self, constraint: &Constraint) {
        let n = constraint.fix.size as usize;
        if n == 0 {
            return;
        }
        let pairs = unsafe {
            std::slice::from_raw_parts(constraint.fix.data as *const super::data::FixPair, n)
        };
        let total = self.state.curr_vertex.ncols();
        for pair in pairs {
            if !pair.kinematic {
                continue;
            }
            let idx = pair.index as usize;
            if idx >= total {
                continue;
            }
            let mut col = self.state.curr_vertex.column_mut(idx);
            col[0] = pair.position[0];
            col[1] = pair.position[1];
            col[2] = pair.position[2];
        }
    }

    pub fn load_state(frame: i32, dirpath: &str) -> Self {
        let (mesh, state) = {
            let path_mesh = format!("{}/meshset.bin.gz", dirpath);
            let path_state = format!("{}/state_{}.bin.gz", dirpath, frame);
            let mesh = super::read(&super::read_gz(path_mesh.as_str()));
            let state = super::read(&super::read_gz(path_state.as_str()));
            (mesh, state)
        };
        Self { state, mesh }
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
        info!("saving dataset to {}", path_dataset);
        super::save(dataset, path_dataset.as_str());
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
        loop {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=clocks.current.sm")
                .arg("--format=csv,noheader,nounits")
                .output()
            {
                if let Ok(clock_str) = String::from_utf8(output.stdout) {
                    let clock = clock_str.trim();
                    info!("GPU SM Clock: {} MHz", clock);

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

            constraint = scene.make_constraint(self.state.time);
            let mut state_saved = false;
            unsafe { update_constraint(&constraint) };
            // Emulated mode: apply the constraint's kinematic positions
            // directly so the next vert_*.bin write reflects the pin
            // trajectory. Production CUDA does this implicitly via
            // update_constraint -> advance.
            #[cfg(feature = "emulated")]
            self.apply_kinematic_constraint(&constraint);
            let new_frame = (self.state.time * sim_args.fps).floor() as i32;
            if new_frame != self.state.curr_frame {
                // Name: Time Per Video Frame
                // Format: list[(frame, ms)]
                // Description:
                // Wall-clock time in milliseconds elapsed between producing
                // the previous output video frame and this one. Because
                // each simulation step can advance less than one frame's
                // worth of time, this aggregates however many solver steps
                // were needed to cross the frame boundary.
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
                // simulation time each output frame corresponds to, which
                // is useful when the playback rate varies over time.
                /*== push "frame_to_time" ==*/
                let mut frame_to_time = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(format!("{}/data/frame_to_time.out", program_args.output).as_str())
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
                let path = format!(
                    "{}/vert_{}.bin.tmp",
                    program_args.output, self.state.curr_frame
                );
                let file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path.clone())
                    .unwrap();
                let mut writer = std::io::BufWriter::with_capacity(64 * 1024, file);
                let surface_vert_count = self.mesh.mesh.mesh.vertex_count;
                // Write vertices in chunks to avoid large RAM allocation
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
                // Write remaining data
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
            for (indices, vx, vy, vz) in scene.get_velocity_overrides(self.state.time, param.dt) {
                unsafe {
                    override_velocity(indices.as_ptr(), indices.len() as u32, vx, vy, vz, param.dt);
                }
            }
            unsafe { refresh_collision_active(self.state.time as f32) };
            let mut result = StepResult::default();
            unsafe { advance(&mut result) };
            // Emulator-only fault injection. When
            // ``PPF_EMULATED_FAIL_AT_FRAME=N`` is set, the first
            // advance after frame N has been written
            // flips intersection_free to false and seeds three
            // synthetic IntersectionRecords (one face-edge, one
            // edge-edge, one collision-mesh) so the existing
            // ``write_intersection_records`` flow drains them into
            // ``output/intersection_records.json``.
            #[cfg(feature = "emulated")]
            {
                if let Some(n) = emulated_intersection::fail_at_frame() {
                    if self.state.curr_frame >= n {
                        emulated_intersection::seed_synthetic_records();
                        result.intersection_free = false;
                    }
                }
            }
            if !result.success() {
                write_intersection_records(&program_args.output);
                panic!("failed to advance");
            }
            // Emulated build: ``advance`` is a no-op stub that leaves
            // result.time at 0, so we'd loop forever on the time test
            // below. Bump time manually by dt; CUDA-side production
            // does this inside the kernel.
            //
            // We also sleep one wall-clock second per step by default so
            // the run paces like a real simulation -- the test rig polls
            // intermediate frames, state transitions, and probe events
            // while the solver is BUSY, and an instant run skips through
            // every BUILDING/RUNNING/SAVING transition before the
            // monitor's 0.25s tick can ever observe them. Override with
            // ``PPF_EMULATED_STEP_MS`` (e.g. 100 for fast tests, 0 for
            // unit tests).
            #[cfg(feature = "emulated")]
            {
                self.state.time += param.dt as f64;
                let step_ms: u64 = std::env::var("PPF_EMULATED_STEP_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1000);
                if step_ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(step_ms));
                }
            }
            #[cfg(not(feature = "emulated"))]
            { self.state.time = result.time; }
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
    let path = format!("{}/intersection_records.json", output_dir);
    if let Ok(file) = std::fs::File::create(&path) {
        let _ = serde_json::to_writer_pretty(file, &json);
        info!("wrote {} intersection records to {}", count, path);
    } else {
        error!("failed to create {}", path);
    }
}
