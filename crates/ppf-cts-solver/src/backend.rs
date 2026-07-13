// File: backend.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::cvec::CVec;
use super::data::{
    Constraint, IntersectionRecord, Mat2x2f, Mat3x3f, RestShapeUpdate, StepResult,
    MAX_INTERSECTION_RECORDS,
};
use super::{mesh::Mesh, DataSet, ParamSet, ProgramArgs, Scene, SimArgs};
use ppf_cts_formats::status::{crash_kind_from_step, Outcome, Phase};
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
    fn update_rest_shape(update: *const RestShapeUpdate);
    fn initialize(data: *const DataSet, param: *const ParamSet) -> bool;
    fn override_velocity(indices: *const u32, count: u32, vx: f32, vy: f32, vz: f32, dt: f32);
    fn gather_current_positions(indices: *const u32, count: u32, out: *mut f32);
    #[allow(clippy::too_many_arguments)]
    fn override_angular_velocity(
        indices: *const u32,
        count: u32,
        wx: f32,
        wy: f32,
        wz: f32,
        cx: f32,
        cy: f32,
        cz: f32,
        dt: f32,
    );
    fn init_collision_windows(
        vert_dmap: *const u32, vert_count: u32,
        windows: *const f32, window_counts: *const u32,
        n_groups: u32,
    );
    fn refresh_collision_active(time: f32);
    fn fetch_intersection_records(out: *mut IntersectionRecord, max_count: u32) -> u32;
}

/// One precomputed rest-shape keyframe: `(time, inv_rest2x2, inv_rest3x3,
/// exclude_face, exclude_tet)`. The exclude masks (1 = element is near-singular
/// at this keyframe and dropped from the energy) come from
/// `builder::compute_inv_rest`.
type RestShapeKeyframe = (f64, Vec<Mat2x2f>, Vec<Mat3x3f>, Vec<u8>, Vec<u8>);

/// Linearly interpolate a precomputed inverse-rest keyframe schedule to
/// `time`, component-wise on the matrices, and pack the result for upload via
/// `update_rest_shape`. Times outside the schedule clamp to the nearest end.
/// Endpoints reproduce `builder::compute_inv_rest` exactly (alpha 0). An
/// element is excluded if it is near-singular in *either* bracketing keyframe
/// (OR of the masks), so a element collapsing between two samples is dropped
/// for the whole interval (its interpolated inv_rest is unused while excluded).
fn interp_rest_shape(keyframes: &[RestShapeKeyframe], time: f64) -> RestShapeUpdate {
    debug_assert!(
        keyframes.windows(2).all(|w| w[0].0 < w[1].0),
        "rest-shape keyframe times must be strictly increasing"
    );
    let n = keyframes.len();
    let (i0, i1, alpha) = if time <= keyframes[0].0 {
        (0, 0, 0.0f32)
    } else if time >= keyframes[n - 1].0 {
        (n - 1, n - 1, 0.0f32)
    } else {
        // Keyframe times are strictly increasing (the producer sorts a set of
        // boundary times), so the bracketing pair is found with a binary search
        // instead of a linear scan. In this interior branch
        // keyframes[0].0 < time < keyframes[n-1].0, so partition_point returns
        // hi in 1..=n-1, keeping lo and keyframes[hi] in bounds.
        let hi = keyframes.partition_point(|k| k.0 < time);
        let lo = hi - 1;
        let (t0, t1) = (keyframes[lo].0, keyframes[hi].0);
        let a = if t1 > t0 {
            ((time - t0) / (t1 - t0)) as f32
        } else {
            0.0
        };
        (lo, hi, a)
    };
    let inv2: Vec<Mat2x2f> = keyframes[i0]
        .1
        .iter()
        .zip(keyframes[i1].1.iter())
        .map(|(x, y)| *x + (*y - *x) * alpha)
        .collect();
    let inv3: Vec<Mat3x3f> = keyframes[i0]
        .2
        .iter()
        .zip(keyframes[i1].2.iter())
        .map(|(x, y)| *x + (*y - *x) * alpha)
        .collect();
    let exclude_face: Vec<u8> = keyframes[i0]
        .3
        .iter()
        .zip(keyframes[i1].3.iter())
        .map(|(a, b)| a | b)
        .collect();
    let exclude_tet: Vec<u8> = keyframes[i0]
        .4
        .iter()
        .zip(keyframes[i1].4.iter())
        .map(|(a, b)| a | b)
        .collect();
    RestShapeUpdate {
        inv_rest2x2: CVec::from(&inv2[..]),
        inv_rest3x3: CVec::from(&inv3[..]),
        exclude_face: CVec::from(&exclude_face[..]),
        exclude_tet: CVec::from(&exclude_tet[..]),
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
    /// Per-object bending reference rest shape: hinge rest angles for the
    /// objects marked in `bend_rest_vertex_mask` are computed from these
    /// positions instead of the initial `vertex`. Unmasked columns equal the
    /// initial vert, so it is a no-op for non-reference objects. `None` when
    /// no object uses a reference rest angle.
    #[serde(default)]
    pub bend_rest_vertex: Option<Matrix3xX<f32>>,
    /// Per-vertex mask (length = vertex count, or empty when
    /// `bend_rest_vertex` is `None`) marking vertices whose hinges take their
    /// rest angle from `bend_rest_vertex`.
    #[serde(default)]
    pub bend_rest_vertex_mask: Vec<bool>,
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
        // Per-frame readback only needs vertex.prev/vertex.curr (fetch())
        // plus the dyn arrays below. The inv_rest / rest-angle DtoH copies
        // are serialized solely by save_state, so they live there now to
        // keep this per-frame write path off that extra bandwidth.
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
            let path_mesh = format!("{dirpath}/meshset.bin.gz");
            let path_state = format!("{dirpath}/{}", ppf_cts_formats::files::state_filename(frame));
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
    /// The frame is emitted at the EXACT frame time `t_frame` (= `frame /
    /// fps`), not at the variable, TOI-limited post-step time the solver
    /// happened to land on. The step that crossed this frame boundary ran
    /// from `t_prev` to `t_curr`, bracketing it, so the written vertex
    /// positions are linearly interpolated between `prev_vertex` and
    /// `curr_vertex` to `t_frame`. This keeps each output frame on the
    /// scene's framerate base instead of drifting by up to one substep,
    /// which otherwise shows up as a sawtooth in relative frame time. For
    /// the rest pose (frame 0) the caller passes `t_prev == t_curr`, which
    /// collapses the interpolation to `curr_vertex`.
    fn write_frame_outputs(
        &mut self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        dataset: &DataSet,
        param: &ParamSet,
        last_time: &mut Instant,
        frame: i32,
        t_frame: f64,
        t_prev: f64,
        t_curr: f64,
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
        // Stamp the frame this fetched state belongs to before any
        // save_state below. The run loop only advances self.state.curr_frame
        // to the batch's final frame *after* write_frame_outputs returns, so
        // without this an auto-save or checkpoint here would name its file
        // state_<frame - 1> (the previous frame), surfacing as an off-by-one
        // in the Resume-From dialog. The loop's frame range is built before
        // this call, so updating curr_frame here does not affect iteration.
        self.state.curr_frame = frame;
        writeln!(time_per_frame, "{} {}", frame, elapsed.as_millis()).unwrap();
        writeln!(frame_to_time, "{} {}", frame, t_frame).unwrap();
        let path = format!(
            "{}/{}.tmp",
            program_args.output,
            ppf_cts_formats::files::vert_filename(frame)
        );
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.clone())
            .unwrap();
        let mut writer = std::io::BufWriter::with_capacity(super::IO_BUF_CAPACITY, file);
        // A faceless point cloud (e.g. a SAND grain set) derives no surface
        // vertices from elements, so vertex_count is 0. Fall back to the full
        // cloud width so the loose verts still reach vert_<i>.bin.
        let surface_vert_count = self
            .mesh
            .mesh
            .mesh
            .vertex_count
            .max(self.mesh.vertex.ncols());
        // Undo the world-scaling applied to all input geometry on ingest
        // (scene.rs), so the written positions are at the user's authored scale.
        // A 1.0 factor (the default) makes this a no-op.
        let inv_world = if sim_args.world_scaling > 0.0 {
            1.0 / sim_args.world_scaling
        } else {
            1.0
        };
        // Linear-interpolation weight that lands the written positions on the
        // exact frame time. `prev_vertex` is the pose at `t_prev` (step start),
        // `curr_vertex` the pose at `t_curr` (step end, which overshoots the
        // boundary); alpha places the output at `t_frame` in between. A zero
        // span (frame 0 rest pose, or a degenerate step) writes `curr_vertex`
        // verbatim. This is output-only: `self.state.*_vertex` stay as the true
        // GPU readback so `save_state` checkpoints the real solver state.
        let alpha = frame_interp_alpha(t_frame, t_prev, t_curr);
        // Write vertices in chunks to avoid large RAM allocation.
        const CHUNK_SIZE: usize = 4096;
        let mut chunk_buf: Vec<f32> = Vec::with_capacity(CHUNK_SIZE);
        for (p, c) in self
            .state
            .prev_vertex
            .columns(0, surface_vert_count)
            .iter()
            .zip(self.state.curr_vertex.columns(0, surface_vert_count).iter())
        {
            let pf = *p;
            let cf = *c;
            chunk_buf.push((pf + alpha * (cf - pf)) * inv_world);
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
        // Build the final path explicitly rather than stripping ".tmp" with
        // String::replace, which would also rewrite any ".tmp" inside the
        // output directory path. This mirrors the temp-path construction above
        // with the same {output}/vert_{frame} stem.
        let final_path = format!(
            "{}/{}",
            program_args.output,
            ppf_cts_formats::files::vert_filename(frame)
        );
        std::fs::rename(&path, &final_path).unwrap();
        super::remove_old_files(
            &program_args.output,
            ppf_cts_formats::files::VERT_PREFIX,
            ppf_cts_formats::files::VERT_SUFFIX,
            sim_args.keep_verts,
            frame,
        );
        // Save a resumable state when this frame hits the auto-save cadence
        // or appears in the explicit checkpoint list. Both use the same
        // `frame` convention so a checkpoint saves the exact frame the
        // artist requested, and a frame matched by both saves only once.
        //
        // Auto-save counts the 1-based output frame: `frame` is solver 0-based
        // with frame 0 the rest pose, and the UI displays it as frame+1, so an
        // interval N must land on the displayed frames N, 2N, 3N (solver frames
        // N-1, 2N-1, ...). Using `(frame + 1) % N == 0` makes the cadence match
        // what the user sees and the Blender->solver checkpoint conversion
        // (display N -> solver N-1).
        let is_auto_save =
            sim_args.auto_save > 0 && (frame + 1) % sim_args.auto_save == 0;
        let is_checkpoint = sim_args
            .checkpoints
            .split(',')
            .filter_map(|t| t.trim().parse::<i32>().ok())
            .any(|c| c == frame);
        if frame > 0 && (is_auto_save || is_checkpoint) {
            info!("saving state (auto_save={is_auto_save}, checkpoint={is_checkpoint})...");
            self.save_state(program_args, sim_args, dataset);
        }
        *last_time = Instant::now();
    }

    fn save_state(
        &self,
        program_args: &ProgramArgs,
        sim_args: &SimArgs,
        dataset: &DataSet,
    ) {
        let path_mesh = format!("{}/meshset.bin.gz", program_args.output);
        let path_dataset = format!("{}/dataset.bin.gz", program_args.output);
        info!(">>> saving state started...");
        // Pull the evolving plastic rest state (inv_rest matrices and the
        // hinge/vertex rest angles) DtoH right before serialization so the
        // dataset carries the current rest pose. The top-of-loop save_and_quit
        // and finished sites run with no advance since the last fetch_state, so
        // the device rest state matches the host buffers; the auto-save site
        // does one extra identical copy after fetch_state, which is harmless.
        unsafe {
            fetch_inv_rest();
            fetch_rest_angles();
        }
        info!("saving dataset to {path_dataset}");
        super::save(dataset, path_dataset.as_str());
        if !std::path::Path::new(&path_mesh).exists() {
            info!("saving meshset to {path_mesh}");
            super::save(&self.mesh, path_mesh.as_str());
        }
        let path_state = format!(
            "{}/{}",
            program_args.output,
            ppf_cts_formats::files::state_filename(self.state.curr_frame)
        );
        info!("saving state to {path_state}...");
        super::save(&self.state, path_state.as_str());
        super::remove_old_files(
            &program_args.output,
            ppf_cts_formats::files::STATE_PREFIX,
            ppf_cts_formats::files::STATE_SUFFIX,
            sim_args.keep_states,
            self.state.curr_frame,
        );
        // A resumable checkpoint now exists on disk; reflect it in the
        // status record's `resumable` flag.
        crate::status_writer::note_saved();
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
            crate::status_writer::progress(
                Phase::Initialized,
                self.state.curr_frame.max(0),
                self.state.time,
            );
        } else {
            // initialize() == false: an init failure (e.g. an intersection
            // in the initial configuration). The panic below stamps the
            // terminal Crashed{Panic} via the hook; a dedicated
            // Crashed{InitIntersection} from an init error_code arrives with
            // the C++ changes in the next increment.
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
                program_args, sim_args, &dataset, &param,
                &mut last_time, 0, 0.0, 0.0, 0.0,
            );
            self.state.curr_frame = 0;
            crate::status_writer::progress(Phase::Running, 0, self.state.time);
        }
        // Precompute the per-keyframe inverse rest matrices for a streamed
        // time-varying rest shape (the captured pull-pin deformation). Reuses
        // the exact build-time formula (builder::compute_inv_rest) so init and
        // animated rest shapes stay identical, then interpolates between
        // keyframes per step. Empty (no per-step upload) when the scene has no
        // schedule.
        let rest_shape_keyframes: Vec<RestShapeKeyframe> =
            match scene.rest_shape_schedule() {
                Some((times, frames)) => {
                    let face_props = dataset.prop.face.as_slice();
                    let tet_props = dataset.prop.tet.as_slice();
                    let face_params = dataset.param_arrays.face.as_slice();
                    let tet_params = dataset.param_arrays.tet.as_slice();
                    times
                        .into_iter()
                        .zip(frames.iter())
                        .map(|(t, rest_v)| {
                            // exclude_singular = true: a folded captured frame can
                            // drive a rest element near-singular; rather than
                            // clamp (which fattens it and hides the bend's
                            // stretch), flag it so update_rest_shape drops it from
                            // the energy. Every non-singular element keeps its
                            // exact rest.
                            let (inv2, inv3, ef, et) = super::builder::compute_inv_rest(
                                rest_v,
                                &self.mesh,
                                face_props,
                                tet_props,
                                face_params,
                                tet_params,
                                true,
                            );
                            (t, inv2, inv3, ef, et)
                        })
                        .collect()
                }
                None => Vec::new(),
            };
        loop {
            // GPU clock sampling is only meaningful on the CUDA build. The
            // emulated/CPU backend has no device, so gating this out avoids a
            // per-step nvidia-smi subprocess spawn (or failed spawn) on that
            // path. The CUDA build's logging is unchanged.
            #[cfg(not(feature = "emulated"))]
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
                crate::status_writer::progress(
                    Phase::Saving,
                    self.state.curr_frame.max(0),
                    self.state.time,
                );
                self.save_state(program_args, sim_args, &dataset);
                std::fs::remove_file(save_and_quit_path).unwrap_or_else(|err| {
                    println!("Failed to delete 'save_and_quit' file: {err}");
                });
                crate::status_writer::terminal(Outcome::SavedAndQuit);
                break;
            }

            if self.state.curr_frame >= sim_args.frames {
                if sim_args.auto_save > 0 || sim_args.save_state_on_finish {
                    info!("simulation finished, saving state...");
                    self.save_state(program_args, sim_args, &dataset);
                } else {
                    info!("simulation finished, not saving state...");
                }
                crate::status_writer::terminal(Outcome::Finished);
                break;
            }

            // Interpolate dynamic params (including a keyframed dt) for this
            // step BEFORE deriving the step's target time. update_param looks
            // up self.state.time (the step START), which is the correct sample
            // point for the dt that governs this step, and it touches neither
            // the pin constraint nor the rest shape. Doing it first means
            // make_constraint, interp_rest_shape, and advance all use the same,
            // freshly-interpolated param.dt; otherwise a dt keyframe would land
            // one step late, targeting pins/rest at the previous step's dt.
            scene.update_param(sim_args, self.state.time, &mut param);

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

            // Stream the time-varying rest shape for this step's target time,
            // right after the pin constraint (same per-frame upload pattern).
            // The elastic kernels re-read inv_rest each Newton iteration, so
            // this drives each element's rest pose toward the captured
            // deformation while pull pins guide it and contact still resolves.
            if !rest_shape_keyframes.is_empty() {
                let rest_shape = interp_rest_shape(&rest_shape_keyframes, target_time);
                unsafe { update_rest_shape(&rest_shape) };
            }

            for (indices, vx, vy, vz) in scene.get_velocity_overrides(self.state.time, param.dt) {
                unsafe {
                    override_velocity(indices.as_ptr(), indices.len() as u32, vx, vy, vz, param.dt);
                }
            }
            // Angular velocity overrides: the spin axis is resolved from the
            // body's LIVE geometry each firing (gather current positions, then
            // a principal-axis solve), so it tracks the simulated (rotated /
            // deformed) pose rather than a frame-0 pose. Runs AFTER the linear
            // loop so a keyframe carrying both yields a full rigid-velocity
            // overwrite: prev = curr - (v_lin + ω × (x - c)) * dt.
            for (indices, pca_index, speed) in
                scene.get_angular_velocity_overrides(self.state.time, param.dt)
            {
                let count = indices.len();
                let mut positions = vec![0.0f32; count * 3];
                unsafe {
                    gather_current_positions(
                        indices.as_ptr(),
                        count as u32,
                        positions.as_mut_ptr(),
                    );
                }
                if let Some((c, axis)) = resolve_principal_axis(&positions, pca_index as usize) {
                    let (wx, wy, wz) = (axis[0] * speed, axis[1] * speed, axis[2] * speed);
                    unsafe {
                        override_angular_velocity(
                            indices.as_ptr(),
                            count as u32,
                            wx,
                            wy,
                            wz,
                            c[0],
                            c[1],
                            c[2],
                            param.dt,
                        );
                    }
                }
            }
            // Fixed world-axis angular overrides (World X/Y/Z / Custom): ω is
            // a world-space vector given directly, so no principal-axis solve
            // is needed; only the live centroid is gathered for the ω × (x-c)
            // pivot.
            for (indices, omega) in
                scene.get_angular_velocity_world_overrides(self.state.time, param.dt)
            {
                let count = indices.len();
                let mut positions = vec![0.0f32; count * 3];
                unsafe {
                    gather_current_positions(
                        indices.as_ptr(),
                        count as u32,
                        positions.as_mut_ptr(),
                    );
                }
                if let Some(c) = centroid_of(&positions) {
                    unsafe {
                        override_angular_velocity(
                            indices.as_ptr(),
                            count as u32,
                            omega[0],
                            omega[1],
                            omega[2],
                            c[0],
                            c[1],
                            c[2],
                            param.dt,
                        );
                    }
                }
            }
            unsafe { refresh_collision_active(self.state.time as f32) };
            let mut result = StepResult::default();
            unsafe { advance(&mut result) };
            if !result.success() {
                write_intersection_records(
                    &program_args.output,
                    if sim_args.world_scaling > 0.0 {
                        1.0 / sim_args.world_scaling
                    } else {
                        1.0
                    },
                );
                // Derive the crash sub-kind from the StepResult booleans the
                // backend already returns (no log parsing). Stamp the terminal
                // record before the panic; the panic hook then sees a terminal
                // outcome already present and skips its Crashed{Panic}.
                let kind = crash_kind_from_step(
                    result.ccd_success,
                    result.pcg_success,
                    result.intersection_free,
                );
                crate::status_writer::terminal_crash(
                    kind,
                    format!(
                        "advance failed at frame {} (ccd={}, pcg={}, intersection_free={})",
                        self.state.curr_frame,
                        result.ccd_success,
                        result.pcg_success,
                        result.intersection_free
                    ),
                );
                panic!("failed to advance");
            }
            // Step start/end times bracket this advance. `t_prev` is the time
            // before the step, `t_curr` the (possibly overshot) time after.
            // Both are handed to write_frame_outputs so each crossed frame is
            // interpolated onto its exact frame time rather than snapped to the
            // overshot post-step time (which drifts as a sawtooth in relative
            // frame time when the substep dt does not divide 1/fps).
            let t_prev = self.state.time;
            self.state.time = result.time;
            let t_curr = self.state.time;

            // Frame detection now happens AFTER advance: the post-step
            // state is what gets recorded, with pin and free verts
            // both at sim time = self.state.time.
            let new_frame = (self.state.time * sim_args.fps).floor() as i32;
            if new_frame > self.state.curr_frame {
                // A single advance can cross more than one frame boundary when
                // the timestep spans several frame periods (dt * fps >= 2).
                // Emit every integer frame in the crossed span so the output
                // stays contiguous (downstream readers index frames
                // sequentially); each crossed index is interpolated to its own
                // exact frame time t_frame = f / fps within [t_prev, t_curr].
                for f in self.state.curr_frame + 1..=new_frame {
                    let t_frame = f as f64 / sim_args.fps;
                    self.write_frame_outputs(
                        program_args, sim_args, &dataset, &param,
                        &mut last_time, f, t_frame, t_prev, t_curr,
                    );
                }
                self.state.curr_frame = new_frame;
                crate::status_writer::progress(Phase::Running, new_frame, self.state.time);
                // Fire the crash injection once on the final frame, preserving
                // the existing per-advance crash semantics.
                if self.state.curr_frame == sim_args.fake_crash_frame {
                    panic!("fake crash!");
                }
            }
        }
        write_current_time_to_file(finished_path.to_str().unwrap()).unwrap();
    }
}

/// Interpolation weight that places an output frame at its exact frame time.
///
/// `t_prev` is the sim time whose pose lives in `prev_vertex` (step start) and
/// `t_curr` the time in `curr_vertex` (step end, which overshoots the frame
/// boundary because the substep dt rarely divides `1/fps`). `t_frame` is the
/// exact boundary `frame / fps`. Returns the lerp weight `alpha` in `[0, 1]`
/// such that `prev + alpha * (curr - prev)` is the pose at `t_frame`. A
/// zero/degenerate span (frame 0 rest pose where `t_prev == t_curr`) returns
/// `1.0`, which writes `curr_vertex` verbatim.
fn frame_interp_alpha(t_frame: f64, t_prev: f64, t_curr: f64) -> f32 {
    let span = t_curr - t_prev;
    if span > 1e-12 {
        (((t_frame - t_prev) / span).clamp(0.0, 1.0)) as f32
    } else {
        1.0
    }
}

/// Resolve the world-space spin axis for an angular velocity override from a
/// body's live positions. `positions` is packed `(x, y, z)` per vertex
/// (length `3 * count`) as gathered from the GPU at the current step.
/// Returns the centroid and the unit principal axis `pca_index`
/// (0 = largest extent ... 2 = thinnest), eigenvalues sorted descending to
/// match the `pdrd_hinge_axis` convention. The eigenvector sign is
/// canonicalized (the component of largest magnitude is made positive) so a
/// signed angular speed has a well-defined handedness that agrees with the
/// Blender overlay. Returns `None` when the body is too small or the chosen
/// axis is degenerate.
/// Mean of the gathered positions (packed `(x, y, z)` per vertex). Used as
/// the pivot for a fixed world-axis angular override. Returns `None` if empty.
fn centroid_of(positions: &[f32]) -> Option<[f32; 3]> {
    let count = positions.len() / 3;
    if count == 0 {
        return None;
    }
    let mut c = [0.0f64; 3];
    for k in 0..count {
        c[0] += positions[3 * k] as f64;
        c[1] += positions[3 * k + 1] as f64;
        c[2] += positions[3 * k + 2] as f64;
    }
    let inv = 1.0 / count as f64;
    Some([(c[0] * inv) as f32, (c[1] * inv) as f32, (c[2] * inv) as f32])
}

fn resolve_principal_axis(positions: &[f32], pca_index: usize) -> Option<([f32; 3], [f32; 3])> {
    let count = positions.len() / 3;
    if count < 3 {
        return None;
    }
    let mut centroid = na::Vector3::<f64>::zeros();
    for k in 0..count {
        centroid += na::Vector3::new(
            positions[3 * k] as f64,
            positions[3 * k + 1] as f64,
            positions[3 * k + 2] as f64,
        );
    }
    centroid /= count as f64;

    let mut cov = na::Matrix3::<f64>::zeros();
    for k in 0..count {
        let r = na::Vector3::new(
            positions[3 * k] as f64,
            positions[3 * k + 1] as f64,
            positions[3 * k + 2] as f64,
        ) - centroid;
        cov += r * r.transpose();
    }

    let eig = cov.symmetric_eigen();
    // Sort axes by DESCENDING eigenvalue (0 = largest extent, 2 = thinnest),
    // matching the hinge / Blender PCA convention.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut axis = eig.eigenvectors.column(order[pca_index]).into_owned();
    let norm = axis.norm();
    if !norm.is_finite() || norm <= 1e-12 {
        return None;
    }
    axis /= norm;

    // Canonicalize sign: component of largest magnitude is made positive.
    let mut jmax = 0usize;
    for j in 1..3 {
        if axis[j].abs() > axis[jmax].abs() {
            jmax = j;
        }
    }
    if axis[jmax] < 0.0 {
        axis = -axis;
    }

    Some((
        [centroid[0] as f32, centroid[1] as f32, centroid[2] as f32],
        [axis[0] as f32, axis[1] as f32, axis[2] as f32],
    ))
}

fn write_current_time_to_file(file_path: &str) -> std::io::Result<()> {
    let now = Local::now();
    let time_str = now.to_rfc3339();
    let path = Path::new(file_path);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{time_str}")?;
    Ok(())
}

fn write_intersection_records(output_dir: &str, inv_world: f32) {
    // The buffer length and the max_count we hand across the FFI both use the
    // Rust mirror of the capacity, which must equal the canonical
    // cpp/data.hpp MAX_INTERSECTION_RECORDS macro that bounds the C++-side
    // writes; if they diverge the C++ getter clamps to its own macro and
    // silently under-reports here.
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
        3 => "point_point",
        _ => "unknown",
    };
    let json_records: Vec<serde_json::Value> = records
        .iter()
        .map(|r| {
            let n0 = r.num_verts0 as usize;
            let n1 = r.num_verts1 as usize;
            // Positions come back at sim scale; divide by world_scaling (via
            // inv_world) so the overlay markers land at the authored scale.
            let mut pos0 = Vec::new();
            for i in 0..n0 {
                pos0.push(vec![
                    r.positions[i * 3] * inv_world,
                    r.positions[i * 3 + 1] * inv_world,
                    r.positions[i * 3 + 2] * inv_world,
                ]);
            }
            let mut pos1 = Vec::new();
            for i in 0..n1 {
                let base = n0 * 3 + i * 3;
                pos1.push(vec![
                    r.positions[base] * inv_world,
                    r.positions[base + 1] * inv_world,
                    r.positions[base + 2] * inv_world,
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
    // Write via a temp file + atomic rename so a crash mid-write never
    // leaves a half-written records file for the overlay / server to read.
    let path = format!("{output_dir}/intersection_records.json");
    let tmp = format!("{path}.tmp");
    match std::fs::File::create(&tmp) {
        Ok(file) => {
            if serde_json::to_writer_pretty(file, &json).is_ok()
                && std::fs::rename(&tmp, &path).is_ok()
            {
                info!("wrote {count} intersection records to {path}");
            } else {
                error!("failed to write {path}");
                let _ = std::fs::remove_file(&tmp);
            }
        }
        Err(_) => error!("failed to create {tmp}"),
    }
}

#[cfg(test)]
mod rest_shape_tests {
    use super::*;

    // The time-varying rest-shape keyframe interpolation used per step.
    // Endpoints must reproduce the keyframes exactly (alpha 0), the interior
    // is a component-wise lerp, and out-of-range times clamp to the ends.
    #[test]
    fn interp_rest_shape_endpoints_midpoint_and_clamp() {
        let i2 = Mat2x2f::identity();
        let i3 = Mat3x3f::identity();
        let keyframes: Vec<RestShapeKeyframe> = vec![
            (0.0_f64, vec![i2], vec![i3], vec![0u8], vec![0u8]),
            (1.0_f64, vec![i2 * 2.0], vec![i3 * 2.0], vec![0u8], vec![0u8]),
        ];

        let r0 = interp_rest_shape(&keyframes, 0.0);
        assert!((r0.inv_rest2x2.as_slice()[0] - i2).amax() < 1e-6);
        assert!((r0.inv_rest3x3.as_slice()[0] - i3).amax() < 1e-6);

        let r1 = interp_rest_shape(&keyframes, 1.0);
        assert!((r1.inv_rest2x2.as_slice()[0] - i2 * 2.0).amax() < 1e-6);
        assert!((r1.inv_rest3x3.as_slice()[0] - i3 * 2.0).amax() < 1e-6);

        let rm = interp_rest_shape(&keyframes, 0.5);
        assert!((rm.inv_rest2x2.as_slice()[0] - i2 * 1.5).amax() < 1e-6);
        assert!((rm.inv_rest3x3.as_slice()[0] - i3 * 1.5).amax() < 1e-6);

        // Below the first keyframe clamps to frame 0.
        let rb = interp_rest_shape(&keyframes, -5.0);
        assert!((rb.inv_rest2x2.as_slice()[0] - i2).amax() < 1e-6);
        // Above the last keyframe clamps to the final frame.
        let ra = interp_rest_shape(&keyframes, 5.0);
        assert!((ra.inv_rest2x2.as_slice()[0] - i2 * 2.0).amax() < 1e-6);
    }

    // The exclude mask is the OR of the two bracketing keyframes, so an element
    // near-singular in either adjacent sample is dropped for the whole interval.
    #[test]
    fn interp_rest_shape_excludes_union_of_bracketing() {
        let i3 = Mat3x3f::identity();
        // two tets: tet0 singular only at the second keyframe, tet1 never.
        let keyframes: Vec<RestShapeKeyframe> = vec![
            (0.0_f64, vec![], vec![i3, i3], vec![], vec![0u8, 0u8]),
            (1.0_f64, vec![], vec![i3, i3], vec![], vec![1u8, 0u8]),
        ];
        // Mid-interval: tet0 excluded (OR), tet1 not.
        let rm = interp_rest_shape(&keyframes, 0.5);
        assert_eq!(rm.exclude_tet.as_slice(), &[1u8, 0u8]);
        // At the first keyframe exactly, still excluded for tet0 (i0==i1==1? no:
        // time 0.0 -> i0=i1=0, mask = 0|0). Confirm the clamp picks frame 0.
        let r0 = interp_rest_shape(&keyframes, 0.0);
        assert_eq!(r0.exclude_tet.as_slice(), &[0u8, 0u8]);
    }

    // A single-keyframe schedule must not panic and returns that frame for any
    // query time (the degenerate static-rest case).
    #[test]
    fn interp_rest_shape_single_keyframe() {
        let i3 = Mat3x3f::identity();
        let keyframes: Vec<RestShapeKeyframe> =
            vec![(0.5_f64, Vec::<Mat2x2f>::new(), vec![i3 * 3.0], vec![], vec![0u8])];
        let r = interp_rest_shape(&keyframes, 2.0);
        assert!(r.inv_rest2x2.as_slice().is_empty());
        assert!((r.inv_rest3x3.as_slice()[0] - i3 * 3.0).amax() < 1e-6);
    }
}

#[cfg(test)]
mod frame_interp_tests {
    use super::frame_interp_alpha;

    // dt=0.01, fps=30 (3.333 substeps/frame): the step that crosses frame 1
    // runs from t=0.03 to 0.04, and the exact boundary 1/30 = 0.03333 sits one
    // third of the way in. Without interpolation the frame would be recorded at
    // the overshot 0.04 (a 0.2-frame error); the weight pulls it back to 0.0333.
    #[test]
    fn alpha_places_frame_on_the_exact_boundary() {
        let a = frame_interp_alpha(1.0 / 30.0, 0.03, 0.04);
        assert!((a - 1.0 / 3.0).abs() < 1e-5, "alpha={a}");
    }

    // A full step that lands exactly on the boundary (alpha 1) writes curr; a
    // boundary sitting exactly on the step start writes prev (alpha 0). These
    // are the integer-substeps-per-frame endpoints, where the output is exact.
    #[test]
    fn alpha_endpoints_are_exact() {
        assert_eq!(frame_interp_alpha(0.04, 0.03, 0.04), 1.0);
        assert_eq!(frame_interp_alpha(0.03, 0.03, 0.04), 0.0);
    }

    // The rest pose (frame 0) is emitted with t_prev == t_curr; the degenerate
    // span must collapse to curr (weight 1.0) instead of dividing by zero.
    #[test]
    fn alpha_zero_span_is_curr() {
        assert_eq!(frame_interp_alpha(0.0, 0.0, 0.0), 1.0);
    }

    // A boundary outside the bracketing step (should never happen given the
    // caller only emits crossed frames) clamps into range rather than
    // extrapolating past prev/curr.
    #[test]
    fn alpha_clamps_out_of_range() {
        assert_eq!(frame_interp_alpha(0.05, 0.03, 0.04), 1.0);
        assert_eq!(frame_interp_alpha(0.02, 0.03, 0.04), 0.0);
    }
}
