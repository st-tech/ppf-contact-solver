// File: run_state.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! The rotational run state a step reads from the step before it, as a
//! checkpoint carries it (`runstate_<N>.bin.gz`).
//!
//! A checkpoint's `state_<N>.bin.gz` holds the two committed poses, the clock
//! and `prev_dt`, which is everything the POSITIONS carry from one step to the
//! next. Three kinds of scene carry more, each in a device buffer the dataset
//! only seeds:
//!
//! - COSSERAT FRAMES. Each rod edge's material frame and the frame of the step
//!   before it: the rotational inertia predicts a frame from the last step's
//!   incremental rotation (`driver/cosserat.rs`), so both the committed frame
//!   and the one before it are state. A resume without them restarts every
//!   frame at its authored orientation under a pose that has moved on.
//! - SAND GRAIN SPIN. Each grain's angular velocity, integrated at the commit
//!   and read by the next step's condense; a resume without it stops every
//!   grain's roll.
//! - PDRD RUNNING ROTATION. Each body's rotation composed from the increments
//!   the line search accepted, and the latch saying it has been seeded. A
//!   resume without it re-fits the rotation to the saved pose, which is the
//!   drift the anchored rigidify exists to avoid.
//!
//! At a commit, the live copy of each equals the committed one (the frames'
//! iterate equals `prev`, the spin equals its snapshot), so the record holds
//! the committed copies and [`RunState::install`] writes them into every
//! buffer the next step reads.
//!
//! WHICH PARTS A SCENE NEEDS IS DECIDED BY THE ALLOCATED STATE, not by the
//! record: a part the scene needs and the record lacks is refused by name, and
//! so is a part sized for another scene. A resume never runs on a part it was
//! not handed.

use ppf_cts_compute::Device;

use super::scene::{Fatal, FatalResult};
use super::state::SolverState;

/// The parts of the run state a scene carries, read off the allocated state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RunStateKinds {
    /// Cosserat frames, four floats each.
    pub frames: usize,
    /// Vertices whose spin a SAND scene integrates (every vertex of a scene
    /// with any grain, as the spin buffers are sized).
    pub spin_vertices: usize,
    /// PDRD rigid bodies.
    pub bodies: usize,
}

impl RunStateKinds {
    pub fn of(state: &SolverState) -> Self {
        Self {
            frames: state.cosserat.slots,
            spin_vertices: if state.sizes.grains > 0 { state.sizes.vertices } else { 0 },
            bodies: state.rigid.n_bodies,
        }
    }

    /// The same parts read off the scene record alone, by the rules the
    /// allocation sizes them by, so a resume can refuse a checkpoint before
    /// it discards anything or allocates a device (`main.rs`). The test below
    /// holds the two readings equal.
    pub fn of_dataset(data: &crate::data::DataSet) -> Self {
        // Safety: the record is live for the call and the array holds its
        // stated length.
        let inertia: &[f32] = unsafe { super::scene::slice(&data.grain_inv_inertia) };
        let grains = inertia.iter().any(|value| *value > 0.0);
        Self {
            frames: data.cosserat_edge.size as usize,
            spin_vertices: if grains { data.vertex.curr.size as usize } else { 0 },
            bodies: data.prop.pdrd_body.size as usize,
        }
    }

    pub fn any(&self) -> bool {
        self.frames > 0 || self.spin_vertices > 0 || self.bodies > 0
    }

    /// The parts, named, for a refusal.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.frames > 0 {
            parts.push(format!("{} Cosserat rod frames", self.frames));
        }
        if self.spin_vertices > 0 {
            parts.push(format!("SAND grain spin over {} vertices", self.spin_vertices));
        }
        if self.bodies > 0 {
            parts.push(format!("the running rotation of {} PDRD bodies", self.bodies));
        }
        parts.join(", ")
    }
}

/// The run state as a checkpoint carries it. Every vector is empty for a part
/// the scene does not have.
#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RunState {
    /// The committed Cosserat frames, four floats a frame (a unit quaternion).
    pub cosserat_prev: Vec<f32>,
    /// The frames one step before them, which the rotational inertia's
    /// predictor differences against.
    pub cosserat_prev2: Vec<f32>,
    /// Each vertex's grain spin as the last commit integrated it, three floats
    /// a vertex.
    pub grain_omega: Vec<f32>,
    /// Each body's running rotation, nine floats a body.
    pub rigid_running_rotation: Vec<f32>,
    /// Whether the running rotation was seeded from a fitted pose.
    pub rigid_rotation_seeded: bool,
}

impl RunState {
    /// The committed run state, read back for a checkpoint.
    pub fn snapshot(state: &SolverState, device: &mut impl Device) -> FatalResult<Self> {
        let kinds = RunStateKinds::of(state);
        let mut out = RunState::default();
        if kinds.frames > 0 {
            let (prev, prev2) = state.cosserat.history(device)?;
            out.cosserat_prev = prev;
            out.cosserat_prev2 = prev2;
        }
        if kinds.spin_vertices > 0 {
            out.grain_omega = vec![0.0; 3 * kinds.spin_vertices];
            state.grain_omega_prev.read(device, 0, &mut out.grain_omega)?;
        }
        if kinds.bodies > 0 {
            out.rigid_running_rotation = vec![0.0; 9 * kinds.bodies];
            state.rigid_running_rotation.read(device, 0, &mut out.rigid_running_rotation)?;
            out.rigid_rotation_seeded = state.rigid_rotation_seeded;
        }
        Ok(out)
    }

    /// Whether this record holds exactly the parts `kinds` names, at their
    /// sizes. A host check, so a resume can refuse before it discards anything.
    pub fn check(&self, kinds: RunStateKinds) -> FatalResult<()> {
        let check = |what: &str, have: usize, per: usize, count: usize| -> FatalResult<()> {
            if have != per * count {
                return Err(Fatal::invariant(format!(
                    "solver driver: the checkpoint's run state holds {have} floats of {what} \
                     where this scene's {count} take {per} each; it was written for another \
                     scene or by a solver that did not keep them, so the resume would run on \
                     state from somewhere else"
                )));
            }
            Ok(())
        };
        check("Cosserat frames", self.cosserat_prev.len(), 4, kinds.frames)?;
        check("previous Cosserat frames", self.cosserat_prev2.len(), 4, kinds.frames)?;
        check("grain spin", self.grain_omega.len(), 3, kinds.spin_vertices)?;
        check("PDRD running rotation", self.rigid_running_rotation.len(), 9, kinds.bodies)?;
        Ok(())
    }

    /// Install a checkpoint's run state, refusing one that does not hold
    /// exactly the parts this scene carries, at this scene's sizes.
    pub fn install(&self, state: &mut SolverState, device: &mut impl Device) -> FatalResult<()> {
        let kinds = RunStateKinds::of(state);
        self.check(kinds)?;
        if kinds.frames > 0 {
            state.cosserat.set_history(device, &self.cosserat_prev, &self.cosserat_prev2)?;
        }
        if kinds.spin_vertices > 0 {
            state.grain_omega.write(device, 0, &self.grain_omega)?;
            state.grain_omega_prev.write(device, 0, &self.grain_omega)?;
        }
        if kinds.bodies > 0 {
            state.rigid_running_rotation.write(device, 0, &self.rigid_running_rotation)?;
            state.rigid_rotation_seeded = self.rigid_rotation_seeded;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! A resume reproduces the uninterrupted run.
    //!
    //! A bent Cosserat chain falls under gravity with frames that start off
    //! their edges, so the shear turns every frame through the run and the
    //! rotational inertia carries that turn from step to step. The scene runs
    //! `2 * HALF` steps uninterrupted; a second state is then allocated over
    //! the pose the first held at `HALF`, as a `--load` resume is, handed the
    //! run state, and run the remaining `HALF` steps. Both end poses must agree.
    //! A third state resumed WITHOUT the run state must not, which is what
    //! shows the frames are run state at all.

    use super::*;
    use crate::cvec::CVec;
    use crate::data::{Barrier, EdgeParam, FrictionMode, ParamSet, PrecondMode, Vec2u, Vec3f};
    use crate::driver::assemble::tests::{install_pattern, install_rod_adjacency};
    use crate::driver::launch::host_device;
    use crate::driver::test_scene::{position, TestScene};
    use ppf_cts_compute::HostDevice;

    const HALF: usize = 12;
    const DT: f32 = 1e-2;

    fn chain_scene(curr: &[Vec3f], prev: &[Vec3f]) -> TestScene {
        let vertices = curr.len();
        let slots = vertices - 1;
        let edges: Vec<Vec2u> = (0..slots as u32).map(|i| Vec2u::new(i, i + 1)).collect();
        let mut scene = TestScene::new(vertices).with_edges(&edges);
        scene.data.vertex.curr = CVec::from(curr);
        scene.data.vertex.prev = CVec::from(prev);
        scene.data.rod_count = slots as u32;
        scene.data.surface_vert_count = vertices as u32;
        scene.data.param_arrays.edge = CVec::from(
            &[EdgeParam {
                stiffness: 1.0e3,
                cosserat_shear: 1.0e2,
                cosserat_bend: 1.0e-1,
                cosserat_twist: 1.0e-1,
                ..EdgeParam::default()
            }][..],
        );
        // THE AUTHORED (REST) LENGTHS, which the resumed scene keeps as the
        // build's dataset does: they are measured off the first pose.
        let rest = authored();
        for (i, prop) in scene.data.prop.edge.as_mut_slice().iter_mut().enumerate() {
            let (a, b) = (rest[i], rest[i + 1]);
            let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
            prop.length = length;
            prop.initial_length = length;
            prop.mass = 0.05;
        }
        for vertex in scene.data.prop.vertex.as_mut_slice() {
            vertex.mass = 0.025;
        }
        // Every frame starts a quarter of a radian off the identity about x,
        // so none sits on its edge and the shear turns them all.
        let tilt = [0.125f32.sin(), 0.0, 0.0, 0.125f32.cos()];
        scene.data.cosserat_edge = CVec::from(&(0..slots as u32).collect::<Vec<_>>()[..]);
        scene.data.cosserat_edge_nodes = CVec::from(&edges[..]);
        scene.data.cosserat_quat = CVec::from(&tilt.repeat(slots)[..]);
        scene.data.cosserat_fixed = CVec::from(&vec![0u32; slots][..]);
        scene.data.cosserat_end_omega = CVec::from(&vec![0.0f32; slots][..]);
        let bend: Vec<Vec2u> = (0..slots as u32 - 1).map(|s| Vec2u::new(s, s + 1)).collect();
        let darboux = vec![Vec3f::new(0.0, 0.0, 0.0); bend.len()];
        let inv_len = vec![10.0f32; bend.len()];
        let metric: Vec<f32> =
            inv_len.iter().flat_map(|l| crate::cosserat::rest_metric([0.0; 3], *l)).collect();
        scene.data.cosserat_bend_slots = CVec::from(&bend[..]);
        scene.data.cosserat_rest_darboux = CVec::from(&darboux[..]);
        scene.data.cosserat_bend_invlen = CVec::from(&inv_len[..]);
        scene.data.cosserat_rest_metric = CVec::from(&metric[..]);
        install_rod_adjacency(&mut scene.data, vertices, &edges);
        install_pattern(&mut scene.data, vertices);
        scene
    }

    /// A zigzag of five nodes, 0.1 m edges, in the x-y plane.
    fn authored() -> Vec<[f32; 3]> {
        vec![[0.0, 0.0, 0.0], [0.08, 0.06, 0.0], [0.16, 0.0, 0.0], [0.24, 0.06, 0.0], [0.32, 0.0, 0.0]]
    }

    fn param() -> ParamSet {
        // Safety: `ParamSet` is a repr(C) record of floats, integers, bools
        // and three C-like enums whose first variant is zero.
        let mut p: ParamSet = unsafe { std::mem::zeroed() };
        p.dt = DT;
        p.prev_dt = DT;
        p.playback = 1.0;
        p.target_toi = 0.25;
        p.stitch_length_factor = 10.0;
        p.cg_max_iter = 10000;
        p.cg_tol = 1e-3;
        p.line_search_max_t = 1.25;
        p.ccd_eps = 1e-7;
        p.max_dx = 1.0;
        p.eiganalysis_eps = 1e-2;
        p.friction_eps = 1e-5;
        p.constraint_tol = 0.01;
        p.gravity = Vec3f::new(0.0, -9.8, 0.0);
        p.barrier = Barrier::Cubic;
        p.friction_mode = FrictionMode::Min;
        p.precond = PrecondMode::BlockJacobi;
        p.csrmat_max_nnz = 1_000_000;
        p.disable_contact = true;
        p.max_newton_steps = 2048;
        p
    }

    struct Run {
        scene: TestScene,
        state: SolverState,
        device: HostDevice,
        param: ParamSet,
    }

    impl Run {
        fn new(curr: &[Vec3f], prev: &[Vec3f], param: ParamSet) -> Self {
            let scene = chain_scene(curr, prev);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }.expect("the chain allocates");
            // What `update_constraint` stashes before every step: no pin, no
            // collider, no torque and no stitch.
            state.stash_pins(&mut device, &[], &[]).unwrap();
            state.stash_colliders(&mut device, &[], &[]).unwrap();
            state.stash_torque(&mut device, &[], &[]).unwrap();
            state.stash_stitches(&mut device, &[]).unwrap();
            // Allocated on every scene by `initialize()`, contact or not.
            // Safety: as above.
            state.analytic = Some(
                unsafe { super::super::collider::Analytic::allocate(&mut device, &scene.data) }
                    .unwrap(),
            );
            Self { scene, state, device, param }
        }

        fn steps(&mut self, count: usize) {
            for _ in 0..count {
                // Safety: the scene is live and the state was allocated for it.
                let result = unsafe {
                    super::super::step::advance(
                        &mut self.device,
                        &self.scene.data,
                        &mut self.param,
                        &mut self.state,
                    )
                }
                .expect("the step runs");
                assert!(result.success(), "a step failed");
            }
        }

        fn pose(&mut self) -> (Vec<f32>, Vec<f32>) {
            self.state.positions.download(&mut self.device).unwrap();
            self.state.positions_prev.download(&mut self.device).unwrap();
            (self.state.positions.host().to_vec(), self.state.positions_prev.host().to_vec())
        }
    }

    /// The device's flat words as the positions a checkpoint carries, bit for
    /// bit, as `copy_to_dataset` hands a resume its pose.
    fn positions(words: &[f32]) -> Vec<Vec3f> {
        words
            .chunks(3)
            .map(|c| Vec3f::new(c[0], c[1], c[2]))
            .collect()
    }

    fn largest_difference(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn a_cosserat_chain_resumed_from_its_run_state_ends_where_the_uninterrupted_run_does() {
        let start: Vec<Vec3f> = authored().iter().map(|p| position(p[0], p[1], p[2])).collect();
        let mut whole = Run::new(&start, &start, param());
        whole.steps(HALF);
        let (curr, prev) = whole.pose();
        let saved = RunState::snapshot(&whole.state, &mut whole.device).unwrap();
        let (whole_time, whole_prev_dt) = (whole.param.time, whole.param.prev_dt);
        let clock = || {
            let mut p = param();
            p.time = whole_time;
            p.prev_dt = whole_prev_dt;
            p
        };
        assert_eq!(saved.cosserat_prev.len(), 16);
        assert_ne!(saved.cosserat_prev, saved.cosserat_prev2, "the frames did not move");
        whole.steps(HALF);
        let (end, _) = whole.pose();

        let (curr, prev) = (positions(&curr), positions(&prev));
        let mut resumed = Run::new(&curr, &prev, clock());
        saved.install(&mut resumed.state, &mut resumed.device).unwrap();
        assert_eq!(RunState::snapshot(&resumed.state, &mut resumed.device).unwrap(), saved);
        resumed.steps(HALF);
        let (resumed_end, _) = resumed.pose();

        let mut forgetful = Run::new(&curr, &prev, clock());
        forgetful.steps(HALF);
        let (forgetful_end, _) = forgetful.pose();

        let start_words: Vec<f32> =
            start.iter().flat_map(|p| [0, 1, 2].map(|k| p[k])).collect();
        let travel = largest_difference(&end, &start_words);
        let difference = largest_difference(&end, &resumed_end);
        let without = largest_difference(&end, &forgetful_end);
        println!(
            "RESULT uninterrupted travel {travel:.3e} m; resumed with the run state ends \
             {difference:.3e} m from it, without it {without:.3e} m"
        );
        assert!(difference <= 1e-6, "the resumed chain ended {difference:.3e} m away");
        assert!(without > 1e3 * difference.max(1e-9), "the frames made no difference ({without:.3e} m)");
    }

    #[test]
    fn a_run_state_for_another_scene_is_refused() {
        let start: Vec<Vec3f> = authored().iter().map(|p| position(p[0], p[1], p[2])).collect();
        let mut run = Run::new(&start, &start, param());
        let mut saved = RunState::snapshot(&run.state, &mut run.device).unwrap();
        // A frame short.
        saved.cosserat_prev.truncate(12);
        assert!(saved.install(&mut run.state, &mut run.device).is_err());
        // Spin for a scene with no grain.
        let mut spin = RunState::snapshot(&run.state, &mut run.device).unwrap();
        spin.grain_omega = vec![0.0; 15];
        assert!(spin.install(&mut run.state, &mut run.device).is_err());
        // An empty record for a scene with frames.
        assert!(RunState::default().install(&mut run.state, &mut run.device).is_err());
        assert_eq!(RunStateKinds::of(&run.state).frames, 4);
        assert_eq!(RunStateKinds::of(&run.state), RunStateKinds::of_dataset(&run.scene.data));
    }
}
