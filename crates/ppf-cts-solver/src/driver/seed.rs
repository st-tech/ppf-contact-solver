// File: crates/ppf-cts-solver/src/driver/seed.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The keyframe velocity seed and the live position readback.
//!
//! This module owns the decomposition and nothing else. Every value it produces
//! comes from `src/kernels/main/override_seed.kernel.cpp` through the ranged
//! entry points in `entrypoints/shim_override_seed.cpp`, which nvcc and the
//! Metal shader compiler compile from the same bytes.
//!
//! | in C++, shared | here, in Rust |
//! |---|---|
//! | `prev = curr - v dt`, `prev -= (omega x (curr - c)) dt`, and the absolute readback, all in the shared body | which vertices, the bounds checks, the fatal report |
//!
//! NOT ONE FLOAT OPERATION HAPPENS IN THIS FILE. The commanded velocity, the
//! angular rate and the pivot pass through as opaque `f32` arguments; they are
//! never scaled, differenced or compared against anything but zero. That is not
//! a stylistic rule: a position expression written a second time in Rust would
//! be a second implementation of the arithmetic three backends are supposed to
//! share, and nothing would check the two against each other.
//!
//! THE ORDER OF THE TWO SEEDS IS PART OF THE CONTRACT. `override_velocity`
//! REPLACES `prev`, and `override_angular_velocity` ACCUMULATES onto it, so a
//! keyframe carrying both yields the full rigid overwrite
//! `prev = curr - (v + omega x (x - c)) dt`. The host's step loop calls them in
//! that order; swapping them would drop the linear half.

use ppf_cts_compute::Device;
use super::kernels::{
    GatherPositionAbsoluteArgs, OverrideAngularSeedListedArgs, OverrideVelocitySeedListedArgs,
};
use super::scene::{Fatal, FatalResult, SceneView};

/// Whether this step can express a velocity at all.
///
/// A COMPARISON, not arithmetic: nothing in this file computes with `dt`, which
/// is handed through to the shared body untouched. Both other backends return
/// early on a non-positive step, so this one does too rather than seeding a
/// displacement of zero or a NaN and letting the integrator find it. NaN is
/// excluded explicitly rather than by relying on it failing `> 0.0`, so the
/// reader does not have to know which way an unordered comparison falls.
fn step_is_usable(dt: f32) -> bool {
    dt.is_finite() && dt > 0.0
}

/// Bounds-check every listed vertex before any of them is written.
///
/// Reported under `DEVICE_ASSERT` rather than `SOLVER_INVARIANT`: this is the
/// index a kernel was about to dereference, which is what a live device assert
/// traps on under CUDA and what Metal cannot see at all (an out-of-bounds read
/// there returns zero and an out-of-bounds write is dropped, both silently).
/// Checking the whole list up front means a rejected keyframe leaves `prev`
/// exactly as it was rather than half seeded.
fn check_indices(who: &str, indices: &[u32], vertex_count: usize) -> FatalResult<()> {
    for (i, vi) in indices.iter().enumerate() {
        if *vi as usize >= vertex_count {
            return Err(Fatal::device_assert(format!(
                "{who} entry {i} names vertex {vi} but the scene has {vertex_count} vertices"
            )));
        }
    }
    Ok(())
}

/// The two vertex buffers must describe the same vertices.
///
/// # Safety
/// `view` must address a live `DataSet`.
unsafe fn paired_vertex_count(view: &SceneView, who: &str) -> FatalResult<usize> {
    let curr = view.vertex_count();
    let prev = view.prev_count();
    if curr != prev {
        return Err(Fatal::invariant(format!(
            "{who}: vertex.curr holds {curr} positions and vertex.prev holds {prev}, so the \
             incoming velocity is not defined"
        )));
    }
    Ok(curr)
}

/// `prev[vi] = curr[vi] - v * dt` for each listed vertex.
///
/// # Safety
/// `view` must address a live `DataSet` with no other reference to its vertex
/// buffers alive for the duration.
/// Stage a caller's index list into the driver's own buffer.
///
/// THE LIST ARRIVES FROM THE FRONTEND as a host slice with no allocation behind
/// it, so it is copied into a buffer the state owns rather than named by
/// address. The buffer is persistent and `Buffer::size` grows only past
/// CAPACITY, so a list no longer than the last one allocates nothing.
fn stage_indices<D: Device>(
    device: &mut D,
    scratch: &mut ppf_cts_compute::Buffer<u32>,
    indices: &[u32],
) -> FatalResult<ppf_cts_compute::Handle> {
    scratch
        .size(device, indices.len(), ppf_cts_compute::AllocLabel("seed.indices"))
        .and_then(|()| scratch.write(device, 0, indices))
        .map_err(|error| {
            Fatal::invariant(format!("solver driver: cannot stage the seed index list: {error:?}"))
        })?;
    Ok(scratch.span(0, indices.len()))
}

pub unsafe fn override_velocity<D: Device>(
    device: &mut D,
    view: &SceneView,
    // THE DEVICE POSITIONS, because the solve reads THOSE. Writing the host
    // `DataSet.vertex.prev` here would leave the seed silently unapplied: it is
    // refreshed FROM the device by `fetch()`, never back into it.
    positions: &mut ppf_cts_compute::ReadbackBuffer<f32>,
    positions_prev: &mut ppf_cts_compute::ReadbackBuffer<f32>,
    scratch: &mut ppf_cts_compute::Buffer<u32>,
    indices: &[u32],
    vx: f32,
    vy: f32,
    vz: f32,
    dt: f32,
) -> FatalResult<()> {
    if indices.is_empty() || !step_is_usable(dt) {
        return Ok(());
    }
    let vertex_count = paired_vertex_count(view, "override_velocity")?;
    check_indices("override_velocity", indices, vertex_count)?;
    // ONE PASS OVER THE WHOLE SPAN, and the reason now lives in the kernel's own
    // declaration rather than in this comment: `Scatter::Atomic` in
    // `super::kernels`, because an index list arrives from a keyframe and
    // nothing guarantees it holds each vertex at most once, so a partition would
    // need a uniqueness proof this backend does not have.
    let args = OverrideVelocitySeedListedArgs {
        curr: positions.handle(),
        prev: positions_prev.handle(),
        indices: stage_indices(device, scratch, indices)?,
        vx,
        vy,
        vz,
        dt,
        count: indices.len() as u32,
        seam_arena_count: 0,
    };
    device.launch("seed.override_velocity", &args, indices.len() as u32)?;
    Ok(())
}

/// `prev[vi] -= (omega x (curr[vi] - c)) * dt` for each listed vertex.
///
/// # Safety
/// As `override_velocity`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn override_angular_velocity<D: Device>(
    device: &mut D,
    view: &SceneView,
    positions: &mut ppf_cts_compute::ReadbackBuffer<f32>,
    positions_prev: &mut ppf_cts_compute::ReadbackBuffer<f32>,
    scratch: &mut ppf_cts_compute::Buffer<u32>,
    indices: &[u32],
    wx: f32,
    wy: f32,
    wz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    dt: f32,
) -> FatalResult<()> {
    if indices.is_empty() || !step_is_usable(dt) {
        return Ok(());
    }
    let vertex_count = paired_vertex_count(view, "override_angular_velocity")?;
    check_indices("override_angular_velocity", indices, vertex_count)?;
    let args = OverrideAngularSeedListedArgs {
        curr: positions.handle(),
        prev: positions_prev.handle(),
        indices: stage_indices(device, scratch, indices)?,
        wx,
        wy,
        wz,
        cx,
        cy,
        cz,
        dt,
        count: indices.len() as u32,
        seam_arena_count: 0,
    };
    device.launch("seed.override_angular_velocity", &args, indices.len() as u32)?;
    Ok(())
}

/// Pack the absolute world position of each listed vertex into `out`.
///
/// The caller solves a centroid, and for a principal-axis keyframe a
/// covariance, from these and hands the result back as the pivot of the angular
/// seed. So they must be the LIVE positions: a pose frozen at build time gives
/// a pivot that does not follow the simulated body, and a zeroed buffer gives a
/// degenerate covariance, which is an angular override that silently does
/// nothing or spins about an arbitrary axis.
///
/// # Safety
/// `view` must address a live `DataSet`, and `out` must be exactly
/// `3 * indices.len()` long.
pub unsafe fn gather_current_positions<D: Device>(
    device: &mut D,
    view: &SceneView,
    // THE DEVICE COPY IS THE LIVE ONE. The host array is refreshed only by
    // `fetch()`, once per output frame, so reading it here would resolve a spin
    // axis from the last frame rather than from the pose the caller means.
    positions: &mut ppf_cts_compute::ReadbackBuffer<f32>,
    scratch: &mut ppf_cts_compute::Buffer<u32>,
    gathered: &mut ppf_cts_compute::ReadbackBuffer<f32>,
    indices: &[u32],
    out: &mut [f32],
) -> FatalResult<()> {
    if indices.is_empty() {
        return Ok(());
    }
    if out.len() != 3 * indices.len() {
        return Err(Fatal::invariant(format!(
            "gather_current_positions was given {} floats for {} vertices, and it packs three per \
             vertex",
            out.len(),
            indices.len()
        )));
    }
    let vertex_count = view.vertex_count();
    check_indices("gather_current_positions", indices, vertex_count)?;
    gathered
        .size(device, out.len(), ppf_cts_compute::AllocLabel("seed.gathered"))
        .map_err(|error| {
            Fatal::invariant(format!("solver driver: cannot size the gather buffer: {error:?}"))
        })?;
    let args = GatherPositionAbsoluteArgs {
        curr: positions.handle(),
        indices: stage_indices(device, scratch, indices)?,
        // The bound the entry point checks each slot against. The host check
        // above runs where the list is built and this one runs where it is
        // used, on every backend, so the two are not redundant.
        vertex_count: vertex_count as u32,
        out: gathered.span(0, out.len()),
        count: indices.len() as u32,
        seam_arena_count: 0,
    };
    device.launch("seed.gather_positions", &args, indices.len() as u32)?;
    // THE CALLER'S SLICE IS ITS OWN MEMORY, not an allocation, so the gathered
    // positions come back through the driver's buffer and are copied out.
    gathered.download(device).map_err(|error| {
        Fatal::invariant(format!("solver driver: cannot read back the gathered positions: {error:?}"))
    })?;
    out.copy_from_slice(&gathered.host()[..out.len()]);
    Ok(())
}

#[cfg(test)]
mod tests {
    /// The two device position buffers, seeded from the scene.
    ///
    /// THE TESTS MIRROR PRODUCTION: the seed kernels write the DEVICE arrays,
    /// and the host `DataSet` is refreshed from them afterwards, which is what
    /// `fetch()` does in a real run. Writing the host arrays directly would
    /// test a path the solver does not take.
    fn device_positions(
        device: &mut impl Device,
        view: &SceneView,
    ) -> (
        ppf_cts_compute::ReadbackBuffer<f32>,
        ppf_cts_compute::ReadbackBuffer<f32>,
    ) {
        let n = 3 * unsafe { view.vertex_count() };
        let mut positions = ppf_cts_compute::ReadbackBuffer::<f32>::default();
        let mut previous = ppf_cts_compute::ReadbackBuffer::<f32>::default();
        positions
            .size(device, n, ppf_cts_compute::AllocLabel("test.seed.curr"))
            .expect("the test allocation succeeds");
        previous
            .size(device, n, ppf_cts_compute::AllocLabel("test.seed.prev"))
            .expect("the test allocation succeeds");
        // Safety: the scene is live and holds `view.vertex_count()` triples.
        unsafe {
            positions
                .seed(device, crate::driver::state::slice_or_empty(view.curr_components(), n))
                .expect("the test seeds curr");
            previous
                .seed(
                    device,
                    crate::driver::state::slice_or_empty(view.prev_components() as *const f32, n),
                )
                .expect("the test seeds prev");
        }
        (positions, previous)
    }

    /// Refresh the scene's host arrays from the device, as `fetch()` does.
    fn refresh_scene(
        device: &mut impl Device,
        positions: &mut ppf_cts_compute::ReadbackBuffer<f32>,
        previous: &mut ppf_cts_compute::ReadbackBuffer<f32>,
        view: &SceneView,
    ) {
        let n = 3 * unsafe { view.vertex_count() };
        positions.download(device).expect("curr downloads");
        previous.download(device).expect("prev downloads");
        // Safety: as above.
        unsafe {
            std::ptr::copy_nonoverlapping(positions.host().as_ptr(), view.curr_components() as *mut f32, n);
            std::ptr::copy_nonoverlapping(previous.host().as_ptr(), view.prev_components(), n);
        }
    }

    use super::*;
    use crate::driver::launch::host_device;
    use crate::driver::test_scene::{position_delta, TestScene};
    use ppf_cts_formats::status::error_code;

    /// One substep, the value the examples run at.
    const DT: f32 = 1.0 / 60.0;

    /// The velocity error a seed of `prev = curr - v * dt` can carry.
    ///
    /// The seed rounds `curr - v * dt` to the float spacing at `curr`'s own
    /// magnitude, and the integrator reads the velocity back as
    /// `(curr - prev) / dt`, which divides that rounding by the substep. These
    /// fixtures place their far vertex and their pivot at 15.0, where the
    /// spacing is at most `15 * f32::EPSILON`, so half of it read back through
    /// `/ DT` bounds the error at about 5.4e-5.
    ///
    /// Derived rather than tuned, so it stays honest if the placement moves. A
    /// seed spelling that is exact only near the origin still fails these on
    /// the far vertex while the near one passes, which is what they are for.
    const SEED_VELOCITY_TOLERANCE: f64 =
        15.0 * (f32::EPSILON as f64) / 2.0 / (DT as f64);

    /// The velocity the seed actually expresses, read back off the positions.
    ///
    /// `(curr - prev) / dt` is exactly what the implicit integrator reads as
    /// the incoming velocity, so this is the quantity the keyframe commanded
    /// and not a proxy for it.
    fn seeded_velocity(scene: &TestScene, vertex: usize) -> [f64; 3] {
        let delta = position_delta(scene.curr(vertex), scene.prev(vertex));
        let dt = DT as f64;
        [delta[0] / dt, delta[1] / dt, delta[2] / dt]
    }

    /// THE RESOLUTION PROPERTY, MEASURED WHERE IT BREAKS.
    ///
    /// Far from the origin is where the seed loses the most: `prev = curr - v
    /// dt` rounds to the spacing of the representation at `curr`'s magnitude,
    /// and dividing that rounding by `dt` is the error in the velocity the
    /// integrator reads back, while at the origin the same seed is exact. So
    /// the far vertex and the near one are asserted to the SAME bound rather
    /// than the far one being allowed to drift, and that is what makes this
    /// test bite: a seed spelling that is exact only near the origin fails it
    /// on the far vertex while the near one still passes, which is the failure
    /// shape in one line.
    #[test]
    fn a_velocity_override_far_from_the_origin_reproduces_the_commanded_velocity() {
        let mut scene = TestScene::new(2);
        scene.place(0, 15.0, -14.5, 3.25);
        scene.place(1, 0.0, 0.0, 0.0);
        let view = scene.view();

        let commanded = [0.25f32, -0.5, 1.0];
        unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = override_velocity(&mut dev, &view, &mut positions, &mut previous, &mut scratch, &[0, 1], commanded[0], commanded[1], commanded[2], DT);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }
            .expect("a well-formed override must be accepted");

        // The error budget is the rounding of `curr - v * dt` at the far
        // vertex's own magnitude, divided by the substep.
        const TOLERANCE: f64 = SEED_VELOCITY_TOLERANCE;
        for vertex in 0..2 {
            let seeded = seeded_velocity(&scene, vertex);
            for axis in 0..3 {
                let error = (seeded[axis] - commanded[axis] as f64).abs();
                assert!(
                    error < TOLERANCE,
                    "vertex {vertex} axis {axis}: seeded {} against commanded {}, error {error:e}",
                    seeded[axis],
                    commanded[axis]
                );
            }
        }
    }

    /// The angular seed produces `omega x r` about the commanded pivot.
    ///
    /// The offsets are exactly representable in `f32` (0.125 is 2^-3), so the
    /// only error left in the comparison is the seed's own: placing the vertex
    /// at an arbitrary offset would fold the `f32` image of the AUTHORED
    /// coordinate into the measurement and hide what is being tested.
    #[test]
    fn an_angular_override_spins_about_the_commanded_pivot() {
        let mut scene = TestScene::new(1);
        scene.place(0, 15.125, 2.0, -3.0);
        let view = scene.view();

        // omega about +z, pivot at the body center: r = (0.125, 0, 0), so
        // v = omega x r = (0, 0.25, 0).
        unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = override_angular_velocity(&mut dev, &view, &mut positions, &mut previous, &mut scratch, &[0], 0.0, 0.0, 2.0, 15.0, 2.0, -3.0, DT);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }
            .expect("a well-formed override must be accepted");

        let seeded = seeded_velocity(&scene, 0);
        let expected = [0.0f64, 0.25, 0.0];
        for axis in 0..3 {
            let error = (seeded[axis] - expected[axis]).abs();
            assert!(
                error < SEED_VELOCITY_TOLERANCE,
                "axis {axis}: seeded {} against expected {}, error {error:e}",
                seeded[axis],
                expected[axis]
            );
        }
    }

    /// The two seeds COMPOSE: linear replaces, angular accumulates.
    ///
    /// A keyframe carrying both must yield the full rigid overwrite
    /// `prev = curr - (v + omega x (x - c)) dt`. Getting the accumulation
    /// backwards would drop the linear half and leave a body that spins in
    /// place instead of travelling.
    #[test]
    fn a_linear_and_an_angular_override_compose_into_one_rigid_field() {
        let mut scene = TestScene::new(1);
        scene.place(0, 15.125, 2.0, -3.0);
        let view = scene.view();

        unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = override_velocity(&mut dev, &view, &mut positions, &mut previous, &mut scratch, &[0], 1.5, 0.0, 0.0, DT);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }.expect("accepted");
        unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = override_angular_velocity(&mut dev, &view, &mut positions, &mut previous, &mut scratch, &[0], 0.0, 0.0, 2.0, 15.0, 2.0, -3.0, DT);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }
            .expect("accepted");

        let seeded = seeded_velocity(&scene, 0);
        let expected = [1.5f64, 0.25, 0.0];
        for axis in 0..3 {
            assert!(
                (seeded[axis] - expected[axis]).abs() < SEED_VELOCITY_TOLERANCE,
                "axis {axis}: seeded {} against expected {}",
                seeded[axis],
                expected[axis]
            );
        }
    }

    /// The gather returns the LIVE absolute positions.
    ///
    /// An implementation that memsets zeros passes every "did it return" test
    /// and then hands the caller a degenerate covariance, so the angular
    /// override silently does nothing or spins about an arbitrary axis. The
    /// assertion is therefore on the values, not on the call.
    #[test]
    fn the_gather_returns_the_live_absolute_positions() {
        let mut scene = TestScene::new(3);
        scene.place(0, 15.0, -14.5, 3.25);
        scene.place(1, 0.0, 0.0, 0.0);
        scene.place(2, -2.5, 1.125, 8.0);
        let view = scene.view();

        // Deliberately out of order and skipping a vertex: the packing is by
        // POSITION IN THE LIST, not by vertex id, and the caller's centroid
        // depends on that.
        let mut out = vec![0.0f32; 6];
        unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = gather_current_positions(&mut dev, &view, &mut positions, &mut scratch, &mut gathered, &[2, 0], &mut out);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }.expect("accepted");
        let expected = [-2.5f32, 1.125, 8.0, 15.0, -14.5, 3.25];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "gathered {out:?} against expected {expected:?}"
            );
        }
        assert!(
            out.iter().any(|v| *v != 0.0),
            "a zeroed gather is the defect this test exists for"
        );
    }

    /// An index outside the scene stops the run and writes nothing.
    #[test]
    fn an_out_of_range_index_fatals_rather_than_writing() {
        let mut scene = TestScene::new(2);
        scene.place(0, 1.0, 2.0, 3.0);
        scene.place(1, 4.0, 5.0, 6.0);
        let before = [scene.prev(0), scene.prev(1)];
        let view = scene.view();

        let fatal = unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = override_velocity(&mut dev, &view, &mut positions, &mut previous, &mut scratch, &[0, 9], 1.0, 0.0, 0.0, DT);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }
            .expect_err("vertex 9 does not exist");
        assert_eq!(
            fatal.code,
            error_code::DEVICE_ASSERT,
            "an index a kernel range was about to dereference is a device assert, \
             not a scene-level invariant"
        );
        assert!(
            fatal.detail.contains("entry 1") && fatal.detail.contains('9'),
            "{:?}",
            fatal.detail
        );
        assert_eq!(
            [scene.prev(0), scene.prev(1)],
            before,
            "the whole list is checked before any of it is written, so vertex 0 \
             must not have been seeded either"
        );

        let fatal =
            unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = override_angular_velocity(&mut dev, &view, &mut positions, &mut previous, &mut scratch, &[5], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, DT);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }
                .expect_err("vertex 5 does not exist");
        assert_eq!(fatal.code, error_code::DEVICE_ASSERT);

        let mut out = vec![0.0f32; 3];
        let fatal = unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = gather_current_positions(&mut dev, &view, &mut positions, &mut scratch, &mut gathered, &[7], &mut out);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }
            .expect_err("vertex 7 does not exist");
        assert_eq!(fatal.code, error_code::DEVICE_ASSERT);
    }

    /// A non-positive step has no velocity to express and touches nothing.
    ///
    /// Both other backends return early on it, so this one does too rather than
    /// dividing by it somewhere downstream.
    #[test]
    fn a_non_positive_step_seeds_nothing() {
        let mut scene = TestScene::new(1);
        scene.place(0, 2.0, 0.0, 0.0);
        let before = scene.prev(0);
        let view = scene.view();
        unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = override_velocity(&mut dev, &view, &mut positions, &mut previous, &mut scratch, &[0], 1.0, 1.0, 1.0, 0.0);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }.expect("accepted");
        unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = override_velocity(&mut dev, &view, &mut positions, &mut previous, &mut scratch, &[0], 1.0, 1.0, 1.0, -1.0);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }.expect("accepted");
        assert_eq!(scene.prev(0), before);
    }

    /// A mismatched output buffer is refused rather than written past.
    #[test]
    fn a_short_gather_buffer_is_refused() {
        let scene = TestScene::new(2);
        let view = scene.view();
        let mut out = vec![0.0f32; 3];
        let fatal = unsafe { let mut dev = host_device();
            // ONE DEVICE. A handle is meaningless against a different
            // allocator, so allocating on one `host_device()` and dispatching
            // against another resolves garbage.
            let (mut positions, mut previous) = device_positions(&mut dev, &view);
            // The staging the production callers take from `SolverState`.
            let mut scratch = ppf_cts_compute::Buffer::<u32>::none();
            let mut gathered = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            let _ = &gathered;
            let r = gather_current_positions(&mut dev, &view, &mut positions, &mut scratch, &mut gathered, &[0, 1], &mut out);
            refresh_scene(&mut dev, &mut positions, &mut previous, &view);
            r }
            .expect_err("three floats cannot hold two vertices");
        assert_eq!(fatal.code, error_code::SOLVER_INVARIANT);
    }
}
