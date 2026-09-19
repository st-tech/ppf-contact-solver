// File: crates/ppf-cts-solver/src/driver/scene.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The typed borrowed views over `DataSet`, and the fatal report they raise.
//!
//! THIS IS THE ONE PLACE RAW POINTERS BECOME SLICES. Every other module in
//! `src/driver` asks here for what it needs and gets a Rust slice with a length,
//! so an out-of-range access is a bounds check rather than a wild write. That
//! matters more on this backend than it looks: THE ADDRESSING IS DIRECT, so
//! these slices alias the host's live buffers, and a stray write here corrupts
//! the caller's own `DataSet` rather than a mirror of it.
//!
//! WHY THERE IS NO MIRROR. `CVec<T>` (`src/cvec.rs`) and the C++ `Vec<T>`
//! (`src/kernels/vec/vec.hpp`) are the same `repr(C)` triple, and `backend.rs`
//! already reads `dataset.vertex.curr.data` by host pointer straight after
//! `fetch()`. Copying every container into a second set would cost about sixty
//! copies per step and would buy nothing: there is no device on the far side of
//! this backend. It is also why `fetch()`, `fetch_inv_rest()` and
//! `fetch_rest_angles()` are legitimately empty here, and why each of them says
//! so in a comment: an empty `fetch()` under any other model is exactly the
//! frozen-animation failure this backend exists to avoid.

use crate::cvec::CVec;
use crate::data::{
    DataSet, EdgeProp, FaceProp, HingeProp, TetProp, Vec2u, Vec3u, Vec4u, VertexProp,
};

/// A fatal reason, in the two parts the status record carries.
///
/// The CODE is what `crates/ppf-cts-formats/src/status/mod.rs` maps to a
/// `CrashKind`, and the DETAIL is what a reader gets instead of generic text
/// naming the code. Both halves are load-bearing: a run that stops with a code
/// and no detail reports every fatal on this backend as the same thing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fatal {
    pub code: u8,
    pub detail: String,
}

impl Fatal {
    /// A violated invariant the host detected before any kernel ran: a null
    /// pointer, a length disagreement, a call before `initialize()`.
    pub fn invariant(detail: impl Into<String>) -> Self {
        Self {
            code: ppf_cts_formats::status::error_code::SOLVER_INVARIANT,
            detail: detail.into(),
        }
    }

    /// An index that would have been dereferenced inside a kernel range.
    ///
    /// This is the CPU analogue of a device-side `assert` trapping, which is
    /// what the CUDA backend gets for free from its live release asserts and
    /// what Metal cannot have at all (it never faults on an out-of-bounds
    /// access: reads return zero, writes are dropped). Reporting it under its
    /// own code rather than folding it into `invariant` is the difference
    /// between "the scene handed us something impossible" and "a kernel was
    /// about to read outside its buffer", which are different bugs in different
    /// places.
    pub fn device_assert(detail: impl Into<String>) -> Self {
        Self {
            code: ppf_cts_formats::status::error_code::DEVICE_ASSERT,
            detail: detail.into(),
        }
    }

    /// An allocation this backend could not make.
    pub fn out_of_memory(detail: impl Into<String>) -> Self {
        Self {
            code: ppf_cts_formats::status::error_code::OOM,
            detail: detail.into(),
        }
    }
}

/// A seam fault becomes a fatal by its KIND, so the run's status names what
/// went wrong rather than reporting every backend refusal as one thing.
///
/// The mapping is the only place the driver interprets a [`Fault`], and each arm
/// is a different bug in a different place: a device diagnostic is a check a
/// kernel range failed, an allocation is the machine, and everything else is a
/// contract this driver broke against the seam before any kernel ran.
impl From<ppf_cts_compute::Fault> for Fatal {
    fn from(fault: ppf_cts_compute::Fault) -> Self {
        use ppf_cts_compute::Fault;
        match &fault {
            Fault::Device { .. } => Fatal::device_assert(format!("solver driver: {fault}")),
            Fault::Alloc { .. } => Fatal::out_of_memory(format!("solver driver: {fault}")),
            Fault::MissingKernel { .. }
            | Fault::StaleRegion { .. }
            | Fault::Shape { .. }
            | Fault::Platform { .. } => Fatal::invariant(format!("solver driver: {fault}")),
        }
    }
}

/// The result shape every fallible operation in this backend returns.
///
/// The FFI entry points are the only place a `Fatal` becomes a process exit, so
/// every rule below is reachable from a unit test with no process to kill. That
/// is the whole reason the logic is written against `Result` rather than
/// exiting where it detects the problem.
pub type FatalResult<T> = Result<T, Fatal>;

/// Borrow a `CVec`'s buffer as a shared slice.
///
/// # Safety
/// `cvec` must describe a live allocation of at least `size` elements, and no
/// mutable reference to it may be live for the returned lifetime.
pub unsafe fn slice<'a, T>(cvec: &CVec<T>) -> &'a [T] {
    if cvec.data.is_null() || cvec.size == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(cvec.data, cvec.size as usize)
    }
}

/// Borrow a `CVec`'s buffer as a mutable slice.
///
/// # Safety
/// `cvec` must describe a live allocation of at least `size` elements, and no
/// other reference to it may be live for the returned lifetime. The backend
/// entry points are called one at a time from the host's step loop, which is
/// what makes that hold; a future phase that runs two of them concurrently owes
/// a different argument here rather than a second call to this function.
///
/// Note which memory this touches and which it does not: the returned slice
/// addresses the HEAP BUFFER the `CVec` points at, allocated and owned
/// elsewhere. The `DataSet` record itself is only ever READ through the
/// `*const` the host passed in, so nothing here writes through a pointer
/// derived from a shared reference to it.
#[allow(clippy::mut_from_ref)]
pub unsafe fn slice_mut<'a, T>(cvec: &CVec<T>) -> &'a mut [T] {
    if cvec.data.is_null() || cvec.size == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(cvec.data, cvec.size as usize)
    }
}

/// The subset of `DataSet` this phase reaches, with every container as a slice.
///
/// It is built fresh at each entry point rather than cached, which is not a
/// style choice: `backend.rs` drops and rebuilds several of these containers
/// between steps, so a cached view would be pointing at freed memory. The
/// `Constraint` is the extreme case and has its own rule; see
/// `super::constraint`.
pub struct SceneView {
    data: *const DataSet,
}

// The view is handed between the FFI entry points, which the host may call from
// any thread. The pointer addresses memory `backend.rs` owns for the whole run.
unsafe impl Send for SceneView {}

// EVERY `*_mut` BELOW HANDS OUT A MUTABLE SLICE FROM A SHARED VIEW, WHICH IS
// THE DESIGN AND NOT AN OVERSIGHT. `clippy::mut_from_ref` is right to stop at
// this in ordinary code: the compiler cannot prove the returned slices do not
// alias. Here it cannot be made to, because decision D1 is direct addressing and
// the point is to reach the caller's own live buffers, several at once (the pin
// rebuild reads the vertex props while writing the face props). What makes it
// sound is stated per method as a safety contract and rests on one fact: the
// backend entry points are called one at a time from the host's step loop. A
// future phase that runs two of them concurrently owes a new argument, not a
// second allow.
#[allow(clippy::mut_from_ref)]
impl SceneView {
    /// # Safety
    /// `data` must point at a live `DataSet` that outlives every borrow taken
    /// through the returned view.
    pub unsafe fn new(data: *const DataSet) -> Self {
        Self { data }
    }

    /// # Safety
    /// The `DataSet` must still be live.
    unsafe fn get(&self) -> &DataSet {
        &*self.data
    }

    /// # Safety
    /// The `DataSet` must still be live.
    pub unsafe fn vertex_count(&self) -> usize {
        self.get().vertex.curr.size as usize
    }

    /// The start-of-step positions, as flat `int32` components.
    ///
    /// Flat rather than typed because this is what crosses the C ABI to
    /// `entrypoints/shim_override_seed.cpp`, which casts it back to `Vec3f`. The
    /// two `static_assert`s there are what make that recovery checked rather
    /// than assumed.
    ///
    /// No PRODUCTION caller: the device holds the authoritative positions and a
    /// live step reads them through the state's buffers. What reads this is
    /// `driver/seed.rs`'s fixture pair `device_positions` and `refresh_scene`,
    /// which seed the device arrays from the scene and copy them back the way
    /// `fetch()` does, serving seven tests there including
    /// `a_velocity_override_far_from_the_origin_reproduces_the_commanded_velocity`
    /// and `the_gather_returns_the_live_absolute_positions`.
    ///
    /// It is also named as a search term: grepping for `curr_components()` and
    /// `prev_components()`, rather than for the buffer, finds every host write
    /// to a `DataSet` array that the device copy would overwrite. A caller of
    /// either is that whole population, so the two accessors are the index into
    /// this file.
    ///
    /// # Safety
    /// The `DataSet` must still be live.
    #[allow(dead_code)]
    pub unsafe fn curr_components(&self) -> *const f32 {
        self.get().vertex.curr.data as *const f32
    }

    /// The previous-step positions, as flat float components.
    ///
    /// No PRODUCTION caller, and named as a search term, both for the reasons
    /// given on [`SceneView::curr_components`]. What reads this is the same
    /// fixture pair `device_positions` and `refresh_scene` in `driver/seed.rs`,
    /// which seeds and refreshes the previous-step array beside the
    /// start-of-step one, for tests including
    /// `an_angular_override_spins_about_the_commanded_pivot` and
    /// `a_non_positive_step_seeds_nothing`.
    ///
    /// # Safety
    /// The `DataSet` must still be live and no other reference to the buffer
    /// may be active.
    #[allow(dead_code)]
    pub unsafe fn prev_components(&self) -> *mut f32 {
        self.get().vertex.prev.data as *mut f32
    }

    /// # Safety
    /// The `DataSet` must still be live.
    pub unsafe fn prev_count(&self) -> usize {
        self.get().vertex.prev.size as usize
    }

    /// # Safety
    /// The `DataSet` must still be live and no other reference to the buffer
    /// may be active.
    pub unsafe fn vertex_props_mut(&self) -> &mut [VertexProp] {
        slice_mut(&self.get().prop.vertex)
    }

    /// # Safety
    /// The `DataSet` must still be live and no other reference to the buffer
    /// may be active.
    pub unsafe fn edge_props_mut(&self) -> &mut [EdgeProp] {
        slice_mut(&self.get().prop.edge)
    }

    /// # Safety
    /// The `DataSet` must still be live and no other reference to the buffer
    /// may be active.
    pub unsafe fn face_props_mut(&self) -> &mut [FaceProp] {
        slice_mut(&self.get().prop.face)
    }

    /// # Safety
    /// The `DataSet` must still be live and no other reference to the buffer
    /// may be active.
    pub unsafe fn hinge_props_mut(&self) -> &mut [HingeProp] {
        slice_mut(&self.get().prop.hinge)
    }

    /// # Safety
    /// The `DataSet` must still be live and no other reference to the buffer
    /// may be active.
    pub unsafe fn tet_props_mut(&self) -> &mut [TetProp] {
        slice_mut(&self.get().prop.tet)
    }

    /// # Safety
    /// The `DataSet` must still be live.
    pub unsafe fn faces(&self) -> &[Vec3u] {
        slice(&self.get().mesh.mesh.face)
    }

    /// # Safety
    /// The `DataSet` must still be live.
    pub unsafe fn edges(&self) -> &[Vec2u] {
        slice(&self.get().mesh.mesh.edge)
    }

    /// # Safety
    /// The `DataSet` must still be live.
    pub unsafe fn tets(&self) -> &[Vec4u] {
        slice(&self.get().mesh.mesh.tet)
    }

    /// # Safety
    /// The `DataSet` must still be live.
    pub unsafe fn hinges(&self) -> &[Vec4u] {
        slice(&self.get().mesh.mesh.hinge)
    }

    /// The streamed inverse rest matrices, as the destination they are written
    /// into.
    ///
    /// # Safety
    /// The `DataSet` must still be live and no other reference to the buffers
    /// may be active.
    pub unsafe fn inv_rest2x2_mut(&self) -> &mut [crate::data::Mat2x2f] {
        slice_mut(&self.get().inv_rest2x2)
    }

    /// # Safety
    /// The `DataSet` must still be live and no other reference to the buffers
    /// may be active.
    pub unsafe fn inv_rest3x3_mut(&self) -> &mut [crate::data::Mat3x3f] {
        slice_mut(&self.get().inv_rest3x3)
    }
}
