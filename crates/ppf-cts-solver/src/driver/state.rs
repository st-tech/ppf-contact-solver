// File: crates/ppf-cts-solver/src/driver/state.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! What the Newton driver keeps between steps.
//!
//! Two things live here and they are here for opposite reasons.
//!
//! THE BUFFERS are here so a step does not allocate, and the reason is not
//! speed alone: an allocation inside a Newton iteration is a failure point in
//! the
//! middle of a solve, where there is no good answer, while an allocation at
//! `initialize()` can be reported before a frame is written.
//!
//! THE PIN COPY is here because it CANNOT be borrowed: `backend.rs` drops the
//! previous `Constraint` and frees its `CVec` buffers as it assigns the new
//! one, so a
//! pointer kept from `update_constraint` to `advance` addresses freed memory.
//! The CUDA backend does not have this problem because `update_constraint`
//! uploads the arrays to the device, which IS a copy. This is that copy.
//!
//! `DataSet::constraint` is not a substitute and reading it would be a silent
//! wrong answer: `backend.rs` never assigns it, so it holds whatever scene build
//! left there. The pins the step must honor arrive only through
//! `update_constraint`.

use crate::data::{
    DataSet, FixPair, Floor, PullPair, Sphere, Stitch, TorqueGroup, TorqueGroupResult,
    TorqueVertex,
};

use ppf_cts_compute::{AllocLabel, Buffer, Device, ReadbackBuffer, StagedBuffer};
use super::scene::{Fatal, FatalResult};

/// The scene's fixed sizes, read once at `initialize()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sizes {
    pub vertices: usize,
    pub tets: usize,
    /// The WHOLE face array, which is not the membrane's range: a solid's
    /// surface triangles sit in it after the shell prefix. Kept beside the
    /// counts the driver indexes with, because a size assertion that reports
    /// the scene it was built for is the one that is readable, and because the
    /// vertex-face adjacency is checked against this length rather than the
    /// prefix.
    #[allow(dead_code)]
    pub faces: usize,
    /// The shell PREFIX of `faces`, which is what the membrane assembly walks
    /// and what `inv_rest2x2` is sized over. The two are equal exactly when the
    /// scene carries no solid, which is what makes confusing them invisible on
    /// every shell-only scene.
    pub shell_faces: usize,
    /// Every adjacent face pair in the mesh, which is NOT the set of bending
    /// elements: a pair whose either side is a solid's surface triangle carries
    /// no bending energy and is marked with bit 0 of `mesh.type.hinge`. The
    /// bending assembly ranges over the whole array and selects inside it, so
    /// this is the length its scratch is sized to.
    pub hinges: usize,
    /// The ROD PREFIX of `mesh.edge`, which is not that array's length: every
    /// face's edges follow the rods in it, because edge-edge contact needs
    /// them. They carry no stretch energy, so the rod layer walks this prefix.
    /// The two are equal exactly when the scene carries no shell and no solid,
    /// which is what makes confusing them invisible on a rod-only scene.
    pub rods: usize,
    /// How many cross-stitches the scene was BUILT with, which is what the
    /// fixed sparsity registered a 6x6 block set for and what the stitch
    /// scratch is sized over. A step whose constraint carries a different count
    /// is refused rather than resized, because the pattern it would push into
    /// was built for this one.
    pub stitches: usize,
    /// How many interior rod vertices the mesh has, which is how many
    /// three-node bending stencils the rod bending layer assembles.
    ///
    /// Derived from the vertex-edge and vertex-face adjacencies, both fixed at
    /// scene build, so the SITES are enumerated once at `initialize()` and only
    /// the per-iteration gate is re-applied. It is not a container length: a
    /// scene with no rods has none however many vertices and edges it carries.
    pub rod_bend_sites: usize,
    /// How many SAND grains the scene carries.
    ///
    /// DERIVED FROM THE SCENE, NOT FROM THE CONTACT LAYER: the host mirror of
    /// `grain_inv_inertia` is scanned once at `initialize()`, that value being
    /// positive for a grain and zero for every vertex of
    /// every non-SAND scene. Reading the count off the contact layer instead
    /// would make it zero under `disable-contact`, which deletes that layer,
    /// and a SAND scene keeps its grains whether or not vertex contact is
    /// assembled: the analytic colliders sit OUTSIDE that gate, so a floor
    /// still condenses spin into the solve and the post-solve integrate still
    /// has to run.
    pub grains: usize,
    #[allow(dead_code)]
    pub fixed_nnz: usize,
}

/// Seed only the COMMITTED pose from the scene, leaving the iterate alone.
///
/// The narrower half of [`reseed_positions`], for a fixture that sets the
/// iterate to something else on purpose: `positions` is the start-of-step pose
/// and tracks the scene, while `eval_x` is whatever the test is probing at.
#[cfg(test)]
pub(crate) fn reseed_committed(
    device: &mut impl Device,
    state: &mut SolverState,
    data: &crate::data::DataSet,
) {
    let n = 3 * state.sizes.vertices;
    if n == 0 {
        return;
    }
    // Safety: the scene is live and holds `state.sizes.vertices` triples.
    let pose = unsafe { crate::driver::state::slice_or_empty(data.vertex.curr.data as *const f32, n) };
    state
        .positions
        .seed(device, pose)
        .expect("the fixture seeds the committed pose");
}

/// The carrier plus the allocations behind it, for a fixture with no state.
///
/// The blocks are returned so the CALLER owns them: a handle names an
/// allocation, so one built from a temporary would dangle the moment the
/// temporary dropped.
#[cfg(test)]
pub(crate) struct TestMesh {
    pub face: Buffer<u32>,
    pub edge: Buffer<u32>,
    pub prop: Buffer<crate::data::VertexProp>,
    pub face_prop: Buffer<crate::data::FaceProp>,
    pub edge_prop: Buffer<crate::data::EdgeProp>,
    pub vertex_param: Buffer<crate::data::VertexParam>,
    pub face_param: Buffer<crate::data::FaceParam>,
    pub edge_param: Buffer<crate::data::EdgeParam>,
    pub fixed_index: Buffer<u32>,
    pub fixed_offset: Buffer<u32>,
    /// A real zero-length allocation, for the adjacencies a fixture omits.
    ///
    /// One buffer serves all six fields: they are all absent, and what a record
    /// needs from an absent table is a RESOLVABLE handle rather than a distinct
    /// one. `Handle::NONE` is not resolvable, which is the whole reason this
    /// field exists.
    pub empty_adjacency: Buffer<u32>,
    /// Allow Existing Intersections' table, staged from `DataSet::start_link`
    /// by the production staging, so a fixture installs links the way a scene
    /// does: by filling the dataset's field.
    pub start_link_index: Buffer<u32>,
    pub start_link_offset: Buffer<u32>,
}

/// One device block from a host slice of any record, for tests.
#[cfg(test)]
pub(crate) fn record_block<T: ppf_cts_compute::Pod + Default>(
    device: &mut impl Device,
    host: &[T],
    label: &'static str,
) -> Buffer<T> {
    let mut buffer = Buffer::none();
    buffer
        .size(device, host.len(), AllocLabel(label))
        .expect("the test allocation succeeds");
    buffer
        .write(device, 0, host)
        .expect("the test upload succeeds");
    buffer
}

#[cfg(test)]
    /// A test pattern's four arrays on the device.
    ///
    /// The production carrier comes from `SolverState`, which a unit test does
    /// not build, so a fixture owns its own and hands out the same shape.
    pub(crate) struct PatternDevice {
        index: ppf_cts_compute::Buffer<u32>,
        offset: ppf_cts_compute::Buffer<u32>,
        transpose_pair: ppf_cts_compute::Buffer<u32>,
        transpose_offset: ppf_cts_compute::Buffer<u32>,
        rows: u32,
    }

#[cfg(test)]
impl PatternDevice {
        pub(crate) fn of(device: &mut impl Device, pattern: &super::fixedcsr::FixedPattern<'_>) -> Self {
            Self {
                index: pattern_array(device, pattern.index, "test.pattern.index"),
                offset: pattern_array(device, pattern.offset, "test.pattern.offset"),
                transpose_pair: pattern_array(
                    device,
                    pattern.transpose_pair,
                    "test.pattern.tpair",
                ),
                transpose_offset: pattern_array(
                    device,
                    pattern.transpose_offset,
                    "test.pattern.toffset",
                ),
                rows: pattern.rows,
            }
        }

        pub(crate) fn refs(&self) -> super::fixedcsr::FixedPatternRefs {
            super::fixedcsr::FixedPatternRefs {
                index: self.index.span(0, self.index.len()),
                offset: self.offset.span(0, self.offset.len()),
                transpose_pair: self.transpose_pair.span(0, self.transpose_pair.len()),
                transpose_offset: self.transpose_offset.span(0, self.transpose_offset.len()),
                rows: self.rows,
            }
        }
    }

#[cfg(test)]
pub(crate) fn pattern_array(
        device: &mut impl Device,
        host: &[u32],
        label: &'static str,
    ) -> ppf_cts_compute::Buffer<u32> {
        let mut buffer = ppf_cts_compute::Buffer::<u32>::none();
        buffer
            .size(device, host.len(), ppf_cts_compute::AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
            .write(device, 0, host)
            .expect("the test upload succeeds");
        buffer
    }


#[cfg(test)]
impl TestMesh {
    pub fn refs(&self) -> crate::driver::contact::MeshRefs {
        crate::driver::contact::MeshRefs {
            face: self.face.handle(),
            edge: self.edge.handle(),
            vertex_prop: self.prop.handle(),
            face_prop: self.face_prop.handle(),
            edge_prop: self.edge_prop.handle(),
            vertex_param: self.vertex_param.handle(),
            face_param: self.face_param.handle(),
            edge_param: self.edge_param.handle(),
            fixed_index: self.fixed_index.handle(),
            fixed_offset: self.fixed_offset.handle(),
            // A TEST MESH CARRIES NO ADJACENCY, and it says so the way a real
            // scene without a table does: a zero `has_*` flag beside a REAL
            // zero-length handle. Not `Handle::NONE`, which the generated entry
            // resolves out of bounds before the body reads the flag.
            vertex_edge_index: self.empty_adjacency.span(0, 0),
            vertex_edge_offset: self.empty_adjacency.span(0, 0),
            has_vertex_edge: 0,
            vertex_face_index: self.empty_adjacency.span(0, 0),
            vertex_face_offset: self.empty_adjacency.span(0, 0),
            has_vertex_face: 0,
            edge_face_index: self.empty_adjacency.span(0, 0),
            edge_face_offset: self.empty_adjacency.span(0, 0),
            has_edge_face: 0,
            start_link: crate::driver::contact::StartLinkRefs {
                index: adjacency_handle(&self.start_link_index),
                offset: adjacency_handle(&self.start_link_offset),
                present: u32::from(self.start_link_offset.len() != 0),
            },
        }
    }
}

/// A raw pointer and a count as a slice, tolerating the EMPTY array.
///
/// `slice::from_raw_parts` is UNDEFINED BEHAVIOR on a null pointer even at
/// length zero, and an empty `DataSet` array is exactly that: nothing was
/// allocated, so `data` is null while `size` is 0. Rust's debug precondition
/// check traps it and the panic is NON-UNWINDING, so it aborts the whole test
/// binary rather than failing one case, which is what the fix-pinned-vertex
/// fixture did: its scene is one vertex over a floor and carries no faces at
/// all.
///
/// # Safety
/// `ptr` must be valid for `len` elements whenever `len` is nonzero.
#[cfg(test)]
pub(crate) unsafe fn slice_or_empty<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        // The ONE place this file may still spell it: everything else in the
        // driver's test code goes through the guard above.
        std::slice::from_raw_parts(ptr, len)
    }
}

/// Build [`TestMesh`] from a live `DataSet`.
///
/// # Safety
/// `data` must be live.
#[cfg(test)]
pub(crate) unsafe fn test_mesh_of(
    device: &mut impl Device,
    data: &crate::data::DataSet,
) -> TestMesh {
    let face = position_block_u32(
        device,
        slice_or_empty(
            data.mesh.mesh.face.data as *const u32,
            3 * data.mesh.mesh.face.size as usize,
        ),
        "test.mesh.face",
    );
    let edge = position_block_u32(
        device,
        slice_or_empty(
            data.mesh.mesh.edge.data as *const u32,
            2 * data.mesh.mesh.edge.size as usize,
        ),
        "test.mesh.edge",
    );
    let host = slice_or_empty(data.prop.vertex.data, data.prop.vertex.size as usize);
    let mut prop = Buffer::<crate::data::VertexProp>::none();
    prop.size(device, host.len(), AllocLabel("test.mesh.prop"))
        .expect("the test allocation succeeds");
    prop.write(device, 0, host).expect("the test upload succeeds");
    let mut empty_adjacency = Buffer::<u32>::none();
    empty_adjacency
        .size(device, 0, AllocLabel("test.mesh.empty_adjacency"))
        .expect("a zero-length allocation succeeds");

    let mut start_link_index = Buffer::<u32>::none();
    let mut start_link_offset = Buffer::<u32>::none();
    stage_adjacency(device, &data.start_link, &mut start_link_index,
                    &mut start_link_offset, "test.mesh.start_link")
        .expect("the fixture stages its link table");

    TestMesh {
        empty_adjacency,
        start_link_index,
        start_link_offset,
        face,
        edge,
        prop,
        face_prop: record_block(
            device,
            slice_or_empty(data.prop.face.data, data.prop.face.size as usize),
            "test.mesh.face_prop",
        ),
        edge_prop: record_block(
            device,
            slice_or_empty(data.prop.edge.data, data.prop.edge.size as usize),
            "test.mesh.edge_prop",
        ),
        vertex_param: record_block(
            device,
            slice_or_empty(
                data.param_arrays.vertex.data,
                data.param_arrays.vertex.size as usize,
            ),
            "test.mesh.vertex_param",
        ),
        face_param: record_block(
            device,
            slice_or_empty(
                data.param_arrays.face.data,
                data.param_arrays.face.size as usize,
            ),
            "test.mesh.face_param",
        ),
        edge_param: record_block(
            device,
            slice_or_empty(
                data.param_arrays.edge.data,
                data.param_arrays.edge.size as usize,
            ),
            "test.mesh.edge_param",
        ),
        fixed_index: {
            let pattern = crate::driver::fixedcsr::borrow_pattern(
                &data.fixed_index_table,
                &data.transpose_table,
            )
            .expect("the fixture builds a valid pattern");
            position_block_u32(device, pattern.index, "test.mesh.fixed_index")
        },
        fixed_offset: {
            let pattern = crate::driver::fixedcsr::borrow_pattern(
                &data.fixed_index_table,
                &data.transpose_table,
            )
            .expect("the fixture builds a valid pattern");
            position_block_u32(device, pattern.offset, "test.mesh.fixed_offset")
        },
    }
}

/// Re-seed a fixture's staged vertex props from the scene it has just edited.
///
/// THE SAME INVARIANT AS [`reseed_positions`], one array over. `allocate` stages
/// the props once, and a fixture that writes `fix_index`, `collider` or
/// `rest_bend_angle` on the `DataSet` afterwards leaves the device copy holding
/// the build-time values. The symptom names nothing: a bending force where the
/// test expects none, because the vertex the fixture prescribed is still free
/// on the device.
#[cfg(test)]
pub(crate) fn reseed_props(
    device: &mut impl Device,
    state: &mut SolverState,
    data: &crate::data::DataSet,
) {
    let n = state.prop_vertex.len();
    if n == 0 {
        return;
    }
    // Safety: the scene is live and carries at least `n` vertex props.
    let host = unsafe { crate::driver::state::slice_or_empty(data.prop.vertex.data, n) };
    state.prop_vertex.at().copy_from_slice(host);
    state
        .prop_vertex
        .upload(device)
        .expect("the fixture re-stages the props");
    // THE FACE PROPS AND MATERIALS TOO, because a kernel reads them on the
    // device now. `shell_stretch_terms` takes the face's own `FaceProp` and
    // `FaceParam` and applies the collider, fixed and PDRD gate there, so a
    // fixture that marks a collider after `allocate` is seen by the gate only
    // if these two travel with the vertex props. The failure is silent on a
    // backend that returns zero for an out-of-range read.
    let faces = state.prop_face.len();
    if faces > 0 {
        // Safety: the scene is live and carries at least `faces` face props.
        let host = unsafe { crate::driver::state::slice_or_empty(data.prop.face.data, faces) };
        state.prop_face.at().copy_from_slice(host);
        state
            .prop_face
            .upload(device)
            .expect("the fixture re-stages the face props");
    }
    let edges = state.prop_edge.len();
    if edges > 0 {
        // Safety: the scene is live and carries at least `edges` edge props.
        let host = unsafe { crate::driver::state::slice_or_empty(data.prop.edge.data, edges) };
        state.prop_edge.at().copy_from_slice(host);
        state
            .prop_edge
            .upload(device)
            .expect("the fixture re-stages the edge props");
    }
    let materials = state.param_face.len();
    if materials > 0 {
        // Safety: as above, for the face materials.
        let host =
            unsafe { crate::driver::state::slice_or_empty(data.param_arrays.face.data, materials) };
        state.param_face.at().copy_from_slice(host);
        state
            .param_face
            .upload(device)
            .expect("the fixture re-stages the face materials");
    }
    // THE TET RECORDS TRAVEL WITH THE REST, and they have to: the solid layer
    // reads `fixed`, `rest_excluded` and the material IN THE THREAD now, so a
    // fixture that flips one of those after `allocate` and does not re-stage
    // hands the kernel the build-time value. That is not a fixture-only
    // hazard; it is the same staleness `update_constraint` would produce.
    let tets = state.prop_tet.len();
    if tets > 0 {
        // Safety: the scene is live and carries at least `tets` tet props.
        let host = unsafe { crate::driver::state::slice_or_empty(data.prop.tet.data, tets) };
        state.prop_tet.at().copy_from_slice(host);
        state
            .prop_tet
            .upload(device)
            .expect("the fixture re-stages the tet props");
    }
    // THE EDGE MATERIALS TRAVEL TOO, for the reason the tet pair does: the rod
    // creep reads `plasticity` off this array IN THE THREAD now, so a fixture
    // that sets a material after `allocate` and does not re-stage hands the
    // kernel the build-time value.
    let edge_materials = state.param_edge.len();
    if edge_materials > 0 {
        // Safety: the scene is live and carries at least that many materials.
        let host = unsafe {
            crate::driver::state::slice_or_empty(data.param_arrays.edge.data, edge_materials)
        };
        state.param_edge.at().copy_from_slice(host);
        state
            .param_edge
            .upload(device)
            .expect("the fixture re-stages the edge materials");
    }
    let tet_materials = state.param_tet.len();
    if tet_materials > 0 {
        // Safety: as above, for the tet materials.
        let host = unsafe {
            crate::driver::state::slice_or_empty(data.param_arrays.tet.data, tet_materials)
        };
        state.param_tet.at().copy_from_slice(host);
        state
            .param_tet
            .upload(device)
            .expect("the fixture re-stages the tet materials");
    }
}

#[cfg(test)]
/// Re-stage the HINGE records and types after a fixture rewrote the scene.
///
/// `allocate` stages these once because they are build-time data: the props
/// carry geometry and exclusion flags, the params the material, and the types
/// are topology. A FIXTURE can rewrite any of them AFTER that, which production
/// cannot, and the bending stiffness now reads all three off the device, so a
/// test that marks a hinge `fixed`, `collider` or solid-surface and does not
/// call this is asserting against the pose `allocate` saw.
///
/// This is `reseed_props` for the hinge side and exists for the same reason.
pub(crate) fn reseed_hinge(
    device: &mut impl Device,
    state: &mut SolverState,
    data: &crate::data::DataSet,
) {
    let hinges = state.prop_hinge.len();
    if hinges > 0 {
        // Safety: the scene is live and carries at least `hinges` hinge props.
        let host = unsafe { slice_or_empty(data.prop.hinge.data, hinges) };
        state.prop_hinge.at().copy_from_slice(host);
        state
            .prop_hinge
            .upload(device)
            .expect("the fixture re-stages the hinge props");
    }
    let materials = state.param_hinge.len();
    if materials > 0 {
        // Safety: the scene is live and carries at least `materials` params.
        let host = unsafe { slice_or_empty(data.param_arrays.hinge.data, materials) };
        state.param_hinge.at().copy_from_slice(host);
        state
            .param_hinge
            .upload(device)
            .expect("the fixture re-stages the hinge materials");
    }
    let kinds = state.hinge_kind.len();
    if kinds > 0 {
        // Safety: the scene is live and owns its hinge type array.
        let kind: &[u8] = unsafe { crate::driver::scene::slice(&data.mesh.ttype.hinge) };
        for (slot, value) in state.hinge_kind.at().iter_mut().zip(kind.iter()) {
            *slot = u32::from(*value);
        }
        state
            .hinge_kind
            .upload(device)
            .expect("the fixture re-stages the hinge types");
    }
}

/// Re-seed a fixture's staged inverse rest matrices from the scene.
///
/// The same invariant as [`reseed_props`], one array over: a fixture that sets
/// a rest matrix on the `DataSet` after `allocate` leaves the staged copy
/// holding the build-time one.
#[cfg(test)]
pub(crate) fn reseed_inv_rest(
    device: &mut impl Device,
    state: &mut SolverState,
    data: &crate::data::DataSet,
) {
    let n2 = state.inv_rest2x2.len();
    if n2 > 0 {
        // Safety: the scene carries `n2` floats of shell rest matrices.
        let host = unsafe { crate::driver::state::slice_or_empty(data.inv_rest2x2.data as *const f32, n2) };
        state.inv_rest2x2.seed(device, host).expect("the fixture re-stages the shell rest matrices");
    }
    let n3 = state.inv_rest3x3.len();
    if n3 > 0 {
        // Safety: as above, for the tet rest matrices.
        let host = unsafe { crate::driver::state::slice_or_empty(data.inv_rest3x3.data as *const f32, n3) };
        state.inv_rest3x3.seed(device, host).expect("the fixture re-stages the tet rest matrices");
    }
}

/// A device block of vertex properties, for tests.
#[cfg(test)]
pub(crate) fn position_block_prop(
    device: &mut impl Device,
    host: &[crate::data::VertexProp],
    label: &'static str,
) -> Buffer<crate::data::VertexProp> {
    let mut buffer = Buffer::none();
    buffer
        .size(device, host.len(), AllocLabel(label))
        .expect("the test allocation succeeds");
    buffer
        .write(device, 0, host)
        .expect("the test upload succeeds");
    buffer
}

/// The scene's device-resident static arrays, as a carrier, for tests.
///
/// A fixture that calls a contact or collider entry point needs the same
/// handles a step would pass, and this builds them from the state the fixture
/// already allocated rather than from its own copies.
/// Re-stage the fixed pattern from a scene a fixture has CHANGED.
///
/// `allocate` stages the pattern once, which is right in production because the
/// pattern is immutable after scene build. A fixture that rewrites
/// `fixed_index_table` afterwards leaves the staged copy naming the OLD one,
/// and the two then disagree: the host check walks the new pattern while the
/// kernel looks a block up in the old one, so a deliberately incomplete pattern
/// silently finds the slot it was supposed to be missing. Same shape as
/// [`reseed_committed`] and [`reseed_props`], for the same reason.
#[cfg(test)]
pub(crate) fn reseed_pattern<D: Device>(
    device: &mut D,
    state: &mut SolverState,
    data: &crate::data::DataSet,
) {
    // Safety: the fixture's tables are live for the call.
    let pattern = unsafe {
        super::fixedcsr::borrow_pattern(&data.fixed_index_table, &data.transpose_table)
    }
    .expect("the fixture's pattern is valid");
    stage_static_u32(device, &mut state.fixed_index, pattern.index, "csr.fixed_index")
        .and_then(|()| stage_static_u32(device, &mut state.fixed_offset, pattern.offset, "csr.fixed_offset"))
        .and_then(|()| stage_static_u32(device, &mut state.transpose_pair, pattern.transpose_pair, "csr.transpose_pair"))
        .and_then(|()| stage_static_u32(device, &mut state.transpose_offset, pattern.transpose_offset, "csr.transpose_offset"))
        .expect("the fixture re-stages its pattern");
    state.fixed_rows = pattern.rows;
}

#[cfg(test)]
pub(crate) fn mesh_refs_of(state: &mut SolverState) -> crate::driver::contact::MeshRefs {
    // DELEGATES rather than rebuilding. This is the fourth place that built the
    // carrier by hand, and adding one field to `MeshRefs` broke all four; there
    // is now one builder and the fixtures use it.
    state.refs()
}

/// The `u32` counterpart of [`position_block`], for topology in tests.
#[cfg(test)]
pub(crate) fn position_block_u32(
    device: &mut impl Device,
    host: &[u32],
    label: &'static str,
) -> Buffer<u32> {
    let mut buffer = Buffer::<u32>::none();
    buffer
        .size(device, host.len(), AllocLabel(label))
        .expect("the test allocation succeeds");
    buffer
        .write(device, 0, host)
        .expect("the test upload succeeds");
    buffer
}

/// Re-seed a fixture's device positions from the scene it has just deformed.
///
/// THE INVARIANT A TEST FIXTURE HAS TO KEEP. `allocate` seeds `positions`,
/// `positions_prev` and `eval_x` from the scene ONCE, which is right in
/// production because the scene does not move afterward. A fixture DOES move
/// it, so without this the kernels read the build-time pose while the test
/// reasons about the deformed one, and the symptom is a zero force or a
/// stitch whose endpoints coincide rather than anything naming a stale buffer.
#[cfg(test)]
pub(crate) fn reseed_positions(
    device: &mut impl Device,
    state: &mut SolverState,
    data: &crate::data::DataSet,
) {
    let n = 3 * state.sizes.vertices;
    if n == 0 {
        return;
    }
    // Safety: the scene is live and holds `state.sizes.vertices` triples.
    let pose = unsafe { crate::driver::state::slice_or_empty(data.vertex.curr.data as *const f32, n) };
    state
        .positions
        .seed(device, pose)
        .expect("the fixture re-seeds the committed pose");
    state
        .eval_x
        .seed(device, pose)
        .expect("the fixture re-seeds the iterate");
}

/// A device allocation holding a host slice of position components, for tests.
///
/// THE POSITIONS ARE DEVICE-RESIDENT, so a test cannot hand a kernel a `&[i32]`
/// directly: it needs an allocation for the handle to name. This is that, in
/// one place rather than once per test module.
#[cfg(test)]
pub(crate) fn position_block(
    device: &mut impl Device,
    host: &[f32],
    label: &'static str,
) -> Buffer<f32> {
    let mut buffer = Buffer::<f32>::none();
    buffer
        .size(device, host.len(), AllocLabel(label))
        .expect("the test allocation succeeds");
    buffer
        .write(device, 0, host)
        .expect("the test upload succeeds");
    buffer
}

/// The per-run state the driver owns.
/// The fold's block width, the same 256 the float fold uses.
pub(crate) const DOF_FOLD_WIDTH: usize = 256;

#[derive(Default)]
pub struct SolverState {
    pub sizes: Sizes,

    /// The committed positions and the previous step's, device-resident.
    ///
    /// THE SPLIT IS FORCED BY THE ABI rather than chosen:
    /// `data.vertex.curr` is a `CVec<Vec3f>` in the `repr(C)`
    /// `DataSet`, shared with the frontend, so it cannot itself be a device
    /// allocation. `fetch` downloads into it, which is that function's whole
    /// contract and the reason it is not empty here.
    pub positions: ReadbackBuffer<f32>,
    pub positions_prev: ReadbackBuffer<f32>,

    /// The Newton iterate, as flat position components (`3 * vertices` int32).
    ///
    /// IT MOVES WITH THE POSITIONS: two records fill one field from this at one
    /// call site and from `data.vertex.curr` at another, and a field is ONE
    /// type in the Rust twin.
    pub eval_x: ReadbackBuffer<f32>,
    /// This step's implicit target, and, from `B20` onward, the iterate's
    /// position at the start of the current iteration.
    ///
    /// ONE BUFFER SERVING TWO ROLES, deliberately: `target` is clobbered at B20
    /// to hold the pre-step `eval_x`, and every CONTINUING iteration rebuilds
    /// it with `compute_target` at the bottom of the loop. A driver that gave
    /// the two roles separate buffers would still have to rebuild the target,
    /// and would hide the fact that the line search reads the OLD positions out
    /// of it.
    pub target: ReadbackBuffer<f32>,
    /// Start-of-step velocity, `3 * vertices` floats. A DEVICE ALLOCATION: the
    /// pass that forms it is the only code that touches it, so nothing reads it
    /// back and nothing seeds it.
    pub velocity: Buffer<f32>,
    /// The external force field: its installed inputs and its two per-vertex
    /// outputs, the acceleration the target adds beside gravity and the air
    /// velocity the drag adds to the scene wind.
    pub field: super::force_field::ForceField,
    /// The Newton right-hand side, `3 * vertices` floats.
    pub force: ReadbackBuffer<f32>,
    /// THE RESIDUAL EVERY CONTACT'S FRICTION ANCHOR IS READ OFF, `3 * vertices`
    /// floats, a COPY of `force` taken twice per assembly: once before the
    /// analytic and collision-mesh contacts, and again before the self-contact
    /// pass, so the second copy carries the first pass's forces.
    ///
    /// IT IS A SEPARATE BUFFER BECAUSE EVERY CONTACT WRITES INTO `force` WHILE
    /// READING THIS. Both halves of the assembly deposit straight into `force`,
    /// so the anchor each friction term reads has to be a snapshot taken before
    /// that half ran; without the copy a term would anchor on forces its own
    /// pass had already added. A DEVICE ALLOCATION: only the copy and
    /// the kernels that read it touch it, so nothing reads it back and nothing
    /// seeds it.
    pub residual: Buffer<f32>,
    /// The search direction, `3 * vertices` floats.
    pub dx: Buffer<f32>,
    /// `C`, the block diagonal, `9 * vertices` floats.
    pub diagonal: Buffer<f32>,
    /// The preconditioner's diagonal and its inverse, `9 * vertices` floats.
    pub precond_diagonal: Buffer<f32>,
    /// The additive Schwarz preconditioner, when the scene asks for one.
    ///
    /// Empty until a solve with `precond=schwarz` builds it, and rebuilt
    /// only when the row count changes: the factorization is quadratic in
    /// the dense-block cap and the partition is a sequential host sweep.
    pub schwarz: super::schwarz::State,
    pub precond_inverse: Buffer<f32>,
    /// Which vertices are Dirichlet-removed this step, one per vertex.
    ///
    /// KERNEL-WRITTEN, WHICH IS WHY IT IS A PLAIN DEVICE BUFFER. `step.rs`
    /// dispatches `step.dof_mask` over the device vertex props and then folds
    /// the mask on the device, so the only thing about it that crosses the seam
    /// is the one-word total below.
    pub dof_mask: Buffer<u32>,
    /// The fold's intermediate levels and its one-word result, the number of
    /// removed vertices.
    pub dof_fold: Buffer<u32>,
    pub dof_total: ReadbackBuffer<u32>,
    /// Per-vertex scratch for the max reductions (`max_u`, the reach from the
    /// origin, `max_dx`) and for the line search's per-vertex time of impact.
    ///
    /// READBACKS: a kernel writes one value per vertex and the host folds the
    /// array to a maximum. The verdict is a function of the positions the
    /// device holds rather than of anything the host built, which is the test
    /// [`ppf_cts_compute::ReadbackBuffer`] states, and the fold itself has no
    /// lane in the neutral vocabulary yet.
    ///
    /// AND THE MISSING LANE IS WHY THE WHOLE ARRAY MOVES. A device max would
    /// move four bytes; the readback here moves one float per vertex because
    /// `kernels/primitives/reduce.kernel.cpp` declares `warp_reduce` and
    /// `block_reduce` as `[[seam::device_fn]]` and declares no
    /// `[[seam::entry]]`, so there is nothing for `super::launch` to bind. The
    /// buffer type is right for what the driver can express today and the
    /// TRANSFER is the divergence.
    pub scalar: ReadbackBuffer<f32>,
    pub scalar_b: ReadbackBuffer<f32>,

    /// `B`, the fixed-pattern Hessian's values, and the `tmp_fixed` snapshot
    /// the contact stiffness reads.
    ///
    /// TWO REAL BUFFERS, not an evaluate / scatter split. `tmp_fixed` is a
    /// second matrix over the same pattern, snapshotted once the elastic and
    /// stitch layers have landed, and the contact assembly reads it as its
    /// stiffness reference while writing into `fixed_values`.
    pub fixed_values: ReadbackBuffer<f32>,
    pub tmp_fixed_values: ReadbackBuffer<f32>,
    /// The pattern the two matrices' sparsity was last validated on.
    ///
    /// `FixedCsr::adopt_validated` walks the whole sparsity when this does not
    /// match the pattern being wrapped, and skips the walk when it does. The
    /// tables are borrowed from the `DataSet`, which `advance` holds by shared
    /// reference, so after the first iteration of a run this always matches.
    pub fixed_pattern_checked: Option<super::fixedcsr::PatternFingerprint>,

    /// The tet elastic pipeline's per-element scratch. Every array is one
    /// element's worth per tet, in tet order.
    pub tet: TetScratch,
    /// The shell membrane pipeline's per-element scratch, one element's worth
    /// per SHELL face, in face order. Sized over the shell prefix and not over
    /// the face array; see [`Sizes::shell_faces`].
    pub face: FaceScratch,
    /// The shell bending pipeline's per-element scratch, one element's worth
    /// per hinge, in hinge order.
    pub hinge: HingeScratch,
    /// The rod stretch pipeline's per-element scratch, one element's worth per
    /// ROD, in edge order. Sized over the rod prefix and not over the edge
    /// array; see [`Sizes::rods`].
    /// The rod bending pipeline's per-element scratch, one element's worth per
    /// interior rod vertex, in ascending vertex order.
    pub rod_bend: RodBendScratch,
    /// The cross-stitch layer's per-element scratch, one element's worth per
    /// stitch, in the authored order of `Constraint::stitch`.
    pub stitch: StitchScratch,
    /// The shell strain limiter's per-element scratch, one element's worth per
    /// SHELL face, in face order.
    pub face_strain: FaceStrainScratch,
    /// The rod strain limiter's per-element scratch, one element's worth per
    /// ROD, in edge order.
    pub rod_strain: RodStrainScratch,
    /// The `max_sigma` indicator's scratch, measured once per step before the
    /// Newton loop.
    pub stretch: StretchScratch,

    /// The plastic creep's per-element scratch, and the record of which of the
    /// four creeps this scene runs. Every group inside it is allocated only
    /// when its own kernel is enabled, so a scene with no plastic material
    /// carries nothing.
    pub plastic: super::plasticity::PlasticScratch,

    /// The PCG solve's own vectors, sized once for the same reason the rest are.
    pub pcg: super::pcg::Workspace,

    /// The aggregate lock's scratch and the two extra vectors its solve needs.
    ///
    /// SIZED ONLY WHEN THE SCENE CARRIES A LOCK, so an unlocked scene allocates
    /// none of it: `Buffer::size` at zero is a no-op and every projector method
    /// returns early on a zero group count.
    /// The three per-vertex blocks a SAND grain's spin is condensed out of.
    ///
    /// PER VERTEX, NOT PER GRAIN: the analytic contact writes every vertex's
    /// slot so the condense pass reads this iteration rather than the last, and
    /// a non-grain leaves zeros, which is what that pass tests.
    pub grain_angular: Buffer<f32>,
    pub grain_coupling: Buffer<f32>,
    pub grain_rotational: Buffer<f32>,
    /// The three per-vertex friction accumulators the grain-grain narrow phase
    /// sums into, and the post-solve integrate consumes.
    ///
    /// HERE RATHER THAN ON THE CONTACT LAYER, which is what `disable-contact`
    /// does not build. The analytic colliders are assembled OUTSIDE that gate,
    /// so a grain resting on a floor has real spin to integrate; owned by the
    /// contact layer these three would have nowhere to accumulate it under that
    /// flag, and such a scene would have to be refused by name rather than run
    /// with its granular physics silently absent. All three are sized per
    /// vertex with the scene, independent of any layer.
    pub grain_torque: Buffer<f32>,
    pub grain_stiffness: Buffer<f32>,
    pub grain_normal: Buffer<f32>,
    /// The scene's per-vertex inverse rolling inertia, staged once: non-zero is
    /// what makes a vertex a grain.
    pub grain_inv_inertia: StagedBuffer<f32>,
    /// The grain's own angular state. `omega` is written by the recover and the
    /// integrate and read by the next step's contact, so it is the one piece of
    /// SAND state that carries ACROSS steps; `previous_omega` is the start-of-
    /// step snapshot the condense's momentum term reads.
    /// PLAIN DEVICE BUFFERS, not staged: the recover and the integrate WRITE
    /// them and the next step's contact narrow phase reads them, so the host is
    /// only their initial author. `SolverState` is their single owner for the
    /// reason the pin indices are staged per step: a second copy inside the
    /// contact layer would be seeded once and never see an integrate.
    pub grain_omega: Buffer<f32>,
    pub grain_omega_prev: Buffer<f32>,
    /// Per grain, the inverse inertia the reduced solve divides by, which is a
    /// different quantity from the rolling inertia above.
    pub grain_inv_inertia_center: StagedBuffer<f32>,

    /// The PDRD reduction's topology and its device image, built once when the
    /// scene is allocated. `map.n_bodies` is zero for every scene that carries
    /// no rigid body, and every PDRD dispatch is gated on it.
    pub rigid: super::rigid_map::RigidMap,
    pub rigid_staged: super::rigid_map::Staged,
    /// Each body's fitted rigid state, rewritten by the polar fit every Newton
    /// iteration.
    pub rigid_state: Buffer<crate::data::PdrdRigidState>,
    /// The RUNNING rotation, nine floats per body, and the per-iteration
    /// rotation increment the solve exports.
    ///
    /// `running_rotation` PERSISTS ACROSS FRAMES and is the reason the rigidify
    /// is anchored: it is the rotation the solve has actually applied, composed
    /// over the run, so refitting to it cannot accumulate the polar fit's own
    /// error as drift.
    pub rigid_running_rotation: Buffer<f32>,
    /// Whether the running rotation has been seeded from a fitted pose yet.
    ///
    /// THE SEED IS ONCE PER RUN, which this flag latches on the first step it
    /// is taken. After the first step the running rotation is
    /// the composition of the increments the line search actually accepted, and
    /// re-fitting it to a contact-sheared pose is exactly the drift the anchored
    /// rigidify exists to avoid.
    pub rigid_rotation_seeded: bool,
    pub rigid_rotation_step: Buffer<f32>,
    /// The preconditioner's assembled 6x6 blocks and their factors, 36 floats
    /// per body each.
    pub rigid_blocks: ReadbackBuffer<f32>,
    pub rigid_factor: StagedBuffer<f32>,
    /// Twelve floats per body for the fit's accumulators and three for the
    /// rigidify's centroid.
    pub rigid_fit_scratch: Buffer<f32>,
    pub rigid_centroid: Buffer<f32>,
    /// The rigidify commit's TARGET POSE, one entry per vertex like `eval_x`.
    ///
    /// It is a second position buffer rather than an in-place rewrite because
    /// the commit is CCD-FILTERED: the sweep needs both endpoints live at once,
    /// the iterate it starts from and the rigid image it moves toward, and the
    /// fraction it returns is then applied between them. Rigidifying in place
    /// would destroy the start pose the sweep is defined against.
    ///
    /// It carries a readback for the same reason `eval_x` does: the host walks
    /// both endpoints during the sweep.
    pub rigid_target: ReadbackBuffer<f32>,
    /// The reduced solve's own vectors.
    pub pcg_rigid: super::pcg::RigidWorkspace,

    pub lock: super::lock::Scratch,
    pub pcg_locked: super::pcg::LockedWorkspace,

    /// The lock arrays as device allocations, staged once: the per-vertex group
    /// index, the group records and the start pose each group's drift is
    /// measured against.
    pub translation_lock: StagedBuffer<crate::data::TranslationLock>,
    pub translation_lock_index: StagedBuffer<u32>,
    pub translation_lock_initial: StagedBuffer<f32>,
    /// The two accumulators `translation_lock::check_invariant` reduces into,
    /// one group-length pair allocated once here rather than per call.
    ///
    /// THEY ARE DIFFERENT KINDS AND THAT IS WHY THE TYPES DIFFER. The drift is
    /// a mass-weighted SUM of perpendicular displacement, three floats per
    /// group. The displacement is a MAXIMUM of magnitudes, and no backend has a
    /// float atomic maximum, so it is stored as the IEEE-754 bit pattern of a
    /// non-negative float, which orders the same way as the float; the host
    /// converts it back after the readback so that no neutral body spells a
    /// bitcast. `solver/translation_lock_check.kernel.cpp` carries the rule.
    ///
    /// Both are allocated once with the scene rather than per call, so a check
    /// inside a Newton iteration allocates nothing.
    /// The per-object statistics channel's three arrays.
    ///
    /// THE TWO INDEX ARRAYS ARE ITS OWN AND ARE NOT `VertexProp::object_index`.
    /// They are a DIFFERENT index space, and conflating the two indexes the
    /// wrong array with nothing to catch it: this is the one channel in the
    /// contact kernels whose index is neither bounds-checked nor in the
    /// caller's own index space. The dynamic and static vertex spaces are
    /// separate again within them.
    ///
    /// The counter is kernel-written and host-read, so it is a readback; the
    /// two index arrays are host-written and kernel-read, so they are staged.
    pub statistics_object_index: StagedBuffer<u32>,
    pub statistics_static_object_index: StagedBuffer<u32>,
    pub statistics_contact_count: ReadbackBuffer<u32>,
    /// The device fold's levels; see [`super::reduce::DeviceFold`]. The step's
    /// maximum displacement, reach and strain ratios are reduced through it
    /// rather than downloaded and folded on the host.
    pub fold: super::reduce::DeviceFold,
    pub lock_drift: ReadbackBuffer<f32>,
    pub lock_max_displacement: ReadbackBuffer<u32>,
    /// `fix_index` per vertex, which `compute_target` reads. KERNEL-WRITTEN,
    /// by `vertex_fix_index_from_records` at the top of a step, and read by the
    /// three `compute_target` calls in that step. Every production reader takes
    /// a `handle()`; the host mirror exists so a test can read the write back.
    ///
    /// IT IS A SECOND DEVICE COPY OF ONE FIELD of [`Self::prop_vertex`], and it
    /// exists because `compute_target_seed`
    /// (`kernels/main/target.kernel.cpp`) declares
    /// `[[seam::device]] [[seam::gather]] const unsigned *fix_index`, and a
    /// record addresses one allocation per field. Removing the copy means
    /// changing that entry to gather `[[seam::pod(44)]] const VertexProp *prop`
    /// and read the field in the body, which is a kernel-side change plus
    /// its three call sites.
    pub fix_index: ReadbackBuffer<u32>,

    /// This step's pins, copied from the incoming `Constraint`.
    pub fix: Buffer<FixPair>,
    pub pull: StagedBuffer<PullPair>,

    /// This step's torque groups and their members, copied from the incoming
    /// `Constraint` beside the pins and for the same reason.
    ///
    /// STAGED, because the host writes them and only kernels read them. The
    /// GROUPS are read by the frame pre-pass and the MEMBERS by both that pass
    /// and the momentum row, which is why the member array outlives the
    /// dispatch that walks it.
    pub torque_group: StagedBuffer<TorqueGroup>,
    pub torque_vertex: StagedBuffer<TorqueVertex>,

    /// The per-group frame the pre-pass writes and the momentum row reads.
    ///
    /// A PLAIN DEVICE BUFFER: nothing on the host ever looks at it. It is an
    /// intermediate between two dispatches of one step, recomputed at the top
    /// of every assembly because the frame is a function of the ITERATE, which
    /// moves under the Newton loop.
    pub torque_result: Buffer<TorqueGroupResult>,
    /// This step's analytic colliders, copied from the same record and for the
    /// same reason: `backend.rs` frees the previous `Constraint` as it assigns
    /// the new one, so a handle kept from `update_constraint` to `advance()`
    /// would address freed memory.
    /// Both are STAGED: the host rebuilds them from the schedule every step and
    /// only the constraint pass reads them, so the mirror is the authority and
    /// `handle()` refuses until this step's `upload` has run. That refusal is
    /// what stops a step assembling against the previous step's colliders.
    pub sphere: StagedBuffer<Sphere>,
    pub floor: StagedBuffer<Floor>,

    /// THE MESH TOPOLOGY, staged once per scene and never refreshed.
    ///
    /// These four are the scene's connectivity, and they are the one class of
    /// `DataSet` array a driver-owned device copy can hold without risking a
    /// stale mirror: the topology is written when the scene is built and by
    /// nothing in the step loop, so there is no host write that could dirty
    /// them behind the upload. Contrast `prop.vertex`, which plasticity writes
    /// through `scene::slice_mut` every step and which therefore stays where
    /// the host owns it.
    ///
    /// Each holds its WHOLE array in `u32` units. The energies that run over a
    /// prefix (rods within `edge`, shell faces within `face`) take a `span`
    /// rather than a second allocation.
    /// The per-vertex properties, device-resident.
    ///
    /// A `StagedBuffer` BECAUSE THE HOST STILL WRITES THEM: the plastic creep
    /// mutates `rest_bend_angle` every step. `at()` hands out the host slice
    /// and marks the device copy stale, and `handle()` PANICS if a dispatch
    /// names it before `upload()`, so a forgotten upload traps rather than
    /// dispatching the previous contents.
    pub prop_vertex: StagedBuffer<crate::data::VertexProp>,
    /// The per-face and per-edge properties and the three element param arrays.
    /// Staged for the same reason as the vertex props: build-time today, but a
    /// staged buffer traps a forgotten upload where a plain one would not.
    /// The fixed sparsity PATTERN, device-resident.
    ///
    /// NO STALENESS RISK AT ALL, which is why this one is a plain `Buffer` and
    /// why the host readers keep reading the `DataSet` copy: the pattern is
    /// built once at scene build and never written again, so the host array and
    /// the device copy agree for the whole run by construction.
    pub fixed_index: Buffer<u32>,
    pub fixed_offset: Buffer<u32>,
    pub transpose_pair: Buffer<u32>,
    pub transpose_offset: Buffer<u32>,
    /// The fixed pattern's row count, from the SAME `borrow_pattern` that
    /// staged the four arrays above.
    ///
    /// STORED RATHER THAN RE-DERIVED, because the alternative is to read it off
    /// `sizes.vertices`, and that is a SECOND SOURCE for a number the pattern
    /// already carries: `borrow_pattern` cross-checks this against
    /// `offset.len() - 1` and against the transpose table, and nothing would
    /// tie a separately-derived vertex count to either.
    pub fixed_rows: u32,

    /// The inverse rest matrices, as flat floats: four per shell face, nine per
    /// tet, which is how every record that names them reads them.
    ///
    /// A `StagedBuffer`, and the census is wrong about why. It records this as
    /// a buffer BOTH the host and a kernel write, with no type serving that.
    /// The kernel writes a SCRATCH (`plastic::face_inverse_rest`) and the HOST
    /// then scatters the rows that yielded into this array, so both writers are
    /// host writers: the creep's selective scatter and a checkpoint restore.
    /// `at()` dirties, `upload()` publishes, and `handle()` panics in between.
    /// KERNEL-WRITTEN AND HOST-READ, which is what picks the type now that the
    /// plastic commit is a kernel. It is also HOST-written, at the scene build
    /// and at a checkpoint restore, which is why it is a `ReadbackBuffer` with
    /// a `seed` rather than a `StagedBuffer`: a both-written buffer takes the
    /// type whose WRITER is the kernel, and the seed is what serves the other
    /// writer. `fetch_inv_rest` owes a download because of this.
    pub inv_rest2x2: ReadbackBuffer<f32>,
    pub inv_rest3x3: ReadbackBuffer<f32>,

    pub prop_face: StagedBuffer<crate::data::FaceProp>,
    /// The per-hinge properties, staged so the bending stiffness reads its
    /// geometry and its exclusion flags off the device.
    ///
    /// `plasticity.rs` creeps `rest_angle` through `scene::slice_mut` and
    /// re-stages this the way it re-stages `prop_vertex`; a write it forgot
    /// would be LOUD, since `at()` marks the mirror stale and `handle()`
    /// refuses while it is.
    pub prop_hinge: StagedBuffer<crate::data::HingeProp>,
    pub param_hinge: StagedBuffer<crate::data::HingeParam>,
    /// The per-hinge element type, widened to `u32` and staged ONCE.
    ///
    /// Bit 0 marks a hinge on a SOLID's surface, which carries no shell bending
    /// energy because the tet does. It is topology and never changes.
    pub hinge_kind: StagedBuffer<u32>,
    pub prop_edge: StagedBuffer<crate::data::EdgeProp>,
    /// The tet props and materials on the device, so the solid layer gathers
    /// each element's material in its own thread rather than on the host every
    /// Newton iteration.
    pub prop_tet: StagedBuffer<crate::data::TetProp>,
    pub param_tet: StagedBuffer<crate::data::TetParam>,
    pub param_vertex: StagedBuffer<crate::data::VertexParam>,
    pub param_face: StagedBuffer<crate::data::FaceParam>,
    pub param_edge: StagedBuffer<crate::data::EdgeParam>,

    pub mesh_face: StagedBuffer<u32>,
    pub mesh_edge: StagedBuffer<u32>,
    /// The rod stretch and strain deposit's precomputed CSR slots, four per
    /// edge, or EMPTY when `PPF_SLOT_REPLAY=0` asked for the search path.
    /// `builder.rs` computes it once for the scene, and the deposit branches on
    /// whether it is present.
    pub edge_hess_slots: StagedBuffer<u32>,
    /// The shell bending deposit's precomputed CSR slots, sixteen per hinge in
    /// the same remapped `(2,1,0,3)` order [`HingeScratch::remapped`] carries.
    pub hinge_hess_slots: StagedBuffer<u32>,
    /// The tet elastic deposit's precomputed CSR slots, sixteen per tet.
    pub tet_hess_slots: StagedBuffer<u32>,
    /// The rod bending deposit's precomputed CSR slots, nine per site.
    pub rod_bend_hess_slots: StagedBuffer<u32>,
    pub mesh_tet: StagedBuffer<u32>,
    pub mesh_hinge: StagedBuffer<u32>,
    /// The three neighbor adjacencies, each as its index array and its row
    /// offsets.
    ///
    /// STAGED RATHER THAN OWNED-BY-KERNEL, because the host WRITES them and a
    /// kernel only reads them: `marshal_neighbor` builds all three once at
    /// scene build and nothing rewrites them during a run.
    ///
    /// AN ABSENT TABLE STAYS ABSENT. A scene need not carry every adjacency,
    /// and the visitors branch on a `has_*` flag rather than on a null pointer,
    /// because a body cannot ask whether a buffer is null on a target where a
    /// buffer cannot be null. So an absent table leaves its buffer unsized and
    /// the record takes `Handle::NONE`, which is the one value that names
    /// nothing rather than arena 0 offset 0.
    pub neighbor_vertex_edge_index: Buffer<u32>,
    pub neighbor_vertex_edge_offset: Buffer<u32>,
    pub neighbor_vertex_face_index: Buffer<u32>,
    pub neighbor_vertex_face_offset: Buffer<u32>,
    pub neighbor_edge_face_index: Buffer<u32>,
    pub neighbor_edge_face_offset: Buffer<u32>,
    /// Allow Existing Intersections' vertex link table, staged once like the
    /// adjacencies above and for the same reason: `builder.rs` builds it at
    /// scene build and nothing rewrites it. A scene that linked nothing stages
    /// real zero-length arrays and reports the table absent.
    pub start_link_index: Buffer<u32>,
    pub start_link_offset: Buffer<u32>,
    /// Staging for the index lists the pin API hands in, and for the positions
    /// it reads back.
    ///
    /// PERSISTENT RATHER THAN PER-CALL. These serve `override_velocity`,
    /// `override_angular_velocity` and `gather_current_positions`, whose
    /// `indices` and `out` arrive from the frontend as host slices with no
    /// allocation behind them. `Buffer::size` grows only past CAPACITY, so a
    /// list no longer than the last one costs no allocation at all.
    pub seed_indices: Buffer<u32>,
    pub seed_out: ReadbackBuffer<f32>,
    /// The staging one `FixedCsr::push_blocks` uploads through.
    ///
    /// ONE AREA FOR THE WHOLE DRIVER, because only one push runs at a time and
    /// three different scratch structs feed it: the tet, face and rod passes
    /// through `state`, the contact narrow phase and the collider. Each builds
    /// its triple in a host `Vec` and the push copies it here, so the buffers
    /// are sized once and grown as a chunk gets wider rather than allocated per
    /// call.
    pub push: PushStaging,

    /// The analytic collider layer. Allocated on EVERY scene, not only one that
    /// wants contact: a sphere and a floor need no bounding hierarchy and no
    /// dynamic sparsity, so this driver assembles them outside the
    /// `disable-contact` gate.
    pub analytic: Option<super::collider::Analytic>,

    /// The contact subsystem: the three trees, the candidate staging and the
    /// dynamic Hessian.
    ///
    /// `None` on a scene that set `disable-contact`, and the driver reads that
    /// as "no broad phase, no barrier, no CCD filter and no intersection gate"
    /// rather than as an empty contact set. The distinction is load-bearing: a
    /// scene that DOES want contact and finds this `None` would run without the
    /// penetration guarantee, so the driver stops instead.
    pub contact: Option<super::contact::Contact>,
}

/// The per-tet arrays the elastic assembly walks, one stage at a time.
///
/// STAGED RATHER THAN FUSED, which is a deliberate difference from
/// `embed_tet_force_hessian`. That function runs the whole chain inside one
/// device thread; this backend runs each stage over all tets so every stage is
/// one call into a shared body over a contiguous range, which is what lets the
/// host compiler vectorize and what keeps the shim a loop with no arithmetic.
/// The arithmetic is identical stage for stage; what differs is only where the
/// intermediates live.
#[derive(Default)]
pub struct TetScratch {
    /// The deformation gradient `F`, 9 floats per tet. A DEVICE ALLOCATION:
    /// the gradient stage writes it and the factorization reads it, with no
    /// host access in between.
    pub deformation: Buffer<f32>,
    /// Its SVD: `U`, the singular values, and `V^T`.
    ///
    /// ALL THREE ARE DEVICE ALLOCATIONS. A record FIELD is one type in the
    /// Rust twin, so a buffer moves with every field that names it at any call
    /// site, and these three are named by the same four records: the
    /// factorization that writes them, the two spectral stages that read them,
    /// and the plastic creep's own copy of the same chain. Every one of those
    /// records is generated, so the component moves whole.
    ///
    /// THE TET SINGULAR VALUES AND THE FACE ONES ARE NOT IN THE SAME STATE,
    /// and the difference is which records name them rather than anything
    /// about the arrays. `state.face.svd_sigma` is reached by
    /// `ShellStrainRestoreSigmaArgs` and `ShellStretchTermsArgs`, whose entry
    /// points are still hand-written, so a shim there would be handed a handle
    /// it has nothing to resolve against.
    pub svd_u: Buffer<f32>,
    pub svd_sigma: Buffer<f32>,
    pub svd_vt: Buffer<f32>,
    /// The material diff table in the singular-value basis. DEVICE
    /// ALLOCATIONS: the table stage writes them, the two spectral stages read
    /// them, and no host code touches them in between.
    pub gradient_sigma: Buffer<f32>,
    pub hessian_sigma: Buffer<f32>,
    /// Whether the element's model id was recognized, one `unsigned` per tet.
    /// The width is the generated record's: a record's buffers address 4-byte
    /// pointees, which is what keeps one free of the padding a later field
    /// could hide in.
    ///
    /// A DEVICE ALLOCATION WITH NO HOST MIRROR, AND NOTHING READS IT. It is the
    /// `[[seam::scatter]]` destination the table stage's return value is
    /// written to, and a generated entry point over a value-returning body must
    /// carry one, so the sink is the shape of the entry rather than a value
    /// this driver wants. The body keeps returning the verdict because that
    /// return value is what stops an unrecognized model id from falling through
    /// to SNHk for every caller of the shared body.
    ///
    /// A [`ReadbackBuffer`] HERE WOULD BUY NOTHING AND COST A QUEUE DRAIN PER
    /// NEWTON ITERATION. The verdict is a pure function of `model`, which the
    /// host builds out of the scene's material table, so the answer is
    /// available before a frame is written: `super::refusal::material_defects`
    /// settles it once at `initialize()` and names what it refuses.
    pub accepted: Buffer<u32>,
    /// The per-element material constants, gathered from the param array.
    ///
    /// THE FIRST THREE AND `damping` ARE STAGED. Each is BUILT ON THE HOST out
    /// of the scene's material table, an element at a time, and then read by a
    /// dispatch, which is the shape [`StagedBuffer`] exists for: a host array,
    /// a device allocation, and one upload between them. `mass` stays a host
    /// array because a hand-written record still names it, and a shim taking a
    /// flat pointer has nothing to resolve a handle against.
    pub model: Buffer<u32>,
    pub mu: Buffer<f32>,
    pub lambda: Buffer<f32>,
    pub mass: Buffer<f32>,
    /// Rayleigh stiffness damping, read by the damping stage alone.
    pub damping: Buffer<f32>,
    /// The force and Hessian in `F` space, then converted to position space,
    /// then mass-scaled into the accumulators the damping stage also writes.
    ///
    /// THE FIRST TWO ARE DEVICE ALLOCATIONS, and they are the first two buffers
    /// this driver moved. They are written by the spectral stage and read by
    /// the converter with no host access in between, and every record that
    /// names either is generated, so nothing above them had to change with
    /// them. What the move buys is not speed: a record naming a host ADDRESS
    /// cannot be dispatched on a backend that resolves an (arena, offset)
    /// handle and has no way to learn what an address means, so these were two
    /// of the fields standing between the tet elastic layer and a GPU backend.
    pub gradient_f: Buffer<f32>,
    pub hessian_f: Buffer<f32>,
    pub converted_force: Buffer<f32>,
    pub converted_hessian: Buffer<f32>,
    pub gradient_x: ReadbackBuffer<f32>,
    pub hessian_x: ReadbackBuffer<f32>,
    /// The push table: one `(row, column, block)` per stored 3x3, 16 per tet.
    pub push_row: Vec<u32>,
    pub push_column: Vec<u32>,
    pub push_block: Vec<f32>,
    pub push_stored: Vec<u32>,
    /// The force scatter's staging: the active tets' four vertex indices and
    /// twelve force floats each, compacted into one ascending run.
    ///
    /// THE SCATTER IS ONE DISPATCH OVER A CONTIGUOUS RANGE, so the elements it
    /// covers have to BE contiguous. The active list is a subset of the tet
    /// range, and the shared body reads its index tuple and its gradient at the
    /// same element, so both are gathered here in ascending active order. That
    /// is the same order a per-element walk would fold them in, which is what
    /// keeps the fp32 running sum in `force` unchanged, and it is the shape
    /// `super::contact` and `super::collider` already scatter through.
    ///
    /// STAGED, because the compaction is a HOST pass over a host active list
    /// and the embed that reads the result is a generated entry taking handles.
    /// The gather is filled through [`StagedBuffer::at`] and uploaded once, and
    /// the handle names the whole allocation: `count` is the live prefix, which
    /// is what the entry guards on. Every stage in this module and the two in
    /// `super::contact` and `super::collider` move together, because the four
    /// force-embed records they share have one field type between them.
    pub scatter_index: StagedBuffer<u32>,
    pub scatter_gradient: StagedBuffer<f32>,
}

/// The per-shell-face arrays the membrane assembly walks, one stage at a time.
///
/// The same staged shape as [`TetScratch`] and for the same reason, with two
/// differences that are the shell's own rather than a choice made here.
///
/// THE DIMENSION IS 3x2, NOT 3x3. A face's deformation gradient maps two
/// material directions into space, so `F` is `Mat3x2f`, its SVD is `svd3x2`,
/// the material table is 2x2, and the Hessian in `F` space is 6x6. Only after
/// [`face_convert_hessian`] does it reach the 9x9 the three vertices carry.
///
/// THERE ARE TWO MATERIAL FAMILIES. ARAP, StVK and SNHk go through the SVD and
/// the spectral force and Hessian; BaraffWitkin goes from `F` to `gradient_f`
/// and `hessian_f` directly. Both write the same two arrays, which is what lets
/// every stage after them be shared.
#[derive(Default)]
pub struct FaceScratch {
    // NO MATERIAL ARRAYS HERE. `model`, `mu`, `lambda`, the mass, the damping
    // and the pressure are all read off the face's own `FaceProp` and
    // `FaceParam` on the device, through `FaceElasticEmbedFromRecordsArgs`, so
    // nothing about a face's material is filled on the host per iteration.
    /// Which faces take the membrane path this iteration, in ascending order.
    ///
    /// READ ONLY FOR ITS EMPTINESS. The dispatch covers the whole shell prefix
    /// and the body's own gates decide what each face
    /// contributes; this list is what lets a step with no active face skip the
    /// launch entirely.
    pub active: Vec<u32>,
}

/// The per-hinge arrays the shell bending assembly walks, one stage at a time.
///
/// The same staged shape as [`TetScratch`] and [`FaceScratch`], with three
/// differences the hinge brings with it.
///
/// THE VERTEX ORDER IS PERMUTED, AND THE PERMUTATION IS NOT COSMETIC.
/// `dihedral_angle::face_compute_force_hessian` rewrites the hinge quadruple to
/// `(h2, h1, h0, h3)` before it reads a position, and the force columns, the
/// 12x12 Hessian's blocks, the damping body's two poses and both scatters are
/// all in that order afterwards. [`HingeScratch::remapped`] carries it, and the
/// areal density is the ONE quantity taken in the mesh order instead, because
/// it is an fp32 running sum over the four vertices and a permutation would
/// change its value.
///
/// THE STIFFNESS IS A SCALAR, NOT A MASS. A hinge's block is scaled by
/// `shell_bend_stiffness`, which folds the Discrete Shells coefficient
/// `|e|^2 / area`, the areal density and the orientation-dependent mix of
/// `bend`, `bend-warp` and `bend-weft` into one number. It multiplies the
/// force, the Hessian AND the lagged damping Hessian, which is why it is formed
/// once here and never re-derived.
///
/// THE DAMPING HESSIAN IS LAGGED, so a second evaluation of the same body at
/// the START-OF-STEP positions is part of the pipeline. `force_raw` and
/// `hessian_raw` carry the iterate's evaluation first and the start-of-step
/// evaluation second; the first is fully consumed into `gradient_x` and
/// `hessian_x` before the second is taken, and the lagged FORCE is discarded,
/// exactly as `embed_hinge_force_hessian` discards it.
#[derive(Default)]
pub struct HingeScratch {
    /// The hinge quadruples in the `(2, 1, 0, 3)` order the dihedral math
    /// wants, four `u32` per hinge.
    ///
    /// STAGED: the permutation is applied on the host, once per assembly, out
    /// of the scene's own hinge list, and the one dispatch that reads it is
    /// `shell_bend_embed`, which gathers both poses through it and reads the
    /// quad back out of it for the scatter and the push.
    pub remapped: StagedBuffer<u32>,
    /// How many hinges [`HingeScratch::remapped`] was last filled for, and the
    /// whole reason the permutation is staged once rather than once per pass.
    ///
    /// The permutation is a function of the scene's hinge list alone, which no
    /// pass rewrites, so refilling it every assembly re-uploaded bytes the
    /// device already held: measured on `drape`, twelve uploads carrying 44.1 MB
    /// where one carries 3.7 MB. A count rather than a flag because the only
    /// thing that can invalidate the buffer is the scene changing size, which
    /// [`SolverState::size`] answers by sizing it again.
    pub remapped_hinges: usize,
    /// The mass-per-area averaged over the hinge's four vertices. A DEVICE
    /// ALLOCATION: the gather writes it, the stiffness pass reads it, and no
    /// host code touches it in between.
    pub areal_density: Buffer<f32>,
    /// The scalar every one of the three blocks is scaled by.
    pub stiffness: ReadbackBuffer<f32>,
    /// Rayleigh bending damping, non-zero only on a hinge this iteration bends.
    ///
    /// KERNEL-WRITTEN AND KERNEL-READ, so a plain buffer: the stiffness pass
    /// writes it out of the same two records it reads the stiffness from, and
    /// the damping stage and the start-of-step degeneracy gate read it back on
    /// the device. Nothing on the host reads it, because "does this scene damp
    /// its bending at all" is a question about the scene's material table and
    /// is answered there.
    pub damping: Buffer<f32>,
    // NO FORCE OR HESSIAN STAGING. `shell_bend_embed` forms both in registers
    // and scatters and pushes them from there, so a hinge's 12x12 never reaches
    // memory, and there is no buffer for the raw, the lagged or the scaled
    // form and no pass that would read one.
    /// The force scatter's staging, gathered over the active hinges in
    /// ascending order; see [`TetScratch::scatter_index`] for why the scatter
    /// needs a contiguous run.
    /// The push table: one `(row, column, block)` per stored 3x3, 16 per hinge.
    pub push_row: Vec<u32>,
    pub push_column: Vec<u32>,
    pub push_block: Vec<f32>,
    pub push_stored: Vec<u32>,
}


/// The per-site arrays the rod bending assembly walks.
///
/// A SITE IS AN INTERIOR ROD VERTEX, NOT AN ELEMENT, which is the one thing
/// that separates this pipeline from the other three. `embed_rod_bend_force_
/// hessian` is dispatched per vertex over `surface_vert_count` and selects the
/// vertices with exactly two incident edges and no incident face, so the
/// element it assembles has no array of its own: its three nodes are the
/// interior vertex and its two edge-neighbors, in the order `(j, i, k)` with
/// `j` from the FIRST incident edge. `builder.rs` derives `j` and `k` the same
/// way when it registers the stencil in the fixed sparsity, so any other order
/// would ask the matrix for a coupling it does not carry.
///
/// THE SITES ARE ENUMERATED ONCE. Both adjacencies are built at scene build and
/// never rewritten, so which vertices are sites cannot change between steps;
/// only the per-iteration gate can. That also sizes this scratch by the number
/// of sites rather than by `surface_vert_count`, which on a shell scene with no
/// rods at all is every vertex and no sites.
///
/// THE STIFFNESS IS A SCALAR, exactly as the shell hinge's is: one number
/// scales the force, the Hessian AND the lagged damping Hessian, and it is
/// formed once by `rod_bend_stiffness` from the two-segment averaged
/// `bend`, the vertex's lumped mass and the two incident rest lengths. There is
/// no directional analogue: `bend-warp` and `bend-weft` are SHELL parameters
/// and a rod carries no material frame to read them in.
#[derive(Default)]
pub struct RodBendScratch {
    /// The three nodes of each site's stencil, `(j, i, k)`, three `u32` per
    /// site. The middle one is the interior vertex.
    ///
    /// STAGED, and uploaded ONCE: neither adjacency the enumeration reads is
    /// rewritten after scene build, so the stencils are formed at allocation
    /// and never again. Four dispatches read them, the force and Hessian pass
    /// at both poses, the damping pass and the plastic creep's turning angle.
    /// The host copy stays readable through [`StagedBuffer::host`], which is
    /// what the interior-vertex lookups, the scatter's index gather and the two
    /// CSR push loops read.
    pub node: StagedBuffer<u32>,
    /// The site's two incident edges, two `u32` per site, which is where the
    /// per-segment material and rest length come from.
    pub edge: Vec<u32>,
    /// The same pair list on the device, so the fused bending kernel can read
    /// each site's two incident edges itself. Filled once beside `edge`: a
    /// site's stencil is fixed at scene build.
    pub edge_device: StagedBuffer<u32>,
    /// The authored rest turning angle per site. STAGED for the reason the five
    /// above are: it is gathered on the host out of the interior vertex's prop,
    /// which the plastic creep rewrites, and read by the force and Hessian pass
    /// at both poses.
    pub rest_angle: StagedBuffer<f32>,
    /// The force scatter's staging, gathered over the active elements in
    /// ascending order; see [`TetScratch::scatter_index`] for why the scatter
    /// needs a contiguous run.
    pub scatter_index: StagedBuffer<u32>,
    pub scatter_gradient: StagedBuffer<f32>,
}

/// The cross-stitch layer's per-element arrays, one element's worth per stitch.
///
/// A STITCH IS SIX SLOTS AND NOT TWO VERTICES. Slots 0 to 2 name the source
/// triangle and 3 to 5 the target, each with barycentric weights that sum to
/// one. An endpoint that is not on a solid degenerates to `{s, s, s}` with
/// weights `{1, 0, 0}`, which recovers single-vertex behavior through the same
/// arithmetic, so nothing here distinguishes the two: the degenerate form folds
/// three contributions onto one vertex and several of its thirty-six blocks
/// onto one CSR slot, which is the sum the barycentric form asks for.
///
/// THE RECORD IS DE-INTERLEAVED. `Stitch` is `{Vec6u, Vec6f, f32}`, and the
/// shared body's entry point reads the indices, the weights and the stiffness
/// as three separate arrays, so a step's records are unpacked into the arrays
/// below once by [`SolverState::stash_stitches`] rather than per Newton
/// iteration.
#[derive(Default)]
pub struct StitchScratch {
    /// Six vertex indices per stitch, in the authored slot order.
    pub index: StagedBuffer<u32>,
    /// Six barycentric weights per stitch, in that same order.
    pub weight: StagedBuffer<f32>,
    /// The per-stitch force factor, applied inside the shared body.
    pub stiffness: StagedBuffer<f32>,
    /// `ParamSet::stitch_length_factor`, one entry per stitch.
    ///
    /// ONE SCENE-WIDE SCALAR HELD PER ELEMENT, because the entry point reads it
    /// at the stitch index alongside the per-stitch stiffness. Refilled every
    /// iteration from the step's own snapshot, so a parameter change between
    /// steps cannot leave a stale factor behind.
    pub length_factor: StagedBuffer<f32>,
    /// The contact gap and the contact offset, indexed BY VERTEX because that
    /// is how the body's gather reads them, and written only at the vertex ids
    /// this step's stitches name. Every other entry is untouched and unread.
    pub vertex_ghat: StagedBuffer<f32>,
    pub vertex_offset: StagedBuffer<f32>,
    /// The 3x6 gradient and the 18x18 Hessian, one element's worth per stitch.
    pub gradient: ReadbackBuffer<f32>,
    pub hessian: ReadbackBuffer<f32>,
    /// The push table: one `(row, column, block)` per stored 3x3, 36 per
    /// stitch.
    pub push_row: Vec<u32>,
    pub push_column: Vec<u32>,
    pub push_block: Vec<f32>,
    pub push_stored: Vec<u32>,
}

/// The shell strain limiter's per-face arrays, one element's worth per SHELL
/// face, in face order.
///
/// SEPARATE FROM `FaceScratch` RATHER THAN SHARING IT, and the reason is not
/// caution about aliasing. The limiter recomputes the deformation gradient and
/// its SVD from the same iterate the membrane used, because
/// `embed_strainlimiting_force_hessian` does: it runs after the membrane, after
/// the tets and after the `tmp_fixed` snapshot, and it takes the SHIFTED
/// singular values, which are a different array from the ones the membrane's
/// material dispatch consumed. Folding the two would make the limiter's answer
/// depend on whether the membrane ran, and the membrane returns early on a scene
/// whose faces are all inert.
#[derive(Default)]
pub struct FaceStrainScratch {
    // NO STAGING. `shell_strain_embed` runs the whole limiter in one kernel and
    // holds the gradient, the SVD, the sigma pair, the stiffness, the spectral
    // and converted forms and both Hessians in registers, so the twenty-two
    // per-face arrays that carried them between fifteen dispatches are gone.
    // What is left is the membership the pass early-outs on.
    /// The line search's per-face time of impact, folded by `reduce::min`.
    pub toi: ReadbackBuffer<f32>,
    /// Faces the dispatch's own gate admits, and the subset of those whose
    /// largest shifted singular value is positive. The second is what is
    /// scattered.
    pub candidate: Vec<u32>,
}

/// The rod strain limiter's per-rod arrays, one element's worth per ROD, in
/// edge order. Sized over the rod prefix and not over the edge array; see
/// [`Sizes::rods`].
#[derive(Default)]
pub struct RodStrainScratch {
    /// The authored `strainlimit`. A rod carries no shrink factors, so the
    /// barrier's ghat and the stiffness's divisor are the same number and there
    /// is one limit here where a face has two.
    pub limit: StagedBuffer<f32>,
    /// `EdgeProp::initial_length`, which is NOT the `length-factor`-scaled
    /// `EdgeProp::length` the stretch energy measures against. A factor of one
    /// makes the two equal, which is what hides an exchange of them.
    pub rest_length: StagedBuffer<f32>,
    pub mass: StagedBuffer<f32>,
    /// The unscaled force (3x2) and Hessian (6x6), the strain the body
    /// reported, and whether it produced a term at all.
    pub force_raw: Buffer<f32>,
    pub hessian_raw: Buffer<f32>,
    pub strain: Buffer<f32>,
    /// `unsigned` rather than a byte, because a generated entry's scatter takes
    /// one of the three scalar types a record field may hold.
    pub ok: ReadbackBuffer<u32>,
    pub stiffness: Buffer<f32>,
    /// The stiffness-scaled force and Hessian the scatter reads.
    pub gradient_x: ReadbackBuffer<f32>,
    pub hessian_x: ReadbackBuffer<f32>,
    /// The force scatter's staging, gathered over the active elements in
    /// ascending order; see [`TetScratch::scatter_index`] for why the scatter
    /// needs a contiguous run.
    pub scatter_index: StagedBuffer<u32>,
    pub scatter_gradient: StagedBuffer<f32>,
    /// The push table: one `(row, column, block)` per stored 3x3, 4 per rod.
    pub push_row: Vec<u32>,
    pub push_column: Vec<u32>,
    pub push_block: Vec<f32>,
    pub push_stored: Vec<u32>,
    /// The line search's per-rod time of impact, folded by `reduce::min`.
    pub toi: ReadbackBuffer<f32>,
    pub candidate: Vec<u32>,
    pub active: Vec<u32>,
}

/// The `max_sigma` indicator's arrays: one element's worth per SHELL face and
/// per ROD.
///
/// TELEMETRY, AND ITS OWN BUFFERS FOR THAT REASON. It is measured once per
/// `advance()` at the START-OF-STEP pose, before the Newton loop, while every
/// other face and rod array in this file is rewritten inside that loop. Sharing
/// one would make an indicator read at the top of the step depend on a buffer
/// the loop below is about to overwrite, which is the kind of coupling that
/// survives review and then breaks when a phase moves.
#[derive(Default)]
pub struct StretchScratch {
    /// `F` at the start-of-step pose and its SVD, per shell face. The singular
    /// values are UNSHIFTED here: the indicator is a stretch RATIO, not the
    /// strain the barrier is a function of. `F` is a DEVICE ALLOCATION, in the
    /// same component as [`FaceScratch::deformation`].
    pub deformation: Buffer<f32>,
    /// DEVICE ALLOCATIONS, in the same component as [`FaceScratch::svd_u`].
    /// WRITTEN AND NEVER READ on this path: the indicator is a function of the
    /// singular values alone, and the factorization writes all three.
    pub svd_u: Buffer<f32>,
    pub sigma: Buffer<f32>,
    pub svd_vt: Buffer<f32>,
    /// The face's two shrink factors, zeroed on a face the indicator's gate
    /// excludes, which is how that face's ratio is made zero.
    /// The two selections the ratio is the product of, and the product.
    pub largest: Buffer<f32>,
    pub shrink_min: Buffer<f32>,
    pub ratio: ReadbackBuffer<f32>,
    /// `EdgeProp::initial_length` per rod, zeroed on a rod the gate excludes,
    /// and the ratio taken against it.
    pub rod_ratio: ReadbackBuffer<f32>,
}

/// Grow a buffer to `len` and zero it, reporting an allocation failure by name.
fn size(buffer: &mut Vec<f32>, len: usize, what: &str) -> FatalResult<()> {
    reserve(buffer, len, what)?;
    buffer.resize(len, 0.0);
    buffer.fill(0.0);
    Ok(())
}

fn size_u32(buffer: &mut Vec<u32>, len: usize, what: &str) -> FatalResult<()> {
    reserve(buffer, len, what)?;
    buffer.resize(len, 0);
    Ok(())
}

/// `try_reserve`, so an out-of-memory is a named refusal rather than an abort.
///
/// A plain `resize` aborts the process on failure, which loses the crash record
/// the host writes from an `atexit` hook. Every buffer here is sized from the
/// scene, so a scene too large for this machine is a legitimate outcome to
/// report.
fn reserve<T>(buffer: &mut Vec<T>, len: usize, what: &str) -> FatalResult<()> {
    if buffer.len() >= len {
        return Ok(());
    }
    buffer.try_reserve_exact(len - buffer.len()).map_err(|error| {
        Fatal::out_of_memory(format!(
            "solver driver: cannot allocate {len} elements ({} MiB) for {what}: {error}",
            (len * std::mem::size_of::<T>()) >> 20
        ))
    })
}

/// Check the vertex-face adjacency the momentum layer walks, ONCE.
///
/// The shared body reads `offset[i] .. offset[i + 1]` for every vertex and then
/// indexes `face[3 * f]`, and it cannot check either: a C++ range takes the
/// pointers it is given. On CUDA an out-of-range read there is a live device
/// assert; on this backend it would be a wild read of the caller's own memory,
/// because decision D1 is direct addressing and these are the host's arrays.
///
/// Once at `initialize()` rather than per step, because the table is built at
/// scene build and never rewritten. That also puts the report before the first
/// frame, which is where a scene defect belongs.
unsafe fn validate_vertex_face_table(
    data: &DataSet,
    vertices: usize,
    faces: usize,
) -> FatalResult<()> {
    let neighbor = &data.mesh.neighbor.vertex.face;
    if neighbor.offset.is_null() || neighbor.size == 0 {
        // No table. `compute_vertex_normal` reads that as "this scene presents
        // no surface" and the driver passes a null offset, so there is nothing
        // to check.
        return Ok(());
    }
    if neighbor.size as usize != vertices {
        return Err(Fatal::invariant(format!(
            "solver driver: the vertex-face adjacency has {} rows and the scene has \
             {vertices} vertices. The momentum layer reads one row per vertex, so a \
             shorter table would be read past its end",
            neighbor.size
        )));
    }
    let offset = std::slice::from_raw_parts(neighbor.offset, vertices + 1);
    if offset[0] != 0 {
        return Err(Fatal::invariant(format!(
            "solver driver: the vertex-face adjacency's first row offset is {}, not 0",
            offset[0]
        )));
    }
    for i in 0..vertices {
        if offset[i] > offset[i + 1] {
            return Err(Fatal::invariant(format!(
                "solver driver: the vertex-face adjacency's row {i} runs {}..{}, which is \
                 backwards",
                offset[i],
                offset[i + 1]
            )));
        }
    }
    let entries = offset[vertices] as usize;
    if entries > neighbor.nnz as usize {
        return Err(Fatal::invariant(format!(
            "solver driver: the vertex-face adjacency's offsets reach {entries} entries over \
             a table declaring {}",
            neighbor.nnz
        )));
    }
    let index = std::slice::from_raw_parts(neighbor.data, entries);
    for (slot, face) in index.iter().enumerate() {
        if *face as usize >= faces {
            return Err(Fatal::device_assert(format!(
                "solver driver: the vertex-face adjacency names face {face} at entry {slot}, and \
                 the scene has {faces} faces"
            )));
        }
    }
    Ok(())
}

/// One `CVecVec` adjacency as its index and offset arrays, validated.
///
/// The same argument as [`validate_vertex_face_table`]: a shared body, or a
/// walk here, reads `offset[i] .. offset[i + 1]` and then indexes with what it
/// finds, and neither can check the table it is handed. `None` is a scene that
/// carries no such table at all, which is not an error: a scene with no edges
/// has no vertex-edge adjacency and no rod bending sites either.
///
/// # Safety
/// `table` must belong to a live `DataSet`.
unsafe fn adjacency<'a>(
    table: &crate::cvecvec::CVecVec<u32>,
    rows: usize,
    what: &str,
) -> FatalResult<Option<(&'a [u32], &'a [u32])>> {
    if table.offset.is_null() || table.size == 0 {
        return Ok(None);
    }
    if (table.size as usize) < rows {
        return Err(Fatal::invariant(format!(
            "solver driver: {what} has {} rows and the walk covers {rows} vertices, so a row would \
             be read past the end of the table",
            table.size
        )));
    }
    let offset = std::slice::from_raw_parts(table.offset, rows + 1);
    if offset[0] != 0 {
        return Err(Fatal::invariant(format!(
            "solver driver: {what}'s first row offset is {}, not 0",
            offset[0]
        )));
    }
    for i in 0..rows {
        if offset[i] > offset[i + 1] {
            return Err(Fatal::invariant(format!(
                "solver driver: {what}'s row {i} runs {}..{}, which is backwards",
                offset[i],
                offset[i + 1]
            )));
        }
    }
    let entries = offset[rows] as usize;
    if entries > table.nnz as usize {
        return Err(Fatal::invariant(format!(
            "solver driver: {what}'s offsets reach {entries} entries over a table declaring {}",
            table.nnz
        )));
    }
    Ok(Some((
        std::slice::from_raw_parts(table.data, entries),
        offset,
    )))
}

/// Enumerate the rod bending sites, ONCE, and fill their node and edge tables.
///
/// The site test is the rod bending element's own: exactly two incident edges
/// and no incident face. `builder.rs` applies the
/// identical test when it registers the `(j, i, k)` stencil in the fixed
/// sparsity, and derives `j` and `k` from the two edges in the same order, so
/// the sites this produces are the ones the matrix carries slots for.
///
/// THE RANGE IS `surface_vert_count`, which is what the rod bending pass is
/// dispatched over. Every rod vertex is a contact vertex
/// (a non-contact vertex is a tet interior Steiner point, which has no edges),
/// so the prefix covers every site; a scene declaring rods and no surface
/// vertices at all is a `DataSet` that did not come from this frontend and
/// stops the run rather than silently assembling no bending.
///
/// # Safety
/// `data` must point at a live `DataSet`, and `vertices` be its vertex count.
unsafe fn enumerate_rod_bend_sites(
    data: &DataSet,
    vertices: usize,
    node: &mut Vec<u32>,
    edge_of_site: &mut Vec<u32>,
) -> FatalResult<usize> {
    node.clear();
    edge_of_site.clear();
    let rods = data.rod_count as usize;
    let surface = data.surface_vert_count as usize;
    if surface > vertices {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene declares {surface} surface vertices over {vertices} \
             vertices. The surface vertices are a PREFIX of the vertex array, so a longer \
             prefix does not describe this mesh"
        )));
    }
    if rods > 0 && surface == 0 {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene carries {rods} rods and declares no surface vertices. A rod \
             vertex is always a contact vertex, so the rod bending dispatch would cover none \
             of them and the term would be silently absent"
        )));
    }
    if surface == 0 {
        return Ok(0);
    }
    let Some((vertex_edge, edge_offset)) =
        adjacency(&data.mesh.neighbor.vertex.edge, surface, "the vertex-edge adjacency")?
    else {
        return Ok(0);
    };
    let Some((_, face_offset)) =
        adjacency(&data.mesh.neighbor.vertex.face, surface, "the vertex-face adjacency")?
    else {
        // No vertex-face table at all, so no vertex has an incident face and
        // the second half of the site test is satisfied by every vertex.
        return enumerate_sites_from(
            data,
            vertices,
            surface,
            vertex_edge,
            edge_offset,
            None,
            node,
            edge_of_site,
        );
    };
    enumerate_sites_from(
        data,
        vertices,
        surface,
        vertex_edge,
        edge_offset,
        Some(face_offset),
        node,
        edge_of_site,
    )
}

/// The walk itself, with the vertex-face row counts optional.
///
/// TWO PASSES OVER THE VERTICES, and the first one exists to size the two
/// tables. Growing them as the sites are found would reallocate once per site,
/// because `reserve` here asks for exactly what is missing rather than for a
/// geometric step; counting first is cheap (two subtractions per vertex) and
/// the alternative of reserving `3 * surface` up front would allocate a table
/// the size of the mesh on a shell scene with no rods at all.
///
/// # Safety
/// As [`enumerate_rod_bend_sites`].
#[allow(clippy::too_many_arguments)]
unsafe fn enumerate_sites_from(
    data: &DataSet,
    vertices: usize,
    surface: usize,
    vertex_edge: &[u32],
    edge_offset: &[u32],
    face_offset: Option<&[u32]>,
    node: &mut Vec<u32>,
    edge_of_site: &mut Vec<u32>,
) -> FatalResult<usize> {
    let mesh_edge: &[crate::data::Vec2u] = super::scene::slice(&data.mesh.mesh.edge);
    let is_site = |i: usize| -> bool {
        edge_offset[i + 1] - edge_offset[i] == 2
            && face_offset.is_none_or(|faces| faces[i + 1] == faces[i])
    };
    let expected = (0..surface).filter(|i| is_site(*i)).count();
    reserve(node, 3 * expected, "the rod bending stencil nodes")?;
    reserve(edge_of_site, 2 * expected, "the rod bending incident edges")?;
    let mut sites = 0usize;
    for i in 0..surface {
        if !is_site(i) {
            continue;
        }
        let first = vertex_edge[edge_offset[i] as usize] as usize;
        let second = vertex_edge[edge_offset[i] as usize + 1] as usize;
        for named in [first, second] {
            if named >= mesh_edge.len() {
                return Err(Fatal::device_assert(format!(
                    "solver driver: the vertex-edge adjacency names edge {named} at vertex {i}, \
                     and the scene has {} edges",
                    mesh_edge.len()
                )));
            }
        }
        // `j` from the FIRST incident edge and `k` from the second, which is
        // the order `builder.rs` registers the stencil in.
        let j = if mesh_edge[first][0] as usize == i {
            mesh_edge[first][1]
        } else {
            mesh_edge[first][0]
        };
        let k = if mesh_edge[second][0] as usize == i {
            mesh_edge[second][1]
        } else {
            mesh_edge[second][0]
        };
        for named in [j, k] {
            if named as usize >= vertices {
                return Err(Fatal::device_assert(format!(
                    "solver driver: rod bending site {i} names neighbor vertex {named} and the \
                     scene has {vertices} vertices"
                )));
            }
        }
        node.push(j);
        node.push(i as u32);
        node.push(k);
        edge_of_site.push(first as u32);
        edge_of_site.push(second as u32);
        sites += 1;
    }
    Ok(sites)
}

/// Copy one topology array onto the device and upload it, once per scene.
///
/// A FREE FUNCTION because four of these are staged in a row and a `?` on each
/// reads better than four inline blocks. The array is the scene's connectivity,
/// which nothing in the step loop rewrites, so this is the only transfer it
/// ever needs.
///
/// # Safety
/// `source` must address `count` live `u32` for the duration of the call.
/// The generic form of [`stage_topology`], for a record array rather than an
/// index array.
unsafe fn stage_records<D: Device, T: ppf_cts_compute::Pod + Default>(
    device: &mut D,
    buffer: &mut StagedBuffer<T>,
    source: *const T,
    count: usize,
    label: &'static str,
) -> FatalResult<()> {
    buffer.size(device, count, AllocLabel(label))?;
    if count > 0 {
        buffer
            .at()
            .copy_from_slice(std::slice::from_raw_parts(source, count));
    }
    buffer.upload(device)?;
    Ok(())
}

/// Upload a build-time `u32` array that is never written again.
/// Stage one adjacency table's index and offset arrays, or neither.
///
/// AN ABSENT TABLE LEAVES BOTH BUFFERS UNSIZED, so `handle_or_none` below hands
/// the record `Handle::NONE` for it. That is the same "names nothing" the host
/// pointer form spelled with a null, said in the one way a handle can say it.
///
/// # Safety
/// `table` must be a live `CVecVec` outliving the staged copy.
unsafe fn stage_adjacency<D: Device, T>(
    device: &mut D,
    table: &crate::cvecvec::CVecVec<T>,
    index: &mut Buffer<u32>,
    offset: &mut Buffer<u32>,
    label: &'static str,
) -> FatalResult<()> {
    if table.offset.is_null() || table.size == 0 {
        // A REAL ZERO-LENGTH ALLOCATION, never the `Handle::NONE` sentinel.
        // The generated entry resolves every `[[seam::device]]` field as
        // `seam_arena_base[handle.arena] + handle.off` and asserts the arena
        // index first, so a sentinel arena of `u32::MAX` is out of bounds
        // before any dereference. This is the same rule the CUDA arena states:
        // an array with zero elements takes a real zero-length handle, because
        // an EMPTY array and a NULL one are different things and only the arena
        // can tell them apart.
        index.size(device, 0, AllocLabel(label))?;
        offset.size(device, 0, AllocLabel(label))?;
        return Ok(());
    }
    let rows = table.size as usize;
    let nnz = table.nnz as usize;
    stage_static_u32(device, index, std::slice::from_raw_parts(table.data as *const u32, nnz), label)?;
    stage_static_u32(device, offset, std::slice::from_raw_parts(table.offset as *const u32, rows + 1), label)?;
    Ok(())
}

/// The handle a staged adjacency hands a record.
///
/// ALWAYS A REAL HANDLE, including for a table the scene does not carry: that
/// case is a zero-length allocation rather than `Handle::NONE`, because the
/// generated entry resolves the field before the body branches on the `has_*`
/// flag, and resolving the sentinel indexes `seam_arena_base` out of bounds.
/// The flag is what says the table is absent; the handle only has to be
/// resolvable.
pub fn adjacency_handle(buffer: &Buffer<u32>) -> ppf_cts_compute::Handle {
    buffer.span(0, buffer.len())
}

fn stage_static_u32<D: Device>(
    device: &mut D,
    buffer: &mut Buffer<u32>,
    host: &[u32],
    label: &'static str,
) -> FatalResult<()> {
    buffer.size(device, host.len(), AllocLabel(label))?;
    if !host.is_empty() {
        buffer.write(device, 0, host)?;
    }
    Ok(())
}

unsafe fn stage_topology<D: Device>(
    device: &mut D,
    buffer: &mut StagedBuffer<u32>,
    source: *const u32,
    count: usize,
    label: &'static str,
) -> FatalResult<()> {
    buffer.size(device, count, AllocLabel(label))?;
    if count > 0 {
        buffer
            .at()
            .copy_from_slice(std::slice::from_raw_parts(source, count));
    }
    buffer.upload(device)?;
    Ok(())
}

/// The device side of a fixed-matrix push.
///
/// `stored` is a `ReadbackBuffer` and the other three are not, which is the
/// vocabulary reading the direction: the host builds the row, column and block
/// triple and a kernel reads it, while the kernel writes the per-block verdict
/// and the host reads THAT. A dropped block is a lost Hessian coupling, so the
/// verdict has to come back.
#[derive(Debug, Default)]
pub struct PushStaging {
    pub row: Buffer<u32>,
    pub column: Buffer<u32>,
    pub block: Buffer<f32>,
    pub stored: ReadbackBuffer<u32>,
    /// How many blocks a DEVICE push offered that the fixed pattern had no slot
    /// for, this pass.
    ///
    /// THE HOST PUSH CARRIES ITS VERDICT PER BLOCK IN `stored` AND THE DEVICE
    /// PUSH CANNOT: a kernel that returned a per-block verdict would be writing
    /// an array the host then walks, which is the staging a device push exists
    /// to avoid. A
    /// count is enough, because the host raises on ANY refusal rather than
    /// naming one: an elastic stencil the pattern cannot hold is a
    /// `builder.rs` defect and the scene does not run either way.
    pub refused: ReadbackBuffer<u32>,
    /// One refused block's row and column, so the fatal can name a block.
    ///
    /// SEPARATE FROM THE COUNTER because the host seam deletes assignment to an
    /// atomic slot, which is right: a plain store to one is a non-atomic write.
    /// This pair is racy by design, several threads may write it and the last
    /// wins, and any winner names a genuine dropped block.
    pub witness: ReadbackBuffer<u32>,
}

impl SolverState {
    /// The mesh handles the contact subsystem reads.
    ///
    /// ONE BUILDER, because a `Contact` is a field of this state and cannot
    /// borrow the state it lives in, so every caller has to hand it the same
    /// carrier. Three places built it by hand and the three drifted apart the
    /// moment a field was added; a method cannot.
    pub fn refs(&mut self) -> super::contact::MeshRefs {
        super::contact::MeshRefs {
            face: self.mesh_face.handle(),
            edge: self.mesh_edge.handle(),
            vertex_prop: self.prop_vertex.handle(),
            face_prop: self.prop_face.handle(),
            edge_prop: self.prop_edge.handle(),
            vertex_param: self.param_vertex.handle(),
            face_param: self.param_face.handle(),
            edge_param: self.param_edge.handle(),
            fixed_index: self.fixed_index.handle(),
            fixed_offset: self.fixed_offset.handle(),
            vertex_edge_index: adjacency_handle(&self.neighbor_vertex_edge_index),
            vertex_edge_offset: adjacency_handle(&self.neighbor_vertex_edge_offset),
            has_vertex_edge: u32::from(self.neighbor_vertex_edge_offset.len() != 0),
            vertex_face_index: adjacency_handle(&self.neighbor_vertex_face_index),
            vertex_face_offset: adjacency_handle(&self.neighbor_vertex_face_offset),
            has_vertex_face: u32::from(self.neighbor_vertex_face_offset.len() != 0),
            edge_face_index: adjacency_handle(&self.neighbor_edge_face_index),
            edge_face_offset: adjacency_handle(&self.neighbor_edge_face_offset),
            has_edge_face: u32::from(self.neighbor_edge_face_offset.len() != 0),
            start_link: super::contact::StartLinkRefs {
                index: adjacency_handle(&self.start_link_index),
                offset: adjacency_handle(&self.start_link_offset),
                present: u32::from(self.start_link_offset.len() != 0),
            },
        }
    }

    /// The fixed pattern's four arrays as handles, for a matrix's `view`.
    ///
    /// A `FixedCsr` owns its values and cannot borrow this state, so the
    /// pattern reaches it as a carrier. Same shape as `MeshRefs`.
    pub fn fixed_pattern_refs(&mut self) -> super::fixedcsr::FixedPatternRefs {
        super::fixedcsr::FixedPatternRefs {
            index: self.fixed_index.handle(),
            offset: self.fixed_offset.handle(),
            transpose_pair: self.transpose_pair.handle(),
            transpose_offset: self.transpose_offset.handle(),
            rows: self.fixed_rows,
        }
    }

    /// Size every buffer for this scene, once, at `initialize()`.
    ///
    /// It takes a [`Device`] because sizing is not the whole of it: the hinge
    /// permutation table is topology, fixed at scene build, and is filled here
    /// by one dispatch so that a caller cannot end up with a sized-but-unfilled
    /// table. Every buffer named here is a driver-owned `Vec` today, which is
    /// the migration debt `ppf_cts_compute::HostRef` records; when they become
    /// arena handles the allocation itself moves onto this same parameter.
    ///
    /// # Safety
    /// `data` must point at a live `DataSet`.
    pub unsafe fn allocate<D: Device>(&mut self, device: &mut D, data: &DataSet) -> FatalResult<()> {
        let vertices = data.vertex.curr.size as usize;
        let tets = data.mesh.mesh.tet.size as usize;
        let faces = data.mesh.mesh.face.size as usize;
        // THE PREFIX, NOT THE ARRAY. A solid's surface triangles follow the
        // shell faces in `mesh.face` and carry no membrane elasticity, so the
        // membrane's scratch is sized over `shell_face_count`.
        //
        // A PREFIX LONGER THAN THE ARRAY STOPS THE RUN rather than being
        // clamped to fit. Clamping would size the scratch correctly and leave
        // the membrane covering fewer faces than the scene declares, with
        // nothing in the output to say which ones were dropped, and it would
        // also defeat the assembly's own bound check by making it unreachable.
        let shell_faces = data.shell_face_count as usize;
        if shell_faces > faces {
            return Err(Fatal::invariant(format!(
                "solver driver: the scene declares {shell_faces} shell faces over a face array of \
                 {faces}. The shell faces are a PREFIX of that array, so a longer prefix does \
                 not describe this mesh"
            )));
        }
        let hinges = data.mesh.mesh.hinge.size as usize;
        // THE PREFIX, NOT THE ARRAY, for the same reason and with the same
        // trap: `mesh.edge` carries every face's edges after the rods, because
        // edge-edge contact needs them, and they carry no stretch energy. The
        // two coincide exactly when the scene has no shell and no solid.
        let edges = data.mesh.mesh.edge.size as usize;
        let rods = data.rod_count as usize;
        if rods > edges {
            return Err(Fatal::invariant(format!(
                "solver driver: the scene declares {rods} rods over an edge array of {edges}. The \
                 rods are a PREFIX of that array, so a longer prefix does not describe this mesh"
            )));
        }
        // THE TOPOLOGY, COPIED ONCE. Safety: each pointer addresses its own
        // array in the live `DataSet` and the widths are the ones the shared
        // records read, three `u32` per face, two per edge, four per tet and
        // four per hinge.
        stage_topology(device, &mut self.mesh_face, data.mesh.mesh.face.data as *const u32,
                       3 * faces, "mesh.face")?;
        stage_topology(device, &mut self.mesh_edge, data.mesh.mesh.edge.data as *const u32,
                       2 * edges, "mesh.edge")?;
        stage_topology(device, &mut self.mesh_tet, data.mesh.mesh.tet.data as *const u32,
                       4 * tets, "mesh.tet")?;
        // THE SLOT TABLE, on the same terms: computed once by `builder.rs` and
        // never written again, so it is staged with the topology rather than
        // with the per-step properties. A zero size is the search-path arm and
        // is staged as a real zero-length allocation, never a null handle.
        stage_topology(device, &mut self.tet_hess_slots,
                       data.tet_hess_slots.data as *const u32,
                       data.tet_hess_slots.size as usize, "tet_hess_slots")?;
        stage_topology(device, &mut self.hinge_hess_slots,
                       data.hinge_hess_slots.data as *const u32,
                       data.hinge_hess_slots.size as usize, "hinge_hess_slots")?;
        stage_topology(device, &mut self.edge_hess_slots,
                       data.edge_hess_slots.data as *const u32,
                       data.edge_hess_slots.size as usize, "edge_hess_slots")?;
        stage_topology(device, &mut self.mesh_hinge, data.mesh.mesh.hinge.data as *const u32,
                       4 * hinges, "mesh.hinge")?;
        // THE PER-VERTEX PROPERTIES. Unlike the topology these are still
        // written on the host, by the plastic creep, so they are staged rather
        // than copied once: the creep writes through `at()` and uploads.
        stage_records(device, &mut self.prop_vertex, data.prop.vertex.data,
                      data.prop.vertex.size as usize, "prop.vertex")?;
        // THE FIXED PATTERN, written once. It is built at scene build and never
        // written again, so this is a plain upload with no mirror to keep.
        //
        // Safety: `borrow_pattern` validates these two tables at `initialize`,
        // and the widths here are the ones it reads.
        {
            let pattern = unsafe { super::fixedcsr::borrow_pattern(
                &data.fixed_index_table,
                &data.transpose_table,
            ) }?;
            stage_static_u32(device, &mut self.fixed_index, pattern.index, "csr.fixed_index")?;
            stage_static_u32(device, &mut self.fixed_offset, pattern.offset, "csr.fixed_offset")?;
            stage_static_u32(device, &mut self.transpose_pair, pattern.transpose_pair,
                             "csr.transpose_pair")?;
            stage_static_u32(device, &mut self.transpose_offset, pattern.transpose_offset,
                             "csr.transpose_offset")?;
            self.fixed_rows = pattern.rows;
        }

        // The three neighbor adjacencies. Each is staged only when the scene
        // carries it; an absent one is left unsized and its record field takes
        // `Handle::NONE` beside a zero `has_*` flag.
        //
        // Safety: each table is a live `CVecVec` inside the `DataSet` and the
        // widths are the ones `adjacency_index` and `adjacency_offset` read.
        unsafe {
            let n = &data.mesh.neighbor;
            stage_adjacency(device, &n.vertex.edge, &mut self.neighbor_vertex_edge_index,
                            &mut self.neighbor_vertex_edge_offset, "neighbor.vertex_edge")?;
            stage_adjacency(device, &n.vertex.face, &mut self.neighbor_vertex_face_index,
                            &mut self.neighbor_vertex_face_offset, "neighbor.vertex_face")?;
            stage_adjacency(device, &n.edge.face, &mut self.neighbor_edge_face_index,
                            &mut self.neighbor_edge_face_offset, "neighbor.edge_face")?;
            // Allow Existing Intersections' table, on the same terms: a
            // scene that linked nothing leaves it unsized and reads absent.
            stage_adjacency(device, &data.start_link, &mut self.start_link_index,
                            &mut self.start_link_offset, "start_link")?;
        }
        // Safety: the two arrays hold one matrix per shell face and per tet,
        // four and nine floats wide.
        {
            let n2 = 4 * data.inv_rest2x2.size as usize;
            let n3 = 9 * data.inv_rest3x3.size as usize;
            self.inv_rest2x2.size(device, n2, AllocLabel("plastic.inv_rest2x2"))?;
            self.inv_rest3x3.size(device, n3, AllocLabel("plastic.inv_rest3x3"))?;
            if n2 > 0 {
                self.inv_rest2x2.seed(device, std::slice::from_raw_parts(
                    data.inv_rest2x2.data as *const f32, n2))?;
            }
            if n3 > 0 {
                self.inv_rest3x3.seed(device, std::slice::from_raw_parts(
                    data.inv_rest3x3.data as *const f32, n3))?;
            }
        }
        // THE STATISTICS CHANNEL. Sized from the scene's own arrays, which are
        // empty for a scene that configures no statistics objects, so the
        // recorders' `statistics_enabled` reads false and every call is a
        // predictable early return rather than a branch on a null.
        {
            let objects = data.statistics_contact_count.size as usize;
            let dynamic = data.statistics_object_index.size as usize;
            let statics = data.statistics_static_object_index.size as usize;
            stage_records(device, &mut self.statistics_object_index,
                          data.statistics_object_index.data, dynamic,
                          "statistics.object_index")?;
            stage_records(device, &mut self.statistics_static_object_index,
                          data.statistics_static_object_index.data, statics,
                          "statistics.static_object_index")?;
            self.statistics_contact_count
                .size(device, objects, AllocLabel("statistics.contact_count"))?;
        }

        // THE AGGREGATE LOCK'S ARRAYS AND SCRATCH, sized only when the scene
        // carries a lock. `translation_lock_initial` is the start pose each
        // group's drift is measured against, staged as its raw components
        // because that is what the entry records address.
        {
            let groups = data.translation_lock.size as usize;
            stage_records(device, &mut self.translation_lock,
                          data.translation_lock.data, groups, "lock.records")?;
            let indexed = data.translation_lock_index.size as usize;
            stage_records(device, &mut self.translation_lock_index,
                          data.translation_lock_index.data, indexed, "lock.index")?;
            let initial = 3 * data.translation_lock_initial.size as usize;
            self.translation_lock_initial
                .size(device, initial, AllocLabel("lock.initial"))?;
            if initial > 0 {
                self.translation_lock_initial
                    .at()
                    .copy_from_slice(std::slice::from_raw_parts(
                        data.translation_lock_initial.data as *const f32,
                        initial,
                    ));
            }
            self.translation_lock_initial.upload(device)?;
            // The invariant check's two reductions, group-length and allocated
            // once. Both are cleared at the top of every check rather than
            // here: the check runs twice per step and each run must start from
            // the identity of its own reduction.
            self.lock_drift
                .size(device, 3 * groups, AllocLabel("lock.drift"))?;
            self.lock_max_displacement
                .size(device, groups, AllocLabel("lock.maxdisp"))?;
            self.lock.allocate(device, groups, vertices)?;
            // THE PDRD REDUCTION'S TOPOLOGY, built and staged once. It is a
            // pure function of the scene, so nothing in the step loop rebuilds
            // it; a scene with no rigid body gets a map whose `n_bodies` is
            // zero and every PDRD dispatch is gated on that.
            self.rigid = super::rigid_map::build(data, vertices)
                .map_err(Fatal::invariant)?;
            self.rigid_staged.stage(device, &self.rigid, data)?;
            let bodies = self.rigid.n_bodies;
            if bodies > 0 {
                self.rigid_state
                    .size(device, bodies, AllocLabel("pdrd.state"))?;
                self.rigid_running_rotation
                    .size(device, 9 * bodies, AllocLabel("pdrd.rrun"))?;
                self.rigid_rotation_step
                    .size(device, 3 * bodies, AllocLabel("pdrd.dtheta"))?;
                self.rigid_blocks
                    .size(device, 36 * bodies, AllocLabel("pdrd.blocks"))?;
                self.rigid_factor
                    .size(device, 36 * bodies, AllocLabel("pdrd.factor"))?;
                self.rigid_fit_scratch
                    .size(device, 12 * bodies, AllocLabel("pdrd.fit_scratch"))?;
                self.rigid_centroid
                    .size(device, 3 * bodies, AllocLabel("pdrd.centroid"))?;
                self.rigid_target
                    .size(device, 3 * vertices, AllocLabel("pdrd.rigid_target"))?;
                self.pcg_rigid.allocate(device, self.rigid.dim, vertices, bodies)?;
                // THE RUNNING ROTATION STARTS AT THE IDENTITY, one per body.
                // Zeroing it would rigidify every body onto a degenerate frame
                // on the first step, which is not a small error: the body would
                // collapse to its centroid.
                let mut identity = vec![0.0f32; 9 * bodies];
                for body in 0..bodies {
                    identity[9 * body] = 1.0;
                    identity[9 * body + 4] = 1.0;
                    identity[9 * body + 8] = 1.0;
                }
                self.rigid_running_rotation.write(device, 0, &identity)?;
                // NAMED ON EVERY RUN THAT HAS A BODY, as the Metal backend does
                // at `main.mm:2055`: a scene whose reduction silently came out
                // the wrong shape reads differently in the log from one that
                // never had a body, and the acceptance fixtures parse this line
                // to check the partition.
                ::log::info!(
                    "PDRD reduced system has {} cloth vertices and {} bodies, \
                     {} reduced DOFs with the body blocks at {}",
                    self.rigid.n_cloth,
                    bodies,
                    self.rigid.dim,
                    self.rigid.body_base
                );
            }
            // The SAND blocks, sized for every vertex whatever the scene holds:
            // the analytic contact writes all of them.
            self.grain_angular
                .size(device, 9 * vertices, AllocLabel("grain.angular"))?;
            self.grain_coupling
                .size(device, 9 * vertices, AllocLabel("grain.coupling"))?;
            self.grain_rotational
                .size(device, 3 * vertices, AllocLabel("grain.rotational"))?;
            self.grain_torque
                .size(device, 3 * vertices, AllocLabel("grain.torque"))?;
            self.grain_stiffness
                .size(device, vertices, AllocLabel("grain.stiffness"))?;
            self.grain_normal
                .size(device, 3 * vertices, AllocLabel("grain.normal"))?;
            let inertia = data.grain_inv_inertia.size as usize;
            self.grain_inv_inertia
                .size(device, vertices.max(inertia), AllocLabel("grain.inv_inertia"))?;
            if inertia > 0 {
                self.grain_inv_inertia.at()[..inertia].copy_from_slice(
                    std::slice::from_raw_parts(data.grain_inv_inertia.data, inertia),
                );
            }
            self.grain_inv_inertia.upload(device)?;
            for (buffer, source, label) in [
                (&mut self.grain_omega, data.grain_omega.data as *const f32, "grain.omega"),
                (&mut self.grain_omega_prev, data.grain_omega_prev.data as *const f32, "grain.omega_prev"),
            ] {
                buffer.size(device, 3 * vertices, AllocLabel(label))?;
                let n = 3 * (data.grain_omega.size as usize).min(vertices);
                if n > 0 {
                    buffer.write(device, 0, std::slice::from_raw_parts(source, n))?;
                }
            }
            let centers = (data.grain_inv_inertia_center.size as usize).min(vertices);
            self.grain_inv_inertia_center.size(
                device,
                vertices,
                AllocLabel("grain.inv_inertia_center"),
            )?;
            if centers > 0 {
                self.grain_inv_inertia_center.at()[..centers].copy_from_slice(
                    std::slice::from_raw_parts(data.grain_inv_inertia_center.data, centers),
                );
            }
            self.grain_inv_inertia_center.upload(device)?;
            if groups > 0 {
                self.pcg_locked.allocate(device, vertices as u32)?;
            }
        }
        stage_records(device, &mut self.prop_face, data.prop.face.data,
                      data.prop.face.size as usize, "prop.face")?;
        stage_records(device, &mut self.prop_hinge, data.prop.hinge.data,
                      data.prop.hinge.size as usize, "prop.hinge")?;
        stage_records(device, &mut self.param_hinge, data.param_arrays.hinge.data,
                      data.param_arrays.hinge.size as usize, "param.hinge")?;
        // THE HINGE TYPES, widened from the scene's `u8`. Topology, so this is
        // the only time it uploads.
        self.hinge_kind.size(device, hinges, AllocLabel("hinge.kind"))?;
        if hinges > 0 {
            let kind: &[u8] = crate::driver::scene::slice(&data.mesh.ttype.hinge);
            for (slot, value) in self.hinge_kind.at().iter_mut().zip(kind.iter()) {
                *slot = u32::from(*value);
            }
        }
        self.hinge_kind.upload(device)?;
        stage_records(device, &mut self.prop_edge, data.prop.edge.data,
                      data.prop.edge.size as usize, "prop.edge")?;
        stage_records(device, &mut self.param_vertex, data.param_arrays.vertex.data,
                      data.param_arrays.vertex.size as usize, "param.vertex")?;
        stage_records(device, &mut self.prop_tet, data.prop.tet.data,
                      data.prop.tet.size as usize, "prop.tet")?;
        stage_records(device, &mut self.param_tet, data.param_arrays.tet.data,
                      data.param_arrays.tet.size as usize, "param.tet")?;
        stage_records(device, &mut self.param_face, data.param_arrays.face.data,
                      data.param_arrays.face.size as usize, "param.face")?;
        stage_records(device, &mut self.param_edge, data.param_arrays.edge.data,
                      data.param_arrays.edge.size as usize, "param.edge")?;

        // ENUMERATED BEFORE THE SIZING BECAUSE IT IS WHAT THE SIZING IS OVER.
        // A site is an interior rod vertex, and neither adjacency it is read
        // from is rewritten after scene build, so this runs once.
        // ENUMERATED INTO A LOCAL, because the stencil table is staged and a
        // staged buffer has no `push`: the walk grows a plain array and the
        // block below sizes the staged pair to it and uploads once. The count
        // it returns is what every rod bending allocation is sized over, so the
        // walk has to run before them either way.
        let mut rod_bend_node = Vec::new();
        let rod_bend_sites = enumerate_rod_bend_sites(
            data,
            vertices,
            &mut rod_bend_node,
            &mut self.rod_bend.edge,
        )?;
        // THE PAIR LIST ON THE DEVICE, uploaded once: the walk above is the
        // only writer and it runs at scene build.
        {
            let pairs = self.rod_bend.edge.len();
            self.rod_bend
                .edge_device
                .size(device, pairs, AllocLabel("rod_bend.edge"))?;
            if pairs > 0 {
                self.rod_bend.edge_device.at()[..pairs]
                    .copy_from_slice(&self.rod_bend.edge);
            }
            self.rod_bend.edge_device.upload(device)?;
        }
        // THE ROD BENDING SLOT TABLE, REPACKED FROM VERTEX ORDER INTO SITE
        // ORDER, which is the one thing about it that is not a verbatim copy.
        // `builder.rs` keys it by surface VERTEX (`9 * surface_vert_count`
        // entries, non-interior vertices left all-sentinel), which suits a
        // dispatch that walks every surface vertex and tests the
        // two-edges-no-face stencil in the thread. This driver
        // walks a COMPACTED site list instead, so the kernel's element index is
        // a site ordinal, and reading the vertex-keyed table with it would
        // deposit one site's blocks at another site's slots.
        //
        // The nine entries per site are copied whole and in order: the builder
        // emits them for the stencil `[j, i, k]`, which is the same order
        // `rod_bend.node` carries, so only the key changes.
        {
            let sites = rod_bend_sites;
            let per_site = 9usize;
            let source: &[u32] = if data.rod_bend_hess_slots.size == 0 {
                &[]
            } else {
                // Safety: the scene is live and the array holds its stated length.
                std::slice::from_raw_parts(
                    data.rod_bend_hess_slots.data as *const u32,
                    data.rod_bend_hess_slots.size as usize,
                )
            };
            let want = if source.is_empty() { 0 } else { per_site * sites };
            self.rod_bend_hess_slots
                .size(device, want, AllocLabel("rod_bend_hess_slots"))?;
            for site in 0..want / per_site {
                // The site's interior vertex is the middle of its `(j, i, k)`
                // triple, which is the row the builder keyed on.
                let interior = rod_bend_node[3 * site + 1] as usize;
                let from = per_site * interior;
                if from + per_site > source.len() {
                    return Err(Fatal::invariant(format!(
                        "solver driver: rod bending site {site} names interior vertex \
                         {interior}, and the slot table covers {} vertices. The table is keyed \
                         by surface vertex, so it must span every vertex a site can name",
                        source.len() / per_site
                    )));
                }
                self.rod_bend_hess_slots.at()[per_site * site..per_site * (site + 1)]
                    .copy_from_slice(&source[from..from + per_site]);
            }
            self.rod_bend_hess_slots.upload(device)?;
        }
        let fixed_nnz = data.fixed_index_table.nnz as usize;
        // THE BUILD-TIME STITCH SET, which is the one the fixed sparsity was
        // registered for: `builder.rs` inserts all thirty-six index pairs of
        // every stitch it was built with. A step's own set arrives through
        // `update_constraint` and is checked against this count there.
        let stitches = data.constraint.stitch.size as usize;
        // THE GRAIN COUNT, scanned once here: a
        // positive rolling inertia marks a grain and every vertex of every
        // non-SAND scene carries zero. Counting rather than stopping at the
        // first one costs the same walk and gives the dispatch extent.
        let grains = if data.grain_inv_inertia.size == 0 {
            0
        } else {
            // Safety: the scene is live and the array holds its stated length.
            let inertia: &[f32] = std::slice::from_raw_parts(
                data.grain_inv_inertia.data,
                data.grain_inv_inertia.size as usize,
            );
            inertia.iter().filter(|value| **value > 0.0).count()
        };
        self.sizes = Sizes {
            vertices,
            tets,
            faces,
            shell_faces,
            hinges,
            rods,
            stitches,
            rod_bend_sites,
            grains,
            fixed_nnz,
        };

        self.positions
            .size(device, 3 * vertices, AllocLabel("step.positions"))?;
        self.positions_prev
            .size(device, 3 * vertices, AllocLabel("step.positions_prev"))?;
        self.eval_x
            .size(device, 3 * vertices, AllocLabel("step.eval_x"))?;
        self.target
            .size(device, 3 * vertices, AllocLabel("step.target"))?;
        // THE ONLY HOST WRITE THESE TAKE: the scene's build-time pose, seeded
        // once. Every write after this is a dispatch or a device copy.
        //
        // Safety: both `CVec`s hold `vertices` position triples, matching the
        // sizing above, and `Vec3f` is three `f32` with no padding.
        let (curr_seed, prev_seed) = unsafe {
            (
                std::slice::from_raw_parts(data.vertex.curr.data as *const f32, 3 * vertices),
                std::slice::from_raw_parts(data.vertex.prev.data as *const f32, 3 * vertices),
            )
        };
        self.positions.seed(device, curr_seed)?;
        self.positions_prev.seed(device, prev_seed)?;
        self.eval_x.seed(device, curr_seed)?;
        self.velocity
            .size(device, 3 * vertices, AllocLabel("step.velocity"))?;
        self.field.allocate(device, vertices)?;
        self.force
            .size(device, 3 * vertices, AllocLabel("step.force"))?;
        self.residual
            .size(device, 3 * vertices, AllocLabel("step.residual"))?;
        self.dx
            .size(device, 3 * vertices, AllocLabel("step.dx"))?;
        self.diagonal
            .size(device, 9 * vertices, AllocLabel("step.diagonal"))?;
        self.precond_diagonal
            .size(device, 9 * vertices, AllocLabel("step.precond_diagonal"))?;
        self.precond_inverse
            .size(device, 9 * vertices, AllocLabel("step.precond_inverse"))?;
        self.dof_mask
            .size(device, vertices, AllocLabel("step.dof_mask"))?;
        // THE FOLD'S SCRATCH, sized for every level at once: level one writes
        // `ceil(n / W)` words, level two `ceil(that / W)`, and the geometric
        // tail sums to under `2 * ceil(n / W)`. One allocation rather than one
        // per level, which is what `Buffer::size` reusing its span is for.
        let blocks = vertices.div_ceil(DOF_FOLD_WIDTH);
        self.dof_fold
            .size(device, 2 * blocks.max(1), AllocLabel("step.dof_fold"))?;
        self.dof_total
            .size(device, 1, AllocLabel("step.dof_total"))?;
        self.fix_index
            .size(device, vertices, AllocLabel("step.fix_index"))?;
        // Sized here for the same reason as everything above: a Newton
        // iteration must not allocate.
        self.pcg.size_for(device, vertices as u32)?;
        self.scalar
            .size(device, vertices.max(1), AllocLabel("step.scalar"))?;
        self.scalar_b
            .size(device, vertices.max(1), AllocLabel("step.scalar_b"))?;
        self.fixed_values
            .size(device, 9 * fixed_nnz, AllocLabel("csr.fixed_value"))?;
        self.tmp_fixed_values
            .size(device, 9 * fixed_nnz, AllocLabel("csr.tmp_fixed_value"))?;

        let t = &mut self.tet;
        t.deformation
            .size(device, 9 * tets, AllocLabel("tet.deformation"))?;
        t.svd_u.size(device, 9 * tets, AllocLabel("tet.svd_u"))?;
        t.svd_sigma
            .size(device, 3 * tets, AllocLabel("tet.svd_sigma"))?;
        t.svd_vt.size(device, 9 * tets, AllocLabel("tet.svd_vt"))?;
        t.gradient_sigma
            .size(device, 3 * tets, AllocLabel("tet.gradient_sigma"))?;
        t.hessian_sigma
            .size(device, 9 * tets, AllocLabel("tet.hessian_sigma"))?;
        t.accepted.size(device, tets, AllocLabel("tet.accepted"))?;
        t.model.size(device, tets, AllocLabel("tet.model"))?;
        t.mu.size(device, tets, AllocLabel("tet.mu"))?;
        t.lambda.size(device, tets, AllocLabel("tet.lambda"))?;
        t.mass
            .size(device, tets, AllocLabel("tet.mass"))?;
        t.damping.size(device, tets, AllocLabel("tet.damping"))?;
        // DEVICE ALLOCATIONS, so what reports a failure here is the seam's
        // allocator rather than `try_reserve`. Both are named refusals at
        // `initialize()`, which is the property the helpers above exist for.
        t.gradient_f
            .size(device, 9 * tets, AllocLabel("tet.gradient_f"))?;
        t.hessian_f
            .size(device, 81 * tets, AllocLabel("tet.hessian_f"))?;
        t.converted_force
            .size(device, 12 * tets, AllocLabel("tet.converted_force"))?;
        t.converted_hessian
            .size(device, 144 * tets, AllocLabel("tet.converted_hessian"))?;
        t.gradient_x
            .size(device, 12 * tets, AllocLabel("tet.gradient_x"))?;
        t.hessian_x
            .size(device, 144 * tets, AllocLabel("tet.hessian_x"))?;
        size_u32(&mut t.push_row, 16 * tets, "the tet push rows")?;
        size_u32(&mut t.push_column, 16 * tets, "the tet push columns")?;
        size(&mut t.push_block, 144 * tets, "the tet push blocks")?;
        size_u32(&mut t.push_stored, 16 * tets, "the tet push verdicts")?;
        t.scatter_index
            .size(device, 4 * tets, AllocLabel("tet.scatter_index"))?;
        t.scatter_gradient
            .size(device, 12 * tets, AllocLabel("tet.scatter_gradient"))?;

        let f = &mut self.face;
        // NO MATERIAL BUFFERS. The membrane reads its material off the face's
        // own records, so no staged array carries it and no per-iteration loop
        // fills one.
        reserve(&mut f.active, shell_faces, "the active face list")?;

        let h = &mut self.hinge;
        h.remapped
            .size(device, 4 * hinges, AllocLabel("hinge.remapped"))?;
        if h.remapped_hinges != hinges {
            h.remapped_hinges = 0;
        }
        h.areal_density
            .size(device, hinges, AllocLabel("hinge.areal_density"))?;
        h.stiffness
            .size(device, hinges, AllocLabel("hinge.stiffness"))?;
        h.damping.size(device, hinges, AllocLabel("hinge.damping"))?;
        size_u32(&mut h.push_row, 16 * hinges, "the hinge push rows")?;
        size_u32(&mut h.push_column, 16 * hinges, "the hinge push columns")?;
        size(&mut h.push_block, 144 * hinges, "the hinge push blocks")?;
        size_u32(&mut h.push_stored, 16 * hinges, "the hinge push verdicts")?;

        let sc = &mut self.stitch;
        sc.index
            .size(device, 6 * stitches, AllocLabel("stitch.index"))?;
        sc.weight
            .size(device, 6 * stitches, AllocLabel("stitch.weight"))?;
        sc.stiffness
            .size(device, stitches, AllocLabel("stitch.stiffness"))?;
        sc.length_factor
            .size(device, stitches, AllocLabel("stitch.length_factor"))?;
        // SIZED OVER THE VERTICES AND NOT OVER THE SLOTS, because the shared
        // body gathers them at the vertex ids the slot array names. Allocated
        // on a stitch-free scene as nothing, since the layer returns before
        // reading them.
        sc.vertex_ghat.size(
            device,
            if stitches == 0 { 0 } else { vertices },
            AllocLabel("stitch.vertex_ghat"),
        )?;
        sc.vertex_offset.size(
            device,
            if stitches == 0 { 0 } else { vertices },
            AllocLabel("stitch.vertex_offset"),
        )?;
        sc.gradient
            .size(device, 18 * stitches, AllocLabel("stitch.gradient"))?;
        sc.hessian
            .size(device, 324 * stitches, AllocLabel("stitch.hessian"))?;
        size_u32(&mut sc.push_row, 36 * stitches, "the stitch push rows")?;
        size_u32(&mut sc.push_column, 36 * stitches, "the stitch push columns")?;
        size(&mut sc.push_block, 324 * stitches, "the stitch push blocks")?;
        size_u32(&mut sc.push_stored, 36 * stitches, "the stitch push verdicts")?;

        let sites = rod_bend_sites;
        let b = &mut self.rod_bend;
        // THE ONE UPLOAD. The stencils were enumerated above and nothing
        // rewrites them, so `handle()` stays current for the run's life.
        b.node
            .size(device, 3 * sites, AllocLabel("rod_bend.node"))?;
        b.node.at().copy_from_slice(&rod_bend_node);
        b.node.upload(device)?;
        b.rest_angle
            .size(device, sites, AllocLabel("rod_bend.rest_angle"))?;
        b.scatter_index
            .size(device, 3 * sites, AllocLabel("rod_bend.scatter_index"))?;
        b.scatter_gradient
            .size(device, 9 * sites, AllocLabel("rod_bend.scatter_gradient"))?;

        // THE TWO STRAIN LIMITERS AND THE STRETCH INDICATOR. Each is sized over
        // the same prefix its dispatch walks, which is `shell_face_count` for
        // the shell half and `rod_count` for the rod half, never the whole face
        // or edge array.
        let s = &mut self.face_strain;
        // NO PUSH OR SCATTER STAGING. The four host push vectors and the
        // scatter gradient would carry a Hessian and a force between dispatches
        // that do not exist here: `shell_strain_embed` scatters and pushes from
        // registers.
        s.toi.size(device, shell_faces, AllocLabel("face_strain.toi"))?;
        reserve(&mut s.candidate, shell_faces, "the limited face list")?;

        let rs = &mut self.rod_strain;
        rs.limit.size(device, rods, AllocLabel("rod_strain.limit"))?;
        rs.rest_length
            .size(device, rods, AllocLabel("rod_strain.rest_length"))?;
        rs.mass.size(device, rods, AllocLabel("rod_strain.mass"))?;
        rs.force_raw
            .size(device, 6 * rods, AllocLabel("rod_strain.force_raw"))?;
        rs.hessian_raw
            .size(device, 36 * rods, AllocLabel("rod_strain.hessian_raw"))?;
        rs.strain.size(device, rods, AllocLabel("rod_strain.strain"))?;
        rs.ok.size(device, rods, AllocLabel("rod_strain.ok"))?;
        rs.stiffness
            .size(device, rods, AllocLabel("rod_strain.stiffness"))?;
        rs.gradient_x
            .size(device, 6 * rods, AllocLabel("rod_strain.gradient_x"))?;
        rs.hessian_x
            .size(device, 36 * rods, AllocLabel("rod_strain.hessian_x"))?;
        size_u32(&mut rs.push_row, 4 * rods, "the rod strain push rows")?;
        size_u32(&mut rs.push_column, 4 * rods, "the rod strain push columns")?;
        size(&mut rs.push_block, 36 * rods, "the rod strain push blocks")?;
        size_u32(&mut rs.push_stored, 4 * rods, "the rod strain push verdicts")?;
        rs.scatter_index
            .size(device, 2 * rods, AllocLabel("rod_strain.scatter_index"))?;
        rs.scatter_gradient
            .size(device, 6 * rods, AllocLabel("rod_strain.scatter_gradient"))?;
        rs.toi.size(device, rods, AllocLabel("rod_strain.toi"))?;
        reserve(&mut rs.candidate, rods, "the limited rod list")?;
        reserve(&mut rs.active, rods, "the active limited rod list")?;

        let st = &mut self.stretch;
        st.deformation
            .size(device, 6 * shell_faces, AllocLabel("stretch.deformation"))?;
        st.svd_u
            .size(device, 6 * shell_faces, AllocLabel("stretch.svd_u"))?;
        st.sigma
            .size(device, 2 * shell_faces, AllocLabel("stretch.sigma"))?;
        st.svd_vt
            .size(device, 4 * shell_faces, AllocLabel("stretch.svd_vt"))?;
        st.largest
            .size(device, shell_faces, AllocLabel("stretch.largest"))?;
        st.shrink_min
            .size(device, shell_faces, AllocLabel("stretch.shrink_min"))?;
        st.ratio
            .size(device, shell_faces.max(1), AllocLabel("stretch.ratio"))?;
        st.rod_ratio
            .size(device, rods.max(1), AllocLabel("stretch.rod_ratio"))?;

        // THE PLASTIC CREEP, sized per enabled kernel rather than over every
        // element class. Each condition tests the material AND the element
        // count, and both halves matter: a tet asset gives its SURFACE faces
        // the object's `plasticity`, so the material test alone would size the
        // face group on a scene with no shell face for the creep to run over.
        // `PlasticKinds` is the same record `backend.rs` asks whether to write
        // a per-frame rest shape, so the pass that mutates and the file that
        // preserves it cannot disagree about which arrays are live.
        let pl = &mut self.plastic;
        pl.kinds = crate::plastic_state::PlasticKinds::of(data);
        let face_creep = if pl.kinds.face { shell_faces } else { 0 };
        pl.face_plasticity
            .size(device, face_creep, AllocLabel("plastic.face_plasticity"))?;
        pl.face_threshold
            .size(device, face_creep, AllocLabel("plastic.face_threshold"))?;
        pl.face_alpha
            .size(device, face_creep, AllocLabel("plastic.face_alpha"))?;
        pl.face_deformation
            .size(device, 6 * face_creep, AllocLabel("plastic.face_deformation"))?;
        pl.face_svd_u
            .size(device, 6 * face_creep, AllocLabel("plastic.face_svd_u"))?;
        pl.face_svd_sigma
            .size(device, 2 * face_creep, AllocLabel("plastic.face_svd_sigma"))?;
        pl.face_svd_vt
            .size(device, 4 * face_creep, AllocLabel("plastic.face_svd_vt"))?;
        pl.face_sigma_new
            .size(device, 2 * face_creep, AllocLabel("plastic.face_sigma_new"))?;
        pl.face_changed
            .size(device, face_creep, AllocLabel("plastic.face_changed"))?;
        pl.face_inverse_rest
            .size(device, 4 * face_creep, AllocLabel("plastic.face_inverse_rest"))?;
        reserve(&mut pl.face_active, face_creep, "the plastic face list")?;

        let tet_creep = if pl.kinds.tet { tets } else { 0 };
        pl.tet_plasticity
            .size(device, tet_creep, AllocLabel("plastic.tet_plasticity"))?;
        pl.tet_threshold
            .size(device, tet_creep, AllocLabel("plastic.tet_threshold"))?;
        pl.tet_alpha
            .size(device, tet_creep, AllocLabel("plastic.tet_alpha"))?;
        pl.tet_deformation
            .size(device, 9 * tet_creep, AllocLabel("plastic.tet_deformation"))?;
        pl.tet_svd_u
            .size(device, 9 * tet_creep, AllocLabel("plastic.tet_svd_u"))?;
        pl.tet_svd_sigma
            .size(device, 3 * tet_creep, AllocLabel("plastic.tet_svd_sigma"))?;
        pl.tet_svd_vt
            .size(device, 9 * tet_creep, AllocLabel("plastic.tet_svd_vt"))?;
        pl.tet_sigma_new
            .size(device, 3 * tet_creep, AllocLabel("plastic.tet_sigma_new"))?;
        pl.tet_changed
            .size(device, tet_creep, AllocLabel("plastic.tet_changed"))?;
        pl.tet_inverse_rest
            .size(device, 9 * tet_creep, AllocLabel("plastic.tet_inverse_rest"))?;
        reserve(&mut pl.tet_active, tet_creep, "the plastic tet list")?;

        let hinge_creep = if pl.kinds.hinge { hinges } else { 0 };
        pl.hinge_node
            .size(device, 4 * hinge_creep, AllocLabel("plastic.hinge_node"))?;
        pl.hinge_plasticity
            .size(device, hinge_creep, AllocLabel("plastic.hinge_plasticity"))?;
        pl.hinge_threshold
            .size(device, hinge_creep, AllocLabel("plastic.hinge_threshold"))?;
        pl.hinge_alpha
            .size(device, hinge_creep, AllocLabel("plastic.hinge_alpha"))?;
        pl.hinge_angle
            .size(device, hinge_creep, AllocLabel("plastic.hinge_angle"))?;
        pl.hinge_rest_angle
            .size(device, hinge_creep, AllocLabel("plastic.hinge_rest_angle"))?;
        size(&mut pl.hinge_rest_scratch, hinge_creep, "the hinge rest-angle seed")?;
        pl.hinge_changed
            .size(device, hinge_creep, AllocLabel("plastic.hinge_changed"))?;
        reserve(&mut pl.hinge_active, hinge_creep, "the plastic hinge list")?;
        // The permutation is topology, fixed at scene build, so it is taken
        // once here rather than on every step.
        let nodes = pl.hinge_node.handle();
        super::plasticity::fill_hinge_nodes(device, self.mesh_hinge.handle(), hinge_creep, nodes)?;

        let rod_creep = if pl.kinds.rod_bend { rod_bend_sites } else { 0 };
        pl.rod_plasticity
            .size(device, rod_creep, AllocLabel("plastic.rod_plasticity"))?;
        pl.rod_threshold
            .size(device, rod_creep, AllocLabel("plastic.rod_threshold"))?;
        pl.rod_alpha
            .size(device, rod_creep, AllocLabel("plastic.rod_alpha"))?;
        pl.rod_angle
            .size(device, rod_creep, AllocLabel("plastic.rod_angle"))?;
        pl.rod_rest_angle
            .size(device, rod_creep, AllocLabel("plastic.rod_rest_angle"))?;
        size(&mut pl.rod_rest_scratch, rod_creep, "the rod rest-angle seed")?;
        pl.rod_changed
            .size(device, rod_creep, AllocLabel("plastic.rod_changed"))?;
        reserve(&mut pl.rod_active, rod_creep, "the plastic rod bending site list")?;

        validate_vertex_face_table(data, vertices, faces)?;
        Ok(())
    }

    /// Replace the stashed pin arrays with this step's.
    ///
    /// Copies rather than borrows; see the module docs for why that is not an
    /// optimization to remove.
    pub fn stash_pins<D: Device>(
        &mut self,
        device: &mut D,
        fix: &[FixPair],
        pull: &[PullPair],
    ) -> FatalResult<()> {
        // A PLAIN DEVICE BUFFER, not a staged one: the rewind pass WRITES these
        // on the device, and the host reads only `len` and `is_empty` off the
        // allocation rather than any contents. Keeping a host copy would give it
        // one that silently diverged the first time the rewind ran.
        self.fix.size(device, fix.len(), AllocLabel("step.fix"))?;
        if !fix.is_empty() {
            self.fix.write(device, 0, fix)?;
        }
        // THE PULL PINS ARE A DEVICE ALLOCATION. They are written once here from
        // the incoming constraint and read only by the momentum pass, so a
        // `StagedBuffer` serves: `handle()` refuses until the upload below has
        // run, which is what stops a step assembling against last step's pins.
        self.pull
            .size(device, pull.len(), AllocLabel("step.pull"))?;
        self.pull.at()[..pull.len()].copy_from_slice(pull);
        self.pull.upload(device)?;
        Ok(())
    }

    /// Re-stage the inverse rest matrices after a streamed rest-shape update.
    ///
    /// A STREAMED REST SHAPE ARRIVES IN THE HOST `DataSet`, while every elastic
    /// and strain-limit dispatch reads
    /// these staged buffers, seeded once at `allocate()`, so without this a
    /// scene carrying rest-shape keyframes simulates against the BUILD-TIME rest
    /// pose for the whole run: no panic, no stale-handle complaint, and a
    /// silently wrong answer for the elastic force, its Hessian, the SVD the
    /// strain limiter takes and the `SL_toi` sweep alike.
    ///
    /// A WHOLESALE COPY IS RIGHT HERE, unlike the vertex props. The creep is the
    /// other author of these two buffers, and the two cannot collide: the
    /// frontend refuses to ship plasticity and a streamed rest shape together,
    /// so there is no crept value to preserve.
    ///
    /// THE STAGED BUFFERS ARE FLAT FLOATS, four and nine per element, which is
    /// the layout `allocate()` gives them and the layout a `Mat2x2f` and a
    /// `Mat3x3f` already have, so each row reinterprets in place.
    pub fn restage_rest_shape<D: Device>(
        &mut self,
        device: &mut D,
        inv2x2: &[crate::data::Mat2x2f],
        inv3x3: &[crate::data::Mat3x3f],
    ) -> FatalResult<()> {
        let n2 = 4 * inv2x2.len();
        if n2 > 0 && self.inv_rest2x2.len() == n2 {
            // Safety: `Mat2x2f` is four contiguous floats and the slice holds
            // `inv2x2.len()` of them.
            let flat = unsafe {
                std::slice::from_raw_parts(inv2x2.as_ptr() as *const f32, n2)
            };
            // A SEED RATHER THAN A STAGED WRITE: the buffer is a readback now
            // that the plastic commit is a kernel, and a restore is the other
            // writer the seed exists for.
            self.inv_rest2x2.seed(device, flat)?;
        }
        let n3 = 9 * inv3x3.len();
        if n3 > 0 && self.inv_rest3x3.len() == n3 {
            // Safety: `Mat3x3f` is nine contiguous floats, same reasoning.
            let flat = unsafe {
                std::slice::from_raw_parts(inv3x3.as_ptr() as *const f32, n3)
            };
            self.inv_rest3x3.seed(device, flat)?;
        }
        Ok(())
    }

    /// Re-stage the per-vertex pin indices after a pin rebuild.
    ///
    /// `constraint::rebuild` rewrites `fix_index` and `pull_index` in the live
    /// `DataSet` every step, so the device copy has to be refreshed at the same
    /// point. Staged once at `allocate()` they
    /// would carry the BUILD-TIME pin set for the whole run: a pin reaching its
    /// `unpin_time` would be released on the host and still prescribed on the
    /// device, so the momentum row's `fix_index > 0` gate would skip the vertex
    /// entirely while `compute_target` had already stopped driving it.
    ///
    /// ONLY THE TWO INDICES ARE COPIED, and that is the whole reason this is a
    /// method rather than a `copy_from_slice`. The staged buffer is the
    /// AUTHORITY for `rest_bend_angle`: `plasticity.rs` creeps it through `at()`
    /// every step and pushes it back to the `DataSet` only at checkpoint time,
    /// so a wholesale copy would reset every crept rod rest angle to its
    /// build-time value.
    pub fn restage_pin_indices<D: Device>(
        &mut self,
        device: &mut D,
        props: &[crate::data::VertexProp],
    ) -> FatalResult<()> {
        let count = self.prop_vertex.len().min(props.len());
        if count == 0 {
            return Ok(());
        }
        {
            let staged = self.prop_vertex.at();
            for i in 0..count {
                staged[i].fix_index = props[i].fix_index;
                staged[i].pull_index = props[i].pull_index;
            }
        }
        self.prop_vertex.upload(device).map_err(|fault| {
            Fatal::device_assert(format!(
                "solver driver: the vertex props failed to reach the device after the pin \
                 rebuild: {fault:?}"
            ))
        })
    }

    /// Re-stage the per-element `fixed` flags after a pin rebuild.
    ///
    /// `constraint::rebuild` recomputes `FaceProp::fixed` and `EdgeProp::fixed`
    /// as the AND over each element's vertices every step, and the CONTACT
    /// kernels read them through `MeshRefs`. Staged once at `allocate()` they
    /// would carry the build-time pin set for the whole run, which is the same
    /// defect as the vertex indices beside it reaching a different consumer.
    ///
    /// A WHOLESALE COPY IS SAFE HERE, unlike the vertex props: nothing else
    /// writes these two buffers, so the `DataSet` is their only author and
    /// there is no crept field to preserve.
    ///
    /// The slices are the scene's own live element props, which is what
    /// `constraint::rebuild` has just written.
    pub fn restage_element_fixed<D: Device>(
        &mut self,
        device: &mut D,
        faces: &[crate::data::FaceProp],
        edges: &[crate::data::EdgeProp],
    ) -> FatalResult<()> {
        let face_count = self.prop_face.len().min(faces.len());
        if face_count > 0 {
            self.prop_face.at()[..face_count].copy_from_slice(&faces[..face_count]);
            self.prop_face.upload(device)?;
        }
        let edge_count = self.prop_edge.len().min(edges.len());
        if edge_count > 0 {
            self.prop_edge.at()[..edge_count].copy_from_slice(&edges[..edge_count]);
            self.prop_edge.upload(device)?;
        }
        Ok(())
    }

    /// Replace this frame's material tables with the streamed ones.
    ///
    /// The animated-parameter path re-derives the four tables per frame and
    /// hands them down through `update_material_params`. Every energy kernel
    /// reads a staged buffer seeded
    /// once at `allocate()`, so each non-empty table is copied in as a staged
    /// write plus an upload, exactly as `restage_rest_shape` is for the rest
    /// pose. Without it a scene carrying material keyframes simulates against
    /// the BUILD-TIME material for the whole run, silently, which is the same
    /// failure and the same silence.
    ///
    /// A TABLE ARRIVES EMPTY WHEN NOTHING ANIMATES IT, and an empty one is
    /// skipped rather than treated as a request to zero the table. The hinge,
    /// edge and
    /// vertex tables are derived from the faces at build, so a schedule that
    /// reaches them ships all four and a face-only schedule ships one.
    ///
    /// THE LENGTHS ARE CHECKED RATHER THAN TRUSTED. These arrive across the C
    /// ABI from the host's own interpolation, and a shorter table would
    /// otherwise leave the tail of the staged buffer holding the build-time
    /// material while the head moved, which no kernel could report. A
    /// mismatch is a host defect, so it fails by name.
    pub fn restage_material_params<D: Device>(
        &mut self,
        device: &mut D,
        faces: &[crate::data::FaceParam],
        vertices: &[crate::data::VertexParam],
        edges: &[crate::data::EdgeParam],
        hinges: &[crate::data::HingeParam],
    ) -> FatalResult<()> {
        fn restage<D: Device, T: Copy + Default + ppf_cts_compute::Pod>(
            device: &mut D,
            buffer: &mut StagedBuffer<T>,
            streamed: &[T],
            what: &str,
        ) -> FatalResult<()> {
            if streamed.is_empty() {
                return Ok(());
            }
            if streamed.len() != buffer.len() {
                return Err(Fatal::invariant(format!(
                    "solver driver: update_material_params was handed {} {what} \
                     materials against the {} this scene was built with",
                    streamed.len(),
                    buffer.len(),
                )));
            }
            buffer.at().copy_from_slice(streamed);
            Ok(buffer.upload(device)?)
        }
        restage(device, &mut self.param_face, faces, "face")?;
        restage(device, &mut self.param_vertex, vertices, "vertex")?;
        restage(device, &mut self.param_edge, edges, "edge")?;
        restage(device, &mut self.param_hinge, hinges, "hinge")?;
        Ok(())
    }

    /// Replace the stashed torque groups and their members with this step's.
    ///
    /// SEPARATE FROM `stash_pins` FOR THE REASON `stash_colliders` IS: they
    /// arrive in the same record and are a different constraint. A group that
    /// switches on at t = 2 appears here and nowhere else, `make_constraint`
    /// being rebuilt from the schedule every step.
    ///
    /// THE RESULT BUFFER IS SIZED HERE AND WRITTEN NOWHERE: the pre-pass fills
    /// every slot it will read, once per assembly, so there is nothing to seed.
    /// Sizing it beside its inputs is what keeps the three lengths agreeing.
    pub fn stash_torque<D: Device>(
        &mut self,
        device: &mut D,
        groups: &[TorqueGroup],
        vertices: &[TorqueVertex],
    ) -> FatalResult<()> {
        self.torque_group
            .size(device, groups.len(), AllocLabel("step.torque.group"))?;
        self.torque_group.at()[..groups.len()].copy_from_slice(groups);
        self.torque_group.upload(device)?;

        self.torque_vertex
            .size(device, vertices.len(), AllocLabel("step.torque.vertex"))?;
        self.torque_vertex.at()[..vertices.len()].copy_from_slice(vertices);
        self.torque_vertex.upload(device)?;

        self.torque_result
            .size(device, groups.len(), AllocLabel("step.torque.result"))?;
        Ok(())
    }

    /// Copy this step's analytic colliders out of the incoming `Constraint`.
    ///
    /// SEPARATE FROM THE PINS ONLY BECAUSE THEY ARRIVE SEPARATELY IN INTENT, not
    /// in the record: both come from `make_constraint`, which the host rebuilds
    /// from the schedule every step, so a collider that switches on at t = 2
    /// appears here and nowhere else.
    pub fn stash_colliders<D: Device>(
        &mut self,
        device: &mut D,
        sphere: &[Sphere],
        floor: &[Floor],
    ) -> FatalResult<()> {
        self.sphere
            .size(device, sphere.len(), AllocLabel("step.sphere"))?;
        self.sphere.at()[..sphere.len()].copy_from_slice(sphere);
        self.sphere.upload(device)?;
        self.floor
            .size(device, floor.len(), AllocLabel("step.floor"))?;
        self.floor.at()[..floor.len()].copy_from_slice(floor);
        self.floor.upload(device)?;
        Ok(())
    }

    /// Unpack this step's cross-stitch records into the flat arrays the shared
    /// body reads.
    ///
    /// COPIED AND DE-INTERLEAVED HERE RATHER THAN PER NEWTON ITERATION, for the
    /// two reasons the pins are copied: the incoming `Constraint` is freed
    /// before `advance()` runs, and `Stitch` interleaves the six indices, the
    /// six weights and the stiffness that the entry point takes as three
    /// separate arrays.
    ///
    /// A COUNT THAT DOES NOT MATCH THE SCENE STOPS THE RUN. The fixed sparsity
    /// carries a 6x6 block set per stitch, registered at scene build from the
    /// set this count was taken from, so a step that brought more stitches than
    /// the scene was built for would push couplings the pattern has no slot
    /// for. That failure is caught again at the push, and it is caught here
    /// first so it names the constraint rather than a block.
    pub fn stash_stitches(
        &mut self,
        device: &mut impl Device,
        stitch: &[Stitch],
    ) -> FatalResult<()> {
        if stitch.len() != self.sizes.stitches {
            return Err(Fatal::invariant(format!(
                "solver driver: this step's constraint carries {} cross-stitches and the scene was \
                 built with {}. The fixed sparsity registers each stitch's thirty-six blocks at \
                 scene build, so a set of a different size does not describe this matrix",
                stitch.len(),
                self.sizes.stitches
            )));
        }
        let s = &mut self.stitch;
        for (element, record) in stitch.iter().enumerate() {
            // THE ONE VALUE THAT COULD BREAK SPD-BY-ASSEMBLY. The spring's two
            // Hessian terms are positive-semidefinite forms and the shared body
            // clamps the coefficients scaling them at zero, so the block it
            // returns is PSD; this factor then multiplies the whole block, and a
            // negative one would flip it. The registry documents the parameter
            // as non-negative, and a scene that arrived with a negative or a
            // non-finite one would surface as a `pAp <= 0` abort several stages
            // away with nothing naming the stitch.
            if !(record.stiffness >= 0.0) {
                return Err(Fatal::invariant(format!(
                    "solver driver: cross-stitch {element} carries stiffness {}, and the stitch \
                     Hessian is a positive-semidefinite block scaled by it, so a negative or \
                     non-finite factor makes the Newton matrix indefinite. `stitch-stiffness` is \
                     non-negative",
                    record.stiffness
                )));
            }
            for slot in 0..6 {
                s.index.at()[6 * element + slot] = record.index[slot];
                s.weight.at()[6 * element + slot] = record.weight[slot];
            }
            s.stiffness.at()[element] = record.stiffness;
        }
        // PUBLISH BEFORE RETURNING. Both arrays are written here and read only
        // by kernels afterwards, and `StagedBuffer::handle` refuses while the
        // host copy is dirty, so a missed upload panics naming the buffer
        // rather than assembling against the previous constraint. This is the
        // point the CUDA backend uploads at.
        s.index.upload(device)?;
        s.weight.upload(device)?;
        s.stiffness.upload(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod restage_tests {
    use super::*;
    use crate::driver::launch::host_device;
    use crate::driver::test_scene::TestScene;

    /// A pin released between steps must reach the DEVICE, not just the host.
    ///
    /// `constraint::rebuild` rewrites `fix_index` in the live `DataSet` every
    /// step and the existing tests in `constraint.rs` cover that. What they
    /// cannot see is the staged buffer the kernels actually read: staged once at
    /// `allocate()`, it carries the build-time pin set for the whole run, so the
    /// momentum row's `fix_index > 0` gate skips a vertex the host has already
    /// released and the vertex assembles no row at all, which is what
    /// `restage_pin_indices` exists to prevent.
    #[test]
    fn a_released_pin_reaches_the_staged_vertex_props() {
        let mut scene = TestScene::new(3);
        scene.vertex_props_mut()[1].fix_index = 2;
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }.expect("the scene allocates");
        assert_eq!(
            state.prop_vertex.host()[1].fix_index, 2,
            "allocate stages the build-time pin set"
        );

        // The pin expires, which `constraint::rebuild` would write here.
        scene.vertex_props_mut()[1].fix_index = 0;
        let props = scene.vertex_props().to_vec();
        state
            .restage_pin_indices(&mut device, &props)
            .expect("the released pin re-stages");
        assert_eq!(
            state.prop_vertex.host()[1].fix_index, 0,
            "the released pin is still prescribed on the device"
        );
    }

    /// A streamed rest shape must reach the DEVICE buffers the kernels read.
    ///
    /// `rest_shape::apply` writes the live `DataSet`, while every elastic and
    /// strain-limit dispatch reads these staged buffers, seeded
    /// once at `allocate()`, so without the re-stage a scene carrying
    /// rest-shape keyframes runs against the BUILD-TIME rest pose: no panic, no
    /// stale-handle complaint, and a wrong elastic force, Hessian, strain SVD
    /// and `SL_toi` sweep alike.
    #[test]
    fn a_streamed_rest_shape_reaches_the_staged_buffers() {
        let mut scene = TestScene::new(3).with_faces(&[crate::data::Vec3u::new(0, 1, 2)]);
        scene.data.inv_rest2x2 =
            crate::cvec::CVec::from(&[crate::data::Mat2x2f::identity()][..]);
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }.expect("the scene allocates");
        assert_eq!(
            state.inv_rest2x2.host()[0], 1.0,
            "allocate stages the build-time rest matrices"
        );

        // The schedule streams a new rest pose, which `rest_shape::apply`
        // writes into the `DataSet`.
        let streamed = [crate::data::Mat2x2f::identity() * 4.0];
        state
            .restage_rest_shape(&mut device, &streamed, &[])
            .expect("the streamed rest shape re-stages");
        assert_eq!(
            state.inv_rest2x2.host()[0], 4.0,
            "the streamed rest matrix never reached the buffer the elastic \
             kernels read"
        );
    }

    /// The `rest_excluded` mask must reach the device in the SAME call that
    /// restages the rest matrices it pairs with.
    ///
    /// `rest_shape::apply` writes the mask into the HOST face props and the
    /// membrane's gate reads it off the DEVICE record
    /// (`kernels/energy/face_force.kernel.cpp`), so the two travel by
    /// different routes. The only other uploader of those props is
    /// `update_constraint`, which owns `fixed` and runs on its own schedule, so
    /// the mask must not be left to it: it would then arrive a frame late, and
    /// on the frame a face is newly flagged near-singular the membrane would
    /// assemble it anyway, against the freshly restaged near-singular
    /// `inv_rest2x2`, which is the exact pairing the exclusion exists to
    /// prevent.
    ///
    /// IT IS INVISIBLE TO EVERY OTHER CHECK. A one-frame stale gate needs a
    /// scene that streams rest-shape keyframes AND flags a face mid-run, which
    /// no fixture does and `--fast-check` could not reach in one frame.
    #[test]
    fn the_rest_excluded_mask_reaches_the_device_with_its_rest_matrices() {
        let mut scene = TestScene::new(3).with_faces(&[crate::data::Vec3u::new(0, 1, 2)]);
        scene.data.inv_rest2x2 =
            crate::cvec::CVec::from(&[crate::data::Mat2x2f::identity()][..]);
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }.expect("the scene allocates");
        assert!(
            !state.prop_face.host()[0].rest_excluded,
            "the face opens unexcluded"
        );

        // The schedule flags the face near-singular, which `rest_shape::apply`
        // writes into the host props, and streams the rest pose that goes with
        // it. This is the pair `update_rest_shape` restages.
        let mut props = scene.face_props().to_vec();
        props[0].rest_excluded = true;
        let streamed = [crate::data::Mat2x2f::identity() * 4.0];
        state
            .restage_rest_shape(&mut device, &streamed, &[])
            .expect("the streamed rest shape re-stages");
        state
            .restage_element_fixed(&mut device, &props, &[])
            .expect("the mask re-stages");

        assert_eq!(
            state.inv_rest2x2.host()[0], 4.0,
            "the streamed rest matrix never reached the buffer the elastic kernels read"
        );
        assert!(
            state.prop_face.host()[0].rest_excluded,
            "the exclusion mask never reached the record the membrane's gate reads, so the \
             face assembles against the near-singular rest matrix restaged beside it"
        );
    }

    /// The re-stage must NOT reset a crept rest angle.
    ///
    /// THE STAGED BUFFER IS THE AUTHORITY for `rest_bend_angle`:
    /// `plasticity.rs` creeps it through `at()` every step and pushes it back to
    /// the `DataSet` only at checkpoint time. A wholesale `copy_from_slice` from
    /// the `DataSet` would therefore look like the obvious way to re-stage the
    /// pin indices and would silently reset every crept rod rest angle to its
    /// build-time value once per step, which is a plasticity that never
    /// accumulates.
    #[test]
    fn the_restage_preserves_a_crept_rest_angle() {
        let mut scene = TestScene::new(3);
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }.expect("the scene allocates");

        // The creep writes the staged buffer, as `plasticity.rs` does, and does
        // NOT write the `DataSet`.
        state.prop_vertex.at()[1].rest_bend_angle = 0.375;
        state.prop_vertex.upload(&mut device).expect("the creep uploads");

        scene.vertex_props_mut()[1].fix_index = 1;
        let props = scene.vertex_props().to_vec();
        state
            .restage_pin_indices(&mut device, &props)
            .expect("the pin re-stages");

        assert_eq!(
            state.prop_vertex.host()[1].fix_index, 1,
            "the new pin index did not reach the staged buffer"
        );
        assert_eq!(
            state.prop_vertex.host()[1].rest_bend_angle, 0.375,
            "the re-stage reset a crept rest angle to its build-time value"
        );
    }
}
