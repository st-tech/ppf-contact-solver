// File: crates/ppf-cts-solver/src/driver/contact.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The contact subsystem: the broad phase, the barrier and friction assembly,
//! the ACCD line search and the intersection gate, wired into one step.
//!
//! # Three things make this solver penetration-free, and the barrier is not one
//!
//! The contact barrier is a CUBIC energy, finite at the surface with a finite
//! gradient there. It has no pole, so it BOUNDS NOTHING: a large enough step
//! walks straight through it. What prevents a crossing is [`Contact::line_search`],
//! which refuses the fraction of a step that would cross, plus
//! [`Contact::check_intersection`], which reports a crossing that got through
//! anyway. The barrier's job is to make the configurations near a contact
//! expensive, not to make them unreachable.
//!
//! All three are live here. A build with the barrier and only one of the other
//! two would COMPLETE and exit 0 with surfaces passed through each other, which
//! is the one outcome worse than refusing the scene, and it is why contact was
//! refused whole until the three arrived together.
//!
//! # The split
//!
//! | shared C++ | here, in Rust |
//! |---|---|
//! | the closest-point coefficients, the barrier, the dynamic stiffness, friction, the widening onto a contact's vertices (`entrypoints/shim_contact.cpp` over the neutral bodies) | which pairs, in what order, and where each result is scattered |
//! | the CCD line search: the traversal, the pair filters and all four ACCD sweeps (`contact/ccd_sweep.kernel.cpp` over `contact/accd.hpp`) | the seed, the dispatch order, and the reduce over the two per-primitive arrays |
//! | the pierce predicate and the allowance rule (`intersection.rs`) | which pairs are candidates |
//!
//! Not one force, Hessian, distance or barrier term is computed in this file,
//! and with the line search on the device not one time of impact is either: no
//! candidate pair is held anywhere.
//!
//! # The scatter is serial, and that is not a performance compromise
//!
//! `compute::atomic_add` on the host seam is a plain read, add and write back,
//! so two contacts sharing a vertex or a CSR slot make a parallel scatter a
//! DATA RACE rather than a different fold order. Every pass below evaluates
//! over disjoint candidate ranges and scatters in one serial pass in ascending
//! pair order, which is also what makes the assembled matrix independent of the
//! thread count.
//!
//! NOTHING HERE CUTS A RANGE ITSELF. Which passes may be split and which must
//! run as one ascending pass is DECLARED, in `super::kernels`, as each row's
//! `Scatter`; the backend reads that and decides the width. The four
//! narrow-phase visitors and the three collision-mesh ones are `Disjoint`,
//! because each candidate pair writes its own staging slot and folds nothing;
//! the force scatters that read those slots are `Atomic`, which is what keeps
//! them serial.
//!
//! # The candidate set is chunked, and the chunk bound is memory
//!
//! One candidate pair stages 144 floats of extended Hessian, so a scene with a
//! million candidates would stage half a gigabyte. The walk therefore runs in
//! fixed-size chunks: evaluate a chunk in parallel, scatter it, move on. Chunks
//! are taken in ascending order and each is scattered before the next is
//! evaluated, so the result is the same at any chunk size.

use crate::data::{DataSet, ParamSet};

use super::ccd::{self, Filter};
use ppf_cts_compute::{AllocLabel, Device, EncoderExt, Fault, Pod, ReadbackBuffer, StagedBuffer};
use super::kernels::OverlapFirstFlaggedLeafArgs;
use super::dyncsr::DynCsrMat;
use super::fixedcsr::FixedCsr;
use super::intersection::{self, Report};
use super::kernels::{
    AabbEdgeContactQueryArgs, AabbEdgeContactQueryMaskedArgs, AabbEdgeScanQueryArgs,
    AabbEdgeScanQueryMaskedArgs,
    AabbLeafActiveArgs, AabbPointContactQueryArgs, AabbPointContactQueryMaskedArgs,
    AabbVertexScanQueryMaskedArgs,
    AabbVertexScanQueryArgs, CcdCollisionEdgeEdgeArgs, CcdCollisionPointFaceC2mArgs,
    CcdCollisionPointFaceM2cArgs, CcdEdgeEdgeArgs, CcdPointFaceArgs, CcdPointPointArgs,
    CollisionEdgeEdgeTraverseArgs,
    CollisionPointFaceC2mTraverseArgs,
    CollisionPointFaceM2cTraverseArgs,
    ContactEdgeEdgeTraverseArgs, ContactPointEdgeTraverseArgs, ContactPointFaceTraverseArgs, ContactPointPointTraverseArgs,
    EdgeCentroidArgs, FaceCentroidArgs, VecFillArgs, VecFillU32Args,
    VertexCentroidArgs};
use super::lbvh::{self, Aabb, Tree};
use super::scene::{Fatal, FatalResult};
use super::bvh;

/// The four contact types, in the order the assembly dispatches them.
///
/// The order is part of the answer: `force` and the two matrices are fp32
/// running sums, so a permutation of the four passes is a different last bit in
/// every entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    PointFace,
    PointEdge,
    PointPoint,
    EdgeEdge,
}

impl Kind {
    const ORDER: [Kind; 4] = [
        Kind::PointFace,
        Kind::PointEdge,
        Kind::PointPoint,
        Kind::EdgeEdge,
    ];
}

/// The contact subsystem's per-run state.
/// Write a rest-pose static array to the device, once.
///
/// # Safety
/// `source` must address `count` elements for the call's duration.
unsafe fn stage_static<D: Device, T: ppf_cts_compute::Pod + Default>(
    device: &mut D,
    buffer: &mut ppf_cts_compute::Buffer<T>,
    source: *const T,
    count: usize,
    label: &'static str,
) -> FatalResult<()> {
    buffer.size(device, count, AllocLabel(label))?;
    if count > 0 {
        buffer.write(device, 0, std::slice::from_raw_parts(source, count))?;
    }
    Ok(())
}

/// The mesh topology, as handles the caller already owns.
///
/// `SolverState` seeds `mesh_face` and `mesh_edge` once at `allocate`, and the
/// mesh does not move after the build, so what the contact subsystem needs is
/// the two handles rather than its own copies. Carried as one value so the five
/// methods that read topology take one parameter instead of two.
#[derive(Clone, Copy)]
pub struct MeshRefs {
    pub face: ppf_cts_compute::Handle,
    pub edge: ppf_cts_compute::Handle,
    /// The per-vertex properties. Staged rather than static, because the
    /// plastic creep still writes them on the host, but device-resident all the
    /// same, so the subsystems that read them take the handle.
    pub vertex_prop: ppf_cts_compute::Handle,
    pub face_prop: ppf_cts_compute::Handle,
    pub edge_prop: ppf_cts_compute::Handle,
    pub vertex_param: ppf_cts_compute::Handle,
    pub face_param: ppf_cts_compute::Handle,
    pub edge_param: ppf_cts_compute::Handle,
    /// The fixed sparsity pattern. Build-time and never rewritten, so the host
    /// readers keep the `DataSet` copy and these agree with it for the run.
    pub fixed_index: ppf_cts_compute::Handle,
    pub fixed_offset: ppf_cts_compute::Handle,
    /// The three neighbor adjacencies, each as its index array and row offsets.
    ///
    /// A table the scene does not carry is `Handle::NONE` here, beside a zero
    /// `has_*` flag the visitors branch on. The flag is what the body reads,
    /// because a body cannot ask whether a buffer is null on a target where a
    /// buffer cannot be null.
    pub vertex_edge_index: ppf_cts_compute::Handle,
    pub vertex_edge_offset: ppf_cts_compute::Handle,
    pub has_vertex_edge: u32,
    pub vertex_face_index: ppf_cts_compute::Handle,
    pub vertex_face_offset: ppf_cts_compute::Handle,
    pub has_vertex_face: u32,
    pub edge_face_index: ppf_cts_compute::Handle,
    pub edge_face_offset: ppf_cts_compute::Handle,
    pub has_edge_face: u32,
    /// Allow Existing Intersections' link table; see [`StartLinkRefs`].
    pub start_link: StartLinkRefs,
}

pub struct Contact {
    /// The device fold's levels; see [`super::reduce::DeviceFold`]. The line
    /// search's two time-of-impact arrays are reduced through it rather than
    /// downloaded and folded on the host.
    fold: super::reduce::DeviceFold,
    /// The ladder that reduces the two overlap report arrays to the one slot
    /// [`Contact::collect_assembly_overlap`] reads.
    overlap_fold: super::reduce::DeviceWordFold,
    /// The scene's per-vertex inverse rolling inertia and angular velocity,
    /// staged once. Non-zero inertia is what makes a vertex a SAND grain; the
    /// omega is what the point-point visitor turns into the contact-point slip
    /// the friction term sees. Both are inert for a scene with no grain.
    grain_inv_inertia: StagedBuffer<f32>,
    /// How many grains the scene carries, so a grain-free scene pays one
    /// comparison and no readback.
    grains: usize,
    /// The three per-vertex friction accumulators, WRITTEN BY THE KERNEL.
    ///
    /// The fused point-point traversal accumulates into them by atomics,
    /// straight into the per-vertex arrays, with no per-pair staging.
    /// The three trees the broad phase walks, over the faces, the edges and
    /// the surface vertices, rebuilt each step.
    face_tree: Tree,
    edge_tree: Tree,
    vertex_tree: Tree,
    /// The Morton codes the tree build sorts, persistent across steps.
    morton: ReadbackBuffer<u32>,
    /// The tree build's own staging, persistent for the same reason.
    tree_build: super::lbvh::BuildScratch,
    /// The query boxes, one per surface vertex and one per edge.
    ///
    /// PLAIN DEVICE BUFFERS, WITH NO HOST MIRROR. A query kernel writes them
    /// and the broad-phase walk reads them through a handle, so no host reader
    /// ever names an element: the box is built inside the dispatch that
    /// traverses with it and the value never leaves the device. A mirror here
    /// would be a download per pass with nothing to consume it.
    point_query: ppf_cts_compute::Buffer<Aabb>,
    edge_query: ppf_cts_compute::Buffer<Aabb>,
    /// The intersection scan's own query boxes, which carry no ghat margin.
    /// Device-resident for the reason above: the scan walks them by handle.
    scan_edge_query: ppf_cts_compute::Buffer<Aabb>,
    scan_vertex_query: ppf_cts_compute::Buffer<Aabb>,
    /// Centroid scratch for the per-step tree rebuild.
    cx: ReadbackBuffer<f32>,
    cy: ReadbackBuffer<f32>,
    cz: ReadbackBuffer<f32>,
    /// The dynamic (contact) Hessian, whose pattern is discovered per step.
    pub matrix: DynCsrMat,
    /// The flattened form the linear operator reads, rebuilt after each
    /// assembly.
    pub flat: super::dyncsr::Flat,
    pub transpose: super::dyncsr::Transpose,
    /// The intersection scan's own device buffers: the two per-element flag
    /// arrays, the bounded record array and its claim counter. Persistent
    /// across steps for the reason every other buffer here is, and the only
    /// state the scan carries now that its four walks run on the device.
    scan: intersection::ScanState,
    /// Per-primitive time of impact, one slot per query, min-folded by the six
    /// CCD sweeps and reduced once by `reduce::min`.
    ///
    /// `toi_vertex` IS AS WIDE AS THE WIDER OF TWO INDEX SPACES: the
    /// collider-to-mesh pass writes at the COLLIDER vertex index into the same
    /// array the three dynamic vertex sweeps write, so the allocation takes the
    /// larger of the two counts.
    toi_vertex: ReadbackBuffer<f32>,
    toi_edge: ReadbackBuffer<f32>,
    /// The overlapping-start report, `ccd::OVERLAP_WORDS` words per query.
    ///
    /// ALLOCATED AS `u32` RATHER THAN AS A RECORD MIRROR so one buffer serves
    /// both the `[[seam::pod(24)]]` field and the `vec_fill_u32` that clears it
    /// before every line search; see `ccd::decode_overlap`.
    overlap_vertex: ReadbackBuffer<u32>,
    overlap_edge: ReadbackBuffer<u32>,
    /// A real zero-length allocation, for a scene that authored no collision
    /// window.
    ///
    /// A generated entry resolves every buffer it is handed BEFORE the body
    /// runs, so an absent mask cannot be `Handle::NONE`, which carries
    /// `u32::MAX` as its arena. The `has_active` flag beside it is what the
    /// body reads; this only has to be resolvable.
    empty_mask: ppf_cts_compute::Buffer<u32>,
    vertices: usize,
    surface_vertices: usize,
    faces: usize,
    edges: usize,
    /// How many contact pairs the last assembly deposited.
    pub assembled: u64,
    /// One tally per self-contact kind, downloaded together after all passes.
    /// The host widens each count before summing, preserving the u64 total.
    assembled_device: ReadbackBuffer<u32>,
    /// The fused collider passes' tally: slot 0 counts ACCEPTED contacts and
    /// slot 1 counts broad-phase CANDIDATES. Separate from
    /// `assembled_device` because the collider passes also report candidates.
    collider_tally: ReadbackBuffer<u32>,

    // The rest-pose STATIC collision mesh. It sits outside the solved namespace
    // and has ONE pose, so its two trees and the query boxes a collider vertex
    // walks the dynamic face tree with are built once at `initialize()` and
    // never refreshed: nothing about them can change between steps.
    collider_face_tree: Tree,
    collider_edge_tree: Tree,
    collider_point_query: ppf_cts_compute::Buffer<Aabb>,
    collider_vertices: usize,
    /// The rest-pose collider mesh, device-resident and seeded once.
    ///
    /// IT JOINS THE POSITION COMPONENT WITHOUT BEING A POSITION: the same
    /// broad-phase records serve it and the deformable arrays, and a record
    /// field is one type. It never moves, so it is written once and never
    /// downloaded.
    collider_vertex: ppf_cts_compute::Buffer<f32>,
    /// The COLLIDER's own per-vertex properties, which are a different array
    /// from the deformable's and are rest-pose static.
    collider_prop: ppf_cts_compute::Buffer<crate::data::VertexProp>,
    /// The REST of the collider's own arrays. All rest-pose static, so each is
    /// written once at `build_collider` and never downloaded; they are separate
    /// buffers from the deformable's because they are separate arrays.
    collider_face: ppf_cts_compute::Buffer<u32>,
    collider_edge: ppf_cts_compute::Buffer<u32>,
    collider_face_prop: ppf_cts_compute::Buffer<crate::data::FaceProp>,
    collider_edge_prop: ppf_cts_compute::Buffer<crate::data::EdgeProp>,
    collider_vertex_param: ppf_cts_compute::Buffer<crate::data::VertexParam>,
    collider_face_param: ppf_cts_compute::Buffer<crate::data::FaceParam>,
    collider_edge_param: ppf_cts_compute::Buffer<crate::data::EdgeParam>,
    collider_faces: usize,
    collider_edges: usize,
}

/// The mirror sibling of [`staged`], for an array the KERNEL writes and a test
/// reads back.
///
/// No PRODUCTION caller: `Contact::allocate` sizes each of its own readback
/// buffers in place. What reads this is the fixture `assemble_against_collider`
/// in this file's test module, which gives the collider assembly a force buffer
/// to read back; the two tests it serves are
/// `a_vertex_over_a_collider_triangle_is_pushed_off_it`, which checks the push
/// barrier's magnitude at a 1 mm gap, and
/// `a_vertex_clear_of_the_collider_contributes_nothing`.
#[allow(dead_code)]
fn readback<T: Pod + Default>(
    device: &mut impl Device,
    count: usize,
    label: &'static str,
) -> Result<ReadbackBuffer<T>, Fault> {
    let mut buffer = ReadbackBuffer::default();
    buffer.size(device, count, AllocLabel(label))?;
    Ok(buffer)
}

impl Contact {
    /// Size every buffer for this scene, once, at `initialize()`.
    ///
    /// # Safety
    /// `data` must address a live `DataSet`.
    pub unsafe fn allocate<D: Device>(device: &mut D, data: &DataSet) -> FatalResult<Self> {
        let vertices = data.vertex.curr.size as usize;
        let surface_vertices = data.surface_vert_count as usize;
        let faces = data.mesh.mesh.face.size as usize;
        let edges = data.mesh.mesh.edge.size as usize;
        if surface_vertices > vertices {
            return Err(Fatal::invariant(format!(
                "solver driver: the scene declares {surface_vertices} contact vertices over a \
                 vertex array of {vertices}. The contact vertices are a PREFIX of that array, \
                 so a longer prefix does not describe this mesh"
            )));
        }
        let collider_vertices = data.constraint.mesh.vertex.size as usize;
        let collider_faces = data.constraint.mesh.face.size as usize;
        let collider_edges = data.constraint.mesh.edge.size as usize;
        // The centroid scratch serves the collider's trees as well as the
        // dynamic mesh's, so it is sized over the widest of all six primitive
        // counts rather than the mesh's three.
        let widest = faces
            .max(edges)
            .max(surface_vertices)
            .max(collider_faces)
            .max(collider_edges)
            .max(collider_vertices);
        let mut live = Contact {
            assembled_device: ReadbackBuffer::default(),
            collider_tally: ReadbackBuffer::default(),
            fold: super::reduce::DeviceFold::default(),
            overlap_fold: super::reduce::DeviceWordFold::default(),
            grain_inv_inertia: Default::default(),
            grains: 0,
            face_tree: Tree::default(),
            edge_tree: Tree::default(),
            vertex_tree: Tree::default(),
            morton: ReadbackBuffer::default(),
            tree_build: super::lbvh::BuildScratch::default(),
            point_query: ppf_cts_compute::Buffer::none(),
            edge_query: ppf_cts_compute::Buffer::none(),
            scan_edge_query: ppf_cts_compute::Buffer::none(),
            scan_vertex_query: ppf_cts_compute::Buffer::none(),
            cx: ReadbackBuffer::default(),
            cy: ReadbackBuffer::default(),
            cz: ReadbackBuffer::default(),
            matrix: DynCsrMat::new(vertices),
            flat: super::dyncsr::Flat::default(),
            transpose: super::dyncsr::Transpose::default(),
            scan: intersection::ScanState::default(),
            toi_vertex: ReadbackBuffer::default(),
            toi_edge: ReadbackBuffer::default(),
            overlap_vertex: ReadbackBuffer::default(),
            overlap_edge: ReadbackBuffer::default(),
            empty_mask: ppf_cts_compute::Buffer::none(),
            vertices,
            surface_vertices,
            faces,
            edges,
            assembled: 0,
            collider_face_tree: Tree::default(),
            collider_edge_tree: Tree::default(),
            collider_point_query: ppf_cts_compute::Buffer::none(),
            collider_vertices,
            collider_vertex: ppf_cts_compute::Buffer::none(),
            collider_prop: ppf_cts_compute::Buffer::none(),
            collider_face: ppf_cts_compute::Buffer::none(),
            collider_edge: ppf_cts_compute::Buffer::none(),
            collider_face_prop: ppf_cts_compute::Buffer::none(),
            collider_edge_prop: ppf_cts_compute::Buffer::none(),
            collider_vertex_param: ppf_cts_compute::Buffer::none(),
            collider_face_param: ppf_cts_compute::Buffer::none(),
            collider_edge_param: ppf_cts_compute::Buffer::none(),
            collider_faces,
            collider_edges,
        };
        // EVERY QUERY BOX IS A DEVICE ALLOCATION AND NONE OF THEM CARRIES A
        // MIRROR: each is written by a query kernel and read by a walk that
        // takes a handle.
        live.edge_query
            .size(device, edges, AllocLabel("contact.edge_query"))?;
        live.scan_edge_query
            .size(device, edges, AllocLabel("contact.scan_edge_query"))?;
        live.scan_vertex_query
            .size(device, surface_vertices, AllocLabel("contact.scan_vertex_query"))?;
        // ONE ALLOCATION OF THE WIDEST ELEMENT COUNT, reused by all five
        // centroid passes; each takes the prefix its own pass writes, which is
        // what `span` is for.
        // THE SAND INPUTS, staged once. A scene with no grain gets zeros, which
        // is what every grain branch in the point-point visitor reads as false.
        {
            let vertices = data.vertex.curr.size as usize;
            live.grain_inv_inertia
                .size(device, vertices, AllocLabel("contact.grain_inv_inertia"))?;
            let inertia = (data.grain_inv_inertia.size as usize).min(vertices);
            if inertia > 0 {
                live.grain_inv_inertia.at()[..inertia].copy_from_slice(
                    std::slice::from_raw_parts(data.grain_inv_inertia.data, inertia),
                );
            }
            live.grain_inv_inertia.upload(device)?;
            live.grains = (0..inertia)
                .filter(|i| live.grain_inv_inertia.host()[*i] > 0.0)
                .count();
        }
        live.cx.size(device, widest, AllocLabel("contact.cx"))?;
        live.cy.size(device, widest, AllocLabel("contact.cy"))?;
        live.cz.size(device, widest, AllocLabel("contact.cz"))?;
        live.point_query
            .size(device, surface_vertices, AllocLabel("contact.point_query"))?;
        // THE CCD LINE SEARCH'S TWO OUTPUT ARRAYS AND THEIR REPORTS. The
        // vertex-space pair is as wide as the wider of the two index spaces
        // that write it, for the reason the field states.
        let ccd_vertex_slots = surface_vertices.max(collider_vertices);
        live.toi_vertex
            .size(device, ccd_vertex_slots, AllocLabel("contact.toi_vertex"))?;
        live.toi_edge
            .size(device, edges, AllocLabel("contact.toi_edge"))?;
        live.overlap_vertex.size(
            device,
            ccd::OVERLAP_WORDS * ccd_vertex_slots,
            AllocLabel("contact.overlap_vertex"),
        )?;
        live.overlap_edge.size(
            device,
            ccd::OVERLAP_WORDS * edges,
            AllocLabel("contact.overlap_edge"),
        )?;
        live.empty_mask
            .size(device, 0, AllocLabel("contact.empty_mask"))?;
        live.collider_point_query.size(
            device,
            collider_vertices,
            AllocLabel("contact.collider_point_query"),
        )?;
        live.build_collider(device, data)?;
        Ok(live)
    }

    /// Build the static collision mesh's two trees and its query boxes.
    ///
    /// ONCE, at `initialize()`, because the collider has one pose:
    /// `update_constraint` deliberately does not re-upload the mesh, so there
    /// is no later pose for a rebuild to track.
    /// The leaves are bounded with the collider's own start and end set to that
    /// one pose, which is the extrapolation-free form of the swept box.
    ///
    /// # Safety
    /// `data` must address a live `DataSet`.
    unsafe fn build_collider<D: Device>(&mut self, device: &mut D, data: &DataSet) -> FatalResult<()> {
        // THROUGH `stage_static`, WHICH IS THE GUARD. A scene with no collider
        // mesh leaves `data.constraint.mesh.vertex.data` NULL with a size of
        // zero, and `slice::from_raw_parts` is undefined behavior on a null
        // pointer even at length zero. Rust's debug precondition check traps
        // it and the panic is NON-UNWINDING, so it aborts the whole test
        // binary rather than failing one case. These two arrays are the
        // collider's own and rest-pose static exactly like the seven below, so
        // they take the same helper rather than a second spelling of the same
        // size-then-write.
        // Safety: the collider mesh holds `collider_vertices` position triples
        // and one prop per collider vertex.
        stage_static(device, &mut self.collider_vertex,
                     data.constraint.mesh.vertex.data as *const f32,
                     3 * self.collider_vertices, "contact.collider_vertex")?;
        stage_static(device, &mut self.collider_prop,
                     data.constraint.mesh.prop.vertex.data,
                     data.constraint.mesh.prop.vertex.size as usize,
                     "contact.collider_prop")?;
        // The rest of the collider's static arrays, written once. Safety: each
        // pointer addresses its own array in the live collider mesh.
        let m = &data.constraint.mesh;
        stage_static(device, &mut self.collider_face,
                     m.face.data as *const u32, 3 * m.face.size as usize, "contact.collider_face")?;
        stage_static(device, &mut self.collider_edge,
                     m.edge.data as *const u32, 2 * m.edge.size as usize, "contact.collider_edge")?;
        stage_static(device, &mut self.collider_face_prop,
                     m.prop.face.data, m.prop.face.size as usize, "contact.collider_face_prop")?;
        stage_static(device, &mut self.collider_edge_prop,
                     m.prop.edge.data, m.prop.edge.size as usize, "contact.collider_edge_prop")?;
        stage_static(device, &mut self.collider_vertex_param,
                     m.param_arrays.vertex.data, m.param_arrays.vertex.size as usize,
                     "contact.collider_vertex_param")?;
        stage_static(device, &mut self.collider_face_param,
                     m.param_arrays.face.data, m.param_arrays.face.size as usize,
                     "contact.collider_face_param")?;
        stage_static(device, &mut self.collider_edge_param,
                     m.param_arrays.edge.data, m.param_arrays.edge.size as usize,
                     "contact.collider_edge_param")?;
        let vertex = self.collider_vertex.handle();
        let face = self.collider_face.handle();
        let edge = self.collider_edge.handle();
        if self.collider_faces > 0 {
            let count = self.collider_faces;
            let centroid = FaceCentroidArgs {
                vert: vertex,
                face: face,
                cx: self.cx.span(0, count),
                cy: self.cy.span(0, count),
                cz: self.cz.span(0, count),
                // The index space the element's slots may name, checked by the
                // entry before it reads a position. Metal returns 0.0 for an
                // out-of-bounds read rather than faulting, so a corrupt index
                // list would be a plausible centroid without this.
                vertex_count: self.collider_vertices as u32,
                count: count as u32,
                seam_arena_count: 0,
};
            device.launch("contact.collider.face_centroid", &centroid, count as u32)?;
            // NO MIRRORS. `build_tree` reduces the bounds, forms the Morton
            // codes, sorts and builds ENTIRELY on the device, taking `cx`,
            // `cy` and `cz` by handle and span; nothing reads their host
            // copies. These three downloads outlived the host tree build they
            // were named for, and on `drape` at 40 frames they were 524 calls
            // and 486.1 MB EACH.
            build_tree(
                device,
                &mut self.cx,
                &mut self.cy,
                &mut self.cz,
                &mut self.morton,
                &mut self.tree_build,
                self.collider_faces,
                &mut self.collider_face_tree,
            )?;
            let pose = vertex;
            lbvh::refresh_face_leaves(
                device,
                &mut self.collider_face_tree,
                pose,
                pose,
                1.0,
                face,
                self.collider_face_prop.handle(),
                self.collider_face_param.handle(),
            )?;
            lbvh::propagate(device, &mut self.collider_face_tree)?;
        }
        if self.collider_edges > 0 {
            let count = self.collider_edges;
            let centroid = EdgeCentroidArgs {
                vert: vertex,
                edge: edge,
                cx: self.cx.span(0, count),
                cy: self.cy.span(0, count),
                cz: self.cz.span(0, count),
                // The index space the element's slots may name, checked by the
                // entry before it reads a position. Metal returns 0.0 for an
                // out-of-bounds read rather than faulting, so a corrupt index
                // list would be a plausible centroid without this.
                vertex_count: self.collider_vertices as u32,
                count: count as u32,
                seam_arena_count: 0,
};
            device.launch("contact.collider.edge_centroid", &centroid, count as u32)?;
            // NO MIRRORS. `build_tree` reduces the bounds, forms the Morton
            // codes, sorts and builds ENTIRELY on the device, taking `cx`,
            // `cy` and `cz` by handle and span; nothing reads their host
            // copies. These three downloads outlived the host tree build they
            // were named for, and on `drape` at 40 frames they were 524 calls
            // and 486.1 MB EACH.
            build_tree(
                device,
                &mut self.cx,
                &mut self.cy,
                &mut self.cz,
                &mut self.morton,
                &mut self.tree_build,
                self.collider_edges,
                &mut self.collider_edge_tree,
            )?;
            let pose = vertex;
            lbvh::refresh_edge_leaves(
                device,
                &mut self.collider_edge_tree,
                pose,
                pose,
                1.0,
                edge,
                self.collider_edge_prop.handle(),
                self.collider_edge_param.handle(),
            )?;
            lbvh::propagate(device, &mut self.collider_edge_tree)?;
        }
        if self.collider_vertices > 0 {
            // A collider vertex walks the DYNAMIC face tree with the same
            // point box a dynamic vertex uses, read off the collider's own
            // materials. No collision window applies: a window masks a
            // primitive of the solved mesh, and the collider carries none.
            let count = self.collider_vertices;
            let query = AabbPointContactQueryArgs {
                x: vertex,
                prop: self.collider_prop.handle(),
                params: self.collider_vertex_param.handle(),
                out: self.collider_point_query.handle(),
                count: count as u32,
                seam_arena_count: 0,
            };
            device.launch("contact.collider.point_query", &query, count as u32)?;
        }
        Ok(())
    }

    /// Rebuild the three trees over the current pose.
    ///
    /// Called once per step: the Morton order is recomputed from the current
    /// positions, so the tree's topology tracks the geometry rather than the
    /// pose it was authored in.
    ///
    /// # Safety
    /// `data` must address a live `DataSet` and `positions` its `3 * vertices`
    /// position components.
    pub unsafe fn rebuild_trees<D: Device>(
        &mut self,
        device: &mut D,
        data: &DataSet,
        mesh: MeshRefs,
        positions: ppf_cts_compute::Handle,
        windows: Windows,
    ) -> FatalResult<()> {
        let face = mesh.face;
        let edge = mesh.edge;
        if self.faces > 0 {
            let count = self.faces;
            let centroid = FaceCentroidArgs {
                vert: positions,
                face: face,
                cx: self.cx.span(0, count),
                cy: self.cy.span(0, count),
                cz: self.cz.span(0, count),
                // The index space the element's slots may name, checked by the
                // entry before it reads a position. Metal returns 0.0 for an
                // out-of-bounds read rather than faulting, so a corrupt index
                // list would be a plausible centroid without this.
                vertex_count: self.vertices as u32,
                count: count as u32,
                seam_arena_count: 0,
};
            device.launch("contact.face_centroid", &centroid, count as u32)?;
            // NO MIRRORS. `build_tree` reduces the bounds, forms the Morton
            // codes, sorts and builds ENTIRELY on the device, taking `cx`,
            // `cy` and `cz` by handle and span; nothing reads their host
            // copies. These three downloads outlived the host tree build they
            // were named for, and on `drape` at 40 frames they were 524 calls
            // and 486.1 MB EACH.
            build_tree(
                device,
                &mut self.cx,
                &mut self.cy,
                &mut self.cz,
                &mut self.morton,
                &mut self.tree_build,
                self.faces,
                &mut self.face_tree,
            )?;
        }
        if self.edges > 0 {
            let count = self.edges;
            let centroid = EdgeCentroidArgs {
                vert: positions,
                edge: edge,
                cx: self.cx.span(0, count),
                cy: self.cy.span(0, count),
                cz: self.cz.span(0, count),
                // The index space the element's slots may name, checked by the
                // entry before it reads a position. Metal returns 0.0 for an
                // out-of-bounds read rather than faulting, so a corrupt index
                // list would be a plausible centroid without this.
                vertex_count: self.vertices as u32,
                count: count as u32,
                seam_arena_count: 0,
};
            device.launch("contact.edge_centroid", &centroid, count as u32)?;
            // NO MIRRORS. `build_tree` reduces the bounds, forms the Morton
            // codes, sorts and builds ENTIRELY on the device, taking `cx`,
            // `cy` and `cz` by handle and span; nothing reads their host
            // copies. These three downloads outlived the host tree build they
            // were named for, and on `drape` at 40 frames they were 524 calls
            // and 486.1 MB EACH.
            build_tree(
                device,
                &mut self.cx,
                &mut self.cy,
                &mut self.cz,
                &mut self.morton,
                &mut self.tree_build,
                self.edges,
                &mut self.edge_tree,
            )?;
        }
        if self.surface_vertices > 0 {
            let count = self.surface_vertices;
            let centroid = VertexCentroidArgs {
                vert: positions,
                cx: self.cx.span(0, count),
                cy: self.cy.span(0, count),
                cz: self.cz.span(0, count),
                count: count as u32,
                seam_arena_count: 0,
};
            device.launch("contact.vertex_centroid", &centroid, count as u32)?;
            // NO MIRRORS. `build_tree` reduces the bounds, forms the Morton
            // codes, sorts and builds ENTIRELY on the device, taking `cx`,
            // `cy` and `cz` by handle and span; nothing reads their host
            // copies. These three downloads outlived the host tree build they
            // were named for, and on `drape` at 40 frames they were 524 calls
            // and 486.1 MB EACH.
            build_tree(
                device,
                &mut self.cx,
                &mut self.cy,
                &mut self.cz,
                &mut self.morton,
                &mut self.tree_build,
                self.surface_vertices,
                &mut self.vertex_tree,
            )?;
        }
        self.refresh_leaves(device, data, mesh, positions, positions, 1.0, windows)
    }

    /// Refresh every leaf box against a swept motion, re-merge, and apply the
    /// collision windows.
    ///
    /// # Safety
    /// `data` must be live and both position arrays `3 * vertices` long.
    pub unsafe fn refresh_leaves<D: Device>(
        &mut self,
        device: &mut D,
        _data: &DataSet,
        mesh: MeshRefs,
        x0: ppf_cts_compute::Handle,
        x1: ppf_cts_compute::Handle,
        extrapolate: f32,
        windows: Windows,
    ) -> FatalResult<()> {
        let face = mesh.face;
        let edge = mesh.edge;
        // The two poses are the same length for every leaf builder: three
        // position components per vertex over the whole solved array, which is
        // what the shared bodies index by the primitive's own vertex ids.
        let start = x0;
        let finish = x1;
        if self.faces > 0 {
            lbvh::refresh_face_leaves(
                device,
                &mut self.face_tree,
                start,
                finish,
                extrapolate,
                face,
                mesh.face_prop,
                mesh.face_param,
            )?;
            set_leaf_active(device, &mut self.face_tree, windows.face)?;
            lbvh::propagate(device, &mut self.face_tree)?;
        }
        if self.edges > 0 {
            lbvh::refresh_edge_leaves(
                device,
                &mut self.edge_tree,
                start,
                finish,
                extrapolate,
                edge,
                mesh.edge_prop,
                mesh.edge_param,
            )?;
            set_leaf_active(device, &mut self.edge_tree, windows.edge)?;
            lbvh::propagate(device, &mut self.edge_tree)?;
        }
        if self.surface_vertices > 0 {
            lbvh::refresh_vertex_leaves(
                device,
                &mut self.vertex_tree,
                start,
                finish,
                extrapolate,
                mesh.vertex_prop,
                mesh.vertex_param,
            )?;
            set_leaf_active(device, &mut self.vertex_tree, windows.vertex)?;
            lbvh::propagate(device, &mut self.vertex_tree)?;
        }
        Ok(())
    }

    /// One Newton iteration's contact assembly.
    ///
    /// `reference` is the ELASTIC SNAPSHOT the dynamic stiffness reads, which is
    /// `tmp_fixed` and never the matrix being written. `force` and `fixed` are
    /// the accumulators, and `self.matrix` takes every block the fixed pattern
    /// has no slot for.
    ///
    /// # Safety
    /// `data` and `param` must be live and every slice sized for the scene.
    #[allow(clippy::too_many_arguments)]
    /// The three per-vertex friction accumulators, uploaded for the integrate.
    ///
    /// The fold writes the host halves, so this is where they reach the device:
    /// once per step, after the last Newton iteration's fold, which is the
    /// converged sum the integrate consumes.
    /// Query boxes must be prepared for `x` and the current materials and
    /// windows. Neither assembly pass modifies those inputs or query boxes.
    pub unsafe fn assemble_prepared<D: Device>(
        &mut self,
        device: &mut D,
        data: &DataSet,
        mesh: MeshRefs,
        param: *const ParamSet,
        x0: ppf_cts_compute::Handle,
        x: ppf_cts_compute::Handle,
        reference: &mut FixedCsr<'_>,
        fixed: &mut FixedCsr<'_>,
        force: ppf_cts_compute::Handle,
        _staging: &mut super::state::PushStaging,
        grain_omega: ppf_cts_compute::Handle,
        // THE THREE PER-VERTEX FRICTION ACCUMULATORS, owned by `SolverState`.
        // Held here they did not exist under `disable-contact`, which is the
        // one thing that gate must not decide: the analytic colliders assemble
        // outside it, so a grain on a floor has spin to integrate either way.
        grain_torque: ppf_cts_compute::Handle,
        grain_stiffness: ppf_cts_compute::Handle,
        grain_normal: ppf_cts_compute::Handle,
        // A COPY OF THE FORCE VECTOR TAKEN BEFORE THIS PASS. See
        // `NarrowPhaseInputs::residual`.
        residual: ppf_cts_compute::Handle,
        statistics: StatisticsRefs,
    ) -> FatalResult<()> {
        self.matrix.start_rebuild();
        self.assembled = 0;
        self.assembled_device.size(device, 4, AllocLabel("contact.assembled"))?;
        device.fill_zero(self.assembled_device.handle(), 4 * std::mem::size_of::<u32>())?;
        // THE SAND ACCUMULATORS ARE ZEROED EACH ITERATION, not each step. A
        // grain's simultaneous contacts must SUM within one iteration, which is
        // what the fold below does, and the post-solve integrate consumes the
        // converged sums once. The clear is gated on the grain count, so a
        // scene with no grain launches nothing.
        if self.grains > 0 {
            // ZEROED ON THE DEVICE, each Newton iteration, because that is
            // where they are written now. A grain's simultaneous contacts must
            // SUM within one iteration and the post-solve integrate consumes
            // the converged sums once, which is why this is per iteration and
            // not per step.
            device.fill_zero(grain_torque, 3 * self.vertices * 4)?;
            device.fill_zero(grain_stiffness, self.vertices * 4)?;
            device.fill_zero(grain_normal, 3 * self.vertices * 4)?;
        }
        let inputs =
            self.narrow_phase_inputs(
                data, mesh, param, x0, x, reference, grain_omega,
                grain_torque, grain_stiffness, grain_normal, residual, statistics,
            );
        for kind in Kind::ORDER {
            // ALL FOUR TRAVERSE AND EMBED IN ONE PASS. Each invokes its
            // narrow phase as a per-hit device functor and builds no pair list,
            // so nothing here walks a tree into one and nothing reads a pair
            // list back to evaluate it in chunks.
            match kind {
                Kind::PointFace => {
                    self.traverse_and_embed_point_face(device, &inputs, fixed, force)?;
                    continue;
                }
                Kind::PointEdge => {
                    self.traverse_and_embed_point_edge(device, &inputs, fixed, force)?;
                    continue;
                }
                Kind::EdgeEdge => {
                    self.traverse_and_embed_edge_edge(device, &inputs, fixed, force)?;
                    continue;
                }
                Kind::PointPoint => {
                    self.traverse_and_embed_point_point(device, &inputs, fixed, force)?;
                    continue;
                }
            }
        }
        self.assembled_device.download(device)?;
        self.assembled = self.assembled_device.host().iter().map(|&count| u64::from(count)).sum();
        self.matrix.finalize();
        // BOTH FILL IN PLACE, reusing the allocations these two fields already
        // hold. This runs once per Newton step, so a form that returned fresh
        // values would allocate seven device buffers per step and, with no
        // `Drop` on a buffer, abandon the previous seven rather than free them.
        self.matrix.to_flat_into(device, &mut self.flat)?;
        self.flat.transpose_into(device, &mut self.transpose)?;
        Ok(())
    }

    /// The vertex and edge query boxes at the Newton iterate.
    ///
    /// ONE DEFINITION FOR BOTH ASSEMBLIES, because the collision-mesh pass walks
    /// the same dynamic-side boxes the self-contact pass does. Built at the
    /// Newton iterate and inflated by the same margin the leaves carry.
    ///
    /// # Safety
    /// `data` must be live and `x` its `3 * vertices` position components.
    /// Prepare iterate boxes for both assembly passes. Swept CCD boxes are
    /// separate and cannot satisfy this preparation.
    pub unsafe fn refresh_contact_queries<D: Device>(
        &mut self,
        device: &mut D,
        _data: &DataSet,
        mesh: MeshRefs,
        x: ppf_cts_compute::Handle,
        windows: Windows,
    ) -> FatalResult<()> {
        if self.surface_vertices > 0 {
            let count = self.surface_vertices;
            // WHICH OF THE TWO ENTRIES, and it is the DRIVER's decision. A
            // scene that authored no collision window has no mask to read, and
            // a record field is a handle with no spelling for absent, so the
            // pair exists and this picks. The two share one body.
            let x = x;
            let prop = mesh.vertex_prop;
            let params = mesh.vertex_param;
            match windows.vertex {
                Some(mask) => {
                    let query = AabbPointContactQueryMaskedArgs {
                        x,
                        prop,
                        params,
                        active: mask,
                        out: self.point_query.handle(),
                        count: count as u32,
                        seam_arena_count: 0,
                    };
                    device.launch("contact.point_query", &query, count as u32)?;
                }
                None => {
                    let query = AabbPointContactQueryArgs {
                        x,
                        prop,
                        params,
                        out: self.point_query.handle(),
                        count: count as u32,
                        seam_arena_count: 0,
                    };
                    device.launch("contact.point_query", &query, count as u32)?;
                }
            }
        }
        if self.edges > 0 {
            let count = self.edges;
            let x = x;
            let edge = mesh.edge;
            let prop = mesh.edge_prop;
            let params = mesh.edge_param;
            match windows.edge {
                Some(mask) => {
                    let query = AabbEdgeContactQueryMaskedArgs {
                        x,
                        edge,
                        prop,
                        params,
                        active: mask,
                        out: self.edge_query.handle(),
                        count: count as u32,
                        seam_arena_count: 0,
                    };
                    device.launch("contact.edge_query", &query, count as u32)?;
                }
                None => {
                    let query = AabbEdgeContactQueryArgs {
                        x,
                        edge,
                        prop,
                        params,
                        out: self.edge_query.handle(),
                        count: count as u32,
                        seam_arena_count: 0,
                    };
                    device.launch("contact.edge_query", &query, count as u32)?;
                }
            }
        }
        Ok(())
    }

    /// The candidate pairs of one contact type, from the broad phase.
    ///
    /// THE WALK IS A DISPATCH, so this takes a device and the query boxes go by
    /// handle rather than by `host()`. The trees are device-resident for the
    /// same reason.
    /// Point-face contact, traversed and embedded in ONE pass.
    ///
    /// THE PAIR LIST NEVER EXISTS. The embed functor is handed to the BVH
    /// traversal: a thread owns a query vertex, the traversal hands it each
    /// overlapping face, and the narrow-phase core runs on the pair in place.
    /// A staged form would walk the tree into a candidate list, download it and
    /// evaluate it in chunks, which is two whole-array transfers per pass.
    ///
    /// # Safety
    /// Every handle the record names must outlive the dispatch.
    unsafe fn traverse_and_embed_point_face<D: Device>(
        &mut self,
        device: &mut D,
        inputs: &NarrowPhaseInputs,
        fixed: &mut FixedCsr<'_>,
        force: ppf_cts_compute::Handle,
    ) -> FatalResult<()> {
        let queries = self.point_query.len();
        if queries == 0 || self.face_tree.node_count == 0 {
            return Ok(());
        }
        let a = inputs;
        let (_fi, _fo, out_fixed_value, _rc) = fixed.device_push_refs();
        // REAL DYNAMIC SLOTS, because a point-face pair NEEDS them: the vertex
        // and the face it touches share no element, so their block is outside
        // the fixed pattern by construction and the dynamic matrix is where it
        // belongs. Zero capacity here is not a stricter rule, it is a dropped
        // coupling.
        //
        // SIZED FROM THE QUERY COUNT, not from a constant. A fused pass has
        // no chunks to reserve against, so its slab must cover the whole pass.
        // Sixteen blocks a QUERY is the worst case expressed against what this
        // pass actually dispatches, bounded by the scene rather than by the
        // candidate count, which is what asked for 4.3 GB once; a fixed
        // per-chunk bound has no meaning here, and measured, edge-edge on
        // `drape` staged more than one in a single pass and stopped on the
        // overflow fatal.
        //
        // An overflow is the loud fatal below rather than a silent drop, and
        // `reserve_device_staging` is what stops the handles being
        // `Handle::NONE`: an unallocated buffer spans to `u32::MAX` and the
        // generated entry's arena assert fires inside a file nobody wrote.
        let capacity = 16 * queries;
        self.matrix.reserve_device_staging(device, capacity)?;
        let dyn_stage = self.matrix.device_staging(device, capacity)?;
        let args = ContactPointFaceTraverseArgs {
            x0: a.x0,
            x: a.x,
            face: a.face,
            vertex_prop: a.vertex_prop,
            start_link_index: a.start_link.index,
            start_link_offset: a.start_link.offset,
            has_start_link: a.start_link.present,
            face_prop: a.face_prop,
            vertex_param: a.vertex_param,
            face_param: a.face_param,
            fixed_index: a.fixed_index,
            fixed_offset: a.fixed_offset,
            fixed_value: a.fixed_value,
            row_count: a.row_count,
            friction_mode: a.friction_mode,
            barrier_id: a.barrier_id,
            friction_eps: a.friction_eps,
            residual: a.residual,
            dt: a.dt,
            out_vertex_force: force,
            out_fixed_value,
            dyn_claim: dyn_stage.claim,
            dyn_row: dyn_stage.row,
            dyn_column: dyn_stage.column,
            dyn_block: dyn_stage.block,
            dyn_capacity: dyn_stage.capacity,
            statistics_contact_count: a.statistics.contact_count,
            statistics_contact_count_size: a.statistics.contact_count_size,
            statistics_object_index: a.statistics.object_index,
            statistics_object_index_size: a.statistics.object_index_size,
            node: self.face_tree.node.handle(),
            node_count: self.face_tree.node_count,
            tree_aabb: self.face_tree.aabb.span(0, self.face_tree.aabb.len()),
            root: self.face_tree.root,
            query: self.point_query.handle(),
            assembled: self.assembled_device.span(0, 1),
            out_overlap: self.overlap_vertex.handle(),
            count: queries as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer the record names is borrowed for the whole call.
        unsafe { device.launch("contact.point_face.traverse", &args, queries as u32) }?;
        self.matrix.take_device_staged(device, capacity).map_err(|fault| {
            let _ = &fault;
            Fatal::invariant(format!(
                "solver driver: the fused point-face traversal staged more than the {capacity} \
                 dynamic blocks reserved for it. Every block past the offer was DROPPED, which \
                 is a lost Hessian coupling; the reserve is one pass wide and needs raising"
            ))
        })?;
        Ok(())
    }

    /// point-edge contact, traversed and embedded in ONE pass.
    ///
    /// The same fusion as [`Self::traverse_and_embed_point_face`], for this
    /// kind; see it for why no pair list exists in this path.
    ///
    /// # Safety
    /// Every handle the record names must outlive the dispatch.
    unsafe fn traverse_and_embed_point_edge<D: Device>(
        &mut self,
        device: &mut D,
        inputs: &NarrowPhaseInputs,
        fixed: &mut FixedCsr<'_>,
        force: ppf_cts_compute::Handle,
    ) -> FatalResult<()> {
        let queries = self.point_query.len();
        if queries == 0 || self.edge_tree.node_count == 0 {
            return Ok(());
        }
        let a = inputs;
        let (_fi, _fo, out_fixed_value, _rc) = fixed.device_push_refs();
        // SIZED FROM THE QUERY COUNT, not from a constant. A fused pass has
        // no chunks to reserve against, so its slab must cover the whole pass.
        // Sixteen blocks a QUERY is the worst case expressed against what this
        // pass actually dispatches, bounded by the scene rather than by the
        // candidate count, which is what asked for 4.3 GB once; a fixed
        // per-chunk bound has no meaning here, and measured, edge-edge on
        // `drape` staged more than one in a single pass and stopped on the
        // overflow fatal.
        //
        // An overflow is the loud fatal below rather than a silent drop, and
        // `reserve_device_staging` is what stops the handles being
        // `Handle::NONE`: an unallocated buffer spans to `u32::MAX` and the
        // generated entry's arena assert fires inside a file nobody wrote.
        let capacity = 16 * queries;
        self.matrix.reserve_device_staging(device, capacity)?;
        let dyn_stage = self.matrix.device_staging(device, capacity)?;
        let record = ContactPointEdgeTraverseArgs {
            statistics_contact_count: a.statistics.contact_count,
            statistics_contact_count_size: a.statistics.contact_count_size,
            statistics_object_index: a.statistics.object_index,
            statistics_object_index_size: a.statistics.object_index_size,
            x0: a.x0,
            x: a.x,
            face: a.face,
            edge: a.edge,
            vertex_prop: a.vertex_prop,
            start_link_index: a.start_link.index,
            start_link_offset: a.start_link.offset,
            has_start_link: a.start_link.present,
            edge_prop: a.edge_prop,
            vertex_param: a.vertex_param,
            edge_param: a.edge_param,
            edge_face_index: a.edge_face_index,
            edge_face_offset: a.edge_face_offset,
            has_edge_face: a.has_edge_face,
            fixed_index: a.fixed_index,
            fixed_offset: a.fixed_offset,
            fixed_value: a.fixed_value,
            row_count: a.row_count,
            friction_mode: a.friction_mode,
            barrier_id: a.barrier_id,
            friction_eps: a.friction_eps,
            residual: a.residual,
            dt: a.dt,
            out_vertex_force: force,
            out_fixed_value,
            dyn_claim: dyn_stage.claim,
            dyn_row: dyn_stage.row,
            dyn_column: dyn_stage.column,
            dyn_block: dyn_stage.block,
            dyn_capacity: dyn_stage.capacity,
            node: self.edge_tree.node.handle(),
            node_count: self.edge_tree.node_count,
            tree_aabb: self.edge_tree.aabb.span(0, self.edge_tree.aabb.len()),
            root: self.edge_tree.root,
            query: self.point_query.handle(),
            assembled: self.assembled_device.span(1, 1),
            out_overlap: self.overlap_vertex.handle(),
            count: queries as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer the record names is borrowed for the whole call.
        unsafe { device.launch("contact.point_edge.traverse", &record, queries as u32) }?;
        self.matrix.take_device_staged(device, capacity).map_err(|fault| {
            let _ = &fault;
            Fatal::invariant(format!(
                "solver driver: the fused point-edge traversal staged more than \
                 the {capacity} dynamic blocks reserved for it; the reserve needs raising"
            ))
        })?;
        Ok(())
    }

    /// edge-edge contact, traversed and embedded in ONE pass.
    ///
    /// The same fusion as [`Self::traverse_and_embed_point_face`], for this
    /// kind; see it for why no pair list exists in this path.
    ///
    /// # Safety
    /// Every handle the record names must outlive the dispatch.
    unsafe fn traverse_and_embed_edge_edge<D: Device>(
        &mut self,
        device: &mut D,
        inputs: &NarrowPhaseInputs,
        fixed: &mut FixedCsr<'_>,
        force: ppf_cts_compute::Handle,
    ) -> FatalResult<()> {
        let queries = self.edge_query.len();
        if queries == 0 || self.edge_tree.node_count == 0 {
            return Ok(());
        }
        let a = inputs;
        let (_fi, _fo, out_fixed_value, _rc) = fixed.device_push_refs();
        // SIZED FROM THE QUERY COUNT, not from a constant. A fused pass has
        // no chunks to reserve against, so its slab must cover the whole pass.
        // Sixteen blocks a QUERY is the worst case expressed against what this
        // pass actually dispatches, bounded by the scene rather than by the
        // candidate count, which is what asked for 4.3 GB once; a fixed
        // per-chunk bound has no meaning here, and measured, edge-edge on
        // `drape` staged more than one in a single pass and stopped on the
        // overflow fatal.
        //
        // An overflow is the loud fatal below rather than a silent drop, and
        // `reserve_device_staging` is what stops the handles being
        // `Handle::NONE`: an unallocated buffer spans to `u32::MAX` and the
        // generated entry's arena assert fires inside a file nobody wrote.
        let capacity = 16 * queries;
        self.matrix.reserve_device_staging(device, capacity)?;
        let dyn_stage = self.matrix.device_staging(device, capacity)?;
        let record = ContactEdgeEdgeTraverseArgs {
            statistics_contact_count: a.statistics.contact_count,
            statistics_contact_count_size: a.statistics.contact_count_size,
            statistics_object_index: a.statistics.object_index,
            statistics_object_index_size: a.statistics.object_index_size,
            x0: a.x0,
            x: a.x,
            edge: a.edge,
            vertex_prop: a.vertex_prop,
            start_link_index: a.start_link.index,
            start_link_offset: a.start_link.offset,
            has_start_link: a.start_link.present,
            edge_prop: a.edge_prop,
            edge_param: a.edge_param,
            fixed_index: a.fixed_index,
            fixed_offset: a.fixed_offset,
            fixed_value: a.fixed_value,
            row_count: a.row_count,
            friction_mode: a.friction_mode,
            barrier_id: a.barrier_id,
            friction_eps: a.friction_eps,
            residual: a.residual,
            dt: a.dt,
            out_vertex_force: force,
            out_fixed_value,
            dyn_claim: dyn_stage.claim,
            dyn_row: dyn_stage.row,
            dyn_column: dyn_stage.column,
            dyn_block: dyn_stage.block,
            dyn_capacity: dyn_stage.capacity,
            node: self.edge_tree.node.handle(),
            node_count: self.edge_tree.node_count,
            tree_aabb: self.edge_tree.aabb.span(0, self.edge_tree.aabb.len()),
            root: self.edge_tree.root,
            query: self.edge_query.handle(),
            assembled: self.assembled_device.span(2, 1),
            out_overlap: self.overlap_edge.handle(),
            count: queries as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer the record names is borrowed for the whole call.
        unsafe { device.launch("contact.edge_edge.traverse", &record, queries as u32) }?;
        self.matrix.take_device_staged(device, capacity).map_err(|fault| {
            let _ = &fault;
            Fatal::invariant(format!(
                "solver driver: the fused edge-edge traversal staged more than \
                 the {capacity} dynamic blocks reserved for it; the reserve needs raising"
            ))
        })?;
        Ok(())
    }

    /// point-point contact, traversed and embedded in ONE pass.
    ///
    /// THE LAST OF THE FOUR. Its SAND grain triple is accumulated into the
    /// per-vertex arrays with atomics, which is the only shape a fused pass can
    /// take: a host fold would need per-pair slots, and a fused pass is
    /// precisely one that never materializes them.
    ///
    /// # Safety
    /// Every handle the record names must outlive the dispatch.
    unsafe fn traverse_and_embed_point_point<D: Device>(
        &mut self,
        device: &mut D,
        inputs: &NarrowPhaseInputs,
        fixed: &mut FixedCsr<'_>,
        force: ppf_cts_compute::Handle,
    ) -> FatalResult<()> {
        let queries = self.point_query.len();
        if queries == 0 || self.vertex_tree.node_count == 0 {
            return Ok(());
        }
        let a = inputs;
        let (_fi, _fo, out_fixed_value, _rc) = fixed.device_push_refs();
        let capacity = 16 * queries;
        self.matrix.reserve_device_staging(device, capacity)?;
        let dyn_stage = self.matrix.device_staging(device, capacity)?;
        let record = ContactPointPointTraverseArgs {
            statistics_contact_count: a.statistics.contact_count,
            statistics_contact_count_size: a.statistics.contact_count_size,
            statistics_object_index: a.statistics.object_index,
            statistics_object_index_size: a.statistics.object_index_size,
            x0: a.x0,
            x: a.x,
            face: a.face,
            edge: a.edge,
            vertex_prop: a.vertex_prop,
            start_link_index: a.start_link.index,
            start_link_offset: a.start_link.offset,
            has_start_link: a.start_link.present,
            vertex_param: a.vertex_param,
            vertex_edge_index: a.vertex_edge_index,
            vertex_edge_offset: a.vertex_edge_offset,
            has_vertex_edge: a.has_vertex_edge,
            vertex_face_index: a.vertex_face_index,
            vertex_face_offset: a.vertex_face_offset,
            has_vertex_face: a.has_vertex_face,
            fixed_index: a.fixed_index,
            fixed_offset: a.fixed_offset,
            fixed_value: a.fixed_value,
            row_count: a.row_count,
            friction_mode: a.friction_mode,
            barrier_id: a.barrier_id,
            friction_eps: a.friction_eps,
            residual: a.residual,
            dt: a.dt,
            grain_inv_inertia: a.grain_inv_inertia,
            grain_omega: a.grain_omega,
            out_vertex_force: force,
            out_fixed_value,
            dyn_claim: dyn_stage.claim,
            dyn_row: dyn_stage.row,
            dyn_column: dyn_stage.column,
            dyn_block: dyn_stage.block,
            dyn_capacity: dyn_stage.capacity,
            grain_torque_vertex: a.grain_torque,
            grain_stiffness_vertex: a.grain_stiffness,
            grain_normal_vertex: a.grain_normal,
            grains_present: u32::from(self.grains > 0),
            node: self.vertex_tree.node.handle(),
            node_count: self.vertex_tree.node_count,
            tree_aabb: self.vertex_tree.aabb.span(0, self.vertex_tree.aabb.len()),
            root: self.vertex_tree.root,
            query: self.point_query.handle(),
            assembled: self.assembled_device.span(3, 1),
            out_overlap: self.overlap_vertex.handle(),
            count: queries as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer the record names is borrowed for the whole call.
        unsafe { device.launch("contact.point_point.traverse", &record, queries as u32) }?;
        self.matrix.take_device_staged(device, capacity).map_err(|_| {
            Fatal::invariant(format!(
                "solver driver: the fused point-point traversal staged more than the \
                 {capacity} dynamic blocks reserved for it; the reserve needs raising"
            ))
        })?;
        Ok(())
    }




    /// The ACCD filter over this step's candidate sweeps.
    ///
    /// SIX DISPATCHES AND NOTHING BETWEEN THEM: one thread per QUERY
    /// primitive, the query box built inside the kernel from that primitive's
    /// own positions and contact margin, the BVH walked there, and the
    /// conservative advance invoked as a per-hit device functor. No candidate
    /// pair is ever held.
    ///
    /// WHAT CROSSES BACK is two float arrays, one slot per primitive, and two
    /// record arrays the host reads only when a fold came out at exactly zero.
    /// The two float arrays should be reduced on the device; no
    /// `[[seam::group]]` entry exists in this tree yet, so the reduce is a
    /// download plus `reduce::min`, which is what the analytic sweep beside
    /// this already does. That is a difference in transport rather than in the
    /// arithmetic.
    ///
    /// Returns the per-primitive times of impact folded to one minimum, and the
    /// first pair that began the step already inside its contact offset. THE
    /// CALLER MUST ASK FOR THE OVERLAP BEFORE USING THE TIME: an overlapping
    /// start returns exactly zero, and reading that as "no progress, try again"
    /// spins forever on a state that cannot resolve.
    ///
    /// # Safety
    /// `data` and `param` must be live, both position arrays `3 * vertices`
    /// long, and the leaves refreshed against this same sweep.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn line_search<D: Device>(
        &mut self,
        device: &mut D,
        // NAMED AND UNREAD, as `refresh_leaves` names it: every array the six
        // sweeps touch reaches them through `mesh` or through a buffer this
        // struct owns, and the `DataSet` stays in the signature because the two
        // call sites hand it to every phase and a phase that stopped taking it
        // would be the odd one out.
        _data: &DataSet,
        mesh: MeshRefs,
        param: &ParamSet,
        x0: ppf_cts_compute::Handle,
        x1: ppf_cts_compute::Handle,
        windows: Windows,
    ) -> FatalResult<Filter> {
        let max_t = param.line_search_max_t;
        let ccd_eps = param.ccd_eps;
        let mut filter = Filter::new(max_t);
        // THE VERTEX-SPACE ARRAY SERVES TWO INDEX SPACES, which is why it is
        // this wide: the collider-to-mesh pass writes at the COLLIDER vertex
        // index into the same array the three dynamic vertex sweeps write, so
        // the allocation takes the larger of the two counts. Sizing it at the
        // surface count alone is an out-of-bounds write that CUDA traps and
        // Metal silently drops, which would leave a time of impact too large.
        let vertex_slots = self.surface_vertices.max(self.collider_vertices);
        if vertex_slots == 0 && self.edges == 0 {
            return Ok(filter);
        }
        // A SCENE WITH NO COLLISION WINDOW STILL NAMES A REAL ALLOCATION. A
        // generated entry resolves every buffer it is handed BEFORE the body
        // runs, so `Handle::NONE` would trip its arena assert; the flag beside
        // it is what the body reads.
        let no_mask = self.empty_mask.handle();
        let (vertex_mask, has_vertex_mask) = match windows.vertex {
            Some(mask) => (mask, 1u32),
            None => (no_mask, 0u32),
        };
        let (edge_mask, has_edge_mask) = match windows.edge {
            Some(mask) => (mask, 1u32),
            None => (no_mask, 0u32),
        };

        // THE SEEDS, on the device. Every sweep MIN-FOLDS into its slot rather
        // than writing it, so each slot has to open at the line-search ceiling,
        // and every overlap slot has to open unflagged or a later read would
        // decode whatever the arena handed out. Metal hands out allocations
        // without zeroing and never faults on an uninitialized read.
        if vertex_slots > 0 {
            let seed = VecFillArgs {
                array: self.toi_vertex.handle(),
                value: max_t,
                count: vertex_slots as u32,
                seam_arena_count: 0,
            };
            device.launch("contact.line_search.toi_vertex", &seed, vertex_slots as u32)?;
            let words = (ccd::OVERLAP_WORDS * vertex_slots) as u32;
            let clear = VecFillU32Args {
                array: self.overlap_vertex.handle(),
                value: 0,
                count: words,
                seam_arena_count: 0,
            };
            device.launch("contact.line_search.overlap_vertex", &clear, words)?;
        }
        if self.edges > 0 {
            let seed = VecFillArgs {
                array: self.toi_edge.handle(),
                value: max_t,
                count: self.edges as u32,
                seam_arena_count: 0,
            };
            device.launch("contact.line_search.toi_edge", &seed, self.edges as u32)?;
            let words = (ccd::OVERLAP_WORDS * self.edges) as u32;
            let clear = VecFillU32Args {
                array: self.overlap_edge.handle(),
                value: 0,
                count: words,
                seam_arena_count: 0,
            };
            device.launch("contact.line_search.overlap_edge", &clear, words)?;
        }

        // THE FOUR VERTEX-SPACE SWEEPS, in a fixed order. The order reaches
        // the answer only through the min-fold, which is order-free, so it is
        // held steady for a reader rather than for the arithmetic.
        //
        // EACH IS GATED ON ITS TREE EXISTING, and the gate is a handle
        // question rather than a node-count one. An unallocated buffer carries
        // `Handle::NONE`, whose arena is `u32::MAX`, and a generated entry
        // asserts every handle's arena BEFORE the body runs, so a sweep cannot
        // be dispatched with the intention of returning early inside it.
        // `rebuild_trees` builds a tree only for a
        // primitive the scene actually has, so a faceless SAND cloud has no
        // face tree at all and the point-face sweep must not be dispatched over
        // it. The tree count is what decides, not the query count.
        if self.surface_vertices > 0 && self.faces > 0 {
            let count = self.surface_vertices as u32;
            let args = CcdPointFaceArgs {
                x0,
                x1,
                face: mesh.face,
                face_count: self.faces as u32,
                vertex_prop: mesh.vertex_prop,
                start_link_index: mesh.start_link.index,
                start_link_offset: mesh.start_link.offset,
                has_start_link: mesh.start_link.present,
                face_prop: mesh.face_prop,
                vertex_param: mesh.vertex_param,
                face_param: mesh.face_param,
                // THE REMAP TREE IS THE VERTEX TREE AND THE TRAVERSAL TREE IS
                // THE FACE TREE, which is why this record names two node
                // arrays: the thread's own surface vertex is remapped through
                // the vertex tree and the query walks the face tree.
                vertex_node: self.vertex_tree.node.handle(),
                node: self.face_tree.node.handle(),
                node_count: self.face_tree.node_count,
                aabb: self.face_tree.aabb.handle(),
                root: self.face_tree.root,
                active: vertex_mask,
                has_active: has_vertex_mask,
                max_t,
                ccd_eps,
                out_toi: self.toi_vertex.handle(),
                out_overlap: self.overlap_vertex.handle(),
                query_count: count,
                count,
                seam_arena_count: 0,
            };
            device.launch("contact.line_search.point_face", &args, count)?;
        }

        // Grain-grain, over the vertex tree. A faceless SAND cloud has no
        // point-face candidate among the mesh primitives, so without this
        // nothing bounds a step that drives two grains through each other, and
        // it is exactly the scene class the guard above skips the face sweep
        // for.
        if self.surface_vertices > 0 {
            let count = self.surface_vertices as u32;
            let args = CcdPointPointArgs {
                x0,
                x1,
                vertex_prop: mesh.vertex_prop,
                start_link_index: mesh.start_link.index,
                start_link_offset: mesh.start_link.offset,
                has_start_link: mesh.start_link.present,
                vertex_param: mesh.vertex_param,
                node: self.vertex_tree.node.handle(),
                node_count: self.vertex_tree.node_count,
                aabb: self.vertex_tree.aabb.handle(),
                root: self.vertex_tree.root,
                active: vertex_mask,
                has_active: has_vertex_mask,
                max_t,
                ccd_eps,
                out_toi: self.toi_vertex.handle(),
                out_overlap: self.overlap_vertex.handle(),
                query_count: count,
                count,
                seam_arena_count: 0,
            };
            device.launch("contact.line_search.point_point", &args, count)?;
        }

        // THE COLLISION-MESH SWEEPS, after the self-contact ones, as
        // `contact::line_search` runs them. The collider has one pose, so its
        // two trees were built at `initialize()` and never refreshed. Each is
        // gated on BOTH sides' trees for the reason the self-contact ones are:
        // `build_collider` builds a tree only for a primitive the collider
        // actually has.
        if self.has_collision_mesh() {
            if self.surface_vertices > 0 && self.collider_faces > 0 {
                let count = self.surface_vertices as u32;
                let args = CcdCollisionPointFaceM2cArgs {
                    x0,
                    x1,
                    collider_vertex: self.collider_vertex.handle(),
                    collider_face: self.collider_face.handle(),
                    collider_face_count: self.collider_faces as u32,
                    vertex_prop: mesh.vertex_prop,
                    start_link_index: mesh.start_link.index,
                    start_link_offset: mesh.start_link.offset,
                    has_start_link: mesh.start_link.present,
                    collider_face_prop: self.collider_face_prop.handle(),
                    vertex_param: mesh.vertex_param,
                    collider_face_param: self.collider_face_param.handle(),
                    node: self.collider_face_tree.node.handle(),
                    node_count: self.collider_face_tree.node_count,
                    aabb: self.collider_face_tree.aabb.handle(),
                    root: self.collider_face_tree.root,
                    active: vertex_mask,
                    has_active: has_vertex_mask,
                    max_t,
                    ccd_eps,
                    out_toi: self.toi_vertex.handle(),
                    out_overlap: self.overlap_vertex.handle(),
                    count,
                    seam_arena_count: 0,
                };
                device.launch("contact.line_search.collision_m2c", &args, count)?;
            }
            if self.collider_vertices > 0 && self.faces > 0 {
                let count = self.collider_vertices as u32;
                let args = CcdCollisionPointFaceC2mArgs {
                    x0,
                    x1,
                    collider_vertex: self.collider_vertex.handle(),
                    face: mesh.face,
                    face_count: self.faces as u32,
                    vertex_prop: mesh.vertex_prop,
                    start_link_index: mesh.start_link.index,
                    start_link_offset: mesh.start_link.offset,
                    has_start_link: mesh.start_link.present,
                    face_prop: mesh.face_prop,
                    collider_vertex_prop: self.collider_prop.handle(),
                    face_param: mesh.face_param,
                    collider_vertex_param: self.collider_vertex_param.handle(),
                    node: self.face_tree.node.handle(),
                    node_count: self.face_tree.node_count,
                    aabb: self.face_tree.aabb.handle(),
                    root: self.face_tree.root,
                    max_t,
                    ccd_eps,
                    out_toi: self.toi_vertex.handle(),
                    out_overlap: self.overlap_vertex.handle(),
                    count,
                    seam_arena_count: 0,
                };
                device.launch("contact.line_search.collision_c2m", &args, count)?;
            }
        }

        // THE VERTEX-SPACE FOLD, taken BEFORE the two edge sweeps because they
        // are seeded from it: an edge
        // collision later than `T_vf` cannot beat an already-found earlier
        // point-face hit, so bounding the edge sweep to `[0, T_vf]` prunes far
        // BVH nodes while leaving the final min bit-identical. The spatial
        // margin is untouched, so coverage of `[0, T_vf]` stays conservative.
        //
        // ONE DOWNLOAD SERVES BOTH READINGS. Nothing writes this array after
        // the four sweeps above, so the value folded here is also the value
        // folded into the filter below.
        let t_vf_seed = if vertex_slots > 0 {
            let values = self.toi_vertex.handle();
            // Safety: the buffer outlives the call and names `vertex_slots`.
            unsafe {
                self.fold
                    .min(device, "contact.toi_vertex", values, vertex_slots as u32, max_t)
            }?
        } else {
            max_t
        };

        if self.edges > 0 {
            let count = self.edges as u32;
            let args = CcdEdgeEdgeArgs {
                x0,
                x1,
                edge: mesh.edge,
                vertex_prop: mesh.vertex_prop,
                start_link_index: mesh.start_link.index,
                start_link_offset: mesh.start_link.offset,
                has_start_link: mesh.start_link.present,
                edge_prop: mesh.edge_prop,
                edge_param: mesh.edge_param,
                node: self.edge_tree.node.handle(),
                node_count: self.edge_tree.node_count,
                aabb: self.edge_tree.aabb.handle(),
                root: self.edge_tree.root,
                active: edge_mask,
                has_active: has_edge_mask,
                t_vf_seed,
                max_t,
                ccd_eps,
                out_toi_ee: self.toi_edge.handle(),
                out_overlap_ee: self.overlap_edge.handle(),
                query_count: count,
                count,
                seam_arena_count: 0,
            };
            device.launch("contact.line_search.edge_edge", &args, count)?;

            if self.has_collision_mesh() && self.collider_edges > 0 {
                let args = CcdCollisionEdgeEdgeArgs {
                    x0,
                    x1,
                    edge: mesh.edge,
                    collider_vertex: self.collider_vertex.handle(),
                    collider_edge: self.collider_edge.handle(),
                    collider_edge_count: self.collider_edges as u32,
                    vertex_prop: mesh.vertex_prop,
                    start_link_index: mesh.start_link.index,
                    start_link_offset: mesh.start_link.offset,
                    has_start_link: mesh.start_link.present,
                    edge_prop: mesh.edge_prop,
                    collider_edge_prop: self.collider_edge_prop.handle(),
                    edge_param: mesh.edge_param,
                    collider_edge_param: self.collider_edge_param.handle(),
                    node: self.collider_edge_tree.node.handle(),
                    node_count: self.collider_edge_tree.node_count,
                    aabb: self.collider_edge_tree.aabb.handle(),
                    root: self.collider_edge_tree.root,
                    active: edge_mask,
                    has_active: has_edge_mask,
                    t_vf_seed,
                    max_t,
                    ccd_eps,
                    out_toi_ee: self.toi_edge.handle(),
                    out_overlap_ee: self.overlap_edge.handle(),
                    count,
                    seam_arena_count: 0,
                };
                device.launch("contact.line_search.collision_edge_edge", &args, count)?;
            }
        }

        filter.fold_time(t_vf_seed);
        let edge_toi = if self.edges > 0 {
            let values = self.toi_edge.handle();
            // Safety: as above.
            unsafe { self.fold.min(device, "contact.toi_edge", values, self.edges as u32, max_t) }?
        } else {
            max_t
        };
        filter.fold_time(edge_toi);

        // THE OVERLAP REPORT IS READ ONLY WHEN THE FOLD CAME OUT AT ZERO, which
        // is what makes a per-query record array free on a healthy step. Every
        // sweep that flags a pair returns exactly zero for it, so a folded time
        // above zero cannot hide a flagged slot; the converse does not hold,
        // which is why the RECORD's own flag decides and not the time.
        if filter.time_of_impact() <= 0.0 {
            self.collect_overlap(device, vertex_slots, &mut filter)?;
        }
        Ok(filter)
    }

    /// Zero both overlap slot arrays. Shared by the assembly and the line
    /// search, which read them under different decoders.
    ///
    /// THE ASSEMBLY CLEARS ONCE, BEFORE ITS FIRST PASS, AND THE CALLER OWNS
    /// THAT because the caller is the only code that knows which pass runs
    /// first. Putting it at the top of [`Contact::assemble`] would be correct
    /// only if the self-contact pass ran before the collision-mesh one, and it
    /// runs the other way round (the analytic and collision-mesh contacts are
    /// assembled first, so a mesh contact's friction anchor reads a residual
    /// with the normal load removed), so a clear there erases every
    /// kind 7, 8 and 9 report the collision-mesh passes have just written,
    /// before the host reads it. Every pass writes its query's slot
    /// first-writer-wins, so a clear BETWEEN two passes silently drops the
    /// earlier one's report; `rig_collider_coincident_pair` is the gate that
    /// covers it.
    pub fn clear_overlap_slots<D: Device>(&mut self, device: &mut D) -> FatalResult<()> {
        let vertex_slots = self.overlap_vertex.len() / ccd::OVERLAP_WORDS;
        if vertex_slots > 0 {
            let words = (ccd::OVERLAP_WORDS * vertex_slots) as u32;
            let clear = VecFillU32Args {
                array: self.overlap_vertex.handle(),
                value: 0,
                count: words,
                seam_arena_count: 0,
            };
            // Safety: the record names one buffer this struct owns, borrowed
            // for the whole call.
            unsafe { device.launch("contact.assembly.overlap_vertex", &clear, words)? };
        }
        if self.edges > 0 {
            let words = (ccd::OVERLAP_WORDS * self.edges) as u32;
            let clear = VecFillU32Args {
                array: self.overlap_edge.handle(),
                value: 0,
                count: words,
                seam_arena_count: 0,
            };
            // Safety: as above.
            unsafe { device.launch("contact.assembly.overlap_edge", &clear, words)? };
        }
        Ok(())
    }

    /// The first pair the ASSEMBLY found already collapsed to its contact
    /// offset, or `None`. Read after both `assemble` and
    /// `assemble_collision_mesh`, since either may have written a slot.
    ///
    /// THE LOWEST-INDEXED FLAGGED SLOT, ascending and across the vertex array
    /// before the edge one, so two runs of one scene name the same pair. A
    /// single latch taken by whichever thread arrived first, under an
    /// `atomicCAS`, would not have that property; a per-query slot needs no
    /// atomic at all, because one thread owns it.
    ///
    /// ONE WORD CROSSES PER NEWTON STEP. The device reduces both report arrays
    /// to the selected slot, and that one record is read only when a slot was
    /// selected; see [`first_assembly_overlap`].
    pub fn collect_assembly_overlap<D: Device>(
        &mut self,
        device: &mut D,
    ) -> FatalResult<Option<ccd::AssemblyOverlap>> {
        let vertex_slots = self.overlap_vertex.len() / ccd::OVERLAP_WORDS;
        first_assembly_overlap(
            device,
            &mut self.overlap_fold,
            &mut self.overlap_vertex,
            vertex_slots,
            &mut self.overlap_edge,
            self.edges,
        )
    }

    fn collect_overlap<D: Device>(
        &mut self,
        device: &mut D,
        vertex_slots: usize,
        filter: &mut Filter,
    ) -> FatalResult<()> {
        if vertex_slots > 0 {
            self.overlap_vertex.download(device)?;
            let words = self.overlap_vertex.host();
            for slot in 0..vertex_slots {
                let base = ccd::OVERLAP_WORDS * slot;
                if let Some(start) = ccd::decode_overlap(&words[base..base + ccd::OVERLAP_WORDS]) {
                    filter.record_overlap(start);
                    return Ok(());
                }
            }
        }
        if self.edges > 0 {
            self.overlap_edge.download(device)?;
            let words = self.overlap_edge.host();
            for slot in 0..self.edges {
                let base = ccd::OVERLAP_WORDS * slot;
                if let Some(start) = ccd::decode_overlap(&words[base..base + ccd::OVERLAP_WORDS]) {
                    filter.record_overlap(start);
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// The final penetration gate.
    ///
    /// # Safety
    /// `data` must be live and `positions` its `3 * vertices` components.
    pub unsafe fn check_intersection<D: Device>(
        &mut self,
        device: &mut D,
        // NAMED AND UNREAD, as `refresh_leaves` names it: every array the four
        // walks touch reaches them through `mesh`, through `positions` or
        // through a buffer this struct owns, and the `DataSet` stays in the
        // signature because both call sites hand it to every phase.
        _data: &DataSet,
        mesh: MeshRefs,
        positions: ppf_cts_compute::Handle,
        // NAMED AND UNREAD. The four walks run on the device and read the pose
        // and the per-vertex properties by handle; these two host mirrors are
        // what the host walk they replaced needed. They stay in the signature
        // because both call sites hand them over and removing a parameter is a
        // change to files this one does not own.
        _positions_host: &[f32],
        _vertex_prop_host: &[crate::data::VertexProp],
        windows: Windows,
    ) -> FatalResult<Report> {
        let scene = intersection::Scene {
            vert: positions,
            face: mesh.face,
            edge: mesh.edge,
            face_prop: mesh.face_prop,
            edge_prop: mesh.edge_prop,
            vert_prop: mesh.vertex_prop,
            edge_param: mesh.edge_param,
            vertex_param: mesh.vertex_param,
            start_link: mesh.start_link,
            faces: self.faces,
            edges: self.edges,
            surface_vertices: self.surface_vertices,
        };
        if self.edges > 0 {
            let count = self.edges;
            let vert = positions;
            let edge = mesh.edge;
            match windows.edge {
                Some(mask) => {
                    let query = AabbEdgeScanQueryMaskedArgs {
                        vert, edge,
                        active: mask,
                        out: self.scan_edge_query.handle(),
                        count: count as u32,
                        seam_arena_count: 0,
                    };
                    device.launch("contact.scan.edge_query", &query, count as u32)?;
                }
                None => {
                    let query = AabbEdgeScanQueryArgs {
                        vert, edge,
                        out: self.scan_edge_query.handle(),
                        count: count as u32,
                        seam_arena_count: 0,
                    };
                    device.launch("contact.scan.edge_query", &query, count as u32)?;
                }
            }
        }
        if self.surface_vertices > 0 {
            let count = self.surface_vertices;
            let vert = positions;
            let prop = mesh.vertex_prop;
            let params = mesh.vertex_param;
            match windows.vertex {
                Some(mask) => {
                    let query = AabbVertexScanQueryMaskedArgs {
                        vert, prop, params,
                        active: mask,
                        out: self.scan_vertex_query.handle(),
                        count: count as u32,
                        seam_arena_count: 0,
                    };
                    device.launch("contact.scan.vertex_query", &query, count as u32)?;
                }
                None => {
                    let query = AabbVertexScanQueryArgs {
                        vert, prop, params,
                        out: self.scan_vertex_query.handle(),
                        count: count as u32,
                        seam_arena_count: 0,
                    };
                    device.launch("contact.scan.vertex_query", &query, count as u32)?;
                }
            }
        }
        // NO READBACK ANYWHERE IN HERE. The query boxes stay on the device
        // and so does everything the four walks read; what comes back at the
        // end is a counter, at most `capacity` records and two flag arrays.
        self.scan
            .begin(device, self.edges, self.surface_vertices)?;
        // EACH WALK IS GATED ON ITS TREE EXISTING, and the gate is a handle
        // question rather than a node-count one. An unallocated buffer carries
        // `Handle::NONE`, whose arena is `u32::MAX`, and a generated entry
        // asserts every handle's arena BEFORE the body runs, so a walk cannot
        // be dispatched with the intention of returning early inside it.
        // `rebuild_trees`
        // builds a tree only for a primitive the scene actually has, so a
        // faceless SAND cloud has no face tree and the face-edge walk must not
        // be dispatched over it.
        let edge_query = self.scan_edge_query.handle();
        let vertex_query = self.scan_vertex_query.handle();
        self.scan
            .scan_face_edge(device, &scene, &mut self.face_tree, edge_query)?;
        self.scan
            .scan_edge_edge(device, &scene, &mut self.edge_tree, edge_query)?;
        self.scan
            .scan_point_point(device, &scene, &mut self.vertex_tree, vertex_query)?;
        // A DYNAMIC EDGE THROUGH THE STATIC COLLISION MESH. The collider is a
        // rest-pose pool outside the solved namespace, so the pair is
        // inter-object by construction and the dynamic edge alone decides
        // whether the crossing is tolerated.
        if self.collider_faces > 0 {
            let collider = intersection::Collider {
                vert: self.collider_vertex.handle(),
                face: self.collider_face.handle(),
                faces: self.collider_faces,
            };
            self.scan.scan_collision_mesh(
                device,
                &scene,
                &collider,
                &mut self.collider_face_tree,
                edge_query,
            )?;
        }
        let report = self.scan.finish(device)?;
        Ok(report)
    }

}

/// The three collision-mesh passes, in the order the constraint assembly
/// dispatches them.
///
/// The order is part of the answer: the force and the fixed matrix are fp32
/// running sums, so a permutation of the three is a different last bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColliderPass {
    /// A dynamic vertex against a collider triangle.
    PointFaceM2c,
    /// A collider vertex against a dynamic triangle.
    PointFaceC2m,
    /// A dynamic edge against a collider edge.
    EdgeEdge,
}

impl ColliderPass {
    fn slot(self) -> usize {
        match self {
            ColliderPass::PointFaceM2c => 0,
            ColliderPass::PointFaceC2m => 1,
            ColliderPass::EdgeEdge => 2,
        }
    }
}

impl Contact {
    /// Whether this scene carries a static collision mesh at all.
    pub fn has_collision_mesh(&self) -> bool {
        self.collider_vertices > 0
    }

    /// One Newton iteration's collision-mesh assembly.
    ///
    /// EVERY BLOCK LANDS IN THE FIXED PATTERN, which is why this runs after the
    /// dynamic matrix has been finalized rather than inside its rebuild window.
    /// A dynamic vertex against a collider triangle contributes to that vertex's
    /// own diagonal; a collider vertex against a dynamic triangle contributes
    /// over that triangle's three vertices; an edge pair contributes over the
    /// dynamic edge's two. `builder.rs` registers all three stencils, so a block
    /// with no slot is a scene the pattern does not describe and the step stops
    /// rather than assembling a Newton matrix missing a coupling.
    ///
    /// # Safety
    /// `data` and `param` must be live and every slice sized for the scene.
    #[allow(clippy::too_many_arguments)]
    /// Query boxes must be prepared for `x` and the current materials and
    /// windows before entering this pass.
    pub unsafe fn assemble_collision_mesh_prepared<D: Device>(
        &mut self,
        device: &mut D,
        data: &DataSet,
        mesh: MeshRefs,
        param: *const ParamSet,
        x0: ppf_cts_compute::Handle,
        x: ppf_cts_compute::Handle,
        reference: &mut FixedCsr<'_>,
        fixed: &mut FixedCsr<'_>,
        force: ppf_cts_compute::Handle,
        _staging: &mut super::state::PushStaging,
        // A COPY OF THE FORCE VECTOR TAKEN BEFORE THIS PASS. See
        // `CollisionInputs::residual`.
        residual: ppf_cts_compute::Handle,
        statistics: StatisticsRefs,
    ) -> FatalResult<()> {
        if !self.has_collision_mesh() {
            return Ok(());
        }
        let inputs =
            self.collision_inputs(data, mesh, param, x0, x, reference, residual, statistics);
        let mut counts = [0usize; 3];
        // THE THREE PASSES, IN THE ORDER `ColliderPass` DECLARES THEM, each
        // traversing and embedding in one dispatch. Written out rather than
        // looped because each names its own record, so a field missing from
        // one is a compile error.
        //
        // Safety: `inputs` names buffers that outlive these calls, and each
        // pass borrows the matrix and the force for the whole dispatch.
        counts[ColliderPass::PointFaceM2c.slot()] = unsafe {
            self.traverse_and_embed_collider_point_face_m2c(device, &inputs, fixed, force)
        }?;
        counts[ColliderPass::PointFaceC2m.slot()] = unsafe {
            self.traverse_and_embed_collider_point_face_c2m(device, &inputs, fixed, force)
        }?;
        counts[ColliderPass::EdgeEdge.slot()] = unsafe {
            self.traverse_and_embed_collider_edge_edge(device, &inputs, fixed, force)
        }?;
        // THE THREE COLLISION-MESH KINDS, COUNTED AS CANDIDATES. They are
        // separate passes over separate candidate lists, and a scene that meant
        // to contact a static collider one way and reached it another way, or
        // not at all, reads as a physics result rather than as a miswired pass.
        //
        // CANDIDATES, NOT ACTIVE PAIRS, and the distinction is why this does
        // not carry the second number the Metal orchestrator prints beside it.
        // That backend reads its compact Hessians back at `initialize()` as
        // part of a mirrored-scene self-check and can say how many pairs
        // survived narrowing and produced a non-zero block; this driver has no
        // such readback, and adding one would be a download per step. Reporting
        // a broad-phase count under the narrowed count's name would be the
        // quieter mistake of the two.
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static SAID: AtomicBool = AtomicBool::new(false);
            // ZERO IS THE INTERESTING CASE HERE, not a reason to stay quiet:
            // a collision window that closes a collider takes every count to
            // zero, and a diagnostic that only speaks when something was found
            // cannot tell that apart from a pass that never ran. The enclosing
            // early return already establishes that the scene HAS a collision
            // mesh, so reaching this line at all is the fact worth reporting.
            //
            // IT IS `debug!` BECAUSE IT IS INSTRUMENTATION, NOT SOLVER
            // BEHAVIOR. This driver's own diagnostics stay off the default
            // transcript, so a transcript diff between two runs shows what the
            // solver did rather than what it counted. The
            // zero-is-interesting argument above and the once-only gate both
            // still apply.
            if !SAID.swap(true, Ordering::Relaxed) {
                ::log::debug!(
                    "collision-mesh broad phase has dynamic-point/static-face {} \
                     candidate(s), static-point/dynamic-face {}, and \
                     dynamic-edge/static-edge {}",
                    counts[0], counts[1], counts[2]
                );
            }
        }
        Ok(())
    }




    /// A dynamic vertex against the collider's face tree.
    ///
    /// TRAVERSES AND EMBEDS IN ONE PASS, so no pair list is built and none is
    /// read back. A staged form would download the whole candidate buffer per
    /// chunk, then the active flag, the arity and the four indices.
    ///
    /// ZERO DYNAMIC SLOTS, WHICH IS THE COLLIDER PATH'S RULE RATHER THAN A
    /// SIZE. A collision-mesh stencil is a vertex's own diagonal, a face's
    /// three vertices or an edge's two, all of which `builder.rs` registers, so
    /// a block the fixed pattern cannot hold means the pattern does not
    /// describe this mesh. `take_device_staged` turns a non-zero claim into the
    /// fatal below, which is where that check has always lived.
    ///
    /// # Safety
    /// Every buffer the record names must outlive the dispatch.
    unsafe fn traverse_and_embed_collider_point_face_m2c<D: Device>(
        &mut self,
        device: &mut D,
        inputs: &CollisionInputs,
        fixed: &mut FixedCsr<'_>,
        force: ppf_cts_compute::Handle,
    ) -> FatalResult<usize> {
        let queries = self.point_query.len();
        if queries == 0 || self.collider_face_tree.node_count == 0 {
            return Ok(0);
        }
        self.collider_tally
            .size(device, 2, AllocLabel("collision.tally"))?;
        { let h = self.collider_tally.handle(); device.fill_zero(h, 8)?; }
        let a = inputs;
        let (_fi, _fo, out_fixed_value, _rc) = fixed.device_push_refs();
        let capacity = 0usize;
        self.matrix.reserve_device_staging(device, capacity)?;
        let dyn_stage = self.matrix.device_staging(device, capacity)?;
        let args = CollisionPointFaceM2cTraverseArgs {
            x0: a.x0,
            x: a.x,
            vertex_prop: a.vertex_prop,
            start_link_index: a.start_link.index,
            start_link_offset: a.start_link.offset,
            has_start_link: a.start_link.present,
            vertex_param: a.vertex_param,
            static_x: a.static_x,
            static_face: a.static_face,
            static_face_prop: a.static_face_prop,
            static_face_param: a.static_face_param,
            fixed_index: a.fixed_index,
            fixed_offset: a.fixed_offset,
            fixed_value: a.fixed_value,
            row_count: a.row_count,
            friction_mode: a.friction_mode,
            friction_eps: a.friction_eps,
            residual: a.residual,
            dt: a.dt,
            out_vertex_force: force,
            out_fixed_value,
            dyn_claim: dyn_stage.claim,
            dyn_row: dyn_stage.row,
            dyn_column: dyn_stage.column,
            dyn_block: dyn_stage.block,
            dyn_capacity: dyn_stage.capacity,
            statistics_contact_count: a.statistics.contact_count,
            statistics_contact_count_size: a.statistics.contact_count_size,
            statistics_object_index: a.statistics.object_index,
            statistics_object_index_size: a.statistics.object_index_size,
            statistics_static_object_index: a.statistics.static_object_index,
            statistics_static_object_index_size: a.statistics.static_object_index_size,
            node: self.collider_face_tree.node.handle(),
            node_count: self.collider_face_tree.node_count,
            tree_aabb: self.collider_face_tree.aabb.span(0, self.collider_face_tree.aabb.len()),
            root: self.collider_face_tree.root,
            query: self.point_query.handle(),
            assembled: self.collider_tally.handle(),
            out_overlap: self.overlap_vertex.handle(),
            count: queries as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer the record names is borrowed for the whole call.
        unsafe { device.launch("collision.point_face_m2c.traverse", &args, queries as u32) }?;
        self.matrix.take_device_staged(device, capacity).map_err(|fault| {
            let _ = &fault;
            Fatal::invariant(
                "solver driver: the fixed sparsity has no slot for a block a collision-mesh \
                 contact contributes. That stencil is a vertex's own diagonal, a face's three \
                 vertices or an edge's two, all of which `builder.rs` registers, so the pattern \
                 does not describe this mesh and the coupling would be dropped"
                    .to_string(),
            )
        })?;
        self.collider_tally.download(device)?;
        self.assembled += u64::from(self.collider_tally.host()[0]);
        Ok(self.collider_tally.host()[1] as usize)
    }

    /// A collider vertex against the DYNAMIC face tree, the one pass that queries the solved mesh.
    ///
    /// TRAVERSES AND EMBEDS IN ONE PASS, so no pair list is built and none is
    /// read back. A staged form would download the whole candidate buffer per
    /// chunk, then the active flag, the arity and the four indices.
    ///
    /// ZERO DYNAMIC SLOTS, WHICH IS THE COLLIDER PATH'S RULE RATHER THAN A
    /// SIZE. A collision-mesh stencil is a vertex's own diagonal, a face's
    /// three vertices or an edge's two, all of which `builder.rs` registers, so
    /// a block the fixed pattern cannot hold means the pattern does not
    /// describe this mesh. `take_device_staged` turns a non-zero claim into the
    /// fatal below, which is where that check has always lived.
    ///
    /// # Safety
    /// Every buffer the record names must outlive the dispatch.
    unsafe fn traverse_and_embed_collider_point_face_c2m<D: Device>(
        &mut self,
        device: &mut D,
        inputs: &CollisionInputs,
        fixed: &mut FixedCsr<'_>,
        force: ppf_cts_compute::Handle,
    ) -> FatalResult<usize> {
        let queries = self.collider_point_query.len();
        if queries == 0 || self.face_tree.node_count == 0 {
            return Ok(0);
        }
        self.collider_tally
            .size(device, 2, AllocLabel("collision.tally"))?;
        { let h = self.collider_tally.handle(); device.fill_zero(h, 8)?; }
        let a = inputs;
        let (_fi, _fo, out_fixed_value, _rc) = fixed.device_push_refs();
        let capacity = 0usize;
        self.matrix.reserve_device_staging(device, capacity)?;
        let dyn_stage = self.matrix.device_staging(device, capacity)?;
        let args = CollisionPointFaceC2mTraverseArgs {
            x0: a.x0,
            x: a.x,
            face: a.face,
            vertex_prop: a.vertex_prop,
            start_link_index: a.start_link.index,
            start_link_offset: a.start_link.offset,
            has_start_link: a.start_link.present,
            face_prop: a.face_prop,
            face_param: a.face_param,
            static_x: a.static_x,
            static_vertex_prop: a.static_vertex_prop,
            static_vertex_param: a.static_vertex_param,
            fixed_index: a.fixed_index,
            fixed_offset: a.fixed_offset,
            fixed_value: a.fixed_value,
            row_count: a.row_count,
            friction_mode: a.friction_mode,
            friction_eps: a.friction_eps,
            residual: a.residual,
            dt: a.dt,
            out_vertex_force: force,
            out_fixed_value,
            dyn_claim: dyn_stage.claim,
            dyn_row: dyn_stage.row,
            dyn_column: dyn_stage.column,
            dyn_block: dyn_stage.block,
            dyn_capacity: dyn_stage.capacity,
            statistics_contact_count: a.statistics.contact_count,
            statistics_contact_count_size: a.statistics.contact_count_size,
            statistics_object_index: a.statistics.object_index,
            statistics_object_index_size: a.statistics.object_index_size,
            statistics_static_object_index: a.statistics.static_object_index,
            statistics_static_object_index_size: a.statistics.static_object_index_size,
            node: self.face_tree.node.handle(),
            node_count: self.face_tree.node_count,
            tree_aabb: self.face_tree.aabb.span(0, self.face_tree.aabb.len()),
            root: self.face_tree.root,
            query: self.collider_point_query.handle(),
            assembled: self.collider_tally.handle(),
            out_overlap: self.overlap_vertex.handle(),
            count: queries as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer the record names is borrowed for the whole call.
        unsafe { device.launch("collision.point_face_c2m.traverse", &args, queries as u32) }?;
        self.matrix.take_device_staged(device, capacity).map_err(|fault| {
            let _ = &fault;
            Fatal::invariant(
                "solver driver: the fixed sparsity has no slot for a block a collision-mesh \
                 contact contributes. That stencil is a vertex's own diagonal, a face's three \
                 vertices or an edge's two, all of which `builder.rs` registers, so the pattern \
                 does not describe this mesh and the coupling would be dropped"
                    .to_string(),
            )
        })?;
        self.collider_tally.download(device)?;
        self.assembled += u64::from(self.collider_tally.host()[0]);
        Ok(self.collider_tally.host()[1] as usize)
    }

    /// A dynamic edge against the collider's edge tree.
    ///
    /// TRAVERSES AND EMBEDS IN ONE PASS, so no pair list is built and none is
    /// read back. A staged form would download the whole candidate buffer per
    /// chunk, then the active flag, the arity and the four indices.
    ///
    /// ZERO DYNAMIC SLOTS, WHICH IS THE COLLIDER PATH'S RULE RATHER THAN A
    /// SIZE. A collision-mesh stencil is a vertex's own diagonal, a face's
    /// three vertices or an edge's two, all of which `builder.rs` registers, so
    /// a block the fixed pattern cannot hold means the pattern does not
    /// describe this mesh. `take_device_staged` turns a non-zero claim into the
    /// fatal below, which is where that check has always lived.
    ///
    /// # Safety
    /// Every buffer the record names must outlive the dispatch.
    unsafe fn traverse_and_embed_collider_edge_edge<D: Device>(
        &mut self,
        device: &mut D,
        inputs: &CollisionInputs,
        fixed: &mut FixedCsr<'_>,
        force: ppf_cts_compute::Handle,
    ) -> FatalResult<usize> {
        let queries = self.edge_query.len();
        if queries == 0 || self.collider_edge_tree.node_count == 0 {
            return Ok(0);
        }
        self.collider_tally
            .size(device, 2, AllocLabel("collision.tally"))?;
        { let h = self.collider_tally.handle(); device.fill_zero(h, 8)?; }
        let a = inputs;
        let (_fi, _fo, out_fixed_value, _rc) = fixed.device_push_refs();
        let capacity = 0usize;
        self.matrix.reserve_device_staging(device, capacity)?;
        let dyn_stage = self.matrix.device_staging(device, capacity)?;
        let args = CollisionEdgeEdgeTraverseArgs {
            x0: a.x0,
            x: a.x,
            edge: a.edge,
            vertex_prop: a.vertex_prop,
            start_link_index: a.start_link.index,
            start_link_offset: a.start_link.offset,
            has_start_link: a.start_link.present,
            edge_prop: a.edge_prop,
            edge_param: a.edge_param,
            static_x: a.static_x,
            static_edge: a.static_edge,
            static_edge_prop: a.static_edge_prop,
            static_edge_param: a.static_edge_param,
            fixed_index: a.fixed_index,
            fixed_offset: a.fixed_offset,
            fixed_value: a.fixed_value,
            row_count: a.row_count,
            friction_mode: a.friction_mode,
            friction_eps: a.friction_eps,
            residual: a.residual,
            dt: a.dt,
            out_vertex_force: force,
            out_fixed_value,
            dyn_claim: dyn_stage.claim,
            dyn_row: dyn_stage.row,
            dyn_column: dyn_stage.column,
            dyn_block: dyn_stage.block,
            dyn_capacity: dyn_stage.capacity,
            statistics_contact_count: a.statistics.contact_count,
            statistics_contact_count_size: a.statistics.contact_count_size,
            statistics_object_index: a.statistics.object_index,
            statistics_object_index_size: a.statistics.object_index_size,
            statistics_static_object_index: a.statistics.static_object_index,
            statistics_static_object_index_size: a.statistics.static_object_index_size,
            node: self.collider_edge_tree.node.handle(),
            node_count: self.collider_edge_tree.node_count,
            tree_aabb: self.collider_edge_tree.aabb.span(0, self.collider_edge_tree.aabb.len()),
            root: self.collider_edge_tree.root,
            query: self.edge_query.handle(),
            assembled: self.collider_tally.handle(),
            out_overlap: self.overlap_edge.handle(),
            count: queries as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer the record names is borrowed for the whole call.
        unsafe { device.launch("collision.edge_edge.traverse", &args, queries as u32) }?;
        self.matrix.take_device_staged(device, capacity).map_err(|fault| {
            let _ = &fault;
            Fatal::invariant(
                "solver driver: the fixed sparsity has no slot for a block a collision-mesh \
                 contact contributes. That stencil is a vertex's own diagonal, a face's three \
                 vertices or an edge's two, all of which `builder.rs` registers, so the pattern \
                 does not describe this mesh and the coupling would be dropped"
                    .to_string(),
            )
        })?;
        self.collider_tally.download(device)?;
        self.assembled += u64::from(self.collider_tally.host()[0]);
        Ok(self.collider_tally.host()[1] as usize)
    }

    /// The inputs half of the collision-mesh passes, which every chunk and all
    /// three passes share. See [`NarrowPhaseInputs`], which it mirrors.
    ///
    /// # Safety
    /// `data`, `param` and both position arrays must outlive the call.
    unsafe fn collision_inputs(
        &self,
        data: &DataSet,
        mesh: MeshRefs,
        param: *const ParamSet,
        x0: ppf_cts_compute::Handle,
        x: ppf_cts_compute::Handle,
        reference: &mut FixedCsr<'_>,
        residual: ppf_cts_compute::Handle,
        statistics: StatisticsRefs,
    ) -> CollisionInputs {
        let pattern = reference.pattern();
        let _statics = &data.constraint.mesh;
        CollisionInputs {
            statistics,
            x0: x0,
            x: x,
            face: mesh.face,
            edge: mesh.edge,
            vertex_prop: mesh.vertex_prop,
            face_prop: mesh.face_prop,
            edge_prop: mesh.edge_prop,
            vertex_param: mesh.vertex_param,
            face_param: mesh.face_param,
            edge_param: mesh.edge_param,
            start_link: mesh.start_link,
            static_x: self.collider_vertex.handle(),
            static_face: self.collider_face.handle(),
            static_edge: self.collider_edge.handle(),
            static_vertex_prop: self.collider_prop.handle(),
            static_face_prop: self.collider_face_prop.handle(),
            static_edge_prop: self.collider_edge_prop.handle(),
            static_vertex_param: self.collider_vertex_param.handle(),
            static_face_param: self.collider_face_param.handle(),
            static_edge_param: self.collider_edge_param.handle(),
            fixed_index: mesh.fixed_index,
            fixed_offset: mesh.fixed_offset,
            fixed_value: reference.value_handle(),
            row_count: pattern.rows,
            friction_mode: (*param).friction_mode as u32,
            friction_eps: (*param).friction_eps,
            residual,
            dt: (*param).dt,
        }
    }
}

/// The collision-window masks, or `None` where the scene authored none.
///
/// `None` is not the same as an all-true mask, and the difference is the whole
/// reason the option is carried: a scene that authored no window has no table
/// at all, so the shim is handed a null pointer and never reads one.
#[derive(Clone, Copy, Default)]
pub struct Windows {
    pub vertex: Option<ppf_cts_compute::Handle>,
    pub edge: Option<ppf_cts_compute::Handle>,
    pub face: Option<ppf_cts_compute::Handle>,
}
/// Take a tree over one primitive's centroids, through the shared Morton
/// pipeline.
///
/// THE CENTROIDS STAY ON THE DEVICE. `scene_bounds` is a host fold and runs
/// against the mirrors BEFORE the handles are taken, because asking for a
/// handle stales them; `primitive_order` is a host sort and runs after the
/// codes come back. Both are driver bookkeeping between two dispatches.
///
/// # Safety
/// As [`bvh::morton_codes`].
/// Rebuild one tree IN PLACE.
///
/// The destination is a parameter rather than a return value because this runs
/// once per step per tree and a `Tree` holds device allocations that nothing
/// reclaims; see `lbvh::build`.
#[allow(clippy::too_many_arguments)]
unsafe fn build_tree<D: Device>(
    device: &mut D,
    cx: &mut ReadbackBuffer<f32>,
    cy: &mut ReadbackBuffer<f32>,
    cz: &mut ReadbackBuffer<f32>,
    codes: &mut ReadbackBuffer<u32>,
    build: &mut super::lbvh::BuildScratch,
    n: usize,
    tree: &mut Tree,
) -> FatalResult<()> {
    // THE BOUNDS ARE REDUCED ON THE DEVICE and stay there. A host reduction
    // would be three O(n) reads of the centroid arrays once per tree per step,
    // and the result is three floats either way.
    let bounds = build.bounds.reduce(device, cx.handle(), cy.handle(), cz.handle(), n as u32)?;
    bvh::morton_codes(
        device,
        cx.span(0, n),
        cy.span(0, n),
        cz.span(0, n),
        codes,
        n,
        bounds,
    )?;
    // THE SORT RUNS ON THE DEVICE. A host sort here would download the codes,
    // sort and gather on the host and upload two arrays back, once per tree per
    // step: the computation relocated rather than a readback.
    let (morton, sorted) = super::lbvh::sort_morton(device, codes, build, n)?;
    lbvh::build_presorted(device, build, morton, sorted, n, tree)
}

/// Which `CcdOverlapRecord::kind` values an ASSEMBLY report carries, as
/// `ccd::decode_assembly_overlap` decodes them: 6 for a self-contact pair, and 7
/// to 9 for the three collision-mesh passes.
const ASSEMBLY_OVERLAP_KINDS: (u32, u32) = (6, 9);

/// The first flagged ASSEMBLY report over the vertex and then the edge report
/// arrays, selected on the device. See [`Contact::collect_assembly_overlap`].
///
/// ONE SUBMIT FOR BOTH LEAF PASSES AND THE WHOLE LADDER, as `bvh.rs` encodes its
/// bounds ladder, then one word read back. A selected word at or past
/// `vertex_slots` is an edge slot, which is what the edge leaf's `base` encodes.
fn first_assembly_overlap<D: Device>(
    device: &mut D,
    fold: &mut super::reduce::DeviceWordFold,
    vertex_reports: &mut ReadbackBuffer<u32>,
    vertex_slots: usize,
    edge_reports: &mut ReadbackBuffer<u32>,
    edge_slots: usize,
) -> FatalResult<Option<ccd::AssemblyOverlap>> {
    if vertex_slots == 0 && edge_slots == 0 {
        return Ok(None);
    }
    if vertex_slots + edge_slots >= u32::MAX as usize {
        return Err(Fatal::invariant(format!(
            "solver driver: {vertex_slots} vertex and {edge_slots} edge overlap slots do not \
             fit below the u32 word that means no slot was flagged"
        )));
    }
    let width = super::reduce::DeviceWordFold::WIDTH;
    let vertex_blocks = (vertex_slots as u32).div_ceil(width);
    let edge_blocks = (edge_slots as u32).div_ceil(width);
    let blocks = vertex_blocks + edge_blocks;
    // SIZED BEFORE ANY SPAN IS TAKEN: a sizing may move the block, and nothing
    // can be allocated inside the region below.
    fold.size_min(device, blocks)?;
    let (kind_first, kind_last) = ASSEMBLY_OVERLAP_KINDS;
    // AN EMPTY ARRAY DISPATCHES NOTHING, because its zero-length allocation has
    // no arena for an entry to resolve.
    let vertex_leaf = (vertex_slots > 0).then(|| OverlapFirstFlaggedLeafArgs {
        overlap: vertex_reports.handle(),
        count: vertex_slots as u32,
        kind_first,
        kind_last,
        base: 0,
        block_size: width,
        out: fold.leaves(0, vertex_blocks),
        blocks: vertex_blocks,
        seam_arena_count: 0,
    });
    let edge_leaf = (edge_slots > 0).then(|| OverlapFirstFlaggedLeafArgs {
        overlap: edge_reports.handle(),
        count: edge_slots as u32,
        kind_first,
        kind_last,
        base: vertex_slots as u32,
        block_size: width,
        out: fold.leaves(vertex_blocks, edge_blocks),
        blocks: edge_blocks,
        seam_arena_count: 0,
    });
    let ladder: &super::reduce::DeviceWordFold = fold;
    device.run("contact.assembly.overlap_select", |encoder| {
        // Safety: both report arrays and the level outlive the region, and each
        // dispatch completes before the next.
        unsafe {
            if let Some(args) = &vertex_leaf {
                encoder.elements(args, vertex_blocks)?;
            }
            if let Some(args) = &edge_leaf {
                encoder.elements(args, edge_blocks)?;
            }
            ladder.encode_min(encoder, blocks)
        }
    })?;
    let selected = fold.read_min(device)?;
    if selected == u32::MAX {
        return Ok(None);
    }
    let selected = selected as usize;
    let (reports, slot, array) = if selected < vertex_slots {
        (vertex_reports, selected, "vertex")
    } else {
        (edge_reports, selected - vertex_slots, "edge")
    };
    let mut words = [0u32; ccd::OVERLAP_WORDS];
    let record_bytes = std::mem::size_of_val(&words);
    // Safety: `u32` is plain data, so the record's bytes are its six words.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), record_bytes) };
    let handle = reports.handle();
    device.read(handle, record_bytes * slot, bytes)?;
    match ccd::decode_assembly_overlap(&words) {
        Some(found) => Ok(Some(found)),
        None => Err(Fatal::invariant(format!(
            "solver driver: the device selected {array} overlap slot {slot}, and that record \
             does not decode as an assembly overlap (flag {}, kind {}). The selection and \
             `ccd::decode_assembly_overlap` disagree about which records count",
            words[0], words[1]
        ))),
    }
}

/// Clear the `active` flag of every leaf outside its collision window.
///
/// A scene that authored no window has nothing to apply and dispatches nothing,
/// which is why this kernel's mask is required rather than optional.
fn set_leaf_active<D: Device>(
    device: &mut D,
    tree: &mut Tree,
    mask: Option<ppf_cts_compute::Handle>,
) -> FatalResult<()> {
    let Some(mask) = mask else {
        return Ok(());
    };
    let leaves = tree.primitive_count() as usize;
    if leaves == 0 {
        return Ok(());
    }
    let args = AabbLeafActiveArgs {
        active: mask,
        nodes: tree.node.span(0, tree.node.len()),
        aabb: tree.aabb.span(0, tree.aabb.len()),
        count: leaves as u32,
        seam_arena_count: 0,
    };
    // Safety: one leaf per thread, each writing its own box, reading a mask the
    // caller sized over the same primitives.
    unsafe { device.launch("contact.leaf_active", &args, leaves as u32) }?;
    // The caller propagates after all leaf changes, including this mask.
    Ok(())
}


/// The inputs half of the narrow phase, which every chunk and all four pair
/// kinds share.
///
/// A DRIVER-LOCAL struct rather than a kernel record. Each of the four
/// generated records names only the arrays its own pair kind reads, so there is
/// no one record left to share; what they still share is this list, which the
/// scene supplies once per step. Building it once and spreading it into four
/// literals keeps the resolution of each buffer in one place.
///
/// THE THREE ADJACENCY FLAGS ARE THE DRIVER'S DECISION, not a null test in a
/// body. A scene either has a vertex-edge table or it does not, which is known
/// here, so it crosses as a flag and the buffer beside it is a real
/// zero-length handle when the flag is zero.
/// The per-object statistics channel's three device arrays, carried together
/// because they are never useful apart.
///
/// THE TWO INDEX MAPS ARE THIS CHANNEL'S OWN AND ARE NOT
/// `VertexProp::object_index`: a different index space, so conflating the two
/// indexes one map with the other's indices. The dynamic
/// and static vertex spaces are separate again within them, which is why the
/// collision-mesh visitors take both and the self-contact visitors take one.
///
/// A scene that configures no statistics objects leaves all three zero-length,
/// `statistics_enabled` reads false on the device, and every recorder call is
/// an early return.
/// Allow Existing Intersections' vertex link table, as the contact, CCD and
/// intersection-scan passes read it through `pair_filter.kernel.cpp`'s
/// `pair_linked_at_start`.
///
/// A CSR over the dynamic vertices, built once by `builder.rs` and staged once
/// by `SolverState::allocate`: row `v` lists the vertices `v` is linked to, a
/// collision-mesh vertex carrying `START_LINK_COLLISION_VERTEX`. `present` is
/// zero for a scene that linked nothing, which is every scene that does not use
/// the option, and the body returns on it before reading either array. The two
/// handles are then REAL zero-length allocations, never `Handle::NONE`, for
/// the arena reason [`StatisticsRefs::none`] states.
#[derive(Clone, Copy)]
pub struct StartLinkRefs {
    pub index: ppf_cts_compute::Handle,
    pub offset: ppf_cts_compute::Handle,
    pub present: u32,
}

#[derive(Clone, Copy)]
pub struct StatisticsRefs {
    pub contact_count: ppf_cts_compute::Handle,
    pub contact_count_size: u32,
    pub object_index: ppf_cts_compute::Handle,
    pub object_index_size: u32,
    pub static_object_index: ppf_cts_compute::Handle,
    pub static_object_index_size: u32,
}

impl StatisticsRefs {
    /// The channel a scene that configures no statistics objects gets, and what
    /// a fixture with no interest in the channel passes.
    ///
    /// REAL ZERO-LENGTH HANDLES, NEVER `Handle::NONE`. A generated entry point
    /// resolves every buffer field against the arena table BEFORE the body
    /// runs, so an optional array must name a bound arena even where the body
    /// will not read it; `Handle::NONE` carries `u32::MAX` and is out of
    /// bounds. That is the arena rule arriving at the seam unchanged, and its
    /// symptom is the same: an assert naming a generated
    /// file nobody wrote, in exactly the scenes that omit the table, which
    /// reads like those scenes rather than like the field.
    ///
    /// It matches what `SolverState::allocate` produces for such a scene rather
    /// than approximating it, so a fixture and a live run take the same path.
    ///
    /// No PRODUCTION caller: a live run gets its channel from
    /// `SolverState::allocate`, whether or not the scene configures statistics
    /// objects. What reads this are three fixtures over scenes that configure
    /// none: `assemble_one` and
    /// `a_vertex_past_the_collider_surface_comes_back_through_the_diagnostic_channel`
    /// in `driver/collider.rs`, and `assemble_against_collider` in this file's
    /// test module.
    #[allow(dead_code)]
    pub fn absent(device: &mut impl Device) -> Result<Self, ppf_cts_compute::Fault> {
        let handle = device.alloc(0, size_of::<u32>(), align_of::<u32>(),
                                  AllocLabel("statistics.absent"))?;
        Ok(Self {
            contact_count: handle,
            contact_count_size: 0,
            object_index: handle,
            object_index_size: 0,
            static_object_index: handle,
            static_object_index_size: 0,
        })
    }
}

struct NarrowPhaseInputs {
    /// The per-object statistics channel; see [`StatisticsRefs`].
    statistics: StatisticsRefs,
    /// The SAND grain inputs and the three per-pair blocks the point-point
    /// visitor writes. Inert for a scene with no grain: `grain_inv_inertia`
    /// is all zeros and every grain branch reads false.
    grain_inv_inertia: ppf_cts_compute::Handle,
    grain_omega: ppf_cts_compute::Handle,
    /// The three per-vertex friction accumulators, owned by `SolverState` for
    /// the same reason `grain_omega` is: they must exist in a scene this layer
    /// is not built for.
    grain_torque: ppf_cts_compute::Handle,
    grain_stiffness: ppf_cts_compute::Handle,
    grain_normal: ppf_cts_compute::Handle,
    dt: f32,
    x0: ppf_cts_compute::Handle,
    x: ppf_cts_compute::Handle,
    face: ppf_cts_compute::Handle,
    edge: ppf_cts_compute::Handle,
    vertex_prop: ppf_cts_compute::Handle,
    face_prop: ppf_cts_compute::Handle,
    edge_prop: ppf_cts_compute::Handle,
    vertex_param: ppf_cts_compute::Handle,
    face_param: ppf_cts_compute::Handle,
    edge_param: ppf_cts_compute::Handle,
    vertex_edge_index: ppf_cts_compute::Handle,
    vertex_edge_offset: ppf_cts_compute::Handle,
    has_vertex_edge: u32,
    vertex_face_index: ppf_cts_compute::Handle,
    vertex_face_offset: ppf_cts_compute::Handle,
    has_vertex_face: u32,
    edge_face_index: ppf_cts_compute::Handle,
    edge_face_offset: ppf_cts_compute::Handle,
    has_edge_face: u32,
    start_link: StartLinkRefs,
    /// The ELASTIC SNAPSHOT the dynamic stiffness reads, which is `tmp_fixed`
    /// and never the matrix being assembled into.
    fixed_index: ppf_cts_compute::Handle,
    fixed_offset: ppf_cts_compute::Handle,
    fixed_value: ppf_cts_compute::Handle,
    row_count: u32,
    /// The three `ParamSet` fields the visitors read, taken here rather than
    /// crossing as a pointer to the whole struct.
    friction_mode: u32,
    barrier_id: u32,
    friction_eps: f32,
    /// THE RESIDUAL EVERY FRICTION TERM ANCHORS ITSELF ON, a COPY of the force
    /// vector taken before this pass rather than the vector itself. Every
    /// contact here deposits into that vector through atomics, so a pair
    /// reading it directly would read a row another pair had already moved.
    residual: ppf_cts_compute::Handle,
}

/// The inputs half of the collision-mesh passes.
///
/// A DRIVER-LOCAL struct, as [`NarrowPhaseInputs`] is and for the same reason.
/// It carries no barrier selector: the collision-mesh passes assemble the
/// one-sided PUSH barrier at all three sites rather than the family the scene
/// selects, so `ParamSet::barrier` is not one of the fields they read.
struct CollisionInputs {
    /// The per-object statistics channel; see [`StatisticsRefs`].
    statistics: StatisticsRefs,
    x0: ppf_cts_compute::Handle,
    x: ppf_cts_compute::Handle,
    face: ppf_cts_compute::Handle,
    edge: ppf_cts_compute::Handle,
    vertex_prop: ppf_cts_compute::Handle,
    face_prop: ppf_cts_compute::Handle,
    edge_prop: ppf_cts_compute::Handle,
    vertex_param: ppf_cts_compute::Handle,
    face_param: ppf_cts_compute::Handle,
    edge_param: ppf_cts_compute::Handle,
    start_link: StartLinkRefs,
    /// The static side, at ONE POSE: it has no start and no end.
    static_x: ppf_cts_compute::Handle,
    static_face: ppf_cts_compute::Handle,
    static_edge: ppf_cts_compute::Handle,
    static_vertex_prop: ppf_cts_compute::Handle,
    static_face_prop: ppf_cts_compute::Handle,
    static_edge_prop: ppf_cts_compute::Handle,
    static_vertex_param: ppf_cts_compute::Handle,
    static_face_param: ppf_cts_compute::Handle,
    static_edge_param: ppf_cts_compute::Handle,
    fixed_index: ppf_cts_compute::Handle,
    fixed_offset: ppf_cts_compute::Handle,
    fixed_value: ppf_cts_compute::Handle,
    row_count: u32,
    friction_mode: u32,
    friction_eps: f32,
    /// THE RESIDUAL EVERY FRICTION TERM ANCHORS ITSELF ON, a COPY of the force
    /// vector taken before this pass rather than the vector itself. Every
    /// contact here deposits into that vector through atomics, so a pair
    /// reading it directly would read a row another pair had already moved.
    residual: ppf_cts_compute::Handle,
    dt: f32,
}

/// # Safety
/// `data`, `param` and both position arrays must outlive the call.
impl Contact {
    unsafe fn narrow_phase_inputs(
        &self,
        _data: &DataSet,
        mesh: MeshRefs,
        param: *const ParamSet,
        x0: ppf_cts_compute::Handle,
        x: ppf_cts_compute::Handle,
        reference: &mut FixedCsr<'_>,
        // THE GRAIN SPIN COMES FROM THE CALLER, `SolverState` being its single
        // owner: the recover and the integrate write it, so a copy held here
        // would be seeded once and never see either.
        grain_omega: ppf_cts_compute::Handle,
        grain_torque: ppf_cts_compute::Handle,
        grain_stiffness: ppf_cts_compute::Handle,
        grain_normal: ppf_cts_compute::Handle,
        residual: ppf_cts_compute::Handle,
        statistics: StatisticsRefs,
    ) -> NarrowPhaseInputs {
        let pattern = reference.pattern();
        NarrowPhaseInputs {
            statistics,
            grain_inv_inertia: self.grain_inv_inertia.handle(),
            grain_omega,
            grain_torque,
            grain_stiffness,
            grain_normal,
            dt: (*param).dt,
            x0: x0,
            x: x,
            face: mesh.face,
            edge: mesh.edge,
            vertex_prop: mesh.vertex_prop,
            face_prop: mesh.face_prop,
            edge_prop: mesh.edge_prop,
            vertex_param: mesh.vertex_param,
            face_param: mesh.face_param,
            edge_param: mesh.edge_param,
            vertex_edge_index: mesh.vertex_edge_index,
            vertex_edge_offset: mesh.vertex_edge_offset,
            has_vertex_edge: mesh.has_vertex_edge,
            vertex_face_index: mesh.vertex_face_index,
            vertex_face_offset: mesh.vertex_face_offset,
            has_vertex_face: mesh.has_vertex_face,
            edge_face_index: mesh.edge_face_index,
            edge_face_offset: mesh.edge_face_offset,
            has_edge_face: mesh.has_edge_face,
            start_link: mesh.start_link,
            fixed_index: mesh.fixed_index,
            fixed_offset: mesh.fixed_offset,
            fixed_value: reference.value_handle(),
            row_count: pattern.rows,
            friction_mode: (*param).friction_mode as u32,
            barrier_id: (*param).barrier as u32,
            friction_eps: (*param).friction_eps,
            residual,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    // The four element records the fixtures build by hand. Production code here
    // names them only through a handle, so they are a test-only import.
    use crate::data::{FaceParam, FaceProp, VertexParam, VertexProp};
    use crate::driver::launch::host_device;

    /// The oracle for [`first_assembly_overlap`]: every vertex report in
    /// ascending slot order, then every edge report, each read through the one
    /// decoder the host reads an assembly report with.
    fn host_first_assembly_overlap(vertex: &[u32], edge: &[u32]) -> Option<ccd::AssemblyOverlap> {
        vertex
            .chunks(ccd::OVERLAP_WORDS)
            .chain(edge.chunks(ccd::OVERLAP_WORDS))
            .find_map(ccd::decode_assembly_overlap)
    }

    /// Report arrays for the given slot counts, each flag written as
    /// `(on the vertex array, slot, kind)` with a record that names its slot.
    fn overlap_reports(
        vertex_slots: usize,
        edge_slots: usize,
        flags: &[(bool, usize, u32)],
    ) -> (Vec<u32>, Vec<u32>) {
        let mut vertex = vec![0u32; ccd::OVERLAP_WORDS * vertex_slots];
        let mut edge = vec![0u32; ccd::OVERLAP_WORDS * edge_slots];
        for &(on_vertex, slot, kind) in flags {
            let words = if on_vertex { &mut vertex } else { &mut edge };
            let record = [
                1u32,
                kind,
                slot as u32,
                slot as u32 + 1,
                (0.25f32 + slot as f32).to_bits(),
                0.5f32.to_bits(),
            ];
            words[ccd::OVERLAP_WORDS * slot..ccd::OVERLAP_WORDS * (slot + 1)]
                .copy_from_slice(&record);
        }
        (vertex, edge)
    }

    fn select_on_device(
        device: &mut impl Device,
        fold: &mut crate::driver::reduce::DeviceWordFold,
        vertex_words: &[u32],
        edge_words: &[u32],
    ) -> Option<ccd::AssemblyOverlap> {
        let mut vertex = readback::<u32>(device, vertex_words.len(), "test.overlap_vertex")
            .expect("the vertex reports size");
        vertex.seed(device, vertex_words).expect("the vertex reports seed");
        let mut edge = readback::<u32>(device, edge_words.len(), "test.overlap_edge")
            .expect("the edge reports size");
        edge.seed(device, edge_words).expect("the edge reports seed");
        first_assembly_overlap(
            device,
            fold,
            &mut vertex,
            vertex_words.len() / ccd::OVERLAP_WORDS,
            &mut edge,
            edge_words.len() / ccd::OVERLAP_WORDS,
        )
        .expect("the selection runs")
    }

    #[test]
    fn the_device_overlap_selection_names_the_pair_the_host_scan_names() {
        let mut device = host_device();
        let mut fold = crate::driver::reduce::DeviceWordFold::default();
        let cases: &[(usize, usize, &[(bool, usize, u32)])] = &[
            // Nothing to reduce, and the healthy case over both arrays.
            (0, 0, &[]),
            (1, 0, &[]),
            (300, 700, &[]),
            // Slot zero, and the last vertex slot in the second leaf block.
            (5, 0, &[(true, 0, 6), (true, 4, 6)]),
            (300, 700, &[(true, 299, 6)]),
            // A vertex report outranks an edge report at a lower slot.
            (300, 700, &[(false, 0, 9), (true, 299, 7)]),
            // Two edge reports straddling a leaf block boundary: the lower wins.
            (300, 700, &[(false, 513, 8), (false, 512, 9)]),
            // A sweep kind, an unknown kind and a kind just below the assembly's
            // are flagged and skipped, as the decoder skips them.
            (300, 700, &[(true, 5, 3), (true, 6, 10), (true, 7, 5), (false, 40, 6)]),
            // Every assembly kind at once: the lowest vertex slot names kind 9.
            (8, 8, &[(true, 3, 9), (true, 4, 6), (false, 0, 7), (false, 1, 8)]),
            // Three rungs over the vertex leaves.
            (65_537, 3, &[(true, 65_536, 6), (false, 2, 7)]),
            // No vertex report array at all.
            (0, 70_000, &[(false, 69_999, 8)]),
        ];
        for &(vertex_slots, edge_slots, flags) in cases {
            let (vertex, edge) = overlap_reports(vertex_slots, edge_slots, flags);
            assert_eq!(
                select_on_device(&mut device, &mut fold, &vertex, &edge),
                host_first_assembly_overlap(&vertex, &edge),
                "vertex_slots={vertex_slots} edge_slots={edge_slots} flags={flags:?}"
            );
        }
    }

    #[test]
    fn the_device_overlap_selection_agrees_with_the_host_scan_on_scattered_flags() {
        let mut device = host_device();
        let mut fold = crate::driver::reduce::DeviceWordFold::default();
        let mut state = 0x9e37_79b9u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for trial in 0..40 {
            let vertex_slots = (next() % 3000) as usize;
            let edge_slots = (next() % 3000) as usize;
            let wanted = next() % 6;
            let mut flags = Vec::new();
            for _ in 0..wanted {
                let on_vertex = next() % 2 == 0;
                let slots = if on_vertex { vertex_slots } else { edge_slots };
                if slots > 0 {
                    flags.push((on_vertex, next() as usize % slots, next() % 12));
                }
            }
            let (vertex, edge) = overlap_reports(vertex_slots, edge_slots, &flags);
            assert_eq!(
                select_on_device(&mut device, &mut fold, &vertex, &edge),
                host_first_assembly_overlap(&vertex, &edge),
                "trial {trial}: vertex_slots={vertex_slots} edge_slots={edge_slots} flags={flags:?}"
            );
        }
    }

    #[test]
    #[ignore = "release timing experiment"]
    fn benchmark_contact_overlap_selection() {
        let mut device = host_device();
        let mut fold = crate::driver::reduce::DeviceWordFold::default();
        for vertex_slots in [16_384usize, 262_144, 1_048_576] {
            // About three edges per vertex, and a healthy step flags nothing,
            // so the host scan runs to the end of both arrays.
            let edge_slots = 3 * vertex_slots;
            let (vertex_words, edge_words) = overlap_reports(vertex_slots, edge_slots, &[]);
            let mut vertex = readback::<u32>(&mut device, vertex_words.len(), "bench.vertex")
                .expect("sizes");
            vertex.seed(&mut device, &vertex_words).expect("seeds");
            let mut edge =
                readback::<u32>(&mut device, edge_words.len(), "bench.edge").expect("sizes");
            edge.seed(&mut device, &edge_words).expect("seeds");
            let mut samples = [Vec::new(), Vec::new()];
            for repetition in 0..10 {
                for mode in [repetition % 2, 1 - repetition % 2] {
                    let start = std::time::Instant::now();
                    let found = if mode == 0 {
                        let _ = vertex.handle();
                        let _ = edge.handle();
                        vertex.download(&mut device).expect("downloads");
                        edge.download(&mut device).expect("downloads");
                        host_first_assembly_overlap(vertex.host(), edge.host())
                    } else {
                        first_assembly_overlap(
                            &mut device,
                            &mut fold,
                            &mut vertex,
                            vertex_slots,
                            &mut edge,
                            edge_slots,
                        )
                        .expect("selects")
                    };
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert!(found.is_none());
                    if repetition > 1 {
                        samples[mode].push(elapsed);
                    }
                }
            }
            for (mode, times) in samples.iter_mut().enumerate() {
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "overlap-select vertex_slots={vertex_slots} edge_slots={edge_slots} path={} \
                     median_ms={:.6} min_ms={:.6} max_ms={:.6}",
                    if mode == 0 { "host" } else { "device" },
                    times[4],
                    times[0],
                    times[7]
                );
            }
        }
    }

    // WHAT REMAINS A DIRECT CALL IN THIS FILE, AND WHY. Nothing in the driver
    // above: every dispatch the contact subsystem makes goes through
    // `Device::launch`. The three below are ORACLES, reached only from the test
    // that follows them, and a test calling a shared body directly is a test
    // rather than a driver reaching around the seam. What that test compares is
    // arithmetic no shared body checks, so the oracle has to be the body itself.
    //
    // Each takes a thread range, so each IS a dispatch by the property that
    // defines one, which is why it may be named here and nowhere above. None
    // has a production caller on any backend: the body they wrap,
    // `extend_contact_force_hessian`, is reached in production from INSIDE
    // the narrow-phase visitors `entrypoints/shim_contact.cpp` and
    // `entrypoints/shim_collider.cpp` dispatch, so these three launchers exist for
    // the arity the test wants to call one at a time.
    extern "C" {
        fn extend_contact2_entry(
            weight: *const f32,
            force: *const f32,
            hessian: *const f32,
            extended_force: *mut f32,
            extended_hessian: *mut f32,
            begin: u32,
            end: u32,
        );
        fn extend_contact3_entry(
            weight: *const f32,
            force: *const f32,
            hessian: *const f32,
            extended_force: *mut f32,
            extended_hessian: *mut f32,
            begin: u32,
            end: u32,
        );
        fn extend_contact4_entry(
            weight: *const f32,
            force: *const f32,
            hessian: *const f32,
            extended_force: *mut f32,
            extended_hessian: *mut f32,
            begin: u32,
            end: u32,
        );
    }

    /// One contact's extended force and Hessian, from the shared body.
    fn extend(arity: usize, weight: &[f32], force: [f32; 3], hessian: [f32; 9]) -> (Vec<f32>, Vec<f32>) {
        let n = arity;
        let mut extended_force = vec![0.0f32; 3 * n];
        let mut extended_hessian = vec![0.0f32; 9 * n * n];
        // Safety: every buffer is sized for one element of this arity, which is
        // what the entry point writes over the range [0, 1).
        unsafe {
            let (w, f, h) = (weight.as_ptr(), force.as_ptr(), hessian.as_ptr());
            let (ef, eh) = (extended_force.as_mut_ptr(), extended_hessian.as_mut_ptr());
            match arity {
                2 => extend_contact2_entry(w, f, h, ef, eh, 0, 1),
                3 => extend_contact3_entry(w, f, h, ef, eh, 0, 1),
                _ => extend_contact4_entry(w, f, h, ef, eh, 0, 1),
            }
        }
        (extended_force, extended_hessian)
    }

    /// THE INDEXING THE SCATTER PERFORMS BY HAND, at every arity.
    ///
    /// `Contact::scatter` reads one 3x3 block out of a `3N x 3N` column-major
    /// matrix staged in a fixed 144-float slot, and the stride it uses is `3 *
    /// arity` rather than 12. That arithmetic is the one piece of the contact
    /// walk that no shared body checks: a wrong stride still produces a
    /// symmetric, plausible matrix, so the only symptom is a trajectory that is
    /// subtly wrong. This compares the extraction against the congruence the
    /// shared body is defined by, `w_i w_j H`.
    #[test]
    fn a_contacts_hessian_blocks_are_read_at_the_stride_its_arity_gives() {
        // Asymmetric on purpose: a symmetric H and equal weights hide a
        // transposed read and a swapped (i, j).
        let hessian = [1.0f32, 2.0, 3.0, 40.0, 50.0, 60.0, 700.0, 800.0, 900.0];
        let force = [7.0f32, -11.0, 13.0];
        for arity in [2usize, 3, 4] {
            let weight: Vec<f32> = (0..arity).map(|i| 1.0 + i as f32).collect();
            let (extended_force, extended_hessian) = extend(arity, &weight, force, hessian);
            let stride = 3 * arity;
            for i in 0..arity {
                for d in 0..3 {
                    let got = extended_force[3 * i + d];
                    let want = weight[i] * force[d];
                    assert_eq!(
                        got, want,
                        "arity {arity}: the force on vertex {i}, component {d}, reads {got} \
                         where the congruence gives {want}"
                    );
                }
            }
            for i in 0..arity {
                for j in 0..arity {
                    for c in 0..3usize {
                        for r in 0..3usize {
                            let got = extended_hessian[stride * (3 * j + c) + 3 * i + r];
                            let want = weight[i] * weight[j] * hessian[3 * c + r];
                            assert_eq!(
                                got, want,
                                "arity {arity}: block ({i}, {j}) element ({r}, {c}) reads \
                                 {got} where the congruence gives {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A 144-float slot holds every arity's matrix in its head, so the same
    /// extraction expression must reach the same numbers when the staging is
    /// the fixed-width buffer the walk actually uses.
    #[test]
    fn the_fixed_width_staging_slot_holds_each_arity_in_its_head() {
        let hessian = [1.0f32, 2.0, 3.0, 40.0, 50.0, 60.0, 700.0, 800.0, 900.0];
        let force = [7.0f32, -11.0, 13.0];
        for arity in [2usize, 3, 4] {
            let weight: Vec<f32> = (0..arity).map(|i| 1.0 + i as f32).collect();
            let (extended_force, extended_hessian) = extend(arity, &weight, force, hessian);
            let mut slot_force = [0.0f32; 12];
            let mut slot_hessian = [0.0f32; 144];
            slot_force[..3 * arity].copy_from_slice(&extended_force);
            slot_hessian[..9 * arity * arity].copy_from_slice(&extended_hessian);
            let stride = 3 * arity;
            for i in 0..arity {
                for j in 0..arity {
                    for c in 0..3usize {
                        for r in 0..3usize {
                            assert_eq!(
                                slot_hessian[stride * (3 * j + c) + 3 * i + r],
                                weight[i] * weight[j] * hessian[3 * c + r],
                                "arity {arity}: the fixed-width slot does not hold block \
                                 ({i}, {j}) where the stride says it does"
                            );
                        }
                    }
                }
            }
            assert_eq!(slot_force[3 * arity..], [0.0f32; 12][3 * arity..]);
        }
    }

    /// A dynamic vertex a millimetre above a static collider triangle, with the
    /// diagonal-only sparsity `builder.rs` registers for every vertex.
    ///
    /// Safety: `DataSet` is a `repr(C)` aggregate of `CVec` handles, plain
    /// integers and nested aggregates of the same, so an all-zero bit pattern is
    /// a valid inhabitant, and a zeroed `CVec` is the null handle `Drop` no-ops
    /// on. This is the fixture pattern `test_scene.rs` established.
    fn scene_over_a_collider(height: f32) -> Box<crate::data::DataSet> {
        use crate::cvec::CVec;
        use crate::cvecvec::CVecVec;
        use crate::driver::test_scene::position;
        use crate::data::{Vec2u, Vec3u};

        let mut data: Box<crate::data::DataSet> = Box::new(unsafe { std::mem::zeroed() });
        let positions = vec![position(0.0, height, 0.0)];
        data.vertex.curr = CVec::from(&positions[..]);
        data.vertex.prev = CVec::from(&positions[..]);
        let mut prop = VertexProp::default();
        prop.mass = 1.0;
        prop.param_index = 0;
        data.prop.vertex = CVec::from(&[prop][..]);
        data.param_arrays.vertex = CVec::from(
            &[VertexParam {
                ghat: 0.01,
                offset: 0.0,
                friction: 0.0,
            }][..],
        );
        data.surface_vert_count = 1;
        let rows: Vec<Vec<u32>> = vec![vec![0]];
        let transpose: Vec<Vec<Vec2u>> = vec![Vec::new()];
        data.fixed_index_table = CVecVec::from(&rows[..]);
        data.transpose_table = CVecVec::from(&transpose[..]);

        // The collider: one triangle in the y = 0 plane, wide enough that the
        // vertex projects into its interior.
        let collider = vec![
            position(-1.0, 0.0, -1.0),
            position(1.0, 0.0, -1.0),
            position(0.0, 0.0, 1.0),
        ];
        data.constraint.mesh.vertex = CVec::from(&collider[..]);
        data.constraint.mesh.face = CVec::from(&[Vec3u::new(0, 1, 2)][..]);
        data.constraint.mesh.prop.vertex = CVec::from(&[VertexProp::default(); 3][..]);
        data.constraint.mesh.prop.face = CVec::from(&[FaceProp::default()][..]);
        data.constraint.mesh.param_arrays.vertex = CVec::from(
            &[VertexParam {
                ghat: 0.01,
                offset: 0.0,
                friction: 0.0,
            }][..],
        );
        let mut face_param = FaceParam::default();
        face_param.ghat = 0.01;
        face_param.offset = 0.0;
        face_param.friction = 0.0;
        data.constraint.mesh.param_arrays.face = CVec::from(&[face_param][..]);
        data
    }

    /// Two free grains on the x axis, `gap` apart, and NOTHING else.
    ///
    /// No faces, no edges and no collider, which is the shape of a faceless
    /// SAND cloud: the point-POINT sweep is the only one that can bound a step
    /// driving these two through each other, every other sweep needing a face
    /// or an edge to query against.
    ///
    /// Safety: as [`scene_over_a_collider`].
    fn scene_of_two_grains(gap: f32) -> Box<crate::data::DataSet> {
        use crate::cvec::CVec;
        use crate::cvecvec::CVecVec;
        use crate::data::Vec2u;
        use crate::driver::test_scene::position;

        let mut data: Box<crate::data::DataSet> = Box::new(unsafe { std::mem::zeroed() });
        let positions = vec![
            position(-0.5 * gap, 0.0, 0.0),
            position(0.5 * gap, 0.0, 0.0),
        ];
        data.vertex.curr = CVec::from(&positions[..]);
        data.vertex.prev = CVec::from(&positions[..]);
        let mut prop = VertexProp::default();
        prop.mass = 1.0;
        prop.param_index = 0;
        data.prop.vertex = CVec::from(&[prop, prop][..]);
        data.param_arrays.vertex = CVec::from(
            &[VertexParam {
                ghat: 0.01,
                offset: 0.0,
                friction: 0.0,
            }][..],
        );
        data.surface_vert_count = 2;
        let rows: Vec<Vec<u32>> = vec![vec![0], vec![1]];
        let transpose: Vec<Vec<Vec2u>> = vec![Vec::new(), Vec::new()];
        data.fixed_index_table = CVecVec::from(&rows[..]);
        data.transpose_table = CVecVec::from(&transpose[..]);
        data
    }

    /// Drive the two grains to `end_gap` and run the CCD line search.
    fn sweep_two_grains(gap: f32, end_gap: f32) -> f32 {
        let data = scene_of_two_grains(gap);
        let param = collision_param();
        // Safety: the boxed scene outlives every borrow below.
        unsafe {
            let mut device = host_device();
            let mut contact = Contact::allocate(&mut device, &data)
                .expect("a scene with vertices allocates");
            let test_mesh = crate::driver::state::test_mesh_of(&mut device, &data);
            let mesh_refs = test_mesh.refs();
            let start_host = crate::driver::state::slice_or_empty(
                data.vertex.curr.data as *const f32,
                3 * data.vertex.curr.size as usize,
            );
            let start_block =
                crate::driver::state::position_block(&mut device, start_host, "test.grain.x0");
            let a = crate::driver::test_scene::position(-0.5 * end_gap, 0.0, 0.0);
            let b = crate::driver::test_scene::position(0.5 * end_gap, 0.0, 0.0);
            let finish_host: [f32; 6] = [a[0], a[1], a[2], b[0], b[1], b[2]];
            let finish_block =
                crate::driver::state::position_block(&mut device, &finish_host, "test.grain.x1");
            let (x0, x1) = (start_block.handle(), finish_block.handle());
            contact
                .rebuild_trees(&mut device, &data, mesh_refs, x0, Windows::default())
                .expect("the trees build");
            contact
                .refresh_leaves(
                    &mut device,
                    &data,
                    mesh_refs,
                    x0,
                    x1,
                    param.line_search_max_t,
                    Windows::default(),
                )
                .expect("the leaves refresh");
            let filter = contact
                .line_search(&mut device, &data, mesh_refs, &param, x0, x1, Windows::default())
                .expect("the line search runs");
            assert!(
                filter.overlapping_start().is_none(),
                "a pair {gap} apart did not begin the step overlapping"
            );
            filter.time_of_impact()
        }
    }

    #[test]
    fn the_line_search_stops_two_grains_driven_through_each_other() {
        // 0.5 apart, commanded to swap places, so the separation closes by 1.0
        // over the step. The advance parks them at `park_floor(0.01) = 1e-4`,
        // so the answer is just under 0.5.
        //
        // THIS IS THE SWEEP A FACELESS CLOUD DEPENDS ON, and it is also the
        // scene shape that has no face tree at all: the point-face and both
        // collision-mesh sweeps must be SKIPPED here rather than dispatched
        // over a tree `rebuild_trees` never built.
        let toi = sweep_two_grains(0.5, -0.5);
        assert!(
            toi > 0.49 && toi < 0.5,
            "the grain sweep returned {toi}, not the ~0.4999 that stops two \
             grains at their contact offsets"
        );
    }

    #[test]
    fn two_grains_moving_apart_keep_the_whole_step() {
        let toi = sweep_two_grains(0.5, 1.5);
        assert_eq!(toi, 1.0, "two grains moving APART had their step cut to {toi}");
    }

    /// Two perpendicular edges, `separation` apart in z, and no faces.
    ///
    /// The only sweep that can bound a step driving these through each other is
    /// the edge-edge one; the four endpoints stay about 1.5 apart throughout, so
    /// the point-point sweep finds nothing and the answer is the edge pass's
    /// alone.
    ///
    /// Safety: as [`scene_over_a_collider`].
    fn scene_of_two_edges(separation: f32) -> Box<crate::data::DataSet> {
        use crate::cvec::CVec;
        use crate::cvecvec::CVecVec;
        use crate::data::{EdgeParam, EdgeProp, Vec2u};
        use crate::driver::test_scene::position;

        let mut data: Box<crate::data::DataSet> = Box::new(unsafe { std::mem::zeroed() });
        let positions = vec![
            position(-1.0, 0.0, 0.0),
            position(1.0, 0.0, 0.0),
            position(0.0, -1.0, separation),
            position(0.0, 1.0, separation),
        ];
        data.vertex.curr = CVec::from(&positions[..]);
        data.vertex.prev = CVec::from(&positions[..]);
        let mut prop = VertexProp::default();
        prop.mass = 1.0;
        prop.param_index = 0;
        data.prop.vertex = CVec::from(&[prop; 4][..]);
        data.param_arrays.vertex = CVec::from(
            &[VertexParam {
                ghat: 0.01,
                offset: 0.0,
                friction: 0.0,
            }][..],
        );
        data.surface_vert_count = 4;
        data.mesh.mesh.edge = CVec::from(&[Vec2u::new(0, 1), Vec2u::new(2, 3)][..]);
        let mut edge_prop = EdgeProp::default();
        edge_prop.mass = 1.0;
        edge_prop.fixed = false;
        edge_prop.param_index = 0;
        data.prop.edge = CVec::from(&[edge_prop; 2][..]);
        let mut edge_param = EdgeParam::default();
        edge_param.ghat = 0.01;
        edge_param.offset = 0.0;
        data.param_arrays.edge = CVec::from(&[edge_param][..]);
        let rows: Vec<Vec<u32>> = vec![vec![0], vec![1], vec![2], vec![3]];
        let transpose: Vec<Vec<Vec2u>> = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        data.fixed_index_table = CVecVec::from(&rows[..]);
        data.transpose_table = CVecVec::from(&transpose[..]);
        data
    }

    /// Drive the second edge from `separation` to `end_separation` in z.
    fn sweep_two_edges(separation: f32, end_separation: f32) -> f32 {
        let data = scene_of_two_edges(separation);
        let param = collision_param();
        // Safety: the boxed scene outlives every borrow below.
        unsafe {
            let mut device = host_device();
            let mut contact = Contact::allocate(&mut device, &data)
                .expect("a scene with vertices allocates");
            let test_mesh = crate::driver::state::test_mesh_of(&mut device, &data);
            let mesh_refs = test_mesh.refs();
            let start_host = crate::driver::state::slice_or_empty(
                data.vertex.curr.data as *const f32,
                3 * data.vertex.curr.size as usize,
            );
            let start_block =
                crate::driver::state::position_block(&mut device, start_host, "test.edge.x0");
            let end = [
                crate::driver::test_scene::position(-1.0, 0.0, 0.0),
                crate::driver::test_scene::position(1.0, 0.0, 0.0),
                crate::driver::test_scene::position(0.0, -1.0, end_separation),
                crate::driver::test_scene::position(0.0, 1.0, end_separation),
            ];
            let finish_host: Vec<f32> = end
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            let finish_block =
                crate::driver::state::position_block(&mut device, &finish_host, "test.edge.x1");
            let (x0, x1) = (start_block.handle(), finish_block.handle());
            contact
                .rebuild_trees(&mut device, &data, mesh_refs, x0, Windows::default())
                .expect("the trees build");
            contact
                .refresh_leaves(
                    &mut device,
                    &data,
                    mesh_refs,
                    x0,
                    x1,
                    param.line_search_max_t,
                    Windows::default(),
                )
                .expect("the leaves refresh");
            let filter = contact
                .line_search(&mut device, &data, mesh_refs, &param, x0, x1, Windows::default())
                .expect("the line search runs");
            assert!(filter.overlapping_start().is_none());
            filter.time_of_impact()
        }
    }

    #[test]
    fn the_line_search_stops_two_edges_swept_across_each_other() {
        // Half a metre apart in z and commanded a full metre down, so an
        // unfiltered step crosses. The advance parks them at
        // `park_floor(0.01) = 1e-4`, so the answer is just under 0.5.
        //
        // IT ALSO EXERCISES THE `T_vf` SEED: the edge sweep opens its register
        // at the point-face minimum rather than at the ceiling, and here that
        // minimum is the whole step, so a seed wired to the wrong value would
        // show as a step cut to something other than this.
        let toi = sweep_two_edges(0.5, -0.5);
        assert!(
            toi > 0.49 && toi < 0.5,
            "the edge-edge sweep returned {toi}, not the ~0.4999 that stops two \
             edges at their contact offsets"
        );
    }

    #[test]
    fn two_edges_moving_apart_keep_the_whole_step() {
        let toi = sweep_two_edges(0.5, 1.5);
        assert_eq!(toi, 1.0, "two edges moving APART had their step cut to {toi}");
    }

    /// One free vertex `height` above a dynamic triangle, and no edges.
    ///
    /// The triangle's own vertices sit at least 1.1 away from the free one, so
    /// the point-POINT sweep finds nothing and the answer is the point-FACE
    /// pass's alone.
    ///
    /// Safety: as [`scene_over_a_collider`].
    fn scene_over_a_dynamic_face(height: f32) -> Box<crate::data::DataSet> {
        use crate::cvec::CVec;
        use crate::cvecvec::CVecVec;
        use crate::data::{Vec2u, Vec3u};
        use crate::driver::test_scene::position;

        let mut data: Box<crate::data::DataSet> = Box::new(unsafe { std::mem::zeroed() });
        let positions = vec![
            position(-1.0, 0.0, -1.0),
            position(1.0, 0.0, -1.0),
            position(0.0, 0.0, 1.0),
            position(0.0, height, 0.0),
        ];
        data.vertex.curr = CVec::from(&positions[..]);
        data.vertex.prev = CVec::from(&positions[..]);
        let mut prop = VertexProp::default();
        prop.mass = 1.0;
        prop.param_index = 0;
        data.prop.vertex = CVec::from(&[prop; 4][..]);
        data.param_arrays.vertex = CVec::from(
            &[VertexParam {
                ghat: 0.01,
                offset: 0.0,
                friction: 0.0,
            }][..],
        );
        data.surface_vert_count = 4;
        data.mesh.mesh.face = CVec::from(&[Vec3u::new(0, 1, 2)][..]);
        let mut face_prop = FaceProp::default();
        face_prop.mass = 1.0;
        face_prop.fixed = false;
        face_prop.param_index = 0;
        data.prop.face = CVec::from(&[face_prop][..]);
        let mut face_param = FaceParam::default();
        face_param.ghat = 0.01;
        face_param.offset = 0.0;
        data.param_arrays.face = CVec::from(&[face_param][..]);
        let rows: Vec<Vec<u32>> = vec![vec![0], vec![1], vec![2], vec![3]];
        let transpose: Vec<Vec<Vec2u>> = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        data.fixed_index_table = CVecVec::from(&rows[..]);
        data.transpose_table = CVecVec::from(&transpose[..]);
        data
    }

    /// Drive the free vertex from `height` to `end_height` over the triangle.
    fn sweep_over_a_dynamic_face(height: f32, end_height: f32) -> f32 {
        sweep_scene_over_a_dynamic_face(scene_over_a_dynamic_face(height), end_height)
    }

    /// The same sweep over a scene the caller has already built, so a test can
    /// install Allow Existing Intersections' links on it first.
    fn sweep_scene_over_a_dynamic_face(
        data: Box<crate::data::DataSet>,
        end_height: f32,
    ) -> f32 {
        let param = collision_param();
        // Safety: the boxed scene outlives every borrow below.
        unsafe {
            let mut device = host_device();
            let mut contact = Contact::allocate(&mut device, &data)
                .expect("a scene with vertices allocates");
            let test_mesh = crate::driver::state::test_mesh_of(&mut device, &data);
            let mesh_refs = test_mesh.refs();
            let start_host = crate::driver::state::slice_or_empty(
                data.vertex.curr.data as *const f32,
                3 * data.vertex.curr.size as usize,
            );
            let start_block =
                crate::driver::state::position_block(&mut device, start_host, "test.face.x0");
            let end = [
                crate::driver::test_scene::position(-1.0, 0.0, -1.0),
                crate::driver::test_scene::position(1.0, 0.0, -1.0),
                crate::driver::test_scene::position(0.0, 0.0, 1.0),
                crate::driver::test_scene::position(0.0, end_height, 0.0),
            ];
            let finish_host: Vec<f32> = end
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            let finish_block =
                crate::driver::state::position_block(&mut device, &finish_host, "test.face.x1");
            let (x0, x1) = (start_block.handle(), finish_block.handle());
            contact
                .rebuild_trees(&mut device, &data, mesh_refs, x0, Windows::default())
                .expect("the trees build");
            contact
                .refresh_leaves(
                    &mut device,
                    &data,
                    mesh_refs,
                    x0,
                    x1,
                    param.line_search_max_t,
                    Windows::default(),
                )
                .expect("the leaves refresh");
            let filter = contact
                .line_search(&mut device, &data, mesh_refs, &param, x0, x1, Windows::default())
                .expect("the line search runs");
            assert!(filter.overlapping_start().is_none());
            filter.time_of_impact()
        }
    }

    #[test]
    fn the_line_search_stops_a_vertex_at_the_dynamic_face_it_is_driven_through() {
        // THE POINT-FACE SWEEP, which is the one the Morton remap is applied
        // at: it walks the FACE tree while remapping the thread through the
        // VERTEX tree, so the record carries two node arrays and a remap taken
        // from the wrong one would query the wrong primitive.
        let toi = sweep_over_a_dynamic_face(0.5, -0.5);
        assert!(
            toi > 0.49 && toi < 0.5,
            "the point-face sweep returned {toi}, not the ~0.4999 that stops the \
             vertex at the triangle it is driven through"
        );
    }

    #[test]
    fn a_vertex_moving_away_from_a_dynamic_face_keeps_the_whole_step() {
        let toi = sweep_over_a_dynamic_face(0.5, 1.5);
        assert_eq!(toi, 1.0, "a vertex moving AWAY had its step cut to {toi}");
    }

    #[test]
    fn a_vertex_linked_at_start_to_the_face_is_not_stopped_by_it() {
        // Allow Existing Intersections: the vertex is linked to ONE vertex of
        // the face, which is enough, so the pair is a neighbor and the sweep
        // does not filter the step against it. The unlinked twin above stops
        // at ~0.5, which is what makes this a result rather than a no-op.
        let mut data = scene_over_a_dynamic_face(0.5);
        data.start_link = crate::cvecvec::CVecVec::from(
            &crate::builder::start_link_table(&[0, 3], 4, 0)[..],
        );
        let toi = sweep_scene_over_a_dynamic_face(data, -0.5);
        assert_eq!(
            toi, 1.0,
            "a vertex linked at start to the face it crosses had its step cut to {toi}"
        );
    }

    #[test]
    fn a_link_elsewhere_does_not_free_the_vertex() {
        // The negative control: a table is present, but it links two vertices
        // of the face to each other and nothing to the swept vertex.
        let mut data = scene_over_a_dynamic_face(0.5);
        data.start_link = crate::cvecvec::CVecVec::from(
            &crate::builder::start_link_table(&[0, 1], 4, 0)[..],
        );
        let toi = sweep_scene_over_a_dynamic_face(data, -0.5);
        assert!(toi > 0.49 && toi < 0.5, "the sweep returned {toi}");
    }

    /// One dynamic edge `height` above a collider edge crossing it, and no faces.
    ///
    /// The collider carries an EDGE and no face, so `build_collider` builds
    /// only its edge tree and the two point-face collision sweeps must be
    /// skipped rather than dispatched over a tree that does not exist.
    ///
    /// Safety: as [`scene_over_a_collider`].
    fn scene_over_a_collider_edge(height: f32) -> Box<crate::data::DataSet> {
        use crate::cvec::CVec;
        use crate::cvecvec::CVecVec;
        use crate::data::{EdgeParam, EdgeProp, Vec2u};
        use crate::driver::test_scene::position;

        let mut data: Box<crate::data::DataSet> = Box::new(unsafe { std::mem::zeroed() });
        let positions = vec![position(-1.0, height, 0.0), position(1.0, height, 0.0)];
        data.vertex.curr = CVec::from(&positions[..]);
        data.vertex.prev = CVec::from(&positions[..]);
        let mut prop = VertexProp::default();
        prop.mass = 1.0;
        prop.param_index = 0;
        data.prop.vertex = CVec::from(&[prop; 2][..]);
        data.param_arrays.vertex = CVec::from(
            &[VertexParam { ghat: 0.01, offset: 0.0, friction: 0.0 }][..],
        );
        data.surface_vert_count = 2;
        data.mesh.mesh.edge = CVec::from(&[Vec2u::new(0, 1)][..]);
        let mut edge_prop = EdgeProp::default();
        edge_prop.mass = 1.0;
        edge_prop.fixed = false;
        edge_prop.param_index = 0;
        data.prop.edge = CVec::from(&[edge_prop][..]);
        let mut edge_param = EdgeParam::default();
        edge_param.ghat = 0.01;
        edge_param.offset = 0.0;
        data.param_arrays.edge = CVec::from(&[edge_param][..]);
        let rows: Vec<Vec<u32>> = vec![vec![0], vec![1]];
        let transpose: Vec<Vec<Vec2u>> = vec![Vec::new(), Vec::new()];
        data.fixed_index_table = CVecVec::from(&rows[..]);
        data.transpose_table = CVecVec::from(&transpose[..]);

        // The collider: one edge along z through the origin, crossing under the
        // dynamic edge.
        let collider = vec![position(0.0, 0.0, -1.0), position(0.0, 0.0, 1.0)];
        data.constraint.mesh.vertex = CVec::from(&collider[..]);
        data.constraint.mesh.prop.vertex = CVec::from(&[VertexProp::default(); 2][..]);
        data.constraint.mesh.param_arrays.vertex = CVec::from(
            &[VertexParam { ghat: 0.01, offset: 0.0, friction: 0.0 }][..],
        );
        data.constraint.mesh.edge = CVec::from(&[Vec2u::new(0, 1)][..]);
        data.constraint.mesh.prop.edge = CVec::from(&[EdgeProp::default()][..]);
        data.constraint.mesh.param_arrays.edge = CVec::from(&[edge_param][..]);
        data
    }

    /// Drive the dynamic edge from `height` to `end_height`.
    fn sweep_over_a_collider_edge(height: f32, end_height: f32) -> f32 {
        let data = scene_over_a_collider_edge(height);
        let param = collision_param();
        // Safety: the boxed scene outlives every borrow below.
        unsafe {
            let mut device = host_device();
            let mut contact = Contact::allocate(&mut device, &data)
                .expect("a scene with vertices allocates");
            let test_mesh = crate::driver::state::test_mesh_of(&mut device, &data);
            let mesh_refs = test_mesh.refs();
            let start_host = crate::driver::state::slice_or_empty(
                data.vertex.curr.data as *const f32,
                3 * data.vertex.curr.size as usize,
            );
            let start_block = crate::driver::state::position_block(
                &mut device, start_host, "test.collider_edge.x0");
            let end = [
                crate::driver::test_scene::position(-1.0, end_height, 0.0),
                crate::driver::test_scene::position(1.0, end_height, 0.0),
            ];
            let finish_host: Vec<f32> = end
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            let finish_block = crate::driver::state::position_block(
                &mut device, &finish_host, "test.collider_edge.x1");
            let (x0, x1) = (start_block.handle(), finish_block.handle());
            contact
                .rebuild_trees(&mut device, &data, mesh_refs, x0, Windows::default())
                .expect("the trees build");
            contact
                .refresh_leaves(
                    &mut device, &data, mesh_refs, x0, x1,
                    param.line_search_max_t, Windows::default(),
                )
                .expect("the leaves refresh");
            let filter = contact
                .line_search(&mut device, &data, mesh_refs, &param, x0, x1, Windows::default())
                .expect("the line search runs");
            assert!(filter.overlapping_start().is_none());
            filter.time_of_impact()
        }
    }

    #[test]
    fn the_line_search_stops_a_dynamic_edge_at_the_collider_edge_it_crosses() {
        // THE COLLIDER EDGE-EDGE SWEEP, which is also seeded from `T_vf` rather
        // than from the ceiling: the point-face pass finds nothing here, so the
        // seed is the whole step and a seed wired to the wrong value would show
        // as a step cut to something other than this.
        let toi = sweep_over_a_collider_edge(0.5, -0.5);
        assert!(
            toi > 0.49 && toi < 0.5,
            "the collider edge sweep returned {toi}, not the ~0.4999 that stops \
             the dynamic edge at the collider edge it crosses"
        );
    }

    #[test]
    fn a_dynamic_edge_moving_away_from_a_collider_edge_keeps_the_whole_step() {
        let toi = sweep_over_a_collider_edge(0.5, 1.5);
        assert_eq!(toi, 1.0, "a dynamic edge moving AWAY had its step cut to {toi}");
    }

    /// A dynamic triangle `height` above a single collider VERTEX.
    ///
    /// The collider carries one vertex and neither a face nor an edge, so the
    /// only collision sweep with anything to do is the collider-to-mesh one,
    /// which dispatches over COLLIDER vertices and walks the DYNAMIC face tree.
    /// It is also the one pass whose query box is UNSWEPT, the collider having
    /// one pose.
    ///
    /// Safety: as [`scene_over_a_collider`].
    fn scene_under_a_collider_vertex(height: f32) -> Box<crate::data::DataSet> {
        use crate::cvec::CVec;
        use crate::cvecvec::CVecVec;
        use crate::data::{Vec2u, Vec3u};
        use crate::driver::test_scene::position;

        let mut data: Box<crate::data::DataSet> = Box::new(unsafe { std::mem::zeroed() });
        let positions = vec![
            position(-1.0, height, -1.0),
            position(1.0, height, -1.0),
            position(0.0, height, 1.0),
        ];
        data.vertex.curr = CVec::from(&positions[..]);
        data.vertex.prev = CVec::from(&positions[..]);
        let mut prop = VertexProp::default();
        prop.mass = 1.0;
        prop.param_index = 0;
        data.prop.vertex = CVec::from(&[prop; 3][..]);
        data.param_arrays.vertex = CVec::from(
            &[VertexParam { ghat: 0.01, offset: 0.0, friction: 0.0 }][..],
        );
        data.surface_vert_count = 3;
        data.mesh.mesh.face = CVec::from(&[Vec3u::new(0, 1, 2)][..]);
        let mut face_prop = FaceProp::default();
        face_prop.mass = 1.0;
        face_prop.fixed = false;
        face_prop.param_index = 0;
        data.prop.face = CVec::from(&[face_prop][..]);
        let mut face_param = FaceParam::default();
        face_param.ghat = 0.01;
        face_param.offset = 0.0;
        data.param_arrays.face = CVec::from(&[face_param][..]);
        let rows: Vec<Vec<u32>> = vec![vec![0], vec![1], vec![2]];
        let transpose: Vec<Vec<Vec2u>> = vec![Vec::new(), Vec::new(), Vec::new()];
        data.fixed_index_table = CVecVec::from(&rows[..]);
        data.transpose_table = CVecVec::from(&transpose[..]);

        data.constraint.mesh.vertex = CVec::from(&[position(0.0, 0.0, 0.0)][..]);
        data.constraint.mesh.prop.vertex = CVec::from(&[VertexProp::default()][..]);
        data.constraint.mesh.param_arrays.vertex = CVec::from(
            &[VertexParam { ghat: 0.01, offset: 0.0, friction: 0.0 }][..],
        );
        data
    }

    /// Drive the dynamic triangle from `height` to `end_height`.
    fn sweep_onto_a_collider_vertex(height: f32, end_height: f32) -> f32 {
        let data = scene_under_a_collider_vertex(height);
        let param = collision_param();
        // Safety: the boxed scene outlives every borrow below.
        unsafe {
            let mut device = host_device();
            let mut contact = Contact::allocate(&mut device, &data)
                .expect("a scene with vertices allocates");
            let test_mesh = crate::driver::state::test_mesh_of(&mut device, &data);
            let mesh_refs = test_mesh.refs();
            let start_host = crate::driver::state::slice_or_empty(
                data.vertex.curr.data as *const f32,
                3 * data.vertex.curr.size as usize,
            );
            let start_block = crate::driver::state::position_block(
                &mut device, start_host, "test.c2m.x0");
            let end = [
                crate::driver::test_scene::position(-1.0, end_height, -1.0),
                crate::driver::test_scene::position(1.0, end_height, -1.0),
                crate::driver::test_scene::position(0.0, end_height, 1.0),
            ];
            let finish_host: Vec<f32> = end
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            let finish_block = crate::driver::state::position_block(
                &mut device, &finish_host, "test.c2m.x1");
            let (x0, x1) = (start_block.handle(), finish_block.handle());
            contact
                .rebuild_trees(&mut device, &data, mesh_refs, x0, Windows::default())
                .expect("the trees build");
            contact
                .refresh_leaves(
                    &mut device, &data, mesh_refs, x0, x1,
                    param.line_search_max_t, Windows::default(),
                )
                .expect("the leaves refresh");
            let filter = contact
                .line_search(&mut device, &data, mesh_refs, &param, x0, x1, Windows::default())
                .expect("the line search runs");
            assert!(filter.overlapping_start().is_none());
            filter.time_of_impact()
        }
    }

    #[test]
    fn the_line_search_stops_a_dynamic_face_at_the_collider_vertex_it_is_driven_onto() {
        // THE COLLIDER-TO-MESH SWEEP, and it is the one whose OUTPUT SLOT is in
        // a foreign index space: it writes at the COLLIDER vertex index into
        // the same array the three dynamic vertex sweeps write, which is why
        // that array is as wide as the wider of the two spaces.
        let toi = sweep_onto_a_collider_vertex(0.5, -0.5);
        assert!(
            toi > 0.49 && toi < 0.5,
            "the collider-to-mesh sweep returned {toi}, not the ~0.4999 that \
             stops the triangle at the collider vertex it is driven onto"
        );
    }

    #[test]
    fn a_dynamic_face_moving_away_from_a_collider_vertex_keeps_the_whole_step() {
        let toi = sweep_onto_a_collider_vertex(0.5, 1.5);
        assert_eq!(toi, 1.0, "a triangle moving AWAY had its step cut to {toi}");
    }

    /// A `ParamSet` with only the fields the collision-mesh pass reads set.
    ///
    /// Safety: as the fixture above.
    fn collision_param() -> ParamSet {
        let mut out: ParamSet = unsafe { std::mem::zeroed() };
        out.line_search_max_t = 1.0;
        out.friction_eps = 1e-5;
        out
    }

    fn assemble_against_collider(height: f32) -> (Vec<f32>, [f32; 9], u64) {
        assemble_scene_against_collider(scene_over_a_collider(height))
    }

    /// The same assembly over a scene the caller has already built.
    fn assemble_scene_against_collider(
        data: Box<crate::data::DataSet>,
    ) -> (Vec<f32>, [f32; 9], u64) {
        let param = collision_param();
        // Safety: the boxed scene outlives every borrow below.
        unsafe {
            // ONE DEVICE FOR BOTH CALLS. The contact state's buffers are
            // handles into the arena that allocated them, and a handle means
            // nothing to a second backend instance: a throwaway device here and
            // another below would dispatch against an arena that was never
            // opened.
            let mut device = host_device();
            // The matrix owns its pattern handles, so the fixture stages the
            // scene's pattern once and hands the same image to both matrices.
            let pattern = super::super::fixedcsr::borrow_pattern(
                &data.fixed_index_table,
                &data.transpose_table,
            )
            .expect("the fixture's pattern is valid");
            let pattern_dev = super::super::state::PatternDevice::of(&mut device, &pattern);
            let mut staging = super::super::state::PushStaging::default();
            let mut reference =
                FixedCsr::from_dataset(&mut device, pattern_dev.refs(), &data).expect("the fixture builds a pattern");
            let mut fixed = FixedCsr::from_dataset(&mut device, pattern_dev.refs(), &data).expect("the fixture builds a pattern");
            let mut contact = Contact::allocate(&mut device, &data)
                .expect("a scene with vertices allocates");
            let mut force = readback::<f32>(&mut device, 3, "test.force")
                .expect("the test allocation succeeds");
            // The positions are device-resident, so a test needs an
            // allocation for the handle to name.
            let positions_host = unsafe {
                crate::driver::state::slice_or_empty(
                    data.vertex.curr.data as *const f32,
                    3 * data.vertex.curr.size as usize,
                )
            };
            let positions_block = crate::driver::state::position_block(
                &mut device,
                positions_host,
                "test.positions",
            );
            let positions = positions_block.handle();
            // The topology as handles, seeded the same way the state seeds it.
            let face_block = crate::driver::state::position_block_u32(
                &mut device,
                unsafe {
                    crate::driver::state::slice_or_empty(
                        data.mesh.mesh.face.data as *const u32,
                        3 * data.mesh.mesh.face.size as usize,
                    )
                },
                "test.face",
            );
            let edge_block = crate::driver::state::position_block_u32(
                &mut device,
                unsafe {
                    crate::driver::state::slice_or_empty(
                        data.mesh.mesh.edge.data as *const u32,
                        2 * data.mesh.mesh.edge.size as usize,
                    )
                },
                "test.edge",
            );
            let prop_block = crate::driver::state::position_block_prop(
                &mut device,
                unsafe {
                    crate::driver::state::slice_or_empty(data.prop.vertex.data, data.prop.vertex.size as usize)
                },
                "test.prop",
            );
            let test_mesh = unsafe { crate::driver::state::test_mesh_of(&mut device, &data) };
            let mesh_refs = test_mesh.refs();
            let statistics =
                crate::driver::contact::StatisticsRefs::absent(&mut device).unwrap();
            let mut residual = readback::<f32>(&mut device, 3, "test.residual")
                .expect("the test allocation succeeds");
            contact.refresh_contact_queries(
                &mut device, &data, mesh_refs, positions, Windows::default(),
            ).expect("the query boxes are prepared");
            contact
                .assemble_collision_mesh_prepared(
                    &mut device,
                    &data,
                    mesh_refs,
                    &param,
                    positions,
                    positions,
                    &mut reference,
                    &mut fixed,
                    force.handle(),
                    &mut staging,
                    // THE RESIDUAL SNAPSHOT. In production it is a copy of the
                    // force vector taken before this pass; this fixture starts
                    // that vector at zero, so a zeroed allocation is the
                    // snapshot it would have made, and a zero residual has no
                    // tangential part, so the friction falls back to the lagged
                    // surrogate. The anchor's own coverage is
                    // `tests/kernels/friction_branches.cpp`.
                    residual.handle(),
                    statistics,
                )
                .expect("the assembly runs");
            {
                force.download(&mut device).expect("the force mirror refreshes");
                // The matrix is device-resident too, and the assembly dispatch
                // above invalidated its mirror.
                fixed.download(&mut device).expect("the matrix reads back");
                (force.host().to_vec(), fixed.read(0, 0), contact.assembled)
            }
        }
    }

    /// Drive the collider fixture's one vertex from `from` to `to` and run the
    /// CCD line search over it.
    ///
    /// THE POINT OF THE FIXTURE IS WHAT IT DOES NOT HAVE. It carries one
    /// dynamic vertex, NO faces and NO edges, so `rebuild_trees` builds only
    /// the vertex tree and the point-face, collider-to-mesh and both edge
    /// sweeps must be skipped rather than dispatched over a tree that was never
    /// allocated. An unallocated buffer is `Handle::NONE`, whose arena is
    /// `u32::MAX`, and a generated entry asserts every handle's arena BEFORE the
    /// body runs, so a missing guard here is a trapped dispatch rather than a
    /// wrong number. This is also the shape of a faceless SAND cloud.
    fn sweep_against_collider(from: f32, to: f32) -> (f32, bool) {
        sweep_against_collider_windowed(from, to, None)
    }

    /// As [`sweep_against_collider`], with an optional per-vertex collision
    /// window.
    ///
    /// NO EXAMPLE SCENE AUTHORS A WINDOW, so the 23-scene sweep exercises only
    /// the unmasked path and the mask is a driver-side flag the six bodies read.
    /// What is untested without this is the `has_active` branch itself: the
    /// `active[element] == 0` test, the flag's pairing with a REAL zero-length
    /// allocation when there is no mask, and the fact that the index it reads is
    /// the query's own.
    fn sweep_against_collider_windowed(
        from: f32,
        to: f32,
        window: Option<&[u32]>,
    ) -> (f32, bool) {
        sweep_scene_against_collider(scene_over_a_collider(from), to, window)
    }

    /// The same sweep over a scene the caller has already built, so a test can
    /// install Allow Existing Intersections' links on it first.
    fn sweep_scene_against_collider(
        data: Box<crate::data::DataSet>,
        to: f32,
        window: Option<&[u32]>,
    ) -> (f32, bool) {
        let param = collision_param();
        // Safety: the boxed scene outlives every borrow below.
        unsafe {
            let mut device = host_device();
            let mut contact = Contact::allocate(&mut device, &data)
                .expect("a scene with vertices allocates");
            let test_mesh = crate::driver::state::test_mesh_of(&mut device, &data);
            let mesh_refs = test_mesh.refs();
            let start_host = crate::driver::state::slice_or_empty(
                data.vertex.curr.data as *const f32,
                3 * data.vertex.curr.size as usize,
            );
            let start_block =
                crate::driver::state::position_block(&mut device, start_host, "test.x0");
            // THE END POSE AS POSITION COMPONENTS, which is what a device block
            // of positions holds.
            let finish = crate::driver::test_scene::position(0.0, to, 0.0);
            let finish_host: [f32; 3] = [finish[0], finish[1], finish[2]];
            let finish_block =
                crate::driver::state::position_block(&mut device, &finish_host, "test.x1");
            let (x0, x1) = (start_block.handle(), finish_block.handle());
            let mut mask = StagedBuffer::<u32>::default();
            let windows = match window {
                Some(bits) => {
                    mask.size(&mut device, bits.len(), AllocLabel("test.window"))
                        .expect("the mask allocation succeeds");
                    mask.at().copy_from_slice(bits);
                    mask.upload(&mut device).expect("the mask uploads");
                    Windows {
                        vertex: Some(mask.handle()),
                        edge: None,
                        face: None,
                    }
                }
                None => Windows::default(),
            };
            contact
                .rebuild_trees(&mut device, &data, mesh_refs, x0, windows)
                .expect("the trees build");
            // THE LEAVES ARE REFRESHED AGAINST THIS SWEEP, exactly as `advance`
            // refreshes them before the line search: a leaf box bounding only
            // the start pose would prune the very pair the sweep is for.
            contact
                .refresh_leaves(
                    &mut device,
                    &data,
                    mesh_refs,
                    x0,
                    x1,
                    param.line_search_max_t,
                    windows,
                )
                .expect("the leaves refresh");
            let filter = contact
                .line_search(&mut device, &data, mesh_refs, &param, x0, x1, windows)
                .expect("the line search runs");
            (
                filter.time_of_impact(),
                filter.overlapping_start().is_some(),
            )
        }
    }

    #[test]
    fn the_line_search_stops_a_vertex_at_the_collider_it_is_driven_through() {
        // The vertex starts half a metre above the triangle and is commanded a
        // full metre down, so an unfiltered step would put it half a metre
        // BELOW. The sweep must return the fraction that stops it short: the
        // gap is 0.5, the travel is 1.0, and `park_floor(ghat) = 1e-2 * 0.01`
        // is the clearance the advance leaves, so the answer is just under 0.5.
        let (toi, overlapping) = sweep_against_collider(0.5, -0.5);
        assert!(!overlapping, "a pair half a metre apart did not begin overlapping");
        assert!(
            toi > 0.49 && toi < 0.5,
            "the collider sweep returned {toi}, not the ~0.4999 that stops the \
             vertex at the triangle it is driven through"
        );
    }

    #[test]
    fn a_windowed_out_vertex_is_not_swept_at_all() {
        // THE SAME GEOMETRY AS THE TEST ABOVE, with the one vertex silenced by
        // a collision window. The sweep must grant the whole step: the query
        // box is inactive, so the traversal prunes at the root and the ACCD
        // advance is never reached. Paired with the unmasked case, which cuts
        // the step to ~0.4999, so a mask read at the wrong index or ignored
        // altogether cannot pass either test.
        let (toi, overlapping) = sweep_against_collider_windowed(0.5, -0.5, Some(&[0]));
        assert!(!overlapping);
        assert_eq!(
            toi, 1.0,
            "a vertex the collision window silences had its step cut to {toi}"
        );
        // AND THE WINDOW ADMITS WHAT IT NAMES, which is the half that fails if
        // the body inverted the test.
        let (toi, _) = sweep_against_collider_windowed(0.5, -0.5, Some(&[1]));
        assert!(
            toi > 0.49 && toi < 0.5,
            "a vertex the window admits was not swept: {toi}"
        );
    }

    #[test]
    fn the_line_search_grants_the_whole_step_to_a_vertex_moving_away() {
        // THE NEGATIVE CASE THE TEST ABOVE NEEDS BESIDE IT: a sweep that
        // returned a filtered time for every pair would pass without it, and so
        // would one whose seed never reached the output array.
        let (toi, overlapping) = sweep_against_collider(0.5, 1.5);
        assert!(!overlapping);
        assert_eq!(
            toi, 1.0,
            "a vertex moving AWAY from the collider had its step cut to {toi}"
        );
    }

    #[test]
    fn a_vertex_over_a_collider_triangle_is_pushed_off_it() {
        // THE VALUE, NOT MERELY THE SIGN. This site assembles the ONE-SIDED
        // PUSH barrier, not the barrier family the self-contact path selects,
        // and the two differ by a factor of two, so the magnitude is what
        // distinguishes them.
        //
        // gap  = 0.001, ghat = 0.01, offset = 0, mass = 1
        // k    = 0 + 1 / 0.001^2         = 1e6
        // d    = 0.001 - 0.01            = -0.009
        // f    = k * -(d^2) / ghat * n   = -8100 * n
        // H    = k * -2 d / ghat * nn^T  = 1.8e6 * nn^T
        let (force, block, count) = assemble_against_collider(0.001);
        assert_eq!(count, 1, "the collider contact was not counted");
        assert!(
            (force[1] + 8100.0).abs() < 1.0,
            "the collider's barrier force is {force:?}, not the -8100 along +y the \
             push barrier gives at a 1 mm gap under a 10 mm contact gap"
        );
        assert!(force[0].abs() < 1e-3 && force[2].abs() < 1e-3, "got {force:?}");
        // Column-major: element (1, 1) is index 3 * 1 + 1.
        assert!(
            (block[4] - 1.8e6).abs() < 1.0e3,
            "the barrier's curvature landed as {} rather than 1.8e6",
            block[4]
        );
    }

    /// The collider fixture with its one dynamic vertex linked at start to the
    /// collider's first vertex, which is combined index 1: the dynamic pool
    /// holds one vertex and the collision mesh follows it.
    fn scene_over_a_collider_linked(height: f32) -> Box<crate::data::DataSet> {
        let mut data = scene_over_a_collider(height);
        data.start_link = crate::cvecvec::CVecVec::from(
            &crate::builder::start_link_table(&[0, 1], 1, 3)[..],
        );
        data
    }

    #[test]
    fn a_vertex_linked_at_start_to_the_collider_is_not_pushed() {
        // The twin of `a_vertex_over_a_collider_triangle_is_pushed_off_it`,
        // which assembles -8100 at this gap: a pair linked at start is a
        // neighbor, so the barrier assembles nothing for it.
        let (force, block, count) =
            assemble_scene_against_collider(scene_over_a_collider_linked(0.001));
        assert_eq!(count, 0, "a linked collider pair was assembled");
        assert!(force.iter().all(|&f| f == 0.0), "got {force:?}");
        assert!(block.iter().all(|&h| h == 0.0), "got {block:?}");
    }

    #[test]
    fn a_vertex_linked_at_start_to_the_collider_is_not_stopped_by_it() {
        // The sweep half of the same pair, which must agree with the assembly
        // above: a pair the barrier does not assemble and the sweep still
        // filtered would stall the vertex at a surface nothing pushes it off.
        let (toi, overlapping) =
            sweep_scene_against_collider(scene_over_a_collider_linked(0.5), -0.5, None);
        assert_eq!(toi, 1.0, "a linked vertex had its step cut to {toi}");
        assert!(!overlapping);
        // And unlinked, the same motion stops at the collider.
        let (toi, _) = sweep_against_collider(0.5, -0.5);
        assert!(toi < 0.5, "the unlinked control returned {toi}");
    }

    #[test]
    fn a_vertex_clear_of_the_collider_contributes_nothing() {
        // The negative case the test above needs beside it: without it, an
        // assembly that pushed on EVERY candidate would pass.
        let (force, block, count) = assemble_against_collider(0.5);
        assert_eq!(count, 0, "a vertex half a metre up was counted as a contact");
        assert!(force.iter().all(|v| *v == 0.0), "got {force:?}");
        assert!(block.iter().all(|v| *v == 0.0), "got {block:?}");
    }
    /// THE MASKED QUERY CLEARS EXACTLY THE PRIMITIVES ITS MASK SILENCES, and
    /// nothing in the notebook sweep reaches this path.
    ///
    /// A collision window is authorable from the frontend and NO example
    /// scene authors one, so the 23-scene sweep exercises only the unmasked
    /// entry. The two share a body, so what is untested without this is the
    /// masked wrapper itself: the `active[element] == 0` test and the record's
    /// field layout. Both are exactly what the deleted shim did, and "exactly
    /// what it did" is a claim worth a test rather than a comment.
    ///
    /// It also pins the PAIRING. The two entries hold two kernel ids and the
    /// launch table is a dense index, so a mis-ordered table would dispatch the
    /// unmasked body for a masked record and silently ignore the window.
    #[test]
    fn the_masked_point_query_clears_only_what_the_mask_silences() {
        use crate::driver::launch::host_device;
        let mut device = host_device();
        let count = 4usize;
        let x: Vec<f32> = (0..count).flat_map(|i| [i as f32, 0.0, 0.0]).collect();
        let prop = vec![VertexProp::default(); count];
        let params = vec![VertexParam::default(); count];
        // Two silenced, two left alone, so a mask read at the wrong index or
        // ignored altogether cannot pass. STAGED, because the record's `active`
        // is a handle now: the production table stages it the same way.
        let mut mask = StagedBuffer::<u32>::default();
        mask.size(&mut device, count, AllocLabel("test.window_mask"))
            .expect("the mask allocation succeeds");
        mask.at().copy_from_slice(&[1, 0, 1, 0]);
        mask.upload(&mut device).expect("the mask uploads");

        // A DEVICE ALLOCATION, because the record's `out` is a handle now.
        let mut masked_out = ReadbackBuffer::<Aabb>::default();
        masked_out
            .size(&mut device, count, AllocLabel("test.masked_query"))
            .expect("the query allocation succeeds");
        let args = AabbPointContactQueryMaskedArgs {
            // Safety: `x` outlives both dispatches below.
            x: crate::driver::state::position_block(&mut device, &x, "test.x").handle(),
            prop: crate::driver::state::record_block(&mut device, &prop, "test.prop").handle(),
            params: crate::driver::state::record_block(&mut device, &params, "test.params").handle(),
            active: mask.handle(),
            out: masked_out.handle(),
            count: count as u32,
            seam_arena_count: 0,
        };
        unsafe { device.launch("test.point_query.masked", &args, count as u32) }
            .expect("the masked query dispatches");

        // A DEVICE ALLOCATION, because the record's `out` is a handle now.
        let mut plain_out = ReadbackBuffer::<Aabb>::default();
        plain_out
            .size(&mut device, count, AllocLabel("test.plain_query"))
            .expect("the query allocation succeeds");
        let args = AabbPointContactQueryArgs {
            // Safety: `x` outlives both dispatches below.
            x: crate::driver::state::position_block(&mut device, &x, "test.x").handle(),
            prop: crate::driver::state::record_block(&mut device, &prop, "test.prop").handle(),
            params: crate::driver::state::record_block(&mut device, &params, "test.params").handle(),
            out: plain_out.handle(),
            count: count as u32,
            seam_arena_count: 0,
        };
        unsafe { device.launch("test.point_query", &args, count as u32) }
            .expect("the unmasked query dispatches");

        masked_out
            .download(&mut device)
            .expect("the masked mirror refreshes");
        plain_out
            .download(&mut device)
            .expect("the plain mirror refreshes");
        let masked = masked_out.host();
        let plain = plain_out.host();
        for i in 0..count {
            assert!(plain[i].active, "the unmasked entry leaves every box in");
            assert_eq!(
                masked[i].active,
                mask.host()[i] != 0,
                "box {i} must follow its own mask entry"
            );
            // The BOX is the same either way: the mask decides participation
            // and nothing else, so a wrapper that perturbed the bounds would
            // fail here rather than in a scene weeks later.
            assert_eq!(masked[i].min, plain[i].min, "box {i} bounds must match");
            assert_eq!(masked[i].max, plain[i].max, "box {i} bounds must match");
        }
    }

}
