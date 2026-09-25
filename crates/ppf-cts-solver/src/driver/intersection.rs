// File: crates/ppf-cts-solver/src/driver/intersection.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! `check_intersection`: the final penetration gate.
//!
//! # What this is for, and why it may not be softened
//!
//! Non-penetration in this solver is enforced STRUCTURALLY, by two things and
//! not by the barrier. The barrier is a cubic energy that is finite at the
//! surface, so it bounds nothing on its own. The two are the ACCD CCD-filtered
//! line search, which refuses a step that would cross, and this scan, which
//! reports a crossing that got through anyway. A contact scene run with the
//! barrier and without both of them COMPLETES, exits 0, and passes surfaces
//! through each other. That outcome is worse than refusing the scene.
//!
//! # The rule this file must not restate
//!
//! Reporting is GATED ON "DYNAMIC" and must stay gated. An intersection between
//! two fully prescribed elements cannot be resolved, because neither side can
//! yield, so reporting it only aborts a run over geometry the solver was never
//! going to fix. `examples/fitting` pins an entire dancing body whose armpits
//! and crotch self-intersect by construction; ungating this aborts it at
//! `initialize`. The same reasoning gives `same_pdrd_body` (a rigid body's
//! self-intersection is fixed and physically meaningless) and the
//! both-collider exclusion (a collider's shape is authored and driven, and
//! rigged colliders ship self-tangled).
//!
//! The intersection ALLOWANCES take the pairs they name out of every pass:
//! contact assembles no barrier for them, the CCD line search does not filter
//! the step against them, and this scan does not report them, so an allowed
//! pair passes through itself freely. They reach exactly one predicate,
//! `isect::intersection_tolerated` in `kernels/contact/intersect_policy.hpp`,
//! which this tree renders into every backend through
//! `contact/pair_filter.kernel.cpp`: `contact_pair_admitted` for the assembly
//! and the sweep, and `intersect_pair_reported`, which is that narrowed by one
//! condition, for the scan. The scan therefore reports only pairs contact acts
//! on, by construction.
//!
//! # Where the work runs
//!
//! ON THE DEVICE, and this module is the LAUNCHER rather than the scan. A
//! dispatch runs one thread per edge, and each thread walks one tree through
//! `aabb::query` with its visitor handed in as a PER-HIT DEVICE FUNCTOR; a
//! further dispatch does the same over the surface vertices with
//! `IntersectPointPointVisitor`. A hit never leaves the device: the functor
//! applies the filters, evaluates the pierce or the proximity, and claims a
//! record slot with an atomic.
//!
//! `contact/intersect_geometry.kernel.cpp` holds all four of those visitors and
//! the four entry points that walk a tree with them, and what crosses to the
//! host is a 4-byte claim counter, at most [`max_records`] records, and two
//! per-element flag arrays folded to a verdict. Nothing else. In particular
//! there is NO candidate pair list: materializing one, downloading it and
//! walking it serially would move a device pass onto the host, and it would
//! force a buffer to carry the list.
//!
//! FOUR DISPATCHES AND NOT TWO. Folding the three edge walks into one kernel
//! would need an absent tree to be PASSED and then skipped inside the body,
//! which `aabb::query`'s `if (bvh.node.size)` would do for free; a handle here
//! carries an arena index that a generated entry resolves BEFORE the body runs,
//! so an absent tree cannot be passed at all and the DISPATCH is what must be
//! skipped. That is the same gate the six CCD sweeps
//! take, for the same reason. The flag array is shared and folded by OR, so
//! splitting the walks changes nothing about the verdict.
//!
//! # The split
//!
//! | shared C++ | here, in Rust |
//! |---|---|
//! | the traversal, the pair filters, the pierce, the two proximities, the allowance rule and the record claim | which tree each walk visits, when a walk is skipped, and how the three readbacks become a report |
//!
//! Not one predicate is stated here.

use ppf_cts_compute::{AllocLabel, Buffer, Device, Handle, ReadbackBuffer};
use super::lbvh::Tree;
use crate::data::IntersectionRecord;
use super::kernels::{
    IntersectScanCollisionMeshArgs, IntersectScanEdgeEdgeArgs, IntersectScanFaceEdgeArgs,
    IntersectScanPointPointArgs, VecFillU32Args,
};
use super::scene::{Fatal, FatalResult};

/// A position triple, the layout of one `Vec3f`.
///
/// Rust never interprets these three components, and with the scan on the
/// device it never reads them either. The alias survives because two other
/// modules spell their element type with it.
pub type PositionTriple = [f32; 3];

// The three CONSTANTS the scan needs from the shared header, and nothing else.
//
// Each is a `#define` or an `enum` in a C++ header the Rust side must agree
// with exactly, and each is READ rather than restated: `MAX_INTERSECTION_RECORDS`
// sizes the record buffer on both sides, `sizeof(IntersectionRecord)` is what
// makes [`RECORD_WORDS`] a decode and not a guess, and `NO_OBJECT_INDEX` is the
// sentinel the collision-mesh walk hands the static side.
//
// They are the same category as `super::pcg`'s per-block inverse and
// `super::step`'s domain query: one value, no thread index, no extent.
extern "C" {
    /// No PRODUCTION caller: the sentinel reaches the walks inside the shared
    /// body, which hands it to the static side of a collision-mesh pair. What
    /// reads it here is `the_record_mirror_matches_cpp` and
    /// `the_allowance_rule_is_the_shared_one`, each asserting it equals the
    /// Rust `NO_OBJECT_INDEX`, which is what keeps an unknown object from
    /// reading as a real one on one side of the seam.
    #[allow(dead_code)]
    fn no_object_index_abi() -> u32;
    fn max_intersection_records_abi() -> u32;
    /// No PRODUCTION caller: a record is decoded through [`RECORD_WORDS`]. What
    /// reads it here is `the_record_mirror_matches_cpp`, which compares it
    /// against `size_of::<IntersectionRecord>()` and against `RECORD_WORDS * 4`,
    /// which is what makes that stride a decode rather than a guess.
    #[allow(dead_code)]
    fn intersection_record_sizeof_abi() -> u32;
    /// No PRODUCTION caller: the scan reaches this same body on the device
    /// through `intersect_pair_reported`. What reads it here is [`tolerated`]
    /// just below, whose own reader is `the_allowance_rule_is_the_shared_one`.
    #[allow(dead_code)]
    fn intersection_tolerated_abi(
        a_object_index: u32,
        a_group_index: u32,
        a_intersect_policy: u8,
        b_object_index: u32,
        b_group_index: u32,
        b_intersect_policy: u8,
        a_pin_allows: i32,
        b_pin_allows: i32,
    ) -> i32;
}

/// The record types, as `IntersectionRecord::type` encodes them.
///
/// Mirrors the `INTERSECT_RECORD_*` enumeration in
/// `kernels/contact/intersect_record.kernel.cpp`, which is where the claim
/// writes them.
///
/// The production reader is [`describe_record`], which names a pair in the
/// initialize failure. `a_crossed_pair_is_reported` asserts that an edge running
/// through a triangle's interior is claimed as this type and not as one of the
/// other three.
pub const RECORD_FACE_EDGE: u32 = 0;
/// `edge_edge_reports_each_unordered_pair_once` asserts that two segments
/// crossing closer than their combined offsets are claimed as this type.
pub const RECORD_EDGE_EDGE: u32 = 1;
/// `a_dynamic_edge_through_the_collision_mesh_is_reported` asserts that a
/// dynamic edge through the rest-pose static mesh is claimed as this type
/// rather than as a dynamic face-edge pair.
pub const RECORD_COLLISION_MESH: u32 = 2;
/// `overlapping_grains_are_reported_once_each` asserts that two grains inside
/// their combined offsets are claimed as this type, the pass a faceless SAND
/// cloud depends on.
pub const RECORD_POINT_POINT: u32 = 3;

/// One record's pair, named by element kind and index, as a message can print
/// it. The indices are the solver's own element arrays, which is what the
/// intersection records the frontend draws are keyed on too.
pub fn describe_record(record: &IntersectionRecord) -> String {
    let (a, b) = (record.elem0, record.elem1);
    match record.itype {
        RECORD_FACE_EDGE => format!("face {a} and edge {b}"),
        RECORD_EDGE_EDGE => format!("edge {a} and edge {b}"),
        RECORD_COLLISION_MESH => format!("collision-mesh face {a} and edge {b}"),
        RECORD_POINT_POINT => format!("vertex {a} and vertex {b}"),
        other => format!("record kind {other}, elements {a} and {b}"),
    }
}

/// How many 32-bit words one [`IntersectionRecord`] occupies.
///
/// THE RECORD ARRAY IS A `u32` BUFFER AND IS DECODED HERE, the same shape the
/// CCD line search's overlap report takes. `IntersectionRecord` is five
/// unsigned counts followed by fifteen floats at four-byte alignment, so a word
/// array carries it exactly and the driver needs no `Pod` mirror of a struct
/// whose only reader is a diagnostic. `the_record_mirror_matches_cpp` compares
/// this against `sizeof` on the C++ side, so a field added on either side is a
/// failing test rather than a misread report.
pub const RECORD_WORDS: usize = 20;

/// How many records the device buffer holds. Read from the shared header rather
/// than restated, because the two must agree for a readback to be safe.
pub fn max_records() -> usize {
    (unsafe { max_intersection_records_abi() }) as usize
}

/// The allowance rule, through the shared body.
///
/// Exposed so a test can drive the rule directly against the same definition
/// the four device visitors reach through `intersect_pair_reported`. The scan
/// itself never calls this: it would be a second statement of a rule that must
/// have one.
///
/// What reads it is `the_allowance_rule_is_the_shared_one`, the only Rust-side
/// gate on the allowance rule, which is guarantee-class: it drives the shared
/// body over the positive and the negative case of every
/// allowance, and the negative cases are the ones that catch over-suppression.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn tolerated(
    a_object_index: u32,
    a_group_index: u32,
    a_intersect_policy: u8,
    b_object_index: u32,
    b_group_index: u32,
    b_intersect_policy: u8,
    a_pin_allows: bool,
    b_pin_allows: bool,
) -> bool {
    unsafe {
        intersection_tolerated_abi(
            a_object_index,
            a_group_index,
            a_intersect_policy,
            b_object_index,
            b_group_index,
            b_intersect_policy,
            i32::from(a_pin_allows),
            i32::from(b_pin_allows),
        ) != 0
    }
}

/// What one scan found.
#[derive(Clone, Default)]
pub struct Report {
    /// The first `max_records()` intersections, in the order the device claimed
    /// them.
    ///
    /// ASCENDING BY QUERY ELEMENT, and that is a property of the declaration
    /// rather than of the arithmetic: the four rows are `Scatter::Claim`, so a
    /// backend runs them as one serial ascending pass and a slot assignment is
    /// reproducible. A backend with real atomics claims in whatever order its
    /// threads arrive, and the SET is the same either way.
    pub records: Vec<IntersectionRecord>,
    /// How many were FOUND, which exceeds `records.len()` once the buffer is
    /// full. Counted past capacity deliberately, by the claim itself: a run
    /// aborted on intersections wants the true count, and a report that stops
    /// counting where it stops storing understates the problem.
    pub found: u32,
    /// Whether NO edge intersects anything, folded on the device over the flag
    /// `check_intersection` accumulates across its three edge walks.
    ///
    /// The verdict is `edge_clear && vert_clear` over two device folds, with no
    /// flag array copied to the host. The per-element flags are still there and
    /// `Scan::flags` downloads them, which only a test asks for.
    ///
    /// No reader, in production or in a test: the field is the sink for the
    /// two-level device fold in `ScanState::fold_flags`, so it carries a result
    /// the scan computes rather than a spare slot.
    #[allow(dead_code)]
    pub edge_clear: bool,
    /// The same for the grain-grain pass over surface vertices, and the sink
    /// for the same fold: no reader in production or in a test.
    #[allow(dead_code)]
    pub vert_clear: bool,
    /// The per-element flags, IN A TEST BUILD ONLY.
    ///
    /// Production reads the verdict off the device fold above and never copies
    /// a flag array to the host. A test asks WHICH element intersected, and
    /// paying two array downloads for that in a test build costs nothing anyone
    /// measures.
    #[cfg(test)]
    pub edge_flag: Vec<bool>,
    #[cfg(test)]
    pub vert_flag: Vec<bool>,
}

impl Report {
    pub fn is_clean(&self) -> bool {
        self.found == 0
    }
}

/// Every scene array the four walks read, by device handle.
///
/// One struct rather than fifteen arguments, carrying exactly what each of the
/// four visitors captures. Every field is an allocation the caller already owns
/// for the whole run; nothing here is staged for the scan.
#[derive(Clone, Copy)]
pub struct Scene {
    /// The committed pose, `3 * vertices` position components.
    pub vert: Handle,
    pub face: Handle,
    pub edge: Handle,
    pub face_prop: Handle,
    pub edge_prop: Handle,
    pub vert_prop: Handle,
    pub edge_param: Handle,
    pub vertex_param: Handle,
    /// Allow Existing Intersections' link table. A linked pair is not
    /// reported, by the same predicate that keeps it out of contact.
    pub start_link: super::contact::StartLinkRefs,
    /// THE THREE BOUNDS THE VISITORS CHECK A LEAF INDEX AGAINST. A leaf
    /// primitive is DATA rather than the thread index, so the entry's own
    /// thread-count guard says nothing about it, and Metal returns 0.0 for an
    /// out-of-bounds read and faults on nothing.
    pub faces: usize,
    pub edges: usize,
    pub surface_vertices: usize,
}

/// The rest-pose collision mesh, which is a disjoint pool with its own
/// positions.
#[derive(Clone, Copy)]
pub struct Collider {
    pub vert: Handle,
    pub face: Handle,
    pub faces: usize,
}

/// The scan's own device buffers, persistent across steps.
///
/// ALLOCATED ONCE AND REUSED. A per-step allocation would leak an arena span,
/// there being no `Drop` to give it back, and a one-frame `--fast-check` would
/// not show it. `size` grows only past capacity, so a steady state costs
/// nothing.
///
/// The two flag arrays and the counter are CLEARED BY A DISPATCH rather than by
/// a host seed, which is what the CCD line search beside this already does for
/// its overlap records. A host seed would be an upload per step of something no
/// host value contributes to.
#[derive(Default)]
pub struct ScanState {
    edge_flag: ReadbackBuffer<u32>,
    vert_flag: ReadbackBuffer<u32>,
    records: ReadbackBuffer<u32>,
    counter: ReadbackBuffer<u32>,
    /// Where the two flag arrays are FOLDED, on the device.
    ///
    /// Each flag array is reduced on the device and only the one-word total
    /// comes back; no flag array is copied to the host. Folding rather than
    /// downloading is what keeps the verdict a device computation: the arrays
    /// are one entry per edge and per surface vertex, so a host walk is a
    /// whole-array transfer per check on the hottest guarantee path there is.
    flag_fold: Buffer<u32>,
    flag_total: ReadbackBuffer<u32>,
    edges: usize,
    surface_vertices: usize,
}

/// The verdict a walk that could not finish leaves.
///
/// A traversal fault reaches the host through the entry's `[[seam::diag]]`
/// lane, so a `launch` that returns an error IS a scan that reached no verdict.
/// The message says so, because the caller's next act is to commit a pose.
fn scan_failed(what: &str, fault: impl std::fmt::Display) -> Fatal {
    Fatal::device_assert(format!(
        "solver driver: the {what} intersection walk did not complete. {fault} \
         An intersecting pair this scan did not visit is a penetration it \
         cannot report, so the run may not proceed"
    ))
}

/// Zero one of the arrays a walk folds into.
///
/// `vec_fill_u32` is the same body the CCD line search clears its overlap
/// records with, and it clears the array ON THE DEVICE. A host seed would upload
/// a page of zeros per step for a value no host reader contributes to.
fn clear<D: Device>(device: &mut D, array: Handle, count: usize) -> FatalResult<()> {
    let args = VecFillU32Args {
        array,
        value: 0,
        count: count as u32,
        seam_arena_count: 0,
    };
    // Safety: the array is one of `ScanState`'s own allocations, sized just
    // above, and the record names nothing else.
    unsafe { device.launch("intersection.clear", &args, count as u32) }?;
    Ok(())
}

impl ScanState {
    /// Size the buffers for this scene and clear what the walks accumulate
    /// into.
    ///
    /// EVERY WALK FOLDS RATHER THAN STORES: a flag is written only on a hit, so
    /// the three edge walks OR into ONE array and the counter is claimed from.
    /// Both therefore have to open at zero, and Metal hands out an allocation
    /// without zeroing it and never faults on an uninitialized read.
    ///
    /// THE RECORD ARRAY TAKES NO CLEAR OF ITS OWN, deliberately: only the slots
    /// below the counter are ever read, and every one of those was written by
    /// the claim that raised it.
    ///
    /// THE THREE CLEARS ARE NOT REDUNDANT WITH `size`. `Buffer::size` zeroes
    /// only the bytes it has just allocated, so a buffer whose length has not
    /// changed comes back carrying the previous step's values, and a per-step
    /// reset that read its zero out of another crate's allocator would not
    /// happen at all. Metal hands out an allocation without zeroing and never
    /// faults on an uninitialized read, so the failure these three prevent is a
    /// scan that quietly starts flagged.
    pub fn begin<D: Device>(
        &mut self,
        device: &mut D,
        edges: usize,
        surface_vertices: usize,
    ) -> FatalResult<()> {
        self.edges = edges;
        self.surface_vertices = surface_vertices;
        // A ZERO-LENGTH ARRAY STILL TAKES A REAL ALLOCATION. A generated entry
        // resolves every handle it is given before the body runs, and
        // `Handle::NONE` carries an out-of-range arena, so a scene with no edge
        // still hands the point-point walk a one-element flag array it will not
        // touch.
        self.edge_flag
            .size(device, edges.max(1), AllocLabel("intersection.edge_flag"))?;
        self.vert_flag.size(
            device,
            surface_vertices.max(1),
            AllocLabel("intersection.vert_flag"),
        )?;
        self.records.size(
            device,
            max_records() * RECORD_WORDS,
            AllocLabel("intersection.records"),
        )?;
        self.counter
            .size(device, 1, AllocLabel("intersection.counter"))?;
        // THE FOLD SCRATCH, sized for the longer of the two arrays. A fold over
        // N elements needs one slot per block at every level, and the geometric
        // series over a width of `DOF_FOLD_WIDTH` is under N/(width-1) slots,
        // so one block's worth beyond the first level is ample; the loop asserts
        // it by construction, never writing past `blocks` at a level.
        let longest = edges.max(surface_vertices).max(1);
        let width = crate::driver::state::DOF_FOLD_WIDTH;
        let slots = longest.div_ceil(width).max(1) * 2;
        self.flag_fold
            .size(device, slots, AllocLabel("intersection.flag_fold"))?;
        self.flag_total
            .size(device, 1, AllocLabel("intersection.flag_total"))?;
        let edge_words = self.edge_flag.len();
        let vert_words = self.vert_flag.len();
        clear(device, self.edge_flag.handle(), edge_words)?;
        clear(device, self.vert_flag.handle(), vert_words)?;
        clear(device, self.counter.handle(), 1)?;
        Ok(())
    }

    /// The DYNAMIC edge against the DYNAMIC face tree.
    ///
    /// `IntersectFaceEdgeVisitor` (`contact/intersect_geometry.kernel.cpp`),
    /// reached as a per-hit functor by `intersect_scan_face_edge`. `edge_query`
    /// is one
    /// box per edge, built on the device by `aabb_edge_scan_query`, so the
    /// collision window has already been applied to `active`.
    ///
    /// # Safety
    /// the scene's handles and `edge_query` must name live allocations on `device`, and `face_tree` must be built and propagated against this same pose.
    pub unsafe fn scan_face_edge<D: Device>(
        &mut self,
        device: &mut D,
        scene: &Scene,
        face_tree: &mut Tree,
        edge_query: Handle,
    ) -> FatalResult<()> {
        if face_tree.is_empty() || scene.edges == 0 || scene.faces == 0 {
            return Ok(());
        }
        let count = scene.edges as u32;
        let args = IntersectScanFaceEdgeArgs {
            vert: scene.vert,
            face: scene.face,
            face_count: scene.faces as u32,
            edge: scene.edge,
            vertex_prop: scene.vert_prop,
            start_link_index: scene.start_link.index,
            start_link_offset: scene.start_link.offset,
            has_start_link: scene.start_link.present,
            face_prop: scene.face_prop,
            edge_prop: scene.edge_prop,
            node: face_tree.node.handle(),
            node_count: face_tree.node_count,
            aabb: face_tree.aabb.handle(),
            root: face_tree.root,
            query: edge_query,
            flag: self.edge_flag.handle(),
            records: self.records.handle(),
            counter: self.counter.handle(),
            capacity: max_records() as u32,
            count,
            seam_arena_count: 0,
        };
        device
            .launch("intersection.face_edge", &args, count)
            .map_err(|fault| scan_failed("face-edge", fault))?;
        Ok(())
    }

    /// Edge against edge, closer than the two contact offsets combined.
    ///
    /// `IntersectEdgeEdgeVisitor` (`contact/intersect_geometry.kernel.cpp`),
    /// including its upper-triangular halving: the found edge must sort BELOW
    /// the query edge, so each unordered pair is measured once and from the
    /// same end.
    ///
    /// # Safety
    /// the scene's handles and `edge_query` must name live allocations on `device`, and `edge_tree` must be built and propagated against this same pose.
    pub unsafe fn scan_edge_edge<D: Device>(
        &mut self,
        device: &mut D,
        scene: &Scene,
        edge_tree: &mut Tree,
        edge_query: Handle,
    ) -> FatalResult<()> {
        if edge_tree.is_empty() || scene.edges == 0 {
            return Ok(());
        }
        let count = scene.edges as u32;
        let args = IntersectScanEdgeEdgeArgs {
            vert: scene.vert,
            edge: scene.edge,
            edge_count: count,
            vertex_prop: scene.vert_prop,
            start_link_index: scene.start_link.index,
            start_link_offset: scene.start_link.offset,
            has_start_link: scene.start_link.present,
            edge_prop: scene.edge_prop,
            edge_param: scene.edge_param,
            node: edge_tree.node.handle(),
            node_count: edge_tree.node_count,
            aabb: edge_tree.aabb.handle(),
            root: edge_tree.root,
            query: edge_query,
            flag: self.edge_flag.handle(),
            records: self.records.handle(),
            counter: self.counter.handle(),
            capacity: max_records() as u32,
            count,
            seam_arena_count: 0,
        };
        device
            .launch("intersection.edge_edge", &args, count)
            .map_err(|fault| scan_failed("edge-edge", fault))?;
        Ok(())
    }

    /// Grain against grain.
    ///
    /// `IntersectPointPointVisitor` (`contact/intersect_geometry.kernel.cpp`).
    /// This pass is what covers a faceless SAND cloud: the edge walks never run
    /// for
    /// one, so without it an overlapping cloud passes the initial check and
    /// aborts mid-advance instead.
    ///
    /// # Safety
    /// the scene's handles and `vertex_query` must name live allocations on `device`, and `vertex_tree` must be built and propagated against this same pose.
    pub unsafe fn scan_point_point<D: Device>(
        &mut self,
        device: &mut D,
        scene: &Scene,
        vertex_tree: &mut Tree,
        vertex_query: Handle,
    ) -> FatalResult<()> {
        if vertex_tree.is_empty() || scene.surface_vertices == 0 {
            return Ok(());
        }
        let count = scene.surface_vertices as u32;
        let args = IntersectScanPointPointArgs {
            vert: scene.vert,
            vertex_count: count,
            vertex_prop: scene.vert_prop,
            start_link_index: scene.start_link.index,
            start_link_offset: scene.start_link.offset,
            has_start_link: scene.start_link.present,
            vertex_param: scene.vertex_param,
            node: vertex_tree.node.handle(),
            node_count: vertex_tree.node_count,
            aabb: vertex_tree.aabb.handle(),
            root: vertex_tree.root,
            query: vertex_query,
            flag: self.vert_flag.handle(),
            records: self.records.handle(),
            counter: self.counter.handle(),
            capacity: max_records() as u32,
            count,
            seam_arena_count: 0,
        };
        device
            .launch("intersection.point_point", &args, count)
            .map_err(|fault| scan_failed("point-point", fault))?;
        Ok(())
    }

    /// A dynamic edge against the rest-pose STATIC collision mesh.
    ///
    /// `IntersectCollisionMeshVisitor`
    /// (`contact/intersect_geometry.kernel.cpp`). The static side is a disjoint
    /// contact-only pool outside the solved namespace, so the pair is
    /// inter-object by construction and has no self-intersection case: the
    /// static side is handed `NO_OBJECT_INDEX` and an empty policy, under which
    /// "either side opts in" reduces to "the dynamic edge opted in". The body
    /// settles that once per edge, OUTSIDE the traversal, so the per-hit functor
    /// carries a flag rather than re-deciding it.
    ///
    /// # Safety
    /// the scene's and the collider's handles and `edge_query` must name live allocations on `device`, and `collider_face_tree` must be built over that collider.
    pub unsafe fn scan_collision_mesh<D: Device>(
        &mut self,
        device: &mut D,
        scene: &Scene,
        collider: &Collider,
        collider_face_tree: &mut Tree,
        edge_query: Handle,
    ) -> FatalResult<()> {
        if collider_face_tree.is_empty() || scene.edges == 0 || collider.faces == 0 {
            return Ok(());
        }
        let count = scene.edges as u32;
        let args = IntersectScanCollisionMeshArgs {
            vert: scene.vert,
            edge: scene.edge,
            vertex_prop: scene.vert_prop,
            start_link_index: scene.start_link.index,
            start_link_offset: scene.start_link.offset,
            has_start_link: scene.start_link.present,
            edge_prop: scene.edge_prop,
            collider_vertex: collider.vert,
            collider_face: collider.face,
            collider_face_count: collider.faces as u32,
            node: collider_face_tree.node.handle(),
            node_count: collider_face_tree.node_count,
            aabb: collider_face_tree.aabb.handle(),
            root: collider_face_tree.root,
            query: edge_query,
            flag: self.edge_flag.handle(),
            records: self.records.handle(),
            counter: self.counter.handle(),
            capacity: max_records() as u32,
            count,
            seam_arena_count: 0,
        };
        device
            .launch("intersection.collision_mesh", &args, count)
            .map_err(|fault| scan_failed("collision-mesh", fault))?;
        Ok(())
    }

    /// The three readbacks, and nothing else.
    ///
    /// A counter, at most `capacity` records, and the two flag arrays. The
    /// record copy is CLAMPED to `capacity`: the counter counts DEMAND, so it
    /// may exceed the array, and reading it as a length would read past the end.
    /// Fold one flag array to a single count, on the device.
    ///
    /// The question is whether ANY flag is set, and a SUM over an array of
    /// zeroes and ones answers it, using the fold this driver already has for
    /// `u32`. What matters is that the array is never walked on the host.
    fn fold_flags<D: Device>(
        &mut self,
        device: &mut D,
        which: &'static str,
        count: usize,
    ) -> FatalResult<u32> {
        if count == 0 {
            return Ok(0);
        }
        let width = crate::driver::state::DOF_FOLD_WIDTH;
        let source = if which == "edge" {
            self.edge_flag.handle()
        } else {
            self.vert_flag.handle()
        };
        let mut input = source;
        let mut remaining = count;
        let mut cursor = 0usize;
        loop {
            let blocks = remaining.div_ceil(width);
            let destination = if blocks == 1 {
                self.flag_total.handle()
            } else {
                self.flag_fold.span(cursor, blocks)
            };
            let args = crate::driver::kernels::VecBlockSumU32Args {
                source: input,
                length: remaining as u32,
                width: width as u32,
                total: destination,
                count: blocks as u32,
                seam_arena_count: 0,
            };
            // Safety: every handle names a live allocation for this call.
            unsafe { device.launch("intersection.flag_fold", &args, blocks as u32) }?;
            if blocks == 1 {
                break;
            }
            input = self.flag_fold.span(cursor, blocks);
            cursor += blocks;
            remaining = blocks;
        }
        self.flag_total.download(device)?;
        Ok(self.flag_total.host()[0])
    }

    /// The per-element flags, downloaded on demand.
    ///
    /// PRODUCTION DOES NOT CALL THIS. The verdict is a device fold; these
    /// arrays say WHICH element intersected, which only a test asks for.
    ///
    /// What reads it is the `#[cfg(test)]` arm of [`ScanState::finish`], which
    /// fills `Report::edge_flag` and `Report::vert_flag`; the tests asserting
    /// on those are `a_crossed_pair_is_reported` (edge 0 is the piercing one),
    /// `overlapping_grains_are_reported_once_each` (grain 1 is the query that
    /// found the overlap) and
    /// `a_dynamic_edge_through_the_collision_mesh_is_reported`.
    #[allow(dead_code)]
    pub fn flags<D: Device>(
        &mut self,
        device: &mut D,
    ) -> FatalResult<(Vec<bool>, Vec<bool>)> {
        let mut edge = vec![false; self.edges];
        if self.edges > 0 {
            self.edge_flag.download(device)?;
            let raw = self.edge_flag.host();
            for (slot, out) in edge.iter_mut().enumerate() {
                *out = raw[slot] != 0;
            }
        }
        let mut vert = vec![false; self.surface_vertices];
        if self.surface_vertices > 0 {
            self.vert_flag.download(device)?;
            let raw = self.vert_flag.host();
            for (slot, out) in vert.iter_mut().enumerate() {
                *out = raw[slot] != 0;
            }
        }
        Ok((edge, vert))
    }

    pub fn finish<D: Device>(&mut self, device: &mut D) -> FatalResult<Report> {
        self.counter.download(device)?;
        let found = self.counter.host()[0];
        let stored = (found as usize).min(max_records());
        let mut records = Vec::with_capacity(stored);
        if stored > 0 {
            self.records.download(device)?;
            let words = self.records.host();
            for slot in 0..stored {
                records.push(decode_record(&words[slot * RECORD_WORDS..][..RECORD_WORDS]));
            }
        }
        // THE VERDICT IS A DEVICE FOLD. Downloading the two arrays to walk
        // them here would be a whole-array transfer per check, twice, on the
        // path that decides whether a pose may be committed.
        let edges = self.edges;
        let surface = self.surface_vertices;
        let edge_clear = self.fold_flags(device, "edge", edges)? == 0;
        let vert_clear = self.fold_flags(device, "vert", surface)? == 0;
        #[cfg(test)]
        let (edge_flag, vert_flag) = self.flags(device)?;
        Ok(Report {
            records,
            found,
            edge_clear,
            vert_clear,
            #[cfg(test)]
            edge_flag,
            #[cfg(test)]
            vert_flag,
        })
    }
}

/// One record, out of the words the device wrote.
///
/// The five counts are read as they lie and the fifteen positions by their bit
/// pattern, which is what makes this a decode rather than a conversion: the
/// device wrote `float`s and no arithmetic happens on either side of the copy.
/// Those positions are ABSOLUTE world coordinates rather than differences,
/// because the record is a diagnostic handed to a user who has to find the
/// geometry in their scene and a relative offset would not answer that.
fn decode_record(words: &[u32]) -> IntersectionRecord {
    let mut record = IntersectionRecord {
        itype: words[0],
        elem0: words[1],
        elem1: words[2],
        num_verts0: words[3],
        num_verts1: words[4],
        positions: [0.0; 15],
    };
    for (slot, out) in record.positions.iter_mut().enumerate() {
        *out = f32::from_bits(words[5 + slot]);
    }
    record
}

#[cfg(test)]
mod tests {
    use super::super::launch::HostDevice;
    use super::super::lbvh::{self, Aabb};
    use super::*;
    use crate::data::{
        EdgeParam, EdgeProp, FaceProp, VertexParam, VertexProp, INTERSECT_ALLOW_INTER_GROUP,
        INTERSECT_ALLOW_INTER_OBJECT, INTERSECT_ALLOW_SELF, NO_GROUP_INDEX, NO_OBJECT_INDEX,
    };
    use ppf_cts_compute::{Buffer, Pod};

    /// A dynamic element: free, massive, not a collider, in no body.
    fn free_face(param_index: u32) -> FaceProp {
        FaceProp {
            mass: 1.0,
            param_index,
            ..Default::default()
        }
    }
    fn free_edge(param_index: u32) -> EdgeProp {
        EdgeProp {
            mass: 1.0,
            param_index,
            ..Default::default()
        }
    }
    fn free_vertex(param_index: u32) -> VertexProp {
        VertexProp {
            mass: 1.0,
            param_index,
            object_index: 0,
            ..Default::default()
        }
    }

    /// A position, in the world units a component carries directly. The three
    /// values pass through, so a test writes its geometry at a human scale and
    /// the scan reads the metres the test wrote.
    fn point(x: f32, y: f32, z: f32) -> PositionTriple {
        [x, y, z]
    }

    /// ONE DEVICE, ONE STAGING SET, ONE SCAN STATE PER FIXTURE.
    ///
    /// The scan reads every scene array by DEVICE HANDLE, because the four
    /// walks run on the device, so a test supplies them the way production
    /// does: as allocations on the fixture's own device. A handle names no
    /// allocator, so a tree or a buffer built on a throwaway device names an
    /// arena the scan cannot reach.
    struct Fixture {
        device: HostDevice,
        scan: ScanState,
        query_boxes: Buffer<Aabb>,
        vert: Buffer<f32>,
        face: Buffer<u32>,
        edge: Buffer<u32>,
        face_prop: Buffer<FaceProp>,
        edge_prop: Buffer<EdgeProp>,
        vert_prop: Buffer<VertexProp>,
        edge_param: Buffer<EdgeParam>,
        vertex_param: Buffer<VertexParam>,
        collider_vert: Buffer<f32>,
        collider_face: Buffer<u32>,
        link_index: Buffer<u32>,
        link_offset: Buffer<u32>,
        link_present: u32,
        faces: usize,
        edges: usize,
        surface_vertices: usize,
        collider_faces: usize,
    }

    /// One array on the device, with a real allocation even when it is empty.
    ///
    /// A generated entry resolves every handle it is handed BEFORE the body
    /// runs, so an absent array takes a zero-length allocation and never
    /// `Handle::NONE`, whose arena is out of range. That is the arena rule,
    /// that every handle names a real allocation, reaching the seam unchanged.
    fn stage<T: Pod>(device: &mut HostDevice, buffer: &mut Buffer<T>, data: &[T]) -> Handle {
        buffer
            .size(
                device,
                data.len().max(1),
                ppf_cts_compute::AllocLabel("test.scene"),
            )
            .expect("the fixture sizes its scene array");
        if !data.is_empty() {
            buffer
                .write(device, 0, data)
                .expect("the fixture stages its scene array");
        }
        buffer.handle()
    }

    fn flat_points(points: &[PositionTriple]) -> Vec<f32> {
        points.iter().flat_map(|p| p.iter().copied()).collect()
    }

    fn flat_u32<const N: usize>(rows: &[[u32; N]]) -> Vec<u32> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    impl Fixture {
        fn new() -> Self {
            Fixture {
                device: super::super::launch::host_device(),
                scan: ScanState::default(),
                query_boxes: Buffer::none(),
                vert: Buffer::none(),
                face: Buffer::none(),
                edge: Buffer::none(),
                face_prop: Buffer::none(),
                edge_prop: Buffer::none(),
                vert_prop: Buffer::none(),
                edge_param: Buffer::none(),
                vertex_param: Buffer::none(),
                collider_vert: Buffer::none(),
                collider_face: Buffer::none(),
                link_index: Buffer::none(),
                link_offset: Buffer::none(),
                link_present: 0,
                faces: 0,
                edges: 0,
                surface_vertices: 0,
                collider_faces: 0,
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn stage_scene(
            &mut self,
            vertex: &[PositionTriple],
            face: &[[u32; 3]],
            edge: &[[u32; 2]],
            face_prop: &[FaceProp],
            edge_prop: &[EdgeProp],
            vert_prop: &[VertexProp],
            edge_param: &[EdgeParam],
            vertex_param: &[VertexParam],
        ) {
            stage(&mut self.device, &mut self.vert, &flat_points(vertex));
            stage(&mut self.device, &mut self.face, &flat_u32(face));
            stage(&mut self.device, &mut self.edge, &flat_u32(edge));
            stage(&mut self.device, &mut self.face_prop, face_prop);
            stage(&mut self.device, &mut self.edge_prop, edge_prop);
            stage(&mut self.device, &mut self.vert_prop, vert_prop);
            stage(&mut self.device, &mut self.edge_param, edge_param);
            stage(&mut self.device, &mut self.vertex_param, vertex_param);
            self.faces = face.len();
            self.edges = edge.len();
            self.surface_vertices = vertex.len();
        }

        /// Allow Existing Intersections' table, one row per dynamic vertex,
        /// exactly as `builder::start_link_table` lays it out.
        fn stage_links(&mut self, rows: &[Vec<u32>]) {
            let mut offset = vec![0u32];
            let mut index = Vec::new();
            for row in rows {
                index.extend_from_slice(row);
                offset.push(index.len() as u32);
            }
            stage(&mut self.device, &mut self.link_index, &index);
            stage(&mut self.device, &mut self.link_offset, &offset);
            self.link_present = 1;
        }

        fn stage_collider(&mut self, vertex: &[PositionTriple], face: &[[u32; 3]]) {
            stage(&mut self.device, &mut self.collider_vert, &flat_points(vertex));
            stage(&mut self.device, &mut self.collider_face, &flat_u32(face));
            self.collider_faces = face.len();
        }

        fn scene(&mut self) -> Scene {
            if self.link_present == 0 {
                // No table: real zero-length handles beside a zero flag.
                stage(&mut self.device, &mut self.link_index, &[]);
                stage(&mut self.device, &mut self.link_offset, &[]);
            }
            Scene {
                vert: self.vert.handle(),
                face: self.face.handle(),
                edge: self.edge.handle(),
                face_prop: self.face_prop.handle(),
                edge_prop: self.edge_prop.handle(),
                vert_prop: self.vert_prop.handle(),
                edge_param: self.edge_param.handle(),
                vertex_param: self.vertex_param.handle(),
                start_link: super::super::contact::StartLinkRefs {
                    index: self.link_index.handle(),
                    offset: self.link_offset.handle(),
                    present: self.link_present,
                },
                faces: self.faces,
                edges: self.edges,
                surface_vertices: self.surface_vertices,
            }
        }

        fn collider(&mut self) -> Collider {
            Collider {
                vert: self.collider_vert.handle(),
                face: self.collider_face.handle(),
                faces: self.collider_faces,
            }
        }

        /// Leaf boxes written straight into a tree, then propagated.
        ///
        /// The production path fills them from geometry through the shared
        /// body; a scan test needs boxes over known geometry, which is the same
        /// thing computed here so the fixture stays readable.
        fn tree(&mut self, boxes: &[Aabb]) -> lbvh::Tree {
            // Each bound is halved before the sum, so a box whose two sides
            // are far apart cannot overflow on the way to its midpoint.
            let mid = |b: &Aabb, k: usize| 0.5 * b.min[k] + 0.5 * b.max[k];
            let cx: Vec<f32> = boxes.iter().map(|b| mid(b, 0)).collect();
            let cy: Vec<f32> = boxes.iter().map(|b| mid(b, 1)).collect();
            let cz: Vec<f32> = boxes.iter().map(|b| mid(b, 2)).collect();
            let mut tree = unsafe {
                lbvh::build_from_centroids(
                    &mut self.device,
                    &mut lbvh::BuildScratch::default(),
                    &cx,
                    &cy,
                    &cz,
                )
            }
            .expect("the tree builds");
            // THE LEAF SEED, read back and written: both arrays are
            // device-resident and production fills the boxes from a `refresh_*`
            // kernel rather than assembling them here.
            let host = lbvh::tree_host(&mut self.device, &mut tree);
            let mut aabb = host.aabb.clone();
            for node in 0..tree.node_count as usize {
                if host.node[2 * node + 1] == 0 {
                    aabb[node] = boxes[(host.node[2 * node] - 1) as usize];
                }
            }
            tree.aabb
                .write(&mut self.device, 0, &aabb)
                .expect("the fixture seeds the leaf boxes");
            unsafe { lbvh::propagate(&mut self.device, &mut tree) }.expect("the boxes merge");
            tree
        }

        /// One box list on the device, for a scan that takes a handle.
        fn query(&mut self, boxes: &[Aabb]) -> Handle {
            self.query_boxes
                .size(
                    &mut self.device,
                    boxes.len().max(1),
                    ppf_cts_compute::AllocLabel("test.query"),
                )
                .and_then(|()| self.query_boxes.write(&mut self.device, 0, boxes))
                .expect("the fixture stages its query boxes");
            self.query_boxes.span(0, boxes.len().max(1))
        }

        fn begin(&mut self) {
            let (edges, verts) = (self.edges, self.surface_vertices);
            self.scan
                .begin(&mut self.device, edges, verts)
                .expect("the scan sizes and clears its buffers");
        }

        fn finish(&mut self) -> Report {
            self.scan
                .finish(&mut self.device)
                .expect("the scan reads its three results back")
        }

        /// The per-element flags, which production never downloads.
        fn flags(&mut self) -> (Vec<bool>, Vec<bool>) {
            self.scan
                .flags(&mut self.device)
                .expect("the scan reads its flag arrays back")
        }
    }

    /// A box grown by `margin` world units on every side.
    ///
    /// The production leaf boxes carry `0.5 * ghat + offset`, applied inside the
    /// shared body; a fixture that needs the same asymmetry between a leaf and a
    /// query says so here rather than pretending the two are the same box.
    fn inflated(mut b: Aabb, margin: f32) -> Aabb {
        for k in 0..3 {
            b.min[k] -= margin;
            b.max[k] += margin;
        }
        b
    }

    fn box_of(points: &[PositionTriple]) -> Aabb {
        let mut b = Aabb {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            active: true,
        };
        for p in points {
            for (k, coordinate) in p.iter().enumerate() {
                b.min[k] = b.min[k].min(*coordinate);
                b.max[k] = b.max[k].max(*coordinate);
            }
        }
        b
    }

    fn face_boxes_of(vertex: &[PositionTriple], face: &[[u32; 3]]) -> Vec<Aabb> {
        face.iter()
            .map(|f| {
                box_of(&[
                    vertex[f[0] as usize],
                    vertex[f[1] as usize],
                    vertex[f[2] as usize],
                ])
            })
            .collect()
    }

    fn edge_boxes_of(vertex: &[PositionTriple], edge: &[[u32; 2]]) -> Vec<Aabb> {
        edge.iter()
            .map(|e| box_of(&[vertex[e[0] as usize], vertex[e[1] as usize]]))
            .collect()
    }

    /// Two triangles CROSSED, so an edge of one genuinely pierces the other.
    ///
    /// Two sheets sliding PARALLEL through each other pierce no edge-triangle
    /// at any pose and would make this fixture prove nothing; the crossing is
    /// what gives it a real answer.
    fn crossed_triangles() -> (Vec<PositionTriple>, Vec<[u32; 3]>, Vec<[u32; 2]>) {
        let vertex = vec![
            // Triangle 0, in the z = 0 plane.
            point(-1.0, -0.5, 0.0),
            point(1.0, -0.5, 0.0),
            point(0.0, 1.0, 0.0),
            // Triangle 1, standing upright and passing through it.
            point(0.0, 0.0, -1.0),
            point(0.0, 0.0, 1.0),
            point(0.5, 0.6, 0.0),
        ];
        let face = vec![[0u32, 1, 2], [3, 4, 5]];
        // Edge 0 of triangle 1 runs from below the z = 0 plane to above it,
        // through the interior of triangle 0.
        let edge = vec![[3u32, 4], [0, 1], [1, 2], [2, 0], [4, 5], [5, 3]];
        (vertex, face, edge)
    }

    /// One face-edge scan over the crossed fixture, with the props a test
    /// varies passed in. Every filter test differs only in those, so the walk
    /// itself is written once.
    fn scan_crossed(
        face_prop: &[FaceProp],
        edge_prop: &[EdgeProp],
        vert_prop: &[VertexProp],
    ) -> Report {
        let mut fx = Fixture::new();
        let (vertex, face, edge) = crossed_triangles();
        fx.stage_scene(
            &vertex,
            &face,
            &edge,
            face_prop,
            edge_prop,
            vert_prop,
            &[EdgeParam::default()],
            &[VertexParam::default()],
        );
        let scene = fx.scene();
        let mut face_tree = fx.tree(&face_boxes_of(&vertex, &face));
        let query = fx.query(&edge_boxes_of(&vertex, &edge));
        fx.begin();
        // Safety: every handle in `scene` names an allocation this
        // fixture staged on this device and holds for the call.
        unsafe { fx.scan.scan_face_edge(&mut fx.device, &scene, &mut face_tree, query) }.unwrap();
        fx.finish()
    }

    #[test]
    fn the_record_mirror_matches_cpp() {
        assert_eq!(
            std::mem::size_of::<IntersectionRecord>(),
            unsafe { intersection_record_sizeof_abi() } as usize,
            "the Rust IntersectionRecord is a different size from the C++ one, \
             so a filled record would be read out of the wrong bytes"
        );
        assert_eq!(
            RECORD_WORDS * 4,
            unsafe { intersection_record_sizeof_abi() } as usize,
            "the record buffer is a u32 array and RECORD_WORDS is its stride, \
             so a record that grew a field would be decoded out of the wrong \
             words with nothing to say so"
        );
        assert_eq!(max_records(), 256);
        assert_eq!(unsafe { no_object_index_abi() }, NO_OBJECT_INDEX);
    }

    #[test]
    fn a_crossed_pair_is_reported() {
        // The positive case, and the one a parallel-sheet fixture cannot make:
        // an edge that genuinely passes through a triangle.
        let report = scan_crossed(
            &[free_face(0), free_face(0)],
            &[free_edge(0); 6],
            &[free_vertex(0); 6],
        );
        assert!(
            report.found > 0,
            "an edge running from z = -1 to z = +1 through the interior of a \
             triangle in the z = 0 plane was not reported; this is the pierce \
             the final penetration gate exists to catch"
        );
        assert!(report.edge_flag[0], "edge 0 is the piercing one");
        assert_eq!(report.records[0].itype, RECORD_FACE_EDGE);
        assert_eq!(report.records[0].elem0, 0, "face 0 is the pierced one");
        assert_eq!(report.records[0].elem1, 0);
        assert_eq!(report.records[0].num_verts0, 3);
        assert_eq!(report.records[0].num_verts1, 2);
    }

    /// The crossed fixture's face-edge scan with a link table installed.
    fn scan_crossed_linked(rows: &[Vec<u32>]) -> Report {
        let mut fx = Fixture::new();
        let (vertex, face, edge) = crossed_triangles();
        fx.stage_scene(
            &vertex,
            &face,
            &edge,
            &[free_face(0), free_face(0)],
            &[free_edge(0); 6],
            &[free_vertex(0); 6],
            &[EdgeParam::default()],
            &[VertexParam::default()],
        );
        fx.stage_links(rows);
        let scene = fx.scene();
        let mut face_tree = fx.tree(&face_boxes_of(&vertex, &face));
        let query = fx.query(&edge_boxes_of(&vertex, &edge));
        fx.begin();
        // Safety: every handle in `scene` names an allocation this
        // fixture staged on this device and holds for the call.
        unsafe { fx.scan.scan_face_edge(&mut fx.device, &scene, &mut face_tree, query) }.unwrap();
        fx.finish()
    }

    #[test]
    fn a_crossed_pair_linked_at_start_is_not_reported() {
        // Allow Existing Intersections: the two triangles started crossed and
        // the build linked every vertex of one to every vertex of the other,
        // which is exactly the table `builder::start_link_table` makes of it.
        let t0 = vec![3u32, 4, 5];
        let t1 = vec![0u32, 1, 2];
        let rows = vec![t0.clone(), t0.clone(), t0, t1.clone(), t1.clone(), t1];
        let report = scan_crossed_linked(&rows);
        assert!(
            report.is_clean(),
            "a pair linked at start is a neighbor, like two elements sharing a \
             vertex, and reporting it aborts the run the user asked for"
        );
    }

    #[test]
    fn a_link_table_that_does_not_name_the_pair_changes_nothing() {
        // The negative control: a table is present and a vertex of each
        // triangle has a row, but no row reaches the other triangle. Without
        // this the test above could pass on a scan that ignored every pair once
        // any table was installed.
        let rows = vec![vec![1u32], vec![0], vec![], vec![4], vec![3], vec![]];
        let report = scan_crossed_linked(&rows);
        assert!(report.found > 0, "an unlinked crossing was not reported");
        assert!(report.edge_flag[0]);
    }

    #[test]
    fn a_separated_pair_is_not_reported() {
        // The negative control: the same geometry, moved apart. Without it the
        // test above could pass on a scan that reports everything.
        let mut fx = Fixture::new();
        let (mut vertex, face, edge) = crossed_triangles();
        for v in vertex.iter_mut().skip(3) {
            v[2] += 10.0;
        }
        fx.stage_scene(
            &vertex,
            &face,
            &edge,
            &[free_face(0); 2],
            &[free_edge(0); 6],
            &[free_vertex(0); 6],
            &[EdgeParam::default()],
            &[VertexParam::default()],
        );
        let scene = fx.scene();
        let mut face_tree = fx.tree(&face_boxes_of(&vertex, &face));
        let query = fx.query(&edge_boxes_of(&vertex, &edge));
        fx.begin();
        // Safety: every handle in `scene` names an allocation this
        // fixture staged on this device and holds for the call.
        unsafe { fx.scan.scan_face_edge(&mut fx.device, &scene, &mut face_tree, query) }.unwrap();
        let report = fx.finish();
        assert!(report.is_clean(), "separated geometry was reported");
    }

    #[test]
    fn a_face_and_an_edge_sharing_a_vertex_are_never_reported() {
        // Every mesh has these at every seam, so reporting one aborts every
        // scene.
        //
        // ONE face and its own three sides, deliberately. The crossed fixture's
        // second triangle is genuinely pierced by a side of the first, which is
        // what the fixture is FOR, so leaving it in would make a failure here
        // ambiguous between the shared-vertex rule and a real pierce.
        let mut fx = Fixture::new();
        let (vertex, _, _) = crossed_triangles();
        let face = vec![[0u32, 1, 2]];
        let edge = vec![[0u32, 1], [1, 2], [2, 0]];
        fx.stage_scene(
            &vertex,
            &face,
            &edge,
            &[free_face(0)],
            &[free_edge(0); 3],
            &[free_vertex(0); 6],
            &[EdgeParam::default()],
            &[VertexParam::default()],
        );
        let scene = fx.scene();
        let face_boxes = face_boxes_of(&vertex, &face);
        let edge_boxes = edge_boxes_of(&vertex, &edge);
        let mut face_tree = fx.tree(&face_boxes);
        let query = fx.query(&edge_boxes);
        fx.begin();
        // Safety: every handle in `scene` names an allocation this
        // fixture staged on this device and holds for the call.
        unsafe { fx.scan.scan_face_edge(&mut fx.device, &scene, &mut face_tree, query) }.unwrap();
        let report = fx.finish();
        assert!(
            report.is_clean(),
            "a face was reported as pierced by one of its own sides"
        );
        // And the broad phase really did offer those pairs, so the silence
        // above is the shared-vertex rule rather than an empty candidate set.
        let query = fx.query(&edge_boxes);
        let mut scratch = lbvh::WalkScratch::default();
        let candidates = lbvh::query_pairs(
            &mut fx.device,
            &mut face_tree,
            query,
            edge_boxes.len(),
            &mut scratch,
        )
        .unwrap();
        assert_eq!(
            candidates.len(),
            2 * edge.len(),
            "each of the three sides must reach the face as a candidate, or \
             this test passes for the wrong reason"
        );
    }

    #[test]
    fn two_prescribed_elements_are_not_reported() {
        // The `either_dyn` gate. `examples/fitting` pins an entire dancing body
        // whose armpits self-intersect by construction; ungating this aborts it
        // at initialize, which is why the gate is not a tuning knob.
        let report = scan_crossed(
            &[FaceProp {
                mass: 1.0,
                fixed: true,
                ..Default::default()
            }; 2],
            &[EdgeProp {
                mass: 1.0,
                fixed: true,
                ..Default::default()
            }; 6],
            &[free_vertex(0); 6],
        );
        assert!(
            report.is_clean(),
            "an intersection between two fully prescribed elements was \
             reported, which aborts a run over geometry the solver cannot fix"
        );
    }

    #[test]
    fn two_zero_mass_elements_are_not_reported() {
        let report = scan_crossed(
            &[FaceProp::default(); 2],
            &[EdgeProp::default(); 6],
            &[free_vertex(0); 6],
        );
        assert!(report.is_clean());
    }

    #[test]
    fn one_pdrd_bodys_self_intersection_is_not_reported() {
        let report = scan_crossed(
            &[free_face(0); 2],
            &[free_edge(0); 6],
            &[VertexProp {
                mass: 1.0,
                pdrd_body_index: 3,
                ..Default::default()
            }; 6],
        );
        assert!(
            report.is_clean(),
            "a rigid body's own self-intersection is fixed and physically \
             meaningless, and reporting it aborts a run that cannot be fixed"
        );
    }

    #[test]
    fn two_colliders_are_not_reported() {
        let report = scan_crossed(
            &[free_face(0); 2],
            &[free_edge(0); 6],
            &[VertexProp {
                mass: 1.0,
                collider: true,
                ..Default::default()
            }; 6],
        );
        assert!(report.is_clean());
    }

    #[test]
    fn the_allowance_rule_is_the_shared_one() {
        // Not a restatement of the rule: every case below is evaluated by
        // `isect::intersection_tolerated`, the one body every backend renders
        // and the same one the four device visitors reach through
        // `intersect_pair_reported`. The NEGATIVE cases are the ones that
        // matter, because over-suppression is invisible in a happy-path test.
        //
        // Arguments: (object, group, policy) for each side, then the two pin
        // bits. Groups 0 and 1 are two ordinary groups.
        //
        // Either pin is enough.
        assert!(tolerated(0, 0, 0, 1, 0, 0, true, false));
        assert!(tolerated(0, 0, 0, 1, 0, 0, false, true));
        assert!(!tolerated(0, 0, 0, 1, 0, 0, false, false));
        // Same object takes the SELF flag and neither cross-object one.
        assert!(tolerated(7, 0, INTERSECT_ALLOW_SELF, 7, 0, 0, false, false));
        let cross = INTERSECT_ALLOW_INTER_OBJECT | INTERSECT_ALLOW_INTER_GROUP;
        assert!(!tolerated(7, 0, cross, 7, 0, cross, false, false));
        // Different objects take the inter-object flag from EITHER side, so a
        // flagged garment covers an unflagged character, in its own group or
        // another.
        let inter_object = INTERSECT_ALLOW_INTER_OBJECT;
        assert!(tolerated(1, 0, inter_object, 2, 0, 0, false, false));
        assert!(tolerated(1, 0, 0, 2, 1, inter_object, false, false));
        assert!(!tolerated(
            1,
            0,
            INTERSECT_ALLOW_SELF,
            2,
            1,
            INTERSECT_ALLOW_SELF,
            false,
            false
        ));
        // The inter-group flag, from EITHER side, covers two objects only when
        // their groups differ: two objects of one group still collide.
        let inter_group = INTERSECT_ALLOW_INTER_GROUP;
        assert!(tolerated(1, 0, inter_group, 2, 1, 0, false, false));
        assert!(tolerated(1, 0, 0, 2, 1, inter_group, false, false));
        assert!(!tolerated(1, 0, inter_group, 2, 0, inter_group, false, false));
        // The collision mesh belongs to no group, so it is another group from
        // every object, and the no-group marker never matches itself.
        assert!(tolerated(
            1,
            0,
            inter_group,
            NO_OBJECT_INDEX,
            NO_GROUP_INDEX,
            0,
            false,
            false
        ));
        assert!(tolerated(
            1,
            NO_GROUP_INDEX,
            inter_group,
            2,
            NO_GROUP_INDEX,
            0,
            false,
            false
        ));
        // And the unknown-object marker must not match itself, or every pair of
        // unknowns would read as one object and take the self allowance.
        assert!(!tolerated(
            NO_OBJECT_INDEX,
            0,
            INTERSECT_ALLOW_SELF,
            NO_OBJECT_INDEX,
            0,
            INTERSECT_ALLOW_SELF,
            false,
            false
        ));
        assert_eq!(unsafe { no_object_index_abi() }, NO_OBJECT_INDEX);
    }

    #[test]
    fn a_tolerated_pierce_is_suppressed_and_an_untolerated_one_is_not() {
        // The allowance reaching a real scan, both ways round. The same
        // geometry with and without the flag, so the only difference is the
        // policy.
        let mut vert_prop = vec![free_vertex(0); 6];
        for v in vert_prop.iter_mut().skip(3) {
            v.object_index = 1;
        }
        // Two different objects, neither flagged: reported.
        let report = scan_crossed(&[free_face(0); 2], &[free_edge(0); 6], &vert_prop);
        assert!(!report.is_clean(), "the unflagged case must still report");

        // One side asks for inter-object intersections to be tolerated.
        for v in vert_prop.iter_mut().skip(3) {
            v.intersect_policy = INTERSECT_ALLOW_INTER_OBJECT;
        }
        let report = scan_crossed(&[free_face(0); 2], &[free_edge(0); 6], &vert_prop);
        assert!(
            report.is_clean(),
            "the inter-object allowance did not reach the scan"
        );
    }

    #[test]
    fn edge_edge_reports_each_unordered_pair_once() {
        // Two crossing segments closer than their combined offsets, plus the
        // upper-triangular halving that keeps one report rather than two.
        let mut fx = Fixture::new();
        let vertex = vec![
            point(-1.0, 0.0, 0.0),
            point(1.0, 0.0, 0.0),
            point(0.0, -1.0, 0.001),
            point(0.0, 1.0, 0.001),
        ];
        let face: Vec<[u32; 3]> = Vec::new();
        let edge = vec![[0u32, 1], [2, 3]];
        // Each edge carries an offset of 0.01, so the pair is reported at a
        // separation of 0.001 and not at 0.1.
        let edge_param = vec![EdgeParam {
            offset: 0.01,
            ..Default::default()
        }];
        fx.stage_scene(
            &vertex,
            &face,
            &edge,
            &[],
            &[free_edge(0); 2],
            &[free_vertex(0); 4],
            &edge_param,
            &[VertexParam::default()],
        );
        let scene = fx.scene();
        // THE QUERY BOX CARRIES NO MARGIN AND THE TREE'S LEAVES DO. That
        // asymmetry is the real pipeline's: `check_intersection` queries with
        // `aabb::make(y0, y1, 0.0f)` while the tree's leaves were built with
        // `0.5 * ghat + offset`. A fixture that inflates neither finds no
        // candidate for two segments crossing 0.001 apart, because each box is
        // flat in the axis that separates them.
        let margin = 0.01f32;
        let edge_query = edge_boxes_of(&vertex, &edge);
        let edge_leaves: Vec<Aabb> = edge_query.iter().map(|b| inflated(*b, margin)).collect();
        let mut edge_tree = fx.tree(&edge_leaves);
        let query = fx.query(&edge_query);
        fx.begin();
        // Safety: every handle in `scene` names an allocation this
        // fixture staged on this device and holds for the call.
        unsafe { fx.scan.scan_edge_edge(&mut fx.device, &scene, &mut edge_tree, query) }.unwrap();
        let report = fx.finish();
        assert_eq!(
            report.found, 1,
            "two crossing edges must be reported exactly once, not once per \
             direction"
        );
        assert_eq!(report.records[0].itype, RECORD_EDGE_EDGE);

        // Move them apart by more than the offsets and it must go quiet. The
        // boxes are inflated by the same margin, so the pair is still a
        // CANDIDATE and the silence comes from the shared distance test rather
        // than from the broad phase.
        let mut fx = Fixture::new();
        let mut apart = vertex.clone();
        apart[2][2] += 0.01;
        apart[3][2] += 0.01;
        fx.stage_scene(
            &apart,
            &face,
            &edge,
            &[],
            &[free_edge(0); 2],
            &[free_vertex(0); 4],
            &edge_param,
            &[VertexParam::default()],
        );
        let scene = fx.scene();
        let query_boxes = edge_boxes_of(&apart, &edge);
        let leaves: Vec<Aabb> = query_boxes.iter().map(|b| inflated(*b, margin)).collect();
        let mut tree = fx.tree(&leaves);
        let query = fx.query(&query_boxes);
        let mut scratch = lbvh::WalkScratch::default();
        assert!(
            !lbvh::query_pairs(
                &mut fx.device,
                &mut tree,
                query,
                query_boxes.len(),
                &mut scratch
            )
            .unwrap()
            .is_empty(),
            "the separated pair must still be a candidate, or the silence below \
             proves nothing about the distance test"
        );
        let query = fx.query(&query_boxes);
        fx.begin();
        // Safety: every handle in `scene` names an allocation this
        // fixture staged on this device and holds for the call.
        unsafe { fx.scan.scan_edge_edge(&mut fx.device, &scene, &mut tree, query) }.unwrap();
        let report = fx.finish();
        assert!(report.is_clean());
    }

    #[test]
    fn overlapping_grains_are_reported_once_each() {
        // The pass a faceless SAND cloud needs: with no edges the two scans
        // above never run, and an overlapping cloud would pass the check and
        // abort mid-advance instead.
        let mut fx = Fixture::new();
        let vertex = vec![
            point(0.0, 0.0, 0.0),
            point(0.01, 0.0, 0.0),
            point(5.0, 0.0, 0.0),
        ];
        fx.stage_scene(
            &vertex,
            &[],
            &[],
            &[],
            &[],
            &[free_vertex(0); 3],
            &[EdgeParam::default()],
            &[VertexParam {
                offset: 0.02,
                ..Default::default()
            }],
        );
        let scene = fx.scene();
        let boxes: Vec<Aabb> = vertex
            .iter()
            .map(|v| {
                let r = 0.02f32;
                Aabb {
                    min: [v[0] - r, v[1] - r, v[2] - r],
                    max: [v[0] + r, v[1] + r, v[2] + r],
                    active: true,
                }
            })
            .collect();
        let mut tree = fx.tree(&boxes);
        let query = fx.query(&boxes);
        fx.begin();
        // Safety: every handle in `scene` names an allocation this
        // fixture staged on this device and holds for the call.
        unsafe { fx.scan.scan_point_point(&mut fx.device, &scene, &mut tree, query) }.unwrap();
        let report = fx.finish();
        assert_eq!(
            report.found, 1,
            "grains 0 and 1 are 0.01 apart with combined offsets of 0.04, and \
             grain 2 is far away"
        );
        assert_eq!(report.records[0].itype, RECORD_POINT_POINT);
        assert_eq!(report.records[0].num_verts0, 1);
        assert!(report.vert_flag[1], "grain 1 is the query that found it");
    }

    #[test]
    fn a_dynamic_edge_through_the_collision_mesh_is_reported() {
        // The static side is a disjoint pool with its own position array, which
        // is why it has its own entry point rather than sharing one.
        let vertex = vec![point(0.0, 0.0, -1.0), point(0.0, 0.0, 1.0)];
        let edge = vec![[0u32, 1]];
        let collision_vertex = vec![
            point(-1.0, -1.0, 0.0),
            point(1.0, -1.0, 0.0),
            point(0.0, 1.0, 0.0),
        ];
        let collision_face = vec![[0u32, 1, 2]];

        let run = |edge_prop: &[EdgeProp]| -> Report {
            let mut fx = Fixture::new();
            fx.stage_scene(
                &vertex,
                &[],
                &edge,
                &[],
                edge_prop,
                &[free_vertex(0); 2],
                &[EdgeParam::default()],
                &[VertexParam::default()],
            );
            fx.stage_collider(&collision_vertex, &collision_face);
            let scene = fx.scene();
            let collider = fx.collider();
            let mut tree = fx.tree(&[box_of(&collision_vertex)]);
            let query = fx.query(&[box_of(&vertex)]);
            fx.begin();
            // Safety: every handle in `scene` names an allocation this
            // fixture staged on this device and holds for the call.
            unsafe { fx.scan.scan_collision_mesh(&mut fx.device, &scene, &collider, &mut tree, query) }.unwrap();
            fx.finish()
        };

        let report = run(&[free_edge(0)]);
        assert_eq!(report.found, 1);
        assert_eq!(report.records[0].itype, RECORD_COLLISION_MESH);
        assert!(report.edge_flag[0]);

        // A zero-mass edge is a static solid and the collision mesh is one too,
        // so the pair can never resolve and must not be reported.
        let report = run(&[EdgeProp::default()]);
        assert!(report.is_clean());
    }

    #[test]
    fn a_dynamic_edge_linked_to_the_collision_mesh_is_not_reported() {
        // The same pierce as above, with the dynamic edge's first vertex linked
        // to a collision-mesh vertex: the tagged index is how the table names
        // that pool, and a link from either end of the edge exempts the pair.
        let vertex = vec![point(0.0, 0.0, -1.0), point(0.0, 0.0, 1.0)];
        let edge = vec![[0u32, 1]];
        let collision_vertex = vec![
            point(-1.0, -1.0, 0.0),
            point(1.0, -1.0, 0.0),
            point(0.0, 1.0, 0.0),
        ];
        let collision_face = vec![[0u32, 1, 2]];
        let tag = crate::data::START_LINK_COLLISION_VERTEX;
        let run = |rows: &[Vec<u32>]| -> Report {
            let mut fx = Fixture::new();
            fx.stage_scene(
                &vertex,
                &[],
                &edge,
                &[],
                &[free_edge(0)],
                &[free_vertex(0); 2],
                &[EdgeParam::default()],
                &[VertexParam::default()],
            );
            fx.stage_collider(&collision_vertex, &collision_face);
            fx.stage_links(rows);
            let scene = fx.scene();
            let collider = fx.collider();
            let mut tree = fx.tree(&[box_of(&collision_vertex)]);
            let query = fx.query(&[box_of(&vertex)]);
            fx.begin();
            // Safety: every handle in `scene` names an allocation this
            // fixture staged on this device and holds for the call.
            unsafe { fx.scan.scan_collision_mesh(&mut fx.device, &scene, &collider, &mut tree, query) }.unwrap();
            fx.finish()
        };
        assert!(run(&[vec![2 | tag], vec![]]).is_clean());
        // The UNTAGGED index 2 names no vertex of this collider face, so the
        // same number without the pool bit links nothing.
        assert_eq!(run(&[vec![2], vec![]]).found, 1);
    }

    #[test]
    fn a_near_coplanar_pierce_is_caught_rather_than_underflowed() {
        // THE DEFECT THE METAL PORT FOUND, and the reason the shared predicate
        // compares two SIGNS instead of the sign of their product. `s1` and `s2`
        // are signed volumes, so an edge lying nearly in the triangle's plane
        // makes both tiny while both are still ordinary normal floats, and their
        // PRODUCT underflows. The product form reported no crossing for 5 of 8
        // genuine crossings on hardware that flushes subnormals.
        //
        // This drives the scan at that magnitude rather than calling the
        // predicate directly, so it covers the position gather in front of it
        // too. The two endpoints sit 7.5e-9 metres either side of the plane,
        // which is the magnitude `intersect_core.hpp` records the product form
        // failing at on the host reference as well as on the GPU.
        let mut fx = Fixture::new();
        let vertex = vec![
            point(-1.0, -0.5, 0.0),
            point(1.0, -0.5, 0.0),
            point(0.0, 1.0, 0.0),
            // A hair below the plane, and the same distance above it.
            point(0.0, 0.0, -7.5e-9),
            point(0.0, 0.0, 7.5e-9),
        ];
        let face = vec![[0u32, 1, 2]];
        let edge = vec![[3u32, 4]];
        fx.stage_scene(
            &vertex,
            &face,
            &edge,
            &[free_face(0)],
            &[free_edge(0)],
            &[free_vertex(0); 5],
            &[EdgeParam::default()],
            &[VertexParam::default()],
        );
        let scene = fx.scene();
        let mut face_tree = fx.tree(&face_boxes_of(&vertex, &face));
        let query = fx.query(&edge_boxes_of(&vertex, &edge));
        fx.begin();
        // Safety: every handle in `scene` names an allocation this
        // fixture staged on this device and holds for the call.
        unsafe { fx.scan.scan_face_edge(&mut fx.device, &scene, &mut face_tree, query) }.unwrap();
        let report = fx.finish();
        assert_eq!(
            report.found, 1,
            "a crossing 7.5e-9 metres either side of the plane was \
             missed. That is the underflow the sign comparison exists to \
             prevent, and it is a missed interpenetration."
        );
    }

    #[test]
    fn the_report_does_not_depend_on_the_thread_count() {
        // WHAT THIS TESTS IS THE BACKEND'S CUT. The walk is one dispatch and
        // `sched` inside the backend decides how it is partitioned, so a report
        // that moved with the pool size would be the partition leaking into the
        // answer. The row is `Scatter::Claim`, which is exactly the declaration
        // that forbids that: a slot taken from a shared counter is reproducible
        // only in ascending order, so the backend runs one serial pass.
        let mut fx = Fixture::new();
        let (vertex, face, edge) = crossed_triangles();
        fx.stage_scene(
            &vertex,
            &face,
            &edge,
            &[free_face(0); 2],
            &[free_edge(0); 6],
            &[free_vertex(0); 6],
            &[EdgeParam::default()],
            &[VertexParam::default()],
        );
        let scene = fx.scene();
        let mut face_tree = fx.tree(&face_boxes_of(&vertex, &face));
        let edge_boxes = edge_boxes_of(&vertex, &edge);

        let query = fx.query(&edge_boxes);
        fx.begin();
        // Safety: every handle in `scene` names an allocation this
        // fixture staged on this device and holds for the call.
        unsafe { fx.scan.scan_face_edge(&mut fx.device, &scene, &mut face_tree, query) }.unwrap();
        let report = fx.finish();
        let reference: Vec<_> = report
            .records
            .iter()
            .map(|r| (r.itype, r.elem0, r.elem1))
            .collect();
        assert!(!reference.is_empty());
        for threads in [1usize, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let query = fx.query(&edge_boxes);
            fx.begin();
            pool.install(|| {
                // Safety: every handle in `scene` names an allocation this
                // fixture staged on this device and holds for the call.
                unsafe { fx.scan.scan_face_edge(&mut fx.device, &scene, &mut face_tree, query) }
            })
            .unwrap();
            let report = fx.finish();
            let got: Vec<_> = report
                .records
                .iter()
                .map(|r| (r.itype, r.elem0, r.elem1))
                .collect();
            assert_eq!(reference, got, "the report moved at {threads} threads");
        }
    }

    #[test]
    fn the_found_count_keeps_rising_past_the_record_buffer() {
        // A run aborted on intersections wants the TRUE count. A report that
        // stops counting where it stops storing understates the problem, and
        // 256 is a small number for a tangled import.
        //
        // THE COUNT IS THE DEVICE'S OWN, claimed past the array by
        // `intersection_record_claim`, so this drives a real scan over more
        // intersections than the buffer holds rather than exercising a host
        // counter. One wide triangle in the z = 0 plane, and 300 short segments
        // standing through its interior.
        let crossings = max_records() + 44;
        let mut fx = Fixture::new();
        let mut vertex = vec![
            point(-1.0, -1.0, 0.0),
            point(1.0, -1.0, 0.0),
            point(0.0, 1.0, 0.0),
        ];
        let mut edge: Vec<[u32; 2]> = Vec::new();
        for slot in 0..crossings {
            // Inside the triangle: a narrow band about the centroid's height.
            let x = -0.5 + (slot as f32) * (1.0 / crossings as f32);
            let base = vertex.len() as u32;
            vertex.push(point(x, -0.5, -0.1));
            vertex.push(point(x, -0.5, 0.1));
            edge.push([base, base + 1]);
        }
        let face = vec![[0u32, 1, 2]];
        fx.stage_scene(
            &vertex,
            &face,
            &edge,
            &[free_face(0)],
            &vec![free_edge(0); edge.len()],
            &vec![free_vertex(0); vertex.len()],
            &[EdgeParam::default()],
            &[VertexParam::default()],
        );
        let scene = fx.scene();
        let mut face_tree = fx.tree(&face_boxes_of(&vertex, &face));
        let query = fx.query(&edge_boxes_of(&vertex, &edge));
        fx.begin();
        // Safety: every handle in `scene` names an allocation this
        // fixture staged on this device and holds for the call.
        unsafe { fx.scan.scan_face_edge(&mut fx.device, &scene, &mut face_tree, query) }.unwrap();
        let report = fx.finish();
        assert_eq!(
            report.found as usize, crossings,
            "every crossing must be counted, whether or not a record was stored \
             for it"
        );
        assert_eq!(
            report.records.len(),
            max_records(),
            "the stored records stop at the buffer's capacity"
        );
    }
}
