// File: crates/ppf-cts-solver/src/driver/lbvh.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The linear BVH: the tree Karras builds from a Morton order, its bounding
//! boxes, and the traversal that turns two trees into candidate pairs.
//!
//! `bvh.rs` supplies the half that comes before this one (scene bounds, Morton
//! codes, the stable order). This module builds the tree from that order,
//! refreshes its boxes against a Newton iterate, and walks it.
//!
//! # The split, and it is the same one everywhere in this backend
//!
//! | shared C++ | here, in Rust |
//! |---|---|
//! | the leaf and internal node, the split search, the depth walk (`lbvh.kernel.cpp`) | the loop bounds, the parent scatter, the level grouping, the buffers |
//! | every box: make, merge, overlap (`aabb.kernel.cpp`) | which box, indexed how |
//! | the traversal stack and its invariants (`aabb_traversal.kernel.cpp`) | which tree, which queries, where the pairs land |
//!
//! **The tree's TOPOLOGY is a computed value, not a layout choice.** Two
//! implementations of the split search are two different trees over the same
//! Morton order, whose traversals find the same contacts in a different order,
//! whose fp32 assembly then sums differently. Nothing asserts. So the split
//! search is shared even though what surrounds it is plain bookkeeping.
//!
//! # Node layout, which the shared bodies fix
//!
//! For `n` primitives there are `2n - 1` nodes (`1` when `n == 1`). Leaves
//! occupy `[0, n)` IN SORTED ORDER, so leaf `i` holds primitive `order[i]`, and
//! internal nodes occupy `[n, 2n - 1)`. Each node is two `u32`:
//!
//! - a leaf is `(primitive + 1, 0)`, so `second == 0` IS the leaf test and the
//!   `+ 1` is what makes primitive 0 distinguishable from it;
//! - an internal node is `(left + 1, right + 1)`.
//!
//! The root is the one internal node no other node points at, and it is
//! reported rather than relocated: a query seeds its stack from `root`.
//!
//! # Determinism
//!
//! Every pass here is either a disjoint per-index write or a serial fold in
//! ascending index, so the tree, its boxes and the pair list are identical at
//! any thread count. That is asserted rather than assumed: the device's level
//! grouping uses an atomic and so permutes a level's members run to run, which
//! is harmless there because a level's nodes are disjoint, and is simply not
//! done here.


use ppf_cts_compute::{AllocLabel, Buffer, Device, EncoderExt, Fault, ReadbackBuffer};
use super::kernels::{
    AabbLeafEdgeArgs, AabbLeafFaceArgs, AabbLeafVertexArgs, AabbMergeLevelArgs, LbvhNodeDepthArgs,
    LbvhNodesArgs, VecFillU32Args, LbvhSetParentArgs, LbvhFindRootArgs,
    LbvhCountLevelsArgs, LbvhScatterLevelsArgs,};
use super::scene::{Fatal, FatalResult};
use super::bvh;

/// A traversal stack that ran out of room, or a node index past the end of the
/// tree, as the shared traversal body recorded it.
///
/// Every one of these is guarantee-class rather than cosmetic. The body's
/// response to a full stack is to `break`, which abandons the unvisited
/// subtrees, and an abandoned subtree is a candidate pair that was never
/// generated: a missed contact, and therefore a possible penetration. So the
/// channel exists and the caller raises on it; it is never counted and ignored.
///
/// No PRODUCTION caller: the walk that raises one is reached from tests alone.
/// What reads it is
/// `a_walk_over_a_corrupt_tree_faults_rather_than_returning_a_short_list`, which
/// points the root's left child past the end of the node array and requires an
/// error carrying the shared body's own file, line and failure count, rather
/// than the short pair list an abandoned walk would otherwise return.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct TraversalFault {
    /// How many checks failed in total, across every chunk.
    pub fail_count: u32,
    /// The four floats the first failing check recorded.
    pub payload: [f32; 4],
    /// Source file and line of that check, inside the shared traversal body.
    pub file: String,
    pub line: u32,
    /// THE REAL FAULT, when the walk failed to RUN rather than tripping a check.
    ///
    /// A transport failure and an abandoned subtree are different verdicts and
    /// must not share a message. `None` means the device genuinely recorded a
    /// failing check and the fields above describe it; `Some` means the walk
    /// never produced a verdict at all, and this is what actually went wrong.
    pub transport: Option<String>,
}

impl TraversalFault {
    /// A walk that could not be PERFORMED, carrying the fault verbatim.
    ///
    /// EVERY FIELD ABOVE IS A PLACEHOLDER ON THIS PATH, which is the whole
    /// reason this constructor exists. Fabricating an all-zero payload and an
    /// empty file, then rendering them through the invariant message below, is
    /// how an allocation failure came to be reported as "BVH traversal
    /// invariant failed ... with payload [0, 0, 0, 0]" for two example scenes:
    /// a guarantee-class verdict about missed contacts, standing in for a
    /// buffer the driver could not size. The zeros were literals, not
    /// measurements, and the empty file is what gives it away, because no real
    /// record can carry one.
    ///
    /// No PRODUCTION caller: its five callers are the transport error paths in
    /// [`walk`], which tests alone reach. The distinction it carries is what
    /// `a_walk_over_a_corrupt_tree_faults_rather_than_returning_a_short_list`
    /// asserts from the other side: that fixture corrupts the tree, so the
    /// device records a real failing check and the fault it reads must carry
    /// `transport: None` and the shared body's own file and line.
    #[allow(dead_code)]
    pub fn transport(error: impl std::fmt::Display) -> Self {
        Self {
            fail_count: 0,
            payload: [0.0; 4],
            file: String::new(),
            line: 0,
            transport: Some(error.to_string()),
        }
    }

    /// The sentence a caller reports a fault with.
    ///
    /// No PRODUCTION caller: nothing outside this module holds a
    /// [`TraversalFault`]. What reads it is
    /// `a_walk_over_a_corrupt_tree_faults_rather_than_returning_a_short_list`,
    /// which requires the message to say the pair list is "incomplete", so a
    /// walk that abandoned subtrees cannot be read as an ordinary short result.
    #[allow(dead_code)]
    pub fn describe(&self) -> String {
        if let Some(detail) = &self.transport {
            return format!(
                "the walk could not be performed: {detail}. This is a TRANSPORT \
                 failure, not a traversal invariant: the broad phase produced no \
                 verdict at all, so nothing is known about the candidate pairs \
                 and the step cannot proceed on them."
            );
        }
        format!(
            "BVH traversal invariant failed {} time(s); first at {}:{} with \
             payload [{}, {}, {}, {}]. A traversal that trips this ABANDONS \
             subtrees, so the candidate pairs it did produce are incomplete and \
             must not be used.",
            self.fail_count,
            self.file,
            self.line,
            self.payload[0],
            self.payload[1],
            self.payload[2],
            self.payload[3]
        )
    }
}

/// The diagnostic record the compiled traversal body writes, one per chunk.
///
/// A re-export rather than a second declaration. The record is TRANSPORT, so it
/// belongs to `ppf-cts-compute` with the merge rule that reads it, and a mirror
/// pair here would be two declarations that can disagree with nothing to catch
/// it. What names it here is `the_diag_record_matches_cpp`, which compares this
/// size against the C++ compiler's own answer, so the alias is `#[cfg(test)]`
/// and the release build carries no import for it.
#[cfg(test)]
pub use ppf_cts_compute::DiagRecord as Diag;

/// The Rust mirror of the device-only `AABB` in `src/kernels/data.hpp`.
///
/// It is NOT part of the wire ABI, so nothing else checks this layout: a drift
/// here is a silent wrong answer rather than a link error. `the_aabb_mirror_matches_cpp`
/// compares the size and every offset against the C++ compiler's own answer.
///
/// `min` and `max` are position components, so they are `f32`, matching the
/// `Vec3f` pair the shared `AABB` declares. The two must agree: the struct is
/// the element type of device buffers the shared bodies fill and read, and a
/// width that matched while the interpretation did not would be read as a box
/// somewhere else entirely.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub active: bool,
}

// SAFETY: the layout is stated above and `the_aabb_mirror_matches_cpp` compares
// its size and every offset against the C++ compiler's own answer, so a device
// allocation of these and the shared bodies' view of them are the same bytes.
// Every field is a float or a `bool` the bodies write as 0 or 1, and the
// padding the 32-byte alignment adds is carried rather than interpreted: Rust
// reads `min`, `max` and `active`, and nothing reads between them.
unsafe impl ppf_cts_compute::Pod for Aabb {}

// Written out rather than derived, because what matters is not the values but
// WHY they are these values, and the derive would carry no explanation of it.
#[allow(clippy::derivable_impls)]
impl Default for Aabb {
    fn default() -> Self {
        // An INACTIVE box, which `aabb_overlap` rejects against anything.
        // Zeroed bounds with `active` set would be a box at the origin, which
        // overlaps whatever is there; a default that silently participates in
        // the broad phase is exactly the plausible-wrong-answer path this
        // project treats as a defect.
        Aabb {
            min: [0.0; 3],
            max: [0.0; 3],
            active: false,
        }
    }
}

// WHAT REMAINS A DIRECT CALL HERE, AND WHY.
//
// NONE OF THE SEVEN IS A DISPATCH: six layout queries and one two-box
// predicate, each taking scalars or two structs and returning one scalar, with
// no thread index, no extent and no buffer. Five of the six queries are what
// `the_aabb_mirror_matches_cpp` compares this module's Rust mirror of the
// device-only `AABB` against, and nothing else compares those two layouts, so a
// drift there is a silent wrong answer rather than a link error; the sixth is
// the traversal stack depth, which a test reads to check the stack is deep
// enough for the trees this backend builds. All seven are the same category as
// `super::pcg`'s per-block inverse and `super::step`'s domain query, and they
// move when `Device` grows a query surface, with those two and not before.
//
// The walk itself is not among them: it goes through `Device::launch` like
// every other pass here, on the per-query slot layout written at [`walk`].
extern "C" {
    /// No PRODUCTION caller: `the_aabb_mirror_matches_cpp` requires
    /// `size_of::<Aabb>()` to equal the C++ compiler's own answer, which is the
    /// only check that a device allocation of `Aabb` and the shared bodies'
    /// view of it are the same bytes.
    #[allow(dead_code)]
    fn aabb_sizeof_abi() -> u32;
    /// No PRODUCTION caller: `the_aabb_mirror_matches_cpp` requires
    /// `offset_of!(Aabb, min)` to equal this, so the minimum Rust carries and
    /// the one the shared bodies read are the same field.
    #[allow(dead_code)]
    fn aabb_offset_min_abi() -> u32;
    /// No PRODUCTION caller: `the_aabb_mirror_matches_cpp` requires
    /// `offset_of!(Aabb, max)` to equal this, on the same terms as
    /// `aabb_offset_min_abi`.
    #[allow(dead_code)]
    fn aabb_offset_max_abi() -> u32;
    /// No PRODUCTION caller: `the_aabb_mirror_matches_cpp` requires
    /// `offset_of!(Aabb, active)` to equal this. That flag decides whether a
    /// box takes part in the broad phase at all, so a drift in its offset reads
    /// a padding byte and admits or prunes leaves at random.
    #[allow(dead_code)]
    fn aabb_offset_active_abi() -> u32;
    /// No PRODUCTION caller: `the_aabb_mirror_matches_cpp` requires
    /// `size_of::<Diag>()` to equal the C++ record's size, so a traversal fault
    /// is read out of the bytes the shared body wrote rather than out of the
    /// wrong ones.
    #[allow(dead_code)]
    fn diag_sizeof_abi() -> u32;
    fn aabb_max_query_abi() -> u32;
    /// No PRODUCTION caller: it is the sole implementation behind [`overlap`],
    /// which `a_traversal_finds_exactly_the_overlapping_pairs` builds its
    /// brute-force expectation from, so the tree walk is compared against the
    /// shared predicate rather than against a second implementation of it.
    #[allow(dead_code)]
    fn aabb_overlap_abi(a: *const Aabb, b: *const Aabb) -> i32;
}

/// The traversal stack depth the shared body was compiled with.
///
/// Read from C++ rather than restated, because a stack overflow is a silent
/// truncation of the walk and the number that decides it must have one home.
///
/// No PRODUCTION caller: the depth is enforced inside the shared traversal body,
/// which records a fault rather than asking anyone. What reads it here is the
/// test that checks the stack is deep enough for the trees this backend builds.
#[allow(dead_code)]
pub fn max_query_depth() -> u32 {
    unsafe { aabb_max_query_abi() }
}

/// Two boxes overlap, as the shared body decides it.
///
/// No PRODUCTION caller: the production passes test boxes inside the shared
/// traversal and scan bodies, which call `aabb_overlap` on the device. Two
/// tests read it. `a_default_box_takes_part_in_nothing` requires a default box
/// to overlap nothing, which is what keeps an unrefreshed leaf out of the broad
/// phase, and `a_traversal_finds_exactly_the_overlapping_pairs` builds its
/// brute-force expectation from it, so the walk is checked against the shared
/// predicate rather than against a second implementation of it.
#[allow(dead_code)]
pub fn overlap(a: &Aabb, b: &Aabb) -> bool {
    unsafe { aabb_overlap_abi(a, b) != 0 }
}

/// A test's view of a device-resident tree.
///
/// THE TESTS ARE THE ONLY HOST READERS LEFT. Production reads neither array on
/// the host now, which is what let both move; a fixture that asserts on the
/// tree's shape brings them back explicitly.
#[cfg(test)]
pub(crate) struct TreeHost {
    pub node: Vec<u32>,
    pub aabb: Vec<Aabb>,
    pub level_data: Vec<u32>,
}

#[cfg(test)]
pub(crate) fn tree_host(device: &mut impl Device, tree: &mut Tree) -> TreeHost {
    let mut node = vec![0u32; tree.node.len()];
    let mut aabb = vec![Aabb::default(); tree.aabb.len()];
    if !node.is_empty() {
        tree.node.download(device).expect("the nodes read back");
        node.copy_from_slice(tree.node.host());
    }
    if !aabb.is_empty() {
        tree.aabb
            .read(device, 0, &mut aabb)
            .expect("the boxes read back");
    }
    let mut level_data = vec![0u32; tree.level_data.len()];
    if !level_data.is_empty() {
        tree.level_data
            .read(device, 0, &mut level_data)
            .expect("the levels read back");
    }
    TreeHost {
        node,
        aabb,
        level_data,
    }
}

/// The buffers one tree BUILD stages through.
///
/// PERSISTENT AND OWNED BY THE CALLER, for the reason [`WalkScratch`] is: a
/// build runs once per tree per step and `Buffer::size` grows only past
/// CAPACITY, so a tree no larger than the last allocates nothing. What passes
/// through here is the Morton sort's own arrays and the four small device
/// arrays the level pass fills.
#[derive(Default)]
pub struct BuildScratch {
    /// The tree's parent links, its root and the level bins, all device
    /// resident: the four passes that fill them are kernels
    /// (`src/kernels/lbvh/lbvh.kernel.cpp`). `root` carries the index in slot 0
    /// and the number of roots found in slot 1, so a tree that produced two
    /// roots is still detectable from the host without downloading the node
    /// array.
    pub parent_device: Buffer<u32>,
    pub root: ReadbackBuffer<u32>,
    pub level_counts: ReadbackBuffer<u32>,
    pub level_cursor: Buffer<u32>,
    /// The per-level starts the scatter reads, staged from the host scan.
    pub level_offset_device: Buffer<u32>,
    /// The device sort's arrays; see [`super::devsort`].
    pub sort: super::devsort::SortScratch,
    /// The centroid-bounds reduction's levels; see [`super::bvh::BoundsScratch`].
    pub bounds: super::bvh::BoundsScratch,
    /// The Morton codes a host caller hands [`build`], staged for the node
    /// pass.
    ///
    /// No PRODUCTION caller: the per-step path sorts on the device and reaches
    /// [`build_presorted`] with handles, so [`build`] is the only writer, and
    /// `build` backs the `tree_of` fixture in this module's tests.
    #[allow(dead_code)]
    pub sorted_codes: Buffer<u32>,
    /// That caller's permutation, staged beside `sorted_codes` and on the same
    /// terms: written by [`build`] alone, which the `tree_of` fixture in this
    /// module's tests reads.
    #[allow(dead_code)]
    pub order: Buffer<u32>,
    /// KERNEL-WRITTEN AND KERNEL-READ. `lbvh_node_depth` fills it and
    /// `lbvh_count_levels` and `lbvh_scatter_levels` read it; nothing on the
    /// host does, so no pass downloads the array.
    pub depth: ReadbackBuffer<u32>,
}

/// The two output buffers a walk writes, plus the diagnostic lane.
///
/// PERSISTENT AND OWNED BY THE CALLER, because a walk runs once per contact
/// kind per Newton step and `Buffer::size` grows only past CAPACITY: a walk no
/// wider than the last allocates nothing. Allocating per call would instead cost
/// one allocation per query chunk per step.
///
/// No PRODUCTION caller: [`query_pairs`] and [`walk`], which take one, are
/// reached from tests alone. Six tests here bind one, from
/// `an_empty_input_builds_an_empty_tree_rather_than_no_tree` through
/// `a_walk_over_a_corrupt_tree_faults_rather_than_returning_a_short_list`, and
/// two in `driver/intersection.rs` bind one to walk a scan's tree themselves,
/// `a_face_and_an_edge_sharing_a_vertex_are_never_reported` and
/// `edge_edge_reports_each_unordered_pair_once`.
#[allow(dead_code)]
#[derive(Default)]
pub struct WalkScratch {
    /// `2 * queries * capacity` slots, each query owning its own run.
    pub out: ReadbackBuffer<u32>,
    /// The count each query WANTED, which is what sizes a retry.
    pub found: ReadbackBuffer<u32>,
}

/// One built tree.
///
/// NOT `Clone` any more: two of its arrays are device allocations, and a clone
/// would hand out a second `Tree` naming the same arena spans, which is an
/// alias rather than a copy.
#[derive(Debug, Default)]
pub struct Tree {
    /// Two `u32` per node, as the shared bodies encode them.
    ///
    /// DEVICE-RESIDENT, which the traversal being a dispatch is what allows. A
    /// `walk` that descended on the host would address these directly, so a
    /// handle would cost a whole-tree download per query pass; `aabb::query` is
    /// a device body instead. No production code reads either array on the
    /// host.
    /// KERNEL-WRITTEN AND KERNEL-READ. `lbvh_nodes` writes it and
    /// `lbvh_set_parent` reads it on the device to find each node's parent.
    /// Nothing downloads the whole array between those two: only `tree_host`
    /// takes one and it is `#[cfg(test)]`.
    pub node: ReadbackBuffer<u32>,
    /// `2n - 1`, or `1` for a single primitive, or `0` for none.
    pub node_count: u32,
    /// Where a query seeds its stack.
    pub root: u32,
    /// `level_offset[d] .. level_offset[d + 1]` indexes `level_data` for depth
    /// `d`. Depth 0 holds the root alone.
    pub level_offset: Vec<u32>,
    /// Node indices grouped by depth.
    ///
    /// THE ORDER WITHIN A DEPTH IS NOT FIXED, and does not need to be: a
    /// level's nodes are independent, each box depending only on children a
    /// level deeper. The host arm's `Scatter::Claim` runs one ascending pass so
    /// it is ascending there; a GPU claims in whatever order its threads
    /// arrive.
    /// DEVICE-RESIDENT AND DEVICE-BUILT. `lbvh_scatter_levels` fills it at the
    /// tree build, claiming a slot within its level through an atomic cursor,
    /// and from there only `propagate` reads it, one `span` per level. Nothing
    /// reads it on the host.
    pub level_data: Buffer<u32>,
    /// One box per node. Leaves are written by a `refresh_*` call and internal
    /// nodes by `propagate`. Device-resident, as [`Self::node`].
    pub aabb: Buffer<Aabb>,
}

impl Tree {
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    /// How many primitives the tree was built over.
    pub fn primitive_count(&self) -> u32 {
        match self.node_count {
            0 => 0,
            1 => 1,
            n => n.div_ceil(2),
        }
    }
}

/// Build the tree over `order`, whose `i`-th entry is the primitive at sorted
/// position `i`, and `sorted_codes`, that primitive's Morton code.
///
/// The two must come from `bvh::primitive_order` applied to the same code
/// array. Passing an unsorted pair is not a detectable error here (any
/// permutation builds SOME tree), so the sort's contract carries it: see
/// `sort.rs`.
///
/// # Safety
/// The two input slices and this function's own node buffer are named to the
/// backend by address, so they must stay alive and unmoved until the dispatch
/// returns. They are the caller's slices and a local, so that holds by
/// construction; the `unsafe` is here because [`Device::launch`] cannot know it.
/// Build the tree INTO a caller-owned `Tree`, which is what keeps it
/// allocation-free after the first step.
///
/// THE SHAPE IS THE POINT, and returning a `Tree` was a LEAK. This runs once per
/// `advance()` for each of the three trees, and its three arrays are device
/// allocations; `Buffer` has no `Drop` on purpose, because a drop cannot reach
/// the device that owns the span, so a returned `Tree` assigned over a live one
/// abandons nine arena spans PER STEP and nothing ever reclaims them. That is
/// the `fn to_flat(&self, device) -> Flat` shape this crate refuses: correct in
/// its contents and wrong in its signature. It cost three example scenes on
/// Metal, which died of `kIOGPUCommandBufferCallbackErrorOutOfMemory` at frames
/// 4, 68 and 102 having reached the same cumulative total from three different
/// rates; CUDA leaks it identically and merely has an order of magnitude more
/// headroom before it matters.
///
/// Reuse is exact rather than approximate: the primitive count is fixed for a
/// run, so `node_count` is identical on every rebuild, and `Buffer::size` on a
/// live handle takes neither the `alloc` branch nor the `grow` branch. What
/// makes the reuse exact is that the build WRITES every slot it later reads,
/// not that the buffer arrives zeroed: `size` zeroes only bytes it has just
/// allocated, so the no-op path leaves the previous contents in place. Measured
/// with that path made to write 0xCD instead: 20 of 20 boundable scenes on CUDA
/// and 27 of 27 Metal fixtures were unaffected.
/// Sort the Morton codes and their permutation ON THE DEVICE.
///
/// THE SORT STAYS ON THE DEVICE. Sorting on the host is the computation itself
/// relocated: it would run once per tree per step and force a readback of the
/// codes to get there, and the permutation would then have to be uploaded again
/// for the node pass that reads it.
///
/// A BITONIC NETWORK RATHER THAN A RADIX SORT, because a network is a sequence
/// of independent compare-exchange passes with no shared counter, so it renders
/// as one neutral body every backend compiles and needs neither a histogram nor
/// an ordered claim. It costs `O(log^2 n)` passes against radix's `O(k)`.
///
/// THE ARRAY IS PADDED TO A POWER OF TWO AND THAT IS NOT OPTIONAL. A bitonic
/// network sorts a power-of-two array; skipping the comparators whose partner
/// is past the data does NOT sort the rest, it leaves the network incomplete,
/// and the tree builder then rejects the result with "the tree has two roots".
/// The padding is `u32::MAX`, which sorts to the tail, so the first `n` entries
/// are the answer.
///
/// THE ORDER IS TOTAL, so this agrees with the stable host sort it replaces:
/// the comparator takes the key first and the ORIGINAL INDEX second, so equal
/// keys keep ascending index order, which is what a stable sort by key gives.
///
/// # Safety
/// `codes` must hold `n` Morton codes and the scratch must outlive the returned
/// spans, which address its own buffers.
pub unsafe fn sort_morton<D: Device>(
    device: &mut D,
    codes: &mut ReadbackBuffer<u32>,
    scratch: &mut BuildScratch,
    n: usize,
) -> FatalResult<(ppf_cts_compute::Handle, ppf_cts_compute::Handle)> {
    let source = codes.handle();
    scratch
        .sort
        .sort(device, "lbvh.morton_sort", source, n)
        .map_err(|error| {
            Fatal::out_of_memory(format!("solver driver: cannot sort the Morton codes: {error:?}"))
        })
}

/// [`build_presorted`], for a caller holding the two arrays as HOST SLICES:
/// `order` as `bvh::primitive_order` produced it, and `sorted_codes` as that
/// primitive's Morton code.
///
/// No PRODUCTION caller: the per-step build sorts on the device and calls
/// [`build_presorted`] with the handles the sort returns. What reads this is
/// the `tree_of` fixture in this module's tests, which builds a tree over codes
/// of a known shape for `a_single_primitive_builds_a_single_leaf`,
/// `an_empty_input_builds_an_empty_tree_rather_than_no_tree` and
/// `the_leaf_margin_matches_cuda`.
///
/// # Safety
/// As [`build_presorted`]: the two slices are staged into `scratch` here, and
/// the handles this hands on name those allocations for the call.
#[allow(dead_code)]
pub unsafe fn build<D: Device>(
    device: &mut D,
    scratch: &mut BuildScratch,
    sorted_codes: &[u32],
    order: &[u32],
    tree: &mut Tree,
) -> FatalResult<()> {
    let n = order.len();
    assert_eq!(
        sorted_codes.len(),
        n,
        "a Morton code per primitive is required"
    );
    if n > 0 {
        scratch
            .sorted_codes
            .size(device, n, AllocLabel("lbvh.sorted_codes"))
            .and_then(|()| scratch.sorted_codes.write(device, 0, sorted_codes))
            .and_then(|()| scratch.order.size(device, n, AllocLabel("lbvh.order")))
            .and_then(|()| scratch.order.write(device, 0, order))
            .map_err(|error| {
                Fatal::out_of_memory(format!(
                    "solver driver: cannot stage the tree build: {error:?}"
                ))
            })?;
    }
    let morton = if n > 0 {
        scratch.sorted_codes.span(0, n)
    } else {
        ppf_cts_compute::Handle::NONE
    };
    let sorted = if n > 0 {
        scratch.order.span(0, n)
    } else {
        ppf_cts_compute::Handle::NONE
    };
    build_presorted(device, scratch, morton, sorted, n, tree)
}

/// As [`build`], on arrays the caller has already sorted ON THE DEVICE.
///
/// THIS IS THE PATH THE PER-STEP TREE BUILD TAKES. `build` above serves a caller
/// that still holds host slices; taking the two handles instead removes the
/// readback of the codes, the host sort, the host gather and the upload that a
/// host sort costs on every tree of every step.
///
/// # Safety
/// `morton` and `sorted` must name live allocations of `n` elements each,
/// already in the order the node builder expects.
pub unsafe fn build_presorted<D: Device>(
    device: &mut D,
    scratch: &mut BuildScratch,
    morton: ppf_cts_compute::Handle,
    sorted: ppf_cts_compute::Handle,
    n: usize,
    tree: &mut Tree,
) -> FatalResult<()> {
    if n == 0 {
        // An EMPTY tree, not a missing one. `lbvh::initialize` sizes all three
        // trees off `max(faces, edges, verts)` for the same reason: a scene
        // class with no primitives of one kind (a faceless SAND cloud has no
        // faces, edges or hinges) must still hand the traversal a valid handle
        // rather than a sentinel that traps when it is merely passed.
        // The buffers are left as they are rather than replaced with empty
        // ones: `node_count == 0` is what every reader tests, so their contents
        // are unreachable, and dropping a live handle here would leak the span
        // this function exists not to leak.
        tree.node_count = 0;
        tree.root = 0;
        tree.level_offset = vec![0];
        return Ok(());
    }

    let node_count = if n == 1 { 1 } else { 2 * n - 1 };
    let node = &mut tree.node;
    node.size(device, 2 * node_count, AllocLabel("lbvh.node"))
        .map_err(|error| Fatal::out_of_memory(format!(
            "solver driver: cannot size the tree's nodes: {error:?}"
        )))?;

    // Node `i` writes only slot `i` (its leaf) and slot `n + i` (the internal
    // node above it), so the row is `Scatter::Disjoint` and the backend may cut
    // the range where it likes.
    let args = LbvhNodesArgs {
        morton,
        sorted,
        primitive_count: n as u32,
        nodes: node.handle(),
        count: n as u32,
        seam_arena_count: 0,
    };
    device.launch("lbvh.nodes", &args, n as u32)?;
    // The handle is taken here so the `&mut tree.node` borrow ends before
    // `build_levels` takes `&mut tree` for the level arrays.
    let node_handle = node.handle();

    // THE PARENT LINKS, THE ROOT AND THE LEVELS, ALL ON THE DEVICE. What
    // crosses back is the root and its count, and the per-level counts. The node
    // array itself does NOT: downloading it whole and folding it serially would
    // move a device pass onto the host.
    let (level_offset, root) =
        build_levels(device, scratch, tree, node_handle, n, node_count)?;

    let aabb = &mut tree.aabb;
    aabb.size(device, node_count, AllocLabel("lbvh.aabb"))
        .map_err(|error| Fatal::out_of_memory(format!(
            "solver driver: cannot size the tree's boxes: {error:?}"
        )))?;
    tree.node_count = node_count as u32;
    tree.root = root;
    tree.level_offset = level_offset;
    Ok(())
}

/// The parent links, the root and the level runs, on the device.
///
/// FOUR KERNELS AND A SMALL SCAN. The alternative shape is three host folds over
/// a downloaded node array, a serial parent sweep, a serial root search and a
/// counting sort by depth, and calling those "index bookkeeping rather than a
/// kernel" does not make them cheap: each one is O(node_count) on the host and
/// each one needs the whole node array copied back first.
///
/// WHAT CROSSES TO THE HOST is the root and its count, then the per-level
/// counts: two words and `LEVEL_CAPACITY` words, against the node array's `2n`.
/// The scan that follows runs over the LEVELS rather than over the nodes, so it
/// is a fixed 64-step loop whatever the scene's size.
///
/// # Safety
/// As [`build`]: `nodes` names `2 * node_count` words the caller keeps alive.
unsafe fn build_levels<D: Device>(
    device: &mut D,
    scratch: &mut BuildScratch,
    tree: &mut Tree,
    nodes: ppf_cts_compute::Handle,
    n: usize,
    node_count: usize,
) -> FatalResult<(Vec<u32>, u32)> {
    /// The deepest tree the level bins are sized for.
    ///
    /// A balanced Morton tree over `n` primitives is `log2(n)` deep, so 64 is
    /// unreachable by construction; a degenerate tree that reached it is a
    /// defect to report rather than a size to grow.
    const LEVEL_CAPACITY: usize = 64;

    scratch
        .parent_device
        .size(device, node_count, AllocLabel("lbvh.parent"))
        .and_then(|()| scratch.root.size(device, 2, AllocLabel("lbvh.root")))
        .and_then(|()| scratch.depth.size(device, node_count, AllocLabel("lbvh.depth")))
        .and_then(|()| {
            scratch
                .level_counts
                .size(device, LEVEL_CAPACITY, AllocLabel("lbvh.level_counts"))
        })
        .and_then(|()| {
            scratch
                .level_cursor
                .size(device, LEVEL_CAPACITY, AllocLabel("lbvh.level_cursor"))
        })
        .map_err(|error| {
            Fatal::out_of_memory(format!(
                "solver driver: cannot size the tree's level pass: {error:?}"
            ))
        })?;

    // THE SENTINEL FIRST, because `lbvh_set_parent` writes only the slots a
    // parent claims and `lbvh_find_root` reads every slot to find the one that
    // was never claimed.
    let fill = VecFillU32Args {
        array: scratch.parent_device.handle(),
        value: u32::MAX,
        count: node_count as u32,
        seam_arena_count: 0,
    };
    device.launch("lbvh.parent_init", &fill, node_count as u32)?;

    let parent = scratch.parent_device.handle();
    let set = LbvhSetParentArgs {
        nodes,
        parent,
        count: node_count as u32,
        seam_arena_count: 0,
    };
    device.launch("lbvh.set_parent", &set, node_count as u32)?;

    // A SINGLE PRIMITIVE HAS NO INTERNAL NODE, so there is nothing for the root
    // search to find and the one leaf is the root. The shape is settled here,
    // before anything is dispatched.
    let root = if n == 1 {
        let single = VecFillU32Args {
            array: parent,
            value: 0,
            count: 1,
            seam_arena_count: 0,
        };
        device.launch("lbvh.parent_single", &single, 1)?;
        scratch.root.seed(device, &[0, 1]).map_err(|error| {
            Fatal::invariant(format!("solver driver: cannot seed the root: {error:?}"))
        })?;
        0
    } else {
        scratch.root.seed(device, &[0, 0]).map_err(|error| {
            Fatal::invariant(format!("solver driver: cannot clear the root: {error:?}"))
        })?;
        let internal = (n - 1) as u32;
        let find = LbvhFindRootArgs {
            parent,
            primitive_count: n as u32,
            root: scratch.root.span(0, 1),
            found: scratch.root.span(1, 1),
            count: internal,
            seam_arena_count: 0,
        };
        device.launch("lbvh.find_root", &find, internal)?;
        scratch.root.download(device).map_err(|error| {
            Fatal::invariant(format!(
                "solver driver: cannot read back the tree's root: {error:?}"
            ))
        })?;
        let found = scratch.root.host()[1];
        // THE HOST'S OLD ASSERTION, KEPT. A serial scan could see every slot and
        // say "the tree has two roots"; a kernel cannot, so the count is what
        // carries the same guarantee. Either answer means the node array is not
        // a tree and every traversal over it would be wrong in a way nothing
        // downstream detects.
        if found != 1 {
            return Err(Fatal::invariant(format!(
                "solver driver: the BVH node array names {found} roots rather than one, \
                 so it is not a tree and every traversal over it would be wrong in a \
                 way nothing downstream can detect"
            )));
        }
        scratch.root.host()[0]
    };

    let depth_args = LbvhNodeDepthArgs {
        parent,
        root: scratch.root.span(0, 1),
        depth: scratch.depth.handle(),
        count: node_count as u32,
        seam_arena_count: 0,
    };
    device.launch("lbvh.depth", &depth_args, node_count as u32)?;

    // A CLEAR, SO A MEMSET. `fill_zero` zeroes the counters on the device
    // rather than copying a page of zeros across the seam, and the counts are
    // read back after the kernel that fills them, so nothing reads the mirror in
    // between.
    {
        let handle = scratch.level_counts.handle();
        device
            .fill_zero(handle, LEVEL_CAPACITY * std::mem::size_of::<u32>())
            .map_err(|error| {
                Fatal::invariant(format!(
                    "solver driver: cannot clear the level counts: {error:?}"
                ))
            })?;
    }
    let depth = scratch.depth.handle();
    let counts = LbvhCountLevelsArgs {
        depth,
        counts: scratch.level_counts.handle(),
        capacity: LEVEL_CAPACITY as u32,
        count: node_count as u32,
        seam_arena_count: 0,
    };
    device.launch("lbvh.count_levels", &counts, node_count as u32)?;
    scratch.level_counts.download(device).map_err(|error| {
        Fatal::invariant(format!(
            "solver driver: cannot read back the level counts: {error:?}"
        ))
    })?;

    // THE OFFSETS ARE SCANNED ON THE HOST, over the LEVELS and not over the
    // nodes. `LEVEL_CAPACITY` is 64, so this is a fixed 64-step loop whatever
    // the scene's size.
    let host_counts = scratch.level_counts.host();
    let levels = host_counts
        .iter()
        .rposition(|&c| c != 0)
        .map_or(0, |last| last + 1);
    let mut level_offset = vec![0u32; levels + 1];
    for level in 0..levels {
        level_offset[level + 1] = level_offset[level] + host_counts[level];
    }
    if level_offset[levels] as usize != node_count {
        return Err(Fatal::invariant(format!(
            "solver driver: the level bins hold {} of the tree's {node_count} nodes, so a \
             node was counted into a level past the {LEVEL_CAPACITY} the bins carry and \
             the propagation would skip it",
            level_offset[levels]
        )));
    }

    let level_data = &mut tree.level_data;
    level_data
        .size(device, node_count, AllocLabel("lbvh.level_data"))
        .map_err(|error| {
            Fatal::out_of_memory(format!(
                "solver driver: cannot size the tree's levels: {error:?}"
            ))
        })?;
    scratch
        .level_offset_device
        .size(device, levels + 1, AllocLabel("lbvh.level_offset"))
        .and_then(|()| scratch.level_offset_device.write(device, 0, &level_offset))
        .and_then(|()| {
            scratch
                .level_cursor
                .size(device, LEVEL_CAPACITY, AllocLabel("lbvh.level_cursor"))
        })
        .map_err(|error| {
            Fatal::out_of_memory(format!(
                "solver driver: cannot stage the level offsets: {error:?}"
            ))
        })?;
    let zero_cursor = VecFillU32Args {
        array: scratch.level_cursor.handle(),
        value: 0,
        count: LEVEL_CAPACITY as u32,
        seam_arena_count: 0,
    };
    device.launch("lbvh.level_cursor", &zero_cursor, LEVEL_CAPACITY as u32)?;
    let scatter = LbvhScatterLevelsArgs {
        depth,
        level_offset: scratch.level_offset_device.span(0, levels + 1),
        cursor: scratch.level_cursor.handle(),
        level_data: level_data.handle(),
        count: node_count as u32,
        seam_arena_count: 0,
    };
    device.launch("lbvh.scatter_levels", &scatter, node_count as u32)?;
    Ok((level_offset, root))
}

/// Refresh the leaf boxes of a FACE tree against a swept motion.
/// The caller must propagate after applying any leaf masks.
///
/// `x0` and `x1` are the two position arrays as `Vec3f`, referenced as raw
/// coordinate triples: nothing on this side converts or rescales one.
/// `extrapolate` scales the motion, matching `lbvh::update_face_aabb`.
///
/// `params` is the scene's whole face parameter array. It is not indexed by the
/// leaf, so its reference carries the array's own length and not the face
/// count: the shared body reaches it through `prop[primitive].param_index`, and
/// that array is deduplicated across objects with identical materials.
///
/// # Safety
/// Every reference must name the array `DataSet` declares for it, alive and
/// unmoved for the call, and the tree must have been built over the same face
/// count.
#[allow(clippy::too_many_arguments)]
pub unsafe fn refresh_face_leaves<D: Device>(
    device: &mut D,
    tree: &mut Tree,
    x0: ppf_cts_compute::Handle,
    x1: ppf_cts_compute::Handle,
    extrapolate: f32,
    face: ppf_cts_compute::Handle,
    prop: ppf_cts_compute::Handle,
    params: ppf_cts_compute::Handle,
) -> FatalResult<()> {
    let n = tree.primitive_count();
    if n == 0 {
        return Ok(());
    }
    let args = AabbLeafFaceArgs {
        x0,
        x1,
        face,
        prop,
        params,
        nodes: tree.node.handle(),
        aabb: tree.aabb.span(0, tree.aabb.len()),
        extrapolate,
        count: n,
        seam_arena_count: 0,
    };
    device.launch("lbvh.leaf.face", &args, n)?;
    Ok(())
}

/// The EDGE tree's leaves. See [`refresh_face_leaves`].
///
/// # Safety
/// As [`refresh_face_leaves`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn refresh_edge_leaves<D: Device>(
    device: &mut D,
    tree: &mut Tree,
    x0: ppf_cts_compute::Handle,
    x1: ppf_cts_compute::Handle,
    extrapolate: f32,
    edge: ppf_cts_compute::Handle,
    prop: ppf_cts_compute::Handle,
    params: ppf_cts_compute::Handle,
) -> FatalResult<()> {
    let n = tree.primitive_count();
    if n == 0 {
        return Ok(());
    }
    let args = AabbLeafEdgeArgs {
        x0,
        x1,
        edge,
        prop,
        params,
        nodes: tree.node.handle(),
        aabb: tree.aabb.span(0, tree.aabb.len()),
        extrapolate,
        count: n,
        seam_arena_count: 0,
    };
    device.launch("lbvh.leaf.edge", &args, n)?;
    Ok(())
}

/// The VERTEX tree's leaves. See [`refresh_face_leaves`].
///
/// # Safety
/// As [`refresh_face_leaves`].
pub unsafe fn refresh_vertex_leaves<D: Device>(
    device: &mut D,
    tree: &mut Tree,
    x0: ppf_cts_compute::Handle,
    x1: ppf_cts_compute::Handle,
    extrapolate: f32,
    prop: ppf_cts_compute::Handle,
    params: ppf_cts_compute::Handle,
) -> FatalResult<()> {
    let n = tree.primitive_count();
    if n == 0 {
        return Ok(());
    }
    let args = AabbLeafVertexArgs {
        x0,
        x1,
        prop,
        params,
        nodes: tree.node.handle(),
        aabb: tree.aabb.span(0, tree.aabb.len()),
        extrapolate,
        count: n,
        seam_arena_count: 0,
    };
    device.launch("lbvh.leaf.vertex", &args, n)?;
    Ok(())
}

/// Re-merge every internal box from the leaves up.
///
/// Levels run deepest first, ONE DISPATCH PER LEVEL, because the sequence of
/// levels is the ordering that matters: a node's children must already carry
/// their merged boxes. Within a level the nodes are distinct and their children
/// sit strictly deeper, which is what makes the row `Scatter::Disjoint`.
///
/// # Safety
/// As [`refresh_face_leaves`].
pub unsafe fn propagate<D: Device>(device: &mut D, tree: &mut Tree) -> FatalResult<()> {
    let levels = tree.level_offset.len().saturating_sub(1);
    if levels <= 1 {
        return Ok(()); // one node, or leaves only: nothing above them to merge
    }
    // EVERY LEVEL IN ONE BOUNDARY. A level's merge reads the level below it, so
    // the passes must stay ORDERED, and consecutive entries of a region are
    // ordered with a full barrier between them on every backend, which is
    // exactly that. What a submit per level adds is a host STALL per level:
    // `Device::launch` is `run` around a single `elements` and a backend
    // synchronizes at the end of every submit, and `PPF_REGION_STATS` measured
    // `lbvh.propagate` at 7,977 submits on ten frames of `drape`, 21.5 percent
    // of every synchronize in the run.
    //
    // `level_offset` IS A HOST ARRAY, not a readback, so nothing in this loop
    // asks the device anything: the bounds come from the level scan that ran
    // before it.
    device.run("lbvh.propagate", |encoder| {
        for level in (0..levels).rev() {
            let begin = tree.level_offset[level] as usize;
            let end = tree.level_offset[level + 1] as usize;
            let size = end - begin;
            if size == 0 {
                continue;
            }
            let args = AabbMergeLevelArgs {
                level: tree.level_data.span(begin, size),
                nodes: tree.node.handle(),
                aabb: tree.aabb.span(0, tree.aabb.len()),
                count: size as u32,
                seam_arena_count: 0,
            };
            // Safety: the tree outlives this region, and every span above is
            // taken from arrays it owns.
            unsafe { encoder.elements(&args, size as u32)? };
        }
        Ok(())
    })?;
    Ok(())
}

/// Every (query, primitive) pair whose boxes overlap, in ascending query order.
///
/// The result is a flat list of two `u32` per pair. It is the CANDIDATE set,
/// not the contact set: no pair filter has been applied, because which filter
/// applies depends on which of the four contact types is being walked and those
/// rules belong beside the reasoning that justifies them.
///
/// # Growing rather than guessing
///
/// The buffer starts at an estimate and the walk reports how many pairs it
/// FOUND, whether or not they fit. That is the same discipline the contact pair
/// cache follows, and for the same reason: a fixed cap that is quietly crossed
/// costs an order of magnitude in traversal time with nothing in the log to say
/// why. Here it would cost correctness instead, since the pairs past the cap are
/// simply absent, so an overflow re-walks with the measured requirement and says
/// what it cost.
///
/// No PRODUCTION caller: the contact and intersection passes fuse the walk into
/// their own kernels (`intersect_scan_*`, `aabb_*_scan_query`), which visit each
/// hit rather than materializing a candidate list. Six tests here read it, from
/// `an_empty_input_builds_an_empty_tree_rather_than_no_tree` through
/// `a_walk_over_a_corrupt_tree_faults_rather_than_returning_a_short_list`, and
/// two in `driver/intersection.rs` read it as a scan's NEGATIVE CONTROL,
/// `a_face_and_an_edge_sharing_a_vertex_are_never_reported` and
/// `edge_edge_reports_each_unordered_pair_once`: each requires the pair to be a
/// candidate here, so the clean report they assert comes from the rule under
/// test rather than from an empty candidate set.
#[allow(dead_code)]
pub fn query_pairs<D: Device>(
    device: &mut D,
    tree: &mut Tree,
    query: ppf_cts_compute::Handle,
    queries: usize,
    scratch: &mut WalkScratch,
) -> Result<Vec<u32>, TraversalFault> {
    // Four pairs per query is the estimate a caller with no history has.
    let mut capacity = DEFAULT_PAIRS_PER_QUERY;
    query_pairs_sized(device, tree, query, queries, scratch, &mut capacity)
}

/// The estimate a caller with no measurement of its own starts from.
///
/// No PRODUCTION caller: [`query_pairs`] and [`query_pairs_sized`] are its only
/// readers and tests alone reach them. What prices it is
/// `an_overflowing_buffer_is_grown_rather_than_truncated`, whose sixty
/// coincident boxes are sixty pairs per query against this four, so the walk
/// must grow rather than return what fit.
#[allow(dead_code)]
pub const DEFAULT_PAIRS_PER_QUERY: usize = 4;

/// [`query_pairs`], with the caller keeping the measured capacity.
///
/// `capacity_per_query` is read as the starting estimate and written back with
/// whatever the walk needed, so a caller that holds it across steps pays one
/// overflow rather than one per step. That is the same discipline the contact
/// pair cache follows: a fixed cap quietly crossed costs an order of magnitude
/// in traversal time with nothing in the log to say why, and a capacity reset
/// on every call reports the growth every time it is paid.
///
/// No PRODUCTION caller: [`query_pairs`] is its only caller and tests alone
/// reach that. The growth it implements is what
/// `an_overflowing_buffer_is_grown_rather_than_truncated` requires: sixty boxes
/// on top of each other against a starting estimate of four must still return
/// every one of the n^2 pairs.
#[allow(dead_code)]
pub fn query_pairs_sized<D: Device>(
    device: &mut D,
    tree: &mut Tree,
    query: ppf_cts_compute::Handle,
    queries: usize,
    scratch: &mut WalkScratch,
    capacity_per_query: &mut usize,
) -> Result<Vec<u32>, TraversalFault> {
    if tree.is_empty() || queries == 0 {
        return Ok(Vec::new());
    }
    if *capacity_per_query == 0 {
        *capacity_per_query = DEFAULT_PAIRS_PER_QUERY;
    }
    loop {
        match walk(device, tree, query, queries, scratch, *capacity_per_query)? {
            Walk::Complete(pairs) => return Ok(pairs),
            Walk::Overflow { found, chunk_span } => {
                let needed = found.div_ceil(chunk_span.max(1));
                let grown = (needed * 2).max(*capacity_per_query * 2);
                log::info!(
                    "cpu contact broad phase: the candidate buffer held {} pairs \
                     per query and the walk found {found} over {chunk_span} \
                     queries, so the tree was walked again at {grown} per query. \
                     A repeated report here is a scene whose candidate density \
                     the estimate does not fit.",
                    *capacity_per_query
                );
                *capacity_per_query = grown;
            }
        }
    }
}

/// What one pass of [`walk`] produced: the pairs, or the requirement that did
/// not fit.
///
/// No PRODUCTION caller: [`walk`] and [`query_pairs_sized`] are its only
/// readers and tests alone reach them. The `Overflow` arm is what
/// `an_overflowing_buffer_is_grown_rather_than_truncated` exercises, requiring
/// the walk to report what it found rather than return the pairs that fit.
#[allow(dead_code)]
enum Walk {
    Complete(Vec<u32>),
    Overflow { found: usize, chunk_span: usize },
}

/// ONE DISPATCH PER WALK, OVER AN OUTPUT THAT IS NOT INDEXED BY ITS THREAD, AND
/// THE LAYOUT THAT SETTLES THAT.
///
/// A query finds however many pairs it finds, so the shape that comes first to
/// mind packs the pairs as they are discovered and returns how many were
/// packed, and those are two things the seam does not carry: a per-partition
/// return value, and a per-partition output buffer whose extent the driver has
/// to know in order to concatenate the results. `Extent::Elements` deliberately
/// hides the partition, since the chunk width is the backend's own.
///
/// WHAT MAKES THIS A `Device::launch` LIKE EVERY OTHER PASS HERE is the
/// per-query slot layout written at the dispatch below: query `i` owns
/// `out[2 * i * capacity .. 2 * (i + 1) * capacity]` and no other thread writes
/// it, so the scatter is `Disjoint`, the pair list is the same whatever cut the
/// backend takes, and neither a per-partition return value nor an ordered claim
/// is needed. What it costs is a buffer of `queries * capacity` rather than
/// `queries * mean density`, and those two differ by orders of magnitude on a
/// scene with one dense region, which is why the capacity is measured and grown
/// by [`query_pairs_sized`] rather than fixed.
///
/// Two other shapes, each weighed against what it would cost, and neither taken:
///
/// - **Claim slots from a shared counter**, which is what `pair_cache_record`
///   does here. That is
///   `Scatter::Claim`, and a Claim through this seam is ONE SERIAL ASCENDING
///   PASS, because a slot assignment is reproducible only in ascending order.
///   The broad phase is the largest parallel region in a contact step, so that
///   trades a correct answer's speed for a smaller buffer.
/// - **Extend the trait with a partitioned dispatch**, which would hand the
///   driver the cut and take a record per part. A GPU backend does not cut a
///   range that way at all, so the primitive would exist for one backend, and
///   a surface only one backend can implement is the fork this seam exists to
///   prevent.
///
/// No PRODUCTION caller: [`query_pairs_sized`] is its only caller, and the
/// contact and intersection passes fuse the walk into their own kernels
/// (`intersect_scan_*`, `aabb_*_scan_query`), which visit each hit rather than
/// materializing a candidate list. Eight tests reach it through
/// [`query_pairs`], among them `a_traversal_finds_exactly_the_overlapping_pairs`,
/// which checks the walk against brute force, and
/// `an_overflowing_buffer_is_grown_rather_than_truncated`, which requires the
/// overflow arm here to report the requirement rather than return what fit.
#[allow(dead_code)]
fn walk<D: Device>(
    device: &mut D,
    tree: &mut Tree,
    query: ppf_cts_compute::Handle,
    queries: usize,
    scratch: &mut WalkScratch,
    capacity_per_query: usize,
) -> Result<Walk, TraversalFault> {
    // ONE DISPATCH, one thread per query box. The partition is the BACKEND'S:
    // the row is declared in `super::kernels` and `sched::chunk_for` inside the
    // backend turns its nanoseconds-per-item into a chunk width, exactly as
    // `fixed_csr_apply_row`'s does. Restating that policy here beside a
    // hand-rolled rayon partition would be a second copy of it.
    //
    // AND THE LAYOUT REMOVES THE ORDERED-CLAIM PROBLEM. A collector that packed
    // sequentially against a count shared across a chunk would need either a
    // serial pass or an ordered claim for a reproducible pair list.
    // A per-query slot needs neither: query `i` owns
    // `out[2 * i * capacity .. 2 * (i + 1) * capacity]` and no other thread
    // writes it, so the layout is the same whatever the cut is.
    scratch
        .out
        .size(device, 2 * queries * capacity_per_query, AllocLabel("lbvh.walk_out"))
        .and_then(|()| scratch.found.size(device, queries, AllocLabel("lbvh.walk_found")))
        .map_err(TraversalFault::transport)?;
    let args = super::kernels::AabbQueryPairsArgs {
        node: tree.node.handle(),
        node_count: tree.node_count,
        aabb: tree.aabb.span(0, tree.aabb.len()),
        root: tree.root,
        query,
        out: scratch.out.handle(),
        found: scratch.found.handle(),
        capacity: capacity_per_query as u32,
        count: queries as u32,
        seam_arena_count: 0,
    };
    // THE INVARIANT CHANNEL IS READ BY `launch` ITSELF, which is what replaces
    // the explicit merge the per-chunk shim needed: a clean boundary returns
    // its `Diag` and a boundary carrying a failure returns `Fault::Device`, so
    // a traversal that abandoned subtrees cannot be read as a result. That is
    // the silent penetration path this check exists to close, and it is now
    // structural rather than a call the caller must remember.
    // Safety: every handle names a live allocation and the count is the query
    // buffer's own length.
    unsafe { device.launch("lbvh.query_pairs", &args, queries as u32) }.map_err(|fault| {
        // THE DEVICE'S OWN RECORD, not a synthetic one. `Fault::Device` carries
        // the `Diag` the traversal wrote, and that record names the FILE and
        // LINE of the failing check inside the shared body. Replacing it with a
        // placeholder would report that a walk faulted without saying where,
        // which is the whole value of the channel.
        match fault {
            // ALREADY MERGED. `launch` collects the boundary's records and
            // hands back one `Diag`, so this reads that record directly.
            Fault::Device { diag, .. } => match diag.first {
                Some(first) => TraversalFault {
                    fail_count: diag.failures as u32,
                    payload: first.payload,
                    file: first.file,
                    line: first.line,
                    transport: None,
                },
                None => TraversalFault::transport(
                    "the device reported failing checks but no record; \
                     the diagnostic channel lost the first one",
                ),
            },
            other => TraversalFault::transport(other),
        }
    })?;
    scratch.found.download(device).map_err(TraversalFault::transport)?;
    let found = scratch.found.host();
    let mut total_found = 0usize;
    let mut overflowed = false;
    for count in found.iter().take(queries) {
        total_found += *count as usize;
        if *count as usize > capacity_per_query {
            overflowed = true;
        }
    }
    if overflowed {
        return Ok(Walk::Overflow {
            found: total_found,
            chunk_span: queries,
        });
    }
    scratch.out.download(device).map_err(TraversalFault::transport)?;
    // COMPACTED IN ASCENDING QUERY ORDER, which the per-query layout makes a
    // plain copy rather than a claim.
    let out = scratch.out.host();
    let mut pairs = Vec::with_capacity(2 * total_found);
    for (i, count) in found.iter().enumerate().take(queries) {
        let base = 2 * i * capacity_per_query;
        pairs.extend_from_slice(&out[base..base + 2 * (*count as usize)]);
    }
    Ok(Walk::Complete(pairs))
}

/// Build a tree over a point cloud, for a caller that has centroids rather than
/// a `DataSet`.
///
/// The three production trees take their centroids from the geometry they
/// index; this is the same pipeline with the centroids supplied, which is what
/// a test needs to build a tree of a known shape.
///
/// No PRODUCTION caller: the three per-step trees take their centroids from the
/// geometry they index and reach [`build_presorted`] through
/// `contact::build_tree`, and the collider's rest-pose tree takes the same
/// path. What reads this is `tree_from_centroids` in this module's tests, which
/// backs the leaf, level, traversal and stack-depth checks, and `Fixture::tree`
/// in `driver/intersection.rs`'s tests, which builds a tree over known boxes
/// for the intersection scans.
///
/// # Safety
/// As [`build`].
#[allow(dead_code)]
pub unsafe fn build_from_centroids<D: Device>(
    device: &mut D,
    scratch: &mut BuildScratch,
    cx: &[f32],
    cy: &[f32],
    cz: &[f32],
) -> FatalResult<Tree> {
    // THE CALLER HAS HOST SLICES, which is what this entry point is for: a
    // fixture holds centroids rather than device buffers. They are staged here
    // and the pipeline below is the same one `contact::build_tree` runs against
    // buffers it already owns.
    let n = cx.len();
    let mut dx = ReadbackBuffer::<f32>::default();
    let mut dy = ReadbackBuffer::<f32>::default();
    let mut dz = ReadbackBuffer::<f32>::default();
    let mut codes = ReadbackBuffer::<u32>::default();
    // SIZED BEFORE SEEDED: `seed` fills both halves and asserts the buffer
    // already carries the length it is handed.
    dx.size(device, n, AllocLabel("lbvh.centroid_x"))
        .and_then(|()| dy.size(device, n, AllocLabel("lbvh.centroid_y")))
        .and_then(|()| dz.size(device, n, AllocLabel("lbvh.centroid_z")))
        .and_then(|()| dx.seed(device, cx))
        .and_then(|()| dy.seed(device, cy))
        .and_then(|()| dz.seed(device, cz))
        .map_err(|error| {
            Fatal::out_of_memory(format!("solver driver: cannot stage the centroids: {error:?}"))
        })?;
    // ON THE DEVICE, like the per-step path. This entry point takes its
    // centroids as host slices and stages them just above, so folding them here
    // would be free; it reduces them on the device anyway, because the Morton
    // pass below reads the bounds as a device ARRAY rather than as six host
    // scalars and a host fold would have to upload them again.
    let bounds = scratch
        .bounds
        .reduce(device, dx.span(0, n), dy.span(0, n), dz.span(0, n), n as u32)?;
    bvh::morton_codes(
        device,
        dx.span(0, n),
        dy.span(0, n),
        dz.span(0, n),
        &mut codes,
        n,
        bounds,
    )?;
    // THE SORT IS THE DEVICE ONE, the same network the per-step build uses.
    // This path runs once per fixture, so a host sort would be affordable, but
    // it would put a second ordering of the same codes in the tree and the two
    // would have to be kept in step; with `devsort::SortScratch` sharing the
    // device one is a call rather than a design.
    let (sorted, order) =
        scratch
            .sort
            .sort(device, "lbvh.rest_sort", codes.handle(), n)
            .map_err(|error| {
                Fatal::out_of_memory(format!(
                    "solver driver: cannot sort the rest-pose Morton codes: {error:?}"
                ))
            })?;
    // A FRESH `Tree` IS CORRECT HERE and is not the leak `build` was changed to
    // avoid, for the same reason.
    let mut tree = Tree::default();
    build_presorted(device, scratch, sorted, order, n, &mut tree)?;
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::super::launch::{host_device, HostDevice};
    use super::*;

    /// A backend for one test.
    ///
    /// Per call rather than shared: every pass here is stateless on the device
    /// (this driver's buffers are still its own `Vec`s, which is what `HostRef`
    /// names), so a fresh one carries nothing between tests and cannot make one
    /// test's result depend on another's.
    fn device() -> HostDevice {
        host_device()
    }

    /// [`build`], for a test that has the sorted codes and the order.
    /// ONE DEVICE PER FIXTURE, and the reason is the tree's own arrays: both
    /// are device allocations now, so a tree built on a throwaway `device()`
    /// names an arena nothing else can reach and every later handle resolves
    /// somewhere unrelated.
    fn tree_of(device: &mut HostDevice, sorted_codes: &[u32], order: &[u32]) -> Tree {
        let mut tree = Tree::default();
        unsafe { build(device, &mut BuildScratch::default(), sorted_codes, order, &mut tree) }
            .expect("the tree builds");
        tree
    }

    /// [`build_from_centroids`], for a test that has a point cloud.
    /// As [`tree_of`], for a fixture that has a point cloud.
    fn tree_from_centroids(
        device: &mut HostDevice,
        cx: &[f32],
        cy: &[f32],
        cz: &[f32],
    ) -> Tree {
        unsafe { build_from_centroids(device, &mut BuildScratch::default(), cx, cy, cz) }.expect("the tree builds")
    }

    /// One box list on the device, for a query that takes a handle.
    ///
    /// The buffer is the caller's so it outlives the dispatch, and a fixture
    /// keeps one and reuses it, which is what the contact state does.
    fn boxes_on_device(
        device: &mut HostDevice,
        boxes: &[Aabb],
        buffer: &mut Buffer<Aabb>,
    ) -> ppf_cts_compute::Handle {
        buffer
            .size(device, boxes.len(), AllocLabel("test.query"))
            .and_then(|()| buffer.write(device, 0, boxes))
            .expect("the fixture stages its query boxes");
        buffer.span(0, boxes.len())
    }

    /// A leaf's primitive, or `None` if the node is internal.
    ///
    /// TAKES THE READ-BACK COPY. Production reads the tree's nodes on the host
    /// nowhere; a fixture that asserts on the tree's shape brings them across
    /// with [`tree_host`] and reads that.
    fn leaf_primitive(host: &TreeHost, node: usize) -> Option<u32> {
        if host.node[2 * node + 1] == 0 {
            Some(host.node[2 * node] - 1)
        } else {
            None
        }
    }

    fn cloud(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut cx = Vec::with_capacity(n);
        let mut cy = Vec::with_capacity(n);
        let mut cz = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 * 0.013;
            cx.push(t.sin() * 3.0 + 0.5);
            cy.push((t * 1.7).cos() * 2.0 - 1.25);
            cz.push((t * 0.37).sin() * 4.0);
        }
        (cx, cy, cz)
    }

    #[test]
    fn the_aabb_mirror_matches_cpp() {
        // `AABB` is device-only, so nothing else compares these two layouts and
        // a drift is a silent wrong answer rather than a link error.
        assert_eq!(
            std::mem::size_of::<Aabb>(),
            unsafe { aabb_sizeof_abi() } as usize,
            "the Rust mirror of AABB is a different size from the C++ struct"
        );
        assert_eq!(
            std::mem::offset_of!(Aabb, min),
            unsafe { aabb_offset_min_abi() } as usize
        );
        assert_eq!(
            std::mem::offset_of!(Aabb, max),
            unsafe { aabb_offset_max_abi() } as usize
        );
        assert_eq!(
            std::mem::offset_of!(Aabb, active),
            unsafe { aabb_offset_active_abi() } as usize
        );
        assert_eq!(
            std::mem::size_of::<Diag>(),
            unsafe { diag_sizeof_abi() } as usize,
            "the Rust mirror of the diagnostic record disagrees with C++, so a \
             traversal fault would be read out of the wrong bytes"
        );
    }

    #[test]
    fn a_default_box_takes_part_in_nothing() {
        // A zeroed box with `active` set is a box AT THE ORIGIN, which overlaps
        // whatever is there. The default must be inactive or an unrefreshed
        // leaf silently joins the broad phase.
        let a = Aabb::default();
        let b = Aabb::default();
        assert!(!overlap(&a, &b));
    }

    #[test]
    fn an_empty_input_builds_an_empty_tree_rather_than_no_tree() {
        // `lbvh::initialize` sizes all three trees off the largest primitive
        // count for exactly this reason: a scene class with none of one kind
        // must still hand the traversal a valid handle.
        let mut device = device();
        let mut tree = tree_of(&mut device, &[], &[]);
        assert!(tree.is_empty());
        assert_eq!(tree.primitive_count(), 0);
        let mut scratch = WalkScratch::default();
        let mut qbuf = Buffer::<Aabb>::none();
        let boxes = [Aabb::default()];
        let query = boxes_on_device(&mut device, &boxes, &mut qbuf);
        assert!(
            query_pairs(&mut device, &mut tree, query, boxes.len(), &mut scratch)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_single_primitive_builds_a_single_leaf() {
        let mut device = device();
        let mut tree = tree_of(&mut device, &[7], &[0]);
        assert_eq!(tree.node_count, 1);
        assert_eq!(tree.root, 0);
        let host = tree_host(&mut device, &mut tree);
        assert_eq!(leaf_primitive(&host, 0), Some(0));
        assert_eq!(tree.primitive_count(), 1);
    }

    #[test]
    fn every_primitive_appears_in_exactly_one_leaf() {
        // The property that makes the tree a partition: a primitive in no leaf
        // is a primitive that can never be found by a query, and one in two
        // leaves is a contact assembled twice.
        let mut device = device();
        for n in [2usize, 3, 5, 64, 1000, 4097] {
            let (cx, cy, cz) = cloud(n);
            let mut tree = tree_from_centroids(&mut device, &cx, &cy, &cz);
            let host = tree_host(&mut device, &mut tree);
            let mut seen = vec![0u32; n];
            for node in 0..tree.node_count as usize {
                if let Some(primitive) = leaf_primitive(&host, node) {
                    seen[primitive as usize] += 1;
                }
            }
            assert!(
                seen.iter().all(|c| *c == 1),
                "at n = {n} the leaves do not partition the primitives: {:?}",
                seen.iter()
                    .enumerate()
                    .filter(|(_, c)| **c != 1)
                    .take(5)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn the_tree_is_connected_and_acyclic_from_its_reported_root() {
        // Reached exactly once each, from the root the query seeds its stack
        // from. A node reachable twice would be traversed twice; one reachable
        // not at all is a subtree the broad phase silently never visits.
        let mut device = device();
        for n in [2usize, 3, 17, 500, 2048] {
            let (cx, cy, cz) = cloud(n);
            let mut tree = tree_from_centroids(&mut device, &cx, &cy, &cz);
            let host = tree_host(&mut device, &mut tree);
            let count = tree.node_count as usize;
            let mut visits = vec![0u32; count];
            let mut stack = vec![tree.root as usize];
            let mut steps = 0usize;
            while let Some(node) = stack.pop() {
                steps += 1;
                assert!(
                    steps <= 4 * count,
                    "the walk from the root did not terminate at n = {n}"
                );
                visits[node] += 1;
                if host.node[2 * node + 1] != 0 {
                    stack.push((host.node[2 * node] - 1) as usize);
                    stack.push((host.node[2 * node + 1] - 1) as usize);
                }
            }
            assert!(
                visits.iter().all(|v| *v == 1),
                "at n = {n} the tree is not a tree: {} nodes were reached a \
                 number of times other than once",
                visits.iter().filter(|v| **v != 1).count()
            );
        }
    }

    #[test]
    fn the_levels_group_every_node_by_its_distance_from_the_root() {
        let mut device = device();
        for n in [2usize, 9, 333] {
            let (cx, cy, cz) = cloud(n);
            let mut tree = tree_from_centroids(&mut device, &cx, &cy, &cz);
            let host = tree_host(&mut device, &mut tree);
            let count = tree.node_count as usize;
            assert_eq!(host.level_data.len(), count, "a node is missing a level");
            let levels = tree.level_offset.len() - 1;
            // Depth 0 is the root alone.
            assert_eq!(tree.level_offset[1] - tree.level_offset[0], 1);
            assert_eq!(host.level_data[0], tree.root);
            // And a node's own depth, recomputed here from the parent links the
            // tree does not keep, must match the level it was filed under.
            let mut depth_of = vec![u32::MAX; count];
            depth_of[tree.root as usize] = 0;
            let mut stack = vec![tree.root as usize];
            while let Some(node) = stack.pop() {
                if host.node[2 * node + 1] == 0 {
                    continue;
                }
                for child in [host.node[2 * node] - 1, host.node[2 * node + 1] - 1] {
                    depth_of[child as usize] = depth_of[node] + 1;
                    stack.push(child as usize);
                }
            }
            for level in 0..levels {
                let begin = tree.level_offset[level] as usize;
                let end = tree.level_offset[level + 1] as usize;
                for node in &host.level_data[begin..end] {
                    assert_eq!(
                        depth_of[*node as usize], level as u32,
                        "node {node} is at depth {} and was filed at level {level}",
                        depth_of[*node as usize]
                    );
                }
                // Ascending within a level, which is what makes the grouping
                // reproducible where the device's atomic scatter is not.
                assert!(
                    host.level_data[begin..end].windows(2).all(|w| w[0] < w[1]),
                    "level {level} is not in ascending node order"
                );
            }
        }
    }

    #[test]
    fn the_tree_does_not_depend_on_the_thread_count() {
        let mut device = device();
        let (cx, cy, cz) = cloud(20_000);
        let mut reference = tree_from_centroids(&mut device, &cx, &cy, &cz);
        // COMPARED AS READ-BACK COPIES: the nodes are device-resident, so two
        // trees hold two arena spans rather than two `Vec`s, and what the test
        // is about is the CONTENTS being identical.
        let reference_host = tree_host(&mut device, &mut reference);
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let mut got = pool.install(|| tree_from_centroids(&mut device, &cx, &cy, &cz));
            let got_host = tree_host(&mut device, &mut got);
            assert_eq!(reference_host.node, got_host.node, "the tree moved at {threads} threads");
            assert_eq!(reference.root, got.root);
            assert_eq!(reference.level_offset, got.level_offset);
            assert_eq!(reference_host.level_data, got_host.level_data);
        }
    }

    /// Leaf boxes for a test, written directly rather than through a `DataSet`.
    ///
    /// The production path fills leaves from geometry through the shared body;
    /// a traversal test needs boxes of a known shape, and the traversal cannot
    /// tell where a box came from.
    /// Seed the leaf boxes and merge them upward.
    ///
    /// THE TREE IS DEVICE-RESIDENT, so this reads the nodes back to find which
    /// slots are leaves, builds the box array on the host, and writes it. That
    /// is a fixture's business: production writes those boxes from a `refresh_*`
    /// kernel and never assembles them here.
    fn set_leaf_boxes(device: &mut HostDevice, tree: &mut Tree, boxes: &[Aabb]) {
        let host = tree_host(device, tree);
        let mut aabb = host.aabb.clone();
        for node in 0..tree.node_count as usize {
            if host.node[2 * node + 1] == 0 {
                let primitive = (host.node[2 * node] - 1) as usize;
                aabb[node] = boxes[primitive];
            }
        }
        tree.aabb
            .write(device, 0, &aabb)
            .expect("the fixture seeds the leaf boxes");
        unsafe { propagate(device, tree) }.expect("the boxes merge");
    }

    #[test]
    fn masked_leaves_need_only_the_final_propagation() {
        use crate::driver::kernels::AabbLeafActiveArgs;
        let mut device = device();
        let mut tree = tree_of(&mut device, &[0, 1, 2, 3], &[0, 1, 2, 3]);
        let nodes = tree_host(&mut device, &mut tree).node;
        let mut active = Buffer::<u32>::default();
        active.size(&mut device, 4, AllocLabel("test.leaf_mask")).unwrap();
        for mask in [[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1]] {
            active.write(&mut device, 0, &mask).unwrap();
            let mut seed = tree_host(&mut device, &mut tree).aabb;
            for node in 0..tree.node_count as usize {
                if nodes[2 * node + 1] == 0 {
                    let primitive = nodes[2 * node] - 1;
                    seed[node] = box_about([primitive as f32 * 100.0, 0.0, 0.0], 20.0);
                }
            }
            let mut expected = None;
            for preliminary_merge in [true, false] {
                tree.aabb.write(&mut device, 0, &seed).unwrap();
                if preliminary_merge {
                    unsafe { propagate(&mut device, &mut tree) }.unwrap();
                }
                let args = AabbLeafActiveArgs {
                    active: active.handle(),
                    nodes: tree.node.handle(),
                    aabb: tree.aabb.handle(),
                    count: 4,
                    seam_arena_count: 0,
                };
                // Safety: one mask per primitive and the complete tree storage.
                unsafe {
                    device.launch("test.leaf_mask", &args, 4).unwrap();
                    propagate(&mut device, &mut tree).unwrap();
                }
                let actual = tree_host(&mut device, &mut tree).aabb;
                if let Some(expected) = &expected {
                    assert_eq!(&actual, expected, "mask {mask:?}");
                } else {
                    expected = Some(actual);
                }
            }
        }
    }

    #[test]
    #[ignore = "release timing experiment"]
    fn benchmark_contact_masked_propagation() {
        use crate::driver::kernels::AabbLeafActiveArgs;
        for count in [2048usize, 16384] {
            let mut device = device();
            let ids: Vec<u32> = (0..count as u32).collect();
            let mut tree = tree_of(&mut device, &ids, &ids);
            let host = tree_host(&mut device, &mut tree);
            let mut seed = host.aabb;
            for node in 0..tree.node_count as usize {
                if host.node[2 * node + 1] == 0 {
                    seed[node] = box_about([host.node[2 * node] as f32 * 100.0, 0.0, 0.0], 20.0);
                }
            }
            let mut active = Buffer::<u32>::default();
            active.size(&mut device, count, AllocLabel("test.mask")).unwrap();
            active.write(&mut device, 0, &ids.iter().map(|&i| i % 2).collect::<Vec<_>>()).unwrap();
            let mut samples = [Vec::new(), Vec::new()];
            for repetition in 0..10 {
                for mode in [repetition % 2, 1 - repetition % 2] {
                    tree.aabb.write(&mut device, 0, &seed).unwrap();
                    let args = AabbLeafActiveArgs {
                        active: active.handle(), nodes: tree.node.handle(),
                        aabb: tree.aabb.handle(), count: count as u32, seam_arena_count: 0,
                    };
                    let start = std::time::Instant::now();
                    // Safety: all buffers are sized over the tree's primitives.
                    unsafe {
                        if mode == 0 { propagate(&mut device, &mut tree).unwrap(); }
                        device.launch("test.mask", &args, count as u32).unwrap();
                        propagate(&mut device, &mut tree).unwrap();
                    }
                    if repetition > 1 {
                        samples[mode].push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            }
            for (mode, times) in samples.iter_mut().enumerate() {
                times.sort_by(f64::total_cmp);
                eprintln!("masked leaves={count} merges={} median_ms={:.6} min_ms={:.6} max_ms={:.6}",
                          if mode == 0 { 2 } else { 1 }, times[4], times[0], times[7]);
            }
        }
    }

    /// A cube of side `2 * r` about `center`.
    ///
    /// The test builds boxes directly in the coordinate units the shared bodies
    /// store, so nothing here converts.
    fn box_about(center: [f32; 3], r: f32) -> Aabb {
        Aabb {
            min: [center[0] - r, center[1] - r, center[2] - r],
            max: [center[0] + r, center[1] + r, center[2] + r],
            active: true,
        }
    }

    #[test]
    fn the_leaf_slots_are_a_permutation_of_the_primitives() {
        // THE MORTON REMAP RESTS ENTIRELY ON THIS. The CCD line search's three
        // self-contact sweeps open with `i = node[2 * thread] - 1`
        // (`contact/ccd_sweep.kernel.cpp`), a query-to-thread permutation:
        // adjacent lanes get spatially adjacent queries, and every query still
        // reads its own primitive and writes its own output slot.
        // `aabb_leaf_active` relies on the same layout.
        //
        // IF IT EVER STOPPED HOLDING, the remap would query one primitive's box
        // and write ANOTHER's slot, with nothing to say so: the answer would
        // still be a float in range and every gate would stay green. That is
        // why this is a test rather than a comment, and it is the only check
        // that would catch it.
        let mut device = device();
        for n in [1usize, 2, 3, 7, 64, 1000] {
            let (cx, cy, cz) = cloud(n);
            let mut tree = tree_from_centroids(&mut device, &cx, &cy, &cz);
            assert_eq!(tree.primitive_count() as usize, n);
            let host = tree_host(&mut device, &mut tree);
            let mut seen = vec![false; n];
            for leaf in 0..n {
                let entry = host.node[2 * leaf];
                assert!(
                    entry > 0,
                    "leaf slot {leaf} of {n} holds {entry}, which is not a \
                     biased primitive index, so the remap would read a node \
                     that is not a leaf"
                );
                let primitive = (entry - 1) as usize;
                assert!(
                    primitive < n,
                    "leaf slot {leaf} of {n} names primitive {primitive}, \
                     outside the index space"
                );
                assert!(
                    !seen[primitive],
                    "primitive {primitive} is named by two leaf slots at \
                     n = {n}, so the remap is not a permutation and two \
                     threads would write one output slot"
                );
                seen[primitive] = true;
            }
            assert!(
                seen.iter().all(|hit| *hit),
                "some primitive is named by no leaf slot at n = {n}, so its \
                 sweep would never be dispatched"
            );
        }
    }

    #[test]
    fn a_traversal_finds_exactly_the_overlapping_pairs() {
        // The claim the broad phase rests on: the tree walk returns the same set
        // brute force does. Checked against the SHARED overlap predicate, not
        // against a second implementation of it.
        let n = 400usize;
        let mut boxes = Vec::with_capacity(n);
        let mut cx = Vec::new();
        let mut cy = Vec::new();
        let mut cz = Vec::new();
        for i in 0..n {
            let t = i as f32 * 0.11;
            let c = [
                t.sin() * 40_000.0,
                (t * 1.3).cos() * 40_000.0,
                (t * 0.7).sin() * 40_000.0,
            ];
            boxes.push(box_about(c, 3_000.0));
            cx.push(c[0]);
            cy.push(c[1]);
            cz.push(c[2]);
        }
        let mut device = device();
        let mut tree = tree_from_centroids(&mut device, &cx, &cy, &cz);
        set_leaf_boxes(&mut device, &mut tree, &boxes);

        // Query with the same boxes, so every primitive at least finds itself.
        let mut scratch = WalkScratch::default();
        let mut qbuf = Buffer::<Aabb>::none();
        let query = boxes_on_device(&mut device, &boxes, &mut qbuf);
        let pairs = query_pairs(&mut device, &mut tree, query, boxes.len(), &mut scratch)
            .expect("the traversal must not fault");
        let mut found: Vec<(u32, u32)> = pairs.chunks(2).map(|p| (p[0], p[1])).collect();
        found.sort_unstable();

        let mut expected: Vec<(u32, u32)> = Vec::new();
        for q in 0..n {
            for p in 0..n {
                if overlap(&boxes[p], &boxes[q]) {
                    expected.push((q as u32, p as u32));
                }
            }
        }
        expected.sort_unstable();
        assert_eq!(
            found, expected,
            "the tree walk and brute force disagree; a MISSING pair here is a \
             candidate contact the solver would never see"
        );
        assert!(
            expected.len() > n,
            "the fixture produced only the self-pairs, so it does not \
             discriminate; make the boxes overlap"
        );
    }

    #[test]
    fn an_inactive_query_finds_nothing() {
        // The collision-window path clears `active` on a query, and the walk
        // must then skip it entirely rather than treat the box as valid.
        let boxes = vec![box_about([0.0, 0.0, 0.0], 1000.0), box_about([500.0, 0.0, 0.0], 1000.0)];
        let cx = vec![0.0f32, 500.0];
        let cy = vec![0.0f32, 0.0];
        let cz = vec![0.0f32, 0.0];
        let mut device = device();
        let mut tree = tree_from_centroids(&mut device, &cx, &cy, &cz);
        set_leaf_boxes(&mut device, &mut tree, &boxes);
        let mut scratch = WalkScratch::default();
        let mut qbuf = Buffer::<Aabb>::none();
        let staged_boxes = boxes_on_device(&mut device, &boxes, &mut qbuf);

        let mut query = boxes.clone();
        let active = boxes_on_device(&mut device, &query, &mut qbuf);
        assert!(
            !query_pairs(&mut device, &mut tree, active, query.len(), &mut scratch)
                .unwrap()
                .is_empty()
        );
        query[0].active = false;
        query[1].active = false;
        // RE-STAGED, because the fixture changed the boxes: the device copy is
        // the one the walk reads and it does not follow the host `Vec`.
        let inactive = boxes_on_device(&mut device, &query, &mut qbuf);
        assert!(
            query_pairs(&mut device, &mut tree, inactive, query.len(), &mut scratch)
                .unwrap()
                .is_empty(),
            "an inactive query box still produced pairs"
        );
    }

    #[test]
    fn the_pair_list_does_not_depend_on_the_thread_count() {
        let n = 3000usize;
        let mut boxes = Vec::with_capacity(n);
        let (cx, cy, cz) = cloud(n);
        for i in 0..n {
            let c = [
                cx[i] * 1_000_000.0,
                cy[i] * 1_000_000.0,
                cz[i] * 1_000_000.0,
            ];
            boxes.push(box_about(c, 40_000.0));
        }
        let mut device = device();
        let mut tree = tree_from_centroids(&mut device, &cx, &cy, &cz);
        set_leaf_boxes(&mut device, &mut tree, &boxes);
        let mut scratch = WalkScratch::default();
        let mut qbuf = Buffer::<Aabb>::none();
        let staged_boxes = boxes_on_device(&mut device, &boxes, &mut qbuf);
        let reference = query_pairs(&mut device, &mut tree, staged_boxes, boxes.len(), &mut scratch).unwrap();
        assert!(
            reference.len() > 2 * n,
            "the fixture found only self-pairs, so it does not discriminate"
        );
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| query_pairs(&mut device, &mut tree, staged_boxes, boxes.len(), &mut scratch).unwrap());
            assert_eq!(
                reference, got,
                "the candidate pair list moved at {threads} threads, so the \
                 assembly that consumes it would sum in a different order"
            );
        }
    }

    #[test]
    fn an_overflowing_buffer_is_grown_rather_than_truncated() {
        // The trap this guards is the one the contact pair cache was burned by:
        // a capacity that is quietly crossed. Here the cost of missing it is
        // not time, it is DROPPED CANDIDATE PAIRS, so the walk must report the
        // requirement and repeat rather than return what fit.
        //
        // 60 boxes all on top of each other gives 60 pairs per query against a
        // starting estimate of 4.
        let n = 60usize;
        let boxes = vec![box_about([0.0, 0.0, 0.0], 1000.0); n];
        let cx = vec![0.0f32; n];
        let cy = vec![0.0f32; n];
        let cz = vec![0.0f32; n];
        let mut device = device();
        let mut tree = tree_from_centroids(&mut device, &cx, &cy, &cz);
        set_leaf_boxes(&mut device, &mut tree, &boxes);
        let mut scratch = WalkScratch::default();
        let mut qbuf = Buffer::<Aabb>::none();
        let staged_boxes = boxes_on_device(&mut device, &boxes, &mut qbuf);
        let pairs = query_pairs(&mut device, &mut tree, staged_boxes, boxes.len(), &mut scratch).unwrap();
        assert_eq!(
            pairs.len(),
            2 * n * n,
            "every box overlaps every box, so the walk must return n^2 pairs; \
             fewer means the overflow silently truncated"
        );
    }

    #[test]
    fn the_traversal_stack_is_deep_enough_for_the_trees_this_backend_builds() {
        // The shared body's stack is fixed at AABB_MAX_QUERY, and running
        // out of it makes the walk ABANDON subtrees. A Morton-ordered tree over
        // a degenerate cloud is the deep case, so it is the one measured.
        let depth = max_query_depth();
        assert!(depth >= 64, "the shared stack shrank to {depth}");
        // Every primitive at one point: the codes are all equal, which is the
        // worst case for the split search and the deepest tree it produces.
        let n = 5000usize;
        let cx = vec![1.0f32; n];
        let cy = vec![1.0f32; n];
        let cz = vec![1.0f32; n];
        let mut device = device();
        let tree = tree_from_centroids(&mut device, &cx, &cy, &cz);
        let levels = (tree.level_offset.len() - 1) as u32;
        // The traversal pushes two children per pop, so it needs headroom of
        // about two per level.
        assert!(
            2 * levels < depth,
            "an all-equal Morton cloud of {n} primitives built a tree {levels} \
             levels deep, and the shared traversal stack holds {depth}. A walk \
             of it would overflow and silently abandon subtrees."
        );
    }

    #[test]
    fn a_walk_over_a_corrupt_tree_faults_rather_than_returning_a_short_list() {
        // The negative control for the diagnostic channel: without it, a node
        // index past the end makes the shared body break out of the walk and
        // return whatever it had, which reads as "no more contacts here".
        let boxes = vec![box_about([0.0, 0.0, 0.0], 1000.0); 8];
        let cx = vec![0.0f32; 8];
        let cy = vec![0.0f32; 8];
        let cz = vec![0.0f32; 8];
        let mut device = device();
        let mut tree = tree_from_centroids(&mut device, &cx, &cy, &cz);
        set_leaf_boxes(&mut device, &mut tree, &boxes);
        let mut scratch = WalkScratch::default();
        let mut qbuf = Buffer::<Aabb>::none();
        let staged_boxes = boxes_on_device(&mut device, &boxes, &mut qbuf);
        // Point the root's left child past the end of the node array. The
        // nodes are device-resident, so the poke is a read, an edit and a
        // write rather than an assignment.
        let root = tree.root as usize;
        let mut nodes = tree_host(&mut device, &mut tree).node;
        nodes[2 * root] = tree.node_count + 99;
        tree.node
            .seed(&mut device, &nodes)
            .expect("the fixture corrupts the tree");
        let fault = query_pairs(&mut device, &mut tree, staged_boxes, boxes.len(), &mut scratch).expect_err(
            "a node index past the end must be reported, not absorbed into a \
             short pair list",
        );
        assert!(fault.fail_count > 0);
        assert!(
            fault.file.contains("aabb_traversal"),
            "the fault should name the shared traversal body, got {}",
            fault.file
        );
        assert!(fault.describe().contains("incomplete"));
    }

    #[test]
    fn the_leaf_margin_matches_cuda() {
        // THE BROAD-PHASE INFLATION IS `0.5f * ghat + offset`, stated once in
        // `aabb_leaf_margin` (`src/kernels/contact/aabb.kernel.cpp`) and reached
        // by every leaf-AABB entry. A margin SMALLER than that generates a
        // smaller candidate set from the same geometry, which is a missed
        // contact and therefore a possible penetration, and it would show up
        // nowhere else, so the value is asserted rather than trusted.
        //
        // Driven through the REAL leaf-AABB entry point rather than compared
        // against a restatement of the formula, so it covers the plumbing too:
        // the parameter indirection, the swept box and the coordinate
        // conversion.
        let ghat = 0.25f32;
        let offset = 0.125f32;
        // Exact in fp32, so a mismatch is a real disagreement and never a
        // rounding difference.
        let expected = 0.5f32 * ghat + offset;
        assert_eq!(expected, 0.25);

        // One vertex, held still at the origin, so the box is the
        // margin and nothing else.
        let positions: [f32; 3] = [0.0, 0.0, 0.0];
        let prop = [crate::data::VertexProp {
            param_index: 0,
            ..Default::default()
        }];
        let params = [crate::data::VertexParam {
            ghat,
            offset,
            friction: 0.0,
        }];
        let mut device = device();
        let mut tree = tree_of(&mut device, &[0], &[0]);
        unsafe {
            // The leaf pass takes the positions as a handle now.
            let pose_block = crate::driver::state::position_block(
                &mut device,
                bytemuck::cast_slice(&positions),
                "test.pose",
            );
            let pose = pose_block.handle();
            let prop_block =
                crate::driver::state::record_block(&mut device, &prop, "test.prop");
            let prop_h = prop_block.handle();
            let param_block =
                crate::driver::state::record_block(&mut device, &params, "test.params");
            let param_h = param_block.handle();
            refresh_vertex_leaves(
                &mut device,
                &mut tree,
                pose,
                pose,
                1.0,
                prop_h,
                param_h,
            )
            .expect("the leaf box is written");
            propagate(&mut device, &mut tree).expect("the tree is propagated");
        }
        // READ BACK: the boxes are device-resident.
        let leaf = tree_host(&mut device, &mut tree).aabb[0];
        assert!(leaf.active, "the leaf box was not written");
        for axis in 0..3 {
            assert_eq!(
                leaf.max[axis], expected,
                "axis {axis}: the leaf half-width is {} and the shared body \
                 spells 0.5 * ghat + offset, which is {expected}",
                leaf.max[axis]
            );
            assert_eq!(leaf.min[axis], -expected);
        }
    }
}
