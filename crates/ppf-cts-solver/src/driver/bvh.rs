// File: crates/ppf-cts-solver/src/driver/bvh.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The BVH skeleton: scene bounds, Morton codes, and the ordering the tree is
//! built from.
//!
//! The split here is sharper than elsewhere, so it is worth stating plainly.
//!
//! | shared C++ | here, in Rust |
//! |---|---|
//! | the bit interleave and the quantizer (`lbvh.kernel.cpp`) | the loop over primitives, the chunking, the bounds reduction, the sort |
//!
//! **A Morton code decides tree TOPOLOGY**, so two implementations of the
//! interleave would be two different trees, whose traversals find the same
//! contacts in different orders, whose fp32 assembly then sums differently.
//! Nothing would assert. That is why the interleave is shared even though it is
//! nine lines of shifts and masks that anyone could retype.
//!
//! The bounds reduction is a min/max monoid, which is associative and
//! commutative over finite floats, so the tree shape does not depend on how it
//! was folded. That makes it genuinely free skeleton, unlike the summation folds
//! in `reduce.rs` whose shape is part of the answer. It is the model case the
//! audit named.

// These are P1 components, landed ahead of the Newton driver that will call
// them. The allow is removed in the change that wires the driver up.
#![allow(dead_code)]

use ppf_cts_compute::{AllocLabel, Device, EncoderExt, ReadbackBuffer};
use super::kernels::{BoundsLeafArgs, BoundsMergeArgs, LbvhMortonFromBoundsArgs};
use super::scene::{Fatal, FatalResult};
// Named only by `primitive_order`, which is itself `#[cfg(test)]`.
#[cfg(test)]
use super::sort;
use rayon::prelude::*;

// TWO SCALAR QUERIES, AND THEY ARE NOT DISPATCHES.
//
// Each takes scalars and returns one scalar: no thread index, no extent, no
// buffer. They are the same category as `super::pcg`'s per-block inverse and
// `super::step`'s domain query, and forcing either into a dispatch shape
// would mean inventing an extent to satisfy a count. What reads them is the
// test that compares this backend's quantizer and interleave against the
// shared bodies rather than against a second implementation of them; production
// reaches the same two bodies through `morton_entry`, which IS
// dispatched, below. If a query surface is ever wanted on `Device`, these move
// with the other two and not before.
extern "C" {
    fn expand_bits_abi(v: u32) -> u32;
    fn morton_code_3d_abi(x: u32, y: u32, z: u32) -> u32;
}

/// An axis-aligned bound over the scene, in the SoA form the Morton pass reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Bounds {
    /// The identity for the monoid below. `FLT_MAX` here is the C++ tree's
    /// sentinel value of 1.0e8, NOT `f32::MAX`: `common.hpp` redefines it, and a
    /// bound seeded with 3.4e38 would produce a quantizer span that rounds every
    /// centroid to the same cell.
    pub const EMPTY: Bounds = Bounds {
        min: [1.0e8; 3],
        max: [-1.0e8; 3],
    };

    fn merge(mut self, other: Bounds) -> Bounds {
        for k in 0..3 {
            self.min[k] = self.min[k].min(other.min[k]);
            self.max[k] = self.max[k].max(other.max[k]);
        }
        self
    }

    fn of_point(p: [f32; 3]) -> Bounds {
        Bounds { min: p, max: p }
    }
}

/// Scene bounds over SoA centroids.
///
/// Folded as a min/max monoid, so the result does not depend on the thread
/// count or the fold order. That is a property of min and max rather than a
/// discipline this code maintains, which is what makes the reduction free here
/// where a summation fold's would not be.
pub fn scene_bounds(cx: &[f32], cy: &[f32], cz: &[f32]) -> Bounds {
    assert_eq!(cx.len(), cy.len());
    assert_eq!(cx.len(), cz.len());
    if cx.is_empty() {
        return Bounds::EMPTY;
    }
    (0..cx.len())
        .into_par_iter()
        .map(|i| Bounds::of_point([cx[i], cy[i], cz[i]]))
        .reduce(|| Bounds::EMPTY, Bounds::merge)
}

/// Morton codes for every centroid, using the shared quantizer and interleave.
///
/// # Safety
/// The three centroid slices and the code buffer are named to the backend by
/// address, so they must stay alive and unmoved until the dispatch returns.
/// They are this function's own locals and its caller's slices, so that holds
/// by construction; the `unsafe` is here because [`Device::launch`] cannot know
/// it.
/// THE THREE CENTROID ARRAYS AND THE BOUNDS ARE ALL HANDLES, and nothing here
/// crosses to the host. The bounds are a device array rather than six SCALARS
/// folded on the host over `cx.host()`, `cy.host()` and `cz.host()`: the
/// reference reduces them with `compute_scene_bounds_kernel` and hands
/// `compute_morton_codes_kernel` the array, which is what a scalar parameter
/// cannot express once the producing pass is a kernel. The codes are not read
/// back either: the sort that consumes them is the device network in
/// [`super::devsort`].
pub unsafe fn morton_codes<D: Device>(
    device: &mut D,
    cx: ppf_cts_compute::Handle,
    cy: ppf_cts_compute::Handle,
    cz: ppf_cts_compute::Handle,
    codes: &mut ReadbackBuffer<u32>,
    n: usize,
    bounds: ppf_cts_compute::Handle,
) -> FatalResult<()> {
    if n == 0 {
        return Ok(());
    }
    codes
        .size(device, n, AllocLabel("bvh.morton_codes"))
        .map_err(|error| {
            Fatal::out_of_memory(format!("solver driver: cannot size the Morton codes: {error:?}"))
        })?;
    let args = LbvhMortonFromBoundsArgs {
        cx,
        cy,
        cz,
        bounds,
        codes: codes.handle(),
        count: n as u32,
        seam_arena_count: 0,
    };
    device.launch("bvh.morton", &args, n as u32)?;
    Ok(())
}

/// The primitive order the tree is built from: ascending Morton code, ties in
/// ascending primitive index.
///
/// THE HOST ORACLE, AND ONLY THAT. Production sorts on the device through
/// [`super::devsort`], whose network carries the same total order; this is what
/// a test compares that order against, which is why it is `#[cfg(test)]` rather
/// than deleted.
///
/// The tie rule is not incidental: Karras's internal-node construction resolves
/// a duplicate-code run by index, so a permuted run yields a different tree.
/// `sort.rs` carries that contract and its tests exercise it.
#[cfg(test)]
pub fn primitive_order(codes: &[u32]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..codes.len() as u32).collect();
    sort::par_stable_sort_by_key(&mut order, codes);
    order
}

/// The shared bit interleave, exposed so a test can compare against it rather
/// than against a second implementation of it.
pub fn expand_bits(v: u32) -> u32 {
    unsafe { expand_bits_abi(v) }
}

/// The shared 3D Morton code.
pub fn morton_code_3d(x: u32, y: u32, z: u32) -> u32 {
    unsafe { morton_code_3d_abi(x, y, z) }
}

#[cfg(test)]
mod tests {
    /// The fixture's centroids on the device, and its codes back.
    ///
    /// `morton_codes` takes handles now, so a test stages its three arrays and
    /// reads the result out of the readback buffer it passed.
    fn codes_of(
        device: &mut impl Device,
        cx: &[f32],
        cy: &[f32],
        cz: &[f32],
        bounds: Bounds,
    ) -> Vec<u32> {
        let n = cx.len();
        let mut dx = ReadbackBuffer::<f32>::default();
        let mut dy = ReadbackBuffer::<f32>::default();
        let mut dz = ReadbackBuffer::<f32>::default();
        let mut codes = ReadbackBuffer::<u32>::default();
        // SIZED BEFORE SEEDED: `seed` fills both halves and asserts the buffer
        // already has the length it is given.
        for (buffer, label) in [
            (&mut dx, "test.cx"),
            (&mut dy, "test.cy"),
            (&mut dz, "test.cz"),
        ] {
            buffer.size(device, n, AllocLabel(label)).expect("the centroid buffer sizes");
        }
        dx.seed(device, cx).expect("cx stages");
        dy.seed(device, cy).expect("cy stages");
        dz.seed(device, cz).expect("cz stages");
        // THE BOUNDS ARE STAGED RATHER THAN REDUCED HERE, so the test still
        // controls them: it hands `codes_of` the bounds it wants the
        // quantization taken against, and the reduction has its own test.
        let mut staged = ReadbackBuffer::<f32>::default();
        staged
            .size(device, 6, AllocLabel("test.bounds"))
            .expect("the bounds size");
        staged
            .seed(
                device,
                &[
                    bounds.min[0], bounds.min[1], bounds.min[2],
                    bounds.max[0], bounds.max[1], bounds.max[2],
                ],
            )
            .expect("the bounds stage");
        // Safety: every handle names a live allocation on this device.
        unsafe {
            morton_codes(
                device,
                dx.span(0, n),
                dy.span(0, n),
                dz.span(0, n),
                &mut codes,
                n,
                staged.span(0, 6),
            )
        }
        .expect("codes");
        codes.download(device).expect("the codes read back");
        codes.host()[..n].to_vec()
    }
    use super::*;

    /// The device reduction against the host fold, at sizes that exercise every
    /// recursion depth.
    ///
    /// THE SIZES ARE CHOSEN AGAINST `BLOCK`, not arbitrarily: one under a
    /// block, exactly a block, one over (the first size that needs a merge at
    /// all) and one that makes the merge level itself more than one block. A
    /// reduction tested only below `BLOCK` never merges and would pass with the
    /// whole loop deleted.
    #[test]
    fn the_device_bounds_match_the_host_fold_at_every_depth() {
        let block = BoundsScratch::BLOCK as usize;
        let mut device = crate::driver::launch::host_device();
        for &n in &[1usize, 2, 7, block - 1, block, block + 1, 3 * block + 5] {
            let (cx, cy, cz) = cloud(n);
            let expected = scene_bounds(&cx, &cy, &cz);
            let mut dx = ReadbackBuffer::<f32>::default();
            let mut dy = ReadbackBuffer::<f32>::default();
            let mut dz = ReadbackBuffer::<f32>::default();
            for (buffer, label) in [
                (&mut dx, "test.bx"),
                (&mut dy, "test.by"),
                (&mut dz, "test.bz"),
            ] {
                buffer.size(&mut device, n, AllocLabel(label)).expect("sized");
            }
            dx.seed(&mut device, &cx).expect("cx");
            dy.seed(&mut device, &cy).expect("cy");
            dz.seed(&mut device, &cz).expect("cz");
            let mut scratch = BoundsScratch::default();
            // Safety: the three arrays outlive the call and name `n` floats.
            let handle = unsafe {
                scratch
                    .reduce(
                        &mut device,
                        dx.span(0, n),
                        dy.span(0, n),
                        dz.span(0, n),
                        n as u32,
                    )
                    .expect("reduced")
            };
            let _ = handle;
            let got = scratch.host_bounds(&mut device);
            for d in 0..3 {
                assert_eq!(got[d], expected.min[d], "n {n}: minimum {d}");
                assert_eq!(got[3 + d], expected.max[d], "n {n}: maximum {d}");
            }
        }
    }

    fn cloud(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut cx = Vec::with_capacity(n);
        let mut cy = Vec::with_capacity(n);
        let mut cz = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 * 0.017;
            cx.push(t.sin() * 3.0 + 0.5);
            cy.push(t.cos() * 2.0 - 1.25);
            cz.push((t * 0.37).sin() * 4.0);
        }
        (cx, cy, cz)
    }

    #[test]
    fn the_bounds_contain_every_point() {
        let (cx, cy, cz) = cloud(10_000);
        let b = scene_bounds(&cx, &cy, &cz);
        for i in 0..cx.len() {
            for (k, v) in [cx[i], cy[i], cz[i]].iter().enumerate() {
                assert!(
                    *v >= b.min[k] && *v <= b.max[k],
                    "point {i} axis {k} value {v} escapes [{}, {}]",
                    b.min[k],
                    b.max[k]
                );
            }
        }
    }

    #[test]
    fn the_bounds_do_not_depend_on_the_thread_count() {
        // Free, because min and max are associative and commutative. Asserted
        // anyway: it is the claim that lets this reduction stay unconstrained
        // where the summation folds are pinned.
        let (cx, cy, cz) = cloud(50_000);
        let reference = scene_bounds(&cx, &cy, &cz);
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| scene_bounds(&cx, &cy, &cz));
            assert_eq!(reference, got, "bounds moved at {threads} threads");
        }
    }

    #[test]
    fn an_empty_cloud_gives_the_empty_bound() {
        assert_eq!(scene_bounds(&[], &[], &[]), Bounds::EMPTY);
    }

    #[test]
    fn expand_bits_interleaves_with_two_zero_bits() {
        // The property the interleave exists for: bit k of the input lands at
        // bit 3k of the output. Checked against the definition rather than
        // against a reimplementation.
        for k in 0..10u32 {
            assert_eq!(
                expand_bits(1 << k),
                1 << (3 * k),
                "bit {k} did not land at 3k"
            );
        }
        assert_eq!(expand_bits(0), 0);
    }

    #[test]
    fn morton_code_interleaves_the_three_axes() {
        // x occupies bits 0, 3, 6...; y bits 1, 4, 7...; z bits 2, 5, 8...
        assert_eq!(morton_code_3d(1, 0, 0), 1);
        assert_eq!(morton_code_3d(0, 1, 0), 2);
        assert_eq!(morton_code_3d(0, 0, 1), 4);
        assert_eq!(morton_code_3d(1, 1, 1), 7);
        assert_eq!(morton_code_3d(2, 0, 0), 1 << 3);
    }

    #[test]
    fn morton_codes_do_not_depend_on_the_chunking() {
        let (cx, cy, cz) = cloud(60_000);
        let b = scene_bounds(&cx, &cy, &cz);
        let mut device = super::super::launch::host_device();
        let reference = codes_of(&mut device, &cx, &cy, &cz, b);
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| {
                let mut device = super::super::launch::host_device();
                codes_of(&mut device, &cx, &cy, &cz, b)
            });
            assert_eq!(reference, got, "codes moved at {threads} threads");
        }
        assert!(
            reference.iter().any(|c| *c != reference[0]),
            "every code came out identical, so the comparison was vacuous"
        );
    }

    #[test]
    fn nearby_points_get_nearby_codes() {
        // The whole point of a Morton order: locality. Two points in the same
        // corner must share a long high-bit prefix, and a point across the scene
        // must not.
        let cx = vec![0.0f32, 0.001, 10.0];
        let cy = vec![0.0f32, 0.001, 10.0];
        let cz = vec![0.0f32, 0.001, 10.0];
        let b = scene_bounds(&cx, &cy, &cz);
        let mut device = super::super::launch::host_device();
        let codes = codes_of(&mut device, &cx, &cy, &cz, b);
        let shared = |a: u32, b: u32| (a ^ b).leading_zeros();
        assert!(
            shared(codes[0], codes[1]) > shared(codes[0], codes[2]),
            "the near pair shares {} leading bits and the far pair {}, so the \
             order is not spatial",
            shared(codes[0], codes[1]),
            shared(codes[0], codes[2])
        );
    }

    #[test]
    fn the_primitive_order_is_ascending_and_ties_break_by_index() {
        // Duplicate codes are the normal case wherever primitives share a cell,
        // and the tie rule is what keeps the tree reproducible.
        let codes: Vec<u32> = (0..5000u32).map(|i| i % 32).collect();
        let order = primitive_order(&codes);
        for w in order.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            assert!(codes[a] <= codes[b], "the order is not ascending by code");
            if codes[a] == codes[b] {
                assert!(
                    w[0] < w[1],
                    "equal codes came out permuted: {} before {}; Karras \
                     resolves a duplicate run by index, so this is a different \
                     tree",
                    w[0],
                    w[1]
                );
            }
        }
    }
}

/// The scene's centroid bounds, reduced on the device.
///
/// SIX FLOATS THAT NEVER REACH THE HOST. `bounds_leaf` and `bounds_merge`
/// (`kernels/primitives/reduce_bounds.kernel.cpp`) write them and
/// `lbvh_morton_from_bounds` reads them straight off the device. Folding the
/// six on the host would instead be `scene_bounds(&cx.host()[..n], ...)`, three
/// O(n) host reads once per tree per step.
///
/// A LEVEL AT A TIME, the same recursion `super::scan` uses and for the same
/// reason: reducing within a block needs `__shared__` and `__syncthreads`,
/// neither of which has a host spelling, so the cooperative layer comes out and
/// one thread walks a block. Minimum and maximum are exact on floats, so every
/// association order gives the same bits.
#[derive(Debug, Default)]
pub struct BoundsScratch {
    /// Six floats per block, ping-ponged between levels: a merge reads one span
    /// and writes another, because a thread reads `BLOCK` records and writes
    /// one and an in-place pass would read a record already replaced.
    ///
    /// A READBACK BUFFER SO A TEST CAN CHECK IT, not because production reads
    /// it: the six floats reach the Morton pass as a handle and never come
    /// down. [`Self::last`] is where the final record landed.
    level: ReadbackBuffer<f32>,
    last: usize,
}

impl BoundsScratch {
    /// The width one thread walks, and the branching factor of the recursion.
    ///
    /// ONE THREAD WALKS THIS MANY CENTROIDS SERIALLY, so it is the leaf pass's
    /// parallelism divided into the count rather than a tuning knob: at 4,096 a
    /// hundred-thousand-leaf tree reduced on about twenty-five threads, and the
    /// pass measured 530 us a call. A cooperative reduction would instead give
    /// one element to one thread and fold through a 256-wide shared tree; this
    /// one has no cooperative body, so the same effect is bought by making the
    /// serial run short and letting the existing merge tree fold the rest, at a
    /// width of 32 rather than 4,096. The merge is 7 us a call, so the
    /// extra levels a smaller width costs are far cheaper than the serial walk
    /// they remove.
    const BLOCK: u32 = 32;

    /// Reduce `count` centroids and return the handle to their six-float bounds.
    ///
    /// # Safety
    /// The three arrays must each name at least `count` floats and outlive the
    /// call.
    pub unsafe fn reduce<D: Device>(
        &mut self,
        device: &mut D,
        cx: ppf_cts_compute::Handle,
        cy: ppf_cts_compute::Handle,
        cz: ppf_cts_compute::Handle,
        count: u32,
    ) -> FatalResult<ppf_cts_compute::Handle> {
        let oom = |error: ppf_cts_compute::Fault| {
            Fatal::out_of_memory(format!("solver driver: cannot size the bounds reduce: {error:?}"))
        };
        // TWO SPANS PER LEVEL AND A PING-PONG BETWEEN THEM, so the widest level
        // and the one above it are both live at once. `first + first / BLOCK`
        // bounds every level below the first, and the `+ 2` covers the tail.
        let first = count.div_ceil(Self::BLOCK).max(1);
        // EVERY LEVEL THE MERGE LOOP WILL WRITE, SUMMED. It folds by `BLOCK`
        // until one entry is left, so the level count is logarithmic in `first`
        // and a span sized for a fixed two of them overruns as soon as `BLOCK`
        // is small enough for the leaf pass to be parallel.
        let mut span = 0usize;
        let mut level = first;
        loop {
            span += level as usize;
            if level <= 1 {
                break;
            }
            level = level.div_ceil(Self::BLOCK);
        }
        let span = span + 2;
        self.level
            .size(device, 6 * span, AllocLabel("bvh.bounds"))
            .map_err(oom)?;

        let mut blocks = first;
        let leaf = LeafBoundsSpans {
            out: self.level.span(0, 6 * blocks as usize),
        };
        let args = BoundsLeafArgs {
            cx,
            cy,
            cz,
            count,
            block_size: Self::BLOCK,
            out: leaf.out,
            blocks,
            seam_arena_count: 0,
        };
        device.launch("bvh.bounds_leaf", &args, blocks)?;

        // Each merge reads the level just written and writes the one after it,
        // at an offset past everything still live.
        // ONE SUBMIT FOR THE WHOLE LADDER, not one per level. Each level
        // reads the one below it, and a stream orders dispatches whether or not
        // they depend on each other, so the levels need no synchronize between
        // them: only the last one's result is read, and that read is a later
        // boundary of its own. A submit ends in a synchronize and one was
        // measured at 15.5 us, so a four-level ladder paid three of those for
        // nothing.
        //
        // THE LEVELS ARE BOUNDED, so they are collected into a fixed array
        // rather than a `Vec`: folding by `BLOCK` reaches one from any `u32`
        // count in at most seven steps, and a host allocation in a per-step
        // path is what this file's own sizing rules exist to avoid.
        let mut ladder: [Option<(BoundsMergeArgs, u32)>; 8] = Default::default();
        let mut rungs = 0usize;
        let mut source_at = 0usize;
        while blocks > 1 {
            let next = blocks.div_ceil(Self::BLOCK);
            let destination_at = source_at + 6 * blocks as usize;
            let merge = BoundsMergeArgs {
                source: self.level.span(source_at, 6 * blocks as usize),
                count: blocks,
                block_size: Self::BLOCK,
                destination: self.level.span(destination_at, 6 * next as usize),
                blocks: next,
                seam_arena_count: 0,
            };
            assert!(
                rungs < ladder.len(),
                "solver driver: the bounds ladder wants more than {} levels at \
                 branching factor {}; the array above is sized for every u32 \
                 count and a deeper one means the factor changed",
                ladder.len(),
                Self::BLOCK
            );
            ladder[rungs] = Some((merge, next));
            rungs += 1;
            source_at = destination_at;
            blocks = next;
        }
        if rungs > 0 {
            device.run("bvh.bounds_merge", |encoder| {
                for rung in ladder.iter().flatten() {
                    // Safety: every span is borrowed for the whole call and the
                    // dispatches complete before `run` returns.
                    unsafe { encoder.elements(&rung.0, rung.1)? };
                }
                Ok(())
            })?;
        }
        self.last = source_at;
        Ok(self.level.span(source_at, 6))
    }

    /// The six floats the reduction settled on, for a test to compare against
    /// the host fold. Production never reads them on the host.
    #[cfg(test)]
    pub fn host_bounds<D: Device>(&mut self, device: &mut D) -> [f32; 6] {
        self.level.download(device).expect("the bounds read back");
        let mut out = [0.0f32; 6];
        out.copy_from_slice(&self.level.host()[self.last..self.last + 6]);
        out
    }
}

/// The leaf level's destination, named so the borrow of `self.level` ends
/// before the record is filled.
struct LeafBoundsSpans {
    out: ppf_cts_compute::Handle,
}
