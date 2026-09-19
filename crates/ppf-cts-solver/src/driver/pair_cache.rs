// File: crates/ppf-cts-solver/src/driver/pair_cache.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The detect-once contact pair cache.
//!
//! A Newton step walks the BVH once, records the candidate pairs it found, and
//! replays that list on every iteration after the first. The four contact types
//! (point-face, point-edge, point-point, edge-edge) each get their own list,
//! because their pairs address different primitive arrays.
//!
//! # It sizes ITSELF, and a constant may not be put back
//!
//! This is the whole reason the module exists rather than four `Vec`s beside
//! the assembly. The CUDA cache once carried a fixed cap of 2^21. On a
//! coarse-collider-against-fine-deformable scene the edge-edge count crossed it
//! BETWEEN two steps, the replay was abandoned, the BVH was re-walked, and
//! `fill_traverse` went from 24 ms to 296 ms with NOTHING IN THE LOG. The
//! behavior was correct the whole time; what was missing was any way to find
//! out. Two properties fix that and both are load-bearing:
//!
//! - **`record` counts every pair whether or not it fit.** The shared body
//!   `pair_cache_record` increments its counter before it checks capacity,
//!   so the REQUIREMENT is known even on the pass that could not meet it. A
//!   cache that stopped counting where it stopped storing would have to guess
//!   how much to grow by.
//! - **Capacity tracks twice the largest count seen**, per type, so one
//!   overflow is paid once rather than every step after it.
//!
//! And when the replay is abandoned, the fallback SAYS SO and says what it
//! cost.
//!
//! # The abandonment is all four types at once
//!
//! Stage 1 either replays a complete pair list or it does not. A partial replay
//! (three types from the cache and one re-walked) would assemble contact from
//! two different detections of the same configuration, so an overflow in any one
//! type invalidates the step's whole cache.
//!
//! # The split
//!
//! | shared C++ | here, in Rust |
//! |---|---|
//! | the record step and its overflow flag (`pair_cache.kernel.cpp`) | the four buffers, their capacities, the high-water mark, the replay decision |

// NO PRODUCTION CALLER, AND THE REASON IS THIS BACKEND'S DYNAMIC MATRIX RATHER
// THAN AN UNFINISHED WIRING. `contact.cu` runs its assembly in two stages, a
// dry pass that reserves the dynamic CSR's slots and a fill pass that writes
// them, and the cache exists so the second stage need not walk the BVH again.
// `src/driver/dyncsr.rs` grows a row as blocks arrive, so this backend has one
// stage and no second traversal to save. What the cache would serve here is a
// two-stage assembly, which is the shape a backend with a fixed-capacity
// dynamic matrix needs; `src/driver/contact.rs` carries its own per-type candidate
// density instead, which is the same self-sizing discipline applied to the
// quantity this walk actually measures.
#![allow(dead_code)]

use ppf_cts_compute::{AllocLabel, Buffer, Device, Fault, ReadbackBuffer};
use super::kernels::PairCacheRecordInterleavedArgs;
use super::scene::FatalResult;

/// The four contact types, which are four separate pair lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairType {
    PointFace,
    PointEdge,
    PointPoint,
    EdgeEdge,
}

impl PairType {
    pub const ALL: [PairType; 4] = [
        PairType::PointFace,
        PairType::PointEdge,
        PairType::PointPoint,
        PairType::EdgeEdge,
    ];

    fn slot(self) -> usize {
        match self {
            PairType::PointFace => 0,
            PairType::PointEdge => 1,
            PairType::PointPoint => 2,
            PairType::EdgeEdge => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PairType::PointFace => "point-face",
            PairType::PointEdge => "point-edge",
            PairType::PointPoint => "point-point",
            PairType::EdgeEdge => "edge-edge",
        }
    }
}

/// NOT `Clone`: two of its fields are device allocations, and a clone would
/// hand out a second cache naming the same arena spans.
#[derive(Debug, Default)]
struct TypeCache {
    /// Two `u32` per pair, `capacity` pairs long.
    ///
    /// KERNEL-WRITTEN AND HOST-READ, which is what picks the type: the record
    /// pass writes it and [`PairCache::replay`] hands the host a slice of it.
    data: ReadbackBuffer<u32>,
    capacity: u32,
    /// The caller's pair list, staged so the record can name it.
    staged_pairs: Buffer<u32>,
    /// The counter and the overflow flag, in ONE buffer of two elements.
    ///
    /// Two allocations for two scalars would be two dispatceh-time bounds and
    /// two downloads; the pass writes both and the host reads both, so they
    /// travel together. Element 0 is the count and element 1 the flag.
    verdict: ReadbackBuffer<u32>,
    /// Pairs OFFERED this detection, which runs past `capacity` on purpose.
    count: u32,
    /// Set by the shared body the first time a pair did not fit.
    overflow: u32,
    /// The largest `count` any detection has reported, which is what the next
    /// capacity is sized from.
    high_water: u32,
}

/// One step's candidate pairs, for all four contact types.
/// NOT `Clone`, as [`TypeCache`].
#[derive(Debug, Default)]
pub struct PairCache {
    types: [TypeCache; 4],
    /// True once a detection did not fit, until the next `begin_detect`.
    abandoned: bool,
    /// How many times the replay has been abandoned over the cache's life.
    /// Reported rather than merely counted: a repeated fallback is a scene
    /// whose candidate density the growth rule is not keeping up with.
    fallbacks: u64,
}

impl PairCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a fresh detection. Capacities are already sized from the last one.
    ///
    /// TAKES A DEVICE, because the growth below is a device allocation now.
    /// `Buffer::size` grows only past CAPACITY, so a detection no larger than
    /// the last allocates nothing and the two-element verdict never regrows.
    pub fn begin_detect<D: Device>(&mut self, device: &mut D) -> Result<(), Fault> {
        for cache in &mut self.types {
            cache.count = 0;
            cache.overflow = 0;
            // The buffer is grown HERE rather than at the end of the previous
            // detection, so a step that never runs does not pay for one.
            let wanted = 2 * cache.high_water.max(1);
            if cache.capacity < wanted {
                cache.capacity = wanted;
            }
            // SIZED ONLY WHEN THE CAPACITY MOVED, AND DELIBERATELY NOT CLEARED
            // OTHERWISE. `Buffer::size` zeroes what it sizes, so calling it on
            // every detection fills the whole pair array every step: at the
            // reference's own starting capacity that is 2^21 pairs, four types,
            // per step, to no purpose. `contact.cu:1496-1497` clears `cc_cnt`
            // and `cc_overflow` and nothing else, its four pair buffers being
            // `pool.get<Vec2u>(cap)` with no clear at all.
            //
            // NOTHING READS AN UNWRITTEN SLOT, which is what licenses that on a
            // target that never faults: `pair_cache_record` claims a slot from
            // the counter and writes BOTH of its words before anyone can reach
            // it, and [`PairCache::replay`] hands out exactly `2 * count`
            // elements, so every element a reader sees was written by this
            // detection. The two-element verdict below is the opposite case, an
            // atomic add and a store that must start from zero, and it is
            // cleared for that reason rather than by habit.
            let elements = 2 * cache.capacity as usize;
            if cache.data.len() != elements {
                cache
                    .data
                    .size(device, elements, AllocLabel("contact.pair_cache"))?;
            }
            // ZEROED EVERY DETECTION, AND EXPLICITLY: the pass writes the
            // counter with an atomic add and the flag by a store, so both must
            // start at zero and neither is written by every thread.
            //
            // `Buffer::size` used to do this as a side effect, zeroing on every
            // call including the no-op where the length had not changed. It
            // zeroes only bytes it has just allocated now, so a detection after
            // the first would inherit the previous one's counter, and the clear
            // this comment always described has to be one.
            cache
                .verdict
                .size(device, 2, AllocLabel("contact.pair_verdict"))?;
            device.fill_zero(
                cache.verdict.handle(),
                2 * std::mem::size_of::<u32>(),
            )?;
        }
        self.abandoned = false;
        Ok(())
    }

    /// Record the candidate pairs of one type.
    ///
    /// `pairs` is the flat `(a, b)` list a traversal produced. The write is
    /// SERIAL, and that is not an optimization left undone:
    /// `compute::atomic_add` on the host seam is a plain read, add and write
    /// back, so a parallel scatter over pairs sharing this counter is a data
    /// race rather than a different fold order. Evaluating in parallel and
    /// scattering serially in ascending order is also what makes the cache
    /// independent of the thread count.
    ///
    /// # Safety
    /// `pairs` and this cache's own buffers are named to the backend by
    /// address, so they must stay alive and unmoved for the dispatch. They are
    /// the caller's slice and this struct's fields, so that holds by
    /// construction; the `unsafe` is here because [`Device::launch`] cannot
    /// know it.
    pub unsafe fn record<D: Device>(
        &mut self,
        device: &mut D,
        kind: PairType,
        pairs: &[u32],
    ) -> FatalResult<()> {
        assert_eq!(
            pairs.len() % 2,
            0,
            "a pair list is two indices per pair, and an odd length means the \
             producer and this consumer disagree about the layout"
        );
        let cache = &mut self.types[kind.slot()];
        let n = (pairs.len() / 2) as u32;
        if n == 0 {
            return Ok(());
        }
        cache
            .staged_pairs
            .size(device, pairs.len(), AllocLabel("contact.pair_input"))
            .and_then(|()| cache.staged_pairs.write(device, 0, pairs))?;
        let args = PairCacheRecordInterleavedArgs {
            pair_data: cache.data.handle(),
            count: cache.verdict.span(0, 1),
            overflow: cache.verdict.span(1, 1),
            capacity: cache.capacity,
            pairs: cache.staged_pairs.span(0, pairs.len()),
            pair_count: n,
            seam_arena_count: 0,
        };
        // The row is `Scatter::Claim`, so the backend runs this as one serial
        // ascending pass over the pairs. That is what the counter requires and
        // what makes the recorded order independent of the thread count.
        device.launch("contact.pair_cache.record", &args, n)?;
        // THE VERDICT AND THE PAIRS COME BACK HERE, because both are read on
        // the host: `finish_detect` folds the counter and the flag, and
        // `replay` hands out a slice of the data. A readback buffer refuses a
        // stale read rather than answering out of the previous detection, so
        // the download is where the write is rather than where the read is.
        cache.verdict.download(device)?;
        cache.data.download(device)?;
        cache.count = cache.verdict.host()[0];
        cache.overflow = cache.verdict.host()[1];
        Ok(())
    }

    /// Close the detection: grow the high-water mark and decide whether the
    /// cache may be replayed.
    ///
    /// Returns `true` when every type fit. On `false` the caller must re-walk
    /// the BVH for ALL FOUR types this step; the next `begin_detect` will have
    /// grown the buffers.
    pub fn finish_detect(&mut self) -> bool {
        let mut fit = true;
        for cache in &mut self.types {
            cache.high_water = cache.high_water.max(cache.count);
            if cache.overflow != 0 {
                fit = false;
            }
        }
        if !fit {
            self.abandoned = true;
            self.fallbacks += 1;
            // The fallback is CORRECT and it is not free, and the defect this
            // log exists for is that it used to be both correct and invisible.
            let detail: Vec<String> = PairType::ALL
                .iter()
                .map(|kind| {
                    let cache = &self.types[kind.slot()];
                    format!(
                        "{} {}/{}{}",
                        kind.name(),
                        cache.count,
                        cache.capacity,
                        if cache.overflow != 0 { " (overflow)" } else { "" }
                    )
                })
                .collect();
            log::info!(
                "cpu contact pair cache: a detection did not fit, so the replay \
                 is abandoned for all four types and the BVH is re-walked this \
                 step (fallback {} of this run). Counts were {}. The buffers \
                 grow to twice the measured requirement before the next step.",
                self.fallbacks,
                detail.join(", ")
            );
        }
        fit
    }

    /// The recorded pairs of one type, or `None` when the last detection
    /// overflowed and the whole cache must be re-walked.
    pub fn replay(&self, kind: PairType) -> Option<&[u32]> {
        if self.abandoned {
            return None;
        }
        let cache = &self.types[kind.slot()];
        Some(&cache.data.host()[..2 * cache.count as usize])
    }

    /// Pairs OFFERED for a type in the last detection, capacity or not.
    pub fn count(&self, kind: PairType) -> u32 {
        self.types[kind.slot()].count
    }

    /// Pairs the buffer can hold for a type.
    pub fn capacity(&self, kind: PairType) -> u32 {
        self.types[kind.slot()].capacity
    }

    /// The largest count ever offered for a type.
    pub fn high_water(&self, kind: PairType) -> u32 {
        self.types[kind.slot()].high_water
    }

    /// How many times the replay has been abandoned.
    pub fn fallback_count(&self) -> u64 {
        self.fallbacks
    }
}

#[cfg(test)]
mod tests {
    use super::super::launch::{host_device, HostDevice};
    use super::*;

    /// A backend for one test.
    ///
    /// ONE PER FIXTURE, and it must be: the cache's buffers are device
    /// allocations now, so a cache sized on one device and recorded into on
    /// another resolves handles against an arena that never held them. The
    /// fixtures below bind this once at the top of the body rather than calling
    /// it at each use.
    fn device() -> HostDevice {
        host_device()
    }

    fn pairs(n: u32) -> Vec<u32> {
        (0..n).flat_map(|i| [i, i + 1000]).collect()
    }

    #[test]
    fn a_fitting_detection_replays_exactly_what_it_recorded() {
        // ONE DEVICE PER FIXTURE: the cache holds device allocations.
        let mut device = device();
        let mut cache = PairCache::new();
        cache.begin_detect(&mut device).expect("the cache sizes");
        // The first detection has a capacity of 2, so give it 2.
        unsafe { cache.record(&mut device, PairType::EdgeEdge, &pairs(2)) }
            .expect("the pairs record");
        assert!(cache.finish_detect());
        assert_eq!(cache.replay(PairType::EdgeEdge).unwrap(), &pairs(2)[..]);
        // The other three recorded nothing and replay nothing, which is not the
        // same as being unavailable.
        for kind in [PairType::PointFace, PairType::PointEdge, PairType::PointPoint] {
            assert_eq!(cache.replay(kind).unwrap().len(), 0);
        }
    }

    #[test]
    fn the_requirement_is_known_even_from_the_pass_that_could_not_meet_it() {
        // ONE DEVICE PER FIXTURE: the cache holds device allocations.
        let mut device = device();
        // THE PROPERTY THE FIXED CAP LACKED. The count runs past capacity, so
        // the cache learns the real requirement from the very step it failed on
        // rather than having to guess how much to grow by.
        let mut cache = PairCache::new();
        cache.begin_detect(&mut device).expect("the cache sizes");
        assert_eq!(cache.capacity(PairType::EdgeEdge), 2);
        unsafe { cache.record(&mut device, PairType::EdgeEdge, &pairs(700)) }
            .expect("the pairs record");
        assert_eq!(
            cache.count(PairType::EdgeEdge),
            700,
            "the counter stopped at the capacity, so the requirement is unknown"
        );
        assert!(!cache.finish_detect());
        assert_eq!(cache.high_water(PairType::EdgeEdge), 700);
    }

    #[test]
    fn an_overflow_abandons_the_replay_for_every_type_not_just_its_own() {
        // ONE DEVICE PER FIXTURE: the cache holds device allocations.
        let mut device = device();
        // A partial replay would assemble contact from two different detections
        // of one configuration.
        let mut cache = PairCache::new();
        cache.begin_detect(&mut device).expect("the cache sizes");
        unsafe { cache.record(&mut device, PairType::PointFace, &pairs(1)) }
            .expect("the pairs record");
        unsafe { cache.record(&mut device, PairType::EdgeEdge, &pairs(500)) }
            .expect("the pairs record");
        assert!(!cache.finish_detect());
        for kind in PairType::ALL {
            assert!(
                cache.replay(kind).is_none(),
                "{} still offered a replay after another type overflowed",
                kind.name()
            );
        }
    }

    #[test]
    fn one_overflow_is_paid_once_rather_than_every_step_after_it() {
        // ONE DEVICE PER FIXTURE: the cache holds device allocations.
        let mut device = device();
        // The growth rule: twice the measured requirement. Without it the same
        // scene re-walks the BVH on every step, which is the 24 ms to 296 ms
        // regression this cache is shaped to avoid.
        let mut cache = PairCache::new();
        cache.begin_detect(&mut device).expect("the cache sizes");
        unsafe { cache.record(&mut device, PairType::EdgeEdge, &pairs(700)) }
            .expect("the pairs record");
        assert!(!cache.finish_detect());
        assert_eq!(cache.fallback_count(), 1);

        cache.begin_detect(&mut device).expect("the cache sizes");
        assert_eq!(
            cache.capacity(PairType::EdgeEdge),
            1400,
            "the buffer must grow to twice the measured requirement"
        );
        unsafe { cache.record(&mut device, PairType::EdgeEdge, &pairs(700)) }
            .expect("the pairs record");
        assert!(cache.finish_detect(), "the second step must fit");
        assert_eq!(cache.fallback_count(), 1, "the fallback was paid twice");
        assert_eq!(cache.replay(PairType::EdgeEdge).unwrap().len(), 1400);
    }

    #[test]
    fn a_fresh_detection_clears_the_previous_overflow() {
        // ONE DEVICE PER FIXTURE: the cache holds device allocations.
        let mut device = device();
        let mut cache = PairCache::new();
        cache.begin_detect(&mut device).expect("the cache sizes");
        unsafe { cache.record(&mut device, PairType::EdgeEdge, &pairs(50)) }
            .expect("the pairs record");
        assert!(!cache.finish_detect());
        cache.begin_detect(&mut device).expect("the cache sizes");
        unsafe { cache.record(&mut device, PairType::EdgeEdge, &pairs(3)) }
            .expect("the pairs record");
        assert!(cache.finish_detect());
        assert_eq!(cache.replay(PairType::EdgeEdge).unwrap(), &pairs(3)[..]);
    }

    #[test]
    fn a_shorter_detection_at_an_unchanged_capacity_replays_only_its_own_pairs() {
        // ONE DEVICE PER FIXTURE: the cache holds device allocations.
        let mut device = device();
        // THE PROPERTY THAT LICENSES NOT CLEARING THE PAIR BUFFER. The
        // reference clears its counter and its overflow flag and leaves the
        // four pair arrays alone (`contact.cu:1496-1497`), and this is why that
        // is safe rather than lucky: a reader is handed exactly the slots this
        // detection claimed, so a shorter detection cannot see the tail of a
        // longer one however stale that tail is. Both detections below run at
        // the SAME capacity, which is the case a per-detection zero fill would
        // otherwise be hiding.
        let mut cache = PairCache::new();
        cache.begin_detect(&mut device).expect("the cache sizes");
        unsafe { cache.record(&mut device, PairType::EdgeEdge, &pairs(6)) }
            .expect("the pairs record");
        cache.finish_detect();

        cache.begin_detect(&mut device).expect("the cache sizes");
        let capacity = cache.capacity(PairType::EdgeEdge);
        assert!(capacity >= 6, "the warm-up must have sized the buffer");
        unsafe { cache.record(&mut device, PairType::EdgeEdge, &pairs(2)) }
            .expect("the pairs record");
        assert!(cache.finish_detect());
        assert_eq!(
            cache.capacity(PairType::EdgeEdge),
            capacity,
            "the capacity must not have moved, or this fixture is not testing \
             the reuse path"
        );
        assert_eq!(
            cache.replay(PairType::EdgeEdge).unwrap(),
            &pairs(2)[..],
            "a replay must hand out this detection's pairs and nothing the \
             previous one left behind"
        );
    }

    #[test]
    fn recording_in_several_calls_matches_recording_in_one() {
        // ONE DEVICE PER FIXTURE: the cache holds device allocations.
        let mut device = device();
        // A traversal arrives in chunks, so the cache must accumulate rather
        // than replace, and the concatenation must be in call order.
        let all = pairs(6);
        let mut split = PairCache::new();
        // One warm-up detection so the buffer is sized for six pairs; the point
        // here is the accumulation, not the growth.
        split.begin_detect(&mut device).expect("the cache sizes");
        unsafe { split.record(&mut device, PairType::PointPoint, &all) }
            .expect("the pairs record");
        split.finish_detect();
        split.begin_detect(&mut device).expect("the cache sizes");
        assert!(split.capacity(PairType::PointPoint) >= 6);
        unsafe { split.record(&mut device, PairType::PointPoint, &all[..4]) }
            .expect("the pairs record");
        unsafe { split.record(&mut device, PairType::PointPoint, &all[4..]) }
            .expect("the pairs record");
        assert!(split.finish_detect());
        assert_eq!(split.replay(PairType::PointPoint).unwrap(), &all[..]);
    }

    #[test]
    #[should_panic(expected = "two indices per pair")]
    fn an_odd_pair_list_is_refused_rather_than_truncated() {
        // ONE DEVICE PER FIXTURE: the cache holds device allocations.
        let mut device = device();
        let mut cache = PairCache::new();
        cache.begin_detect(&mut device).expect("the cache sizes");
        unsafe { cache.record(&mut device, PairType::PointEdge, &[1, 2, 3]) }
            .expect("the pairs record");
    }
}
