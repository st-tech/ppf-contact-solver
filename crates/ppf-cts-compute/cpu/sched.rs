// File: crates/ppf-cts-compute/cpu/sched.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Scheduling policy: how work is cut up and how many threads run it.
//!
//! Scheduling computes no values, so this is skeleton and has no shared body.
//! It exists as a module rather than as constants beside each kernel because the
//! policy is one decision applied in many places, and two copies of a policy
//! drift the way two copies of a kernel do, just more quietly: nothing fails,
//! the backend merely stops being tuned the way its measurements said.
//!
//! # The chunk rule, and why it is a rule rather than a number
//!
//! Measured on this project's reference host: a rayon region costs about 3
//! microseconds at its floor and 8 to 17 for a real one. So a chunk must carry
//! enough work to dwarf that, and few enough items that the tail does not idle
//! most of the pool. The rule that follows is **size a chunk so one call carries
//! roughly 10 to 30 microseconds of work**, which for a body costing `c`
//! nanoseconds per item is `10_000 / c` to `30_000 / c` items.
//!
//! Both ends were measured on the elastic body at about 237 ns per face:
//! 64 to 256 faces per chunk was the optimum (6.3x to 6.5x on 8 threads), 16 was
//! worse from region overhead, and 65536 was much worse from load imbalance,
//! because three chunks cannot fill eight threads. A number would encode the
//! elastic body's cost into kernels that cost something else; the rule scales.
//!
//! # Threads are capped per phase, not globally
//!
//! Also measured: DRAM saturates at 2 to 4 threads on the reference host, so a
//! bandwidth-bound phase gains nothing above 4 and measurably loses at 8 (44.1
//! GB/s at 4 against 26.5 at 8). A compute-bound phase does not have that
//! ceiling: the elastic assembly scaled 7.13x on 8 threads. So the cap belongs
//! to the phase and there is no single right thread count for the backend.

// These are P1 components, landed ahead of the Newton driver that will call
// them. The allow is removed in the change that wires the driver up.
#![allow(dead_code)]

/// What limits a phase, which is what decides its thread cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// Limited by memory bandwidth: a vector op, a SpMV, a buffer copy. More
    /// threads past the memory system's saturation point cost rather than gain.
    Bandwidth,
    /// Limited by arithmetic: the elastic and contact bodies, the eigensolvers.
    /// These scale with cores.
    Compute,
    /// Limited by dependent loads: a BVH descent. Scaling comes from running
    /// independent queries, not from more threads on one.
    Latency,
}

/// Threads to use for a phase, given what limits it.
///
/// Never more than the pool has. The bandwidth cap is the measured saturation
/// point rather than a guess, and it is deliberately not raised on a machine
/// with more cores: the limit is the memory system, so more cores do not move
/// it.
pub fn threads_for(bound: Bound, available: usize) -> usize {
    let available = available.max(1);
    match bound {
        Bound::Bandwidth => available.min(4),
        Bound::Compute => available,
        Bound::Latency => available,
    }
}

/// Items per chunk for a body costing `nanos_per_item`.
///
/// Applies the rule above: target the middle of the 10 to 30 microsecond band,
/// which is 20 microseconds of work. Clamped so a very cheap body does not
/// produce an unbounded chunk and a very expensive one still yields at least
/// one item.
pub fn chunk_for(nanos_per_item: f64, total_items: usize) -> usize {
    const TARGET_NANOS: f64 = 20_000.0;
    const MAX_CHUNK: usize = 1 << 20;

    if total_items == 0 {
        return 1;
    }
    let cost = nanos_per_item.max(0.001);
    let ideal = (TARGET_NANOS / cost).round() as usize;
    ideal.clamp(1, MAX_CHUNK).min(total_items.max(1))
}

/// Below this much total work, run serially: the region costs more than the job.
///
/// Expressed in nanoseconds of total work rather than in items, because the
/// threshold is about the region's cost and not about the item count. The bound
/// is the measured upper end of a real region, so a job below it cannot repay
/// even one.
pub fn should_run_serially(nanos_per_item: f64, total_items: usize) -> bool {
    const REGION_NANOS: f64 = 17_000.0;
    nanos_per_item.max(0.0) * total_items as f64 <= REGION_NANOS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chunk_rule_reproduces_the_measured_optimum() {
        // The elastic body was measured at about 237 ns per face, and 64 to 256
        // faces per chunk measured best. The rule must land inside that band
        // without being told about it.
        let chunk = chunk_for(237.0, 200_000);
        assert!(
            (64..=256).contains(&chunk),
            "the rule gave {chunk}, outside the measured optimum of 64 to 256"
        );
    }

    #[test]
    fn a_cheaper_body_gets_a_proportionally_larger_chunk() {
        let expensive = chunk_for(237.0, 1_000_000);
        let cheap = chunk_for(2.0, 1_000_000);
        assert!(
            cheap > expensive * 10,
            "a body about 100x cheaper should take a much larger chunk; got \
             {cheap} against {expensive}"
        );
    }

    #[test]
    fn a_chunk_never_exceeds_the_work_available() {
        assert_eq!(chunk_for(0.5, 10), 10);
        assert_eq!(chunk_for(1000.0, 0), 1);
    }

    #[test]
    fn a_chunk_is_never_zero() {
        // A body so expensive that the ideal chunk rounds below one still has to
        // make progress.
        assert_eq!(chunk_for(1.0e9, 1000), 1);
    }

    #[test]
    fn bandwidth_bound_phases_are_capped_where_dram_saturates() {
        // Measured: 44.1 GB/s at 4 threads against 26.5 at 8, so 8 is worse than
        // 4 and the cap is not a conservatism.
        assert_eq!(threads_for(Bound::Bandwidth, 8), 4);
        assert_eq!(threads_for(Bound::Bandwidth, 64), 4);
        // Never more than exist.
        assert_eq!(threads_for(Bound::Bandwidth, 2), 2);
    }

    #[test]
    fn compute_bound_phases_use_every_core() {
        // Measured: the elastic assembly scaled 7.13x on 8 threads.
        assert_eq!(threads_for(Bound::Compute, 8), 8);
        assert_eq!(threads_for(Bound::Compute, 64), 64);
    }

    #[test]
    fn a_thread_count_is_never_zero() {
        assert_eq!(threads_for(Bound::Bandwidth, 0), 1);
        assert_eq!(threads_for(Bound::Compute, 0), 1);
    }

    #[test]
    fn a_job_smaller_than_one_region_runs_serially() {
        // 100 items at 2 ns is 200 ns of work against a region costing up to
        // 17 us: parallelising it is pure loss.
        assert!(should_run_serially(2.0, 100));
        // 200k faces at 237 ns is 47 ms: obviously worth a region.
        assert!(!should_run_serially(237.0, 200_000));
    }
}
