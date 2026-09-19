// File: devsort.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! The device sort over `(key, index)` pairs, shared by every caller that needs
//! one.
//!
//! # A radix sort, as the reference sorts
//!
//! `radix_sort_pairs` is what the reference uses, and this is the same
//! algorithm: four bits a pass, a per-block histogram, an exclusive scan over
//! it, and a stable scatter. It costs `O(k)` passes where a bitonic network
//! costs `O(log^2 n)`, and that difference is the whole reason for the choice.
//!
//! MEASURED: a bitonic `bitonic_step` accounted for 13,547 of this tree's
//! 31,017 kernel launches on ten frames of `drape`, against the reference's
//! 22,129 TOTAL. A network needs neither a histogram nor a ballot and so
//! renders as one body every backend compiles; letting a cooperative body carry
//! a serial twin removes that reason.
//!
//! # The order is total, so it agrees with a stable sort by key
//!
//! A radix pass is stable and the passes run least significant digit first, so
//! equal keys keep ascending index order, which is what the network's
//! `(key, index)` comparator gave and what a stable `sort_by_key` gives.
//!

use ppf_cts_compute::{AllocLabel, Buffer, Device, EncoderExt, Fault, Handle};

/// Threads per group in a radix pass, and why it is not the block width.
///
/// A GROUP HANDLES THIS MANY KEYS, one to a lane, so the block count is
/// `keys / RADIX_LANES` and the group-local histogram is shared by exactly these
/// lanes. 64 is two simdgroups on both device arms, which keeps the cross-warp
/// step of the scatter's rank to a single pass over two entries.
const RADIX_LANES: u32 = 64;

use super::kernels::{RadixHistogramArgs, RadixScatterArgs};

/// Persistent arrays for a stable radix sort over the logical key count.
///
/// PERSISTENT AND OWNED BY THE CALLER, because a sort runs once per tree per
/// step and `Buffer::size` grows only past CAPACITY: a sort no larger than the
/// last allocates nothing. `iota` is the identity permutation, held so the
/// per-step seed is a device copy rather than a host upload; it is refilled only
/// when it grows.
#[derive(Debug, Default)]
pub struct SortScratch {
    pub key: Buffer<u32>,
    pub index: Buffer<u32>,
    pub iota: Buffer<u32>,
    pub iota_len: usize,
    /// The pass's destination pair. A radix pass cannot sort in place: a key's
    /// destination is decided by a prefix over the whole array, so the source
    /// has to stay readable while it is written.
    pub key_alt: Buffer<u32>,
    pub index_alt: Buffer<u32>,
    /// One count per (digit, block), scanned in place into the offsets the
    /// scatter reads.
    pub histogram: Buffer<u32>,
    pub scan: super::scan::ScanScratch,
}

impl SortScratch {
    /// Sort `count` keys read from `source`, and the identity permutation with
    /// them. Returns the sorted key and permutation spans, both `count` long.
    ///
    /// Partial groups participate in every barrier but contribute only valid
    /// keys. Allocation capacity may exceed `count`; it is not sorted.
    ///
    /// # Safety
    /// `source` must name at least `count` keys and outlive the call.
    pub unsafe fn sort<D: Device>(
        &mut self,
        device: &mut D,
        region: &'static str,
        source: Handle,
        count: usize,
    ) -> Result<(Handle, Handle), Fault> {
        let length = count.max(1);
        self.key.size(device, length, AllocLabel("devsort.key"))?;
        self.index.size(device, length, AllocLabel("devsort.index"))?;
        // EVERY ALLOCATION IS SIZED BEFORE ANY SPAN IS TAKEN, and that ordering
        // is the whole of it: `Buffer::size` may GROW the arena and move a
        // block, so a handle taken before a later `size` can address memory the
        // allocator has since handed to something else. Sizing the radix
        // scratch after these two spans made the sort return its own histogram,
        // and the seam says so in as many words at `be_replay`: "a grow may have
        // moved a block".
        const RADIX_BITS: u32 = 4;
        const RADIX_SIZE: u32 = 1 << RADIX_BITS;
        const KEY_BITS: u32 = 32;
        let lanes = RADIX_LANES;
        let blocks = (length as u32).div_ceil(lanes);
        // ONE SLOT MORE THAN THE HISTOGRAM HAS COUNTS, because
        // `ScanScratch::exclusive` writes the exclusive prefix into the first
        // `count` and THE GRAND TOTAL INTO THE NEXT. Sizing it to the counts
        // alone let that write land in whatever the allocator had put after it,
        // which here was `key_alt`: the sort returned its own running total as a
        // key, and pass 1 read it back as the value to sort.
        self.histogram.size(
            device,
            (RADIX_SIZE * blocks) as usize + 1,
            AllocLabel("devsort.histogram"),
        )?;
        self.key_alt.size(device, length, AllocLabel("devsort.key_alt"))?;
        self.index_alt
            .size(device, length, AllocLabel("devsort.index_alt"))?;
        self.scan.size(device, RADIX_SIZE * blocks)?;
        if self.iota_len < length {
            self.iota.size(device, length, AllocLabel("devsort.iota"))?;
            let identity: Vec<u32> = (0..length as u32).collect();
            self.iota.write(device, 0, &identity)?;
            self.iota_len = length;
        }
        if count == 0 {
            return Ok((self.key.span(0, 0), self.index.span(0, 0)));
        }
        device.copy(
            self.key.handle(),
            0,
            source,
            0,
            count * std::mem::size_of::<u32>(),
        )?;
        device.copy(
            self.index.handle(),
            0,
            self.iota.handle(),
            0,
            count * std::mem::size_of::<u32>(),
        )?;

        let key = self.key.span(0, count);
        let permutation = self.index.span(0, count);
        // THE WHOLE NETWORK IN ONE BOUNDARY, which is what makes the sort
        // affordable. `Device::launch` is `run` around a single `elements` and
        // a backend SYNCHRONIZES at the end of every submit, so dispatching
        // each comparator pass on its own stalls the host once per pass: the
        // network is `O(log^2 n)` passes, 210 of them at a million keys, and
        // `PPF_REGION_STATS` measured `lbvh.morton_sort` at 14,128 submits on
        // ten frames of `drape`, 38 percent of every synchronize in the run and
        // the largest single source of them in the tree.
        //
        // NOTHING BETWEEN THE PASSES ASKS THE HOST ANYTHING. Every argument is
        // a loop index or a span fixed before the loop, so the double loop is a
        // pure encoding walk. Consecutive entries of a region are ordered with
        // a full barrier between them on every backend, which is exactly what a
        // bitonic network needs of its passes, so this is the SAME sequence with
        // the stalls removed rather than a reordering.
        // FOUR BITS A PASS, LEAST SIGNIFICANT FIRST, which is what makes the
        // whole sequence stable: each pass preserves the order the pass before
        // it produced, so after the last one the keys are in ascending order and
        // equal keys are still in ascending index order.
        //
        // EIGHT PASSES OVER A 32-BIT KEY, each a histogram, a scan and a
        // scatter, against the network's `k(k+1)/2` compare-exchange steps: 136
        // for a 65,536-element sort. That ratio is the change.
        // THE PING-PONG. A pass reads one pair and writes the other, so after an
        // even number of passes the answer is back where the caller expects it.
        // `KEY_BITS / RADIX_BITS` is eight, which is even, so no final copy is
        // needed and none is made.
        // ONE SUBMIT FOR THE WHOLE SORT, which is what the network it replaces
        // cost and what a pass-per-submit radix throws away.
        //
        // NOTHING BETWEEN THE PASSES ASKS THE HOST ANYTHING: every argument is a
        // loop index or a span fixed before the loop, and consecutive entries of
        // ONE region are barrier-ordered on every backend, which is exactly what
        // a histogram, its scan and the scatter that reads the scan need of each
        // other. Dispatching them one at a time was measured at 20,038
        // synchronizes against 14,784 for the network, for the same work.
        let scan = &mut self.scan;
        let histogram_buffer = &self.histogram;
        let key_alt = &self.key_alt;
        let index_alt = &self.index_alt;
        device.run(region, |encoder| {
        let mut shift = 0u32;
        while shift < KEY_BITS {
            let even = (shift / RADIX_BITS) % 2 == 0;
            let (keys_in, values_in, keys_out, values_out) = if even {
                (key, permutation, key_alt.span(0, count),
                 index_alt.span(0, count))
            } else {
                (key_alt.span(0, count), index_alt.span(0, count),
                 key, permutation)
            };
            let histogram = histogram_buffer.span(0, (RADIX_SIZE * blocks) as usize);
            let hist_args = RadixHistogramArgs {
                keys: keys_in,
                count: count as u32,
                shift,
                radix_size: RADIX_SIZE,
                radix_mask: RADIX_SIZE - 1,
                block_histograms: histogram,
                block_count: blocks,
                groups: blocks,
                seam_arena_count: 0,
            };
            // Safety: every span outlives the call and the caller owns the
            // scratch for the whole sort.
            // Safety: every span outlives the region and the caller owns the
            // scratch for the whole sort.
            unsafe { encoder.groups(&hist_args, blocks, lanes) }?;
            // THE SCAN IS OVER (digit, block) IN THAT ORDER, which is what makes
            // a block's destinations disjoint from every other block's: the
            // prefix of digit d block b counts every key of a smaller digit and
            // every key of the same digit in an earlier block.
            // Safety: the histogram outlives the call.
            // Safety: the histogram outlives the region, and the scratch was
            // sized above: allocation cannot happen inside a region.
            unsafe {
                scan.encode_exclusive(encoder, histogram, RADIX_SIZE * blocks)
            }?;
            let scatter_args = RadixScatterArgs {
                keys_in,
                values_in,
                keys_out,
                values_out,
                count: count as u32,
                shift,
                radix_size: RADIX_SIZE,
                radix_mask: RADIX_SIZE - 1,
                global_offsets: histogram,
                block_count: blocks,
                groups: blocks,
                seam_arena_count: 0,
            };
            // Safety: as above; `keys_in` and `keys_out` are different spans.
            // Safety: as above; `keys_in` and `keys_out` are different spans.
            unsafe { encoder.groups(&scatter_args, blocks, lanes) }?;
            shift += RADIX_BITS;
        }
        Ok(())
        })?;
        // THE ONE CHECK THAT READS THE FORM THE DEVICE ACTUALLY RAN.
        //
        // `radix_scatter` has a COOPERATIVE body and a SERIAL TWIN under rule
        // (1-LANE), and only the twin is what `cargo test` exercises: the host
        // backend renders the serial form, so every unit test of this sort has
        // tested the half that CUDA and Metal do not run. The ballot form was
        // checked against a MODEL of itself and against the twin's output in
        // that model, which is not the same as checking the code the GPU
        // executes.
        //
        // Ascending AND stable, because a radix pass's whole contract is
        // stability: an unstable scatter still sorts and would reorder the
        // Morton keys, which changes every contact pair the BVH finds without
        // any of it looking wrong.
        if std::env::var_os("PPF_SORT_VERIFY").is_some() {
            let mut keys = vec![0u32; count];
            let mut order = vec![0u32; count];
            self.key.read(device, 0, &mut keys)?;
            self.index.read(device, 0, &mut order)?;
            for i in 1..count {
                let ascending = keys[i - 1] < keys[i]
                    || (keys[i - 1] == keys[i] && order[i - 1] < order[i]);
                if !ascending {
                    return Err(Fault::Shape {
                        kernel: "devsort.verify",
                        detail: format!(
                            "the sort is not ascending and stable at slot {i}: \
                             ({}, {}) then ({}, {}). A radix pass that loses \
                             stability still sorts, and reorders the Morton \
                             keys, so every contact pair the BVH finds changes \
                             with nothing looking wrong",
                            keys[i - 1],
                            order[i - 1],
                            keys[i],
                            order[i]
                        ),
                    });
                }
            }
        }
        Ok((self.key.span(0, count), self.index.span(0, count)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;

    /// The sort's own test, which it did not have.
    ///
    /// IT HAD NONE AND THAT IS WHY THE RADIX SWITCH FAILED IN A GALERKIN
    /// COARSENING TEST rather than here. The only coverage was a caller three
    /// layers up, whose message named a split or merged run; a wrong
    /// permutation reaches it as a wrong dedup count and says nothing about
    /// which key moved.
    fn sorted(keys: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut device = host_device();
        let mut scratch = SortScratch::default();
        let mut source = ppf_cts_compute::Buffer::<u32>::none();
        source
            .size(&mut device, keys.len().max(1), AllocLabel("test.sort.source"))
            .expect("the source sizes");
        source.write(&mut device, 0, keys).expect("the source uploads");
        // Safety: the source outlives the call and names `keys.len()` keys.
        let (key, index) = unsafe {
            scratch
                .sort(&mut device, "test.sort", source.handle(), keys.len())
                .expect("the sort runs")
        };
        let mut got_key = vec![0u32; keys.len()];
        let mut got_index = vec![0u32; keys.len()];
        if !keys.is_empty() {
            scratch.key.read(&mut device, 0, &mut got_key).expect("keys read back");
            scratch
                .index
                .read(&mut device, 0, &mut got_index)
                .expect("permutation reads back");
        }
        let _ = (key, index);
        (got_key, got_index)
    }

    #[test]
    fn the_device_sort_is_ascending_and_stable() {
        // EQUAL KEYS ARE THE POINT: a radix pass is stable and the passes run
        // least significant digit first, so equal keys must keep ascending
        // index order. An unstable sort passes an "is it ascending" check and
        // fails this one.
        let keys: Vec<u32> = (0..1000u32)
            .map(|i| ((i.wrapping_mul(2_654_435_761) >> 4) % 97))
            .collect();
        let (got_key, got_index) = sorted(&keys);
        let mut want: Vec<(u32, u32)> =
            keys.iter().copied().zip(0u32..).collect();
        want.sort_by_key(|(k, i)| (*k, *i));
        for (slot, (k, i)) in want.iter().enumerate() {
            assert_eq!(
                (got_key[slot], got_index[slot]),
                (*k, *i),
                "slot {slot}: the device sort gave ({}, {}) against ({k}, {i})",
                got_key[slot],
                got_index[slot]
            );
        }
    }

    #[test]
    fn the_device_sort_handles_lengths_around_a_block() {
        // A BLOCK IS `RADIX_LANES` KEYS, so the interesting lengths are the ones
        // that do not fill one, fill exactly one, and spill into a second.
        for count in [0usize, 1, 63, 64, 65, 127, 128, 129] {
            let keys: Vec<u32> =
                (0..count as u32).map(|i| (count as u32 - i) * 7 % 251).collect();
            let (got_key, _) = sorted(&keys);
            let mut want = keys.clone();
            want.sort_unstable();
            assert_eq!(got_key, want, "a sort of {count} keys came back wrong");
        }
    }

    #[test]
    fn radix_stability_survives_partial_groups_and_scratch_reuse() {
        let mut device = host_device();
        let mut scratch = SortScratch::default();
        let mut source = Buffer::<u32>::none();
        source.size(&mut device, 4097, AllocLabel("test.sort.source")).unwrap();
        for count in [0usize, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                      255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097,
                      65, 1, 0, 129] {
            let keys: Vec<u32> = (0..count)
                .map(|i| if i % 3 == 0 { u32::MAX } else { (i % 7) as u32 })
                .collect();
            source.write(&mut device, 0, &keys).unwrap();
            // Safety: source and scratch remain alive throughout the sort.
            unsafe { scratch.sort(&mut device, "test.sort.reuse", source.handle(), count) }
                .unwrap();
            let mut got_key = vec![0; count];
            let mut got_index = vec![0; count];
            scratch.key.read(&mut device, 0, &mut got_key).unwrap();
            scratch.index.read(&mut device, 0, &mut got_index).unwrap();
            let mut want: Vec<_> = keys.into_iter().zip(0u32..).collect();
            want.sort_unstable();
            assert_eq!(got_key.into_iter().zip(got_index).collect::<Vec<_>>(),
                       want, "count={count}");
        }
    }

    #[test]
    #[ignore = "release timing experiment"]
    fn benchmark_contact_radix_lengths() {
        let mut device = host_device();
        let mut scratch = SortScratch::default();
        let mut source = Buffer::<u32>::none();
        source.size(&mut device, 131073, AllocLabel("test.sort.source")).unwrap();
        for count in [32768usize, 32769, 65535, 65536, 65537, 131073] {
            let keys: Vec<u32> = (0..count as u32)
                .map(|i| if i % 7 == 0 { u32::MAX } else { i.wrapping_mul(2654435761) })
                .collect();
            source.write(&mut device, 0, &keys).unwrap();
            let mut samples = Vec::new();
            for batch in 0..7 {
                let start = std::time::Instant::now();
                for _ in 0..5 {
                    // Safety: source and scratch outlive all dispatches.
                    unsafe { scratch.sort(&mut device, "test.sort.bench", source.handle(), count) }
                        .unwrap();
                }
                if batch > 0 {
                    samples.push(start.elapsed().as_secs_f64() * 1000.0 / 5.0);
                }
            }
            let mut actual = vec![0; count];
            scratch.key.read(&mut device, 0, &mut actual).unwrap();
            let mut order = vec![0; count];
            scratch.index.read(&mut device, 0, &mut order).unwrap();
            let mut want: Vec<_> = keys.into_iter().zip(0u32..).collect();
            want.sort_unstable();
            assert_eq!(actual.into_iter().zip(order).collect::<Vec<_>>(), want);
            samples.sort_by(f64::total_cmp);
            eprintln!("radix count={count} median_ms={:.6} min_ms={:.6} max_ms={:.6}",
                      samples[3], samples[0], samples[5]);
        }
    }
}
