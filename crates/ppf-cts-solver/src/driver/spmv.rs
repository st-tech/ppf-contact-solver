// File: crates/ppf-cts-solver/src/driver/spmv.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The sparse matvec SKELETON.
//!
//! This module owns the decomposition and nothing else. Every value it produces
//! comes from `src/kernels/solver/spmv.kernel.cpp`, which every backend compiles
//! from the same bytes, so there is one implementation of the arithmetic and the
//! backends cannot drift in it.
//!
//! The split, restated because it is the whole architecture in one place:
//!
//! | in C++, shared | here, in Rust |
//! |---|---|
//! | the 3x3 block matvec and the per-row accumulation, including its ORDER | which rows go to which thread, the chunk size, the thread count |
//!
//! Reordering the work INSIDE a row would change the fp32 result, so the shared
//! body does the whole row and this module never reaches inside one. Choosing
//! which rows a thread walks cannot change any row's value, which is why the
//! partition is free and why the result does not depend on the thread count.

// These are P1 components, landed ahead of the Newton driver that will call
// them. Until `advance()` is built they have no caller outside their own tests,
// which is dead code by the compiler's reckoning and deliberate by the plan's:
// each slice lands with its own gates rather than waiting for a driver that
// would then arrive untested. The allow is removed in the change that wires the
// driver up, and it is scoped per module so a genuinely unused item elsewhere
// still surfaces.
#![allow(dead_code)]

use ppf_cts_compute::{Device, Fault};
use super::kernels::FixedCsrProductRowArgs;

// THE TWO ENTRY POINTS THIS MODULE NAMES DIRECTLY, AND THEY ARE ORACLES.
//
// The matvec itself is behind the seam: [`fixed_csr_spmv`] dispatches
// `fixed_csr_apply_row`, and the block-diagonal apply `z = P^-1 r` reaches
// the same `mat3_mul` body through its own generated entry point, which is
// `id::MAT3_MUL` in the kernel table. So the BODY is dispatched; what these two
// declarations add is a way to call it on ONE block and one vector, with no
// extent and no thread index.
//
// That is what a test needs and is the reason they exist: the tests below check
// the row applier against the block matvec it is built from, and a comparison
// whose reference travelled through the same table lookup, record check and
// launch as the thing under test would be checking that machinery against
// itself. Routing them through the seam to lower a symbol count would cost the
// independence and buy nothing, since the body is already reached from
// production only through a dispatch.
extern "C" {
    fn mat3_mul_abi(matrix: *const f32, vector: *const f32, out: *mut f32);
    fn mat3_transpose_mul_abi(matrix: *const f32, vector: *const f32, out: *mut f32);
}

/// One fixed-pattern matrix in the flat form the skeleton walks.
pub struct FixedCsrView {
    pub index: ppf_cts_compute::Handle,
    pub offset: ppf_cts_compute::Handle,
    pub value: ppf_cts_compute::Handle,
    pub transpose_pair: ppf_cts_compute::Handle,
    pub transpose_offset: ppf_cts_compute::Handle,
    pub rows: u32,
}

/// `result = M x` for the fixed-pattern matrix, over all rows.
///
/// THE COST PER ROW AND THE PARTITION ARE THE BACKEND'S, and this function no
/// longer states either. `fixed_csr_apply_row`'s row in the kernel table
/// carries the nanoseconds per item, `Scatter::Disjoint` is what permits the
/// cut, and `sched::chunk_for` inside the backend turns the first into a chunk
/// width. Stating the cost or the partition again here would put a second copy
/// of that policy where nothing compares the two.
///
/// # Panics
/// If `result` or `x` is not `3 * rows` long, or an offset array is short. The
/// check is here rather than in the entry point because a length mismatch is a
/// caller defect and this is the side that knows the lengths.
///
/// # Safety
/// Every slice must outlive the dispatch.
pub unsafe fn fixed_csr_spmv<D: Device>(
    device: &mut D,
    matrix: &FixedCsrView,
    // THE TWO VECTORS AS HANDLES. Production reaches this arithmetic through
    // `Operator::apply`, whose `x` and `result` are the PCG's own device
    // buffers; this entry is the one a test drives directly, and it takes the
    // same form so the two cannot drift.
    x: ppf_cts_compute::Handle,
    result: ppf_cts_compute::Handle,
    rows_hint: usize,
) -> Result<(), Fault> {
    let rows = matrix.rows;
    let expected = 3 * rows as usize;
    assert_eq!(rows_hint, expected, "x and result must be 3 * rows long");
    assert_eq!(x.size as usize, expected, "x must be 3 * rows long");
    assert_eq!(result.size as usize, expected, "result must be 3 * rows long");
    // THE TWO OFFSET LENGTHS ARE STILL CHECKED HERE, against the handle rather
    // than a slice. A `Handle` carries `size` in ELEMENTS, so the migration took
    // nothing away, and `fixed_csr_apply_row` declares no `[[seam::bound]]`, so
    // dropping these would leave the invariant checked on NEITHER side: this
    // kernel walks `offset[row] .. offset[row + 1]`, a range that is data rather
    // than a thread index, which is exactly the shape no device-side bound can
    // reach.
    assert_eq!(
        matrix.offset.size as usize,
        rows as usize + 1,
        "offset must be rows + 1 long"
    );
    assert_eq!(
        matrix.transpose_offset.size as usize,
        rows as usize + 1,
        "transpose_offset must be rows + 1 long"
    );
    if rows == 0 {
        return Ok(());
    }

    let args = FixedCsrProductRowArgs {
        index: matrix.index,
        offset: matrix.offset,
        value: matrix.value,
        transpose_pair: matrix.transpose_pair,
        transpose_offset: matrix.transpose_offset,
        x,
        result,
        count: rows,
        seam_arena_count: 0,
    };
    device.launch("spmv.fixed_csr", &args, rows)?;
    Ok(())
}

/// The shared 3x3 matvec, for callers that need one block rather than a row.
pub fn mat3_mul(matrix: &[f32; 9], vector: &[f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    unsafe { mat3_mul_abi(matrix.as_ptr(), vector.as_ptr(), out.as_mut_ptr()) };
    out
}

/// The shared transposed 3x3 matvec.
pub fn mat3_transpose_mul(matrix: &[f32; 9], vector: &[f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    unsafe { mat3_transpose_mul_abi(matrix.as_ptr(), vector.as_ptr(), out.as_mut_ptr()) };
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;
    use ppf_cts_compute::sched;

    /// Seeding a readback buffer leaves the mirror answerable with no download.
    ///
    /// THE PROPERTY THAT MATTERS IS THE ONE AFTER THE DISPATCH: `handle()` is
    /// how a record names the buffer, and asking for it is the event that
    /// stales the mirror, because the kernel it goes to may write. So a host
    /// reader downloads first, and this checks both halves of that: the seed
    /// answers without one, and a `host()` after a `handle()` traps rather than
    /// answering out of the contents from before the dispatch.
    #[test]
    fn a_seeded_readback_buffer_answers_without_a_download() {
        use ppf_cts_compute::{AllocLabel, ReadbackBuffer};
        let mut device = host_device();
        let mut buffer: ReadbackBuffer<i32> = ReadbackBuffer::default();
        buffer
            .size(&mut device, 4, AllocLabel("test.seed"))
            .unwrap();
        buffer.seed(&mut device, &[7, 8, 9, 10]).expect("the seed runs");
        assert_eq!(buffer.host(), &[7, 8, 9, 10]);

        // The device half took it too, which a download proves.
        let _ = buffer.handle();
        buffer.download(&mut device).unwrap();
        assert_eq!(buffer.host(), &[7, 8, 9, 10]);
    }

    /// A HOST READ AFTER A DISPATCH MUST TRAP, not answer stale. This is what
    /// makes an explicit download the only way to read the positions on the
    /// host, and therefore what keeps the readback visible where it is paid.
    #[test]
    #[should_panic(expected = "not downloaded since")]
    fn reading_a_readback_buffer_after_a_dispatch_named_it_traps() {
        use ppf_cts_compute::{AllocLabel, ReadbackBuffer};
        let mut device = host_device();
        let mut buffer: ReadbackBuffer<i32> = ReadbackBuffer::default();
        buffer
            .size(&mut device, 4, AllocLabel("test.seed"))
            .unwrap();
        buffer.seed(&mut device, &[7, 8, 9, 10]).unwrap();
        let _ = buffer.handle();
        let _ = buffer.host();
    }

    /// A readback-to-readback copy moves the device half and STALES the mirror.
    ///
    /// This is the shape the position commit takes: previous gets current, then
    /// current gets the iterate, both entirely on the device. The mirror going
    /// stale is the correct half of it, since the device half changed and the
    /// host half did not, so `fetch` must download before it reads.
    #[test]
    fn a_readback_copy_moves_the_device_half_and_stales_the_mirror() {
        use ppf_cts_compute::{AllocLabel, ReadbackBuffer};
        let mut device = host_device();
        let mut src: ReadbackBuffer<i32> = ReadbackBuffer::default();
        let mut dst: ReadbackBuffer<i32> = ReadbackBuffer::default();
        src.size(&mut device, 4, AllocLabel("test.rb.src")).unwrap();
        dst.size(&mut device, 4, AllocLabel("test.rb.dst")).unwrap();
        src.seed(&mut device, &[7, 8, 9, 10]).unwrap();
        dst.seed(&mut device, &[-1, -1, -1, -1]).unwrap();

        dst.copy_from(&mut device, &src).expect("the copy runs");

        // The mirror is stale, so the answer comes from a download.
        dst.download(&mut device).unwrap();
        assert_eq!(dst.host(), &[7, 8, 9, 10]);
    }

    /// The device-to-device copy verb, which no dispatch exercises.
    ///
    /// `write` moves host to device and `read` moves device to host; a driver
    /// that owns its buffers also needs device to device, and without it a seed
    /// round-trips through the host, which is two transfers to do no work.
    ///
    /// THE DESTINATION IS POISONED FIRST, so a copy that silently did nothing
    /// fails here rather than passing on the allocator's zeroes.
    #[test]
    fn a_device_to_device_copy_moves_the_source_bytes() {
        use ppf_cts_compute::{AllocLabel, Buffer};
        let mut device = host_device();
        let mut src: Buffer<i32> = Buffer::none();
        let mut dst: Buffer<i32> = Buffer::none();
        src.size(&mut device, 4, AllocLabel("test.copy.src")).unwrap();
        dst.size(&mut device, 4, AllocLabel("test.copy.dst")).unwrap();
        src.write(&mut device, 0, &[7, 8, 9, 10]).unwrap();
        dst.write(&mut device, 0, &[-1, -1, -1, -1]).unwrap();

        dst.copy_from(&mut device, &src).expect("the copy runs");

        let mut back = [0i32; 4];
        dst.read(&mut device, 0, &mut back).unwrap();
        assert_eq!(back, [7, 8, 9, 10]);
    }

    /// A SHORT COPY MUST TRAP RATHER THAN TRUNCATE: it would leave the tail of
    /// the destination holding what it held before, which reads as a plausible
    /// result rather than as an error.
    #[test]
    #[should_panic(expected = "same length")]
    fn a_device_copy_between_different_lengths_is_refused() {
        use ppf_cts_compute::{AllocLabel, Buffer};
        let mut device = host_device();
        let mut src: Buffer<i32> = Buffer::none();
        let mut dst: Buffer<i32> = Buffer::none();
        src.size(&mut device, 4, AllocLabel("test.copy.src")).unwrap();
        dst.size(&mut device, 2, AllocLabel("test.copy.dst")).unwrap();
        let _ = dst.copy_from(&mut device, &src);
    }

    /// A diagonal-plus-one-offdiagonal matrix, big enough to cross both the
    /// serial threshold and several chunks.
    fn fixture(rows: u32) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>) {
        let n = rows as usize;
        let mut index = Vec::new();
        let mut offset = vec![0u32];
        let mut value = Vec::new();
        // Upper triangle stored: row i keeps (i, i) and (i, i+1).
        for i in 0..n {
            index.push(i as u32);
            let d = 1.0 + (i % 7) as f32 * 0.25;
            value.extend_from_slice(&[d, 0.1, 0.0, 0.1, d, 0.2, 0.0, 0.2, d]);
            if i + 1 < n {
                index.push(i as u32 + 1);
                let o = 0.5 - (i % 5) as f32 * 0.05;
                value.extend_from_slice(&[o, 0.0, 0.03, 0.0, o, 0.0, 0.03, 0.0, o]);
            }
            offset.push(index.len() as u32);
        }
        // The transpose table: row j names every (source_row, slot) that stored
        // a block in column j. Only the off-diagonals contribute.
        let mut transpose_pair = Vec::new();
        let mut transpose_offset = vec![0u32];
        for j in 0..n {
            if j > 0 {
                let source = j - 1;
                let slot = offset[source] as usize + 1;
                transpose_pair.push(source as u32);
                transpose_pair.push(slot as u32);
            }
            transpose_offset.push((transpose_pair.len() / 2) as u32);
        }
        let x: Vec<f32> = (0..3 * n).map(|k| ((k % 13) as f32 - 6.0) * 0.37).collect();
        (index, offset, value, transpose_pair, transpose_offset, x)
    }

    /// One host array on the device, which is what a handle-taking view needs.
    fn on_device<T: ppf_cts_compute::Pod + Default>(
        device: &mut impl Device,
        host: &[T],
        label: &'static str,
    ) -> ppf_cts_compute::Buffer<T> {
        let mut buffer = ppf_cts_compute::Buffer::<T>::none();
        buffer
            .size(device, host.len(), ppf_cts_compute::AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
            .write(device, 0, host)
            .expect("the test upload succeeds");
        buffer
    }

    fn run(rows: u32) -> Vec<f32> {
        let (index, offset, value, transpose_pair, transpose_offset, x) = fixture(rows);
        // ONE DEVICE for the uploads and the dispatch alike.
        let mut device = host_device();
        let index_d = on_device(&mut device, &index, "test.index");
        let offset_d = on_device(&mut device, &offset, "test.offset");
        let value_d = on_device(&mut device, &value, "test.value");
        let tp_d = on_device(&mut device, &transpose_pair, "test.tpair");
        let to_d = on_device(&mut device, &transpose_offset, "test.toffset");
        let view = FixedCsrView {
            index: index_d.span(0, index_d.len()),
            offset: offset_d.span(0, offset_d.len()),
            value: value_d.span(0, value_d.len()),
            transpose_pair: tp_d.span(0, tp_d.len()),
            transpose_offset: to_d.span(0, to_d.len()),
            rows,
        };
        let mut result = vec![0.0f32; 3 * rows as usize];
        let x_d = on_device(&mut device, &x, "test.x");
        let mut out_d = on_device(&mut device, &result, "test.result");
        // Safety: every handle names a live allocation on this device.
        unsafe {
            fixed_csr_spmv(
                &mut device,
                &view,
                x_d.span(0, x.len()),
                out_d.span(0, result.len()),
                x.len(),
            )
        }
        .expect("the fixed-pattern matvec dispatches");
        out_d
            .read(&mut device, 0, &mut result)
            .expect("the result reads back");
        result
    }

    /// The property the whole split rests on: the decomposition is free.
    ///
    /// ONE UNPARTITIONED CALL IS NO LONGER SOMETHING THIS SIDE CAN ASK FOR, and
    /// that is the point of the seam rather than a hole in this test: the
    /// backend decides whether a range is cut and how wide the pieces are. What
    /// it decides on is total work, so the same fixture at two SIZES straddles
    /// the decision: 3000 rows is under `sched`'s region threshold and runs as
    /// one serial pass, 8192 rows is over it and is cut into chunks that run in
    /// parallel.
    ///
    /// The two are comparable because the fixture is row-local. Row `i` holds a
    /// diagonal block keyed on `i % 7` and one off-diagonal keyed on `i % 5`,
    /// its transposed entry names row `i - 1`, and `x[k]` depends only on `k`,
    /// so for `0 < i < rows - 1` the answer at row `i` does not depend on how
    /// many rows follow it. The comparison therefore runs over the interior of
    /// the smaller problem, and a mismatch means the partition reached inside a
    /// row.
    #[test]
    fn chunking_does_not_change_a_single_bit() {
        let serial = run(3000);
        let chunked = run(8192);
        assert!(
            sched::should_run_serially(5.0, 3000) && !sched::should_run_serially(5.0, 8192),
            "the two sizes no longer straddle the backend's serial threshold, so \
             this test compares two runs of the same path"
        );
        // The interior: row 0 has no transposed entry and the last row has no
        // off-diagonal, so neither is row-local.
        for i in 3..(3 * 2999) {
            let (a, b) = (serial[i], chunked[i]);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i} differs: serial {a} against chunked {b}. The \
                 partition is not allowed to change a value."
            );
        }
        assert!(
            serial.iter().any(|v| *v != 0.0),
            "the fixture produced an all-zero result, so the comparison was vacuous"
        );
    }

    /// The result must not depend on how many threads rayon happens to use.
    #[test]
    fn the_answer_does_not_depend_on_the_thread_count() {
        let rows = 8192;
        let reference = run(rows);
        for threads in [1usize, 2, 3, 7] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| run(rows));
            for (i, (a, b)) in reference.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "element {i} moved at {threads} threads; determinism is a \
                     stated guarantee, not an accident of scheduling"
                );
            }
        }
    }

    /// The row applier must be the block matvec applied to the row, so a row
    /// with one diagonal block equals one `mat3_mul`. This checks the shim wires
    /// the shared body up correctly rather than checking the arithmetic, which
    /// has one implementation and is CUDA's to validate.
    #[test]
    fn a_one_block_row_equals_the_shared_block_matvec() {
        let matrix = [1.5f32, 0.25, -0.5, 0.25, 2.0, 0.75, -0.5, 0.75, 3.25];
        let x = [0.5f32, -1.25, 2.0];
        let index = [0u32];
        let offset = [0u32, 1];
        let transpose_offset = [0u32, 0];
        let mut device = host_device();
        let index_d = on_device(&mut device, &index, "test.index");
        let offset_d = on_device(&mut device, &offset, "test.offset");
        let value_d = on_device(&mut device, &matrix, "test.value");
        let tp_d = on_device(&mut device, &[0u32; 0], "test.tpair");
        let to_d = on_device(&mut device, &transpose_offset, "test.toffset");
        let view = FixedCsrView {
            index: index_d.span(0, index_d.len()),
            offset: offset_d.span(0, offset_d.len()),
            value: value_d.span(0, value_d.len()),
            transpose_pair: tp_d.span(0, tp_d.len()),
            transpose_offset: to_d.span(0, to_d.len()),
            rows: 1,
        };
        let mut result = vec![0.0f32; 3];
        let x_d = on_device(&mut device, &x, "test.x");
        let mut out_d = on_device(&mut device, &result, "test.result");
        // Safety: every handle names a live allocation on this device.
        unsafe {
            fixed_csr_spmv(
                &mut device,
                &view,
                x_d.span(0, x.len()),
                out_d.span(0, result.len()),
                x.len(),
            )
        }
        .expect("the fixed-pattern matvec dispatches");
        out_d
            .read(&mut device, 0, &mut result)
            .expect("the result reads back");
        let direct = mat3_mul(&matrix, &x);
        for k in 0..3 {
            assert_eq!(result[k].to_bits(), direct[k].to_bits());
        }
    }

    /// The transposed form is the transpose, checked against the forward form on
    /// a symmetric matrix where the two must agree exactly.
    #[test]
    fn transpose_mul_agrees_with_mul_on_a_symmetric_block() {
        let symmetric = [2.0f32, 0.5, -1.0, 0.5, 3.0, 0.25, -1.0, 0.25, 4.0];
        let v = [1.0f32, -2.0, 0.5];
        let a = mat3_mul(&symmetric, &v);
        let b = mat3_transpose_mul(&symmetric, &v);
        for k in 0..3 {
            assert_eq!(a[k].to_bits(), b[k].to_bits());
        }
    }
}
