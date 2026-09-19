//! The assembled-system dump, env-gated by `PPF_DUMP_LINSYS=<k>`: on the k-th
//! solve (0-based) it writes the assembled Newton system `M x = b`, with
//! `M = A_dyn + B_fixed + C_diag` stored as the upper-triangle off-diagonal COO
//! plus one 3x3 block per row, to `$PPF_DUMP_DIR/linsys_<k>.bin`, and then exits
//! 0. Off by default, and a diagnostic rather than a production path: two builds
//! are pointed at the same solve index and their dumps compared, which is how a
//! divergence in the ASSEMBLED SYSTEM is told from one in the SOLVER. The file
//! layout is the one this module writes below, and nothing else reads it. The
//! COO pass runs on the device through the `dump_linsys_row_to_coo` entry
//! (`src/kernels/main/dump_linsys.kernel.cpp`); this file is its host half.
use std::io::Write;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::OnceLock;

use ppf_cts_compute::{AllocLabel, Device, Handle, ReadbackBuffer};

use super::kernels::DumpLinsysRowToCooArgs;
use super::operator::Operator;
use super::scene::FatalResult;

const MAGIC: u32 = 0x4C53_5953;

fn target() -> i64 {
    static TARGET: OnceLock<i64> = OnceLock::new();
    *TARGET.get_or_init(|| {
        std::env::var("PPF_DUMP_LINSYS")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(-1)
    })
}

static COUNTER: AtomicI64 = AtomicI64::new(0);

/// Called once per solve, immediately before PCG on both the locked and the
/// unlocked path (`step.rs`); a no-op unless this is the requested solve, in
/// which case the process ends here after writing the file.
pub fn maybe_dump<D: Device>(
    device: &mut D,
    operator: &Operator,
    force: Handle,
    eval_x: Handle,
    vertices: usize,
) -> FatalResult<()> {
    let target = target();
    if target < 0 {
        return Ok(());
    }
    let solve = COUNTER.fetch_add(1, Ordering::SeqCst);
    if solve != target {
        return Ok(());
    }
    let n = vertices;
    let fixed_nnz = operator.fixed.value.size as usize / 9;
    let dyn_nnz = operator
        .dynamic
        .as_ref()
        .map(|d| d.value.size as usize / 9)
        .unwrap_or(0);
    // Upper-bound the off-diagonal COO by the total stored nnz of both CSRs.
    // Every entry the passes emit comes from a stored block of one of the two,
    // so the bound cannot be exceeded, and `.max(1)` keeps the allocation legal
    // when both are empty.
    let capacity = (fixed_nnz + dyn_nnz).max(1);

    let mut cursor = ReadbackBuffer::<u32>::default();
    cursor.size(device, 1, AllocLabel("dump_linsys.cursor"))?;
    cursor.seed(device, &[0u32])?;
    let mut out_row = ReadbackBuffer::<u32>::default();
    out_row.size(device, capacity, AllocLabel("dump_linsys.row"))?;
    let mut out_column = ReadbackBuffer::<u32>::default();
    out_column.size(device, capacity, AllocLabel("dump_linsys.column"))?;
    let mut out_block = ReadbackBuffer::<f32>::default();
    out_block.size(device, 9 * capacity, AllocLabel("dump_linsys.block"))?;
    // The diagonal opens as C, and the two passes add each CSR's own diagonal
    // blocks into it, so what is written out is the full diagonal of `M`.
    let mut diagonal = ReadbackBuffer::<f32>::default();
    diagonal.size(device, 9 * n, AllocLabel("dump_linsys.diagonal"))?;
    device.copy(diagonal.handle(), 0, operator.diagonal, 0, 36 * n)?;
    let mut rhs = ReadbackBuffer::<f32>::default();
    rhs.size(device, 3 * n, AllocLabel("dump_linsys.rhs"))?;
    device.copy(rhs.handle(), 0, force, 0, 12 * n)?;
    // THE ITERATE TOO, as raw int32 words, so two builds' iterates can be
    // compared bit for bit. It goes to its own `evalx_<k>.bin` below, leaving
    // the `linsys_<k>.bin` layout exactly as the module doc states it.
    let mut iterate = ReadbackBuffer::<i32>::default();
    iterate.size(device, 3 * n, AllocLabel("dump_linsys.iterate"))?;
    device.copy(iterate.handle(), 0, eval_x, 0, 12 * n)?;

    // A CSR offset array has n + 1 entries; the entry reads `row_begin[row]`
    // and `row_end[row]`, so the second view is the same array one element on.
    let one_on = |offset: Handle| Handle {
        arena: offset.arena,
        off: offset.off + 4,
        size: n as u32,
        allocated: n as u32,
    };
    let mut pass = |device: &mut D, offset: Handle, index: Handle, value: Handle| -> FatalResult<()> {
        let args = DumpLinsysRowToCooArgs {
            row_begin: offset,
            row_end: one_on(offset),
            column: index,
            block: value,
            cursor: cursor.handle(),
            out_row: out_row.handle(),
            out_column: out_column.handle(),
            out_block: out_block.handle(),
            diagonal: diagonal.handle(),
            capacity: capacity as u32,
            count: n as u32,
            seam_arena_count: 0,
        };
        // SAFETY: every handle names a live allocation sized for `n` rows or
        // `capacity` entries, which is what the entry's bounds are declared over.
        unsafe { device.launch("step.dump_linsys", &args, n as u32) }?;
        Ok(())
    };
    if let Some(dynamic) = operator.dynamic.as_ref() {
        pass(device, dynamic.offset, dynamic.index, dynamic.value)?;
    }
    pass(device, operator.fixed.offset, operator.fixed.index, operator.fixed.value)?;

    cursor.download(device)?;
    out_row.download(device)?;
    out_column.download(device)?;
    out_block.download(device)?;
    diagonal.download(device)?;
    rhs.download(device)?;
    iterate.download(device)?;
    let n_off = (cursor.host()[0] as usize).min(capacity);

    let dir = std::env::var("PPF_DUMP_DIR").unwrap_or_else(|_| ".".to_string());
    let path = format!("{dir}/linsys_{solve}.bin");
    let file = std::fs::File::create(&path)
        .unwrap_or_else(|e| panic!("dump_linsys: cannot open {path}: {e}"));
    let mut w = std::io::BufWriter::new(file);
    let put_u32 = |w: &mut std::io::BufWriter<std::fs::File>, v: u32| w.write_all(&v.to_le_bytes()).expect("write");
    let put_f32s = |w: &mut std::io::BufWriter<std::fs::File>, v: &[f32]| {
        for x in v {
            w.write_all(&x.to_le_bytes()).expect("write");
        }
    };
    put_u32(&mut w, MAGIC);
    put_u32(&mut w, n as u32);
    put_u32(&mut w, n_off as u32);
    let (rows, cols, blocks) = (out_row.host(), out_column.host(), out_block.host());
    for k in 0..n_off {
        put_u32(&mut w, rows[k]);
        put_u32(&mut w, cols[k]);
        put_f32s(&mut w, &blocks[9 * k..9 * k + 9]);
    }
    put_f32s(&mut w, &diagonal.host()[..9 * n]);
    put_f32s(&mut w, &rhs.host()[..3 * n]);
    w.flush().expect("flush");
    let xpath = format!("{dir}/evalx_{solve}.bin");
    let mut xw = std::io::BufWriter::new(std::fs::File::create(&xpath).expect("create evalx"));
    for v in &iterate.host()[..3 * n] {
        xw.write_all(&v.to_le_bytes()).expect("write");
    }
    xw.flush().expect("flush");
    eprintln!("[dump_linsys] wrote {path}: nrow={n} n_offdiag={n_off} (solve #{solve})");
    std::process::exit(0);
}
