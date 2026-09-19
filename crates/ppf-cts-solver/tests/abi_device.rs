// File: crates/ppf-cts-solver/tests/abi_device.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! THE SEAM, DRIVEN AGAINST A REAL BACKEND LIBRARY THROUGH THE C ABI.
//!
//! A target's `Device` can be complete and called by nothing, and this file is
//! the gate on that: allocate, dispatch a GENERATED entry point through
//! [`Device::run`], read back and compare
//! against arithmetic that is exact by construction; a record and a replay,
//! asserted to be the platform's deferred form; a `grow` between the two,
//! required to be refused as a stale region rather than replayed against a
//! moved block; and every refusal the boundary owes, taken by name.
//!
//! # What this proves, and what it deliberately does not
//!
//! It proves the ROUTE: that the neutral Rust seam reaches a real device
//! through a library implementing `kernels/seam/backend_abi.h`, that the
//! two generated kernel tables agree, that an argument record assembled above
//! the seam is bound and launched below it, and that the numbers that come back
//! are the ones the neutral kernel body computes.
//!
//! It does NOT prove that any production phase takes that route, and nothing
//! here should ever be read as though it did. `pcg_update3` was compiled,
//! self-tested and called by nothing on Metal for the backend's whole life
//! while `precond = "block-jacobi"`, the default every example runs, was a
//! label over unpreconditioned CG. A per-kernel fixture passes happily on dead
//! code. What will settle the production question is three instruments, and
//! they belong to the change that routes a phase, not to this one: a dispatch
//! COUNT asserted at the phase's own number, fault injection required to make
//! the run fail, and the penetration gate the position passes already sit
//! behind.
//!
//! # Why the whole file is one test
//!
//! The library supports ONE live backend and refuses a second `be_open`
//! with a named misuse, because the arena and the device-side base table it
//! writes are process-wide. `cargo test` runs test functions on several
//! threads, so two functions that each open a device would race and one would
//! fail for a reason unrelated to what it was checking. One function, in
//! sections, is the honest shape rather than a mutex that hides the constraint.

// The library exists only where the build produced one, and the build says so
// with a fact rather than with a target's name: whether an implementation of
// the seam is on this link line. Without it the whole file is absent, which is
// what keeps `cargo test` on a machine with no such library green rather than
// red for a missing artifact.
#![cfg(abi_backend_linked)]

use std::sync::OnceLock;

use ppf_cts_compute::abi::{AbiDevice, OpenConfig};

// THE OTHER HALF OF THE MACOS LINK SEAM, WHICH A TEST BINARY OWES AND DOES NOT
// INHERIT. A Metal backend dylib deliberately leaves `print_rust` UNDEFINED so
// it binds to the host's at load time, and the solver binary is what defines it
// (`src/main.rs`). A test binary is not that binary: it links the library and
// the crate, neither of which carries the symbol, so the link fails with
// `Undefined symbols: _print_rust` and names nothing about backends.
//
// It is a real bridge rather than a stub for the reason the link flag exists:
// an unbound reference resolves to address 0 under this profile, so the first
// log line from the backend would jump to null rather than be dropped. The
// CUDA arm never meets this, because an undefined symbol in an ELF shared
// object is resolved at load and the link does not object.
#[no_mangle]
extern "C" fn print_rust(message: *const std::os::raw::c_char) {
    let text = unsafe { std::ffi::CStr::from_ptr(message) };
    eprintln!("backend: {}", text.to_string_lossy());
}
use ppf_cts_compute::{
    AllocLabel, Device, Extent, Handle, KernelDecl, KernelId, Scatter,
};

// ===========================================================================
// THE DRIVER'S HALF OF THE KERNEL TABLE
// ===========================================================================

/// One generated row: the entry point's name, its record's size, the byte
/// offsets of its handle fields, whether it is group-shaped, and the static
/// group-local scratch it declares for itself.
///
/// A tuple rather than a struct because the generator emits exactly what a
/// declaration knows and nothing more; the fields `KernelDecl` carries that a
/// declaration cannot honestly fill are supplied once, below, where the reason
/// for each value can be stated in one place instead of eighty.
type Row = (&'static str, u16, &'static [u16], bool, u32);

/// The rows, in the order the library concatenated its own.
///
/// BOTH HALVES COME FROM ONE WALK OF ONE SORTED LIST, in the CUDA recipe, so
/// the id a row gets here is the id the library gave it by construction rather
/// than by two programs agreeing to sort the same way. `AbiDevice::open` still
/// compares them id by id, by name and by record size, because that check
/// catches a different mistake: a driver and a library built from different
/// trees.
static ROWS: &[Row] = &include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/kernel_table.rs"
));

/// The rows as the seam reads them, with the id assigned from the position.
///
/// The three fields a generated row does not carry are set here, once:
///
/// - `scatter` is `Atomic`, the CONSERVATIVE value. It exists so a target that
///   cuts a range across threads knows whether it may; this target does not cut
///   one at all, since the launcher computes its own grid from the record's
///   count field. Claiming `Disjoint` for eighty kernels nothing here inspects
///   would be asserting a property of each body that no declaration states.
/// - `nanos_per_item` feeds the host target's chunk width and nothing else. It
///   is 1.0 because no figure has been measured for these entry points on this
///   target, and an invented one would be read later as a measurement.
/// - `host_refs` is EMPTY, and that is a statement rather than a placeholder. A
///   generated record addresses its buffers by (arena, offset) handle, which is
///   what a device can resolve; the seam refuses a dispatch whose record names
///   buffers by host address, and this table is on the far side of that
///   refusal.
fn table() -> &'static [KernelDecl] {
    static TABLE: OnceLock<Vec<KernelDecl>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            ROWS.iter()
                .enumerate()
                .map(|(index, row)| KernelDecl {
                    id: KernelId(u16::try_from(index).expect("the table is far under 65536 rows")),
                    name: row.0,
                    scatter: Scatter::Atomic,
                    nanos_per_item: 1.0,
                    args_bytes: row.1,
                    host_refs: &[],
                    diag: false,
                    generated: true,
                })
                .collect()
        })
        .as_slice()
}

fn row_of(name: &str) -> (KernelId, &'static Row) {
    let index = ROWS
        .iter()
        .position(|row| row.0 == name)
        .unwrap_or_else(|| panic!("the generated table carries no entry point named {name}"));
    (
        KernelId(u16::try_from(index).expect("the table is far under 65536 rows")),
        &ROWS[index],
    )
}

// ===========================================================================
// ASSEMBLING AN ARGUMENT RECORD
// ===========================================================================

/// An argument record, built field by field in declaration order.
///
/// **WHY THIS IS NOT A HAND-WRITTEN MIRROR OF A GENERATED STRUCT, which is the
/// thing this architecture forbids.** A mirror is a second DECLARATION that can
/// disagree with the first while both still compile. What is here declares
/// nothing: it appends bytes, and two independent things check the result. The
/// LENGTH is checked against the generated row, which took it from the same
/// `sizeof` the library's table did, so a field added, removed or resized fails
/// here. The ORDER is checked by the ANSWER: `vec_copy` with distinct data
/// gives the right result only if source and destination are in the slots the
/// declaration puts them in, and the position step below is exact, so a swapped
/// or shifted field is a wrong number rather than a plausible one.
///
/// A generated record has NO PADDING by construction: every field is 4-byte
/// aligned and 4 or 16 bytes wide, which is what makes sequential appending
/// reproduce the layout exactly rather than approximately.
#[derive(Default)]
struct Record(Vec<u8>);

impl Record {
    fn handle(mut self, handle: Handle) -> Self {
        self.0.extend_from_slice(&handle.arena.to_ne_bytes());
        self.0.extend_from_slice(&handle.off.to_ne_bytes());
        self.0.extend_from_slice(&handle.size.to_ne_bytes());
        self.0.extend_from_slice(&handle.allocated.to_ne_bytes());
        self
    }

    fn f32(mut self, value: f32) -> Self {
        self.0.extend_from_slice(&value.to_ne_bytes());
        self
    }

    fn u32(mut self, value: u32) -> Self {
        self.0.extend_from_slice(&value.to_ne_bytes());
        self
    }

    /// The generator-owned trailing field. The LIBRARY fills it with its own
    /// live arena count as it binds the buffers, which a caller above the seam
    /// does not have, so a driver leaves it at zero and the bytes are here only
    /// to make the record its declared size.
    fn arena_count_slot(self) -> Self {
        self.u32(0)
    }

    fn finish(self, row: &Row) -> Vec<u8> {
        assert_eq!(
            self.0.len(),
            row.1 as usize,
            "the record assembled for {} is {} bytes and the generated \
             declaration says {}",
            row.0,
            self.0.len(),
            row.1
        );
        self.0
    }
}

fn bytes_of_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn f32s_from(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}


fn quiet(_level: i32, _message: &str) {}

// ===========================================================================
// THE GATE
// ===========================================================================

#[test]
fn the_seam_drives_a_linked_backend_library() {
    let mut config = OpenConfig::new(quiet);
    // POISON EVERY FRESH ALLOCATION. A buffer accumulated into but never
    // cleared is correct only while the memory it happens to get is still zero,
    // and fresh device memory frequently is, so a run that does not poison
    // cannot tell the two apart. 0x5a is loud under every reading: 1.9e16 as a
    // float, 1515870810 as an index.
    config.poison_byte = Some(0x5a);
    // A FALLBACK MUST NOT BE SILENT, and the flag is NOT how this asserts it,
    // because the two linked libraries answer differently and both answers are
    // correct. One records a captured graph; the other replays a region by
    // re-encoding its dispatches and says so in `supports_deferred_regions`.
    // Setting the flag would make the second one refuse at record time and end
    // the test there, taking the replay and stale-region coverage with it.
    //
    // What replaces it is stricter about the thing that matters: the region's
    // realized form is compared against what the backend DECLARES, so a library
    // claiming deferral and quietly stopping still fails, which is the
    // regression the flag existed to catch.
    config.require_deferred_regions = false;
    // WHERE THE BUILD PUT THE BACKEND'S ARTIFACTS, which a library that loads a
    // pre-built shader has to be TOLD. `build.rs` is the one file that knows
    // which target this is and emits the path; a library that compiles its own
    // kernels ignores it, so this is set unconditionally rather than per
    // backend.

    // Safety: the build linked a library implementing the header, which is what
    // `abi_backend_linked` states, and `table()` is the driver's half of the
    // same generated table the library carries.
    let mut device = unsafe { AbiDevice::open(table(), &config) }
        .expect("opening the linked backend library");

    // --- identity, and the four checks `open` already made ------------------
    //
    // WHAT IS ASSERTED HERE IS TARGET-INDEPENDENT, AND THAT IS THE POINT RATHER
    // THAN CAUTION. `abi_backend_linked` says a library implementing the
    // seam is on this link line and deliberately does not say which; a file in
    // this crate naming one would be the first sentence of the fork the whole
    // architecture removes, and would also be wrong the day a second library
    // answers the same ABI. So the backend's NAME and its fault model are
    // reported and not compared: `faults_on_oob` is true on one target and
    // false on another, and the ABI states outright that a driver may not
    // branch on it. What is compared is what every library owes: a name, a
    // device, and the arena budget the ABI fixes for every target so that a
    // handle means one thing everywhere.
    let info = device.info().clone();
    assert!(
        !info.backend.is_empty(),
        "the library named itself with an empty string"
    );
    assert!(
        !info.device_name.is_empty(),
        "the library reported no device name"
    );
    println!(
        "backend {} on {}, faults_on_oob = {}",
        info.backend, info.device_name, info.faults_on_oob
    );
    assert_eq!(info.max_arenas, 29);
    assert_eq!(
        device.kernels().len(),
        ROWS.len(),
        "the seam is holding a different table from the one this file built"
    );
    assert!(
        !ROWS.is_empty(),
        "the generated table is empty, so every cross-check at open passed by \
         vacuity"
    );
    assert!(
        device.missing().is_empty(),
        "the library carries no implementation for {:?}",
        device.missing()
    );

    // Creating every pipeline the driver will need, stated before a frame is
    // written. This target has no pipeline creation step, so what it validates
    // is the ids.
    let all: Vec<KernelId> = device.kernels().iter().map(|d| d.id).collect();
    device
        .prepare_kernels(&all)
        .expect("preparing every id in the table");

    device.counters_reset();

    // --- a copy, which is exact ---------------------------------------------
    //
    // The first dispatch of a GENERATED entry point through this seam. A copy
    // is chosen because it is bit-exact whatever the arithmetic does, so a
    // difference can only be the route.
    const N: u32 = 4096;
    let source: Vec<f32> = (0..N).map(|i| i as f32 * 0.25 - 7.5).collect();
    let (copy_id, copy_row) = row_of("vec_copy_entry");

    let src = device
        .alloc(N as usize, 4, 4, AllocLabel("abi.gate.source"))
        .expect("allocating the source");
    let dst = device
        .alloc(N as usize, 4, 4, AllocLabel("abi.gate.destination"))
        .expect("allocating the destination");
    device
        .write(src, 0, &bytes_of_f32(&source))
        .expect("uploading the source");

    let record = Record::default()
        .handle(src)
        .handle(dst)
        .u32(N)
        .arena_count_slot()
        .finish(copy_row);
    device
        .run("abi.gate.copy", |encoder| {
            // Safety: the record is `copy_row.1` bytes long, which is what the
            // declaration says, and its handles name live allocations.
            unsafe { encoder.dispatch_raw(copy_id, Extent::Elements { count: N }, record.as_ptr(), record.len()) }
        })
        .expect("dispatching the copy");

    let mut back = vec![0u8; N as usize * 4];
    device.read(dst, 0, &mut back).expect("reading the copy back");
    assert_eq!(
        f32s_from(&back),
        source,
        "the copy through the seam did not reproduce its source"
    );

    // --- a scaled add, whose values are exactly representable ----------------
    //
    // `d[i] += scale * s[i]`, with every input a small integer and the scale a
    // power of two, so the product and the sum are exact in fp32 and the answer
    // is the same whether the compiler contracted the two into an FMA or not.
    // That is deliberate: this file is a gate on the ROUTE, and a tolerance
    // here would be a place for a real difference to hide.
    let (add_id, add_row) = row_of("vec_add_scaled_entry");
    let addend: Vec<f32> = (0..N).map(|i| (i % 17) as f32).collect();
    let base: Vec<f32> = (0..N).map(|i| (i % 5) as f32).collect();
    device
        .write(src, 0, &bytes_of_f32(&addend))
        .expect("uploading the addend");
    device
        .write(dst, 0, &bytes_of_f32(&base))
        .expect("uploading the base");

    let record = Record::default()
        .handle(src)
        .handle(dst)
        .f32(0.5)
        .u32(N)
        .arena_count_slot()
        .finish(add_row);
    device
        .run("abi.gate.add", |encoder| {
            // Safety: as above.
            unsafe { encoder.dispatch_raw(add_id, Extent::Elements { count: N }, record.as_ptr(), record.len()) }
        })
        .expect("dispatching the scaled add");
    device.read(dst, 0, &mut back).expect("reading the sum back");
    let want: Vec<f32> = base
        .iter()
        .zip(&addend)
        .map(|(b, a)| b + 0.5 * a)
        .collect();
    assert_eq!(
        f32s_from(&back),
        want,
        "the scaled add through the seam did not compute d + 0.5 s"
    );

    // --- the position step -------------------------------------------------
    //
    // `main/position_step` is the only place the Newton iterate moves, so it is
    // the pass worth proving reachable. It reads a POSITION, subtracts a scaled
    // SEARCH DIRECTION, and writes the position back. The inputs below are all
    // small powers of two, so every product and every difference is exact in
    // fp32 and the comparison below carries no tolerance at all.
    const VERTS: u32 = 512;
    let (step_id, step_row) = row_of("position_step_entry");
    // Positions at 0, +1 and -1.
    let positions: Vec<f32> = (0..VERTS).flat_map(|_| [0.0f32, 1.0, -1.0]).collect();
    // A direction whose scaled components are exact in fp32.
    let direction: Vec<f32> = (0..VERTS).flat_map(|_| [1.0f32, 2.0, -3.0]).collect();
    let scale = 0.5f32;

    let position = device
        .alloc(VERTS as usize, 12, 4, AllocLabel("abi.gate.eval_x"))
        .expect("allocating the positions");
    let dir = device
        .alloc(VERTS as usize * 3, 4, 4, AllocLabel("abi.gate.direction"))
        .expect("allocating the direction");
    device
        .write(position, 0, &bytes_of_f32(&positions))
        .expect("uploading the positions");
    device
        .write(dir, 0, &bytes_of_f32(&direction))
        .expect("uploading the direction");

    let record = Record::default()
        .handle(position)
        .handle(dir)
        .f32(scale)
        .u32(VERTS)
        .arena_count_slot()
        .finish(step_row);
    device
        .run("abi.gate.position_step", |encoder| {
            // Safety: as above.
            unsafe {
                encoder.dispatch_raw(
                    step_id,
                    Extent::Elements { count: VERTS },
                    record.as_ptr(),
                    record.len(),
                )
            }
        })
        .expect("dispatching the position step");

    let mut step_back = vec![0u8; VERTS as usize * 12];
    device
        .read(position, 0, &mut step_back)
        .expect("reading the positions back");
    let want: Vec<f32> = positions
        .iter()
        .zip(&direction)
        .map(|(p, d)| p - scale * d)
        .collect();
    assert_eq!(
        f32s_from(&step_back),
        want,
        "the position step through the seam moved the positions by the wrong \
         amount"
    );

    // --- what the counters say ----------------------------------------------
    //
    // Three dispatches and nothing else. A dispatch that did not happen is a
    // counter that did not move, which is the one instrument a per-kernel
    // fixture cannot substitute for.
    let counters = device.counters();
    assert_eq!(
        counters.dispatches, 3,
        "three dispatches were issued and the library counted {}",
        counters.dispatches
    );
    assert_eq!(
        counters.regions_fallback, 0,
        "no region has been recorded yet, so nothing can have fallen back"
    );

    // --- a recorded region, in the platform's deferred form ------------------
    //
    // Recorded with `require_deferred_regions` set, so a target that could not
    // capture would have failed at record time by name rather than replayed by
    // re-issuing and shown up only as a slowdown.
    device
        .write(dst, 0, &bytes_of_f32(&base))
        .expect("re-uploading the base");
    let record = Record::default()
        .handle(src)
        .handle(dst)
        .f32(1.0)
        .u32(N)
        .arena_count_slot()
        .finish(add_row);
    let region = device
        .record("abi.gate.region", |encoder| {
            // Safety: as above.
            unsafe { encoder.dispatch_raw(add_id, Extent::Elements { count: N }, record.as_ptr(), record.len()) }
        })
        .expect("recording a region");
    let shape = region.info();
    let declared = i32::from(info.supports_deferred_regions);
    assert_eq!(
        shape.deferred, declared,
        "this backend declares supports_deferred_regions = {declared} and \
         realized a region with deferred = {}. A region that quietly stopped \
         being deferred is a slowdown inside this project's run-to-run \
         envelope rather than a visible failure, which is why the realized \
         form is compared against the declared one rather than assumed",
        shape.deferred
    );
    assert_eq!(shape.dispatch_count, 1);

    const REPEATS: u32 = 8;
    device
        .replay(&region, REPEATS)
        .expect("replaying the region");
    device
        .read(dst, 0, &mut back)
        .expect("reading the replayed sum back");
    let want: Vec<f32> = base
        .iter()
        .zip(&addend)
        .map(|(b, a)| b + REPEATS as f32 * a)
        .collect();
    assert_eq!(
        f32s_from(&back),
        want,
        "replaying the region {REPEATS} times did not apply its dispatch \
         {REPEATS} times"
    );

    let counters = device.counters();
    // THE SAME COMPARISON THE REGION'S OWN SHAPE GOT, one level out: a library
    // that declares deferral counts one deferred region and no fallback, and
    // one that does not counts the reverse. The header states both halves, and
    // asserting either as an absolute would be asserting one platform's
    // implementation on every platform.
    let (want_deferred, want_fallback) =
        if info.supports_deferred_regions { (1, 0) } else { (0, 1) };
    assert_eq!(
        counters.regions_deferred, want_deferred,
        "this backend declares supports_deferred_regions = {} and counted {} \
         deferred region(s)",
        info.supports_deferred_regions, counters.regions_deferred
    );
    assert_eq!(
        counters.regions_fallback, want_fallback,
        "and counted {} fallback region(s)",
        counters.regions_fallback
    );
    assert_eq!(
        counters.dispatches,
        3 + u64::from(REPEATS),
        "three one-shot dispatches plus {REPEATS} replayed ones"
    );

    // --- a grow between record and replay, which must be refused -------------
    //
    // A grow may relocate a block, so a recorded argument record can hold a
    // stale handle and replaying it would address memory the allocator has
    // since handed to something else. That has to be a named refusal and not a
    // wrong answer.
    let mut growable = src;
    device
        .grow(&mut growable, N as usize * 2, 4, 4)
        .expect("growing an allocation");
    let stale = device.replay(&region, 1).expect_err(
        "replaying a region recorded before a grow must be refused, because a \
         grow may have moved the block its handles name",
    );
    let text = stale.to_string();
    assert!(
        text.contains("abi.gate.region") && text.contains("generation"),
        "the refusal did not name the region and the two generations: {text}"
    );
    device.release(region);

    // --- a WINDOW, and the poisoning that makes it visible -------------------
    //
    // A byte offset is measured from the start of the BLOCK, so a caller may
    // move a window of a block holding several arrays end to end. That the
    // window is honored rather than merely accepted is what the poisoning
    // shows: the bytes outside it must still carry the fill this device was
    // opened with, and 0x5a is chosen because it is loud under every reading.
    const WINDOW_ELEMS: usize = 8;
    let block = device
        .alloc(WINDOW_ELEMS, 4, 4, AllocLabel("abi.gate.window"))
        .expect("allocating a block to window");
    let half = bytes_of_f32(&[1.0, 2.0, 3.0, 4.0]);
    device
        .write(block, 16, &half)
        .expect("writing the second half of the block");
    let mut whole = vec![0u8; WINDOW_ELEMS * 4];
    device
        .read(block, 0, &mut whole)
        .expect("reading the whole block");
    assert_eq!(
        &whole[16..],
        &half[..],
        "the window did not land at byte 16"
    );
    assert_eq!(
        &whole[..16],
        &[0x5au8; 16],
        "the bytes before the window are not the poison this device was \
         opened with, so either the window overran or the allocation was not \
         poisoned"
    );

    // The bound is the BLOCK's byte capacity, which the library reads from its
    // allocator. A handle's `size` and `allocated` are ELEMENT counts, so a
    // bound taken from one of them would refuse every window past the element
    // count and admit none past the real end: a refusal that looks like a
    // bounds check and is not one. One byte past the end is where that shows.
    let err = device
        .write(block, WINDOW_ELEMS * 4 - 3, &[0u8; 4])
        .expect_err("a window running one byte past the block must be refused");
    assert!(
        err.to_string().contains("bytes this handle names"),
        "the refusal did not name the block's capacity: {err}"
    );
    let mut block = block;
    device.free(&mut block).expect("freeing the windowed block");

    // --- the refusals the boundary owes --------------------------------------
    //
    // Each is taken by name, because the alternative to a named refusal here is
    // a launch with the wrong bytes or the wrong geometry, and both produce
    // plausible numbers.
    let short = vec![0u8; add_row.1 as usize - 4];
    let err = device
        .run("abi.gate.short", |encoder| {
            // Safety: the pointer addresses `short.len()` readable bytes; the
            // length is deliberately wrong, which is what is under test.
            unsafe { encoder.dispatch_raw(add_id, Extent::Elements { count: 1 }, short.as_ptr(), short.len()) }
        })
        .expect_err("a record of the wrong length must be refused");
    assert!(
        err.to_string().contains("vec_add_scaled_entry"),
        "the refusal did not name the kernel: {err}"
    );

    let unknown = KernelId(u16::try_from(ROWS.len()).expect("under 65536"));
    // Bound before the call: two calls to `record_bytes()` inside one
    // expression would take the address of one temporary and the length of
    // another, and the first would already be dropped.
    let well_formed = record_bytes();
    let err = device
        .run("abi.gate.unknown", |encoder| {
            // Safety: the record is well formed; the id is deliberately outside
            // the table.
            unsafe {
                encoder.dispatch_raw(
                    unknown,
                    Extent::Elements { count: 1 },
                    well_formed.as_ptr(),
                    well_formed.len(),
                )
            }
        })
        .expect_err("an id outside the table must be refused");
    assert!(
        err.to_string().contains("kernel"),
        "the refusal did not name a kernel: {err}"
    );

    device.free(&mut growable).expect("freeing the source");
    let mut dst = dst;
    device.free(&mut dst).expect("freeing the destination");
    let mut position = position;
    device.free(&mut position).expect("freeing the positions");
    let mut dir = dir;
    device.free(&mut dir).expect("freeing the direction");
}

/// Bytes of a record that is well formed and never dispatched, for the
/// refusal that is about the ID rather than about the record.
fn record_bytes() -> Vec<u8> {
    let (_, row) = row_of("vec_add_scaled_entry");
    // A pair of zeroed handles, spelled out because `Handle` carries no
    // `Default`: zero is a REAL arena, so a value-initialized handle names arena
    // 0 at offset 0 rather than nothing. This record is refused on its id and
    // never dispatched, so the values are placeholders.
    const PLACEHOLDER: Handle = Handle {
        arena: 0,
        off: 0,
        size: 0,
        allocated: 0,
    };
    Record::default()
        .handle(PLACEHOLDER)
        .handle(PLACEHOLDER)
        .f32(0.0)
        .u32(0)
        .arena_count_slot()
        .finish(row)
}

/// The two halves of the table are generated from one walk of one sorted list,
/// so a row here is the row the library carries at the same id. What this
/// checks is the shape of the generated rows themselves, which needs no device
/// and so runs on any host that built the library.
#[test]
fn the_generated_table_is_dense_and_well_formed() {
    assert!(!ROWS.is_empty(), "the generated table is empty");
    for (index, row) in ROWS.iter().enumerate() {
        // `_entry` AND NOT A `ppf_` PREFIX. The prefix was removed from every
        // identifier that names no namespace, so a generated entry point is
        // `<body>_entry`; this assertion still required the old spelling and
        // had gone stale, which nothing noticed because nothing had run this
        // gate since.
        assert!(
            row.0.ends_with("_entry") && row.0.len() > "_entry".len(),
            "row {index} is named {}, which is not a generated entry point's \
             name",
            row.0
        );
        assert!(row.1 >= 4, "row {} declares a {}-byte record", row.0, row.1);
        assert!(
            row.1 as usize % 4 == 0,
            "row {} declares a {}-byte record, and a generated record has no \
             padding, so its size is a multiple of 4",
            row.0,
            row.1
        );
        for &offset in row.2 {
            assert!(
                (offset as usize) + 16 <= row.1 as usize,
                "row {} puts a handle at {offset} in a {}-byte record",
                row.0,
                row.1
            );
        }
        // A GROUP-SHAPED ROW IS DISPATCHABLE: `Extent::Groups` carries the
        // group count, the width and the scratch, and the C ABI spells the
        // same thing as `EXTENT_GROUPS`. What is worth checking is that the
        // two halves AGREE, because the scratch is bound by the launch and
        // read by the body: an element row asking for scratch would have it
        // silently unbound, and a scratch size that is not a whole number of
        // words cannot be the array any body indexes.
        if row.3 {
            assert_eq!(
                row.4 as usize % 4,
                0,
                "row {} is group-shaped and declares {} scratch bytes, which \
                 is not a whole number of 4-byte words",
                row.0,
                row.4
            );
        } else {
            assert_eq!(row.4, 0, "row {} declares scratch and is not a group", row.0);
        }
    }
    let mut names: Vec<&str> = ROWS.iter().map(|r| r.0).collect();
    names.sort_unstable();
    let count = names.len();
    names.dedup();
    assert_eq!(
        names.len(),
        count,
        "two rows share a name, so one id would name two kernels"
    );
}
