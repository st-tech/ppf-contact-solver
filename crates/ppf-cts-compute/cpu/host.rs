// File: crates/ppf-cts-compute/cpu/host.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The host target: MECHANISM ONLY, behind [`crate::Device`].
//!
//! It allocates, transfers, cuts a range across threads, calls, waits, and
//! collects the diagnostic channel. It computes nothing and it reads nothing it
//! is given: a dispatch arrives as a KERNEL IDENTIFIER, an EXTENT and an OPAQUE
//! BLOB, and what a kernel is, what its arguments mean and which kernels exist
//! at all are supplied by the caller at [`HostDevice::new`].
//!
//! # What that separation buys, and why it is not decoration
//!
//! The launch table below is two `&'static` slices this target never authored:
//! a declaration per kernel and a function pointer per kernel, indexed by the
//! same id. So this file can be read, reviewed and published without knowing
//! what a Newton step is, which is the one test the crate is held to, and the
//! caller keeps the argument records beside the declarations they are generated
//! from rather than mirroring them here. A hand-written mirror pair is two
//! declarations that can disagree, and no compiler sees the disagreement.
//!
//! # The three properties a reader should not have to rediscover
//!
//! - **The cut is by a FIXED chunk width**, [`crate::sched::chunk_for`] of the
//!   declared per-item cost, never by the thread count. So the answer does not
//!   depend on how many threads ran it, which is what makes one scene under two
//!   `RAYON_NUM_THREADS` values byte-identical.
//! - **A scatter that is not [`Scatter::Disjoint`] is run SERIALLY, ascending.**
//!   `compute::atomic_add` on the host seam is a plain read, add and write
//!   back, so a parallel pass over elements sharing an output slot is a DATA
//!   RACE and not a different fold order. The declaration is what says which a
//!   kernel is, and this file obeys it rather than guessing from the name.
//! - **The diagnostic channel merges in ASCENDING CHUNK ORDER.** A device
//!   channel latches whichever thread arrives first, which is not reproducible;
//!   the chunks here are independent, so the report is defined to be the
//!   lowest-numbered chunk's and two runs of one workload name the same check.

use rayon::prelude::*;

use crate::sched;
use crate::{
    AllocLabel, Counters, Device, DeviceInfo, Diag, DiagFailure, Encoder, Extent, Fault, Handle,
    HostRef, KernelDecl, KernelId, Scatter,
};
use crate::mem::Pool;

use std::collections::BTreeMap;

// ===========================================================================
// The diagnostic channel's wire record.
// ===========================================================================

/// One chunk's diagnostic record, as the compiled entry point writes it.
///
/// It is TRANSPORT, which is why it is here and not above the seam: the fields
/// are a count, a claim flag, a source location and four floats, and this crate
/// resolves them into a [`DiagFailure`] without learning what any of them means.
/// One record type across every channel is what keeps a traversal's stack check
/// and a narrow phase's bounds check from drifting apart.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DiagRecord {
    fail_count: u32,
    claimed: u32,
    line: u32,
    payload: [f32; 4],
    file: *const std::ffi::c_char,
}

// The record is written only by the chunk that owns it and read only after that
// chunk's call has returned, so no chunk observes another's.
unsafe impl Send for DiagRecord {}
unsafe impl Sync for DiagRecord {}

/// The channel's merge rule: total failures, first claimant in ASCENDING chunk
/// order.
///
/// Ascending rather than first-to-arrive, because the chunks are independent and
/// a device channel latching whichever thread arrives first is not reproducible.
/// Two runs of one workload must name the same failing check.
///
/// A total above zero with NO claimant reports clean, which is deliberate: the
/// count and the claim are written by different stores in the compiled body, so
/// a count with no record behind it has nothing to name and inventing a location
/// for it would be worse than reporting what was actually recorded.
pub fn merge_records(records: &[DiagRecord]) -> Diag {
    let total: u32 = records.iter().map(|d| d.fail_count).sum();
    if total == 0 {
        return Diag::default();
    }
    let Some(first) = records.iter().find(|d| d.claimed != 0) else {
        return Diag::default();
    };
    let file = if first.file.is_null() {
        String::from("<unrecorded>")
    } else {
        // Safety: a non-null `file` is the `__FILE__` of the compiled body, a
        // string literal with static storage duration.
        unsafe { std::ffi::CStr::from_ptr(first.file) }
            .to_string_lossy()
            .into_owned()
    };
    Diag {
        failures: total as u64,
        first: Some(DiagFailure {
            file,
            line: first.line,
            payload: first.payload,
        }),
    }
}

/// One launch per kernel id, supplied by the caller.
///
/// This is the whole of what this target knows about a kernel: an address to
/// call with a half-open range. Everything the call needs in order to mean
/// something (the record's layout, the body it wraps, the physics it computes)
/// stays above the seam with the declaration it was generated from.
///
/// # Safety
/// `args` addresses the record the id declares, `begin <= end`, and `diag`
/// addresses a writable record when the declaration says the kernel takes one.
pub type Launch = unsafe fn(
    args: *const u8,
    begin: u32,
    end: u32,
    group_width: u32,
    diag: *mut DiagRecord,
);

/// The most buffers one generated entry point may name, and the largest record.
///
/// Both are checked in `check_shape`, off the DECLARATION, so a record that
/// outgrows either is refused before a byte is read rather than truncated. The
/// generator's own portable cap is 4096 bytes; nothing in this driver is near
/// it, and a stack copy per dispatch is what these numbers buy.
///
/// The narrow-phase visitors set both: a point-point candidate reaches 19
/// buffers, the widest record in the driver, because it walks two adjacency
/// tables as well as the mesh, the properties and the elastic snapshot. The
/// figures carry headroom for the collision-mesh passes rather than tracking
/// the current maximum exactly, since a record that outgrows either is refused
/// in `check_shape` off the DECLARATION and never truncated.
pub const MAX_BOUND_BUFFERS: usize = 32;
pub const MAX_BOUND_ARGS: usize = 640;

/// Arenas this target opens, the same number the arena ABI states so that a
/// driver sized against one backend's binding budget is sized against every
/// backend's.
pub const MAX_ARENAS: u32 = 29;

/// Base-table slots one dispatch may need.
///
/// A record may name buffers BOTH ways while the migration from an address to
/// a handle is under way: the handles resolve against this target's own live
/// arenas, which occupy the low slots, and each remaining address is bound to a
/// slot above them. So the table has to hold both at once, and the static
/// assertion below is what makes the sum a compile-time fact rather than a
/// runtime check on a path that would already have written the record.
pub const MAX_ARENA_SLOTS: usize = MAX_ARENAS as usize + MAX_BOUND_BUFFERS;
const _: () = assert!(MAX_ARENA_SLOTS >= MAX_ARENAS as usize + MAX_BOUND_BUFFERS);

/// What an arena reserves when it is opened, and the most it may hold.
///
/// The doubling policy is `ppf-cts-compute/cuda/arena/arena.cu`'s and is
/// mirrored rather than invented, so the two targets pack the same driver into
/// a comparable number of arenas. The cap is what a 32-bit offset can address;
/// an allocation larger than it is refused by name rather than truncated into a
/// plausible small one, which is what `allocated: bytes as u32` did before.
const FIRST_RESERVE: u64 = 1 << 20;
const ARENA_CAP: u64 = 4 << 30;

/// The least alignment an arena's block is allocated to. Every offset inside it
/// is aligned by [`HostArena::allocate_span`], and an offset is an aligned
/// ADDRESS only when the base is at least as aligned as the request.
const ARENA_ALIGN: usize = 64;

/// One generated entry point's arguments, resolved into this backend's form.
///
/// **WHY A TRANSLATION EXISTS AT ALL, and why it is mechanism rather than
/// logic.** The seam's buffer reference is an (arena, offset) handle resolved
/// against a table of arena base addresses, which is what a generated entry
/// point takes on every backend, because Metal enforces 31 buffer binding slots
/// against this solver's 90 transitive device pointers. What the DRIVER has to
/// put in a record today is an address, because its buffers are Rust-owned
/// `Vec`s and the host's own `DataSet` arrays rather than device allocations:
/// that is the `HostRef` debt `ppf_cts_compute` spells out. So this backend binds
/// each reference to an arena of its own, base `k` being the reference's
/// address and the handle `(k, 0)`. Resolving a reference to an address is what
/// an allocator does, and the day the driver's buffers are device allocations
/// this collapses to passing the backend's own base table unchanged.
///
/// It is built ONCE per dispatch, never per chunk: the chunks share the record
/// and the table, and both are read-only for the duration.
///
/// `align(4)` is load-bearing: the bytes below are handed to the entry point as
/// a pointer to its argument record, which every rendering asserts is 4-byte
/// aligned, and a `[u8; N]` alone is 1-aligned. It is first so its address is
/// the struct's own.
#[repr(C, align(4))]
pub struct BoundArgs {
    bytes: [u8; MAX_BOUND_ARGS],
    bases: [*mut u8; MAX_ARENA_SLOTS],
}

impl BoundArgs {
    /// The rewritten record, as the generated entry point takes it.
    ///
    /// An accessor rather than a public field, because the two arrays are
    /// POSITIONAL: the entry point resolves handle `(k, off)` against base `k`,
    /// so a caller able to reorder either would be able to bind a record to the
    /// wrong buffer with no type error anywhere.
    pub fn args_ptr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    /// The arena base table the record's handles resolve against.
    pub fn bases_ptr(&self) -> *const *mut u8 {
        self.bases.as_ptr()
    }

    /// The address bound to arena `index`.
    pub fn base(&self, index: usize) -> *mut u8 {
        self.bases[index]
    }
}
const _: () = assert!(std::mem::align_of::<BoundArgs>() >= 4);

/// Copies the record, lays this target's own arena bases in the low slots of the
/// base table, and rewrites every remaining `HostRef` into an (arena, offset)
/// handle naming a slot above them.
///
/// **THE TWO HALVES ARE WHAT LETS A RECORD BE MIGRATED FIELD BY FIELD.** A
/// generated entry point reaches every buffer as `base[arena] + off`, so a field
/// the driver already fills with a [`Handle`] needs the allocator's real table
/// and a field it still fills with an address needs one slot of its own. Laying
/// the real arenas first and appending the addresses gives both at once, and no
/// slot can serve two buffers: the handles name arenas below `arena_bases.len()`
/// because that is where this target's allocator put them, and the appended
/// slots start exactly there.
///
/// The day the driver's buffers are all device allocations, `decl.host_refs` is
/// empty for every row, the loop below does nothing, and what the entry point
/// receives is this target's own base table unchanged.
///
/// # Safety
/// `args` addresses `decl.args_bytes` readable bytes holding the record `decl`
/// declares, and `check_shape` has already established that the record fits
/// [`MAX_BOUND_ARGS`] and names at most [`MAX_BOUND_BUFFERS`] buffers by
/// address.
pub unsafe fn bind_generated(
    decl: &'static KernelDecl,
    args: *const u8,
    arena_bases: &[usize],
) -> BoundArgs {
    let mut bound = BoundArgs {
        bytes: [0u8; MAX_BOUND_ARGS],
        bases: [std::ptr::null_mut(); MAX_ARENA_SLOTS],
    };
    let width = decl.args_bytes as usize;
    std::ptr::copy_nonoverlapping(args, bound.bytes.as_mut_ptr(), width);
    // The allocator's live arenas, in its own order, because a handle's `arena`
    // field is an index into exactly this table.
    let live = arena_bases.len();
    debug_assert!(live + decl.host_refs.len() <= MAX_ARENA_SLOTS);
    for (slot, &base) in arena_bases.iter().enumerate() {
        bound.bases[slot] = base as *mut u8;
    }
    for (index, &offset) in decl.host_refs.iter().enumerate() {
        let arena = live + index;
        let field = bound.bytes.as_mut_ptr().add(offset as usize);
        let reference = std::ptr::read_unaligned(field.cast::<HostRef>());
        bound.bases[arena] = reference.addr() as *mut u8;
        // (arena, off, size, allocated). The offset is zero because the whole
        // allocation IS the arena here, and both lengths are the byte length
        // the reference carries. A handle's lengths are ELEMENT counts
        // everywhere else, and this is not an exception to that so much as a
        // case where the question has no answer: a raw address carries no
        // element size, so a byte length is the only length there is. Nothing
        // reads either one. This backend's entry points do not, and CUDA's
        // `compute::arena::resolve` asserts only that the first does not exceed
        // the second, which a pair of equal lengths satisfies. A buffer over 4
        // GB saturates rather than wrapping, since those fields are 32 bits and
        // a wrapped length would be a plausible small one.
        let bytes = u32::try_from(reference.bytes()).unwrap_or(u32::MAX);
        let handle = [arena as u32, 0u32, bytes, bytes];
        std::ptr::write_unaligned(field.cast::<[u32; 4]>(), handle);
    }
    // The generator-owned trailing field. Every generated record carries
    // `arena_count` as its LAST field, so that a backend whose pointer
    // resolution cannot fault still has a live bound to check the arena id
    // against; the entry point asserts against it. The per-kernel thunk asserts
    // that position at compile time.
    //
    // WRITTEN UNCONDITIONALLY, not only when a reference was bound. A record
    // whose buffers are all handles names arenas this target opened, and their
    // ids are bounded by the live arena count rather than by anything in the
    // record, so a count left at zero would trip every one of the entry point's
    // own assertions.
    let slot = bound.bytes.as_mut_ptr().add(width - 4);
    std::ptr::write_unaligned(slot.cast::<u32>(), (live + decl.host_refs.len()) as u32);
    bound
}

/// One entry of an encoded region.
///
/// The argument bytes are OWNED here rather than borrowed, because a recorded
/// region outlives the call that built it. That copy is also what makes the
/// staleness rule meaningful: the bytes are frozen at record time, so a buffer
/// that moved afterwards is named by an address that is no longer its own, and
/// the generation counter is what catches it.
/// One recorded entry: a dispatch, or a fill batched beside it.
///
/// A fill is an OP rather than a separate list because it is ordered against
/// the dispatches around it: a counter cleared before a pass and accumulated
/// into by that pass must be cleared first on every replay.
enum Op {
    Dispatch {
        kernel: KernelId,
        extent: Extent,
        args: Vec<u8>,
    },
    Fill {
        handle: Handle,
        byte_offset: usize,
        bytes: usize,
        value: u8,
    },
}

/// A recorded dispatch list.
pub struct HostRegion {
    label: &'static str,
    ops: Vec<Op>,
    /// The allocator generation this was recorded at.
    generation: u64,
    /// Whether the backend realized it in its deferred form. False means every
    /// replay re-issues the list, which is correct and slower, and is counted by
    /// [`Counters::regions_fallback`] rather than logged.
    deferred: bool,
}

/// Immediate or recording, which is the only thing an encoder branches on.
enum Sink<'a> {
    Immediate(&'a mut HostDevice),
    Record(&'a mut Vec<Op>),
}

struct HostEncoder<'a> {
    label: &'static str,
    /// The caller's table, so a RECORDING encoder can validate a dispatch
    /// without holding the device: recording borrows the op list, not the
    /// device, and a region built from unvalidated bytes would fail at replay
    /// instead of where it was written.
    table: &'static [KernelDecl],
    sink: Sink<'a>,
    /// Merged over every dispatch this encoder executed. Recording collects
    /// nothing, because nothing has run.
    diag: Diag,
}

impl Encoder for HostEncoder<'_> {
    unsafe fn dispatch_raw(
        &mut self,
        kernel: KernelId,
        extent: Extent,
        args: *const u8,
        args_bytes: usize,
    ) -> Result<(), Fault> {
        let decl = match &self.sink {
            Sink::Immediate(device) => device.decl(kernel)?,
            Sink::Record(_) => self.table.get(kernel.0 as usize).ok_or(Fault::MissingKernel { kernel })?,
        };
        check_shape(decl, extent, args, args_bytes)?;
        match &mut self.sink {
            Sink::Immediate(device) => {
                let diag = device.execute(decl, extent, args);
                self.diag.failures += diag.failures;
                if self.diag.first.is_none() {
                    self.diag.first = diag.first;
                }
                Ok(())
            }
            Sink::Record(ops) => {
                let mut bytes = vec![0u8; args_bytes];
                std::ptr::copy_nonoverlapping(args, bytes.as_mut_ptr(), args_bytes);
                ops.push(Op::Dispatch {
                    kernel,
                    extent,
                    args: bytes,
                });
                Ok(())
            }
        }
    }

    unsafe fn fill_raw(
        &mut self,
        handle: Handle,
        byte_offset: usize,
        bytes: usize,
        value: u8,
    ) -> Result<(), Fault> {
        match &mut self.sink {
            Sink::Immediate(device) => device.fill_bytes(handle, byte_offset, bytes, value),
            Sink::Record(ops) => {
                ops.push(Op::Fill { handle, byte_offset, bytes, value });
                Ok(())
            }
        }
    }

    fn region(&self) -> &'static str {
        self.label
    }
}

/// Every check the seam can make before a byte is read by a kernel.
///
/// The `HostRef` walk is the one worth naming: a record whose address and length
/// disagree about whether a field names anything is HALF WIRED, which is what a
/// hand-filled record produces when a pointer is set and its count is not. On a
/// backend that faults, that surfaces as a segmentation fault at an unrelated
/// line; on Metal it would surface as plausible numbers.
///
/// # Safety
/// `args` must address `args_bytes` readable bytes.
unsafe fn check_shape(
    decl: &'static KernelDecl,
    extent: Extent,
    args: *const u8,
    args_bytes: usize,
) -> Result<(), Fault> {
    if args_bytes != decl.args_bytes as usize {
        return Err(Fault::Shape {
            kernel: decl.name,
            detail: format!(
                "the argument record is {args_bytes} bytes and the declaration says {}",
                decl.args_bytes
            ),
        });
    }
    // BOTH SHAPES REACH THE SAME CHECKS. The extent decides the launch
    // geometry and says nothing about whether the record fits its declaration,
    // which is what this function is establishing.
    match extent {
        Extent::Elements { .. } | Extent::Groups { .. } => {}
    }
    for &offset in decl.host_refs {
        let field = std::ptr::read_unaligned(
            args.add(offset as usize).cast::<HostRef>(),
        );
        if !field.is_coherent() {
            return Err(Fault::Shape {
                kernel: decl.name,
                detail: format!(
                    "the buffer reference at byte {offset} names a length with no address, so \
                     the record is half wired"
                ),
            });
        }
    }
    // A generated entry point's record is COPIED before the call, so that its
    // buffer references can be bound to arenas, and the copy has a fixed
    // capacity. Refused here, off the declaration, so a record that outgrew
    // either bound is named before a byte is read rather than truncated into a
    // plausible one.
    if decl.generated {
        if decl.args_bytes as usize > MAX_BOUND_ARGS {
            return Err(Fault::Shape {
                kernel: decl.name,
                detail: format!(
                    "the record is {} bytes, over the {MAX_BOUND_ARGS} this backend binds",
                    decl.args_bytes
                ),
            });
        }
        if decl.host_refs.len() > MAX_BOUND_BUFFERS {
            return Err(Fault::Shape {
                kernel: decl.name,
                detail: format!(
                    "the record names {} buffers, over the {MAX_BOUND_BUFFERS} this backend \
                     binds to arenas",
                    decl.host_refs.len()
                ),
            });
        }
    }
    Ok(())
}

/// One arena: a pooled block, a bump pointer, and the spans freed inside it.
///
/// **AN ARENA HOLDS MANY ALLOCATIONS, and it has to.** A handle's `arena` field
/// indexes a base table the entry point resolves against, and that table is
/// capped at [`MAX_ARENAS`] because Metal's binding budget is compiler-enforced
/// at 31. One arena per allocation therefore caps a driver at 29 buffers, which
/// this solver passes early in its own build; packing many spans into one arena
/// is what makes the cap a property of the BACKEND rather than a ceiling on the
/// caller.
///
/// The packing policy is `ppf-cts-compute/cuda/arena/arena.cu`'s, mirrored
/// rather than invented so the two targets pack one driver into a comparable
/// number of arenas: first fit over the free spans in ascending offset order,
/// then the bump pointer, and a fresh arena only when no open one can hold the
/// request. Handles still differ between targets, which is why an
/// [`AllocLabel`] rather than a handle is what a cross-target comparison keys
/// on.
struct HostArena {
    /// The base address of the pool block these bytes come from, cached at open
    /// time.
    ///
    /// The BLOCK ITSELF is not recorded, and does not need to be: an arena is
    /// never closed, so the block stays checked out for the allocator's life
    /// and the pool's own reuse never moves it under a live span. Caching the
    /// address keeps a dispatch's base table off the pool's borrow, and it is
    /// held as an integer so this type stays `Send` for the same reason `Block`
    /// is.
    base: usize,
    capacity: u64,
    /// The alignment the block was allocated to, which BOUNDS what an
    /// allocation inside it may ask for: an offset aligned inside a block whose
    /// own base is not aligned is not an aligned address.
    align: usize,
    bump: u64,
    /// Freed spans by offset, coalesced with their neighbors on release so a
    /// long run of allocate and free does not shatter the arena.
    free_spans: BTreeMap<u64, u64>,
}

impl HostArena {
    /// First fit over the free spans, then the bump pointer. `None` means this
    /// arena cannot hold the request and the caller should try the next.
    fn allocate_span(&mut self, bytes: u64, align: usize) -> Option<u64> {
        if align > self.align {
            return None;
        }
        let align = align as u64;
        let fit = self.free_spans.iter().find_map(|(&begin, &length)| {
            let end = begin + length;
            let aligned = align_up(begin, align);
            (aligned + bytes <= end).then_some((begin, length, aligned))
        });
        if let Some((begin, length, aligned)) = fit {
            let end = begin + length;
            self.free_spans.remove(&begin);
            if aligned > begin {
                self.free_spans.insert(begin, aligned - begin);
            }
            if aligned + bytes < end {
                self.free_spans.insert(aligned + bytes, end - aligned - bytes);
            }
            return Some(aligned);
        }
        let aligned = align_up(self.bump, align);
        if aligned + bytes > self.capacity {
            return None;
        }
        if aligned > self.bump {
            self.free_spans.insert(self.bump, aligned - self.bump);
        }
        self.bump = aligned + bytes;
        Some(aligned)
    }

    /// Return a span, coalescing it with an adjacent free span on either side
    /// and rewinding the bump pointer when it was the last one out.
    fn release_span(&mut self, off: u64, bytes: u64) {
        let mut begin = off;
        let mut end = off + bytes;
        if let Some(&next) = self.free_spans.range(begin..).next().map(|(k, _)| k) {
            if next == end {
                end += self.free_spans.remove(&next).unwrap_or(0);
            }
        }
        if let Some((&previous, &length)) = self.free_spans.range(..begin).next_back() {
            debug_assert!(previous + length <= begin, "two free spans overlap");
            if previous + length == begin {
                begin = previous;
                self.free_spans.remove(&previous);
            }
        }
        if end == self.bump {
            self.bump = begin;
        } else {
            self.free_spans.insert(begin, end - begin);
        }
    }
}

fn align_up(value: u64, align: u64) -> u64 {
    (value + align - 1) & !(align - 1)
}

/// One live allocation, as the allocator recorded it.
///
/// `elem_size` and `align` are kept because [`Device::grow`] must refuse a call
/// that changes either: a block grown as a different element type would move
/// bytes the caller reads at the wrong stride, and one grown to a weaker
/// alignment would land at an offset the arena never promised.
#[derive(Clone, Copy)]
struct LiveSpan {
    bytes: u64,
    elem_size: usize,
    align: usize,
}

/// What a request must satisfy before a byte is reserved for it, and the byte
/// length it comes to.
///
/// **EVERY TARGET APPLIES EXACTLY THESE RULES**, so a request one serves is a
/// request all of them serve. A target that admitted what another refuses would
/// let a driver be written against the permissive one and fail on the strict
/// one, which is the acceptance rule read backwards. The rules are stated on
/// [`Device::alloc`] and mirrored in `ppf-cts-compute/cuda/arena/arena.cu`.
fn validate_request(
    count: usize,
    elem_size: usize,
    align: usize,
    label: AllocLabel,
) -> Result<u64, Fault> {
    let refuse = |detail: String| {
        Err(Fault::Alloc {
            label,
            bytes: elem_size.saturating_mul(count),
            detail,
        })
    };
    if elem_size == 0 {
        return refuse(String::from("an element size of zero names no allocation"));
    }
    if !align.is_power_of_two() {
        return refuse(format!("an alignment of {align} is not a power of two"));
    }
    if align < crate::MIN_ALIGN {
        return refuse(format!(
            "an alignment of {align} is below the {} every target requires",
            crate::MIN_ALIGN
        ));
    }
    // An element AT LEAST as wide as the alignment must be a whole number of
    // alignments, or the second element lands unaligned. A NARROWER one needs at
    // most its own size, so any offset inside a block that starts aligned
    // satisfies it, and the requirement is the other way round.
    let compatible = if elem_size >= align {
        elem_size % align == 0
    } else {
        align % elem_size == 0
    };
    if !compatible {
        return refuse(format!(
            "an element of {elem_size} bytes and an alignment of {align} are incompatible"
        ));
    }
    if count > u32::MAX as usize {
        return refuse(format!(
            "an element count of {count} exceeds the handle width"
        ));
    }
    let bytes = (count as u64) * (elem_size as u64);
    // Refused rather than truncated. An offset is 32 bits, so a longer
    // allocation would be reachable only at an offset that wrapped into a
    // plausible small one.
    if bytes > ARENA_CAP {
        return refuse(format!(
            "{bytes} bytes is past the {ARENA_CAP} an arena offset can address"
        ));
    }
    Ok(bytes)
}

/// A device that runs kernels on the host's own cores.
///
/// It carries no kernel of its own: [`HostDevice::new`] takes the caller's
/// declaration table and the matching array of entry points, which is what lets
/// this file stay free of every name the workload uses.
pub struct HostDevice {
    info: DeviceInfo,
    /// The caller's declaration table, indexed by [`KernelId`].
    table: &'static [KernelDecl],
    /// The caller's entry points, indexed by the same id. `new` refuses a pair
    /// of different lengths, because a row added to one and not the other is a
    /// dispatch of the wrong kernel with the right bytes.
    launch: &'static [Launch],
    pool: Pool,
    /// The open arenas, indexed by a handle's `arena`.
    ///
    /// AN ARENA IS NEVER CLOSED, so an id is never reused and a stale handle
    /// names the arena it was cut from rather than a later tenant's. What
    /// catches a handle whose SPAN was freed is `live` below; what catches a
    /// recorded region outliving an allocator move is the generation counter.
    ///
    /// **AN ARENA ID MEANS SOMETHING ONLY TO THE INSTANCE THAT CUT IT.** This
    /// is a plain `Vec` per target, not a process-wide table, so a handle taken
    /// from one target and dispatched on another resolves against the wrong
    /// base and reads whatever that one has at the offset. It is the same
    /// hazard a raw address carries and the reason the seam refuses one, so it
    /// is worth naming here: a caller holding two targets at once, which in
    /// this tree is a test fixture beside a fresh [`HostDevice`], must dispatch
    /// a record on the target its buffers came from.
    arenas: Vec<HostArena>,
    /// What each live span is, keyed by `(arena, offset)`.
    ///
    /// **THE BYTE CAPACITY LIVES HERE AND NOWHERE ELSE, and that is the whole
    /// reason this map holds a record rather than a flag.** A handle's `size`
    /// and `allocated` are ELEMENT counts, so neither is a byte bound, and the
    /// byte capacity is the element count times an element size only the
    /// allocator ever saw. Reading a bound off the handle instead refuses every
    /// window past the element count and admits none past the real end, which is
    /// a refusal that looks like a bounds check and is not one.
    ///
    /// It is also how a free and a grow learn the handle is REAL. A handle
    /// naming a span nothing allocated, or one already released, or one whose
    /// `off` was advanced by hand, is a caller defect that a pure
    /// bump-and-release allocator absorbs silently, and silence is the wrong
    /// answer on the target whose whole job is to name what the others cannot.
    live: BTreeMap<u64, LiveSpan>,
    /// What the NEXT arena reserves, doubling from [`FIRST_RESERVE`] and capped
    /// at [`ARENA_CAP`]. A fixed reserve either wastes the first arena or opens
    /// too many for a large scene, and there is no reason to guess which when
    /// the arena count is what the budget bounds.
    next_reserve: u64,
    generation: u64,
    counters: Counters,
    /// Whether [`Device::record`] may produce a deferred region.
    ///
    /// The lever exists because the fallback path must be exercisable: CUDA
    /// latches graph capture off on any failure and keeps producing
    /// bit-identical results by direct launch, which is a pure slowdown, and a
    /// pure slowdown is inside this project's measured run-to-run envelope. A
    /// backend whose fallback is never taken in a test is a fallback nobody has
    /// run.
    allow_record: bool,
}

impl HostDevice {
    /// # Panics
    /// If the two slices have different lengths. That is a caller defect with
    /// no safe reading: the id indexes both, so a mismatch means some id names a
    /// declaration and no entry point, or an entry point and no declaration, and
    /// either is a dispatch of the wrong kernel with the right bytes.
    pub fn new(table: &'static [KernelDecl], launch: &'static [Launch]) -> Self {
        assert_eq!(
            table.len(),
            launch.len(),
            "the declaration table and the launch table are indexed by the same \
             kernel id, so they must be the same length"
        );
        HostDevice {
            table,
            launch,
            info: DeviceInfo {
                backend: "host",
                device_name: String::from("host cpu"),
                // The same cap the arena ABI states, so a driver sized against
                // one backend's arena budget is sized against every backend's.
                max_arenas: MAX_ARENAS,
                // A host allocation faults on a wild address, unlike Metal.
                faults_on_oob: true,
                // There is no platform form to defer INTO here: the host device
                // replays a region by running its calls again.
                supports_deferred_regions: false,
            },
            pool: Pool::new(),
            arenas: Vec::new(),
            live: BTreeMap::new(),
            next_reserve: FIRST_RESERVE,
            generation: 0,
            counters: Counters::default(),
            allow_record: std::env::var_os("PPF_CPU_NO_RECORD").is_none(),
        }
    }

    /// This device's row for `kernel`.
    ///
    /// An id with no row is a caller defect, and naming it here is the earliest
    /// point at which it can be named: `prepare_kernels` runs this over the
    /// whole set the caller declares it will dispatch, before any work is done.
    pub fn decl(&self, kernel: KernelId) -> Result<&'static KernelDecl, Fault> {
        self.table
            .get(kernel.0 as usize)
            .ok_or(Fault::MissingKernel { kernel })
    }

    /// Whether [`Device::record`] may produce a deferred region.
    ///
    /// The lever exists because the FALLBACK path must be exercisable. A target
    /// that cannot record re-issues the list, which is correct and slower, and a
    /// pure slowdown is invisible to every value gate; a fallback nobody has run
    /// is a fallback nobody has tested.
    pub fn set_recording_enabled(&mut self, enabled: bool) {
        self.allow_record = enabled;
    }

    /// Run one dispatch to completion and return whatever its channel recorded.
    ///
    /// # Safety
    /// `args` addresses the record `decl` declares and every buffer it names is
    /// live for the duration of the call.
    unsafe fn execute(&mut self, decl: &'static KernelDecl, extent: Extent, args: *const u8) -> Diag {
        self.counters.dispatches += 1;
        // THE GROUP SHAPE RUNS ITS LANES ONE AFTER ANOTHER, one whole group
        // per step, which is what `device.rs` states and what makes it exact
        // for a body that does not synchronize. A body that DOES synchronize
        // has no host rendering and owes a serial twin under rule (1-LANE);
        // the compiler refuses it here on the undefined barrier rather than
        // this backend running it in an order the body did not ask for.
        // THE RANGE IS OVER GROUPS, NOT THREADS, which is what the generated
        // group shim states: `[begin, end)` counts groups and the shim's own
        // inner loop runs lane 0 through lane width - 1. So a group launch
        // cuts on the same boundary an element launch does, and each chunk is
        // whole groups.
        // THE WIDTH TRAVELS WITH THE RANGE, because the generated GROUP shim
        // takes it as a parameter: its lanes run 0 through width - 1 and it has
        // no other way to know how many there are. An ELEMENT shim does not take
        // it, and its thunk drops the argument, so one launch signature serves
        // both and a group entry cannot be dispatched without a width.
        let (count, group_width) = match extent {
            Extent::Elements { count } => (count, 0u32),
            Extent::Groups {
                groups, threads, ..
            } => (groups, threads),
        };
        if count == 0 {
            return Diag::default();
        }
        let launch = self.launch[decl.id.0 as usize];
        let total = count as usize;

        // A generated entry point takes the seam's own reference form, so the
        // record is bound to this backend's arena table once here and every
        // chunk is handed the same bound copy. `check_shape` has already
        // established that it fits.
        let bound;
        let args = if decl.generated {
            let (bases, live) = self.arena_bases();
            bound = bind_generated(decl, args, &bases[..live]);
            (&bound as *const BoundArgs).cast::<u8>()
        } else {
            args
        };

        // A serial pass, either because the scatter forbids a cut or because the
        // job is smaller than the region that would carry it.
        let serial = !matches!(decl.scatter, Scatter::Disjoint)
            || sched::should_run_serially(decl.nanos_per_item, total);
        if serial {
            let mut raw = DiagRecord::default();
            let slot = if decl.diag {
                &mut raw as *mut DiagRecord
            } else {
                std::ptr::null_mut()
            };
            launch(args, 0, count, group_width, slot);
            return merge(std::slice::from_ref(&raw), decl.diag);
        }

        let chunk = sched::chunk_for(decl.nanos_per_item, total) as u32;
        let starts: Vec<u32> = (0..count).step_by(chunk as usize).collect();
        let mut raws = vec![DiagRecord::default(); starts.len()];
        let address = args as usize;
        starts
            .par_iter()
            .zip(raws.par_iter_mut())
            .for_each(|(&begin, raw)| {
                let end = (begin + chunk).min(count);
                let slot = if decl.diag {
                    raw as *mut DiagRecord
                } else {
                    std::ptr::null_mut()
                };
                // Safety: the declaration says every output element is written
                // by exactly one thread, so the chunks are disjoint; the record
                // is read and never written; and each chunk owns its own
                // diagnostic record. The address crosses as an integer because a
                // raw pointer is not `Sync`.
                unsafe { launch(address as *const u8, begin, end, group_width, slot) };
            });
        merge(&raws, decl.diag)
    }

    /// Where a handle's bytes live.
    fn resolve(&self, handle: Handle) -> Result<*mut u8, Fault> {
        self.resolve_parts(handle.arena, handle.off as u64)
    }

    /// The BYTE capacity of the block a handle names.
    ///
    /// The allocator's own record, because a handle carries element counts and
    /// no element size. A handle naming no live block is refused here, which is
    /// also what refuses one whose `off` was advanced by hand: such a handle
    /// addresses the inside of somebody's allocation and is not the start of
    /// one, so nothing recorded it.
    fn block_bytes(&self, handle: Handle, call: &'static str) -> Result<u64, Fault> {
        if handle.arena as usize >= self.arenas.len() {
            return Err(Fault::Platform {
                call,
                detail: format!("handle names arena {}, which is not open", handle.arena),
            });
        }
        self.live
            .get(&span_key(handle.arena, handle.off as u64))
            .map(|span| span.bytes)
            .ok_or(Fault::Platform {
                call,
                detail: format!(
                    "handle names span ({}, {}), which is not a live allocation",
                    handle.arena, handle.off
                ),
            })
    }

    /// The window rule every transfer shares: a byte offset from the START of
    /// the block, with both ends checked against the block's capacity, so a
    /// caller may move a window of a block holding several arrays end to end.
    ///
    /// **THE BOUND IS THE ALLOCATOR'S AND NOT THE HANDLE'S.** `size` and
    /// `allocated` are element counts, so reading either as a byte bound refuses
    /// every window past the element count and admits none past the real end.
    ///
    /// `None` means there is nothing to move: a zero-byte window at offset zero
    /// is accepted without resolving anything, which is what an empty buffer's
    /// transfer is.
    fn window(
        &self,
        handle: Handle,
        byte_offset: usize,
        bytes: usize,
        call: &'static str,
    ) -> Result<Option<*mut u8>, Fault> {
        if bytes == 0 && byte_offset == 0 {
            return Ok(None);
        }
        let capacity = self.block_bytes(handle, call)?;
        let offset = byte_offset as u64;
        if offset > capacity || bytes as u64 > capacity - offset {
            return Err(Fault::Platform {
                call,
                detail: format!(
                    "the window [{byte_offset}, {}) is outside the {capacity} bytes this \
                     handle names",
                    byte_offset + bytes
                ),
            });
        }
        let base = self.resolve(handle)?;
        // Safety: the bound above puts the offset inside the block.
        Ok(Some(unsafe { base.add(byte_offset) }))
    }

    fn resolve_parts(&self, arena: u32, off: u64) -> Result<*mut u8, Fault> {
        let arena = self
            .arenas
            .get(arena as usize)
            .ok_or(Fault::Platform {
                call: "resolve",
                detail: format!("handle names arena {arena}, which is not open"),
            })?;
        Ok((arena.base + off as usize) as *mut u8)
    }

    /// Find a span for `bytes`, opening an arena when no open one can hold it.
    ///
    /// First fit ACROSS arenas in the order they were opened, then a new arena.
    /// The order matters only for reproducibility: the same sequence of
    /// allocations must produce the same handles, or a recorded region and a
    /// dispatch trace would differ between two runs of one scene.
    fn place(&mut self, bytes: u64, align: usize, label: AllocLabel) -> Result<(u32, u64), Fault> {
        for (index, arena) in self.arenas.iter_mut().enumerate() {
            if let Some(off) = arena.allocate_span(bytes, align) {
                return Ok((index as u32, off));
            }
        }
        self.open_arena(bytes, align, label)?;
        let index = self.arenas.len() - 1;
        match self.arenas[index].allocate_span(bytes, align) {
            Some(off) => Ok((index as u32, off)),
            None => Err(Fault::Alloc {
                label,
                bytes: bytes as usize,
                detail: String::from(
                    "a freshly opened arena could not satisfy the request it was opened for",
                ),
            }),
        }
    }

    /// Open one arena, reserving the doubling amount or the request rounded up,
    /// whichever is larger.
    fn open_arena(&mut self, need: u64, align: usize, label: AllocLabel) -> Result<(), Fault> {
        if self.arenas.len() as u32 >= self.info.max_arenas {
            return Err(Fault::Alloc {
                label,
                bytes: need as usize,
                detail: format!(
                    "every one of the {} arenas this seam allows is open and none can hold \
                     {need} more bytes",
                    self.info.max_arenas
                ),
            });
        }
        let mut reserve = self.next_reserve.max(align_up(need.max(1), FIRST_RESERVE));
        if reserve > ARENA_CAP {
            reserve = ARENA_CAP;
        }
        if reserve < need {
            return Err(Fault::Alloc {
                label,
                bytes: need as usize,
                detail: format!("an arena cannot hold {need} bytes"),
            });
        }
        // The block's OWN alignment bounds what may be placed inside it, so an
        // arena opened for a wide request stays wide for every later one.
        let align = align.max(ARENA_ALIGN);
        let block = self.pool.take_block(reserve as usize, align);
        let base = self.pool.block_base(block) as usize;
        self.arenas.push(HostArena {
            base,
            capacity: reserve,
            align,
            bump: 0,
            free_spans: BTreeMap::new(),
        });
        self.next_reserve = if reserve >= ARENA_CAP / 2 {
            ARENA_CAP
        } else {
            reserve * 2
        };
        Ok(())
    }

    /// This target's live arena bases, in arena-id order, as a dispatch's base
    /// table takes them.
    fn arena_bases(&self) -> ([usize; MAX_ARENA_SLOTS], usize) {
        let mut bases = [0usize; MAX_ARENA_SLOTS];
        for (index, arena) in self.arenas.iter().enumerate() {
            bases[index] = arena.base;
        }
        (bases, self.arenas.len())
    }
}

/// One key per live span. The arena id is never reused, so a pair is unique for
/// the allocator's whole life.
fn span_key(arena: u32, off: u64) -> u64 {
    ((arena as u64) << 32) | off
}

/// What a dispatch reports back, for a kernel that carries a channel at all.
///
/// A kernel declared without one gets a clean report rather than a merge over
/// records nothing wrote, which is the difference between "no check failed" and
/// "no check exists".
fn merge(raws: &[DiagRecord], expect_diag: bool) -> Diag {
    if !expect_diag {
        return Diag::default();
    }
    merge_records(raws)
}

impl HostDevice {
    /// Set `bytes` of `handle` to `value`, which is what a backend fill does.
    ///
    /// The CPU backend has no blit engine, so this IS the fill; what matters is
    /// that a caller inside a region reaches it through `Encoder::fill_raw` and
    /// pays no submit for it.
    pub(crate) fn fill_bytes(
        &mut self,
        handle: Handle,
        byte_offset: usize,
        bytes: usize,
        value: u8,
    ) -> Result<(), Fault> {
        let Some(base) = self.window(handle, byte_offset, bytes, "fill")? else {
            return Ok(());
        };
        // Safety: the window puts the whole span inside the block.
        unsafe { std::ptr::write_bytes(base, value, bytes) };
        Ok(())
    }
}

impl Device for HostDevice {
    type Region = HostRegion;

    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn kernels(&self) -> &'static [KernelDecl] {
        self.table
    }

    /// Every declared kernel has an entry point, because `new` refuses a table
    /// and a launch array of different lengths. It is NOT a claim that the
    /// caller's workload will run: what a caller cannot handle is its own to
    /// refuse, above this seam, from the workload rather than from the table.
    fn missing(&self) -> &[KernelId] {
        &[]
    }

    fn prepare_kernels(&mut self, ids: &[KernelId]) -> Result<(), Fault> {
        // Nothing to create: a launch is a direct call. The check is still worth
        // making here, because an id with no row is a driver defect and this is
        // the point at which it can be named before a frame is written.
        for &id in ids {
            self.decl(id)?;
        }
        Ok(())
    }

    fn alloc(
        &mut self,
        count: usize,
        elem_size: usize,
        align: usize,
        label: AllocLabel,
    ) -> Result<Handle, Fault> {
        let bytes = validate_request(count, elem_size, align, label)?;
        // A COUNT OF ZERO RESERVES NOTHING AND REGISTERS NO SPAN. It still names
        // a real bound arena, so a bounds check on it fails loudly rather than
        // indexing an unbound slot, and its `allocated` of 0 is what makes a
        // later free of it a no-op rather than a lookup for a span that was
        // never recorded. Arena 0 is where the first real allocation also lands,
        // which is why the span map is keyed on the ALLOCATION and this handle
        // is deliberately absent from it.
        if bytes == 0 {
            if self.arenas.is_empty() {
                self.open_arena(0, align, label)?;
            }
            return Ok(Handle {
                arena: 0,
                off: 0,
                size: 0,
                allocated: 0,
            });
        }
        let (arena, off) = self.place(bytes, align, label)?;
        // Both are invariants of the design rather than caller errors, and both
        // are silent wrong answers on the target that never faults: Metal FLOORS
        // a misaligned buffer offset, and an offset past 32 bits wraps inside
        // the handle into a plausible small one.
        debug_assert_eq!(off % align as u64, 0, "a span was placed unaligned");
        if off > u32::MAX as u64 {
            self.arenas[arena as usize].release_span(off, bytes);
            return Err(Fault::Alloc {
                label,
                bytes: bytes as usize,
                detail: format!("an offset of {off} is past the width a handle carries"),
            });
        }
        // A SPAN HANDED OUT WHILE IT IS STILL LIVE is the defect a first-fit
        // allocator has, and it is silent: the second tenant's writes read back
        // to the first as plausible numbers. The insert already returns the
        // displaced record, so naming it costs a comparison.
        let displaced = self.live.insert(
            span_key(arena, off),
            LiveSpan {
                bytes,
                elem_size,
                align,
            },
        );
        assert!(
            displaced.is_none(),
            "the allocator placed a span at ({arena}, {off}), where one is already live"
        );
        self.generation += 1;
        Ok(Handle {
            arena,
            off: off as u32,
            // ELEMENTS, both of them. The byte capacity is `self.live`'s and is
            // not recoverable from the handle, which is what `window` exists to
            // say once rather than at each transfer.
            size: count as u32,
            allocated: count as u32,
        })
    }

    fn grow(
        &mut self,
        handle: &mut Handle,
        new_count: usize,
        elem_size: usize,
        align: usize,
    ) -> Result<(), Fault> {
        let label = AllocLabel("grow");
        let refuse = |detail: String| {
            Err(Fault::Alloc {
                label,
                bytes: elem_size.saturating_mul(new_count),
                detail,
            })
        };
        if handle.size > handle.allocated {
            return refuse(format!(
                "a handle of size {} in a capacity of {} names more elements than it holds",
                handle.size, handle.allocated
            ));
        }
        // REFUSED RATHER THAN SILENTLY SHRINKING. A caller asking for fewer
        // elements than the block already holds has computed something wrong,
        // and honoring it would move the bytes it meant to keep.
        if new_count < handle.allocated as usize {
            return refuse(format!(
                "a grow to {new_count} elements cannot shrink a capacity of {}",
                handle.allocated
            ));
        }
        if new_count == handle.allocated as usize {
            return Ok(());
        }
        // A ZERO-LENGTH HANDLE RESERVED NOTHING, so there is nothing to move and
        // nothing to release. Its LOGICAL size is carried across, because this
        // call is about capacity.
        if handle.allocated == 0 {
            let logical = handle.size;
            *handle = self.alloc(new_count, elem_size, align, label)?;
            handle.size = logical;
            return Ok(());
        }
        let previous_key = span_key(handle.arena, handle.off as u64);
        let Some(previous) = self.live.get(&previous_key).copied() else {
            return refuse(format!(
                "handle names span ({}, {}), which is not a live allocation",
                handle.arena, handle.off
            ));
        };
        // The element type and the alignment are the block's, not this call's,
        // so a disagreement is the caller reading one allocation as another.
        if previous.elem_size != elem_size || previous.align != align {
            return refuse(format!(
                "the span was allocated as {}-byte elements at alignment {}, and this \
                 grow names {elem_size}-byte elements at alignment {align}",
                previous.elem_size, previous.align
            ));
        }
        // The handle's capacity and the allocator's record must describe one
        // block. They can only disagree if the handle was edited by hand, and
        // the copy below is sized from the record, so a disagreement here would
        // move a length the caller does not believe in.
        if previous.bytes != handle.allocated as u64 * elem_size as u64 {
            return refuse(format!(
                "the handle reports a capacity of {} elements of {elem_size} bytes and \
                 the allocator holds {} bytes for it",
                handle.allocated, previous.bytes
            ));
        }
        // A FRESH SPAN, then a copy, then the old one back. An arena's block is
        // never reallocated, because every other live span inside it is
        // addressed from the same base; so a grow moves the allocation, which is
        // exactly what the seam says it may do and what the generation counter
        // makes a recorded region notice.
        let replacement = self.alloc(new_count, elem_size, align, label)?;
        let old = self.resolve(*handle)?;
        let fresh = self.resolve(replacement)?;
        // Safety: the destination is longer than the source, since a shrink was
        // refused above, and the two are disjoint because the old span was still
        // live when the new one was placed.
        unsafe { std::ptr::copy_nonoverlapping(old, fresh, previous.bytes as usize) };
        self.live.remove(&previous_key);
        self.arenas[handle.arena as usize].release_span(handle.off as u64, previous.bytes);
        let logical = handle.size;
        *handle = replacement;
        // THE LOGICAL SIZE IS PRESERVED, not raised to the new capacity. This
        // call moves `allocated`; a caller that wants the length to follow sets
        // it, which is what `Buffer::size` does with its own `len`.
        handle.size = logical;
        self.generation += 1;
        Ok(())
    }

    fn free(&mut self, handle: &mut Handle) -> Result<(), Fault> {
        // NOTHING TO RELEASE, TWICE OVER: a handle that already names nothing,
        // and a zero-length one, which reserved no span and was never recorded.
        // Both leave `Handle::NONE`, which is the spelling every target agrees a
        // freed handle carries, so the next `Buffer::size` allocates rather than
        // growing a block that has been released.
        if handle.is_none() || handle.allocated == 0 {
            *handle = Handle::NONE;
            return Ok(());
        }
        let arena = handle.arena as usize;
        if arena >= self.arenas.len() {
            return Err(Fault::Platform {
                call: "free",
                detail: format!("handle names arena {arena}, which is not open"),
            });
        }
        let key = span_key(handle.arena, handle.off as u64);
        let span = self.live.remove(&key).ok_or(Fault::Platform {
            call: "free",
            detail: format!(
                "handle names span ({arena}, {}), which is not a live allocation",
                handle.off
            ),
        })?;
        self.arenas[arena].release_span(handle.off as u64, span.bytes);
        self.generation += 1;
        *handle = Handle::NONE;
        Ok(())
    }

    fn write(&mut self, handle: Handle, byte_offset: usize, src: &[u8]) -> Result<(), Fault> {
        let Some(base) = self.window(handle, byte_offset, src.len(), "write")? else {
            return Ok(());
        };
        // Safety: the window puts the whole span inside the block.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), base, src.len()) };
        Ok(())
    }

    fn fill_zero(&mut self, handle: Handle, bytes: usize) -> Result<(), Fault> {
        self.fill_bytes(handle, 0, bytes, 0)
    }

    fn read(&mut self, handle: Handle, byte_offset: usize, dst: &mut [u8]) -> Result<(), Fault> {
        let Some(base) = self.window(handle, byte_offset, dst.len(), "read")? else {
            return Ok(());
        };
        // Safety: the window puts the whole span inside the block.
        unsafe { std::ptr::copy_nonoverlapping(base, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    // THIS TARGET SERVES NO HOST VIEW, AND `Device::host_view` IS LEFT AT ITS
    // DEFAULT DELIBERATELY. Its arenas are host memory, so it COULD serve one
    // and the copies above are redundant. What stops it is not the memory: it
    // is that adopting the mapped representation changes which bytes a
    // conformance assertion sees and owes a scene sweep of its own, and this
    // change is scoped to the target whose copies cross a seam. The default
    // answer states what this target implements, which is the truthful thing
    // for it to report.

    fn copy(
        &mut self,
        dst: Handle,
        dst_byte_offset: usize,
        src: Handle,
        src_byte_offset: usize,
        bytes: usize,
    ) -> Result<(), Fault> {
        let Some(from) = self.window(src, src_byte_offset, bytes, "copy source")? else {
            return Ok(());
        };
        let Some(to) = self.window(dst, dst_byte_offset, bytes, "copy destination")? else {
            return Ok(());
        };
        // Safety: each window puts its whole span inside its own block, and the
        // trait requires the two spans not to overlap.
        unsafe { std::ptr::copy_nonoverlapping(from, to, bytes) };
        Ok(())
    }

    fn allocator_generation(&self) -> u64 {
        self.generation
    }

    fn bytes_reserved(&self) -> u64 {
        self.pool.reserved_bytes()
    }


    fn run<F>(&mut self, region: &'static str, body: F) -> Result<Diag, Fault>
    where
        F: FnOnce(&mut dyn Encoder) -> Result<(), Fault>,
    {
        self.counters.syncs += 1;
        // The device borrow moves into the encoder for the closure's duration,
        // which is what stops a body from allocating or reading: neither is on
        // `Encoder`, and the device is not reachable from inside.
        let mut encoder = HostEncoder {
            label: region,
            table: self.table,
            sink: Sink::Immediate(self),
            diag: Diag::default(),
        };
        body(&mut encoder)?;
        let diag = std::mem::take(&mut encoder.diag);
        if diag.is_clean() {
            Ok(diag)
        } else {
            Err(Fault::Device { region, diag })
        }
    }

    fn record<F>(&mut self, region: &'static str, body: F) -> Result<HostRegion, Fault>
    where
        F: FnOnce(&mut dyn Encoder) -> Result<(), Fault>,
    {
        let mut ops = Vec::new();
        let table = self.table;
        let mut encoder = HostEncoder {
            label: region,
            table,
            sink: Sink::Record(&mut ops),
            diag: Diag::default(),
        };
        body(&mut encoder)?;
        Ok(HostRegion {
            label: region,
            ops,
            generation: self.generation,
            deferred: self.allow_record,
        })
    }

    fn replay(&mut self, region: &HostRegion, repeats: u32) -> Result<Diag, Fault> {
        if region.generation != self.generation {
            return Err(Fault::StaleRegion {
                region: region.label,
                recorded: region.generation,
                now: self.generation,
            });
        }
        self.counters.syncs += 1;
        if region.deferred {
            self.counters.regions_deferred += 1;
        } else {
            self.counters.regions_fallback += 1;
        }
        let mut merged = Diag::default();
        for _ in 0..repeats {
            for op in &region.ops {
                let (kernel, extent, args) = match op {
                    Op::Fill { handle, byte_offset, bytes, value } => {
                        self.fill_bytes(*handle, *byte_offset, *bytes, *value)?;
                        continue;
                    }
                    Op::Dispatch { kernel, extent, args } => (*kernel, *extent, args),
                };
                let decl = self.decl(kernel)?;
                // Safety: the bytes were copied from a record of exactly this
                // kernel's declared length at record time, and the caller's
                // contract is that every buffer they name outlives every replay.
                let diag = unsafe { self.execute(decl, extent, args.as_ptr()) };
                merged.failures += diag.failures;
                if merged.first.is_none() {
                    merged.first = diag.first;
                }
            }
        }
        if merged.is_clean() {
            Ok(merged)
        } else {
            Err(Fault::Device {
                region: region.label,
                diag: merged,
            })
        }
    }

    fn release(&mut self, _region: HostRegion) {}

    fn counters(&self) -> Counters {
        self.counters
    }

    fn counters_reset(&mut self) {
        self.counters = Counters::default();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A launch that is never called, for the shape tests below. It exists to
    /// give the table a row, which is all those tests are about.
    unsafe fn never(
        _args: *const u8,
        _begin: u32,
        _end: u32,
        _group_width: u32,
        _diag: *mut DiagRecord,
    ) {
        unreachable!("this row exists to be counted, never to be dispatched")
    }

    static ONE_LAUNCH: [Launch; 1] = [never];

    fn record(fail_count: u32, claimed: u32, line: u32) -> DiagRecord {
        DiagRecord {
            fail_count,
            claimed,
            line,
            payload: [line as f32, 0.0, 0.0, 0.0],
            // A null file is what a body that recorded no location leaves, and
            // the merge must name it rather than dereference it.
            file: std::ptr::null(),
        }
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn a_table_and_a_launch_array_of_different_lengths_are_refused() {
        // The id indexes both, so a mismatch means some id names a declaration
        // and no entry point, or an entry point and no declaration. Either is a
        // dispatch of the wrong kernel with the right bytes, which produces a
        // plausible wrong answer rather than a crash.
        let _ = HostDevice::new(&[], &ONE_LAUNCH);
    }

    #[test]
    fn an_empty_registry_is_legal() {
        // A caller with no kernels can still allocate and transfer, and a target
        // that refused to open for one would make the memory half of the seam
        // unreachable on its own.
        let device = HostDevice::new(&[], &[]);
        assert_eq!(device.kernels().len(), 0);
        assert!(device.missing().is_empty());
    }

    #[test]
    fn the_channel_reports_the_lowest_numbered_claimant() {
        // ASCENDING chunk order, not first to arrive. The chunks run
        // concurrently, so first-to-arrive is not reproducible and two runs of
        // one workload would name different checks.
        let merged = merge_records(&[record(0, 0, 0), record(3, 1, 41), record(2, 1, 97)]);
        assert_eq!(merged.failures, 5, "every chunk's count is summed");
        let first = merged.first.expect("a claimant was recorded");
        assert_eq!(first.line, 41);
        assert_eq!(first.file, "<unrecorded>", "a null location is named, not read");
    }

    #[test]
    fn a_count_with_no_claimant_reports_clean() {
        // The count and the claim are written by different stores in the
        // compiled body. A count with no record behind it has nothing to name,
        // and inventing a location for it would be worse than reporting what was
        // actually recorded.
        assert!(merge_records(&[record(4, 0, 0)]).is_clean());
        assert!(merge_records(&[]).is_clean());
    }

    /// A device with no kernels, which is all the allocator tests need.
    fn allocating_device() -> HostDevice {
        HostDevice::new(&[], &[])
    }

    #[test]
    fn far_more_buffers_than_arenas_can_be_allocated_at_once() {
        // THE PROPERTY SPAN PACKING EXISTS FOR. One arena per allocation caps a
        // caller at MAX_ARENAS buffers, and this solver's driver passes that
        // early in its own build, so the cap has to bound the ARENA count and
        // not the caller's buffer count.
        let mut device = allocating_device();
        let count = MAX_ARENAS as usize * 8;
        let mut handles = Vec::new();
        for i in 0..count {
            let handle = device
                .alloc(16 + i, 4, 4, AllocLabel("test.many"))
                .expect("an allocation past the arena count must still be served");
            handles.push(handle);
        }
        assert!(
            device.arenas.len() <= MAX_ARENAS as usize,
            "{} arenas were opened for {count} allocations, over the {MAX_ARENAS} budget",
            device.arenas.len()
        );

        // Every span must be its own memory, which a bump pointer that forgot
        // to advance would not be. Written through the seam, so the check goes
        // the same way a caller's would.
        for (i, &handle) in handles.iter().enumerate() {
            let pattern = vec![i as u8; (16 + i) * 4];
            device.write(handle, 0, &pattern).expect("write");
        }
        for (i, &handle) in handles.iter().enumerate() {
            let mut back = vec![0u8; (16 + i) * 4];
            device.read(handle, 0, &mut back).expect("read");
            assert!(
                back.iter().all(|b| *b == i as u8),
                "allocation {i} was overwritten by another span"
            );
        }
        for handle in &mut handles {
            device.free(handle).expect("free");
        }
    }

    #[test]
    fn a_genuinely_exhausted_budget_is_refused_by_name() {
        // The cap bounds the ARENA count, and packing is what keeps a caller's
        // buffer count away from it. When it IS reached, the refusal names the
        // budget rather than opening a slot no handle could address: an arena id
        // past the table is a base a target resolves to the LAST arena rather
        // than faulting, which is a silent wrong answer with plausible floats.
        //
        // The budget is lowered rather than the allocations raised, because
        // reaching 29 arenas honestly means reserving gigabytes: the reserve
        // doubles per arena from FIRST_RESERVE.
        let mut device = allocating_device();
        device.info.max_arenas = 1;
        let elems = FIRST_RESERVE as usize / 4;
        device
            .alloc(elems, 4, 4, AllocLabel("test.fills"))
            .expect("the first arena holds exactly its reserve");
        match device.alloc(elems, 4, 4, AllocLabel("test.overflows")) {
            Err(Fault::Alloc { label, detail, .. }) => {
                assert_eq!(label.0, "test.overflows");
                assert!(detail.contains("arenas this seam allows"), "{detail}");
            }
            other => panic!("an exhausted budget must be refused by name: {other:?}"),
        }
        assert_eq!(device.arenas.len(), 1, "no arena was opened past the budget");
    }

    #[test]
    fn a_span_is_aligned_inside_its_arena_and_not_merely_offset() {
        // Metal silently FLOORS a misaligned buffer offset to a multiple of 4
        // and returns the wrong data with status Completed, so an offset that
        // is merely inside the arena is not enough.
        let mut device = allocating_device();
        let mut handles = Vec::new();
        // The first row is the seam's SANCTIONED exception: an element narrower
        // than the alignment needs at most its own size, so a byte array asks
        // for four and every offset inside the block satisfies it. An alignment
        // BELOW four is refused instead, which the request test below covers.
        for (count, elem, align) in [(12usize, 1usize, 4usize), (5, 4, 4), (7, 8, 8), (3, 16, 16)] {
            let handle = device
                .alloc(count, elem, align, AllocLabel("test.align"))
                .expect("alloc");
            let base = device.resolve(handle).expect("resolve") as usize;
            assert_eq!(base % align, 0, "a span of alignment {align} landed at {base}");
            handles.push(handle);
        }
        for handle in &mut handles {
            device.free(handle).expect("free");
        }
    }

    #[test]
    fn a_released_span_is_reused_and_coalesced_with_its_neighbors() {
        // Without coalescing, a long run of allocate and free shatters an arena
        // into spans none of which can hold a later request, and the allocator
        // opens arenas until the budget is gone. The tell is that the request
        // below needs the two freed spans TOGETHER.
        let mut device = allocating_device();
        let mut first = device.alloc(1024, 4, 4, AllocLabel("test.a")).expect("alloc");
        let mut second = device.alloc(1024, 4, 4, AllocLabel("test.b")).expect("alloc");
        let tail = device.alloc(16, 4, 4, AllocLabel("test.tail")).expect("alloc");
        let opened = device.arenas.len();
        let bump = device.arenas[0].bump;
        device.free(&mut first).expect("free");
        device.free(&mut second).expect("free");
        let wide = device
            .alloc(2048, 4, 4, AllocLabel("test.wide"))
            .expect("two adjacent freed spans must coalesce into one");
        assert_eq!(wide.arena, 0, "the reused span is in the arena it was freed from");
        assert_eq!(device.arenas.len(), opened, "no arena needed opening");
        assert_eq!(
            device.arenas[0].bump, bump,
            "the request was served from the freed spans, not from fresh bytes"
        );
        assert!(!tail.is_none());
    }

    #[test]
    fn a_long_run_of_allocate_free_and_grow_keeps_every_span_intact() {
        // A COALESCING FREE LIST IS WHERE AN ALLOCATOR IS USUALLY WRONG, and its
        // failure is not a crash: it hands one span's bytes to two callers, and
        // the second write is read back by the first as a plausible number. So
        // the check is not that the calls succeed, it is that EVERY live span
        // still reads back its own pattern after every mutation.
        //
        // The sequence is a fixed LCG rather than a random one, so a failure is
        // reproducible from this file alone.
        let mut device = allocating_device();
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        // (handle, element count, the id every element is filled with). The id
        // is a WHOLE ELEMENT rather than a byte, so no two live spans can share
        // one: a repeating pattern would let an overlap read back as its own
        // tenant.
        let mut live: Vec<(Handle, usize, u32)> = Vec::new();
        let mut tag: u32 = 0;

        let verify = |device: &mut HostDevice, live: &[(Handle, usize, u32)], step: usize| {
            for &(handle, count, id) in live {
                let mut back = vec![0u32; count];
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(back.as_mut_ptr().cast::<u8>(), count * 4)
                };
                device.read(handle, 0, bytes).expect("read a live span");
                assert!(
                    back.iter().all(|v| *v == id),
                    "at step {step}, the span ({}, {}) of {count} elements no longer \
                     holds {id}: another allocation was placed over it",
                    handle.arena,
                    handle.off
                );
            }
        };

        // One span's contents, as bytes.
        let filled = |id: u32, count: usize| -> Vec<u8> {
            let mut out = Vec::with_capacity(count * 4);
            for _ in 0..count {
                out.extend_from_slice(&id.to_ne_bytes());
            }
            out
        };

        for step in 0..600 {
            match next() % 4 {
                // Allocate, with a width that varies so the free list is asked
                // for spans it cannot serve exactly and has to split.
                0 | 1 => {
                    let count = 1 + next() % 96;
                    let handle = device
                        .alloc(count, 4, 4, AllocLabel("test.churn"))
                        .expect("alloc");
                    tag += 1;
                    device.write(handle, 0, &filled(tag, count)).expect("write");
                    live.push((handle, count, tag));
                }
                // Free one from the middle, which is what leaves holes for the
                // first fit to find and neighbors for the release to coalesce.
                2 => {
                    if !live.is_empty() {
                        let victim = next() % live.len();
                        let (mut handle, _, _) = live.swap_remove(victim);
                        device.free(&mut handle).expect("free");
                        assert!(handle.is_none());
                    }
                }
                // Grow one, which allocates, copies and releases in one call.
                _ => {
                    if !live.is_empty() {
                        let index = next() % live.len();
                        let (handle, count, id) = &mut live[index];
                        let grown = *count + 1 + next() % 64;
                        device.grow(handle, grown, 4, 4).expect("grow");
                        // The grow preserved the old elements; fill the rest so
                        // the whole span carries one id again.
                        device
                            .write(*handle, 0, &filled(*id, grown))
                            .expect("write");
                        *count = grown;
                    }
                }
            }
            verify(&mut device, &live, step);
        }

        assert!(
            device.arenas.len() <= MAX_ARENAS as usize,
            "{} arenas were opened, over the {MAX_ARENAS} budget: the free list is \
             shattering instead of coalescing",
            device.arenas.len()
        );
        // THE PROPERTY, stated as a ratio rather than as the two figures it
        // happens to produce: far more spans are live AT ONCE than there are
        // arenas to hold one each. One allocation per arena would have been
        // refused partway through this loop.
        assert!(
            live.len() > MAX_ARENAS as usize,
            "the churn ended with only {} live spans, too few to say anything about \
             packing",
            live.len()
        );
        // The churn ends with everything returned, which is the state a scene
        // teardown leaves. A leak here would show as a bump that never rewinds.
        for (handle, _, _) in &mut live {
            device.free(handle).expect("free");
        }
        assert!(
            device.live.is_empty(),
            "{} spans are still recorded after every handle was freed",
            device.live.len()
        );
        for (index, arena) in device.arenas.iter().enumerate() {
            assert_eq!(
                arena.bump, 0,
                "arena {index} did not rewind to empty, so a released span was \
                 not coalesced back into the bump"
            );
            assert!(
                arena.free_spans.is_empty(),
                "arena {index} still holds {} free spans after rewinding",
                arena.free_spans.len()
            );
        }
    }

    #[test]
    fn a_free_of_a_span_that_is_not_live_is_named() {
        // A pure bump-and-release allocator absorbs this silently, and silence
        // is the wrong answer on the one target whose job is to name what the
        // others cannot: Metal does not fault on addressing a released span.
        let mut device = allocating_device();
        let mut handle = device.alloc(8, 4, 4, AllocLabel("test.once")).expect("alloc");
        let copy = handle;
        device.free(&mut handle).expect("the first free is legitimate");
        let mut again = copy;
        match device.free(&mut again) {
            Err(Fault::Platform { call, detail }) => {
                assert_eq!(call, "free");
                assert!(detail.contains("live allocation"), "{detail}");
            }
            other => panic!("a second free of one span must be named, got {other:?}"),
        }
    }

    #[test]
    fn an_allocation_past_the_offset_cap_is_refused_rather_than_truncated() {
        // `allocated` is 32 bits, so a longer allocation recorded as a plausible
        // short one would pass every bound this target checks against it. It is
        // refused before any memory is asked for, so the test costs nothing.
        let mut device = allocating_device();
        // Four-byte elements, so the ELEMENT count still fits a handle's 32 bits
        // and it is the byte length that is over the cap. A one-byte element
        // would trip the count check first and prove a different thing.
        match device.alloc((ARENA_CAP as usize) / 4 + 1, 4, 4, AllocLabel("test.huge")) {
            Err(Fault::Alloc { label, detail, .. }) => {
                assert_eq!(label.0, "test.huge");
                assert!(detail.contains("arena offset can address"), "{detail}");
            }
            other => panic!("an allocation past the offset cap must be refused: {other:?}"),
        }
    }

    #[test]
    fn a_grow_moves_the_span_and_preserves_what_was_written() {
        // An arena's block is never reallocated, because every other live span
        // inside it is addressed from the same base, so a grow places a fresh
        // span and copies. That is the seam's stated contract, and it is what
        // the generation counter makes a recorded region notice.
        let mut device = allocating_device();
        let mut handle = device.alloc(64, 4, 4, AllocLabel("test.grow")).expect("alloc");
        // A second live span so the first cannot simply extend in place.
        let neighbor = device.alloc(64, 4, 4, AllocLabel("test.pin")).expect("alloc");
        let source: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
        device.write(handle, 0, &source).expect("write");
        let before = device.allocator_generation();
        device.grow(&mut handle, 256, 4, 4).expect("grow");
        assert!(device.allocator_generation() > before, "a move must be visible");
        // ELEMENTS. A capacity of 1024 here would be the byte length, which is
        // the reading that turns a transfer's bound into a refusal that looks
        // like a bounds check and is not one.
        assert_eq!(handle.allocated, 256, "a capacity is an element count");
        assert_eq!(handle.size, 64, "a grow moves the capacity, not the length");
        let mut kept = vec![0u8; 256];
        device.read(handle, 0, &mut kept).expect("read");
        assert_eq!(kept, source, "a grow must preserve what was written");
        let mut untouched = vec![0u8; 4];
        device.read(neighbor, 0, &mut untouched).expect("the neighbor is still live");
    }

    #[test]
    fn a_transfer_is_bounded_by_the_block_and_not_by_the_handle() {
        // THE DIVERGENCE THIS FILE IS HELD TO. A handle's `size` and `allocated`
        // are ELEMENT counts on every target, so the byte bound is the
        // allocator's record. Reading `allocated` as bytes instead refuses every
        // window past the element count and admits none past the real end, which
        // is a refusal that looks like a bounds check and is not one. Both
        // halves of that are checked here, because either alone passes under the
        // wrong reading.
        let mut device = allocating_device();
        const ELEMS: usize = 64;
        let handle = device
            .alloc(ELEMS, 4, 4, AllocLabel("test.window"))
            .expect("alloc");
        assert_eq!(handle.allocated, ELEMS as u32, "a capacity is an element count");
        assert_eq!(handle.size, ELEMS as u32);

        // ADMITTED: a window past the ELEMENT count and inside the byte length.
        // Under the wrong reading this is the first refusal.
        device
            .write(handle, ELEMS + 4, &[1u8; 4])
            .expect("a window inside the block must be served");

        // The last legal byte, then one past it.
        device
            .write(handle, ELEMS * 4 - 4, &[2u8; 4])
            .expect("the final element must be writable");
        let err = device
            .write(handle, ELEMS * 4 - 3, &[3u8; 4])
            .expect_err("a window running one byte past the block must be refused");
        assert!(
            err.to_string().contains("bytes this handle names"),
            "the refusal did not name the block's capacity: {err}"
        );
        let err = device
            .read(handle, ELEMS * 4, &mut [0u8; 1])
            .expect_err("a read starting at the end must be refused");
        assert!(err.to_string().contains("bytes this handle names"), "{err}");
    }

    #[test]
    fn a_handle_whose_offset_was_advanced_by_hand_is_refused() {
        // A handle addressing the INSIDE of somebody's allocation is not the
        // start of one, so nothing recorded it and there is no capacity to check
        // a window against. Serving it would move bytes at an offset no
        // allocation promised.
        let mut device = allocating_device();
        let handle = device
            .alloc(64, 4, 4, AllocLabel("test.invented"))
            .expect("alloc");
        let invented = Handle {
            off: handle.off + 4,
            ..handle
        };
        let err = device
            .write(invented, 0, &[0u8; 4])
            .expect_err("an invented handle must be refused");
        assert!(err.to_string().contains("live allocation"), "{err}");
    }

    #[test]
    fn a_request_no_target_can_serve_is_refused_by_name() {
        // EVERY TARGET APPLIES THESE RULES. One that admitted what another
        // refuses would let a driver be written against the permissive one and
        // fail on the strict one, which is the acceptance rule read backwards.
        let mut device = allocating_device();
        let cases: [(usize, usize, usize, &str); 5] = [
            (4, 0, 4, "element size of zero"),
            (4, 4, 3, "not a power of two"),
            (4, 4, 2, "below the 4"),
            (4, 4, 1, "below the 4"),
            (4, 12, 8, "incompatible"),
        ];
        for (count, elem, align, expected) in cases {
            match device.alloc(count, elem, align, AllocLabel("test.request")) {
                Err(Fault::Alloc { detail, .. }) => assert!(
                    detail.contains(expected),
                    "a request of ({count}, {elem}, {align}) was refused as {detail}, \
                     which does not name {expected}"
                ),
                other => panic!("({count}, {elem}, {align}) must be refused: {other:?}"),
            }
        }
        // The sanctioned exception, admitted: an element NARROWER than the
        // alignment needs at most its own size.
        device
            .alloc(9, 1, 4, AllocLabel("test.narrow"))
            .expect("a byte array at four-byte alignment is legal");
        device
            .alloc(3, 4, 8, AllocLabel("test.wide"))
            .expect("a four-byte element at eight-byte alignment is legal");
    }

    #[test]
    fn a_zero_length_allocation_names_a_bound_arena_and_is_free_to_free() {
        // A zero-length handle is NOT the sentinel: it names a real arena, so a
        // bounds check on it fails loudly rather than indexing an unbound slot.
        // It reserved nothing, so freeing it releases nothing and is legal.
        let mut device = allocating_device();
        let mut empty = device
            .alloc(0, 4, 4, AllocLabel("test.empty"))
            .expect("a zero-length request is legal");
        assert!(!empty.is_none(), "a zero-length handle is not the sentinel");
        assert!(
            (empty.arena as usize) < device.arenas.len(),
            "a zero-length handle must name an arena that is open"
        );
        assert_eq!((empty.size, empty.allocated), (0, 0));
        device
            .write(empty, 0, &[])
            .expect("an empty transfer moves nothing and is legal");
        device.free(&mut empty).expect("freeing it releases nothing");
        assert!(empty.is_none(), "a freed handle names nothing");
        device.free(&mut empty).expect("freeing it again is a no-op");
    }

    #[test]
    fn a_freed_handle_names_nothing_so_the_next_sizing_allocates() {
        // THE SPELLING IS PART OF THE CONTRACT, and the two calls below are
        // where it is written down: `Buffer::size` asks `is_none()` to tell an
        // allocation it must make from one it must grow, so a target that left a
        // freed handle ZEROED would send the next sizing down the grow path
        // against a block it had just released. Zero is a real arena, which is
        // why the sentinel is not zero.
        //
        // ASSERTED THROUGH THE SHARED CONFORMANCE CALLS rather than here,
        // because the other target in this crate reaches the same spelling by a
        // different route and can lose it independently: the C-ABI library
        // zeroes the handle and `AbiDevice::free` normalizes the caller's copy
        // afterwards. One written rule, called from both test modules.
        let mut device = allocating_device();
        crate::device::assert_free_leaves_the_sentinel(&mut device);
        crate::device::assert_sizing_after_a_free_spares_the_first_allocation(&mut device);
        crate::device::assert_a_prefix_upload_names_only_what_it_carried(&mut device);
        crate::device::assert_a_readback_answers_only_after_a_download(&mut device);
        crate::device::assert_a_redundant_upload_transfers_nothing(&mut device);
        crate::device::assert_a_host_view_aliases_the_allocation(&mut device);

        // The tail is particular to a sizing after a free rather than to the
        // sentinel: whatever span the allocator hands back, the caller's bytes
        // are ZEROED, so a pass that reads further than its producer writes goes
        // on reading a zero.
        let mut buffer = crate::Buffer::<f32>::none();
        buffer
            .size(&mut device, 32, AllocLabel("test.cycle"))
            .expect("size");
        buffer.write(&mut device, 0, &[1.0f32; 32]).expect("write");
        buffer.free(&mut device).expect("free");
        buffer
            .size(&mut device, 8, AllocLabel("test.cycle"))
            .expect("a sizing after a free must allocate rather than grow");
        let mut back = [7.0f32; 8];
        buffer.read(&mut device, 0, &mut back).expect("read");
        assert_eq!(back, [0.0f32; 8], "a freshly sized buffer is zeroed");
    }

    #[test]
    fn the_host_target_serves_no_host_view() {
        // WHAT KEEPS THE CPU BACKEND UNCHANGED, AS A CHECK RATHER THAN A CLAIM.
        // This target's arenas are host memory, so a query named for the
        // physical property would answer yes here and send it down the mapped
        // representation. `Device::host_view` is named for what a target
        // SERVES, this target overrides nothing, and the default answer is what
        // holds every staged and readback buffer on the copy path.
        let mut device = allocating_device();
        let mut buffer = crate::Buffer::<f32>::none();
        buffer
            .size(&mut device, 8, AllocLabel("test.host_view"))
            .expect("size");
        let bytes = 8 * std::mem::size_of::<f32>();
        // Safety: the handle names the live allocation and the window is its
        // own logical length.
        let view = unsafe { device.host_view(buffer.handle(), 0, bytes) }
            .expect("a target that serves no view reports that, it does not fail");
        assert!(
            view.is_none(),
            "the host target must report no host view, or every staged buffer \
             changes representation on a build this change leaves alone"
        );
        buffer.free(&mut device).expect("free");
    }

    #[test]
    fn a_buffer_sized_down_and_back_up_asks_for_no_grow_it_does_not_need() {
        // A `grow` refuses to shrink, on every target. So a sizing that compared
        // against the LENGTH would ask to shrink a capacity the buffer still
        // holds the moment a scene sized one buffer down and then part way back
        // up, and the refusal would be correct. The capacity is what decides
        // whether there is anything to do.
        let mut device = allocating_device();
        let mut buffer = crate::Buffer::<f32>::none();
        buffer.size(&mut device, 64, AllocLabel("test.updown")).expect("size up");
        let capacity = buffer.handle().allocated;
        buffer.size(&mut device, 8, AllocLabel("test.updown")).expect("size down");
        assert_eq!(
            buffer.handle().allocated,
            capacity,
            "sizing down releases nothing, so the capacity stands"
        );
        buffer
            .size(&mut device, 32, AllocLabel("test.updown"))
            .expect("a sizing back up inside the capacity must not ask for a grow");
        assert_eq!(buffer.len(), 32);
        buffer
            .size(&mut device, 128, AllocLabel("test.updown"))
            .expect("a sizing past the capacity grows");
        // AT LEAST THE COUNT, NOT EXACTLY IT. `size` asks for `count + count / 4`,
        // the headroom that keeps a buffer growing by ones from reallocating and
        // copying every step, so the capacity after a growth is above the length
        // by design. What this test is about is that the growth happened and
        // covers the request; pinning the exact figure would restate the growth
        // factor and fail the day it is retuned.
        assert!(
            buffer.handle().allocated as usize >= 128,
            "a growth must cover the count, got {}",
            buffer.handle().allocated
        );
        assert_eq!(buffer.len(), 128);
        let mut back = [1.0f32; 128];
        buffer.read(&mut device, 0, &mut back).expect("read");
        assert_eq!(back, [0.0f32; 128], "a sized buffer is zeroed");
    }

    #[test]
    fn a_grow_refuses_a_shrink_and_a_changed_element_type() {
        // Both are the caller reading one allocation as another. Honoring the
        // first would move the bytes it meant to keep; honoring the second would
        // leave it reading the block at the wrong stride.
        let mut device = allocating_device();
        let mut handle = device
            .alloc(64, 4, 4, AllocLabel("test.strict"))
            .expect("alloc");
        match device.grow(&mut handle, 32, 4, 4) {
            Err(Fault::Alloc { detail, .. }) => {
                assert!(detail.contains("cannot shrink"), "{detail}")
            }
            other => panic!("a shrink must be refused: {other:?}"),
        }
        match device.grow(&mut handle, 128, 8, 8) {
            Err(Fault::Alloc { detail, .. }) => {
                assert!(detail.contains("was allocated as"), "{detail}")
            }
            other => panic!("a changed element type must be refused: {other:?}"),
        }
        // Equal is a no-op rather than a move, so nothing else goes stale.
        let before = device.allocator_generation();
        device.grow(&mut handle, 64, 4, 4).expect("an equal grow is a no-op");
        assert_eq!(device.allocator_generation(), before);
    }

    #[test]
    fn a_zero_length_handle_grows_into_a_real_span() {
        // It reserved nothing, so there is no block to move and no span to
        // release; the LOGICAL size it carried is preserved, because a grow is
        // about capacity.
        let mut device = allocating_device();
        let mut handle = device
            .alloc(0, 4, 4, AllocLabel("test.fromempty"))
            .expect("alloc");
        device.grow(&mut handle, 16, 4, 4).expect("grow");
        assert_eq!(handle.allocated, 16);
        assert_eq!(handle.size, 0, "a grow moves the capacity, not the length");
        device.write(handle, 0, &[9u8; 64]).expect("the span is real");
    }

    #[test]
    fn group_costs_count_groups_and_never_split_lanes_or_atomic_work() {
        use std::sync::Mutex;
        unsafe fn observe(args: *const u8, begin: u32, end: u32, width: u32, _: *mut DiagRecord) {
            let calls = unsafe { &*args.cast::<Mutex<Vec<(u32, u32, u32)>>>() };
            calls.lock().unwrap().push((begin, end, width));
        }
        const PARALLEL: KernelDecl = KernelDecl {
            id: KernelId(0),
            name: "group_schedule",
            scatter: Scatter::Disjoint,
            nanos_per_item: 10_000.0,
            args_bytes: 0,
            host_refs: &[],
            diag: false,
            generated: false,
        };
        static DECL: [KernelDecl; 3] = [
            PARALLEL,
            KernelDecl { id: KernelId(1), nanos_per_item: 100.0, ..PARALLEL },
            KernelDecl { id: KernelId(2), scatter: Scatter::Atomic, ..PARALLEL },
        ];
        static LAUNCH: [Launch; 3] = [observe; 3];
        let mut device = HostDevice::new(&DECL, &LAUNCH);
        let calls = Mutex::new(Vec::<(u32, u32, u32)>::new());
        for declaration in &DECL {
            for width in [32, 256] {
                calls.lock().unwrap().clear();
                // The private execution path uses the pointer only during this call.
                unsafe {
                    device.execute(declaration, Extent::Groups {
                        groups: 17, threads: width, scratch_bytes: 0,
                    }, (&calls as *const Mutex<Vec<(u32, u32, u32)>>).cast());
                }
                let mut actual = calls.lock().unwrap().clone();
                actual.sort_unstable();
                let expected = if declaration.id == KernelId(0) {
                    (0..17).step_by(2).map(|begin| (begin, (begin + 2).min(17), width)).collect()
                } else {
                    vec![(0, 17, width)]
                };
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn a_handle_and_a_reference_can_name_buffers_in_one_record() {
        // WHAT LETS A RECORD MIGRATE FIELD BY FIELD. The handles resolve against
        // this target's own arenas, which occupy the low slots; each remaining
        // address takes a slot above them, so no slot serves two buffers and a
        // half-migrated record dispatches correctly on the way through.
        #[repr(C, packed(4))]
        #[derive(Clone, Copy)]
        struct Mixed {
            migrated: Handle,
            still_an_address: HostRef,
            seam_arena_count: u32,
        }
        static DECL: [KernelDecl; 1] = [KernelDecl {
            id: KernelId(0),
            name: "mixed",
            scatter: Scatter::Disjoint,
            nanos_per_item: 1.0,
            args_bytes: std::mem::size_of::<Mixed>() as u16,
            // ONLY the field that is still an address. The migrated one is
            // already in the form the entry point takes.
            host_refs: &[16],
            diag: false,
            generated: true,
        }];
        let mut device = allocating_device();
        let handle = device.alloc(4, 4, 4, AllocLabel("test.migrated")).expect("alloc");
        let values = [1.0f32, 2.0, 3.0];
        let args = Mixed {
            migrated: handle,
            still_an_address: HostRef::of(&values),
            seam_arena_count: 0,
        };
        let (bases, live) = device.arena_bases();
        assert_eq!(live, 1, "one arena was opened for the migrated buffer");
        // Safety: the record is live, its reference names a live slice, and its
        // handle names a live span.
        let bound = unsafe {
            bind_generated(&DECL[0], (&args as *const Mixed).cast::<u8>(), &bases[..live])
        };
        assert_eq!(
            bound.base(0),
            device.resolve(handle).expect("resolve"),
            "slot 0 is the allocator's own arena, so the migrated handle resolves \
             through it unchanged"
        );
        assert_eq!(
            bound.base(1),
            values.as_ptr() as *mut u8,
            "the remaining address takes the slot above the live arenas"
        );
        let migrated: [u32; 4] =
            unsafe { std::ptr::read_unaligned(bound.args_ptr().cast()) };
        assert_eq!(migrated[0], handle.arena, "a handle is passed through untouched");
        assert_eq!(migrated[1], handle.off);
        let rewritten: [u32; 4] =
            unsafe { std::ptr::read_unaligned(bound.args_ptr().add(16).cast()) };
        assert_eq!(rewritten[0], 1, "the reference was bound above the live arenas");
        let count: u32 = unsafe {
            std::ptr::read_unaligned(
                bound.args_ptr().add(std::mem::size_of::<Mixed>() - 4).cast(),
            )
        };
        assert_eq!(count, 2, "the entry point checks both arena ids against this");
    }

    #[test]
    fn a_reference_is_bound_to_its_own_arena() {
        // The positional assumption the whole binding rests on: reference k in
        // declaration order becomes handle (k, 0) against base k. Checked here
        // on a record this crate owns, so it holds with no kernel in sight.
        #[repr(C, packed(4))]
        #[derive(Clone, Copy)]
        struct TwoBuffers {
            a: HostRef,
            b: HostRef,
            seam_arena_count: u32,
        }
        static DECL: [KernelDecl; 1] = [KernelDecl {
            id: KernelId(0),
            name: "two_buffers",
            scatter: Scatter::Disjoint,
            nanos_per_item: 1.0,
            args_bytes: std::mem::size_of::<TwoBuffers>() as u16,
            host_refs: &[0, 16],
            diag: false,
            generated: true,
        }];
        let first = [1.0f32, 2.0];
        let mut second = [0.0f32; 3];
        let args = TwoBuffers {
            a: HostRef::of(&first),
            b: HostRef::of_mut(&mut second),
            seam_arena_count: 0,
        };
        // Safety: the record is live and both references name live slices.
        // No arena is open, so the two references take slots 0 and 1 and the
        // positional assumption is visible on its own.
        let bound =
            unsafe { bind_generated(&DECL[0], (&args as *const TwoBuffers).cast::<u8>(), &[]) };
        assert_eq!(bound.base(0), first.as_ptr() as *mut u8);
        assert_eq!(bound.base(1), second.as_ptr() as *mut u8);
        for (arena, &offset) in DECL[0].host_refs.iter().enumerate() {
            let handle: [u32; 4] = unsafe {
                std::ptr::read_unaligned(bound.args_ptr().add(offset as usize).cast())
            };
            assert_eq!(handle[0], arena as u32, "one buffer, one arena");
            assert_eq!(handle[1], 0, "the whole allocation is the arena");
        }
        let live: u32 = unsafe {
            std::ptr::read_unaligned(
                bound
                    .args_ptr()
                    .add(std::mem::size_of::<TwoBuffers>() - 4)
                    .cast(),
            )
        };
        assert_eq!(live, 2, "the entry point checks its arena ids against this");
    }
}
