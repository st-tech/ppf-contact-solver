// File: crates/ppf-cts-compute/src/device.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The target surface, and the whole of it.
//!
//! A target owns device memory allocation, host-to-device and device-to-host
//! transfer, free, and kernel launch, plus the platform machinery those four
//! require. It owns nothing else: no convergence test, no phase ordering, no
//! fallback, no parameter interpretation and no decision about whether a term is
//! assembled. Everything in that second list lives above this seam, in the
//! caller, written once. That split is the rule the seam exists to enforce,
//! and this file is its Rust spelling; the same surface takes a C spelling
//! when a target is a separately linked library.
//!
//! Three measurements are load-bearing in that design and are stated here,
//! because a later reader will otherwise take them for taste:
//!
//! - **Execution is immediate.** The CUDA implementation this seam was
//!   measured against ends every non-queue dispatch with
//!   `cudaStreamSynchronize(0)`, and 122 of its 128 macro dispatch sites
//!   reachable from `advance` take that form. So an immediate primitive costs
//!   a CUDA target nothing across essentially the whole solver.
//! - **Exactly one construct defers**, [`Device::record`] plus
//!   [`Device::replay`], because exactly one loop needs it: CUDA's PCG issues
//!   five launches per iteration captured into one graph replay and reads the
//!   residual every fourth iteration, which is 0.25 host round trips per
//!   iteration. A trait forcing a host sync per dispatch would destroy that.
//! - **A backend that cannot record still works by re-issuing**, and the
//!   fallback is counted rather than logged. CUDA already falls back from a
//!   failed graph capture to direct launches with bit-identical numbers and only
//!   a slowdown, and a pure slowdown is inside this project's measured 44
//!   percent run-to-run envelope, so no value gate can see it.
//!   [`Counters::regions_fallback`] is the only thing that can, which is why it
//!   is on the seam rather than in a backend's private statistics.
//!
//! # What is NOT here yet, and why each is named rather than stubbed
//!
//! - **A COOPERATIVE BODY'S SERIAL TWIN.** [`Extent::Groups`] is here now and
//!   reaches all three backends, so a dispatch can name a group; the C ABI's
//!   `EXTENT_GROUPS` and the generator's `[[seam::group]]` carry the shape and
//!   all three targets render it. On the host the lanes run one after another,
//!   one whole group per step, which is exact for a body that does not
//!   synchronize and refused by the compiler for one that does, because the
//!   host seam leaves `compute::threadgroup_barrier` undefined.
//!   WHAT IS STILL MISSING IS THE TWIN SELECTION: a body that folds a group
//!   through a barrier has no host rendering, and the one exception to writing
//!   a kernel once admits a SERIAL TWIN for it, for lane cooperation only,
//!   that the generator would emit for `cpp` and `rust`. Until the generator
//!   selects between the two, a reduction reaches the CPU backend only as a
//!   body written without a barrier.
//! - **`Encoder::fill`**, the ordered device memset. Every clear on the host
//!   target today is a Rust `slice.fill`, because the buffer being cleared is
//!   caller-owned rather than a [`Handle`]. It lands with the handle migration.
//! - **The trace oracle** (`TraceSink`, [`AllocLabel`] masking). That is the
//!   cross-target acceptance instrument, and it is worth nothing with one target
//!   behind the seam.
//!
//! # What is deliberately NOT here, and never will be: a query surface
//!
//! A caller may still name a compiled HELPER directly, and that is a decision
//! rather than a backlog. A helper takes no thread range, so it is a layout
//! query, a compile-time constant, a per-pair predicate or a per-block
//! operation, and a `Device::query(QueryId, ..)` was weighed for them and
//! refused.
//!
//! A method on this trait is a value a TARGET ANSWERS. A helper is a call into a
//! neutral body that every compiler builds from identical bytes, and its whole
//! guarantee is that no target answers it differently. Behind a `query` that
//! guarantee becomes a convention each implementation must independently honor,
//! and the value returned would be the ANSWER rather than the MEANS, which rule
//! (1b) keeps off a target surface outright. A dispatch escapes that objection
//! because a target's only contribution to one is how the range is cut.
//!
//! The boundary is therefore drawn by the property that defines a dispatch, and
//! it is checked rather than described: rule 10 of
//! `.github/workflows/scripts/check-shared-wiring.py` fails the build on a
//! caller's module binding a LAUNCHER, on a binding no translation unit defines,
//! on a recorded exception that stops occurring, and on a binding kept with no
//! reason at its declaration.

// The trait is the whole target surface, and a caller reaches only the part its
// converted phases need. An unused method here is a method a target still owes
// rather than dead code, and each one is named in the module comment above.
#![allow(dead_code)]

use std::fmt;

// ===========================================================================
// THE KERNEL TABLE'S VOCABULARY
//
// THE TABLE ITSELF IS THE CALLER'S, and that follows from what this crate is
// rather than from a choice of where to put a file: a row names a kernel, so a
// crate holding the rows
// could not be published on its own and used by a program that is not a physics
// solver. What is here is the SHAPE of a row, which says only how a range may
// be cut and how wide the arguments are. A caller renders its rows from the
// same `[[seam::entry]]` declaration its entry points come from, so nothing in a
// row may acquire a branch, a phase order or a physical quantity.
// ===========================================================================

/// A dense index into the kernel table.
///
/// A kernel must have a NAME the driver can pass. That is what the generated
/// entry-point layer buys and what CUDA's 132 anonymous extended `__device__`
/// lambdas do not have today.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct KernelId(pub u16);

/// How a kernel writes its outputs, which decides whether a backend may split
/// its range across threads AT ALL.
///
/// This is the single most load-bearing field in the table.
/// `compute::atomic_add` on the host seam
/// (`crates/ppf-cts-solver/src/kernels/seam/seam_host.h`) is a plain read, add
/// and write back, so a parallel scatter over elements sharing a
/// vertex or a CSR slot is a DATA RACE and not a different fold order. Before
/// the seam that rule lived as a comment at each call site ("SERIAL, BY
/// CONTRACT"); declaring it here makes it a property the target reads rather
/// than a property a future edit can forget.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scatter {
    /// Every output element is written by exactly one thread, so the range may
    /// be cut anywhere. The cut is by a FIXED chunk size rather than by the
    /// thread count, which is what makes the answer independent of how many
    /// threads run it.
    Disjoint,
    /// Threads accumulate into shared slots. One serial ascending pass.
    Atomic,
    /// Threads take numbered slots out of a shared counter. One serial ascending
    /// pass, because a slot assignment is deterministic only in ascending order.
    Claim,
}

/// One kernel's declaration.
///
/// `Bound` (bandwidth, compute, latency) is deliberately ABSENT. It belongs here
/// the moment a target acts on it, and [`crate::sched::threads_for`] is the
/// function that would; no target caps its pool per phase today, and a field no
/// implementation reads is worse than a missing one.
#[derive(Clone, Copy, Debug)]
pub struct KernelDecl {
    pub id: KernelId,
    /// The entry point's name. Identical across renderings once the entry
    /// points are generated, which is what lets a trace be compared.
    pub name: &'static str,
    pub scatter: Scatter,
    /// Estimated nanoseconds per dispatch unit, the input to
    /// [`crate::sched::chunk_for`]: one element for [`Extent::Elements`],
    /// one whole group for [`Extent::Groups`], not one lane or logical row.
    ///
    /// A group estimate assumes the entry's intended launch width. The scheduler
    /// does not multiply it by the lane count. Call sites share this estimate
    /// rather than maintaining separate copies that can drift.
    pub nanos_per_item: f64,
    /// `size_of` the argument record.
    pub args_bytes: u16,
    /// Byte offsets of the [`HostRef`] fields inside the argument record.
    ///
    /// Read by the backend, not decoration: a `HostRef` whose address and length
    /// disagree about whether it is empty is a half-wired record, which is the
    /// defect class a hand-written mirror produces and the compiler cannot see.
    pub host_refs: &'static [u16],
    /// Whether this kernel takes a diagnostic record, so the backend supplies
    /// one per chunk and merges them.
    pub diag: bool,
    /// Whether the entry point was rendered from an `[[seam::args]]
    /// [[seam::entry]]` declaration by the caller's transcompiler.
    ///
    /// It is a property of the ENTRY POINT rather than of the target, which is why
    /// a target may read it: a generated entry takes the seam's reference form
    /// everywhere, and a hand-written one takes whatever its author wrote, so
    /// only the first can have its buffer references bound. Counting the rows
    /// that carry it measures how far a caller's conversion has got.
    pub generated: bool,
}

/// A buffer the DRIVER owns, reaching a kernel by address.
///
/// **This type is the migration debt, and it is spelled as its own type so the
/// debt is countable.** The seam's real buffer reference is [`Handle`], an
/// (arena, offset) pair, because Metal enforces 31 buffer binding slots against
/// this solver's 90 transitive device pointers and a raw device address has no
/// bound and no provenance on a backend that never faults. A `HostRef` cannot
/// be implemented by CUDA or by Metal at all.
///
/// It exists because the driver's buffers are Rust-owned `Vec`s and the host's
/// own `DataSet` arrays rather than device allocations, and migrating 26,000
/// lines of driver to arena handles is its own stage. Every record field that is
/// a `HostRef` is one field that must become a `Handle` before the phase it
/// belongs to can dispatch on a GPU backend, and `git grep -c HostRef` is the
/// count. The layout is chosen so that migration moves nothing: a `HostRef` is
/// 16 bytes, exactly a `Handle`, so swapping the type in a record leaves every
/// other field's offset where it was.
///
/// **`packed(4)` is load-bearing, not a size optimization.** A generated
/// argument record has NO PADDING by construction: every field is 4-byte
/// aligned and 4 or 16 bytes wide, so the record's size is the sum of its
/// fields and the generated size assertion is total. A `HostRef` holding two
/// `u64`s at natural alignment would be 8-byte aligned, which pads a record
/// ending in scalars out to the next multiple of 8 and breaks exactly that
/// property. The seam's reference is four `u32`s, so 4 is the alignment a
/// buffer reference is defined to have; a `HostRef` standing in for one has to
/// occupy the same shape and not merely the same number of bytes.
#[repr(C, packed(4))]
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct HostRef {
    addr: u64,
    bytes: u64,
}
const _: () = assert!(std::mem::size_of::<HostRef>() == 16);
const _: () = assert!(std::mem::size_of::<HostRef>() == std::mem::size_of::<Handle>());
const _: () = assert!(std::mem::align_of::<HostRef>() == std::mem::align_of::<Handle>());

impl HostRef {
    /// Names nothing. Distinct from a zero-length reference to a real buffer,
    /// which keeps its address.
    pub const fn none() -> Self {
        HostRef { addr: 0, bytes: 0 }
    }

    pub fn of<T>(slice: &[T]) -> Self {
        HostRef {
            addr: slice.as_ptr() as u64,
            bytes: std::mem::size_of_val(slice) as u64,
        }
    }

    pub fn of_mut<T>(slice: &mut [T]) -> Self {
        HostRef {
            addr: slice.as_mut_ptr() as u64,
            bytes: std::mem::size_of_val(slice) as u64,
        }
    }

    /// From a raw pointer and an element count, for the host's own `DataSet`
    /// arrays, which arrive as pointers rather than as slices.
    ///
    /// # Safety
    /// The pointer must address `count` elements of `T` that stay alive and
    /// unmoved until the dispatch has executed. A null pointer must come with a
    /// zero count, which is how a scene with no such array is spelled.
    pub unsafe fn at<T>(ptr: *const T, count: usize) -> Self {
        HostRef {
            addr: ptr as u64,
            bytes: (std::mem::size_of::<T>() * count) as u64,
        }
    }

    /// As [`HostRef::at`], for a buffer the kernel writes.
    ///
    /// # Safety
    /// As [`HostRef::at`], and no other reference to the same elements may be
    /// live across the dispatch.
    pub unsafe fn at_mut<T>(ptr: *mut T, count: usize) -> Self {
        HostRef {
            addr: ptr as u64,
            bytes: (std::mem::size_of::<T>() * count) as u64,
        }
    }

    pub fn addr(self) -> u64 {
        self.addr
    }

    /// The length in BYTES, which is what the reference was built from: a
    /// generated record names an allocation and not an element count, and the
    /// element width is the entry point's business.
    pub fn bytes(self) -> u64 {
        self.bytes
    }

    pub fn is_none(self) -> bool {
        self.addr == 0
    }

    /// Whether the two halves agree about whether this names anything.
    ///
    /// ONE DIRECTION ONLY, and the asymmetry is the same one the arena model
    /// has. A reference with a length and no address is a half-wired record: a
    /// count was filled in and the pointer beside it was not, which is what a
    /// hand-filled record produces and what no compiler sees. The reverse is
    /// legitimate and common: an EMPTY buffer keeps a real address, exactly as a
    /// zero-length [`Handle`] names a real bound arena rather than the `NONE`
    /// sentinel, so that a bounds check on it fails loudly instead of indexing
    /// something unbound. An empty `Vec` in Rust is that case, and several
    /// scenes carry one (an operator with no transposed couplings, a scene with
    /// no pins).
    pub fn is_coherent(self) -> bool {
        self.addr != 0 || self.bytes == 0
    }
}

/// (arena, offset, size, allocated), byte-identical to `::ArenaHandle`
/// (`crates/ppf-cts-solver/src/kernels/arena_handle.hpp`) and to `BeHandle`
/// (`crates/ppf-cts-solver/src/kernels/seam/backend_abi.h`), which CUDA and
/// Metal already share.
///
/// **`off` IS A BYTE OFFSET AND `size` AND `allocated` ARE ELEMENT COUNTS.**
/// The three units are not uniform and cannot be made so: an offset has to be
/// bytes because a generated entry point resolves `base[arena] + off` without
/// knowing any element type, and a length has to be elements because that is
/// what a caller asked for and what a kernel indexes with. What follows, and it
/// is the whole reason this paragraph exists: **`allocated` IS NOT A BYTE
/// BOUND**, so a transfer's window must be checked against the byte capacity
/// the ALLOCATOR recorded, never against a field of the handle. Reading it as
/// bytes refuses every window past the element count and admits none past the
/// real end, which is a refusal that looks like a bounds check and is not one.
/// Every target states the same rule at its own window helper, because the
/// driver fills one record for all of them.
///
/// `allocated` is not redundant with `size`: a caller that wants spare capacity
/// allocates the capacity and lowers `size` itself, which is what [`Buffer`]
/// does across a `grow`.
// NO `Default`. A value-initialized handle is all zeros, which names ARENA 0 AT
// OFFSET 0: a real arena, and the very address the first allocation lands at, so
// a transfer through one would be served against somebody else's block rather
// than refused. The sentinel is `Handle::NONE` and it has to be asked for by
// name. The C ABI header states the same reasoning at `HANDLE_NONE_ARENA`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Handle {
    pub arena: u32,
    pub off: u32,
    pub size: u32,
    pub allocated: u32,
}
const _: () = assert!(std::mem::size_of::<Handle>() == 16);

/// The least alignment any allocation may ask for.
///
/// Metal silently FLOORS a misaligned buffer offset to a multiple of 4 and
/// returns the wrong data with status Completed and no diagnostic, so four is
/// the smallest alignment at which an offset means what it says on every
/// target. A request below it is refused rather than raised, because raising it
/// would answer a question the caller got wrong instead of asking it.
pub const MIN_ALIGN: usize = 4;

impl Handle {
    /// Names nothing. Distinct from a zero-LENGTH handle, which names a real
    /// bound arena so a bounds check on it fails loudly rather than indexing an
    /// unbound slot.
    pub const NONE: Handle = Handle {
        arena: u32::MAX,
        off: 0,
        size: 0,
        allocated: 0,
    };

    pub fn is_none(self) -> bool {
        self.arena == u32::MAX
    }
}

/// Plain old data: what a transfer may move byte for byte.
///
/// # Safety
/// Every bit pattern of `size_of::<T>()` bytes is a valid `T`, `T` holds no
/// padding, and `T` owns nothing. A transfer copies the bytes and asks nothing
/// else of the type, so a `T` holding a pointer, a length or a niche would be
/// reconstituted from whatever the device left there.
pub unsafe trait Pod: Copy + 'static {}

// The scalar widths a device buffer is made of. A caller's own `#[repr(C)]`
// record implements this beside its declaration, where the layout is stated.
unsafe impl Pod for u8 {}
unsafe impl Pod for i8 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for i16 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for i32 {}
unsafe impl Pod for u64 {}
unsafe impl Pod for i64 {}
unsafe impl Pod for f32 {}

/// An allocation the caller owns, with the element type it was sized for.
///
/// **THIS IS WHAT REPLACES A DRIVER-OWNED `Vec`, and the difference that
/// matters is not ownership but PROVENANCE.** A `Vec`'s bytes are the host
/// process's own, so a record naming one carries an address, which is what
/// [`HostRef`] spells and what no device backend can resolve: a backend library
/// resolves an (arena, offset) handle against a base table it maintains, and it
/// has no way to learn what a host address means. A `Buffer`'s bytes come from
/// [`Device::alloc`], so what a record carries is that handle.
///
/// It is deliberately NOT a smart pointer. There is no `Deref`, no indexing and
/// no iteration, because a target whose device memory the host cannot address
/// is one this type must serve, and an accessor here would compile against a
/// target whose arenas happen to be host memory and be a wild read on one whose
/// are not. Reading is [`Buffer::read`] and writing is [`Buffer::write`], both
/// explicit, both a transfer, and both stay a COPY on every target: their other
/// end is caller-owned memory, which no view of an allocation reaches. Where a
/// target serves a host view of the allocation ITSELF that is
/// [`Device::host_view`], and what consumes it is [`StagedBuffer`] and
/// [`ReadbackBuffer`], whose host side is an array of their own that such a
/// view replaces.
///
/// Freeing is EXPLICIT, through [`Buffer::free`], and `Drop` deliberately does
/// nothing: a drop cannot reach the device that owns the span, and a `Drop`
/// that silently leaked would be indistinguishable from one that worked. A
/// buffer sized once at `initialize()` and reused for the run's life is the
/// shape this is for.
pub struct Buffer<T> {
    handle: Handle,
    len: usize,
    marker: std::marker::PhantomData<T>,
}

impl<T: Pod> Default for Buffer<T> {
    fn default() -> Self {
        Buffer::none()
    }
}

impl<T> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Buffer {{ handle: {:?}, len: {} }}", self.handle, self.len)
    }
}

/// What a sizing did to the allocation, which is the one fact
/// [`Buffer::size`] knows and the two shapes above it could not ask for.
///
/// **IT IS NOT A DETAIL OF THE ALLOCATOR, IT IS WHAT THE BYTES HOLD.**
/// [`Buffer::size`] zeroes an allocation only when the bytes are NEW, so the
/// two answers are the difference between "the allocation reads
/// `T::default()`" and "the allocation reads whatever it last held". A caller
/// that keeps a host side beside the allocation has to say the same thing
/// about both halves, and it cannot without being told which of the two
/// happened.
///
/// WHY THE VERDICT IS NOT ON [`Buffer::size`]'s OWN RETURN. Every one of its
/// several hundred call sites states `-> Result<(), Fault>`, several of them
/// by chaining `and_then(|()| ..)`, and a widened return would rewrite files
/// that have nothing to do with this. [`Buffer::size_reporting`] carries it
/// and [`Buffer::size`] is the discarding wrapper.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sizing {
    /// The bytes are NEW, either the first allocation or a growth past the
    /// capacity, and the call zeroed them.
    Fresh,
    /// The allocation was already long enough, so nothing was allocated,
    /// nothing moved, and it still holds whatever it last held.
    Reused,
}

impl<T: Pod> Buffer<T> {
    /// Names nothing, and holds no allocation.
    pub const fn none() -> Self {
        Buffer {
            handle: Handle::NONE,
            len: 0,
            marker: std::marker::PhantomData,
        }
    }

    /// The handle a record's buffer field takes.
    pub fn handle(&self) -> Handle {
        self.handle
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// A handle naming `count` elements starting at element `first`.
    ///
    /// This is what a `&buffer[first..first + count]` argument becomes. The
    /// offset is in the handle rather than in the entry point, because a
    /// generated entry resolves `base[arena] + off` and knows nothing about the
    /// caller's slicing.
    ///
    /// # Panics
    /// If the span runs past the allocation, which is a caller defect a device
    /// that never faults would turn into a plausible wrong answer.
    pub fn span(&self, first: usize, count: usize) -> Handle {
        let width = std::mem::size_of::<T>();
        assert!(
            first + count <= self.len,
            "a span of {count} elements at {first} runs past a buffer of {}",
            self.len
        );
        Handle {
            arena: self.handle.arena,
            off: self.handle.off + (first * width) as u32,
            size: count as u32,
            // ELEMENTS, as everywhere else. `off` is the only byte quantity a
            // handle carries.
            allocated: count as u32,
        }
    }

    /// Size the buffer to `count` elements, zeroed, allocating or growing.
    ///
    /// ZEROED, because the buffer it replaces was a `Vec` a caller sized with a
    /// zero fill, so a pass that reads further than its producer writes must go
    /// on reading a zero. It is not a substitute for a per-step clear: fresh
    /// device memory is undefined on every backend, and a target that happened
    /// to hand back zeros is what hides a missing one.
    ///
    /// AND ZEROED ONLY WHERE THE BYTES ARE NEW, which is the qualifier
    /// [`Sizing`] exists to carry: see [`Buffer::size_reporting`], which is
    /// this call with its verdict kept rather than dropped.
    pub fn size(
        &mut self,
        device: &mut impl Device,
        count: usize,
        label: AllocLabel,
    ) -> Result<(), Fault> {
        self.size_reporting(device, count, label)?;
        Ok(())
    }

    /// [`Buffer::size`], answering which of the two things it did.
    ///
    /// THE IMPLEMENTATION IS HERE AND THE WRAPPER IS ABOVE, rather than the
    /// other way round, because the verdict is a fact this function computes
    /// on the way past and every caller that drops it is dropping something
    /// real. [`StagedBuffer`] and [`ReadbackBuffer`] are the callers that
    /// cannot drop it: they carry a host side beside the allocation, and
    /// whether that host side may keep what it holds is exactly this answer.
    ///
    /// `pub(crate)` because nothing outside this crate holds a second half of
    /// a buffer. A caller with only the allocation has nothing to reconcile.
    pub(crate) fn size_reporting(
        &mut self,
        device: &mut impl Device,
        count: usize,
        label: AllocLabel,
    ) -> Result<Sizing, Fault> {
        let width = std::mem::size_of::<T>();
        // FLOORED AT [`MIN_ALIGN`], which is a request the seam sanctions rather
        // than a rounding: a one-byte element needs at most its own alignment,
        // so any offset inside a block that starts aligned satisfies it, and
        // asking for less is refused by every target. Natural alignment is what
        // decides the value for every wider type.
        let align = std::mem::align_of::<T>().max(MIN_ALIGN);
        // ZEROED WHEN THE BYTES ARE NEW, AND NOT OTHERWISE. This zeroed on
        // EVERY call, including the no-op where the buffer is already long
        // enough, and `allocate` re-sizes every buffer once a step so a Newton
        // iteration never allocates. That was 9,742 device memsets moving
        // 1,691 MB on a three-frame `drape`, against the reference's 105
        // moving 4 KB.
        //
        // A CALLER THAT WANTS A CLEARED BUFFER MUST CLEAR IT, and the ones that
        // do already did: `intersection.rs` keeps its three explicit clears and
        // its comment says why, having anticipated this exact change. Metal
        // hands out an allocation without zeroing and never faults on an
        // uninitialised read, so a reliance would be silent, which is why this
        // was not changed on the argument that nothing looked dependent.
        //
        // MEASURED INSTEAD `[2026-09-01]`: the no-op path was made to write
        // 0xCD rather than skip, so a reliance failed loudly. Under that,
        // CUDA ran 20 of 20 boundable scenes and Metal 27 of 27 fixtures,
        // excluding `allow_intersection`, whose own rate is unchanged by it
        // (one pass in four, poisoned and not).
        let fresh = self.handle.is_none() || count > self.handle.allocated as usize;
        // HEADROOM, BECAUSE A GROWTH COPIES THE WHOLE BUFFER. The arena's grow
        // allocates a fresh span and copies the old contents device-to-device
        // (`cuda/arena/arena.cu:356`), so asking for EXACTLY the count makes a
        // buffer that gains an element a step reallocate and copy every step.
        // Measured that way on `bench_drape` at 40 frames: 5,615 grow-copies
        // moving 45.4 GB and 3,806 fresh zeroes moving 44.6 GB, about 140
        // growths a frame, against a reference that moves 14 MB and 0.125 MB.
        // The rule that breaks is that a buffer a per-step routine needs is
        // allocated once and reused.
        //
        // A QUARTER, NOT A DOUBLING. Geometric growth by any factor amortizes
        // the copies to O(log n); the factor decides how much memory is held
        // and never used. A quarter bounds that waste at 25 percent, which the
        // `large-*` scenes can afford where a doubling might not, and still
        // turns a buffer that grows by ones into a growth every quarter of its
        // length rather than every step.
        let want = count + count / 4;
        if self.handle.is_none() {
            self.handle = device.alloc(want, width, align, label)?;
        } else if count > self.handle.allocated as usize {
            // AGAINST THE CAPACITY, NOT THE LENGTH. A `grow` refuses to shrink,
            // so a buffer sized down and then part way back up would be asking
            // to shrink a capacity it still holds, and the refusal would be
            // correct. What it needs in that case is nothing at all: the bytes
            // are already there, and only `len` moves.
            device.grow(&mut self.handle, want, width, align)?;
        }
        // THE HANDLE'S SIZE IS THE LOGICAL LENGTH, NEVER THE CAPACITY. A
        // `Handle` carries both, `size` in elements is what a caller's bound
        // check reads and what a kernel's extent is taken from, and
        // `allocated` is what the arena holds. Letting the headroom reach
        // `size` makes `offset.size == rows + 1` false at
        // `Operator::encode_apply`, which is where the two solver tests caught
        // it, and would have widened every `[[seam::bound]]` derived from a
        // handle by a quarter.
        self.handle.size = count as u32;
        self.len = count;
        if fresh {
            // THE COUNT, NOT THE CAPACITY. Zeroing the headroom would hide a
            // read past the caller's own length behind a zero rather than
            // whatever the arena last held there, and Metal never faults on an
            // uninitialised read, so that reliance would be silent.
            device.fill_zero(self.handle, count * width)?;
            return Ok(Sizing::Fresh);
        }
        Ok(Sizing::Reused)
    }

    /// Release the allocation. Idempotent on a buffer that names nothing.
    pub fn free(&mut self, device: &mut impl Device) -> Result<(), Fault> {
        device.free(&mut self.handle)?;
        self.len = 0;
        Ok(())
    }

    /// Copy `src` into the buffer starting at element `first`.
    pub fn write(
        &mut self,
        device: &mut impl Device,
        first: usize,
        src: &[T],
    ) -> Result<(), Fault> {
        let width = std::mem::size_of::<T>();
        // Safety: `T: Pod`, so the elements are plain bytes with no padding.
        let bytes = unsafe {
            std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), std::mem::size_of_val(src))
        };
        device.write(self.handle, first * width, bytes)
    }

    /// Copy the buffer's elements from `first` into `dst`.
    /// Fill this buffer from another, entirely on the device.
    ///
    /// # Panics
    /// If the two lengths differ, which is a caller defect rather than a
    /// device one and is worth trapping at the side that knows both.
    pub fn copy_from(&mut self, device: &mut impl Device, src: &Buffer<T>) -> Result<(), Fault> {
        assert_eq!(
            self.len, src.len,
            "a device-to-device copy needs both buffers the same length"
        );
        if self.len == 0 {
            return Ok(());
        }
        device.copy(
            self.handle,
            0,
            src.handle,
            0,
            self.len * std::mem::size_of::<T>(),
        )
    }

    pub fn read(
        &self,
        device: &mut impl Device,
        first: usize,
        dst: &mut [T],
    ) -> Result<(), Fault> {
        let width = std::mem::size_of::<T>();
        let len = std::mem::size_of_val(dst);
        // Safety: `T: Pod`, so any byte pattern of that width is a valid `T`.
        let bytes = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), len) };
        device.read(self.handle, first * width, bytes)
    }

    /// One element, for a host-side verdict a kernel wrote.
    pub fn read_one(&self, device: &mut impl Device, index: usize) -> Result<T, Fault>
    where
        T: Default,
    {
        let mut value = [T::default()];
        self.read(device, index, &mut value)?;
        Ok(value[0])
    }
}

/// Whether an elided upload should be checked against the device instead of
/// trusted, from `PPF_VERIFY_UPLOAD_ELISION`.
///
/// **THIS EXISTS BECAUSE THE ELISION RESTS ON A CLAIM ABOUT THE WHOLE TREE,
/// not on anything the type system enforces.** [`StagedBuffer::upload_span`]
/// skips a transfer whose bytes are already on the device, which is only true
/// while nothing but the host writes that allocation. That holds by
/// construction today, [`StagedBuffer::handle`] taking `&self` and the
/// kernel-written direction being [`ReadbackBuffer`], but a violation would be
/// SILENT: a dispatch would read what a kernel left rather than what the host
/// staged. Turning this on downloads every elided span and compares it, which
/// converts the claim into a measurement at the cost of making the elision
/// slower than the copy it removes. Run a scene under it after changing who
/// writes a staged buffer.
fn verify_upload_elision() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PPF_VERIFY_UPLOAD_ELISION").is_ok_and(|v| v != "0"))
}

/// A staged prefix's contents at the moment an upload declared it finished,
/// kept only under `PPF_VERIFY_UPLOAD_ELISION`.
///
/// **IT IS HOW THAT KNOB REACHES THE MAPPED REPRESENTATION.** `verify_elided`
/// answers the question by downloading the span an upload skipped and comparing
/// it, which where the host side IS the allocation is a comparison of memory
/// with itself and can never fail. So the same claim, that nothing but the host
/// writes a staged allocation, is checked FORWARD instead: the prefix is
/// fingerprinted when the upload returns and answered for at the next host
/// access. A difference means something wrote the allocation in between, and on
/// that representation the only candidate is a kernel.
struct UploadWitness {
    /// The prefix the upload named, which is the span the claim covers.
    count: usize,
    /// That prefix's bytes when the upload returned.
    hash: u64,
}

/// FNV-1a over a prefix's bytes, for [`UploadWitness`].
///
/// A HASH RATHER THAN A COPY, because this is answered at every `at` and every
/// `host` rather than once per transfer, and what it must not miss is THAT the
/// bytes changed rather than which of them did. `verify_elided` keeps the byte
/// position because it already holds both arrays; here holding a second array
/// would be the copy the representation exists to remove.
fn prefix_hash<T: Pod>(cells: &[T], count: usize) -> u64 {
    // THE BOUND IS CHECKED WHERE THE `unsafe` IS. A witness outlives nothing
    // that can shorten the host side, `size` and `free` both dropping it, so
    // this holds by construction; it is asserted because what it guards is a
    // raw read rather than an index.
    assert!(
        count <= cells.len(),
        "a witness of {count} elements runs past a staged buffer of {}",
        cells.len()
    );
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    if count == 0 {
        return hash;
    }
    // Safety: `T: Pod`, so the elements are plain bytes with no padding, which
    // is the same reasoning `Buffer::write` makes, and `count` is a prefix of
    // this slice.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            cells.as_ptr().cast::<u8>(),
            count * std::mem::size_of::<T>(),
        )
    };
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Whether a host view should be proved to alias the allocation, from
/// `PPF_VERIFY_HOST_VIEW`.
///
/// **THE ELISION VERIFIER CANNOT ASK THIS QUESTION.** `verify_elided` compares
/// the host side against a download; on the mapped representation that is a
/// comparison of memory with itself and can never fail, so a knob that looked
/// like coverage would be coverage of nothing. What is worth checking there is
/// the claim underneath: that the address the backend handed back addresses the
/// same bytes `read` returns. Turning this on checks that at every call that
/// has a device, in both directions, and restores what it touched.
fn verify_host_view() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PPF_VERIFY_HOST_VIEW").is_ok_and(|v| v != "0"))
}

/// A window of an allocation the host can address directly, where the target
/// serves one.
///
/// A raw pointer rather than a slice, because the length and the element type
/// belong to the buffer that owns it and a stored slice would need a lifetime
/// this struct cannot name. `len` is a COUNT of `T`, matching every other
/// length in this file.
struct HostView<T> {
    ptr: std::ptr::NonNull<T>,
    len: usize,
}

// SEND, AND DELIBERATELY NOT SYNC. The pointer names memory the backend owns,
// and the buffer holding it already carries a `Handle` naming the same bytes,
// so moving the buffer to another thread moves nothing the handle did not
// already move. `Sync` would say two threads may address the window at once,
// which the seam does not promise: a backend is `Send` and not `Sync`, so one
// thread drives one device.
//
// IT IS REQUIRED RATHER THAN CONVENIENT. The driver holds a table of staged
// allocations in a `static Mutex<CollisionWindows>`, and `Mutex<T>` is `Sync`
// only where `T` is `Send`, so a staged buffer that lost `Send` would stop that
// table compiling.
unsafe impl<T: Pod> Send for HostView<T> {}

/// Where a staged or readback buffer's host-side elements live.
///
/// TWO REPRESENTATIONS OF ONE THING, chosen by the target rather than by the
/// caller. On a target whose allocations the host cannot address, the elements
/// are an array this buffer owns and the transfer between it and the allocation
/// is a copy. On a target that serves a host view, the elements ARE the
/// allocation, the second array does not exist, and the transfer is nothing at
/// all.
///
/// THE ACCOUNTING ABOVE THIS TYPE DOES NOT CHANGE WITH THE REPRESENTATION, and
/// that is the property the whole design rests on. `clean` and `fresh` mean what
/// they meant, are moved by the same calls, and refuse the same handles, because
/// what they protect is a rule about which elements a DISPATCH may name, and
/// that rule is the same whether or not a copy is what put the bytes there. A
/// representation that relaxed the refusal would let a driver defect pass on one
/// target and trap on another, which is the acceptance rule inverted.
enum HostCells<T> {
    Owned(Vec<T>),
    Mapped(HostView<T>),
}

// NO BOUND ON `T` HERE, because two callers need these under `T: Pod` alone:
// the `const fn none()` twins, which a `static` is built from, and the `Debug`
// impls. Only [`take_cells`] needs `Default`, and it states that bound itself.
impl<T> HostCells<T> {
    const fn empty() -> Self {
        HostCells::Owned(Vec::new())
    }

    /// Take the host side out, leaving nothing behind, and reduce it to what
    /// can survive a sizing.
    ///
    /// A VIEW IS DROPPED HERE AND NEVER RETURNED, because the caller runs
    /// [`Buffer::size`] next, that call may grow the block, and a growth
    /// relocates it. Only the view's LENGTH survives, which is all the fill in
    /// [`take_cells`] needs of it: where the host side IS the allocation, the
    /// elements are still in the allocation, and what the next call has to know
    /// is how many of them were ever the caller's.
    fn carry(&mut self) -> Carried<T> {
        match std::mem::replace(self, HostCells::empty()) {
            HostCells::Owned(owned) => Carried {
                len: owned.len(),
                owned,
            },
            HostCells::Mapped(view) => Carried {
                owned: Vec::new(),
                len: view.len,
            },
        }
    }

    fn len(&self) -> usize {
        match self {
            HostCells::Owned(owned) => owned.len(),
            HostCells::Mapped(view) => view.len,
        }
    }

    fn is_mapped(&self) -> bool {
        matches!(self, HostCells::Mapped(_))
    }

    fn as_slice(&self) -> &[T] {
        match self {
            HostCells::Owned(owned) => owned,
            // Safety: the pointer came from `Device::host_view` for this
            // buffer's own live allocation and this many elements, it is
            // re-taken whenever the block could have moved, and no device work
            // is in flight at any point a caller runs.
            HostCells::Mapped(view) => unsafe {
                std::slice::from_raw_parts(view.ptr.as_ptr(), view.len)
            },
        }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            HostCells::Owned(owned) => owned,
            // Safety: as `as_slice`, and the borrow is exclusive.
            HostCells::Mapped(view) => unsafe {
                std::slice::from_raw_parts_mut(view.ptr.as_ptr(), view.len)
            },
        }
    }
}

/// What a sizing took out of a buffer's host side, and how much of it there
/// was.
///
/// TWO FIELDS BECAUSE THE TWO REPRESENTATIONS CARRY DIFFERENT THINGS. An owned
/// array travels: it is a `Vec` this struct now holds, and nothing can move
/// under it. A view does not, because the block it names may relocate, so only
/// its length comes across and the elements stay where they are, in the
/// allocation.
struct Carried<T> {
    /// The array from before, EMPTY where the previous representation was
    /// mapped and where there was no previous host side at all.
    owned: Vec<T>,
    /// How many leading elements the previous host side held, on either
    /// representation.
    len: usize,
}

/// Point the host side at `count` elements of `buffer`, taking a host view
/// where the target serves one.
///
/// CALLED AFTER [`Buffer::size`], because that call is what allocates or grows
/// the block, and a view taken before it would name the block a growth moved
/// away from. The caller takes its previous cells through [`HostCells::carry`]
/// BEFORE that call, which drops any view rather than returning one, so a view
/// of a moved block does not exist even for an instant.
///
/// **WHAT IT FILLS IS DECIDED BY `sizing`, AND THE TWO REPRESENTATIONS MUST
/// COME OUT HOLDING THE SAME BYTES.** [`Buffer::size`] zeroes the allocation
/// only when the bytes are NEW, so the two cases are different questions:
///
/// - [`Sizing::Fresh`]: nothing is carried over. The allocation was zeroed, and
///   the host side is filled with `T::default()` so a `T` whose default is not
///   all-zero bytes reads the same on both. The owned representation builds a
///   new array, which is the same fill by another spelling.
/// - [`Sizing::Reused`]: nothing was zeroed, so nothing is overwritten either.
///   The mapped host side keeps what the allocation holds because it IS the
///   allocation; the owned one keeps its array; and the ELEMENTS PAST THE
///   PREVIOUS LENGTH, which no host write ever reached, take `T::default()` on
///   both. A `Vec::resize` does that half on the copy path and the tail fill
///   does it on the mapped one.
///
/// THE FILL USED TO BE UNCONDITIONAL, and that made a re-size at an unchanged
/// count read `T::default()` on the mapped representation while the copy one
/// read the previous contents from the allocation, which is one code path with
/// one set of accounting handing two targets different simulation input. The
/// caller withdraws the accounting on [`Sizing::Reused`] rather than papering
/// over it here, so what a dispatch may name is the same on both.
/// Whether the elements past the carried length must read as `T::default()`
/// on both representations after a sizing.
///
/// THE TWO BUFFER KINDS ANSWER DIFFERENTLY, and the difference is not a
/// preference. A staged buffer's host side is what the caller writes and then
/// uploads, so the copy arm's `resize` defaults that tail and the mapped arm
/// has to write the same defaults into the allocation or the two disagree
/// before anything is uploaded. A readback buffer's host side is what a
/// download fills FROM the allocation, so defaulting the tail on the mapped
/// arm writes device bytes the copy arm never writes: after a download the
/// copy arm answers out of the arena and the mapped arm out of its own
/// defaults, which is the same fork in the other direction. Leaving the tail
/// alone is what makes both answer out of the arena.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tail {
    /// Default the elements past the carried length on both representations.
    Defaulted,
    /// Leave them as the allocation holds them, for both representations.
    AsAllocated,
}

fn take_cells<T: Pod + Default>(
    device: &mut impl Device,
    buffer: &Buffer<T>,
    count: usize,
    sizing: Sizing,
    carried: Carried<T>,
    tail: Tail,
) -> Result<HostCells<T>, Fault> {
    if count == 0 {
        return Ok(HostCells::empty());
    }
    // HOW MANY LEADING ELEMENTS SURVIVE, on either representation and by the
    // same rule, which is what makes the two answers equal.
    let kept = match sizing {
        Sizing::Fresh => 0,
        Sizing::Reused => carried.len.min(count),
    };
    let bytes = count * std::mem::size_of::<T>();
    // Safety: the handle names this buffer's live allocation and the window
    // starts at the block and runs for its logical length.
    let view = unsafe { device.host_view(buffer.handle(), 0, bytes)? };
    let Some(ptr) = view else {
        if kept == 0 {
            // `vec![T::default(); count]` RATHER THAN A CLEARED ARRAY, and the
            // spelling is load-bearing: for a `T` whose default is all-zero
            // bytes it is a zeroed ALLOCATION rather than a memset, which is
            // what a growth costs here and must go on costing. Nothing
            // is carried over in this case, so there is nothing to keep.
            return Ok(HostCells::Owned(vec![T::default(); count]));
        }
        // `truncate` THEN `resize`, because `kept` is the shorter of the two
        // lengths and a `resize` alone would extend from the OLD length,
        // carrying elements past `kept` that the mapped arm below defaults.
        let mut owned = carried.owned;
        owned.truncate(kept);
        owned.resize(count, T::default());
        return Ok(HostCells::Owned(owned));
    };
    let ptr = ptr.cast::<T>();
    assert_eq!(
        ptr.as_ptr() as usize % std::mem::align_of::<T>(),
        0,
        "a host view must be aligned for its element type: the seam allocates \
         at the element's own alignment, so a misaligned view is the backend \
         reporting the wrong address"
    );
    let mut cells = HostCells::Mapped(HostView { ptr, len: count });
    // THE TAIL ONLY, WHICH ON THE PER-STEP PATH IS NOTHING AT ALL. A buffer
    // re-sized to the length it already holds leaves `kept == count` and this
    // fills an empty slice, so a per-step sizing moves no host bytes. What it
    // does write is the elements past the previous length, which the owned arm
    // above writes through its `resize`.
    if tail == Tail::Defaulted {
        cells.as_mut_slice()[kept..].fill(T::default());
    }
    Ok(cells)
}

/// Re-resolve a held view and require the address to be unchanged.
///
/// THE CHEAP HALF OF [`verify_host_view_aliases`], split out because it is the
/// half a debug build can afford at every call: it allocates nothing and moves
/// no bytes. A view is dropped and re-taken only where the block can move, so an
/// address that has changed under one means something relocated the block
/// without the buffer being told.
fn verify_host_view_address<T: Pod + Default>(
    device: &mut impl Device,
    buffer: &Buffer<T>,
    cells: &HostCells<T>,
) {
    let len = cells.len();
    let HostCells::Mapped(view) = cells else {
        return;
    };
    if len == 0 {
        return;
    }
    let bytes = len * std::mem::size_of::<T>();
    // Safety: the handle names the live allocation and the window is its own
    // logical length.
    let again = unsafe { device.host_view(buffer.handle(), 0, bytes) }
        .expect("PPF_VERIFY_HOST_VIEW: the view could not be re-taken")
        .expect("PPF_VERIFY_HOST_VIEW: the target stopped serving a view");
    assert_eq!(
        again.as_ptr(),
        view.ptr.as_ptr().cast::<u8>(),
        "PPF_VERIFY_HOST_VIEW: the allocation's host address moved while a \
         view of it was held, so something relocated the block without this \
         buffer re-taking the view"
    );
}

/// Prove that a host view really aliases the allocation.
///
/// PANICS on a mismatch rather than returning a `Fault`, on the same grounds
/// `verify_elided` does: a difference means every access through the view was
/// already reading or writing the wrong memory, and there is no recovery to
/// offer.
fn verify_host_view_aliases<T: Pod + Default>(
    device: &mut impl Device,
    buffer: &Buffer<T>,
    cells: &mut HostCells<T>,
) {
    let len = cells.len();
    if len == 0 || !cells.is_mapped() {
        return;
    }
    // THE CURRENT ADDRESS, FIRST.
    verify_host_view_address(device, buffer, cells);

    // PASSIVELY, SECOND: the current contents must come back equal through the
    // ordinary transfer.
    let bytes = len * std::mem::size_of::<T>();
    let mut mirror = vec![T::default(); len];
    buffer
        .read(device, 0, &mut mirror)
        .expect("PPF_VERIFY_HOST_VIEW: the allocation could not be read back");
    {
        // Safety: `T: Pod`, so both are plain bytes of a known width.
        let seen =
            unsafe { std::slice::from_raw_parts(cells.as_slice().as_ptr().cast::<u8>(), bytes) };
        let back = unsafe { std::slice::from_raw_parts(mirror.as_ptr().cast::<u8>(), bytes) };
        if let Some(at) = seen.iter().zip(back).position(|(a, b)| a != b) {
            panic!(
                "PPF_VERIFY_HOST_VIEW: the host view and the allocation differ at \
                 byte {at}, so the address the backend returned does not name the \
                 bytes `read` returns"
            );
        }
    }

    // ACTIVELY, LAST, because equal contents prove nothing about an array that
    // is all one value, which a freshly sized buffer is. One element's bytes are
    // inverted through the view, read back through the seam, and restored.
    let width = std::mem::size_of::<T>();
    // Safety: as above.
    let saved: Vec<u8> =
        unsafe { std::slice::from_raw_parts(cells.as_slice().as_ptr().cast::<u8>(), width) }
            .to_vec();
    let probe: Vec<u8> = saved.iter().map(|byte| !*byte).collect();
    {
        // Safety: as above, and the borrow is exclusive.
        let cell = unsafe {
            std::slice::from_raw_parts_mut(cells.as_mut_slice().as_mut_ptr().cast::<u8>(), width)
        };
        cell.copy_from_slice(&probe);
    }
    let mut one = [T::default()];
    let read = buffer.read(device, 0, &mut one);
    {
        // RESTORED BEFORE THE STATUS IS READ, so a failed read leaves the
        // allocation holding what it held.
        // Safety: as above.
        let cell = unsafe {
            std::slice::from_raw_parts_mut(cells.as_mut_slice().as_mut_ptr().cast::<u8>(), width)
        };
        cell.copy_from_slice(&saved);
    }
    read.expect("PPF_VERIFY_HOST_VIEW: the probe could not be read back");
    // Safety: as above.
    let came_back = unsafe { std::slice::from_raw_parts(one.as_ptr().cast::<u8>(), width) };
    assert_eq!(
        came_back,
        &probe[..],
        "PPF_VERIFY_HOST_VIEW: a write through the host view did not reach the \
         allocation, so the backend handed back a copy rather than a view"
    );
}

/// Run whichever host-view check this build asks for, at a call that has a
/// device.
///
/// ONE CALL RATHER THAN TWO, so the knob and the debug build cannot both fire
/// and pay for the address check twice. A debug build gets the address
/// re-resolution, which allocates nothing; `PPF_VERIFY_HOST_VIEW` gets that plus
/// the byte comparison and the active probe.
fn maybe_verify_host_view<T: Pod + Default>(
    device: &mut impl Device,
    buffer: &Buffer<T>,
    cells: &mut HostCells<T>,
) {
    if verify_host_view() {
        verify_host_view_aliases(device, buffer, cells);
    } else if cfg!(debug_assertions) {
        verify_host_view_address(device, buffer, cells);
    }
}

/// A device allocation with a host copy the caller fills, and the upload between
/// them.
///
/// **WHY A SECOND SHAPE EXISTS BESIDE [`Buffer`].** Some of a caller's buffers
/// are produced by one kernel and consumed by the next, and those are a plain
/// `Buffer`: the host never sees the bytes. Others are BUILT ON THE HOST, an
/// element at a time, out of facts the device has no access to, and then read by
/// a kernel. On a target whose memory the host cannot address there is no way to
/// write those in place, so the shape is a host array, a device allocation, and
/// one transfer; on a target that serves a host view the same shape is one
/// array and no transfer, which is what [`Device::host_view`] chooses between.
/// Either way it is the transfer half of the seam that decides, not the caller.
///
/// **FORGETTING THE UPLOAD IS THE DEFECT THIS TYPE EXISTS TO MAKE LOUD.** A
/// stale device copy is a plausible wrong answer, not a crash: the kernel reads
/// the PREVIOUS step's parameters and the run completes. So [`StagedBuffer::at`]
/// invalidates the device copy and [`StagedBuffer::handle`] refuses to hand out
/// a handle until [`StagedBuffer::upload`] has made it current again. The check
/// is an assert rather than a `Result` because a record literal is where a
/// handle is asked for, and a violated invariant there is a caller defect with
/// no correct recovery. The refusal is the SAME on both representations: what it
/// records is whether the caller has declared itself finished writing, which a
/// target that needs no transfer still has to be told.
///
/// **HOW MUCH IS CURRENT IS A LENGTH, NOT A FLAG, because a caller that
/// compacts into a prefix uploads a prefix.** An element scatter gathers its
/// active elements into one run starting at zero and dispatches over exactly
/// that run, and the run is a small fraction of the allocation, so transferring
/// the whole array on every dispatch would move the chunk rather than the
/// answer. [`StagedBuffer::upload_span`] transfers a prefix and hands back a
/// handle naming exactly it, which is one call because it is one decision: a
/// caller stating the length twice can state it two ways, and the direction
/// that hurts is silent, since the tail the handle names but the transfer
/// missed holds whatever the device held before. A prefix upload therefore
/// leaves the buffer NOT current as a whole, and [`StagedBuffer::handle`],
/// which names the whole allocation, keeps refusing until a full
/// [`StagedBuffer::upload`] has run.
pub struct StagedBuffer<T> {
    cells: HostCells<T>,
    device: Buffer<T>,
    /// How many leading elements of the host side the device copy matches. A
    /// full upload sets it to the length; `at` sets it to zero, because the
    /// slice it hands out covers the whole array and it cannot see which
    /// elements move.
    clean: usize,
    /// What the last upload left, under `PPF_VERIFY_UPLOAD_ELISION` and never
    /// otherwise. See [`UploadWitness`]: it is how that knob reaches the mapped
    /// representation, which `verify_elided` cannot.
    witness: Option<UploadWitness>,
}

impl<T: Pod> StagedBuffer<T> {
    /// Names nothing, and holds no allocation.
    ///
    /// THE CONST TWIN OF [`StagedBuffer::default`], and it exists for one
    /// reason: a caller held in a `static` has to be constructed in a const
    /// context, which `Default` cannot serve. The driver's collision-window
    /// table is exactly that, a `static Mutex<..>` whose masks are staged
    /// allocations, so without this the table could not hold one at all.
    pub const fn none() -> Self {
        StagedBuffer {
            cells: HostCells::empty(),
            device: Buffer::none(),
            clean: 0,
            witness: None,
        }
    }
}

impl<T: Pod> Default for StagedBuffer<T> {
    fn default() -> Self {
        StagedBuffer::none()
    }
}

impl<T: Pod> fmt::Debug for StagedBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StagedBuffer {{ len: {}, clean: {}, mapped: {}, {:?} }}",
            self.cells.len(),
            self.clean,
            self.cells.is_mapped(),
            self.device
        )
    }
}

impl<T: Pod + Default> StagedBuffer<T> {
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.len() == 0
    }

    /// The host copy, read only. Reading does not dirty it.
    ///
    /// UNDER `PPF_VERIFY_UPLOAD_ELISION` THIS IS ONE OF THE TWO PLACES THE
    /// WITNESS IS ANSWERED, because a read is where a kernel's write to a
    /// staged allocation would be observed on the mapped representation.
    pub fn host(&self) -> &[T] {
        self.check_upload_witness();
        self.cells.as_slice()
    }

    /// The host copy, to fill. Every element the caller means to change must be
    /// written through this, and the next [`StagedBuffer::handle`] is refused
    /// until the upload has run.
    ///
    /// ON THE MAPPED REPRESENTATION THE ELEMENTS ARE ALREADY WHERE A DISPATCH
    /// READS THEM, AND `clean` IS ZEROED ANYWAY. What it records is not where
    /// the bytes are but whether the caller has declared it finished writing,
    /// and a dispatch over a half-filled array is the defect either way. Keeping
    /// the refusal identical is what makes a scene that passes on one target
    /// pass on the others.
    pub fn at(&mut self) -> &mut [T] {
        // ANSWERED BEFORE THE CALLER CAN WRITE, AND DROPPED AFTER. The witness
        // covers the span between an upload and the next host access, and this
        // call is the end of that span: whatever the caller writes next is the
        // host writing, which is the one writer the claim allows.
        self.check_upload_witness();
        self.witness = None;
        self.clean = 0;
        self.cells.as_mut_slice()
    }

    /// Answer the witness the last upload left, where one was left at all.
    ///
    /// NOTHING BUT AN `Option` TEST WITHOUT THE KNOB, and nothing at all on the
    /// copy representation, which is checked by `verify_elided` instead. The
    /// hash is over the uploaded prefix, so a caller reading a long array in a
    /// loop pays it once per call; that is the same trade
    /// `PPF_VERIFY_UPLOAD_ELISION` already makes, which downloads the elided
    /// span rather than skipping it.
    fn check_upload_witness(&self) {
        let Some(witness) = &self.witness else {
            return;
        };
        assert_eq!(
            prefix_hash(self.cells.as_slice(), witness.count),
            witness.hash,
            "PPF_VERIFY_UPLOAD_ELISION: an upload of {} elements carried nothing, \
             because the host side IS the allocation, and those elements have \
             changed since. Something other than the host has written this \
             allocation, so the elision in `StagedBuffer::upload_span` is unsound \
             for it",
            witness.count
        );
    }

    /// The handle a record's buffer field takes.
    ///
    /// # Panics
    /// If the host copy has been written and not uploaded, which would dispatch
    /// against the previous contents.
    pub fn handle(&self) -> Handle {
        assert!(
            self.clean == self.cells.len(),
            "a staged buffer of {} elements is current for {} of them, so this \
             dispatch would read the previous contents past that point",
            self.cells.len(),
            self.clean
        );
        self.device.handle()
    }

    /// `count` elements starting `first` elements in, as one record field.
    ///
    /// FOR A CALLER WHOSE ARRAY IS A PREFIX OF THIS ONE. The mesh topology is
    /// the case that wanted it: rods are a prefix of `mesh.edge` and shell
    /// faces a prefix of `mesh.face`, so one upload serves both the whole array
    /// and the prefix that carries an energy.
    ///
    /// # Panics
    /// Under the same condition [`StagedBuffer::handle`] does, because a span
    /// of a stale mirror is stale in exactly the same way, and past the end of
    /// the allocation, which [`Buffer::span`] checks.
    pub fn span(&self, first: usize, count: usize) -> Handle {
        assert!(
            self.clean == self.cells.len(),
            "a staged buffer of {} elements is current for {} of them, so this \
             dispatch would read the previous contents past that point",
            self.cells.len(),
            self.clean
        );
        self.device.span(first, count)
    }

    /// Size both halves to `count` elements, zeroed where the allocation is
    /// fresh and preserved where it is not.
    ///
    /// **WHY IT IS NOT SIMPLY "ZEROED".** [`Buffer::size`] zeroes an allocation
    /// only when the bytes are NEW, so at a length the buffer already holds it
    /// moves nothing and the allocation goes on holding what it last held. This
    /// call reports that through [`Sizing`] rather than hiding it: on
    /// [`Sizing::Fresh`] both halves carry `T::default()` and the whole length
    /// is current, and on [`Sizing::Reused`] both halves keep what they held
    /// and `clean` is ZERO, so the next [`StagedBuffer::handle`] is refused
    /// until an upload has made the two agree.
    ///
    /// THE REFUSAL IS THE POINT, AND IT IS THE SAME ON BOTH REPRESENTATIONS.
    /// `clean = count` after a reused sizing was a claim the copy path could not
    /// keep, the host side reading `T::default()` while the allocation read the
    /// previous contents, and a dispatch taken on that claim read the older
    /// bytes. Withdrawing it costs an upload a caller was already owed and
    /// leaves nothing for a target to answer differently.
    pub fn size(
        &mut self,
        device: &mut impl Device,
        count: usize,
        label: AllocLabel,
    ) -> Result<(), Fault> {
        // THE VIEW IS DROPPED FIRST, because `Buffer::size` may grow the block
        // and a growth relocates it. `carry` takes the host side out and drops
        // any view rather than handing one back, which is what makes a view of a
        // moved block unrepresentable rather than merely unlikely.
        let carried = self.cells.carry();
        // AND THE ACCOUNTING IS WITHDRAWN BEFORE THE TWO FALLIBLE CALLS, not
        // after them. On either `?` the host side is gone, and a `clean` left at
        // its previous value would then describe a buffer that no longer exists:
        // the two fields agree at every point, including the ones this call
        // does not reach.
        self.clean = 0;
        self.witness = None;
        let sizing = self.device.size_reporting(device, count, label)?;
        // A staged buffer's tail is the caller's to write, so both representations
        // must show the default there before an upload.
        self.cells =
            take_cells(device, &self.device, count, sizing, carried, Tail::Defaulted)?;
        if sizing == Sizing::Fresh {
            // BOTH HALVES CARRY `T::default()` after a fresh sizing, the
            // allocation through `Buffer::size` and the host side through
            // `take_cells`, so the whole length is current.
            self.clean = count;
        }
        Ok(())
    }

    /// Copy the host contents to the device.
    pub fn upload(&mut self, device: &mut impl Device) -> Result<(), Fault> {
        let count = self.cells.len();
        self.upload_span(device, count)?;
        Ok(())
    }

    /// Copy the first `count` elements to the device, and name exactly them.
    ///
    /// FOR A CALLER THAT COMPACTS INTO A PREFIX. The returned handle spans the
    /// elements this call transferred and no others, so the length cannot be
    /// stated one way for the transfer and another for the dispatch.
    ///
    /// # Panics
    /// If `count` runs past the host copy, which is a caller defect: the
    /// allocation and the host array are sized together, so a longer prefix
    /// names elements neither half has.
    pub fn upload_span(
        &mut self,
        device: &mut impl Device,
        count: usize,
    ) -> Result<Handle, Fault> {
        assert!(
            count <= self.cells.len(),
            "a prefix of {count} elements runs past a staged buffer of {}",
            self.cells.len()
        );
        // NOTHING TO CARRY WHERE THE HOST SIDE IS THE ALLOCATION. `at` wrote the
        // elements where a dispatch reads them, so the transfer the other
        // representation makes here has already happened.
        //
        // `clean` MOVES EXACTLY AS IT DOES ON THE COPY PATH, and deliberately
        // not to the full length even though every element is in fact current.
        // Setting it further would hand out a whole-array handle here that the
        // copy path refuses, so a driver that forgot a full upload would trap on
        // one target and run on another. The refusal is worth more than the
        // handle.
        if self.cells.is_mapped() {
            maybe_verify_host_view(device, &self.device, &mut self.cells);
            if self.clean < count {
                self.clean = count;
            }
            // AND `PPF_VERIFY_UPLOAD_ELISION` REACHES THIS REPRESENTATION HERE.
            // `verify_elided` below cannot: it compares the host side against a
            // download, which is memory against itself once the two are one
            // array. The claim is the same one either way, that nothing but the
            // host writes a staged allocation, so it is checked forward instead.
            // The previous witness is answered before this one replaces it, so a
            // kernel write between two uploads is caught as well as one between
            // an upload and a read.
            if verify_upload_elision() {
                self.check_upload_witness();
                self.witness = Some(UploadWitness {
                    count,
                    hash: prefix_hash(self.cells.as_slice(), count),
                });
            }
            return Ok(self.device.span(0, count));
        }

        // THE ELISION, AND `clean` IS EXACTLY THE FACT IT NEEDS. It counts the
        // leading elements the device copy already matches, and only three
        // things move it: `at` zeroes it because the slice it hands out covers
        // the whole array, `size` sets it to the length ONLY when the sizing was
        // fresh, because that is the case in which both halves are known equal,
        // and withdraws it to zero otherwise, and this one sets it to what it
        // transferred. So
        // `clean >= count` means these very elements are already there and the
        // copy would rewrite them with themselves.
        //
        // WHAT MAKES IT SAFE IS THAT NOTHING ELSE WRITES THE ALLOCATION.
        // A `StagedBuffer` is the host-written, kernel-READ direction, and
        // `handle` is `&self`, so a dispatch cannot take a mutable view of one.
        // The direction where a kernel writes is `ReadbackBuffer`, which is a
        // different type. If that ever stops holding, this returns stale
        // contents silently, which is why `PPF_VERIFY_UPLOAD_ELISION` exists.
        //
        // AND IT MUST NOT TOUCH `clean` ON THE WAY OUT. Assigning `count` here
        // would SHRINK a longer clean prefix on a caller that asked for less,
        // and the next `handle` would then refuse a mirror that is in fact
        // whole.
        if self.clean >= count {
            if verify_upload_elision() {
                self.verify_elided(device, count);
            }
            return Ok(self.device.span(0, count));
        }
        {
            // DESTRUCTURED RATHER THAN `mem::take`d. The two fields are
            // disjoint, so borrowing them separately is what the compiler needs
            // and no swap is required to satisfy it.
            let Self { cells, device: buffer, .. } = self;
            buffer.write(device, 0, &cells.as_slice()[..count])?;
        }
        // NOT `max`: a shorter prefix after a longer one leaves the tail
        // holding what the host wrote LAST time, and a caller whose run shrank
        // is exactly the case that would silently read it.
        self.clean = count;
        Ok(self.device.span(0, count))
    }

    /// Download the span an upload just skipped and compare it byte for byte.
    ///
    /// PANICS on a mismatch rather than returning a `Fault`, because a
    /// difference here means the elision's premise is false and every earlier
    /// skip in the run was already wrong. There is no recovery to offer and no
    /// caller that could act on one.
    ///
    /// IT ASKS NOTHING ON THE MAPPED REPRESENTATION AND IS NOT REACHED THERE.
    /// There is no elision to check when there is no copy, and comparing the
    /// host side against a download of the same bytes could not fail. What is
    /// worth checking on that representation is the claim underneath it, which
    /// is `PPF_VERIFY_HOST_VIEW`.
    fn verify_elided(&self, device: &mut impl Device, count: usize) {
        if count == 0 {
            return;
        }
        let mut mirror = vec![T::default(); count];
        self.device
            .read(device, 0, &mut mirror)
            .expect("PPF_VERIFY_UPLOAD_ELISION: the device copy could not be read back");
        // Safety: `T: Pod`, so both slices are plain bytes of a known width,
        // which is the same reasoning `Buffer::read` makes just above.
        let staged = unsafe {
            std::slice::from_raw_parts(
                self.cells.as_slice().as_ptr().cast::<u8>(),
                count * std::mem::size_of::<T>(),
            )
        };
        let actual = unsafe {
            std::slice::from_raw_parts(
                mirror.as_ptr().cast::<u8>(),
                count * std::mem::size_of::<T>(),
            )
        };
        if let Some(at) = staged.iter().zip(actual).position(|(a, b)| a != b) {
            panic!(
                "PPF_VERIFY_UPLOAD_ELISION: an upload of {count} elements was skipped as \
                 redundant, but the device copy differs from the host copy at byte {at}. \
                 Something other than the host has written this allocation, so the elision \
                 in `StagedBuffer::upload_span` is unsound for it"
            );
        }
    }

    /// Release the allocation and drop any view of it.
    ///
    /// `pub(crate)` because nothing outside this crate frees a staged buffer:
    /// the driver sizes once and reuses. It exists so the conformance
    /// assertions can release what they allocate WITHOUT leaving a view of a
    /// freed block behind, which reaching through the private `device` field
    /// would do.
    pub(crate) fn free(&mut self, device: &mut impl Device) -> Result<(), Fault> {
        self.cells = HostCells::empty();
        self.clean = 0;
        self.witness = None;
        self.device.free(device)
    }

    /// Whether the host side is the allocation itself.
    ///
    /// For the conformance assertions, which have to state a different BYTE
    /// expectation per representation while stating the same ACCOUNTING
    /// expectation on both.
    pub(crate) fn is_mapped(&self) -> bool {
        self.cells.is_mapped()
    }
}

/// A device allocation with a host mirror a kernel fills, and the download
/// between them.
///
/// **THE OPPOSITE DIRECTION TO [`StagedBuffer`], AND THE ONE WHOSE MISTAKE IS
/// SILENT.** A staged buffer is built on the host and read by a kernel; this one
/// is written by a kernel and read by the host, which is what a per-element
/// VERDICT is: a face whose singular values put it over its strain limit, a
/// rest shape the creep rewrote. The host cannot compute either, because the
/// value falls out of work the kernel did, and neither can be left on the
/// device, because what consumes it is a host branch.
///
/// **THAT FIRST CONDITION IS THE ONE TO CHECK, AND A VERDICT CAN FAIL IT.** A
/// per-element answer computed by a kernel is not automatically device
/// knowledge: a kernel that classifies an element by a value the HOST built,
/// a material id being the case in this tree, returns something the host could
/// have decided before the run. What that costs is not the transfer's
/// nanoseconds but a queue drain per dispatch on a backend that has a queue,
/// charged wherever the dispatch sits, which for an assembly stage is every
/// Newton iteration. Before reaching for this type, ask which of the kernel's
/// inputs the verdict is a function of; if they are all host-built, the answer
/// belongs at scene build.
///
/// **WHAT MAKES THE DIRECTION HARDER IS THAT THE WRITER IS NOT THE CALLER.** A
/// staged buffer knows when it goes stale: the host dirties it through
/// [`StagedBuffer::at`], so the type sees the write. Here the DEVICE writes, and
/// a buffer cannot observe a dispatch. So the invalidation is taken at the last
/// moment the buffer is involved before the kernel runs, which is
/// [`ReadbackBuffer::handle`]: a caller asks for a handle exactly to put it in a
/// record, and a record naming this buffer is a kernel about to write it.
/// [`ReadbackBuffer::host`] then refuses until [`ReadbackBuffer::download`] has
/// made the mirror current again.
///
/// **THAT RULE OVER-INVALIDATES, ON PURPOSE.** A kernel that only READS through
/// the handle marks the mirror stale anyway, and the caller pays a download it
/// did not need. The two errors are not symmetric: over-invalidating costs a
/// transfer, and under-invalidating returns the PREVIOUS iteration's verdict
/// from an array that still holds plausible values, which no fault catches on a
/// target that never faults. A needless transfer is the direction to err in.
///
/// **THE MIRROR IS THE WHOLE ARRAY, with no prefix form.** [`StagedBuffer`] has
/// one because an element scatter compacts its active run into a prefix and
/// transferring the tail would move the chunk rather than the answer. A verdict
/// has the opposite shape: the dispatch covers the whole element range, and the
/// host reads a SCATTERED subset of it, so there is no prefix to name. If a
/// caller ever wants less than all of it, the thing to add is the span it
/// actually reads, not a guessed one.
pub struct ReadbackBuffer<T> {
    cells: HostCells<T>,
    device: Buffer<T>,
    /// Whether the mirror matches what the device holds. A bool rather than a
    /// length, because a download is all or nothing here; see the note above on
    /// why there is no prefix form.
    fresh: bool,
}

impl<T: Pod> ReadbackBuffer<T> {
    /// Names nothing and holds no allocation, so a `static` can hold one.
    ///
    /// Mirrors [`StagedBuffer::none`] and exists for the same reason: a table
    /// living behind a `Mutex` in a `static` is built in a `const` context,
    /// and a buffer the host SEEDS and a kernel then WRITES belongs to this
    /// type rather than to the staged one.
    pub const fn none() -> Self {
        ReadbackBuffer {
            cells: HostCells::empty(),
            device: Buffer::none(),
            fresh: false,
        }
    }
}

impl<T: Pod> Default for ReadbackBuffer<T> {
    fn default() -> Self {
        ReadbackBuffer {
            cells: HostCells::empty(),
            device: Buffer::none(),
            fresh: true,
        }
    }
}

impl<T: Pod> fmt::Debug for ReadbackBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ReadbackBuffer {{ len: {}, fresh: {}, mapped: {}, {:?} }}",
            self.cells.len(),
            self.fresh,
            self.cells.is_mapped(),
            self.device
        )
    }
}

impl<T: Pod + Default> ReadbackBuffer<T> {
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.len() == 0
    }

    /// The mirror, which is what the host reads the verdict out of.
    ///
    /// # Panics
    /// If a handle has been handed out and not downloaded since, which would
    /// answer out of the contents from before that dispatch.
    pub fn host(&self) -> &[T] {
        assert!(
            self.fresh,
            "a readback buffer of {} elements is not current: it was named by a \
             dispatch and not downloaded since, or a sizing of it failed part \
             way and withdrew the claim. Either way this read would answer out \
             of contents that are not the ones asked for",
            self.cells.len()
        );
        self.cells.as_slice()
    }

    /// The handle a record's buffer field takes.
    ///
    /// TAKES `&mut self` BECAUSE ASKING IS THE EVENT: the record this handle
    /// goes into is a kernel that will write the buffer, so the mirror is stale
    /// from here until the next download.
    pub fn handle(&mut self) -> Handle {
        self.fresh = false;
        self.device.handle()
    }

    /// A handle naming `count` elements starting at element `first`.
    ///
    /// The prefix form of [`Self::handle`], and it exists for the same reason
    /// [`Buffer::span`] does: one allocation sized for the widest caller is
    /// handed to a narrower pass as `&buffer[first..first + count]`, and the
    /// bound the entry point checks should be that pass's count rather than the
    /// whole allocation. Converting such a site to the whole-buffer handle
    /// would compile and would WIDEN the bound, which is the wrong direction.
    ///
    /// STALES THE WHOLE MIRROR, not the span, because `fresh` is a bool: a
    /// download is all or nothing here, so a partial write invalidates the
    /// whole host copy rather than part of it.
    ///
    /// # Panics
    /// If the span runs past the allocation, as [`Buffer::span`] does.
    pub fn span(&mut self, first: usize, count: usize) -> Handle {
        self.fresh = false;
        self.device.span(first, count)
    }

    /// Size both halves to `count` elements, zeroed where the allocation is
    /// fresh and preserved where it is not.
    ///
    /// **THE TWIN OF [`StagedBuffer::size`], AND THE SAME QUALIFIER APPLIES.**
    /// [`Buffer::size`] zeroes an allocation only when the bytes are NEW, so at
    /// a length the buffer already holds nothing moves. On [`Sizing::Fresh`]
    /// both halves carry `T::default()` and the mirror is current; on
    /// [`Sizing::Reused`] both halves keep what they held and the mirror is
    /// STALE, so [`ReadbackBuffer::host`] refuses until a download or a seed.
    ///
    /// THAT REFUSAL IS WHY THE PRESERVED BYTES ARE NOT A HAZARD HERE. A caller
    /// that sized and read without downloading was reading the previous
    /// dispatch's verdict on the mapped representation and `T::default()` on the
    /// copy one; now it reads neither and is told which call it owes.
    pub fn size(
        &mut self,
        device: &mut impl Device,
        count: usize,
        label: AllocLabel,
    ) -> Result<(), Fault> {
        // THE VIEW IS DROPPED FIRST, because `Buffer::size` may grow the block
        // and a growth relocates it. `carry` takes the host side out and drops
        // any view rather than handing one back, which is what makes a view of a
        // moved block unrepresentable rather than merely unlikely.
        let carried = self.cells.carry();
        // AND THE FLAG IS WITHDRAWN BEFORE THE TWO FALLIBLE CALLS, not after
        // them. On either `?` the mirror is gone, and a `fresh` left true would
        // then let `host` answer out of an EMPTY slice with no assert, so a
        // caller folding over it on the error path would fold over nothing.
        self.fresh = false;
        let sizing = self.device.size_reporting(device, count, label)?;
        // A readback buffer's tail is filled by a download from the allocation, so
        // defaulting it here would write device bytes the copy arm never writes and
        // the two would answer differently after that download.
        self.cells =
            take_cells(device, &self.device, count, sizing, carried, Tail::AsAllocated)?;
        if sizing == Sizing::Fresh {
            // BOTH HALVES CARRY `T::default()` after a fresh sizing, the
            // allocation through `Buffer::size` and the mirror through
            // `take_cells`, so the mirror is current.
            self.fresh = true;
        }
        Ok(())
    }

    /// Seed BOTH halves from the host, leaving the mirror current.
    ///
    /// THE ONE HOST WRITE A READBACK BUFFER TAKES, and deliberately not the
    /// general `write` a [`Buffer`] has. A buffer the host may overwrite at any
    /// time cannot keep a freshness flag honest, because the flag would have to
    /// track two writers. A SEED is different: it establishes an initial state
    /// on both halves at once, so the mirror is current by construction rather
    /// than by a claim, and it is the only place `fresh` is set without a
    /// download.
    ///
    /// WHAT IT IS FOR: a buffer that is host-SEEDED once, kernel-written after
    /// that, and host-read. There was no type for it, and the scene's
    /// build-time positions are the case that wanted one. Note what this does
    /// NOT add: a way to keep reading the mirror after a dispatch has named the
    /// buffer. A host reader still calls [`Self::download`] first, which is
    /// what makes the readback visible at the site that pays for it.
    ///
    /// # Panics
    /// If `src` is not this buffer's length, which is a caller defect and is
    /// worth trapping at the side that knows both lengths.
    pub fn seed(&mut self, device: &mut impl Device, src: &[T]) -> Result<(), Fault>
    where
        T: Copy,
    {
        assert_eq!(
            src.len(),
            self.cells.len(),
            "a readback buffer is seeded with exactly its own length"
        );
        if src.is_empty() {
            return Ok(());
        }
        // ONE HOST WRITE WHERE THE MIRROR IS THE ALLOCATION, rather than a host
        // copy followed by a transfer of the same bytes. The seed's meaning is
        // unchanged: both halves hold `src` and `fresh` is set by construction.
        if self.cells.is_mapped() {
            self.cells.as_mut_slice().copy_from_slice(src);
            maybe_verify_host_view(device, &self.device, &mut self.cells);
        } else {
            let Self { cells, device: buffer, .. } = self;
            cells.as_mut_slice().copy_from_slice(src);
            buffer.write(device, 0, src)?;
        }
        self.fresh = true;
        Ok(())
    }

    /// Fill this buffer from another, entirely on the device.
    ///
    /// THE MIRROR GOES STALE, because the device half changed and the host half
    /// did not. That is the same rule [`Self::handle`] follows and for the same
    /// reason: a host read after this must download rather than answer out of
    /// the contents from before.
    ///
    /// # Panics
    /// If the two lengths differ.
    pub fn copy_from(&mut self, device: &mut impl Device, src: &ReadbackBuffer<T>) -> Result<(), Fault> {
        assert_eq!(
            self.cells.len(),
            src.cells.len(),
            "a device-to-device copy needs both buffers the same length"
        );
        if self.cells.len() == 0 {
            return Ok(());
        }
        // STALE ON BOTH REPRESENTATIONS. A device-to-device copy changes the
        // bytes the view addresses, and what `fresh` records is that the caller
        // has reached the point at which reading is legitimate, which this call
        // is not.
        self.fresh = false;
        self.device.copy_from(device, &src.device)?;
        if self.cells.is_mapped() {
            maybe_verify_host_view(device, &self.device, &mut self.cells);
        }
        Ok(())
    }

    /// Copy the device contents into the mirror, which makes [`Self::host`]
    /// answerable again.
    pub fn download(&mut self, device: &mut impl Device) -> Result<(), Fault> {
        // AN EMPTY MIRROR IS ALREADY CURRENT, and it has no allocation to read
        // from: `size(.., 0, ..)` reserves nothing, so resolving the handle
        // would refuse with "arena 0 is not open". A scene reaches this
        // legitimately, a collider-free one sizing every collider array to
        // zero, and the caller should not have to guard each download for it.
        if self.cells.len() == 0 {
            self.fresh = true;
            return Ok(());
        }
        // NOTHING TO CARRY. What a download does on the other representation is
        // bring the kernel's writes into a second array; here the kernel wrote
        // the array the host reads. The FLAG still moves, because what it
        // records is that the caller has reached the point at which reading is
        // legitimate, which is what `host` refuses without, and that rule is the
        // same on every target.
        if self.cells.is_mapped() {
            maybe_verify_host_view(device, &self.device, &mut self.cells);
            self.fresh = true;
            return Ok(());
        }
        {
            // DESTRUCTURED RATHER THAN `mem::take`d. The two fields are
            // disjoint, so borrowing them separately is what the compiler needs
            // and no swap is required to satisfy it.
            let Self { cells, device: buffer, .. } = self;
            buffer.read(device, 0, cells.as_mut_slice())?;
        }
        self.fresh = true;
        Ok(())
    }

    /// Release the allocation and drop any view of it.
    ///
    /// The twin of [`StagedBuffer::free`], and it exists for the same reason:
    /// releasing through the private `device` field would leave a view of a
    /// freed block behind.
    pub(crate) fn free(&mut self, device: &mut impl Device) -> Result<(), Fault> {
        self.cells = HostCells::empty();
        self.fresh = false;
        self.device.free(device)
    }

    /// Whether the mirror is the allocation itself.
    ///
    /// The twin of [`StagedBuffer::is_mapped`], for the conformance assertions.
    pub(crate) fn is_mapped(&self) -> bool {
        self.cells.is_mapped()
    }
}

/// A stable name for an allocation, given at [`Device::alloc`].
///
/// Handles differ between backends because arena packing does; labels do not, so
/// a label is what a cross-backend comparison and an out-of-memory report both
/// key on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AllocLabel(pub &'static str);

/// How much work one dispatch covers, and the only launch geometry a call site
/// supplies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Extent {
    /// One item per element, guarded in-kernel against a count carried in the
    /// argument record. The chunk width is the backend's own, computed from the
    /// declared cost, because it computes no value.
    Elements { count: u32 },
    /// One GROUP per element, the group's threads running cooperatively.
    ///
    /// THE SHAPE A REDUCTION NEEDS, and the one an `Elements` launch cannot
    /// express: the guard is on the GROUP index, which is uniform across the
    /// group, so a group returns whole and a `compute::threadgroup_barrier()`
    /// in the body is still reached by every thread of every group that did
    /// not. An element launch's guard is per THREAD and has no such property.
    ///
    /// `threads` is the group width the entry was written against, and
    /// `scratch_bytes` the group-local array it declares for itself. Both come
    /// from the generated declaration rather than from the call site: the
    /// entry's `[[seam::scratch]]` states the length and the driver may not
    /// size it.
    ///
    /// ON THE HOST the lanes run one after another, one whole group per step.
    /// That is exact for a body that does not synchronize, and WRONG for one
    /// that does, which is why a cooperative body owes a SERIAL TWIN rather
    /// than being run as-is.
    Groups {
        groups: u32,
        threads: u32,
        scratch_bytes: u32,
    },
}

/// Implemented by every argument record, and by nothing else.
///
/// # Safety
/// The implementor is `#[repr(C)]`, is plain old data, contains only [`HostRef`]
/// / [`Handle`] fields and 4-byte scalars, contains no `f64` (MSL has none) and
/// no `float3` or `float3x3` (16 B and 48 B in MSL against `Vec3f` at 12 and
/// `Mat3x3f` at 36), and its `HOST_REFS` names the byte offset of every
/// `HostRef` field it has.
pub unsafe trait KernelArgs: Copy + Sized + 'static {
    const KERNEL: KernelId;

    /// The entry point's name, for a diagnostic raised before the table is
    /// reached. A generated record carries the name every rendering exports;
    /// a hand-written stand-in has none to give.
    const NAME: &'static str = "<a hand-written argument record>";

    /// The guard bound the record carries, for a record that carries one.
    ///
    /// A GENERATED record always does: `[[seam::count]]` is required of an
    /// entry declaration, because Metal never faults on an out-of-bounds access
    /// and every backend rounds its launch up to whole threadgroups, so the
    /// bound has to travel with the arguments. That makes the count exist TWICE
    /// at a dispatch, once in the record and once in [`Extent::Elements`], and
    /// the two disagreeing is silent: the entry point clamps to its own field,
    /// so a record left at zero would run no elements and report nothing.
    /// [`EncoderExt::elements`] compares them and faults.
    ///
    /// `None` is the hand-written stand-in records, whose shims take `begin` and
    /// `end` and carry no bound at all. Converting every shim to a generated
    /// entry point removes them, and this default disappears with the last of
    /// them.
    fn guard_count(&self) -> Option<u32> {
        None
    }

    /// Byte offsets of this record's handle fields, and the alignment each
    /// field's POINTEE asks for, positionally matched.
    ///
    /// They exist so a dispatch can check every handle it is about to bind
    /// WITHOUT knowing the pointee types: the offsets say where the handles
    /// are in the opaque blob and the alignments say what each one owes. That
    /// is the check that would otherwise fall to `compute::arena::resolve` on
    /// the DEVICE, once per thread, for a fact that is fixed before the launch
    /// and identical in every thread of every block.
    ///
    /// Empty on a hand-written record, which carries no generated tables and
    /// is therefore not validated here.
    fn handle_offsets() -> &'static [u16] {
        &[]
    }
    fn handle_aligns() -> &'static [u16] {
        &[]
    }
}

// ===========================================================================
// DIAGNOSTICS
// ===========================================================================

/// One failing check, as the driver reads it.
///
/// The backend owns the TRANSPORT, which includes resolving whatever the channel
/// records into a file name: a `const char *` here, a file id injected by the
/// shader assembler on Metal, a device-side `__FILE__` on CUDA. What crosses the
/// seam is the resolved record, so a driver never learns which of those the
/// backend used.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagFailure {
    pub file: String,
    pub line: u32,
    pub payload: [f32; 4],
}

impl fmt::Display for DiagFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{} [{:.6e} {:.6e} {:.6e} {:.6e}]",
            self.file, self.line, self.payload[0], self.payload[1], self.payload[2],
            self.payload[3]
        )
    }
}

/// What a boundary reports back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Diag {
    /// Threads that failed a check anywhere in the region.
    pub failures: u64,
    /// The first claimant's record.
    ///
    /// FIRST IN ASCENDING CHUNK ORDER, not first in time. A device channel
    /// latches whichever thread arrives first, which is not reproducible; the
    /// chunks here are independent, so the report is defined to be the
    /// lowest-numbered chunk's and two runs of one scene name the same check.
    pub first: Option<DiagFailure>,
}

impl Diag {
    pub fn is_clean(&self) -> bool {
        self.failures == 0 && self.first.is_none()
    }
}

/// Anything the seam refuses or the platform reported.
#[derive(Clone, Debug)]
pub enum Fault {
    /// A device assertion or bounds check fired inside `region`.
    Device { region: &'static str, diag: Diag },
    /// This backend carries no implementation of this kernel. Reachable only
    /// through a driver defect: the refusal gate reads [`Device::missing`] at
    /// `initialize()` and refuses the scene by name and count before a step runs.
    MissingKernel { kernel: KernelId },
    /// A recorded region replayed after the allocator moved underneath it.
    /// `grow` may relocate a block, so a recorded argument record can name a
    /// stale handle, and Metal does not fault on addressing freed memory.
    StaleRegion {
        region: &'static str,
        recorded: u64,
        now: u64,
    },
    /// An extent whose variant the kernel does not accept, an argument block
    /// whose length disagrees with the declaration, or a record whose
    /// [`HostRef`] fields are half-wired.
    Shape {
        kernel: &'static str,
        detail: String,
    },
    Alloc {
        label: AllocLabel,
        bytes: usize,
        detail: String,
    },
    /// Anything the platform reported.
    Platform {
        call: &'static str,
        detail: String,
    },
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::Device { region, diag } => {
                write!(f, "device diagnostic in {region}: {} failing check(s)", diag.failures)?;
                if let Some(first) = &diag.first {
                    write!(f, ", first at {first}")?;
                }
                Ok(())
            }
            Fault::MissingKernel { kernel } => {
                write!(f, "backend carries no kernel {}", kernel.0)
            }
            Fault::StaleRegion {
                region,
                recorded,
                now,
            } => write!(
                f,
                "recorded region {region} was captured at allocator generation \
                 {recorded} and the allocator is now at {now}"
            ),
            Fault::Shape { kernel, detail } => write!(f, "{kernel}: {detail}"),
            Fault::Alloc {
                label,
                bytes,
                detail,
            } => write!(f, "allocating {bytes} B for {}: {detail}", label.0),
            Fault::Platform { call, detail } => write!(f, "{call}: {detail}"),
        }
    }
}

/// Facts about the device. Reporting and limits only.
///
/// A driver that BRANCHED on one of these would be a backend owning policy
/// through the back door, which is the one failure a trait cannot prevent by
/// itself. The mechanical guard is a wiring rule rather than a comment: no
/// backend name and no backend-conditional compilation in driver code.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub backend: &'static str,
    pub device_name: String,
    pub max_arenas: u32,
    /// False on Metal, and never a licence to skip a bounds check: it exists so
    /// a fatal can name the fault model that produced its verdict.
    pub faults_on_oob: bool,
    /// Whether a recorded region is realized in a form the platform REPLAYS
    /// without re-encoding it, which one target does with a captured graph and
    /// another does not.
    ///
    /// A CAPABILITY, NOT A PREFERENCE, and it is here for the same reason
    /// `faults_on_oob` is: a gate that asserts a region came back deferred has
    /// to compare against what the backend declares, or it is asserting one
    /// platform's implementation on every platform. A backend that declares
    /// this and then hands back a re-encoded region is a slowdown inside the
    /// run-to-run envelope rather than a visible failure, which is exactly what
    /// comparing the two catches.
    pub supports_deferred_regions: bool,
}

/// Counters a GATE asserts, not diagnostics a human reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Host round trips: one per [`Device::run`] and one per [`Device::replay`],
    /// whatever the repeat count. A per-phase assertion on this is what stops a
    /// shared driver from silently reacquiring the syncs the seam exists to
    /// remove.
    pub syncs: u64,
    pub dispatches: u64,
    /// Regions the backend realized in its deferred form.
    pub regions_deferred: u64,
    /// Regions the backend had to fall back to ordinary dispatches for.
    ///
    /// MUST BE ASSERTED, never merely logged: see the module comment.
    pub regions_fallback: u64,
}

// ===========================================================================
// THE SEAM
// ===========================================================================

/// Appends work to the region being built.
///
/// An `Encoder` cannot read and cannot allocate, and both restrictions are
/// load-bearing rather than tidy. A host read inside a region would be a host
/// dependency no backend can express inside a captured graph or a committed
/// command buffer; an allocation inside one is what CUDA's capture forbids.
/// Neither is expressible through this trait, so neither needs to be checked.
///
/// Consecutive entries are ordered with a full memory barrier between them,
/// which is what all three backends already give: same-stream launch ordering on
/// CUDA, a compute encoder's default serial dispatch type on Metal, and sequence
/// on the CPU. There is therefore no barrier primitive and nothing for a backend
/// to decide.
pub trait Encoder {
    /// Append one dispatch.
    ///
    /// # Safety
    /// `args` must address `args_bytes` readable bytes laid out as the record
    /// `kernel` declares, and every buffer those bytes name must stay alive and
    /// unmoved until the dispatch has executed. In immediate mode that is until
    /// this call returns; in recording mode the bytes are copied here and the
    /// BUFFERS must outlive every replay.
    unsafe fn dispatch_raw(
        &mut self,
        kernel: KernelId,
        extent: Extent,
        args: *const u8,
        args_bytes: usize,
    ) -> Result<(), Fault>;

    /// Append one FILL, batched with the dispatches around it.
    ///
    /// **A PER-STEP CLEAR IS NOT A SIZING CLEAR.** [`Device::fill_zero`] opens
    /// an immediate encoder and SUBMITS, which is right for `initialize()` and
    /// wrong inside a step: a backend synchronizes at the end of every submit,
    /// so a four-byte counter reset before a dispatch costs a full device stall.
    /// Measured on `drape` at three frames, an immediate fill per clear issued
    /// 472 memsets against the reference's 81, and every one of them was its
    /// own submit. Appending
    /// the fill to the region that follows it costs no submit at all.
    ///
    /// # Safety
    /// `handle` must name a live allocation of at least `byte_offset + bytes`,
    /// and it must stay alive and unmoved until the fill has executed, which in
    /// recording mode is until every replay has.
    unsafe fn fill_raw(
        &mut self,
        handle: Handle,
        byte_offset: usize,
        bytes: usize,
        value: u8,
    ) -> Result<(), Fault>;

    /// The enclosing region's label, so a fault names the phase.
    fn region(&self) -> &'static str;
}

/// Zero `bytes` of `handle`, appended to the region being built.
///
/// The safe spelling of [`Encoder::fill_raw`] for the common case.
pub fn encode_zero(
    encoder: &mut dyn Encoder,
    handle: Handle,
    bytes: usize,
) -> Result<(), Fault> {
    // Safety: the caller owns the allocation for the region's lifetime, which
    // is the same precondition every dispatch in the region carries.
    unsafe { encoder.fill_raw(handle, 0, bytes, 0) }
}

/// Check every handle a record is about to bind, on the HOST, once.
///
/// THIS IS THE CHECK THAT WOULD OTHERWISE FALL TO `compute::arena::resolve` ON
/// THE DEVICE, and making it here is worth about half the run time of the
/// smallest kernels. A
/// handle's `size` against its `allocated`, and its `off` against the alignment
/// its pointee asks for, are fixed before the launch and identical in every
/// thread of every block, so asserting them per thread computed one verdict
/// tens of millions of times. A generated entry resolves every buffer it takes,
/// fourteen of them for the PCG folds, which put about 56 assert branches in
/// front of a body that folds 64 lanes.
///
/// IT FAILS EARLIER AND LOUDER THAN A DEVICE TEST WOULD: a `Fault` naming the
/// kernel and the field index, raised before anything is dispatched, rather
/// than a trap inside one thread of one block with no context. On Metal, where
/// an out-of-bounds read returns 0.0 and faults on nothing, a device test is
/// not a backstop at all.
///
/// A hand-written record carries no tables and is not validated; that is the
/// same gap `guard_count` has and it closes with the last of them.
fn validate_handles<A: KernelArgs>(args: &A) -> Result<(), Fault> {
    let offsets = A::handle_offsets();
    let aligns = A::handle_aligns();
    if offsets.is_empty() {
        return Ok(());
    }
    debug_assert_eq!(
        offsets.len(),
        aligns.len(),
        "the generator emits both tables from one walk of the handle fields"
    );
    let base = (args as *const A).cast::<u8>();
    for (index, (&offset, &align)) in offsets.iter().zip(aligns).enumerate() {
        // Safety: `offset` is generated from this record's own layout and the
        // record is `#[repr(C)]`, so it names a `Handle` inside `args`. The
        // read is unaligned-safe because nothing promises the blob's alignment.
        let handle: Handle =
            unsafe { std::ptr::read_unaligned(base.add(offset as usize).cast::<Handle>()) };
        if handle.size > handle.allocated {
            return Err(Fault::Shape {
                kernel: A::NAME,
                detail: format!(
                    "handle {index} names {} elements of a {}-element \
                     allocation, so the record is malformed",
                    handle.size, handle.allocated
                ),
            });
        }
        if align != 0 && handle.off % u32::from(align) != 0 {
            return Err(Fault::Shape {
                kernel: A::NAME,
                detail: format!(
                    "handle {index} sits at byte offset {} in its arena, which \
                     is not a multiple of the {align} its pointee asks for",
                    handle.off
                ),
            });
        }
    }
    Ok(())
}

/// Typed sugar, as an extension trait so [`Encoder`] stays object-safe and
/// [`Device::run`] can take `&mut dyn Encoder`.
pub trait EncoderExt: Encoder {
    /// Append one dispatch of `count` elements.
    ///
    /// # Safety
    /// Every buffer `args` names must satisfy [`Encoder::dispatch_raw`]'s
    /// contract. This is the safe-looking wrapper over an inherently unchecked
    /// operation, and it is `unsafe` for that reason.
    unsafe fn elements<A: KernelArgs>(&mut self, args: &A, count: u32) -> Result<(), Fault> {
        validate_handles(args)?;
        if let Some(declared) = args.guard_count() {
            if declared != count {
                return Err(Fault::Shape {
                    kernel: A::NAME,
                    detail: format!(
                        "the record's guard count is {declared} and the \
                         dispatch extent is {count}. The entry point clamps to \
                         the record, so the difference would be work silently \
                         not done"
                    ),
                });
            }
        }
        self.dispatch_raw(
            A::KERNEL,
            Extent::Elements { count },
            (args as *const A).cast::<u8>(),
            std::mem::size_of::<A>(),
        )
    }

    /// Append one dispatch of `groups` GROUPS, each `threads` wide.
    ///
    /// THE WIDTH IS THE CALLER'S, THE SCRATCH IS NOT, and that is why only one
    /// of them is a parameter. A `[[seam::group]]` declaration states its own
    /// `[[seam::scratch(N)]]` and the entry allocates it STATICALLY; the
    /// extent's `scratch_bytes` is for DYNAMIC scratch, which nothing in this
    /// tree uses and which the CUDA backend refuses outright for an entry that
    /// declares its own. Passing the generated `<NAME>_SCRATCH_BYTES` here was
    /// tried and is exactly the mistake: it reads like the amount to ask for and
    /// is the amount already reserved, so the dispatch asks for a second copy
    /// and the ABI rejects it by name.
    ///
    /// THE GUARD IS CHECKED AGAINST THE GROUP COUNT for the reason
    /// [`EncoderExt::elements`] checks it against the element count: the entry
    /// clamps to the record, so a difference is work silently not done.
    ///
    /// # Safety
    /// As [`EncoderExt::elements`].
    unsafe fn groups<A: KernelArgs>(
        &mut self,
        args: &A,
        groups: u32,
        threads: u32,
    ) -> Result<(), Fault> {
        validate_handles(args)?;
        if let Some(declared) = args.guard_count() {
            if declared != groups {
                return Err(Fault::Shape {
                    kernel: A::NAME,
                    detail: format!(
                        "the record's guard count is {declared} and the \
                         dispatch names {groups} groups. The entry point \
                         clamps to the record, so the difference would be work \
                         silently not done"
                    ),
                });
            }
        }
        if threads == 0 {
            return Err(Fault::Shape {
                kernel: A::NAME,
                detail: "a group launch of zero-wide groups runs no thread and \
                         is a dispatch that cannot do its work"
                    .to_string(),
            });
        }
        self.dispatch_raw(
            A::KERNEL,
            Extent::Groups {
                groups,
                threads,
                // NONE REQUESTED. The entry's own `[[seam::scratch(N)]]` is
                // already reserved; this field is the dynamic kind.
                scratch_bytes: 0,
            },
            (args as *const A).cast::<u8>(),
            std::mem::size_of::<A>(),
        )
    }
}
impl<E: Encoder + ?Sized> EncoderExt for E {}

/// The backend surface.
///
/// Generic rather than object-safe on purpose: `build.rs` links exactly one
/// backend, so a driver written `fn advance<D: Device>` monomorphizes once and
/// no call in any loop goes through a vtable.
pub trait Device {
    /// A recorded region. `cudaGraphExec_t` plus its direct-launch fallback list
    /// on CUDA, an encoded dispatch list on Metal, a call list on the CPU.
    type Region;

    // ---- identity and capability ------------------------------------------

    fn info(&self) -> &DeviceInfo;

    /// The kernel table for this backend.
    fn kernels(&self) -> &'static [KernelDecl];

    /// Kernels this backend does not carry. The DRIVER maps these to named
    /// features and refuses a scene by name and count at `initialize()`; the
    /// backend never decides what to refuse.
    ///
    /// BORROWED FROM THE TARGET RATHER THAN `'static`, because for a target
    /// behind a C ABI the answer is not a compile-time constant: it is
    /// `be_kernel_present` asked once per id when the library is opened, so
    /// it lives exactly as long as the open backend does. A `'static` return
    /// forces such a target to leak the vector to satisfy a signature, which is
    /// a lifetime laundering rather than a fact about the kernel set.
    fn missing(&self) -> &[KernelId];

    /// Create whatever these ids need, now.
    ///
    /// The DRIVER states the set, at `initialize()`, before a frame is written.
    /// Pipeline creation is the larger of Metal's two startup costs (16.2 s for
    /// 129 pipelines against 6.4 s to compile the source), and an id that cannot
    /// be supplied must fail HERE by name rather than mid-step.
    fn prepare_kernels(&mut self, ids: &[KernelId]) -> Result<(), Fault>;

    // ---- memory -----------------------------------------------------------

    /// Allocates `count` elements of `elem_size` bytes at `align`.
    ///
    /// **THE REQUEST IS VALIDATED AT ALLOCATION TIME**, which is the earliest
    /// point the question can be asked and the only point at which it can be
    /// answered loudly: Metal silently FLOORS a misaligned buffer offset to a
    /// multiple of 4 and returns the wrong data with status Completed and no
    /// diagnostic. `elem_size` must be non-zero; `align` must be a power of two
    /// and at least [`MIN_ALIGN`]; and `align` must divide `elem_size`, with one
    /// sanctioned exception, an element SMALLER than the alignment, which needs
    /// at most its own size and so is satisfied by any offset inside a block
    /// that starts aligned. Every target applies exactly these rules, so a
    /// request one serves is a request all of them serve.
    ///
    /// A count of 0 yields a real zero-length handle naming a bound arena, never
    /// the [`Handle::NONE`] sentinel, so a bounds check on it fails loudly
    /// instead of indexing an unbound slot.
    ///
    /// FRESH BYTES ARE UNSPECIFIED. A target that happened to hand back zeros is
    /// what hides a missing clear.
    fn alloc(
        &mut self,
        count: usize,
        elem_size: usize,
        align: usize,
        label: AllocLabel,
    ) -> Result<Handle, Fault>;

    /// Grows an allocation to `new_count` elements, preserving its contents.
    ///
    /// The block MAY MOVE, so every copy of the handle held elsewhere is stale,
    /// a recorded region included, which is what the generation counter catches.
    /// `elem_size` and `align` must match the original allocation and are
    /// checked. A `new_count` below the current capacity is REFUSED rather than
    /// silently shrinking, and the LOGICAL size the handle carries is preserved:
    /// this call moves `allocated`, and a caller that wants the length to follow
    /// sets it, which is what [`Buffer::size`] does.
    fn grow(
        &mut self,
        handle: &mut Handle,
        new_count: usize,
        elem_size: usize,
        align: usize,
    ) -> Result<(), Fault>;

    /// Leaves the handle at [`Handle::NONE`], on every target.
    ///
    /// THE SPELLING IS PART OF THE CONTRACT rather than an implementation
    /// detail. A caller asks `is_none()` to tell an allocation it must make from
    /// one it must grow, so a target that left a freed handle zeroed would send
    /// the next [`Buffer::size`] down the grow path against a block it had just
    /// released. Freeing a handle that already names nothing is legal and does
    /// nothing.
    fn free(&mut self, handle: &mut Handle) -> Result<(), Fault>;

    /// Host to device. `byte_offset` is measured from the start of the block, so
    /// a caller may write a window of a block that carries several arrays end to
    /// end, and both ends are checked against the block's BYTE capacity, which
    /// only the allocator knows (see [`Handle`]). A handle whose `off` has been
    /// advanced by hand names no live block and is refused, which is the correct
    /// answer to a handle that has been invented.
    fn write(&mut self, handle: Handle, byte_offset: usize, src: &[u8]) -> Result<(), Fault>;

    /// Device to host, same window rule.
    fn read(&mut self, handle: Handle, byte_offset: usize, dst: &mut [u8]) -> Result<(), Fault>;

    /// Copy `bytes` from one allocation to another WITHOUT touching the host.
    ///
    /// THE MISSING HALF OF "COPY AND TRANSFER". `write` moves host to device
    /// and `read` moves device to host; a driver that owns its buffers also
    /// needs device to device, and without it a seed like the Newton iterate's
    /// has to round-trip through the host, which is two transfers and a stall
    /// on a real GPU to do no work.
    ///
    /// It is admissible under this crate's closed list by name: allocation,
    /// free, copy and transfer, and kernel launch. This is the copy.
    ///
    /// The two spans must not overlap.
    fn copy(
        &mut self,
        dst: Handle,
        dst_byte_offset: usize,
        src: Handle,
        src_byte_offset: usize,
        bytes: usize,
    ) -> Result<(), Fault>;

    /// Bumped by every `alloc`, `grow` and `free`.
    fn allocator_generation(&self) -> u64;

    fn bytes_reserved(&self) -> u64;

    // ---- execution --------------------------------------------------------

    /// Encode and execute once, wait, and collect the diagnostic channel.
    ///
    /// A clean boundary returns its `Diag`; a boundary carrying a failure
    /// returns `Fault::Device`, so a driver cannot proceed past a device assert
    /// by ignoring a value.
    fn run<F>(&mut self, region: &'static str, body: F) -> Result<Diag, Fault>
    where
        F: FnOnce(&mut dyn Encoder) -> Result<(), Fault>;

    /// Encode into a replayable region without executing. The closure runs
    /// exactly once, at record time.
    fn record<F>(&mut self, region: &'static str, body: F) -> Result<Self::Region, Fault>
    where
        F: FnOnce(&mut dyn Encoder) -> Result<(), Fault>;

    /// Execute a recorded region exactly `repeats` times, back to back, with no
    /// host round trip anywhere inside, then wait and collect the diagnostic
    /// channel ONCE for the whole batch.
    ///
    /// THE CONTRACT THAT FOLLOWS, and it must be read before recording anything:
    /// a failure raised on repeat `k` is reported after repeat `repeats - 1` has
    /// run. A body may therefore be recorded only if its failure mode is LATCH
    /// AND CONTINUE. Nothing whose failure must stop the step before the next
    /// dispatch is encoded belongs in a region; those stay one-shot [`Device::run`]
    /// calls, which is where the contact, CCD and intersection kernels are.
    fn replay(&mut self, region: &Self::Region, repeats: u32) -> Result<Diag, Fault>;

    fn release(&mut self, region: Self::Region);

    fn counters(&self) -> Counters;

    fn counters_reset(&mut self);

    // ---- provided ---------------------------------------------------------

    /// Zero the first `bytes` of an allocation.
    ///
    /// Provided rather than required, because the chunked transfer below is
    /// correct on any target that implements the seam's four memory calls, and
    /// a target with a native fill (`cudaMemsetAsync`, a blit encoder) overrides
    /// it with one. It is a `Device` call rather than an [`Encoder`] one because
    /// what it is for is sizing at `initialize()`, outside any region.
    ///
    /// The chunk is what keeps a large buffer from needing a host copy of its
    /// own size: a scene's largest array is measured in hundreds of megabytes,
    /// and a zeroing that allocated one would be a second peak.
    fn fill_zero(&mut self, handle: Handle, bytes: usize) -> Result<(), Fault> {
        const CHUNK: usize = 1 << 16;
        let zeros = [0u8; CHUNK];
        let mut written = 0;
        while written < bytes {
            let span = CHUNK.min(bytes - written);
            self.write(handle, written, &zeros[..span])?;
            written += span;
        }
        Ok(())
    }

    /// A host-addressable view of a window inside an allocation, where this
    /// target serves one.
    ///
    /// PROVIDED AND DEFAULTING TO ABSENT, on the same grounds [`Device::fill_zero`]
    /// is provided: the copy through [`Device::write`] and [`Device::read`] is
    /// correct on any target that implements the seam's memory calls, and a
    /// target that can do better says so by overriding. A target that answers
    /// `None` is stating what it implements, not what its memory is; the host
    /// target's arenas are host memory and it still answers `None`, because
    /// serving a view there is a separate change that owes its own scene sweep.
    ///
    /// `Ok(None)` is the answer for a target that serves no view and for a
    /// zero-byte window. An `Err` is a real refusal: a handle naming no live
    /// block, a window outside it, or a call made where a view may not be handed
    /// out.
    ///
    /// # Safety
    /// The returned pointer addresses `bytes` bytes of the allocation `handle`
    /// names. It may be read and written only while no device work is in flight,
    /// which is every point at which caller code runs: every execution entry
    /// point of a target that answers here completes its device work before
    /// returning. It is invalidated by [`Device::grow`] or [`Device::free`] on
    /// THIS handle and by nothing else, so a caller that re-takes it whenever it
    /// sizes or releases its own allocation holds a valid one.
    unsafe fn host_view(
        &mut self,
        handle: Handle,
        byte_offset: usize,
        bytes: usize,
    ) -> Result<Option<std::ptr::NonNull<u8>>, Fault> {
        let _ = (handle, byte_offset, bytes);
        Ok(None)
    }

    /// The one-dispatch case, which is what most of the driver is.
    ///
    /// # Safety
    /// As [`EncoderExt::elements`].
    unsafe fn launch<A: KernelArgs>(
        &mut self,
        region: &'static str,
        args: &A,
        count: u32,
    ) -> Result<Diag, Fault> {
        self.run(region, |e| e.elements(args, count))
    }

    /// One dispatch of a GROUP-shaped entry, submitted on its own.
    ///
    /// [`Device::launch`]'s shape for the other extent. A caller that wants
    /// several dispatches in one submit uses [`Device::run`] and
    /// [`EncoderExt::groups`]; this is the one-shot form, and it waits.
    ///
    /// # Safety
    /// As [`Device::launch`].
    unsafe fn groups_launch<A: KernelArgs>(
        &mut self,
        region: &'static str,
        args: &A,
        groups: u32,
        threads: u32,
    ) -> Result<Diag, Fault> {
        self.run(region, |e| e.groups(args, groups, threads))
    }
}

/// THE TWO CONFORMANCE ASSERTIONS EVERY [`Device`] MUST PASS, so that
/// [`Device::free`]'s "leaves the handle at [`Handle::NONE`], on every target"
/// is checked on each target rather than restated at each one.
///
/// **WHY THIS IS NOT A TEST IN ONE MODULE.** The two implementations reach the
/// same spelling by different routes and can lose it independently.
/// [`crate::host::HostDevice`] writes the sentinel itself, so its own code is
/// the evidence. The C-ABI target does not: the library ZEROES the handle,
/// which is right for the library because its own bounds checks then reject it,
/// and the caller's copy is NORMALIZED afterwards by
/// `abi::device_impl::AbiDevice::free`. Deleting that one line leaves every
/// existing test green on any host, GPU or not, because nothing asked what a
/// freed handle spells. So the assertion lives once and is called from both
/// test modules, and a third target added to this crate owes the same two
/// calls.
///
/// It is written against `impl Device` and uses only the seam's own memory
/// calls, so it needs no GPU and no kernel: a target that can allocate, write,
/// read and free can run it.
#[cfg(test)]
pub(crate) fn assert_free_leaves_the_sentinel(device: &mut impl Device) {
    let mut scratch: Buffer<f32> = Buffer::none();
    scratch
        .size(device, 4, AllocLabel("test.free.scratch"))
        .expect("four floats");
    assert!(
        !scratch.handle().is_none(),
        "a sized buffer must name a real block, or the free below asserts nothing"
    );

    scratch.free(device).expect("the block is released");

    // EQUALITY, NOT `is_none()`. The sentinel is a whole handle, and a target
    // that set the arena while leaving a stale `off`, `size` or `allocated`
    // behind would satisfy `is_none()` and still hand the next span a length it
    // no longer owns.
    assert_eq!(
        scratch.handle(),
        Handle::NONE,
        "a freed handle spells Handle::NONE on every target. A target that left it \
         ZEROED names arena 0 at offset 0, which is a real arena and the very \
         address the first allocation lands at"
    );
    assert_eq!(scratch.len(), 0, "a released buffer holds no elements");

    // Freeing again is legal and does nothing, which is the property that lets a
    // caller release unconditionally.
    scratch.free(device).expect("a second free is a no-op");
    assert_eq!(scratch.handle(), Handle::NONE);
}

/// THE CONSEQUENCE OF THE SENTINEL, asserted through the data rather than
/// through the handle, so it fails even if the handle assertion above is
/// weakened or removed.
///
/// [`Buffer::size`] asks `is_none()` to tell an allocation it must make from one
/// it must grow. A freed handle left at all zeros answers that question wrongly:
/// `is_none()` is false, the requested count exceeds the `allocated` of 0, and
/// the grow path runs against ARENA 0 AT OFFSET 0, which is a live block
/// belonging to somebody else. What follows is not a crash on any target and is
/// silent on one that never faults: the block is resized, the zero fill that
/// [`Buffer::size`] owes its caller runs over it, and the other buffer's
/// contents are gone while every call reported success.
#[cfg(test)]
pub(crate) fn assert_sizing_after_a_free_spares_the_first_allocation(device: &mut impl Device) {
    // ALLOCATED FIRST, so it is the block a zeroed handle would name.
    let mut keep: Buffer<f32> = Buffer::none();
    keep.size(device, 4, AllocLabel("test.free.keep"))
        .expect("four floats");
    let pattern = [1.0f32, 2.0, 3.0, 4.0];
    keep.write(device, 0, &pattern).expect("a whole-block write");

    let mut scratch: Buffer<f32> = Buffer::none();
    scratch
        .size(device, 4, AllocLabel("test.free.scratch"))
        .expect("four floats");
    scratch.free(device).expect("the block is released");

    // LARGER than what it held, so a handle that survived the free takes the
    // grow path rather than finding the capacity already there.
    scratch
        .size(device, 8, AllocLabel("test.free.scratch"))
        .expect("eight floats");

    let mut back = [0.0f32; 4];
    keep.read(device, 0, &mut back).expect("a whole-block read");
    assert_eq!(
        back, pattern,
        "sizing a released buffer overwrote the first allocation. The freed handle \
         was left naming arena 0 at offset 0 rather than Handle::NONE, so \
         Buffer::size grew and zeroed a block it did not own"
    );
    assert_eq!(keep.len(), 4, "the surviving buffer keeps its length");
}

/// THE THIRD CONFORMANCE ASSERTION: a prefix upload carries exactly its prefix,
/// and the buffer stays refused as a whole until a full upload has run.
///
/// **WHY IT IS A CONFORMANCE CALL AND NOT ONE TEST.** The accounting is target
/// independent, but what it is accounting FOR is not: the transfer is the
/// seam's own `write`, whose bound each target derives for itself, and the
/// handle comes from [`Buffer::span`], whose arithmetic is in element widths a
/// target's allocator chose. A target that carried more than the prefix would
/// pass any assertion made on the length alone, so this one READS THE DEVICE
/// BACK and requires the tail to be untouched.
///
/// The failure it exists to catch is silent in the direction that matters. A
/// caller compacts an active run into a prefix, uploads it, and dispatches over
/// it; if a later handle named the whole allocation, the tail would hold the
/// PREVIOUS run, which is a plausible wrong answer on a target that never
/// faults rather than a trap.
///
/// It needs no GPU and no kernel: allocate, write, read and free are enough.
#[cfg(test)]
pub(crate) fn assert_a_prefix_upload_names_only_what_it_carried(device: &mut impl Device) {
    let mut staged: StagedBuffer<u32> = StagedBuffer::default();
    staged
        .size(device, 8, AllocLabel("test.prefix.staged"))
        .expect("eight words");
    staged.at().copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

    let handle = staged
        .upload_span(device, 3)
        .expect("a three-element prefix");
    assert_eq!(
        handle.size, 3,
        "the handle a prefix upload returns names the elements it carried"
    );
    assert_eq!(
        handle.allocated, 3,
        "and no more of them, or a bound read off the handle admits the tail"
    );

    // READ BACK THROUGH THE ALLOCATION ITSELF, which this module can reach and
    // a caller cannot: what is under test is whether the tail moved, and no
    // public handle names it while the buffer is refused.
    let mut read = [0u32; 8];
    staged
        .device
        .read(device, 0, &mut read)
        .expect("the eight read back");
    assert_eq!(
        &read[..3],
        &[1, 2, 3],
        "the prefix reached the device"
    );
    if staged.is_mapped() {
        assert_eq!(
            &read[3..],
            &[4, 5, 6, 7, 8],
            "WHERE THE HOST SIDE IS THE ALLOCATION THE TAIL IS THERE, and the \
             property the copy path spells in bytes is spelled here by the \
             refusal below: the handle this call returned names three elements \
             and the whole-array handle is refused, so no dispatch can name the \
             tail whatever it holds"
        );
    } else {
        assert_eq!(
            &read[3..],
            &[0, 0, 0, 0, 0],
            "and NOTHING PAST IT did. `size` zeroed the allocation, so a target \
             that carried the whole host array would show 4 through 8 here"
        );
    }

    // THE WHOLE-BUFFER HANDLE STAYS REFUSED, because five elements the host
    // wrote never left it. The refusal is an assert, so the check is that it
    // panics; a target that cleared the accounting on a partial transfer would
    // hand out a handle naming the stale tail.
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| staged.handle()));
    assert!(
        refused.is_err(),
        "a buffer current for 3 of 8 elements must refuse a handle naming all 8"
    );

    staged.upload(device).expect("the whole array");
    let _ = staged.handle();
    staged
        .device
        .read(device, 0, &mut read)
        .expect("the eight read back");
    assert_eq!(
        read,
        [1, 2, 3, 4, 5, 6, 7, 8],
        "a full upload carries the rest and makes the whole buffer current"
    );
    staged.free(device).expect("the block is released");
}

/// THE FOURTH CONFORMANCE ASSERTION: a readback answers out of the device, and
/// refuses to answer at all between the dispatch and the download.
///
/// **WHY IT IS A CONFORMANCE CALL AND NOT ONE TEST**, on the same grounds as the
/// prefix assertion above: the accounting is target independent, but the
/// transfer under it is the seam's own `read`, which each target implements by a
/// different route. A target whose `read` returned the host's own previous
/// contents would satisfy every assertion made on the accounting alone, so this
/// one puts bytes on the DEVICE that the mirror has never held and requires them
/// to come back.
///
/// The failure it exists to catch is the silent one this type was added for. A
/// verdict array is written by a kernel and read by a host branch; if the mirror
/// answered without a download it would answer out of the PREVIOUS iteration,
/// which on a scene whose active set barely moves is a plausible verdict rather
/// than a trap, and no target faults on it.
///
/// It needs no GPU and no kernel: the device-side write stands in for what a
/// kernel would have left behind, since what is under test is the direction and
/// the accounting, not the arithmetic of any particular body.
#[cfg(test)]
pub(crate) fn assert_a_readback_answers_only_after_a_download(device: &mut impl Device) {
    let mut readback: ReadbackBuffer<u32> = ReadbackBuffer::default();
    readback
        .size(device, 4, AllocLabel("test.readback"))
        .expect("four words");

    // SIZED AND NOT YET DISPATCHED, so the mirror is current: both halves were
    // zeroed by `size`, and a caller reading here gets the zero every other
    // buffer in this crate starts at.
    assert_eq!(readback.host(), &[0, 0, 0, 0], "a sized readback reads zero");

    // THE DISPATCH. Asking for the handle is what marks the mirror stale, since
    // the buffer cannot see the launch that follows it.
    let _ = readback.handle();

    // WHAT THE KERNEL WOULD HAVE LEFT BEHIND, written through the allocation
    // this module can reach and a caller cannot. These four words have never
    // been in the mirror, so a target answering out of the host array shows
    // zeros below.
    readback
        .device
        .write(device, 0, &[7u32, 0, 9, 0])
        .expect("the verdicts reach the device");

    // THE MIRROR REFUSES. The refusal is an assert, so the check is that it
    // panics: a caller reading here would take a branch on the contents from
    // before the dispatch.
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        readback.host().to_vec()
    }));
    assert!(
        refused.is_err(),
        "a readback named by a dispatch must refuse to answer until it is \
         downloaded"
    );

    readback.download(device).expect("the four come back");
    assert_eq!(
        readback.host(),
        &[7, 0, 9, 0],
        "a download answers out of the DEVICE, not out of what the mirror held"
    );

    // AND STALE AGAIN on the next dispatch, so the guard is not a one-shot that
    // a first download disarms for the rest of the run.
    let _ = readback.handle();
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        readback.host().to_vec()
    }));
    assert!(
        refused.is_err(),
        "a second dispatch makes the mirror stale again; the download does not \
         disarm the guard"
    );

    readback.download(device).expect("the four come back again");

    // A REUSED SIZING WITHDRAWS THE CLAIM, on both representations. Re-sizing to
    // a length the allocation already covers copies nothing and zeroes nothing,
    // so the mirror is no longer known to match the device and `fresh` goes
    // back to false. The staged twin is asserted the same way; this is the half
    // where the tail rule below lives, so it is asserted here rather than
    // assumed from that one.
    readback
        .size(device, 4, AllocLabel("test.readback"))
        .expect("a re-size to the length it already holds");
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        readback.host().to_vec()
    }));
    assert!(
        refused.is_err(),
        "a reused sizing withdraws the mirror's claim, so a read before a \
         download must refuse"
    );
    readback.download(device).expect("a download makes it answerable again");
    assert_eq!(
        readback.host(),
        &[7, 0, 9, 0],
        "a reused sizing preserves the allocation, so the download brings back \
         what the device still holds"
    );

    // THE TAIL PAST THE PREVIOUS LENGTH IS THE ALLOCATION'S, NOT A DEFAULT.
    // Growing inside the capacity writes no device bytes on either
    // representation, so what a download brings back for those elements is
    // whatever the arena holds. Defaulting them on the mapped arm alone would
    // put the two targets on different bytes after this very download, which is
    // the fork this rule exists to prevent, so the assertion is that both
    // answer the same thing rather than that they answer any particular thing.
    // FIVE, NOT SIX, AND THE FIGURE IS THE POINT. `size(4)` asks the allocator
    // for `4 + 4 / 4`, so the capacity is five: a growth to five is inside it
    // and reuses the block, while a growth to six exceeds it and is a fresh
    // allocation the sizing zeroes. Only the first is the case this asserts.
    readback
        .size(device, 5, AllocLabel("test.readback"))
        .expect("a growth inside the capacity");
    readback.download(device).expect("five come back");
    assert_eq!(
        readback.host()[..4],
        [7, 0, 9, 0],
        "a growth inside the capacity reuses the block, so the elements the \
         device already held survive it"
    );
    assert_eq!(readback.host().len(), 5, "the mirror is the new length");

    readback.free(device).expect("the block is released");
}

/// A staged upload whose bytes are already there transfers nothing, and does
/// not forget what it knows.
///
/// **THE SECOND HALF IS THE SUBTLE ONE.** If `upload_span` assigned the prefix
/// it transferred to `clean` unconditionally, asking for a SHORTER prefix than
/// the mirror already holds would throw away the knowledge that the tail is
/// current, and the next `handle` would refuse a buffer that is in fact whole.
/// It cannot simply take the larger of the two either: when the host HAS been
/// rewritten, a shorter prefix really does leave the tail holding the previous
/// contents, and claiming otherwise is how a caller silently reads them. The
/// two cases are told apart by whether `at` was called, which is what `clean`
/// records.
pub(crate) fn assert_a_redundant_upload_transfers_nothing(device: &mut impl Device) {
    let mut staged = StagedBuffer::<u32>::default();
    staged
        .size(device, 4, AllocLabel("test.elision"))
        .expect("four elements are staged");

    // This sizing is fresh, so both halves are zeroed and `clean` is the length:
    // all four are current before anything is written and an upload here has
    // nothing to do. A REUSED sizing withdraws `clean` instead, which is why
    // this test sizes a buffer it just created.
    staged.at().copy_from_slice(&[5, 6, 7, 8]);
    staged.upload(device).expect("the four go across");
    assert_eq!(
        staged.host(),
        &[5, 6, 7, 8],
        "an upload does not disturb the host copy it sends"
    );

    // THE ELISION. Nothing has been written since, so this transfers nothing
    // and the mirror stays whole.
    staged
        .upload(device)
        .expect("a redundant upload is still a success");
    let _ = staged.handle();

    // A SHORTER PREFIX MUST NOT SHRINK WHAT IS KNOWN. Two elements are asked
    // for, four are current, and `handle` must still answer for all four.
    staged
        .upload_span(device, 2)
        .expect("a shorter prefix of a current mirror is a no-op");
    let _ = staged.handle();

    // AND A REWRITE STILL GOES ACROSS, which is the case the elision must not
    // swallow: `at` zeroes `clean`, so the next upload transfers.
    staged.at().copy_from_slice(&[1, 2, 3, 4]);
    staged.upload(device).expect("the rewrite goes across");
    let mut mirror = [0u32; 4];
    staged
        .device
        .read(device, 0, &mut mirror)
        .expect("the four come back");
    assert_eq!(
        mirror,
        [1, 2, 3, 4],
        "a staged buffer written through `at` must reach the device; the \
         elision may only skip a transfer whose bytes are already there"
    );

    // A SHORTER PREFIX AFTER A REWRITE IS THE OPPOSITE CASE and must NOT be
    // elided: `at` has zeroed `clean`, so the two elements really do have to
    // travel, and the tail is then genuinely stale.
    staged.at().copy_from_slice(&[9, 9, 9, 9]);
    staged
        .upload_span(device, 2)
        .expect("two elements go across");
    staged
        .device
        .read(device, 0, &mut mirror)
        .expect("the four come back again");
    // WHERE THE HOST SIDE IS THE ALLOCATION the tail is already current, `at`
    // having written it in place, and the refusal below is unchanged anyway:
    // `clean` shrank to the prefix, so no dispatch may name the whole array.
    let expected = if staged.is_mapped() {
        [9, 9, 9, 9]
    } else {
        [9, 9, 3, 4]
    };
    assert_eq!(
        mirror, expected,
        "a shorter prefix transfers exactly its own elements and leaves the \
         tail holding what the device had before, which is why `clean` shrinks \
         to the prefix and the whole-array `handle` below is refused"
    );
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| staged.handle()));
    assert!(
        refused.is_err(),
        "after a partial upload the mirror is current for a prefix only, and a \
         dispatch over the whole array must be refused"
    );

    staged.free(device).expect("the block is released");
}

/// THE FIFTH CONFORMANCE ASSERTION: whichever representation a target chose, it
/// is the one the target's own transfer agrees with.
///
/// **WHY IT IS A CONFORMANCE CALL.** The four above assert the accounting, and
/// the accounting is identical on both representations by construction. What is
/// NOT identical is where the bytes are, and a target that reported a host view
/// and then handed back a staging copy would satisfy every one of them while
/// every `at` wrote into memory no dispatch reads. That failure is silent on a
/// target that never faults, so it is asserted directly and in both directions:
/// the mapped arm requires a host write to reach the allocation with NO upload,
/// and the copy arm requires it NOT to.
///
/// **THE SECOND HALF IS THE RE-SIZE, AND IT ASSERTS ONE ANSWER ON BOTH ARMS.**
/// [`Buffer::size`] zeroes an allocation only when the bytes are NEW, so a
/// buffer re-sized to a length it already holds keeps what it last held. That
/// is a fact about the allocation, and it has to become the same fact about the
/// host side however the host side is represented: the mapped one keeps it by
/// not writing the allocation, the owned one by keeping its array, and
/// [`StagedBuffer::size`] withdraws `clean` so neither can be dispatched
/// against until an upload has run. A version of this assertion that stated one
/// expectation per arm would be recording a backend fork as intended behavior,
/// which is the acceptance rule inverted.
pub(crate) fn assert_a_host_view_aliases_the_allocation(device: &mut impl Device) {
    let mut staged: StagedBuffer<u32> = StagedBuffer::default();
    staged
        .size(device, 4, AllocLabel("test.hostview"))
        .expect("four words");

    // A HOST WRITE AND NO UPLOAD. Read through the allocation itself, which this
    // module can reach and a caller cannot.
    staged.at().copy_from_slice(&[7, 7, 7, 7]);
    let mut read = [0u32; 4];
    staged
        .device
        .read(device, 0, &mut read)
        .expect("the four read back");
    if staged.is_mapped() {
        assert_eq!(
            read,
            [7, 7, 7, 7],
            "a target reporting a host view must let a host write REACH the \
             allocation with no transfer. Reading zeros here means the address \
             it handed back names a staging copy, so every `at` in the run wrote \
             memory no dispatch reads"
        );
    } else {
        assert_eq!(
            read,
            [0, 0, 0, 0],
            "a target serving no host view must leave the allocation at what \
             `size` zeroed until an upload runs, or the copy path is not the \
             path it is taking"
        );
    }

    // AND THE ACCOUNTING IS THE SAME ON BOTH, which is what makes a scene that
    // passes on one target pass on the others: the whole-array handle is refused
    // until an upload has run, whether or not one moves any bytes.
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| staged.handle()));
    assert!(
        refused.is_err(),
        "a staged buffer written through `at` and not uploaded must refuse a \
         handle on every target, however its host side is represented"
    );
    staged.upload(device).expect("the four go across");
    let _ = staged.handle();
    staged
        .device
        .read(device, 0, &mut read)
        .expect("the four read back");
    assert_eq!(
        read,
        [7, 7, 7, 7],
        "after the upload the allocation holds what the host wrote, on both \
         representations"
    );

    // THE NO-OP RE-SIZE, at the length the buffer already holds, so
    // `Buffer::size` allocates nothing and zeroes nothing. ONE EXPECTATION, ON
    // BOTH ARMS.
    staged
        .size(device, 4, AllocLabel("test.hostview"))
        .expect("the same four words");
    assert_eq!(
        staged.host(),
        &[7, 7, 7, 7],
        "a re-size that allocated nothing preserves what the host side held, on \
         BOTH representations: the mapped one by not writing the allocation, the \
         copy one by keeping its array. The two answering differently here is a \
         target reading different simulation input from one code path, which is \
         the fork this assertion exists to refuse"
    );
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| staged.handle()));
    assert!(
        refused.is_err(),
        "and a re-size that zeroed nothing withdraws the claim that the two \
         halves agree, on both representations. The copy path's allocation holds \
         the last upload while its host side holds its array, so a handle granted \
         here would dispatch against whichever of the two the target happens to \
         keep"
    );
    staged.upload(device).expect("the four go across again");
    let _ = staged.handle();
    staged
        .device
        .read(device, 0, &mut read)
        .expect("the four read back");
    assert_eq!(
        read,
        [7, 7, 7, 7],
        "and after the upload the allocation holds what the host side held, on \
         both representations"
    );

    // A GROWTH IS FRESH, so `Buffer::size` zeroes the allocation and both
    // halves read the element's default with no upload owed.
    staged
        .size(device, 16, AllocLabel("test.hostview"))
        .expect("sixteen words");
    assert_eq!(
        staged.host(),
        &[0u32; 16],
        "a fresh allocation reads its element's default on both representations"
    );
    let _ = staged.handle();
    let mut wide = [1u32; 16];
    staged
        .device
        .read(device, 0, &mut wide)
        .expect("the sixteen read back");
    assert_eq!(
        wide,
        [0u32; 16],
        "and `Buffer::size` zeroed the allocation to match, which is what makes \
         the whole length current with no transfer"
    );

    // AND THE HALF OF A REUSED SIZING THAT IS NOT PRESERVED: the elements past
    // the previous length are ones no host write ever reached, so they take the
    // element's default while the ones below it keep what they held. Sixteen
    // words were asked for with a quarter of headroom, so eighteen is inside the
    // capacity and is reused.
    staged.at().fill(5);
    staged.upload(device).expect("the sixteen go across");
    staged
        .size(device, 18, AllocLabel("test.hostview"))
        .expect("eighteen words");
    let mut expected = [0u32; 18];
    expected[..16].fill(5);
    assert_eq!(
        staged.host(),
        &expected,
        "a reused sizing keeps the elements the host side already had and \
         defaults the ones past its previous length, on both representations. \
         The mapped one fills exactly that tail and the copy one resizes into \
         it, so a target cannot answer this with a different array"
    );

    staged.free(device).expect("the block is released");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_host_reference_is_coherent_or_it_is_a_defect() {
        assert!(HostRef::none().is_coherent());
        let values = [1.0f32, 2.0, 3.0];
        assert!(HostRef::of(&values).is_coherent());
        // An EMPTY slice keeps a real address and is legitimate, the same way a
        // zero-length handle names a real bound arena. Several scenes carry one.
        let empty: [f32; 0] = [];
        assert!(HostRef::of(&empty).is_coherent());
        // A length with no address is the half-wired record: a count filled in
        // and the pointer beside it left null.
        let half_wired = HostRef {
            addr: 0,
            bytes: 64,
        };
        assert!(!half_wired.is_coherent());
    }

    #[test]
    fn a_host_reference_is_exactly_a_handle_wide() {
        // The migration from HostRef to Handle must move no other field in a
        // record, which is only true while the two are the same width.
        assert_eq!(
            std::mem::size_of::<HostRef>(),
            std::mem::size_of::<Handle>()
        );
    }

    /// A staged buffer with a host side and no allocation, for the two witness
    /// tests below.
    ///
    /// BUILT FROM THE FIELDS rather than through `size`, because what is under
    /// test is the witness and not the sizing, and a device would only decide
    /// which representation the buffer got. The witness is armed and answered
    /// identically on both, the mapped one being the only one that ARMS it in
    /// production.
    fn witnessed(cells: Vec<u32>, count: usize) -> StagedBuffer<u32> {
        let hash = prefix_hash(&cells, count);
        StagedBuffer {
            cells: HostCells::Owned(cells),
            device: Buffer::none(),
            clean: count,
            witness: Some(UploadWitness { count, hash }),
        }
    }

    #[test]
    fn an_upload_witness_passes_while_only_the_host_writes() {
        // THE CASE THAT MUST NOT FIRE, and it is the whole run: an upload
        // fingerprints its prefix, the caller reads it back, and nothing has
        // written the allocation in between.
        let staged = witnessed(vec![3, 1, 4, 1], 4);
        assert_eq!(staged.host(), &[3, 1, 4, 1]);

        // AND A PREFIX WITNESS SAYS NOTHING ABOUT THE TAIL, which the upload
        // that armed it did not name either.
        let mut staged = witnessed(vec![3, 1, 4, 1], 2);
        staged.cells.as_mut_slice()[3] = 9;
        assert_eq!(staged.host(), &[3, 1, 4, 9]);
    }

    #[test]
    fn an_upload_witness_catches_a_write_the_host_did_not_make() {
        // A KERNEL WRITING A STAGED ALLOCATION, which on the mapped
        // representation is a write straight into the array `host` answers out
        // of. `verify_elided` cannot see it there, the download it compares
        // against being the same memory, so this is the check that does.
        let mut staged = witnessed(vec![3, 1, 4, 1], 4);
        staged.cells.as_mut_slice()[2] = 0;
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            staged.host().to_vec()
        }));
        assert!(
            refused.is_err(),
            "a read after something other than the host wrote the uploaded \
             prefix must be refused, or PPF_VERIFY_UPLOAD_ELISION covers the \
             copy representation only"
        );

        // AND AT `at` TOO, because a caller that writes one element and uploads
        // would otherwise carry the foreign write across with it.
        let mut staged = witnessed(vec![3, 1, 4, 1], 4);
        staged.cells.as_mut_slice()[0] = 0;
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            staged.at()[0] = 1;
        }));
        assert!(refused.is_err(), "and `at` answers for it as well as `host`");

        // A BUFFER WITH NO WITNESS ANSWERS FOR NOTHING, which is every buffer
        // in a run without the knob and every buffer on the copy path.
        let mut staged = witnessed(vec![3, 1, 4, 1], 4);
        staged.witness = None;
        staged.cells.as_mut_slice()[2] = 0;
        assert_eq!(staged.host(), &[3, 1, 0, 1]);
    }
}

#[cfg(test)]
mod handle_validation_tests {
    use super::*;

    /// A record shaped like a generated one: two handles at known offsets, the
    /// second asking for a 32-byte pointee as an `AABB` field does.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TwoHandles {
        a: Handle,
        b: Handle,
        count: u32,
    }
    unsafe impl KernelArgs for TwoHandles {
        const KERNEL: KernelId = KernelId(0);
        const NAME: &'static str = "two_handles";
        fn guard_count(&self) -> Option<u32> {
            Some(self.count)
        }
        fn handle_offsets() -> &'static [u16] {
            &[0, 16]
        }
        fn handle_aligns() -> &'static [u16] {
            &[4, 32]
        }
    }
    fn record(a: Handle, b: Handle) -> TwoHandles {
        TwoHandles { a, b, count: 1 }
    }
    fn ok(off: u32) -> Handle {
        Handle { arena: 0, off, size: 4, allocated: 8 }
    }

    /// THE WELL-FORMED CASE PASSES, or every rejection below proves nothing.
    #[test]
    fn a_well_formed_record_is_accepted() {
        assert!(validate_handles(&record(ok(4), ok(64))).is_ok());
    }

    /// `size <= allocated`, which the device asserted once per THREAD.
    #[test]
    fn a_handle_longer_than_its_allocation_is_refused() {
        let mut bad = ok(4);
        bad.size = 9;
        let err = validate_handles(&record(bad, ok(64))).unwrap_err();
        let text = format!("{err}");
        assert!(
            text.contains("9 elements of a 8-element allocation"),
            "the fault must name both counts: {text}"
        );
    }

    /// `off % alignof(T) == 0`, the check 194c measured as the costly one and
    /// the one a host validator could not make until the generator emitted the
    /// pointee alignments beside the offsets.
    #[test]
    fn a_handle_misaligned_for_its_pointee_is_refused() {
        // 64 is fine for a 32-aligned pointee; 48 is not, and it IS a legal
        // offset for a 4-aligned one, so only the table distinguishes them.
        assert!(validate_handles(&record(ok(4), ok(64))).is_ok());
        let err = validate_handles(&record(ok(4), ok(48))).unwrap_err();
        let text = format!("{err}");
        assert!(
            text.contains("not a multiple of the 32"),
            "the fault must name the alignment the pointee asked for: {text}"
        );
    }

    /// A HAND-WRITTEN RECORD CARRIES NO TABLES AND IS NOT VALIDATED, which is
    /// the same gap `guard_count` has. Stated as a test so the day the last one
    /// is converted, this is what changes.
    #[test]
    fn a_record_without_tables_is_skipped_rather_than_rejected() {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Bare {
            a: Handle,
        }
        unsafe impl KernelArgs for Bare {
            const KERNEL: KernelId = KernelId(0);
        }
        let mut bad = ok(4);
        bad.size = 99;
        assert!(validate_handles(&Bare { a: bad }).is_ok());
    }
}
