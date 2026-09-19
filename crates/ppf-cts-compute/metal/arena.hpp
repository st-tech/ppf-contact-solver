// File: arena.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The arena-plus-offset allocator for the Metal backend, answering the same
// interface the CUDA target answers with a per-container `cudaMalloc` in
// `cuda/cuda_vec.hpp`.
//
// Why it exists: DataSet carries 84 transitive device pointers (the contact
// assembly kernel capture carries 135, about 60 after maximal flattening) and
// Metal offers 31 buffer binding slots, enforced by the shader compiler
// ([[buffer(31)]] is a compile error). Binding one buffer per container is
// therefore impossible, not merely awkward. Instead every container is packed
// into a small number of large arena buffers and addressed as
// (arena id, byte offset). The handle aggregate travels to the kernel by
// setBytes and the arenas are bound once per encoder.
//
// The model is verified end to end, 19 passed and 0 failed over three
// repeats: handles resolve byte-identically for float, uint, uchar, a 12-byte
// V3 and a 36-byte M9; VecVec's two pointers resolve independently; the
// allocated counter crosses intact; and handles manufactured inside device
// memory by one kernel resolve correctly in another, which is what lets
// DynCSRMat::Row cross a kernel boundary without a raw device address.
//
// This is not a foreign idea imported for Metal. cpp/solver/solver.cu:231-232
// already addresses the PCG value buffer as base + ref_value[k], so that
// buffer's ELEMENTS are offsets rather than pointers. This generalizes the
// same indirection up to the container handles themselves.
//
// THREAD SAFETY: none. The solver drives the allocator from one thread, and a
// lock added here would be untested code claiming a guarantee nothing
// verifies. Call it from one thread.

#ifndef METAL_ARENA_HPP
#define METAL_ARENA_HPP

#include "arena_handle.hpp"

#include <cstddef>
#include <string>

namespace metal_backend {

struct Context;

// A device allocation, addressed as (arena id, byte offset) rather than as
// a pointer. Four fields, mirroring Vec<T> in cpp/vec/vec.hpp, whose THIRD
// field is load-bearing: Vec<T>::alloc sets 'allocated' before the malloc
// and schwarz.cu's ensure() reads it against grow_cap.
using ::ArenaHandle;

// A single arena may not exceed what the handle's 32-bit 'off' field can
// address, which is 4 GiB, even though this device reports a maxBufferLength
// of 30150672384 B (28.08 GB). The PORT owns the smaller bound, so when a
// request does not fit the current arena the allocator opens another one
// rather than growing past the addressable span and handing out offsets that
// silently wrap.
//
// The two limits coincided exactly on the paravirtual device the early
// measurements were taken on (maxBufferLength was itself 4 GB there), which is
// why this distinction was invisible then. It is real on the M4 Pro. Do not
// "simplify" the cap away by deferring to maxBufferLength.
constexpr unsigned long long kArenaCapBytes = 4ull << 30;

// The arena COUNT is bounded by the binding budget, not by memory. There are
// 31 slots (indices 0 to 30); the handle block takes one and the diagnostic
// buffer takes one, leaving 29 for arenas. Measured: 29 arenas plus handle
// plus diagnostic bind at highest index 30 and run correctly, while 30 arenas
// fail to compile.
constexpr unsigned kMaxArenas = 29;

// Returned by allocator_buffer_id for an out-of-range arena index. Never a
// valid binding id.
constexpr unsigned kInvalidBufferId = 0xFFFFFFFFu;

// The binding-slot policy for the whole backend, kept here because this is the
// module that owns the budget the arena count is taken out of. Arena i binds at
// index i (0 to kMaxArenas - 1), the handle block binds at
// kHandleBlockBindingIndex, and the one dedicated reservation below binds at
// kDedicatedBindingIndex, which is the top slot. Arenas grow upward from 0, so
// reserving the top is what keeps an arena from displacing the channel.
constexpr unsigned kHandleBlockBindingIndex = kMaxArenas;    // 29
constexpr unsigned kDedicatedBindingIndex = kMaxArenas + 1;  // 30

struct Allocator;

Allocator *allocator_create(Context *ctx);
void allocator_destroy(Allocator *a);

// Reserves a whole buffer, plus one buffer id and one binding index, for a
// channel that cannot live inside an arena: today only the diagnostic channel
// (diagnostics.hpp), which is bound at a fixed index on every dispatch and read
// by the host after every command buffer. A suballocation could not be bound on
// its own, and the binding index has to come out of the same 31-slot budget the
// arenas draw from, which only this module tracks.
//
// The buffer is host-visible (Shared) and lives until allocator_destroy;
// '*out_host_ptr' is the mapping to read it through. 'name' appears in the
// error messages. There is exactly ONE dedicated slot, so a second call is
// refused loudly rather than displacing an arena.
bool alloc_reserve_dedicated(Allocator *a, const char *name, size_t bytes,
                             unsigned *out_buffer_id,
                             unsigned *out_binding_index, void **out_host_ptr,
                             std::string *err);

// Allocates 'count' elements of 'elem_size' bytes, aligned to 'align'.
//
// ASSERTS NATURAL ALIGNMENT AT ALLOCATION TIME. This is not defensive
// hardening, it is the only thing that catches a misaligned offset at all:
// Metal silently FLOORS a misaligned setBuffer offset to a multiple of 4
// and returns the wrong data with status Completed and no diagnostic.
// Returns false and fills 'err' on any failure.
//
// 'align' is the alignment of the BLOCK's first byte. It must be a power of
// two and at least 4, because a sub-4 offset is exactly what gets floored.
// Element i sits at off + i * elem_size, so 'align' must also divide
// 'elem_size', with one sanctioned exception: an element SMALLER than the
// alignment (a uchar array) needs at most elem_size alignment of its own, so
// any byte offset inside a block that starts aligned satisfies it.
//
// On success 'out' gets size == allocated == count. The allocator never reads
// 'size'; a caller that wants spare capacity allocates the capacity and lowers
// 'size' itself, which is what Vec<T>::alloc's alloc_factor does today.
//
// A count of 0 yields a zero-length handle addressing arena 0 at offset 0.
// It consumes no space, and because it names a real bound arena rather than an
// out-of-range id, a bounds check on it fails loudly instead of indexing an
// unbound slot.
bool allocator_alloc(Allocator *a, size_t count, size_t elem_size,
                     size_t align, ArenaHandle *out, std::string *err);

// Grows an existing allocation, preserving its contents. The block may move to
// a different arena, so any copy of the handle held elsewhere is stale after
// this call. 'new_count' below the current capacity is refused loudly rather
// than silently shrinking; equal to it is a no-op. 'elem_size' and 'align'
// must match the original allocation, which is checked.
bool allocator_grow(Allocator *a, ArenaHandle *h, size_t new_count,
                    size_t elem_size, size_t align, std::string *err);

// Frees the block and zeroes the handle, so a use after free addresses arena 0
// with allocated 0 and every bounds check below rejects it.
bool allocator_free(Allocator *a, ArenaHandle *h, std::string *err);

// Host to device and back. 'h' addresses the destination or source.
//
// Arenas use MTLResourceStorageModeShared, so both directions are a memcpy
// through the buffer's contents pointer with no blit. Ordering is the caller's
// responsibility: a write is visible to the device only if it completes before
// the command buffer that reads it is committed, and a read is meaningful only
// after waitUntilCompleted on the command buffer that produced the data.
bool allocator_write(Allocator *a, const ArenaHandle &h, const void *src,
                     size_t bytes, std::string *err);
bool allocator_read(Allocator *a, const ArenaHandle &h, void *dst,
                    size_t bytes, std::string *err);

// Reads 'bytes' starting 'offset' bytes INTO the block 'h' names, rather than
// at its start. For a block that carries several arrays end to end, which is
// how the LBVH keeps a whole tree under one handle in a 31-slot binding budget.
//
// A handle with 'off' advanced by hand cannot serve here: the allocator finds a
// block by its exact starting offset, so a derived handle names no live block
// and is refused, which is the correct answer to a handle that has been
// invented. This entry point asks the block that really exists for a window
// inside itself, and both ends of that window are checked against its capacity.
bool allocator_read_at(Allocator *a, const ArenaHandle &h, size_t offset,
                       void *dst, size_t bytes, std::string *err);

// The mirror image: writes 'bytes' starting 'offset' bytes INTO the block,
// rather than at its start, on exactly the terms allocator_read_at reads one.
//
// A handle with 'off' advanced by hand cannot serve here either, and for the
// same reason: the allocator finds a block by its exact starting offset, so a
// derived handle names no live block. This asks the block that really exists
// for a window inside itself, and both ends of that window are checked against
// its capacity.
bool allocator_write_at(Allocator *a, const ArenaHandle &h, size_t offset,
                        const void *src, size_t bytes, std::string *err);

// Block to block, both windows checked as `allocator_read_at` and
// `allocator_write_at` check theirs, and no host bounce between them. The
// arenas are host-visible here, so this is one memcpy and crosses no bus.
//
// The two blocks may be the same one, in which case the windows must not
// overlap: this is `memcpy` and not `memmove`, so an overlapping move is the
// caller's to split.
bool allocator_copy_at(Allocator *a, const ArenaHandle &dst, size_t dst_offset,
                       const ArenaHandle &src, size_t src_offset, size_t bytes,
                       std::string *err);

// The host address of a window inside a live block, for a caller that reads or
// writes the block's bytes directly rather than through a copy.
//
// Arenas use MTLResourceStorageModeShared, so an arena's bytes are already
// mapped on the host and this hands back a pointer into that mapping. There is
// no second allocation and no memcpy: the address names the same bytes a
// dispatch reads and writes.
//
// The window is checked at both ends exactly as allocator_read_at checks its
// own, and a handle with 'off' advanced by hand names no live block and is
// refused, so this is not a way to manufacture an address the transfer calls
// would have rejected. 'bytes' of 0 succeeds with '*out' null, which keeps a
// zero-length window from producing an address a caller could build an empty
// span from.
//
// THE ADDRESS IS INVALIDATED BY allocator_grow OR allocator_free ON THIS
// HANDLE AND BY NOTHING ELSE. An allocation, a growth or a free on any other
// handle leaves this block where it is: an arena is created once and never
// resized, a request that does not fit the open arenas opens another one, and
// a free only returns a span to a free list. A caller that re-takes the
// address whenever it grows or releases its own block therefore holds a valid
// one.
//
// ORDERING IS THE CALLER'S, on exactly the terms allocator_write and
// allocator_read state it. This returns an address. It performs no
// synchronization and it does not make one unnecessary: a host access through
// the returned pointer is meaningful only while no command buffer that touches
// the block is in flight.
bool allocator_host_ptr(Allocator *a, const ArenaHandle &h, size_t offset,
                        size_t bytes, void **out, std::string *err);

// The block's BYTE capacity, which a handle does not carry: `size` and
// `allocated` are ELEMENT counts, so a caller that wants to bound a byte window
// has to ask the allocator. A bound taken from an element count refuses every
// window past the element count and admits none past the real end, which is a
// refusal that looks like a bounds check and is not one.
bool allocator_block_bytes(Allocator *a, const ArenaHandle &h,
                           unsigned long long *out, std::string *err);

// Sets the first 'bytes' of the block 'h' addresses to a repeated byte value,
// on the DEVICE, through a blit. Argument order follows memset. 'bytes' beyond
// the block's capacity is refused; zero fills nothing and succeeds.
//
// This is what a kernel that ACCUMULATES into a buffer needs before it runs.
// The device path rather than a memset through the arena's host mapping is
// argued in metal_context.hpp at context_fill_buffer; the short version is that
// it is ordered by the queue and costs no host bandwidth. Ordering against the
// dispatches on either side is still the caller's, as it is for
// allocator_write.
bool allocator_fill(Allocator *a, const ArenaHandle &h, unsigned char value,
                    size_t bytes, std::string *err);

// The Metal buffer id backing an arena, for context_bind_buffer. Returns
// kInvalidBufferId for an out-of-range arena index, and asserts, which traps
// in this project's builds because the production build defines no NDEBUG.
unsigned allocator_buffer_id(Allocator *a, unsigned arena);
unsigned allocator_arena_count(Allocator *a);

// Binds every arena slot required by the fixed 29-arena shader ABI. Unused
// slots alias arena 0, so every declared MSL buffer argument is valid while
// handle bounds remain governed by allocator_arena_count.
void allocator_bind_all(Allocator *a, Context *ctx, unsigned command);

// Total bytes handed out and total reserved, for diagnostics.
unsigned long long allocator_bytes_used(Allocator *a);
unsigned long long allocator_bytes_reserved(Allocator *a);

} // namespace metal_backend

#endif
