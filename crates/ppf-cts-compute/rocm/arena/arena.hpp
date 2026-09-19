// File: arena.hpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE ARENA ALLOCATOR FOR THE CUDA TARGET, in the namespace `compute::arena`.
//
// Handing out and reclaiming device memory is one of the five verbs this crate
// exists for, so the declaration lives here with the `arena.cu` that answers
// it. The namespace is named for the thing rather than for a compiler, because
// every target answers the same vocabulary: the Metal backend answers it from
// `metal/arena.hpp` and the host target from its own base table.
//
// What makes this one nvcc's is the two `__constant__` members below. `g_bases`
// is the base table a kernel resolves through and `resolve` is the resolution
// itself, so a compiler that does not know the CUDA address spaces cannot read
// this file at all.
//
// An allocation is an `ArenaHandle` (`arena_handle.hpp`, in the neutral kernel
// tree the build puts on the include path): an arena id and a byte offset, not
// a pointer. Every generated argument record is built out of that handle and
// every generated entry point resolves one, so the vocabulary here is the one
// the whole seam already speaks, whichever target is running.

#ifndef ARENA_HPP
#define ARENA_HPP

#include "arena_handle.hpp"

#include <cassert>
#include <cstddef>
#include <string>

// WHY THIS IS A SEPARATE FILE FROM cuda/arena/arena.hpp RATHER THAN A SHARED
// ONE. An arena is ALLOCATION, which belongs inside a backend library, and
// each backend already owns its own: the CUDA one beside this, and
// metal/arena.hpp. The declarations below are nearly identical across the three
// because the (arena, offset) data model is the same everywhere; what differs is
// the definitions, which live in each backend's own translation unit. Sharing
// the header would put one backend's file in another's compilation, which is
// exactly what that separation exists to prevent.
namespace compute::arena {

using Handle = ::ArenaHandle;

constexpr unsigned kMaxArenas = 29;
constexpr unsigned long long kMaxArenaBytes = 4ull << 30;

struct Config {
    unsigned long long first_reserve = 64ull << 20;
    unsigned long long arena_cap = kMaxArenaBytes;
    unsigned max_arenas = kMaxArenas;
};

struct Allocator;

Allocator *create(const Config &config = Config{});
void destroy(Allocator *allocator);
Allocator *active();

bool alloc(Allocator *allocator, size_t count, size_t elem_size, size_t align,
           Handle *out, std::string *error);
bool grow(Allocator *allocator, Handle *handle, size_t new_count,
          size_t elem_size, size_t align, std::string *error);
bool free(Allocator *allocator, Handle *handle, std::string *error);

bool write(Allocator *allocator, const Handle &handle, const void *source,
           size_t bytes, std::string *error);
bool read(Allocator *allocator, const Handle &handle, void *destination,
          size_t bytes, std::string *error);

Handle alloc_or_exit(size_t count, size_t elem_size, size_t align);
void free_or_exit(Handle *handle);

void *host_resolve(Allocator *allocator, const Handle &handle,
                   std::string *error);

// The BYTE capacity of the block a handle names.
//
// A handle's `size` and `allocated` are ELEMENT counts, so neither is a byte
// bound, and a caller that needs one (a windowed transfer, a fill, a fresh
// allocation's poisoning) has to ask the allocator, which is the only thing
// that knows the element size the block was made with. Returns false and sets
// `error` for a handle naming no live block, so an invented handle is refused
// rather than measured.
bool block_bytes(Allocator *allocator, const Handle &handle,
                 unsigned long long *out, std::string *error);
unsigned arena_count(const Allocator *allocator);
unsigned long long bytes_used(const Allocator *allocator);
unsigned long long bytes_reserved(const Allocator *allocator);

// THE BINDING TABLE LIVES IN THE CONSTANT BANK, AND THAT IS A REGISTER
// DECISION RATHER THAN A BANDWIDTH ONE.
//
// A `__device__` global is mutable storage the compiler must assume can change,
// so it cannot rematerialize an address it has already resolved: it keeps every
// resolved pointer LIVE for the whole body. A generated entry resolves one
// pointer per buffer field, and the SpMV takes seventeen, which is how
// `operator_apply_dynamic_folded_entry` came to need 71 registers against the
// reference kernel's 40 for the same arithmetic.
//
// `__constant__` cannot change while a kernel runs, so the compiler is free to
// recompute an address from the handle instead of holding it, and the load is a
// broadcast out of the constant cache rather than a global one. Measured on
// `drape` `[L40S, 2026-09-04]`: 71 registers to 56 with NO spill, and
// per launch 27.39 us to 26.98 on the SpMV, 6.59 to 6.23 on the row update and
// 3.15 to 2.70 on the alpha fold, the small kernels gaining most because they
// had the least work to hide the loads behind.
//
// CAPPING REGISTERS DOES NOT SUBSTITUTE FOR THIS, which was measured first:
// `-maxrregcount=48` at the device link reaches 48 registers and 80 bytes of
// SPILL, and the SpMV gets slightly SLOWER (27.78 us). The occupancy it buys is
// paid straight back in local traffic.
//
// IT IS DEVICE-ONLY, WHICH IS WHAT MAKES IT SAFE. A namespace-scope
// `__constant__` that a HOST reader also names is a silent wrong-answer trap,
// correct on the device and zero on the host, and that applies to any symbol
// BOTH sides read. These two are read on the device only: `ArenaPtr::get` in
// this backend's vector header reads them under a device-compile guard and
// reads `g_host_bases` and `g_host_arena_count` on the host branch. Do not
// give either of these a host reader.
//
// 29 arenas of one pointer each is 232 bytes, against the bank's 64 KB.
extern __constant__ unsigned char *g_bases[kMaxArenas];
extern __constant__ unsigned g_arena_count;

template <class T> __device__ T *resolve(const Handle &handle) {
    // TWO OF THE FOUR CHECKS A RESOLUTION OWES ARE THE HOST'S, and the two
    // that remain need state only the dispatch has.
    //
    // `size <= allocated` and `off % alignof(T) == 0` are properties of the
    // HANDLE, fixed before the launch and identical in every thread of every
    // block, so testing them here computed one verdict once per thread: a
    // generated entry resolves every buffer it takes, fourteen for the PCG
    // folds, which put about 56 assert branches in front of a body that folds
    // 64 lanes. `EncoderExt::validate_handles` checks both once per
    // dispatch, reading the offsets and the pointee alignments the generator
    // emits beside each record, and raises a `Fault` naming the kernel and the
    // field instead of trapping one thread with no context.
    //
    // THE ARENA PAIR IS CHECKED IN A DIFFERENT PLACE FROM THE FIRST TWO,
    // because it is a different KIND of fact. `size <= allocated` and the
    // alignment are properties of the HANDLE, fixed the moment it is made, so
    // they belong wherever they are cheapest to check. `arena < count` is a
    // property of the DEVICE at the instant of dispatch: an arena can be added
    // between a region being recorded and being replayed, so it cannot be
    // checked when the record is filled.
    //
    // `backend.cu`'s `run_items` checks it at DISPATCH, once per handle per
    // dispatch, against the same live count the launcher writes into
    // `seam_arena_count`, and names the kernel and the arena when it fails.
    // That is the replay-time check this fact needs, and `run_items` is where
    // it already lives.
    //
    // WHAT THE DEVICE-SIDE CHECK COST: 1,460 ms of 22,018 on `drape`, 6.6
    // percent of GPU time, and about 8.5 registers of mean pressure across 284
    // kernels, which on a memory-latency-bound solver is occupancy. The
    // check did not become cheaper by being written once per dispatch; it
    // stopped being written once per handle per THREAD, which is where the
    // whole of that went.
    //
    // DO NOT PUT THEM BACK HERE. Restoring the assert restores the cost, and
    // the invariant is not weaker for being checked on the host: a malformed
    // arena is refused before any thread runs, with the kernel's name.
    return reinterpret_cast<T *>(g_bases[handle.arena] + handle.off);
}

template <class T> __device__ const T *resolve_const(const Handle &handle) {
    return resolve<T>(handle);
}

} // namespace compute::arena

#endif
