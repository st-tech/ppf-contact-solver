// File: arena.mm
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Implementation of the arena-plus-offset allocator declared in arena.hpp.
//
// Each arena is one MTLBuffer. Space inside an arena is handed out by a bump
// pointer plus a coalescing free list, because both halves are needed:
// schwarz.cu's ensure() does free-then-reallocate against grow_cap today, and
// the CUDA-side allocator that mirrors this design has to support the same
// operations.
//
// Storage mode is MTLResourceStorageModeShared, so the host reads and writes
// an arena through a plain pointer with no blit. Measured aside, recorded so
// nobody builds on it: a Private buffer on this hardware returns a NON-NULL
// and coherent contents pointer (probes/binding_results.txt,
// storage_modes.private_contents_pointer), which Apple documents as NULL.
// Nothing here relies on that.

#include "arena.hpp"
#include "metal_context.hpp"

#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace metal_backend {

// ---------------------------------------------------------------------------
// Context seam.
//
// The entire dependency on the context module is these entry points, declared
// in metal_context.hpp:
//
//   unsigned context_new_buffer(Context *, unsigned long long length,
//                               std::string *err);
//   void    *context_buffer_contents(Context *, unsigned buffer_id);
//   unsigned long long context_buffer_length(Context *, unsigned buffer_id);
//   unsigned long long context_max_buffer_length(Context *);
//   bool     context_fill_buffer(Context *, unsigned buffer_id,
//                                unsigned long long offset,
//                                unsigned long long length,
//                                unsigned char value, std::string *err);
//
// The context CREATES and OWNS the arena buffers, because a buffer id can only
// be resolved back to an MTLBuffer by the module that made it and no header
// outside a .mm may name MTLBuffer. The id handed back here is exactly what
// context_bind_buffer takes, which is what allocator_buffer_id returns. Storage
// is Shared, so context_buffer_contents is a host mapping of the same memory
// and this file needs no Metal type of its own.
// ---------------------------------------------------------------------------

namespace {

using u64 = unsigned long long;

// The first arena reserves this much; each additional arena reserves twice the
// previous one, up to kArenaCapBytes. A scene that needs a lot of memory
// therefore reaches it in a few arenas without a small scene reserving a large
// buffer it never touches, and the arena COUNT stays far below the 29 the
// binding budget allows.
constexpr u64 kFirstArenaReserve = 64ull << 20;

// A new arena is rounded up to this granularity so a single oversized request
// does not create an arena sized to the exact request and immediately full.
constexpr u64 kArenaReserveGranularity = 1ull << 20;

std::string fmt(const char *f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

bool fail(std::string *err, std::string msg) {
    if (err) {
        *err = std::move(msg);
    }
    return false;
}

bool is_power_of_two(u64 v) { return v != 0 && (v & (v - 1)) == 0; }

u64 align_up(u64 v, u64 a) { return (v + a - 1) & ~(a - 1); }

// ---------------------------------------------------------------------------
// PPF_METAL_POISON_ALLOC: what a fresh allocation contains.
//
// Unset, an allocation contains whatever the arena held, which is what the
// backend ships. Set to "0" it is zeroed; set to anything else it is filled
// with kPoisonByte.
//
// WHY THIS BACKEND NEEDS IT. Metal never faults on an uninitialized or
// out-of-bounds read: it returns the bytes that are there, and a fresh
// MTLBuffer's bytes are zero. A buffer the backend forgets to initialize
// therefore reads as zero, the solver produces a plausible trajectory, and
// nothing appears in any log. The defect only becomes visible once an earlier
// block has been freed and the space recycled, at which point it surfaces as a
// failure in whatever unrelated change happened to add that allocation. Nothing
// else in the backend can make this class of defect loud: there is no device
// assert on an uninitialized read, and the value read is a valid float.
//
// A deterministic non-zero fill converts it into a repeatable difference. The
// three-way comparison is the instrument: a run that fails under the pattern,
// passes under "0" and passes unset is a genuine dependence on zeroed memory
// and names the pass that owes a clear, while a run that fails under all three
// is an unrelated defect. A run that SUCCEEDS under the pattern with a
// different answer is the worst case and the one this exists to catch, since
// under zeros it is indistinguishable from a correct run.
//
// 0x5a is chosen to be loud in every interpretation: as a float,
// 0x5a5a5a5a is 1.9e16, far outside any physical quantity in a scene; as an
// index it is 1515870810, past every array; and as a byte pattern it is
// recognizable in a memory dump. Zero would be indistinguishable from a fresh
// buffer, and 0xFF makes a float NaN, which propagates and hides where it
// entered.
//
// The fill covers every block allocator_alloc hands out, which includes the
// TAIL of an allocator_grow (grow allocates through allocator_alloc and then
// copies the old bytes over the head, so the new span keeps the pattern). It
// does not cover the one dedicated reservation, which is the diagnostic
// channel; diag_reset writes that before every dispatch.
//
// Cost is one memset per allocation, proportional to the bytes allocated, paid
// only when the variable is set. The Metal fixture suite is run both ways, by
// the `fixtures` and `fixtures-poisoned` targets in this directory's Makefile.
// ---------------------------------------------------------------------------

constexpr unsigned char kPoisonByte = 0x5a;

enum class PoisonMode { Off, Zero, Pattern };

PoisonMode poison_mode() {
    // Read once. An EMPTY value counts as Off, because `VAR= command` is the
    // shell's idiom for not setting it, and getenv returns a non-null pointer
    // to an empty string for that, which a bare null check would take as on.
    static const PoisonMode mode = [] {
        const char *env = std::getenv("PPF_METAL_POISON_ALLOC");
        if (!env || env[0] == '\0') {
            return PoisonMode::Off;
        }
        return env[0] == '0' ? PoisonMode::Zero : PoisonMode::Pattern;
    }();
    return mode;
}

// Key into the live-block table. An (arena, offset) pair is unique because a
// block occupies its offset exclusively until it is freed.
u64 block_key(unsigned arena, u64 off) {
    return (static_cast<u64>(arena) << 32) | off;
}

struct Arena {
    unsigned char *host = nullptr;    // the context's host mapping, cached
    unsigned buffer_id = kInvalidBufferId;
    u64 capacity = 0;
    u64 bump = 0;                     // first byte never yet handed out
    std::map<u64, u64> free_spans;    // offset -> bytes, disjoint, coalesced,
                                      // all strictly below bump
};

struct Block {
    unsigned arena = 0;
    u64 off = 0;
    u64 bytes = 0;
    size_t elem_size = 0;
    size_t align = 0;
};

// First fit inside one arena: the free list first, then the bump. Returns the
// byte offset of the block, which is a multiple of 'align' by construction.
bool alloc_in_arena(Arena &ar, u64 bytes, u64 align, u64 *out_off) {
    for (auto it = ar.free_spans.begin(); it != ar.free_spans.end(); ++it) {
        const u64 span_off = it->first;
        const u64 span_bytes = it->second;
        const u64 aligned = align_up(span_off, align);
        if (aligned + bytes <= span_off + span_bytes) {
            const u64 tail_off = aligned + bytes;
            const u64 head_bytes = aligned - span_off;
            const u64 tail_bytes = span_off + span_bytes - tail_off;
            ar.free_spans.erase(it);
            if (head_bytes) {
                ar.free_spans[span_off] = head_bytes;
            }
            if (tail_bytes) {
                ar.free_spans[tail_off] = tail_bytes;
            }
            *out_off = aligned;
            return true;
        }
    }
    const u64 aligned = align_up(ar.bump, align);
    if (aligned + bytes <= ar.capacity) {
        if (aligned > ar.bump) {
            ar.free_spans[ar.bump] = aligned - ar.bump;
        }
        ar.bump = aligned + bytes;
        *out_off = aligned;
        return true;
    }
    return false;
}

// Returns the block to the free list, coalescing with the neighbor on each
// side, and retracts the bump when the result reaches it.
void free_in_arena(Arena &ar, u64 off, u64 bytes) {
    u64 begin = off;
    u64 end = off + bytes;
    assert(end <= ar.bump && "freed block lies past the arena's bump pointer");

    auto next = ar.free_spans.lower_bound(off);
    if (next != ar.free_spans.end() && next->first == end) {
        end = next->first + next->second;
        next = ar.free_spans.erase(next);
    }
    if (next != ar.free_spans.begin()) {
        auto prev = std::prev(next);
        assert(prev->first + prev->second <= begin &&
               "free list overlaps the block being freed (double free)");
        if (prev->first + prev->second == begin) {
            begin = prev->first;
            ar.free_spans.erase(prev);
        }
    }
    if (end == ar.bump) {
        ar.bump = begin;
    } else {
        ar.free_spans[begin] = end - begin;
    }
}

} // namespace

struct Allocator {
    Context *ctx = nullptr;
    std::vector<Arena> arenas;
    std::map<u64, Block> live;
    u64 bytes_used = 0;
    u64 next_reserve = kFirstArenaReserve;
    // The one dedicated (non-arena) reservation, see alloc_reserve_dedicated.
    // Empty until it is taken; the name is kept so a second request can say
    // which channel already holds the slot.
    std::string dedicated_name;
    unsigned dedicated_buffer_id = 0;
    u64 dedicated_bytes = 0;
};

namespace {

// Opens one more arena, large enough to hold 'need_bytes' at offset 0.
bool open_arena(Allocator *a, u64 need_bytes, std::string *err) {
    if (a->arenas.size() >= kMaxArenas) {
        return fail(err,
                    fmt("arena budget exhausted: %zu arenas are already open "
                        "and the limit is %u. Metal exposes 31 buffer binding "
                        "slots (indices 0 to 30, enforced by the shader "
                        "compiler); the handle block and the diagnostic buffer "
                        "take one each, leaving %u for arenas.",
                        a->arenas.size(), kMaxArenas, kMaxArenas));
    }
    if (need_bytes > kArenaCapBytes) {
        return fail(err,
                    fmt("single allocation of %llu bytes exceeds the %llu byte "
                        "arena cap. The cap is the span a handle's 32-bit "
                        "'off' field can address, not a device limit.",
                        need_bytes, kArenaCapBytes));
    }

    u64 reserve = a->next_reserve;
    if (reserve < need_bytes) {
        reserve = align_up(need_bytes, kArenaReserveGranularity);
    }
    // Two independent ceilings, and the arena must respect the LOWER of them.
    // kArenaCapBytes is what the handle's 32-bit offset can address (4 GiB).
    // maxBufferLength is what the device will hand out, and on Apple silicon it
    // scales with installed RAM, so it can be either side of 4 GiB: 28.08 GB on
    // the machine this was developed on, but roughly 3.4 GB on an 8 GB Mac.
    // Clamping to kArenaCapBytes alone would make the doubling policy request
    // 4 GiB on such a machine, get refused by the device, and fail the whole
    // allocation with 29 binding slots and plenty of memory still free.
    const u64 device_cap = context_max_buffer_length(a->ctx);
    const u64 cap = device_cap && device_cap < kArenaCapBytes ? device_cap
                                                             : kArenaCapBytes;
    if (reserve > cap) {
        reserve = cap;
    }
    if (need_bytes > cap) {
        // A single block larger than either ceiling cannot be placed in any
        // arena, so say which ceiling stopped it rather than reporting a
        // generic allocation failure.
        if (err) {
            *err = fmt("allocation of %llu bytes exceeds the %llu byte ceiling "
                       "for one arena (%s)",
                       (unsigned long long)need_bytes, (unsigned long long)cap,
                       cap == kArenaCapBytes
                           ? "a 32-bit arena offset addresses 4 GiB"
                           : "the device's maxBufferLength");
        }
        return false;
    }
    assert(reserve >= need_bytes);

    std::string buf_err;
    const unsigned buffer_id = context_new_buffer(a->ctx, reserve, &buf_err);
    if (buffer_id == 0) {
        return fail(err, fmt("could not create arena %zu of %llu bytes: %s",
                             a->arenas.size(), reserve, buf_err.c_str()));
    }
    void *contents = context_buffer_contents(a->ctx, buffer_id);
    if (!contents) {
        return fail(err, fmt("arena buffer of %llu bytes returned a null "
                             "contents pointer",
                             reserve));
    }

    Arena ar;
    ar.host = static_cast<unsigned char *>(contents);
    ar.buffer_id = buffer_id;
    ar.capacity = reserve;
    a->arenas.push_back(std::move(ar));

    a->next_reserve = (a->next_reserve >= kArenaCapBytes / 2)
                          ? kArenaCapBytes
                          : a->next_reserve * 2;
    return true;
}

// Validates the request and reduces it to a byte count. Every rejection here
// is a caller error that Metal itself would not report: a misaligned offset is
// silently floored to a multiple of 4 and returns the data at the floored
// address with status Completed and no diagnostic.
bool check_request(size_t count, size_t elem_size, size_t align, u64 *out_bytes,
                   std::string *err) {
    if (elem_size == 0) {
        return fail(err, "elem_size must be non-zero");
    }
    if (!is_power_of_two(align)) {
        return fail(err, fmt("align %zu is not a power of two", align));
    }
    if (align < 4) {
        return fail(err,
                    fmt("align %zu is below the minimum of 4. Metal floors a "
                        "setBuffer offset to a multiple of 4 without any "
                        "diagnostic, so a sub-4 alignment is unrepresentable "
                        "rather than merely inefficient.",
                        align));
    }
    // Element i sits at off + i * elem_size. When the element is at least as
    // large as the alignment, 'align' must divide 'elem_size' or element 1
    // already lands off the grid. A smaller element (a uchar array) needs at
    // most elem_size alignment of its own, which any byte offset inside an
    // aligned block satisfies.
    if (elem_size >= align ? (elem_size % align != 0)
                           : (align % elem_size != 0)) {
        return fail(err, fmt("elem_size %zu and align %zu are incompatible: "
                             "element i sits at off + i * %zu, which would "
                             "leave elements after the first off the %zu byte "
                             "grid",
                             elem_size, align, elem_size, align));
    }
    if (count > UINT32_MAX) {
        return fail(err, fmt("element count %zu does not fit the handle's "
                             "32-bit 'size' and 'allocated' counters",
                             count));
    }
    if (count != 0 &&
        static_cast<u64>(count) > UINT64_MAX / static_cast<u64>(elem_size)) {
        return fail(err, fmt("allocation of %zu x %zu bytes overflows",
                             count, elem_size));
    }
    *out_bytes = static_cast<u64>(count) * static_cast<u64>(elem_size);
    return true;
}

// Places 'bytes' in the first arena with room, opening a new arena when none
// has room. 'bytes' must be non-zero.
//
// First fit, scanning arenas in order, so freed space is reused before new
// space is reserved. The scan is linear in the number of free spans, which is
// the right trade here: the solver allocates on the order of a hundred
// containers at scene assembly and grows a handful of them per step.
bool place(Allocator *a, u64 bytes, u64 align, unsigned *out_arena, u64 *out_off,
           std::string *err) {
    assert(bytes != 0);
    for (size_t i = 0; i < a->arenas.size(); ++i) {
        u64 off = 0;
        if (alloc_in_arena(a->arenas[i], bytes, align, &off)) {
            *out_arena = static_cast<unsigned>(i);
            *out_off = off;
            return true;
        }
    }
    if (!open_arena(a, bytes, err)) {
        return false;
    }
    const size_t last = a->arenas.size() - 1;
    u64 off = 0;
    if (!alloc_in_arena(a->arenas[last], bytes, align, &off)) {
        return fail(err, fmt("internal error: %llu bytes at alignment %llu do "
                             "not fit a freshly opened arena of %llu bytes",
                             bytes, align, a->arenas[last].capacity));
    }
    *out_arena = static_cast<unsigned>(last);
    *out_off = off;
    return true;
}

// Locates the live block a handle names, or reports why it does not name one.
Block *find_block(Allocator *a, const ArenaHandle &h, std::string *err) {
    if (h.arena >= a->arenas.size()) {
        fail(err, fmt("handle names arena %u but only %zu are open", h.arena,
                      a->arenas.size()));
        return nullptr;
    }
    auto it = a->live.find(block_key(h.arena, h.off));
    if (it == a->live.end()) {
        fail(err, fmt("handle (arena %u, off %u) is not a live allocation",
                      h.arena, h.off));
        return nullptr;
    }
    return &it->second;
}

} // namespace

Allocator *allocator_create(Context *ctx) {
    if (!ctx) {
        return nullptr;
    }
    Allocator *a = new Allocator();
    a->ctx = ctx;
    return a;
}

void allocator_destroy(Allocator *a) {
    if (!a) {
        return;
    }
    // The arena buffers belong to the Context, which releases them in
    // context_destroy. There is no per-buffer release here, which is why the
    // header requires the Context to outlive the Allocator.
    delete a;
}

bool alloc_reserve_dedicated(Allocator *a, const char *name, size_t bytes,
                             unsigned *out_buffer_id,
                             unsigned *out_binding_index, void **out_host_ptr,
                             std::string *err) {
    if (!a) {
        return fail(err, "allocator is null");
    }
    if (!name || !*name) {
        return fail(err, "a dedicated reservation must be named, so a later "
                         "request can report which channel holds the slot");
    }
    if (!out_buffer_id || !out_binding_index || !out_host_ptr) {
        return fail(err, "alloc_reserve_dedicated: null output argument");
    }
    if (bytes == 0) {
        return fail(err, fmt("dedicated reservation '%s' asked for 0 bytes", name));
    }
    if (!a->dedicated_name.empty()) {
        return fail(err,
                    fmt("dedicated reservation '%s' refused: binding index %u is "
                        "the only slot outside the arena pool and '%s' already "
                        "holds it. Metal exposes 31 slots and the shader "
                        "compiler enforces the limit, so a second dedicated "
                        "buffer would have to displace an arena.",
                        name, kDedicatedBindingIndex,
                        a->dedicated_name.c_str()));
    }

    std::string buf_err;
    const unsigned buffer_id =
        context_new_buffer(a->ctx, static_cast<u64>(bytes), &buf_err);
    if (buffer_id == 0) {
        return fail(err, fmt("dedicated reservation '%s' of %zu bytes failed: %s",
                             name, bytes, buf_err.c_str()));
    }
    void *contents = context_buffer_contents(a->ctx, buffer_id);
    if (!contents) {
        return fail(err, fmt("dedicated reservation '%s' returned a null "
                             "contents pointer",
                             name));
    }

    a->dedicated_name = name;
    a->dedicated_buffer_id = buffer_id;
    a->dedicated_bytes = static_cast<u64>(bytes);

    *out_buffer_id = buffer_id;
    *out_binding_index = kDedicatedBindingIndex;
    *out_host_ptr = contents;
    return true;
}

bool allocator_alloc(Allocator *a, size_t count, size_t elem_size, size_t align,
                     ArenaHandle *out, std::string *err) {
    if (!a) {
        return fail(err, "allocator is null");
    }
    if (!out) {
        return fail(err, "output handle is null");
    }
    *out = ArenaHandle{0, 0, 0, 0};

    u64 bytes = 0;
    if (!check_request(count, elem_size, align, &bytes, err)) {
        return false;
    }

    if (bytes == 0) {
        // A zero-length handle still has to name a bound arena, so that a
        // device-side bounds check on it fails loudly rather than indexing an
        // arena slot that was never bound.
        if (a->arenas.empty() && !open_arena(a, 0, err)) {
            return false;
        }
        return true;
    }

    unsigned arena = 0;
    u64 off = 0;
    if (!place(a, bytes, static_cast<u64>(align), &arena, &off, err)) {
        return false;
    }

    // Both are invariants of the design rather than caller errors, and both are
    // silent wrong answers on Metal if they ever fail: a misaligned offset is
    // floored, and an offset past 32 bits wraps inside the handle.
    assert(off % static_cast<u64>(align) == 0);
    assert(off <= UINT32_MAX);
    if (off > UINT32_MAX) {
        free_in_arena(a->arenas[arena], off, bytes);
        return fail(err, fmt("offset %llu in arena %u exceeds the 32-bit span a "
                             "handle can address",
                             off, arena));
    }

    Block blk;
    blk.arena = arena;
    blk.off = off;
    blk.bytes = bytes;
    blk.elem_size = elem_size;
    blk.align = align;
    a->live[block_key(arena, off)] = blk;
    a->bytes_used += bytes;

    // The debug fill, off unless PPF_METAL_POISON_ALLOC is set. See the block
    // beside kPoisonByte above for what it is and why this backend needs one.
    // Host-side, through the arena's Shared mapping, because this runs at
    // allocation time with no dispatch of any kind in flight.
    const PoisonMode poison = poison_mode();
    if (poison != PoisonMode::Off) {
        std::memset(a->arenas[arena].host + off,
                    poison == PoisonMode::Zero ? 0x00 : kPoisonByte,
                    static_cast<size_t>(bytes));
    }

    out->arena = arena;
    out->off = static_cast<unsigned>(off);
    out->size = static_cast<unsigned>(count);
    out->allocated = static_cast<unsigned>(count);
    return true;
}

bool allocator_grow(Allocator *a, ArenaHandle *h, size_t new_count,
                    size_t elem_size, size_t align, std::string *err) {
    if (!a) {
        return fail(err, "allocator is null");
    }
    if (!h) {
        return fail(err, "handle is null");
    }
    u64 new_bytes = 0;
    if (!check_request(new_count, elem_size, align, &new_bytes, err)) {
        return false;
    }
    if (h->size > h->allocated) {
        return fail(err, fmt("handle reports size %u above its capacity %u",
                             h->size, h->allocated));
    }
    if (new_count < h->allocated) {
        return fail(err, fmt("grow to %zu elements would shrink a %u element "
                             "allocation; shrinking is not supported",
                             new_count, h->allocated));
    }
    if (new_count == h->allocated) {
        return true;
    }

    // A zero-capacity handle owns no block, so growing it is a plain
    // allocation. Its logical size is 0 by construction, so nothing is copied.
    if (h->allocated == 0) {
        ArenaHandle grown{};
        if (!allocator_alloc(a, new_count, elem_size, align, &grown, err)) {
            return false;
        }
        grown.size = h->size;
        *h = grown;
        return true;
    }

    Block *old = find_block(a, *h, err);
    if (!old) {
        return false;
    }
    if (old->elem_size != elem_size || old->align != align) {
        return fail(err, fmt("grow uses elem_size %zu align %zu but the block "
                             "was allocated with elem_size %zu align %zu",
                             elem_size, align, old->elem_size, old->align));
    }
    if (old->bytes != static_cast<u64>(h->allocated) * elem_size) {
        return fail(err, fmt("handle reports capacity %u x %zu bytes but the "
                             "allocator holds %llu bytes for it",
                             h->allocated, elem_size, old->bytes));
    }

    const unsigned old_arena = old->arena;
    const u64 old_off = old->off;
    const u64 old_bytes = old->bytes;

    // Allocate before freeing, so a failure leaves the original block intact.
    // Peak usage is therefore old plus new for the duration of the copy.
    ArenaHandle grown{};
    if (!allocator_alloc(a, new_count, elem_size, align, &grown, err)) {
        return false;
    }
    std::memcpy(a->arenas[grown.arena].host + grown.off,
                a->arenas[old_arena].host + old_off,
                static_cast<size_t>(old_bytes));

    a->live.erase(block_key(old_arena, old_off));
    free_in_arena(a->arenas[old_arena], old_off, old_bytes);
    a->bytes_used -= old_bytes;

    grown.size = h->size;
    *h = grown;
    return true;
}

bool allocator_free(Allocator *a, ArenaHandle *h, std::string *err) {
    if (!a) {
        return fail(err, "allocator is null");
    }
    if (!h) {
        return fail(err, "handle is null");
    }
    // A zero-capacity handle never reserved anything, so there is nothing to
    // return. A real block always has a capacity of at least one element, so
    // this cannot swallow a live block.
    if (h->allocated == 0) {
        *h = ArenaHandle{0, 0, 0, 0};
        return true;
    }

    Block *blk = find_block(a, *h, err);
    if (!blk) {
        return false;
    }
    const unsigned arena = blk->arena;
    const u64 off = blk->off;
    const u64 bytes = blk->bytes;
    a->live.erase(block_key(arena, off));
    free_in_arena(a->arenas[arena], off, bytes);
    a->bytes_used -= bytes;

    *h = ArenaHandle{0, 0, 0, 0};
    return true;
}

bool allocator_write(Allocator *a, const ArenaHandle &h, const void *src,
                     size_t bytes, std::string *err) {
    if (!a) {
        return fail(err, "allocator is null");
    }
    if (bytes == 0) {
        return true;
    }
    if (!src) {
        return fail(err, "source pointer is null");
    }
    Block *blk = find_block(a, h, err);
    if (!blk) {
        return false;
    }
    // Metal checks nothing here: a write past the end of a binding is silently
    // dropped and a write 1 GiB past the end completes with status Completed
    // and no error, so this bound is the only one there is.
    if (static_cast<u64>(bytes) > blk->bytes) {
        return fail(err, fmt("write of %zu bytes exceeds the %llu byte capacity "
                             "of the block at (arena %u, off %u)",
                             bytes, blk->bytes, h.arena, h.off));
    }
    std::memcpy(a->arenas[blk->arena].host + blk->off, src, bytes);
    return true;
}

bool allocator_read(Allocator *a, const ArenaHandle &h, void *dst, size_t bytes,
                    std::string *err) {
    if (!a) {
        return fail(err, "allocator is null");
    }
    if (bytes == 0) {
        return true;
    }
    if (!dst) {
        return fail(err, "destination pointer is null");
    }
    Block *blk = find_block(a, h, err);
    if (!blk) {
        return false;
    }
    if (static_cast<u64>(bytes) > blk->bytes) {
        return fail(err, fmt("read of %zu bytes exceeds the %llu byte capacity "
                             "of the block at (arena %u, off %u)",
                             bytes, blk->bytes, h.arena, h.off));
    }
    std::memcpy(dst, a->arenas[blk->arena].host + blk->off, bytes);
    return true;
}

bool allocator_read_at(Allocator *a, const ArenaHandle &h, size_t offset,
                       void *dst, size_t bytes, std::string *err) {
    if (!a) {
        return fail(err, "allocator is null");
    }
    if (bytes == 0) {
        return true;
    }
    if (!dst) {
        return fail(err, "destination pointer is null");
    }
    Block *blk = find_block(a, h, err);
    if (!blk) {
        return false;
    }
    // Both ends, and the subtraction rather than offset + bytes, so a length
    // that would wrap is refused instead of passing the test.
    if (static_cast<u64>(offset) > blk->bytes ||
        static_cast<u64>(bytes) > blk->bytes - static_cast<u64>(offset)) {
        return fail(err, fmt("read of %zu bytes at offset %zu lies outside the "
                             "%llu byte block at (arena %u, off %u)",
                             bytes, offset, blk->bytes, h.arena, h.off));
    }
    std::memcpy(dst, a->arenas[blk->arena].host + blk->off + offset, bytes);
    return true;
}

bool allocator_host_ptr(Allocator *a, const ArenaHandle &h, size_t offset,
                        size_t bytes, void **out, std::string *err) {
    if (!a || !out) {
        return fail(err, "allocator or output slot is null");
    }
    *out = nullptr;
    if (bytes == 0) {
        return true;
    }
    Block *blk = find_block(a, h, err);
    if (!blk) {
        return false;
    }
    // Both ends, and the subtraction rather than offset + bytes, so a length
    // that would wrap is refused instead of passing the test.
    if (static_cast<u64>(offset) > blk->bytes ||
        static_cast<u64>(bytes) > blk->bytes - static_cast<u64>(offset)) {
        return fail(err, fmt("a host view of %zu bytes at offset %zu lies "
                             "outside the %llu byte block at (arena %u, off "
                             "%u)",
                             bytes, offset, blk->bytes, h.arena, h.off));
    }
    *out = a->arenas[blk->arena].host + blk->off + offset;
    return true;
}

bool allocator_copy_at(Allocator *a, const ArenaHandle &dst, size_t dst_offset,
                       const ArenaHandle &src, size_t src_offset, size_t bytes,
                       std::string *err) {
    if (!a) {
        return fail(err, "allocator is null");
    }
    if (bytes == 0) {
        return true;
    }
    Block *dst_blk = find_block(a, dst, err);
    if (!dst_blk) {
        return false;
    }
    Block *src_blk = find_block(a, src, err);
    if (!src_blk) {
        return false;
    }
    // Both ends of both windows, and the subtraction rather than
    // offset + bytes, so a length that would wrap is refused.
    if (static_cast<u64>(dst_offset) > dst_blk->bytes ||
        static_cast<u64>(bytes) > dst_blk->bytes - static_cast<u64>(dst_offset)) {
        return fail(err, fmt("copy of %zu bytes to offset %zu lies outside the "
                             "%llu byte block at (arena %u, off %u)",
                             bytes, dst_offset, dst_blk->bytes, dst.arena,
                             dst.off));
    }
    if (static_cast<u64>(src_offset) > src_blk->bytes ||
        static_cast<u64>(bytes) > src_blk->bytes - static_cast<u64>(src_offset)) {
        return fail(err, fmt("copy of %zu bytes from offset %zu lies outside "
                             "the %llu byte block at (arena %u, off %u)",
                             bytes, src_offset, src_blk->bytes, src.arena,
                             src.off));
    }
    std::memcpy(a->arenas[dst_blk->arena].host + dst_blk->off + dst_offset,
                a->arenas[src_blk->arena].host + src_blk->off + src_offset,
                bytes);
    return true;
}

bool allocator_block_bytes(Allocator *a, const ArenaHandle &h,
                           unsigned long long *out, std::string *err) {
    if (!a || !out) {
        return fail(err, "allocator or output slot is null");
    }
    Block *blk = find_block(a, h, err);
    if (!blk) {
        return false;
    }
    *out = blk->bytes;
    return true;
}

bool allocator_write_at(Allocator *a, const ArenaHandle &h, size_t offset,
                        const void *src, size_t bytes, std::string *err) {
    if (!a) {
        return fail(err, "allocator is null");
    }
    if (bytes == 0) {
        return true;
    }
    if (!src) {
        return fail(err, "source pointer is null");
    }
    Block *blk = find_block(a, h, err);
    if (!blk) {
        return false;
    }
    // Both ends, and the subtraction rather than offset + bytes, so a length
    // that would wrap is refused instead of passing the test.
    if (static_cast<u64>(offset) > blk->bytes ||
        static_cast<u64>(bytes) > blk->bytes - static_cast<u64>(offset)) {
        return fail(err, fmt("write of %zu bytes at offset %zu lies outside "
                             "the %llu byte block at (arena %u, off %u)",
                             bytes, offset, blk->bytes, h.arena, h.off));
    }
    std::memcpy(a->arenas[blk->arena].host + blk->off + offset, src, bytes);
    return true;
}

bool allocator_fill(Allocator *a, const ArenaHandle &h, unsigned char value,
                    size_t bytes, std::string *err) {
    if (!a) {
        return fail(err, "allocator is null");
    }
    if (bytes == 0) {
        return true;
    }
    Block *blk = find_block(a, h, err);
    if (!blk) {
        return false;
    }
    if (static_cast<u64>(bytes) > blk->bytes) {
        return fail(err, fmt("fill of %zu bytes exceeds the %llu byte capacity "
                             "of the block at (arena %u, off %u)",
                             bytes, blk->bytes, h.arena, h.off));
    }
    return context_fill_buffer(a->ctx, a->arenas[blk->arena].buffer_id,
                               blk->off, static_cast<u64>(bytes), value, err);
}

unsigned allocator_buffer_id(Allocator *a, unsigned arena) {
    assert(a && arena < a->arenas.size() &&
           "allocator_buffer_id called with an out-of-range arena index");
    if (!a || arena >= a->arenas.size()) {
        return kInvalidBufferId;
    }
    return a->arenas[arena].buffer_id;
}

unsigned allocator_arena_count(Allocator *a) {
    return a ? static_cast<unsigned>(a->arenas.size()) : 0u;
}

void allocator_bind_all(Allocator *a, Context *ctx, unsigned command) {
    assert(a && ctx);
    assert(!a->arenas.empty());
    for (unsigned slot = 0; slot < kMaxArenas; ++slot) {
        const unsigned arena =
            slot < static_cast<unsigned>(a->arenas.size()) ? slot : 0u;
        context_bind_buffer(ctx, command, slot,
                            allocator_buffer_id(a, arena), 0);
    }
}

unsigned long long allocator_bytes_used(Allocator *a) {
    return a ? a->bytes_used : 0ull;
}

unsigned long long allocator_bytes_reserved(Allocator *a) {
    if (!a) {
        return 0ull;
    }
    // The dedicated buffer counts too: it is device memory this allocator
    // asked the context for, so leaving it out would under-report what the
    // backend holds.
    unsigned long long total = a->dedicated_bytes;
    for (const Arena &ar : a->arenas) {
        total += ar.capacity;
    }
    return total;
}

} // namespace metal_backend
