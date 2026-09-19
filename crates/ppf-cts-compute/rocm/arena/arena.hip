// File: arena.cu
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "arena/arena.hpp"

#include "../hip_utils.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace compute::arena {

// CONSTANT, NOT DEVICE: see the reasoning at the declarations in `arena.hpp`.
// `hipMemcpyToSymbol` writes a `__constant__` symbol exactly as it writes a
// `__device__` one, offset form included, so the two writers below are
// unchanged.
__constant__ unsigned char *g_bases[kMaxArenas];
__constant__ unsigned g_arena_count;
unsigned char *g_host_bases[kMaxArenas];
unsigned g_host_arena_count;

namespace {

using u64 = unsigned long long;
bool allocator_active = false;
Allocator *active_allocator = nullptr;

struct Arena {
    unsigned char *base = nullptr;
    u64 capacity = 0;
    u64 bump = 0;
    std::map<u64, u64> free_spans;
};

struct Block {
    unsigned arena = 0;
    u64 off = 0;
    u64 bytes = 0;
    size_t elem_size = 0;
    size_t align = 0;
};

bool fail(std::string *error, std::string message) {
    if (error) {
        *error = std::move(message);
    }
    return false;
}

bool is_power_of_two(u64 value) {
    return value != 0 && (value & (value - 1)) == 0;
}

u64 align_up(u64 value, u64 align) {
    return (value + align - 1) & ~(align - 1);
}

u64 key(unsigned arena, u64 off) {
    return (static_cast<u64>(arena) << 32) | off;
}

bool validate_request(size_t count, size_t elem_size, size_t align,
                      u64 *bytes, std::string *error) {
    if (elem_size == 0) {
        return fail(error, "arena element size must be non-zero");
    }
    if (!is_power_of_two(align)) {
        return fail(error, "arena alignment must be a power of two");
    }
    if (align < 4) {
        return fail(error, "arena alignment must be at least four bytes");
    }
    if (elem_size >= align ? elem_size % align != 0
                           : align % elem_size != 0) {
        return fail(error, "arena element size and alignment are incompatible");
    }
    if (count > UINT32_MAX) {
        return fail(error, "arena element count exceeds the handle width");
    }
    if (count != 0 && static_cast<u64>(count) > UINT64_MAX / elem_size) {
        return fail(error, "arena allocation byte count overflows");
    }
    *bytes = static_cast<u64>(count) * elem_size;
    return true;
}

bool allocate_span(Arena &arena, u64 bytes, u64 align, u64 *out) {
    for (auto it = arena.free_spans.begin(); it != arena.free_spans.end();
         ++it) {
        const u64 begin = it->first;
        const u64 end = begin + it->second;
        const u64 aligned = align_up(begin, align);
        if (aligned + bytes > end) {
            continue;
        }
        arena.free_spans.erase(it);
        if (aligned > begin) {
            arena.free_spans[begin] = aligned - begin;
        }
        if (aligned + bytes < end) {
            arena.free_spans[aligned + bytes] = end - aligned - bytes;
        }
        *out = aligned;
        return true;
    }
    const u64 aligned = align_up(arena.bump, align);
    if (aligned + bytes > arena.capacity) {
        return false;
    }
    if (aligned > arena.bump) {
        arena.free_spans[arena.bump] = aligned - arena.bump;
    }
    arena.bump = aligned + bytes;
    *out = aligned;
    return true;
}

void release_span(Arena &arena, u64 off, u64 bytes) {
    u64 begin = off;
    u64 end = off + bytes;
    auto next = arena.free_spans.lower_bound(begin);
    if (next != arena.free_spans.end() && next->first == end) {
        end += next->second;
        next = arena.free_spans.erase(next);
    }
    if (next != arena.free_spans.begin()) {
        auto previous = std::prev(next);
        assert(previous->first + previous->second <= begin);
        if (previous->first + previous->second == begin) {
            begin = previous->first;
            arena.free_spans.erase(previous);
        }
    }
    if (end == arena.bump) {
        arena.bump = begin;
    } else {
        arena.free_spans[begin] = end - begin;
    }
}

} // namespace

struct Allocator {
    Config config;
    std::vector<Arena> arenas;
    std::map<u64, Block> live;
    u64 used = 0;
    u64 reserved = 0;
    u64 next_reserve = 0;
};

namespace {

bool open_arena(Allocator *allocator, u64 need, std::string *error) {
    if (allocator->arenas.size() >= allocator->config.max_arenas) {
        return fail(error, "HIP arena binding budget exhausted");
    }
    if (need > allocator->config.arena_cap) {
        return fail(error, "HIP arena allocation exceeds the 32-bit offset cap");
    }
    u64 reserve = allocator->next_reserve;
    if (reserve < need) {
        reserve = align_up(need, 1ull << 20);
    }
    if (reserve > allocator->config.arena_cap) {
        reserve = allocator->config.arena_cap;
    }
    if (reserve < need) {
        return fail(error, "HIP arena cannot hold the requested allocation");
    }

    Arena arena;
    HIP_HANDLE_ERROR(hipMalloc(&arena.base, reserve));
    arena.capacity = reserve;
    const unsigned index = static_cast<unsigned>(allocator->arenas.size());
    HIP_HANDLE_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(g_bases), &arena.base, sizeof(arena.base), index * sizeof(arena.base),
        hipMemcpyHostToDevice));
    allocator->arenas.push_back(std::move(arena));
    const unsigned count = static_cast<unsigned>(allocator->arenas.size());
    g_host_bases[index] = allocator->arenas.back().base;
    g_host_arena_count = count;
    HIP_HANDLE_ERROR(
        hipMemcpyToSymbol(HIP_SYMBOL(g_arena_count), &count, sizeof(count)));
    allocator->reserved += reserve;
    allocator->next_reserve =
        reserve >= allocator->config.arena_cap / 2
            ? allocator->config.arena_cap
            : reserve * 2;
    return true;
}

bool place(Allocator *allocator, u64 bytes, u64 align, unsigned *arena_id,
           u64 *off, std::string *error) {
    for (size_t i = 0; i < allocator->arenas.size(); ++i) {
        if (allocate_span(allocator->arenas[i], bytes, align, off)) {
            *arena_id = static_cast<unsigned>(i);
            return true;
        }
    }
    if (!open_arena(allocator, bytes, error)) {
        return false;
    }
    *arena_id = static_cast<unsigned>(allocator->arenas.size() - 1);
    return allocate_span(allocator->arenas.back(), bytes, align, off) ||
           fail(error, "fresh HIP arena could not satisfy its opening request");
}

Block *find(Allocator *allocator, const Handle &handle, std::string *error) {
    if (handle.arena >= allocator->arenas.size()) {
        fail(error, "HIP arena handle names an unopened arena");
        return nullptr;
    }
    auto it = allocator->live.find(key(handle.arena, handle.off));
    if (it == allocator->live.end()) {
        fail(error, "HIP arena handle does not name a live allocation");
        return nullptr;
    }
    return &it->second;
}

} // namespace

Allocator *create(const Config &config) {
    if (config.first_reserve == 0 || config.first_reserve > config.arena_cap ||
        config.arena_cap > kMaxArenaBytes || config.max_arenas == 0 ||
        config.max_arenas > kMaxArenas) {
        return nullptr;
    }
    if (allocator_active) {
        return nullptr;
    }
    auto *allocator = new Allocator();
    allocator->config = config;
    allocator->next_reserve = config.first_reserve;
    allocator_active = true;
    active_allocator = allocator;
    return allocator;
}

void destroy(Allocator *allocator) {
    if (!allocator) {
        return;
    }
    for (Arena &arena : allocator->arenas) {
        HIP_HANDLE_ERROR(hipFree(arena.base));
    }
    unsigned char *zero_bases[kMaxArenas]{};
    HIP_HANDLE_ERROR(
        hipMemcpyToSymbol(HIP_SYMBOL(g_bases), zero_bases, sizeof(zero_bases)));
    const unsigned zero = 0;
    HIP_HANDLE_ERROR(
        hipMemcpyToSymbol(HIP_SYMBOL(g_arena_count), &zero, sizeof(zero)));
    std::memset(g_host_bases, 0, sizeof(g_host_bases));
    g_host_arena_count = 0;
    delete allocator;
    allocator_active = false;
    active_allocator = nullptr;
}

Allocator *active() { return active_allocator; }

Handle alloc_or_exit(size_t count, size_t elem_size, size_t align) {
    Handle handle{};
    std::string error;
    if (!active_allocator ||
        !alloc(active_allocator, count, elem_size, align, &handle, &error)) {
        std::fprintf(stderr, "### hip arena allocation failed: %s\n",
                     error.c_str());
        g_ppf_fatal_code = 2;
        std::exit(1);
    }
    return handle;
}

void free_or_exit(Handle *handle) {
    std::string error;
    if (!active_allocator || !free(active_allocator, handle, &error)) {
        std::fprintf(stderr, "### hip arena free failed: %s\n", error.c_str());
        g_ppf_fatal_code = 2;
        std::exit(1);
    }
}

bool alloc(Allocator *allocator, size_t count, size_t elem_size, size_t align,
           Handle *out, std::string *error) {
    if (!allocator || !out) {
        return fail(error, "HIP arena allocator or output handle is null");
    }
    *out = Handle{0, 0, 0, 0};
    u64 bytes = 0;
    if (!validate_request(count, elem_size, align, &bytes, error)) {
        return false;
    }
    if (bytes == 0) {
        return !allocator->arenas.empty() || open_arena(allocator, 0, error);
    }
    unsigned arena_id = 0;
    u64 off = 0;
    if (!place(allocator, bytes, align, &arena_id, &off, error)) {
        return false;
    }
    if (off > UINT32_MAX || off % align != 0) {
        release_span(allocator->arenas[arena_id], off, bytes);
        return fail(error, "HIP arena produced an unrepresentable offset");
    }
    allocator->live[key(arena_id, off)] =
        Block{arena_id, off, bytes, elem_size, align};
    allocator->used += bytes;
    *out = Handle{arena_id, static_cast<unsigned>(off),
                  static_cast<unsigned>(count), static_cast<unsigned>(count)};
    return true;
}

bool grow(Allocator *allocator, Handle *handle, size_t new_count,
          size_t elem_size, size_t align, std::string *error) {
    if (!allocator || !handle) {
        return fail(error, "HIP arena allocator or handle is null");
    }
    if (handle->size > handle->allocated) {
        return fail(error, "HIP arena handle size exceeds capacity");
    }
    if (new_count < handle->allocated) {
        return fail(error, "HIP arena grow cannot shrink an allocation");
    }
    if (new_count == handle->allocated) {
        return true;
    }
    if (handle->allocated == 0) {
        const unsigned logical_size = handle->size;
        if (!alloc(allocator, new_count, elem_size, align, handle, error)) {
            return false;
        }
        handle->size = logical_size;
        return true;
    }
    Block *old = find(allocator, *handle, error);
    if (!old) {
        return false;
    }
    if (old->elem_size != elem_size || old->align != align) {
        return fail(error, "HIP arena grow changed element type or alignment");
    }
    const Block previous = *old;
    Handle replacement{};
    if (!alloc(allocator, new_count, elem_size, align, &replacement, error)) {
        return false;
    }
    HIP_HANDLE_ERROR(hipMemcpy(
        allocator->arenas[replacement.arena].base + replacement.off,
        allocator->arenas[previous.arena].base + previous.off, previous.bytes,
        hipMemcpyDeviceToDevice));
    allocator->live.erase(key(previous.arena, previous.off));
    release_span(allocator->arenas[previous.arena], previous.off,
                 previous.bytes);
    allocator->used -= previous.bytes;
    replacement.size = handle->size;
    *handle = replacement;
    return true;
}

bool free(Allocator *allocator, Handle *handle, std::string *error) {
    if (!allocator || !handle) {
        return fail(error, "HIP arena allocator or handle is null");
    }
    if (handle->allocated == 0) {
        *handle = Handle{0, 0, 0, 0};
        return true;
    }
    Block *block = find(allocator, *handle, error);
    if (!block) {
        return false;
    }
    const Block released = *block;
    allocator->live.erase(key(released.arena, released.off));
    release_span(allocator->arenas[released.arena], released.off,
                 released.bytes);
    allocator->used -= released.bytes;
    *handle = Handle{0, 0, 0, 0};
    return true;
}

void *host_resolve(Allocator *allocator, const Handle &handle,
                   std::string *error) {
    Block *block = allocator ? find(allocator, handle, error) : nullptr;
    return block ? allocator->arenas[block->arena].base + block->off : nullptr;
}

bool block_bytes(Allocator *allocator, const Handle &handle,
                 unsigned long long *out, std::string *error) {
    if (!allocator || !out) {
        return fail(error, "HIP arena allocator or output is null");
    }
    Block *block = find(allocator, handle, error);
    if (!block) {
        return false;
    }
    *out = block->bytes;
    return true;
}

bool write(Allocator *allocator, const Handle &handle, const void *source,
           size_t bytes, std::string *error) {
    if (bytes == 0) {
        return true;
    }
    Block *block = allocator ? find(allocator, handle, error) : nullptr;
    if (!block || (!source && bytes != 0)) {
        return block ? fail(error, "HIP arena write source is null") : false;
    }
    if (bytes > block->bytes) {
        return fail(error, "HIP arena write exceeds allocation capacity");
    }
    HIP_HANDLE_ERROR(hipMemcpy(allocator->arenas[block->arena].base + block->off,
                                 source, bytes, hipMemcpyHostToDevice));
    return true;
}

bool read(Allocator *allocator, const Handle &handle, void *destination,
          size_t bytes, std::string *error) {
    if (bytes == 0) {
        return true;
    }
    Block *block = allocator ? find(allocator, handle, error) : nullptr;
    if (!block || (!destination && bytes != 0)) {
        return block ? fail(error, "HIP arena read destination is null") : false;
    }
    if (bytes > block->bytes) {
        return fail(error, "HIP arena read exceeds allocation capacity");
    }
    HIP_HANDLE_ERROR(hipMemcpy(destination,
                                 allocator->arenas[block->arena].base + block->off,
                                 bytes, hipMemcpyDeviceToHost));
    return true;
}

unsigned arena_count(const Allocator *allocator) {
    return allocator ? static_cast<unsigned>(allocator->arenas.size()) : 0;
}

unsigned long long bytes_used(const Allocator *allocator) {
    return allocator ? allocator->used : 0;
}

unsigned long long bytes_reserved(const Allocator *allocator) {
    return allocator ? allocator->reserved : 0;
}

} // namespace compute::arena
