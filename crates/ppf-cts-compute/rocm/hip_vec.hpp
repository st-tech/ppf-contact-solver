// File: cuda_vec.hpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE CUDA DEFINITION OF THE OWNING CONTAINERS, whose names are neutral and
// whose bodies are not.
//
// `ArenaPtr<T>`, `ArenaVec<T>` and `ArenaVecVec<T>` are the names the neutral
// tree declares its owning arrays with, and each backend defines them for
// itself: what nvcc gets is below, resolving a handle through
// `compute::arena::g_bases` on the device and `g_host_bases` on the host. The
// names carry no compiler in them because they say what the thing IS, an
// allocation out of an arena, and the same declaration has to be satisfiable by
// three targets. The FILE is named for the target because it is one target's
// answer, and it lives under `cuda/` for the same reason.
//
// The view forms are `Vec<T>` and `VecVec<T>` in the neutral tree, which each
// of these converts to on the way in. Reserving the bytes belongs here;
// reading them belongs there.

#ifndef HIP_VEC_HPP
#define HIP_VEC_HPP

// The neutral tree, through the include path the CUDA build supplies
// (-I$(KERNEL_ROOT) in ../Makefile). vec/vec.hpp brings the plain-old-data view
// of `Vec<T>` and `VecVec<T>` this header hands allocations back as, and that
// is the whole of what it needs: naming the scene model instead (`data.hpp`)
// would put every solver type this file never mentions into the include
// closure of an allocator. The dependency points one way: the neutral tree
// names nothing in this crate.
#include "vec/vec.hpp"
#include "arena/arena.hpp"
#include "hip_utils.hpp"      // g_device_alloc_count / g_device_free_count

namespace compute::arena {
extern unsigned char *g_host_bases[kMaxArenas];
extern unsigned g_host_arena_count;
}

// A device pointer as an (arena, byte offset) pair. `arena == kNullArena` is
// the null pointer value and means exactly one thing: this handle names no
// allocation, because it was never allocated or has been freed. An EMPTY array
// is NOT that: it takes a real zero-length handle from the arena (see
// ArenaVec::alloc_aligned), so its base address is well defined and `get()`
// stays legal. Keeping the two apart is what lets the assert in `get()` stay
// strict, since every remaining trip is a genuinely dead handle.
template <class T> struct ArenaPtr {
    static constexpr unsigned kNullArena = 0xFFFFFFFFu;

    unsigned arena{kNullArena};
    unsigned off{0};

    __host__ __device__ ArenaPtr() = default;
    __host__ __device__ ArenaPtr(unsigned arena_, unsigned off_)
        : arena(arena_), off(off_) {}
    __host__ __device__ ArenaPtr(std::nullptr_t) {}
    __host__ __device__ ArenaPtr &operator=(std::nullptr_t) {
        arena = kNullArena;
        off = 0;
        return *this;
    }
    __host__ __device__ bool operator==(std::nullptr_t) const {
        return !static_cast<bool>(*this);
    }
    __host__ __device__ bool operator!=(std::nullptr_t) const {
        return static_cast<bool>(*this);
    }
    __host__ __device__ T *get() const {
#ifdef __HIP_DEVICE_COMPILE__
        assert(arena < compute::arena::g_arena_count);
        return reinterpret_cast<T *>(compute::arena::g_bases[arena] + off);
#else
        assert(arena < compute::arena::g_host_arena_count);
        return reinterpret_cast<T *>(compute::arena::g_host_bases[arena] + off);
#endif
    }
    __host__ __device__ operator T *() const { return get(); }
    __host__ __device__ explicit operator bool() const {
#ifdef __HIP_DEVICE_COMPILE__
        return arena < compute::arena::g_arena_count;
#else
        return arena < compute::arena::g_host_arena_count;
#endif
    }
    __host__ __device__ T &operator[](unsigned i) const { return get()[i]; }
    __host__ __device__ ArenaPtr operator+(unsigned i) const {
        return ArenaPtr(arena, off + i * sizeof(T));
    }
};

template <class T> struct ArenaVec {
    ArenaPtr<T> data;
    unsigned size{0};
    unsigned allocated{0};

    static ArenaVec alloc(unsigned n, unsigned alloc_factor = 1) {
        return alloc_aligned(n, alignof(T), alloc_factor);
    }
    // An empty array takes a handle too. compute::arena::alloc answers a
    // zero-element request with a resolvable zero-length handle (it opens an
    // arena first if none is), so an empty array's base address is well
    // defined. Kernels form that address before consulting the length, because
    // the shared-math seam takes a raw base pointer plus the counts that bound
    // it (fixed_csr_apply_row, aabb_query): a zero-length range is read
    // zero times, so the address is materialized and never dereferenced. Making
    // that case null instead conflates "empty" with "dead handle" and traps in
    // get() for any scene that leaves an array empty. A granular scene is the
    // shipped example: its vertices carry no face, edge, tet or stitch, so its
    // whole fixed Hessian pattern is empty and the SpMV trips on every row.
    static ArenaVec alloc_aligned(unsigned n, size_t requested_align,
                                 unsigned alloc_factor = 1) {
        ArenaVec out;
        const unsigned capacity = alloc_factor * n;
        const size_t natural = alignof(T) < 4 ? 4 : alignof(T);
        const size_t align =
            requested_align < natural ? natural : requested_align;
        const ArenaHandle h =
            compute::arena::alloc_or_exit(capacity, sizeof(T), align);
        out.data = ArenaPtr<T>(h.arena, h.off);
        out.size = n;
        out.allocated = capacity;
        if (capacity) {
            // Counted per REAL device reservation, which is what the per-step
            // alloc/free tally in advance() reports. A zero-length handle
            // reserves no bytes, so counting it would put a phantom allocation
            // in that log; free() is gated the same way so the two balance.
            ++g_device_alloc_count;
        }
        return out;
    }
    static ArenaVec reserve(unsigned n) {
        ArenaVec out = alloc(n);
        out.size = 0;
        return out;
    }
    bool free() {
        if (!data) {
            return false;
        }
        // compute::arena::free returns a zero-length handle to the null state
        // without touching the block map, so an empty array releases nothing
        // and reports nothing, matching alloc_aligned above.
        const bool reserved = allocated != 0;
        ArenaHandle h{data.arena, data.off, size, allocated};
        compute::arena::free_or_exit(&h);
        data = nullptr;
        size = 0;
        allocated = 0;
        if (reserved) {
            ++g_device_free_count;
        }
        return reserved;
    }
    void resize(unsigned new_size) {
        assert(new_size <= allocated);
        size = new_size;
    }
    __host__ __device__ T &operator[](unsigned i) {
        assert(i < size);
        return data[i];
    }
    __host__ __device__ const T &operator[](unsigned i) const {
        assert(i < size);
        return data[i];
    }
    __host__ __device__ operator Vec<T>() const {
        Vec<T> out;
        out.data = data ? data.get() : nullptr;
        out.size = size;
        out.allocated = allocated;
        return out;
    }
    // FILLED THROUGH `kernels::fill_view`, WHICH vec/vec.hpp DECLARES, RATHER
    // THAN THROUGH `kernels::set`. Both dispatch the same fill and are
    // instantiated for the same eight types; the difference is what naming one
    // drags in. `set` is declared in primitives/vec_ops.hpp beside the linear
    // solve's scalar_div, its breakdown latch and its indirect combines, so a
    // container that named it would put the PCG's interface into the include
    // closure of an allocator. `fill_view` takes no queue for the same reason
    // this header can reach it: a declaration every compiler reads cannot spell
    // `hipStream_t`.
    ArenaVec<T> clear(const T value = T()) {
        if (data && size) {
            kernels::fill_view(data.get(), size, value);
        }
        return *this;
    }
};

template <class T> struct ArenaVecVec {
    ArenaPtr<T> data;
    ArenaPtr<unsigned> offset;
    unsigned size{0};
    unsigned nnz{0};
    unsigned nnz_allocated{0};
    unsigned offset_allocated{0};

    static ArenaVecVec alloc(unsigned nrow, unsigned max_nnz) {
        ArenaVecVec out;
        out.size = nrow;
        out.nnz_allocated = max_nnz;
        out.offset_allocated = nrow + 1;
        // A table with rows but no entries is ordinary (no fixed Hessian block
        // exists in a scene with no elements), and its value array still needs
        // a base address the row loops can offset from, so it is allocated at
        // zero length rather than left null.
        ArenaVec<T> values = ArenaVec<T>::alloc(max_nnz);
        out.data = values.data;
        ArenaVec<unsigned> offsets = ArenaVec<unsigned>::alloc(nrow + 1);
        out.offset = offsets.data;
        return out;
    }
    bool free() {
        bool freed = false;
        if (data) {
            ArenaVec<T> values{data, nnz, nnz_allocated};
            freed = values.free() || freed;
            data = nullptr;
        }
        if (offset) {
            ArenaVec<unsigned> offsets{offset, size + 1, offset_allocated};
            freed = offsets.free() || freed;
            offset = nullptr;
        }
        size = nnz = nnz_allocated = offset_allocated = 0;
        return freed;
    }
    __host__ __device__ T &operator()(unsigned i, unsigned j) {
        assert(i < size);
        const unsigned k = offset[i] + j;
        assert(k < offset[i + 1]);
        return data[k];
    }
    __host__ __device__ const T &operator()(unsigned i, unsigned j) const {
        assert(i < size);
        const unsigned k = offset[i] + j;
        assert(k < offset[i + 1]);
        return data[k];
    }
    __host__ __device__ unsigned count(unsigned i) const {
        assert(i < size);
        return size ? offset[i + 1] - offset[i] : 0;
    }
    __host__ __device__ unsigned count() const {
        return size ? offset[size] : 0;
    }
    __host__ __device__ operator VecVec<T>() const {
        VecVec<T> out;
        out.data = data ? data.get() : nullptr;
        out.offset = offset ? offset.get() : nullptr;
        out.size = size;
        out.nnz = nnz;
        out.nnz_allocated = nnz_allocated;
        out.offset_allocated = offset_allocated;
        return out;
    }
};

static_assert(sizeof(ArenaVec<float>) == sizeof(ArenaHandle),
              "ArenaVec must retain the four-word handle layout");

#endif
