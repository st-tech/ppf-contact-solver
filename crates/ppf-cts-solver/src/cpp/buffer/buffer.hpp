// File: buffer.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef BUFFER_HPP
#define BUFFER_HPP

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "../vec/vec.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <utility>

namespace buffer {

// Forward declaration
class MemoryPool;

// RAII wrapper for Vec<T> that auto-releases buffer on destruction
template<typename T>
class PooledVec : public Vec<T> {
private:
    MemoryPool* pool_;
    bool released_;

public:
    // Constructor from pool and vec
    PooledVec(MemoryPool* pool, const Vec<T>& vec);

    // Destructor - auto-release if not manually released
    ~PooledVec();

    // Move constructor
    PooledVec(PooledVec&& other) noexcept;

    // Move assignment
    PooledVec& operator=(PooledVec&& other) noexcept;

    // Delete copy operations to prevent double-release
    PooledVec(const PooledVec&) = delete;
    PooledVec& operator=(const PooledVec&) = delete;

    // Manual release (optional - for explicit control)
    void release();

    // Check if buffer has been released
    bool is_released() const { return released_; }

    // Get underlying Vec<T> (for passing to functions expecting Vec<T>)
    Vec<T> as_vec() const {
        return Vec<T>{this->data, this->size, this->allocated};
    }
};

// Memory pool allocator for temporary buffers
class MemoryPool {
private:
    // Storage for actual buffer data
    std::vector<Vec<float>> buffers_;
    // Track which buffers are currently in use
    std::vector<bool> in_use_;

public:
    // Allocate a buffer and reinterpret as requested type.
    // count: number of elements of type T needed.
    // Buffers are sorted smallest-first, so this picks the tightest available
    // fit. If nothing fits, the pool grows to a HIGH-WATER mark rather than
    // accumulating: it reallocates the largest free buffer in place instead of
    // orphaning it and adding a new one (the old behavior monotonically grew
    // the pool when fed schwarz's nnz-sized requests and caused the OOM). A
    // brand-new buffer is added only when every buffer is checked out, i.e.
    // when concurrent demand genuinely rises. Once a scene reaches its peak
    // contact count the sizes stop changing and get<T> performs no cudaMalloc.
    // Returns PooledVec<T> which auto-releases on destruction.
    template<typename T>
    PooledVec<T> get(size_t count) {
        // Pool storage is float-backed; round the request up to whole floats.
        size_t float_count = (count * sizeof(T) + sizeof(float) - 1) / sizeof(float);
        if (float_count == 0) {
            float_count = 1;
        }

        // 1. Reuse the smallest free buffer that already fits.
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (!in_use_[i] && buffers_[i].size >= float_count) {
                in_use_[i] = true;
                return PooledVec<T>(this, Vec<T>{
                    reinterpret_cast<T*>(buffers_[i].data),
                    static_cast<unsigned>(count),
                    buffers_[i].allocated});
            }
        }

        // 2. Nothing fits. Grow the largest free buffer in place (free +
        // realloc; a free buffer holds no live data so no copy is needed). This
        // keeps the buffer COUNT bounded by peak concurrency and the sizes at
        // their high-water mark, so the pool stops growing once the workload
        // stabilizes. Only when no buffer is free do we add one.
        //
        // Allocate with 1.5x headroom (amortized growth, like std::vector) so a
        // contact-driven request (schwarz Galerkin / radix / scan, whose size
        // tracks num_contact) that creeps to a new high reuses the slack
        // instead of reallocating every step. Without this, each tiny new high
        // churns a free+alloc pair even though memory is already plateaued.
        size_t alloc_floats = float_count + float_count / 2;
        long grow = -1;
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (!in_use_[i] &&
                (grow < 0 || buffers_[i].size > buffers_[grow].size)) {
                grow = static_cast<long>(i);
            }
        }
        if (grow >= 0) {
            buffers_[grow].free();
            buffers_[grow] = Vec<float>::alloc(static_cast<unsigned>(alloc_floats));
        } else {
            add_buffer(Vec<float>::alloc(static_cast<unsigned>(alloc_floats)));
        }
        sort_buffers();

        // The buffer we just grew/added is now the tightest fit; claim it.
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (!in_use_[i] && buffers_[i].size >= float_count) {
                in_use_[i] = true;
                return PooledVec<T>(this, Vec<T>{
                    reinterpret_cast<T*>(buffers_[i].data),
                    static_cast<unsigned>(count),
                    buffers_[i].allocated});
            }
        }

        // Should never reach here.
        assert(false && "MemoryPool: Failed to allocate buffer");
        return PooledVec<T>(this, Vec<T>{nullptr, 0, 0});
    }

    // Pre-allocate a set of backing buffers once (at solver init), so the hot
    // loop draws from them with no cudaMalloc. `slots` is a list of
    // (float_count, how_many). Any existing buffers are released first, so this
    // is safe to call once per session. The sizing need only be approximate:
    // the high-water get<T> growth above covers any request it under-sizes, and
    // a too-large request is bounded by reuse, so this just front-loads the
    // common buffers to setup time.
    void reserve(const std::vector<std::pair<size_t, size_t>>& slots) {
        clear();
        for (const auto& slot : slots) {
            size_t fc = slot.first ? slot.first : 1;
            for (size_t k = 0; k < slot.second; ++k) {
                add_buffer(Vec<float>::alloc(static_cast<unsigned>(fc)));
            }
        }
        sort_buffers();
    }

    // Release every backing buffer and reset. Must be called with nothing
    // checked out (between solves); reserve() uses it to re-seed a session.
    void clear() {
        for (size_t i = 0; i < buffers_.size(); ++i) {
            assert(!in_use_[i] && "MemoryPool::clear() with buffers in use");
            buffers_[i].free();
        }
        buffers_.clear();
        in_use_.clear();
    }

    // Release a buffer back to the pool
    template<typename T>
    void release(const Vec<T>& vec) {
        void* ptr = reinterpret_cast<void*>(vec.data);

        // Find buffer by address
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (reinterpret_cast<void*>(buffers_[i].data) == ptr) {
                assert(in_use_[i] && "MemoryPool: Attempting to release buffer that is not in use");
                in_use_[i] = false;
                return;
            }
        }

        assert(false && "MemoryPool: Buffer not found in pool");
    }

    // Add a buffer to the pool
    void add_buffer(Vec<float> buffer) {
        buffers_.push_back(buffer);
        in_use_.push_back(false);
    }

    // Sort buffers by size (smallest first) for efficient allocation
    void sort_buffers() {
        // Create index array to track original positions
        std::vector<size_t> indices(buffers_.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        // Sort indices by buffer size
        std::sort(indices.begin(), indices.end(),
                  [this](size_t a, size_t b) {
                      return buffers_[a].size < buffers_[b].size;
                  });

        // Reorder buffers and in_use flags according to sorted indices
        std::vector<Vec<float>> sorted_buffers;
        std::vector<bool> sorted_in_use;
        sorted_buffers.reserve(buffers_.size());
        sorted_in_use.reserve(in_use_.size());

        for (size_t idx : indices) {
            sorted_buffers.push_back(buffers_[idx]);
            sorted_in_use.push_back(in_use_[idx]);
        }

        buffers_ = std::move(sorted_buffers);
        in_use_ = std::move(sorted_in_use);
    }
};

// PooledVec method implementations (defined after MemoryPool is complete)

template<typename T>
PooledVec<T>::PooledVec(MemoryPool* pool, const Vec<T>& vec)
    : Vec<T>(vec), pool_(pool), released_(false) {}

template<typename T>
PooledVec<T>::~PooledVec() {
    if (!released_ && pool_ && this->data) {
        // Create temporary Vec to pass to release
        Vec<T> vec_copy = {this->data, this->size, this->allocated};
        pool_->release(vec_copy);
    }
}

template<typename T>
PooledVec<T>::PooledVec(PooledVec&& other) noexcept
    : Vec<T>(other), pool_(other.pool_), released_(other.released_) {
    other.released_ = true;
    other.pool_ = nullptr;
}

template<typename T>
PooledVec<T>& PooledVec<T>::operator=(PooledVec&& other) noexcept {
    if (this != &other) {
        // Release current buffer if we have one
        if (!released_ && pool_ && this->data) {
            Vec<T> vec_copy = {this->data, this->size, this->allocated};
            pool_->release(vec_copy);
        }
        Vec<T>::operator=(other);
        pool_ = other.pool_;
        released_ = other.released_;
        other.released_ = true;
        other.pool_ = nullptr;
    }
    return *this;
}

template<typename T>
void PooledVec<T>::release() {
    if (!released_ && pool_ && this->data) {
        Vec<T> vec_copy = {this->data, this->size, this->allocated};
        pool_->release(vec_copy);
        released_ = true;
    }
}

// Get the global memory pool instance
MemoryPool& get();

// Pre-seed the global pool at solver init with the mesh/body-bounded buffers
// the hot loop reuses, sized from the worst-case counts known at initialize().
// schwarz's contact/nnz-driven buffers are intentionally NOT pre-sized here;
// they grow the same pool to its high-water mark at runtime and then stabilize.
void reserve_for_mesh(unsigned n_verts, unsigned n_edges, unsigned n_faces,
                      unsigned n_bodies);

} // namespace buffer

#endif // BUFFER_HPP
