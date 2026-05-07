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
    // Allocate a buffer and reinterpret as requested type
    // count: number of elements of type T needed
    // Note: Buffers are sorted by size (smallest first), so this picks
    // the smallest available buffer that fits, maximizing memory efficiency
    // If no suitable buffer exists, allocates a new one on-the-fly
    // Returns PooledVec<T> which auto-releases on destruction
    template<typename T>
    PooledVec<T> get(size_t count) {
        // Calculate required float count based on size
        size_t float_count = (count * sizeof(T) + sizeof(float) - 1) / sizeof(float);

        // Find smallest available buffer with sufficient capacity
        // Since buffers are sorted by size, first match is the smallest fit
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (!in_use_[i] && buffers_[i].size >= float_count) {
                in_use_[i] = true;
                // Reinterpret float buffer as requested type
                Vec<T> vec{
                    reinterpret_cast<T*>(buffers_[i].data),
                    static_cast<unsigned>(count),
                    buffers_[i].allocated
                };
                return PooledVec<T>(this, vec);
            }
        }

        // No suitable buffer found - allocate a new one on-the-fly
        Vec<float> new_buffer = Vec<float>::alloc(float_count);
        add_buffer(new_buffer);
        // Sort to maintain smallest-first order
        sort_buffers();

        // Find and mark the newly added buffer (must exist now)
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (!in_use_[i] && buffers_[i].size >= float_count) {
                in_use_[i] = true;
                Vec<T> vec{
                    reinterpret_cast<T*>(buffers_[i].data),
                    static_cast<unsigned>(count),
                    buffers_[i].allocated
                };
                return PooledVec<T>(this, vec);
            }
        }

        // Should never reach here
        assert(false && "MemoryPool: Failed to allocate buffer");
        Vec<T> empty{nullptr, 0, 0};
        return PooledVec<T>(this, empty);
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

} // namespace buffer

#endif // BUFFER_HPP
