// File: buffer.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef BUFFER_HPP
#define BUFFER_HPP

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "../vec/vec.hpp"
#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace buffer {

class MemoryPool;

template <typename T> class PooledVec : public Vec<T> {
  private:
    MemoryPool *pool_;
    bool released_;

  public:
    PooledVec(MemoryPool *pool, const Vec<T> &vec);
    ~PooledVec();
    PooledVec(PooledVec &&other) noexcept;
    PooledVec &operator=(PooledVec &&other) noexcept;
    PooledVec(const PooledVec &) = delete;
    PooledVec &operator=(const PooledVec &) = delete;
    void release();
    bool is_released() const { return released_; }
    Vec<T> as_vec() const {
        return Vec<T>{this->data, this->size, this->allocated};
    }
};

class MemoryPool {
  private:
    std::vector<Vec<float>> buffers_;
    std::vector<bool> in_use_;

  public:
    template <typename T> PooledVec<T> get(size_t count) {
        size_t float_count =
            (count * sizeof(T) + sizeof(float) - 1) / sizeof(float);

        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (!in_use_[i] && buffers_[i].size >= float_count) {
                in_use_[i] = true;
                Vec<T> vec{reinterpret_cast<T *>(buffers_[i].data),
                           static_cast<unsigned>(count), buffers_[i].allocated};
                return PooledVec<T>(this, vec);
            }
        }

        Vec<float> new_buffer = Vec<float>::alloc(float_count);
        add_buffer(new_buffer);
        sort_buffers();

        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (!in_use_[i] && buffers_[i].size >= float_count) {
                in_use_[i] = true;
                Vec<T> vec{reinterpret_cast<T *>(buffers_[i].data),
                           static_cast<unsigned>(count), buffers_[i].allocated};
                return PooledVec<T>(this, vec);
            }
        }

        assert(false && "MemoryPool: Failed to allocate buffer");
        Vec<T> empty{nullptr, 0, 0};
        return PooledVec<T>(this, empty);
    }

    template <typename T> void release(const Vec<T> &vec) {
        void *ptr = reinterpret_cast<void *>(vec.data);

        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (reinterpret_cast<void *>(buffers_[i].data) == ptr) {
                assert(in_use_[i] && "MemoryPool: Attempting to release buffer "
                                     "that is not in use");
                in_use_[i] = false;
                return;
            }
        }

        assert(false && "MemoryPool: Buffer not found in pool");
    }

    void add_buffer(Vec<float> buffer) {
        buffers_.push_back(buffer);
        in_use_.push_back(false);
    }

    void sort_buffers() {
        std::vector<size_t> indices(buffers_.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
            return buffers_[a].size < buffers_[b].size;
        });

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

template <typename T>
PooledVec<T>::PooledVec(MemoryPool *pool, const Vec<T> &vec)
    : Vec<T>(vec), pool_(pool), released_(false) {}

template <typename T> PooledVec<T>::~PooledVec() {
    if (!released_ && pool_ && this->data) {
        Vec<T> vec_copy = {this->data, this->size, this->allocated};
        pool_->release(vec_copy);
    }
}

template <typename T>
PooledVec<T>::PooledVec(PooledVec &&other) noexcept
    : Vec<T>(other), pool_(other.pool_), released_(other.released_) {
    other.released_ = true;
    other.pool_ = nullptr;
}

template <typename T>
PooledVec<T> &PooledVec<T>::operator=(PooledVec &&other) noexcept {
    if (this != &other) {
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

template <typename T> void PooledVec<T>::release() {
    if (!released_ && pool_ && this->data) {
        Vec<T> vec_copy = {this->data, this->size, this->allocated};
        pool_->release(vec_copy);
        released_ = true;
    }
}

MemoryPool &get();

} // namespace buffer

#endif // BUFFER_HPP
