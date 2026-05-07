// File: buffer.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "buffer.hpp"

namespace buffer {

// Global memory pool instance
// Buffers are allocated on-demand when get<T>() is called
static MemoryPool global_pool;

MemoryPool &get() { return global_pool; }

} // namespace buffer
