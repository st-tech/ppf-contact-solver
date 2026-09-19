// File: mem.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// HOST-DEVICE TRANSFER FOR THE CUDA TARGET.
//
// Copying bytes between host and device is one of the five verbs this crate
// exists for, so the helpers that spell it belong here rather than beside the
// kernel bodies: `hipMemcpy` is a mechanism, and a tree whose contents belong
// to no compiler in particular cannot name it. The signatures are stated over
// the containers the neutral tree declares (`Vec<T>`, `VecVec<T>`) and the
// arena-backed allocations `cuda_vec.hpp` defines (`ArenaPtr<T>`,
// `ArenaVec<T>`), so a caller says what to move without saying how.
//
// The overload set is deliberately narrow: a raw device pointer or an
// `ArenaPtr` in either direction, and a whole container to the device. There is
// no container-to-host form, because a device container's `size` is authored on
// the device and a host copy of it would need the count read back first.

#ifndef HIP_MEM_HPP
#define HIP_MEM_HPP

#include "hip_utils.hpp" // HIP_HANDLE_ERROR
#include "hip_vec.hpp"   // ArenaPtr<T>, ArenaVec<T>
#include "vec/vec.hpp"    // Vec<T>, VecVec<T>, through -I$(KERNEL_ROOT)
#include <cassert>

namespace mem {

template <typename A>
void copy_from_device_to_host(const A *dev_src, A *host_dst,
                              unsigned count = 1) {
    if (count) {
        HIP_HANDLE_ERROR(hipMemcpy(host_dst, dev_src, count * sizeof(A),
                                     hipMemcpyDeviceToHost));
    }
}

template <typename A>
void copy_from_device_to_host(ArenaPtr<A> dev_src, A *host_dst,
                              unsigned count = 1) {
    if (count) {
        copy_from_device_to_host(dev_src.get(), host_dst, count);
    }
}

template <typename A>
void copy_from_host_to_device(const A *host_src, A *dev_dst,
                              unsigned count = 1) {
    if (count) {
        HIP_HANDLE_ERROR(hipMemcpy(dev_dst, host_src, count * sizeof(A),
                                     hipMemcpyHostToDevice));
    }
}

template <typename A>
void copy_from_host_to_device(const A *host_src, ArenaPtr<A> dev_dst,
                              unsigned count = 1) {
    if (count) {
        copy_from_host_to_device(host_src, dev_dst.get(), count);
    }
}

template <typename T>
void copy_to_device(const Vec<T> &host_src, Vec<T> &dev_dst) {
    assert(host_src.size <= dev_dst.allocated);
    dev_dst.size = host_src.size;
    copy_from_host_to_device(host_src.data, dev_dst.data, host_src.size);
}

template <typename T>
void copy_to_device(const Vec<T> &host_src, ArenaVec<T> &dev_dst) {
    assert(host_src.size <= dev_dst.allocated);
    dev_dst.size = host_src.size;
    copy_from_host_to_device(host_src.data, dev_dst.data, host_src.size);
}

template <typename T>
void copy_to_device(const VecVec<T> &host_src, VecVec<T> &dev_dst) {
    assert(host_src.nnz <= dev_dst.nnz_allocated);
    assert(host_src.size <= dev_dst.offset_allocated);
    dev_dst.size = host_src.size;
    dev_dst.nnz = host_src.nnz;
    copy_from_host_to_device(host_src.data, dev_dst.data, host_src.nnz);
    copy_from_host_to_device(host_src.offset, dev_dst.offset,
                             host_src.size + 1);
}

// WAIT FOR EVERY DISPATCH THE DEVICE HAS BEEN GIVEN TO FINISH.
//
// A wait is mechanism in the same sense the copies above are, and it is here
// for the same reason: a caller in the neutral tree has to be able to order a
// dispatch against a host read without naming the runtime that does it. A
// caller that reads a kernel's output through the host pointer overloads above,
// rather than through a blocking copy, has no other way to say it.
//
// SEPARATE FROM THE COPIES ON PURPOSE, and it is not the same statement as the
// blocking copy that usually follows it. `hipMemcpy` on the default stream
// orders the transfer behind the launches already issued, so the two coincide
// for THAT read and only for that read; a caller waiting before reading
// something the copy does not name, or before timing, is saying the other
// thing. Do not fold one into the other.
inline void wait_for_device() { HIP_HANDLE_ERROR(hipDeviceSynchronize()); }

} // namespace mem

#endif
