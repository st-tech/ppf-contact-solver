// File: fix.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef FIX_HPP
#define FIX_HPP

#include "../../common.hpp"
// THE VOCABULARY, NOT `data.hpp`. This header names exactly three spellings,
// `Vec3f`, `Vec3f` and `Mat3x3f`, and they live in the type-alias header that
// `data.hpp` itself includes. Naming `data.hpp` here dragged in `vec/vec.hpp`
// and with it `<cassert>`, which the Metal shader compiler cannot open, so an
// entry harness reaching this file failed on a header it does not use.
#include "../../linalg/type_aliases.hpp"

// THE ADDRESS-SPACE MACRO, guarded exactly as `contact/accd.hpp` guards it. A
// shared header is not a neutral kernel body: MSL requires an address space on
// every reference type, while CUDA and the host have one and are handed the
// name defined empty.
#ifndef SM_THREAD
#define SM_THREAD
#define FIX_UNDEF_THREAD
#endif

namespace fix {

// `inline`, like `accd::park_floor` and for the same reason: more than one
// translation unit includes this header on a host build, and a definition
// without it is a duplicate symbol at link rather than a compile error, so the
// failure appears only once a second includer exists.
__device__ inline Vec3f gradient(SM_THREAD const Vec3f &x,
                                 SM_THREAD const Vec3f &y) {
    return (x - y).cast<float>();
}

__device__ inline Mat3x3f hessian() { return Mat3x3f::Identity(); }

} // namespace fix

#ifdef FIX_UNDEF_THREAD
#undef SM_THREAD
#undef FIX_UNDEF_THREAD
#endif

#endif
