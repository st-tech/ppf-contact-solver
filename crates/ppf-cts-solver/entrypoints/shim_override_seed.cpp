// File: entrypoints/shim_override_seed.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The CPU backend's entry points for the keyframe velocity seed and the live
// position readback.
//
// A SECOND SHIM TRANSLATION UNIT, not an extension of kernel_shim.cpp: the
// shims split by subject once one file grows past comfortable reading, and this
// pair is one such split. Both files are the same thing, a loop and a gather
// around a shared body with no arithmetic of their own, and both include the
// `cpp` RENDERING of a neutral kernel source rather than the source itself,
// from $OUT_DIR/kernelgen where crates/ppf-cts-solver/build.rs put it. Nothing
// is written under src/, because two build scripts watch that tree recursively.
//
// If a float expression ever appears below, the architecture's one hard
// constraint has been crossed. Everything here is an index, a bound and a
// pointer; every value comes from main/override_seed.kernel.cpp, which nvcc and
// the Metal shader compiler compile from the same bytes.

// data.hpp declares the shared vector and scalar types this file names, and
// reaches the backend seam (src/kernels/seam/seam.hpp) that supplies the seam
// names.
// Named by a relative path, and src/kernels is deliberately NOT on the include
// path: a rendering sits at the same relative path under $OUT_DIR/kernelgen
// that its neutral source has under src/kernels, so an -I on both would resolve
// the quoted include below to whichever came first, and compiling the neutral
// source instead of its rendering produces a translation unit that happens to
// work.
#include "../src/kernels/data.hpp"

#include <cstdint>

#include "main/override_seed.kernel.cpp"

// THE LAYOUT CONTRACT AT THE SEAM. Rust hands these buffers over as
// `*const i32` / `*mut i32` because a flat 32-bit word is a type the C ABI can
// state, while `Vec3f` is not. The buffer's real element type is `Vec3f` on
// both sides (Rust `CVec<Vec3f>` allocated it), so the cast recovers the
// original object type rather than inventing one. These two assertions are what
// makes that recovery checkable rather than assumed: a widened component or a
// padded vector would fail the build here instead of silently reading every
// third component of someone else's vertex.
static_assert(sizeof(Vec3f) == 3 * sizeof(int32_t),
              "a position is three 32-bit components with no padding");
static_assert(alignof(Vec3f) == alignof(int32_t),
              "a position aligns like its 32-bit components");

extern "C" {

// WHY THESE THREE TAKE A RANGE AND ARE STILL CALLED ONCE.
//
// The `[begin, end)` shape matches every other entry point in this backend, so
// a later phase can partition them without changing the C++ side. Today the
// Rust caller passes the whole span in one call, deliberately: an index list
// arrives from a keyframe and this backend has no proof that it holds each
// vertex at most once. The linear seed is idempotent under a duplicate, but the
// ANGULAR seed reads `prev` and writes it back, so a duplicated index makes the
// result depend on the order the two writes land in, and under a parallel
// partition it is a data race outright. The work is a keyframe-sized subset of
// the vertices, so serial costs nothing measurable and buys determinism with no
// argument to get wrong.

// THE ABSOLUTE GATHER IS NOT HERE. It declares its own entry point beside its
// body (`src/kernels/main/override_seed.kernel.cpp`) and the range shim is
// rendered into `entrypoints/entries.cpp`.


} // extern "C"
