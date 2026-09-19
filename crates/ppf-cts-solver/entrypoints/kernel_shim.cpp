// File: entrypoints/kernel_shim.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The CPU backend's kernel entry points.
//
// This file is the LAUNCHER half of the CPU backend: a loop, a gather and a
// scatter, and no arithmetic of its own. Every value it produces comes from a
// neutral kernel body under ../src/kernels, which nvcc and the Metal shader compiler
// compile from the same bytes. If a float expression ever appears below, the
// architecture's one hard constraint has been crossed: a second implementation
// of a numerical kernel is a fork, not a port.
//
// What it includes is not the neutral source but the `cpp` RENDERING of it,
// which crates/ppf-cts-solver/build.rs produces with ppf-cts-compute/seam/kernelgen.py into
// $OUT_DIR/kernelgen and puts on this file's include path. That is the same
// mechanism ppf-cts-compute/cuda/Makefile and ppf-cts-compute/metal/Makefile
// use, and it is why nothing here spells an address space or an execution
// space: the renderer dropped both for this target. Nothing is written under
// src/, because two build scripts watch that tree recursively.
//
// The entry shape is a chunk `[begin, end)` rather than a single element, for
// the two reasons the measurements support: it is what lets the host compiler
// vectorize the loop, and it amortizes the rayon region, whose floor is about
// 3 us against a per-row cost far below that. It is NOT for FFI amortization; a
// call was measured at 1.33 ns.
//
// A handful of entries are NOT ranged, and each is a fold operator or a single
// verdict rather than an element pass: `aabb_join_abi`, `_merge_active` and
// `_overlap`, `pcg_alpha_entry` and `_beta`, `block_jacobi_invert_abi`.
// Wrapping a fold operator in a loop here would put the fold's SHAPE in this
// file, and this backend's determinism guarantee rests on that shape living in
// one place, `src/driver/reduce.rs`, independent of how rayon split the work. For
// the same reason no entry point below sums, minimizes or maximizes across its
// range: where a body produces a per-element term that has to be reduced (the
// PCG `rz` and `error`, the strain-limit and CCD toi values, the per-face vertex
// normal term), the range writes the terms and the caller folds them.
//
// TWO SAFETY RULES THE ENTRY POINTS CANNOT ENFORCE, both the caller's:
//
//   1. Anything named `_scatter_`, plus `fixed_csr_atomic_push_entry`,
//      reaches `compute::atomic_add`, which the host seam spells as a plain
//      read, add and write back on the premise that one thread runs the body.
//      Run those serially, or partition so no two ranges touch a destination. A
//      plain `+=` from two threads is a data race, not a fold-order difference,
//      and nothing here will say so. The rule survives generation unchanged:
//      the scatters and the pair-cache claim that moved to
//      `entrypoints/entries.cpp` carry it too, because an entry point covers
//      whatever range it is handed and cannot state how that range may be cut.
//   2. Where a body returns a validity flag, the range writes it per element and
//      the caller must honor it. A rejected hinge or edge still has bytes in its
//      force and Hessian slots, and they are not the zero contribution.
//
// WHAT IS WIRED AND WHAT IS NOT. `build.rs`'s KERNELS is the list, and its
// comment there records why ten of the neutral bodies are absent: three do not
// compile for a host at all (they are written against warp and threadgroup
// primitives `seam_host.h` deliberately leaves undefined), and seven serve
// capabilities `src/driver/refusal.rs` refuses by name. Several bodies below have
// no Rust caller yet and are staged for the driver; that is deliberate, and a
// body wired here still costs compile time, so nothing is added without a use
// in view for it.

// data.hpp first, and only here: it declares Vec3f, Mat3x3f and their siblings,
// and it is the one header that reaches the backend seam (seam/seam.hpp),
// which supplies the seam names the bodies below call. A kernel body includes
// nothing of its own beyond what the renderer rewrote into an absolute path.
//
// IT IS NAMED BY A RELATIVE PATH, AND src/kernels IS DELIBERATELY NOT ON THE INCLUDE
// PATH. A rendering sits at the same relative path under $OUT_DIR/kernelgen that
// its neutral source has under src/kernels, so an -I on both would resolve every
// `#include "solver/spmv.kernel.cpp"` below to whichever came first. Compiling
// the neutral source instead of the rendering is not a link error: g++ merely
// warns that it is ignoring the `[[seam::device]]` attributes and produces a
// translation unit that happens to work, so the mistake would be silent on this
// backend and would go on hiding whatever the renderer is there to catch.
#include "../src/kernels/data.hpp"

// Four more shared headers, by the same relative spelling and for the same
// reason: a body reaches them by name and this translation unit has to have
// them already declared. They are ordinary headers, not kernel bodies, so the
// renderer never sees them and there is no rendering to confuse them with.
//
//   * eigsolve.hpp   the symmetric 2x2 and 3x3 eigensolvers the two
//                    eigenanalysis bodies call as `linalg::eig::symm*`.
//   * cubic / quadratic / logarithm  the three barrier shapes
//                    `barrier/contact_barrier.kernel.cpp` dispatches over. All
//                    three are required: `barrier.hpp` is not a substitute,
//                    because the body names the three namespaces directly.
#include "../src/kernels/linalg/eigsolve.hpp"

#include "../src/kernels/barrier/cubic.hpp"
#include "../src/kernels/barrier/logarithm.hpp"
#include "../src/kernels/barrier/quadratic.hpp"

#include <cstddef>
#include <cstdint>

// The position arrays and the dense-matrix pack are handed across the boundary
// as the flat scalar arrays Rust already owns, so the two reinterpretations
// below have to be exactly reinterpretations. `SMat` is documented as POD
// `T[R*C]` at natural alignment and a position component is one 32-bit word,
// but a layout change on either side would otherwise be a silent misread rather
// than a build failure.
static_assert(sizeof(float) == sizeof(int32_t),
              "a position component must be one 32-bit word for the int32 "
              "array views below to be reinterpretations, not conversions");
static_assert(sizeof(Vec3f) == 3 * sizeof(int32_t),
              "Vec3f must be three packed 32-bit components");
static_assert(sizeof(Mat3x3f) == 9 * sizeof(float),
              "SMat must be packed column-major storage with no padding");
static_assert(alignof(Mat3x3f) == alignof(float),
              "SMat must carry natural alignment, not a vector alignment");

// The two views every entry point below uses to reach the caller's arrays.
// They compute nothing; they are the reinterpretation the static_asserts above
// license, named once so no call site spells a cast.
static inline const Vec3f *position_abi(const int32_t *p) {
    return reinterpret_cast<const Vec3f *>(p);
}
static inline Vec3f *position_mut_abi(int32_t *p) {
    return reinterpret_cast<Vec3f *>(p);
}

// The same reinterpretation for the dense pack. `M` is an SMat, whose whole
// storage is one `float[R * C]` in column-major order, so an array of them and a
// flat float array with R * C floats per element are the same bytes. The two
// assertions are what makes that a checked claim rather than a remembered one,
// and they are inside the helper so every use site is covered by construction.
template <class M> static inline const M *mat_abi(const float *p) {
    static_assert(sizeof(M) == sizeof(float) * M::Size,
                  "an SMat must be exactly its elements, with no padding");
    static_assert(alignof(M) == alignof(float),
                  "an SMat must carry natural alignment");
    return reinterpret_cast<const M *>(p);
}
template <class M> static inline M *mat_mut_abi(float *p) {
    static_assert(sizeof(M) == sizeof(float) * M::Size,
                  "an SMat must be exactly its elements, with no padding");
    static_assert(alignof(M) == alignof(float),
                  "an SMat must carry natural alignment");
    return reinterpret_cast<M *>(p);
}

// And the same for the small INDEX vectors (Vec2u, Vec3u, Vec4u, Vec6u), which
// are the same SMat storage over `unsigned`. Kept separate from the float form
// rather than folded into it: the two happen to have the same element width on
// every target this builds for, so one helper would type-check a float array
// handed to an index parameter and say nothing.
template <class M> static inline const M *uvec_abi(const uint32_t *p) {
    static_assert(sizeof(M) == sizeof(uint32_t) * M::Size,
                  "an index vector must be exactly its elements");
    static_assert(alignof(M) == alignof(uint32_t),
                  "an index vector must carry natural alignment");
    return reinterpret_cast<const M *>(p);
}

// ---------------------------------------------------------------------------
// The sparse matvec.
// ---------------------------------------------------------------------------

#include "solver/spmv.kernel.cpp"

extern "C" {

// THE WHOLE-ROW APPLY IS NOT HERE, AND NEITHER IS ITS ARGUMENT RECORD. Six
// base pointers, one thread index and one `Vec3f` out, so
// `fixed_csr_apply_row` declares its own entry point beside the body
// (`src/kernels/solver/spmv.kernel.cpp`) and the range shim is rendered into
// `entrypoints/entries.cpp`. No struct of raw pointers goes with it either: a
// hand-written one would need a matching record in `src/driver/spmv.rs`, two
// declarations that can disagree, while a generated record is one declaration
// both sides take. Each row writes only its own three floats, so a partition by
// row is disjoint by construction and needs no atomic; the kernel table's
// `Scatter::Disjoint` is where that lives.

// The bare 3x3 block matvec, exposed so a Rust test can compare against the
// shared body directly rather than against a second implementation of it.
void mat3_mul_abi(const float *matrix, const float *vector, float *out) {
    const Vec3f r = mat3_mul(matrix, vector);
    out[0] = r[0];
    out[1] = r[1];
    out[2] = r[2];
}

void mat3_transpose_mul_abi(const float *matrix, const float *vector,
                                float *out) {
    const Vec3f r = mat3_transpose_mul(matrix, vector);
    out[0] = r[0];
    out[1] = r[1];
    out[2] = r[2];
}

} // extern "C"

// ---------------------------------------------------------------------------
// Vector ops. Same shape as above: a loop around a shared body, no arithmetic.
//
// The bodies are templates over the element type. Every entry point here
// instantiates them on `float` and only on `float`: widening at the seam is one
// of the routes float64 reaches a kernel, and this backend has no SASS guard to
// catch it after the fact.
// ---------------------------------------------------------------------------

#include "primitives/vec_ops.kernel.cpp"

extern "C" {

// `vec_copy_entry`, `_vec_add_scaled_range`, `_vec_combine_range` and
// `_vec_fill_range` were here and are GENERATED now, from the
// [[seam::args]] [[seam::entry]] declarations beside their bodies in
// primitives/vec_ops.kernel.cpp, into entrypoints/entries.cpp, so this block holds no
// entry point of its own. The bodies stay included here because the other
// entry points below call them.

} // extern "C"

// ---------------------------------------------------------------------------
// Block-Jacobi. The inverse comes from the shared body; the VERDICT is returned
// rather than trapped here, because raising is the backend's job and Rust raises
// through a panic mapped to a CrashKind rather than through a C++ assert.
// ---------------------------------------------------------------------------

#include "solver/block_jacobi.kernel.cpp"

extern "C" {

// Inverts one 3x3 diagonal block. Returns 1 when the block was usable and 0 when
// the assembly upstream is broken, in which case `inverse` is NOT written and
// the caller must not use it.
int block_jacobi_invert_abi(const float *block, float *inverse,
                                float *lambda_max_out) {
    Mat3x3f m;
    for (unsigned c = 0; c < 3; ++c) {
        for (unsigned r = 0; r < 3; ++r) {
            m(r, c) = block[3 * c + r];
        }
    }
    const BlockJacobiInverse result = block_jacobi_invert(m);
    *lambda_max_out = result.lambda_max;
    if (!result.valid) {
        return 0;
    }
    for (unsigned c = 0; c < 3; ++c) {
        for (unsigned r = 0; r < 3; ++r) {
            inverse[3 * c + r] = result.inverse(r, c);
        }
    }
    return 1;
}

// THE APPLY IS NOT HERE. `z = P^-1 r` is one 3x3 block times one vector per
// row, which is `mat3_mul` over a range, so that body declares its own
// entry point beside itself (`src/kernels/solver/spmv.kernel.cpp`) and the range
// shim is rendered into `entrypoints/entries.cpp`. The declaration carries the
// row's geometry, a stride of nine floats for the block against three for the
// vector, which is where that addressing belongs rather than in a hand-written
// loop.

} // extern "C"

// ---------------------------------------------------------------------------
// CSR row bookkeeping: the pure index algorithms.
//
// These are the heapsort, bisect, merge and compaction that Rust must CALL
// rather than reimplement. They are not physics, but they are kernels by the
// rule that matters: they compute values (an ordering, a compacted run length),
// and a second copy would be a second set of behavior in the structure the whole
// linear solve reads. The compaction in particular rests on an invariant whose
// violation leaves the row NUMERICALLY correct and merely 40 seconds slower per
// step, so nothing would report the divergence.
// ---------------------------------------------------------------------------

#include "csrmat/dynamic_csr.kernel.cpp"

extern "C" {

void sort_pattern_abi(uint32_t *pattern, uint32_t count) {
    dynamic_csr_sort_pattern(pattern, count);
}

uint32_t find_sorted_abi(const uint32_t *pattern, uint32_t count,
                             uint32_t key) {
    return dynamic_csr_find_sorted(pattern, count, key);
}

void merge_runs_abi(const uint32_t *a, uint32_t na, const uint32_t *b,
                        uint32_t nb, uint32_t *out) {
    dynamic_csr_merge_runs(a, na, b, nb, out);
}

// Compacts a row in place, returning the surviving entry count. `index` and
// `value` are the row's parallel arrays; `carried` is where the carried run ends
// and the appended run begins. `appended_begin_out` reports where the surviving
// carried entries end, which the caller needs to keep both runs ascending.
uint32_t row_dedupe_abi(uint32_t *index, float *value, uint32_t count,
                            uint32_t carried, uint32_t *appended_begin_out) {
    // The C ABI keeps a POINTER, which is what a foreign caller can spell; the
    // neutral body takes a thread-space reference, because MSL has separate
    // address spaces and the device-side caller passes a local.
    return dynamic_csr_finalize(index, reinterpret_cast<Mat3x3f *>(value),
                                    count, carried, *appended_begin_out);
}

} // extern "C"

// ---------------------------------------------------------------------------
// The FIXED-pattern CSR: the two scalar helpers the driver calls directly.
//
// The block PUSH is not here. It is a generated entry point, declared beside
// its body in `csrmat/fixed_csr.kernel.cpp`, and its per-block verdict rides
// `[[seam::scatter]]`: the body returns false for a block whose (i, j) is not in
// the sparsity pattern and every CUDA caller ignores it, which is how the
// rod-bend stencil bug shipped an indefinite matrix. A generated scatter stores
// that return value by construction, so the verdict does not depend on a
// launcher remembering to write it.
// ---------------------------------------------------------------------------

#include "csrmat/fixed_csr.kernel.cpp"

extern "C" {

// The slot holding block (i, j), or 0xFFFFFFFF when the pattern has none.
// `transpose_out` reports that the stored block is the transpose of the one
// asked for, which is how the upper-triangle-only storage answers a lower query.
uint32_t fixed_csr_find_abi(const uint32_t *index, const uint32_t *offset,
                                uint32_t row_count, uint32_t i, uint32_t j,
                                int *transpose_out) {
    const FixedCsrLookup lookup =
        fixed_csr_find(index, offset, row_count, i, j);
    *transpose_out = lookup.transpose ? 1 : 0;
    return lookup.slot;
}

// Reads block (i, j) into nine column-major floats. A pair the pattern does not
// carry reads as the zero block, which is what the matrix means by its absence.
void fixed_csr_read_abi(const uint32_t *index, const uint32_t *offset,
                            const float *value, uint32_t row_count, uint32_t i,
                            uint32_t j, float *out) {
    const Mat3x3f block =
        fixed_csr_read(index, offset, value, row_count, i, j);
    for (unsigned c = 0; c < 3; ++c) {
        for (unsigned r = 0; r < 3; ++r) {
            out[3 * c + r] = block(r, c);
        }
    }
}

} // extern "C"

// ---------------------------------------------------------------------------
// PCG scalars and the fused update.
//
// The two scalar verdicts carry the curvature-noise rule: a `p^T A p` below the
// round-off of its own sum is unresolvable rather than negative, so the body
// classifies it and the caller truncates instead of aborting. Reproducing that
// classification in Rust would put the solver's abort decision in two places.
//
// `pcg_update3_entry` writes the per-row `rz` and `error` terms instead
// of summing them. The sum is a reduction, and this backend's determinism rests
// on the reduction having ONE fixed shape independent of the thread count, which
// is src/driver/reduce.rs's contract; a chunk-local sum here would be a second
// shape that moves with the partition.
// ---------------------------------------------------------------------------

#include "solver/pcg.kernel.cpp"

extern "C" {

} // extern "C"

// ---------------------------------------------------------------------------
// THE NEWTON SEED IS NOT HERE. `compute_target` declares its own entry
// point beside the body (`src/kernels/main/target.kernel.cpp`), so the range
// shim is rendered into `entrypoints/entries.cpp`.
//
// The body remains a template over the fix-pin record, because MSL needs the
// address space on the pointer and the other backends still deduce it; the
// concrete instantiation on `FixPair`, the same repr(C) record Rust's data.rs
// mirrors, is named once beside the declaration rather than here.
//
// Positions cross as flat triples and are reinterpreted, never converted: the
// generated entry gathers them as `Vec3f`, so no reinterpreting helper is
// needed on this side at all.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// LBVH: Morton codes and the quantizer. A Morton code decides tree topology, so
// two implementations would be two trees finding contacts in two orders.
// ---------------------------------------------------------------------------

#include "lbvh/lbvh.kernel.cpp"

extern "C" {

// MORTON CODES ARE NOT HERE. Three element gathers and one element scatter, so
// `lbvh_morton_from_bounds` declares its own entry point beside the body
// (`src/kernels/lbvh/lbvh.kernel.cpp`) and the range shim is rendered into
// `entrypoints/entries.cpp`. The quantizer's bounds are the scene bounds the
// caller reduced, and they cross as six scalars because every thread reads the
// same box.

uint32_t expand_bits_abi(uint32_t v) { return lbvh_expand_bits(v); }

uint32_t morton_code_3d_abi(uint32_t x, uint32_t y, uint32_t z) {
    return lbvh_morton_code_3d(x, y, z);
}

} // extern "C"

// ---------------------------------------------------------------------------
// The two SVDs.
//
// Both are deterministic one-sided Jacobi with a FIXED number of unrolled
// sweeps and no convergence break, so the answer does not depend on how the
// work was scheduled. That property is why they are shared rather than taken
// from a host library: a library SVD would converge to a different iterate and
// every eigen-clamped Hessian downstream would differ.
//
// `_rv` is the reflection-corrected variant (a rotation, det = +1). The two are
// separate entry points because they are separate functions upstream, and a
// caller that wants one must not silently get the other.
// ---------------------------------------------------------------------------

#include "utility/svd3x2.kernel.cpp"
#include "utility/svd3x3.kernel.cpp"

extern "C" {

// NEITHER SVD IS HERE. All three factorizations declare their own entry points
// beside their bodies, `svd3x2` in `src/kernels/utility/svd3x2.kernel.cpp` and
// `svd3x3` with `svd3x3_rv` in `src/kernels/utility/svd3x3.kernel.cpp`, and
// the range shims are rendered into `entrypoints/entries.cpp`. The three outputs of
// each are element GATHERS rather than scatters: the body returns void and
// writes through references, so `buffer[index]` is the destination.
//
// The plain 3x3 form has no dispatch in this driver, which is a reason for its
// id to sit past `id::COUNT` rather than a reason to keep a hand-written
// launcher for it.

} // extern "C"

// ---------------------------------------------------------------------------
// Spectral force and Hessian, shells (2x2 in the material frame) and tets (3x3).
//
// These are where SPD-by-assembly is delivered for the elastic terms: the
// Hessian entry point eigen-decomposes in the singular-value basis and clamps
// each mode, so the block it returns is PSD by construction. The eps argument is
// the mode-separation floor the body needs, not a tolerance to widen.
// ---------------------------------------------------------------------------

// NOTHING FROM THIS SECTION IS COMPILED HERE ANY MORE, and the bodies are not
// included either: all four of the spectral entry points are element gathers
// into one element scatter, so each declares its own entry point beside its
// body (`src/kernels/eigenanalysis/{face,tet}_eigenanalysis.kernel.cpp`) and the
// range shims are rendered into `entrypoints/entries.cpp`.

// ---------------------------------------------------------------------------
// The vertex normal.
// ---------------------------------------------------------------------------

#include "utility/vertex_normal.kernel.cpp"

extern "C" {

// THE TWO DEFORMATION-GRADIENT ENTRIES ARE NOT HERE. Each reads the element's
// positions THROUGH its own index list, which the declaration form expresses as
// a `[[seam::through]]` buffer over an `[[seam::indices(N)]]` list with a
// `[[seam::bound]]` beside it, so both declare their own entry points beside
// their bodies (`src/kernels/utility/face_deformation.kernel.cpp` and
// `src/kernels/utility/tet_convert.kernel.cpp`) and the range shims are
// rendered into `entrypoints/entries.cpp`. Positions cross as flat 32-bit
// words, which is a reinterpretation of the same bytes and not an encoding.

// THE FOUR MATERIAL-FRAME CONVERTERS ARE NOT HERE. Two element gathers into
// one element scatter each, so all four declare their own entry points beside
// their bodies (`src/kernels/utility/{face,tet}_convert.kernel.cpp`) and the range
// shims are rendered into `entrypoints/entries.cpp`.

// THE SHAPE-FUNCTION GRADIENTS ARE NOT HERE. One element gather and one run of
// four `Vec3f` the body writes itself, so the declaration beside the body
// (`src/kernels/utility/tet_convert.kernel.cpp`) states that run as a stride of
// four and the range shim is rendered into `entrypoints/entries.cpp`.

// THE FINALIZE STAGE IS NOT HERE. One element gather into one element
// scatter, so it declares its own entry point beside the body
// (`src/kernels/utility/vertex_normal.kernel.cpp`) and the range shim is rendered
// into `entrypoints/entries.cpp`. The per-face term above stays hand-written: it
// reads its three positions through the face's index triple, which is an
// indirect gather and none of the three shapes an entry declaration expresses.

} // extern "C"

// ---------------------------------------------------------------------------
// Bending: the shell hinge and the rod vertex.
//
// Both are the true analytical force, and both are ALWAYS PSD-projected through
// the analytic eigensystem inside the body. There is no unprojected variant to
// select and none is exposed here.
//
// The shell stiffness is DIRECTIONAL: `sin2` is the squared sine of the angle
// from the hinge's shared edge to the UV warp axis, and a NEGATIVE value is the
// no-UV sentinel that makes the result isotropic. Passing a positive warp or
// weft alongside that sentinel is an authoring error the caller must catch; the
// body cannot, because both readings are legal float.
//
// The shell hinge's force and Hessian report validity per hinge in `ok`, from
// the entry point named below rather than from this file. A zero there is a
// degenerate hinge (a zero normal or a zero shared edge), which the caller must
// skip rather than scatter: the force and Hessian written for it are not
// meaningful.
// ---------------------------------------------------------------------------

#include "energy/model/rod_bend.kernel.cpp"
#include "energy/model/rod_bend_stiffness.kernel.cpp"
#include "energy/model/shell_bend.kernel.cpp"
#include "energy/model/shell_bend_stiffness.kernel.cpp"

extern "C" {

float shell_bend_directional_abi(float bend, float warp, float weft,
                                     float sin2) {
    return shell_bend_directional(bend, warp, weft, sin2);
}

// THE HINGE'S AREAL DENSITY IS NOT HERE. It reads the per-vertex mass and area
// arrays at the hinge's own four slots, so `shell_bend_areal_density_gathered`
// declares its own entry point beside the body
// (`src/kernels/energy/model/shell_bend_stiffness.kernel.cpp`) and the range
// shim is rendered into `entrypoints/entries.cpp`. The hinge is the entry's
// `[[seam::indices(4)]]` list and the two arrays are read `[[seam::through]]`
// it, which also gives the slots a BOUND: a hand-written loop subscripts both
// arrays with an unchecked index, and Metal answers an out-of-bounds read with
// 0.0 rather than a fault.

// THE HINGE STIFFNESS IS NOT HERE. Seven per-hinge floats in and one out, so
// `shell_bend_stiffness` declares its own entry point beside the body
// (`src/kernels/energy/model/shell_bend_stiffness.kernel.cpp`) and the range
// shim is rendered into `entrypoints/entries.cpp`.

// The (2, 1, 0, 3) permutation `dihedral_angle::remap` applies before any
// dihedral quantity reads a position. A hinge's four indices in that order are
// what every stage below takes.
// THE HINGE'S FORCE AND HESSIAN ARE NOT HERE. They read four positions through
// the hinge's own index list, which is `[[seam::indices(4)]]` over a
// `[[seam::through]]` buffer, so
// `shell_bend_force_hessian_verdict` declares its own entry point beside
// the body (`src/kernels/energy/model/shell_bend.kernel.cpp`) and the range
// shim is rendered into `entrypoints/entries.cpp`. The declaration also gives
// those four slots a BOUND, which a hand-written loop does not have: it
// subscripts the position array with an unchecked index, and Metal answers an
// out-of-bounds read with 0.0 rather than a fault.

float rod_bend_segment_average_abi(float first, float second) {
    return rod_bend_segment_average(first, second);
}

// THE ROD BENDING STIFFNESS IS NOT HERE, for the reason the shell form is not:
// `rod_bend_stiffness` declares its own entry point beside the body
// (`src/kernels/energy/model/rod_bend_stiffness.kernel.cpp`).

} // extern "C"

// ---------------------------------------------------------------------------
// Cross-stitch.
//
// Six slots: 0..2 the source triangle, 3..5 the target. A non-SOLID endpoint
// degenerates to {s, s, s} with weights {1, 0, 0} and takes the same path, so
// this entry point has no branch for it either.
//
// The gather is by slot rather than by element because that is the record's own
// shape (Stitch{Vec6u, Vec6f}), and the per-vertex ghat and offset are read at
// the same six indices.
// ---------------------------------------------------------------------------

#include "energy/model/stitch.kernel.cpp"

extern "C" {

} // extern "C"

// ---------------------------------------------------------------------------
// The barrier family and the elasticity-inclusive dynamic stiffness.
//
// `Barrier` crosses the boundary as its underlying integer, because an
// `enum class` is not a C type. The mapping is the enumerator order in data.hpp
// (0 cubic, 1 quadratic, 2 logarithmic) and the cast below is the only place it
// is spelled; Rust's mirror must not spell a second one.
//
// The stiffness body is a template over the contact ARITY, and the three
// arities the solver builds are the three instantiated here: 2 for point-point,
// 3 for point-triangle against a static, 4 for the vertex-face and edge-edge
// pairs. A fourth arity would be a new contact type, not a missing entry point.
// ---------------------------------------------------------------------------

#include "barrier/contact_barrier.kernel.cpp"
#include "barrier/contact_stiffness.kernel.cpp"

namespace {

} // namespace

extern "C" {

void barrier_gradient_entry(const float *gap, float ghat, float offset,
                                    uint32_t kind, float *gradient,
                                    uint32_t begin, uint32_t end) {
    const Barrier barrier = static_cast<Barrier>(kind);
    for (uint32_t i = begin; i < end; ++i) {
        gradient[i] = barrier_gradient(gap[i], ghat, offset, barrier);
    }
}

} // extern "C"

// ---------------------------------------------------------------------------
// Strain limiting: the shell and rod terms, and the two TOI probes.
//
// The TOI entries return one value PER ELEMENT and never a minimum. Taking the
// minimum is a reduction, and this backend's determinism rests on the reduction
// shape being fixed by src/driver/reduce.rs rather than by how the range was cut.
//
// EVERY LIMIT IS A PER-ELEMENT ARRAY, not a scalar, and that is not a
// generalization for its own sake: `strainlimit` is a MATERIAL parameter, so one
// range covers faces whose limits differ, and a scalar could describe only a
// scene authored with a single material. A non-positive entry is how the caller
// spells "this element is not in the term this iteration", which is the same
// inert-seed shape the elastic layers use: the CUDA dispatch's other gates
// (`fixed`, `rest_excluded`, and for a rod a non-positive `initial_length`) are
// folded into the seed by the caller, and every entry below writes a fully
// defined zero for such an element rather than leaving the previous iteration's
// bytes in its slots.
//
// TWO DIFFERENT LIMITS LIVE IN ONE FACE and swapping them is a scale error no
// shrink-free scene can see; `strainlimiting/shell_strain.kernel.cpp` states
// the same distinction beside the two bodies that take them. The barrier's own
// ghat is the AUTHORED `strainlimit`; the stiffness and both line searches
// measure against the SHRINK-CORRECTED value `shell_effective_strain_limit`
// returns. The two differ for any face whose `shrink_x` or `shrink_y` is below
// one, and the parameters below are named for which of the two they take.
//
// Nothing here may be relaxed to make a scene pass. A strain limit that
// truncates every step is a diagnosis (see the PCG denominator rule), not a
// number to loosen.
// ---------------------------------------------------------------------------

#include "strainlimiting/rod_strain.kernel.cpp"
#include "strainlimiting/shell_strain.kernel.cpp"
#include "strainlimiting/strain_toi.kernel.cpp"

extern "C" {

// NEITHER HALF OF THE STRAIN LIMITER'S COORDINATE IS HERE ANY MORE. The shift
// is `svd3x2_shifted` and the restore `shell_strain_restore_sigma`,
// both neutral bodies with generated entry points in
// `utility/svd3x2.kernel.cpp`, so the three compilers build them from one set
// of bytes rather than from a launcher only one of them reads. The round trip
// is still two passes and not one, for the fp32 reason that file states.

// THE BARRIER'S DERIVATIVE PAIR IS NOT HERE. It is
// `shell_strain_diff_table_gated` in
// `src/kernels/strainlimiting/shell_strain.kernel.cpp`, a neutral body with a
// generated entry point. The gate on a non-positive authored limit is in that
// body too, since a branch on a physical quantity belongs with the physics and
// not in a range shim.

float shell_effective_strain_limit_abi(float strain_limit, float shrink_x,
                                           float shrink_y) {
    return shell_effective_strain_limit(strain_limit, shrink_x, shrink_y);
}

// THE LARGEST SINGULAR VALUE IS NOT HERE. One element gather into one element
// scatter, so `shell_max_strain` declares its own entry point beside the
// body (`src/kernels/strainlimiting/strain_toi.kernel.cpp`) and the range shim is
// rendered into `entrypoints/entries.cpp`.

// The two SELECTIONS the `max_sigma` indicator is built from: the larger of a
// face's two singular values, and the smaller of its two shrink factors.
// Neither produces a new value, and the
// PRODUCT of the two is not taken here: the caller forms it through
// `element_add_scaled_entry`, which is `vec_add_scaled`, for the
// same reason the per-element mass scales go through it.
//
// A face outside the driver's dispatch gate reaches this with both shrink
// factors seeded to zero, so its product is zero and the max reduction ignores
// it, which is what that dispatch's cleared scratch does.
// THE STRETCH INDICATOR IS NOT HERE AT ALL. `shell_stretch_terms` declares
// its own entry point beside its body
// (`src/kernels/utility/face_deformation.kernel.cpp`, beside the gradient whose
// singular values it selects from), and the rod's own half declares one beside
// `rod_stretch_ratio` in `src/kernels/main/stretch.kernel.cpp`, beside the
// segment length it is formed from. The gate on a non-positive rest length is
// `rod_stretch_ratio_gated` in that same file, a branch on a physical quantity
// and so a branch that belongs in a body. Both range shims are rendered into
// `entrypoints/entries.cpp`.

// NEITHER TIME OF IMPACT IS HERE. The shell's and the rod's are
// `shell_strain_toi_gated` and `rod_strain_toi_gated` in
// `src/kernels/strainlimiting/strain_toi.kernel.cpp`, neutral bodies with
// generated entry points. Each reads its element's positions through an index
// list, which `[[seam::indices]]` and `[[seam::through]]` express, and each
// holds its gate on a non-positive limit inside the body, since a branch on a
// physical quantity belongs with the physics and not in a range shim.

} // extern "C"

// ---------------------------------------------------------------------------
// The push barrier and the friction model.
//
// Both are rank-structured PSD by construction: the push Hessian is a
// non-negative curvature times `n n^T`, and the friction Hessian is the
// tangential projection scaled by a non-negative lambda. Neither may be
// assembled without the other's sign discipline, and neither is clamped here.
// ---------------------------------------------------------------------------

#include "energy/model/friction.kernel.cpp"
#include "energy/model/push.kernel.cpp"

extern "C" {

// THE FOUR PUSH-BARRIER ENTRY POINTS ARE NOT HERE. Each is element gathers
// into one element scatter, so each declares its own entry point beside its
// body (`src/kernels/energy/model/push.kernel.cpp`) and the range shims are
// rendered into `entrypoints/entries.cpp`. The body stays included above, because
// `contact/analytic_contact.kernel.cpp` below calls all four.

// THE FRICTION EVALUATION IS NOT HERE. Its ten buffers are all element
// gathers, the six outputs included, so it declares its own entry point beside
// the body (`src/kernels/energy/model/friction.kernel.cpp`) and the range shim is
// rendered into `entrypoints/entries.cpp`. The struct of ten raw pointers goes
// with it: a hand-written record is a mirror no other backend shares.

} // extern "C"

// ---------------------------------------------------------------------------
// The analytic collider contact.
//
// This is the branch that runs EVEN WHEN `disable-contact` is set, because an
// invisible wall or sphere is not a mesh pair: it needs no BVH and no dynamic
// CSR, which is what makes it the cheapest way to exercise the barrier, the
// stiffness and the line search.
//
// `analytic_grain_schur` is deliberately NOT exposed. It is the SAND
// rolling degree of freedom's condensation, and the grain angular DOF is one of
// the capabilities this backend does not carry, so an entry point for it would
// be reachable code for a refused feature.
// ---------------------------------------------------------------------------

#include "contact/analytic_contact.kernel.cpp"

extern "C" {

// THE FRICTION COMBINER IS NOT HERE. Two element gathers into one element
// scatter, so it declares its own entry point beside the body
// (`src/kernels/contact/analytic_contact.kernel.cpp`) and the range shim is
// rendered into `entrypoints/entries.cpp`. An entry point names the body it wraps,
// so the symbol is `combine_friction_values_entry`.

struct AnalyticContactArgs {
    const float *local_hessian;   // 9 per contact
    const float *slip;            // 3 per contact
    const float *normal;          // 3 per contact
    const float *signed_distance; // 1 per contact
    const float *physical_gap;    // 1 per contact
    const float *ghat;            // 1 per contact
    const float *friction;        // 1 per contact
    const float *mass;            // 1 per contact
    const uint8_t *kinematic;     // 1 per contact
    const uint8_t *with_friction; // 1 per contact
    float *force;                 // 3 per contact
    float *hessian;               // 9 per contact
    float *friction_hessian;      // 9 per contact
    float *friction_gradient;     // 3 per contact
    float *out_normal;            // 3 per contact
    float *stiffness;             // 1 per contact
    uint8_t *valid_gap;           // 1 per contact
};

} // extern "C"

// ---------------------------------------------------------------------------
// Contact assembly: the slot lookup and the barycentric extension.
//
// The extension spreads one contact's 3-vector force and 3x3 Hessian over the N
// vertices its weights name, as the congruence `w_i w_j H`. A congruence cannot
// introduce a negative mode, so the extended block is PSD whenever the 3x3 one
// is, which is what keeps SPD-by-assembly true through the widening.
// ---------------------------------------------------------------------------

#include "contact/contact_assembly.kernel.cpp"

namespace {

template <unsigned N>
void extend_contact_entry(const float *weight, const float *force,
                                 const float *hessian, float *extended_force,
                                 float *extended_hessian, uint32_t begin,
                                 uint32_t end) {
    const SVecf<N> *w = mat_abi<SVecf<N>>(weight);
    const Vec3f *f = mat_abi<Vec3f>(force);
    const Mat3x3f *h = mat_abi<Mat3x3f>(hessian);
    SMatf<3, N> *ef = mat_mut_abi<SMatf<3, N>>(extended_force);
    SMatf<3 * N, 3 * N> *eh =
        mat_mut_abi<SMatf<3 * N, 3 * N>>(extended_hessian);
    for (uint32_t i = begin; i < end; ++i) {
        extend_contact_force_hessian<N>(w[i], f[i], h[i], ef[i], eh[i]);
    }
}

} // namespace

extern "C" {

// THE SLOT LOOKUP IS NOT HERE. The CSR pattern reaches the body as base
// pointers, the pair as two element gathers and the resolved slot as one
// element scatter, so it declares its own entry point beside the body
// (`src/kernels/contact/contact_assembly.kernel.cpp`) and the range shim is
// rendered into `entrypoints/entries.cpp`. 0xFFFFFFFF still marks a pair below the
// diagonal, which the upper-triangle storage accounts for through its
// transpose and has no slot of its own.

void extend_contact2_entry(const float *weight, const float *force,
                                   const float *hessian,
                                   float *extended_force,
                                   float *extended_hessian, uint32_t begin,
                                   uint32_t end) {
    extend_contact_entry<2>(weight, force, hessian, extended_force,
                                   extended_hessian, begin, end);
}

void extend_contact3_entry(const float *weight, const float *force,
                                   const float *hessian,
                                   float *extended_force,
                                   float *extended_hessian, uint32_t begin,
                                   uint32_t end) {
    extend_contact_entry<3>(weight, force, hessian, extended_force,
                                   extended_hessian, begin, end);
}

void extend_contact4_entry(const float *weight, const float *force,
                                   const float *hessian,
                                   float *extended_force,
                                   float *extended_hessian, uint32_t begin,
                                   uint32_t end) {
    extend_contact_entry<4>(weight, force, hessian, extended_force,
                                   extended_hessian, begin, end);
}

} // extern "C"

// ---------------------------------------------------------------------------
// The detect-once pair cache.
//
// `count` is a CLAIM atomic: it exists so each recorded pair gets a slot no
// other pair got, and no row partition can deliver that, so this one is not
// made safe by the scatter rule the rest of the file relies on. Under a
// parallel dispatch the host seam's plain read-add-write races.
//
// The body counts EVERY pair, whether or not it fit, and sets `overflow` when
// one did not. That is deliberate: the requirement is then known even when it
// was not met, which is what lets the capacity be sized from measurement rather
// than from a constant. Do not read a full cache as the pair count.
// ---------------------------------------------------------------------------

#include "contact/pair_cache.kernel.cpp"

extern "C" {

// THE RECORDER IS NOT HERE. The cache, its counter and its overflow flag reach
// the body as base pointers and the two endpoint arrays as element gathers, so
// it declares its own entry point beside the body
// (`src/kernels/contact/pair_cache.kernel.cpp`) and the range shim is rendered into
// `entrypoints/entries.cpp`. The claim is still not made safe by a partition, and
// that is still the caller's to keep.

} // extern "C"

// ---------------------------------------------------------------------------
// Bounding volumes: the leaf builders, the fold operator, and the overlap test.
//
// `AABB` crosses as itself rather than as loose floats. It is a box of two
// `Vec3f` corners plus an active flag, padded to a 32-byte stride so a random
// node load is one sector, and taking it apart at the boundary would cost that
// layout.
//
// THE FOLD IS NOT HERE. `join` and `merge_active` are exposed as the single
// binary operator they are, and the propagate up the tree is the caller's,
// because a fold's SHAPE is what decides determinism on this backend and it
// belongs in one place with the other reductions. Both operators are min/max
// only, so the result is order-independent anyway; the discipline still holds
// because it is the discipline, not because this case needs it.
// ---------------------------------------------------------------------------

#include "contact/aabb.kernel.cpp"

static_assert(sizeof(AABB) == 32, "AABB is padded to a 32-byte node stride");
static_assert(offsetof(AABB, min) == 0, "AABB layout: min first");
static_assert(offsetof(AABB, max) == 12, "AABB layout: max after min");
static_assert(offsetof(AABB, active) == 24, "AABB layout: active after max");

extern "C" {

void aabb_join_abi(const AABB *a, const AABB *b, AABB *out) {
    *out = aabb_join(*a, *b);
}

void aabb_merge_active_abi(const AABB *a, const AABB *b, AABB *out) {
    *out = aabb_merge_active(*a, *b);
}

int aabb_overlap_abi(const AABB *a, const AABB *b) {
    return aabb_overlap(*a, *b) ? 1 : 0;
}

} // extern "C"

// ---------------------------------------------------------------------------
// The bitonic sorting-network step.
//
// The step is shared because it defines the COMPARATOR, and the comparator is
// what fixes the total order the Morton sort produces; two orders are two trees.
// The schedule around it (which subsequence and stride, in which order) is the
// caller's, as is the choice to run a different sort algorithm entirely, so long
// as the resulting order is this one.
// ---------------------------------------------------------------------------

#include "lbvh/bitonic.kernel.cpp"

extern "C" {

// THE STEP IS NOT HERE. Both arrays reach the body as base pointers and the
// thread index is forwarded, because a comparator addresses its own element and
// its partner's and the partner is the body's arithmetic, so it declares its
// own entry point beside the body (`src/kernels/lbvh/bitonic.kernel.cpp`) and the
// range shim is rendered into `entrypoints/entries.cpp`.

} // extern "C"

// ---------------------------------------------------------------------------
// Plasticity: the creep of the REST shape.
//
// These update the rest state in place and are handed the RAW substep dt, not
// the TOI-shortened local one. That is what the CUDA path does and the
// difference is visible in the result, so it is reproduced rather than tidied.
//
// `changed` is reported per element because a run that creeps must write its
// rest shape to the per-frame plastic file and a run that does not must not
// write one at all; the flag is what separates the two.
//
// SEAM_MOVE: the three entry points left here are hand-written rather than
// generated. Each returns a bool the launcher widens to a byte, and a generated
// entry writes the body's return value at the thread index without converting
// it, so the widening has nowhere to live. Each is otherwise a loop, a gather
// and a write around a neutral body and carries no arithmetic of its own.
// ---------------------------------------------------------------------------

#include "plasticity/plasticity.kernel.cpp"

// NOTHING FROM THE CREEP IS HERE ANY MORE. Every pass declares its own entry
// point beside its body in `src/kernels/plasticity/plasticity.kernel.cpp`: the
// creep rate is one per-element float and the step in and one float out, the
// two inverse-rest passes read their positions through the element's own index
// list, which `[[seam::indices(N)]]` and `[[seam::through]]` name, and the
// three creep passes go through the `plasticity_creep_*` wrappers. Those
// three were the last to move: each launcher widened the body's `bool` to a
// byte, and a byte is not a width a record field may address. The verdict is an
// `unsigned` now, which is what the tet material table's `accepted` and the
// face table's `dispatch` already were, so the widening is in a neutral body
// and the three per-element verdicts in this tree agree on one width.

// ---------------------------------------------------------------------------
// The element scatters.
//
// ONE IS LEFT HERE, and it is the arity-6 Hessian scatter. The four force
// scatters and the arity-2, arity-3 and arity-4 Hessian scatters each declare
// an entry point beside their body, so their range shims are rendered into
// `entrypoints/entries.cpp` from the same declaration that produces the CUDA
// `__global__`, the MSL `kernel void` and the Rust twin the driver fills. What
// keeps this one here is its slot table: a stride of thirty-six, which the
// arena's handle can express, against a `SMatf<18, 18>` gather that
// `data.hpp` gives no alias to, and a template argument list in a parameter
// type is a construct the transcompiler refuses by name.
//
// A SCATTER, AND THE HOST SEAM SPELLS ITS ATOMIC AS A PLAIN READ, ADD AND WRITE
// BACK, on the premise that one thread runs the body. That premise is the whole
// safety argument, and it is the CALLER'S to keep: run a range serially, or
// partition so that no two ranges reach the same destination. A plain `+=` from
// two rayon threads is a data race, not merely a different fold order, and it
// will not announce itself. The destination is a CSR value slot, and two
// stitches can share a block.
//
// A 0xFFFFFFFF slot marks a block the upper-triangle storage does not hold, and
// the body skips it. The per-component zero test and the column-major element
// order inside it are the ones the CSR's own push uses, so routing a Hessian
// through the slot table does not change the float fold order.
// ---------------------------------------------------------------------------

#include "utility/stitch_scatter.kernel.cpp"

extern "C" {

} // extern "C"
