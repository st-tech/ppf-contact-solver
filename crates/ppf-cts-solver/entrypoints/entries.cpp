// File: entrypoints/entries.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The CPU backend's GENERATED kernel entry points.
//
// Nothing here is written by hand except the include list. Each line below
// names one artifact `ppf-cts-compute/seam/kernelgen.py --target cpp --emit entry`
// rendered from an `[[seam::args]] [[seam::entry]]` declaration beside its
// neutral body, and that one declaration also produces the CUDA `__global__`,
// the MSL `kernel void` and the Rust `#[repr(C)]` twin `src/driver/kernels.rs`
// fills. An entry point is an argument record plus a thread index plus a call
// into the neutral body, all three mechanical, so a hand-written one is a
// mirror pair with nothing linking its halves. An entry point is therefore
// generated and never hand-written, and the launchers still spread across
// `kernel_shim.cpp`, `shim_step.cpp`, `shim_contact.cpp` and
// `shim_override_seed.cpp` are what remains of the other way;
// `check-shared-wiring.py` rule 9 counts them and holds the count down. They
// group by the shape that blocks each: most need the BODY changed rather than
// this file extended, a
// few need a name in the shared type vocabulary or an arena that can promise a
// pointee's own stride, and a handful are blocked by nothing but a CALL SITE
// that does not go through the `Device` seam.
//
// THIS IS THE ONLY TRANSLATION UNIT THAT MAY INCLUDE THEM. A rendering DEFINES
// its `ppf_<stem>_entry`, and `#pragma once` does not reach across
// translation units, so a second includer is a duplicate symbol at link. Each
// rendering says so in its own banner.
//
// The renderings live under `$OUT_DIR/kernelgen`, mirroring their path under
// `src/kernels`, and `build.rs` puts that root on this file's include path.
// Nothing is written under `crates/ppf-cts-solver/src`, because two build
// scripts watch that tree recursively.

// data.hpp first, and for the reason `kernel_shim.cpp` states: it declares the
// shared types and it is the one header that reaches the backend seam
// (seam/seam.hpp), which supplies the seam names a neutral body calls.
// A generated entry adds `arena_handle.hpp` and the body rendering itself, by
// absolute path and by its own directory, so this file names neither.
#include "../src/kernels/data.hpp"

// The symmetric 2x2 and 3x3 eigensolvers the two spectral Hessian bodies
// reach. `data.hpp` does not carry them, so a translation unit that compiles
// those bodies must name this header itself, exactly as `kernel_shim.cpp`
// does; without it the bodies fail at `linalg::eig::symm2x2`.
#include "../src/kernels/linalg/eigsolve.hpp"

// Per-element vector arithmetic: copy, add-scaled and combine. The fourth
// entry this file's declaration set carries, `vec_combine_indirect`, is
// rendered with them and is compiled here for the same reason CUDA compiles
// it, but the CPU driver does not dispatch it: its PCG holds alpha and beta as
// host scalars, so it has no coefficient in device memory to point at.
#include "primitives/vec_ops.entry.cpp"
#include "primitives/radix.entry.cpp"
#include "primitives/scan_levels.entry.cpp"
#include "primitives/reduce_bounds.entry.cpp"
#include "primitives/reduce_scalar.entry.cpp"

// The block-diagonal apply, `z = P^-1 r`, which is one 3x3 block times one
// vector per row, and the FIXED matrix's whole-row apply beside it: both are
// declared in `src/kernels/solver/spmv.kernel.cpp` and render into this one
// artifact. The first entry is declared over `mat3_mul` because that is
// what the pass computes; the row's geometry (a stride of nine floats against
// three) is in the declaration. The second takes its six arrays as BASE
// pointers, because the row's own slot range is the body's business.
#include "solver/spmv.entry.cpp"

// The rescaled Newton step: one position in, one position out, on the same
// buffer. The gather and the scatter name the same parameter, which is the
// in-place element update a launcher spells `x[i] = f(x[i], ...)`.
#include "main/position_step.entry.cpp"

// The arity-1 force scatter. SERIAL BY CONTRACT, and the contract lives in the
// kernel table's `Scatter::Atomic` rather than here: `compute::atomic_add` is a
// plain read, add and write back on the host seam, so two threads folding
// contributions that land on one vertex would be a data race. A generated
// entry point covers whatever range it is handed and says nothing about how
// that range may be cut, which is why the declaration alone cannot make one
// parallel.
#include "utility/vertex_scatter.entry.cpp"

// The per-vertex magnitude of the Newton search direction. `direction` stays a
// BASE pointer because the body addresses its own triple at `3 * vert`, so the
// thread index is forwarded rather than spent on a gather.
#include "main/dx_norm.entry.cpp"
#include "main/dump_linsys.entry.cpp"
#include "schwarz/schwarz.entry.cpp"

// The accept lerp: commit the fraction of the proposed motion the CCD line
// search certified. Two positions in, one position out.
#include "main/position_accept.entry.cpp"

// The PCG dot product's per-row term and the round-off bound it is judged
// against, in ONE pass over the two vectors. Both stay base pointers because
// the body addresses its own triple, and the two destinations are non-const
// gathers because a scatter carries one return value.
#include "solver/pcg.entry.cpp"

// One row of the block-Jacobi preconditioner, inverted through a floored
// symmetric eigendecomposition. Both arrays stay base pointers because the body
// addresses its own nine floats, and the verdict leaves through the diagnostic
// lane rather than a scatter: a block that is not positive definite is an
// assembly defect upstream, so the host discards the whole dispatch on it.
#include "solver/block_jacobi.entry.cpp"
#include "solver/translation_lock_check.entry.cpp"
#include "solver/translation_lock_frames.entry.cpp"
#include "solver/translation_lock_rows.entry.cpp"

// The Newton iterate's seed on every prescribed row: four element gathers and
// no scatter, because only a removed row has a closed-form correction and only
// a removed row is written. Both positions are read as `Vec3f`, and only their
// difference leaves the body.
#include "main/dx_seed.entry.cpp"

// The kinematic pin's rewind to the fraction of its scheduled step the clock
// was actually shortened to. One element gather of the pin's own record and no
// scatter: the body returns void and writes through the gathered element, and
// the gate on `kinematic` is inside the body, where a branch on what the scene
// contains belongs.
#include "main/rewind_fix.entry.cpp"

// The absolute-position gather the angular seed's pivot needs. One slot per
// element read through `[[seam::indices(1)]]`, the position at that slot as the
// body's only argument, and the triple it returns as the scatter.
#include "main/override_seed.entry.cpp"

// The start of a step: the incoming velocity, its squared speed and the
// vertex's distance from the origin, in ONE pass over the two position
// arrays. Three element gathers and three per-element destinations, none of
// them a scatter, because a scatter carries the body's one return value and
// this body writes all three on every path.
#include "main/velocity.entry.cpp"

// The `fix-xz` horizontal drag, position half: pull a risen vertex's x and z
// back toward its previous pose, leaving a DOF-removed vertex alone. Two
// positions and the DOF mask in, one position out, on the same buffer. The
// mask is read per element rather than spent on a launcher predicate, because
// a generated entry always writes what the body returns and the body returns
// the skipped vertex's own position.
#include "main/fix_xz_drag.entry.cpp"

// The Newton seed: where an unconstrained vertex would be at the end of the
// step under its own momentum and gravity, and exactly its prescribed position
// when the vertex is fix-pinned. Three element gathers into one element
// scatter. It carries its own name rather than the
// template's, which the shared-wiring census requires of two bodies in one
// file. The pin array stays a BASE pointer because
// the body wants the pin at `fix_index - 1`, and gravity is three scalars
// rather than a buffer because every thread reads the same vector.
#include "main/target.entry.cpp"

// The LBVH's Morton codes, from centroids already in SoA form. Three element
// gathers, the scene box as six scalars, one element scatter. A Morton code
// decides tree topology, so two statements of this arithmetic would be two
// trees finding contacts in two orders. The node-depth walk renders into the
// same artifact: it forwards the thread index to a body that climbs `parent`
// from that node, so the array is a base pointer and the addressing stays
// where the walk is.
#include "lbvh/lbvh.entry.cpp"

// The elastic pipeline's per-element stages, in the order the driver
// dispatches them. Each is element gathers into one element scatter, or, where the body
// returns void and writes its outputs through references, element gathers all
// the way: a gather hands the body `buffer[index]`, which is an lvalue, so a
// non-const gather is the write.
//
// The two SVDs. `svd3x3.entry.cpp` renders only the reflection-corrected form,
// which is the one this driver dispatches.
#include "utility/svd3x2.entry.cpp"
#include "utility/svd3x3.entry.cpp"

// The spectral force and Hessian, one pair per element arity.
#include "eigenanalysis/face_eigenanalysis.entry.cpp"
#include "eigenanalysis/tet_eigenanalysis.entry.cpp"

// The material-frame converters, which take a quantity in the element's own
// frame and its inverse rest shape to the world frame. `tet_convert` carries
// the tet deformation gradient beside them: that one reads its four positions
// THROUGH the element's index list, and the face twin below is the same shape.
#include "utility/face_convert.entry.cpp"
#include "utility/tet_convert.entry.cpp"

// The shell face's deformation gradient, an indirect gather over the face's
// three vertex slots, with the bound each slot is checked against carried in
// the record. The stretch indicator's two per-face selections render into the
// same artifact: the larger of the gradient's singular values and the smaller
// of the face's authored shrink factors, two element gathers in and two
// non-const element gathers out, because a scatter carries one return value.
#include "utility/face_deformation.entry.cpp"

// The material diff table, one entry per element arity: four element gathers
// into this element's own slot in two destinations, plus the verdict on its
// model id as the scatter. The verdict is the RETURN value because a scatter
// carries one and the table's two halves cannot both be it, and the caller must
// honor it: a table the dispatch did not write holds the zero fill, which is
// not a zero contribution.
//
// IT BRINGS `model/{arap,stvk,snhk,detsqr}.hpp` INTO THIS TRANSLATION UNIT.
// Those four declared their functions without `inline` until this entry existed
// and now say `SM_INLINE`, for the reason `model/fix.hpp` states in its own
// header: a definition without it is a duplicate symbol at link rather than a
// compile error, so the failure appears only once a second includer exists, and
// a generated entry point is what makes a second includer.
#include "energy/model/material_diff_table.entry.cpp"

// The BaraffWitkin membrane's staged pass: four element gathers into this
// face's own slot in two destinations. It runs AFTER the spectral stages and
// overwrites what they left, which is correct because a BaraffWitkin face's
// diff table is zero; a face naming any other model is left alone by the body's
// own early return.
#include "energy/model/baraffwitkin.entry.cpp"

// The two bending stiffnesses, one per element arity: per-element floats in,
// one float out. The shell form folds the directional warp and weft weights in
// here, which is what covers the lagged damping Hessian by construction.
#include "energy/model/rod_bend_stiffness.entry.cpp"
#include "energy/model/shell_bend_stiffness.entry.cpp"

// The shell hinge's exact bending force and PSD-projected Hessian, reading its
// four positions through the hinge's own index list. The verdict is the scatter
// and the force and the Hessian are non-const gathers, which is what a body
// that returns one value and writes two more through references renders as.
#include "energy/model/shell_bend.entry.cpp"
#include "energy/model/pdrd_rigid.entry.cpp"
#include "energy/model/pdrd_lock_projector.entry.cpp"

// The rod's turning angle and its bending pair, each reading the site's three
// nodes through the site's own index list. The angle is the body's return
// value, so it is the scatter; the force and the Hessian are two outputs, so
// they are non-const gathers, which is the write.
#include "energy/model/rod_bend.entry.cpp"

// The rod segment's Hookean stretch gradient and Hessian, reading its two
// positions through the edge's own index pair. It ASSIGNS both outputs rather
// than accumulating, so the damping entry below it is the accumulating half and
// the order of the two dispatches is load-bearing.
#include "energy/rod_force.entry.cpp"

// THE TET ELASTIC LAYER OF ONE ELEMENT, EVALUATED AND EMBEDDED. It is the one
// entry in this file that both COMPUTES a Newton contribution and PLACES it:
// the deformation gradient, the reflection-corrected factorization, the
// material diff table, the spectral force, the fused 12x12 spectral Hessian and
// the Rayleigh damping block stay in the thread, and what leaves are the four
// vertices' force rows and the sixteen 3x3 blocks the element pushes into the
// fixed matrix. The shape is one element per thread, with nothing staged
// through per-element arrays and no host transpose between the stages.
//
// The tet's four vertex slots arrive TWICE and neither arrival is spare: as the
// index list the entry reads the two position buffers through and checks
// against `vertex_count`, and as a gathered quadruple the body addresses the
// force rows and the CSR blocks with. An index list is not forwarded.
#include "energy/tet_force.entry.cpp"

// THE SHELL MEMBRANE AND PRESSURE LAYER OF ONE FACE, EVALUATED AND EMBEDDED,
// on the same terms as the solid one above and from the same reference block:
// `embed_face_force_hessian` runs elasticity and pressure as two sibling
// terms and embeds each itself. The deformation gradient, the SVD, the
// material diff table, the spectral force, the PSD-projected 6x6 Hessian, the
// material-frame conversion and the Rayleigh damping block stay in the thread,
// and what leaves are the three vertices' force rows and the nine 3x3 blocks
// the face pushes into the fixed matrix. Nothing is staged through per-element
// arrays and no host transpose sits between the stages.
//
// The face's three vertex slots arrive TWICE and neither arrival is spare, for
// the reason the tet's four do.
#include "energy/face_force.entry.cpp"

// The stretch indicator's rod half: two positions through the edge's own index
// pair, the initial length as a gather, the ratio as the scatter. The gate on a
// non-positive initial length is inside the body, where a branch on a physical
// quantity belongs, and a fixed segment reaches it with a zero length so one
// comparison serves both exclusions.
#include "main/stretch.entry.cpp"

// The three broad-phase leaf boxes, one per primitive arity, and the margin
// they share. Every buffer is a BASE POINTER because the tree's node entry is
// the primitive index PLUS ONE, so it cannot serve as a checked slot list; the
// box is the body's return value and reaches memory as a scatter. `AABB` is
// `alignas(32)` and crosses as a `[[seam::pod(32)]]` pointee.
#include "contact/aabb.entry.cpp"
#include "contact/aabb_traversal.entry.cpp"
#include "energy/model/face_pressure.entry.cpp"

// The cross-stitch's force and 18x18 Hessian, over its six barycentric slots.
// Six positions, six contact gaps and six offsets read THROUGH one six-slot
// index list, the weights as this element's own run, the gradient as a
// non-const gather, and the Hessian as a BASE POINTER: at 1296 bytes a gathered
// element would be copied into thread space on MSL and would not fit a device
// stack, while the 72-byte gradient is unremarkable.
#include "energy/model/stitch.entry.cpp"

// The plastic creep rate, `1 - exp(-plasticity * dt)`. `dt` is the same for
// every element, so it arrives in the record rather than through a buffer.
#include "plasticity/plasticity.entry.cpp"

// The one-sided push barrier's energy, curvature, gradient and Hessian. Each
// is per-contact gathers plus the one `ghat` the whole dispatch shares, into
// one element scatter.
#include "energy/model/push.entry.cpp"

// The friction model, ten per-contact buffers wide: the body returns void and
// writes its six outputs through references, so every one of them is an
// element gather.
#include "energy/model/friction.entry.cpp"

// The friction combiner, which resolves the two materials' coefficients at a
// contact under one rule per dispatch.
#include "contact/analytic_contact.entry.cpp"

// The area-weighted vertex normal's finalize stage. The per-face term beside
// it stays hand-written in `kernel_shim.cpp`: it reads its three positions
// through the face's own index triple, which is an indirect gather.
#include "utility/vertex_normal.entry.cpp"

// The contact assembly's slot lookup, which resolves one (row, column) pair to
// its slot in the FIXED pattern.
#include "contact/contact_assembly.entry.cpp"

// The four narrow-phase visitors, one per pair kind. Each turns a broad-phase
// candidate into an extended force and Hessian or into nothing, and each
// declares a `[[seam::diag]]` lane: the separation assert they carry is the
// penetration-free guarantee rather than instrumentation, so the channel is
// part of the entry's signature.
#include "contact/contact_narrow.entry.cpp"

// The three static-collision-mesh visitors. Each names only the arrays its own
// pass reads, which is why the collider side appears in three shapes: its
// faces for M2C, its vertices for C2M, its edges for edge-edge.
#include "contact/collision_narrow.entry.cpp"

// THE CCD LINE SEARCH, one dispatch per (query kind, tree) pair. Each walks a
// BVH with the ACCD advance as a per-hit device functor, folds a time of impact
// in a register and writes one slot of a per-primitive array; nothing is
// materialized between the traversal and the advance. No candidate pair list
// exists at all, so none is written, downloaded or swept on the host.
#include "contact/ccd_sweep.entry.cpp"

// THE FINAL PENETRATION GATE, in the same shape one stage later: one dispatch
// per query element, three of them over edges and one over surface vertices,
// each walking a BVH with the intersection tester as a per-hit device functor.
// Three things leave the device and every one is a scalar or a bounded array:
// a 4-byte claim counter, at most `capacity` records, and two per-element flag
// arrays.
//
// The claim is why these four rows are `Scatter::Claim` rather than `Disjoint`.
// Threads take numbered slots out of one counter, and a slot assignment is
// reproducible only in ascending order.
#include "contact/intersect_geometry.entry.cpp"

// The two per-vertex analytic-collider passes: the assembly and the swept
// line-search test. Both carry a `[[seam::diag]]` lane, the non-negative gap
// they assert being the penetration-free guarantee rather than instrumentation.
#include "contact/vertex_constraint.entry.cpp"

// One vertex's momentum row, with the air damper, the soft PULL pin and the two
// global drags that share it.
#include "main/momentum.entry.cpp"

// One torque group's frame: the centroid, the principal axis and the radius
// normalization every member of the group scales its own force by.
#include "energy/model/torque.entry.cpp"

// The SAND grain's three rows: the Schur condense, the recover and the
// post-solve integrate.
#include "energy/model/sand_rigid.entry.cpp"

// One step of the bitonic sorting network, the comparator that fixes the total
// order the Morton sort produces.
#include "lbvh/bitonic.entry.cpp"

// The three Hessian scatters, one per element arity: a run of slot ids, one
// element Hessian and the CSR value array. SERIAL BY CONTRACT, exactly as the
// arity-1 force scatter above, and for the same reason.
//
// The hinge and rod artifacts carry a second entry point each, the arity-4 and
// arity-2 force scatters, because a rendering covers a whole source file and
// those declarations sit beside these. The face and stitch force scatters are
// the two lines below, their own files having no Hessian entry: the face's is
// in `face_hessian_scatter.kernel.cpp`, and the stitch's 18x18 now has the name
// `Mat18x18f` in the shared type vocabulary, which is what a generated entry
// point's parameter type needs: the transcompiler splits a parameter list on
// its top-level commas and refuses a template spelling saying so.
#include "utility/face_hessian_scatter.entry.cpp"
#include "utility/hinge_scatter.entry.cpp"
#include "utility/collision_window.entry.cpp"
#include "utility/rod_scatter.entry.cpp"
#include "utility/face_scatter.entry.cpp"
#include "utility/stitch_scatter.entry.cpp"

// The two per-element strain readings, the largest shell singular value and the
// rod's length ratio, and beside them the two line-search times of impact. The
// readings are element gathers into one element scatter; each time of impact
// reads its element's positions through an index list, and gates on a
// non-positive limit inside its body.
#include "strainlimiting/strain_toi.entry.cpp"

// The Dirichlet elimination's second pass: make every removed row the identity
// carrying its prescribed increment. Three element gathers read, one non-const
// element gather written, the force as a base pointer with the thread index
// forwarded beside it, and no scatter. The gate on the DOF mask is inside the
// body, where a branch on what the scene contains belongs.
#include "main/dirichlet.entry.cpp"

// The rod strain limiter's force, Hessian and strain, at the authored limit.
// Two positions through the edge's own index pair, two element gathers read, the
// barrier kind as a scalar every thread shares, three non-const gathers written,
// and the verdict as the scatter. Both gates, on a non-positive limit and on a
// non-positive rest length, are inside the body rather than in the caller.
#include "strainlimiting/rod_strain.entry.cpp"

// The strain limiter's barrier derivative pair, at the AUTHORED limit. Four
// element gathers, two read and two written, and the gate a face with no
// `strainlimit` takes is inside the body rather than in this backend.
#include "strainlimiting/shell_strain.entry.cpp"

// The detect-once pair cache's recorder. Its counter is a CLAIM atomic, so no
// range partition makes it safe and the caller runs it as one ascending pass.
#include "contact/pair_cache.entry.cpp"

// The DYNAMIC matrix's two whole-matrix passes: the prefix sum that turns
// per-row counts into row offsets, and the transpose index the symmetric matvec
// reads to reach the lower triangle. Each is dispatched over ONE element and
// walks the whole row range itself, because a prefix sum and a transpose
// scatter are both sequential in the order the rows are walked. The thread
// index is forwarded and guards the pass, so a wider extent runs it once rather
// than N racing times.
#include "csrmat/dynamic_csr.entry.cpp"

// The fixed-pattern CSR block push. Two coordinate gathers and a 3x3 block
// gather per element, the pattern and the value array as base pointers because
// the body walks a row whose width is data, and the per-block verdict as the
// scatter that carries the body's return value.
//
// A SCATTER, NOT A LAUNCHER'S BOOKKEEPING. A block whose (row, column) is
// outside the build-time sparsity is refused by the body and the false reaches
// the driver here by construction, which is what a hand-written range shim had
// to remember to do. The fold underneath is `compute::atomic_add`, so this row
// carries the same serial-or-disjoint rule as the scatters above.
#include "csrmat/fixed_csr.entry.cpp"

// Rayleigh stiffness damping, one entry per element arity. Each reads two
// position buffers THROUGH the element's own slots, the per-element damping
// coefficient as a gather, and the gradient and Hessian as NON-CONST gathers,
// which is the write. The two bending members take a third gather, the LAGGED
// start-of-step Hessian, which is what makes their dissipation unconditional.
//
// READ-MODIFY-WRITE, NOT A FILL. This pass adds its force into the gradient the
// elastic assembly left and scales that Hessian in place, so the two gathered
// elements carry a value in as well as out. It is `Scatter::Disjoint` all the
// same: an element entry covers one ELEMENT per thread and each element owns
// its own slot in both destinations, so no two threads name the same one.
#include "utility/tet_damping.entry.cpp"
#include "utility/face_damping.entry.cpp"
#include "utility/rod_damping.entry.cpp"
#include "utility/hinge_damping.entry.cpp"
#include "utility/rod_bend_damping.entry.cpp"
