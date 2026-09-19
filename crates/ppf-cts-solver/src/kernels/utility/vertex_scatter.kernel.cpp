// File: vertex_scatter.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++, no preprocessor conditional and no macro
// of its own. ppf-cts-compute/seam/kernelgen.py renders it into the three forms the three
// compilers read. The two facts a compiler cannot infer are C++ attributes:
// `[[seam::device_fn]]` is the execution space, and `[[seam::thread]]` and
// `[[seam::device]]` are the address spaces MSL requires on every reference and
// pointer.

// The one-node scatter, the arity-1 member of the set the two-node rod scatter
// (rod_scatter.kernel.cpp), the three-node face scatter (face_scatter.kernel.cpp)
// and the four-node hinge scatter (hinge_scatter.kernel.cpp) already belong to.
// One body per node count, because MSL has no template over an address space
// the way a `__device__` template can serve every arity at once.
//
// Its contributors are the terms whose force lands on a single vertex: the
// analytic sphere and floor barriers, the barrier that holds a PDRD anchor, and
// a dynamic vertex against a static collision-mesh triangle.
//
// `vert` rather than `vertex`, which MSL reserves as a shader-stage qualifier
// alongside `kernel` and `fragment`; a parameter of that name parses there as a
// stage declaration and yields a cascade of errors pointing at lines that are
// fine.
//
// A zero component is added rather than skipped, matching the face, hinge and
// rod scatters. That is not a value change: the accumulator is cleared to +0.0
// before the pass and every contribution folds onto it, and IEEE
// round-to-nearest gives +0.0 for (+0.0) + (-0.0) and for x + (-x), so the
// buffer can never come to hold -0.0, which is the only float a +0.0 addition
// would alter.
[[seam::device_fn]] inline void
vertex_atomic_embed_force(unsigned vert,
                              const Vec3f &gradient,
                              compute::atomic_float_t *force) {
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        compute::atomic_add(force + 3 * vert + dimension, gradient[dimension]);
    }
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `vertex_atomic_embed_force_entry` shim a host C++
// compiler compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// THE THREAD INDEX REACHES THE BODY THROUGH THE GATHERS AND NOWHERE ELSE. This
// body takes no index: it takes the vertex the contribution lands on and the
// contribution itself, so the declaration carries no `[[seam::index]]` and the
// generator names the index it guards on. Both lists are compacted, one entry
// per contribution, so the two gathers read the same element of each.
//
// THE SCATTER IS THE BODY'S, AND IT STAYS SERIAL. `force` arrives as a base
// pointer because the destination is chosen by `vert`, not by the thread index,
// so two contributions can land on one vertex. `compute::atomic_add` is a plain
// read, add and write back on the host seam, which makes a parallel pass over
// these elements a data race rather than a different fold order. Nothing in
// this declaration says otherwise: the entry point covers whatever range it is
// handed, and the kernel table's `Scatter::Atomic` is what keeps that range one
// ascending pass.
//
// `compute::atomic_float_t` is the seam's own name for the destination's
// element type, `float` under nvcc and on the host and `atomic_float` under
// MSL. It is declared here rather than as `float` for exactly that reason: a
// `float *` would compile on two backends and fail on the third, and its 4
// bytes are asserted in every C++ rendering.
[[seam::entry(count)]] void vertex_atomic_embed_force(
    const unsigned *vert,
    const Vec3f *gradient,
    compute::atomic_float_t *force,
    unsigned count);

// THE PIN INDEX, COPIED OUT OF THE VERTEX RECORD IN ITS OWN THREAD.
//
// `main.cu` never gathers this: it reads `prop_vertex[i].fix_index` directly
// wherever it needs it, because its dataset is device-resident. This driver
// kept a flat `fix_index` array beside the record and refilled it from the
// HOST every advance, then uploaded it, which is an O(vertices) pass and one
// transfer per step for a field the device already holds.
//
// It stays a separate array rather than becoming a record read at each call
// site: three `compute_target` dispatches take it as a plain buffer, and
// widening all three to carry `VertexProp` would be a larger change than the
// one this removes.
[[seam::entry(element)]]
[[seam::device_fn]] inline void vertex_fix_index_from_records(
    const VertexProp &prop,
    unsigned *fix_index, unsigned element) {
    fix_index[element] = prop.fix_index;
}

// THE PIN DOF-REMOVAL MASK, built where `main.cu:653-662` builds it: a
// dispatch over the vertices reading the record, not a host pass and an
// upload.
//
// THE TWO TESTS ARE THE REFERENCE'S AND BOTH ARE LOAD-BEARING. A vertex with
// no fix pin is not eliminated, and a pinned vertex INSIDE a PDRD body is not
// either: it owns no per-vertex degree of freedom, the solve being reduced
// through the rigid Jacobian, so a Dirichlet row for it is not representable
// and its anchor keeps the barrier instead.
//
// `disable` IS THE A/B SWITCH, `PPF_DISABLE_PIN_DOF_REMOVAL`, and it arrives
// as a value rather than being read here because an environment variable is a
// property of the process and not of the element.
[[seam::entry(element)]]
[[seam::device_fn]] inline void vertex_dof_removal_mask(
    const VertexProp &prop, unsigned disable,
    unsigned *mask, unsigned element) {
    mask[element] = (disable == 0u && prop.fix_index > 0u &&
                     prop.pdrd_body_index == 0u)
                        ? 1u
                        : 0u;
}

