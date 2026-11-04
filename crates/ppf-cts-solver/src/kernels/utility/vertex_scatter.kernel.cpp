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

