// File: intersect_ffi.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Host-callable C ABI over the shared edge-triangle pierce predicate that
// lives in the CUDA solver tree (ppf-cts-solver/src/cpp/contact/
// intersect_core.hpp). This is the single source of truth: the same
// templated routine is instantiated for float by the device contact
// kernels and for double here, so the Rust build-time self-intersection
// check (crates/ppf-cts-core/src/kernels/intersection.rs) runs the exact
// same geometry instead of a parallel Rust port.
//
// Compiled by ppf-cts-core/build.rs via the `cc` crate (plain host C++, no
// nvcc, no CUDA runtime) and linked into every consumer of ppf-cts-core
// (the _ppf_cts_py cdylib, the solver, the server, and the crate's own
// unit tests). The predicate header is dependency-free and STL-free, so
// this translation unit pulls in no C++ standard library symbols.

#include "contact/intersect_core.hpp"

extern "C" bool ppf_isect_edge_triangle_intersect(const double *e0,
                                                  const double *e1,
                                                  const double *v0,
                                                  const double *v1,
                                                  const double *v2) {
    return ppf_isect::edge_triangle_intersect<double>(e0, e1, v0, v1, v2);
}
