// File: build.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Compiles the host-callable C ABI shim (cpp/intersect_ffi.cpp) over the
// shared edge-triangle pierce predicate that lives in the CUDA solver tree
// (../ppf-cts-solver/src/cpp/contact/intersect_core.hpp). This makes the
// device intersection predicate the single source of truth: the Rust
// build-time self-intersection check links this object and calls the same
// geometry the GPU kernels run, instead of a parallel Rust port.
//
// This is a plain host C++ compile (no nvcc, no CUDA): the predicate
// header is dependency-free and STL-free, so it also builds on a no-CUDA
// host (macOS emulated path) with the system C++ compiler.

use std::path::Path;

fn main() {
    // The shared predicate header is authored in the solver's C++ tree
    // (sibling crate, source-file dependency only, not a Cargo dependency,
    // so there is no dependency cycle).
    let solver_cpp = "../ppf-cts-solver/src/cpp";
    let header = Path::new(solver_cpp).join("contact/intersect_core.hpp");

    println!("cargo:rerun-if-changed=cpp/intersect_ffi.cpp");
    println!("cargo:rerun-if-changed={}", header.display());

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/intersect_ffi.cpp")
        .include(solver_cpp)
        .compile("ppf_isect_ffi");
}
