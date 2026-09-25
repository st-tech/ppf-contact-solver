// File: crates/ppf-cts-compute/src/lib.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! A general compute API: allocate, free, transfer and launch, on one of
//! several targets, plus the platform machinery those four need.
//!
//! # The one test this crate is held to
//!
//! *Could this crate be published on its own and used by a program that is not
//! a physics solver?* Everything here answers yes. Nothing that answers no may
//! be added, and the list of what answers no is specific rather than a mood: a
//! convergence test, a phase ordering, a fallback, a parameter interpretation, a
//! retry, a capacity decision, a branch on a physical quantity, and any data
//! type that crosses the seam and names a solver concept.
//!
//! That is the rule, and this crate is where it is enforced by construction
//! rather than by review.
//!
//! # What that costs, and how the cost is paid
//!
//! A kernel's argument record names solver concepts and is generated from a
//! declaration that lives in the solver crate, so this crate cannot hold one.
//! A dispatch therefore carries a KERNEL IDENTIFIER, an EXTENT and an OPAQUE
//! ARGUMENT BLOB: bytes are bound and launched, never read. The records, their
//! layout assertions and the kernel table stay above the seam, and the
//! dependency edge points one way.

// The seam's own vocabulary and the two traits that carry it. Re-exported flat
// so a caller writes `ppf_cts_compute::Device` rather than naming an internal
// module: the crate IS the seam, and a module path between the two would
// suggest there is more than one.
mod device;
pub use device::*;

// The C ABI target: one `Device` over any library exporting `be_*`. It
// names no target, so there is one of it and not three, and which library was
// loaded is what `be_backend_name` reports.
pub mod abi;

// THE CPU TARGET'S SOURCES ARE IN `cpu/`, beside `cuda/`, `rocm/` and
// `metal/`, so every target is one top-level directory of this crate and
// `src/` holds only what no single target owns: the seam, the C ABI client and
// the build API's module root. The `#[path]` attributes below are what that
// layout costs, since cargo looks for a module under `src/` by default.
//
// The modules are named `host`, not `cpu`, because that is the target's name at
// the seam, and the seam is wider than the CPU backend: the solver's own unit
// tests run on `HostDevice` in every build, whichever GPU backend the binary
// links, and the solver crate names no backend in any symbol it spells,
// which includes this type's.

// The scratch pool. Allocation computes no value, so it is mechanism by
// definition: it decides where bytes live and never what is in them.
#[path = "../cpu/mem.rs"]
pub mod mem;
// How a range is cut across threads. It is on this side of the seam because a
// backend's only contribution to a dispatch is how the range is cut, and it is
// a MODULE rather than a flat re-export because a caller outside a dispatch
// (a driver walking a tree with its own parallel loop) reads the same two
// functions and should be seen to be reading the backend's rule.
#[path = "../cpu/sched.rs"]
pub mod sched;

// The host target. Compiled unconditionally: it is pure Rust over the caller's
// own entry points, so it builds on every platform, and a target that only
// compiles where its vendor toolchain is installed would make the seam itself
// unbuildable on the machines that review it.
#[path = "../cpu/host.rs"]
pub mod host;
pub use host::{DiagRecord, HostDevice, Launch};

// The build-script API: how each target's sources are rendered and compiled.
//
// Behind a feature so a RUNTIME dependent never pulls `cc`. With resolver 2 a
// build-dependency's features are not unified with a normal dependency's, so
// one crate can take this module in its build script and not in its binary.
#[cfg(feature = "build")]
pub mod build;
