// File: generated_entries.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Every `rust` entry rendering, compiled.
//!
//! # Why this module exists
//!
//! One `[[seam::args]] [[seam::entry]]` declaration renders four ways, and a
//! rendering nothing compiles is a rendering nothing checks. That is not a
//! hypothetical: the MSL rendering of every gathered struct element did not
//! compile at all, across 71 sites in 26 declarations, and stayed that way
//! because no shader compiler had ever read one.
//!
//! The `rust` renderings were in the same position by halves.
//! [`super::kernels`] `include!`s the ones this driver DISPATCHES, which is
//! what puts them in front of rustc; the rest are rendered by the build script
//! and read by nothing. This module reads all of them, so a rendering that
//! stops compiling, or whose layout assertions stop holding, fails the build
//! rather than waiting for a driver to want it.
//!
//! # What it is not
//!
//! It is NOT a kernel table and nothing here is dispatchable. The kernel ids
//! are local stand-ins, all zero, because a real id is a dense index over what
//! the driver dispatches and only the driver can assign one; a record here
//! carries the stand-in and is never handed to a backend. The types are the
//! REAL [`HostRef`], [`Handle`], [`KernelId`] and [`KernelArgs`], so what is
//! checked is what the driver would get: the field types, the layout
//! assertions, and the trait implementation.
//!
//! A record this module compiles and [`super::kernels`] also `include!`s is
//! compiled twice, as two distinct types in two modules. That is deliberate and
//! costs nothing at run time: neither copy is constructed here.
//!
//! # Keeping it complete
//!
//! The list is GENERATED, into `$OUT_DIR/kernelgen/entries_manifest.rs` by this
//! crate's `build.rs`, and included below. It holds one `include!` per neutral
//! source that declares an entry point and one stand-in id per entry point
//! those renderings name.
//!
//! It used to be written out here, for a reason that reads well and turns out
//! to hold for only half of it: a declaration that does not reach a list is a
//! silent gap, and a name in a list with nothing behind it is a compile error.
//! The second half is still true and the first is now unreachable. The
//! declarations ARE the list: `build.rs` gets them from the recipe that
//! rendered the artifacts, so nothing transcribes anything and there is no step
//! at which a declaration can fail to arrive. A new `[[seam::entry]]` is
//! compiled here with no edit to this file.
//!
//! What generation does NOT decide, and must not: which kernels this backend
//! compiles at all. That is `build.rs`'s `KERNELS`, a curated list rather than
//! a directory walk, because it is this backend's capability policy and
//! [`super::refusal`] refuses those capabilities by name at `initialize()`.
//!
//! The C++ half of the same job, `entrypoints/entries.cpp`, is still a list a
//! human maintains, and `build.rs`'s `check_entry_coverage` still compares it
//! against the declarations. It carries per-entry prose about what each
//! declaration gathers and scatters, which a generated file has no place for.
//!
//! One check survives generation and is worth more than the list was:
//! `check_recognizer_agreement` compares the LEXICAL test that decides whether
//! a kernel gets entry artifacts against the transcompiler's own span parser,
//! per declaration. A declaration only one of them accepts leaves a file that
//! compiles and defines nothing, which no include list has ever been able to
//! see.

#![allow(dead_code)]

use ppf_cts_compute::{Handle, HostRef, KernelArgs, KernelId};

/// The value every stand-in id in the manifest carries.
///
/// Zero on purpose. A rendering references `id::<STEM>` so that a missing or
/// renamed id is a compile error in the driver rather than a literal in two
/// places; here there is no driver, so the value is meaningless and the NAME is
/// the whole content. Nothing in this module is dispatched, so no two records
/// sharing an id can collide.
const STANDIN: KernelId = KernelId(0);

// The manifest: the `id` module and one `include!` per rendering. The artifacts
// live under `$OUT_DIR/kernelgen`, mirroring their path under `src/kernels`,
// and are written by `ppf-cts-compute`'s host recipe from the same declaration
// the C++ renderings come from.
include!(concat!(env!("OUT_DIR"), "/kernelgen/entries_manifest.rs"));
