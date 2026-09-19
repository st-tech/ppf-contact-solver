// File: crates/ppf-cts-compute/src/build/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The build RECIPE for each target, as a library a build script calls.
//!
//! # Why a recipe here and an invocation there, rather than one build script
//!
//! Rendering a neutral kernel to a target and compiling the result is
//! backend-specific work, so it belongs to this crate:
//! the renderings, the compiler flags and the artifact guards are all statements
//! about how a device is made to compute, never about what is computed.
//!
//! It cannot be a build SCRIPT here, and the reason is cargo rather than taste.
//! A dependency's build script runs before the dependent's and is given no way
//! to see the dependent's files, so a script in this crate could reach the
//! kernel sources only by a relative path out of its own tree. That would work,
//! and it would fail the one test this crate is held to: it could no longer be
//! published on its own.
//!
//! So the split is: the caller names its own sources and its own output
//! directory, and this module decides everything else. The caller's build script
//! is the process, which is what puts the `cargo:` lines this module prints on
//! the right stdout.
//!
//! # What the caller keeps, and it is exactly one kind of thing
//!
//! Every input below is a NAME from the caller's own domain: which kernels,
//! which translation units, where they live. None of them is a decision about
//! how to compile, which is why none of them is made here.

pub mod host;
