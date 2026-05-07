// File: crates/ppf-cts-core/src/lib.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Pure data model, state machine, and numeric kernels for the
//! ppf-contact-solver Rust workspace. Free of PyO3, tokio, and disk
//! I/O, so the algorithmic core is testable on its own with `cargo
//! test -p ppf-cts-core`.
//!
//! # What's in this crate
//!
//! - [`datamodel`], [`state`], [`transitions`]: typed scene state and
//!   the engine state machine.
//! - [`kernels`]: rasterizer, SDF, marching cubes, self-intersection,
//!   frame mapping, shared by Python and server.
//! - [`effects`], [`events`]: side-effect descriptors and the events
//!   fed back in.
//! - [`parsers`], [`extra`], [`utils`], [`cancel`] (incl. the
//!   [`CancelHandle`] re-export): support helpers.
//!
//! Wire and persistence types live in [`ppf_cts_formats`]; PyO3
//! bindings in `ppf-cts-py`; the tokio engine host in `ppf-cts-server`.

pub mod cancel;
pub mod datamodel;
pub mod effects;
pub mod events;
pub mod extra;
pub mod kernels;
pub mod parsers;
pub mod state;
pub mod transitions;
pub mod utils;

pub use cancel::CancelHandle;
