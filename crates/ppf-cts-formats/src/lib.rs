// File: crates/ppf-cts-formats/src/lib.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Wire and persistence formats shared across the ppf-contact-solver
//! ecosystem.
//!
//! These types are exchanged between the Blender addon (CBOR
//! producer), the Rust server (CBOR consumer), and the PyO3 frontend
//! bindings. Every cross-language payload is wrapped in an
//! [`Envelope`] so a peer with a mismatched [`SCHEMA_VERSION`] can be
//! refused explicitly rather than misinterpreted.
//!
//! # Public surface
//!
//! - [`envelope`]: the [`Envelope`] container, [`FormatError`], and
//!   [`SCHEMA_VERSION`].
//! - [`kinds`]: concrete payload bodies ([`ParamPayload`],
//!   [`ScenePayload`]).
//! - [`status`]: the structured solver run-status record ([`RunStatus`])
//!   and its liveness [`status::lock`], the single source of truth for
//!   solver lifecycle and outcome.
//!
//! Consumers: [`ppf_cts_core`] (kernels and state), `ppf-cts-py`
//! (PyO3 bindings), and `ppf-cts-server` (engine host).

pub mod envelope;
pub mod files;
pub mod kinds;
pub mod statistics;
pub mod status;

pub use envelope::{Envelope, FormatError, SCHEMA_VERSION};
pub use kinds::{ParamPayload, ScenePayload};
pub use statistics::{
    ObjectStatistics, StatisticChannel, StatisticsFrame, StatisticsInput, StatisticsInputObject,
    StatisticsManifest, StatisticsObject, StatisticsValidationError, STATISTICS_VERSION,
};
pub use status::{RunStatus, STATUS_VERSION};

/// A digest of the sources this crate's formats and the parameter registry are
/// built from, computed by `build.rs`.
///
/// Two binaries reporting the same stamp were built from the same
/// `ppf-cts-formats` and `ppf-cts-core` sources, so the session one writes is
/// the session the other reads. The frontend compares the loaded Python
/// extension's stamp with a solver's `--probe` answer before a run takes its
/// solver from another backend's directory. `build.rs` says what is digested
/// and why a version number cannot stand in for it.
pub const SOURCE_STAMP: &str = env!("PPF_SOURCE_STAMP");
