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
pub mod status;

pub use envelope::{Envelope, FormatError, SCHEMA_VERSION};
pub use kinds::{ParamPayload, ScenePayload};
pub use status::{RunStatus, STATUS_VERSION};
