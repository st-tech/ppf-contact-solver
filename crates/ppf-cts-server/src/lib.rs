// File: crates/ppf-cts-server/src/lib.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// ppf-cts-server: tokio engine host for the contact solver.
//
// Provides the engine framework: state holding, transition dispatch,
// effect execution, and a file-watching monitor task. The
// frontend-dependent effects (DoSpawnBuild, DoLaunchSolver,
// DoLoadApp, DoRequestSaveAndQuit, DoKillSolver) are implemented in
// `executor.rs`.

//! Tokio engine host for the ppf-contact-solver. Owns the tokio
//! runtime, the wire protocol (TCMD, JSON, BDAT framing at
//! [`PROTOCOL_VERSION`]), the file-watching monitor task, and the
//! effects executor. The pure state machine is delegated to
//! [`ppf_cts_core`]; shared payload types come from
//! [`ppf_cts_formats`].
//!
//! # Public surface
//!
//! - [`engine::ServerEngine`]: holds state and dispatches transitions.
//! - [`executor::EffectExecutor`] / [`executor::DefaultExecutor`]:
//!   side-effect surface (build, solver launch, GPU checks).
//! - [`config::EngineConfig`], [`config::HardwareInfo`]: runtime
//!   configuration and detected hardware metadata.
//! - [`monitor`], [`serve`], [`upload`], [`protocol`]: network and
//!   filesystem-facing pieces.
//!
//! Binary entry at `main.rs`; library surface mostly internal.

pub mod config;
pub(crate) mod easy_parse;
pub mod engine;
pub(crate) mod error;
pub mod executor;
pub mod hardware;
pub mod monitor;
pub(crate) mod protocol;
pub(crate) mod response;
pub mod serve;
pub(crate) mod upload;
pub(crate) mod wire;

pub use config::{EngineConfig, HardwareInfo};
pub use engine::ServerEngine;
pub use executor::{DefaultExecutor, EffectExecutor};

/// Wire-format protocol version. The addon refuses to connect to a server
/// with a different value (strict-equality handshake), so a stale or
/// mismatched server is rejected and forced to restart.
///
/// Single-sourced in `blender_addon/protocol_version.toml`: baked in here by
/// `build.rs` (`env!`) and read by the addon's `core/protocol.py` at run time,
/// so the two separately shipped halves cannot drift. See that file for the
/// bump policy and full changelog.
pub const PROTOCOL_VERSION: &str = env!("PPF_PROTOCOL_VERSION");
