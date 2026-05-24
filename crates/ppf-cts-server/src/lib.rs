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

/// Wire-format protocol version. Locked in for client compatibility;
/// the addon refuses to connect to a server with a different value.
/// Must match `blender_addon/core/protocol.py:PROTOCOL_VERSION`.
///
/// BUMP THIS whenever the `data` / `param` payload schema OR the
/// addon-encoder / frontend-decoder contract changes. The handshake
/// is the only thing that catches an addon paired with a server whose
/// `frontend` decoder disagrees about payload shape; an un-bumped
/// version there silently mis-decodes instead of erroring.
///
/// 0.06: deforming STATIC mesh colliders. A STATIC object whose
/// modifier stack deforms vertices (Armature, MeshDeform, Lattice,
/// shape keys, etc.) can now carry a `static_deform_animation`
/// payload alongside `vert` / `transform`: a per-frame absolute
/// vertex buffer in solver world space, captured from Blender's
/// depsgraph. The decoder builds a zero-stiffness pin shell whose
/// every vertex is driven by `MoveByOperation` segments derived
/// from consecutive frames, just like the per-vertex pin animation
/// in 0.05 but spanning the whole mesh. Mutually exclusive with
/// `transform_animation` and `static_ops`; an old decoder would
/// ignore the new field and play the rest-pose mesh.
///
/// 0.05: keyframed pin animation. `param.pin_config[uuid]` stays
/// `{vertex_index: PinData}`, but each `PinData` now carries its own
/// single-entry `pin_anim` (`{that_vertex: PinAnim}`) and the
/// `frontend` decoder builds genuine per-vertex `MoveByOperation`
/// deltas from it instead of broadcasting one vertex's track. An old
/// decoder would treat the new payload as a rigid translation.
///
/// 0.04: TCMD requests are length-prefixed (`[4-byte u32 BE length]`
/// after the `b"TCMD"` header) instead of EOF-terminated. The
/// previous design relied on the client calling `shutdown(SHUT_WR)`
/// to signal end-of-input, but tokio on Windows did not surface that
/// half-close as `Ok(0)` to AsyncRead, so the server's read loop
/// hung forever and the connection sat in FIN_WAIT_2 indefinitely.
pub const PROTOCOL_VERSION: &str = "0.06";
