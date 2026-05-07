// File: crates/ppf-cts-formats/src/kinds/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Concrete payload schemas. Two cross-language wire formats land here:
//
//   * `ScenePayload`: replaces data.pickle (mesh + transforms + pins +
//                     static animation). Mirrors the producer at
//                     blender_addon/core/encoder/mesh.py:280-462 and
//                     the consumer at frontend/_decoder_.py:851-1430.
//   * `ParamPayload`: replaces param.pickle (sim params + materials +
//                     pin keyframes + colliders + cross-stitch).
//                     Mirrors blender_addon/core/encoder/params.py and
//                     frontend/_decoder_.py:449-849.
//
// Two persistence-only formats (app_state.pickle, app.pickle) are NOT
// mirrored here because they currently snapshot full Python object
// graphs (BlenderApp, AssetManager, SceneManager). Those graphs become
// Rust structs in later phases and persist via bincode through whichever
// types own them. Replicating today's pickle layout would create dead
// schema we'd then throw away.

pub mod param;
pub mod scene;

pub use param::ParamPayload;
pub use scene::ScenePayload;

pub const KIND_SCENE: &str = "Scene";
pub const KIND_PARAM: &str = "Param";
