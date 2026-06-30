// File: args.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use clap::Parser;

#[derive(Parser, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[clap(author, version, about, long_about = None)]
pub struct ProgramArgs {
    #[clap(long, default_value = "")]
    pub path: String,

    #[clap(long, default_value = "output")]
    pub output: String,

    #[clap(long, default_value_t = 0)]
    pub load: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimArgs {
    pub disable_contact: bool,
    pub keep_states: i32,
    pub keep_verts: i32,
    pub dt: f32,
    pub inactive_momentum: bool,
    pub playback: f32,
    pub min_newton_steps: u32,
    pub target_toi: f32,
    pub air_friction: f32,
    pub line_search_max_t: f32,
    pub constraint_ghat: f32,
    pub constraint_tol: f32,
    pub fps: f64,
    pub cg_max_iter: u32,
    pub cg_tol: f32,
    pub ccd_reduction: f32,
    pub ccd_max_iter: u32,
    pub max_dx: f32,
    pub eiganalysis_eps: f32,
    pub friction_eps: f32,
    pub csrmat_max_nnz: u32,
    pub frames: i32,
    pub auto_save: i32,
    #[serde(default)]
    pub save_state_on_finish: bool,
    /// Comma-separated solver frame indices at which to write a resumable
    /// state, independent of `auto_save`. Empty string disables explicit
    /// checkpoints. `#[serde(default)]` keeps older `param.toml` files
    /// (written before this field existed) loadable for resume.
    #[serde(default)]
    pub checkpoints: String,
    pub barrier: String,
    pub friction_mode: String,
    pub stitch_length_factor: f32,
    pub air_density: f32,
    pub isotropic_air_friction: f32,
    pub gravity: [f32; 3],
    pub wind: [f32; 3],
    pub include_face_mass: bool,
    pub fix_xz: f32,
    pub fake_crash_frame: i32,
    /// Linear-solve preconditioner: "block-jacobi" (default) or "schwarz".
    /// Block-Jacobi is the default since the PCG inner loop became
    /// device-resident: its many cheap iterations beat Schwarz's few expensive
    /// ones, and it does not OOM on heavy-contact scenes. Parsed in `make_param`
    /// (mirrors the `barrier` / `friction_mode` idiom). `#[serde(default)]` keeps
    /// older `param.toml` files (written before this field existed) loadable;
    /// they default to the block-Jacobi preconditioner.
    #[serde(default = "default_precond")]
    pub precond: String,
    /// Number of additive levels for the Schwarz preconditioner (1 =
    /// single-level smoother, 2 = two-level coarse correction). Only used
    /// when `precond` is "schwarz". `#[serde(default)]` keeps older
    /// `param.toml` files (written before this field existed) loadable;
    /// they default to the two-level coarse correction.
    #[serde(default = "default_schwarz_levels")]
    pub schwarz_levels: u32,
    /// Uniform spatial scale applied to all input geometry on ingest; output
    /// positions are divided back by it on write. 1.0 is a no-op. Lets a mesh
    /// authored at the wrong scale (e.g. a 15 m human) simulate at the right
    /// physical size (1.5 m) and still be written back at the authored scale.
    /// `#[serde(default = "default_world_scaling")]` keeps older `param.toml`
    /// files (written before this field existed) loadable for resume; they
    /// default to 1.0 (no scaling).
    #[serde(default = "default_world_scaling")]
    pub world_scaling: f32,
}

fn default_precond() -> String {
    "block-jacobi".to_string()
}

fn default_schwarz_levels() -> u32 {
    2
}

fn default_world_scaling() -> f32 {
    1.0
}
