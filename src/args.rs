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
    pub fitting: bool,
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
    pub ccd_eps: f32,
    pub ccd_reduction: f32,
    pub ccd_max_iter: u32,
    pub max_dx: f32,
    pub eiganalysis_eps: f32,
    pub friction_eps: f32,
    pub csrmat_max_nnz: u32,
    pub bvh_alloc_factor: u32,
    pub frames: i32,
    pub auto_save: i32,
    pub barrier: String,
    pub stitch_stiffness: f32,
    pub air_density: f32,
    pub isotropic_air_friction: f32,
    pub gravity: f32,
    pub wind: f32,
    pub wind_dim: u8,
    pub include_face_mass: bool,
    pub fix_xz: f32,
    pub fake_crash_frame: i32,
}
