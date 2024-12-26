// File: args.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use clap::Parser;

#[derive(Parser, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    #[clap(long, default_value = "")]
    pub path: String,

    #[clap(long, default_value = "output")]
    pub output: String,

    #[clap(long, default_value_t = 1e-3)]
    pub dt: f32,

    #[clap(long, default_value_t = 0)]
    pub min_newton_steps: u32,

    #[clap(long, default_value_t = 0.25)]
    pub target_toi: f32,

    #[clap(long, default_value_t = 1e-3)]
    pub fitting_dt: f32,

    #[clap(long, default_value_t = 0.2)]
    pub aerial_friction: f32,

    #[clap(long, default_value_t = 1.25)]
    pub line_search_max_t: f32,

    #[clap(long, default_value_t = 1e-3)]
    pub contact_ghat: f32,

    #[clap(long, default_value_t = 0.0)]
    pub contact_offset: f32,

    #[clap(long, default_value_t = 5e-3)]
    pub rod_offset: f32,

    #[clap(long, default_value_t = 1e-3)]
    pub constraint_ghat: f32,

    #[clap(long, default_value_t = 0.025)]
    pub strain_limit_tau: f32,

    #[clap(long, default_value_t = 0.025)]
    pub strain_limit_eps: f32,

    #[clap(long, default_value_t = 1024)]
    pub binary_search_max_iter: u32,

    #[clap(long)]
    pub disable_strain_limit: bool,

    #[clap(long, default_value_t = 0.75)]
    pub dt_decrease_factor: f32,

    #[clap(long, default_value_t = 60.0)]
    pub fps: f64,

    #[clap(long, default_value_t = 10000)]
    pub cg_max_iter: u32,

    #[clap(long, default_value_t = 1e-3)]
    pub cg_tol: f32,

    #[clap(long)]
    pub enable_retry: bool,

    #[clap(long, default_value_t = 1024)]
    pub ccd_max_iters: u32,

    #[clap(long, default_value_t = 0.01)]
    pub ccd_reduction: f32,

    #[clap(long, default_value_t = 1e-2)]
    pub eiganalysis_eps: f32,

    #[clap(long, default_value_t = 0.5)]
    pub friction: f32,

    #[clap(long, default_value_t = 1e-5)]
    pub friction_eps: f32,

    #[clap(long, default_value_t = 70000000)]
    pub csrmat_max_nnz: u32,

    #[clap(long, default_value_t = 2)]
    pub bvh_alloc_factor: u32,

    #[clap(long, default_value_t = 300)]
    pub frames: i32,

    #[clap(long, default_value = "baraffwitkin")]
    pub model_shell: String,

    #[clap(long, default_value = "snhk")]
    pub model_tet: String,

    #[clap(long, default_value = "cubic")]
    pub barrier: String,

    #[clap(long, default_value_t = 100.0)]
    pub area_young_mod: f32,

    #[clap(long, default_value_t = 0.25)]
    pub area_poiss_rat: f32,

    #[clap(long, default_value_t = 500.0)]
    pub volume_young_mod: f32,

    #[clap(long, default_value_t = 0.35)]
    pub volume_poiss_rat: f32,

    #[clap(long, default_value_t = 1e4)]
    pub rod_young_mod: f32,

    #[clap(long, default_value_t = 1.0)]
    pub stitch_stiffness: f32,

    #[clap(long, default_value_t = 1e3)]
    pub area_density: f32,

    #[clap(long, default_value_t = 1e3)]
    pub volume_density: f32,

    #[clap(long, default_value_t = 1e3)]
    pub rod_density: f32,

    #[clap(long, default_value_t = 1e-3)]
    pub air_density: f32,

    #[clap(long, default_value_t = 0.0)]
    pub isotropic_aerial_friction: f32,

    #[clap(long, default_value_t = 1.0)]
    pub bend: f32,

    #[clap(long, default_value_t = 1e-3)]
    pub rod_bend: f32,

    #[clap(long, default_value_t = -9.8)]
    pub gravity: f32,

    #[clap(long, default_value_t = 0.0)]
    pub wind: f32,

    #[clap(long, default_value_t = 0)]
    pub wind_dim: u8,

    #[clap(long)]
    pub include_face_mass: bool,

    #[clap(long, default_value_t = 0.0)]
    pub fix_xz: f32,
}
