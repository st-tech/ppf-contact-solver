// File: args.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use clap::Parser;

#[derive(Parser, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    // Do not list
    #[clap(long, default_value = "")]
    pub path: String,

    // Do not list
    #[clap(long, default_value = "output")]
    pub output: String,

    // Name: Step Size
    // Recommended Range: 1e-3 to 1e-2
    // Description:
    // Step size for the simulation.
    // Small step size will make the simulation more accurate.
    // Large step size may make the simulation faster but can introduce noticeable damping.
    // We recommend to use 1e-3 for most cases. Setting a larger value can results in
    // PCG solver divergence.
    #[clap(long, default_value_t = 1e-3)]
    pub dt: f32,

    // Name: Fitting Mode
    // Description:
    // Enable fitting mode for the simulation.
    // This mode adjusts the simulation step size and disable inertia.
    #[clap(long)]
    pub fitting: bool,

    // Name: Playback Speed
    // Description:
    // The speed at which the simulation is played back.
    // 1.0 means unaltered playback, values greater than 1.0 speed up the playback,
    // and values less than 1.0 slow down the playback.
    #[clap(long, default_value_t = 1.0)]
    pub playback: f32,

    // Name: Lower Bound of Newton Steps
    // Recommended Range: 0 to 32
    // Description:
    // Minimal Newton's steps that must be consumed to advance a step.
    // The solver tries to determine the actual Newton's step count, but this number
    // assumes that static friction is not present.
    // If static friction is present, we recommend setting this value to 32.
    // Also, the step size should be somewhat large (e.g., 1e-2) to more accurately account for
    // static friction.
    #[clap(long, default_value_t = 0)]
    pub min_newton_steps: u32,

    // Name: Target Accumulated Time of Impact (TOI)
    // Recommended Range: 0.1 to 0.5
    // Description:
    // At each inner Newton's step, the minimal TOI for continuous collision detection is computed.
    // This TOI is used to rewind back the time to the TOI and a new Newton's step is computed.
    // We accumulate this TOI, and if the accumulated TOI exceeds this value, we terminate the
    // Newton's loop and compute the actual step size advanced.
    #[clap(long, default_value_t = 0.25)]
    pub target_toi: f32,

    // Name: Air Tangental Friction
    // Recommended Range: 0 to 1
    // Description:
    // When an object is moving in the air, both drag and lift forces are computed.
    // This value controls the ratio of the tangential friction to the normal friction.
    #[clap(long, default_value_t = 0.2)]
    pub air_friction: f32,

    // Name: Extended Line Search Maximum Time
    // Recommended Range: 1.25 to 1.75
    // Description:
    // Continuous Collision Detection (CCD) is used to determine the time of impact (TOI),
    // but if we advance the time to the actual TOI, contact gaps can be nearly zero.
    // Such a small gap can cause the solver to diverge, so we extend the time to the TOI by this
    // factor, and the actual TOI is recaled by dividing by this factor.
    // For example, if the actual TOI is 1.0 and this value is 1.25, the actual TOI is 1.0/1.25.
    #[clap(long, default_value_t = 1.25)]
    pub line_search_max_t: f32,

    // Name: Contact Gap
    // Recommended Range: 0.25e-3 to 1e-2
    // Description:
    // This value dictates the maximal gap distance when nearly touching objects are considered
    // in contact. A contact barrier is then activated to prevent the objects from penetrating.
    // Since we employ single precision floating point, we recommend setting this value to be
    // larger than 1e-4.
    #[clap(long, default_value_t = 1e-3)]
    pub contact_ghat: f32,

    // Name: Shell Contact Offset
    // Recommended Range: 0 to 1e-2
    // Description:
    // This value is used to offset the contact to give a visible thickness to shell contact surfaces.
    #[clap(long, default_value_t = 0.0)]
    pub contact_offset: f32,

    // Name: Offset for Static Mesh
    // Recommended Range: 0 to 1e-2
    // Description:
    // This value is used to offset the contact to give a visible thickness to surfaces on static
    // mesh surfaces.
    #[clap(long, default_value_t = 0.0)]
    pub static_mesh_offset: f32,

    // Name: Rod Contact Offset
    // Recommended Range: 0 to 1e-2
    // Description:
    // This value is used to offset the contact to give a visible thickness to rod contact surfaces.
    #[clap(long, default_value_t = 5e-3)]
    pub rod_offset: f32,

    // Name: Gap Distance for Boundary Conditions
    // Reommended Range: 1e-3 to 1e-2
    // Description:
    // For boundary conditions such as pinned vertices and wall conditions, this value is used
    // to determine the gap distance to activate the barrier to enforce the boundary conditions.
    #[clap(long, default_value_t = 1e-3)]
    pub constraint_ghat: f32,

    // Name: Moving Constraint Minimum Gap Tolerance
    // Recommended Range: 1e-3 to 0.1
    // Description:
    // For moving constraints, the gap distance can be negative at Newton's steps.
    // This value multiplies the constraint gap is used to cap such small gaps.
    #[clap(long, default_value_t = 0.01)]
    pub constraint_tol: f32,

    // Name: Strain Limit Epsilon
    // Recommended Range: 0 to 0.05
    // Description:
    // After the strain limit is activated, this value is used to control the maximal stretch.
    // For example, if this value is 0.05, the maximal stretch ratio is 5%.
    #[clap(long, default_value_t = 0.05)]
    pub strain_limit_eps: f32,

    // Name: Flag to Disable Strain Limit
    // Description:
    // When this flag is enabled, the strain limit is disabled regardless of `strain_limit_eps`.
    #[clap(long)]
    pub disable_strain_limit: bool,

    // Name: Frame Per Second for Video Frames
    // Description:
    // The time interval for writing meshes are determined by this value so that
    // when a video is generated, the frame rate corresponds to this value.
    #[clap(long, default_value_t = 60.0)]
    pub fps: f64,

    // Name: Maximum Number of PCG Iterations
    // Description:
    // When PCG iterations exceed this value, the solver is regarded as diverged.
    #[clap(long, default_value_t = 10000)]
    pub cg_max_iter: u32,

    // Name: Relative Tolerance for PCG
    // Recommended Range: 1e-4 to 1e-3
    // Description:
    // The relative tolerance for the PCG solver. The solver is terminated when the relative
    // residual reaches below this value.
    #[clap(long, default_value_t = 1e-3)]
    pub cg_tol: f32,

    // Name: ACCD Reduction Factor
    // Recommended Range: 1e-2 to 0.5
    // Description:
    // ACCD needs some small number to determine that the gap distance is close enough to the surface.
    // This factor is multiplied to the initial gap to set this threshold.
    #[clap(long, default_value_t = 0.01)]
    pub ccd_reduction: f32,

    // Name: Maximum CCD Iterations
    // Description:
    // The maximum number of iterations for ACCD.
    #[clap(long, default_value_t = 4096)]
    pub ccd_max_iter: u32,

    // Name: Maximum Search Direction
    // Recommended Range: 0.01m to 1m
    // Description:
    // This parameter defines the maximum allowable search direction magnitude
    // during optimization processes. It helps in controlling the step size
    // and ensuring stability.
    #[clap(long, default_value_t = 1.0)]
    pub max_dx: f32,

    // Name: Epsilon for Eigenvalue Analysis
    // Recommended Range: 1e-3 to 1e-2
    // Description:
    // When two singular values are close to each other, our analysis can be unstable
    // as it involves division by the difference of the two singular values.
    // In such cases, we switch to a stable approximation.
    #[clap(long, default_value_t = 1e-2)]
    pub eiganalysis_eps: f32,

    // Name: Friction Coefficient
    // Allowed Range: 0 to 1
    // Description:
    // Friction coefficient for the contact.
    // 0.0 means no friction and 1.0 means full friction.
    #[clap(long, default_value_t = 0.5)]
    pub friction: f32,

    // Name: Epsilon for Friction
    // Recommended Range: 1e-5 to 1e-4
    // Description:
    // We employ a quadratic friction model to approximate Coulomb friction.
    // In doing so, we need to set a small value to avoid division by zero.
    // This occurs when the motion undergoes static friction.
    #[clap(long, default_value_t = 1e-5)]
    pub friction_eps: f32,

    // Name: Maximal Matrix Entries for Contact Matrix Entries on the GPU
    // Recommended Range: 1e6 to 1e9
    // Description:
    // We pre-alocate the contact matrix entries to avoid cost arising from
    // dynamic memory allocation. This value is manually chosen by the user.
    // If this value is too large, the GPU runs out of memory.
    // On the other hand, if this value is too small, the simulation may fail
    // due to the lack of memory for contact matrix assembly.
    #[clap(long, default_value_t = 10000000)]
    pub csrmat_max_nnz: u32,

    // Name: Extra Memory Allocation Factor for BVH on the GPU
    // Recommended Range: 2 to 3
    // Description:
    // We pre-allocate the memory for the BVH on the GPU to avoid dynamic
    // memory allocation. BVH size can dynamically grow during the simulation.
    // This number is multiplied to the initial BVH size to allocate the memory.
    #[clap(long, default_value_t = 2)]
    pub bvh_alloc_factor: u32,

    // Name: Maximal Frame Count to Simulate
    // Description:
    // This number dictates the maximal number of frames to simulate.
    #[clap(long, default_value_t = 300)]
    pub frames: i32,

    // Name: Constitutive Model for Shells
    // Choices: baraffwitkin, shhk (Stable Neo-Hookean), arap (As-Rigid-As-Possible), stvk (St. Venant-Kirchhoff)
    // Description:
    // Physical model for the Finite Element shell.
    #[clap(long, default_value = "baraffwitkin")]
    pub model_shell: String,

    // Name: Constitutive Model for Volumetric Solids
    // Choices: snhk (Stable Neo-Hookean), arap (As-Rigid-As-Possible), stvk (St. Venant-Kirchhoff)
    // Description:
    // Physical model for the Finite Element volumetric solid.
    #[clap(long, default_value = "snhk")]
    pub model_tet: String,

    // Name: Barrier Model for Contact
    // Choices: cubic, quad, log
    // Description:
    // Contact barrier potential model.
    #[clap(long, default_value = "cubic")]
    pub barrier: String,

    // Name: Young's Modulus for Shells
    // Recommended Range: 100 to 1e4
    // Description:
    // This value is not the actual Young's modulus but the one divided by
    // the density of the material.
    #[clap(long, default_value_t = 100.0)]
    pub area_young_mod: f32,

    // Name: Poisson's Ratio for Shells
    // Allowed Range: 0.0 to 0.4999
    // Description:
    // Poisson's ratio for the shell material, encoding how much the material
    // responds to compression.
    // 0.0 means the material does not respond to any compression, while 0.5
    // means completely incompressible. 0.5 leads to numerical instability, so it
    // must be less than 0.5.
    #[clap(long, default_value_t = 0.25)]
    pub area_poiss_rat: f32,

    // Name: Young's Modulus for Volumetric Solids
    // Recommended Range: 100 to 1e4
    // Description:
    // This value is not the actual Young's modulus but the one divided by
    // the density of the material.
    #[clap(long, default_value_t = 500.0)]
    pub volume_young_mod: f32,

    // Name: Poisson's Ratio for Volume Solids
    // Allowed Range: 0.0 to 0.4999
    // Description:
    // Poisson's ratio for the volumetric solid material, encoding how much the material
    // responds to compression.
    // 0.0 means the material does not respond to any compression, while 0.5
    // means completely incompressible. 0.5 leads to numerical instability, so it
    // must be less than 0.5.
    #[clap(long, default_value_t = 0.35)]
    pub volume_poiss_rat: f32,

    // Name: Young's Modulus for Rods
    // Recommended Range: 1e3 to 1e6
    // Description:
    // Unlike shells or volumetric solids, this value is the actual Young's modulus.
    #[clap(long, default_value_t = 1e4)]
    pub rod_young_mod: f32,

    // Name: Stiffness Factor for Stitches
    // Recommended Range: 0.5 to 2.0
    // Description:
    // Stiffness factor for the stitches.
    #[clap(long, default_value_t = 1.0)]
    pub stitch_stiffness: f32,

    // Name: Shell Density
    // Description:
    // Material density for the shell.
    #[clap(long, default_value_t = 1e3)]
    pub area_density: f32,

    // Name: Volume Density
    // Description:
    // Material density for the volumetric solid.
    #[clap(long, default_value_t = 1e3)]
    pub volume_density: f32,

    // Name: Rod Density
    // Description:
    // Material density per unit length for the rod.
    #[clap(long, default_value_t = 1.0)]
    pub rod_density: f32,

    // Name: Air Density
    // Description:
    // Air density for drag and lift force computation.
    #[clap(long, default_value_t = 1e-3)]
    pub air_density: f32,

    // Name: Air Dragging Coefficient
    // Description:
    // Per-vertex air dragging coefficient.
    #[clap(long, default_value_t = 0.0)]
    pub isotropic_air_friction: f32,

    // Name: Bend Stiffness for Shells
    // Recommended Range: 0.0 to 1e2
    // Description:
    // This bending stiffness not scaled by the density.
    // If you change the density, you need to adjust this value accordingly.
    // This behavior also change by the material thickness.
    #[clap(long, default_value_t = 1.0)]
    pub bend: f32,

    // Name: Bend Stiffness for Rods
    // Recommended Range: 0.0 to 10
    // Description:
    // Rod bending stiffness.
    // The actual force is amplified by the rod mass, which
    // includes rod density.
    #[clap(long, default_value_t = 2.0)]
    pub rod_bend: f32,

    // Name: Gravity Coefficient
    #[clap(long, default_value_t = -9.8)]
    pub gravity: f32,

    // Name: Wind Coefficient
    // Description:
    // The wind strength.
    #[clap(long, default_value_t = 0.0)]
    pub wind: f32,

    // Name: Wind Direction
    // Description:
    // The wind direction.
    #[clap(long, default_value_t = 0)]
    pub wind_dim: u8,

    // Name: Flag to Include Shell Mass for Volume Solids
    // Description:
    // The mass of volume solids is computed by their tetrahedral elements.
    // If this option is enabled, the mass of the shell is included for the
    // surface elements of the volume solids.
    #[clap(long)]
    pub include_face_mass: bool,

    // Name: Whether to fix xz positions
    // Description:
    // For falling objects if their y position is higher than this value,
    // their xz positions are fixed.
    #[clap(long, default_value_t = 0.0)]
    pub fix_xz: f32,

    // Name: Fake Crash Frame
    // Description:
    // Frame number at which to intentionally crash the simulation for testing purposes.
    // Set to -1 to disable fake crashing.
    #[clap(long, default_value_t = -1)]
    pub fake_crash_frame: i32,
}
