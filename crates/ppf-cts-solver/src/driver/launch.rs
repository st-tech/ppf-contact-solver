// File: crates/ppf-cts-solver/src/driver/launch.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The LAUNCH TABLE: one entry point per kernel id, and nothing else.
//!
//! The device that consumes this table is `ppf_cts_compute::HostDevice`, which
//! allocates, cuts a range across threads, calls and collects the diagnostic
//! channel without knowing what a kernel is. What it cannot supply is the part
//! that names the workload, and this file is that part: the compiled entry
//! points, the argument shapes the hand-written ones take, and the array pairing
//! each with a [`ppf_cts_compute::KernelId`].
//!
//! # Why the pairing lives here and not in the compute crate
//!
//! A row of [`LAUNCH`] names a symbol like `face_baraffwitkin_entry`, and
//! an argument shape like `LbvhNodesArgs` names a bounding-volume tree. Both are
//! statements about the simulation, so a crate holding them could not be
//! published on its own and used by a program that is not a physics solver,
//! which is the one test `ppf-cts-compute` is held to.
//! The table is therefore handed DOWN at `ppf_cts_compute::HostDevice::new`,
//! and a dispatch reaches the device as a kernel identifier, an extent and an
//! opaque blob.
//!
//! # What a thunk is for, and why it is not just a function pointer
//!
//! Each row casts the argument bytes back to the record `super::kernels`
//! declares and calls the shim. That is the Rust stand-in for the generated
//! `ppf_<stem>_entry` entry point; once the entry emitter lands, the thunks are
//! rendered from the same declaration as the record and this table is generated
//! with them. Until then a thunk is the only thing that can
//! catch a record disagreeing with a shim's parameter list, and it catches it at
//! COMPILE time, because it forwards the parsed fields to a signature the same
//! compiler reads.
//!
//! # The two rules the table states rather than implements
//!
//! Both are fields of the declaration in `super::kernels`, read by the device
//! and obeyed there: a kernel whose scatter is not `Scatter::Disjoint` runs as
//! one serial ascending pass, because `compute::atomic_add` on the host seam is
//! a plain read, add and write back and a parallel pass over shared output
//! slots is a data race; and a cut is by a FIXED chunk width from the declared
//! per-item cost, never by the thread count, which is what makes the answer
//! independent of how many threads ran it.

use ppf_cts_compute::host::{BoundArgs, Launch, MAX_BOUND_ARGS};
use ppf_cts_compute::DiagRecord;
use super::kernels::{self, id};

// The device every driver module runs against is `ppf_cts_compute::HostDevice`,
// re-exported rather than wrapped: the compute crate's device is complete on
// its own, and a newtype here would be twenty methods of delegation whose only
// effect is to hide which crate the seam lives in.
pub use ppf_cts_compute::HostDevice;

/// The backend THIS BUILD drives, named once.
///
/// A TYPE ALIAS AND NOT A TRAIT OBJECT, because `Device` carries an associated
/// `Region` and a generic `launch<A: KernelArgs>`, so it is not object-safe and
/// `Box<dyn Device>` cannot exist. One build drives one backend, which the
/// build script already enforces, so the choice is a compile-time one.
///
/// WHAT THIS IS FOR: the driver names this rather than a concrete backend, so
/// pointing it at the C ABI target is a `cfg` here rather than an edit in
/// twenty modules. `ppf_cts_compute::AbiDevice` is the other arm, and it drives
/// any library exporting `be_*`, which is what the CUDA and Metal backends in
/// that crate already do.
#[cfg(not(abi_backend_linked))]
pub type Backend = HostDevice;

/// The C ABI arm. One build links one backend library, and this is the device
/// over whichever one it linked: nothing here names CUDA or Metal, because
/// `AbiDevice` does not.
#[cfg(abi_backend_linked)]
pub type Backend = ppf_cts_compute::abi::AbiDevice;

/// The backend this build drives, constructed.
///
/// Pairs with [`Backend`]: a module that needs one asks here rather than
/// naming a constructor a different arm would not have.
#[cfg(not(abi_backend_linked))]
pub fn backend() -> Backend {
    host_device()
}

/// The backend this build drives, constructed, or why it could not be.
///
/// `ppf-contact-solver --probe` asks this rather than [`backend`], whose
/// failure is a panic: reporting that failure as an answer is the probe's job.
/// The host device cannot fail to construct.
#[cfg(not(abi_backend_linked))]
pub fn try_backend() -> Result<Backend, String> {
    Ok(host_device())
}

/// The C ABI arm's constructor.
///
/// FOUR CROSS-CHECKS HAPPEN INSIDE `open`, and a failure there is fatal rather
/// than a fallback: a library that disagrees with this build about the ABI
/// version, the record layout or the kernel table would otherwise dispatch the
/// wrong bytes to the right kernel. There is no host device to fall back to,
/// and offering one would be the silent-wrong-answer path the seam exists to
/// close.
#[cfg(abi_backend_linked)]
pub fn backend() -> Backend {
    match try_backend() {
        Ok(device) => device,
        Err(fault) => panic!(
            "\n\n  the linked backend library could not be opened: {fault}\n\n"
        ),
    }
}

/// The C ABI arm's constructor, returning the fault rather than panicking on
/// it. [`backend`] is this plus the panic; `--probe` is this plus a report.
#[cfg(abi_backend_linked)]
pub fn try_backend() -> Result<Backend, String> {
    // THE LIBRARY'S MESSAGES REACH THIS PROCESS'S LOG, which is the whole
    // reason `OpenConfig` takes a sink rather than defaulting: `new`'s
    // production default DROPS them, and a backend that cannot say anything is
    // one whose first complaint is a wrong answer with no explanation.
    fn sink(level: i32, message: &str) {
        // The ABI's levels, narrowest first. An unrecognized level is reported
        // rather than dropped, because a library saying something in a dialect
        // this build does not know is still saying something.
        match level {
            0 => ::log::debug!("{message}"),
            1 => ::log::info!("{message}"),
            2 => ::log::warn!("{message}"),
            _ => ::log::error!("{message}"),
        }
    }
    let config = ppf_cts_compute::abi::OpenConfig::new(sink);
    // NO `library_dir` IS NAMED HERE, AND THAT IS WHAT MAKES A SHIPPED COPY
    // WORK. A library that loads a pre-built shader has to find one, and the
    // obvious way to tell it is an absolute path the build script recorded at
    // compile time. That path names the machine that built the binary. It is
    // correct there and resolves to nothing anywhere else, so a bundle carrying
    // it fails to open its backend on any other machine while every load command
    // looks right, which is the hardest shape of failure to read.
    //
    // Left empty, the library resolves the directory it was itself loaded from.
    // The generated entry library sits beside it in both layouts this project
    // ships, the build tree's library directory and the bundle's own, so the
    // question has the same answer in both and neither needs telling.
    // Safety: the linked library implements `backend_abi.h`, and the table is
    // this tree's own generated one, which `open` cross-checks against the
    // library's rather than assuming.
    unsafe { ppf_cts_compute::abi::AbiDevice::open(&kernels::TABLE, &config) }
        .map_err(|fault| fault.to_string())
}

/// A host device carrying this solver's kernels.
///
/// The one place the two tables meet. `HostDevice::new` refuses a pair of
/// different lengths, so a row added to `super::kernels::TABLE` and not to
/// [`LAUNCH`] fails here rather than dispatching the wrong kernel with the right
/// bytes.
#[cfg_attr(abi_backend_linked, allow(dead_code))]
pub fn host_device() -> HostDevice {
    HostDevice::new(&kernels::TABLE, &LAUNCH)
}

#[cfg(test)]
#[path = "scheduling_tests.rs"]
pub(super) mod scheduling_tests;

// ===========================================================================
// The shims. The ONLY place in the tree below the seam that names one.
// ===========================================================================

// THE GENERATED ENTRY POINTS' DECLARATIONS, written by
// `ppf-cts-compute/seam/kernelgen.py --emit externs` and included from
// `OUT_DIR`. There is one per `[[seam::args]] [[seam::entry]]` declaration in
// the neutral tree, in sorted source order.
//
// A DECLARATION'S ARITY IS A FUNCTION OF THE ENTRY AND NOT A CHOICE: a group
// symbol takes an extra `group_width` between the arena base and the range,
// and a `[[seam::diag]]` entry takes a channel after it. Rust believes the
// declaration, so a hand-written one that disagreed miscounted arguments at
// RUN TIME rather than failing to build. `check-launch-seam.py` was added to
// police that and found four live mismatches on its first run; generating
// them removes the class instead of checking it.
//
// WHAT THE DECLARATIONS CANNOT CARRY IS THE PROSE THAT WOULD SIT BETWEEN
// THEM, because it describes GROUPS of kernels and no single declaration
// knows its group. It is kept here:
//   GENERATED entry points, rendered from the declarations beside their
//   bodies and compiled by `entrypoints/entries.cpp`. They share one signature,
//   which is the whole point of generating them: the record, the arena base
//   table, and the half-open range. A hand-written shim below takes its
//   arguments flat, so each needs its own declaration and its own thunk.
//   The contact subsystem's per-primitive centroid passes. Their positions
//   cross as raw words, three per vertex: this seam never interprets one, so
//   nothing here can round or reinterpret a coordinate on the way through.
//   The broad phase's construction passes. `Aabb` is this driver's mirror of
//   the device-only C++ `AABB`, and `super::lbvh::the_aabb_mirror_matches_cpp`
//   is what compares the two layouts: nothing else does, so a drift here is a
//   wrong answer rather than a link error.
//   The dynamic matrix's serial exclusive scan takes no thread range: it walks
//   the whole row range itself and returns the total it laid out. The thunk
//   below discards that return, because a dispatch produces no value and the
//   total is the last entry of the offset array the body just wrote.
//   The transpose's two halves DO take a thread range: one row per thread,
//   with the shared slot each reaches claimed through an atomic rather than
//   decided by the thread index.
//   THE SIX PASSES THE DYNAMIC MATRIX IS BUILT BY, over flat device storage.
//   Four are one row per thread and two are one contribution per thread; the
//   compaction carries the disjointness check on a diagnostic lane, so its
//   entry takes the extra record every `[[seam::diag]]` entry does.
//   The four narrow-phase visitors, which take one record and a diagnostic
//   slot. Each is called once per chunk, so building the shim record below is
//   paid per chunk and not per pair.
//   The three collision-mesh visitors.

include!(concat!(env!("OUT_DIR"), "/launch_externs.rs"));



// ===========================================================================
// The launch table.
// ===========================================================================

// ===========================================================================
// Binding a generated entry point's arguments.
// ===========================================================================

/// One thunk per GENERATED entry point.
///
/// It is the same three lines every time, because a generated entry point has
/// one signature: the record, the arena base table, the half-open range. What
/// it adds over calling the symbol directly is the compile-time check that this
/// record really is the shape `bind_generated` assumes, which is a positional
/// assumption that no signature can carry.
macro_rules! generated_thunk {
    ($name:ident, $record:ty, $symbol:ident) => {
        #[cfg_attr(abi_backend_linked, allow(dead_code))]
        unsafe fn $name(
            args: *const u8,
            begin: u32,
            end: u32,
            _group_width: u32,
            _diag: *mut DiagRecord,
        ) {
            const _: () = assert!(
                std::mem::offset_of!($record, seam_arena_count)
                    == std::mem::size_of::<$record>() - 4,
                "a generated record carries arena_count as its last field"
            );
            const _: () = assert!(std::mem::size_of::<$record>() <= MAX_BOUND_ARGS);
            let bound = &*args.cast::<BoundArgs>();
            $symbol(bound.args_ptr(), bound.bases_ptr(), begin, end);
        }
    };
}

/// As `generated_thunk!`, for an entry that declares a diagnostic lane.
///
/// The two differ in one parameter and are two macros rather than one with an
/// optional tail, because the lane is not an option a call site may forget: an
/// entry either declares `[[seam::diag]]` or it does not, and the generator
/// renders a different signature for each. Passing the channel to an entry that
/// does not take it, or dropping it from one that does, is then a name error at
/// the `extern` rather than a silent argument-count mismatch across the ABI.
macro_rules! generated_thunk_diag {
    ($name:ident, $record:ty, $symbol:ident) => {
        #[cfg_attr(abi_backend_linked, allow(dead_code))]
        unsafe fn $name(
            args: *const u8,
            begin: u32,
            end: u32,
            _group_width: u32,
            diag: *mut DiagRecord,
        ) {
            const _: () = assert!(
                std::mem::offset_of!($record, seam_arena_count)
                    == std::mem::size_of::<$record>() - 4,
                "a generated record carries arena_count as its last field"
            );
            const _: () = assert!(std::mem::size_of::<$record>() <= MAX_BOUND_ARGS);
            let bound = &*args.cast::<BoundArgs>();
            $symbol(bound.args_ptr(), bound.bases_ptr(), begin, end, diag);
        }
    };
}

/// As `generated_thunk!`, for a `[[seam::group]]` entry.
///
/// THE GROUP SHIM TAKES THE WIDTH AND THE ELEMENT SHIM DOES NOT, and the
/// generator renders the two signatures differently: a group shim's lanes run 0
/// through `width - 1` and it has no other way to know how many there are. So
/// this is a third macro rather than a flag on the first, for the reason
/// `generated_thunk_diag!` is a second: passing a width to an entry that does
/// not take it, or dropping it from one that does, is then a name error at the
/// `extern` rather than an argument-count mismatch across the ABI that shifts
/// `begin` into the width's place and reads garbage as the range.
macro_rules! generated_thunk_group {
    ($name:ident, $record:ty, $symbol:ident) => {
        #[cfg_attr(abi_backend_linked, allow(dead_code))]
        unsafe fn $name(
            args: *const u8,
            begin: u32,
            end: u32,
            group_width: u32,
            _diag: *mut DiagRecord,
        ) {
            const _: () = assert!(
                std::mem::offset_of!($record, seam_arena_count)
                    == std::mem::size_of::<$record>() - 4,
                "a generated record carries arena_count as its last field"
            );
            const _: () = assert!(std::mem::size_of::<$record>() <= MAX_BOUND_ARGS);
            let bound = &*args.cast::<BoundArgs>();
            $symbol(bound.args_ptr(), bound.bases_ptr(), group_width, begin, end);
        }
    };
}

// One wrapper per generated entry point, rendered by
// `ppf-cts-compute/seam/kernelgen.py --emit thunks` and written to `OUT_DIR` by
// this crate's `build.rs`.
//
// Every line of a thunk is a restatement of the entry declaration beside the
// kernel body: which macro, from whether the entry is group-shaped or declares
// a `[[seam::diag]]` lane; the argument record, from the entry's name in camel
// case; the wrapper's own name, from that name under a `launch_` prefix. So the
// block is derived rather than written, and a wrapper cannot name a record its
// declaration has stopped carrying.
//
// The table covers every entry point the tree DECLARES, including those this
// driver never dispatches. A library built from this tree carries all of them,
// and a kernel id is a table INDEX, so the launch table and the id table must
// agree position by position whether or not a row is reached.
//
// The four narrow-phase entries declare a diagnostic lane, so their wrappers
// take the channel as a fifth argument. That is the `[[seam::diag]]` attribute
// arriving in the signature, and it is the declaration's business rather than
// the caller's: which channel a dispatch reports through is a property of the
// process, not of the kernel.
include!(concat!(env!("OUT_DIR"), "/launch_thunks.rs"));

/// One thunk per id, in id order.
#[cfg_attr(abi_backend_linked, allow(dead_code))]
static LAUNCH: [Launch; id::COUNT] = [
    launch_aabb_leaf_face,
    launch_aabb_leaf_edge,
    launch_aabb_leaf_vertex,
    launch_aabb_leaf_active,
    launch_aabb_point_contact_query,
    launch_aabb_point_contact_query_masked,
    launch_aabb_edge_contact_query,
    launch_aabb_edge_contact_query_masked,
    launch_aabb_edge_scan_query,
    launch_aabb_edge_scan_query_masked,
    launch_aabb_vertex_scan_query,
    launch_aabb_vertex_scan_query_masked,
    launch_aabb_merge_level,
    launch_aabb_query_pairs,
    launch_combine_friction_values,
    launch_overlap_first_flagged_leaf,
    launch_ccd_point_face,
    launch_ccd_point_point,
    launch_ccd_edge_edge,
    launch_ccd_collision_point_face_m2c,
    launch_ccd_collision_point_face_c2m,
    launch_ccd_collision_edge_edge,
    launch_collision_point_face_m2c,
    launch_collision_point_face_c2m,
    launch_collision_edge_edge,
    launch_collision_point_face_m2c_traverse,
    launch_collision_point_face_c2m_traverse,
    launch_collision_edge_edge_traverse,
    launch_contact_fixed_slot,
    launch_contact_embed_hessian_blocks,
    launch_contact_embed_force_terms,
    launch_contact_point_face_traverse,
    launch_contact_point_face,
    launch_contact_point_edge_traverse,
    launch_contact_point_edge,
    launch_contact_point_point_traverse,
    launch_contact_point_point,
    launch_contact_edge_edge_traverse,
    launch_contact_edge_edge,
    launch_intersect_scan_face_edge,
    launch_intersect_scan_edge_edge,
    launch_intersect_scan_point_point,
    launch_intersect_scan_collision_mesh,
    launch_pair_cache_record,
    launch_pair_cache_record_interleaved,
    launch_vertex_constraint,
    launch_vertex_constraint_sweep,
    launch_dyn_count_transpose_pass,
    launch_dyn_scatter_transpose_pass,
    launch_dyn_row_begin_pass,
    launch_dyn_dry_push_pass,
    launch_dyn_row_seed_pass,
    launch_dyn_push_pass,
    launch_dyn_row_compact_pass,
    launch_dyn_row_emit_pass,
    launch_fixed_push_element_blocks_at,
    launch_fixed_push_element_blocks_gated_at,
    launch_fixed_push_element_blocks_live,
    launch_fixed_push_element_blocks_gated,
    launch_fixed_push_element_blocks,
    launch_fixed_csr_atomic_push,
    launch_precond_diagonal,
    launch_precond_diagonal_dynamic,
    launch_face_spectral_force,
    launch_face_spectral_hessian,
    launch_tet_spectral_force,
    launch_tet_spectral_hessian,
    launch_external_field,
    launch_face_elastic_embed_from_records,
    launch_face_baraffwitkin,
    launch_face_pressure_embed,
    launch_friction_evaluate,
    launch_tet_material_diff_table,
    launch_face_material_diff_table,
    launch_pdrd_project_body_dofs_row,
    launch_pdrd_copy_state_rotation_row,
    launch_pdrd_compose_running_rotation_row,
    launch_pdrd_prolong_row,
    launch_pdrd_restrict_row,
    launch_pdrd_seed_restrict_row,
    launch_pdrd_copy_projected_cloth_row,
    launch_pdrd_translation_lock_particular_row,
    launch_pdrd_extract_body_rotation_row,
    launch_pdrd_scatter_rotated_rest_row,
    launch_pdrd_precond_body_row,
    launch_pdrd_precond_cloth_row,
    launch_pdrd_rigidify_centroid_row,
    launch_pdrd_rigidify_write_row,
    launch_pdrd_fit_centroid_row,
    launch_pdrd_fit_covariance_row,
    launch_pdrd_fit_finish_row,
    launch_pdrd_assemble_inertia_row,
    launch_pdrd_assemble_sandwich_row,
    launch_push_energy,
    launch_push_curvature,
    launch_push_gradient,
    launch_push_hessian,
    launch_rod_bend_angle,
    launch_rod_bend_force_hessian,
    launch_rod_bend_embed,
    launch_rod_bend_stiffness,
    launch_sand_grain_integrate_row,
    launch_sand_grain_condense_row,
    launch_sand_grain_recover_row,
    launch_shell_bend_force_hessian_checked,
    launch_shell_bend_embed,
    launch_shell_bend_angle,
    launch_shell_bend_remap,
    launch_shell_bend_stiffness_and_damping,
    launch_shell_bend_stiffness,
    launch_shell_bend_areal_density_from_records,
    launch_shell_bend_areal_density_gathered,
    launch_stitch_force_hessian_gathered,
    launch_torque_group_frame,
    launch_rod_stretch_diff_table,
    launch_rod_stretch_embed,
    launch_tet_spectral_convert_hessian,
    launch_tet_elastic_embed,
    launch_tet_material_from_records,
    launch_bitonic_step,
    launch_lbvh_morton_from_bounds,
    launch_lbvh_nodes,
    launch_lbvh_node_depth,
    launch_face_centroid,
    launch_edge_centroid,
    launch_vertex_centroid,
    launch_lbvh_set_parent,
    launch_lbvh_find_root,
    launch_lbvh_count_levels,
    launch_lbvh_scatter_levels,
    launch_dirichlet_prescribe_gated,
    launch_dirichlet_lift_row,
    launch_dump_linsys_row_to_coo,
    launch_dx_magnitude,
    launch_dx_seed,
    launch_fix_xz_drag,
    launch_momentum_embed,
    launch_gather_position_absolute,
    launch_override_velocity_seed_listed,
    launch_override_angular_seed_listed,
    launch_position_accept,
    launch_position_step,
    launch_rewind_fix,
    launch_rod_stretch_ratio_gated,
    launch_compute_target_seed,
    launch_velocity_terms,
    launch_plasticity_alpha,
    launch_plasticity_face_inverse_rest,
    launch_plasticity_tet_inverse_rest,
    launch_plasticity_creep_singular2,
    launch_plasticity_creep_singular3,
    launch_plasticity_creep_rest_angle,
    launch_plasticity_commit_face,
    launch_plasticity_commit_tet,
    launch_plasticity_face_from_records,
    launch_plasticity_hinge_from_records,
    launch_plasticity_tet_from_records,
    launch_plasticity_rod_from_records,
    launch_radix_histogram,
    launch_radix_scatter,
    launch_bounds_leaf,
    launch_bounds_merge,
    launch_reduce_min_leaf,
    launch_reduce_max_leaf,
    launch_reduce_min_u32_leaf,
    launch_reduce_sum_u32_leaf,
    launch_reduce_sum_wide_merge,
    launch_scan_block_total,
    launch_scan_block_apply,
    launch_scan_zero,
    launch_vec_combine_indirect,
    launch_vec_add_scaled_indirect,
    launch_vec_copy,
    launch_vec_add_scaled,
    launch_vec_combine,
    launch_vec_fill,
    launch_vec_fill_u32,
    launch_element_add_scaled,
    launch_vec_block_sum,
    launch_vec_block_sum_u32,
    launch_vec_block_sum_cooperative,
    launch_vec_block_sum_abs_cooperative,
    launch_vec_block_sum_pair_cooperative,
    launch_vec_block_sum_dual_cooperative,
    launch_vec_block_sum_abs,
    launch_vec_block_sum_pair,
    launch_vec_block_sum_dual,
    launch_schwarz_count_members,
    launch_schwarz_scatter_members,
    launch_schwarz_domain_inverse_size,
    launch_schwarz_fine_graph_count,
    launch_schwarz_fine_graph_fill,
    launch_schwarz_factor_gather,
    launch_schwarz_factor_floor,
    launch_schwarz_factor_cholesky_diagonal,
    launch_schwarz_factor_cholesky_column,
    launch_schwarz_factor_inverse_column,
    launch_schwarz_factor_pack,
    launch_schwarz_apply_gather,
    launch_schwarz_apply_lower,
    launch_schwarz_apply_upper,
    launch_schwarz_restrict_row,
    launch_schwarz_prolong_row,
    launch_schwarz_compose_map_row,
    launch_schwarz_level0_count,
    launch_schwarz_level0_fill,
    launch_schwarz_coarse_gather,
    launch_schwarz_galerkin_key,
    launch_schwarz_galerkin_edge_flag,
    launch_schwarz_galerkin_edge_head,
    launch_schwarz_galerkin_segment_sum,
    launch_block_jacobi_invert_row,
    launch_pcg_dot_terms,
    launch_pcg_update_row,
    launch_pcg_update_row_folded,
    launch_pcg_alpha_terms,
    launch_pcg_beta_terms,
    launch_pcg_alpha_resident,
    launch_pcg_fold_alpha,
    launch_pcg_beta_resident,
    launch_pcg_fold_beta,
    launch_pcg_rigid_group_l1,
    launch_operator_apply,
    launch_operator_apply_dynamic,
    launch_operator_apply_folded,
    launch_operator_apply_dynamic_folded,
    launch_operator_apply_symmetric_folded,
    launch_mat3_mul,
    launch_fixed_csr_product_row,
    launch_translation_lock_drift_row,
    launch_lock_frame_clear_row,
    launch_lock_frame_center_of_mass_row,
    launch_lock_center_of_mass_accumulate_row,
    launch_lock_inertia_accumulate_row,
    launch_lock_row_sums_accumulate_row,
    launch_lock_refine_toward_rhs_row,
    launch_lock_project_out_rows_row,
    launch_lock_seed_free_solution_row,
    launch_lock_torque_accumulate_row,
    launch_lock_constraint_assemble_row,
    launch_rod_strain_force_hessian_gated,
    launch_rod_strain_stiffness_gated,
    launch_shell_strain_diff_table_from_records,
    launch_shell_strain_diff_table_gated,
    launch_shell_strain_stiffness_from_records,
    launch_shell_strain_stiffness_gated,
    launch_shell_strain_embed,
    launch_shell_max_strain,
    launch_rod_strain_value,
    launch_shell_strain_toi_from_records,
    launch_shell_strain_toi_gated,
    launch_rod_strain_toi_gated,
    launch_collision_window_vertex,
    launch_collision_window_face,
    launch_collision_window_edge,
    launch_face_convert_force,
    launch_face_convert_hessian,
    launch_face_damping,
    launch_face_deformation_gradient,
    launch_shell_stretch_terms,
    launch_face_atomic_embed_hessian_slots,
    launch_face_atomic_embed_force,
    launch_face_live_embed_force,
    launch_face_active_embed_force,
    launch_hinge_damping,
    launch_hinge_atomic_embed_force,
    launch_hinge_live_embed_force,
    launch_hinge_active_embed_force,
    launch_hinge_atomic_embed_hessian_slots,
    launch_rod_bend_damping,
    launch_rod_damping,
    launch_rod_atomic_embed_force,
    launch_rod_live_embed_force,
    launch_rod_active_embed_force,
    launch_rod_packed_embed_force,
    launch_rod_atomic_embed_hessian_slots,
    launch_stitch_atomic_embed_force,
    launch_svd3x2,
    launch_svd3x2_shifted,
    launch_shell_strain_restore_sigma,
    launch_svd3x3_rv,
    launch_svd3x3,
    launch_tet_convert_force,
    launch_tet_convert_hessian,
    launch_tet_shape_gradients,
    launch_tet_deformation_gradient,
    launch_tet_damping,
    launch_vertex_normal_finalize,
    launch_vertex_atomic_embed_force,
    launch_vertex_fix_index_from_records,
    launch_vertex_dof_removal_mask,
];

// ===========================================================================
// The device.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::kernels::VecAddScaledArgs;
    use ppf_cts_compute::host::bind_generated;
    use ppf_cts_compute::{AllocLabel, Device, EncoderExt, Extent, Fault};

    /// Both vectors are DEVICE allocations, because the record's two buffer
    /// fields are handles: this test is about the record and the extent agreeing
    /// on a count, and the arrays only have to exist somewhere the dispatch can
    /// resolve.
    fn add_scaled_args(
        source: ppf_cts_compute::Handle,
        destination: ppf_cts_compute::Handle,
        scale: f32,
        count: u32,
    ) -> VecAddScaledArgs {
        VecAddScaledArgs {
            source,
            destination,
            scale,
            count,
            seam_arena_count: 0,
        }
    }

    /// A device vector holding `values`.
    fn device_vec(
        device: &mut super::HostDevice,
        values: &[f32],
        label: &'static str,
    ) -> ppf_cts_compute::Buffer<f32> {
        let mut buffer = ppf_cts_compute::Buffer::<f32>::none();
        buffer
            .size(device, values.len(), AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
            .write(device, 0, values)
            .expect("the test upload succeeds");
        buffer
    }

    #[test]
    fn the_record_and_the_extent_must_agree_about_the_count() {
        let mut device = host_device();
        let source_host = vec![1.0f32, 2.0, 3.0, 4.0];
        let destination_host = vec![10.0f32; 4];
        let source = device_vec(&mut device, &source_host, "test.src");
        let mut destination = device_vec(&mut device, &destination_host, "test.dst");
        // A generated entry point clamps its loop to the count in its OWN
        // record, because Metal never faults out of bounds and every backend
        // rounds a launch up to whole threadgroups. So a record whose count
        // disagrees with the extent runs the smaller of the two and reports
        // nothing, which is work silently not done.
        let mut args = add_scaled_args(source.handle(), destination.handle(), 2.0, 4);
        args.count = 2;
        let result = device.run("test", |e| unsafe { e.elements(&args, 4) });
        match result {
            Err(Fault::Shape { kernel, detail }) => {
                assert_eq!(kernel, kernels::VEC_ADD_SCALED_NAME);
                assert!(detail.contains('2') && detail.contains('4'), "{detail}");
            }
            other => panic!("a disagreeing guard count must be refused, got {other:?}"),
        }
        let mut back = vec![0.0f32; 4];
        destination
            .read(&mut device, 0, &mut back)
            .expect("the destination reads back");
        assert_eq!(back, vec![10.0f32; 4], "nothing may have run");
    }

    #[test]
    fn binding_gives_every_buffer_its_own_arena_and_a_live_count() {
        // THE SUBJECT IS THE HOST-REFERENCE BINDING, so this names a record that
        // still HAS host references. `VecAddScaledArgs` served here until its two
        // buffers moved to handles, at which point `decl.host_refs` went empty
        // and the loop below would have asserted nothing at all: a pass bought by
        // the coverage vanishing. `vec_copy` has the same two-buffer shape and
        // still binds by address, so the machinery is exercised as before.
        let device = host_device();
        let source = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut destination = vec![0.0f32; 4];
        let args = kernels::VecCopyArgs {
            source: ppf_cts_compute::HostRef::of(&source),
            destination: ppf_cts_compute::HostRef::of_mut(&mut destination),
            count: source.len() as u32,
            seam_arena_count: 0,
        };
        let decl = device.decl(kernels::id::VEC_COPY).expect("declared");
        assert!(decl.generated, "this kernel's entry point is generated");
        // Safety: the record is live and its two references name live slices.
        // No arena is open on this device, so the two references take slots 0
        // and 1 and the positional half of the binding is visible on its own.
        let bound = unsafe {
            bind_generated(decl, (&args as *const kernels::VecCopyArgs).cast::<u8>(), &[])
        };
        assert_eq!(bound.base(0), source.as_ptr() as *mut u8);
        assert_eq!(bound.base(1), destination.as_ptr() as *mut u8);
        for (arena, &offset) in decl.host_refs.iter().enumerate() {
            let handle: [u32; 4] = unsafe {
                std::ptr::read_unaligned(bound.args_ptr().add(offset as usize).cast())
            };
            assert_eq!(handle[0], arena as u32, "one buffer, one arena");
            assert_eq!(handle[1], 0, "the whole allocation is the arena");
            assert_eq!(handle[2], 16, "four floats");
        }
        let width = decl.args_bytes as usize;
        let live: u32 =
            unsafe { std::ptr::read_unaligned(bound.args_ptr().add(width - 4).cast()) };
        assert_eq!(live, 2, "the entry point checks its arena ids against this");
    }

    #[test]
    fn a_dispatch_reaches_the_shared_body() {
        let mut device = host_device();
        let source_host = vec![1.0f32, 2.0, 3.0, 4.0];
        let source = device_vec(&mut device, &source_host, "test.src");
        let destination_host = vec![10.0f32; 4];
        let mut destination =
            device_vec(&mut device, &destination_host, "test.dst");
        let args = add_scaled_args(source.handle(), destination.handle(), 2.0, source_host.len() as u32);
        // Safety: both buffers outlive the call and neither is aliased.
        unsafe { device.launch("test", &args, 4) }.expect("the dispatch must run");
        let mut back = vec![0.0f32; destination_host.len()];
        destination
            .read(&mut device, 0, &mut back)
            .expect("the destination reads back");
        assert_eq!(back, vec![12.0, 14.0, 16.0, 18.0]);
        assert_eq!(device.counters().dispatches, 1);
        assert_eq!(device.counters().syncs, 1);
    }

    #[test]
    fn a_record_whose_length_disagrees_with_the_declaration_is_refused() {
        // The record and the shim's parameter list are two declarations that can
        // disagree, so the seam checks the only thing it can see: the length.
        let mut device = host_device();
        let source_host = vec![1.0f32; 4];
        let source = device_vec(&mut device, &source_host, "test.src");
        let destination_host = vec![0.0f32; 4];
        let mut destination =
            device_vec(&mut device, &destination_host, "test.dst");
        let args = add_scaled_args(source.handle(), destination.handle(), 1.0, source_host.len() as u32);
        let result = device.run("test", |e| {
            // Safety: the pointer is valid; the LENGTH is deliberately wrong.
            unsafe {
                e.dispatch_raw(
                    kernels::id::VEC_ADD_SCALED,
                    Extent::Elements { count: 4 },
                    (&args as *const VecAddScaledArgs).cast::<u8>(),
                    7,
                )
            }
        });
        match result {
            Err(Fault::Shape { kernel, detail }) => {
                // The ENTRY POINT's name, which is what a generated
                // declaration exports and what every rendering of it spells,
                // rather than the neutral body it wraps.
                assert_eq!(kernel, kernels::VEC_ADD_SCALED_NAME);
                assert!(detail.contains('7'), "the fault must name the length: {detail}");
            }
            other => panic!("a wrong record length must be refused, got {other:?}"),
        }
        let mut back = vec![0.0f32; 4];
        destination
            .read(&mut device, 0, &mut back)
            .expect("the destination reads back");
        assert_eq!(back, vec![0.0f32; 4], "nothing may have run");
    }

    #[test]
    fn a_half_wired_buffer_reference_is_refused() {
        let mut device = host_device();
        // THE SUBJECT IS A HOST REFERENCE, so this names a record that still has
        // one: `VecAddScaledArgs` served here until its buffers became handles,
        // and a handle has no half-wired form to build.
        let source_host = vec![1.0f32; 4];
        let mut destination_host = vec![0.0f32; 4];
        let mut args = kernels::VecCopyArgs {
            source: ppf_cts_compute::HostRef::of(&source_host),
            destination: ppf_cts_compute::HostRef::of_mut(&mut destination_host),
            count: 4,
            seam_arena_count: 0,
        };
        // A length with no address: what a hand-filled record produces when a
        // count is set and the pointer beside it is not.
        args.source = unsafe { ppf_cts_compute::HostRef::at::<f32>(std::ptr::null(), 4) };
        let result = device.run("test", |e| unsafe { e.elements(&args, 4) });
        assert!(
            matches!(result, Err(Fault::Shape { .. })),
            "a half-wired reference must be refused"
        );
    }

    #[test]
    fn the_answer_does_not_depend_on_how_the_seam_cuts_the_range() {
        // The whole point of cutting by a FIXED chunk width rather than by the
        // thread count. Run the same dispatch at a size the scheduler runs
        // serially and at one it parallelizes, against a reference computed one
        // element at a time.
        let mut device = host_device();
        for count in [4usize, 400_000] {
            let source_host: Vec<f32> = (0..count).map(|i| (i % 17) as f32 * 0.25).collect();
            let destination_host: Vec<f32> = (0..count).map(|i| (i % 7) as f32).collect();
            let expected: Vec<f32> = source_host
                .iter()
                .zip(destination_host.iter())
                .map(|(s, d)| d + 1.5 * s)
                .collect();
            let source = device_vec(&mut device, &source_host, "test.src");
            let mut destination = device_vec(&mut device, &destination_host, "test.dst");
            let args = add_scaled_args(
                source.handle(),
                destination.handle(),
                1.5,
                count as u32,
            );
            unsafe { device.launch("test", &args, count as u32) }.expect("dispatch");
            let mut back = vec![0.0f32; count];
            destination
                .read(&mut device, 0, &mut back)
                .expect("the destination reads back");
            assert_eq!(back, expected, "at {count} elements");
        }
    }

    #[test]
    fn a_region_replays_without_one_sync_per_repeat() {
        // The property the deferred construct exists for: N repeats, ONE host
        // round trip. Measured on the counter rather than asserted in prose.
        let mut device = host_device();
        let source_host = vec![1.0f32; 8];
        let source = device_vec(&mut device, &source_host, "test.src");
        let destination_host = vec![0.0f32; 8];
        let mut destination =
            device_vec(&mut device, &destination_host, "test.dst");
        let args = add_scaled_args(source.handle(), destination.handle(), 1.0, source_host.len() as u32);
        let region = device
            .record("test.iteration", |e| unsafe { e.elements(&args, 8) })
            .expect("record");
        device.counters_reset();
        device.replay(&region, 10).expect("replay");
        let mut back = vec![0.0f32; destination_host.len()];
        destination
            .read(&mut device, 0, &mut back)
            .expect("the destination reads back");
        assert_eq!(back, vec![10.0f32; 8], "ten repeats must have run");
        assert_eq!(device.counters().syncs, 1, "ten repeats, one host round trip");
        assert_eq!(device.counters().dispatches, 10);
        assert_eq!(device.counters().regions_deferred, 1);
        assert_eq!(device.counters().regions_fallback, 0);
        device.release(region);
    }

    #[test]
    fn a_backend_that_cannot_record_still_runs_and_says_so() {
        // CUDA falls back from a failed graph capture to direct launches with
        // bit-identical results and only a slowdown, which no value gate can
        // see. The counter is the only thing that can, so it is asserted.
        let mut device = host_device();
        device.set_recording_enabled(false);
        let source_host = vec![2.0f32; 8];
        let source = device_vec(&mut device, &source_host, "test.src");
        let destination_host = vec![0.0f32; 8];
        let mut destination =
            device_vec(&mut device, &destination_host, "test.dst");
        let args = add_scaled_args(source.handle(), destination.handle(), 1.0, source_host.len() as u32);
        let region = device
            .record("test.iteration", |e| unsafe { e.elements(&args, 8) })
            .expect("a backend that cannot record must still build a region");
        device.counters_reset();
        device.replay(&region, 3).expect("replay");
        let mut back = vec![0.0f32; destination_host.len()];
        destination
            .read(&mut device, 0, &mut back)
            .expect("the destination reads back");
        assert_eq!(
            back,
            vec![6.0f32; 8],
            "the fallback must produce the same answer"
        );
        assert_eq!(device.counters().regions_fallback, 1);
        assert_eq!(device.counters().regions_deferred, 0);
    }

    #[test]
    fn a_region_recorded_before_the_allocator_moved_is_refused() {
        let mut device = host_device();
        let source_host = vec![1.0f32; 8];
        let source = device_vec(&mut device, &source_host, "test.src");
        let destination_host = vec![0.0f32; 8];
        let mut destination =
            device_vec(&mut device, &destination_host, "test.dst");
        let args = add_scaled_args(source.handle(), destination.handle(), 1.0, source_host.len() as u32);
        let region = device
            .record("test.iteration", |e| unsafe { e.elements(&args, 8) })
            .expect("record");
        let mut handle = device
            .alloc(16, 4, 4, AllocLabel("scratch"))
            .expect("alloc");
        match device.replay(&region, 1) {
            Err(Fault::StaleRegion { recorded, now, .. }) => assert!(recorded < now),
            other => panic!("a region recorded before an allocation must be refused: {other:?}"),
        }
        device.free(&mut handle).expect("free");
    }

    #[test]
    fn allocation_round_trips_through_the_seam() {
        let mut device = host_device();
        let mut handle = device
            .alloc(64, 4, 4, AllocLabel("scratch"))
            .expect("alloc");
        assert_eq!(handle.size, 64);
        let source: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
        device.write(handle, 0, &source).expect("write");
        let mut back = vec![0u8; 256];
        device.read(handle, 0, &mut back).expect("read");
        assert_eq!(back, source);
        device
            .grow(&mut handle, 128, 4, 4)
            .expect("grow must preserve contents");
        let mut kept = vec![0u8; 256];
        device.read(handle, 0, &mut kept).expect("read");
        assert_eq!(kept, source, "a grow must preserve what was written");
        device.free(&mut handle).expect("free");
        assert!(handle.is_none());
    }

    #[test]
    fn a_write_past_the_block_is_refused_rather_than_landing_elsewhere() {
        let mut device = host_device();
        let mut handle = device.alloc(4, 4, 4, AllocLabel("scratch")).expect("alloc");
        let result = device.write(handle, 0, &[0u8; 32]);
        assert!(matches!(result, Err(Fault::Platform { .. })));
        device.free(&mut handle).expect("free");
    }

    #[test]
    fn every_declared_kernel_has_a_launch() {
        // The table and the launch array are indexed by the same id, so a row
        // added to one and not the other is a dispatch of the wrong kernel with
        // the right bytes. The lengths are what make that impossible.
        assert_eq!(LAUNCH.len(), kernels::TABLE.len());
        let device = host_device();
        assert!(device.missing().is_empty());
        assert_eq!(device.kernels().len(), LAUNCH.len());
    }
}
