// File: crates/ppf-cts-solver/src/driver/kernels.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The kernel table: one declaration per entry point the driver dispatches.
//!
//! **WHAT IS NOT `include!`d HERE IS A STAND-IN FOR GENERATED CODE, AND NOTHING
//! IN THIS FILE MAY ACQUIRE LOGIC.** The rule is this: a contributor writes one
//! `[[seam::entry]]` declaration beside the neutral body, and
//! `ppf-cts-compute/seam/kernelgen.py --emit entry` renders the
//! argument record and the entry point for `cu`, `metal`, `cpp` and `rust`. The
//! `rust` target is exactly what a record below looks like, and the `include!`
//! lines are the ones that come from it. Every other record here is written by
//! hand, which is a fork by hand: a record here and the shim's parameter list
//! are two declarations that can disagree, and the only thing that catches a
//! disagreement is the C++ compiler refusing the thunk in
//! `super::launch`.
//!
//! What it may hold: an argument record, its [`KernelId`], and its
//! [`KernelDecl`]. What it may not hold: a branch, a phase order, a convergence
//! test, a physical quantity, or any arithmetic at all.
//!
//! # Why the table is here and the launch is not
//!
//! A declaration is neutral: the name, the scatter shape, the per-item cost and
//! the record layout are the same facts on every backend. The LAUNCH is not, so
//! it lives in the backend (`super::launch::LAUNCH`), which is what makes
//! this table the thing a second backend implements against rather than a thing
//! it re-derives.

// Records for phases not yet converted are absent rather than stubbed, so
// nothing here is unused; the allow covers the constructors a test does not
// reach.
#![allow(dead_code)]

use std::mem::{size_of};

use ppf_cts_compute::{Handle, HostRef, KernelArgs, KernelDecl, KernelId, Scatter};

/// The ids, dense from zero so the table can be indexed by them.
pub mod id {
    use super::KernelId;
    pub const AABB_LEAF_FACE: KernelId = KernelId(0);
    pub const AABB_LEAF_EDGE: KernelId = KernelId(1);
    pub const AABB_LEAF_VERTEX: KernelId = KernelId(2);
    pub const AABB_LEAF_ACTIVE: KernelId = KernelId(3);
    pub const AABB_POINT_CONTACT_QUERY: KernelId = KernelId(4);
    pub const AABB_POINT_CONTACT_QUERY_MASKED: KernelId = KernelId(5);
    pub const AABB_EDGE_CONTACT_QUERY: KernelId = KernelId(6);
    pub const AABB_EDGE_CONTACT_QUERY_MASKED: KernelId = KernelId(7);
    pub const AABB_EDGE_SCAN_QUERY: KernelId = KernelId(8);
    pub const AABB_EDGE_SCAN_QUERY_MASKED: KernelId = KernelId(9);
    pub const AABB_VERTEX_SCAN_QUERY: KernelId = KernelId(10);
    pub const AABB_VERTEX_SCAN_QUERY_MASKED: KernelId = KernelId(11);
    pub const AABB_MERGE_LEVEL: KernelId = KernelId(12);
    /// The broad phase, one thread per query box.
    ///
    /// DISPATCHED. The traversal itself runs on the DEVICE: `aabb_query_pairs`
    /// is a `[[seam::device_fn]]` in `kernels/contact/aabb_traversal.kernel.cpp`
    /// and this entry calls it in-thread, so neither tree array needs host
    /// residency. A host-side walk over rayon would pin both.
    pub const AABB_QUERY_PAIRS: KernelId = KernelId(13);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const COMBINE_FRICTION_VALUES: KernelId = KernelId(14);
    /// The CCD line search's six fused sweeps, one per (query kind, tree)
    /// pair. Each walks a BVH with the ACCD advance as a per-hit device
    /// functor and writes one slot of a per-primitive time-of-impact array.
    pub const CCD_POINT_FACE: KernelId = KernelId(16);
    pub const CCD_POINT_POINT: KernelId = KernelId(17);
    pub const CCD_EDGE_EDGE: KernelId = KernelId(18);
    pub const CCD_COLLISION_POINT_FACE_M2C: KernelId = KernelId(19);
    pub const CCD_COLLISION_POINT_FACE_C2M: KernelId = KernelId(20);
    pub const CCD_COLLISION_EDGE_EDGE: KernelId = KernelId(21);
    pub const COLLISION_POINT_FACE_M2C: KernelId = KernelId(22);
    pub const COLLISION_POINT_FACE_C2M: KernelId = KernelId(23);
    pub const COLLISION_EDGE_EDGE: KernelId = KernelId(24);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const CONTACT_FIXED_SLOT: KernelId = KernelId(28);
    pub const CONTACT_POINT_FACE: KernelId = KernelId(32);
    pub const CONTACT_POINT_FACE_TRAVERSE: KernelId = KernelId(31);
    pub const CONTACT_POINT_EDGE: KernelId = KernelId(34);
    pub const CONTACT_POINT_EDGE_TRAVERSE: KernelId = KernelId(33);
    pub const CONTACT_POINT_POINT: KernelId = KernelId(36);
    pub const CONTACT_POINT_POINT_TRAVERSE: KernelId = KernelId(35);
    pub const CONTACT_EDGE_EDGE: KernelId = KernelId(38);
    pub const CONTACT_EDGE_EDGE_TRAVERSE: KernelId = KernelId(37);
    /// The final penetration gate's four fused walks, one per (query kind,
    /// tree) pair. Each walks a BVH with the intersection tester as a per-hit
    /// device functor, sets this element's flag if the walk found anything, and
    /// claims a record slot out of one shared counter.
    pub const INTERSECT_SCAN_FACE_EDGE: KernelId = KernelId(39);
    pub const INTERSECT_SCAN_EDGE_EDGE: KernelId = KernelId(40);
    pub const INTERSECT_SCAN_POINT_POINT: KernelId = KernelId(41);
    pub const INTERSECT_SCAN_COLLISION_MESH: KernelId = KernelId(42);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const PAIR_CACHE_RECORD: KernelId = KernelId(43);
    pub const PAIR_CACHE_RECORD_INTERLEAVED: KernelId = KernelId(44);
    pub const VERTEX_CONSTRAINT: KernelId = KernelId(45);
    pub const VERTEX_CONSTRAINT_SWEEP: KernelId = KernelId(46);
    pub const DYN_COUNT_TRANSPOSE_PASS: KernelId = KernelId(47);
    pub const DYN_SCATTER_TRANSPOSE_PASS: KernelId = KernelId(48);
    /// The dynamic contact matrix, as six element-wise passes over flat device
    /// storage: order the carried pattern and open each row's reserve, count one
    /// slot per contribution the pattern does not already hold, lay the pattern
    /// into the slab, fill, compact, and hand on the merged pattern with the
    /// contiguous copy the sparse matvec reads. The two scans between those
    /// passes are `super::scan::ScanScratch::exclusive`.
    pub const DYN_ROW_BEGIN_PASS: KernelId = KernelId(49);
    pub const DYN_DRY_PUSH_PASS: KernelId = KernelId(50);
    pub const DYN_ROW_SEED_PASS: KernelId = KernelId(51);
    pub const DYN_PUSH_PASS: KernelId = KernelId(52);
    pub const DYN_ROW_COMPACT_PASS: KernelId = KernelId(53);
    pub const DYN_ROW_EMIT_PASS: KernelId = KernelId(54);
    pub const FIXED_CSR_ATOMIC_PUSH: KernelId = KernelId(60);
    pub const FIXED_PUSH_ELEMENT_BLOCKS: KernelId = KernelId(59);
    pub const FIXED_PUSH_ELEMENT_BLOCKS_AT: KernelId = KernelId(55);
    pub const FIXED_PUSH_ELEMENT_BLOCKS_GATED_AT: KernelId = KernelId(56);
    pub const FIXED_PUSH_ELEMENT_BLOCKS_GATED: KernelId = KernelId(58);
    pub const FIXED_PUSH_ELEMENT_BLOCKS_LIVE: KernelId = KernelId(57);
    pub const PRECOND_DIAGONAL: KernelId = KernelId(61);
    pub const PRECOND_DIAGONAL_DYNAMIC: KernelId = KernelId(62);
    pub const FACE_SPECTRAL_FORCE: KernelId = KernelId(63);
    pub const FACE_SPECTRAL_HESSIAN: KernelId = KernelId(64);
    pub const TET_SPECTRAL_FORCE: KernelId = KernelId(65);
    pub const TET_SPECTRAL_HESSIAN: KernelId = KernelId(66);
    /// The shell membrane layer of ONE face: the deformation gradient, the
    /// SVD, the material table, the spectral force, the PSD-projected 6x6
    /// Hessian, the material-frame conversion, the Rayleigh damping block, the
    /// per-face pressure term and both scatters, in one dispatch, and it
    /// crosses to the host zero times.
    /// The same embed reading its material off `FaceProp` and `FaceParam`
    /// rather than off six arrays the host flattened and uploaded every pass.
    pub const FACE_ELASTIC_EMBED_FROM_RECORDS: KernelId = KernelId(68);
    pub const FACE_BARAFFWITKIN: KernelId = KernelId(69);
    /// The per-face inflation pressure, added to the face accumulators.
    ///
    /// DISPATCHED. It is the last term `refusal.rs` named that the elastic
    /// path did not already carry, and the body returns at once on a face whose
    /// own pressure is not positive.
    pub const FACE_PRESSURE_EMBED: KernelId = KernelId(70);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const FRICTION_EVALUATE: KernelId = KernelId(71);
    pub const TET_MATERIAL_DIFF_TABLE: KernelId = KernelId(72);
    pub const TET_MATERIAL_FROM_RECORDS: KernelId = KernelId(118);
    pub const FACE_MATERIAL_DIFF_TABLE: KernelId = KernelId(73);
    /// `pdrd_project_body_dofs_row`: each body's forbidden degrees of freedom
    /// removed from a reduced vector, which every vector the reduced solve
    /// touches passes through.
    pub const PDRD_PROJECT_BODY_DOFS_ROW: KernelId = KernelId(74);
    /// `pdrd_copy_state_rotation_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_COPY_STATE_ROTATION_ROW: KernelId = KernelId(75);
    /// `pdrd_compose_running_rotation_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_COMPOSE_RUNNING_ROTATION_ROW: KernelId = KernelId(76);
    /// `pdrd_prolong_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_PROLONG_ROW: KernelId = KernelId(77);
    /// `pdrd_restrict_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_RESTRICT_ROW: KernelId = KernelId(78);
    /// `pdrd_seed_restrict_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_SEED_RESTRICT_ROW: KernelId = KernelId(79);
    /// `pdrd_copy_projected_cloth_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_COPY_PROJECTED_CLOTH_ROW: KernelId = KernelId(80);
    /// `pdrd_translation_lock_particular_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_TRANSLATION_LOCK_PARTICULAR_ROW: KernelId = KernelId(81);
    /// `pdrd_extract_body_rotation_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_EXTRACT_BODY_ROTATION_ROW: KernelId = KernelId(82);
    /// `pdrd_scatter_rotated_rest_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_SCATTER_ROTATED_REST_ROW: KernelId = KernelId(83);
    /// `pdrd_precond_body_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_PRECOND_BODY_ROW: KernelId = KernelId(84);
    /// `pdrd_precond_cloth_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_PRECOND_CLOTH_ROW: KernelId = KernelId(85);
    /// `pdrd_rigidify_centroid_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_RIGIDIFY_CENTROID_ROW: KernelId = KernelId(86);
    /// `pdrd_rigidify_write_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_RIGIDIFY_WRITE_ROW: KernelId = KernelId(87);
    /// `pdrd_fit_centroid_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_FIT_CENTROID_ROW: KernelId = KernelId(88);
    /// `pdrd_fit_covariance_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_FIT_COVARIANCE_ROW: KernelId = KernelId(89);
    /// `pdrd_fit_finish_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_FIT_FINISH_ROW: KernelId = KernelId(90);
    /// `pdrd_assemble_inertia_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_ASSEMBLE_INERTIA_ROW: KernelId = KernelId(91);
    /// `pdrd_assemble_sandwich_row`, one row of the PDRD reduced six-DOF solve.
    pub const PDRD_ASSEMBLE_SANDWICH_ROW: KernelId = KernelId(92);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const PUSH_ENERGY: KernelId = KernelId(93);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const PUSH_CURVATURE: KernelId = KernelId(94);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const PUSH_GRADIENT: KernelId = KernelId(95);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const PUSH_HESSIAN: KernelId = KernelId(96);
    pub const ROD_BEND_ANGLE: KernelId = KernelId(97);
    pub const ROD_BEND_FORCE_HESSIAN: KernelId = KernelId(98);
    pub const ROD_BEND_EMBED: KernelId = KernelId(99);
    pub const ROD_BEND_STIFFNESS: KernelId = KernelId(100);
    /// One grain's post-solve spin integrate, from the converged friction torque.
    ///
    /// DISPATCHED by `super::step`, at the three points `main.cu` dispatches
    /// them: the condense after the whole Newton system is assembled, the
    /// recover straight after the solve, and the integrate after the commit.
    pub const SAND_GRAIN_INTEGRATE_ROW: KernelId = KernelId(101);
    /// One grain's spin condensed out of the Newton system, before the solve.
    ///
    /// DISPATCHED by `super::step`, at the three points `main.cu` dispatches
    /// them: the condense after the whole Newton system is assembled, the
    /// recover straight after the solve, and the integrate after the commit.
    pub const SAND_GRAIN_CONDENSE_ROW: KernelId = KernelId(102);
    /// One grain's angular velocity recovered from the solved translation increment.
    ///
    /// DISPATCHED by `super::step`, at the three points `main.cu` dispatches
    /// them: the condense after the whole Newton system is assembled, the
    /// recover straight after the solve, and the integrate after the commit.
    pub const SAND_GRAIN_RECOVER_ROW: KernelId = KernelId(103);
    pub const SHELL_BEND_FORCE_HESSIAN_CHECKED: KernelId = KernelId(104);
    pub const SHELL_BEND_EMBED: KernelId = KernelId(105);
    pub const SHELL_BEND_ANGLE: KernelId = KernelId(106);
    pub const SHELL_BEND_REMAP: KernelId = KernelId(107);
    pub const SHELL_BEND_STIFFNESS: KernelId = KernelId(109);
    pub const SHELL_BEND_STIFFNESS_AND_DAMPING: KernelId = KernelId(108);
    pub const SHELL_BEND_AREAL_DENSITY_GATHERED: KernelId = KernelId(111);
    pub const SHELL_BEND_AREAL_DENSITY_FROM_RECORDS: KernelId = KernelId(110);
    pub const STITCH_FORCE_HESSIAN_GATHERED: KernelId = KernelId(112);
    /// The number of ids [`super::TABLE`] and the backend's launch table cover.
    /// One torque group's centroid, principal axis and radius normalization.
    ///
    /// DISPATCHED, as the pre-pass `momentum` needs: the per-vertex torque
    /// force scales by its group's frame, so the frame must exist before any
    /// member's row is assembled. It is one thread per group, and the three
    /// walks cannot be fused: the covariance is about the centroid and the
    /// perpendicular radius is about the axis the covariance produces.
    pub const TORQUE_GROUP_FRAME: KernelId = KernelId(113);
    pub const ROD_STRETCH_DIFF_TABLE: KernelId = KernelId(114);
    pub const ROD_STRETCH_EMBED: KernelId = KernelId(115);
    /// The tet elastic layer of ONE element: the deformation gradient, its
    /// factorization, the material table, the spectral force, the fused 12x12
    /// Hessian, the Rayleigh damping block and the two scatters, in one
    /// dispatch, and it crosses to the host zero times.
    pub const TET_ELASTIC_EMBED: KernelId = KernelId(117);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const BITONIC_STEP: KernelId = KernelId(119);
    pub const LBVH_MORTON_FROM_BOUNDS: KernelId = KernelId(120);
    pub const LBVH_NODES: KernelId = KernelId(121);
    pub const LBVH_NODE_DEPTH: KernelId = KernelId(122);
    pub const FACE_CENTROID: KernelId = KernelId(123);
    pub const EDGE_CENTROID: KernelId = KernelId(124);
    pub const VERTEX_CENTROID: KernelId = KernelId(125);
    pub const DIRICHLET_PRESCRIBE_GATED: KernelId = KernelId(130);
    pub const DIRICHLET_LIFT_ROW: KernelId = KernelId(131);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const DUMP_LINSYS_ROW_TO_COO: KernelId = KernelId(132);
    pub const DX_MAGNITUDE: KernelId = KernelId(133);
    pub const DX_SEED: KernelId = KernelId(134);
    pub const FIX_XZ_DRAG: KernelId = KernelId(135);
    pub const MOMENTUM_EMBED: KernelId = KernelId(136);
    pub const GATHER_POSITION_ABSOLUTE: KernelId = KernelId(137);
    pub const OVERRIDE_VELOCITY_SEED_LISTED: KernelId = KernelId(138);
    pub const OVERRIDE_ANGULAR_SEED_LISTED: KernelId = KernelId(139);
    pub const POSITION_ACCEPT: KernelId = KernelId(140);
    pub const POSITION_STEP: KernelId = KernelId(141);
    pub const REWIND_FIX: KernelId = KernelId(142);
    pub const ROD_STRETCH_RATIO_GATED: KernelId = KernelId(143);
    pub const COMPUTE_TARGET_SEED: KernelId = KernelId(144);
    pub const EXTERNAL_FIELD: KernelId = KernelId(67);
    pub const VELOCITY_TERMS: KernelId = KernelId(145);
    pub const PLASTICITY_ALPHA: KernelId = KernelId(146);
    pub const PLASTICITY_FACE_FROM_RECORDS: KernelId = KernelId(154);
    pub const PLASTICITY_TET_FROM_RECORDS: KernelId = KernelId(156);
    pub const PLASTICITY_HINGE_FROM_RECORDS: KernelId = KernelId(155);
    pub const PLASTICITY_ROD_FROM_RECORDS: KernelId = KernelId(157);
    pub const PLASTICITY_FACE_INVERSE_REST: KernelId = KernelId(147);
    pub const PLASTICITY_TET_INVERSE_REST: KernelId = KernelId(148);
    pub const PLASTICITY_CREEP_SINGULAR2: KernelId = KernelId(149);
    pub const PLASTICITY_CREEP_SINGULAR3: KernelId = KernelId(150);
    pub const PLASTICITY_CREEP_REST_ANGLE: KernelId = KernelId(151);
    /// `p = z + beta p` with `beta` read from a device scalar.
    ///
    /// DISPATCHED. It stood here undispatched for a while, with the boilerplate
    /// note the ids below still carry, and the PCG direction update reaches it
    /// now that the recurrence keeps its coefficients on the device.
    ///
    /// **AN UNDISPATCHED ROW'S DECLARATION IS A PLACEHOLDER, AND IT BECOMES
    /// LOAD-BEARING THE DAY SOMETHING LAUNCHES IT.** This row said
    /// `Scatter::Atomic` and 1.0 ns, chosen because neither could be wrong
    /// about a kernel nobody launched. On this backend `Atomic` means ONE
    /// SERIAL ASCENDING PASS, so the first dispatch of it ran a `3 * vertices`
    /// vector update single-threaded beside chunked neighbors. Check the row
    /// against the BODY when you give a kernel its first caller.
    pub const VEC_COMBINE_INDIRECT: KernelId = KernelId(170);
    /// `x += alpha p` and `r -= alpha Ap` with `alpha` still on the device.
    pub const VEC_ADD_SCALED_INDIRECT: KernelId = KernelId(171);
    pub const VEC_COPY: KernelId = KernelId(172);
    pub const VEC_ADD_SCALED: KernelId = KernelId(173);
    pub const VEC_COMBINE: KernelId = KernelId(174);
    /// `vec_fill` writes one value to every element.
    ///
    /// DISPATCHED, unlike the ids below it. It is what SEEDS a device buffer the
    /// host would otherwise clear with `slice::fill`, which is the shape a time
    /// of impact has: seeded at the line-search ceiling, then reduced INTO by
    /// the kernel, then read back. Neither staged nor readback storage covers
    /// that on its own, so the seed is a dispatch rather than a fourth buffer
    /// type. CUDA's own `fill_kernel` reaches the same body.
    pub const VEC_FILL: KernelId = KernelId(175);
    /// DISPATCHED by `super::dyncsr`, at 12 call sites (dyncsr.rs:860, ...).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const VEC_FILL_U32: KernelId = KernelId(176);
    pub const ELEMENT_ADD_SCALED: KernelId = KernelId(177);
    /// One level of a blocked fold: one element sums one block of the input.
    ///
    /// A reduction reaches this backend only as a body written without a
    /// threadgroup barrier, which `crates/ppf-cts-compute/src/device.rs`
    /// states; a caller folds to a scalar by dispatching this level by level
    /// with kernel completion as the barrier.
    pub const VEC_BLOCK_SUM: KernelId = KernelId(178);
    pub const VEC_BLOCK_SUM_U32: KernelId = KernelId(179);
    pub const VEC_BLOCK_SUM_COOPERATIVE: KernelId = KernelId(180);
    pub const RADIX_HISTOGRAM: KernelId = KernelId(158);
    pub const RADIX_SCATTER: KernelId = KernelId(159);
    pub const VEC_BLOCK_SUM_ABS_COOPERATIVE: KernelId = KernelId(181);
    pub const VEC_BLOCK_SUM_PAIR_COOPERATIVE: KernelId = KernelId(182);
    pub const VEC_BLOCK_SUM_DUAL_COOPERATIVE: KernelId = KernelId(183);
    /// The same level over magnitudes, which only the FIRST level of an L1
    /// norm takes: every level above it folds totals that are already
    /// non-negative.
    pub const VEC_BLOCK_SUM_ABS: KernelId = KernelId(184);
    pub const VEC_BLOCK_SUM_PAIR: KernelId = KernelId(185);
    /// Two folds of DIFFERENT lengths in one dispatch, which the pair cannot
    /// express: each source carries its own length and its own block count and
    /// the dispatch covers the larger. The PCG's `||r||_1` and `r . z` chains
    /// are its callers.
    pub const VEC_BLOCK_SUM_DUAL: KernelId = KernelId(186);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:86).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_COUNT_MEMBERS: KernelId = KernelId(187);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:115).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_SCATTER_MEMBERS: KernelId = KernelId(188);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:125).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_DOMAIN_INVERSE_SIZE: KernelId = KernelId(189);
    /// DISPATCHED by `super::schwarz`, at 2 call sites (schwarz.rs:206, ...).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_FINE_GRAPH_COUNT: KernelId = KernelId(190);
    /// DISPATCHED by `super::schwarz`, at 2 call sites (schwarz.rs:229, ...).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_FINE_GRAPH_FILL: KernelId = KernelId(191);
    /// DISPATCHED by `super::schwarz`, at 2 call sites (schwarz.rs:306, ...).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_FACTOR_GATHER: KernelId = KernelId(192);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:370).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_FACTOR_FLOOR: KernelId = KernelId(193);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:381).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_FACTOR_CHOLESKY_DIAGONAL: KernelId = KernelId(194);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:398).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_FACTOR_CHOLESKY_COLUMN: KernelId = KernelId(195);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:410).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_FACTOR_INVERSE_COLUMN: KernelId = KernelId(196);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:421).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_FACTOR_PACK: KernelId = KernelId(197);
    /// DISPATCHED by `super::schwarz`, at 2 call sites (schwarz.rs:471, ...).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_APPLY_GATHER: KernelId = KernelId(198);
    /// DISPATCHED by `super::schwarz`, at 2 call sites (schwarz.rs:482, ...).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_APPLY_LOWER: KernelId = KernelId(199);
    /// DISPATCHED by `super::schwarz`, at 2 call sites (schwarz.rs:494, ...).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_APPLY_UPPER: KernelId = KernelId(200);
    /// DISPATCHED by `super::schwarz`, at 2 call sites (schwarz.rs:613, ...).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_RESTRICT_ROW: KernelId = KernelId(201);
    /// DISPATCHED by `super::schwarz`, at 2 call sites (schwarz.rs:643, ...).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_PROLONG_ROW: KernelId = KernelId(202);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:670).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_COMPOSE_MAP_ROW: KernelId = KernelId(203);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:725).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_LEVEL0_COUNT: KernelId = KernelId(204);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:749).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_LEVEL0_FILL: KernelId = KernelId(205);
    /// DISPATCHED by `super::schwarz`, at 1 call site (schwarz.rs:1132).
    ///
    /// ITS ROW BELOW IS THEREFORE LOAD-BEARING. A row for an entry nothing
    /// launches is written conservatively, because neither the scatter nor
    /// the per-item cost can be wrong about a kernel nobody reaches; both
    /// become real the day something does. Check the row against the BODY
    /// rather than against this line.
    pub const SCHWARZ_COARSE_GATHER: KernelId = KernelId(206);
    /// The Galerkin coarsening's four rows, all DISPATCHED by
    /// [`super::schwarz::galerkin`]: the keys and their identity permutation,
    /// the equal-key run flags, each run's head, and the segmented sum of the
    /// nine floats each run carries.
    pub const SCHWARZ_GALERKIN_KEY: KernelId = KernelId(207);
    pub const SCHWARZ_GALERKIN_EDGE_FLAG: KernelId = KernelId(208);
    pub const SCHWARZ_GALERKIN_EDGE_HEAD: KernelId = KernelId(209);
    pub const SCHWARZ_GALERKIN_SEGMENT_SUM: KernelId = KernelId(210);
    /// One row of the block-Jacobi preconditioner, inverted on the device.
    ///
    /// DISPATCHED. It replaces a per-row HOST loop that called the same shared
    /// body through a C-ABI helper, which is why the neutral body predates the
    /// entry point: the physics was already single-sourced and only the walk
    /// over the rows was the driver's.
    pub const BLOCK_JACOBI_INVERT_ROW: KernelId = KernelId(211);
    pub const PCG_DOT_TERMS: KernelId = KernelId(212);
    pub const PCG_UPDATE_ROW: KernelId = KernelId(213);
    /// One PDRD body's six reduced wrench rows, folded into an L1 norm on the
    /// DEVICE. The value is renumbered from the canonical walk by
    /// `renumber-kernel-ids.py`; what matters here is that the name exists.
    pub const PCG_RIGID_GROUP_L1: KernelId = KernelId(221);
    pub const PCG_UPDATE_ROW_FOLDED: KernelId = KernelId(214);
    pub const PCG_FOLD_ALPHA: KernelId = KernelId(218);
    pub const PCG_FOLD_BETA: KernelId = KernelId(220);
    pub const PCG_ALPHA_TERMS: KernelId = KernelId(215);
    pub const PCG_BETA_TERMS: KernelId = KernelId(216);
    /// The same two coefficients over scalars that never leave the device.
    ///
    /// They call the same `pcg_alpha` and `pcg_beta` the parameter forms
    /// above call, so a solve is classified by ONE rule however the numbers
    /// arrive; what differs is that a fold wrote these into a device buffer
    /// and reading them back to fill a `float` field would stall the
    /// recurrence once per coefficient.
    pub const PCG_ALPHA_RESIDENT: KernelId = KernelId(217);
    pub const PCG_BETA_RESIDENT: KernelId = KernelId(219);
    pub const OPERATOR_APPLY: KernelId = KernelId(222);
    pub const OPERATOR_APPLY_DYNAMIC: KernelId = KernelId(223);
    pub const OPERATOR_APPLY_FOLDED: KernelId = KernelId(224);
    pub const OPERATOR_APPLY_DYNAMIC_FOLDED: KernelId = KernelId(225);
    pub const OPERATOR_APPLY_SYMMETRIC_FOLDED: KernelId = KernelId(226);
    pub const MAT3_MUL: KernelId = KernelId(227);
    pub const FIXED_CSR_PRODUCT_ROW: KernelId = KernelId(228);
    /// One locked group's centroid drift and largest member displacement, read back to verify the lock held.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const TRANSLATION_LOCK_DRIFT_ROW: KernelId = KernelId(229);
    /// Zero one group's frame before the two accumulation passes fill it.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const LOCK_FRAME_CLEAR_ROW: KernelId = KernelId(230);
    /// One group's centroid, the accumulated sum divided by its total mass.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const LOCK_FRAME_CENTER_OF_MASS_ROW: KernelId = KernelId(231);
    /// One member's mass-weighted position into its group's centroid sum.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const LOCK_CENTER_OF_MASS_ACCUMULATE_ROW: KernelId = KernelId(232);
    /// One member's contribution to its group's inertia about that centroid.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const LOCK_INERTIA_ACCUMULATE_ROW: KernelId = KernelId(233);
    /// One member's contribution to C v, the constraint-space image of the vector being projected.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const LOCK_ROW_SUMS_ACCUMULATE_ROW: KernelId = KernelId(234);
    /// One residual correction of the free part toward the group's right-hand side.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const LOCK_REFINE_TOWARD_RHS_ROW: KernelId = KernelId(235);
    /// Remove the constraint-space component from one member's three rows.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const LOCK_PROJECT_OUT_ROWS_ROW: KernelId = KernelId(236);
    /// The affine feasible correction q: exact pin increments on removed rows, the minimum-norm free part elsewhere.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const LOCK_SEED_FREE_SOLUTION_ROW: KernelId = KernelId(237);
    /// One member's contribution to the group's best-fit angular increment, for the read-only tangent check.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const LOCK_TORQUE_ACCUMULATE_ROW: KernelId = KernelId(238);
    /// One group's constraint rows and their Gram matrix, from the frame.
    ///
    /// DISPATCHED by `super::lock`, which transcribes `FullProjector` in
    /// `src/kernels/solver/translation_lock.hpp` dispatch for dispatch.
    pub const LOCK_CONSTRAINT_ASSEMBLE_ROW: KernelId = KernelId(239);
    pub const ROD_STRAIN_FORCE_HESSIAN_GATED: KernelId = KernelId(240);
    pub const ROD_STRAIN_STIFFNESS_GATED: KernelId = KernelId(241);
    pub const SHELL_STRAIN_DIFF_TABLE_GATED: KernelId = KernelId(243);
    pub const SHELL_STRAIN_DIFF_TABLE_FROM_RECORDS: KernelId = KernelId(242);
    pub const SHELL_STRAIN_STIFFNESS_FROM_RECORDS: KernelId = KernelId(244);
    pub const SHELL_STRAIN_EMBED: KernelId = KernelId(246);
    pub const SHELL_STRAIN_TOI_FROM_RECORDS: KernelId = KernelId(249);
    pub const SHELL_STRAIN_STIFFNESS_GATED: KernelId = KernelId(245);
    pub const SHELL_MAX_STRAIN: KernelId = KernelId(247);
    pub const ROD_STRAIN_VALUE: KernelId = KernelId(248);
    pub const SHELL_STRAIN_TOI_GATED: KernelId = KernelId(250);
    pub const ROD_STRAIN_TOI_GATED: KernelId = KernelId(251);
    pub const FACE_CONVERT_FORCE: KernelId = KernelId(255);
    pub const FACE_CONVERT_HESSIAN: KernelId = KernelId(256);
    pub const FACE_DAMPING: KernelId = KernelId(257);
    pub const FACE_DEFORMATION_GRADIENT: KernelId = KernelId(258);
    pub const SHELL_STRETCH_TERMS: KernelId = KernelId(259);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const FACE_ATOMIC_EMBED_HESSIAN_SLOTS: KernelId = KernelId(260);
    pub const FACE_ATOMIC_EMBED_FORCE: KernelId = KernelId(261);
    pub const HINGE_DAMPING: KernelId = KernelId(264);
    pub const HINGE_ATOMIC_EMBED_FORCE: KernelId = KernelId(265);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const HINGE_ATOMIC_EMBED_HESSIAN_SLOTS: KernelId = KernelId(268);
    pub const ROD_BEND_DAMPING: KernelId = KernelId(269);
    pub const ROD_DAMPING: KernelId = KernelId(270);
    pub const ROD_ATOMIC_EMBED_FORCE: KernelId = KernelId(271);
    pub const COLLISION_WINDOW_VERTEX: KernelId = KernelId(252);
    pub const COLLISION_WINDOW_FACE: KernelId = KernelId(253);
    pub const COLLISION_WINDOW_EDGE: KernelId = KernelId(254);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const ROD_ATOMIC_EMBED_HESSIAN_SLOTS: KernelId = KernelId(275);
    pub const STITCH_ATOMIC_EMBED_FORCE: KernelId = KernelId(276);
    pub const SVD3X2: KernelId = KernelId(277);
    pub const SVD3X2_SHIFTED: KernelId = KernelId(278);
    pub const SHELL_STRAIN_RESTORE_SIGMA: KernelId = KernelId(279);
    pub const SVD3X3_RV: KernelId = KernelId(280);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const SVD3X3: KernelId = KernelId(281);
    pub const TET_CONVERT_FORCE: KernelId = KernelId(282);
    pub const TET_CONVERT_HESSIAN: KernelId = KernelId(283);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const TET_SHAPE_GRADIENTS: KernelId = KernelId(284);
    pub const TET_DEFORMATION_GRADIENT: KernelId = KernelId(285);
    pub const TET_DAMPING: KernelId = KernelId(286);
    /// Declared by the neutral tree and NOT dispatched by this driver.
    ///
    /// It carries a row because a library built from this tree carries one:
    /// `AbiDevice::open` refuses a pair of tables of different lengths, and a
    /// kernel id is a table INDEX, so every entry the tree declares must sit at
    /// the same position on both sides whether or not this driver reaches it.
    ///
    /// A row is not an invitation. `KernelArgs::KERNEL` ties an id to its
    /// record TYPE, so the only way to dispatch this is to construct its own
    /// arguments, which is a deliberate act rather than a slip.
    pub const VERTEX_NORMAL_FINALIZE: KernelId = KernelId(287);
    pub const VERTEX_ATOMIC_EMBED_FORCE: KernelId = KernelId(288);
    pub const VERTEX_FIX_INDEX_FROM_RECORDS: KernelId = KernelId(289);
    pub const VERTEX_DOF_REMOVAL_MASK: KernelId = KernelId(290);

    /// THE ELEMENT-WISE MULTI-LEVEL SCAN, the exclusive scan every pass that
    /// needs one dispatches. Three ids because
    /// the scan is three passes per level and a recursion between them, not
    /// because it is three kernels: `scan_block_total` sums each block,
    /// `driver::scan` recurses on the block sums, and `scan_block_apply` walks
    /// each block writing its exclusive prefix from that block's own base.
    /// `scan_zero` is the recursion's base and its extent is the array's own
    /// length.
    pub const SCAN_BLOCK_TOTAL: KernelId = KernelId(167);
    pub const SCAN_BLOCK_APPLY: KernelId = KernelId(168);
    pub const SCAN_ZERO: KernelId = KernelId(169);

    /// THE TREE'S PARENT LINKS, ROOT AND LEVELS, as four device passes rather
    /// than a host fold over a downloaded node array.
    pub const LBVH_SET_PARENT: KernelId = KernelId(126);
    pub const LBVH_FIND_ROOT: KernelId = KernelId(127);
    pub const LBVH_COUNT_LEVELS: KernelId = KernelId(128);
    pub const LBVH_SCATTER_LEVELS: KernelId = KernelId(129);

    /// THE SCENE'S CENTROID BOUNDS, reduced a level at a time rather than by
    /// one cooperative kernel; `reduce_bounds.kernel.cpp` states why the
    /// cooperative layer comes out.
    pub const BOUNDS_LEAF: KernelId = KernelId(160);
    pub const BOUNDS_MERGE: KernelId = KernelId(161);

    /// THE MINIMUM AND MAXIMUM OF A FLOAT ARRAY. One entry each: the leaf pass
    /// reads a float array and writes one float per block, which is the same
    /// shape the merge level needs, so the recursion re-dispatches it rather
    /// than needing a second kernel.
    pub const REDUCE_MIN_LEAF: KernelId = KernelId(162);
    pub const REDUCE_MAX_LEAF: KernelId = KernelId(163);

    /// THE SMALLEST WORD AND THE UNWRAPPED TOTAL OF AN UNSIGNED ARRAY. The
    /// minimum re-dispatches its own leaf as the merge, as the float pair does;
    /// the total's leaf reads single words and writes (low, high) pairs, so its
    /// merge over pairs is a second entry.
    pub const REDUCE_MIN_U32_LEAF: KernelId = KernelId(164);
    pub const REDUCE_SUM_U32_LEAF: KernelId = KernelId(165);
    pub const REDUCE_SUM_WIDE_MERGE: KernelId = KernelId(166);

    /// THE FIRST FLAGGED OVERLAP REPORT of one block, which
    /// `REDUCE_MIN_U32_LEAF` then reduces to the one slot the host reads.
    pub const OVERLAP_FIRST_FLAGGED_LEAF: KernelId = KernelId(15);

    /// THE PLASTIC COMMIT, which publishes a crept row into the rest shape the
    /// elastic kernels read. Three host scatters before this.
    pub const PLASTICITY_COMMIT_FACE: KernelId = KernelId(152);
    pub const PLASTICITY_COMMIT_TET: KernelId = KernelId(153);

    /// The number of rows in [`TABLE`], which is every entry point the
    /// neutral tree declares. A library built from this tree carries the
    /// same number in the same order, because both come from one walk.
    pub const CONTACT_EMBED_HESSIAN_BLOCKS: KernelId = KernelId(29);
    pub const CONTACT_EMBED_FORCE_TERMS: KernelId = KernelId(30);
    pub const COLLISION_POINT_FACE_M2C_TRAVERSE: KernelId = KernelId(25);
    pub const COLLISION_POINT_FACE_C2M_TRAVERSE: KernelId = KernelId(26);
    pub const COLLISION_EDGE_EDGE_TRAVERSE: KernelId = KernelId(27);
    pub const HINGE_ACTIVE_EMBED_FORCE: KernelId = KernelId(267);
    pub const HINGE_LIVE_EMBED_FORCE: KernelId = KernelId(266);
    pub const FACE_ACTIVE_EMBED_FORCE: KernelId = KernelId(263);
    pub const FACE_LIVE_EMBED_FORCE: KernelId = KernelId(262);
    pub const ROD_ACTIVE_EMBED_FORCE: KernelId = KernelId(273);
    pub const ROD_LIVE_EMBED_FORCE: KernelId = KernelId(272);
    pub const ROD_PACKED_EMBED_FORCE: KernelId = KernelId(274);
    pub const TET_SPECTRAL_CONVERT_HESSIAN: KernelId = KernelId(116);
    pub const COUNT: usize = 291;
}

// ===========================================================================
// The records.
//
// Every buffer field is a `HostRef`, which is the migration debt this stage
// leaves: see that type's comment. A `HostRef` is a `Handle` wide, so replacing
// one moves no other field.
// ===========================================================================

// ---------------------------------------------------------------------------
// THE GENERATED RECORDS.
//
// `VecCopyArgs`, `VecAddScaledArgs`, `VecCombineArgs` and
// `VecCombineIndirectArgs` are NOT written here. They are rendered by
// `ppf-cts-compute/seam/kernelgen.py --target rust --emit entry` from the
// `[[seam::args]] [[seam::entry]]` declarations beside their bodies in
// `src/kernels/primitives/vec_ops.kernel.cpp`, and the SAME declaration renders the
// `ppf_<stem>_entry` entry point `entrypoints/entries.cpp` compiles. That is
// what makes a record and its entry point one declaration instead of two that
// can disagree, which is what everything above this line still is.
//
// The artifact carries its own layout assertions, written against the same
// literal offsets the C++ renderings assert, so the four languages cannot
// drift. It names `HostRef`, `KernelId`, `KernelArgs` and the `id` module from
// this module's scope and declares none of them.
include!(concat!(env!("OUT_DIR"), "/kernelgen/primitives/vec_ops.entry.rs"));
include!(concat!(env!("OUT_DIR"), "/kernelgen/primitives/radix.entry.rs"));
include!(concat!(env!("OUT_DIR"), "/kernelgen/primitives/scan_levels.entry.rs"));
include!(concat!(env!("OUT_DIR"), "/kernelgen/primitives/reduce_bounds.entry.rs"));
include!(concat!(env!("OUT_DIR"), "/kernelgen/primitives/reduce_scalar.entry.rs"));

// The block-diagonal apply `z = P^-1 r`, rendered from the declaration beside
// `mat3_mul` in `src/kernels/solver/spmv.kernel.cpp`. It is the same body the
// operator's own row apply calls, dispatched one 3x3 block per row: the row's
// geometry is the declaration's two strides, nine floats for the block against
// three for the vector.
//
// Beside it the whole Newton operator, `result = (A + B + C) x` accumulated in
// one fp32 running sum, in the two forms `the_operator_record_names_every_
// buffer_it_carries` pins: with a dynamic contact matrix and without.
include!(concat!(env!("OUT_DIR"), "/kernelgen/solver/spmv.entry.rs"));

// The rescaled Newton step applied to the positions, rendered from the
// declaration beside `position_step`. `eval_x` is read and written at the same
// element, which is the in-place update the range shim spelled by hand.
include!(concat!(env!("OUT_DIR"), "/kernelgen/main/position_step.entry.rs"));

// The absolute-position gather, rendered from the declaration beside its body.
// One slot per element rather than a fixed run of them, and the bound each slot
// is checked against carried in the record.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/main/override_seed.entry.rs"
));

// THE MODULES WHOSE ENTRY POINTS THIS DRIVER DOES NOT DISPATCH. They are
// included for their RECORDS and their names, because `TABLE` carries a row
// for every entry the neutral tree declares: a library built from this tree
// carries one too, and a kernel id is a table INDEX, so the two must agree
// position by position or a dispatch runs the wrong kernel with the right
// bytes.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/contact/analytic_contact.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/contact/contact_assembly.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/friction.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/push.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/lbvh/bitonic.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/main/dump_linsys.entry.rs"
));

// The Schwarz preconditioner's domain construction.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/schwarz/schwarz.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/utility/face_hessian_scatter.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/utility/vertex_normal.entry.rs"
));

// The arity-1 force scatter, rendered from the declaration beside
// `vertex_atomic_embed_force`. Its `Scatter::Atomic` row below is what
// keeps the range one ascending pass; a generated entry point covers whatever
// range it is handed and cannot say otherwise.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/utility/vertex_scatter.entry.rs"
));

// The arity-2, arity-3, arity-4 and arity-6 force scatters, rendered from the
// declarations beside their bodies. Each is the arity-1 form above with a wider
// index vector and a wider gradient: two element gathers, a base destination and
// the guard count.
//
// The rod and hinge artifacts carry a SECOND record each, the Hessian scatter
// into precomputed fixed-CSR value slots, because a rendering covers a whole
// source file and those two declarations sit beside these. This driver builds no
// slot table (`FixedCsr::push_blocks` searches the row and reports a block the
// pattern has no slot for), so neither is dispatched here and neither has a
// table row; each names an id past `id::COUNT`, where a dispatch would fail by
// name.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/utility/collision_window.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/utility/rod_scatter.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/utility/face_scatter.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/utility/hinge_scatter.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/utility/stitch_scatter.entry.rs"
));

// The per-vertex magnitude of the search direction, rendered from the
// declaration beside `dx_magnitude`. `direction` reaches the body as a
// BASE pointer, because the body addresses its own triple at `3 * vert`.
include!(concat!(env!("OUT_DIR"), "/kernelgen/main/dx_norm.entry.rs"));

// The accept lerp, rendered from the declaration beside
// `position_accept`. `proposed` is read and written at the same element,
// which is the in-place update the range shim spelled by hand.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/main/position_accept.entry.rs"
));

// THE ELASTIC PIPELINE'S PER-ELEMENT STAGES, rendered from the declarations
// beside their bodies. Each record below is one declaration, not two: the same
// text produces the `ppf_<stem>_entry` shim `entrypoints/entries.cpp` compiles.
//
// The two SVDs write their three factors through references rather than
// returning them, so their outputs are element GATHERS and the records carry
// no scatter. The spectral and converter pairs return one element each.
include!(concat!(env!("OUT_DIR"), "/kernelgen/utility/svd3x2.entry.rs"));
include!(concat!(env!("OUT_DIR"), "/kernelgen/utility/svd3x3.entry.rs"));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/eigenanalysis/face_eigenanalysis.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/eigenanalysis/tet_eigenanalysis.entry.rs"
));
include!(concat!(env!("OUT_DIR"), "/kernelgen/utility/face_convert.entry.rs"));
include!(concat!(env!("OUT_DIR"), "/kernelgen/utility/tet_convert.entry.rs"));
// The shell face's deformation gradient and, from the declaration beside
// `shell_stretch_terms` in the same file, the stretch indicator's two
// per-face selections: the larger of that gradient's singular values and the
// smaller of the face's authored shrink factors, whose product is the stretch
// the material asks for.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/utility/face_deformation.entry.rs"
));

// The rod segment's Hookean stretch gradient and Hessian: an indirect gather
// over the edge's two vertex slots, the rest length and the force weight as
// element gathers, and the two outputs as non-const gathers. It ASSIGNS both,
// so the damping record below carries the accumulating half and the two
// dispatches are ordered.
include!(concat!(env!("OUT_DIR"), "/kernelgen/energy/rod_force.entry.rs"));

// THE TET ELASTIC LAYER, WHOLE. One dispatch per tet carries the deformation
// gradient, its factorization, the material table, the spectral force, the
// fused 12x12 spectral Hessian, the Rayleigh damping block and both scatters,
// which is what `embed_tet_force_hessian` does inside one CUDA thread. The
// element's four vertex slots arrive twice: once as the index list the entry
// reads the two position buffers through and bounds-checks, and once as a
// gathered quadruple, because the force rows and the CSR blocks the BODY
// writes are addressed by those same four numbers and an index list is not
// forwarded. Both fields carry one handle.
include!(concat!(env!("OUT_DIR"), "/kernelgen/energy/tet_force.entry.rs"));

// THE SHELL MEMBRANE LAYER, WHOLE, on the same terms as the solid one above.
// One dispatch per face carries the deformation gradient, the SVD, the
// material table, the spectral force, the PSD-projected 6x6 Hessian, the
// material-frame conversion, the Rayleigh damping block, the per-face
// pressure term and both scatters, which is what `embed_face_force_hessian`
// does inside one CUDA thread. The element's three vertex slots arrive twice:
// once as the index list the entry reads the two position buffers through and
// bounds-checks, and once as a gathered triple, because the force rows and the
// CSR blocks the BODY writes are addressed by those same three numbers and an
// index list is not forwarded. Both fields carry one handle.
include!(concat!(env!("OUT_DIR"), "/kernelgen/energy/face_force.entry.rs"));

// The dynamic matrix's three index-building passes.
//
// The ROW OFFSETS are one pass over one element with the thread index forwarded
// as the pass's own guard, because that body carries a running total across the
// whole row range: it is a serial exclusive scan, and the count of rows travels
// in `row_count_size` while the counts themselves travel in the buffer.
//
// The TRANSPOSE INDEX the symmetric matvec reads to reach the lower triangle is
// TWO passes of ONE ROW PER THREAD: a COUNT that adds into the column's
// arrival counter and a SCATTER that
// takes each arrival the next place in its column's run out of a claim counter.
// Both reach a shared slot through an index the row STORES rather than through
// the thread index, so both carry a scatter that is not `Disjoint` and neither
// can declare a device-side bound on the slots it touches.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/csrmat/dynamic_csr.entry.rs"
));

// The three broad-phase leaf boxes. Every buffer is a base pointer, the thread
// index is forwarded, and the box is the scatter.
include!(concat!(env!("OUT_DIR"), "/kernelgen/contact/aabb.entry.rs"));

// The cross-stitch's force and Hessian. Three gathers through one six-slot
// index list, the weights as a strided run, one non-const element gather out
// and the Hessian as a base pointer the body indexes itself.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/stitch.entry.rs"
));

// The Dirichlet elimination's second pass. Three element gathers in, one
// non-const element gather out, the force as a base pointer the body addresses
// at `3 * row`, and the thread index forwarded beside it.
include!(concat!(env!("OUT_DIR"), "/kernelgen/main/dirichlet.entry.rs"));

// The rod strain limiter's force, Hessian, strain and verdict. Two positions
// through the edge's own slot pair, two element gathers in, three non-const
// gathers out and one element scatter. The barrier kind rides as an `unsigned`
// because a record field is a scalar and `Barrier` is this tree's own
// enumeration; the cast is in the neutral body, so no backend spells it.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/strainlimiting/rod_strain.entry.rs"
));

// The stretch indicator's rod half. An indirect gather over the edge's two
// vertex slots, the initial length as an element gather, one element scatter.
// A DIAGNOSTIC that is still a kernel body: the maximum over the mesh is the
// `max_sigma` channel a reader compares against the strain limit, and that
// comparison is only worth anything if two backends report one number.
include!(concat!(env!("OUT_DIR"), "/kernelgen/main/stretch.entry.rs"));

// Rayleigh stiffness damping, one record per element arity: an indirect gather
// over the element's own vertex slots on TWO position buffers at once, the
// Newton iterate and the start of the step. One index list serves both, and the
// bound each slot is checked against is carried in the record. The two bending
// members carry a third gather, the LAGGED start-of-step Hessian.
//
// ITS GRADIENT AND HESSIAN ARE NON-CONST GATHERS. The body adds its force into
// the one and scales the other in place, so both carry the elastic assembly's
// value in as well as the damped value out, which is what a gather with no
// `const` and no scatter expresses.
include!(concat!(env!("OUT_DIR"), "/kernelgen/utility/tet_damping.entry.rs"));
include!(concat!(env!("OUT_DIR"), "/kernelgen/utility/face_damping.entry.rs"));
include!(concat!(env!("OUT_DIR"), "/kernelgen/utility/rod_damping.entry.rs"));
include!(concat!(env!("OUT_DIR"), "/kernelgen/utility/hinge_damping.entry.rs"));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/utility/rod_bend_damping.entry.rs"
));

// The material diff table, one record per element arity, rendered from the
// declarations beside `tet_material_diff_table` and its face sibling. The
// table's two halves are strides of one and the VERDICT is the scatter, because
// a scatter carries the body's return value and only one of the three outputs
// can be it. Its buffer is `unsigned` rather than a byte: a generated record
// addresses 4-byte pointees, which is what keeps a record free of the padding a
// later field could hide in.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/material_diff_table.entry.rs"
));

// The BaraffWitkin membrane's staged pass, rendered from the declaration
// beside `face_baraffwitkin`. The two destinations are strides of one
// rather than scatters: a scatter carries the body's RETURN value, and this
// body has two outputs and a case, a face naming another model, where it
// writes neither.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/baraffwitkin.entry.rs"
));

// The two bending stiffnesses and the plastic creep rate, rendered from the
// declarations beside their bodies. Each is per-element floats in and one
// float out, so each record is gathers plus one scatter.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/shell_bend_stiffness.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/rod_bend_stiffness.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/plasticity/plasticity.entry.rs"
));

// The two per-element strain readings, the largest shell singular value and the
// rod's length ratio, and beside them the two line-search times of impact. Each
// reading is element gathers into one element scatter, and what dispatches
// those two is the acceptance rig, which measures the shared body rather than a
// second implementation of it. Each time of impact reads its element's
// positions through an index list and gates on a non-positive limit inside its
// body, and both are on the step's own path.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/strainlimiting/strain_toi.entry.rs"
));

// The strain limiter's barrier derivative pair, at the AUTHORED limit. The
// no-limit case is decided in the body rather than by a launcher, so the branch
// a face with no `strainlimit` takes is written once for the three compilers;
// `strainlimiting/shell_strain.kernel.cpp` states why the comparison is
// negated.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/strainlimiting/shell_strain.entry.rs"
));

// The per-row terms of a dot product and of the round-off bound it is judged
// against, rendered from the declaration beside `pcg_dot_terms`. Both
// vectors stay BASE pointers because the body addresses its own triple at
// `3 * row`, and the two destinations are non-const gathers because a scatter
// carries one return value and this pass produces two.
//
// Beside them the step length and the search-direction coefficient, each
// dispatched over ONE element. Both are kernels rather than host helpers
// because the step length carries a VERDICT: CUDA evaluates the same
// arithmetic per thread inside `cg_fused_update_kernel`, and classifying the
// curvature on the host would be a second rule for when a solve aborts.
include!(concat!(env!("OUT_DIR"), "/kernelgen/solver/pcg.entry.rs"));

// One row of the block-Jacobi preconditioner, and the diagnostic lane its
// positive-definiteness verdict leaves through.
include!(concat!(env!("OUT_DIR"), "/kernelgen/solver/block_jacobi.entry.rs"));

// Start-of-step velocity, squared speed and distance from the origin, rendered
// from the declaration beside `velocity_terms`.
//
// THREE DESTINATIONS, SO NONE OF THEM IS A SCATTER. A scatter carries the
// body's one RETURN value, and this body writes three slots on every path, so
// each is a `[[seam::stride(1)]]` pointer to this vertex's own slot. The two
// positions and the `VertexProp` are element gathers.
include!(concat!(env!("OUT_DIR"), "/kernelgen/main/velocity.entry.rs"));

// The `fix-xz` horizontal drag's position half, rendered from the declaration
// beside `fix_xz_drag`. `eval_x` is read and written at the same element,
// which is the in-place update the range shim spelled by hand. The DOF mask is
// a third element gather rather than a launcher predicate: a generated entry
// always writes what the body returns, and the body returns a skipped vertex's
// own position.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/main/fix_xz_drag.entry.rs"
));

// The implicit target every vertex is solved toward, rendered from the
// declaration beside `compute_target`. The pin array reaches the body as a
// BASE pointer, because the pin it wants is at `fix_index - 1`; gravity is
// three record scalars rather than a sixth handle, because it is one
// scene-wide vector every thread reads identically and a uniform routed
// through a device allocation would cost a bounds check to deliver a constant.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/main/target.entry.rs"
));

// The external force field: grids and the compiled script, evaluated per
// vertex at the step's starting position. Its output is the per-vertex buffer
// `compute_target_seed` adds beside gravity and, for an air-velocity grid, the
// flow `momentum_embed` adds to the scene wind.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/external_field.entry.rs"
));

// The LBVH's Morton codes, rendered from the declaration beside
// `lbvh_morton_from_bounds`. Three element gathers of the SoA centroids
// into one element scatter, with the scene box as six record scalars: a
// three-float member would be a `float3` on Metal, 16 bytes against 12, and a
// record's layout is the one thing every backend must agree on.
//
// The node-depth walk renders into this same artifact, from the declaration
// beside `lbvh_node_depth`. Its `parent` array stays a BASE pointer,
// because the body climbs the links from a node it computes itself rather than
// reading a fixed run, and the thread index is forwarded to it instead of
// being spent on a gather.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/lbvh/lbvh.entry.rs"
));

// The shell hinge's exact bending force and PSD-projected Hessian, rendered
// from the declaration beside `shell_bend_force_hessian_checked`. Four
// positions read through the hinge's own index list, the rest angle gathered,
// the force and Hessian written back through non-const gathers, and a
// degenerate hinge raised through the diagnostic lane.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/shell_bend.entry.rs"
));

// The rod's turning angle and its bending pair, rendered from the two
// declarations beside their bodies. Three nodes read through the site's own
// index list, the angle scattered as the body's return value, and the force and
// Hessian written back through non-const gathers because a scatter carries one
// return value and these are two outputs.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/rod_bend.entry.rs"
));

// The kinematic pin's rewind, rendered from the declaration beside
// `rewind_fix`. One element gather of the pin's own record and no scatter:
// the body writes through the gathered element, and the gate on `kinematic`
// sits in the body rather than in a launcher.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/main/rewind_fix.entry.rs"
));

// The Newton iterate's seed, rendered from the declaration beside
// `dx_seed`. Four element gathers and no scatter: the gate on whether the
// row was removed is inside the body, so only a prescribed row is written and a
// free one is left as the driver cleared it.
include!(concat!(env!("OUT_DIR"), "/kernelgen/main/dx_seed.entry.rs"));

// The two per-vertex analytic-collider passes, rendered from the declarations
// beside their bodies in `src/kernels/contact/vertex_constraint.kernel.cpp`.
//
// A vertex is tested against EVERY analytic collider, so both records carry the
// sphere and floor arrays as base pointers with their counts beside them rather
// than a candidate pair list. Both declare a `[[seam::diag]]` lane: the
// non-negative gap they assert is the penetration-free guarantee rather than
// instrumentation.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/contact/vertex_constraint.entry.rs"
));

// The broad phase. `out` is a BASE POINTER because a thread writes a RANGE of
// it, the `capacity` slots its own query owns; what the body RETURNS is the
// count that query wanted, and that is the one `[[seam::scatter]]`. The
// `[[seam::diag]]` lane is the traversal's own: a stack overflow abandons
// subtrees, so its report is a fault rather than a statistic.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/contact/aabb_traversal.entry.rs"
));

// The per-face inflation pressure. Two accumulators and no scatter: a thread
// ADDS into a fixed run of each, which is the `[[seam::stride(N)]]` shape, and
// the body returns nothing for a scatter to carry.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/face_pressure.entry.rs"
));

// One torque group's frame. The group is the element and its result is the
// group's own slot; every other array is flat and bound, because what this
// kernel reads out of them is decided by the group's own fields rather than by
// the thread index. It carries the diagnostic channel for the three
// indirections that follow from that.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/torque.entry.rs"
));

// The SAND grain's three rows.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/sand_rigid.entry.rs"
));

// The PDRD rigid body's eighteen rows.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/pdrd_rigid.entry.rs"
));

// The body-DOF projector.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/energy/model/pdrd_lock_projector.entry.rs"
));

// THE AGGREGATE LOCK'S ELEVEN ROWS, which are the entries `super::lock`
// dispatches.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/solver/translation_lock_check.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/solver/translation_lock_frames.entry.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/solver/translation_lock_rows.entry.rs"
));

// ---------------------------------------------------------------------------
// THE ELEMENT ENERGIES. `super::assemble` owns the walk over them; these are
// the stages it dispatches. Every one of them writes element `i` and nothing
// else, so its row below is `Scatter::Disjoint`, except the force scatters,
// which reach `compute::atomic_add` and are declared `Scatter::Atomic`.
// ---------------------------------------------------------------------------

// One vertex's momentum row, rendered from the declaration beside its body in
// `src/kernels/main/momentum.kernel.cpp`.
//
// THE `ParamSet` IS NOT IN THE RECORD, the eight fields the body reads are, and
// `wind` arrives as its three components: a generated record holds 4-byte
// scalars and 16-byte handles, and `ParamSet::time` is an `f64` that MSL has no
// type for.
include!(concat!(env!("OUT_DIR"), "/kernelgen/main/momentum.entry.rs"));

// ---------------------------------------------------------------------------
// THE BROAD PHASE'S CONSTRUCTION PASSES.
//
// The tree is built once per step and its boxes are refreshed against every
// Newton iterate, so these stand in front of contact, the CCD filter and the
// intersection gate. What each of them writes is indexed by its own thread,
// which is why every row below is `Scatter::Disjoint`; the one pass in this
// subsystem that is NOT is the traversal, whose pairs take slots claimed as
// they are found, and `super::lbvh::walk` states why it is still a direct call.
//
// A LEAF'S PARAMETER ARRAY IS NOT INDEXED BY THE LEAF. The face and edge leaf
// passes reach theirs through `prop[primitive].param_index`, and that array is
// deduplicated across objects with identical materials, so its length is
// unrelated to the primitive count. The VERTEX pass carries no index array at
// all: leaf `i` holds primitive `nodes[2 * i] - 1`, which IS the vertex.
//
// THE INTERNAL-NODE MERGE IS DISPATCHED PER LEVEL, and the sequence of levels is
// the ordering that matters. Within one level the nodes are distinct and their
// children sit strictly deeper, which is what makes that row
// `Scatter::Disjoint` even though its box is indexed by the node rather than by
// the thread.
// ---------------------------------------------------------------------------

// The detect-once contact pair cache's record step, in the two shapes one body
// is reached in. `count` runs past `capacity` on purpose, counting every pair
// whether or not it fit, so the requirement is known even on the step that
// could not meet it; `overflow` is the flag stage 1 reads to decide the replay
// must be abandoned and the BVH re-walked.
//
// THE SLOT IS CLAIMED FROM `count` INSIDE THE BODY rather than chosen by the
// thread index, which is what makes the row `Scatter::Claim` and the pass one
// serial ascending dispatch: `compute::atomic_add` on the host seam is a plain
// read, add and write back, so a cut range is a data race and not a different
// fold order.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/contact/pair_cache.entry.rs"
));

// One batch of 3x3 blocks folded into the fixed-pattern Hessian, and the
// per-block verdict naming whether the pattern carried a slot for it, rendered
// from the declaration beside `fixed_csr_atomic_push`.
//
// THE VERDICT IS A SCATTER, WHICH IS WHY IT CANNOT BE DROPPED. The body returns
// false for a block whose `(row, column)` is outside the build-time sparsity,
// and every CUDA caller of the same body ignores that verdict, which is how the
// rod-bend `(j, k)` stencil bug shipped an indefinite matrix.
// `[[seam::scatter]]` carries the body's return value to this block's slot, so
// the entry point stores it by construction rather than by a launcher
// remembering to.
//
// `stored` holds one `u32` per block rather than one byte: a record holds only
// 4-byte scalars and 16-byte handles, so it has no padding and its size is the
// sum of its fields, and a narrower slot would move no size.
//
// Beside it the block-Jacobi preconditioner's diagonal, `A(i,i) + B(i,i) +
// C[i]`, in the same two forms as the operator it preconditions.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/csrmat/fixed_csr.entry.rs"
));

// ---------------------------------------------------------------------------
// THE CONTACT SUBSYSTEM'S PER-PRIMITIVE PASSES.
//
// The Morton sort keys, the query boxes the three trees are walked with, and
// the collision-window mask applied to the leaves. Every one writes the element
// its own thread index names, so every row is `Scatter::Disjoint`; the four
// narrow-phase visitors below are the passes that are not.
//
// THE BOXES COME IN THREE KINDS AND THE DIFFERENCE IS WHAT EACH IS WALKED FOR.
// An ASSEMBLY box is the primitive at the Newton iterate inflated by the same
// margin its leaf carries. A LINE SEARCH box is SWEPT, bounding the primitive
// over the whole candidate step rather than at either end, because a pair that
// ends the step apart may have crossed inside it. An INTERSECTION SCAN box
// carries no margin at all, which is the box the final penetration gate walks.
//
// THE COLLISION WINDOW IS OPTIONAL ON A QUERY AND REQUIRED ON THE LEAF FLAG.
// A query's mask naming NOTHING is how a scene that authored no window arrives,
// which is a different thing from an all-true mask and leaves every box taking
// part; the pass that clears the flags has nothing to apply without one and is
// not dispatched at all. That mask is indexed by the PRIMITIVE a leaf holds
// rather than by the thread.
// ---------------------------------------------------------------------------

// The four narrow-phase visitors, rendered from the declarations beside their
// bodies in `src/kernels/contact/contact_narrow.kernel.cpp`.
//
// ONE DECLARATION PER PAIR KIND, not one record behind four ids. The four
// differ in which primitives a candidate pair names, so each names only the
// arrays its own kind reads: point-point takes no face property and edge-edge
// takes no vertex parameter. A record shared by all four would carry every
// array for every kind and leave a reader unable to tell which pass reads
// which.
//
// EVERY BUFFER IS A BASE POINTER, because a visitor addresses `pair` at
// `2 * k`, the outputs at `4 * k`, `12 * k` and `144 * k`, and the mesh and
// property arrays at indices the pair NAMES. The outputs are indexed by the
// candidate pair, one slot each, which is what makes every row
// `Scatter::Disjoint`: the fold into the force vector and the two matrices
// happens afterwards, in the driver's own serial scatter.
//
// THE `ParamSet` IS NOT IN THESE RECORDS, its three read fields are. A record
// holds 4-byte scalars, so the barrier selector, the friction mode and the
// friction floor cost three fields against one buffer and remove the record's
// dependency on that struct's layout.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/contact/contact_narrow.entry.rs"
));

// The three static-collision-mesh visitors, rendered from the declarations
// beside their bodies in `src/kernels/contact/collision_narrow.kernel.cpp`.
//
// ONE DECLARATION PER PASS. Each names only the arrays its own pass reads,
// which is why the collider side appears in three shapes: its faces for M2C,
// its vertices for C2M, its edges for edge-edge. The static side has ONE POSE,
// so it has no start and no end.
//
// THE DYNAMIC `VertexProp` IS IN ALL THREE and the static one only in C2M,
// because the mass and the elastic snapshot are gathered over the DYNAMIC
// vertices alone. The collider contributes no row: it is a rest-pose
// contact-only pool outside the solved namespace, so it pushes and is never
// pushed, and that asymmetry is its definition rather than an omission.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/contact/collision_narrow.entry.rs"
));

// THE CCD LINE SEARCH, one dispatch per (query kind, tree) pair. Each carries
// its own tree and its own query index space, so no two of the six records are
// the same shape; what they share is the pair of per-primitive OUTPUT arrays,
// `out_toi` and `out_overlap`, which every sweep min-folds into at the slot its
// own query owns.
//
// `out_toi` AND `out_overlap` ARE BASE POINTERS RATHER THAN SCATTERS, twice
// over: the slot is the Morton-remapped primitive rather than the thread index,
// and the write is a fold into a value the caller seeded rather than a store.
//
// THE `ParamSet` IS NOT IN THESE RECORDS EITHER, its two read fields are, which
// is what keeps `accd.hpp` reachable from a Metal shader that cannot read
// data.hpp in any position.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/contact/ccd_sweep.entry.rs"
));

// THE FINAL PENETRATION GATE'S FOUR FUSED WALKS.
//
// `query`, `flag` and `records` are BASE POINTERS and none of them is a
// scatter, which is the seam's rule read rather than worked around. `query` is
// read at this thread's own index and copied into thread space by the body;
// `flag` is written ONLY on a hit, so an unconditional per-thread store would
// clear the verdict another scan over the same array had already reached; and
// `records` is claimed at an index no thread owns.
//
// THE RECORD ARRAY'S CAPACITY IS A FIELD RATHER THAN A CONSTANT, so no backend
// carries a copy of a size the caller decided. The counter counts PAST it on
// purpose: a run aborted on intersections wants the true demand, and a report
// that stops counting where it stops storing understates the problem.
include!(concat!(
    env!("OUT_DIR"),
    "/kernelgen/contact/intersect_geometry.entry.rs"
));

// ===========================================================================
// The table.
// ===========================================================================


const fn decl(
    id: KernelId,
    name: &'static str,
    scatter: Scatter,
    nanos_per_item: f64,
    args_bytes: usize,
    host_refs: &'static [u16],
    diag: bool,
) -> KernelDecl {
    KernelDecl {
        id,
        name,
        scatter,
        nanos_per_item,
        args_bytes: args_bytes as u16,
        host_refs,
        diag,
        generated: false,
    }
}

/// As [`decl`], for an entry point rendered by `ppf-cts-compute/seam/kernelgen.py`.
///
/// The one thing the flag changes is how the backend HANDS the arguments over:
/// a generated entry point takes the seam's own reference form, an (arena,
/// offset) handle resolved against a table of arena base addresses, where a
/// hand-written shim takes flat pointers. Nothing about the work differs, which
/// is why this is a separate constructor rather than a parameter on every row:
/// the rows that carry `true` are the count of how far the conversion has got.
const fn decl_generated(
    id: KernelId,
    name: &'static str,
    scatter: Scatter,
    nanos_per_item: f64,
    args_bytes: usize,
    host_refs: &'static [u16],
) -> KernelDecl {
    KernelDecl {
        id,
        name,
        scatter,
        nanos_per_item,
        args_bytes: args_bytes as u16,
        host_refs,
        diag: false,
        generated: true,
    }
}

/// As [`decl_generated`], for an entry that declares a `[[seam::diag]]` lane.
///
/// The lane is what lets a NEUTRAL body report a failed check: the three
/// targets bind three different things behind one alias, so the body takes the
/// handle by value and passes it to `DIAG_ASSERT4` without ever dereferencing
/// it. Setting `diag` here is what makes the host allocate a record per chunk
/// and merge them in ascending chunk order.
///
/// Two constructors rather than one with a flag, for the reason
/// `generated_thunk_diag!` is a second macro: an entry either declares the lane
/// or it does not, and the generator renders a different signature for each, so
/// a row that disagrees with its rendering is a name error at the `extern`
/// rather than a null handed to a body that will write through it.
const fn decl_generated_diag(
    id: KernelId,
    name: &'static str,
    scatter: Scatter,
    nanos_per_item: f64,
    args_bytes: usize,
    host_refs: &'static [u16],
) -> KernelDecl {
    KernelDecl {
        id,
        name,
        scatter,
        nanos_per_item,
        args_bytes: args_bytes as u16,
        host_refs,
        diag: true,
        generated: true,
    }
}

/// The declarations, indexed by [`KernelId`].
///
/// Costs estimate one dispatch unit: an element or an entire group, not a lane.
/// The row estimates include a vector update at 0.5 ns, the composed operator at
/// 6.0, its preconditioner at 8.0, and the block-Jacobi apply at 4.0. The three
/// Disjoint folded PCG entries scale their row estimate by the number of rows
/// a group owns. The
/// per-vertex step passes ran serially over the whole range before the seam and
/// so carried no figure; they are stated at the shape of the work they do, and
/// a wrong figure here costs scheduling and never an answer, because a
/// `Disjoint` range is cut at a fixed chunk width and every element is written
/// by exactly one thread.
pub static TABLE: [KernelDecl; id::COUNT] = [
    decl_generated(
        id::AABB_LEAF_FACE,
        AABB_LEAF_FACE_NAME,
        Scatter::Disjoint,
        30.0,
        size_of::<AabbLeafFaceArgs>(),
        AABB_LEAF_FACE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_LEAF_EDGE,
        AABB_LEAF_EDGE_NAME,
        Scatter::Disjoint,
        25.0,
        size_of::<AabbLeafEdgeArgs>(),
        AABB_LEAF_EDGE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_LEAF_VERTEX,
        AABB_LEAF_VERTEX_NAME,
        Scatter::Disjoint,
        20.0,
        size_of::<AabbLeafVertexArgs>(),
        AABB_LEAF_VERTEX_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_LEAF_ACTIVE,
        AABB_LEAF_ACTIVE_NAME,
        Scatter::Disjoint,
        3.0,
        size_of::<AabbLeafActiveArgs>(),
        AABB_LEAF_ACTIVE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_POINT_CONTACT_QUERY,
        AABB_POINT_CONTACT_QUERY_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<AabbPointContactQueryArgs>(),
        AABB_POINT_CONTACT_QUERY_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_POINT_CONTACT_QUERY_MASKED,
        AABB_POINT_CONTACT_QUERY_MASKED_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<AabbPointContactQueryMaskedArgs>(),
        AABB_POINT_CONTACT_QUERY_MASKED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_EDGE_CONTACT_QUERY,
        AABB_EDGE_CONTACT_QUERY_NAME,
        Scatter::Disjoint,
        12.0,
        size_of::<AabbEdgeContactQueryArgs>(),
        AABB_EDGE_CONTACT_QUERY_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_EDGE_CONTACT_QUERY_MASKED,
        AABB_EDGE_CONTACT_QUERY_MASKED_NAME,
        Scatter::Disjoint,
        12.0,
        size_of::<AabbEdgeContactQueryMaskedArgs>(),
        AABB_EDGE_CONTACT_QUERY_MASKED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_EDGE_SCAN_QUERY,
        AABB_EDGE_SCAN_QUERY_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<AabbEdgeScanQueryArgs>(),
        AABB_EDGE_SCAN_QUERY_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_EDGE_SCAN_QUERY_MASKED,
        AABB_EDGE_SCAN_QUERY_MASKED_NAME,
        Scatter::Disjoint,
        10.0,
        size_of::<AabbEdgeScanQueryMaskedArgs>(),
        AABB_EDGE_SCAN_QUERY_MASKED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_VERTEX_SCAN_QUERY,
        AABB_VERTEX_SCAN_QUERY_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<AabbVertexScanQueryArgs>(),
        AABB_VERTEX_SCAN_QUERY_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_VERTEX_SCAN_QUERY_MASKED,
        AABB_VERTEX_SCAN_QUERY_MASKED_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<AabbVertexScanQueryMaskedArgs>(),
        AABB_VERTEX_SCAN_QUERY_MASKED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::AABB_MERGE_LEVEL,
        AABB_MERGE_LEVEL_NAME,
        Scatter::Disjoint,
        15.0,
        size_of::<AabbMergeLevelArgs>(),
        AABB_MERGE_LEVEL_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::AABB_QUERY_PAIRS,
        AABB_QUERY_PAIRS_NAME,
        // A query writes only its OWN run of `out` and its own slot of `found`,
        // so no two threads name one word and the range may be cut anywhere.
        Scatter::Disjoint,
        400.0,
        size_of::<AabbQueryPairsArgs>(),
        AABB_QUERY_PAIRS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::COMBINE_FRICTION_VALUES,
        COMBINE_FRICTION_VALUES_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<CombineFrictionValuesArgs>(),
        COMBINE_FRICTION_VALUES_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::OVERLAP_FIRST_FLAGGED_LEAF,
        OVERLAP_FIRST_FLAGGED_LEAF_NAME,
        // DISJOINT: a thread reads one block of reports, which no pass writes
        // while it runs, and writes one word of its own.
        Scatter::Disjoint,
        512.0,
        size_of::<OverlapFirstFlaggedLeafArgs>(),
        OVERLAP_FIRST_FLAGGED_LEAF_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CCD_POINT_FACE,
        CCD_POINT_FACE_NAME,
        Scatter::Disjoint,
        3000.0,
        size_of::<CcdPointFaceArgs>(),
        CCD_POINT_FACE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CCD_POINT_POINT,
        CCD_POINT_POINT_NAME,
        Scatter::Disjoint,
        3000.0,
        size_of::<CcdPointPointArgs>(),
        CCD_POINT_POINT_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CCD_EDGE_EDGE,
        CCD_EDGE_EDGE_NAME,
        Scatter::Disjoint,
        3000.0,
        size_of::<CcdEdgeEdgeArgs>(),
        CCD_EDGE_EDGE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CCD_COLLISION_POINT_FACE_M2C,
        CCD_COLLISION_POINT_FACE_M2C_NAME,
        Scatter::Disjoint,
        3000.0,
        size_of::<CcdCollisionPointFaceM2cArgs>(),
        CCD_COLLISION_POINT_FACE_M2C_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CCD_COLLISION_POINT_FACE_C2M,
        CCD_COLLISION_POINT_FACE_C2M_NAME,
        Scatter::Disjoint,
        3000.0,
        size_of::<CcdCollisionPointFaceC2mArgs>(),
        CCD_COLLISION_POINT_FACE_C2M_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CCD_COLLISION_EDGE_EDGE,
        CCD_COLLISION_EDGE_EDGE_NAME,
        Scatter::Disjoint,
        3000.0,
        size_of::<CcdCollisionEdgeEdgeArgs>(),
        CCD_COLLISION_EDGE_EDGE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::COLLISION_POINT_FACE_M2C,
        COLLISION_POINT_FACE_M2C_NAME,
        Scatter::Disjoint,
        900.0,
        size_of::<CollisionPointFaceM2cArgs>(),
        COLLISION_POINT_FACE_M2C_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::COLLISION_POINT_FACE_C2M,
        COLLISION_POINT_FACE_C2M_NAME,
        Scatter::Disjoint,
        900.0,
        size_of::<CollisionPointFaceC2mArgs>(),
        COLLISION_POINT_FACE_C2M_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::COLLISION_EDGE_EDGE,
        COLLISION_EDGE_EDGE_NAME,
        Scatter::Disjoint,
        900.0,
        size_of::<CollisionEdgeEdgeArgs>(),
        COLLISION_EDGE_EDGE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::COLLISION_POINT_FACE_M2C_TRAVERSE,
        COLLISION_POINT_FACE_M2C_TRAVERSE_NAME,
        // ATOMIC for the same reason the self-contact traversal is: the embed
        // accumulates the per-vertex force and the fixed matrix's blocks, and
        // two queries touching one vertex overlap by construction.
        Scatter::Atomic,
        // A query walks the tree and embeds every hit, so its cost is the
        // traversal plus however many pairs it finds.
        2000.0,
        size_of::<CollisionPointFaceM2cTraverseArgs>(),
        COLLISION_POINT_FACE_M2C_TRAVERSE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::COLLISION_POINT_FACE_C2M_TRAVERSE,
        COLLISION_POINT_FACE_C2M_TRAVERSE_NAME,
        // ATOMIC for the same reason the self-contact traversal is: the embed
        // accumulates the per-vertex force and the fixed matrix's blocks, and
        // two queries touching one vertex overlap by construction.
        Scatter::Atomic,
        // A query walks the tree and embeds every hit, so its cost is the
        // traversal plus however many pairs it finds.
        2000.0,
        size_of::<CollisionPointFaceC2mTraverseArgs>(),
        COLLISION_POINT_FACE_C2M_TRAVERSE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::COLLISION_EDGE_EDGE_TRAVERSE,
        COLLISION_EDGE_EDGE_TRAVERSE_NAME,
        // ATOMIC for the same reason the self-contact traversal is: the embed
        // accumulates the per-vertex force and the fixed matrix's blocks, and
        // two queries touching one vertex overlap by construction.
        Scatter::Atomic,
        // A query walks the tree and embeds every hit, so its cost is the
        // traversal plus however many pairs it finds.
        2000.0,
        size_of::<CollisionEdgeEdgeTraverseArgs>(),
        COLLISION_EDGE_EDGE_TRAVERSE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::CONTACT_FIXED_SLOT,
        CONTACT_FIXED_SLOT_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<ContactFixedSlotArgs>(),
        CONTACT_FIXED_SLOT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::CONTACT_EMBED_HESSIAN_BLOCKS,
        CONTACT_EMBED_HESSIAN_BLOCKS_NAME,
        // CLAIM, because the slot each block writes is taken from an atomic
        // cursor rather than derived from the thread index. That is what lets
        // the host arm reproduce the sequential order the host loop had while a
        // GPU claims in arrival order.
        Scatter::Claim,
        120.0,
        size_of::<ContactEmbedHessianBlocksArgs>(),
        CONTACT_EMBED_HESSIAN_BLOCKS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::CONTACT_EMBED_FORCE_TERMS,
        CONTACT_EMBED_FORCE_TERMS_NAME,
        // ATOMIC, because the fold is a float atomic into the per-vertex force
        // and two contacts commonly share a vertex. The host arm must not cut
        // the range.
        Scatter::Atomic,
        60.0,
        size_of::<ContactEmbedForceTermsArgs>(),
        CONTACT_EMBED_FORCE_TERMS_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CONTACT_POINT_FACE_TRAVERSE,
        CONTACT_POINT_FACE_TRAVERSE_NAME,
        // ATOMIC for the same reason the pair form is: the embed accumulates
        // the per-vertex force and the fixed matrix's blocks, and two queries
        // touching one vertex overlap by construction.
        Scatter::Atomic,
        // A query walks the tree and embeds every hit, so its cost is the
        // traversal plus however many pairs it finds.
        2000.0,
        size_of::<ContactPointFaceTraverseArgs>(),
        CONTACT_POINT_FACE_TRAVERSE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CONTACT_POINT_FACE,
        CONTACT_POINT_FACE_NAME,
        // ATOMIC BECAUSE THE NARROW PHASE NOW DEPOSITS. It accumulates the
        // per-vertex force and the fixed matrix's blocks with
        // `compute::atomic_add`, and two contacts sharing a vertex overlap by
        // construction. `Disjoint` runs as concurrent rayon chunks on the host
        // backend, where the float seam atomic is a plain read, add and write
        // back, so the row would license a race that CUDA and Metal cannot
        // show. `check-atomic-scatter.py` is what caught it.
        Scatter::Atomic,
        900.0,
        size_of::<ContactPointFaceArgs>(),
        CONTACT_POINT_FACE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CONTACT_POINT_EDGE_TRAVERSE,
        CONTACT_POINT_EDGE_TRAVERSE_NAME,
        Scatter::Atomic,
        2000.0,
        size_of::<ContactPointEdgeTraverseArgs>(),
        CONTACT_POINT_EDGE_TRAVERSE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CONTACT_POINT_EDGE,
        CONTACT_POINT_EDGE_NAME,
        // ATOMIC BECAUSE THE NARROW PHASE NOW DEPOSITS. It accumulates the
        // per-vertex force and the fixed matrix's blocks with
        // `compute::atomic_add`, and two contacts sharing a vertex overlap by
        // construction. `Disjoint` runs as concurrent rayon chunks on the host
        // backend, where the float seam atomic is a plain read, add and write
        // back, so the row would license a race that CUDA and Metal cannot
        // show. `check-atomic-scatter.py` is what caught it.
        Scatter::Atomic,
        900.0,
        size_of::<ContactPointEdgeArgs>(),
        CONTACT_POINT_EDGE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CONTACT_POINT_POINT_TRAVERSE,
        CONTACT_POINT_POINT_TRAVERSE_NAME,
        Scatter::Atomic,
        2200.0,
        size_of::<ContactPointPointTraverseArgs>(),
        CONTACT_POINT_POINT_TRAVERSE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CONTACT_POINT_POINT,
        CONTACT_POINT_POINT_NAME,
        // ATOMIC BECAUSE THE NARROW PHASE NOW DEPOSITS. It accumulates the
        // per-vertex force and the fixed matrix's blocks with
        // `compute::atomic_add`, and two contacts sharing a vertex overlap by
        // construction. `Disjoint` runs as concurrent rayon chunks on the host
        // backend, where the float seam atomic is a plain read, add and write
        // back, so the row would license a race that CUDA and Metal cannot
        // show. `check-atomic-scatter.py` is what caught it.
        Scatter::Atomic,
        900.0,
        size_of::<ContactPointPointArgs>(),
        CONTACT_POINT_POINT_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CONTACT_EDGE_EDGE_TRAVERSE,
        CONTACT_EDGE_EDGE_TRAVERSE_NAME,
        Scatter::Atomic,
        2000.0,
        size_of::<ContactEdgeEdgeTraverseArgs>(),
        CONTACT_EDGE_EDGE_TRAVERSE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::CONTACT_EDGE_EDGE,
        CONTACT_EDGE_EDGE_NAME,
        // ATOMIC BECAUSE THE NARROW PHASE NOW DEPOSITS. It accumulates the
        // per-vertex force and the fixed matrix's blocks with
        // `compute::atomic_add`, and two contacts sharing a vertex overlap by
        // construction. `Disjoint` runs as concurrent rayon chunks on the host
        // backend, where the float seam atomic is a plain read, add and write
        // back, so the row would license a race that CUDA and Metal cannot
        // show. `check-atomic-scatter.py` is what caught it.
        Scatter::Atomic,
        900.0,
        size_of::<ContactEdgeEdgeArgs>(),
        CONTACT_EDGE_EDGE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::INTERSECT_SCAN_FACE_EDGE,
        INTERSECT_SCAN_FACE_EDGE_NAME,
        Scatter::Claim,
        2000.0,
        size_of::<IntersectScanFaceEdgeArgs>(),
        INTERSECT_SCAN_FACE_EDGE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::INTERSECT_SCAN_EDGE_EDGE,
        INTERSECT_SCAN_EDGE_EDGE_NAME,
        Scatter::Claim,
        2000.0,
        size_of::<IntersectScanEdgeEdgeArgs>(),
        INTERSECT_SCAN_EDGE_EDGE_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::INTERSECT_SCAN_POINT_POINT,
        INTERSECT_SCAN_POINT_POINT_NAME,
        Scatter::Claim,
        2000.0,
        size_of::<IntersectScanPointPointArgs>(),
        INTERSECT_SCAN_POINT_POINT_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::INTERSECT_SCAN_COLLISION_MESH,
        INTERSECT_SCAN_COLLISION_MESH_NAME,
        Scatter::Claim,
        2000.0,
        size_of::<IntersectScanCollisionMeshArgs>(),
        INTERSECT_SCAN_COLLISION_MESH_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PAIR_CACHE_RECORD,
        PAIR_CACHE_RECORD_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<PairCacheRecordArgs>(),
        PAIR_CACHE_RECORD_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PAIR_CACHE_RECORD_INTERLEAVED,
        PAIR_CACHE_RECORD_INTERLEAVED_NAME,
        Scatter::Claim,
        2.0,
        size_of::<PairCacheRecordInterleavedArgs>(),
        PAIR_CACHE_RECORD_INTERLEAVED_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::VERTEX_CONSTRAINT,
        VERTEX_CONSTRAINT_NAME,
        Scatter::Disjoint,
        120.0,
        size_of::<VertexConstraintArgs>(),
        VERTEX_CONSTRAINT_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::VERTEX_CONSTRAINT_SWEEP,
        VERTEX_CONSTRAINT_SWEEP_NAME,
        Scatter::Disjoint,
        60.0,
        size_of::<VertexConstraintSweepArgs>(),
        VERTEX_CONSTRAINT_SWEEP_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DYN_COUNT_TRANSPOSE_PASS,
        DYN_COUNT_TRANSPOSE_PASS_NAME,
        // THE COUNTS ARE SHARED SLOTS: every row of the matrix adds into the
        // counter of whatever column it names, so two rows naming one column
        // meet there. On the host seam an atomic add is a plain read, add and
        // write back, which is why this is `Atomic` and not `Disjoint`.
        Scatter::Atomic,
        20.0,
        size_of::<DynCountTransposePassArgs>(),
        DYN_COUNT_TRANSPOSE_PASS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DYN_SCATTER_TRANSPOSE_PASS,
        DYN_SCATTER_TRANSPOSE_PASS_NAME,
        // A NUMBERED SLOT OUT OF A SHARED COUNTER, which is `Claim`: each
        // arrival takes the next place in its column's run out of one counter
        // over the whole grid, so WHICH place an arrival gets is not a property
        // this preserves; that every arrival gets one place and no place is
        // handed out twice is.
        Scatter::Claim,
        30.0,
        size_of::<DynScatterTransposePassArgs>(),
        DYN_SCATTER_TRANSPOSE_PASS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DYN_ROW_BEGIN_PASS,
        DYN_ROW_BEGIN_PASS_NAME,
        // ROW r ORDERS ROW r's PATTERN AND WRITES ROW r's RESERVE, and nothing
        // else, so a backend is free to cut the range.
        Scatter::Disjoint,
        20.0,
        size_of::<DynRowBeginPassArgs>(),
        DYN_ROW_BEGIN_PASS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DYN_DRY_PUSH_PASS,
        DYN_DRY_PUSH_PASS_NAME,
        // THE RESERVE IS A SHARED SLOT: two contributions to one row meet in
        // that row's counter, and on the host seam an atomic add is a plain
        // read, add and write back, so a cut range would be a data race.
        Scatter::Atomic,
        10.0,
        size_of::<DynDryPushPassArgs>(),
        DYN_DRY_PUSH_PASS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DYN_ROW_SEED_PASS,
        DYN_ROW_SEED_PASS_NAME,
        Scatter::Disjoint,
        30.0,
        size_of::<DynRowSeedPassArgs>(),
        DYN_ROW_SEED_PASS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DYN_PUSH_PASS,
        DYN_PUSH_PASS_NAME,
        // A NUMBERED SLOT OUT OF A SHARED COUNTER, which is `Claim`: a
        // contribution whose column the row does not carry takes the next place
        // in that row's slab, out of one counter over the whole grid.
        Scatter::Claim,
        25.0,
        size_of::<DynPushPassArgs>(),
        DYN_PUSH_PASS_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::DYN_ROW_COMPACT_PASS,
        DYN_ROW_COMPACT_PASS_NAME,
        Scatter::Disjoint,
        60.0,
        size_of::<DynRowCompactPassArgs>(),
        DYN_ROW_COMPACT_PASS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DYN_ROW_EMIT_PASS,
        DYN_ROW_EMIT_PASS_NAME,
        Scatter::Disjoint,
        30.0,
        size_of::<DynRowEmitPassArgs>(),
        DYN_ROW_EMIT_PASS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FIXED_PUSH_ELEMENT_BLOCKS_AT,
        FIXED_PUSH_ELEMENT_BLOCKS_AT_NAME,
        // ATOMIC: two elements sharing a vertex push the same block, which is
        // the ordinary case for an assembled Hessian, so the host arm must not
        // cut the range into concurrent chunks over a float accumulation.
        Scatter::Atomic,
        // An arity-4 stencil offers sixteen blocks, each a lookup and nine
        // atomic adds.
        900.0,
        size_of::<FixedPushElementBlocksAtArgs>(),
        FIXED_PUSH_ELEMENT_BLOCKS_AT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FIXED_PUSH_ELEMENT_BLOCKS_GATED_AT,
        FIXED_PUSH_ELEMENT_BLOCKS_GATED_AT_NAME,
        // ATOMIC: two elements sharing a vertex push the same block, which is
        // the ordinary case for an assembled Hessian, so the host arm must not
        // cut the range into concurrent chunks over a float accumulation.
        Scatter::Atomic,
        // An arity-4 stencil offers sixteen blocks, each a lookup and nine
        // atomic adds.
        900.0,
        size_of::<FixedPushElementBlocksGatedAtArgs>(),
        FIXED_PUSH_ELEMENT_BLOCKS_GATED_AT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FIXED_PUSH_ELEMENT_BLOCKS_LIVE,
        FIXED_PUSH_ELEMENT_BLOCKS_LIVE_NAME,
        // ATOMIC for the reason the compacted form is: two elements sharing a
        // vertex push the same block, so the host arm must not cut the range
        // into concurrent chunks over a float accumulation.
        Scatter::Atomic,
        // An arity-4 stencil offers sixteen blocks, each a lookup and nine
        // atomic adds. The gate makes the AVERAGE cheaper and the worst case is
        // the compacted form's, which is what a per-item cost states.
        900.0,
        size_of::<FixedPushElementBlocksLiveArgs>(),
        FIXED_PUSH_ELEMENT_BLOCKS_LIVE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FIXED_PUSH_ELEMENT_BLOCKS_GATED,
        FIXED_PUSH_ELEMENT_BLOCKS_GATED_NAME,
        // ATOMIC for the reason the compacted form is: two elements sharing a
        // vertex push the same block, so the host arm must not cut the range
        // into concurrent chunks over a float accumulation.
        Scatter::Atomic,
        // An arity-4 stencil offers sixteen blocks, each a lookup and nine
        // atomic adds. The gate makes the AVERAGE cheaper and the worst case is
        // the compacted form's, which is what a per-item cost states.
        900.0,
        size_of::<FixedPushElementBlocksGatedArgs>(),
        FIXED_PUSH_ELEMENT_BLOCKS_GATED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FIXED_PUSH_ELEMENT_BLOCKS,
        FIXED_PUSH_ELEMENT_BLOCKS_NAME,
        // ATOMIC: two elements sharing a vertex push the same block, which is
        // the ordinary case for an assembled Hessian, so the host arm must not
        // cut the range into concurrent chunks over a float accumulation.
        Scatter::Atomic,
        // An arity-4 stencil offers sixteen blocks, each a lookup and nine
        // atomic adds.
        900.0,
        size_of::<FixedPushElementBlocksArgs>(),
        FIXED_PUSH_ELEMENT_BLOCKS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FIXED_CSR_ATOMIC_PUSH,
        FIXED_CSR_ATOMIC_PUSH_NAME,
        Scatter::Atomic,
        15.0,
        size_of::<FixedCsrAtomicPushArgs>(),
        FIXED_CSR_ATOMIC_PUSH_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PRECOND_DIAGONAL,
        PRECOND_DIAGONAL_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<PrecondDiagonalArgs>(),
        PRECOND_DIAGONAL_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PRECOND_DIAGONAL_DYNAMIC,
        PRECOND_DIAGONAL_DYNAMIC_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<PrecondDiagonalDynamicArgs>(),
        PRECOND_DIAGONAL_DYNAMIC_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_SPECTRAL_FORCE,
        FACE_SPECTRAL_FORCE_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<FaceSpectralForceArgs>(),
        FACE_SPECTRAL_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_SPECTRAL_HESSIAN,
        FACE_SPECTRAL_HESSIAN_NAME,
        Scatter::Disjoint,
        90.0,
        size_of::<FaceSpectralHessianArgs>(),
        FACE_SPECTRAL_HESSIAN_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::TET_SPECTRAL_FORCE,
        TET_SPECTRAL_FORCE_NAME,
        Scatter::Disjoint,
        12.0,
        size_of::<TetSpectralForceArgs>(),
        TET_SPECTRAL_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::TET_SPECTRAL_HESSIAN,
        TET_SPECTRAL_HESSIAN_NAME,
        Scatter::Disjoint,
        200.0,
        size_of::<TetSpectralHessianArgs>(),
        TET_SPECTRAL_HESSIAN_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::EXTERNAL_FIELD,
        EXTERNAL_FIELD_NAME,
        // Each vertex writes its own three accelerations, its own air
        // velocity and its own outside flag; every other access is a read.
        Scatter::Disjoint,
        8.0,
        size_of::<ExternalFieldArgs>(),
        EXTERNAL_FIELD_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::FACE_ELASTIC_EMBED_FROM_RECORDS,
        FACE_ELASTIC_EMBED_FROM_RECORDS_NAME,
        // ATOMIC for the reason the row above is: it composes that body and
        // scatters through the same two accumulators.
        Scatter::Atomic,
        // The same work plus one record load and one indexed material load.
        420.0,
        size_of::<FaceElasticEmbedFromRecordsArgs>(),
        FACE_ELASTIC_EMBED_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_BARAFFWITKIN,
        FACE_BARAFFWITKIN_NAME,
        Scatter::Disjoint,
        40.0,
        size_of::<FaceBaraffwitkinArgs>(),
        FACE_BARAFFWITKIN_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_PRESSURE_EMBED,
        FACE_PRESSURE_EMBED_NAME,
        // A face writes only its own run of each accumulator, so no two threads
        // name one word and the range may be cut anywhere.
        Scatter::Disjoint,
        40.0,
        size_of::<FacePressureEmbedArgs>(),
        FACE_PRESSURE_EMBED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FRICTION_EVALUATE,
        FRICTION_EVALUATE_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<FrictionEvaluateArgs>(),
        FRICTION_EVALUATE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::TET_MATERIAL_DIFF_TABLE,
        TET_MATERIAL_DIFF_TABLE_NAME,
        Scatter::Disjoint,
        20.0,
        size_of::<TetMaterialDiffTableArgs>(),
        TET_MATERIAL_DIFF_TABLE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_MATERIAL_DIFF_TABLE,
        FACE_MATERIAL_DIFF_TABLE_NAME,
        Scatter::Disjoint,
        14.0,
        size_of::<FaceMaterialDiffTableArgs>(),
        FACE_MATERIAL_DIFF_TABLE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_PROJECT_BODY_DOFS_ROW,
        PDRD_PROJECT_BODY_DOFS_ROW_NAME,
        // A body rewrites its own six reduced rows.
        Scatter::Disjoint,
        30.0,
        size_of::<PdrdProjectBodyDofsRowArgs>(),
        PDRD_PROJECT_BODY_DOFS_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_COPY_STATE_ROTATION_ROW,
        PDRD_COPY_STATE_ROTATION_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdCopyStateRotationRowArgs>(),
        PDRD_COPY_STATE_ROTATION_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_COMPOSE_RUNNING_ROTATION_ROW,
        PDRD_COMPOSE_RUNNING_ROTATION_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdComposeRunningRotationRowArgs>(),
        PDRD_COMPOSE_RUNNING_ROTATION_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_PROLONG_ROW,
        PDRD_PROLONG_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdProlongRowArgs>(),
        PDRD_PROLONG_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_RESTRICT_ROW,
        PDRD_RESTRICT_ROW_NAME,
        // ATOMIC, NOT DISJOINT: six atomics per body vertex into its six reduced rows.
        // Threads accumulate into shared slots, so the range may not be cut and
        // the pass runs once, serially ascending. Declaring it disjoint would
        // let a backend split the range and make the answer depend on how many
        // threads ran it.
        Scatter::Atomic,
        40.0,
        size_of::<PdrdRestrictRowArgs>(),
        PDRD_RESTRICT_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_SEED_RESTRICT_ROW,
        PDRD_SEED_RESTRICT_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdSeedRestrictRowArgs>(),
        PDRD_SEED_RESTRICT_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_COPY_PROJECTED_CLOTH_ROW,
        PDRD_COPY_PROJECTED_CLOTH_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdCopyProjectedClothRowArgs>(),
        PDRD_COPY_PROJECTED_CLOTH_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_TRANSLATION_LOCK_PARTICULAR_ROW,
        PDRD_TRANSLATION_LOCK_PARTICULAR_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdTranslationLockParticularRowArgs>(),
        PDRD_TRANSLATION_LOCK_PARTICULAR_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_EXTRACT_BODY_ROTATION_ROW,
        PDRD_EXTRACT_BODY_ROTATION_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdExtractBodyRotationRowArgs>(),
        PDRD_EXTRACT_BODY_ROTATION_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_SCATTER_ROTATED_REST_ROW,
        PDRD_SCATTER_ROTATED_REST_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdScatterRotatedRestRowArgs>(),
        PDRD_SCATTER_ROTATED_REST_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_PRECOND_BODY_ROW,
        PDRD_PRECOND_BODY_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdPrecondBodyRowArgs>(),
        PDRD_PRECOND_BODY_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::PDRD_PRECOND_CLOTH_ROW,
        PDRD_PRECOND_CLOTH_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdPrecondClothRowArgs>(),
        PDRD_PRECOND_CLOTH_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_RIGIDIFY_CENTROID_ROW,
        PDRD_RIGIDIFY_CENTROID_ROW_NAME,
        // ATOMIC, NOT DISJOINT: three atomics per body vertex into its centroid.
        // Threads accumulate into shared slots, so the range may not be cut and
        // the pass runs once, serially ascending. Declaring it disjoint would
        // let a backend split the range and make the answer depend on how many
        // threads ran it.
        Scatter::Atomic,
        40.0,
        size_of::<PdrdRigidifyCentroidRowArgs>(),
        PDRD_RIGIDIFY_CENTROID_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_RIGIDIFY_WRITE_ROW,
        PDRD_RIGIDIFY_WRITE_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdRigidifyWriteRowArgs>(),
        PDRD_RIGIDIFY_WRITE_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_FIT_CENTROID_ROW,
        PDRD_FIT_CENTROID_ROW_NAME,
        // ATOMIC, NOT DISJOINT: three atomics per body vertex into its centroid sum.
        // Threads accumulate into shared slots, so the range may not be cut and
        // the pass runs once, serially ascending. Declaring it disjoint would
        // let a backend split the range and make the answer depend on how many
        // threads ran it.
        Scatter::Atomic,
        40.0,
        size_of::<PdrdFitCentroidRowArgs>(),
        PDRD_FIT_CENTROID_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_FIT_COVARIANCE_ROW,
        PDRD_FIT_COVARIANCE_ROW_NAME,
        // ATOMIC, NOT DISJOINT: the covariance accumulated over the body's vertices.
        // Threads accumulate into shared slots, so the range may not be cut and
        // the pass runs once, serially ascending. Declaring it disjoint would
        // let a backend split the range and make the answer depend on how many
        // threads ran it.
        Scatter::Atomic,
        40.0,
        size_of::<PdrdFitCovarianceRowArgs>(),
        PDRD_FIT_COVARIANCE_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_FIT_FINISH_ROW,
        PDRD_FIT_FINISH_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdFitFinishRowArgs>(),
        PDRD_FIT_FINISH_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_ASSEMBLE_INERTIA_ROW,
        PDRD_ASSEMBLE_INERTIA_ROW_NAME,
        // A body or a vertex writes its own slot.
        Scatter::Disjoint,
        40.0,
        size_of::<PdrdAssembleInertiaRowArgs>(),
        PDRD_ASSEMBLE_INERTIA_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PDRD_ASSEMBLE_SANDWICH_ROW,
        PDRD_ASSEMBLE_SANDWICH_ROW_NAME,
        // ATOMIC, NOT DISJOINT: each vertex's sandwich folded into its body's 6x6.
        // Threads accumulate into shared slots, so the range may not be cut and
        // the pass runs once, serially ascending. Declaring it disjoint would
        // let a backend split the range and make the answer depend on how many
        // threads ran it.
        Scatter::Atomic,
        40.0,
        size_of::<PdrdAssembleSandwichRowArgs>(),
        PDRD_ASSEMBLE_SANDWICH_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PUSH_ENERGY,
        PUSH_ENERGY_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<PushEnergyArgs>(),
        PUSH_ENERGY_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PUSH_CURVATURE,
        PUSH_CURVATURE_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<PushCurvatureArgs>(),
        PUSH_CURVATURE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PUSH_GRADIENT,
        PUSH_GRADIENT_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<PushGradientArgs>(),
        PUSH_GRADIENT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PUSH_HESSIAN,
        PUSH_HESSIAN_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<PushHessianArgs>(),
        PUSH_HESSIAN_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_BEND_ANGLE,
        ROD_BEND_ANGLE_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<RodBendAngleArgs>(),
        ROD_BEND_ANGLE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_BEND_FORCE_HESSIAN,
        ROD_BEND_FORCE_HESSIAN_NAME,
        Scatter::Disjoint,
        70.0,
        size_of::<RodBendForceHessianArgs>(),
        ROD_BEND_FORCE_HESSIAN_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_BEND_EMBED,
        ROD_BEND_EMBED_NAME,
        Scatter::Atomic,
        70.0,
        size_of::<RodBendEmbedArgs>(),
        ROD_BEND_EMBED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_BEND_STIFFNESS,
        ROD_BEND_STIFFNESS_NAME,
        Scatter::Disjoint,
        1.5,
        size_of::<RodBendStiffnessArgs>(),
        ROD_BEND_STIFFNESS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SAND_GRAIN_INTEGRATE_ROW,
        SAND_GRAIN_INTEGRATE_ROW_NAME,
        // A grain writes its own slot and nothing else.
        Scatter::Disjoint,
        80.0,
        size_of::<SandGrainIntegrateRowArgs>(),
        SAND_GRAIN_INTEGRATE_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SAND_GRAIN_CONDENSE_ROW,
        SAND_GRAIN_CONDENSE_ROW_NAME,
        // A grain writes its own slot and nothing else.
        Scatter::Disjoint,
        60.0,
        size_of::<SandGrainCondenseRowArgs>(),
        SAND_GRAIN_CONDENSE_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SAND_GRAIN_RECOVER_ROW,
        SAND_GRAIN_RECOVER_ROW_NAME,
        // A grain writes its own slot and nothing else.
        Scatter::Disjoint,
        40.0,
        size_of::<SandGrainRecoverRowArgs>(),
        SAND_GRAIN_RECOVER_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::SHELL_BEND_FORCE_HESSIAN_CHECKED,
        SHELL_BEND_FORCE_HESSIAN_CHECKED_NAME,
        Scatter::Disjoint,
        90.0,
        size_of::<ShellBendForceHessianCheckedArgs>(),
        SHELL_BEND_FORCE_HESSIAN_CHECKED_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::SHELL_BEND_EMBED,
        SHELL_BEND_EMBED_NAME,
        // ATOMIC: the force scatter and the CSR push both accumulate into
        // slots several hinges share, so the host arm runs one ascending pass,
        // which is the order the separate scatter and push passes ran in.
        Scatter::Atomic,
        120.0,
        size_of::<ShellBendEmbedArgs>(),
        SHELL_BEND_EMBED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_BEND_ANGLE,
        SHELL_BEND_ANGLE_NAME,
        Scatter::Disjoint,
        10.0,
        size_of::<ShellBendAngleArgs>(),
        SHELL_BEND_ANGLE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_BEND_REMAP,
        SHELL_BEND_REMAP_NAME,
        Scatter::Disjoint,
        2.0,
        size_of::<ShellBendRemapArgs>(),
        SHELL_BEND_REMAP_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_BEND_STIFFNESS_AND_DAMPING,
        SHELL_BEND_STIFFNESS_AND_DAMPING_NAME,
        // DISJOINT, like the row above: each element writes its own slot.
        Scatter::Disjoint,
        // The same arithmetic plus a record load and an indexed material load,
        // and a gate that returns before either for an excluded hinge.
        12.0,
        size_of::<ShellBendStiffnessAndDampingArgs>(),
        SHELL_BEND_STIFFNESS_AND_DAMPING_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_BEND_STIFFNESS,
        SHELL_BEND_STIFFNESS_NAME,
        Scatter::Disjoint,
        2.0,
        size_of::<ShellBendStiffnessArgs>(),
        SHELL_BEND_STIFFNESS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_BEND_AREAL_DENSITY_FROM_RECORDS,
        SHELL_BEND_AREAL_DENSITY_FROM_RECORDS_NAME,
        Scatter::Disjoint,
        4.0,
        size_of::<ShellBendArealDensityFromRecordsArgs>(),
        SHELL_BEND_AREAL_DENSITY_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_BEND_AREAL_DENSITY_GATHERED,
        SHELL_BEND_AREAL_DENSITY_GATHERED_NAME,
        Scatter::Disjoint,
        4.0,
        size_of::<ShellBendArealDensityGatheredArgs>(),
        SHELL_BEND_AREAL_DENSITY_GATHERED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::STITCH_FORCE_HESSIAN_GATHERED,
        STITCH_FORCE_HESSIAN_GATHERED_NAME,
        Scatter::Disjoint,
        120.0,
        size_of::<StitchForceHessianGatheredArgs>(),
        STITCH_FORCE_HESSIAN_GATHERED_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::TORQUE_GROUP_FRAME,
        TORQUE_GROUP_FRAME_NAME,
        // A group writes its own result slot and nothing else. It READS other
        // groups' members only in the sense that the member array is shared;
        // the run it walks is its own, so no two threads write one word.
        Scatter::Disjoint,
        // Three walks over the group's members plus a symmetric 3x3
        // eigendecomposition, so the per-item cost is the group's size rather
        // than a constant. This is the figure for a small group; a large one
        // pays proportionally more and the range may be cut anywhere.
        200.0,
        size_of::<TorqueGroupFrameArgs>(),
        TORQUE_GROUP_FRAME_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_STRETCH_DIFF_TABLE,
        ROD_STRETCH_DIFF_TABLE_NAME,
        Scatter::Disjoint,
        20.0,
        size_of::<RodStretchDiffTableArgs>(),
        ROD_STRETCH_DIFF_TABLE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_STRETCH_EMBED,
        ROD_STRETCH_EMBED_NAME,
        Scatter::Atomic,
        20.0,
        size_of::<RodStretchEmbedArgs>(),
        ROD_STRETCH_EMBED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::TET_SPECTRAL_CONVERT_HESSIAN,
        TET_SPECTRAL_CONVERT_HESSIAN_NAME,
        Scatter::Disjoint,
        160.0,
        size_of::<TetSpectralConvertHessianArgs>(),
        TET_SPECTRAL_CONVERT_HESSIAN_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::TET_ELASTIC_EMBED,
        TET_ELASTIC_EMBED_NAME,
        // TWO TETS SHARING A VERTEX HAVE ONE FORCE DESTINATION AND TWO SHARING
        // AN EDGE HAVE ONE CSR SLOT, so the fold is a data race unless the
        // whole range runs as one ascending pass. `compute::atomic_add` is a
        // plain read, add and write back on the host seam, and the fp32
        // running sums in `force` and in the matrix's values depend on the
        // ORDER besides, which is what makes this backend's answer independent
        // of the thread count.
        Scatter::Atomic,
        // The sum of the stages it replaces (the deformation gradient, the
        // factorization, the material table, the two spectral passes, the two
        // converters and the damping) plus the two scatters. `Atomic` runs the
        // range whole, so this figure sets no chunk width; it is stated at the
        // shape of the work for a reader comparing rows.
        620.0,
        size_of::<TetElasticEmbedArgs>(),
        TET_ELASTIC_EMBED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::TET_MATERIAL_FROM_RECORDS,
        TET_MATERIAL_FROM_RECORDS_NAME,
        Scatter::Disjoint,
        20.0,
        size_of::<TetMaterialFromRecordsArgs>(),
        TET_MATERIAL_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::BITONIC_STEP,
        BITONIC_STEP_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<BitonicStepArgs>(),
        BITONIC_STEP_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LBVH_MORTON_FROM_BOUNDS,
        LBVH_MORTON_FROM_BOUNDS_NAME,
        Scatter::Disjoint,
        3.0,
        size_of::<LbvhMortonFromBoundsArgs>(),
        LBVH_MORTON_FROM_BOUNDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LBVH_NODES,
        LBVH_NODES_NAME,
        Scatter::Disjoint,
        60.0,
        size_of::<LbvhNodesArgs>(),
        LBVH_NODES_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LBVH_NODE_DEPTH,
        LBVH_NODE_DEPTH_NAME,
        Scatter::Disjoint,
        40.0,
        size_of::<LbvhNodeDepthArgs>(),
        LBVH_NODE_DEPTH_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_CENTROID,
        FACE_CENTROID_NAME,
        Scatter::Disjoint,
        15.0,
        size_of::<FaceCentroidArgs>(),
        FACE_CENTROID_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::EDGE_CENTROID,
        EDGE_CENTROID_NAME,
        Scatter::Disjoint,
        10.0,
        size_of::<EdgeCentroidArgs>(),
        EDGE_CENTROID_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VERTEX_CENTROID,
        VERTEX_CENTROID_NAME,
        Scatter::Disjoint,
        5.0,
        size_of::<VertexCentroidArgs>(),
        VERTEX_CENTROID_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LBVH_SET_PARENT,
        LBVH_SET_PARENT_NAME,
        // DISJOINT WITHOUT AN ATOMIC: a child has exactly one parent, so the
        // two writes each thread makes are the only writes to those slots.
        Scatter::Disjoint,
        2.0,
        size_of::<LbvhSetParentArgs>(),
        LBVH_SET_PARENT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LBVH_FIND_ROOT,
        LBVH_FIND_ROOT_NAME,
        // ATOMIC: the counter beside the index is what lets the host fail on a
        // tree with two roots, which a serial host scan would assert instead.
        Scatter::Atomic,
        2.0,
        size_of::<LbvhFindRootArgs>(),
        LBVH_FIND_ROOT_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::LBVH_COUNT_LEVELS,
        LBVH_COUNT_LEVELS_NAME,
        Scatter::Atomic,
        2.0,
        size_of::<LbvhCountLevelsArgs>(),
        LBVH_COUNT_LEVELS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LBVH_SCATTER_LEVELS,
        LBVH_SCATTER_LEVELS_NAME,
        // CLAIM: the slot within a level comes from an atomic cursor.
        Scatter::Claim,
        2.0,
        size_of::<LbvhScatterLevelsArgs>(),
        LBVH_SCATTER_LEVELS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DIRICHLET_PRESCRIBE_GATED,
        DIRICHLET_PRESCRIBE_GATED_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<DirichletPrescribeGatedArgs>(),
        DIRICHLET_PRESCRIBE_GATED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DIRICHLET_LIFT_ROW,
        DIRICHLET_LIFT_ROW_NAME,
        // SERIAL BY CONTRACT, carried across the conversion unchanged: the lift
        // folds into ANOTHER row's force through a plain read, add and write
        // back, so two threads walking two rows that couple to one free row
        // would race. A generated entry says nothing about how its range may be
        // cut, so the rule lives here.
        Scatter::Atomic,
        4.0,
        size_of::<DirichletLiftRowArgs>(),
        DIRICHLET_LIFT_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DUMP_LINSYS_ROW_TO_COO,
        DUMP_LINSYS_ROW_TO_COO_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<DumpLinsysRowToCooArgs>(),
        DUMP_LINSYS_ROW_TO_COO_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DX_MAGNITUDE,
        DX_MAGNITUDE_NAME,
        Scatter::Disjoint,
        0.5,
        size_of::<DxMagnitudeArgs>(),
        DX_MAGNITUDE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::DX_SEED,
        DX_SEED_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<DxSeedArgs>(),
        DX_SEED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FIX_XZ_DRAG,
        FIX_XZ_DRAG_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<FixXzDragArgs>(),
        FIX_XZ_DRAG_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::MOMENTUM_EMBED,
        MOMENTUM_EMBED_NAME,
        // Each vertex's force and diagonal block are written by its own thread
        // and by no other; the incident-face walk inside the body is a read.
        Scatter::Disjoint,
        20.0,
        size_of::<MomentumEmbedArgs>(),
        MOMENTUM_EMBED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::GATHER_POSITION_ABSOLUTE,
        GATHER_POSITION_ABSOLUTE_NAME,
        Scatter::Disjoint,
        0.5,
        size_of::<GatherPositionAbsoluteArgs>(),
        GATHER_POSITION_ABSOLUTE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::OVERRIDE_VELOCITY_SEED_LISTED,
        OVERRIDE_VELOCITY_SEED_LISTED_NAME,
        // SERIAL BY CONTRACT, and the declaration cannot say so: the index list
        // arrives from a keyframe with no proof it holds each vertex at most
        // once, and the angular seed reads `prev` and writes it back. A
        // generated entry covers whatever range it is handed and states nothing
        // about how that range may be cut, so the rule lives here.
        Scatter::Atomic,
        1.0,
        size_of::<OverrideVelocitySeedListedArgs>(),
        OVERRIDE_VELOCITY_SEED_LISTED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::OVERRIDE_ANGULAR_SEED_LISTED,
        OVERRIDE_ANGULAR_SEED_LISTED_NAME,
        // SERIAL BY CONTRACT, and the declaration cannot say so: the index list
        // arrives from a keyframe with no proof it holds each vertex at most
        // once, and the angular seed reads `prev` and writes it back. A
        // generated entry covers whatever range it is handed and states nothing
        // about how that range may be cut, so the rule lives here.
        Scatter::Atomic,
        1.0,
        size_of::<OverrideAngularSeedListedArgs>(),
        OVERRIDE_ANGULAR_SEED_LISTED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::POSITION_ACCEPT,
        POSITION_ACCEPT_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<PositionAcceptArgs>(),
        POSITION_ACCEPT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::POSITION_STEP,
        POSITION_STEP_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<PositionStepArgs>(),
        POSITION_STEP_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::REWIND_FIX,
        REWIND_FIX_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<RewindFixArgs>(),
        REWIND_FIX_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_STRETCH_RATIO_GATED,
        ROD_STRETCH_RATIO_GATED_NAME,
        Scatter::Disjoint,
        4.0,
        size_of::<RodStretchRatioGatedArgs>(),
        ROD_STRETCH_RATIO_GATED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::COMPUTE_TARGET_SEED,
        COMPUTE_TARGET_SEED_NAME,
        Scatter::Disjoint,
        3.0,
        size_of::<ComputeTargetSeedArgs>(),
        COMPUTE_TARGET_SEED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VELOCITY_TERMS,
        VELOCITY_TERMS_NAME,
        Scatter::Disjoint,
        2.0,
        size_of::<VelocityTermsArgs>(),
        VELOCITY_TERMS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_ALPHA,
        PLASTICITY_ALPHA_NAME,
        Scatter::Disjoint,
        2.0,
        size_of::<PlasticityAlphaArgs>(),
        PLASTICITY_ALPHA_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_FACE_INVERSE_REST,
        PLASTICITY_FACE_INVERSE_REST_NAME,
        Scatter::Disjoint,
        14.0,
        size_of::<PlasticityFaceInverseRestArgs>(),
        PLASTICITY_FACE_INVERSE_REST_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_TET_INVERSE_REST,
        PLASTICITY_TET_INVERSE_REST_NAME,
        Scatter::Disjoint,
        20.0,
        size_of::<PlasticityTetInverseRestArgs>(),
        PLASTICITY_TET_INVERSE_REST_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_CREEP_SINGULAR2,
        PLASTICITY_CREEP_SINGULAR2_NAME,
        Scatter::Disjoint,
        6.0,
        size_of::<PlasticityCreepSingular2Args>(),
        PLASTICITY_CREEP_SINGULAR2_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_CREEP_SINGULAR3,
        PLASTICITY_CREEP_SINGULAR3_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<PlasticityCreepSingular3Args>(),
        PLASTICITY_CREEP_SINGULAR3_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_CREEP_REST_ANGLE,
        PLASTICITY_CREEP_REST_ANGLE_NAME,
        Scatter::Disjoint,
        4.0,
        size_of::<PlasticityCreepRestAngleArgs>(),
        PLASTICITY_CREEP_REST_ANGLE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_COMMIT_FACE,
        PLASTICITY_COMMIT_FACE_NAME,
        // DISJOINT: an element writes its own row of the destination and no
        // other thread's.
        Scatter::Disjoint,
        2.0,
        size_of::<PlasticityCommitFaceArgs>(),
        PLASTICITY_COMMIT_FACE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_COMMIT_TET,
        PLASTICITY_COMMIT_TET_NAME,
        // DISJOINT: an element writes its own row of the destination and no
        // other thread's.
        Scatter::Disjoint,
        2.0,
        size_of::<PlasticityCommitTetArgs>(),
        PLASTICITY_COMMIT_TET_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_FACE_FROM_RECORDS,
        PLASTICITY_FACE_FROM_RECORDS_NAME,
        Scatter::Disjoint,
        2.0,
        size_of::<PlasticityFaceFromRecordsArgs>(),
        PLASTICITY_FACE_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_HINGE_FROM_RECORDS,
        PLASTICITY_HINGE_FROM_RECORDS_NAME,
        Scatter::Disjoint,
        2.0,
        size_of::<PlasticityHingeFromRecordsArgs>(),
        PLASTICITY_HINGE_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_TET_FROM_RECORDS,
        PLASTICITY_TET_FROM_RECORDS_NAME,
        Scatter::Disjoint,
        2.0,
        size_of::<PlasticityTetFromRecordsArgs>(),
        PLASTICITY_TET_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PLASTICITY_ROD_FROM_RECORDS,
        PLASTICITY_ROD_FROM_RECORDS_NAME,
        Scatter::Disjoint,
        2.0,
        size_of::<PlasticityRodFromRecordsArgs>(),
        PLASTICITY_ROD_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::RADIX_HISTOGRAM,
        RADIX_HISTOGRAM_NAME,
        // DISJOINT: one GROUP owns one block of keys and one column of the
        // block histogram. The lanes share the group-local counts, which is not
        // a scatter and is not visible to the cut.
        Scatter::Disjoint,
        // Per GROUP: a block's keys read once and one increment each.
        64.0,
        size_of::<RadixHistogramArgs>(),
        RADIX_HISTOGRAM_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::RADIX_SCATTER,
        RADIX_SCATTER_NAME,
        // DISJOINT: the destinations come from this block's own prefix, so two
        // blocks never write one slot. That is the property the scan before it
        // establishes.
        Scatter::Disjoint,
        96.0,
        size_of::<RadixScatterArgs>(),
        RADIX_SCATTER_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::BOUNDS_LEAF,
        BOUNDS_LEAF_NAME,
        // DISJOINT: a thread owns one block of the input and writes one
        // six-float record of its own.
        Scatter::Disjoint,
        4096.0,
        size_of::<BoundsLeafArgs>(),
        BOUNDS_LEAF_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::BOUNDS_MERGE,
        BOUNDS_MERGE_NAME,
        Scatter::Disjoint,
        4096.0,
        size_of::<BoundsMergeArgs>(),
        BOUNDS_MERGE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::REDUCE_MIN_LEAF,
        REDUCE_MIN_LEAF_NAME,
        Scatter::Disjoint,
        256.0,
        size_of::<ReduceMinLeafArgs>(),
        REDUCE_MIN_LEAF_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::REDUCE_MAX_LEAF,
        REDUCE_MAX_LEAF_NAME,
        Scatter::Disjoint,
        256.0,
        size_of::<ReduceMaxLeafArgs>(),
        REDUCE_MAX_LEAF_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::REDUCE_MIN_U32_LEAF,
        REDUCE_MIN_U32_LEAF_NAME,
        // DISJOINT: a thread owns one block of the input and writes one word
        // of its own, as the float minimum does.
        Scatter::Disjoint,
        256.0,
        size_of::<ReduceMinU32LeafArgs>(),
        REDUCE_MIN_U32_LEAF_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::REDUCE_SUM_U32_LEAF,
        REDUCE_SUM_U32_LEAF_NAME,
        // DISJOINT: one block of words in, its own (low, high) pair out.
        Scatter::Disjoint,
        256.0,
        size_of::<ReduceSumU32LeafArgs>(),
        REDUCE_SUM_U32_LEAF_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::REDUCE_SUM_WIDE_MERGE,
        REDUCE_SUM_WIDE_MERGE_NAME,
        // DISJOINT: one block of pairs in, its own pair out, into a span the
        // source does not overlap.
        Scatter::Disjoint,
        512.0,
        size_of::<ReduceSumWideMergeArgs>(),
        REDUCE_SUM_WIDE_MERGE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCAN_BLOCK_TOTAL,
        SCAN_BLOCK_TOTAL_NAME,
        // DISJOINT: one GROUP owns one block of the array and the one slot of
        // `total` that names it. The lanes fold through group-local scratch,
        // which is not a scatter and is not visible to the cut.
        Scatter::Disjoint,
        // Per GROUP: the block's elements read once, which is what the lanes
        // divide between them rather than what any one of them does.
        256.0,
        size_of::<ScanBlockTotalArgs>(),
        SCAN_BLOCK_TOTAL_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCAN_BLOCK_APPLY,
        SCAN_BLOCK_APPLY_NAME,
        Scatter::Disjoint,
        256.0,
        size_of::<ScanBlockApplyArgs>(),
        SCAN_BLOCK_APPLY_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCAN_ZERO,
        SCAN_ZERO_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<ScanZeroArgs>(),
        SCAN_ZERO_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_COMBINE_INDIRECT,
        VEC_COMBINE_INDIRECT_NAME,
        // DISJOINT AND 0.5, WHICH IS `vec_combine`'S ROW, because this body is
        // `vec_combine` with one coefficient read from a device scalar: it
        // writes `destination[index]` and nothing else, so the range may be
        // cut anywhere, and the deliberate aliasing of `p` reads element `i`
        // before writing element `i` inside one chunk.
        //
        // IT CARRIED `Atomic` AND 1.0 WHILE NOTHING DISPATCHED IT, as a
        // placeholder that could not be wrong about a kernel nobody launched.
        // It became wrong the moment the PCG direction update started using
        // it: `Atomic` is not merely conservative on this backend, it forces
        // ONE SERIAL ASCENDING PASS (`host.rs:786`), so the update over
        // `3 * vertices` floats ran single-threaded while every other vector
        // update beside it ran across chunks. Measured on `drape`, 245,760
        // floats an iteration.
        Scatter::Disjoint,
        0.5,
        size_of::<VecCombineIndirectArgs>(),
        VEC_COMBINE_INDIRECT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_ADD_SCALED_INDIRECT,
        VEC_ADD_SCALED_INDIRECT_NAME,
        Scatter::Disjoint,
        0.5,
        size_of::<VecAddScaledIndirectArgs>(),
        VEC_ADD_SCALED_INDIRECT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_COPY,
        VEC_COPY_NAME,
        Scatter::Disjoint,
        0.5,
        size_of::<VecCopyArgs>(),
        VEC_COPY_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_ADD_SCALED,
        VEC_ADD_SCALED_NAME,
        Scatter::Disjoint,
        0.5,
        size_of::<VecAddScaledArgs>(),
        VEC_ADD_SCALED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_COMBINE,
        VEC_COMBINE_NAME,
        Scatter::Disjoint,
        0.5,
        size_of::<VecCombineArgs>(),
        VEC_COMBINE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_FILL,
        VEC_FILL_NAME,
        Scatter::Disjoint,
        0.5,
        size_of::<VecFillArgs>(),
        VEC_FILL_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_FILL_U32,
        VEC_FILL_U32_NAME,
        // DISJOINT, AND IT WAS A PLACEHOLDER UNTIL SOMETHING LAUNCHED IT. The
        // body is `array[index] = value;` at the thread index and nothing else
        // (`primitives/vec_ops.kernel.cpp`), which is the definition of
        // disjoint; the row said `Atomic` because it was written for an entry
        // the driver did not yet reach, where neither field can be wrong. It is
        // reached now, from the CSR transpose twice per rebuild and from the
        // CCD, contact and intersection clears, over vertex-count arrays, and
        // an `Atomic` row runs ONE SERIAL ASCENDING PASS on the host backend
        // while the identically shaped float `vec_fill` above it runs chunked.
        // The cost matches that sibling for the same reason: one store.
        Scatter::Disjoint,
        0.5,
        size_of::<VecFillU32Args>(),
        VEC_FILL_U32_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ELEMENT_ADD_SCALED,
        ELEMENT_ADD_SCALED_NAME,
        Scatter::Disjoint,
        // The strides dispatched here run from 1 to 324 floats, so no single
        // figure is the body's cost. This one sits in the middle of them, and
        // a wrong figure costs scheduling and never an answer: every element
        // writes its own span and nothing else.
        20.0,
        size_of::<ElementAddScaledArgs>(),
        ELEMENT_ADD_SCALED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_BLOCK_SUM,
        VEC_BLOCK_SUM_NAME,
        // DISJOINT: one element owns one block of the input and writes one
        // slot of the output, so the range is cut anywhere. The cost is per
        // ELEMENT and an element is a whole block, which is why this is 256
        // times a vector update rather than beside one.
        Scatter::Disjoint,
        128.0,
        size_of::<VecBlockSumArgs>(),
        VEC_BLOCK_SUM_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_BLOCK_SUM_U32,
        VEC_BLOCK_SUM_U32_NAME,
        // DISJOINT: one element owns one block of the input and writes one
        // slot of the output, so the range is cut anywhere. The cost is per
        // ELEMENT and an element is a whole block, which is why this is 256
        // times a vector update rather than beside one.
        Scatter::Disjoint,
        128.0,
        size_of::<VecBlockSumU32Args>(),
        VEC_BLOCK_SUM_U32_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_BLOCK_SUM_COOPERATIVE,
        VEC_BLOCK_SUM_COOPERATIVE_NAME,
        // DISJOINT for the reason the element form is: one GROUP owns one block
        // of the input and one slot of the output, so the range is cut on any
        // group boundary. The lanes inside a group share their fold through the
        // group-local scratch, which is not a scatter and is not visible to the
        // cut.
        Scatter::Disjoint,
        // Per GROUP, and a group is a whole block, so this is the element form's
        // cost: the same values are read and summed, by more threads.
        128.0,
        size_of::<VecBlockSumCooperativeArgs>(),
        VEC_BLOCK_SUM_COOPERATIVE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_BLOCK_SUM_ABS_COOPERATIVE,
        VEC_BLOCK_SUM_ABS_COOPERATIVE_NAME,
        // DISJOINT: one GROUP owns one block of the input and one slot of each
        // output. The lanes share their fold through the group-local scratch,
        // which is not a scatter and is not visible to the cut.
        Scatter::Disjoint,
        128.0,
        size_of::<VecBlockSumAbsCooperativeArgs>(),
        VEC_BLOCK_SUM_ABS_COOPERATIVE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_BLOCK_SUM_PAIR_COOPERATIVE,
        VEC_BLOCK_SUM_PAIR_COOPERATIVE_NAME,
        // DISJOINT: one GROUP owns one block of the input and one slot of each
        // output. The lanes share their fold through the group-local scratch,
        // which is not a scatter and is not visible to the cut.
        Scatter::Disjoint,
        128.0,
        size_of::<VecBlockSumPairCooperativeArgs>(),
        VEC_BLOCK_SUM_PAIR_COOPERATIVE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_BLOCK_SUM_DUAL_COOPERATIVE,
        VEC_BLOCK_SUM_DUAL_COOPERATIVE_NAME,
        // DISJOINT: one GROUP owns one block of the input and one slot of each
        // output. The lanes share their fold through the group-local scratch,
        // which is not a scatter and is not visible to the cut.
        Scatter::Disjoint,
        128.0,
        size_of::<VecBlockSumDualCooperativeArgs>(),
        VEC_BLOCK_SUM_DUAL_COOPERATIVE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_BLOCK_SUM_ABS,
        VEC_BLOCK_SUM_ABS_NAME,
        Scatter::Disjoint,
        128.0,
        size_of::<VecBlockSumAbsArgs>(),
        VEC_BLOCK_SUM_ABS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_BLOCK_SUM_PAIR,
        VEC_BLOCK_SUM_PAIR_NAME,
        // DISJOINT, like both bodies it fuses. Each element writes its own
        // block's slot in each of the two outputs and reads nobody else's, and
        // sharing a dispatch does not make two threads share a slot.
        Scatter::Disjoint,
        // Twice one fold's per-item cost, because it does both.
        256.0,
        size_of::<VecBlockSumPairArgs>(),
        VEC_BLOCK_SUM_PAIR_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VEC_BLOCK_SUM_DUAL,
        VEC_BLOCK_SUM_DUAL_NAME,
        // DISJOINT, like every body it fuses. Each element writes its own
        // block's slot in each of the two outputs and reads nobody else's, and
        // sharing a dispatch does not make two threads share a slot. A thread
        // past a source's own count writes nothing for that source, which
        // removes slots rather than sharing them.
        Scatter::Disjoint,
        // Twice one fold's per-item cost, because it does both. A thread past
        // one source's count does less, so this is the bound rather than the
        // mean.
        256.0,
        size_of::<VecBlockSumDualArgs>(),
        VEC_BLOCK_SUM_DUAL_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_COUNT_MEMBERS,
        SCHWARZ_COUNT_MEMBERS_NAME,
        Scatter::Atomic,
        1.0,
        size_of::<SchwarzCountMembersArgs>(),
        SCHWARZ_COUNT_MEMBERS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_SCATTER_MEMBERS,
        SCHWARZ_SCATTER_MEMBERS_NAME,
        Scatter::Atomic,
        1.0,
        size_of::<SchwarzScatterMembersArgs>(),
        SCHWARZ_SCATTER_MEMBERS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_DOMAIN_INVERSE_SIZE,
        SCHWARZ_DOMAIN_INVERSE_SIZE_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzDomainInverseSizeArgs>(),
        SCHWARZ_DOMAIN_INVERSE_SIZE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_FINE_GRAPH_COUNT,
        SCHWARZ_FINE_GRAPH_COUNT_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzFineGraphCountArgs>(),
        SCHWARZ_FINE_GRAPH_COUNT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_FINE_GRAPH_FILL,
        SCHWARZ_FINE_GRAPH_FILL_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzFineGraphFillArgs>(),
        SCHWARZ_FINE_GRAPH_FILL_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_FACTOR_GATHER,
        SCHWARZ_FACTOR_GATHER_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzFactorGatherArgs>(),
        SCHWARZ_FACTOR_GATHER_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_FACTOR_FLOOR,
        SCHWARZ_FACTOR_FLOOR_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzFactorFloorArgs>(),
        SCHWARZ_FACTOR_FLOOR_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_FACTOR_CHOLESKY_DIAGONAL,
        SCHWARZ_FACTOR_CHOLESKY_DIAGONAL_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzFactorCholeskyDiagonalArgs>(),
        SCHWARZ_FACTOR_CHOLESKY_DIAGONAL_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_FACTOR_CHOLESKY_COLUMN,
        SCHWARZ_FACTOR_CHOLESKY_COLUMN_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzFactorCholeskyColumnArgs>(),
        SCHWARZ_FACTOR_CHOLESKY_COLUMN_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_FACTOR_INVERSE_COLUMN,
        SCHWARZ_FACTOR_INVERSE_COLUMN_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzFactorInverseColumnArgs>(),
        SCHWARZ_FACTOR_INVERSE_COLUMN_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_FACTOR_PACK,
        SCHWARZ_FACTOR_PACK_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzFactorPackArgs>(),
        SCHWARZ_FACTOR_PACK_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_APPLY_GATHER,
        SCHWARZ_APPLY_GATHER_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzApplyGatherArgs>(),
        SCHWARZ_APPLY_GATHER_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_APPLY_LOWER,
        SCHWARZ_APPLY_LOWER_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzApplyLowerArgs>(),
        SCHWARZ_APPLY_LOWER_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_APPLY_UPPER,
        SCHWARZ_APPLY_UPPER_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzApplyUpperArgs>(),
        SCHWARZ_APPLY_UPPER_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_RESTRICT_ROW,
        SCHWARZ_RESTRICT_ROW_NAME,
        Scatter::Atomic,
        1.0,
        size_of::<SchwarzRestrictRowArgs>(),
        SCHWARZ_RESTRICT_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_PROLONG_ROW,
        SCHWARZ_PROLONG_ROW_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzProlongRowArgs>(),
        SCHWARZ_PROLONG_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_COMPOSE_MAP_ROW,
        SCHWARZ_COMPOSE_MAP_ROW_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzComposeMapRowArgs>(),
        SCHWARZ_COMPOSE_MAP_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_LEVEL0_COUNT,
        SCHWARZ_LEVEL0_COUNT_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzLevel0CountArgs>(),
        SCHWARZ_LEVEL0_COUNT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_LEVEL0_FILL,
        SCHWARZ_LEVEL0_FILL_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzLevel0FillArgs>(),
        SCHWARZ_LEVEL0_FILL_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_COARSE_GATHER,
        SCHWARZ_COARSE_GATHER_NAME,
        // EVERY OUTPUT ELEMENT IS WRITTEN BY EXACTLY ONE THREAD, so the range may
        // be cut. The body reaches no `compute::atomic_*`: the four Schwarz
        // bodies that do are `count_members`, `scatter_members`, `restrict_row`
        // and `galerkin_edge_head`, and each of those is declared `Atomic`
        // beside this. The per-item cost below is still an unmeasured
        // placeholder, and it is what decides whether this range is cut at all.
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzCoarseGatherArgs>(),
        SCHWARZ_COARSE_GATHER_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_GALERKIN_KEY,
        SCHWARZ_GALERKIN_KEY_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<SchwarzGalerkinKeyArgs>(),
        SCHWARZ_GALERKIN_KEY_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_GALERKIN_EDGE_FLAG,
        SCHWARZ_GALERKIN_EDGE_FLAG_NAME,
        Scatter::Disjoint,
        0.5,
        size_of::<SchwarzGalerkinEdgeFlagArgs>(),
        SCHWARZ_GALERKIN_EDGE_FLAG_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_GALERKIN_EDGE_HEAD,
        SCHWARZ_GALERKIN_EDGE_HEAD_NAME,
        Scatter::Atomic,
        1.0,
        size_of::<SchwarzGalerkinEdgeHeadArgs>(),
        SCHWARZ_GALERKIN_EDGE_HEAD_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SCHWARZ_GALERKIN_SEGMENT_SUM,
        SCHWARZ_GALERKIN_SEGMENT_SUM_NAME,
        Scatter::Disjoint,
        9.0,
        size_of::<SchwarzGalerkinSegmentSumArgs>(),
        SCHWARZ_GALERKIN_SEGMENT_SUM_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::BLOCK_JACOBI_INVERT_ROW,
        BLOCK_JACOBI_INVERT_ROW_NAME,
        Scatter::Disjoint,
        3.0,
        size_of::<BlockJacobiInvertRowArgs>(),
        BLOCK_JACOBI_INVERT_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PCG_DOT_TERMS,
        PCG_DOT_TERMS_NAME,
        Scatter::Disjoint,
        0.5,
        size_of::<PcgDotTermsArgs>(),
        PCG_DOT_TERMS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PCG_UPDATE_ROW,
        PCG_UPDATE_ROW_NAME,
        // DISJOINT: a row writes its own three components of the iterate, the
        // residual and the preconditioned residual, and its own two term
        // slots. No two rows share an element, which is why one dispatch can
        // do what four did.
        Scatter::Disjoint,
        // Four passes' worth of work at one row: two fused multiply-adds a
        // component, a 3x3 apply, and the row's dot.
        192.0,
        size_of::<PcgUpdateRowArgs>(),
        PCG_UPDATE_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PCG_UPDATE_ROW_FOLDED,
        PCG_UPDATE_ROW_FOLDED_NAME,
        // DISJOINT: one GROUP owns one run of rows, the rows of the three
        // vectors its lanes wrote, and one slot of each partial array.
        Scatter::Disjoint,
        8.0 * super::operator::APPLY_GROUP as f64,
        size_of::<PcgUpdateRowFoldedArgs>(),
        PCG_UPDATE_ROW_FOLDED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PCG_ALPHA_TERMS,
        PCG_ALPHA_TERMS_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<PcgAlphaTermsArgs>(),
        PCG_ALPHA_TERMS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PCG_BETA_TERMS,
        PCG_BETA_TERMS_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<PcgBetaTermsArgs>(),
        PCG_BETA_TERMS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PCG_ALPHA_RESIDENT,
        PCG_ALPHA_RESIDENT_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<PcgAlphaResidentArgs>(),
        PCG_ALPHA_RESIDENT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PCG_FOLD_ALPHA,
        PCG_FOLD_ALPHA_NAME,
        // DISJOINT: ONE group, which owns every scalar it writes.
        Scatter::Disjoint,
        4.0,
        size_of::<PcgFoldAlphaArgs>(),
        PCG_FOLD_ALPHA_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PCG_BETA_RESIDENT,
        PCG_BETA_RESIDENT_NAME,
        // ATOMIC, WHICH IS THE HONEST DECLARATION EVEN AT ONE ELEMENT. The
        // body writes its three verdict slots at its own index, which would be
        // `Disjoint`, and it also carries the `rz0 <- rz1` roll, which writes a
        // slot no index reaches. `Atomic` never lets the host arm cut a range
        // it must not cut, so a dispatch of this at any extent stays one
        // ascending pass; the caller dispatches exactly one element, so the
        // choice costs nothing and states the property rather than relying on
        // the extent to hide it.
        Scatter::Atomic,
        1.0,
        size_of::<PcgBetaResidentArgs>(),
        PCG_BETA_RESIDENT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PCG_FOLD_BETA,
        PCG_FOLD_BETA_NAME,
        Scatter::Disjoint,
        4.0,
        size_of::<PcgFoldBetaArgs>(),
        PCG_FOLD_BETA_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::PCG_RIGID_GROUP_L1,
        PCG_RIGID_GROUP_L1_NAME,
        // DISJOINT: a body writes its own slot of the norm array and nothing
        // else, which is what the body's single `norm = ...` says.
        Scatter::Disjoint,
        0.5,
        size_of::<PcgRigidGroupL1Args>(),
        PCG_RIGID_GROUP_L1_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::OPERATOR_APPLY,
        OPERATOR_APPLY_NAME,
        Scatter::Disjoint,
        6.0,
        size_of::<OperatorApplyArgs>(),
        OPERATOR_APPLY_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::OPERATOR_APPLY_DYNAMIC,
        OPERATOR_APPLY_DYNAMIC_NAME,
        Scatter::Disjoint,
        6.0,
        size_of::<OperatorApplyDynamicArgs>(),
        OPERATOR_APPLY_DYNAMIC_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::OPERATOR_APPLY_FOLDED,
        OPERATOR_APPLY_FOLDED_NAME,
        // DISJOINT: one GROUP owns one contiguous run of rows, one slot of each
        // partial array, and the rows of `result` its own lanes wrote. The
        // lanes share their two folds through group-local scratch, which is not
        // a scatter and is not visible to the cut.
        Scatter::Disjoint,
        6.0 * super::operator::APPLY_GROUP as f64,
        size_of::<OperatorApplyFoldedArgs>(),
        OPERATOR_APPLY_FOLDED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::OPERATOR_APPLY_DYNAMIC_FOLDED,
        OPERATOR_APPLY_DYNAMIC_FOLDED_NAME,
        Scatter::Disjoint,
        // Eight lanes share a row; the scheduler cuts whole groups.
        6.0 * (super::operator::APPLY_GROUP / super::operator::SPMV_ROW_LANES) as f64,
        size_of::<OperatorApplyDynamicFoldedArgs>(),
        OPERATOR_APPLY_DYNAMIC_FOLDED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::OPERATOR_APPLY_SYMMETRIC_FOLDED,
        OPERATOR_APPLY_SYMMETRIC_FOLDED_NAME,
        // ATOMIC, because the body scatters each stored block's transpose into
        // a row that is not the thread's own. That is what makes it symmetric
        // and it is why the row cannot be `Disjoint`.
        Scatter::Atomic,
        6.0,
        size_of::<OperatorApplySymmetricFoldedArgs>(),
        OPERATOR_APPLY_SYMMETRIC_FOLDED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::MAT3_MUL,
        MAT3_MUL_NAME,
        Scatter::Disjoint,
        4.0,
        size_of::<Mat3MulArgs>(),
        MAT3_MUL_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FIXED_CSR_PRODUCT_ROW,
        FIXED_CSR_PRODUCT_ROW_NAME,
        Scatter::Disjoint,
        5.0,
        size_of::<FixedCsrProductRowArgs>(),
        FIXED_CSR_PRODUCT_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::TRANSLATION_LOCK_DRIFT_ROW,
        TRANSLATION_LOCK_DRIFT_ROW_NAME,
        // MANY THREADS ACCUMULATE INTO ONE GROUP SLOT, so the range may NOT be
        // cut: `compute::atomic_add` is a plain read-modify-write on the host arm
        // (`seam/seam_host.h`), and a chunked Disjoint pass loses updates.
        Scatter::Atomic,
        12.0,
        size_of::<TranslationLockDriftRowArgs>(),
        TRANSLATION_LOCK_DRIFT_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LOCK_FRAME_CLEAR_ROW,
        LOCK_FRAME_CLEAR_ROW_NAME,
        // One group writes its own frame and nothing else.
        Scatter::Disjoint,
        4.0,
        size_of::<LockFrameClearRowArgs>(),
        LOCK_FRAME_CLEAR_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LOCK_FRAME_CENTER_OF_MASS_ROW,
        LOCK_FRAME_CENTER_OF_MASS_ROW_NAME,
        // One group writes its own frame's centroid.
        Scatter::Disjoint,
        6.0,
        size_of::<LockFrameCenterOfMassRowArgs>(),
        LOCK_FRAME_CENTER_OF_MASS_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LOCK_CENTER_OF_MASS_ACCUMULATE_ROW,
        LOCK_CENTER_OF_MASS_ACCUMULATE_ROW_NAME,
        // MANY THREADS ACCUMULATE INTO ONE GROUP SLOT, so the range may NOT be
        // cut: `compute::atomic_add` is a plain read-modify-write on the host arm
        // (`seam/seam_host.h`), and a chunked Disjoint pass loses updates.
        Scatter::Atomic,
        10.0,
        size_of::<LockCenterOfMassAccumulateRowArgs>(),
        LOCK_CENTER_OF_MASS_ACCUMULATE_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LOCK_INERTIA_ACCUMULATE_ROW,
        LOCK_INERTIA_ACCUMULATE_ROW_NAME,
        // MANY THREADS ACCUMULATE INTO ONE GROUP SLOT, so the range may NOT be
        // cut: `compute::atomic_add` is a plain read-modify-write on the host arm
        // (`seam/seam_host.h`), and a chunked Disjoint pass loses updates.
        Scatter::Atomic,
        20.0,
        size_of::<LockInertiaAccumulateRowArgs>(),
        LOCK_INERTIA_ACCUMULATE_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LOCK_ROW_SUMS_ACCUMULATE_ROW,
        LOCK_ROW_SUMS_ACCUMULATE_ROW_NAME,
        // MANY THREADS ACCUMULATE INTO ONE GROUP SLOT, so the range may NOT be
        // cut: `compute::atomic_add` is a plain read-modify-write on the host arm
        // (`seam/seam_host.h`), and a chunked Disjoint pass loses updates.
        Scatter::Atomic,
        20.0,
        size_of::<LockRowSumsAccumulateRowArgs>(),
        LOCK_ROW_SUMS_ACCUMULATE_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LOCK_REFINE_TOWARD_RHS_ROW,
        LOCK_REFINE_TOWARD_RHS_ROW_NAME,
        // One vertex rewrites its own three components.
        Scatter::Disjoint,
        30.0,
        size_of::<LockRefineTowardRhsRowArgs>(),
        LOCK_REFINE_TOWARD_RHS_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LOCK_PROJECT_OUT_ROWS_ROW,
        LOCK_PROJECT_OUT_ROWS_ROW_NAME,
        // One vertex rewrites its own three components.
        Scatter::Disjoint,
        30.0,
        size_of::<LockProjectOutRowsRowArgs>(),
        LOCK_PROJECT_OUT_ROWS_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LOCK_SEED_FREE_SOLUTION_ROW,
        LOCK_SEED_FREE_SOLUTION_ROW_NAME,
        // One vertex writes its own three components.
        Scatter::Disjoint,
        40.0,
        size_of::<LockSeedFreeSolutionRowArgs>(),
        LOCK_SEED_FREE_SOLUTION_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated_diag(
        id::LOCK_TORQUE_ACCUMULATE_ROW,
        LOCK_TORQUE_ACCUMULATE_ROW_NAME,
        // MANY THREADS ACCUMULATE INTO ONE GROUP SLOT, so the range may NOT be
        // cut: `compute::atomic_add` is a plain read-modify-write on the host arm
        // (`seam/seam_host.h`), and a chunked Disjoint pass loses updates.
        Scatter::Atomic,
        20.0,
        size_of::<LockTorqueAccumulateRowArgs>(),
        LOCK_TORQUE_ACCUMULATE_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::LOCK_CONSTRAINT_ASSEMBLE_ROW,
        LOCK_CONSTRAINT_ASSEMBLE_ROW_NAME,
        // MANY THREADS ACCUMULATE INTO ONE GROUP SLOT, so the range may NOT be
        // cut: `compute::atomic_add` is a plain read-modify-write on the host arm
        // (`seam/seam_host.h`), and a chunked Disjoint pass loses updates.
        Scatter::Atomic,
        60.0,
        size_of::<LockConstraintAssembleRowArgs>(),
        LOCK_CONSTRAINT_ASSEMBLE_ROW_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_STRAIN_FORCE_HESSIAN_GATED,
        ROD_STRAIN_FORCE_HESSIAN_GATED_NAME,
        Scatter::Disjoint,
        30.0,
        size_of::<RodStrainForceHessianGatedArgs>(),
        ROD_STRAIN_FORCE_HESSIAN_GATED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_STRAIN_STIFFNESS_GATED,
        ROD_STRAIN_STIFFNESS_GATED_NAME,
        Scatter::Disjoint,
        30.0,
        size_of::<RodStrainStiffnessGatedArgs>(),
        ROD_STRAIN_STIFFNESS_GATED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_STRAIN_DIFF_TABLE_FROM_RECORDS,
        SHELL_STRAIN_DIFF_TABLE_FROM_RECORDS_NAME,
        Scatter::Disjoint,
        10.0,
        size_of::<ShellStrainDiffTableFromRecordsArgs>(),
        SHELL_STRAIN_DIFF_TABLE_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_STRAIN_DIFF_TABLE_GATED,
        SHELL_STRAIN_DIFF_TABLE_GATED_NAME,
        Scatter::Disjoint,
        10.0,
        size_of::<ShellStrainDiffTableGatedArgs>(),
        SHELL_STRAIN_DIFF_TABLE_GATED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_STRAIN_STIFFNESS_FROM_RECORDS,
        SHELL_STRAIN_STIFFNESS_FROM_RECORDS_NAME,
        Scatter::Disjoint,
        60.0,
        size_of::<ShellStrainStiffnessFromRecordsArgs>(),
        SHELL_STRAIN_STIFFNESS_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_STRAIN_STIFFNESS_GATED,
        SHELL_STRAIN_STIFFNESS_GATED_NAME,
        Scatter::Disjoint,
        60.0,
        size_of::<ShellStrainStiffnessGatedArgs>(),
        SHELL_STRAIN_STIFFNESS_GATED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_STRAIN_EMBED,
        SHELL_STRAIN_EMBED_NAME,
        // ATOMIC: the force scatter and the CSR push both accumulate into slots
        // several faces share, so the host arm runs one ascending pass, which
        // is the order the separate scatter and push passes ran in.
        Scatter::Atomic,
        140.0,
        size_of::<ShellStrainEmbedArgs>(),
        SHELL_STRAIN_EMBED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_MAX_STRAIN,
        SHELL_MAX_STRAIN_NAME,
        Scatter::Disjoint,
        120.0,
        size_of::<ShellMaxStrainArgs>(),
        SHELL_MAX_STRAIN_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_STRAIN_VALUE,
        ROD_STRAIN_VALUE_NAME,
        Scatter::Disjoint,
        2.0,
        size_of::<RodStrainValueArgs>(),
        ROD_STRAIN_VALUE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_STRAIN_TOI_FROM_RECORDS,
        SHELL_STRAIN_TOI_FROM_RECORDS_NAME,
        Scatter::Disjoint,
        60.0,
        size_of::<ShellStrainToiFromRecordsArgs>(),
        SHELL_STRAIN_TOI_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_STRAIN_TOI_GATED,
        SHELL_STRAIN_TOI_GATED_NAME,
        Scatter::Disjoint,
        60.0,
        size_of::<ShellStrainToiGatedArgs>(),
        SHELL_STRAIN_TOI_GATED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_STRAIN_TOI_GATED,
        ROD_STRAIN_TOI_GATED_NAME,
        Scatter::Disjoint,
        20.0,
        size_of::<RodStrainToiGatedArgs>(),
        ROD_STRAIN_TOI_GATED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::COLLISION_WINDOW_VERTEX,
        COLLISION_WINDOW_VERTEX_NAME,
        // SERIAL BY CONTRACT, for the reason the arity-4 scatter states.
        Scatter::Disjoint,
        2.0,
        size_of::<CollisionWindowVertexArgs>(),
        COLLISION_WINDOW_VERTEX_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::COLLISION_WINDOW_FACE,
        COLLISION_WINDOW_FACE_NAME,
        // SERIAL BY CONTRACT, for the reason the arity-4 scatter states.
        Scatter::Disjoint,
        2.0,
        size_of::<CollisionWindowFaceArgs>(),
        COLLISION_WINDOW_FACE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::COLLISION_WINDOW_EDGE,
        COLLISION_WINDOW_EDGE_NAME,
        // SERIAL BY CONTRACT, for the reason the arity-4 scatter states.
        Scatter::Disjoint,
        2.0,
        size_of::<CollisionWindowEdgeArgs>(),
        COLLISION_WINDOW_EDGE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_CONVERT_FORCE,
        FACE_CONVERT_FORCE_NAME,
        Scatter::Disjoint,
        6.0,
        size_of::<FaceConvertForceArgs>(),
        FACE_CONVERT_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_CONVERT_HESSIAN,
        FACE_CONVERT_HESSIAN_NAME,
        Scatter::Disjoint,
        60.0,
        size_of::<FaceConvertHessianArgs>(),
        FACE_CONVERT_HESSIAN_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_DAMPING,
        FACE_DAMPING_NAME,
        Scatter::Disjoint,
        40.0,
        size_of::<FaceDampingArgs>(),
        FACE_DAMPING_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_DEFORMATION_GRADIENT,
        FACE_DEFORMATION_GRADIENT_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<FaceDeformationGradientArgs>(),
        FACE_DEFORMATION_GRADIENT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_STRETCH_TERMS,
        SHELL_STRETCH_TERMS_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<ShellStretchTermsArgs>(),
        SHELL_STRETCH_TERMS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_ATOMIC_EMBED_HESSIAN_SLOTS,
        FACE_ATOMIC_EMBED_HESSIAN_SLOTS_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<FaceAtomicEmbedHessianSlotsArgs>(),
        FACE_ATOMIC_EMBED_HESSIAN_SLOTS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_ATOMIC_EMBED_FORCE,
        FACE_ATOMIC_EMBED_FORCE_NAME,
        // SERIAL BY CONTRACT, for the reason the arity-4 scatter states.
        Scatter::Atomic,
        3.0,
        size_of::<FaceAtomicEmbedForceArgs>(),
        FACE_ATOMIC_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_LIVE_EMBED_FORCE,
        FACE_LIVE_EMBED_FORCE_NAME,
        // ATOMIC for the reason both other forms are: two faces sharing a
        // vertex land on one destination, and the row is what keeps the host
        // arm's range one ascending pass, which is what makes the gated form
        // reach the embeds in the compacted run's own order.
        Scatter::Atomic,
        60.0,
        size_of::<FaceLiveEmbedForceArgs>(),
        FACE_LIVE_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::FACE_ACTIVE_EMBED_FORCE,
        FACE_ACTIVE_EMBED_FORCE_NAME,
        // ATOMIC for the same reason the compacted form is: two elements
        // sharing a vertex land on one destination.
        Scatter::Atomic,
        60.0,
        size_of::<FaceActiveEmbedForceArgs>(),
        FACE_ACTIVE_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::HINGE_DAMPING,
        HINGE_DAMPING_NAME,
        Scatter::Disjoint,
        60.0,
        size_of::<HingeDampingArgs>(),
        HINGE_DAMPING_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::HINGE_ATOMIC_EMBED_FORCE,
        HINGE_ATOMIC_EMBED_FORCE_NAME,
        // SERIAL BY CONTRACT. The body is `compute::atomic_add` into four
        // vertices, which the host seam spells as a plain read, add and write
        // back, and two elements sharing a vertex have the same destination.
        Scatter::Atomic,
        4.0,
        size_of::<HingeAtomicEmbedForceArgs>(),
        HINGE_ATOMIC_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::HINGE_LIVE_EMBED_FORCE,
        HINGE_LIVE_EMBED_FORCE_NAME,
        // ATOMIC for the reason both other forms are: two hinges sharing a
        // vertex land on one destination, and the host seam's float add is a
        // plain read, add and write back, so the row is what keeps the range one
        // ascending pass. That is also what makes the gated form reach the
        // embeds in the compacted run's own order.
        Scatter::Atomic,
        60.0,
        size_of::<HingeLiveEmbedForceArgs>(),
        HINGE_LIVE_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::HINGE_ACTIVE_EMBED_FORCE,
        HINGE_ACTIVE_EMBED_FORCE_NAME,
        // ATOMIC for the same reason the compacted form is: two hinges sharing
        // a vertex land on one destination, and the host seam's float add is a
        // plain read, add and write back, so the row is what keeps the range
        // one ascending pass.
        Scatter::Atomic,
        60.0,
        size_of::<HingeActiveEmbedForceArgs>(),
        HINGE_ACTIVE_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::HINGE_ATOMIC_EMBED_HESSIAN_SLOTS,
        HINGE_ATOMIC_EMBED_HESSIAN_SLOTS_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<HingeAtomicEmbedHessianSlotsArgs>(),
        HINGE_ATOMIC_EMBED_HESSIAN_SLOTS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_BEND_DAMPING,
        ROD_BEND_DAMPING_NAME,
        Scatter::Disjoint,
        45.0,
        size_of::<RodBendDampingArgs>(),
        ROD_BEND_DAMPING_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_DAMPING,
        ROD_DAMPING_NAME,
        Scatter::Disjoint,
        20.0,
        size_of::<RodDampingArgs>(),
        ROD_DAMPING_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_ATOMIC_EMBED_FORCE,
        ROD_ATOMIC_EMBED_FORCE_NAME,
        // SERIAL BY CONTRACT, for the reason the arity-4 scatter states.
        Scatter::Atomic,
        2.0,
        size_of::<RodAtomicEmbedForceArgs>(),
        ROD_ATOMIC_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_LIVE_EMBED_FORCE,
        ROD_LIVE_EMBED_FORCE_NAME,
        // ATOMIC, as the active form beside it: two rods sharing a node fold
        // into the same three slots, so the host arm runs one ascending pass.
        Scatter::Atomic,
        6.0,
        size_of::<RodLiveEmbedForceArgs>(),
        ROD_LIVE_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_ACTIVE_EMBED_FORCE,
        ROD_ACTIVE_EMBED_FORCE_NAME,
        // ATOMIC for the same reason the compacted form is: two elements
        // sharing a vertex land on one destination.
        Scatter::Atomic,
        60.0,
        size_of::<RodActiveEmbedForceArgs>(),
        ROD_ACTIVE_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_PACKED_EMBED_FORCE,
        ROD_PACKED_EMBED_FORCE_NAME,
        // ATOMIC as its siblings are: two rods sharing a vertex overlap.
        Scatter::Atomic,
        60.0,
        size_of::<RodPackedEmbedForceArgs>(),
        ROD_PACKED_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::ROD_ATOMIC_EMBED_HESSIAN_SLOTS,
        ROD_ATOMIC_EMBED_HESSIAN_SLOTS_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<RodAtomicEmbedHessianSlotsArgs>(),
        ROD_ATOMIC_EMBED_HESSIAN_SLOTS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::STITCH_ATOMIC_EMBED_FORCE,
        STITCH_ATOMIC_EMBED_FORCE_NAME,
        // SERIAL BY CONTRACT, for the reason the arity-4 scatter states: two
        // stitches sharing a vertex have the same destination.
        Scatter::Atomic,
        6.0,
        size_of::<StitchAtomicEmbedForceArgs>(),
        STITCH_ATOMIC_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SVD3X2,
        SVD3X2_NAME,
        Scatter::Disjoint,
        120.0,
        size_of::<Svd3x2Args>(),
        SVD3X2_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SVD3X2_SHIFTED,
        SVD3X2_SHIFTED_NAME,
        Scatter::Disjoint,
        120.0,
        size_of::<Svd3x2ShiftedArgs>(),
        SVD3X2_SHIFTED_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SHELL_STRAIN_RESTORE_SIGMA,
        SHELL_STRAIN_RESTORE_SIGMA_NAME,
        Scatter::Disjoint,
        1.0,
        size_of::<ShellStrainRestoreSigmaArgs>(),
        SHELL_STRAIN_RESTORE_SIGMA_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SVD3X3_RV,
        SVD3X3_RV_NAME,
        Scatter::Disjoint,
        180.0,
        size_of::<Svd3x3RvArgs>(),
        SVD3X3_RV_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::SVD3X3,
        SVD3X3_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<Svd3x3Args>(),
        SVD3X3_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::TET_CONVERT_FORCE,
        TET_CONVERT_FORCE_NAME,
        Scatter::Disjoint,
        8.0,
        size_of::<TetConvertForceArgs>(),
        TET_CONVERT_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::TET_CONVERT_HESSIAN,
        TET_CONVERT_HESSIAN_NAME,
        Scatter::Disjoint,
        120.0,
        size_of::<TetConvertHessianArgs>(),
        TET_CONVERT_HESSIAN_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::TET_SHAPE_GRADIENTS,
        TET_SHAPE_GRADIENTS_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<TetShapeGradientsArgs>(),
        TET_SHAPE_GRADIENTS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::TET_DEFORMATION_GRADIENT,
        TET_DEFORMATION_GRADIENT_NAME,
        Scatter::Disjoint,
        12.0,
        size_of::<TetDeformationGradientArgs>(),
        TET_DEFORMATION_GRADIENT_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::TET_DAMPING,
        TET_DAMPING_NAME,
        Scatter::Disjoint,
        60.0,
        size_of::<TetDampingArgs>(),
        TET_DAMPING_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VERTEX_NORMAL_FINALIZE,
        VERTEX_NORMAL_FINALIZE_NAME,
        // CONSERVATIVE, and deliberately not a measurement. `Atomic`
        // never lets the host arm cut a range it must not cut, and
        // the nanosecond figure only sets that arm's chunk width. A
        // kernel this driver does not dispatch has neither measured.
        Scatter::Atomic,
        1.0,
        size_of::<VertexNormalFinalizeArgs>(),
        VERTEX_NORMAL_FINALIZE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VERTEX_ATOMIC_EMBED_FORCE,
        VERTEX_ATOMIC_EMBED_FORCE_NAME,
        // SERIAL BY CONTRACT. The body is an atomic embed, which the host seam
        // spells as a plain read, add and write back. The declaration cannot
        // carry this: an entry point covers the range it is handed, and this
        // row is what decides that the range is never cut.
        Scatter::Atomic,
        2.0,
        size_of::<VertexAtomicEmbedForceArgs>(),
        VERTEX_ATOMIC_EMBED_FORCE_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VERTEX_FIX_INDEX_FROM_RECORDS,
        VERTEX_FIX_INDEX_FROM_RECORDS_NAME,
        // SERIAL BY CONTRACT. The body is an atomic embed, which the host seam
        // spells as a plain read, add and write back. The declaration cannot
        // carry this: an entry point covers the range it is handed, and this
        // row is what decides that the range is never cut.
        Scatter::Disjoint,
        2.0,
        size_of::<VertexFixIndexFromRecordsArgs>(),
        VERTEX_FIX_INDEX_FROM_RECORDS_HOST_REF_OFFSETS,
    ),
    decl_generated(
        id::VERTEX_DOF_REMOVAL_MASK,
        VERTEX_DOF_REMOVAL_MASK_NAME,
        // SERIAL BY CONTRACT. The body is an atomic embed, which the host seam
        // spells as a plain read, add and write back. The declaration cannot
        // carry this: an entry point covers the range it is handed, and this
        // row is what decides that the range is never cut.
        Scatter::Disjoint,
        2.0,
        size_of::<VertexDofRemovalMaskArgs>(),
        VERTEX_DOF_REMOVAL_MASK_HOST_REF_OFFSETS,
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_indexed_by_its_own_ids() {
        // The backend indexes the table and the launch array by `KernelId`, so
        // a row whose id is not its own index would dispatch a different
        // kernel with the right argument bytes, which is a silent wrong answer
        // rather than a crash.
        for (index, row) in TABLE.iter().enumerate() {
            assert_eq!(
                row.id.0 as usize, index,
                "kernel {} sits at index {index} and claims id {}",
                row.name, row.id.0
            );
        }
    }

    #[test]
    fn every_declared_argument_size_is_the_records_own() {
        // A declaration that disagreed with `size_of` would make every dispatch
        // of that kernel fail the length check, which is loud; the assertion is
        // here so the failure names the record rather than a call site.
        assert_eq!(
            TABLE[id::OPERATOR_APPLY.0 as usize].args_bytes as usize,
            size_of::<OperatorApplyArgs>()
        );
        assert_eq!(
            TABLE[id::VEC_COMBINE.0 as usize].args_bytes as usize,
            size_of::<VecCombineArgs>()
        );
    }

    #[test]
    fn every_host_reference_offset_is_inside_its_record() {
        for row in TABLE.iter() {
            for &offset in row.host_refs {
                assert!(
                    offset as usize + size_of::<HostRef>() <= row.args_bytes as usize,
                    "{}: a host reference at {offset} does not fit in {} bytes",
                    row.name,
                    row.args_bytes
                );
                // FOUR, NOT EIGHT, and the difference is the whole reason
                // `HostRef` is `#[repr(C, packed(4))]`. A generated record has
                // no padding by construction, so a scalar may sit before a
                // buffer reference and put it at a 4-byte offset:
                // `face_spectral_hessian` has `eps` ahead of its scatter
                // buffer and lands one at 84. Four is the alignment the seam's
                // reference is DEFINED to have (it is four `u32`s), and
                // `ppf_cts_compute::host::bind_generated` reads and writes each
                // field with `read_unaligned` / `write_unaligned` for that
                // reason. Requiring eight here would be requiring the padding
                // the record is built not to have.
                assert_eq!(
                    offset as usize % 4,
                    0,
                    "{}: a host reference at {offset} is misaligned",
                    row.name
                );
            }
        }
    }

    /// AN INDIRECT VECTOR UPDATE CARRIES ITS DIRECT TWIN'S ROW, because it is
    /// the same body with one coefficient read from a buffer instead of from
    /// the record.
    ///
    /// **WHAT THIS EXISTS TO CATCH, MEASURED RATHER THAN IMAGINED.**
    /// `VEC_COMBINE_INDIRECT` carried `Scatter::Atomic` and 1.0 ns while
    /// nothing dispatched it, which was a placeholder that could not be wrong
    /// about a kernel nobody launched. `Atomic` means ONE SERIAL ASCENDING
    /// PASS on this backend (`host.rs:786`), so the first caller ran a
    /// `3 * vertices` vector update single-threaded beside chunked
    /// neighbors doing the same shape of work. Nothing failed and no gate
    /// moved: it is a wall-clock defect with a correct answer.
    ///
    /// A scatter is a property of the BODY, and these two bodies write
    /// `destination[index]` and nothing else, exactly as their twins do. So
    /// the rows must agree, and a future edit that moves one has to move both
    /// or say here why they differ.
    #[test]
    fn an_indirect_vector_update_carries_its_direct_twins_row() {
        for (indirect, direct) in [
            (id::VEC_COMBINE_INDIRECT, id::VEC_COMBINE),
            (id::VEC_ADD_SCALED_INDIRECT, id::VEC_ADD_SCALED),
        ] {
            let a = &TABLE[indirect.0 as usize];
            let b = &TABLE[direct.0 as usize];
            assert!(
                matches!(a.scatter, Scatter::Disjoint),
                "{} must be Disjoint: it writes its own element and nothing \
                 else, and Atomic would run the whole vector serially",
                a.name
            );
            assert_eq!(
                format!("{:?}", a.scatter),
                format!("{:?}", b.scatter),
                "{} and {} are one body with one argument moved, so their \
                 scatters must agree",
                a.name,
                b.name
            );
            assert_eq!(
                a.nanos_per_item, b.nanos_per_item,
                "{} and {} are one body with one argument moved, so their \
                 per-item costs must agree",
                a.name, b.name
            );
        }
    }

    /// NO TWO KERNEL IDS NAME THE SAME NUMBER, and an id past [`id::COUNT`]
    /// really is past it.
    ///
    /// `Device::decl` is `table.get(id)`, so an id inside the table resolves to
    /// whatever row sits at that index. That is exactly what the "deliberately
    /// past COUNT" ids promise CANNOT happen to them: their doc says a dispatch
    /// answers `Fault::MissingKernel` by name. Six of them sat at 108 to 113,
    /// which the six AABB query ids already held, so a dispatch would have
    /// launched an unrelated kernel with a mis-typed record instead. Nothing
    /// dispatched them, so nothing failed; `check_shape` would have caught it
    /// only where the two records happen to differ in size, which is a
    /// coincidence and not a guarantee.
    ///
    /// This reads the source rather than the constants because the defect is
    /// two names sharing a NUMBER, and a test written against the constants
    /// would compare each to itself.
    #[test]
    fn no_two_kernel_ids_name_the_same_number() {
        let source = include_str!("kernels.rs");
        let mut seen: std::collections::HashMap<u32, &str> =
            std::collections::HashMap::new();
        let mut past = Vec::new();
        for line in source.lines() {
            let Some(rest) = line.trim().strip_prefix("pub const ") else {
                continue;
            };
            let Some((name, tail)) = rest.split_once(": KernelId = KernelId(") else {
                continue;
            };
            let Some((number, _)) = tail.split_once(')') else {
                continue;
            };
            let number: u32 = number.parse().expect("a kernel id is a number");
            if let Some(previous) = seen.insert(number, name) {
                panic!(
                    "kernel id {number} is claimed by both {previous} and {name}; \
                     `Device::decl` indexes the table by this number, so the \
                     second would dispatch the first's kernel"
                );
            }
            if number as usize >= id::COUNT {
                past.push((name, number));
            }
        }
        // The count is DERIVED from the same source, so this cannot drift.
        assert_eq!(
            seen.len(),
            id::COUNT + past.len(),
            "every id below COUNT has a row and every id past it is documented \
             as undispatched; found {} ids for COUNT {} and {} past it",
            seen.len(),
            id::COUNT,
            past.len()
        );
        for (name, number) in &past {
            assert!(
                TABLE.get(*number as usize).is_none(),
                "{name} is past COUNT and must have no row, but the table has \
                 one at {number}"
            );
        }
    }

    #[test]
    fn the_operator_record_names_every_buffer_it_carries() {
        // A record that named one fewer would pass every other check here and
        // leave a field unvalidated, which is the half-wired case the walk
        // exists for.
        //
        // BOTH FORMS ARE PINNED, and the DIFFERENCE between them is the check
        // that matters: the operator comes as a pair, and the seven the dynamic
        // form adds are exactly the contact matrix's, which all arrive from one
        // `Option`. A dynamic record that named six would be the partial wiring
        // its own doc says cannot be mistaken for a matrix.
        // BOTH FORMS ARE NOW ENTIRELY HANDLE-ADDRESSED, so `host_refs` can no
        // longer carry this check: it reads zero for both, and a difference of
        // zero would pass while saying nothing. The fixed matrix's five arrays
        // went to the device with the CSR migration and the contact matrix's
        // seven went with it, joining `x` and `result`, which had gone earlier
        // with the PCG workspace.
        //
        // THE INVARIANT IS UNCHANGED AND IS NOW MEASURED IN THE RECORD'S SIZE.
        // The dynamic form carries exactly seven more buffers than the plain
        // one, they are the contact matrix's own, and they all arrive from one
        // `Option`. A dynamic record that named six would be the partial wiring
        // its own doc says cannot be mistaken for a matrix, and it would show
        // up here as a record one handle too small.
        let plain_refs = TABLE[id::OPERATOR_APPLY.0 as usize].host_refs.len();
        let dynamic_refs = TABLE[id::OPERATOR_APPLY_DYNAMIC.0 as usize].host_refs.len();
        assert_eq!(plain_refs, 0, "the form with no dynamic matrix is fully resident");
        assert_eq!(dynamic_refs, 0, "the form that carries one is fully resident");

        let plain = TABLE[id::OPERATOR_APPLY.0 as usize].args_bytes as usize;
        let dynamic = TABLE[id::OPERATOR_APPLY_DYNAMIC.0 as usize].args_bytes as usize;
        let handle = std::mem::size_of::<ppf_cts_compute::Handle>();
        assert!(dynamic > plain, "the dynamic form must carry more, not less");
        assert_eq!(
            dynamic - plain,
            7 * handle,
            "the contact matrix's own buffers: seven handles, {handle} bytes each"
        );
        // THE ABSOLUTE SIZES ARE PINNED TOO, and the difference alone is not
        // enough: a buffer dropped from BOTH records leaves the difference
        // intact, which is the half-wired case that would otherwise walk past
        // every assertion above. `size_of` is exact for a `repr(C)` record, so
        // these are the two records' real widths rather than a lower bound.
        assert_eq!(
            plain,
            std::mem::size_of::<OperatorApplyArgs>(),
            "the plain operator record's width"
        );
        assert_eq!(
            dynamic,
            std::mem::size_of::<OperatorApplyDynamicArgs>(),
            "the dynamic operator record's width"
        );
    }
}
