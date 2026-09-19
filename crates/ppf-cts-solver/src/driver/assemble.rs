// File: crates/ppf-cts-solver/src/driver/assemble.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Assembling one Newton system: the momentum layer, the five elastic layers,
//! and the two strain limiters.
//!
//! It also carries the `max_sigma` stretch indicator, which is not part of any
//! system: it is an element walk measured once per step at the start-of-step
//! pose, and it lives here because that is what this module owns.
//!
//! # What this module owns and what it must not
//!
//! It owns the WALK: which elements are in this iteration, in what order, and
//! where their results are scattered. Every value comes from a shared body, and
//! every one of them is reached by dispatching a named kernel through
//! [`Device`]: this module names no entry point except the two `ppf_*_abi`
//! scalar queries below, which have no thread index and so no dispatch to be.
//! There is no float arithmetic here, including the per-element mass scale,
//! which goes through `vec_add_scaled` for exactly that reason.
//!
//! # The order is part of the answer
//!
//! `force` and the two Hessians are fp32 running sums, so re-ordering the
//! contributions changes the assembled system in its last bits. The order is
//! momentum, then elastic, then stitch, then the SHELL strain limit and the ROD
//! strain limit, then contact, and it is fixed rather than incidental. Inside
//! the elastic layer the five dispatches run as ROD BEND, ROD, SHELL FACE, TET
//! and HINGE, so the two rod terms come first, the membrane is assembled before
//! the tets, and the shell hinges last.
//!
//! THE TWO STRAIN LIMITERS SIT ON THE FAR SIDE OF THE `tmp_fixed` SNAPSHOT, and
//! that is not an ordering preference: each scales its blocks by a stiffness
//! that CONTRACTS that snapshot, so it has to run where the snapshot still
//! holds the elastic-only matrix, and its own blocks have to land in the live
//! one the solve reads.
//!
//! # Two scatters, both serial, and the reason is not tidiness
//!
//! `hinge_atomic_embed_force` and `FixedCsr::push_blocks` reach
//! `compute::atomic_add`, which `seam/seam_host.h` spells as a plain read,
//! add and write back on the premise that one thread runs a shared body. Two
//! tets sharing a vertex have the same force destination, two sharing an edge
//! have the same CSR slot, two hinges over one triangle strip share three of
//! their four vertices, and two rod bending sites one vertex apart share two of
//! their three nodes, so a parallel scatter is a DATA RACE rather than a
//! different fold order. The split is therefore taken in its simplest form:
//! the per-element evaluation is a range that could be parallel, and the
//! scatter is one serial pass in ascending element index. That order is also
//! what makes this backend's answer independent of the thread count.
//!
//! **THE SERIAL PASS IS THE KERNEL TABLE'S DECLARATION, NOT A LOOP HERE.**
//! Each force scatter is one `Scatter::Atomic` row in `super::kernels`, which is
//! what stops the backend cutting its range, and this module hands it one
//! dispatch covering the whole pass. A dispatch covers a contiguous range and an
//! active list is a subset of one, so the active elements are gathered into an
//! ascending run first and the dispatch walks that run: the fold reaches the
//! elements in ascending element index either way, which is what the fp32
//! running sum in `force` depends on. It is the shape `super::contact` and
//! `super::collider` scatter through as well. The one layer whose EVALUATION is
//! over a subset too is the rod stretch, for a reason stated on `rod_stretch`
//! itself, and there each active rod is a one-element dispatch inside a single
//! region.
//!
//! **THE SHELL MEMBRANE IS ONE DISPATCH THAT EVALUATES AND SCATTERS, and is
//! where the split above does NOT apply.** The shell face body holds a
//! triangle's whole elastic chain in one thread and embeds the result itself,
//! so [`shell_membrane`] dispatches one `Scatter::Atomic` kernel over the shell
//! prefix rather than staging the arithmetic and folding afterwards. The split
//! buys nothing there: the evaluation could be cut, but the scatter it feeds
//! cannot, and separating them costs fifteen dispatches, thirteen staging
//! arrays and a host transpose that a fused pass needs none of. The four other
//! elastic layers still stage; fusing the solid one has to preserve
//! SPD-by-assembly, which is the constraint any attempt is judged against.

use crate::data::{
    DataSet, EdgeParam, EdgeProp, FaceParam, FaceProp, HingeParam, HingeProp, ParamSet,
    TetParam, TetProp, VertexParam, VertexProp,
};

use ppf_cts_compute::{Device, EncoderExt, Fault};
use super::fixedcsr::FixedCsr;
use super::kernels::{VecFillArgs,
    ElementAddScaledArgs,             FaceElasticEmbedFromRecordsArgs, TorqueGroupFrameArgs,
    FaceDeformationGradientArgs, HingeLiveEmbedForceArgs, 
    
    MomentumEmbedArgs,
    ShellBendEmbedArgs,
    RodLiveEmbedForceArgs, 
    RodStrainForceHessianGatedArgs, RodStrainStiffnessGatedArgs, RodStrainToiGatedArgs,
    
    RodStretchRatioGatedArgs,
        RodBendEmbedArgs,
        TetElasticEmbedArgs, TetMaterialFromRecordsArgs,
        RodStretchEmbedArgs,
        ShellStrainEmbedArgs,
        ShellStrainToiFromRecordsArgs,
    ShellStretchTermsArgs, StitchAtomicEmbedForceArgs, StitchForceHessianGatheredArgs, Svd3x2Args,
    Svd3x3RvArgs,
    TetConvertForceArgs, TetSpectralConvertHessianArgs, TetDampingArgs, TetDeformationGradientArgs, TetMaterialDiffTableArgs,
    TetSpectralForceArgs,
};
// THE NAMES ONLY THE TEST MODULE BELOW REACHES. It opens with `use super::*`,
// so an import here is what puts them in its scope; keeping them out of the
// lists above is what keeps the release build free of an unused import.
#[cfg(test)]
use crate::data::Model;
#[cfg(test)]
use super::kernels::{
    FaceMaterialDiffTableArgs, RodStrainValueArgs, ShellBendArealDensityFromRecordsArgs,
    ShellBendStiffnessAndDampingArgs, ShellMaxStrainArgs,
};
use super::scene::{self, Fatal, FatalResult};
use super::state::SolverState;


// THE TWO SHIMS THIS MODULE STILL NAMES DIRECTLY, and they are named together
// here so the exception is visible rather than buried.
//
// NEITHER IS A DISPATCH, and that is the reason rather than an excuse. Each is
// a scalar function of scalars: it has no thread index, no extent, no argument
// record and no buffer, so there is nothing for [`Device`] to carry and no
// launch geometry a call site could state. Both are read inside a host loop
// that is gathering material for a stage that IS dispatched, so wrapping one in
// a range shim would mean allocating an array per input, dispatching over it,
// and reading the answers back, which is a second walk over the same elements
// to compute what the walk already had in hand.
//
// The precedents are `super::pcg`'s per-block inverse and `super::step`'s
// domain query, and the shape of the exception is the same in all three:
// a value the shared bodies define, read one at a time by the driver, with no
// range for the seam to describe. `ppf_cts_compute` grows a QUERY the day one of
// them needs to be answered by a backend rather than by a shared body compiled
// into this process, and none of the three does today.
extern "C" {
    /// The mean of one quantity over an interior rod vertex's two segments.
    ///
    /// Read once per bending site while the material is gathered: each site
    /// names two edges through an adjacency, so a range would have to carry the
    /// gather it cannot see.
    ///
    /// No PRODUCTION caller: the assembly dispatch forms the average inside the
    /// bending kernel, in the site's own thread. What reads this is the oracle
    /// `RodBend::stiffness` in this file's test module, which rebuilds the
    /// site's stiffness scalar from the shared bodies rather than from a
    /// production buffer, for
    /// `the_rod_bending_force_is_the_gradient_of_the_turning_angle_energy`,
    /// `the_rod_bending_hessian_is_the_angle_gradient_outer_product_at_the_rest_angle`
    /// and
    /// `the_bending_stiffness_halves_when_the_segment_length_and_lumped_mass_double`.
    #[allow(dead_code)]
    fn rod_bend_segment_average_abi(first: f32, second: f32) -> f32;
}

/// A0 of the assembly: inertia, the aerodynamic term, the pull springs, the
/// isotropic drag and the `fix-xz` drag, per free vertex.
///
/// ONE DISPATCH OVER THE WHOLE VERTEX RANGE. The body reads a vertex's incident
/// faces through the neighbor table and writes only that vertex's force and
/// diagonal block, so nothing scatters: the kernel's row declares
/// `Scatter::Disjoint`, and whether the range is cut and where is the backend's
/// decision to make off that declaration.
///
/// # Safety
/// `data` and `param` must be live, and `state`'s buffers sized for the scene.
pub unsafe fn momentum<D: Device>(
    device: &mut D,
    data: &DataSet,
    param: *const ParamSet,
    state: &mut SolverState,
    dt: f32,
) -> FatalResult<()> {
    let vertices = state.sizes.vertices;
    if vertices == 0 {
        return Ok(());
    }
    // THE TORQUE FRAME IS A PRE-PASS, AND IT RUNS EVERY ASSEMBLY. Each group's
    // centroid, principal axis and radius normalization are functions of the
    // ITERATE, which moves under the Newton loop, so a frame computed once per
    // step would scale this iteration's force by last iteration's geometry.
    //
    // IT CANNOT BE FOLDED INTO THE ROW BELOW: a frame is a property of the
    // whole group and no single member can reach it, which is what makes the
    // two dispatches two rather than one.
    let torque_groups = state.torque_group.len();
    if torque_groups > 0 {
        let frame_args = TorqueGroupFrameArgs {
            group: state.torque_group.handle(),
            member: state.torque_vertex.handle(),
            member_count: state.torque_vertex.len() as u32,
            position: state.eval_x.handle(),
            vertex_count: vertices as u32,
            prop: state.prop_vertex.handle(),
            result: state.torque_result.handle(),
            count: torque_groups as u32,
            seam_arena_count: 0,
        };
        device.launch("assemble.torque.frame", &frame_args, torque_groups as u32)?;
    }
    let _neighbor = &data.mesh.neighbor.vertex.face;
    // WHETHER THE SCENE HAS A VERTEX-FACE TABLE IS THE DRIVER'S TO SAY, which
    // is the same question `compute_vertex_normal` asks. It crosses as a flag
    // rather than a null pointer, because a body cannot ask whether a buffer is
    // null on a target where a buffer cannot be null; the buffers beside a
    // zero flag name nothing and the body never reads them.
    // THE VERTEX-FACE ADJACENCY IS THE STATE'S, staged once at allocate. An
    // absent table is `Handle::NONE` beside a zero flag, which is the same
    // "names nothing" the null pointer spelled, said the one way a handle can.
    let has_neighbor = u32::from(state.neighbor_vertex_face_offset.len() != 0);
    let neighbor_index = super::state::adjacency_handle(&state.neighbor_vertex_face_index);
    let neighbor_offset = super::state::adjacency_handle(&state.neighbor_vertex_face_offset);
    // Copied out before the record is built: indexing through a raw-pointer
    // dereference autorefs, which is a reference into memory this function does
    // not own.
    let wind = (*param).wind;
    let args = MomentumEmbedArgs {
        eval_x: state.eval_x.handle(),
        current: state.positions.handle(),
        target: state.target.handle(),
        prop: state.prop_vertex.handle(),
        pull: state.pull.handle(),
        pull_count: state.pull.len() as u32,
        // THE MEMBER ARRAY IS SCANNED WHOLE, as the pin array above is: a
        // vertex can be named by more than one group and every match
        // contributes, so there is no per-vertex run to hand it instead.
        torque_vertex: state.torque_vertex.handle(),
        torque_vertex_count: state.torque_vertex.len() as u32,
        torque_result: state.torque_result.handle(),
        torque_group_count: torque_groups as u32,
        neighbor_index,
        neighbor_offset,
        has_neighbor,
        face: state.mesh_face.handle(),
        dt,
        inactive_momentum: u32::from((*param).inactive_momentum),
        time_f32: (*param).time_f32,
        wind_x: wind[0],
        wind_y: wind[1],
        wind_z: wind[2],
        air_density: (*param).air_density,
        air_friction: (*param).air_friction,
        isotropic_air_friction: (*param).isotropic_air_friction,
        fix_xz: (*param).fix_xz,
        force: state.force.handle(),
        diagonal: state.diagonal.handle(),
        count: vertices as u32,
        seam_arena_count: 0,
    };
    device.launch("assemble.momentum", &args, vertices as u32)?;
    Ok(())
}

/// The tet elastic force and Hessian, staged over all active tets.
///
/// Eleven passes over the active tets. The
/// gate is `!fixed && !rest_excluded` from the dispatch plus `mu > 0` from the
/// body, and both halves matter: `fixed` and `rest_excluded` say the element
/// carries no energy this step, and `mu == 0` is how a PDRD element and a
/// zero-stiffness material both arrive.
///
/// # Safety
/// `data` must be live and `state` sized for the scene.
pub unsafe fn tet_elastic<D: Device>(
    device: &mut D,
    data: &DataSet,
    eiganalysis_eps: f32,
    state: &mut SolverState,
    fixed: &mut FixedCsr<'_>,
    dt: f32,
) -> FatalResult<()> {
    let tets = state.sizes.tets;
    if tets == 0 {
        return Ok(());
    }
    let mesh: &[crate::data::Vec4u] = scene::slice(&data.mesh.mesh.tet);
    let props: &[TetProp] = scene::slice(&data.prop.tet);
    let params: &[TetParam] = scene::slice(&data.param_arrays.tet);
    let inv_rest: &[crate::data::Mat3x3f] = scene::slice(&data.inv_rest3x3);
    if props.len() != tets || mesh.len() != tets {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene has {tets} tets, {} tet props and {} tet index records",
            props.len(),
            mesh.len()
        )));
    }
    if inv_rest.len() < tets {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene has {tets} tets and {} inverse rest matrices",
            inv_rest.len()
        )));
    }

    // THE MATERIAL IS GATHERED ON THE DEVICE, from the tet's own records: the
    // dispatch gates on `prop.tet[i]` and reads
    // `param_arrays.tet[prop.param_index]` in the same thread. Gathering the
    // material on the host instead would mean walking every tet every Newton
    // iteration, gathering five constants, building an active list and
    // uploading four arrays, which is the computation itself relocated onto the
    // host.
    //
    // WHAT STAYS ON THE HOST IS VALIDATION, and it is not a readback: `props`,
    // `params` and `mesh` are the scene's own host-resident arrays. It is also
    // not per-iteration work of any size that matters, because the values it
    // checks are fixed for the scene rather than rebuilt per step.
    for (index, prop) in props.iter().enumerate() {
        if params.get(prop.param_index as usize).is_none() {
            return Err(Fatal::device_assert(format!(
                "solver driver: tet {index} names material {} but the scene carries {} tet \
                 materials",
                prop.param_index,
                params.len()
            )));
        }
        for k in 0..4 {
            if mesh[index][k] as usize >= state.sizes.vertices {
                return Err(Fatal::device_assert(format!(
                    "solver driver: tet {index} names vertex {} but the scene has {} vertices",
                    mesh[index][k], state.sizes.vertices
                )));
            }
        }
    }

    // THE STAGES RUN OVER THE WHOLE TET RANGE, not over the active list, and
    // the scatter at the end runs over the active list only. Ranging the
    // arithmetic over every tet keeps each call one contiguous span, which is
    // what the shim's chunk shape is for; an inactive tet's intermediates are
    // computed and then never scattered, so they cost time and change nothing.
    // An inactive tet carries the inert seed written above, so every stage is
    // well defined on it.
    let count = tets as u32;
    let inv_rest = state.inv_rest3x3.handle();

    // THE MATERIAL PASS, which replaces those four uploads and the host gather
    // that filled them. It writes the inert seed for every gated-out tet, so
    // the stages below stay total over the whole range.
    let material_args = TetMaterialFromRecordsArgs {
        prop: state.prop_tet.handle(),
        tet_param: state.param_tet.handle(),
        model: state.tet.model.handle(),
        mu: state.tet.mu.handle(),
        lambda: state.tet.lambda.handle(),
        mass: state.tet.mass.handle(),
        damping: state.tet.damping.handle(),
        count,
        seam_arena_count: 0,
    };
    // Safety: every array is borrowed for the whole call.
    unsafe { device.launch("assemble.tet.material", &material_args, count) }?;

    // ONE KERNEL. It computes a tet's force and Hessian and pushes both from
    // inside the element's own thread; the staged path below writes a 12x12
    // through device memory and pushes it in a second pass, which is eight
    // dispatches instead of one.
    //
    // THE CONSTRAINT A FUSED TET LAYER IS JUDGED AGAINST IS SPD-BY-ASSEMBLY.
    // Every block that can be indefinite is PSD-projected in the thread that
    // forms it, and a fusion that loses that projection produces an indefinite
    // matrix the linear solve reports as a `pAp<=0` breakdown rather than as a
    // wrong number. `examples/cards` is the scene that exercises it and
    // completes with `shape=pass` on this path.
    //
    // `examples/plastic` FAILS ABOUT A QUARTER OF ITS RUNS, AND SO DOES THE
    // STAGED PATH: **thirty-six PAIRED runs across two boxes say 25 of 36
    // against 27 of 36**, which is no difference at all. That scene does not
    // always complete, and this measurement is that property rather than a
    // defect in either path. Three runs a side is too few to see it, and
    // suggested the opposite. The staged chain stays reachable under
    // `PPF_TET_SPLIT=1`, which is the lever that answers the question again in
    // one run if a future scene disagrees.
    //
    // WHAT IT BUYS: `trapped` falls from 114.98 s to 106.48 and
    // `matrix_assembly` from 11,057 ms to 3,654.
    if std::env::var_os("PPF_TET_SPLIT").is_none() {
        let (index, offset, value, row_count) = fixed.device_push_refs();
        let embed_args = TetElasticEmbedArgs {
            x: state.eval_x.handle(),
            current: state.positions.handle(),
            tet: state.mesh_tet.handle(),
            vertex_count: state.sizes.vertices as u32,
            tet_vertex: state.mesh_tet.handle(),
            inverse_rest: inv_rest,
            model: state.tet.model.handle(),
            mu: state.tet.mu.handle(),
            lambda: state.tet.lambda.handle(),
            mass: state.tet.mass.handle(),
            deform_damping: state.tet.damping.handle(),
            dt,
            eigenanalysis_eps: eiganalysis_eps,
            force: state.force.handle(),
            index,
            offset,
            value,
            row_count,
            count,
            seam_arena_count: 0,
        };
        unsafe { device.launch("assemble.tet.elastic_embed", &embed_args, count) }?;
        return Ok(());
    }

    let deformation_args = TetDeformationGradientArgs {
        x: state.eval_x.handle(),
        tet: state.mesh_tet.handle(),
        // The bound the entry point checks the tet's four slots against, which
        // the record carries because a slot is data rather than the thread
        // index and the count guard says nothing about it.
        vertex_count: state.sizes.vertices as u32,
        inverse_rest: inv_rest,
        deformation: state.tet.deformation.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.tet.deformation_gradient", &deformation_args, count)?;
    let svd_args = Svd3x3RvArgs {
        input: state.tet.deformation.handle(),
        u: state.tet.svd_u.handle(),
        sigma: state.tet.svd_sigma.handle(),
        vt: state.tet.svd_vt.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.tet.svd3x3_rv", &svd_args, count)?;
    let table_args = TetMaterialDiffTableArgs {
        model: state.tet.model.handle(),
        sigma: state.tet.svd_sigma.handle(),
        mu: state.tet.mu.handle(),
        lambda: state.tet.lambda.handle(),
        gradient_sigma: state.tet.gradient_sigma.handle(),
        hessian_sigma: state.tet.hessian_sigma.handle(),
        accepted: state.tet.accepted.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.tet.material_diff_table", &table_args, count)?;
    // THE VERDICT IS NOT READ BACK, AND THE HOST ALREADY KNOWS IT. The body
    // returns a code saying whether it had an arm for the element's model id,
    // and that code is a pure function of `model`, which the host builds out of
    // the scene's material table. `super::refusal::material_defects` asks the
    // question once at `initialize()` and refuses the scene by name and count.
    // Asking the device instead would cost a download inside every Newton
    // iteration, which on a queued backend is a wait for the queue to drain,
    // for an answer the host settled before the first frame.
    //
    // THE SINK IS THE SHAPE OF THE ENTRY POINT. A generated entry over a
    // value-returning body must carry a `[[seam::scatter]]` destination for
    // that value, and the body returns the verdict for a reason of its own: it
    // is what stops an unrecognized id from falling through to SNHk for every
    // caller of the shared body. The tests below read each code straight off
    // the kernel, so the vocabulary the gate mirrors cannot drift from it.
    let force_args = TetSpectralForceArgs {
        gradient_sigma: state.tet.gradient_sigma.handle(),
        u: state.tet.svd_u.handle(),
        vt: state.tet.svd_vt.handle(),
        force: state.tet.gradient_f.handle(),
        count,
        seam_arena_count: 0,
    };
    let rest_ref = inv_rest;
    // THE SPECTRAL HESSIAN AND ITS CONVERSION IN ONE DISPATCH. The pair this
    // replaces wrote the 9x9 to `hessian_f` and read all 81 floats of it back
    // to build the 12x12; the composition keeps it in the thread's registers
    // and calls the same two bodies in the same order.
    let hessian_args = TetSpectralConvertHessianArgs {
        gradient_sigma: state.tet.gradient_sigma.handle(),
        hessian_sigma: state.tet.hessian_sigma.handle(),
        u: state.tet.svd_u.handle(),
        sigma: state.tet.svd_sigma.handle(),
        vt: state.tet.svd_vt.handle(),
        inverse_rest: rest_ref,
        mass: state.tet.mass.handle(),
        eigenanalysis_eps: eiganalysis_eps,
        hessian: state.tet.hessian_x.handle(),
        count,
        seam_arena_count: 0,
    };
    // The two converters write into `gradient_f` / `hessian_f`'s successors,
    // and the mass scale accumulates into a destination that opens at zero.
    // `dedx` and `d2edx2` are `Zero()` at the top of the device body, and this
    // is that zero.
    // THE SEED, on the device: this array opens at zero and the scale
    // below accumulates into it.
    let gradient_seed = VecFillArgs {
        array: state.tet.gradient_x.handle(),
        value: 0.0,
        count: state.tet.gradient_x.len() as u32,
        seam_arena_count: 0,
    };
    let gradient_seed_count = state.tet.gradient_x.len() as u32;
    // THE SEED, on the device: this array opens at zero and the scale
    // below accumulates into it.
    let _hessian_seed = VecFillArgs {
        array: state.tet.hessian_x.handle(),
        value: 0.0,
        count: state.tet.hessian_x.len() as u32,
        seam_arena_count: 0,
    };
    let _hessian_seed_count = state.tet.hessian_x.len() as u32;
    // `inv_rest_ptr` is the host's own `DataSet` array, which arrives as a
    // pointer rather than as a slice: nine floats per tet.
    let convert_force_args = TetConvertForceArgs {
        gradient_f: state.tet.gradient_f.handle(),
        inverse_rest: rest_ref,
        mass: state.tet.mass.handle(),
        force: state.tet.gradient_x.handle(),
        count,
        seam_arena_count: 0,
    };
    // SIX DISPATCHES, ONE BOUNDARY. `Device::launch` is `run` around a single
    // `elements`, and a backend SYNCHRONIZES at the end of every submit, so six
    // launches are six host round trips where the work between them is only
    // building the next record. Consecutive entries of one region are ordered
    // with a full barrier between them on every backend, which is the property
    // `driver/pcg.rs` already relies on for its seed phase, so this is the same
    // sequence with five stalls removed rather than a reordering.
    //
    // MEASURED: the end-of-submit wait is 75.3 percent of this tree's device
    // API time, at 36,714 waits over ten frames of `drape`, and per-launch
    // submits are where they come from.
    device.run("assemble.tet.spectral", |encoder| {
        encoder.elements(&force_args, count)?;
        encoder.elements(&hessian_args, count)?;
        encoder.elements(&gradient_seed, gradient_seed_count)?;
        encoder.elements(&convert_force_args, count)
    })?;
    // THE MASS SCALE IS NOT A PASS OF ITS OWN: the two converters above apply
    // it as they write. Scaling afterwards would be two more dispatches over
    // the materialized packs, 12 floats a tet and 144, read in full only to be
    // multiplied.
    // Rayleigh stiffness damping, which reads the iterate and the START of the
    // step. Passing the same array twice makes it identically zero, which is a
    // silent no-op rather than an error, so the two must not be confused.
    let damping_args = TetDampingArgs {
        x: state.eval_x.handle(),
        current: state.positions.handle(),
        tet: state.mesh_tet.handle(),
        // The bound the entry point checks the tet's four slots against, which
        // the record carries because a slot is data rather than the thread
        // index and the count guard says nothing about it. Both position
        // buffers are read at those slots, so one bound serves them.
        vertex_count: state.sizes.vertices as u32,
        beta: state.tet.damping.handle(),
        dt,
        gradient: state.tet.gradient_x.handle(),
        hessian: state.tet.hessian_x.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.tet.damping", &damping_args, count)?;

    // THE SCATTER, serial and in ascending tet index. The force goes to the
    // four vertices through the arity-4 embed the hinge scatter provides, and
    // the Hessian's sixteen blocks go to the CSR through `push_blocks`.
    //
    // THE ACTIVE TETS ARE GATHERED INTO ONE ASCENDING RUN FIRST, because the
    // scatter is one dispatch over a contiguous range and the active list is a
    // subset of the tet range. The gather is in ascending tet index, so the fold
    // reaches the four-vertex embeds in that order, which is what the fp32
    // running sum in `force` depends on; the kernel's `Scatter::Atomic` row is
    // what keeps the pass serial.
    // THE MIRROR, before the scatter reads it. Every writer of this array
    // has run by here, and `host()` refuses between a handle and a
    // download rather than folding the previous iteration's values.
    // OVER THE ACTIVE LIST ON THE DEVICE; `hinge_active_embed_force` states
    // the argument in full: the Hessian half of this pass already runs that
    // way, and the order is unchanged because `active` is ascending and the
    // slot is the thread index.
    // GATED IN THE THREAD ON THE TET'S OWN `mu`, which the material pass wrote
    // and which is zero for exactly the tets a host-built list would omit. The
    // order is the same either way: the gate skips the same elements and the
    // survivors are visited in the same ascending sequence.
    let scatter_args = HingeLiveEmbedForceArgs {
        live: state.tet.mu.handle(),
        hinge: state.mesh_tet.handle(),
        gradient: state.tet.gradient_x.handle(),
        force: state.force.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.tet.scatter_force", &scatter_args, count)?;
    push_tet_hessians(device, mesh, state, fixed)
}

/// Fold every active tet's 12x12 Hessian into the fixed matrix.
///
/// ROW-MAJOR OVER THE SIXTEEN BLOCKS, which is `atomic_embed_hessian<4>`'s own
/// `(ii, jj)` order. The block at `(ii, jj)` is the 3x3 sub-block of the 12x12
/// at rows `3 * ii` and columns `3 * jj`, and `Mat12x12f` is column-major, so
/// the extraction below reads column `3 * jj + c`, row `3 * ii + r`.
fn push_tet_hessians<D: Device>(

    device: &mut D,
    _mesh: &[crate::data::Vec4u],
    state: &mut SolverState,
    fixed: &mut FixedCsr<'_>,
) -> FatalResult<()> {
    // ON THE DEVICE, in the element's own thread, and the RUN IS DECIDED THERE
    // TOO: the gate is the tet's own `mu`, which the material pass wrote and
    // which is zero for exactly the tets a host active list would omit.
    let count = state.sizes.tets;
    if count == 0 {
        return Ok(());
    }
    let live = state.tet.mu.handle();
    let index = state.mesh_tet.handle();
    let hessian = state.tet.hessian_x.handle();
    // THE TWO ARMS OF THE SLOT LOOKUP: replay the precomputed slot when the
    // scene carries the table, search the row when `PPF_SLOT_REPLAY=0` asked
    // for the A/B arm.
    if state.tet_hess_slots.len() > 0 {
        let slots = state.tet_hess_slots.handle();
        // Safety: every handle names a live allocation and outlives the dispatch.
        return unsafe {
            fixed.push_element_blocks_gated_at(device, live, slots, hessian, 4, count as u32)
        };
    }
    // Safety: every handle names a live allocation and outlives the dispatch.
    unsafe {
        fixed.push_element_blocks_gated(
            device,
            &mut state.push,
            live,
            index,
            hessian,
            4,
            count as u32,
            "tet",
        )
    }
}

/// The face verdicts `face_material_diff_table` returns, mirrored from the
/// `FACE_TABLE_*` enumerators beside that body in
/// `src/kernels/energy/model/material_diff_table.kernel.cpp`.
///
/// Four-valued rather than a boolean because four shell models exist and the
/// diff table carries three of them. A body that reported a BaraffWitkin face
/// as merely "accepted" would hand it a zero table under a name that says the
/// table is its material.
///
/// READ ONLY BY THE TESTS. The assembly takes no branch on a verdict, because a
/// verdict is a function of the element's material and
/// `super::refusal::material_defects` settles it at `initialize()`. What these
/// codes are for is the pairing: the tests below read each one straight off the
/// kernel, so the vocabulary that gate mirrors is checked against the body that
/// defines it rather than assumed to agree with it.
#[cfg(test)]
mod face_verdict {
    /// The model id is not one any backend implements.
    pub const UNKNOWN: u32 = 0;
    /// The diff table carries this face's material.
    pub const DIFF_TABLE: u32 = 1;
    /// BaraffWitkin: the material comes from `face_baraffwitkin_entry`.
    pub const BARAFF_WITKIN: u32 = 2;
    /// A recognized model with no elastic energy (`Model::Pdrd`).
    pub const NO_ENERGY: u32 = 3;
}

/// The tet verdicts `tet_material_diff_table` returns, mirrored from the
/// `TET_TABLE_*` enumerators beside that body in
/// `src/kernels/energy/model/material_diff_table.kernel.cpp`.
///
/// TWO-VALUED WHERE THE FACE SET IS FOUR-VALUED, and the difference is the
/// shell model set's rather than this file's: a tet has no BaraffWitkin form
/// and no second material family, so the only question a solid's dispatch
/// answers is whether the id was one it knows.
///
/// READ ONLY BY THE TESTS, on the grounds [`face_verdict`] states.
#[cfg(test)]
mod tet_verdict {
    /// The model id is not one the solid table has an arm for.
    pub const UNKNOWN: u32 = 0;
    /// The table carries this tet's material, or the model has no energy.
    pub const ACCEPTED: u32 = 1;
}

/// The shell membrane force and Hessian, in ONE dispatch over the shell prefix.
///
/// A thread holds one triangle's
/// deformation gradient, its factorization, the material diff table, the
/// spectral force, the PSD-projected 6x6 Hessian, the material-frame conversion
/// and the Rayleigh damping block in registers, embeds the result into the force
/// vector and the fixed matrix itself, and crosses to the host ZERO times.
///
/// # The per-face pressure is the SECOND term of the same body
///
/// `face_elastic_embed` runs elasticity and pressure as SIBLING blocks over one
/// face, `mu > 0` and `pressure > 0`, each embedding its own gradient and
/// Hessian, with the stiffness damping applied INSIDE the first one on the
/// elastic accumulators alone. So the pressure term is never damped and never
/// has to be ordered against the damping: the two terms share nothing but their
/// destination. That is why this pass carries no pressure dispatch of its own
/// and no note about where it must sit.
///
/// # The gate is in two halves
///
/// `!fixed && !rest_excluded && !collider` comes from the dispatch
/// and `mu > 0` / `pressure > 0` from the body. `fixed` and
/// `rest_excluded` say the face carries no energy this step, `collider` says its
/// shape is held by its pins rather than by stiffness of its own, and `mu == 0`
/// is how a PDRD face and a zero-stiffness material both arrive. THE FIRST HALF
/// IS DECIDED HERE AND THE SECOND ON THE DEVICE, which is not a relocation:
/// `data.prop.face` is a HOST array in this driver (`plasticity.rs` writes the
/// props through `scene::slice_mut` every step, so the component has not moved),
/// and the material table it indexes is host memory too. So the loop below
/// resolves each face's material ONCE into the six staged arrays and seeds an
/// excluded face inert, which is exactly what the device gate then reads. Every
/// face in the prefix is written, active or not, so none can read a value left
/// by a previous iteration.
///
/// THE RANGE IS THE SHELL PREFIX, NOT THE FACE ARRAY. `mesh.face` carries a
/// solid's surface triangles after the shell faces, and `inv_rest2x2` has
/// exactly `shell_face_count` entries, so a dispatch ranged over `face.size`
/// would read the rest matrices past their end on every tetrahedralized scene
/// and be correct on every shell-only one.
///
/// # The range is every shell face, and the fold is over the active ones
///
/// The dispatch covers the whole prefix, and a face the
/// gate excludes returns before writing anything: not a zero force, not a zero
/// block, not a slot lookup. So the sequence of additions landing on each
/// accumulator is the ascending run of ACTIVE faces either way, which is what
/// the fp32 running sums in `force` and in the matrix's values depend on. The
/// kernel's `Scatter::Atomic` row is what keeps that range one pass: two faces
/// sharing a vertex have one force destination and two sharing an edge have one
/// CSR slot.
///
/// # Safety
/// `data` must be live and `state` sized for the scene.
pub unsafe fn shell_membrane<D: Device>(
    device: &mut D,
    data: &DataSet,
    eiganalysis_eps: f32,
    state: &mut SolverState,
    fixed: &mut FixedCsr<'_>,
    dt: f32,
) -> FatalResult<()> {
    let faces = state.sizes.shell_faces;
    if faces == 0 {
        return Ok(());
    }
    // THE TWO COUNTS, SAID OUT LOUD ONCE PER RUN. `shell_face_count` is a
    // PREFIX of `mesh.face` and the two are equal exactly when the scene
    // carries no solid, which is what made confusing them invisible: a
    // preflight comparing a per-face slot table against the prefix refused
    // EVERY tetrahedralized scene on one backend while every shell-only scene
    // agreed. A reader, or a gate, can now see which number this dispatch
    // walked without inferring it from the scene.
    //
    // IT IS `debug!` BECAUSE IT IS INSTRUMENTATION, NOT SOLVER BEHAVIOR. This
    // driver's own diagnostics stay off the default transcript, so a transcript
    // diff between two runs shows what the solver did rather than what it
    // counted. The argument above and the once-only gate both still apply.
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static SAID: AtomicBool = AtomicBool::new(false);
        if !SAID.swap(true, Ordering::Relaxed) {
            ::log::debug!(
                "face assembly dispatch passes with {faces} shell face(s) of {} \
                 mirrored face(s)",
                data.mesh.mesh.face.size
            );
        }
    }
    let mesh: &[crate::data::Vec3u] = scene::slice(&data.mesh.mesh.face);
    let props: &[FaceProp] = scene::slice(&data.prop.face);
    let params: &[FaceParam] = scene::slice(&data.param_arrays.face);
    let inv_rest: &[crate::data::Mat2x2f] = scene::slice(&data.inv_rest2x2);
    // Each of the three is indexed over the PREFIX, so each must cover it. The
    // rest matrices are the one that is sized over the prefix itself rather
    // than over the whole face array, which is why they are reported apart:
    // reading them short is the failure this range exists to make impossible.
    if props.len() < faces || mesh.len() < faces {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene declares {faces} shell faces, and carries {} face props and \
             {} face index records",
            props.len(),
            mesh.len()
        )));
    }
    if inv_rest.len() < faces {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene declares {faces} shell faces and {} inverse rest matrices. \
             The membrane walks the shell PREFIX of mesh.face, and inv_rest2x2 is sized over \
             that prefix, so a shorter array cannot describe this mesh",
            inv_rest.len()
        )));
    }

    // WHICH FACES ARE IN. Rebuilt every iteration because `fixed` is rebuilt
    // every step by `update_constraint` and `rest_excluded` by
    // `update_rest_shape`. Gathering the material constants in the same pass
    // keeps the param indirection out of the dispatch below, which then reads
    // contiguous arrays.
    let f = &mut state.face;
    f.active.clear();
    // NO MATERIAL SEEDING, AND NO SIX STAGED ARRAYS. The dispatch below is
    // `FaceElasticEmbedFromRecordsArgs` and reads the face's own `FaceProp` and
    // `FaceParam` off the device, so a host gather would fill arrays nothing
    // uploads and nothing names a handle for. Such arrays cost a pass over
    // every face every Newton iteration and reach the kernel not at all, which
    // is silent in both directions.
    //
    // WHAT THIS LOOP STILL DOES IS THE ACTIVE LIST AND THE TWO REFUSALS, which
    // are not dead: the emptiness test below skips the whole pass, and the
    // material and vertex-bound checks name a bad `DataSet` where an
    // out-of-range read on Metal would silently return zero.
    for (index, prop) in props[..faces].iter().enumerate() {
        if prop.fixed || prop.rest_excluded || prop.collider {
            continue;
        }
        let material = params.get(prop.param_index as usize).ok_or_else(|| {
            Fatal::device_assert(format!(
                "solver driver: face {index} names material {} but the scene carries {} face \
                 materials",
                prop.param_index,
                params.len()
            ))
        })?;
        // ELASTICITY AND PRESSURE ARE INDEPENDENT TERMS, and a face carrying
        // either one is in. The body states the rule as two sibling blocks,
        // `mu > 0` and `pressure > 0`,
        // so an inflated membrane with no elastic stiffness assembles its
        // pressure and nothing else. Admitting only `mu > 0` here would leave
        // such a face inert at both gates in the body, which is silent: the
        // term is present, correct, and never dispatched.
        let stiff = material.mu > 0.0;
        let inflated = material.pressure > 0.0;
        if !stiff && !inflated {
            continue;
        }
        for k in 0..3 {
            if mesh[index][k] as usize >= state.sizes.vertices {
                return Err(Fatal::device_assert(format!(
                    "solver driver: face {index} names vertex {} but the scene has {} vertices",
                    mesh[index][k], state.sizes.vertices
                )));
            }
        }
        // THE MATERIAL IS NOT COPIED ANYWHERE. The body reads `model`, `mu`,
        // `lambda`, the mass and the damping off the face's own records, and
        // applies the same `mu > 0` and `pressure > 0` gates as two sibling
        // blocks. What is wanted from this iteration is the membership, and the
        // two refusals above.
        f.active.push(index as u32);
    }
    if f.active.is_empty() {
        return Ok(());
    }

    let count = faces as u32;

    // NO MATERIAL UPLOAD. The dispatch below reads `FaceProp` and `FaceParam`
    // off the device, in the face's own thread, so the six arrays a host gather
    // would fill and send are not there: measured on `drape` at 3 frames they
    // were 72 host-to-device calls carrying 44.4 MB.

    // The matrix's own pattern and values, taken together. A push looks its
    // block up in the pattern the matrix was BUILT over, which is why the
    // handles come off the matrix rather than off the state.
    let matrix = fixed.view();
    let args = FaceElasticEmbedFromRecordsArgs {
        x: state.eval_x.handle(),
        // The START of the step, which the Rayleigh damping term differences
        // the iterate against. Passing the same buffer twice makes the damping
        // identically zero, which is a silent no-op rather than an error, so
        // the two must not be confused.
        current: state.positions.handle(),
        face: state.mesh_face.handle(),
        // The bound the entry point checks the face's three slots against,
        // which the record carries because a slot is data rather than the
        // thread index and the count guard says nothing about it. Both
        // position buffers are read at those slots, so one bound serves them.
        vertex_count: state.sizes.vertices as u32,
        // THE SAME BUFFER AGAIN, and the second field is not spare: the entry
        // reads the slot list above and does not forward it, while the body
        // addresses the force rows and the CSR blocks by the element's own
        // three vertices. Twelve bytes at the element, which is the same
        // memory.
        face_vertex: state.mesh_face.handle(),
        inverse_rest: state.inv_rest2x2.handle(),
        // THE RECORDS, not six flattened arrays. `prop` is gathered per face
        // and carries the mass and the material's index; `face_param` is
        // indexed per MATERIAL, deduplicated across objects that share one, so
        // the body reaches it through `prop.param_index` rather than through
        // the element index.
        prop: state.prop_face.handle(),
        face_param: state.param_face.handle(),
        dt,
        eigenanalysis_eps: eiganalysis_eps,
        force: state.force.handle(),
        index: matrix.index,
        offset: matrix.offset,
        value: matrix.value,
        row_count: matrix.rows,
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.face.elastic", &args, count)?;
    Ok(())
}

/// The shell hinge bending force and Hessian, staged over every hinge.
///
/// THE FIFTH AND LAST OF THE ELASTIC DISPATCHES, so it is assembled after the
/// membrane and after the tets and not before either: `force` and the two
/// Hessians are fp32 running sums.
///
/// # A hinge is not a bending element merely by existing
///
/// `mesh.hinge` is built over every adjacent face pair in the mesh, a solid's
/// surface triangles included. The dispatch gate is
/// `!fixed && !collider && (mesh.type.hinge[i] & 1) == 0`, and the body then
/// requires `stiff_k > 0`. Bit 0 marks a pair with a SOLID surface face on
/// either side, which carries no bending energy at all; `fixed` says every one
/// of the four vertices is an exact Dirichlet row, so the hinge cannot move;
/// and `collider` says the surface is held by its pins rather than by stiffness
/// of its own. The three do not alias, which is why the dispatch tests each.
///
/// # What the stiffness scalar carries, and why it is formed before anything
///
/// `shell_bend_stiffness` folds three things into one number: the Discrete
/// Shells coefficient `|e|^2 / (A1 + A2)`, which is what makes the bent shape
/// independent of mesh resolution; the areal density, which makes it
/// independent of density; and `shell_bend_directional`, which mixes
/// `bend` with `bend-warp` and `bend-weft` according to where the shared edge
/// sits in the UV material frame. It multiplies the force, the Hessian AND the
/// lagged damping Hessian, so forming it once is what keeps the three
/// consistent. Every input is non-negative, so the scalar cannot turn negative
/// and flip the sign of an already PSD-projected block.
///
/// THE AREAL DENSITY IS THE ONE QUANTITY TAKEN IN THE MESH ORDER. It is an fp32
/// running sum over the hinge's four vertices, so the permutation the dihedral
/// math wants would change its value. The mass and area are therefore read
/// BEFORE the permutation is applied.
///
/// # The damping Hessian is lagged
///
/// `hinge_add_stiffness_damping_lagged` reads a Hessian evaluated at the
/// START-OF-STEP positions, not at the Newton iterate, which makes the damping
/// force the gradient of a convex potential and therefore guaranteed
/// dissipative. That is a SECOND evaluation of the same body, taken only when
/// some hinge this iteration bends asks for damping: with every `beta` zero the
/// damping body returns before reading the lagged Hessian, so the stage
/// produces no value that is read.
///
/// # Safety
/// `data` must be live and `state` sized for the scene.
/// Turn a degenerate-hinge device assert into the fatal that names the hinge.
///
/// THE PAYLOAD IS THE HINGE AND ITS THREE DEGENERACIES. `DIAG_ASSERT4` carries
/// the element index and the two squared normals and the shared edge length
/// beside it, so the message can say WHICH of the three went to zero. A single
/// word per hinge could say only that the angle was undefined, and leave the
/// reader to work out why.
fn hinge_degenerate(fault: Fault, mesh: &[crate::data::Vec4u], pose: &str) -> Fatal {
    match fault {
        Fault::Device { diag, .. } => {
            let named = diag.first.as_ref().map_or(String::new(), |first| {
                // The first parameter is the hinge index, which names the four
                // vertices the scene author has to look at.
                let index = first.payload[0] as usize;
                if index < mesh.len() {
                    format!(
                        ", first at hinge {index} (vertices {}, {}, {}, {})",
                        mesh[index][0], mesh[index][1], mesh[index][2], mesh[index][3]
                    )
                } else {
                    format!(", first at {first}")
                }
            });
            Fatal::device_assert(format!(
                "solver driver: a shell hinge is degenerate {pose}: one of its two triangles has \
                 a zero normal, or its shared edge has zero length, so the dihedral angle it \
                 bends about is undefined. The check failed {} time(s){named}",
                diag.failures
            ))
        }
        other => Fatal::from(other),
    }
}

pub unsafe fn shell_bending<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
    fixed: &mut FixedCsr<'_>,
    dt: f32,
) -> FatalResult<()> {
    let hinges = state.sizes.hinges;
    if hinges == 0 {
        return Ok(());
    }
    let vertices = state.sizes.vertices;
    let mesh: &[crate::data::Vec4u] = scene::slice(&data.mesh.mesh.hinge);
    let props: &[HingeProp] = scene::slice(&data.prop.hinge);
    let params: &[HingeParam] = scene::slice(&data.param_arrays.hinge);
    let kind: &[u8] = scene::slice(&data.mesh.ttype.hinge);
    let vertex_props: &[VertexProp] = state.prop_vertex.host();
    if props.len() != hinges || mesh.len() != hinges {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene has {hinges} hinges, {} hinge props and {} hinge index \
             records",
            props.len(),
            mesh.len()
        )));
    }
    // A HINGE WITH NO TYPE BYTE STOPS THE RUN, and neither reading is safe to
    // guess. `mesh.type.hinge` is the only thing that separates a shell's
    // bending hinge from a solid's surface hinge, so a missing byte means the
    // element's class is unknown: bending a solid's surface adds an energy the
    // reference does not assemble, and skipping a shell's hinge drops one it
    // does. `builder.rs` sizes the table over every hinge, so this reports a
    // `DataSet` that came from somewhere else.
    if kind.len() < hinges {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene has {hinges} hinges and a hinge type table of {}. Bit 0 of \
             that table is what marks a hinge with a solid's surface face on either side, which \
             carries no bending energy, so a hinge with no byte has no known element class",
            kind.len()
        )));
    }
    if vertex_props.len() < vertices {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene has {vertices} vertices and {} vertex props. The hinge's \
             areal density is averaged over its four vertices' mass and area",
            vertex_props.len()
        )));
    }

    let h = &mut state.hinge;
    // WHICH HINGES ARE IN, in two parts, because the second half of the gate is
    // a value the shared bodies produce. This pass applies the three flag tests
    // and gathers the material; the `stiff_k > 0` half is applied after the
    // stiffness range below.
    //
    // THE LOOP WRITES NOTHING. It applies the three flag tests only to decide
    // which hinges the authoring checks below apply to; the stiffness itself is
    // formed on the device from the records, which is what the note at the foot
    // of this loop records.
    // ONCE PER SCENE, NOT ONCE PER ASSEMBLY. Every input this loop reads is
    // immutable for the run: `mesh`, `props`, `kind` and `params` are all
    // borrowed from the `DataSet`, which `builder.rs` constructs once and
    // nothing writes afterwards -- the driver's only `&mut DataSet` is a test
    // helper in `refusal.rs`. So a hinge that names a vertex the scene lacks,
    // or that asks for anisotropy while carrying the no-UV sentinel, is as
    // wrong on the first Newton iteration as on the thousandth, and checking it
    // on each cost 1.83 s of a 117 s `trapped` run.
    //
    // THE LATCH GUARDS THE CHECK AS WELL AS THE FILL, and what makes that
    // sound is the immutability above rather than the two happening to share a
    // latch.
    if h.remapped_hinges != hinges {
        for index in 0..hinges {
        let quad = mesh[index];
        for k in 0..4 {
            if quad[k] as usize >= vertices {
                return Err(Fatal::device_assert(format!(
                    "solver driver: hinge {index} names vertex {} but the scene has {vertices} \
                     vertices",
                    quad[k]
                )));
            }
        }
        if props[index].fixed || props[index].collider || (kind[index] & 1) != 0 {
            continue;
        }
        let material = params.get(props[index].param_index as usize).ok_or_else(|| {
            Fatal::device_assert(format!(
                "solver driver: hinge {index} names material {} but the scene carries {} hinge \
                 materials",
                props[index].param_index,
                params.len()
            ))
        })?;
        // ANISOTROPY WITHOUT A DIRECTION IS AN AUTHORING ERROR, and the shared
        // body cannot report it: a negative `uv_edge_sin2` is the no-UV
        // sentinel and reads as isotropic, so a scene asking for warp or weft
        // stiffness on a mesh with no UV would quietly get neither. `scene.rs`
        // raises this when the scene is built; this is the same condition on
        // the same two numbers, for a `DataSet` that did not come through it.
        if props[index].uv_edge_sin2 < 0.0
            && (material.bend_warp != 0.0 || material.bend_weft != 0.0)
        {
            return Err(Fatal::invariant(format!(
                "solver driver: hinge {index} (edge {}-{}) asks for directional bending \
                 (bend-warp={}, bend-weft={}) and carries the no-UV sentinel, so it has no \
                 material direction to weight them by. Give the mesh a UV map, or leave both at \
                 0.0",
                quad[0], quad[1], material.bend_warp, material.bend_weft
            )));
        }
        // NO MATERIAL FLATTENING. `shell_bend_stiffness_from_records` reads
        // `bend`, `bend_warp` and `bend_weft` off `HingeParam` and
        // `uv_edge_sin2`, `length` and `area` off `HingeProp`, both on the
        // device, and applies this loop's own exclusion gate there. The six
        // arrays this filled were 72 host-to-device calls carrying 66 MB on
        // `drape` at 3 frames.
        //
        // THE LOOP STAYS for the authoring check above, which names the hinge
        // when a scene asks for anisotropy on a mesh with no UV, and for the
        // permutation and rest angle below.
    }
        // THE (2, 1, 0, 3) PERMUTATION, which `dihedral_angle::remap` applies
        // in place before the dihedral math reads a position. Everything after
        // it is in this order: the force's four columns, the 12x12 Hessian's
        // blocks, the damping body's two poses, and both scatters.
        //
        // Filled in the same guarded pass as the checks above and uploaded on
        // its own line, `handle()` refusing a `u32` set.
        let remapped = h.remapped.at();
        for index in 0..hinges {
            let quad = mesh[index];
            remapped[4 * index] = quad[2];
            remapped[4 * index + 1] = quad[1];
            remapped[4 * index + 2] = quad[0];
            remapped[4 * index + 3] = quad[3];
        }
        h.remapped.upload(device)?;
        h.remapped_hinges = hinges;
    }

    let count = hinges as u32;
    // MESH ORDER, not the permutation: see the note on the areal density above.
    // THE TWO PREP DISPATCHES ARE GONE, and their work is inside
    // `shell_bend_embed` now: it recovers the mesh order from the remapped
    // quad by the same `(2,1,0,3)` involution, sums the areal density in that
    // order because an fp32 running sum is order-dependent, and forms the
    // stiffness and the damping from the hinge's own records. It is ONE
    // dispatch over the hinges.
    //
    // BOTH ENTRIES STAY DECLARED. They are the oracle the bending fixture
    // dispatches to re-derive a stiffness the assembly does not store, which
    // is the only way that test can check the scale without a second
    // implementation of the four-term average this file calls load-bearing.
    // NO MIRROR AND NO ACTIVE LIST. The dispatch covers the full element count
    // and gates on the element's own props, and both of this pass's consumers
    // take that shape, so the stiffness the gate reads never leaves the device.
    // Building the list on the host would cost a
    // download of the whole stiffness array and two uploads of the run it
    // produced, 168.0 MB over ten frames of `drape`, to reach a decision the
    // kernel makes from a value it is already holding.

    // ONE KERNEL FORMS THE HINGE AND EMBEDS IT: the force and the Hessian are
    // computed at the iterate, scaled, damped against the start-of-step pose
    // where the material asks, then scattered and pushed from registers. The
    // 12x12 never reaches memory. Splitting it would take four passes, the
    // checked evaluation into `hessian_x`, the lagged evaluation and the
    // damping add over it, the force scatter reading `gradient_x` back and the
    // CSR push reading `hessian_x` back, about 69 MB each way per assembly per
    // Newton step for the Hessian alone.
    //
    // THE DAMPING QUESTION IS ASKED IN THE KERNEL, per hinge. Asking on the
    // host whether ANY material damps
    // would run a separate pass over every hinge whenever one does; here every
    // hinge reads its own damping and evaluates the lagged pose only when it is
    // positive.
    //
    // THE ORDER IS THE SEPARATE PASSES' ORDER on the host arm: the kernel's
    // `Scatter::Atomic` row keeps that pass serial and ascending, the force and
    // the Hessian go to different arrays so interleaving one hinge's two embeds
    // changes neither array's accumulation sequence, and a hinge outside the
    // gate is skipped exactly as the scatter and the push each skipped it.
    FixedCsr::seed_push_staging(device, &mut state.push)?;
    let (fixed_index, fixed_offset, fixed_value, row_count) = fixed.device_push_refs();
    let embed_args = ShellBendEmbedArgs {
        x: state.eval_x.handle(),
        current: state.positions.handle(),
        // THE REMAPPED LIST, named twice: once for the gather with its bound,
        // once as a plain pointer for the scatter and the push to read the
        // quad through, since a gather brings the positions and not the
        // indices they were reached by.
        hinge: state.hinge.remapped.handle(),
        vertex_count: state.sizes.vertices as u32,
        quad: state.hinge.remapped.handle(),
        prop: state.prop_hinge.handle(),
        hinge_param: state.param_hinge.handle(),
        vertex_prop: state.prop_vertex.handle(),
        kind: state.hinge_kind.handle(),
        dt,
        force: state.force.handle(),
        fixed_index,
        fixed_offset,
        fixed_value,
        row_count,
        refused: state.push.refused.handle(),
        witness: state.push.witness.handle(),
        hess_slots: state.hinge_hess_slots.handle(),
        has_hess_slots: u32::from(state.hinge_hess_slots.len() > 0),
        count,
        seam_arena_count: 0,
    };
    // A DEGENERATE HINGE STOPS THE RUN FROM INSIDE THE KERNEL, as it did from
    // the separate evaluation: a zero triangle normal or a zero shared edge
    // leaves the dihedral angle undefined, and the diagnostic names which.
    device
        .run("assemble.hinge.embed", |encoder| {
            // Safety: every array is borrowed for the whole call.
            unsafe { encoder.elements(&embed_args, count) }
        })
        .map_err(|fault| hinge_degenerate(fault, mesh, "at the iterate or the start of the step"))?;
    fixed.report_refusals(device, &mut state.push, "shell hinge")
}


/// One edge's material, with the indirection checked rather than indexed.
///
/// `param_index` deduplicates identical materials across objects, so it is an
/// index into a shorter array than the edge array and a stale one names a
/// material that is not there. Both rod layers reach it, and a Rust index panic
/// there would report a line number where a named failure reports the edge.
fn material_of<'a>(
    props: &[EdgeProp],
    params: &'a [EdgeParam],
    edge: usize,
) -> FatalResult<&'a EdgeParam> {
    params.get(props[edge].param_index as usize).ok_or_else(|| {
        Fatal::device_assert(format!(
            "solver driver: edge {edge} names material {} but the scene carries {} edge materials",
            props[edge].param_index,
            params.len()
        ))
    })
}

/// The rod bending force and Hessian, staged over every interior rod vertex.
///
/// THE FIRST OF THE FIVE ELASTIC DISPATCHES, so it is assembled before the rod
/// stretch, the membrane, the tets and the hinges.
///
/// THE DISPATCH IS PER VERTEX AND THE ELEMENT HAS NO ARRAY. The range is the
/// surface vertex count and the body selects the vertices with
/// exactly two incident edges and no incident face; the element it then
/// assembles is the three-node stencil `(j, i, k)`, the interior vertex between
/// its two edge-neighbors. Those sites are fixed at scene build, so
/// [`super::state::SolverState::allocate`] enumerates them once and this walks
/// the result.
///
/// THE GATE COMPOSES INTO ONE TEST, as the shell hinge's does. `fix_index == 0`
/// comes from the dispatch and `mass > 0 && stiff_k > 0` from the body, and
/// `rod_bend_stiffness` is `bend * mass * ref^2 / voronoi^2` with `bend`
/// and `mass` non-negative and `voronoi` positive, so a positive stiffness is
/// exactly a positive `bend` and a positive `mass`. A site the dispatch
/// excludes is seeded with a zero `bend` AND a zero `mass`, which no material
/// can revive, so the active list is `{ s : stiffness[s] > 0 }` and the two
/// halves cannot drift apart.
///
/// THERE IS NO DIRECTIONAL ANALOGUE. `bend-warp` and `bend-weft` are shell
/// parameters read in the hinge's UV material frame; a rod carries no material
/// frame, and `EdgeParam` has no field for either.
///
/// # Safety
/// `data` must be live and `state` sized for the scene.
pub unsafe fn rod_bend<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
    fixed: &mut FixedCsr<'_>,
    dt: f32,
) -> FatalResult<()> {
    let sites = state.sizes.rod_bend_sites;
    if sites == 0 {
        return Ok(());
    }
    let _ = data;
    let vertices = state.sizes.vertices;
    let count = sites as u32;
    let (fixed_index, fixed_offset, fixed_value, row_count) = fixed.device_push_refs();
    FixedCsr::seed_push_staging(device, &mut state.push)?;
    // ONE DISPATCH OVER THE BENDING SITES. The stiffness and the damping
    // are formed in the thread from the site's two incident edges and its
    // interior vertex rather than staged by the host, so nothing here gathers a
    // material, sizes a per-site array or reads one back.
    let embed_args = RodBendEmbedArgs {
        x: state.eval_x.handle(),
        current: state.positions.handle(),
        node: state.rod_bend.node.handle(),
        vertex_count: vertices as u32,
        // THE SAME ARRAY AS `node`, in its other role: the entry reads the three
        // slots through the field above and checks them against the bound, and
        // the body walks this one for the scatter's rows and columns.
        node_slots: state.rod_bend.node.handle(),
        site_edge: state.rod_bend.edge_device.handle(),
        edge_prop: state.prop_edge.handle(),
        edge_param: state.param_edge.handle(),
        vertex_prop: state.prop_vertex.handle(),
        dt,
        force: state.force.handle(),
        fixed_index,
        fixed_offset,
        fixed_value,
        row_count,
        refused: state.push.refused.handle(),
        witness: state.push.witness.handle(),
        hess_slots: state.rod_bend_hess_slots.handle(),
        has_hess_slots: u32::from(state.rod_bend_hess_slots.len() > 0),
        count,
        seam_arena_count: 0,
    };
    // Safety: every array is borrowed for the whole call.
    unsafe { device.launch("assemble.rod_bend.embed", &embed_args, count) }?;
    fixed.report_refusals(device, &mut state.push, "rod bend")
}


/// The rod stretch force and Hessian, over every active rod.
///
/// THE SECOND OF THE FIVE ELASTIC DISPATCHES, so it is assembled after the rod
/// bending and before the membrane. The gate is `!fixed` from the dispatch plus
/// `stiffness > 0` from the body.
///
/// THE RANGE IS THE ROD PREFIX, NOT THE EDGE ARRAY. `mesh.edge` carries every
/// face's edges after the rods, because edge-edge contact needs them, and
/// `prop.edge`, `param_arrays.edge` and `edge_hess_slots` all span that whole
/// array. A face edge has a zero mass and a live stiffness from the material it
/// inherited, so a walk over `edge.size` would assemble a stretch energy on
/// every shell's edges. The two counts coincide exactly when the scene carries
/// no shell and no solid.
///
/// TWO REST LENGTHS EXIST AND THEY ARE ONE LETTER APART. The stretch measures
/// against `EdgeProp::length`, the `length-factor`-scaled rest length; the
/// strain limiter measures against `EdgeProp::initial_length`. A `length-factor`
/// of 1.0 makes them equal, which is what hides an exchange of the two.
///
/// THE EVALUATION IS OVER THE ACTIVE LIST, not over the whole prefix, and this
/// is the one place this backend's element layers differ in shape.
/// `hook::make_diff_table` divides by the segment's CURRENT length, which is
/// not a value the caller can seed: an inactive tet is given a zero mass and
/// every stage stays well defined on it, and there is no equivalent here. The
/// body is therefore never evaluated outside the gate on any backend.
///
/// # Safety
/// `data` must be live and `state` sized for the scene.
pub unsafe fn rod_stretch<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
    fixed: &mut FixedCsr<'_>,
    dt: f32,
) -> FatalResult<()> {
    let rods = state.sizes.rods;
    if rods == 0 {
        return Ok(());
    }
    let vertices = state.sizes.vertices;
    let mesh: &[crate::data::Vec2u] = scene::slice(&data.mesh.mesh.edge);
    let props: &[EdgeProp] = scene::slice(&data.prop.edge);
    let params: &[EdgeParam] = scene::slice(&data.param_arrays.edge);
    if props.len() < rods || mesh.len() < rods {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene declares {rods} rods, with {} edge props and {} edge index \
             records. The rods are a PREFIX of the edge array, so both must cover it",
            props.len(),
            mesh.len()
        )));
    }

    // WHAT THE HOST STILL CHECKS, and why it is not a readback. `props` is the
    // scene's own host-resident array, so this loop reads no device memory and
    // costs no transfer; what it buys is a LOUD failure on a divisor the shared
    // body does not guard. `rod_stretch_diff_table` divides by the rest length,
    // and a zero writes a non-finite gradient that no later multiply recovers,
    // so a scene that reached here without `scene.rs`'s positive-length
    // assertion, or with a `length-factor` of zero, must be named rather than
    // silently turned into a NaN.
    //
    // THE GATE ITSELF IS NOT HERE. Which rods carry a stretch energy is decided
    // in the element's own thread, off the edge's prop and its material, so
    // this loop builds no list, stages no stiffness, mass, damping or rest
    // length, and compacts nothing.
    for index in 0..rods {
        if props[index].fixed {
            continue;
        }
        if !(material_of(props, params, index)?.stiffness > 0.0) {
            continue;
        }
        for k in 0..2 {
            if mesh[index][k] as usize >= vertices {
                return Err(Fatal::device_assert(format!(
                    "solver driver: rod {index} names vertex {} but the scene has {vertices} \
                     vertices",
                    mesh[index][k]
                )));
            }
        }
        if !(props[index].length > 0.0) {
            return Err(Fatal::invariant(format!(
                "solver driver: rod {index} has rest length {}, and the stretch energy measures \
                 its strain as a ratio against it. `length-factor` scales the drawn length, so a \
                 factor of zero produces this",
                props[index].length
            )));
        }
    }

    // ONE DISPATCH OVER THE ROD PREFIX: the edge, its prop and its material are
    // read in the thread,
    // the gate is applied there, the diff table and the Rayleigh damping are
    // built in registers, and the force and the nine-block Hessian are
    // scattered without either ever reaching memory as a whole-array pass.
    let count = rods as u32;
    let (fixed_index, fixed_offset, fixed_value, row_count) = fixed.device_push_refs();
    FixedCsr::seed_push_staging(device, &mut state.push)?;
    let embed_args = RodStretchEmbedArgs {
        x: state.eval_x.handle(),
        current: state.positions.handle(),
        edge: state.mesh_edge.handle(),
        vertex_count: vertices as u32,
        // THE SAME ARRAY AS `edge`, IN ITS OTHER ROLE: the entry reads the two
        // slots through the field above and checks them against the bound, and
        // the body walks this one to give the scatter and the CSR lookup their
        // row and column.
        edge_slots: state.mesh_edge.handle(),
        prop: state.prop_edge.handle(),
        edge_param: state.param_edge.handle(),
        dt,
        hess_slots: state.edge_hess_slots.handle(),
        has_hess_slots: u32::from(state.edge_hess_slots.len() > 0),
        force: state.force.handle(),
        fixed_index,
        fixed_offset,
        fixed_value,
        row_count,
        refused: state.push.refused.handle(),
        witness: state.push.witness.handle(),
        count,
        seam_arena_count: 0,
    };
    // Safety: every array is borrowed for the whole call.
    unsafe { device.launch("assemble.rod.embed", &embed_args, count) }?;
    fixed.report_refusals(device, &mut state.push, "rod stretch")
}


/// The cross-stitch force and Hessian, over every stitch this step carries.
///
/// Dispatched AFTER the five elastic layers and BEFORE the `tmp_fixed`
/// snapshot. That position is load-bearing rather than cosmetic: the
/// snapshot is the matrix the contact stiffness contracts, so a stitch
/// assembled on the far side of it would leave every contact on a stitched
/// scene measuring its stiffness against a matrix the stitch blocks are missing
/// from.
///
/// # A stitch is six slots, and the degenerate form is not a special case
///
/// Slots 0 to 2 name the source triangle and 3 to 5 the target, each with
/// barycentric weights summing to one. An endpoint that is not on a solid
/// arrives as `{s, s, s}` with weights `{1, 0, 0}`, which recovers
/// single-vertex behavior through the same arithmetic. Nothing in this walk
/// distinguishes the two: the repeated slots fold three force contributions
/// onto one vertex and several of the thirty-six Hessian blocks onto one CSR
/// entry, which is the sum the barycentric form asks for when all the weight
/// sits on one slot.
///
/// # There is no per-element gate
///
/// The dispatch covers the whole stitch array with no test, and the
/// body answers a zero stiffness with a zero gradient and a zero Hessian, so a
/// stitch with no stiffness contributes nothing through the same path rather
/// than through a branch. The active-list shape the elastic layers use would
/// change nothing here, and would put a gate in this file that the body already
/// answers.
///
/// # The Hessian is PSD by construction, and this walk cannot lose that
///
/// Both of the body's terms are positive-semidefinite forms, `g g^T` and
/// `dtdx^T dtdx`, and the two coefficients scaling them are clamped at zero
/// inside it, which is the projection. The per-stitch stiffness is a
/// non-negative force factor, which [`SolverState::stash_stitches`] refuses to
/// accept otherwise, since it multiplies the whole block. Nothing here
/// rescales, projects or clamps a block: the walk stages the body's output and
/// folds it, so the projection stays where it is made.
///
/// # Safety
/// `data` must be live and `state` sized for the scene.
pub unsafe fn stitch<D: Device>(
    device: &mut D,
    data: &DataSet,
    stitch_length_factor: f32,
    state: &mut SolverState,
    fixed: &mut FixedCsr<'_>,
) -> FatalResult<()> {
    let stitches = state.sizes.stitches;
    if stitches == 0 {
        return Ok(());
    }
    let vertices = state.sizes.vertices;
    let props: &[VertexProp] = state.prop_vertex.host();
    let params: &[VertexParam] = scene::slice(&data.param_arrays.vertex);
    if props.len() < vertices {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene declares {vertices} vertices with {} vertex props, and a \
             cross-stitch reads a vertex's material through them",
            props.len()
        )));
    }

    // THE PER-VERTEX GATHER, at the vertex ids this step's stitches name and
    // nowhere else. The shared body reads the contact gap and the contact
    // offset at those same ids, so the two arrays are vertex-indexed rather
    // than slot-indexed and every entry outside the named set is untouched and
    // unread.
    //
    // `param_index` deduplicates identical materials across objects, so it
    // indexes a shorter array than the vertex one and a stale value names a
    // material that is not there. Checked rather than indexed, so the failure
    // names the stitch and the slot instead of reporting a line number.
    state.stitch.length_factor.at().fill(stitch_length_factor);
    for element in 0..stitches {
        for slot in 0..6usize {
            let vertex = state.stitch.index.host()[6 * element + slot] as usize;
            if vertex >= vertices {
                return Err(Fatal::device_assert(format!(
                    "solver driver: cross-stitch {element} slot {slot} names vertex {vertex} but \
                     the scene has {vertices} vertices"
                )));
            }
            let material = props[vertex].param_index as usize;
            let Some(vertex_param) = params.get(material) else {
                return Err(Fatal::invariant(format!(
                    "solver driver: cross-stitch {element} slot {slot} is vertex {vertex}, whose \
                     param_index is {material} in a vertex param array of {}",
                    params.len()
                )));
            };
            state.stitch.vertex_ghat.at()[vertex] = vertex_param.ghat;
            state.stitch.vertex_offset.at()[vertex] = vertex_param.offset;
        }
    }

    // THE UPLOADS, after the gather loop above and before the first handle.
    // `handle()` refuses a dirty staged buffer, so a gather that returned
    // early leaves the host copy for the next one rather than dispatching it.
    state.stitch.length_factor.upload(device)?;
    state.stitch.vertex_ghat.upload(device)?;
    state.stitch.vertex_offset.upload(device)?;

    // THE EVALUATION, one call over the whole range. The body writes only this
    // stitch's own gradient and Hessian and reads shared inputs, so nothing
    // scatters here and a partition WOULD be sound.
    let count = stitches as u32;
    let stitch_args = StitchForceHessianGatheredArgs {
        x: state.eval_x.handle(),
        stitch_index: state.stitch.index.handle(),
        // The bound each of the six slots is checked against. The three buffers
        // read through them are all vertex-indexed, which is what lets one
        // index list serve the positions, the contact gaps and the offsets.
        vertex_count: state.sizes.vertices as u32,
        vertex_ghat: state.stitch.vertex_ghat.handle(),
        vertex_offset: state.stitch.vertex_offset.handle(),
        stitch_weight: state.stitch.weight.handle(),
        length_factor: state.stitch.length_factor.handle(),
        stiffness: state.stitch.stiffness.handle(),
        gradient: state.stitch.gradient.handle(),
        hessian: state.stitch.hessian.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.stitch.force_hessian", &stitch_args, count)?;
    // THE MIRRORS, refreshed while the results are the ones this launch
    // wrote. `host()` refuses after a handle has gone out and before a
    // download, so the scan below and the Hessian push cannot read the
    // previous iteration's contents.
    state.stitch.gradient.download(device)?;
    state.stitch.hessian.download(device)?;

    // THE RESULT IS REFUSED IF IT IS NOT A NUMBER, and the check is here rather
    // than several stages later at the solve. The body divides by two lengths
    // this walk may not form for itself: the rest length, which is the
    // barycentric mean of the six slots' contact gaps halved, and the measured
    // length, which is zero when the two endpoints coincide exactly. So the
    // divisors cannot be tested before the call; what can be tested is whether
    // the result is foldable. The gradient alone is scanned because it passes
    // through BOTH divisions, so a non-finite Hessian has a non-finite gradient
    // beside it.
    for element in 0..stitches {
        for component in 0..18usize {
            let value = state.stitch.gradient.host()[18 * element + component];
            if !value.is_finite() {
                return Err(Fatal::device_assert(format!(
                    "solver driver: cross-stitch {element} produced a force of {value}, so its \
                     spring has no length to measure against: either the six slots' contact gaps \
                     are all zero, or the two endpoints are at the same point"
                )));
            }
        }
    }

    // THE SCATTER, serial and in ascending stitch index. One call, because the
    // entry point's own loop is that pass: `compute::atomic_add` on the host
    // seam is a plain read, add and write back, and two stitches sharing a
    // vertex have the same destination.
    let scatter_args = StitchAtomicEmbedForceArgs {
        index: state.stitch.index.handle(),
        gradient: state.stitch.gradient.handle(),
        force: state.force.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.stitch.scatter_force", &scatter_args, count)?;
    push_stitch_hessians(device, state, fixed)
}

/// Fold every stitch's 18x18 Hessian into the fixed matrix.
///
/// ROW-MAJOR OVER THE THIRTY-SIX BLOCKS, which is `atomic_embed_hessian<6>`'s
/// own `(ii, jj)` order. `SMatf<18, 18>` is column-major, so the block at
/// `(ii, jj)` reads column `3 * jj + c`, row `3 * ii + r`.
///
/// THE REPEATS OF A DEGENERATE STITCH ARE LEFT IN. Its repeated slot indices
/// make several of the thirty-six `(row, column)` pairs the same pair, and the
/// push accumulates, so they fold onto one CSR entry. Collapsing them first
/// would drop contributions the barycentric form asks for.
fn push_stitch_hessians<D: Device>(
    device: &mut D,
    state: &mut SolverState,
    fixed: &mut FixedCsr<'_>,
) -> FatalResult<()> {
    let stitches = state.sizes.stitches;
    let mut written = 0usize;
    for element in 0..stitches {
        let base = 324 * element;
        for ii in 0..6usize {
            for jj in 0..6usize {
                let slot = written + 6 * ii + jj;
                state.stitch.push_row[slot] = state.stitch.index.host()[6 * element + ii];
                state.stitch.push_column[slot] = state.stitch.index.host()[6 * element + jj];
                for c in 0..3usize {
                    for r in 0..3usize {
                        state.stitch.push_block[9 * slot + 3 * c + r] =
                            state.stitch.hessian.host()[base + 18 * (3 * jj + c) + 3 * ii + r];
                    }
                }
            }
        }
        written += 36;
    }
    let scratch = &mut state.stitch;
    let rows = &scratch.push_row[..written];
    let columns = &scratch.push_column[..written];
    let blocks = &scratch.push_block[..9 * written];
    let stored = &mut scratch.push_stored[..written];
    fixed.push_blocks(device, &mut state.push, rows, columns, blocks, stored)?;
    // The lower triangle is declined by design; a missing upper-triangle slot
    // is a lost coupling.
    for k in 0..written {
        if stored[k] == 0 && rows[k] <= columns[k] {
            return Err(Fatal::invariant(format!(
                "solver driver: the scene's fixed sparsity has no slot for Hessian block ({}, {}), \
                 so a cross-stitch's coupling would be dropped and the Newton matrix would be \
                 missing it. Every stitch's thirty-six index pairs are registered by the scene \
                 builder's fixed_index_table",
                rows[k], columns[k]
            )));
        }
    }
    Ok(())
}

/// The shell strain-limit barrier's force and Hessian, staged over every shell
/// face.
///
/// Dispatched AFTER the elastic layers, after the stitch layer and after the
/// `tmp_fixed` snapshot, and BEFORE the contact assembly. That position is not a detail: the
/// stiffness this term scales by is a contraction of the SNAPSHOT, so the term
/// has to run where the snapshot is still the elastic-only matrix, and its own
/// blocks have to land in the live matrix the solve reads.
///
/// # What this term is, since its name misleads
///
/// It is not a clamp that engages near the limit. The table it drives is
/// `shell_strain_diff_table` with `ghat` equal to the limit, so at strain
/// `s` the cubic gradient is `2 s^2 / limit` and the curvature `4 s / limit`: it
/// is live at ANY positive stretch and grows from zero. A backend carrying only
/// the line-search half would commit a visibly different trajectory rather than
/// a marginally different one, which is why the two halves are one capability.
///
/// # Two limits live in one face and they are one letter apart in the source
///
/// The barrier's own ghat is the AUTHORED `strainlimit`. The stiffness, and
/// both line searches, measure against `shell_effective_strain_limit`, which
/// divides `1 + limit` by the smaller shrink factor. The two are equal exactly
/// when `shrink-x` and `shrink-y` are both one, which is every scene that does
/// not author shrink, so no shrink-free test can see them exchanged.
///
/// # The gate is NOT the membrane's gate
///
/// `!fixed && !rest_excluded && strainlimit > 0`, with NO `collider` test,
/// while the membrane's dispatch DOES exclude a collider. The difference is
/// deliberate rather than an oversight: a spring-held collider's vertices are
/// free, and the limiter is what stops the collider's own mesh from stretching
/// past the limit it was authored with.
///
/// A SECOND HALF OF THE GATE IS A VALUE THE SVD PRODUCES, the largest SHIFTED
/// singular value being positive, so the active list is built in two passes as
/// the rod bending layer's is. The line search does NOT apply that second
/// half.
///
/// # The stiffness reads the start-of-step pose
///
/// The stiffness contracts the matrix row against `vertex.curr`, the
/// start-of-step pose, not the Newton iterate, because the matrix row it
/// contracts belongs to that pose. The deformation gradient above it reads the
/// iterate. Handing one where
/// the other belongs is a plausible wrong scale on every scene.
///
/// # Safety
/// `data` must be live, `state` sized for the scene, and `reference` must be
/// the `tmp_fixed` snapshot over the scene's own pattern.
pub unsafe fn shell_strain<D: Device>(
    device: &mut D,
    data: &DataSet,
    eiganalysis_eps: f32,
    barrier: u32,
    state: &mut SolverState,
    reference: &mut FixedCsr<'_>,
    fixed: &mut FixedCsr<'_>,
) -> FatalResult<()> {
    let faces = state.sizes.shell_faces;
    if faces == 0 {
        return Ok(());
    }
    let mesh: &[crate::data::Vec3u] = scene::slice(&data.mesh.mesh.face);
    let props: &[FaceProp] = scene::slice(&data.prop.face);
    let params: &[FaceParam] = scene::slice(&data.param_arrays.face);
    let inv_rest: &[crate::data::Mat2x2f] = scene::slice(&data.inv_rest2x2);
    if props.len() < faces || mesh.len() < faces {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene declares {faces} shell faces, and carries {} face props and \
             {} face index records",
            props.len(),
            mesh.len()
        )));
    }
    if inv_rest.len() < faces {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene declares {faces} shell faces and {} inverse rest matrices. \
             The strain limiter walks the shell PREFIX of mesh.face, and inv_rest2x2 is sized \
             over that prefix, so a shorter array cannot describe this mesh",
            inv_rest.len()
        )));
    }

    // WHICH FACES ARE IN, part one: the dispatch's own gate, shared with the
    // line search below, which applies the identical test.
    if !gather_face_strain_gate(mesh, props, params, state)? {
        return Ok(());
    }

    let count = faces as u32;
    let _inv_rest = state.inv_rest2x2.handle();
    let _pattern = reference.pattern();

    // ONE KERNEL RUNS THE WHOLE LIMITER: one dispatch over the shell face
    // count carrying the gate, the deformation gradient, the SVD, the diff
    // table, the stiffness, the spectral force and Hessian, both conversions
    // and both embeds. Splitting it takes fifteen dispatches over sixteen
    // per-face staging arrays, materializing the Hessian three times on its way
    // to the CSR, 36 then 81 then 81 floats a face, and 251 floats a face in
    // all.
    //
    // TWO POSES AND TWO MATRICES, which is why the record names four handles
    // that look redundant. The gradient is taken at the ITERATE and the
    // stiffness at the START OF STEP; the stiffness READS the snapshot matrix
    // and the push WRITES the live one. Both triples come from their own
    // `device_push_refs`, so neither can drift from the matrix it belongs to.
    FixedCsr::seed_push_staging(device, &mut state.push)?;
    let (reference_index, reference_offset, reference_value, reference_rows) =
        reference.device_push_refs();
    let (fixed_index, fixed_offset, fixed_value, row_count) = fixed.device_push_refs();
    let embed_args = ShellStrainEmbedArgs {
        x: state.eval_x.handle(),
        current: state.positions.handle(),
        face: state.mesh_face.handle(),
        vertex_count: state.sizes.vertices as u32,
        // THE SAME ARRAY AS `face`, IN ITS OTHER ROLE: the entry reads the three
        // slots through the field above and checks them against the bound, and
        // the body walks this one to hand the CSR lookup and the scatter their
        // row and column.
        face_slots: state.mesh_face.handle(),
        inverse_rest: state.inv_rest2x2.handle(),
        prop: state.prop_face.handle(),
        face_param: state.param_face.handle(),
        barrier,
        eiganalysis_eps,
        reference_index,
        reference_offset,
        reference_value,
        reference_rows,
        force: state.force.handle(),
        fixed_index,
        fixed_offset,
        fixed_value,
        row_count,
        refused: state.push.refused.handle(),
        witness: state.push.witness.handle(),
        count,
        seam_arena_count: 0,
    };
    // Safety: every array is borrowed for the whole call.
    unsafe { device.launch("assemble.strain.embed", &embed_args, count) }?;
    fixed.report_refusals(device, &mut state.push, "shell strain limit")
}

/// The rod strain-limit barrier's force and Hessian, over every rod.
///
/// Dispatched immediately after the shell half and against the same
/// `tmp_fixed` snapshot.
///
/// THE RANGE IS THE ROD PREFIX, NOT THE EDGE ARRAY, for the reason the stretch
/// layer states: `mesh.edge` carries every face's edges after the rods.
///
/// TWO REST LENGTHS EXIST AND THIS ONE IS `initial_length`. The stretch energy
/// measures against `EdgeProp::length`, the `length-factor`-scaled value; the
/// limiter measures against `EdgeProp::initial_length`, the drawn one. A
/// `length-factor` of one makes them equal, which is what hides an exchange.
///
/// A FALSE VERDICT IS A SKIP HERE AND NOT A STOP, which is the opposite of the
/// shell hinge's geometry verdict. `rod_strain_force_hessian` returns false on
/// a rod that is not stretched at all, which is an ordinary state and
/// contributes nothing; a hinge that cannot be decomposed is a degenerate
/// element and stops the run. Do not unify the two.
///
/// # Safety
/// `data` must be live, `state` sized for the scene, and `reference` must be
/// the `tmp_fixed` snapshot over the scene's own pattern.
pub unsafe fn rod_strain<D: Device>(
    device: &mut D,
    data: &DataSet,
    barrier: u32,
    state: &mut SolverState,
    reference: &mut FixedCsr<'_>,
    fixed: &mut FixedCsr<'_>,
) -> FatalResult<()> {
    let rods = state.sizes.rods;
    if rods == 0 {
        return Ok(());
    }
    let vertices = state.sizes.vertices;
    let mesh: &[crate::data::Vec2u] = scene::slice(&data.mesh.mesh.edge);
    let props: &[EdgeProp] = scene::slice(&data.prop.edge);
    let params: &[EdgeParam] = scene::slice(&data.param_arrays.edge);
    if props.len() < rods || mesh.len() < rods {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene declares {rods} rods, with {} edge props and {} edge index \
             records. The rods are a PREFIX of the edge array, so both must cover it",
            props.len(),
            mesh.len()
        )));
    }

    // WHICH RODS ARE IN, part one: the dispatch's gate plus its rest-length
    // test, shared with the line search below, which applies the identical
    // pair.
    if !gather_rod_strain_gate(mesh, props, params, vertices, state)? {
        return Ok(());
    }

    // THE UPLOADS, one per path that names a handle. The gather is shared
    // with the line search, so each caller publishes for itself.
    state.rod_strain.limit.upload(device)?;
    state.rod_strain.rest_length.upload(device)?;
    state.rod_strain.mass.upload(device)?;
    let count = rods as u32;
    let pattern = reference.pattern();
    let force_args = RodStrainForceHessianGatedArgs {
        x: state.eval_x.handle(),
        edge: state.mesh_edge.span(0, 2 * rods),
        // The bound each of the edge's two slots is checked against. Only the
        // element-indexed references are per rod; `x` is the whole array and
        // the edge's slots address it.
        vertex_count: vertices as u32,
        rest_length: state.rod_strain.rest_length.handle(),
        limit: state.rod_strain.limit.handle(),
        barrier,
        force: state.rod_strain.force_raw.handle(),
        hessian: state.rod_strain.hessian_raw.handle(),
        strain: state.rod_strain.strain.handle(),
        ok: state.rod_strain.ok.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.rod_strain.force_hessian", &force_args, count)?;
    // NO DOWNLOAD AND NO HOST WALK. The body's verdict stays on the device and
    // every pass below gates on it: each dispatch covers the rod count and asks
    // the rod's own records and its own SVD. Compacting a run on the host would
    // mean downloading the whole `ok` array every step and walking it, which is
    // a whole-array `download()` in a per-step path.
    let stiffness_args = RodStrainStiffnessGatedArgs {
        x: state.positions.handle(),
        edge: state.mesh_edge.span(0, 2 * rods),
        vertex_count: vertices as u32,
        // THE SAME ARRAY AS `edge`, IN ITS OTHER ROLE. The entry reads the two
        // slots through the field above and checks them against the bound; the
        // body walks this one to hand the CSR lookup its row and column, which
        // `[[seam::indices]]` consumes. The two fields must name one array.
        edge_slots: state.mesh_edge.span(0, 2 * rods),
        index: state.fixed_index.handle(),
        offset: state.fixed_offset.handle(),
        value: reference.value_handle(),
        row_count: pattern.rows,
        mass: state.rod_strain.mass.handle(),
        strain: state.rod_strain.strain.handle(),
        limit: state.rod_strain.limit.handle(),
        stiffness: state.rod_strain.stiffness.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.rod_strain.stiffness", &stiffness_args, count)?;
    // `H = stiffness * H` and `Fmat = stiffness * Fmat`, as an accumulate into
    // a destination that opens at zero.
    // THE SEED, on the device: this array opens at zero and the scale
    // below accumulates into it.
    let seed = VecFillArgs {
        array: state.rod_strain.gradient_x.handle(),
        value: 0.0,
        count: state.rod_strain.gradient_x.len() as u32,
        seam_arena_count: 0,
    };
    device.launch("assemble.rod_strain.gradient_x.fill", &seed, state.rod_strain.gradient_x.len() as u32)?;
    // THE SEED, on the device: this array opens at zero and the scale
    // below accumulates into it.
    let seed = VecFillArgs {
        array: state.rod_strain.hessian_x.handle(),
        value: 0.0,
        count: state.rod_strain.hessian_x.len() as u32,
        seam_arena_count: 0,
    };
    device.launch("assemble.rod_strain.hessian_x.fill", &seed, state.rod_strain.hessian_x.len() as u32)?;
    let scale_args_11 = ElementAddScaledArgs {
        source: state.rod_strain.force_raw.handle(),
        destination: state.rod_strain.gradient_x.handle(),
        scale: state.rod_strain.stiffness.handle(),
        stride: 6,
        count: count * 6,
        seam_arena_count: 0,
    };
    device.launch("assemble.rod_strain.stiffness_scale_force", &scale_args_11, count * 6)?;
    let scale_args_12 = ElementAddScaledArgs {
        source: state.rod_strain.hessian_raw.handle(),
        destination: state.rod_strain.hessian_x.handle(),
        scale: state.rod_strain.stiffness.handle(),
        stride: 36,
        count: count * 36,
        seam_arena_count: 0,
    };
    device.launch("assemble.rod_strain.stiffness_scale_hessian", &scale_args_12, count * 36)?;

    // THE SCATTER, serial and in ascending rod index.
    // THE MIRROR, before the scatter reads it. Every writer of this array
    // has run by here, and `host()` refuses between a handle and a
    // download rather than folding the previous iteration's values.
    // OVER THE ACTIVE LIST ON THE DEVICE; `hinge_active_embed_force` states
    // the argument in full: the Hessian half of this pass already runs that
    // way, and the order is unchanged because `active` is ascending and the
    // slot is the thread index.
    // OVER EVERY ROD WITH THE RUN DECIDED IN THE KERNEL, which is the shape the
    // hinge, membrane and shell strain layers already take. The gate skips
    // exactly the rods a compacted run would omit and the survivors are visited
    // in the same ascending sequence, so the fp32 fold is the same; and no list
    // has to be uploaded, which would be a second transfer per step carrying a
    // decision the device has already made.
    let scatter_args = RodLiveEmbedForceArgs {
        live: state.rod_strain.ok.handle(),
        edge: state.mesh_edge.handle(),
        gradient: state.rod_strain.gradient_x.handle(),
        force: state.force.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("assemble.rod_strain.scatter_force", &scatter_args, count)?;
    push_rod_strain_hessians(device, mesh, state, fixed)
}

/// Fold every active rod's 6x6 strain-limit Hessian into the fixed matrix.
///
/// ROW-MAJOR OVER THE FOUR BLOCKS, which is `atomic_embed_hessian<2>`'s own
/// `(ii, jj)` order. `Mat6x6f` is column-major, so the block at `(ii, jj)` reads
/// column `3 * jj + c`, row `3 * ii + r`.
fn push_rod_strain_hessians<D: Device>(

    device: &mut D,
    _mesh: &[crate::data::Vec2u],
    state: &mut SolverState,
    fixed: &mut FixedCsr<'_>,
) -> FatalResult<()> {
    // ON THE DEVICE, in the element's own thread, OVER EVERY ROD WITH THE RUN
    // DECIDED IN THE KERNEL. This took an active list the host had compacted
    // out of a downloaded verdict and uploaded again; the verdict never leaves
    // the device now, so the gated form reads it directly.
    //
    // THE GATE IS LOAD-BEARING, as it is for the hinge: a rod outside the run
    // has a Hessian the stiffness left at zero, and adding 0.0f is exact, so
    // pushing it looks free. It is not, because `builder.rs` has no reason to
    // have registered its stencil and `fixed_csr_atomic_push` counts a refusal.
    let count = state.sizes.rods;
    if count == 0 {
        return Ok(());
    }
    let live = state.rod_strain.ok.handle();
    let index = state.mesh_edge.handle();
    let hessian = state.rod_strain.hessian_x.handle();
    // Safety: every handle names a live allocation and outlives the dispatch.
    unsafe {
        fixed.push_element_blocks_live(
            device,
            &mut state.push,
            live,
            index,
            hessian,
            2,
            count as u32,
            "rod strain",
        )
    }
}

/// Which shell faces the strain limiter covers this iteration, and the two
/// limits and the mass each of them carries.
///
/// ONE FUNCTION FOR BOTH HALVES OF THE LIMITER, because both apply one test,
/// `!fixed && !rest_excluded && strainlimit > 0`. Two copies could drift, and a
/// line search that bounded a step the assembly had assembled no barrier for,
/// or the reverse, would be a limiter that half exists.
///
/// EVERY FACE'S SLOTS ARE WRITTEN, in or out, so no stage can read a value a
/// previous iteration left. A zero in both limits is the inert seed, and every
/// shim stage tests it: a face outside the gate gets a zero table, a zero
/// stiffness, a zero contribution and a time of impact of `max_t`, rather than a
/// division by a zero ghat.
///
/// Returns whether any face is in.
///
/// # Safety
/// The three slices must cover the shell prefix and `state` be sized for it.
unsafe fn gather_face_strain_gate(
    mesh: &[crate::data::Vec3u],
    props: &[FaceProp],
    params: &[FaceParam],
    state: &mut SolverState,
) -> FatalResult<bool> {
    let faces = state.sizes.shell_faces;
    let vertices = state.sizes.vertices;
    let s = &mut state.face_strain;
    s.candidate.clear();
    for index in 0..faces {
        let prop = &props[index];
        // NO `collider` TEST, which the membrane's dispatch does have. A
        // spring-held collider's vertices are free, and the limit it was
        // authored with is what stops its own mesh from stretching past it.
        if prop.fixed || prop.rest_excluded {
            continue;
        }
        let material = params.get(prop.param_index as usize).ok_or_else(|| {
            Fatal::device_assert(format!(
                "solver driver: face {index} names material {} but the scene carries {} face \
                 materials",
                prop.param_index,
                params.len()
            ))
        })?;
        if !(material.strainlimit > 0.0) {
            continue;
        }
        for k in 0..3 {
            if mesh[index][k] as usize >= vertices {
                return Err(Fatal::device_assert(format!(
                    "solver driver: face {index} names vertex {} but the scene has {vertices} \
                     vertices",
                    mesh[index][k]
                )));
            }
        }
        s.candidate.push(index as u32);
    }
    Ok(!s.candidate.is_empty())
}

/// Which rods the strain limiter covers this iteration, and the limit, rest
/// length and mass each carries. The rod counterpart of
/// [`gather_face_strain_gate`], sharing its reasons.
///
/// THE REST LENGTH IS `EdgeProp::initial_length`, not the
/// `length-factor`-scaled `EdgeProp::length` the stretch energy measures
/// against, and a non-positive one takes the rod out rather than being divided
/// by: `rod_strain_force_hessian` and `rod_strain_toi` both divide by it
/// with no guard, and the CUDA dispatch answers that with an early return at
/// `:134` and an `if` at `:179`.
///
/// # Safety
/// The three slices must cover the rod prefix and `state` be sized for it.
unsafe fn gather_rod_strain_gate(
    mesh: &[crate::data::Vec2u],
    props: &[EdgeProp],
    params: &[EdgeParam],
    vertices: usize,
    state: &mut SolverState,
) -> FatalResult<bool> {
    let rods = state.sizes.rods;
    let r = &mut state.rod_strain;
    r.candidate.clear();
    r.active.clear();
    for index in 0..rods {
        r.limit.at()[index] = 0.0;
        r.rest_length.at()[index] = 0.0;
        r.mass.at()[index] = 0.0;
        if props[index].fixed {
            continue;
        }
        let material = material_of(props, params, index)?;
        if !(material.strainlimit > 0.0) {
            continue;
        }
        if !(props[index].initial_length > 0.0) {
            continue;
        }
        for k in 0..2 {
            if mesh[index][k] as usize >= vertices {
                return Err(Fatal::device_assert(format!(
                    "solver driver: rod {index} names vertex {} but the scene has {vertices} \
                     vertices",
                    mesh[index][k]
                )));
            }
        }
        r.limit.at()[index] = material.strainlimit;
        r.rest_length.at()[index] = props[index].initial_length;
        r.mass.at()[index] = props[index].mass;
        r.candidate.push(index as u32);
    }
    Ok(!r.candidate.is_empty())
}

/// The shell strain limiter's per-face time of impact, into
/// `state.face_strain.toi`.
///
/// The second half of the limiter and the one that makes the guarantee: the
/// barrier raises the cost of stretching and this REFUSES the fraction of the
/// step that would cross the limit. The driver folds it into the step's `toi`
/// beside the contact CCD.
///
/// THE SWEEP RUNS FROM `target` TO `eval_x`, and at STEP B29 `target` is holding
/// the iterate's PRE-STEP positions rather than the implicit target, so
/// `target` is what the body reads as the sweep's START. Exchanging the two
/// measures the reverse sweep and returns a plausible number.
///
/// IT MEASURES AGAINST THE SHRINK-CORRECTED LIMIT, as the stiffness does, and
/// applies NO `maxCoeff > 0` test: a face that is not stretched now can be
/// stretched past the limit by the step, which is the case this exists for.
///
/// One value per face and no reduction. The fold is the driver's, so that this
/// backend's answer does not depend on how the range was cut.
///
/// # Safety
/// `data` must be live and `state` sized for the scene.
pub unsafe fn shell_strain_toi<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
    max_t: f32,
) -> FatalResult<()> {
    let faces = state.sizes.shell_faces;
    if faces == 0 {
        return Ok(());
    }
    let mesh: &[crate::data::Vec3u] = scene::slice(&data.mesh.mesh.face);
    let props: &[FaceProp] = scene::slice(&data.prop.face);
    let params: &[FaceParam] = scene::slice(&data.param_arrays.face);
    let inv_rest: &[crate::data::Mat2x2f] = scene::slice(&data.inv_rest2x2);
    if props.len() < faces || mesh.len() < faces || inv_rest.len() < faces {
        return Err(Fatal::invariant(format!(
            "solver driver: the strain-limit line search walks {faces} shell faces over {} face \
             props, {} face index records and {} inverse rest matrices",
            props.len(),
            mesh.len(),
            inv_rest.len()
        )));
    }
    if !gather_face_strain_gate(mesh, props, params, state)? {
        // THE SEED, on the device, for the branch that dispatches nothing.
        // The gate found no candidate, so no kernel will write this array, and
        // the reduction that follows still has to see the ceiling in every slot.
        let seed = VecFillArgs {
            array: state.face_strain.toi.handle(),
            value: max_t,
            count: faces as u32,
            seam_arena_count: 0,
        };
        device.launch("assemble.strain.toi.fill", &seed, faces as u32)?;
        return Ok(());
    }
    let toi_args = ShellStrainToiFromRecordsArgs {
        start: state.target.handle(),
        finish: state.eval_x.handle(),
        face: state.mesh_face.handle(),
        // The bound the entry point checks the face's three slots against.
        vertex_count: state.sizes.vertices as u32,
        inverse_rest: state.inv_rest2x2.handle(),
        // THE LIMIT IS EVALUATED IN THE KERNEL from the face's own records, so
        // this path stages nothing and owes no upload. Naming a staged array
        // that a host gather had just rewritten would carry an upload of its
        // own.
        prop: state.prop_face.handle(),
        face_param: state.param_face.handle(),
        max_t,
        toi: state.face_strain.toi.handle(),
        count: faces as u32,
        seam_arena_count: 0,
    };
    device.launch("assemble.strain.toi", &toi_args, faces as u32)?;
    // THE RESULT STAYS ON THE DEVICE. `step.rs` folds this array with
    // `DeviceFold::min` and reads back one float, so downloading the whole
    // per-element mirror here would move it across the bus for nobody.
    Ok(())
}

/// The rod strain limiter's per-rod time of impact, into
/// `state.rod_strain.toi`.
///
/// Folded into the step's `toi` beside the shell half. The sweep and the fold
/// have the shell half's shape and reasons.
///
/// # Safety
/// `data` must be live and `state` sized for the scene.
pub unsafe fn rod_strain_toi<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
    max_t: f32,
) -> FatalResult<()> {
    let rods = state.sizes.rods;
    if rods == 0 {
        return Ok(());
    }
    let vertices = state.sizes.vertices;
    let mesh: &[crate::data::Vec2u] = scene::slice(&data.mesh.mesh.edge);
    let props: &[EdgeProp] = scene::slice(&data.prop.edge);
    let params: &[EdgeParam] = scene::slice(&data.param_arrays.edge);
    if props.len() < rods || mesh.len() < rods {
        return Err(Fatal::invariant(format!(
            "solver driver: the rod strain-limit line search walks {rods} rods over {} edge props \
             and {} edge index records",
            props.len(),
            mesh.len()
        )));
    }
    if !gather_rod_strain_gate(mesh, props, params, vertices, state)? {
        // THE SEED, on the device, for the branch that dispatches nothing.
        // The gate found no candidate, so no kernel will write this array, and
        // the reduction that follows still has to see the ceiling in every slot.
        let seed = VecFillArgs {
            array: state.rod_strain.toi.handle(),
            value: max_t,
            count: rods as u32,
            seam_arena_count: 0,
        };
        device.launch("assemble.rod_strain.toi.fill", &seed, rods as u32)?;
        return Ok(());
    }
    // THE UPLOADS, as the assembly path above; this one names the limit and
    // the rest length.
    state.rod_strain.limit.upload(device)?;
    state.rod_strain.rest_length.upload(device)?;
    let toi_args = RodStrainToiGatedArgs {
        start: state.target.handle(),
        finish: state.eval_x.handle(),
        edge: state.mesh_edge.span(0, 2 * rods),
        // The bound the entry point checks the edge's two slots against.
        vertex_count: state.sizes.vertices as u32,
        rest_length: state.rod_strain.rest_length.handle(),
        limit: state.rod_strain.limit.handle(),
        max_t,
        toi: state.rod_strain.toi.handle(),
        count: rods as u32,
        seam_arena_count: 0,
    };
    device.launch("assemble.rod_strain.toi", &toi_args, rods as u32)?;
    // THE RESULT STAYS ON THE DEVICE. `step.rs` folds this array with
    // `DeviceFold::min` and reads back one float, so downloading the whole
    // per-element mirror here would move it across the bus for nobody.
    Ok(())
}

/// The per-element stretch ratios the `max_sigma` indicator is reduced from.
///
/// Measured once per `advance()` at the START-OF-STEP pose and before the
/// Newton loop. TELEMETRY: nothing in the solve reads it and a
/// scene runs the same trajectory with and without it. It is here because it is
/// one of the three indicator streams a lost PCG residual denominator is read
/// off, and a collapsing `SL_toi` is only diagnostic beside an exploding
/// `max_sigma`.
///
/// THE SHELL RATIO IS NOT THE STRAIN THE LIMITER IS A FUNCTION OF. It is
/// `max(sigma) * min(shrink_x, shrink_y)`, a ratio about one, where the barrier
/// takes `max(sigma) - 1`. Neither the shifted SVD above nor
/// `shell_max_strain` can stand in: adding one back to a shifted value is
/// not the identity in fp32.
///
/// THE GATE IS ITS OWN, and it is the only place `pdrd_body_index` is read on
/// this backend: a PDRD body's rigid fit always carries a small singular-value
/// residual that would dominate the indicator, and a STATIC collider's distance
/// from its rest shape reflects what its pins allowed rather than material
/// stretch. `rest_excluded` is deliberately NOT tested: an element dropped from
/// the elastic energy still has a stretch worth reporting.
///
/// Writes `state.stretch.ratio` over the shell prefix and
/// `state.stretch.rod_ratio` over the rod prefix; the caller reduces both.
///
/// **BOTH RESULTS STAY ON THE DEVICE.** The caller folds them with
/// [`DeviceFold::max`] and four bytes come back, so neither array is
/// downloaded. A test that wants to read one calls `download` itself; doing it
/// here would move a per-face array across the bus on every step for a mirror
/// production never reads.
///
/// # Safety
/// `data` must be live and `state` sized for the scene.
pub unsafe fn stretch_indicator<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
) -> FatalResult<()> {
    let faces = state.sizes.shell_faces;
    let rods = state.sizes.rods;
    let curr = state.positions.handle();
    if faces > 0 {
        let mesh: &[crate::data::Vec3u] = scene::slice(&data.mesh.mesh.face);
        let props: &[FaceProp] = scene::slice(&data.prop.face);
        let _params: &[FaceParam] = scene::slice(&data.param_arrays.face);
        let _vertex_props: &[VertexProp] = state.prop_vertex.host();
        let inv_rest: &[crate::data::Mat2x2f] = scene::slice(&data.inv_rest2x2);
        if props.len() < faces || mesh.len() < faces || inv_rest.len() < faces {
            return Err(Fatal::invariant(format!(
                "solver driver: the stretch indicator walks {faces} shell faces over {} face \
                 props, {} face index records and {} inverse rest matrices",
                props.len(),
                mesh.len(),
                inv_rest.len()
            )));
        }
        // NO HOST LOOP AND NO TWO ARRAYS. `shell_stretch_terms` reads the
        // face's own material and applies the same gate in the face's own
        // thread, taking `min(shrink_x, shrink_y)` off the record it is already
        // holding. A host gather would maintain a per-face array of each, and
        // every input to it is fixed for the life of the run, so every step
        // after the first would recompute the previous step's answer and upload
        // it.
        let count = faces as u32;
        let deformation_args = FaceDeformationGradientArgs {
            x: curr,
            face: state.mesh_face.handle(),
            vertex_count: state.sizes.vertices as u32,
            inverse_rest: state.inv_rest2x2.handle(),
            deformation: state.stretch.deformation.handle(),
            count,
            seam_arena_count: 0,
        };
        device.launch(
            "assemble.stretch.deformation_gradient",
            &deformation_args,
            count,
        )?;
        let svd_args = Svd3x2Args {
            input: state.stretch.deformation.handle(),
            u: state.stretch.svd_u.handle(),
            sigma: state.stretch.sigma.handle(),
            vt: state.stretch.svd_vt.handle(),
            count,
            seam_arena_count: 0,
        };
        device.launch("assemble.stretch.svd3x2", &svd_args, count)?;
        let terms_args = ShellStretchTermsArgs {
            sigma: state.stretch.sigma.handle(),
            face: state.mesh_face.handle(),
            prop: state.prop_face.handle(),
            face_param: state.param_face.handle(),
            vertex_prop: state.prop_vertex.handle(),
            largest: state.stretch.largest.handle(),
            shrink_min: state.stretch.shrink_min.handle(),
            count,
            seam_arena_count: 0,
        };
        device.launch("assemble.stretch.terms", &terms_args, count)?;
        // The PRODUCT of the two selections, through `vec_add_scaled` into a
        // destination that opens at zero, for the same reason the per-element
        // mass scales go through it.
        // THE SEED, on the device: this array opens at zero and the scale
        // below accumulates into it.
        let seed = VecFillArgs {
            array: state.stretch.ratio.handle(),
            value: 0.0,
            count: faces as u32,
            seam_arena_count: 0,
        };
        device.launch("assemble.stretch.ratio.fill", &seed, faces as u32)?;
        let scale_args_13 = ElementAddScaledArgs {
            source: state.stretch.largest.handle(),
            destination: state.stretch.ratio.handle(),
            scale: state.stretch.shrink_min.handle(),
            stride: 1,
            count,
            seam_arena_count: 0,
        };
        device.launch("assemble.stretch.shrink_scale", &scale_args_13, count)?;
    }
    if rods > 0 {
        let mesh: &[crate::data::Vec2u] = scene::slice(&data.mesh.mesh.edge);
        let props: &[EdgeProp] = scene::slice(&data.prop.edge);
        if props.len() < rods || mesh.len() < rods {
            return Err(Fatal::invariant(format!(
                "solver driver: the stretch indicator walks {rods} rods over {} edge props and {} \
                 edge index records",
                props.len(),
                mesh.len()
            )));
        }
        let vertices = state.sizes.vertices;
        // NO HOST LOOP AND NO REST-LENGTH ARRAY. `rod_stretch_ratio_gated`
        // reads the segment's own record and applies the `fixed` gate there, as
        // the shell half above does. The slot bound is the entry's
        // `vertex_count`, which every backend checks.
        let ratio_args = RodStretchRatioGatedArgs {
            x: curr,
            edge: state.mesh_edge.span(0, 2 * rods),
            // The bound each of the edge's two slots is checked against. The
            // loop above already refuses an out-of-range slot with a message
            // naming the rod; this is the entry point's own check, which every
            // backend runs and which Metal has nothing else standing in for.
            vertex_count: vertices as u32,
            prop: state.prop_edge.handle(),
            ratio: state.stretch.rod_ratio.handle(),
            count: rods as u32,
            seam_arena_count: 0,
        };
        device.launch("assemble.stretch.rod_ratio", &ratio_args, rods as u32)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;
    use ppf_cts_compute::{AllocLabel, Buffer, HostDevice};
    use crate::driver::test_scene::TestScene;
    use crate::cvec::CVec;
    use crate::cvecvec::CVecVec;
    use crate::data::{
        FaceParam, HingeParam, Mat2x2f, Stitch, TetParam, TorqueGroup, TorqueVertex, Vec2u, Vec3u,
        Vec4u, Vec6f, Vec6u,
    };

    /// The registry's `eiganalysis-eps` default, which is the mode-separation
    /// floor the spectral Hessian needs and not a tolerance to widen.
    const EPS: f32 = 1e-2;

    /// Every quantity below is derived from an energy by hand, so the only
    /// error a comparison has to absorb is fp32 round-off through the SVD and
    /// the two converters. Four decimal digits is far inside that and far
    /// outside any wrong-material answer.
    const TOLERANCE: f32 = 1e-4;

    /// One triangle at a stated deformation, with the whole assembly around it.
    ///
    /// THE REST SHAPE IS THE UNIT RIGHT TRIANGLE IN THE XY PLANE, so
    /// `inv_rest2x2` is the identity, `F` is exactly the pair of deformed edge
    /// vectors, and `convert_force`'s three material gradients are
    /// `(-1, -1)`, `(1, 0)` and `(0, 1)`. That is what makes an expected value
    /// derivable here rather than recorded from a run.
    struct Membrane {
        scene: TestScene,
        state: SolverState,
        /// ONE DEVICE FOR THE FIXTURE'S LIFE, on the grounds
        /// [`Bending::device`] states: a handle names an arena of the allocator
        /// that opened it, so sizing on one device and dispatching on another
        /// resolves a handle against a table that never held it.
        device: HostDevice,
    }

    /// The result of one assembly, owned, so the matrix's borrow of the scene
    /// ends before anything is asserted.
    ///
    /// Shape only in `Debug`: a matrix over a real scene carries millions of
    /// floats, and an assertion that printed them would be unreadable.
    #[derive(Debug)]
    struct Assembled {
        /// `3 * vertices` floats: the energy GRADIENT, which is what `force`
        /// carries. `position_step` spells the Newton step `x - dx`, so a
        /// stretched vertex's entry points along the stretch.
        force: Vec<f32>,
        /// The assembled matrix over the gathered vertices, block by block
        /// through the shared read, so the lower triangle comes back
        /// transposed the way the SpMV sees it. Row-major and `width` wide.
        dense: Vec<f32>,
        width: usize,
    }

    impl Assembled {
        fn vertex_force(&self, vertex: usize) -> [f32; 3] {
            [
                self.force[3 * vertex],
                self.force[3 * vertex + 1],
                self.force[3 * vertex + 2],
            ]
        }

        /// `v^T H v`, in `f64` because this is a test reading a result rather
        /// than solver arithmetic: the quantity under test is a sign, and
        /// measuring it in the precision that produced it would fold the
        /// measurement into the answer.
        fn quadratic_form(&self, v: &[f64]) -> f64 {
            assert_eq!(v.len(), self.width);
            let mut total = 0.0;
            for row in 0..self.width {
                for column in 0..self.width {
                    total +=
                        v[row] * f64::from(self.dense[self.width * row + column]) * v[column];
                }
            }
            total
        }
    }

    // ------------------------------------------------------------------
    // Solid (tet) elasticity.
    // ------------------------------------------------------------------

    /// A single tet, assembled through the production driver.
    ///
    /// THIS FIXTURE EXISTS BECAUSE NOTHING CALLED `tet_elastic` AT ALL. Section
    /// 180b measured that: making the tet deposit a no-op left every solver
    /// test passing, so the layer's composition was checked only by a scene.
    /// The four tet tests that predate this one all reach `tet_dispatch_of` or
    /// `refusal::material_defects` and never the driver.
    struct Solid {
        scene: TestScene,
        state: SolverState,
        /// ONE DEVICE FOR THE FIXTURE'S LIFE, on the grounds [`Bending`]
        /// states: a handle names an arena of the allocator that opened it.
        device: HostDevice,
    }

    impl Solid {
        /// The unit corner tet, `(0,0,0) (1,0,0) (0,1,0) (0,0,1)`, whose rest
        /// shape matrix is the identity so `inv_rest3x3` is too.
        fn unit_tet(material: TetParam, mass: f32) -> Self {
            let mut scene = TestScene::new(4).with_tets(&[Vec4u::new(0, 1, 2, 3)]);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.0, 0.0, 0.0);
            scene.place(2, 0.0, 1.0, 0.0);
            scene.place(3, 0.0, 0.0, 1.0);
            scene.data.param_arrays.tet = CVec::from(&[material][..]);
            scene.data.inv_rest3x3 = CVec::from(&[crate::data::Mat3x3f::identity()][..]);
            {
                let props = scene.data.prop.tet.as_mut_slice();
                props[0].mass = mass;
                props[0].volume = 1.0 / 6.0;
            }
            install_pattern(&mut scene.data, 4);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            Self { scene, state, device }
        }

        /// Move vertex 1 along x, which puts `F` at `diag(factor, 1, 1)`.
        fn stretch_to(&mut self, factor: f32) {
            self.scene.place(1, factor, 0.0, 0.0);
        }

        fn tet_prop_mut(&mut self) -> &mut crate::data::TetProp {
            &mut self.scene.data.prop.tet.as_mut_slice()[0]
        }

        fn assemble(&mut self, dt: f32) -> FatalResult<Assembled> {
            let Solid { scene, state, device } = self;
            clear_force(device, state);
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            // Safety: the scene is live, the state was allocated for it, and
            // the matrix borrows the scene's own pattern tables.
            let (force, dense) = unsafe {
                // SEEDED FROM THE SCENE: the fixture deforms it AFTER
                // `allocate` seeded the device, so a device copy would hand the
                // kernel the undeformed shape.
                let pose = crate::driver::state::slice_or_empty(
                    scene.data.vertex.curr.data as *const f32,
                    3 * state.sizes.vertices,
                );
                state.positions.seed(device, pose).expect("committed pose");
                state.eval_x.seed(device, pose).expect("iterate");
                let mut fixed = FixedCsr::adopt_from_dataset(
                    device, state.fixed_pattern_refs(), &scene.data, Default::default())?;
                tet_elastic(device, &scene.data, EPS, state, &mut fixed, dt)?;
                fixed.download(device)?;
                (force_of(device, state), dense_of(&fixed, 4))
            };
            Ok(Assembled { force, dense, width: 12 })
        }
    }

    #[test]
    fn the_tet_slot_replay_deposit_agrees_with_the_row_search() {
        // THE ARM PRODUCTION TAKES. `builder.rs` ships `tet_hess_slots` by
        // default, so a real scene deposits through `push_element_blocks_at`,
        // while every tet fixture builds its `DataSet` directly and leaves the
        // table empty. Measured before this test existed: making that deposit
        // a no-op left all 453 tests passing.
        //
        // A tet's off-diagonal 3x3 blocks are asymmetric, so this case also
        // discriminates a transposed or permuted slot index, which the rod
        // stretch case structurally cannot.
        let material = TetParam { model: Model::Arap, mu: 1000.0, lambda: 500.0,
                                  ..TetParam::default() };
        let mut search = Solid::unit_tet(material, 0.5);
        search.stretch_to(1.25);
        let by_search = search.assemble(0.01).expect("the search path assembles");

        let mut replay = Solid::unit_tet(material, 0.5);
        install_tet_hess_slots(&mut replay.scene.data, 4, &[Vec4u::new(0, 1, 2, 3)]);
        replay.state = SolverState::default();
        // Safety: the scene lives in its box for the whole test.
        unsafe { replay.state.allocate(&mut replay.device, &replay.scene.data) }
            .expect("the fixture scene reallocates");
        replay.stretch_to(1.25);
        let by_replay = replay.assemble(0.01).expect("the replay path assembles");

        for v in 0..4 {
            assert_close(by_replay.vertex_force(v), by_search.vertex_force(v),
                         "the tet slot replay force");
        }
        assert_eq!(by_replay.dense.len(), by_search.dense.len());
        let mut wrote = false;
        for (k, (a, b)) in by_replay.dense.iter().zip(by_search.dense.iter()).enumerate() {
            assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0),
                    "slot {k}: replay {a} against search {b}");
            wrote = wrote || a.abs() > 0.0;
        }
        assert!(wrote, "the replay deposit wrote no block at all");
    }

    #[test]
    fn a_stretched_tet_carries_a_force_and_a_psd_hessian() {
        // THE WHOLE DRIVER, not a kernel: this is the first test that calls
        // `tet_elastic`. A stretched tet must pull back, and the assembled
        // Hessian must be positive semi-definite, which is the property
        // SPD-by-assembly rests on and which a lost PSD projection breaks.
        let material = TetParam { model: Model::Arap, mu: 1000.0, lambda: 500.0,
                                  ..TetParam::default() };
        let mut fixture = Solid::unit_tet(material, 0.5);
        fixture.stretch_to(1.25);
        let assembled = fixture.assemble(0.01).expect("the tet assembles");

        let pulled: f32 = (0..4).map(|v| assembled.vertex_force(v)[0].abs()).sum();
        assert!(pulled > 0.0, "a stretched tet carries no force at all");
        // The elastic force is internal, so it sums to zero over the element.
        for axis in 0..3 {
            let net: f32 = (0..4).map(|v| assembled.vertex_force(v)[axis]).sum();
            assert!(
                net.abs() <= 1e-3 * pulled.max(1.0),
                "axis {axis}: the internal force sums to {net}, not zero"
            );
        }
        // PSD, on the twelve axis directions and a handful of mixed ones. A
        // negative quadratic form here is what `pAp <= 0` reports at run time.
        for k in 0..12usize {
            let mut v = vec![0.0f64; 12];
            v[k] = 1.0;
            assert!(
                assembled.quadratic_form(&v) >= -1e-3,
                "direction {k} has a negative curvature"
            );
        }
    }

    #[test]
    fn a_fixed_tet_carries_no_elastic_energy() {
        // The dispatch's own gate, which the driver applies on the host today
        // and which any conversion to an in-thread gate must preserve
        // exactly.
        let material = TetParam { model: Model::Arap, mu: 1000.0, lambda: 500.0,
                                  ..TetParam::default() };
        let mut fixture = Solid::unit_tet(material, 0.5);
        fixture.tet_prop_mut().fixed = true;
        fixture.stretch_to(1.25);
        let assembled = fixture.assemble(0.01).expect("the tet assembles");
        for v in 0..4 {
            assert_close(assembled.vertex_force(v), [0.0, 0.0, 0.0],
                         "a fixed tet's force");
        }
        assert!(
            assembled.dense.iter().all(|&x| x == 0.0),
            "a fixed tet deposited a Hessian block"
        );
    }

    /// The upper-triangle sparsity over `vertices`, every pair stored, plus its
    /// transpose mirror, built the way `builder.rs` builds them.
    fn install_pattern(data: &mut crate::data::DataSet, vertices: usize) {
        let rows: Vec<Vec<u32>> = (0..vertices)
            .map(|i| ((i as u32)..(vertices as u32)).collect())
            .collect();
        let mut transpose: Vec<Vec<Vec2u>> = vec![Vec::new(); vertices];
        let mut slot = 0u32;
        for (i, row) in rows.iter().enumerate() {
            for &j in row {
                if i as u32 != j {
                    transpose[j as usize].push(Vec2u::new(i as u32, slot));
                }
                slot += 1;
            }
        }
        data.fixed_index_table = CVecVec::from(&rows[..]);
        data.transpose_table = CVecVec::from(&transpose[..]);
    }

    /// The slot table `builder.rs` would ship for a scene whose pattern came
    /// from [`install_pattern`], four per edge in `ii * 2 + jj` order.
    ///
    /// MIRRORS TWO THINGS AND BOTH ARE LOAD-BEARING: the flat value layout,
    /// where row `i` occupies `[row_offset[i], row_offset[i + 1])` and column
    /// `j` sits at `j - i` within it because `install_pattern` gives row `i`
    /// the columns `i..vertices`; and the lower-triangle sentinel, which is
    /// how a block `FixedCSRMat::push` would decline is encoded.
    fn install_edge_hess_slots(
        data: &mut crate::data::DataSet,
        vertices: usize,
        edges: &[Vec2u],
    ) {
        const SENTINEL: u32 = 0xFFFF_FFFF;
        let mut row_offset = vec![0u32; vertices + 1];
        for i in 0..vertices {
            row_offset[i + 1] = row_offset[i] + (vertices - i) as u32;
        }
        let mut slots = Vec::with_capacity(4 * edges.len());
        for edge in edges {
            for ii in 0..2usize {
                for jj in 0..2usize {
                    let row = edge[ii];
                    let column = edge[jj];
                    slots.push(if row > column {
                        SENTINEL
                    } else {
                        row_offset[row as usize] + (column - row)
                    });
                }
            }
        }
        data.edge_hess_slots = CVec::from(&slots[..]);
    }

    /// The rod bending slot table `builder.rs` would ship, KEYED BY VERTEX.
    ///
    /// THE KEY IS THE WHOLE POINT OF THIS HELPER. `builder.rs:1559` allocates
    /// `9 * surface_vert_count` entries and fills the nine for vertex `i` only
    /// when `i` is an interior rod vertex, leaving every other vertex
    /// all-sentinel, because rod bending is dispatched over every surface
    /// vertex and the stencil is tested in the thread. This driver walks a
    /// COMPACTED site list, so `SolverState::allocate` has to repack the table
    /// into site order; a fixture that handed the kernel a site-keyed table
    /// would test the repack against itself and prove nothing.
    fn install_rod_bend_hess_slots(
        data: &mut crate::data::DataSet,
        vertices: usize,
        edges: &[Vec2u],
    ) {
        const SENTINEL: u32 = 0xFFFF_FFFF;
        let mut row_offset = vec![0u32; vertices + 1];
        for i in 0..vertices {
            row_offset[i + 1] = row_offset[i] + (vertices - i) as u32;
        }
        let slot_of = |i: u32, j: u32| -> u32 {
            if i > j {
                SENTINEL
            } else {
                row_offset[i as usize] + (j - i)
            }
        };
        let mut slots = vec![SENTINEL; 9 * vertices];
        for i in 0..vertices as u32 {
            let incident: Vec<&Vec2u> =
                edges.iter().filter(|e| e[0] == i || e[1] == i).collect();
            if incident.len() != 2 {
                continue;
            }
            let other = |e: &Vec2u| if e[0] == i { e[1] } else { e[0] };
            // `[j, i, k]`, the builder's stencil order and the order
            // `rod_bend.node` carries.
            let element = [other(incident[0]), i, other(incident[1])];
            for a in 0..3usize {
                for b in 0..3usize {
                    slots[9 * i as usize + a * 3 + b] = slot_of(element[a], element[b]);
                }
            }
        }
        data.rod_bend_hess_slots = CVec::from(&slots[..]);
    }

    /// The tet slot table `builder.rs` would ship, sixteen per tet.
    ///
    /// KEYED BY TET, which is the element index the dispatch walks, so unlike
    /// the rod bending table this one needs no repack. It is here because the
    /// slot arm is what PRODUCTION takes and no fixture reached it: measured,
    /// removing the tet slot deposit entirely left every solver test passing.
    fn install_tet_hess_slots(
        data: &mut crate::data::DataSet,
        vertices: usize,
        tets: &[Vec4u],
    ) {
        const SENTINEL: u32 = 0xFFFF_FFFF;
        let mut row_offset = vec![0u32; vertices + 1];
        for i in 0..vertices {
            row_offset[i + 1] = row_offset[i] + (vertices - i) as u32;
        }
        let mut slots = Vec::with_capacity(16 * tets.len());
        for tet in tets {
            for ii in 0..4usize {
                for jj in 0..4usize {
                    let (row, column) = (tet[ii], tet[jj]);
                    slots.push(if row > column {
                        SENTINEL
                    } else {
                        row_offset[row as usize] + (column - row)
                    });
                }
            }
        }
        data.tet_hess_slots = CVec::from(&slots[..]);
    }

    /// The hinge slot table `builder.rs` would ship, sixteen per hinge.
    ///
    /// IN THE REMAPPED `(2,1,0,3)` ORDER, which is the whole subtlety.
    /// `dihedral_angle::face_compute_force_hessian` permutes the quadruple
    /// before it reads a position, so the 12x12 Hessian's blocks are in that
    /// order and the table has to be too, or `slot[ii * 4 + jj]` names a
    /// different pair than the block it receives.
    fn install_hinge_hess_slots(
        data: &mut crate::data::DataSet,
        vertices: usize,
        hinges: &[Vec4u],
    ) {
        const SENTINEL: u32 = 0xFFFF_FFFF;
        let mut row_offset = vec![0u32; vertices + 1];
        for i in 0..vertices {
            row_offset[i + 1] = row_offset[i] + (vertices - i) as u32;
        }
        let mut slots = Vec::with_capacity(16 * hinges.len());
        for hinge in hinges {
            let remapped = [hinge[2], hinge[1], hinge[0], hinge[3]];
            for ii in 0..4usize {
                for jj in 0..4usize {
                    let (row, column) = (remapped[ii], remapped[jj]);
                    slots.push(if row > column {
                        SENTINEL
                    } else {
                        row_offset[row as usize] + (column - row)
                    });
                }
            }
        }
        data.hinge_hess_slots = CVec::from(&slots[..]);
    }

    impl Membrane {
        fn unit_triangle(material: FaceParam, mass: f32) -> Self {
            let mut scene = TestScene::new(3).with_faces(&[Vec3u::new(0, 1, 2)]);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.0, 0.0, 0.0);
            scene.place(2, 0.0, 1.0, 0.0);
            scene.data.param_arrays.face = CVec::from(&[material][..]);
            scene.data.inv_rest2x2 = CVec::from(&[Mat2x2f::identity()][..]);
            scene.data.prop.face.as_mut_slice()[0].mass = mass;
            install_pattern(&mut scene.data, 3);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            Self { scene, state, device }
        }

        /// Move vertex 1 along x, which puts `F` at `diag(factor, 1)`.
        ///
        /// The two factors used below, 1.25 and 0.5, are dyadic, so they are
        /// exactly representable and exact in fp32, and the singular values are
        /// the factors and not a quantization of them.
        fn scale_along_x(&mut self, factor: f32) {
            self.scene.place(1, factor, 0.0, 0.0);
            // The scene moved, so the device positions must move with it.
            crate::driver::state::reseed_positions(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
        }

        fn assemble(&mut self, dt: f32) -> FatalResult<Assembled> {
            self.assemble_faces(dt, 3)
        }

        fn assemble_faces(&mut self, dt: f32, gathered: usize) -> FatalResult<Assembled> {
            let Membrane { scene, state, device } = self;
            clear_force(device, state);
            // `positions` IS the start-of-step pose, so it tracks the scene at
            // every fixture entry. A test that wants the iterate somewhere else
            // seeds `eval_x` separately below; one that does not gets both at
            // the scene, which is what a step would see.
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            // Safety: the scene is live, the state was allocated for it, and the
            // matrix borrows the scene's own pattern tables.
            let (force, dense) = unsafe {
                // SEEDED FROM THE SCENE, NOT COPIED FROM `positions`. The
                // fixture deforms the scene AFTER `allocate` seeded the device
                // buffers, so `positions` still holds the build-time pose and a
                // device copy would hand the kernel the undeformed shape.
                //
                // Safety: the scene is live and holds `vertices` triples.
                let pose = crate::driver::state::slice_or_empty(
                    scene.data.vertex.curr.data as *const f32,
                    3 * state.sizes.vertices,
                );
                state
                    .positions
                    .seed(device, pose)
                    .expect("the fixture seeds the committed pose");
                state
                    .eval_x
                    .seed(device, pose)
                    .expect("the fixture seeds the iterate");
                let mut fixed = FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default())?;
                shell_membrane(device, &scene.data, EPS, state, &mut fixed, dt)?;
                // THE MATRIX IS DEVICE-RESIDENT, so the dense reference below reads
                // the mirror and the mirror is stale the moment the assembly
                // dispatch names the buffer. The download costs a copy and buys
                // the read its guarantee.
                fixed.download(device)?;
                let width = 3 * gathered;
                let mut dense = vec![0.0f32; width * width];
                for i in 0..gathered {
                    for j in 0..gathered {
                        let block = fixed.read(i as u32, j as u32);
                        for r in 0..3 {
                            for c in 0..3 {
                                dense[width * (3 * i + r) + 3 * j + c] = block[3 * c + r];
                            }
                        }
                    }
                }
                (force_of(device, state), dense)
            };
            Ok(Assembled {
                force,
                dense,
                width: 3 * gathered,
            })
        }
    }

    fn material(model: Model, mu: f32, lambda: f32) -> FaceParam {
        FaceParam {
            model,
            mu,
            lambda,
            ..FaceParam::default()
        }
    }

    fn assert_close(actual: [f32; 3], expected: [f32; 3], what: &str) {
        let scale = expected
            .iter()
            .fold(1.0f32, |acc, value| acc.max(value.abs()));
        for k in 0..3 {
            assert!(
                (actual[k] - expected[k]).abs() <= TOLERANCE * scale,
                "{what}: component {k} is {} and the derived value is {} (whole vector {actual:?} \
                 against {expected:?})",
                actual[k],
                expected[k]
            );
        }
    }


    /// A cross-stitch fixture: the scene, the state allocated for it, and the
    /// full upper-triangle sparsity so no coupling can be declined for a reason
    /// other than the one under test.
    ///
    /// ONE VERTEX MATERIAL FOR THE WHOLE SCENE, because `param_index` defaults
    /// to zero on every `VertexProp` and the stitch's rest length is the
    /// barycentric mean of the six slots' contact gaps: a per-slot gap would
    /// make the expected value depend on which slot a weight landed on, which
    /// is a second thing to get wrong in a test that is checking the first.
    struct Seam {
        scene: TestScene,
        state: SolverState,
        /// ONE DEVICE FOR THE FIXTURE'S LIFE, on the grounds
        /// [`Bending::device`] states: a handle names an arena of the allocator
        /// that opened it, so sizing on one device and dispatching on another
        /// resolves a handle against a table that never held it.
        device: HostDevice,
    }

    impl Seam {
        fn new(vertices: usize, ghat: f32, offset: f32, record: Stitch) -> Self {
            let mut scene = TestScene::new(vertices);
            scene.data.param_arrays.vertex = CVec::from(
                &[VertexParam {
                    ghat,
                    offset,
                    friction: 0.0,
                }][..],
            );
            scene.data.constraint.stitch = CVec::from(&[record][..]);
            install_pattern(&mut scene.data, vertices);
            let mut state = SolverState::default();
            // Safety: the scene lives in its box for the whole test.
            let mut device = host_device();
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            state
                .stash_stitches(&mut device, &[record])
                .expect("the fixture's stitch set is the one the scene was built with");
            Self { scene, state, device }
        }

        fn place(&mut self, index: usize, x: f32, y: f32, z: f32) {
            self.scene.place(index, x, y, z);
        }

        fn assemble(&mut self, length_factor: f32, gathered: usize) -> FatalResult<Assembled> {
            let Seam { scene, state, device } = self;
            clear_force(device, state);
            // `positions` IS the start-of-step pose, so it tracks the scene at
            // every fixture entry. A test that wants the iterate somewhere else
            // seeds `eval_x` separately below; one that does not gets both at
            // the scene, which is what a step would see.
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            // Safety: the scene is live, the state was allocated for it, and the
            // matrix borrows the scene's own pattern tables.
            let (force, dense) = unsafe {
                // SEEDED FROM THE SCENE, NOT COPIED FROM `positions`. The
                // fixture deforms the scene AFTER `allocate` seeded the device
                // buffers, so `positions` still holds the build-time pose and a
                // device copy would hand the kernel the undeformed shape.
                //
                // Safety: the scene is live and holds `vertices` triples.
                let pose = crate::driver::state::slice_or_empty(
                    scene.data.vertex.curr.data as *const f32,
                    3 * state.sizes.vertices,
                );
                state
                    .positions
                    .seed(device, pose)
                    .expect("the fixture seeds the committed pose");
                state
                    .eval_x
                    .seed(device, pose)
                    .expect("the fixture seeds the iterate");
                let mut fixed = FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default())?;
                stitch(device, &scene.data, length_factor, state, &mut fixed)?;
                // THE MATRIX IS DEVICE-RESIDENT, so the dense reference below reads
                // the mirror and the mirror is stale the moment the assembly
                // dispatch names the buffer. The download costs a copy and buys
                // the read its guarantee.
                fixed.download(device)?;
                let width = 3 * gathered;
                let mut dense = vec![0.0f32; width * width];
                for i in 0..gathered {
                    for j in 0..gathered {
                        let block = fixed.read(i as u32, j as u32);
                        for r in 0..3 {
                            for c in 0..3 {
                                dense[width * (3 * i + r) + 3 * j + c] = block[3 * c + r];
                            }
                        }
                    }
                }
                (force_of(device, state), dense)
            };
            Ok(Assembled {
                force,
                dense,
                width: 3 * gathered,
            })
        }
    }

    /// A stitch record from six slot indices, six weights and a stiffness.
    fn seam(index: [u32; 6], weight: [f32; 6], stiffness: f32) -> Stitch {
        Stitch {
            index: Vec6u::from_row_slice(&index),
            weight: Vec6f::from_row_slice(&weight),
            stiffness,
        }
    }

    /// The vertex-to-vertex form: one vertex named three times per endpoint,
    /// with all the weight on the first slot, which is what a stitch between two
    /// non-solid endpoints is authored as.
    fn degenerate(a: u32, b: u32, stiffness: f32) -> Stitch {
        seam(
            [a, a, a, b, b, b],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            stiffness,
        )
    }

    #[test]
    fn a_stretched_stitch_pulls_its_two_endpoints_together() {
        // The vertex-to-vertex form, derived from the body by hand. Both
        // endpoints carry ghat = 0.5 and offset = 0, so the rest length is the
        // barycentric mean of the six gaps halved, l0 = (0.5 + 0.5) / 2 = 0.5,
        // and with length-factor 4 the cap is 4 * 0.5 = 2, which the measured
        // length of 1 does not reach.
        //
        //   t = x(a) - x(b) = (-1, 0, 0),  l = 1,  n = t / l = (-1, 0, 0)
        //   dedt = (l / l0 - 1) n = (1 / 0.5 - 1) n = (-1, 0, 0)
        //
        // The gradient's six columns are w0..w2 dedt and -w3..-w5 dedt, times
        // the stiffness of 3, and the scatter folds the three repeats of each
        // endpoint onto that endpoint's own force.
        let mut fixture = Seam::new(2, 0.5, 0.0, degenerate(0, 1, 3.0));
        fixture.place(0, 0.0, 0.0, 0.0);
        fixture.place(1, 1.0, 0.0, 0.0);

        let assembled = fixture.assemble(4.0, 2).expect("the stitch assembles");
        assert_close(assembled.vertex_force(0), [-3.0, 0.0, 0.0], "endpoint a");
        assert_close(assembled.vertex_force(1), [3.0, 0.0, 0.0], "endpoint b");

        // The Hessian, from the same derivation:
        //   r  = (l - l0) / l = 0.5
        //   c0 = max(0, 1 - r) / l0 = 1,  c1 = max(0, r / l0) = 1
        //   H  = c0 g g^T + c1 dtdx^T dtdx, so the (a, a) block is
        //        diag(1, 0, 0) + I = diag(2, 1, 1), times the stiffness of 3.
        let expect = |row: usize, column: usize, value: f32| {
            let actual = assembled.dense[assembled.width * row + column];
            assert!(
                (actual - value).abs() <= TOLERANCE * value.abs().max(1.0),
                "matrix entry ({row}, {column}) is {actual} and the derived value is {value}"
            );
        };
        expect(0, 0, 6.0);
        expect(1, 1, 3.0);
        expect(2, 2, 3.0);
        expect(0, 3, -6.0);
        expect(1, 4, -3.0);
        expect(3, 0, -6.0);
        expect(3, 3, 6.0);
        expect(0, 1, 0.0);
    }

    #[test]
    fn a_stitch_pulls_on_its_barycentric_point_and_not_on_its_first_slot() {
        // The whole reason a stitch carries six slots. The source endpoint is
        // the midpoint of vertices 0 and 1 (weights 0.5, 0.5, 0), the target is
        // vertex 3, and vertex 2 sits on the source triangle with weight zero.
        // Every gap is 0.5, so l0 is again 0.5:
        //   (0.5 (0.5) + 0.5 (0.5) + 0 + 1 (0.5)) / 2 = 0.5
        //
        //   source = (0, 1, 0),  target = (1, 1, 0),  t = (-1, 0, 0),  l = 1
        //   dedt = (1 / 0.5 - 1) n = (-1, 0, 0)
        //
        // With stiffness 3 the source's pull is SPLIT by the weights, half to
        // vertex 0 and half to vertex 1, and vertex 2 takes none. A walk that
        // read the endpoint off slot 0 alone would put all of it on vertex 0.
        let record = seam(
            [0, 1, 2, 3, 3, 3],
            [0.5, 0.5, 0.0, 1.0, 0.0, 0.0],
            3.0,
        );
        let mut fixture = Seam::new(4, 0.5, 0.0, record);
        fixture.place(0, 0.0, 0.0, 0.0);
        fixture.place(1, 0.0, 2.0, 0.0);
        fixture.place(2, 0.0, 0.0, 2.0);
        fixture.place(3, 1.0, 1.0, 0.0);

        let assembled = fixture.assemble(4.0, 4).expect("the stitch assembles");
        assert_close(assembled.vertex_force(0), [-1.5, 0.0, 0.0], "source slot 0");
        assert_close(assembled.vertex_force(1), [-1.5, 0.0, 0.0], "source slot 1");
        assert_close(assembled.vertex_force(2), [0.0, 0.0, 0.0], "source slot 2");
        assert_close(assembled.vertex_force(3), [3.0, 0.0, 0.0], "target");
    }

    #[test]
    fn a_stitch_stretched_past_its_length_cap_saturates() {
        // The cap enters the body as a MINIMUM on the measured length, not as a
        // branch, and it is what stops a stitch across a wide gap from pulling
        // without bound. Gap 0.5 gives l0 = 0.5, length-factor 1.5 gives a cap
        // of 0.75, and the endpoints are 2 apart:
        //   l = min(0.75, 2) = 0.75,  n = t / l = (-8/3, 0, 0)
        //   dedt = (0.75 / 0.5 - 1) n = (-4/3, 0, 0)
        // Uncapped the same configuration would read n = (-1, 0, 0) and
        // dedt = (-3, 0, 0), so the two answers are not close.
        let mut fixture = Seam::new(2, 0.5, 0.0, degenerate(0, 1, 1.0));
        fixture.place(0, 0.0, 0.0, 0.0);
        fixture.place(1, 2.0, 0.0, 0.0);

        let assembled = fixture.assemble(1.5, 2).expect("the stitch assembles");
        let capped = 4.0 / 3.0;
        assert_close(assembled.vertex_force(0), [-capped, 0.0, 0.0], "endpoint a");
        assert_close(assembled.vertex_force(1), [capped, 0.0, 0.0], "endpoint b");
    }

    #[test]
    fn the_stitch_hessian_is_positive_semidefinite_and_blind_to_translation() {
        // SPD-by-assembly, at the one place this term could lose it. Both of the
        // body's terms are PSD forms and the two coefficients scaling them are
        // clamped at zero inside it, so the assembled block set can only fail
        // this test by being scattered wrong: a block placed at the wrong
        // (row, column) breaks the symmetry the form needs.
        //
        // Translation is checked beside it because it is the null direction a
        // spring must have, and a sign error in one endpoint's column would
        // leave the matrix PSD while making a rigid shift cost energy.
        let mut fixture = Seam::new(2, 0.5, 0.0, degenerate(0, 1, 3.0));
        fixture.place(0, 0.0, 0.0, 0.0);
        fixture.place(1, 1.0, 0.0, 0.0);
        let assembled = fixture.assemble(4.0, 2).expect("the stitch assembles");

        for direction in 0..3usize {
            let mut shift = vec![0.0f64; 6];
            shift[direction] = 1.0;
            shift[3 + direction] = 1.0;
            let energy = assembled.quadratic_form(&shift);
            assert!(
                energy.abs() < 1e-5,
                "a rigid shift along axis {direction} costs {energy}, and a stitch cannot resist \
                 one"
            );
        }
        // A spread of probe directions, including ones that mix the two
        // endpoints and the three axes, so a single wrong block cannot hide.
        let probes: [[f64; 6]; 5] = [
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
            [1.0, -2.0, 3.0, 0.5, 0.25, -1.5],
            [-1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ];
        for probe in probes {
            let energy = assembled.quadratic_form(&probe);
            assert!(
                energy >= -1e-5,
                "the stitch Hessian is indefinite along {probe:?}, which reads {energy}"
            );
        }
    }

    #[test]
    fn a_stitch_naming_a_vertex_the_scene_does_not_have_stops_the_step() {
        // The gather runs over the six slots before anything is evaluated, so
        // the failure names the stitch and the slot rather than trapping inside
        // a shared body with a raw index.
        let mut fixture = Seam::new(2, 0.5, 0.0, degenerate(0, 1, 3.0));
        fixture.place(0, 0.0, 0.0, 0.0);
        fixture.place(1, 1.0, 0.0, 0.0);
        fixture.state.stitch.index.at()[3] = 7;

        let failure = fixture
            .assemble(4.0, 2)
            .expect_err("a slot outside the vertex range must stop the step");
        let text = format!("{failure:?}");
        assert!(
            text.contains("slot 3") && text.contains("vertex 7"),
            "the refusal must name the stitch's slot and the vertex, got {text}"
        );
    }

    #[test]
    fn a_stitch_coupling_a_pair_the_sparsity_has_no_slot_for_stops_the_step() {
        // THE FAILURE THIS BACKEND MUST NOT INHERIT. `FixedCSRMat::push` returns
        // false on a block the pattern does not carry and its CUDA callers
        // ignore the verdict, so a missing slot silently drops a coupling and
        // leaves the Newton matrix indefinite. Here it stops the step by name.
        let record = degenerate(0, 1, 3.0);
        let mut scene = TestScene::new(2);
        scene.data.param_arrays.vertex = CVec::from(
            &[VertexParam {
                ghat: 0.5,
                offset: 0.0,
                friction: 0.0,
            }][..],
        );
        scene.data.constraint.stitch = CVec::from(&[record][..]);
        // The DIAGONAL ONLY, which is a pattern the (0, 1) coupling is absent
        // from. `install_pattern` would have carried it.
        let rows: Vec<Vec<u32>> = (0..2u32).map(|i| vec![i]).collect();
        scene.data.fixed_index_table = CVecVec::from(&rows[..]);
        scene.data.transpose_table = CVecVec::from(&vec![Vec::<Vec2u>::new(); 2][..]);
        scene.place(0, 0.0, 0.0, 0.0);
        scene.place(1, 1.0, 0.0, 0.0);

        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        let failure = unsafe {
            state
                .allocate(&mut device, &scene.data)
                .expect("the fixture scene allocates");
            state
                .stash_stitches(&mut device, &[record])
                .expect("one stitch, as built");
            // SEEDED FROM THE SCENE, NOT COPIED FROM `positions`. The
                // fixture deforms the scene AFTER `allocate` seeded the device
                // buffers, so `positions` still holds the build-time pose and a
                // device copy would hand the kernel the undeformed shape.
                //
                // Safety: the scene is live and holds `vertices` triples.
                let pose = crate::driver::state::slice_or_empty(
                    scene.data.vertex.curr.data as *const f32,
                    3 * state.sizes.vertices,
                );
                state
                    .positions
                    .seed(&mut device, pose)
                    .expect("the fixture seeds the committed pose");
                state
                    .eval_x
                    .seed(&mut device, pose)
                    .expect("the fixture seeds the iterate");
            let mut fixed =
                FixedCsr::adopt_from_dataset(&mut device, state.fixed_pattern_refs(), &scene.data, Default::default()).expect("a diagonal matrix");
            stitch(&mut device, &scene.data, 4.0, &mut state, &mut fixed)
                .expect_err("a dropped coupling must stop the step")
        };
        let text = format!("{failure:?}");
        assert!(
            text.contains("(0, 1)"),
            "the refusal must name the block that was dropped, got {text}"
        );
    }

    #[test]
    fn a_stitch_with_a_negative_stiffness_stops_the_run() {
        // The spring's Hessian is PSD out of the shared body and this factor
        // multiplies the whole block, so a negative one flips it and the Newton
        // matrix is indefinite. Refused where the constraint arrives, so the
        // message names the stitch instead of surfacing as a `pAp <= 0` abort
        // in the linear solve.
        let fixture = Seam::new(2, 0.5, 0.0, degenerate(0, 1, 3.0));
        let Seam { mut state, mut device, .. } = fixture;
        let failure = state
            .stash_stitches(&mut device, &[degenerate(0, 1, -1.0)])
            .expect_err("a negative stitch stiffness must be refused");
        let text = format!("{failure:?}");
        assert!(
            text.contains("-1"),
            "the refusal must name the stiffness it found, got {text}"
        );
    }

    #[test]
    fn a_step_bringing_a_different_number_of_stitches_stops_the_run() {
        // The fixed sparsity registers each stitch's thirty-six index pairs at
        // scene build, so a constraint carrying a set of another size does not
        // describe this matrix. Caught where the constraint arrives, so the
        // message names the two counts.
        let record = degenerate(0, 1, 3.0);
        let fixture = Seam::new(2, 0.5, 0.0, record);
        let Seam { mut state, mut device, .. } = fixture;
        let failure = state
            .stash_stitches(&mut device, &[record, record])
            .expect_err("a larger stitch set must be refused");
        let text = format!("{failure:?}");
        assert!(
            text.contains('2') && text.contains('1'),
            "the refusal must name both counts, got {text}"
        );
    }

    /// Rayleigh damping must NOT scale the pressure term.
    ///
    /// The body runs elasticity and pressure as SIBLING blocks over one face,
    /// with the stiffness damping applied inside the `mu > 0` block on the
    /// elastic accumulators alone; the pressure block below it embeds its own
    /// gradient and Hessian separately, undamped.
    ///
    /// THIS BACKEND CAN GET IT WRONG BY ORDERING ALONE, because it accumulates
    /// both terms into one pair of per-face buffers and damps them in a stage of
    /// its own. Measured with the dispatch in the wrong place: the pressure
    /// Hessian came out about 76x its correct value, which is `1 + beta/dt` for
    /// this fixture.
    ///
    /// THE COMPARISON IS A DOUBLE DIFFERENCE, and it has to be. Damping SHOULD
    /// scale the elastic Hessian, so a damped face and an undamped one differ
    /// for a correct reason and comparing their totals proves nothing. What
    /// must match is the contribution PRESSURE makes: assemble with and without
    /// pressure at each damping setting, and the two differences are the
    /// pressure block alone, with everything elastic canceled out of both.
    #[test]
    fn rayleigh_damping_does_not_scale_the_pressure_term() {
        // The pressure block, isolated at one damping setting.
        let contribution = |damping: f32| {
            let mut bare = material(Model::Arap, 100.0, 50.0);
            bare.deform_damping = damping;
            let mut inflated = bare;
            inflated.pressure = 250.0;
            let mut without = Membrane::unit_triangle(bare, 2.0);
            let flat = without.assemble(0.01).expect("the bare face assembles");
            let mut with = Membrane::unit_triangle(inflated, 2.0);
            let full = with.assemble(0.01).expect("the inflated face assembles");
            let hessian: Vec<f32> = full
                .dense
                .iter()
                .zip(flat.dense.iter())
                .map(|(a, b)| a - b)
                .collect();
            let force: Vec<f32> = full
                .force
                .iter()
                .zip(flat.force.iter())
                .map(|(a, b)| a - b)
                .collect();
            (force, hessian)
        };

        let (quiet_force, quiet_hessian) = contribution(0.0);
        let (loud_force, loud_hessian) = contribution(0.75);

        let scale = quiet_hessian
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        assert!(
            scale > 1.0e-4,
            "the fixture must carry a pressure Hessian to compare: {quiet_hessian:?}"
        );
        for (k, (a, b)) in quiet_hessian.iter().zip(loud_hessian.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1.0e-3 * scale,
                "entry {k}: damping scaled the pressure Hessian, {a} against \
                 {b}. `assemble.face.pressure` must be dispatched AFTER \
                 `assemble.face.damping`, not before"
            );
        }
        let force_scale = quiet_force
            .iter()
            .fold(1.0f32, |acc, value| acc.max(value.abs()));
        for (k, (a, b)) in quiet_force.iter().zip(loud_force.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1.0e-3 * force_scale,
                "component {k}: damping changed the pressure force, {a} against {b}"
            );
        }
    }

    /// A pressurized face carries a force its unpressurized twin does not.
    ///
    /// THE SUBJECT IS THE WIRING, not the arithmetic. `face_pressure_gradient`
    /// is a shared body with its own conditioning, tested where it is defined;
    /// what is new is that a dispatch reaches it, that the per-face pressure
    /// arrives through the deduplicated material table, and that the result
    /// lands in the same accumulator the elastic term uses. A term computed
    /// correctly and added to nothing would pass every check but this one.
    ///
    /// THE REST SHAPE IS THE FIXTURE ON PURPOSE, because the membrane force
    /// vanishes there for every model: whatever this face carries is the
    /// pressure term alone, with nothing to subtract.
    #[test]
    fn a_pressurized_face_carries_a_force_its_unpressurized_twin_does_not() {
        let quiet = material(Model::Arap, 100.0, 50.0);
        let mut pressurized = quiet;
        pressurized.pressure = 250.0;

        let mut without = Membrane::unit_triangle(quiet, 2.0);
        let calm = without.assemble(0.01).expect("the rest shape assembles");
        for vertex in 0..3 {
            assert_close(
                [
                    calm.force[3 * vertex],
                    calm.force[3 * vertex + 1],
                    calm.force[3 * vertex + 2],
                ],
                [0.0, 0.0, 0.0],
                "an unpressurized face at its rest shape",
            );
        }

        let mut with = Membrane::unit_triangle(pressurized, 2.0);
        let inflated = with.assemble(0.01).expect("the pressurized face assembles");
        let magnitude: f32 = inflated
            .force
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        assert!(
            magnitude > 1.0e-4,
            "a face with pressure 250 carried no force: {:?}. The term is \
             `assemble.face.pressure`, and a zero here is the dispatch not \
             reaching the accumulator rather than the body being wrong",
            inflated.force
        );

        // AND IT IS THE PRESSURE THAT DID IT, not the fixture: the same face
        // with the pressure removed is the calm one above, so the difference
        // isolates the term.
        assert!(
            inflated.force != calm.force,
            "the pressurized and unpressurized faces carried the same force"
        );
    }

    /// A face that is inflated but carries NO elastic stiffness still assembles
    /// its pressure.
    ///
    /// The body runs `mu > 0` and `pressure > 0` as sibling blocks, so the
    /// two terms are independent and a stiffness-free membrane is a supported
    /// scene rather than a degenerate one. THE FAILURE THIS GUARDS IS SILENT:
    /// an active-face list admitting only `mu > 0` leaves such a face computing
    /// a correct pressure gradient into a slot that no scatter reads, so every
    /// kernel is right, every buffer holds the right bytes, and the face does
    /// not move.
    #[test]
    fn an_inflated_face_with_no_stiffness_still_assembles_its_pressure() {
        let mut limp = material(Model::Arap, 0.0, 0.0);
        limp.pressure = 250.0;
        let mut fixture = Membrane::unit_triangle(limp, 2.0);
        let assembled = fixture
            .assemble(0.01)
            .expect("a stiffness-free inflated face assembles");
        let magnitude: f32 = assembled
            .force
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        assert!(
            magnitude > 1.0e-4,
            "an inflated face with mu = 0 carried no force: {:?}",
            assembled.force
        );
    }

    #[test]
    fn a_face_at_its_rest_shape_carries_no_membrane_force() {
        // Every one of the four shell models is stress free at `F = I`, and
        // each reaches that answer differently: the three diff-table models
        // through a table that vanishes at `sigma = (1, 1)`, BaraffWitkin
        // through a stretch term that vanishes at unit column norms and a shear
        // term that vanishes at orthogonal columns. A model wired to the wrong
        // arm would still pass this one, which is why the two tests below carry
        // the derived magnitudes.
        for model in [
            Model::Arap,
            Model::StVK,
            Model::BaraffWitkin,
            Model::SNHk,
        ] {
            let mut fixture = Membrane::unit_triangle(material(model, 100.0, 50.0), 2.0);
            let assembled = fixture.assemble(0.01).expect("the rest shape assembles");
            for vertex in 0..3 {
                assert_close(
                    assembled.vertex_force(vertex),
                    [0.0, 0.0, 0.0],
                    &format!("{model:?} at rest, vertex {vertex}"),
                );
            }
        }
    }

    #[test]
    fn a_stretched_corotated_face_matches_the_gradient_derived_from_its_energy() {
        // Corotated linear (the namespace keeps the ARAP name), 2D form:
        //   E(a) = mu ((a0 - 1)^2 + (a1 - 1)^2) + lmd/2 (a0 + a1 - 2)^2
        // At F = diag(5/4, 1) the singular values are (5/4, 1), so
        //   dE/da = (2 mu (1/4) + lmd (1/4), lmd (1/4)) = (62.5, 12.5)
        // with mu = 100 and lmd = 50. U and V are the identity here, so
        // dE/dF is diag(62.5, 12.5), and convert_force's three gradients
        // (-1, -1), (1, 0) and (0, 1) carry it to the vertices. Times the
        // face's mass of 2:
        //   v0 = (-125, -25, 0)   v1 = (125, 0, 0)   v2 = (0, 25, 0)
        let mut fixture = Membrane::unit_triangle(material(Model::Arap, 100.0, 50.0), 2.0);
        fixture.scale_along_x(1.25);
        let assembled = fixture.assemble(0.01).expect("the stretched face assembles");

        assert_close(assembled.vertex_force(0), [-125.0, -25.0, 0.0], "vertex 0");
        assert_close(assembled.vertex_force(1), [125.0, 0.0, 0.0], "vertex 1");
        assert_close(assembled.vertex_force(2), [0.0, 25.0, 0.0], "vertex 2");

        // The energy is translation invariant, so its gradient sums to zero
        // over the face. Exact rather than approximate here, because
        // convert_force builds the first vertex's gradient as the negated sum
        // of the other two.
        for axis in 0..3 {
            let net: f32 = (0..3).map(|v| assembled.force[3 * v + axis]).sum();
            assert!(
                net.abs() <= TOLERANCE * 125.0,
                "the membrane gradient must sum to zero on axis {axis}, got {net}"
            );
        }
    }

    #[test]
    fn baraffwitkin_is_assembled_from_its_own_arm_and_not_from_the_diff_table() {
        // THE TRAP THIS TEST EXISTS FOR. BaraffWitkin is not a diff-table
        // model: it goes from F to the gradient directly and never forms an
        // SVD. A dispatch that folded it into the spectral arm would hand it
        // the zero table and assemble NO membrane at all, and one that let it
        // fall through to SNHk would assemble a different material. Both
        // outcomes complete and look plausible; the derived value separates
        // them from the right answer.
        //
        // At F = diag(5/4, 1) the stretch term is
        //   mu [ fu (|fu| - 1) / |fu| , fv (|fv| - 1) / |fv| ]
        // whose second column vanishes at a unit column norm, leaving
        // dE/dF = diag(mu/4, 0) = diag(25, 0); the shear term vanishes because
        // the columns are orthogonal. Times the mass of 2:
        //   v0 = (-50, 0, 0)   v1 = (50, 0, 0)   v2 = (0, 0, 0)
        let mut fixture =
            Membrane::unit_triangle(material(Model::BaraffWitkin, 100.0, 50.0), 2.0);
        fixture.scale_along_x(1.25);
        let assembled = fixture.assemble(0.01).expect("the stretched face assembles");

        assert_close(assembled.vertex_force(0), [-50.0, 0.0, 0.0], "vertex 0");
        assert_close(assembled.vertex_force(1), [50.0, 0.0, 0.0], "vertex 1");
        assert_close(assembled.vertex_force(2), [0.0, 0.0, 0.0], "vertex 2");

        // And it is a DIFFERENT answer from the spectral arm's on the same
        // geometry and the same constants, which is what makes the assertion
        // above a dispatch test rather than a magnitude test.
        let mut corotated = Membrane::unit_triangle(material(Model::Arap, 100.0, 50.0), 2.0);
        corotated.scale_along_x(1.25);
        let other = corotated.assemble(0.01).expect("assembles");
        assert!(
            (other.vertex_force(1)[0] - assembled.vertex_force(1)[0]).abs() > 1.0,
            "the two arms must not agree on this face, or the test cannot tell them apart"
        );
    }

    #[test]
    fn a_pure_shear_reaches_the_baraffwitkin_shear_term_alone() {
        // BaraffWitkin is TWO terms and the test above exercises one of them.
        // Placing vertex 2 on the 3-4-5 triangle's short leg keeps BOTH column
        // norms at 1, so the stretch term vanishes on each and what is left is
        // the shear alone: with fu = (1, 0, 0) and fv = (3/5, 4/5, 0) their dot
        // product is 3/5, and
        //   dE/dF = lmd (fu . fv) [ fv , fu ] = 50 (3/5) [ fv , fu ]
        // Carried to the vertices by convert_force's (-1, -1), (1, 0), (0, 1)
        // and scaled by the mass of 2:
        //   v1 = 2 (30) fv = (36, 48, 0), v2 = 2 (30) fu = (60, 0, 0),
        //   v0 = -(v1 + v2) = (-96, -48, 0)
        let mut fixture =
            Membrane::unit_triangle(material(Model::BaraffWitkin, 100.0, 50.0), 2.0);
        fixture.scene.place(2, 0.6, 0.8, 0.0);
        let assembled = fixture.assemble(0.01).expect("the sheared face assembles");

        assert_close(assembled.vertex_force(1), [36.0, 48.0, 0.0], "vertex 1");
        assert_close(assembled.vertex_force(2), [60.0, 0.0, 0.0], "vertex 2");
        assert_close(assembled.vertex_force(0), [-96.0, -48.0, 0.0], "vertex 0");
    }

    #[test]
    fn the_projected_hessian_is_positive_semidefinite_under_compression() {
        // SPD BY ASSEMBLY, AT THE CONFIGURATION WHERE IT IS NOT FREE. Under
        // compression the analytic eigensystem's rotation modes carry negative
        // eigenvalues: at sigma = (1, 1/2) with mu = 100 and lmd = 50 the
        // corotated table gives dE/da = (-25, -125), so the twist mode reads
        // (deda0 + deda1) / (s0 + s1) = -100 and the two scaling modes -25 and
        // -250. Three of the six modes are clamped to zero here, against none
        // at rest and none under tension, which is why this is the
        // configuration the projection has to be checked at.
        let mut fixture = Membrane::unit_triangle(material(Model::Arap, 100.0, 50.0), 2.0);
        fixture.scale_along_x(0.5);
        let assembled = fixture.assemble(0.01).expect("the compressed face assembles");

        let magnitude: f64 = assembled
            .dense
            .iter()
            .fold(0.0f64, |acc, value| acc.max(f64::from(value.abs())));
        assert!(magnitude > 1.0, "the fixture must assemble a nonzero Hessian");

        // A deterministic probe set: the nine axes, then 200 vectors from a
        // fixed linear congruential sequence, so the test is reproducible and
        // does not depend on a random seed.
        let mut probes: Vec<[f64; 9]> = Vec::new();
        for axis in 0..9 {
            let mut v = [0.0f64; 9];
            v[axis] = 1.0;
            probes.push(v);
        }
        let mut bits: u64 = 0x2545_F491_4F6C_DD1D;
        for _ in 0..200 {
            let mut v = [0.0f64; 9];
            for slot in v.iter_mut() {
                bits = bits.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *slot = ((bits >> 33) as f64 / f64::from(1u32 << 31)) - 1.0;
            }
            probes.push(v);
        }
        for probe in &probes {
            let form = assembled.quadratic_form(probe);
            assert!(
                form >= -1e-4 * magnitude,
                "the projected membrane Hessian must be positive semidefinite, and this probe \
                 gives {form} against a largest entry of {magnitude}"
            );
        }

        // And it annihilates a rigid translation exactly, which is the other
        // half of the same energy's invariance and is what a wrong
        // convert_hessian or a mis-ordered block extraction would break.
        for axis in 0..3 {
            let mut translation = [0.0f64; 9];
            for vertex in 0..3 {
                translation[3 * vertex + axis] = 1.0;
            }
            let form = assembled.quadratic_form(&translation);
            assert!(
                form.abs() <= 1e-4 * magnitude,
                "a rigid translation must carry no membrane energy, and axis {axis} gives {form}"
            );
        }
    }

    #[test]
    fn rayleigh_damping_scales_the_hessian_and_leaves_the_force_alone_at_a_zero_step() {
        // The damping term reads the iterate and the START of the step. This
        // fixture opens the iterate AT the start of the step, so the
        // displacement is zero and the gradient half of the term is zero,
        // while the Hessian is still multiplied by 1 + beta / dt. With
        // beta = 0.1 and dt = 0.01 that factor is 11 exactly, so the term is
        // visible in the matrix and absent from the right-hand side, which is
        // what says the stage ran at all.
        let mut plain = Membrane::unit_triangle(material(Model::Arap, 100.0, 50.0), 2.0);
        plain.scale_along_x(1.25);
        let undamped = plain.assemble(0.01).expect("assembles");

        let mut damped_material = material(Model::Arap, 100.0, 50.0);
        damped_material.deform_damping = 0.1;
        let mut damped = Membrane::unit_triangle(damped_material, 2.0);
        damped.scale_along_x(1.25);
        let damped = damped.assemble(0.01).expect("assembles");

        assert_close(damped.vertex_force(1), undamped.vertex_force(1), "vertex 1");
        let magnitude: f64 = undamped
            .dense
            .iter()
            .fold(0.0f64, |acc, value| acc.max(f64::from(value.abs())));
        // Without this the comparison below is vacuous: an assembly that ran
        // nothing leaves both matrices at zero, and 11 times zero is zero.
        assert!(
            magnitude > 1.0,
            "the fixture must assemble a nonzero Hessian for the scale factor to be visible"
        );
        for slot in 0..undamped.dense.len() {
            let expected = 11.0 * undamped.dense[slot];
            assert!(
                f64::from((damped.dense[slot] - expected).abs()) <= 1e-4 * magnitude,
                "entry {slot} is {} and 1 + beta/dt times the undamped one is {expected}",
                damped.dense[slot]
            );
        }
    }

    #[test]
    fn a_collider_face_carries_no_membrane_energy() {
        // The third flag of the dispatch gate, and the one that only comes
        // alive for a spring-held STATIC: an exactly pinned collider had its
        // DOF removed and `fixed` already covered it. A collider's shape is
        // held by its pins, so assembling stiffness for it would fight them.
        let mut fixture = Membrane::unit_triangle(material(Model::Arap, 100.0, 50.0), 2.0);
        fixture.scale_along_x(1.25);
        fixture.scene.data.prop.face.as_mut_slice()[0].collider = true;
        let assembled = fixture.assemble(0.01).expect("assembles");
        for vertex in 0..3 {
            assert_close(
                assembled.vertex_force(vertex),
                [0.0, 0.0, 0.0],
                &format!("collider vertex {vertex}"),
            );
        }
    }

    #[test]
    fn the_membrane_walks_the_shell_prefix_and_not_a_solids_surface_triangles() {
        // THE TRAP THE SURVEY NAMES FIRST. `mesh.face` carries a solid's
        // surface triangles after the shell faces, and `inv_rest2x2` is sized
        // over the shell PREFIX alone, so a stage ranged over `face.size` reads
        // the rest matrices past their end on every tetrahedralized scene and
        // is correct on every shell-only one. Here face 1 is a solid's surface
        // triangle on its own three vertices: it must contribute nothing, and
        // the assembly must not reach for a rest matrix it does not have.
        let mut scene = TestScene::new(6)
            .with_faces(&[Vec3u::new(0, 1, 2), Vec3u::new(3, 4, 5)]);
        scene.place(0, 0.0, 0.0, 0.0);
        scene.place(1, 1.25, 0.0, 0.0);
        scene.place(2, 0.0, 1.0, 0.0);
        scene.place(3, 0.0, 0.0, 1.0);
        scene.place(4, 1.25, 0.0, 1.0);
        scene.place(5, 0.0, 1.0, 1.0);
        scene.data.shell_face_count = 1;
        scene.data.param_arrays.face = CVec::from(&[material(Model::Arap, 100.0, 50.0)][..]);
        scene.data.inv_rest2x2 = CVec::from(&[Mat2x2f::identity()][..]);
        for prop in scene.data.prop.face.as_mut_slice() {
            prop.mass = 2.0;
        }
        install_pattern(&mut scene.data, 6);
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }.expect("allocates");
        assert_eq!(state.sizes.faces, 2);
        assert_eq!(state.sizes.shell_faces, 1);

        let mut fixture = Membrane { scene, state, device };
        let assembled = fixture.assemble_faces(0.01, 6).expect("assembles");

        assert_close(assembled.vertex_force(0), [-125.0, -25.0, 0.0], "shell vertex 0");
        assert_close(assembled.vertex_force(1), [125.0, 0.0, 0.0], "shell vertex 1");
        for vertex in 3..6 {
            assert_close(
                assembled.vertex_force(vertex),
                [0.0, 0.0, 0.0],
                &format!("solid surface vertex {vertex}"),
            );
        }
    }

    #[test]
    fn a_shell_prefix_longer_than_the_face_array_stops_the_run() {
        // The other end of the same prefix relation, and the one that is caught
        // at `initialize()` rather than inside a step. Clamping it to fit would
        // leave the membrane covering fewer faces than the scene declares, with
        // nothing in the output naming the ones it dropped.
        let mut scene = TestScene::new(3).with_faces(&[Vec3u::new(0, 1, 2)]);
        scene.data.shell_face_count = 4;
        let mut state = SolverState::default();
        // Safety: the scene lives in its box for the whole test.
        let fatal = unsafe { state.allocate(&mut host_device(), &scene.data) }
            .expect_err("a prefix longer than the array it indexes must stop the run");
        assert!(
            fatal.detail.contains("shell faces") && fatal.detail.contains('4'),
            "the message must name the prefix and the array, got {}",
            fatal.detail
        );
    }

    #[test]
    fn an_inverse_rest_array_short_of_the_shell_prefix_stops_the_run() {
        // The bound the trap above turns on, checked before any range reads it.
        // A `DataSet` whose rest matrices do not cover the shell prefix does
        // not describe this mesh, and reading past them would be a wild read of
        // the host's own memory rather than a fault.
        let mut fixture = Membrane::unit_triangle(material(Model::Arap, 100.0, 50.0), 2.0);
        fixture.scene.data.inv_rest2x2 = CVec::new();
        let fatal = fixture
            .assemble(0.01)
            .expect_err("a rest array shorter than the shell prefix must stop the run");
        assert!(
            fatal.detail.contains("inverse rest"),
            "the message must name what is short, got {}",
            fatal.detail
        );
    }

    #[test]
    fn a_membrane_hessian_block_the_sparsity_cannot_hold_is_reported_rather_than_dropped() {
        // THE GUARANTEE-CLASS CHECK, AND IT MOVED WITH THE ASSEMBLY. While the
        // membrane staged its 9x9 into a host mirror, the driver read
        // `push_stored` back and raised on any block with `row <= column` that
        // found no slot. The layer is one device dispatch now, so the same
        // predicate is a `DIAG_ASSERT4` inside `face_elastic_embed` and reaches
        // the host through the `[[seam::diag]]` lane instead.
        //
        // WHAT IT PREVENTS IS NOT A ROUNDING DIFFERENCE. A block the pattern
        // cannot hold is a LOST HESSIAN COUPLING: the Newton matrix is
        // assembled missing a pair this element's stencil writes, so it is no
        // longer the SPD-by-assembly matrix the PCG guards are entitled to
        // assume. `FixedCSRMat::push`'s CUDA callers ignore that verdict, which
        // is how the rod-bend `(j, k)` stencil bug shipped, masked by damping
        // and a small `dt`.
        //
        // The lower triangle is the OTHER refusal and is not a defect: the
        // matrix stores only `i <= j`. Every other membrane fixture here builds
        // the full upper triangle through `install_pattern` and none of them
        // trips this, which is the paired positive case.
        let mut fixture = Membrane::unit_triangle(material(Model::Arap, 100.0, 50.0), 2.0);
        // THE DIAGONAL ONLY, a pattern the (0, 1) coupling is absent from.
        let rows: Vec<Vec<u32>> = (0..3u32).map(|i| vec![i]).collect();
        fixture.scene.data.fixed_index_table = CVecVec::from(&rows[..]);
        fixture.scene.data.transpose_table = CVecVec::from(&vec![Vec::<Vec2u>::new(); 3][..]);
        fixture.scale_along_x(1.25);
        let fatal = fixture
            .assemble(0.01)
            .expect_err("a dropped coupling must stop the step");
        assert_eq!(
            fatal.code,
            ppf_cts_formats::status::error_code::DEVICE_ASSERT,
            "a failing kernel check is a device assert, not an invariant: {fatal:?}"
        );
        // THE NEUTRAL BODY'S OWN FILE, which is what the lane buys: the check
        // sits in a kernel body all three backends compile, and `__FILE__`
        // there names that body rather than whichever launcher called it.
        assert!(
            fatal.detail.contains("face_force.kernel.cpp"),
            "the channel must carry the file the check is in, got {}",
            fatal.detail
        );
    }

    /// The three verdicts, read straight off the kernel.
    ///
    /// AT THE KERNEL AND NOT THROUGH A SCENE, because an unrecognized model id
    /// cannot be spelled in Rust: `Model` is a `repr(C)` enum and writing a
    /// discriminant outside its five variants is undefined. The record takes a
    /// raw `u32` array, which is exactly the surface a scene built by something
    /// other than this frontend would arrive on.
    fn dispatch_of(model: u32) -> (u32, [f32; 2], [f32; 4]) {
        // THE THREE MATERIAL CONSTANTS ARE DEVICE ALLOCATIONS, so this stages
        // them exactly as the driver does rather than pointing the record at
        // its own stack. The device therefore has to outlive the launch, which
        // is why it is a binding here and was a temporary before.
        let mut device = host_device();
        let mut models = one_element_buffer(&mut device, "test.model", &[model]);
        let mut mu = one_element_buffer(&mut device, "test.mu", &[100.0f32]);
        let mut lambda = one_element_buffer(&mut device, "test.lambda", &[50.0f32]);
        // sigma well away from (1, 1), so a table that was written carries
        // nonzero entries and "zero" means "not written".
        let sigma = [1.5f32, 0.75];
        let mut sigma_in = one_element_buffer(&mut device, "test.sigma", &sigma);
        // THE TABLE'S OWN OUTPUTS ARE DEVICE ALLOCATIONS TOO, and they are
        // staged with a nonzero fill on purpose: every caller of this helper
        // asserts that an unrecognized model leaves the pair ZERO, and a
        // destination that started at zero cannot tell "the kernel wrote zero"
        // from "the kernel wrote nothing".
        let mut gradient_out = one_element_buffer(&mut device, "test.gradient", &[f32::NAN; 2]);
        let mut hessian_out = one_element_buffer(&mut device, "test.hessian", &[f32::NAN; 4]);
        // THE VERDICT IS A DEVICE ALLOCATION TOO, and seeded with a sentinel
        // for the reason the pair above is: the assertions below read it as the
        // kernel's answer, and a destination starting at a real code could not
        // tell that answer from the kernel writing nothing.
        let mut verdict = one_element_buffer(&mut device, "test.verdict", &[255u32]);
        let args = FaceMaterialDiffTableArgs {
            model: models.handle(),
            sigma: sigma_in.handle(),
            mu: mu.handle(),
            lambda: lambda.handle(),
            gradient_sigma: gradient_out.handle(),
            hessian_sigma: hessian_out.handle(),
            dispatch: verdict.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        // Safety: every array is one element wide and the extent is one element.
        unsafe { device.launch("test.face_material_diff_table", &args, 1) }
            .expect("the material table runs");
        let mut gradient = [0.0f32; 2];
        let mut hessian = [0.0f32; 4];
        gradient_out
            .read(&mut device, 0, &mut gradient)
            .expect("the gradient reads back");
        hessian_out
            .read(&mut device, 0, &mut hessian)
            .expect("the Hessian reads back");
        let code = verdict
            .read_one(&mut device, 0)
            .expect("the verdict reads back");
        models.free(&mut device).expect("the model buffer frees");
        mu.free(&mut device).expect("the shear modulus buffer frees");
        lambda.free(&mut device).expect("the Lame buffer frees");
        gradient_out.free(&mut device).expect("the gradient buffer frees");
        hessian_out.free(&mut device).expect("the Hessian buffer frees");
        verdict.free(&mut device).expect("the verdict buffer frees");
        sigma_in.free(&mut device).expect("the sigma buffer frees");
        (code, gradient, hessian)
    }

    /// The solid verdict, read straight off the kernel.
    ///
    /// The tet counterpart of [`dispatch_of`], and it exists for the same
    /// reason: an unrecognized model id cannot be spelled in Rust, and the
    /// record takes a raw `u32`, which is the surface a scene built by
    /// something other than this frontend would arrive on.
    fn tet_dispatch_of(model: u32) -> (u32, [f32; 3], [f32; 9]) {
        let mut device = host_device();
        let mut models = one_element_buffer(&mut device, "test.tet_model", &[model]);
        let mut mu = one_element_buffer(&mut device, "test.tet_mu", &[100.0f32]);
        let mut lambda = one_element_buffer(&mut device, "test.tet_lambda", &[50.0f32]);
        // Well away from (1, 1, 1), so a table that was written is nonzero and
        // "zero" means "not written"; the destinations open at NaN for the same
        // reason the face helper's do.
        let sigma = [1.5f32, 0.75, 1.25];
        let mut sigma_in = one_element_buffer(&mut device, "test.tet_sigma", &sigma);
        let mut gradient_out = one_element_buffer(&mut device, "test.tet_gradient", &[f32::NAN; 3]);
        let mut hessian_out = one_element_buffer(&mut device, "test.tet_hessian", &[f32::NAN; 9]);
        let mut verdict = one_element_buffer(&mut device, "test.tet_verdict", &[255u32]);
        let args = TetMaterialDiffTableArgs {
            model: models.handle(),
            sigma: sigma_in.handle(),
            mu: mu.handle(),
            lambda: lambda.handle(),
            gradient_sigma: gradient_out.handle(),
            hessian_sigma: hessian_out.handle(),
            accepted: verdict.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        // Safety: every array is one element wide and the extent is one element.
        unsafe { device.launch("test.tet_material_diff_table", &args, 1) }
            .expect("the material table runs");
        let mut gradient = [0.0f32; 3];
        let mut hessian = [0.0f32; 9];
        gradient_out
            .read(&mut device, 0, &mut gradient)
            .expect("the gradient reads back");
        hessian_out
            .read(&mut device, 0, &mut hessian)
            .expect("the Hessian reads back");
        let code = verdict
            .read_one(&mut device, 0)
            .expect("the verdict reads back");
        models.free(&mut device).expect("the model buffer frees");
        mu.free(&mut device).expect("the shear modulus buffer frees");
        lambda.free(&mut device).expect("the Lame buffer frees");
        sigma_in.free(&mut device).expect("the sigma buffer frees");
        gradient_out.free(&mut device).expect("the gradient buffer frees");
        hessian_out.free(&mut device).expect("the Hessian buffer frees");
        verdict.free(&mut device).expect("the verdict buffer frees");
        (code, gradient, hessian)
    }

    /// A one-element device allocation holding `values`.
    ///
    /// A test's own staging, and deliberately a plain [`Buffer`] rather than a
    /// `StagedBuffer`: the contents are a literal written once, so there is no
    /// host array for a later pass to dirty.
    /// Clear the Newton right-hand side ON THE DEVICE, as the step's own opening
    /// does. It is a dispatch rather than a host `fill` because the array is a
    /// device allocation now, and its mirror refuses a read that follows a
    /// handle without a download.
    /// A ring of vertices driven by one torque group.
    ///
    /// THE FIXTURE OWNS BOTH DISPATCHES, because the term needs both: the frame
    /// pre-pass over the groups and the momentum row over the vertices. A test
    /// that ran only the second would read whatever the result buffer happened
    /// to hold, which on a fresh arena is zero and looks exactly like a group
    /// commanding no torque.
    struct Torqued {
        scene: TestScene,
        state: SolverState,
        device: HostDevice,
    }

    impl Torqued {
        /// Four vertices on an ellipse in the xy plane, one group turning
        /// about the axis their own covariance picks out.
        ///
        /// PLANAR ON PURPOSE: the principal axis of a flat ring is its normal,
        /// so the commanded rotation is in the plane the members lie in and
        /// every member has a real perpendicular radius. A collinear set would
        /// leave `inv_r_perp_sq_sum` at zero and receive no force, which is a
        /// case worth testing separately but not the one that shows the term
        /// arrives at all.
        ///
        /// ELLIPTICAL RATHER THAN CIRCULAR, and that is not cosmetic. A circle
        /// has covariance `diag(c, c, 0)`, whose two equal eigenvalues leave the
        /// in-plane eigenvectors undetermined; that is the eigensolver's
        /// tie-breaking under test, not the torque wiring, and it is covered
        /// where `torque_sym_eig3x3` is. Unequal radii make all three
        /// eigenvalues distinct, so the axis this fixture asks for is the one
        /// unambiguous answer. The centroid stays at the origin either way,
        /// which keeps the frame easy to reason about.
        fn ring(magnitude: f32) -> Self {
            let mut scene = TestScene::new(4);
            scene.place(0, 2.0, 0.0, 0.0);
            scene.place(1, 0.0, 1.0, 0.0);
            scene.place(2, -2.0, 0.0, 0.0);
            scene.place(3, 0.0, -1.0, 0.0);
            for prop in scene.vertex_props_mut() {
                prop.mass = 1.0;
            }
            scene.data.constraint.torque_groups = CVec::from(
                &[TorqueGroup {
                    // The axis of LEAST extent, which for a planar ring is its
                    // normal: `torque_sym_eig3x3` orders the eigenvectors by
                    // descending eigenvalue, so component 2 is the smallest.
                    axis_component: 2,
                    vertex_start: 0,
                    vertex_count: 4,
                    hint_vertex: 0,
                }][..],
            );
            let members: Vec<TorqueVertex> = (0..4)
                .map(|index| TorqueVertex {
                    magnitude,
                    index,
                    group_id: 0,
                })
                .collect();
            scene.data.constraint.torque_vertices = CVec::from(&members[..]);
            install_pattern(&mut scene.data, 4);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            Self { scene, state, device }
        }

        /// One momentum assembly, torque frame included, returning the gradient.
        fn assemble(&mut self, dt: f32) -> FatalResult<Vec<f32>> {
            let Torqued { scene, state, device } = self;
            clear_force(device, state);
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // THE ITERATE IS SEEDED TOO, and it is not optional here: BOTH
            // torque passes read `eval_x`, the frame because a group's centroid
            // and axis are properties of where the group IS this iteration, and
            // the row because the force is about that centroid. Left at the
            // origin every member coincides with the centroid, the
            // covariance vanishes, `torque_rperp_finish` returns zero and the
            // group receives no force, which reads exactly like a term that is
            // not wired.
            //
            // Safety: the scene is live and holds `vertices` triples.
            unsafe {
                let pose = crate::driver::state::slice_or_empty(
                    scene.data.vertex.curr.data as *const f32,
                    3 * state.sizes.vertices,
                );
                state
                    .eval_x
                    .seed(device, pose)
                    .expect("the fixture seeds the iterate");
            }
            // THE STASH IS WHAT A STEP DOES, and the fixture must do it too:
            // `update_constraint` copies the groups and members to the device
            // every step, and `momentum` reads the stash rather than the scene.
            //
            // THE PINS ARE STASHED EMPTY FOR THE SAME REASON, and the emptiness
            // is not why it is needed: an unstashed `pull` is left at
            // `Handle::NONE`, and a generated entry resolves every buffer it is
            // handed BEFORE the body branches on any count, so the sentinel
            // trips the arena assert rather than being read as "no pins". A
            // zero-length stash is a real allocation and reads as none.
            state.stash_pins(device, &[], &[])?;
            let groups = scene.data.constraint.torque_groups.as_slice().to_vec();
            let members = scene.data.constraint.torque_vertices.as_slice().to_vec();
            state.stash_torque(device, &groups, &members)?;
            // Safety: `ParamSet` is a repr(C) plain-old-data aggregate with no
            // Drop, no references and no niche-optimized fields, so an all-zero
            // bit pattern is a valid inhabitant. Zero is also the quiet setting
            // for every knob the momentum row reads, which is what leaves the
            // torque term the only thing acting.
            let param = unsafe { std::mem::zeroed::<ParamSet>() };
            // Safety: the scene is live and the state was allocated for it.
            unsafe { momentum(device, &scene.data, &param, state, dt) }?;
            Ok(force_of(device, state))
        }
    }

    /// A commanded torque reaches the vertices it names.
    ///
    /// THE SUBJECT IS THE WIRING, as it is for the pressure term: every float
    /// operation here is in `energy/model/torque.kernel.cpp`, tested where it is
    /// defined. What is new is that the frame pre-pass runs, that its result
    /// reaches the momentum row through a buffer rather than a recomputation,
    /// and that the row's contribution lands in the assembled gradient. A term
    /// computed correctly into a buffer nobody reads passes everything else.
    ///
    /// THE COMPARISON IS AGAINST THE TWIN, NOT AGAINST ZERO. The momentum row
    /// is active in both: with the target at the origin and `dt = 0.01` each
    /// unit-mass vertex carries `mass / dt^2` of inertia, which is the right
    /// answer and not the term under test. Differencing the two assemblies
    /// leaves the torque alone.
    #[test]
    fn a_commanded_torque_reaches_the_vertices_it_names() {
        let applied = torque_contribution(10.0);
        let magnitude = applied.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        assert!(
            magnitude > 1.0e-4,
            "a group commanding torque 10 changed nothing: {applied:?}. The \
             term is `assemble.torque.frame` feeding the momentum row, and a \
             zero here is one of the two dispatches not reaching the other"
        );

        // AND A GROUP COMMANDING NOTHING CHANGES NOTHING, which is the other
        // half: without it a fixture that perturbed the assembly some other way
        // would satisfy the assertion above.
        let none = torque_contribution(0.0);
        for (vertex, value) in none.iter().enumerate() {
            assert!(
                value.abs() < 1.0e-6,
                "a group commanding no torque moved component {vertex} by \
                 {value}"
            );
        }
    }

    /// The torque is a COUPLE: it turns the group without pushing it.
    ///
    /// This is the property that separates a wired torque from an arbitrary
    /// force of the right size. `force_i = (axis x r_perp_i) * scale` sums to
    /// zero over the group, so its contribution carries a rotation and no net
    /// translation. A sign error, a wrong axis, or a frame computed about the
    /// origin instead of the centroid all break this while leaving the
    /// magnitude assertion above satisfied.
    #[test]
    fn a_torque_group_carries_no_net_linear_force() {
        let applied = torque_contribution(10.0);
        let mut net = [0.0f32; 3];
        for vertex in 0..4 {
            for k in 0..3 {
                net[k] += applied[3 * vertex + k];
            }
        }
        let scale = applied.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        for k in 0..3 {
            assert!(
                net[k].abs() < 1.0e-4 * scale.max(1.0),
                "component {k} of the net force is {}, against a per-vertex \
                 scale of {scale}: a torque must turn the group without \
                 pushing it",
                net[k]
            );
        }
    }

    /// A CIRCULAR group comes out with a zero axis.
    ///
    /// THIS TEST PINS A DEFECT, ON PURPOSE, AND THE DEFECT IS IN THE SHARED
    /// EIGENSOLVER.
    /// A ring of members equidistant from their centroid has covariance
    /// `diag(c, c, 0)`, whose two equal in-plane eigenvalues leave their
    /// eigenvectors undetermined. The two columns are solved independently and
    /// can return the SAME direction, at which point the Gram-Schmidt pass
    /// produces a zero `e1` and `e2 = e0 x e1` goes to zero with it. The third
    /// eigenvalue is distinct and its eigenvector well determined, so the axis
    /// the group actually asks for is destroyed by the degeneracy of the other
    /// two, and a zero axis crossed with any radius is no force: the group does
    /// not turn, silently.
    ///
    /// `eigen_sym3x3` carries no degenerate guard, and it is a neutral body,
    /// so every backend answers a circular group the same way. What this test
    /// buys is that a repair cannot land quietly: it turns this red, and the
    /// answer has to be changed deliberately rather than drifted into.
    ///
    /// THE CENTROID AND THE NORMALIZATION ARE STILL ASSERTED, because they are
    /// correct and they localize the defect: it is the axis alone that is lost,
    /// not the frame pre-pass generally.
    #[test]
    fn a_circular_group_reproduces_the_reference_zero_axis() {
        let frame = circular_frame();
        let axis = frame.axis;
        let norm =
            (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        assert!(
            norm < 1.0e-6,
            "the reference leaves a degenerate ring's axis at zero; this \
             backend returned {axis:?} of length {norm}. If the reference was \
             repaired, repair this too rather than relaxing the test"
        );
        // FOUR MEMBERS AT UNIT RADIUS, each fully perpendicular to the plane
        // normal, so the summed squared radius is 4 and its reciprocal 0.25.
        // This survives the zero axis because `r - axis * axis.dot(r)` with a
        // zero axis is just `r`, which is what the members' radii already are.
        assert!(
            (frame.inv_r_perp_sq_sum - 0.25).abs() < 1.0e-5,
            "expected 1/4 for four unit radii, got {}",
            frame.inv_r_perp_sq_sum
        );
    }

    /// The frame `torque_group_frame` computes for a unit circle of four
    /// members in the xy plane, read back from the device.
    fn circular_frame() -> crate::data::TorqueGroupResult {
        let mut scene = TestScene::new(4);
        scene.place(0, 1.0, 0.0, 0.0);
        scene.place(1, 0.0, 1.0, 0.0);
        scene.place(2, -1.0, 0.0, 0.0);
        scene.place(3, 0.0, -1.0, 0.0);
        for prop in scene.vertex_props_mut() {
            prop.mass = 1.0;
        }
        scene.data.constraint.torque_groups = CVec::from(
            &[TorqueGroup {
                axis_component: 2,
                vertex_start: 0,
                vertex_count: 4,
                hint_vertex: 0,
            }][..],
        );
        let members: Vec<TorqueVertex> = (0..4)
            .map(|index| TorqueVertex {
                magnitude: 10.0,
                index,
                group_id: 0,
            })
            .collect();
        scene.data.constraint.torque_vertices = CVec::from(&members[..]);
        install_pattern(&mut scene.data, 4);
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }
            .expect("the fixture scene allocates");
        crate::driver::state::reseed_committed(&mut device, &mut state, &scene.data);
        crate::driver::state::reseed_props(&mut device, &mut state, &scene.data);
        // Safety: the scene is live and holds four triples.
        unsafe {
            let pose = crate::driver::state::slice_or_empty(
                scene.data.vertex.curr.data as *const f32,
                12,
            );
            state
                .eval_x
                .seed(&mut device, pose)
                .expect("the fixture seeds the iterate");
        }
        let groups = scene.data.constraint.torque_groups.as_slice().to_vec();
        state
            .stash_torque(&mut device, &groups, &members)
            .expect("the fixture stashes its group");
        // READ BACK THROUGH A BUFFER OF THE TEST'S OWN, because the driver's
        // result buffer is a plain device allocation: nothing in a step ever
        // looks at it, so it carries no host mirror to read.
        let mut out: ppf_cts_compute::ReadbackBuffer<crate::data::TorqueGroupResult> =
            Default::default();
        out.size(&mut device, 1, ppf_cts_compute::AllocLabel("test.torque.frame"))
            .expect("the readback buffer sizes");
        let args = TorqueGroupFrameArgs {
            group: state.torque_group.handle(),
            member: state.torque_vertex.handle(),
            member_count: 4,
            position: state.eval_x.handle(),
            vertex_count: 4,
            prop: state.prop_vertex.handle(),
            result: out.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        // Safety: every handle names a live allocation of the stated length and
        // the extent is the one group.
        unsafe { device.launch("test.torque.frame", &args, 1) }
            .expect("the frame pre-pass dispatches");
        out.download(&mut device).expect("the frame reads back");
        out.host()[0]
    }

    /// The gradient a commanded torque ADDS, isolated by differencing one
    /// assembly against the same scene commanding none.
    fn torque_contribution(magnitude: f32) -> Vec<f32> {
        let mut quiet = Torqued::ring(0.0);
        let calm = quiet.assemble(0.01).expect("an untorqued ring assembles");
        let mut driven = Torqued::ring(magnitude);
        let turned = driven.assemble(0.01).expect("a torqued ring assembles");
        turned
            .iter()
            .zip(calm.iter())
            .map(|(with, without)| with - without)
            .collect()
    }

    fn clear_force(device: &mut HostDevice, state: &mut SolverState) {
        let n = 3 * state.sizes.vertices;
        let args = crate::driver::kernels::VecFillArgs {
            array: state.force.handle(),
            value: 0.0,
            count: n as u32,
            seam_arena_count: 0,
        };
        // Safety: the array is one allocation of `n` floats and the extent is `n`.
        unsafe { device.launch("test.force.fill", &args, n as u32) }
            .expect("the right-hand side clears");
    }

    /// The right-hand side, read back off its mirror.
    fn force_of(device: &mut HostDevice, state: &mut SolverState) -> Vec<f32> {
        state
            .force
            .download(device)
            .expect("the force mirror refreshes");
        state.force.host().to_vec()
    }

    fn one_element_buffer<T: ppf_cts_compute::Pod>(
        device: &mut HostDevice,
        label: &'static str,
        values: &[T],
    ) -> Buffer<T> {
        let mut buffer = Buffer::<T>::none();
        buffer
            .size(device, values.len(), AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer.write(device, 0, values).expect("the test upload succeeds");
        buffer
    }

    #[test]
    fn an_unrecognized_face_model_is_refused_rather_than_assembled_as_snhk() {
        // The usual dispatch shape (`if ARAP ... else if StVK ... else SNHk`)
        // turns a bad id into a silently wrong MATERIAL, which runs to
        // completion and reads as a physics disagreement rather than a defect.
        let (verdict, gradient, hessian) = dispatch_of(9);
        assert_eq!(verdict, face_verdict::UNKNOWN);
        assert_eq!(gradient, [0.0, 0.0]);
        assert_eq!(hessian, [0.0; 4]);
    }

    #[test]
    fn a_pdrd_face_is_admitted_by_the_table_with_no_elastic_energy() {
        // THE MODEL ID IS NOT VALIDATED: the whole term is guarded on
        // `mu > 0`, and a PDRD face carries `mu == 0` because its shape is held
        // by the reduced rigid solve. A table that validated ids would have to
        // admit a non-elastic model anyway, or it would refuse a scene the
        // solver supports.
        let (verdict, gradient, hessian) = dispatch_of(Model::Pdrd as u32);
        assert_eq!(verdict, face_verdict::NO_ENERGY);
        assert_eq!(gradient, [0.0, 0.0]);
        assert_eq!(hessian, [0.0; 4]);
    }

    #[test]
    fn baraffwitkin_reports_that_the_table_is_not_its_material() {
        // The verdict that makes the two-family split checkable. A boolean
        // here would have to call this face "accepted" while handing it a zero
        // table, so a caller that skipped the BaraffWitkin range would assemble
        // a cloth with no membrane and nothing in the output would say so.
        let (verdict, gradient, hessian) = dispatch_of(Model::BaraffWitkin as u32);
        assert_eq!(verdict, face_verdict::BARAFF_WITKIN);
        assert_eq!(gradient, [0.0, 0.0]);
        assert_eq!(hessian, [0.0; 4]);

        // While the three models the table does carry report the other code and
        // a table that was actually written.
        for model in [Model::Arap, Model::StVK, Model::SNHk] {
            let (verdict, gradient, _) = dispatch_of(model as u32);
            assert_eq!(verdict, face_verdict::DIFF_TABLE, "{model:?}");
            assert!(
                gradient.iter().any(|value| *value != 0.0),
                "{model:?} must write a table at a deformed sigma"
            );
        }
    }

    #[test]
    fn an_unrecognized_tet_model_is_refused_rather_than_assembled_as_snhk() {
        // The solid half of the same rule, and it had no case of its own until
        // the gate at `initialize()` began mirroring this verdict: a table the
        // dispatch did not write holds the zero fill, which is not a zero
        // CONTRIBUTION but an unknown material.
        let (verdict, gradient, hessian) = tet_dispatch_of(9);
        assert_eq!(verdict, tet_verdict::UNKNOWN);
        assert_eq!(gradient, [0.0; 3]);
        assert_eq!(hessian, [0.0; 9]);
    }

    #[test]
    fn baraffwitkin_has_no_solid_form_and_the_tet_table_says_so() {
        // A MEMBRANE MODEL MAPS TWO MATERIAL DIRECTIONS INTO SPACE, so the
        // solid dispatch has three arms where the face's has four and this id
        // reaches the same refusal an id no backend knows does. It is the one
        // `Model` variant the solid table has no answer for, which is what
        // `super::super::refusal::material_defects` counts over the tets.
        let (verdict, gradient, hessian) = tet_dispatch_of(Model::BaraffWitkin as u32);
        assert_eq!(verdict, tet_verdict::UNKNOWN);
        assert_eq!(gradient, [0.0; 3]);
        assert_eq!(hessian, [0.0; 9]);

        // While the three the table does carry report the other code and a
        // table that was actually written.
        for model in [Model::Arap, Model::StVK, Model::SNHk] {
            let (verdict, gradient, _) = tet_dispatch_of(model as u32);
            assert_eq!(verdict, tet_verdict::ACCEPTED, "{model:?}");
            assert!(
                gradient.iter().any(|value| *value != 0.0),
                "{model:?} must write a table"
            );
        }
    }

    #[test]
    fn a_pdrd_tet_is_admitted_by_the_table_with_no_elastic_energy() {
        // The solid counterpart of the face case: a rigid body's tets carry
        // model id 4 because their shape is held by the reduced rigid solve,
        // and the id is not validated, so a table that validated it would have
        // to admit a non-elastic model anyway. Zero is the correct energy for
        // an element that has none.
        let (verdict, gradient, hessian) = tet_dispatch_of(Model::Pdrd as u32);
        assert_eq!(verdict, tet_verdict::ACCEPTED);
        assert_eq!(gradient, [0.0; 3]);
        assert_eq!(hessian, [0.0; 9]);
    }

    /// The five `Model` variants, and the gate stops exactly the ones the
    /// kernel has no answer for.
    ///
    /// TWO LISTS ARE THE DEFECT THIS EXISTS AGAINST. The assembly takes no
    /// branch on a verdict, because the verdict is a function of the element's
    /// material and `super::super::refusal::material_defects` settles it at
    /// `initialize()`. That leaves two enumerations of the same vocabulary, one
    /// in the neutral body and one in the gate, and they can disagree in two
    /// directions: a scene refused that the backend could assemble is loud and
    /// merely annoying, and a scene assembled that the kernel has no arm for
    /// writes a zero table under a material the scene did not ask for. Reading
    /// both answers here and comparing them makes the second impossible rather
    /// than unlikely.
    #[test]
    fn the_material_gate_stops_exactly_what_the_face_dispatch_cannot_assemble() {
        for model in [
            Model::Arap,
            Model::StVK,
            Model::BaraffWitkin,
            Model::SNHk,
            Model::Pdrd,
        ] {
            // `mu` matches the helper's, so the two answers are taken about the
            // same element: a face with a positive shear modulus, which is what
            // puts it in the elastic pass at all.
            let (verdict, _, _) = dispatch_of(model as u32);
            let kernel_stops =
                verdict == face_verdict::UNKNOWN || verdict == face_verdict::NO_ENERGY;
            let mut scene = TestScene::new(3).with_faces(&[Vec3u::new(0, 1, 2)]);
            scene.data.param_arrays.face = CVec::from(&[material(model, 100.0, 50.0)][..]);
            let stopped = !crate::driver::refusal::material_defects(&scene.data).is_empty();
            assert_eq!(
                stopped, kernel_stops,
                "the gate and the face dispatch disagree about {model:?}"
            );
        }
    }

    #[test]
    fn the_material_gate_stops_exactly_what_the_tet_dispatch_cannot_assemble() {
        // The solid half, and it is a separate case rather than a parameter:
        // the two dispatches recognize different sets, BaraffWitkin being a
        // shell material and PDRD reaching a different code on each side.
        for model in [
            Model::Arap,
            Model::StVK,
            Model::BaraffWitkin,
            Model::SNHk,
            Model::Pdrd,
        ] {
            let (verdict, _, _) = tet_dispatch_of(model as u32);
            let kernel_stops = verdict == tet_verdict::UNKNOWN;
            let mut scene = TestScene::new(4).with_tets(&[Vec4u::new(0, 1, 2, 3)]);
            scene.data.param_arrays.tet = CVec::from(
                &[TetParam {
                    model,
                    mu: 100.0,
                    lambda: 50.0,
                    ..TetParam::default()
                }][..],
            );
            let stopped = !crate::driver::refusal::material_defects(&scene.data).is_empty();
            assert_eq!(
                stopped, kernel_stops,
                "the gate and the solid dispatch disagree about {model:?}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Shell bending.
    // ------------------------------------------------------------------

    // THE DIHEDRAL ANGLE IS DISPATCHED, NOT NAMED. Its entry point is
    // generated, so it reaches a backend through `Device::launch` and nothing
    // outside `entrypoints/` spells its symbol; the oracle below fills the same
    // record production fills and dispatches it on a host device of its own.
    //
    // The test uses the angle as an ENERGY, which is what makes the
    // finite-difference check below a real one: the force the assembly scatters
    // is `k (theta - theta0) grad(theta)`, and `grad(theta)` is a separate
    // closed form that this compares against the derivative of `theta` itself.

    /// The step a central difference moves one position component by, in
    /// world units.
    ///
    /// A central difference divides by the step it took, so the perturbation
    /// has to land exactly. `2^-11` is a power of two and every coordinate
    /// these fixtures place is of order one, so adding it to a component is
    /// exact in fp32 and the two probes stay symmetric about the base pose. It
    /// is large enough that the fp32 angle's own round-off (about `1e-7` on an
    /// angle of order one) contributes `2e-4` to the quotient and small enough
    /// that the central difference's truncation is four orders below that.
    const FD_STEP: f32 = 1.0 / 2048.0;
    /// The same step in `f64`, which is the precision each quotient below is
    /// formed in.
    const FD_STEP_F64: f64 = FD_STEP as f64;

    /// One hinge with the whole bending assembly around it.
    ///
    /// FOUR VERTICES AND NO FACES. The bending assembly reads `mesh.hinge`, the
    /// hinge props and params, the hinge type table and the per-vertex mass and
    /// area, and nothing else; a face array would be scenery. The hinge is
    /// `(0, 1, 2, 3)`, so vertices 0 and 1 are the shared edge and 2 and 3 are
    /// the two flaps, which is the order `builder.rs` builds and the order the
    /// `(2, 1, 0, 3)` permutation is defined against.
    struct Bending {
        scene: TestScene,
        state: SolverState,
        /// ONE DEVICE FOR THE FIXTURE'S LIFE. A handle names an arena of the
        /// allocator that opened it, so a fixture that sized its state on one
        /// device and dispatched on another would be resolving a handle against
        /// a table that never held it. That was invisible while every buffer
        /// was a host `Vec` carrying its own address, and it is a named refusal
        /// now: `resolve: handle names arena 0, which is not open`.
        device: HostDevice,
    }

    /// The rest and bent poses, in meters. Every coordinate but the two on the
    /// bent flap is dyadic, so it is exactly representable and exact in fp32.
    const SHARED_A: [f32; 3] = [0.0, 0.0, 0.0];
    const SHARED_B: [f32; 3] = [1.0, 0.0, 0.0];
    const FLAP_A: [f32; 3] = [0.5, 1.0, 0.0];
    const FLAP_B_FLAT: [f32; 3] = [0.5, -1.0, 0.0];
    const FLAP_B_BENT: [f32; 3] = [0.5, -0.6, -0.8];
    /// A second bend, so a test that needs two DIFFERENT poses of one hinge
    /// has one that is not the flat book. Flat is the degenerate case for the
    /// analytic eigensystem, not merely a special value; see
    /// `a_hinge_at_its_rest_angle_carries_no_force`.
    const FLAP_B_START: [f32; 3] = [0.5, -0.8, -0.6];

    fn hinge_material(bend: f32) -> HingeParam {
        HingeParam {
            bend,
            ..HingeParam::default()
        }
    }

    impl Bending {
        /// The hinge at `flap_b`, with `bend` and no anisotropy or damping.
        fn book(flap_b: [f32; 3], material: HingeParam) -> Self {
            Self::configured(flap_b, material, 1.0, 1.0, 2.0, 1.0, 0.0)
        }

        /// Every quantity the stiffness scalar is composed from, so a test can
        /// move one at a time.
        ///
        /// `length` and `area` are the hinge's shared-edge length and its two
        /// triangles' combined rest area, which is what `HingeProp` carries;
        /// they are AUTHORED here rather than measured off the pose, exactly as
        /// `builder.rs` authors them, so a test can vary the coefficient
        /// without moving the geometry the angle is read from.
        fn configured(
            flap_b: [f32; 3],
            material: HingeParam,
            length: f32,
            area: f32,
            vertex_mass: f32,
            vertex_area: f32,
            uv_edge_sin2: f32,
        ) -> Self {
            let mut scene = TestScene::new(4).with_hinges(&[Vec4u::new(0, 1, 2, 3)]);
            scene.place(0, SHARED_A[0], SHARED_A[1], SHARED_A[2]);
            scene.place(1, SHARED_B[0], SHARED_B[1], SHARED_B[2]);
            scene.place(2, FLAP_A[0], FLAP_A[1], FLAP_A[2]);
            scene.place(3, flap_b[0], flap_b[1], flap_b[2]);
            scene.data.param_arrays.hinge = CVec::from(&[material][..]);
            // Bit 0 clear: a shell's own hinge, which is the one that bends.
            scene.data.mesh.ttype.hinge = CVec::from(&[0u8][..]);
            let prop = &mut scene.data.prop.hinge.as_mut_slice()[0];
            prop.length = length;
            prop.area = area;
            prop.rest_angle = 0.0;
            prop.uv_edge_sin2 = uv_edge_sin2;
            for vertex in scene.data.prop.vertex.as_mut_slice() {
                vertex.mass = vertex_mass;
                vertex.area = vertex_area;
            }
            install_pattern(&mut scene.data, 4);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            Self {
                scene,
                state,
                device,
            }
        }

        fn hinge_prop_mut(&mut self) -> &mut crate::data::HingeProp {
            &mut self.scene.data.prop.hinge.as_mut_slice()[0]
        }

        /// The start-of-step pose, as flat position components.
        fn start_of_step(&self) -> Vec<f32> {
            let curr = self.scene.data.vertex.curr.as_slice();
            let mut out = Vec::with_capacity(3 * curr.len());
            for position in curr {
                for k in 0..3 {
                    out.push(position[k]);
                }
            }
            out
        }

        /// Assemble with the Newton iterate at `iterate`, which is what the
        /// elastic block is evaluated at; `vertex.curr` stays where the fixture
        /// placed it and is what the lagged damping block is evaluated at.
        fn assemble_at(&mut self, iterate: &[f32], dt: f32) -> FatalResult<Assembled> {
            let Bending {
                scene,
                state,
                device,
            } = self;
            clear_force(device, state);
            // `positions` IS the start-of-step pose, so it tracks the scene at
            // every fixture entry. A test that wants the iterate somewhere else
            // seeds `eval_x` separately below; one that does not gets both at
            // the scene, which is what a step would see.
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // AND THE HINGE RECORDS, because the bending stiffness reads its
            // geometry, its material and its type off the device now. A fixture
            // that marks a hinge `fixed`, `collider` or solid-surface does it
            // AFTER `allocate` staged them, which production never does.
            crate::driver::state::reseed_hinge(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            state.eval_x.seed(device, iterate).expect("the fixture seeds the buffer");
            // Safety: the scene is live, the state was allocated for it, and the
            // matrix borrows the scene's own pattern tables.
            let (force, dense) = unsafe {
                let mut fixed = FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default())?;
                shell_bending(device, &scene.data, state, &mut fixed, dt)?;
                // THE MATRIX IS DEVICE-RESIDENT, so the dense reference below reads
                // the mirror and the mirror is stale the moment the assembly
                // dispatch names the buffer. The download costs a copy and buys
                // the read its guarantee.
                fixed.download(device)?;
                let width = 12;
                let mut dense = vec![0.0f32; width * width];
                for i in 0..4 {
                    for j in 0..4 {
                        let block = fixed.read(i as u32, j as u32);
                        for r in 0..3 {
                            for c in 0..3 {
                                dense[width * (3 * i + r) + 3 * j + c] = block[3 * c + r];
                            }
                        }
                    }
                }
                (force_of(device, state), dense)
            };
            Ok(Assembled {
                force,
                dense,
                width: 12,
            })
        }

        /// Assemble with the iterate at the start-of-step pose.
        fn assemble(&mut self, dt: f32) -> FatalResult<Assembled> {
            let iterate = self.start_of_step();
            self.assemble_at(&iterate, dt)
        }

        /// The stiffness scalar the last assembly formed.
        ///
        /// DOWNLOADS FIRST, because the assembly ends with the scale that reads
        /// this array, and taking a handle is what marks the mirror stale: the
        /// buffer type cannot tell a kernel that READS a scale from one that
        /// writes its destination. The contents are the same either way, so the
        /// download costs a copy and buys the read its guarantee.
        /// AN ORACLE, DISPATCHED, rather than a read of a production buffer.
        ///
        /// The assembly forms this value in the element's own thread and stores
        /// it nowhere, so there is no array left to download. It dispatches the
        /// SAME two shared bodies the fused kernel calls, which is what keeps
        /// the test and the assembly from drifting: re-deriving the four-term
        /// fp32 average on the host would be a second implementation of the
        /// summation `shell_bend_stiffness.kernel.cpp` calls order-dependent.
        fn stiffness(&mut self) -> f64 {
            let Bending { scene, state, device } = self;
            crate::driver::state::reseed_props(device, state, &scene.data);
            crate::driver::state::reseed_hinge(device, state, &scene.data);
            let count = state.sizes.hinges as u32;
            let density_args = ShellBendArealDensityFromRecordsArgs {
                vertex_prop: state.prop_vertex.handle(),
                hinge: state.mesh_hinge.handle(),
                vertex_count: state.sizes.vertices as u32,
                areal_density: state.hinge.areal_density.handle(),
                count,
                seam_arena_count: 0,
            };
            // Safety: every handle names a live allocation for the whole call.
            unsafe { device.launch("oracle.hinge.areal_density", &density_args, count) }
                .expect("the areal density oracle dispatches");
            let stiffness_args = ShellBendStiffnessAndDampingArgs {
                prop: state.prop_hinge.handle(),
                hinge_param: state.param_hinge.handle(),
                kind: state.hinge_kind.handle(),
                areal_density: state.hinge.areal_density.handle(),
                stiffness: state.hinge.stiffness.handle(),
                damping: state.hinge.damping.handle(),
                count,
                seam_arena_count: 0,
            };
            // Safety: as above.
            unsafe { device.launch("oracle.hinge.bend_stiffness", &stiffness_args, count) }
                .expect("the stiffness oracle dispatches");
            state
                .hinge
                .stiffness
                .download(device)
                .expect("the stiffness oracle reads back");
            f64::from(state.hinge.stiffness.host()[0])
        }

        /// The dihedral angle at `iterate`, read through the shared body.
        fn angle_at(&self, iterate: &[f32]) -> f64 {
            // The (2, 1, 0, 3) permutation of hinge (0, 1, 2, 3), which is the
            // order the dihedral math is written in.
            let remapped: [u32; 4] = [2, 1, 0, 3];
            let mut device = crate::driver::launch::host_device();
            let mut angle_out =
                one_element_buffer(&mut device, "test.shell_bend.angle", &[0.0f32]);
            let mut remapped_in =
                one_element_buffer(&mut device, "test.shell_bend.hinge", &remapped);
            // THE RECORD PRODUCTION FILLS, DISPATCHED THE WAY PRODUCTION
            // DISPATCHES IT, so this oracle exercises the binding and the guard
            // as well as the body.
            let args = crate::driver::kernels::ShellBendAngleArgs {
                // Safety: `iterate` is three position components per vertex and
                // outlives the dispatch below, `remapped` names four in-range
                // vertices of it, and the angle is read back off its own device
                // allocation once the call has returned.
                x: crate::driver::state::position_block(&mut device, iterate, "test.iterate").handle(),
                hinge: remapped_in.handle(),
                vertex_count: (iterate.len() / 3) as u32,
                angle: angle_out.handle(),
                count: 1,
                seam_arena_count: 0,
            };
            // Safety: every reference in the record names live storage that
            // outlives this dispatch.
            unsafe { device.launch("test.shell_bend.angle", &args, 1) }
                .expect("the dihedral angle dispatches");
            let angle = angle_out
                .read_one(&mut device, 0)
                .expect("the dihedral angle reads back");
            angle_out
                .free(&mut device)
                .expect("the angle buffer frees");
            remapped_in
                .free(&mut device)
                .expect("the stencil buffer frees");
            f64::from(angle)
        }

        /// `E = 1/2 k (theta - theta0)^2` at `iterate`, in `f64`.
        ///
        /// `f64` because this is a test measuring a result: the quantity being
        /// differenced is an angle of order one perturbed by `2^-11`, and
        /// forming the difference in the precision that produced it would fold
        /// the measurement into the answer.
        fn energy_at(&self, iterate: &[f32], stiffness: f64, rest_angle: f64) -> f64 {
            let difference = self.angle_at(iterate) - rest_angle;
            0.5 * stiffness * difference * difference
        }
    }

    /// Every component of the assembled force, as `f64`.
    fn force_vector(assembled: &Assembled) -> Vec<f64> {
        (0..12).map(|k| f64::from(assembled.force[k])).collect()
    }

    #[test]
    fn a_hinge_at_its_rest_angle_carries_no_force() {
        // The force is `k (theta - theta0) grad(theta)`, so a hinge held at its
        // own rest angle carries none of it, EXACTLY and not approximately.
        // Taking the rest angle from the pose rather than the pose from the
        // rest angle is what makes this test the rest angle's wiring as well:
        // a `HingeProp::rest_angle` that never reached the body would leave the
        // bent hinge below carrying its full force.
        let mut fixture = Bending::book(FLAP_B_BENT, hinge_material(1.0e5));
        let pose = fixture.start_of_step();
        let angle = fixture.angle_at(&pose) as f32;
        assert!(angle.abs() > 0.5, "the fixture must be bent, got {angle} rad");
        fixture.hinge_prop_mut().rest_angle = angle;
        let assembled = fixture.assemble(0.01).expect("the hinge assembles");
        for vertex in 0..4 {
            assert_eq!(
                assembled.vertex_force(vertex),
                [0.0, 0.0, 0.0],
                "vertex {vertex} of a hinge at its rest angle"
            );
        }
        assert!(
            fixture.stiffness() > 0.0,
            "the fixture must actually be a bending element: stiffness was {}",
            fixture.stiffness()
        );
        // AND IT STILL RESISTS BEING BENT: the rest angle removes the force,
        // not the stiffness.
        let trace: f64 = (0..12).map(|k| f64::from(assembled.dense[12 * k + k])).sum();
        assert!(
            trace > 0.0,
            "a hinge at its rest angle still carries bending stiffness, and its \
             Hessian trace was {trace}"
        );
    }

    #[test]
    fn an_exactly_flat_hinge_carries_no_force_and_no_hessian() {
        // FLAT IS THE DEGENERATE CASE FOR THE ANALYTIC EIGENSYSTEM, which is
        // worth recording where a reader will meet it. The two flap directions
        // measured across the shared edge are exactly anti-parallel there, so
        // the binormal the Wu and Kim decomposition is built on is undefined
        // and `shell_bend_force_hessian` returns a ZERO Hessian rather than
        // a projection of something it cannot form. The FORCE is still exact,
        // and it is zero here because the angle is exactly zero and so is the
        // rest angle. This is the shared body's own behavior, identical on
        // every backend, and not something this assembly decides.
        let mut fixture = Bending::book(FLAP_B_FLAT, hinge_material(1.0e5));
        let assembled = fixture.assemble(0.01).expect("the flat hinge assembles");
        assert_eq!(fixture.angle_at(&fixture.start_of_step()), 0.0);
        for dof in 0..12 {
            assert_eq!(assembled.force[dof], 0.0, "dof {dof} of a flat hinge");
        }
        for (slot, value) in assembled.dense.iter().enumerate() {
            assert_eq!(*value, 0.0, "slot {slot} of a flat hinge's Hessian");
        }
    }

    #[test]
    fn the_bending_force_is_the_gradient_of_the_hinge_energy() {
        // THE TEST THE WHOLE SUBSYSTEM ANSWERS TO, and the one an absent or
        // mis-permuted assembly cannot pass. The scattered force must be
        // dE/dx for E = 1/2 k (theta - theta0)^2, component by component and
        // vertex by vertex, against a central difference of the angle body's
        // own output. It pins four separate things at once: that the analytic
        // angle gradient is the derivative of the angle, that the stiffness
        // scalar multiplies it, that the (2, 1, 0, 3) permutation is applied
        // consistently between the evaluation and the scatter, and that each
        // column lands on the vertex it belongs to.
        let mut fixture = Bending::book(FLAP_B_BENT, hinge_material(1.0e5));
        let base = fixture.start_of_step();
        let assembled = fixture.assemble_at(&base, 0.01).expect("the bent hinge assembles");
        let stiffness = fixture.stiffness();
        let rest_angle = 0.0;
        let angle = fixture.angle_at(&base);
        assert!(
            angle.abs() > 0.5,
            "the fixture must be meaningfully bent; the angle was {angle} rad"
        );

        let analytic = force_vector(&assembled);
        let scale = analytic.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(
            scale > 1.0e-3,
            "a bent hinge must carry a force to compare against, got {scale}"
        );
        let mut probe = base.clone();
        for dof in 0..12 {
            probe[dof] = base[dof] + FD_STEP;
            let forward = fixture.energy_at(&probe, stiffness, rest_angle);
            probe[dof] = base[dof] - FD_STEP;
            let backward = fixture.energy_at(&probe, stiffness, rest_angle);
            probe[dof] = base[dof];
            let derivative = (forward - backward) / (2.0 * FD_STEP_F64);
            assert!(
                (analytic[dof] - derivative).abs() <= 1.0e-2 * scale,
                "dof {dof} (vertex {}, component {}): the assembly scattered {} and the energy's \
                 own derivative is {}",
                dof / 3,
                dof % 3,
                analytic[dof],
                derivative
            );
        }
    }

    #[test]
    fn the_hinge_hessian_is_the_angle_gradient_outer_product_at_the_rest_angle() {
        // WHAT PINS THE SIXTEEN BLOCKS TO THE RIGHT PAIRS OF VERTICES. At
        // theta == theta0 the exact Hessian of E = 1/2 k (theta - theta0)^2 is
        // k g g^T for g = grad(theta), because the term carrying d2theta/dx2 is
        // multiplied by (theta - theta0) and vanishes. That matrix is PSD
        // already, so the analytic projection returns it unchanged and the
        // assembled 12x12 is exactly it. `g` is measured here by differencing
        // the angle body, so a scatter that named the mesh order where the
        // evaluation used the permuted one puts every off-diagonal block on the
        // wrong pair of vertices and fails.
        let mut fixture = Bending::book(FLAP_B_BENT, hinge_material(1.0e5));
        let pose = fixture.start_of_step();
        let rest_angle = fixture.angle_at(&pose) as f32;
        fixture.hinge_prop_mut().rest_angle = rest_angle;
        let assembled = fixture.assemble(0.01).expect("the hinge assembles");
        let stiffness = fixture.stiffness();

        let mut gradient = [0.0f64; 12];
        let mut probe = pose.clone();
        for dof in 0..12 {
            probe[dof] = pose[dof] + FD_STEP;
            let forward = fixture.angle_at(&probe);
            probe[dof] = pose[dof] - FD_STEP;
            let backward = fixture.angle_at(&probe);
            probe[dof] = pose[dof];
            gradient[dof] = (forward - backward) / (2.0 * FD_STEP_F64);
        }
        let largest = gradient.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(largest > 0.1, "the angle gradient is too small to compare against");
        let scale = stiffness * largest * largest;
        for row in 0..12 {
            for column in 0..12 {
                let expected = stiffness * gradient[row] * gradient[column];
                let actual = f64::from(assembled.dense[12 * row + column]);
                assert!(
                    (actual - expected).abs() <= 1.0e-2 * scale,
                    "block ({}, {}) component ({}, {}): the assembly holds {actual} and \
                     k g g^T is {expected}",
                    row / 3,
                    column / 3,
                    row % 3,
                    column % 3
                );
            }
        }
    }

    #[test]
    fn the_bending_force_sums_to_zero_over_the_four_vertices() {
        // A bending energy depends on shape alone, so translating the hinge
        // changes nothing and the four columns must cancel. A scatter that
        // dropped a column or wrote one twice fails here.
        let mut fixture = Bending::book(FLAP_B_BENT, hinge_material(1.0e5));
        let assembled = fixture.assemble(0.01).expect("the bent hinge assembles");
        for component in 0..3 {
            let total: f64 = (0..4)
                .map(|vertex| f64::from(assembled.force[3 * vertex + component]))
                .sum();
            let scale = (0..4)
                .map(|vertex| f64::from(assembled.force[3 * vertex + component]).abs())
                .fold(1.0e-6, f64::max);
            assert!(
                total.abs() <= 1.0e-5 * scale,
                "component {component} of the bending force sums to {total} over the hinge"
            );
        }
    }

    #[test]
    fn the_hinge_hessian_is_positive_semidefinite_where_the_exact_one_is_not() {
        // SPD-BY-ASSEMBLY. The exact dihedral Hessian is generally indefinite,
        // which is why the shared body projects it through the Wu and Kim 2023
        // analytic eigensystem, and the rest of the assembly must not undo
        // that: the stiffness scalar is non-negative by construction and the
        // lagged damping block is the same projected matrix again. A rest angle
        // far from the pose is what makes the unprojected matrix indefinite.
        let mut fixture = Bending::book(FLAP_B_BENT, hinge_material(1.0e5));
        fixture.hinge_prop_mut().rest_angle = 2.5;
        let assembled = fixture.assemble(0.01).expect("the bent hinge assembles");
        // A deterministic spread of probe directions rather than a random one,
        // so a failure is reproducible: every axis, every axis pair, and two
        // dense mixtures.
        let mut probes: Vec<Vec<f64>> = Vec::new();
        for k in 0..12 {
            let mut v = vec![0.0; 12];
            v[k] = 1.0;
            probes.push(v);
        }
        for k in 0..12 {
            let mut v = vec![0.0; 12];
            v[k] = 1.0;
            v[(k + 5) % 12] = -1.0;
            probes.push(v);
        }
        probes.push((0..12).map(|k| ((k * 7) % 13) as f64 - 6.0).collect());
        probes.push((0..12).map(|k| 6.0 - ((k * 11) % 17) as f64).collect());
        let largest = assembled
            .dense
            .iter()
            .fold(0.0f64, |acc, v| acc.max(f64::from(*v).abs()));
        assert!(largest > 0.0, "the fixture assembled no Hessian at all");
        for (index, probe) in probes.iter().enumerate() {
            let quadratic = assembled.quadratic_form(probe);
            assert!(
                quadratic >= -1.0e-4 * largest,
                "probe {index} gives v^T H v = {quadratic}, so the hinge block is not PSD"
            );
        }
    }

    #[test]
    fn the_stiffness_carries_the_resolution_and_density_normalizations() {
        // `shell_bend_stiffness` is BEND_SCALE * bend * |e|^2 / area *
        // areal_density, and the assembled force is that scalar times a
        // geometric factor the material cannot change. So each input moves the
        // whole force by an exactly known ratio, which pins every one of them
        // to the right slot without restating the calibration constant. A
        // swapped `length` and `area`, or an areal density read as a raw mass,
        // changes at least one of these ratios.
        let reference = Bending::book(FLAP_B_BENT, hinge_material(1.0e5))
            .assemble(0.01)
            .expect("the reference hinge assembles");
        let reference = force_vector(&reference);
        let scale = reference.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(scale > 1.0e-3, "the reference hinge carries no force");

        let cases: [(&str, f64, Bending); 5] = [
            (
                "twice the bending stiffness",
                2.0,
                Bending::book(FLAP_B_BENT, hinge_material(2.0e5)),
            ),
            (
                "twice the shared-edge length",
                4.0,
                Bending::configured(
                    FLAP_B_BENT,
                    hinge_material(1.0e5),
                    2.0,
                    1.0,
                    2.0,
                    1.0,
                    0.0,
                ),
            ),
            (
                "twice the incident area",
                0.5,
                Bending::configured(
                    FLAP_B_BENT,
                    hinge_material(1.0e5),
                    1.0,
                    2.0,
                    2.0,
                    1.0,
                    0.0,
                ),
            ),
            (
                "twice the vertex mass",
                2.0,
                Bending::configured(
                    FLAP_B_BENT,
                    hinge_material(1.0e5),
                    1.0,
                    1.0,
                    4.0,
                    1.0,
                    0.0,
                ),
            ),
            (
                "twice the vertex area",
                0.5,
                Bending::configured(
                    FLAP_B_BENT,
                    hinge_material(1.0e5),
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    0.0,
                ),
            ),
        ];
        for (what, expected, mut fixture) in cases {
            let assembled = fixture.assemble(0.01).expect("the varied hinge assembles");
            let varied = force_vector(&assembled);
            for dof in 0..12 {
                assert!(
                    (varied[dof] - expected * reference[dof]).abs() <= 1.0e-5 * expected * scale,
                    "{what}: dof {dof} is {} against the expected {} times the reference {}",
                    varied[dof],
                    expected,
                    reference[dof]
                );
            }
        }
    }

    #[test]
    fn an_edge_along_warp_takes_the_weft_stiffness() {
        // THE 90-DEGREE CONVENTION, through the whole assembly rather than
        // through `shell_bend_directional` alone. A hinge folds ABOUT its
        // shared edge, so the surface curves ACROSS it: an edge lying along
        // warp (sin^2 = 0) bends the sheet in the weft sense and picks up
        // `bend-weft`, and an edge along weft (sin^2 = 1) picks up `bend-warp`.
        // Exchanging the two leaves a build that is still directional, still
        // non-negative and still SPD, and drapes wrong by 90 degrees, which is
        // why this is asserted where the two numbers actually reach a force.
        const BEND: f32 = 1.0e5;
        const WARP: f32 = 3.0e5;
        const WEFT: f32 = 7.0e5;
        let directional = HingeParam {
            bend: BEND,
            bend_warp: WARP,
            bend_weft: WEFT,
            ..HingeParam::default()
        };
        let along_warp = Bending::configured(
            FLAP_B_BENT,
            directional,
            1.0,
            1.0,
            2.0,
            1.0,
            0.0,
        )
        .assemble(0.01)
        .expect("the warp-aligned hinge assembles");
        let along_weft = Bending::configured(
            FLAP_B_BENT,
            directional,
            1.0,
            1.0,
            2.0,
            1.0,
            1.0,
        )
        .assemble(0.01)
        .expect("the weft-aligned hinge assembles");
        let isotropic_weft = Bending::book(FLAP_B_BENT, hinge_material(BEND + WEFT))
            .assemble(0.01)
            .expect("the isotropic comparison assembles");
        let isotropic_warp = Bending::book(FLAP_B_BENT, hinge_material(BEND + WARP))
            .assemble(0.01)
            .expect("the isotropic comparison assembles");

        let scale = force_vector(&isotropic_weft)
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(scale > 1.0e-3, "the comparison hinge carries no force");
        for (label, actual, expected) in [
            ("an edge along warp", &along_warp, &isotropic_weft),
            ("an edge along weft", &along_weft, &isotropic_warp),
        ] {
            let actual = force_vector(actual);
            let expected = force_vector(expected);
            for dof in 0..12 {
                assert!(
                    (actual[dof] - expected[dof]).abs() <= 1.0e-5 * scale,
                    "{label}: dof {dof} is {} and the isotropic equivalent is {}",
                    actual[dof],
                    expected[dof]
                );
            }
        }
        // And the two are not the same answer, so the assertion above is
        // measuring something: WARP and WEFT differ, so the two orientations
        // must give different forces.
        let a = force_vector(&along_warp);
        let b = force_vector(&along_weft);
        assert!(
            (0..12).any(|k| (a[k] - b[k]).abs() > 1.0e-3 * scale),
            "the two orientations gave the same force, so the directional term is not reaching \
             the hinge at all"
        );
    }

    #[test]
    fn the_no_uv_sentinel_with_a_directional_stiffness_stops_the_run() {
        // Anisotropic bending is meaningless without a direction. A negative
        // `uv_edge_sin2` is the no-UV sentinel and the shared body reads it as
        // isotropic, so a scene asking for warp or weft on a mesh with no UV
        // would quietly get neither.
        let directional = HingeParam {
            bend: 1.0e5,
            bend_warp: 3.0e5,
            ..HingeParam::default()
        };
        let mut fixture = Bending::configured(
            FLAP_B_BENT,
            directional,
            1.0,
            1.0,
            2.0,
            1.0,
            -1.0,
        );
        let error = fixture
            .assemble(0.01)
            .expect_err("a directional hinge with no UV direction must stop the run");
        let detail = format!("{error:?}");
        assert!(
            detail.contains("bend-warp") && detail.contains("no-UV"),
            "the refusal must name the parameter and the sentinel, got {detail}"
        );
        // While the same mesh with no anisotropy asked for is perfectly usable.
        let mut isotropic = Bending::configured(
            FLAP_B_BENT,
            hinge_material(1.0e5),
            1.0,
            1.0,
            2.0,
            1.0,
            -1.0,
        );
        isotropic
            .assemble(0.01)
            .expect("the no-UV sentinel is isotropic, not an error on its own");
    }

    #[test]
    fn a_hinge_that_is_not_a_bending_element_is_left_alone() {
        // The three flags the dispatch tests, one at a time. Bit 0 of
        // `mesh.type.hinge` marks a pair with a SOLID surface face on either
        // side, which carries no bending energy at all; `fixed` says all four
        // vertices are exact Dirichlet rows; `collider` says the surface is
        // held by its pins rather than by stiffness of its own. None of the
        // three aliases the others, so each is checked on its own.
        for (what, prepare) in [
            (
                "a solid's surface hinge",
                Box::new(|fixture: &mut Bending| {
                    fixture.scene.data.mesh.ttype.hinge.as_mut_slice()[0] = 1;
                }) as Box<dyn Fn(&mut Bending)>,
            ),
            (
                "a fully pinned hinge",
                Box::new(|fixture: &mut Bending| {
                    fixture.hinge_prop_mut().fixed = true;
                }),
            ),
            (
                "a collider's hinge",
                Box::new(|fixture: &mut Bending| {
                    fixture.hinge_prop_mut().collider = true;
                }),
            ),
        ] {
            let mut fixture = Bending::book(FLAP_B_BENT, hinge_material(1.0e5));
            prepare(&mut fixture);
            let assembled = fixture.assemble(0.01).expect("the inert hinge assembles");
            for dof in 0..12 {
                assert_eq!(
                    assembled.force[dof], 0.0,
                    "{what} scattered a force at dof {dof}"
                );
            }
            for (slot, value) in assembled.dense.iter().enumerate() {
                assert_eq!(*value, 0.0, "{what} scattered a Hessian at slot {slot}");
            }
        }
    }

    #[test]
    fn a_hinge_with_no_type_byte_stops_the_run() {
        // `mesh.type.hinge` is the ONLY thing separating a shell's bending
        // hinge from a solid's surface hinge, so a missing byte leaves the
        // element's class unknown and neither reading is safe to guess:
        // bending a solid's surface adds an energy that surface does not
        // carry, and skipping a shell's hinge drops one it does.
        let mut fixture = Bending::book(FLAP_B_BENT, hinge_material(1.0e5));
        fixture.scene.data.mesh.ttype.hinge = CVec::new();
        let error = fixture
            .assemble(0.01)
            .expect_err("an unclassified hinge must stop the run");
        let detail = format!("{error:?}");
        assert!(
            detail.contains("hinge type table"),
            "the failure must name the table it could not read, got {detail}"
        );
    }

    #[test]
    fn a_degenerate_hinge_stops_the_run_rather_than_bending_by_nothing() {
        // A zero triangle normal or a zero shared edge leaves the dihedral
        // angle undefined, and the shared body answers by writing a zero force
        // and Hessian, and raises the same three conditions on the diagnostic
        // lane (`DIAG_ASSERT4` in `shell_bend.kernel.cpp`). Scattering that
        // zero without reading the lane would be a silently missing bending
        // term rather than a loud failure.
        let mut fixture = Bending::book(FLAP_B_BENT, hinge_material(1.0e5));
        // Flap A onto the shared edge's own line, so its triangle has no area.
        fixture.scene.place(2, 0.5, 0.0, 0.0);
        let error = fixture
            .assemble(0.01)
            .expect_err("a degenerate hinge must stop the run");
        let detail = format!("{error:?}");
        assert!(
            detail.contains("degenerate"),
            "the failure must say the hinge geometry is degenerate, got {detail}"
        );
    }

    #[test]
    fn the_bending_damping_hessian_is_taken_at_the_start_of_the_step() {
        // LAGGED, WHICH IS WHAT MAKES THE DAMPING GUARANTEED DISSIPATIVE. The
        // damping block is (beta/dt) K, with K the hinge Hessian at the
        // START-OF-STEP positions and not at the Newton iterate, so the damping
        // force is the gradient of a convex potential. This measures exactly
        // that: the difference the damping makes to the Hessian must be
        // (beta/dt) times the UNDAMPED Hessian evaluated at the start-of-step
        // pose, and the difference it makes to the force must be that same
        // matrix applied to the displacement. An implementation that reused the
        // iterate's own Hessian passes neither, because the two poses here are
        // different bends of the same hinge.
        const BETA: f32 = 0.25;
        const DT: f32 = 0.02;

        // Two DIFFERENT bends of the same hinge. Neither may be the flat book,
        // whose Hessian is identically zero (see
        // `an_exactly_flat_hinge_carries_no_force_and_no_hessian`), because a
        // zero lagged Hessian would make this test pass on an implementation
        // that assembled no damping at all.
        let mut damped = Bending::book(FLAP_B_START, hinge_material(1.0e5));
        damped.scene.data.param_arrays.hinge.as_mut_slice()[0].bend_damping = BETA;
        let start = damped.start_of_step();

        // The iterate: the same hinge with its second flap bent away.
        let mut iterate = start.clone();
        let bent = crate::driver::test_scene::position(
            FLAP_B_BENT[0],
            FLAP_B_BENT[1],
            FLAP_B_BENT[2],
        );
        for k in 0..3 {
            iterate[9 + k] = bent[k];
        }

        let with_damping = damped
            .assemble_at(&iterate, DT)
            .expect("the damped hinge assembles");

        let mut undamped = Bending::book(FLAP_B_START, hinge_material(1.0e5));
        let at_iterate = undamped
            .assemble_at(&iterate, DT)
            .expect("the undamped hinge assembles");
        let at_start = undamped
            .assemble_at(&start, DT)
            .expect("the start-of-step hinge assembles");

        let scale = f64::from(BETA) / f64::from(DT);
        let lagged = &at_start.dense;
        let magnitude = lagged.iter().fold(0.0f64, |acc, v| acc.max(f64::from(*v).abs()));
        assert!(
            magnitude > 0.0,
            "the start-of-step pose must carry a Hessian for the damping to be built on"
        );

        // 1. The Hessian difference IS the scaled lagged Hessian.
        for slot in 0..144 {
            let difference = f64::from(with_damping.dense[slot]) - f64::from(at_iterate.dense[slot]);
            let expected = scale * f64::from(lagged[slot]);
            assert!(
                (difference - expected).abs() <= 1.0e-4 * scale * magnitude,
                "slot {slot}: the damping added {difference} and (beta/dt) K_lag is {expected}"
            );
        }

        // 2. The force difference is that same matrix applied to the
        //    displacement from the start of the step to the iterate.
        let displacement: Vec<f64> = (0..12)
            .map(|k| f64::from(iterate[k]) - f64::from(start[k]))
            .collect();
        let moved = displacement.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(moved > 0.1, "the two poses must differ for this to measure anything");
        for row in 0..12 {
            let mut expected = 0.0;
            for column in 0..12 {
                expected += f64::from(lagged[12 * row + column]) * displacement[column];
            }
            expected *= scale;
            let difference =
                f64::from(with_damping.force[row]) - f64::from(at_iterate.force[row]);
            assert!(
                (difference - expected).abs() <= 1.0e-3 * scale * magnitude * moved,
                "row {row}: the damping force is {difference} and (beta/dt) K_lag (x - x^n) is \
                 {expected}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Rods: the Hookean stretch and the turning-angle bending.
    // ------------------------------------------------------------------

    // The turning angle at an interior rod vertex is dispatched rather than
    // named, on the same terms as the shell angle above. The tests below use it
    // as an ENERGY: the force the assembly scatters is
    // `k (theta - theta0) grad(theta)`, and `grad(theta)` is a separate closed
    // form, so differencing `theta` itself compares two independent derivations
    // rather than one against itself.

    /// Install an explicit fixed sparsity, row by row, upper triangle only.
    ///
    /// `install_pattern` stores every pair; this takes the rows verbatim, which
    /// is what lets a test withhold one coupling and watch the assembly refuse
    /// to drop it.
    fn install_rows(data: &mut crate::data::DataSet, rows: &[Vec<u32>]) {
        let mut transpose: Vec<Vec<Vec2u>> = vec![Vec::new(); rows.len()];
        let mut slot = 0u32;
        for (i, row) in rows.iter().enumerate() {
            for &j in row {
                if i as u32 != j {
                    transpose[j as usize].push(Vec2u::new(i as u32, slot));
                }
                slot += 1;
            }
        }
        data.fixed_index_table = CVecVec::from(rows);
        data.transpose_table = CVecVec::from(&transpose[..]);
    }

    /// The vertex-edge and vertex-face adjacencies a rod scene needs.
    ///
    /// The rod bending sites are read off these two and off nothing else, so a
    /// fixture that omits them has no sites whatever its geometry looks like.
    /// The face table is present and EMPTY rather than absent, which is the
    /// shape `builder.rs` produces for a rod-only mesh.
    fn install_rod_adjacency(data: &mut crate::data::DataSet, vertices: usize, edges: &[Vec2u]) {
        let mut incident: Vec<Vec<u32>> = vec![Vec::new(); vertices];
        for (index, edge) in edges.iter().enumerate() {
            incident[edge[0] as usize].push(index as u32);
            incident[edge[1] as usize].push(index as u32);
        }
        data.mesh.neighbor.vertex.edge = CVecVec::from(&incident[..]);
        let faces: Vec<Vec<u32>> = vec![Vec::new(); vertices];
        data.mesh.neighbor.vertex.face = CVecVec::from(&faces[..]);
    }

    /// The assembled matrix over the first `gathered` vertices, block by block
    /// through the shared read, row-major and `3 * gathered` wide.
    ///
    /// THE CALLER MUST HAVE DOWNLOADED the matrix: the values are device-side,
    /// and a read off a mirror an assembly dispatch has invalidated panics
    /// naming the buffer rather than answering out of the previous step.
    ///
    /// # Safety
    /// `fixed` must borrow a live scene.
    unsafe fn dense_of(fixed: &FixedCsr<'_>, gathered: usize) -> Vec<f32> {
        let width = 3 * gathered;
        let mut dense = vec![0.0f32; width * width];
        for i in 0..gathered {
            for j in 0..gathered {
                let block = fixed.read(i as u32, j as u32);
                for r in 0..3 {
                    for c in 0..3 {
                        dense[width * (3 * i + r) + 3 * j + c] = block[3 * c + r];
                    }
                }
            }
        }
        dense
    }

    /// Every vertex's flat position components, which is the shape an iterate has.
    fn positions_of(scene: &TestScene) -> Vec<f32> {
        let curr = scene.data.vertex.curr.as_slice();
        let mut out = Vec::with_capacity(3 * curr.len());
        for position in curr {
            for k in 0..3 {
                out.push(position[k]);
            }
        }
        out
    }

    fn rod_material(stiffness: f32) -> EdgeParam {
        EdgeParam {
            stiffness,
            ..EdgeParam::default()
        }
    }

    /// One rod segment with the whole stretch assembly around it.
    ///
    /// TWO VERTICES ON THE X AXIS, one unit apart, with the rest length
    /// AUTHORED rather than measured off the pose: `EdgeProp::length` is the
    /// drawn length times `length-factor`, so a fixture that took it from the
    /// geometry could not put the segment at a stated strain.
    struct Stretch {
        scene: TestScene,
        state: SolverState,
        /// ONE DEVICE FOR THE FIXTURE'S LIFE, on the grounds
        /// [`Bending::device`] states: a handle names an arena of the allocator
        /// that opened it, so sizing on one device and dispatching on another
        /// resolves a handle against a table that never held it.
        device: HostDevice,
    }

    impl Stretch {
        fn segment(rest: f32, stiffness: f32, mass: f32, damping: f32) -> Self {
            let edges = [Vec2u::new(0, 1)];
            let mut scene = TestScene::new(2).with_edges(&edges);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.0, 0.0, 0.0);
            scene.data.rod_count = 1;
            scene.data.surface_vert_count = 2;
            scene.data.param_arrays.edge = CVec::from(
                &[EdgeParam {
                    deform_damping: damping,
                    ..rod_material(stiffness)
                }][..],
            );
            let prop = &mut scene.data.prop.edge.as_mut_slice()[0];
            prop.length = rest;
            // DELIBERATELY NOT `rest`. `EdgeProp` carries two rest lengths one
            // letter apart: the stretch measures against `length`, the
            // `length-factor`-scaled one, and the strain limiter against
            // `initial_length`, the drawn one. A `length-factor` of 1.0 makes
            // them equal, which is exactly what would hide an exchange of the
            // two, so this fixture keeps them apart.
            prop.initial_length = 0.5 * rest;
            prop.mass = mass;
            install_rod_adjacency(&mut scene.data, 2, &edges);
            install_pattern(&mut scene.data, 2);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            Self { scene, state, device }
        }

        /// Put vertex 1 at `length` along x, which is exactly representable
        /// for every dyadic value used below.
        fn stretch_to(&mut self, length: f32) {
            self.scene.place(1, length, 0.0, 0.0);
            // The scene moved, so the device positions must move with it.
            crate::driver::state::reseed_positions(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
        }

        fn assemble_at(&mut self, iterate: &[f32], dt: f32) -> FatalResult<Assembled> {
            let Stretch { scene, state, device } = self;
            clear_force(device, state);
            // `positions` IS the start-of-step pose, so it tracks the scene at
            // every fixture entry. A test that wants the iterate somewhere else
            // seeds `eval_x` separately below; one that does not gets both at
            // the scene, which is what a step would see.
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            state.eval_x.seed(device, iterate).expect("the fixture seeds the buffer");
            // Safety: the scene is live, the state was allocated for it, and the
            // matrix borrows the scene's own pattern tables.
            let (force, dense) = unsafe {
                let mut fixed = FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default())?;
                rod_stretch(device, &scene.data, state, &mut fixed, dt)?;
                // The dense reference reads the mirror, which the assembly
                // dispatch above invalidated. Download before reading.
                fixed.download(device)?;
                (force_of(device, state), dense_of(&fixed, 2))
            };
            Ok(Assembled {
                force,
                dense,
                width: 6,
            })
        }

        fn assemble(&mut self, dt: f32) -> FatalResult<Assembled> {
            let iterate = positions_of(&self.scene);
            self.assemble_at(&iterate, dt)
        }

        /// `E = 1/2 * stiffness * mass * (l - l0)^2 / l0` at `iterate`, in `f64`.
        ///
        /// THE ENERGY IS A LINEAR SPRING OF CONSTANT `stiffness * mass / l0`,
        /// which is what the shared body's diff table is the derivative pair of:
        /// its Hessian's longitudinal coefficient is `weight / l` and its
        /// transverse one `weight (l - l0) / (l0 l)`, and both match this energy
        /// with `weight = stiffness * mass`. Stating it here rather than reading
        /// the body's own gradient back is what makes the comparison below a
        /// test rather than a restatement.
        fn energy_at(&self, iterate: &[f32], rest: f64, weight: f64) -> f64 {
            let mut squared = 0.0f64;
            for k in 0..3 {
                let difference = f64::from(iterate[3 + k]) - f64::from(iterate[k]);
                squared += difference * difference;
            }
            let extension = squared.sqrt() - rest;
            0.5 * weight * extension * extension / rest
        }
    }

    #[test]
    fn a_rod_at_its_rest_length_carries_no_stretch_force() {
        // `l == l0` makes the extension exactly zero, so the force is zero
        // EXACTLY and not approximately: the body forms `(l / l0 - 1) * n` and
        // the left factor is a difference of two equal fp32 numbers.
        let mut fixture = Stretch::segment(1.0, 2000.0, 0.25, 0.0);
        let assembled = fixture.assemble(0.01).expect("the rod assembles");
        for vertex in 0..2 {
            assert_close(
                assembled.vertex_force(vertex),
                [0.0, 0.0, 0.0],
                "a rod at its rest length",
            );
        }
    }

    #[test]
    fn the_rod_bend_slot_table_is_repacked_from_vertex_order_into_site_order() {
        // THE A/B THAT CATCHES A ROW-KEY ERROR, which neither other slot case
        // structurally can. A rod stretch table is keyed by the edge and a
        // hinge table by the hinge, and both of those dispatches walk their
        // whole array, so there the element index and the table row are the
        // same number. Rod bending is the one place they differ: `builder.rs`
        // keys by surface VERTEX because rod bending is dispatched over every
        // surface vertex, while this driver walks a COMPACTED site list, so
        // `SolverState::allocate` repacks the table before the kernel sees it.
        //
        // This fixture's only site is ordinal 0 and its interior vertex is 1,
        // so the two keys disagree and staging the table verbatim deposits the
        // site's blocks at another vertex's slots, or at an all-sentinel row,
        // which drops them.
        let mut search = RodBend::strand(FAR_BENT, BEND_2_5);
        let by_search = search.assemble(0.01).expect("the search path assembles");

        let mut replay = RodBend::strand(FAR_BENT, BEND_2_5);
        install_rod_bend_hess_slots(
            &mut replay.scene.data,
            3,
            &[Vec2u::new(0, 1), Vec2u::new(1, 2)],
        );
        replay.state = SolverState::default();
        // Safety: the scene lives in its box for the whole test.
        unsafe { replay.state.allocate(&mut replay.device, &replay.scene.data) }
            .expect("the fixture scene reallocates");
        let by_replay = replay.assemble(0.01).expect("the replay path assembles");

        for v in 0..3 {
            assert_close(
                by_replay.vertex_force(v),
                by_search.vertex_force(v),
                "the rod bending slot replay force",
            );
        }
        assert_eq!(by_replay.dense.len(), by_search.dense.len());
        for (k, (a, b)) in by_replay.dense.iter().zip(by_search.dense.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "slot {k}: replay {a} against search {b}"
            );
        }
        // AND SOMETHING LANDED. Staging the vertex-keyed table verbatim reads
        // row 0, which for this fixture is all sentinel, so every block is
        // dropped and the two matrices would agree at zero.
        assert!(
            by_replay.dense.iter().any(|v| v.abs() > 0.0),
            "the replay deposit wrote no block at all"
        );
    }

    #[test]
    fn the_hinge_slot_replay_deposit_agrees_with_the_row_search() {
        // THE SAME A/B ON AN ELEMENT WHOSE OFF-DIAGONAL BLOCKS ARE ASYMMETRIC,
        // which the rod case cannot be. A hinge's 12x12 Hessian has no symmetry
        // between its (ii, jj) and (jj, ii) 3x3 blocks, so this case
        // discriminates a transposed or permuted slot index where the rod one
        // reads the same nine floats either way.
        //
        // AND IT IS THE CASE THE REMAP MATTERS FOR: the table is built in the
        // `(2,1,0,3)` order the force and Hessian are already in, so a table
        // built in mesh order would fail here and pass no other test.
        let material = HingeParam { bend: 4.0, ..HingeParam::default() };

        let mut search = Bending::book([0.3, 0.8, 0.2], material);
        let by_search = search.assemble(0.01).expect("the search path assembles");

        let mut replay = Bending::book([0.3, 0.8, 0.2], material);
        install_hinge_hess_slots(&mut replay.scene.data, 4, &[Vec4u::new(0, 1, 2, 3)]);
        replay.state = SolverState::default();
        // Safety: the scene lives in its box for the whole test.
        unsafe { replay.state.allocate(&mut replay.device, &replay.scene.data) }
            .expect("the fixture scene reallocates");
        let by_replay = replay.assemble(0.01).expect("the replay path assembles");

        for v in 0..4 {
            assert_close(
                by_replay.vertex_force(v),
                by_search.vertex_force(v),
                "the hinge slot replay force",
            );
        }
        assert_eq!(by_replay.dense.len(), by_search.dense.len());
        let mut asymmetric = false;
        for (k, (a, b)) in by_replay.dense.iter().zip(by_search.dense.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "slot {k}: replay {a} against search {b}"
            );
            asymmetric = asymmetric || a.abs() > 0.0;
        }
        assert!(asymmetric, "the replay deposit wrote no block at all");
    }

    #[test]
    fn the_slot_replay_deposit_agrees_with_the_row_search() {
        // THE DEFAULT DEPOSIT AGAINST ITS OWN A/B ARM.
        // `builder.rs` precomputes one CSR slot per stored block and the
        // deposit uses it; `PPF_SLOT_REPLAY=0` ships an empty
        // table and the row search runs instead. Both must land the same
        // Hessian, which is what makes the table an optimization of the lookup
        // rather than a second opinion about the matrix.
        //
        // THIS TEST EXISTS BECAUSE THE FIXTURES SHIP NO TABLE. Every other rod
        // test builds its `DataSet` directly rather than through `builder.rs`,
        // so `edge_hess_slots` is empty and the slot branch is not reached:
        // measured, deleting that branch's deposit entirely left all 458 tests
        // passing.
        let (rest, stiffness, mass) = (0.8f32, 2000.0f32, 0.25f32);

        let mut search = Stretch::segment(rest, stiffness, mass, 0.0);
        search.stretch_to(1.0);
        let by_search = search.assemble(0.01).expect("the search path assembles");

        let mut replay = Stretch::segment(rest, stiffness, mass, 0.0);
        install_edge_hess_slots(&mut replay.scene.data, 2, &[Vec2u::new(0, 1)]);
        // The table is staged with the topology, so the state must be
        // reallocated against the scene that now carries it.
        replay.state = SolverState::default();
        // Safety: the scene lives in its box for the whole test.
        unsafe { replay.state.allocate(&mut replay.device, &replay.scene.data) }
            .expect("the fixture scene reallocates");
        replay.stretch_to(1.0);
        let by_replay = replay.assemble(0.01).expect("the replay path assembles");

        for v in 0..2 {
            assert_close(
                by_replay.vertex_force(v),
                by_search.vertex_force(v),
                "the slot replay force",
            );
        }
        assert_eq!(
            by_replay.dense.len(),
            by_search.dense.len(),
            "the two deposits fill the same matrix"
        );
        let mut moved = false;
        for (k, (a, b)) in by_replay.dense.iter().zip(by_search.dense.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "slot {k}: replay {a} against search {b}"
            );
            moved = moved || a.abs() > 0.0;
        }
        // AND THE MATRIX IS NOT EMPTY, or the comparison above is two zeros
        // agreeing and says nothing about either path.
        assert!(moved, "the replay deposit wrote no block at all");
        // WHAT THIS TEST CANNOT SEE, ON THIS ELEMENT TYPE. Transposing the
        // slot index to `arity * jj + ii` still passes, and that is a property
        // of a rod rather than a gap in the assertions: the stretch Hessian's
        // off-diagonal 3x3 is itself symmetric, so writing it where its
        // transpose belongs writes the same nine floats, and the block the
        // transpose displaces is the lower-triangle sentinel. An element whose
        // off-diagonal blocks are asymmetric, a hinge or a tet, would
        // discriminate it, so extending this deposit to those types needs its
        // own case rather than this one widened.
    }

    #[test]
    fn the_stretch_is_a_linear_spring_of_the_derived_constant() {
        // A rod stretched to 1.25 of its rest length pulls its two ends toward
        // each other with `k * (l - l0)`, `k = stiffness * mass / l0`. The
        // assembled quantity is the energy GRADIENT, so vertex 1, which sits at
        // +x, carries `+k (l - l0)` along x.
        let (rest, stiffness, mass) = (0.8f32, 2000.0f32, 0.25f32);
        let mut fixture = Stretch::segment(rest, stiffness, mass, 0.0);
        fixture.stretch_to(1.0);
        let assembled = fixture.assemble(0.01).expect("the rod assembles");
        let spring = stiffness * mass / rest;
        let expected = spring * (1.0 - rest);
        assert_close(
            assembled.vertex_force(1),
            [expected, 0.0, 0.0],
            "the stretched end of a rod",
        );
        assert_close(
            assembled.vertex_force(0),
            [-expected, 0.0, 0.0],
            "the anchored end of a rod",
        );
    }

    #[test]
    fn the_stretch_force_is_the_gradient_of_the_spring_energy() {
        // Every one of the six components, against a central difference of the
        // energy the docstring on `energy_at` states. A permuted scatter, a
        // dropped `1 / l0`, or a weight that is not `stiffness * mass` all fail
        // this; a sign error fails it on every component at once.
        let (rest, stiffness, mass) = (0.75f32, 1500.0f32, 0.5f32);
        let mut fixture = Stretch::segment(rest, stiffness, mass, 0.0);
        // Off-axis, so all three dimensions carry a component and a swapped
        // pair of columns cannot hide.
        fixture.scene.place(1, 0.5, 0.5, 0.25);
        let base = positions_of(&fixture.scene);
        let assembled = fixture.assemble(0.01).expect("the rod assembles");
        let weight = f64::from(stiffness) * f64::from(mass);
        for component in 0..6usize {
            let mut forward = base.clone();
            let mut backward = base.clone();
            forward[component] += FD_STEP;
            backward[component] -= FD_STEP;
            let derivative = (fixture.energy_at(&forward, f64::from(rest), weight)
                - fixture.energy_at(&backward, f64::from(rest), weight))
                / (2.0 * FD_STEP_F64);
            let actual = f64::from(assembled.force[component]);
            let scale = derivative.abs().max(1.0);
            assert!(
                (actual - derivative).abs() <= 2e-3 * scale,
                "component {component}: the assembly says {actual} and the energy's central \
                 difference says {derivative}"
            );
        }
    }

    #[test]
    fn the_stretch_force_sums_to_zero_over_the_two_vertices() {
        // The energy depends on the two positions only through their
        // difference, so a translation of the whole rod changes nothing and the
        // two forces are equal and opposite. It is what makes the rod carry no
        // net momentum of its own.
        let mut fixture = Stretch::segment(0.75, 1500.0, 0.5, 0.0);
        fixture.scene.place(1, 0.5, 0.5, 0.25);
        let assembled = fixture.assemble(0.01).expect("the rod assembles");
        for k in 0..3 {
            let total = assembled.force[k] + assembled.force[3 + k];
            assert!(
                total.abs() <= 1e-3,
                "component {k} of the rod's net force is {total}"
            );
        }
    }

    #[test]
    fn the_stretch_hessian_carries_the_spring_constant_under_tension() {
        // Under tension no clamp binds, so the assembled block is the EXACT
        // Hessian of the spring energy and its curvature along the stretch
        // mode is the spring constant times the square of that mode's effect on
        // the length. Moving the two ends apart by one each changes the length
        // by two, so `v^T H v = 4 * stiffness * mass / l0`.
        let (rest, stiffness, mass) = (0.8f32, 2000.0f32, 0.5f32);
        let mut fixture = Stretch::segment(rest, stiffness, mass, 0.0);
        fixture.stretch_to(1.0);
        let assembled = fixture.assemble(0.01).expect("the rod assembles");
        let mut longitudinal = vec![0.0f64; 6];
        longitudinal[0] = 1.0;
        longitudinal[3] = -1.0;
        let spring = f64::from(stiffness) * f64::from(mass) / f64::from(rest);
        let expected = 4.0 * spring;
        let measured = assembled.quadratic_form(&longitudinal);
        assert!(
            (measured - expected).abs() <= 1e-3 * expected,
            "the stretched rod's longitudinal curvature is {measured} against the derived \
             {expected}"
        );
    }

    #[test]
    fn the_stretch_hessian_is_positive_semidefinite_under_compression() {
        // A COMPRESSED SPRING HAS NEGATIVE TRANSVERSE CURVATURE, so the exact
        // Hessian is indefinite here and the assembled one must not be. The
        // exact transverse coefficient is `weight (l - l0) / (l0 l)`, which at
        // `l = l0 / 2` and `weight = 1000` puts the probe below at -4000; the
        // shared body clamps it with `max(0, r)`, leaving a rank-1 multiple of
        // the length gradient's outer product, which a transverse mode is
        // orthogonal to. So the projected curvature is EXACTLY zero here, not
        // merely non-negative. `pAp <= 0` hard-errors on the strength of it.
        let (rest, stiffness, mass) = (1.0f32, 2000.0f32, 0.5f32);
        let mut fixture = Stretch::segment(rest, stiffness, mass, 0.0);
        fixture.stretch_to(0.5);
        let assembled = fixture.assemble(0.01).expect("the rod assembles");
        for axis in 1..3usize {
            let mut probe = vec![0.0f64; 6];
            probe[axis] = 1.0;
            probe[3 + axis] = -1.0;
            let measured = assembled.quadratic_form(&probe);
            assert!(
                measured.abs() <= 1e-3,
                "a compressed rod's transverse curvature along axis {axis} is {measured}, and \
                 the unprojected form would put it at -4000"
            );
        }
        // And the rod still resists being compressed further: the longitudinal
        // mode keeps the `weight / l` term the clamp does not touch.
        let mut longitudinal = vec![0.0f64; 6];
        longitudinal[0] = 1.0;
        longitudinal[3] = -1.0;
        let expected = 4.0 * f64::from(stiffness) * f64::from(mass) / 0.5;
        let measured = assembled.quadratic_form(&longitudinal);
        assert!(
            (measured - expected).abs() <= 1e-3 * expected,
            "the compressed rod's longitudinal curvature is {measured} against the derived \
             {expected}"
        );
    }

    #[test]
    fn rod_rayleigh_damping_scales_the_hessian_and_leaves_the_force_alone_at_a_zero_step() {
        // The damping force is `(beta / dt) K (x - x^n)`, so an iterate AT the
        // start of the step carries none of it while the Hessian is still
        // scaled by `1 + beta / dt`. That asymmetry is what makes passing one
        // pose for both a silent defect rather than a loud one.
        let (rest, stiffness, mass, dt) = (0.8f32, 1500.0f32, 0.5f32, 0.01f32);
        let beta = 0.02f32;
        let mut undamped = Stretch::segment(rest, stiffness, mass, 0.0);
        undamped.stretch_to(1.0);
        let plain = undamped.assemble(dt).expect("the rod assembles");
        let mut damped = Stretch::segment(rest, stiffness, mass, beta);
        damped.stretch_to(1.0);
        let inflated = damped.assemble(dt).expect("the rod assembles");
        assert_close(
            inflated.vertex_force(1),
            plain.vertex_force(1),
            "the damping force at a zero step",
        );
        let scale = 1.0 + f64::from(beta) / f64::from(dt);
        let mut probe = vec![0.0f64; 6];
        probe[0] = 1.0;
        probe[3] = -1.0;
        let measured = inflated.quadratic_form(&probe);
        let expected = scale * plain.quadratic_form(&probe);
        assert!(
            (measured - expected).abs() <= 1e-3 * expected.abs(),
            "the damped Hessian's curvature is {measured} against the derived {expected}"
        );
    }

    #[test]
    fn a_fixed_rod_carries_no_stretch_energy() {
        // `EdgeProp::fixed` is the dispatch's own test: every vertex of the rod
        // is prescribed, so its rows leave the Newton system and an energy
        // written about it would be assembled and then eliminated.
        let mut fixture = Stretch::segment(0.8, 1500.0, 0.5, 0.0);
        fixture.stretch_to(1.0);
        fixture.scene.data.prop.edge.as_mut_slice()[0].fixed = true;
        let assembled = fixture.assemble(0.01).expect("the rod assembles");
        for vertex in 0..2 {
            assert_close(
                assembled.vertex_force(vertex),
                [0.0, 0.0, 0.0],
                "a fixed rod",
            );
        }
        assert_eq!(
            assembled.dense.iter().filter(|value| **value != 0.0).count(),
            0,
            "a fixed rod put a block in the matrix"
        );
    }

    #[test]
    fn the_stretch_walks_the_rod_prefix_and_not_a_shells_edges() {
        // THE TRAP THIS COVERS. `mesh.edge` carries every face's edges after
        // the rods, and `prop.edge` and `param_arrays.edge` span that whole
        // array, so a walk over `edge.size` would assemble a stretch energy on
        // every shell's edges. A face edge has a zero mass and a LIVE stiffness
        // inherited from its faces' material, which is why the mass alone does
        // not save it: the weight would be zero but the element would still be
        // in the matrix, and this fixture's face edge is stretched well past
        // its rest length.
        let edges = [Vec2u::new(0, 1), Vec2u::new(2, 3)];
        let mut scene = TestScene::new(4).with_edges(&edges);
        scene.place(0, 0.0, 0.0, 0.0);
        scene.place(1, 1.0, 0.0, 0.0);
        scene.place(2, 0.0, 1.0, 0.0);
        scene.place(3, 2.0, 1.0, 0.0);
        scene.data.rod_count = 1;
        scene.data.surface_vert_count = 4;
        scene.data.param_arrays.edge = CVec::from(&[rod_material(1500.0)][..]);
        for prop in scene.data.prop.edge.as_mut_slice() {
            prop.length = 0.5;
            prop.initial_length = 0.25;
            prop.mass = 0.5;
        }
        install_rod_adjacency(&mut scene.data, 4, &edges);
        install_pattern(&mut scene.data, 4);
        let mut state = SolverState::default();
        // Safety: the scene lives in its box for the whole test.
        let mut device = host_device();
        unsafe { state.allocate(&mut device, &scene.data) }
            .expect("the fixture scene allocates");
        clear_force(&mut device, &mut state);
        // Safety: as above.
        let (force, dense) = unsafe {
            // SEEDED FROM THE SCENE, NOT COPIED FROM `positions`. The
                // fixture deforms the scene AFTER `allocate` seeded the device
                // buffers, so `positions` still holds the build-time pose and a
                // device copy would hand the kernel the undeformed shape.
                //
                // Safety: the scene is live and holds `vertices` triples.
                let pose = crate::driver::state::slice_or_empty(
                    scene.data.vertex.curr.data as *const f32,
                    3 * state.sizes.vertices,
                );
                state
                    .positions
                    .seed(&mut device, pose)
                    .expect("the fixture seeds the committed pose");
                state
                    .eval_x
                    .seed(&mut device, pose)
                    .expect("the fixture seeds the iterate");
            let mut fixed = FixedCsr::adopt_from_dataset(&mut device, state.fixed_pattern_refs(), &scene.data, Default::default())
                .expect("the fixture pattern is adopted");
            rod_stretch(&mut device, &scene.data, &mut state, &mut fixed, 0.01)
                .expect("the rod assembles");
            // The dense reference reads the mirror, which the assembly
            // dispatch above invalidated. Download before reading.
            fixed.download(&mut device).expect("the matrix reads back");
            (force_of(&mut device, &mut state), dense_of(&fixed, 4))
        };
        assert!(
            force[0..6].iter().any(|value| *value != 0.0),
            "the rod itself carried no force, so this fixture measures nothing"
        );
        for vertex in 2..4usize {
            assert_eq!(
                [force[3 * vertex], force[3 * vertex + 1], force[3 * vertex + 2]],
                [0.0, 0.0, 0.0],
                "the face edge's vertex {vertex} carries a stretch force"
            );
        }
        for row in 6..12usize {
            for column in 6..12usize {
                assert_eq!(
                    dense[12 * row + column],
                    0.0,
                    "the face edge put a block at ({row}, {column})"
                );
            }
        }
    }

    #[test]
    fn a_rod_with_a_zero_rest_length_stops_the_run() {
        // The shared body divides by the rest length and has no guard for it,
        // so a zero is a scene whose strain is not a number. `scene.rs` asserts
        // a positive drawn length at build and scales it by `length-factor`, so
        // this reports a factor of zero or a `DataSet` from somewhere else.
        let mut fixture = Stretch::segment(1.0, 1500.0, 0.5, 0.0);
        fixture.scene.data.prop.edge.as_mut_slice()[0].length = 0.0;
        let error = fixture
            .assemble(0.01)
            .expect_err("a zero rest length must stop the run");
        assert!(
            error.detail.contains("rest length"),
            "the report does not name the rest length: {}",
            error.detail
        );
    }

    /// One interior rod vertex with the whole bending assembly around it.
    ///
    /// THREE VERTICES AND TWO SEGMENTS. Vertex 1 is the interior one: it has
    /// exactly two incident edges and no incident face, which is the rod
    /// bending site test. The stencil is then `(0, 1, 2)`,
    /// with `0` from the FIRST incident edge.
    ///
    /// THE TWO SEGMENT LENGTHS AND THE LUMPED MASS ARE AUTHORED, exactly as the
    /// shell hinge fixture authors its shared-edge length and incident area, so
    /// a test can vary the stiffness coefficient without moving the geometry
    /// the turning angle is read from.
    struct RodBend {
        scene: TestScene,
        state: SolverState,
        /// ONE DEVICE FOR THE FIXTURE'S LIFE, as `Bending` above.
        device: HostDevice,
    }

    /// The rest pose, in meters, every coordinate dyadic and so exactly
    /// representable and exact in fp32. `FAR_BENT` puts the turning angle at `3 pi / 4`
    /// against a rest angle of `pi`, which is a quarter turn from straight.
    const NEAR: [f32; 3] = [-1.0, 0.0, 0.0];
    const INTERIOR: [f32; 3] = [0.0, 0.0, 0.0];
    const FAR_BENT: [f32; 3] = [0.5, 0.5, 0.0];
    /// A second bend, so a test needing two DIFFERENT poses of one site has one
    /// that is not the straight rod. Straight is the degenerate case for the
    /// analytic eigensystem rather than merely a special value.
    const FAR_START: [f32; 3] = [0.5, 0.25, 0.0];
    const FAR_STRAIGHT: [f32; 3] = [1.0, 0.0, 0.0];

    /// `bend` for a stiffness scalar of 2.5, which is DELIBERATELY NOT ONE:
    /// `rod_bend_stiffness` is `bend * mass * ref^2 / voronoi^2`, and with
    /// the fixture's unit lengths and unit mass a `bend` of 1e4 would put the
    /// scalar at exactly 1.0, where a lost stiffness scale is invisible.
    const BEND_2_5: f32 = 2.5e4;

    impl RodBend {
        fn strand(far: [f32; 3], bend: f32) -> Self {
            Self::configured(far, bend, 0.0, 1.0, 1.0, 1.0)
        }

        /// Every quantity the stiffness scalar is composed from.
        fn configured(
            far: [f32; 3],
            bend: f32,
            bend_damping: f32,
            length0: f32,
            length1: f32,
            vertex_mass: f32,
        ) -> Self {
            let edges = [Vec2u::new(0, 1), Vec2u::new(1, 2)];
            let mut scene = TestScene::new(3).with_edges(&edges);
            scene.place(0, NEAR[0], NEAR[1], NEAR[2]);
            scene.place(1, INTERIOR[0], INTERIOR[1], INTERIOR[2]);
            scene.place(2, far[0], far[1], far[2]);
            scene.data.rod_count = 2;
            scene.data.surface_vert_count = 3;
            scene.data.param_arrays.edge = CVec::from(
                &[EdgeParam {
                    bend,
                    bend_damping,
                    ..rod_material(1000.0)
                }][..],
            );
            {
                let props = scene.data.prop.edge.as_mut_slice();
                props[0].length = length0;
                props[0].initial_length = length0;
                props[1].length = length1;
                props[1].initial_length = length1;
                // The stretch is not under test here and is switched off by the
                // dispatch's own flag rather than by a zero stiffness, which
                // would be the bending gate's spelling and not the stretch's.
                for prop in props.iter_mut() {
                    prop.fixed = true;
                    prop.mass = 0.0;
                }
            }
            for vertex in scene.data.prop.vertex.as_mut_slice() {
                vertex.mass = vertex_mass;
                vertex.rest_bend_angle = std::f32::consts::PI;
            }
            install_rod_adjacency(&mut scene.data, 3, &edges);
            install_pattern(&mut scene.data, 3);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            Self {
                scene,
                state,
                device,
            }
        }

        fn assemble_at(&mut self, iterate: &[f32], dt: f32) -> FatalResult<Assembled> {
            let RodBend {
                scene,
                state,
                device,
            } = self;
            clear_force(device, state);
            // `positions` IS the start-of-step pose, so it tracks the scene at
            // every fixture entry. A test that wants the iterate somewhere else
            // seeds `eval_x` separately below; one that does not gets both at
            // the scene, which is what a step would see.
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            state.eval_x.seed(device, iterate).expect("the fixture seeds the buffer");
            // Safety: the scene is live, the state was allocated for it, and the
            // matrix borrows the scene's own pattern tables.
            let (force, dense) = unsafe {
                let mut fixed = FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default())?;
                rod_bend(device, &scene.data, state, &mut fixed, dt)?;
                // The dense reference reads the mirror, which the assembly
                // dispatch above invalidated. Download before reading.
                fixed.download(device)?;
                (force_of(device, state), dense_of(&fixed, 3))
            };
            Ok(Assembled {
                force,
                dense,
                width: 9,
            })
        }

        fn assemble(&mut self, dt: f32) -> FatalResult<Assembled> {
            let iterate = positions_of(&self.scene);
            self.assemble_at(&iterate, dt)
        }

        /// The stiffness scalar the last assembly formed.
        ///
        /// AN ORACLE, DISPATCHED, rather than a read of a production buffer.
        /// The assembly forms this in the element's own thread and stores it
        /// nowhere, so a fixture reading a staged array would be reading an
        /// artifact of a decomposition this assembly does not perform. It
        /// dispatches the
        /// SAME shared body the assembly calls, on the fixture's own records,
        /// which is what keeps the two from drifting.
        fn stiffness(&mut self) -> f64 {
            // Safety: the scene lives in its box for the fixture's whole life
            // and both arrays hold their stated lengths.
            let (props, params): (&[EdgeProp], &[EdgeParam]) = unsafe {
                (
                    scene::slice(&self.scene.data.prop.edge),
                    scene::slice(&self.scene.data.param_arrays.edge),
                )
            };
            let interior = self.state.rod_bend.node.host()[1] as usize;
            let (first, second) = (
                self.state.rod_bend.edge[0] as usize,
                self.state.rod_bend.edge[1] as usize,
            );
            let mass = self.state.prop_vertex.host()[interior].mass;
            // Safety: both shims are the shared bodies compiled into this test
            // binary, and take scalars only.
            let bend = unsafe {
                rod_bend_segment_average_abi(
                    params[props[first].param_index as usize].bend,
                    params[props[second].param_index as usize].bend,
                )
            };
            let device = &mut self.device;
            let mut put = |value: f32, label: &'static str| {
                let mut b = ppf_cts_compute::StagedBuffer::<f32>::default();
                b.size(device, 1, ppf_cts_compute::AllocLabel(label)).expect("sized");
                b.at()[0] = value;
                let h = b.upload_span(device, 1).expect("uploaded");
                (b, h)
            };
            let (_b0, bend_h) = put(bend, "oracle.bend");
            let (_b1, mass_h) = put(mass, "oracle.mass");
            let (_b2, l0_h) = put(props[first].length, "oracle.length0");
            let (_b3, l1_h) = put(props[second].length, "oracle.length1");
            let mut out = ppf_cts_compute::ReadbackBuffer::<f32>::default();
            out.size(device, 1, ppf_cts_compute::AllocLabel("oracle.stiffness"))
                .expect("sized");
            let args = crate::driver::kernels::RodBendStiffnessArgs {
                bend: bend_h,
                mass: mass_h,
                length0: l0_h,
                length1: l1_h,
                stiffness: out.handle(),
                count: 1,
                seam_arena_count: 0,
            };
            // Safety: every handle outlives the dispatch.
            unsafe { device.launch("oracle.rod_bend.stiffness", &args, 1) }
                .expect("the stiffness oracle dispatches");
            out.download(device).expect("the oracle reads back");
            f64::from(out.host()[0])
        }

        /// The turning angle at `iterate`, read through the shared body.
        fn angle_at(&self, iterate: &[f32]) -> f64 {
            // THE FIXTURE'S OWN GEOMETRY, not the order the assembly
            // enumerated. Vertex 1 is the interior one here by construction, so
            // an assembly that put a different vertex in the middle measures a
            // different energy and the comparisons below see it; reading the
            // enumeration back would make both halves move together and hide
            // exactly that.
            let mut device = crate::driver::launch::host_device();
            // THE STENCIL IS A DEVICE ALLOCATION, and it has to be cut from
            // THIS target rather than borrowed from the fixture's: a
            // `HostDevice` keeps its arenas per instance, so a handle from
            // another one would resolve against the wrong base and read
            // whatever sits at that offset.
            let stencil = one_element_buffer(&mut device, "test.rod_bend.stencil", &[0u32, 1, 2]);
            let mut angle_out =
                one_element_buffer(&mut device, "test.rod_bend.angle", &[0.0f32]);
            // Dispatched the way production dispatches it; the shell oracle
            // above states why.
            let args = crate::driver::kernels::RodBendAngleArgs {
                // Safety: `iterate` is three position components per vertex and
                // outlives the dispatch below, the stencil names three in-range
                // vertices of it, and the angle is read back off its own device
                // allocation once the call has returned.
                x: crate::driver::state::position_block(&mut device, iterate, "test.iterate").handle(),
                node_index: stencil.handle(),
                vertex_count: (iterate.len() / 3) as u32,
                angle: angle_out.handle(),
                count: 1,
                seam_arena_count: 0,
            };
            // Safety: as the shell oracle above.
            unsafe { device.launch("test.rod_bend.angle", &args, 1) }
                .expect("the turning angle dispatches");
            let angle = angle_out
                .read_one(&mut device, 0)
                .expect("the turning angle reads back");
            angle_out
                .free(&mut device)
                .expect("the angle buffer frees");
            f64::from(angle)
        }

        /// `E = 1/2 k (theta - theta0)^2` at `iterate`, in `f64`.
        fn energy_at(&self, iterate: &[f32], stiffness: f64, rest_angle: f64) -> f64 {
            let difference = self.angle_at(iterate) - rest_angle;
            0.5 * stiffness * difference * difference
        }
    }

    #[test]
    fn the_rod_bending_stencil_is_the_interior_vertex_between_its_two_neighbors() {
        // `(j, i, k)` with `j` from the FIRST incident edge, which is the order
        // `builder.rs` registers the `(j, k)` coupling in. Any other order asks
        // the fixed sparsity for a block it does not carry, and on a symmetric
        // fixture that is invisible.
        let mut fixture = RodBend::strand(FAR_BENT, BEND_2_5);
        fixture.assemble(0.01).expect("the site assembles");
        assert_eq!(
            fixture.state.rod_bend.node.host()[0..3],
            [0, 1, 2],
            "the stencil is not (j, i, k)"
        );
        assert_eq!(fixture.state.sizes.rod_bend_sites, 1);
    }

    #[test]
    fn a_straight_rod_carries_no_bending_force_and_no_hessian() {
        // The rest angle is `pi`, which builder.rs writes for every vertex, so
        // a rod drawn straight sits at its own rest angle and the force is zero
        // for the same reason the shell hinge's is. The Hessian is zero for a
        // SECOND reason that outlives the first: the two segments are exactly
        // antiparallel, so their cross product vanishes and the binormal the
        // analytic eigensystem is built on is undefined. The shared body
        // answers that with a zero Hessian on every backend, so a perfectly
        // straight rod's first Newton system carries no bending stiffness at
        // all and it appears as soon as the pose leaves the line.
        let mut fixture = RodBend::strand(FAR_STRAIGHT, BEND_2_5);
        let assembled = fixture.assemble(0.01).expect("the site assembles");
        for vertex in 0..3 {
            assert_close(
                assembled.vertex_force(vertex),
                [0.0, 0.0, 0.0],
                "a straight rod",
            );
        }
        assert_eq!(
            assembled.dense.iter().filter(|value| **value != 0.0).count(),
            0,
            "a straight rod put a bending block in the matrix"
        );
    }


    #[test]
    fn the_rod_bending_force_is_the_gradient_of_the_turning_angle_energy() {
        // All nine components, against a central difference of
        // `1/2 k (theta - theta0)^2` built from the shared ANGLE body. The
        // force is `k (theta - theta0) grad(theta)` with `grad(theta)` a
        // separate closed form, so this compares two independent derivations.
        // A permuted stencil, a lost stiffness scale or a sign error all fail
        // it.
        let mut fixture = RodBend::strand(FAR_BENT, BEND_2_5);
        let base = positions_of(&fixture.scene);
        let assembled = fixture.assemble(0.01).expect("the site assembles");
        let stiffness = fixture.stiffness();
        assert!(stiffness > 0.0, "the fixture's site does not bend");
        let rest = f64::from(std::f32::consts::PI);
        for component in 0..9usize {
            let mut forward = base.clone();
            let mut backward = base.clone();
            forward[component] += FD_STEP;
            backward[component] -= FD_STEP;
            let derivative = (fixture.energy_at(&forward, stiffness, rest)
                - fixture.energy_at(&backward, stiffness, rest))
                / (2.0 * FD_STEP_F64);
            let actual = f64::from(assembled.force[component]);
            let scale = derivative.abs().max(1e-2);
            assert!(
                (actual - derivative).abs() <= 5e-3 * scale,
                "component {component}: the assembly says {actual} and the energy's central \
                 difference says {derivative}"
            );
        }
    }

    #[test]
    fn the_rod_bending_force_sums_to_zero_over_the_three_nodes() {
        // The turning angle depends on the three positions only through two
        // differences, so translating the site changes nothing.
        let mut fixture = RodBend::strand(FAR_BENT, BEND_2_5);
        let assembled = fixture.assemble(0.01).expect("the site assembles");
        for k in 0..3 {
            let total = assembled.force[k] + assembled.force[3 + k] + assembled.force[6 + k];
            assert!(
                total.abs() <= 1e-4,
                "component {k} of the site's net force is {total}"
            );
        }
    }

    #[test]
    fn the_rod_bending_hessian_is_the_angle_gradient_outer_product_at_the_rest_angle() {
        // At `theta == theta0` the exact Hessian is `k g g^T` with
        // `g = grad(theta)`: the term carrying the angle's own second
        // derivative is multiplied by `theta - theta0` and vanishes. That
        // matrix is already PSD, so the Wu and Kim projection returns it
        // unchanged and the assembled blocks must equal it. This is what pins
        // the nine blocks to the right pairs of vertices; the force test above
        // cannot, because a force is one vector per node.
        let mut fixture = RodBend::strand(FAR_BENT, BEND_2_5);
        let base = positions_of(&fixture.scene);
        // The rest angle IS this pose's angle, so the site is at rest.
        let mut probe = RodBend::strand(FAR_BENT, BEND_2_5);
        probe.assemble(0.01).expect("the site assembles");
        let here = probe.angle_at(&base);
        for vertex in fixture.scene.data.prop.vertex.as_mut_slice() {
            vertex.rest_bend_angle = here as f32;
        }
        let assembled = fixture.assemble(0.01).expect("the site assembles");
        let stiffness = fixture.stiffness();
        let mut gradient = [0.0f64; 9];
        for component in 0..9usize {
            let mut forward = base.clone();
            let mut backward = base.clone();
            forward[component] += FD_STEP;
            backward[component] -= FD_STEP;
            gradient[component] =
                (fixture.angle_at(&forward) - fixture.angle_at(&backward)) / (2.0 * FD_STEP_F64);
        }
        let scale = gradient
            .iter()
            .fold(0.0f64, |acc, value| acc.max(value.abs()));
        assert!(scale > 0.1, "the angle gradient is degenerate at this pose");
        for row in 0..9usize {
            for column in 0..9usize {
                let expected = stiffness * gradient[row] * gradient[column];
                let actual = f64::from(assembled.dense[9 * row + column]);
                assert!(
                    (actual - expected).abs() <= 5e-3 * stiffness * scale * scale,
                    "block entry ({row}, {column}) is {actual} against the derived {expected}"
                );
            }
        }
    }

    #[test]
    fn the_rod_bending_hessian_is_positive_semidefinite_where_the_exact_one_is_not() {
        // Away from the rest angle the exact Hessian carries the angle's second
        // derivative and is generally indefinite. The analytic eigensystem
        // drops every non-positive mode, so no probe can find a negative
        // curvature, and `pAp <= 0` hard-errors on the strength of that.
        let mut fixture = RodBend::strand(FAR_START, BEND_2_5);
        let assembled = fixture.assemble(0.01).expect("the site assembles");
        let mut worst = f64::INFINITY;
        // A deterministic spread of probes rather than one direction: the
        // negative mode of an unprojected turning-angle Hessian is not axis
        // aligned.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..256 {
            let mut probe = vec![0.0f64; 9];
            for slot in probe.iter_mut() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *slot = ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            }
            worst = worst.min(assembled.quadratic_form(&probe));
        }
        assert!(
            worst >= -1e-6,
            "a probe found curvature {worst}, so the PSD projection was lost"
        );
    }

    #[test]
    fn the_bending_stiffness_halves_when_the_segment_length_and_lumped_mass_double() {
        // RESOLUTION INDEPENDENCE, as a ratio rather than as a formula. The
        // per-vertex stiffness is `bend * mass * ref^2 / voronoi^2`, and
        // `builder.rs` lumps half of each incident segment's mass onto the
        // vertex, so halving a rod's segment count doubles both the Voronoi
        // length and the lumped mass and the coefficient must fall by two. A
        // coefficient of `bend * mass` alone would DOUBLE instead, leaving two
        // resolutions of one physical rod describing different rods.
        let mut fine = RodBend::configured(FAR_BENT, BEND_2_5, 0.0, 1.0, 1.0, 1.0);
        fine.assemble(0.01).expect("the site assembles");
        let mut coarse = RodBend::configured(FAR_BENT, BEND_2_5, 0.0, 2.0, 2.0, 2.0);
        coarse.assemble(0.01).expect("the site assembles");
        let ratio = coarse.stiffness() / fine.stiffness();
        assert!(
            (ratio - 0.5).abs() <= 1e-5,
            "doubling the segment length and the lumped mass moved the stiffness by {ratio}, not \
             by a half"
        );
    }

    #[test]
    fn a_prescribed_interior_vertex_carries_no_rod_bending() {
        // The dispatch's own test: the interior vertex's row leaves the
        // Newton system entirely, so an energy written about it would be
        // assembled and then eliminated.
        let mut fixture = RodBend::strand(FAR_BENT, BEND_2_5);
        fixture.scene.data.prop.vertex.as_mut_slice()[1].fix_index = 1;
        let assembled = fixture.assemble(0.01).expect("the site assembles");
        for vertex in 0..3 {
            assert_close(
                assembled.vertex_force(vertex),
                [0.0, 0.0, 0.0],
                "a prescribed interior rod vertex",
            );
        }
        assert_eq!(
            assembled.dense.iter().filter(|value| **value != 0.0).count(),
            0,
            "a prescribed interior vertex put a bending block in the matrix"
        );
    }

    #[test]
    fn a_vertex_with_an_incident_face_is_not_a_rod_bending_site() {
        // The site test is two incident edges AND no incident face. A vertex
        // where a strand meets a sheet has both, and its bending is the shell
        // hinge's business rather than the rod's; counting it here would
        // assemble a turning-angle energy on top of the hinge's.
        let edges = [Vec2u::new(0, 1), Vec2u::new(1, 2)];
        // The face the adjacency names has to exist: `SolverState::allocate`
        // checks the vertex-face table against the face array, which is the
        // gate that turns a wild read in a shared body into a named error.
        let mut scene = TestScene::new(3)
            .with_edges(&edges)
            .with_faces(&[Vec3u::new(0, 1, 2)]);
        scene.place(0, NEAR[0], NEAR[1], NEAR[2]);
        scene.place(1, INTERIOR[0], INTERIOR[1], INTERIOR[2]);
        scene.place(2, FAR_BENT[0], FAR_BENT[1], FAR_BENT[2]);
        scene.data.rod_count = 2;
        scene.data.surface_vert_count = 3;
        scene.data.param_arrays.edge = CVec::from(
            &[EdgeParam {
                bend: 1.0e4,
                ..rod_material(1000.0)
            }][..],
        );
        for prop in scene.data.prop.edge.as_mut_slice() {
            prop.length = 1.0;
            prop.initial_length = 1.0;
            prop.fixed = true;
        }
        for vertex in scene.data.prop.vertex.as_mut_slice() {
            vertex.mass = 1.0;
            vertex.rest_bend_angle = std::f32::consts::PI;
        }
        install_rod_adjacency(&mut scene.data, 3, &edges);
        // One incident face on the interior vertex, which is all the site test
        // reads: the face array itself is not consulted.
        let faces: Vec<Vec<u32>> = vec![Vec::new(), vec![0], Vec::new()];
        scene.data.mesh.neighbor.vertex.face = CVecVec::from(&faces[..]);
        install_pattern(&mut scene.data, 3);
        let mut state = SolverState::default();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut host_device(), &scene.data) }
            .expect("the fixture scene allocates");
        assert_eq!(
            state.sizes.rod_bend_sites, 0,
            "a vertex with an incident face was counted as a rod bending site"
        );
    }

    #[test]
    fn a_fixed_pattern_without_the_neighbor_coupling_stops_the_run() {
        // THE STENCIL THIS PROJECT HAS ALREADY LOST ONCE. `(i, j)` and `(i, k)`
        // are edges and are in the pattern for that reason alone; `(j, k)` is
        // neither an edge nor a shell hinge, so it is there only because
        // `builder.rs` registers the rod-bend stencil. Without it
        // `FixedCSRMat::push` declines those two blocks and returns a verdict
        // its CUDA callers ignore, the rank-1 PSD bending Hessian arrives
        // indefinite, and the symptom is a `pAp <= 0` abort that damping and a
        // small `dt` hide.
        let mut fixture = RodBend::strand(FAR_BENT, BEND_2_5);
        // The pattern an edge walk alone would produce: the diagonal and the
        // two edges, and no (0, 2).
        install_rows(
            &mut fixture.scene.data,
            &[vec![0, 1], vec![1, 2], vec![2]],
        );
        let error = fixture
            .assemble(0.01)
            .expect_err("a missing rod-bend slot must stop the run");
        assert!(
            error.detail.contains("(0, 2)"),
            "the report does not name the dropped block: {}",
            error.detail
        );
    }

    #[test]
    fn the_rod_bending_damping_hessian_is_taken_at_the_start_of_the_step() {
        // The damping block is `(beta / dt) K_lag` with `K_lag` the stencil
        // Hessian at the START of the step, which is what makes the
        // turning-angle damping unconditionally dissipative. Evaluating it at
        // the iterate instead is a silent difference on a scene where the two
        // poses are close, so the test moves them far apart: the fixture sits
        // at `FAR_START` and the iterate is assembled at `FAR_BENT`.
        let dt = 0.01f32;
        let beta = 0.05f32;
        let mut damped = RodBend::configured(FAR_START, BEND_2_5, beta, 1.0, 1.0, 1.0);
        let mut iterate = positions_of(&damped.scene);
        {
            let bent = crate::driver::test_scene::position(FAR_BENT[0], FAR_BENT[1], FAR_BENT[2]);
            for k in 0..3 {
                iterate[6 + k] = bent[k];
            }
        }
        let with_damping = damped.assemble_at(&iterate, dt).expect("the site assembles");

        // The same two poses, undamped, plus a separate assembly whose ITERATE
        // is the start-of-step pose: that second one's Hessian is `K_lag`.
        let mut plain = RodBend::configured(FAR_START, BEND_2_5, 0.0, 1.0, 1.0, 1.0);
        let elastic = plain.assemble_at(&iterate, dt).expect("the site assembles");
        let mut lagged = RodBend::configured(FAR_START, BEND_2_5, 0.0, 1.0, 1.0, 1.0);
        let start = positions_of(&lagged.scene);
        let at_start = lagged.assemble_at(&start, dt).expect("the site assembles");

        let mut probe = vec![0.0f64; 9];
        probe[6] = 1.0;
        probe[7] = -1.0;
        probe[8] = 0.5;
        let measured = with_damping.quadratic_form(&probe);
        let expected = elastic.quadratic_form(&probe)
            + f64::from(beta) / f64::from(dt) * at_start.quadratic_form(&probe);
        assert!(
            (measured - expected).abs() <= 1e-3 * expected.abs(),
            "the damped curvature is {measured} against the derived {expected}, so the lagged \
             Hessian was not taken at the start of the step"
        );
        // AND THE ITERATE'S OWN HESSIAN IS THE WRONG ONE, so the assertion above
        // is not satisfied by both readings. This is the comparison that fails
        // if the second evaluation is dropped.
        let wrong = elastic.quadratic_form(&probe) * (1.0 + f64::from(beta) / f64::from(dt));
        assert!(
            (measured - wrong).abs() > 1e-2 * wrong.abs(),
            "the two poses give the same curvature here, so this fixture cannot tell them apart"
        );
    }

    // -----------------------------------------------------------------------
    // Strain limiting: the shell term, the rod term, the two times of impact
    // and the `max_sigma` indicator.
    // -----------------------------------------------------------------------

    // A LAUNCHER NAMED IN A TEST, WHICH IS THE ONE PLACE THAT IS NOT A DRIVER
    // REACHING AROUND THE SEAM.
    //
    // It takes a thread range, so unlike the scalar queries this module's
    // production half stopped naming, it IS a dispatch and would go through
    // `Device::launch` anywhere above. Here it is an ORACLE: the strain-limit
    // tests need the barrier's own gradient at one gap to derive an expected
    // value, and the range they dispatch it over is one element. Routing that
    // through the table, the record check and the launch would compare the
    // assembly against a reference that travelled the same machinery, which is
    // the independence `barrier_gradient` below exists for.
    //
    // No production caller in this file: the assembly reaches the same body
    // through the dispatched diff-table stage, which is stated at
    // `barrier_gradient`.
    extern "C" {
        fn barrier_gradient_entry(
            gap: *const f32,
            ghat: f32,
            offset: f32,
            kind: u32,
            gradient: *mut f32,
            begin: u32,
            end: u32,
        );
    }

    /// `Barrier::Cubic`, which is the registry's default and the shape every
    /// expected value below is derived against.
    const CUBIC: u32 = crate::data::Barrier::Cubic as u32;

    /// The barrier's own gradient at one gap, through the shared body.
    ///
    /// THE DERIVATION USES IT AND THE ASSEMBLY DOES NOT REACH IT THIS WAY. The
    /// assembly goes gap -> diff table -> spectral force -> material-frame
    /// converter -> stiffness scale -> scatter, none of which this call touches,
    /// so an expected value built from it is a check of that chain rather than a
    /// restatement of it. What it avoids is hard-coding a number that depends on
    /// which barrier shape the scene selected.
    fn barrier_gradient(gap: f32, ghat: f32) -> f32 {
        let gaps = [gap];
        let mut out = [0.0f32];
        // Safety: one element in, one out, both on the stack.
        unsafe {
            barrier_gradient_entry(
                gaps.as_ptr(),
                ghat,
                0.0,
                CUBIC,
                out.as_mut_ptr(),
                0,
                1,
            )
        };
        out[0]
    }

    fn limited_material(strainlimit: f32, shrink_x: f32, shrink_y: f32) -> FaceParam {
        FaceParam {
            strainlimit,
            shrink_x,
            shrink_y,
            ..FaceParam::default()
        }
    }

    /// One triangle with the whole strain-limit assembly around it.
    ///
    /// THE REST SHAPE IS THE UNIT RIGHT TRIANGLE IN THE XY PLANE, as the
    /// membrane fixture's is, so `inv_rest2x2` is the identity, `F` is exactly
    /// the pair of deformed edge vectors and `convert_force`'s three material
    /// gradients are `(-1, -1)`, `(1, 0)` and `(0, 1)`. A stretch along x then
    /// puts `F` at `diag(a, 1)`, whose SVD is exact, and every expected value
    /// below follows by hand.
    struct ShellStrain {
        scene: TestScene,
        state: SolverState,
        /// ONE DEVICE FOR THE FIXTURE'S LIFE, on the grounds
        /// [`Bending::device`] states: a handle names an arena of the allocator
        /// that opened it, so sizing on one device and dispatching on another
        /// resolves a handle against a table that never held it.
        device: HostDevice,
    }

    impl ShellStrain {
        fn unit_triangle(material: FaceParam, mass: f32) -> Self {
            let mut scene = TestScene::new(3).with_faces(&[Vec3u::new(0, 1, 2)]);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.0, 0.0, 0.0);
            scene.place(2, 0.0, 1.0, 0.0);
            scene.data.param_arrays.face = CVec::from(&[material][..]);
            scene.data.inv_rest2x2 = CVec::from(&[Mat2x2f::identity()][..]);
            scene.data.prop.face.as_mut_slice()[0].mass = mass;
            install_pattern(&mut scene.data, 3);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            Self { scene, state, device }
        }

        /// Move vertex 1 along x, which puts `F` at `diag(factor, 1)`.
        ///
        /// The factors used below are dyadic, so they are exactly representable
        /// and exact in fp32 and the singular values are the factors themselves.
        fn scale_along_x(&mut self, factor: f32) {
            self.scene.place(1, factor, 0.0, 0.0);
            // The scene moved, so the device positions must move with it.
            crate::driver::state::reseed_positions(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
        }

        /// Assemble at `iterate`, against a reference matrix whose three
        /// diagonal blocks are `reference_scale * I`.
        ///
        /// A DIAGONAL REFERENCE IS WHAT MAKES THE STIFFNESS DERIVABLE: the 9x9
        /// the shared body contracts is then `k * I`, so
        /// `shape.dot(local * shape)` is `k` times the squared distance of the
        /// face's three vertices from their own centroid, which a test can form
        /// from the pose without touching the assembly.
        fn assemble_at(
            &mut self,
            iterate: &[f32],
            reference_scale: f32,
        ) -> FatalResult<Assembled> {
            let ShellStrain { scene, state, device } = self;
            clear_force(device, state);
            // `positions` IS the start-of-step pose, so it tracks the scene at
            // every fixture entry. A test that wants the iterate somewhere else
            // seeds `eval_x` separately below; one that does not gets both at
            // the scene, which is what a step would see.
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            state.eval_x.seed(device, iterate).expect("the fixture seeds the buffer");
            // Safety: the scene is live, the state was allocated for it, and both
            // matrices borrow the scene's own pattern tables.
            let (force, dense) = unsafe {
                let mut reference = FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default())?;
                if reference_scale != 0.0 {
                    let rows = [0u32, 1, 2];
                    let columns = [0u32, 1, 2];
                    let mut blocks = [0.0f32; 27];
                    for vertex in 0..3usize {
                        for k in 0..3usize {
                            blocks[9 * vertex + 4 * k] = reference_scale;
                        }
                    }
                    let mut stored = [0u32; 3];
                    reference.push_blocks(
                        device,
                        &mut state.push,
                        &rows,
                        &columns,
                        &blocks,
                        &mut stored,
                    )?;
                    assert_eq!(stored, [1, 1, 1], "the fixture's reference blocks must land");
                }
                let mut fixed = FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default())?;
                shell_strain(
                    device,
                    &scene.data,
                    EPS,
                    CUBIC,
                    state,
                    &mut reference,
                    &mut fixed,
                )?;
                // The dense reference reads the mirror, which the assembly
                // dispatch above invalidated. Download before reading.
                fixed.download(device)?;
                (force_of(device, state), dense_of(&fixed, 3))
            };
            Ok(Assembled {
                force,
                dense,
                width: 9,
            })
        }

        fn assemble(&mut self) -> FatalResult<Assembled> {
            let iterate = positions_of(&self.scene);
            self.assemble_at(&iterate, 0.0)
        }

        /// The squared distance of the START-OF-STEP pose's three vertices from
        /// their own centroid, in `f64`, which is what the shared stiffness body
        /// contracts the reference matrix along.
        fn shape_squared(&self) -> f64 {
            let curr = self.scene.data.vertex.curr.as_slice();
            let mut center = [0.0f64; 3];
            for vertex in 0..3usize {
                for k in 0..3usize {
                    center[k] += f64::from(curr[vertex][k]) / 3.0;
                }
            }
            let mut total = 0.0f64;
            for vertex in 0..3usize {
                for k in 0..3usize {
                    let offset = f64::from(curr[vertex][k]) - center[k];
                    total += offset * offset;
                }
            }
            total
        }
    }

    #[test]
    fn an_unstretched_face_carries_no_strain_limit_term() {
        // The entry gate is `svd.S.maxCoeff() > 0` on the SHIFTED singular
        // values, so a face at its rest shape has no term at all. That is not a
        // rounding statement: the barrier's own table skips a non-positive
        // strain, and this checks the face never reaches the scatter either.
        let mut fixture = ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
        let assembled = fixture.assemble().expect("the assembly runs");
        for vertex in 0..3 {
            for k in 0..3 {
                assert_eq!(
                    assembled.vertex_force(vertex)[k],
                    0.0,
                    "vertex {vertex} component {k} carries a force at the rest shape"
                );
            }
        }
        assert!(
            assembled.dense.iter().all(|value| *value == 0.0),
            "an unstretched face wrote a Hessian block"
        );
    }

    /// A PINNED FACE AND A REST-EXCLUDED ONE CARRY NO STRAIN TERM EITHER.
    ///
    /// THE LIMITER'S GATE IS NOT THE MEMBRANE'S, and the difference is
    /// deliberate: `gather_face_strain_gate` has NO `collider` test, because a
    /// spring-held collider's vertices are free and the limit it was authored
    /// with still applies to them. Its three exclusions are `fixed`,
    /// `rest_excluded` and `strainlimit > 0`, and asserting a collider here
    /// fails against correct code, which is how this test found that its first
    /// version had the membrane's gate written into it.
    ///
    /// `a_face_with_no_strain_limit_carries_no_term` below covers the third of
    /// them. These two had none,
    /// and they are the two that a conversion would put at risk: all three
    /// strain dispatches take `count = faces`, so the exclusion is carried by
    /// the SEEDED ZERO in `authored_limit`, `effective_limit` and `mass`, not by
    /// the dispatch extent. A pass that read `FaceParam` and `FaceProp` off the
    /// device instead would hand a collider's face its real limit unless it made
    /// these tests itself.
    ///
    /// MEASURED on the elastic side: the build with the gate missing passed
    /// every unit test, every static gate and all 23 scenes, and `disp_at_5`
    /// could not separate it from a correct one. So this exists BEFORE the
    /// conversion rather than after it.
    ///
    /// TWO FACES, for the same reason: with one face the
    /// candidate list empties and the assembly returns before dispatching, so
    /// the assertion would hold with the gate deleted.
    #[test]
    fn a_pinned_and_a_rest_excluded_face_carry_no_strain_term() {
        fn pair(flag: Option<usize>) -> ShellStrain {
            let mut scene = TestScene::new(4)
                .with_faces(&[Vec3u::new(0, 1, 2), Vec3u::new(1, 3, 2)]);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.25, 0.0, 0.0);
            scene.place(2, 0.0, 1.0, 0.0);
            scene.place(3, 1.25, 1.0, 0.0);
            scene.data.param_arrays.face =
                CVec::from(&[limited_material(0.05, 1.0, 1.0)][..]);
            scene.data.inv_rest2x2 =
                CVec::from(&[Mat2x2f::identity(), Mat2x2f::identity()][..]);
            for face in scene.data.prop.face.as_mut_slice() {
                face.mass = 2.0;
            }
            if let Some(case) = flag {
                let face = &mut scene.data.prop.face.as_mut_slice()[0];
                if case == 0 {
                    face.fixed = true;
                } else {
                    face.rest_excluded = true;
                }
            }
            install_pattern(&mut scene.data, 4);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            ShellStrain { scene, state, device }
        }

        // THE UNEXCLUDED TWIN FIRST: a fixture carrying no strain force would
        // pass both exclusions below while testing nothing.
        let mut twin = pair(None);
        let assembled = twin.assemble().expect("the twin assembles");
        let reference = assembled.vertex_force(0);
        assert!(
            reference.iter().any(|component| component.abs() > 1.0e-3),
            "the unexcluded twin must carry a real strain force at vertex 0, or \
             the exclusions below prove nothing: got {reference:?}"
        );

        for case in 0..2usize {
            let mut fixture = pair(Some(case));
            let excluded = fixture.assemble().expect("the excluded face assembles");
            let gated = excluded.vertex_force(0);
            assert!(
                gated.iter().all(|component| *component == 0.0),
                "case {case}: vertex 0 belongs only to the excluded face and must \
                 carry no strain force, got {gated:?}"
            );
            let live = excluded.vertex_force(3);
            assert!(
                live.iter().any(|component| component.abs() > 1.0e-3),
                "case {case}: vertex 3 belongs to the face that is NOT excluded, \
                 so a zero here means the dispatch never ran and the assertion \
                 above proves nothing: got {live:?}"
            );
        }
    }

    #[test]
    fn a_face_with_no_strain_limit_carries_no_term() {
        // The dispatch's own gate: `strainlimit > 0`.
        let mut fixture = ShellStrain::unit_triangle(FaceParam::default(), 2.0);
        fixture.scale_along_x(1.25);
        let assembled = fixture.assemble().expect("the assembly runs");
        assert_eq!(assembled.vertex_force(1), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn the_strain_force_is_the_barrier_gradient_at_the_authored_limit() {
        // THE DERIVATION, and every later test varies one input of it.
        //
        // `F = diag(1.25, 1)`, so the singular values are `(1, 1.25)` ascending
        // and the shifted pair is `(0, 0.25)`. Only the second is positive, so
        // `deda = (0, -barrier_gradient(limit - 0.25, limit))`, and
        // `U diag(deda) V^T` puts that one number at `dedF(0, 0)`. The three
        // material gradients `(-1, -1)`, `(1, 0)`, `(0, 1)` then send it to
        // `-d` at vertex 0, `+d` at vertex 1 and nothing at vertex 2.
        //
        // The stiffness is `shape . (H shape) + mass / gap^2` with `H` zero
        // here, so it is `2.0 / 0.25^2 = 32`.
        let mut fixture = ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
        fixture.scale_along_x(1.25);
        let assembled = fixture.assemble().expect("the assembly runs");

        let deda = -barrier_gradient(0.25, 0.5);
        let stiffness = 2.0f32 / (0.25f32 * 0.25f32);
        let expected = stiffness * deda;
        assert!(
            expected.abs() > 1e-3,
            "the fixture must put a real force on the vertex, got {expected:e}"
        );

        let at_one = assembled.vertex_force(1);
        let at_zero = assembled.vertex_force(0);
        let at_two = assembled.vertex_force(2);
        assert!(
            (at_one[0] - expected).abs() <= TOLERANCE * expected.abs(),
            "vertex 1 carries {} against the derived {expected}",
            at_one[0]
        );
        assert!(
            (at_zero[0] + expected).abs() <= TOLERANCE * expected.abs(),
            "vertex 0 carries {} against the derived {}",
            at_zero[0],
            -expected
        );
        for k in 1..3 {
            assert!(at_one[k].abs() <= TOLERANCE * expected.abs());
            assert!(at_zero[k].abs() <= TOLERANCE * expected.abs());
        }
        for k in 0..3 {
            assert!(
                at_two[k].abs() <= TOLERANCE * expected.abs(),
                "vertex 2 is off the stretch axis and must carry nothing, got {}",
                at_two[k]
            );
        }
    }

    #[test]
    fn the_stiffness_contracts_the_reference_matrix_at_the_start_of_step_pose() {
        // TWO THINGS AT ONCE, and neither is visible without the other. The
        // stiffness's first term is the face's own assembled Hessian contracted
        // along its centered shape, which is what carries the surrounding
        // elasticity into the barrier's scale; and the pose that shape is taken
        // at is `data.vertex.curr`, not the Newton iterate.
        //
        // The fixture puts the two poses far apart: the start-of-step triangle
        // is twice the size of the iterate's, so a stiffness formed at the wrong
        // one is off by a factor of about three here.
        let mut fixture = ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
        fixture.scale_along_x(1.25);
        let iterate = positions_of(&fixture.scene);
        // The start-of-step pose, which only `curr` carries from here on.
        fixture.scene.place(1, 2.0, 0.0, 0.0);
        fixture.scene.place(2, 0.0, 2.0, 0.0);
        let start_shape = fixture.shape_squared();
        let iterate_shape = {
            let mut probe = ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
            probe.scale_along_x(1.25);
            probe.shape_squared()
        };
        assert!(
            (start_shape - iterate_shape).abs() > 0.5 * start_shape,
            "the two poses must differ enough for this test to distinguish them"
        );

        const REFERENCE: f32 = 3.0;
        let assembled = fixture
            .assemble_at(&iterate, REFERENCE)
            .expect("the assembly runs");

        let deda = -barrier_gradient(0.25, 0.5);
        let stiffness = f64::from(REFERENCE) * start_shape + 2.0 / (0.25 * 0.25);
        let expected = stiffness * f64::from(deda);
        let measured = f64::from(assembled.vertex_force(1)[0]);
        assert!(
            (measured - expected).abs() <= 1e-3 * expected.abs(),
            "vertex 1 carries {measured} against the derived {expected}; a stiffness formed \
             at the iterate would read {}",
            (f64::from(REFERENCE) * iterate_shape + 2.0 / (0.25 * 0.25)) * f64::from(deda)
        );
    }

    #[test]
    fn the_barrier_takes_the_authored_limit_and_the_stiffness_the_shrink_corrected_one() {
        // TRAP 1, and no shrink-free scene can see it. `shell_strain.kernel.cpp`
        // states it at its own definition: the table is handed the AUTHORED
        // `strainlimit` and the stiffness the value corrected for shrink, which
        // here are 0.5 and `(1 + 0.5) / 0.5 - 1 = 2.0`. Exchanging them changes
        // the barrier's gap from 0.25 to 1.75 and the stiffness's divisor from
        // 3.0625 to 0.0625, so the two readings differ by orders of magnitude.
        let mut fixture =
            ShellStrain::unit_triangle(limited_material(0.5, 0.5, 1.0), 2.0);
        fixture.scale_along_x(1.25);
        let assembled = fixture.assemble().expect("the assembly runs");

        let deda = -barrier_gradient(0.25, 0.5);
        let stiffness = 2.0f32 / (1.75f32 * 1.75f32);
        let expected = stiffness * deda;
        let swapped = {
            let deda_swapped = -barrier_gradient(1.75, 2.0);
            (2.0f32 / (0.25f32 * 0.25f32)) * deda_swapped
        };
        assert!(
            (expected - swapped).abs() > 0.1 * expected.abs(),
            "the fixture must separate the two readings, got {expected} against {swapped}"
        );
        let measured = assembled.vertex_force(1)[0];
        assert!(
            (measured - expected).abs() <= TOLERANCE * expected.abs(),
            "vertex 1 carries {measured} against the derived {expected}; the exchanged \
             assignment would read {swapped}"
        );
    }

    #[test]
    fn a_collider_face_is_still_strain_limited() {
        // TRAP 4. The membrane's dispatch excludes a collider and the limiter's
        // does NOT, and the two
        // must not be harmonized: a spring-held collider's vertices are free, so
        // the limit it was authored with is what stops its own mesh stretching
        // past it.
        let mut fixture = ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
        fixture.scene.data.prop.face.as_mut_slice()[0].collider = true;
        fixture.scale_along_x(1.25);
        let assembled = fixture.assemble().expect("the assembly runs");

        let expected = (2.0f32 / (0.25f32 * 0.25f32)) * -barrier_gradient(0.25, 0.5);
        let measured = assembled.vertex_force(1)[0];
        assert!(
            (measured - expected).abs() <= TOLERANCE * expected.abs(),
            "a collider face must carry the same term as any other, got {measured} against \
             {expected}"
        );
    }

    #[test]
    fn a_fixed_or_rest_excluded_face_carries_no_strain_limit_term() {
        // The other two halves of the same gate. `fixed` says every vertex is an
        // exact Dirichlet row, so the face cannot yield; `rest_excluded` says
        // `update_rest_shape` found its streamed rest shape near-singular this
        // frame, so a strain measured against it is not a number.
        for excluded in [0usize, 1] {
            let mut fixture =
                ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
            {
                let prop = &mut fixture.scene.data.prop.face.as_mut_slice()[0];
                if excluded == 0 {
                    prop.fixed = true;
                } else {
                    prop.rest_excluded = true;
                }
            }
            fixture.scale_along_x(1.25);
            let assembled = fixture.assemble().expect("the assembly runs");
            assert_eq!(
                assembled.vertex_force(1),
                [0.0, 0.0, 0.0],
                "an excluded face still carried a term (case {excluded})"
            );
        }
    }

    #[test]
    fn the_strain_hessian_is_psd_and_annihilates_a_translation() {
        // TWO PROPERTIES, and each catches a different defect.
        //
        // PSD is the one the solver's `pAp <= 0` guard rests on: the spectral
        // Hessian eigen-clamps in the singular-value basis, so the block it
        // returns is PSD by construction and the stiffness that scales it is a
        // non-negative number. A lost projection, or a stiffness that came back
        // negative, shows up here.
        //
        // ANNIHILATING A TRANSLATION is the structural one. The energy is a
        // function of the deformation gradient alone, which is translation
        // invariant, and `convert_hessian`'s three material gradients sum to
        // zero, so a uniform shift of all three vertices is exactly in the null
        // space. A block written to the wrong pair of vertices breaks that while
        // leaving the matrix symmetric and PSD.
        let mut fixture = ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
        fixture.scale_along_x(1.25);
        let assembled = fixture.assemble().expect("the assembly runs");

        let scale: f64 = assembled
            .dense
            .iter()
            .fold(0.0f64, |best, value| best.max(f64::from(value.abs())));
        assert!(scale > 1e-3, "the fixture must assemble a real Hessian");

        let probes: [[f64; 9]; 5] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, -0.7, 0.2, 0.9, 0.1, -0.4, -0.6, 0.5, 0.8],
            [-1.0, 0.25, 0.5, 0.75, -0.25, 0.125, 0.625, -0.875, 1.0],
            [0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 0.05, -0.05, 0.15],
        ];
        for probe in probes {
            let value = assembled.quadratic_form(&probe);
            assert!(
                value >= -1e-5 * scale,
                "the strain Hessian is indefinite: v^T H v = {value:e} against a largest \
                 entry of {scale:e}"
            );
        }
        for axis in 0..3usize {
            let mut translation = [0.0f64; 9];
            for vertex in 0..3usize {
                translation[3 * vertex + axis] = 1.0;
            }
            let value = assembled.quadratic_form(&translation);
            assert!(
                value.abs() <= 1e-5 * scale,
                "a uniform translation along axis {axis} is not in the Hessian's null space: \
                 v^T H v = {value:e} against a largest entry of {scale:e}"
            );
        }
    }

    #[test]
    fn the_shell_strain_line_search_stops_at_the_limit() {
        // THE HALF THAT MAKES THE LIMIT A LIMIT. The barrier raises the cost of
        // stretching; this REFUSES the fraction of the step that would cross it.
        //
        // The sweep runs from the rest triangle to `diag(2, 1)`, so
        // `F(t) = diag(1 + t, 1)` and the strain is exactly `t`. A limit of 0.5
        // therefore has to stop the step at 0.5, and the bisection returns the
        // last `t` STRICTLY below it.
        let mut fixture = ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
        let start = positions_of(&fixture.scene);
        fixture.scale_along_x(2.0);
        let finish = positions_of(&fixture.scene);

        let ShellStrain { scene, state, device } = &mut fixture;
        state.target.seed(device, &start).expect("the fixture seeds the buffer");
        state.eval_x.seed(device, &finish).expect("the fixture seeds the buffer");
        // Safety: the scene is live and the state was allocated for it.
        unsafe { shell_strain_toi(device, &scene.data, state, 1.0) }.expect("the line search runs");
        // The limiter leaves its per-element times on the device, so a test
        // that reads the mirror asks for it.
        state.face_strain.toi.download(device).expect("the times come back");
        let toi = state.face_strain.toi.host()[0];
        assert!(
            (toi - 0.5).abs() < 1e-4,
            "the line search returned {toi} where the limit is crossed at exactly 0.5"
        );
        // AND THE POSE IT ACCEPTS IS INSIDE THE LIMIT, which is the guarantee
        // rather than the arithmetic. Asserting a strict `toi < 0.5` instead
        // would be asserting about fp32's grid: the strain is measured by
        // subtracting one from a number near 1.5, so the last representable step
        // in `t` is coarser than the bisection's own window and the returned
        // fraction can sit an ulp above the analytic crossing while the strain
        // there is still below the limit.
        assert!(
            interpolated_face_strain(toi) <= 0.5,
            "the accepted fraction {toi} puts the face at a strain of {}, past the limit",
            interpolated_face_strain(toi)
        );
    }

    /// The largest strain of `F(t) = diag(1 + t, 1)`, through the shared body.
    ///
    /// That is the deformation gradient the sweep in the two shell line-search
    /// tests interpolates: the rest triangle's `F` is the identity and the
    /// stretched one's is `diag(2, 1)`, both exact, so `f0 + t * df` is this.
    fn interpolated_face_strain(t: f32) -> f32 {
        // Column-major 3x2: column 0 is `(1 + t, 0, 0)`, column 1 is `(0, 1, 0)`.
        let deformation = [1.0 + t, 0.0, 0.0, 0.0, 1.0, 0.0];
        // ONE DEVICE, and the two arrays on it: a handle names no allocator, so
        // staging on a throwaway `host_device()` and dispatching on another
        // would resolve against an arena that never held them.
        let mut device = host_device();
        let mut deformation_d = ppf_cts_compute::Buffer::<f32>::none();
        let mut strain_d = ppf_cts_compute::ReadbackBuffer::<f32>::default();
        deformation_d
            .size(&mut device, deformation.len(), ppf_cts_compute::AllocLabel("test.deformation"))
            .and_then(|()| deformation_d.write(&mut device, 0, &deformation))
            .and_then(|()| strain_d.size(&mut device, 1, ppf_cts_compute::AllocLabel("test.strain")))
            .expect("the fixture stages its arrays");
        let args = ShellMaxStrainArgs {
            deformation: deformation_d.span(0, deformation.len()),
            strain: strain_d.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        // Safety: both handles name live allocations on this device.
        unsafe { device.launch("test.shell_max_strain", &args, 1) }
            .expect("the shell strain reading dispatches");
        strain_d.download(&mut device).expect("the strain reads back");
        strain_d.host()[0]
    }

    #[test]
    fn the_shell_line_search_does_not_bound_a_step_inside_the_limit() {
        // The other side of the same gate, and the one a limiter that clamped
        // unconditionally would fail: a step that ends at a strain of 0.25
        // against a limit of 0.5 is taken whole.
        let mut fixture = ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
        let start = positions_of(&fixture.scene);
        fixture.scale_along_x(1.25);
        let finish = positions_of(&fixture.scene);

        let ShellStrain { scene, state, device } = &mut fixture;
        state.target.seed(device, &start).expect("the fixture seeds the buffer");
        state.eval_x.seed(device, &finish).expect("the fixture seeds the buffer");
        // Safety: as above.
        unsafe { shell_strain_toi(device, &scene.data, state, 1.0) }.expect("the line search runs");
        // The limiter leaves its per-element times on the device, so a test
        // that reads the mirror asks for it.
        state.face_strain.toi.download(device).expect("the times come back");
        assert_eq!(
            state.face_strain.toi.host()[0], 1.0,
            "a step that stays inside the limit must not be truncated"
        );
    }

    #[test]
    fn the_shell_line_search_measures_the_shrink_corrected_limit() {
        // The line search takes `shell_effective_strain_limit`, as the
        // stiffness does and unlike the
        // barrier. With both shrink factors at 0.5 the authored 0.5 corrects to
        // 2.0, and the same sweep to `diag(2, 1)` reaches a strain of 1.0 at
        // `t = 1`, so nothing bounds the step at all.
        let mut fixture = ShellStrain::unit_triangle(limited_material(0.5, 0.5, 0.5), 2.0);
        let start = positions_of(&fixture.scene);
        fixture.scale_along_x(2.0);
        let finish = positions_of(&fixture.scene);

        let ShellStrain { scene, state, device } = &mut fixture;
        state.target.seed(device, &start).expect("the fixture seeds the buffer");
        state.eval_x.seed(device, &finish).expect("the fixture seeds the buffer");
        // Safety: as above.
        unsafe { shell_strain_toi(device, &scene.data, state, 1.0) }.expect("the line search runs");
        // The limiter leaves its per-element times on the device, so a test
        // that reads the mirror asks for it.
        state.face_strain.toi.download(device).expect("the times come back");
        assert_eq!(
            state.face_strain.toi.host()[0], 1.0,
            "the authored limit of 0.5 corrects to 2.0 here, which this sweep never reaches"
        );
    }

    /// One rod segment with the whole strain-limit assembly around it.
    ///
    /// TWO REST LENGTHS, DELIBERATELY DIFFERENT. `EdgeProp::length` is what the
    /// stretch energy measures against and `EdgeProp::initial_length` what the
    /// limiter measures against; a fixture that set them equal could not tell an
    /// exchange of the two apart.
    struct RodStrain {
        scene: TestScene,
        state: SolverState,
        /// ONE DEVICE FOR THE FIXTURE'S LIFE, on the grounds
        /// [`Bending::device`] states: a handle names an arena of the allocator
        /// that opened it, so sizing on one device and dispatching on another
        /// resolves a handle against a table that never held it.
        device: HostDevice,
    }

    impl RodStrain {
        fn segment(limit: f32, initial_length: f32, mass: f32) -> Self {
            let edges = [Vec2u::new(0, 1)];
            let mut scene = TestScene::new(2).with_edges(&edges);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.0, 0.0, 0.0);
            scene.data.rod_count = 1;
            scene.data.surface_vert_count = 2;
            scene.data.param_arrays.edge = CVec::from(
                &[EdgeParam {
                    strainlimit: limit,
                    ..EdgeParam::default()
                }][..],
            );
            let prop = &mut scene.data.prop.edge.as_mut_slice()[0];
            prop.initial_length = initial_length;
            // HALF THE VALUE THE LIMITER READS. Reading this one instead would
            // put the segment at a strain of 1.5 rather than 0.25 in the test
            // below, past the limit and on the far side of the barrier's gap.
            prop.length = 0.5 * initial_length;
            prop.mass = mass;
            install_rod_adjacency(&mut scene.data, 2, &edges);
            install_pattern(&mut scene.data, 2);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            Self { scene, state, device }
        }

        fn stretch_to(&mut self, length: f32) {
            self.scene.place(1, length, 0.0, 0.0);
            // The scene moved, so the device positions must move with it.
            crate::driver::state::reseed_positions(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
        }

        fn assemble(&mut self) -> FatalResult<Assembled> {
            let iterate = positions_of(&self.scene);
            let RodStrain { scene, state, device } = self;
            clear_force(device, state);
            // `positions` IS the start-of-step pose, so it tracks the scene at
            // every fixture entry. A test that wants the iterate somewhere else
            // seeds `eval_x` separately below; one that does not gets both at
            // the scene, which is what a step would see.
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            state.eval_x.seed(device, &iterate).expect("the fixture seeds the buffer");
            // Safety: the scene is live, the state was allocated for it, and both
            // matrices borrow the scene's own pattern tables.
            let (force, dense) = unsafe {
                let mut reference = FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default())?;
                let mut fixed = FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default())?;
                rod_strain(device, &scene.data, CUBIC, state, &mut reference, &mut fixed)?;
                // The dense reference reads the mirror, which the assembly
                // dispatch above invalidated. Download before reading.
                fixed.download(device)?;
                (force_of(device, state), dense_of(&fixed, 2))
            };
            Ok(Assembled {
                force,
                dense,
                width: 6,
            })
        }
    }

    #[test]
    fn a_rod_inside_its_limit_carries_the_barrier_gradient_over_its_rest_length() {
        // THE DERIVATION. The segment sits at 1.25 against an `initial_length`
        // of 1.0, so the strain is 0.25 and the gap against a limit of 0.5 is
        // 0.25. The body's gradient is `-barrier_gradient(gap, limit)` and its
        // Jacobian is `-n / l0` at the first node and `+n / l0` at the second,
        // with `n` the unit segment direction, which here is the x axis. The
        // stiffness is `mass / gap^2` with a zero reference matrix, so it is
        // `2.0 / 0.0625 = 32`.
        //
        // READING `EdgeProp::length` INSTEAD would put the strain at 1.5 and the
        // gap at -1.0, a different branch of the barrier entirely.
        let mut fixture = RodStrain::segment(0.5, 1.0, 2.0);
        fixture.stretch_to(1.25);
        let assembled = fixture.assemble().expect("the assembly runs");

        let gradient = -barrier_gradient(0.25, 0.5);
        let stiffness = 2.0f32 / (0.25f32 * 0.25f32);
        let expected = stiffness * gradient / 1.0;
        assert!(expected.abs() > 1e-3, "the fixture must put a real force on the rod");

        let at_one = assembled.vertex_force(1);
        let at_zero = assembled.vertex_force(0);
        assert!(
            (at_one[0] - expected).abs() <= TOLERANCE * expected.abs(),
            "vertex 1 carries {} against the derived {expected}",
            at_one[0]
        );
        assert!(
            (at_zero[0] + expected).abs() <= TOLERANCE * expected.abs(),
            "vertex 0 carries {} against the derived {}",
            at_zero[0],
            -expected
        );
        for k in 1..3 {
            assert!(at_one[k].abs() <= TOLERANCE * expected.abs());
            assert!(at_zero[k].abs() <= TOLERANCE * expected.abs());
        }
    }

    #[test]
    fn a_rod_at_its_rest_length_carries_no_strain_limit_term() {
        // The body reports `false` for a segment that is not stretched, and a
        // false is a SKIP here rather than a stop: the caller does nothing with
        // the element and assembles no term for it.
        let mut fixture = RodStrain::segment(0.5, 1.0, 2.0);
        let assembled = fixture.assemble().expect("the assembly runs");
        assert_eq!(assembled.vertex_force(0), [0.0, 0.0, 0.0]);
        assert_eq!(assembled.vertex_force(1), [0.0, 0.0, 0.0]);
        assert!(assembled.dense.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn a_rod_with_no_strain_limit_or_a_zero_rest_length_carries_no_term() {
        // Two halves of the rod gate, and the second one is a division the
        // shared body does not guard: `rod_strain_force_hessian` divides the
        // current length by the rest length with no test, so a zero there has to
        // be taken out before the call rather than answered afterwards.
        let mut unlimited = RodStrain::segment(0.0, 1.0, 2.0);
        unlimited.stretch_to(1.25);
        let assembled = unlimited.assemble().expect("the assembly runs");
        assert_eq!(assembled.vertex_force(1), [0.0, 0.0, 0.0]);

        let mut restless = RodStrain::segment(0.5, 1.0, 2.0);
        restless.scene.data.prop.edge.as_mut_slice()[0].initial_length = 0.0;
        restless.stretch_to(1.25);
        let assembled = restless.assemble().expect("the assembly runs");
        assert_eq!(assembled.vertex_force(1), [0.0, 0.0, 0.0]);
        assert!(
            assembled.force.iter().all(|value| value.is_finite()),
            "a zero rest length must be taken out of the range, not divided by"
        );
    }

    #[test]
    fn the_rod_strain_hessian_is_psd_and_annihilates_a_translation() {
        // The rod's Hessian is PSD by structure rather than by projection:
        // `curvature * J J^T` is a clamped rank-1, and the geometric part is
        // `[[1, -1], [-1, 1]]` tensored with `I - n n^T`, a Kronecker product of
        // two PSD matrices, scaled by a clamped gradient. Both terms vanish on a
        // uniform translation.
        let mut fixture = RodStrain::segment(0.5, 1.0, 2.0);
        fixture.stretch_to(1.25);
        let assembled = fixture.assemble().expect("the assembly runs");

        let scale: f64 = assembled
            .dense
            .iter()
            .fold(0.0f64, |best, value| best.max(f64::from(value.abs())));
        assert!(scale > 1e-3, "the fixture must assemble a real Hessian");

        let probes: [[f64; 6]; 4] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
            [0.3, -0.7, 0.2, 0.9, 0.1, -0.4],
            [-1.0, 0.25, 0.5, 0.75, -0.25, 0.125],
        ];
        for probe in probes {
            let value = assembled.quadratic_form(&probe);
            assert!(
                value >= -1e-5 * scale,
                "the rod strain Hessian is indefinite: v^T H v = {value:e}"
            );
        }
        for axis in 0..3usize {
            let mut translation = [0.0f64; 6];
            translation[axis] = 1.0;
            translation[3 + axis] = 1.0;
            let value = assembled.quadratic_form(&translation);
            assert!(
                value.abs() <= 1e-5 * scale,
                "a uniform translation along axis {axis} is not in the null space: {value:e}"
            );
        }
    }

    #[test]
    fn the_rod_line_search_stops_at_the_limit() {
        // The sweep runs from the segment at its rest length to twice it, so the
        // strain is exactly `t` and a limit of 0.05 has to stop the step there.
        let mut fixture = RodStrain::segment(0.05, 1.0, 2.0);
        let start = positions_of(&fixture.scene);
        fixture.stretch_to(2.0);
        let finish = positions_of(&fixture.scene);

        let RodStrain { scene, state, device } = &mut fixture;
        state.target.seed(device, &start).expect("the fixture seeds the buffer");
        state.eval_x.seed(device, &finish).expect("the fixture seeds the buffer");
        // Safety: the scene is live and the state was allocated for it.
        unsafe { rod_strain_toi(device, &scene.data, state, 1.0) }.expect("the line search runs");
        // The limiter leaves its per-element times on the device, so a test
        // that reads the mirror asks for it.
        state.rod_strain.toi.download(device).expect("the times come back");
        let toi = state.rod_strain.toi.host()[0];
        assert!(
            (toi - 0.05).abs() < 1e-4,
            "the line search returned {toi} where the limit is crossed at exactly 0.05"
        );
        // AND THE POSE IT ACCEPTS IS INSIDE THE LIMIT. As in the shell case, the
        // strain is a difference against one, so the fraction can land an ulp
        // above the analytic crossing while the strain the body measures there
        // is still below the limit. Measured here: 0.050000012 against 0.05.
        let strain = interpolated_rod_strain(toi);
        assert!(
            strain <= 0.05,
            "the accepted fraction {toi} puts the segment at a strain of {strain}, past the \
             limit"
        );
    }

    /// The strain of `d(t) = (1 + t, 0, 0)` against a rest length of one,
    /// through the shared body.
    ///
    /// That is what the rod sweep interpolates: the segment starts at its rest
    /// length along x and ends at twice it, both exactly representable.
    fn interpolated_rod_strain(t: f32) -> f32 {
        let difference = [1.0 + t, 0.0, 0.0];
        let rest = [1.0f32];
        // ONE DEVICE, as `interpolated_face_strain` above.
        let mut device = host_device();
        let mut difference_d = ppf_cts_compute::Buffer::<f32>::none();
        let mut rest_d = ppf_cts_compute::Buffer::<f32>::none();
        let mut strain_d = ppf_cts_compute::ReadbackBuffer::<f32>::default();
        difference_d
            .size(&mut device, 3, ppf_cts_compute::AllocLabel("test.difference"))
            .and_then(|()| difference_d.write(&mut device, 0, &difference))
            .and_then(|()| rest_d.size(&mut device, 1, ppf_cts_compute::AllocLabel("test.rest")))
            .and_then(|()| rest_d.write(&mut device, 0, &rest))
            .and_then(|()| strain_d.size(&mut device, 1, ppf_cts_compute::AllocLabel("test.strain")))
            .expect("the fixture stages its arrays");
        let args = RodStrainValueArgs {
            difference: difference_d.span(0, 3),
            rest_length: rest_d.span(0, 1),
            strain: strain_d.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        // Safety: every handle names a live allocation on this device.
        unsafe { device.launch("test.rod_strain_value", &args, 1) }
            .expect("the rod strain reading dispatches");
        strain_d.download(&mut device).expect("the strain reads back");
        strain_d.host()[0]
    }

    #[test]
    fn the_stretch_indicator_reports_the_largest_ratio_and_its_shrink() {
        // `max_sigma` is a RATIO about one, not the strain the barrier is a
        // function of, and it is scaled by the smaller of the face's two shrink
        // factors.
        let mut fixture = ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
        fixture.scene.place(1, 1.25, 0.0, 0.0);
        {
            let ShellStrain { scene, state, device } = &mut fixture;
            // The scene was placed directly, so the committed pose follows it.
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            // Safety: the scene is live and the state was allocated for it.
            unsafe { stretch_indicator(device, &scene.data, state) }.expect("the indicator runs");
            // The indicator leaves both results on the device, so a test
            // that reads the mirror asks for it.
            state.stretch.ratio.download(device).expect("the ratios come back");
            state.stretch.rod_ratio.download(device).expect("the rod ratios come back");
            assert!(
                (state.stretch.ratio.host()[0] - 1.25).abs() < 1e-5,
                "an unshrunk face stretched to 1.25 must report 1.25, got {}",
                state.stretch.ratio.host()[0]
            );
        }

        let mut shrunk = ShellStrain::unit_triangle(limited_material(0.5, 0.5, 1.0), 2.0);
        shrunk.scene.place(1, 1.25, 0.0, 0.0);
        {
            let ShellStrain { scene, state, device } = &mut shrunk;
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            // Safety: as above.
            unsafe { stretch_indicator(device, &scene.data, state) }.expect("the indicator runs");
            // The indicator leaves both results on the device, so a test
            // that reads the mirror asks for it.
            state.stretch.ratio.download(device).expect("the ratios come back");
            state.stretch.rod_ratio.download(device).expect("the rod ratios come back");
            assert!(
                (state.stretch.ratio.host()[0] - 0.625).abs() < 1e-5,
                "a face with a 0.5 shrink factor must report 1.25 * 0.5, got {}",
                state.stretch.ratio.host()[0]
            );
        }
    }

    /// A FACE THE HOST DECLINED TO ADMIT CARRIES NO ELASTIC TERM, whatever its
    /// material says.
    ///
    /// THE GATE IS NOT A SEED VALUE. A `shell_membrane` that seeded `mu` to
    /// zero for every face and wrote the material only for the ones it admitted
    /// would let a `fixed`, `collider` or `rest_excluded` face reach the body
    /// with `mu == 0`, and its `mu > 0` test would skip it; the dispatch covers
    /// EVERY face rather than an active list, so that seed would be the whole
    /// gate. The body reads `FaceParam` off the device instead, so this test has
    /// to be explicit, and a collider's faces do carry stiffness:
    /// `_setup_pin_shell` zeroes the material and a later `param.clear_all()`
    /// discards those writes, so a pin shell arrives with the triangle
    /// DEFAULTS.
    ///
    /// IT TAKES TWO FACES, AND ONE WOULD NOT DO. With one
    /// face, excluding it leaves the active list EMPTY and `shell_membrane`
    /// returns before dispatching anything, so the assertion would hold with the
    /// gate deleted. The second face keeps the dispatch alive; vertex 0 belongs to
    /// the excluded face alone and vertex 3 to the other, so one asserts the
    /// exclusion and the other asserts the pass still ran.
    ///
    /// NOTHING ELSE IN THE TREE CATCHES THIS. The build without the gate passed
    /// 612 unit tests, six static gates and 23 of 23 scenes producing moving
    /// output, and `disp_at_5` on `drape` could not separate it from a correct
    /// build either.
    #[test]
    fn the_face_elastic_embed_excludes_a_collider_a_pinned_and_a_rest_excluded_face() {
        // Two triangles sharing the edge 1-2, stretched along x so both carry a
        // real membrane force. Vertex 0 is exclusive to face 0 and vertex 3 to
        // face 1.
        fn pair(flag: Option<usize>) -> Membrane {
            let mut scene = TestScene::new(4)
                .with_faces(&[Vec3u::new(0, 1, 2), Vec3u::new(1, 3, 2)]);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.25, 0.0, 0.0);
            scene.place(2, 0.0, 1.0, 0.0);
            scene.place(3, 1.25, 1.0, 0.0);
            let stiff = material(Model::Arap, 120.0, 60.0);
            scene.data.param_arrays.face = CVec::from(&[stiff][..]);
            scene.data.inv_rest2x2 =
                CVec::from(&[Mat2x2f::identity(), Mat2x2f::identity()][..]);
            for face in scene.data.prop.face.as_mut_slice() {
                face.mass = 2.0;
            }
            if let Some(case) = flag {
                let face = &mut scene.data.prop.face.as_mut_slice()[0];
                match case {
                    0 => face.collider = true,
                    1 => face.fixed = true,
                    _ => face.rest_excluded = true,
                }
            }
            install_pattern(&mut scene.data, 4);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            Membrane { scene, state, device }
        }

        // THE UNEXCLUDED TWIN FIRST, because a fixture that produces no force
        // would pass every exclusion below while testing nothing.
        let mut twin = pair(None);
        let assembled = twin.assemble_faces(0.01, 4).expect("the twin assembles");
        let reference = assembled.vertex_force(0);
        assert!(
            reference.iter().any(|component| component.abs() > 1.0),
            "the unexcluded twin must carry a real elastic force at vertex 0, \
             or the exclusions below prove nothing: got {reference:?}"
        );

        for case in 0..3usize {
            let mut fixture = pair(Some(case));
            let excluded = fixture
                .assemble_faces(0.01, 4)
                .expect("the excluded face assembles");
            let gated = excluded.vertex_force(0);
            assert!(
                gated.iter().all(|component| *component == 0.0),
                "case {case}: vertex 0 belongs only to the excluded face and \
                 must carry no elastic force, got {gated:?}"
            );
            let live = excluded.vertex_force(3);
            assert!(
                live.iter().any(|component| component.abs() > 1.0),
                "case {case}: vertex 3 belongs to the face that is NOT excluded, \
                 so a zero here means the dispatch never ran and the assertion \
                 above proves nothing: got {live:?}"
            );
        }
    }

    #[test]
    fn the_stretch_indicator_excludes_a_collider_a_pinned_face_and_a_rigid_body() {
        // ITS OWN GATE, which is not the limiter's. A PDRD body's rigid fit
        // always carries a small singular-value residual that would dominate the
        // indicator, and a collider's distance from its rest shape reflects what
        // its pins allowed rather than material stretch.
        for case in 0..3usize {
            let mut fixture =
                ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
            fixture.scene.place(1, 1.25, 0.0, 0.0);
            match case {
                0 => fixture.scene.data.prop.face.as_mut_slice()[0].collider = true,
                1 => fixture.scene.data.prop.face.as_mut_slice()[0].fixed = true,
                _ => fixture.scene.vertex_props_mut()[0].pdrd_body_index = 1,
            }
            let ShellStrain { scene, state, device } = &mut fixture;
            // The scene was placed directly, so the committed pose follows it.
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            // Safety: the scene is live and the state was allocated for it.
            unsafe { stretch_indicator(device, &scene.data, state) }.expect("the indicator runs");
            // The indicator leaves both results on the device, so a test
            // that reads the mirror asks for it.
            state.stretch.ratio.download(device).expect("the ratios come back");
            state.stretch.rod_ratio.download(device).expect("the rod ratios come back");
            assert_eq!(
                state.stretch.ratio.host()[0], 0.0,
                "case {case} must be excluded from the stretch indicator"
            );
        }
    }

    #[test]
    fn the_stretch_indicator_reports_a_rod_against_its_initial_length() {
        // The rod half of the same indicator, and it takes `initial_length`
        // rather than the `length-factor`-scaled `length` the stretch energy
        // measures against.
        let mut fixture = RodStrain::segment(0.05, 1.0, 2.0);
        fixture.stretch_to(1.5);
        let RodStrain { scene, state, device } = &mut fixture;
        // Safety: the scene is live and the state was allocated for it.
        unsafe { stretch_indicator(device, &scene.data, state) }.expect("the indicator runs");
            // The indicator leaves both results on the device, so a test
            // that reads the mirror asks for it.
            state.stretch.ratio.download(device).expect("the ratios come back");
            state.stretch.rod_ratio.download(device).expect("the rod ratios come back");
        assert!(
            (state.stretch.rod_ratio.host()[0] - 1.5).abs() < 1e-5,
            "a segment at 1.5 against a rest length of 1.0 must report 1.5, got {}; reading \
             `length` instead would report 3.0",
            state.stretch.rod_ratio.host()[0]
        );
    }


    /// One column of a central difference of `gradient`, per iterate component.
    ///
    /// `step` is a dyadic world length, so the perturbation is exactly
    /// representable and the two probes are symmetric about the base pose.
    fn finite_difference<F>(base: &[f32], step: f32, mut gradient: F) -> Vec<f64>
    where
        F: FnMut(&[f32]) -> Vec<f64>,
    {
        let width = base.len();
        assert!(step > 0.0, "the step must be a positive world length");
        let mut out = vec![0.0f64; width * width];
        let mut probe = base.to_vec();
        for column in 0..width {
            probe.copy_from_slice(base);
            probe[column] = base[column] + step;
            let forward = gradient(&probe);
            probe[column] = base[column] - step;
            let backward = gradient(&probe);
            for row in 0..width {
                out[width * row + column] =
                    (forward[row] - backward[row]) / (2.0 * f64::from(step));
            }
        }
        out
    }

    /// The largest singular value of the fixture triangle's deformation
    /// gradient, in `f64`, from the pose and from nothing else.
    ///
    /// The rest shape is the unit right triangle and `inv_rest2x2` is the
    /// identity, so `F` is the pair of deformed edge vectors and its singular
    /// values are the square roots of the eigenvalues of the 2x2 Gram matrix.
    /// Computed here rather than read out of the SVD body so the stiffness the
    /// test divides by is derived independently of the chain under test.
    fn largest_singular_value(pose: &[f32]) -> f64 {
        let component = |vertex: usize, axis: usize| {
            f64::from(pose[3 * vertex + axis]) - f64::from(pose[axis])
        };
        let mut gram = [0.0f64; 3];
        for axis in 0..3usize {
            let first = component(1, axis);
            let second = component(2, axis);
            gram[0] += first * first;
            gram[1] += first * second;
            gram[2] += second * second;
        }
        let mean = 0.5 * (gram[0] + gram[2]);
        let half = 0.5 * (gram[0] - gram[2]);
        let radius = (half * half + gram[1] * gram[1]).sqrt();
        (mean + radius).sqrt()
    }

    /// The fixture rod's strain, in `f64`, from the pose.
    fn rod_strain_of(pose: &[f32], rest_length: f64) -> f64 {
        let mut squared = 0.0f64;
        for axis in 0..3usize {
            let difference = f64::from(pose[3 + axis]) - f64::from(pose[axis]);
            squared += difference * difference;
        }
        squared.sqrt() / rest_length - 1.0
    }

    #[test]
    fn the_strain_hessian_is_the_derivative_of_the_barrier_it_scales() {
        // WHAT THIS PINS THAT PSD CANNOT. The spectral Hessian's six modal
        // eigenvalues include `gradient_sigma[i] / sigma[i]` and
        // `(gradient_sigma[0] + gradient_sigma[1]) / (sigma[0] + sigma[1])`,
        // which read the RESTORED singular values. Feeding them the shifted ones
        // instead changes those by factors of five and nine here and leaves the
        // result symmetric, PSD and translation-annihilating, so only a value
        // comparison sees it.
        //
        // THE STIFFNESS IS DIVIDED OUT ON BOTH SIDES, and that is the property
        // being stated rather than a convenience. The dynamic stiffness is a
        // SCALE the assembly applies to an already-formed force and Hessian, not
        // a factor of the energy being differentiated: it varies with the pose
        // through its own `limit - strain` term, so a raw difference of the
        // assembled force carries `d(stiffness)/dx` as well and is larger than
        // the assembled Hessian by that much. Measured here: 145.9 against
        // 81.8 on the leading entry.
        //
        // NONE OF THE SIX MODES IS CLAMPED at this configuration (two are
        // exactly zero and that is their true value), so the projected Hessian
        // IS the second derivative and the comparison is exact rather than
        // one-sided.
        let mut fixture = ShellStrain::unit_triangle(limited_material(0.5, 1.0, 1.0), 2.0);
        fixture.scale_along_x(1.25);
        let base = positions_of(&fixture.scene);
        let assembled = fixture
            .assemble_at(&base, 0.0)
            .expect("the assembly runs");

        // `mass / (limit - strain)^2`, which is the whole stiffness when the
        // reference matrix is zero.
        let stiffness_at = |pose: &[f32]| -> f64 {
            let gap = 0.5 - (largest_singular_value(pose) - 1.0);
            2.0 / (gap * gap)
        };
        let base_stiffness = stiffness_at(&base);

        const STEP: f32 = 1.0 / 256.0;
        let numeric = finite_difference(&base, STEP, |probe| {
            let scale = stiffness_at(probe);
            fixture
                .assemble_at(probe, 0.0)
                .expect("the assembly runs")
                .force
                .iter()
                .map(|value| f64::from(*value) / scale)
                .collect()
        });

        let scale = assembled
            .dense
            .iter()
            .fold(0.0f64, |best, value| best.max(f64::from(value.abs())));
        assert!(scale > 1e-3, "the fixture must assemble a real Hessian");
        for row in 0..9usize {
            for column in 0..9usize {
                let analytic = f64::from(assembled.dense[9 * row + column]);
                let measured = base_stiffness * numeric[9 * row + column];
                assert!(
                    (analytic - measured).abs() <= 2e-2 * scale,
                    "entry ({row}, {column}) is {analytic} against a central difference of \
                     {measured}, over a largest entry of {scale}"
                );
            }
        }
    }

    #[test]
    fn the_rod_strain_hessian_is_the_derivative_of_the_barrier_it_scales() {
        // The rod's counterpart, with the same stiffness division and the same
        // reason for it. Its Hessian is the barrier's curvature times the strain
        // Jacobian's outer product plus the geometric term the strain's own
        // second derivative contributes, both clamped at zero and neither
        // clamped here, so it is the exact second derivative.
        let mut fixture = RodStrain::segment(0.5, 1.0, 2.0);
        fixture.stretch_to(1.25);
        let base = positions_of(&fixture.scene);
        let assembled = fixture.assemble().expect("the assembly runs");

        let stiffness_at = |pose: &[f32]| -> f64 {
            let gap = 0.5 - rod_strain_of(pose, 1.0);
            2.0 / (gap * gap)
        };
        let base_stiffness = stiffness_at(&base);

        const STEP: f32 = 1.0 / 256.0;
        let numeric = finite_difference(&base, STEP, |probe| {
            let scale = stiffness_at(probe);
            let RodStrain { scene, state, device } = &mut fixture;
            clear_force(device, state);
            // `positions` IS the start-of-step pose, so it tracks the scene at
            // every fixture entry. A test that wants the iterate somewhere else
            // seeds `eval_x` separately below; one that does not gets both at
            // the scene, which is what a step would see.
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            // A fixture may have REWRITTEN the pattern since `allocate` staged
            // it, and the two must not disagree.
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            state.eval_x.seed(device, probe).expect("the fixture seeds the buffer");
            // Safety: the scene is live and the state was allocated for it.
            unsafe {
                let mut reference =
                    FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default()).expect("wraps");
                let mut fixed =
                    FixedCsr::adopt_from_dataset(device, state.fixed_pattern_refs(), &scene.data, Default::default()).expect("wraps");
                rod_strain(device, &scene.data, CUBIC, state, &mut reference, &mut fixed)
                    .expect("the assembly runs");
            }
            fixture
                .state
                .force
                .download(&mut fixture.device)
                .expect("the force mirror refreshes");
            fixture
                .state
                .force
                .host()
                .iter()
                .map(|value| f64::from(*value) / scale)
                .collect()
        });

        let scale = assembled
            .dense
            .iter()
            .fold(0.0f64, |best, value| best.max(f64::from(value.abs())));
        assert!(scale > 1e-3, "the fixture must assemble a real Hessian");
        for row in 0..6usize {
            for column in 0..6usize {
                let analytic = f64::from(assembled.dense[6 * row + column]);
                let measured = base_stiffness * numeric[6 * row + column];
                assert!(
                    (analytic - measured).abs() <= 2e-2 * scale,
                    "entry ({row}, {column}) is {analytic} against a central difference of \
                     {measured}, over a largest entry of {scale}"
                );
            }
        }
    }
}
