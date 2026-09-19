// File: crates/ppf-cts-solver/src/driver/plasticity.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Permanent deformation: the creep of the REST shape.
//!
//! Four passes, one per element class, and each is `cpp/plasticity/
//! plasticity.cu`'s own dispatch: shell faces creep `inv_rest2x2`, tets creep
//! `inv_rest3x3`, hinges creep `HingeProp::rest_angle`, and interior rod
//! vertices creep `VertexProp::rest_bend_angle`. Plasticity adds no field to
//! the scene. It changes four that the elastic layers re-read every Newton
//! iteration, which is why a checkpoint has to carry them: see
//! `crates/ppf-cts-solver/src/plastic_state.rs`, whose per-frame file is what
//! pairs a saved pose with the rest shape it was solved against.
//!
//! ON THIS BACKEND THE CHECKPOINT NEEDS NO COPY BACK. Decision D1 is direct
//! addressing, so the four arrays this module writes ARE the host's own, and
//! `backend.rs` serializes them straight out of the `DataSet`. That is what
//! `fetch_inv_rest()` and `fetch_rest_angles()` say in their comments, and it
//! is only true while every write below lands in the scene's buffers rather
//! than in a mirror.
//!
//! THEY RUN ONCE PER STEP, AFTER THE COMMIT, on the committed pose, which is
//! where `main.cu:1663-1675` puts them. The `dt` they take is the RAW substep
//! (`ParamSet::dt`), not the playback-scaled or TOI-shortened `dt` the Newton
//! loop ran on: `main.cu` hands the plasticity kernels the untouched `ParamSet`
//! while passing its local `dt` to everything else. The creep rate depends on
//! it, so the distinction is reproduced rather than tidied.
//!
//! # The staged shape, and where the gates live
//!
//! Every pass runs its stages over the WHOLE element range and scatters over an
//! ACTIVE LIST, which is this backend's shape for every element layer. An
//! inactive element is still evaluated and its result is then never read, so no
//! stage needs a branch and each call is one contiguous span.
//!
//! Every one of those stages is reached by dispatching a named kernel through
//! [`Device`], and this module names no backend entry point directly. The
//! scatters that follow a pass are the module's own and stay on the host: each
//! copies out of a scratch into the scene's rest shape, over the active list,
//! in ascending element order.
//!
//! Two gates decide the active list, and both are the reference kernel's own:
//! a material with `plasticity <= 0`, and, for a hinge, `HingeProp::fixed`.
//! Neither is a size or a container test, so both are applied here rather than
//! inside a shared body.
//!
//! A THIRD GATE IS THE `changed` VERDICT, and it is not the same kind of thing.
//! The reference kernel returns before writing when no singular value crossed
//! its threshold and when a rest angle is inside its dead zone, so an element
//! that did not yield must keep the exact bytes it had. The two rest-matrix
//! ranges therefore write to a scratch and this module copies only where the
//! shared body reported a yield; recomputing an unyielded element's rest matrix
//! from its own SVD would return a value differing in the last bits and would
//! make the rest shape drift on a scene that is not deforming at all.
//!
//! ONLY THOSE TWO READ THE VERDICT BACK. `plasticity.cu` keeps `changed` in a
//! register and never materializes it, so every readback of it here is a port
//! artifact rather than something the reference does. The two ANGLE ranges need
//! none: their creep is an in-place update of the value the host seeded, so an
//! element inside its dead zone still holds the exact bytes it started with and
//! the scatter is a no-op on it. The two REST-MATRIX ranges still need it,
//! because their kernel writes a fresh matrix for every element in range and the
//! host cannot otherwise tell a yield from a recomputation. Removing that last
//! pair is a residency change rather than a driver one: the reference creeps
//! `inv_rest2x2` and `inv_rest3x3` in place on the device, and this backend
//! cannot until those two arrays are kernel-written buffers.

use crate::data::{
    DataSet, EdgeParam, EdgeProp, FaceParam, FaceProp, HingeParam, HingeProp, TetParam, TetProp,
};
use crate::plastic_state::PlasticKinds;

use ppf_cts_compute::{Buffer, Device, ReadbackBuffer};
use super::kernels::{
    PlasticityCommitFaceArgs, PlasticityCommitTetArgs,
    FaceDeformationGradientArgs, PlasticityAlphaArgs, PlasticityFaceFromRecordsArgs, PlasticityTetFromRecordsArgs, PlasticityRodFromRecordsArgs, PlasticityHingeFromRecordsArgs, PlasticityFaceInverseRestArgs,
    PlasticityCreepRestAngleArgs, PlasticityCreepSingular2Args,
    PlasticityCreepSingular3Args,
    PlasticityTetInverseRestArgs, RodBendAngleArgs, ShellBendAngleArgs, ShellBendRemapArgs,
    Svd3x2Args, Svd3x3RvArgs, TetDeformationGradientArgs,
};
use super::scene::{self, Fatal, FatalResult};
use super::state::SolverState;

/// The per-element scratch the four passes stage through.
///
/// Each group is sized only when its kernel is enabled, so a scene with no
/// plastic material allocates nothing at all and a hinge-only one does not
/// carry the tet rest matrices. That is the same pair of conditions
/// `PlasticKinds` applies, and it is decided once from the built scene because
/// both halves of it are build-time constants.
#[derive(Default)]
pub struct PlasticScratch {
    /// Which of the four kernels this scene runs.
    pub kinds: PlasticKinds,

    /// Per shell face: the two material constants, the creep rate, the
    /// deformation gradient and its SVD, the crept singular values, the yield
    /// verdict and the rest matrix the yield implies.
    ///
    /// THE PLASTICITY RATE AND THE THRESHOLD ARE BOTH STAGED, because both are
    /// built on the host out of the scene's material table and read only by the
    /// creep passes. The creep STEP FRACTION beside them is neither:
    /// `plasticity_alpha` computes it on the device from the rate and the
    /// step, so it is a device allocation like the rest of the chain.
    ///
    /// THE YIELD VERDICT IS A READBACK, the same direction and the same cadence
    /// as `face_inverse_rest` beside it: the creep pass writes one word per
    /// element saying which way the branch went, and the scatter below the
    /// dispatch reads it to decide which rest matrices to copy out. It is a
    /// function of the singular values the kernel computed, so the host cannot
    /// derive it at scene build.
    /// KERNEL-WRITTEN NOW, by `plasticity_face_from_records`, so the type
    /// moves with the writer: `host()` refuses until a download rather than
    /// handing back whatever the host last staged.
    pub face_plasticity: ReadbackBuffer<f32>,
    pub face_threshold: ReadbackBuffer<f32>,
    pub face_alpha: Buffer<f32>,
    /// DEVICE ALLOCATIONS, in the same components as the elastic layer's.
    pub face_deformation: Buffer<f32>,
    /// DEVICE ALLOCATIONS, in the same component as the elastic layer's pair.
    pub face_svd_u: Buffer<f32>,
    pub face_svd_sigma: Buffer<f32>,
    pub face_svd_vt: Buffer<f32>,
    pub face_sigma_new: Buffer<f32>,
    pub face_changed: ReadbackBuffer<u32>,
    /// A READBACK, and the one charged per FRAME rather than per Newton
    /// iteration: the creep runs once a step, and the scatter below the
    /// dispatch reads the rows of the faces that yielded.
    pub face_inverse_rest: ReadbackBuffer<f32>,
    pub face_active: Vec<u32>,

    /// Per tet, the same chain at 3x3. `tet_plasticity` is staged for the
    /// reason given on `face_plasticity`.
    /// Kernel-written, as the face pair above.
    pub tet_plasticity: ReadbackBuffer<f32>,
    pub tet_threshold: ReadbackBuffer<f32>,
    pub tet_alpha: Buffer<f32>,
    pub tet_deformation: Buffer<f32>,
    /// DEVICE ALLOCATIONS, and they move with the elastic layer's trio rather
    /// than on their own: `Svd3x3RvArgs.u`, `.sigma` and `.vt` are filled here
    /// and in `assemble`, and a record field is one type in the Rust twin.
    pub tet_svd_u: Buffer<f32>,
    pub tet_svd_sigma: Buffer<f32>,
    pub tet_svd_vt: Buffer<f32>,
    pub tet_sigma_new: Buffer<f32>,
    pub tet_changed: ReadbackBuffer<u32>,
    /// A READBACK, as [`PlasticScratch::face_inverse_rest`] is.
    pub tet_inverse_rest: ReadbackBuffer<f32>,
    pub tet_active: Vec<u32>,

    /// Per hinge. `node` is the `(2, 1, 0, 3)` permutation the dihedral math
    /// reads its positions in, built once because the mesh topology is fixed at
    /// scene build.
    pub hinge_node: Buffer<u32>,
    /// Staged, for the reason given on `face_plasticity`.
    /// Kernel-written, as the other three pairs.
    pub hinge_plasticity: ReadbackBuffer<f32>,
    pub hinge_threshold: ReadbackBuffer<f32>,
    pub hinge_alpha: Buffer<f32>,
    pub hinge_angle: Buffer<f32>,
    /// The host scratch the seed below is built in.
    ///
    /// A FIELD RATHER THAN A LOCAL, because this runs every step and a local
    /// would allocate each time. It is host-only: nothing dispatches against
    /// it, it is the staging the seed copies from.
    pub hinge_rest_scratch: Vec<f32>,
    /// The rest angle the creep pass rewrites.
    ///
    /// HOST-SEEDED, KERNEL-WRITTEN, HOST-READ, which is the shape
    /// `ReadbackBuffer::seed` exists for: the gather below fills both halves
    /// from `props`, the creep kernel writes the device half, and the scatter
    /// reads the mirror back into `props`. A read that has missed the download
    /// panics naming this buffer rather than scattering the pre-creep angle.
    pub hinge_rest_angle: ReadbackBuffer<f32>,
    pub hinge_changed: ReadbackBuffer<u32>,
    pub hinge_active: Vec<u32>,

    /// Per interior rod vertex, indexed by BENDING SITE and not by vertex. The
    /// sites are the ones `SolverState` enumerated at `initialize()`, which
    /// apply the same two-edges-and-no-face test the reference kernel does.
    /// Staged, for the reason given on `face_plasticity`.
    /// Kernel-written, as the face and tet pairs above.
    pub rod_plasticity: ReadbackBuffer<f32>,
    pub rod_threshold: ReadbackBuffer<f32>,
    pub rod_alpha: Buffer<f32>,
    pub rod_angle: Buffer<f32>,
    /// As [`Self::hinge_rest_scratch`], for the rod pass.
    pub rod_rest_scratch: Vec<f32>,
    /// The rod rest angle the creep pass rewrites. As
    /// [`Self::hinge_rest_angle`].
    pub rod_rest_angle: ReadbackBuffer<f32>,
    pub rod_changed: ReadbackBuffer<u32>,
    pub rod_active: Vec<u32>,
}

impl PlasticScratch {
    /// True when some kernel can creep the rest shape, so the driver has a pass
    /// to run at all.
    pub fn any(&self) -> bool {
        self.kinds.any()
    }
}

/// Run every enabled creep on the committed pose.
///
/// # Safety
/// `data` must address a live `DataSet` whose committed positions are in
/// `vertex.curr`, and `state` must have been allocated for that scene.
pub unsafe fn creep<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
    dt: f32,
) -> FatalResult<()> {
    if !state.plastic.any() {
        return Ok(());
    }
    // The COMMITTED pose. `main.cu` runs these after `vertex.curr` has taken
    // the accepted iterate, so the rest shape follows the pose the frame
    // reports rather than the last Newton iterate before the line search.
    // The COMMITTED pose, which is a device buffer now. The creep runs after
    // the commit, so this is `state.positions` rather than the iterate.
    let pose = state.positions.handle();
    if state.plastic.kinds.face {
        face(device, data, state, pose, dt)?;
    }
    if state.plastic.kinds.tet {
        tet(device, data, state, pose, dt)?;
    }
    if state.plastic.kinds.hinge {
        hinge(device, data, state, pose, dt)?;
    }
    if state.plastic.kinds.rod_bend {
        rod_bend(device, data, state, pose, dt)?;
    }
    Ok(())
}

/// Shell faces: creep `inv_rest2x2` over the SHELL PREFIX.
///
/// The prefix, not the face array: a solid's surface triangles follow the shell
/// faces in `mesh.face` and have no entry in `inv_rest2x2` at all, so a pass
/// ranged over `face.size` would read and write past the end of the rest
/// matrices on every tetrahedralized scene and be correct on every shell-only
/// one.
///
/// # Safety
/// As [`creep`].
unsafe fn face<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
    pose: ppf_cts_compute::Handle,
    dt: f32,
) -> FatalResult<()> {
    let faces = state.sizes.shell_faces;
    if faces == 0 {
        return Ok(());
    }
    let props: &[FaceProp] = scene::slice(&data.prop.face);
    let params: &[FaceParam] = scene::slice(&data.param_arrays.face);
    let mesh: &[crate::data::Vec3u] = scene::slice(&data.mesh.mesh.face);
    let inv_rest: &mut [crate::data::Mat2x2f] = scene::slice_mut(&data.inv_rest2x2);
    if props.len() < faces || mesh.len() < faces || inv_rest.len() < faces {
        return Err(Fatal::invariant(format!(
            "solver driver: the plastic creep walks {faces} shell faces and the scene carries {} \
             face props, {} face index records and {} inverse rest matrices",
            props.len(),
            mesh.len(),
            inv_rest.len()
        )));
    }

    // EARLY OUT ON THE MATERIALS, WHICH IS O(materials) RATHER THAN O(faces).
    // Whether any face can creep is a property of the scene's materials and
    // does not change per step, so the question is answered over the material
    // table rather than by walking every face. The walk this replaces ran on
    // EVERY scene with faces, because the list it built was only consulted
    // afterwards.
    if !params.iter().any(|material| material.plasticity > 0.0) {
        return Ok(());
    }

    let p = &mut state.plastic;
    let count = faces as u32;
    // THE MATERIAL, GATHERED ON THE DEVICE, which is where `plasticity.cu:34`
    // reads it: `param_face[prop_face[i].param_index]` inside the dispatch.
    // The two uploads this replaces are gone with the host gather that filled
    // them.
    let material_args = PlasticityFaceFromRecordsArgs {
        prop: state.prop_face.handle(),
        face_param: state.param_face.handle(),
        plasticity: p.face_plasticity.handle(),
        threshold: p.face_threshold.handle(),
        count,
        seam_arena_count: 0,
    };
    // Safety: every array is borrowed for the whole call.
    unsafe { device.launch("plasticity.face.material", &material_args, count) }?;

    let alpha_args = PlasticityAlphaArgs {
        plasticity: p.face_plasticity.handle(),
        dt,
        alpha: p.face_alpha.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.face.alpha", &alpha_args, count)?;
    let deformation_args = FaceDeformationGradientArgs {
        x: pose,
        face: state.mesh_face.handle(),
        // The bound the entry point checks the face's three slots against, as
        // the inverse-rest pass below states.
        vertex_count: state.sizes.vertices as u32,
        inverse_rest: state.inv_rest2x2.handle(),
        deformation: p.face_deformation.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.face.deformation", &deformation_args, count)?;
    let svd_args = Svd3x2Args {
        input: p.face_deformation.handle(),
        u: p.face_svd_u.handle(),
        sigma: p.face_svd_sigma.handle(),
        vt: p.face_svd_vt.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.face.svd3x2", &svd_args, count)?;
    let singular_args = PlasticityCreepSingular2Args {
        singular_values: p.face_svd_sigma.handle(),
        threshold: p.face_threshold.handle(),
        alpha: p.face_alpha.handle(),
        updated: p.face_sigma_new.handle(),
        changed: p.face_changed.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.face.singular", &singular_args, count)?;
    let rest_args = PlasticityFaceInverseRestArgs {
        x: pose,
        face: state.mesh_face.handle(),
        // The size of the index space the face's three slots may name, which
        // the entry point checks each of them against before it reads a
        // position. Metal returns 0.0 for an out-of-bounds read rather than
        // faulting, so a corrupt index list is a plausible rest shape without
        // this.
        vertex_count: state.sizes.vertices as u32,
        u: p.face_svd_u.handle(),
        singular: p.face_sigma_new.handle(),
        vt: p.face_svd_vt.handle(),
        inverse_rest: p.face_inverse_rest.handle(),
        count: count as u32,
        seam_arena_count: 0,
    };
    device.launch("plasticity.face.inverse_rest", &rest_args, count)?;
    // THE COMMIT, ON THE DEVICE. Only a face that is active AND yielded is
    // written; every other one keeps the bytes it had. Downloading the verdict
    // and the crept rows, walking the active list serially and uploading the
    // destination would move the commit itself off the device.
    let commit = PlasticityCommitFaceArgs {
        plasticity: p.face_plasticity.handle(),
        changed: p.face_changed.handle(),
        inverse_rest: p.face_inverse_rest.handle(),
        destination: state.inv_rest2x2.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.face.commit", &commit, count)?;
    Ok(())
}

/// Tets: creep `inv_rest3x3`.
///
/// # Safety
/// As [`creep`].
unsafe fn tet<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
    pose: ppf_cts_compute::Handle,
    dt: f32,
) -> FatalResult<()> {
    let tets = state.sizes.tets;
    if tets == 0 {
        return Ok(());
    }
    let props: &[TetProp] = scene::slice(&data.prop.tet);
    let params: &[TetParam] = scene::slice(&data.param_arrays.tet);
    let mesh: &[crate::data::Vec4u] = scene::slice(&data.mesh.mesh.tet);
    let inv_rest: &mut [crate::data::Mat3x3f] = scene::slice_mut(&data.inv_rest3x3);
    if props.len() < tets || mesh.len() < tets || inv_rest.len() < tets {
        return Err(Fatal::invariant(format!(
            "solver driver: the plastic creep walks {tets} tets and the scene carries {} tet \
             props, {} tet index records and {} inverse rest matrices",
            props.len(),
            mesh.len(),
            inv_rest.len()
        )));
    }

    // AS THE FACE LAYER ABOVE: the early out is over the MATERIALS, which is
    // O(materials) and scene-static, and the gather is a dispatch.
    if !params.iter().any(|material| material.plasticity > 0.0) {
        return Ok(());
    }

    let p = &mut state.plastic;
    let count = tets as u32;
    let material_args = PlasticityTetFromRecordsArgs {
        prop: state.prop_tet.handle(),
        tet_param: state.param_tet.handle(),
        plasticity: p.tet_plasticity.handle(),
        threshold: p.tet_threshold.handle(),
        count,
        seam_arena_count: 0,
    };
    // Safety: every array is borrowed for the whole call.
    unsafe { device.launch("plasticity.tet.material", &material_args, count) }?;

    let alpha_args = PlasticityAlphaArgs {
        plasticity: p.tet_plasticity.handle(),
        dt,
        alpha: p.tet_alpha.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.tet.alpha", &alpha_args, count)?;
    let deformation_args = TetDeformationGradientArgs {
        x: pose,
        tet: state.mesh_tet.handle(),
        vertex_count: state.sizes.vertices as u32,
        inverse_rest: state.inv_rest3x3.handle(),
        deformation: p.tet_deformation.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.tet.deformation", &deformation_args, count)?;
    // `svd3x3_rv`, the reflection-corrected factorization, which is what
    // `update_tet_plasticity` takes and is not the plain `svd3x3` the elastic
    // layer uses.
    let svd_args = Svd3x3RvArgs {
        input: p.tet_deformation.handle(),
        u: p.tet_svd_u.handle(),
        sigma: p.tet_svd_sigma.handle(),
        vt: p.tet_svd_vt.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.tet.svd3x3_rv", &svd_args, count)?;
    let singular_args = PlasticityCreepSingular3Args {
        singular_values: p.tet_svd_sigma.handle(),
        threshold: p.tet_threshold.handle(),
        alpha: p.tet_alpha.handle(),
        updated: p.tet_sigma_new.handle(),
        changed: p.tet_changed.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.tet.singular", &singular_args, count)?;
    let rest_args = PlasticityTetInverseRestArgs {
        x: pose,
        tet: state.mesh_tet.handle(),
        // The bound the entry point checks the tet's four slots against; see
        // the face pass above.
        vertex_count: state.sizes.vertices as u32,
        u: p.tet_svd_u.handle(),
        singular: p.tet_sigma_new.handle(),
        vt: p.tet_svd_vt.handle(),
        inverse_rest: p.tet_inverse_rest.handle(),
        count: count as u32,
        seam_arena_count: 0,
    };
    device.launch("plasticity.tet.inverse_rest", &rest_args, count)?;
    // THE COMMIT, ON THE DEVICE; see the face pass above.
    let commit = PlasticityCommitTetArgs {
        plasticity: p.tet_plasticity.handle(),
        changed: p.tet_changed.handle(),
        inverse_rest: p.tet_inverse_rest.handle(),
        destination: state.inv_rest3x3.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.tet.commit", &commit, count)?;
    Ok(())
}

/// Shell hinges: creep `HingeProp::rest_angle`.
///
/// DISPATCHED OVER EVERY HINGE IN THE MESH, a solid's surface hinges included,
/// which is why this is reachable on a scene carrying no shell face at all. The
/// bending ENERGY additionally skips a collider hinge and one whose type byte
/// marks it as a solid's, and this does not: the reference kernel's only tests
/// are `fixed` and the material.
///
/// # Safety
/// As [`creep`].
unsafe fn hinge<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
    pose: ppf_cts_compute::Handle,
    dt: f32,
) -> FatalResult<()> {
    let hinges = state.sizes.hinges;
    if hinges == 0 {
        return Ok(());
    }
    let props: &mut [HingeProp] = scene::slice_mut(&data.prop.hinge);
    let params: &[HingeParam] = scene::slice(&data.param_arrays.hinge);
    if props.len() < hinges {
        return Err(Fatal::invariant(format!(
            "solver driver: the plastic creep walks {hinges} hinges and the scene carries {} hinge \
             props",
            props.len()
        )));
    }

    // AS THE FACE, TET AND ROD LAYERS: the early out is over the MATERIALS,
    // which is scene-static and O(materials), and the gather is a dispatch.
    if !params.iter().any(|material| material.plasticity > 0.0) {
        return Ok(());
    }

    let p = &mut state.plastic;
    let count = hinges as u32;
    let material_args = PlasticityHingeFromRecordsArgs {
        prop: state.prop_hinge.handle(),
        hinge_param: state.param_hinge.handle(),
        plasticity: p.hinge_plasticity.handle(),
        threshold: p.hinge_threshold.handle(),
        count,
        seam_arena_count: 0,
    };
    // Safety: every array is borrowed for the whole call.
    unsafe { device.launch("plasticity.hinge.material", &material_args, count) }?;

    // THE REST ANGLE IS SEEDED FROM THE PROPS, and it has to be: the creep is
    // an IN-PLACE update that leaves a hinge inside its dead zone untouched,
    // so the slot must already hold that hinge's own angle or the write-back
    // would carry a zero into a hinge that simply did not yield. Removing this
    // with the host gather is what broke `a_fixed_hinge_does_not_creep`.
    for index in 0..hinges {
        p.hinge_rest_scratch[index] = props[index].rest_angle;
    }
    p.hinge_rest_angle
        .seed(device, &p.hinge_rest_scratch[..hinges])?;
    let alpha_args = PlasticityAlphaArgs {
        plasticity: p.hinge_plasticity.handle(),
        dt,
        alpha: p.hinge_alpha.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.hinge.alpha", &alpha_args, count)?;
    let angle_args = ShellBendAngleArgs {
        x: pose,
        hinge: p.hinge_node.handle(),
        // The bound the entry point checks the hinge's four slots against,
        // which the record carries because a slot is data rather than the
        // thread index.
        vertex_count: state.sizes.vertices as u32,
        angle: p.hinge_angle.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.hinge.angle", &angle_args, count)?;
    // IN PLACE ON THE GATHERED COPY. The shared body leaves a rest angle inside
    // its dead zone untouched, so an inactive hinge's slot still holds the value
    // gathered above and the scatter below is a no-op on it. The scatter still
    // walks the active list, so an inactive hinge is never written at all.
    let rest_args = PlasticityCreepRestAngleArgs {
        angle: p.hinge_angle.handle(),
        threshold: p.hinge_threshold.handle(),
        alpha: p.hinge_alpha.handle(),
        rest_angle: p.hinge_rest_angle.handle(),
        changed: p.hinge_changed.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.hinge.rest_angle", &rest_args, count)?;
    // THE CREPT ANGLES COME BACK, AND THE VERDICT DOES NOT. The reference keeps
    // `changed` in a register and writes `prop.rest_angle` in place on the
    // device (`plasticity.cu:182-185`), so it reads no verdict array at all;
    // this pass reads the angles back only because `HingeProp` is still
    // host-resident. The verdict would be a SECOND per-element array crossing
    // the seam to decide a branch the device has already taken: the creep is an
    // in-place update, so a hinge inside its dead zone still holds the exact
    // value seeded from `props` above and the write below is a no-op on it.
    p.hinge_rest_angle.download(device)?;

    // THE ONE COMMIT STILL ON THE HOST, and the reason is a data layout rather
    // than an omission. The face and tet commits are kernels because their
    // destination is a FLAT array; this one writes `HingeProp::rest_angle`, a
    // MEMBER of another array's element, and a record addresses one allocation
    // per field, so a dispatch into a member cannot be written without
    // splitting that member out first, which is a data-layout change rather
    // than a mechanical one.
    // OVER EVERY HINGE, because there is no compacted list any more. The
    // kernel leaves an unyielded angle UNTOUCHED and the mirror was seeded
    // from these same props, so a hinge that did not creep writes back the
    // value it already held.
    let mut crept = false;
    for index in 0..hinges {
        props[index].rest_angle = p.hinge_rest_angle.host()[index];
        crept = true;
    }
    // THE STAGED MIRROR FOLLOWS THE SCENE, exactly as `prop_vertex` does above.
    // The bending stiffness reads `HingeProp` off the device now, so a creep
    // that moved the scene's copy and left the device's behind would assemble
    // against the previous step's hinges. It cannot go unnoticed, `at()` marks
    // the mirror stale and `handle()` refuses while it is, but it would be a
    // panic mid-solve rather than a correct step, so it is re-staged here where
    // the write happens.
    //
    // NONE OF THE SIX FIELDS THE STIFFNESS READS ACTUALLY CREEPS, only
    // `rest_angle` does, so this is owed to the NEXT reader rather than to that
    // kernel. That is the reason to do it at the write rather than at the read.
    if crept {
        let hinges = state.prop_hinge.len().min(props.len());
        if hinges > 0 {
            state.prop_hinge.at()[..hinges].copy_from_slice(&props[..hinges]);
            state.prop_hinge.upload(device)?;
        }
    }
    Ok(())
}

/// Interior rod vertices: creep `VertexProp::rest_bend_angle`.
///
/// The material is the AVERAGE of the two incident edges' materials, both the
/// creep rate and the dead zone, which is `update_rod_bend_plasticity`'s own
/// two-segment average and the same one the bending stiffness takes.
///
/// # Safety
/// As [`creep`].
unsafe fn rod_bend<D: Device>(
    device: &mut D,
    data: &DataSet,
    state: &mut SolverState,
    pose: ppf_cts_compute::Handle,
    dt: f32,
) -> FatalResult<()> {
    let sites = state.sizes.rod_bend_sites;
    if sites == 0 {
        return Ok(());
    }
    // Read before the destructuring below, which reborrows the two scratch
    // fields for the rest of the function.
    let vertices = state.sizes.vertices;
    // THE STAGED COPY, NOT THE `DataSet` ONE. The kernels read
    // `state.prop_vertex`, and the host `DataSet` array is refreshed FROM it by
    // `fetch_rest_angles()`, never back into it, so a creep written to the
    // `DataSet` would be silently discarded.
    let _edge_props: &[EdgeProp] = scene::slice(&data.prop.edge);
    let edge_params: &[EdgeParam] = scene::slice(&data.param_arrays.edge);
    if state.prop_vertex.len() < state.sizes.vertices {
        return Err(Fatal::invariant(format!(
            "solver driver: the plastic creep reads {} vertex props and the scene has {} vertices",
            state.prop_vertex.len(),
            state.sizes.vertices
        )));
    }

    let SolverState {
        plastic: p,
        rod_bend: b,
        prop_vertex,
        prop_edge,
        param_edge,
        ..
    } = state;
    p.rod_active.clear();
    for site in 0..sites {
        let interior = b.node.host()[3 * site + 1] as usize;
        p.rod_rest_scratch[site] = prop_vertex.host()[interior].rest_bend_angle;
    }
    // Seeds both halves, as the hinge pass does.
    p.rod_rest_angle.seed(device, &p.rod_rest_scratch[..sites])?;
    // AS THE FACE AND TET LAYERS: the early out is over the MATERIALS and the
    // gather is a dispatch. The rod one is PER SITE and averages the two
    // incident edges, which is why it reads the pair list rather than an edge.
    if !edge_params.iter().any(|material| material.plasticity > 0.0) {
        return Ok(());
    }

    let count = sites as u32;
    let material_args = PlasticityRodFromRecordsArgs {
        site_edge: b.edge_device.handle(),
        edge_prop: prop_edge.handle(),
        edge_param: param_edge.handle(),
        plasticity: p.rod_plasticity.handle(),
        threshold: p.rod_threshold.handle(),
        count,
        seam_arena_count: 0,
    };
    // Safety: every array is borrowed for the whole call.
    unsafe { device.launch("plasticity.rod.material", &material_args, count) }?;
    let p = &mut state.plastic;
    let alpha_args = PlasticityAlphaArgs {
        plasticity: p.rod_plasticity.handle(),
        dt,
        alpha: p.rod_alpha.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.rod_bend.alpha", &alpha_args, count)?;
    let angle_args = RodBendAngleArgs {
        x: pose,
        node_index: b.node.handle(),
        // The bound the entry point checks the site's three slots against,
        // which the record carries because a slot is data rather than the
        // thread index.
        vertex_count: vertices as u32,
        angle: p.rod_angle.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.rod_bend.angle", &angle_args, count)?;
    let rest_args = PlasticityCreepRestAngleArgs {
        angle: p.rod_angle.handle(),
        threshold: p.rod_threshold.handle(),
        alpha: p.rod_alpha.handle(),
        rest_angle: p.rod_rest_angle.handle(),
        changed: p.rod_changed.handle(),
        count,
        seam_arena_count: 0,
    };
    device.launch("plasticity.rod_bend.rest_angle", &rest_args, count)?;
    // THE CREPT ANGLES COME BACK, AND THE VERDICT DOES NOT, for the reason the
    // hinge pass above gives: the reference creeps `vp.rest_bend_angle` in
    // place on the device and materializes no verdict, and the creep here is
    // the same in-place update, so a site inside its dead zone still holds the
    // exact value seeded from `prop_vertex` above.
    p.rod_rest_angle.download(device)?;

    // ONE SITE PER INTERIOR VERTEX, so no two writes below land on the same
    // vertex and the scatter needs no accumulation.
    //
    // THE SEED IS WHAT SAYS WHETHER A SITE MOVED. `at()` marks the whole staged
    // buffer stale, so the upload below is owed the moment this loop touches
    // it; the test that avoids paying it on a step where nothing yielded is the
    // host's own seed against what came back, which is exact because the kernel
    // leaves an unyielded angle untouched.
    // OVER EVERY SITE, because there is no compacted list any more. The test
    // just below is what made the list unnecessary rather than merely
    // redundant: the kernel leaves an unyielded angle UNTOUCHED, so a site
    // that did not creep compares equal to its own seed and is skipped here
    // exactly as it was skipped by not being in the list.
    let mut crept = false;
    for site in 0..sites {
        let angle = p.rod_rest_angle.host()[site];
        if angle == p.rod_rest_scratch[site] {
            continue;
        }
        let interior = b.node.host()[3 * site + 1] as usize;
        prop_vertex.at()[interior].rest_bend_angle = angle;
        crept = true;
    }
    // ONLY WHEN SOMETHING MOVED. `at()` marks the device copy stale whether or
    // not a value changed, and `handle()` panics while it is, so an upload is
    // owed the moment the loop above touched it.
    if crept {
        prop_vertex.upload(device)?;
    }
    Ok(())
}

/// Fill the hinge permutation table, once, from the mesh's hinge indices.
///
/// # Safety
/// `data` must address a live `DataSet` and `node` be `4 * hinges` long.
pub unsafe fn fill_hinge_nodes<D: Device>(
    device: &mut D,
    // The hinge topology, as the handle the state already stages.
    hinge: ppf_cts_compute::Handle,
    hinges: usize,
    node: ppf_cts_compute::Handle,
) -> FatalResult<()> {
    if hinges == 0 {
        return Ok(());
    }
    let args = ShellBendRemapArgs {
        hinge,
        remapped: node,
        count: hinges as u32,
        seam_arena_count: 0,
    };
    device.launch("plasticity.hinge.remap", &args, hinges as u32)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;
    use ppf_cts_compute::{AllocLabel, HostDevice};
    use crate::cvec::CVec;
    use crate::cvecvec::CVecVec;
    use crate::data::{Mat2x2f, Mat3x3f, Vec2u, Vec3u, Vec4u};

    use super::super::test_scene::TestScene;

    /// The dead-zone creep the shared body applies to one scalar, spelled out
    /// here rather than called, so a test compares two independent statements
    /// of the rule. `plasticity` and `dt` enter only through `alpha`.
    fn expected_scalar(value: f32, rest: f32, threshold: f32, plasticity: f32, dt: f32) -> f32 {
        let delta = value - rest;
        if delta.abs() <= threshold {
            return rest;
        }
        let alpha = 1.0 - (-plasticity * dt).exp();
        let target = if delta > 0.0 { threshold } else { -threshold };
        rest + alpha * (delta - target)
    }

    /// The crept singular value: the same rule stated about a deviation from 1.
    fn expected_singular(sigma: f32, threshold: f32, plasticity: f32, dt: f32) -> f32 {
        if (sigma - 1.0).abs() <= threshold {
            return sigma;
        }
        let alpha = 1.0 - (-plasticity * dt).exp();
        let target = if sigma < 1.0 { 1.0 - threshold } else { 1.0 + threshold };
        sigma + alpha * (target - sigma)
    }

    fn close(actual: f32, expected: f32, what: &str) {
        let tolerance = 1e-6 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what}: got {actual}, expected {expected}"
        );
    }

    /// A single shell face on the plane, with an identity rest matrix, so its
    /// deformation gradient is the edge matrix itself.
    struct Face {
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

    impl Face {
        fn new(plasticity: f32, threshold: f32) -> Self {
            let mut scene = TestScene::new(3).with_faces(&[Vec3u::new(0, 1, 2)]);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.0, 0.0, 0.0);
            scene.place(2, 0.0, 1.0, 0.0);
            scene.data.param_arrays.face = CVec::from(
                &[FaceParam {
                    plasticity,
                    plasticity_threshold: threshold,
                    ..FaceParam::default()
                }][..],
            );
            scene.data.inv_rest2x2 = CVec::from(&[Mat2x2f::identity()][..]);
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

        /// Stretch along x, which puts the deformation gradient at
        /// `diag(factor, 1)` and its singular values at `(factor, 1)`.
        fn stretch(&mut self, factor: f32) {
            self.scene.place(1, factor, 0.0, 0.0);
        }

        /// A SHEARED stretch, whose SVD carries a real rotation on both sides.
        /// A diagonal configuration reconstructs `U diag(sigma) V^T` exactly in
        /// fp32 and inverts exactly at 1.25, so it cannot tell an unwritten
        /// rest matrix from one rebuilt out of its own factors; this one can.
        fn shear(&mut self) {
            self.scene.place(1, 1.3, 0.0, 0.0);
            self.scene.place(2, 0.2, 0.9, 0.0);
            // The scene moved, so the device positions must move with it.
            crate::driver::state::reseed_positions(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
        }

        fn creep(&mut self, dt: f32) {
            // The creep reads the COMMITTED pose and the staged props, both
            // device-resident, so they track the scene the fixture just edited.
            crate::driver::state::reseed_committed(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            crate::driver::state::reseed_props(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            crate::driver::state::reseed_inv_rest(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            // Safety: the scene is live and the state was allocated for it.
            unsafe { super::creep(&mut self.device, &self.scene.data, &mut self.state, dt) }
                .expect("the creep runs on the fixture");
            // AND THE CREPT VALUES COME BACK, which is what `fetch_rest_angles`
            // and `fetch_inv_rest` do in a real run: the creep writes the staged
            // copies, and the `DataSet` arrays this fixture reads are refreshed
            // from them.
            let n = self.state.prop_vertex.len();
            if n > 0 {
                // Safety: the scene carries at least `n` vertex props.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.prop_vertex.host().as_ptr(),
                        self.scene.data.prop.vertex.data,
                        n,
                    );
                }
            }
            // THE MIRRORS FIRST. The commit is a kernel now, so the device is
            // the authority and `host()` refuses a mirror that has not been
            // downloaded since a dispatch named the buffer. `fetch_inv_rest`
            // owes the same download for the same reason.
            self.state
                .inv_rest2x2
                .download(&mut self.device)
                .expect("the shell rest matrices read back");
            self.state
                .inv_rest3x3
                .download(&mut self.device)
                .expect("the tet rest matrices read back");
            let n2 = self.state.inv_rest2x2.len();
            if n2 > 0 {
                // Safety: the scene carries `n2` floats of shell rest matrices.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.inv_rest2x2.host().as_ptr(),
                        self.scene.data.inv_rest2x2.data as *mut f32,
                        n2,
                    );
                }
            }
            let n3 = self.state.inv_rest3x3.len();
            if n3 > 0 {
                // Safety: as above, for the tet rest matrices.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.inv_rest3x3.host().as_ptr(),
                        self.scene.data.inv_rest3x3.data as *mut f32,
                        n3,
                    );
                }
            }
        }

        fn rest(&self) -> Mat2x2f {
            self.scene.data.inv_rest2x2.as_slice()[0]
        }
    }

    #[test]
    fn a_stretched_face_creeps_its_rest_matrix_toward_the_yield_surface() {
        // The rest matrix this produces is derivable by hand. With the face in
        // the xy plane and stretched along x, the tangent frame is the identity
        // on the first two rows, so the projected edge matrix is
        // `diag(factor, 1)` and the crept rest matrix is
        // `diag(sigma_new / factor, 1)`.
        const FACTOR: f32 = 1.25;
        const PLASTICITY: f32 = 4.0;
        const THRESHOLD: f32 = 0.1;
        const DT: f32 = 0.05;

        let mut face = Face::new(PLASTICITY, THRESHOLD);
        face.stretch(FACTOR);
        face.creep(DT);

        let sigma_new = expected_singular(FACTOR, THRESHOLD, PLASTICITY, DT);
        let rest = face.rest();
        close(rest[(0, 0)], sigma_new / FACTOR, "the crept warp rest entry");
        close(rest[(1, 1)], 1.0, "the unyielded weft rest entry");
        close(rest[(0, 1)], 0.0, "the off-diagonal rest entry");
        close(rest[(1, 0)], 0.0, "the off-diagonal rest entry");
        // The crept spectrum, which is what the yield rule is about: the new
        // rest matrix must reproduce the singular values the shared body
        // returned, so the face reads as less stretched than it did.
        assert!(
            sigma_new < FACTOR && sigma_new > 1.0 + THRESHOLD,
            "the crept singular value {sigma_new} must lie between the yield surface and the \
             deformed state"
        );
    }

    #[test]
    fn the_face_creep_rate_follows_the_substep_it_is_given() {
        // `alpha = 1 - exp(-plasticity * dt)`, so a longer substep creeps
        // further. This is what makes passing the RAW substep rather than the
        // TOI-shortened one a visible decision rather than a tidy-up.
        const FACTOR: f32 = 1.25;
        const PLASTICITY: f32 = 4.0;
        const THRESHOLD: f32 = 0.1;

        let mut short = Face::new(PLASTICITY, THRESHOLD);
        short.stretch(FACTOR);
        short.creep(0.01);
        let mut long = Face::new(PLASTICITY, THRESHOLD);
        long.stretch(FACTOR);
        long.creep(0.10);

        close(
            short.rest()[(0, 0)],
            expected_singular(FACTOR, THRESHOLD, PLASTICITY, 0.01) / FACTOR,
            "the short substep's crept rest entry",
        );
        close(
            long.rest()[(0, 0)],
            expected_singular(FACTOR, THRESHOLD, PLASTICITY, 0.10) / FACTOR,
            "the long substep's crept rest entry",
        );
        assert!(
            long.rest()[(0, 0)] < short.rest()[(0, 0)],
            "a longer substep must creep further"
        );
    }

    #[test]
    fn a_face_inside_its_dead_zone_keeps_the_exact_bytes_of_its_rest_matrix() {
        // THE `changed` VERDICT, and why the crept rest matrix is staged rather
        // than written in place. This face is stretched, but by less than its
        // threshold, so the reference kernel returns before writing. Rebuilding
        // its rest matrix from its own SVD would land within round-off of the
        // identity and NOT on it, and the error would accumulate over a run on
        // a scene that is barely deforming at all.
        let mut face = Face::new(4.0, 0.5);
        face.shear();
        let before = face.rest();
        face.creep(0.05);
        let after = face.rest();
        for slot in 0..4 {
            assert_eq!(
                before.as_slice()[slot].to_bits(),
                after.as_slice()[slot].to_bits(),
                "an unyielded face's rest matrix must be bit-identical, entry {slot}"
            );
        }
    }

    #[test]
    fn a_face_whose_material_asks_for_no_creep_allocates_nothing_and_is_untouched() {
        let mut face = Face::new(0.0, 0.1);
        face.stretch(2.0);
        assert!(
            !face.state.plastic.any(),
            "a scene with no plastic material runs no creep at all"
        );
        assert!(
            face.state.plastic.face_plasticity.is_empty(),
            "and allocates none of the creep's scratch"
        );
        face.creep(0.05);
        assert_eq!(face.rest(), Mat2x2f::identity());
    }

    #[test]
    fn the_face_creep_walks_the_shell_prefix() {
        // A solid's surface triangles sit in `mesh.face` AFTER the shell
        // prefix and have no entry in `inv_rest2x2` at all, so a pass ranged
        // over `face.size` would read and write past the end of the rest
        // matrices on every tetrahedralized scene and be correct on every
        // shell-only one.
        let mut scene = TestScene::new(4).with_faces(&[
            Vec3u::new(0, 1, 2),
            Vec3u::new(0, 2, 3),
            Vec3u::new(1, 2, 3),
            Vec3u::new(0, 1, 3),
        ]);
        scene.place(0, 0.0, 0.0, 0.0);
        scene.place(1, 1.25, 0.0, 0.0);
        scene.place(2, 0.0, 1.0, 0.0);
        scene.place(3, 0.0, 0.0, 1.0);
        // Two of the four faces are the shell; the other two belong to a solid.
        scene.data.shell_face_count = 2;
        scene.data.param_arrays.face = CVec::from(
            &[FaceParam {
                plasticity: 4.0,
                plasticity_threshold: 0.1,
                ..FaceParam::default()
            }][..],
        );
        scene.data.inv_rest2x2 = CVec::from(&[Mat2x2f::identity(); 2][..]);
        let mut state = SolverState::default();
        // ONE DEVICE, for the reason recorded on the fixture structs above.
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }
            .expect("the fixture scene allocates");
        assert_eq!(
            state.plastic.face_plasticity.len(),
            2,
            "the creep's scratch is sized over the shell prefix"
        );
        // Safety: as above.
        unsafe { super::creep(&mut device, &scene.data, &mut state, 0.05) }.expect("the creep runs");
        // THE WITNESS IS THE MATERIAL PASS, NOT AN ACTIVE LIST. The creep does
        // not compact a list on the host; it gathers the material on the device
        // over the shell prefix and every stage runs over that range.
        // So what says "only the shell prefix creeps" is that the material
        // pass wrote a rate for exactly those faces, which is also the array
        // the dispatches read.
        state
            .plastic
            .face_plasticity
            .download(&mut device)
            .expect("the creep rate reads back");
        let rates = state.plastic.face_plasticity.host();
        assert_eq!(
            rates.len(),
            2,
            "only the shell prefix may be creeping; the two solid surface triangles have no \
             rest matrix to write"
        );
        assert!(
            rates.iter().all(|rate| *rate > 0.0),
            "every shell face in this fixture carries a creeping material, so the device \
             gather must have written a rate for both: {rates:?}"
        );
    }

    /// A single tet with an identity rest matrix.
    struct Tet {
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

    impl Tet {
        fn new(plasticity: f32, threshold: f32) -> Self {
            let mut scene = TestScene::new(4).with_tets(&[Vec4u::new(0, 1, 2, 3)]);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.0, 0.0, 0.0);
            scene.place(2, 0.0, 1.0, 0.0);
            scene.place(3, 0.0, 0.0, 1.0);
            scene.data.param_arrays.tet = CVec::from(
                &[TetParam {
                    plasticity,
                    plasticity_threshold: threshold,
                    ..TetParam::default()
                }][..],
            );
            scene.data.inv_rest3x3 = CVec::from(&[Mat3x3f::identity()][..]);
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

        fn stretch(&mut self, factor: f32) {
            self.scene.place(1, factor, 0.0, 0.0);
        }

        /// As `Face::shear`, and for the same reason.
        fn shear(&mut self) {
            self.scene.place(1, 1.3, 0.0, 0.0);
            self.scene.place(2, 0.2, 0.9, 0.0);
            self.scene.place(3, 0.1, 0.3, 1.1);
            // The scene moved, so the device positions must move with it.
            crate::driver::state::reseed_positions(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
        }

        fn creep(&mut self, dt: f32) {
            // The creep reads the COMMITTED pose and the staged props, both
            // device-resident, so they track the scene the fixture just edited.
            crate::driver::state::reseed_committed(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            crate::driver::state::reseed_props(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            crate::driver::state::reseed_inv_rest(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            // Safety: the scene is live and the state was allocated for it.
            unsafe { super::creep(&mut self.device, &self.scene.data, &mut self.state, dt) }
                .expect("the creep runs on the fixture");
            // AND THE CREPT VALUES COME BACK, which is what `fetch_rest_angles`
            // and `fetch_inv_rest` do in a real run: the creep writes the staged
            // copies, and the `DataSet` arrays this fixture reads are refreshed
            // from them.
            let n = self.state.prop_vertex.len();
            if n > 0 {
                // Safety: the scene carries at least `n` vertex props.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.prop_vertex.host().as_ptr(),
                        self.scene.data.prop.vertex.data,
                        n,
                    );
                }
            }
            // THE MIRRORS FIRST. The commit is a kernel now, so the device is
            // the authority and `host()` refuses a mirror that has not been
            // downloaded since a dispatch named the buffer. `fetch_inv_rest`
            // owes the same download for the same reason.
            self.state
                .inv_rest2x2
                .download(&mut self.device)
                .expect("the shell rest matrices read back");
            self.state
                .inv_rest3x3
                .download(&mut self.device)
                .expect("the tet rest matrices read back");
            let n2 = self.state.inv_rest2x2.len();
            if n2 > 0 {
                // Safety: the scene carries `n2` floats of shell rest matrices.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.inv_rest2x2.host().as_ptr(),
                        self.scene.data.inv_rest2x2.data as *mut f32,
                        n2,
                    );
                }
            }
            let n3 = self.state.inv_rest3x3.len();
            if n3 > 0 {
                // Safety: as above, for the tet rest matrices.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.inv_rest3x3.host().as_ptr(),
                        self.scene.data.inv_rest3x3.data as *mut f32,
                        n3,
                    );
                }
            }
        }

        fn rest(&self) -> Mat3x3f {
            self.scene.data.inv_rest3x3.as_slice()[0]
        }
    }

    #[test]
    fn a_stretched_tet_creeps_its_rest_matrix_toward_the_yield_surface() {
        // A tet's edge matrix is square, so the crept rest matrix is
        // `edges^-1 * U diag(sigma_new) V^T`, which for this configuration is
        // `diag(sigma_new / factor, 1, 1)`.
        const FACTOR: f32 = 1.25;
        const PLASTICITY: f32 = 4.0;
        const THRESHOLD: f32 = 0.1;
        const DT: f32 = 0.05;

        let mut tet = Tet::new(PLASTICITY, THRESHOLD);
        tet.stretch(FACTOR);
        tet.creep(DT);

        let sigma_new = expected_singular(FACTOR, THRESHOLD, PLASTICITY, DT);
        let rest = tet.rest();
        close(rest[(0, 0)], sigma_new / FACTOR, "the crept rest entry");
        close(rest[(1, 1)], 1.0, "the unyielded rest entry");
        close(rest[(2, 2)], 1.0, "the unyielded rest entry");
    }

    #[test]
    fn a_tet_inside_its_dead_zone_keeps_the_exact_bytes_of_its_rest_matrix() {
        let mut tet = Tet::new(4.0, 0.5);
        tet.shear();
        let before = tet.rest();
        tet.creep(0.05);
        let after = tet.rest();
        for slot in 0..9 {
            assert_eq!(
                before.as_slice()[slot].to_bits(),
                after.as_slice()[slot].to_bits(),
                "an unyielded tet's rest matrix must be bit-identical, entry {slot}"
            );
        }
    }

    #[test]
    fn the_crept_rest_shape_is_what_the_checkpoint_serializes() {
        // THE CHECKPOINT HALF, and the reason this backend needs no copy back.
        // `backend.rs` writes the per-frame `plastic_<N>.bin.gz` out of the
        // `DataSet`'s own arrays after calling `fetch_inv_rest()`, which is
        // empty here because decision D1 is direct addressing. That is only
        // true while the creep writes into the scene's buffers, so this asks
        // `PlasticState::extract` what it would have stored.
        const FACTOR: f32 = 1.25;
        const PLASTICITY: f32 = 4.0;
        const THRESHOLD: f32 = 0.1;
        const DT: f32 = 0.05;

        let mut face = Face::new(PLASTICITY, THRESHOLD);
        face.stretch(FACTOR);
        face.creep(DT);

        let kinds = PlasticKinds::of(&face.scene.data);
        assert!(kinds.face && kinds.needs_inv_rest());
        let saved = crate::plastic_state::PlasticState::extract(&face.scene.data, kinds);
        assert_eq!(saved.inv_rest2x2.len(), 1);
        let sigma_new = expected_singular(FACTOR, THRESHOLD, PLASTICITY, DT);
        close(
            saved.inv_rest2x2[0][(0, 0)],
            sigma_new / FACTOR,
            "the rest matrix the checkpoint would carry",
        );
        assert_ne!(
            saved.inv_rest2x2[0], Mat2x2f::identity(),
            "a checkpoint carrying the BUILD-TIME rest shape pairs this frame's pose with a \
             rest shape from a different time"
        );
    }

    // TWO LAUNCHERS RE-DECLARED IN A TEST, AND PRODUCTION REACHES BOTH THROUGH
    // THE SEAM.
    //
    // Each takes a thread range, so each IS a dispatch: `creep` above launches
    // them as `id::SHELL_BEND_ANGLE` and `id::ROD_BEND_ANGLE`. What is here is
    // the same body called flat, so the test can read a rest angle without
    // building a device, a table lookup and a record around one number it then
    // compares against.
    //
    // THE STALE-MIRROR HAZARD IS REAL AND THE COMPILER IS WHAT CATCHES IT. This
    // declaration and the backend's are two spellings of one symbol in one
    // crate, so converting either launcher to a generated entry point, whose
    // signature is the record plus the arena bases plus the range, makes the two
    // disagree and rustc reports `clashing_extern_declarations`. That is what it
    // did when `assemble.rs` carried a second, older declaration of
    // `svd3x2_entry`, and it is why a re-declaration is written beside
    // the test that needs it rather than hidden in a helper module.
    /// One hinge over two triangles sharing an edge, folded out of plane.
    struct Hinge {
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

    impl Hinge {
        fn new(plasticity: f32, threshold: f32, rest_angle: f32) -> Self {
            let mut scene = TestScene::new(4).with_hinges(&[Vec4u::new(0, 1, 2, 3)]);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.0, 0.0, 0.0);
            scene.place(2, 0.5, 0.75, 0.0);
            scene.place(3, 0.5, -0.5, 0.375);
            // The PREVIOUS pose is deliberately different, so a creep reading
            // the wrong array measures a different angle and this fixture says
            // so rather than agreeing by accident.
            for slot in scene.data.vertex.prev.as_mut_slice() {
                *slot = super::super::test_scene::position(0.0, 0.0, 0.0);
            }
            scene.data.param_arrays.hinge = CVec::from(
                &[HingeParam {
                    plasticity,
                    plasticity_threshold: threshold,
                    ..HingeParam::default()
                }][..],
            );
            scene.data.prop.hinge.as_mut_slice()[0].rest_angle = rest_angle;
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

        /// The hinge's dihedral angle at the committed pose, through the same
        /// entry point the creep reads it with but from this test's own call.
        /// DISPATCHES ON THE FIXTURE'S OWN DEVICE, not a fresh one. The stencil
        /// below is a HANDLE into the arena `allocate` filled, and a handle
        /// means nothing to another backend instance: a second `host_device()`
        /// would resolve the same arena index against its own empty table and
        /// read four zeros, which is a degenerate hinge and an angle of zero.
        fn angle(&mut self) -> f32 {
            let mut angle_out = Buffer::<f32>::none();
            angle_out
                .size(&mut self.device, 1, AllocLabel("test.hinge.angle"))
                .expect("the angle allocation succeeds");
            let vertices = self.state.sizes.vertices;
            // THE RECORD PRODUCTION FILLS, DISPATCHED THE WAY PRODUCTION
            // DISPATCHES IT: this entry point is generated, so it reaches a
            // backend through `Device::launch` and nothing here spells its
            // symbol.
            let args = crate::driver::kernels::ShellBendAngleArgs {
                // Safety: the scene is live, the stencil table was filled at
                // allocation and names in-range vertices of it, and `out` is a
                // device allocation the call fills and the line below reads back.
                x: self.state.positions.handle(),
                // The stencil's four slots, as a prefix of the creep array
                // rather than a pointer into it.
                hinge: self.state.plastic.hinge_node.span(0, 4),
                vertex_count: vertices as u32,
                angle: angle_out.handle(),
                count: 1,
                seam_arena_count: 0,
            };
            // Safety: every reference in the record names live storage that
            // outlives this dispatch.
            unsafe { self.device.launch("test.hinge.angle", &args, 1) }
                .expect("the dihedral angle dispatches");
            let angle = angle_out
                .read_one(&mut self.device, 0)
                .expect("the dihedral angle reads back");
            angle_out
                .free(&mut self.device)
                .expect("the angle buffer frees");
            angle
        }

        fn creep(&mut self, dt: f32) {
            // The creep reads the COMMITTED pose and the staged props, both
            // device-resident, so they track the scene the fixture just edited.
            crate::driver::state::reseed_committed(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            crate::driver::state::reseed_props(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            // THE HINGE RECORDS TOO, because the creep reads `fixed` and the
            // material IN THE THREAD now. A fixture that flips `fixed` after
            // `allocate` and re-stages only the vertex and face props hands
            // the kernel the build-time hinge, which is what
            // `a_fixed_hinge_does_not_creep` caught.
            crate::driver::state::reseed_hinge(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            crate::driver::state::reseed_inv_rest(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            // Safety: the scene is live and the state was allocated for it.
            unsafe { super::creep(&mut self.device, &self.scene.data, &mut self.state, dt) }
                .expect("the creep runs on the fixture");
            // AND THE CREPT VALUES COME BACK, which is what `fetch_rest_angles`
            // and `fetch_inv_rest` do in a real run: the creep writes the staged
            // copies, and the `DataSet` arrays this fixture reads are refreshed
            // from them.
            let n = self.state.prop_vertex.len();
            if n > 0 {
                // Safety: the scene carries at least `n` vertex props.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.prop_vertex.host().as_ptr(),
                        self.scene.data.prop.vertex.data,
                        n,
                    );
                }
            }
            // THE MIRRORS FIRST. The commit is a kernel now, so the device is
            // the authority and `host()` refuses a mirror that has not been
            // downloaded since a dispatch named the buffer. `fetch_inv_rest`
            // owes the same download for the same reason.
            self.state
                .inv_rest2x2
                .download(&mut self.device)
                .expect("the shell rest matrices read back");
            self.state
                .inv_rest3x3
                .download(&mut self.device)
                .expect("the tet rest matrices read back");
            let n2 = self.state.inv_rest2x2.len();
            if n2 > 0 {
                // Safety: the scene carries `n2` floats of shell rest matrices.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.inv_rest2x2.host().as_ptr(),
                        self.scene.data.inv_rest2x2.data as *mut f32,
                        n2,
                    );
                }
            }
            let n3 = self.state.inv_rest3x3.len();
            if n3 > 0 {
                // Safety: as above, for the tet rest matrices.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.inv_rest3x3.host().as_ptr(),
                        self.scene.data.inv_rest3x3.data as *mut f32,
                        n3,
                    );
                }
            }
        }

        fn rest_angle(&self) -> f32 {
            self.scene.data.prop.hinge.as_slice()[0].rest_angle
        }
    }

    #[test]
    fn a_folded_hinge_creeps_its_rest_angle_by_the_dead_zone_rule() {
        const PLASTICITY: f32 = 4.0;
        const THRESHOLD: f32 = 0.05;
        const REST: f32 = 0.0;
        const DT: f32 = 0.05;

        let mut hinge = Hinge::new(PLASTICITY, THRESHOLD, REST);
        let angle = hinge.angle();
        assert!(
            (angle - REST).abs() > THRESHOLD,
            "the fixture must fold the hinge past its dead zone, got {angle}"
        );
        hinge.creep(DT);
        close(
            hinge.rest_angle(),
            expected_scalar(angle, REST, THRESHOLD, PLASTICITY, DT),
            "the crept hinge rest angle",
        );
    }

    #[test]
    fn the_crept_rest_angle_is_what_the_checkpoint_serializes() {
        // THE SECOND HALF OF THE CHECKPOINT PAIRING, over the rest ANGLES
        // rather than the rest matrices. `backend.rs` calls `fetch_rest_angles()`
        // before serializing and that is empty here for the same reason
        // `fetch_inv_rest()` is, so the file is right only while the creep
        // writes `HingeProp::rest_angle` in the scene itself. A pass that
        // scattered into a mirror would leave every assertion about the creep
        // passing and the per-frame file carrying the build-time angle.
        const PLASTICITY: f32 = 4.0;
        const THRESHOLD: f32 = 0.05;
        const REST: f32 = 0.0;
        const DT: f32 = 0.05;

        let mut hinge = Hinge::new(PLASTICITY, THRESHOLD, REST);
        let angle = hinge.angle();
        hinge.creep(DT);

        let kinds = PlasticKinds::of(&hinge.scene.data);
        assert!(kinds.hinge && kinds.needs_rest_angles());
        let saved = crate::plastic_state::PlasticState::extract(&hinge.scene.data, kinds);
        assert_eq!(saved.hinge_rest_angle.len(), 1);
        close(
            saved.hinge_rest_angle[0],
            expected_scalar(angle, REST, THRESHOLD, PLASTICITY, DT),
            "the rest angle the checkpoint would carry",
        );
        assert_ne!(
            saved.hinge_rest_angle[0], REST,
            "a checkpoint carrying the BUILD-TIME rest angle pairs this frame's pose with a \
             rest shape from a different time"
        );
    }

    #[test]
    fn a_fixed_hinge_does_not_creep() {
        // `update_hinge_plasticity`'s one flag test, and the only one it has:
        // it consults neither `collider` nor the hinge's type byte.
        let mut hinge = Hinge::new(4.0, 0.05, 0.0);
        hinge.scene.data.prop.hinge.as_mut_slice()[0].fixed = true;
        hinge.creep(0.05);
        assert_eq!(hinge.rest_angle().to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn a_hinge_inside_its_dead_zone_does_not_creep() {
        let mut hinge = Hinge::new(4.0, 0.05, 0.0);
        let angle = hinge.angle();
        // A rest angle at the current fold: the deviation is exactly zero, so
        // the hinge is inside any dead zone.
        hinge.scene.data.prop.hinge.as_mut_slice()[0].rest_angle = angle;
        hinge.creep(0.05);
        assert_eq!(hinge.rest_angle().to_bits(), angle.to_bits());
    }

    /// Three vertices and two rod segments, bent at the interior one.
    struct RodBend {
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

    impl RodBend {
        fn new(first: EdgeParam, second: EdgeParam, rest_angle: f32) -> Self {
            let edges = [Vec2u::new(0, 1), Vec2u::new(1, 2)];
            let mut scene = TestScene::new(3).with_edges(&edges);
            scene.place(0, 0.0, 0.0, 0.0);
            scene.place(1, 1.0, 0.0, 0.0);
            scene.place(2, 1.75, 0.5, 0.0);
            scene.data.rod_count = 2;
            scene.data.surface_vert_count = 3;
            scene.data.param_arrays.edge = CVec::from(&[first, second][..]);
            {
                let props = scene.data.prop.edge.as_mut_slice();
                props[0].param_index = 0;
                props[1].param_index = 1;
            }
            scene.data.prop.vertex.as_mut_slice()[1].rest_bend_angle = rest_angle;
            let mut incident: Vec<Vec<u32>> = vec![Vec::new(); 3];
            for (index, edge) in edges.iter().enumerate() {
                incident[edge[0] as usize].push(index as u32);
                incident[edge[1] as usize].push(index as u32);
            }
            scene.data.mesh.neighbor.vertex.edge = CVecVec::from(&incident[..]);
            scene.data.mesh.neighbor.vertex.face = CVecVec::from(&vec![Vec::new(); 3][..]);
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

        /// `&mut self`, AND ON THE FIXTURE'S OWN TARGET rather than on a
        /// fresh one, because the stencil table is a device allocation now: a
        /// `HostDevice` keeps its arenas per instance, so the handle below
        /// resolves only against the target that cut it. That is also what
        /// production does, which the hinge oracle above can only approximate
        /// while its own table is still a host array.
        fn angle(&mut self) -> f32 {
            let mut angle_out = Buffer::<f32>::none();
            angle_out
                .size(&mut self.device, 1, AllocLabel("test.rod_bend.angle"))
                .expect("the angle allocation succeeds");
            let vertices = self.state.sizes.vertices;
            // Dispatched the way production dispatches it; the hinge oracle
            // above states why.
            let args = crate::driver::kernels::RodBendAngleArgs {
                // Safety: the scene is live, the sites were enumerated at
                // allocation and name in-range vertices of it, and `out` is a
                // device allocation the call fills and the line below reads back.
                x: self.state.positions.handle(),
                node_index: self.state.rod_bend.node.handle(),
                vertex_count: vertices as u32,
                angle: angle_out.handle(),
                count: 1,
                seam_arena_count: 0,
            };
            // Safety: as the hinge oracle above.
            unsafe { self.device.launch("test.rod_bend.angle", &args, 1) }
                .expect("the turning angle dispatches");
            let angle = angle_out
                .read_one(&mut self.device, 0)
                .expect("the turning angle reads back");
            angle_out
                .free(&mut self.device)
                .expect("the angle buffer frees");
            angle
        }

        fn creep(&mut self, dt: f32) {
            // The creep reads the COMMITTED pose and the staged props, both
            // device-resident, so they track the scene the fixture just edited.
            crate::driver::state::reseed_committed(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            crate::driver::state::reseed_props(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            crate::driver::state::reseed_inv_rest(
                &mut self.device,
                &mut self.state,
                &self.scene.data,
            );
            // Safety: the scene is live and the state was allocated for it.
            unsafe { super::creep(&mut self.device, &self.scene.data, &mut self.state, dt) }
                .expect("the creep runs on the fixture");
            // AND THE CREPT VALUES COME BACK, which is what `fetch_rest_angles`
            // and `fetch_inv_rest` do in a real run: the creep writes the staged
            // copies, and the `DataSet` arrays this fixture reads are refreshed
            // from them.
            let n = self.state.prop_vertex.len();
            if n > 0 {
                // Safety: the scene carries at least `n` vertex props.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.prop_vertex.host().as_ptr(),
                        self.scene.data.prop.vertex.data,
                        n,
                    );
                }
            }
            // THE MIRRORS FIRST. The commit is a kernel now, so the device is
            // the authority and `host()` refuses a mirror that has not been
            // downloaded since a dispatch named the buffer. `fetch_inv_rest`
            // owes the same download for the same reason.
            self.state
                .inv_rest2x2
                .download(&mut self.device)
                .expect("the shell rest matrices read back");
            self.state
                .inv_rest3x3
                .download(&mut self.device)
                .expect("the tet rest matrices read back");
            let n2 = self.state.inv_rest2x2.len();
            if n2 > 0 {
                // Safety: the scene carries `n2` floats of shell rest matrices.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.inv_rest2x2.host().as_ptr(),
                        self.scene.data.inv_rest2x2.data as *mut f32,
                        n2,
                    );
                }
            }
            let n3 = self.state.inv_rest3x3.len();
            if n3 > 0 {
                // Safety: as above, for the tet rest matrices.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.state.inv_rest3x3.host().as_ptr(),
                        self.scene.data.inv_rest3x3.data as *mut f32,
                        n3,
                    );
                }
            }
        }

        fn rest_angle(&self) -> f32 {
            self.scene.data.prop.vertex.as_slice()[1].rest_bend_angle
        }
    }

    fn rod_material(plasticity: f32, threshold: f32) -> EdgeParam {
        EdgeParam {
            plasticity,
            plasticity_threshold: threshold,
            ..EdgeParam::default()
        }
    }

    #[test]
    fn a_bent_rod_creeps_its_rest_turning_angle_using_the_two_segment_average() {
        // The two incident segments carry DIFFERENT materials, so a creep
        // reading one of them rather than averaging both lands somewhere else.
        const REST: f32 = 0.0;
        const DT: f32 = 0.05;
        let mut rod = RodBend::new(rod_material(2.0, 0.02), rod_material(6.0, 0.06), REST);
        assert_eq!(
            rod.state.sizes.rod_bend_sites, 1,
            "the fixture must enumerate its one interior rod vertex"
        );
        let angle = rod.angle();
        rod.creep(DT);
        close(
            rod.rest_angle(),
            expected_scalar(angle, REST, 0.04, 4.0, DT),
            "the crept rod rest turning angle",
        );
    }

    #[test]
    fn a_rod_whose_two_segments_average_to_no_creep_is_untouched() {
        // Both averages are taken over the two segments, so a strand with no
        // plastic material anywhere runs nothing; the zero rate is what the
        // reference kernel gates on.
        let mut rod = RodBend::new(rod_material(0.0, 0.02), rod_material(0.0, 0.06), 0.0);
        assert!(!rod.state.plastic.any());
        rod.creep(0.05);
        assert_eq!(rod.rest_angle().to_bits(), 0.0f32.to_bits());
    }
}
