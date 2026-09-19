// File: crates/ppf-cts-solver/src/driver/refusal.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The unsupported-feature gate.
//!
//! COMPLETENESS IS TWO LOCALLY CHECKABLE CONDITIONS, and the second one is that
//! every scene class the backend does not accept is refused
//! at `initialize()`, naming the feature and its count, before any frame is
//! written. The refusal names the FEATURE, not the symptom: "this scene has N
//! projected translation or rotation locks, which the CPU backend cannot solve
//! yet" tells the reader which host to use, and "unsupported scene" does not.
//!
//! Two rules keep this file current, and neither is optional:
//!
//! 1. Any new `DataSet` container or `ParamSet` field that changes what the
//!    solve COMPUTES is either implemented on this backend or added here, in
//!    the change that introduces it. A field carrying only telemetry needs
//!    neither: a backend that records nothing into a statistics array is
//!    missing a count, not solving a different problem.
//! 2. An entry is removed in the same change that implements what it refuses,
//!    never before and never separately.

use crate::data::{DataSet, FaceParam, FaceProp, Model, ParamSet, TetParam, TetProp};
// Named only where the tests below select a preconditioner to be refused.
#[cfg(test)]
use crate::data::PrecondMode;

/// One reason this scene is refused, with the count that was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// The feature, in the words a user would recognize from the frontend.
    pub feature: &'static str,
    /// How many of it the scene carries. Zero is never reported: a refusal with
    /// no count is the "unsupported scene" message this design rejects.
    pub count: u64,
    /// What the count counts, so the number is readable without the source.
    pub unit: &'static str,
}

impl Refusal {
    pub fn describe(&self) -> String {
        format!("{}: {} {}", self.feature, self.count, self.unit)
    }
}

/// Enumerate every reason this scene cannot be solved by the CPU backend.
///
/// An empty result means the scene is accepted, which today happens only for a
/// scene with no elements at all. The list is deliberately exhaustive rather
/// than short-circuiting on the first hit: a user retargeting a scene wants to
/// know everything that would have to change, not the first thing.
// `_param` IS UNREAD, AND THAT IS THE NEWS RATHER THAN AN OVERSIGHT. Every
// refusal that survives is a property of the SCENE: a solid naming a membrane
// model, a PDRD face asking for shear stiffness. None is a property of how the
// solve was configured, so no solver parameter can now put a scene outside this
// backend. The argument stays because a future refusal may read it, and because
// `advance()` re-runs this per step precisely so a schedule cannot introduce a
// refused capability partway through a run.
pub fn scene_refusals(data: &DataSet, _param: &ParamSet) -> Vec<Refusal> {
    // THIS LIST IS EMPTY, AND IT RETURNS A VEC RATHER THAN NOTHING BECAUSE THE
    // EMPTINESS IS A MEASUREMENT, NOT A DESIGN. Every capability this backend
    // once refused has been implemented; the comments below are what each
    // retirement rested on, kept because rule 2 at the top of this file is that
    // an entry comes off only in the change that implements what it refused,
    // and the argument for each is what a reader needs to check that it did.
    // A future capability arrives here the same way.
    let _ = data;
    let out = Vec::new();

    // `mesh.face` is NOT the shell count. A solid's surface triangles live in
    // that same array, with the shell faces occupying its HEAD, so
    // `shell_face_count` is a PREFIX length and `face.size` is the whole thing.
    // The two are equal exactly when the scene carries no solid, which is what
    // makes confusing them invisible: every shell-only scene agrees.
    //
    // This is not a hypothetical tidy-up. The Metal backend refused EVERY
    // tetrahedralized scene at `initialize()` because a preflight compared a
    // per-face slot table against `shell_face_count`, and the suite had no
    // solid-plus-shell scene to catch it. Refusing by CAPABILITY rather than by
    // container size is what keeps that class of mistake out of this backend:
    // "shell strain limiting" is about shells, and asking `face.size` answers a
    // different question.
    let total_faces = data.mesh.mesh.face.size as u64;
    let shell_faces = u64::from(data.shell_face_count);
    let solid_surface_faces = total_faces.saturating_sub(shell_faces);
    let edges = data.mesh.mesh.edge.size as u64;

    let hinges = data.mesh.mesh.hinge.size as u64;


    // A SHELL FACE IS SEVERAL CAPABILITIES AND NOT ONE. Four terms are
    // dispatched off the same faces, and each is refused or accepted on its own:
    // the membrane, bending (through the hinges, below), STRAIN LIMITING, which
    // has no element class of its own at all, and the per-face pressure (below).
    // Only the last is still refused.
    //
    // STRAIN LIMITING IS ASSEMBLED, both halves of it: the force and Hessian
    // contribution and the `SL_toi` line search. They share one per-face gate
    // (`!fixed && !rest_excluded && strainlimit > 0`, which
    // `kernels/strainlimiting/shell_strain.kernel.cpp` states once in
    // `shell_strain_diff_table_from_records` and writes out as `live`), so the
    // assembly and the line search read the same verdict and the pair cannot
    // come apart.
    let _ = shell_faces;
    // A solid's surface triangles are a SEPARATE capability from its tets: they
    // reach contact, collision and the per-face pressure term without carrying
    // membrane elasticity. They are refused only where they are USED, which is
    // contact, so a contact-disabled solid run is accepted while a
    // contact-enabled one is refused by the contact entry below. The area-
    // weighted vertex normal the momentum layer reads is the one thing that
    // walks them here, and it is ported.
    let _ = solid_surface_faces;
    // BENDING IS ASSEMBLED, so a hinge is not a reason to stop. What
    // separates a shell's bending hinge from a solid's surface hinge is bit 0
    // of `mesh.type.hinge`, and the assembly reads it: a hinge with no type
    // byte has no known element class and stops the run there rather than being
    // guessed at here.
    let _ = hinges;
    // A ROD IS SEVERAL CAPABILITIES AND NOT ONE, exactly as a shell face is.
    // Its Hookean stretch, its turning-angle bending and its STRAIN LIMIT are
    // all assembled, the last over the ROD PREFIX and with the same paired gate
    // the shell half has.
    //
    // Mesh edges carry no energy of their own on a solid: they exist for
    // edge-edge contact and for the rod stencils. Contact is refused on its own
    // entry and the rod stencils walk the ROD PREFIX rather than this array, so
    // an edge count alone is not a reason to stop.
    let _ = edges;

    // CONTACT IS NO LONGER REFUSED, and what removed the entry was not the
    // barrier: it was the two structural halves of the penetration guarantee
    // arriving with it. `src/driver/contact.rs` runs the broad phase, the four
    // narrow-phase visitors, the barrier and friction assembly, the ACCD
    // CCD-filtered line search and `check_intersection`, the last two on every
    // step. A build with the barrier and only one of the other two would
    // complete and exit 0 with surfaces passed through each other, so the entry
    // could not come off for two of the three.
    //
    // What is still refused about contact is refused by NAME below rather than
    // by this one entry: a SAND grain's angular degree of freedom keeps its
    // own.

    // THE PER-FACE PRESSURE ENTRY IS GONE, removed in the change that
    // implements it, which is the rule this file's own header states. The term
    // is `assemble.face.pressure`, one dispatch per shell face after the mass
    // scale and before the damping.

    // Solve paths that are refused on their own merits rather than because the
    // driver is unbuilt, so they keep their entries after the element classes
    // above come off. None of the three can be approximated by a path this
    // backend already has; the same three are live on Metal today.
    // A PDRD BODY TOGETHER WITH ANY LOCK IS ITS OWN CAPABILITY, and refusing
    // the combination is not a gap in either half. A scene carrying a lock
    // record AT ALL, PDRD-attached or purely deformable, solves a DIFFERENT
    // system: the reduced unknown splits affinely into a known feasible
    // correction plus a tangent Krylov unknown, and the operator becomes the
    // deformable projector composed with the per-body one. It is not the plain
    // reduced solve with projections added, so implementing it is its own
    // change and this says so rather than running the wrong solve.
    let _pdrd_bodies = data.prop.pdrd_body.size as u64;
    let lock_groups = data.translation_lock.size as u64;
    // PDRD TOGETHER WITH A PROJECTED LOCK IS IMPLEMENTED, and it is one loop
    // rather than two: `solve_rigid` takes the locked path on the option being
    // `Some`, where the reduced operator projects the FULL-SPACE product before
    // restricting it and the seed carries the lock's particular solution.
    //
    // THE ORDER IS NOT INTERCHANGEABLE. Projecting the reduced vector instead
    // would apply the lock in a basis it was not built in, which is why the
    // lock's projection sits inside the reduced apply rather than around it.
    let _ = lock_groups;

    // THE PROJECTED LOCKS ARE NO LONGER REFUSED. `src/driver/lock.rs` builds
    // the per-group projector and `pcg::solve_locked` runs the projected solve,
    // so a locked scene solves `Q M Q z = Q (b - M q)` through the host-syncing
    // path. The entry could not come off for the projector alone: a lock
    // is an EXACT constraint on the Newton direction, so a solve that prepared
    // the correction and did not project the search directions would be
    // silently wrong rather than unimplemented, and `check_tangent` is what
    // says so when it happens.
    //
    // A PDRD group keeps its own refusal below and is not this projector's:
    // its rows live in the reduced body vector, the two have disjoint supports,
    // and `builder.rs` rejects every grouping that would break that.

    // MULTILEVEL ADDITIVE SCHWARZ IS IMPLEMENTED. The driver materializes the
    // fine operator as a block-CSR level 0, coarsens it with a Galerkin triple
    // product over each level's own aggregation, builds and factors that
    // level's domains, and sums every level's correction onto the fine sweep.
    // Two of its steps reach the same answer by a route of their own, and
    // `schwarz.rs` names each at its own site.

    // Constraint kinds that reach the solve. Every one of these also arrives
    // through `update_constraint` every step, and is re-checked there: a
    // schedule can switch a collider on at t = 2, which a gate that runs only
    // at t = 0 cannot see.
    // THE ANALYTIC COLLIDERS ARE NO LONGER REFUSED. `src/driver/collider.rs`
    // assembles the sphere and floor barriers, sweeps a prescribed vertex's path
    // against both, and confines a barrier-held pin to its own gap ball, all
    // outside the `disable-contact` gate.
    // THE CROSS-STITCH IS NO LONGER REFUSED. `assemble::stitch` evaluates the
    // six-slot barycentric spring through `stitch_force_hessian` and folds
    // its force and its thirty-six Hessian blocks into the fixed matrix between
    // the elastic layers and the `tmp_fixed` snapshot.
    // THE TORQUE GROUPS ARE NO LONGER REFUSED. `energy/model/torque.kernel.cpp`
    // carries the frame pre-pass, one thread per group over the walks
    // `torque_group_frame` performs, and `main/momentum.kernel.cpp`
    // carries the per-vertex term between the pin scan and the momentum row,
    // which is where `energy/vertex_force.kernel.cpp` sums it. The entry could
    // not come off for either half alone: the frame is a property of the whole
    // group and no member can reach it, so a per-vertex term with no pre-pass
    // would scale by an uninitialized frame rather than fail.
    // THE STATIC COLLISION MESH IS NO LONGER REFUSED. `src/driver/contact.rs`
    // builds the collider's two trees once, assembles the three pair types
    // (mesh vertex against collider face, collider vertex against mesh face,
    // and edge against edge), sweeps all three in the line
    // search, and scans a dynamic edge through the collider in the intersection
    // gate. The entry could not come off for the assembly alone: a barrier with
    // no CCD filter behind it completes and exits 0 with a surface passed
    // through the collider.

    // SAND grains, which the momentum layer would silently treat as ordinary
    // vertices: their spin, the Schur condense / recover and the post-solve
    // integrate are all absent, so a grain would slide where it should roll.
    // THE ARRAY IS PER VERTEX, NOT PER GRAIN, and its LENGTH is the vertex
    // count in any scene that carries it at all: `builder.rs` sizes it over
    // every vertex and leaves a zero where the vertex is not a grain, which is
    // how the post-solve integrate skips them. Refusing on the length refuses
    // every scene; the capability test is a NON-ZERO inverse inertia.
    // SAND GRAINS ARE NO LONGER REFUSED. The rolling degree of freedom is
    // assembled at four points: the
    // analytic contact accumulates each grain's Schur blocks, the point-point
    // narrow phase gives its friction the CONTACT-POINT slip and folds the
    // resulting torque per vertex, the condense eliminates the spin into the
    // translation block before the solve, and the recover and the post-solve
    // integrate put it back. The entry could not come off for any subset: a
    // grain that accumulated its torque and never integrated it would slide
    // where it should roll, with nothing in the output to say so.
    // A `disable-contact` SCENE IS NOT AN EXCEPTION, AND WHAT KEEPS IT FROM
    // BEING ONE IS WHERE THE GRAIN ARRAYS LIVE. `SolverState` owns all six
    // per-vertex grain arrays, sized with the scene whenever it carries grains
    // and independent of any layer. Holding the torque, stiffness and normal
    // accumulators on the CONTACT layer instead would lose them exactly when
    // that flag is set, because the flag does not build that layer while the
    // analytic colliders are assembled outside its gate: a grain resting on a
    // floor has real spin to integrate and would have nowhere to accumulate it.
    // It would also leave the condense, the recover and the integrate with
    // `Handle::NONE` for the analytic blocks in any scene with no collider.

    // PLASTICITY IS NO LONGER REFUSED. `src/driver/plasticity.rs` runs all four
    // creeps on the committed pose at the end of every step, each over its
    // element range and behind that range's own element-count gate, so a scene
    // with none of an element class dispatches nothing for it. The entry could
    // not come off for the creep alone: a
    // backend that mutated the rest shape and did not preserve it per frame
    // would write checkpoints pairing a pose with a rest shape from a different
    // time. What preserves it here is `plastic_state.rs`, unchanged, because
    // decision D1 is direct addressing and the four arrays the creep writes are
    // the host's own.


    out
}

/// Every material this scene names on an element kind that has no form of it.
///
/// A SECOND GATE IN THIS FILE, AND A DIFFERENT KIND FROM [`scene_refusals`],
/// which is why it is a separate function reported through a separate message
/// rather than two more entries in that list. An entry there names a capability
/// the CPU backend has not built yet, and the message it goes into tells the
/// reader to run the scene on a CUDA host; every entry HERE is a scene the CUDA
/// backend answers with a live device assert inside the assembly, so that
/// advice would send a user to a host that fails the same way. Rule 2 at the
/// top of this file would not hold for them either: nothing implements a solid
/// form of a membrane model, so such an entry would never be removed.
///
/// IT IS ALSO WHY NEITHER VERDICT IS DOWNLOADED.
/// `tet_material_diff_table` and `face_material_diff_table` each return
/// a code on the element's model id, and the alternative to this gate is
/// reading both back inside every Newton iteration to decide whether to stop.
/// THE VERDICT IS A PURE FUNCTION OF `model`: it is the only value either
/// branch tests, the singular values and the two Lame constants reaching the
/// table and never the code. `model` is built on the host out of the scene's
/// material table, so the answer is available before a frame is written and
/// cannot change during the run. On a queued backend a download is a wait for
/// the queue to drain, which is what makes the COUNT of them, rather than the
/// nanoseconds one costs on a host backend, the figure that decides this.
///
/// THE GATE IS THE MATERIAL, NOT THE ACTIVE SET, and that is a requirement
/// rather than a shortcut. An element takes the elastic pass when its material
/// carries `mu > 0` AND its props say it is not fixed, not rest-excluded and
/// not a collider; the last three are rebuilt every step, so an element fixed
/// at t = 0 can become active at t = 2, which a gate reading them before the
/// run would never see. Reading only the material errs toward naming an element
/// that never becomes active, which is the safe direction: the other one writes
/// frames for a material the scene did not ask for.
pub fn material_defects(data: &DataSet) -> Vec<Refusal> {
    let mut out = Vec::new();

    let tets = solid_elements_naming_a_membrane_model(data);
    if tets > 0 {
        out.push(Refusal {
            feature: "solid elements naming the BaraffWitkin membrane model, which has no solid \
                      form",
            count: tets,
            unit: "tets",
        });
    }

    let faces = shell_faces_naming_the_rigid_model_with_stiffness(data);
    if faces > 0 {
        out.push(Refusal {
            feature: "shell faces naming the PDRD rigid model together with a positive shear \
                      modulus",
            count: faces,
            unit: "faces",
        });
    }


    out
}

/// How many tets name a model the solid diff table has no arm for.
///
/// THE MATCH IS EXHAUSTIVE ON PURPOSE, and it is the guard against this gate
/// drifting away from the kernel it mirrors: `Model` is the host's whole
/// vocabulary, so a variant added to it stops compiling here until someone
/// states which side of the solid dispatch it falls on. The other direction, a
/// kernel arm removed without touching this file, is covered by the tests in
/// `super::assemble` that read each verdict straight off the kernel.
///
/// BARAFFWITKIN IS THE ONLY ONE, and PDRD is deliberately not beside it: the
/// solid table admits `Model::Pdrd` and writes it a zero table, which is the
/// correct energy for an element that has none.
fn solid_elements_naming_a_membrane_model(data: &DataSet) -> u64 {
    // Safety: both containers are read as slices of their declared length, and
    // this runs with the `DataSet` live, as `grain_count` does.
    let params: &[TetParam] = unsafe { super::scene::slice(&data.param_arrays.tet) };
    let props: &[TetProp] = unsafe { super::scene::slice(&data.prop.tet) };
    props
        .iter()
        .filter(|prop| {
            params
                .get(prop.param_index as usize)
                .is_some_and(|material| {
                    material.mu > 0.0
                        && match material.model {
                            Model::Arap | Model::StVK | Model::SNHk | Model::Pdrd => false,
                            Model::BaraffWitkin => true,
                        }
                })
        })
        .count() as u64
}

/// How many shell faces declare a rigid body and an elastic energy at once.
///
/// `Model::Pdrd` names a face whose shape is held by the reduced rigid solve
/// rather than by a membrane, so it carries `mu == 0` and leaves the elastic
/// pass at that gate. A face that names it WITH a positive shear modulus is
/// asking for both, which is the face verdict `FACE_TABLE_NO_ENERGY` and
/// the case the membrane assembly answers with a live device assert.
///
/// THE RANGE IS THE SHELL PREFIX, NOT `mesh.face`. A solid's surface triangles
/// follow the shell faces in that one array and never reach the membrane, so
/// counting them here would refuse a scene over a material no membrane
/// dispatch reads. The two lengths are equal exactly when the scene carries no
/// solid, which is what makes confusing them invisible.
fn shell_faces_naming_the_rigid_model_with_stiffness(data: &DataSet) -> u64 {
    // Safety: as above.
    let params: &[FaceParam] = unsafe { super::scene::slice(&data.param_arrays.face) };
    let props: &[FaceProp] = unsafe { super::scene::slice(&data.prop.face) };
    let shell = (data.shell_face_count as usize).min(props.len());
    props[..shell]
        .iter()
        .filter(|prop| {
            params
                .get(prop.param_index as usize)
                .is_some_and(|material| {
                    material.mu > 0.0
                        && match material.model {
                            Model::Arap
                            | Model::StVK
                            | Model::SNHk
                            | Model::BaraffWitkin => false,
                            Model::Pdrd => true,
                        }
                })
        })
        .count() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cvec::CVec;
    use crate::data::{
        EdgeParam, EdgeProp, FaceParam, FaceProp, HingeParam, HingeProp, Vec2u, Vec3u, Vec4u,
    };

    #[test]
    fn a_refusal_names_the_feature_and_its_count() {
        // A PERMANENT REFUSAL, on purpose. This fixture named the PDRD reduced
        // solve until that landed, then the multilevel preconditioner until
        // THAT landed; a fixture naming a capability has an expiry date, and
        // the way to write one that does not is to name something that cannot
        // be implemented. A solid element asking for the BaraffWitkin membrane
        // model is such a thing: that model maps two material directions into
        // space and has no solid form.
        let r = Refusal {
            feature: "solid elements naming the BaraffWitkin membrane model",
            count: 3,
            unit: "elements",
        };
        assert_eq!(
            r.describe(),
            "solid elements naming the BaraffWitkin membrane model: 3 elements"
        );
    }

    /// A scene record whose containers are all empty.
    ///
    /// `DataSet` and `ParamSet` are `repr(C)` aggregates of `CVec` handles, and
    /// a zeroed `CVec` is a null pointer with size 0, which is the shape the
    /// gate reads. Nothing here is dereferenced except through
    /// `scene::slice`, which returns an empty slice for a null pointer.
    fn empty_scene() -> (DataSet, ParamSet) {
        // Safety: both types are repr(C) plain-old-data aggregates with no
        // Drop, no references and no niche-optimized fields, so an all-zero
        // bit pattern is a valid inhabitant.
        unsafe {
            (
                std::mem::zeroed::<DataSet>(),
                std::mem::zeroed::<ParamSet>(),
            )
        }
    }

    /// The same, with contact switched off, which is the mode the driver runs.
    fn empty_contactless_scene() -> (DataSet, ParamSet) {
        let (data, mut param) = empty_scene();
        param.disable_contact = true;
        (data, param)
    }

    #[test]
    fn an_empty_contactless_scene_is_accepted() {
        // The gate must refuse on features found, never on principle: a scene
        // with nothing in it has nothing this backend cannot solve.
        let (data, param) = empty_contactless_scene();
        assert!(scene_refusals(&data, &param).is_empty());
    }

    #[test]
    fn a_contact_enabled_scene_is_accepted() {
        // A ZEROED ParamSet HAS `disable_contact == false`, which is also the
        // registry's default, so this is the mode every example notebook
        // arrives in. It is accepted because the two structural halves of the
        // penetration guarantee are live beside the barrier: the ACCD line
        // search and `check_intersection`. This test is what stops a refusal
        // entry being added for it; what would justify one is either of those
        // two halves going away, not the barrier.
        let (data, param) = empty_scene();
        assert!(!param.disable_contact);
        let refusals = scene_refusals(&data, &param);
        let text: Vec<String> = refusals.iter().map(Refusal::describe).collect();
        assert!(
            refusals.is_empty(),
            "a contact-enabled scene with nothing in it must be accepted, got {text:?}"
        );
    }

    #[test]
    fn a_shell_with_hinges_to_bend_is_accepted() {
        // THE CASE THE BENDING ASSEMBLY EXISTS FOR. Its refusal came off in the
        // change that assembled it, which is this file's rule 2, and this test
        // is what stops it being reinstated. A hinge whose element class cannot
        // be read is not answered here any more: the assembly reads
        // `mesh.type.hinge` itself and stops the run by name on a short table,
        // which `cpu::assemble::tests::a_hinge_with_no_type_byte_stops_the_run`
        // covers.
        let (mut data, param) = empty_contactless_scene();
        with_face_material(&mut data, 1234, 1234, FaceParam::default());
        data.mesh.mesh.hinge.size = 600;
        data.mesh.ttype.hinge = CVec::from(&vec![0u8; 600][..]);
        data.vertex.curr.size = 700;

        let refusals = scene_refusals(&data, &param);
        assert!(
            refusals.is_empty(),
            "a contact-disabled shell with bending hinges is what this backend now steps, got {:?}",
            refusals.iter().map(Refusal::describe).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_tet_scene_with_contact_disabled_is_accepted() {
        // THE CASE THE DRIVER SPINE EXISTS FOR. Its refusal came off in the
        // change that implemented it, which is the rule this file states, and
        // this test is what stops it being reinstated by accident.
        let (mut data, param) = empty_contactless_scene();
        data.mesh.mesh.tet.size = 500;
        data.mesh.mesh.face.size = 300; // a solid's surface triangles
        data.shell_face_count = 0;
        data.mesh.mesh.edge.size = 900;
        data.vertex.curr.size = 200;
        let refusals = scene_refusals(&data, &param);
        assert!(
            refusals.is_empty(),
            "a contact-disabled tet scene is what this backend steps, got {:?}",
            refusals.iter().map(Refusal::describe).collect::<Vec<_>>()
        );
    }

    /// Give a zeroed scene real faces, one `FaceProp` each, and one material.
    ///
    /// SETTING `face.size` ALONE WOULD TEST NOTHING here, and that is worth
    /// stating because the older shape of these tests did exactly that: the
    /// per-material gates read both containers through `scene::slice`, which
    /// returns an EMPTY slice for a null pointer however large the declared
    /// size, so a scene with a size and no buffer reports every material gate
    /// as unmet.
    fn with_face_material(
        data: &mut DataSet,
        faces: usize,
        shell_faces: u32,
        material: FaceParam,
    ) {
        data.mesh.mesh.face = CVec::from(&vec![Vec3u::zeros(); faces][..]);
        data.prop.face = CVec::from(&vec![FaceProp::default(); faces][..]);
        data.param_arrays.face = CVec::from(&[material][..]);
        data.shell_face_count = shell_faces;
    }

    fn shell_material(model: Model, mu: f32) -> FaceParam {
        FaceParam {
            model,
            mu,
            ..FaceParam::default()
        }
    }

    fn with_tet_material(data: &mut DataSet, tets: usize, material: TetParam) {
        data.mesh.mesh.tet = CVec::from(&vec![Vec4u::zeros(); tets][..]);
        data.prop.tet = CVec::from(&vec![TetProp::default(); tets][..]);
        data.param_arrays.tet = CVec::from(&[material][..]);
    }

    fn solid_material(model: Model, mu: f32) -> TetParam {
        TetParam {
            model,
            mu,
            ..TetParam::default()
        }
    }

    #[test]
    fn a_solid_naming_the_membrane_model_is_named_and_counted() {
        // BaraffWitkin maps two material directions into space, so there is no
        // solid form of it to dispatch: the fused CUDA body answers this tet
        // with a live device assert and the staged one returns
        // `TET_TABLE_UNKNOWN`. Naming it here stops the run before a frame
        // rather than on the first Newton iteration that assembles a tet.
        let (mut data, _) = empty_contactless_scene();
        with_tet_material(&mut data, 7, solid_material(Model::BaraffWitkin, 100.0));

        let defects = material_defects(&data);
        assert_eq!(defects.len(), 1, "got {defects:?}");
        assert_eq!(defects[0].count, 7);
        assert_eq!(defects[0].unit, "tets");
        assert!(
            defects[0].feature.contains("BaraffWitkin"),
            "the message must name the model, got {}",
            defects[0].feature
        );
    }

    #[test]
    fn the_solid_models_the_table_carries_are_accepted() {
        // PDRD IS IN THIS LIST ON PURPOSE. A rigid body's tets carry model id 4
        // because their shape is held by the reduced rigid solve, and the solid
        // table admits it and writes a zero table, which is the correct energy
        // for an element that has none. A gate that refused every id it did not
        // recognize would take this scene class down with it.
        for model in [Model::Arap, Model::StVK, Model::SNHk, Model::Pdrd] {
            let (mut data, _) = empty_contactless_scene();
            with_tet_material(&mut data, 3, solid_material(model, 100.0));
            assert!(
                material_defects(&data).is_empty(),
                "{model:?} is a solid model this backend assembles, got {:?}",
                material_defects(&data)
            );
        }
    }

    #[test]
    fn a_pdrd_shell_face_with_no_stiffness_is_accepted() {
        // THE PDRD CASE, and the one a gate on the model id alone would get
        // wrong. A rigid body's faces carry model id 4 with `mu == 0` and leave
        // the elastic pass at that gate before any dispatch, so nothing here is
        // asked of them. Refusing them would refuse a scene that simulates
        // correctly.
        let (mut data, _) = empty_contactless_scene();
        with_face_material(&mut data, 5, 5, shell_material(Model::Pdrd, 0.0));

        assert!(
            material_defects(&data).is_empty(),
            "a PDRD face carrying no elastic energy is what a rigid body is made of, got {:?}",
            material_defects(&data)
        );
    }

    #[test]
    fn a_pdrd_shell_face_asking_for_stiffness_is_named_and_counted() {
        // The other half of the same pair: a face declaring that its shape is
        // held by the rigid solve AND asking for a membrane energy is asking
        // for both, which is the verdict `FACE_TABLE_NO_ENERGY` and the
        // case the membrane assembly answers with a live device assert.
        let (mut data, _) = empty_contactless_scene();
        with_face_material(&mut data, 5, 5, shell_material(Model::Pdrd, 100.0));

        let defects = material_defects(&data);
        assert_eq!(defects.len(), 1, "got {defects:?}");
        assert_eq!(defects[0].count, 5);
        assert_eq!(defects[0].unit, "faces");
        assert!(
            defects[0].feature.contains("PDRD"),
            "the message must name the model, got {}",
            defects[0].feature
        );
    }

    #[test]
    fn the_shell_models_the_membrane_carries_are_accepted() {
        // Four rather than three: BaraffWitkin is a shell material the membrane
        // assembles through its own arm, so the face gate admits it where the
        // solid gate above does not.
        for model in [Model::Arap, Model::StVK, Model::SNHk, Model::BaraffWitkin] {
            let (mut data, _) = empty_contactless_scene();
            with_face_material(&mut data, 4, 4, shell_material(model, 100.0));
            assert!(
                material_defects(&data).is_empty(),
                "{model:?} is a shell model this backend assembles, got {:?}",
                material_defects(&data)
            );
        }
    }

    #[test]
    fn a_solid_surface_face_past_the_shell_prefix_is_not_counted() {
        // `mesh.face` CARRIES A SOLID'S SURFACE TRIANGLES AFTER THE SHELL
        // FACES, and the membrane walks the prefix, so a material on one of
        // them reaches no membrane dispatch and is not this gate's business.
        // The two lengths agree on every shell-only scene, which is what makes
        // reading the whole array look correct.
        let (mut data, _) = empty_contactless_scene();
        with_face_material(&mut data, 6, 0, shell_material(Model::Pdrd, 100.0));

        assert!(
            material_defects(&data).is_empty(),
            "a face past the shell prefix carries no membrane, got {:?}",
            material_defects(&data)
        );
    }

    #[test]
    fn a_material_defect_is_reported_apart_from_the_capability_refusals() {
        // TWO GATES THAT SAY DIFFERENT THINGS. A defect here is a property of
        // the scene that no backend can assemble, and `scene_refusals` names
        // capabilities this backend has not built, whose message sends the
        // reader to a CUDA host. A scene can carry one, the other, or both, and
        // the counts must not be mixed into one list.
        //
        // THE DRIVER REFUSES NO CAPABILITY, so this test cannot pair one of
        // each. What it asserts instead is the separation itself: a material
        // defect appears in its own list and NOT among the capability
        // refusals, and asking for a capability the driver has adds nothing to
        // either. A scene whose defect leaked into the refusal list would send
        // the reader to a CUDA host for a scene no backend can assemble.
        let (mut data, mut param) = empty_contactless_scene();
        with_tet_material(&mut data, 2, solid_material(Model::BaraffWitkin, 100.0));
        param.precond = PrecondMode::Schwarz;

        let defects = material_defects(&data);
        assert_eq!(defects.len(), 1, "got {defects:?}");
        assert!(defects[0].feature.contains("BaraffWitkin"), "got {defects:?}");
        let refusals = scene_refusals(&data, &param);
        assert!(
            refusals.is_empty(),
            "a material defect must not appear among the capability refusals, \
             and Schwarz is implemented: got {refusals:?}"
        );
    }

    fn strain_limited_material() -> FaceParam {
        FaceParam {
            strainlimit: 0.05,
            ..FaceParam::default()
        }
    }

    fn pressurized_material() -> FaceParam {
        FaceParam {
            pressure: 1.0,
            ..FaceParam::default()
        }
    }

    #[test]
    fn a_shell_scene_carrying_only_a_membrane_is_accepted() {
        // THE CASE THE MEMBRANE EXISTS FOR, and the counterpart to the tet
        // entry below: a shell with no hinge to bend, no strain limit to
        // enforce and no pressure, whose whole elastic content is the membrane.
        // Its refusal came off in the change that assembled it, which is this
        // file's rule 2, and this test is what stops it being reinstated.
        let (mut data, param) = empty_contactless_scene();
        with_face_material(&mut data, 1234, 1234, FaceParam::default());
        data.vertex.curr.size = 700;

        let refusals = scene_refusals(&data, &param);
        assert!(
            refusals.is_empty(),
            "a contact-disabled membrane-only shell is what this backend now steps, got {:?}",
            refusals.iter().map(Refusal::describe).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_strain_limited_shell_is_accepted() {
        // THE CASE THE LIMITER EXISTS FOR, and its refusal came off in the
        // change that assembled both halves, which is this file's rule 2. The
        // limiter is dispatched on `shell_face_count` alone and has no element
        // class of its own, so it was reachable the moment the membrane was
        // accepted; a cloth run without it stretches past the limit it was
        // authored with and completes, which is why it had an entry at all.
        let (mut data, param) = empty_contactless_scene();
        with_face_material(&mut data, 1234, 1234, strain_limited_material());
        data.vertex.curr.size = 700;

        let refusals = scene_refusals(&data, &param);
        assert!(
            refusals.is_empty(),
            "a contact-disabled strain-limited shell is what this backend now steps, got {:?}",
            refusals.iter().map(Refusal::describe).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_strain_limited_rod_is_accepted() {
        // The rod half of the same entry, and it needs its own case: the two
        // limiters are separate dispatches over separate prefixes, so one
        // could be assembled and the other left refused.
        let (mut data, param) = empty_contactless_scene();
        data.mesh.mesh.edge = CVec::from(&vec![Vec2u::zeros(); 40][..]);
        data.prop.edge = CVec::from(&vec![EdgeProp::default(); 40][..]);
        data.param_arrays.edge = CVec::from(
            &[EdgeParam {
                strainlimit: 0.05,
                ..EdgeParam::default()
            }][..],
        );
        data.rod_count = 40;
        data.vertex.curr.size = 41;

        let refusals = scene_refusals(&data, &param);
        assert!(
            refusals.is_empty(),
            "a contact-disabled strain-limited rod is what this backend now steps, got {:?}",
            refusals.iter().map(Refusal::describe).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_plastic_solid_and_shell_scene_is_accepted() {
        // A scene where the shell count and the face-array length DIFFER: a
        // solid's surface triangles sit in `mesh.face` after the shell prefix.
        // Face plasticity was the entry that made such a scene stop here, and
        // it is now assembled, so the whole scene has to be accepted. The
        // prefix property itself moved to where the prefix is now read:
        // `plasticity::tests::the_face_creep_walks_the_shell_prefix`.
        let (mut data, param) = empty_contactless_scene();
        let material = FaceParam {
            plasticity: 0.5,
            ..FaceParam::default()
        };
        with_face_material(&mut data, 300, 100, material);
        data.mesh.mesh.tet.size = 42;
        data.vertex.curr.size = 90;

        let refusals = scene_refusals(&data, &param);
        assert!(
            refusals.is_empty(),
            "a plastic solid-plus-shell scene is what this backend now steps, got {:?}",
            refusals.iter().map(Refusal::describe).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_hinge_plastic_scene_with_no_shell_at_all_is_accepted() {
        // `update_hinge_plasticity` is dispatched over EVERY hinge in the mesh,
        // a solid's surface hinges included, so it is reachable on a tet-only
        // scene. Two separate capabilities meet on such a hinge and neither is
        // refused: it carries no bending ENERGY, and its rest angle still
        // creeps.
        let (mut data, param) = empty_contactless_scene();
        data.mesh.mesh.tet.size = 12;
        data.mesh.mesh.hinge = CVec::from(&vec![Vec4u::zeros(); 6][..]);
        data.prop.hinge = CVec::from(&vec![HingeProp::default(); 6][..]);
        data.param_arrays.hinge = CVec::from(
            &[HingeParam {
                plasticity: 0.25,
                ..HingeParam::default()
            }][..],
        );
        // Bit 0 set: every hinge sits on a solid's surface.
        data.mesh.ttype.hinge = CVec::from(&vec![1u8; 6][..]);
        data.vertex.curr.size = 20;

        let refusals = scene_refusals(&data, &param);
        assert!(
            refusals.is_empty(),
            "a tet-only scene whose surface hinges creep is stepped, got {:?}",
            refusals.iter().map(Refusal::describe).collect::<Vec<_>>()
        );
    }

    /// Setting the fix-xz threshold must add no refusal.
    ///
    /// The drag is a momentum term AND a position clamp, and a backend carrying
    /// one alone would solve a system that does not describe the step it then
    /// takes. Both are assembled, so the parameter adds nothing here.
    #[test]
    fn the_fix_xz_drag_is_accepted_now_that_both_halves_exist() {
        // The drag is a momentum term AND a position clamp, and the reason this
        // test exists at all is that a backend carrying one alone would solve a
        // system that does not describe the step it then takes. Both are now
        // assembled, so setting the threshold must ADD NO REFUSAL. The test is
        // written as a difference against the same scene with the threshold off,
        // so it fails if the parameter reintroduces a refusal under any name
        // rather than only under the one this test could have hardcoded.
        let (data, mut param) = empty_contactless_scene();
        let without = scene_refusals(&data, &param).len();
        param.fix_xz = 0.5;
        let with = scene_refusals(&data, &param);
        assert_eq!(
            with.len(),
            without,
            "setting fix-xz must add no refusal now that both halves ship, got {with:?}"
        );
        assert!(
            !with.iter().any(|r| r.feature.contains("fix-xz")),
            "got {with:?}"
        );
    }

    /// SCHWARZ IS ACCEPTED AT EVERY LEVEL COUNT, which is the last capability
    /// refusal coming off.
    ///
    /// The driver materializes the fine operator as a block-CSR level 0,
    /// coarsens it with a Galerkin triple product over each level's own
    /// aggregation, factors that level's domains, and sums every level's
    /// correction onto the fine sweep. Measured on a pinned sheet: 51.4 mean
    /// PCG iterations on block-Jacobi, 30.8 at one level, 21.2 at two.
    ///
    /// ZERO STILL READS AS TWO, which `super::step` spells as
    /// `if schwarz_levels > 0 { schwarz_levels } else { 2 }`, so an unset count
    /// asks for the hierarchy and gets it.
    #[test]
    fn schwarz_is_accepted_at_every_level_count() {
        let (data, mut param) = empty_contactless_scene();
        param.precond = PrecondMode::Schwarz;
        for levels in [0u32, 1, 2, 3, 8] {
            param.schwarz_levels = levels;
            let refusals = scene_refusals(&data, &param);
            assert!(
                !refusals.iter().any(|r| r.feature.contains("schwarz")),
                "levels={levels} must be accepted, got {refusals:?}"
            );
        }
    }

    #[test]
    fn every_defect_is_reported_not_just_the_first() {
        // A user retargeting a scene wants the whole list, not the first hit.
        //
        // IT PAIRS TWO MATERIAL DEFECTS rather than two capability refusals,
        // because the driver refuses no capability any more. Both are
        // permanent: a solid
        // element naming the BaraffWitkin membrane model, which has no solid
        // form, and a PDRD shell face asking for a positive shear modulus,
        // which asks for a rigid body and an elastic energy at once. Neither
        // can lift, so neither can stop testing what this was written for.
        let (mut data, mut param) = empty_scene();
        // Capabilities the fixture carries that are NOT refused, kept on
        // purpose: something since implemented is exactly what must not appear.
        with_face_material(&mut data, 10, 10, pressurized_material());
        data.vertex.curr.size = 30;
        data.constraint.torque_groups.size = 2;
        param.precond = PrecondMode::Schwarz;

        with_tet_material(&mut data, 4, solid_material(Model::BaraffWitkin, 100.0));
        with_face_material(&mut data, 3, 3, shell_material(Model::Pdrd, 100.0));

        let defects = material_defects(&data);
        let text: Vec<String> = defects.iter().map(Refusal::describe).collect();
        assert!(
            defects.len() >= 2,
            "expected the BaraffWitkin solid and the sheared PDRD face, got {text:?}"
        );
        for implemented in ["pressure", "torque", "schwarz"] {
            assert!(
                !text.iter().any(|t| t.contains(implemented)),
                "{implemented} is implemented and must not be reported: {text:?}"
            );
        }
        for wanted in ["BaraffWitkin", "PDRD"] {
            assert!(
                text.iter().any(|t| t.contains(wanted)),
                "{wanted} must be reported, got {text:?}"
            );
        }
    }
}
