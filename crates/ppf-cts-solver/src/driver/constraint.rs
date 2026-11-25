// File: crates/ppf-cts-solver/src/driver/constraint.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The per-step pin rebuild: `update_constraint`.
//!
//! The host builds a fresh `Constraint` every step and hands it over here. Two
//! things follow from it, and the whole module is those two things:
//!
//!   1. `VertexProp::fix_index` and `pull_index` are rebuilt from the new pin
//!      set, 1-based, with 0 meaning "not pinned".
//!   2. `Face/Edge/Tet/HingeProp::fixed` is rebuilt as the AND over the
//!      element's vertices.
//!
//! THE SECOND ONE IS WHY THIS CANNOT BE DONE ONCE AT BUILD TIME. An element
//! whose vertices are all pinned carries no elastic or bending energy, and the
//! energy kernels skip it on that flag. When a pin reaches its `unpin_time` the
//! element has to become elastic again, and nothing else in the step says so.
//! An empty `update_constraint` therefore does not merely freeze the pins: it
//! also leaves every element that started fully pinned permanently inert, and
//! both failures produce a full run of plausible frames with nothing in the
//! output to say so.
//!
//! WHAT THIS DELIBERATELY DOES NOT DO, and the reason is not obvious: it does
//! NOT write a pin's prescribed position into `vertex.curr`. Production CUDA
//! does not, and neither does Metal, because `compute_target`
//! (`kernels/main/target.kernel.cpp`) seeds a fix-pinned vertex's Newton target
//! from `constraint.fix` DURING the step: writing the pin here as well would apply
//! the step's motion twice, and would corrupt the pinned vertex's incoming
//! velocity through the `prev` snapshot that goes with it.
//!
//! NOTHING FROM THE `Constraint` IS CACHED ACROSS A CALL, and it is the one
//! place direct addressing is not simply cheaper than mirroring: `backend.rs`
//! drops the previous `Constraint` and frees its `CVec` buffers as it assigns
//! the new one, so a pointer kept from last step addresses freed memory. A
//! later phase that needs the pin set while the step runs takes it the same way
//! this function does, from the argument, on every call.
//!
//! NO PLASTIC READBACK IS NEEDED HERE, and its absence is worth stating because
//! both other backends carry one. They mirror `prop.vertex` and `prop.hinge` to
//! a device, so before overwriting the host copy they must pull back the rest
//! angles the plasticity kernels crept in place. This backend has one copy of
//! each array, which is the array the creep would have written, so there is
//! nothing to pull back and nothing that can be clobbered.

use crate::data::Constraint;

use super::scene::{Fatal, FatalResult, SceneView};

/// Rebuild the pin indices and the per-element `fixed` flags.
///
/// # Safety
/// `view` must address a live `DataSet` and `constraint` a live `Constraint`,
/// with no other reference to either alive for the duration.
pub unsafe fn rebuild(view: &SceneView, constraint: &Constraint) -> FatalResult<()> {
    let vertex_props = view.vertex_props_mut();
    let vertex_count = vertex_props.len();

    let fix = super::scene::slice(&constraint.fix);
    let pull = super::scene::slice(&constraint.pull);

    // VALIDATE FIRST, WRITE SECOND. Both other backends validate inside the
    // write loop and stop the process where they find the problem, which is
    // fine for them because the process is going away. Here the check is
    // hoisted so a rejected constraint leaves the scene EXACTLY as it was: the
    // failure is then testable without a process to kill, and a caller that
    // somehow survives it has not been handed a half-rebuilt pin set.
    for (i, pin) in fix.iter().enumerate() {
        if pin.index as usize >= vertex_count {
            return Err(Fatal::invariant(format!(
                "constraint.fix[{i}] names vertex {} but the scene has {vertex_count} vertices",
                pin.index
            )));
        }
    }
    for (i, pin) in pull.iter().enumerate() {
        if pin.index as usize >= vertex_count {
            return Err(Fatal::invariant(format!(
                "constraint.pull[{i}] names vertex {} but the scene has {vertex_count} vertices",
                pin.index
            )));
        }
    }

    // A FIX pin on a vertex inside a PDRD rigid body is the one penalty pin
    // left in the solver, and a backend that assembles no pin barrier would
    // hold such an anchor with nothing at all. It is not re-checked here
    // because `refusal.rs` refuses any scene carrying a PDRD body at
    // `initialize()`, by name and by count, which is strictly earlier and
    // strictly louder. That entry and this comment come off together.

    for prop in vertex_props.iter_mut() {
        prop.fix_index = 0;
        prop.pull_index = 0;
    }
    for (i, pin) in fix.iter().enumerate() {
        vertex_props[pin.index as usize].fix_index = i as u32 + 1;
    }
    for (i, pin) in pull.iter().enumerate() {
        vertex_props[pin.index as usize].pull_index = i as u32 + 1;
    }

    // Re-borrowed read-only so the element loops below can read the pin state
    // while they write their own arrays.
    let pinned: &[crate::data::VertexProp] = vertex_props;
    let is_fixed = |vi: u32| -> bool { pinned[vi as usize].fix_index > 0 };

    // Each element array is walked over its PROP length, which is what both
    // other backends do, reading the connectivity array at the same index. That
    // is an unchecked read on both of them; here the two lengths are compared
    // once, so a mesh and a prop array that disagree stop the run instead of
    // reading past the end of the connectivity.
    let faces = view.faces();
    let face_props = view.face_props_mut();
    check_pairing("face", face_props.len(), faces.len())?;
    for (i, prop) in face_props.iter_mut().enumerate() {
        let f = faces[i];
        check_element("face", i, &[f[0], f[1], f[2]], vertex_count)?;
        prop.fixed = is_fixed(f[0]) && is_fixed(f[1]) && is_fixed(f[2]);
    }

    let edges = view.edges();
    let edge_props = view.edge_props_mut();
    check_pairing("edge", edge_props.len(), edges.len())?;
    for (i, prop) in edge_props.iter_mut().enumerate() {
        let e = edges[i];
        check_element("edge", i, &[e[0], e[1]], vertex_count)?;
        prop.fixed = is_fixed(e[0]) && is_fixed(e[1]);
    }

    let tets = view.tets();
    let tet_props = view.tet_props_mut();
    check_pairing("tet", tet_props.len(), tets.len())?;
    for (i, prop) in tet_props.iter_mut().enumerate() {
        let t = tets[i];
        check_element("tet", i, &[t[0], t[1], t[2], t[3]], vertex_count)?;
        prop.fixed = is_fixed(t[0]) && is_fixed(t[1]) && is_fixed(t[2]) && is_fixed(t[3]);
    }

    let hinges = view.hinges();
    let hinge_props = view.hinge_props_mut();
    check_pairing("hinge", hinge_props.len(), hinges.len())?;
    for (i, prop) in hinge_props.iter_mut().enumerate() {
        let h = hinges[i];
        check_element("hinge", i, &[h[0], h[1], h[2], h[3]], vertex_count)?;
        prop.fixed = is_fixed(h[0]) && is_fixed(h[1]) && is_fixed(h[2]) && is_fixed(h[3]);
    }

    Ok(())
}

/// The pin set the last [`rebuild`] was run on: which vertex each fix pin and
/// each pull pin names, in the constraint's own order.
///
/// EVERYTHING [`rebuild`] WRITES IS A FUNCTION OF THESE TWO LISTS and of the
/// mesh's connectivity, which does not change during a run: it clears every
/// pin index, assigns them in list order, and recomputes every `fixed` flag
/// from the fix indices. It reads no position, no weight and nothing a
/// previous call left. So a call whose two lists equal the remembered ones
/// would write, field for field, what the arrays already hold, and the walk
/// over every vertex, face, edge, tetrahedron and hinge can be skipped.
///
/// Measured on `trapped` (135,396 vertices, no pin at all): the rebuild and
/// the two re-stage comparisons behind it were 8.5 ms a step, 7.9 s of a
/// 76 s run of 929 steps.
///
/// THE REMEMBERED LISTS ARE COPIES. `backend.rs` frees the previous
/// `Constraint` as it assigns the next, which is why nothing else here keeps
/// a pointer into one.
///
#[derive(Default)]
pub struct PinSet {
    fix: Vec<u32>,
    pull: Vec<u32>,
    /// False until a rebuild has run, so the first call never skips.
    known: bool,
    /// How many times the remembered pins have been replaced, for whoever
    /// stages something that depends on which vertices are pinned.
    generation: u64,
}

impl PinSet {
    /// Whether `constraint` names the pins the last rebuild was run on.
    pub fn matches(&self, fix: &[crate::data::FixPair], pull: &[crate::data::PullPair]) -> bool {
        self.known
            && self.fix.len() == fix.len()
            && self.pull.len() == pull.len()
            && self.fix.iter().zip(fix).all(|(seen, pin)| *seen == pin.index)
            && self.pull.iter().zip(pull).all(|(seen, pin)| *seen == pin.index)
    }

    /// Record the pins a rebuild has just been run on.
    pub fn remember(&mut self, fix: &[crate::data::FixPair], pull: &[crate::data::PullPair]) {
        self.fix.clear();
        self.fix.extend(fix.iter().map(|pin| pin.index));
        self.pull.clear();
        self.pull.extend(pull.iter().map(|pin| pin.index));
        self.known = true;
        self.generation = self.generation.wrapping_add(1);
    }

    /// Changes whenever [`Self::remember`] records a new pin set.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Whether a skipped rebuild is checked instead of trusted, from
/// `PPF_VERIFY_PIN_SKIP`.
///
/// THE SKIP RESTS ON A CLAIM ABOUT THE WHOLE TREE that no type enforces:
/// nothing but [`rebuild`] writes the pin indices and the `fixed` flags once a
/// run has started. A violation would be silent, a step simulated against a
/// stale pin set, so the knob turns the claim into a check: every skipped step
/// runs the rebuild anyway and stops the run if it changed anything.
pub fn verify_pin_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PPF_VERIFY_PIN_SKIP").is_ok_and(|v| v != "0"))
}

/// Everything [`rebuild`] writes, read back.
unsafe fn rebuilt_fields(view: &SceneView) -> (Vec<(u32, u32)>, [Vec<bool>; 4]) {
    (
        view.vertex_props_mut().iter().map(|p| (p.fix_index, p.pull_index)).collect(),
        [
            view.face_props_mut().iter().map(|p| p.fixed).collect(),
            view.edge_props_mut().iter().map(|p| p.fixed).collect(),
            view.tet_props_mut().iter().map(|p| p.fixed).collect(),
            view.hinge_props_mut().iter().map(|p| p.fixed).collect(),
        ],
    )
}

/// Run the rebuild a step skipped and report whether it changed anything.
///
/// # Safety
/// As [`rebuild`].
pub unsafe fn verify_skipped(view: &SceneView, constraint: &Constraint) -> FatalResult<()> {
    let before = rebuilt_fields(view);
    rebuild(view, constraint)?;
    let after = rebuilt_fields(view);
    if before.0 != after.0 {
        return Err(Fatal::invariant(
            "PPF_VERIFY_PIN_SKIP: a skipped pin rebuild would have changed a vertex's fix or              pull index, so something other than the rebuild wrote them since the last one",
        ));
    }
    for (kind, (a, b)) in ["face", "edge", "tet", "hinge"].iter().zip(before.1.iter().zip(after.1.iter())) {
        if a != b {
            return Err(Fatal::invariant(format!(
                "PPF_VERIFY_PIN_SKIP: a skipped pin rebuild would have changed a {kind}'s fixed                  flag, so something other than the rebuild wrote it since the last one"
            )));
        }
    }
    Ok(())
}

/// The prop array may not be longer than the connectivity it indexes.
fn check_pairing(kind: &str, props: usize, connectivity: usize) -> FatalResult<()> {
    if props > connectivity {
        return Err(Fatal::invariant(format!(
            "the scene carries {props} {kind} props against {connectivity} {kind} elements, so \
             rebuilding the fixed flags would read past the end of the connectivity"
        )));
    }
    Ok(())
}

/// Every vertex an element names must exist.
fn check_element(
    kind: &str,
    index: usize,
    vertices: &[u32],
    vertex_count: usize,
) -> FatalResult<()> {
    for vi in vertices {
        if *vi as usize >= vertex_count {
            return Err(Fatal::device_assert(format!(
                "{kind} {index} names vertex {vi} but the scene has {vertex_count} vertices"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::test_scene::{constraint, fix_pin, pull_pin, TestScene};
    use crate::data::{Vec2u, Vec3u, Vec4u};
    use ppf_cts_formats::status::error_code;

    /// The skip's whole premise: the lists a rebuild ran on are recognized,
    /// and nothing else is. A pin set never rebuilt matches nothing, so the
    /// first step always rebuilds.
    #[test]
    fn a_pin_set_matches_only_the_lists_it_remembered() {
        let mut seen = PinSet::default();
        let (fix, pull) = ([fix_pin(3), fix_pin(1)], [pull_pin(2)]);
        assert!(!seen.matches(&fix, &pull), "a pin set no rebuild has run on matched");
        assert!(!PinSet::default().matches(&[], &[]), "the empty set matched before any rebuild");
        seen.remember(&fix, &pull);
        assert!(seen.matches(&fix, &pull));
        // A pin that moves, changes weight or turns kinematic names the same
        // vertex, and the rebuild reads nothing else.
        let mut moved = [fix_pin(3), fix_pin(1)];
        moved[0].kinematic = true;
        moved[0].position = crate::driver::test_scene::position(1.0, 2.0, 3.0);
        assert!(seen.matches(&moved, &pull));
        // Another vertex, another order, one pin fewer, one pin more, and a pin
        // changing kind are each a different set.
        assert!(!seen.matches(&[fix_pin(3), fix_pin(0)], &pull));
        assert!(!seen.matches(&[fix_pin(1), fix_pin(3)], &pull));
        assert!(!seen.matches(&[fix_pin(3)], &pull));
        assert!(!seen.matches(&[fix_pin(3), fix_pin(1), fix_pin(0)], &pull));
        assert!(!seen.matches(&[fix_pin(3)], &[pull_pin(2), pull_pin(1)]));
        assert!(!seen.matches(&fix, &[]));
    }

    /// What the skip relies on, checked against the rebuild itself: a second
    /// rebuild on the same lists writes what the first left, whatever stood in
    /// the arrays before, and a rebuild on other lists does not.
    #[test]
    fn a_rebuild_on_the_same_pins_writes_what_is_already_there() {
        let scene = TestScene::new(4)
            .with_faces(&[Vec3u::new(0, 1, 2), Vec3u::new(1, 2, 3)])
            .with_edges(&[Vec2u::new(0, 1), Vec2u::new(2, 3)]);
        let view = scene.view();
        let read = |view: &SceneView| unsafe {
            (
                view.vertex_props_mut().iter().map(|p| (p.fix_index, p.pull_index)).collect::<Vec<_>>(),
                view.face_props_mut().iter().map(|p| p.fixed).collect::<Vec<_>>(),
                view.edge_props_mut().iter().map(|p| p.fixed).collect::<Vec<_>>(),
            )
        };
        let pins = constraint(&[fix_pin(0), fix_pin(1), fix_pin(2)], &[pull_pin(3)]);
        unsafe { rebuild(&view, &pins) }.expect("the rebuild runs");
        let first = read(&view);
        assert_eq!(first.1, vec![true, false]);
        // Scribble over every output, then rebuild on the same lists.
        unsafe {
            for prop in view.vertex_props_mut().iter_mut() {
                prop.fix_index = 9;
                prop.pull_index = 9;
            }
            for prop in view.face_props_mut().iter_mut() {
                prop.fixed = !prop.fixed;
            }
            for prop in view.edge_props_mut().iter_mut() {
                prop.fixed = !prop.fixed;
            }
            rebuild(&view, &pins).expect("the rebuild runs");
        }
        assert_eq!(read(&view), first, "a rebuild read something a previous call left");
        let fewer = constraint(&[fix_pin(0), fix_pin(1)], &[pull_pin(3)]);
        unsafe { rebuild(&view, &fewer) }.expect("the rebuild runs");
        assert_ne!(read(&view), first);
    }

    /// The check behind `PPF_VERIFY_PIN_SKIP` passes on arrays a rebuild left
    /// and refuses arrays something else wrote since: one index, one flag.
    #[test]
    fn the_skip_check_refuses_fields_written_since_the_last_rebuild() {
        let scene = TestScene::new(4)
            .with_faces(&[Vec3u::new(0, 1, 2)])
            .with_edges(&[Vec2u::new(0, 1)])
            .with_tets(&[Vec4u::new(0, 1, 2, 3)])
            .with_hinges(&[Vec4u::new(0, 1, 2, 3)]);
        let view = scene.view();
        let pins = constraint(&[fix_pin(0), fix_pin(1), fix_pin(2)], &[pull_pin(3)]);
        unsafe {
            rebuild(&view, &pins).expect("the rebuild runs");
            verify_skipped(&view, &pins).expect("nothing was written since the rebuild");
            view.vertex_props_mut()[3].pull_index = 0;
            assert!(verify_skipped(&view, &pins).is_err(), "a changed pull index passed");
            // The refused check has rebuilt the arrays, so they pass again.
            verify_skipped(&view, &pins).expect("the check restores what it compares");
            view.tet_props_mut()[0].fixed = true;
            assert!(verify_skipped(&view, &pins).is_err(), "a changed tet flag passed");
            view.face_props_mut()[0].fixed = false;
            assert!(verify_skipped(&view, &pins).is_err(), "a changed face flag passed");
        }
    }

    /// THE DEFECT AN EMPTY `update_constraint` HIDES, and it is not the pins.
    ///
    /// An element whose vertices are all pinned is skipped by the energy
    /// kernels. When one of those pins reaches its `unpin_time` the host stops
    /// sending it, and the element has to become elastic again. Nothing else in
    /// the step says so, so a backend that does not rebuild this flag leaves the
    /// element permanently inert: it never stretches, never bends and never
    /// contributes a force, and the run completes with plausible frames.
    #[test]
    fn a_pin_reaching_its_unpin_time_makes_its_element_elastic_again() {
        let scene = TestScene::new(4)
            .with_faces(&[Vec3u::new(0, 1, 2)])
            .with_edges(&[Vec2u::new(0, 1)])
            .with_tets(&[Vec4u::new(0, 1, 2, 3)])
            .with_hinges(&[Vec4u::new(0, 1, 2, 3)]);
        let view = scene.view();

        // Every vertex pinned: the face, edge, tet and hinge are all inert.
        let all = constraint(&[fix_pin(0), fix_pin(1), fix_pin(2), fix_pin(3)], &[]);
        unsafe { rebuild(&view, &all) }.expect("a well-formed constraint must be accepted");
        assert!(scene.face_props()[0].fixed);
        assert!(scene.edge_props()[0].fixed);
        assert!(scene.tet_props()[0].fixed);
        assert!(scene.hinge_props()[0].fixed);
        assert_eq!(
            scene
                .vertex_props()
                .iter()
                .map(|p| p.fix_index)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4],
            "fix_index is 1-based, with 0 reserved for an unpinned vertex"
        );

        // Vertex 1's pin expires. Every element that touches it is elastic
        // again, and the one that does not (the edge here still does) follows
        // its own vertices.
        let fewer = constraint(&[fix_pin(0), fix_pin(2), fix_pin(3)], &[]);
        unsafe { rebuild(&view, &fewer) }.expect("a well-formed constraint must be accepted");
        assert_eq!(
            scene.vertex_props()[1].fix_index,
            0,
            "the expired pin is gone"
        );
        assert_eq!(
            scene
                .vertex_props()
                .iter()
                .map(|p| p.fix_index)
                .collect::<Vec<_>>(),
            vec![1, 0, 2, 3],
            "the surviving pins are renumbered against the NEW list, not the old one"
        );
        assert!(
            !scene.face_props()[0].fixed,
            "the face must be elastic again"
        );
        assert!(!scene.edge_props()[0].fixed);
        assert!(!scene.tet_props()[0].fixed);
        assert!(!scene.hinge_props()[0].fixed);

        // An element none of whose vertices lost a pin stays inert.
        let solo = TestScene::new(3).with_faces(&[Vec3u::new(0, 1, 2)]);
        let solo_view = solo.view();
        let three = constraint(&[fix_pin(0), fix_pin(1), fix_pin(2)], &[]);
        unsafe { rebuild(&solo_view, &three) }.expect("accepted");
        assert!(solo.face_props()[0].fixed);
    }

    /// A PULL pin does not make an element inert.
    ///
    /// `fixed` is read off `fix_index` alone, because a pull pin is a soft
    /// spring that yields to elasticity and contact: an element held only by
    /// pull pins still carries its full elastic energy.
    #[test]
    fn a_pull_pin_leaves_its_element_elastic() {
        let scene = TestScene::new(3).with_faces(&[Vec3u::new(0, 1, 2)]);
        let view = scene.view();
        let pulled = constraint(&[], &[pull_pin(0), pull_pin(1), pull_pin(2)]);
        unsafe { rebuild(&view, &pulled) }.expect("accepted");
        assert_eq!(
            scene
                .vertex_props()
                .iter()
                .map(|p| p.pull_index)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(
            !scene.face_props()[0].fixed,
            "a pull pin is compliant, so it must not suppress the elastic term"
        );
    }

    /// An out-of-range constraint index FATALS RATHER THAN WRITING.
    ///
    /// Both other backends stop the process where they find it, mid-rebuild,
    /// which is fine because the process is going away. Here the whole pin set
    /// is validated first, so the rejection is testable and the scene is left
    /// exactly as it was rather than half rebuilt. The alternative is what CUDA
    /// does: `host_vprop[constraint->fix[i].index]` with no check at all, which
    /// is a wild write into whatever follows the prop array.
    #[test]
    fn an_out_of_range_constraint_index_fatals_rather_than_writing() {
        let mut scene = TestScene::new(3).with_faces(&[Vec3u::new(0, 1, 2)]);
        // A recognizable prior state, so "did not write" is checkable rather
        // than indistinguishable from "wrote a zero".
        for (i, prop) in scene.vertex_props_mut().iter_mut().enumerate() {
            prop.fix_index = 7 + i as u32;
            prop.pull_index = 11 + i as u32;
        }
        let view = scene.view();

        let bad = constraint(&[fix_pin(0), fix_pin(99)], &[]);
        let fatal = unsafe { rebuild(&view, &bad) }.expect_err("vertex 99 does not exist");
        assert_eq!(fatal.code, error_code::SOLVER_INVARIANT);
        assert!(
            fatal.detail.contains("constraint.fix[1]")
                && fatal.detail.contains("99")
                && fatal.detail.contains("3 vertices"),
            "the report must name the entry, the vertex and the scene size, got {:?}",
            fatal.detail
        );
        assert_eq!(
            scene
                .vertex_props()
                .iter()
                .map(|p| (p.fix_index, p.pull_index))
                .collect::<Vec<_>>(),
            vec![(7, 11), (8, 12), (9, 13)],
            "a rejected constraint must leave the scene untouched"
        );

        // The pull list is validated on the same rule.
        let bad_pull = constraint(&[], &[pull_pin(4)]);
        let fatal = unsafe { rebuild(&view, &bad_pull) }.expect_err("vertex 4 does not exist");
        assert_eq!(fatal.code, error_code::SOLVER_INVARIANT);
        assert!(
            fatal.detail.contains("constraint.pull[0]"),
            "{:?}",
            fatal.detail
        );
    }

    /// An element naming a vertex outside the scene is reported as the index a
    /// kernel was about to dereference, not as a scene-level invariant.
    #[test]
    fn an_element_naming_a_missing_vertex_is_reported_under_device_assert() {
        let scene = TestScene::new(2).with_faces(&[Vec3u::new(0, 1, 5)]);
        let view = scene.view();
        let empty = constraint(&[], &[]);
        let fatal = unsafe { rebuild(&view, &empty) }.expect_err("vertex 5 does not exist");
        assert_eq!(fatal.code, error_code::DEVICE_ASSERT);
        assert!(fatal.detail.contains("face 0"), "{:?}", fatal.detail);
    }

    /// A scene with no pins at all is accepted and clears every index.
    #[test]
    fn an_empty_constraint_unpins_everything() {
        let mut scene = TestScene::new(2).with_edges(&[Vec2u::new(0, 1)]);
        for prop in scene.vertex_props_mut() {
            prop.fix_index = 3;
        }
        let view = scene.view();
        unsafe { rebuild(&view, &constraint(&[], &[])) }.expect("accepted");
        assert!(scene.vertex_props().iter().all(|p| p.fix_index == 0));
        assert!(!scene.edge_props()[0].fixed);
    }
}
