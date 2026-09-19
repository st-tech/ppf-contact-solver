// File: rigid_map.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! The PDRD reduction's topology: which vertices belong to which rigid body,
//! where each free vertex sits in the reduced vector, and what each body's
//! joint and lock constraints are.
//!
//! A transcription of `build_rigid_map` in
//! `src/kernels/energy/model/pdrd_rigid.hpp`, which is itself the neutral form
//! of the reference's. That function is host orchestration around the eighteen
//! generated rows; the rows are shared and the orchestration is not, so the
//! Rust driver needs its own copy for the same reason it has its own Newton
//! loop.
//!
//! WHAT THE REDUCTION IS. A PDRD body's vertices carry no per-vertex degrees of
//! freedom: the whole body moves by six numbers, three translational and three
//! rotational, and its vertices follow. So the Newton system is solved in a
//! REDUCED vector of `3 * n_cloth + 6 * n_bodies` floats rather than `3 * nrow`,
//! and `vbody` and `cloth_off` are the map between the two.
//!
//! THE PARTITION IS A PURE FUNCTION OF THE SCENE and is fixed for the lifetime
//! of a solve, which is why the reference caches it behind `built` and rebuilds
//! only when the counts change. Nothing in the step loop can move a vertex
//! between a body and the cloth.

use super::lock_math::{
    rotation_lock_enabled, rotation_mode_valid, translation_lock_enabled,
    translation_mode_valid,
};
use crate::data::{DataSet, TranslationLock, Vec3f, TRANSLATION_LOCK_AXIS};

/// A vertex that belongs to no rigid body, and a body with no lock: the
/// reference's `RIGID_UNSET`, which is `PDRD_CLOTH_MARKER`.
pub const RIGID_UNSET: u32 = 0xffff_ffff;

/// `PDRD_JOINT_FREE`: a full six-DOF body, as against a hinge.
pub const PDRD_JOINT_FREE: u32 = 0;

/// `PDRD_JOINT_HINGE`: translation locked, spin about the axle. The value is
/// the neutral kernel's own (`energy/model/pdrd_rigid.kernel.cpp`), which is
/// where the enumeration lives; this is a host-side reader of the same number.
pub const PDRD_JOINT_HINGE: u32 = 1;

/// The reduction's topology.
///
/// EVERY ARRAY IS HOST-SIDE HERE, because every consumer of it is: the reduced
/// solve's dispatches take these as device buffers, and the driver stages them
/// once when the topology is built rather than rebuilding them per step.
#[derive(Default, Debug)]
pub struct RigidMap {
    pub nrow: usize,
    pub n_bodies: usize,
    pub n_cloth: usize,
    /// `3 * n_cloth + 6 * n_bodies`, the reduced vector's length.
    pub dim: usize,
    /// `3 * n_cloth`, where the bodies' six-vectors begin.
    pub body_base: usize,
    /// Per vertex: its 0-based body id, or `RIGID_UNSET` for a free vertex.
    pub vbody: Vec<u32>,
    /// Per vertex: its float offset in the reduced vector, meaningful only for
    /// a free one.
    pub cloth_off: Vec<u32>,
    /// Per body: the joint mode and, for a hinge, its world axle.
    pub jmode: Vec<u32>,
    pub jaxis: Vec<Vec3f>,
    /// Per body: the compact index of the translation or rotation lock that
    /// constrains it, or `RIGID_UNSET`, and that lock's world axis.
    pub tlock: Vec<u32>,
    pub tlock_axis: Vec<Vec3f>,
    /// Axis or all-axes mode per body. THE MODE CARRIES THE ENABLE BIT, so this
    /// travels with the axis rather than being inferable from it.
    pub tlock_mode: Vec<u32>,
    pub rlock: Vec<u32>,
    pub rlock_axis: Vec<Vec3f>,
    pub rlock_mode: Vec<u32>,
    pub any_joint: bool,
    pub any_translation_lock: bool,
    pub any_rotation_lock: bool,
}

impl RigidMap {
    /// Whether the reduced solve has anything to project.
    ///
    /// `launch_project_bodies` returns immediately on this, so the driver tests
    /// it before dispatching rather than dispatching a no-op.
    pub fn needs_projection(&self) -> bool {
        self.n_bodies > 0
            && (self.any_joint || self.any_translation_lock || self.any_rotation_lock)
    }
}

/// Build the reduction's topology from the scene.
///
/// # Safety
/// `data` must be live and carry `nrow` vertex props.
pub unsafe fn build(data: &DataSet, nrow: usize) -> Result<RigidMap, String> {
    let n_bodies = data.prop.pdrd_body.size as usize;
    let mut map = RigidMap {
        nrow,
        n_bodies,
        vbody: vec![RIGID_UNSET; nrow],
        cloth_off: vec![0; nrow],
        ..RigidMap::default()
    };

    // THE PARTITION. A vertex whose `pdrd_body_index` is zero is free and takes
    // the next three floats of the reduced vector; one that names a body is
    // carried by that body's six and takes none. The index is 1-BASED in the
    // scene and 0-based here, which is the reference's own convention and the
    // reason `idx1 - 1` appears rather than a cast.
    let props: &[crate::data::VertexProp] = super::scene::slice(&data.prop.vertex);
    let mut n_cloth = 0usize;
    for vertex in 0..nrow.min(props.len()) {
        let one_based = props[vertex].pdrd_body_index;
        if one_based == 0 {
            map.vbody[vertex] = RIGID_UNSET;
            map.cloth_off[vertex] = 3 * n_cloth as u32;
            n_cloth += 1;
        } else {
            map.vbody[vertex] = one_based - 1;
        }
    }
    map.n_cloth = n_cloth;
    map.body_base = 3 * n_cloth;
    map.dim = 3 * n_cloth + 6 * n_bodies;

    if n_bodies == 0 {
        return Ok(map);
    }

    let bodies: &[crate::data::PdrdBodyProp] = super::scene::slice(&data.prop.pdrd_body);
    map.jmode = vec![PDRD_JOINT_FREE; n_bodies];
    map.jaxis = vec![Vec3f::zeros(); n_bodies];
    map.tlock = vec![RIGID_UNSET; n_bodies];
    map.tlock_axis = vec![Vec3f::zeros(); n_bodies];
    map.tlock_mode = vec![TRANSLATION_LOCK_AXIS; n_bodies];
    map.rlock = vec![RIGID_UNSET; n_bodies];
    map.rlock_axis = vec![Vec3f::zeros(); n_bodies];
    map.rlock_mode = vec![0; n_bodies];
    let mut hinged = 0usize;
    for body in 0..n_bodies.min(bodies.len()) {
        map.jmode[body] = bodies[body].joint_mode;
        map.jaxis[body] = bodies[body].joint_axis;
        if bodies[body].joint_mode != PDRD_JOINT_FREE {
            map.any_joint = true;
        }
        if bodies[body].joint_mode == PDRD_JOINT_HINGE {
            hinged += 1;
        }
    }
    // WHICH PROJECTOR THE REDUCED SOLVE GOT, and how many bodies asked for one.
    // `identity` and `active` are different solves, and a scene that silently
    // took the first while its hinge was meant to bind looks like a physics
    // result rather than like a missing constraint.
    ::log::info!(
        "PDRD reduced projector {}, {hinged} hinged body(ies)",
        if map.needs_projection() { "active" } else { "identity" }
    );

    // THE LOCKS THAT NAME A BODY, which are the ones this projector does NOT
    // handle: a group whose `pdrd_body_index` is non-zero lives in the reduced
    // body vector and PDRD's own exact six-DOF projector removes its rows, so
    // the deformable projector builds none for it. The two have disjoint
    // supports and `builder.rs` rejects every grouping that would break that,
    // which is what makes the assertion below an invariant rather than a
    // validation.
    let locks: &[TranslationLock] = super::scene::slice(&data.translation_lock);
    for (compact, lock) in locks.iter().enumerate() {
        if lock.pdrd_body_index == 0 {
            continue;
        }
        let body = (lock.pdrd_body_index - 1) as usize;
        if body >= n_bodies {
            return Err(format!(
                "solver driver: locked group {} names PDRD body {} but the scene carries {} \
                 bodies",
                lock.dmap_index, lock.pdrd_body_index, n_bodies
            ));
        }
        // ENABLEMENT IS THE MODE, NOT THE AXIS: an all-axes lock ships a zero
        // axis, so an axis test would leave tlock or rlock at RIGID_UNSET and
        // silently drop the lock while the UI still showed it as set.
        if translation_lock_enabled(lock) {
            if !translation_mode_valid(lock.translation_mode) {
                return Err(format!(
                    "solver driver: locked group {} has invalid translation mode {}",
                    lock.dmap_index, lock.translation_mode
                ));
            }
            if map.tlock[body] != RIGID_UNSET {
                return Err(format!(
                    "solver driver: PDRD body {} is named by two translation locks, groups {} \
                     and {}. One body carries at most one of each kind, which builder.rs \
                     enforces at scene build",
                    body, map.tlock[body], compact
                ));
            }
            map.tlock[body] = compact as u32;
            map.tlock_axis[body] = lock.axis;
            map.tlock_mode[body] = lock.translation_mode;
            map.any_translation_lock = true;
        }
        if rotation_lock_enabled(lock) {
            if !rotation_mode_valid(lock.rotation_mode) {
                return Err(format!(
                    "solver driver: locked group {} has invalid rotation mode {}",
                    lock.dmap_index, lock.rotation_mode
                ));
            }
            if map.rlock[body] != RIGID_UNSET {
                return Err(format!(
                    "solver driver: PDRD body {} is named by two rotation locks, groups {} and \
                     {}",
                    body, map.rlock[body], compact
                ));
            }
            map.rlock[body] = compact as u32;
            map.rlock_axis[body] = lock.rotation_axis;
            map.rlock_mode[body] = lock.rotation_mode;
            map.any_rotation_lock = true;
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cvec::CVec;
    use crate::driver::test_scene::{position, TestScene};

    fn body(joint_mode: u32) -> crate::data::PdrdBodyProp {
        crate::data::PdrdBodyProp {
            joint_mode,
            ..Default::default()
        }
    }

    /// A scene with no PDRD body reduces to itself.
    ///
    /// The identity case, and the one every non-PDRD scene takes: every vertex
    /// is free, the reduced vector is the full one, and nothing is projected.
    #[test]
    fn a_scene_with_no_body_reduces_to_itself() {
        let scene = TestScene::new(4);
        // Safety: the scene lives in its box for the whole test.
        let map = unsafe { build(&scene.data, 4) }.expect("an unreduced scene builds");
        assert_eq!(map.n_cloth, 4);
        assert_eq!(map.n_bodies, 0);
        assert_eq!(map.dim, 12, "the reduced vector is the full one");
        assert_eq!(map.body_base, 12);
        assert_eq!(map.vbody, vec![RIGID_UNSET; 4]);
        assert_eq!(map.cloth_off, vec![0, 3, 6, 9]);
        assert!(!map.needs_projection());
    }

    /// A body's vertices leave the reduced vector and its six DOFs replace them.
    ///
    /// THE OFFSETS ARE THE POINT. A free vertex's slot is its position among the
    /// FREE vertices, not among all of them, so a body in the middle of the
    /// array shifts every later free vertex down. Getting that wrong puts one
    /// vertex's correction on another, which is a silently wrong solve rather
    /// than a crash.
    #[test]
    fn a_bodys_vertices_leave_the_reduced_vector() {
        let mut scene = TestScene::new(5);
        // Vertices 1 and 2 belong to body 1 (1-based in the scene).
        scene.vertex_props_mut()[1].pdrd_body_index = 1;
        scene.vertex_props_mut()[2].pdrd_body_index = 1;
        scene.data.prop.pdrd_body = CVec::from(&[body(PDRD_JOINT_FREE)][..]);
        // Safety: as above.
        let map = unsafe { build(&scene.data, 5) }.expect("a reduced scene builds");

        assert_eq!(map.n_cloth, 3, "three vertices stay free");
        assert_eq!(map.n_bodies, 1);
        assert_eq!(map.body_base, 9, "the body's six DOFs start after the free nine");
        assert_eq!(map.dim, 15, "nine free floats plus one body's six");
        assert_eq!(
            map.vbody,
            vec![RIGID_UNSET, 0, 0, RIGID_UNSET, RIGID_UNSET],
            "the scene's 1-based body index becomes 0-based here"
        );
        // Vertices 3 and 4 are the SECOND and THIRD free vertices, so their
        // offsets are 3 and 6 rather than 9 and 12.
        assert_eq!(map.cloth_off[0], 0);
        assert_eq!(map.cloth_off[3], 3);
        assert_eq!(map.cloth_off[4], 6);
        assert!(
            !map.needs_projection(),
            "a free body with no lock has nothing to project"
        );

        // EVERY CLOTH OFFSET LIES BELOW `body_base`, which is the invariant the
        // layout rests on and the one a naive offset breaks. Numbering a free
        // vertex by its position among ALL vertices rather than among the FREE
        // ones puts the later ones on top of the bodies' six-vectors, and the
        // restrict-then-prolong round trip still passes: the cloth values are
        // stored and read back from the same wrong slots, and what is destroyed
        // is the body rows nobody looked at. So the round trip cannot be the
        // guard for this and the bound is asserted here instead.
        for vertex in 0..map.nrow {
            if map.vbody[vertex] != RIGID_UNSET {
                continue;
            }
            let offset = map.cloth_off[vertex] as usize;
            assert!(
                offset + 3 <= map.body_base,
                "free vertex {vertex} takes reduced slots {}..{} , which reaches \
                 into the bodies' region at {}",
                offset,
                offset + 3,
                map.body_base
            );
        }
    }

    /// A hinge, or a lock naming a body, is what makes the projection necessary.
    #[test]
    fn a_joint_or_a_lock_is_what_needs_projecting() {
        let mut scene = TestScene::new(3);
        scene.vertex_props_mut()[0].pdrd_body_index = 1;
        scene.data.prop.pdrd_body = CVec::from(&[body(1)][..]);
        // Safety: as above.
        let map = unsafe { build(&scene.data, 3) }.expect("a hinged scene builds");
        assert!(map.any_joint, "a non-free joint mode must be seen");
        assert!(map.needs_projection());

        // A lock naming the body, which PDRD's own projector removes rather
        // than the deformable one.
        let mut locked = TestScene::new(3);
        locked.vertex_props_mut()[0].pdrd_body_index = 1;
        locked.data.prop.pdrd_body = CVec::from(&[body(PDRD_JOINT_FREE)][..]);
        locked.data.translation_lock = CVec::from(
            &[TranslationLock {
                axis: Vec3f::new(0.0, 1.0, 0.0),
                translation_mode: TRANSLATION_LOCK_AXIS,
                total_mass: 1.0,
                pdrd_body_index: 1,
                dmap_index: 7,
                rotation_axis: Vec3f::zeros(),
                rotation_mode: 0,
                anchor: position(0.0, 0.0, 0.0),
            }][..],
        );
        // Safety: as above.
        let map = unsafe { build(&locked.data, 3) }.expect("a locked body builds");
        assert!(map.any_translation_lock);
        assert!(!map.any_rotation_lock, "the rotation axis is all zeros");
        assert_eq!(map.tlock[0], 0, "the body names the lock's compact index");
        assert!(map.needs_projection());
    }

    /// Two locks of one kind on one body is an invariant violation, not a
    /// tolerance.
    ///
    /// `builder.rs` rejects the grouping at scene build, so reaching here means
    /// the two projectors' disjoint supports have been broken and the reduced
    /// solve would remove one row twice.
    #[test]
    fn two_locks_of_one_kind_on_one_body_is_refused() {
        let mut scene = TestScene::new(2);
        scene.vertex_props_mut()[0].pdrd_body_index = 1;
        scene.data.prop.pdrd_body = CVec::from(&[body(PDRD_JOINT_FREE)][..]);
        let one = TranslationLock {
            axis: Vec3f::new(0.0, 1.0, 0.0),
            translation_mode: TRANSLATION_LOCK_AXIS,
            total_mass: 1.0,
            pdrd_body_index: 1,
            dmap_index: 7,
            rotation_axis: Vec3f::zeros(),
            rotation_mode: 0,
            anchor: position(0.0, 0.0, 0.0),
        };
        scene.data.translation_lock = CVec::from(&[one, one][..]);
        // Safety: as above.
        let fault = unsafe { build(&scene.data, 2) }.expect_err("two locks must be refused");
        assert!(
            fault.contains("two translation locks"),
            "the message must say what is wrong: {fault}"
        );
    }
}

/// The reduction's topology as DEVICE buffers.
///
/// STAGED ONCE, when the topology is built. The partition is a pure function of
/// the scene and fixed for the lifetime of a solve, so nothing in the step loop
/// re-uploads these; the reference caches the same arrays behind its `built`
/// flag for the same reason.
#[derive(Default)]
pub struct Staged {
    pub vbody: ppf_cts_compute::StagedBuffer<u32>,
    pub cloth_off: ppf_cts_compute::StagedBuffer<u32>,
    /// Per vertex, three floats: `p_k = R_b ybar_k`, the body-rotated rest
    /// vector. UNLIKE THE REST OF THIS STRUCT IT IS NOT TOPOLOGY: the rotation
    /// moves every step, so a kernel writes it and the reduction reads it.
    pub prot: ppf_cts_compute::Buffer<f32>,
    pub jmode: ppf_cts_compute::StagedBuffer<u32>,
    pub jaxis: ppf_cts_compute::StagedBuffer<f32>,
    pub tlock: ppf_cts_compute::StagedBuffer<u32>,
    pub tlock_axis: ppf_cts_compute::StagedBuffer<f32>,
    pub tlock_mode: ppf_cts_compute::StagedBuffer<u32>,
    pub rlock: ppf_cts_compute::StagedBuffer<u32>,
    pub rlock_axis: ppf_cts_compute::StagedBuffer<f32>,
    pub rlock_mode: ppf_cts_compute::StagedBuffer<u32>,
    /// The scene's own PDRD arrays as device buffers: the body vertex list, the
    /// rest shape centered on each body, and the per-body properties. Staged
    /// once with the topology, being scene data that no step rewrites.
    pub vert_list: ppf_cts_compute::StagedBuffer<u32>,
    pub rest_centered: ppf_cts_compute::StagedBuffer<f32>,
    pub body_prop: ppf_cts_compute::StagedBuffer<crate::data::PdrdBodyProp>,
    /// AN EMPTY DYNAMIC MATRIX, for a scene that assembles no contact at all.
    ///
    /// The reference has no such case to handle: `DynCSRMat::alloc(nrow, ...)`
    /// is always allocated, so a contact-free scene reaches the sandwich row
    /// with real `dyn_row_offsets` of `nrow + 1` zeros and its slot loop runs
    /// zero times. This driver carries the dynamic matrix as an `Option` that
    /// is `None` when the scene configures no contact, and the three handles
    /// were being filled with `Handle::NONE`, whose arena is `u32::MAX`: a
    /// generated entry resolves every buffer it is handed BEFORE the body
    /// runs, so the dispatch trapped on its own arena assert in exactly the
    /// scenes that have nothing for it to read.
    ///
    /// Zero-length would clear that assert and not be enough. The body indexes
    /// `dyn_offset[v]` and `dyn_offset[v + 1]` unconditionally, so the offsets
    /// have to be a real `nrow + 1` run of zeros; only the index and value
    /// arrays are genuinely empty. That is the reference's shape, reproduced
    /// rather than invented.
    pub empty_dyn_offset: ppf_cts_compute::Buffer<u32>,
    pub empty_dyn_index: ppf_cts_compute::Buffer<u32>,
    pub empty_dyn_value: ppf_cts_compute::Buffer<f32>,
}

impl Staged {
    /// Stage the topology.
    ///
    /// A triple is staged as three flat floats, which is the layout a `Vec3f`
    /// already has and the layout the entry records address.
    ///
    /// # Safety
    /// `data` must be live and its PDRD arrays sized as the scene declares.
    pub unsafe fn stage<D: ppf_cts_compute::Device>(
        &mut self,
        device: &mut D,
        map: &RigidMap,
        data: &DataSet,
    ) -> Result<(), ppf_cts_compute::Fault> {
        use ppf_cts_compute::AllocLabel;
        let nrow = map.nrow;
        let bodies = map.n_bodies;

        self.vbody.size(device, nrow, AllocLabel("pdrd.vbody"))?;
        self.vbody.at()[..nrow].copy_from_slice(&map.vbody);
        self.vbody.upload(device)?;

        self.cloth_off.size(device, nrow, AllocLabel("pdrd.cloth_off"))?;
        self.cloth_off.at()[..nrow].copy_from_slice(&map.cloth_off);
        self.cloth_off.upload(device)?;

        self.prot.size(device, 3 * nrow, AllocLabel("pdrd.prot"))?;

        // Sized with the topology, beside every other array whose extent is the
        // row count. `Buffer::size` grows only past capacity, so re-staging is
        // allocation-free, and a fresh allocation is zeroed, which is the only
        // content this ever needs: an all-zero offset run is an empty row for
        // every vertex.
        self.empty_dyn_offset
            .size(device, nrow + 1, AllocLabel("pdrd.empty_dyn_offset"))?;
        self.empty_dyn_index
            .size(device, 1, AllocLabel("pdrd.empty_dyn_index"))?;
        self.empty_dyn_value
            .size(device, 9, AllocLabel("pdrd.empty_dyn_value"))?;

        if bodies == 0 {
            return Ok(());
        }
        // The scene's own PDRD arrays. `rest_centered` stages as three flat
        // floats per body vertex, the layout a `Vec3f` already has.
        let listed = data.pdrd_vert_list.size as usize;
        self.vert_list.size(device, listed, AllocLabel("pdrd.vert_list"))?;
        if listed > 0 {
            self.vert_list.at()[..listed].copy_from_slice(
                std::slice::from_raw_parts(data.pdrd_vert_list.data, listed),
            );
        }
        self.vert_list.upload(device)?;

        let centred = 3 * data.pdrd_rest_centered.size as usize;
        self.rest_centered
            .size(device, centred, AllocLabel("pdrd.rest_centered"))?;
        if centred > 0 {
            self.rest_centered.at()[..centred].copy_from_slice(
                std::slice::from_raw_parts(
                    data.pdrd_rest_centered.data as *const f32,
                    centred,
                ),
            );
        }
        self.rest_centered.upload(device)?;

        self.body_prop.size(device, bodies, AllocLabel("pdrd.body_prop"))?;
        self.body_prop.at()[..bodies].copy_from_slice(
            std::slice::from_raw_parts(data.prop.pdrd_body.data, bodies),
        );
        self.body_prop.upload(device)?;
        for (buffer, source, label) in [
            (&mut self.jmode, &map.jmode, "pdrd.jmode"),
            (&mut self.tlock, &map.tlock, "pdrd.tlock"),
            (&mut self.tlock_mode, &map.tlock_mode, "pdrd.tlock_mode"),
            (&mut self.rlock, &map.rlock, "pdrd.rlock"),
            (&mut self.rlock_mode, &map.rlock_mode, "pdrd.rlock_mode"),
        ] {
            buffer.size(device, bodies, AllocLabel(label))?;
            buffer.at()[..bodies].copy_from_slice(source);
            buffer.upload(device)?;
        }
        for (buffer, source, label) in [
            (&mut self.jaxis, &map.jaxis, "pdrd.jaxis"),
            (&mut self.tlock_axis, &map.tlock_axis, "pdrd.tlock_axis"),
            (&mut self.rlock_axis, &map.rlock_axis, "pdrd.rlock_axis"),
        ] {
            buffer.size(device, 3 * bodies, AllocLabel(label))?;
            {
                let staged = buffer.at();
                for (body, axis) in source.iter().enumerate() {
                    staged[3 * body] = axis[0];
                    staged[3 * body + 1] = axis[1];
                    staged[3 * body + 2] = axis[2];
                }
            }
            buffer.upload(device)?;
        }
        Ok(())
    }
}
