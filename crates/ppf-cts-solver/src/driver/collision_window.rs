// File: crates/ppf-cts-solver/src/driver/collision_window.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Collision windows: `init_collision_windows` and `refresh_collision_active`.
//!
//! An object may be authored to collide only during named time intervals. The
//! host builds one table for the whole scene, a per-vertex group id plus up to
//! eight `[start, end)` intervals per group, and hands it over once at
//! `initialize()`. Every step then asks which vertices, faces and edges are
//! collidable at the current time.
//!
//! A GROUP WITH NO INTERVALS IS ALWAYS ACTIVE, which is the default and is why
//! an empty implementation looks correct on every scene that authored none. It
//! is also exactly why an empty one is dangerous on a scene that did: leaving
//! everything permanently collidable makes an authored non-collision interval
//! silently do nothing, and the run completes with plausible frames.
//!
//! AN ELEMENT IS ACTIVE IF ANY OF ITS VERTICES IS. That direction is
//! deliberate on both other backends and is the safe one: a face straddling a
//! group boundary keeps colliding, so the window can only ever remove collision
//! from geometry entirely inside a closed group.
//!
//! WHAT CONSUMES THE FLAGS IS PHASE 8, and nothing reads them yet. They are
//! computed now anyway, because the alternative is to leave a surface that
//! reports success while answering a question it never asked; and because the
//! validations below, which are the loud half, are worth having the moment the
//! table can arrive at all.

// The masks have no consumer until the contact filters land, which is dead code
// by the compiler's reckoning and deliberate by the plan's: the table and its
// validations land with their own tests rather than waiting for a reader.
#![allow(dead_code)]

use ppf_cts_compute::{ReadbackBuffer, AllocLabel, Device, Handle, StagedBuffer};
use ppf_cts_core::datamodel::object::MAX_COLLISION_WINDOWS;

use super::scene::{Fatal, FatalResult, SceneView};

/// The scene's collision-window table and the masks derived from it.
#[derive(Default)]
pub struct CollisionWindows {
    /// Group id per vertex, indexed by global vertex.
    vertex_group: Vec<u32>,
    /// `[start, end)` pairs, `MAX_COLLISION_WINDOWS * 2` floats per group.
    windows: Vec<f32>,
    /// How many of each group's eight slots are in use.
    window_count: Vec<u32>,
    group_count: u32,
    /// `u32` rather than `bool`, because a generated entry point's buffer
    /// pointee is one of float, int or unsigned: a byte mask has no lane. The
    /// cost is three per-primitive arrays at four bytes instead of one, which
    /// is small beside the position and Hessian arrays beside them.
    /// ALL THREE ARE STAGED ALLOCATIONS. The host recomputes them from the
    /// clock every step and only the masked broad-phase queries read them,
    /// which is the `StagedBuffer` shape; `handle()` refuses until this step's
    /// upload has run, so a query cannot read the previous step's windows.
    /// The window TABLE on the device, staged once when the table is
    /// installed and not uploaded again, because a window is authored geometry
    /// rather than per-step state.
    vertex_group_device: StagedBuffer<u32>,
    windows_device: StagedBuffer<f32>,
    window_count_device: StagedBuffer<u32>,
    /// The face and edge index lists on the device.
    ///
    /// A STAGED COPY RATHER THAN A READ OF THE DATASET'S OWN INDEX ARRAYS. The
    /// entry that refreshes the masks holds a `SceneView`, while the solver
    /// state that owns the mesh sits behind a SECOND mutex that
    /// `step::advance` already takes in the other order, so reaching the mesh
    /// from here would invert a documented lock pair. The copy costs
    /// `3 * faces + 2 * edges` words and no ordering risk, and it is refreshed
    /// on exactly the condition the masks are.
    face_index_device: StagedBuffer<u32>,
    edge_index_device: StagedBuffer<u32>,
    vertex_active: ReadbackBuffer<u32>,
    face_active: ReadbackBuffer<u32>,
    edge_active: ReadbackBuffer<u32>,
    initialized: bool,
}

impl CollisionWindows {
    /// The empty table, const so the backend can hold one in a `static`.
    pub const fn new() -> Self {
        Self {
            vertex_group: Vec::new(),
            windows: Vec::new(),
            window_count: Vec::new(),
            group_count: 0,
            vertex_group_device: StagedBuffer::none(),
            windows_device: StagedBuffer::none(),
            window_count_device: StagedBuffer::none(),
            face_index_device: StagedBuffer::none(),
            edge_index_device: StagedBuffer::none(),
            vertex_active: ReadbackBuffer::none(),
            face_active: ReadbackBuffer::none(),
            edge_active: ReadbackBuffer::none(),
            initialized: false,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// The per-vertex mask, or `None` when the scene authored no windows.
    ///
    /// A HANDLE RATHER THAN A SLICE, and the `Option` is what carries "no
    /// table": a record field is a handle with no spelling for absent, which is
    /// why the masked and unmasked queries are two entry points over one body
    /// and the driver picks between them on exactly this `Option`.
    pub fn vertex_active(&mut self) -> Option<Handle> {
        // `&mut` BECAUSE A HANDLE STALES THE MIRROR: a kernel writes this mask
        // now, so taking its handle is what marks the host copy out of date.
        if !self.initialized {
            return None;
        }
        Some(self.vertex_active.handle())
    }

    pub fn face_active(&mut self) -> Option<Handle> {
        // `&mut` BECAUSE A HANDLE STALES THE MIRROR: a kernel writes this mask
        // now, so taking its handle is what marks the host copy out of date.
        if !self.initialized {
            return None;
        }
        Some(self.face_active.handle())
    }

    pub fn edge_active(&mut self) -> Option<Handle> {
        // `&mut` BECAUSE A HANDLE STALES THE MIRROR: a kernel writes this mask
        // now, so taking its handle is what marks the host copy out of date.
        if !self.initialized {
            return None;
        }
        Some(self.edge_active.handle())
    }

    /// Install the table.
    ///
    /// Everything is validated before anything is stored, so a rejected table
    /// leaves the previous state intact and a scene never runs on half a table.
    ///
    /// # Safety
    /// `view` must address a live `DataSet`. `vertex_group` must hold
    /// `vertex_count` entries, `windows` `n_groups * MAX_COLLISION_WINDOWS * 2`
    /// floats, and `window_count` `n_groups` entries.
    pub unsafe fn initialize(
        &mut self,
        device: &mut impl Device,
        view: &SceneView,
        vertex_group: &[u32],
        windows: &[f32],
        window_count: &[u32],
        group_count: u32,
    ) -> FatalResult<()> {
        // Metal's three validations, which CUDA does not make and which are the
        // reason a malformed table stops the run here instead of reading past
        // the end of an array with no diagnostic anywhere.
        if group_count == 0 {
            return Err(Fatal::invariant(
                "the collision-window table declares no groups, so no vertex can name one",
            ));
        }
        let scene_vertices = view.vertex_count();
        if vertex_group.len() != scene_vertices {
            return Err(Fatal::invariant(format!(
                "the collision-window table covers {} vertices and the scene has {scene_vertices}",
                vertex_group.len()
            )));
        }
        let expected_windows = group_count as usize * MAX_COLLISION_WINDOWS * 2;
        if windows.len() != expected_windows {
            return Err(Fatal::invariant(format!(
                "the collision-window table carries {} interval bounds for {group_count} groups, \
                 and the wire layout is {MAX_COLLISION_WINDOWS} intervals of two per group \
                 ({expected_windows})",
                windows.len()
            )));
        }
        if window_count.len() != group_count as usize {
            return Err(Fatal::invariant(format!(
                "the collision-window table carries {} per-group counts for {group_count} groups",
                window_count.len()
            )));
        }
        for (group, count) in window_count.iter().enumerate() {
            if *count as usize > MAX_COLLISION_WINDOWS {
                return Err(Fatal::invariant(format!(
                    "collision-window group {group} declares {count} intervals, above the \
                     {MAX_COLLISION_WINDOWS}-interval wire limit"
                )));
            }
        }
        for (vertex, group) in vertex_group.iter().enumerate() {
            if *group >= group_count {
                return Err(Fatal::invariant(format!(
                    "collision-window vertex {vertex} names group {group} of {group_count}"
                )));
            }
        }

        let face_count = view.faces().len();
        let edge_count = view.edges().len();

        // Allocations are reported rather than aborted: a table this size is
        // proportional to the scene, so running out here is a legitimate
        // outcome and it has its own code in the status record.
        let stored_group = copy_or_report(vertex_group, "collision-window vertex groups")?;
        let stored_windows = copy_or_report(windows, "collision-window intervals")?;
        let stored_count = copy_or_report(window_count, "collision-window counts")?;
        // Everything starts collidable, which is what the masks mean before the
        // first `refresh_collision_active` has had a time to evaluate them at.
        //
        // FILLED IN PLACE, NOT REBUILT. Freeing the six device arrays and
        // allocating replacements would reset the same state, but the shape
        // used here is a caller-owned buffer [`set_collidable`] sizes, because
        // a helper RETURNING a buffer strands the one it is assigned over.
        // `Buffer::size` reuses the allocation and grows only past capacity, so
        // a second `initialize` over the same scene neither frees nor
        // allocates.
        //
        // THE TABLE IS MARKED UNINSTALLED FOR THE DURATION, which is what keeps
        // this function's promise now that the masks are the fields themselves:
        // a validation failure above returns before anything is touched, and an
        // allocation failure below leaves a table that reports itself absent
        // rather than one whose masks are half the scene's length.
        self.initialized = false;
        set_collidable(device, &mut self.vertex_active, scene_vertices, "window.vertex_active")?;
        set_collidable(device, &mut self.face_active, face_count, "window.face_active")?;
        set_collidable(device, &mut self.edge_active, edge_count, "window.edge_active")?;

        self.vertex_group = stored_group;
        self.windows = stored_windows;
        self.window_count = stored_count;
        self.group_count = group_count;
        // THE TABLE GOES TO THE DEVICE ONCE, here, because that is where it
        // stops changing: `refresh` evaluates it against a time and writes no
        // part of it back.
        stage_u32(device, &mut self.vertex_group_device, &self.vertex_group,
                  "window.vertex_group")?;
        stage_f32(device, &mut self.windows_device, &self.windows,
                  "window.windows")?;
        stage_u32(device, &mut self.window_count_device, &self.window_count,
                  "window.window_count")?;
        self.initialized = true;
        Ok(())
    }

    /// Recompute the three masks for `time`.
    ///
    /// A backend with no table installed does nothing, which is not a silent
    /// fallback: a scene that authored no windows has nothing to evaluate and
    /// every element stays collidable, which is what the absence means.
    ///
    /// # Safety
    /// `view` must address a live `DataSet`.
    pub unsafe fn refresh(
        &mut self,
        device: &mut impl Device,
        view: &SceneView,
        time: f32,
    ) -> FatalResult<()> {
        if !self.initialized {
            return Ok(());
        }

        // The element counts are re-read every step because the mesh arrays are
        // the live ones; CUDA reallocates its device masks on the same
        // condition. A change here is a scene whose element count moved under
        // the table, which nothing in the step loop does today.
        let faces = view.faces();
        let edges = view.edges();
        if self.face_active.len() != faces.len() {
            set_collidable(device, &mut self.face_active, faces.len(), "window.face_active")?;
        }
        if self.edge_active.len() != edges.len() {
            set_collidable(device, &mut self.edge_active, edges.len(), "window.edge_active")?;
        }
        // THE TOPOLOGY TRAVELS WITH THE MASK IT INDEXES, on the same condition,
        // so the two can never disagree about how many elements there are.
        if self.face_index_device.len() != 3 * faces.len() {
            let flat: Vec<u32> = faces.iter().flat_map(|f| [f[0], f[1], f[2]]).collect();
            stage_u32(device, &mut self.face_index_device, &flat, "window.face_index")?;
        }
        if self.edge_index_device.len() != 2 * edges.len() {
            let flat: Vec<u32> = edges.iter().flat_map(|e| [e[0], e[1]]).collect();
            stage_u32(device, &mut self.edge_index_device, &flat, "window.edge_index")?;
        }

        // THREE DISPATCHES: one over the vertices off the window table, then
        // one each over the faces and the edges propagating it. A host loop
        // over every vertex, face and edge followed by three uploads would
        // produce the same masks and is the rule (1a-0) shape the driver may
        // not have, so the evaluation stays on the device.
        //
        // THE THREE ARE ORDERED AND THE ORDER IS LOAD-BEARING: the element
        // passes READ the vertex mask this pass writes, so they cannot be
        // merged into it and cannot precede it.
        let stride = (MAX_COLLISION_WINDOWS * 2) as u32;
        let vertices = self.vertex_active.len() as u32;
        let vertex_args = crate::driver::kernels::CollisionWindowVertexArgs {
            vertex_group: self.vertex_group_device.handle(),
            windows: self.windows_device.handle(),
            window_count: self.window_count_device.handle(),
            stride,
            time,
            vertex_active: self.vertex_active.handle(),
            count: vertices,
            seam_arena_count: 0,
        };
        // Safety: every handle names a live allocation for the whole call.
        unsafe { device.launch("window.vertex_active", &vertex_args, vertices) }?;

        let face_count = self.face_active.len() as u32;
        if face_count > 0 {
            let face_args = crate::driver::kernels::CollisionWindowFaceArgs {
                vertex_active: self.vertex_active.handle(),
                face: self.face_index_device.handle(),
                face_active: self.face_active.handle(),
                count: face_count,
                seam_arena_count: 0,
            };
            // Safety: as above.
            unsafe { device.launch("window.face_active", &face_args, face_count) }?;
        }
        let edge_count = self.edge_active.len() as u32;
        if edge_count > 0 {
            let edge_args = crate::driver::kernels::CollisionWindowEdgeArgs {
                vertex_active: self.vertex_active.handle(),
                edge: self.edge_index_device.handle(),
                edge_active: self.edge_active.handle(),
                count: edge_count,
                seam_arena_count: 0,
            };
            // Safety: as above.
            unsafe { device.launch("window.edge_active", &edge_args, edge_count) }?;
        }
        Ok(())
    }
}

fn copy_or_report<T: Copy>(source: &[T], what: &str) -> FatalResult<Vec<T>> {
    let mut out: Vec<T> = Vec::new();
    out.try_reserve_exact(source.len()).map_err(|_| {
        Fatal::out_of_memory(format!(
            "cannot allocate the {what} ({} entries)",
            source.len()
        ))
    })?;
    out.extend_from_slice(source);
    Ok(out)
}

/// Size a CALLER-OWNED mask for `len` elements with every element COLLIDABLE,
/// staged and uploaded.
///
/// One is what the table means before the first `refresh` has had a time to
/// evaluate the windows at, and it is also what a group with no interval means
/// forever.
///
/// # It takes the buffer rather than returning one, and that is the whole point
///
/// A helper that RETURNS a `StagedBuffer` is assigned over a field, and the
/// buffer that field held is then dropped. A dropped buffer does not free: a
/// `Drop` cannot reach the device that owns the span, so the arena span is
/// stranded for the run's life with nothing reporting it. That is the shape the
/// tree has already been burned by once, in `lbvh::build`.
///
/// The explicit form of the same lifetime is to release each array, when one is
/// held, before allocating it again. `Buffer::size` REUSES the allocation and
/// grows only past capacity, which is that accounting with no free left to
/// forget: the element count seen moving here is the face or edge count, which
/// grows into the same span rather than into a new one.
fn stage_u32(
    device: &mut impl Device,
    buffer: &mut StagedBuffer<u32>,
    host: &[u32],
    label: &'static str,
) -> FatalResult<()> {
    buffer.size(device, host.len(), AllocLabel(label))?;
    if !host.is_empty() {
        buffer.at().copy_from_slice(host);
    }
    buffer.upload(device)?;
    Ok(())
}

fn stage_f32(
    device: &mut impl Device,
    buffer: &mut StagedBuffer<f32>,
    host: &[f32],
    label: &'static str,
) -> FatalResult<()> {
    buffer.size(device, host.len(), AllocLabel(label))?;
    if !host.is_empty() {
        buffer.at().copy_from_slice(host);
    }
    buffer.upload(device)?;
    Ok(())
}

fn set_collidable(
    device: &mut impl Device,
    mask: &mut ReadbackBuffer<u32>,
    len: usize,
    label: &'static str,
) -> FatalResult<()> {
    // A `ReadbackBuffer` BECAUSE A KERNEL WRITES IT NOW, and the host seeds
    // it: `refresh` fills all three with a dispatch, while this seed is what
    // they hold before the first `refresh` has a time to evaluate. That is the
    // both-written shape the buffer vocabulary names, and `host()` refuses
    // until a `download` rather than handing back this seed as if it were the
    // dispatch's answer.
    mask.size(device, len, AllocLabel(label))?;
    let ones = vec![1u32; len];
    mask.seed(device, &ones)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::test_scene::TestScene;
    use crate::data::{Vec2u, Vec3u};
    use ppf_cts_formats::status::error_code;

    /// One group's slice of the flat interval table.
    fn table(groups: &[&[(f32, f32)]]) -> (Vec<f32>, Vec<u32>) {
        let mut windows = vec![0.0f32; groups.len() * MAX_COLLISION_WINDOWS * 2];
        let mut counts = vec![0u32; groups.len()];
        for (group, intervals) in groups.iter().enumerate() {
            counts[group] = intervals.len() as u32;
            for (i, (start, end)) in intervals.iter().enumerate() {
                windows[group * MAX_COLLISION_WINDOWS * 2 + i * 2] = *start;
                windows[group * MAX_COLLISION_WINDOWS * 2 + i * 2 + 1] = *end;
            }
        }
        (windows, counts)
    }

    /// A window that closes actually closes.
    ///
    /// The defect an empty `refresh_collision_active` hides is exactly this:
    /// everything stays collidable forever, so an authored non-collision
    /// interval silently does nothing and the run completes looking fine.
    /// Bring all three mirrors current.
    ///
    /// THE MASKS ARE KERNEL-WRITTEN NOW, so `host()` refuses until a download
    /// rather than handing back the seed. A test that read the mirror without
    /// this would be reading `set_collidable`'s all-ones fill and calling it
    /// the dispatch's answer.
    fn sync(table: &mut CollisionWindows, device: &mut impl Device) {
        table.vertex_active.download(device).expect("the vertex mask reads back");
        table.face_active.download(device).expect("the face mask reads back");
        table.edge_active.download(device).expect("the edge mask reads back");
    }

    #[test]
    fn a_closed_window_deactivates_its_vertices_faces_and_edges() {
        // Vertices 0 and 1 are in group 1, which collides only on [1, 2).
        // Vertex 2 is in group 0, which has no interval and is always active.
        let scene = TestScene::new(3)
            .with_faces(&[Vec3u::new(0, 1, 2), Vec3u::new(0, 1, 1)])
            .with_edges(&[Vec2u::new(0, 1), Vec2u::new(1, 2)]);
        let view = scene.view();
        let (windows, counts) = table(&[&[], &[(1.0, 2.0)]]);
        // ONE DEVICE FOR THE WHOLE TEST. A handle names an arena index and
        // nothing that opened it, so allocating on one throwaway device and
        // dispatching on a second resolves it into a different allocation.
        let mut device = super::super::launch::host_device();
        let mut table_state = CollisionWindows::new();
        unsafe { table_state.initialize(&mut device, &view, &[1, 1, 0], &windows, &counts, 2) }
            .expect("a well-formed table must be accepted");

        unsafe { table_state.refresh(&mut device, &view, 1.5) }.expect("accepted");
        sync(&mut table_state, &mut device);
        assert_eq!(table_state.vertex_active.host(),
            &[1, 1, 1][..]);

        unsafe { table_state.refresh(&mut device, &view, 2.5) }.expect("accepted");
        sync(&mut table_state, &mut device);
        assert_eq!(
            table_state.vertex_active.host(),
            &[0, 0, 1][..],
            "a group with no interval is always active; a group past its last \
             interval is not"
        );
        assert_eq!(
            table_state.face_active.host(),
            &[1, 0][..],
            "an element is active if ANY of its vertices is, so the face that \
             touches the always-on group survives and the one that does not \
             goes quiet"
        );
        assert_eq!(table_state.edge_active.host(),
            &[0, 1][..]);

        // The interval is half open: active at its start, inactive at its end.
        unsafe { table_state.refresh(&mut device, &view, 1.0) }.expect("accepted");
        sync(&mut table_state, &mut device);
        assert_eq!(table_state.vertex_active.host(),
            &[1, 1, 1][..]);
        unsafe { table_state.refresh(&mut device, &view, 2.0) }.expect("accepted");
        sync(&mut table_state, &mut device);
        assert_eq!(table_state.vertex_active.host(),
            &[0, 0, 1][..]);
    }

    /// A vertex naming a group that does not exist stops the run.
    ///
    /// CUDA indexes `d_wc[dm]` with no check, so this is a device-side read
    /// past the end of the count array with nothing reported anywhere.
    #[test]
    fn a_vertex_naming_a_missing_group_is_refused() {
        let scene = TestScene::new(2);
        let view = scene.view();
        let (windows, counts) = table(&[&[(0.0, 1.0)]]);
        // ONE DEVICE FOR THE WHOLE TEST. A handle names an arena index and
        // nothing that opened it, so allocating on one throwaway device and
        // dispatching on a second resolves it into a different allocation.
        let mut device = super::super::launch::host_device();
        let mut table_state = CollisionWindows::new();
        let fatal = unsafe { table_state.initialize(&mut device, &view, &[0, 4], &windows, &counts, 1) }
            .expect_err("group 4 of 1 does not exist");
        assert_eq!(fatal.code, error_code::SOLVER_INVARIANT);
        assert!(fatal.detail.contains("vertex 1"), "{:?}", fatal.detail);
        assert!(
            !table_state.is_initialized(),
            "a rejected table must not be installed"
        );
    }

    /// A group declaring more intervals than the wire carries is refused.
    #[test]
    fn a_group_above_the_interval_limit_is_refused() {
        let scene = TestScene::new(1);
        let view = scene.view();
        let (windows, mut counts) = table(&[&[(0.0, 1.0)]]);
        counts[0] = MAX_COLLISION_WINDOWS as u32 + 1;
        // ONE DEVICE FOR THE WHOLE TEST. A handle names an arena index and
        // nothing that opened it, so allocating on one throwaway device and
        // dispatching on a second resolves it into a different allocation.
        let mut device = super::super::launch::host_device();
        let mut table_state = CollisionWindows::new();
        let fatal = unsafe { table_state.initialize(&mut device, &view, &[0], &windows, &counts, 1) }
            .expect_err("nine intervals do not fit in eight slots");
        assert_eq!(fatal.code, error_code::SOLVER_INVARIANT);
        assert!(fatal.detail.contains("wire limit"), "{:?}", fatal.detail);
    }

    /// A table that does not cover the scene is refused.
    #[test]
    fn a_table_of_the_wrong_length_is_refused() {
        let scene = TestScene::new(3);
        let view = scene.view();
        let (windows, counts) = table(&[&[]]);
        // ONE DEVICE FOR THE WHOLE TEST. A handle names an arena index and
        // nothing that opened it, so allocating on one throwaway device and
        // dispatching on a second resolves it into a different allocation.
        let mut device = super::super::launch::host_device();
        let mut table_state = CollisionWindows::new();
        let fatal = unsafe { table_state.initialize(&mut device, &view, &[0, 0], &windows, &counts, 1) }
            .expect_err("two group ids do not cover three vertices");
        assert_eq!(fatal.code, error_code::SOLVER_INVARIANT);
        assert!(fatal.detail.contains("3"), "{:?}", fatal.detail);
    }

    /// An element count that moves REUSES the mask rather than stranding it.
    ///
    /// THE DEFECT THIS GUARDS IS SILENT AND PERMANENT. Rebuilding a mask
    /// through a helper that RETURNS a `StagedBuffer`, assigned over the field,
    /// drops the buffer the field held, and a dropped buffer does not free,
    /// because `Drop` cannot reach the device that owns the span. Nothing
    /// fails, nothing is logged, and the arena keeps the span for the run. The
    /// caller-owned form `stage_u32` takes is what rules that out, and this
    /// test is what holds it to it.
    ///
    /// THE ALLOCATOR'S GENERATION IS THE WITNESS, not its reserved bytes: the
    /// host pool's byte total is a high-water mark that a stranded span does
    /// not move, so an accounting test written that way passes over the defect.
    /// `allocator_generation` is bumped by every `alloc`, `grow` and `free`,
    /// which is exactly the event a reused mask must not cause. It has to be
    /// read across a CYCLE rather than at one point, because the first widening
    /// legitimately allocates; what proves the reuse is that going back down and
    /// up again allocates nothing further.
    #[test]
    fn a_changed_element_count_reuses_the_mask_rather_than_stranding_it() {
        let narrow = TestScene::new(3)
            .with_faces(&[Vec3u::new(0, 1, 2)])
            .with_edges(&[Vec2u::new(0, 1)]);
        let wide = TestScene::new(3)
            .with_faces(&[
                Vec3u::new(0, 1, 2),
                Vec3u::new(0, 1, 2),
                Vec3u::new(0, 1, 2),
                Vec3u::new(0, 1, 2),
                Vec3u::new(0, 1, 2),
                Vec3u::new(0, 1, 2),
            ])
            .with_edges(&[
                Vec2u::new(0, 1),
                Vec2u::new(0, 1),
                Vec2u::new(0, 1),
                Vec2u::new(0, 1),
                Vec2u::new(0, 1),
                Vec2u::new(0, 1),
            ]);
        let narrow_view = narrow.view();
        let wide_view = wide.view();
        let (windows, counts) = table(&[&[]]);
        // ONE DEVICE FOR THE WHOLE TEST. A handle names an arena index and
        // nothing that opened it, so allocating on one throwaway device and
        // dispatching on a second resolves it into a different allocation.
        let mut device = super::super::launch::host_device();
        let mut table_state = CollisionWindows::new();
        unsafe {
            table_state.initialize(&mut device, &narrow_view, &[0, 0, 0], &windows, &counts, 1)
        }
        .expect("a well-formed table must be accepted");
        unsafe { table_state.refresh(&mut device, &narrow_view, 0.0) }.expect("accepted");

        // The one legitimate widening: six faces and six edges do not fit in one.
        unsafe { table_state.refresh(&mut device, &wide_view, 0.0) }.expect("accepted");
        let settled = device.allocator_generation();

        // Down and back up twice. Every one of these four calls takes the
        // resize branch, and none of them may reach the allocator.
        for _ in 0..2 {
            unsafe { table_state.refresh(&mut device, &narrow_view, 0.0) }.expect("accepted");
            unsafe { table_state.refresh(&mut device, &wide_view, 0.0) }.expect("accepted");
        }
        assert_eq!(
            device.allocator_generation(),
            settled,
            "a mask whose element count moved must be sized in place; a helper \
             that returns a buffer allocates a second span and strands the one \
             it replaces, and nothing frees it"
        );
        sync(&mut table_state, &mut device);
        assert_eq!(
            table_state.face_active.host(),
            &[1, 1, 1, 1, 1, 1][..],
            "and the reused mask must still read as fully collidable"
        );
    }

    /// With no table installed, a refresh does nothing and reports nothing.
    ///
    /// That is not a silent fallback: a scene that authored no window has
    /// nothing to evaluate, and every element is collidable, which is what the
    /// absence means.
    #[test]
    fn a_scene_with_no_windows_has_no_masks() {
        let scene = TestScene::new(2);
        let view = scene.view();
        // ONE DEVICE FOR THE WHOLE TEST. A handle names an arena index and
        // nothing that opened it, so allocating on one throwaway device and
        // dispatching on a second resolves it into a different allocation.
        let mut device = super::super::launch::host_device();
        let mut table_state = CollisionWindows::new();
        unsafe { table_state.refresh(&mut device, &view, 3.0) }.expect("accepted");
        sync(&mut table_state, &mut device);
        assert!(table_state.vertex_active().is_none());
        assert!(table_state.face_active().is_none());
        assert!(table_state.edge_active().is_none());
    }
}
