// File: friction_memory.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! The friction's memory of where each contact stuck, carried from one
//! committed step to the next (`contact/friction_memory.kernel.cpp` states the
//! law).
//!
//! # Two halves, and only one of them is read during a step
//!
//! The MEMORY is the last committed step's pairs, sorted by key: every contact
//! pass of every Newton iteration reads it and none writes it. The RECORD is
//! what the current iteration's passes append, one entry per frictional pair,
//! and it is cleared at the start of each iteration's assembly, so at the end
//! of the step it holds the last iteration's pairs. [`FrictionMemory::commit`]
//! turns that record into the next step's memory, evaluating each pair's slip
//! at the pose actually committed. A step that is not committed leaves the
//! memory as it was.
//!
//! # The record is the step's friction
//!
//! Only a step's FIRST iteration writes the record, at the step's start pose;
//! every later iteration opens it with capacity zero, so the record stays as
//! that iteration wrote it, and every iteration applies friction from it
//! through [`launch_frozen`] (`contact/friction_memory.kernel.cpp`, "THE RECORD
//! IS ALSO THE STEP'S FRICTION", states why).
//!
//! # The vertices' cone sums and loads
//!
//! Closing an iteration also gathers its record into each vertex's lagged cone
//! sum, `sum |w| m mu N` over the pairs at the vertex, and writes each vertex's
//! non-friction contact load, what the passes deposited less the friction they
//! deposited. The next iteration's passes read both, so each pair's release
//! test sees its share of what friction at the vertex has to carry
//! (`contact/friction_memory.kernel.cpp`, "THE ANCHOR'S RELEASE TEST IS PER
//! VERTEX", states the law). Both are allocated once, five floats a vertex for
//! the sums and three for the friction the passes deposit.
//!
//! # Capacity
//!
//! The record has a fixed capacity per iteration, and a pass past it counts the
//! pair without writing it. Every iteration reads its count back (one word,
//! beside the assembly tallies the step already reads), and once a count passes
//! half the capacity the next iteration opens with twice that count, so an
//! iteration can overflow only by more than doubling the count of the one before
//! it. THE GROWTH WAITS FOR THE NEXT OPEN, never the close that measured it: a
//! grown buffer does not keep its contents, and the last iteration's record is
//! what the commit reads. An overflowed record is incomplete, and
//! [`FrictionMemory::commit`] refuses a step whose LAST iteration overflowed
//! rather than carry a memory with pairs missing from it, since a missing pair
//! would restart from zero slip without a word.

use ppf_cts_compute::{AllocLabel, Buffer, Device, Handle};

use super::devsort::SortScratch;
use super::dyncsr::DeviceStaging;
use super::kernels::{
    FrictionConeGatherArgs, FrictionFrozenEmbedArgs, FrictionLoadPrepareArgs, FrictionLoadSettleArgs, FrictionMemoryGatherWordArgs, FrictionMemoryHeadArgs, FrictionMemoryMergeArgs,
    FrictionMemorySettleArgs,
};
use super::scan::ScanScratch;
use super::scene::{Fatal, FatalResult};

/// Vertices and floats a record entry holds, as the kernel's
/// `FRICTION_MEMORY_VERTICES` and `FRICTION_MEMORY_FLOATS`.
const RECORD_VERTICES: usize = 4;
/// Words a record entry holds, as the kernel's `FRICTION_RECORD_WORDS`: its
/// vertices, then one Cosserat frame slot per side.
const RECORD_WORDS: usize = RECORD_VERTICES + 2;
const RECORD_FLOATS: usize = 16;

/// Floats a vertex's sums take, as the kernel's `FRICTION_CONE_FLOATS`: the
/// cone sum, the number of pairs in it, and the non-friction contact load.
const CONE_FLOATS: usize = 5;

/// Floats a memory entry holds, as the kernel's `FRICTION_MEMORY_SLIP`: the
/// carried slip and the slide of the step that left it.
const MEMORY_FLOATS: usize = 4;

/// The element limit a pair key leaves its first index,
/// `FRICTION_ELEMENT_LIMIT` in the kernel: the kind takes the top four bits.
pub const ELEMENT_LIMIT: usize = 1 << 28;

/// The memory as a checkpoint carries it (`friction_<N>.bin.gz`): the last
/// committed step's pairs in the sorted order the lookup bisects, four floats
/// each (the carried slip, then the slide of the step that left it), and each
/// vertex's sums at that commit, `CONE_FLOATS` floats a vertex, whose
/// non-friction contact load the next step's friction reads. A resumed run
/// installs it before its first step, so a stuck contact holds across the
/// resume as it holds across a step.
#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FrictionMemoryState {
    pub word0: Vec<u32>,
    pub word1: Vec<u32>,
    pub slip: Vec<f32>,
    pub committed_cone: Vec<f32>,
}

impl FrictionMemoryState {
    /// Whether this memory can be installed in a scene of `vertices`
    /// vertices: the lengths agree and the keys ascend strictly, since the
    /// lookup bisects them and a disorder would find the wrong pair's slip.
    /// A host check, so a resume can refuse a checkpoint before it discards
    /// anything (`main.rs`).
    pub fn check(&self, vertices: usize) -> FatalResult<()> {
        let state = self;
        let count = state.word0.len();
        if state.word1.len() != count || state.slip.len() != MEMORY_FLOATS * count {
            return Err(Fatal::invariant(format!(
                "solver driver: the checkpoint's friction memory holds {} first words, {} \
                 second words and {} slip floats, where one pair is one of each and \
                 {MEMORY_FLOATS} floats",
                count,
                state.word1.len(),
                state.slip.len()
            )));
        }
        if state.committed_cone.len() != CONE_FLOATS * vertices {
            return Err(Fatal::invariant(format!(
                "solver driver: the checkpoint's friction memory holds {} committed vertex \
                 sums where this scene's {} vertices take {CONE_FLOATS} each; it was written \
                 for another scene or by a solver that did not keep them",
                state.committed_cone.len(),
                vertices
            )));
        }
        if let Some(at) = (1..count).find(|&i| {
            (state.word0[i - 1], state.word1[i - 1]) >= (state.word0[i], state.word1[i])
        }) {
            return Err(Fatal::invariant(format!(
                "solver driver: the checkpoint's friction memory is not in ascending key order \
                 at entry {at}, so a contact's lookup could find another pair's slip"
            )));
        }
        Ok(())
    }
}

/// The Cosserat frames' handles the friction passes read
/// (`super::cosserat::State::frames_for_friction`): per mesh edge the frame slot
/// riding on it, the iterate's and the step's start frames, and the frames'
/// gradient and own blocks, which a framed pair's torque and stiffness join.
#[derive(Clone, Copy)]
pub struct FrictionFrames {
    pub edge_slot: Handle,
    pub quat: Handle,
    pub quat_start: Handle,
    pub gradient: Handle,
    pub self_block: Handle,
}

/// No frames at all, which only a memory not yet allocated holds:
/// `FrictionMemory::allocate` replaces it with its placeholders before any pass
/// can read it.
impl Default for FrictionFrames {
    fn default() -> Self {
        Self {
            edge_slot: Handle::NONE,
            quat: Handle::NONE,
            quat_start: Handle::NONE,
            gradient: Handle::NONE,
            self_block: Handle::NONE,
        }
    }
}

/// What the augmented solve's operator needs to apply a framed pair's
/// frozen friction across its frames matrix free (`friction_frame_apply`):
/// the step's start pose the Jacobian is read at, the frozen record and its
/// capacity, and each entry's tangential stiffness.
#[derive(Clone, Copy)]
pub struct FrameCoupling {
    pub x0: Handle,
    /// Per vertex, one where the Dirichlet pass removed its row
    /// (`vertex_dof_removal_mask`): the coupling stays off it.
    pub dof_mask: Handle,
    pub record_vertex: Handle,
    pub record_float: Handle,
    pub record_count: Handle,
    pub capacity: u32,
    pub frozen_stiffness: Handle,
}

/// The handles every contact pass takes, rebuilt each iteration because a
/// grown record moves its arrays.
#[derive(Clone, Copy)]
pub struct FrictionMemoryRefs {
    pub memory_word0: Handle,
    pub memory_word1: Handle,
    pub memory_slip: Handle,
    pub memory_count: u32,
    pub record_word0: Handle,
    pub record_word1: Handle,
    pub record_vertex: Handle,
    pub record_float: Handle,
    pub record_count: Handle,
    pub record_capacity: u32,
    /// The capacity the step's first iteration recorded into, over which the
    /// frozen pass reads the record.
    pub frozen_capacity: u32,
    /// The vertices' cone sums and loads of the last closed iteration.
    pub friction_cone: Handle,
    /// The load every friction term reads, three floats a vertex: the residual
    /// snapshot `open` was handed plus the vertex's lagged non-friction contact
    /// load (`friction_load_prepare`).
    pub friction_load: Handle,
    /// The scene's Cosserat frames (`FrictionMemory::set_frames`).
    pub frames: FrictionFrames,
    /// Per record entry, the frozen pass's tangential stiffness of a framed
    /// pair, which the matrix-free frame coupling reads.
    pub frozen_stiffness: Handle,
}

#[derive(Default)]
pub struct FrictionMemory {
    /// The scene's Cosserat frames, set once their buffers exist; until then
    /// (and on a memory no scene's frames are handed to) the placeholders below.
    frames: FrictionFrames,
    /// A one-word edge-to-frame map naming no frame and nine zero floats: the
    /// frames a memory reads before `set_frames`, which no entry can name.
    edge_none: Buffer<u32>,
    frame_none: Buffer<f32>,
    /// Per record entry, a framed pair's tangential stiffness, written by the
    /// frozen pass.
    frozen_stiffness: Buffer<f32>,
    memory_word0: Buffer<u32>,
    memory_word1: Buffer<u32>,
    memory_slip: Buffer<f32>,
    memory_count: usize,
    record_word0: Buffer<u32>,
    record_word1: Buffer<u32>,
    record_vertex: Buffer<u32>,
    record_float: Buffer<f32>,
    record_count: Buffer<u32>,
    capacity: usize,
    /// The vertices' lagged cone sums, [`CONE_FLOATS`] per vertex.
    cone: Buffer<f32>,
    /// The sums as the last commit left them; the load reads these.
    committed_cone: Buffer<f32>,
    /// The friction load `open` forms, three floats a vertex.
    load: Buffer<f32>,
    vertices: usize,
    /// The capacity the next iteration opens with.
    wanted: usize,
    /// The count the last iteration recorded, which may exceed `capacity`.
    recorded: usize,
    /// Whether the last iteration's record holds every pair it counted.
    complete: bool,
    /// The two sorts of the commit, by the second key word and then, stably,
    /// by the first; the first's permutation is read after the second runs,
    /// so they cannot share a scratch.
    by_second: SortScratch,
    by_first: SortScratch,
    gathered: Buffer<u32>,
    /// The committed pairs in key order before equal keys are merged, and the
    /// flags (then, scanned, the slots) that merge them; `head` holds one
    /// more slot than there are pairs, for the count the scan writes there.
    sorted_word0: Buffer<u32>,
    sorted_word1: Buffer<u32>,
    sorted_slip: Buffer<f32>,
    sorted_force: Buffer<f32>,
    head: Buffer<u32>,
    slot: Buffer<u32>,
    scan: ScanScratch,
    /// The frozen pass's dynamic slots where there is no dynamic matrix (a
    /// scene with `disable-contact`): one claim word read after the pass, and
    /// rows, columns and a block it can never write, since their capacity is
    /// zero. Every pair such a scene records is analytic, whose blocks the
    /// fixed pattern holds, so a nonzero claim is a fatal.
    fixed_only_claim: Buffer<u32>,
    fixed_only_index: Buffer<u32>,
    fixed_only_block: Buffer<f32>,
}

impl FrictionMemory {
    /// Size the memory for a scene, once, at `initialize()`.
    ///
    /// `floor` is the record's starting capacity, a count the scene's own
    /// sizes make likely to suffice (the capacity grows past it on demand),
    /// and zero for a scene with no friction anywhere, whose passes never
    /// record. The arrays are allocated either way, because a generated entry
    /// resolves every buffer it names whether or not the body reads it.
    /// `vertices` sizes the cone sums, which open at zero: a vertex with no
    /// sum gives every pair its whole residual.
    pub fn allocate(device: &mut impl Device, floor: usize, vertices: usize) -> FatalResult<Self> {
        let mut memory = Self { complete: true, vertices, ..Self::default() };
        memory.record_count.size(device, 1, AllocLabel("friction_memory.record_count"))?;
        memory.fixed_only_claim.size(device, 1, AllocLabel("friction_memory.fixed_only_claim"))?;
        memory.fixed_only_index.size(device, 1, AllocLabel("friction_memory.fixed_only_index"))?;
        memory.fixed_only_block.size(device, 9, AllocLabel("friction_memory.fixed_only_block"))?;
        memory.cone.size(device, CONE_FLOATS * vertices.max(1), AllocLabel("friction_memory.cone"))?;
        device.fill_zero(memory.cone.handle(), CONE_FLOATS * vertices.max(1) * std::mem::size_of::<f32>())?;
        memory.committed_cone.size(device, CONE_FLOATS * vertices.max(1), AllocLabel("friction_memory.committed_cone"))?;
        device.fill_zero(memory.committed_cone.handle(), CONE_FLOATS * vertices.max(1) * std::mem::size_of::<f32>())?;
        memory.load.size(device, 3 * vertices.max(1), AllocLabel("friction_memory.load"))?;
        device.fill_zero(memory.load.handle(), 3 * vertices.max(1) * std::mem::size_of::<f32>())?;
        memory.edge_none.size(device, 1, AllocLabel("friction_memory.edge_none"))?;
        memory.edge_none.write(device, 0, &[u32::MAX])?;
        memory.frame_none.size(device, 9, AllocLabel("friction_memory.frame_none"))?;
        device.fill_zero(memory.frame_none.handle(), 9 * std::mem::size_of::<f32>())?;
        let none = memory.frame_none.handle();
        memory.frames = FrictionFrames {
            edge_slot: memory.edge_none.handle(),
            quat: none,
            quat_start: none,
            gradient: none,
            self_block: none,
        };
        memory.size_record(device, floor)?;
        memory.wanted = floor;
        memory.size_memory(device, 0)?;
        Ok(memory)
    }

    /// The frame coupling of this step's frozen record against the start pose
    /// `x0`, for a scene whose friction can name a frame; None where it cannot
    /// (no frames, or no record), and the augmented solve adds nothing.
    pub fn frame_coupling(&self, x0: Handle, dof_mask: Handle, framed: bool) -> Option<FrameCoupling> {
        if !framed || self.capacity == 0 {
            return None;
        }
        Some(FrameCoupling {
            x0,
            dof_mask,
            record_vertex: self.record_vertex.handle(),
            record_float: self.record_float.handle(),
            record_count: self.record_count.handle(),
            capacity: self.capacity as u32,
            frozen_stiffness: self.frozen_stiffness.handle(),
        })
    }

    /// Hand over the scene's Cosserat frames, once their buffers exist and
    /// whenever they are reallocated: every pass that reads a framed pair's
    /// slip takes them from here.
    pub fn set_frames(&mut self, frames: FrictionFrames) {
        self.frames = frames;
    }

    fn size_record(&mut self, device: &mut impl Device, capacity: usize) -> FatalResult<()> {
        let length = capacity.max(1);
        self.record_word0.size(device, length, AllocLabel("friction_memory.record_word0"))?;
        self.record_word1.size(device, length, AllocLabel("friction_memory.record_word1"))?;
        self.record_vertex.size(
            device,
            RECORD_WORDS * length,
            AllocLabel("friction_memory.record_vertex"),
        )?;
        self.frozen_stiffness.size(device, length, AllocLabel("friction_memory.frozen_stiffness"))?;
        self.record_float.size(
            device,
            RECORD_FLOATS * length,
            AllocLabel("friction_memory.record_float"),
        )?;
        self.capacity = capacity;
        Ok(())
    }

    fn size_memory(&mut self, device: &mut impl Device, count: usize) -> FatalResult<()> {
        let length = count.max(1);
        self.memory_word0.size(device, length, AllocLabel("friction_memory.word0"))?;
        self.memory_word1.size(device, length, AllocLabel("friction_memory.word1"))?;
        self.memory_slip.size(device, MEMORY_FLOATS * length, AllocLabel("friction_memory.slip"))?;
        Ok(())
    }

    /// The committed memory, read back for a checkpoint.
    pub fn snapshot(&self, device: &mut impl Device) -> FatalResult<FrictionMemoryState> {
        let count = self.memory_count;
        let mut state = FrictionMemoryState {
            word0: vec![0; count],
            word1: vec![0; count],
            slip: vec![0.0; MEMORY_FLOATS * count],
            committed_cone: vec![0.0; CONE_FLOATS * self.vertices],
        };
        if self.vertices > 0 {
            self.committed_cone.read(device, 0, &mut state.committed_cone)?;
        }
        if count > 0 {
            self.memory_word0.read(device, 0, &mut state.word0)?;
            self.memory_word1.read(device, 0, &mut state.word1)?;
            self.memory_slip.read(device, 0, &mut state.slip)?;
        }
        Ok(state)
    }

    /// Install a checkpoint's memory, which must be what [`Self::snapshot`]
    /// wrote: the lengths agree and the keys ascend strictly, since the lookup
    /// bisects them and a disorder would find the wrong pair's slip.
    pub fn restore(
        &mut self,
        device: &mut impl Device,
        state: &FrictionMemoryState,
    ) -> FatalResult<()> {
        state.check(self.vertices)?;
        let count = state.word0.len();
        self.size_memory(device, count)?;
        if count > 0 {
            self.memory_word0.write(device, 0, &state.word0)?;
            self.memory_word1.write(device, 0, &state.word1)?;
            self.memory_slip.write(device, 0, &state.slip)?;
        }
        // THE LAGGED CONE SUMS TOO, not only the committed copy. The first
        // Newton iteration of a step reads its cone sums and pair counts from
        // `cone`, the last iteration's close, before any close of its own; a
        // commit copies `cone` into `committed_cone` and nothing writes either
        // again before that read, so at a checkpoint the two are equal. Left at
        // its allocation's zeros, the resumed step's first iteration would
        // give every pair the whole vertex's share where the uninterrupted run
        // splits it.
        if self.vertices > 0 {
            self.committed_cone.write(device, 0, &state.committed_cone)?;
            self.cone.write(device, 0, &state.committed_cone)?;
        }
        self.memory_count = count;
        Ok(())
    }

    /// Open an iteration's record, form the load every friction term reads from
    /// the iteration's residual snapshot and the last commit's non-friction
    /// contact load (`friction_load_prepare`, which says why it is formed here
    /// rather than by each pair), and hand out the handles.
    ///
    /// A step's `first_iteration` records: the record grows to what the last
    /// count asked for and its count is cleared. Any later iteration leaves the
    /// record as it is and hands out a capacity of zero, so the passes write
    /// nothing and the record stays the step's frozen friction. The one
    /// exception is a record that overflowed: it is missing pairs, so the next
    /// iteration records again, at its own pose, into the grown capacity.
    ///
    /// # Safety
    /// `residual` must name three floats per vertex of the scene the memory
    /// was sized for.
    pub unsafe fn open(
        &mut self,
        device: &mut impl Device,
        residual: Handle,
        first_iteration: bool,
    ) -> FatalResult<FrictionMemoryRefs> {
        let recording = first_iteration || !self.complete;
        if recording {
            if self.wanted > self.capacity {
                self.size_record(device, self.wanted)?;
            }
            device.fill_zero(self.record_count.handle(), std::mem::size_of::<u32>())?;
        }
        if self.vertices > 0 {
            let prepare = FrictionLoadPrepareArgs {
                residual,
                cone: self.committed_cone.handle(),
                load: self.load.handle(),
                count: self.vertices as u32,
                seam_arena_count: 0,
            };
            // Safety: the caller's residual and both of ours hold the scene's
            // vertices.
            unsafe { device.launch("friction_memory.load_prepare", &prepare, self.vertices as u32) }?;
        }
        Ok(FrictionMemoryRefs {
            memory_word0: self.memory_word0.handle(),
            memory_word1: self.memory_word1.handle(),
            memory_slip: self.memory_slip.handle(),
            memory_count: self.memory_count as u32,
            record_word0: self.record_word0.handle(),
            record_word1: self.record_word1.handle(),
            record_vertex: self.record_vertex.handle(),
            record_float: self.record_float.handle(),
            record_count: self.record_count.handle(),
            record_capacity: if recording { self.capacity as u32 } else { 0 },
            frozen_capacity: self.capacity as u32,
            friction_cone: self.cone.handle(),
            friction_load: self.load.handle(),
            frames: self.frames,
            frozen_stiffness: self.frozen_stiffness.handle(),
        })
    }

    /// Close an iteration's record: read how many pairs it counted, ask the
    /// next iteration for twice that once it passes half the capacity, and
    /// gather the pairs it wrote into the vertices' cone sums and, with their
    /// friction recomputed at the iterate `x` against the step's start `x0`,
    /// the assembled `force` less the contact passes' starting `residual` and
    /// that friction into each vertex's non-friction contact load, both of
    /// which the next iteration reads. The record itself is left as it is, for
    /// the commit.
    ///
    /// # Safety
    /// `x0` and `x` must name the scene's vertex positions and `force` and
    /// `residual` three floats a vertex of it.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn close(
        &mut self,
        device: &mut impl Device,
        x0: Handle,
        x: Handle,
        friction_eps: f32,
        force: Handle,
        residual: Handle,
    ) -> FatalResult<()> {
        self.recorded = self.record_count.read_one(device, 0)? as usize;
        self.complete = self.recorded <= self.capacity;
        if 2 * self.recorded > self.capacity {
            self.wanted = self.wanted.max(2 * self.recorded);
        }
        device.fill_zero(
            self.cone.handle(),
            CONE_FLOATS * self.vertices.max(1) * std::mem::size_of::<f32>(),
        )?;
        let written = self.recorded.min(self.capacity);
        if written > 0 {
            let gather = FrictionConeGatherArgs {
                x0,
                x,
                record_word0: self.record_word0.handle(),
                record_vertex: self.record_vertex.handle(),
                record_float: self.record_float.handle(),
                friction_eps,
                quat: self.frames.quat,
                quat_start: self.frames.quat_start,
                cone: self.cone.handle(),
                count: written as u32,
                seam_arena_count: 0,
            };
            // Safety: the record holds `written` entries, each naming vertices
            // below the scene's count the cone sums were sized for.
            unsafe { device.launch("friction_memory.cone_gather", &gather, written as u32) }?;
        }
        if self.vertices > 0 {
            let settle = FrictionLoadSettleArgs {
                force,
                residual,
                cone: self.cone.handle(),
                count: self.vertices as u32,
                seam_arena_count: 0,
            };
            // Safety: the caller's two arrays and both of ours hold the scene's
            // vertices.
            unsafe { device.launch("friction_memory.load_settle", &settle, self.vertices as u32) }?;
        }
        Ok(())
    }

    /// The step's friction where the scene has no dynamic matrix
    /// (`disable-contact`): [`launch_frozen`] with no dynamic slots. Every pair
    /// such a scene records is an analytic contact, whose blocks the fixed
    /// pattern or the block diagonal holds, so a block it could not place is a defect and stops the
    /// step rather than dropping a coupling.
    ///
    /// # Safety
    /// As [`launch_frozen`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn embed_frozen_fixed_only(
        &mut self,
        device: &mut impl Device,
        refs: &FrictionMemoryRefs,
        x0: Handle,
        x: Handle,
        friction_eps: f32,
        fixed: (Handle, Handle, Handle, u32),
        force: Handle,
        diagonal: Handle,
    ) -> FatalResult<()> {
        device.fill_zero(self.fixed_only_claim.handle(), std::mem::size_of::<u32>())?;
        let staging = DeviceStaging {
            claim: self.fixed_only_claim.handle(),
            row: self.fixed_only_index.handle(),
            column: self.fixed_only_index.handle(),
            block: self.fixed_only_block.handle(),
            capacity: 0,
        };
        // Safety: the caller's handles as `launch_frozen` asks; ours are
        // allocated and never written at capacity zero.
        unsafe { launch_frozen(device, refs, x0, x, friction_eps, fixed, force, diagonal, &staging) }?;
        let unplaced = self.fixed_only_claim.read_one(device, 0)?;
        if unplaced > 0 {
            return Err(Fatal::invariant(format!(
                "solver driver: the frozen friction pass offered {unplaced} blocks the fixed \
                 pattern has no slot for, on a scene with no dynamic matrix; every pair such a \
                 scene records is analytic, so a pair outside the pattern was recorded"
            )));
        }
        Ok(())
    }

    /// Each vertex's non-friction contact load from the last close, three
    /// floats a vertex, for tests.
    #[cfg(test)]
    pub fn loads(&self, device: &mut impl Device) -> Vec<f32> {
        let mut cone = vec![0f32; CONE_FLOATS * self.vertices];
        self.cone.read(device, 0, &mut cone).unwrap();
        cone.chunks(CONE_FLOATS).flat_map(|c| c[2..5].to_vec()).collect()
    }

    /// Turn the last iteration's record into the memory the next step reads,
    /// its slips evaluated at the committed pose `x` against the step's start
    /// `x0`.
    ///
    /// # Safety
    /// `x0` and `x` must name the scene's vertex positions.
    pub unsafe fn commit<D: Device>(
        &mut self,
        device: &mut D,
        x0: Handle,
        x: Handle,
        friction_eps: f32,
    ) -> FatalResult<()> {
        if !self.complete {
            return Err(Fatal::invariant(format!(
                "solver driver: the friction memory's last record counted {} frictional \
                 pair(s) against a capacity of {}, so {} of them were not written. Committing \
                 it would restart those contacts from zero slip; the capacity grows to twice \
                 any count past half of it, so this is a count that more than doubled between \
                 the step's last two Newton iterations",
                self.recorded,
                self.capacity,
                self.recorded - self.capacity
            )));
        }
        // THE LOAD THE NEXT STEP'S FRICTION READS IS THIS COMMIT'S
        // (`friction_load_prepare` says why), whether or not any pair is left.
        {
            let Self { committed_cone, cone, .. } = self;
            committed_cone.copy_from(device, cone)?;
        }
        let count = self.recorded;
        if count == 0 {
            self.memory_count = 0;
            return Ok(());
        }
        let length = count.max(1);
        // EVERY ALLOCATION BEFORE ANY SPAN, for the reason `SortScratch::sort`
        // gives: a later `size` may move a block an earlier handle names. The
        // scan sizes its own scratch too, so it is sized here first, which
        // makes its own call a no-op, and the memory is sized for every pair
        // being distinct, which the merge can only reduce.
        self.size_memory(device, count)?;
        self.gathered.size(device, length, AllocLabel("friction_memory.gathered"))?;
        self.sorted_word0.size(device, length, AllocLabel("friction_memory.sorted_word0"))?;
        self.sorted_word1.size(device, length, AllocLabel("friction_memory.sorted_word1"))?;
        self.sorted_slip
            .size(device, MEMORY_FLOATS * length, AllocLabel("friction_memory.sorted_slip"))?;
        self.sorted_force.size(device, length, AllocLabel("friction_memory.sorted_force"))?;
        self.head.size(device, length, AllocLabel("friction_memory.head"))?;
        self.slot.size(device, length + 1, AllocLabel("friction_memory.slot"))?;
        self.scan.size(device, count as u32)?;
        // THE ORDER IS LEXICOGRAPHIC IN (word0, word1), the order the lookup
        // bisects: a stable radix sort by the second word, then a stable one by
        // the first over the order the first left.
        let (_, by_second) =
            self.by_second.sort(device, "friction_memory.sort_second", self.record_word1.handle(), count)?;
        let gather = FrictionMemoryGatherWordArgs {
            word: self.record_word0.handle(),
            order: by_second,
            out: self.gathered.handle(),
            count: count as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer the record names outlives the call.
        unsafe { device.launch("friction_memory.gather_word", &gather, count as u32) }?;
        let (_, by_first) =
            self.by_first.sort(device, "friction_memory.sort_first", self.gathered.handle(), count)?;
        let settle = FrictionMemorySettleArgs {
            x0,
            x,
            record_word0: self.record_word0.handle(),
            record_word1: self.record_word1.handle(),
            record_vertex: self.record_vertex.handle(),
            record_float: self.record_float.handle(),
            first: by_second,
            second: by_first,
            friction_eps,
            quat: self.frames.quat,
            quat_start: self.frames.quat_start,
            sorted_word0: self.sorted_word0.handle(),
            sorted_word1: self.sorted_word1.handle(),
            sorted_slip: self.sorted_slip.handle(),
            sorted_force: self.sorted_force.handle(),
            count: count as u32,
            seam_arena_count: 0,
        };
        // Safety: as above, and `x0` and `x` are the caller's.
        unsafe { device.launch("friction_memory.settle", &settle, count as u32) }?;
        // ONE ENTRY PER KEY: flag each run's first entry, scan the flags into
        // slots (the scan writes the run count one past the end), and keep one
        // entry per run. The flags stay for the merge to find a run's end by,
        // so the scan works on a copy.
        let head = FrictionMemoryHeadArgs {
            sorted_word0: self.sorted_word0.handle(),
            sorted_word1: self.sorted_word1.handle(),
            head: self.head.handle(),
            count: count as u32,
            seam_arena_count: 0,
        };
        // Safety: as above.
        unsafe { device.launch("friction_memory.head", &head, count as u32) }?;
        device.copy(
            self.slot.handle(),
            0,
            self.head.handle(),
            0,
            count * std::mem::size_of::<u32>(),
        )?;
        // Safety: `slot` holds `count + 1` words and outlives the call.
        unsafe { self.scan.exclusive(device, "friction_memory.scan", self.slot.handle(), count as u32) }?;
        let unique = self.slot.read_one(device, count)? as usize;
        let merge = FrictionMemoryMergeArgs {
            sorted_word0: self.sorted_word0.handle(),
            sorted_word1: self.sorted_word1.handle(),
            sorted_slip: self.sorted_slip.handle(),
            sorted_force: self.sorted_force.handle(),
            head: self.head.handle(),
            slot: self.slot.handle(),
            entries: count as u32,
            memory_word0: self.memory_word0.handle(),
            memory_word1: self.memory_word1.handle(),
            memory_slip: self.memory_slip.handle(),
            count: count as u32,
            seam_arena_count: 0,
        };
        // Safety: as above.
        unsafe { device.launch("friction_memory.merge", &merge, count as u32) }?;
        self.memory_count = unique;
        Ok(())
    }
}

/// THE STEP'S FRICTION, applied from the frozen record (`friction_frozen_embed`,
/// `contact/contact_narrow.kernel.cpp`): every iteration of a step, after the
/// contact passes, into the same force vector and matrix they assemble. A
/// vertex's own block goes to the block diagonal, a coupling to the fixed
/// pattern, and one outside it to `staging`, the dynamic matrix's slots.
///
/// # Safety
/// `x0` and `x` must name the scene's vertex positions, `force` three floats a
/// vertex, `diagonal` the operator's block diagonal, nine floats a vertex,
/// `fixed` the fixed matrix's index, offset, value and row count, and
/// `staging` live slots of its stated capacity.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_frozen(
    device: &mut impl Device,
    refs: &FrictionMemoryRefs,
    x0: Handle,
    x: Handle,
    friction_eps: f32,
    fixed: (Handle, Handle, Handle, u32),
    force: Handle,
    diagonal: Handle,
    staging: &DeviceStaging,
) -> FatalResult<()> {
    let entries = refs.frozen_capacity;
    if entries == 0 {
        return Ok(());
    }
    let (fixed_index, fixed_offset, out_fixed_value, row_count) = fixed;
    let args = FrictionFrozenEmbedArgs {
        x0,
        x,
        record_word0: refs.record_word0,
        record_vertex: refs.record_vertex,
        record_float: refs.record_float,
        record_count: refs.record_count,
        record_capacity: entries,
        friction_eps,
        fixed_index,
        fixed_offset,
        row_count,
        out_vertex_force: force,
        out_diagonal: diagonal,
        out_fixed_value,
        dyn_claim: staging.claim,
        dyn_row: staging.row,
        dyn_column: staging.column,
        dyn_block: staging.block,
        dyn_capacity: staging.capacity,
        quat: refs.frames.quat,
        quat_start: refs.frames.quat_start,
        frame_gradient: refs.frames.gradient,
        frame_self_block: refs.frames.self_block,
        frozen_stiffness: refs.frozen_stiffness,
        count: entries,
        seam_arena_count: 0,
    };
    // Safety: the record holds `entries` slots and the caller vouches for the rest.
    unsafe { device.launch("friction_memory.frozen_embed", &args, entries) }?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;

    /// A test's record words in the vertices-only layout, each entry's four
    /// vertices followed by two unused frame slots.
    fn pad_words(vertex: &[u32]) -> Vec<u32> {
        vertex
            .chunks(RECORD_VERTICES)
            .flat_map(|entry| entry.iter().copied().chain([u32::MAX; RECORD_WORDS - RECORD_VERTICES]))
            .collect()
    }

    /// A test's record floats in the frameless layout, each entry's fourteen
    /// floats followed by two zero side radii.
    fn pad_floats(float: &[f32]) -> Vec<f32> {
        float.chunks(RECORD_FLOATS - 2).flat_map(|entry| entry.iter().copied().chain([0.0; 2])).collect()
    }

    /// Open an iteration against a zero residual snapshot.
    fn open(memory: &mut FrictionMemory, device: &mut impl Device) -> FatalResult<FrictionMemoryRefs> {
        let floats = 3 * memory.vertices.max(1);
        let mut residual = Buffer::<f32>::none();
        residual.size(device, floats, AllocLabel("test.residual")).unwrap();
        device.fill_zero(residual.handle(), floats * std::mem::size_of::<f32>()).unwrap();
        // Safety: the residual holds three floats per vertex.
        unsafe { memory.open(device, residual.handle(), true) }
    }

    /// Close an iteration with nothing moved and a zero force and residual.
    fn close(memory: &mut FrictionMemory, device: &mut impl Device) {
        let floats = 3 * memory.vertices.max(1);
        let mut zero = Buffer::<f32>::none();
        zero.size(device, floats, AllocLabel("test.zero")).unwrap();
        device.fill_zero(zero.handle(), floats * std::mem::size_of::<f32>()).unwrap();
        let mut pose = Buffer::<f32>::none();
        pose.size(device, floats, AllocLabel("test.pose")).unwrap();
        device.fill_zero(pose.handle(), floats * std::mem::size_of::<f32>()).unwrap();
        // Safety: every array holds the scene's vertices.
        unsafe {
            memory.close(device, pose.handle(), pose.handle(), 1e-5, zero.handle(), zero.handle())
        }
        .unwrap();
    }

    /// Two crossing edges a gap apart: turning the pair as one body leaves it no
    /// slip beyond the second order of the angle, where the bare relative
    /// displacement reads the angle times the gap; sliding one edge along
    /// itself keeps the whole slide.
    #[test]
    fn a_pair_turning_as_one_body_does_not_slip_and_a_slide_does() {
        let mut device = host_device();
        let mut memory = FrictionMemory::allocate(&mut device, 8, 8).unwrap();
        let _ = open(&mut memory, &mut device).unwrap();
        let kind = 4u32 << 28; // FRICTION_KIND_EDGE_EDGE
        let word0 = [kind, kind | 1];
        let word1 = [0u32, 1];
        let vertex = [0u32, 1, 2, 3, 4, 5, 6, 7];
        // Edge a along x at height `gap` over edge b along y; the closest
        // points are both midpoints, so the weights are a half each.
        let pair = [0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, -0.5, 1.0, 0.5, 1.0, 1.0];
        let float = [pair, pair].concat();
        memory.record_word0.write(&mut device, 0, &word0).unwrap();
        memory.record_word1.write(&mut device, 0, &word1).unwrap();
        memory.record_vertex.write(&mut device, 0, &pad_words(&vertex)).unwrap();
        memory.record_float.write(&mut device, 0, &pad_floats(&float)).unwrap();
        memory.record_count.write(&mut device, 0, &[2]).unwrap();
        close(&mut memory, &mut device);
        let gap = 0.01f64;
        let cross = [[-1.0, 0.0, gap], [1.0, 0.0, gap], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        // Pair 0 turns by `angle` about a skew axis through the origin
        // (Rodrigues); pair 1 has edge a slide `slide` along x.
        let angle = 1e-3f64;
        let axis = {
            let a = [1.0f64, 2.0, 3.0];
            let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            a.map(|c| c / n)
        };
        let turn = |p: [f64; 3]| {
            let (c, s) = (angle.cos(), angle.sin());
            let kxp = [
                axis[1] * p[2] - axis[2] * p[1],
                axis[2] * p[0] - axis[0] * p[2],
                axis[0] * p[1] - axis[1] * p[0],
            ];
            let kdp = axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2];
            [0, 1, 2].map(|d| p[d] * c + kxp[d] * s + axis[d] * kdp * (1.0 - c))
        };
        let slide = 1e-5f64;
        // Float positions round to about 1e-7 of their size, so the scene is
        // shrunk a thousandfold; the bare measure then reads 1e-8 for the turn.
        let scale = 1e-3f64;
        let (turn_tol, slide_tol) = (1e-9f32, 1e-9f32);
        let pos = |p: [f64; 3]| p.map(|x| (x * scale) as f32);
        let mut start = Vec::new();
        let mut pose = Vec::new();
        for p in cross {
            start.extend(pos(p));
            pose.extend(pos(turn(p)));
        }
        for (k, p) in cross.iter().enumerate() {
            start.extend(pos(*p));
            let moved = if k < 2 { [p[0] + slide, p[1], p[2]] } else { *p };
            pose.extend(pos(moved));
        }
        let mut x0 = Buffer::<f32>::none();
        let mut x = Buffer::<f32>::none();
        x0.size(&mut device, 24, AllocLabel("test.x0")).unwrap();
        x.size(&mut device, 24, AllocLabel("test.x")).unwrap();
        x0.write(&mut device, 0, &start).unwrap();
        x.write(&mut device, 0, &pose).unwrap();
        // A stick range far above both slips, so the return map keeps them whole.
        // Safety: both pose buffers hold the eight vertices the records name.
        unsafe { memory.commit(&mut device, x0.handle(), x.handle(), 1.0) }.unwrap();
        let mut slip = [0f32; 8];
        memory.memory_slip.read(&mut device, 0, &mut slip).unwrap();
        let norm = |v: &[f32]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        // The bare measure would read angle * gap * scale for the turn.
        let turned = norm(&slip[0..3]);
        assert!(turned < turn_tol, "a rigid turn slipped {turned}");
        let slid = &slip[MEMORY_FLOATS..MEMORY_FLOATS + 3];
        assert!((slid[0] - (slide * scale) as f32).abs() < slide_tol, "slide {slid:?}");
        assert!(slid[1].abs() < slide_tol && slid[2].abs() < slide_tol, "slide {slid:?}");
    }

    /// A slip carried into the step turns with the pair: two crossing edges
    /// holding a carried slip along x, pivoted together about their normal,
    /// commit that slip turned by the pivot, where re-projecting it alone would
    /// leave it pointing where the pair no longer slides.
    #[test]
    fn a_carried_slip_turns_with_its_pair() {
        let mut device = host_device();
        let mut memory = FrictionMemory::allocate(&mut device, 8, 4).unwrap();
        let _ = open(&mut memory, &mut device).unwrap();
        let carried = 1e-3f32;
        let pair = [carried, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, -0.5, 1.0, 0.5, 1.0, 1.0];
        memory.record_word0.write(&mut device, 0, &[4u32 << 28]).unwrap(); // FRICTION_KIND_EDGE_EDGE
        memory.record_word1.write(&mut device, 0, &[0u32]).unwrap();
        memory.record_vertex.write(&mut device, 0, &pad_words(&[0u32, 1, 2, 3])).unwrap();
        memory.record_float.write(&mut device, 0, &pad_floats(&pair)).unwrap();
        memory.record_count.write(&mut device, 0, &[1]).unwrap();
        close(&mut memory, &mut device);
        let gap = 0.01f64;
        let cross = [[-1.0, 0.0, gap], [1.0, 0.0, gap], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        // A pivot about z through the origin, which is the pair's own normal
        // axis, so the relative step of the two midpoints is zero.
        let angle = 1e-2f64;
        let pivot = |p: [f64; 3]| {
            let (c, s) = (angle.cos(), angle.sin());
            [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
        };
        let pos = |p: [f64; 3]| p.map(|x| x as f32);
        let start: Vec<f32> = cross.iter().flat_map(|p| pos(*p)).collect();
        let pose: Vec<f32> = cross.iter().flat_map(|p| pos(pivot(*p))).collect();
        let mut x0 = Buffer::<f32>::none();
        let mut x = Buffer::<f32>::none();
        x0.size(&mut device, 12, AllocLabel("test.x0")).unwrap();
        x.size(&mut device, 12, AllocLabel("test.x")).unwrap();
        x0.write(&mut device, 0, &start).unwrap();
        x.write(&mut device, 0, &pose).unwrap();
        // Safety: both pose buffers hold the four vertices the record names.
        unsafe { memory.commit(&mut device, x0.handle(), x.handle(), 1.0) }.unwrap();
        let mut slip = [0f32; 4];
        memory.memory_slip.read(&mut device, 0, &mut slip).unwrap();
        // The carried slip turned by the pivot, to first order in the angle.
        let want = [carried, carried * angle as f32, 0.0];
        for d in 0..3 {
            assert!((slip[d] - want[d]).abs() < 1e-7, "axis {d}: {:?} against {want:?}", &slip[..3]);
        }
    }

    /// A side with a Cosserat frame slips by its surface: two crossing edges
    /// held still while side a's frame rolls by `angle` about its own axis,
    /// with side b's edge, which has no frame, unmoved. The committed slip is
    /// side a's surface's step, `r_a n x theta_a`, where the pair rule would
    /// read no motion at all.
    #[test]
    fn a_framed_side_slips_by_its_surface() {
        let mut device = host_device();
        let mut memory = FrictionMemory::allocate(&mut device, 8, 4).unwrap();
        let (radius_a, radius_b, angle) = (1e-2f32, 2e-2f32, 1e-3f32);
        // The step's start frame is the identity; the iterate's has rolled by
        // `angle` about x, edge a's direction.
        let mut quat_start = Buffer::<f32>::none();
        let mut quat = Buffer::<f32>::none();
        let mut scratch = Buffer::<f32>::none();
        quat_start.size(&mut device, 4, AllocLabel("test.quat_start")).unwrap();
        quat.size(&mut device, 4, AllocLabel("test.quat")).unwrap();
        scratch.size(&mut device, 9, AllocLabel("test.scratch")).unwrap();
        quat_start.write(&mut device, 0, &[0.0, 0.0, 0.0, 1.0]).unwrap();
        quat.write(&mut device, 0, &[(0.5 * angle).sin(), 0.0, 0.0, (0.5 * angle).cos()]).unwrap();
        let none = memory.frames.edge_slot;
        memory.set_frames(FrictionFrames {
            edge_slot: none,
            quat: quat.handle(),
            quat_start: quat_start.handle(),
            gradient: scratch.handle(),
            self_block: scratch.handle(),
        });
        let _ = open(&mut memory, &mut device).unwrap();
        // Edge a along x over edge b along y, the normal +z from b toward a.
        let pair = [0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, -0.5, 1.0, 0.5, 1.0, 1.0, radius_a, radius_b];
        memory.record_word0.write(&mut device, 0, &[4u32 << 28]).unwrap(); // FRICTION_KIND_EDGE_EDGE
        memory.record_word1.write(&mut device, 0, &[0u32]).unwrap();
        memory.record_vertex.write(&mut device, 0, &[0u32, 1, 2, 3, 0, u32::MAX]).unwrap();
        memory.record_float.write(&mut device, 0, &pair).unwrap();
        memory.record_count.write(&mut device, 0, &[1]).unwrap();
        close(&mut memory, &mut device);
        let gap = 0.03f64;
        let cross = [[-1.0, 0.0, gap], [1.0, 0.0, gap], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        let pos = |p: [f64; 3]| p.map(|x| x as f32);
        let still: Vec<f32> = cross.iter().flat_map(|p| pos(*p)).collect();
        let mut x = Buffer::<f32>::none();
        x.size(&mut device, 12, AllocLabel("test.x")).unwrap();
        x.write(&mut device, 0, &still).unwrap();
        // Safety: the pose buffer holds the four vertices the record names.
        unsafe { memory.commit(&mut device, x.handle(), x.handle(), 1.0) }.unwrap();
        let mut slip = [0f32; 4];
        memory.memory_slip.read(&mut device, 0, &mut slip).unwrap();
        // n x theta_a = z x (angle x) = angle y.
        let want = [0.0, radius_a * angle, 0.0];
        for d in 0..3 {
            assert!((slip[d] - want[d]).abs() < 1e-9, "axis {d}: {:?} against {want:?}", &slip[..3]);
        }
    }

    /// A framed pair turned rigidly does not slip: both sides carry frames, and
    /// the four vertices and both frames turn together by one small rotation.
    /// The centerlines' relative step, `phi u x (c_a - c_b)`, and the frames'
    /// surface steps, `(r_a + r_b) n x phi u`, cancel to first order, so the
    /// committed slip is of second order in `phi`, whatever the axis.
    #[test]
    fn a_framed_pair_turned_rigidly_does_not_slip() {
        let (radius_a, radius_b, angle) = (1e-2f64, 2e-2f64, 1e-3f64);
        let gap = radius_a + radius_b;
        let s3 = 1.0 / 3.0f64.sqrt();
        for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [s3, s3, s3]] {
            let mut device = host_device();
            let mut memory = FrictionMemory::allocate(&mut device, 8, 4).unwrap();
            let half = 0.5 * angle;
            let turned = [axis[0] * half.sin(), axis[1] * half.sin(), axis[2] * half.sin(), half.cos()].map(|x| x as f32);
            let mut quat_start = Buffer::<f32>::none();
            let mut quat = Buffer::<f32>::none();
            let mut scratch = Buffer::<f32>::none();
            quat_start.size(&mut device, 8, AllocLabel("test.quat_start")).unwrap();
            quat.size(&mut device, 8, AllocLabel("test.quat")).unwrap();
            scratch.size(&mut device, 18, AllocLabel("test.scratch")).unwrap();
            quat_start.write(&mut device, 0, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
            quat.write(&mut device, 0, &[turned, turned].concat()).unwrap();
            let none = memory.frames.edge_slot;
            memory.set_frames(FrictionFrames {
                edge_slot: none,
                quat: quat.handle(),
                quat_start: quat_start.handle(),
                gradient: scratch.handle(),
                self_block: scratch.handle(),
            });
            let _ = open(&mut memory, &mut device).unwrap();
            // Edge a along x over edge b along y, the normal +z from b toward a.
            let pair = [0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, -0.5, 1.0, 0.5, 1.0, 1.0, radius_a as f32, radius_b as f32];
            memory.record_word0.write(&mut device, 0, &[4u32 << 28]).unwrap(); // FRICTION_KIND_EDGE_EDGE
            memory.record_word1.write(&mut device, 0, &[0u32]).unwrap();
            memory.record_vertex.write(&mut device, 0, &[0u32, 1, 2, 3, 0, 1]).unwrap();
            memory.record_float.write(&mut device, 0, &pair).unwrap();
            memory.record_count.write(&mut device, 0, &[1]).unwrap();
            close(&mut memory, &mut device);
            let cross = [[-1.0, 0.0, gap], [1.0, 0.0, gap], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
            // Rodrigues: R p = p cos + (u x p) sin + u (u . p)(1 - cos)
            let rotate = |p: [f64; 3]| {
                let (c, s) = (angle.cos(), angle.sin());
                let d = axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2];
                let x = [axis[1] * p[2] - axis[2] * p[1], axis[2] * p[0] - axis[0] * p[2], axis[0] * p[1] - axis[1] * p[0]];
                [0, 1, 2].map(|i| p[i] * c + x[i] * s + axis[i] * d * (1.0 - c))
            };
            let pos = |p: [f64; 3]| p.map(|x| x as f32);
            let start: Vec<f32> = cross.iter().flat_map(|p| pos(*p)).collect();
            let moved: Vec<f32> = cross.iter().flat_map(|p| pos(rotate(*p))).collect();
            let mut x0 = Buffer::<f32>::none();
            let mut x = Buffer::<f32>::none();
            x0.size(&mut device, 12, AllocLabel("test.x0")).unwrap();
            x.size(&mut device, 12, AllocLabel("test.x")).unwrap();
            x0.write(&mut device, 0, &start).unwrap();
            x.write(&mut device, 0, &moved).unwrap();
            // Safety: both pose buffers hold the four vertices the record names.
            unsafe { memory.commit(&mut device, x0.handle(), x.handle(), 1.0) }.unwrap();
            let mut slip = [0f32; 4];
            memory.memory_slip.read(&mut device, 0, &mut slip).unwrap();
            let size = ((slip[0] as f64).powi(2) + (slip[1] as f64).powi(2) + (slip[2] as f64).powi(2)).sqrt();
            let first_order = gap * angle;
            assert!(size < 1e-2 * first_order, "axis {axis:?}: slip {:?}, against a first-order frame step of {first_order:e}", &slip[..3]);
        }
    }

    /// A framed side that rolls on the other without slipping records no slip:
    /// edge a's centerline moves `step` along edge b while a's frame turns by
    /// `-step / r_a` about a's own axis, so a's contact point stays where it
    /// was on b. The centerlines' relative step and a's surface step cancel.
    #[test]
    fn a_framed_side_rolling_without_slipping_does_not_slip() {
        let (radius_a, radius_b, step) = (1e-2f64, 2e-2f64, 1e-5f64);
        let gap = radius_a + radius_b;
        let mut device = host_device();
        let mut memory = FrictionMemory::allocate(&mut device, 8, 4).unwrap();
        let half = 0.5 * (-step / radius_a);
        let mut quat_start = Buffer::<f32>::none();
        let mut quat = Buffer::<f32>::none();
        let mut scratch = Buffer::<f32>::none();
        quat_start.size(&mut device, 8, AllocLabel("test.quat_start")).unwrap();
        quat.size(&mut device, 8, AllocLabel("test.quat")).unwrap();
        scratch.size(&mut device, 18, AllocLabel("test.scratch")).unwrap();
        quat_start.write(&mut device, 0, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        quat.write(&mut device, 0, &[half.sin() as f32, 0.0, 0.0, half.cos() as f32, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let none = memory.frames.edge_slot;
        memory.set_frames(FrictionFrames {
            edge_slot: none,
            quat: quat.handle(),
            quat_start: quat_start.handle(),
            gradient: scratch.handle(),
            self_block: scratch.handle(),
        });
        let _ = open(&mut memory, &mut device).unwrap();
        // Edge a along x over edge b along y, the normal +z from b toward a.
        let pair = [0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, -0.5, 1.0, 0.5, 1.0, 1.0, radius_a as f32, radius_b as f32];
        memory.record_word0.write(&mut device, 0, &[4u32 << 28]).unwrap(); // FRICTION_KIND_EDGE_EDGE
        memory.record_word1.write(&mut device, 0, &[0u32]).unwrap();
        memory.record_vertex.write(&mut device, 0, &[0u32, 1, 2, 3, 0, 1]).unwrap();
        memory.record_float.write(&mut device, 0, &pair).unwrap();
        memory.record_count.write(&mut device, 0, &[1]).unwrap();
        close(&mut memory, &mut device);
        let cross = [[-1.0, 0.0, gap], [1.0, 0.0, gap], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        let rolled = [[-1.0, step, gap], [1.0, step, gap], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        let pos = |p: [f64; 3]| p.map(|x| x as f32);
        let start: Vec<f32> = cross.iter().flat_map(|p| pos(*p)).collect();
        let moved: Vec<f32> = rolled.iter().flat_map(|p| pos(*p)).collect();
        let mut x0 = Buffer::<f32>::none();
        let mut x = Buffer::<f32>::none();
        x0.size(&mut device, 12, AllocLabel("test.x0")).unwrap();
        x.size(&mut device, 12, AllocLabel("test.x")).unwrap();
        x0.write(&mut device, 0, &start).unwrap();
        x.write(&mut device, 0, &moved).unwrap();
        // Safety: both pose buffers hold the four vertices the record names.
        unsafe { memory.commit(&mut device, x0.handle(), x.handle(), 1.0) }.unwrap();
        let mut slip = [0f32; 4];
        memory.memory_slip.read(&mut device, 0, &mut slip).unwrap();
        let size = ((slip[0] as f64).powi(2) + (slip[1] as f64).powi(2) + (slip[2] as f64).powi(2)).sqrt();
        assert!(size < 1e-2 * step, "slip {:?} against a roll of {step:e}", &slip[..3]);
    }

    /// A vertex over a framed edge that rolls under it records no slip: the
    /// edge's centerline moves `step` across while its frame turns by
    /// `step / r_b` about the edge's own axis, so its touching surface stays
    /// under the vertex. Read through the edge's own turn instead, the roll is
    /// a slip of `step`.
    #[test]
    fn a_framed_edge_rolling_under_a_vertex_does_not_slip() {
        let (radius_a, radius_b, step) = (1e-2f64, 2e-2f64, 1e-5f64);
        let gap = radius_a + radius_b;
        let mut device = host_device();
        let mut memory = FrictionMemory::allocate(&mut device, 8, 4).unwrap();
        let half = 0.5 * (step / radius_b);
        let mut quat_start = Buffer::<f32>::none();
        let mut quat = Buffer::<f32>::none();
        let mut scratch = Buffer::<f32>::none();
        quat_start.size(&mut device, 4, AllocLabel("test.quat_start")).unwrap();
        quat.size(&mut device, 4, AllocLabel("test.quat")).unwrap();
        scratch.size(&mut device, 9, AllocLabel("test.scratch")).unwrap();
        quat_start.write(&mut device, 0, &[0.0, 0.0, 0.0, 1.0]).unwrap();
        quat.write(&mut device, 0, &[half.sin() as f32, 0.0, 0.0, half.cos() as f32]).unwrap();
        let none = memory.frames.edge_slot;
        memory.set_frames(FrictionFrames {
            edge_slot: none,
            quat: quat.handle(),
            quat_start: quat_start.handle(),
            gradient: scratch.handle(),
            self_block: scratch.handle(),
        });
        let _ = open(&mut memory, &mut device).unwrap();
        // Vertex 0 (side a) over the middle of edge 1-2 along x (side b), the
        // normal +z from b toward a; only side b has a frame, slot 0.
        let pair = [0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -0.5, -0.5, 0.0, 1.0, 0.5, 1.0, 1.0, radius_a as f32, radius_b as f32];
        memory.record_word0.write(&mut device, 0, &[1u32 << 28]).unwrap(); // FRICTION_KIND_POINT_SURFACE
        memory.record_word1.write(&mut device, 0, &[0u32]).unwrap();
        memory.record_vertex.write(&mut device, 0, &[0u32, 1, 2, u32::MAX, u32::MAX, 0]).unwrap();
        memory.record_float.write(&mut device, 0, &pair).unwrap();
        memory.record_count.write(&mut device, 0, &[1]).unwrap();
        close(&mut memory, &mut device);
        let start = [[0.0, 0.0, gap], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let rolled = [[0.0, 0.0, gap], [-1.0, step, 0.0], [1.0, step, 0.0]];
        let pos = |p: [f64; 3]| p.map(|x| x as f32);
        let start: Vec<f32> = start.iter().flat_map(|p| pos(*p)).collect();
        let moved: Vec<f32> = rolled.iter().flat_map(|p| pos(*p)).collect();
        let mut x0 = Buffer::<f32>::none();
        let mut x = Buffer::<f32>::none();
        x0.size(&mut device, 9, AllocLabel("test.x0")).unwrap();
        x.size(&mut device, 9, AllocLabel("test.x")).unwrap();
        x0.write(&mut device, 0, &start).unwrap();
        x.write(&mut device, 0, &moved).unwrap();
        // Safety: both pose buffers hold the three vertices the record names.
        unsafe { memory.commit(&mut device, x0.handle(), x.handle(), 1.0) }.unwrap();
        let mut slip = [0f32; 4];
        memory.memory_slip.read(&mut device, 0, &mut slip).unwrap();
        let size = ((slip[0] as f64).powi(2) + (slip[1] as f64).powi(2) + (slip[2] as f64).powi(2)).sqrt();
        assert!(size < 1e-2 * step, "slip {:?} against a roll of {step:e}", &slip[..3]);
    }

    /// The commit's two halves: the memory comes out in the lexicographic key
    /// order the lookup bisects, and each slip is the record's rule evaluated
    /// at the committed pose, in its tangent plane, and return mapped.
    #[test]
    fn a_commit_sorts_the_record_and_evaluates_each_slip_at_the_committed_pose() {
        let mut device = host_device();
        let mut memory = FrictionMemory::allocate(&mut device, 8, 2).unwrap();
        let _ = open(&mut memory, &mut device).unwrap();
        // Three pairs, recorded out of order: keys (2, 5), (1, 9), (2, 3).
        let word0 = [2u32, 1, 2];
        let word1 = [5u32, 9, 3];
        let unused = u32::MAX;
        // Pair 0 is vertex 0 against a static floor; pair 1 is vertices 0 and 1
        // with weights 1 and -1; pair 2 is vertex 1, carrying an offset.
        let vertex = [0, unused, unused, unused, 0, 1, unused, unused, 1, unused, unused, unused];
        let up = [0.0f32, 0.0, 1.0];
        let float = [
            [0.0, 0.0, 0.0, up[0], up[1], up[2], 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0],
            [0.0, 0.0, 0.0, up[0], up[1], up[2], 1.0, -1.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0],
            [2e-8, 0.0, 5.0, up[0], up[1], up[2], 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0],
        ]
        .concat();
        memory.record_word0.write(&mut device, 0, &word0).unwrap();
        memory.record_word1.write(&mut device, 0, &word1).unwrap();
        memory.record_vertex.write(&mut device, 0, &pad_words(&vertex)).unwrap();
        memory.record_float.write(&mut device, 0, &pad_floats(&float)).unwrap();
        memory.record_count.write(&mut device, 0, &[3]).unwrap();
        close(&mut memory, &mut device);
        // Vertex 0 moves 3e-8 along x and 1 along z; vertex 1 moves 1e-8 along x.
        let pos = |p: [f64; 3]| p.map(|x| x as f32);
        let start = [pos([0.0, 0.0, 0.0]), pos([1e-3, 0.0, 0.0])].concat();
        let pose = [pos([3e-8, 0.0, 1.0]), pos([1e-3 + 1e-8, 0.0, 0.0])].concat();
        let mut x0 = Buffer::<f32>::none();
        let mut x = Buffer::<f32>::none();
        x0.size(&mut device, 6, AllocLabel("test.x0")).unwrap();
        x.size(&mut device, 6, AllocLabel("test.x")).unwrap();
        x0.write(&mut device, 0, &start).unwrap();
        x.write(&mut device, 0, &pose).unwrap();
        let eps = 1e-6;
        // Safety: both pose buffers hold the two vertices the records name.
        unsafe { memory.commit(&mut device, x0.handle(), x.handle(), eps) }.unwrap();
        assert_eq!(memory.memory_count, 3);
        let mut got0 = [0u32; 3];
        let mut got1 = [0u32; 3];
        let mut slip = [0f32; 12];
        memory.memory_word0.read(&mut device, 0, &mut got0).unwrap();
        memory.memory_word1.read(&mut device, 0, &mut got1).unwrap();
        memory.memory_slip.read(&mut device, 0, &mut slip).unwrap();
        assert_eq!((got0, got1), ([1, 2, 2], [9, 3, 5]));
        // (1, 9): x0 - x1 moved 3e-8 - 1e-8 along x, and its z is projected out.
        // (2, 3): offset 2e-8 plus vertex 1's 1e-8, the offset's z projected out.
        // (2, 5): vertex 0's 3e-8 along x.
        let want = [[2e-8f32, 0.0, 0.0], [3e-8, 0.0, 0.0], [3e-8, 0.0, 0.0]];
        for (entry, w) in want.iter().enumerate() {
            for d in 0..3 {
                let g = slip[MEMORY_FLOATS * entry + d];
                // An f32 resolves about 1.2e-10 at 1e-3, the largest coordinate here.
                assert!((g - w[d]).abs() < 1e-9, "entry {entry} axis {d}: {g} against {}", w[d]);
            }
        }
        // THE SLIDE is the step's own tangential motion, without the carried
        // offset: 2e-8, 1e-8 (the offset's 2e-8 is not motion) and 3e-8.
        for (entry, w) in [2e-8f32, 1e-8, 3e-8].iter().enumerate() {
            let g = slip[MEMORY_FLOATS * entry + 3];
            assert!((g - w).abs() < 1e-9, "entry {entry} slide: {g} against {w}");
        }

        // PAST THE STICK RANGE the slip is cut back to it along its direction.
        let _ = open(&mut memory, &mut device).unwrap();
        memory.record_word0.write(&mut device, 0, &[1]).unwrap();
        memory.record_word1.write(&mut device, 0, &[1]).unwrap();
        memory.record_vertex.write(&mut device, 0, &pad_words(&[0, unused, unused, unused])).unwrap();
        memory
            .record_float
            .write(&mut device, 0, &[0.0, 0.0, 0.0, up[0], up[1], up[2], 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0])
            .unwrap();
        memory.record_count.write(&mut device, 0, &[1]).unwrap();
        close(&mut memory, &mut device);
        let far = [pos([3e-6, 4e-6, 0.0]), pos([1e-3, 0.0, 0.0])].concat();
        x.write(&mut device, 0, &far).unwrap();
        // Safety: as above.
        unsafe { memory.commit(&mut device, x0.handle(), x.handle(), eps) }.unwrap();
        memory.memory_slip.read(&mut device, 0, &mut slip[..4]).unwrap();
        for (d, w) in [0.6e-6f32, 0.8e-6, 0.0].iter().enumerate() {
            assert!((slip[d] - w).abs() < 1e-10, "axis {d}: {} against {w}", slip[d]);
        }
    }

    /// Two contacts that share a key become one entry, the one bearing the
    /// larger normal force, whatever order the passes appended them in.
    #[test]
    fn equal_keys_merge_into_the_entry_with_the_larger_normal_force() {
        let unused = u32::MAX;
        let pos = |p: [f64; 3]| p.map(|x| x as f32);
        for order in [[0usize, 1, 2], [2, 1, 0]] {
            let mut device = host_device();
            let mut memory = FrictionMemory::allocate(&mut device, 8, 2).unwrap();
            let _ = open(&mut memory, &mut device).unwrap();
            // Keys (4, 7) twice, carrying 1e-8 and 3e-8 along x under normal
            // forces 2 and 5, and (4, 9) once.
            let entries = [
                ((4u32, 7u32), 1e-8f32, 2.0f32),
                ((4, 7), 3e-8, 5.0),
                ((4, 9), 2e-8, 1.0),
            ];
            let mut word0 = Vec::new();
            let mut word1 = Vec::new();
            let mut vertex = Vec::new();
            let mut float = Vec::new();
            for &i in &order {
                let ((a, b), carried, force) = entries[i];
                word0.push(a);
                word1.push(b);
                vertex.extend([0, unused, unused, unused]);
                float.extend([carried, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, force, 0.5, 1.0, 1.0]);
            }
            memory.record_word0.write(&mut device, 0, &word0).unwrap();
            memory.record_word1.write(&mut device, 0, &word1).unwrap();
            memory.record_vertex.write(&mut device, 0, &pad_words(&vertex)).unwrap();
            memory.record_float.write(&mut device, 0, &pad_floats(&float)).unwrap();
            memory.record_count.write(&mut device, 0, &[3]).unwrap();
            close(&mut memory, &mut device);
            // Nothing moves, so each slip is its carried offset.
            let mut pose = Buffer::<f32>::none();
            pose.size(&mut device, 3, AllocLabel("test.pose")).unwrap();
            pose.write(&mut device, 0, &pos([0.0, 0.0, 0.0])).unwrap();
            // Safety: the pose holds the one vertex the records name.
            unsafe { memory.commit(&mut device, pose.handle(), pose.handle(), 1e-6) }.unwrap();
            let state = memory.snapshot(&mut device).unwrap();
            assert_eq!((state.word0.clone(), state.word1.clone()), (vec![4, 4], vec![7, 9]),
                       "order {order:?}");
            assert!((state.slip[0] - 3e-8).abs() < 1e-12, "order {order:?}: kept {}", state.slip[0]);
            assert!((state.slip[MEMORY_FLOATS] - 2e-8).abs() < 1e-12, "order {order:?}");
        }
    }

    /// Closing an iteration gathers its record into the vertices' cone sums,
    /// |w| m mu N and one pair per vertex, and writes each vertex's
    /// non-friction contact load: the force less the residual less the friction
    /// each pair's rule gives at the iterate.
    #[test]
    fn closing_an_iteration_gathers_cone_sums_and_non_friction_loads() {
        let mut device = host_device();
        let mut memory = FrictionMemory::allocate(&mut device, 8, 2).unwrap();
        let _ = open(&mut memory, &mut device).unwrap();
        let unused = u32::MAX;
        let floor = 9u32 << 28;
        let edge_edge = 4u32 << 28;
        // An edge-edge pair over vertices 0 and 1 (weights 0.25, -0.75, N 2,
        // mu 0.5, deposited twice) carrying a slip of 2e-6 along x inside the
        // stick range 1e-5, a floor contact on vertex 0 (N 3, mu 0.4) and one
        // with no cone on vertex 1 (N 0).
        let word0 = [edge_edge, floor, floor | 1];
        let word1 = [0u32, 0, 0];
        let vertex = [0, 1, unused, unused, 0, unused, unused, unused, 1, unused, unused, unused];
        let float = [
            [2e-6, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25, -0.75, 0.0, 0.0, 2.0, 0.5, 2.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0, 0.4, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0],
        ]
        .concat();
        memory.record_word0.write(&mut device, 0, &word0).unwrap();
        memory.record_word1.write(&mut device, 0, &word1).unwrap();
        memory.record_vertex.write(&mut device, 0, &pad_words(&vertex)).unwrap();
        memory.record_float.write(&mut device, 0, &pad_floats(&float)).unwrap();
        memory.record_count.write(&mut device, 0, &[3]).unwrap();
        // The passes started from residual (1, 0, 0) on vertex 0 and deposited
        // up to force (1, 5, 0). The pair's friction is mu N u / eps =
        // (0.2, 0, 0), deposited twice at weights 0.25 and -0.75: (0.1, 0, 0)
        // on vertex 0 and (-0.3, 0, 0) on vertex 1.
        let mut force = Buffer::<f32>::none();
        let mut residual = Buffer::<f32>::none();
        force.size(&mut device, 6, AllocLabel("test.force")).unwrap();
        residual.size(&mut device, 6, AllocLabel("test.residual")).unwrap();
        force.write(&mut device, 0, &[1.0, 5.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        residual.write(&mut device, 0, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let mut pose = Buffer::<f32>::none();
        pose.size(&mut device, 6, AllocLabel("test.pose")).unwrap();
        device.fill_zero(pose.handle(), 6 * std::mem::size_of::<f32>()).unwrap();
        // Safety: every array holds the two vertices.
        unsafe {
            memory.close(&mut device, pose.handle(), pose.handle(), 1e-5, force.handle(), residual.handle())
        }
        .unwrap();
        let mut cone = [0f32; 2 * CONE_FLOATS];
        memory.cone.read(&mut device, 0, &mut cone).unwrap();
        // Per vertex: cone sum, pairs, load.
        let want = [1.7f32, 2.0, -0.1, 5.0, 0.0, 1.5, 1.0, 0.3, 0.0, 0.0];
        for (i, (g, w)) in cone.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "slot {i}: {g} against {w}");
        }
        // THE LOAD IS THE LAST COMMIT'S, NOT THE LAST ITERATION'S: before any
        // commit the next open reads the plain residual, whatever the iteration
        // just gathered.
        let mut snapshot = Buffer::<f32>::none();
        snapshot.size(&mut device, 6, AllocLabel("test.snapshot")).unwrap();
        snapshot.write(&mut device, 0, &[0.5, 0.0, 1.0, 0.0, 2.0, 0.0]).unwrap();
        // Safety: the snapshot holds the two vertices.
        let refs = unsafe { memory.open(&mut device, snapshot.handle(), true) }.unwrap();
        assert_eq!(refs.friction_load, memory.load.handle());
        let mut load = [0f32; 6];
        memory.load.read(&mut device, 0, &mut load).unwrap();
        assert_eq!(load, [0.5, 0.0, 1.0, 0.0, 2.0, 0.0], "an uncommitted load was read");
        // ONCE THE STEP COMMITS, each vertex's load is its residual plus the load
        // that commit left: (0.5, 0, 1) + (-0.1, 5, 0) and (0, 2, 0) + (0.3, 0, 0).
        // The open above restarted the record, so it is written again.
        memory.record_word0.write(&mut device, 0, &word0).unwrap();
        memory.record_word1.write(&mut device, 0, &word1).unwrap();
        memory.record_vertex.write(&mut device, 0, &pad_words(&vertex)).unwrap();
        memory.record_float.write(&mut device, 0, &pad_floats(&float)).unwrap();
        memory.record_count.write(&mut device, 0, &[3]).unwrap();
        // Safety: every array holds the two vertices.
        unsafe {
            memory.close(&mut device, pose.handle(), pose.handle(), 1e-5, force.handle(), residual.handle())
        }
        .unwrap();
        // Safety: the pose holds the two vertices the records name.
        unsafe { memory.commit(&mut device, pose.handle(), pose.handle(), 1e-5) }.unwrap();
        // Safety: as above.
        let _ = unsafe { memory.open(&mut device, snapshot.handle(), true) }.unwrap();
        memory.load.read(&mut device, 0, &mut load).unwrap();
        let want_load = [0.4f32, 5.0, 1.0, 0.3, 2.0, 0.0];
        for (i, (g, w)) in load.iter().zip(want_load.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "load {i}: {g} against {w}");
        }
        // The next close starts the sums over.
        memory.record_count.write(&mut device, 0, &[0]).unwrap();
        close(&mut memory, &mut device);
        memory.cone.read(&mut device, 0, &mut cone).unwrap();
        assert!(cone.iter().all(|&c| c == 0.0), "{cone:?}");
    }

    /// A snapshot restores to the same memory, and a disordered one is refused.
    #[test]
    fn a_snapshot_restores_to_the_same_memory_and_a_disordered_one_is_refused() {
        let mut device = host_device();
        let mut memory = FrictionMemory::allocate(&mut device, 4, 1).unwrap();
        let state = FrictionMemoryState {
            word0: vec![1, 2, 2],
            word1: vec![9, 3, 5],
            slip: vec![1.0, 2.0, 3.0, 0.1, 4.0, 5.0, 6.0, 0.2, 7.0, 8.0, 9.0, 0.3],
            committed_cone: vec![0.5, 1.0, 2.0, 3.0, 4.0],
        };
        memory.restore(&mut device, &state).unwrap();
        assert_eq!(memory.snapshot(&mut device).unwrap(), state);
        // The lagged sums the next step's first iteration reads are the
        // committed ones, as a commit leaves them.
        let mut cone = vec![0f32; CONE_FLOATS];
        memory.cone.read(&mut device, 0, &mut cone).unwrap();
        assert_eq!(cone, state.committed_cone);
        // Sums for another vertex count are another scene's.
        let mut other = state.clone();
        other.committed_cone.push(0.0);
        assert!(memory.restore(&mut device, &other).is_err());
        let mut disordered = state.clone();
        disordered.word1.swap(1, 2);
        disordered.word0.swap(0, 1);
        assert!(memory.restore(&mut device, &disordered).is_err());
    }

    /// A record past its capacity grows it and refuses to commit.
    #[test]
    fn an_overflowed_record_is_refused_at_commit_and_grows_the_capacity() {
        let mut device = host_device();
        let mut memory = FrictionMemory::allocate(&mut device, 2, 1).unwrap();
        let refs = open(&mut memory, &mut device).unwrap();
        assert_eq!(refs.record_capacity, 2);
        memory.record_count.write(&mut device, 0, &[5]).unwrap();
        close(&mut memory, &mut device);
        let mut pose = Buffer::<f32>::none();
        pose.size(&mut device, 3, AllocLabel("test.pose")).unwrap();
        // Safety: nothing is read from the pose when the commit refuses.
        let refused = unsafe { memory.commit(&mut device, pose.handle(), pose.handle(), 1e-6) };
        assert!(refused.is_err(), "an incomplete record was committed");
        assert_eq!(open(&mut memory, &mut device).unwrap().record_capacity, 10);
    }
}
