// File: crates/ppf-cts-solver/src/driver/force_field.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The external force field on the device.
//!
//! The host reads and verifies the field once (`crate::force_field`) and
//! installs it here; it is uploaded ONCE and never re-staged per step, because
//! a field object's track and a script are authored inputs rather than
//! per-step state. Every step then runs ONE dispatch, `external_field`, over
//! the vertices at the step's starting position, velocity and time, and the
//! two outputs are what the inertial target and the aerodynamic drag read.
//! The one input that moves is a Fluid field's volume: its frames stay on
//! disk, and the two bracketing the instant a step evaluates are resident in
//! slots sized once at install (`prepare_fluid`).
//!
//! A FIELD'S ORIGIN REACHES THE DEVICE IN SOLVER UNITS. The session carries
//! it in double precision; the host applies World Scaling in f64, refuses an
//! origin past [`ORIGIN_REACH`], and rounds once to the float the kernel
//! subtracts from a vertex.
//!
//! THE ACCELERATION BUFFER ALWAYS EXISTS, zero-filled, because the target
//! kernel gathers it unconditionally: a scene with no field pays one 12-byte
//! read per vertex and adds exactly zero, which is one code path rather than a
//! second entry point for "no field". Everything else takes a real zero-length
//! handle when absent, never `Handle::NONE`, which a generated entry asserts
//! against.

use ppf_cts_compute::{AllocLabel, Buffer, Device, Handle, Pod, StagedBuffer};

use super::scene::{Fatal, FatalResult};
use crate::force_field::{
    FieldType, Fluid, ForceFieldData, FIELD_HEADER_WORDS, FIELD_TIMING_FLOATS, FLUID_BRICK_FLOATS,
    FLUID_CHANNELS, FLUID_EMPTY, ORIGIN_REACH, SCRIPT_HEADER_WORDS, TRACK_FILE_FLOATS,
    TRACK_FLOATS, TRACK_ORIGIN_FLOATS,
};

/// Words per resident fluid frame, `FLUID_FRAME_WORDS` in the kernel: table
/// offset, data offset and resolution.
const FLUID_FRAME_WORDS: usize = 5;
/// Words per Fluid field, `FLUID_WORDS`: two resident frames.
const FLUID_WORDS: usize = 2 * FLUID_FRAME_WORDS;
/// Floats per resident fluid frame, `FLUID_FRAME_FLOATS`: the domain's axes
/// (the columns local X, Y, Z), its cell along local X, Y and Z, then the
/// domain corner in solver units.
const FLUID_FRAME_FLOATS: usize = 15;
/// Where a resident frame's domain corner starts in its floats,
/// `FLUID_FRAME_CORNER`.
const FLUID_FRAME_CORNER: usize = 12;
/// Floats per Fluid field, `FLUID_FLOATS`: the blend, then two resident
/// frames.
const FLUID_FLOATS: usize = 1 + 2 * FLUID_FRAME_FLOATS;
/// Frames each Fluid field keeps on the HOST beside its two device slots, as
/// they were read from disk. A step's Newton iterations read the volume at
/// the step's full end and its trailing iteration at the end the line search
/// committed, so a cut step whose committed end falls back across a frame
/// boundary brackets three consecutive frames before the next step brackets
/// the later two again; three entries serve that back and forth with no
/// second disk read. Each entry is a whole frame, so the count is kept to
/// what that pattern needs.
const FLUID_CACHE_FRAMES: usize = 3;

/// A frame as it was read from disk and verified: its table, and its bricks
/// in scene units, before World Scaling.
struct CachedFrame {
    frame: Option<usize>,
    table: Vec<u32>,
    bricks: Vec<f32>,
}

/// One Fluid field's residency: its volume (frames stay on disk), where its
/// two slots sit in the device tables, which frame each slot holds, and the
/// host cache the slots are filled from.
struct FluidResidency {
    /// The field's index in the manifest, which every message names.
    field: usize,
    volume: Fluid,
    table_base: [usize; 2],
    data_base: [usize; 2],
    /// Each slot's size: the longest table and the most bricks of any frame.
    table_slot: usize,
    data_slot: usize,
    resident: [Option<usize>; 2],
    /// Per frame, the axes, the cell, then the domain corner in solver units
    /// (held to the reach once, at install), as the device reads them.
    frame_floats: Vec<[f32; FLUID_FRAME_FLOATS]>,
    /// [`FLUID_CACHE_FRAMES`] entries (fewer for a shorter volume), each
    /// sized at install for the largest frame.
    cache: Vec<CachedFrame>,
    /// Frames read from disk since install.
    disk_reads: usize,
}

impl FluidResidency {
    /// The cache entry holding frame `k`. On a miss the frame is read from
    /// disk, held to what the loader measured, into the entry whose frame is
    /// farthest from the bracket `(k0, k1)` being made resident.
    fn cached(&mut self, k: usize, (k0, k1): (usize, usize)) -> FatalResult<usize> {
        if let Some(entry) = self.cache.iter().position(|c| c.frame == Some(k)) {
            return Ok(entry);
        }
        let distance = |frame: Option<usize>| {
            frame.map_or(usize::MAX, |f| f.abs_diff(k0).min(f.abs_diff(k1)))
        };
        let entry = (0..self.cache.len())
            .max_by_key(|&e| distance(self.cache[e].frame))
            .expect("a fluid keeps at least one cache entry");
        let CachedFrame { frame, table, bricks } = &mut self.cache[entry];
        // Unclaimed until the read succeeds, so a failed read leaves no entry
        // claiming a frame it does not hold.
        *frame = None;
        self.volume.read_frame(k, table, bricks).map_err(|e| {
            Fatal::invariant(format!(
                "solver driver: force field {} fluid frame {k} no longer reads as it did at \
                 load: {e}",
                self.field
            ))
        })?;
        *frame = Some(k);
        self.disk_reads += 1;
        Ok(entry)
    }
}

#[derive(Default)]
pub struct ForceField {
    installed: bool,
    vertices: usize,
    field_count: u32,
    script_count: u32,
    has_air: bool,
    /// Whether some field has a term evaluated at the Newton iterate, so the
    /// assembly launches `external_field_implicit` only when there is one.
    has_implicit: bool,
    has_weight: bool,
    has_mask: bool,
    /// `1 / world_scaling`: the field is authored in scene units.
    inv_world_scaling: f32,
    field_header: StagedBuffer<u32>,
    field_timing: StagedBuffer<f32>,
    field_origin: StagedBuffer<f32>,
    field_track: StagedBuffer<f32>,
    /// Every Texture field's image, LINEAR RGB texels concatenated.
    field_image: StagedBuffer<f32>,
    script_header: StagedBuffer<u32>,
    script_code: StagedBuffer<u32>,
    script_constants: StagedBuffer<f32>,
    weight: StagedBuffer<f32>,
    target_mask: StagedBuffer<u32>,
    /// The Fluid fields' resident frames: per field, [`FLUID_WORDS`] words
    /// and [`FLUID_FLOATS`] floats saying where each frame sits and how it is
    /// placed, then the brick tables and the bricks of every slot. SIZED
    /// ONCE at install for the largest frame of each field, so a step that
    /// crosses a fluid frame overwrites a slot and allocates nothing, on the
    /// device or on the host.
    fluids: Vec<FluidResidency>,
    world_scaling: f32,
    fluid_words: StagedBuffer<u32>,
    fluid_floats: StagedBuffer<f32>,
    /// What `fluid_words` and `fluid_floats` last uploaded, and the values
    /// the current call builds, compared before anything is uploaded.
    fluid_words_sent: Vec<u32>,
    fluid_floats_sent: Vec<f32>,
    fluid_words_next: Vec<u32>,
    fluid_floats_next: Vec<f32>,
    fluid_table: StagedBuffer<u32>,
    fluid_data: StagedBuffer<f32>,
    /// `3 * vertices` floats, the acceleration the target adds beside gravity.
    acceleration: Buffer<f32>,
    /// `3 * vertices` floats when a Wind field exists, else empty.
    air_velocity: Buffer<f32>,
}

fn stage<T: Pod + Default>(
    device: &mut impl Device,
    buffer: &mut StagedBuffer<T>,
    host: &[T],
    label: &'static str,
) -> FatalResult<()> {
    buffer.size(device, host.len(), AllocLabel(label))?;
    buffer.at()[..host.len()].copy_from_slice(host);
    buffer.upload(device)?;
    Ok(())
}

impl ForceField {
    /// Size the always-present outputs and give every absent input a real
    /// zero-length handle. Runs at `allocate()`, before any field is
    /// installed, and is a no-op on the inputs of an installed field.
    pub fn allocate(&mut self, device: &mut impl Device, vertices: usize) -> FatalResult<()> {
        self.vertices = vertices;
        self.acceleration
            .size(device, 3 * vertices, AllocLabel("field.acceleration"))?;
        if !self.installed {
            stage::<u32>(device, &mut self.field_header, &[], "field.header")?;
            stage::<f32>(device, &mut self.field_timing, &[], "field.timing")?;
            stage::<f32>(device, &mut self.field_origin, &[], "field.origin")?;
            stage::<f32>(device, &mut self.field_track, &[], "field.track")?;
            stage::<f32>(device, &mut self.field_image, &[], "field.image")?;
            stage::<u32>(device, &mut self.script_header, &[], "field.script_header")?;
            stage::<u32>(device, &mut self.script_code, &[], "field.script_code")?;
            stage::<f32>(device, &mut self.script_constants, &[], "field.script_constants")?;
            stage::<f32>(device, &mut self.weight, &[], "field.weight")?;
            stage::<u32>(device, &mut self.target_mask, &[], "field.target_mask")?;
            stage::<u32>(device, &mut self.fluid_words, &[], "field.fluid_words")?;
            stage::<f32>(device, &mut self.fluid_floats, &[], "field.fluid_floats")?;
            stage::<u32>(device, &mut self.fluid_table, &[], "field.fluid_table")?;
            stage::<f32>(device, &mut self.fluid_data, &[], "field.fluid_data")?;
            self.air_velocity.size(device, 0, AllocLabel("field.air_velocity"))?;
        }
        Ok(())
    }

    /// Install a verified field. Replaces whatever was installed, which is
    /// how a held run takes an updated field between frames.
    pub fn install(
        &mut self,
        device: &mut impl Device,
        field: &ForceFieldData,
        world_scaling: f32,
    ) -> FatalResult<()> {
        let vertices = self.vertices;
        if !(world_scaling > 0.0 && world_scaling.is_finite()) {
            return Err(Fatal::invariant(format!(
                "solver driver: the force field was installed with world scaling {world_scaling}"
            )));
        }
        self.inv_world_scaling = 1.0 / world_scaling;
        for (what, len) in [
            ("weights", field.weight.as_ref().map(Vec::len)),
            ("target masks", field.target_mask.as_ref().map(Vec::len)),
        ] {
            if let Some(len) = len {
                if len != vertices {
                    return Err(Fatal::invariant(format!(
                        "solver driver: the force field carries {len} {what} for {vertices} vertices"
                    )));
                }
            }
        }
        // Every field's header, timing, origins and track concatenated,
        // each field locating its samples by its first sample and count.
        let mut header: Vec<u32> = Vec::with_capacity(FIELD_HEADER_WORDS * field.fields.len());
        let mut timing: Vec<f32> = Vec::with_capacity(FIELD_TIMING_FLOATS * field.fields.len());
        let mut origins: Vec<f32> = Vec::new();
        let mut track: Vec<f32> = Vec::new();
        let mut image: Vec<f32> = Vec::new();
        // Every Fluid field's two slots, laid out end to end, and every
        // frame's corner held to the reach HERE, before the
        // first step, rather than at the step that first brackets it.
        self.world_scaling = world_scaling;
        self.fluids.clear();
        let (mut table_len, mut data_len) = (0usize, 0usize);
        for (index, record) in field.fields.iter().enumerate() {
            let Some(volume) = &record.fluid else {
                continue;
            };
            let table = volume.table_capacity();
            let data = volume.brick_capacity() * FLUID_BRICK_FLOATS;
            let frames = volume.frames();
            // The loader's verification guarantees both; a record built
            // anywhere else is refused here rather than at its first bracket.
            if frames == 0
                || volume.brick_counts.len() != frames
                || volume.velocity_bound.len() != frames
            {
                return Err(Fatal::invariant(format!(
                    "solver driver: force field {index} is a fluid of {frames} frames with {} \
                     brick counts and {} velocity bounds measured, which the loader refuses",
                    volume.brick_counts.len(),
                    volume.velocity_bound.len()
                )));
            }
            let mut frame_floats = Vec::with_capacity(frames);
            for k in 0..frames {
                let corner = volume.corner(k);
                let mut floats = [0.0f32; FLUID_FRAME_FLOATS];
                for axis in 0..3 {
                    let solver = corner[axis] * world_scaling as f64;
                    if !(solver.abs() <= ORIGIN_REACH) {
                        return Err(Fatal::invariant(format!(
                            "solver driver: force field {index} fluid frame {k} has its domain \
                             corner at {solver} solver units, past the {ORIGIN_REACH} a \
                             field may sit from the solver's origin"
                        )));
                    }
                    floats[FLUID_FRAME_CORNER + axis] = solver as f32;
                }
                // THE VELOCITY IS SCALED BY WORLD SCALING when a frame is
                // copied into its slot (`prepare_fluid`), so the largest
                // component the loader measured must stay a float after it.
                let scaled = volume.velocity_bound[k] * world_scaling;
                if !scaled.is_finite() {
                    return Err(Fatal::invariant(format!(
                        "solver driver: force field {index} fluid frame {k} holds a velocity of \
                         {} scene units per second, which World Scaling {world_scaling} takes \
                         past single precision",
                        volume.velocity_bound[k]
                    )));
                }
                for c in 0..3 {
                    let axis = volume.axis(k, c);
                    for r in 0..3 {
                        floats[3 * c + r] = axis[r] as f32;
                    }
                }
                for (c, cell) in volume.cell(k).into_iter().enumerate() {
                    floats[9 + c] = cell as f32;
                }
                frame_floats.push(floats);
            }
            let cache = (0..FLUID_CACHE_FRAMES.min(frames))
                .map(|_| CachedFrame {
                    frame: None,
                    table: Vec::with_capacity(table),
                    bricks: Vec::with_capacity(data),
                })
                .collect();
            self.fluids.push(FluidResidency {
                field: index,
                volume: volume.clone(),
                table_base: [table_len, table_len + table],
                data_base: [data_len, data_len + data],
                table_slot: table,
                data_slot: data,
                resident: [None, None],
                frame_floats,
                cache,
                disk_reads: 0,
            });
            table_len += 2 * table;
            data_len += 2 * data;
        }
        if u32::try_from(data_len).is_err() || u32::try_from(table_len).is_err() {
            return Err(Fatal::invariant(format!(
                "solver driver: the fluid volumes need {data_len} resident floats and \
                 {table_len} table entries, past the kernel's 32-bit offsets"
            )));
        }
        let mut fluid_index = 0u32;
        for (index, record) in field.fields.iter().enumerate() {
            let first = u32::try_from(origins.len() / TRACK_ORIGIN_FLOATS).map_err(|_| {
                Fatal::invariant("solver driver: the force field's tracks exceed 32-bit offsets")
            })?;
            let samples = record.samples();
            header.extend_from_slice(&[
                record.kind as u32,
                record.shape as u32,
                record.falloff as u32,
                record.z_direction as u32,
                record.flags,
                record.seed,
                record.target,
                first,
                samples as u32,
            ]);
            match &record.texture {
                Some(t) => {
                    let offset = u32::try_from(image.len()).map_err(|_| {
                        Fatal::invariant(
                            "solver driver: the force field's images exceed 32-bit offsets",
                        )
                    })?;
                    header.extend_from_slice(&[
                        offset, t.width, t.height, t.mode, t.extension, t.flags,
                    ]);
                    image.extend_from_slice(&t.pixels);
                }
                None if record.kind == FieldType::Fluid => {
                    let use_density = record.fluid.as_ref().is_some_and(|f| f.use_density);
                    header.extend_from_slice(&[fluid_index, u32::from(use_density), 0, 0, 0, 0]);
                    fluid_index += 1;
                }
                None => header.extend_from_slice(&[0; 6]),
            }
            timing.extend_from_slice(&[record.start, record.step]);
            for k in 0..samples {
                for axis in 0..3 {
                    let solver = record.origin[k][axis] * world_scaling as f64;
                    if !(solver.abs() <= ORIGIN_REACH) {
                        return Err(Fatal::invariant(format!(
                            "solver driver: force field {index} sample {k} has its origin at \
                             {solver} solver units, past the {ORIGIN_REACH} a field may sit \
                             from the solver's origin"
                        )));
                    }
                    origins.push(solver as f32);
                }
                track.extend_from_slice(
                    &record.track[TRACK_FILE_FLOATS * k..TRACK_FILE_FLOATS * (k + 1)],
                );
                // The origin's motion to the next sample, in scene units: the
                // kernel subtracts a fraction of it from the difference
                // at this sample, so it is a small per-frame quantity and
                // nothing large cancels. The last sample holds still.
                for axis in 0..3 {
                    let delta = if k + 1 < samples {
                        record.origin[k + 1][axis] - record.origin[k][axis]
                    } else {
                        0.0
                    };
                    track.push(delta as f32);
                }
            }
        }
        debug_assert_eq!(track.len(), TRACK_FLOATS * (origins.len() / TRACK_ORIGIN_FLOATS));
        stage(device, &mut self.field_header, &header, "field.header")?;
        stage(device, &mut self.field_timing, &timing, "field.timing")?;
        stage(device, &mut self.field_origin, &origins, "field.origin")?;
        stage(device, &mut self.field_track, &track, "field.track")?;
        stage(device, &mut self.field_image, &image, "field.image")?;
        // The fluid slots start EMPTY (every table entry names no brick) and
        // are filled by `prepare_fluid` before the first step reads them.
        stage(device, &mut self.fluid_table, &vec![FLUID_EMPTY; table_len], "field.fluid_table")?;
        stage(device, &mut self.fluid_data, &vec![0.0; data_len], "field.fluid_data")?;
        let nfluid = self.fluids.len();
        stage(device, &mut self.fluid_words, &vec![0; FLUID_WORDS * nfluid], "field.fluid_words")?;
        stage(device, &mut self.fluid_floats, &vec![0.0; FLUID_FLOATS * nfluid], "field.fluid_floats")?;
        self.fluid_words_sent = vec![0; FLUID_WORDS * nfluid];
        self.fluid_floats_sent = vec![0.0; FLUID_FLOATS * nfluid];
        self.fluid_words_next = vec![0; FLUID_WORDS * nfluid];
        self.fluid_floats_next = vec![0.0; FLUID_FLOATS * nfluid];
        // Every script's code and constants concatenated, each located by its
        // row of the script table.
        let mut script_header: Vec<u32> =
            Vec::with_capacity(SCRIPT_HEADER_WORDS * field.scripts.len());
        let mut code: Vec<u32> = Vec::new();
        let mut constants: Vec<f32> = Vec::new();
        for script in &field.scripts {
            script_header.extend_from_slice(&[
                code.len() as u32,
                script.code.len() as u32,
                constants.len() as u32,
                script.constants.len() as u32,
                script.target,
            ]);
            code.extend_from_slice(&script.code);
            constants.extend_from_slice(&script.constants);
        }
        stage(device, &mut self.script_header, &script_header, "field.script_header")?;
        stage(device, &mut self.script_code, &code, "field.script_code")?;
        stage(device, &mut self.script_constants, &constants, "field.script_constants")?;
        let weight: &[f32] = field.weight.as_deref().unwrap_or(&[]);
        stage(device, &mut self.weight, weight, "field.weight")?;
        let mask: &[u32] = field.target_mask.as_deref().unwrap_or(&[]);
        stage(device, &mut self.target_mask, mask, "field.target_mask")?;
        self.has_air = field.has_air();
        self.has_implicit = field.has_implicit();
        self.air_velocity.size(
            device,
            if self.has_air { 3 * vertices } else { 0 },
            AllocLabel("field.air_velocity"),
        )?;
        self.field_count = field.fields.len() as u32;
        self.script_count = field.scripts.len() as u32;
        self.has_weight = field.weight.is_some();
        self.has_mask = field.target_mask.is_some();
        self.installed = true;
        Ok(())
    }

    /// Evaluate the field for this step, at `positions` (the positions the
    /// step starts from), the velocity `(positions - previous) / previous_dt`
    /// and `time` (its starting clock). `surface` is the vertex-face table
    /// (index, offset, whether it exists) and the faces, which Wind Factor
    /// folds a vertex normal from as the drag does.
    pub fn evaluate(
        &mut self,
        device: &mut impl Device,
        positions: Handle,
        previous: Handle,
        previous_dt: f32,
        prop: Handle,
        surface: (Handle, Handle, u32, Handle),
        time: f32,
    ) -> FatalResult<()> {
        if !self.installed || self.vertices == 0 {
            return Ok(());
        }
        let count = self.vertices as u32;
        let args = crate::driver::kernels::ExternalFieldArgs {
            current: positions,
            previous,
            previous_dt,
            prop,
            neighbor_index: surface.0,
            neighbor_offset: surface.1,
            has_neighbor: surface.2,
            face: surface.3,
            weight: self.weight.handle(),
            has_weight: u32::from(self.has_weight),
            target_mask: self.target_mask.handle(),
            has_mask: u32::from(self.has_mask),
            field_header: self.field_header.handle(),
            field_timing: self.field_timing.handle(),
            field_origin: self.field_origin.handle(),
            field_track: self.field_track.handle(),
            field_count: self.field_count,
            field_image: self.field_image.handle(),
            script_header: self.script_header.handle(),
            script_count: self.script_count,
            script_code: self.script_code.handle(),
            script_constants: self.script_constants.handle(),
            time,
            inv_world_scaling: self.inv_world_scaling,
            acceleration: self.acceleration.handle(),
            air_velocity: self.air_velocity.handle(),
            has_air: u32::from(self.has_air),
            count,
            seam_arena_count: 0,
        };
        // Safety: every handle names a live allocation for the whole call.
        unsafe { device.launch("field.external", &args, count) }?;
        Ok(())
    }

    /// Make each Fluid field's two resident frames the two bracketing
    /// `time`, and write the blend between them.
    ///
    /// A FRAME ENTERS A SLOT ONLY WHEN THE BRACKET MOVES ONTO IT, so a step
    /// crossing a fluid frame copies ONE frame and keeps the other slot. The
    /// copy comes from the host cache, and the cache reads the disk only for a
    /// frame it does not hold: the Newton iterations read the volume at the
    /// step's full end and the trailing iteration at the end the line search
    /// committed, so a cut step can bracket both sides of a frame boundary in
    /// turn, and that back and forth costs a copy rather than a read.
    ///
    /// THE VELOCITY TAKES WORLD SCALING IN THE COPY, once: the volume holds
    /// scene units per second and the kernel compares it against a vertex
    /// velocity in solver units per second. The density is a ratio and is
    /// copied as it is. The step's Playback Speed is NOT applied here: a
    /// slot keeps its frame across steps whose playback differs, so the
    /// kernel divides by it per call ([`ForceField::implicit`]).
    ///
    /// The tables and the bricks are uploaded whole, and only when a slot
    /// changed; the words and floats only when they differ from what was
    /// last uploaded, which the blend makes about once per step.
    fn prepare_fluid(&mut self, device: &mut impl Device, time: f32) -> FatalResult<()> {
        if self.fluids.is_empty() {
            return Ok(());
        }
        let world_scaling = self.world_scaling;
        let mut data_dirty = false;
        for (index, fluid) in self.fluids.iter_mut().enumerate() {
            let (k0, k1, b) = fluid.volume.bracket(time);
            for want in [k0, k1] {
                if fluid.resident.contains(&Some(want)) {
                    continue;
                }
                // The slot holding neither wanted frame.
                let slot = (0..2)
                    .find(|&s| !matches!(fluid.resident[s], Some(f) if f == k0 || f == k1))
                    .expect("two slots hold at most the two wanted frames");
                // The slot is claimed by nothing while it is rewritten, so a
                // failure below leaves no slot naming a frame it half holds.
                fluid.resident[slot] = None;
                let entry = fluid.cached(want, (k0, k1))?;
                let CachedFrame { table, bricks, .. } = &fluid.cache[entry];
                if table.len() > fluid.table_slot || bricks.len() > fluid.data_slot {
                    return Err(Fatal::invariant(format!(
                        "solver driver: force field {} fluid frame {want} has {} table entries \
                         and {} floats, past the slot of {} and {} its volume was sized for",
                        fluid.field,
                        table.len(),
                        bricks.len(),
                        fluid.table_slot,
                        fluid.data_slot
                    )));
                }
                let tb = fluid.table_base[slot];
                self.fluid_table.at()[tb..tb + table.len()].copy_from_slice(table);
                let db = fluid.data_base[slot];
                let target = &mut self.fluid_data.at()[db..db + bricks.len()];
                let mut finite = true;
                for (out, cell) in target
                    .chunks_exact_mut(FLUID_CHANNELS)
                    .zip(bricks.chunks_exact(FLUID_CHANNELS))
                {
                    for c in 0..3 {
                        out[c] = cell[c] * world_scaling;
                        finite &= out[c].is_finite();
                    }
                    out[3] = cell[3];
                }
                // Install held every frame's largest velocity to this, so only
                // a file that changed under the run reaches it.
                if !finite {
                    return Err(Fatal::invariant(format!(
                        "solver driver: force field {} fluid frame {want} holds a velocity World \
                         Scaling {world_scaling} takes past single precision; the files changed \
                         under the run",
                        fluid.field
                    )));
                }
                fluid.resident[slot] = Some(want);
                data_dirty = true;
            }
            self.fluid_floats_next[FLUID_FLOATS * index] = b;
            for (which, frame) in [k0, k1].into_iter().enumerate() {
                let slot = fluid
                    .resident
                    .iter()
                    .position(|&r| r == Some(frame))
                    .expect("both bracketing frames are resident");
                let w = FLUID_WORDS * index + FLUID_FRAME_WORDS * which;
                let words = &mut self.fluid_words_next[w..w + FLUID_FRAME_WORDS];
                // Install refused tables past 32-bit offsets.
                words[0] = fluid.table_base[slot] as u32;
                words[1] = fluid.data_base[slot] as u32;
                words[2..5].copy_from_slice(&fluid.volume.resolution[frame]);
                let f = FLUID_FLOATS * index + 1 + FLUID_FRAME_FLOATS * which;
                self.fluid_floats_next[f..f + FLUID_FRAME_FLOATS]
                    .copy_from_slice(&fluid.frame_floats[frame]);
            }
        }
        if data_dirty {
            self.fluid_table.upload(device)?;
            self.fluid_data.upload(device)?;
        }
        // `at()` hands out the host side for writing, which withdraws the
        // buffer's claim to be current until the next upload, so the last
        // uploaded values are compared from a copy of their own. Every word
        // and float of `_next` was written above, so the swap leaves `_sent`
        // exact and `_next` free to be overwritten next call.
        if self.fluid_words_sent != self.fluid_words_next {
            self.fluid_words.at().copy_from_slice(&self.fluid_words_next);
            self.fluid_words.upload(device)?;
            std::mem::swap(&mut self.fluid_words_sent, &mut self.fluid_words_next);
        }
        if self.fluid_floats_sent != self.fluid_floats_next {
            self.fluid_floats.at().copy_from_slice(&self.fluid_floats_next);
            self.fluid_floats.upload(device)?;
            std::mem::swap(&mut self.fluid_floats_sent, &mut self.fluid_floats_next);
        }
        Ok(())
    }

    /// Add the field terms evaluated at the Newton iterate `eval_x` (Drag,
    /// Harmonic, a Vortex's damping, a Fluid) to the momentum row's `force`
    /// and `diagonal`, with the step's velocity `(eval_x - current) / dt` and
    /// the step's starting `time`. A no-op when no field has such a term.
    ///
    /// A FLUID'S VOLUME IS READ AT `fluid_time`, THE STEP'S END, BLENDED
    /// between the two frames bracketing it. That is this solver's own
    /// choice, not Blender's: stepping from frame F to F + 1, Blender holds
    /// frame F + 1's grid for every substep of the step. Blending keeps the
    /// push continuous in time where a step ends between two frames, and it
    /// agrees with Blender at every step that ends on a frame.
    ///
    /// A FLUID'S VELOCITY IS DIVIDED BY `playback`, the step's Playback
    /// Speed. Its frames sit on the scene clock, so its velocity is per clock
    /// second, while `playback` physical seconds pass per clock second and
    /// the vertex velocity is per physical second. It is an argument of every
    /// call rather than a value fixed at install, because a dynamic Playback
    /// Speed changes it from step to step.
    #[allow(clippy::too_many_arguments)]
    pub fn implicit(
        &mut self,
        device: &mut impl Device,
        eval_x: Handle,
        current: Handle,
        prop: Handle,
        force: Handle,
        diagonal: Handle,
        time: f32,
        dt: f32,
        playback: f32,
        fluid_time: f32,
    ) -> FatalResult<()> {
        if !self.installed || !self.has_implicit || self.vertices == 0 {
            return Ok(());
        }
        // The kernel divides a fluid's velocity by it, so a Playback Speed
        // that is not a positive float would reach the force as an infinity
        // or a NaN; it is named here instead, with the field, before the
        // dispatch.
        if let Some(fluid) = self.fluids.first() {
            if !(playback > 0.0 && playback.is_finite()) {
                return Err(Fatal::invariant(format!(
                    "solver driver: force field {} is a fluid and the step's playback is \
                     {playback}; its velocity is divided by the playback, which must be \
                     positive and finite",
                    fluid.field
                )));
            }
        }
        self.prepare_fluid(device, fluid_time)?;
        let count = self.vertices as u32;
        let args = crate::driver::kernels::ExternalFieldImplicitArgs {
            eval_x,
            current,
            prop,
            weight: self.weight.handle(),
            has_weight: u32::from(self.has_weight),
            target_mask: self.target_mask.handle(),
            has_mask: u32::from(self.has_mask),
            field_header: self.field_header.handle(),
            field_timing: self.field_timing.handle(),
            field_origin: self.field_origin.handle(),
            field_track: self.field_track.handle(),
            field_count: self.field_count,
            fluid_words: self.fluid_words.handle(),
            fluid_floats: self.fluid_floats.handle(),
            fluid_table: self.fluid_table.handle(),
            fluid_data: self.fluid_data.handle(),
            time,
            dt,
            playback,
            inv_world_scaling: self.inv_world_scaling,
            force,
            diagonal,
            count,
            seam_arena_count: 0,
        };
        // Safety: every handle names a live allocation for the whole call.
        unsafe { device.launch("field.implicit", &args, count) }?;
        Ok(())
    }

    /// The per-vertex acceleration the target adds beside gravity.
    pub fn acceleration(&self) -> Handle {
        self.acceleration.handle()
    }

    /// The per-vertex air velocity and whether the drag should read it.
    pub fn air_velocity(&self) -> (Handle, bool) {
        (self.air_velocity.handle(), self.installed && self.has_air)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::VertexProp;
    use crate::driver::launch::host_device;
    use crate::force_field::test_session::{
        brick, track_row, transform, write_fluid, write_words, FluidSettings, Frame, IDENTITY,
    };
    use crate::force_field::{
        load_dir, FieldRecord, FieldShape, Falloff, Texture, ZDirection, ALL_TARGETS,
        TRACK_NOISE_SCALE, texture_flag,
    };
    use ppf_cts_compute::HostDevice;
    use std::path::Path;

    /// Frame `k` of the streaming volume, which says which frame it is: every
    /// velocity component is `k + 1` (signed per axis) and every density
    /// `10 + k`. Even frames are one 8^3 brick, odd ones two bricks named in
    /// reverse order, and frame 4 is EMPTY, so the slots see tables and
    /// brick counts of every size the volume holds.
    fn streaming_frame(k: usize) -> Frame {
        let value = (k + 1) as f32;
        let cell = [value, -value, 0.5 * value, 10.0 + k as f32];
        let corner = [0.1 * k as f64, -0.05 * k as f64, 0.0];
        let transform = transform(corner, IDENTITY, [0.1, 0.2, 0.3]);
        if k == 4 {
            return Frame { resolution: [0, 0, 0], table: vec![], bricks: vec![], transform };
        }
        if k % 2 == 0 {
            Frame { resolution: [8, 8, 8], table: vec![0], bricks: brick(cell), transform }
        } else {
            Frame {
                resolution: [16, 8, 8],
                table: vec![1, 0],
                bricks: [brick(cell), brick(cell)].concat(),
                transform,
            }
        }
    }

    fn streaming(dir: &Path, settings: &FluidSettings) -> Vec<Frame> {
        let frames: Vec<Frame> = (0..6).map(streaming_frame).collect();
        write_fluid(dir, &frames, settings);
        frames
    }

    /// The field at `dir`, loaded and installed for `vertices` vertices.
    fn installed(dir: &Path, vertices: usize, world_scaling: f32) -> (HostDevice, ForceField) {
        let data = load_dir(dir, vertices).unwrap().expect("the directory carries a field");
        install(&data, vertices, world_scaling).expect("the field installs")
    }

    fn install(
        data: &ForceFieldData,
        vertices: usize,
        world_scaling: f32,
    ) -> FatalResult<(HostDevice, ForceField)> {
        let mut device = host_device();
        let mut field = ForceField::default();
        field.allocate(&mut device, vertices)?;
        field.install(&mut device, data, world_scaling)?;
        Ok((device, field))
    }

    /// Everything the device reads of fluid field 0 agrees with the frames
    /// bracketing `(k0, k1, b)`: each word and float names the slot its
    /// frame sits in and that frame's own placement, and each resident slot
    /// holds its frame's table and its bricks, the velocity scaled by World
    /// Scaling ONCE and the density as it is.
    fn assert_resident(field: &ForceField, frames: &[Frame], (k0, k1, b): (usize, usize, f32)) {
        let fluid = &field.fluids[0];
        let ws = field.world_scaling;
        assert_eq!(field.fluid_floats_sent[0], b, "the blend");
        for (which, k) in [k0, k1].into_iter().enumerate() {
            let slot = fluid
                .resident
                .iter()
                .position(|&r| r == Some(k))
                .unwrap_or_else(|| panic!("frame {k} is not resident: {:?}", fluid.resident));
            let w = FLUID_FRAME_WORDS * which;
            let words = &field.fluid_words_sent[w..w + FLUID_FRAME_WORDS];
            assert_eq!(words[0] as usize, fluid.table_base[slot]);
            assert_eq!(words[1] as usize, fluid.data_base[slot]);
            assert_eq!(&words[2..5], &frames[k].resolution);
            let f = 1 + FLUID_FRAME_FLOATS * which;
            let floats = &field.fluid_floats_sent[f..f + FLUID_FRAME_FLOATS];
            for (c, value) in floats[..FLUID_FRAME_CORNER].iter().enumerate() {
                assert_eq!(*value, frames[k].transform[3 + c] as f32, "frame {k} float {c}");
            }
            for axis in 0..3 {
                let corner = (frames[k].transform[axis] * ws as f64) as f32;
                assert_eq!(floats[FLUID_FRAME_CORNER + axis], corner, "frame {k} axis {axis}");
            }
            let tb = fluid.table_base[slot];
            let table = &field.fluid_table.host()[tb..tb + frames[k].table.len()];
            assert_eq!(table, &frames[k].table[..], "frame {k}'s table in slot {slot}");
            let db = fluid.data_base[slot];
            let data = &field.fluid_data.host()[db..db + frames[k].bricks.len()];
            for (i, (got, raw)) in data.iter().zip(&frames[k].bricks).enumerate() {
                let want = if i % FLUID_CHANNELS == 3 { *raw } else { raw * ws };
                assert_eq!(*got, want, "frame {k} float {i} in slot {slot}");
            }
        }
    }

    fn prepare(device: &mut HostDevice, field: &mut ForceField, time: f32) {
        field.prepare_fluid(device, time).expect("the fluid prepares");
    }

    /// FORWARD STREAMING READS EACH FRAME ONCE: a step crossing a fluid frame
    /// copies the ONE new frame into the slot of the frame it leaves, keeps
    /// the other, and a step inside a bracket reads nothing.
    #[test]
    fn a_forward_stream_reads_each_frame_once() {
        let dir = tempfile::tempdir().unwrap();
        let frames = streaming(dir.path(), &FluidSettings::default());
        let (mut device, mut field) = installed(dir.path(), 1, 1.5);
        let reads = |field: &ForceField| field.fluids[0].disk_reads;
        prepare(&mut device, &mut field, 0.05);
        assert_resident(&field, &frames, (0, 1, 0.5));
        assert_eq!(reads(&field), 2);
        prepare(&mut device, &mut field, 0.07);
        assert_eq!(reads(&field), 2, "a step inside the bracket read the disk");
        let mut previous = field.fluids[0].resident;
        for (k, time) in [(1usize, 0.15f32), (2, 0.25), (3, 0.35), (4, 0.45)] {
            prepare(&mut device, &mut field, time);
            let (k0, k1, b) = field.fluids[0].volume.bracket(time);
            assert_eq!((k0, k1), (k, k + 1));
            assert_resident(&field, &frames, (k0, k1, b));
            assert_eq!(reads(&field), k + 2, "crossing into frame {} read more than it", k + 1);
            // The frame the step keeps stays in its slot.
            let kept = previous.iter().position(|&r| r == Some(k)).unwrap();
            assert_eq!(field.fluids[0].resident[kept], Some(k));
            previous = field.fluids[0].resident;
        }
        // Past the end the last frame holds in both bracket positions, read
        // from the slot it already sits in.
        prepare(&mut device, &mut field, 3.0);
        assert_resident(&field, &frames, (5, 5, 0.0));
        assert_eq!(reads(&field), 6);
        let w = &field.fluid_words_sent;
        assert_eq!(w[..FLUID_FRAME_WORDS], w[FLUID_FRAME_WORDS..FLUID_WORDS]);
    }

    /// A JUMP OVER SEVERAL FRAMES REPLACES BOTH SLOTS IN ONE CALL, and a
    /// jump onto an empty frame leaves its slot naming a zero domain.
    #[test]
    fn a_jump_over_several_frames_replaces_both_slots_in_one_call() {
        let dir = tempfile::tempdir().unwrap();
        let frames = streaming(dir.path(), &FluidSettings::default());
        let (mut device, mut field) = installed(dir.path(), 1, 1.0);
        prepare(&mut device, &mut field, 0.05);
        assert_eq!(field.fluids[0].resident, [Some(0), Some(1)]);
        let time = 0.425;
        prepare(&mut device, &mut field, time);
        let bracket = field.fluids[0].volume.bracket(time);
        assert_eq!((bracket.0, bracket.1), (4, 5));
        assert_resident(&field, &frames, bracket);
        let mut resident = field.fluids[0].resident.to_vec();
        resident.sort();
        assert_eq!(resident, vec![Some(4), Some(5)]);
        assert_eq!(field.fluids[0].disk_reads, 4);
        let slot = field.fluids[0].resident.iter().position(|&r| r == Some(4)).unwrap();
        assert_eq!(field.fluids[0].table_base[slot], field.fluid_words_sent[0] as usize);
        assert_eq!(&field.fluid_words_sent[2..5], &[0, 0, 0]);
    }

    /// THE TOI SWING COSTS NO DISK READ. A step's Newton iterations read the
    /// volume at its full end and its trailing iteration at the end the line
    /// search committed, so a cut step brackets (2, 3), then (1, 2), then the
    /// next step (2, 3) again. Frames 1 and 3 come back from the host cache:
    /// their files are deleted before the swing, so a disk read would fail,
    /// and the velocity is scaled from the cached raw frame each time, never
    /// twice.
    #[test]
    fn a_toi_cut_swing_across_a_frame_boundary_reads_nothing_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let frames = streaming(dir.path(), &FluidSettings::default());
        let (mut device, mut field) = installed(dir.path(), 1, 2.0);
        prepare(&mut device, &mut field, 0.15);
        prepare(&mut device, &mut field, 0.25);
        assert_eq!(field.fluids[0].disk_reads, 3);
        let volume = field.fluids[0].volume.clone();
        for k in [1, 3] {
            std::fs::remove_file(&volume.tables[k]).unwrap();
            std::fs::remove_file(&volume.bricks[k]).unwrap();
        }
        for time in [0.18f32, 0.28, 0.19, 0.29, 0.195] {
            prepare(&mut device, &mut field, time);
            assert_resident(&field, &frames, field.fluids[0].volume.bracket(time));
        }
        assert_eq!(field.fluids[0].disk_reads, 3, "the swing read a frame from disk");
        // Moving on reads the next frame and evicts the cached frame farthest
        // from the bracket, which is frame 1.
        prepare(&mut device, &mut field, 0.35);
        assert_eq!(field.fluids[0].disk_reads, 4);
        let cached: Vec<Option<usize>> = field.fluids[0].cache.iter().map(|c| c.frame).collect();
        assert!(!cached.contains(&Some(1)) && cached.contains(&Some(4)), "{cached:?}");
    }

    /// BEFORE THE FIRST FRAME the first frame holds, in ONE slot that both
    /// bracket positions name, and only it is read.
    #[test]
    fn before_the_first_frame_the_first_frame_holds_in_one_slot() {
        let dir = tempfile::tempdir().unwrap();
        let settings = FluidSettings { start: 0.5, ..FluidSettings::default() };
        let frames = streaming(dir.path(), &settings);
        let (mut device, mut field) = installed(dir.path(), 1, 1.0);
        prepare(&mut device, &mut field, 0.1);
        assert_resident(&field, &frames, (0, 0, 0.0));
        assert_eq!(field.fluids[0].disk_reads, 1);
        assert_eq!(field.fluids[0].resident.iter().filter(|r| r.is_some()).count(), 1);
        let w = &field.fluid_words_sent;
        assert_eq!(w[..FLUID_FRAME_WORDS], w[FLUID_FRAME_WORDS..FLUID_WORDS]);
        prepare(&mut device, &mut field, 0.55);
        assert_resident(&field, &frames, field.fluids[0].volume.bracket(0.55));
        assert_eq!(field.fluids[0].disk_reads, 2);
    }

    /// A FRAME RE-READ WITH A COUNT OTHER THAN ITS LOAD'S IS A FATAL, naming
    /// the field, the frame, the file and both counts, and the slot it was
    /// headed for is left naming no frame rather than half of one.
    #[test]
    fn a_reread_frame_whose_brick_count_changed_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        streaming(dir.path(), &FluidSettings::default());
        let (mut device, mut field) = installed(dir.path(), 1, 1.0);
        prepare(&mut device, &mut field, 0.05);
        // Frame 2 held one brick at load; it now holds a self-consistent two.
        let path = field.fluids[0].volume.bricks[2].clone();
        write_words(&path, &[brick([1.0; 4]), brick([2.0; 4])].concat());
        write_words(&field.fluids[0].volume.tables[2], &[1u32]);
        let fatal = field.prepare_fluid(&mut device, 0.15).unwrap_err();
        assert!(
            fatal.detail.contains("force field 0 fluid frame 2 no longer reads as it did at load")
                && fatal.detail.contains("bricks-00002.bin): 2 bricks (4096 floats), 1 at load"),
            "{}",
            fatal.detail
        );
        assert!(!field.fluids[0].resident.contains(&Some(2)));
        assert!(!field.fluids[0].resident.contains(&Some(0)));
        assert!(field.fluids[0].cache.iter().all(|c| c.frame != Some(2)));
    }

    /// INSTALL HOLDS EVERY FRAME, not only the ones a step brackets: a
    /// domain corner past the origin's reach on the LAST frame, and a
    /// velocity World Scaling takes past single precision, stop the run
    /// before its first step, naming the field by its MANIFEST index.
    #[test]
    fn install_refuses_a_late_frame_past_reach_or_past_single_precision() {
        let dir = tempfile::tempdir().unwrap();
        streaming(dir.path(), &FluidSettings::default());
        let mut data = load_dir(dir.path(), 1).unwrap().unwrap();
        // A Force field ahead of the fluid, so the fluid is manifest field 1.
        let mut force = data.fields[0].clone();
        force.kind = FieldType::Force;
        force.fluid = None;
        data.fields.insert(0, force);
        install(&data, 1, 2.0).expect("the volume as written installs");

        let mut far = data.clone();
        far.fields[1].fluid.as_mut().unwrap().transform[5][0] = 6.0e5;
        let fatal = install(&far, 1, 2.0).err().expect("a corner past reach installs");
        assert!(
            fatal.detail.contains("force field 1 fluid frame 5 has its domain corner at 1200000"),
            "{}",
            fatal.detail
        );

        let mut fast = data.clone();
        fast.fields[1].fluid.as_mut().unwrap().velocity_bound[3] = 3.0e38;
        let fatal = install(&fast, 1, 2.0).err().expect("an overflowing velocity installs");
        assert!(
            fatal.detail.contains("force field 1 fluid frame 3 holds a velocity of"),
            "{}",
            fatal.detail
        );
        install(&fast, 1, 1.0).expect("the same velocity at World Scaling 1 is a float");

        // A volume that skipped the loader's measurement has nothing to size
        // its slots from.
        let mut unmeasured = data.clone();
        unmeasured.fields[1].fluid.as_mut().unwrap().velocity_bound.clear();
        let fatal = install(&unmeasured, 1, 1.0).err().expect("an unmeasured volume installs");
        assert!(
            fatal.detail.contains("force field 1 is a fluid of 6 frames with 6 brick counts and 0"),
            "{}",
            fatal.detail
        );
    }

    /// A buffer of `values` on the device.
    fn device_buffer<T: Pod + Default>(device: &mut HostDevice, values: &[T]) -> Buffer<T> {
        let mut buffer = Buffer::none();
        buffer.size(device, values.len(), AllocLabel("test.field")).unwrap();
        if !values.is_empty() {
            buffer.write(device, 0, values).unwrap();
        }
        buffer
    }

    /// Scene-unit points as the device holds positions, at `world_scaling`.
    fn positions(points: &[[f64; 3]], world_scaling: f64) -> Vec<f32> {
        points
            .iter()
            .flat_map(|p| p.map(|x| (x * world_scaling) as f32))
            .collect()
    }

    fn read<T: Pod + Default + Clone>(device: &mut HostDevice, buffer: &Buffer<T>, n: usize) -> Vec<T> {
        let mut out = vec![T::default(); n];
        buffer.read(device, 0, &mut out).unwrap();
        out
    }

    /// One `external_field_implicit` dispatch over `points` (scene units) at
    /// rest, `eval_x == current`, at Playback Speed `playback`, returning each
    /// vertex's force and diagonal block (nine floats).
    fn implicit_at(
        data: &ForceFieldData,
        points: &[[f64; 3]],
        world_scaling: f32,
        playback: f32,
        fluid_time: f32,
    ) -> FatalResult<(Vec<f32>, Vec<f32>)> {
        let n = points.len();
        let (mut device, mut field) = install(data, n, world_scaling).unwrap();
        let placed = positions(points, world_scaling as f64);
        let position = device_buffer(&mut device, &placed);
        let props = vec![VertexProp { mass: 1.0, ..VertexProp::default() }; n];
        let prop = device_buffer(&mut device, &props);
        let force = device_buffer(&mut device, &vec![0.0f32; 3 * n]);
        let diagonal = device_buffer(&mut device, &vec![0.0f32; 9 * n]);
        field.implicit(
            &mut device,
            position.handle(),
            position.handle(),
            prop.handle(),
            force.handle(),
            diagonal.handle(),
            0.0,
            0.01,
            playback,
            fluid_time,
        )?;
        Ok((read(&mut device, &force, 3 * n), read(&mut device, &diagonal, 9 * n)))
    }

    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        for (c, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= 1e-4 * (1.0 + w.abs()), "{what} component {c}: {g} against {w}");
        }
    }

    /// THE KERNEL READS THE CELL PER AXIS AND U IN SOLVER UNITS PER PHYSICAL
    /// SECOND. One frame of an 8^3 domain whose cell is 0.1, 0.2 and 0.05
    /// along x, y and z, so it spans 0.8 x 1.6 x 0.4 scene units, a uniform
    /// velocity (1, 2, 3) scene units per clock second and Strength 1, Flow
    /// 1, no falloff. A vertex at rest inside reads `f = -m U ws / playback`
    /// and the damping `m / dt`, which no Playback Speed touches; past any
    /// face it reads nothing, damping included. A vertex at y = 1.5 is
    /// inside only because y has its own cell. A NEGATIVE weight reverses the
    /// push and adds no Hessian.
    #[test]
    fn the_fluid_push_reads_a_cell_per_axis_and_velocity_in_solver_units() {
        let dir = tempfile::tempdir().unwrap();
        let frame = Frame {
            resolution: [8, 8, 8],
            table: vec![0],
            bricks: brick([1.0, 2.0, 3.0, 0.5]),
            transform: transform([0.0; 3], IDENTITY, [0.1, 0.2, 0.05]),
        };
        write_fluid(dir.path(), &[frame], &FluidSettings::default());
        let mut data = load_dir(dir.path(), 5).unwrap().unwrap();
        data.weight = Some(vec![1.0, 1.0, 1.0, 1.0, -1.0]);
        let points = [
            [0.7, 1.5, 0.35],
            [0.7, 1.7, 0.35],
            [0.85, 0.5, 0.2],
            [0.05, 0.1, 0.025],
            [0.7, 1.5, 0.35],
        ];
        for (ws, playback) in [(1.0f32, 1.0f32), (2.0, 1.0), (2.0, 0.5), (1.0, 4.0)] {
            let (force, diagonal) = implicit_at(&data, &points, ws, playback, 0.0).unwrap();
            let push = [-ws / playback, -2.0 * ws / playback, -3.0 * ws / playback];
            let damping = 1.0 / 0.01;
            let what = format!("ws {ws} playback {playback}");
            for (i, (want, h)) in [
                (push, damping),
                ([0.0; 3], 0.0),
                ([0.0; 3], 0.0),
                (push, damping),
                (push.map(|x| -x), 0.0),
            ]
            .into_iter()
            .enumerate()
            {
                assert_close(&force[3 * i..3 * i + 3], &want, &format!("{what} vertex {i} force"));
                let block = &diagonal[9 * i..9 * i + 9];
                let want_block = [h, 0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0, h];
                assert_close(block, &want_block, &format!("{what} vertex {i} diagonal"));
            }
        }
    }

    /// A FLUID UNDER A PLAYBACK SPEED THAT IS NOT A POSITIVE FLOAT IS A FATAL
    /// before the dispatch, naming the field by its manifest index and the
    /// value, since its velocity is divided by it. A scene with no fluid
    /// reads no playback in the field, so the same value passes there.
    #[test]
    fn a_fluid_refuses_a_playback_that_is_not_positive() {
        let dir = tempfile::tempdir().unwrap();
        let frame = Frame {
            resolution: [8, 8, 8],
            table: vec![0],
            bricks: brick([1.0, 0.0, 0.0, 0.0]),
            transform: transform([0.0; 3], IDENTITY, [0.1; 3]),
        };
        write_fluid(dir.path(), &[frame], &FluidSettings::default());
        let data = load_dir(dir.path(), 1).unwrap().unwrap();
        let points = [[0.4, 0.4, 0.4]];
        for playback in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let fatal = implicit_at(&data, &points, 1.0, playback, 0.0)
                .err()
                .unwrap_or_else(|| panic!("playback {playback} dispatched"));
            assert!(
                fatal.detail.contains(&format!(
                    "force field 0 is a fluid and the step's playback is {playback}"
                )),
                "{}",
                fatal.detail
            );
        }
        let mut drag = data.fields[0].clone();
        drag.kind = FieldType::Drag;
        drag.fluid = None;
        let no_fluid = ForceFieldData { fields: vec![drag], ..ForceFieldData::default() };
        implicit_at(&no_fluid, &points, 1.0, 0.0, 0.0).expect("a drag reads no playback");
    }

    /// AN EMPTY FRAME IS ABSENT FLUID. A volume of one empty frame (zero-length
    /// tables and data) pushes nothing anywhere and dispatches cleanly, the
    /// point exactly on its corner included: there `g = 0` passes a `[0, n]`
    /// test with `n = 0`, so only the kernel's empty-frame guard keeps it from
    /// reading a table that does not exist. Halfway between a full frame and
    /// an empty one the velocity and the presence each blend to one half, so
    /// the push, their product, is a quarter and the damping a half, as the
    /// reference evaluator blends them.
    #[test]
    fn an_empty_frame_is_absent_fluid() {
        let full = Frame {
            resolution: [8, 8, 8],
            table: vec![0],
            bricks: brick([1.0, 0.0, 0.0, 0.0]),
            transform: transform([0.0; 3], IDENTITY, [0.1; 3]),
        };
        let empty = Frame {
            resolution: [0, 0, 0],
            table: vec![],
            bricks: vec![],
            transform: transform([0.0; 3], IDENTITY, [0.1; 3]),
        };
        let points = [[0.4, 0.4, 0.4], [0.0, 0.0, 0.0]];

        let dir = tempfile::tempdir().unwrap();
        write_fluid(dir.path(), &[empty.clone()], &FluidSettings::default());
        let data = load_dir(dir.path(), 2).unwrap().unwrap();
        let (force, diagonal) = implicit_at(&data, &points, 1.0, 1.0, 0.0).unwrap();
        assert!(force.iter().chain(&diagonal).all(|v| *v == 0.0), "{force:?} {diagonal:?}");

        let dir = tempfile::tempdir().unwrap();
        write_fluid(dir.path(), &[full, empty], &FluidSettings::default());
        let data = load_dir(dir.path(), 2).unwrap().unwrap();
        let (force, diagonal) = implicit_at(&data, &points, 1.0, 1.0, 0.05).unwrap();
        for i in 0..2 {
            assert_close(&force[3 * i..3 * i + 3], &[-0.25, 0.0, 0.0], "half present force");
            assert_close(&diagonal[9 * i..9 * i + 1], &[50.0], "half present damping");
        }
    }

    /// EVERY IMPLICIT HESSIAN TERM IS GATED ON `m * weight * coefficient`.
    /// A Drag and a Harmonic spring with damping, at a vertex moving through
    /// them, with weight 1 and weight -1: the negative weight reverses every
    /// gradient exactly and contributes no Hessian at all.
    #[test]
    fn a_negative_weight_reverses_every_implicit_term_and_adds_no_hessian() {
        let base = |kind: FieldType| {
            let mut row = track_row(2.0, 0.0);
            row[22] = 0.5; // linear drag
            row[23] = 0.25; // quadratic drag
            row[24] = 0.3; // harmonic damping
            row[25] = 0.1; // rest length
            FieldRecord {
                kind,
                shape: FieldShape::Point,
                falloff: Falloff::Sphere,
                z_direction: ZDirection::Both,
                flags: 0,
                seed: 0,
                start: 0.0,
                step: 0.04,
                origin: vec![[0.0; 3]],
                track: row,
                target: ALL_TARGETS,
                texture: None,
                fluid: None,
            }
        };
        let data = ForceFieldData {
            fields: vec![base(FieldType::Drag), base(FieldType::Harmonic)],
            scripts: vec![],
            weight: Some(vec![1.0, -1.0]),
            target_mask: None,
        };
        let n = 2;
        let (mut device, mut field) = install(&data, n, 1.0).unwrap();
        let start = positions(&[[0.3, 0.2, -0.4], [0.3, 0.2, -0.4]], 1.0);
        let moved = positions(&[[0.31, 0.19, -0.38], [0.31, 0.19, -0.38]], 1.0);
        let current = device_buffer(&mut device, &start);
        let eval_x = device_buffer(&mut device, &moved);
        let props = vec![VertexProp { mass: 1.0, ..VertexProp::default() }; n];
        let prop = device_buffer(&mut device, &props);
        let force = device_buffer(&mut device, &vec![0.0f32; 3 * n]);
        let diagonal = device_buffer(&mut device, &vec![0.0f32; 9 * n]);
        field
            .implicit(
                &mut device,
                eval_x.handle(),
                current.handle(),
                prop.handle(),
                force.handle(),
                diagonal.handle(),
                0.0,
                0.01,
                1.0,
                0.0,
            )
            .unwrap();
        let force = read(&mut device, &force, 3 * n);
        let diagonal = read(&mut device, &diagonal, 9 * n);
        assert!(diagonal[..9].iter().any(|v| *v != 0.0), "the positive weight added no Hessian");
        assert!(
            (0..3).all(|r| diagonal[4 * r] > 0.0),
            "the positive weight's diagonal is not positive: {:?}",
            &diagonal[..9]
        );
        assert!(diagonal[9..].iter().all(|v| *v == 0.0), "a negative weight added {:?}", &diagonal[9..]);
        let reversed: Vec<f32> = force[..3].iter().map(|v| -v).collect();
        assert_close(&force[3..], &reversed, "the negative weight's force");
        assert!(force[..3].iter().any(|v| v.abs() > 1e-3), "{force:?}");
    }

    /// A TEXTURE WITH Z UP READS BLENDER'S AXES. A 2 x 2 image whose top row
    /// is red and whose bottom row is black, under two points on Blender's
    /// horizontal plane, Blender y = 0.5 and y = -0.5, which are the solver's
    /// z = -0.5 and z = 0.5. With Z up the lookup reads the image at
    /// Blender's (x, y), so the two read different rows, and the push
    /// `0.5 - color` (a Blender vector) arrives in solver axes, Blender's
    /// (x, y, z) at the solver's (x, z, -y). In solver axes both points read
    /// the image at the same y = 0 and the same row. In object coordinates
    /// the lookup is the field's own frame either way and only the output
    /// turns.
    #[test]
    fn a_z_up_texture_reads_blender_axes_and_pushes_in_solver_axes() {
        let image = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let points = [[0.5, 0.0, -0.5], [0.5, 0.0, 0.5]];
        let red = [-0.5f32, 0.5, 0.5];
        let black = [0.5f32, 0.5, 0.5];
        let turned = |v: [f32; 3]| [v[0], v[2], -v[1]];
        for (flags, want) in [
            (texture_flag::Z_UP, [turned(red), turned(black)]),
            (0, [red, red]),
            (texture_flag::Z_UP | texture_flag::OBJECT_COORDINATES, [turned(red), turned(red)]),
        ] {
            let record = FieldRecord {
                kind: FieldType::Texture,
                shape: FieldShape::Point,
                falloff: Falloff::Sphere,
                z_direction: ZDirection::Both,
                flags: 0,
                seed: 0,
                start: 0.0,
                step: 0.04,
                origin: vec![[0.0; 3]],
                track: track_row(1.0, 0.0),
                target: ALL_TARGETS,
                texture: Some(Texture {
                    width: 2,
                    height: 2,
                    mode: 0,
                    extension: 0,
                    flags,
                    pixels: image.clone(),
                }),
                fluid: None,
            };
            crate::force_field::verify_field(&record).unwrap();
            let data = ForceFieldData { fields: vec![record], ..ForceFieldData::default() };
            let n = points.len();
            let (mut device, mut field) = install(&data, n, 1.0).unwrap();
            let placed = positions(&points, 1.0);
            let position = device_buffer(&mut device, &placed);
            let props = vec![VertexProp { mass: 1.0, ..VertexProp::default() }; n];
            let prop = device_buffer(&mut device, &props);
            let none: Buffer<u32> = device_buffer(&mut device, &[]);
            field
                .evaluate(
                    &mut device,
                    position.handle(),
                    position.handle(),
                    0.01,
                    prop.handle(),
                    (none.handle(), none.handle(), 0, none.handle()),
                    0.0,
                )
                .unwrap();
            let acceleration = read(&mut device, &field.acceleration, 3 * n);
            for (i, w) in want.iter().enumerate() {
                assert_close(&acceleration[3 * i..3 * i + 3], w, &format!("flags {flags} point {i}"));
            }
        }
    }

    /// AN OBJECT COORDINATES TEXTURE READS THE OBJECT'S SCALED SPACE: its
    /// image at the point in the field's local frame divided axis by axis by
    /// the object's scale (the noise scale), as Blender reads it under the
    /// object's full inverse matrix. A 4 x 4 image whose pixel (column, row)
    /// is `(0.1 column, 0.1 row, 0)`, no interpolation, Clip Cube, on a field
    /// object offset from the origin, turned a quarter about Z and scaled
    /// (2, 0.5, 0.5). The local point (0.8, 0.3, 0) reads column 2, row 3 at
    /// (0.4, 0.6, 0), where the unscaled point would read column 3, row 2; the
    /// local point (0.1, 0.1, 0.8) sits at z = 1.6, outside the cube, where
    /// the unscaled point is inside. World coordinates ignore the scale: the
    /// first point's world position (-0.1, 0.7, 0.3) reads column 1, row 3.
    #[test]
    fn an_object_coordinates_texture_reads_the_scaled_object_space() {
        let mut image = Vec::new();
        for row in 0..4 {
            for column in 0..4 {
                image.extend_from_slice(&[0.1 * column as f32, 0.1 * row as f32, 0.0]);
            }
        }
        let origin = [0.2, -0.1, 0.3];
        let local = [[0.8, 0.3, 0.0], [0.1, 0.1, 0.8]];
        // Local X along world Y, local Y along world -X.
        let points: Vec<[f64; 3]> = local
            .iter()
            .map(|p| [origin[0] - p[1], origin[1] + p[0], origin[2] + p[2]])
            .collect();
        let push = |c: [f32; 3]| c.map(|v| 0.5 - v);
        for (flags, want) in [
            (
                texture_flag::OBJECT_COORDINATES,
                [push([0.2, 0.3, 0.0]), push([0.0; 3])],
            ),
            (0, [push([0.1, 0.3, 0.0]), push([0.0; 3])]),
        ] {
            let mut track = track_row(1.0, 0.0);
            track[..9].copy_from_slice(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
            track[TRACK_NOISE_SCALE..TRACK_NOISE_SCALE + 3].copy_from_slice(&[2.0, 0.5, 0.5]);
            let record = FieldRecord {
                kind: FieldType::Texture,
                shape: FieldShape::Point,
                falloff: Falloff::Sphere,
                z_direction: ZDirection::Both,
                flags: 0,
                seed: 0,
                start: 0.0,
                step: 0.04,
                origin: vec![origin],
                track,
                target: ALL_TARGETS,
                texture: Some(Texture {
                    width: 4,
                    height: 4,
                    mode: 0,
                    extension: 3,
                    flags,
                    pixels: image.clone(),
                }),
                fluid: None,
            };
            crate::force_field::verify_field(&record).unwrap();
            let data = ForceFieldData { fields: vec![record], ..ForceFieldData::default() };
            let n = points.len();
            let (mut device, mut field) = install(&data, n, 1.0).unwrap();
            let placed = positions(&points, 1.0);
            let position = device_buffer(&mut device, &placed);
            let props = vec![VertexProp { mass: 1.0, ..VertexProp::default() }; n];
            let prop = device_buffer(&mut device, &props);
            let none: Buffer<u32> = device_buffer(&mut device, &[]);
            field
                .evaluate(
                    &mut device,
                    position.handle(),
                    position.handle(),
                    0.01,
                    prop.handle(),
                    (none.handle(), none.handle(), 0, none.handle()),
                    0.0,
                )
                .unwrap();
            let acceleration = read(&mut device, &field.acceleration, 3 * n);
            for (i, w) in want.iter().enumerate() {
                assert_close(&acceleration[3 * i..3 * i + 3], w, &format!("flags {flags} point {i}"));
            }
        }
    }
}
