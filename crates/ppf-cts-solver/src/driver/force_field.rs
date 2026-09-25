// File: crates/ppf-cts-solver/src/driver/force_field.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The external force field on the device.
//!
//! The host reads and verifies the field once (`crate::force_field`) and
//! installs it here; it is uploaded ONCE and never re-staged per step, because
//! a grid and a script are authored inputs rather than per-step state. Every
//! step then runs ONE dispatch, `external_field`, over the vertices at the
//! step's starting position and time, and the two outputs are what the
//! inertial target and the aerodynamic drag read.
//!
//! THE ACCELERATION BUFFER ALWAYS EXISTS, zero-filled, because the target
//! kernel gathers it unconditionally: a scene with no field pays one 12-byte
//! read per vertex and adds exactly zero, which is one code path rather than a
//! second entry point for "no field". Everything else takes a real zero-length
//! handle when absent, never `Handle::NONE`, which a generated entry asserts
//! against.

use ppf_cts_compute::{AllocLabel, Buffer, Device, Handle, Pod, ReadbackBuffer, StagedBuffer};

use super::scene::{Fatal, FatalResult};
use crate::force_field::{
    ForceFieldData, GRID_BOX_FLOATS, GRID_HEADER_WORDS, SCRIPT_HEADER_WORDS,
};

#[derive(Default)]
pub struct ForceField {
    installed: bool,
    vertices: usize,
    grid_count: u32,
    script_count: u32,
    has_air: bool,
    has_weight: bool,
    has_mask: bool,
    /// `1 / world_scaling`: the field is authored in scene units.
    inv_world_scaling: f32,
    grid_header: StagedBuffer<u32>,
    grid_box: StagedBuffer<f32>,
    grid_times: StagedBuffer<f32>,
    grid_data: StagedBuffer<f32>,
    script_header: StagedBuffer<u32>,
    script_code: StagedBuffer<u32>,
    script_constants: StagedBuffer<f32>,
    weight: StagedBuffer<f32>,
    target_mask: StagedBuffer<u32>,
    /// `3 * vertices` floats, the acceleration the target adds beside gravity.
    acceleration: Buffer<f32>,
    /// `3 * vertices` floats when an air-velocity grid exists, else empty.
    air_velocity: Buffer<f32>,
    /// One flag per vertex: free, and outside every grid. Read back only when
    /// a frame is written, to count.
    outside: ReadbackBuffer<u32>,
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
        self.outside.size(device, vertices, AllocLabel("field.outside"))?;
        if !self.installed {
            stage::<u32>(device, &mut self.grid_header, &[], "field.grid_header")?;
            stage::<f32>(device, &mut self.grid_box, &[], "field.grid_box")?;
            stage::<f32>(device, &mut self.grid_times, &[], "field.grid_times")?;
            stage::<f32>(device, &mut self.grid_data, &[], "field.grid_data")?;
            stage::<u32>(device, &mut self.script_header, &[], "field.script_header")?;
            stage::<u32>(device, &mut self.script_code, &[], "field.script_code")?;
            stage::<f32>(device, &mut self.script_constants, &[], "field.script_constants")?;
            stage::<f32>(device, &mut self.weight, &[], "field.weight")?;
            stage::<u32>(device, &mut self.target_mask, &[], "field.target_mask")?;
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
        let mut header: Vec<u32> = Vec::with_capacity(GRID_HEADER_WORDS * field.grids.len());
        let mut boxes: Vec<f32> = Vec::with_capacity(GRID_BOX_FLOATS * field.grids.len());
        let mut times: Vec<f32> = Vec::new();
        let mut data: Vec<f32> = Vec::with_capacity(field.grid_bytes() / 4);
        for grid in &field.grids {
            let data_offset = u32::try_from(data.len()).map_err(|_| {
                Fatal::invariant(
                    "solver driver: the force field's grids exceed the kernel's 32-bit offsets",
                )
            })?;
            header.extend_from_slice(&[
                grid.dims[0],
                grid.dims[1],
                grid.dims[2],
                grid.dims[3],
                grid.kind.code(),
                data_offset,
                times.len() as u32,
                grid.target,
            ]);
            boxes.extend_from_slice(&grid.min);
            boxes.extend_from_slice(&grid.max);
            times.extend_from_slice(&grid.times);
            data.extend_from_slice(&grid.data);
        }
        stage(device, &mut self.grid_header, &header, "field.grid_header")?;
        stage(device, &mut self.grid_box, &boxes, "field.grid_box")?;
        stage(device, &mut self.grid_times, &times, "field.grid_times")?;
        stage(device, &mut self.grid_data, &data, "field.grid_data")?;
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
        self.air_velocity.size(
            device,
            if self.has_air { 3 * vertices } else { 0 },
            AllocLabel("field.air_velocity"),
        )?;
        self.grid_count = field.grids.len() as u32;
        self.script_count = field.scripts.len() as u32;
        self.has_weight = field.weight.is_some();
        self.has_mask = field.target_mask.is_some();
        self.installed = true;
        Ok(())
    }

    /// Evaluate the field for this step, at `positions` (the positions the
    /// step starts from) and `time` (its starting clock).
    pub fn evaluate(
        &mut self,
        device: &mut impl Device,
        positions: Handle,
        prop: Handle,
        time: f32,
    ) -> FatalResult<()> {
        if !self.installed || self.vertices == 0 {
            return Ok(());
        }
        let count = self.vertices as u32;
        let args = crate::driver::kernels::ExternalFieldArgs {
            current: positions,
            prop,
            weight: self.weight.handle(),
            has_weight: u32::from(self.has_weight),
            target_mask: self.target_mask.handle(),
            has_mask: u32::from(self.has_mask),
            grid_header: self.grid_header.handle(),
            grid_box: self.grid_box.handle(),
            grid_times: self.grid_times.handle(),
            grid_data: self.grid_data.handle(),
            grid_count: self.grid_count,
            script_header: self.script_header.handle(),
            script_count: self.script_count,
            script_code: self.script_code.handle(),
            script_constants: self.script_constants.handle(),
            time,
            inv_world_scaling: self.inv_world_scaling,
            acceleration: self.acceleration.handle(),
            air_velocity: self.air_velocity.handle(),
            has_air: u32::from(self.has_air),
            outside: self.outside.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: every handle names a live allocation for the whole call.
        unsafe { device.launch("field.external", &args, count) }?;
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

    /// How many free vertices the last evaluation found outside every grid,
    /// or `None` when no grid is installed and the question has no meaning.
    pub fn outside_count(&mut self, device: &mut impl Device) -> FatalResult<Option<usize>> {
        if !self.installed || self.grid_count == 0 {
            return Ok(None);
        }
        self.outside.download(device)?;
        Ok(Some(self.outside.host().iter().filter(|&&v| v != 0).count()))
    }
}
