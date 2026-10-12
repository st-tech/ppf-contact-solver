// File: force_field.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The external force field's session inputs: field objects, compiled
//! scripts, a per-vertex weight and a per-vertex target mask, read from
//! `<session>/bin/force_field/`.
//!
//! A FIELD OBJECT IS A RECORD, NOT A SAMPLING. Each one is a Blender-style
//! field (Force, Wind, Vortex, Turbulence, Magnetic) given by its discrete
//! settings and a TRACK of per-frame samples (origin, axes, the animatable
//! values), which the kernel evaluates exactly at each vertex. What each type
//! computes was measured on Blender as a black box; nothing per vertex or
//! per volume is shipped for it.
//!
//! EACH SOURCE CARRIES A TARGET BIT. A field or a script meant for some
//! objects only names a bit below [`ALL_TARGETS`], and the per-vertex mask
//! says which bits reach each vertex; [`ALL_TARGETS`] reaches every vertex,
//! and a field whose sources all do ships no mask.
//!
//! EVERYTHING IS VERIFIED HERE, BEFORE THE DEVICE SEES IT. The kernel
//! (`kernels/energy/external_field.kernel.cpp`) reads the records and runs the
//! bytecode on trust, with guarantee-class asserts as the only net, so a
//! malformed file must stop the run at load with the file and field named,
//! never reach a dispatch. That includes a bytecode file that did not come from
//! the frontend's compiler: the proof below (forward jumps only, every operand
//! in range, the stack depth bounded on every path, every path returning) is
//! what the kernel's termination and its fixed-size stack rest on.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// The directory under the session holding the field's files.
pub const FIELD_DIR: &str = "bin/force_field";
/// The manifest inside it.
pub const MANIFEST: &str = "force_field.toml";
/// The bytecode format this solver runs. The frontend writes the same number
/// and a mismatch is refused, so a program is never run under a table it was
/// not compiled against.
pub const SCRIPT_VERSION: u32 = 3;

/// The interpreter's thread-local capacities, `FIELD_STACK` and `FIELD_VARS`
/// in the kernel.
pub const STACK_CAPACITY: usize = 32;
pub const VAR_CAPACITY: u32 = 64;

/// Words per field in the header table, `FIELD_HEADER` in the kernel: type,
/// shape, falloff, Z direction, flags, seed, target bit, first track sample,
/// track sample count, then a Texture's image offset, width, height, mode,
/// extension and texture flags.
pub const FIELD_HEADER_WORDS: usize = 15;
/// Floats per field in the timing table, `FIELD_TIMING` in the kernel: the
/// instant of the first track sample and the spacing of the samples.
pub const FIELD_TIMING_FLOATS: usize = 2;
/// Floats per track sample in the session file: the three axes (local X, Y,
/// Z as columns, solver world axes), then the nine scalars [`TRACK_SIZE`]
/// lists, then the noise scale along local X, Y and Z, then the wind factor,
/// linear drag, quadratic drag, harmonic damping, rest length, inflow and a
/// Texture's nabla.
pub const TRACK_FILE_FLOATS: usize = 28;
/// Floats per track sample on the device, `FIELD_TRACK` in the kernel: the
/// file's 28, then the origin's motion to the next sample (scene units).
pub const TRACK_FLOATS: usize = 31;
/// A Texture's nabla in a track sample.
pub const TRACK_NABLA: usize = 27;
/// Floats per track sample in the device's origin table,
/// `FIELD_TRACK_ORIGIN` in the kernel: the origin, in solver units.
pub const TRACK_ORIGIN_FLOATS: usize = 3;
/// How far from the solver's origin a field may sit, in solver units. The
/// device holds an origin in single precision, which at this distance still
/// places it within 1/32 of a unit.
pub const ORIGIN_REACH: f64 = 1.0e6;

/// A track sample after its nine axis floats holds, in order: strength,
/// falloff power, min distance, max distance, radial power, radial min,
/// radial max, size and noise amount (`FIELD_TRACK_*` in the kernel). The
/// host reads only the size.
pub const TRACK_SIZE: usize = 16;
/// The noise amount in a track sample.
pub const TRACK_NOISE: usize = 17;
/// The first of the three noise-scale floats in a track sample: the field
/// object's scale along its local X, Y and Z, which sizes Noise Amount's gusts
/// and scales an Object Coordinates texture's lookup.
pub const TRACK_NOISE_SCALE: usize = 18;
/// The wind factor in a track sample.
pub const TRACK_WIND_FACTOR: usize = 21;
/// A Vortex's inflow in a track sample.
pub const TRACK_INFLOW: usize = 26;

/// Words per script in the script table, `FIELD_SCRIPT_HEADER` in the kernel:
/// code offset, code length, constant offset, constant count, target bit.
pub const SCRIPT_HEADER_WORDS: usize = 5;
/// The target bit of a source that reaches every vertex, `FIELD_ALL_TARGETS`.
pub const ALL_TARGETS: u32 = 32;
/// The largest octave count NOISE and CURL take, `FIELD_MAX_OCTAVES`.
pub const MAX_OCTAVES: usize = 8;

/// A field object's type, `FieldType` in the kernel. The types Blender has
/// that are not here are refused by the add-on by name; the ones [`load`]
/// refuses say so there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Force = 0,
    Wind = 1,
    Vortex = 2,
    Turbulence = 3,
    Magnetic = 4,
    /// Evaluated at the Newton iterate by `external_field_implicit`.
    Drag = 5,
    Harmonic = 6,
    /// An image read at the point.
    Texture = 7,
    /// A fluid's velocity volume, evaluated at the Newton iterate.
    Fluid = 8,
}

/// Cells along each axis of a fluid brick, `FLUID_BRICK` in the kernel.
pub const FLUID_BRICK: usize = 8;
/// Cells per brick, `FLUID_BRICK_CELLS`.
pub const FLUID_BRICK_CELLS: usize = FLUID_BRICK * FLUID_BRICK * FLUID_BRICK;
/// Floats per cell (velocity in the domain's axes, then density),
/// `FLUID_CHANNELS`.
pub const FLUID_CHANNELS: usize = 4;
/// A table entry naming no brick, `FLUID_EMPTY`.
pub const FLUID_EMPTY: u32 = u32::MAX;
/// Floats per brick, [`FLUID_BRICK_CELLS`] cells of [`FLUID_CHANNELS`].
pub const FLUID_BRICK_FLOATS: usize = FLUID_BRICK_CELLS * FLUID_CHANNELS;
/// Cells along a frame's axis, at most: the kernel's cell and table index
/// arithmetic is exact in 32 bits up to here.
pub const FLUID_MAX_RESOLUTION: u32 = 1 << 15;
/// Floats per frame in a fluid's transform file: the domain corner (3, scene
/// units, solver axes), its axes (9, the columns local X, Y, Z), then the
/// cell's size along local X, Y and Z (3, scene units).
pub const FLUID_TRANSFORM_FLOATS: usize = 15;
/// Where a transform row's three cell sizes start.
pub const FLUID_TRANSFORM_CELL: usize = 12;

/// A Fluid field's velocity volume: one frame per instant, each with its own
/// resolution (a VDB sequence's active box changes from frame to frame), a
/// brick table and the bricks it names, kept ON DISK and
/// read when the driver needs the frame. The loader has
/// verified every frame, so a read that fails afterwards means the files
/// changed under the run.
///
/// A FRAME MAY BE EMPTY: a resolution of `[0, 0, 0]`, no table entry and no
/// brick. The fluid is absent there, so it neither pushes nor damps.
///
/// THE FRAMES SIT ON THE SCENE CLOCK, and the velocity is scene units per
/// clock second. The solve brings it to the vertex velocity's units, solver
/// units per physical second, in two places: World Scaling when the driver
/// copies a frame into its slot, and the step's Playback Speed in the kernel,
/// per call (`driver::force_field`).
#[derive(Debug, Clone)]
pub struct Fluid {
    /// Per frame, cells along each axis; `[0, 0, 0]` for an empty frame.
    pub resolution: Vec<[u32; 3]>,
    pub use_density: bool,
    /// The instant of frame 0 and the spacing of the rest, on the scene
    /// clock.
    pub start: f32,
    pub step: f32,
    /// Per frame, [`FLUID_TRANSFORM_FLOATS`] values.
    pub transform: Vec<[f64; FLUID_TRANSFORM_FLOATS]>,
    pub tables: Vec<PathBuf>,
    pub bricks: Vec<PathBuf>,
    /// The bricks each frame held when the loader measured it. They size the
    /// resident slots, so every later read of the frame must reproduce them.
    pub brick_counts: Vec<usize>,
    /// Per frame, the largest magnitude of a velocity component at load
    /// (scene units per second), which install holds against World Scaling.
    pub velocity_bound: Vec<f32>,
}

impl Fluid {
    pub fn frames(&self) -> usize {
        self.transform.len()
    }

    /// Whether frame `k` holds no fluid at all.
    pub fn is_empty_frame(&self, k: usize) -> bool {
        self.resolution[k] == [0, 0, 0]
    }

    /// Frame `k`'s bricks along each axis.
    pub fn brick_dims(&self, k: usize) -> [usize; 3] {
        self.resolution[k].map(|n| (n as usize).div_ceil(FLUID_BRICK))
    }

    /// Frame `k`'s table entries.
    pub fn table_len(&self, k: usize) -> usize {
        let b = self.brick_dims(k);
        b[0] * b[1] * b[2]
    }

    /// Frame `k`'s domain corner, scene units and solver axes.
    pub fn corner(&self, k: usize) -> [f64; 3] {
        let t = &self.transform[k];
        [t[0], t[1], t[2]]
    }

    /// Frame `k`'s local axis `c`, a column of its rotation.
    pub fn axis(&self, k: usize, c: usize) -> [f64; 3] {
        let t = &self.transform[k];
        [t[3 + 3 * c], t[4 + 3 * c], t[5 + 3 * c]]
    }

    /// Frame `k`'s cell along local X, Y and Z, scene units.
    pub fn cell(&self, k: usize) -> [f64; 3] {
        let t = &self.transform[k];
        [t[FLUID_TRANSFORM_CELL], t[FLUID_TRANSFORM_CELL + 1], t[FLUID_TRANSFORM_CELL + 2]]
    }

    /// The longest table of any frame, what a resident slot is sized for.
    pub fn table_capacity(&self) -> usize {
        (0..self.frames()).map(|k| self.table_len(k)).max().unwrap_or(0)
    }

    /// The most bricks any frame holds, what a resident slot is sized for.
    pub fn brick_capacity(&self) -> usize {
        self.brick_counts.iter().copied().max().unwrap_or(0)
    }

    /// The two frames bracketing `time` and the blend toward the second;
    /// the end frame holds outside the sequence, as a track's does.
    pub fn bracket(&self, time: f32) -> (usize, usize, f32) {
        let n = self.frames();
        if n <= 1 {
            return (0, 0, 0.0);
        }
        let u = (time - self.start) / self.step;
        if u >= (n - 1) as f32 {
            (n - 1, n - 1, 0.0)
        } else if u > 0.0 {
            let k = u as usize;
            (k, k + 1, (u - k as f32).clamp(0.0, 1.0))
        } else {
            (0, 0, 0.0)
        }
    }

    /// MEASURE frame `k`: read its table and bricks into `table` and
    /// `bricks`, verify them against the volume, and return how many bricks
    /// the frame holds and its largest velocity component. The loader's read,
    /// before anything is sized from the frame.
    pub fn measure_frame(
        &self,
        k: usize,
        table: &mut Vec<u32>,
        bricks: &mut Vec<f32>,
    ) -> Result<(usize, f32), String> {
        read_words_into(&self.tables[k], None, table)?;
        read_words_into(&self.bricks[k], None, bricks)?;
        verify_fluid_frame(self, k, table, bricks)?;
        let bound = bricks
            .chunks_exact(FLUID_CHANNELS)
            .flat_map(|cell| cell[..3].iter())
            .fold(0.0f32, |acc, v| acc.max(v.abs()));
        Ok((bricks.len() / FLUID_BRICK_FLOATS, bound))
    }

    /// Frame `k` as the run reads it after load, into `table` and `bricks`,
    /// whose storage is reused: the files must hold exactly the table length
    /// and the brick count the loader measured, which is what keeps every
    /// copy inside the resident slot the loader's counts sized. A file of any
    /// other length is refused before a byte of it is read.
    pub fn read_frame(
        &self,
        k: usize,
        table: &mut Vec<u32>,
        bricks: &mut Vec<f32>,
    ) -> Result<(), String> {
        let Some(&count) = self.brick_counts.get(k) else {
            return Err(format!(
                "frame {k} was never measured: the volume carries {} brick counts for {} frames",
                self.brick_counts.len(),
                self.frames()
            ));
        };
        let changed = |path: &PathBuf, now: usize, then: usize, unit: &str| {
            format!(
                "frame {k} ({}): {now} {unit}, {then} at load; the files changed under the run",
                path.display()
            )
        };
        let entries = word_count(&self.tables[k])?;
        if entries != self.table_len(k) {
            return Err(changed(&self.tables[k], entries, self.table_len(k), "table entries"));
        }
        let floats = word_count(&self.bricks[k])?;
        if floats % FLUID_BRICK_FLOATS != 0 || floats / FLUID_BRICK_FLOATS != count {
            return Err(changed(
                &self.bricks[k],
                floats / FLUID_BRICK_FLOATS,
                count,
                &format!("bricks ({floats} floats)"),
            ));
        }
        read_words_into(&self.tables[k], Some(self.table_len(k)), table)?;
        read_words_into(&self.bricks[k], Some(count * FLUID_BRICK_FLOATS), bricks)?;
        verify_fluid_frame(self, k, table, bricks)
    }
}

/// One frame's table and bricks against the volume: the table covers every
/// brick, every entry names a brick the frame holds or none, an empty frame
/// holds no brick, and every value is finite.
pub fn verify_fluid_frame(
    fluid: &Fluid,
    k: usize,
    table: &[u32],
    bricks: &[f32],
) -> Result<(), String> {
    let what = |p: &PathBuf| format!("frame {k} ({})", p.display());
    if table.len() != fluid.table_len(k) {
        return Err(format!(
            "{}: {} table entries for {} bricks of a {:?} domain",
            what(&fluid.tables[k]),
            table.len(),
            fluid.table_len(k),
            fluid.resolution[k]
        ));
    }
    let per = FLUID_BRICK_FLOATS;
    if bricks.len() % per != 0 {
        return Err(format!(
            "{}: {} floats is not a whole number of {per}-float bricks",
            what(&fluid.bricks[k]),
            bricks.len()
        ));
    }
    let count = bricks.len() / per;
    if fluid.is_empty_frame(k) && count != 0 {
        return Err(format!(
            "{}: an empty frame (a [0, 0, 0] domain) holds {count} bricks; it holds none",
            what(&fluid.bricks[k])
        ));
    }
    if let Some(bad) = table.iter().position(|&s| s != FLUID_EMPTY && s as usize >= count) {
        return Err(format!(
            "{}: entry {bad} names brick {} of the {count} the frame holds",
            what(&fluid.tables[k]),
            table[bad]
        ));
    }
    if let Some(bad) = bricks.iter().position(|v| !v.is_finite()) {
        let cell = bad / FLUID_CHANNELS;
        return Err(format!(
            "{}: brick {} cell {} channel {} is {}",
            what(&fluid.bricks[k]),
            cell / FLUID_BRICK_CELLS,
            cell % FLUID_BRICK_CELLS,
            bad % FLUID_CHANNELS,
            bricks[bad]
        ));
    }
    Ok(())
}

/// The volume's settings and transforms, every frame read once and verified,
/// so a malformed file stops the run at load rather than at the step that
/// first reaches it.
pub fn verify_fluid(fluid: &Fluid) -> Result<(), String> {
    let n = fluid.frames();
    if n == 0 {
        return Err("the fluid has no frame".into());
    }
    if fluid.tables.len() != n || fluid.bricks.len() != n || fluid.resolution.len() != n {
        return Err(format!(
            "{n} transforms, {} resolutions, {} tables and {} brick files; give one of \
             each per frame",
            fluid.resolution.len(),
            fluid.tables.len(),
            fluid.bricks.len()
        ));
    }
    for (k, res) in fluid.resolution.iter().enumerate() {
        if *res != [0, 0, 0] && res.iter().any(|&r| r == 0 || r > FLUID_MAX_RESOLUTION) {
            return Err(format!(
                "fluid frame {k}: a domain of {res:?} cells; each axis holds 1 to \
                 {FLUID_MAX_RESOLUTION} cells, and an empty frame is [0, 0, 0]"
            ));
        }
    }
    if !fluid.start.is_finite() {
        return Err(format!("the fluid starts at {}", fluid.start));
    }
    if n > 1 && !(fluid.step.is_finite() && fluid.step > 0.0) {
        return Err(format!("{n} fluid frames {} s apart", fluid.step));
    }
    // EVERY FRAME'S TRANSFORM IS VERIFIED, AN EMPTY ONE'S TOO: the driver
    // reads every frame's placement at install, whether or not it holds
    // any fluid, so a transform no frame could use is still a malformed file.
    for (k, t) in fluid.transform.iter().enumerate() {
        if let Some(bad) = t.iter().position(|v| !v.is_finite()) {
            return Err(format!("fluid frame {k}: transform value {bad} is {}", t[bad]));
        }
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        for c in 0..3 {
            for d in c..3 {
                let want = if c == d { 1.0 } else { 0.0 };
                let product = dot(fluid.axis(k, c), fluid.axis(k, d));
                if (product - want).abs() > 1.0e-4 {
                    return Err(format!(
                        "fluid frame {k}: the domain's axes are not orthonormal \
                         (axis {c} . axis {d} = {product})"
                    ));
                }
            }
        }
        // THE DEVICE READS THE CELL AS A FLOAT AND DIVIDES BY IT, so the
        // value verified is that float, and its reciprocal must be a normal
        // float too. A cell whose float image overflows has a zero
        // reciprocal, which reads every point of the world as inside the
        // domain; one whose reciprocal overflows reads every point as outside
        // it; and one whose reciprocal is subnormal reads every point as
        // inside it on a backend that flushes subnormals. All three are
        // silent, so all three are refused here.
        for (axis, &cell) in fluid.cell(k).iter().enumerate() {
            let single = cell as f32;
            if !(cell > 0.0 && single.is_normal() && (1.0f32 / single).is_normal()) {
                return Err(format!(
                    "fluid frame {k}: a cell of {cell} along local axis {axis}; a cell is \
                     positive, and it and its reciprocal are normal single-precision \
                     floats, which is what the device reads"
                ));
            }
        }
    }
    Ok(())
}

/// A Texture field's image and how it is read, `FieldTexture` in the kernel.
#[derive(Debug, Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    /// 0 RGB, 1 Gradient, 2 Curl.
    pub mode: u32,
    /// 0 Extend, 1 Repeat, 2 Clip, 3 Clip Cube.
    pub extension: u32,
    /// [`texture_flag`] bits.
    pub flags: u32,
    /// `width * height` LINEAR RGB texels, row 0 at the bottom.
    pub pixels: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldShape {
    Point = 0,
    Plane = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Falloff {
    Sphere = 0,
    Tube = 1,
    Cone = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZDirection {
    Both = 0,
    Positive = 1,
    Negative = 2,
}

/// The header's flag bits, `FieldFlag` in the kernel.
pub mod flag {
    pub const USE_MIN: u32 = 1;
    pub const USE_MAX: u32 = 2;
    pub const USE_RADIAL_MIN: u32 = 4;
    pub const USE_RADIAL_MAX: u32 = 8;
    pub const GRAVITATION: u32 = 16;
    pub const GLOBAL_COORDINATES: u32 = 32;
}

/// A Texture's flag bits, the texture flags of `FieldTexture` in the kernel.
pub mod texture_flag {
    /// Bilinear between pixel centers rather than the nearest pixel.
    pub const INTERPOLATE: u32 = 1;
    /// The field object's own space rather than world coordinates: the point
    /// in the field's local frame divided, axis by axis, by the object's
    /// scale along that axis (the track's noise scale), which is where
    /// Blender reads it, under the object's full inverse matrix (measured).
    pub const OBJECT_COORDINATES: u32 = 2;
    /// BLENDER'S AXES (Z up): a world-coordinate lookup reads the image at
    /// the point's Blender coordinates, and the push the image gives is a
    /// Blender-world vector, turned to solver axes. The add-on sets it, so a
    /// Blender texture pushes the way Blender's does.
    pub const Z_UP: u32 = 4;
}

/// One field object.
#[derive(Debug, Clone)]
pub struct FieldRecord {
    pub kind: FieldType,
    pub shape: FieldShape,
    pub falloff: Falloff,
    pub z_direction: ZDirection,
    /// [`flag`] bits.
    pub flags: u32,
    pub seed: u32,
    /// The instant of the first track sample, and the spacing of the rest.
    pub start: f32,
    pub step: f32,
    /// One origin per track sample, in SCENE units and solver world axes.
    pub origin: Vec<[f64; 3]>,
    /// [`TRACK_FILE_FLOATS`] floats per track sample.
    pub track: Vec<f32>,
    /// The target bit, below [`ALL_TARGETS`], or [`ALL_TARGETS`] for every
    /// vertex.
    pub target: u32,
    /// A Texture field's image; `None` for every other type.
    pub texture: Option<Texture>,
    /// A Fluid field's volume; `None` for every other type.
    pub fluid: Option<Fluid>,
}

impl FieldRecord {
    pub fn samples(&self) -> usize {
        self.origin.len()
    }

    /// Whether this field has a term `external_field_implicit` evaluates at
    /// the Newton iterate: Drag, Harmonic, and a Vortex's damping (always on a
    /// Plane, by its Inflow on a Point).
    pub fn is_implicit(&self) -> bool {
        match self.kind {
            FieldType::Drag | FieldType::Harmonic | FieldType::Fluid => true,
            FieldType::Vortex => {
                self.shape == FieldShape::Plane
                    || (0..self.samples())
                        .any(|k| self.track[TRACK_FILE_FLOATS * k + TRACK_INFLOW] != 0.0)
            }
            _ => false,
        }
    }

    /// Whether this field adds to the air velocity rather than the
    /// acceleration.
    pub fn is_air(&self) -> bool {
        self.kind == FieldType::Wind
    }
}

/// One compiled script.
#[derive(Debug, Clone)]
pub struct Script {
    pub code: Vec<u32>,
    pub constants: Vec<f32>,
    /// The source the frontend compiled, kept for the log.
    pub source: String,
    /// As [`FieldRecord::target`].
    pub target: u32,
}

/// Everything the field needs, as the loader read and verified it.
#[derive(Debug, Clone, Default)]
pub struct ForceFieldData {
    pub fields: Vec<FieldRecord>,
    pub scripts: Vec<Script>,
    /// One weight per dynamic vertex, or `None` for all ones.
    pub weight: Option<Vec<f32>>,
    /// One target mask per dynamic vertex, bit `b` set when the sources with
    /// target bit `b` reach it; `None` when every source reaches every vertex.
    pub target_mask: Option<Vec<u32>>,
}

impl ForceFieldData {
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty() && self.scripts.is_empty()
    }

    pub fn has_air(&self) -> bool {
        self.fields.iter().any(FieldRecord::is_air)
    }

    pub fn has_implicit(&self) -> bool {
        self.fields.iter().any(FieldRecord::is_implicit)
    }

    /// One line for the log.
    pub fn describe(&self) -> String {
        let mut parts: Vec<String> = self
            .fields
            .iter()
            .map(|f| {
                let volume = match &f.fluid {
                    Some(v) => {
                        let largest = v.resolution.iter().fold([0u32; 3], |acc, r| {
                            [acc[0].max(r[0]), acc[1].max(r[1]), acc[2].max(r[2])]
                        });
                        let empty = (0..v.frames()).filter(|&k| v.is_empty_frame(k)).count();
                        format!(
                            ", a volume of {} frames ({empty} empty) at most {largest:?} cells \
                             and {} bricks a frame",
                            v.frames(),
                            v.brick_capacity()
                        )
                    }
                    None => String::new(),
                };
                format!(
                    "{:?} field ({:?}, {:?} falloff) over {} track samples{volume}",
                    f.kind,
                    f.shape,
                    f.falloff,
                    f.samples()
                )
            })
            .collect();
        for script in &self.scripts {
            parts.push(format!(
                "script of {} instructions and {} constants{}",
                script.code.len(),
                script.constants.len(),
                if script.target == ALL_TARGETS {
                    String::new()
                } else {
                    format!(" (target {})", script.target)
                }
            ));
        }
        if self.weight.is_some() {
            parts.push("per-vertex weights".into());
        }
        if self.target_mask.is_some() {
            parts.push("per-vertex targets".into());
        }
        format!("force field: {}", parts.join(", "))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestField {
    #[serde(rename = "type")]
    kind: String,
    shape: String,
    falloff: String,
    z_direction: String,
    #[serde(default)]
    use_min: bool,
    #[serde(default)]
    use_max: bool,
    #[serde(default)]
    use_radial_min: bool,
    #[serde(default)]
    use_radial_max: bool,
    #[serde(default)]
    gravitation: bool,
    #[serde(default)]
    global_coordinates: bool,
    #[serde(default)]
    seed: u32,
    start: f32,
    step: f32,
    origin: String,
    track: String,
    #[serde(default = "all_targets")]
    targets: u32,
    texture: Option<ManifestTexture>,
    fluid: Option<ManifestFluid>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestFluid {
    /// One per frame.
    resolution: Vec<[u32; 3]>,
    use_density: bool,
    start: f32,
    step: f32,
    transform: String,
    tables: Vec<String>,
    bricks: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestTexture {
    image: String,
    width: u32,
    height: u32,
    mode: String,
    extension: String,
    interpolation: bool,
    object_coordinates: bool,
    /// Blender's axes, [`texture_flag::Z_UP`]; absent means solver axes.
    #[serde(default)]
    z_up: bool,
}

fn all_targets() -> u32 {
    ALL_TARGETS
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestScript {
    version: u32,
    code: String,
    constants: String,
    source: String,
    #[serde(default = "all_targets")]
    targets: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWeight {
    data: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    #[serde(default)]
    field: Vec<ManifestField>,
    #[serde(default)]
    script: Vec<ManifestScript>,
    weight: Option<ManifestWeight>,
    targets: Option<ManifestWeight>,
}

/// Read and verify the field of the session at `session`, for a scene of
/// `n_vert` dynamic vertices. `Ok(None)` when the session carries no field.
pub fn load(session: &str, n_vert: usize) -> Result<Option<ForceFieldData>, String> {
    let dir = Path::new(session).join(FIELD_DIR);
    load_dir(&dir, n_vert)
}

/// [`load`], from the field directory itself.
pub fn load_dir(dir: &Path, n_vert: usize) -> Result<Option<ForceFieldData>, String> {
    let manifest_path = dir.join(MANIFEST);
    if !manifest_path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest: Manifest = toml::from_str(&text)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;

    let mut out = ForceFieldData::default();
    for (index, entry) in manifest.field.iter().enumerate() {
        let what = format!("{} field {index}", manifest_path.display());
        let field = read_field(dir, entry).map_err(|e| format!("{what}: {e}"))?;
        verify_field(&field).map_err(|e| format!("{what}: {e}"))?;
        out.fields.push(field);
    }
    for (index, entry) in manifest.script.iter().enumerate() {
        let what = format!("{} script {index}", manifest_path.display());
        if entry.version != SCRIPT_VERSION {
            return Err(format!(
                "{what}: bytecode version {} but this solver runs version {SCRIPT_VERSION}; \
                 rebuild the scene with the frontend that ships with this solver",
                entry.version
            ));
        }
        let code = read_u32(&dir.join(&entry.code))?;
        let constants = read_f32(&dir.join(&entry.constants))?;
        let source = std::fs::read_to_string(dir.join(&entry.source)).unwrap_or_default();
        let script = Script { code, constants, source, target: entry.targets };
        verify_script(&script).map_err(|e| format!("{what}: {e}"))?;
        out.scripts.push(script);
    }
    if let Some(entry) = &manifest.weight {
        let path = dir.join(&entry.data);
        let weight = read_f32(&path)?;
        if weight.len() != n_vert {
            return Err(format!(
                "{}: {} weights for a scene of {n_vert} dynamic vertices",
                path.display(),
                weight.len()
            ));
        }
        if let Some(bad) = weight.iter().position(|w| !w.is_finite()) {
            return Err(format!("{}: weight {bad} is {}", path.display(), weight[bad]));
        }
        out.weight = Some(weight);
    }
    if let Some(entry) = &manifest.targets {
        let path = dir.join(&entry.data);
        let mask = read_u32(&path)?;
        if mask.len() != n_vert {
            return Err(format!(
                "{}: {} target masks for a scene of {n_vert} dynamic vertices",
                path.display(),
                mask.len()
            ));
        }
        out.target_mask = Some(mask);
    }
    // A TARGET BIT WITH NO MASK WOULD REACH NOTHING, silently: every vertex
    // would read a zero mask. So a source that names a bit requires the mask,
    // and a bit out of range is refused rather than read as "every vertex".
    let bits = out
        .fields
        .iter()
        .map(|f| f.target)
        .chain(out.scripts.iter().map(|s| s.target));
    for bit in bits {
        if bit > ALL_TARGETS {
            return Err(format!(
                "{}: target bit {bit} is past the {ALL_TARGETS} a mask holds",
                manifest_path.display()
            ));
        }
        if bit < ALL_TARGETS && out.target_mask.is_none() {
            return Err(format!(
                "{}: a source targets bit {bit} and the manifest carries no [targets] mask",
                manifest_path.display()
            ));
        }
    }
    if out.is_empty() {
        return Err(format!(
            "{}: the manifest names neither a field nor a script",
            manifest_path.display()
        ));
    }
    Ok(Some(out))
}

fn read_bytes(path: &PathBuf) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_f32(path: &PathBuf) -> Result<Vec<f32>, String> {
    let bytes = read_bytes(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: {} bytes is not a whole number of f32", path.display(), bytes.len()));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn read_f64(path: &PathBuf) -> Result<Vec<f64>, String> {
    let bytes = read_bytes(path)?;
    if bytes.len() % 8 != 0 {
        return Err(format!("{}: {} bytes is not a whole number of f64", path.display(), bytes.len()));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

fn read_u32(path: &PathBuf) -> Result<Vec<u32>, String> {
    let bytes = read_bytes(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: {} bytes is not a whole number of u32", path.display(), bytes.len()));
    }
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// How many four-byte little-endian words the file at `path` holds, from its
/// length alone.
fn word_count(path: &Path) -> Result<usize, String> {
    let bytes = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.len();
    if bytes % 4 != 0 {
        return Err(format!("{}: {bytes} bytes is not a whole number of 4-byte words", path.display()));
    }
    usize::try_from(bytes / 4).map_err(|_| format!("{}: {bytes} bytes is past this host", path.display()))
}

/// The four-byte little-endian words of the file at `path` (a `u32` or an
/// `f32` each), read into `out`, whose storage is reused: a caller that sized
/// it once reads every later frame without allocating. With `expect`, a file
/// holding any other number of words is refused before a byte is read.
fn read_words_into<T: bytemuck::Pod + Default>(
    path: &Path,
    expect: Option<usize>,
    out: &mut Vec<T>,
) -> Result<(), String> {
    const { assert!(std::mem::size_of::<T>() == 4) };
    use std::io::Read;
    let words = word_count(path)?;
    if let Some(want) = expect {
        if words != want {
            return Err(format!("{}: {words} words where {want} are expected", path.display()));
        }
    }
    out.clear();
    out.resize(words, T::default());
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    file.read_exact(bytemuck::cast_slice_mut::<T, u8>(out))
        .map_err(|e| format!("{}: {e} reading {words} words", path.display()))?;
    // The files are little-endian, which is this host's order on every
    // platform the solver ships for; the swap is spelled out so a big-endian
    // host reads them right as well.
    for word in bytemuck::cast_slice_mut::<T, u32>(out) {
        *word = u32::from_le(*word);
    }
    Ok(())
}

/// The record a manifest entry and its two files describe, its spellings
/// mapped and its flags gathered, not yet verified.
fn read_field(dir: &Path, entry: &ManifestField) -> Result<FieldRecord, String> {
    let kind = match entry.kind.as_str() {
        "force" => FieldType::Force,
        "wind" => FieldType::Wind,
        "vortex" => FieldType::Vortex,
        "turbulence" => FieldType::Turbulence,
        "magnetic" => FieldType::Magnetic,
        "drag" => FieldType::Drag,
        "harmonic" => FieldType::Harmonic,
        "texture" => FieldType::Texture,
        "fluid" => FieldType::Fluid,
        other => return Err(format!("field type {other:?} is not one this solver evaluates")),
    };
    let shape = match entry.shape.as_str() {
        "point" => FieldShape::Point,
        "plane" => FieldShape::Plane,
        other => return Err(format!("shape {other:?} is neither \"point\" nor \"plane\"")),
    };
    let falloff = match entry.falloff.as_str() {
        "sphere" => Falloff::Sphere,
        "tube" => Falloff::Tube,
        "cone" => Falloff::Cone,
        other => return Err(format!("falloff {other:?} is not sphere, tube or cone")),
    };
    let z_direction = match entry.z_direction.as_str() {
        "both" => ZDirection::Both,
        "positive" => ZDirection::Positive,
        "negative" => ZDirection::Negative,
        other => return Err(format!("z direction {other:?} is not both, positive or negative")),
    };
    let mut flags = 0u32;
    for (on, bit) in [
        (entry.use_min, flag::USE_MIN),
        (entry.use_max, flag::USE_MAX),
        (entry.use_radial_min, flag::USE_RADIAL_MIN),
        (entry.use_radial_max, flag::USE_RADIAL_MAX),
        (entry.gravitation, flag::GRAVITATION),
        (entry.global_coordinates, flag::GLOBAL_COORDINATES),
    ] {
        if on {
            flags |= bit;
        }
    }
    let origin_flat = read_f64(&dir.join(&entry.origin))?;
    if origin_flat.len() % 3 != 0 {
        return Err(format!("{} origin values is not a whole number of points", origin_flat.len()));
    }
    let origin: Vec<[f64; 3]> = origin_flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let track = read_f32(&dir.join(&entry.track))?;
    let texture = match &entry.texture {
        None => None,
        Some(t) => Some(Texture {
            width: t.width,
            height: t.height,
            mode: match t.mode.as_str() {
                "rgb" => 0,
                "gradient" => 1,
                "curl" => 2,
                other => return Err(format!("texture mode {other:?} is not rgb, gradient or curl")),
            },
            extension: match t.extension.as_str() {
                "extend" => 0,
                "repeat" => 1,
                "clip" => 2,
                "clip_cube" => 3,
                other => {
                    return Err(format!(
                        "texture extension {other:?} is not extend, repeat, clip or clip_cube"
                    ))
                }
            },
            flags: [
                (t.interpolation, texture_flag::INTERPOLATE),
                (t.object_coordinates, texture_flag::OBJECT_COORDINATES),
                (t.z_up, texture_flag::Z_UP),
            ]
            .into_iter()
            .filter(|&(on, _)| on)
            .fold(0, |bits, (_, bit)| bits | bit),
            pixels: read_f32(&dir.join(&t.image))?,
        }),
    };
    let fluid = match &entry.fluid {
        None => None,
        Some(f) => {
            let path = dir.join(&f.transform);
            let flat = read_f64(&path)?;
            if flat.len() % FLUID_TRANSFORM_FLOATS != 0 {
                return Err(format!(
                    "{}: {} values is not a whole number of {FLUID_TRANSFORM_FLOATS}-value \
                     frames (corner, axes, cell along each axis)",
                    path.display(),
                    flat.len()
                ));
            }
            let transform = flat
                .chunks_exact(FLUID_TRANSFORM_FLOATS)
                .map(|c| <[f64; FLUID_TRANSFORM_FLOATS]>::try_from(c).unwrap())
                .collect();
            let frames = f.resolution.len();
            let mut fluid = Fluid {
                resolution: f.resolution.clone(),
                use_density: f.use_density,
                start: f.start,
                step: f.step,
                transform,
                tables: f.tables.iter().map(|t| dir.join(t)).collect(),
                bricks: f.bricks.iter().map(|b| dir.join(b)).collect(),
                brick_counts: Vec::with_capacity(frames),
                velocity_bound: Vec::with_capacity(frames),
            };
            verify_fluid(&fluid)?;
            // Every frame is read once here, which is what verifies it before
            // the run and what measures the slots the driver sizes.
            let (mut table, mut bricks) = (Vec::new(), Vec::new());
            for k in 0..fluid.frames() {
                let (count, bound) = fluid.measure_frame(k, &mut table, &mut bricks)?;
                fluid.brick_counts.push(count);
                fluid.velocity_bound.push(bound);
            }
            Some(fluid)
        }
    };
    Ok(FieldRecord {
        fluid,
        kind,
        shape,
        falloff,
        z_direction,
        flags,
        seed: entry.seed,
        start: entry.start,
        step: entry.step,
        origin,
        track,
        target: entry.targets,
        texture,
    })
}

/// A record the kernel can evaluate without a bounds or a domain question.
pub fn verify_field(field: &FieldRecord) -> Result<(), String> {
    let samples = field.samples();
    if samples == 0 {
        return Err("the track has no sample".into());
    }
    match (&field.texture, field.kind == FieldType::Texture) {
        (None, true) => return Err("a texture field carries no image".into()),
        (Some(_), false) => return Err("only a texture field carries an image".into()),
        (Some(t), true) => {
            if t.width == 0 || t.height == 0 {
                return Err(format!("a {}x{} image", t.width, t.height));
            }
            let want = 3 * t.width as u64 * t.height as u64;
            if t.pixels.len() as u64 != want {
                return Err(format!(
                    "{} image floats for {}x{} RGB texels",
                    t.pixels.len(),
                    t.width,
                    t.height
                ));
            }
            if let Some(bad) = t.pixels.iter().position(|v| !v.is_finite()) {
                return Err(format!("image float {bad} is {}", t.pixels[bad]));
            }
        }
        (None, false) => {}
    }
    if field.fluid.is_some() != (field.kind == FieldType::Fluid) {
        return Err(if field.fluid.is_some() {
            "only a fluid field carries a volume".into()
        } else {
            "a fluid field carries no volume".into()
        });
    }
    if let Some(fluid) = &field.fluid {
        // The driver sizes the resident slots from the measured counts, so a
        // volume that skipped the loader's measurement cannot be installed.
        if fluid.brick_counts.len() != fluid.frames() || fluid.velocity_bound.len() != fluid.frames()
        {
            return Err(format!(
                "the fluid's {} frames were not all measured ({} brick counts, {} velocity \
                 bounds)",
                fluid.frames(),
                fluid.brick_counts.len(),
                fluid.velocity_bound.len()
            ));
        }
        if field.flags & flag::GRAVITATION != 0 {
            return Err("a fluid field with Gravitation; Blender's Fluid Flow applies none \
                        (measured), so this solver would push where Blender does not"
                .into());
        }
    }
    if field.track.len() != TRACK_FILE_FLOATS * samples {
        return Err(format!(
            "{} track floats for {samples} samples of {TRACK_FILE_FLOATS}",
            field.track.len()
        ));
    }
    if !field.start.is_finite() {
        return Err(format!("the track starts at {}", field.start));
    }
    if samples > 1 && !(field.step.is_finite() && field.step > 0.0) {
        return Err(format!("{samples} track samples {} s apart", field.step));
    }
    for (k, o) in field.origin.iter().enumerate() {
        if o.iter().any(|v| !v.is_finite()) {
            return Err(format!("origin {k} is {o:?}"));
        }
    }
    if let Some(bad) = field.track.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "track sample {} value {} is {}",
            bad / TRACK_FILE_FLOATS,
            bad % TRACK_FILE_FLOATS,
            field.track[bad]
        ));
    }
    for k in 0..samples {
        let row = &field.track[TRACK_FILE_FLOATS * k..TRACK_FILE_FLOATS * (k + 1)];
        // THE AXES ARE A ROTATION. The kernel takes the transpose as the
        // inverse, so a scaled or sheared frame would read every distance
        // wrong rather than fail. The object's scale reaches the field only
        // as the noise scale, which is measured to be all it does in
        // Blender: it sizes Noise Amount's gusts, and an Object Coordinates
        // texture reads its image at the local point divided by it.
        let axis = |c: usize| [row[3 * c], row[3 * c + 1], row[3 * c + 2]];
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        for c in 0..3 {
            for d in c..3 {
                let want = if c == d { 1.0 } else { 0.0 };
                if (dot(axis(c), axis(d)) - want).abs() > 1.0e-4 {
                    return Err(format!(
                        "track sample {k}: the axes are not orthonormal (axis {c} . axis {d} = {})",
                        dot(axis(c), axis(d))
                    ));
                }
            }
        }
        // The kernel divides by each noise scale, so its reciprocal must be a
        // float too: a positive NORMAL value, never a subnormal one.
        for c in 0..3 {
            let s = row[TRACK_NOISE_SCALE + c];
            if !(s > 0.0 && s.is_normal()) {
                return Err(format!(
                    "track sample {k}: noise scale {c} is {s}, not a positive normal float"
                ));
            }
        }
        let wind_factor = row[TRACK_WIND_FACTOR];
        if field.kind == FieldType::Wind && wind_factor != 0.0 {
            return Err(format!(
                "track sample {k}: a wind with wind factor {wind_factor}; a wind acts \
                 through the drag, which weighs a surface by how it faces the flow already"
            ));
        }
        if let Some(t) = &field.texture {
            let nabla = row[TRACK_NABLA];
            if t.mode != 0 && !(nabla > 0.0) {
                return Err(format!(
                    "track sample {k}: a gradient or curl texture with nabla {nabla}; its \
                     differences are taken nabla apart, so it must be positive"
                ));
            }
        }
        if field.kind == FieldType::Fluid {
            // The explicit kernel skips a Fluid field, so these two would be
            // silently ignored there.
            for (what, at) in [("noise amount", TRACK_NOISE), ("wind factor", TRACK_WIND_FACTOR)] {
                if row[at] != 0.0 {
                    return Err(format!(
                        "track sample {k}: a fluid field with {what} {}; it acts through the \
                         fluid's velocity alone",
                        row[at]
                    ));
                }
            }
        }
        let size = row[TRACK_SIZE];
        if field.kind == FieldType::Turbulence && !(size > 0.0) {
            return Err(format!(
                "track sample {k}: a turbulence of size {size}; its swirls are size across, \
                 so the size must be positive"
            ));
        }
    }
    Ok(())
}

/// The opcodes, `FieldOp` in the kernel.
pub mod op {
    pub const CONST: u32 = 1;
    pub const LOAD: u32 = 2;
    pub const STORE: u32 = 3;
    pub const ADD: u32 = 4;
    pub const SUB: u32 = 5;
    pub const MUL: u32 = 6;
    pub const DIV: u32 = 7;
    pub const NEG: u32 = 8;
    pub const MOD: u32 = 9;
    pub const POW: u32 = 10;
    pub const LT: u32 = 11;
    pub const LE: u32 = 12;
    pub const GT: u32 = 13;
    pub const GE: u32 = 14;
    pub const EQ: u32 = 15;
    pub const NE: u32 = 16;
    pub const NOT: u32 = 17;
    pub const AND: u32 = 18;
    pub const OR: u32 = 19;
    pub const JUMP: u32 = 20;
    pub const JUMP_IF_FALSE: u32 = 21;
    pub const RETURN: u32 = 22;
    /// The unary functions, `SQRT` through `COSH`.
    pub const UNARY_FIRST: u32 = 32;
    pub const UNARY_LAST: u32 = 46;
    /// The binary functions, `ATAN2` through `HYPOT`.
    pub const BINARY_FUNC_FIRST: u32 = 48;
    pub const BINARY_FUNC_LAST: u32 = 51;
    /// Pop x, y, z, w, seed; the operand is the octave count. w is how far
    /// the pattern has evolved.
    pub const NOISE: u32 = 52;
    pub const CURL: u32 = 53;
}

/// How an instruction moves the stack: `(pops, pushes)`, or `None` for a word
/// outside the table.
fn stack_effect(opcode: u32) -> Option<(usize, usize)> {
    use op::*;
    Some(match opcode {
        CONST | LOAD => (0, 1),
        NOISE => (5, 1),
        CURL => (5, 3),
        STORE | JUMP_IF_FALSE => (1, 0),
        JUMP => (0, 0),
        RETURN => (3, 0),
        NEG | NOT => (1, 1),
        ADD | SUB | MUL | DIV | MOD | POW | LT | LE | GT | GE | EQ | NE | AND | OR => (2, 1),
        c if (UNARY_FIRST..=UNARY_LAST).contains(&c) => (1, 1),
        c if (BINARY_FUNC_FIRST..=BINARY_FUNC_LAST).contains(&c) => (2, 1),
        _ => return None,
    })
}

/// A program the kernel can run without a bounds question: every word in the
/// table, every operand in range, every jump forward, one stack depth per
/// instruction on every path into it, never past the capacity, and every path
/// ending in a RETURN.
pub fn verify_script(script: &Script) -> Result<(), String> {
    let code = &script.code;
    let length = code.len();
    if length == 0 {
        return Err("the program is empty".into());
    }
    if length > (1 << 24) {
        return Err(format!("{length} instructions exceed the 24-bit jump operand"));
    }
    if let Some(bad) = script.constants.iter().position(|v| !v.is_finite()) {
        return Err(format!("constant {bad} is {}", script.constants[bad]));
    }
    // FORWARD-ONLY JUMPS MAKE ONE PASS IN ORDER A COMPLETE DATAFLOW: every
    // predecessor of an instruction precedes it, so its depth is settled
    // before it is reached.
    let mut depth: Vec<Option<usize>> = vec![None; length + 1];
    depth[0] = Some(0);
    for pc in 0..length {
        let Some(here) = depth[pc] else {
            // Unreachable code is harmless to the kernel, which never runs it.
            continue;
        };
        let word = code[pc];
        let opcode = word & 0xff;
        let arg = (word >> 8) as usize;
        let Some((pops, pushes)) = stack_effect(opcode) else {
            return Err(format!("instruction {pc}: opcode {opcode} is not in the table"));
        };
        if here < pops {
            return Err(format!("instruction {pc}: pops {pops} from a stack of {here}"));
        }
        let after = here - pops + pushes;
        if after > STACK_CAPACITY {
            return Err(format!(
                "instruction {pc}: the stack reaches {after}, past the interpreter's {STACK_CAPACITY}"
            ));
        }
        match opcode {
            op::CONST if arg >= script.constants.len() => {
                return Err(format!(
                    "instruction {pc}: constant {arg} of {}",
                    script.constants.len()
                ))
            }
            op::LOAD | op::STORE if arg as u32 >= VAR_CAPACITY => {
                return Err(format!("instruction {pc}: variable {arg} of {VAR_CAPACITY}"))
            }
            op::NOISE | op::CURL if !(1..=MAX_OCTAVES).contains(&arg) => {
                return Err(format!(
                    "instruction {pc}: {arg} octaves, and noise takes 1 to {MAX_OCTAVES}"
                ))
            }
            _ => {}
        }
        let mut flow = |target: usize, d: usize| -> Result<(), String> {
            match depth[target] {
                None => {
                    depth[target] = Some(d);
                    Ok(())
                }
                Some(seen) if seen == d => Ok(()),
                Some(seen) => Err(format!(
                    "instruction {target} is reached with stack depths {seen} and {d}"
                )),
            }
        };
        match opcode {
            op::RETURN => {}
            op::JUMP | op::JUMP_IF_FALSE => {
                if arg <= pc || arg > length {
                    return Err(format!(
                        "instruction {pc}: a jump to {arg}, and only forward jumps inside the \
                         program are admitted"
                    ));
                }
                flow(arg, after)?;
                if opcode == op::JUMP_IF_FALSE {
                    flow(pc + 1, after)?;
                }
            }
            _ => flow(pc + 1, after)?,
        }
    }
    if depth[length].is_some() {
        return Err("a path runs past the end of the program without returning".into());
    }
    Ok(())
}

/// Session files for tests, written as the frontend's `write_session` writes
/// them: a manifest and the binary files it names, little-endian.
#[cfg(test)]
pub(crate) mod test_session {
    use super::*;

    /// The solver's own axes, as columns.
    pub(crate) const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    /// A quarter turn about Z: local X along world Y, local Y along world -X.
    pub(crate) const QUARTER_TURN_Z: [[f64; 3]; 3] =
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

    /// One frame of a test volume.
    #[derive(Clone)]
    pub(crate) struct Frame {
        pub resolution: [u32; 3],
        pub table: Vec<u32>,
        pub bricks: Vec<f32>,
        pub transform: [f64; FLUID_TRANSFORM_FLOATS],
    }

    /// The field around a test volume: its Fluid Flow settings.
    pub(crate) struct FluidSettings {
        pub start: f32,
        pub step: f32,
        pub use_density: bool,
        pub strength: f32,
        pub flow: f32,
    }

    impl Default for FluidSettings {
        fn default() -> Self {
            FluidSettings { start: 0.0, step: 0.1, use_density: false, strength: 1.0, flow: 1.0 }
        }
    }

    /// A transform row: the corner, the axes given as columns, the cell
    /// along each local axis.
    pub(crate) fn transform(
        corner: [f64; 3],
        axes: [[f64; 3]; 3],
        cell: [f64; 3],
    ) -> [f64; FLUID_TRANSFORM_FLOATS] {
        let mut t = [0.0; FLUID_TRANSFORM_FLOATS];
        t[..3].copy_from_slice(&corner);
        for (c, axis) in axes.iter().enumerate() {
            t[3 + 3 * c..6 + 3 * c].copy_from_slice(axis);
        }
        t[FLUID_TRANSFORM_CELL..].copy_from_slice(&cell);
        t
    }

    /// One brick whose every cell holds `cell` (velocity, then density).
    pub(crate) fn brick(cell: [f32; 4]) -> Vec<f32> {
        cell.repeat(FLUID_BRICK_CELLS)
    }

    /// A track sample whose frame is the solver's, with no falloff, the
    /// given Strength and Inflow (a Fluid Flow's Flow), and the rest at their
    /// defaults.
    pub(crate) fn track_row(strength: f32, inflow: f32) -> Vec<f32> {
        let mut row = vec![0.0f32; TRACK_FILE_FLOATS];
        row[0] = 1.0;
        row[4] = 1.0;
        row[8] = 1.0;
        row[9] = strength;
        row[TRACK_SIZE] = 1.0;
        for c in 0..3 {
            row[TRACK_NOISE_SCALE + c] = 1.0;
        }
        row[TRACK_INFLOW] = inflow;
        row
    }

    /// Write `values` to `path` as little-endian words.
    pub(crate) fn write_words<T: bytemuck::Pod>(path: &Path, values: &[T]) {
        assert!(cfg!(target_endian = "little"), "the test files are written in host order");
        std::fs::write(path, bytemuck::cast_slice::<T, u8>(values)).unwrap();
    }

    /// The `[[field]]` lines every test field starts with, and its origin
    /// and track files.
    fn field_header(dir: &Path, kind: &str, track: &[f32]) -> String {
        write_words(&dir.join("field-0-origin.bin"), &[0.0f64; 3]);
        write_words(&dir.join("field-0-track.bin"), track);
        format!(
            "[[field]]\ntype = \"{kind}\"\nshape = \"point\"\nfalloff = \"sphere\"\n\
             z_direction = \"both\"\nstart = 0.0\nstep = 0.04\n\
             origin = \"field-0-origin.bin\"\ntrack = \"field-0-track.bin\"\n"
        )
    }

    /// A field directory holding one Fluid Flow over `frames`.
    pub(crate) fn write_fluid(dir: &Path, frames: &[Frame], settings: &FluidSettings) {
        let mut manifest = field_header(dir, "fluid", &track_row(settings.strength, settings.flow));
        std::fs::create_dir_all(dir.join("field-0-fluid")).unwrap();
        let transform: Vec<f64> = frames.iter().flat_map(|f| f.transform).collect();
        write_words(&dir.join("field-0-fluid/transform.bin"), &transform);
        let (mut tables, mut bricks, mut resolution) = (Vec::new(), Vec::new(), Vec::new());
        for (k, frame) in frames.iter().enumerate() {
            let table = format!("field-0-fluid/table-{k:05}.bin");
            let data = format!("field-0-fluid/bricks-{k:05}.bin");
            write_words(&dir.join(&table), &frame.table);
            write_words(&dir.join(&data), &frame.bricks);
            tables.push(format!("\"{table}\""));
            bricks.push(format!("\"{data}\""));
            let [x, y, z] = frame.resolution;
            resolution.push(format!("[{x}, {y}, {z}]"));
        }
        manifest += &format!(
            "[field.fluid]\nresolution = [{}]\nuse_density = {}\nstart = {:?}\nstep = {:?}\n\
             transform = \"field-0-fluid/transform.bin\"\ntables = [{}]\nbricks = [{}]\n",
            resolution.join(", "),
            settings.use_density,
            settings.start,
            settings.step,
            tables.join(", "),
            bricks.join(", ")
        );
        std::fs::write(dir.join(MANIFEST), manifest).unwrap();
    }

    /// A field directory holding one RGB Texture over a 2 x 2 image, with
    /// `extra` added to its `[field.texture]` table.
    pub(crate) fn write_texture(dir: &Path, extra: &str) {
        let mut manifest = field_header(dir, "texture", &track_row(1.0, 0.0));
        write_words(&dir.join("field-0-image.bin"), &[0.25f32; 12]);
        manifest += &format!(
            "[field.texture]\nimage = \"field-0-image.bin\"\nwidth = 2\nheight = 2\n\
             mode = \"rgb\"\nextension = \"extend\"\ninterpolation = true\n\
             object_coordinates = false\n{extra}"
        );
        std::fs::write(dir.join(MANIFEST), manifest).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(opcode: u32, arg: u32) -> u32 {
        opcode | (arg << 8)
    }

    fn script(code: Vec<u32>, constants: Vec<f32>) -> Script {
        Script { code, constants, source: String::new(), target: ALL_TARGETS }
    }

    #[test]
    fn a_straight_program_that_returns_three_values_verifies() {
        let s = script(
            vec![w(op::LOAD, 0), w(op::CONST, 0), w(op::LOAD, 2), w(op::RETURN, 0)],
            vec![1.0],
        );
        verify_script(&s).unwrap();
    }

    #[test]
    fn a_backward_jump_is_refused_so_every_program_terminates() {
        let s = script(vec![w(op::JUMP, 0)], vec![]);
        assert!(verify_script(&s).unwrap_err().contains("forward"));
    }

    #[test]
    fn falling_off_the_end_is_refused() {
        let s = script(vec![w(op::LOAD, 0), w(op::STORE, 4)], vec![]);
        assert!(verify_script(&s).unwrap_err().contains("without returning"));
    }

    #[test]
    fn a_branch_that_leaves_the_stack_unbalanced_is_refused() {
        // if x: push 1 ; then both arms meet at the return with different depths.
        let s = script(
            vec![
                w(op::LOAD, 0),
                w(op::JUMP_IF_FALSE, 3),
                w(op::CONST, 0),
                w(op::CONST, 0),
                w(op::CONST, 0),
                w(op::CONST, 0),
                w(op::RETURN, 0),
            ],
            vec![0.0],
        );
        assert!(verify_script(&s).unwrap_err().contains("stack depths"));
    }

    #[test]
    fn an_operand_out_of_range_is_refused() {
        let s = script(vec![w(op::CONST, 5), w(op::RETURN, 0)], vec![1.0]);
        assert!(verify_script(&s).unwrap_err().contains("constant 5"));
        let s = script(vec![w(op::LOAD, 64), w(op::RETURN, 0)], vec![]);
        assert!(verify_script(&s).unwrap_err().contains("variable 64"));
    }

    #[test]
    fn noise_pops_five_and_curl_pushes_three_with_bounded_octaves() {
        let s = script(
            vec![
                w(op::LOAD, 0),
                w(op::LOAD, 1),
                w(op::LOAD, 2),
                w(op::LOAD, 3),
                w(op::CONST, 0),
                w(op::CURL, 3),
                w(op::RETURN, 0),
            ],
            vec![7.0],
        );
        verify_script(&s).unwrap();
        let mut bad = s.clone();
        bad.code[5] = w(op::CURL, 9);
        assert!(verify_script(&bad).unwrap_err().contains("octaves"));
        let mut short = s.clone();
        short.code[5] = w(op::NOISE, 1);
        assert!(verify_script(&short).unwrap_err().contains("pops 3"));
        let mut four = s.clone();
        four.code.remove(3);
        assert!(verify_script(&four).unwrap_err().contains("pops 5"));
    }

    #[test]
    fn a_stack_past_capacity_is_refused() {
        let mut code: Vec<u32> = (0..33).map(|_| w(op::LOAD, 0)).collect();
        code.push(w(op::RETURN, 0));
        assert!(verify_script(&script(code, vec![])).unwrap_err().contains("past the interpreter"));
    }

    fn record(kind: FieldType, samples: usize) -> FieldRecord {
        let mut track = Vec::new();
        for _ in 0..samples {
            let mut row = vec![0.0f32; TRACK_FILE_FLOATS];
            row[0] = 1.0;
            row[4] = 1.0;
            row[8] = 1.0;
            row[9] = 1.0;
            row[TRACK_SIZE] = 1.0;
            for c in 0..3 {
                row[TRACK_NOISE_SCALE + c] = 1.0;
            }
            track.extend_from_slice(&row);
        }
        FieldRecord {
            texture: None,
            fluid: None,
            kind,
            shape: FieldShape::Point,
            falloff: Falloff::Sphere,
            z_direction: ZDirection::Both,
            flags: 0,
            seed: 0,
            start: 0.0,
            step: 0.04,
            origin: vec![[0.0; 3]; samples],
            track,
            target: ALL_TARGETS,
        }
    }

    #[test]
    fn a_record_with_a_rotation_and_positive_scales_verifies() {
        verify_field(&record(FieldType::Force, 3)).unwrap();
    }

    #[test]
    fn a_scaled_or_sheared_frame_is_refused() {
        let mut f = record(FieldType::Force, 1);
        f.track[0] = 2.0;
        assert!(verify_field(&f).unwrap_err().contains("orthonormal"));
        let mut f = record(FieldType::Force, 1);
        f.track[3] = 0.5;
        assert!(verify_field(&f).unwrap_err().contains("orthonormal"));
    }

    #[test]
    fn a_track_that_does_not_match_its_origins_is_refused() {
        let mut f = record(FieldType::Wind, 2);
        f.track.pop();
        assert!(verify_field(&f).unwrap_err().contains("track floats"));
        let mut f = record(FieldType::Wind, 2);
        f.step = 0.0;
        assert!(verify_field(&f).unwrap_err().contains("apart"));
        let mut f = record(FieldType::Wind, 2);
        f.origin[1][2] = f64::NAN;
        assert!(verify_field(&f).unwrap_err().contains("origin 1"));
    }

    #[test]
    fn a_texture_needs_its_whole_image_and_a_positive_nabla_for_differences() {
        let image = |w: u32, h: u32, mode: u32| Texture {
            width: w,
            height: h,
            mode,
            extension: 0,
            flags: 1,
            pixels: vec![0.5; (3 * w * h) as usize],
        };
        let mut f = record(FieldType::Texture, 1);
        assert!(verify_field(&f).unwrap_err().contains("no image"));
        f.texture = Some(image(4, 2, 0));
        verify_field(&f).unwrap();
        f.texture.as_mut().unwrap().pixels.pop();
        assert!(verify_field(&f).unwrap_err().contains("RGB texels"));
        f.texture = Some(image(4, 2, 1));
        f.track[TRACK_NABLA] = 0.0;
        assert!(verify_field(&f).unwrap_err().contains("nabla"));
        let mut g = record(FieldType::Force, 1);
        g.texture = Some(image(1, 1, 0));
        assert!(verify_field(&g).unwrap_err().contains("only a texture field"));
    }

    #[test]
    fn a_wind_factor_on_a_wind_is_refused_and_on_a_force_admitted() {
        let mut f = record(FieldType::Wind, 1);
        f.track[TRACK_WIND_FACTOR] = 1.0;
        assert!(verify_field(&f).unwrap_err().contains("wind factor"));
        let mut f = record(FieldType::Force, 1);
        f.track[TRACK_WIND_FACTOR] = 0.5;
        verify_field(&f).unwrap();
    }

    #[test]
    fn a_turbulence_of_size_zero_and_a_zero_noise_scale_are_refused() {
        let mut f = record(FieldType::Turbulence, 1);
        f.track[TRACK_SIZE] = 0.0;
        assert!(verify_field(&f).unwrap_err().contains("size"));
        let mut f = record(FieldType::Force, 1);
        f.track[TRACK_NOISE_SCALE + 1] = 0.0;
        assert!(verify_field(&f).unwrap_err().contains("noise scale"));
        // A subnormal scale is positive, and its reciprocal is infinite.
        let mut f = record(FieldType::Force, 1);
        f.track[TRACK_NOISE_SCALE + 2] = f32::MIN_POSITIVE / 4.0;
        assert!(verify_field(&f).unwrap_err().contains("noise scale 2"));
    }

    fn fluid(frames: usize) -> Fluid {
        let t = test_session::transform([0.0; 3], test_session::IDENTITY, [0.1, 0.2, 0.3]);
        Fluid {
            resolution: vec![[9, 3, 2]; frames],
            use_density: false,
            start: 0.5,
            step: 0.1,
            transform: vec![t; frames],
            tables: (0..frames).map(|k| PathBuf::from(format!("table-{k}"))).collect(),
            bricks: (0..frames).map(|k| PathBuf::from(format!("bricks-{k}"))).collect(),
            brick_counts: vec![1; frames],
            velocity_bound: vec![0.0; frames],
        }
    }

    #[test]
    fn a_fluid_bracket_blends_inside_and_holds_its_ends() {
        let f = fluid(3);
        assert_eq!(f.bracket(0.0), (0, 0, 0.0));
        let (k0, k1, b) = f.bracket(0.625);
        assert_eq!((k0, k1), (1, 2));
        assert!((b - 0.25).abs() < 1e-5);
        assert_eq!(f.bracket(9.0), (2, 2, 0.0));
        assert_eq!(fluid(1).bracket(3.0), (0, 0, 0.0));
    }

    #[test]
    fn a_fluid_frame_is_held_to_its_table_and_its_bricks() {
        let f = fluid(1);
        // 9 x 3 x 2 cells are 2 x 1 x 1 bricks.
        assert_eq!(f.table_len(0), 2);
        let brick = vec![0.0f32; FLUID_BRICK_CELLS * FLUID_CHANNELS];
        verify_fluid_frame(&f, 0, &[FLUID_EMPTY, 0], &brick).unwrap();
        let short = verify_fluid_frame(&f, 0, &[0], &brick).unwrap_err();
        assert!(short.contains("1 table entries for 2 bricks"), "{short}");
        let past = verify_fluid_frame(&f, 0, &[1, 0], &brick).unwrap_err();
        assert!(past.contains("names brick 1 of the 1"), "{past}");
        let ragged = verify_fluid_frame(&f, 0, &[0, 0], &brick[1..]).unwrap_err();
        assert!(ragged.contains("not a whole number"), "{ragged}");
        let mut nan = brick.clone();
        nan[4 * 70 + 3] = f32::NAN;
        let bad = verify_fluid_frame(&f, 0, &[0, FLUID_EMPTY], &nan).unwrap_err();
        assert!(bad.contains("brick 0 cell 70 channel 3"), "{bad}");
    }

    #[test]
    fn a_fluid_needs_a_rotation_a_cell_and_a_spacing() {
        verify_fluid(&fluid(2)).unwrap();
        let mut sheared = fluid(1);
        sheared.transform[0][4] = 0.5;
        assert!(verify_fluid(&sheared).unwrap_err().contains("not orthonormal"));
        let mut flat = fluid(1);
        flat.transform[0][FLUID_TRANSFORM_CELL] = 0.0;
        let message = verify_fluid(&flat).unwrap_err();
        assert!(message.contains("a cell of 0 along local axis 0"), "{message}");
        let mut still = fluid(2);
        still.step = 0.0;
        assert!(verify_fluid(&still).unwrap_err().contains("apart"));
        let mut partial = fluid(1);
        partial.resolution = vec![[0, 1, 1]];
        let message = verify_fluid(&partial).unwrap_err();
        assert!(message.contains("an empty frame is [0, 0, 0]"), "{message}");
        let mut wide = fluid(1);
        wide.resolution = vec![[FLUID_MAX_RESOLUTION + 1, 1, 1]];
        assert!(verify_fluid(&wide).unwrap_err().contains("a domain of"));
    }

    /// EACH AXIS CARRIES ITS OWN CELL, and each is held to what the device
    /// reads: a float, divided into one. The f64 value 1e39 is finite and
    /// positive, and its float image is infinite, which reads every point of
    /// the world as inside the domain; 3e38 is a float whose reciprocal is
    /// subnormal; 1e-39 is itself subnormal. All three were admitted by a
    /// test in double precision.
    #[test]
    fn a_fluid_cell_is_held_to_the_float_the_device_reads_axis_by_axis() {
        let accept = |cell: [f64; 3]| {
            let mut f = fluid(1);
            f.transform[0][FLUID_TRANSFORM_CELL..].copy_from_slice(&cell);
            verify_fluid(&f)
        };
        accept([0.1, 0.25, 4.0]).unwrap();
        accept([1.0e-30, 1.0e30, 1.0]).unwrap();
        for (axis, bad) in [
            (0, 1.0e39),
            (1, 3.0e38),
            (2, 1.0e-39),
            (1, -0.2),
            (2, 0.0),
        ] {
            let mut cell = [0.5; 3];
            cell[axis] = bad;
            let message = accept(cell).unwrap_err();
            assert!(
                message.contains(&format!("a cell of {bad} along local axis {axis}")),
                "{message}"
            );
        }
        let mut nan = fluid(1);
        nan.transform[0][FLUID_TRANSFORM_CELL + 1] = f64::NAN;
        assert!(verify_fluid(&nan).unwrap_err().contains("transform value 13"));
    }

    /// AN EMPTY FRAME IS `[0, 0, 0]` WITH NO TABLE AND NO BRICK: the fluid is
    /// absent there. A frame with some axes empty is refused rather than
    /// read as either.
    #[test]
    fn an_empty_frame_is_a_zero_domain_with_no_table_and_no_brick() {
        let mut f = fluid(3);
        f.resolution[1] = [0, 0, 0];
        verify_fluid(&f).unwrap();
        assert!(f.is_empty_frame(1) && !f.is_empty_frame(0));
        assert_eq!(f.table_len(1), 0);
        verify_fluid_frame(&f, 1, &[], &[]).unwrap();
        let brick = vec![0.0f32; FLUID_BRICK_FLOATS];
        let message = verify_fluid_frame(&f, 1, &[], &brick).unwrap_err();
        assert!(message.contains("an empty frame (a [0, 0, 0] domain) holds 1 bricks"), "{message}");
        let message = verify_fluid_frame(&f, 1, &[0], &brick).unwrap_err();
        assert!(message.contains("1 table entries for 0 bricks"), "{message}");
    }

    #[test]
    fn a_fluid_field_with_gravitation_or_unmeasured_frames_is_refused() {
        let mut f = record(FieldType::Fluid, 1);
        f.fluid = Some(fluid(2));
        verify_field(&f).unwrap();
        f.flags |= flag::GRAVITATION;
        assert!(verify_field(&f).unwrap_err().contains("Gravitation"));
        f.flags = 0;
        f.fluid.as_mut().unwrap().brick_counts.pop();
        let message = verify_field(&f).unwrap_err();
        assert!(message.contains("were not all measured (1 brick counts"), "{message}");
        f.fluid = Some(fluid(2));
        f.fluid.as_mut().unwrap().velocity_bound.clear();
        assert!(verify_field(&f).unwrap_err().contains("0 velocity bounds"));
        // Gravitation stays legal on every other type.
        let mut g = record(FieldType::Force, 1);
        g.flags |= flag::GRAVITATION;
        verify_field(&g).unwrap();
    }

    /// THE LOADER ROUND TRIP: a manifest and its files, as the frontend
    /// writes them, read back frame for frame. Three frames of different
    /// resolutions, one EMPTY, each with its own per-axis cell, rotation and
    /// corner, and tables that name their bricks out of order and leave some
    /// empty, so a reader that assumed an identity table, one cell or one
    /// resolution reads a different volume.
    #[test]
    fn a_multi_frame_fluid_reads_back_frame_for_frame() {
        use test_session::*;
        let dir = tempfile::tempdir().unwrap();
        let frames = vec![
            Frame {
                // 9 x 3 x 2 cells are 2 x 1 x 1 bricks, named in reverse.
                resolution: [9, 3, 2],
                table: vec![1, 0],
                bricks: [brick([1.0, 2.0, 3.0, 0.5]), brick([-4.0, 5.0, 6.0, 0.25])].concat(),
                transform: transform([1.0, -2.0, 0.5], QUARTER_TURN_Z, [0.1, 0.2, 0.3]),
            },
            Frame {
                resolution: [0, 0, 0],
                table: vec![],
                bricks: vec![],
                transform: transform([0.0, 0.0, 0.0], IDENTITY, [1.0, 1.0, 1.0]),
            },
            Frame {
                // 17 x 8 x 9 cells are 3 x 1 x 2 bricks, three of six held.
                resolution: [17, 8, 9],
                table: vec![FLUID_EMPTY, 2, FLUID_EMPTY, 0, FLUID_EMPTY, 1],
                bricks: [
                    brick([7.0, 0.0, 0.0, 1.0]),
                    brick([0.0, -8.5, 0.0, 2.0]),
                    brick([0.0, 0.0, 9.0, 3.0]),
                ]
                .concat(),
                transform: transform([0.25, 0.5, -0.75], IDENTITY, [0.05, 0.05, 0.4]),
            },
        ];
        write_fluid(dir.path(), &frames, &FluidSettings::default());
        let data = load_dir(dir.path(), 4).unwrap().expect("the session carries a field");
        assert_eq!(data.fields.len(), 1);
        let field = &data.fields[0];
        assert_eq!(field.kind, FieldType::Fluid);
        assert!(field.is_implicit());
        let volume = field.fluid.as_ref().unwrap();
        assert_eq!(volume.frames(), 3);
        assert_eq!(volume.resolution, vec![[9, 3, 2], [0, 0, 0], [17, 8, 9]]);
        assert_eq!(volume.brick_counts, vec![2, 0, 3]);
        assert_eq!(volume.velocity_bound, vec![6.0, 0.0, 9.0]);
        assert_eq!((volume.table_capacity(), volume.brick_capacity()), (6, 3));
        assert!(volume.is_empty_frame(1));
        assert_eq!((volume.start, volume.step), (0.0, 0.1));
        for (k, frame) in frames.iter().enumerate() {
            assert_eq!(volume.transform[k], frame.transform, "frame {k}'s transform");
            assert_eq!(volume.cell(k), [
                frame.transform[12],
                frame.transform[13],
                frame.transform[14]
            ]);
            let (mut table, mut bricks) = (Vec::new(), Vec::new());
            volume.read_frame(k, &mut table, &mut bricks).unwrap();
            assert_eq!(table, frame.table, "frame {k}'s table");
            assert_eq!(bricks, frame.bricks, "frame {k}'s bricks");
        }
        assert_eq!(volume.axis(0, 0), [0.0, 1.0, 0.0]);
        assert_eq!(volume.axis(0, 1), [-1.0, 0.0, 0.0]);
        assert!(data.describe().contains("a volume of 3 frames (1 empty) at most [17, 8, 9] cells"));
    }

    /// A RE-READ IS HELD TO THE LOAD: a frame whose files changed under the
    /// run is refused, naming the file, the frame and both counts, before a
    /// byte of it is read, so it can never be copied past the slot its
    /// load-time count sized. The table is held the same way.
    #[test]
    fn a_reread_frame_must_hold_the_table_and_bricks_load_measured() {
        use test_session::*;
        let dir = tempfile::tempdir().unwrap();
        let one = |value: f32| Frame {
            resolution: [8, 8, 8],
            table: vec![0],
            bricks: brick([value, 0.0, 0.0, 0.0]),
            transform: transform([0.0; 3], IDENTITY, [0.1; 3]),
        };
        write_fluid(dir.path(), &[one(1.0), one(2.0)], &FluidSettings::default());
        let data = load_dir(dir.path(), 1).unwrap().unwrap();
        let volume = data.fields[0].fluid.as_ref().unwrap();
        let (mut table, mut bricks) = (Vec::new(), Vec::new());
        volume.read_frame(1, &mut table, &mut bricks).unwrap();
        // A second, self-consistent brick: the frame verifies against itself
        // and not against the load.
        let grown = [brick([2.0, 0.0, 0.0, 0.0]), brick([3.0, 0.0, 0.0, 0.0])].concat();
        write_words(&volume.bricks[1], &grown);
        let message = volume.read_frame(1, &mut table, &mut bricks).unwrap_err();
        assert!(
            message.contains("frame 1 (")
                && message.contains("bricks-00001.bin): 2 bricks (4096 floats), 1 at load; \
                                     the files changed under the run"),
            "{message}"
        );
        write_words(&volume.tables[0], &[0u32, 0]);
        let message = volume.read_frame(0, &mut table, &mut bricks).unwrap_err();
        assert!(message.contains("2 table entries, 1 at load"), "{message}");
        // A frame the loader never measured cannot be read as a run reads.
        let mut unmeasured = volume.clone();
        unmeasured.brick_counts.clear();
        assert!(unmeasured.read_frame(0, &mut table, &mut bricks).unwrap_err().contains("never measured"));
    }

    /// A transform file of the 13-value layout (one cell for every axis) is
    /// refused rather than read with its frames sheared into one another.
    #[test]
    fn a_transform_of_the_wrong_width_is_refused() {
        use test_session::*;
        let dir = tempfile::tempdir().unwrap();
        let frame = Frame {
            resolution: [8, 8, 8],
            table: vec![FLUID_EMPTY],
            bricks: vec![],
            transform: transform([0.0; 3], IDENTITY, [0.1; 3]),
        };
        write_fluid(dir.path(), &[frame], &FluidSettings::default());
        write_words(&dir.path().join("field-0-fluid/transform.bin"), &[0.0f64; 13]);
        let message = load_dir(dir.path(), 1).unwrap_err();
        assert!(message.contains("13 values is not a whole number of 15-value frames"), "{message}");
    }

    /// `z_up` reaches the texture's flags; a manifest without it keeps the
    /// solver's axes.
    #[test]
    fn a_texture_takes_z_up_from_its_manifest() {
        use test_session::*;
        for (key, want) in [("z_up = true\n", texture_flag::Z_UP), ("", 0)] {
            let dir = tempfile::tempdir().unwrap();
            write_texture(dir.path(), key);
            let data = load_dir(dir.path(), 1).unwrap().unwrap();
            let flags = data.fields[0].texture.as_ref().unwrap().flags;
            assert_eq!(flags & texture_flag::Z_UP, want, "{key:?}");
            assert_eq!(flags & texture_flag::INTERPOLATE, texture_flag::INTERPOLATE);
            assert_eq!(flags & texture_flag::OBJECT_COORDINATES, 0);
        }
    }

    #[test]
    fn a_fluid_field_takes_its_volume_and_no_noise_or_wind_factor() {
        let mut f = record(FieldType::Fluid, 1);
        assert!(verify_field(&f).unwrap_err().contains("carries no volume"));
        f.fluid = Some(fluid(1));
        verify_field(&f).unwrap();
        f.track[TRACK_NOISE] = 0.5;
        assert!(verify_field(&f).unwrap_err().contains("noise amount"));
        let mut g = record(FieldType::Force, 1);
        g.fluid = Some(fluid(1));
        assert!(verify_field(&g).unwrap_err().contains("only a fluid field"));
        assert!(record(FieldType::Fluid, 1).is_implicit());
    }
}
