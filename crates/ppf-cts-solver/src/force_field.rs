// File: force_field.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The external force field's session inputs: sampled grids, compiled
//! scripts, a per-vertex weight and a per-vertex target mask, read from
//! `<session>/bin/force_field/`.
//!
//! EACH SOURCE CARRIES A TARGET BIT. A grid or a script meant for some objects
//! only names a bit below [`ALL_TARGETS`], and the per-vertex mask says which
//! bits reach each vertex; [`ALL_TARGETS`] reaches every vertex, and a field
//! whose sources all do ships no mask.
//!
//! EVERYTHING IS VERIFIED HERE, BEFORE THE DEVICE SEES IT. The kernel
//! (`kernels/energy/external_field.kernel.cpp`) indexes the grids and runs the
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

/// Words per grid in the header table and floats per grid in the box table,
/// `FIELD_GRID_HEADER` and `FIELD_GRID_BOX` in the kernel.
pub const GRID_HEADER_WORDS: usize = 8;
pub const GRID_BOX_FLOATS: usize = 6;
/// Words per script in the script table, `FIELD_SCRIPT_HEADER` in the kernel:
/// code offset, code length, constant offset, constant count, target bit.
pub const SCRIPT_HEADER_WORDS: usize = 5;
/// The target bit of a source that reaches every vertex, `FIELD_ALL_TARGETS`.
pub const ALL_TARGETS: u32 = 32;
/// The largest octave count NOISE and CURL take, `FIELD_MAX_OCTAVES`.
pub const MAX_OCTAVES: usize = 8;

/// What a grid feeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridKind {
    /// Added to the inertial target beside gravity, in m/s^2.
    Acceleration,
    /// Added to the scene wind inside the aerodynamic drag, in m/s.
    AirVelocity,
}

impl GridKind {
    /// The number the kernel's header carries.
    pub fn code(self) -> u32 {
        match self {
            GridKind::Acceleration => 0,
            GridKind::AirVelocity => 1,
        }
    }
}

/// One sampled grid.
#[derive(Debug, Clone)]
pub struct Grid {
    pub kind: GridKind,
    /// `[W, H, D, T]`.
    pub dims: [u32; 4],
    pub min: [f32; 3],
    pub max: [f32; 3],
    /// T strictly increasing instants, in seconds.
    pub times: Vec<f32>,
    /// `[T][D][H][W][3]` floats.
    pub data: Vec<f32>,
    /// The target bit, below [`ALL_TARGETS`], or [`ALL_TARGETS`] for every
    /// vertex.
    pub target: u32,
}

/// One compiled script.
#[derive(Debug, Clone)]
pub struct Script {
    pub code: Vec<u32>,
    pub constants: Vec<f32>,
    /// The source the frontend compiled, kept for the log.
    pub source: String,
    /// As [`Grid::target`].
    pub target: u32,
}

/// Everything the field needs, as the loader read and verified it.
#[derive(Debug, Clone, Default)]
pub struct ForceFieldData {
    pub grids: Vec<Grid>,
    pub scripts: Vec<Script>,
    /// One weight per dynamic vertex, or `None` for all ones.
    pub weight: Option<Vec<f32>>,
    /// One target mask per dynamic vertex, bit `b` set when the sources with
    /// target bit `b` reach it; `None` when every source reaches every vertex.
    pub target_mask: Option<Vec<u32>>,
}

impl ForceFieldData {
    pub fn is_empty(&self) -> bool {
        self.grids.is_empty() && self.scripts.is_empty()
    }

    pub fn has_air(&self) -> bool {
        self.grids.iter().any(|g| g.kind == GridKind::AirVelocity)
    }

    /// The bytes the grids hold, which is what the frontend's estimate named.
    pub fn grid_bytes(&self) -> usize {
        self.grids.iter().map(|g| g.data.len() * 4).sum()
    }

    /// One line for the log.
    pub fn describe(&self) -> String {
        let mut parts: Vec<String> = self
            .grids
            .iter()
            .map(|g| {
                format!(
                    "{:?} grid {}x{}x{}x{} over [{:?}, {:?}]",
                    g.kind, g.dims[0], g.dims[1], g.dims[2], g.dims[3], g.min, g.max
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
        format!(
            "force field: {} ({:.1} MB of grids)",
            parts.join(", "),
            self.grid_bytes() as f64 / 1.0e6
        )
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestGrid {
    kind: String,
    dims: [u32; 4],
    min: [f32; 3],
    max: [f32; 3],
    times: Vec<f32>,
    data: String,
    #[serde(default = "all_targets")]
    targets: u32,
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
    grid: Vec<ManifestGrid>,
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
    for (index, entry) in manifest.grid.iter().enumerate() {
        let what = format!("{} grid {index}", manifest_path.display());
        let kind = match entry.kind.as_str() {
            "acceleration" => GridKind::Acceleration,
            "air-velocity" => GridKind::AirVelocity,
            other => {
                return Err(format!(
                    "{what}: kind {other:?} is neither \"acceleration\" nor \"air-velocity\""
                ))
            }
        };
        let data = read_f32(&dir.join(&entry.data))?;
        let grid = Grid {
            kind,
            dims: entry.dims,
            min: entry.min,
            max: entry.max,
            times: entry.times.clone(),
            data,
            target: entry.targets,
        };
        verify_grid(&grid).map_err(|e| format!("{what}: {e}"))?;
        out.grids.push(grid);
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
        .grids
        .iter()
        .map(|g| g.target)
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
            "{}: the manifest names neither a grid nor a script",
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

fn read_u32(path: &PathBuf) -> Result<Vec<u32>, String> {
    let bytes = read_bytes(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: {} bytes is not a whole number of u32", path.display(), bytes.len()));
    }
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// A grid the kernel can index without a bounds question.
pub fn verify_grid(grid: &Grid) -> Result<(), String> {
    let [w, h, d, t] = grid.dims;
    if w < 2 || h < 2 || d < 2 {
        return Err(format!(
            "spatial extents {w}x{h}x{d}: every extent needs at least 2 samples, the two \
             corners of its cell"
        ));
    }
    if t < 1 {
        return Err("the grid has no time sample".into());
    }
    let expected = (w as u64) * (h as u64) * (d as u64) * (t as u64) * 3;
    if expected > u32::MAX as u64 {
        return Err(format!("{expected} floats do not fit the kernel's 32-bit offsets"));
    }
    if grid.data.len() as u64 != expected {
        return Err(format!(
            "{} floats, but {w}x{h}x{d}x{t} samples of 3 components need {expected}",
            grid.data.len()
        ));
    }
    if grid.times.len() != t as usize {
        return Err(format!("{} instants for {t} time samples", grid.times.len()));
    }
    if grid.times.iter().any(|x| !x.is_finite()) {
        return Err("an instant is not finite".into());
    }
    if grid.times.windows(2).any(|p| p[1] <= p[0]) {
        return Err(format!("the instants {:?} are not strictly increasing", grid.times));
    }
    for axis in 0..3 {
        let (lo, hi) = (grid.min[axis], grid.max[axis]);
        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
            return Err(format!("axis {axis} of the box is [{lo}, {hi}], which is not an interval"));
        }
    }
    if let Some(bad) = grid.data.iter().position(|v| !v.is_finite()) {
        return Err(format!("sample value {bad} is {}", grid.data[bad]));
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

    #[test]
    fn a_grid_must_match_its_declared_shape_and_order_its_instants() {
        let mut grid = Grid {
            kind: GridKind::Acceleration,
            dims: [2, 2, 2, 2],
            min: [0.0; 3],
            max: [1.0; 3],
            times: vec![0.0, 1.0],
            data: vec![0.0; 2 * 2 * 2 * 2 * 3],
            target: ALL_TARGETS,
        };
        verify_grid(&grid).unwrap();
        grid.times = vec![1.0, 1.0];
        assert!(verify_grid(&grid).unwrap_err().contains("strictly increasing"));
        grid.times = vec![0.0, 1.0];
        grid.data.pop();
        assert!(verify_grid(&grid).unwrap_err().contains("need"));
        grid.data.push(f32::NAN);
        assert!(verify_grid(&grid).unwrap_err().contains("NaN"));
        grid.data = vec![0.0; 48];
        grid.dims = [1, 2, 2, 2];
        grid.data = vec![0.0; 24];
        assert!(verify_grid(&grid).unwrap_err().contains("at least 2"));
    }
}
