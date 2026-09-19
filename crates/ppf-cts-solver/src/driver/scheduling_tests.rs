//! Real generated entries through the dispatcher, with only declaration costs varied.
//! Modes are row-cost baseline (0), production group costs (1), forced serial (2),
//! and one whole group per chunk (3). No scatter classification is changed.
use super::*;
use ppf_cts_compute::{AllocLabel, Buffer, Device, EncoderExt, Handle, KernelDecl};
use std::sync::OnceLock;
use std::time::Instant;

pub(crate) fn device(mode: usize) -> HostDevice {
    static TABLES: OnceLock<[Vec<KernelDecl>; 4]> = OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        std::array::from_fn(|mode| {
            let mut table = kernels::TABLE.to_vec();
            for (id, old, scaled) in [
                (kernels::id::PCG_UPDATE_ROW_FOLDED, 8.0, 8.0 * 256.0),
                (kernels::id::OPERATOR_APPLY_FOLDED, 6.0, 6.0 * 256.0),
                (kernels::id::OPERATOR_APPLY_DYNAMIC_FOLDED, 6.0, 6.0 * 32.0),
            ] {
                table[id.0 as usize].nanos_per_item = match mode {
                    0 => old,
                    1 => {
                        assert_eq!(table[id.0 as usize].nanos_per_item, scaled);
                        scaled
                    }
                    2 => 0.0,
                    _ => 20_000.0,
                };
            }
            table
        })
    });
    HostDevice::new(&tables[mode], &LAUNCH)
}

struct Fixture {
    device: HostDevice,
    floats: Vec<Buffer<f32>>,
    indices: Vec<Buffer<u32>>,
}

impl Fixture {
    fn floats(&mut self, values: &[f32]) -> Handle {
        let mut buffer = Buffer::none();
        buffer
            .size(&mut self.device, values.len(), AllocLabel("test.schedule"))
            .unwrap();
        buffer.write(&mut self.device, 0, values).unwrap();
        let handle = buffer.handle();
        self.floats.push(buffer);
        handle
    }

    fn indices(&mut self, values: &[u32]) -> Handle {
        let mut buffer = Buffer::none();
        buffer
            .size(&mut self.device, values.len(), AllocLabel("test.schedule"))
            .unwrap();
        buffer.write(&mut self.device, 0, values).unwrap();
        let handle = buffer.handle();
        self.indices.push(buffer);
        handle
    }

    fn read(&mut self, handle: Handle) -> Vec<u32> {
        let buffer = self
            .floats
            .iter()
            .find(|buffer| buffer.handle() == handle)
            .unwrap();
        let mut values = vec![0.0; buffer.len()];
        buffer.read(&mut self.device, 0, &mut values).unwrap();
        values.into_iter().map(f32::to_bits).collect()
    }
}

fn exercise(
    rows: usize,
    width: usize,
    mode: usize,
    rounds: usize,
    repeats: usize,
) -> (Vec<f64>, Vec<u32>) {
    let mut f = Fixture {
        device: device(mode),
        floats: vec![],
        indices: vec![],
    };
    let diagonal: Vec<f32> = (0..9 * rows)
        .map(|i| if i % 9 % 4 == 0 { 1.0 } else { 0.0 })
        .collect();
    let diagonal = f.floats(&diagonal);
    let x = f.floats(
        &(0..3 * rows)
            .map(|i| (i % 31) as f32 * 0.03125 - 0.5)
            .collect::<Vec<_>>(),
    );
    let result = f.floats(&vec![0.0; 3 * rows]);
    let groups = rows.div_ceil(if width == 0 { 256 } else { 32 });
    let curvature = f.floats(&vec![0.0; groups]);
    let absolute = f.floats(&vec![0.0; groups]);
    let empty_index = f.indices(&[]);
    let empty_value = f.floats(&[]);
    let empty_offsets = f.indices(&vec![0; rows + 1]);
    let mut index = Vec::new();
    let mut offset = vec![0];
    let mut transpose = vec![Vec::new(); rows];
    for row in 0..rows {
        for column in row + 1..(row + 1 + width).min(rows) {
            transpose[column].push((row as u32, index.len() as u32));
            index.push(column as u32);
        }
        offset.push(index.len() as u32);
    }
    let mut ti = Vec::new();
    let mut tv = Vec::new();
    let mut to = vec![0];
    for entries in transpose {
        for (row, slot) in entries {
            ti.push(row);
            tv.push(slot);
        }
        to.push(ti.len() as u32);
    }
    let values = f.floats(
        &(0..index.len() * 9)
            .map(|i| if i % 9 % 4 == 0 { 0.125 } else { 0.0 })
            .collect::<Vec<_>>(),
    );
    let dynamic = kernels::OperatorApplyDynamicFoldedArgs {
        dyn_index: f.indices(&index),
        dyn_value: values,
        dyn_offset: f.indices(&offset),
        dyn_reference_index: f.indices(&ti),
        dyn_reference_value: f.indices(&tv),
        dyn_reference_offset: f.indices(&to),
        dyn_global_value: values,
        index: empty_index,
        offset: empty_offsets,
        value: empty_value,
        transpose_pair: empty_index,
        transpose_offset: empty_offsets,
        diagonal,
        diagonal_shift: 0.0,
        x,
        result,
        curvature_total: curvature,
        absolute_total: absolute,
        rows: rows as u32,
        count: groups as u32,
        seam_arena_count: 0,
    };
    let fixed = kernels::OperatorApplyFoldedArgs {
        index: empty_index,
        offset: empty_offsets,
        value: empty_value,
        transpose_pair: empty_index,
        transpose_offset: empty_offsets,
        diagonal,
        diagonal_shift: 0.0,
        x,
        result,
        curvature_total: curvature,
        absolute_total: absolute,
        rows: rows as u32,
        count: groups as u32,
        seam_arena_count: 0,
    };
    let update_groups = rows.div_ceil(256);
    let update = kernels::PcgUpdateRowFoldedArgs {
        direction: x,
        product_direction: x,
        alpha: f.floats(&[0.0001]),
        inverse_diagonal: diagonal,
        iterate: f.floats(&vec![0.0; rows * 3]),
        residual: f.floats(&vec![1.0; rows * 3]),
        preconditioned: f.floats(&vec![0.0; rows * 3]),
        product_total: f.floats(&vec![0.0; update_groups]),
        residual_total: f.floats(&vec![0.0; update_groups]),
        rows: rows as u32,
        count: update_groups as u32,
        seam_arena_count: 0,
    };
    let mut times = Vec::new();
    for _ in 0..rounds {
        for kind in 0..2 {
            let start = Instant::now();
            for _ in 0..repeats {
                f.device
                    .run("test.schedule", |encoder| unsafe {
                        if kind == 1 {
                            encoder.groups(&update, update_groups as u32, 256)
                        } else if width == 0 {
                            encoder.groups(&fixed, groups as u32, 256)
                        } else {
                            encoder.groups(&dynamic, groups as u32, 256)
                        }
                    })
                    .unwrap();
            }
            times.push(start.elapsed().as_secs_f64() * 1e6 / repeats as f64);
        }
    }
    let output = [
        result,
        curvature,
        absolute,
        update.iterate,
        update.residual,
        update.preconditioned,
        update.product_total,
        update.residual_total,
    ]
    .into_iter()
    .flat_map(|handle| f.read(handle))
    .collect();
    (times, output)
}

#[test]
fn scheduling_preserves_group_results_bitwise() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        for width in [0, 4, 16] {
            let (_, serial) = exercise(4097, width, 2, 1, 1);
            for mode in [0, 1, 3] {
                assert_eq!(
                    serial,
                    exercise(4097, width, mode, 1, 1).1,
                    "scheduling changed a group result at width {width}, mode {mode}"
                );
            }
        }
    });
}

#[test]
#[ignore = "release dispatcher timing, run alone with --nocapture"]
fn benchmark_group_scheduling() {
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for rows in [1025, 81920, 199999, 200001, 600001] {
                for width in [0, 4, 16] {
                    let mut samples: [Vec<Vec<f64>>; 3] = std::array::from_fn(|_| vec![vec![], vec![]]);
                    for trial in 0..7 {
                        let order = if trial % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
                        let mut expected = None;
                        for mode in order {
                            let (times, output) = exercise(rows, width, mode, 2, 8);
                            if let Some(ref expected) = expected { assert_eq!(&output, expected); }
                            else { expected = Some(output); }
                            for kind in 0..2 { samples[mode][kind].push(times[2 + kind]); }
                        }
                    }
                    for (mode, samples) in samples.iter_mut().enumerate() {
                        for (kind, values) in samples.iter_mut().enumerate() {
                            values.sort_by(f64::total_cmp);
                            println!("schedule threads={threads} rows={rows} width={width} mode={mode} kind={kind} us={:.3} min={:.3} max={:.3}",
                                values[3], values[0], values[6]);
                        }
                    }
                }
            }
        });
    }
}
