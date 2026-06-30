// File: main.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

mod args;
mod backend;
mod builder;
mod cvec;
mod cvecvec;
mod data;
mod mesh;
mod raw_vec;
mod scene;
mod status_writer;
mod triutils;

use args::{ProgramArgs, SimArgs};
use backend::MeshSet;
use clap::Parser;
use data::{DataSet, EdgeParam, EdgeProp, ParamSet};
use mesh::Mesh as SimMesh;
use {builder::Props, cvec::CVec};
use rayon::prelude::*;
use std::collections::HashMap;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use log::*;
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use scene::Scene;
use std::ffi::CString;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::raw::c_char;

extern crate nalgebra as na;

extern "C" {
    fn set_log_path(data_dir: *const c_char);
}

#[no_mangle]
extern "C" fn print_rust(message: *const libc::c_char) {
    let message = unsafe { std::ffi::CStr::from_ptr(message) }
        .to_str()
        .unwrap();
    info!("{message}");
}

fn main() {
    let mut program_args = ProgramArgs::parse();
    // `--load -1` is the "resume from latest checkpoint" sentinel the
    // server emits when the user clicks Resume (see
    // ppf-cts-server::executor::solver::launch_solver). The rest of
    // this binary keys off `program_args.load > 0` for the resume
    // path, so resolve the sentinel to a real frame index before
    // anything reads it. Without this, `--load -1` falls into the
    // fresh-start branch *and* trips `remove_files("vert"|"state",
    // ...)` in `setup()` starting at index 0, which destroys the
    // very checkpoint and vert frames the user wanted to resume from.
    if program_args.load < 0 {
        let output_dir = std::path::Path::new(&program_args.output);
        let saved = ppf_cts_core::datamodel::list_saved_states(output_dir);
        match saved.into_iter().max() {
            Some(n) => {
                program_args.load = n as i32;
            }
            None => {
                eprintln!(
                    "--load {} (resume from latest) but no state_<N>.bin.gz found in {}",
                    program_args.load, program_args.output
                );
                std::process::exit(1);
            }
        }
    }
    // Initialize the logger before emitting any diagnostics. The git
    // info! lines below (and the print_rust FFI bridge)
    // route to the global NOP logger until log4rs is installed inside
    // setup(), so they are silently dropped if setup() runs later.
    // setup() only reads program_args (load/output, both resolved by
    // the --load block above) and operates on args.output, while
    // Scene::new below reads args.path, so there is no read-after-delete
    // hazard from running setup() first.
    setup(&program_args);

    // Acquire the liveness lock and begin the structured status record
    // right after setup()'s output-dir wipe (load==0) and before any
    // backend work, so the lock covers the whole compute lifetime and a
    // fresh run never reads its own scrubbed record. launch_id only needs
    // to be unique across launches in the same output dir, so a
    // pid + wall-clock mix is sufficient (not a security token).
    let launch_id = {
        let pid = std::process::id() as u64;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let mixed = pid
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(nanos)
            & 0xFFFF_FFFF_FFFF;
        format!("{mixed:012x}")
    };
    status_writer::init(&program_args.output, launch_id, program_args.load > 0);

    let info = git_info::get();
    info!(
        "git branch: {}",
        info.current_branch.unwrap_or("Unknown".to_string())
    );
    info!(
        "git hash: {}",
        info.head.last_commit_hash.unwrap_or("Unknown".to_string())
    );
    let mut scene = Scene::new(&program_args);
    let sim_args = scene.args();

    if program_args.load > 0 {
        let mut param = builder::make_param(&sim_args);
        info!("Loading dataset...");
        let mut dataset = read(&read_gz(&format!("{}/dataset.bin.gz", program_args.output)));
        info!("Loading backend state...");
        let mut backend = backend::Backend::load_state(program_args.load, &program_args.output);
        info!("Data loaded successfully");
        param.time = backend.state.time;
        param.prev_dt = backend.state.prev_dt;
        builder::copy_to_dataset(
            &backend.state.curr_vertex,
            &backend.state.prev_vertex,
            &mut dataset,
        );
        backend.run(&program_args, &sim_args, dataset, param, scene);
    } else {
        info!("Initializing mesh...");
        let mut backend = backend::Backend::new(scene.make_mesh());
        let mesh = &backend.mesh;
        let face_area = triutils::face_areas(&mesh.vertex, &mesh.mesh.mesh.face);
        let tet_volumes = triutils::tet_volumes(&mesh.vertex, &mesh.mesh.mesh.tet);

        info!("Building material properties...");
        let mut props = scene.make_props(mesh, &face_area, &tet_volumes);

        info!("Computing constraints and parameters...");
        let time = backend.state.time;
        let temp_constraint = scene.make_constraint(time);
        scene.export_param_summary(&program_args, &props, &face_area, &tet_volumes);

        let mut total_rod_mass = 0.0;
        let mut total_area_mass = 0.0;
        let mut total_vol_mass = 0.0;

        for prop in props.face.iter() {
            total_area_mass += prop.mass;
        }

        info!("Building edge parameter map...");
        // Build edge param map for deduplication
        let mut edge_param_map: HashMap<EdgeParam, u32> = HashMap::new();
        for (i, param) in props.edge_params.iter().enumerate() {
            edge_param_map.insert(*param, i as u32);
        }

        info!("Processing edge properties...");
        // Process rod edges (sequential, small count)
        for i in 0..mesh.mesh.mesh.rod_count {
            let prop = props.edge[i];
            total_rod_mass += prop.mass;
        }

        // Process non-rod edges (parallelized)
        // Step 1: Parallel computation
        let rod_count = mesh.mesh.mesh.rod_count;
        let edge_count = mesh.mesh.mesh.edge.ncols();
        let non_rod_edge_data: Vec<(f32, EdgeParam)> = (rod_count..edge_count)
            .into_par_iter()
            .map(|i| {
                let rod = mesh.mesh.mesh.edge.column(i);
                let x0 = mesh.vertex.column(rod[0]);
                let x1 = mesh.vertex.column(rod[1]);
                let length = (x1 - x0).norm();
                let param = builder::averaged_edge_param(
                    &mesh.mesh.neighbor.edge.face[i],
                    &props.face,
                    &props.face_params,
                    |j, _| face_area[j],
                );
                (length, param)
            })
            .collect();

        // Step 2: Sequential deduplication
        for (length, param) in non_rod_edge_data {
            let param_idx = *edge_param_map.entry(param).or_insert_with(|| {
                let new_idx = props.edge_params.len() as u32;
                props.edge_params.push(param);
                new_idx
            });
            props.edge.push(EdgeProp {
                length,
                initial_length: length,
                mass: 0.0,
                fixed: false,
                param_index: param_idx,
            });
        }

        for prop in props.tet.iter() {
            total_vol_mass += prop.mass;
        }

        let mut mass_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}/data/total_mass.out", program_args.output).as_str())
            .unwrap();
        writeln!(
            mass_file,
            "{total_rod_mass} {total_area_mass} {total_vol_mass}"
        )
        .unwrap();
        let velocity = scene.get_initial_velocity();

        info!("Building dataset (this may take a while)...");
        let pdrd_data = builder::PdrdSceneData {
            body_rows: scene.pdrd_body_rows(),
            vert_index: scene.pdrd_vert_index(),
            vert_list: scene.pdrd_vert_list(),
            rest_centered: scene.pdrd_rest_centered(),
        };
        let dataset = builder::build(
            &sim_args,
            mesh,
            &velocity,
            &mut props,
            temp_constraint,
            pdrd_data,
        );
        info!("Dataset built successfully");

        let param = builder::make_param(&sim_args);
        backend.run(&program_args, &sim_args, dataset, param, scene);
    }
}

fn remove_old_files(dirpath: &str, prefix: &str, suffix: &str, keep_number: i32, max_frame: i32) {
    if keep_number <= 0 {
        return;
    }

    let mut n = 0;
    for i in 0..=max_frame {
        let path = format!("{dirpath}/{prefix}{i}{suffix}");
        if std::path::Path::new(&path).exists() {
            n += 1;
        }
    }

    let mut i = 0;
    while n > keep_number {
        let filename = format!("{prefix}{i}{suffix}");
        let path = format!("{dirpath}/{filename}");
        if std::path::Path::new(&path).exists() {
            info!("Removing {filename}...");
            std::fs::remove_file(path).unwrap_or(());
            n -= 1;
        }
        i += 1;
        if i > max_frame {
            break;
        }
    }
}

fn remove_files(dirpath: &str, base: &str, ext: &str, load: i32) {
    let mut count = load + 1;
    loop {
        let path = format!("{dirpath}/{base}_{count}.{ext}");
        if std::path::Path::new(&path).exists() {
            std::fs::remove_file(path).unwrap_or(());
        } else {
            break;
        }
        count += 1;
    }
}

fn remove_files_in_dir(path: &str) -> std::io::Result<()> {
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.into_iter().flatten() {
            // ``save_and_quit`` is a control sentinel the server may
            // write between launch and the solver's setup(). Wiping
            // it here would lose the save request and the solver
            // would run to completion. Preserve it. The server's
            // launch_solver already cleared stale sentinels before
            // spawning, so anything present now is fresh.
            if entry.file_name() == "save_and_quit" {
                continue;
            }
            let path = entry.path();
            if path.is_file() {
                std::fs::remove_file(path)?;
            } else if path.is_dir() {
                std::fs::remove_dir_all(path)?;
            }
        }
    }
    Ok(())
}

/// Read the simulation time recorded for `load` from
/// ``<output>/data/frame_to_time.out``. That stream is written as
/// ``{frame} {time}`` once per output frame (see
/// Backend::write_frame_outputs), so the row whose first column equals
/// `load` gives the sim time at the resume checkpoint. Returns None when
/// the file is missing or has no matching row; callers then skip
/// time-keyed pruning rather than guess a threshold.
fn loaded_time_for_frame(output: &str, load: i32) -> Option<f64> {
    let contents = std::fs::read_to_string(format!("{output}/data/frame_to_time.out")).ok()?;
    for line in contents.lines() {
        let mut cols = line.split_whitespace();
        let frame: i32 = cols.next()?.parse().ok()?;
        if frame == load {
            return cols.next()?.parse().ok();
        }
    }
    None
}

/// On resume, drop rows that belong to frames the run is about to
/// resimulate from the ``<output>/data/*.out`` streams. All of these are
/// opened append-only (Rust-side time_per_frame/frame_to_time/clock and
/// the C++ SimpleLog metrics), so without this the frames after the
/// checkpoint would get duplicate keys, skewing the averaged-metric
/// readers in ppf-cts-core's session::log.
///
/// time_per_frame.out and frame_to_time.out are keyed by integer frame
/// index, so keep rows with ``frame <= load``. Every other stream is
/// keyed by sim time (the C++ SimpleLog writes ``{time} {value}``, and
/// clock.out matches), so keep rows with ``time <= loaded_time``. Lines
/// whose first column is not the expected numeric key are left in place
/// to avoid silently dropping data we don't understand.
fn prune_out_streams(output: &str, load: i32) {
    let data_dir = format!("{output}/data");
    let loaded_time = loaded_time_for_frame(output, load);
    let entries = match std::fs::read_dir(&data_dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("out") {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => continue,
        };
        let frame_keyed = name == "time_per_frame.out" || name == "frame_to_time.out";
        // Without a sim-time threshold there is nothing to prune the
        // time-keyed streams against, so leave them untouched.
        if !frame_keyed && loaded_time.is_none() {
            continue;
        }
        let contents = match std::fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(_) => continue,
        };
        let mut kept = String::with_capacity(contents.len());
        for line in contents.lines() {
            let keep = match line.split_whitespace().next() {
                Some(first) if frame_keyed => match first.parse::<i32>() {
                    Ok(frame) => frame <= load,
                    Err(_) => true,
                },
                Some(first) => match first.parse::<f64>() {
                    Ok(time) => time <= loaded_time.unwrap(),
                    Err(_) => true,
                },
                None => true,
            };
            if keep {
                kept.push_str(line);
                kept.push('\n');
            }
        }
        std::fs::write(&path, kept).unwrap_or(());
    }
}

fn setup(program_args: &ProgramArgs) {
    if program_args.load == 0 {
        remove_files_in_dir(&program_args.output).unwrap();
    } else {
        remove_files(&program_args.output, "vert", "bin", program_args.load);
        // State checkpoints are sparse on disk: save_state writes
        // state_<N>.bin.gz only every auto_save frames, and
        // remove_old_files keeps just keep_states of them at the
        // highest indices. So the indices are scattered high values,
        // not contiguous from load+1. A contiguous break-on-gap scan
        // (like the vert one above) would stop at the first absent
        // index after the resume point and leave newer checkpoints
        // behind, which a later `--load -1` (resume from latest) would
        // then resolve to and jump past the frame the user branched
        // from. Enumerate the actual on-disk set and drop every index
        // newer than the resume frame instead of assuming contiguity.
        let output_dir = std::path::Path::new(&program_args.output);
        for n in ppf_cts_core::datamodel::list_saved_states(output_dir) {
            if (n as i32) > program_args.load {
                let path = format!(
                    "{}/{}",
                    program_args.output,
                    ppf_cts_formats::files::state_filename(n as i32)
                );
                std::fs::remove_file(path).unwrap_or(());
            }
        }
        // The .out streams in output/data are append-only, so the resume
        // run would re-append rows for frames load+1.. that the original
        // run already recorded. Prune them to frames/time <= the
        // checkpoint so resumed frames don't produce duplicate keys.
        prune_out_streams(&program_args.output, program_args.load);
    }

    if !std::path::Path::new(&program_args.output).exists() {
        std::fs::create_dir_all(&program_args.output).unwrap_or(());
    }
    // ``output/data/`` holds the per-frame .out streams (total_mass,
    // time_per_frame, frame_to_time, ...). Nothing else creates it,
    // so a fresh run on a clean output dir hits NotFound when the
    // first writeln! fires. Make sure the dir exists up-front.
    let data_dir = format!("{}/data", program_args.output);
    if !std::path::Path::new(&data_dir).exists() {
        std::fs::create_dir_all(&data_dir).unwrap_or(());
    }
    // The server's launch_solver removes any stale ``save_and_quit``
    // sentinel before spawning this process, so anything present here
    // was written between spawn and setup() and reflects the user's
    // intent. Don't preemptively delete it: the run() loop will pick
    // it up on the first iteration and call save_state.

    let pattern = "[{d(%Y-%m-%d %H:%M:%S)}] {m}{n}";
    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build();
    let config = Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(Root::builder().appender("stdout").build(LevelFilter::Info))
        .unwrap();

    log4rs::init_config(config).unwrap();

    info!("{}", std::env::args().collect::<Vec<_>>().join(" "));
    let data_dir = CString::new(format!("{}/data", program_args.output)).unwrap();
    unsafe {
        set_log_path(data_dir.as_ptr());
    }
}

fn decompress_from_gz(compressed_data: Vec<u8>, path: &str) -> Vec<u8> {
    let mut decoder = GzDecoder::new(compressed_data.as_slice());
    let mut decompressed_data = Vec::new();
    if let Err(e) = decoder.read_to_end(&mut decompressed_data) {
        panic!(
            "Failed to decompress '{path}': {e}. The file is likely truncated \
             or corrupt, for example an auto-save that was interrupted or an \
             incomplete file transfer. Resuming from this checkpoint is not \
             possible; re-run the simulation or resume from a valid checkpoint."
        );
    }
    decompressed_data
}

/// 64 KiB I/O write buffer.
const IO_BUF_CAPACITY: usize = 64 * 1024;

/// Streams serialization directly to a gzip-compressed file without buffering entire data in RAM.
///
/// The bytes are written to a temporary sibling file and then atomically
/// renamed into `path`. A process killed mid-write (e.g. an interrupted
/// auto-save) therefore leaves either the previous valid file or the fully
/// written new one, never a truncated gzip that fails to decompress on resume
/// with `UnexpectedEof`. The temp file is fsynced before the rename so a crash
/// right after the rename cannot expose partially flushed contents.
fn save<T: serde::Serialize>(obj: &T, path: &str) {
    let tmp_path = format!("{}.tmp.{}", path, std::process::id());
    {
        let file = std::fs::File::create(&tmp_path).unwrap();
        let buf_writer = std::io::BufWriter::with_capacity(IO_BUF_CAPACITY, file);
        let mut encoder = GzEncoder::new(buf_writer, Compression::default());
        bincode::serialize_into(&mut encoder, obj).unwrap();
        let buf_writer = encoder.finish().unwrap();
        let file = buf_writer.into_inner().unwrap();
        file.sync_all().unwrap();
    }
    std::fs::rename(&tmp_path, path).unwrap();
}

fn read_gz(path: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read '{path}': {e}"));
    decompress_from_gz(bytes, path)
}

fn read<'de, T: serde::Deserialize<'de>>(data: &'de [u8]) -> T {
    bincode::deserialize(data).unwrap()
}

#[cfg(test)]
mod tests {
    use super::prune_out_streams;
    use std::collections::HashSet;
    use std::fs;

    fn write_stream(data_dir: &std::path::Path, name: &str, body: &str) {
        fs::write(data_dir.join(name), body).unwrap();
    }

    #[test]
    fn prune_out_streams_drops_resimulated_frames() {
        let output = std::env::temp_dir().join(format!(
            "ppf_cts_prune_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let data_dir = output.join("data");
        fs::create_dir_all(&data_dir).unwrap();

        let load = 3;
        // frame_to_time maps frame -> sim time; the row for `load` sets the
        // sim-time threshold the time-keyed streams are pruned against.
        write_stream(
            &data_dir,
            "frame_to_time.out",
            "0 0\n1 0.04\n2 0.08\n3 0.12\n4 0.16\n5 0.2\n",
        );
        write_stream(
            &data_dir,
            "time_per_frame.out",
            "0 10\n1 11\n2 12\n3 13\n4 14\n5 15\n",
        );
        // clock.out (and the C++ SimpleLog streams) are keyed by sim time.
        write_stream(
            &data_dir,
            "clock.out",
            "0.000000 1500\n0.040000 1500\n0.080000 1500\n0.120000 1500\n0.160000 1400\n",
        );
        write_stream(
            &data_dir,
            "advance.total_mass.out",
            "0.000000 12.5\n0.080000 12.5\n0.120000 12.5\n0.200000 12.5\n",
        );

        prune_out_streams(output.to_str().unwrap(), load);

        // Frame-keyed streams keep only frame <= load with no duplicate keys.
        for name in ["frame_to_time.out", "time_per_frame.out"] {
            let contents = fs::read_to_string(data_dir.join(name)).unwrap();
            let mut seen = HashSet::new();
            for line in contents.lines() {
                let frame: i32 = line.split_whitespace().next().unwrap().parse().unwrap();
                assert!(frame <= load, "{name}: frame {frame} > load {load} survived");
                assert!(seen.insert(frame), "{name}: duplicate frame {frame}");
            }
            assert_eq!(seen.len(), (load + 1) as usize, "{name}: wrong row count");
        }

        // Time-keyed streams keep only rows up to the checkpoint sim time.
        for name in ["clock.out", "advance.total_mass.out"] {
            let contents = fs::read_to_string(data_dir.join(name)).unwrap();
            let mut seen: HashSet<u64> = HashSet::new();
            for line in contents.lines() {
                let time: f64 = line.split_whitespace().next().unwrap().parse().unwrap();
                assert!(time <= 0.12 + 1e-9, "{name}: time {time} > loaded time survived");
                assert!(seen.insert(time.to_bits()), "{name}: duplicate time {time}");
            }
        }

        fs::remove_dir_all(&output).unwrap_or(());
    }

    #[test]
    fn save_round_trips_and_leaves_no_temp_file() {
        let dir = std::env::temp_dir().join(format!(
            "ppf_cts_save_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dataset.bin.gz");
        let path_str = path.to_str().unwrap();

        let original: Vec<u64> = (0..10_000u64).collect();
        super::save(&original, path_str);

        // The atomically renamed file decompresses and deserializes cleanly,
        // and no `.tmp.<pid>` sibling is left behind after a successful save.
        let loaded: Vec<u64> = super::read(&super::read_gz(path_str));
        assert_eq!(loaded, original);
        let temp_left = fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| e.file_name().to_string_lossy().contains(".tmp."));
        assert!(!temp_left, "temp file was not renamed away");

        // A pre-existing truncated file must not block a fresh valid save.
        fs::write(&path, b"\x1f\x8b\x08not a complete gzip stream").unwrap();
        super::save(&original, path_str);
        let reloaded: Vec<u64> = super::read(&super::read_gz(path_str));
        assert_eq!(reloaded, original);

        fs::remove_dir_all(&dir).unwrap_or(());
    }
}
