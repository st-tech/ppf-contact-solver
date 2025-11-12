// File: main.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

mod args;
mod backend;
mod builder;
mod bvh;
mod cvec;
mod cvecvec;
mod data;
mod mesh;
mod scene;
mod triutils;

use args::{ProgramArgs, SimArgs};
use backend::MeshSet;
use clap::Parser;
use data::{BvhSet, DataSet, EdgeParam, EdgeProp, ParamSet};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use log::*;
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use mesh::Mesh as SimMesh;
use more_asserts::*;
use scene::Scene;
use std::collections::HashMap;
use std::ffi::CString;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::raw::c_char;
use {builder::Props, cvec::CVec};

extern crate nalgebra as na;

extern "C" {
    fn set_log_path(data_dir: *const c_char);
}

#[no_mangle]
extern "C" fn print_rust(message: *const libc::c_char) {
    let message = unsafe { std::ffi::CStr::from_ptr(message) }
        .to_str()
        .unwrap();
    info!("{}", message);
}

fn main() {
    let program_args = ProgramArgs::parse();
    let info = git_info::get();
    info!(
        "git branch: {}",
        info.current_branch.unwrap_or("Unknown".to_string())
    );
    info!(
        "git hash: {}",
        info.head.last_commit_hash.unwrap_or("Unknown".to_string())
    );
    info!("float range: from {} to {}", f32::MIN, f32::MAX);
    let mut scene = Scene::new(&program_args);
    let sim_args = scene.args();
    setup(&program_args);

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
        let mut edge_prop = Vec::new();
        for i in 0..mesh.mesh.mesh.edge.ncols() {
            let rod = mesh.mesh.mesh.edge.column(i);
            let x0 = mesh.vertex.column(rod[0]);
            let x1 = mesh.vertex.column(rod[1]);
            if i < mesh.mesh.mesh.rod_count {
                let prop = props.edge[i];
                total_rod_mass += prop.mass;
                edge_prop.push(prop);
            } else {
                let length = (x1 - x0).norm();
                let mut ghat_sum = 0.0;
                let mut offset_sum = 0.0;
                let mut friction_sum = 0.0;
                let mut area_sum = 0.0;
                for &j in mesh.mesh.neighbor.edge.face[i].iter() {
                    let face_prop = props.face.get(j).unwrap();
                    let face_param = &props.face_params[face_prop.param_index as usize];
                    let area = face_area[j];
                    ghat_sum += area * face_param.ghat;
                    friction_sum += area * face_param.friction;
                    offset_sum += area * face_param.offset;
                    area_sum += area;
                }
                assert_gt!(area_sum, 0.0);
                let ghat = ghat_sum / area_sum;
                let offset = offset_sum / area_sum;
                let friction = friction_sum / area_sum;

                // Create edge param and deduplicate
                let param = EdgeParam {
                    stiffness: 0.0,
                    bend: 0.0,
                    ghat,
                    offset,
                    friction,
                };
                let param_idx = *edge_param_map.entry(param).or_insert_with(|| {
                    let new_idx = props.edge_params.len() as u32;
                    props.edge_params.push(param);
                    new_idx
                });

                props.edge.push(EdgeProp {
                    length,
                    mass: 0.0,
                    fixed: false,
                    param_index: param_idx,
                });
            }
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
            "{} {} {}",
            total_rod_mass, total_area_mass, total_vol_mass
        )
        .unwrap();
        let velocity = scene.get_initial_velocity();

        info!("Building dataset (this may take a while)...");
        let dataset = builder::build(&sim_args, mesh, &velocity, &mut props, temp_constraint);
        info!("Dataset built successfully");

        let param = builder::make_param(&sim_args);
        backend.run(&program_args, &sim_args, dataset, param, scene);
    }
}

fn remove_old_states(program_args: &ProgramArgs, keep_states: i32, max_frame: i32) {
    if keep_states > 0 {
        let mut n = 0;
        for i in 0..=max_frame {
            let path = format!("{}/state_{}.bin.gz", program_args.output, i);
            if std::path::Path::new(&path).exists() {
                n += 1;
            }
        }
        info!("Removing old states...");
        let mut i = 0;
        while n > keep_states {
            let path = format!("{}/state_{}.bin.gz", program_args.output, i);
            if std::path::Path::new(&path).exists() {
                info!("Removing {}...", path);
                std::fs::remove_file(path).unwrap_or(());
                n -= 1;
            }
            i += 1;
            if i > max_frame {
                break;
            }
        }
    }
}

fn remove_old_files(dirpath: &str, prefix: &str, suffix: &str, keep_number: i32, max_frame: i32) {
    if keep_number <= 0 {
        return;
    }

    let keep_number = keep_number.max(1); // Ensure at least 1 file is kept

    let mut n = 0;
    for i in 0..=max_frame {
        let path = format!("{}/{}{}{}", dirpath, prefix, i, suffix);
        if std::path::Path::new(&path).exists() {
            n += 1;
        }
    }

    let mut i = 0;
    while n > keep_number {
        let filename = format!("{}{}{}", prefix, i, suffix);
        let path = format!("{}/{}", dirpath, filename);
        if std::path::Path::new(&path).exists() {
            info!("Removing {}...", filename);
            std::fs::remove_file(path).unwrap_or(());
            n -= 1;
        }
        i += 1;
        if i > max_frame {
            break;
        }
    }
}

fn remove_files(base: &str, ext: &str, args: &ProgramArgs) {
    let mut count = args.load + 1;
    loop {
        let path = format!("{}/{}_{}.{}", args.output, base, count, ext);
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

fn setup(program_args: &ProgramArgs) {
    if program_args.load == 0 {
        remove_files_in_dir(&program_args.output).unwrap();
    } else {
        remove_files("vert", "bin", program_args);
        remove_files("state", "bin.gz", program_args);
    }

    if !std::path::Path::new(&program_args.output).exists() {
        std::fs::create_dir_all(&program_args.output).unwrap_or(());
    }
    let save_and_quit_path =
        std::path::Path::new(program_args.output.as_str()).join("save_and_quit");
    if save_and_quit_path.exists() {
        std::fs::remove_file(save_and_quit_path).unwrap_or_else(|err| {
            println!("Failed to delete 'save_and_quit' file: {}", err);
        });
    }

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

fn compress_to_gz(data: Vec<u8>) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    encoder.finish().unwrap()
}

fn decompress_from_gz(compressed_data: Vec<u8>) -> Vec<u8> {
    let mut decoder = GzDecoder::new(compressed_data.as_slice());
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data).unwrap();
    decompressed_data
}

fn save<T: serde::Serialize>(obj: &T, path: &str) {
    let data = bincode::serialize(obj).unwrap();
    std::fs::write(path, compress_to_gz(data)).unwrap();
}

fn read_gz(path: &str) -> Vec<u8> {
    decompress_from_gz(std::fs::read(path).unwrap())
}

fn read<'de, T: serde::Deserialize<'de>>(data: &'de [u8]) -> T {
    bincode::deserialize(data).unwrap()
}
