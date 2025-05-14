// File: main.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
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

use args::Args;
use backend::MeshSet;
use clap::Parser;
use data::{BvhSet, DataSet, FaceProp, ParamSet, TetProp};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use log::*;
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use mesh::Mesh as SimMesh;
use scene::Scene;
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
    let mut args = Args::parse();
    let info = git_info::get();
    info!(
        "git branch: {}",
        info.current_branch.unwrap_or("Unknown".to_string())
    );
    info!(
        "git hash: {}",
        info.head.last_commit_hash.unwrap_or("Unknown".to_string())
    );
    let mut scene = Scene::new(&args);
    scene.override_args(&mut args);
    setup(&args);

    if args.load > 0 {
        let mut param = builder::make_param(&args);
        println!("loading dataset...");
        let mut dataset = read(&read_gz(&format!("{}/dataset.bin.gz", args.output)));
        println!("loading backend...");
        let mut backend = backend::Backend::load_state(&args, args.load, &args.output);
        param.time = backend.state.time;
        param.prev_dt = backend.state.prev_dt;
        builder::copy_to_dataset(
            &backend.state.curr_vertex,
            &backend.state.prev_vertex,
            &mut dataset,
        );
        backend.run(&args, dataset, param, scene);
    } else {
        let mut backend = backend::Backend::new(&args, scene.make_mesh(&args));
        let mesh = &backend.mesh;
        let face_area = triutils::face_areas(&mesh.vertex, &mesh.mesh.mesh.face);
        let tet_volumes = triutils::tet_volumes(&mesh.vertex, &mesh.mesh.mesh.tet);
        let mut props = Props {
            rod: Vec::new(),
            face: Vec::new(),
            tet: Vec::new(),
        };
        let mut total_rod_mass = 0.0;
        let mut total_area_mass = 0.0;
        let mut total_vol_mass = 0.0;
        for i in 0..mesh.mesh.mesh.rod_count {
            let rod = mesh.mesh.mesh.edge.column(i);
            let x0 = mesh.vertex.column(rod[0]);
            let x1 = mesh.vertex.column(rod[1]);
            let r = args.rod_offset;
            let mut length = (x1 - x0).map(f32::from).norm();
            if let Some(seg_len_factor) = mesh.mesh.mesh.rod_length_factor.as_ref() {
                length *= seg_len_factor[i];
            }
            let mass = length * args.rod_density;
            props.rod.push(data::RodProp {
                length,
                radius: r,
                stiffness: args.rod_young_mod,
                mass,
            });
            total_rod_mass += mass;
        }
        for (i, &area) in face_area.iter().enumerate() {
            let volume = 2.0 * area * args.contact_ghat;
            let (mu, lambda) = builder::convert_prop(args.area_young_mod, args.area_poiss_rat);
            let mass = if args.include_face_mass || i < mesh.mesh.mesh.shell_face_count {
                args.area_density * volume
            } else {
                0.0
            };
            props.face.push(FaceProp {
                area,
                mass,
                mu,
                lambda,
            });
            if i < mesh.mesh.mesh.shell_face_count {
                total_area_mass += mass;
            }
        }
        for &vol in tet_volumes.iter() {
            let (mu, lambda) = builder::convert_prop(args.volume_young_mod, args.volume_poiss_rat);
            let mass = vol * args.volume_density;
            props.tet.push(TetProp {
                mass,
                volume: vol,
                mu,
                lambda,
            });
            total_vol_mass += mass;
        }
        let mut mass_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}/data/total_mass.out", args.output).as_str())
            .unwrap();
        writeln!(
            mass_file,
            "{} {} {}",
            total_rod_mass, total_area_mass, total_vol_mass
        )
        .unwrap();
        let velocity = scene.get_initial_velocity(&args, mesh.vertex.ncols());
        let time = backend.state.time;
        let dataset = builder::build(
            &args,
            mesh,
            &velocity,
            &props,
            scene.make_constraint(&args, time, mesh),
        );
        let param = builder::make_param(&args);
        backend.run(&args, dataset, param, scene);
    }
}

fn remove_old_files(dirpath: &str, prefix: &str, suffix: &str, keep_number: i32, max_frame: i32) {
    if keep_number > 0 {
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
}

fn remove_files(base: &str, ext: &str, args: &Args) {
    let mut count = args.load + 1;
    loop {
        let path = format!("{}/{}_{}.{}", args.output, base, count, ext);
        if std::path::Path::new(&path).exists() {
            std::fs::remove_file(path).unwrap_or(());
        }
        count += 1;
        if count > args.frames {
            break;
        }
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

fn setup(args: &Args) {
    if args.load == 0 {
        remove_files_in_dir(&args.output).unwrap();
    } else {
        remove_files("vert", "bin", args);
        remove_files("state", "bin.gz", args);
    }

    if !std::path::Path::new(&args.output).exists() {
        std::fs::create_dir_all(&args.output).unwrap_or(());
    }
    let save_and_quit_path = std::path::Path::new(args.output.as_str()).join("save_and_quit");
    if save_and_quit_path.exists() {
        std::fs::remove_file(save_and_quit_path).unwrap_or_else(|err| {
            println!("Failed to delete 'save_and_quit' file: {}", err);
        });
    }

    let pattern = "{d(%Y-%m-%d %H:%M:%S)} [{t}] {m}{n}";
    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build();
    let config = Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(Root::builder().appender("stdout").build(LevelFilter::Info))
        .unwrap();

    log4rs::init_config(config).unwrap();

    info!("{}", std::env::args().collect::<Vec<_>>().join(" "));
    let data_dir = CString::new(format!("{}/data", args.output)).unwrap();
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
