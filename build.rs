// File: build.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use std::env;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let num_threads = num_cpus::get();
    println!("cargo:rerun-if-changed=src/cpp");
    println!("cargo:rerun-if-changed=eigsys/eig-hpp");
    let output = Command::new("make")
        .current_dir("src/cpp")
        .arg(format!("OUT_DIR={}", out_dir))
        .arg(format!("-j{}", num_threads))
        .output()
        .expect("Failed to execute make command");

    if !output.status.success() {
        let error_message = String::from_utf8(output.stderr).unwrap();
        println!("make command failed with output: {}", error_message);
        std::process::exit(1);
    }

    let mut dir = std::env::current_dir().expect("Failed to get current directory");
    dir.push(out_dir);
    dir.push("lib");

    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=dylib=simplelog");
    println!("cargo:rustc-link-lib=dylib=simbackend_cuda");
}
