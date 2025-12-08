// File: build.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use std::env;

fn main() {
    #[cfg(not(target_os = "windows"))]
    {
        use std::process::Command;

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

    #[cfg(target_os = "windows")]
    {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let lib_dir = format!("{}\\src\\cpp\\build\\lib", manifest_dir);
        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:rustc-link-lib=dylib=libsimbackend_cuda");

        let cuda_path = env::var("CUDA_PATH")
            .unwrap_or_else(|_| r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8".to_string());
        let cuda_lib_path = format!("{}\\lib\\x64", cuda_path);
        println!("cargo:rustc-link-search=native={}", cuda_lib_path);
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cusparse");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
