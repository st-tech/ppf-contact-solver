// File: build.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use std::env;

fn main() {
    // The ``emulated`` feature swaps the backing C++ library. The
    // CUDA path links against ``libsimbackend_cuda`` as a shared
    // library (it's huge, includes all the kernels). The emulator
    // path links ``libsimbackend_cpu`` *statically* into the Rust
    // binary, sidestepping Windows DLL search-order quirks the rig
    // hits when launching the solver via a child process. Rust code
    // doesn't need a single ``#[cfg(feature = "emulated")]`` outside
    // this file to honor the switch.
    let emulated = env::var("CARGO_FEATURE_EMULATED").is_ok();
    let (cpp_dir, lib_name) = if emulated {
        ("src/cpp_emul", "simbackend_cpu")
    } else {
        ("src/cpp", "simbackend_cuda")
    };

    if emulated {
        println!("cargo:warning=building with --features emulated; CUDA disabled (libsimbackend_cpu, static)");
    }

    #[cfg(not(target_os = "windows"))]
    {
        use std::process::Command;

        let out_dir = env::var("OUT_DIR").unwrap();
        let num_threads = num_cpus::get();
        println!("cargo:rerun-if-changed={cpp_dir}");
        if !emulated {
            println!("cargo:rerun-if-changed=../../eigsys/eig-hpp");
        }
        let output = Command::new("make")
            .current_dir(cpp_dir)
            .arg(format!("OUT_DIR={out_dir}"))
            .arg(format!("-j{num_threads}"))
            .output()
            .expect("Failed to execute make command");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("make command failed.\nstdout:\n{stdout}\nstderr:\n{stderr}");
            std::process::exit(1);
        }

        let mut dir = std::env::current_dir().expect("Failed to get current directory");
        dir.push(out_dir);
        dir.push("lib");

        println!("cargo:rustc-link-search=native={}", dir.display());
        if emulated {
            // Static archive: the C++ TU is small enough that linking
            // it directly into the Rust binary is cheaper than fixing
            // every host's dlopen / DLL-search quirks.
            println!("cargo:rustc-link-lib=static={lib_name}");
            // Pull in the C++ standard library so libstdc++ symbols
            // (operator new, the std::vector destructor in the synthetic
            // intersection-records buffer) resolve at link time.
            #[cfg(target_os = "macos")]
            println!("cargo:rustc-link-lib=dylib=c++");
            #[cfg(all(unix, not(target_os = "macos")))]
            println!("cargo:rustc-link-lib=dylib=stdc++");
        } else {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
            println!("cargo:rustc-link-lib=dylib=simplelog");
            println!("cargo:rustc-link-lib=dylib={lib_name}");
        }
    }

    #[cfg(target_os = "windows")]
    {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let lib_dir = format!(
            "{manifest_dir}\\{cpp_dir}\\build\\lib",
            cpp_dir = cpp_dir.replace('/', "\\")
        );
        println!("cargo:rustc-link-search=native={lib_dir}");

        if emulated {
            // Static archive (libsimbackend_cpu.lib) produced by
            // build-emul.bat via cl /c + lib.exe.
            println!("cargo:rustc-link-lib=static=lib{lib_name}");
        } else {
            println!("cargo:rustc-link-lib=dylib=lib{lib_name}");
            let cuda_path = env::var("CUDA_PATH")
                .expect("CUDA_PATH environment variable must be set");
            let cuda_lib_path = format!("{cuda_path}\\lib\\x64");
            println!("cargo:rustc-link-search=native={cuda_lib_path}");
            println!("cargo:rustc-link-lib=dylib=cudart");
        }
    }
}
