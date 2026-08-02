// File: build.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use std::env;

/// True when this machine can build the real CUDA backend, in which case an
/// emulated build is almost certainly a mistake. Windows exposes the toolkit
/// via ``CUDA_PATH``; Unix has ``nvcc`` on ``PATH``.
fn cuda_toolkit_present() -> bool {
    if env::var("CUDA_PATH").map_or(false, |p| !p.trim().is_empty()) {
        return true;
    }
    std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

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

    // Guard: never build the EMULATED (CPU stub) backend on a machine that can
    // build CUDA. The emulator produces NO real physics yet is written to the
    // same target/release/ path as the real solver/server, so it silently
    // replaces the CUDA binary a live session depends on (this has bitten us:
    // a rig build left an emulated server quietly serving fake results). The
    // test rig, which deliberately runs emulated on a CUDA host, opts in with
    // PPF_ALLOW_EMULATED=1.
    println!("cargo:rerun-if-env-changed=PPF_ALLOW_EMULATED");
    if emulated
        && env::var("PPF_ALLOW_EMULATED").is_err()
        && cuda_toolkit_present()
    {
        panic!(
            "\n\n  Refusing to build the EMULATED (CPU stub) backend: a CUDA \
             toolkit (nvcc / CUDA_PATH) is present on this machine.\n  The \
             emulated binary produces no real physics and overwrites the real \
             CUDA solver/server at target/release/, so a connected session \
             would silently get fake results.\n  Build the real backend \
             instead:  cargo build --release   (default features, real CUDA).\n  \
             The emulated backend builds with  cargo build-emul   on macOS / a \
             no-nvcc host.\n  If you truly need the emulator here (e.g. the \
             test rig on a CUDA host), set  PPF_ALLOW_EMULATED=1.\n\n"
        );
    }

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
        if emulated {
            // cpp_emul/main.cpp includes shared headers from ../cpp
            // (data.hpp, etc.). Without watching that directory, cargo
            // skips re-running make when a shared struct like FixPair or
            // ParamSet changes, leaving a stale static lib whose layout
            // disagrees with the Rust repr(C) structs -> SIGBUS at run.
            println!("cargo:rerun-if-changed=src/cpp");
        } else {
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

        if !emulated {
            // The device image just produced must run entirely in single
            // precision. Read that off the compiled SASS, not the source.
            fp64_guard::check(&dir.join(format!("lib{lib_name}.so")));
        }

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

// Guard against double precision reaching the GPU.
//
// The solver is single precision on the device, and that is a correctness
// property rather than a preference: the barrier stiffness, the conservative
// advance and the CSR assembly are all reasoned about in float. Source review
// cannot enforce it, because a double can arrive without the word appearing
// anywhere in the source. The library trig and exponential functions are the
// standing example: their slow-path argument reduction is double precision, so
// a kernel that merely calls sinf emits I2F.F64 and DMUL.
//
// So the check reads the SASS of the device image that was just built, which is
// what actually runs. It reports per kernel, and fails on any kernel not on the
// list of sites that already contain FP64. That list is the point: it does not
// bless those sites, it pins them, so the count cannot quietly grow and each
// entry stays visible until it is dealt with.
mod fp64_guard {
    use std::path::Path;
    use std::process::Command;

    // SASS mnemonics that execute in double precision, including the
    // conversions, which is how an integer widened to float arrives here.
    const FP64_OPS: [&str; 10] = [
        "DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "DDIV", "F2F.F64", "I2F.F64",
        "F2I.F64", "MUFU.RCP64H",
    ];

    // No kernel may contain FP64. The list is empty and is meant to stay that
    // way: the device runs single precision, and the transcendentals that used
    // to break that are replaced in float_math.hpp. It exists as a list rather
    // than as a bare zero so that a site which genuinely cannot avoid double
    // has somewhere to be recorded and argued about, not so that one can be
    // added to quiet a failure.
    const KNOWN: [&str; 0] = [];

    // Whether `line` contains `op` as a whole mnemonic rather than as part of a
    // longer one. Without this, DMNMX matches inside VIADDMNMX.U32, a 32-bit
    // integer add-min-max, and the guard reports double precision that is not
    // there. A mnemonic carries dot-separated modifiers, so a dot after the
    // match is allowed, while an alphanumeric on either side means the match
    // landed inside a different instruction.
    fn contains_mnemonic(line: &str, op: &str) -> bool {
        let b = line.as_bytes();
        let mut from = 0;
        while let Some(rel) = line[from..].find(op) {
            let s = from + rel;
            let e = s + op.len();
            let before_ok = s == 0
                || !(b[s - 1].is_ascii_alphanumeric() || b[s - 1] == b'_'
                     || b[s - 1] == b'.');
            let after_ok =
                e >= b.len() || !(b[e].is_ascii_alphanumeric() || b[e] == b'_');
            if before_ok && after_ok {
                return true;
            }
            from = s + 1;
        }
        false
    }

    pub fn check(image: &Path) {
        if std::env::var("PPF_ALLOW_FP64").is_ok() {
            println!("cargo:warning=PPF_ALLOW_FP64 set: skipping the device \
                      single-precision check");
            return;
        }
        if !image.exists() {
            return;
        }
        let Ok(out) = Command::new("cuobjdump").arg("--dump-sass").arg(image).output() else {
            // A CUDA install without cuobjdump cannot answer the question. Say
            // so, rather than passing silently on no evidence.
            println!("cargo:warning=cuobjdump not found: the device \
                      single-precision check did not run");
            return;
        };
        let sass = String::from_utf8_lossy(&out.stdout);

        // An empty or unparsable dump means the check learned nothing. Treating
        // that as a pass is the trap this exists to avoid, so it is reported.
        let instr = sass.lines().filter(|l| l.trim_start().starts_with("/*")).count();
        if instr == 0 {
            println!("cargo:warning=the device image produced no readable SASS: \
                      the single-precision check did not run");
            return;
        }

        // Report what was actually read, so a disagreement between this and a
        // manual inspection names the image rather than needing to be guessed
        // at. An instruction count is the evidence that the dump parsed.
        println!(
            "cargo:warning=device single-precision check: {} instruction(s) in {}",
            instr,
            image.display()
        );

        let mut current = String::new();
        let mut offenders: Vec<(String, usize)> = Vec::new();
        for line in sass.lines() {
            if let Some(i) = line.find("Function : ") {
                current = line[i + "Function : ".len()..].trim().to_string();
            } else if FP64_OPS.iter().any(|op| contains_mnemonic(line, op)) {
                if KNOWN.iter().any(|k| current.contains(k)) {
                    continue;
                }
                match offenders.iter_mut().find(|(f, _)| *f == current) {
                    Some((_, n)) => *n += 1,
                    None => offenders.push((current.clone(), 1)),
                }
            }
        }
        if offenders.is_empty() {
            return;
        }
        for (f, n) in &offenders {
            println!("cargo:warning=FP64 in device code: {n} instruction(s) in {f}");
        }
        panic!(
            "double precision reached the GPU in {} kernel(s) not previously \
             carrying it. The solver runs single precision on the device. A \
             common cause is a library call whose argument reduction is double \
             (sinf, cosf, expf, logf): use the float-only intrinsic, or reduce \
             the argument in float first. Another is an integer widened to \
             float, which CUDA routes through double; narrow it first. Set \
             PPF_ALLOW_FP64=1 to build anyway while investigating.",
            offenders.len()
        );
    }
}
