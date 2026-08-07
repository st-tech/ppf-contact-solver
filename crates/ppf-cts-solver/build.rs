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

/// The CUDA release every host in this project builds with.
///
/// It is a hard requirement rather than a preference: `main.cu` reads
/// `cudaDeviceProp::kernelExecTimeoutEnabled`, which CUDA 13 removed, so a
/// newer toolkit does not merely behave differently, it fails to compile. This
/// constant is the only place that dependency is written down.
const REQUIRED_CUDA_RELEASE: &str = "release 12.8";

/// Fail before `make` runs if the toolkit is not [`REQUIRED_CUDA_RELEASE`].
///
/// This is the one chokepoint every path that compiles the solver crosses, so
/// it is where the check is worth having: the per-example CI workflows, the dev
/// hosts, and any workflow added later that forgets to pin all arrive here. The
/// alternative is what prompted it, three `cudaDeviceProp` errors forty lines
/// into `main.cu` on a runner whose AMI had silently moved its default to 13.2.
///
/// It interrogates `src/cpp/Makefile`'s own `NVCC`, which is the ABSOLUTE path
/// `/usr/local/cuda/bin/nvcc`. Asking `PATH` instead would reproduce the exact
/// blind spot this exists to close, since the two are separate selectors and it
/// is the symlink that decides what actually compiles.
fn require_cuda_12_8() {
    use std::process::Command;

    let makefile = std::fs::read_to_string("src/cpp/Makefile")
        .expect("build.rs: cannot read src/cpp/Makefile to locate nvcc");
    let nvcc = makefile
        .lines()
        .find_map(|l| l.strip_prefix("NVCC="))
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| "nvcc".to_string());

    let out = match Command::new(&nvcc).arg("--version").output() {
        Ok(o) if o.status.success() => o,
        Ok(o) => panic!(
            "build.rs: `{nvcc} --version` failed with status {}.\n\
             The solver needs CUDA {REQUIRED_CUDA_RELEASE}.",
            o.status
        ),
        Err(e) => panic!(
            "build.rs: cannot run `{nvcc} --version`: {e}.\n\
             The solver compiles through that exact path, so it must be a CUDA \
             {REQUIRED_CUDA_RELEASE} toolkit."
        ),
    };

    let banner = String::from_utf8_lossy(&out.stdout);
    if !banner.contains(REQUIRED_CUDA_RELEASE) {
        let found = banner
            .lines()
            .find(|l| l.contains("release "))
            .unwrap_or("unknown")
            .trim();
        panic!(
            "build.rs: wrong CUDA toolkit.\n\
             \x20 required : {REQUIRED_CUDA_RELEASE}\n\
             \x20 found    : {found}\n\
             \x20 via      : {nvcc}\n\
             `main.cu` reads cudaDeviceProp::kernelExecTimeoutEnabled, which \
             CUDA 13 removed, so this would fail to compile.\n\
             On an AWS Deep Learning AMI several toolkits are installed at once \
             and /usr/local/cuda points at the newest by default; repoint that \
             symlink at /usr/local/cuda-12.8 (see the CUDA pin in \
             .github/workflows/template/aws-template.yml)."
        );
    }
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
        if !emulated {
            require_cuda_12_8();
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

    // The architectures whose SASS the image carries, as the image itself
    // reports them. Reading the list off the artifact rather than repeating the
    // Makefile's -gencode list means the two cannot drift: an arch added there
    // is checked here without a second edit, and one dropped is not looked for.
    fn image_arches(image: &Path) -> Vec<String> {
        let Ok(out) = Command::new("cuobjdump").arg("--list-elf").arg(image).output() else {
            return Vec::new();
        };
        let listing = String::from_utf8_lossy(&out.stdout);
        let mut archs: Vec<String> = Vec::new();
        for line in listing.lines() {
            let mut from = 0;
            while let Some(rel) = line[from..].find("sm_") {
                let s = from + rel;
                let digits: String =
                    line[s + 3..].chars().take_while(|c| c.is_ascii_digit()).collect();
                from = s + 3;
                if digits.is_empty() {
                    continue;
                }
                let arch = format!("sm_{digits}");
                if !archs.contains(&arch) {
                    archs.push(arch);
                }
            }
        }
        archs
    }

    // Whether `line` is a disassembled instruction, which cuobjdump prefixes
    // with its address as `/*0a30*/`. It also emits a continuation line holding
    // the raw encoding, `/* 0x000e220000000800 */`, and that one must not count:
    // a bare "starts with /*" tallies both and reports exactly twice the
    // instructions the image holds, which is worse than a useless number,
    // because it disagrees with what a reader counting the same dump by hand
    // gets (`cuobjdump --dump-sass <image> | grep -cE '^\s+/\*[0-9a-f]+\*/'`)
    // and so reads as a coverage gap that is not there. Requiring hex between
    // the delimiters separates them: the encoding line opens with a space and
    // an 0x prefix.
    fn is_instruction(line: &str) -> bool {
        let t = line.trim_start();
        let Some(rest) = t.strip_prefix("/*") else {
            return false;
        };
        let Some(end) = rest.find("*/") else {
            return false;
        };
        !rest[..end].is_empty() && rest[..end].bytes().all(|b| b.is_ascii_hexdigit())
    }

    // Scan one dump, accumulating per-kernel FP64 counts into `offenders` and
    // returning the instruction count that proves the dump parsed.
    fn scan(sass: &str, offenders: &mut Vec<(String, usize)>) -> usize {
        let mut current = String::new();
        let mut instr = 0usize;
        for line in sass.lines() {
            if is_instruction(line) {
                instr += 1;
            }
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
        instr
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

        // Dump one architecture per process, all at once. A single --dump-sass
        // over the whole image walks the cubins one after another and takes
        // 18.0 s on a release build (measured, L40S / nvcc 12.8), which is a
        // third of the parallelized link it follows; requesting the same
        // architectures individually and concurrently takes 4.1 s. This reads
        // the identical SASS, not a sample of it: the per-arch instruction
        // counts sum to exactly the 1,132,816 the single dump reports.
        //
        // -arch is a filter, and asking for one the image does not carry exits
        // 0 with an EMPTY dump (verified with sm_70), so a mis-enumeration here
        // would silently check nothing. Every arch is therefore required to
        // yield instructions, and anything short of that falls back to the one
        // whole-image dump below rather than reporting a pass it did not earn.
        let archs = image_arches(image);
        let mut offenders: Vec<(String, usize)> = Vec::new();
        let mut instr = 0usize;
        if !archs.is_empty() {
            let handles: Vec<_> = archs
                .iter()
                .map(|arch| {
                    let (arch, image) = (arch.clone(), image.to_path_buf());
                    std::thread::spawn(move || {
                        let out = Command::new("cuobjdump")
                            .arg("--dump-sass")
                            .arg("-arch")
                            .arg(&arch)
                            .arg(&image)
                            .output();
                        (arch, out)
                    })
                })
                .collect();
            let mut complete = true;
            for h in handles {
                let Ok((arch, Ok(out))) = h.join() else {
                    complete = false;
                    continue;
                };
                let n = scan(&String::from_utf8_lossy(&out.stdout), &mut offenders);
                if n == 0 {
                    println!(
                        "cargo:warning={arch} is in the device image but dumped no \
                         SASS: falling back to a whole-image single-precision check"
                    );
                    complete = false;
                }
                instr += n;
            }
            if !complete {
                offenders.clear();
                instr = 0;
            }
        }

        if instr == 0 {
            let Ok(out) = Command::new("cuobjdump").arg("--dump-sass").arg(image).output() else {
                // A CUDA install without cuobjdump cannot answer the question.
                // Say so, rather than passing silently on no evidence.
                println!("cargo:warning=cuobjdump not found: the device \
                          single-precision check did not run");
                return;
            };
            offenders.clear();
            instr = scan(&String::from_utf8_lossy(&out.stdout), &mut offenders);
        }

        // An empty or unparsable dump means the check learned nothing. Treating
        // that as a pass is the trap this exists to avoid, so it is reported.
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
