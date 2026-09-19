// File: build.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Compiles the host-callable C ABI shim (cpp/intersect_ffi.cpp) over the
// shared edge-triangle pierce predicate that lives in the CUDA solver tree
// (../ppf-cts-solver/src/kernels/contact/intersect_core.hpp). This makes the
// device intersection predicate the single source of truth: the Rust
// build-time self-intersection check links this object and calls the same
// geometry the GPU kernels run, instead of a parallel Rust port.
//
// This is a plain host C++ compile (no nvcc, no CUDA): the predicate
// header is dependency-free and STL-free, so it also builds on a no-CUDA
// host (macOS, or any CPU-backend host) with the system C++ compiler.
//
// It also generates the cubin list behind `utils::SUPPORTED_SM` from the
// same `cuda_arch.txt` the two CUDA builds read, so the architectures the
// run-time gate accepts cannot differ from the ones actually linked.
//
// On macOS this also compiles cpp/metal_probe.mm, the tiny Objective-C++
// runtime probe backing the Metal arm of `check_gpu` (src/utils.rs). It is
// intentionally separate from crates/ppf-cts-compute/metal/: check_gpu is a
// pre-flight capability query, not part of the real backend.

use std::path::Path;

/// Architectures the solver ships a cubin for, read from the manifest the
/// CUDA builds link against. It lives with the CUDA target in
/// `ppf-cts-compute`, which is where every backend-specific thing lives; this
/// crate reads it rather than keeping a copy.
///
/// Emitted as a bare array literal that `utils.rs` wraps, so the constant
/// keeps its documentation at the place a reader looks for it.
///
/// Every failure here is fatal on purpose. Falling back to a built-in list
/// would reintroduce exactly the second copy this file exists to remove, and
/// falling back to an empty one would reject every GPU at run time while
/// pointing at the device rather than at the unreadable manifest.
fn generate_supported_sm(cuda_dir: &str) {
    let manifest = Path::new(cuda_dir).join("cuda_arch.txt");
    println!("cargo:rerun-if-changed={}", manifest.display());

    let text = std::fs::read_to_string(&manifest).unwrap_or_else(|e| {
        panic!("cannot read the CUDA architecture manifest {}: {e}", manifest.display())
    });

    let mut cubins: Vec<u32> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }
        let mut field = line.split_whitespace();
        let (Some("cubin"), Some(value)) = (field.next(), field.next()) else {
            continue;
        };
        let sm = value.parse::<u32>().unwrap_or_else(|e| {
            panic!("{}:{}: cubin value {value:?} is not a number: {e}", manifest.display(), n + 1)
        });
        cubins.push(sm);
    }
    if cubins.is_empty() {
        panic!("{}: no 'cubin' lines found", manifest.display());
    }

    let body = cubins.iter().map(u32::to_string).collect::<Vec<_>>().join(", ");
    let out = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"))
        .join("cuda_arch_cubins.rs");
    std::fs::write(&out, format!("[{body}]\n"))
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", out.display()));
}

/// The specific AMD parts the shipped code objects cover, behind
/// `utils::SUPPORTED_GFX`.
///
/// TWO MANIFESTS, AND NEITHER IS A COPY OF THE OTHER. `rocm_arch.txt` says what
/// is BUILT, a list mostly of GENERIC targets; `generic_targets.txt` says which
/// physical parts each generic target covers, read from the pinned toolchain's
/// own documentation. A device reports a specific name (`gfx1100`), never a
/// generic one, so the run-time question can only be answered by expanding the
/// first through the second. Doing it here means the accepted set cannot drift
/// from the linked set, which is the reason `generate_supported_sm` above reads
/// `cuda_arch.txt` rather than keeping its own list.
///
/// A bare `gfxNNNN` in `rocm_arch.txt` is its own coverage and needs no entry in
/// the generic table; a `gfxN-generic` with no entry is fatal, because silently
/// covering nothing would refuse the very hardware it was added for.
///
/// Every failure is fatal for the reason the CUDA generator gives: a built-in
/// fallback reintroduces the second copy, and an empty one rejects every GPU
/// while pointing at the device rather than at the unreadable manifest.
/// Add a THIRD-PARTY include root, so the compiler does not diagnose headers
/// this repository does not own. See the call sites for why that matters.
fn system_include(build: &mut cc::Build, dir: &Path) {
    // MSVC spells this `/external:I` and needs `/external:W0` beside it to mean
    // anything, and both are refused by older cl.exe. The noise this suppresses
    // was measured on the GNU side, so MSVC takes a plain `-I` rather than a
    // flag that could fail the compile outright.
    if build.get_compiler().is_like_msvc() {
        build.include(dir);
    } else {
        build.flag(format!("-isystem{}", dir.display()));
    }
}

fn generate_supported_gfx(rocm_dir: &str) {
    let arch = Path::new(rocm_dir).join("rocm_arch.txt");
    let generic = Path::new(rocm_dir).join("generic_targets.txt");
    println!("cargo:rerun-if-changed={}", arch.display());
    println!("cargo:rerun-if-changed={}", generic.display());

    let read = |path: &Path| {
        std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read the ROCm manifest {}: {e}", path.display()))
    };
    let fields = |text: &str, keyword: &'static str| -> Vec<Vec<String>> {
        text.lines()
            .map(|l| l.split(';').next().unwrap_or("").trim().to_string())
            .filter(|l| !l.is_empty())
            .filter_map(|l| {
                let mut f = l.split_whitespace().map(str::to_string).collect::<Vec<_>>();
                (!f.is_empty() && f[0] == keyword).then(|| {
                    f.remove(0);
                    f
                })
            })
            .collect()
    };

    let mut covers: Vec<(String, Vec<String>)> = Vec::new();
    for row in fields(&read(&generic), "generic") {
        let (name, parts) = row.split_first().unwrap_or_else(|| {
            panic!("{}: a 'generic' line names no target", generic.display())
        });
        if parts.is_empty() {
            panic!("{}: generic target {name} lists no parts", generic.display());
        }
        covers.push((name.clone(), parts.to_vec()));
    }

    let targets = fields(&read(&arch), "target");
    if targets.is_empty() {
        panic!("{}: no 'target' lines found", arch.display());
    }

    let mut parts: Vec<String> = Vec::new();
    for row in &targets {
        let target = &row[0];
        match covers.iter().find(|(name, _)| name == target) {
            Some((_, members)) => parts.extend(members.iter().cloned()),
            // A bare gfxNNNN covers exactly itself. Anything else naming a
            // family we have no table for is refused rather than assumed.
            None if !target.contains("-generic") => parts.push(target.clone()),
            None => panic!(
                "{}: target {target} is generic and {} lists no parts for it. \
                 A generic target with no membership would accept no device, \
                 refusing the hardware it was added to support.",
                arch.display(),
                generic.display()
            ),
        }
    }
    parts.sort();
    parts.dedup();

    let body = parts.iter().map(|p| format!("{p:?}")).collect::<Vec<_>>().join(", ");
    let out = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"))
        .join("rocm_arch_gfx.rs");
    std::fs::write(&out, format!("[{body}]\n"))
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", out.display()));
}

fn main() {
    // The shared predicate header is authored in the solver's C++ tree
    // (sibling crate, source-file dependency only, not a Cargo dependency,
    // so there is no dependency cycle).
    let solver_cpp = "../ppf-cts-solver/src/kernels";
    let header = Path::new(solver_cpp).join("contact/intersect_core.hpp");

    println!("cargo:rerun-if-changed=cpp/intersect_ffi.cpp");
    println!("cargo:rerun-if-changed={}", header.display());

    // The architecture manifest is the CUDA target's, and the CUDA target is
    // ppf-cts-compute's, so it is read from there rather than from the neutral
    // tree above.
    generate_supported_sm("../ppf-cts-compute/cuda");
    // Unconditional, exactly as the CUDA one is: the constant documents what the
    // ROCm build would accept and is read by tests and messages on every host,
    // so gating it on the feature would leave it compiled by nothing.
    generate_supported_gfx("../ppf-cts-compute/rocm");

    // THE HOST BUILD CHECK DOES NOT CONTRACT, ON ANY COMPILER.
    //
    // This instantiates the pierce predicate for double, behind the Rust
    // build-time self-intersection check. It is not the device-parity oracle the
    // kernel shim is (that one contracts on purpose, see
    // ppf-cts-compute/src/build/host.rs); what it owes is the same verdict on
    // every platform that builds a scene. Left to their defaults the compilers
    // disagree: GCC in ISO mode (`cc` passes -std=c++17) and cl.exe since Visual
    // Studio 2022 do not contract, clang does, and on aarch64 FMA is in the base
    // instruction set, so a near-degenerate edge could get one verdict on
    // macOS arm64 and another on x86-64 Linux. No contraction is the one policy
    // all three honor identically, so it is the one pinned.
    let mut isect = cc::Build::new();
    isect
        .cpp(true)
        .std("c++17")
        .file("cpp/intersect_ffi.cpp")
        .include(solver_cpp);
    if isect.get_compiler().is_like_msvc() {
        isect.flag("/fp:precise");
    } else {
        isect.flag("-ffp-contract=off");
    }
    isect.compile("isect_ffi");

    if cfg!(target_os = "macos") {
        println!("cargo:rerun-if-changed=cpp/metal_probe.mm");
        cc::Build::new()
            .file("cpp/metal_probe.mm")
            .flag("-fobjc-arc")
            .compile("metal_probe");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        // IOKit carries the live GPU utilization counter. Metal has no
        // device-wide one, so the "Realtime Statistics" GPU rows come from the
        // IOAccelerator registry entry instead: the same place Activity Monitor
        // reads, and readable without root.
        println!("cargo:rustc-link-lib=framework=IOKit");
    }

    // The ROCm arm's probe, on the same terms as the Metal one above and under
    // the same reasoning: `check_gpu` is a pre-flight query, so it asks the
    // BUNDLED runtime directly rather than loading the real backend or shelling
    // out to `rocminfo`, which is a ROCm package and would put an installation
    // back in front of an end user.
    //
    // GATED ON THE FEATURE, because it includes <hip/hip_runtime.h> and a host
    // without ROCm has no such header. A default build must not acquire a
    // toolchain requirement it does not use.
    //
    // THE INCLUDE PATH IS THE SDK'S OWN, resolved the way the solver's build
    // script resolves it, and a build that asks for this feature without one is
    // told so here rather than at a confusing compile error inside the header.
    // WINDOWS COMPILES IT TOO, and adds no requirement by doing so: a ROCm
    // build's backend DLL already imports the HIP runtime, so a machine that can
    // run this build already has `amdhip64` beside it. macOS is the only host
    // excluded, ROCm having no macOS distribution at all.
    if cfg!(feature = "rocm") && (cfg!(target_os = "linux") || cfg!(target_os = "windows")) {
        println!("cargo:rerun-if-changed=cpp/rocm_probe.cpp");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-env-changed=HIP_PLATFORM");
        let root = std::env::var("ROCM_PATH")
            .or_else(|_| std::env::var("HIP_PATH"))
            .unwrap_or_else(|_| "/opt/rocm".to_string());
        let include = Path::new(&root).join("include");
        if !include.join("hip").join("hip_runtime.h").exists() {
            panic!(
                "--features rocm needs the HIP headers: {} holds no \
                 hip/hip_runtime.h. Set ROCM_PATH (or HIP_PATH) to a ROCm SDK \
                 root; the probe behind check_gpu's ROCm arm is compiled \
                 against them rather than declaring hipDeviceProp_t itself.",
                include.display()
            );
        }
        // THE PLATFORM MACRO IS PASSED EXPLICITLY. hipcc would supply it; `cc`
        // drives the host compiler, so without it hip_runtime.h cannot tell
        // which arm to take and fails on an ambiguous platform.
        let nvidia = std::env::var("HIP_PLATFORM").as_deref() == Ok("nvidia");
        // WHICH PLATFORM THIS BUILD IS FOR, published so the Rust side asks the
        // same question the probe was compiled for. Without it `check_gpu`
        // would gate an NVIDIA staging build on SUPPORTED_GFX and refuse every
        // machine it is meant to run on.
        // Declared as well as emitted, or every compile of this crate warns
        // `unexpected_cfgs` at the two sites that read it.
        println!("cargo::rustc-check-cfg=cfg(rocm_platform_nvidia)");
        if nvidia {
            println!("cargo:rustc-cfg=rocm_platform_nvidia");
        }
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .std("c++17")
            .file("cpp/rocm_probe.cpp");
        // THE SDK'S HEADERS COME IN AS `-isystem`, NOT `-I`, AND THAT IS ABOUT
        // SIGNAL RATHER THAN TASTE. Under `-I` the compiler diagnoses the
        // vendor's own headers: on the NVIDIA staging platform
        // `hip/nvidia_detail/nvidia_hip_runtime_api.h` alone emits a long run of
        // `-Wmissing-field-initializers` from one `CUDA_MEMCPY2D cudaCopy = {0}`,
        // and cargo prefixes every line with `warning: ppf-cts-core@0.1.0:`, so
        // a warning this crate really owns arrives buried in third-party noise
        // it cannot fix. That is not hypothetical: an `unused_doc_comments`
        // warning of ours sat unread in exactly that run. `-isystem` suppresses
        // diagnostics from the headers below it and changes nothing about what
        // is found. This arm is `target_os = "linux"` only, so the spelling is
        // always GCC's or Clang's.
        system_include(&mut build, &include);
        if nvidia {
            // THE STAGING PLATFORM NEEDS `hip/nvidia_detail`, AND NOT EVERY SDK
            // HAS IT. ROCm 7.2.4 carries that directory; TheRock 10.0.0 does
            // not, on either OS. Without this check the compile fails inside
            // hip_runtime.h on a missing include, which reads as a broken SDK
            // rather than as the wrong SDK for this platform.
            if !include.join("hip").join("nvidia_detail").is_dir() {
                panic!(
                    "HIP_PLATFORM=nvidia needs {}, which this SDK does not \
                     carry. ROCm 7.2.4 has it and TheRock 10.0.0 does not, so \
                     point ROCM_PATH at a 7.2.4 SDK or build for the AMD \
                     platform instead.",
                    include.join("hip").join("nvidia_detail").display()
                );
            }
            build.define("__HIP_PLATFORM_NVIDIA__", None);
            // AND THE CUDA HEADERS, because on this platform HIP is a shim:
            // `hip/nvidia_detail/nvidia_hip_runtime.h` includes
            // <cuda_runtime.h> on its first lines. hipcc would add this path;
            // `cc` drives the host compiler, so without it the compile fails
            // inside a HIP header naming a CUDA one, which reads as a broken
            // ROCm install rather than a missing include path.
            let cuda = std::env::var("CUDA_PATH")
                .unwrap_or_else(|_| "/usr/local/cuda".to_string());
            let cuda_include = Path::new(&cuda).join("include");
            if !cuda_include.join("cuda_runtime.h").exists() {
                panic!(
                    "HIP_PLATFORM=nvidia compiles HIP through CUDA, and {} holds \
                     no cuda_runtime.h. Set CUDA_PATH to a CUDA toolkit root.",
                    cuda_include.display()
                );
            }
            println!("cargo:rerun-if-env-changed=CUDA_PATH");
            system_include(&mut build, &cuda_include);
            // THE IMPORT LIBRARY DIRECTORY IS NAMED PER OPERATING SYSTEM. A Linux
            // toolkit keeps libcudart in lib64, and the Windows toolkit keeps
            // cudart.lib in lib\x64, where lib64 does not exist. Naming lib64 on
            // Windows leaves the final link with no cudart.lib to find, and
            // LNK1181 on a CUDA import library reads as a missing CUDA install
            // rather than as this path.
            let cuda_lib = if cfg!(target_os = "windows") {
                Path::new(&cuda).join("lib").join("x64")
            } else {
                Path::new(&cuda).join("lib64")
            };
            println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        } else {
            build.define("__HIP_PLATFORM_AMD__", None);
        }
        build.compile("rocm_probe");
        // The runtime the probe calls into. On the AMD platform that is the
        // library the distribution ships beside the binary; on the NVIDIA
        // staging platform HIP is a shim over CUDA and the symbols come from
        // cudart, which the backend library already brings in.
        println!("cargo:rustc-link-search=native={}", Path::new(&root).join("lib").display());
        if nvidia {
            println!("cargo:rustc-link-lib=dylib=cudart");
        } else {
            println!("cargo:rustc-link-lib=dylib=amdhip64");
        }
    }
}
