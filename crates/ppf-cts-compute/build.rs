// File: crates/ppf-cts-compute/build.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Publishes this crate's library directory through the `links` manifest
// channel, and does nothing else.
//
// WHY A BUILD SCRIPT EXISTS AT ALL WHEN IT COMPILES NOTHING YET. The `links`
// key in Cargo.toml requires one, and `links` is what makes
// DEP_PPFCTSCOMPUTE_LIBDIR reach a dependent's build script. That channel is
// the only way the two link arguments a C++ backend needs can be emitted by the
// crate that will link them: `cargo:rustc-link-arg` from a dependency never
// reaches the dependent binary's link, so the directory travels as metadata and
// the dependent emits the flag. Wiring the channel now, while the directory is
// still empty, is what makes it a mechanism the solver's build script can
// assert on rather than one nobody has run.
//
// The directory is inside OUT_DIR, so nothing is ever written under a watched
// `src/`: two build scripts in this workspace watch a source tree recursively
// and cargo reads a directory in `rerun-if-changed` as "rerun if any descendant
// changes", so one artifact inside costs a two-crate rebuild on every
// subsequent build.

use std::path::PathBuf;

fn main() {
    // Nothing under `src` decides this script's output, so nothing is watched:
    // an unconditional rerun-if-changed on a path cargo would not otherwise
    // consult marks the crate dirty on every build for no reason.
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(
        std::env::var("OUT_DIR").expect("cargo sets OUT_DIR for every build script"),
    );
    let libdir = out_dir.join("lib");
    std::fs::create_dir_all(&libdir)
        .unwrap_or_else(|e| panic!("build.rs: cannot create {}: {e}", libdir.display()));

    // Whether a library implementing `kernels/seam/backend_abi.h` is on
    // this build's link line, which is what decides whether `src/abi.rs`
    // compiles its extern block. Without the feature the block is compiled only
    // by this crate's own test build, where its test double defines the
    // symbols in Rust.
    println!("cargo:rustc-check-cfg=cfg(backend_abi)");
    // TWO WAYS TO TURN IT ON, AND THE SECOND ONE EXISTS BECAUSE THE FIRST
    // CANNOT REACH A DEFAULT BUILD.
    //
    // The FEATURE is the dependent's explicit statement that such a library
    // WILL be on the link line.
    //
    // But a plain `cargo build --release` names no feature and still links one:
    // the CUDA backend is reached ONLY through this ABI now that the neutral
    // Rust driver drives it, so a default build on a CUDA host needs the block.
    // A feature cannot say that, because cargo resolves the dependency graph
    // before any build script runs, and this script runs BEFORE the dependent's
    // own, so it cannot be told either. What it can do is ask the same question
    // the dependent's build script asks, which is the one below: is a CUDA
    // toolkit present, since that is what decides whether `libppfbe_cuda` gets
    // built and linked.
    //
    // `host-only` IS THE OPT-OUT AND IT IS REQUIRED, not a convenience. A CPU
    // build on a CUDA host links no `be_*` library at all, so the probe alone
    // would compile the block into a binary with nothing to resolve it, which
    // is the loud-failure case the feature comment describes. The CPU backend
    // turns `host-only` on, and features being additive is what makes that
    // reliable: nothing else can turn it off again.
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=PPF_CUDA_ROOT");
    let host_only = std::env::var_os("CARGO_FEATURE_HOST_ONLY").is_some();
    // BOTH REAL BACKENDS REACH THE DRIVER THROUGH THIS BLOCK NOW, so the probe
    // asks about both. macOS builds Metal, which links `libppfbe_metal`; a host
    // with a CUDA toolkit builds CUDA, which links `libppfbe_cuda`. The
    // question is the same one either way: will a library exporting `be_*` be
    // on this binary's link line.
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    // A ROCm build links `libppfbe_rocm`, which exports the same `be_*` surface,
    // so the extern block in `src/abi.rs` must be compiled for it too. It is a
    // FEATURE rather than a host probe on purpose: the dev fleet carries a ROCm
    // SDK beside a CUDA toolkit, so probing would turn this on for every build
    // on those machines.
    let rocm = std::env::var_os("CARGO_FEATURE_ROCM").is_some();
    let backend_abi = std::env::var_os("CARGO_FEATURE_BACKEND_ABI").is_some()
        || rocm
        || (!host_only && (macos || cuda_toolkit_present()));
    if backend_abi {
        println!("cargo:rustc-cfg=backend_abi");
    }

    // WINDOWS ONLY, AND FOR THE BUILD-SCRIPT LINK RATHER THAN THE FINAL BINARY.
    //
    // When `backend_abi` is on, `src/abi.rs` compiles the extern block, so
    // `AbiDevice`'s methods reference `be_*`. A dependent that names this crate
    // in `[build-dependencies]` (ppf-cts-solver does, for the transcompiler)
    // links that code into its BUILD SCRIPT executable. On Unix the reference is
    // dropped as unreachable and the build-script link is clean with nothing on
    // the line; MSVC keeps it, so the build-script link fails on undefined
    // `be_*` before the build script ever runs to emit the flags the FINAL
    // binary needs.
    //
    // A dependency's `rustc-link-lib` DOES reach a build script that links it
    // (unlike `rustc-link-arg`, and unlike the final-binary case this crate
    // deliberately leaves to the dependent), so naming the library here resolves
    // those `be_*` at the build-script link. This is possible on Windows and not
    // on Unix for one reason: build-win-native/build.bat builds the backend DLL
    // (and its import lib) BEFORE it runs cargo, so the library already exists
    // when the build script links, whereas the Unix build builds it later inside
    // the dependent's own build script. The path matches where build.bat writes
    // it: <this crate>/cuda/build/lib. The final binary gets the same library
    // from the dependent's build script; a second mention here is harmless.
    //
    // AND WHICH LIBRARY IT IS DEPENDS ON THE BACKEND, which the paragraph above
    // was written before ROCm existed and so did not say. A `--features rocm`
    // build links `libppfbe_rocm` out of this crate's `rocm/build/lib`, and
    // naming the CUDA import library instead fails the BUILD-SCRIPT link with
    // `LNK1181: cannot open input file 'libsimbackend_cuda.lib'` on any machine
    // that has not also built the CUDA backend in the same tree. The message
    // names a CUDA artifact during a ROCm build, which reads as a stray CUDA
    // dependency rather than as this line.
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if backend_abi && windows {
        let manifest = std::env::var("CARGO_MANIFEST_DIR")
            .expect("cargo sets CARGO_MANIFEST_DIR for every build script");
        let (backend_dir, import_lib) = if rocm {
            ("rocm", "libppfbe_rocm")
        } else {
            ("cuda", "libsimbackend_cuda")
        };
        let lib_dir = format!("{manifest}\\{backend_dir}\\build\\lib");
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=dylib={import_lib}");
    }

    // Read back by a direct dependent as DEP_PPFCTSCOMPUTE_LIBDIR.
    println!("cargo:LIBDIR={}", libdir.display());
    // The libraries a dependent must put on its own link line, comma separated,
    // and empty while this crate builds no artifact. A dependent reads the
    // emptiness as "nothing to link", never as "the key is missing", which is
    // why it is published rather than omitted.
    println!("cargo:LIBS=");
}

/// True when this machine can build the real CUDA backend, and therefore when a
/// default build on a non-macOS host will put `libppfbe_cuda` on the link line.
///
/// SPELLED EXACTLY AS `ppf-cts-solver/build.rs` SPELLS IT, on purpose: the two
/// scripts must agree about whether that library exists, and they cannot share
/// code because this crate is that script's build dependency and cannot depend
/// on itself. Windows exposes the toolkit via `CUDA_PATH`; a toolkit outside
/// `/usr/local/cuda` is named by `PPF_CUDA_ROOT`, the variable the CUDA recipe
/// compiles through; otherwise the recipe's own default root
/// `/usr/local/cuda/bin/nvcc` is asked, and `PATH` last. If one spelling
/// changes, the other has to change with it, and the symptom of a disagreement
/// is an undefined `be_open` at link.
///
/// ASKING `PATH` ALONE ANSWERS NO ON A MACHINE THAT BUILDS CUDA, which is why
/// the default root is consulted: `cuda/Makefile` compiles through
/// `$(PPF_CUDA_ROOT)/bin/nvcc` and never through `PATH`, and a provisioned
/// Linux host puts that toolkit's `bin` on the `PATH` of login shells only. The
/// solver's build script carries the measured case.
///
/// A set `PPF_CUDA_ROOT` answers yes without looking inside it. Naming a
/// toolkit root is a request for the CUDA backend, and the solver's build
/// script then fails by name when that root holds no 12.8 `nvcc`, which says
/// more than quietly selecting another backend would.
fn cuda_toolkit_present() -> bool {
    if std::env::var("CUDA_PATH").map_or(false, |p| !p.trim().is_empty()) {
        return true;
    }
    if std::env::var("PPF_CUDA_ROOT").map_or(false, |p| !p.trim().is_empty()) {
        return true;
    }
    // The recipe's own default root. macOS is excluded because a default build
    // there is Metal: a CUDA toolkit under this path would take a Mac off its
    // own backend, and the automatic arm asks about CUDA first.
    if !cfg!(target_os = "macos") && std::path::Path::new("/usr/local/cuda/bin/nvcc").exists() {
        return true;
    }
    std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
