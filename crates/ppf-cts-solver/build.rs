// File: build.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use std::env;
use std::path::Path;
use std::process::Command;

// The build RECIPE for every target. This script names its own sources and its
// own output directory and decides nothing else about how they are compiled.
use ppf_cts_compute as compute;

/// Which C++ backend this build links. The whole backend swap is this one
/// value: it picks a source directory and a library name and nothing else in
/// the Rust tree knows a backend exists. Keep it that way. If a backend starts
/// needing `#[cfg]` outside this file, the seam is in the wrong place.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Cuda,
    Metal,
    /// The ROCm backend, crates/ppf-cts-compute/rocm. A real device backend
    /// reached through the same C ABI CUDA and Metal are, so nothing above
    /// `Device` changes for it. NEVER selected automatically: the dev fleet
    /// carries a ROCm SDK beside a CUDA toolkit, so a host probe that chose it
    /// would move every default build off the reference backend without saying
    /// so.
    Rocm,
    /// The CPU backend, `src/driver`. Written in Rust, so there is no C++ backend
    /// library to build and nothing to link; what `build.rs` compiles for it is
    /// the ranged kernel entry points in `entrypoints/`, which include the shared
    /// kernel bodies. NEVER selected automatically: it is real but roughly 30x
    /// slower than a GPU, and it is written to the same `target/release/` path.
    Cpu,
}

impl Backend {
    /// The C++ backend directory `make` is run in, relative to this crate's
    /// manifest. `None` for a backend implemented in Rust, which is the whole
    /// of what makes it different here.
    ///
    /// Both leave this crate, and that is the architecture rather than a
    /// layout preference: every backend-specific thing lives in
    /// `ppf-cts-compute`. What stays behind is the
    /// neutral kernel tree the recipes there compile, which this crate passes
    /// as `KERNEL_ROOT` so neither recipe hardcodes a path back into here.
    fn cpp_dir(self) -> Option<&'static str> {
        match self {
            Backend::Cuda => Some("../ppf-cts-compute/cuda"),
            Backend::Metal => Some("../ppf-cts-compute/metal"),
            Backend::Rocm => Some("../ppf-cts-compute/rocm"),
            Backend::Cpu => None,
        }
    }

    /// The BACKEND LOGIC tree the recipe compiles beside its own sources,
    /// relative to this crate's manifest. `None` where nothing has been split
    /// out yet.
    ///
    /// It is a third crate because it fits neither of the other two: it is
    /// backend-specific, so it cannot come back here (no backend name appears
    /// anywhere in this crate), and it is none of the five verbs
    /// `ppf-cts-compute` is held to, so it cannot stay there. The crate exists
    /// to be deleted, one file at a time, as each becomes neutral Rust in
    /// `src/driver` or a generated entry point.
    fn logic_dir(self) -> Option<&'static str> {
        match self {
            // GONE. The CUDA orchestrator was this backend's logic tree until
            // the neutral driver took over driving it; what a CUDA build
            // compiles now is `ppf-cts-compute/cuda` plus one object per
            // generated entry point, and none of it is simulation.
            Backend::Cuda => None,
            // NEVER HAD ONE. This backend was added after the orchestrators were
            // deleted, so it has only ever been mechanism plus generated entry
            // points, which is what rule (1b) asks of a backend library.
            Backend::Rocm => None,
            // GONE, on the same terms as CUDA's: the neutral driver drives
            // this backend and the orchestrator that was its logic tree is
            // deleted. What a Metal build compiles is `ppf-cts-compute/metal`
            // plus one object per generated entry point, and none of it is
            // simulation.
            Backend::Metal => None,
            // The CPU backend has no C++ tree of its own to split: what it
            // compiles is the neutral kernel bodies through `entrypoints/`.
            Backend::Cpu => None,
        }
    }

    fn lib_name(self) -> Option<&'static str> {
        match self {
            Backend::Cuda => Some("simbackend_cuda"),
            Backend::Metal => Some("simbackend_metal"),
            Backend::Rocm => Some("ppfbe_rocm"),
            Backend::Cpu => None,
        }
    }

    /// Whether this build links a [`ppf_cts_compute::Device`] implementation
    /// the neutral Rust driver in `src/driver` drives, and therefore whether
    /// that module is compiled at all.
    ///
    /// THE DRIVER IS NOT A BACKEND, and this method is where that stops being a
    /// claim and becomes a build fact. It holds the Newton loop, the linear
    /// solve, the assembly order and every dispatch decision, written once for
    /// every target. What
    /// decides whether it is compiled is not which target this is but whether
    /// anything below it implements the seam, and that question is answered
    /// here, in the one file that is allowed to know a target's name.
    ///
    /// It answers false for a C++ backend for a reason that fails SILENTLY if
    /// it is got wrong. `src/driver/mod.rs` defines the whole `advance()`
    /// surface with `#[no_mangle]`, and each C++ backend library exports the
    /// same symbols; building both is neither a compile error nor a link error,
    /// because the binary's own definitions win at dynamic link and the library
    /// is simply never called. Every value gate then stays green over a path
    /// nothing takes. Section 12.4 A of the plan states it; this is where it is
    /// enforced.
    fn links_neutral_driver(self, abi_backend: bool) -> bool {
        match self {
            // CUDA ANSWERS YES, FULL STOP. The driver drives it: it refuses no
            // capability the orchestrator implemented, and a CUDA build links
            // `libppfbe_cuda` and nothing else. `abi_backend` is no longer
            // consulted here because there is no configuration in which a CUDA
            // build wants the C++ `advance()` surface instead.
            Backend::Cuda => {
                let _ = abi_backend;
                true
            }
            // METAL ANSWERS YES, FULL STOP, on the same terms CUDA does. The
            // driver runs all 23 acceptance scenes on that backend and all 27
            // of its fixtures, so the C++ orchestrator it would replace is
            // deleted and there is no configuration in which a Metal build
            // wants a `advance()` surface other than this one.
            Backend::Metal => {
                let _ = abi_backend;
                true
            }
            Backend::Cpu => true,
            // YES, FULL STOP, on the same terms as CUDA. This backend was added
            // after the orchestrators were deleted, so the neutral driver is the
            // only thing that has ever driven it and there is no C++ `advance()`
            // surface for `abi_backend` to select between.
            Backend::Rocm => true,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Backend::Cuda => "CUDA",
            Backend::Rocm => "ROCm (HIP)",
            // The qualifier remains while Phase 4 is incomplete because Metal
            // is auto-selected on macOS. The current slice performs a real
            // CCD-filtered Newton position commit, but the full production
            // state machine is still gated.
            Backend::Metal => "Metal (PHASE 4 INTEGRATION SLICE, GATED)",
            Backend::Cpu => {
                "CPU (Rust; solids, shell membranes and bending, rod stretch and bending, \
                 both strain limiters, cross-stitches, self-contact, analytic colliders, the \
                 static collision mesh, plasticity, per-face pressure, applied torque, \
                 projected translation and rotation locks, granular SAND, PDRD rigid \
                 bodies with or without a projected lock, and multilevel additive \
                 Schwarz; no scene class is refused, and the two material defects \
                 that remain are properties no backend can assemble)"
            }
        }
    }
}

/// True when this machine can build the ROCm backend.
///
/// SPELLED THE SAME WAY IN `ppf-cts-compute/build.rs`, for the reason that
/// script's `cuda_toolkit_present` states: the two must agree about whether a
/// `be_*` library will be on the link line, and they cannot share code because
/// that crate is this script's build dependency. A disagreement surfaces as an
/// undefined `be_open` at link.
///
/// BOTH SPELLINGS OF THE DRIVER ARE TESTED, because Windows names it
/// `hipcc.exe` and a check for the extensionless name alone answers NO on every
/// correct Windows ROCm installation. The failure would not name the extension:
/// `--features rocm` would report that no ROCm toolchain was found on a machine
/// whose SDK is right there, and the obvious next move is to go looking for a
/// broken SDK.
fn rocm_toolkit_present() -> bool {
    // The ROCm recipe compiles through whichever SDK these name, for whichever
    // platform HIP_PLATFORM selects, so a change to any of them has to re-run
    // this script: a cached build would otherwise keep a library for the other
    // SDK or the other platform.
    for key in ["ROCM_PATH", "HIP_PATH", "HIP_PLATFORM"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    for key in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(root) = env::var(key) {
            let bin = Path::new(root.trim()).join("bin");
            if !root.trim().is_empty()
                && (bin.join("hipcc").exists() || bin.join("hipcc.exe").exists())
            {
                return true;
            }
        }
    }
    std::process::Command::new("hipcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// True when this machine can build the real CUDA backend.
///
/// THE QUESTION IS ABOUT THE PATH THE RECIPE COMPILES THROUGH, NOT ABOUT
/// ``PATH``. `ppf-cts-compute/cuda/Makefile` compiles with
/// ``$(PPF_CUDA_ROOT)/bin/nvcc``, which is ``/usr/local/cuda/bin/nvcc`` unless
/// that variable names another root, and it consults ``PATH`` for nothing. A
/// provisioned Linux host carries the toolkit at that default root and puts its
/// ``bin`` on the ``PATH`` of login shells only, so a non-interactive step (a
/// GitHub Actions `run:` line, a plain `ssh host cmd`) finds no ``nvcc`` on
/// ``PATH`` while the recipe beside it compiles fine. Asking ``PATH`` alone
/// therefore answers NO on a machine that builds CUDA, and the automatic arm of
/// [`select_backend`] refuses the build for want of a toolkit that is installed.
/// Measured on the self-hosted CI runner, whose toolkit is at that default root:
/// `cuobjdump` is not on ``PATH`` there either, which is the same fact reported
/// by the arch and FP64 guards as a skipped check.
///
/// Windows exposes the toolkit via ``CUDA_PATH``. ``PATH`` is still asked, last,
/// so a toolkit reachable only that way still selects CUDA and
/// [`require_cuda_12_8`] then names the path the recipe uses, which is the
/// actionable half.
///
/// Spelled exactly as `ppf-cts-compute/build.rs` spells it, which says why the
/// two must agree.
///
/// A set ``PPF_CUDA_ROOT`` answers yes without looking inside it: naming a
/// toolkit root asks for CUDA, and [`require_cuda_12_8`] then fails by name if
/// that root holds no 12.8 ``nvcc``, rather than a different backend being
/// selected quietly.
fn cuda_toolkit_present() -> bool {
    if env::var("CUDA_PATH").map_or(false, |p| !p.trim().is_empty()) {
        return true;
    }
    if env::var("PPF_CUDA_ROOT").map_or(false, |p| !p.trim().is_empty()) {
        return true;
    }
    // The recipe's own default root. macOS is excluded because a default build
    // there is Metal: a CUDA toolkit under this path would take a Mac off its
    // own backend, and the automatic arm asks about CUDA first.
    if !cfg!(target_os = "macos") && Path::new("/usr/local/cuda/bin/nvcc").exists() {
        return true;
    }
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `None` when this machine can build the real Metal backend; otherwise the
/// reason it cannot, phrased so the hard error below can quote it. Metal needs
/// macOS and a macOS SDK carrying `Metal.framework`, which arrives with the
/// Xcode command line tools.
fn metal_unavailable_reason() -> Option<String> {
    if !cfg!(target_os = "macos") {
        return Some("Metal needs macOS, and this is not a macOS host".to_string());
    }
    let sdk = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output();
    match sdk {
        Ok(out) if out.status.success() => {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let framework = format!("{path}/System/Library/Frameworks/Metal.framework");
            if Path::new(&framework).exists() {
                None
            } else {
                Some(format!(
                    "the macOS SDK at {path} carries no Metal.framework"
                ))
            }
        }
        _ => Some(
            "xcrun could not report a macOS SDK path; install the Xcode \
             command line tools with  xcode-select --install"
                .to_string(),
        ),
    }
}

/// The CUDA release every host in this project builds with.
///
/// It is a hard requirement rather than a preference: the CUDA backend's
/// `be_open` preflight reads `cudaDeviceProp::kernelExecTimeoutEnabled`, which
/// CUDA 13 removed, so that file does not merely behave differently on a newer
/// toolkit, it fails to compile. Whether anything ELSE here breaks on 13 is
/// unmeasured, since every host and every CI leg builds with 12.8. This
/// constant is the only place the pin is written down.
const REQUIRED_CUDA_RELEASE: &str = "release 12.8";

/// Fail before `make` runs if the toolkit is not [`REQUIRED_CUDA_RELEASE`].
///
/// This is the one chokepoint every path that compiles the solver crosses, so
/// it is where the check is worth having: the per-example CI workflows, the dev
/// hosts, and any workflow added later that forgets to pin all arrive here. The
/// alternative is what prompted it, three `cudaDeviceProp` errors on a runner
/// whose AMI had silently moved its default to 13.2.
///
/// It interrogates the nvcc the CUDA recipe itself compiles with, by asking the
/// recipe through its `print-nvcc` target. That is `$(PPF_CUDA_ROOT)/bin/nvcc`,
/// the absolute path `/usr/local/cuda/bin/nvcc` unless the variable is exported.
/// Asking `PATH` instead would reproduce the exact blind spot this exists to
/// close, since the two are separate selectors and it is the recipe's path that
/// decides what actually compiles. Asking the recipe rather than reading its
/// text is what keeps an override from being checked in one place and ignored
/// in the other.
fn require_cuda_12_8(cpp_dir: &str, kernel_root: &Path) {
    use std::process::Command;

    println!("cargo:rerun-if-env-changed=PPF_CUDA_ROOT");
    let query = Command::new("make")
        .current_dir(cpp_dir)
        .arg("-s")
        .arg(format!("KERNEL_ROOT={}", kernel_root.display()))
        .arg("print-nvcc")
        .output()
        .unwrap_or_else(|e| panic!("build.rs: cannot run make in {cpp_dir} to locate nvcc: {e}"));
    if !query.status.success() {
        panic!(
            "build.rs: `make print-nvcc` in {cpp_dir} failed with status {}.\n{}",
            query.status,
            String::from_utf8_lossy(&query.stderr)
        );
    }
    let nvcc = String::from_utf8_lossy(&query.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .last()
        .map(str::to_string)
        .unwrap_or_else(|| panic!("build.rs: `make print-nvcc` in {cpp_dir} printed nothing"));

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
             The CUDA backend reads cudaDeviceProp::kernelExecTimeoutEnabled, which \
             CUDA 13 removed, so this would fail to compile.\n\
             On an AWS Deep Learning AMI several toolkits are installed at once \
             and /usr/local/cuda points at the newest by default; repoint that \
             symlink at /usr/local/cuda-12.8 (see the CUDA pin in \
             .github/workflows/template/aws-template.yml), or export \
             PPF_CUDA_ROOT naming a 12.8 toolkit root, which is how \
             build-linux-native/build.sh selects the toolkit it provisions."
        );
    }
}

/// Picks the backend. Plain `cargo build --release` (no features) selects the
/// REAL backend for the host, and hard-errors when the host has none. The
/// explicit `--features cuda` / `--features metal` overrides exist for forcing
/// one on a host that could serve either.
/// Record WHICH BACKEND these artifacts are, and refuse to overwrite another.
///
/// **THE HAZARD IS SILENT AND THIS FILE ALREADY NAMES IT.** Every backend links
/// the same `[[bin]]` name, because launch scripts, the Windows batch files and
/// the CI runners all invoke `ppf-contact-solver` by that name and Cargo cannot
/// vary a bin name by feature. So `--features cpu` after a CUDA build used to
/// overwrite `target/<profile>/ppf-contact-solver` in place, and what that costs
/// is recorded here: "a live session then gets fake results from a
/// binary that looks right". A CPU build is roughly 30x slower and incomplete,
/// so the swap is not merely surprising, it invalidates whatever ran next.
///
/// TWO THINGS, AND THEY ANSWER DIFFERENT QUESTIONS. `PPF_BACKEND` is compiled
/// INTO the binary, so an artifact can be asked what it targets long after the
/// build that made it: `ppf-contact-solver --backend` prints it and `strings`
/// finds it. The marker records what a given `target/<profile>` HOLDS, so the
/// next build can tell whether it is about to replace a different backend.
///
/// THE REFUSAL NAMES THE FIX RATHER THAN JUST THE PROBLEM. Two backends can be
/// held at once, in separate target directories, and one environment variable
/// drives both the build and the frontend that loads the cdylib out of it.
fn record_backend(backend: Backend) {
    let name = match backend {
        Backend::Cuda => "cuda",
        Backend::Metal => "metal",
        Backend::Cpu => "cpu",
        Backend::Rocm => "rocm",
    };
    println!("cargo:rustc-env=PPF_BACKEND={name}");
    // OUT_DIR is `<target>/<profile>/build/<pkg>-<hash>/out`, so the profile
    // directory the artifacts land in is three ancestors up.
    let out = env::var("OUT_DIR").expect("OUT_DIR");
    let Some(profile_dir) = Path::new(&out).ancestors().nth(3) else {
        return;
    };
    let marker = profile_dir.join(".ppf-backend");
    if let Ok(previous) = std::fs::read_to_string(&marker) {
        let previous = previous.trim().to_string();
        if !previous.is_empty() && previous != name {
            panic!(
                "\n\n  {} already holds a {previous} build, and this one is \
                 {name}.\n\n  Every backend links the same executable name, so \
                 finishing here would overwrite the {previous} binary in place \
                 and leave nothing to say it had changed. A {name} build is not \
                 a drop-in for a {previous} one.\n\n  Hold both at once by \
                 giving this backend its own target directory, which the \
                 frontend reads too, so the cdylib is found beside its own \
                 binary:\n\n      CARGO_TARGET_DIR=target/{name} cargo build \
                 --release --features {name}\n      CARGO_TARGET_DIR=target/{name} \
                 python3 examples/run_suite.py --backend {name} --fast-check\n\n  \
                 Or replace this one deliberately with `cargo clean --release`.\n\n",
                profile_dir.display()
            );
        }
    }
    let _ = std::fs::write(&marker, name);
}

fn select_backend() -> Backend {
    let force_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let force_metal = env::var("CARGO_FEATURE_METAL").is_ok();
    let force_cpu = env::var("CARGO_FEATURE_CPU").is_ok();
    let force_rocm = env::var("CARGO_FEATURE_ROCM").is_ok();

    let forced: Vec<&str> = [
        (force_cuda, "cuda"),
        (force_metal, "metal"),
        (force_cpu, "cpu"),
        (force_rocm, "rocm"),
    ]
    .iter()
    .filter(|(on, _)| *on)
    .map(|(_, name)| *name)
    .collect();
    if forced.len() > 1 {
        panic!(
            "\n\n  Backend features {} were requested together. Exactly one \
             backend is linked into the solver, so asking for several is \
             ambiguous rather than additive.\n\n",
            forced.join(" and ")
        );
    }

    if force_cpu {
        guard_cpu();
        return Backend::Cpu;
    }
    // ROCm is EXPLICIT ONLY and is never reached by the automatic arm below.
    // The dev fleet carries a ROCm SDK beside a CUDA toolkit, so a host probe
    // that selected it would move every default build off the reference backend
    // with nothing saying so.
    if force_rocm {
        if !rocm_toolkit_present() {
            panic!(
                "\n\n  --features rocm was requested but no ROCm toolchain was \
                 found: neither ROCM_PATH nor HIP_PATH names a directory with \
                 bin/hipcc, and hipcc is not on PATH.\n\n"
            );
        }
        return Backend::Rocm;
    }
    if force_cuda {
        if !cuda_toolkit_present() {
            panic!(
                "\n\n  --features cuda was requested but no CUDA toolkit was \
                 found: nothing at /usr/local/cuda/bin/nvcc, no CUDA_PATH or \
                 PPF_CUDA_ROOT naming a toolkit root, and no nvcc on PATH.\n\n"
            );
        }
        return Backend::Cuda;
    }
    if force_metal {
        if let Some(reason) = metal_unavailable_reason() {
            panic!("\n\n  --features metal was requested but {reason}.\n\n");
        }
        return Backend::Metal;
    }

    // Automatic: the real backend this host can build.
    if cuda_toolkit_present() {
        return Backend::Cuda;
    }
    let metal_reason = metal_unavailable_reason();
    if metal_reason.is_none() {
        return Backend::Metal;
    }

    // Neither. Fail loudly and name what is missing. THERE IS NOTHING TO FALL
    // BACK TO, and that is deliberate: a build that produced no physics while
    // writing to the same `target/release/` path would let a machine run,
    // report success and hand back fake results with nothing saying so. A tree
    // holds a real backend or it holds nothing.
    panic!(
        "\n\n  No real solver backend can be built on this machine, so there is \
         nothing to link.\n  CUDA: nothing at /usr/local/cuda/bin/nvcc, no \
         CUDA_PATH or PPF_CUDA_ROOT naming a toolkit root, and no nvcc on \
         PATH.\n  Metal: \
         {}.\n  Install one of the two. The only other backend is never \
         selected for you:\n    --features cpu builds the Rust CPU backend, \
         driven by the same neutral Newton driver as CUDA and Metal, about 30x \
         slower, refusing by name the few capabilities it does not carry yet \
         (src/driver/refusal.rs).\n\n",
        metal_reason.unwrap_or_default()
    );
}

/// Announce, loudly, that a GPU-capable machine is being given the CPU backend.
///
/// It keys on "a real GPU backend is available HERE" rather than on CUDA
/// specifically, so a Metal-capable Mac and a CUDA host are treated the same:
/// `--features cpu` is honored on both, and both get the banner.
///
/// This does NOT refuse, and the distinction it rests on is worth stating. A
/// build that computed NO PHYSICS while taking the same path would deserve a
/// hard refusal, because a machine that silently ended up on it would get fake
/// results and could not tell.
/// The CPU backend computes the same physics from the same neutral kernel
/// bodies, so selecting it on a CUDA or Metal host is a legitimate thing to
/// want: a parity run against the GPU, a determinism check, a debugging session
/// on a laptop, or simply preferring a slow correct answer to none.
///
/// `--features cpu` is itself the explicit request. Nobody types it by accident,
/// so demanding a second opt-in through an environment variable was friction
/// rather than a safeguard, and it made the supported way to select this backend
/// look unsupported.
///
/// What survives is the SURPRISE, which was the real hazard: the build lands at
/// the same `target/release/` path as the GPU solver, so a session already
/// connected to that path picks up a backend roughly 30x slower without being
/// told. That is worth a banner, not a wall.
fn guard_cpu() {
    let real = if cuda_toolkit_present() {
        Some("CUDA (/usr/local/cuda, CUDA_PATH, PPF_CUDA_ROOT or nvcc on PATH)")
    } else if metal_unavailable_reason().is_none() {
        Some("Metal (macOS with a Metal-capable SDK)")
    } else {
        None
    };
    if let Some(real) = real {
        for line in [
            "".to_string(),
            "  ******************************************************************".to_string(),
            "  * BUILDING THE CPU BACKEND ON A GPU-CAPABLE MACHINE              *".to_string(),
            "  ******************************************************************".to_string(),
            format!("  This machine can build the REAL {real} backend, and you asked"),
            "  for the CPU one. That is supported and this build will proceed.".to_string(),
            "".to_string(),
            "  Two consequences, so neither is a surprise later:".to_string(),
            "    - it is roughly 30x slower than the GPU backend, and".to_string(),
            "    - it overwrites the solver at target/release/, so a session".to_string(),
            "      already connected to that path now gets the slow backend.".to_string(),
            "".to_string(),
            "  To go back:  cargo build --release   (no flags; picks CUDA or".to_string(),
            "  Metal by host)".to_string(),
            "".to_string(),
        ] {
            println!("cargo:warning={line}");
        }
    }
}

/// The neutral kernel bodies the CPU backend's entry points include.
///
/// Each is a path under `src/kernels` with the `.kernel.cpp` suffix left off. The
/// list is short and explicit rather than a directory walk because it is the
/// answer to "what does this backend compile", and it grows in the same change
/// that gives a body a caller in one of the `entrypoints/*.cpp` shims. A body with
/// no caller there is one this backend has not reached yet, which is the same
/// statement `src/driver/refusal.rs` makes to a user at run time.
/// Some of the neutral bodies are deliberately ABSENT and their absence is not
/// a gap to close. `primitives/{reduce,scan}` are device-only by design: each is
/// written against a warp or threadgroup primitive that `seam_host.h` leaves
/// undefined on purpose, so they do not compile here. THE SCAN IS ONE THAT CAME
/// BACK, and it came back by splitting rather than by weakening:
/// `primitives/scan_levels` is the element-wise multi-level scan the driver
/// dispatches and it is in the list below, while `primitives/scan` keeps the
/// cooperative transcription that no backend can run on all three targets at
/// once.
///
/// `primitives/radix` IS THE OTHER ONE, and it came back by a different route:
/// the one authorized lane exception admits a SERIAL
/// TWIN beside a cooperative body, so the histogram and the scatter now render
/// a form this host compiles. **The reason it was excluded was that it did not
/// compile here, and that reason is gone**; leaving it out on the strength of a
/// stale rationale would have kept the CPU backend on a bitonic network for no
/// remaining reason. The other three,
/// `energy/model/{pdrd_rigid,pdrd_lock_projector}` and `schwarz/schwarz` DO
/// compile on this host (measured);
/// they are left out because this backend does not implement the capabilities
/// they serve, and `src/driver/refusal.rs` says so by name at `initialize()`. An
/// entry here for a refused capability would be reachable code for a feature
/// that cannot run.
///
/// THE LIST SHRINKS AS REFUSALS COME OFF, and the two that left it are the
/// worked examples: `energy/model/face_pressure` and `energy/model/torque` are
/// both in the list below now, each having landed in the change that removed its
/// entry from `refusal.rs`. `sand_rigid` needs `contact/grain_pair` with it,
/// the per-grain angular accumulators being contact's to produce.
const KERNELS: [&str; 87] = [
    // The linear solve.
    "solver/spmv",
    "solver/block_jacobi",
    "solver/translation_lock_check",
    "solver/translation_lock_frames",
    "solver/translation_lock_rows",
    "solver/pcg",
    "primitives/vec_ops",
    "csrmat/dynamic_csr",
    "csrmat/fixed_csr",
    // The step: the Newton driver's position arithmetic, written once here so
    // no backend re-spells it.
    "main/target",
    "main/override_seed",
    "main/velocity",
    "main/rewind_fix",
    "main/dx_seed",
    "main/dx_norm",
    // The linear system dump, which renders an entry point like any other. It
    // is a DIAGNOSTIC rather than a step of the solve, and that is exactly why
    // it was missing here for a while: nothing in a run reaches it, so nothing
    // in a run noticed. The library's table is built by walking the tree, so it
    // carried the entry while this list did not, and the two tables then had
    // different lengths.
    "main/dump_linsys",
    // The Schwarz preconditioner's domain construction. The leaf math beside it
    // has been neutral for a while; these are the first entry points over it.
    "schwarz/schwarz",
    "main/dirichlet",
    "main/fix_xz_drag",
    "main/position_step",
    "main/position_accept",
    // The stretch indicator's rod half. Its ungated body is what
    // `cpp/main/main.cu` calls, so the two backends report one number for one
    // pose, which is the whole reason a diagnostic is a kernel body.
    "main/stretch",
    // Elasticity: the deformation gradients, the SVDs, the spectral
    // decompositions and the material-frame converters.
    "utility/svd3x2",
    "utility/svd3x3",
    "utility/face_deformation",
    "utility/face_convert",
    "utility/tet_convert",
    "utility/vertex_normal",
    "eigenanalysis/face_eigenanalysis",
    "eigenanalysis/tet_eigenanalysis",
    // Rayleigh stiffness damping, one body per element arity.
    "utility/face_damping",
    "utility/tet_damping",
    "utility/rod_damping",
    "utility/hinge_damping",
    "utility/rod_bend_damping",
    // Bending and stitching.
    // The elastic model ids, an enumeration with no body of its own. It is
    // here because a rendered body reaches another rendering by RELATIVE path,
    // so a neutral source this backend compiles can only include a
    // `.kernel.cpp` that is also rendered into the mirrored tree.
    "energy/elastic_model",
    // The material diff table, one body per element arity. Each composes its
    // material in `energy/model/{arap,stvk,snhk}.hpp`, which the fused bodies in
    // `energy/face_force.kernel.cpp` and `energy/tet_force.kernel.cpp` also
    // call, so the two arrangements of the same material share one definition.
    "energy/model/material_diff_table",
    // The BaraffWitkin membrane as a staged pass. Its material is composed in
    // `energy/model/baraffwitkin.hpp`, which the fused body in
    // `energy/face_force.kernel.cpp` also calls, so the two arrangements of the
    // same material share one definition.
    "energy/model/baraffwitkin",
    // The per-face inflation pressure. NO LONGER ABSENT: the doc above listed
    // it among the bodies this backend does not implement, and the entry beside
    // it is what removed that refusal.
    "energy/model/face_pressure",
    // The torque group's frame. Its nine bodies were already neutral and
    // reached by `energy.cu`; what this adds is the per-group entry that runs
    // the three walks, which is the pre-pass the momentum row's torque term
    // needs before any member's row is assembled.
    "energy/model/torque",
    // The SAND grain's contact-point slip and its share of the friction force.
    // `contact/contact_narrow` includes it, so its rendering must sit beside
    // that one's: a generated rendering resolves a quoted include against the
    // generated tree, not the source tree.
    "contact/grain_pair",
    // The SAND grain's three rows: the Schur condense before the solve, the
    // recover after it, and the post-solve integrate.
    "energy/model/sand_rigid",
    // The rod segment's Hookean stretch term. Its body composes
    // `energy/model/hook.hpp`, which the fused body in the same file also
    // calls, so the two arrangements of the same spring share one definition.
    "energy/rod_force",
    // The shell membrane layer, whole: one entry over the fused body that
    // holds the deformation gradient, the SVD, the material table, the
    // spectral force, the PSD-projected 6x6 Hessian, the material-frame
    // conversion and the damping in registers and embeds both the membrane and
    // the per-face pressure itself. Its materials come from the same
    // `energy/model/{arap,snhk,stvk}.hpp` and `energy/model/baraffwitkin.hpp`
    // the staged passes above compose.
    "energy/face_force",
    // The tet elastic layer, whole: one entry over the fused body that holds
    // the deformation gradient, the factorization, the material table, the
    // spectral force, the 12x12 spectral Hessian and the damping in registers
    // and embeds the result itself. Its material comes from the same
    // `energy/model/{arap,snhk,stvk}.hpp` the staged diff table above composes.
    "energy/tet_force",
    "energy/model/shell_bend_stiffness",
    "energy/model/shell_bend",
    "energy/model/rod_bend_stiffness",
    "energy/model/rod_bend",
    "energy/model/stitch",
    "energy/model/pdrd_rigid",
    "energy/model/pdrd_lock_projector",
    // The barrier family and strain limiting.
    "barrier/contact_barrier",
    "barrier/contact_stiffness",
    "strainlimiting/shell_strain",
    "strainlimiting/rod_strain",
    "strainlimiting/strain_toi",
    // Contact.
    "energy/model/push",
    "energy/model/friction",
    "contact/analytic_contact",
    // Which pairs a pass may act on, stated once. It declares no entry point
    // and is listed here because two rendered bodies include it: the
    // intersection rule's own callers and, since the CCD line search returned
    // to the device, `contact_pair_admitted` beside them.
    "contact/pair_filter",
    // THE FINAL PENETRATION GATE, fused: one dispatch per query element, the
    // tester invoked as a per-hit device functor inside the traversal, and
    // nothing materialized between the two. `contact/intersect_record` carries
    // no entry point of its own and is listed for the reason
    // `contact/pair_filter` is: the scan's rendering includes it, and a
    // rendered body resolves a quoted include against the generated tree.
    "contact/intersect_geometry",
    "contact/intersect_record",
    "contact/contact_assembly",
    // THE PER-OBJECT STATISTICS RECORDERS, carrying no entry point of their
    // own and listed for the reason `contact/pair_filter` is: the three
    // narrow-phase renderings include them, and a rendered body resolves a
    // quoted include against the generated tree.
    "contact/contact_statistics",
    "contact/contact_narrow",
    "contact/collision_narrow",
    // The CCD line search, fused: one dispatch per (query kind, tree) pair,
    // the ACCD advance invoked as a per-hit device functor inside the
    // traversal, and nothing materialized between the phases. It replaces a
    // broad phase that wrote a candidate pair list the reference never builds.
    "contact/ccd_sweep",
    "contact/vertex_constraint",
    "main/momentum",
    "contact/pair_cache",
    // The broad phase.
    "contact/aabb",
    "contact/aabb_traversal",
    "lbvh/lbvh",
    "lbvh/bitonic",
    // Plasticity.
    "plasticity/plasticity",
    // The element scatters.
    "utility/face_scatter",
    "utility/face_hessian_scatter",
    "utility/hinge_scatter",
    "utility/collision_window",
    "utility/rod_scatter",
    "utility/stitch_scatter",
    "utility/vertex_scatter",
    // LAST ON PURPOSE: the canonical kernel id order is this list's order,
    // and these three carry the three highest ids. Inserting a file
    // mid-list renumbers every entry below it, which the driver TABLE
    // then contradicts, because a kernel id is a table INDEX.
    "primitives/reduce_scalar",
    "primitives/reduce_bounds",
    "primitives/radix",
    "primitives/scan_levels",
];

/// Buffer fields the driver fills with an (arena, offset) handle rather than a
/// host address, as `(kernel stem, "Record.field")`.
///
/// **THIS LIST IS A RATCHET, AND IT ONLY GROWS.** A generated record's Rust twin
/// spells a buffer field `HostRef` until its name appears here and `Handle`
/// afterwards, so an entry is the statement that the DRIVER's buffer behind it
/// is a device allocation. A backend library resolves an (arena, offset) handle
/// and cannot resolve a host address, so a phase can be dispatched on a GPU
/// backend exactly when every field of every record it uses is named here;
/// `ppf_cts_compute::abi::AbiDevice` refuses the rest by name rather than
/// launching a wild pointer.
///
/// IT LIVES IN THE BUILD AND NOT IN A NEUTRAL KERNEL SOURCE. A body says nothing
/// about where its caller's buffers live, and a migration state written into one
/// would be a kernel taking a branch on its caller, which is forbidden.
///
/// A name no declaration in the named file carries is a build failure, so a
/// typo cannot quietly leave a buffer unmigrated: the transcompiler prints the
/// buffer fields that file does declare.
const HANDLE_FIELDS: &[(&str, &str)] = &[
    ("main/velocity", "VelocityTermsArgs.prop"),
    ("main/dx_seed", "DxSeedArgs.prop"),
    ("strainlimiting/strain_toi", "ShellStrainToiGatedArgs.inverse_rest"),
    // THE INVERSE REST MATRICES. Both sides write them: the creep scatters
    // from a kernel and a checkpoint restore writes them wholesale, which
    // ReadbackBuffer serves now that it has a host seed.
    ("utility/face_convert", "FaceConvertForceArgs.inverse_rest"),
    ("utility/face_convert", "FaceConvertHessianArgs.inverse_rest"),
    ("utility/face_deformation", "FaceDeformationGradientArgs.inverse_rest"),
    // THE COMPOSED SPECTRAL HESSIAN, which reaches the same buffers as the
    // pair it replaces and so is device-resident on the same terms.
    ("energy/tet_force", "TetSpectralConvertHessianArgs.gradient_sigma"),
    ("energy/tet_force", "TetSpectralConvertHessianArgs.hessian_sigma"),
    ("energy/tet_force", "TetSpectralConvertHessianArgs.u"),
    ("energy/tet_force", "TetSpectralConvertHessianArgs.sigma"),
    ("energy/tet_force", "TetSpectralConvertHessianArgs.vt"),
    ("energy/tet_force", "TetSpectralConvertHessianArgs.inverse_rest"),
    ("energy/tet_force", "TetSpectralConvertHessianArgs.mass"),
    ("energy/tet_force", "TetSpectralConvertHessianArgs.hessian"),
    ("utility/tet_convert", "TetConvertForceArgs.inverse_rest"),
    // THE PER-ELEMENT MASS, gathered by the converter so the scale rides the
    // conversion rather than a second pass over the materialized pack.
    ("utility/tet_convert", "TetConvertForceArgs.mass"),
    ("utility/tet_convert", "TetConvertHessianArgs.mass"),
    ("utility/tet_convert", "TetConvertHessianArgs.inverse_rest"),
    ("utility/tet_convert", "TetDeformationGradientArgs.inverse_rest"),
    // THE PER-OBJECT STATISTICS CHANNEL. The counter is written by an ATOMIC
    // from inside the contact kernels, exactly as `contact.cu` writes it, so
    // it is device-resident on every backend rather than a host fold over a
    // downloaded pair list. Its two index maps are read on the device beside
    // it.
    ("contact/contact_narrow", "ContactPointFaceArgs.statistics_contact_count"),
    ("contact/contact_narrow", "ContactPointFaceArgs.statistics_object_index"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.statistics_contact_count"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.statistics_object_index"),
    ("contact/contact_narrow", "ContactPointPointArgs.statistics_contact_count"),
    ("contact/contact_narrow", "ContactPointPointArgs.statistics_object_index"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.statistics_contact_count"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.statistics_object_index"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.statistics_contact_count"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.statistics_object_index"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.statistics_static_object_index"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.statistics_contact_count"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.statistics_object_index"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.statistics_static_object_index"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.statistics_contact_count"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.statistics_object_index"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.statistics_static_object_index"),
    ("contact/vertex_constraint", "VertexConstraintArgs.statistics_contact_count"),
    ("contact/vertex_constraint", "VertexConstraintArgs.statistics_object_index"),
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF. It is a COPY of the force
    // vector, made on the device by the same `fill`/`copy` the force itself
    // lives under, so it is device-resident on exactly the terms `force` is.
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.residual"),
    ("contact/contact_narrow", "ContactPointFaceArgs.residual"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.residual"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.residual"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.residual"),
    ("contact/contact_narrow", "ContactPointPointArgs.residual"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.residual"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.residual"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.residual"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.residual"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.residual"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.residual"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.residual"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.residual"),
    ("contact/vertex_constraint", "VertexConstraintArgs.residual"),
    ("contact/vertex_constraint", "VertexConstraintArgs.fixed_index"),
    ("contact/vertex_constraint", "VertexConstraintArgs.fixed_offset"),
    ("contact/contact_narrow", "ContactPointFaceArgs.fixed_index"),
    ("contact/contact_narrow", "ContactPointFaceArgs.fixed_offset"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.fixed_index"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.fixed_offset"),
    ("contact/contact_narrow", "ContactPointPointArgs.fixed_index"),
    ("contact/contact_narrow", "ContactPointPointArgs.fixed_offset"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.fixed_index"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.fixed_offset"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.fixed_index"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.fixed_offset"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.fixed_index"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.fixed_offset"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.fixed_index"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.fixed_offset"),
    ("contact/contact_narrow", "ContactPointFaceArgs.face_prop"),
    ("contact/contact_narrow", "ContactPointFaceArgs.vertex_param"),
    ("contact/contact_narrow", "ContactPointFaceArgs.face_param"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.edge_prop"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.vertex_param"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.edge_param"),
    ("contact/contact_narrow", "ContactPointPointArgs.vertex_param"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.edge_prop"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.edge_param"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.face_prop"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.face_param"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.static_vertex_prop"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.static_vertex_param"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.static_x"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.vertex_param"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.static_face_prop"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.static_face_param"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.static_face"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.static_x"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.edge_prop"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.edge_param"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.static_edge_prop"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.static_edge_param"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.static_edge"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.static_x"),
    ("contact/aabb", "AabbPointContactQueryArgs.params"),
    ("contact/aabb", "AabbPointContactQueryMaskedArgs.params"),
    ("contact/aabb", "AabbVertexScanQueryArgs.params"),
    ("contact/aabb", "AabbVertexScanQueryMaskedArgs.params"),
    ("contact/aabb", "AabbLeafVertexArgs.params"),
    ("contact/aabb", "AabbLeafFaceArgs.face"),
    ("contact/aabb", "AabbLeafFaceArgs.prop"),
    ("contact/aabb", "AabbLeafFaceArgs.params"),
    ("contact/aabb", "AabbLeafEdgeArgs.edge"),
    ("contact/aabb", "AabbLeafEdgeArgs.prop"),
    ("contact/aabb", "AabbLeafEdgeArgs.params"),
    ("lbvh/lbvh", "EdgeCentroidArgs.edge"),
    ("lbvh/lbvh", "FaceCentroidArgs.face"),
    ("contact/aabb", "AabbEdgeContactQueryArgs.prop"),
    ("contact/aabb", "AabbEdgeContactQueryMaskedArgs.prop"),
    ("contact/aabb", "AabbEdgeContactQueryArgs.params"),
    ("contact/aabb", "AabbEdgeContactQueryMaskedArgs.params"),
    ("contact/contact_narrow", "ContactPointFaceArgs.vertex_prop"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.vertex_prop"),
    ("contact/contact_narrow", "ContactPointPointArgs.vertex_prop"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.vertex_prop"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.vertex_prop"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.vertex_prop"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.vertex_prop"),
    // THE PER-VERTEX PROPERTIES, staged rather than static because the plastic
    // creep still writes them on the host. StagedBuffer::handle panics if a
    // dispatch names them before the upload, so a forgotten upload traps.
    ("main/momentum", "MomentumEmbedArgs.prop"),
    ("contact/vertex_constraint", "VertexConstraintArgs.vertex_prop"),
    ("contact/vertex_constraint", "VertexConstraintSweepArgs.vertex_prop"),
    ("contact/aabb", "AabbPointContactQueryArgs.prop"),
    ("contact/aabb", "AabbPointContactQueryMaskedArgs.prop"),
    ("contact/aabb", "AabbVertexScanQueryArgs.prop"),
    ("contact/aabb", "AabbVertexScanQueryMaskedArgs.prop"),
    ("contact/aabb", "AabbLeafVertexArgs.prop"),
    ("contact/contact_narrow", "ContactPointFaceArgs.face"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.face"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.edge"),
    ("contact/contact_narrow", "ContactPointPointArgs.face"),
    ("contact/contact_narrow", "ContactPointPointArgs.edge"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.edge"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.face"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.edge"),
    // THE MESH TOPOLOGY the broad phase reads. `SolverState` already seeds
    // mesh_face and mesh_edge at allocate, so these fields name a buffer that
    // was device-resident before this change and was simply not being used.
    ("contact/aabb", "AabbEdgeContactQueryMaskedArgs.edge"),
    ("contact/aabb", "AabbEdgeContactQueryArgs.edge"),
    ("contact/aabb", "AabbEdgeScanQueryMaskedArgs.edge"),
    ("contact/aabb", "AabbEdgeScanQueryArgs.edge"),
    // THE KEYFRAME SEED AND THE GATHER, which reach the SAME positions. The
    // host arrays are refreshed FROM the device by fetch() and never back into
    // it, so a host-side write here is silently lost and a host-side read is a
    // frame behind.
    ("main/override_seed", "OverrideVelocitySeedListedArgs.curr"),
    ("main/override_seed", "OverrideVelocitySeedListedArgs.prev"),
    ("main/override_seed", "OverrideAngularSeedListedArgs.curr"),
    ("main/override_seed", "OverrideAngularSeedListedArgs.prev"),
    ("main/override_seed", "GatherPositionAbsoluteArgs.curr"),
    ("contact/vertex_constraint", "VertexConstraintArgs.eval_x"),
    ("contact/vertex_constraint", "VertexConstraintArgs.current"),
    ("contact/vertex_constraint", "VertexConstraintSweepArgs.x0"),
    ("contact/vertex_constraint", "VertexConstraintSweepArgs.x1"),
    ("main/velocity", "VelocityTermsArgs.current"),
    ("main/velocity", "VelocityTermsArgs.previous"),
    ("main/fix_xz_drag", "FixXzDragArgs.previous"),
    ("contact/aabb", "AabbLeafFaceArgs.x0"),
    ("contact/aabb", "AabbLeafFaceArgs.x1"),
    ("contact/aabb", "AabbLeafEdgeArgs.x0"),
    ("contact/aabb", "AabbLeafEdgeArgs.x1"),
    ("contact/aabb", "AabbLeafVertexArgs.x0"),
    ("contact/aabb", "AabbLeafVertexArgs.x1"),
    ("plasticity/plasticity", "PlasticityFaceInverseRestArgs.x"),
    ("plasticity/plasticity", "PlasticityTetInverseRestArgs.x"),
    ("plasticity/plasticity", "PlasticityFaceFromRecordsArgs.prop"),
    ("plasticity/plasticity", "PlasticityFaceFromRecordsArgs.face_param"),
    ("plasticity/plasticity", "PlasticityFaceFromRecordsArgs.plasticity"),
    ("plasticity/plasticity", "PlasticityFaceFromRecordsArgs.threshold"),
    ("plasticity/plasticity", "PlasticityTetFromRecordsArgs.prop"),
    ("plasticity/plasticity", "PlasticityTetFromRecordsArgs.tet_param"),
    ("plasticity/plasticity", "PlasticityTetFromRecordsArgs.plasticity"),
    ("plasticity/plasticity", "PlasticityTetFromRecordsArgs.threshold"),
    ("plasticity/plasticity", "PlasticityHingeFromRecordsArgs.prop"),
    ("plasticity/plasticity", "PlasticityHingeFromRecordsArgs.hinge_param"),
    ("plasticity/plasticity", "PlasticityHingeFromRecordsArgs.plasticity"),
    ("plasticity/plasticity", "PlasticityHingeFromRecordsArgs.threshold"),
    ("plasticity/plasticity", "PlasticityRodFromRecordsArgs.site_edge"),
    ("plasticity/plasticity", "PlasticityRodFromRecordsArgs.edge_prop"),
    ("plasticity/plasticity", "PlasticityRodFromRecordsArgs.edge_param"),
    ("plasticity/plasticity", "PlasticityRodFromRecordsArgs.plasticity"),
    ("plasticity/plasticity", "PlasticityRodFromRecordsArgs.threshold"),
    ("energy/model/shell_bend", "ShellBendAngleArgs.x"),
    ("energy/model/rod_bend", "RodBendAngleArgs.x"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.x"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.current"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.node"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.node_slots"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.site_edge"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.edge_prop"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.edge_param"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.vertex_prop"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.force"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.fixed_index"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.fixed_offset"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.fixed_value"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.refused"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.witness"),
    ("energy/model/rod_bend", "RodBendEmbedArgs.hess_slots"),
    ("main/fix_xz_drag", "FixXzDragArgs.eval_x"),
    ("main/position_accept", "PositionAcceptArgs.origin"),
    ("main/position_accept", "PositionAcceptArgs.proposed"),
    ("main/target", "ComputeTargetSeedArgs.current"),
    ("main/target", "ComputeTargetSeedArgs.previous"),
    ("main/target", "ComputeTargetSeedArgs.target"),
    ("main/dx_seed", "DxSeedArgs.target"),
    ("main/position_step", "PositionStepArgs.eval_x"),
    ("main/stretch", "RodStretchRatioGatedArgs.x"),
    ("contact/contact_narrow", "ContactPointFaceArgs.x0"),
    ("contact/contact_narrow", "ContactPointFaceArgs.x"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.x0"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.x"),
    ("contact/contact_narrow", "ContactPointPointArgs.x0"),
    ("contact/contact_narrow", "ContactPointPointArgs.x"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.x0"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.x"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.x0"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.x"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.x0"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.x"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.x0"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.x"),
    // THE POSITION COMPONENT, and it is one unit because residency is keyed by
    // <Record>.<field>: the iterate, the committed positions, the implicit
    // target, every broad-phase query and centroid that reads them, and the
    // STATIC COLLIDER MESH, which is not a position at all and joins because
    // the same records serve both. A field left behind here leaves one pass
    // reading a host address while the rest read a handle.
    ("main/momentum", "MomentumEmbedArgs.eval_x"),
    ("main/momentum", "MomentumEmbedArgs.current"),
    ("main/momentum", "MomentumEmbedArgs.target"),
    ("utility/tet_convert", "TetDeformationGradientArgs.x"),
    ("utility/tet_damping", "TetDampingArgs.x"),
    ("utility/tet_damping", "TetDampingArgs.current"),
    ("utility/face_deformation", "FaceDeformationGradientArgs.x"),
    ("utility/face_damping", "FaceDampingArgs.x"),
    ("utility/face_damping", "FaceDampingArgs.current"),
    ("energy/model/shell_bend", "ShellBendForceHessianCheckedArgs.x"),
    ("utility/hinge_damping", "HingeDampingArgs.x"),
    ("utility/hinge_damping", "HingeDampingArgs.current"),
    ("energy/model/rod_bend", "RodBendForceHessianArgs.x"),
    ("utility/rod_bend_damping", "RodBendDampingArgs.x"),
    ("utility/rod_bend_damping", "RodBendDampingArgs.current"),
    ("energy/model/stitch", "StitchForceHessianGatheredArgs.x"),
    ("strainlimiting/rod_strain", "RodStrainForceHessianGatedArgs.x"),
    ("strainlimiting/rod_strain", "RodStrainStiffnessGatedArgs.x"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessGatedArgs.x"),
    ("strainlimiting/strain_toi", "ShellStrainToiGatedArgs.finish"),
    ("strainlimiting/strain_toi", "ShellStrainToiGatedArgs.start"),
    ("strainlimiting/strain_toi", "RodStrainToiGatedArgs.finish"),
    ("strainlimiting/strain_toi", "RodStrainToiGatedArgs.start"),
    ("solver/spmv", "FixedCsrProductRowArgs.index"),
    ("solver/spmv", "FixedCsrProductRowArgs.offset"),
    ("solver/spmv", "FixedCsrProductRowArgs.value"),
    ("solver/spmv", "FixedCsrProductRowArgs.transpose_pair"),
    ("solver/spmv", "FixedCsrProductRowArgs.transpose_offset"),
    ("csrmat/dynamic_csr", "DynScatterTransposePassArgs.transpose_offset"),
    ("csrmat/dynamic_csr", "DynScatterTransposePassArgs.transpose_index"),
    ("csrmat/dynamic_csr", "DynScatterTransposePassArgs.transpose_value"),
    ("csrmat/dynamic_csr", "DynScatterTransposePassArgs.cursor"),
    ("csrmat/dynamic_csr", "DynCountTransposePassArgs.transpose_count"),
    ("csrmat/fixed_csr", "FixedCsrAtomicPushArgs.value"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessGatedArgs.index"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessGatedArgs.offset"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessGatedArgs.value"),
    ("strainlimiting/rod_strain", "RodStrainStiffnessGatedArgs.index"),
    ("strainlimiting/rod_strain", "RodStrainStiffnessGatedArgs.offset"),
    ("strainlimiting/rod_strain", "RodStrainStiffnessGatedArgs.value"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.fixed_value"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.fixed_value"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.fixed_value"),
    ("main/momentum", "MomentumEmbedArgs.neighbor_index"),
    ("main/momentum", "MomentumEmbedArgs.neighbor_offset"),
    ("main/momentum", "MomentumEmbedArgs.face"),
    ("contact/contact_narrow", "ContactPointPointArgs.vertex_edge_index"),
    ("contact/contact_narrow", "ContactPointPointArgs.vertex_edge_offset"),
    ("contact/contact_narrow", "ContactPointPointArgs.vertex_face_index"),
    ("contact/contact_narrow", "ContactPointPointArgs.vertex_face_offset"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.edge_face_index"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.edge_face_offset"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.fixed_value"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.fixed_value"),
    ("contact/contact_narrow", "ContactPointFaceArgs.fixed_value"),
    ("contact/contact_narrow", "ContactPointPointArgs.fixed_value"),
    ("contact/vertex_constraint", "VertexConstraintArgs.fixed_value"),
    ("contact/vertex_constraint", "VertexConstraintArgs.vertex_param"),
    ("plasticity/plasticity", "PlasticityCreepRestAngleArgs.rest_angle"),
    ("main/override_seed", "OverrideVelocitySeedListedArgs.indices"),
    ("main/override_seed", "OverrideAngularSeedListedArgs.indices"),
    ("main/override_seed", "GatherPositionAbsoluteArgs.indices"),
    ("main/override_seed", "GatherPositionAbsoluteArgs.out"),
    ("csrmat/fixed_csr", "FixedCsrAtomicPushArgs.index"),
    ("csrmat/fixed_csr", "FixedCsrAtomicPushArgs.offset"),
    ("solver/spmv", "FixedCsrProductRowArgs.x"),
    ("solver/spmv", "FixedCsrProductRowArgs.result"),
    ("csrmat/fixed_csr", "FixedCsrAtomicPushArgs.row"),
    ("csrmat/fixed_csr", "FixedCsrAtomicPushArgs.column"),
    ("csrmat/fixed_csr", "FixedCsrAtomicPushArgs.block"),
    ("csrmat/fixed_csr", "FixedCsrAtomicPushArgs.stored"),
    ("energy/rod_force", "RodStretchDiffTableArgs.x"),
    ("energy/rod_force", "RodStretchDiffTableArgs.rest_length"),
    ("energy/rod_force", "RodStretchDiffTableArgs.weight"),
    ("energy/rod_force", "RodStretchDiffTableArgs.gradient"),
    ("energy/rod_force", "RodStretchDiffTableArgs.hessian"),
    ("energy/rod_force", "RodStretchEmbedArgs.x"),
    ("energy/rod_force", "RodStretchEmbedArgs.current"),
    ("energy/rod_force", "RodStretchEmbedArgs.edge"),
    ("energy/rod_force", "RodStretchEmbedArgs.edge_slots"),
    ("energy/rod_force", "RodStretchEmbedArgs.prop"),
    ("energy/rod_force", "RodStretchEmbedArgs.edge_param"),
    ("energy/rod_force", "RodStretchEmbedArgs.hess_slots"),
    ("energy/rod_force", "RodStretchEmbedArgs.force"),
    ("energy/rod_force", "RodStretchEmbedArgs.fixed_index"),
    ("energy/rod_force", "RodStretchEmbedArgs.fixed_offset"),
    ("energy/rod_force", "RodStretchEmbedArgs.fixed_value"),
    ("energy/rod_force", "RodStretchEmbedArgs.refused"),
    ("energy/rod_force", "RodStretchEmbedArgs.witness"),
    ("utility/rod_damping", "RodDampingArgs.x"),
    ("utility/rod_damping", "RodDampingArgs.current"),
    ("utility/rod_damping", "RodDampingArgs.beta"),
    ("utility/rod_damping", "RodDampingArgs.gradient"),
    ("utility/rod_damping", "RodDampingArgs.hessian"),
    ("contact/aabb", "AabbLeafActiveArgs.nodes"),
    ("contact/aabb", "AabbLeafActiveArgs.aabb"),
    ("lbvh/lbvh", "LbvhNodesArgs.nodes"),
    ("contact/aabb", "AabbLeafFaceArgs.nodes"),
    ("contact/aabb", "AabbLeafFaceArgs.aabb"),
    ("contact/aabb", "AabbLeafEdgeArgs.nodes"),
    ("contact/aabb", "AabbLeafEdgeArgs.aabb"),
    ("contact/aabb", "AabbLeafVertexArgs.nodes"),
    ("contact/aabb", "AabbLeafVertexArgs.aabb"),
    ("contact/aabb", "AabbMergeLevelArgs.nodes"),
    ("contact/aabb", "AabbMergeLevelArgs.aabb"),
    // THE CCD LINE SEARCH'S SIX FUSED SWEEPS. Every buffer they name was
    // already a device allocation before this entry existed; what the sweeps
    // add is a device READER for each, in place of the host walk that used to
    // download a candidate pair list and advance over rayon.
    // THE INTERSECTION SCAN'S FOUR FUSED WALKS. Every input was already a
    // device allocation before these entries existed; the three outputs (the
    // per-element flag, the bounded record array and its claim counter) are new
    // and are device allocations from the day they are written, because the
    // reference reads exactly those three back and nothing else.
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.vert"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.face"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.edge"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.vertex_prop"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.face_prop"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.edge_prop"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.node"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.aabb"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.query"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.flag"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.records"),
    ("contact/intersect_geometry", "IntersectScanFaceEdgeArgs.counter"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.vert"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.edge"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.vertex_prop"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.edge_prop"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.edge_param"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.node"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.aabb"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.query"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.flag"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.records"),
    ("contact/intersect_geometry", "IntersectScanEdgeEdgeArgs.counter"),
    ("contact/intersect_geometry", "IntersectScanPointPointArgs.vert"),
    ("contact/intersect_geometry", "IntersectScanPointPointArgs.vertex_prop"),
    ("contact/intersect_geometry", "IntersectScanPointPointArgs.vertex_param"),
    ("contact/intersect_geometry", "IntersectScanPointPointArgs.node"),
    ("contact/intersect_geometry", "IntersectScanPointPointArgs.aabb"),
    ("contact/intersect_geometry", "IntersectScanPointPointArgs.query"),
    ("contact/intersect_geometry", "IntersectScanPointPointArgs.flag"),
    ("contact/intersect_geometry", "IntersectScanPointPointArgs.records"),
    ("contact/intersect_geometry", "IntersectScanPointPointArgs.counter"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.vert"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.edge"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.vertex_prop"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.edge_prop"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.collider_vertex"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.collider_face"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.node"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.aabb"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.query"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.flag"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.records"),
    ("contact/intersect_geometry", "IntersectScanCollisionMeshArgs.counter"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.x0"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.x1"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.face"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.vertex_prop"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.face_prop"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.vertex_param"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.face_param"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.vertex_node"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.node"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.aabb"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.active"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.out_toi"),
    ("main/dump_linsys", "DumpLinsysRowToCooArgs.row_begin"),
    ("main/dump_linsys", "DumpLinsysRowToCooArgs.row_end"),
    ("main/dump_linsys", "DumpLinsysRowToCooArgs.column"),
    ("main/dump_linsys", "DumpLinsysRowToCooArgs.block"),
    ("main/dump_linsys", "DumpLinsysRowToCooArgs.cursor"),
    ("main/dump_linsys", "DumpLinsysRowToCooArgs.out_row"),
    ("main/dump_linsys", "DumpLinsysRowToCooArgs.out_column"),
    ("main/dump_linsys", "DumpLinsysRowToCooArgs.out_block"),
    ("main/dump_linsys", "DumpLinsysRowToCooArgs.diagonal"),
    ("contact/ccd_sweep", "CcdPointFaceArgs.out_overlap"),
    ("contact/ccd_sweep", "CcdPointPointArgs.x0"),
    ("contact/ccd_sweep", "CcdPointPointArgs.x1"),
    ("contact/ccd_sweep", "CcdPointPointArgs.vertex_prop"),
    ("contact/ccd_sweep", "CcdPointPointArgs.vertex_param"),
    ("contact/ccd_sweep", "CcdPointPointArgs.node"),
    ("contact/ccd_sweep", "CcdPointPointArgs.aabb"),
    ("contact/ccd_sweep", "CcdPointPointArgs.active"),
    ("contact/ccd_sweep", "CcdPointPointArgs.out_toi"),
    ("contact/ccd_sweep", "CcdPointPointArgs.out_overlap"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.x0"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.x1"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.edge"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.vertex_prop"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.edge_prop"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.edge_param"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.node"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.aabb"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.active"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.out_toi_ee"),
    ("contact/ccd_sweep", "CcdEdgeEdgeArgs.out_overlap_ee"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.x0"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.x1"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.collider_vertex"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.collider_face"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.vertex_prop"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.collider_face_prop"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.vertex_param"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.collider_face_param"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.node"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.aabb"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.active"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.out_toi"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceM2cArgs.out_overlap"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.x0"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.x1"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.collider_vertex"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.face"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.face_prop"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.collider_vertex_prop"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.face_param"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.collider_vertex_param"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.node"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.aabb"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.out_toi"),
    ("contact/ccd_sweep", "CcdCollisionPointFaceC2mArgs.out_overlap"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.x0"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.x1"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.edge"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.collider_vertex"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.collider_edge"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.edge_prop"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.collider_edge_prop"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.edge_param"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.collider_edge_param"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.node"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.aabb"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.active"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.out_toi_ee"),
    ("contact/ccd_sweep", "CcdCollisionEdgeEdgeArgs.out_overlap_ee"),
    ("contact/aabb_traversal", "AabbQueryPairsArgs.node"),
    ("contact/aabb_traversal", "AabbQueryPairsArgs.aabb"),
    ("contact/aabb_traversal", "AabbQueryPairsArgs.query"),
    ("contact/aabb_traversal", "AabbQueryPairsArgs.out"),
    ("contact/aabb_traversal", "AabbQueryPairsArgs.found"),
    ("lbvh/lbvh", "LbvhMortonFromBoundsArgs.cx"),
    ("lbvh/lbvh", "LbvhMortonFromBoundsArgs.cy"),
    ("lbvh/lbvh", "LbvhMortonFromBoundsArgs.cz"),
    ("lbvh/lbvh", "LbvhMortonFromBoundsArgs.codes"),
    ("lbvh/lbvh", "LbvhNodesArgs.morton"),
    ("lbvh/lbvh", "LbvhNodesArgs.sorted"),
    ("lbvh/lbvh", "LbvhNodeDepthArgs.parent"),
    ("lbvh/lbvh", "LbvhNodeDepthArgs.depth"),
    ("contact/pair_cache", "PairCacheRecordInterleavedArgs.pair_data"),
    ("contact/pair_cache", "PairCacheRecordInterleavedArgs.count"),
    ("contact/pair_cache", "PairCacheRecordInterleavedArgs.overflow"),
    ("contact/pair_cache", "PairCacheRecordInterleavedArgs.pairs"),
    ("contact/aabb", "AabbMergeLevelArgs.level"),
    ("strainlimiting/strain_toi", "ShellMaxStrainArgs.deformation"),
    ("strainlimiting/strain_toi", "ShellMaxStrainArgs.strain"),
    ("strainlimiting/strain_toi", "RodStrainValueArgs.difference"),
    ("strainlimiting/strain_toi", "RodStrainValueArgs.rest_length"),
    ("strainlimiting/strain_toi", "RodStrainValueArgs.strain"),
    ("energy/model/face_pressure", "FacePressureEmbedArgs.vert"),
    ("energy/model/face_pressure", "FacePressureEmbedArgs.face"),
    ("energy/model/face_pressure", "FacePressureEmbedArgs.pressure"),
    ("energy/model/face_pressure", "FacePressureEmbedArgs.gradient"),
    ("energy/model/face_pressure", "FacePressureEmbedArgs.hessian"),
    ("energy/model/torque", "TorqueGroupFrameArgs.group"),
    ("energy/model/torque", "TorqueGroupFrameArgs.member"),
    ("energy/model/torque", "TorqueGroupFrameArgs.position"),
    ("energy/model/torque", "TorqueGroupFrameArgs.prop"),
    ("energy/model/torque", "TorqueGroupFrameArgs.result"),
    ("main/momentum", "MomentumEmbedArgs.torque_vertex"),
    ("main/momentum", "MomentumEmbedArgs.torque_result"),
    // The SAND grain's Schur blocks, which the analytic contact accumulates and
    // the condense pass reads. Handles, not host pointers: this runs inside the
    // Newton loop.
    ("contact/vertex_constraint", "VertexConstraintArgs.grain_inv_inertia"),
    ("contact/vertex_constraint", "VertexConstraintArgs.out_grain_angular"),
    ("contact/vertex_constraint", "VertexConstraintArgs.out_grain_coupling"),
    ("contact/vertex_constraint", "VertexConstraintArgs.out_grain_rotational"),
    ("contact/contact_narrow", "ContactPointPointArgs.grain_inv_inertia"),
    ("contact/contact_narrow", "ContactPointPointArgs.grain_omega"),
    ("contact/contact_narrow", "ContactPointPointArgs.out_grain_torque"),
    ("contact/contact_narrow", "ContactPointPointArgs.out_grain_stiffness"),
    ("contact/contact_narrow", "ContactPointPointArgs.out_grain_normal"),
    // The SAND grain's three rows, every buffer field: all three run inside
    // the step and the first two inside the Newton loop.
    ("energy/model/sand_rigid", "SandGrainIntegrateRowArgs.inverse_rolling_inertia"),
    ("energy/model/sand_rigid", "SandGrainIntegrateRowArgs.angular"),
    ("energy/model/sand_rigid", "SandGrainIntegrateRowArgs.prop"),
    ("energy/model/sand_rigid", "SandGrainIntegrateRowArgs.params"),
    ("energy/model/sand_rigid", "SandGrainIntegrateRowArgs.curr"),
    ("energy/model/sand_rigid", "SandGrainIntegrateRowArgs.prev"),
    ("energy/model/sand_rigid", "SandGrainIntegrateRowArgs.torque"),
    ("energy/model/sand_rigid", "SandGrainIntegrateRowArgs.normal_sum"),
    ("energy/model/sand_rigid", "SandGrainIntegrateRowArgs.angular_stiffness"),
    ("energy/model/sand_rigid", "SandGrainIntegrateRowArgs.omega"),
    ("energy/model/sand_rigid", "SandGrainCondenseRowArgs.inverse_center_inertia"),
    ("energy/model/sand_rigid", "SandGrainCondenseRowArgs.angular"),
    ("energy/model/sand_rigid", "SandGrainCondenseRowArgs.coupling"),
    ("energy/model/sand_rigid", "SandGrainCondenseRowArgs.rotational_gradient"),
    ("energy/model/sand_rigid", "SandGrainCondenseRowArgs.previous_omega"),
    ("energy/model/sand_rigid", "SandGrainCondenseRowArgs.diagonal"),
    ("energy/model/sand_rigid", "SandGrainCondenseRowArgs.force"),
    ("energy/model/sand_rigid", "SandGrainRecoverRowArgs.inverse_center_inertia"),
    ("energy/model/sand_rigid", "SandGrainRecoverRowArgs.angular"),
    ("energy/model/sand_rigid", "SandGrainRecoverRowArgs.coupling"),
    ("energy/model/sand_rigid", "SandGrainRecoverRowArgs.rotational_gradient"),
    ("energy/model/sand_rigid", "SandGrainRecoverRowArgs.previous_omega"),
    ("energy/model/sand_rigid", "SandGrainRecoverRowArgs.increment"),
    ("energy/model/sand_rigid", "SandGrainRecoverRowArgs.omega"),
    // THE PDRD REDUCED SOLVE'S EIGHTEEN ROWS, every buffer field. The whole
    // solve runs inside the Newton loop, so a host pointer here would be a
    // download per dispatch on a GPU backend.
    ("energy/model/pdrd_rigid", "PdrdCopyStateRotationRowArgs.state"),
    ("energy/model/pdrd_rigid", "PdrdCopyStateRotationRowArgs.rotation"),
    ("energy/model/pdrd_rigid", "PdrdComposeRunningRotationRowArgs.running"),
    ("energy/model/pdrd_rigid", "PdrdComposeRunningRotationRowArgs.dtheta"),
    ("energy/model/pdrd_rigid", "PdrdProlongRowArgs.vertex_body"),
    ("energy/model/pdrd_rigid", "PdrdProlongRowArgs.cloth_offset"),
    ("energy/model/pdrd_rigid", "PdrdProlongRowArgs.rotated_rest"),
    ("energy/model/pdrd_rigid", "PdrdProlongRowArgs.reduced"),
    ("energy/model/pdrd_rigid", "PdrdProlongRowArgs.full"),
    ("energy/model/pdrd_rigid", "PdrdRestrictRowArgs.vertex_body"),
    ("energy/model/pdrd_rigid", "PdrdRestrictRowArgs.cloth_offset"),
    ("energy/model/pdrd_rigid", "PdrdRestrictRowArgs.rotated_rest"),
    ("energy/model/pdrd_rigid", "PdrdRestrictRowArgs.full"),
    ("energy/model/pdrd_rigid", "PdrdRestrictRowArgs.cloth_out"),
    ("energy/model/pdrd_rigid", "PdrdRestrictRowArgs.reduced"),
    ("energy/model/pdrd_rigid", "PdrdSeedRestrictRowArgs.vertex_body"),
    ("energy/model/pdrd_rigid", "PdrdSeedRestrictRowArgs.cloth_offset"),
    ("energy/model/pdrd_rigid", "PdrdSeedRestrictRowArgs.full"),
    ("energy/model/pdrd_rigid", "PdrdSeedRestrictRowArgs.reduced"),
    ("energy/model/pdrd_rigid", "PdrdCopyProjectedClothRowArgs.vertex_body"),
    ("energy/model/pdrd_rigid", "PdrdCopyProjectedClothRowArgs.cloth_offset"),
    ("energy/model/pdrd_rigid", "PdrdCopyProjectedClothRowArgs.full"),
    ("energy/model/pdrd_rigid", "PdrdCopyProjectedClothRowArgs.reduced"),
    ("energy/model/pdrd_rigid", "PdrdTranslationLockParticularRowArgs.body_lock"),
    ("energy/model/pdrd_rigid", "PdrdTranslationLockParticularRowArgs.joint_mode"),
    ("energy/model/pdrd_rigid", "PdrdTranslationLockParticularRowArgs.locks"),
    ("energy/model/pdrd_rigid", "PdrdTranslationLockParticularRowArgs.drift"),
    ("energy/model/pdrd_rigid", "PdrdTranslationLockParticularRowArgs.reduced"),
    ("energy/model/pdrd_rigid", "PdrdExtractBodyRotationRowArgs.reduced"),
    ("energy/model/pdrd_rigid", "PdrdExtractBodyRotationRowArgs.rotation_out"),
    ("energy/model/pdrd_rigid", "PdrdScatterRotatedRestRowArgs.vert_list"),
    ("energy/model/pdrd_rigid", "PdrdScatterRotatedRestRowArgs.prop"),
    ("energy/model/pdrd_rigid", "PdrdScatterRotatedRestRowArgs.state"),
    ("energy/model/pdrd_rigid", "PdrdScatterRotatedRestRowArgs.rest_centered"),
    ("energy/model/pdrd_rigid", "PdrdScatterRotatedRestRowArgs.rotated_rest"),
    ("energy/model/pdrd_rigid", "PdrdPrecondBodyRowArgs.factor"),
    ("energy/model/pdrd_rigid", "PdrdPrecondBodyRowArgs.residual"),
    ("energy/model/pdrd_rigid", "PdrdPrecondBodyRowArgs.out"),
    ("energy/model/pdrd_rigid", "PdrdPrecondClothRowArgs.vertex_body"),
    ("energy/model/pdrd_rigid", "PdrdPrecondClothRowArgs.cloth_offset"),
    ("energy/model/pdrd_rigid", "PdrdPrecondClothRowArgs.inverse_diagonal"),
    ("energy/model/pdrd_rigid", "PdrdPrecondClothRowArgs.residual"),
    ("energy/model/pdrd_rigid", "PdrdPrecondClothRowArgs.out"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyCentroidRowArgs.vert_list"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyCentroidRowArgs.prop"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyCentroidRowArgs.body_prop"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyCentroidRowArgs.positions"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyCentroidRowArgs.centroid"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyWriteRowArgs.vert_list"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyWriteRowArgs.prop"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyWriteRowArgs.body_prop"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyWriteRowArgs.positions"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyWriteRowArgs.centroid"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyWriteRowArgs.running_rotation"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyWriteRowArgs.rest_centered"),
    ("energy/model/pdrd_rigid", "PdrdRigidifyWriteRowArgs.out"),
    ("energy/model/pdrd_rigid", "PdrdFitCentroidRowArgs.vert_list"),
    ("energy/model/pdrd_rigid", "PdrdFitCentroidRowArgs.prop"),
    ("energy/model/pdrd_rigid", "PdrdFitCentroidRowArgs.body_prop"),
    ("energy/model/pdrd_rigid", "PdrdFitCentroidRowArgs.positions"),
    ("energy/model/pdrd_rigid", "PdrdFitCentroidRowArgs.scratch"),
    ("energy/model/pdrd_rigid", "PdrdFitCovarianceRowArgs.vert_list"),
    ("energy/model/pdrd_rigid", "PdrdFitCovarianceRowArgs.prop"),
    ("energy/model/pdrd_rigid", "PdrdFitCovarianceRowArgs.body_prop"),
    ("energy/model/pdrd_rigid", "PdrdFitCovarianceRowArgs.positions"),
    ("energy/model/pdrd_rigid", "PdrdFitCovarianceRowArgs.rest_centered"),
    ("energy/model/pdrd_rigid", "PdrdFitCovarianceRowArgs.centroid"),
    ("energy/model/pdrd_rigid", "PdrdFitCovarianceRowArgs.scratch"),
    ("energy/model/pdrd_rigid", "PdrdFitFinishRowArgs.vert_list"),
    ("energy/model/pdrd_rigid", "PdrdFitFinishRowArgs.body_prop"),
    ("energy/model/pdrd_rigid", "PdrdFitFinishRowArgs.positions"),
    ("energy/model/pdrd_rigid", "PdrdFitFinishRowArgs.scratch"),
    ("energy/model/pdrd_rigid", "PdrdFitFinishRowArgs.state"),
    ("energy/model/pdrd_rigid", "PdrdAssembleInertiaRowArgs.state"),
    ("energy/model/pdrd_rigid", "PdrdAssembleInertiaRowArgs.blocks"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.vert_list"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.prop"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.state"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.rest_centered"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.dyn_index"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.dyn_offset"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.dyn_value"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.fixed_index"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.fixed_offset"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.fixed_value"),
    ("energy/model/pdrd_rigid", "PdrdAssembleSandwichRowArgs.blocks"),
    // The body-DOF projector, which every reduced vector passes through.
    ("energy/model/pdrd_lock_projector", "PdrdProjectBodyDofsRowArgs.joint_mode"),
    ("energy/model/pdrd_lock_projector", "PdrdProjectBodyDofsRowArgs.joint_axis"),
    ("energy/model/pdrd_lock_projector", "PdrdProjectBodyDofsRowArgs.translation_lock"),
    ("energy/model/pdrd_lock_projector", "PdrdProjectBodyDofsRowArgs.translation_axis"),
    ("energy/model/pdrd_lock_projector", "PdrdProjectBodyDofsRowArgs.translation_mode"),
    ("energy/model/pdrd_lock_projector", "PdrdProjectBodyDofsRowArgs.rotation_lock"),
    ("energy/model/pdrd_lock_projector", "PdrdProjectBodyDofsRowArgs.rotation_axis"),
    ("energy/model/pdrd_lock_projector", "PdrdProjectBodyDofsRowArgs.rotation_mode"),
    ("energy/model/pdrd_lock_projector", "PdrdProjectBodyDofsRowArgs.reduced"),
    // THE MORTON SORT'S TWO ARRAYS. The sort runs once per tree per step and the
    // reference runs it ON THE DEVICE, so a host pointer here would be the sort
    // itself relocated, which is what rule (1a-0) forbids rather than a readback
    // it permits.
    ("lbvh/bitonic", "BitonicStepArgs.key"),
    ("lbvh/bitonic", "BitonicStepArgs.index"),
    // THE AGGREGATE LOCK'S ELEVEN ROWS, every buffer field of every record. The
    // projector runs inside the Newton solve, so a host pointer here would be a
    // download per dispatch on a GPU backend and would put the residency
    // ratchet back above zero.
    ("solver/translation_lock_check", "TranslationLockDriftRowArgs.lock_index"),
    ("solver/translation_lock_check", "TranslationLockDriftRowArgs.locks"),
    ("solver/translation_lock_check", "TranslationLockDriftRowArgs.positions"),
    ("solver/translation_lock_check", "TranslationLockDriftRowArgs.initial"),
    ("solver/translation_lock_check", "TranslationLockDriftRowArgs.prop"),
    ("solver/translation_lock_check", "TranslationLockDriftRowArgs.drift"),
    ("solver/translation_lock_check", "TranslationLockDriftRowArgs.max_displacement"),
    ("solver/translation_lock_frames", "LockFrameClearRowArgs.frames"),
    ("solver/translation_lock_frames", "LockFrameCenterOfMassRowArgs.locks"),
    ("solver/translation_lock_frames", "LockFrameCenterOfMassRowArgs.mass_weighted_sum"),
    ("solver/translation_lock_frames", "LockFrameCenterOfMassRowArgs.frames"),
    ("solver/translation_lock_frames", "LockCenterOfMassAccumulateRowArgs.lock_index"),
    ("solver/translation_lock_frames", "LockCenterOfMassAccumulateRowArgs.locks"),
    ("solver/translation_lock_frames", "LockCenterOfMassAccumulateRowArgs.prop"),
    ("solver/translation_lock_frames", "LockCenterOfMassAccumulateRowArgs.positions"),
    ("solver/translation_lock_frames", "LockCenterOfMassAccumulateRowArgs.mass_weighted_sum"),
    ("solver/translation_lock_frames", "LockInertiaAccumulateRowArgs.lock_index"),
    ("solver/translation_lock_frames", "LockInertiaAccumulateRowArgs.locks"),
    ("solver/translation_lock_frames", "LockInertiaAccumulateRowArgs.prop"),
    ("solver/translation_lock_frames", "LockInertiaAccumulateRowArgs.positions"),
    ("solver/translation_lock_frames", "LockInertiaAccumulateRowArgs.frames"),
    ("solver/translation_lock_frames", "LockInertiaAccumulateRowArgs.inertia"),
    ("solver/translation_lock_rows", "LockRowSumsAccumulateRowArgs.lock_index"),
    ("solver/translation_lock_rows", "LockRowSumsAccumulateRowArgs.locks"),
    ("solver/translation_lock_rows", "LockRowSumsAccumulateRowArgs.prop"),
    ("solver/translation_lock_rows", "LockRowSumsAccumulateRowArgs.positions"),
    ("solver/translation_lock_rows", "LockRowSumsAccumulateRowArgs.dof_mask"),
    ("solver/translation_lock_rows", "LockRowSumsAccumulateRowArgs.frames"),
    ("solver/translation_lock_rows", "LockRowSumsAccumulateRowArgs.values"),
    ("solver/translation_lock_rows", "LockRowSumsAccumulateRowArgs.sums"),
    ("solver/translation_lock_rows", "LockRefineTowardRhsRowArgs.lock_index"),
    ("solver/translation_lock_rows", "LockRefineTowardRhsRowArgs.locks"),
    ("solver/translation_lock_rows", "LockRefineTowardRhsRowArgs.prop"),
    ("solver/translation_lock_rows", "LockRefineTowardRhsRowArgs.positions"),
    ("solver/translation_lock_rows", "LockRefineTowardRhsRowArgs.dof_mask"),
    ("solver/translation_lock_rows", "LockRefineTowardRhsRowArgs.frames"),
    ("solver/translation_lock_rows", "LockRefineTowardRhsRowArgs.sums"),
    ("solver/translation_lock_rows", "LockRefineTowardRhsRowArgs.values"),
    ("solver/translation_lock_rows", "LockProjectOutRowsRowArgs.lock_index"),
    ("solver/translation_lock_rows", "LockProjectOutRowsRowArgs.locks"),
    ("solver/translation_lock_rows", "LockProjectOutRowsRowArgs.prop"),
    ("solver/translation_lock_rows", "LockProjectOutRowsRowArgs.positions"),
    ("solver/translation_lock_rows", "LockProjectOutRowsRowArgs.dof_mask"),
    ("solver/translation_lock_rows", "LockProjectOutRowsRowArgs.frames"),
    ("solver/translation_lock_rows", "LockProjectOutRowsRowArgs.sums"),
    ("solver/translation_lock_rows", "LockProjectOutRowsRowArgs.values"),
    ("solver/translation_lock_rows", "LockSeedFreeSolutionRowArgs.lock_index"),
    ("solver/translation_lock_rows", "LockSeedFreeSolutionRowArgs.locks"),
    ("solver/translation_lock_rows", "LockSeedFreeSolutionRowArgs.prop"),
    ("solver/translation_lock_rows", "LockSeedFreeSolutionRowArgs.positions"),
    ("solver/translation_lock_rows", "LockSeedFreeSolutionRowArgs.dof_mask"),
    ("solver/translation_lock_rows", "LockSeedFreeSolutionRowArgs.frames"),
    ("solver/translation_lock_rows", "LockSeedFreeSolutionRowArgs.seed"),
    ("solver/translation_lock_rows", "LockSeedFreeSolutionRowArgs.values"),
    ("solver/translation_lock_rows", "LockTorqueAccumulateRowArgs.lock_index"),
    ("solver/translation_lock_rows", "LockTorqueAccumulateRowArgs.locks"),
    ("solver/translation_lock_rows", "LockTorqueAccumulateRowArgs.prop"),
    ("solver/translation_lock_rows", "LockTorqueAccumulateRowArgs.positions"),
    ("solver/translation_lock_rows", "LockTorqueAccumulateRowArgs.frames"),
    ("solver/translation_lock_rows", "LockTorqueAccumulateRowArgs.step"),
    ("solver/translation_lock_rows", "LockTorqueAccumulateRowArgs.torque"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.lock_index"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.locks"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.prop"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.positions"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.initial"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.dof_mask"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.frames"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.seed"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.drift"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.fixed"),
    ("solver/translation_lock_rows", "LockConstraintAssembleRowArgs.gram"),
    ("solver/pcg", "PcgAlphaTermsArgs.value_out"),
    ("solver/pcg", "PcgRigidGroupL1Args.reduced"),
    ("solver/pcg", "PcgRigidGroupL1Args.norm"),
    ("solver/pcg", "PcgAlphaTermsArgs.noise_out"),
    ("solver/pcg", "PcgAlphaTermsArgs.cause_out"),
    ("solver/pcg", "PcgBetaTermsArgs.value_out"),
    ("solver/pcg", "PcgBetaTermsArgs.noise_out"),
    ("solver/pcg", "PcgBetaTermsArgs.cause_out"),
    // The Schwarz domain construction. Every one of these is device-resident
    // from the start, which is the easy direction: a field is a handle unless
    // something on the host reads the buffer, and nothing here does. The
    // counting and the claim both happen in a kernel, and the scan between them
    // is `dyn_build_offsets_pass`, which is a kernel too.
    // THE SCAN'S THREE ENTRIES. Every array they touch is device resident by
    // construction: the offsets they scan are the CSR's own, and the block sums
    // are a device scratch that no host code reads.
    // THE TREE'S PARENT LINKS, ROOT AND LEVELS. Every array here is device
    // resident: the node array the passes read is the tree's own, and what the
    // host reads back is the root and the per-level counts rather than any of
    // these.
    ("lbvh/lbvh", "LbvhSetParentArgs.nodes"),
    ("lbvh/lbvh", "LbvhSetParentArgs.parent"),
    ("lbvh/lbvh", "LbvhFindRootArgs.parent"),
    ("lbvh/lbvh", "LbvhFindRootArgs.root"),
    ("lbvh/lbvh", "LbvhFindRootArgs.found"),
    ("lbvh/lbvh", "LbvhCountLevelsArgs.depth"),
    ("lbvh/lbvh", "LbvhCountLevelsArgs.counts"),
    ("lbvh/lbvh", "LbvhScatterLevelsArgs.depth"),
    ("lbvh/lbvh", "LbvhScatterLevelsArgs.level_offset"),
    ("lbvh/lbvh", "LbvhScatterLevelsArgs.cursor"),
    ("lbvh/lbvh", "LbvhScatterLevelsArgs.level_data"),
    ("lbvh/lbvh", "LbvhNodeDepthArgs.root"),
    // THE CENTROID BOUNDS, device resident: `bounds_leaf` and `bounds_merge`
    // reduce them and the Morton pass reads them, with no host step between.
    ("lbvh/lbvh", "LbvhMortonFromBoundsArgs.bounds"),
    ("plasticity/plasticity", "PlasticityCommitFaceArgs.plasticity"),
    ("plasticity/plasticity", "PlasticityCommitFaceArgs.changed"),
    ("plasticity/plasticity", "PlasticityCommitFaceArgs.inverse_rest"),
    ("plasticity/plasticity", "PlasticityCommitFaceArgs.destination"),
    ("plasticity/plasticity", "PlasticityCommitTetArgs.plasticity"),
    ("plasticity/plasticity", "PlasticityCommitTetArgs.changed"),
    ("plasticity/plasticity", "PlasticityCommitTetArgs.inverse_rest"),
    ("plasticity/plasticity", "PlasticityCommitTetArgs.destination"),
    ("primitives/reduce_scalar", "ReduceMinLeafArgs.values"),
    ("primitives/reduce_scalar", "ReduceMinLeafArgs.out"),
    ("primitives/reduce_scalar", "ReduceMaxLeafArgs.values"),
    ("primitives/reduce_scalar", "ReduceMaxLeafArgs.out"),
    // THE UNSIGNED MINIMUM AND TOTAL, device resident: each ladder reads what
    // the pass before it wrote, and the host reads only the result words.
    ("primitives/reduce_scalar", "ReduceMinU32LeafArgs.values"),
    ("primitives/reduce_scalar", "ReduceMinU32LeafArgs.out"),
    ("primitives/reduce_scalar", "ReduceSumU32LeafArgs.values"),
    ("primitives/reduce_scalar", "ReduceSumU32LeafArgs.out"),
    ("primitives/reduce_scalar", "ReduceSumWideMergeArgs.source"),
    ("primitives/reduce_scalar", "ReduceSumWideMergeArgs.destination"),
    // THE OVERLAP SELECTION'S LEAF, device resident on the same terms. It only
    // reads the report arrays, which the line search's own reader still
    // downloads whenever a sweep has flagged a start.
    ("contact/ccd_sweep", "OverlapFirstFlaggedLeafArgs.overlap"),
    ("contact/ccd_sweep", "OverlapFirstFlaggedLeafArgs.out"),
    ("primitives/reduce_bounds", "BoundsLeafArgs.cx"),
    ("primitives/reduce_bounds", "BoundsLeafArgs.cy"),
    ("primitives/reduce_bounds", "BoundsLeafArgs.cz"),
    ("primitives/reduce_bounds", "BoundsLeafArgs.out"),
    ("primitives/reduce_bounds", "BoundsMergeArgs.source"),
    ("primitives/reduce_bounds", "BoundsMergeArgs.destination"),
    ("primitives/scan_levels", "ScanBlockTotalArgs.data"),
    ("primitives/scan_levels", "ScanBlockTotalArgs.total"),
    ("primitives/scan_levels", "ScanBlockApplyArgs.data"),
    ("primitives/scan_levels", "ScanBlockApplyArgs.base"),
    ("primitives/scan_levels", "ScanZeroArgs.data"),
    ("primitives/vec_ops", "VecFillU32Args.array"),
    ("schwarz/schwarz", "SchwarzCountMembersArgs.aggregate"),
    ("schwarz/schwarz", "SchwarzCountMembersArgs.offset"),
    ("schwarz/schwarz", "SchwarzScatterMembersArgs.aggregate"),
    ("schwarz/schwarz", "SchwarzScatterMembersArgs.offset"),
    ("schwarz/schwarz", "SchwarzScatterMembersArgs.cursor"),
    ("schwarz/schwarz", "SchwarzScatterMembersArgs.members"),
    ("schwarz/schwarz", "SchwarzDomainInverseSizeArgs.offset"),
    ("schwarz/schwarz", "SchwarzDomainInverseSizeArgs.inverse_offset"),
    ("schwarz/schwarz", "SchwarzFineGraphCountArgs.dynamic_offset"),
    ("schwarz/schwarz", "SchwarzFineGraphCountArgs.reference_offset"),
    ("schwarz/schwarz", "SchwarzFineGraphCountArgs.fixed_offset"),
    ("schwarz/schwarz", "SchwarzFineGraphCountArgs.transpose_offset"),
    ("schwarz/schwarz", "SchwarzFineGraphCountArgs.count"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.dynamic_index"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.dynamic_value"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.dynamic_offset"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.reference_index"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.reference_value"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.reference_offset"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.global_value"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.fixed_index"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.fixed_offset"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.fixed_value"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.transpose_pair"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.transpose_offset"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.graph_offset"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.column"),
    ("schwarz/schwarz", "SchwarzFineGraphFillArgs.weight"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.aggregate_offset"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.members"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.dense_offset"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.dynamic_index"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.dynamic_value"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.dynamic_offset"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.reference_index"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.reference_value"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.reference_offset"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.global_value"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.fixed_index"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.fixed_offset"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.fixed_value"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.transpose_pair"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.transpose_offset"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.diagonal"),
    ("schwarz/schwarz", "SchwarzFactorGatherArgs.dense"),
    ("schwarz/schwarz", "SchwarzFactorFloorArgs.aggregate_offset"),
    ("schwarz/schwarz", "SchwarzFactorFloorArgs.dense_offset"),
    ("schwarz/schwarz", "SchwarzFactorFloorArgs.dense"),
    ("schwarz/schwarz", "SchwarzFactorFloorArgs.floor_out"),
    ("schwarz/schwarz", "SchwarzFactorCholeskyDiagonalArgs.aggregate_offset"),
    ("schwarz/schwarz", "SchwarzFactorCholeskyDiagonalArgs.dense_offset"),
    ("schwarz/schwarz", "SchwarzFactorCholeskyDiagonalArgs.floor_in"),
    ("schwarz/schwarz", "SchwarzFactorCholeskyDiagonalArgs.dense"),
    ("schwarz/schwarz", "SchwarzFactorCholeskyColumnArgs.aggregate_offset"),
    ("schwarz/schwarz", "SchwarzFactorCholeskyColumnArgs.dense_offset"),
    ("schwarz/schwarz", "SchwarzFactorCholeskyColumnArgs.dense"),
    ("schwarz/schwarz", "SchwarzFactorInverseColumnArgs.aggregate_offset"),
    ("schwarz/schwarz", "SchwarzFactorInverseColumnArgs.dense_offset"),
    ("schwarz/schwarz", "SchwarzFactorInverseColumnArgs.dense"),
    ("schwarz/schwarz", "SchwarzFactorInverseColumnArgs.work"),
    ("schwarz/schwarz", "SchwarzFactorPackArgs.aggregate_offset"),
    ("schwarz/schwarz", "SchwarzFactorPackArgs.dense_offset"),
    ("schwarz/schwarz", "SchwarzFactorPackArgs.inverse_offset"),
    ("schwarz/schwarz", "SchwarzFactorPackArgs.work"),
    ("schwarz/schwarz", "SchwarzFactorPackArgs.packed"),
    ("schwarz/schwarz", "SchwarzApplyGatherArgs.aggregate_offset"),
    ("schwarz/schwarz", "SchwarzApplyGatherArgs.members"),
    ("schwarz/schwarz", "SchwarzApplyGatherArgs.x"),
    ("schwarz/schwarz", "SchwarzApplyGatherArgs.residual_local"),
    ("schwarz/schwarz", "SchwarzApplyLowerArgs.aggregate_offset"),
    ("schwarz/schwarz", "SchwarzApplyLowerArgs.inverse_offset"),
    ("schwarz/schwarz", "SchwarzApplyLowerArgs.packed"),
    ("schwarz/schwarz", "SchwarzApplyLowerArgs.residual_local"),
    ("schwarz/schwarz", "SchwarzApplyLowerArgs.y_local"),
    ("schwarz/schwarz", "SchwarzApplyUpperArgs.aggregate_offset"),
    ("schwarz/schwarz", "SchwarzApplyUpperArgs.members"),
    ("schwarz/schwarz", "SchwarzApplyUpperArgs.inverse_offset"),
    ("schwarz/schwarz", "SchwarzApplyUpperArgs.packed"),
    ("schwarz/schwarz", "SchwarzApplyUpperArgs.y_local"),
    ("schwarz/schwarz", "SchwarzApplyUpperArgs.result"),
    ("schwarz/schwarz", "SchwarzRestrictRowArgs.map_fine"),
    ("schwarz/schwarz", "SchwarzRestrictRowArgs.x"),
    ("schwarz/schwarz", "SchwarzRestrictRowArgs.coarse"),
    ("schwarz/schwarz", "SchwarzProlongRowArgs.map_fine"),
    ("schwarz/schwarz", "SchwarzProlongRowArgs.coarse"),
    ("schwarz/schwarz", "SchwarzProlongRowArgs.z"),
    ("schwarz/schwarz", "SchwarzComposeMapRowArgs.previous_map"),
    ("schwarz/schwarz", "SchwarzComposeMapRowArgs.previous_aggregate"),
    ("schwarz/schwarz", "SchwarzComposeMapRowArgs.map_fine"),
    ("schwarz/schwarz", "SchwarzLevel0CountArgs.dynamic_offset"),
    ("schwarz/schwarz", "SchwarzLevel0CountArgs.reference_offset"),
    ("schwarz/schwarz", "SchwarzLevel0CountArgs.fixed_offset"),
    ("schwarz/schwarz", "SchwarzLevel0CountArgs.transpose_offset"),
    ("schwarz/schwarz", "SchwarzLevel0CountArgs.count"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.dynamic_index"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.dynamic_value"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.dynamic_offset"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.reference_index"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.reference_value"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.reference_offset"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.global_value"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.fixed_index"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.fixed_offset"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.fixed_value"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.transpose_pair"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.transpose_offset"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.diagonal"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.coarse_offset"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.column"),
    ("schwarz/schwarz", "SchwarzLevel0FillArgs.value"),
    ("schwarz/schwarz", "SchwarzCoarseGatherArgs.aggregate_offset"),
    ("schwarz/schwarz", "SchwarzCoarseGatherArgs.members"),
    ("schwarz/schwarz", "SchwarzCoarseGatherArgs.dense_offset"),
    ("schwarz/schwarz", "SchwarzCoarseGatherArgs.matrix_offset"),
    ("schwarz/schwarz", "SchwarzCoarseGatherArgs.matrix_column"),
    ("schwarz/schwarz", "SchwarzCoarseGatherArgs.matrix_value"),
    ("schwarz/schwarz", "SchwarzCoarseGatherArgs.dense"),
    ("schwarz/schwarz", "SchwarzGalerkinKeyArgs.aggregate"),
    ("schwarz/schwarz", "SchwarzGalerkinKeyArgs.offset"),
    ("schwarz/schwarz", "SchwarzGalerkinKeyArgs.column"),
    ("schwarz/schwarz", "SchwarzGalerkinKeyArgs.key"),
    ("schwarz/schwarz", "SchwarzGalerkinKeyArgs.permutation"),
    ("schwarz/schwarz", "SchwarzGalerkinEdgeFlagArgs.key"),
    ("schwarz/schwarz", "SchwarzGalerkinEdgeFlagArgs.edge"),
    ("schwarz/schwarz", "SchwarzGalerkinEdgeHeadArgs.key"),
    ("schwarz/schwarz", "SchwarzGalerkinEdgeHeadArgs.edge"),
    ("schwarz/schwarz", "SchwarzGalerkinEdgeHeadArgs.column"),
    ("schwarz/schwarz", "SchwarzGalerkinEdgeHeadArgs.edge_start"),
    ("schwarz/schwarz", "SchwarzGalerkinEdgeHeadArgs.row_count"),
    ("schwarz/schwarz", "SchwarzGalerkinSegmentSumArgs.edge_start"),
    ("schwarz/schwarz", "SchwarzGalerkinSegmentSumArgs.permutation"),
    ("schwarz/schwarz", "SchwarzGalerkinSegmentSumArgs.source_value"),
    ("schwarz/schwarz", "SchwarzGalerkinSegmentSumArgs.value"),
    ("csrmat/dynamic_csr", "DynScatterTransposePassArgs.row_offset"),
    ("csrmat/dynamic_csr", "DynScatterTransposePassArgs.index"),
    ("csrmat/dynamic_csr", "DynCountTransposePassArgs.row_offset"),
    ("csrmat/dynamic_csr", "DynCountTransposePassArgs.index"),
    ("main/dirichlet", "DirichletLiftRowArgs.offset"),
    ("main/dirichlet", "DirichletLiftRowArgs.column"),
    ("main/dirichlet", "DirichletLiftRowArgs.value"),
    ("main/dirichlet", "DirichletLiftRowArgs.eval_x"),
    ("main/dirichlet", "DirichletLiftRowArgs.target"),
    ("main/dirichlet", "DirichletPrescribeGatedArgs.eval_x"),
    ("main/dirichlet", "DirichletPrescribeGatedArgs.target"),
    ("main/dx_seed", "DxSeedArgs.eval_x"),
    ("contact/aabb", "AabbEdgeScanQueryMaskedArgs.vert"),
    ("contact/aabb", "AabbEdgeScanQueryArgs.vert"),
    ("contact/aabb", "AabbVertexScanQueryMaskedArgs.vert"),
    ("contact/aabb", "AabbVertexScanQueryArgs.vert"),
    ("lbvh/lbvh", "FaceCentroidArgs.vert"),
    ("lbvh/lbvh", "EdgeCentroidArgs.vert"),
    ("lbvh/lbvh", "VertexCentroidArgs.vert"),
    ("contact/aabb", "AabbPointContactQueryMaskedArgs.x"),
    ("contact/aabb", "AabbPointContactQueryArgs.x"),
    ("contact/aabb", "AabbEdgeContactQueryMaskedArgs.x"),
    ("contact/aabb", "AabbEdgeContactQueryArgs.x"),
    // `state.tet.gradient_f` and `state.tet.hessian_f`: the tet elastic force
    // and Hessian in the deformation-gradient basis. Written by the spectral
    // stage and read by the converter, with no host access in between, and
    // every record naming either is generated. That last part is what decides
    // the order rather than the size: a HAND-WRITTEN entry point takes flat
    // pointers, so its shim can be handed an address and cannot be handed a
    // handle, and a buffer one of them names cannot move until its entry point
    // is generated.
    ("eigenanalysis/tet_eigenanalysis", "TetSpectralForceArgs.force"),
    ("eigenanalysis/tet_eigenanalysis", "TetSpectralHessianArgs.hessian"),
    // THE TET SPECTRAL CHAIN'S INTERMEDIATES: the reflection-corrected SVD's
    // `U` and `V^T`, and the material diff table's two derivative arrays. Each
    // is written by one dispatch and read by the next with no host access
    // between them, and every record naming any of them is generated.
    //
    // THE SINGULAR VALUES ARE HERE TOO, so the three arrays one factorization
    // writes now move together. They were held back while
    // `PlasticityCreepSingular3Args` was a hand-written entry point: a shim
    // there takes a flat pointer and has nothing to resolve a handle against,
    // and a field is one type in the Rust twin, so `Svd3x3RvArgs.sigma` could
    // not move while that call site stood. Generating that entry point is what
    // released the whole component.
    //
    // THE FACE ANALOGUE IS RELEASED THE SAME WAY. It was blocked by
    // `ShellStrainRestoreSigmaArgs` and `ShellStretchTermsArgs`, which name
    // `state.face.svd_sigma` and were both hand-written; generating them freed
    // the component exactly as generating `PlasticityCreepSingular3Args` freed
    // the tet one. It is a THREE-buffer component, and the two fields that
    // chain it are worth naming: `Svd3x2Args.sigma` is dispatched over both
    // `state.face.svd_sigma` and `state.stretch.sigma`, and
    // `FaceSpectralHessianArgs.sigma` over both `state.face.svd_sigma` and
    // `state.face_strain.restored_sigma`, so no two of the three could have
    // moved apart from the third.
    ("utility/svd3x2", "Svd3x2Args.sigma"),
    ("utility/svd3x2", "ShellStrainRestoreSigmaArgs.restored"),
    ("energy/model/material_diff_table", "FaceMaterialDiffTableArgs.sigma"),
    ("eigenanalysis/face_eigenanalysis", "FaceSpectralHessianArgs.sigma"),
    ("utility/face_deformation", "ShellStretchTermsArgs.sigma"),
    // `state.rod_strain.strain`: the per-rod strain reading, written by the
    // gated force-and-Hessian pass and read by the stiffness pass, with no host
    // access between them.
    ("strainlimiting/rod_strain", "RodStrainForceHessianGatedArgs.strain"),
    ("strainlimiting/rod_strain", "RodStrainStiffnessGatedArgs.strain"),
    // `state.plastic.face_svd_sigma` joins the same component: the face creep
    // chain dispatches `Svd3x2Args.sigma` over it, which is the field that
    // already spans `state.face.svd_sigma` and `state.stretch.sigma`. It is
    // reached through a BARE LOCAL (`let p = &mut state.plastic`), so a scan
    // keyed on a `state.` path does not see it at all.
    ("plasticity/plasticity", "PlasticityCreepSingular2Args.singular_values"),
    // `state.plastic.hinge_angle` and `state.plastic.rod_angle`: the crept
    // dihedral and turning angles. Written by the bending passes and read by
    // the creep pass; the VERDICT beside them still comes back to the host,
    // and the angles themselves no longer do.
    ("plasticity/plasticity", "PlasticityCreepRestAngleArgs.angle"),
    ("energy/model/shell_bend", "ShellBendAngleArgs.angle"),
    ("energy/model/rod_bend", "RodBendAngleArgs.angle"),
    // `state.stitch.weight` and `state.stitch.stiffness`: written once per
    // constraint update from the incoming records and read only by kernels
    // afterwards, which is what a `StagedBuffer` is for. They are the first
    // host-POPULATED buffers to move; every field above was written by a
    // kernel to begin with, so a plain `Buffer` served it.
    ("energy/model/stitch", "StitchForceHessianGatheredArgs.stitch_weight"),
    ("energy/model/stitch", "StitchForceHessianGatheredArgs.stiffness"),
    // The rest of the stitch inputs. The slot indices are host-written AND
    // host-read (the slot bound check, and the CSR push that reads a row and
    // column out of them), which a `StagedBuffer` serves: `host()` answers off
    // the copy the host itself wrote and does not dirty it.
    ("energy/model/stitch", "StitchForceHessianGatheredArgs.stitch_index"),
    ("energy/model/stitch", "StitchForceHessianGatheredArgs.length_factor"),
    ("energy/model/stitch", "StitchForceHessianGatheredArgs.vertex_ghat"),
    ("energy/model/stitch", "StitchForceHessianGatheredArgs.vertex_offset"),
    ("utility/stitch_scatter", "StitchAtomicEmbedForceArgs.index"),
    // The stitch force and Hessian, which the kernel WRITES and the host then
    // reads: the finiteness scan, and the push that folds the 18x18 into the
    // fixed matrix. That is a `ReadbackBuffer`, whose `host()` refuses between
    // a handle going out and the download, so neither reader can answer out of
    // the previous iteration. The force is also read BACK by the scatter, which
    // is why it is listed twice.
    ("energy/model/stitch", "StitchForceHessianGatheredArgs.gradient"),
    ("energy/model/stitch", "StitchForceHessianGatheredArgs.hessian"),
    ("utility/stitch_scatter", "StitchAtomicEmbedForceArgs.gradient"),
    // The analytic collider sweep's infeasibility flags. The kernel clears
    // every slot before it tests, so nothing seeds them and the mirror IS the
    // answer. Its sibling `out_toi` cannot follow yet: the kernel READS it,
    // seeded by the caller at the line-search ceiling, and a seed needs a
    // device-side fill this driver has no dispatch for.
    ("contact/vertex_constraint", "VertexConstraintSweepArgs.out_infeasible"),
    // The rod strain limiter's per-rod verdict, the same shape `hinge.ok`
    // already had: the kernel writes it, the host walks the candidates and
    // keeps the ones it admitted.
    ("strainlimiting/rod_strain", "RodStrainForceHessianGatedArgs.ok"),
    // The stretch indicator's per-face shrink pair and per-rod rest length,
    // gathered on the host from the material set and read only by kernels.
    // The stretch terms read the face's own material now, as the reference
    // does, so the two host-filled shrink arrays are gone and the records they
    // named with them.
    ("utility/face_deformation", "ShellStretchTermsArgs.face"),
    ("utility/face_deformation", "ShellStretchTermsArgs.prop"),
    ("utility/face_deformation", "ShellStretchTermsArgs.face_param"),
    ("utility/face_deformation", "ShellStretchTermsArgs.vertex_prop"),
    ("main/stretch", "RodStretchRatioGatedArgs.prop"),
    // The rod strain limiter's gate arrays, the rod analogue of the face gate
    // and gathered by the same shape of loop. The gather is shared by the
    // assembly and the line search, so BOTH upload: an upload belongs on every
    // path that names a handle, not on the first one written.
    ("strainlimiting/rod_strain", "RodStrainForceHessianGatedArgs.limit"),
    ("strainlimiting/rod_strain", "RodStrainForceHessianGatedArgs.rest_length"),
    ("strainlimiting/rod_strain", "RodStrainStiffnessGatedArgs.limit"),
    ("strainlimiting/rod_strain", "RodStrainStiffnessGatedArgs.mass"),
    ("strainlimiting/strain_toi", "RodStrainToiGatedArgs.limit"),
    ("strainlimiting/strain_toi", "RodStrainToiGatedArgs.rest_length"),
    // The per-edge query boxes the broad phase walks. Written by the four
    // query kernels (contact and swept, each masked and not) and read by the
    // HOST BVH traversal, so the buffer carries a mirror and each launch
    // downloads: `candidates()` takes no device and cannot download for itself.
    // This is also the first `Aabb` to live on the device, which is what its
    // `Pod` impl beside the declaration is for.
    ("contact/aabb", "AabbEdgeContactQueryArgs.out"),
    ("contact/aabb", "AabbEdgeContactQueryMaskedArgs.out"),
    // `vec_fill` seeds a DEVICE buffer, so its one array is a handle. Nothing
    // else would be worth dispatching: the host can already clear a host slice.
    ("primitives/vec_ops", "VecFillArgs.array"),
    // And the sweep's time of impact, which is what the seed above was for.
    // It is the first buffer of the seed-then-reduce-into-then-read shape to
    // move: the host no longer touches it in either direction.
    ("contact/vertex_constraint", "VertexConstraintSweepArgs.out_toi"),
    // The intersection scan's query boxes, the same shape as the broad-phase
    // edge boxes: written by the four scan-query kernels and walked by the
    // host traversal, so both carry a mirror and both download before the walk.
    ("contact/aabb", "AabbEdgeScanQueryArgs.out"),
    ("contact/aabb", "AabbEdgeScanQueryMaskedArgs.out"),
    ("contact/aabb", "AabbVertexScanQueryArgs.out"),
    ("contact/aabb", "AabbVertexScanQueryMaskedArgs.out"),
    // The two strain limiters' times of impact. Their entries SCATTER a
    // value-returning body, so the dispatch writes every element and only the
    // no-candidate branch needs the device seed.
    ("strainlimiting/strain_toi", "ShellStrainToiGatedArgs.toi"),
    ("strainlimiting/strain_toi", "ShellStrainToiFromRecordsArgs.start"),
    ("strainlimiting/strain_toi", "ShellStrainToiFromRecordsArgs.finish"),
    ("strainlimiting/strain_toi", "ShellStrainToiFromRecordsArgs.face"),
    ("strainlimiting/strain_toi", "ShellStrainToiFromRecordsArgs.inverse_rest"),
    ("strainlimiting/strain_toi", "ShellStrainToiFromRecordsArgs.prop"),
    ("strainlimiting/strain_toi", "ShellStrainToiFromRecordsArgs.face_param"),
    ("strainlimiting/strain_toi", "ShellStrainToiFromRecordsArgs.toi"),
    ("strainlimiting/strain_toi", "RodStrainToiGatedArgs.toi"),
    // The per-rod stretch ratio, scattered by a value-returning body and
    // folded by the caller. Its shell sibling `stretch.ratio` cannot follow it:
    // that one is a `destination` of `ElementAddScaledArgs`, a field shared
    // across many call sites, so it moves with that whole component or not at
    // all.
    ("main/stretch", "RodStretchRatioGatedArgs.ratio"),
    // THE ELEMENT ASSEMBLY STAGING, one whole component. `ElementAddScaledArgs`
    // has three buffer fields and each is its own component, because the three
    // name disjoint buffer sets: this is the SOURCE half, the per-element force
    // and Hessian each model writes before its scale is applied. Thirteen of its
    // fourteen buffers are written and read only by kernels; the fourteenth,
    // `state.rod.stiffness`, is host-gathered and is staged.
    ("primitives/vec_ops", "ElementAddScaledArgs.source"),
    ("utility/face_convert", "FaceConvertForceArgs.force"),
    ("utility/face_convert", "FaceConvertHessianArgs.hessian"),
    ("utility/tet_convert", "TetConvertForceArgs.force"),
    ("utility/tet_convert", "TetConvertHessianArgs.hessian"),
    ("energy/model/rod_bend", "RodBendForceHessianArgs.force"),
    ("energy/model/rod_bend", "RodBendForceHessianArgs.hessian"),
    ("energy/model/shell_bend", "ShellBendForceHessianCheckedArgs.force"),
    ("energy/model/shell_bend", "ShellBendForceHessianCheckedArgs.hessian"),
    ("strainlimiting/rod_strain", "RodStrainForceHessianGatedArgs.force"),
    ("strainlimiting/rod_strain", "RodStrainForceHessianGatedArgs.hessian"),
    ("utility/face_deformation", "ShellStretchTermsArgs.largest"),
    // THE SCALE half of the same triple, and a component of its own because it
    // names a disjoint buffer set: the per-element masses and stiffnesses each
    // model scales its staged force and Hessian by. All three buffer types appear
    // in it, which is what the classification is for: the masses are
    // host-gathered and staged, the shell and rod BENDING stiffnesses are written
    // by a kernel and read by the host gate that selects active elements, and the
    // two strain-limiter stiffnesses and the shrink minimum are touched by
    // kernels alone.
    ("primitives/vec_ops", "ElementAddScaledArgs.scale"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessArgs.stiffness"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessAndDampingArgs.prop"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessAndDampingArgs.hinge_param"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessAndDampingArgs.kind"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessAndDampingArgs.areal_density"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessAndDampingArgs.stiffness"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessAndDampingArgs.damping"),
    ("energy/model/rod_bend_stiffness", "RodBendStiffnessArgs.stiffness"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessGatedArgs.stiffness"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessFromRecordsArgs.x"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.x"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.current"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.face"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.face_slots"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.inverse_rest"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.prop"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.face_param"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.reference_index"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.reference_offset"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.reference_value"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.force"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.fixed_index"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.fixed_offset"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.fixed_value"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.refused"),
    ("strainlimiting/shell_strain", "ShellStrainEmbedArgs.witness"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessFromRecordsArgs.face"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessFromRecordsArgs.face_slots"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessFromRecordsArgs.index"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessFromRecordsArgs.offset"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessFromRecordsArgs.value"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessFromRecordsArgs.prop"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessFromRecordsArgs.face_param"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessFromRecordsArgs.shifted_sigma"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessFromRecordsArgs.stiffness"),
    ("strainlimiting/rod_strain", "RodStrainStiffnessGatedArgs.stiffness"),
    ("utility/face_deformation", "ShellStretchTermsArgs.shrink_min"),
    // THE DESTINATION half, and the last of the three. These are the
    // accumulators every element model folds into: the per-element gradient and
    // Hessian, the two lagged damping Hessians and the stretch ratio. Each opens
    // at zero, which used to be a host `fill` and is now a `vec_fill` dispatch,
    // and each is read back by the host pass that scatters it into the matrix.
    // The damping records name the same arrays, so they convert with it.
    ("primitives/vec_ops", "ElementAddScaledArgs.destination"),
    ("utility/tet_damping", "TetDampingArgs.gradient"),
    ("utility/tet_damping", "TetDampingArgs.hessian"),
    ("utility/face_damping", "FaceDampingArgs.gradient"),
    ("utility/face_damping", "FaceDampingArgs.hessian"),
    ("utility/hinge_damping", "HingeDampingArgs.gradient"),
    ("utility/hinge_damping", "HingeDampingArgs.hessian"),
    ("utility/hinge_damping", "HingeDampingArgs.lagged"),
    ("utility/rod_bend_damping", "RodBendDampingArgs.gradient"),
    ("utility/rod_bend_damping", "RodBendDampingArgs.hessian"),
    ("utility/rod_bend_damping", "RodBendDampingArgs.lagged"),
    // The Dirichlet mask: which vertices had their per-vertex DOF eliminated.
    // Host-written once per step from the vertex props, host-SUMMED for the
    // one-time count, and read by the lift, the prescribe and the drag. A
    // `StagedBuffer` serves all three: the sum reads the copy `at()` just wrote.
    ("main/dirichlet", "DirichletLiftRowArgs.dof_mask"),
    ("main/dirichlet", "DirichletPrescribeGatedArgs.dof_mask"),
    ("main/fix_xz_drag", "FixXzDragArgs.dof_removed"),
    // The analytic collider pass's four per-vertex outputs. The kernel writes
    // them and the serial deposit below reads all four, so they refresh together
    // after the launch rather than one at a time.
    ("contact/vertex_constraint", "VertexConstraintArgs.force"),
    ("contact/vertex_constraint", "VertexConstraintArgs.diagonal"),
    ("contact/vertex_constraint", "VertexConstraintArgs.out_count"),
    // The centroid coordinates every BVH build sorts on. One allocation of the
    // widest element count serves all five passes, so each takes the PREFIX its
    // own pass writes through `ReadbackBuffer::span`: the whole-buffer handle
    // would compile and would widen the bound the entry checks, which is the
    // wrong direction for a conversion to move it.
    ("lbvh/lbvh", "FaceCentroidArgs.cx"),
    ("lbvh/lbvh", "FaceCentroidArgs.cy"),
    ("lbvh/lbvh", "FaceCentroidArgs.cz"),
    ("lbvh/lbvh", "EdgeCentroidArgs.cx"),
    ("lbvh/lbvh", "EdgeCentroidArgs.cy"),
    ("lbvh/lbvh", "EdgeCentroidArgs.cz"),
    ("lbvh/lbvh", "VertexCentroidArgs.cx"),
    ("lbvh/lbvh", "VertexCentroidArgs.cy"),
    ("lbvh/lbvh", "VertexCentroidArgs.cz"),
    // The hinge creep stencil: which four vertices each crept hinge names.
    // Written by the remap pass and read by the angle pass, both kernels, so a
    // plain `Buffer` serves it; the two oracles that name it take a prefix or
    // stage their own literal rather than pointing at host storage.
    ("energy/model/shell_bend", "ShellBendRemapArgs.hinge"),
    ("energy/model/shell_bend", "ShellBendRemapArgs.remapped"),
    ("energy/model/shell_bend", "ShellBendAngleArgs.hinge"),
    // The per-vertex query boxes, the point analogue of the edge ones above and
    // the same shape: written by the four query kernels and walked by the host
    // broad phase, so each launch refreshes the mirror it wrote.
    ("contact/aabb", "AabbPointContactQueryArgs.out"),
    ("contact/aabb", "AabbPointContactQueryMaskedArgs.out"),
    // The PCG per-row fold outputs, written by one pass and reduced on the host
    // in the same block shape. They refresh together because one pass writes
    // both, which is what makes the magnitude a bound on the sum beside it.
    ("solver/pcg", "PcgDotTermsArgs.product"),
    ("solver/pcg", "PcgDotTermsArgs.absolute_product"),
    // The block-Jacobi preconditioner, both halves. The assembled diagonal is
    // written by the operator pass and read by the inversion; the inverse is
    // written by the inversion and read by the preconditioner apply. Neither is
    // touched by the host any more, which is what moving the per-row inversion
    // off it bought.
    ("csrmat/fixed_csr", "PrecondDiagonalArgs.out"),
    ("csrmat/fixed_csr", "PrecondDiagonalDynamicArgs.out"),
    ("solver/block_jacobi", "BlockJacobiInvertRowArgs.diagonal"),
    ("solver/block_jacobi", "BlockJacobiInvertRowArgs.inverse"),
    ("solver/spmv", "Mat3MulArgs.matrix"),
    // THE NEWTON RIGHT-HAND SIDE. Every element model scatters into it and the
    // linear solve reads it as `b`, so it is named by one field per scatter pass
    // plus the two Dirichlet rows. Nothing in production reads it on the host:
    // the only host touch was the opening `fill(0.0)`, which is a `vec_fill`
    // dispatch now, and the mirror exists for the tests that compare it.
    ("utility/face_scatter", "FaceAtomicEmbedForceArgs.force"),
    ("utility/hinge_scatter", "HingeAtomicEmbedForceArgs.force"),
    ("utility/rod_scatter", "RodAtomicEmbedForceArgs.force"),
    ("utility/collision_window", "CollisionWindowVertexArgs.vertex_group"),
    ("utility/collision_window", "CollisionWindowVertexArgs.windows"),
    ("utility/collision_window", "CollisionWindowVertexArgs.window_count"),
    ("utility/collision_window", "CollisionWindowVertexArgs.vertex_active"),
    ("utility/collision_window", "CollisionWindowFaceArgs.vertex_active"),
    ("utility/collision_window", "CollisionWindowFaceArgs.face"),
    ("utility/collision_window", "CollisionWindowFaceArgs.face_active"),
    ("utility/collision_window", "CollisionWindowEdgeArgs.vertex_active"),
    ("utility/collision_window", "CollisionWindowEdgeArgs.edge"),
    ("utility/collision_window", "CollisionWindowEdgeArgs.edge_active"),
    ("utility/stitch_scatter", "StitchAtomicEmbedForceArgs.force"),
    ("utility/vertex_scatter", "VertexAtomicEmbedForceArgs.force"),
    ("utility/vertex_scatter", "VertexFixIndexFromRecordsArgs.prop"),
    ("utility/vertex_scatter", "VertexFixIndexFromRecordsArgs.fix_index"),
    ("utility/vertex_scatter", "VertexDofRemovalMaskArgs.prop"),
    ("utility/vertex_scatter", "VertexDofRemovalMaskArgs.mask"),
    ("main/momentum", "MomentumEmbedArgs.force"),
    ("main/dirichlet", "DirichletLiftRowArgs.force"),
    ("main/dirichlet", "DirichletPrescribeGatedArgs.force"),
    // The Newton matvec's input and output vectors. They are the PCG's own `p`
    // and `ap`, so they move with the workspace rather than with the matrix.
    ("solver/spmv", "OperatorApplyArgs.x"),
    ("solver/spmv", "OperatorApplyArgs.result"),
    ("solver/spmv", "OperatorApplyArgs.absolute"),
    ("solver/spmv", "OperatorApplyArgs.curvature"),
    ("solver/spmv", "OperatorApplyDynamicArgs.dyn_index"),
    ("solver/spmv", "OperatorApplyDynamicArgs.dyn_value"),
    ("solver/spmv", "OperatorApplyDynamicArgs.dyn_offset"),
    ("solver/spmv", "OperatorApplyDynamicArgs.dyn_reference_index"),
    ("solver/spmv", "OperatorApplyDynamicArgs.dyn_reference_value"),
    ("solver/spmv", "OperatorApplyDynamicArgs.dyn_reference_offset"),
    ("solver/spmv", "OperatorApplyDynamicArgs.dyn_global_value"),
    ("solver/spmv", "OperatorApplyDynamicArgs.index"),
    ("solver/spmv", "OperatorApplyDynamicArgs.offset"),
    ("solver/spmv", "OperatorApplyDynamicArgs.value"),
    ("solver/spmv", "OperatorApplyDynamicArgs.transpose_pair"),
    ("solver/spmv", "OperatorApplyDynamicArgs.transpose_offset"),
    // The FOLDED pair. Same buffers as the element forms above and the same
    // residency, plus the two per-GROUP partial arrays the fold that follows
    // reads: one float per group rather than one per row.
    ("solver/spmv", "OperatorApplyFoldedArgs.index"),
    ("solver/spmv", "OperatorApplyFoldedArgs.offset"),
    ("solver/spmv", "OperatorApplyFoldedArgs.value"),
    ("solver/spmv", "OperatorApplyFoldedArgs.transpose_pair"),
    ("solver/spmv", "OperatorApplyFoldedArgs.transpose_offset"),
    ("solver/spmv", "OperatorApplyFoldedArgs.diagonal"),
    ("solver/spmv", "OperatorApplyFoldedArgs.x"),
    ("solver/spmv", "OperatorApplyFoldedArgs.result"),
    ("solver/spmv", "OperatorApplyFoldedArgs.curvature_total"),
    ("solver/spmv", "OperatorApplyFoldedArgs.absolute_total"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.dyn_index"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.dyn_value"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.dyn_offset"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.dyn_reference_index"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.dyn_reference_value"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.dyn_reference_offset"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.dyn_global_value"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.index"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.offset"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.value"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.transpose_pair"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.transpose_offset"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.diagonal"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.x"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.result"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.curvature_total"),
    ("solver/spmv", "OperatorApplyDynamicFoldedArgs.absolute_total"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.dyn_index"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.dyn_value"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.dyn_offset"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.index"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.offset"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.value"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.diagonal"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.x"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.result"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.curvature_total"),
    ("solver/spmv", "OperatorApplySymmetricFoldedArgs.absolute_total"),
    ("csrmat/fixed_csr", "PrecondDiagonalDynamicArgs.index"),
    ("csrmat/fixed_csr", "PrecondDiagonalDynamicArgs.offset"),
    ("csrmat/fixed_csr", "PrecondDiagonalDynamicArgs.value"),
    ("csrmat/fixed_csr", "PrecondDiagonalDynamicArgs.dyn_index"),
    ("csrmat/fixed_csr", "PrecondDiagonalDynamicArgs.dyn_offset"),
    ("csrmat/fixed_csr", "PrecondDiagonalDynamicArgs.dyn_value"),
    ("solver/spmv", "OperatorApplyArgs.index"),
    ("solver/spmv", "OperatorApplyArgs.offset"),
    ("solver/spmv", "OperatorApplyArgs.value"),
    ("solver/spmv", "OperatorApplyArgs.transpose_pair"),
    ("solver/spmv", "OperatorApplyArgs.transpose_offset"),
    ("csrmat/fixed_csr", "PrecondDiagonalArgs.index"),
    ("csrmat/fixed_csr", "PrecondDiagonalArgs.offset"),
    ("csrmat/fixed_csr", "PrecondDiagonalArgs.value"),
    ("solver/spmv", "OperatorApplyDynamicArgs.x"),
    ("solver/spmv", "OperatorApplyDynamicArgs.result"),
    ("solver/spmv", "OperatorApplyDynamicArgs.absolute"),
    ("solver/spmv", "OperatorApplyDynamicArgs.curvature"),
    // THE PCG WORKSPACE. The recurrence was already entirely dispatched, so
    // these vectors were host-resident only because they were passed as slices.
    // They move as ONE component with the right-hand side above and the search
    // direction below, because the seeded residual puts `b` in the same
    // `source_a` the in-place combine uses for `z`.
    ("primitives/vec_ops", "VecAddScaledArgs.source"),
    ("primitives/vec_ops", "VecAddScaledArgs.destination"),
    ("primitives/vec_ops", "VecCombineArgs.source_a"),
    ("primitives/vec_ops", "VecCombineArgs.source_b"),
    ("primitives/vec_ops", "VecCombineArgs.destination"),
    ("solver/spmv", "Mat3MulArgs.vector"),
    ("solver/spmv", "Mat3MulArgs.result"),
    ("solver/pcg", "PcgDotTermsArgs.a"),
    ("solver/pcg", "PcgDotTermsArgs.b"),
    // THE RECURRENCE'S OWN SCALARS, which never leave the device now. Every
    // one of these fields names a span of the workspace's scalar buffer or of
    // the fold scratch beside it, both device allocations sized once at
    // `initialize()`; the host reads one small probe per iteration instead of
    // four full-length vectors. `cg_device` (`cpp/solver/solver.cu:921`) keeps
    // the same thirteen floats on the GPU for a whole solve.
    ("primitives/vec_ops", "VecBlockSumArgs.source"),
    ("primitives/vec_ops", "VecBlockSumU32Args.source"),
    ("primitives/vec_ops", "VecBlockSumU32Args.total"),
    ("primitives/vec_ops", "VecBlockSumCooperativeArgs.source"),
    ("primitives/vec_ops", "VecBlockSumCooperativeArgs.total"),
    ("primitives/radix", "RadixHistogramArgs.keys"),
    ("primitives/radix", "RadixHistogramArgs.block_histograms"),
    ("primitives/radix", "RadixScatterArgs.keys_in"),
    ("primitives/radix", "RadixScatterArgs.values_in"),
    ("primitives/radix", "RadixScatterArgs.keys_out"),
    ("primitives/radix", "RadixScatterArgs.values_out"),
    ("primitives/radix", "RadixScatterArgs.global_offsets"),
    ("primitives/vec_ops", "VecBlockSumAbsCooperativeArgs.source"),
    ("primitives/vec_ops", "VecBlockSumAbsCooperativeArgs.total"),
    ("primitives/vec_ops", "VecBlockSumPairCooperativeArgs.first_source"),
    ("primitives/vec_ops", "VecBlockSumPairCooperativeArgs.second_source"),
    ("primitives/vec_ops", "VecBlockSumPairCooperativeArgs.first_total"),
    ("primitives/vec_ops", "VecBlockSumPairCooperativeArgs.second_total"),
    ("primitives/vec_ops", "VecBlockSumDualCooperativeArgs.first_source"),
    ("primitives/vec_ops", "VecBlockSumDualCooperativeArgs.second_source"),
    ("primitives/vec_ops", "VecBlockSumDualCooperativeArgs.first_total"),
    ("primitives/vec_ops", "VecBlockSumDualCooperativeArgs.second_total"),
    ("primitives/vec_ops", "VecBlockSumArgs.total"),
    ("primitives/vec_ops", "VecBlockSumAbsArgs.source"),
    ("primitives/vec_ops", "VecBlockSumAbsArgs.total"),
    ("primitives/vec_ops", "VecBlockSumPairArgs.first_source"),
    ("primitives/vec_ops", "VecBlockSumPairArgs.second_source"),
    ("primitives/vec_ops", "VecBlockSumPairArgs.first_total"),
    ("primitives/vec_ops", "VecBlockSumPairArgs.second_total"),
    ("primitives/vec_ops", "VecBlockSumDualArgs.first_source"),
    ("primitives/vec_ops", "VecBlockSumDualArgs.second_source"),
    ("primitives/vec_ops", "VecBlockSumDualArgs.first_total"),
    ("primitives/vec_ops", "VecBlockSumDualArgs.second_total"),
    ("solver/pcg", "PcgUpdateRowArgs.direction"),
    ("solver/pcg", "PcgUpdateRowArgs.product_direction"),
    ("solver/pcg", "PcgUpdateRowArgs.alpha"),
    ("solver/pcg", "PcgUpdateRowArgs.inverse_diagonal"),
    ("solver/pcg", "PcgUpdateRowArgs.iterate"),
    ("solver/pcg", "PcgUpdateRowArgs.residual"),
    ("solver/pcg", "PcgUpdateRowArgs.preconditioned"),
    ("solver/pcg", "PcgUpdateRowArgs.term_product"),
    ("solver/pcg", "PcgUpdateRowArgs.term_magnitude"),
    ("solver/pcg", "PcgUpdateRowFoldedArgs.direction"),
    ("solver/pcg", "PcgUpdateRowFoldedArgs.product_direction"),
    ("solver/pcg", "PcgUpdateRowFoldedArgs.alpha"),
    ("solver/pcg", "PcgUpdateRowFoldedArgs.inverse_diagonal"),
    ("solver/pcg", "PcgUpdateRowFoldedArgs.iterate"),
    ("solver/pcg", "PcgUpdateRowFoldedArgs.residual"),
    ("solver/pcg", "PcgUpdateRowFoldedArgs.preconditioned"),
    ("solver/pcg", "PcgUpdateRowFoldedArgs.product_total"),
    ("solver/pcg", "PcgUpdateRowFoldedArgs.residual_total"),
    ("solver/pcg", "PcgFoldAlphaArgs.curvature_source"),
    ("solver/pcg", "PcgFoldAlphaArgs.absolute_source"),
    ("solver/pcg", "PcgFoldAlphaArgs.curvature_out"),
    ("solver/pcg", "PcgFoldAlphaArgs.absolute_out"),
    ("solver/pcg", "PcgFoldAlphaArgs.rz"),
    ("solver/pcg", "PcgFoldAlphaArgs.iteration_counter"),
    ("solver/pcg", "PcgFoldAlphaArgs.value_out"),
    ("solver/pcg", "PcgFoldAlphaArgs.noise_out"),
    ("solver/pcg", "PcgFoldAlphaArgs.cause_out"),
    ("solver/pcg", "PcgFoldAlphaArgs.break_cause"),
    ("solver/pcg", "PcgFoldAlphaArgs.break_value"),
    ("solver/pcg", "PcgFoldAlphaArgs.break_iteration"),
    ("solver/pcg", "PcgFoldAlphaArgs.break_fired"),
    ("solver/pcg", "PcgFoldBetaArgs.product_source"),
    ("solver/pcg", "PcgFoldBetaArgs.residual_source"),
    ("solver/pcg", "PcgFoldBetaArgs.product_out"),
    ("solver/pcg", "PcgFoldBetaArgs.residual_out"),
    ("solver/pcg", "PcgFoldBetaArgs.rz_previous"),
    ("solver/pcg", "PcgFoldBetaArgs.iteration_counter"),
    ("solver/pcg", "PcgFoldBetaArgs.value_out"),
    ("solver/pcg", "PcgFoldBetaArgs.noise_out"),
    ("solver/pcg", "PcgFoldBetaArgs.cause_out"),
    ("solver/pcg", "PcgFoldBetaArgs.break_cause"),
    ("solver/pcg", "PcgFoldBetaArgs.break_value"),
    ("solver/pcg", "PcgFoldBetaArgs.break_iteration"),
    ("solver/pcg", "PcgFoldBetaArgs.break_fired"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksArgs.active"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksArgs.index"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksArgs.hessian"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksArgs.fixed_index"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksArgs.fixed_offset"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksArgs.fixed_value"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksArgs.refused"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksArgs.witness"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksAtArgs.active"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksAtArgs.slots"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksAtArgs.hessian"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksAtArgs.fixed_value"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedAtArgs.live"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedAtArgs.slots"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedAtArgs.hessian"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedAtArgs.fixed_value"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedArgs.live"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedArgs.index"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedArgs.hessian"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedArgs.fixed_index"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedArgs.fixed_offset"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedArgs.fixed_value"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedArgs.refused"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksGatedArgs.witness"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksLiveArgs.live"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksLiveArgs.index"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksLiveArgs.hessian"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksLiveArgs.fixed_index"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksLiveArgs.fixed_offset"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksLiveArgs.fixed_value"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksLiveArgs.refused"),
    ("csrmat/fixed_csr", "FixedPushElementBlocksLiveArgs.witness"),
    ("primitives/vec_ops", "VecAddScaledIndirectArgs.source"),
    ("primitives/vec_ops", "VecAddScaledIndirectArgs.destination"),
    ("primitives/vec_ops", "VecAddScaledIndirectArgs.coefficient"),
    ("primitives/vec_ops", "VecCombineIndirectArgs.source_a"),
    ("primitives/vec_ops", "VecCombineIndirectArgs.source_b"),
    ("primitives/vec_ops", "VecCombineIndirectArgs.destination"),
    ("primitives/vec_ops", "VecCombineIndirectArgs.coefficient_b"),
    // The contact Hessian embed: every array it touches is device-resident,
    // which is the point of the conversion. It reads the staged pairs, pushes
    // into the fixed matrix's own values, and appends a refused block to the
    // dynamic matrix's staging, so nothing crosses the seam but the claim.
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.active"),
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.arity"),
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.index"),
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.hessian"),
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.fixed_index"),
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.fixed_offset"),
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.fixed_value"),
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.claim"),
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.stage_row"),
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.stage_column"),
    ("contact/contact_assembly", "ContactEmbedHessianBlocksArgs.stage_block"),
    ("contact/contact_assembly", "ContactEmbedForceTermsArgs.active"),
    ("contact/contact_assembly", "ContactEmbedForceTermsArgs.arity"),
    ("contact/contact_assembly", "ContactEmbedForceTermsArgs.index"),
    ("contact/contact_assembly", "ContactEmbedForceTermsArgs.gradient"),
    ("contact/contact_assembly", "ContactEmbedForceTermsArgs.force"),
    ("solver/pcg", "PcgAlphaResidentArgs.rz"),
    ("solver/pcg", "PcgAlphaResidentArgs.p_ap"),
    ("solver/pcg", "PcgAlphaResidentArgs.absolute_sum"),
    ("solver/pcg", "PcgAlphaResidentArgs.iteration_counter"),
    ("solver/pcg", "PcgAlphaResidentArgs.value_out"),
    ("solver/pcg", "PcgAlphaResidentArgs.noise_out"),
    ("solver/pcg", "PcgAlphaResidentArgs.cause_out"),
    // The three STICKY breakdown latch slots, device-resident because they are
    // read and written across iterations by the kernel itself: the host reads
    // them only at a scheduled residual check.
    ("solver/pcg", "PcgAlphaResidentArgs.break_cause"),
    ("solver/pcg", "PcgAlphaResidentArgs.break_value"),
    ("solver/pcg", "PcgAlphaResidentArgs.break_iteration"),
    ("solver/pcg", "PcgAlphaResidentArgs.break_fired"),
    ("solver/pcg", "PcgBetaResidentArgs.rz_next"),
    ("solver/pcg", "PcgBetaResidentArgs.rz_previous"),
    ("solver/pcg", "PcgBetaResidentArgs.iteration_counter"),
    ("solver/pcg", "PcgBetaResidentArgs.value_out"),
    ("solver/pcg", "PcgBetaResidentArgs.noise_out"),
    ("solver/pcg", "PcgBetaResidentArgs.break_cause"),
    ("solver/pcg", "PcgBetaResidentArgs.break_value"),
    ("solver/pcg", "PcgBetaResidentArgs.break_iteration"),
    ("solver/pcg", "PcgBetaResidentArgs.break_fired"),
    ("solver/pcg", "PcgBetaResidentArgs.cause_out"),
    // The search direction: seeded on the prescribed rows, solved for by the
    // PCG as its `x`, then measured and applied. It is the same component.
    ("main/dx_seed", "DxSeedArgs.dx"),
    ("main/dx_norm", "DxMagnitudeArgs.direction"),
    ("main/position_step", "PositionStepArgs.direction"),
    // The Newton block diagonal. Momentum and the Dirichlet rows accumulate
    // into it, the matvec and the preconditioner pass read it, and nothing in
    // production touches it on the host: its opening zero is a `vec_fill`
    // dispatch and the operator carries a handle rather than a slice.
    ("main/momentum", "MomentumEmbedArgs.diagonal"),
    // The PULL pins, written once per constraint update from the incoming set
    // and read by the momentum pass alone. Their count stays a host read, which
    // a `StagedBuffer` answers off its own copy.
    ("main/momentum", "MomentumEmbedArgs.pull"),
    // The FIX pins. Unlike the pull set these are WRITTEN on the device by the
    // rewind pass, and the host reads only how many there are, so they carry no
    // mirror at all: a host copy would diverge the first time a rewind ran.
    ("main/target", "ComputeTargetSeedArgs.fix"),
    ("main/rewind_fix", "RewindFixArgs.fix"),
    // The analytic collider reads the same pin set, in both its assembly and
    // its sweep.
    ("contact/vertex_constraint", "VertexConstraintArgs.fix_pair"),
    ("contact/vertex_constraint", "VertexConstraintSweepArgs.fix_pair"),
    // THE TWO ANALYTIC COLLIDER ARRAYS, read by the same two entry points. The
    // host rebuilds both from the schedule every step, so a collider switching
    // on partway through a run arrives here and nowhere else, and only the
    // constraint pass reads them: that is the `StagedBuffer` shape, where the
    // mirror answers `len` without a readback and `handle()` refuses until this
    // step's upload has run. Their counts stay host reads for exactly that
    // reason, and both arrays are commonly EMPTY, which takes a real
    // zero-length handle rather than a null one.
    ("contact/vertex_constraint", "VertexConstraintArgs.sphere"),
    ("contact/vertex_constraint", "VertexConstraintArgs.floor"),
    ("contact/vertex_constraint", "VertexConstraintSweepArgs.sphere"),
    ("contact/vertex_constraint", "VertexConstraintSweepArgs.floor"),
    // THE CONTACT NARROW-PHASE STAGING, and it is the largest single residency
    // unit in the driver: six buffers spent across seven records, 56 call sites.
    // The visitor WRITES `active`, `arity`, `index`, `force` and `hessian` and
    // the host compaction READS them, which is `ReadbackBuffer`; `pair` is the
    // chunk of candidates the HOST fills from the caller's slice, which is
    // `StagedBuffer` and is re-uploaded per chunk because the window moves.
    ("contact/contact_narrow", "ContactPointPointArgs.pair"),
    ("contact/contact_narrow", "ContactPointPointArgs.active"),
    ("contact/contact_narrow", "ContactPointPointArgs.arity"),
    ("contact/contact_narrow", "ContactPointPointArgs.out_index"),
    ("contact/contact_narrow", "ContactPointPointArgs.out_vertex_force"),
    ("contact/contact_narrow", "ContactPointPointArgs.out_fixed_value"),
    ("contact/contact_narrow", "ContactPointPointArgs.dyn_claim"),
    ("contact/contact_narrow", "ContactPointPointArgs.dyn_row"),
    ("contact/contact_narrow", "ContactPointPointArgs.dyn_column"),
    ("contact/contact_narrow", "ContactPointPointArgs.dyn_block"),
    ("contact/contact_narrow", "ContactPointPointArgs.out_overlap"),
    // THE THREE FUSED COLLIDER TRAVERSALS. Every buffer is device-resident for
    // the same reason the self-contact traversals' are: the pass finds its own
    // candidates and deposits them without the host seeing either, so nothing
    // here can be a `HostRef`.
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.x0"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.x"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.vertex_prop"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.vertex_param"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.static_x"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.static_face"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.static_face_prop"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.static_face_param"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.fixed_index"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.fixed_offset"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.fixed_value"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.out_vertex_force"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.out_fixed_value"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.dyn_claim"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.dyn_row"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.dyn_column"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.dyn_block"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.statistics_contact_count"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.statistics_object_index"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.statistics_static_object_index"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.node"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.tree_aabb"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.query"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.assembled"),
    ("contact/collision_narrow", "CollisionPointFaceM2cTraverseArgs.out_overlap"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.x0"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.x"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.face"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.vertex_prop"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.face_prop"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.face_param"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.static_x"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.static_vertex_prop"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.static_vertex_param"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.fixed_index"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.fixed_offset"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.fixed_value"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.out_vertex_force"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.out_fixed_value"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.dyn_claim"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.dyn_row"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.dyn_column"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.dyn_block"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.statistics_contact_count"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.statistics_object_index"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.statistics_static_object_index"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.node"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.tree_aabb"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.query"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.assembled"),
    ("contact/collision_narrow", "CollisionPointFaceC2mTraverseArgs.out_overlap"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.x0"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.x"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.edge"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.vertex_prop"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.edge_prop"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.edge_param"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.static_x"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.static_edge"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.static_edge_prop"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.static_edge_param"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.fixed_index"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.fixed_offset"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.fixed_value"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.out_vertex_force"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.out_fixed_value"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.dyn_claim"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.dyn_row"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.dyn_column"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.dyn_block"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.statistics_contact_count"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.statistics_object_index"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.statistics_static_object_index"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.node"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.tree_aabb"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.query"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.assembled"),
    ("contact/collision_narrow", "CollisionEdgeEdgeTraverseArgs.out_overlap"),
    // The active-subset force scatter: every buffer is device-resident, the
    // element reached through `active` rather than through a compacted upload.
    ("utility/face_scatter", "FaceActiveEmbedForceArgs.active"),
    ("utility/face_scatter", "FaceActiveEmbedForceArgs.face"),
    ("utility/face_scatter", "FaceActiveEmbedForceArgs.gradient"),
    ("utility/face_scatter", "FaceActiveEmbedForceArgs.force"),
    ("utility/face_scatter", "FaceLiveEmbedForceArgs.live"),
    ("utility/face_scatter", "FaceLiveEmbedForceArgs.face"),
    ("utility/face_scatter", "FaceLiveEmbedForceArgs.gradient"),
    ("utility/face_scatter", "FaceLiveEmbedForceArgs.force"),
    ("utility/rod_scatter", "RodPackedEmbedForceArgs.active"),
    ("utility/rod_scatter", "RodPackedEmbedForceArgs.edge"),
    ("utility/rod_scatter", "RodPackedEmbedForceArgs.gradient"),
    ("utility/rod_scatter", "RodPackedEmbedForceArgs.force"),
    ("utility/rod_scatter", "RodActiveEmbedForceArgs.active"),
    ("utility/rod_scatter", "RodLiveEmbedForceArgs.live"),
    ("utility/rod_scatter", "RodLiveEmbedForceArgs.edge"),
    ("utility/rod_scatter", "RodLiveEmbedForceArgs.gradient"),
    ("utility/rod_scatter", "RodLiveEmbedForceArgs.force"),
    ("utility/rod_scatter", "RodActiveEmbedForceArgs.edge"),
    ("utility/rod_scatter", "RodActiveEmbedForceArgs.gradient"),
    ("utility/rod_scatter", "RodActiveEmbedForceArgs.force"),
    ("utility/hinge_scatter", "HingeActiveEmbedForceArgs.active"),
    ("utility/hinge_scatter", "HingeActiveEmbedForceArgs.hinge"),
    ("utility/hinge_scatter", "HingeActiveEmbedForceArgs.gradient"),
    ("utility/hinge_scatter", "HingeActiveEmbedForceArgs.force"),
    ("utility/hinge_scatter", "HingeLiveEmbedForceArgs.live"),
    ("utility/hinge_scatter", "HingeLiveEmbedForceArgs.hinge"),
    ("utility/hinge_scatter", "HingeLiveEmbedForceArgs.gradient"),
    ("utility/hinge_scatter", "HingeLiveEmbedForceArgs.force"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.statistics_contact_count"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.statistics_object_index"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.fixed_index"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.fixed_offset"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.vertex_param"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.vertex_prop"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.face"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.edge"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.x0"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.x"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.vertex_edge_index"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.vertex_edge_offset"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.vertex_face_index"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.vertex_face_offset"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.fixed_value"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.grain_inv_inertia"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.grain_omega"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.out_vertex_force"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.out_fixed_value"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.dyn_claim"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.dyn_row"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.dyn_column"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.dyn_block"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.grain_torque_vertex"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.grain_stiffness_vertex"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.grain_normal_vertex"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.node"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.tree_aabb"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.query"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.assembled"),
    ("contact/contact_narrow", "ContactPointPointTraverseArgs.out_overlap"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.pair"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.active"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.arity"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.out_index"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.out_vertex_force"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.out_fixed_value"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.dyn_claim"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.dyn_row"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.dyn_column"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.dyn_block"),
    ("contact/contact_narrow", "ContactPointEdgeArgs.out_overlap"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.statistics_contact_count"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.statistics_object_index"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.fixed_index"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.fixed_offset"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.edge_prop"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.vertex_param"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.edge_param"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.vertex_prop"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.face"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.edge"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.x0"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.x"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.edge_face_index"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.edge_face_offset"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.fixed_value"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.out_vertex_force"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.out_fixed_value"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.dyn_claim"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.dyn_row"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.dyn_column"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.dyn_block"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.node"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.tree_aabb"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.query"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.assembled"),
    ("contact/contact_narrow", "ContactPointEdgeTraverseArgs.out_overlap"),
    ("contact/contact_narrow", "ContactPointFaceArgs.pair"),
    ("contact/contact_narrow", "ContactPointFaceArgs.active"),
    ("contact/contact_narrow", "ContactPointFaceArgs.arity"),
    ("contact/contact_narrow", "ContactPointFaceArgs.out_index"),
    ("contact/contact_narrow", "ContactPointFaceArgs.out_vertex_force"),
    ("contact/contact_narrow", "ContactPointFaceArgs.out_fixed_value"),
    ("contact/contact_narrow", "ContactPointFaceArgs.dyn_claim"),
    ("contact/contact_narrow", "ContactPointFaceArgs.dyn_row"),
    ("contact/contact_narrow", "ContactPointFaceArgs.dyn_column"),
    ("contact/contact_narrow", "ContactPointFaceArgs.dyn_block"),
    ("contact/contact_narrow", "ContactPointFaceArgs.out_overlap"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.statistics_contact_count"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.statistics_object_index"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.fixed_index"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.fixed_offset"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.face_prop"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.vertex_param"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.face_param"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.vertex_prop"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.face"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.x0"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.x"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.fixed_value"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.out_vertex_force"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.out_fixed_value"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.dyn_claim"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.dyn_row"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.dyn_column"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.dyn_block"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.node"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.tree_aabb"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.query"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.assembled"),
    ("contact/contact_narrow", "ContactPointFaceTraverseArgs.out_overlap"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.pair"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.active"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.arity"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.out_index"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.out_vertex_force"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.out_fixed_value"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.dyn_claim"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.dyn_row"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.dyn_column"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.dyn_block"),
    ("contact/contact_narrow", "ContactEdgeEdgeArgs.out_overlap"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.statistics_contact_count"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.statistics_object_index"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.fixed_index"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.fixed_offset"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.edge_prop"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.edge_param"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.vertex_prop"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.edge"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.x0"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.x"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.fixed_value"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.out_vertex_force"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.out_fixed_value"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.dyn_claim"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.dyn_row"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.dyn_column"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.dyn_block"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.node"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.tree_aabb"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.query"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.assembled"),
    ("contact/contact_narrow", "ContactEdgeEdgeTraverseArgs.out_overlap"),
    // The collision-mesh half, the same six buffers through the same Stage.
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.pair"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.active"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.arity"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.out_index"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.out_force"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.out_hessian"),
    ("contact/collision_narrow", "CollisionPointFaceC2mArgs.out_overlap"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.pair"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.active"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.arity"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.out_index"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.out_force"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.out_hessian"),
    ("contact/collision_narrow", "CollisionPointFaceM2cArgs.out_overlap"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.pair"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.active"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.arity"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.out_index"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.out_force"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.out_hessian"),
    ("contact/collision_narrow", "CollisionEdgeEdgeArgs.out_overlap"),
    // THE COLLISION-WINDOW MASKS, recomputed on the host from the clock every
    // step and read only by the masked broad-phase queries and the leaf pass.
    // `None` still means no table at all, which is why the masked and unmasked
    // queries stay two entry points over one body rather than one entry taking
    // a handle that could name nothing.
    ("contact/aabb", "AabbPointContactQueryMaskedArgs.active"),
    ("contact/aabb", "AabbEdgeContactQueryMaskedArgs.active"),
    ("contact/aabb", "AabbEdgeScanQueryMaskedArgs.active"),
    ("contact/aabb", "AabbVertexScanQueryMaskedArgs.active"),
    ("contact/aabb", "AabbLeafActiveArgs.active"),
    ("utility/rod_damping", "RodDampingArgs.edge"),
    ("energy/rod_force", "RodStretchDiffTableArgs.edge"),
    ("plasticity/plasticity", "PlasticityFaceInverseRestArgs.face"),
    ("plasticity/plasticity", "PlasticityTetInverseRestArgs.tet"),
    // THE MESH TOPOLOGY. The scene's connectivity is written when the scene is
    // built and by nothing in the step loop, so a driver-owned device copy of
    // it cannot go stale: it is staged once in `SolverState::allocate` and
    // never refreshed. That immutability is the whole argument, and it is what
    // separates these from `prop.vertex`, which plasticity rewrites every step
    // through a path no mirror could see.
    ("utility/face_damping", "FaceDampingArgs.face"),
    ("utility/face_deformation", "FaceDeformationGradientArgs.face"),
    ("strainlimiting/rod_strain", "RodStrainForceHessianGatedArgs.edge"),
    ("strainlimiting/rod_strain", "RodStrainStiffnessGatedArgs.edge"),
    ("strainlimiting/rod_strain", "RodStrainStiffnessGatedArgs.edge_slots"),
    ("strainlimiting/strain_toi", "RodStrainToiGatedArgs.edge"),
    ("main/stretch", "RodStretchRatioGatedArgs.edge"),
    ("energy/model/shell_bend_stiffness", "ShellBendArealDensityGatheredArgs.hinge"),
    ("energy/model/shell_bend_stiffness", "ShellBendArealDensityFromRecordsArgs.hinge"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessGatedArgs.face"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessGatedArgs.face_slots"),
    ("strainlimiting/strain_toi", "ShellStrainToiGatedArgs.face"),
    ("utility/tet_damping", "TetDampingArgs.tet"),
    ("utility/tet_convert", "TetDeformationGradientArgs.tet"),
    ("main/dirichlet", "DirichletPrescribeGatedArgs.diagonal"),
    ("solver/spmv", "OperatorApplyArgs.diagonal"),
    ("solver/spmv", "OperatorApplyDynamicArgs.diagonal"),
    ("csrmat/fixed_csr", "PrecondDiagonalArgs.diagonal"),
    ("csrmat/fixed_csr", "PrecondDiagonalDynamicArgs.diagonal"),
    // The face strain limiter's shrink-corrected limit and face mass, written
    // by the same gate loop that already staged `authored_limit`. The limit is
    // named by BOTH the assembly and the line-search path, so both upload; the
    // mass is named by the assembly path alone.
    ("strainlimiting/shell_strain", "ShellStrainStiffnessGatedArgs.effective_limit"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessGatedArgs.mass"),
    ("strainlimiting/strain_toi", "ShellStrainToiGatedArgs.effective_limit"),
    ("utility/svd3x3", "Svd3x3RvArgs.u"),
    ("utility/svd3x3", "Svd3x3RvArgs.sigma"),
    ("utility/svd3x3", "Svd3x3RvArgs.vt"),
    ("energy/model/material_diff_table", "TetMaterialDiffTableArgs.sigma"),
    ("energy/model/material_diff_table", "TetMaterialDiffTableArgs.gradient_sigma"),
    ("energy/model/material_diff_table", "TetMaterialDiffTableArgs.hessian_sigma"),
    ("eigenanalysis/tet_eigenanalysis", "TetSpectralForceArgs.gradient_sigma"),
    ("eigenanalysis/tet_eigenanalysis", "TetSpectralForceArgs.u"),
    ("eigenanalysis/tet_eigenanalysis", "TetSpectralForceArgs.vt"),
    ("eigenanalysis/tet_eigenanalysis", "TetSpectralHessianArgs.sigma"),
    ("eigenanalysis/tet_eigenanalysis", "TetSpectralHessianArgs.gradient_sigma"),
    ("eigenanalysis/tet_eigenanalysis", "TetSpectralHessianArgs.hessian_sigma"),
    ("eigenanalysis/tet_eigenanalysis", "TetSpectralHessianArgs.u"),
    ("eigenanalysis/tet_eigenanalysis", "TetSpectralHessianArgs.vt"),
    ("plasticity/plasticity", "PlasticityCreepSingular3Args.singular_values"),
    ("plasticity/plasticity", "PlasticityTetInverseRestArgs.u"),
    ("plasticity/plasticity", "PlasticityTetInverseRestArgs.vt"),
    // THE PER-ELEMENT MATERIAL CONSTANTS, tet and shell. Each is BUILT ON THE
    // HOST out of the scene's material table and rebuilt every iteration,
    // because which elements are in is rebuilt every iteration, so each is a
    // `StagedBuffer` for the reason the bending sets are.
    //
    // TWO RECORDS READ THE SHELL SET AND ONE READS THE TET SET, which is the
    // whole of why the shell half could not move earlier: the BaraffWitkin arm
    // is the second reader, and it was a hand-written entry point until its
    // declaration moved beside its body.
    //
    // THE VERDICT ARRAYS, and no host code reads either one. Each is the
    // `[[seam::scatter]]` destination its table stage's return value is written
    // to, which a generated entry over a value-returning body has to carry, and
    // the body returns the verdict so that an unrecognized model id cannot fall
    // through to SNHk for any caller of the shared body.
    //
    // THE HOST SETTLES THE SAME QUESTION AT `initialize()`, in
    // `src/driver/refusal.rs`, because the verdict is a pure function of
    // `model` and `model` is built on the host out of the scene's material
    // table. Downloading it instead would charge a synchronization per NEWTON
    // ITERATION, since which elements are active is rebuilt every iteration:
    // on a host backend that is a `memcpy`, and on a queued backend it is a
    // wait for the queue to drain. A verdict the host cannot derive is a
    // different case and belongs on the seam rather than around it, which is
    // what `ReadbackBuffer` is for.
    ("energy/model/material_diff_table", "TetMaterialDiffTableArgs.model"),
    ("energy/model/material_diff_table", "TetMaterialDiffTableArgs.mu"),
    ("energy/model/material_diff_table", "TetMaterialDiffTableArgs.lambda"),
    ("energy/model/material_diff_table", "TetMaterialDiffTableArgs.accepted"),
    ("energy/model/material_diff_table", "FaceMaterialDiffTableArgs.model"),
    ("energy/model/material_diff_table", "FaceMaterialDiffTableArgs.mu"),
    ("energy/model/material_diff_table", "FaceMaterialDiffTableArgs.lambda"),
    ("energy/model/material_diff_table", "FaceMaterialDiffTableArgs.dispatch"),
    ("energy/model/baraffwitkin", "FaceBaraffwitkinArgs.model"),
    ("energy/model/baraffwitkin", "FaceBaraffwitkinArgs.mu"),
    ("energy/model/baraffwitkin", "FaceBaraffwitkinArgs.lambda"),
    // THE STEP'S THREE PER-VERTEX SCRATCH ARRAYS AND THE PIN INDEX GATHER.
    //
    // The start-of-step VELOCITY is a device allocation nothing reads: the pass
    // that forms it is the only code in this driver that names it, so it needs
    // neither a mirror nor a seed.
    //
    // THE TWO SCALAR ARRAYS ARE READBACKS, and they are the case that type's
    // own note asks a caller to think about before reaching for it. A kernel
    // writes one value per vertex and the host folds the array to a maximum:
    // `max_u` and the coordinate reach at the top of a step, `max_dx` inside the
    // Newton loop. The verdict is a function of the positions the device holds
    // rather than of anything the host built, so it is device knowledge; what
    // the download costs is a queue drain per Newton iteration on a backend
    // with a queue, and the fold that would remove it has no lane in the
    // neutral vocabulary yet.
    //
    // ONE BUFFER, TWO RECORD FIELDS, for `state.scalar`: the start-of-step pass
    // writes it as `speed_squared` and the Newton loop writes it as
    // `magnitude`, which is main.cu's own reuse of one scratch array.
    //
    // THE PIN INDEX IS STAGED: the gather at the top of a step copies it out of
    // the scene's vertex props and the three `compute_target` calls in that
    // step read the one upload.
    ("main/velocity", "VelocityTermsArgs.velocity"),
    ("main/velocity", "VelocityTermsArgs.speed_squared"),
    ("main/velocity", "VelocityTermsArgs.reach"),
    ("main/dx_norm", "DxMagnitudeArgs.magnitude"),
    ("main/target", "ComputeTargetSeedArgs.fix_index"),
    // THE PLASTIC CREEP'S DEAD ZONES, the same shape one layer down: each is
    // built on the host out of the scene's material table, in the same loop
    // that fills the creep RATE beside it, and read only by the creep pass. So
    // each is staged for the reason the rate is, and uploads with it.
    //
    // ONE RECORD FIELD SERVES THE HINGE AND THE ROD, which is why the rest
    // angle's threshold moves two buffers rather than one: both creep passes
    // dispatch `PlasticityCreepRestAngleArgs`.
    //
    // THE YIELD VERDICTS, the other direction: each creep pass writes one word
    // per element saying whether that element crossed its threshold, and the
    // host scatter below the dispatch reads it to decide which rest matrices to
    // copy out. That is a `ReadbackBuffer`, and the answer is a function of the
    // singular values the kernel computed rather than of anything the host
    // built, which is the test that type states.
    //
    // ONE FIELD SERVES THE HINGE AND THE ROD, as the rest angle's threshold
    // does: both creep passes dispatch `PlasticityCreepRestAngleArgs`, so the
    // two buffers move together or not at all.
    ("plasticity/plasticity", "PlasticityCreepSingular2Args.changed"),
    ("plasticity/plasticity", "PlasticityCreepSingular3Args.changed"),
    ("plasticity/plasticity", "PlasticityCreepRestAngleArgs.changed"),
    ("plasticity/plasticity", "PlasticityCreepSingular2Args.threshold"),
    ("plasticity/plasticity", "PlasticityCreepSingular3Args.threshold"),
    ("plasticity/plasticity", "PlasticityCreepRestAngleArgs.threshold"),
    // THE FACE SVD'S `U` AND `V^T`, the shell analogue of the tet pair above
    // and a wider component than it: four owners (the membrane's, the strain
    // limiter's, the stretch indicator's and the plastic creep's) and five
    // record fields, because the limiter takes the SHIFTED factorization and
    // the spectral Hessian names `U` under a second field name. None of the
    // four is touched by host code between the dispatch that writes it and the
    // dispatches that read it; the stretch indicator's pair is written and
    // never read at all, the indicator being a function of the singular values
    // alone.
    //
    // THE SHIFTED FACTORIZATION IS WHAT UNBLOCKED THEM. It was the one record
    // in the component whose entry point was hand-written, so nothing else in
    // it could move while it stood.
    ("utility/svd3x2", "Svd3x2Args.u"),
    ("utility/svd3x2", "Svd3x2Args.vt"),
    ("utility/svd3x2", "Svd3x2ShiftedArgs.u"),
    ("utility/svd3x2", "Svd3x2ShiftedArgs.vt"),
    // THE LARGEST SHIFTED SINGULAR VALUE, a readback: the host reads it to
    // build the strain limiter's active list, which is the second half of
    // that layer's entry gate. One float per face per Newton iteration.
    ("utility/svd3x2", "Svd3x2ShiftedArgs.largest_shifted"),
    // `state.face_strain.shifted_sigma`: the singular values shifted by one,
    // which is the strain the limiter barrier is a function of. Written by the
    // SVD stage and read by the three strain stages with no host access in
    // between, so it completes the component its neighbors `svd_u` and
    // `svd_vt` were already in.
    ("utility/svd3x2", "Svd3x2ShiftedArgs.shifted_sigma"),
    ("utility/svd3x2", "ShellStrainRestoreSigmaArgs.shifted"),
    ("strainlimiting/shell_strain", "ShellStrainDiffTableGatedArgs.shifted_sigma"),
    ("strainlimiting/shell_strain", "ShellStrainStiffnessGatedArgs.shifted_sigma"),
    ("eigenanalysis/face_eigenanalysis", "FaceSpectralForceArgs.u"),
    ("eigenanalysis/face_eigenanalysis", "FaceSpectralForceArgs.vt"),
    ("eigenanalysis/face_eigenanalysis", "FaceSpectralHessianArgs.u2"),
    ("eigenanalysis/face_eigenanalysis", "FaceSpectralHessianArgs.vt"),
    ("plasticity/plasticity", "PlasticityFaceInverseRestArgs.u"),
    ("plasticity/plasticity", "PlasticityFaceInverseRestArgs.vt"),
    // THE CREPT REST MATRICES, the other direction again and the one place
    // it is charged per FRAME rather than per Newton iteration: the creep
    // runs once a step, and the host loop below the dispatch scatters the
    // rows of the elements that yielded into the scene's own `inv_rest`
    // array. Four floats per face and nine per tet, downloaded once.
    ("plasticity/plasticity", "PlasticityFaceInverseRestArgs.inverse_rest"),
    ("plasticity/plasticity", "PlasticityTetInverseRestArgs.inverse_rest"),
    // THE DEFORMATION GRADIENTS, the input side of the factorizations above.
    // A face's `F` is read by three stages and a tet's by one, and no host code
    // touches either between the dispatch that writes it and the dispatches
    // that read it. Their entry points had to be generated first: each reads
    // its element's positions through an index list, which is the indirect
    // gather, and it was the last shape in this chain a hand-written launcher
    // still owned.
    ("utility/face_deformation", "FaceDeformationGradientArgs.deformation"),
    ("utility/tet_convert", "TetDeformationGradientArgs.deformation"),
    ("utility/svd3x2", "Svd3x2Args.input"),
    ("utility/svd3x2", "Svd3x2ShiftedArgs.input"),
    ("utility/svd3x3", "Svd3x3RvArgs.input"),
    ("energy/model/baraffwitkin", "FaceBaraffwitkinArgs.deformation"),
    ("utility/tet_convert", "TetConvertForceArgs.gradient_f"),
    ("utility/tet_convert", "TetConvertHessianArgs.hessian_f"),
    // `state.face.gradient_f` and `state.face.hessian_f`, the shell membrane's
    // force and Hessian in the deformation-gradient basis. The same shape as
    // the tet pair above and moved for the same reason, once the last of the
    // four records naming them stopped being hand-written: the spectral force,
    // the spectral Hessian and the BaraffWitkin arm write them, the two
    // converters read them, and no host code touches them in between.
    ("eigenanalysis/face_eigenanalysis", "FaceSpectralForceArgs.force"),
    ("eigenanalysis/face_eigenanalysis", "FaceSpectralHessianArgs.hessian"),
    ("energy/model/baraffwitkin", "FaceBaraffwitkinArgs.gradient_f"),
    ("energy/model/baraffwitkin", "FaceBaraffwitkinArgs.hessian_f"),
    // THE MATERIAL DIFF TABLE IN THE SINGULAR-VALUE BASIS, the shell analogue
    // of `state.tet.gradient_sigma` and `state.tet.hessian_sigma`, which have
    // been device allocations since the tet chain moved. Two owners, the
    // membrane's and the strain limiter's, and seven record fields between
    // them, because the limiter's writer names the pair `deda` and `d2ed2a`,
    // the barrier's own coordinates rather than the material's. In each owner
    // one stage writes the pair and the spectral force and Hessian read it,
    // with no host access in between.
    //
    // THE GATED DIFF TABLE IS WHAT UNBLOCKED THEM. It was the one record in the
    // component whose entry point was hand-written, so nothing else in it could
    // move while it stood.
    ("energy/model/material_diff_table", "FaceMaterialDiffTableArgs.gradient_sigma"),
    ("energy/model/material_diff_table", "FaceMaterialDiffTableArgs.hessian_sigma"),
    ("eigenanalysis/face_eigenanalysis", "FaceSpectralForceArgs.gradient_sigma"),
    ("eigenanalysis/face_eigenanalysis", "FaceSpectralHessianArgs.gradient_sigma"),
    ("eigenanalysis/face_eigenanalysis", "FaceSpectralHessianArgs.hessian_sigma"),
    // THE AUTHORED STRAIN LIMIT, the barrier's own coordinate origin. STAGED:
    // the gate loop fills it on the host out of the scene's material table, and
    // the diff table is the only record that names it. The EFFECTIVE limit
    // beside it, the shrink-corrected one, cannot follow: two hand-written
    // records name that.
    ("strainlimiting/shell_strain", "ShellStrainDiffTableGatedArgs.authored_limit"),
    ("strainlimiting/shell_strain", "ShellStrainDiffTableGatedArgs.deda"),
    ("strainlimiting/shell_strain", "ShellStrainDiffTableGatedArgs.d2ed2a"),
    ("strainlimiting/shell_strain", "ShellStrainDiffTableFromRecordsArgs.shifted_sigma"),
    ("strainlimiting/shell_strain", "ShellStrainDiffTableFromRecordsArgs.prop"),
    ("strainlimiting/shell_strain", "ShellStrainDiffTableFromRecordsArgs.face_param"),
    ("strainlimiting/shell_strain", "ShellStrainDiffTableFromRecordsArgs.deda"),
    ("strainlimiting/shell_strain", "ShellStrainDiffTableFromRecordsArgs.d2ed2a"),
    ("strainlimiting/shell_strain", "ShellStrainDiffTableFromRecordsArgs.largest_shifted"),
    ("strainlimiting/shell_strain", "ShellStrainDiffTableFromRecordsArgs.live"),
    // THE CREEP STEP FRACTION AND THE CREPT SINGULAR VALUES, the two device
    // intermediates of the plastic chain. `plasticity_alpha` computes the
    // fraction from the scene's rate and the step and the three creep passes
    // read it; each creep pass writes the crept values and the matching
    // inverse-rest pass reads them. No host code touches either in between: the
    // rate and the threshold beside them ARE host-built, which is why they are
    // not here.
    //
    // THE THREE CREEP PASSES ARE WHAT UNBLOCKED THEM. Each of these buffers was
    // named by one of them, and a hand-written entry point takes flat pointers,
    // so it has nothing to resolve a handle against.
    ("plasticity/plasticity", "PlasticityAlphaArgs.alpha"),
    ("plasticity/plasticity", "PlasticityCreepSingular2Args.alpha"),
    ("plasticity/plasticity", "PlasticityCreepSingular3Args.alpha"),
    ("plasticity/plasticity", "PlasticityCreepRestAngleArgs.alpha"),
    ("plasticity/plasticity", "PlasticityCreepSingular2Args.updated"),
    ("plasticity/plasticity", "PlasticityCreepSingular3Args.updated"),
    ("plasticity/plasticity", "PlasticityFaceInverseRestArgs.singular"),
    ("plasticity/plasticity", "PlasticityTetInverseRestArgs.singular"),
    ("utility/face_convert", "FaceConvertForceArgs.gradient_f"),
    ("utility/face_convert", "FaceConvertHessianArgs.hessian_f"),
    // The two bending layers' per-element material parameters. These are BUILT
    // ON THE HOST, out of the scene's material table, so each is a
    // `StagedBuffer`: a host array the assembly fills element by element, a
    // device allocation, and one upload between them. Their consumer is the
    // only record that names them, and it is generated.
    // THE HINGE'S REST ANGLE AND ITS GEOMETRY VERDICT, the two buffers
    // generating this entry point released. The rest angle is STAGED: it is
    // gathered on the host out of the hinge props in the same loop that fills
    // the six material buffers below, so it uploads with them. The verdict is a
    // READBACK, the direction the plastic creep's verdicts take: the entry
    // writes one word per hinge and the host loop below each dispatch reads it
    // to decide whether to stop the step, and it is a function of the positions
    // the device holds rather than of anything the host built.
    //
    // TWO CALL SITES AND ONE FIELD EACH, because the layer evaluates the same
    // body twice: at the iterate, and at the start-of-step pose the lagged
    // damping Hessian is built from.
    ("energy/model/shell_bend", "ShellBendForceHessianCheckedArgs.prop"),
    ("energy/model/shell_bend", "ShellBendForceHessianCheckedArgs.live"),
    ("energy/model/shell_bend", "ShellBendForceHessianCheckedArgs.scale"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.x"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.current"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.hinge"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.quad"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.prop"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.hinge_param"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.vertex_prop"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.kind"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.force"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.fixed_index"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.fixed_offset"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.fixed_value"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.refused"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.witness"),
    ("energy/model/shell_bend", "ShellBendEmbedArgs.hess_slots"),
    // THE HINGE'S AREAL DENSITY AND THE TWO PER-VERTEX ARRAYS IT AVERAGES. The
    // density is a device allocation of the plainest kind: the gather writes it,
    // the stiffness pass below reads it, and no host code touches it in between,
    // which is why one buffer accounts for two record fields here.
    //
    // The mass and the area are STAGED, for the reason the six material buffers
    // below are: each is gathered on the host out of the scene's vertex props,
    // element by element, and read only by that one pass. They are flat arrays
    // rather than the `VertexProp` they come from because a shared body takes
    // the arrays it was written against and not a struct whose layout it would
    // have to know.
    ("energy/model/shell_bend_stiffness", "ShellBendArealDensityGatheredArgs.vertex_mass"),
    ("energy/model/shell_bend_stiffness", "ShellBendArealDensityGatheredArgs.vertex_area"),
    ("energy/model/shell_bend_stiffness", "ShellBendArealDensityGatheredArgs.areal_density"),
    ("energy/model/shell_bend_stiffness", "ShellBendArealDensityFromRecordsArgs.vertex_prop"),
    ("energy/model/shell_bend_stiffness", "ShellBendArealDensityFromRecordsArgs.areal_density"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessArgs.areal_density"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessArgs.bend"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessArgs.warp"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessArgs.weft"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessArgs.sin2"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessArgs.length"),
    ("energy/model/shell_bend_stiffness", "ShellBendStiffnessArgs.area"),
    ("energy/model/rod_bend_stiffness", "RodBendStiffnessArgs.bend"),
    ("energy/model/rod_bend_stiffness", "RodBendStiffnessArgs.mass"),
    ("energy/model/rod_bend_stiffness", "RodBendStiffnessArgs.length0"),
    ("energy/model/rod_bend_stiffness", "RodBendStiffnessArgs.length1"),
    // The four plastic creep rates, one per element class. Built on the host
    // out of the scene's material table, so each is a `StagedBuffer` for the
    // reason the bending sets are. ONE FIELD, FOUR BUFFERS: the record is
    // filled at four call sites from four arrays, and a field is one type in
    // the Rust twin, so the four move together or not at all. They can move
    // because `plasticity_alpha` is the only record naming any of them;
    // their siblings in the same chain (the dead zone, the crept singular
    // values, the yield verdict) are each named by a hand-written record too.
    ("plasticity/plasticity", "PlasticityAlphaArgs.plasticity"),
    // THE ELEMENT SCATTER INPUTS, one connected component covering nine owners
    // and four records. Each is COMPACTED ON THE HOST: an assembly pass walks
    // its active elements in ascending order and writes the vertex indices and
    // the contribution into one run starting at zero, so each is a
    // `StagedBuffer` for the reason the bending sets are. The prefix is what
    // the dispatch covers, and the record's `count` is that prefix; the handle
    // names the whole allocation because a generated entry gathers element
    // `index` and guards on the count rather than on the buffer's length.
    //
    // THE UNIT IS THE COMPONENT, NOT THE BUFFER. A record field is one type in
    // the Rust twin, so every buffer that fills it at any call site moves with
    // it: the tet and hinge stages share `HingeAtomicEmbedForceArgs`, three
    // shell and rod stages share the face record, two share the rod record, and
    // the contact narrow phase fills all four from ITS staging arrays while the
    // analytic collider fills the vertex one from its own. Moving any one of
    // them alone would leave a sibling call site handing an address to a field
    // that is now a handle.
    // THE FOUR ELEMENT DAMPING COEFFICIENTS, tet, face, hinge and rod bending
    // site. Each is BUILT ON THE HOST out of the scene's material table, in the
    // same loop that fills the material or stiffness set beside it, and read by
    // that element class's damping dispatch alone, which is the shape the
    // bending sets have. The hinge's and the site's are gathered a second time
    // over the ACTIVE list, after the stiffness gate has run, so each of those
    // two uploads sits below its gate loop rather than with its siblings.
    //
    // THE HOST STILL READS ITS OWN COPY, through `StagedBuffer::host`, to
    // answer whether any active element asks for damping at all; that is the
    // host array the type keeps, not a download.
    //
    // GENERATING THE DAMPING FAMILY IS WHAT RELEASED THEM. Each of the four was
    // named by a hand-written entry point, and a shim there takes a flat
    // pointer and has nothing to resolve a handle against.
    // THE ROD BENDING STENCILS, `(j, i, k)` per site with the interior vertex
    // in the middle. UPLOADED ONCE, at allocation: neither adjacency the
    // enumeration reads is rewritten after scene build, so the table is formed
    // there and never again. Four records read it, the force and Hessian pass
    // at both poses, the damping pass and the plastic creep's turning angle,
    // and the host copy stays readable through `StagedBuffer::host` for the
    // interior-vertex lookups, the scatter's index gather and the CSR pushes.
    //
    // THE TWO TEST ORACLES MOVED WITH IT, and they had to: a `HostDevice` keeps
    // its arenas per INSTANCE, so a handle cut from a fixture's target resolves
    // against the wrong base on the fresh one those oracles used to make. The
    // plastic one now dispatches on the fixture's own target; the assembly one
    // keeps its independent stencil and allocates it on the target it
    // dispatches on.
    ("energy/model/rod_bend", "RodBendForceHessianArgs.node_index"),
    ("energy/model/rod_bend", "RodBendAngleArgs.node_index"),
    ("utility/rod_bend_damping", "RodBendDampingArgs.node_index"),
    // THE PERMUTED HINGE QUADRUPLES AND THE ROD BENDING REST ANGLES, the two
    // host-built arrays their bending layers read at every pose. The
    // permutation is applied on the host, once per assembly, and read by the
    // hinge force and Hessian pass at both poses and by the damping pass; the
    // rest angles are gathered out of the interior vertex's prop, which the
    // plastic creep rewrites, and read by the rod bending pass at both poses.
    // Each host copy stays readable through `StagedBuffer::host`, which is what
    // the scatters' index gathers and the CSR push loops read.
    ("energy/model/shell_bend", "ShellBendForceHessianCheckedArgs.hinge"),
    ("utility/hinge_damping", "HingeDampingArgs.hinge"),
    ("energy/model/rod_bend", "RodBendForceHessianArgs.rest_angle"),
    ("utility/tet_damping", "TetDampingArgs.beta"),
    // THE FUSED TET ELASTIC LAYER. Every buffer it names is already a device
    // allocation: the two position buffers and the tet table the staged chain
    // used, the inverse rest matrices, the five staged material arrays, the
    // Newton right-hand side and the fixed matrix's pattern and values.
    ("energy/tet_force", "TetElasticEmbedArgs.x"),
    ("energy/tet_force", "TetElasticEmbedArgs.current"),
    ("energy/tet_force", "TetElasticEmbedArgs.tet"),
    ("energy/tet_force", "TetElasticEmbedArgs.tet_vertex"),
    ("energy/tet_force", "TetElasticEmbedArgs.inverse_rest"),
    ("energy/tet_force", "TetElasticEmbedArgs.model"),
    ("energy/tet_force", "TetElasticEmbedArgs.mu"),
    ("energy/tet_force", "TetElasticEmbedArgs.lambda"),
    ("energy/tet_force", "TetElasticEmbedArgs.mass"),
    ("energy/tet_force", "TetElasticEmbedArgs.deform_damping"),
    ("energy/tet_force", "TetElasticEmbedArgs.force"),
    ("energy/tet_force", "TetElasticEmbedArgs.index"),
    ("energy/tet_force", "TetElasticEmbedArgs.offset"),
    ("energy/tet_force", "TetElasticEmbedArgs.value"),
    ("energy/tet_force", "TetMaterialFromRecordsArgs.prop"),
    ("energy/tet_force", "TetMaterialFromRecordsArgs.tet_param"),
    ("energy/tet_force", "TetMaterialFromRecordsArgs.model"),
    ("energy/tet_force", "TetMaterialFromRecordsArgs.mu"),
    ("energy/tet_force", "TetMaterialFromRecordsArgs.lambda"),
    ("energy/tet_force", "TetMaterialFromRecordsArgs.mass"),
    ("energy/tet_force", "TetMaterialFromRecordsArgs.damping"),
    // THE FUSED SHELL MEMBRANE LAYER, on the same terms as the solid one
    // above: every buffer it names is already a device allocation. The two
    // position buffers and the face table the staged chain used, the inverse
    // rest matrices, the six staged material arrays (the pressure joining the
    // five the solid has), the Newton right-hand side and the fixed matrix's
    // pattern and values.
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.x"),
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.current"),
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.face"),
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.face_vertex"),
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.inverse_rest"),
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.prop"),
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.face_param"),
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.force"),
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.index"),
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.offset"),
    ("energy/face_force", "FaceElasticEmbedFromRecordsArgs.value"),
    ("utility/face_damping", "FaceDampingArgs.beta"),
    ("utility/hinge_damping", "HingeDampingArgs.beta"),
    ("utility/rod_bend_damping", "RodBendDampingArgs.beta"),
    ("utility/face_scatter", "FaceAtomicEmbedForceArgs.face"),
    ("utility/face_scatter", "FaceAtomicEmbedForceArgs.gradient"),
    ("utility/hinge_scatter", "HingeAtomicEmbedForceArgs.hinge"),
    ("utility/hinge_scatter", "HingeAtomicEmbedForceArgs.gradient"),
    ("utility/rod_scatter", "RodAtomicEmbedForceArgs.edge"),
    ("utility/rod_scatter", "RodAtomicEmbedForceArgs.gradient"),
    ("utility/vertex_scatter", "VertexAtomicEmbedForceArgs.vert"),
    ("utility/vertex_scatter", "VertexAtomicEmbedForceArgs.gradient"),
    // THE DYNAMIC CONTACT MATRIX, whose storage moved from a host `Vec` per row
    // to the reference's flat slab: one row-offset array, one flat column array
    // and one flat block array, with a `Row` reduced to an offset into them. The
    // six passes below are the whole of `cpp/csrmat/csrmat.cu`'s lifecycle, so
    // every array a step of that lifecycle touches is named here at once; a
    // field left behind would leave one pass reading a host address while the
    // rest read a handle, which is the shape that cannot be dispatched at all.
    ("csrmat/dynamic_csr", "DynRowBeginPassArgs.fixed_offset"),
    ("csrmat/dynamic_csr", "DynRowBeginPassArgs.fixed_index"),
    ("csrmat/dynamic_csr", "DynRowBeginPassArgs.reserve"),
    ("csrmat/dynamic_csr", "DynDryPushPassArgs.fixed_offset"),
    ("csrmat/dynamic_csr", "DynDryPushPassArgs.fixed_index"),
    ("csrmat/dynamic_csr", "DynDryPushPassArgs.push_row"),
    ("csrmat/dynamic_csr", "DynDryPushPassArgs.push_column"),
    ("csrmat/dynamic_csr", "DynDryPushPassArgs.reserve"),
    ("csrmat/dynamic_csr", "DynRowSeedPassArgs.fixed_offset"),
    ("csrmat/dynamic_csr", "DynRowSeedPassArgs.fixed_index"),
    ("csrmat/dynamic_csr", "DynRowSeedPassArgs.dyn_offset"),
    ("csrmat/dynamic_csr", "DynRowSeedPassArgs.dyn_index"),
    ("csrmat/dynamic_csr", "DynRowSeedPassArgs.dyn_value"),
    ("csrmat/dynamic_csr", "DynRowSeedPassArgs.head"),
    ("csrmat/dynamic_csr", "DynPushPassArgs.fixed_offset"),
    ("csrmat/dynamic_csr", "DynPushPassArgs.dyn_offset"),
    ("csrmat/dynamic_csr", "DynPushPassArgs.dyn_index"),
    ("csrmat/dynamic_csr", "DynPushPassArgs.dyn_value"),
    ("csrmat/dynamic_csr", "DynPushPassArgs.push_row"),
    ("csrmat/dynamic_csr", "DynPushPassArgs.push_column"),
    ("csrmat/dynamic_csr", "DynPushPassArgs.push_block"),
    ("csrmat/dynamic_csr", "DynPushPassArgs.head"),
    ("csrmat/dynamic_csr", "DynPushPassArgs.refused"),
    ("csrmat/dynamic_csr", "DynRowCompactPassArgs.fixed_offset"),
    ("csrmat/dynamic_csr", "DynRowCompactPassArgs.fixed_index"),
    ("csrmat/dynamic_csr", "DynRowCompactPassArgs.dyn_offset"),
    ("csrmat/dynamic_csr", "DynRowCompactPassArgs.dyn_index"),
    ("csrmat/dynamic_csr", "DynRowCompactPassArgs.dyn_value"),
    ("csrmat/dynamic_csr", "DynRowCompactPassArgs.head"),
    ("csrmat/dynamic_csr", "DynRowCompactPassArgs.split"),
    ("csrmat/dynamic_csr", "DynRowEmitPassArgs.dyn_offset"),
    ("csrmat/dynamic_csr", "DynRowEmitPassArgs.dyn_index"),
    ("csrmat/dynamic_csr", "DynRowEmitPassArgs.dyn_value"),
    ("csrmat/dynamic_csr", "DynRowEmitPassArgs.head"),
    ("csrmat/dynamic_csr", "DynRowEmitPassArgs.split"),
    ("csrmat/dynamic_csr", "DynRowEmitPassArgs.pattern"),
    ("csrmat/dynamic_csr", "DynRowEmitPassArgs.flat_index"),
    ("csrmat/dynamic_csr", "DynRowEmitPassArgs.flat_value"),
];

/// Compiles the CPU backend's kernel entry points, through the compute crate's
/// recipe.
///
/// EVERYTHING THIS FUNCTION DECIDES IS A NAME FROM THIS CRATE'S OWN DOMAIN:
/// which kernels, which translation units, where they live. How they are
/// rendered, which compiler flags they get and what the ISA baseline is are
/// decisions about how a device is made to compute, so they live in
/// `ppf-cts-compute`. The invocation cannot live there
/// too: cargo runs a dependency's build script first and gives it no way to see
/// this crate's files, so a recipe reaching `src/kernels` from that crate would make
/// it unpublishable on its own.
fn build_kernels() {
    let out_dir = std::path::PathBuf::from(env::var("OUT_DIR").unwrap());

    // The shared headers the shim reaches through data.hpp: the coordinate
    // scalar, the linalg pack, the seam. Cargo reads a directory here as "any
    // descendant",
    // which is what is wanted: a changed struct layout must recompile the shim
    // or its repr(C) view disagrees with the Rust one at run time. It is watched
    // HERE rather than inside the recipe because these are this crate's sources
    // and the recipe never learns they exist.
    println!("cargo:rerun-if-changed=src/kernels");
    println!("cargo:rerun-if-changed=entrypoints");

    let artifact = compute::build::host::compile_shim(&compute::build::host::HostShim {
        kernel_root: Path::new("src/kernels"),
        kernels: &KERNELS,
        translation_units: &[
            "entrypoints/kernel_shim.cpp",
            // The GENERATED entry points, in the one translation unit allowed to
            // define them: each rendering carries a `#pragma once`, which does
            // not reach across translation units, so a second includer would be
            // a duplicate symbol at link.
            "entrypoints/entries.cpp",
            // A second shim translation unit, which is the split the plan
            // anticipates rather than a new mechanism: every file here is a loop
            // and a gather around a shared body, and grouping them by subsystem
            // is what keeps one file from becoming the whole backend.
            "entrypoints/shim_override_seed.cpp",
            // The Newton driver's own entry points: the position steps, the
            // momentum layer, the tet material table, the operator A + B + C
            // and the PCG inner-product terms. Separate from
            // kernel_shim.cpp because that file is the ELEMENT layer (flat arrays
            // of one element kind) while this one takes the scene records.
            "entrypoints/shim_step.cpp",
            // The contact entry points: the broad phase and the intersection
            // gate. A separate translation unit rather than more of
            // `kernel_shim.cpp`, because it defines `DIAG_ASSERT4` before it
            // includes the traversal body, and that macro would otherwise be in
            // scope for every other body the shim compiles.
            "entrypoints/shim_contact.cpp",
            // Geometry outside the solved namespace: the analytic sphere and
            // floor colliders and the rest-pose static collision mesh. Its own
            // translation unit because it includes `energy/model/fix.hpp`, whose
            // functions are declared without `inline`, so a second file
            // including it would be a duplicate symbol at link.
        ],
        out_dir: &out_dir,
        lib_name: "ppf_kernels",
        handle_fields: HANDLE_FIELDS,
    });

    // EVERY DECLARATION IS COMPILED BY BOTH HALVES, AND THE TWO HALVES ARE
    // ARRANGED DIFFERENTLY ON PURPOSE.
    //
    // A rendering no compiler reads is not single-sourced, it is untested, and
    // that is measured rather than argued: the MSL rendering of every gathered
    // struct element was wrong for as long as nothing compiled one. The C++
    // half is decided by a hand-written include list in
    // `entrypoints/entries.cpp`, so a declaration missing from it is a silent
    // gap and `check_entry_coverage` names it. The Rust half is not a list at
    // all any more: `write_entry_manifest` emits one `include!` per declaration
    // the recipe rendered, so the gap it would report cannot be written.
    //
    // What remains to check there is not a list against the declarations but
    // one RECOGNIZER against another, which `check_recognizer_agreement` does.
    // It runs first: it decides whether the artifacts the other two read say
    // what their sources declare.
    check_recognizer_agreement(&artifact);
    check_entry_coverage(&artifact);
    write_entry_manifest(&artifact);
    write_entry_source_list(&artifact);

    // Read at run time so `initialize()` can refuse a CPU that cannot run what
    // was built for it. Emitted HERE because the code that reads it is this
    // crate's; the recipe decided the value and says nothing about who is told.
    println!(
        "cargo:rustc-env=PPF_HOST_BASELINE_BUILT={}",
        artifact.baseline
    );
}

/// The text the transcompiler leaves in a body rendering where an entry
/// declaration stood.
///
/// One per declaration its span parser accepted, replacing the declaration line
/// for line so the rendering's line count is unchanged. Counting it reads that
/// parser's verdict off an artifact instead of re-implementing the parse, which
/// is the whole point: a check comparing a rule against itself cannot fail.
const ENTRY_MARKER: &str = "[kernelgen] entry declaration:";

/// The kernel-id names a `rust` entry rendering references.
///
/// A rendering names each of its entry points exactly once, as
/// `const KERNEL: KernelId = id::<NAME>;`, so that a missing or renamed id is a
/// compile error in the driver rather than a literal in two places. Reading
/// them back off the rendering is not a second derivation of anything: it is
/// the same text rustc will read, so a manifest built from it cannot name a set
/// the renderings do not.
///
/// The banner every rendering carries spells `id::<STEM>` in prose. It is
/// excluded by the anchor rather than by stripping comments: a line that is a
/// comment does not begin with `const`.
fn entry_ids(text: &str) -> Vec<&str> {
    const OPEN: &str = "const KERNEL: KernelId = id::";
    text.lines()
        .filter_map(|line| {
            let name = line.trim().strip_prefix(OPEN)?.strip_suffix(';')?;
            let named = !name.is_empty()
                && name
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_');
            named.then_some(name)
        })
        .collect()
}

/// Fails the build when the two recognizers that read an entry declaration
/// disagree about how many there are.
///
/// WHY THERE ARE TWO, AND WHY THIS IS THE CHECK WORTH KEEPING. A declaration is
/// read twice, by code that shares nothing. `ppf-cts-compute`'s host recipe
/// applies a LEXICAL test, one of the two marking attributes at column zero,
/// and that test alone decides whether a kernel gets entry artifacts at all.
/// The transcompiler applies its own SPAN PARSER, which accepts the attribute
/// anywhere in a line's code and follows the declaration to the `;` that closes
/// it; its verdict is visible in the body rendering, which is produced for
/// every kernel whatever the lexical test said.
///
/// They agree on every declaration in this tree. Where they would not, the
/// failure is silent in both directions and neither include list sees it:
///
/// - A declaration the lexical test misses (indented, or preceded by a token)
///   renders no entry artifacts, and the parser still marks it in the body.
/// - A declaration the lexical test sees and the parser does not yields an
///   entry rendering that is a banner and nothing else. It compiles, it links,
///   and it defines no record, so every list naming it goes on passing.
///
/// Comparing the counts per file catches both, and catches them at DECLARATION
/// granularity rather than at file granularity: a source carrying two
/// declarations of which one is recognized fails here, where a set comparison
/// would call the file covered.
///
/// The `rust` entry rendering's own count is compared as well. It comes from a
/// third piece of the generator, the renderer rather than the parser, and it is
/// what `write_entry_manifest` reads, so the manifest is built from a number
/// two independent readings have already agreed on.
fn check_recognizer_agreement(artifact: &compute::build::host::HostShimArtifact) {
    let kernel_root = Path::new("src/kernels");
    let generated = &artifact.generated_root;
    let read = |path: &Path| -> String {
        std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("build.rs: cannot read {}: {e}", path.display()))
    };
    let mut disagreed: Vec<String> = Vec::new();
    let mut total = 0usize;
    for kernel in KERNELS {
        let source = read(&kernel_root.join(format!("{kernel}.kernel.cpp")));
        let lexical = source
            .lines()
            .filter(|l| {
                l.starts_with("[[seam::args]]") || l.starts_with("[[seam::entry]]")
                    || l.starts_with("[[seam::entry(")
            })
            .count();
        let parsed = read(&generated.join(format!("{kernel}.kernel.cpp")))
            .matches(ENTRY_MARKER)
            .count();
        total += parsed;
        if lexical != parsed {
            disagreed.push(format!(
                "  {kernel}: the lexical test reads {lexical} entry \
                 declaration(s), the transcompiler's parser {parsed}"
            ));
            continue;
        }
        if parsed == 0 {
            continue;
        }
        let rendered = entry_ids(&read(&generated.join(format!("{kernel}.entry.rs")))).len();
        if rendered != parsed {
            disagreed.push(format!(
                "  {kernel}: the transcompiler's parser reads {parsed} entry \
                 declaration(s) and its rust rendering names {rendered} kernel id(s)"
            ));
        }
    }
    assert!(
        total > 0,
        "build.rs: no rendered body carries `{ENTRY_MARKER}`, so this check \
         compared every count against zero and would pass whatever the sources \
         say. Either no kernel in KERNELS declares an entry, or the \
         transcompiler no longer marks one."
    );
    if !disagreed.is_empty() {
        panic!(
            "build.rs: {} neutral source(s) are read differently by the two \
             recognizers that decide what an entry rendering contains:\n{}\n\
             Neither include list can see this: a declaration only one of them \
             accepts still leaves a file that compiles. Fix the declaration, or \
             the recognizer that is wrong about it.",
            disagreed.len(),
            disagreed.join("\n")
        );
    }
}

/// Fails the build when a neutral source declares an entry point that
/// `entrypoints/entries.cpp` does not name.
///
/// WHAT IT READS AND WHY IT IS TEXTUAL. The declarations are not re-derived
/// here: `artifact.declaring` is the recipe's own verdict, the one that decided
/// whether the renderings exist, so this cannot disagree with what is on disk.
/// The include list is read as text because that is what it is; a `#include` is
/// not a value this script can ask for.
///
/// It checks COVERAGE only, never the reverse. An include naming a rendering
/// that does not exist already fails, loudly, at the compiler.
///
/// ONLY THE C++ HALF IS LEFT, and that is a narrowing rather than a loss. The
/// Rust half was a second hand-written list and is now generated from
/// `artifact.declaring` by `write_entry_manifest`, so checking it would compare
/// a list against the thing that produced it. This one still reads a file a
/// human maintains, so it can still fail.
fn check_entry_coverage(artifact: &compute::build::host::HostShimArtifact) {
    const LIST: &str = "entrypoints/entries.cpp";
    println!("cargo:rerun-if-changed={LIST}");
    assert!(
        !artifact.declaring.is_empty(),
        "build.rs: no kernel in KERNELS declares an entry point, so this check \
         would compare a list against nothing and pass."
    );
    // LINE COMMENTS ARE STRIPPED FIRST, and that is the difference between a
    // check and a formality. That file names a rendering in prose as well as in
    // an include, and commenting an include out is how one stops being
    // compiled, so a bare substring search over the raw text would go on
    // passing over exactly the change it exists to catch. Measured: commenting
    // out one `#include` left it silent. It has no string literal carrying
    // `//`.
    let raw = std::fs::read_to_string(LIST)
        .unwrap_or_else(|e| panic!("build.rs: cannot read {LIST}: {e}"));
    let mut text = String::new();
    for line in raw.lines() {
        text.push_str(match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        });
        text.push('\n');
    }
    let missing: Vec<String> = artifact
        .declaring
        .iter()
        .filter(|kernel| !text.contains(&format!("{kernel}.entry.cpp")))
        .map(|kernel| {
            format!("  {kernel} is named by none of {LIST}, so the host C++ compiler never reads its entry.cpp rendering")
        })
        .collect();
    if !missing.is_empty() {
        panic!(
            "build.rs: {} generated entry rendering(s) would be compiled by \
             nothing:\n{}\nAdd the include. A rendering no compiler reads is not \
             single-sourced, it is untested.",
            missing.len(),
            missing.join("\n")
        );
    }
}

/// Writes the manifest `src/driver/generated_entries.rs` includes: every `rust`
/// entry rendering, and a stand-in id for every entry point they name.
///
/// WHY THIS IS GENERATED AND THE C++ LIST IS NOT. Both lists were hand-written,
/// for a reason that reads well and holds only for one of them: a declaration
/// that does not reach a list is a silent gap, and a name in a list with
/// nothing behind it is a compile error. The second half is still true here and
/// the first is now unreachable, because the list IS `artifact.declaring` and
/// nothing transcribes it. The C++ list carries per-entry prose about the
/// declarations it names, which a generated file has no place for, so it stays
/// a list with `check_entry_coverage` behind it.
///
/// WHAT IS READ, AND WHAT IS NOT DERIVED. `KERNELS` is not touched here: it is
/// this backend's capability policy, the one place a human states which kernels
/// it compiles, and discovering it by walking the tree would compile entry
/// points for capabilities `src/driver/refusal.rs` refuses at `initialize()`.
/// The id NAMES are likewise not derived: an entry point is named by its
/// neutral body, one source can declare several, and only the transcompiler
/// parses that. They are read back off the renderings it has just written,
/// which is the same text rustc will read.
///
/// It lands in `OUT_DIR` beside the renderings it lists. Nothing may be built
/// under `src/`, because two build scripts watch that tree recursively and read
/// a directory as "any descendant", so one artifact inside costs a full rebuild
/// of both crates on every subsequent build.
///
/// TWO RENDERINGS CLAIMING ONE ID IS NOT CHECKED HERE, and the reason is that
/// nothing can reach the check. An id name is its entry point's name, so two
/// claimants are two entry points with one symbol. Within a file the
/// transcompiler refuses the second by name. Across files both `entry.cpp`
/// renderings are included by `entrypoints/entries.cpp`, which
/// `check_entry_coverage` requires, so the host C++ compiler meets the
/// duplicated argument record before this function runs: measured by declaring
/// a second `dx_magnitude` in another kernel, which failed at
/// `redefinition of 'struct DxMagnitudeArgs'`. If that translation unit is
/// ever split, a duplicate still fails loudly, as a duplicate `const` in the
/// generated `mod id`.
/// The neutral sources that declare an entry, for the backend recipes to read.
///
/// **THE RECIPES USED TO GREP FOR THE ATTRIBUTE SPELLING, AND A SPELLING CHANGE
/// LEFT THE SET SHORT RATHER THAN EMPTY.** Their own guard catches an EMPTY set,
/// which is the loud case; a set missing only the declarations whose spelling
/// moved compiles a library that references every absent `<name>_entry_launch`
/// and links clean, and the binary is what fails, naming symbols rather than the
/// cause. This list is the RENDERER's own answer, one line per source it
/// actually rendered an entry for, so no recipe has to recognize a spelling.
fn write_entry_source_list(artifact: &compute::build::host::HostShimArtifact) {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    let path = Path::new(&out_dir).join("entry-sources.txt");
    let mut body = String::new();
    for kernel in &artifact.declaring {
        body.push_str(kernel);
        body.push_str(".kernel.cpp\n");
    }
    std::fs::write(&path, body).expect("writing entry-sources.txt");
}

fn write_entry_manifest(artifact: &compute::build::host::HostShimArtifact) {
    let generated = &artifact.generated_root;
    let mut ids: Vec<(String, &str)> = Vec::new();
    let mut includes = String::new();
    for kernel in &artifact.declaring {
        let rendering = generated.join(format!("{kernel}.entry.rs"));
        let text = std::fs::read_to_string(&rendering)
            .unwrap_or_else(|e| panic!("build.rs: cannot read {}: {e}", rendering.display()));
        for name in entry_ids(&text) {
            ids.push((name.to_string(), kernel));
        }
        includes.push_str(&format!(
            "include!(concat!(env!(\"OUT_DIR\"), \"/kernelgen/{kernel}.entry.rs\"));\n"
        ));
    }
    ids.sort();
    let mut manifest = String::from(
        "// GENERATED by crates/ppf-cts-solver/build.rs. Do not edit, and do not\n\
         // check in: this file is written to OUT_DIR on every build of the CPU\n\
         // backend and `src/driver/generated_entries.rs` includes it.\n\
         //\n\
         // One `include!` per neutral source in that script's KERNELS list that\n\
         // declares an entry point, and one stand-in id per entry point those\n\
         // renderings name. Both are read off what this build has just rendered,\n\
         // so a declaration cannot fail to reach the list and a name here cannot\n\
         // fail to have a rendering behind it.\n\n",
    );
    manifest.push_str("mod id {\n    use super::{KernelId, STANDIN};\n");
    for (name, kernel) in &ids {
        manifest.push_str(&format!("    // {kernel}\n    pub const {name}: KernelId = STANDIN;\n"));
    }
    manifest.push_str("}\n\n");
    manifest.push_str(&includes);
    let path = generated.join("entries_manifest.rs");
    std::fs::write(&path, manifest)
        .unwrap_or_else(|e| panic!("build.rs: cannot write {}: {e}", path.display()));
}

/// The compute crate's library directory, read back through the `links`
/// manifest channel.
///
/// THIS IS AN ASSERTION, NOT A LOOKUP, and the reason is the cargo rule that
/// decides how this workspace links. `cargo:rustc-link-arg` emitted by a
/// DEPENDENCY never reaches the dependent binary's link, while
/// `rustc-link-lib` and `rustc-link-search` do. Two flags a C++ backend needs
/// are exactly the kind that does not propagate: `-Wl,-rpath,<libdir>`, and on
/// macOS `-Wl,-exported_symbol,_print_rust`, without which the backend dylib's
/// undefined `print_rust` resolves to address zero and the solver segfaults at
/// its first log line. So the directory travels as metadata and THIS script
/// emits the flags.
///
/// A missing key means the `links` key was dropped from
/// `crates/ppf-cts-compute/Cargo.toml`, and the failure that would follow is
/// silent on the build that drops it and fatal at load time on the next macOS
/// backend build. Naming it here, on every build and on every host, is what
/// turns a load-time crash into a build-time message.
fn compute_libdir() -> std::path::PathBuf {
    let dir = env::var("DEP_PPFCTSCOMPUTE_LIBDIR").unwrap_or_else(|_| {
        panic!(
            "build.rs: DEP_PPFCTSCOMPUTE_LIBDIR is unset, so ppf-cts-compute is \
             not publishing its library directory. Restore the `links` key in \
             crates/ppf-cts-compute/Cargo.toml: the two link arguments a C++ \
             backend needs cannot be emitted by the crate that produces the \
             library, so they travel through this channel and are emitted here."
        )
    });
    std::path::PathBuf::from(dir)
}

/// The hand-written Rust seam must agree with the entry declarations.
///
/// `driver/launch.rs` declares an `extern` for every generated entry point and
/// picks a thunk macro for it. Both are decided entirely by two facts the
/// generator already computes, whether the entry is group-shaped and whether it
/// takes a `[[seam::diag]]` channel, and both are written by hand.
///
/// A DISAGREEMENT IS NOT A COMPILE ERROR, which is why this runs at BUILD time
/// rather than by convention. Rust believes the declaration: a group symbol
/// takes an extra `group_width` between the arena base and the range, so a
/// four-parameter declaration against a five-parameter definition reads the
/// width as `begin`. A missing `diag` is worse in the other direction, since
/// the callee then reads a register the caller never set and writes a failing
/// assert through it.
///
/// The BODY is checked on the same terms by `check-address-spaces.py`, which
/// runs here too: Metal is the only target with more than one address space, so
/// a parameter marked `[[seam::thread]]` that its entry hands a device pointer
/// builds clean on CUDA and on the host and fails the shader compile at run
/// time, on a Mac, after every build leg has gone green.
///
/// They are skipped when python3 is absent rather than failing the build, on
/// the same terms as any other optional check: the host target already needs
/// python3 to render its kernels, so a build that got this far has it.
fn check_launch_seam() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf);
    let Some(root) = root else { return };
    println!("cargo:rerun-if-changed=src/driver/launch.rs");
    // BOTH CHECKS ANSWER THE SAME QUESTION FROM OPPOSITE SIDES, which is why
    // they run together: one compares the Rust seam against the declaration and
    // one compares the BODY against it. Neither failure is a compile error on
    // this host, and the second is not a compile error on any host that is not
    // a Mac.
    for (name, what) in [
        ("check-launch-seam.py",
         "the hand-written seam in src/driver/launch.rs disagrees with the \
          entry declarations"),
        ("check-address-spaces.py",
         "a neutral kernel body's address spaces disagree with how its entry \
          hands the buffers over, which fails the Metal shader compile at run \
          time and nothing earlier"),
    ] {
        let script = root.join(".github/workflows/scripts").join(name);
        if !script.exists() {
            continue;
        }
        println!("cargo:rerun-if-changed={}", script.display());
        let out = match Command::new("python3").arg(&script).output() {
            Ok(out) => out,
            // No python3 is not this check's failure to report.
            Err(_) => continue,
        };
        if !out.status.success() {
            panic!(
                "ppf-cts-solver: {what}.\n{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}

/// The `extern` block for every generated entry point, written by the generator.
///
/// A DECLARATION'S ARITY IS A FUNCTION OF THE ENTRY, not a choice: a group
/// symbol takes an extra `group_width` between the arena base and the range, and
/// a `[[seam::diag]]` entry takes a channel after it. Rust believes whatever the
/// declaration says, so a hand-written one that disagrees miscounts arguments at
/// RUN TIME rather than failing to build; `check-launch-seam.py` was added to
/// police exactly that and found four live mismatches on its first run.
///
/// Generating them removes the class instead of checking it. The prose that used
/// to sit between the declarations is kept in `driver/launch.rs`, above the
/// include, because it describes groups of kernels and no single declaration
/// knows it.
fn generate_launch_externs() {
    let Some(body) = render_kernel_fragments("externs", |line| {
        line.starts_with("    fn ") || line.starts_with("        ") || line == "    );"
    }) else {
        return;
    };
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    let target = Path::new(&out_dir).join("launch_externs.rs");
    let text = format!(
        "// Generated by ppf-cts-compute/seam/kernelgen.py --emit externs.\n\
         // One declaration per generated entry point, in sorted source order.\n\
         //\n\
         // THE TABLE IS COMPLETE AND A GIVEN BUILD DISPATCHES A SUBSET. An\n\
         // abi_backend_linked build reaches its kernels through the backend\n\
         // library rather than through these symbols, so every one of them is\n\
         // unreachable there and rustc is right to say so. Which entries a\n\
         // build actually dispatches is not a question the dead-code lint can\n\
         // answer, and check-launch-seam.py is what does.\n\
         #[cfg_attr(abi_backend_linked, allow(dead_code))]\n\
         extern \"C\" {{\n{body}}}\n"
    );
    std::fs::write(&target, text).expect("writing launch_externs.rs");
}

/// The `generated_thunk*!` invocations, rendered from the same declarations.
///
/// A thunk is the safe Rust wrapper around one `extern` symbol, and every line
/// of it followed from the entry declaration already: which macro, from whether
/// the entry is group-shaped or carries a diagnostic lane; the argument record,
/// from the entry's name in camel case; the wrapper's own name, from that name
/// with a `launch_` prefix. Twenty-five launchers were named against that last
/// rule and were renamed so it holds with no exception, which is what made the
/// block derivable rather than merely repetitive.
///
/// What that buys is the same thing the extern block bought: a thunk cannot go
/// on naming a record the declaration has stopped carrying, because there is no
/// second place for the two to disagree.
fn generate_launch_thunks() {
    let Some(body) = render_kernel_fragments("thunks", |line| {
        line.starts_with("generated_thunk") || line.starts_with("    ") || line == ");"
    }) else {
        return;
    };
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    let target = Path::new(&out_dir).join("launch_thunks.rs");
    let text = format!(
        "// Generated by ppf-cts-compute/seam/kernelgen.py --emit thunks.\n\
         // One wrapper per generated entry point, in sorted source order.\n{body}"
    );
    std::fs::write(&target, text).expect("writing launch_thunks.rs");
}

/// Render one emit kind over every neutral kernel and keep the lines that pass.
///
/// The generator renders a banner beside its output, and which lines carry the
/// content differs per emit kind, so the caller supplies the predicate. Sources
/// are sorted, which is what makes the concatenation reproducible.
fn render_kernel_fragments(emit: &str, keep: impl Fn(&str) -> bool) -> Option<String> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.parent().and_then(Path::parent)?;
    let generator = root.join("crates/ppf-cts-compute/seam/kernelgen.py");
    let kernels = manifest.join("src/kernels");
    if !generator.exists() || !kernels.is_dir() {
        return None;
    }
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    let fragment = Path::new(&out_dir).join(format!("launch_{emit}.fragment"));
    let mut sources: Vec<std::path::PathBuf> = Vec::new();
    collect_kernel_sources(&kernels, &mut sources);
    sources.sort();
    let mut body = String::new();
    for source in &sources {
        let out = Command::new("python3")
            .arg(&generator)
            .arg("--target").arg("rust")
            .arg("--emit").arg(emit)
            .arg("--out").arg(&fragment)
            .arg("--kernel-root").arg(&kernels)
            .arg(source)
            .output();
        let Ok(out) = out else { return None };
        if !out.status.success() {
            panic!(
                "ppf-cts-solver: rendering the {} block for {} failed.\n{}{}",
                emit,
                source.display(),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
        if let Ok(text) = std::fs::read_to_string(&fragment) {
            for line in text.lines() {
                if keep(line) {
                    body.push_str(line);
                    body.push('\n');
                }
            }
        }
    }
    Some(body)
}

fn collect_kernel_sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_kernel_sources(&path, out);
        } else if path.to_string_lossy().ends_with(".kernel.cpp") {
            out.push(path);
        }
    }
}

fn main() {

    // Every CUDA architecture this project targets comes from cuda_arch.txt.
    // `arch_guard` below checks the ARTIFACT matches the manifest; this checks
    // the SOURCE never states an architecture of its own, which is the failure
    // the artifact check cannot see: a build file pinned to its own arch
    // produces a perfectly self-consistent image of the wrong thing, and the
    // kernel-test harness sat at a hardcoded sm_70 through several floor moves
    // exactly that way. Runs unconditionally, CUDA present or not.
    arch_literal_guard::check();

    generate_launch_externs();
    generate_launch_thunks();
    check_launch_seam();
    compute_libdir();
    let backend = select_backend();
    record_backend(backend);
    // Publish the chosen backend to the Rust side. The driver in backend.rs is
    // shared by all three, and a few of its steps are meaningful only against a
    // CUDA device (the per-step nvidia-smi clock sample is the live one). The
    // feature flags cannot answer that question: a normal build enables none of
    // them and lets this file pick, so `feature = "cuda"` is false on the very
    // build that links CUDA. Deciding it here is deciding it where it is known.
    println!("cargo:rustc-check-cfg=cfg(cuda_backend)");
    if backend == Backend::Cuda {
        println!("cargo:rustc-cfg=cuda_backend");
    }
    // Whether `src/main.rs` compiles the neutral driver. It is decided here
    // rather than spelled at the module, because the module is the one thing in
    // the crate that must not know a target exists: gating it on a backend's
    // own name would say the driver belongs to that backend, and it belongs to
    // all of them. `Backend::links_neutral_driver` carries the reason a C++
    // backend answers false, and why getting that wrong is silent.
    println!("cargo:rustc-check-cfg=cfg(neutral_driver)");
    // Whether this build LINKS a library implementing the seam's C ABI, so the
    // `ppf_cts_compute::abi::AbiDevice` route to it can be exercised. The name
    // states the fact and not the target, because the fact is what a Rust file
    // may read: rule (1d) keeps every backend name in this file and in
    // Cargo.toml, and a test gated on a target's name would be the first
    // sentence of a fork.
    println!("cargo:rustc-check-cfg=cfg(abi_backend_linked)");
    // The DRIVER feature, not the ABI one. `cuda-abi` builds and links the
    // library; this asks for the neutral driver to be what calls it. They are
    // separate because `cuda-abi` is a build gate that must keep producing a
    // binary the C++ orchestrator runs.
    // NEITHER TARGET CONSULTS IT ANY MORE: both answer yes unconditionally,
    // because both orchestrators are deleted and the driver is the only
    // `advance()` either one has.
    let abi_backend_feature = env::var("CARGO_FEATURE_CUDA_DRIVER").is_ok()
        || env::var("CARGO_FEATURE_METAL_ABI").is_ok();
    if backend.links_neutral_driver(abi_backend_feature) {
        println!("cargo:rustc-cfg=neutral_driver");
    }
    if backend != Backend::Cuda {
        println!("cargo:warning=building the {} backend", backend.label());
    }

    // The CPU backend is Rust, so there is no C++ backend library to make and
    // nothing to link against. Everything below this point is about producing
    // and validating one, so it returns here rather than threading a `None`
    // through it: no `make`, no CUDA toolkit pin, no arch guard, no FP64 SASS
    // guard, and no `rustc-link-lib` for a backend library. `cc` emits its own
    // link directives for the kernel entry points it just compiled.
    // THE ABI FEATURE IS CHECKED BEFORE ANY BACKEND RETURNS, because the
    // failure it prevents is a link error two crates away: the feature turns on
    // the compute crate's `extern "C"` block, and if this build then produces
    // no library exporting `be_*` the binary fails to link with a list of
    // undefined symbols and nothing naming the feature that asked for them.
    if env::var("CARGO_FEATURE_CUDA_ABI").is_ok() {
        if backend != Backend::Cuda {
            panic!(
                "\n\n  --features cuda-abi builds the CUDA target's C ABI \
                 library, and this build selected {} instead. The two \
                 artifacts come from one recipe and one toolchain, so the \
                 feature cannot be combined with another backend.\n\n",
                backend.label()
            );
        }
        if cfg!(target_os = "windows") {
            panic!(
                "\n\n  --features cuda-abi has no Windows path yet: the C ABI \
                 library is produced by the make recipe, and the Windows build \
                 goes through build-win-native/build.bat instead.\n\n"
            );
        }
    }
    if env::var("CARGO_FEATURE_METAL_ABI").is_ok() && backend != Backend::Metal {
        panic!(
            "\n\n  --features metal-abi builds the Metal target's C ABI \
             library, and this build selected {} instead. The two artifacts \
             come from one recipe and one toolchain, so the feature cannot be \
             combined with another backend.\n\n",
            backend.label()
        );
    }
    if backend == Backend::Cpu {
        build_kernels();
        return;
    }
    // THE DRIVER'S HOST ARM NEEDS ITS ENTRY POINTS WHEREVER THE DRIVER IS
    // COMPILED, and that is now more than one backend. `launch.rs` declares the
    // generated entry points and the remaining hand-written shims in an
    // unconditional `extern "C"` block, so a build with the driver live and
    // this call skipped fails at link with a list of undefined symbols naming
    // every one of them. That is the loud direction, and it is why this sits
    // beside the CPU return rather than inside it.
    //
    // The objects it produces are not what a C ABI build DISPATCHES through:
    // that goes through `be_encode_dispatch` into the linked library. They are
    // what the host arm and the driver's own tests bind to.
    if backend.links_neutral_driver(abi_backend_feature) {
        build_kernels();
    }

    // Safe for every remaining backend: only `Backend::Cpu` answers `None`, and
    // it returned above.
    let cpp_dir = backend.cpp_dir().expect("a C++ backend names a directory");
    let lib_name = backend.lib_name().expect("a C++ backend names a library");
    // Set beside every `abi_backend_linked` emission below, and asserted at the
    // end of this function. The two link blocks are written per PLATFORM, so a
    // cfg emitted in one and forgotten in the other compiles and tests clean
    // everywhere; this is what turns that into a build failure.
    #[allow(unused_mut)]
    let mut abi_linked = false;

    #[cfg(not(target_os = "windows"))]
    {
        let out_dir = env::var("OUT_DIR").unwrap();
        let num_threads = num_cpus::get();
        println!("cargo:rerun-if-changed={cpp_dir}");
        // EVERY C++ BACKEND COMPILES THE NEUTRAL TREE, so every one of them
        // watches it. It holds the shared headers and the kernel bodies
        // (data.hpp, the shared linear algebra, SimpleLog). Without this, cargo
        // skips re-running make when a shared struct like FixPair or ParamSet
        // changes, leaving a stale library whose layout disagrees with the Rust
        // repr(C) structs, which surfaces as a SIGBUS at run time rather than
        // as a build error.
        println!("cargo:rerun-if-changed=src/kernels");
        match backend {
            Backend::Metal => {}
            Backend::Cuda => {
                println!("cargo:rerun-if-changed=../../eigsys/eig-hpp");
            }
            // Nothing outside its own directory and the neutral kernel tree,
            // both already watched above.
            Backend::Rocm => {}
            // Returned above, before `cpp_dir` was unwrapped. Spelled out
            // rather than folded into a wildcard so a fourth C++ backend added
            // later has to answer this question instead of inheriting an arm.
            Backend::Cpu => unreachable!("the CPU backend builds no C++ library"),
        }
        // WHERE THE NEUTRAL KERNEL TREE IS, NAMED BY THE CRATE THAT OWNS IT.
        // The recipe belongs to `ppf-cts-compute` and these sources belong to
        // this crate, so the path between them is an argument rather than a
        // constant on either side: a recipe there that hardcoded a path back
        // into here could no longer be published on its own, which is the one
        // test that crate is held to.
        let kernel_root = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("src")
            .join("kernels");
        // THE CUDA AND ROCm RECIPES ARE HANDED BOTH PATHS RELATIVE TO THEIR OWN
        // DIRECTORY, because their compilers record the path they are given in
        // every `__FILE__`. The device image keeps its asserts live, so host
        // code and device code alike carry the source path of each assert, and
        // an absolute spelling puts this machine's build directory into the
        // library that ships, which build-linux-native/bundle.sh refuses. A
        // compiler flag does not remove it on NVIDIA's side: measured on nvcc
        // 12.8, `-Xcompiler -ffile-prefix-map` rewrites the host half and leaves
        // the device half, whose frontend expands `__FILE__` before the host
        // compiler runs, and the ROCm recipe drives nvcc on its NVIDIA platform.
        // On AMD's platform, measured with TheRock 10.0.0's hipcc, absolute
        // spellings left 272 build-tree paths in libppfbe_rocm.so. Relative
        // spellings leave every half naming a file relative to the recipe, which
        // is still a diagnostic a reader can follow and names no machine. The
        // distance is computed here, so neither crate writes it down. Metal
        // keeps absolute paths; only the CUDA and ROCm paths were measured.
        let (kernel_root_arg, out_dir_arg) = if matches!(backend, Backend::Cuda | Backend::Rocm) {
            /// `target` spelled relative to `base`, both canonicalized first.
            fn relative_to(base: &Path, target: &Path) -> std::path::PathBuf {
                let canon = |p: &Path| {
                    p.canonicalize()
                        .unwrap_or_else(|e| panic!("build.rs: {} is unreadable: {e}", p.display()))
                };
                let (base, target) = (canon(base), canon(target));
                let base: Vec<_> = base.components().collect();
                let target: Vec<_> = target.components().collect();
                let common = base.iter().zip(&target).take_while(|(a, b)| a == b).count();
                let mut relative = std::path::PathBuf::new();
                for _ in common..base.len() {
                    relative.push("..");
                }
                for component in &target[common..] {
                    relative.push(component.as_os_str());
                }
                relative
            }
            (
                relative_to(Path::new(cpp_dir), &kernel_root),
                relative_to(Path::new(cpp_dir), Path::new(&out_dir)),
            )
        } else {
            (kernel_root.clone(), std::path::PathBuf::from(&out_dir))
        };
        // Toolkit pin and the SASS single-precision guard are both CUDA facts:
        // one interrogates the recipe's nvcc, the other reads cubins with
        // cuobjdump. Metal has neither, so they key on the selected backend
        // rather than on the backend merely not being CUDA.
        if backend == Backend::Cuda {
            require_cuda_12_8(cpp_dir, &kernel_root_arg);
        }
        // THE C ABI LIBRARY IS A SECOND ARTIFACT FROM THE SAME RECIPE, asked
        // for by name so a build that is not routing the driver at this target
        // does not pay a second five-architecture device link for an image it
        // will not load.
        // Already validated above, before any backend returned.
        // EITHER FEATURE ASKS THE RECIPE FOR THE ABI LIBRARY, and which one is
        // valid was already settled above: the pre-flight panics if the feature
        // and the selected backend disagree, so reading both here cannot ask a
        // recipe for an artifact it does not build.
        let abi_backend = env::var("CARGO_FEATURE_CUDA_ABI").is_ok()
            || env::var("CARGO_FEATURE_METAL_ABI").is_ok();
        let mut make = Command::new("make");
        make.current_dir(cpp_dir)
            .arg(format!("OUT_DIR={}", out_dir_arg.display()))
            .arg(format!("KERNEL_ROOT={}", kernel_root_arg.display()));
        if backend == Backend::Cuda || backend == Backend::Metal || abi_backend {
            // THE ABI LIBRARY AND NOTHING ELSE. On CUDA `all` builds an
            // orchestrator that no longer exists; on Metal the feature that
            // asks for this library is the same one that routes the driver at
            // it, so the orchestrator it would replace is not built either.
            make.arg("abi");
        }
        // WHERE THE BACKEND LOGIC TREE IS, on the same terms as the neutral
        // one: absolute, because make runs with `cpp_dir` as its working
        // directory, and named by the caller so the recipe hardcodes no path
        // between crates.
        if let Some(logic_dir) = backend.logic_dir() {
            let logic_root = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join(logic_dir);
            let logic_root = logic_root
                .canonicalize()
                .unwrap_or_else(|e| panic!("build.rs: {} is unreadable: {e}", logic_root.display()));
            // Watched for the same reason the neutral tree is: cargo would
            // otherwise skip make after an edit there and leave a library built
            // from the previous sources.
            println!("cargo:rerun-if-changed={logic_dir}");
            make.arg(format!("BACKEND_LOGIC_ROOT={}", logic_root.display()));
        }
        let output = make
            .arg(format!("-j{num_threads}"))
            .output()
            .expect("Failed to execute make command");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("make command failed.\nstdout:\n{stdout}\nstderr:\n{stderr}");
            std::process::exit(1);
        }

        // A CHECK THE RECIPE COULD NOT RUN IS SAID OUT LOUD, because nothing
        // above would say it. `make`'s output is captured and printed only on
        // FAILURE, so a recipe that skipped something and exited 0 is silent
        // here, which is the shape of a gate reporting success over nothing. A
        // recipe leaves such an account in `build-check-status.txt` and deletes
        // it when the check ran, and this turns it into a cargo warning, which
        // cargo prints for the crate being built. The name is the mechanism's,
        // not any one check's: this script learns nothing about which target
        // wrote it or what it was checking.
        let status_note = Path::new(&out_dir).join("build-check-status.txt");
        if let Ok(text) = std::fs::read_to_string(&status_note) {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                println!("cargo:warning={line}");
            }
        }

        let mut dir = std::env::current_dir().expect("Failed to get current directory");
        dir.push(out_dir);
        dir.push("lib");

        // Both guards below read a CUDA cubin image, so they are gated on the
        // CUDA backend specifically. A Metal build produces
        // libsimbackend_metal.dylib, which
        // carries no cubins at all, so either guard would be reading the wrong
        // artifact there.
        // THE METAL ABI LIBRARY IS THE ONLY ONE, so this is no longer a
        // question the caller answers.
        let abi_metal = backend == Backend::Metal;
        if backend == Backend::Cuda {
            // THE IMAGE THE GUARD READS is whichever library this backend
            // built: CUDA builds the ABI one now and Metal still builds its
            // orchestrator.
            let image = if backend == Backend::Cuda {
                dir.join("libppfbe_cuda.so")
            } else {
                dir.join(format!("lib{lib_name}.so"))
            };
            // The image must carry exactly the architectures the manifest asked
            // for. Checked before the SASS scan below, since that scan
            // enumerates the same architectures and a mis-enumeration there is
            // this failure seen one step later.
            arch_guard::check(&image, cpp_dir);
            // The device image just produced must run entirely in single
            // precision. Read that off the compiled SASS, not the source.
            fp64_guard::check(&image);
        }

        println!("cargo:rustc-link-search=native={}", dir.display());
        // NO `PPF_BACKEND_LIBRARY_DIR` IS EMITTED, deliberately. This used to
        // record where the backend put its artifacts so a library that loads a
        // pre-built shader could be told. What it recorded was an absolute path
        // into THIS machine's build tree, compiled into the binary, so a copy of
        // that binary on another machine pointed at a directory that does not
        // exist there. The Metal library resolves its own location instead, and
        // the CUDA one embeds its kernels and never needed a path at all.
        match backend {
            Backend::Metal => {
                // Dynamic, like the CUDA backend, and here it is required
                // rather than preferred: the dylib deliberately leaves
                // print_rust undefined so it binds to this binary's exported
                // symbol at load time. Measured over six link configurations:
                // a dylib carrying its own stub swallows every backend log line
                // instead. SimpleLog is compiled into the dylib, so unlike the
                // CUDA arm there is no separate simplelog library to link.
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
                if !abi_metal {
                    println!("cargo:rustc-link-lib=dylib={lib_name}");
                }
                // The other half of that seam, and it is load-bearing: the
                // BINARY has to export _print_rust for the dylib's undefined
                // reference to bind to. It does not by default here, because
                // the workspace release profile sets strip = "symbols", and an
                // unbound dynamic_lookup reference resolves to address 0, so
                // the first log line from the backend jumps to null. Measured:
                // without this flag the solver segfaults inside initialize()
                // at its first logging::info.
                //
                // `-bins`, NOT the unscoped form, because the unscoped one
                // reaches every target of this crate INCLUDING the test
                // binaries, and only the `[[bin]]` defines `print_rust`
                // (`main.rs:65`). Asking the linker to export a symbol a test
                // binary does not have makes it an `<initial-undefines>` and
                // the link fails, so `cargo test` did not build at all on
                // macOS: measured on `kernel_gates`, `Undefined symbols for
                // architecture arm64: "_print_rust"`. The tests still link the
                // dylib and still get the rpath above; they simply never call
                // into it, every gate under `tests/kernels/` being a host C++
                // program the harness compiles itself.
                println!("cargo:rustc-link-arg-bins=-Wl,-exported_symbol,_print_rust");
                if abi_metal {
                    // INSTEAD OF THE ORCHESTRATOR, NOT BESIDE IT. Linking both
                    // is neither a compile nor a link error and the binary's
                    // own `advance()` wins, so the orchestrator's would be dead
                    // weight nothing calls and nothing reports.
                    println!("cargo:rustc-link-lib=dylib=ppfbe_metal");
                    println!("cargo:rustc-cfg=abi_backend_linked");
                    abi_linked = true;
                }
            }
            Backend::Cuda => {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
                println!("cargo:rustc-link-lib=dylib=simplelog");
                // THE ABI LIBRARY IS THE ONLY ONE. The orchestrator's
                // `libsimbackend_cuda` is gone with the tree that built it.
                println!("cargo:rustc-link-lib=dylib=ppfbe_cuda");
                println!("cargo:rustc-cfg=abi_backend_linked");
                abi_linked = true;
                if false {
                    // The rpath above already names this directory, so the ABI
                    // library needs only to be put on the link line. It is a
                    // SECOND library beside the orchestrator's rather than a
                    // replacement: nothing in the binary calls it, and what
                    // reaches it is the seam's own `extern "C"` block, which
                    // the cfg below turns on.
                    println!("cargo:rustc-link-lib=dylib=ppfbe_cuda");
                    println!("cargo:rustc-cfg=abi_backend_linked");
                }
            }
            // `abi_backend_linked` IS EMITTED HERE AND NOWHERE ELSE FOR THIS
            // BACKEND, and forgetting it is a defect that shipped once:
            // `driver/launch.rs` resolves `Backend` to
            // `HostDevice` without this cfg, so a build that LINKS the library
            // and omits the cfg runs the whole solve on the host renderings
            // while both identity strings still say the accelerated backend.
            Backend::Rocm => {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
                println!("cargo:rustc-link-lib=dylib=ppfbe_rocm");
                println!("cargo:rustc-cfg=abi_backend_linked");
                abi_linked = true;
            }
            Backend::Cpu => unreachable!("the CPU backend links no backend library"),
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Metal never reaches this branch: it needs macOS, and select_backend
        // refuses --features metal anywhere else.
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let lib_dir = format!(
            "{manifest_dir}\\{cpp_dir}\\build\\lib",
            cpp_dir = cpp_dir.replace('/', "\\")
        );
        println!("cargo:rustc-link-search=native={lib_dir}");

        {
            println!("cargo:rustc-link-lib=dylib=lib{lib_name}");
            // THE SOLVER ITSELF IMPORTS A DEVICE RUNTIME ONLY ON CUDA. It calls
            // nothing but the `be_*` ABI, and which runtime the backend DLL
            // imports is the DLL's own business: `libppfbe_rocm.dll` imports
            // `amdhip64` when it was built for AMD and `cudart` when it was
            // built through nvcc on the staging platform. Naming `cudart` here
            // for a ROCm build would also make the build REQUIRE `CUDA_PATH`,
            // so a Windows machine carrying only a ROCm SDK would fail on a
            // missing CUDA toolkit while building a backend that does not use
            // one. The CUDA arm keeps the import because `driver/` calls
            // `cudaGetLastError` and friends through the CUDA prologue.
            if !matches!(backend, Backend::Rocm) {
                let cuda_path = env::var("CUDA_PATH")
                    .expect("CUDA_PATH environment variable must be set");
                let cuda_lib_path = format!("{cuda_path}\\lib\\x64");
                println!("cargo:rustc-link-search=native={cuda_lib_path}");
                println!("cargo:rustc-link-lib=dylib=cudart");
            }
            // THE DISPATCH PATH IS A cfg, NOT A LINK, AND OMITTING IT RUNS THE
            // WHOLE SOLVE ON THE HOST. `driver/launch.rs` selects
            // `Backend = HostDevice` under `cfg(not(abi_backend_linked))` and
            // `Backend = AbiDevice` under it, so a build that LINKS the backend
            // library without setting this dispatches to the statically linked
            // host renderings instead and never calls the library it linked.
            // The note at the top of this file already describes the shape:
            // the binary's own definitions win and "every value gate then stays
            // green over a path nothing takes".
            //
            // MEASURED, AND IT SHIPPED. Windows omitted this while every other
            // platform emitted it, so the Windows bundle ran the neutral kernels
            // on the CPU under a binary that answers `--backend cuda`. On
            // `examples/headless.py`, one commit, one L40S: 84 msec per step on
            // Linux against 14,583 msec in the Windows bundle, with identical
            // physics counters (newton_steps 8, iter 175 against 155,
            // num_contact 5663 against 5587) and the GPU sitting at its 210 MHz
            // idle clock rather than 2520 MHz. Nothing was red: the host
            // renderings compute the same numbers, only about 170x slower, so
            // the whole CI suite passed over a GPU that was never used.
            println!("cargo:rustc-cfg=abi_backend_linked");
            abi_linked = true;
        }
    }

    // A GPU BACKEND THAT DID NOT SET `abi_backend_linked` IS A HOST-EXECUTING
    // BUILD WEARING A GPU NAME, so it fails HERE rather than shipping. This is
    // a property of the backend rather than of the platform, which is what the
    // Windows omission above proved the hard way: the two link blocks are
    // written per platform, so a cfg emitted in one and forgotten in the other
    // is invisible to every compiler and every test. `Backend::Cpu` returns
    // long before this point, and Metal reaches it only through the
    // `abi_metal` arm that sets the cfg beside its link line.
    assert!(
        !matches!(backend, Backend::Cuda | Backend::Metal | Backend::Rocm) || abi_linked,
        "the {} build links its backend library but never set \
         `abi_backend_linked`, so `driver::launch::Backend` would resolve to \
         `HostDevice` and the solve would run on the CPU with the library \
         linked and never called. Emit `cargo:rustc-cfg=abi_backend_linked` \
         beside the link lines for this platform.",
        backend.label()
    );
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
// The architectures the image actually carries must be the ones cuda_arch.txt
// asked for. The Makefile already derives its -gencode list from that manifest,
// so this is not checking for a second hand-maintained copy; it is checking that
// the toolchain emitted what it was told to. A cubin can go missing without the
// build failing: a stale object from an earlier arch list reaches the link, an
// nvcc version drops a target it does not know, or a hand-run make overrides the
// flags. Each of those ships a binary that the run-time gate then advertises
// support the image does not have, which is the exact failure a user meets as
// "no kernel image is available for execution on the device".
mod arch_literal_guard {
    use std::path::{Path, PathBuf};

    /// Directories that may contain a CUDA build file. Naming DIRECTORIES
    /// rather than files is what lets this catch a build file nobody has
    /// written yet, which a fixed list of paths cannot; `check` asserts each one
    /// exists, so a root that moves is a build failure rather than a silent
    /// hole.
    ///
    /// The compute crate rather than this one, because rule (1d) admits no
    /// backend name in the solver crate: every CUDA build file in this tree is
    /// under crates/ppf-cts-compute, and naming the whole crate covers the
    /// Metal and host recipes beside them at no cost, since neither can carry a
    /// CUDA architecture without this guard wanting to hear about it.
    const ROOTS: &[&str] = &["../ppf-cts-compute", "../../build-win-native"];

    /// The flag contexts an architecture can appear in. A digit straight after
    /// one of these is a literal; a `$`, `!` or `%` is a make / cmd.exe variable
    /// expansion, which is what deriving from the manifest looks like in each of
    /// the two build languages.
    const CONTEXTS: &[&str] = &[
        "arch=compute_",
        "arch=sm_",
        "code=sm_",
        "code=lto_",
        "code=compute_",
        // The AMD flag. It is listed WITHOUT the leading dashes for the same
        // reason the CUDA ones are: `--offload-arch=` and `--amdgpu-target=`
        // both end in `arch=` or `target=`, and matching the tail is what makes
        // one entry cover every spelling a recipe might use.
        "offload-arch=gfx",
        "amdgpu-target=gfx",
    ];

    fn is_build_file(p: &Path) -> bool {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name == "Makefile" || name.ends_with(".mk") || name.ends_with(".bat")
    }

    /// Collects every build file under `dir`, and watches each directory it
    /// actually descends into rather than the root, so that a build file added
    /// later re-runs this script without a build OUTPUT directory doing the
    /// same. cargo reads a directory in `rerun-if-changed` as "rerun if any
    /// descendant changes" and does not consult .gitignore, so watching the
    /// compute crate whole would put crates/ppf-cts-compute/build-tests, where
    /// that crate's regression binaries land, on the watch list and charge a
    /// full solver relink to every run of them.
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        println!("cargo:rerun-if-changed={}", dir.display());
        for e in entries.flatten() {
            let p = e.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Build outputs and vendored trees carry generated makefiles that
            // are not ours to police.
            if matches!(name, "target" | "build-tests" | ".git" | "node_modules") {
                continue;
            }
            if p.is_dir() {
                walk(&p, out);
            } else if is_build_file(&p) {
                out.push(p);
            }
        }
    }

    fn offenders(text: &str) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        for (n, line) in text.lines().enumerate() {
            for ctx in CONTEXTS {
                let mut from = 0usize;
                while let Some(i) = line[from..].find(ctx) {
                    let at = from + i + ctx.len();
                    // A DIGIT for the CUDA contexts, and for the AMD ones the
                    // context already consumed the `gfx`, so anything at all
                    // there is a literal. Spelled as "not a variable expansion"
                    // rather than "is a digit" because `gfx11-generic` starts
                    // with a digit and `gfx9-generic` does too, while a make or
                    // cmd.exe expansion starts with `$`, `!` or `%`.
                    let next = line[at..].chars().next();
                    let literal = if ctx.contains("gfx") {
                        next.is_some_and(|c| !matches!(c, '$' | '!' | '%'))
                    } else {
                        next.is_some_and(|c| c.is_ascii_digit())
                    };
                    if literal {
                        out.push((n + 1, line.trim().to_string()));
                    }
                    from = at;
                }
            }
        }
        out
    }

    pub fn check() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        for rel in ROOTS {
            let dir = root.join(rel);
            // PER ROOT, not over the total. The aggregate emptiness check below
            // is satisfied by any one root carrying build files, so a root that
            // has moved leaves its own tree unchecked while the guard reports a
            // pass, which is the same "proved nothing" failure one level down.
            assert!(
                dir.is_dir(),
                "arch_literal_guard root {} does not exist: it would check \
                 nothing under it while the other roots kept the guard green",
                dir.display()
            );
            walk(&dir, &mut files);
        }
        if files.is_empty() {
            panic!(
                "arch_literal_guard found no build files under {ROOTS:?}: it proved \
                 nothing. Fix the roots rather than letting the guard pass on no \
                 evidence."
            );
        }
        let mut found: Vec<String> = Vec::new();
        for f in &files {
            println!("cargo:rerun-if-changed={}", f.display());
            let Ok(text) = std::fs::read_to_string(f) else { continue };
            for (line_no, line) in offenders(&text) {
                let shown = f.strip_prefix(&root).unwrap_or(f);
                found.push(format!("  {}:{}: {}", shown.display(), line_no, line));
            }
        }
        if !found.is_empty() {
            panic!(
                "hardcoded CUDA architecture in a build file:\n{}\n\n\
                 cuda_arch.txt is the single source for every architecture this \
                 project targets, and each consumer derives from it. Read the floor \
                 or the cubin list from that file instead of naming an architecture \
                 here; the Makefile beside it has the awk one-liner both build paths \
                 use.",
                found.join("\n")
            );
        }
    }
}

mod arch_guard {
    use std::path::Path;
    use std::process::Command;

    /// Architectures `cuda_arch.txt` asks for, ascending.
    ///
    /// This parser is a second reader of the manifest, not a second copy of the
    /// list: it and `ppf-cts-core`'s exist so each build script can resolve the
    /// one file for itself, and a disagreement between them is a parser defect
    /// rather than drift. Any failure is fatal, for the reason the manifest
    /// gives: proceeding on an empty list checks nothing and reports a pass.
    fn manifest_arches(cpp_dir: &str) -> Vec<u32> {
        let manifest = Path::new(cpp_dir).join("cuda_arch.txt");
        let text = std::fs::read_to_string(&manifest).unwrap_or_else(|e| {
            panic!("cannot read {}: {e}", manifest.display())
        });
        let mut archs: Vec<u32> = text
            .lines()
            .filter_map(|line| {
                let mut field = line.trim().split_whitespace();
                match (field.next(), field.next()) {
                    (Some("cubin"), Some(v)) => Some(v.parse::<u32>().unwrap_or_else(|e| {
                        panic!("{}: cubin value {v:?} is not a number: {e}", manifest.display())
                    })),
                    _ => None,
                }
            })
            .collect();
        archs.sort_unstable();
        archs.dedup();
        if archs.is_empty() {
            panic!("{}: no 'cubin' lines found", manifest.display());
        }
        archs
    }

    /// Architectures the image carries, ascending, or `None` when `cuobjdump`
    /// is not installed.
    ///
    /// The distinction matters: an absent tool cannot answer the question,
    /// while a tool that answers with nothing is reporting an image with no
    /// device code. Collapsing the two into an empty list is how a check comes
    /// to pass on no evidence.
    fn image_arches(image: &Path) -> Option<Vec<u32>> {
        let out = Command::new("cuobjdump").arg("--list-elf").arg(image).output().ok()?;
        let listing = String::from_utf8_lossy(&out.stdout);
        let mut archs: Vec<u32> = Vec::new();
        for line in listing.lines() {
            let mut from = 0;
            while let Some(rel) = line[from..].find("sm_") {
                let s = from + rel;
                from = s + 3;
                let digits: String =
                    line[s + 3..].chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(v) = digits.parse::<u32>() {
                    archs.push(v);
                }
            }
        }
        archs.sort_unstable();
        archs.dedup();
        Some(archs)
    }

    pub fn check(image: &Path, cpp_dir: &str) {
        // A MISSING IMAGE IS A HARD ERROR, NOT A SKIP. This guard used to
        // return here in silence, so a build whose artifact path had moved
        // stayed green while the shipped-cubin list stopped being checked at
        // all, and the only evidence was the absence of a warning line nobody
        // reads for. `make` has already succeeded by the time this runs, so the
        // image is either where the recipe put it or the two disagree, and both
        // are defects.
        assert!(
            image.exists(),
            "build.rs: the CUDA recipe reported success and {} does not exist, \
             so the shipped-architecture check has nothing to read. The \
             artifact path and the recipe that writes it disagree; fix one \
             rather than letting this guard pass over a missing image.",
            image.display()
        );
        let want = manifest_arches(cpp_dir);
        let Some(got) = image_arches(image) else {
            println!(
                "cargo:warning=cuobjdump not found: the device image was not \
                 checked against cuda_arch.txt, so a missing cubin would not be \
                 caught until a run fails on the affected GPU"
            );
            return;
        };
        if got == want {
            // Report the evidence, so a later disagreement names the image.
            let listed =
                want.iter().map(|a| format!("sm_{a}")).collect::<Vec<_>>().join(", ");
            println!("cargo:warning=device image carries {listed}");
            return;
        }
        let missing: Vec<u32> = want.iter().copied().filter(|a| !got.contains(a)).collect();
        let extra: Vec<u32> = got.iter().copied().filter(|a| !want.contains(a)).collect();
        let render = |v: &[u32]| {
            if v.is_empty() {
                "none".to_string()
            } else {
                v.iter().map(|a| format!("sm_{a}")).collect::<Vec<_>>().join(", ")
            }
        };
        panic!(
            "the device image does not carry the architectures cuda_arch.txt \
             asks for. Requested: {}. Present: {}. Missing: {}. Unexpected: {}. \
             The manifest drives the -gencode list, so a mismatch means the \
             toolchain did not emit what it was asked for: a stale object from \
             an earlier arch list, or an nvcc that does not know a target. \
             Remove the build directory and rebuild; if it persists, the nvcc \
             in use cannot target the missing architecture.",
            render(&want),
            render(&got),
            render(&missing),
            render(&extra)
        );
    }
}

mod fp64_guard {
    use std::path::Path;
    use std::process::Command;

    // SASS mnemonics that execute in double precision.
    //
    // The arithmetic ones are D-prefixed and are matched as whole mnemonics.
    // The CONVERSIONS are not, and listing them as whole opcodes is a real hole
    // rather than a style question: a conversion spells BOTH widths, so the
    // double-to-float narrowing is `F2F.F32.F64`, and the string "F2F.F64"
    // does not occur in it anywhere. An image whose only double precision was a
    // narrowing therefore PASSED. Measured on a stale artifact carrying known
    // FP64: the whole-opcode list found 230 instructions and missed 120
    // `F2F.F32.F64`, the true total being 350.
    //
    // So conversions are caught by the `.F64` MODIFIER instead, checked
    // separately below. That is precise as well as complete: 64-bit INTEGER and
    // 64-bit WIDTH operations spell themselves `.S64`, `.U64` or a bare `.64`
    // (`IADD.64`, `LDG.E.64`, `ISETP.NE.S64`, `SHF.R.U64`), never `.F64`.
    const FP64_OPS: [&str; 7] = [
        "DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "DDIV", "MUFU.RCP64H",
    ];

    // The double-precision modifier, matched anywhere in an opcode's dotted
    // modifier list. This is what catches every conversion in either direction
    // (`I2F.F64.S64`, `F2F.F32.F64`, `F2I.S64.F64`) without enumerating the
    // cross product of widths.
    const FP64_MODIFIER: &str = ".F64";

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
    // Whether the opcode on `line` carries the double-precision modifier.
    //
    // Restricted to the OPCODE, which is the first whitespace-delimited token
    // after the address comment, so a `.F64` appearing in an operand, a comment
    // or a demangled function name cannot raise a false alarm. The modifier
    // itself is unambiguous: 64-bit integer and 64-bit width operations spell
    // `.S64`, `.U64` or a bare `.64`, never `.F64`.
    fn contains_fp64_modifier(line: &str) -> bool {
        line.split_whitespace()
            .find(|tok| {
                !tok.starts_with("/*") && !tok.starts_with("@") && !tok.is_empty()
            })
            .is_some_and(|opcode| opcode.contains(FP64_MODIFIER))
    }

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
            } else if FP64_OPS.iter().any(|op| contains_mnemonic(line, op))
                || contains_fp64_modifier(line)
            {
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
        // A MISSING IMAGE IS A HARD ERROR, NOT A SKIP, for the reason
        // arch_guard::check states above and with more at stake: this is the
        // check that holds the device to single precision, and a silent return
        // would retire that guarantee without a line of output.
        assert!(
            image.exists(),
            "build.rs: the CUDA recipe reported success and {} does not exist, \
             so the device single-precision check has nothing to read. The \
             artifact path and the recipe that writes it disagree; fix one \
             rather than letting this guard pass over a missing image.",
            image.display()
        );

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
