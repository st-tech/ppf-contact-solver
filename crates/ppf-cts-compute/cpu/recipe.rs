// File: crates/ppf-cts-compute/cpu/recipe.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Rendering neutral kernels for the host target and compiling them.
//!
//! What is produced is the LAUNCHER half of a backend: a loop, a gather and a
//! scatter around a shared body, with no arithmetic of its own. The bodies
//! arrive as the `cpp` renderings this module asks the transcompiler for, so
//! every target compiles one source and cannot drift in the arithmetic.

use std::path::{Path, PathBuf};
use std::process::Command;

/// What a caller names, and it is all it names.
///
/// Every field is a fact from the caller's own domain: where the neutral
/// sources are, which of them this target wants, which hand-written translation
/// units call them, and where the output goes. Nothing here decides how to
/// compile, and nothing here names a file this crate owns.
pub struct HostShim<'a> {
    /// Root of the neutral kernel sources.
    pub kernel_root: &'a Path,
    /// Stems under `kernel_root`, without the `.kernel.cpp` suffix.
    pub kernels: &'a [&'a str],
    /// The hand-written translation units that call the rendered bodies, in the
    /// order the caller wants them compiled.
    pub translation_units: &'a [&'a str],
    /// Where renderings and objects land. NEVER inside a watched source tree:
    /// a build script that watches a directory recursively is re-run by any
    /// descendant changing, so one artifact inside costs a full rebuild of that
    /// crate on every subsequent build.
    pub out_dir: &'a Path,
    /// The static library's name.
    pub lib_name: &'a str,
    /// Buffer fields the caller already fills with an (arena, offset) handle
    /// rather than a host address, as `(kernel stem, "Record.field")`.
    ///
    /// THIS RECIPE NEVER READS THE STRINGS. It selects the pairs whose stem it
    /// is rendering and hands the rest to the transcompiler, which refuses a
    /// name no declaration in that file carries. What a field means, and why it
    /// has moved, are the caller's business; that a record may name buffers
    /// both ways at once is the seam's.
    pub handle_fields: &'a [(&'a str, &'a str)],
}

/// What the caller has to know afterward, and nothing more.
pub struct HostShimArtifact {
    /// The ISA baseline the compiler was ASKED for.
    ///
    /// Returned rather than published here because the caller's own code reads
    /// it at run time to refuse a CPU that cannot run what was built for it.
    /// `flag_if_supported` silently drops an unknown `-march`, so what is
    /// recorded is the request and the agreement between the two is checked at
    /// run time rather than assumed.
    pub baseline: String,
    /// Where the renderings landed.
    pub generated_root: PathBuf,
    /// The caller's stems that declared an entry point, so have an
    /// `<stem>.entry.cpp` and an `<stem>.entry.rs` under `generated_root`.
    /// Sorted, and a subset of `HostShim::kernels`.
    ///
    /// Returned because the caller decides who COMPILES a rendering, and it
    /// cannot decide that without knowing which renderings exist. This is the
    /// recipe's own verdict, the one that actually produced the files, so a
    /// caller reading it is not re-deriving anything and the two cannot
    /// disagree. It says nothing about what any kernel computes.
    pub declaring: Vec<String>,
}

/// Renders every kernel the caller named and compiles the shim around them.
///
/// The `cargo:` lines this prints go to the CALLER's build script stdout, which
/// is the process this runs in, so the link directives `cc` emits reach the
/// crate that links them. That is the whole reason the recipe is a library
/// function and the invocation is the caller's: `cargo:rustc-link-lib` and
/// `cargo:rustc-link-search` propagate from a dependency, but the two link
/// ARGUMENTS a dynamically linked backend needs do not, so nothing may assume a
/// build script boundary sits between deciding and emitting.
///
/// # Panics
/// On any failure, with the tool's own output. A rendering that half-succeeded
/// or a compile that was skipped is a silently wrong binary, and this project
/// treats a loud failure as strictly better than a quiet path that masks one.
pub fn compile_shim(plan: &HostShim) -> HostShimArtifact {
    let (generated_root, declaring) = render_kernels(plan);
    let baseline = compile(plan, &generated_root);
    HostShimArtifact {
        baseline,
        generated_root,
        declaring,
    }
}

/// This crate's transcompiler, which turns one neutral kernel source into the
/// form each target compiles.
///
/// It is resolved against this crate's own manifest directory rather than named
/// by the caller, and that is the rule rather than a convenience: rendering a
/// kernel for a target is backend-specific work, so the renderer belongs to this
/// crate, and a caller that had to pass its path would be naming a file it does
/// not own and could not move. `CARGO_MANIFEST_DIR` is expanded when THIS crate
/// is compiled, so it points at this source tree whether the crate is consumed
/// as a path dependency or unpacked from a registry.
///
/// It is the VALIDATOR as well as the renderer: a construct it does not
/// understand is an error with a file, line and column, never a pass-through.
fn transcompiler() -> PathBuf {
    let seam = Path::new(env!("CARGO_MANIFEST_DIR")).join("seam/kernelgen.py");
    assert!(
        seam.exists(),
        "ppf-cts-compute: the transcompiler is missing at {}. It ships with this \
         crate; a build cannot render a kernel without it.",
        seam.display()
    );
    seam
}

/// Renders the named kernels to their `cpp` form and returns the generated root
/// with the stems that also got entry artifacts.
///
/// The renderings land under `<out_dir>/kernelgen`, mirroring their path under
/// the kernel root, exactly as the Metal build places theirs.
///
/// The transcompiler is the validator as well as the renderer, so a failure here
/// is reported with its own output rather than summarized.
fn render_kernels(plan: &HostShim) -> (PathBuf, Vec<String>) {
    let generator = transcompiler();
    println!("cargo:rerun-if-changed={}", generator.display());
    let generated_root = plan.out_dir.join("kernelgen");
    let mut declaring: Vec<String> = Vec::new();
    for kernel in plan.kernels {
        let source = plan.kernel_root.join(format!("{kernel}.kernel.cpp"));
        if !source.exists() {
            panic!(
                "ppf-cts-compute: {} names no neutral kernel source. The caller's \
                 kernel list names what its shim includes, so an entry with no \
                 file behind it is a rename that did not reach that list.",
                source.display()
            );
        }
        println!("cargo:rerun-if-changed={}", source.display());
        let rendered = generated_root.join(format!("{kernel}.kernel.cpp"));
        std::fs::create_dir_all(rendered.parent().unwrap()).unwrap_or_else(|e| {
            panic!("ppf-cts-compute: cannot create {}: {e}", rendered.display())
        });
        let out = Command::new("python3")
        .arg("-B")
            .arg(&generator)
            .arg("--target")
            .arg("cpp")
            .arg("--out")
            .arg(&rendered)
            .arg(&source)
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "ppf-cts-compute: cannot run python3 {}: {e}. The host \
                     target renders its kernels at build time, so python3 is a \
                     build requirement for it.",
                    generator.display()
                )
            });
        if !out.status.success() {
            panic!(
                "ppf-cts-compute: rendering {} failed.\nstdout:\n{}\nstderr:\n{}",
                source.display(),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
        if render_entry_artifacts(
            &generator,
            plan.kernel_root,
            &source,
            kernel,
            &generated_root,
            plan.handle_fields,
        ) {
            declaring.push((*kernel).to_string());
        }
    }
    declaring.sort();
    (generated_root, declaring)
}

/// Renders the entry point and argument record a neutral kernel declares.
///
/// An entry point is an argument record plus a thread index plus a call into
/// the neutral body, all three
/// mechanical, so writing one by hand is writing a mirror pair by hand. What
/// this produces is the `cpp` half, which the caller's entry translation unit
/// compiles, and the `rust` half, which the caller's kernel table includes. Both
/// come from the one declaration beside the body, so the record cannot disagree
/// with the shim.
///
/// WHICH KERNELS: the ones that declare an entry, read off the source rather
/// than kept in a list here. A declaration opens at column zero with one of the
/// two attributes that mark it, so a mention of either in prose (which is
/// always inside a `//` comment) is not one, and neither is the
/// `[[seam::device_fn]]` that opens an ordinary body. A miss here is loud
/// rather than silent: `cpp_cpu/entries.cpp` names the artifact it expects, so
/// a declaration this test did not see fails the build at that include.
///
/// A kernel that declares an entry and is NOT compiled by the caller's entry
/// translation unit is not an error: the declaration may exist for another
/// target, and the host target fails at LINK if a dispatch names a symbol it did
/// not compile.
///
/// Returns whether the source declared an entry, which is the same verdict that
/// decided whether the two artifacts exist. The caller is told rather than left
/// to re-derive it, so nothing downstream can spell this test a second way.
fn render_entry_artifacts(
    generator: &Path,
    kernel_root: &Path,
    source: &Path,
    kernel: &str,
    generated_root: &Path,
    handle_fields: &[(&str, &str)],
) -> bool {
    let text = std::fs::read_to_string(source)
        .unwrap_or_else(|e| panic!("ppf-cts-compute: cannot read {}: {e}", source.display()));
    let declares_entry = text
        .lines()
        .any(|line| {
            line.starts_with("[[seam::args]]")
                || line.starts_with("[[seam::entry]]")
                || line.starts_with("[[seam::entry(")
        });
    if !declares_entry {
        return false;
    }
    for (target, suffix) in [("cpp", "entry.cpp"), ("rust", "entry.rs")] {
        let rendered = generated_root.join(format!("{kernel}.{suffix}"));
        let mut command = Command::new("python3");
        command
            .arg("-B")
            .arg(generator)
            .arg("--target")
            .arg(target)
            .arg("--emit")
            .arg("entry")
            .arg("--out")
            .arg(&rendered)
            // The generated entry includes headers from the caller's kernel
            // tree, and this script does not sit inside it, so the root is
            // passed rather than deduced.
            .arg("--kernel-root")
            .arg(kernel_root);
        // Only the Rust twin has a choice to make: every C++ rendering spells a
        // buffer field as an `ArenaHandle` whatever the caller puts in it, and
        // the transcompiler refuses the flag on those targets rather than
        // ignoring it.
        if target == "rust" {
            for (stem, field) in handle_fields {
                if *stem == kernel {
                    command.arg("--handle-field").arg(field);
                }
            }
        }
        let out = command
            .arg(source)
            .output()
            .unwrap_or_else(|e| {
                panic!("ppf-cts-compute: cannot run python3 {}: {e}", generator.display())
            });
        if !out.status.success() {
            panic!(
                "ppf-cts-compute: rendering the {target} entry for {} failed.\nstdout:\n{}\nstderr:\n{}",
                source.display(),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
    true
}

/// Compiles the caller's translation units against the renderings.
///
/// Returns the ISA baseline the compiler was asked for.
fn compile(plan: &HostShim, generated_root: &Path) -> String {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        // ONLY the generated root. Putting `src/cpp` here as well would make
        // every `#include "solver/spmv.kernel.cpp"` ambiguous, because a
        // rendering sits at the same relative path as its neutral source.
        .include(generated_root)
        // And the path discipline above is not the only thing standing between
        // this build and the wrong file, because a rule that holds only while
        // everyone remembers it is not a rule. A neutral kernel carries
        // `[[seam::device_fn]]` and `[[seam::thread]]`, kernelgen.py removes
        // both when it renders, so those attributes reach a compiler in exactly
        // one situation: the neutral source was compiled instead of its
        // rendering. Unrecognized attributes are merely IGNORED by default, so
        // that mistake otherwise produces a clean object built from the wrong
        // file. Promoting the warning makes it a build error where it happens.
        .flag_if_supported("-Werror=attributes")
        .flag_if_supported("-Werror=unknown-attributes")
        // Explanations for WHY these are separate translation units belong with
        // the files, so the caller supplies the list in the order it wants.
        .files(plan.translation_units.iter().map(Path::new));

    // EVERY FLAG ABOVE AND BELOW IS SPELLED FOR GCC AND CLANG, AND `cl.exe` TAKES
    // NONE OF THEM. `flag_if_supported` drops an unknown flag without a word, so
    // on MSVC the attribute guard, the contraction policy and the ISA baseline
    // would each vanish while the build stayed green. Every one of them therefore
    // has an MSVC spelling below, passed with `flag` rather than
    // `flag_if_supported`, so a toolset that cannot honor one fails here instead
    // of producing a binary that quietly lacks it.
    let msvc = build.get_compiler().is_like_msvc();
    if msvc {
        // Warning C5030 is cl.exe's "attribute is not recognized", the same
        // diagnostic the two GNU spellings above promote.
        build.flag("/we5030");
    }

    // __FILE__ MUST NOT NAME THE MACHINE THAT BUILT THIS.
    //
    // The diagnostic macros embed `__FILE__`, and a kernel body reaches this
    // compiler through an include resolved against an absolute root, so every
    // assert in every body lays an absolute build-tree path into the binary.
    // Measured on macOS: the CPU solver carried 15 of them and the Linux one
    // 95, naming both the crate's own `src/kernels` and the generated
    // renderings under OUT_DIR.
    //
    // WHY IT IS A SHIPPING PROBLEM AND NOT A COSMETIC ONE. The macOS
    // distribution is gated by `build-mac-native/bundle.sh`, whose gate C
    // refuses any file anywhere in the payload that names the build tree, and
    // that gate is deliberately not widenable: a path compiled into a binary is
    // exactly what a load-command walk cannot see, which is the reason it
    // exists. So without this the CPU solver cannot be packaged at all.
    //
    // Mapping to a RELATIVE path rather than to a placeholder, because the
    // string still has a reader: an assert that fires names a file, and
    // `src/kernels/energy/model/shell_bend.kernel.cpp` locates it in any
    // checkout while an absolute path locates it only on the machine that is
    // not the one reporting the failure.
    //
    // `flag_if_supported` is doing real work here rather than being defensive:
    // MSVC has no equivalent, so a Windows build keeps its absolute `__FILE__`.
    // Nothing gates that today (`build-win-native/bundle.bat` has no gate C),
    // and this is the place a fix would go if one is ever wanted.
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        build.flag_if_supported(format!("-fmacro-prefix-map={manifest}/="));
    }
    build.flag_if_supported(format!(
        "-fmacro-prefix-map={}/=",
        generated_root.display()
    ));

    // Contraction is pinned rather than inherited. A host build that does not
    // contract compares badly against a device build that does, and the failure
    // is a wrong tolerance rather than a wrong answer, so it is easy to misread.
    // GNU maps `-ffp-contract=on` onto `off` before version 14, so `fast` is the
    // spelling that means what it says on both compilers.
    //
    // cl.exe has a policy of its own, and since Visual Studio 2022 its default
    // `/fp:precise` no longer contracts; `/fp:contract` is what permits fusion
    // under it. Both are stated so a change to the toolset's default cannot move
    // this backend onto the other side of the line.
    if msvc {
        build.flag("/fp:precise");
        build.flag("/fp:contract");
    } else {
        build.flag_if_supported("-ffp-contract=fast");
    }
    // THE FLAG IS NOT ENOUGH, AND ON ITS OWN IT IS THE TRAP IT LOOKS LIKE THE FIX
    // FOR. A contraction POLICY only decides whether the compiler is permitted to
    // fuse; it cannot fuse into an instruction the target does not have, and the
    // baseline `x86-64` ISA has no FMA. Measured on g++ 13.3 on this tree, over
    // `a * b + c`: `-O3 -ffp-contract=fast` emits 0 `vfmadd`, and the same line
    // with `-march=x86-64-v3` emits 1. So without a baseline carrying FMA the
    // shared bodies compile here as a NON-CONTRACTING oracle while nvcc
    // (`--fmad=true`) and the Metal shader compiler both contract, which is
    // exactly the wrong-tolerance failure the comment above warns about, arrived
    // at while appearing to have been prevented.
    //
    // x86-64-v3 (AVX2 + FMA, Haswell 2013 and later) is the baseline.
    // `PPF_HOST_BASELINE` overrides for an older machine, and takes the wrong
    // tolerance knowingly rather than silently.
    //
    // AARCH64 GETS A BASELINE TOO, AND THE REASON IS NOT CONTRACTION. NEON
    // carries FMA at the architectural baseline, so the fp-contract argument
    // above is already satisfied here and this flag is not needed for the
    // oracle to agree with CUDA and Metal. What it is for is TUNING: with no
    // `-mcpu` the compiler schedules for a generic ARMv8 core, and every Mac
    // this backend runs on is at least an Apple M1. `apple-m1` is therefore
    // the floor rather than a guess about the host, which is what makes it
    // safe to bake in: a binary built for it runs on every later Apple part,
    // and `-mcpu=native` would not be, because a build machine can be newer
    // than the machine that runs the artifact.
    //
    // WHAT THIS IS NOT: a width lever. NEON is 128-bit and M1/M2 carry no
    // SVE, so there is no ARM analogue of the x86 AVX-512 step. Whether the
    // tuning model alone moves anything is a MEASUREMENT, and the x86
    // precedent is discouraging: the x86 baseline lever moved a microbenchmark
    // and made no end-to-end difference at all. The lever exists here so the question can
    // be asked and overridden, not because the answer is assumed.
    //
    // RETURNED rather than published here. The variable it ends up in is read
    // by the CALLER's own code, so the caller emits it: this module decides
    // which instructions the compiler may use, which is mechanism, and says
    // nothing about who is told.
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=PPF_HOST_BASELINE");
    let requested = std::env::var("PPF_HOST_BASELINE").ok();
    let baseline = match arch.as_deref() {
        Ok("x86_64") => {
            let baseline = requested.unwrap_or_else(|| "x86-64-v3".to_string());
            if msvc {
                if let Some(level) = msvc_x86_arch(&baseline) {
                    build.flag(level);
                }
            } else {
                build.flag_if_supported(format!("-march={baseline}"));
            }
            baseline
        }
        Ok("aarch64") if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") => {
            let baseline = requested.unwrap_or_else(|| "apple-m1".to_string());
            // `-mcpu` rather than `-march`: on aarch64 clang spells the
            // core model that way, and `flag_if_supported` drops it silently
            // if the compiler disagrees, which is why the REQUEST is what is
            // recorded and the agreement is checked at run time.
            build.flag_if_supported(format!("-mcpu={baseline}"));
            baseline
        }
        // EVERY OTHER AARCH64 TARGET ASKS FOR NO FLOOR, AND THAT IS A DECISION,
        // NOT AN OMISSION. Contraction needs none, because FMA is in the base
        // A64 instruction set. And unlike the Mac there is no single part every
        // machine is at least: Linux and Windows on ARM span Graviton, Ampere and
        // Snapdragon cores that share only the architecture baseline, so any
        // `-mcpu` baked in here would be a guess about the machine that runs the
        // artifact. The compiler's generic ARMv8 scheduling is the floor, and
        // "none" is what the run-time check reads as nothing to verify.
        //
        // An override is honored rather than ignored, because a variable the
        // operator set and the build silently disregarded is a result they did
        // not ask for. cl.exe has no `-mcpu`, so there it is refused by name.
        Ok("aarch64") => match requested {
            None => String::from("none"),
            Some(core) if msvc => panic!(
                "ppf-cts-compute: PPF_HOST_BASELINE={core} cannot be honored by cl.exe \
                 for aarch64, which takes no -mcpu. Unset it: the build targets the \
                 ARMv8 baseline, which already carries the FMA contraction needs."
            ),
            Some(core) => {
                build.flag_if_supported(format!("-mcpu={core}"));
                core
            }
        },
        _ => String::from("none"),
    };
    // Two warning classes come from the SHARED headers rather than from the
    // shim, and both are properties of headers written for nvcc. `#pragma
    // unroll` in `linalg/eigsolve.hpp` is advice to nvcc about a loop whose trip
    // count is a compile-time constant either way, and ignoring it is the
    // correct host behavior rather than a problem to fix. The unused-function
    // set is the `static` helpers a header defines and this translation unit
    // does not reach: measured with g++ 13.3 at `-Wall -Wextra`, `eigvalues3`,
    // `eigvectors3x3` and `symm2x2` from that same header, plus `logging::info`
    // from `common.hpp`. Silencing exactly these two keeps a warning from the
    // shim itself visible, which is the reason to leave the rest of the warning
    // set on.
    build.flag_if_supported("-Wno-unknown-pragmas");
    build.flag_if_supported("-Wno-unused-function");
    build.compile(plan.lib_name);
    baseline
}

/// The `/arch:` flag for an x86-64 microarchitecture level, on cl.exe.
///
/// cl.exe has no `-march`, and its `/arch:` levels are named for instruction set
/// extensions rather than for the psABI levels this recipe records, so the
/// translation is written out and a level it does not know is refused rather
/// than guessed.
///
/// `x86-64` and `x86-64-v2` both build at cl.exe's own SSE2 default. For v2 that
/// is a lower set than requested and never a higher one, so the binary runs on
/// every machine the request covers; and neither level carries FMA, which is
/// the only thing this baseline exists to provide, so nothing the build is for
/// is lost between them.
///
/// # Panics
/// On a level with no translation here.
fn msvc_x86_arch(level: &str) -> Option<&'static str> {
    match level {
        "x86-64" | "x86-64-v2" => None,
        "x86-64-v3" => Some("/arch:AVX2"),
        "x86-64-v4" => Some("/arch:AVX512"),
        other => panic!(
            "ppf-cts-compute: PPF_HOST_BASELINE={other} has no cl.exe spelling. cl.exe \
             takes /arch: levels rather than -march, and this recipe translates \
             x86-64, x86-64-v2, x86-64-v3 and x86-64-v4; name one of those."
        ),
    }
}
