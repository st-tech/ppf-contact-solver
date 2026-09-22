// File: crates/ppf-cts-solver/tests/kernel_gates.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! REGRESSION GATES OVER THE NEUTRAL KERNEL HEADERS, compiled and run by
//! `cargo test`.
//!
//! Each one under `tests/kernels/` is a self-contained C++ program that
//! includes a neutral header, exercises it, prints a table a reader can follow,
//! and returns non-zero if the property it guards has been lost. This file
//! compiles each with the host compiler and asserts the exit status.
//!
//! WHY THEY ARE HOST PROGRAMS. Every one of them was a `.cu` in the holding
//! pen, compiled by nvcc and launched as a kernel, and none of what they assert
//! needs a device: they are properties of fp32 arithmetic and of the type
//! system, which a host compiler reads out of the same headers. The ones that
//! genuinely need a device are the ones whose subject names a warp or
//! threadgroup operation, and those are NOT here: `seam_host.h` leaves
//! `compute::simd_width`, `compute::shuffle_down` and `compute::threadgroup_barrier`
//! deliberately undefined, because inventing single-thread bodies for them
//! would replace a compile error with an oracle computing something else.
//!
//! WHAT A DEVICE RUN WOULD ADD, stated so nobody assumes it is nothing: a
//! device has its own flush-to-zero policy, and the MEASURED miss counts these
//! gates print differ between one and a host. None of them ASSERTS a count. The
//! underflow they turn on is twenty decades below the smallest fp32 subnormal,
//! so it is zero on any fp32 hardware, gradual or flushing.

use std::path::PathBuf;
use std::process::Command;

fn kernels_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/kernels")
}

fn gate_source(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/kernels")
        .join(format!("{name}.cpp"))
}

fn compiler() -> String {
    std::env::var("CXX").unwrap_or_else(|_| "c++".to_string())
}

fn have_compiler() -> bool {
    Command::new(compiler()).arg("--version").output().is_ok()
}

/// Compiles one gate and runs it, returning its stdout on success.
///
/// `-O2` because two of these sample hundreds of millions of float pairs and an
/// unoptimized build turns a second into a minute. It changes no verdict: none
/// of them is a timing test, and `-ffast-math` is NOT passed, which would.
fn run_gate(name: &str) -> String {
    let out_dir = std::env::temp_dir().join(format!("ppf-kernel-gate-{name}"));
    let mut compile = Command::new(compiler());
    compile
        .arg("-std=c++17")
        .arg("-O2")
        // A gate that includes a NEUTRAL BODY directly sees its
        // `[[seam::thread]]` and `[[seam::device_fn]]` attributes, which the
        // transcompiler normally strips for the host and which a host compiler
        // warns about one line at a time. They are the declaration language,
        // not code, and ignoring them is what the host rendering does too.
        .arg("-Wno-attributes")
        // One gate puts its third stage behind a deadline, because the property
        // it guards is that a call RETURNS at all: an unguarded rescale spins
        // forever on a collapsed frame, and a hang is a test that never reports
        // rather than one that fails. Reaching that deadline takes a thread.
        .arg("-pthread")
        .arg(format!("-I{}", kernels_root().display()))
        .arg("-o")
        .arg(&out_dir)
        .arg(gate_source(name));
    let built = compile.output().expect("the host compiler runs");
    assert!(
        built.status.success(),
        "{name} did not compile:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(&out_dir).output().expect("the gate runs");
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(
        run.status.success(),
        "{name} FAILED. Its own report follows, and the line naming the \
         property is the one to read:\n{stdout}\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    stdout
}

/// The edge-triangle pierce predicate must not be written as a PRODUCT of two
/// signed volumes: the product underflows while both factors are still ordinary
/// normal floats, and the crossing is silently missed. This is the final
/// penetration gate's own predicate, so a miss there is a penetration.
#[test]
fn the_pierce_predicate_does_not_underflow() {
    if !have_compiler() {
        eprintln!("SKIPPED: no C++ compiler on this host");
        return;
    }
    let out = run_gate("pierce_underflow");
    assert!(out.contains("PASS"), "{out}");
    // THE GATE MUST HAVE PROVED SOMETHING. If the product form missed nothing,
    // the eps sweep no longer reaches the underflow band and the gate passed
    // over an empty question, which its own report says in as many words.
    assert!(
        !out.contains("product form missed 0 of"),
        "the product form missed nothing, so this gate proved nothing:\n{out}"
    );
}

/// `normalize()` must not guard on `norm() > 0`: `squaredNorm()` underflows well
/// before `norm()` would, so the guard passes and the division returns a
/// NON-ZERO vector left unnormalized. Every consumer of a unit normal then
/// works with a direction whose length is not one.
#[test]
fn the_staged_and_fused_tet_hessians_agree() {
    // THE FUSED TET PATH IS DISPATCHED BY NOTHING, so its Hessian has never run
    // in a scene. `energy/tet_force.kernel.cpp`'s `tet_elastic_embed` calls
    // `tet_spectral_hessian_fused`, which folds the spectral Hessian, the 9x9
    // to 12x12 conversion and the mass into one accumulate; the staged assembly
    // the driver runs instead dispatches `tet_spectral_hessian` and
    // `tet_convert_hessian` and scales separately.
    //
    // The comment above the fused body asserts the two are byte-identical
    // expression for expression. `c4bcd850` reverted the driver at the fused
    // path for losing SPD-by-assembly on `examples/cards`, so that assertion is
    // worth holding to a measurement rather than reading: if the two Hessians
    // ever diverge, the fused path assembles a different matrix and the
    // revert's cause is here rather than in the wiring.
    if !have_compiler() {
        eprintln!("no host C++ compiler; skipping");
        return;
    }
    let out = run_gate("tet_hessian_agreement");
    assert!(
        out.contains("agree on 64 cases"),
        "the staged and fused tet Hessians disagree:\n{out}"
    );
}

#[test]
fn normalize_does_not_underflow_its_guard() {
    if !have_compiler() {
        eprintln!("SKIPPED: no C++ compiler on this host");
        return;
    }
    let out = run_gate("normalize_underflow");
    assert!(out.contains("PASS"), "{out}");
    // THE GATE MUST HAVE PROVED SOMETHING, on the same terms as the pierce one:
    // if the pre-fix form failed nothing, the magnitude sweep no longer reaches
    // the underflow band and the gate passed over an empty question.
    assert!(
        !out.contains("pre-fix form failed 0 of"),
        "the pre-fix form failed nothing, so this gate proved nothing:\n{out}"
    );
}

/// Regularized Coulomb friction has two branches for the FORCE and one Hessian
/// for both, and that Hessian is a MAJORIZER rather than the exact second
/// derivative: past the cone the capped potential is linear along the slip, so
/// the exact form `lambda * (P - s s^T)` is singular there and a solver whose
/// only line search is CCD cannot use it. What releases a saturated contact is
/// the ANCHOR the surrogate is tight at, not the shape of the Hessian: inside
/// the cone it is the current slip and the surrogate is the lagged one exactly,
/// and past it the anchor is the sliding slip of the pair-local model, so one
/// Newton step lands on the physical slip instead of creeping by a multiple of
/// `friction_eps` forever. This gate pins both, plus the two slip predictions
/// the anchor is built from.
#[test]
fn friction_is_a_majorizer_whose_anchor_releases_a_saturated_contact() {
    if !have_compiler() {
        eprintln!("SKIPPED: no C++ compiler on this host");
        return;
    }
    let out = run_gate("friction_branches");
    assert!(out.contains("friction surrogate and anchor passed"), "{out}");
}

/// The three intersection allowances take the pairs they cover out of contact,
/// the line search and the intersection report, and which pairs they cover is
/// a truth table rather than a rule of thumb: "either
/// side opts in" for the pin and inter-object cases, one side asked for
/// self-intersection, and an unknown object identity tolerates nothing. A
/// policy that read only the first side would answer a one-sided table
/// correctly and be wrong on half of every real scene, which is why every case
/// is also evaluated with its two sides exchanged.
#[test]
fn the_intersection_allowance_truth_table_holds() {
    if !have_compiler() {
        eprintln!("SKIPPED: no C++ compiler on this host");
        return;
    }
    let out = run_gate("intersect_allowance");
    assert!(out.contains("truth table passed"), "{out}");
    assert!(!out.contains("[FAIL]"), "{out}");
}

/// `FixedCSRMat::push` silently drops an out-of-pattern block, so the row a
/// dynamic matrix finalizes has to be exactly right: a missing sparsity slot
/// drops a Hessian block and leaves an indefinite matrix, which is the rod-bend
/// `(j, k)` stencil bug and is masked by damping and a small `dt`. This checks
/// the finalize against a straightforward oracle over 102 cases, including the
/// cold start, the steady state, and rows whose blocks all cancelled to zero.
#[test]
fn a_dynamic_csr_row_dedupes_against_its_oracle() {
    if !have_compiler() {
        eprintln!("SKIPPED: no C++ compiler on this host");
        return;
    }
    let out = run_gate("csr_row_dedupe");
    assert!(out.contains("PASS:"), "{out}");
}

/// The row search bisects a sorted pattern rather than scanning it, which is
/// what took a coarse-collider assembly from 37.0 s to 0.35 s per Newton step.
/// A bisection that is right on ordinary rows and wrong on an edge answers a
/// query with a slot holding a different column, which pushes a block into the
/// wrong place rather than failing. Checked against `std::sort` and a linear
/// scan over 66 cases.
#[test]
fn the_csr_row_pattern_sorts_finds_and_merges() {
    if !have_compiler() {
        eprintln!("SKIPPED: no C++ compiler on this host");
        return;
    }
    let out = run_gate("csr_row_pattern");
    assert!(out.contains("PASS:"), "{out}");
}

/// ACCD must not be handed a frame it cannot normalize. `rescale` divides
/// `0.99f` by the sweep's largest coordinate magnitude, and two frames send
/// that quotient to infinity: one with no extent at all, where two primitives
/// are coincident, and one whose extent underflows far enough that the quotient
/// leaves the float range. The loop below the quotient exits only when
/// `scale <= s`, so an infinite scale never satisfies it and the call NEVER
/// RETURNS: on CUDA that is a hung device with no assert and no output, and on
/// Metal there is no assert to lose. Guarding the QUOTIENT rather than the
/// divisor is what covers both, and a NaN extent with it.
///
/// The gate is therefore in three stages and the third is the one that matters:
/// it runs the four collapsed CCD entry points behind a deadline, because the
/// property under test is termination. A hang IS the failure, and reporting it
/// as one is why the deadline exists.
#[test]
fn accd_returns_on_a_degenerate_frame() {
    if !have_compiler() {
        eprintln!("SKIPPED: no C++ compiler on this host");
        return;
    }
    let out = run_gate("accd_degenerate");
    assert!(out.contains("PASS:"), "{out}");
    // THE COLLAPSE MUST HAVE HAPPENED. Each degenerate input is preceded by a
    // witness line reporting the extent the frame actually presents, against a
    // control that reports a nonzero one. If a witness stops reading zero the
    // input is no longer degenerate and the stage below it proves nothing.
    assert!(
        !out.contains("[FAIL]"),
        "a witness stopped reporting a collapsed extent:\n{out}"
    );
}

/// The distance a closest-point routine reports is what ACCD divides by to
/// bound its conservative advance, so reporting it too LARGE is the unsafe
/// direction: the advance is sized for a separation the pair does not have and
/// the sweep can step across the true time of impact. Both routines in
/// `contact/distance.hpp` once CLASSIFIED a feature and trusted it, which can
/// only over-report, and both let a penetration through: 54.6% of constructed
/// colliding edge-edge sweeps below 0.1 rad were certified collision-free, and
/// a barycentric-sign classification aborted `examples/large-animals` on a
/// 153-degree sliver. The fix is to build every candidate feature, score each
/// by the distance it realizes, and keep the smallest, so no parallel-edge or
/// obtuse-triangle threshold is ever needed.
///
/// Six gates, each against a reference in double that shares no implementation
/// with the routine it checks: the edge-edge and point-triangle reports against
/// truth, the `large-animals` sliver on its own, the barycentric inside verdict,
/// and the CCD itself over twenty thousand constructed crossings, head-on and
/// with a tangential glide up to a thousand times the approach rate. The two
/// CCD gates allow NOTHING: one certified crossing is one trajectory the line
/// search would have accepted through a collision.
///
/// It runs for about seventeen seconds, nearly all of it the double-precision
/// reference, which bisects a convex derivative at four hundred points of every
/// sweep. That cost is the gate: a reference sharing the arithmetic of the code
/// it checks could share its blind spot.
#[test]
fn the_closest_point_routines_do_not_over_report() {
    if !have_compiler() {
        eprintln!("SKIPPED: no C++ compiler on this host");
        return;
    }
    let out = run_gate("distance_coeff");
    assert!(out.contains("all contact-distance gates passed"), "{out}");
    // THE CROSSINGS MUST HAVE BEEN CONSTRUCTED. Both CCD gates pass vacuously
    // if the sampler stops producing sweeps that genuinely reach contact, and
    // each says so in its own report rather than leaving it to be inferred.
    assert!(
        !out.contains("no crossing was constructed"),
        "a CCD gate had nothing to certify, so it proved nothing:\n{out}"
    );
}

/// Lock Translation and Lock Rotation are EXACT constraints on the Newton
/// direction, never a penalty: no energy term, no Hessian block, no stiffness,
/// so nothing here is a tolerance and a row is either annihilated or the
/// constraint is not being enforced.
///
/// Two properties carry the feature. ONE PROJECTOR, NEVER A COMPOSITION: two
/// projections in sequence do not commute, so a second reintroduces a component
/// the first removed, and the three intersection cases are what detect that,
/// including one that feeds the projector a DUPLICATED constraint so the
/// orthonormalization is load-bearing rather than incidental. And the rotation
/// rows are ANCHOR-RELATIVE: the moment arm is measured from the group's own
/// anchor, so a row built from an absolute position is wrong by the anchor's
/// offset, an error that vanishes at the origin and grows across the domain.
///
/// The gate also reads the body's `[[seam::diag]]` channel, which the CUDA test
/// it replaces bound to the global record and never looked at. A basis that
/// failed to build leaves the reduced vector untouched, which is what a lock
/// with nothing to do also looks like, so the channel is the only thing that
/// separates them.
#[test]
fn the_aggregate_lock_projector_holds_its_constraints() {
    if !have_compiler() {
        eprintln!("SKIPPED: no C++ compiler on this host");
        return;
    }
    let out = run_gate("translation_lock");
    assert!(out.contains("PASS: aggregate lock projector gates passed"), "{out}");
    assert!(!out.contains("FAIL"), "{out}");
}

/// The block-Jacobi diagonal inverter tests its INPUT for finiteness, not its
/// eigenvalues, and this gate holds the two properties that make the input-side
/// test the only one that works.
///
/// `symm3x3` reduces its scale with `fmaxf` from 0.0f, and IEEE 754 maxNum
/// returns the non-NaN operand, so an all-NaN block leaves the scale at exactly
/// zero and takes the `scale <= 0.0f` early return: finite eigenvalues and an
/// identity basis. The same property then hides a NaN behind any real operand
/// in the inverter's own max over the spectrum. A guard built on
/// `isfinite(lmax)` therefore passes a block that is NaN in every entry, and
/// the positivity test below it reports a non-positive block when the real
/// fault is a non-finite one.
#[test]
fn a_nan_block_reaches_the_eigensolver_with_finite_eigenvalues() {
    if !have_compiler() {
        eprintln!("SKIPPED: no C++ compiler on this host");
        return;
    }
    let out = run_gate("eigen_nan_input");
    assert!(
        out.contains("a max-side finiteness test cannot see a NaN input"),
        "{out}"
    );
}
