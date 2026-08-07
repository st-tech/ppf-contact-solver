// File: cache_path_gates.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Regression gates for the tetrahedralize cache filename.
//
// Two composers produce the same name and must agree:
//
//   writer  frontend/_mesh_.py    -> tetra_cache_name + cache_path
//   planner frontend/_decoder_.py -> tetra_cache_name + cache_path
//
// Both reach the name through one composer, so they agree for any kwargs
// set by construction. The gate below fixes that property so a second
// composer cannot be reintroduced without failing here.
//
// The composed value is a single PATH COMPONENT, so it is bounded by the
// filesystem's per-component limit (255 bytes on NTFS, APFS and ext4
// alike), not by the total-path limit. Overrunning it makes `open()` fail
// with an errno the caller cannot act on.
//
// These tests live in an integration-test file rather than in the inline
// `#[cfg(test)] mod tests` of either source module, so a fix to those
// modules edits nothing here.
//
// The three gates below are ARMED against defects that are open. Each
// carries `#[should_panic(expected = ...)]` naming the wrong value the
// defect produces, which is the Rust counterpart of the Python gates'
// `pytest.mark.xfail(strict=True)`: the test runs in the blocking unit
// job on every platform, passes while the defect is present, and fails
// with "test did not panic as expected" the moment it is fixed. Answer
// that failure by deleting the attribute, which turns the test into the
// permanent regression gate its name describes.

use ppf_cts_core::datamodel::mesh::{cache_path, tetra_cache_name, tetrahedralize_arg_str};

/// A SHA-256 digest rendered as hex. Both composers interpolate one twice.
const HASH: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

/// The largest per-component length a filesystem this project ships on
/// accepts. NTFS, APFS and ext4 all cap a name at 255 bytes.
const MAX_COMPONENT_BYTES: usize = 255;

/// The seven per-object fTetWild overrides the Blender add-on can send,
/// each rendered the way `frontend/_mesh_.py` renders it: `str(v)` over
/// the value the encoder read out of the RNA property.
///
/// The three float fields are Blender `FloatProperty`, i.e. float32, so
/// widening one to a Python float prints its full float64 image: the 0.05
/// default reads back as `0.05000000074505806`, 19 characters for a value
/// the artist typed as 4. That expansion is why the component overruns at
/// realistic settings rather than at contrived ones.
fn all_overrides() -> Vec<(String, String)> {
    [
        ("edge_length_fac", "0.05000000074505806"),
        ("epsilon", "0.0010000000474974513"),
        ("stop_energy", "10.0"),
        ("num_opt_iter", "80"),
        ("optimize", "True"),
        ("simplify", "True"),
        ("coarsen", "False"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Compose the writer's filename component the way `Mesh.tetrahedralize`
/// does: build the arg string, wrap it in the `_tetrahedralize_` body, and
/// hand that to `cache_path` as the `name`.
fn writer_component(kwargs: &[(String, String)]) -> String {
    let arg_str = tetrahedralize_arg_str(&[], kwargs);
    let name = format!("{HASH}_tetrahedralize_{arg_str}.npz");
    file_name_of(&cache_path("cache", HASH, &name))
}

/// Compose the planner's filename component the way
/// `SceneDecoder._tetra_cache_path` does.
fn planner_component() -> String {
    file_name_of(&cache_path("cache", HASH, &tetra_cache_name(HASH, &[], &[])))
}

fn file_name_of(p: &std::path::Path) -> String {
    p.file_name()
        .expect("cache_path always ends in a filename")
        .to_string_lossy()
        .into_owned()
}

/// G1a. `cache_path` appends `.npz`, and every caller hands it a `name`
/// that already ends in `.npz`, so the composed component carries the
/// extension twice. Fixing the length alone leaves this in place, so it is
/// asserted separately.
///
/// PENDING dev-preexisting-fixes | G1a: `cache_path` appends a second
/// `.npz`. Delete the `should_panic` attribute when that is fixed.
#[test]
#[should_panic(expected = "component carries .npz twice")]
fn cache_path_does_not_double_the_npz_extension() {
    for n in 0..=all_overrides().len() {
        let component = writer_component(&all_overrides()[..n]);
        assert!(
            !component.ends_with(".npz.npz"),
            "{n} override(s): component carries .npz twice: {component}"
        );
    }
    assert!(
        !planner_component().ends_with(".npz.npz"),
        "planner component carries .npz twice: {}",
        planner_component()
    );
}

/// G1b. With every fTetWild override set to the value the add-on's own RNA
/// defaults produce, the component must still fit one filesystem name. The
/// bound is computed, never compared against a recorded number: a recorded
/// number would pin today's rendering rather than the property that the
/// name has to be openable.
///
/// PENDING dev-preexisting-fixes | G1b: the component reaches 266 bytes at
/// five overrides. Delete the `should_panic` attribute when that is fixed.
#[test]
#[should_panic(expected = "over the 255-byte filesystem limit")]
fn cache_path_component_fits_a_255_byte_filesystem() {
    let overrides = all_overrides();
    for n in 0..=overrides.len() {
        let component = writer_component(&overrides[..n]);
        assert!(
            component.len() <= MAX_COMPONENT_BYTES,
            "{n} override(s): component is {} bytes, over the {MAX_COMPONENT_BYTES}-byte \
             filesystem limit: {component}",
            component.len()
        );
    }
}

/// G2. The planner stats a path to decide whether an object's
/// tetrahedralization is already cached. It must stat the path the writer
/// writes, or an object carrying any override is reported "new" on every
/// build and re-runs fTetWild forever.
///
/// The kwargs here must stay NON-EMPTY. The two composers agree exactly
/// when the arg string is empty, so a zero-override case certifies
/// nothing; the assertion below therefore starts at one override.
///
/// PENDING dev-preexisting-fixes | G2: the planner composes an empty-arg
/// cache name. Delete the `should_panic` attribute when that is fixed.
#[test]
#[should_panic(expected = "the planner probes a different file than the writer writes")]
fn planner_probe_path_matches_the_writer_path() {
    let overrides = all_overrides();
    for n in 1..=overrides.len() {
        let writer = writer_component(&overrides[..n]);
        assert_eq!(
            planner_component(),
            writer,
            "{n} override(s): the planner probes a different file than the writer writes"
        );
    }
}

/// The zero-override case is the one configuration where the two composers
/// already agree. It is asserted so the gate above cannot be "fixed" by
/// making the planner's own default drift.
#[test]
fn planner_and_writer_agree_with_no_overrides() {
    assert_eq!(planner_component(), writer_component(&[]));
}
