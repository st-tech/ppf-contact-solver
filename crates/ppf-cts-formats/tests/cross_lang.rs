// File: crates/ppf-cts-formats/tests/cross_lang.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Cross-language CBOR roundtrip. Reads fixtures emitted by Python's
// cbor2 (via tests/scripts/gen_fixtures.py) and decodes them through
// the same serde types the Rust runtime will use. This is the
// regression test that catches addon ↔ Rust schema drift.
//
// Regenerate with:
//   .venv/bin/python crates/ppf-cts-formats/tests/scripts/gen_fixtures.py

use ppf_cts_formats::envelope::from_cbor;
use ppf_cts_formats::kinds::param::PinOperation;
use ppf_cts_formats::kinds::{ParamPayload, ScenePayload, KIND_PARAM, KIND_SCENE};

/// Tolerance: producer-side numpy.float32 → .item() carries f32
/// quantization out to the wire, so 0.001 lands as ~0.001000000047...
/// in f64 form. Anything within ~1e-6 is the addon round-tripping
/// correctly; tighter equality would be a test bug.
fn approx_eq(a: f64, b: f64) {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!(
        diff <= 1e-6 * scale,
        "assertion failed: |{a} - {b}| = {diff} > {} (1e-6 relative)",
        1e-6 * scale,
    );
}

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cbor_fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {path:?}: {e}"))
}

#[test]
fn scene_fixture_decodes() {
    let bytes = fixture("scene.cbor");
    let payload: ScenePayload = from_cbor(KIND_SCENE, &bytes).expect("scene decode");

    assert_eq!(payload.len(), 3, "expected 3 groups (SHELL/ROD/STATIC)");
    assert_eq!(payload[0].group_type, "SHELL");
    assert_eq!(payload[1].group_type, "ROD");
    assert_eq!(payload[2].group_type, "STATIC");

    let canonical = &payload[0].objects[0];
    assert_eq!(canonical.uuid, "uuid-shell-1");
    assert_eq!(canonical.vert.as_deref().unwrap().len(), 4);
    assert_eq!(canonical.face.as_deref().unwrap()[0], [0, 1, 2]);
    assert_eq!(canonical.pin.as_deref().unwrap(), &[0, 3]);
    let stitch = canonical.stitch.as_ref().unwrap();
    assert_eq!(stitch.0, vec![[0, 1]]);
    assert_eq!(stitch.1, vec![1.0]);

    let dup = &payload[0].objects[1];
    assert_eq!(dup.mesh_ref.as_deref(), Some("uuid-shell-1"));
    assert!(dup.vert.is_none());
    let m = dup.transform.unwrap();
    assert_eq!(m[0][3], 2.5, "translated +2.5 on X");

    let rod = &payload[1].objects[0];
    assert_eq!(rod.edge.as_deref().unwrap().len(), 2);

    let static_anim = &payload[2].objects[0];
    let anim = static_anim.transform_animation.as_ref().unwrap();
    assert_eq!(anim.time, vec![0.0, 1.0, 2.0]);

    let static_ops = &payload[2].objects[1];
    let ops = static_ops.static_ops.as_ref().unwrap();
    assert_eq!(ops.len(), 3);
    assert_eq!(ops[0].op_type, "MOVE_BY");
    assert_eq!(ops[0].delta.unwrap(), [1.0, 0.0, 0.0]);
    assert_eq!(ops[1].op_type, "SPIN");
    assert_eq!(ops[1].angular_velocity.unwrap(), 90.0);
    assert_eq!(ops[2].op_type, "SCALE");
    assert_eq!(ops[2].factor.unwrap(), 0.5);
}

#[test]
fn param_fixture_decodes() {
    let bytes = fixture("param.cbor");
    let payload: ParamPayload = from_cbor(KIND_PARAM, &bytes).expect("param decode");

    // Scene-level params: kebab-case keys decode into snake_case fields.
    approx_eq(payload.scene.dt, 1e-3);
    assert_eq!(payload.scene.gravity, [0.0, -9.8, 0.0]); // f64 in producer, exact
    assert_eq!(payload.scene.frames, 59);
    assert_eq!(payload.scene.fps, 60);
    assert_eq!(payload.scene.friction_mode, "min");
    assert_eq!(payload.scene.csrmat_max_nnz, 10_000_000);
    approx_eq(payload.scene.inactive_momentum.unwrap(), 0.5);
    assert!(!payload.scene.disable_contact);

    // Two groups (SHELL, SOLID) wrapped in 3-tuples.
    assert_eq!(payload.group.len(), 2);
    let (shell_params, shell_names, shell_uuids) = &payload.group[0];
    assert_eq!(shell_params.model.as_deref(), Some("baraff-witkin"));
    approx_eq(shell_params.young_mod.unwrap(), 1000.0);
    approx_eq(shell_params.shrink_x.unwrap(), 1.0);
    assert!(shell_params.shrink.is_none(), "SHELL omits 'shrink'");
    assert_eq!(shell_names, &vec!["cloth_a".to_string()]);
    assert_eq!(shell_uuids, &vec!["uuid-shell-1".to_string()]);

    let (solid_params, _, _) = &payload.group[1];
    assert_eq!(solid_params.model.as_deref(), Some("snhk"));
    approx_eq(solid_params.shrink.unwrap(), 1.0);
    assert!(solid_params.shrink_x.is_none(), "SOLID omits 'shrink-x'");
    let cw = solid_params.collision_windows.as_ref().unwrap();
    assert_eq!(cw["uuid-solid-1"], vec![(0.5, 1.5), (2.0, 3.0)]);

    // pin_config: int vertex keys preserved through CBOR.
    let shell_pins = &payload.pin_config["uuid-shell-1"];
    let v0 = &shell_pins[&0];
    assert_eq!(v0.unpin_time, Some(1.0));
    assert_eq!(v0.pin_group_id.as_deref(), Some("uuid-shell-1:vg-a"));
    assert_eq!(v0.operations.len(), 2);
    match &v0.operations[0] {
        PinOperation::Spin(s) => assert_eq!(s.angular_velocity, 360.0),
        other => panic!("expected Spin, got {other:?}"),
    }
    match &v0.operations[1] {
        PinOperation::MoveBy(m) => assert_eq!(m.delta, [0.5, 0.0, 0.0]),
        other => panic!("expected MoveBy, got {other:?}"),
    }
    let v3 = &shell_pins[&3];
    assert!(v3.operations.is_empty());

    // Cross-stitch.
    let cs = payload.cross_stitch.as_ref().unwrap();
    assert_eq!(cs.len(), 1);
    assert_eq!(cs[0].stitch_stiffness, 0.75);
    assert_eq!(cs[0].target_points.as_ref().unwrap(), &vec![[1.0, 2.0, 3.0]]);

    // Invisible colliders.
    let ic = payload.invisible_colliders.as_ref().unwrap();
    assert_eq!(ic.walls.len(), 1);
    assert_eq!(ic.walls[0].keyframes.len(), 2);
    assert_eq!(ic.spheres.len(), 1);
    assert_eq!(ic.spheres[0].radius, 0.5);
}
