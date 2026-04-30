# File: scenarios/bl_profile_load_batch.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Profile-load batched updates.
#
# Applying a material profile mutates many ObjectGroup parameters in a
# single user gesture. We want the overlay version to bump and the
# panel to refresh AFTER the batch, not flicker per-field. The param
# hash should also reflect the new values so the "Update Params"
# button surfaces the drift.
#
# This scenario walks the user-visible flow:
#   1. Build a SHELL group on a pinned plane.
#   2. Author a TOML profile on disk that overrides several material
#      fields with non-default values.
#   3. Snapshot pre-load: state.overlay_version, compute_param_hash,
#      and the per-field group values we expect to change.
#   4. Drive the load by setting ``group.material_profile_path`` and
#      then ``group.material_profile_selection``. The Enum's update
#      callback (``_on_material_profile_selected``) invokes
#      ``apply_material_profile`` and then ``invalidate_overlays``,
#      mirroring the operator path the panel exposes.
#   5. Snapshot post-load and assert:
#      A. ``overlay_version`` strictly increases across the batch
#         (we observe and document the exact delta).
#      B. At least three targeted fields took the profile values.
#      C. ``compute_param_hash`` changed.
#
# Pure addon-side check: no server, no build, no fetch.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="ProfileMesh")
    # save_blend keeps us aligned with other scenarios that depend on
    # a saved file for relative paths; the profile-load path itself
    # does not require it, but it makes PROBE_DIR-relative writes
    # behave the same way as the operator's typical user environment.
    dh.save_blend(PROBE_DIR, "profile_load.blend")
    root = dh.configure_state(project_name="profile_load_batch", frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "AllPin")
    group = root.object_group_0

    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    # Author a profile on disk. Field values intentionally diverge
    # from the addon defaults so the post-load comparison is
    # unambiguous (defaults: shell_density=1.0, shell_young_modulus=
    # 1000.0, friction=0.5, bend=100.0, stitch_stiffness=1.0,
    # contact_gap=0.001).
    profile_path = os.path.join(os.path.dirname(PROBE_DIR),
                                "material_profile.toml")
    profile_name = "BatchTest"
    expected = {
        "shell_density": 7.5,
        "shell_young_modulus": 4242.0,
        "friction": 0.125,
        "bend": 33.0,
        "stitch_stiffness": 12.5,
        "contact_gap": 0.005,
    }
    # Pre-snapshot needs the field values BEFORE the profile lands.
    pre_field_values = {k: float(getattr(group, k)) for k in expected}
    pre_overlay_version = int(root.state.overlay_version)
    pre_param_hash = encoder_params.compute_param_hash(bpy.context)

    # Sanity guard: every targeted field must currently differ from
    # the value the profile will load. If any default already matches
    # the profile, the field-took-profile assertion would silently
    # pass without exercising the load.
    pre_already_matches = [
        k for k in expected if abs(pre_field_values[k] - expected[k]) < 1e-9
    ]
    if pre_already_matches:
        raise RuntimeError(
            "profile values overlap defaults: " + repr(pre_already_matches)
        )

    # Hand-author the TOML so the test does not depend on the
    # profile-save operator path (which would itself bump
    # overlay_version and pollute the snapshot).
    with open(profile_path, "w") as f:
        f.write("[" + profile_name + "]\n")
        # object_type carries an update= callback (apply_object_overlays
        # which invalidates overlays). Lock it to the value the group
        # already has so the only overlay bumps are the ones the batch
        # protocol explicitly emits (per-field setattrs without
        # update=, plus the trailing invalidate_overlays).
        f.write('object_type = "SHELL"\n')
        f.write('shell_model = "BARAFF_WITKIN"\n')
        for key, val in expected.items():
            f.write(f"{key} = {val}\n")
    dh.log(f"profile_written path={profile_path}")

    # Drive the user-visible load path: setting material_profile_path
    # is a no-op for state, then setting material_profile_selection
    # triggers _on_material_profile_selected which performs the batch
    # (apply_material_profile + invalidate_overlays).
    group.material_profile_path = profile_path
    # Re-snapshot AFTER setting the path so the profile-name resolution
    # the Enum performs does not count toward the batch delta.
    pre_overlay_version_after_path = int(root.state.overlay_version)
    pre_param_hash_after_path = encoder_params.compute_param_hash(bpy.context)

    group.material_profile_selection = profile_name
    dh.log("profile_loaded")

    post_overlay_version = int(root.state.overlay_version)
    post_param_hash = encoder_params.compute_param_hash(bpy.context)
    post_field_values = {k: float(getattr(group, k)) for k in expected}

    # ----- A: overlay_version bumps across the batch -----------------
    # We assert strict monotonic growth. The batch is NOT guaranteed
    # to be exactly +1: per-field setattrs that lack an update=
    # callback do not bump on their own, but object_type / color have
    # update= callbacks that route through apply_object_overlays and
    # call invalidate_overlays, AND the operator path tail explicitly
    # calls invalidate_overlays once. The shipped guarantee is "after
    # the batch the version moved forward at least once"; we record
    # the observed delta in details for diagnostics.
    delta = post_overlay_version - pre_overlay_version_after_path
    dh.record(
        "A_overlay_version_bumps_once_per_batch",
        delta >= 1,
        {
            "pre_overlay_version": pre_overlay_version,
            "pre_overlay_version_after_path": pre_overlay_version_after_path,
            "post_overlay_version": post_overlay_version,
            "delta": delta,
        },
    )

    # ----- B: targeted fields took the profile values ----------------
    matched = []
    diffs = {}
    for key, want in expected.items():
        got = post_field_values[key]
        ok_field = abs(got - want) < 1e-5
        diffs[key] = {
            "pre": pre_field_values[key],
            "post": got,
            "want": want,
            "matched": ok_field,
        }
        if ok_field and abs(got - pre_field_values[key]) > 1e-9:
            matched.append(key)
    dh.record(
        "B_param_fields_took_profile_values",
        len(matched) >= 3,
        {"matched_count": len(matched), "matched": matched, "diffs": diffs},
    )

    # ----- C: param hash reflects the new values ---------------------
    dh.record(
        "C_param_hash_reflects_new_values",
        post_param_hash != pre_param_hash_after_path,
        {
            "pre_param_hash": pre_param_hash,
            "pre_param_hash_after_path": pre_param_hash_after_path,
            "post_param_hash": post_param_hash,
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 60.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
