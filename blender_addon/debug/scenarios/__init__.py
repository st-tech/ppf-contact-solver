# File: scenarios/__init__.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Phase 1 scenario registry. Each scenario module exports a ``run(ctx) -> dict``
# function that returns ``{"status": "pass"|"fail", "violations": [...]}``.
#
# Scenarios are protocol-level: they talk to the debug server via the same
# JSON-over-TCP wire the addon's communicator uses, so production code on
# the server side (transitions, monitor, response generation, atomic upload)
# is exercised end-to-end. Phase 2 will add Blender-driven counterparts
# that exercise the addon UI through the same lifecycle.

import sys

from . import server_smoke
from . import upload_id_changes
from . import bl_connect_local
from . import bl_connect_win_native

# Pin fidelity matrix. Each scenario builds a scene with one pin op
# (or composed ops), runs through the full pipeline, and diffs the
# fetched PC2 against frontend.FixedScene.time(t).
from . import bl_pin_animation_fidelity
from . import bl_pin_spin_centroid
from . import bl_pin_spin_fixed
from . import bl_pin_spin_max_towards
from . import bl_pin_spin_vertex
from . import bl_pin_scale_centroid
from . import bl_pin_scale_fixed
from . import bl_pin_scale_max_towards
from . import bl_pin_torque
from . import bl_pin_compose_move_spin
from . import bl_pin_compose_spin_move
from . import bl_pin_compose_full

# UI / state-machine integration scenarios. These exercise
# overlay invalidation, race-condition surfaces, fetch-clear-refetch,
# geometry hash propagation, and parameter transfer.
from . import bl_overlay_invalidation
from . import bl_race_state_machine
from . import bl_fetch_clear_refetch
from . import bl_geometry_hash
from . import bl_param_change
from . import bl_friction_mode
from . import bl_save_resume
from . import bl_load_disconnect
from . import bl_open_mainfile_disconnect
from . import bl_param_dirty
from . import bl_run_consistency
from . import bl_drape_ready_to_run
from . import bl_shallow_copy
from . import bl_shared_object_data
from . import bl_transition_chains
from . import bl_chain_lifecycle
from . import bl_chain_save_resume
from . import bl_chain_param_repeat
from . import bl_chain_abort_recovery
from . import bl_chain_reconnect
from . import bl_chain_data_evolution
from . import bl_chain_server_restart_after_run
from . import bl_pc2_migration
from . import bl_ngon_rejection
from . import bl_upload_id_desync_recovery
from . import bl_mesh_cache_self_heal
from . import bl_live_frame_end_tracking
from . import bl_fetch_failed_watchdog
from . import bl_async_op_cancelled_redraws
from . import bl_profile_load_batch
from . import bl_pin_rod_curve
from . import bl_static_op_anim
from . import bl_multi_group
from . import bl_collider_keyframes
from . import bl_stitch_merge
from . import bl_velocity_keyframes
from . import bl_bake_animation
from . import bl_mcp_roundtrip
from . import bl_addon_reload_handoff
from . import bl_ftetwild_overrides
from . import bl_project_rename_resync
from . import bl_intersection_records_roundtrip
from . import bl_violation_overlay_classification
from . import bl_self_intersection_build_reject

# Copy/paste roundtrip coverage. Material Params and Pin Operations
# expose COPYDOWN / PASTEDOWN buttons backed by a WindowManager-scoped
# clipboard; the cross-type scenario verifies that the paste filter
# only forwards model-applicable scalars.
from . import bl_copy_paste_material_params
from . import bl_copy_paste_pin_ops
from . import bl_copy_paste_cross_type_material

# Operator-poll regression: the Transfer button used to remain
# clickable for one event-loop tick after Run.execute because its
# poll only checked the protocol-version and the cached server
# response.
from . import bl_transfer_disabled_during_run
from . import bl_transfer_skip_delete_when_no_data


REGISTRY = {
    # Server-only protocol checks. These don't require a build, so
    # they don't need real addon-encoded data.pickle and run against
    # the real frontend without issue.
    "server_smoke": server_smoke,
    "upload_id_changes": upload_id_changes,

    # Blender-driven scenarios. These produce real ``data.pickle`` via
    # the addon's encoder, exercising the full pipeline:
    # addon -> upload -> frontend.populate -> frontend.make ->
    # Rust binary (--features emulated) -> vert_*.bin -> fetch.
    "bl_connect_local": bl_connect_local,
    "bl_connect_win_native": bl_connect_win_native,

    # Pin-op fidelity matrix. Each scenario cross-checks the Rust
    # solver's per-frame pin trajectory against frontend.FixedScene
    # .time(t) -- the same source of truth that frontend.preview()
    # uses in a Jupyter notebook.
    "bl_pin_animation_fidelity": bl_pin_animation_fidelity,  # MOVE_BY
    "bl_pin_spin_centroid": bl_pin_spin_centroid,
    "bl_pin_spin_fixed": bl_pin_spin_fixed,
    "bl_pin_spin_max_towards": bl_pin_spin_max_towards,
    "bl_pin_spin_vertex": bl_pin_spin_vertex,
    "bl_pin_scale_centroid": bl_pin_scale_centroid,
    "bl_pin_scale_fixed": bl_pin_scale_fixed,
    "bl_pin_scale_max_towards": bl_pin_scale_max_towards,
    "bl_pin_torque": bl_pin_torque,
    "bl_pin_compose_move_spin": bl_pin_compose_move_spin,
    "bl_pin_compose_spin_move": bl_pin_compose_spin_move,
    "bl_pin_compose_full": bl_pin_compose_full,

    # UI / state-machine integration
    "bl_overlay_invalidation": bl_overlay_invalidation,
    "bl_race_state_machine": bl_race_state_machine,
    "bl_fetch_clear_refetch": bl_fetch_clear_refetch,
    "bl_geometry_hash": bl_geometry_hash,
    "bl_param_change": bl_param_change,
    "bl_friction_mode": bl_friction_mode,
    "bl_save_resume": bl_save_resume,
    "bl_load_disconnect": bl_load_disconnect,
    "bl_open_mainfile_disconnect": bl_open_mainfile_disconnect,
    "bl_param_dirty": bl_param_dirty,
    "bl_run_consistency": bl_run_consistency,
    "bl_drape_ready_to_run": bl_drape_ready_to_run,
    "bl_shallow_copy": bl_shallow_copy,
    "bl_shared_object_data": bl_shared_object_data,
    "bl_transition_chains": bl_transition_chains,
    "bl_chain_lifecycle": bl_chain_lifecycle,
    "bl_chain_save_resume": bl_chain_save_resume,
    "bl_chain_param_repeat": bl_chain_param_repeat,
    "bl_chain_abort_recovery": bl_chain_abort_recovery,
    "bl_chain_reconnect": bl_chain_reconnect,
    "bl_chain_data_evolution": bl_chain_data_evolution,
    "bl_chain_server_restart_after_run": bl_chain_server_restart_after_run,
    "bl_pc2_migration": bl_pc2_migration,
    "bl_ngon_rejection": bl_ngon_rejection,

    # Tier 1: bug-fix-driven coverage (commits ea4303cb, 92546e18, a8766a08,
    # ff0d20ca, ...).
    "bl_upload_id_desync_recovery": bl_upload_id_desync_recovery,
    "bl_mesh_cache_self_heal": bl_mesh_cache_self_heal,
    "bl_live_frame_end_tracking": bl_live_frame_end_tracking,
    "bl_fetch_failed_watchdog": bl_fetch_failed_watchdog,
    "bl_async_op_cancelled_redraws": bl_async_op_cancelled_redraws,
    "bl_profile_load_batch": bl_profile_load_batch,

    # Tier 2: feature-coverage gaps. Each scenario authors a specific
    # primitive (rod curve, static op, multi-group, collider keyframe,
    # stitch, velocity keyframe, bake) and verifies the encoded /
    # simulated round-trip end-to-end.
    "bl_pin_rod_curve": bl_pin_rod_curve,
    "bl_static_op_anim": bl_static_op_anim,
    "bl_multi_group": bl_multi_group,
    "bl_collider_keyframes": bl_collider_keyframes,
    "bl_stitch_merge": bl_stitch_merge,
    "bl_velocity_keyframes": bl_velocity_keyframes,
    "bl_bake_animation": bl_bake_animation,

    # Tier 3: nice-to-have coverage that needed extra rig plumbing
    # (MCP HTTP, addon reload handoff, fTetWild overrides, project
    # rename resync).
    "bl_mcp_roundtrip": bl_mcp_roundtrip,
    "bl_addon_reload_handoff": bl_addon_reload_handoff,
    "bl_ftetwild_overrides": bl_ftetwild_overrides,
    "bl_project_rename_resync": bl_project_rename_resync,

    # Tier 1.5: solver intersection feedback round-trip. Both rely on
    # the PPF_EMULATED_FAIL_AT_FRAME (Rust) and PPF_EMULATED_VIOLATIONS
    # (Python emulator) knobs; the second can run client-side only by
    # injecting a synthetic ServerPolled.
    "bl_intersection_records_roundtrip": bl_intersection_records_roundtrip,
    "bl_violation_overlay_classification": bl_violation_overlay_classification,
    "bl_self_intersection_build_reject": bl_self_intersection_build_reject,

    # Copy/paste clipboards (Material Params, Pin Operations).
    "bl_copy_paste_material_params": bl_copy_paste_material_params,
    "bl_copy_paste_pin_ops": bl_copy_paste_pin_ops,
    "bl_copy_paste_cross_type_material": bl_copy_paste_cross_type_material,

    # Operator-poll regression for Transfer-during-Run.
    "bl_transfer_disabled_during_run": bl_transfer_disabled_during_run,
    "bl_transfer_skip_delete_when_no_data": bl_transfer_skip_delete_when_no_data,
}


def _platform_supported(mod) -> bool:
    """True if *mod* declares no PLATFORMS attribute, or the current
    ``sys.platform`` matches one of its declared prefixes. Lets a
    scenario opt out of OSes where its connect path doesn't apply
    (e.g. bl_connect_local on macOS/Windows, bl_connect_win_native on
    Linux/macOS)."""
    plats = getattr(mod, "PLATFORMS", None)
    if plats is None:
        return True
    return any(sys.platform.startswith(p) for p in plats)


def server_only_names() -> list[str]:
    """Names of scenarios that don't require Blender. Useful for CI
    runs on hosts without a Blender install."""
    return [
        n for n, mod in REGISTRY.items()
        if not getattr(mod, "NEEDS_BLENDER", False)
        and _platform_supported(mod)
    ]


def get(name: str):
    """Return the scenario module, or None if unknown."""
    return REGISTRY.get(name)


def all_names() -> list[str]:
    return [n for n, m in REGISTRY.items() if _platform_supported(m)]
