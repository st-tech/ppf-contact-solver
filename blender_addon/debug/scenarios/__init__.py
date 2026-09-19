# File: scenarios/__init__.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Scenario registry. Each scenario module exports a ``run(ctx) -> dict``
# function that returns ``{"status": "pass"|"fail", "violations": [...]}``.
#
# Protocol-level scenarios talk to the debug server via the same
# JSON-over-TCP wire the addon's communicator uses, so production code on
# the server side (transitions, monitor, response generation, atomic upload)
# is exercised end-to-end. The ``bl_*`` scenarios drive Blender itself and
# exercise the addon UI through the same lifecycle.

import os
import sys


REPO_ROOT_POSIX: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
).replace("\\", "/")
"""Repo root with forward-slash separators. Driver string-substitution
on Windows would otherwise emit backslash escapes."""

from . import rig_lock_axes
from . import rig_lock_axes_projector
from . import rig_intersection_allowances
from . import rig_intersection_allowance_isolation
from . import bl_server_stop_is_real
from . import bl_force_terminate_port
from . import rig_collider_coincident_pair
from . import rig_degenerate_rest_shape
from . import bl_fetch_frame_discovery
from . import rig_degenerate_tet_rest_shape
from . import rig_coincident_contact_pair
from . import rig_backend_cleared_midflight
from . import rig_device_diagnostic_channel
from . import rig_launch_config
from . import rig_solver_log_format
from . import rig_remote_kill_port_scope
from . import rig_session_artifact_identity
from . import server_smoke
from . import upload_id_changes
from . import bl_connect_linux_native
from . import bl_connect_win_native
from . import bl_connection_path_validation
from . import bl_connection_path_relative
from . import bl_connection_failure_reporting
from . import bl_remote_device_select
from . import bl_cbor2_missing_reported
from . import bl_docker_connect_gate
from . import bl_ssh_proxy_jump
from . import bl_win_native_root_resolve
from . import bl_win_native_bundle_layout
from . import bl_mac_native_root_resolve
from . import bl_mac_native_real_solve
from . import bl_native_attach_build_check
from . import bl_retired_connection_migration
from . import bl_native_device_real_solve
from . import bl_solver_gpu_select
from . import bl_direct_disk_transfer
from . import bl_rust_binary_protocol

# Pin fidelity matrix. Each scenario builds a scene with one pin op
# (or composed ops), runs through the full pipeline, and diffs the
# fetched PC2 against frontend.FixedScene.time(t).
from . import bl_pin_animation_fidelity
from . import bl_driven_pin_frame_exact
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
from . import bl_pin_op_type_enum_stable
from . import bl_pin_vgroup_enum_ref
from . import bl_enum_props_guard
from . import bl_i18n_no_leaks
from . import bl_solver_crash_kind_localized
from . import bl_solver_crash_open_session_folder
from . import bl_pin_capture_deformation
from . import bl_pin_capture_deformation_persistence
from . import bl_pinned_anim_transform_display
from . import bl_recapture_all_deformations
from . import bl_geonode_deform_input
from . import bl_geonode_capture_range_frame_count
from . import bl_static_smooth_by_angle_no_capture
from . import bl_autosave_modal_residency
from . import bl_cache_placement_heal
from . import bl_static_keyframe_capture_hint
from . import bl_bake_aborts_unfetched
from . import bl_pin_make_keyframe_writes_fcurves
from . import bl_pin_overlay_follows_edit_mode
from . import bl_pin_create_clears_stale_membership

# UI / state-machine integration scenarios. These exercise
# overlay invalidation, race-condition surfaces, fetch-clear-refetch,
# geometry hash propagation, and parameter transfer.
from . import bl_overlay_invalidation
from . import bl_overlay_respects_shading
from . import bl_overlay_respects_user_obj_color
from . import bl_cleanup_respects_user_obj_color
from . import bl_race_state_machine
from . import bl_fetch_clear_refetch
from . import bl_geometry_hash
from . import bl_param_change
from . import bl_friction_mode
from . import bl_young_mod_density_normalize
from . import bl_save_resume
from . import bl_resume_from_frame
from . import bl_save_state_on_finish
from . import bl_load_disconnect
from . import bl_open_mainfile_disconnect
from . import bl_param_dirty
from . import bl_run_consistency
from . import bl_drape_ready_to_run
from . import bl_elastic_drape
from . import bl_sand_roundtrip
from . import bl_angular_spin
from . import bl_world_spin
from . import bl_timeline_statistics
from . import bl_stale_statistics_manifest
from . import bl_bend_reference_shell
from . import bl_bend_reference_rod
from . import bl_bend_reference_rod_curve
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
from . import bl_ngon_triangulation
from . import bl_duplicate_face_rejection
from . import bl_hanging_stitch_vertex_rejection
from . import bl_isolated_vertex_rejection
from . import bl_degenerate_tessellation_rejection
from . import bl_degenerate_tessellation_repair
from . import bl_mesh_cleaning
from . import bl_upload_id_desync_recovery
from . import bl_mesh_cache_self_heal
from . import bl_live_frame_end_tracking
from . import bl_fetch_failed_watchdog
from . import bl_server_unknown_recovery
from . import bl_launch_error_unsticks
from . import bl_build_worker_faulthandler
from . import bl_profile_load_batch
from . import bl_pin_rod_curve
from . import bl_static_deform_anim
from . import bl_static_deform_first_frame
from . import bl_static_fcurve_anim
from . import bl_static_op_anim
from . import bl_static_panel_draws
from . import bl_fps_source
from . import bl_time_scale_encoding
from . import bl_time_scale_kinematic_invariance
from . import bl_frame_start
from . import bl_frame_start_leadin
from . import bl_group_slot_reuse
from . import bl_multi_group
from . import bl_collider_keyframes
from . import bl_stitch_merge
from . import bl_merge_pair_no_stitch_rejection
from . import bl_post_snap_toggle
from . import bl_solid_solid_stitch
from . import bl_static_stitch
from . import bl_shell_static_stitch
from . import bl_static_snap_guard
from . import bl_static_soft_constraint
from . import bl_deformed_target_snap
from . import bl_snap_parented_move
from . import bl_transform_translation_roundtrip
from . import bl_velocity_keyframes

# world_scaling coordinate round-trip suite. The Rust solver scales all
# input geometry by state.world_scaling on ingest and divides per-frame
# output back by it, so authored-scale geometry / motion must survive the
# round-trip. Kinematic rigs diff against the scale-agnostic frontend
# reference; scale-invariance rigs run the same scene at two sizes and
# assert the 10x relationship; an encoder rig checks relative-vs-absolute
# gap scaling; a resume rig checks no double-scaling across a checkpoint.
from . import bl_world_scaling_move_by
from . import bl_world_scaling_move_by_shrink
from . import bl_world_scaling_spin
from . import bl_world_scaling_spin_absolute
from . import bl_world_scaling_scale_op
from . import bl_world_scaling_static_pin
from . import bl_world_scaling_shell_drape
from . import bl_world_scaling_velocity
from . import bl_world_scaling_velocity_schedule
from . import bl_world_scaling_solid_tet
from . import bl_world_scaling_rod
from . import bl_world_scaling_sand
from . import bl_world_scaling_multi_group
from . import bl_world_scaling_colliders
from . import bl_world_scaling_encoder_scales
from . import bl_world_scaling_resume
from . import bl_world_scaling_pdrd

from . import bl_pdrd_hinge
from . import bl_lock_translation
from . import bl_lock_rotation
from . import bl_lock_rotation_solve
from . import bl_lock_rotation_prohibit
from . import bl_lock_translation_free
from . import bl_lock_translation_pinned
from . import bl_pdrd_anchor_release
from . import bl_pdrd_driven_translate
from . import bl_pdrd_driven_rotate_vertex
from . import bl_pdrd_panel_draws
from . import bl_bake_animation
from . import bl_export_cache
from . import bl_mcp_mesh_cleaning
from . import bl_mcp_roundtrip
from . import bl_mcp_doc_coverage
from . import bl_mcp_deformation_and_bake
from . import bl_mcp_object_settings
from . import bl_mcp_collision_windows
from . import bl_mcp_connection_refusals
from . import bl_mcp_console_and_diagnostics
from . import bl_mcp_curve_authoring
from . import bl_mcp_dynamic_parameters
from . import bl_mcp_group_lifecycle
from . import bl_mcp_invisible_colliders
from . import bl_mcp_merge_and_snap
from . import bl_mcp_presets_and_profiles
from . import bl_mcp_ordered_collections
from . import bl_mcp_geometry_repair
from . import bl_mcp_group_material_readback
from . import bl_mcp_pin_keyframes
from . import bl_mcp_bend_reference
from . import bl_mcp_modal_jobs
from . import bl_mcp_connection_surface
from . import bl_mcp_error_semantics
from . import bl_mcp_legacy_era
from . import bl_mcp_material_maps
from . import bl_mcp_object_locks
from . import bl_mcp_prompts
from . import bl_mcp_resources
from . import bl_mcp_scene_inspection
from . import bl_mcp_scene_parameters
from . import bl_mcp_statistics
from . import bl_mcp_streaming_tool_call
from . import bl_mcp_tool_schema_invariants
from . import bl_mcp_ui_element_status
from . import bl_mcp_vertex_groups
from . import bl_mcp_transport_conformance
from . import bl_addon_reload_handoff
from . import bl_ftetwild_overrides
from . import bl_project_rename_resync
from . import bl_violation_overlay_classification
from . import bl_intersection_allowances
from . import bl_self_intersection_build_reject
from . import bl_solid_zero_volume_reject
from . import bl_solid_fix_weight_threshold
from . import bl_tetgen_solid_build
from . import bl_real_solid_smoke
from . import bl_ssh_remote_solve
from . import bl_ssh_remote_solid
from . import bl_real_shell_drape
from . import bl_real_frame_start_drape
from . import bl_solid_overlap_pin_last_wins
from . import bl_solid_spin_flip_per_pin
from . import bl_pin_reorder_and_gating

# Copy/paste roundtrip coverage. Material Params and Pin Operations
# expose COPYDOWN / PASTEDOWN buttons backed by a WindowManager-scoped
# clipboard; the cross-type scenario verifies that the paste filter
# only forwards model-applicable scalars.
from . import bl_copy_paste_material_params
from . import bl_copy_paste_pin_ops
from . import bl_copy_paste_cross_type_material
from . import bl_material_keyframe_animates
from . import bl_material_lock_guards
from . import bl_material_map_animates
from . import bl_material_map_every_key
from . import bl_material_map_refusals
from . import bl_material_map_panel_draws
from . import bl_material_map_sample_ops
from . import bl_solid_spatial_material_map
from . import bl_spatial_material_map
from . import bl_material_preset_apply

# Operator-poll regression: the Transfer button must be disabled in the
# same event-loop tick as Run.execute. A poll that consults only the
# protocol version and the cached server response leaves it clickable
# for that tick, because the cached response still reads READY.
from . import bl_transfer_disabled_during_run
from . import bl_transfer_skip_delete_when_no_data

# UX progress bars: live solver progress during simulation, and the
# fetch-animation download/apply progress sequence.
from . import bl_progress_simulating
from . import bl_progress_fetching

# Realtime Statistics: the live ``summary`` dict the addon panel
# renders inside the "Realtime Statistics" box during a sim.
from . import bl_realtime_stats_shown

# Abort resolution: a pending abort keeps polling until the solver is
# confirmed terminal, so "Aborting..." never sticks (com.busy() clears).
from . import bl_abort_resolves

# Clear-Local-Animation enable signal: stateless, fast object-modifier scan
# (scene_has_solver_cache) that reflects cache presence on every redraw.
from . import bl_clear_anim_poll


REGISTRY = {
    # Server-only protocol checks. These don't require a build, so
    # they don't need real addon-encoded data.pickle and run against
    # the real frontend without issue.
    "server_smoke": server_smoke,
    "upload_id_changes": upload_id_changes,

    # Every Lock Translation / Lock Rotation mode at its two gates: the bytes
    # that reach the session directory, and the projector that reads them. The
    # The `bl_lock_*` scenarios cover the addon encoder and the same physics
    # driven through Blender; neither of these needs it.
    "rig_lock_axes": rig_lock_axes,
    "rig_lock_axes_projector": rig_lock_axes_projector,

    # The issue-#138 intersection allowances, at their three gates: the
    # solver's live scan, the scene-build check, and whether an allowance
    # stays inside the pairs that asked for it. None needs Blender.
    "rig_intersection_allowances": rig_intersection_allowances,
    "rig_intersection_allowance_isolation": rig_intersection_allowance_isolation,

    # The solver's build-time rest-shape gate: a near-collinear shell face is
    # finite and invertible, so only its conditioning gives it away (issue
    # #144). `bl_degenerate_tessellation_rejection` covers the addon-side gate
    # that refuses the same geometry a step earlier; both grant the same set.
    "bl_server_stop_is_real": bl_server_stop_is_real,
    # Force Terminate Process: ends the local server from the refused, disconnected
    # state, which is the one state Stop Server cannot reach.
    "bl_force_terminate_port": bl_force_terminate_port,
    "rig_collider_coincident_pair": rig_collider_coincident_pair,
    "rig_degenerate_rest_shape": rig_degenerate_rest_shape,
    "bl_fetch_frame_discovery": bl_fetch_frame_discovery,
    "rig_degenerate_tet_rest_shape": rig_degenerate_tet_rest_shape,
    "rig_coincident_contact_pair": rig_coincident_contact_pair,

    # A disconnect landing mid-operation is a transport failure, never an
    # AttributeError: disconnect runs on the main thread and the worker does
    # not serialize against it.
    "rig_backend_cleared_midflight": rig_backend_cleared_midflight,

    # A failed DEVICE check reaches the host on this machine's build, which
    # only firing one can settle: a healthy run never touches the channel.
    "rig_device_diagnostic_channel": rig_device_diagnostic_channel,

    # How the rig LAUNCHES Blender (window size, PPF_BLENDER_WINDOW
    # parsing, display probing). Server-only so it does not need the
    # Blender it configures.
    "rig_launch_config": rig_launch_config,

    # The identity the runner's cached session artifacts carry, driven
    # through the real fetch entry points against a fake backend. Needs
    # neither Blender nor a solver, so it runs wherever the rig does.
    "rig_solver_log_format": rig_solver_log_format,
    "rig_session_artifact_identity": rig_session_artifact_identity,

    # The remote kill ends the server on the port it was given and no
    # other, checked against stand-in processes. Needs neither Blender nor
    # a solver; the POSIX half is skipped on Windows, where there is no
    # /bin/sh to build a stand-in from.
    "rig_remote_kill_port_scope": rig_remote_kill_port_scope,

    # Blender-driven scenarios. These produce real ``data.pickle`` via
    # the addon's encoder, exercising the full pipeline:
    # addon -> upload -> frontend.populate -> frontend.make ->
    # solver binary -> vert_*.bin -> fetch.
    "bl_connect_linux_native": bl_connect_linux_native,
    "bl_connect_win_native": bl_connect_win_native,
    "bl_connection_path_validation": bl_connection_path_validation,
    "bl_connection_path_relative": bl_connection_path_relative,
    "bl_remote_device_select": bl_remote_device_select,
    "bl_connection_failure_reporting": bl_connection_failure_reporting,
    "bl_cbor2_missing_reported": bl_cbor2_missing_reported,
    "bl_docker_connect_gate": bl_docker_connect_gate,
    "bl_ssh_proxy_jump": bl_ssh_proxy_jump,
    "bl_win_native_root_resolve": bl_win_native_root_resolve,
    "bl_win_native_bundle_layout": bl_win_native_bundle_layout,
    "bl_mac_native_root_resolve": bl_mac_native_root_resolve,
    "bl_mac_native_real_solve": bl_mac_native_real_solve,
    "bl_native_attach_build_check": bl_native_attach_build_check,
    "bl_retired_connection_migration": bl_retired_connection_migration,
    "bl_native_device_real_solve": bl_native_device_real_solve,
    "bl_solver_gpu_select": bl_solver_gpu_select,
    "bl_direct_disk_transfer": bl_direct_disk_transfer,
    "bl_rust_binary_protocol": bl_rust_binary_protocol,

    # Pin-op fidelity matrix. Each scenario cross-checks the Rust
    # solver's per-frame pin trajectory against frontend.FixedScene
    # .time(t) -- the same source of truth that frontend.preview()
    # uses in a Jupyter notebook.
    "bl_pin_animation_fidelity": bl_pin_animation_fidelity,  # MOVE_BY
    # Frame-writer exact-pose gate: output frames fall between substeps and a
    # driven fix pin must still land dead on its prescribed path (the
    # driven-collider jitter regression).
    "bl_driven_pin_frame_exact": bl_driven_pin_frame_exact,
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
    "bl_pin_op_type_enum_stable": bl_pin_op_type_enum_stable,
    "bl_pin_vgroup_enum_ref": bl_pin_vgroup_enum_ref,
    "bl_enum_props_guard": bl_enum_props_guard,
    "bl_i18n_no_leaks": bl_i18n_no_leaks,
    "bl_solver_crash_kind_localized": bl_solver_crash_kind_localized,
    "bl_solver_crash_open_session_folder": bl_solver_crash_open_session_folder,
    "bl_pin_capture_deformation": bl_pin_capture_deformation,
    "bl_pin_capture_deformation_persistence": bl_pin_capture_deformation_persistence,
    "bl_pinned_anim_transform_display": bl_pinned_anim_transform_display,
    "bl_recapture_all_deformations": bl_recapture_all_deformations,
    "bl_geonode_deform_input": bl_geonode_deform_input,
    "bl_geonode_capture_range_frame_count": bl_geonode_capture_range_frame_count,
    "bl_static_smooth_by_angle_no_capture": bl_static_smooth_by_angle_no_capture,
    "bl_autosave_modal_residency": bl_autosave_modal_residency,
    "bl_cache_placement_heal": bl_cache_placement_heal,
    "bl_static_keyframe_capture_hint": bl_static_keyframe_capture_hint,
    "bl_bake_aborts_unfetched": bl_bake_aborts_unfetched,
    "bl_pin_make_keyframe_writes_fcurves": bl_pin_make_keyframe_writes_fcurves,
    "bl_pin_overlay_follows_edit_mode": bl_pin_overlay_follows_edit_mode,
    "bl_pin_create_clears_stale_membership": bl_pin_create_clears_stale_membership,

    # UI / state-machine integration
    "bl_overlay_invalidation": bl_overlay_invalidation,
    "bl_overlay_respects_shading": bl_overlay_respects_shading,
    "bl_overlay_respects_user_obj_color": bl_overlay_respects_user_obj_color,
    "bl_cleanup_respects_user_obj_color": bl_cleanup_respects_user_obj_color,
    "bl_race_state_machine": bl_race_state_machine,
    "bl_fetch_clear_refetch": bl_fetch_clear_refetch,
    "bl_geometry_hash": bl_geometry_hash,
    "bl_param_change": bl_param_change,
    "bl_friction_mode": bl_friction_mode,
    "bl_young_mod_density_normalize": bl_young_mod_density_normalize,
    "bl_save_resume": bl_save_resume,
    "bl_resume_from_frame": bl_resume_from_frame,
    "bl_save_state_on_finish": bl_save_state_on_finish,
    "bl_load_disconnect": bl_load_disconnect,
    "bl_open_mainfile_disconnect": bl_open_mainfile_disconnect,
    "bl_param_dirty": bl_param_dirty,
    "bl_run_consistency": bl_run_consistency,
    "bl_drape_ready_to_run": bl_drape_ready_to_run,
    "bl_elastic_drape": bl_elastic_drape,
    "bl_sand_roundtrip": bl_sand_roundtrip,
    "bl_angular_spin": bl_angular_spin,
    "bl_world_spin": bl_world_spin,
    "bl_timeline_statistics": bl_timeline_statistics,
    "bl_stale_statistics_manifest": bl_stale_statistics_manifest,
    "bl_bend_reference_shell": bl_bend_reference_shell,
    "bl_bend_reference_rod": bl_bend_reference_rod,
    "bl_bend_reference_rod_curve": bl_bend_reference_rod_curve,
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
    "bl_ngon_triangulation": bl_ngon_triangulation,
    "bl_duplicate_face_rejection": bl_duplicate_face_rejection,
    "bl_hanging_stitch_vertex_rejection": bl_hanging_stitch_vertex_rejection,
    "bl_isolated_vertex_rejection": bl_isolated_vertex_rejection,
    "bl_degenerate_tessellation_rejection":
        bl_degenerate_tessellation_rejection,
    "bl_degenerate_tessellation_repair":
        bl_degenerate_tessellation_repair,
    "bl_mesh_cleaning": bl_mesh_cleaning,

    # Tier 1: bug-fix-driven coverage (commits ea4303cb, 92546e18, a8766a08,
    # ff0d20ca, ...).
    "bl_upload_id_desync_recovery": bl_upload_id_desync_recovery,
    "bl_mesh_cache_self_heal": bl_mesh_cache_self_heal,
    "bl_live_frame_end_tracking": bl_live_frame_end_tracking,
    "bl_fetch_failed_watchdog": bl_fetch_failed_watchdog,
    "bl_server_unknown_recovery": bl_server_unknown_recovery,
    "bl_launch_error_unsticks": bl_launch_error_unsticks,
    "bl_build_worker_faulthandler": bl_build_worker_faulthandler,
    "bl_profile_load_batch": bl_profile_load_batch,

    # Tier 2: feature-coverage gaps. Each scenario authors a specific
    # primitive (rod curve, static op, multi-group, collider keyframe,
    # stitch, velocity keyframe, bake) and verifies the encoded /
    # simulated round-trip end-to-end.
    "bl_pin_rod_curve": bl_pin_rod_curve,
    "bl_static_deform_anim": bl_static_deform_anim,
    "bl_static_deform_first_frame": bl_static_deform_first_frame,
    "bl_static_fcurve_anim": bl_static_fcurve_anim,
    "bl_static_op_anim": bl_static_op_anim,
    "bl_static_panel_draws": bl_static_panel_draws,
    "bl_fps_source": bl_fps_source,
    "bl_time_scale_encoding": bl_time_scale_encoding,
    "bl_time_scale_kinematic_invariance": bl_time_scale_kinematic_invariance,
    "bl_frame_start": bl_frame_start,
    "bl_frame_start_leadin": bl_frame_start_leadin,
    "bl_group_slot_reuse": bl_group_slot_reuse,
    "bl_multi_group": bl_multi_group,
    "bl_collider_keyframes": bl_collider_keyframes,
    "bl_stitch_merge": bl_stitch_merge,
    "bl_merge_pair_no_stitch_rejection": bl_merge_pair_no_stitch_rejection,
    "bl_post_snap_toggle": bl_post_snap_toggle,
    "bl_solid_solid_stitch": bl_solid_solid_stitch,
    "bl_static_stitch": bl_static_stitch,
    "bl_shell_static_stitch": bl_shell_static_stitch,
    "bl_static_snap_guard": bl_static_snap_guard,
    "bl_static_soft_constraint": bl_static_soft_constraint,
    "bl_deformed_target_snap": bl_deformed_target_snap,
    "bl_snap_parented_move": bl_snap_parented_move,
    "bl_transform_translation_roundtrip": bl_transform_translation_roundtrip,
    "bl_velocity_keyframes": bl_velocity_keyframes,

    # world_scaling coordinate round-trip suite.
    "bl_world_scaling_move_by": bl_world_scaling_move_by,
    "bl_world_scaling_move_by_shrink": bl_world_scaling_move_by_shrink,
    "bl_world_scaling_spin": bl_world_scaling_spin,
    "bl_world_scaling_spin_absolute": bl_world_scaling_spin_absolute,
    "bl_world_scaling_scale_op": bl_world_scaling_scale_op,
    "bl_world_scaling_static_pin": bl_world_scaling_static_pin,
    "bl_world_scaling_shell_drape": bl_world_scaling_shell_drape,
    "bl_world_scaling_velocity": bl_world_scaling_velocity,
    "bl_world_scaling_velocity_schedule": bl_world_scaling_velocity_schedule,
    "bl_world_scaling_solid_tet": bl_world_scaling_solid_tet,
    "bl_world_scaling_rod": bl_world_scaling_rod,
    "bl_world_scaling_sand": bl_world_scaling_sand,
    "bl_world_scaling_multi_group": bl_world_scaling_multi_group,
    "bl_world_scaling_colliders": bl_world_scaling_colliders,
    "bl_world_scaling_encoder_scales": bl_world_scaling_encoder_scales,
    "bl_world_scaling_resume": bl_world_scaling_resume,
    "bl_world_scaling_pdrd": bl_world_scaling_pdrd,

    "bl_pdrd_hinge": bl_pdrd_hinge,
    "bl_lock_translation": bl_lock_translation,
    "bl_lock_rotation": bl_lock_rotation,
    "bl_lock_rotation_solve": bl_lock_rotation_solve,
    "bl_lock_rotation_prohibit": bl_lock_rotation_prohibit,
    "bl_lock_translation_free": bl_lock_translation_free,
    "bl_lock_translation_pinned": bl_lock_translation_pinned,
    "bl_pdrd_anchor_release": bl_pdrd_anchor_release,
    "bl_pdrd_driven_translate": bl_pdrd_driven_translate,
    "bl_pdrd_driven_rotate_vertex": bl_pdrd_driven_rotate_vertex,
    "bl_pdrd_panel_draws": bl_pdrd_panel_draws,
    "bl_bake_animation": bl_bake_animation,
    "bl_export_cache": bl_export_cache,

    # Tier 3: nice-to-have coverage that needed extra rig plumbing
    # (MCP HTTP, addon reload handoff, fTetWild overrides, project
    # rename resync).
    "bl_mcp_mesh_cleaning": bl_mcp_mesh_cleaning,
    "bl_mcp_roundtrip": bl_mcp_roundtrip,
    "bl_mcp_transport_conformance": bl_mcp_transport_conformance,
    "bl_mcp_doc_coverage": bl_mcp_doc_coverage,
    "bl_mcp_deformation_and_bake": bl_mcp_deformation_and_bake,
    "bl_mcp_object_settings": bl_mcp_object_settings,
    "bl_mcp_collision_windows": bl_mcp_collision_windows,
    "bl_mcp_connection_refusals": bl_mcp_connection_refusals,
    "bl_mcp_console_and_diagnostics": bl_mcp_console_and_diagnostics,
    "bl_mcp_curve_authoring": bl_mcp_curve_authoring,
    "bl_mcp_dynamic_parameters": bl_mcp_dynamic_parameters,
    "bl_mcp_group_lifecycle": bl_mcp_group_lifecycle,
    "bl_mcp_invisible_colliders": bl_mcp_invisible_colliders,
    "bl_mcp_merge_and_snap": bl_mcp_merge_and_snap,
    "bl_mcp_presets_and_profiles": bl_mcp_presets_and_profiles,
    "bl_mcp_ordered_collections": bl_mcp_ordered_collections,
    "bl_mcp_geometry_repair": bl_mcp_geometry_repair,
    "bl_mcp_group_material_readback": bl_mcp_group_material_readback,
    "bl_mcp_pin_keyframes": bl_mcp_pin_keyframes,
    "bl_mcp_bend_reference": bl_mcp_bend_reference,
    "bl_mcp_modal_jobs": bl_mcp_modal_jobs,
    "bl_mcp_connection_surface": bl_mcp_connection_surface,
    "bl_mcp_error_semantics": bl_mcp_error_semantics,
    "bl_mcp_legacy_era": bl_mcp_legacy_era,
    "bl_mcp_material_maps": bl_mcp_material_maps,
    "bl_mcp_object_locks": bl_mcp_object_locks,
    "bl_mcp_prompts": bl_mcp_prompts,
    "bl_mcp_resources": bl_mcp_resources,
    "bl_mcp_scene_inspection": bl_mcp_scene_inspection,
    "bl_mcp_scene_parameters": bl_mcp_scene_parameters,
    "bl_mcp_statistics": bl_mcp_statistics,
    "bl_mcp_streaming_tool_call": bl_mcp_streaming_tool_call,
    "bl_mcp_tool_schema_invariants": bl_mcp_tool_schema_invariants,
    "bl_mcp_ui_element_status": bl_mcp_ui_element_status,
    "bl_mcp_vertex_groups": bl_mcp_vertex_groups,
    "bl_addon_reload_handoff": bl_addon_reload_handoff,
    "bl_ftetwild_overrides": bl_ftetwild_overrides,
    "bl_project_rename_resync": bl_project_rename_resync,

    # Tier 1.5: solver intersection feedback, client side. This one runs by
    # injecting a synthetic ServerPolled, so it needs no solver-side fault
    # injection to reach the overlay classification it asserts.
    "bl_violation_overlay_classification": bl_violation_overlay_classification,

    # The ADDON half of the intersection allowances: that the two group
    # checkboxes and the per-pin one reach the built session at all. The
    # rig_intersection_allowances scenario covers what they mean once they
    # get there, and it does not load Blender.
    # It sits after the two entries above because it belongs with the
    # intersection cluster, and below their comment rather than inside it
    # because it drives an ordinary build and reads the session directory:
    # it sets neither of those knobs and injects no ServerPolled.
    "bl_intersection_allowances": bl_intersection_allowances,

    "bl_self_intersection_build_reject": bl_self_intersection_build_reject,
    "bl_solid_zero_volume_reject": bl_solid_zero_volume_reject,
    "bl_solid_fix_weight_threshold": bl_solid_fix_weight_threshold,
    "bl_tetgen_solid_build": bl_tetgen_solid_build,
    "bl_real_solid_smoke": bl_real_solid_smoke,
    "bl_ssh_remote_solve": bl_ssh_remote_solve,
    "bl_ssh_remote_solid": bl_ssh_remote_solid,
    "bl_real_shell_drape": bl_real_shell_drape,
    "bl_real_frame_start_drape": bl_real_frame_start_drape,
    "bl_solid_overlap_pin_last_wins": bl_solid_overlap_pin_last_wins,
    "bl_solid_spin_flip_per_pin": bl_solid_spin_flip_per_pin,
    "bl_pin_reorder_and_gating": bl_pin_reorder_and_gating,

    # Copy/paste clipboards (Material Params, Pin Operations).
    "bl_copy_paste_material_params": bl_copy_paste_material_params,
    "bl_copy_paste_pin_ops": bl_copy_paste_pin_ops,
    "bl_copy_paste_cross_type_material": bl_copy_paste_cross_type_material,
    "bl_material_keyframe_animates": bl_material_keyframe_animates,
    "bl_material_lock_guards": bl_material_lock_guards,
    "bl_material_map_animates": bl_material_map_animates,
    "bl_material_map_every_key": bl_material_map_every_key,
    "bl_material_map_refusals": bl_material_map_refusals,
    "bl_material_map_panel_draws": bl_material_map_panel_draws,
    "bl_material_map_sample_ops": bl_material_map_sample_ops,
    "bl_solid_spatial_material_map": bl_solid_spatial_material_map,
    "bl_spatial_material_map": bl_spatial_material_map,
    "bl_material_preset_apply": bl_material_preset_apply,

    # Operator-poll regression for Transfer-during-Run.
    "bl_transfer_disabled_during_run": bl_transfer_disabled_during_run,
    "bl_transfer_skip_delete_when_no_data": bl_transfer_skip_delete_when_no_data,

    # UX progress bars.
    "bl_progress_simulating": bl_progress_simulating,
    "bl_progress_fetching": bl_progress_fetching,

    # Realtime Statistics box.
    "bl_realtime_stats_shown": bl_realtime_stats_shown,

    # Abort-state resolution.
    "bl_abort_resolves": bl_abort_resolves,

    # Clear-Local-Animation stateless enable signal.
    "bl_clear_anim_poll": bl_clear_anim_poll,
}


def _platform_supported(mod) -> bool:
    """True if *mod* declares no PLATFORMS attribute, or the current
    ``sys.platform`` matches one of its declared prefixes. Lets a
    scenario opt out of OSes where its connect path doesn't apply
    (e.g. bl_connect_linux_native on macOS/Windows, bl_connect_win_native on
    Linux/macOS)."""
    plats = getattr(mod, "PLATFORMS", None)
    if plats is None:
        return True
    return any(sys.platform.startswith(p) for p in plats)


# THERE IS NO DEFAULT BACKEND, and an undeclared scenario is an ERROR.
#
# A default would let a scenario be selected for a backend nobody had
# established it against, and the inherited claim would be invisible in the
# scenario's own source. Every scenario names in its ``BACKENDS`` tuple the
# backend it was RUN against, and one that names none is refused at import
# rather than quietly acquiring a claim its author never made.
#
# Nor is there a stand-in backend to default to. Every backend this rig can
# select computes real physics; none of them is a no-op that reports a pass
# without simulating.


# Backends this rig can target. `real` is a backend that computes real physics,
# which is CUDA, Metal or the Rust CPU backend depending on what the tree was
# built for; `runtests` reports which one a run actually used.
SELECTABLE_BACKENDS = ("real",)


class BackendUnavailable(RuntimeError):
    """Raised when a run asks for a backend the rig cannot target.

    Carries the reason in its message so the caller can print it verbatim
    instead of restating it at each call site."""


def resolve_backend(backend: str) -> str:
    """Return *backend* if the rig can target it, else raise.

    This is the single gate on backend NAMES. It exists because argparse
    ``choices=`` answers an unknown backend with "invalid choice", which
    reads as a typo rather than as a name this rig cannot target."""
    if backend not in SELECTABLE_BACKENDS:
        raise BackendUnavailable(
            f"backend {backend!r} is not a backend this rig knows. "
            f"Selectable: {', '.join(SELECTABLE_BACKENDS)}"
        )
    return backend


def backend_unsupported_reason(mod, backend: str) -> str | None:
    """``None`` if *mod* can run against *backend*, else why it cannot.

    A reason, not a bool: a scenario dropped from a selection has to be
    reportable by name, or a suite that lost its backend prints a smaller
    green summary that nobody diffs.

    A scenario that declares NOTHING is refused rather than defaulted. There is
    no default to fall back on, and inventing one here would hand the scenario a
    claim its author never made; see the note above ``SELECTABLE_BACKENDS``."""
    declared = getattr(mod, "BACKENDS", None)
    if declared is None:
        return (
            "declares no BACKENDS. Every scenario must name the backend it was "
            "RUN against, because there is no default: add "
            "``BACKENDS = (\"real\",)`` once it has actually been run."
        )
    declared = tuple(declared)
    if backend in declared:
        return None
    return (
        f"declares BACKENDS={declared!r}, which does not include {backend!r}"
    )


def _backend_supported(mod, backend: str) -> bool:
    """True if *mod* supports the requested solver *backend*."""
    return backend_unsupported_reason(mod, backend) is None


def unrunnable_names(backend: str) -> dict[str, str]:
    """Registered scenarios this platform could host but *backend* cannot,
    mapped to the reason.

    Platform-gated scenarios are NOT in here: a macOS-only scenario on
    Linux is a routing fact that predates any backend, while an entry here
    is coverage the rig has lost."""
    out: dict[str, str] = {}
    for n, m in REGISTRY.items():
        if not _platform_supported(m):
            continue
        reason = backend_unsupported_reason(m, backend)
        if reason is not None:
            out[n] = reason
    return out


def server_only_names(backend: str) -> list[str]:
    """Names of scenarios that don't require Blender. Useful for CI
    runs on hosts without a Blender install.

    ``backend`` is required: there is no default backend to fall back on,
    and inventing one is how a caller ends up selecting a suite it did not
    ask for."""
    return [
        n for n, mod in REGISTRY.items()
        if not getattr(mod, "NEEDS_BLENDER", False)
        and _platform_supported(mod)
        and _backend_supported(mod, backend)
    ]


def get(name: str):
    """Return the scenario module, or None if unknown."""
    return REGISTRY.get(name)


def all_names(backend: str) -> list[str]:
    """Scenario names runnable on this platform against *backend*.

    ``backend`` is REQUIRED and has no default: a default would select
    whatever it happened to name and report that as if it were the suite.
    Callers say what they target, and ``unrunnable_names`` says what that
    costs."""
    return [
        n for n, m in REGISTRY.items()
        if _platform_supported(m) and _backend_supported(m, backend)
    ]
