# File: __init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from bpy.app.handlers import persistent  # pyright: ignore

from .ui import main_panel, console, solver, state
from .ui import dynamics  # Import the dynamics package
from . import mesh_ops  # Import the mesh_ops package (snap + merge only)
from .ops import zozo_contact_solver

reload_server = None

# One-shot timers scheduled during register() that we must be able to
# deregister on unregister() so they can't fire into a torn-down
# module namespace after a reload that happens inside their delay.
_deferred_timers: list = []


def _persist_session_id(*_args):
    """save_pre hook: mirror engine.state.session_id into the scene
    PropertyGroup so the id persists across Blender close/reopen.

    The engine keeps the canonical value; the PropertyGroup exists only
    so the saved .blend remembers which session produced its PC2 files,
    MESH_CACHE binds, and remote directory.
    """
    import bpy  # pyright: ignore
    try:
        from .core import facade
        from .models.groups import get_addon_data, has_addon_data
        sid = facade.engine.state.session_id or ""
        for scene in bpy.data.scenes:
            if not has_addon_data(scene):
                continue
            root = get_addon_data(scene)
            if hasattr(root, "state") and root.state.last_session_id != sid:
                root.state.last_session_id = sid
    except Exception:
        pass


def _save_manifest_post(*_args):
    """save_post hook: write the project manifest next to the .blend.

    Runs AFTER migrate_pc2_on_save so the PC2 inventory reflects the
    final on-disk file set.  Silent on failure — manifest is advisory.
    """
    try:
        from .core.manifest import write_manifest_now
        write_manifest_now()
    except Exception:
        pass


@persistent
def _disconnect_on_load(*_args):
    """load_pre hook: drop any active server connection before the new
    .blend takes over.

    The communicator and engine are module-level singletons that
    survive ``wm.read_homefile`` / ``wm.open_mainfile``; without this
    hook the new scene inherits the previous project's live socket and
    ONLINE phase, even though its own scene-level state defaults to
    OFFLINE. The result is an inconsistency where the UI thinks it's
    still connected to a server bound to a different project.
    Dispatching DisconnectRequested here clears engine.state and tears
    down ``runner._backend`` so the new scene starts with a clean
    slate.

    Marked ``@persistent`` so the handler survives file loads: Blender
    wipes non-persistent handlers each time a new file is opened, which
    would silently disable this disconnect on every load after the
    first one.
    """
    try:
        from .core.facade import communicator
        communicator.disconnect()
    except Exception:
        pass


def _disconnect_at_exit():
    """atexit hook: disconnect on Blender shutdown so a Windows-native
    server.py subprocess doesn't outlive its parent and become an
    orphan squatting on its port. Blender doesn't expose a quit_pre
    handler, so atexit is the only reliable seam to drive a clean
    disconnect on exit.
    """
    try:
        from .core.facade import communicator
        communicator.disconnect()
    except Exception:
        pass


def _reconcile_manifest_on_load(*_args):
    """load_post hook: auto-migrate legacy data, then reconcile manifest.

    Auto-migration runs BEFORE the first depsgraph update so code paths
    that now require UUIDs (merge_ops cleanup, encoder, overlay) don't
    silently drop un-migrated data.
    """
    try:
        from .core.migrate import needs_migration, migrate_legacy_data
        if needs_migration():
            result = migrate_legacy_data()
            from .models.console import console
            console.write(f"[auto-migrate] {result}")
    except Exception:
        pass
    try:
        from .core.manifest import reconcile_on_load
        reconcile_on_load()
    except Exception:
        pass


def _register_deferred_timer(fn, first_interval: float) -> None:
    import bpy  # pyright: ignore
    bpy.app.timers.register(fn, first_interval=first_interval)
    _deferred_timers.append(fn)


def _clear_deferred_timers() -> None:
    import bpy  # pyright: ignore
    for fn in _deferred_timers:
        if bpy.app.timers.is_registered(fn):
            try:
                bpy.app.timers.unregister(fn)
            except ValueError:
                pass
    _deferred_timers.clear()


def register():
    global reload_server
    try:
        _register_body()
    except Exception:
        # Partial registration would leave Blender with half-set-up
        # classes and scheduled timers that Blender's addon framework
        # won't tear down. Clean up on our own before re-raising so
        # the user gets a working "disabled" state.
        try:
            unregister()
        except Exception as e:
            print(f"Rollback unregister failed: {e}")
        raise


def _register_body():
    global reload_server
    state.register()
    main_panel.register()
    solver.register()
    dynamics.register()
    mesh_ops.register()
    console.register()
    zozo_contact_solver.register()

    # Start the persistent engine timer
    from .core.facade import ensure_engine_timer
    ensure_engine_timer()

    # Migrate existing groups to UUID system
    import bpy  # pyright: ignore

    # Clean up any existing groups with messy names on startup
    def cleanup_group_names():
        mutated = False
        try:
            if hasattr(bpy.context, "scene") and bpy.context.scene:
                from .models.groups import iterate_active_object_groups

                scene = bpy.context.scene

                # Clean up active groups using display indices (1, 2, 3, 4, 5)
                for display_index, group in enumerate(
                    iterate_active_object_groups(scene)
                ):
                    # Ensure UUID exists
                    if not group.uuid:
                        group.ensure_uuid()
                    # Clean up messy names using display index
                    if not group.name or "Group Group" in group.name:
                        group.name = f"Group {display_index + 1}"
                        mutated = True
                        print(f"Cleaned up group name to: {group.name}")

                print("Group cleanup completed")
        except Exception as e:
            print(f"Group cleanup failed: {e}")
        if mutated:
            try:
                from .core.utils import redraw_all_areas
                redraw_all_areas(bpy.context)
            except Exception:
                pass
        return None  # Remove timer

    _register_deferred_timer(cleanup_group_names, 1.0)

    # Register UUID rename detection handler
    from .core.uuid_registry import register as uuid_register
    uuid_register()

    # Register save_post handler for PC2 temp→data migration. Compare
    # existing handlers by __name__ rather than identity so any stale
    # handler from an earlier module generation (pre-reload) gets
    # detected and isn't double-added.
    from .core.pc2 import (
        ensure_curve_handler,
        migrate_pc2_on_save,
        reset_render_counter,
        warn_missing_frames_on_render,
    )

    if not any(
        getattr(h, "__name__", "") == "migrate_pc2_on_save"
        for h in bpy.app.handlers.save_post
    ):
        bpy.app.handlers.save_post.append(migrate_pc2_on_save)

    if not any(
        getattr(h, "__name__", "") == "reset_render_counter"
        for h in bpy.app.handlers.render_init
    ):
        bpy.app.handlers.render_init.append(reset_render_counter)
    if not any(
        getattr(h, "__name__", "") == "warn_missing_frames_on_render"
        for h in bpy.app.handlers.render_pre
    ):
        bpy.app.handlers.render_pre.append(warn_missing_frames_on_render)

    # Session-id persistence: write the live engine's session id into the
    # scene PropertyGroup just before the user saves, so the id survives
    # close/reopen and can be compared against the remote server to
    # detect orphaned sims on reconnect.
    if not any(
        getattr(h, "__name__", "") == "_persist_session_id"
        for h in bpy.app.handlers.save_pre
    ):
        bpy.app.handlers.save_pre.append(_persist_session_id)

    # Manifest lifecycle: save_post writes (after pc2 migration has
    # moved temp files into place), load_post reconciles against the
    # current addon version and logs orphans.
    if not any(
        getattr(h, "__name__", "") == "_save_manifest_post"
        for h in bpy.app.handlers.save_post
    ):
        bpy.app.handlers.save_post.append(_save_manifest_post)
    if not any(
        getattr(h, "__name__", "") == "_reconcile_manifest_on_load"
        for h in bpy.app.handlers.load_post
    ):
        bpy.app.handlers.load_post.append(_reconcile_manifest_on_load)
    if not any(
        getattr(h, "__name__", "") == "_disconnect_on_load"
        for h in bpy.app.handlers.load_pre
    ):
        bpy.app.handlers.load_pre.append(_disconnect_on_load)

    # atexit: cover Blender shutdown (no quit_pre handler exists). Idempotent
    # under register/unregister churn — atexit dedupes on the function object.
    import atexit as _atexit
    _atexit.unregister(_disconnect_at_exit)
    _atexit.register(_disconnect_at_exit)

    # Register curve handlers (persistent — survive file loads)
    ensure_curve_handler()

    # Note: MESH_CACHE self-heal is driven by the PPF_OT_FramePump modal
    # (registered further down), not from here — ID writes from addon
    # register context are unreliable on Blender 5.x when the reload
    # originates from the debug-server TCP handler.

    # Restart servers that were running before a manual reload
    def _restart_servers_after_reload():
        from .core.reload_server import get_restart_server_state, ReloadServer, get_reload_server_status
        state = get_restart_server_state()
        if state is None:
            return None  # fresh launch, not a reload
        global reload_server
        if state.get("debug") and not get_reload_server_status():
            reload_server = ReloadServer()
            reload_server.start()
        if state.get("mcp"):
            try:
                from .mcp.mcp_server import start_mcp_server, is_mcp_running
                from .models.defaults import DEFAULT_MCP_PORT
                if not is_mcp_running():
                    start_mcp_server(DEFAULT_MCP_PORT)
            except Exception as e:
                print(f"MCP restart after reload failed: {e}")
        # Kick the 3D-view panels to redraw so the Start/Stop button
        # reflects the just-restarted server without a manual hover.
        try:
            wm = bpy.context.window_manager
            if wm:
                for window in wm.windows:
                    for area in window.screen.areas:
                        if area.type == "VIEW_3D":
                            area.tag_redraw()
        except Exception:
            pass
        return None

    _register_deferred_timer(_restart_servers_after_reload, 1.0)

    # Frame-pump modal: drives apply_animation + MESH_CACHE self-heal
    # from a modal-operator timer context (the only context where
    # Blender 5.x permits the ID writes those involve).
    from .core import frame_pump
    frame_pump.register()

    # Everything above succeeded — tell the persistent tick it's safe to
    # read/modify PropertyGroup state. Must be the final line of register().
    try:
        from .core.facade import mark_addon_ready
        mark_addon_ready(True)
    except Exception as e:
        print(f"mark_addon_ready failed: {e}")


def unregister():
    global reload_server

    # Mark the addon unsafe for any in-flight timer tick before anything
    # else. This must happen first so the tick gate flips immediately,
    # preventing a mid-unregister tick from touching partially-freed
    # PropertyGroup state (has triggered Blender segfaults on reload).
    try:
        from .core.facade import mark_addon_ready
        mark_addon_ready(False)
    except Exception as e:
        print(f"mark_addon_ready(False) failed: {e}")

    # Deregister any one-shot timers scheduled at register-time that
    # haven't fired yet. After reload they would call into a freed
    # module's globals.
    try:
        _clear_deferred_timers()
    except Exception as e:
        print(f"_clear_deferred_timers failed: {e}")

    # Tell the frame-pump modal to exit on its next TIMER tick. Must
    # happen before unregister_class() so the modal instance detaches
    # cleanly (Blender will otherwise crash if we unregister a class
    # whose modal is still alive).
    try:
        from .core import frame_pump
        frame_pump.unregister()
    except Exception as e:
        print(f"frame_pump.unregister failed: {e}")

    # Stop engine worker thread and timer
    try:
        from .core.facade import cleanup
        cleanup()
    except Exception as e:
        print(f"Error during engine cleanup: {e}")

    # Stop reload server. Do NOT clear the restart-state sentinel in
    # driver_namespace here: _reload_phase1_disable writes it just
    # before calling addon_disable (which runs this function), so
    # popping here would drop the handoff the next register() needs to
    # rebuild the servers.
    if reload_server:
        reload_server.stop()
        reload_server = None

    # Remove handlers. Match by __name__ rather than object identity so
    # stale handlers from a previous module generation (before reload)
    # get removed even though the function object they point at no
    # longer equals the one we just imported.
    try:
        import bpy  # pyright: ignore
        from .core.pc2 import remove_curve_handler
        remove_curve_handler()
        for handler_list in (bpy.app.handlers.frame_change_pre,
                             bpy.app.handlers.frame_change_post,
                             bpy.app.handlers.depsgraph_update_post):
            for h in list(handler_list):
                if getattr(h, "__name__", "") == "curve_frame_change_handler":
                    handler_list.remove(h)
        for h in list(bpy.app.handlers.save_post):
            if getattr(h, "__name__", "") == "migrate_pc2_on_save":
                bpy.app.handlers.save_post.remove(h)
        for h in list(bpy.app.handlers.save_pre):
            if getattr(h, "__name__", "") == "_persist_session_id":
                bpy.app.handlers.save_pre.remove(h)
        for h in list(bpy.app.handlers.save_post):
            if getattr(h, "__name__", "") == "_save_manifest_post":
                bpy.app.handlers.save_post.remove(h)
        for h in list(bpy.app.handlers.load_post):
            if getattr(h, "__name__", "") == "_reconcile_manifest_on_load":
                bpy.app.handlers.load_post.remove(h)
        for h in list(bpy.app.handlers.load_pre):
            if getattr(h, "__name__", "") == "_disconnect_on_load":
                bpy.app.handlers.load_pre.remove(h)
        import atexit as _atexit
        _atexit.unregister(_disconnect_at_exit)
        for h in list(bpy.app.handlers.render_init):
            if getattr(h, "__name__", "") == "reset_render_counter":
                bpy.app.handlers.render_init.remove(h)
        for h in list(bpy.app.handlers.render_pre):
            if getattr(h, "__name__", "") == "warn_missing_frames_on_render":
                bpy.app.handlers.render_pre.remove(h)
    except Exception:
        pass

    # Unregister UUID handlers
    try:
        from .core.uuid_registry import unregister as uuid_unregister
        uuid_unregister()
    except Exception:
        pass

    # Reset heal-log dedup set so a broken PC2 from the prior session
    # doesn't silently suppress logging after re-enable.
    try:
        from .core import client as _client_mod
        _client_mod._heal_logged.clear()
    except Exception:
        pass

    # Cleanup MCP server before unregistering other components
    try:
        from .mcp.mcp_server import cleanup_mcp_server

        cleanup_mcp_server()
    except ImportError:
        pass  # MCP module not available
    except Exception as e:
        print(f"Error during MCP cleanup: {e}")

    zozo_contact_solver.unregister()
    console.unregister()
    mesh_ops.unregister()
    dynamics.unregister()
    solver.unregister()
    main_panel.unregister()
    state.unregister()
