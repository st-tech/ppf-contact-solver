#!/usr/bin/env bash
# capture.sh — Launch Blender headlessly, capture UI widget screenshots, quit.
#
# ============================================================================
# NOTES FOR LLM AGENTS
# ============================================================================
#
# This script has two modes: non-interactive (default) and --interactive.
#
# NON-INTERACTIVE MODE (recommended for screenshot generation):
#   Launches Blender, captures the requested widget(s), annotates PNGs,
#   and quits. One Blender process per run. This is the RELIABLE path.
#
#   Prints "CAPTURE_DONE" on stdout when finished — grep for it when
#   piping into other tools. Each run takes ~15-20s (startup + work + quit).
#
#   THREE WAYS TO CAPTURE:
#
#   1. Specific widgets (widget highlighted with red box + caption):
#        bash capture.sh --profile ~/connection_profile.toml -o shots/ \
#          "MAIN_PT_RemotePanel:Connect" \
#          "SSH_PT_SolverPanel:Transfer"
#
#   2. Every widget in one or more panels (same highlight style):
#        bash capture.sh --profile ~/connection_profile.toml -o shots/ \
#          --all MAIN_PT_RemotePanel --all SSH_PT_SolverPanel
#
#   3. Clean panel overview (no widget highlight, filename = <panel_id>.png):
#        bash capture.sh --profile ~/connection_profile.toml -o shots/ \
#          --panel-only \
#          --all MAIN_PT_RemotePanel \
#          --all SSH_PT_SolverPanel \
#          --all SSH_PT_ObjectGroupsManager \
#          --all DYNAMICS_PT_Groups
#
#   ALWAYS batch multiple panels in one run. Re-launching Blender per
#   panel costs ~15s each. The script happily hides all non-target
#   panels in every iteration and restores them between targets.
#
# INTERACTIVE MODE (--interactive):
#   Launches Blender and stays alive. Starts reload + MCP servers on
#   auto-picked ports. Prints CAPTURE_READY with port numbers.
#   The LLM can then drive Blender via debug/main.py.
#
#   KNOWN ISSUE: The MCP server has a 5-second task-poll timeout
#   (hardcoded in mcp/task_system.py). Any exec that takes longer
#   (e.g. widget capture with pixel-diffing) returns a timeout error
#   even though the code runs to completion inside Blender. This makes
#   interactive mode UNRELIABLE for capture work. Use it only for
#   short commands (connect, status checks, scene setup).
#
#   For screenshot capture, use non-interactive mode instead.
#
# SPLASH SCREEN: Blender shows a splash on fresh launch. This script
#   dismisses it via --enable-event-simulate + ESC keypress in the
#   _setup_sidebar() function. If the splash still appears in your
#   screenshots, it means the ESC didn't fire — check that
#   --enable-event-simulate is on the Blender launch line.
#
# PORT ISOLATION: Every run auto-picks 2 ephemeral ports (MCP and
#   reload) so parallel instances never collide. The ports are freed
#   when Blender exits.
#
# PROJECT NAME ISOLATION: Every session gets a unique project name
#   (capture-<uuid8>) so remote directories don't clash when a capture
#   runs alongside another capture.sh or a user's live Blender session.
#
# PROFILE: --profile PATH[:ENTRY] loads a TOML connection profile and
#   applies it before capture so connection fields are filled and the
#   panel shows real data instead of empty defaults.
#
# ============================================================================
set -eo pipefail
# Note: -u is NOT set because TARGETS[@] triggers "unbound variable"
# when the array is empty (common in --interactive and --all modes).

usage() {
    cat << 'EOF'
capture.sh — Capture annotated screenshots of Blender addon UI widgets.

Launches Blender (window hidden), locates each requested widget by pixel-
diffing, saves a cropped+annotated PNG per widget, then quits.  Multiple
instances can run in parallel (each picks a free TCP port automatically).

USAGE
    bash capture.sh -o DIR  PANEL:LABEL [PANEL:LABEL ...]
    bash capture.sh -o DIR  --all PANEL [--all PANEL ...]
    bash capture.sh -o DIR  --panel-only --all PANEL [--all PANEL ...]
    bash capture.sh -o DIR  --interactive [--profile PATH]
    bash capture.sh -h

REQUIRED
    -o DIR              Output directory.  Each widget gets its own PNG
                        named <PANEL>__<LABEL>.png, plus a manifest.json.

TARGETS (positional arguments)
    PANEL:LABEL         Capture the widget whose draw_string is LABEL in
                        panel PANEL.  LABEL is the exact text shown in the
                        UI (e.g. "Connect", "FPS: 60", "Debug Options").

    PANEL:op=OPID       Capture an icon-only operator button by its
                        bl_idname (e.g. "MAIN_PT_RemotePanel:op=ssh.save_profile").

OPTIONS
    --all PANEL         Capture every widget found in PANEL (auto-discovers
                        labels from the introspect tree).  Can be passed
                        multiple times to process several panels in one run.

    --panel-only        Skip per-widget enumeration; save one cropped PNG
                        per --all panel named <PANEL>.png (no widget
                        highlight).  Ideal for docs overview shots.

    --expand            Auto-expand all collapsible sections (show_wind,
                        show_advanced_parameters, …) before capturing.

    --profile PATH[:ENTRY]
                        Load a TOML connection profile so fields are filled.
                        If ENTRY is omitted, the first entry is used.

    --pre-python PATH   Execute a Python file inside Blender after the
                        connection profile is applied, before sidebar setup
                        and capture. Use this to seed scene state (e.g.
                        dyn_params, collider lists) so panels render with
                        realistic content. The script is exec'd with `bpy`
                        already imported in its globals.

    --interactive       Launch Blender and stay alive for LLM control.
                        Prints CAPTURE_READY with ports. See notes above
                        about the 5s MCP timeout limitation.

    --sidebar-width N   Logical sidebar width in pixels (default: 560).

    --hide P1,P2,...    Comma-separated extra panel ids to force-hide.

    -h, --help          Show this help and exit.

AVAILABLE PANELS
    MAIN_PT_RemotePanel              "Backend Communicator"
    SSH_PT_SolverPanel               "Solver"
    SSH_PT_ObjectGroupsManager       "Scene Configuration"
    DYNAMICS_PT_Groups               "Dynamics Groups"
    SNAPMERGE_PT_SnapAndMerge        "Snap and Merge"
    VISUALIZATION_PT_Visualization   "Visualization"

EXAMPLES
    # Capture the Connect button from Backend Communicator:
    bash capture.sh --profile ~/connection_profile.toml -o /tmp/shots \
        "MAIN_PT_RemotePanel:Connect"

    # Capture multiple widgets in one run (shares Blender startup):
    bash capture.sh --profile ~/connection_profile.toml -o /tmp/shots \
        "MAIN_PT_RemotePanel:Connect" \
        "MAIN_PT_RemotePanel:Open Profile" \
        "MAIN_PT_RemotePanel:Debug Options"

    # Clean panel overviews for docs (one <PANEL>.png per --all panel):
    bash capture.sh --profile ~/connection_profile.toml -o docs/images/ \
        --panel-only \
        --all MAIN_PT_RemotePanel \
        --all SSH_PT_SolverPanel \
        --all SSH_PT_ObjectGroupsManager \
        --all DYNAMICS_PT_Groups \
        --all SNAPMERGE_PT_SnapAndMerge \
        --all VISUALIZATION_PT_Visualization

    # Every widget in two panels with sections expanded:
    bash capture.sh --profile ~/connection_profile.toml -o /tmp/shots \
        --expand \
        --all SSH_PT_ObjectGroupsManager \
        --all DYNAMICS_PT_Groups

    # Interactive mode (for LLM-driven workflows):
    bash capture.sh --interactive --profile ~/connection_profile.toml \
        -o /tmp/shots &
    # Then use: python blender_addon/debug/main.py --mcp-port <PORT> exec '...'
    # WARNING: any exec taking >5s will timeout. Use non-interactive for captures.

OUTPUT
    DIR/<PANEL>__<LABEL>.png   Per-widget capture: cropped to the panel,
                               with the widget highlighted in a red box
                               plus a caption above it.

    DIR/<PANEL>.png            --panel-only capture: cropped to the panel,
                               no widget highlight, no caption.

    DIR/manifest.json          Machine-readable results for every capture.
    DIR/.scratch/              Working files (baselines, swap screenshots).

    Stdout prints "CAPTURE_DONE" when a non-interactive run finishes.
EOF
    exit 0
}

BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
if [ ! -x "$BLENDER" ]; then
    echo "Error: Blender not found at $BLENDER" >&2
    exit 1
fi

# --- Parse arguments --------------------------------------------------------
OUTDIR=""
EXPAND=0
INTERACTIVE=0
PANEL_ONLY=0
SIDEBAR_WIDTH=560
HIDE_PANELS=""
PROFILE=""
PRE_PYTHON=""
TARGETS=()
ALL_PANELS=()  # Array — multiple --all flags allowed

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)    usage ;;
        -o)           OUTDIR="$2"; shift 2 ;;
        --expand)     EXPAND=1; shift ;;
        --all)        ALL_PANELS+=("$2"); shift 2 ;;
        --panel-only) PANEL_ONLY=1; shift ;;
        --sidebar-width) SIDEBAR_WIDTH="$2"; shift 2 ;;
        --hide)       HIDE_PANELS="$2"; shift 2 ;;
        --profile)    PROFILE="$2"; shift 2 ;;
        --pre-python) PRE_PYTHON="$2"; shift 2 ;;
        --interactive) INTERACTIVE=1; shift ;;
        -*)           echo "Unknown option: $1" >&2; exit 1 ;;
        *)            TARGETS+=("$1"); shift ;;
    esac
done

if [ -z "$OUTDIR" ]; then
    echo "Usage: capture.sh -o OUTDIR [--expand] [--all PANEL] panel:label ..." >&2
    exit 1
fi
mkdir -p "$OUTDIR"

# --panel-only implies we don't need a per-widget target; use first --all panel
# (or error out if none given).
if [ "$PANEL_ONLY" -eq 1 ] && [ "${#ALL_PANELS[@]}" -eq 0 ] && [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "Error: --panel-only requires --all PANEL [--all PANEL ...]" >&2
    exit 1
fi

if [ "$INTERACTIVE" -eq 0 ] && [ "${#TARGETS[@]}" -eq 0 ] && [ "${#ALL_PANELS[@]}" -eq 0 ]; then
    echo "Error: specify at least one panel:label or --all PANEL, or use --interactive" >&2
    exit 1
fi

# --- Pick free ports --------------------------------------------------------
# Auto-pick 2 ephemeral ports so parallel instances never collide.
# Ports are released immediately after binding so Blender can re-bind them.
FREE_PORTS=$(python3 -c "
import socket
ports = []
for _ in range(2):
    s = socket.socket(); s.bind(('', 0)); ports.append(s.getsockname()[1]); s.close()
print(' '.join(str(p) for p in ports))
")
read MCP_PORT RELOAD_PORT <<< "$FREE_PORTS"

# --- Build the Python payload -----------------------------------------------
# We embed the entire capture logic as a Python script that Blender executes
# on startup, then quits.

# Serialize TARGETS and ALL_PANELS arrays to JSON via argv. Empty arrays
# expand to no args, so sys.argv[1:] is [] and we get "[]" back.
TARGETS_JSON=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${TARGETS[@]}")
ALL_PANELS_JSON=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${ALL_PANELS[@]}")

PYTHON_SCRIPT=$(cat << 'PYEOF'
import bpy, sys, os, json, ctypes

OUTDIR = os.environ["_CAPTURE_OUTDIR"]
EXPAND = os.environ.get("_CAPTURE_EXPAND", "0") == "1"
PANEL_ONLY = os.environ.get("_CAPTURE_PANEL_ONLY", "0") == "1"
ALL_PANELS = json.loads(os.environ.get("_CAPTURE_ALL_PANELS", "[]"))
SIDEBAR_WIDTH = int(os.environ.get("_CAPTURE_SIDEBAR_WIDTH", "560"))
HIDE_PANELS = [p for p in os.environ.get("_CAPTURE_HIDE_PANELS", "").split(",") if p]
TARGETS = json.loads(os.environ.get("_CAPTURE_TARGETS", "[]"))
PROFILE = os.environ.get("_CAPTURE_PROFILE", "")
INTERACTIVE = os.environ.get("_CAPTURE_INTERACTIVE", "0") == "1"
MCP_PORT = int(os.environ.get("_CAPTURE_MCP_PORT", "9633"))
RELOAD_PORT = int(os.environ.get("_CAPTURE_RELOAD_PORT", "8765"))
PRE_PYTHON = os.environ.get("_CAPTURE_PRE_PYTHON", "")

# The addon is enabled before this script runs via --addons on the
# Blender command line, so its modules are already in sys.modules.
# Strip a known leaf so the resolved name is the addon's root package
# under either the extension layout (bl_ext.user_default.<id>) or a
# legacy single-segment layout.
_PKG = next(n.removesuffix(".ui.solver") for n in sys.modules if n.endswith(".ui.solver"))
cap = __import__(_PKG + ".ui.capture", fromlist=["capture"])
groups_mod = __import__(_PKG + ".models.groups", fromlist=["groups"])

def _run_pre_python():
    """Exec an optional prep script path after the profile is applied.
    Lets callers seed per-run scene state (e.g. dyn_params) so panels
    render with non-default content."""
    if not PRE_PYTHON:
        return
    with open(PRE_PYTHON) as f:
        src = f.read()
    exec(compile(src, PRE_PYTHON, "exec"), {"bpy": bpy, "__name__": "__main__"})
    print(f"capture.sh: ran pre-python {PRE_PYTHON}")

def _apply_connection_profile():
    """Load a TOML connection profile and apply it so fields are filled."""
    if not PROFILE:
        return
    # Parse PATH or PATH:ENTRY. If PROFILE doesn't exist as-is and contains
    # a colon, treat the tail as the entry name.
    if ":" in PROFILE and not os.path.exists(PROFILE):
        path, entry = PROFILE.rsplit(":", 1)
    else:
        path, entry = PROFILE, ""

    profile_mod = __import__(_PKG + ".core.profile", fromlist=["profile"])
    profiles = profile_mod.load_profiles(path)
    if not profiles:
        print(f"capture.sh: WARNING: no profiles found in {path}")
        return

    if not entry:
        entry = sorted(profiles.keys())[0]
    if entry not in profiles:
        print(f"capture.sh: WARNING: profile '{entry}' not found, available: {sorted(profiles.keys())}")
        return

    root = groups_mod.get_addon_data()
    profile_mod.apply_profile(profiles[entry], root.ssh_state)
    print(f"capture.sh: applied connection profile '{entry}' from {path}")

def _setup_sidebar():
    """Dismiss splash, open the sidebar, and switch to the addon tab."""
    win = bpy.context.window_manager.windows[0]

    # Dismiss splash popup
    bpy.context.preferences.view.show_splash = False
    try:
        win.event_simulate('ESC', 'PRESS')
        win.event_simulate('ESC', 'RELEASE')
    except Exception:
        pass
    cap.force_redraw(iterations=3)

    # Open sidebar
    for a in win.screen.areas:
        if a.type == "VIEW_3D":
            for s in a.spaces:
                if s.type == "VIEW_3D":
                    s.show_region_ui = True

    cap.force_redraw(iterations=3)

    # Widen sidebar
    for a in win.screen.areas:
        if a.type == "VIEW_3D":
            for s in a.spaces:
                if s.type == "VIEW_3D":
                    s.show_region_ui = False
    cap.force_redraw(iterations=1)
    for a in win.screen.areas:
        if a.type == "VIEW_3D":
            for r in a.regions:
                if r.type == "UI":
                    ptr = r.as_pointer()
                    ctypes.c_int16.from_address(ptr + 198).value = SIDEBAR_WIDTH
    for a in win.screen.areas:
        if a.type == "VIEW_3D":
            for s in a.spaces:
                if s.type == "VIEW_3D":
                    s.show_region_ui = True

    # Switch to addon tab
    addon_cat = bpy.types.MAIN_PT_RemotePanel.bl_category
    for a in win.screen.areas:
        if a.type == "VIEW_3D":
            for r in a.regions:
                if r.type == "UI":
                    r.active_panel_category = addon_cat

    cap.force_redraw(iterations=5)

def _run_interactive():
    """Interactive mode: apply profile, set unique ports+project, start
    reload+MCP servers, open sidebar, then stay alive for LLM control."""
    try:
        _assign_unique_project_name()

        root = groups_mod.get_addon_data()
        # Override ports to auto-picked values
        root.state.reload_port = RELOAD_PORT
        root.state.mcp_port = MCP_PORT

        _apply_connection_profile()
        _run_pre_python()
        _setup_sidebar()

        # Start reload server on the session-specific port
        reload_mod = __import__(_PKG + ".core.reload_server", fromlist=["reload_server"])
        reload_mod.start_reload_server(RELOAD_PORT)

        # Start MCP server on the session-specific port
        mcp_mod = __import__(_PKG + ".mcp.mcp_server", fromlist=["mcp_server"])
        mcp_mod.start_mcp_server(MCP_PORT)

        # Print connection info so the LLM knows how to reach us
        print(f"CAPTURE_READY reload_port={RELOAD_PORT} mcp_port={MCP_PORT} project={root.state.project_name}")
        sys.stdout.flush()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"capture.sh: interactive setup FAILED: {e}", file=sys.stderr)
        bpy.ops.wm.quit_blender()
    return None  # don't re-schedule; Blender stays alive

def _assign_unique_project_name():
    """Stamp a unique project name on every session. Remote project
    directories are keyed on this, so two parallel Blender instances
    (e.g. another capture.sh run, or a user's live session) would
    otherwise clobber each other's state."""
    import uuid as _uuid
    root = groups_mod.get_addon_data()
    root.state.project_name = f"capture-{_uuid.uuid4().hex[:8]}"

def _run_capture():
    try:
        _assign_unique_project_name()
        _apply_connection_profile()
        _run_pre_python()
        _do_capture()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"capture.sh: FAILED: {e}", file=sys.stderr)
    bpy.ops.wm.quit_blender()
    return None

def _do_capture():
    _setup_sidebar()

    # Determine which widgets to capture
    work = []  # list of (panel_id, draw_string, op_idname_or_None)

    # --panel-only mode: skip widget enumeration, just record one placeholder
    # target per panel so each panel gets exactly one cropped output PNG.
    if PANEL_ONLY:
        for panel_id in ALL_PANELS:
            work.append((panel_id, "__PANEL_OVERVIEW__", None))
    else:
        # Enumerate every widget in every --all panel.
        for all_panel in ALL_PANELS:
            tree = cap.introspect_panel(all_panel)
            def collect(items, _p=all_panel):
                for item in items:
                    ds = item.get("draw_string", "")
                    op = item.get("operator", "")
                    t = item.get("type", "")
                    if isinstance(t, str) and t.startswith("LAYOUT_"):
                        if isinstance(item.get("items"), list):
                            collect(item["items"])
                    elif t != 24:
                        if ds:
                            work.append((_p, ds, None))
                        elif op:
                            oid = op[len("bpy.ops."):].split("(")[0]
                            work.append((_p, "", oid))
            for root in tree:
                if root.get("type") == "LAYOUT_ROOT":
                    collect(root.get("items", []))
    for spec in TARGETS:
        if ":" not in spec:
            print(f"capture.sh: skipping malformed target {spec!r}")
            continue
        panel, rest = spec.split(":", 1)
        if rest.startswith("op="):
            work.append((panel, "", rest[3:]))
        else:
            work.append((panel, rest, None))

    if not work:
        print("capture.sh: nothing to capture")
        return

    # Group work by panel (dict preserves insertion order on py3.7+)
    by_panel: dict = {}
    for panel, ds, op in work:
        by_panel.setdefault(panel, []).append((ds, op))

    # Discover all addon panels so we can hide the ones we don't need
    all_addon_panels = []
    addon_cat = bpy.types.MAIN_PT_RemotePanel.bl_category
    for attr in dir(bpy.types):
        cls = getattr(bpy.types, attr)
        if getattr(cls, "bl_category", "") == addon_cat and hasattr(cls, "bl_idname"):
            all_addon_panels.append(cls.bl_idname)

    scratch = os.path.join(OUTDIR, ".scratch")
    os.makedirs(scratch, exist_ok=True)

    results = []
    panel_bboxes = {}

    @classmethod
    def _false_poll(cls, context):
        return False

    for panel, items in by_panel.items():
        # Hide every addon panel EXCEPT the target one
        saved_polls = {}
        others = [p for p in all_addon_panels if p != panel]
        # Also hide any explicitly requested panels
        for extra in HIDE_PANELS:
            if extra and extra not in others:
                others.append(extra)
        for pid in others:
            cls = getattr(bpy.types, pid, None)
            if cls is None:
                continue
            saved_polls[pid] = cls.__dict__.get("poll")
            cls.poll = _false_poll

        cap.force_redraw(iterations=5)

        # Close collapsible sections that do NOT contain any of the
        # requested widgets.  For each open show_* bool, tentatively
        # close it, re-introspect, and check if any requested target
        # vanished.  If yes → reopen; if no → leave closed.
        requested_labels = set(ds for ds, op in items if ds)
        requested_ops = set(op for ds, op in items if op)
        owners_cache = cap._collect_prop_owners("")
        collapsed = []
        try:
            tree_pre = cap.introspect_panel(panel)
            show_props = []
            def _find_shows(items):
                for item in items:
                    rna = item.get("rna", "")
                    parsed = cap._parse_rna_ref(rna) if rna else None
                    if parsed:
                        rc, ra, _ = parsed
                        if ra.startswith("show_"):
                            owner = cap._find_prop_owner_by_rna_class(owners_cache, rc)
                            if owner:
                                bl = getattr(type(owner), "bl_rna", None)
                                pr = bl.properties.get(ra) if bl else None
                                if pr and pr.type == "BOOLEAN" and getattr(owner, ra):
                                    show_props.append((owner, ra))
                    sub = item.get("items")
                    if isinstance(sub, list):
                        _find_shows(sub)
            for root in tree_pre:
                if root.get("type") == "LAYOUT_ROOT":
                    _find_shows(root.get("items", []))

            def _tree_labels(tree):
                found = set()
                def walk(items):
                    for item in items:
                        ds = item.get("draw_string", "")
                        if ds: found.add(ds)
                        op = item.get("operator", "")
                        if op: found.add(op[len("bpy.ops."):].split("(")[0])
                        sub = item.get("items")
                        if isinstance(sub, list): walk(sub)
                for root in tree:
                    if root.get("type") == "LAYOUT_ROOT":
                        walk(root.get("items", []))
                return found

            for owner, attr in show_props:
                setattr(owner, attr, False)
                cap.force_redraw(iterations=2)
                tree_closed = cap.introspect_panel(panel)
                labels_after = _tree_labels(tree_closed)
                lost = (requested_labels - labels_after) | (requested_ops - labels_after)
                if lost:
                    setattr(owner, attr, True)
                else:
                    collapsed.append((owner, attr, True))
            cap.force_redraw(iterations=3)
        except Exception:
            pass

        # Measure tight panel crop via noop-stub diff
        baseline = os.path.join(scratch, f"baseline_{panel}.png")
        cap.screenshot(baseline)
        noop_png = os.path.join(scratch, f"noop_{panel}.png")
        try:
            with cap.stub_panel_draw(panel, cap.stub_noop()):
                cap.screenshot(noop_png)
            tight_bbox = cap.diff_bbox(baseline, noop_png, threshold=5)
        except Exception:
            tight_bbox = None
        if tight_bbox:
            # The noop diff captures the panel content but not the header
            # (header doesn't change). Expand upward by ~1.5 row heights
            # to include the panel header bar.
            header_margin = int(cap.row_height_px() * 1.5)
            tight_bbox = cap.Rect(
                tight_bbox.left, max(tight_bbox.top - header_margin, 0),
                tight_bbox.right, tight_bbox.bottom,
            )
            panel_bboxes[panel] = list(tight_bbox)
        # Else: panel is collapsed (bl_options DEFAULT_CLOSED) and has no
        # drawable body to diff against. Leaving panel_bboxes[panel] unset
        # causes downstream code to skip it with a clear error message.

        # Re-take baseline (stub leaves drift)
        cap.force_redraw(iterations=5)
        cap.screenshot(baseline)
        loc = cap.WidgetLocator(baseline, scratch)

        # Expand sections if requested (re-open what we closed + more)
        expand_ctx = None
        if EXPAND:
            # First restore closed sections
            for owner, attr, old in collapsed:
                setattr(owner, attr, old)
            collapsed.clear()
            expand_ctx = loc.auto_expand_sections(panel)
            expand_ctx.__enter__()
            cap.force_redraw(iterations=3)
            cap.screenshot(baseline)
            loc = cap.WidgetLocator(baseline, scratch)

        seen_labels = set()
        for ds, op in items:
            key = f"{ds or op}"
            if key in seen_labels:
                continue
            seen_labels.add(key)

            # --panel-only placeholder: skip widget locator, just emit a
            # manifest entry. The host-side annotation step does the actual
            # crop (Blender's bundled Python lacks Pillow on some setups).
            if ds == "__PANEL_OVERVIEW__":
                crop_bbox = panel_bboxes.get(panel)
                if crop_bbox is None:
                    results.append({
                        "panel": panel, "draw_string": "", "op_idname": "",
                        "status": "error",
                        "error": (
                            f"{panel} is collapsed (DEFAULT_CLOSED) and has "
                            "no drawable body. Open it manually in a saved "
                            ".blend or use --interactive to expand it first."
                        ),
                    })
                    continue
                results.append({
                    "panel": panel,
                    "draw_string": "",
                    "op_idname": "",
                    "status": "ok",
                    "panel_only": True,
                    "file": f"{panel}.png",
                    "baseline": os.path.basename(baseline),
                })
                continue

            entry = {"panel": panel, "draw_string": ds, "op_idname": op}
            try:
                kw = {}
                if op:
                    kw["op_idname"] = op
                elif not ds:
                    continue
                rect = loc.widget_rect(panel, ds, **kw)
                entry["rect"] = list(rect)
                entry["wh"] = [rect.width, rect.height]
                entry["status"] = "ok"

                label = ds or op
                safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
                entry["file"] = f"{panel}__{safe}.png"
                entry["baseline"] = os.path.basename(baseline)
            except cap.WidgetNotRendered as e:
                entry["status"] = "skipped"
                entry["reason"] = str(e)
            except Exception as e:
                entry["status"] = "error"
                entry["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            results.append(entry)

        # Clean up expand context + restore collapsed sections
        if expand_ctx is not None:
            expand_ctx.__exit__(None, None, None)
        for owner, attr, old in collapsed:
            setattr(owner, attr, old)

        # Restore polls
        for pid, poll in saved_polls.items():
            cls = getattr(bpy.types, pid)
            if poll is None:
                if "poll" in cls.__dict__:
                    del cls.poll
            else:
                cls.poll = poll
        cap.force_redraw(iterations=2)

    # Write results manifest
    manifest_data = {
        "panel_bboxes": panel_bboxes,
        "results": results,
    }
    manifest = os.path.join(OUTDIR, "manifest.json")
    with open(manifest, "w") as f:
        json.dump(manifest_data, f, indent=2)

    ok = sum(1 for r in results if r.get("status") == "ok")
    skip = sum(1 for r in results if r.get("status") == "skipped")
    err = sum(1 for r in results if r.get("status") == "error")
    print(f"capture.sh: {ok} captured, {skip} skipped, {err} errors → {OUTDIR}")

if INTERACTIVE:
    bpy.app.timers.register(_run_interactive, first_interval=3.0)
else:
    bpy.app.timers.register(_run_capture, first_interval=3.0)
PYEOF
)

# --- Launch Blender (hidden window) -----------------------------------------
export _CAPTURE_OUTDIR="$OUTDIR"
export _CAPTURE_EXPAND="$EXPAND"
export _CAPTURE_PANEL_ONLY="$PANEL_ONLY"
export _CAPTURE_ALL_PANELS="$ALL_PANELS_JSON"
export _CAPTURE_SIDEBAR_WIDTH="$SIDEBAR_WIDTH"
export _CAPTURE_HIDE_PANELS="$HIDE_PANELS"
export _CAPTURE_TARGETS="$TARGETS_JSON"
export _CAPTURE_PROFILE="$PROFILE"
export _CAPTURE_PRE_PYTHON="$PRE_PYTHON"
export _CAPTURE_INTERACTIVE="$INTERACTIVE"
export _CAPTURE_MCP_PORT="$MCP_PORT"
export _CAPTURE_RELOAD_PORT="$RELOAD_PORT"

# Launch Blender in the background, then immediately hide its window.
# Screenshots still work because the window exists (just not visible).
"$BLENDER" --enable-event-simulate --addons bl_ext.user_default.ppf_contact_solver --python-expr "$PYTHON_SCRIPT" 2>&1 &
BLENDER_PID=$!

# Wait briefly for the window to appear, then hide it
sleep 1
osascript -e 'tell application "System Events" to set visible of process "Blender" to false' 2>/dev/null || true

if [ "$INTERACTIVE" -eq 1 ]; then
    # Interactive mode: print ports and wait. The LLM drives via
    # debug/main.py and is responsible for quitting Blender.
    echo ""
    echo "=== INTERACTIVE MODE ==="
    echo "  Blender PID:  $BLENDER_PID"
    echo "  Reload port:  $RELOAD_PORT"
    echo "  MCP port:     $MCP_PORT"
    echo ""
    echo "Drive via:"
    echo "  python blender_addon/debug/main.py --mcp-port $MCP_PORT exec '<code>'"
    echo "  python blender_addon/debug/main.py --mcp-port $MCP_PORT tools"
    echo ""
    echo "Quit Blender when done:"
    echo "  python blender_addon/debug/main.py --mcp-port $MCP_PORT exec 'bpy.ops.wm.quit_blender()'"
    echo ""
    echo "Waiting for Blender (PID $BLENDER_PID) to exit..."
    wait "$BLENDER_PID" 2>/dev/null || true
    echo "Blender exited."
    exit 0
fi

# --- Non-interactive: wait for Blender to finish (it quits after capture) ---
wait "$BLENDER_PID" 2>/dev/null || true

# --- Host-side annotation (uses Pillow from host Python) --------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$OUTDIR/manifest.json" ]; then
    python3 -c "
import json, sys, os
sys.path.insert(0, os.path.join('${SCRIPT_DIR}', 'ui'))
from capture import Rect, annotate_image

outdir = '$OUTDIR'
scratch = os.path.join(outdir, '.scratch')
with open(os.path.join(outdir, 'manifest.json')) as f:
    data = json.load(f)

panel_bboxes = {k: Rect(*v) for k, v in data.get('panel_bboxes', {}).items()}
results = data.get('results', [])

from PIL import Image as _Image

def _optimize(path):
    '''Palette-quantize + optimize a PNG in place.
    UI screenshots are mostly flat colors, so 32 colors is visually
    lossless for typical addon panels and ~4x smaller than truecolor PNG.
    Keeps full resolution so crops stay crisp.'''
    try:
        src_size = os.path.getsize(path)
        img = _Image.open(path).convert('RGB')
        q = img.quantize(colors=32, method=_Image.Quantize.FASTOCTREE)
        q.save(path, optimize=True)
        dst_size = os.path.getsize(path)
        return src_size, dst_size
    except Exception as e:
        return None, f'optimize failed: {e}'

for r in results:
    label = r.get('draw_string') or r.get('op_idname') or r.get('panel') or '?'
    if r.get('panel_only') and r.get('status') == 'ok':
        # Panel-only: crop the baseline to the panel bbox and save directly
        # (no red-box annotation).
        panel = r['panel']
        crop = panel_bboxes.get(panel)
        if crop is None:
            print(f'  [-] {panel:30s} no crop bbox')
            continue
        baseline = os.path.join(scratch, r.get('baseline', 'baseline.png'))
        out_path = os.path.join(outdir, r['file'])
        _Image.open(baseline).crop(tuple(crop.expand(8))).save(out_path)
        s0, s1 = _optimize(out_path)
        size_note = f'{s0//1024}K->{s1//1024}K' if isinstance(s1, int) else str(s1)
        print(f'  [+] {panel:30s} {r[\"file\"]} ({size_note})')
        continue
    if r.get('status') == 'ok' and r.get('rect') and r.get('file'):
        rect = Rect(*r['rect'])
        panel = r.get('panel', '')
        baseline = os.path.join(scratch, r.get('baseline', 'baseline.png'))
        crop = panel_bboxes.get(panel)
        if crop is not None:
            crop = crop.expand(8)
        else:
            crop = rect.expand(80)
        out_path = os.path.join(outdir, r['file'])
        annotations = [
            {'rect': rect, 'style': 'box', 'padding': 4, 'width': 3},
            {'rect': rect, 'style': 'caption', 'text': label[:30],
             'font_size': 18, 'offset': (0, -22)},
        ]
        annotate_image(baseline, annotations, out_path, crop=crop)
        s0, s1 = _optimize(out_path)
        size_note = f'{s0//1024}K->{s1//1024}K' if isinstance(s1, int) else str(s1)
        print(f'  [+] {label:30s} {r[\"file\"]} ({size_note})')
    else:
        info = r.get('reason', r.get('error', r.get('status', '')))[:60]
        print(f'  [-] {label:30s} {info}')
"
    echo ""
    echo "Output: $OUTDIR"
    echo "CAPTURE_DONE"
else
    echo "Error: no manifest produced" >&2
    exit 1
fi
