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
# PLATFORM: Linux is the supported host for capture. The script starts its
#   OWN Xvfb (searching :200..:229) and sizes Blender to fill it, so a run
#   does not depend on a desktop session being present, does not put a
#   window in front of whoever is using one, and produces the same pixels
#   on a workstation and in CI. Override the size with PPF_BLENDER_WINDOW
#   (WxH, default 1920x1800); set PPF_BLENDER_DISPLAY=inherit to render
#   onto the ambient DISPLAY instead when you want to WATCH a capture.
#
#   Two host packages are required beyond Blender itself:
#     xvfb    — the private display
#     ffmpeg  — reads the finished frame off the X server
#   and Pillow is needed by the host Python that does the cropping and
#   annotation after Blender exits.
#
#   WHY THE FRAME COMES FROM THE X SERVER, NOT FROM BLENDER: Blender's
#   screen.screenshot operator copies the GL front buffer. Under a
#   software rasterizer on a bare Xvfb there is no compositor preserving
#   that buffer across the swap, so it reads back BLACK — and it does so
#   without failing, which is the dangerous part. The blank frame then
#   diffs to nothing and the run reports "this panel is collapsed and has
#   no drawable body", pointing at the panel instead of at the capture.
#   ui/capture.py:screenshot() therefore grabs the display with ffmpeg and
#   rejects a uniform frame outright. PPF_CAPTURE_BACKEND=blender forces
#   the old operator path if you ever need to compare them.
#
#   This is also why the window is sized to the whole framebuffer and left
#   mapped: the X server's copy of the screen is the capture, so anything
#   that hides or shrinks the window removes it from the output.
#
# ============================================================================
set -eo pipefail
# Note: -u is NOT set because TARGETS[@] triggers "unbound variable"
# when the array is empty (common in --interactive and --all modes).

usage() {
    cat << 'EOF'
capture.sh — Capture annotated screenshots of Blender addon UI widgets.

Starts a private Xvfb, launches Blender filling it, locates each requested
widget by pixel-diffing, saves a cropped+annotated PNG per widget, then
quits.  Multiple instances can run in parallel (each picks a free TCP port
and its own display automatically).

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

    --open-closed-panels
                        Re-register DEFAULT_CLOSED panels without that flag
                        so they draw a body.  Needed for Snap and Merge,
                        Utility Tools, Visualization and Object Statistics,
                        which are otherwise captured as empty.  Distinct
                        from --expand, which opens sections WITHIN a panel
                        that is already open.

    --profile PATH[:ENTRY]
                        Load a TOML connection profile so fields are filled.
                        If ENTRY is omitted, the first entry is used.

    --pre-python PATH   Execute a Python file inside Blender after the
                        connection profile is applied, before sidebar setup
                        and capture. Use this to seed scene state (e.g.
                        dyn_params, collider lists) so panels render with
                        realistic content. The script is exec'd with `bpy`
                        already imported in its globals.

    --blend-file PATH   Load a .blend file at Blender startup, BEFORE the
                        capture timer fires. Use this when you need a
                        prepared scene (mesh + addon group state already
                        on disk) instead of building it from scratch in
                        --pre-python. Loading the file via the command
                        line avoids the GUI-mode hang that an in-timer
                        ``bpy.ops.wm.open_mainfile`` triggers.

    --interactive       Launch Blender and stay alive for LLM control.
                        Prints CAPTURE_READY with ports. See notes above
                        about the 5s MCP timeout limitation.

    --sidebar-width N   Logical sidebar width in pixels (default: 560).

    --hide P1,P2,...    Comma-separated extra panel ids to force-hide.

    -h, --help          Show this help and exit.

ENVIRONMENT
    PPF_BLENDER_BIN     Blender binary to use.  Otherwise resolved from
                        PATH and the usual per-OS install locations.

    PPF_BLENDER_WINDOW  Capture size as WxH (default 1920x1800).  Sets both
                        the Xvfb framebuffer and Blender's window, which are
                        kept equal so the UI fills the frame.

    PPF_BLENDER_DISPLAY inherit — render onto the ambient DISPLAY instead of
                        a private Xvfb.  Use when you want to watch a run.

    PPF_CAPTURE_BACKEND blender — take frames with Blender's screen.screenshot
                        operator instead of reading the X server.  Black under
                        software GL; kept for comparison only.

REQUIREMENTS (Linux)
    xvfb and ffmpeg on PATH, plus Pillow importable by the host python3
    (used for cropping and annotation after Blender exits).

AVAILABLE PANELS
    MAIN_PT_RemotePanel              "Backend Communicator"
    SSH_PT_SolverPanel               "Solver"
    SSH_PT_ObjectGroupsManager       "Scene Configuration"
    DYNAMICS_PT_Groups               "Dynamics Groups"
    SNAPMERGE_PT_SnapAndMerge        "Snap and Merge"          (closed)
    UTILITY_PT_UtilityTools          "Utility Tools"           (closed)
    VISUALIZATION_PT_Visualization   "Visualization"           (closed)
    STATISTICS_PT_Statistics         "Object Statistics"       (closed)

    "(closed)" marks bl_options={'DEFAULT_CLOSED'}: the panel draws no body
    until something opens it, so a plain run captures nothing and reports it
    as having no drawable body.  --open-closed-panels re-registers those
    classes without the flag so they come up expanded.

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

# Resolve a Blender binary for the running OS, honoring the same
# PPF_BLENDER_BIN override the rig and launch.sh honor, so one export
# drives all three. Hardcoding the macOS bundle path made this script
# refuse to start anywhere else, which is why the docs screenshots could
# only ever be regenerated from a Mac.
find_blender() {
    if [ -n "${PPF_BLENDER_BIN:-}" ] && [ -x "${PPF_BLENDER_BIN}" ]; then
        echo "$PPF_BLENDER_BIN"
        return 0
    fi
    case "$(uname -s)" in
    Darwin)
        local bundle="/Applications/Blender.app/Contents/MacOS/Blender"
        [ -x "$bundle" ] && { echo "$bundle"; return 0; }
        command -v blender 2>/dev/null && return 0
        ;;
    Linux)
        command -v blender 2>/dev/null && return 0
        local candidate
        for candidate in $(ls -d /opt/blender-*/blender \
                                 /usr/local/blender-*/blender \
                                 /opt/blender/blender 2>/dev/null | sort -rV); do
            [ -x "$candidate" ] && { echo "$candidate"; return 0; }
        done
        ;;
    esac
    return 1
}

BLENDER="$(find_blender)" || {
    echo "Error: no Blender found for $(uname -s)." >&2
    echo "       Set PPF_BLENDER_BIN, or install Blender (./install-blender.sh)." >&2
    exit 1
}

# Check the host tools up front. Every one of these is needed only AFTER
# Blender has run — the display before it starts, ffmpeg per frame, Pillow
# in the annotation pass at the very end — so without this check the
# common failure is a clean two-minute capture that dies on the last step
# with an ImportError and leaves no usable PNG.
missing=""
if [ "$(uname -s)" = Linux ] && [ "${PPF_BLENDER_DISPLAY:-}" != inherit ]; then
    command -v Xvfb >/dev/null 2>&1 || missing="$missing xvfb"
fi
if [ "${PPF_CAPTURE_BACKEND:-}" != blender ] && [ "$(uname -s)" = Linux ]; then
    command -v ffmpeg >/dev/null 2>&1 || missing="$missing ffmpeg"
fi
python3 -c "import PIL" >/dev/null 2>&1 || missing="$missing python3-pillow"
if [ -n "$missing" ]; then
    echo "Error: capture.sh needs:$missing" >&2
    echo "       apt-get install xvfb ffmpeg   # and: pip install pillow" >&2
    echo "       (Pillow must be importable by the python3 on PATH, which" >&2
    echo "        does the cropping and annotation after Blender exits.)" >&2
    exit 1
fi

# --- Parse arguments --------------------------------------------------------
OUTDIR=""
EXPAND=0
INTERACTIVE=0
PANEL_ONLY=0
OPEN_CLOSED=0
SIDEBAR_WIDTH=560
HIDE_PANELS=""
PROFILE=""
PRE_PYTHON=""
BLEND_FILE=""
TARGETS=()
ALL_PANELS=()  # Array — multiple --all flags allowed

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)    usage ;;
        -o)           OUTDIR="$2"; shift 2 ;;
        --expand)     EXPAND=1; shift ;;
        --all)        ALL_PANELS+=("$2"); shift 2 ;;
        --panel-only) PANEL_ONLY=1; shift ;;
        --open-closed-panels) OPEN_CLOSED=1; shift ;;
        --sidebar-width) SIDEBAR_WIDTH="$2"; shift 2 ;;
        --hide)       HIDE_PANELS="$2"; shift 2 ;;
        --profile)    PROFILE="$2"; shift 2 ;;
        --pre-python) PRE_PYTHON="$2"; shift 2 ;;
        --blend-file) BLEND_FILE="$2"; shift 2 ;;
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
OPEN_CLOSED = os.environ.get("_CAPTURE_OPEN_CLOSED", "0") == "1"
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
    # __file__ is part of the contract: a prep script that lives beside
    # sibling helpers (docs/tools/ has several) can only find them by its
    # own path, and exec() does not supply one.
    exec(compile(src, PRE_PYTHON, "exec"),
         {"bpy": bpy, "__name__": "__main__",
          "__file__": os.path.abspath(PRE_PYTHON)})
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

def _open_closed_panels(panel_ids):
    """Re-register the named panels without their DEFAULT_CLOSED flag.

    A panel's open/closed state is decided when the region first builds
    its panel list, and DEFAULT_CLOSED means it builds closed and draws no
    body. Clearing bl_options on the live class is not enough — the region
    has already made its decision — so the class is unregistered and
    registered again, which makes the region rebuild it fresh.

    bl_order is preserved across the round trip, so panels keep their
    place in the sidebar rather than jumping to the bottom.
    """
    for pid in sorted(set(panel_ids)):
        cls = getattr(bpy.types, pid, None)
        if cls is None:
            print(f"capture.sh: no panel {pid} to open")
            continue
        opts = set(getattr(cls, "bl_options", None) or set())
        if "DEFAULT_CLOSED" not in opts:
            continue
        opts.discard("DEFAULT_CLOSED")
        try:
            bpy.utils.unregister_class(cls)
            cls.bl_options = opts
            bpy.utils.register_class(cls)
            print(f"capture.sh: opened {pid}")
        except Exception as e:
            print(f"capture.sh: could not open {pid}: {e}")


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

    # Switch to addon tab. The redraw above matters: a UI region that has
    # not been laid out yet carries no panel categories, and assigning one
    # that the region does not know about raises rather than being ignored.
    cap.force_redraw(iterations=2)
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
    if OPEN_CLOSED:
        wanted = list(ALL_PANELS) + [t.split(":", 1)[0] for t in TARGETS if ":" in t]
        _open_closed_panels(wanted)
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

# --- Display and window size ------------------------------------------------
# This script reads pixels back off the screen, so the display is part of
# the OUTPUT, not just somewhere to open a window. Two things follow.
#
# First, the run owns its display. Rendering onto whatever DISPLAY happens
# to be exported puts the capture at the mercy of the desktop behind it —
# its resolution, its DPI, a screensaver, another window overlapping the
# one being read — and makes the same command produce different PNGs on a
# laptop and on a CI runner. A private Xvfb makes the geometry a property
# of the run. Set PPF_BLENDER_DISPLAY=inherit to use the ambient DISPLAY
# instead, which is what to reach for when you want to WATCH the capture.
# The search starts at :200 to stay clear of a desktop session (:0), of
# the :99 that CI and install-blender.sh use, and of the :100.. range the
# test rig searches (debug/blender_harness.py), so a capture can run
# alongside all three.
#
# Second, Blender is told to fill that display. Without a window manager
# there is nothing to maximize a window, and Blender left to itself opens
# at whatever size its userpref remembers, so panels get cropped narrower
# on one machine than another. Sizing the window to the framebuffer makes
# the sidebar render at full height and keeps crops crisp.
# 1800 rows rather than a monitor-shaped 1080: the sidebar is the subject,
# and a panel taller than the region simply stops being drawn at the
# bottom. That loss is invisible in the result — the crop ends on a clean
# widget boundary and looks like the whole panel — so the default buys
# headroom for the longest panel instead of leaving each caller to notice.
CAPTURE_GEOMETRY="${PPF_BLENDER_WINDOW:-1920x1800}"
case "$CAPTURE_GEOMETRY" in
    *x*) CAPTURE_W="${CAPTURE_GEOMETRY%x*}"; CAPTURE_H="${CAPTURE_GEOMETRY#*x}" ;;
    *)   echo "Error: PPF_BLENDER_WINDOW='$CAPTURE_GEOMETRY' is not WxH" >&2; exit 1 ;;
esac

XVFB_PID=""
cleanup_display() {
    [ -n "$XVFB_PID" ] && kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup_display EXIT

if [ "$(uname -s)" = Linux ] && \
   [ "${PPF_BLENDER_DISPLAY:-}" != inherit ] && command -v Xvfb >/dev/null 2>&1; then
    for _n in $(seq 200 229); do
        [ -e "/tmp/.X11-unix/X$_n" ] && continue
        Xvfb ":$_n" -screen 0 "${CAPTURE_W}x${CAPTURE_H}x24" -nolisten tcp \
            >/dev/null 2>&1 &
        XVFB_PID=$!
        # Xvfb either binds the display or exits; give it a moment, then
        # confirm the socket exists rather than trusting the spawn.
        for _t in 1 2 3 4 5 6 7 8 9 10; do
            [ -e "/tmp/.X11-unix/X$_n" ] && break
            sleep 0.5
        done
        if [ -e "/tmp/.X11-unix/X$_n" ] && kill -0 "$XVFB_PID" 2>/dev/null; then
            export DISPLAY=":$_n"
            echo "capture.sh: private display $DISPLAY at ${CAPTURE_W}x${CAPTURE_H}"
            break
        fi
        kill "$XVFB_PID" 2>/dev/null || true
        XVFB_PID=""
    done
    if [ -z "$XVFB_PID" ]; then
        echo "Error: could not start Xvfb on any display in :200..:229" >&2
        exit 1
    fi
elif [ "$(uname -s)" = Linux ] && [ -z "${DISPLAY:-}" ]; then
    echo "Error: no DISPLAY and Xvfb is not installed (apt-get install xvfb)." >&2
    exit 1
fi

WINDOW_ARGS=(--window-geometry 0 0 "$CAPTURE_W" "$CAPTURE_H" --no-window-focus)

# --- Launch Blender (hidden window) -----------------------------------------
export _CAPTURE_OUTDIR="$OUTDIR"
export _CAPTURE_EXPAND="$EXPAND"
export _CAPTURE_PANEL_ONLY="$PANEL_ONLY"
export _CAPTURE_OPEN_CLOSED="$OPEN_CLOSED"
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
#
# A --blend-file goes BEFORE the option flags so Blender opens it on the
# command line. We deliberately avoid ``bpy.ops.wm.open_mainfile`` from a
# pre-python script: in GUI mode, replacing the active blend inside a
# timer callback invalidates the window-manager that owns the timer and
# wedges the main event loop.
BLEND_ARGS=()
if [ -n "$BLEND_FILE" ]; then
    if [ ! -f "$BLEND_FILE" ]; then
        echo "Error: --blend-file '$BLEND_FILE' does not exist" >&2
        exit 1
    fi
    BLEND_ARGS+=("$BLEND_FILE")
fi
"$BLENDER" "${WINDOW_ARGS[@]}" "${BLEND_ARGS[@]}" --enable-event-simulate --addons bl_ext.user_default.ppf_contact_solver --python-expr "$PYTHON_SCRIPT" 2>&1 &
BLENDER_PID=$!

# Give the window a moment to map before anything tries to read it. On a
# private Xvfb there is nobody to hide it from, which is the point: the
# frame the X server holds IS the capture, so the window must stay mapped
# and on top rather than being pushed out of sight.
sleep 1

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
