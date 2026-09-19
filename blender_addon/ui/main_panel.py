# File: main_panel.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Main panel (MAIN_PT_RemotePanel), GlobalStateWatcher, and aggregated
# operator registration.  Extracted from ui/client.py.

import os
import textwrap
from dataclasses import dataclass

import bpy  # pyright: ignore
from bpy.types import Panel  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_

from ..core.client import RemoteStatus
from ..core.client import communicator as com
from ..core.status import crash_cause_summary
from ..core.derived import (
    is_server_busy_from_response as is_running,
    is_sim_running_from_response as is_simulating,
)
from ..models.groups import get_addon_data, has_addon_data
from ..core.module import cbor2_available, get_install_error_message, get_install_result, get_installing_status, module_exists
from ..core.reload_server import get_reload_server_status
from ..core.utils import (
    WINDOWS_MAX_PATH,
    find_invalid_name_char,
    find_invalid_path_char,
    find_shell_unsafe_path_char,
    get_category_name,
    resolve_local_path,
    windows_long_paths_enabled,
    windows_path_too_long,
)

from .connection_ops import (
    REMOTE_OT_Abort,
    REMOTE_OT_CancelConnect,
    REMOTE_OT_CancelStartServer,
    REMOTE_OT_Connect,
    REMOTE_OT_Disconnect,
    REMOTE_OT_OpenProfile,
    REMOTE_OT_RefreshGpuDevices,
    REMOTE_OT_StartServer,
    REMOTE_OT_StopServer,
    SOLVER_OT_ForceTerminatePort,
    is_local_server_type,
    force_terminate_status,
    NATIVE_PATH_FIELDS,
    REMOTE_SERVER_TYPES,
    SERVER_TYPE_BACKENDS,
    classes as connection_classes,
)
from .install_ops import (
    REMOTE_OT_InstallCbor2,
    REMOTE_OT_InstallDocker,
    REMOTE_OT_InstallParamiko,
    classes as install_classes,
)
from .mcp_ops import classes as mcp_classes
from .debug_ops import (
    DEBUG_OT_BrowseLogPath,
    DEBUG_OT_ClearLogPath,
    DEBUG_OT_Compile,
    DEBUG_OT_DataReceive,
    DEBUG_OT_DataSend,
    DEBUG_OT_DeleteLog,
    DEBUG_OT_ExecuteServer,
    DEBUG_OT_ExecuteShell,
    DEBUG_OT_GitPull,
    DEBUG_OT_GitPullLocal,
    DEBUG_OT_RenderAnimation,
    DEBUG_OT_StopRender,
    is_render_anim_running,
    get_render_anim_progress,
    WM_OT_OpenGitHubLink,
    classes as debug_classes,
)
from .addon_ops import classes as addon_classes
from .jupyter_ops import classes as jupyter_classes
from .solver_control_ops import (
    SOLVER_OT_OpenSessionFolder,
    SOLVER_OT_SaveAndQuit,
    SOLVER_OT_ShowConsole,
    SOLVER_OT_Terminate,
    SOLVER_OT_UpdateStatus,
    classes as solver_control_classes,
)
from .geometry_cleanup_ops import (
    MESH_OT_RemoveIsolatedVertices,
    MESH_OT_TriangulateDegenerateFaces,
    classes as geometry_cleanup_classes,
)


# Tiny TTL cache for the port-error probe in the panel draw. The panel
# can redraw many times per second; a TCP probe on every paint would be
# wasteful even at sub-millisecond cost. ``_PROBE_TTL_S`` keeps the
# answer fresh enough that the user sees the stale-error suppression
# kick in within a frame or two of attaching.
#
# Only one port is ever in play at a time (the one named in the current
# error), so a single (port, timestamp, ours) tuple caps the cache at one
# entry instead of accumulating a dict key per distinct port hit.
_PROBE_TTL_S = 1.5

# Character budget for the crash detail row. A panel label renders one line
# and Blender clips whatever does not fit without marking the cut, so the
# ellipsis has to be added here for the reader to know the line continues in
# the Console.
_CRASH_DETAIL_CHARS = 96
_probe_cache: tuple[int, float, bool] | None = None


# Hardware keys the panel reads but does not list. "GPU Index" is the
# machine-readable half of the "GPU" row, which already leads with the same
# number, so listing it would print that number twice; it stays on the wire
# because the GPU-selection check compares against it rather than parsing a
# display string.
_UNDISPLAYED_HARDWARE_KEYS = frozenset({"GPU Index"})


# The server reports frame indices 0-based; Blender's timeline is 1-based.
# This offset documents the convention in one place for the display-side
# remaps below.
REMOTE_FRAME_OFFSET = 1


def remote_frame_to_blender(value, *, grouped=False) -> str:
    """Convert a remote 0-based frame index into a Blender 1-based frame for
    display. Strips thousands separators before parsing so the grouped
    ``Total Frames`` string is tolerated, adds ``REMOTE_FRAME_OFFSET``, and
    returns the result thousands-grouped when *grouped* is set. A value that
    isn't an integer is returned unchanged (coerced to ``str``).
    """
    try:
        n = int(str(value).replace(",", "")) + REMOTE_FRAME_OFFSET
    except (ValueError, TypeError):
        return str(value)
    return f"{n:,}" if grouped else str(n)


def _our_server_responding_in_error(error_msg: str) -> bool:
    """True when the panel's port-in-use error names a port that now
    answers a ppf-cts-server TCMD ping. Used to suppress the stale
    error + Force Terminate Process button after the spawn path's attach branch
    takes over.
    """
    import re
    import time

    global _probe_cache
    m = re.search(r"\bPort\s+(\d+)", error_msg)
    if not m:
        return False
    port = int(m.group(1))
    now = time.monotonic()
    if _probe_cache and _probe_cache[0] == port and (now - _probe_cache[1]) < _PROBE_TTL_S:
        return _probe_cache[2]
    from ..core.connection import _probe_ppf_cts_server
    ours = _probe_ppf_cts_server(port, timeout=0.5)
    _probe_cache = (port, now, ours)
    return ours


def _draw_error_lines(layout, error: str) -> None:
    """Draw a connection error across as many label rows as it needs.

    A single label renders one line and Blender clips the overflow without
    marking the cut, which is the same limit ``_CRASH_DETAIL_CHARS`` exists
    for. A connection refusal is not a summary with a fuller report behind it
    though: it IS the whole report, and the part that gets clipped is the part
    that says what to do (which folder to pick, which flag to add). So it is
    wrapped rather than truncated, and the icon goes on the first row so the
    block reads as one error.
    """
    lines = textwrap.wrap(error, width=_CRASH_DETAIL_CHARS) or [error]
    for i, line in enumerate(lines):
        layout.label(text=line, icon="ERROR" if i == 0 else "BLANK1")


def _crash_detail_line(server_error: str, crash_kind: str) -> str:
    """The solver's own one-line detail out of a rendered crash report.

    The report's first line is ``"<summary>: <detail>"``, and the panel draws
    the summary separately as a localized headline, so only the part after the
    colon is left here. When the report has no detail (an empty ``detail``
    field), the split yields nothing and no second row is drawn.

    The line is truncated to a panel-width budget, because a label renders one
    line and Blender clips the overflow with no indication that it did. The
    Console keeps the untruncated report.
    """
    first_line = server_error.split("\n", 1)[0]
    summary = crash_cause_summary(crash_kind)
    prefix = f"{summary}: "
    detail = first_line[len(prefix):] if first_line.startswith(prefix) else first_line
    detail = detail.strip()
    if len(detail) > _CRASH_DETAIL_CHARS:
        detail = detail[: _CRASH_DETAIL_CHARS - 1].rstrip() + "…"
    return detail


def _draw_path_warning(layout, path, *, shell_bound: bool = True) -> bool:
    """Draw a one-line warning when *path* holds a character its backend
    cannot carry, and return ``True`` so the caller can skip any follow-up
    status line.  Draws nothing and returns ``False`` for a valid (or blank)
    path.

    ``shell_bound`` is what the path's backend does with it. A REMOTE path is
    interpolated into a shell command on the solver host, so it refuses
    whitespace as well as metacharacters. A native solver root, Windows or
    macOS, is only ever an ``os.path.join`` base and a ``subprocess.Popen``
    ``cwd``, so it refuses metacharacters alone; each caller passes what is true of its own
    backend, and the Connect gate reads the matching predicate.
    """
    bad = (find_invalid_path_char(path) if shell_bound
           else find_shell_unsafe_path_char(path))
    if bad is None:
        return False
    layout.label(
        text=("Path should not contain spaces or special characters"
              if shell_bound else "Path should not contain special characters"),
        icon="ERROR",
    )
    return True


def _draw_native_gpu_backend(layout, props, root, builds) -> None:
    """Draw the GPU Backend selector where the root offers a choice of one.

    DRAWN ONLY WHERE THERE IS SOMETHING TO SAY, unlike the Compute Device row
    above it: a Windows x64 distribution carries CUDA and ROCm together and the
    artist picks between them, while a macOS root has one accelerator and a
    CUDA-or-ROCm row there would be a control that can never mean anything. It
    is also drawn when the saved choice names a backend this root does not
    hold, so a `.blend` carrying that choice can be corrected rather than
    silently ignored.

    WHICH BACKENDS ARE OFFERED IS A FACT ABOUT THE DISK, as with the device
    row. Whether one has a usable GPU is a question for the solver, asked when
    the server is spawned; asking it here would run a solver on every redraw.
    """
    from ..core.connection import GPU_BACKEND_AUTO, native_gpu_backend_choice_open

    found = builds(root)
    named = [name for name in found if name]
    selected = props.native_gpu_backend
    missing = selected != GPU_BACKEND_AUTO and selected.lower() not in named
    if len(named) < 2 and not missing:
        return
    row = layout.row(align=True)
    row.enabled = native_gpu_backend_choice_open(found, selected)
    row.prop(props, "native_gpu_backend", text=iface_("GPU Backend"))
    if missing:
        layout.label(
            text=iface_("This folder has no {} build").format(selected),
            icon="ERROR",
        )


def _draw_native_device(layout, props, root, resolver, builds=None) -> None:
    """Draw the GPU/CPU selector for a local native backend, ALWAYS.

    DRAWN EVEN WHEN ONLY ONE DEVICE IS BUILT, disabled with a line saying why,
    rather than hidden. A control that disappears when its precondition is
    unmet costs the artist the discoverability of the feature: they cannot tell
    "this build has no CPU solver" from "this add-on cannot do that".

    WHICH DEVICES ARE OFFERED IS A FACT ABOUT THE DISK. The backend is chosen
    when the solver is BUILT, so a device is available exactly when a server
    binary for it exists under the selected root. A machine with a GPU whose
    tree has only ever built the CPU backend can honestly offer only CPU, and
    the reverse is the common case: an ordinary GPU checkout has no CPU build
    until someone asks for one.
    """
    from ..core.connection import DEVICE_CPU, DEVICE_GPU, native_device_choice_open

    root = resolve_local_path(root or "").rstrip("/\\")
    have = {device: bool(resolver(root, device)) for device in (DEVICE_GPU, DEVICE_CPU)}
    row = layout.row(align=True)
    # Only a root that holds BOTH gives the artist a choice to make; with one,
    # the property cannot be moved onto something that is not there. It stays
    # open while the selection names the ABSENT build, because the property
    # defaults to GPU and a folder holding only the CPU build would otherwise
    # lock it on a device Connect can only refuse.
    row.enabled = native_device_choice_open(have, props.native_device)
    row.prop(props, "native_device", expand=True)
    if builds is not None and props.native_device == DEVICE_GPU:
        _draw_native_gpu_backend(layout, props, root, builds)
    if not any(have.values()):
        # The path warning above already said the root is wrong; adding a
        # second complaint here would be noise.
        return
    if not have[DEVICE_CPU]:
        # THE COMMAND IS SPELLED WITH ITS TARGET DIRECTORY, because a bare
        # `--features cpu` does not produce a build this panel can offer
        # ALONGSIDE the GPU one, and reaching this branch means a GPU build is
        # here (a root holding neither returned above). Every backend links the
        # same executable name, so `build.rs` refuses to put the CPU build into
        # a `target/release` that already holds the GPU one and names this
        # variable in its own refusal. Naming only the feature flag sent the
        # artist to a command that either stops at that refusal or, on a
        # cleaned tree, replaces the GPU build they still want.
        layout.label(text=iface_("CPU build not found. To add one:"), icon="INFO")
        layout.label(
            text="CARGO_TARGET_DIR=target/cpu cargo build --release --features cpu"
        )
    elif not have[DEVICE_GPU]:
        if props.native_device == DEVICE_GPU:
            # The selection names the absent build, which is where a CPU-only
            # folder starts; the selector above is open for exactly this, so
            # the line says what to pick.
            layout.label(
                text=iface_("GPU build not found under this root. Choose CPU to run the CPU build."),
                icon="ERROR",
            )
        else:
            layout.label(text=iface_("GPU build not found under this root"), icon="INFO")
    # NO LINE FOR A CPU SELECTION THAT IS SIMPLY WORKING. The two branches
    # above report a build that is ABSENT, which the artist has to act on; a
    # note that the backend they deliberately picked is the slower one reports
    # nothing they did not just decide, and it sat under the selector on every
    # redraw for the whole session.


def _draw_native_status(layout, server_type, path) -> None:
    """Draw the native solver-path validity line(s) for *path*.

    Resolves *path* to the real solver root (walking up from a selected
    subdirectory such as ``target/release``, ``target/cuda/release``, ``bin``,
    or an embedded ``python`` folder), then draws a CHECKMARK when a root is
    found, adding a second line naming the resolved root when it differs from
    what the user selected, or an ERROR when no ancestor holds a server.
    No-op for a blank path.

    ONE DRAWER FOR THE THREE NATIVES. What differs between them is the resolver
    and the executable's name, and both come from the backend's own entry in
    `core.connection`, so neither is spelled here.
    """
    from ..core.connection import native_resolvers

    backend_type = SERVER_TYPE_BACKENDS[server_type]
    path = resolve_local_path(path or "").rstrip("/\\")
    if not path:
        return
    resolve_root = native_resolvers(backend_type)[3]
    resolved = resolve_root(path)
    if resolved is None:
        exe = "ppf-cts-server.exe" if backend_type == "win_native" else "ppf-cts-server"
        layout.label(text=f"{exe} not found", icon="ERROR")
        return
    layout.label(text="Solver path valid", icon="CHECKMARK")
    if os.path.normpath(resolved) != os.path.normpath(path):
        layout.label(text=iface_("Using solver root: {path}").format(path=resolved))


def _draw_remote_device(layout, props) -> None:
    """Draw the Compute Device and GPU Backend rows for a REMOTE connection.

    WHY THIS IS NOT `_draw_native_device`. That one answers from the local
    filesystem on every redraw, which it can afford because the builds are on
    this machine. Here they are on the solver host, so the answer comes from
    the listing `core.remote_builds` took once over the connection, and the
    rows exist only while there is a connection to have taken it.

    DRAWN AFTER CONNECT, NOT BEFORE, which is the opposite of the native rows
    and follows from the same fact. Before connecting there is nothing to ask,
    so a row here would offer a choice against no information; the GPU picker
    directly below has always worked this way for the same reason. The choice
    reaches the server at Start Server, so the rows are disabled once one is
    running: Stop Server, pick, Start Server is how a running solver is moved,
    exactly as it is for the GPU.

    THE SELECTION IS READ AT START SERVER, not held from connect, which is what
    makes these rows mean anything: they are drawn only once a connection is up,
    so a value captured at connect would leave a control the artist can move and
    that changes no run. `REMOTE_OT_StartServer` passes what they hold, exactly
    as it passes the GPU index.
    """
    from ..core import remote_builds
    from ..core.connection import (
        DEVICE_CPU,
        DEVICE_GPU,
        native_device_choice_open,
        remote_server_binary,
    )

    if props.server_type not in REMOTE_SERVER_TYPES:
        return
    if not com.is_connected():
        return

    col = layout.column()
    col.enabled = not com.is_server_running() and not com.is_server_launching()

    error = remote_builds.probe_error()
    if error:
        # THE FAILED PROBE GETS A LINE, and the rows still get drawn. Which
        # builds the host has is unknown, not known to be none, and hiding the
        # rows would say the second. The launch refuses by name if the choice
        # cannot be served, and the reason here is what the artist acts on.
        col.prop(props, "native_device", expand=True)
        col.label(text=error, icon="ERROR")
        return
    if not remote_builds.has_probed():
        return

    # THE ROOT THE PROBE RECORDED, never one derived here. The listing's keys
    # are absolute directories under the root the probe was pointed at, which is
    # the backend's `current_directory`, the SOLVER root. Deriving one here from
    # `normalized_remote_root()` gets the DATA root
    # (`<share>/ppf-cts/git-<branch>/<project>`) instead, and every lookup then
    # misses, which is indistinguishable from a host holding no build: the rows
    # go dead and the panel says "No solver build found under ..." while the
    # listing in `remote_builds` holds both. That shipped, on every remote
    # transport at once, because this is the only place the two spellings could
    # differ and nothing compared them.
    root = remote_builds.probed_root()
    listing = remote_builds.cached_builds()
    have = {
        device: bool(remote_server_binary(root, listing, device))
        for device in (DEVICE_GPU, DEVICE_CPU)
    }
    row = col.row(align=True)
    row.enabled = native_device_choice_open(have, props.native_device)
    row.prop(props, "native_device", expand=True)

    if props.native_device == DEVICE_GPU:
        _draw_remote_gpu_backend(col, props, root, listing)

    if not any(have.values()):
        col.label(
            text=iface_("No solver build found under {path}").format(path=root),
            icon="ERROR",
        )
        return
    if not have[props.native_device]:
        other = DEVICE_CPU if props.native_device == DEVICE_GPU else DEVICE_GPU
        col.label(
            text=iface_("The solver host has no {device} build here; it has {other}").format(
                device=props.native_device, other=other
            ),
            icon="ERROR",
        )


def _draw_remote_gpu_backend(layout, props, root, listing) -> None:
    """Draw the GPU Backend row where a remote root offers a choice of one.

    The same rule the native row follows: drawn only where there is something
    to say, which is a host holding more than one GPU build, or a saved choice
    that host does not have.
    """
    from ..core.connection import (
        GPU_BACKEND_AUTO,
        native_gpu_backend_choice_open,
        remote_gpu_builds,
    )

    found = remote_gpu_builds(root, listing)
    named = [name for name in found if name]
    selected = props.native_gpu_backend
    missing = selected != GPU_BACKEND_AUTO and selected.lower() not in named
    if len(named) < 2 and not missing:
        return
    row = layout.row(align=True)
    row.enabled = native_gpu_backend_choice_open(found, selected)
    row.prop(props, "native_gpu_backend", text=iface_("GPU Backend"))
    if missing:
        layout.label(
            text=iface_("The solver host has no {} build").format(selected),
            icon="ERROR",
        )


def _draw_gpu_section(layout, props) -> None:
    """Draw the GPU picker and its outcome line, or nothing when disconnected.

    This is the one row in the Connection box that its precondition HIDES
    rather than disables. The device list is read from the solver host over the
    connection, so with no connection there is nothing to offer, and a dropdown
    holding only Automatic would say nothing the Connect button directly above
    it does not already say.

    Disabled while the server is up, because the choice is applied at Start
    Server: Stop Server, pick, Start Server is how a running solver is moved.
    """
    from ..core.connection import DEVICE_CPU

    # The macOS native solver runs on the system default Metal device, so there
    # is nothing to pick and the dropdown has nothing to offer. The confirmation
    # line below gates itself on the backend the SERVER reports, which is the
    # answer that also holds for a Metal or CPU server reached some other way;
    # this return is about the PICKER.
    if props.server_type == "MAC_NATIVE":
        return
    # A CPU RUN HAS NO GPU TO PICK. The dropdown would offer the solver host's
    # cards for a server that opens none of them, and the launch would write a
    # CUDA_VISIBLE_DEVICES in front of a binary with no CUDA in it. The
    # confirmation line below says nothing for a non-CUDA server already, so
    # what is left without this is a control that means nothing.
    if props.native_device == DEVICE_CPU:
        return
    if not com.is_connected():
        return
    col = layout.column()
    col.enabled = not com.is_server_running() and not com.is_server_launching()
    _draw_gpu_picker(col, props)
    _draw_gpu_confirmation(layout, props)


def _draw_gpu_picker(layout, props) -> None:
    """Draw the GPU dropdown and its refresh button, and nothing else.

    The selected entry already names the device, and a selection the host
    cannot satisfy already reads ``<index>: not detected`` with an error icon,
    so restating either underneath would only repeat the row above. The one
    thing the dropdown cannot express is that enumeration failed outright,
    since that leaves it holding just Automatic, so that gets a line.
    """
    from ..core.gpu_devices import gpu_probe_error

    row = layout.row(align=True)
    row.prop(props, "solver_gpu")
    row.operator(REMOTE_OT_RefreshGpuDevices.bl_idname, text="", icon="FILE_REFRESH")

    probe_error = gpu_probe_error()
    if probe_error:
        layout.label(text=probe_error, icon="ERROR")


def _draw_gpu_confirmation(layout, props) -> None:
    """Say so when the server is not on the GPU that was picked, and nothing
    otherwise.

    The GPU the server is actually on is named by the Remote Hardware block, so
    this row exists only for the cases that block cannot express: the two
    disagreeing, which happens when Start Server attached to a server it did
    not launch, and a server too old to report its device at all, where the
    comparison cannot run and silence would read as agreement.
    """
    from ..core.gpu_devices import AUTOMATIC

    hardware = com.response.get("hardware") or {}
    if not hardware:
        return
    # EVERY LINE BELOW IS A CUDA QUESTION, so a server that is not on CUDA is
    # not asked it. Metal opens the system default device and offers no way to
    # name another, and a CPU build has no device at all, so neither reports a
    # GPU index and neither has anything for this to confirm. Read from the
    # server's own "Backend" rather than from the add-on's connection type:
    # the two agree only for a server this add-on launched itself, and an SSH
    # connection to a Mac is exactly where they do not.
    backend = str(hardware.get("Backend", "cuda"))
    if backend != "cuda":
        return
    reported = hardware.get("GPU Index")
    if reported is None:
        layout.label(text="Server does not report which GPU it is on", icon="QUESTION")
        return
    reported = int(reported)
    name = str(hardware.get("GPU", ""))
    if reported < 0:
        alert = layout.column(align=True)
        alert.alert = True
        alert.label(text=name or "Server resolved no CUDA device", icon="ERROR")
        return
    selected = props.solver_gpu_index
    selected_uuid = props.solver_gpu_uuid
    from ..core.gpu_devices import find_device_by_uuid
    selected_device = find_device_by_uuid(selected_uuid)
    if selected_device is not None:
        selected = selected_device.index
    if selected in (AUTOMATIC, reported):
        # Agreement needs no line of its own: the Remote Hardware GPU row names
        # the device the server is on, index included, and repeating it here
        # would say the same thing twice.
        return
    # Reached when the add-on attached to a server it did not launch. Stop
    # Server then Start Server relaunches it on the selection, which the
    # backend still holds.
    alert = layout.column(align=True)
    alert.alert = True
    alert.label(
        text=iface_("Solver is on GPU {actual}, not the selected GPU {wanted}").format(
            actual=reported, wanted=selected
        ),
        icon="ERROR",
    )
    alert.label(text="Press Stop Server, then Start Server, to move it")


def _draw_force_terminate(layout, props, context) -> None:
    """Draw the Force Terminate Process button and its status line.

    Drawn only in the states ``_force_terminate_offered`` names, where a
    server process has to be ended and Stop Server cannot reach it. It is
    enabled when no job is running, and carries a three-branch line under
    it: a server of ours is listening on the port, nothing is, or the port
    is held by another program. That line is what tells the artist whether
    the kill has anything to act on before they press it.

    FOR A REMOTE TYPE (SSH, Docker, Docker over SSH) THE ROW IS DRAWN ONLY
    WHILE CONNECTED. This is a deliberate departure from drawing every
    conditional button disabled with a status line: that rule is for a
    feature awaiting a precondition the artist can meet, and a disconnected
    remote kill has NO transport to act through, so a disabled button there
    would offer an action that cannot exist in that state. Connected, the
    kill runs through the live backend and leaves the server stopped, so
    Start Server is the next step.
    """
    if not is_local_server_type(props.server_type) and not com.is_connected():
        return
    row = layout.row()
    row.enabled = SOLVER_OT_ForceTerminatePort.poll(context)
    row.operator(SOLVER_OT_ForceTerminatePort.bl_idname, icon="X")
    text, icon = force_terminate_status(props)
    layout.label(text=text, icon=icon)


def _names_port_in_use(err_lower: str) -> bool:
    """True when a lowercased error says a port is taken.

    Covers ``PortInUseByForeignProcess``'s ``"Port N is in use"`` and the
    remote launch's ``"Server port N is already in use on the remote host"``.
    The ``"already running on port N"`` refusal is deliberately NOT one of
    them, because it is the one that must not be probed for staleness: there
    our own server answering IS the report.
    """
    return "in use" in err_lower and "port" in err_lower


def _force_terminate_offered(error: str, *, stale_port_error: bool) -> bool:
    """True when the panel is reporting a failure whose way out is Force
    Terminate Process, which is the only time its row is drawn.

    THREE STATES, AND THEY ARE THE WHOLE LIST:

    - A PROTOCOL VERSION MISMATCH, read off ``RemoteStatus`` rather than off
      the message because two of the malformed-response paths set the state
      with no error text at all. A server orphaned from an earlier binary
      keeps serving its old ``PROTOCOL_VERSION`` across the update the artist
      just made, so the restart has to end the process, not the image.
    - ANY ERROR WHILE NOT CONNECTED, which is what a refused or lost
      connection leaves behind. Stop Server is reachable only through a
      connection, so there the button is the only way to end a server
      process, whatever the refusal says.
    - AN ERROR NAMING A HELD PORT, connected or not: ``"Port N is in use"``,
      ``"A solver server is already running on port N"``
      (``NativeServerMismatch``), and the remote launch's ``"Server port N is
      already in use on the remote host"``, which a CONNECTED Start Server
      raises and the connected kill clears through the live backend.

    Being disconnected is what separates the second from the errors that are
    not connection failures: a build or transfer refusal (stray isolated
    vertices, no usable rest shape) can only be raised while ONLINE, and
    there the server answers Start Server and Stop Server. A standing kill
    button outside these states is an invitation to end a healthy server.

    A stale port error is not offered either: ``_our_server_responding_in_error``
    has just found our own server answering on the port the message names.
    """
    if com.info.status == RemoteStatus.PROTOCOL_VERSION_MISMATCH:
        return True
    if stale_port_error or not error:
        return False
    if not com.is_connected():
        return True
    err_lower = error.lower()
    return _names_port_in_use(err_lower) or "already running on port" in err_lower


def _draw_long_path_warning(layout, path, project_name) -> bool:
    """Draw a warning when the build pipeline's deepest cache file under
    *path* would reach the Windows ``MAX_PATH`` limit for *project_name*, and
    return ``True``. Draws nothing and returns ``False`` otherwise.

    This is what makes a too-long Windows solver path fail loudly here, at the
    time it is set, instead of as a bare ``FileNotFoundError`` deep inside a
    later Transfer. Stays silent when Windows long-path support is enabled,
    since the limit no longer applies there.
    """
    if windows_long_paths_enabled():
        return False
    projected = windows_path_too_long(resolve_local_path(path), project_name)
    if projected is None:
        return False
    layout.label(
        text=iface_("Path too long: cache files reach {chars} chars (Windows limit {limit})").format(
            chars=projected, limit=WINDOWS_MAX_PATH
        ),
        icon="ERROR",
    )
    layout.label(text="Use a shorter solver path, or enable Windows long paths")
    return True


def _draw_name_warning(layout, name) -> bool:
    """Draw a one-line warning when the project *name* holds a space or a
    character that isn't filename-safe, and return ``True``. Draws nothing and
    returns ``False`` for a valid (or blank) name.
    """
    if find_invalid_name_char(name) is None:
        return False
    layout.label(text="Project name should not contain spaces or special characters", icon="ERROR")
    return True


def _draw_install_prompt(layout, *, operator_idname, module_label) -> None:
    """Draw the install operator for *module_label* followed by its current
    status: an in-progress notice while installing, the install error message
    (or a generic failure fallback) when the last attempt failed, or a prompt
    that the module still needs to be installed otherwise.
    """
    layout.operator(operator_idname)
    if get_installing_status():
        layout.label(text="Installing...", icon="FILE_REFRESH")
    else:
        install_result = get_install_result()
        if install_result is False:
            error_msg = get_install_error_message()
            if error_msg:
                layout.label(text=error_msg, icon="ERROR")
            else:
                layout.label(text=iface_("{module_label} installation failed.").format(module_label=module_label), icon="ERROR")
        else:
            layout.label(text=iface_("{module_label} needs to be installed.").format(module_label=module_label), icon="ERROR")


class MAIN_OT_ProjectNameFromFile(bpy.types.Operator):
    """Copy the .blend filename into the Project Name field"""

    bl_idname = "main.project_name_from_file"
    bl_label = "Use Filename"

    def execute(self, context):
        filepath = bpy.data.filepath
        name = os.path.splitext(os.path.basename(filepath))[0]
        get_addon_data(context.scene).state.project_name = name
        return {"FINISHED"}


class MAIN_PT_RemotePanel(Panel):
    """Backend Communicator panel: connection settings, server status, transfer, and statistics."""

    bl_label = "Backend Communicator"
    bl_idname = "MAIN_PT_RemotePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()

    @classmethod
    def poll(cls, context):
        return has_addon_data(context.scene)

    def draw(self, context):
        layout = self.layout
        root = get_addon_data(context.scene)
        state = root.state
        props = root.ssh_state

        # cbor2 encodes every Transfer regardless of backend, so prompt
        # for it up front whenever the bundled wheel is missing.
        if not cbor2_available():
            _draw_install_prompt(
                layout,
                operator_idname=REMOTE_OT_InstallCbor2.bl_idname,
                module_label="cbor2",
            )

        if (
            props.server_type == "COMMAND"
            or props.server_type == "CUSTOM"
            or "SSH" in props.server_type
        ):
            if not module_exists(["paramiko"]):
                _draw_install_prompt(
                    layout,
                    operator_idname=REMOTE_OT_InstallParamiko.bl_idname,
                    module_label="Paramiko",
                )
        elif props.server_type == "DOCKER" and not module_exists(["docker"]):
            _draw_install_prompt(
                layout,
                operator_idname=REMOTE_OT_InstallDocker.bl_idname,
                module_label="Docker-Py",
            )

        profile_row = layout.row(align=True)
        profile_row.enabled = com.is_connected() is False
        if props.profile_path and os.path.isfile(bpy.path.abspath(props.profile_path)):
            profile_row.prop(props, "profile_selection", text="Profile")
            profile_row.operator(
                REMOTE_OT_OpenProfile.bl_idname, text="", icon="FILEBROWSER"
            )
            profile_row.operator(
                "ssh.clear_profile", text="", icon="X"
            )
            profile_row.operator(
                "ssh.reload_profile", text="", icon="FILE_REFRESH"
            )
            profile_row.operator(
                "ssh.save_profile", text="", icon="FILE_TICK"
            )
        else:
            profile_row.operator(
                REMOTE_OT_OpenProfile.bl_idname, text="Open Profile", icon="FILEBROWSER"
            )
            profile_row.operator(
                "ssh.save_profile", text="", icon="FILE_TICK"
            )

        # Connection section (expandable)
        box = layout.box()
        row = box.row(align=True)
        row.alignment = 'LEFT'
        row.prop(
            state, "show_connection",
            icon="TRIA_DOWN" if state.show_connection else "TRIA_RIGHT",
            emboss=False, text="",
        )
        row.label(text="Connection", icon="LINKED" if com.is_connected() else "UNLINKED")

        if state.show_connection:
            col = box.column()
            col.enabled = com.is_connected() is False
            col.prop(props, "server_type")
            if props.server_type == "COMMAND":
                col.prop(props, "command")
            elif props.server_type == "CUSTOM":
                col.prop(props, "host")
                col.prop(props, "port")
                col.prop(props, "username")
                col.prop(props, "key_path")
                _draw_path_warning(col, props.key_path)
                col.prop(props, "proxy_jump")
            elif props.server_type == "DOCKER":
                col.prop(props, "container")
            elif props.server_type == "DOCKER_SSH":
                col.prop(props, "host")
                col.prop(props, "port")
                col.prop(props, "username")
                col.prop(props, "key_path")
                _draw_path_warning(col, props.key_path)
                col.prop(props, "proxy_jump")
                col.prop(props, "container")
            elif props.server_type == "DOCKER_SSH_COMMAND":
                col.prop(props, "command")
                col.prop(props, "container")
            if props.server_type in NATIVE_PATH_FIELDS:
                field = NATIVE_PATH_FIELDS[props.server_type]
                path = getattr(props, field)
                col.prop(props, field)
                # Held to the same rule the Connect gate uses for this
                # backend, so the warning and the button never disagree.
                if not _draw_path_warning(col, path, shell_bound=False):
                    _draw_native_status(col, props.server_type, path)
                    from ..core.connection import native_resolvers
                    resolver, _, builds, resolve_root = native_resolvers(
                        SERVER_TYPE_BACKENDS[props.server_type]
                    )
                    _draw_native_device(
                        col,
                        props,
                        resolve_root(resolve_local_path(path or "")) or path,
                        resolver,
                        builds,
                    )
                    if props.server_type == "WIN_NATIVE":
                        _draw_long_path_warning(col, path, state.project_name)
            elif props.server_type in ["CUSTOM", "COMMAND"]:
                col.prop(props, "ssh_remote_path")
                _draw_path_warning(col, props.ssh_remote_path)
            else:
                col.prop(props, "docker_path")
                _draw_path_warning(col, props.docker_path)

            # Drawn on the box rather than inside col, which is disabled while
            # connected: these are reachable exactly then, since both the
            # build and the GPU are applied at Start Server, and the
            # confirmation lines report the running server.
            _draw_remote_device(box, props)
            _draw_gpu_section(box, props)

            row = box.row(align=True)
            row.enabled = not com.is_server_running() and not com.is_server_launching()
            row.prop(state, "project_name", text="Project Name")
            if bpy.data.filepath and state.project_name.strip() in ("", "unnamed"):
                row.operator("main.project_name_from_file", text="", icon="COPYDOWN")
            _draw_name_warning(box, state.project_name)
            if "DOCKER" in props.server_type:
                box.prop(props, "docker_port")

            row = box.row()
            if com.is_connecting():
                sub = row.row()
                sub.enabled = False
                sub.operator(REMOTE_OT_Connect.bl_idname, text="Connecting...", icon="LINKED")
            else:
                row.operator(REMOTE_OT_Connect.bl_idname, icon="LINKED")
            row.operator(REMOTE_OT_Disconnect.bl_idname, icon="UNLINKED")

            row = box.row()
            if com.is_server_launching():
                sub = row.row()
                sub.enabled = False
                sub.operator(REMOTE_OT_StartServer.bl_idname, text="Server Starting...", icon="PLAY")
            else:
                row.operator(REMOTE_OT_StartServer.bl_idname, icon="PLAY")
            row.operator(REMOTE_OT_StopServer.bl_idname, icon="CANCEL")

            if not com.is_connected() and not com.is_connecting():
                row = box.row()
                row.label(text="Fill entries and click \"Connect\"", icon="INFO")
            elif com.is_connected() and not com.is_server_running() and not com.is_server_launching():
                row = box.row()
                row.label(text="Click \"Start Server on Remote\"", icon="INFO")

        status = com.info.status
        # After a crash the server stays in the failed state but keeps the
        # saved checkpoints, so surface both: "Simulation Failed (Resumable)"
        # rather than a bare "Resumable" (which hides the failure) or a bare
        # "Simulation Failed" (which hides that a resume is still possible).
        status_text = iface_(status.value)
        if status == RemoteStatus.SIMULATION_FAILED and len(com.saved_state_frames()) > 0:
            status_text = iface_("{status} (Resumable)").format(status=iface_(status.value))
        message = com.message or iface_("Status: {status}").format(status=status_text)
        if com.is_connecting():
            layout.label(text=message, icon=status.icon)
            layout.operator(REMOTE_OT_CancelConnect.bl_idname, text="Cancel", icon="X")
        elif com.is_server_launching():
            layout.label(text=message, icon=status.icon)
            layout.operator(REMOTE_OT_CancelStartServer.bl_idname, text="Cancel", icon="X")
        elif status == RemoteStatus.ABORTING:
            layout.label(text="Status: Aborting...", icon="CANCEL")
        elif status in (RemoteStatus.DATA_SENDING, RemoteStatus.BUILDING):
            if com.info.traffic:
                layout.label(
                    text=f"{message} ({com.info.traffic})",
                    icon=status.icon,
                )
            else:
                layout.label(text=message, icon=status.icon)
            layout.operator(REMOTE_OT_Abort.bl_idname, text="Cancel", icon="X")
        elif com.info.traffic:
            layout.label(
                text=f"{message} ({com.info.traffic})",
                icon=com.info.status.icon,
            )
        else:
            layout.label(text=message, icon=com.info.status.icon)
        error = com.error
        err_lower = error.lower()
        # Defense in depth for the port refusal: when the message says the
        # port is in use, probe it live before showing. If our own
        # ppf-cts-server is now responding there (e.g. the user clicked
        # Connect a second time and the spawn path's attach branch took
        # over), the error is stale, so suppress the label AND the button and
        # do not tempt the user into killing our own running server. The
        # "already running" refusal is different: there our server IS
        # answering, and that is the problem being reported.
        is_port_error = bool(error) and _names_port_in_use(err_lower)
        stale_port_error = is_port_error and _our_server_responding_in_error(error)
        if error and not stale_port_error:
            _draw_error_lines(layout, error)
            # A repair the artist can press sits next to the refusal that asks
            # for it. These two are BUILD refusals, which can only be raised
            # while connected; a CONNECTION failure takes the Force Terminate
            # Process row below instead.
            if "isolated vert" in err_lower:
                # Stray faceless vertices on a STATIC collider abort the
                # build; offer a one-click cleanup (see the encoder's
                # detect_isolated_vertices check / geometry_cleanup_ops).
                layout.operator(
                    MESH_OT_RemoveIsolatedVertices.bl_idname, icon="TRASH",
                )
            elif "no usable rest shape" in err_lower:
                # The failing Transfer already opened a dialog carrying
                # this button, but a dismissed dialog must not take the
                # repair with it, so the panel keeps offering it for as
                # long as the error stands. Only when triangulating is the
                # right repair, which the message says: a face that is
                # already a triangle has no other split, and one that is
                # degenerate itself has no sound split at all. Splitting
                # either would change the mesh without fixing it.
                if "triangulate faces" in err_lower:
                    layout.operator(
                        MESH_OT_TriangulateDegenerateFaces.bl_idname,
                        icon="MOD_TRIANGULATE",
                    )
        # The way out sits next to the refusal that names it, and the
        # Connection box is collapsible, so the row goes here rather than
        # beside Stop Server: a refused Connect must never leave the artist
        # with a server process and nothing to end it with.
        if _force_terminate_offered(error, stale_port_error=stale_port_error):
            _draw_force_terminate(layout, props, context)

        server_error = com.server_error
        crash_kind = com.crash_kind
        if server_error and crash_kind:
            # A crash report is a multi-line document (cause, detail, then the
            # solver's stdout and stderr tails), and a panel label renders one
            # truncated line. Draw the localized cause plus the first line of
            # detail here, and leave the full report to the Console, which the
            # same response already logged line by line.
            layout.label(
                text=iface_("Solver failed: {cause}").format(
                    cause=iface_(crash_cause_summary(crash_kind))
                ),
                icon="ERROR",
            )
            detail = _crash_detail_line(server_error, crash_kind)
            if detail:
                # Drawn untranslated: it is machine data (a CUDA error name, a
                # signal name, a file and line), not prose.
                layout.label(text=detail)
        elif server_error:
            # Any server error can carry supporting lines below its first one
            # (a build failure appends the worker's traceback), and
            # `layout.label` draws a multi-line string on a single line with
            # the breaks as glyphs. Show the headline; the Show Console button
            # on the next row opens the full report, one line each.
            headline = server_error.partition("\n")[0]
            layout.label(text=iface_("Remote: {error}").format(error=headline), icon="ERROR")

        row = layout.row()
        row.operator(SOLVER_OT_UpdateStatus.bl_idname, icon="FILE_REFRESH")
        row.operator(SOLVER_OT_ShowConsole.bl_idname, icon="CONSOLE")
        row.prop(state, "debug_mode")
        if crash_kind:
            # Drawn whenever a crash is being reported, disabled with a reason
            # when the folder cannot be resolved, so the route to the logs is
            # discoverable rather than appearing only in the case that works.
            col = layout.column(align=True)
            col.enabled = bool(SOLVER_OT_OpenSessionFolder.session_path())
            col.operator(SOLVER_OT_OpenSessionFolder.bl_idname, icon="FILE_FOLDER")
            if not col.enabled:
                col.label(text=iface_("The session folder is not known for this run."))

        # Remote Hardware info (shown when connected)
        hardware = com.response.get("hardware", {})
        if hardware and com.is_connected():
            hw_box = layout.box()
            row = hw_box.row()
            row.prop(
                state,
                "show_hardware",
                icon="TRIA_DOWN" if state.show_hardware else "TRIA_RIGHT",
                emboss=False,
                icon_only=True,
            )
            row.label(text="Remote Hardware", icon="DESKTOP")
            if state.show_hardware:
                col = hw_box.column(align=True)
                for key, value in hardware.items():
                    if key in _UNDISPLAYED_HARDWARE_KEYS:
                        continue
                    row = col.row(align=True)
                    row.label(text=key)
                    row.label(text=str(value))

        # Scene-encode / drift-check progress. This runs on the main thread
        # inside Transfer / Run before any server status exists, so it has its
        # own snapshot (see core.encode_progress); showing it here gives a
        # labeled bar from the moment of the click that flows straight into the
        # server's build/sim bar below.
        from ..core import encode_progress
        if encode_progress.is_active():
            done, total, label = encode_progress.snapshot()
            factor = min(1.0, done / total) if total else 0.0
            layout.progress(
                factor=factor, type="BAR", text=label or "Preparing scene data...",
            )

        response = com.response
        if com.info.status.in_progress() or is_running(response):
            progress_text = com.message or com.info.status.value
            layout.progress(
                factor=com.info.progress, type="BAR", text=progress_text
            )
            if com.info.status.abortable():
                layout.operator(REMOTE_OT_Abort.bl_idname, icon="CANCEL", text="Abort")
            row = layout.row()
            if is_simulating(response):
                row.operator(SOLVER_OT_SaveAndQuit.bl_idname, icon="FILE_TICK")
                row.operator(SOLVER_OT_Terminate.bl_idname, icon="CANCEL")
        live_summary = com.response.get("summary", {})
        average_summary = com.response.get("average_summary", {})
        displayed_summary = live_summary if is_simulating(response) else average_summary
        stats_label = "Realtime Statistics" if is_simulating(response) else "Average Statistics"
        show_stats_box = is_simulating(response) or bool(average_summary)

        if show_stats_box:
            stats_box = layout.box()
            stats_box.prop(
                state,
                "show_statistics",
                icon="TRIA_DOWN" if state.show_statistics else "TRIA_RIGHT",
                emboss=False,
                text=stats_label,
            )

            def add_statistic_row(col, label, value):
                row = col.row(align=True)
                row.label(text=label)
                row.label(text=value)

            if state.show_statistics:
                col = stats_box.column(align=True)
                for key, value in displayed_summary.items():
                    # Remap remote frame index to Blender frame (0-based → 1-based)
                    if key == "frame":
                        value = remote_frame_to_blender(value)
                    add_statistic_row(col, key, value)

        scene_info = com.response.get("scene_info", {})
        if scene_info:
            # Remap remote frame indices to Blender frames (0-based → 1-based)
            scene_info = dict(scene_info)
            if "Total Frames" in scene_info:
                scene_info["Total Frames"] = remote_frame_to_blender(
                    scene_info["Total Frames"], grouped=True
                )
            if "Last Saved" in scene_info and scene_info["Last Saved"] != "None":
                scene_info["Last Saved"] = remote_frame_to_blender(
                    scene_info["Last Saved"], grouped=True
                )
        if scene_info:
            info_box = layout.box()
            info_box.prop(
                state,
                "show_scene_info",
                icon="TRIA_DOWN" if state.show_scene_info else "TRIA_RIGHT",
                emboss=False,
                text="Scene Info",
            )
            if state.show_scene_info:
                col = info_box.column(align=True)
                for key, value in scene_info.items():
                    row = col.row(align=True)
                    row.label(text=key)
                    row.label(text=str(value))

        if state.debug_mode:
            box = layout.box()
            row = box.row()
            row.label(text="Shell Calls", icon="CONSOLE")

            prop_row = box.row()
            prop_row.enabled = com.is_server_running()
            prop_row.prop(state, "server_script")
            box.operator(DEBUG_OT_ExecuteServer.bl_idname)

            prop_row = box.row()
            prop_row.enabled = com.is_connected()
            prop_row.prop(state, "shell_command")
            row = box.row()
            row.operator(DEBUG_OT_ExecuteShell.bl_idname)
            prop_col = row.column()
            prop_col.enabled = com.is_connected()
            prop_col.prop(state, "use_shell")

            label_row = box.row()
            label_row.enabled = com.is_connected()
            label_row.label(text="Data Transfer Tests", icon="ARROW_LEFTRIGHT")
            row = box.row(align=True)
            row.operator(DEBUG_OT_DataSend.bl_idname, icon="EXPORT")
            row.operator(DEBUG_OT_DataReceive.bl_idname, icon="IMPORT")
            prop_row = box.row()
            prop_row.enabled = com.is_connected()
            prop_row.prop(state, "data_size", text="Data Size (MB)")

            col = box.column()
            col.label(text="Options", icon="PREFERENCES")
            col.prop(state, "max_console_lines")

            col = box.column()
            col.label(text="Console Log Export", icon="TEXT")
            row = col.row(align=True)
            row.prop(state, "log_file_path", text="")
            row.operator(DEBUG_OT_BrowseLogPath.bl_idname, icon="FILEBROWSER", text="")
            row.operator(DEBUG_OT_ClearLogPath.bl_idname, icon="X", text="")
            row = col.row(align=True)
            row.enabled = bool(state.log_file_path)
            row.operator(DEBUG_OT_DeleteLog.bl_idname, icon="TRASH", text="Delete Log")

            label_row = box.row()
            label_row.enabled = com.is_connected()
            label_row.label(text="GitHub Repo on Remote", icon="URL")
            row = box.row(align=True)
            row.operator(DEBUG_OT_GitPull.bl_idname, icon="IMPORT")
            row.operator(DEBUG_OT_Compile.bl_idname, icon="FILE_REFRESH")
            box.operator(WM_OT_OpenGitHubLink.bl_idname, icon="URL")

            col = box.column()
            col.label(text="GitHub Repo on Local", icon="URL")
            col.operator(DEBUG_OT_GitPullLocal.bl_idname, icon="IMPORT")

            col = box.column()
            col.label(text="UUID Migration", icon="FILE_REFRESH")
            col.operator("debug.run_uuid_migration", icon="FILE_REFRESH")
            if state.uuid_migration_result:
                col.label(text=state.uuid_migration_result, icon="INFO")

            col = box.column()
            col.label(text="Render", icon="RENDER_ANIMATION")
            row = col.row(align=True)
            running = is_render_anim_running()
            r1 = row.row(align=True)
            r1.enabled = not running
            r1.operator(DEBUG_OT_RenderAnimation.bl_idname,
                        text="Render Animation", icon="RENDER_ANIMATION")
            r2 = row.row(align=True)
            r2.enabled = running
            r2.operator(DEBUG_OT_StopRender.bl_idname,
                        text="Stop", icon="PAUSE")
            if running:
                current, total, current_frame = get_render_anim_progress()
                pct = (current / total) if total > 0 else 0.0
                col.row().progress(
                    factor=pct,
                    type="BAR",
                    text=iface_("Frame {frame}  ({current}/{total}, {pct:.0f}%)").format(frame=current_frame, current=current, total=total, pct=pct*100),
                )

            col = box.column()
            col.label(text="Add-on Local Debug Server", icon="TOOL_SETTINGS")

            row = col.row(align=True)
            debug_running = get_reload_server_status()
            if debug_running:
                row.operator("addon.stop_reload_server", text="Stop", icon="PAUSE")
            else:
                row.operator("addon.start_reload_server", text="Start", icon="PLAY")
            sub = row.row()
            sub.enabled = not debug_running
            sub.prop(state, "reload_port", text="Port")
            reload_row = col.row(align=True)
            reload_row.operator(
                "addon.trigger_reload", text="Reload Add-on Now", icon="FILE_REFRESH"
            )
            reload_row.operator(
                "addon.trigger_full_reload", text="Full Reload", icon="FILE_REFRESH"
            )



@dataclass
class GlobalStateWatcher:
    last_state: RemoteStatus = RemoteStatus.UNKNOWN
    last_install_status: bool = False
    last_install_result: bool | None = None
    last_progress: float = 0.0
    last_is_connected: bool | None = None
    last_is_connecting: bool | None = None
    last_is_server_running: bool | None = None
    last_is_server_launching: bool | None = None
    last_message: str | None = None
    last_traffic: str | None = None

    def has_changed(self):
        return (
            self.last_state != com.info.status
            or self.last_progress != com.info.progress
            or self.last_install_status != get_installing_status()
            or self.last_install_result != get_install_result()
            or self.last_is_connected != com.is_connected()
            or self.last_is_connecting != com.is_connecting()
            or self.last_is_server_running != com.is_server_running()
            or self.last_is_server_launching != com.is_server_launching()
            or self.last_message != com.message
            or self.last_traffic != com.info.traffic
        )

    def reset(self):
        self.last_progress = com.info.progress
        self.last_state = com.info.status
        self.last_install_status = get_installing_status()
        self.last_install_result = get_install_result()
        self.last_is_connected = com.is_connected()
        self.last_is_connecting = com.is_connecting()
        self.last_is_server_running = com.is_server_running()
        self.last_is_server_launching = com.is_server_launching()
        self.last_message = com.message
        self.last_traffic = com.info.traffic


global_state = GlobalStateWatcher()


def refresh_ssh_panel():
    """refresh the ssh connection panel ui."""
    global global_state
    if global_state.has_changed():
        global_state.reset()
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()


classes = (
    connection_classes
    + install_classes
    + mcp_classes
    + solver_control_classes
    + list(geometry_cleanup_classes)
    + [MAIN_OT_ProjectNameFromFile, MAIN_PT_RemotePanel]
    + debug_classes
    + addon_classes
    + list(jupyter_classes)
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
