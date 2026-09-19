# File: connection_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# SSH/Docker/Local connection operators.

import time

import bpy  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_  # pyright: ignore
from bpy.props import StringProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ..core.async_op import AsyncOperator
from ..core.client import communicator as com
from ..core.module import module_exists
from ..core.ssh_command import parse_ssh_command
from ..core.utils import (
    find_invalid_name_char,
    find_invalid_path_char,
    find_shell_unsafe_path_char,
    get_timer_wait_time,
    redraw_all_areas,
    resolve_local_path,
)
from ..models.groups import get_addon_data

# THE CONNECTION TYPES WHOSE SERVER RUNS ON THIS MACHINE, mapped to the property
# holding the folder it runs from.
#
# One per platform, because a solver root is a path on THIS machine and the
# three platforms spell and pick one differently; the artist sees one "Solver
# Path" field either way, since only their own platform's type is usable.
#
# ONE MAP, SO THE PANEL, THE CONNECT GATE AND THE FORCE-TERMINATE BUTTON CANNOT
# DISAGREE. A list of native types in each of them is a place a type can be
# added to two of the three, and that gap is silent: the gate simply returns
# None, which reads as "cannot connect" with no reason given, and the button
# simply hides.
NATIVE_PATH_FIELDS = {
    "WIN_NATIVE": "win_native_path",
    "MAC_NATIVE": "mac_native_path",
    "LINUX_NATIVE": "linux_native_path",
}

# The backend each native connection type creates. The panel needs the backend's
# name to reach its resolvers, and the two spellings differ only in case and
# order, which is exactly the kind of thing that is worth writing down once
# rather than deriving with `.lower()` at four call sites.
SERVER_TYPE_BACKENDS = {
    "WIN_NATIVE": "win_native",
    "MAC_NATIVE": "mac_native",
    "LINUX_NATIVE": "linux_native",
}

# The connection types that reach a server on ANOTHER machine, which is where
# the solver builds have to be asked for rather than looked at.
REMOTE_SERVER_TYPES = (
    "CUSTOM",
    "COMMAND",
    "DOCKER",
    "DOCKER_SSH",
    "DOCKER_SSH_COMMAND",
)


def _refresh_ssh_panel_bridge():
    """Forward to main_panel.refresh_ssh_panel (the canonical implementation).

    This wrapper uses a lazy import to break a circular dependency:
    main_panel.py imports operator classes from connection_ops.py, so
    connection_ops.py cannot import from main_panel.py at module level.
    """
    from .main_panel import refresh_ssh_panel
    refresh_ssh_panel()


class REMOTE_OT_Connect(Operator):
    """Establish an SSH connection and execute a command asynchronously.

    Note: This operator intentionally does NOT use AsyncOperator because
    its modal covers only the handshake: it polls until the connection is
    up, fails, or times out, then finishes.

    It must NOT stay alive for the connection's lifetime. Blender skips
    its auto-save for as long as any modal operator handler is attached to
    a window, re-arming the auto-save timer every 10 ms instead of
    writing, so a modal held open across a working session leaves the
    user with no recovery file at all. Watching for the disconnect and
    refreshing the SSH panel need no modal context, since both only tag a
    redraw, and run from the persistent tick in ``core.facade`` instead.
    """

    bl_idname = "ssh.run_command"
    bl_label = "Connect"

    _timer = None
    _connection_established = False
    _start_time: float = 0.0
    timeout: float = 60.0

    def get_remote_path(self, props):
        """Return the solver directory for the selected connection type.

        The SSH and Docker paths name a directory on the SOLVER HOST, where the
        client's .blend location has no meaning, so they are returned verbatim.
        A native path names a directory on the machine Blender runs on and goes
        through ``resolve_local_path`` at its own call site instead, because the
        picker stores it in Blender's ``//``-relative notation whenever the
        .blend is saved and relative paths are enabled, and ``os.path`` cannot
        read that form.
        """
        if props.server_type in ["CUSTOM", "COMMAND"]:
            return props.ssh_remote_path
        else:
            return props.docker_path

    @classmethod
    def poll(cls, context):
        # A SECOND REQUEST WHILE ONE IS HANDSHAKING CHANGES NOTHING, because
        # the reducer accepts a connect only from the offline phase. Refusing
        # it here is what the MCP tool already does (``_require_offline``), and
        # it keeps a click that would start a modal watching an attempt it did
        # not start from being possible at all.
        if com.is_connecting():
            return False
        root = get_addon_data(context.scene)
        props = root.ssh_state
        state = root.state
        project_name_valid = (
            state.project_name.strip() != ""
            and find_invalid_name_char(state.project_name) is None
        )

        if props.server_type == "COMMAND":
            return (
                not com.is_connected()
                and props.command.strip() != ""
                and find_invalid_path_char(props.ssh_remote_path) is None
                and module_exists(["paramiko"])
                and project_name_valid
            )
        elif props.server_type == "CUSTOM":
            return (
                not com.is_connected()
                and props.host.strip() != ""
                and find_invalid_path_char(props.key_path) is None
                and find_invalid_path_char(props.ssh_remote_path) is None
                and module_exists(["paramiko"])
                and project_name_valid
            )
        elif props.server_type == "DOCKER":
            # Local Docker reaches the daemon over the Docker socket, so no
            # SSH key takes part in the connection and the panel does not draw
            # the SSH Key field in this mode. Gating the button on that field
            # made the button unpressable for a reason nothing on screen could
            # explain: the default key path is derived from the user's home
            # directory, so a Windows account whose name holds a space put a
            # space in it, and the shell-safety test rejected a value the user
            # could neither see nor edit here.
            return (
                not com.is_connected()
                and props.container.strip() != ""
                and find_invalid_path_char(props.docker_path) is None
                and module_exists(["docker"])
                and project_name_valid
            )
        elif props.server_type == "DOCKER_SSH":
            return (
                not com.is_connected()
                and props.host.strip() != ""
                and props.container.strip() != ""
                and find_invalid_path_char(props.key_path) is None
                and find_invalid_path_char(props.docker_path) is None
                and module_exists(["paramiko"])
                and project_name_valid
            )
        elif props.server_type == "DOCKER_SSH_COMMAND":
            return (
                not com.is_connected()
                and props.command.strip() != ""
                and props.container.strip() != ""
                and find_invalid_path_char(props.docker_path) is None
                and module_exists(["paramiko"])
                and project_name_valid
            )
        elif props.server_type in NATIVE_PATH_FIELDS:
            # A NATIVE ROOT NEVER REACHES A SHELL (it is an os.path.join base, a
            # Popen argv element, and that Popen's cwd), so it is held to the
            # metacharacter rule only. A space is ordinary in a path on all
            # three platforms, and this button is the only place the user could
            # act on a refusal of one, with no field on screen to change and
            # nothing wrong with what they picked.
            path = getattr(props, NATIVE_PATH_FIELDS[props.server_type])
            return (
                not com.is_connected()
                and path.strip() != ""
                and find_shell_unsafe_path_char(path) is None
                and project_name_valid
            )

    def _connect_ssh(self, **kwargs) -> bool:
        """Ask the facade for an SSH connection; report a bad jump spec.

        The jump chain is resolved before anything is dispatched, so a spec
        that names no host, or one whose hosts jump back to each other, is
        refused here with the reason rather than surfacing as a traceback from
        the operator.
        """
        try:
            com.connect_ssh(**kwargs)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return False
        return True

    def execute(self, context):
        root = get_addon_data(context.scene)
        props = root.ssh_state
        com.set_project_name(root.state.project_name)
        if props.server_type == "COMMAND" or props.server_type == "DOCKER_SSH_COMMAND":
            try:
                parsed = parse_ssh_command(props.command)
            except ValueError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            if not parsed.host:
                self.report(
                    {"ERROR"},
                    iface_("Failed to parse command. Ensure it includes host."),
                )
                return {"CANCELLED"}
            container = props.container if "DOCKER" in props.server_type else None
            # The panel's own Proxy Jump field is not drawn for the command
            # backends, where -J is where a jump host belongs, so the command
            # is the only source here.
            if not self._connect_ssh(
                host=parsed.host,
                port=parsed.port or 22,
                username=parsed.username,
                key_path=parsed.key_path,
                path=self.get_remote_path(props),
                container=container,
                server_port=props.docker_port,
                proxy_jump=parsed.proxy_jump,
                device=props.native_device,
                gpu_backend=props.native_gpu_backend,
            ):
                return {"CANCELLED"}
        elif props.server_type == "CUSTOM" or props.server_type == "DOCKER_SSH":
            container = props.container if "DOCKER" in props.server_type else None
            if not self._connect_ssh(
                host=props.host,
                port=props.port,
                username=props.username,
                key_path=props.key_path,
                path=self.get_remote_path(props),
                container=container,
                server_port=props.docker_port,
                proxy_jump=props.proxy_jump.strip(),
                device=props.native_device,
                gpu_backend=props.native_gpu_backend,
            ):
                return {"CANCELLED"}
        elif props.server_type == "DOCKER":
            com.connect_docker(
                props.container,
                self.get_remote_path(props),
                server_port=props.docker_port,
                device=props.native_device,
                gpu_backend=props.native_gpu_backend,
            )
        elif props.server_type == "WIN_NATIVE":
            win_path = resolve_local_path(props.win_native_path)
            if not win_path:
                self.report({"ERROR"}, iface_("Solver path is not set"))
                return {"CANCELLED"}
            com.connect_win_native(
                win_path,
                props.docker_port,
                props.native_device,
                props.native_gpu_backend,
            )
        elif props.server_type == "MAC_NATIVE":
            mac_path = resolve_local_path(props.mac_native_path)
            if not mac_path:
                self.report({"ERROR"}, iface_("Solver path is not set"))
                return {"CANCELLED"}
            com.connect_mac_native(
                mac_path, props.docker_port, props.native_device
            )
        elif props.server_type == "LINUX_NATIVE":
            linux_path = resolve_local_path(props.linux_native_path)
            if not linux_path:
                self.report({"ERROR"}, iface_("Solver path is not set"))
                return {"CANCELLED"}
            com.connect_linux_native(
                linux_path,
                props.docker_port,
                props.native_device,
                props.native_gpu_backend,
            )

        self._connection_established = False
        self._start_time = time.time()
        self._timer = context.window_manager.event_timer_add(
            get_timer_wait_time(), window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _detach_timer(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        # Check is_connected() before the cancel/timeout branches: a fast connect
        # (e.g. a native) can reach ONLINE before the first timer tick, which would
        # otherwise be misread as a cancellation since is_connecting() is False.
        if com.is_connected():
            self._connection_established = True
            self._detach_timer(context)
            redraw_all_areas(context)
            return {"FINISHED"}
        # Detect cancellation (user canceled or connection failed). Reaching
        # here means the connection is not up, so the established flag can
        # only be False and does not need testing.
        if not com.is_connecting():
            self._detach_timer(context)
            return {"CANCELLED"}
        # Still handshaking, so the timeout applies on every tick that
        # gets this far.
        if time.time() - self._start_time > self.timeout:
            self._detach_timer(context)
            # THE TIMEOUT TEARS THE ATTEMPT DOWN, exactly as Cancel does. The
            # phase is CONNECTING and this operator is the only thing watching
            # it, so returning without disconnecting leaves the state machine
            # in a phase nothing owns: the reducer accepts a connect request
            # only from OFFLINE, so every later click is dropped without a word
            # and the panel reads "Connecting..." until Blender is restarted.
            com.disconnect()
            self.report({"ERROR"}, iface_("Connection timed out"))
            redraw_all_areas(context)
            return {"CANCELLED"}
        _refresh_ssh_panel_bridge()
        return {"PASS_THROUGH"}


class REMOTE_OT_CancelConnect(Operator):
    """Cancel a pending connection attempt."""

    bl_idname = "ssh.cancel_connect"
    bl_label = "Cancel"

    @classmethod
    def poll(cls, _):
        return com.is_connecting()

    def execute(self, context):
        com.disconnect()
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_Disconnect(Operator):
    """Disconnect the SSH connection."""

    bl_idname = "ssh.disconnect"
    bl_label = "Disconnect"

    @classmethod
    def poll(cls, _):
        return com.is_connected() and not com.info.status.abortable()

    def execute(self, context):
        com.disconnect()
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_Abort(Operator):
    """Abort the current in-progress operation."""

    bl_idname = "ssh.abort"
    bl_label = "Abort"

    def execute(self, context):
        com.abort()
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_StartServer(AsyncOperator):
    """Start the remote server process."""

    bl_idname = "ssh.start_server"
    bl_label = "Start Server on Remote"

    timeout: float = 60.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        return (
            com.is_connected() and com.is_server_running() is False and not com.busy()
        )

    def execute(self, context):
        from ..core.gpu_devices import selected_device, validate_selection

        props = get_addon_data(context.scene).ssh_state
        selected = props.solver_gpu_index
        device = selected_device(selected, props.solver_gpu_uuid)
        if device is not None:
            selected = device.index
            props.solver_gpu_index = selected
            props.solver_gpu_uuid = device.uuid
        # Refuse a GPU the solver host does not have rather than starting a
        # server against an empty CUDA device set, which surfaces much later as
        # a solver error that names no GPU.
        try:
            validate_selection(selected, props.solver_gpu_uuid)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        com.start_server(
            selected,
            props.solver_gpu_uuid,
            props.native_device,
            props.native_gpu_backend,
        )
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return com.is_server_running()

    def is_cancelled(self) -> bool:
        return not com.is_server_launching() and not com.is_server_running()

    def on_complete(self, context):
        redraw_all_areas(context)


class REMOTE_OT_CancelStartServer(Operator):
    """Cancel a pending server start."""

    bl_idname = "ssh.cancel_start_server"
    bl_label = "Cancel"

    @classmethod
    def poll(cls, _):
        return com.is_server_launching()

    def execute(self, context):
        com.stop_server()
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_StopServer(AsyncOperator):
    """Stop the remote server process."""

    bl_idname = "ssh.stop_server"
    bl_label = "Stop Server on Remote"

    timeout: float = 60.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        return (
            com.is_connected()
            and com.is_server_running()
            and not com.info.status.abortable()
        )

    def execute(self, context):
        com.stop_server()
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return not com.is_server_running()

    def on_complete(self, context):
        redraw_all_areas(context)


# Server types whose server runs on THIS machine, on a loopback port. For
# these the Force Terminate Process button works without a connection: the port is
# inspectable from here, and a native Connect is refused exactly while a
# server from an earlier session still holds it.
LOCAL_SERVER_TYPES = tuple(NATIVE_PATH_FIELDS)

# The panel redraws many times a second, so the loopback probe behind the
# Force Terminate Process status line is cached for this long. One (port, timestamp,
# verdict) tuple: only the configured port is ever asked.
_LISTENER_TTL_S = 1.5
_listener_cache: tuple[int, float, str] | None = None

# Verdicts of ``local_listener_verdict``.
LISTENER_OURS = "ours"
LISTENER_FOREIGN = "foreign"
LISTENER_FREE = "free"


def is_local_server_type(server_type: str) -> bool:
    return server_type in LOCAL_SERVER_TYPES


def local_listener_verdict(port: int) -> str:
    """Whether a ppf-cts-server, some other program, or nothing listens on
    loopback *port*, cached for ``_LISTENER_TTL_S``."""
    import time

    global _listener_cache
    from ..core.connection import _port_is_in_use, _probe_ppf_cts_server

    now = time.monotonic()
    if (
        _listener_cache
        and _listener_cache[0] == port
        and now - _listener_cache[1] < _LISTENER_TTL_S
    ):
        return _listener_cache[2]
    if _probe_ppf_cts_server(port, timeout=0.5):
        verdict = LISTENER_OURS
    elif _port_is_in_use(port):
        verdict = LISTENER_FOREIGN
    else:
        verdict = LISTENER_FREE
    _listener_cache = (port, now, verdict)
    return verdict


def force_terminate_status(props) -> tuple[str, str]:
    """The status line under the Force Terminate Process button, as ``(text, icon)``.

    Three answers: a server of ours is listening on the port, nothing is,
    or the port cannot be checked from here and why. A local type is asked
    on the loopback; a remote type is reachable only through the connection,
    and a panel draw must never issue a backend command, so its answer is
    read off the engine's own view of the server.
    """
    port = props.docker_port
    if is_local_server_type(props.server_type):
        verdict = local_listener_verdict(port)
        if verdict == LISTENER_OURS:
            return (
                iface_("A solver server is listening on port {port}").format(port=port),
                "CHECKMARK",
            )
        if verdict == LISTENER_FOREIGN:
            return (
                iface_("Port {port} is held by another program").format(port=port),
                "ERROR",
            )
        return (
            iface_("Nothing is listening on port {port}").format(port=port),
            "INFO",
        )
    if com.is_server_running():
        return (
            iface_("A solver server is listening on port {port} of the solver host").format(port=port),
            "CHECKMARK",
        )
    return (
        iface_("Cannot check port {port} from here: Force Terminate Process ends the ppf-cts-server on that port").format(port=port),
        "QUESTION",
    )


class SOLVER_OT_ForceTerminatePort(AsyncOperator):
    """End the ppf-cts-server process, whether or not this add-on started it.

    Two paths, decided by whether the add-on is connected:

    - Not connected, local server type: the loopback port is killed directly
      (``kill_local_server``), and the connection state is reset the way
      Disconnect resets it, so the refusal the artist was looking at is gone
      and a fresh Connect starts clean. This is the state the button exists
      for: a native Connect refused because a server from an earlier session
      still holds the port, where Stop Server is unreachable.
    - Connected, any server type: the kill goes through the live backend on
      the worker thread (``KillServerRequested``, the same effect Stop Server
      on Remote runs), and the state is left the way Stop Server leaves it:
      connected to the host, server stopped, Start Server next.

    A remote type with no connection has no transport to act through, so
    the poll refuses it and the panel does not draw the button then.
    """

    bl_idname = "solver.force_terminate_port"
    bl_label = "Force Terminate Process"
    bl_description = (
        "End the solver server process on the configured port, whether or "
        "not this add-on started it"
    )

    timeout: float = 60.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, context):
        props = get_addon_data(context.scene).ssh_state
        idle = (
            not com.busy()
            and not com.is_connecting()
            and not com.is_server_launching()
            and not com.is_server_stopping()
            and not com.info.status.abortable()
        )
        if com.is_connected():
            return idle
        return idle and is_local_server_type(props.server_type)

    def execute(self, context):
        from ..core.server_kill import kill_local_server

        props = get_addon_data(context.scene).ssh_state
        if com.is_connected():
            com.kill_server()
            if not com.is_server_stopping():
                # The transition guard refused it: something started between
                # the poll and the click.
                self.report({"WARNING"}, iface_("Force Terminate Process: the connection is busy, try again"))
                return {"CANCELLED"}
            self.setup_modal(context)
            return {"RUNNING_MODAL"}
        if not is_local_server_type(props.server_type):
            self.report(
                {"ERROR"},
                iface_("Force Terminate Process needs a connection for this server type"),
            )
            return {"CANCELLED"}
        report = kill_local_server(props.docker_port)
        # The refusal is in the connection error, and a fresh Connect must
        # start from a clean state: this is what Disconnect does.
        com.disconnect()
        self._report(report, next_step=iface_("Press Connect to start a new one."))
        redraw_all_areas(context)
        return {"FINISHED"}

    def is_complete(self) -> bool:
        return not com.is_server_stopping()

    def on_complete(self, context):
        report = com.last_kill_report
        if report is None:
            self.report({"WARNING"}, iface_("Force Terminate Process finished without a report"))
        else:
            self._report(
                report, next_step=iface_("Press Start Server to launch a new one.")
            )
        redraw_all_areas(context)

    def _report(self, report, *, next_step: str) -> None:
        """Say what was killed (pid, port, where) or why nothing was."""
        pids = ", ".join(str(p) for p in report.killed)
        if not report.checked:
            self.report(
                {"ERROR"},
                iface_("Could not inspect port {port} ({where}): {error}").format(
                    port=report.port, where=report.where, error=report.error
                ),
            )
            return
        if report.nothing_found:
            self.report(
                {"WARNING"},
                iface_("No ppf-cts-server was listening on port {port} ({where}).").format(
                    port=report.port, where=report.where
                )
                + " " + next_step,
            )
            return
        text = iface_("Killed pid {pids} on port {port} ({where}).").format(
            pids=pids, port=report.port, where=report.where
        )
        level = "INFO"
        if report.survivors:
            left = ", ".join(str(p) for p in report.survivors)
            text += " " + iface_("Still running after the kill: pid {pids}.").format(pids=left)
            level = "WARNING"
        if report.error:
            text += " " + report.error
            level = "WARNING"
        self.report({level}, text + " " + next_step)


class REMOTE_OT_OpenProfile(Operator):
    """Open a TOML connection profile file."""

    bl_idname = "ssh.open_profile"
    bl_label = "Open Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "connection_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ..core.profile import get_profile_names

        root = get_addon_data(context.scene)
        props = root.ssh_state
        props.profile_path = self.filepath
        abs_path = bpy.path.abspath(self.filepath)
        names = get_profile_names(abs_path)
        if names:
            props.profile_selection = names[0]
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_ClearProfile(Operator):
    """Clear the loaded connection profile."""

    bl_idname = "ssh.clear_profile"
    bl_label = "Clear Profile"

    def execute(self, context):
        root = get_addon_data(context.scene)
        root.ssh_state.profile_path = ""
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_ReloadProfile(Operator):
    """Re-apply the currently selected connection profile."""

    bl_idname = "ssh.reload_profile"
    bl_label = "Reload Profile"

    def execute(self, context):
        from ..core.profile import apply_profile, load_profiles

        root = get_addon_data(context.scene)
        props = root.ssh_state
        if not props.profile_path or props.profile_selection == "NONE":
            return {"CANCELLED"}
        abs_path = bpy.path.abspath(props.profile_path)
        profiles = load_profiles(abs_path)
        profile = profiles.get(props.profile_selection)
        if profile is None:
            return {"CANCELLED"}
        apply_profile(profile, props)
        redraw_all_areas(context)
        return {"FINISHED"}


class REMOTE_OT_RefreshGpuDevices(Operator):
    """Re-run nvidia-smi and rebuild the GPU list."""

    bl_idname = "ssh.refresh_gpu_devices"
    bl_label = "Refresh GPU List"

    def execute(self, context):
        # The solver host is reachable only through the connection, whichever
        # backend it is, and that command belongs on the worker thread that
        # owns it. The list lands in the cache and the panel redraws with it.
        if not com.is_connected():
            self.report(
                {"ERROR"},
                iface_("Connect first: the GPU list is read from the solver host"),
            )
            return {"CANCELLED"}
        com.refresh_solver_host_gpus()
        return {"FINISHED"}


class REMOTE_OT_SaveProfile(Operator):
    """Save current connection settings to a profile."""

    bl_idname = "ssh.save_profile"
    bl_label = "Save Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    entry_name: StringProperty(name="Entry Name", default="Default")  # pyright: ignore

    def invoke(self, context, event):
        root = get_addon_data(context.scene)
        props = root.ssh_state
        if props.profile_path and props.profile_selection != "NONE":
            return self.execute(context)
        if not self.filepath:
            self.filepath = "connection_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ..core.profile import read_connection_profile, save_profile_entry

        root = get_addon_data(context.scene)
        props = root.ssh_state
        data = read_connection_profile(props)
        if props.profile_path and props.profile_selection != "NONE":
            abs_path = bpy.path.abspath(props.profile_path)
            save_profile_entry(abs_path, props.profile_selection, data)
            self.report({"INFO"}, iface_("Saved to '{name}'").format(name=props.profile_selection))
        else:
            save_profile_entry(self.filepath, self.entry_name, data)
            props.profile_path = self.filepath
            props.profile_selection = self.entry_name
            self.report({"INFO"}, iface_("Saved to '{name}'").format(name=self.entry_name))
        redraw_all_areas(context)
        return {"FINISHED"}


classes = [
    REMOTE_OT_Connect,
    REMOTE_OT_CancelConnect,
    REMOTE_OT_Disconnect,
    REMOTE_OT_Abort,
    REMOTE_OT_StartServer,
    REMOTE_OT_CancelStartServer,
    REMOTE_OT_StopServer,
    SOLVER_OT_ForceTerminatePort,
    REMOTE_OT_OpenProfile,
    REMOTE_OT_ClearProfile,
    REMOTE_OT_ReloadProfile,
    REMOTE_OT_RefreshGpuDevices,
    REMOTE_OT_SaveProfile,
]
