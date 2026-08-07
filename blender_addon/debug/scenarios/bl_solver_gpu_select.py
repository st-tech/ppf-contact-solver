# File: scenarios/bl_solver_gpu_select.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Acceptance gate for the solver GPU picker.
#
# The add-on starts the solver server on every backend, so on every backend the
# server's environment is out of the user's reach and there is nowhere to say
# which GPU to use. One mechanism covers all of them: Connect enumerates the
# solver host, the panel picks from that list, and Start Server applies the
# choice as CUDA_VISIBLE_DEVICES. Only the delivery differs, because Windows
# Native spawns a child process and the rest write a shell command. The device
# list this scenario works from is a recorded nvidia-smi answer, so every
# assertion holds on a machine with no NVIDIA driver, which is what lets the
# whole contract be checked on macOS as well as on a GPU host.
#
#   A. items_carry_device_numbers: the dropdown offers Automatic plus one entry
#      per device, and an entry's numeric ID is its CUDA index plus one. That
#      mapping is what keeps a saved .blend pointing at the device it named.
#   B. selection_roundtrips: picking an entry writes the CUDA index onto the
#      saved integer, and Automatic writes the sentinel.
#   C. automatic_preserves_inherited_env: Automatic sets nothing, so a
#      CUDA_VISIBLE_DEVICES exported before Blender started is inherited.
#   D. explicit_overrides_inherited_env: a picked device replaces an inherited
#      value rather than losing to it.
#   E. absent_device_is_named_and_refused: a selection the solver host cannot
#      satisfy stays visible in the dropdown and fails validation, and with no
#      device list at all validation stays silent (no evidence to contradict).
#   F. panel_draws_only_the_picker: the row carries the dropdown and the
#      refresh button and no status prose, is drawn only once connected, since
#      the device list is read from the solver host, and gets one line when
#      enumeration failed outright, which the dropdown cannot express.
#   G. start_applies_the_given_device: each start uses the device it is handed,
#      which is how Stop, pick another GPU, Start moves a running solver.
#   H. parser_rejects_empty_and_keeps_names: an empty nvidia-smi answer raises
#      instead of reading as "one GPU, all fine", and a comma inside a device
#      name does not shift the fields.
#   I. confirmation_only_reports_disagreement: the outcome row is silent when
#      the server is on the picked GPU, since Remote Hardware already names it,
#      and speaks up when they differ or the server is too old to say.
#   J. launch_is_logged: every server start writes which device it used to the
#      add-on console, so a session traced back later can name its GPU.
#   K. list_belongs_to_the_connection: the devices come from the solver host
#      and are dropped when the connection ends, so a later connection to a
#      different machine cannot inherit them.
#   L. shell_launch_carries_the_device: the backends started through a shell
#      get CUDA_VISIBLE_DEVICES in front of the server command.
#   M. connect_does_not_start_the_server: Windows Native connects without
#      launching, so the GPU can be picked before Start Server the same way it
#      is on every other backend.
#   N. win_native_waits_for_protocol: a Windows child is not reported ready
#      until it answers the solver protocol, and an early process exit is named.
#   O. failed_win_native_start_is_reaped: a child that never becomes ready is
#      stopped and cleared before the launch error returns control to the UI.
#   P. saved_uuid_survives_index_reorder: a saved choice follows the same
#      physical GPU when nvidia-smi assigns it a different index.
#   Q. probe_is_bounded: Connect and Refresh cap nvidia-smi so a stuck driver
#      cannot occupy the add-on's sole connection worker indefinitely.
#   R. uuid_overrides_automatic_index: a non-empty saved UUID is always a
#      selection, even if an older or inconsistent file pairs it with -1.
#
# Pure UI + logic scenario: no server, no solver, no transfer.

from __future__ import annotations


from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# No physics: the picker is add-on logic and panel drawing, so the emulated
# build exercises it exactly as the real one does.
BACKENDS = ("emulated",)


_DRIVER_TEMPLATE = r"""
import os, time, traceback
import bpy
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


class _Row:
    "Recording stand-in for a UILayout row."

    def __init__(self, sink):
        self.sink = sink

    def prop(self, data, name, **kw):
        self.sink.append(("prop", name))

    def operator(self, idname, **kw):
        self.sink.append(("operator", idname))

    def label(self, text="", **kw):
        self.sink.append(("label", text))


class _Layout(_Row):
    # Sub-layouts record into the same sink, and are layouts themselves, since
    # the panel nests a row inside a column.
    def row(self, **kw):
        return _Layout(self.sink)

    def column(self, **kw):
        return _Layout(self.sink)


# Two devices, one of them carrying a comma in its name, which is what a
# field-order assumption in the parser would trip over.
RECORDED = (
    "0, GPU-11111111-2222-3333-4444-555555555555, NVIDIA L40S\n"
    "1, GPU-66666666-7777-8888-9999-000000000000, NVIDIA RTX A4000, Inc.\n"
)


try:
    gpu = __import__(pkg + ".core.gpu_devices", fromlist=["load_devices"])
    panel = __import__(pkg + ".ui.main_panel", fromlist=["_draw_gpu_picker"])
    state_mod = __import__(pkg + ".ui.state", fromlist=["_get_solver_gpu_items"])
    backends = __import__(pkg + ".core.backends", fromlist=["WinNativeBackend"])
    conn = __import__(pkg + ".core.connection", fromlist=["spawn_win_native_server"])
    client = __import__(pkg + ".core.client", fromlist=["communicator"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])

    props = groups.get_addon_data(bpy.context.scene).ssh_state
    saved_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    saved_index = props.solver_gpu_index
    saved_uuid = props.solver_gpu_uuid

    devices = gpu.load_devices(RECORDED)
    build_items = state_mod._get_solver_gpu_items

    # ----- A: one entry per device, numbered by CUDA index -------------
    props.solver_gpu_index = gpu.AUTOMATIC
    props.solver_gpu_uuid = ""
    entries = build_items(props, bpy.context)
    got = [(entry[0], entry[4]) for entry in entries]
    expected = [("AUTO", 0), ("0", 1), ("1", 2)]
    names_ok = (len(entries) == 3
                and "L40S" in entries[1][1]
                and "A4000" in entries[2][1])
    record("A_items_carry_device_numbers", got == expected and names_ok,
           {"entries": [(e[0], e[1], e[4]) for e in entries],
            "parsed": [tuple(d) for d in devices]})

    # ----- B: the dropdown writes the saved CUDA index ------------------
    # Assigning by identifier goes through Blender's own resolution of the
    # dynamic items, so this also proves the registered property offers them.
    props.solver_gpu = "1"
    picked = props.solver_gpu_index
    picked_uuid = props.solver_gpu_uuid
    read_back = props.solver_gpu
    props.solver_gpu = "AUTO"
    auto = props.solver_gpu_index
    props.solver_gpu = "0"
    first = props.solver_gpu_index
    record("B_selection_roundtrips",
           picked == 1 and read_back == "1"
           and picked_uuid == devices[1].uuid
           and auto == gpu.AUTOMATIC and first == 0,
           {"picked": picked, "read_back": read_back,
            "picked_uuid": picked_uuid, "auto": auto, "first": first})

    # ----- C: Automatic leaves an inherited value alone -----------------
    inherited = gpu.apply_selection({"CUDA_VISIBLE_DEVICES": "3"}, gpu.AUTOMATIC)
    empty = gpu.apply_selection({}, gpu.AUTOMATIC)
    record("C_automatic_preserves_inherited_env",
           inherited.get("CUDA_VISIBLE_DEVICES") == "3"
           and "CUDA_VISIBLE_DEVICES" not in empty,
           {"inherited": inherited, "empty": empty})

    # ----- D: an explicit pick wins over an inherited value -------------
    overridden = gpu.apply_selection({"CUDA_VISIBLE_DEVICES": "3"}, 1)
    fresh = gpu.apply_selection({}, 0)
    record("D_explicit_overrides_inherited_env",
           overridden.get("CUDA_VISIBLE_DEVICES") == devices[1].uuid
           and fresh.get("CUDA_VISIBLE_DEVICES") == devices[0].uuid,
           {"overridden": overridden, "fresh": fresh})

    # ----- E: a device this machine lacks is named, and refused ---------
    props.solver_gpu_index = 5
    props.solver_gpu_uuid = "GPU-missing"
    stale = [(entry[0], entry[4]) for entry in build_items(props, bpy.context)]
    stale_listed = (
        ("MISSING", gpu.STALE_SELECTION_ID) in stale
        and props.solver_gpu == "MISSING"
    )
    props.solver_gpu_uuid = ""
    legacy_stale = props.solver_gpu == "MISSING"
    props.solver_gpu_uuid = "GPU-missing"
    refused = False
    try:
        gpu.validate_selection(5, props.solver_gpu_uuid)
    except ValueError as exc:
        refused = "5" in str(exc)
    # With nothing enumerated there is no evidence against the request, so
    # validation must stay silent rather than invent a device list.
    gpu.load_devices("")
    silent = True
    try:
        gpu.validate_selection(5, props.solver_gpu_uuid)
    except ValueError:
        silent = False
    probe_error_recorded = bool(gpu.gpu_probe_error())
    gpu.load_devices(RECORDED)
    record("E_absent_device_is_named_and_refused",
           stale_listed and legacy_stale
           and refused and silent and probe_error_recorded,
           {"stale_items": stale, "legacy_stale": legacy_stale,
            "refused": refused, "silent": silent,
            "probe_error": gpu.gpu_probe_error()})

    # ----- F: the row is always drawn, and names the environment --------
    # No status prose under the row, whatever is picked: the dropdown entry
    # already names the device, and a stale pick already reads "not detected".
    quiet = {}
    for pick in (gpu.AUTOMATIC, 1, 5):
        props.solver_gpu_index = pick
        props.solver_gpu_uuid = ""
        picked_sink = []
        panel._draw_gpu_picker(_Layout(picked_sink), props)
        quiet[pick] = picked_sink
    props.solver_gpu_index = 1
    props.solver_gpu_uuid = devices[1].uuid
    sink = quiet[1]
    drew_prop = ("prop", "solver_gpu") in sink
    drew_refresh = any(k == "operator" and v == "ssh.refresh_gpu_devices"
                       for k, v in sink)
    no_prose = all(not [v for k, v in rows if k == "label"]
                   for rows in quiet.values())
    # A failed probe is the one thing the dropdown cannot say, so it gets a line.
    gpu.record_probe_failure("nvidia-smi failed on the solver host: boom")
    failed_sink = []
    panel._draw_gpu_picker(_Layout(failed_sink), props)
    says_failure = any(k == "label" and "boom" in v for k, v in failed_sink)
    gpu.load_devices(RECORDED)
    # The whole section is hidden while disconnected: the list is read from
    # the solver host, so there is nothing to offer before there is one.
    props.solver_gpu_index = 1
    props.solver_gpu_uuid = ""
    original_connected = client.communicator.__class__.is_connected
    offline, online = [], []
    try:
        client.communicator.__class__.is_connected = lambda _self: False
        panel._draw_gpu_section(_Layout(offline), props)
        client.communicator.__class__.is_connected = lambda _self: True
        panel._draw_gpu_section(_Layout(online), props)
    finally:
        client.communicator.__class__.is_connected = original_connected
    record("F_panel_draws_only_the_picker",
           drew_prop and drew_refresh and no_prose and says_failure
           and offline == []
           and ("prop", "solver_gpu") in online,
           {"quiet": quiet, "failed": failed_sink,
            "offline": offline, "online": online})

    # ----- G: a Stop/Start cycle re-launches on the same GPU ------------
    seen = {}

    def _fake_spawn(
        root, port, cuda_device=gpu.AUTOMATIC, cuda_device_uuid=""
    ):
        seen["args"] = (root, port, cuda_device, cuda_device_uuid)
        return None

    original_spawn = conn.spawn_win_native_server
    conn.spawn_win_native_server = _fake_spawn
    try:
        backend = backends.WinNativeBackend(
            directory="C:\\dev", port=59999, process=None)
        # is_alive() would probe a socket; the point here is the launch
        # argument, so hold the backend in the stopped state directly.
        backend.is_alive = lambda: False
        backend.start_server(1)
        first = seen.get("args")
        backend.start_server(3)
        moved = seen.get("args")
        backend.start_server()
        default = seen.get("args")
    finally:
        conn.spawn_win_native_server = original_spawn
    record("G_start_applies_the_given_device",
           first == ("C:\\dev", 59999, 1, "")
           and moved == ("C:\\dev", 59999, 3, "")
           and default == ("C:\\dev", 59999, gpu.AUTOMATIC, ""),
           {"first": first, "moved": moved, "default": default})

    # ----- I: the outcome row reads the server, not the intent ----------
    # Connect attaches when a server already holds the port, so the selection
    # can reach nothing. What the panel reports then has to come from the
    # server's own answer.
    original_response = client.communicator.__class__.response
    cases = {}

    def _confirmation_text(hardware, selected):
        props.solver_gpu_index = selected
        sink = []
        client.communicator.__class__.response = property(lambda _self: {"hardware": hardware})
        try:
            panel._draw_gpu_confirmation(_Layout(sink), props)
        finally:
            client.communicator.__class__.response = original_response
        return " | ".join(v for k, v in sink if k == "label")

    agreed = _confirmation_text({"GPU Index": 2, "GPU": "2: NVIDIA L4"}, 2)
    disagreed = _confirmation_text({"GPU Index": 0, "GPU": "0: NVIDIA L4"}, 2)
    under_auto = _confirmation_text({"GPU Index": 3, "GPU": "3: NVIDIA L4"}, gpu.AUTOMATIC)
    silent_server = _confirmation_text({"GPU": "NVIDIA L4"}, 2)
    no_device = _confirmation_text(
        {"GPU Index": -1, "GPU": "CUDA_VISIBLE_DEVICES is set to '9', which selects no GPU"}, 9)
    cases = {
        "agreed": agreed,
        "disagreed": disagreed,
        "under_auto": under_auto,
        "silent_server": silent_server,
        "no_device": no_device,
    }
    # The GPU Index row feeds the comparison but is never listed: the GPU row
    # already leads with the same number.
    hidden = panel._UNDISPLAYED_HARDWARE_KEYS
    cases["hidden_rows"] = sorted(hidden)
    record("I_confirmation_only_reports_disagreement",
           "GPU Index" in hidden
           and agreed == "" and under_auto == ""
           and "not the selected GPU 2" in disagreed
           and "Stop Server" in disagreed
           and "does not report" in silent_server
           and "selects no GPU" in no_device,
           {"cases": cases})

    # ----- J: every server start is recorded in the console -------------
    # A session read back later has no panel to consult, so the device has to
    # be in the log.
    launched_named = gpu.describe_launch(1, True)
    launched_auto = gpu.describe_launch(gpu.AUTOMATIC, True)
    attached = gpu.describe_launch(1, False)
    record("J_launch_is_logged",
           "GPU 1" in launched_named
           and devices[1].uuid in launched_named
           and "Automatic" in launched_auto
           and "attached" in attached and devices[1].uuid not in attached,
           {"launched_named": launched_named, "launched_auto": launched_auto,
            "attached": attached})

    # ----- H: the parser refuses an empty answer, and keeps names -------
    raised = False
    try:
        gpu.parse_nvidia_smi_devices("\n\n")
    except gpu.GpuProbeError:
        raised = True
    comma_name = devices[1].name if len(devices) > 1 else ""
    record("H_parser_rejects_empty_and_keeps_names",
           raised and comma_name == "NVIDIA RTX A4000, Inc."
           and devices[1].uuid.startswith("GPU-6666"),
           {"raised": raised, "name": comma_name})

    # ----- K: the list belongs to the connection ------------------------
    # It is read from the solver host, so it must not outlive the connection:
    # the next one may reach a different machine.
    gpu.forget_devices()
    after_forget = (len(gpu.cached_gpu_devices()), gpu.has_probed(),
                    gpu.gpu_probe_error())
    # Automatic, so the list holds nothing but Automatic: a stored index would
    # legitimately add its own "not detected" entry, which subtest E covers.
    props.solver_gpu_index = gpu.AUTOMATIC
    empty_items = [e[0] for e in build_items(props, bpy.context)]
    gpu.record_probe_failure("nvidia-smi failed on the solver host: boom")
    failure = (gpu.gpu_probe_error(), len(gpu.cached_gpu_devices()),
               gpu.has_probed())
    gpu.load_devices(RECORDED)
    record("K_list_belongs_to_the_connection",
           after_forget == (0, False, "")
           and empty_items == ["AUTO"]
           and failure[0].endswith("boom") and failure[1] == 0 and failure[2],
           {"after_forget": after_forget, "empty_items": empty_items,
            "failure": failure})

    # ----- Q: the solver-host probe carries a hard timeout ---------------
    probe_call = {}

    class _ProbeBackend:
        backend_type = "win_native"

        def exec_command(self, command, **kwargs):
            probe_call["command"] = command
            probe_call.update(kwargs)
            return {
                "exit_code": 0,
                "stdout": RECORDED.splitlines(),
                "stderr": [],
            }

    runner = client.communicator._runner
    runner._probe_solver_host_gpus(_ProbeBackend())
    record("Q_probe_is_bounded",
           probe_call.get("timeout") == gpu.PROBE_TIMEOUT_SECONDS
           and probe_call.get("shell") is False
           and tuple(probe_call.get("command", ())) == gpu.NVIDIA_SMI_ARGS,
           probe_call)

    # ----- L: a shell-started server gets the variable on its command ---
    prefixes = (gpu.shell_prefix(gpu.AUTOMATIC), gpu.shell_prefix(1))
    launched_script = {}

    class _FakeBackend:
        backend_type = "ssh"
        server_port = 9090
        current_directory = "/home/u/ppf-contact-solver"

        def exec_command(self, command, shell=False, cwd=None, timeout=None):
            if "start_server.sh\n" in command:
                launched_script["script"] = command
            # Fail the script invocation so the launch stops here, with the
            # command already recorded, instead of waiting on a server that
            # will never come up.
            return {"exit_code": 1, "stdout": [], "stderr": []}

    saved_backend = runner._backend
    runner._backend = _FakeBackend()
    try:
        runner._do_launch_server(1)
    except Exception:
        pass
    finally:
        runner._backend = saved_backend
    script = launched_script.get("script", "")
    record("L_shell_launch_carries_the_device",
           prefixes == ("", f"CUDA_VISIBLE_DEVICES={devices[1].uuid} ")
           and f"CUDA_VISIBLE_DEVICES={devices[1].uuid} /home/u/ppf-contact-solver/target/release/ppf-cts-server" in script,
           {"prefixes": prefixes, "script": script[:400]})

    # ----- M: connecting does not start the server ----------------------
    # The GPU is picked between Connect and Start Server, which requires
    # Connect to leave the server stopped.
    spawned = []
    original_spawn2 = conn.spawn_win_native_server
    conn.spawn_win_native_server = lambda *a, **k: spawned.append(a) or None
    try:
        info, process = conn.connect_win_native(<<REPO_ROOT_REPR>>, 59997)
    finally:
        conn.spawn_win_native_server = original_spawn2
    record("M_connect_does_not_start_the_server",
           not spawned and process is None
           and info.server_running is False
           and info.type == "win_native",
           {"spawned": spawned, "server_running": info.server_running})

    # ----- N: Windows waits for a protocol-ready child ------------------
    class _ExitedProcess:
        def poll(self):
            return 23

    class _WaitingBackend:
        backend_type = "win_native"
        server_port = 59996

        def __init__(self, process):
            self._process = process

    saved_backend = runner._backend
    original_probe = conn._probe_ppf_cts_server
    ready = False
    exit_error = ""
    try:
        runner._backend = _WaitingBackend(None)
        conn._probe_ppf_cts_server = lambda *a, **k: True
        runner._wait_for_win_native_server(timeout=0.01)
        ready = True

        runner._backend = _WaitingBackend(_ExitedProcess())
        conn._probe_ppf_cts_server = lambda *a, **k: False
        try:
            runner._wait_for_win_native_server(timeout=0.01)
        except RuntimeError as exc:
            exit_error = str(exc)
    finally:
        conn._probe_ppf_cts_server = original_probe
        runner._backend = saved_backend
    record("N_win_native_waits_for_protocol",
           ready and "exited with code 23" in exit_error,
           {"ready": ready, "exit_error": exit_error})

    # ----- O: a failed Windows launch cannot strand its child ------------
    class _FailedLaunchBackend:
        backend_type = "win_native"
        server_port = 59995

        def __init__(self):
            self._process = object()
            self.stopped = False

        def start_server(self, cuda_device, cuda_device_uuid=""):
            return True

        def stop_server(self):
            self.stopped = True
            self._process = None

    class _CaptureEngine:
        def __init__(self):
            self.events = []

        def dispatch(self, event):
            self.events.append(event)

    saved_backend = runner._backend
    saved_engine = runner._engine
    saved_wait = runner._wait_for_win_native_server
    failed_backend = _FailedLaunchBackend()
    capture = _CaptureEngine()

    def _fail_wait():
        raise TimeoutError("not ready")

    try:
        runner._backend = failed_backend
        runner._engine = capture
        runner._wait_for_win_native_server = _fail_wait
        runner._do_launch_server(1)
    finally:
        runner._wait_for_win_native_server = saved_wait
        runner._engine = saved_engine
        runner._backend = saved_backend
    launch_errors = [
        event for event in capture.events
        if type(event).__name__ == "ErrorOccurred"
    ]
    record("O_failed_win_native_start_is_reaped",
           failed_backend.stopped and failed_backend._process is None
           and len(launch_errors) == 1
           and "not ready" in launch_errors[0].error,
           {"stopped": failed_backend.stopped,
            "event_types": [type(event).__name__ for event in capture.events]})

    # ----- P: a saved UUID follows the physical card across reorder -------
    gpu.load_devices(RECORDED)
    props.solver_gpu = "1"
    stable_uuid = props.solver_gpu_uuid
    reordered = (
        f"0, {devices[1].uuid}, {devices[1].name}\n"
        f"1, {devices[0].uuid}, {devices[0].name}\n"
    )
    gpu.load_devices(reordered)
    reordered_identifier = props.solver_gpu
    reordered_index = gpu.selected_device(
        props.solver_gpu_index, props.solver_gpu_uuid
    ).index
    record("P_saved_uuid_survives_index_reorder",
           stable_uuid == devices[1].uuid
           and reordered_identifier == "0"
           and reordered_index == 0
           and gpu.selection_token(
               props.solver_gpu_index, props.solver_gpu_uuid
           ) == devices[1].uuid,
           {"stable_uuid": stable_uuid,
            "identifier": reordered_identifier,
            "resolved_index": reordered_index})

    # ----- R: a UUID is authoritative even beside the Automatic index ----
    props.solver_gpu_index = gpu.AUTOMATIC
    props.solver_gpu_uuid = "GPU-missing"
    inconsistent_items = [
        (entry[0], entry[4]) for entry in build_items(props, bpy.context)
    ]
    inconsistent_refused = False
    try:
        gpu.validate_selection(
            props.solver_gpu_index, props.solver_gpu_uuid
        )
    except ValueError:
        inconsistent_refused = True
    record("R_uuid_overrides_automatic_index",
           ("MISSING", gpu.STALE_SELECTION_ID) in inconsistent_items
           and props.solver_gpu == "MISSING"
           and inconsistent_refused
           and gpu.shell_prefix(
               props.solver_gpu_index, props.solver_gpu_uuid
           ) == "CUDA_VISIBLE_DEVICES=GPU-missing "
           and gpu.apply_selection(
               {}, props.solver_gpu_index, props.solver_gpu_uuid
           ).get("CUDA_VISIBLE_DEVICES") == "GPU-missing",
           {"items": inconsistent_items,
            "refused": inconsistent_refused})

    props.solver_gpu_index = saved_index
    props.solver_gpu_uuid = saved_uuid
    if saved_env is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved_env

    log("checks=" + str(len(result["checks"])) + " done")
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    # repr() so Windows backslashes survive into the driver source intact.
    return _DRIVER_TEMPLATE.replace("<<REPO_ROOT_REPR>>", repr(REPO_ROOT_POSIX))


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
