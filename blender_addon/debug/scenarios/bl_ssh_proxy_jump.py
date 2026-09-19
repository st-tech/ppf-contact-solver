# File: scenarios/bl_ssh_proxy_jump.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# ProxyJump support for every SSH-backed connection type: the solver host is
# reached through one or more jump hosts, named either by ~/.ssh/config or by
# -J on the SSH Command form. This scenario covers the whole path from the
# text a user types to the paramiko chain the backend opens:
#
#   A. core.ssh_config reads the ProxyJump keyword out of a config file and
#      resolves it into the ordered hop list, with a hop that carries a
#      ProxyJump of its own contributing its hops ahead of itself.
#   B. a jump loop and a malformed spec raise instead of resolving to a
#      connection against the wrong host.
#   C. core.ssh_command reads -J, and consumes each option's argument, so
#      `ssh -J me@bastion host` does not mistake the jump host for the
#      destination.
#   D. the facade turns a jump spec into the resolved chain it hands the
#      backend, preferring the panel field over the config entry.
#   E. core.backends opens the hops in order, tunnels each one through the
#      previous, forwards the last channel to the target, and closes what it
#      opened both on a mid-chain failure and on disconnect.
#   F. the panel draws the field for both field-entry SSH forms, and a
#      connection profile carries it.
#
# A and B and C are smoke checks here, enough to show the two modules behave
# under Blender's own Python. Their full tables (every spec form, every
# refusal, ssh's own option set) are host-side pytest, since they are pure
# text-to-parameters and need no Blender:
# ``addon_host_tests/_ssh_proxy_jump_.py``.
#
# Pure introspection scenario: no server, no solver, no transfer, and no
# network. The paramiko chain is exercised against a stub client, since a
# real one would need a bastion and a solver host.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)


_DRIVER_TEMPLATE = r"""
import inspect, os, shutil, sys, tempfile, time, traceback, types
import bpy
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


def raises(fn, *args, **kwargs):
    "Return the ValueError message, or None when the call did not raise."
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        return str(exc)
    return None


CONFIG_TEXT = '''
Host gpu-host
    HostName 192.0.2.5
    User ubuntu
    IdentityFile ~/.ssh/id_gpu
    ProxyJump inner

Host inner
    HostName inner.example.com
    Port 2202
    User inner-user
    IdentityFile ~/.ssh/id_inner
    ProxyJump outer

Host outer
    HostName outer.example.com
    Port 2201
    User outer-user

Host loop-a
    HostName a.example.com
    ProxyJump loop-b

Host loop-b
    HostName b.example.com
    ProxyJump loop-a

Host *
    User fallback-user
'''

tmpdir = tempfile.mkdtemp(prefix="proxyjump_")
config_path = os.path.join(tmpdir, "config")
with open(config_path, "w") as f:
    f.write(CONFIG_TEXT)

try:
    ssh_config = __import__(pkg + ".core.ssh_config",
                           fromlist=["resolve_ssh_config"])
    ssh_command = __import__(pkg + ".core.ssh_command",
                             fromlist=["parse_ssh_command"])
    backends = __import__(pkg + ".core.backends", fromlist=["create_backend"])
    client_mod = __import__(pkg + ".core.client", fromlist=["communicator"])
    profile_mod = __import__(pkg + ".core.profile",
                             fromlist=["read_connection_profile"])
    main_panel = __import__(pkg + ".ui.main_panel", fromlist=["classes"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])

    resolve = ssh_config.resolve_ssh_config
    chain = ssh_config.resolve_jump_chain
    parse = ssh_command.parse_ssh_command

    # ----- A: the config keyword and the resolved chain ---------------
    entry = resolve("gpu-host", 22, config_path)
    record("A_proxyjump_read_from_config",
           entry.proxy_jump == "inner" and entry.hostname == "192.0.2.5",
           {"proxy_jump": entry.proxy_jump, "hostname": entry.hostname})

    hops = chain("inner", 22, config_path)
    # "inner" jumps through "outer", so the chain reaches outward from this
    # machine: outer first, then inner, then the target the caller connects to.
    record("A_chain_is_ordered_outward",
           [h.hostname for h in hops] == ["outer.example.com", "inner.example.com"],
           {"hops": [(h.hostname, h.port, h.user) for h in hops]})

    record("A_hop_carries_its_own_config",
           hops[0].port == 2201 and hops[0].user == "outer-user"
           and hops[1].port == 2202 and hops[1].user == "inner-user"
           and hops[1].identity_file == os.path.expanduser("~/.ssh/id_inner"),
           {"hops": [(h.hostname, h.port, h.user, h.identity_file) for h in hops]})

    # ----- B: a spec that cannot be honored is refused ------------------
    loop_msg = raises(chain, "loop-a", 22, config_path)
    record("B_jump_loop_raises",
           loop_msg is not None and "loop" in loop_msg.lower(),
           {"message": loop_msg})

    record("B_malformed_spec_raises",
           raises(chain, "user@", 22, config_path) is not None)

    # ----- C: the SSH Command form -------------------------------------
    cmd = parse("ssh -J me@bastion -p 2222 -i ~/.ssh/id_gpu gpu-host")
    record("C_dash_j_parsed",
           cmd.host == "gpu-host" and cmd.proxy_jump == "me@bastion"
           and cmd.port == 2222
           and cmd.key_path == os.path.expanduser("~/.ssh/id_gpu"),
           {"host": cmd.host, "proxy_jump": cmd.proxy_jump, "port": cmd.port,
            "key_path": cmd.key_path})

    # The option argument holds an "@" and names no destination: reading it
    # as the destination is what a bare token scan does.
    record("C_jump_host_is_not_the_destination",
           parse("ssh -J me@bastion gpu-host").host == "gpu-host",
           {"host": parse("ssh -J me@bastion gpu-host").host})

    # ----- D: the facade resolves the chain it hands the backend --------
    com = client_mod.communicator
    Entry = ssh_config.SSHConfigEntry
    fake_hosts = {
        "gpu-host": Entry("192.0.2.5", 22, "ubuntu", "/keys/gpu", "bastion"),
        "bastion": Entry("bastion.example.com", 2201, "jump", "/keys/jump", None),
        "other": Entry("other.example.com", 2202, "other-user", None, None),
    }

    def fake_resolve(host, default_port=22, config_path=None):
        if host in fake_hosts:
            return fake_hosts[host]
        return Entry(host, default_port, None, None, None)

    captured = []
    real_resolve = ssh_config.resolve_ssh_config
    ssh_config.resolve_ssh_config = fake_resolve
    com._dispatch_and_tick = lambda event: captured.append(event)
    try:
        com.connect_ssh(host="gpu-host", port=22, username="", key_path="",
                        path="~/ppf-contact-solver", proxy_jump=None)
        from_config = captured[-1].config
        com.connect_ssh(host="gpu-host", port=22, username="", key_path="",
                        path="~/ppf-contact-solver", proxy_jump="other")
        from_field = captured[-1].config
        no_jump_msg = None
        try:
            com.connect_ssh(host="gpu-host", port=22, username="", key_path="",
                            path="~/ppf-contact-solver", proxy_jump="user@")
        except ValueError as exc:
            no_jump_msg = str(exc)
    finally:
        ssh_config.resolve_ssh_config = real_resolve
        del com._dispatch_and_tick

    record("D_config_proxyjump_is_used",
           from_config["jumps"] == [{"host": "bastion.example.com", "port": 2201,
                                     "username": "jump", "key_path": "/keys/jump"}],
           {"jumps": from_config["jumps"]})

    record("D_field_outranks_config",
           from_field["jumps"] == [{"host": "other.example.com", "port": 2202,
                                    "username": "other-user", "key_path": None}],
           {"jumps": from_field["jumps"]})

    record("D_bad_spec_reaches_the_caller",
           no_jump_msg is not None and len(captured) == 2,
           {"message": no_jump_msg, "dispatched": len(captured)})

    # ----- E: the paramiko chain ---------------------------------------
    events = []
    refuse = set()
    # Kept beside `events` rather than inside it, so the tuples the order and
    # credential checks below compare keep their shape.
    connect_kwargs = []
    channel_timeouts = []

    class FakeChannel:
        def __init__(self, source, dest):
            self.source = source
            self.dest = dest

    class FakeTransport:
        def __init__(self, client):
            self.client = client
            self.keepalive = None

        def set_keepalive(self, interval):
            self.keepalive = interval

        def is_active(self):
            return True

        # THE SIGNATURE IS PART OF WHAT THIS DOUBLE PROMISES. A stand-in that
        # does not take a parameter the real `paramiko.Transport.open_channel`
        # takes turns a correct call into a TypeError that only a rig run can
        # see: adding the handshake cap to the real call passed every unit test
        # (they hold a real paramiko) and failed here, on two CI legs at once.
        # The cap is asserted below rather than merely absorbed, so the next
        # change to it has to come through this scenario.
        def open_channel(self, kind=None, dest_addr=None, src_addr=None,
                         timeout=None):
            events.append(("channel", self.client.hostname, kind, tuple(dest_addr)))
            channel_timeouts.append(timeout)
            return FakeChannel(self.client.hostname, tuple(dest_addr))

    class FakeSSHClient:
        def __init__(self):
            self.hostname = None
            self.kwargs = None
            self.transport = FakeTransport(self)

        def set_missing_host_key_policy(self, policy):
            pass

        def connect(self, **kwargs):
            self.hostname = kwargs.get("hostname")
            self.kwargs = kwargs
            connect_kwargs.append(dict(kwargs))
            events.append(("connect", self.hostname, kwargs.get("port"),
                           kwargs.get("username"), kwargs.get("key_filename"),
                           getattr(kwargs.get("sock"), "dest", None)))
            if self.hostname in refuse:
                raise OSError("connection refused")

        def get_transport(self):
            return self.transport

        def close(self):
            events.append(("close", self.hostname))

    fake_paramiko = types.ModuleType("paramiko")
    fake_paramiko.SSHClient = FakeSSHClient
    fake_paramiko.AutoAddPolicy = object
    real_paramiko = sys.modules.get("paramiko")
    sys.modules["paramiko"] = fake_paramiko

    ssh_cfg = {
        "host": "192.0.2.5", "port": 22, "username": "ubuntu",
        "key_path": "/keys/gpu", "path": "~/ppf-contact-solver", "container": "",
        "keepalive_interval": 30, "server_port": 9090,
        "jumps": [
            {"host": "outer.example.com", "port": 2201,
             "username": "outer-user", "key_path": "/keys/outer"},
            {"host": "inner.example.com", "port": 2202,
             "username": "inner-user", "key_path": None},
        ],
    }
    try:
        backend = backends.create_backend("ssh", dict(ssh_cfg))
        connects = [e for e in events if e[0] == "connect"]
        channels = [e for e in events if e[0] == "channel"]
        record("E_hops_open_in_order",
               [c[1] for c in connects]
               == ["outer.example.com", "inner.example.com", "192.0.2.5"],
               {"connects": connects})

        record("E_each_hop_tunnels_the_next",
               [c[3] for c in channels]
               == [("inner.example.com", 2202), ("192.0.2.5", 22)]
               and [c[1] for c in channels]
               == ["outer.example.com", "inner.example.com"],
               {"channels": channels})

        # The first hop is reached directly; every later connect rides the
        # channel the previous hop opened.
        record("E_target_rides_the_last_channel",
               connects[0][5] is None
               and connects[1][5] == ("inner.example.com", 2202)
               and connects[2][5] == ("192.0.2.5", 22),
               {"socks": [c[5] for c in connects]})

        record("E_hop_credentials_are_its_own",
               connects[0][2:5] == (2201, "outer-user", "/keys/outer")
               and connects[1][2:5] == (2202, "inner-user", None)
               and connects[2][2:5] == (22, "ubuntu", "/keys/gpu"),
               {"connects": [c[1:5] for c in connects]})

        # Every hop and every forwarded channel carries the handshake cap.
        # An unbounded handshake does not present as a failure but as a hang:
        # a `connect` to a host that is powered off waits out the OS TCP retry
        # schedule with nothing on screen, and a bastion that accepts and then
        # forwards nothing holds the channel open just as long. The three
        # phases are named separately because paramiko caps them separately.
        cap = backends.SSH_HANDSHAKE_TIMEOUT_S
        hop_caps = [[k.get("timeout"), k.get("banner_timeout"),
                     k.get("auth_timeout")] for k in connect_kwargs]
        # `list(...)`, because the recorded detail has to be what was
        # ASSERTED. These two accumulate for the rest of section E (the
        # refused hop and the direct connect both add to them), and a detail
        # holding the live object reads three channel caps under a check that
        # tested two.
        record("E_every_hop_and_channel_is_time_capped",
               cap > 0
               and hop_caps == [[cap, cap, cap]] * 3
               and channel_timeouts == [cap, cap],
               {"cap": cap, "hop_caps": list(hop_caps),
                "channel_caps": list(channel_timeouts)})

        record("E_keepalive_reaches_every_hop",
               all(c.transport.keepalive == 30
                   for c in backend._jump_clients + [backend._instance]),
               {"keepalives": [c.transport.keepalive
                               for c in backend._jump_clients]})

        events.clear()
        backend.disconnect()
        record("E_disconnect_closes_the_chain",
               [e[1] for e in events]
               == ["192.0.2.5", "inner.example.com", "outer.example.com"],
               {"closed": [e[1] for e in events]})

        # A hop that refuses tears down what was already opened, and names
        # itself rather than surfacing as a bare socket error.
        events.clear()
        refuse.add("inner.example.com")
        failure = None
        try:
            backends.create_backend("ssh", dict(ssh_cfg))
        except Exception as exc:
            failure = str(exc)
        refuse.clear()
        record("E_failed_hop_is_named",
               failure is not None and "inner.example.com:2202" in failure,
               {"error": failure})
        record("E_failed_chain_is_torn_down",
               [e[1] for e in events if e[0] == "close"] == ["outer.example.com"]
               and not [e for e in events
                        if e[0] == "connect" and e[1] == "192.0.2.5"],
               {"events": events})

        # With no jump host the backend connects straight to the target, so
        # the default path is unchanged.
        events.clear()
        direct_cfg = dict(ssh_cfg)
        direct_cfg["jumps"] = []
        direct = backends.create_backend("ssh", direct_cfg)
        record("E_no_jump_connects_directly",
               [e[1] for e in events if e[0] == "connect"] == ["192.0.2.5"]
               and not [e for e in events if e[0] == "channel"]
               and direct._jump_clients == [],
               {"events": events})
        direct.disconnect()
    finally:
        if real_paramiko is None:
            sys.modules.pop("paramiko", None)
        else:
            sys.modules["paramiko"] = real_paramiko

    # ----- F: the panel field and the connection profile ----------------
    panel_source = inspect.getsource(main_panel)
    record("F_panel_draws_the_field",
           panel_source.count('col.prop(props, "proxy_jump")') == 2,
           {"rows": panel_source.count('col.prop(props, "proxy_jump")')})

    props = groups.get_addon_data(bpy.context.scene).ssh_state
    saved = props.proxy_jump
    try:
        props.proxy_jump = "me@bastion:2201"
        data = profile_mod.read_connection_profile(props)
        props.proxy_jump = ""
        applied = profile_mod.apply_profile(dict(data), props)
        record("F_profile_round_trip",
               applied and data.get("proxy_jump") == "me@bastion:2201"
               and props.proxy_jump == "me@bastion:2201",
               {"saved": data.get("proxy_jump"), "restored": props.proxy_jump})
    finally:
        props.proxy_jump = saved

    log("checks=" + str(len(result["checks"])) + " done")
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
