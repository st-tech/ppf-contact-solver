"""The add-on half of the server's idle stop, in the parts that need no Blender.

The server stops itself after `--idle-timeout` seconds with no request and
nothing running, and a connected, idle add-on keeps it up by pinging. What is
pinned here is the add-on's spelling of both: the flag every launch passes, the
one test of a ping too slow to keep a server up, the ping thread polling only
a running server it is connected to, and the idle time reaching the launch
effect from Start Server.
`bl_server_ping_keeps_it_up` drives the same code against a real server.
"""

from dataclasses import replace

import pytest

from conftest import load_addon_module


@pytest.fixture(scope="module")
def connection():
    return load_addon_module("core.connection")


@pytest.fixture(scope="module")
def machine():
    state = load_addon_module("core.state")
    transitions = load_addon_module("core.transitions")
    events = load_addon_module("core.events")
    effects = load_addon_module("core.effects")
    return state, transitions, events, effects


def test_every_launch_spells_the_flag_one_way(connection):
    assert connection.idle_timeout_args(None) == []
    assert connection.idle_timeout_args(0) == ["--idle-timeout", "0"]
    assert connection.idle_timeout_args(3600) == ["--idle-timeout", "3600"]


@pytest.mark.parametrize("idle, ping, outlasts", [
    (3600, 10, False),
    (3, 1, False),
    (3, 3, True),
    (3, 5, True),
    (0, 3600, False),   # 0 never stops, so no ping is too slow
])
def test_a_ping_too_slow_is_one_test(connection, idle, ping, outlasts):
    assert connection.ping_outlasts_idle_stop(idle, ping) is outlasts


def test_the_ping_thread_polls_a_running_server_and_stops_when_offline(machine):
    # The real runner and its real ping thread, against a stand-in engine and
    # backend: the poll it queues is counted where the worker would send it.
    import time

    state_mod, _, _, _ = machine
    runner_mod = load_addon_module("core.effect_runner")

    class Engine:
        state = state_mod.AppState(phase=state_mod.Phase.ONLINE,
                                   server=state_mod.Server.RUNNING)

        def dispatch(self, _event):
            pass

    engine = Engine()
    runner = runner_mod.EffectRunner(engine)
    sent = []
    try:
        runner._do_query = lambda request: sent.append(request)
        runner._backend = object()
        runner.ping_interval_s = 0.1
        time.sleep(1.2)
        online = len(sent)
        assert online >= 3, f"{online} pings in 1.2 s"
        assert all(request == {} for request in sent), "a ping is a status poll"
        engine.state = replace(engine.state, phase=state_mod.Phase.OFFLINE)
        time.sleep(0.4)
        settled = len(sent)
        time.sleep(1.0)
        assert len(sent) == settled, "pinged while offline"
        engine.state = replace(engine.state, phase=state_mod.Phase.ONLINE,
                               server=state_mod.Server.UNKNOWN)
        time.sleep(1.0)
        assert len(sent) == settled, "pinged a server that is not running"
    finally:
        runner.stop()


def test_start_server_carries_the_idle_time_to_the_launch(machine):
    state_mod, transitions, events, effects = machine
    online = state_mod.AppState(phase=state_mod.Phase.ONLINE)
    _, sent = transitions.transition(online, events.StartServerRequested(idle_timeout=3))
    launches = [e for e in sent if isinstance(e, effects.DoLaunchServer)]
    assert len(launches) == 1 and launches[0].idle_timeout == 3
