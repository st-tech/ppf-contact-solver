# File: addon_host_tests/_build_verdict_survives_polls_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A verdict is read the same way every time the server reports it.
#
# The server reports a refused build as ``NO_BUILD`` carrying the error, and
# it says so on every status response until an upload replaces the payload.
# The add-on turns that into ``Solver.FAILED``. Responses keep arriving after
# the add-on has stopped waiting for the build: the ping polls a connected,
# idle server, and a poll that was already in flight lands behind the one that
# ended the build. Each of those carries the same refusal and has to read the
# same way, or the panel falls back to waiting for a build that was refused
# and a caller waiting for the verdict never sees it.
#
# Covers, on the pure state machine:
#   * a refused build reads FAILED, and so does the same response again;
#   * the refusal is written to the Console once, however often it is
#     reported, and written again when the same payload is refused again;
#   * the next upload, which carries no error, ends the verdict;
#   * a refusal about another upload does not inherit it;
#   * a crashed run's report is written once, and a different report again.

from __future__ import annotations

import pytest

from conftest import load_addon_module


@pytest.fixture(scope="module")
def machine():
    state = load_addon_module("core.state")
    transitions = load_addon_module("core.transitions")
    events = load_addon_module("core.events")
    protocol = load_addon_module("core.protocol")
    return state, transitions, events, protocol


REFUSAL = "ValidationError: 2 self-intersections (2 tri-tri)."
CRASH = "Solver crashed: the linear solve did not converge."


def response(protocol, status, upload_id, error=""):
    answer = {
        "status": status,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "upload_id": upload_id,
    }
    if error:
        answer["error"] = error
    return answer


def poll(machine, state, answer):
    _state_mod, transitions, events, _protocol = machine
    new_state, effects = transitions.transition(
        state, events.ServerPolled(response=answer))
    logged = [e.message for e in effects if type(e).__name__ == "DoLog"]
    return new_state, logged


def building(state_mod, upload_id="old"):
    """Where the add-on stands once its build request has gone out."""
    return state_mod.AppState(
        phase=state_mod.Phase.ONLINE,
        server=state_mod.Server.RUNNING,
        solver=state_mod.Solver.BUILDING,
        activity=state_mod.Activity.BUILDING,
        server_upload_id=upload_id,
        active_upload_id="",
    )


def refused_once(machine):
    state_mod, _transitions, _events, protocol = machine
    state, _ = poll(machine, building(state_mod),
                    response(protocol, protocol.STATUS_BUILDING, "new"))
    return poll(machine, state,
                response(protocol, protocol.STATUS_NO_BUILD, "new", REFUSAL))


def test_a_refused_build_reads_failed_and_ends_the_activity(machine):
    state_mod, _transitions, _events, _protocol = machine
    state, logged = refused_once(machine)
    assert state.solver == state_mod.Solver.FAILED
    assert state.activity == state_mod.Activity.IDLE
    assert state.server_error == REFUSAL
    assert REFUSAL in logged


def test_the_same_response_again_still_reads_failed(machine):
    # The poll that was in flight behind the verdict, and every ping after it.
    state_mod, _transitions, _events, protocol = machine
    state, _ = refused_once(machine)
    for _ in range(3):
        state, _ = poll(machine, state, response(
            protocol, protocol.STATUS_NO_BUILD, "new", REFUSAL))
        assert state.solver == state_mod.Solver.FAILED
        assert state.activity == state_mod.Activity.IDLE
        assert state.server_error == REFUSAL


def test_a_refusal_is_written_once_however_often_it_is_reported(machine):
    _state_mod, _transitions, _events, protocol = machine
    state, _ = refused_once(machine)
    for _ in range(3):
        state, logged = poll(machine, state, response(
            protocol, protocol.STATUS_NO_BUILD, "new", REFUSAL))
        assert logged == []


def test_the_next_upload_ends_the_verdict(machine):
    # An upload drops the error of the build it invalidates, so the response
    # that names the new upload carries none.
    state_mod, _transitions, _events, protocol = machine
    state, _ = refused_once(machine)
    state, _ = poll(machine, state, response(
        protocol, protocol.STATUS_NO_BUILD, "newer"))
    assert state.solver == state_mod.Solver.NO_BUILD
    assert state.server_error == ""


def test_a_refusal_about_another_upload_does_not_inherit_the_verdict(machine):
    # The verdict is this upload's. A refusal the add-on did not build for,
    # under an id it has not read as failed, is not promoted by it.
    state_mod, _transitions, _events, protocol = machine
    state, _ = refused_once(machine)
    state, _ = poll(machine, state, response(
        protocol, protocol.STATUS_NO_BUILD, "someone-elses", REFUSAL))
    assert state.solver == state_mod.Solver.NO_BUILD


def test_the_same_payload_refused_again_is_written_again(machine):
    # Pressing Build on the refused payload is a new build with a new verdict,
    # which happens to say the same thing.
    state_mod, transitions, events, protocol = machine
    state, _ = refused_once(machine)
    state, _effects = transitions.transition(state, events.BuildRequested())
    assert state.activity == state_mod.Activity.BUILDING
    state, logged = poll(machine, state, response(
        protocol, protocol.STATUS_NO_BUILD, "new", REFUSAL))
    assert state.solver == state_mod.Solver.FAILED
    assert REFUSAL in logged


def test_a_crashed_run_is_written_once_and_a_new_report_again(machine):
    state_mod, _transitions, _events, protocol = machine
    state = state_mod.AppState(
        phase=state_mod.Phase.ONLINE,
        server=state_mod.Server.RUNNING,
        solver=state_mod.Solver.RUNNING,
        server_upload_id="u",
    )
    state, logged = poll(machine, state, response(
        protocol, protocol.STATUS_FAILED, "u", CRASH))
    assert state.solver == state_mod.Solver.FAILED
    assert logged == [CRASH]
    for _ in range(3):
        state, logged = poll(machine, state, response(
            protocol, protocol.STATUS_FAILED, "u", CRASH))
        assert state.solver == state_mod.Solver.FAILED
        assert logged == []
    other = "Solver crashed: an intersection was detected."
    state, logged = poll(machine, state, response(
        protocol, protocol.STATUS_FAILED, "u", other))
    assert logged == [other]
