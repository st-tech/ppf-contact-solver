# File: addon_host_tests/_violation_overlay_dismissal_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The violation overlay of a refused Transfer is taken down by an edit that
# moves the geometry it points at, and stays down.
#
# A refused build reports its violations as positions in the scene as it was
# uploaded, and the viewport draws them. Resolve Intersections then moves the
# objects, so the highlights mark places the surfaces have left
# (``ViolationsDismissed``). The server knows nothing of the edit: it holds
# the upload it refused and reports that upload's violations on every status
# response, so clearing the client's list once lasts until the next poll.
# The dismissal is therefore keyed on the upload id.
#
# Covers, on the pure state machine:
#   * a refused upload's violations are mirrored into the state;
#   * dismissing them empties the list, records the upload, and redraws;
#   * the next status response for the SAME upload, carrying the violations
#     again, leaves the list empty, and so does one carrying the error alone;
#   * the next upload's violations are shown;
#   * with nothing uploaded, a dismissal records no upload, so a server whose
#     responses carry an empty upload id is not silenced.

from __future__ import annotations

import pytest

from conftest import load_addon_module


@pytest.fixture(scope="module")
def machine():
    state = load_addon_module("core.state")
    transitions = load_addon_module("core.transitions")
    events = load_addon_module("core.events")
    effects = load_addon_module("core.effects")
    protocol = load_addon_module("core.protocol")
    return state, transitions, events, effects, protocol


CROSSING = [{"type": "self_intersection", "count": 2, "pairs": []}]
LEFT_OVER = [{"type": "self_intersection", "count": 1, "pairs": []}]
REFUSAL = "ValidationError: 2 self-intersections (2 tri-tri)."


def refused(protocol, upload_id, violations=None):
    response = {
        "status": protocol.STATUS_NO_BUILD,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "upload_id": upload_id,
        "error": REFUSAL,
    }
    if violations is not None:
        response["violations"] = violations
    return response


def online(state_mod):
    return state_mod.AppState(
        phase=state_mod.Phase.ONLINE, server=state_mod.Server.RUNNING)


def poll(machine, state, response):
    _state_mod, transitions, events, _effects, _protocol = machine
    new_state, _sent = transitions.transition(
        state, events.ServerPolled(response=response))
    return new_state


def test_a_refused_upload_shows_its_violations(machine):
    state_mod, _transitions, _events, _effects, protocol = machine
    state = poll(machine, online(state_mod), refused(protocol, "A", CROSSING))
    assert state.violations == CROSSING
    assert state.server_upload_id == "A"


def test_dismissing_empties_the_list_records_the_upload_and_redraws(machine):
    state_mod, transitions, events, effects, protocol = machine
    state = poll(machine, online(state_mod), refused(protocol, "A", CROSSING))
    state, sent = transitions.transition(state, events.ViolationsDismissed())
    assert state.violations == []
    assert state.violations_dismissed_for == "A"
    assert any(isinstance(e, effects.DoRedrawUI) for e in sent)
    # The refusal itself is the server's verdict on what it holds, and stays.
    assert state.server_error == REFUSAL


def test_the_same_upload_reported_again_stays_dismissed(machine):
    state_mod, transitions, events, _effects, protocol = machine
    state = poll(machine, online(state_mod), refused(protocol, "A", CROSSING))
    state, _sent = transitions.transition(state, events.ViolationsDismissed())
    # With the violations in the response, and with the error alone.
    for response in (refused(protocol, "A", CROSSING), refused(protocol, "A")):
        state = poll(machine, state, response)
        assert state.violations == []
        assert state.server_error == REFUSAL


def test_the_next_upload_shows_its_own(machine):
    state_mod, transitions, events, _effects, protocol = machine
    state = poll(machine, online(state_mod), refused(protocol, "A", CROSSING))
    state, _sent = transitions.transition(state, events.ViolationsDismissed())
    state = poll(machine, state, refused(protocol, "B", LEFT_OVER))
    assert state.violations == LEFT_OVER


def test_a_dismissal_with_nothing_uploaded_silences_nothing(machine):
    state_mod, transitions, events, _effects, protocol = machine
    state, _sent = transitions.transition(
        online(state_mod), events.ViolationsDismissed())
    assert state.violations_dismissed_for == ""
    state = poll(machine, state, refused(protocol, "", CROSSING))
    assert state.violations == CROSSING
