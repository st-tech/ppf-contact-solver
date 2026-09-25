# File: addon_host_tests/_material_solver_units_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The one conversion from an authored material value to the solver's units.
#
# Three callers share it: a group's static parameter, a sampled keyframe, and a
# spatial map's target. A map blends between two of them, so a conversion that
# reached one end and not the other would blend across two different units.

import types

import pytest

from conftest import load_addon_module


@pytest.fixture(scope="module")
def mm():
    return load_addon_module("models.material_maps")


def make_group(object_type="SHELL", **overrides):
    """A stand-in for an ObjectGroup carrying only what the conversion reads."""
    group = types.SimpleNamespace(
        name="Group",
        object_type=object_type,
        shell_density=1000.0,
        solid_density=1000.0,
        rod_density=1000.0,
        young_mod_density_normalized=True,
        enable_inflate=True,
        enable_strain_limit=True,
        enable_plasticity=True,
        enable_bend_plasticity=True,
        shrink_x=1.0,
        shrink_y=1.0,
        pin_vertex_groups=[],
    )
    for key, value in overrides.items():
        setattr(group, key, value)
    return group


def captured_pull_pin():
    return types.SimpleNamespace(use_pull=True, has_captured_anim=True)


@pytest.mark.parametrize(
    "object_type,density_prop",
    [("SOLID", "solid_density"), ("SHELL", "shell_density"), ("ROD", "rod_density")],
)
def test_young_mod_normalizes_by_the_group_density(mm, object_type, density_prop):
    group = make_group(
        object_type, young_mod_density_normalized=False, **{density_prop: 1200.0}
    )
    assert mm.to_solver_value(group, "young-mod", 6000.0) == pytest.approx(5.0)


@pytest.mark.parametrize("object_type", ["SOLID", "SHELL", "ROD"])
def test_young_mod_is_untouched_when_the_field_is_already_normalized(mm, object_type):
    group = make_group(object_type, young_mod_density_normalized=True)
    assert mm.to_solver_value(group, "young-mod", 6000.0) == pytest.approx(6000.0)


@pytest.mark.parametrize("object_type", ["PDRD", "STATIC", "SAND"])
def test_young_mod_is_untouched_where_the_type_carries_no_density(mm, object_type):
    group = make_group(object_type, young_mod_density_normalized=False)
    assert mm.density_prop(object_type) is None
    assert mm.to_solver_value(group, "young-mod", 6000.0) == pytest.approx(6000.0)


def test_strain_limit_converts_percent_to_a_fraction(mm):
    group = make_group("SHELL")
    assert mm.to_solver_value(group, "strain-limit", 5.0) == pytest.approx(0.05)


def test_strain_limit_is_closed_by_its_own_checkbox(mm):
    group = make_group("SHELL", enable_strain_limit=False)
    assert mm.to_solver_value(group, "strain-limit", 5.0) is None
    assert "enable_strain_limit" in mm.gate_reason(group, "strain-limit")


def test_strain_limit_is_closed_by_a_shrunk_shell(mm):
    group = make_group("SHELL", shrink_x=0.9)
    assert mm.to_solver_value(group, "strain-limit", 5.0) is None
    assert "shrunk" in mm.gate_reason(group, "strain-limit")


def test_pressure_is_closed_by_its_own_checkbox(mm):
    group = make_group("SHELL", enable_inflate=False)
    assert mm.to_solver_value(group, "pressure", 300.0) is None
    assert mm.to_solver_value(make_group("SHELL"), "pressure", 300.0) == 300.0


def tracking_pin(**overrides):
    pin = types.SimpleNamespace(
        name="Rig", use_pull=False, has_captured_anim=True,
        track_rest_pose_deformation=True,
    )
    for key, value in overrides.items():
        setattr(pin, key, value)
    return pin


@pytest.mark.parametrize("object_type", ["SOLID", "SHELL"])
@pytest.mark.parametrize(
    "key", ["plasticity", "plasticity-threshold", "bend-plasticity",
            "bend-plasticity-threshold"]
)
def test_a_capture_alone_leaves_plasticity_open(mm, object_type, key):
    # A captured pin that does not track the rest shape streams nothing into
    # it, so plasticity is the artist's own setting and reaches the solver.
    group = make_group(object_type, pin_vertex_groups=[captured_pull_pin()])
    assert mm.to_solver_value(group, key, 0.5) == pytest.approx(0.5)
    assert mm.gate_reason(group, key) is None
    assert mm.rest_shape_plasticity_conflict(group) is None


@pytest.mark.parametrize("use_pull", [False, True])
def test_a_tracked_rest_shape_conflicts_with_plasticity(mm, use_pull):
    pin = tracking_pin(use_pull=use_pull)
    group = make_group("SOLID", pin_vertex_groups=[pin])
    assert mm.pin_tracks_rest_shape(group, pin)
    assert mm.rest_shape_plasticity_conflict(group) is pin
    assert mm.rest_shape_plasticity_conflict(
        make_group("SOLID", enable_plasticity=False, pin_vertex_groups=[pin])
    ) is None


@pytest.mark.parametrize(
    "object_type,overrides",
    [("SHELL", {}), ("SOLID", {"track_rest_pose_deformation": False}),
     ("SOLID", {"has_captured_anim": False})],
)
def test_only_a_tracking_solid_pin_tracks(mm, object_type, overrides):
    pin = tracking_pin(**overrides)
    group = make_group(object_type, pin_vertex_groups=[pin])
    assert not mm.pin_tracks_rest_shape(group, pin)
    assert mm.rest_shape_plasticity_conflict(group) is None


def test_an_uncaptured_pull_pin_leaves_plasticity_open(mm):
    pin = types.SimpleNamespace(use_pull=True, has_captured_anim=False)
    group = make_group("SHELL", pin_vertex_groups=[pin])
    assert mm.to_solver_value(group, "plasticity", 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "key", ["bend", "friction", "deformation-damping", "bending-damping",
            "bend-warp", "bend-weft"]
)
def test_an_ungated_unscaled_key_round_trips(mm, key):
    group = make_group("SHELL")
    assert mm.to_solver_value(group, key, 0.25) == pytest.approx(0.25)
    assert mm.gate_reason(group, key) is None


def test_a_solid_maps_only_the_parameters_its_elements_read(mm):
    solid_keys = {
        key for key, per_type in mm.MATERIAL_MAP_BASE_PROP.items()
        if "SOLID" in per_type
    }
    assert solid_keys == {"young-mod", "friction", "deformation-damping", "plasticity"}


def test_the_map_parameter_ids_are_permanent(mm):
    # Blender stores an enum's number, not its identifier, so a renumbering
    # repoints every saved map at a different parameter.
    assert {key: num for key, _label, _desc, num in mm.MATERIAL_MAP_KEYS} == {
        "young-mod": 1,
        "bend": 2,
        "friction": 3,
        "deformation-damping": 4,
        "bending-damping": 5,
        "pressure": 6,
        "strain-limit": 7,
        "plasticity": 8,
        "bend-plasticity": 9,
        "bend-warp": 10,
        "bend-weft": 11,
    }


def make_map(parameter, target, enabled=True):
    return types.SimpleNamespace(
        parameter=parameter, target_value=target, enabled=enabled
    )


def test_a_positive_slider_asks_for_anisotropic_bending(mm):
    group = make_group("SHELL", bend_warp=0.0, bend_weft=0.0, material_maps=[])
    assert mm.wants_anisotropic_bending(group) is False
    group.bend_warp = 300.0
    assert mm.wants_anisotropic_bending(group) is True


@pytest.mark.parametrize("key", ["bend-warp", "bend-weft"])
def test_a_map_target_asks_for_it_with_the_slider_at_zero(mm, key):
    # The natural way to paint anisotropy into a region. A predicate watching
    # only the sliders leaves this unwarned until the build refuses it.
    group = make_group(
        "SHELL", bend_warp=0.0, bend_weft=0.0,
        material_maps=[make_map(key, 600.0)],
    )
    assert mm.wants_anisotropic_bending(group) is True


def test_a_disabled_or_zero_map_asks_for_nothing(mm):
    group = make_group(
        "SHELL", bend_warp=0.0, bend_weft=0.0,
        material_maps=[make_map("bend-warp", 600.0, enabled=False)],
    )
    assert mm.wants_anisotropic_bending(group) is False
    group.material_maps = [make_map("bend-warp", 0.0)]
    assert mm.wants_anisotropic_bending(group) is False
    # A map on some other parameter is not a request for anisotropy either.
    group.material_maps = [make_map("bend", 5000.0)]
    assert mm.wants_anisotropic_bending(group) is False
