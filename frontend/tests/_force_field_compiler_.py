# File: _force_field_compiler_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The force-field script compiler, field objects and their reference
# evaluator, and the session writer, without a solver.
#
# The compiler's contract has two halves, and each is pinned here: what it
# ACCEPTS it compiles to bytecode that computes what the Python computes (run
# through the reference interpreter, the kernel's semantics in float64), and
# what it REFUSES it refuses by line, before anything reaches a solver. The
# solver's own loader re-verifies the bytecode (crates/ppf-cts-solver/src/
# force_field.rs has those tests); these are the frontend's.

import math
import os
import tomllib

import numpy as np
import pytest

from frontend._field_eval_ import evaluate
from frontend._force_field_ import (
    SCRIPT_VERSION,
    ForceField,
    ForceFieldScriptError,
    compile_script,
    make_field,
    resolve_targets,
    run_bytecode,
    track_array,
    write_session,
)

SWIRL = """
import math

def eval(x, y, z, t):
    r = math.sqrt(x * x + z * z) + 1e-6
    s = 2.0 * math.sin(3.0 * t)
    if r > 1.5:
        return (0.0, 0.0, 0.0)
    acc = 0.0
    for i in range(3):
        acc += i * 0.5
    a, b = x ** 2, (y ** -2 if y != 0 else 0.0)
    k = 1.0 if 0 < x < 0.5 and not y > 0 else -1.0
    return (-z / r * s + acc, 0.5 * k + max(a, 0.1, b * 0),
            x / r * s + (x // 0.3) + x % 0.7 + abs(z) ** 1.5)
"""


def _python(source):
    ns = {"math": math}
    exec(source, ns)
    return ns["eval"]


def test_bytecode_computes_what_the_python_computes():
    compiled = compile_script(SWIRL)
    fn = _python(SWIRL)
    rng = np.random.default_rng(3)
    for _ in range(200):
        x, y, z = rng.uniform(-2.0, 2.0, 3)
        t = rng.uniform(0.0, 5.0)
        want = fn(x, y, z, t)
        got = run_bytecode(compiled.code, compiled.constants, x, y, z, t)
        assert np.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_z_up_maps_blender_axes_both_ways():
    # A Blender script sees (x, y, z) Z-up and returns a Z-up vector; the
    # solver feeds it Y-up positions and wants a Y-up answer.
    compiled = compile_script(
        "def eval(x, y, z, t):\n    return (x + 10.0 * y, 2.0 * z, 0.0)\n", z_up=True)
    xb, yb, zb = 0.3, -0.7, 1.1
    solver_point = (xb, zb, -yb)
    ax, ay, az = run_bytecode(compiled.code, compiled.constants, *solver_point, 0.0)
    # Blender answer (xb + 10 yb, 2 zb, 0) in solver axes is (bx, bz, -by).
    assert np.allclose((ax, ay, az), (xb + 10.0 * yb, 0.0, -2.0 * zb))


def test_small_integer_powers_are_exact_for_negative_bases():
    compiled = compile_script("def eval(x, y, z, t):\n    return (x ** 3, x ** -2, x ** 0)\n")
    got = run_bytecode(compiled.code, compiled.constants, -2.0, 0.0, 0.0, 0.0)
    assert np.allclose(got, (-8.0, 0.25, 1.0))


def test_a_conditional_return_of_two_tuples_compiles():
    src = "def eval(x, y, z, t):\n    return (1.0, 0.0, 0.0) if x > 0 else (-1.0, 0.0, 0.0)\n"
    compiled = compile_script(src)
    assert run_bytecode(compiled.code, compiled.constants, 1.0, 0, 0, 0)[0] == 1.0
    assert run_bytecode(compiled.code, compiled.constants, -1.0, 0, 0, 0)[0] == -1.0


@pytest.mark.parametrize("source, line, fragment", [
    ("def eval(x, y, z, t):\n    import os\n    return (0, 0, 0)\n", 2, "Import"),
    ("def eval(x, y, z, t):\n    while x:\n        x = 0\n    return (0, 0, 0)\n", 2, "While"),
    ("def eval(x, y, z, t):\n    if x > 0:\n        q = 1.0\n    return (q, 0, 0)\n", 4, "before it is assigned"),
    ("def eval(x, y, z, t):\n    if x > 0:\n        return (1, 0, 0)\n", 1, "without returning"),
    ("def eval(x, y, z, t):\n    return (os.sep, 0, 0)\n", 2, "os.sep"),
    ("def eval(x, y, z, t):\n    return (x, y)\n", 2, "three numbers"),
    ("def eval(x, y, z):\n    return (x, y, z)\n", 1, "four plain arguments"),
    ("def eval(x, y, z, t):\n    return (math.gamma(x), 0, 0)\n", 2, "math.gamma"),
    ("def eval(x, y, z, t)\n    return 1\n", 1, "syntax error"),
    ("def eval(x, y, z, t):\n    for i in range(3):\n        return (1, 0, 0)\n    return (0, 0, 0)\n", 2, "inside a loop"),
    ("def eval(x, y, z, t):\n    n = int(x)\n    for i in range(n):\n        x += 1\n    return (x, 0, 0)\n", 2, "int(x)"),
])
def test_unsupported_constructs_are_refused_by_line(source, line, fragment):
    with pytest.raises(ForceFieldScriptError) as info:
        compile_script(source)
    assert info.value.lineno == line
    assert fragment in str(info.value)


def test_a_function_without_readable_source_is_refused_by_name():
    ns = {}
    exec("def eval(x, y, z, t):\n    return (0.0, 0.0, 0.0)\n", ns)
    with pytest.raises(ForceFieldScriptError, match="cannot be read"):
        compile_script(ns["eval"])


def test_a_field_is_validated_into_per_sample_arrays():
    f = make_field("vortex", strength=[0.0, 1.0, 2.0], step=0.5, min_distance=0.3)
    assert f["strength"].shape == (3,) and f["position"].shape == (3, 3)
    assert f["axes"].shape == (3, 3, 3) and f["use_min"] and not f["use_max"]
    with pytest.raises(ValueError, match="step="):
        make_field("force", strength=[1.0, 2.0])
    with pytest.raises(ValueError, match="not a rotation"):
        make_field("force", axes=np.eye(3) * 2.0)
    with pytest.raises(ValueError, match="size above 0"):
        make_field("turbulence", size=0.0)
    with pytest.raises(ValueError, match="needs image="):
        make_field("texture")
    with pytest.raises(ValueError, match="only a texture field takes image"):
        make_field("force", image=np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="not one of"):
        make_field("boid")
    with pytest.raises(ValueError, match="track samples and another setting has"):
        make_field("force", power=[0.0, 1.0, 2.0], strength=[1.0, 2.0], step=0.1)


def test_the_session_writer_lays_out_what_the_solver_reads(tmp_path):
    field = ForceField()
    field.add("wind", position=[(0.0, 1.0, 0.0), (0.5, 1.0, 0.0)], step=0.04,
              seed=7, groups=["a"])
    field.script("def eval(x, y, z, t):\n    return (0.0, 1.0, 0.0)\n")
    field.script("def eval(x, y, z, t):\n    return (1.0, 0.0, 0.0)\n", groups=["b", "a"])
    weight = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    fields, scripts, mask = resolve_targets(field, {"a": [0], "b": [2]}, 3)
    write_session(str(tmp_path), fields, scripts, weight, mask)
    base = tmp_path / "bin" / "force_field"
    manifest = tomllib.loads((base / "force_field.toml").read_text())
    (f,) = manifest["field"]
    assert f["type"] == "wind" and f["seed"] == 7 and f["step"] == 0.04
    origin = np.fromfile(base / f["origin"], dtype="<f8").reshape(-1, 3)
    assert np.array_equal(origin, [[0.0, 1.0, 0.0], [0.5, 1.0, 0.0]])
    track = np.fromfile(base / f["track"], dtype="<f4").reshape(-1, 28)
    assert track.shape == (2, 28)
    # The axes' columns first: a new Blender field's local Z is solver +Y.
    assert np.array_equal(track[0, 6:9], [0.0, 1.0, 0.0])
    assert np.array_equal(track_array(fields[0][0]), track)
    assert [sc["targets"] for sc in manifest["script"]] == [32, 1]
    assert f["targets"] == 0
    assert all(sc["version"] == SCRIPT_VERSION for sc in manifest["script"])
    code = np.fromfile(base / manifest["script"][1]["code"], dtype="<u4")
    assert len(code) == len(field.scripts[1][0].code)
    assert np.array_equal(np.fromfile(base / "weight.bin", dtype="<f4"), weight)
    # Vertex 0 is in "a" (bits 0 and 1), vertex 2 in "b" (bit 1), vertex 1 in none.
    assert list(np.fromfile(base / "targets.bin", dtype="<u4")) == [0b11, 0, 0b10]


# Values MEASURED on Blender 5.2 as a black box,
# with the field's local frame equal to the world's.
IDENTITY = np.eye(3)


def accel(kind, points, velocities=None, **settings):
    points = np.asarray(points, dtype=float)
    if velocities is None:
        velocities = np.zeros_like(points)
    return evaluate(make_field(kind, axes=IDENTITY, **settings), points, velocities)


def test_the_reference_evaluator_reproduces_the_falloff_measurements():
    d = np.array([[0.25, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]])

    def norm(a):
        return list(np.round(np.linalg.norm(a, axis=1), 5))

    assert norm(accel("force", d, power=2)[0]) == [0.64, 0.44444, 0.25, 0.11111, 0.04]
    assert norm(accel("force", d, power=2, min_distance=1.0)[0]) == [
        1.0, 1.0, 1.0, 0.25, 0.0625]
    assert norm(accel("force", d, max_distance=1.5)[0]) == [1.0, 1.0, 1.0, 0.0, 0.0]
    p = np.array([[1, 0, 1], [2, 0, 0.5], [0, 0, -1], [1, 1, -2]], dtype=float)
    tube = accel("force", p, falloff="tube", power=1, radial_power=2)[0]
    assert np.allclose(np.linalg.norm(tube, axis=1)[[0, 2]], [0.125, 0.5], atol=1e-4)
    cone = accel("force", p, falloff="cone", power=1, radial_power=1)[0]
    # The probe printed these components to four decimals.
    assert np.allclose(cone, [[0.0077, 0, 0.0077], [0.0084, 0, 0.0021], [0, 0, -0.0028],
                              [0.0009, 0.0009, -0.0019]], atol=5e-5)
    zpos = accel("force", p, power=1, z_direction="positive")[0]
    assert np.all(zpos[2:] == 0.0) and np.all(np.linalg.norm(zpos[:2], axis=1) > 0.0)
    grav = accel("force", np.array([[0, 2, 0], [1, 0, 1], [2, 0, -1]], dtype=float),
                 gravitation=True)[0]
    assert np.allclose(np.linalg.norm(grav, axis=1), [0.25, 0.5, 0.2])


def test_the_reference_evaluator_reproduces_the_type_measurements():
    p = np.array([[1, 0, 0], [0, 2, 0], [1, 0, 1], [2, 0, -1]], dtype=float)
    vortex = accel("vortex", p)[0]
    assert np.allclose(vortex, [[0, 1, 0], [-2, 0, 0], [0, 1.4142, 0], [0, 2.2361, 0]],
                       atol=1e-4)
    plane = accel("vortex", p, shape="plane", strength=2.0)[0]
    assert np.allclose(plane, [[-4, 2, 0], [-4, -8, 0], [-4, 2, 0], [-8, 4, 0]], atol=1e-4)
    plane_p1 = accel("vortex", p, shape="plane", strength=2.0, power=1)[0]
    assert np.allclose(plane_p1[1], [-1.3333, -0.8889, 0], atol=1e-4)
    v = np.tile([0.0, 2.0, 0.0], (4, 1))
    magnet = accel("magnetic", p[:2], v[:2], strength=3.0, power=1)[0]
    assert np.allclose(magnet, [[0, 0, 0], [0, 0, 2]], atol=1e-6)
    magnet_plane = accel("magnetic", p, np.tile([1.0, 0, 0], (4, 1)), shape="plane")[0]
    assert np.allclose(magnet_plane, [[0, -1, 0]] * 4)
    plane_force = accel("force", p, shape="plane", power=1)[0]
    assert np.allclose(plane_force[:, 2], [1.0, 1.0, 0.5, -0.5])
    wind_a, wind_air = accel("wind", p, shape="plane")
    assert np.all(wind_a == 0.0) and np.allclose(wind_air, [[0, 0, 1]] * 4)


def test_the_track_interpolates_and_holds_its_ends():
    f = make_field("force", axes=IDENTITY, shape="plane", strength=[1.0, 3.0], step=1.0,
                   position=[(0, 0, 0), (0, 0, 2)])
    point = np.array([[0.0, 0.0, 1.5]])
    # Halfway, the origin is at z = 1 and the strength 2: the point is above.
    assert np.allclose(evaluate(f, point, time=0.5)[0], [[0, 0, 2.0]])
    assert np.allclose(evaluate(f, point, time=-1.0)[0], [[0, 0, 1.0]])
    assert np.allclose(evaluate(f, point, time=9.0)[0], [[0, 0, -3.0]])


def test_targets_are_resolved_by_set_and_refused_when_unknown():
    field = ForceField()
    field.script("def eval(x, y, z, t):\n    return (0.0, 1.0, 0.0)\n", groups=["a", "b"])
    field.script("def eval(x, y, z, t):\n    return (0.0, 2.0, 0.0)\n", groups=["b", "a"])
    _, scripts, mask = resolve_targets(field, {"a": [0], "b": [1]}, 2)
    assert [bit for _, bit in scripts] == [0, 0]  # one set, one bit
    assert list(mask) == [1, 1]
    field.script("def eval(x, y, z, t):\n    return (0.0, 3.0, 0.0)\n", groups=["c"])
    with pytest.raises(ValueError, match="no object"):
        resolve_targets(field, {"a": [0], "b": [1]}, 2)
    with pytest.raises(ValueError, match="reach nothing"):
        ForceField().add("force", groups=[])
    everyone = ForceField().script("def eval(x, y, z, t):\n    return (0.0, 0.0, 0.0)\n")
    assert resolve_targets(everyone, {}, 4)[2] is None


def test_noise_builtins_compile_and_match_the_reference():
    from frontend._noise_ import curl_noise, noise

    src = (
        "def eval(x, y, z, t):\n"
        "    n = noise(x, y, z, octaves=3, seed=4)\n"
        "    cx, cy, cz = curl_noise(x, y, z, 2, 7)\n"
        "    return (cx + n, cy, cz)\n"
    )
    compiled = compile_script(src)
    got = run_bytecode(compiled.code, compiled.constants, 0.3, -0.2, 1.7, 0.0)
    c = curl_noise(0.3, -0.2, 1.7, 2, 7)
    assert np.allclose(got, (c[0] + noise(0.3, -0.2, 1.7, 3, 4), c[1], c[2]))
    returned = compile_script("def eval(x, y, z, t):\n    return curl_noise(x, y, z, seed=3)\n")
    assert np.allclose(run_bytecode(returned.code, returned.constants, 0.2, 0.3, 0.4, 0.0),
                       curl_noise(0.2, 0.3, 0.4, 1, 3))


def test_evolving_and_decaying_noise_compiles_and_matches_the_reference():
    from frontend._noise_ import curl_noise, noise

    src = (
        "def eval(x, y, z, t):\n"
        "    n = noise(x, y, z, octaves=2, seed=4, time=t, frequency=1.5, decay=0.3)\n"
        "    cx, cy, cz = curl_noise(x, y, z, 3, 7, t, 0.5, 2.0)\n"
        "    return (cx + n, cy, cz)\n"
    )
    compiled = compile_script(src)
    for t in (0.0, 0.4, 1.3):
        got = run_bytecode(compiled.code, compiled.constants, 0.3, -0.2, 1.7, t)
        c = curl_noise(0.3, -0.2, 1.7, 3, 7, time=t, frequency=0.5, decay=2.0)
        n = noise(0.3, -0.2, 1.7, 2, 4, time=t, frequency=1.5, decay=0.3)
        assert np.allclose(got, (c[0] + n, c[1], c[2]))
    returned = compile_script(
        "def eval(x, y, z, t):\n    return curl_noise(x, y, z, time=t, decay=1.0)\n")
    assert np.allclose(run_bytecode(returned.code, returned.constants, 0.2, 0.3, 0.4, 0.7),
                       curl_noise(0.2, 0.3, 0.4, time=0.7, decay=1.0))


def test_the_pattern_evolves_in_place_and_decays():
    from frontend._noise_ import noise

    rng = np.random.default_rng(2)
    p = rng.uniform(-3, 3, (2000, 3)).T
    still = noise(*p, time=0.0)
    # Evolving changes the pattern without sliding it: the field one time
    # unit later is uncorrelated with the one before, but a short step apart
    # it is nearly the same.
    near = noise(*p, time=0.02, frequency=1.0)
    far = noise(*p, time=3.0, frequency=1.0)
    assert np.corrcoef(still, near)[0, 1] > 0.99
    assert abs(np.corrcoef(still, far)[0, 1]) < 0.2
    # frequency 0 freezes it; decay scales it by exp(-decay * time).
    assert np.allclose(noise(*p, time=5.0, frequency=0.0), still)
    assert np.allclose(noise(*p, time=2.0, frequency=0.0, decay=0.5),
                       still * np.exp(-1.0))


@pytest.mark.parametrize("source, fragment", [
    ("def eval(x, y, z, t):\n    return (noise(x, y, z, frequency=2.0), 0, 0)\n", "needs time"),
    ("def eval(x, y, z, t):\n    return (noise(x, y, z, decay=1.0), 0, 0)\n", "needs time"),
    ("def eval(x, y, z, t):\n    return (noise(x, y, z, speed=1.0), 0, 0)\n", "'speed'"),
    ("def eval(x, y, z, t):\n    return (curl_noise(x, y, z), 0, 0)\n", "three numbers"),
    ("def eval(x, y, z, t):\n    return (noise(x, y, z, octaves=9), 0, 0)\n", "1 to 8"),
    ("def eval(x, y, z, t):\n    return (noise(x, y, z, octaves=t), 0, 0)\n", "1 to 8"),
    ("def eval(x, y, z, t):\n    a, b = curl_noise(x, y, z)\n    return (a, b, 0)\n", "three names"),
])
def test_noise_misuse_is_refused(source, fragment):
    with pytest.raises(ForceFieldScriptError, match=fragment):
        compile_script(source)


def test_curl_noise_is_divergence_free():
    from frontend._noise_ import curl_noise

    rng = np.random.default_rng(5)
    p = rng.uniform(-3, 3, (100, 3))
    h = 1e-5
    div = np.zeros(len(p))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        hi = np.array(curl_noise(*(p + e).T, 2, 1))[k]
        lo = np.array(curl_noise(*(p - e).T, 2, 1))[k]
        div += (hi - lo) / (2 * h)
    assert np.abs(div).max() < 1e-6


def test_every_listed_builtin_compiles_and_nothing_else_does(capsys):
    from frontend import _script_api_ as api
    from frontend._force_field_ import ForceField

    args = {"sqrt": "(x * x + 1.0)", "log": "(x * x + 1.0)", "asin": "(0.5)",
            "acos": "(0.5)"}
    for name, sig, _ in api.MATH:
        arity = sig.count(",") + 1
        arg = args.get(name, "(x)") if arity == 1 else "(x, 2.0)"
        compile_script(f"import math\n\ndef eval(x, y, z, t):\n"
                       f"    return (math.{name}{arg}, 0.0, 0.0)\n")
    for name, _, _ in api.MATH_CONSTANTS:
        compile_script(f"import math\n\ndef eval(x, y, z, t):\n"
                       f"    return (math.{name}, 0.0, 0.0)\n")
    compile_script("def eval(x, y, z, t):\n"
                   "    a = 0.0\n"
                   "    for i in range(3):\n"
                   "        a = a + abs(min(x, y)) + max(x, y, z) + float(i)\n"
                   "    n = noise(x, y, z)\n"
                   "    cx, cy, cz = curl_noise(x, y, z)\n"
                   "    return (a + n + cx, cy, cz)\n")
    with pytest.raises(ForceFieldScriptError, match="built-in functions are noise, curl_noise"):
        compile_script("def eval(x, y, z, t):\n    return (round(x), 0.0, 0.0)\n")
    ForceField.builtins()
    printed = capsys.readouterr().out
    assert all(sig in printed for _, entries in api.SECTIONS for _, sig, _ in entries)


def test_the_add_on_carries_byte_identical_copies_of_the_noise_and_the_laws():
    # The add-on cannot import the frontend, so it carries copies; Visualize
    # draws what the solver computes only while they stay identical.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for ours, theirs in (("frontend/_noise_.py", "blender_addon/core/noise.py"),
                         ("frontend/_field_eval_.py", "blender_addon/core/field_eval.py")):
        with open(os.path.join(root, ours), "rb") as a, open(os.path.join(root, theirs), "rb") as b:
            assert a.read() == b.read(), f"{theirs} is no longer a copy of {ours}"


def test_noise_amount_gusts_are_divergence_free_bounded_and_drift_with_a_wind():
    rng = np.random.default_rng(5)
    p = rng.uniform(-1.0, 1.0, (40, 3))
    settings = dict(shape="plane", strength=3.0, noise=0.5, noise_scale=(0.4, 0.4, 0.4),
                    seed=9)
    plain = make_field("wind", axes=IDENTITY, shape="plane", strength=3.0)
    noisy = make_field("wind", axes=IDENTITY, **settings)
    gust = evaluate(noisy, p)[1] - evaluate(plain, p)[1]
    assert np.all(evaluate(noisy, p)[0] == 0.0)
    assert 0.1 < float(np.abs(gust).max()) < 10.0 * 0.5 * 3.0
    # Divergence-free at a uniform scale: the curl of noise potentials.
    h = 1e-4
    div = sum((evaluate(noisy, p + h * e)[1][:, k] - evaluate(noisy, p - h * e)[1][:, k])
              / (2 * h) for k, e in enumerate(np.eye(3)))
    assert float(np.abs(div).max()) < 1e-5 * float(np.abs(gust).max()) / 0.4
    # Carried by the flow: the gust at time t is the one at time 0, displaced
    # downstream by strength * t along the wind's axis.
    t = 0.37
    later = evaluate(noisy, p, time=t)[1] - evaluate(plain, p, time=t)[1]
    upstream = evaluate(noisy, p - np.array([0.0, 0.0, 3.0 * t]))[1] - \
        evaluate(plain, p - np.array([0.0, 0.0, 3.0 * t]))[1]
    assert np.allclose(later, upstream, atol=1e-12)
    # And a field that is not a Wind keeps its pattern in place.
    force = make_field("force", axes=IDENTITY, shape="plane", noise=0.5, seed=9)
    assert np.allclose(evaluate(force, p, time=t)[0], evaluate(force, p)[0])


def test_wind_factor_reproduces_the_cloth_measurements():
    # A Plane Force along Z on a cloth facing it, at 45 degrees and edge-on
    # (normals in the field's frame), against Blender's cloth: at Wind Factor
    # 1 only the component along the normal remains, at 0.5 the even blend.
    s45 = math.sqrt(0.5)
    normals = np.array([[0, 0, 1], [0, -s45, s45], [0, 1, 0], [0, 0, 0]], dtype=float)
    p = np.array([[0.0, 0.0, 1.0]] * 4)
    want = {1.0: [[0, 0, 1], [0, -0.5, 0.5], [0, 0, 0], [0, 0, 1]],
            0.5: [[0, 0, 1], [0, -0.25, 0.75], [0, 0, 0.5], [0, 0, 1]]}
    for wf, expected in want.items():
        f = make_field("force", axes=IDENTITY, shape="plane", wind_factor=wf)
        got = evaluate(f, p, normals=normals)[0]
        # The last vertex presents no surface, so its push stays whole.
        assert np.allclose(got, expected, atol=1e-12), (wf, got)
    with pytest.raises(ValueError, match="needs the vertices' normals"):
        evaluate(make_field("force", wind_factor=1.0), p)
    with pytest.raises(ValueError, match="takes no wind_factor"):
        make_field("wind", wind_factor=1.0)


def test_the_decoder_refuses_an_add_on_still_sending_grids():
    # An add-on older than this solver sends sampled grids; reading past the
    # key would leave its fields doing nothing, unsaid.
    from frontend import App
    from frontend._decoder_ import ParamDecoder

    decoder = ParamDecoder()
    decoder._data = {"force_field": {"grids": [{"kind": "acceleration"}], "scripts": []}}
    scene = App.create("decoder_grid_refusal").scene.create()
    with pytest.raises(ValueError, match="no longer takes"):
        decoder.apply_force_field(scene)
    decoder._data = {"force_field": {"fields": [{"type": "wind", "strength": 2.0,
                                                 "groups": None}], "scripts": []}}
    decoder.apply_force_field(scene)
    assert [f["type"] for f in scene.force_field.fields] == ["wind"]


def test_the_velocity_laws_reproduce_the_measurements():
    # Drag, Harmonic and a Vortex's damping, from a known velocity, against
    # the values measured on Blender.
    p = np.array([[1, 0, 0], [0, 2, 0]], dtype=float)
    v = np.tile([0.0, 2.0, 0.0], (2, 1))
    drag = accel("drag", p, v, linear_drag=1.0, quadratic_drag=0.0)[0]
    assert np.allclose(drag, [[0, -4, 0], [0, -4, 0]])  # Linear Drag multiplies |v| v
    drag = accel("drag", p, v, linear_drag=0.0, quadratic_drag=1.0)[0]
    assert np.allclose(drag, [[0, -2, 0], [0, -2, 0]])
    spring = accel("harmonic", p, v, strength=4.0, harmonic_damping=0.5)[0]
    assert np.allclose(spring, [[-4, -4, 0], [0, -12, 0]])
    repel = accel("harmonic", p, v, strength=-4.0, harmonic_damping=0.5)[0]
    assert np.allclose(repel, [[4, -4, 0], [0, 4, 0]])  # sqrt(|s|)
    rest = accel("harmonic", np.array([[0, 2, 0], [1, 0, 1]], dtype=float),
                 rest_length=1.0)[0]
    assert np.allclose(rest, [[0, -1, 0], [-0.29289, 0, -0.29289]], atol=1e-4)
    plane = accel("harmonic", np.array([[1, 0, 1], [0, 2, -0.5]], dtype=float),
                  np.tile([0.0, 0.0, 1.0], (2, 1)), strength=4.0, shape="plane", power=1)[0]
    assert np.allclose(plane[:, 2], [-6.0, -2.6667], atol=1e-4)
    # A Plane Vortex damps the velocity with Inflow 0; Inflow adds to it.
    q = np.array([[1, 0, 1], [0, 2, -0.5], [3, 0, 0], [0.5, 0, 0]], dtype=float)
    vq = np.tile([0.0, 2.0, 0.0], (4, 1))
    for inflow, power, want in ((0.0, 1, [1.0, 1.0, 1.0, 1.0]),
                                (2.0, 1, [2.0, 1.6667, 1.5, 2.3333]),
                                (1.0, 2, [1.25, 1.1111, 1.0625, 1.4444])):
        a = accel("vortex", q, vq, shape="plane", strength=0.0, inflow=inflow, power=power)[0]
        assert np.allclose(-a[:, 1] / 2.0, want, atol=1e-4), (inflow, power, a)
    cut = accel("vortex", q, vq, shape="plane", strength=0.0, inflow=1.0, power=1,
                max_distance=2.5)[0]
    assert cut[2, 1] == 0.0
    point = accel("vortex", p, v, strength=0.0, inflow=1.0, power=1)[0]
    assert np.allclose(point[:, 1], [-1.0, -0.6667], atol=1e-4)


def test_the_image_texture_reproduces_the_measurements():
    # A 4x4 image, columns 0 and 1 black, 2 and 3 white, read nearest, as
    # measured on Blender.
    step = np.zeros((4, 4, 3))
    step[:, 2:] = 1.0
    nearest = dict(image=step, interpolation=False)
    at = np.array([[-0.5, 0.1, 0.0], [0.5, 0.1, 0.0]])
    rgb = accel("texture", at, **nearest)[0]
    assert np.allclose(rgb, [[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
    assert np.allclose(accel("texture", at, strength=2.0, **nearest)[0], 2.0 * rgb)
    # Forward differences 0.25 apart across the step at x = -0.1.
    edge = np.array([[-0.1, 0.1, 0.0]])
    grad = accel("texture", edge, texture_mode="gradient", nabla=0.25, **nearest)[0]
    assert np.allclose(grad, [[-4.0, 0.0, 0.0]])
    green = np.zeros((4, 4, 3))
    green[:, 2:, 1] = 1.0
    curl = accel("texture", edge, texture_mode="curl", nabla=0.25,
                 **dict(nearest, image=green))[0]
    assert np.allclose(curl, [[0.0, 0.0, 4.0]])
    grad_g = accel("texture", edge, texture_mode="gradient", nabla=0.25,
                   **dict(nearest, image=green))[0]
    assert np.allclose(grad_g, [[-4.0 / 3.0, 0.0, 0.0]])
    # World coordinates by default: moving the field changes nothing; with
    # object coordinates the image follows it.
    moved = accel("texture", at, position=(1.0, 0.0, 0.0), **nearest)[0]
    assert np.allclose(moved, rgb)
    follows = accel("texture", at, position=(1.0, 0.0, 0.0), object_coordinates=True,
                    **nearest)[0]
    assert np.allclose(follows, [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    # Extensions outside [-1, 1]: clamp, color 0, wrap; clip_cube also in z.
    gray = np.tile(np.array([0.0, 1 / 3, 2 / 3, 1.0])[None, :, None], (4, 1, 3))
    out = np.array([[1.2, 0.1, 0], [1.6, 0.1, 0], [-1.4, 0.1, 0], [2.6, 0.1, 0]])
    for ext, want in (("extend", [1.0, 1.0, 0.0, 1.0]), ("clip", [0, 0, 0, 0]),
                      ("repeat", [0.0, 1 / 3, 1.0, 1.0])):
        a = accel("texture", out, extension=ext, **dict(nearest, image=gray))[0]
        assert np.allclose(0.5 - a[:, 0], want), (ext, a)
    z = np.array([[0.1, 0.1, 0.5], [0.1, 0.1, 1.5]])
    cube = accel("texture", z, extension="clip_cube", **dict(nearest, image=np.ones((4, 4, 3))))[0]
    assert np.allclose(0.5 - cube[:, 0], [1.0, 0.0])
    # The falloff applies, in the field's frame.
    red = np.zeros((4, 4, 3))
    red[..., 0] = 1.0
    fall = accel("texture", np.array([[0.5, 0.1, 0.0]]), power=1.0,
                 **dict(nearest, image=red))[0]
    assert np.allclose(fall, [[-0.3312, 0.3312, 0.3312]], atol=2e-4)


# --- fluid volumes -----------------------------------------------------------

from frontend._field_eval_ import FLUID_EMPTY, fluid_at, fluid_bricks  # noqa: E402
from frontend._force_field_ import FluidVolume  # noqa: E402

# Measured on Blender 5.2: a 16^3 gas domain
# spanning [-1, 1], the eight cells around a particle at P, their velocities
# in m/s (the grid's units times 2.5 * cell), and the acceleration Blender's
# Fluid Flow field gave the particle at Strength 1, Flow 0.
MEASURED_P = (-0.2725, 0.0925, 0.0425)
MEASURED_CELLS = {
    (5, 8, 7): (1.3790665566921234, -0.007542700041085482, -0.2523529902100563),
    (6, 8, 7): (0.8763257414102554, -0.009664656827226281, -0.1857936941087246),
    (5, 9, 7): (1.4708778262138367, 0.2504746802151203, -0.22281261160969734),
    (6, 9, 7): (0.9522947669029236, 0.16467614099383354, -0.176386758685112),
    (5, 8, 8): (1.3954004645347595, -0.007641594856977463, -0.007641639094799757),
    (6, 8, 8): (0.8794601261615753, -0.009664741810411215, -0.00966471154242754),
    (5, 9, 8): (1.4775988459587097, 0.2503391169011593, -0.00785330543294549),
    (6, 9, 8): (0.9637552499771118, 0.16555659472942352, -0.010368686635047197),
}
MEASURED_A = (1.24839947555362, 0.04727130686818749, -0.043118868288447)
# The same domain, Use Smoke Density on, Flow 1, a particle at rest-frame
# velocity (0.5, 0, 0) at the center of cell (5, 8, 8), density 0.8799.
MEASURED_RHO = 0.8799306154251099
MEASURED_A_DENSITY = (0.7878899750386918, -0.0067240727414635435, -0.006724112759231316)


def _measured_volume(rho_at=None):
    u = np.zeros((16, 16, 16, 3))
    rho = np.zeros((16, 16, 16))
    for cell, value in MEASURED_CELLS.items():
        u[cell] = value
    if rho_at is not None:
        rho[rho_at] = MEASURED_RHO
    return FluidVolume([u], cell=0.125, corner=(-1.0, -1.0, -1.0), density=[rho])


def _fluid_field(volume, **kw):
    kw.setdefault("axes", np.eye(3))
    return make_field("fluid", fluid=volume, **kw)


def test_the_fluid_flow_law_reproduces_the_measurements():
    # Flow 0: the push is the fluid's velocity, trilinear between centers.
    f = _fluid_field(_measured_volume())
    a, _ = evaluate(f, [MEASURED_P], velocities=[[0.3, -0.2, 0.1]])
    assert np.allclose(a[0], MEASURED_A, atol=2e-6)
    # Use Smoke Density and Flow 1: s rho (U - v).
    center = (-1.0 + 0.125 * 5.5, -1.0 + 0.125 * 8.5, -1.0 + 0.125 * 8.5)
    f = _fluid_field(_measured_volume(rho_at=(5, 8, 8)), flow=1.0, use_density=True)
    a, _ = evaluate(f, [center], velocities=[[0.5, 0.0, 0.0]])
    assert np.allclose(a[0], MEASURED_A_DENSITY, atol=2e-6)
    # Flow 1 without density: s (U - v) inside, and NOTHING outside the
    # domain, the damping included (measured: 0 at (1.3, 0, 0) moving at
    # (0.5, 0.2, 0); (0.8954, -0.20764, -0.00764) at cell (5, 8, 8)).
    f = _fluid_field(_measured_volume(), flow=1.0)
    a, _ = evaluate(f, [[1.3, 0.0, 0.0], center], velocities=[[0.5, 0.2, 0.0]] * 2)
    assert np.all(a[0] == 0.0)
    assert np.allclose(a[1], (0.8954, -0.20764, -0.00764), atol=1e-5)


def test_a_fluid_is_zero_outside_clamped_inside_and_blended_in_time():
    rng = np.random.default_rng(3)
    u0 = rng.normal(size=(5, 3, 4, 3))
    u1 = rng.normal(size=(5, 3, 4, 3))
    axes = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    vol = FluidVolume([u0, u1], cell=0.5, corner=(1.0, 2.0, 3.0), axes=axes, step=0.1).as_dict()
    corner = np.array([1.0, 2.0, 3.0])

    def world(g):  # grid coordinates (in cells) to a world point
        return corner + axes @ (0.5 * np.asarray(g, dtype=np.float64))

    # A cell center reads the cell, turned to world axes.
    U, _, _ = fluid_at(vol, [world((2.5, 1.5, 0.5))], 0.0)
    assert np.allclose(U[0], axes @ u0[2, 1, 0])
    # Inside the outer half cell, the outer cell (clamped).
    U, _, _ = fluid_at(vol, [world((0.1, 0.2, 3.9))], 0.0)
    assert np.allclose(U[0], axes @ u0[0, 0, 3])
    # Past a face, nothing.
    U, _, _ = fluid_at(vol, [world((5.01, 1.0, 1.0)), world((1.0, -0.01, 1.0))], 0.0)
    assert np.all(U == 0.0)
    # Halfway between two centers, their mean.
    U, _, _ = fluid_at(vol, [world((2.0, 1.5, 0.5))], 0.0)
    assert np.allclose(U[0], axes @ (0.5 * (u0[1, 1, 0] + u0[2, 1, 0])))
    # A quarter of the way to frame 1, the frames blended; past it, frame 1.
    U, _, _ = fluid_at(vol, [world((2.5, 1.5, 0.5))], 0.025)
    assert np.allclose(U[0], axes @ (0.75 * u0[2, 1, 0] + 0.25 * u1[2, 1, 0]))
    U, _, _ = fluid_at(vol, [world((2.5, 1.5, 0.5))], 7.0)
    assert np.allclose(U[0], axes @ u1[2, 1, 0])


def test_fluid_bricks_keep_only_occupied_bricks_and_read_back_exactly():
    u = np.zeros((20, 9, 8, 3))
    u[3, 4, 5] = (1.0, 2.0, 3.0)
    rho = np.zeros((20, 9, 8))
    rho[17, 8, 0] = 0.5
    table, bricks = fluid_bricks(u, rho)
    assert table.shape == (3 * 2 * 1,)
    assert len(bricks) == 2 and np.sum(table != FLUID_EMPTY) == 2
    vol = FluidVolume([u], cell=1.0, density=[rho]).as_dict()
    U, R, _ = fluid_at(vol, [[3.5, 4.5, 5.5], [17.5, 8.5, 0.5], [10.5, 1.5, 1.5]], 0.0)
    assert np.allclose(U, [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(R, [0.0, 0.5, 0.0])


def test_fluid_settings_are_refused_where_they_do_not_belong():
    vol = FluidVolume([np.zeros((2, 2, 2, 3))], cell=1.0)
    with pytest.raises(ValueError, match="needs fluid="):
        make_field("fluid")
    with pytest.raises(ValueError, match="takes flow=, not inflow="):
        make_field("fluid", fluid=vol, inflow=1.0)
    with pytest.raises(ValueError, match="takes no noise"):
        make_field("fluid", fluid=vol, noise=0.5)
    with pytest.raises(ValueError, match="only a fluid field takes flow="):
        make_field("force", flow=1.0)
    with pytest.raises(ValueError, match="only a fluid field takes fluid="):
        make_field("force", fluid=vol)
    with pytest.raises(ValueError, match="not finite at cell"):
        FluidVolume([np.full((2, 2, 2, 3), np.nan)], cell=1.0)
    with pytest.raises(ValueError, match="need step="):
        FluidVolume([np.zeros((2, 2, 2, 3))] * 2, cell=1.0)


def test_a_fluid_field_writes_the_volume_the_solver_reads(tmp_path):
    u = np.zeros((9, 2, 2, 3))
    u[8, 1, 1] = (1.0, 0.0, 0.0)
    vol = FluidVolume([u, 2 * u], cell=0.25, corner=(0.5, 0.0, 0.0), step=0.04)
    ff = ForceField().add("fluid", fluid=vol, strength=3.0, flow=1.0, use_density=False)
    fields, scripts, mask = resolve_targets(ff, {}, 4)
    write_session(str(tmp_path), fields, scripts, mask=mask)
    d = tmp_path / "bin" / "force_field"
    m = tomllib.loads((d / "force_field.toml").read_text())
    fl = m["field"][0]["fluid"]
    assert fl["resolution"] == [[9, 2, 2], [9, 2, 2]] and len(fl["tables"]) == 2
    table = np.fromfile(d / fl["tables"][1], dtype="<u4")
    bricks = np.fromfile(d / fl["bricks"][1], dtype="<f4").reshape(-1, 512, 4)
    assert table.tolist() == [FLUID_EMPTY, 0] and bricks[0, 0 + 8 * (1 + 8 * 1), 0] == 2.0
    # Per frame the corner, the axes as columns and the cell along each axis.
    transform = np.fromfile(d / fl["transform"], dtype="<f8").reshape(2, 15)
    assert np.allclose(transform[0], [0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.25, 0.25, 0.25])
    # Flow rides in the track's inflow slot, as Blender stores it.
    assert np.allclose(track_array(fields[0][0])[:, 26], 1.0)


def _chunks(data: bytes, size: int) -> list:
    return [data[i:i + size] for i in range(0, len(data), size)]


def test_the_decoder_turns_the_add_ons_fluid_bricks_into_a_volume():
    # The add-on sends a Fluid Flow's volume as per-frame brick bytes
    # (`blender_addon/core/force_field.py` encode), each whole or as a list of
    # chunks; the decoder must hand them to the public `FluidVolume.from_bricks`,
    # and the field must read what the frames held.
    from frontend import App
    from frontend._decoder_ import ParamDecoder

    u0 = np.zeros((9, 4, 4, 3))
    u0[8, 2, 1] = (1.0, -2.0, 0.5)
    frames = [fluid_bricks(u0), fluid_bricks(2.0 * u0)]
    axes = np.eye(3)
    payload = {
        "type": "fluid", "strength": 3.0, "flow": 1.0, "use_density": False,
        "groups": None, "axes": axes.tolist(),
        "fluid": {
            "resolution": [9, 4, 4],
            # Frame 0 whole, frame 1 chunked, as a large frame travels.
            "tables": [frames[0][0].astype("<u4").tobytes(),
                       _chunks(frames[1][0].astype("<u4").tobytes(), 4)],
            "bricks": [frames[0][1].astype("<f4").tobytes(),
                       _chunks(frames[1][1].astype("<f4").tobytes(), 1000)],
            "corner": [[0.0, 0.0, 0.0]] * 2, "axes": [axes.tolist()] * 2,
            "cell": [[0.5, 0.5, 0.5]] * 2, "start": 0.0, "step": 0.1,
        },
    }
    decoder = ParamDecoder()
    decoder._data = {"force_field": {"fields": [payload], "scripts": []}}
    scene = App.create("decoder_fluid").scene.create()
    decoder.apply_force_field(scene)
    (field,) = scene.force_field.fields
    center = [[0.5 * 8.5, 0.5 * 2.5, 0.5 * 1.5]]
    U, _, presence = fluid_at(field["fluid"], center, 0.05)
    assert np.allclose(U[0], 1.5 * np.array([1.0, -2.0, 0.5])) and presence[0] == 1.0
    a, _ = evaluate(field, center, velocities=[[0.0, 0.0, 0.0]], fluid_time=0.1)
    assert np.allclose(a[0], 3.0 * 2.0 * np.array([1.0, -2.0, 0.5]))


def test_fluid_frames_may_differ_in_size_and_place():
    # A VDB sequence's active box changes from frame to frame, so each frame
    # carries its own resolution and corner; the same world point reads the
    # same cell from both.
    u0 = np.zeros((3, 2, 2, 3))
    u0[2, 1, 1] = (1.0, 0.0, 0.0)
    u1 = np.zeros((12, 9, 2, 3))
    u1[3, 1, 1] = (3.0, 0.0, 0.0)
    vol = FluidVolume([u0, u1], cell=0.5, corner=[(1.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
                      step=1.0).as_dict()
    point = [[2.25, 0.75, 0.75]]
    assert np.allclose(fluid_at(vol, point, 0.0)[0][0], (1.0, 0.0, 0.0))
    assert np.allclose(fluid_at(vol, point, 1.0)[0][0], (3.0, 0.0, 0.0))
    assert np.allclose(fluid_at(vol, point, 0.5)[0][0], (2.0, 0.0, 0.0))


# --- fluid validation, empty frames, per-axis cells and memory ---------------

import pickle  # noqa: E402
import shutil  # noqa: E402
import tracemalloc  # noqa: E402

from frontend._force_field_ import hold_fluid_frames  # noqa: E402


def test_use_density_needs_a_volume_built_with_density():
    # Without density the channel is zero, so use_density would weigh the
    # push by zero everywhere: refused rather than run as a field that does
    # nothing.
    u = np.ones((2, 2, 2, 3))
    bare = FluidVolume([u], cell=0.5)
    assert not bare.has_density
    with pytest.raises(ValueError, match="built without density="):
        ForceField().add("fluid", fluid=bare, strength=10.0, flow=1.0, use_density=True)
    dense = FluidVolume([u], cell=0.5, density=[np.ones((2, 2, 2))])
    assert dense.has_density and make_field("fluid", fluid=dense, use_density=True)["use_density"]
    # The decoder's entry is told, never left to guess from the values: a
    # smoke before it emits has a density of zero everywhere.
    table, bricks = fluid_bricks(u)
    with pytest.raises(TypeError, match="has_density"):
        FluidVolume.from_bricks((2, 2, 2), [table], [bricks], cell=0.5)
    told = FluidVolume.from_bricks((2, 2, 2), [table], [bricks], cell=0.5, has_density=False)
    with pytest.raises(ValueError, match="built without density="):
        make_field("fluid", fluid=told, use_density=True)


def test_settings_a_fluid_or_its_kind_would_ignore_are_refused():
    vol = FluidVolume([np.zeros((2, 2, 2, 3))], cell=1.0)
    # Blender applies no Gravitation to a Fluid Flow (measured).
    with pytest.raises(ValueError, match="takes no gravitation"):
        make_field("fluid", fluid=vol, gravitation=True)
    with pytest.raises(ValueError, match="takes no wind_factor"):
        make_field("fluid", fluid=vol, wind_factor=0.5)
    with pytest.raises(ValueError, match="only a fluid field takes fluid= and use_density="):
        make_field("force", use_density=True)
    with pytest.raises(TypeError, match="takes a FluidVolume"):
        make_field("fluid", fluid={})
    with pytest.raises(ValueError, match="only a texture field takes z_up="):
        make_field("force", z_up=True)
    for axes in (2.0 * np.eye(3), np.diag([1.0, 1.0, -1.0])):
        with pytest.raises(ValueError, match="not a rotation"):
            FluidVolume([np.zeros((2, 2, 2, 3))], cell=1.0, axes=axes)


def test_a_noise_scale_the_solver_cannot_read_is_refused():
    # Positive in float64, but zero, subnormal or infinite as the float32
    # the solver reads, which refuses all three at load.
    for bad in (0.0, -1.0, 1e-39, 1e-50, 1e39):
        with pytest.raises(ValueError, match="noise_scale must be positive along every axis"):
            make_field("force", noise_scale=(1.0, bad, 1.0))
    make_field("force", noise_scale=(1e-37, 1.0, 3e38))


def test_a_fluid_value_past_float32_is_refused_by_frame_and_cell():
    # Finite in float64, infinite as the float32 the solver reads.
    u = np.zeros((3, 4, 9, 3))
    u[1, 2, 8, 0] = 1e39
    with pytest.raises(ValueError, match=r"fluid frame 1: the fluid velocity is not finite "
                                         r"at cell \(1, 2, 8\): 1e\+39, past the float32"):
        FluidVolume([np.zeros_like(u), u], cell=0.5, step=0.1)
    rho = np.zeros((3, 4, 9))
    rho[2, 0, 3] = np.nan
    with pytest.raises(ValueError, match=r"fluid frame 0: the fluid density is not finite "
                                         r"at cell \(2, 0, 3\)"):
        FluidVolume([np.zeros((3, 4, 9, 3))], cell=0.5, density=[rho])
    # The add-on's cut, the same function, refuses it the same way.
    with pytest.raises(ValueError, match=r"at cell \(1, 2, 8\)"):
        fluid_bricks(u)
    # A brick that arrives already cut is refused at the cell it covers:
    # brick slot 1 is the second along x, and its cell (1, 1, 0) is (9, 1, 0).
    table, bricks = fluid_bricks(np.ones((10, 2, 2, 3)))
    bricks = bricks.copy()
    bricks[1, 1 + 8 * 1, 2] = np.inf
    with pytest.raises(ValueError, match=r"fluid frame 0: the velocity component 2 at "
                                         r"cell \(9, 1, 0\) is inf"):
        FluidVolume.from_bricks((10, 2, 2), [table], [bricks], cell=0.5, has_density=False)


def test_a_fluid_cell_is_per_axis_and_held_to_what_float32_can_read(tmp_path):
    u = np.zeros((4, 2, 3, 3))
    u[3, 1, 2] = (1.0, 2.0, 3.0)
    vol = FluidVolume([u], cell=(0.5, 0.25, 2.0), corner=(1.0, 0.0, -1.0))
    assert vol.cell.shape == (1, 3)
    rec = vol.as_dict()
    # Cell (3, 1, 2)'s center, each axis at its own cell size.
    center = np.array([1.0 + 3.5 * 0.5, 1.5 * 0.25, -1.0 + 2.5 * 2.0])
    U, _, presence = fluid_at(rec, [center], 0.0)
    assert np.allclose(U[0], (1.0, 2.0, 3.0)) and presence[0] == 1.0
    # The box is n_i cells of cell_i along each axis.
    far = np.array([1.0 + 4 * 0.5, 2 * 0.25, -1.0 + 3 * 2.0])
    probes = [far - 1e-9, far + [1e-6, 0, 0], far + [0, 1e-6, 0], far + [0, 0, 1e-6]]
    assert fluid_at(rec, probes, 0.0)[2].tolist() == [1.0, 0.0, 0.0, 0.0]
    # The session writer gives the solver the three sizes per frame.
    fields, scripts, mask = resolve_targets(ForceField().add("fluid", fluid=vol), {}, 1)
    write_session(str(tmp_path), fields, scripts, mask=mask)
    d = tmp_path / "bin" / "force_field"
    fl = tomllib.loads((d / "force_field.toml").read_text())["field"][0]["fluid"]
    transform = np.fromfile(d / fl["transform"], dtype="<f8").reshape(-1, 15)
    assert np.allclose(transform[0, 12:], (0.5, 0.25, 2.0))
    # Every spelling of cell=, and the one that is ambiguous.
    one = np.zeros((2, 2, 2, 3))
    assert np.allclose(FluidVolume([one], cell=0.5).cell, [[0.5] * 3])
    assert np.allclose(FluidVolume([one, one], cell=[0.5, 0.25], step=1.0).cell,
                       [[0.5] * 3, [0.25] * 3])
    assert np.allclose(FluidVolume([one, one], cell=(0.1, 0.2, 0.3), step=1.0).cell,
                       [[0.1, 0.2, 0.3]] * 2)
    assert np.allclose(FluidVolume([one] * 3, cell=[[0.1, 0.2, 0.3]] * 3, step=1.0).cell,
                       [[0.1, 0.2, 0.3]] * 3)
    with pytest.raises(ValueError, match="ambiguous"):
        FluidVolume([one] * 3, cell=(0.1, 0.2, 0.3), step=1.0)
    # The solver reads the cell in float32 and divides by it: a size whose
    # float32 image, or whose reciprocal, is zero, subnormal or infinite would
    # place nothing, or everything, inside the domain.
    for bad in (0.0, -1.0, 1e39, 1e38, 1e-39, 1e-50, (0.5, 0.5, np.nan)):
        with pytest.raises(ValueError, match="cell size"):
            FluidVolume([one], cell=bad)


def test_an_empty_fluid_frame_has_no_domain(tmp_path):
    u = np.ones((4, 4, 4, 3))
    vol = FluidVolume([u, np.zeros((0, 0, 0, 3)), np.zeros((5, 0, 3, 3)), u],
                      cell=0.5, step=0.1)
    assert vol.resolution == [(4, 4, 4), (0, 0, 0), (0, 0, 0), (4, 4, 4)]
    assert len(vol.tables[1]) == 0 and vol.bricks[1].shape == (0, 512, 4)
    rec = vol.as_dict()
    center = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    # Outside everywhere, the domain's corner included: no fluid, no damping.
    for t in (0.1, 0.2):
        U, rho, presence = fluid_at(rec, center, t)
        assert np.all(U == 0.0) and np.all(rho == 0.0) and np.all(presence == 0.0)
    field = make_field("fluid", fluid=vol, strength=2.0, flow=1.0, axes=np.eye(3))
    a, _ = evaluate(field, center, velocities=[[1.0, 0.0, 0.0]] * 2, fluid_time=0.1)
    assert np.all(a == 0.0)
    # Blended with a full frame, the presence fades in.
    assert np.isclose(fluid_at(rec, center, 0.25)[2][0], 0.5)
    fields, scripts, mask = resolve_targets(ForceField().add("fluid", fluid=vol), {}, 1)
    write_session(str(tmp_path), fields, scripts, mask=mask)
    d = tmp_path / "bin" / "force_field"
    fl = tomllib.loads((d / "force_field.toml").read_text())["field"][0]["fluid"]
    assert fl["resolution"] == [[4, 4, 4], [0, 0, 0], [0, 0, 0], [4, 4, 4]]
    assert (d / fl["tables"][1]).stat().st_size == 0
    assert (d / fl["bricks"][1]).stat().st_size == 0
    # The decoder's entry takes the same frame.
    empty = FluidVolume.from_bricks(
        [[4, 4, 4], [0, 0, 0]], [vol.tables[0], np.zeros(0, np.uint32)],
        [vol.bricks[0], np.zeros(0, np.float32)], cell=0.5, has_density=False, step=0.1)
    assert empty.resolution[1] == (0, 0, 0)


def test_from_bricks_refuses_what_the_solver_would_and_names_the_frame():
    table, bricks = fluid_bricks(np.ones((9, 2, 2, 3)))  # two bricks
    ok = dict(cell=0.5, has_density=False)
    # A float table is not cast into a slot it never named.
    with pytest.raises(TypeError, match="fluid frame 0: a brick table of float64"):
        FluidVolume.from_bricks((9, 2, 2), [np.array([0.7, 1.0])], [bricks], **ok)
    with pytest.raises(ValueError, match="table entry 0 is -1"):
        FluidVolume.from_bricks((9, 2, 2), [np.array([-1, 1])], [bricks], **ok)
    with pytest.raises(ValueError, match="table entry 1 is 2, which names none of the 2"):
        FluidVolume.from_bricks((9, 2, 2), [np.array([0, 2], np.uint32)], [bricks], **ok)
    for resolution in ((9, 2), 9, (9.0, 2.0, 2.0), [(9, 2, 2)] * 2):
        with pytest.raises(ValueError, match="resolution="):
            FluidVolume.from_bricks(resolution, [table], [bricks], **ok)
    with pytest.raises(ValueError, match="3 table entries"):
        FluidVolume.from_bricks((9, 2, 2), [np.array([0, 1, FLUID_EMPTY], np.uint32)],
                                [bricks], **ok)
    with pytest.raises(ValueError, match="bricks of shape"):
        FluidVolume.from_bricks((9, 2, 2), [table], [bricks.reshape(-1, 4)], **ok)
    with pytest.raises(ValueError, match="1 brick tables and 2 brick sets"):
        FluidVolume.from_bricks((9, 2, 2), [table], [bricks, bricks], **ok)
    with pytest.raises(ValueError, match="a domain of"):
        FluidVolume.from_bricks((40000, 1, 1), [np.full(5000, FLUID_EMPTY, np.uint32)],
                                [np.zeros((0, 512, 4), np.float32)], **ok)
    with pytest.raises(ValueError, match="a domain of"):
        FluidVolume([np.zeros((32769, 1, 1, 3))], cell=1.0)
    # An empty frame, any zero extent, holds no brick, as the solver's loader
    # requires; its table is empty, so no entry would have caught the bricks.
    for empty in ((0, 0, 0), (0, 5, 5)):
        with pytest.raises(ValueError, match=rf"fluid frame 1: an empty frame \(a "
                                             rf"\[{empty[0]}, {empty[1]}, {empty[2]}\] "
                                             r"domain\) holds 2 bricks; it holds none"):
            FluidVolume.from_bricks([(9, 2, 2), empty], [table, np.zeros(0, np.uint32)],
                                    [bricks, bricks], step=0.1, **ok)
    # The flat floats a file holds read as the bricks.
    assert FluidVolume.from_bricks((9, 2, 2), [table], [bricks.ravel()], **ok).bricks[0].shape \
        == (2, 512, 4)


def test_a_fluid_volume_reads_its_frames_one_at_a_time():
    # A generator of dense frames streams: one is alive at a time (with one
    # slab of its cut), not the whole sequence before the first is cut.
    n, count = 48, 30
    dense = n ** 3 * 3 * 8

    def frame(k):
        u = np.zeros((n, n, n, 3))
        u[k % n, :4, :4] = 1.0  # a sliver, so few bricks are kept
        return u

    tracemalloc.start()
    try:
        vol = FluidVolume((frame(k) for k in range(count)), cell=0.1, step=0.1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    kept = sum(t.nbytes + b.nbytes for t, b in zip(vol.tables, vol.bricks))
    assert vol.frames == count
    assert peak - kept < 1.5 * dense, (peak, kept, dense)
    rho = np.zeros((n, n, n))
    with pytest.raises(ValueError, match="density frames end at frame 1"):
        FluidVolume((frame(k) for k in range(3)), cell=0.1, step=0.1, density=iter([rho]))
    with pytest.raises(ValueError, match="more density frames"):
        FluidVolume(iter([frame(0)]), cell=0.1, density=iter([rho, rho]))
    with pytest.raises(ValueError, match="2 density frames for 1 velocity frames"):
        FluidVolume([frame(0)], cell=0.1, density=[rho, rho])
    # One frame where a sequence belongs would otherwise be read as nx frames.
    with pytest.raises(ValueError, match=r"velocity= takes a sequence of frames.*\[velocity\]"):
        FluidVolume(frame(0), cell=0.1)
    with pytest.raises(ValueError, match=r"give \[density\]"):
        FluidVolume([frame(0)], cell=0.1, density=rho)


def test_fluid_bricks_holds_a_slab_not_the_whole_box():
    # A sparse frame in a large box: the cut needs one slab and the kept
    # bricks, whatever the box, not several dense copies of it.
    n = 128
    nb = n // 8
    u = np.zeros((n, n, n, 3))
    picks = np.random.default_rng(0).choice(nb ** 3, size=nb ** 3 // 50, replace=False)
    for p in picks:  # 2% of the bricks hold one value each
        u[8 * (p % nb) + 3, 8 * ((p // nb) % nb) + 4, 8 * (p // nb ** 2) + 5] = (1.0, 2.0, 3.0)
    tracemalloc.start()
    try:
        table, bricks = fluid_bricks(u)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(bricks) == len(picks) and np.sum(table != FLUID_EMPTY) == len(picks)
    assert peak < u.nbytes / 8 + 3 * (bricks.nbytes + table.nbytes), (peak, u.nbytes, bricks.nbytes)


def _random_volume(frames=4, n=24, seed=4):
    rng = np.random.default_rng(seed)
    return FluidVolume([rng.normal(size=(n, n, n, 3)) for _ in range(frames)],
                       cell=0.1, step=0.05)


def test_a_written_volume_pickles_as_a_reference_to_the_session_files(tmp_path):
    vol = _random_volume()
    brick_bytes = sum(b.nbytes for b in vol.bricks)
    ff = ForceField().add("fluid", fluid=vol, strength=2.0)
    # Before a session holds the frames, a pickle carries them.
    before = pickle.dumps(ff)
    assert len(before) > brick_bytes
    fields, scripts, mask = resolve_targets(ff, {}, 1)
    write_session(str(tmp_path), fields, scripts, mask=mask)
    after = pickle.dumps(ff)
    assert len(after) < 0.1 * brick_bytes
    # Read back, the frames come from the session's own files, exactly.
    back = pickle.loads(after).fields[0]["fluid"]
    for k in range(vol.frames):
        assert np.array_equal(back["bricks"][k], vol.bricks[k])
        assert np.array_equal(back["tables"][k], vol.tables[k])
    point = [[1.2, 1.2, 1.2]]
    assert np.array_equal(fluid_at(back, point, 0.07)[0], fluid_at(vol.as_dict(), point, 0.07)[0])
    # A changed file, and a removed one, fail by name.
    d = tmp_path / "bin" / "force_field" / "field-0-fluid"
    data = np.fromfile(d / "bricks-00002.bin", dtype="<f4")
    data[7] += 1.0
    data.tofile(d / "bricks-00002.bin")
    with pytest.raises(ValueError, match="fluid frame 2 was written to 1 place, and none "
                                         "holds it any more: .* hold other values"):
        pickle.loads(after).fields[0]["fluid"]["bricks"][2]
    shutil.rmtree(tmp_path / "bin")
    with pytest.raises(FileNotFoundError, match="fluid frame 0 was written to 1 place, and "
                                                "none holds it any more: .* is gone"):
        pickle.loads(after).fields[0]["fluid"]["bricks"][0]
    # The one pickled before any write needs no file, and the live volume
    # kept its frames.
    assert np.array_equal(pickle.loads(before).fields[0]["fluid"]["bricks"][3], vol.bricks[3])
    assert len(pickle.dumps(ff)) > brick_bytes


def test_a_volume_read_back_from_a_session_can_be_written_over_it(tmp_path):
    vol = _random_volume(frames=2, n=12)
    brick_bytes = sum(b.nbytes for b in vol.bricks)
    fields, scripts, mask = resolve_targets(ForceField().add("fluid", fluid=vol), {}, 1)
    write_session(str(tmp_path), fields, scripts, mask=mask)
    directory = tmp_path / "bin" / "force_field"
    back = pickle.loads(pickle.dumps(fields))
    # Held before its directory is cleared, it is written anew there, and
    # then refers to the new files again.
    hold_fluid_frames(back, str(directory))
    shutil.rmtree(directory)
    write_session(str(tmp_path), back, scripts, mask=mask)
    assert len(pickle.dumps(back)) < 0.1 * brick_bytes
    assert np.array_equal(back[0][0]["fluid"]["bricks"][1], vol.bricks[1])
    # Without the hold, the cleared file is named.
    again = pickle.loads(pickle.dumps(back))
    shutil.rmtree(directory)
    with pytest.raises(FileNotFoundError, match="fluid frame 0 was written to 1 place"):
        write_session(str(tmp_path), again, scripts, mask=mask)


def test_the_app_state_and_session_pickles_hold_no_second_copy_of_the_bricks(tmp_path):
    # The add-on path: the decoder takes the volume out of its payload, the
    # scene builds and exports a session, and the state pickled afterwards
    # (what app_state.pickle and fixed_session.pickle hold) refers to the
    # session's brick files rather than carrying them again.
    from frontend._asset_ import AssetManager
    from frontend._decoder_ import ParamDecoder
    from frontend._mesh_ import MeshManager
    from frontend._plot_ import PlotManager
    from frontend._scene_ import Scene

    asset = AssetManager()
    V, F = MeshManager(str(tmp_path / "cache")).square(res=3)
    asset.add.tri("sheet", V, F)
    scene = Scene("fluid_pickles", PlotManager(), asset)
    scene.add("sheet")
    rng = np.random.default_rng(8)
    frames = [fluid_bricks(rng.normal(size=(16, 16, 16, 3))) for _ in range(6)]
    brick_bytes = sum(b.nbytes for _, b in frames)
    payload = {
        "type": "fluid", "strength": 2.0, "flow": 1.0, "use_density": False,
        "groups": None, "axes": np.eye(3).tolist(),
        "fluid": {
            "resolution": [[16, 16, 16]] * 6,
            "tables": [_chunks(t.astype("<u4").tobytes(), 1 << 12) for t, _ in frames],
            "bricks": [_chunks(b.astype("<f4").tobytes(), 1 << 16) for _, b in frames],
            "corner": [[0.0, 0.0, 0.0]] * 6, "axes": [np.eye(3).tolist()] * 6,
            "cell": [[0.1, 0.1, 0.1]] * 6, "start": 0.0, "step": 0.04,
        },
    }
    decoder = ParamDecoder()
    decoder._data = {"force_field": {"fields": [payload], "scripts": []}}
    decoder.apply_force_field(scene)
    assert "fluid" not in payload and len(pickle.dumps(decoder)) < 0.1 * brick_bytes
    with pytest.raises(RuntimeError, match="already gave its force field"):
        decoder.apply_force_field(Scene("again", PlotManager(), asset))
    fixed = scene.build(quiet=True)
    session = tmp_path / "session"
    fixed.export_fixed(str(session), delete_exist=True)
    state = pickle.dumps((decoder, scene, fixed))
    assert len(state) < 0.1 * brick_bytes, (len(state), brick_bytes)
    # Read back, the scene exports again over its own session directory.
    recovered = pickle.loads(state)[2]
    recovered.export_fixed(str(session), delete_exist=True)
    volume = recovered._force_field["fields"][0][0]["fluid"]
    for k, (t, b) in enumerate(frames):
        assert np.array_equal(volume["tables"][k], t)
        assert np.array_equal(volume["bricks"][k], b)
    assert len(pickle.dumps(recovered)) < 0.1 * brick_bytes


def test_the_decoder_refuses_a_malformed_byte_payload_by_name():
    from frontend import App
    from frontend._decoder_ import ParamDecoder

    table, bricks = fluid_bricks(np.ones((2, 2, 2, 3)))

    def decode(**fluid):
        volume = {"resolution": [2, 2, 2], "tables": [table.tobytes()],
                  "bricks": [bricks.tobytes()], "corner": [[0.0] * 3],
                  "axes": [np.eye(3).tolist()], "cell": [[0.5] * 3], "start": 0.0,
                  "step": 1.0}
        volume.update(fluid)
        decoder = ParamDecoder()
        decoder._data = {"force_field": {"fields": [
            {"type": "fluid", "groups": None, "fluid": volume}], "scripts": []}}
        decoder.apply_force_field(App.create("decoder_fluid_bytes").scene.create())

    with pytest.raises(ValueError, match=r"chunk 1 is str, not bytes"):
        decode(bricks=[[bricks.tobytes(), "x"]])
    with pytest.raises(ValueError, match="table is str; expected bytes or a list of bytes"):
        decode(tables=[table.tobytes().hex()])
    with pytest.raises(ValueError, match="table: chunk 0 is int, not bytes"):
        decode(tables=[table.tolist()])
    with pytest.raises(ValueError, match="not a whole number of 4-byte values"):
        decode(bricks=[bricks.tobytes()[:-1]])
    with pytest.raises(ValueError, match=r"one \(cx, cy, cz\) per frame"):
        decode(cell=[0.5])


def test_a_z_up_texture_reads_and_pushes_in_blender_axes(tmp_path):
    # A Blender point (bx, by, bz) is the solver point (bx, bz, -by), and a
    # Blender vector likewise. With z_up the field reads its image at the
    # point's Blender coordinates and returns Blender's push in solver axes.
    def to_solver(v):
        v = np.asarray(v, dtype=float)
        return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)

    # Measured: a red texel pushes (-0.5, 0.5, 0.5) in Blender's axes, read on
    # Blender's horizontal plane, whatever the height.
    red_top = np.zeros((4, 4, 3))
    red_top[2:, :, 0] = 1.0
    f = make_field("texture", image=red_top, interpolation=False, axes=IDENTITY, z_up=True)
    blender = [(0.1, 0.5, 0.0), (0.1, 0.5, 0.7), (0.1, -0.5, 0.0)]
    a = evaluate(f, to_solver(blender))[0]
    assert np.allclose(a, to_solver([(-0.5, 0.5, 0.5)] * 2 + [(0.5, 0.5, 0.5)]))
    # Every mode and extension: the z_up field at a solver point is the
    # solver-axes image of the plain field at that point's Blender coordinates
    # (world), or of the plain field at the same point (object coordinates).
    rng = np.random.default_rng(6)
    image = rng.random((5, 7, 3))
    pb = rng.uniform(-1.2, 1.2, (20, 3))
    ps = to_solver(pb)
    for mode in ("rgb", "gradient", "curl"):
        for ext in ("extend", "clip_cube"):
            kw = dict(image=image, texture_mode=mode, extension=ext, axes=IDENTITY, nabla=0.05)
            got = evaluate(make_field("texture", z_up=True, **kw), ps)[0]
            assert np.allclose(got, to_solver(evaluate(make_field("texture", **kw), pb)[0]))
            got = evaluate(make_field("texture", z_up=True, object_coordinates=True, **kw), ps)[0]
            want = evaluate(make_field("texture", object_coordinates=True, **kw), ps)[0]
            assert np.allclose(got, to_solver(want))
    # The session writer tells the solver.
    ff = ForceField().add("texture", image=image, z_up=True).add("texture", image=image)
    fields, scripts, mask = resolve_targets(ff, {}, 1)
    write_session(str(tmp_path), fields, scripts, mask=mask)
    manifest = tomllib.loads((tmp_path / "bin" / "force_field" / "force_field.toml").read_text())
    assert [f["texture"]["z_up"] for f in manifest["field"]] == [True, False]


def test_the_compile_worker_runs_as_a_package_module():
    # The server answers Compile and Check with `python -m
    # frontend._force_field_ check`; the module is a package module only.
    import json
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    request = json.dumps({"source": "def eval(x, y, z, t):\n    return (x, 0.0, 0.0)\n",
                          "z_up": True})
    out = subprocess.run([sys.executable, "-m", "frontend._force_field_", "check"],
                         input=request, capture_output=True, text=True, cwd=root,
                         timeout=300)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["ok"]
    with open(os.path.join(root, "frontend", "_force_field_.py"), encoding="utf-8") as f:
        assert "except ImportError" not in f.read()


def _documented_args(fn_or_class):
    """The names an ``Args:`` section documents, each entry's names on one
    line as napoleon needs them; a wrapped name list is refused."""
    import inspect
    import re

    doc = inspect.cleandoc(fn_or_class.__doc__)
    block = doc.split("Args:\n", 1)[1]
    names = []
    for line in block.splitlines():
        if not line.strip():
            break
        if line.startswith("        "):
            continue  # a description's continuation
        entry = re.match(r"^    ([a-z_][a-z_0-9]*(?:, [a-z_][a-z_0-9]*)*): \S", line)
        assert entry, f"an Args entry napoleon cannot read: {line!r}"
        names += entry.group(1).split(", ")
    return names


def test_the_public_docstrings_document_every_argument_once():
    import inspect

    for fn, doc in ((ForceField.add, ForceField.add), (FluidVolume.__init__, FluidVolume),
                    (FluidVolume.from_bricks, FluidVolume.from_bricks)):
        params = [p for p in inspect.signature(fn).parameters if p not in ("self", "cls")]
        documented = _documented_args(doc)
        assert sorted(documented) == sorted(params), (fn.__qualname__, documented, params)
    # noise_scale is the field object's scale, and its own entry says the two
    # things it scales: Noise Amount's gusts and a Use Coordinates lookup.
    import re

    entry = re.search(r"\n\s*noise_scale:(.*?)(?=\n\s*\w[\w, ]*:\s)", ForceField.add.__doc__, re.S)
    assert entry is not None, "ForceField.add documents noise_scale in an entry of its own"
    text = " ".join(entry.group(1).split())
    assert "object_coordinates" in text and "Noise Amount" in text, text
    # Public text names no plan or notes document.
    for obj in (ForceField, ForceField.add, FluidVolume, FluidVolume.from_bricks):
        assert not re.search(r"\w+\.md\b|plan section", obj.__doc__), obj


def test_density_and_presence_blend_between_frames_of_different_extent():
    # Frame 0 covers x in [0, 2), frame 1 x in [1, 5), each with its own
    # density; between them both quantities blend, and a point only frame 0
    # covers fades out.
    u0, rho0 = np.ones((4, 2, 2, 3)), np.full((4, 2, 2), 0.25)
    u1, rho1 = 3.0 * np.ones((8, 2, 2, 3)), np.full((8, 2, 2), 0.75)
    vol = FluidVolume([u0, u1], cell=0.5, corner=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                      density=[rho0, rho1], step=1.0).as_dict()
    both, only0 = [1.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    U, rho, presence = fluid_at(vol, [both, only0], 0.25)
    assert np.allclose(rho, [0.75 * 0.25 + 0.25 * 0.75, 0.75 * 0.25])
    assert np.allclose(presence, [1.0, 0.75])
    assert np.allclose(U[:, 0], [0.75 * 1.0 + 0.25 * 3.0, 0.75 * 1.0])


def test_the_decoder_takes_per_frame_resolutions_transforms_and_density():
    # What the add-on sends for a VDB sequence whose active box changes: one
    # resolution, corner and cell per frame, the table length changing too.
    from frontend import App
    from frontend._decoder_ import ParamDecoder

    u0, rho0 = np.zeros((9, 4, 4, 3)), np.zeros((9, 4, 4))
    u0[8, 2, 1], rho0[8, 2, 1] = (1.0, 0.0, 0.0), 0.5
    u1, rho1 = np.zeros((20, 9, 8, 3)), np.zeros((20, 9, 8))
    u1[17, 8, 0], rho1[17, 8, 0] = (0.0, 2.0, 0.0), 0.25
    frames = [fluid_bricks(u0, rho0), fluid_bricks(u1, rho1)]
    assert [len(t) for t, _ in frames] == [2, 6]
    corners = [[0.0, 0.0, 0.0], [-1.0, 0.5, 0.0]]
    cells = [[0.5, 0.25, 0.5], [0.25, 0.25, 0.125]]
    payload = {
        "type": "fluid", "strength": 1.0, "flow": 0.0, "use_density": True,
        "groups": None, "axes": np.eye(3).tolist(),
        "fluid": {
            "resolution": [[9, 4, 4], [20, 9, 8]],
            "tables": [_chunks(t.astype("<u4").tobytes(), 8) for t, _ in frames],
            "bricks": [_chunks(b.astype("<f4").tobytes(), 4096) for _, b in frames],
            "corner": corners, "axes": [np.eye(3).tolist()] * 2, "cell": cells,
            "start": 0.0, "step": 0.1,
        },
    }
    decoder = ParamDecoder()
    decoder._data = {"force_field": {"fields": [payload], "scripts": []}}
    scene = App.create("decoder_fluid_frames").scene.create()
    decoder.apply_force_field(scene)
    (field,) = scene.force_field.fields
    volume = field["fluid"]
    assert volume["resolution"] == [(9, 4, 4), (20, 9, 8)] and volume["has_density"]
    p0 = np.array(corners[0]) + (np.array([8, 2, 1]) + 0.5) * cells[0]
    p1 = np.array(corners[1]) + (np.array([17, 8, 0]) + 0.5) * cells[1]
    U, rho, _ = fluid_at(volume, [p0, p1], 0.0)
    assert np.allclose(U[0], (1.0, 0.0, 0.0)) and np.isclose(rho[0], 0.5)
    U, rho, _ = fluid_at(volume, [p0, p1], 0.1)
    assert np.allclose(U[1], (0.0, 2.0, 0.0)) and np.isclose(rho[1], 0.25)
    # Use Smoke Density weighs the push by that density.
    a, _ = evaluate(field, [p1], velocities=[[0.0, 0.0, 0.0]], fluid_time=0.1)
    assert np.allclose(a[0], 0.25 * np.array([0.0, 2.0, 0.0]))


# --- the field object's scale, playback, and fluid frames across saves -------


def test_an_object_coordinates_texture_is_read_under_the_field_objects_scale():
    # Blender reads an Object Coordinates image under the field object's whole
    # inverse world matrix, its scale included (measured), so the lookup point
    # is S^-1 R^T (p - c), and the differences nabla apart are taken there.
    # The vector stays a world-axes one.
    rng = np.random.default_rng(12)
    image = rng.random((6, 5, 3))
    c, s = math.cos(0.7), math.sin(0.7)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    origin = np.array([0.3, -0.2, 0.5])
    scale = np.array([1.3, 0.8, 1.1])
    points = rng.uniform(-1.5, 1.5, (25, 3))
    lookup = ((points - origin) @ rotation) / scale
    for mode in ("rgb", "gradient", "curl"):
        kw = {"image": image, "texture_mode": mode, "nabla": 0.05,
              "object_coordinates": True}
        got = evaluate(make_field("texture", position=origin, axes=rotation,
                                  noise_scale=scale, **kw), points)[0]
        want = evaluate(make_field("texture", axes=IDENTITY, **kw), lookup)[0]
        assert np.allclose(got, want, atol=1e-12), mode
        # In world coordinates the object's scale reaches nothing.
        world = dict(kw, object_coordinates=False)
        assert np.array_equal(
            evaluate(make_field("texture", noise_scale=scale, **world), points)[0],
            evaluate(make_field("texture", **world), points)[0])
    # Stripes black, white, black, white across x in [-1, 1], read nearest: a
    # point 0.8 along local X reads white unscaled and, at scale 2, the black
    # stripe at 0.4.
    stripes = np.zeros((1, 4, 3))
    stripes[:, 1::2] = 1.0
    at = [[0.8, 0.0, 0.0]]
    kw = {"image": stripes, "interpolation": False, "object_coordinates": True,
          "axes": IDENTITY}
    assert np.allclose(evaluate(make_field("texture", **kw), at)[0], -0.5)
    assert np.allclose(evaluate(make_field("texture", noise_scale=(2.0, 1.0, 1.0), **kw),
                                at)[0], 0.5)


def test_a_fluids_velocity_is_divided_by_the_playback_speed():
    # U is per second of the scene clock and a step moves the vertices by
    # dt * playback seconds of motion, so the solver compares U / playback
    # with the vertex's velocity; the Flow damping is per second of motion.
    center = (-1.0 + 0.125 * 5.5, -1.0 + 0.125 * 8.5, -1.0 + 0.125 * 8.5)
    v = [[0.5, 0.2, 0.0]]
    f = _fluid_field(_measured_volume(), strength=2.0, flow=1.0)
    U = fluid_at(f["fluid"], [center], 0.0)[0][0]
    assert np.allclose(evaluate(f, [center], velocities=v)[0][0], 2.0 * (U - v[0]))
    assert np.allclose(evaluate(f, [center], velocities=v, playback=2.0)[0][0],
                       2.0 * (U / 2.0 - v[0]))
    # Only a fluid reads it.
    drag = make_field("drag", axes=IDENTITY)
    assert np.array_equal(evaluate(drag, [center], velocities=v, playback=3.0)[0],
                          evaluate(drag, [center], velocities=v)[0])
    for bad in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="not a positive speed"):
            evaluate(f, [center], velocities=v, playback=bad)


def test_a_volumes_frames_cannot_change_once_it_holds_them(tmp_path):
    # A pickle may name the files written for a frame in place of its values,
    # so an edit of a frame would be lost by the next save; it is refused.
    vol = _random_volume(frames=2, n=12)
    with pytest.raises(ValueError, match="read-only"):
        vol.bricks[0][0, 0, 0] = 123.0
    with pytest.raises(ValueError, match="read-only"):
        vol.tables[1][0] = 0
    fields, scripts, mask = resolve_targets(ForceField().add("fluid", fluid=vol), {}, 1)
    write_session(str(tmp_path), fields, scripts, mask=mask)
    with pytest.raises(ValueError, match="read-only"):
        vol.bricks[1][0, 0, 0] = 123.0
    # A frame read back from its files, too.
    back = pickle.loads(pickle.dumps(vol))
    with pytest.raises(ValueError, match="read-only"):
        back.bricks[0][0, 0, 0] = 123.0
    # from_bricks keeps its own copy of an array the caller can still write...
    table, bricks = fluid_bricks(np.ones((9, 2, 2, 3)))
    ok = {"cell": 0.5, "has_density": False}
    taken = FluidVolume.from_bricks((9, 2, 2), [table], [bricks], **ok)
    bricks[0, 0, 0] = 123.0
    table[0] = FLUID_EMPTY
    assert taken.bricks[0][0, 0, 0] == 1.0 and taken.tables[0][0] == 0
    # ...and keeps the decoder's read-only views of bytes as they are.
    data = fluid_bricks(np.ones((9, 2, 2, 3)))[1].tobytes()
    view = np.frombuffer(data, dtype="<f4")
    kept = FluidVolume.from_bricks((9, 2, 2), [np.array([0, 1], np.uint32)], [view], **ok)
    assert np.shares_memory(kept.bricks[0], view)
    # A buffer that is not an ndarray but can still be written (an
    # array.array, a bytearray) is copied too, whatever type it arrives as.
    import array as pyarray

    raw = bytearray(fluid_bricks(np.ones((9, 2, 2, 3)))[1].tobytes())
    held = pyarray.array("I", [0, 1])
    via = FluidVolume.from_bricks((9, 2, 2), [held], [np.frombuffer(raw, dtype="<f4")], **ok)
    raw[0:4] = np.float32(123.0).tobytes()
    held[0] = FLUID_EMPTY
    assert via.bricks[0][0, 0, 0] == 1.0 and via.tables[0][0] == 0
    # A frame unpickled before any session write is read-only as well.
    fresh = pickle.loads(pickle.dumps(_random_volume(frames=1, n=10)))
    with pytest.raises(ValueError, match="read-only"):
        fresh.bricks[0][0, 0, 0] = 123.0


def test_a_resident_frame_refers_only_to_files_that_hold_it_byte_for_byte(tmp_path):
    # Two volumes of the same brick layout written to the same paths in turn:
    # the files hold the second, at the first's sizes, so the first must carry
    # its bricks rather than name them.
    u = np.zeros((16, 16, 16, 3))
    u[..., 0] = 2.0
    first, second = FluidVolume([u], cell=0.1), FluidVolume([2.0 * u], cell=0.1)
    for vol in (first, second):
        fields, scripts, mask = resolve_targets(ForceField().add("fluid", fluid=vol), {}, 1)
        if (tmp_path / "bin").exists():
            shutil.rmtree(tmp_path / "bin")
        write_session(str(tmp_path), fields, scripts, mask=mask)
        if vol is first:
            # Its own files: a reference.
            assert len(pickle.dumps(first)) < 0.1 * first.bricks[0].nbytes
    back = pickle.loads(pickle.dumps(first))
    assert np.array_equal(back.bricks[0], first.bricks[0])
    assert len(pickle.dumps(first)) > first.bricks[0].nbytes
    assert len(pickle.dumps(second)) < 0.1 * second.bricks[0].nbytes


def _fluid_app(tmp_path, monkeypatch, name, vol):
    """An App in ``tmp_path`` whose scene holds a sheet and a fluid field."""
    from frontend import App

    data = str(tmp_path / "data")
    monkeypatch.setattr(App, "get_data_dirpath", staticmethod(lambda: data))
    app = App.create(name, cache_dir=str(tmp_path / "cache"))
    V, F = app.mesh.square(res=3)
    app.asset.add.tri("sheet", V, F)
    scene = app.scene.create()
    scene.add("sheet").at(0, 0.5, 0)
    scene.force_field.add("fluid", fluid=vol, flow=1.0)
    return app, scene


def _session_fluid(fixed_session, index=0):
    return fixed_session.fixed_scene._force_field["fields"][index][0]["fluid"]


def _listing(path):
    return sorted(os.path.relpath(os.path.join(root, f), path)
                  for root, _, files in os.walk(path) for f in files)


def test_a_rebuild_reads_the_fluid_from_any_session_that_still_holds_it(tmp_path, monkeypatch):
    # Two sessions built from one scene each hold a copy of the fluid, and a
    # saved app names both; deleting the one written last leaves the other to
    # read, and the rebuild must not clear the session that holds it first.
    from frontend import App

    vol = _random_volume(frames=3, n=16)
    app, scene = _fluid_app(tmp_path, monkeypatch, "fluid_two_sessions", vol)
    fixed = scene.build()
    a = app.session.create(fixed, "a").build()
    app.session.create(fixed, "b").build()
    app.save()
    loaded = App.load("fluid_two_sessions")
    loaded.session.delete("b")
    rebuilt = loaded.session.select("a").build()
    for k in range(vol.frames):
        assert np.array_equal(_session_fluid(rebuilt)["bricks"][k], vol.bricks[k])
    assert np.array_equal(_session_fluid(App.recover("a"))["bricks"][2], vol.bricks[2])
    # A frame that no copy holds any more fails before the session is
    # cleared, rather than after it.
    again = App.load("fluid_two_sessions")
    fluid = os.path.join(a.info.path, "bin", "force_field", "field-0-fluid")
    data = np.fromfile(os.path.join(fluid, "bricks-00001.bin"), dtype="<f4")
    (data + 1.0).tofile(os.path.join(fluid, "bricks-00001.bin"))
    before = _listing(a.info.path)
    with pytest.raises(ValueError, match=r"fluid frame 1 was written to 2 places, and none "
                                         r"holds it any more: .*/b/.* is gone; .*/a/.* hold "
                                         r"other values"):
        again.session.select("a").build()
    assert _listing(a.info.path) == before and "fixed_session.pickle" in before


@pytest.mark.parametrize("update", ["moved", "dropped", "replaced", "kept"])
def test_an_update_of_a_held_run_leaves_its_saved_session_rebuildable(
        tmp_path, monkeypatch, update):
    # The saved session (fixed_session.pickle) keeps the scene as built; an
    # update rewrites the files the built fluid was read from, so it must save
    # that state again, or a recovered session can no longer be rebuilt.
    from frontend import App
    from frontend._session_ import FixedSession

    vol = _random_volume(frames=3, n=16)
    app, scene = _fluid_app(tmp_path, monkeypatch, f"fluid_update_{update}", vol)
    session = app.session.create(scene.build(), "held").build()
    # A held run, as far as the frontend's half of an update needs one.
    monkeypatch.setattr(FixedSession, "_require_held", lambda self, what: 0)
    os.makedirs(session.output.path, exist_ok=True)
    field = scene.force_field.clear()
    if update == "moved":
        field.add("wind", strength=1.0).add("fluid", fluid=vol, flow=1.0)
    elif update == "dropped":
        field.add("wind", strength=1.0)
    elif update == "replaced":  # the same brick layout, other values
        field.add("fluid", fluid=_random_volume(frames=3, n=16, seed=5), flow=1.0)
    else:
        field.add("fluid", fluid=vol, flow=2.0)
    session.update_force_field(field)
    recovered = App.recover("held")
    # A recovered held run can be steered, and rebuilt, alike.
    recovered.update_force_field(ForceField().add("wind", strength=2.0))
    rebuilt = App.recover("held").session.build()
    for k in range(vol.frames):
        assert np.array_equal(_session_fluid(rebuilt)["bricks"][k], vol.bricks[k])
        assert np.array_equal(_session_fluid(rebuilt)["tables"][k], vol.tables[k])


def test_a_fixed_session_pickles_without_its_solver_process(tmp_path, monkeypatch):
    # The process handle holds locks that do not pickle; a running session
    # is saved without it, and one read back finds the solver otherwise.
    import subprocess
    import sys

    vol = _random_volume(frames=1, n=8)
    app, scene = _fluid_app(tmp_path, monkeypatch, "fluid_process", vol)
    session = app.session.create(scene.build(), "proc").build()
    session._process = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        back = pickle.loads(pickle.dumps(session))
    finally:
        session._process.wait()
    assert back._process is None and session._process is not None
