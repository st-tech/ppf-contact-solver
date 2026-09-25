# File: _force_field_compiler_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The force-field script compiler and the session writer, without a solver.
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

from frontend._force_field_ import (
    SCRIPT_VERSION,
    SCRIPT_VERSION,
    ForceField,
    ForceFieldScriptError,
    compile_script,
    estimate_line,
    make_grid,
    resolve_targets,
    run_bytecode,
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


def test_a_grid_is_validated_and_estimated(capsys):
    grid = make_grid(np.zeros((3, 4, 5, 6, 3)), (0, 0, 0), (1, 1, 1), times=[0.0, 1.0, 2.0])
    assert grid.dims == (6, 5, 4, 3)
    assert capsys.readouterr().out.strip() == estimate_line(6, 5, 4, 3)
    with pytest.raises(ValueError, match="strictly increasing"):
        make_grid(np.zeros((2, 2, 2, 2, 3)), (0, 0, 0), (1, 1, 1), times=[1.0, 1.0])
    with pytest.raises(ValueError, match="at least 2"):
        make_grid(np.zeros((1, 2, 2, 3)), (0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError, match="not a box"):
        make_grid(np.zeros((2, 2, 2, 3)), (0, 0, 0), (1, 0, 1))
    with pytest.raises(ValueError, match="NaN"):
        make_grid(np.full((2, 2, 2, 3), np.nan), (0, 0, 0), (1, 1, 1))


def test_the_session_writer_lays_out_what_the_solver_reads(tmp_path):
    field = ForceField()
    field.grid(np.ones((2, 2, 3, 3)), (-1, 0, -1), (1, 2, 1), kind="air-velocity",
               groups=["a"])
    field.script("def eval(x, y, z, t):\n    return (0.0, 1.0, 0.0)\n")
    field.script("def eval(x, y, z, t):\n    return (1.0, 0.0, 0.0)\n", groups=["b", "a"])
    weight = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    grids, scripts, mask = resolve_targets(field, {"a": [0], "b": [2]}, 3)
    write_session(str(tmp_path), grids, scripts, weight, mask)
    base = tmp_path / "bin" / "force_field"
    manifest = tomllib.loads((base / "force_field.toml").read_text())
    (grid,) = manifest["grid"]
    assert grid["kind"] == "air-velocity" and grid["dims"] == [3, 2, 2, 1]
    assert os.path.getsize(base / grid["data"]) == 3 * 2 * 2 * 1 * 3 * 4
    assert [sc["targets"] for sc in manifest["script"]] == [32, 1]
    assert grid["targets"] == 0
    assert all(sc["version"] == SCRIPT_VERSION for sc in manifest["script"])
    code = np.fromfile(base / manifest["script"][1]["code"], dtype="<u4")
    assert len(code) == len(field.scripts[1][0].code)
    assert np.array_equal(np.fromfile(base / "weight.bin", dtype="<f4"), weight)
    # Vertex 0 is in "a" (bits 0 and 1), vertex 2 in "b" (bit 1), vertex 1 in none.
    assert list(np.fromfile(base / "targets.bin", dtype="<u4")) == [0b11, 0, 0b10]


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
        ForceField().grid(np.zeros((2, 2, 2, 3)), (0, 0, 0), (1, 1, 1), groups=[])
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
