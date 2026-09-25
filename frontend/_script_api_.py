# File: _script_api_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""What a force field script may call: the ONE list every surface shows.

THIS FILE EXISTS TWICE, BYTE FOR BYTE: as ``frontend/_script_api_.py`` and as
``blender_addon/core/script_api.py``, because the add-on ships without the
frontend and both show the list (the compiler in its errors and in
``scene.force_field.builtins()``, the add-on in its Built-in Functions popup,
its new-script template, its drawing of the script, MCP and the Python API).
``addon_host_tests/_force_field_noise_copies_.py`` fails when the two differ,
and ``frontend/tests/_force_field_compiler_.py`` fails when the compiler
admits a call this list does not name, or refuses one it does.

Plain data, no imports, so either side can load it anywhere.
"""

# (name, signature, what it gives), per section, in the order shown.
NOISE = (
    ("noise", "noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0)",
     "Smooth random value in about -1 to 1"),
    ("curl_noise", "curl_noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0)",
     "Smooth random swirl with no sources or sinks, three values: "
     "return it, or unpack it as cx, cy, cz = curl_noise(...)"),
)

NOISE_ARGUMENTS = (
    ("octaves", "Layers of finer detail, a whole number 1 to 8 written in the script"),
    ("seed", "Picks another random pattern"),
    ("time", "Pass t to let the pattern change over time; without it the pattern stays still"),
    ("frequency", "How many times per second the pattern changes, in place"),
    ("decay", "How fast it fades: multiplied by exp(-decay * time)"),
)

MATH = (
    ("sqrt", "math.sqrt(x)", "Square root"),
    ("sin", "math.sin(x)", "Sine, x in radians"),
    ("cos", "math.cos(x)", "Cosine, x in radians"),
    ("tan", "math.tan(x)", "Tangent, x in radians"),
    ("asin", "math.asin(x)", "Arc sine"),
    ("acos", "math.acos(x)", "Arc cosine"),
    ("atan", "math.atan(x)", "Arc tangent"),
    ("atan2", "math.atan2(y, x)", "Angle of the point (x, y)"),
    ("sinh", "math.sinh(x)", "Hyperbolic sine"),
    ("cosh", "math.cosh(x)", "Hyperbolic cosine"),
    ("tanh", "math.tanh(x)", "Hyperbolic tangent"),
    ("exp", "math.exp(x)", "e to the power x"),
    ("log", "math.log(x)", "Natural logarithm"),
    ("pow", "math.pow(x, y)", "x to the power y"),
    ("hypot", "math.hypot(x, y)", "Length of the vector (x, y)"),
    ("fabs", "math.fabs(x)", "Absolute value"),
    ("floor", "math.floor(x)", "Largest whole number not above x"),
    ("ceil", "math.ceil(x)", "Smallest whole number not below x"),
)

MATH_CONSTANTS = (
    ("pi", "math.pi", "3.14159..."),
    ("e", "math.e", "2.71828..."),
    ("tau", "math.tau", "2 pi"),
)

PYTHON = (
    ("abs", "abs(x)", "Absolute value"),
    ("min", "min(a, b, ...)", "Smallest of two or more values"),
    ("max", "max(a, b, ...)", "Largest of two or more values"),
    ("float", "float(x)", "x as a number"),
    ("range", "for i in range(n):", "A loop run a fixed number of times, n written in the script"),
)

LANGUAGE = (
    "import math, then def eval(x, y, z, t): returning (ax, ay, az) on every path",
    "Arithmetic (+ - * / % **), comparisons, and, or, not",
    "if / elif / else, and x if c else y",
    "Local variables, and for loops over range(n)",
)

SECTIONS = (
    ("Noise", NOISE),
    ("Math", MATH),
    ("Math constants", MATH_CONSTANTS),
    ("Python", PYTHON),
)


def names() -> set:
    """Every callable name a script may use, ``math.`` ones without the
    prefix (``range`` included, though it is only a loop's)."""
    return {name for _, entries in SECTIONS if entries is not MATH_CONSTANTS
            for name, _, _ in entries}


def reference_text() -> str:
    """The whole list as plain text, one entry per line."""
    lines = ["A force field script may use:"]
    lines += [f"  {line}" for line in LANGUAGE]
    for title, entries in SECTIONS:
        lines.append("")
        lines.append(f"{title}:")
        width = max(len(sig) for _, sig, _ in entries)
        for _, sig, what in entries:
            lines.append(f"  {sig.ljust(width)}  {what}")
        if entries is NOISE:
            lines.append("  Arguments:")
            for arg, what in NOISE_ARGUMENTS:
                lines.append(f"    {arg}: {what}")
    return "\n".join(lines)
