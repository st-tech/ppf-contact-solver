#!/usr/bin/env python3
"""Acceptance gate: the ROCm build never selects a fast-math device library.

THE HAZARD IS NOT SPEED, IT IS NON-PENETRATION. On ROCm the floating-point
policy is chosen by LINKING a different device-library variant, and the variants
are real files: `oclc_correctly_rounded_sqrt_{on,off}.bc`,
`oclc_unsafe_math_{on,off}.bc`, `oclc_finite_only_{on,off}.bc` and
`oclc_daz_opt_{on,off}.bc` all ship in `rocm-device-libs`. `-ffast-math` selects
the `off` variant of correctly rounded sqrt, and this tree MANDATES a correctly
rounded square root at the ACCD sites, so that variant is a question about
whether the solver can still guarantee non-penetration rather than a question
about throughput. The rule is therefore absolute: no ROCm compile selects fast
math.

A library built with the wrong variant LOADS, DISPATCHES AND RETURNS PLAUSIBLE
WRONG NUMBERS, which is why this is a build-time gate: nothing downstream would
report it.

WHY IT RESOLVES THE FLAGS THROUGH THE RECIPE RATHER THAN GREPPING THE FILE.
`check-metal-math-mode.py` learned this the expensive way: the same spelling
sitting in an unused variable satisfies a grep and reaches no compile. So this
asks `make -n` what the compiler would actually be invoked with, on BOTH
platform branches, and reads the answer.
WHY IT IS TRUSTED: it was FAULT-INJECTED before being relied on. With
`-ffast-math` added to HIPCC_FLAGS it names the flag, prints the offending
command line and exits 1 on both platform branches; with the injection removed
it passes. A gate that has never failed is a gate nobody has checked.
"""
import os, re, subprocess, sys

# scripts/ -> workflows/ -> .github/ -> repo root: four levels, not three.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
RECIPE = os.path.join(ROOT, "crates", "ppf-cts-compute", "rocm")

# Any of these on a compile line means a fast-math device library.
FORBIDDEN = [
    "-ffast-math",
    "-funsafe-math-optimizations",
    "-ffinite-math-only",
    "-fno-honor-infinities",
    "-fno-honor-nans",
    "-munsafe-fp-atomics",
    # The control globals, if a recipe ever linked one directly.
    "oclc_correctly_rounded_sqrt_off",
    "oclc_unsafe_math_on",
    "oclc_finite_only_on",
]


def recipe_lines(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    # ROCM_PATH only has to be non-empty for `make -n`: nothing is executed.
    env.setdefault("ROCM_PATH", "/nonexistent-rocm")
    env.setdefault("HIPCC", "hipcc-not-run")
    # `-B` (always-make) is REQUIRED, not a convenience. Without it `make -n`
    # emits nothing once the tree is built, because every target is up to date,
    # and this gate would then read ZERO compiler invocations. Its own emptiness
    # check turns that into a loud failure rather than a false pass, which is
    # how the flaw was found, but a gate whose verdict depends on whether
    # someone has built recently is not a gate. `-B` makes the recipes print
    # regardless of timestamps, and `-n` still executes nothing.
    out = subprocess.run(["make", "-n", "-B", "abi"], cwd=RECIPE, env=env,
                         capture_output=True, text=True)
    if not out.stdout.strip():
        sys.exit("check-rocm-fp-flags: `make -n abi` produced no command lines "
                 f"in {RECIPE}; either the recipe moved or this gate cannot "
                 "see what the compiler is invoked with, and either way it "
                 "refuses to report a pass it did not earn.\n" + out.stderr[:600])
    return [l for l in out.stdout.splitlines() if "hipcc" in l or "HIPCC" in l]


def main():
    problems = []
    checked = 0
    for label, extra in (("AMD", {}), ("NVIDIA", {"HIP_PLATFORM": "nvidia"})):
        lines = recipe_lines(extra)
        if not lines:
            sys.exit(f"check-rocm-fp-flags: the {label} branch produced no "
                     "compiler invocation, so this gate read nothing.")
        checked += len(lines)
        for line in lines:
            for flag in FORBIDDEN:
                if flag in line:
                    problems.append((label, flag, line.strip()[:110]))
        print(f"  {label:<7} {len(lines)} compiler invocations read")

    print(f"\nchecked {checked} invocations against "
          f"{len(FORBIDDEN)} forbidden spellings")
    if problems:
        print("\nFAILED: a fast-math spelling reaches a compile")
        for label, flag, line in problems:
            print(f"  {label}: {flag}\n    {line}")
        sys.exit(1)
    print("\nOK: no ROCm compile selects a fast-math device library")


if __name__ == "__main__":
    main()
