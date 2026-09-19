#!/usr/bin/env python3
"""Every Metal compile that SHIPS states `-fmetal-math-mode=safe`.

SAFE IS A CORRECTNESS SETTING AND NOT A PERFORMANCE KNOB. Fast math DELETES
Kahan compensation outright, zero threads keep it, and it damages the
eigenvalue-floored inverse on the large majority of threads, the worst
returning a value nine orders out. `MTLMathModeRelaxed` is not a middle
ground and `options:nil` is a correctness bug.

WHY A GATE AND NOT A CODE REVIEW. There are THREE Metal compile paths and they
are held together by nothing but this file:

  * the run-time compile from SOURCE, which sets `mathMode` Safe on the options
    object and errors if it does not read back as Safe
    (`metal/metal_context.mm`),
  * the macOS bundle's offline compile, which REFUSES to build an artifact
    whose flags lack the exact spelling (`build-mac-native/build.sh`),
  * and the entries compile, `xcrun metal $(METAL_ENTRY_FLAGS)`, whose `.air`
    objects `xcrun metallib` links into `ppf_entries.metallib`, the library the
    backend loads by path.

The third one passed NO math flag, and the comment above it said the compile
"never ships" while four lines in the same file trace its output into the
shipped library. Nothing noticed, because a library compiled under fast math
loads, dispatches and returns plausible numbers. Two of the three paths policed
themselves and the third did not.

WHAT THIS CANNOT DO, stated so nobody reads more into a pass than is there: it
cannot tell you what `xcrun metal` does by DEFAULT, which is a property of the
toolchain on the machine. That is the point. A default is not a guarantee, and
this gate exists so the answer stops mattering.
"""
import pathlib, re, sys

ROOT = pathlib.Path(".")
MAKEFILE = ROOT / "crates/ppf-cts-compute/metal/Makefile"
BUNDLE = ROOT / "build-mac-native/build.sh"
SAFE = "-fmetal-math-mode=safe"

bad = []
checked = 0

if not MAKEFILE.is_file():
    sys.exit(f"check-metal-math-mode: no such file: {MAKEFILE}. This gate would "
             f"check nothing and report a pass, which is not a pass.")
make_raw = MAKEFILE.read_text()
# A RECIPE IS ONE LOGICAL LINE, and these span several: the `xcrun metal`
# invocation puts its `-o` on a continuation. Matching the physical first line
# sees the compiler and not its output, which is how the first draft of this
# gate found no compile at all and said so.
make = re.sub(r"\\\n\s*", " ", make_raw)

# THE VARIABLE THE SHIPPING RECIPE ACTUALLY USES, resolved from the recipe
# rather than assumed. A `-fmetal-math-mode=safe` sitting in some other variable
# in the same file would satisfy a naive grep and reach no compile.
#
# A COMPILE, NOT A PROBE. `xcrun metal --version` and `xcrun metallib --version`
# are availability checks that emit nothing, and holding them to a math mode
# would be a finding over nothing. A compile is the invocation that names an
# output, so the `-o` is what separates the two.
recipes = [r for r in re.findall(r"^\t.*xcrun\s+metal\s+(.*)$", make, re.M)
           if "-o " in r and "--version" not in r]
if not recipes:
    sys.exit("check-metal-math-mode: found no `xcrun metal` recipe producing an "
             f"output in {MAKEFILE}. The compile this gate exists to check has "
             f"moved or been renamed; find it rather than letting this report "
             f"clean.")

for recipe in recipes:
    checked += 1
    used = re.findall(r"\$\((\w+)\)", recipe)
    # Expand one level of make variable, which is all these recipes use.
    text = recipe
    for name in used:
        m = re.search(rf"^{name}\s*[:?]?=\s*(.*)$", make_raw, re.M)
        if m:
            text += " " + m.group(1)
    if SAFE not in text:
        bad.append(f"  {MAKEFILE}: an `xcrun metal` recipe compiles without "
                   f"{SAFE}\n      recipe flags resolved to: {text.strip()[:120]}")

# The bundle's own refusal must stay a refusal. It is the other half of the
# standard, and a gate that let it soften would be checking one path of two.
if not BUNDLE.is_file():
    sys.exit(f"check-metal-math-mode: no such file: {BUNDLE}")
bundle = BUNDLE.read_text()
checked += 1
if SAFE not in bundle or "die" not in bundle:
    bad.append(f"  {BUNDLE}: no longer refuses an artifact whose offline flags "
               f"lack {SAFE}")

print(f"check-metal-math-mode: {checked} shipping Metal compile paths examined")
if bad:
    print("check-metal-math-mode: FAIL")
    print("\n".join(bad))
    print(f"\nEvery Metal compile whose output ships states {SAFE}. It is a\n"
          "correctness setting: fast math deletes the Kahan compensation this\n"
          "solver depends on, and a library built without it loads, dispatches\n"
          "and returns plausible wrong numbers.")
    sys.exit(1)
print(f"OK: every shipping Metal compile states {SAFE}")
