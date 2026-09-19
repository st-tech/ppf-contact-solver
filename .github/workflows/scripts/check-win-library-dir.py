#!/usr/bin/env python3
"""Every statement of the Windows backend library directory must agree with
`build-win-native/build.bat`, which is the one that PUTS the DLL there.

WHY THIS EXISTS. The directory moved when the backend moved to
`ppf-cts-compute`, and FIVE consumers kept naming the solver crate's old trees:
`crates/ppf-cts-core/src/datamodel/session/scripts.rs` (which writes
`command.bat`, so it decides where every frontend-launched solve looks) plus
four PowerShell and batch PATHs. They all agreed once and drifted together, and
nothing noticed for one reason: the Windows solver did not LOAD the library at
all. It dispatched into its own host renderings, so no PATH was ever consulted
and any value worked equally well, including a wrong one. The day the solver
actually imported the DLL, every one of those five became an immediate
`STATUS_DLL_NOT_FOUND` (exit 0xC0000135, no output) and every Windows dev tree
failed at solver start.

So the hazard is not that a path is wrong today. It is that a path nothing reads
cannot be wrong until something reads it, and by then it is five paths.

WHAT IS CHECKED. `build.bat` is the source of truth because it is what compiles
and links the DLL into that directory; `LIB_DIR` is resolved through its own
variable chain rather than restated here, so moving the library means editing one
file and this check follows. Every other file that names the directory must name
the same one.

THIS CHECK FAILS LOUDLY WHEN IT CANNOT READ, never quietly. A regex that stops
matching is how a counting gate goes blind and then reports clean over nothing,
so an unresolvable `LIB_DIR`, a file that has lost the directory entirely, or a
consumer list that matches nothing are all errors rather than passes.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUILD_BAT = ROOT / "build-win-native" / "build.bat"

# Files that must agree with build.bat. Each is a place a reader or a process
# looks for the DLL; the comment says which.
CONSUMERS = [
    # Writes command.bat's LIB_PATH_DEV, so it decides where EVERY
    # frontend-launched solve looks. The root one.
    "crates/ppf-cts-core/src/datamodel/session/scripts.rs",
    # PATH for the CI rig and the per-example Windows runs.
    ".github/workflows/scripts/win/run-blender-rig.ps1",
    ".github/workflows/scripts/win/run-example.ps1",
    ".github/workflows/scripts/win/run-iteration.ps1",
    # The bundler reads the DLL from here; the launcher build.bat writes puts it
    # on PATH; fast-check-all runs against it.
    "build-win-native/bundle.bat",
    "build-win-native/fast-check-all.bat",
    "build-win-native/TESTING.md",
]


def fail(message):
    print(f"check-win-library-dir: FAILED  {message}", file=sys.stderr)
    sys.exit(1)


def canonical_lib_dir():
    """Resolve build.bat's LIB_DIR through its own variable chain.

    Derived rather than restated: the point of the check is that one file
    decides, so hardcoding the answer here would make this a second copy of the
    very thing it is meant to pin.
    """
    if not BUILD_BAT.is_file():
        fail(f"{BUILD_BAT} is missing, so the source of truth cannot be read")
    text = BUILD_BAT.read_text(encoding="utf-8", errors="replace")
    variables = {}
    for name, value in re.findall(r"(?mi)^set ([A-Z_]+)=(.+?)\s*$", text):
        variables[name.upper()] = value
    if "LIB_DIR" not in variables:
        fail("build.bat declares no LIB_DIR; this check can no longer find the "
             "directory it is supposed to pin")
    seen = set()
    value = variables["LIB_DIR"]
    while "%" in value:
        match = re.search(r"%([A-Za-z_]+)%", value)
        if match is None:
            break
        key = match.group(1).upper()
        if key in seen:
            fail(f"build.bat's LIB_DIR expands %{key}% in a cycle")
        seen.add(key)
        if key not in variables:
            # SRC_DIR is the checkout root and is computed, not `set` to a
            # literal, so it resolves to nothing and the remainder is the
            # repo-relative directory, which is exactly what the consumers say.
            value = value.replace(match.group(0), "")
            continue
        value = value.replace(match.group(0), variables[key])
    parts = [p for p in re.split(r"[\\/]+", value) if p]
    if not parts:
        fail(f"build.bat's LIB_DIR resolved to nothing (raw: {variables['LIB_DIR']!r})")
    return parts


# How a comment opens, per file type. Checked against CODE rather than against
# the whole file, because a comment naming the right directory would otherwise
# satisfy this while the operative line named a wrong one. Measured: an early
# version of this check passed over a `scripts.rs` whose `join` had been
# repointed, because the explanatory comment above it still spelled the
# directory correctly. A gate that accepts prose as evidence of behavior
# reports on the comment and not on the code.
COMMENT_PREFIXES = {
    ".rs": ("//",),
    ".ps1": ("#",),
    ".bat": ("rem ", "::"),
}


def code_only(path, text):
    """*text* with comment-only content removed, by the file's own syntax.

    Markdown has no code to separate, so it is returned whole: a document is
    prose throughout and is checked for naming the directory at all.
    """
    prefixes = COMMENT_PREFIXES.get(path.suffix.lower())
    if prefixes is None:
        return text
    kept = []
    for line in text.splitlines():
        stripped = line.lstrip()
        lowered = stripped.lower()
        if any(lowered.startswith(prefix) for prefix in prefixes):
            continue
        # A trailing comment on a code line: keep the code before it.
        for prefix in prefixes:
            if prefix in ("rem ", "::"):
                continue
            index = stripped.find(prefix)
            if index > 0:
                line = line[: line.find(prefix)]
                break
        kept.append(line)
    return "\n".join(kept)


def main():
    want = canonical_lib_dir()
    # Matched separator-agnostically: these files spell the same directory with
    # backslashes, forward slashes and doubled backslashes, and the check is
    # about the DIRECTORY rather than the spelling.
    pattern = re.compile(r"[\\/]+".join(re.escape(p) for p in want))
    # Any other `build/lib` under a crate is a stale sibling of this directory.
    stale = re.compile(r"crates[\\/]+[A-Za-z0-9_-]+[\\/]+src[\\/]+[A-Za-z0-9_-]+"
                       r"[\\/]+build[\\/]+lib")
    print("check-win-library-dir: build.bat's LIB_DIR is "
          f"{'/'.join(want)}")
    problems = []
    for rel in CONSUMERS:
        path = ROOT / rel
        if not path.is_file():
            problems.append(f"{rel}: listed as a consumer but the file is missing")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # THE OPERATIVE LINES, not the commentary around them.
        if not pattern.search(code_only(path, text)):
            problems.append(
                f"{rel}: its CODE does not name {'/'.join(want)}, which is "
                f"build.bat's LIB_DIR. A Windows solve launched through it would "
                f"fail with STATUS_DLL_NOT_FOUND. (A comment naming it does not "
                f"count.)")
        for hit in set(stale.findall(text)):
            problems.append(
                f"{rel}: still names {hit}, a stale library directory. The "
                f"backend library lives at {'/'.join(want)}.")
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        fail(f"{len(problems)} disagreement(s) with build.bat")
    print(f"check-win-library-dir: OK  ({len(CONSUMERS)} consumers agree)")


if __name__ == "__main__":
    main()
