#!/usr/bin/env python3
# File: build-win-native/scripts/source-wheel.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Builds one frontend package's wheel from its upstream source, at the commit
# scripts\downloads.txt pins, for the interpreter running this script, and checks
# what it built. warmup.bat runs it on ARM64 for each package PyPI publishes no
# win_arm64 wheel of, with the portable MSVC environment loaded:
#
#     python scripts\source-wheel.py PACKAGE OUT_DIR
#
# It is the Windows counterpart of build-linux-native/scripts/source-wheel.sh and
# makes the same promises. It is Python rather than a .bat because it unpacks a
# wheel, parses a disassembly and compares records, none of which cmd.exe does
# reliably.
#
# It is safe to run twice: a wheel already in OUT_DIR for this interpreter whose
# source record matches the pins is checked again rather than rebuilt.
#
# THE SOURCE IS FETCHED, NEVER CARRIED. This repository holds no third-party
# code. The package is fetched at its pinned commit into downloads\src\<package>,
# where a host with no route to the repository can be given a copy, and is built
# from there unmodified. Anything a build needs to differ is a compiler setting
# or an environment variable below, never an edit to the source.
#
# FLOATING-POINT CONTRACTION IS OFF, AND CHECKED ON THE RESULT. These packages
# carry exact geometric predicates, which are exact only when every product is
# rounded on its own; a fused multiply-add breaks the error-free transformation
# they rest on. MSVC does not contract under its default /fp:precise unless
# /fp:contract is also given, measured on x64, and this script passes
# /fp:precise explicitly through the CL variable so the setting is stated rather
# than inherited. For ARM64, where FMA is in the base instruction
# set, MSVC's default has not been measured anywhere in this project, so the built
# extension's instructions are counted and a single fused one fails the build.
#
# THE BUILD TOOLS ARE PINNED BY HASH. scripts\wheel-build-requirements.txt is
# installed with --require-hashes into a throwaway environment and the build runs
# with --no-build-isolation, so no build requirement is resolved at build time.
# PPF_WIN_WHEELHOUSE replaces the index for that install, for a host with no route
# to PyPI.
#
# WHAT IT LEAVES: OUT_DIR\<wheel>.whl, and OUT_DIR\<wheel>.sources.txt naming the
# repositories, commits and settings it was built from, which bundle.bat copies
# into the distribution's licenses\.
import argparse
import hashlib
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_WIN = os.path.dirname(HERE)
SOURCES_ROOT = os.path.join(BUILD_WIN, "downloads", "src")
WORK_ROOT = os.path.join(BUILD_WIN, ".wheel-build")
REQUIREMENTS = os.path.join(HERE, "wheel-build-requirements.txt")
CL_SETTING = "/fp:precise"

# EACH PACKAGE IS DESCRIBED BY WHAT DIFFERS BETWEEN THEM AND NOTHING ELSE: its
# repository and commit keys in downloads.txt, its submodules as (path, url key,
# commit key), and the extension module inside the wheel whose instructions are
# counted.
PACKAGES = {
    "triangle": {
        "repo": ("URL_TRIANGLE_GIT", "TRIANGLE_COMMIT"),
        "submodules": [("c", "URL_TRIANGLE_C_GIT", "TRIANGLE_C_COMMIT")],
        # The interpreter tag in the name is optional: a stable-ABI build names
        # its module without one.
        "extension": re.compile(r"^triangle/core(?:\.[^/]*)?\.pyd$"),
    },
    "tetgen": {
        "repo": ("URL_TETGEN_GIT", "TETGEN_COMMIT"),
        "submodules": [],
        # tetgen builds with nanobind, which targets the stable ABI from CPython
        # 3.12. On the windows-11-arm runner the wheel came out cp312-abi3 and
        # a pattern requiring an interpreter tag matched 0 files in it, as a
        # stable-ABI module name carries none. On cp311 the name is tagged.
        "extension": re.compile(r"^tetgen/_tetgen(?:\.[^/]*)?\.pyd$"),
    },
}

# The PE Machine field and the fused multiply-add mnemonics `dumpbin /disasm`
# prints for each architecture: vfmadd231sd and its family on x64, fmadd, fmsub,
# fnmadd and fnmsub on ARM64, where the vector forms are fmla and fmls.
MACHINES = {
    "AMD64": (0x8664, re.compile(r"\bvfn?m(?:add|sub)\w*\b", re.IGNORECASE)),
    "ARM64": (0xAA64, re.compile(r"\b(?:fn?m(?:add|sub)|fml[as])\b", re.IGNORECASE)),
}


def die(message, *lines):
    print(f"ERROR: {message}", file=sys.stderr)
    for line in lines:
        print(f"       {line}", file=sys.stderr)
    sys.exit(1)


def manifest():
    """KEY=VALUE pairs of scripts\\downloads.txt, read as load-downloads.bat reads them."""
    values = {}
    with open(os.path.join(HERE, "downloads.txt"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value
    return values


def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, **kwargs)


def git(*args, cwd=None, check=True):
    # core.filemode=false because NTFS carries no executable bit. A checkout made
    # on Windows records that itself, but one relayed from a Linux host keeps
    # `filemode = true` in its .git/config, and every executable upstream file
    # then reads as a local change: measured on a relayed tetgen checkout, whose
    # tools/audit_wheel.sh showed `mode change 100755 => 100644` and nothing else.
    result = run(["git", "-c", "core.filemode=false", *args], cwd=cwd, capture_output=True)
    if check and result.returncode:
        die(f"git {' '.join(args)} failed in {cwd or os.getcwd()}",
            *(result.stderr or result.stdout).strip().splitlines()[-10:])
    return result.stdout.strip()


def pe_machine(path):
    """The Machine field of a PE image, read off its own header."""
    with open(path, "rb") as handle:
        data = handle.read(4096)
    if data[:2] != b"MZ" or len(data) < 0x40:
        die(f"{path} is not a PE image")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[offset:offset + 4] != b"PE\0\0":
        die(f"{path} has an MZ header and no PE signature")
    return struct.unpack_from("<H", data, offset + 4)[0]


def main():
    ap = argparse.ArgumentParser(description="Build a frontend wheel from pinned upstream source.")
    ap.add_argument("package", choices=sorted(PACKAGES))
    ap.add_argument("out_dir")
    args = ap.parse_args()

    spec = PACKAGES[args.package]
    pins = manifest()
    for key in (spec["repo"], *[(u, c) for _p, u, c in spec["submodules"]]):
        for name in key:
            if not pins.get(name):
                die(f"scripts\\downloads.txt defines no {name}")
    repo_url, repo_commit = (pins[k] for k in spec["repo"])
    submodules = [(path, pins[u], pins[c]) for path, u, c in spec["submodules"]]

    machine = platform.machine().upper()
    if machine not in MACHINES:
        die(f"this builds on x64 and ARM64 Windows, and the interpreter reports {machine}")
    want_machine, fused_pattern = MACHINES[machine]
    py_tag = "cp%d%d" % sys.version_info[:2]
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    if not platform_tag.startswith("win"):
        die(f"this builds Windows wheels, and the interpreter reports platform {platform_tag}")

    for tool in ("git", "cl", "link", "dumpbin"):
        if shutil.which(tool) is None:
            die(f"required tool not found on PATH: {tool}",
                "Run this with MinGit on PATH and the portable MSVC environment loaded,",
                "as warmup.bat does.")

    with open(REQUIREMENTS, "rb") as handle:
        requirements_digest = hashlib.sha256(handle.read()).hexdigest()
    record = "\n".join(
        [f"{args.package}, built from upstream source for {py_tag} {platform_tag}",
         f"  {repo_url} @ {repo_commit}"]
        + [f"  submodule {p}: {u} @ {c}" for p, u, c in submodules]
        + [f"  compiler: MSVC with CL={CL_SETTING}",
           f"  build tools: scripts/wheel-build-requirements.txt, sha256 {requirements_digest}"]
    ) + "\n"

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def check_wheel(wheel):
        """The one extension module inside is built for this host and fuses nothing."""
        with zipfile.ZipFile(wheel) as archive:
            names = [n for n in archive.namelist() if spec["extension"].match(n)]
            if len(names) != 1:
                print(f"{wheel} holds {len(names)} files matching {spec['extension'].pattern}, "
                      f"and exactly one extension module is expected", file=sys.stderr)
                return False
            with tempfile.TemporaryDirectory() as scratch:
                extension = archive.extract(names[0], scratch)
                found = pe_machine(extension)
                if found != want_machine:
                    print(f"{names[0]} is PE machine 0x{found:04X}, and this host is "
                          f"0x{want_machine:04X}", file=sys.stderr)
                    return False
                disasm = run(["dumpbin", "/nologo", "/disasm", extension], capture_output=True)
                if disasm.returncode:
                    print(f"dumpbin /disasm failed on {names[0]}:\n{disasm.stdout[-2000:]}",
                          file=sys.stderr)
                    return False
                # THE LINE COUNT IS ASSERTED BEFORE THE VERDICT. An empty
                # disassembly would report zero fused instructions over nothing.
                lines = disasm.stdout.count("\n")
                fused = len(fused_pattern.findall(disasm.stdout))
        print(f"  {os.path.basename(wheel)}: machine 0x{found:04X}, {lines} disassembly lines, "
              f"{fused} fused multiply-add instructions")
        if lines < 1000:
            print(f"{names[0]} disassembled to {lines} lines, which is not a disassembly of an "
                  f"extension module; no statement about contraction can be made", file=sys.stderr)
            return False
        if fused:
            print(f"{names[0]} carries {fused} fused multiply-add instructions, and its "
                  f"predicates are exact only without them", file=sys.stderr)
            return False
        return True

    suffix = f"-{py_tag}-"
    existing = [f for f in os.listdir(out_dir)
                if f.startswith(args.package + "-") and f.endswith(f"-{platform_tag}.whl")
                and (suffix in f or "-abi3-" in f)]
    if len(existing) > 1:
        die(f"{out_dir} holds {len(existing)} {args.package} wheels for {platform_tag}",
            "Remove them and re-run.")
    if existing:
        wheel = os.path.join(out_dir, existing[0])
        record_path = wheel[:-4] + ".sources.txt"
        if os.path.exists(record_path) and open(record_path, encoding="utf-8").read() == record:
            print(f"Found {existing[0]}, built from the pinned sources. Checking rather than rebuilding.")
            if not check_wheel(wheel):
                die(f"the {args.package} wheel in {out_dir} does not verify (see above)",
                    f"Remove it and re-run: del {wheel[:-4]}.*")
            return

    # -----------------------------------------------------------------------
    # The source, at its pins
    # -----------------------------------------------------------------------
    src = os.path.join(SOURCES_ROOT, args.package)
    if not os.path.isdir(os.path.join(src, ".git")):
        # Fetched into a .part directory first, so an interrupted fetch is never
        # taken for a complete one.
        part = src + ".part"
        print(f"Fetching {repo_url} at {repo_commit}")
        shutil.rmtree(part, ignore_errors=True)
        os.makedirs(SOURCES_ROOT, exist_ok=True)
        git("init", "-q", part)
        fetch = run(["git", "-C", part, "fetch", "-q", "--depth", "1", repo_url, repo_commit],
                    capture_output=True)
        if fetch.returncode:
            die(f"could not fetch {repo_commit} from {repo_url}",
                "On a host that cannot reach it, place a clone at that commit, with its",
                f"submodules, in {src} and re-run.")
        git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD", cwd=part)
        for path, url, commit in submodules:
            print(f"Fetching submodule {path} from {url} at {commit}")
            target = os.path.join(part, path)
            shutil.rmtree(target, ignore_errors=True)
            git("init", "-q", target)
            fetch = run(["git", "-C", target, "fetch", "-q", "--depth", "1", url, commit],
                        capture_output=True)
            if fetch.returncode:
                die(f"could not fetch {commit} from {url}")
            git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD", cwd=target)
        os.replace(part, src)

    repair = f"rmdir /s /q {src} and re-run"
    if git("rev-parse", "HEAD", cwd=src) != repo_commit:
        die(f"{src} is not at the pinned commit {repo_commit}", f"Repair with: {repair}")
    if git("status", "--porcelain", "--ignore-submodules=all", cwd=src):
        die(f"{src} carries local changes",
            f"The build uses upstream's source unmodified. Repair with: {repair}")
    for path, url, commit in submodules:
        recorded = ""
        for line in git("ls-tree", "HEAD", path, cwd=src).splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[1] == "commit":
                recorded = fields[2]
        if recorded != commit:
            die(f"{repo_url} at {repo_commit} records {path} at {recorded or 'nothing'}, "
                f"and downloads.txt pins {commit}",
                "A submodule pin that disagrees with its parent tree builds a combination",
                "upstream never had. Move the two pins together.")
        sub = os.path.join(src, path)
        if git("rev-parse", "HEAD", cwd=sub, check=False) != commit:
            die(f"{sub} is not at the pinned commit {commit}", f"Repair with: {repair}")
        if git("status", "--porcelain", cwd=sub):
            die(f"{sub} carries local changes", f"Repair with: {repair}")
    print(f"Source: {src} at {repo_commit}")

    include = sysconfig.get_paths()["include"]
    if not os.path.exists(os.path.join(include, "Python.h")):
        die(f"no Python.h for {sys.executable} (looked in {include})",
            "An extension module is compiled against its interpreter's headers, which the",
            "embeddable python.org distribution does not carry.")

    # -----------------------------------------------------------------------
    # The build
    # -----------------------------------------------------------------------
    work = os.path.join(WORK_ROOT, f"{args.package}-{py_tag}-{platform_tag}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "dist"))
    # A copy without the repositories' metadata, so the build's own output lands
    # in the work directory and downloads\src stays exactly what was fetched.
    shutil.copytree(src, os.path.join(work, "src"), ignore=shutil.ignore_patterns(".git"))

    # Every interpreter this script starts sees only its own installation.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONNOUSERSITE"] = "1"
    venv = os.path.join(work, "venv")
    if run([sys.executable, "-m", "venv", venv], env=env).returncode:
        die(f"python -m venv failed for {sys.executable}")
    venv_python = os.path.join(venv, "Scripts", "python.exe")
    pip_source = []
    if os.environ.get("PPF_WIN_WHEELHOUSE"):
        pip_source = ["--no-index", "--find-links", os.environ["PPF_WIN_WHEELHOUSE"]]
    if run([venv_python, "-m", "pip", "install", "-q", *pip_source, "--require-hashes",
            "--only-binary=:all:", "-r", REQUIREMENTS], env=env).returncode:
        die("installing the pinned build tools failed",
            "Every tool and hash is in scripts\\wheel-build-requirements.txt.")

    build_env = dict(env)
    build_env["CL"] = CL_SETTING
    # THE COMPILER IS THE ONE ON PATH, AND setuptools HAS TO BE TOLD SO. Without
    # DISTUTILS_USE_SDK setuptools ignores the loaded environment and searches the
    # registry and vswhere for a Visual Studio installation, finds none of the
    # portable one, and fails with "Microsoft Visual C++ 14.0 or greater is
    # required" beside a working cl.exe. Measured with the portable MSVC
    # environment loaded. CMake, which builds tetgen, reads PATH on its own.
    build_env["DISTUTILS_USE_SDK"] = "1"
    build_env["MSSdk"] = "1"
    # THE CMAKE GENERATOR AND BUILD TYPE ARE STATED, NOT DISCOVERED. Left to
    # itself, scikit-build-core configured tetgen with "NMake Makefiles" and the
    # Debug flags `/Od /RTC1`, and tetgen's own `/O2` then failed the compile
    # with D8016 ('/RTC1' and '/O2' command-line options are incompatible).
    # Measured on Windows. Ninja is pinned in the build tools above, and
    # Release is what a shipped extension is built as.
    build_env["CMAKE_GENERATOR"] = "Ninja"
    build_env["SKBUILD_CMAKE_BUILD_TYPE"] = "Release"
    # The build tools are the venv's, and CMake and Ninja are console scripts in
    # its Scripts directory, which scikit-build-core finds on PATH.
    build_env["PATH"] = os.path.join(venv, "Scripts") + os.pathsep + env.get("PATH", "")
    log_path = os.path.join(work, "build.log")
    print(f"Building {args.package} for {py_tag} {platform_tag} with CL={CL_SETTING}")
    with open(log_path, "w", encoding="utf-8") as log:
        result = run([venv_python, "-m", "pip", "wheel", "-v", "--no-build-isolation",
                      "--no-deps", "-w", os.path.join(work, "dist"), os.path.join(work, "src")],
                     env=build_env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = open(log_path, encoding="utf-8", errors="replace").read().splitlines()[-40:]
        print("\n".join(tail), file=sys.stderr)
        die(f"pip wheel failed; the last lines of {log_path} are above")

    built = [f for f in os.listdir(os.path.join(work, "dist"))
             if f.startswith(args.package + "-") and f.endswith(".whl")]
    if len(built) != 1:
        die(f"pip wheel left {len(built)} {args.package} wheels in {work}\\dist, and one is expected")
    if not built[0].endswith(f"-{platform_tag}.whl"):
        die(f"the built wheel {built[0]} is not tagged {platform_tag}")
    built_path = os.path.join(work, "dist", built[0])
    if not check_wheel(built_path):
        die(f"the {args.package} wheel just built does not verify (see above)")

    for stale in os.listdir(out_dir):
        if stale.startswith(args.package + "-") and (
                stale.endswith(f"-{platform_tag}.whl") or stale.endswith(".sources.txt")):
            os.remove(os.path.join(out_dir, stale))
    shutil.move(built_path, os.path.join(out_dir, built[0]))
    with open(os.path.join(out_dir, built[0][:-4] + ".sources.txt"), "w", encoding="utf-8") as handle:
        handle.write(record)
    shutil.rmtree(work, ignore_errors=True)
    print(f"  [OK] {built[0]} in {out_dir}")


main()
