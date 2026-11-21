# File: build-win-native/fast_check_all.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# What fast-check-all.bat used to do in batch, in Python: run the frontend's
# unit tests, check that every notebook under examples\fail-examples fails, then
# run every example named in examples.txt in fast-check mode (one frame), and
# write fast-check-results.log beside this file. fast-check-all.bat only finds
# the interpreter and starts this script.
#
# WHY NOT BATCH. The folder this runs from may be named with any character a
# Windows file name can hold, such as "ppf-contact-solver (1)", an ampersand, a
# percent sign or an exclamation mark. The batch version needed delayed
# expansion for its loops, and delayed expansion removes an exclamation mark from
# every path it touches; its unquoted assignments split at an ampersand; and
# CALL mangled a percent sign or a caret. Python passes each path to each child
# process as one argument, whatever it holds.
#
# Two layouts, as before: in a source tree this file is in build-win-native\,
# beside its python\ and mingit\, and the project root is its parent; in a
# distribution it is in the root, beside python\, bin\ and examples\.

import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NOPAUSE = any(a.lower() == "/nopause" for a in sys.argv[1:])


def pause():
    if not NOPAUSE:
        print("Press any key to exit...", flush=True)
        try:
            import msvcrt

            msvcrt.getch()
        except ImportError:
            input()


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


parent = os.path.dirname(HERE)
if os.path.exists(os.path.join(parent, "frontend")):
    # Source tree: this script is in build-win-native\.
    tree_root = parent
    build_win = HERE
    python_dir = os.path.join(build_win, "python")
    examples_dir = os.path.join(tree_root, "examples")
    # Where a checkout's backend library sits for either GPU backend, and the
    # ROCm SDK's runtime beside it. A directory that does not exist is skipped
    # by the DLL search, so a CUDA tree is unaffected by the ROCm entries.
    bin_dirs = [
        os.path.join(tree_root, "crates", "ppf-cts-compute", "cuda", "build", "lib"),
        os.path.join(tree_root, "crates", "ppf-cts-compute", "rocm", "build", "lib"),
        os.path.join(build_win, "rocm", "bin"),
    ]
    examples_txt = os.path.join(tree_root, ".github", "workflows", "scripts", "examples.txt")
    if not os.path.exists(examples_txt):
        examples_txt = os.path.join(build_win, "examples.txt")
else:
    # Distribution: this script is in its root.
    tree_root = HERE
    build_win = HERE
    python_dir = os.path.join(tree_root, "python")
    examples_dir = os.path.join(tree_root, "examples")
    bin_dirs = [os.path.join(tree_root, "bin")]
    examples_txt = os.path.join(tree_root, "examples.txt")

fast_check_dir = os.path.join(HERE, "fast_check")
fail_check_dir = os.path.join(HERE, "fail_check")
inject_script = os.path.join(HERE, "inject_fast_check.py")
mingit_dir = os.path.join(build_win, "mingit", "cmd")
python = os.path.join(python_dir, "python.exe")
log_file = os.path.join(HERE, "fast-check-results.log")

env = dict(os.environ)
env["PYTHONPATH"] = tree_root
# NOTHING IS PINNED WHERE A GPU BACKEND IS BUILT. Each backend has its own
# target\<backend>, the frontend searches them, and which one a run uses is
# resolved when it starts (App.get_backend), exactly as the launchers leave it.
# A tree holding only the CPU build has nothing to resolve, and naming it keeps
# the frontend from searching for a GPU build that was never made. KEYED ON THE
# GPU BUILDS THEMSELVES, never on target\release: a distribution carrying several
# backends has no such directory.
has_gpu_build = any(
    os.path.exists(os.path.join(tree_root, "target", b, "release", "ppf-contact-solver.exe"))
    for b in ("cuda", "rocm", "metal")
)
cpu_target = os.path.join(tree_root, "target", "cpu")
if not has_gpu_build and os.path.exists(os.path.join(cpu_target, "release", "ppf-contact-solver.exe")):
    env["CARGO_TARGET_DIR"] = cpu_target
    print(f"This tree holds only the CPU build: CARGO_TARGET_DIR={cpu_target}")
# Use the local CUDA in a source tree (build-win-native\cuda); in a distribution
# the CUDA DLLs are in bin\.
cuda_bin = []
if os.path.exists(os.path.join(build_win, "cuda", "bin", "x64")):
    env["CUDA_PATH"] = os.path.join(build_win, "cuda")
    cuda_bin = [os.path.join(build_win, "cuda", "bin", "x64")]
else:
    env.pop("CUDA_PATH", None)
env["PATH"] = os.pathsep.join(
    [python_dir, os.path.join(python_dir, "Scripts")] + bin_dirs + [mingit_dir] + cuda_bin
    + [env.get("PATH", "")]
)
# Point Python at certifi's CA bundle. urllib.request.urlretrieve (used by some
# examples to fetch mesh assets) doesn't auto-load Windows' system cert store
# under the embedded Python distribution, so without this the clean-environment
# verify run fails with SSLCertVerificationError on any HTTPS asset download.
cacert = os.path.join(python_dir, "Lib", "site-packages", "certifi", "cacert.pem")
env["SSL_CERT_FILE"] = cacert
env["REQUESTS_CA_BUNDLE"] = cacert


def clear_cache():
    # clear-cache.bat sits beside this file in both layouts. It is started by
    # its bare name from inside this folder, so cmd never parses the path.
    subprocess.run(
        [os.environ.get("ComSpec", "cmd.exe"), "/d", "/c", "clear-cache.bat /nopause"],
        cwd=HERE,
        env=env,
    )


def remove(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def to_script(notebook, out_dir):
    """nbconvert NOTEBOOK into OUT_DIR; the script's path, or None."""
    rc = subprocess.run(
        [python, "-m", "nbconvert", "--to", "script", notebook, "--output-dir", out_dir],
        env=env,
    ).returncode
    if rc != 0:
        return None, rc
    script = os.path.join(out_dir, os.path.splitext(os.path.basename(notebook))[0] + ".py")
    if not os.path.exists(script):
        return None, rc
    # Insert App.set_fast_check() after the "app = App" line.
    subprocess.run([python, inject_script, script], env=env)
    return script, 0


def fail():
    remove(fast_check_dir)
    remove(fail_check_dir)
    clear_cache()
    print()
    pause()
    sys.exit(1)


def main():
    clear_cache()
    print()

    if not os.path.exists(examples_txt):
        print(f'ERROR: examples.txt not found at "{examples_txt}"')
        fail()
    if not os.path.exists(python):
        print(f'ERROR: Embedded Python not found at "{python}"')
        print("Please run warmup.bat first to set up the environment.")
        fail()

    # Unit tests (equivalent to warmup.py run_tests).
    print("=== Running Unit Tests ===\n", flush=True)
    rc = subprocess.run(
        [python, "-c",
         "from frontend.tests._runner_ import run_all_tests; import sys; "
         "sys.exit(0 if run_all_tests() else 1)"],
        env=env,
        cwd=tree_root,
    ).returncode
    if rc != 0:
        print("\n=== UNIT TESTS FAILED ===")
        fail()
    print("\n=== Unit Tests Passed ===\n")

    # Fail-examples: every one of these notebooks is designed to fail.
    print("=== Running Fail-Examples Tests ===")
    print("(These notebooks are designed to fail)\n", flush=True)
    fail_examples_dir = os.path.join(examples_dir, "fail-examples")
    if os.path.isdir(fail_examples_dir):
        remove(fail_check_dir)
        os.makedirs(fail_check_dir)
        count = 0
        for name in sorted(os.listdir(fail_examples_dir)):
            if not name.lower().endswith(".ipynb"):
                continue
            stem = os.path.splitext(name)[0]
            print(f"[FAIL-TEST] {stem}", flush=True)
            script, rc = to_script(os.path.join(fail_examples_dir, name), fail_check_dir)
            if script is None:
                print("           [ERROR] nbconvert failed")
                fail()
            rc = subprocess.run(
                [python, script], env=env, cwd=examples_dir,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode
            if rc == 0:
                print("           [ERROR] Expected to fail but passed!")
                fail()
            print("           FAILED (as expected)")
            count += 1
        remove(fail_check_dir)
        print(f"\nAll {count} fail-examples failed as expected.")
    else:
        print(f'fail-examples directory not found at "{fail_examples_dir}"')
    print()

    print("=== Fast Check All Examples ===")
    print(f'Using: "{examples_txt}"\n')
    remove(fast_check_dir)
    os.makedirs(fast_check_dir)

    # examples.txt: one notebook name per line; blank lines and lines starting
    # with # are skipped, as the batch FOR /F with eol=# skipped them.
    with open(examples_txt, encoding="utf-8") as f:
        notebooks = [line.strip() for line in f]
    notebooks = [n for n in notebooks if n and not n.startswith("#")]
    total = len(notebooks)

    log = open(log_file, "w", encoding="utf-8")

    def both(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    log.write("=== Fast Check Results ===\n")
    log.write(f"Started: {stamp()}\n")
    log.write(f"Total tests: {total}\n\n")
    print("Converting notebooks to Python and running tests...")
    print(f"Total tests to run: {total}")
    print(f'Log file: "{log_file}"\n', flush=True)

    passed = []
    for current, notebook in enumerate(notebooks, 1):
        path = os.path.join(examples_dir, notebook + ".ipynb")
        if not os.path.exists(path):
            print(f"[SKIP] {notebook} - notebook not found")
            continue
        both(f"[TEST {current}/{total}] {notebook}")
        script, rc = to_script(path, fast_check_dir)
        if script is None:
            print(f"\n=== FAILED: {notebook} [nbconvert error code: {rc}] ===")
            log.close()
            fail()
        rc = subprocess.run([python, script], env=env, cwd=examples_dir).returncode
        if rc != 0:
            print(f"\n=== FAILED: {notebook} ===")
            log.write("       FAILED\n")
            log.close()
            fail()
        both("       PASSED")
        passed.append(notebook)

    print("\n=== Cleaning up ===", flush=True)
    remove(fast_check_dir)
    clear_cache()

    banner = "============================================"
    summary = f"=== ALL {len(passed)}/{total} TESTS PASSED ==="
    print(f"\n{banner}\n{summary}\n{banner}\n")
    print("Passed tests:")
    log.write(f"\n{banner}\n{summary}\n{banner}\n\nPassed tests:\n")
    for i, name in enumerate(passed, 1):
        both(f"  {i}. {name}")
    print()
    log.write(f"Completed: {stamp()}\n")
    log.close()
    print(f'Log saved to: "{log_file}"\n')
    pause()
    sys.exit(0)


if __name__ == "__main__":
    main()
