#!/usr/bin/env python3
# File: build-win-native/scripts/build_rocm.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The Windows build of the ROCm backend library, the counterpart of what
# crates/ppf-cts-compute/rocm/Makefile does on Linux. It renders every neutral
# *.kernel.cpp for the `hip` target, compiles this directory's mechanism and one
# object per generated entry, and links libppfbe_rocm.dll.
#
# WHY THIS IS PYTHON RATHER THAN A .bat, and why it is one script rather than the
# gen_cuda.py / build.bat split beside it. The kernel id is a table INDEX and the
# table is a SORTED WALK of the neutral tree, so the file order here must be
# byte-identical to the Makefile's `find ... | sort`, which cmd.exe cannot
# produce. gen_cuda.py stops at the rendering because build.bat already owned the
# CUDA compile; nothing on Windows owns the ROCm compile, so this script carries
# it. A rendering also embeds the ABSOLUTE path of the shared header its argument
# record is written against, so the rendering must happen on the machine that
# compiles it and cannot be copied in from Linux.
#
# THE TWO PLATFORMS ARE DIFFERENT BUILDS OF ONE SOURCE, exactly as the Makefile's
# platform branch says. `--platform amd` compiles for the targets in
# rocm_arch.txt and produces a library nothing on this fleet can execute;
# `--platform nvidia` compiles the same HIP through nvcc and is the only way this
# backend RUNS on Windows. `be_backend_name()` reports `hip-nvidia` for the
# second, and `ppf-contact-solver --backend` prints the linked name, so the two
# cannot be confused by a later reader. A green `--platform nvidia` leg therefore
# establishes that these HIP sources compile and run, and nothing at all about
# the code hipcc generates for the AMD targets.
import argparse
import concurrent.futures
import os
import subprocess
import sys


def fail(message):
    print(f"\nFAIL: {message}")
    sys.exit(1)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def diagnose(result, what):
    """Print what the compiler actually said, not a line matching 'error'.

    A grep for `error` reports NOTHING when a driver fails before it compiles
    anything, and an empty failure line reads as a mystery: measured here, the
    first NVIDIA-platform run printed `backend/backend.hip: FAIL` followed by a
    blank line, and the cause (a HIP header tree with no nvidia_detail) was in
    the output that was thrown away.
    """
    print(f"\nFAIL: {what}")
    for stream, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        text = (text or "").strip()
        if text:
            print(f"--- {stream} (last 40 lines) ---")
            print("\n".join(text.splitlines()[-40:]))
    sys.exit(1)


def targets_from_manifest(path):
    """The AMD target list, read from rocm_arch.txt as the Makefile reads it.

    An empty list stops the build rather than linking a library carrying no
    device code, which would surface much later as every GPU being rejected at
    run time and would name the device rather than the missing manifest.
    """
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(";", 1)[0].split()
            if len(parts) >= 2 and parts[0] == "target":
                out.append(parts[1])
    if not out:
        fail(f"no 'target' line in {path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--platform", choices=("amd", "nvidia"), required=True)
    ap.add_argument("--kernel-root", required=True,
                    help="the neutral kernel tree of ppf-cts-solver")
    ap.add_argument("--rocm-dir", required=True,
                    help="crates/ppf-cts-compute/rocm")
    ap.add_argument("--cuda-prologue-dir", required=True,
                    help="crates/ppf-cts-compute/cuda, needed by the NVIDIA leg")
    ap.add_argument("--kernelgen", required=True,
                    help="crates/ppf-cts-compute/seam/kernelgen.py")
    ap.add_argument("--gen-def", required=True,
                    help="build-win-native/scripts/gen_def.py")
    ap.add_argument("--out-dir", required=True,
                    help="build directory; renderings and objects land under it")
    ap.add_argument("--hip-include", required=True,
                    help="directory holding hip/hip_runtime.h; named with -I on "
                         "the NVIDIA leg and validated but not added on the AMD "
                         "one, where hipcc supplies its own")
    ap.add_argument("--rocm-path", default=os.environ.get("ROCM_PATH", ""),
                    help="ROCm SDK root, for bin/hipcc.exe on the AMD leg")
    ap.add_argument("--cuda-path", default=os.environ.get("CUDA_PATH", ""),
                    help="CUDA toolkit root, for bin/nvcc.exe on the NVIDIA leg")
    ap.add_argument("--arch", default="sm_89",
                    help="NVIDIA architecture for the staging leg")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                    help="compiles run at once; every one is its own process")
    args = ap.parse_args()

    nvidia = args.platform == "nvidia"
    kernels = os.path.abspath(args.kernel_root)
    rocm = os.path.abspath(args.rocm_dir)
    out = os.path.abspath(args.out_dir)
    # The renderings are shared between the platforms because they are the same
    # bytes; the OBJECTS are not, and must not share a directory. See the
    # Makefile's note on the same hazard: a switch of platform changes no source,
    # so a shared object tree is up to date by mtime and the link would silently
    # take the other platform's objects.
    gen = os.path.join(out, "kernelgen")
    obj = os.path.join(out, "obj-" + args.platform)
    libdir = os.path.join(out, "lib")

    if not os.path.exists(os.path.join(kernels, "data.hpp")):
        fail(f"--kernel-root {kernels} does not hold data.hpp")

    # THE HIP HEADER TREE IS CHECKED BY CONTENT, NOT BY EXISTENCE, and this is
    # the check that would have saved a session. `hip/hip_runtime.h` dispatches
    # on a plain macro:
    #
    #     #elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
    #     #include <hip/nvidia_detail/nvidia_hip_runtime.h>
    #
    # so a distribution that ships hip_runtime.h WITHOUT hip/nvidia_detail leaves
    # the NVIDIA arm including a file that is not there. MEASURED: the AMD
    # "TheRock" Windows tarball (therock-dist-windows-multiarch-10.0.0) carries
    # 0 headers under hip/nvidia_detail, while the 7.2.4 Linux hip-dev package
    # carries 12. The failure does NOT name the missing directory: the include
    # fails, hipStream_t is never declared, and every generated launcher reports
    # `identifier "hipStream_t" is undefined` in a file nobody wrote. Supply the
    # headers from a distribution that has them.
    hip_inc = os.path.abspath(args.hip_include)
    runtime = os.path.join(hip_inc, "hip", "hip_runtime.h")
    if not os.path.exists(runtime):
        fail(f"--hip-include {hip_inc} does not hold hip/hip_runtime.h")
    if nvidia:
        shim = os.path.join(hip_inc, "hip", "nvidia_detail", "nvidia_hip_runtime.h")
        if not os.path.exists(shim):
            fail(f"--hip-include {hip_inc} holds hip/hip_runtime.h but no "
                 f"hip/nvidia_detail/nvidia_hip_runtime.h, so the NVIDIA arm of "
                 f"that header includes a file that is not there and every "
                 f"generated launcher fails on an undefined hipStream_t. Point "
                 f"--hip-include at a HIP distribution that ships nvidia_detail.")

    if nvidia:
        cc = os.path.join(args.cuda_path, "bin", "nvcc.exe")
        if not os.path.exists(cc):
            fail(f"--cuda-path {args.cuda_path} has no bin/nvcc.exe")
        # NVCC DIRECTLY RATHER THAN hipcc. On this platform hipcc's whole job is
        # to add the macro and the include path and then call nvcc, and its
        # Windows wrapper has its own argument-splitting behavior to work around
        # (`-Xcompiler -fPIC` as two tokens is measured to lose the flag). What
        # selects the platform is `__HIP_PLATFORM_NVIDIA__`, which hip_runtime.h
        # tests, so naming the macro and the headers here needs no driver.
        #
        # `-x cu` because nvcc dispatches on the FILE EXTENSION and does not know
        # `.hip`. It is a COMPILE-ONLY flag: on a link line it makes the compiler
        # read the `.obj` files as source text.
        #
        # `-rdc=true` is nvcc's `-fgpu-rdc`, and it is required rather than an
        # optimization: the arena's device-side globals are defined in the ROCm
        # directory's own translation unit and referenced from every generated
        # entry.
        #
        # The CUDA prologue directory is required and reads like a mistake.
        # `seam/seam.hpp` tests `__CUDACC__` BEFORE `__HIPCC__`, and nvcc defines
        # it, so seam_cuda.cuh is the correct prologue here and seam_hip.hiph is
        # NOT exercised by this leg, which is the blind spot this platform
        # branch leaves for an AMD build to cover.
        flags = ["-std=c++17", "-O3", "-x", "cu", f"-arch={args.arch}",
                 "-rdc=true", "-D__HIP_PLATFORM_NVIDIA__",
                 "--expt-relaxed-constexpr", "-Xcompiler=/MD",
                 "-I", hip_inc, "-I", os.path.abspath(args.cuda_prologue_dir)]
        # THE LINK NEEDS `/MD` AND `-lcudart` TOO, not only the compile, and the
        # failure names neither. The objects are compiled `/MD`, so every CRT
        # call goes through an import thunk; a link line that does not select
        # the same CRT leaves those unresolved as `__imp__fdsign` and
        # `__imp__wassert`, which read as a missing math or assert
        # implementation rather than as a CRT selection. Measured: 34 unresolved
        # externals across all 84 objects, from a build whose every compile
        # succeeded. `-lcudart` is the runtime this platform's HIP calls resolve
        # to. build.bat's own nvcc device link spells both the same way.
        link_flags = ["-shared", f"-arch={args.arch}", "-rdc=true",
                      "-Xcompiler=/MD", "-lcudart"]
        print(f"platform: nvidia ({args.arch}), nvcc, HIP headers {hip_inc}")
    else:
        cc = os.path.join(args.rocm_path, "bin", "hipcc.exe")
        if not os.path.exists(cc):
            fail(f"--rocm-path {args.rocm_path} has no bin/hipcc.exe")
        offload = [f"--offload-arch={t}"
                   for t in targets_from_manifest(os.path.join(rocm, "rocm_arch.txt"))]
        # NO FAST MATH, AND THAT IS A CORRECTNESS SETTING rather than a speed
        # one: -ffast-math selects device-library variants through the oclc_*
        # control globals, one of which turns off the correctly rounded sqrtf
        # this tree mandates at the ACCD sites.
        flags = ["-std=c++17", "-O3", "-fgpu-rdc"] + offload
        link_flags = ["-shared", "-fgpu-rdc"] + offload
        print(f"platform: amd, hipcc, targets "
              f"{' '.join(t.split('=')[1] for t in offload)}")

    sources = sorted(
        os.path.join(root, name)
        for root, _dirs, names in os.walk(kernels)
        for name in names if name.endswith(".kernel.cpp"))
    if not sources:
        fail(f"no *.kernel.cpp under {kernels}")

    # RENDERED N AT A TIME, IN THE SORTED ORDER. Four kernelgen processes per
    # file, 93 files, was over 370 interpreter starts one after another (the
    # CUDA half of the same render measured 4 min 10 s serially on 8 vCPUs).
    # Each file is independent; the fragments are collected by index over the
    # sorted list, so kernel_table.inc is byte-for-byte the serial walk's.
    print(f"=== render ({len(sources)} neutral sources, {args.jobs} at a time) ===")

    def render_one(source):
        stem = os.path.relpath(source, kernels)[: -len(".kernel.cpp")]
        base = os.path.join(gen, stem)
        os.makedirs(os.path.dirname(base), exist_ok=True)
        # `diagfile` is rendered for EVERY source, entry or not: an assert does
        # not need an entry to exist. It needs `--kernel-root` because the path
        # it stores is relative to that root.
        for emit, suffix in (("entry", ".entry.hip"), ("args", ".args.hiph"),
                             ("body", ".kernel.hip"), ("table", ".table.inc"),
                             ("diagfile", ".diagfile.inc")):
            cmd = [args.python, args.kernelgen, "--target", "hip",
                   "--emit", emit, "--out", base + suffix]
            if emit in ("table", "diagfile"):
                cmd += ["--kernel-root", kernels]
            cmd.append(source)
            result = run(cmd)
            if result.returncode:
                return (result, f"render {stem} {emit}")
        return (None, base + ".table.inc", base + ".diagfile.inc")

    fragments = []
    diag_fragments = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(render_one, source) for source in sources]
        for future in futures:
            result = future.result()
            if result[0] is not None:
                pool.shutdown(wait=False, cancel_futures=True)
                diagnose(result[0], result[1])
            fragments.append(result[1])
            diag_fragments.append(result[2])
    table = os.path.join(gen, "kernel_table.inc")
    with open(table, "w", encoding="utf-8") as handle:
        for fragment in fragments:
            handle.write(open(fragment, encoding="utf-8").read())
    print(f"  kernel_table.inc: {os.path.getsize(table)} bytes")
    # diag_files.inc turns a device report's file id back into a path.
    # `diagnostics.hip` includes it and is compiled below, so this build owes
    # the file exactly as the Makefile does.
    diag_table = os.path.join(gen, "diag_files.inc")
    with open(diag_table, "w", encoding="utf-8") as handle:
        for fragment in diag_fragments:
            handle.write(open(fragment, encoding="utf-8").read())
    print(f"  diag_files.inc: {os.path.getsize(diag_table)} bytes")

    def compile_one(source, output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
        return run([cc] + flags +
                   ["-I", kernels, "-I", rocm, "-I", gen,
                    "-I", os.path.dirname(source),
                    "-include", "data.hpp", "-include", "linalg/eigsolve.hpp",
                    "-c", source, "-o", output])

    # THE COMPILES RUN IN PARALLEL AND THE LINK ORDER DOES NOT. Each object is
    # its own process with its own output file, so they share nothing, and on the
    # AMD platform each is a full device compile for every target in
    # rocm_arch.txt: one at a time, the eight-target build produced about 1.5
    # objects a minute. The objects list is still built in the fixed
    # order below, mechanism first and entries in the sorted walk, so the link
    # line does not depend on which compile finished first, and a failure is
    # reported for the first failing object in that order.
    jobs = []
    for rel in ("backend/backend.hip", "arena/arena.hip",
                "diagnostics/diagnostics.hip", "utility/dispatcher.hip"):
        output = os.path.join(obj, "mech_" + rel.replace("/", "_")[:-4] + ".obj")
        jobs.append((f"compile {rel}", os.path.join(rocm, rel), output))
    mechanism = len(jobs)
    entryless = 0
    for source in sources:
        stem = os.path.relpath(source, kernels)[: -len(".kernel.cpp")]
        entry = os.path.join(gen, stem + ".entry.hip")
        if "declares no [[seam::args]]" in open(entry, encoding="utf-8").read():
            entryless += 1
            continue
        output = os.path.join(obj, "entry_" + stem.replace(os.sep, "_") + ".obj")
        jobs.append((f"compile entry {stem}", entry, output))

    print(f"=== {mechanism} mechanism and {len(jobs) - mechanism} entry objects, "
          f"{args.jobs} at a time ===")
    objects = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(compile_one, source, output) for _what, source, output in jobs]
        for (what, _source, output), future in zip(jobs, futures):
            result = future.result()
            if result.returncode:
                pool.shutdown(wait=False, cancel_futures=True)
                diagnose(result, what)
            print(f"  {what}: ok")
            objects.append(output)
    print(f"  compiled={len(jobs) - mechanism} entry-less={entryless}")

    print("=== link ===")
    os.makedirs(libdir, exist_ok=True)
    library = os.path.join(libdir, "libppfbe_rocm.dll")
    # WINDOWS EXPORTS NOTHING FROM A DLL UNLESS TOLD TO, and has no
    # export-all-symbols equivalent. `extern "C"` is enough on Linux, where a
    # shared library exports every default-visibility symbol; here it is not, and
    # the failure is SILENT: the DLL links and its import library comes out 1192
    # bytes with no be_* symbol in it. gen_def.py reads the EXPORTS list from the
    # ABI HEADER, which is the single source of truth, so the list cannot drift
    # from the declarations. build.bat already uses it for the CUDA DLL.
    module = os.path.join(out, "backend-exports.def")
    result = run([args.python, args.gen_def,
                  os.path.join(kernels, "seam", "backend_abi.h")])
    if result.returncode or not result.stdout.strip():
        diagnose(result, "gen_def.py produced no export list")
    open(module, "w", encoding="utf-8").write(result.stdout)
    exports = sum(1 for line in result.stdout.splitlines()
                  if line.strip() and not line.strip().upper().startswith("EXPORTS"))
    print(f"  exports: {exports}")

    # ON THE AMD PLATFORM THE OBJECTS GO IN A RESPONSE FILE. hipcc runs clang
    # through the shell, so the link line meets cmd.exe's 8191-character limit,
    # and with every object named by its absolute path it passes that: measured
    # with 84 objects under C:\ppf-contact-solver, "The command line is too
    # long." clang expands @file itself and reads it with GNU quoting, where a
    # backslash is an escape, so a path written as-is arrives as
    # C:ppf-contact-solvercrates...
    # and every object is "no such file". The paths are written with forward
    # slashes, which Windows accepts. nvcc on the NVIDIA platform is handed the
    # objects directly, as before.
    if nvidia:
        inputs = objects
    else:
        response = os.path.join(out, "link-objects.rsp")
        with open(response, "w", encoding="utf-8") as handle:
            for obj in objects:
                handle.write('"' + obj.replace("\\", "/") + '"\n')
        inputs = ["@" + response]
    result = run([cc] + link_flags + ["-Xlinker", "/DEF:" + module,
                                      "-o", library] + inputs)
    if result.returncode:
        diagnose(result, "link")
    implib = library[:-4] + ".lib"
    if not os.path.exists(implib):
        fail(f"the DLL linked but produced no import library beside it: {implib}")
    print(f"  {library}: {os.path.getsize(library)} bytes")
    print(f"  {implib}: {os.path.getsize(implib)} bytes")
    print("BUILD_ROCM_DONE")


main()
