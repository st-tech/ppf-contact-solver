#!/usr/bin/env python3
# File: build-win-native/scripts/gen_cuda.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The transcompiler-generation half of the Windows CUDA build, the counterpart
# of what crates/ppf-cts-compute/cuda/Makefile does on Linux/macOS. build.bat
# calls this once; it renders every neutral *.kernel.cpp into the .cu / .args.cuh
# it needs, the .entry.cu for the kernels that declare an entry, and assembles
# the kernel table. It then prints the list of entry .cu files build.bat must
# compile, one per line, on stdout.
#
# WHY THIS IS PYTHON RATHER THAN build.bat. The kernel id is a table INDEX and
# the table is a SORTED WALK of the neutral tree, so the file order here must be
# byte-identical to the Linux `find ... | sort`. cmd.exe cannot enumerate a tree
# in a stable sorted order, and reproducing the Makefile's find/patsubst/grep-l
# in batch is where a silent divergence would hide. This mirrors the Makefile's
# logic exactly, in the one language the transcompiler is already written in.
#
# THE FILES ARE RENDERED N AT A TIME, AND THE ORDER OF EVERYTHING WRITTEN IS
# STILL THE SORTED WALK. Each kernel file is rendered by four or five fresh
# kernelgen processes, and with 93 files that is over 400 interpreter starts:
# measured serially on an 8 vCPU builder, 4 min 10 s with one core busy, which
# after the compiles were pooled was two thirds of the whole CUDA library
# phase. The per-file work is therefore submitted to a pool, and every list
# the tables and the entry list are built from is collected BY INDEX in the
# sorted order, never in completion order, so the kernel ids and the bytes of
# kernel_table.inc are what the serial walk produced.
import argparse
import concurrent.futures
import os
import re
import subprocess
import sys

ENTRY_ATTR = re.compile(r'^\[\[seam::(args|entry)(\(.*\))?\]\]')


def find_kernels(root):
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith('.kernel.cpp'):
                full = os.path.join(dirpath, name)
                out.append(os.path.relpath(full, root).replace('\\', '/'))
    # The Linux build is `find | sed 's|root/||' | sort`. Match that ordering,
    # which the kernel-id numbering depends on.
    return sorted(out)


def has_entry(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return any(ENTRY_ATTR.match(line) for line in handle)


class RenderFailed(Exception):
    pass


def run(kernelgen, args):
    # Raised, not exited: this runs on a pool thread, where sys.exit would
    # end the thread and nothing else. Output is captured so the pool's
    # workers do not interleave on the console; a failure prints its own.
    result = subprocess.run([sys.executable, '-B', kernelgen] + args,
                            capture_output=True, text=True, errors='replace')
    if result.returncode != 0:
        raise RenderFailed('kernelgen failed (exit %d): %s\n%s%s' % (
            result.returncode, ' '.join(args), result.stdout, result.stderr))


def render_one(root, gen, kernelgen, rel):
    """Render one kernel file.

    Returns (table.inc, table.rs, diagfile.inc, entry.cu or None).
    """
    src = os.path.join(root, rel.replace('/', os.sep))
    stem = rel[:-len('.kernel.cpp')]
    base = os.path.join(gen, stem.replace('/', os.sep))
    os.makedirs(os.path.dirname(base) or gen, exist_ok=True)

    run(kernelgen, ['--target', 'cu', '--out', base + '.kernel.cu', src])
    run(kernelgen, ['--target', 'cu', '--emit', 'args',
                    '--kernel-root', root, '--out', base + '.args.cuh', src])
    run(kernelgen, ['--target', 'cu', '--emit', 'table',
                    '--kernel-root', root, '--out', base + '.table.inc', src])
    run(kernelgen, ['--target', 'rust', '--emit', 'table',
                    '--kernel-root', root, '--out', base + '.table.rs', src])
    # EVERY source gets a diagnostic-file row, entry or not: an assert does not
    # need an entry to exist. `--kernel-root` is what the stored path is
    # relative to, so passing it is not optional here.
    run(kernelgen, ['--target', 'cu', '--emit', 'diagfile',
                    '--kernel-root', root, '--out', base + '.diagfile.inc', src])
    entry = None
    if has_entry(src):
        run(kernelgen, ['--target', 'cu', '--emit', 'entry',
                        '--kernel-root', root, '--out', base + '.entry.cu', src])
        entry = base + '.entry.cu'
    return (base + '.table.inc', base + '.table.rs',
            base + '.diagfile.inc', entry)


def pool_width():
    override = os.environ.get('PPF_WIN_BUILD_JOBS')
    if override:
        return max(1, int(override))
    return os.cpu_count() or 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernel-root', required=True)
    ap.add_argument('--kernelgen', required=True)
    ap.add_argument('--gen-dir', required=True)
    args = ap.parse_args()

    root = os.path.abspath(args.kernel_root)
    gen = os.path.abspath(args.gen_dir)
    kernelgen = os.path.abspath(args.kernelgen)

    kernels = find_kernels(root)
    if not kernels:
        sys.stderr.write('no *.kernel.cpp under %s\n' % root)
        sys.exit(1)

    entry_srcs = []
    table_inc_parts = []
    table_rs_parts = []
    diagfile_parts = []
    entry_cus = []

    # Submitted in sorted order and READ BACK in that same order (zip over the
    # sorted list, not as_completed), so every list below is the serial walk's.
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_width()) as pool:
        futures = [pool.submit(render_one, root, gen, kernelgen, rel) for rel in kernels]
        for rel, future in zip(kernels, futures):
            try:
                table_inc, table_rs, diagfile, entry = future.result()
            except RenderFailed as exc:
                pool.shutdown(wait=False, cancel_futures=True)
                sys.stderr.write('%s\n' % exc)
                sys.exit(1)
            table_inc_parts.append(table_inc)
            table_rs_parts.append(table_rs)
            diagfile_parts.append(diagfile)
            if entry is not None:
                entry_srcs.append(rel)
                entry_cus.append(entry)

    if not entry_srcs:
        sys.stderr.write('no kernel declares [[seam::args]] or [[seam::entry]]\n')
        sys.exit(1)

    # kernel_table.inc is the concatenation in sorted order; kernel_table.rs the
    # same wrapped in a Rust array literal. Mirrors the Makefile's cat rules.
    with open(os.path.join(gen, 'kernel_table.inc'), 'w', encoding='utf-8') as out:
        for part in table_inc_parts:
            with open(part, 'r', encoding='utf-8') as handle:
                out.write(handle.read())
    # diag_files.inc turns a device report's file id back into a path.
    # `diagnostics.cu` includes it, and that translation unit is compiled here
    # too, so the Windows build owes this file exactly as the Makefile does.
    with open(os.path.join(gen, 'diag_files.inc'), 'w', encoding='utf-8') as out:
        for part in diagfile_parts:
            with open(part, 'r', encoding='utf-8') as handle:
                out.write(handle.read())
    with open(os.path.join(gen, 'kernel_table.rs'), 'w', encoding='utf-8') as out:
        out.write('[\n')
        for part in table_rs_parts:
            with open(part, 'r', encoding='utf-8') as handle:
                out.write(handle.read())
        out.write(']\n')
    with open(os.path.join(gen, 'entry-sources.txt'), 'w', encoding='utf-8') as out:
        for rel in entry_srcs:
            out.write(rel + '\n')

    # The entry objects build.bat must compile, absolute paths, one per line.
    for path in entry_cus:
        sys.stdout.write(path + '\n')


if __name__ == '__main__':
    main()
