#!/usr/bin/env bash
# File: build-linux-native/config.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Local settings for the Linux build, sourced by warmup.sh, build.sh, bundle.sh
# and the start.sh that build.sh generates. Source it, do not execute it.
#
# Every setting reads an already-exported value first, so anything here can be
# overridden for one run from the environment:
#
#     PORT=8888 ./build.sh
#
# There is no `set -euo pipefail` in this file. It is sourced, and those
# options would leak into the shell of whoever sources it.
#
# THIS FILE IS TRACKED IN GIT. Nothing machine-specific belongs in the tracked
# copy; override for a run from the environment instead.

# JupyterLab port for the launcher build.sh writes.
PORT="${PORT:-8080}"

# The backends build.sh builds and bundle.sh ships, space separated. Empty means
# the default decided in scripts/platform.sh: CUDA and the CPU backend, on x86_64
# and aarch64 alike. Set it to "rocm cpu" for the ROCm backend (x86_64 only), or
# to "cpu" for a CPU-only build. The CPU backend is part of every set.
PPF_LINUX_BACKENDS="${PPF_LINUX_BACKENDS:-}"

# The Python that carries the frontend dependencies for the developer path.
# Leave empty and warmup.sh asks warmup.py's get_venv_path() for the canonical
# location, which is $HOME/.local/share/ppf-cts/venv. Set it to keep a separate
# environment, for example on a host where other work shares that one.
PPF_CTS_VENV="${PPF_CTS_VENV:-}"

# The interpreter warmup.sh builds that environment with. Leave empty and it
# searches PATH for python3.13, 3.12, 3.11, 3.10 and python3, taking the first
# that reports 3.10 or newer.
PPF_CTS_PYTHON="${PPF_CTS_PYTHON:-}"

# A directory of wheels to install from INSTEAD of a package index. Empty means
# the index. Set it on a host with no route to PyPI, after filling the
# directory elsewhere with `pip download --only-binary=:all:` for the same
# Python version and platform. warmup.sh passes --no-index when it is set, so an
# incomplete directory fails by package name rather than reaching for a network.
PPF_LINUX_WHEELHOUSE="${PPF_LINUX_WHEELHOUSE:-}"
