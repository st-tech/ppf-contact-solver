#!/usr/bin/env bash
# File: build-mac-native/config.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Local settings for the macOS build, sourced by warmup.sh, build.sh,
# bundle.sh and the start.sh that build.sh generates. Source it, do not
# execute it.
#
# Every setting reads an already-exported value first, so anything here can be
# overridden for one run from the environment:
#
#     PORT=8888 ./build.sh
#
# There is no `set -euo pipefail` in this file. It is sourced, and those
# options would leak into the shell of whoever sources it.
#
# THIS FILE IS TRACKED IN GIT. NO SECRET GOES IN IT. The signing settings at
# the bottom name an identity and a keychain profile, which are not secrets;
# an app-specific password is, and belongs in the environment or in a
# notarytool keychain profile, never in this file.

# JupyterLab port for the launcher build.sh writes.
PORT="${PORT:-8080}"

# The Python that carries the frontend dependencies. Leave empty and warmup.sh
# asks warmup.py's get_venv_path() for the canonical location, which is
# $HOME/.local/share/ppf-cts/venv and is also what the Metal fixture target
# defaults FIXTURE_PYTHON to (crates/ppf-cts-compute/metal/Makefile).
# Set it to put the environment somewhere else.
PPF_CTS_VENV="${PPF_CTS_VENV:-}"

# The interpreter warmup.sh builds that environment with. Leave empty and it
# searches PATH for python3.13, 3.12, 3.11, 3.10 and python3, taking the first
# that reports 3.10 or newer. macOS ships 3.9, which the frontend's PEP 604
# unions do not parse, so a stock machine needs one installed.
PPF_CTS_PYTHON="${PPF_CTS_PYTHON:-}"

# ---------------------------------------------------------------------------
# Code signing and notarization
# ---------------------------------------------------------------------------
#
# THIS IS THE SINGLE POINT WHERE APPLE DEVELOPER CREDENTIALS ARE SUPPLIED.
# None of them are available in this project today, so bundle.sh ad-hoc signs
# by default and does not notarize at all. What each setting does:
#
# MAC_CODESIGN_IDENTITY
#     The name of a codesigning identity in the login keychain, as
#     `security find-identity -v -p codesigning` prints it. For a bundle that
#     will be distributed this is a "Developer ID Application: NAME (TEAMID)"
#     certificate. Empty means bundle.sh signs ad-hoc (`codesign --sign -`),
#     which is required for an arm64 binary to run at all once its load
#     commands have been rewritten, and which Gatekeeper rejects on any
#     machine that did not build it.
#
# MAC_NOTARY_KEYCHAIN_PROFILE
#     The name of a notarytool keychain profile, created once with
#     `xcrun notarytool store-credentials`. bundle.sh does not notarize; it
#     prints the exact commands and names this profile in them. The
#     alternative to a profile is the triple Apple ID, team ID and an
#     app-specific password, which bundle.sh also prints. Do not put that
#     password here.
MAC_CODESIGN_IDENTITY="${MAC_CODESIGN_IDENTITY:-}"
MAC_NOTARY_KEYCHAIN_PROFILE="${MAC_NOTARY_KEYCHAIN_PROFILE:-}"
