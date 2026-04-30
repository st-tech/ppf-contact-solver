#!/bin/bash
# Build the full docs site from scratch.
#
#   - Bootstraps a local virtualenv under docs/.venv on first run.
#   - Regenerates the Blender Python API reference from blender_addon/ops/api.py
#     (fails loudly if any @blender_api symbol is missing a docstring).
#   - Runs sphinx-build, which pulls in the frontend package via autodoc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
REQS="$SCRIPT_DIR/requirements.txt"
STAMP="$VENV/.requirements.stamp"

# Pick a Python >= 3.10 interpreter. The frontend package (pulled in by
# autodoc) uses PEP 604 "X | Y" union syntax, which 3.9 rejects at import
# time — and macOS's /usr/bin/python3 is still 3.9.
find_python() {
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

if [ ! -x "$VENV/bin/python" ]; then
    PYTHON="$(find_python)" || {
        echo "ERROR: need python >= 3.10 on PATH (tried python3.13 … python3.10, python3)" >&2
        exit 1
    }
    echo "==> Creating docs venv at $VENV (using $PYTHON -> $(command -v "$PYTHON"))"
    "$PYTHON" -m venv "$VENV"
fi

# Reinstall deps when requirements.txt is newer than the last install.
if [ ! -f "$STAMP" ] || [ "$REQS" -nt "$STAMP" ]; then
    echo "==> Installing docs requirements"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r "$REQS"
    touch "$STAMP"
fi

echo "==> Regenerating Blender Python API reference"
"$VENV/bin/python" "$SCRIPT_DIR/generate_blender_api_reference.py"

echo "==> Regenerating MCP tool reference"
"$VENV/bin/python" "$SCRIPT_DIR/generate_mcp_reference.py"

echo "==> Regenerating frontend parameter / log reference"
"$VENV/bin/python" "$SCRIPT_DIR/generate_frontend_params_reference.py"

echo "==> Removing previous build output"
rm -rf "$SCRIPT_DIR/_build"

echo "==> Running sphinx-build"
# -W --keep-going: treat warnings as errors but keep building so every
# problem is reported in one pass.  This is what traces docstring-level
# issues (napoleon parse errors, malformed ``Example:`` code blocks,
# unknown directives) through to a failing exit code — otherwise the
# build would succeed silently and the broken page would ship.
"$VENV/bin/sphinx-build" -W --keep-going -b html "$SCRIPT_DIR" "$SCRIPT_DIR/_build"

echo "==> Done. Open $SCRIPT_DIR/_build/index.html"
