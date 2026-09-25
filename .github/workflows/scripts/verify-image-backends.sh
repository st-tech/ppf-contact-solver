#!/bin/bash
# File: .github/workflows/scripts/verify-image-backends.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Assert that a built image carries BOTH backends and that the two things which
# read them agree about it.
#
#   verify-image-backends.sh <image> [<docker>]
#
# RUN WITHOUT `--gpus`, ON PURPOSE. The image ships a CUDA build and a CPU
# build, and the case the CPU one exists for is a machine with no usable GPU.
# Dropping the flag produces exactly that machine even on a GPU host, so this
# checks what a reader on a laptop gets: both build directories present and
# marked, the CPU solver answering for itself, and the frontend's automatic rule
# landing on it rather than failing or running the wrong one.
#
# WHAT WOULD GO WRONG WITHOUT IT. The two builds live in two directories because
# `crates/ppf-cts-solver/build.rs` refuses to put a second backend in a
# directory that already holds one, and nothing in a normal image test looks at
# the second directory: a dropped COPY, a marker left behind, or a `--features
# cpu` build that silently stopped being made would leave an image that passes
# every GPU smoke test and offers one device. The marker is what both readers
# key on, so it is checked by content and not just for existence.
set -euo pipefail

IMAGE=${1:?usage: verify-image-backends.sh <image> [<docker>]}
DOCKER=${2:-docker}
ROOT=/root/ppf-contact-solver

run() { $DOCKER run --rm "$IMAGE" /bin/sh -c "$1"; }

echo "== markers =="
run "cat $ROOT/target/release/.ppf-backend; echo; cat $ROOT/target/cpu/release/.ppf-backend; echo"

echo "== both directories hold a full set, and each solver answers for itself =="
run "
set -e
cd $ROOT
test \"\$(cat target/release/.ppf-backend)\" = cuda
test \"\$(cat target/cpu/release/.ppf-backend)\" = cpu
test -x target/release/ppf-cts-server
test -x target/cpu/release/ppf-cts-server
test -f target/release/lib_ppf_cts_py.so
test -f target/cpu/release/lib_ppf_cts_py.so
test \"\$(./target/cpu/release/ppf-contact-solver --backend)\" = cpu
echo 'both build directories are complete'
"

echo "== the frontend sees both, and with no GPU it chooses the CPU one and says why =="
run "
cd $ROOT
PYTHONPATH=$ROOT /root/.local/share/ppf-cts/venv/bin/python /dev/stdin <<'PYEOF'
from frontend import App

built = sorted(App.list_backends())
print('built:', built)
assert built == ['cpu', 'cuda'], built

# No GPU is visible in this container, so the automatic rule has to land on the
# CPU build. It prints one line saying so before answering, which is the whole
# point: a substitution the reader is not told about is the thing to avoid.
chosen = App.get_backend()
print('chosen:', chosen)
assert chosen == 'cpu', chosen
PYEOF
"

echo "BOTH_BACKENDS_OK"
