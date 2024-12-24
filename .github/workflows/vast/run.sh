# File: run.sh
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

SSH=/tmp/vast-ci/ssh-command.sh
REMOTE_WORKDIR=/root/ppf-contact-solver

ARG=$1
if [ "$ARG" = "warmup" ]; then
    $SSH "apt update"
    $SSH "cd ppf-contact-solver; python3 warmup.py"
elif [ "$ARG" = "build" ]; then
    $SSH "cd ppf-contact-solver; /root/.cargo/bin/cargo build --release"
elif [ "$ARG" = "convert" ]; then
    $SSH "cd ppf-contact-solver/examples; jupyter nbconvert --to script *.ipynb"
    $SSH "touch /root/ppf-contact-solver/examples/.CLI"
elif [ "$ARG" = "run" ]; then
    EXAMPLE=$2
    $SSH "cd ppf-contact-solver/examples; python3 $EXAMPLE"
fi