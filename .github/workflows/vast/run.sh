# File: run.sh
# Author: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

SSH=/tmp/vast-ci/ssh-command.sh
REMOTE_WORKDIR=/root/ppf-contact-solver

ARG=$1
if [ "$ARG" = "build" ]; then
    $SSH "cd ppf-contact-solver && /root/.cargo/bin/cargo build --release"
elif [ "$ARG" = "convert" ]; then
    $SSH "cd ppf-contact-solver/examples && jupyter nbconvert --to script *.ipynb"
    $SSH "touch /root/ppf-contact-solver/frontend/.CI"
elif [ "$ARG" = "run" ]; then
    EXAMPLE=$2
    $SSH "PYTHONPATH=/root/ppf-contact-solver python3 ppf-contact-solver/examples/$EXAMPLE"
fi