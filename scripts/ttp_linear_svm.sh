#!/bin/bash

cd $(dirname $0)

python docker_launcher.py \
    --tag-prefix "linear_svm" \
    --world-size 3 \
    $@ \
    ../examples/mpc_linear_svm
