#!/bin/bash

cd $(dirname $0)

python docker_launcher.py \
    --tag-prefix "test" \
    --world-size 3 \
    $@ \
    ../examples/mpc_linear_svm
