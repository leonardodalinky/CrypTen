#!/bin/bash

cd $(dirname $0)

python docker_launcher.py \
    --tag-prefix "linear_svm" \
    $@ \
    ../examples/mpc_linear_svm
