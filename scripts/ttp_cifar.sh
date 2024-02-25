#!/bin/bash

cd $(dirname $0)

python docker_launcher.py \
    --tag-prefix "test" \
    --world-size 3 \
    $@ \
    ../examples/mpc_cifar \
    --epochs 0 \
    --device cuda \
    --batch-size 4 \
    --print-freq 10 \
    --model-location model/cifar-model-checkpoint.pth.tar \
    --evaluate \
    --skip-plaintext \
    --data-dir downloads/cifar
