#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=1 --> RTX2070
# "resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta", "conv2d"
CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n resnet50
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n resnext50_32x4d
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n nasneta
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n nasrnn
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n bert

# Relay examples
#CUDA_VISIBLE_DEVICES=0 python3 testing/test_relay_build.py -n annotate_test
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n conv2d
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n conv2d+relu
#CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build.py -n conv2d+relu_x2


