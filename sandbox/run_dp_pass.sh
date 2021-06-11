#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=1 --> RTX2070
# "resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta", "conv2d"
CUDA_VISIBLE_DEVICES=1 python3 testing/test_relay_build -n resnet50
#CUDA_VISIBLE_DEVICES=1 python3 testing.test_relay_build -n resnext50_32x4d
#CUDA_VISIBLE_DEVICES=1 python3 testing.test_relay_build -n nasrnn
#CUDA_VISIBLE_DEVICES=1 python3 testing.test_relay_build -n nasneta
#CUDA_VISIBLE_DEVICES=1 python3 testing.test_relay_build -n bert
#CUDA_VISIBLE_DEVICES=1 python3 testing.test_relay_build -n conv2d
#CUDA_VISIBLE_DEVICES=1 python3 testing.test_relay_build -n conv2d+relu
#CUDA_VISIBLE_DEVICES=1 python3 testing.test_relay_build -n conv2d+relu_x2


