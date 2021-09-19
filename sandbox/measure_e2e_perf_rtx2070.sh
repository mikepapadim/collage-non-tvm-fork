#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=1 --> RTX2070
# "resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta"
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n nasneta -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n resnet50_3d -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n bert -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n dcgan -hw rtx2070
CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n bert_full -hw rtx2070

#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n resnet50 -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n nasrnn -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n mobilenet_v2 -hw rtx2070


# PyTorch measurement
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n resnext50_32x4d -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n nasneta -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n bert -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n resnet50_3d -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n dcgan -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n bert_full -hw rtx2070

#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n resnet50 -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n nasrnn -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n mobilenet_v2 -hw rtx2070

# TF, TF-XLA measurement
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda TF_XLA_FLAGS="--tf_xla_auto_jit=2"
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=1 python3 testing/baselines/tf2/bert.py -hw rtx2070
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=1 python3 testing/baselines/tf2/dcgan.py -hw rtx2070
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=1 python3 testing/baselines/tf2/resnext50.py -hw rtx2070
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=1 python3 testing/baselines/tf2/nasnet_a.py -hw rtx2070
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=1 python3 testing/baselines/tf2/resnet50_3d.py -hw rtx2070

# Batch size of 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n resnet50 -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n nasneta -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n resnet50_3d -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n mobilenet_v2 -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n bert -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_end_to_end.py -n nasrnn -hw rtx2070 -bs 8

# PyTorch measurement
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n resnext50_32x4d -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n nasneta -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n bert -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n resnet50_3d -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n dcgan -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n resnet50 -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n nasrnn -hw rtx2070 -bs 8
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_pytorch.py -n mobilenet_v2 -hw rtx2070 -bs 8

# TF, TF-XLA measurement
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda TF_XLA_FLAGS="--tf_xla_auto_jit=2"
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=1 python3 testing/baselines/tf2/bert.py -hw rtx2070 -bs 8
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=1 python3 testing/baselines/tf2/dcgan.py -hw rtx2070 -bs 8
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=1 python3 testing/baselines/tf2/resnext50.py -hw rtx2070 -bs 8
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=1 python3 testing/baselines/tf2/nasnet_a.py -hw rtx2070 -bs 8
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=1 python3 testing/baselines/tf2/resnet50_3d.py -hw rtx2070 -bs 8