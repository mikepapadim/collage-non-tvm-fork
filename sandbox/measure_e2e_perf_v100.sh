#!/bin/bash

# 1) Make sure you have right PROJ_PATH

# Currently available networks:
# resnet50, resnext50, nasnetamobile
#PROJ_PATH=~/backend-aware-graph-opt
#cd $PROJ_PATH

# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=0 --> RTX2070
# "resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta"
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasneta -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50_3d -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n dcgan -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n bert_full -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n yolov3 -hw v100

#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n bert -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasrnn -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n mobilenet_v2 -hw v100

# PyTorch, TF, TF-XLA measurement
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n resnext50_32x4d -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n nasneta -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n resnet50_3d -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n dcgan -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n bert_full -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n yolov3 -hw v100

#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n bert -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n resnet50 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n mobilenet_v2 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n nasrnn -hw v100

# Different batch size test
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw v100 -bs 4
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n resnext50_32x4d -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n resnext50_32x4d -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n resnext50_32x4d -hw v100 -bs 4

XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/resnext50.py -hw v100 -bs 16
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/resnext50.py -hw v100 -bs 8
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/resnext50.py -hw v100 -bs 4

# TF, TF-XLA measurement
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda TF_XLA_FLAGS="--tf_xla_auto_jit=2"
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/bert.py -hw v100
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/dcgan.py -hw v100
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/resnext50.py -hw v100
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/nasnet_a.py -hw v100
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/resnet50_3d.py -hw v100

# Batch size of 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n bert -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasneta -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50_3d -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n dcgan -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n mobilenet_v2 -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50 -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasrnn -hw v100 -bs 16

# PyTorch, TF, TF-XLA measurement
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n resnet50 -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n resnext50_32x4d -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n nasneta -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n nasrnn -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n bert -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n mobilenet_v2 -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n resnet50_3d -hw v100 -bs 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch.py -n dcgan -hw v100 -bs 16

# TF, TF-XLA measurement
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda TF_XLA_FLAGS="--tf_xla_auto_jit=2"
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/bert.py -hw v100 -bs 16
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/dcgan.py -hw v100 -bs 16
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/resnext50.py -hw v100 -bs 16
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/nasnet_a.py -hw v100 -bs 16
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 python3 testing/baselines/tf2/resnet50_3d.py -hw v100 -bs 16


# Batch size of 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50 -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasneta -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50_3d -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n mobilenet_v2 -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n bert -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasrnn -hw v100 -bs 8
