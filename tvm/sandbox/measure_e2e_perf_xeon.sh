#!/bin/bash

# 1) Make sure you have right PROJ_PATH

# Currently available networks:
#PROJ_PATH=~/backend-aware-graph-opt
#cd $PROJ_PATH

# "resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta"
#python3 testing/measure_end_to_end.py -n resnet50 -hw xeon
#python3 testing/measure_end_to_end.py -n mobilenet_v2 -hw xeon

# Worse than the best baseline
#python3 testing/measure_end_to_end.py -n nasneta -hw xeon
#python3 testing/measure_end_to_end.py -n dcgan -hw xeon

# Pass
#python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw xeon
#python3 testing/measure_end_to_end.py -n resnet50_3d -hw xeon
#python3 testing/measure_end_to_end.py -n bert_full -hw xeon

#python3 testing/measure_end_to_end.py -n bert -hw xeon

# Fail
#python3 testing/measure_end_to_end.py -n nasrnn -hw xeon

# PyTorch measurement
python3 testing/measure_pytorch.py -n resnext50_32x4d -hw xeon
python3 testing/measure_pytorch.py -n nasneta -hw xeon
python3 testing/measure_pytorch.py -n resnet50_3d -hw xeon
python3 testing/measure_pytorch.py -n dcgan -hw xeon
python3 testing/measure_pytorch.py -n bert_full -hw xeon

# TF, TF-XLA measurement
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda TF_XLA_FLAGS="--tf_xla_auto_jit=2"
python3 testing/baselines/tf2/resnext50.py -hw xeon
python3 testing/baselines/tf2/nasnet_a.py -hw xeon
python3 testing/baselines/tf2/resnet50_3d.py -hw xeon
python3 testing/baselines/tf2/dcgan.py -hw xeon
python3 testing/baselines/tf2/bert_full_scratch.py -hw xeon
