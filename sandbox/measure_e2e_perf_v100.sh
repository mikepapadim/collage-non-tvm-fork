#!/bin/bash

# 1) Make sure you have right PROJ_PATH

# Currently available networks:
# resnet50, resnext50, nasnetamobile
#PROJ_PATH=~/backend-aware-graph-opt
#cd $PROJ_PATH

# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=0 --> RTX2070
# "resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta"
CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasneta -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50_3d -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n bert -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n dcgan -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasrnn -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n mobilenet_v2 -hw v100

# PyTorch, TF, TF-XLA measurement
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch_tf.py -n resnet50 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch_tf.py -n resnext50_32x4d -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch_tf.py -n nasneta -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch_tf.py -n nasrnn -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch_tf.py -n bert -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch_tf.py -n mobilenet_v2 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch_tf.py -n resnet50_3d -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_pytorch_tf.py -n dcgan -hw v100

# Batch size of 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50 -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasneta -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n resnet50_3d -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n mobilenet_v2 -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n bert -hw v100 -bs 8
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_end_to_end.py -n nasrnn -hw v100 -bs 8
