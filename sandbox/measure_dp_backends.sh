#!/bin/bash

# 1) Make sure you have right PROJ_PATH

# Currently available networks:
# resnet50, resnext50, nasnetamobile
#PROJ_PATH=~/backend-aware-graph-opt
#cd $PROJ_PATH

# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=0 --> RTX2070
# "resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta"
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_dp_backends.py -n resnext50_32x4d -hw rtx2070
CUDA_VISIBLE_DEVICES=1 python3 testing/measure_dp_backends.py -n dcgan -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_dp_backends.py -n nasneta -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_dp_backends.py -n resnet50_3d -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/measure_dp_backends.py -n bert_full -hw rtx2070

#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_dp_backends.py -n resnext50_32x4d -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_dp_backends.py -n nasneta -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_dp_backends.py -n resnet50_3d -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_dp_backends.py -n dcgan -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/measure_dp_backends.py -n bert_full -hw v100
