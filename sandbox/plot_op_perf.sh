#!/bin/bash

# 1) Make sure you have right PROJ_PATH

# Currently available networks:
# resnet50, resnext50, nasnetamobile

# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=1 --> RTX2070
CUDA_VISIBLE_DEVICES=1 python3 testing/plot_ops.py 