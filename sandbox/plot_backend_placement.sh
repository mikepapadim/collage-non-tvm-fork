#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=1 --> RTX2070

## Make sure you uncomment visualization code in "get_user_fusion" function in _optimizer.py
CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n resnext50_32x4d -hw rtx2070
CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n nasneta -hw rtx2070
CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n bert -hw rtx2070
CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n resnet50_3d -hw rtx2070
CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n dcgan -hw rtx2070

# Excluded models for the paper
#CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n resnet50 -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n nasrnn -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n mobilenet_v2 -hw rtx2070
