#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=1 --> RTX2070

## Make sure you uncomment visualization code in "get_user_fusion" function in _optimizer.py
python3 testing/plot_e2e_perf.py -hw rtx2070 -bs 1
python3 testing/plot_e2e_perf.py -hw v100 -bs 1
python3 testing/plot_e2e_perf.py -hw xeon -bs 1
python3 testing/plot_e2e_perf.py -hw diff_batch_v100 -bs 1


# Excluded models for the paper
#CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n resnet50 -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n nasrnn -hw rtx2070
#CUDA_VISIBLE_DEVICES=1 python3 testing/plot_backend_placement.py -n mobilenet_v2 -hw rtx2070
