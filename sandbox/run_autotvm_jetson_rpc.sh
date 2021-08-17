#!/bin/bash

# 1) Make sure you have right PROJ_PATH
# 2) Don't forget to type following command before you autotune ops
# - conda activate tvm_fleet

# Currently available networks:
# resnet50, resnext50, nasnetamobile
#PROJ_PATH=~/backend-aware-graph-opt
#cd $PROJ_PATH

# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=1 --> RTX2070
CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay_rpc.py -tu autotvm -t cuda -th llvm -dt float32 -n resnet50 -l autotvm_ops -bs 1 -hw jetson-1
#CUDA_VISIBLE_DEVICES=1 python3 testing/autotune_relay_rpc.py -tu autotvm -t cuda -th llvm -dt float32 -n resnext50_32x4d -l autotvm_ops -bs 1 -hw jetson-1
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay_rpc.py -tu autotvm -t cuda -th llvm -dt float32 -n nasneta -l autotvm_ops -bs 1 -hw jetson-1
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay_rpc.py -tu autotvm -t cuda -th llvm -dt float32 -n nasrnn -l autotvm_ops -bs 1 -hw jetson-1
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay_rpc.py -tu autotvm -t cuda -th llvm -dt float32 -n resnet50_3d -l autotvm_ops -bs 1 -hw jetson-1
#CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/ejaeaesa/tvm/python:$PYTHONPATH /opt/anaconda3/envs/tvm_fleet/bin/python testing/autotune_relay_rpc.py -tu autotvm -t cuda -th llvm -dt float32 -n mobilenet_v2 -l autotvm_ops -bs 1 -hw jetson-1
#CUDA_VISIBLE_DEVICES=1 python3 testing/autotune_relay_rpc.py -tu autotvm -t cuda -th llvm -dt float32 -n mobilenet_v2 -l autotvm_ops -bs 1 -hw jetson-1