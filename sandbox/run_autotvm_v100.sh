#!/bin/bash

# 1) Make sure you have right PROJ_PATH
# 2) Don't forget to type following command before you autotune ops
# - conda activate tvm_fleet

# Currently available networks:
# resnet50, resnext50, nasnetamobile
#PROJ_PATH=~/backend-aware-graph-opt
#cd $PROJ_PATH

#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnext50_32x4d -l autotvm_ops -bs 1 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n nasneta -l autotvm_ops -bs 1 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n bert -l autotvm_ops -bs 1 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnet50_3d -l autotvm_ops -bs 1 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n dcgan -l autotvm_ops -bs 1 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n nasrnn -l autotvm_ops -bs 1 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnet50 -l autotvm_ops -bs 1 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n mobilenet_v2 -l autotvm_ops -bs 1 -hw v100


# Increasing Batch Size for ResNext
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnext50_32x4d -l autotvm_ops -bs 16 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnext50_32x4d -l autotvm_ops -bs 8 -hw v100
CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnext50_32x4d -l autotvm_ops -bs 4 -hw v100

# Batch size of 16
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnet50_3d -l autotvm_ops -bs 16 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n nasneta -l autotvm_ops -bs 16 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n dcgan -l autotvm_ops -bs 16 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnext50_32x4d -l autotvm_ops -bs 16 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n bert -l autotvm_ops -bs 16 -hw v100


# Batch size of 8
# Note: Only mobilenet has the measurement issue of single operator (e.g., relu); weirdly, not on the fused operator thought.
# I think this is the issue of fallback schedule of TVM when we didn't autotune ops. Let's check.
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnext50_32x4d -l autotvm_ops -bs 8 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n dcgan -l autotvm_ops -bs 8 -hw v100
CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n nasneta -l autotvm_ops -bs 8 -hw v100
CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnet50_3d -l autotvm_ops -bs 8 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n bert -l autotvm_ops -bs 8 -hw v100

#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n mobilenet_v2 -l autotvm_ops -bs 8 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n resnet50 -l autotvm_ops -bs 8 -hw v100
#CUDA_VISIBLE_DEVICES=0 python3 testing/autotune_relay.py -tu autotvm -t cuda -th llvm -dt float32 -n nasrnn -l autotvm_ops -bs 8 -hw v100
