#!/bin/bash

# 1) Make sure you have right PROJ_PATH
# 2) Don't forget to type following command before you autotune ops
# - conda activate tvm_fleet

# Currently available networks:
# resnet50, resnext50, nasnetamobile
#PROJ_PATH=~/backend-aware-graph-opt
#cd $PROJ_PATH

# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# --> RTX2070
#python3 testing/autotune_relay.py -tu autotvm -t llvm -th llvm -dt float32 -n nasneta -l autotvm_ops -bs 1 -hw xeon
#python3 testing/autotune_relay.py -tu autotvm -t llvm -th llvm -dt float32 -n resnet50_3d -l autotvm_ops -bs 1 -hw xeon
#python3 testing/autotune_relay.py -tu autotvm -t llvm -th llvm -dt float32 -n bert_full -l autotvm_ops -bs 1 -hw xeon
#python3 testing/autotune_relay.py -tu autotvm -t llvm -th llvm -dt float32 -n yolov3 -l autotvm_ops -bs 1 -hw xeon


# Error
#python3 testing/autotune_relay.py -tu autotvm -t llvm -th llvm -dt float32 -n dcgan -l autotvm_ops -bs 1 -hw xeon

# Error
#python3 testing/autotune_relay.py -tu autotvm -t llvm -th llvm -dt float32 -n bert -l autotvm_ops -bs 1 -hw xeon

# Error
#python3 testing/autotune_relay.py -tu autotvm -t llvm -th llvm -dt float32 -n resnext50_32x4d -l autotvm_ops -bs 1 -hw xeon

# We don't need it anymore
#python3 testing/autotune_relay.py -tu autotvm -t llvm -th llvm -dt float32 -n mobilenet_v2 -l autotvm_ops -bs 1 -hw xeon
#python3 testing/autotune_relay.py -tu autotvm -t llvm -th llvm -dt float32 -n nasrnn -l autotvm_ops -bs 1 -hw xeon
#python3 testing/autotune_relay.py -tu autotvm -t llvm -th llvm -dt float32 -n resnet50 -l autotvm_ops -bs 1 -hw xeon
