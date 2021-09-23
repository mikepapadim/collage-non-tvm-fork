#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 --> RTX3070
# CUDA_VISIBLE_DEVICES=1 --> RTX2070

#python3 testing/export_torch_to_onnx.py -n bert_full -bs 1

# Don't use it. It has the error of "RuntimeError: Tensors must have same number of dimensions: got 2 and 1"
#python3 testing/export_torch_to_onnx.py -n yolov3 -bs 1

