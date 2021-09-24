#!/bin/bash

# 1) Make sure you have right PROJ_PATH

# Currently available networks:
#PROJ_PATH=~/backend-aware-graph-opt
#cd $PROJ_PATH

# "resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta"
#python3 testing/measure_end_to_end.py -n resnet50 -hw xeon
#python3 testing/measure_end_to_end.py -n mobilenet_v2 -hw xeon

# Pass
#python3 testing/measure_end_to_end.py -n resnext50_32x4d -hw xeon
#python3 testing/measure_end_to_end.py -n dcgan -hw xeon
#python3 testing/measure_end_to_end.py -n bert -hw xeon
#python3 testing/measure_end_to_end.py -n resnet50_3d -hw xeon

# Fail
python3 testing/measure_end_to_end.py -n nasneta -hw xeon

#python3 testing/measure_end_to_end.py -n nasrnn -hw xeon
