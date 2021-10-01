#!/bin/bash
TVM_PATH=~/tvm

# Re-build TVM
cd $TVM_PATH/build
cp ../cmake/config.cmake.cpu config.cmake
cmake ..
make -j64
. /opt/intel/oneapi/setvars.sh
