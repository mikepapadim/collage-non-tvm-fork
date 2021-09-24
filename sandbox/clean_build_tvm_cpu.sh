#!/bin/bash
TVM_PATH=~/tvm

# Re-build TVM
rm -rf $TVM_PATH/build
mkdir $TVM_PATH/build
cd $TVM_PATH
cp cmake/config.cmake.cpu build/config.cmake
cd build
cmake ..
make clean && make -j64
