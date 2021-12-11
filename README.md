# Collage
System for automated integration of deep learning backends. 

# Installation
Since our implementation uses TVM as the main code generator, install tvm under `tvm/`. [TVM installation guide](https://tvm.apache.org/docs/install/index.html)
1. Install dependencies
```
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```
```
pip3 install --user numpy decorator attrs tornado psutil xgboost cloudpickle pytest
```

2. Create build directory and go to build directory
```
mkdir tvm/build && cd tvm/build
```
3. Prepare `cmake` configuration file. Make sure backend libaries of interest are built together. We provide cmake config that we used for our GPU/CPU experiments (`config.cmake.gpu`, `config.cmake.cpu`) in `tvm/cmake/`. Users may copy it to their build directory and rename it to `config.cmake`
 ```
 cp ../cmake/config.cmake.gpu config.cmake
 ```
4. Run `cmake` and `make`
```
cmake .. && make -j$(nproc)
```
5. Declare following environment variables
```
export COLLAGE_HOME=/path/to/collage/repo
export COLLAGE_TVM_HOME=${COLLAGE_HOME}/tvm
export PYTHONPATH=${COLLAGE_TVM_HOME}/python:${COLLAGE_HOME}/python:${PYTHONPATH}
```

# Demo
Install the following dependencies for deep learning models used for demo.
```
pip3 install --user torch torchvision tqdm
```

We provide two demos (`demo_performance.py`, `demo_customization.py`) under `demo/`. 
* `demo_performance.py` shows how collage optimizes given workloads with popular backends that Collage provides by default.
* `demo_customization.py` shows how users can register new backend with their custom codegen, pattern, pattern rule.

For the best result, it is highly recommend to create the tuning log by using `autotune_tvm_ops.py` before running those demos.


# Note
* As Collage uses TVM as its code generator, it cannot support backends that TVM is unable to build. Tested backends are
  * TVM w/o tuning
  * TVM w/ AutoTVM
  * cuDNN
  * cuBLAS
  * TensorRT
  * MKL
  * DNNL
* Since Collage is essentially a profile-guided search, variance in the measurement may affect the final backend placement. For the best result, multiple runs are highly recommended. 
* Due to some issues in the implementation, current evolutionary search only supports network implemented in `get_network_from_torch()`. If an user want to try new network, the network must be implemented within this function.


# Cite
```
@article{jeon2021collage,
  title={Collage: Automated Integration of Deep Learning Backends},
  author={Jeon, Byungsoo and Park, Sunghyun and Liao, Peiyuan and Xu, Sheng and Chen, Tianqi and Jia, Zhihao},
  journal={arXiv preprint arXiv:2111.00655},
  year={2021}
}
```
