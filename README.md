# Collage
System for automated integration of deep learning backends. Our implementation uses TVM as its code generator. 

# Installation
1. Go to `tvm/` and install tvm. Make sure backend libaries of interest are built together. [TVM installation guide](https://tvm.apache.org/docs/install/index.html)
   We provide cmake config that we used for our GPU/CPU experiments (`config.cmake.gpu`, `config.cmake.cpu`) in `tvm/cmake/`.
   Users may copy it to their build directory and rename it to `config.cmake` before running `cmake` command. 
2. Declare following environment variables
```
export COLLAGE_HOME=/path/to/collage/repo
export COLLAGE_TVM_HOME=${COLLAGE_HOME}/tvm
export PYTHONPATH=${COLLAGE_TVM_HOME}/python:${COLLAGE_HOME}/python:${PYTHONPATH}
```

# Demo
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
