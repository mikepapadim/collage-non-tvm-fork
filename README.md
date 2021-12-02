# Collage
System for automated integration of deep learning backends. Our implementation uses TVM as its code generator. 

# Installation
1. Go to `tvm/` and install tvm. Make sure backend libaries of interest are built together. [TVM installation guide](https://tvm.apache.org/docs/install/index.html)
2. Add following environment variables
```
export COLLAGE_HOME=/path/to/collage/repo
export COLLAGE_TVM_HOME=${COLLAGE_HOME}/tvm
export PYTHONPATH=${COLLAGE_TVM_HOME}/python:${COLLAGE_HOME}/python:${PYTHONPATH}
```

# Demo
1. `cd demo/`
2. `python3 demo.py`

# Known issues
* As Collage uses TVM as its code generator, it cannot support backends that TVM is unable to build. Tested backends are
  * TVM w/o tuning
  * TVM w/ AutoTVM
  * cuDNN
  * cuBLAS
  * TensorRT
  * MKL
  * DNNL
* Since Collage is essentially a profile-guided search, variance in the measurement may affect the final backend placement. For the best result, multiple runs are highly recommended. 


# Cite
```
@article{jeon2021collage,
  title={Collage: Automated Integration of Deep Learning Backends},
  author={Jeon, Byungsoo and Park, Sunghyun and Liao, Peiyuan and Xu, Sheng and Chen, Tianqi and Jia, Zhihao},
  journal={arXiv preprint arXiv:2111.00655},
  year={2021}
}
```
