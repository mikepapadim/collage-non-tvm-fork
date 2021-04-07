import tvm.relay.testing as testing
from tvm import relay

from .op_config import MeasuredConfigs
from .backend_op import get_optimal_backendop, BackendOpLib
from .target import Target
from .op_type import OpType

# Run on the GPU 1 (RTX 2070)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from ..workloads.relay_workloads import get_network_from_relay

def log_backend_op_perf(b_op_lib, expr, target):
    assert type(target) != list
    
    for pattern in b_op_lib.get_all_patterns():
        if pattern.get_pattern().match(expr):
            print("PATTERN:\n", pattern.get_pattern())
            res = get_optimal_backendop(b_op_lib, expr, pattern, [target])
            if res == None:
                print("No satisfying backend operators")
            else:
                op, cost = res
                print("best backendop: %s, cost: %.5f ms" % (op, cost))

# traverse all subgraphs of a computation graph and evaluate all matchings between backend operators and subgraphs
def log_network_backend_ops_perf_on_target(b_op_lib, target, network_name, batch_size = 1):
    
    mod, _, _, _ = get_network_from_relay(network_name, batch_size)

#     # Resnet-8
#     image_shape = (3, 28, 28)
#     mod, params = testing.resnet.get_workload(num_layers=8, batch_size=batch_size, image_shape=image_shape)

#     # Resnet-18
#     image_shape = (3, 224, 224)
#     mod, params = testing.resnet.get_workload(num_layers=18, batch_size=batch_size, image_shape=image_shape)

    relay.analysis.post_order_visit(mod['main'], lambda expr: log_backend_op_perf(b_op_lib, expr, target))


def add_all_backend_ops_to_lib(b_op_lib, target):
    t_name = target.name()

    for op_type in OpType:
        # Skip diamond pattern for now
        if op_type==OpType.DIAMOND:
            continue

        op_name, op_depth = op_type.name(), op_type.depth()
        b_op_lib.add_backendop(f"{t_name}_{op_name}", target, op_type, op_depth)

def add_all_backend_ops_to_lib_except_fused(b_op_lib, target):
    t_name = target.name()
    op_to_skip = [OpType.DIAMOND, OpType.ADD] #OpType.CONV2D_BN, OpType.CONV2D_BN_RELU, 
                  #OpType.BN_RELU, OpType.CONV2D_BIAS_ADD_RELU
    
    for op_type in OpType:
        # Skip diamond pattern for now
        if op_type in op_to_skip:
            continue

        op_name, op_depth = op_type.name(), op_type.depth()
        b_op_lib.add_backendop(f"{t_name}_{op_name}", target, op_type, op_depth)

        
# TODO: add in some constraints

measured_configs = MeasuredConfigs()
measured_configs.load_from_log()

backendop_lib = BackendOpLib(measured_configs)

# CUDNN
# TODO: discuss with Soo.
backendop_lib.add_backendop("cudnn_conv2d", Target.CUDNN, OpType.CONV2D, 1)
backendop_lib.add_backendop("cudnn_softmax", Target.CUDNN, OpType.SOFTMAX, 1)
backendop_lib.add_backendop("cudnn_biasadd", Target.CUDNN, OpType.BIAS_ADD, 1)
backendop_lib.add_backendop("cudnn_relu", Target.CUDNN, OpType.RELU, 1)
backendop_lib.add_backendop("cudnn_bn", Target.CUDNN, OpType.BN, 1)

# measure_cost doesn't work, we need to fix this later.
# backendop_lib.add_backendop("cudnn_maxpool2d", Target.CUDNN, OpType.MAX_POOL2D, 1)

# conv_bias_add_relu --> ResNet doesn't have this pattern, so it wouldn't be measured
backendop_lib.add_backendop("cudnn_conv2d+biasadd+relu", Target.CUDNN, OpType.CONV2D_BIAS_ADD_RELU, 3)
# backendop_lib.add_backendop("cudnn_add", Target.CUDNN, OpType.ADD, 1)


# Non-existing patterns
#backendop_lib.add_backendop("cudnn_dense", Target.CUDNN, OpType.DENSE, 1) #TODO: matmul?
#backendop_lib.add_backendop("cudnn_batchflatten", Target.CUDNN, OpType.BATCH_FLATTEN, 1)
#backendop_lib.add_backendop("cudnn_globalavgpool2d", Target.CUDNN, OpType.GLOBAL_AVG_POOL2D, 1)
#backendop_lib.add_backendop("cudnn_conv2d+bn", Target.CUDNN, OpType.CONV2D_BN, 2)
#backendop_lib.add_backendop("cudnn_bn+relu", Target.CUDNN, OpType.BN_RELU, 2)
#backendop_lib.add_backendop("cudnn_conv2d+bn+relu", Target.CUDNN, OpType.CONV2D_BN_RELU, 3)
#backendop_lib.add_backendop("cudnn_diamond", Target.CUDNN, OpType.DIAMOND, 6)


# TENSORRT
add_all_backend_ops_to_lib(backendop_lib, Target.TENSORRT)

# CUBLAS
# TODO: Add patterns. matmul, batch matmul
backendop_lib.add_backendop("cublas_dense", Target.CUBLAS, OpType.DENSE, 1)

# TVM_GPU
add_all_backend_ops_to_lib(backendop_lib, Target.TVM_GPU)
# add_all_backend_ops_to_lib_except_fused(backendop_lib, Target.TVM_GPU)

# TVM_GPU_NO_TUNING
add_all_backend_ops_to_lib(backendop_lib, Target.TVM_GPU_NO_TUNING)
# add_all_backend_ops_to_lib_except_fused(backendop_lib, Target.TVM_GPU_NO_TUNING)


# TVM_CPU; Exclude it for GPU testing
# Fix: Extend this to automatically select backend library based on HW info
# add_all_backend_ops_to_lib(backendop_lib, Target.TVM_CPU)

# OUTDATED: We left it just for reference.
# backendop_lib.add_backendop("tvmgpu_add", Target.TVM_GPU, OpType.ADD, 1)
# backendop_lib.add_backendop("tvmgpu_conv2d", Target.TVM_GPU, OpType.CONV2D, 1)
# backendop_lib.add_backendop("tvmgpu_bn", Target.TVM_GPU, OpType.BN, 1)
# backendop_lib.add_backendop("tvmgpu_relu", Target.TVM_GPU, OpType.RELU, 1)
# backendop_lib.add_backendop("tvmgpu_softmax", Target.TVM_GPU, OpType.SOFTMAX, 1)
# backendop_lib.add_backendop("tvmgpu_biasadd", Target.TVM_GPU, OpType.BIAS_ADD, 1)
# backendop_lib.add_backendop("tvmgpu_dense", Target.TVM_GPU, OpType.DENSE, 1)
# backendop_lib.add_backendop("tvmgpu_batchflatten", Target.TVM_GPU, OpType.BATCH_FLATTEN, 1)
# backendop_lib.add_backendop("tvmgpu_globalavgpool2d", Target.TVM_GPU, OpType.GLOBAL_AVG_POOL2D, 1)
# backendop_lib.add_backendop("tvmgpu_maxpool2d", Target.TVM_GPU, OpType.MAX_POOL2D, 1)
# backendop_lib.add_backendop("tvmgpu_conv2d+bn", Target.TVM_GPU, OpType.CONV2D_BN, 2)
# backendop_lib.add_backendop("tvmgpu_bn+relu", Target.TVM_GPU, OpType.BN_RELU, 2)
# backendop_lib.add_backendop("tvmgpu_conv2d+bn+relu", Target.TVM_GPU, OpType.CONV2D_BN_RELU, 3)
# backendop_lib.add_backendop("tvmgpu_diamond", Target.TVM_GPU, OpType.DIAMOND, 6)


# Note: Execute one target at a time. Otherwise, you will get random error.
# Warning: Larger batch size leads to the error: 
# TVMError: CUDALaunch Error: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
# log_network_backend_ops_perf_on_target(backendop_lib, Target.TVM_GPU, 'resnet-50', batch_size = 16)
# log_network_backend_ops_perf_on_target(backendop_lib, Target.TENSORRT, 'resnet-50', batch_size = 16)
# log_network_backend_ops_perf_on_target(backendop_lib, Target.CUDNN, 'resnet-50', batch_size = 16)
# log_network_backend_ops_perf_on_target(backendop_lib, Target.CUBLAS, 'resnet-50', batch_size = 16)
# log_network_backend_ops_perf_on_target(backendop_lib, Target.TVM_GPU_NO_TUNING, 'resnet-50', batch_size = 16)
# log_network_backend_ops_perf_on_target(backendop_lib, Target.TVM_CPU, 'resnet-50', batch_size = 1)

# log_network_backend_ops_perf_on_target(backendop_lib, Target.TVM_GPU, 'bert')
# log_network_backend_ops_perf_on_target(backendop_lib, Target.TENSORRT, 'bert')
# log_network_backend_ops_perf_on_target(backendop_lib, Target.CUDNN, 'bert')
# log_network_backend_ops_perf_on_target(backendop_lib, Target.CUBLAS, 'bert')
# log_network_backend_ops_perf_on_target(backendop_lib, Target.TVM_CPU, 'bert')

measured_configs.save_to_log()
