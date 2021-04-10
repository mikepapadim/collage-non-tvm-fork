import tvm.relay.testing as testing
from tvm import relay

from .op_config import MeasuredConfigs
from .backend_op import get_optimal_backendop
from .backend_op_lib import BackendOpLib
from .target import Target
from .op_type import OpType

# It won't work because comp graph from onnx doesn't have type information
# In other words, Relay type inference is needed to get type info

if __name__ == "__main__":
    networks = ['resnet50']
    # targets = [Target.TVM_GPU, Target.TENSORRT, Target.CUDNN, Target.CUBLAS, Target.TVM_GPU_NO_TUNING, Target.TVM_CPU]
    targets = [Target.TVM_GPU_NO_TUNING]
    batch_size = 1

    backendop_lib = BackendOpLib.get()
    backendop_lib.measure_backend_ops(networks, targets, batch_size)
    backendop_lib.save_to_log()

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


