from enum import Enum
from tvm import relay
import tvm.relay.testing as testing
import tvm
import numpy as np
# import tvm.contrib.graph_executor as runtime
from tvm import autotvm, auto_scheduler

import os
from pathlib import Path

from .utils import *

from tvm.contrib import graph_executor as runtime
import logging
# from tvm.contrib import graph_executor
from ..utility.debug_helper import printe

# only collect results whose standard deviation is below this
MAX_STD_MEASURE_RTX = 5E-04
MAX_STD_MEASURE_XEON = 0.005 # 0.0025
# MAX_STD_MEASURE_GPU = 5E-03

# This is for operator measurement
# NUM_REPEATS = 1 # Debug / This lead to the best end-to-end perf of DP on ResNet-50
NUM_REPEATS = 3 # Finalized one by Sung
# NUM_MEASUREMENTS_PER_REPEAT = 1 # Debug
# NUM_MEASUREMENTS_PER_REPEAT = 10 # Finalized one by Sung
NUM_MEASUREMENTS_PER_REPEAT = 20
# NUM_MEASUREMENTS_PER_REPEAT = 100

# This is for network measurement
NUM_REPEATS_E2E = 3
NUM_MEASUREMENTS_PER_REPEAT_E2E = 20
OPT_LEVEL = OptLevel(3)
EXTERNAL_COMPILERS = ['tensorrt']
# XEON_BUILD_TARGET = 'llvm'
XEON_BUILD_TARGET = 'llvm -mcpu=skylake-avx512'
NVIDIA_GPUS = ['rtx2070', 'rtx3070', 'jetson', 'v100']
INTEL_CPUS = ['xeon']

cur_dir_path = Path(__file__).parent.absolute()
LOG_PATH = f"{cur_dir_path}/../logs"
EVAL_RESULT_LOG_PATH = f"{LOG_PATH}/eval_results"
BEST_MATCH_LOG = f"{EVAL_RESULT_LOG_PATH}/best_match"
USER_DEFINED_MATCH_LOG = f"{LOG_PATH}/user_defined_match.log"
HW_FUNC_ATTR = "TargetHW"
BATCH_SIZE_ATTR = "BatchSize"
BACKEND_OP_ATTR = "BackendOP"
NETWORK_FUNC_ATTR = "Network"
SINGLE_BACKEND_ATTR = "SingleBackend"

# AUTOTVM_LOG = f"{LOG_PATH}/autotvm_ops.json"
# Temporary autoscheduler log file
# FIXME(@Soo): Accumulate autoscheduler logs to the same file
# AUTOSCH_LOG = "/home/byungsoj/backend-aware-graph-opt/package/autotune/tmp/autosch_ops.json.resnet50.tmp"
AUTOSCH_LOG = f"{LOG_PATH}/autosch_ops.json"

"""
measure
- 1) network
    > Skip while loop
- 2) operator
    > Keep as is
"""

def get_autotvm_log_path(hw_name):
    return f"{LOG_PATH}/autotvm_ops_{hw_name}.json"

def get_build_target(hw_name):
    if hw_name in NVIDIA_GPUS:
        build_target = 'cuda'
    elif hw_name in INTEL_CPUS:
        build_target = XEON_BUILD_TARGET
    else:
        raise Exception(f"{hw_name} is unexpected hw, we need to set build target for this hw.")

    return build_target

def get_backends(hw_name):
    if hw_name in NVIDIA_GPUS:
        backends = [Target.AUTOTVM, Target.CUDNN, Target.TENSORRT, Target.CUBLAS]
    elif hw_name in INTEL_CPUS:
        backends = [Target.AUTOTVM, Target.MKL, Target.MKLDNN, Target.DNNL]
    else:
        raise Exception(f"{hw_name} is unexpected hw, we need to set default backends for this hw.")

    return backends

# For example, it is TensorRT for NVIDIA GPUs, and OpenVINO (For now, OneDNN) for Intel CPU.
def get_graph_level_opt_backend_name(hw_name):
    if hw_name in NVIDIA_GPUS:
        backend_name = Target.TENSORRT.name()
    elif hw_name in INTEL_CPUS:
        backend_name = Target.DNNL.name()
    else:
        raise Exception(f"{hw_name} is unexpected hw, we need to set default backends for this hw.")

    return backend_name

def get_max_std_for_measurement(hw_name, mean_perf):
    if hw_name in NVIDIA_GPUS:
        max_std = max(MAX_STD_MEASURE_RTX, MAX_STD_MEASURE_RTX*mean_perf)
    elif hw_name in INTEL_CPUS:
        max_std = max(MAX_STD_MEASURE_XEON, MAX_STD_MEASURE_XEON*mean_perf)
    else:
        raise Exception(f"{hw_name} is unexpected hw, we need to set default backends for this hw.")

    return max_std

def measure(ftimer, is_net, hw_name, *args):
    # Dummy run to check whether it runs correctly e.g., segfault due to large workspace
    import sys

    #try:
    #    ftimer(*args)
    #except:
    #    return float('inf'), 0

    # Warm-up Phase: Run without measurement
    # TimeEvaluator itself come with the warmup,
    # so we don't need this part technically.
    for i in range(3):
         ftimer(*args)

    mean_perf, std_perf = None, None
    # Measure performance. Continue until we get results within the max standard deviation
    while True:
        perfs = np.array(ftimer(*args).results) * 1000  # convert to millisecond
        mean_perf, std_perf = np.mean(perfs), np.std(perfs)
        logging.info(f"Mean, std of perf : {mean_perf}, {std_perf}")

        # If mean_perf is more than 1 ms, then we should reduce threshold not to take too long,
        # e.g., BERT or Conv3D ops
        # Otherwise, we keep MAX_STD_MEASURE_RTX no matter how small the mean_perf is.
        # MAX_STD_MEASURE_RTX much of variance shouldn't matter anyway for end-to-end perf.
        threshold = get_max_std_for_measurement(hw_name, mean_perf)
        # if is_net or std_perf <= MAX_STD_MEASURE_RTX:
        if std_perf <= threshold:
            break

    return mean_perf, std_perf

# (id, parameter, name)
class Target(Enum):
    # NVIDIA GPU
    CUDNN = (1, "cudnn")
    TENSORRT = (2, "tensorrt")
    CUBLAS = (3, "cublas")
    TVM_DEFAULT = (4, "tvm-default")
    AUTOTVM = (5, "autotvm")
    AUTOSCH = (6, "autosch")

    # Intel CPU
    DNNL = (7, "dnnl") # not implemented
    MKL = (8, "mkl")  # not implemented
    MKLDNN = (9, "mkldnn")  # not implemented
#     TENSORFLOWXLA = (10, "tensorflowxla") # not implemented

    def id(self):
        return self.value[0]

    def name(self):
        return self.value[1]

    def __str__(self):
        return self.name()

target_id_to_target = {}
for target in Target:
    target_id_to_target[target.id()] = target

class TargetCostFunc:
    def __init__(self):
        pass

    # Placeholder. It shouldn't be called.
    def measure_cost(self):
        assert False


class TVMSubGraphCostFunc_AutoSch(TargetCostFunc):
    def __init__(self):
        super().__init__()

    # measure the cost of running an expression on a target, in milliseconds.
    # We assume that the target has a backend operator satisfying the configuration of the expr
    @staticmethod
    def measure_cost(name, expr, target, hw_name):
        # Create workload
        inputs = relay.analysis.free_vars(expr)
        expr_func = relay.Function(inputs, expr)
        net, params = testing.create_workload(expr_func)

        assert (os.path.exists(AUTOSCH_LOG))

        # AutoScheduler codes
        # FIXME(@Soo): We should redesign Target class to deal with new TVM build interface
        target_str = get_build_target(hw_name)
        with auto_scheduler.ApplyHistoryBest(AUTOSCH_LOG):
            with tvm.transform.PassContext(opt_level=OPT_LEVEL.get(), config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(net, target_str, params=params)

        dev = tvm.device(target_str, 0)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        setup_mod_inputs(module)
        # data_shape = get_data_shape(expr)
        # data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        # module.set_input("data", data)
        ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

        return measure(ftimer, False, hw_name)

class TVMSubGraphCostFunc_AutoTVM(TargetCostFunc):
    def __init__(self):
        super().__init__()

    # measure the cost of running an expression on a target, in milliseconds.
    # We assume that the target has a backend operator satisfying the configuration of the expr
    @staticmethod
    def measure_cost(name, expr, target, hw_name):
        # Create workload
        inputs = relay.analysis.free_vars(expr)
        expr_func = relay.Function(inputs, expr)
        net, params = testing.create_workload(expr_func)

        assert(os.path.exists(get_autotvm_log_path(hw_name)))

        # print(f"Measure autotvm log: {get_autotvm_log_path(hw_name)}")
        # AutoTVM codes
        # Compile kernels with history best records
        with autotvm.apply_history_best(get_autotvm_log_path(hw_name)):
            target_str = get_build_target(hw_name)
            with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
                lib = relay.build_module.build(net, target=target_str, params=params)

            dev = tvm.device(target_str, 0)
            module = runtime.GraphModule(lib["default"](dev))

            # Setup execution
            setup_mod_inputs(module)
            # data_shape = get_data_shape(expr)
            # data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
            # module.set_input("data", data)
            ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

        return measure(ftimer, False, hw_name)

class TVMSubGraphCostFunc_OpMeasurement(TargetCostFunc):
    def __init__(self):
        super().__init__()

    # measure the cost of running an expression on a target, in milliseconds.
    # We assume that the target has a backend operator satisfying the configuration of the expr
    @staticmethod
    def measure_cost(name, expr, target, hw_name):
        # Create workload
        inputs = relay.analysis.free_vars(expr)
        expr_func = relay.Function(inputs, expr)

        # Add attributes to execute our backend pipeline based on the operator name we assign to expr
        from ..optimizer.custom_fusion_pass import CustomFusionPass
        expr_func = expr_func.with_attr("CustomFusionPass", CustomFusionPass.OP_MEASUREMENT)
        default_op_group_id = 0
        annotation = create_backend_op_annotation(default_op_group_id, name)
        expr_func = expr_func.with_attr(BACKEND_OP_ATTR, annotation)
        expr_func = expr_func.with_attr(HW_FUNC_ATTR, hw_name)

        net, params = testing.create_workload(expr_func)

        # Build the subgraph
        target_str = get_build_target(hw_name)

        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build_module.build(net, target=target_str, params=params)

        dev = tvm.device(target_str, 0)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        setup_mod_inputs(module)
        # data_shape = get_data_shape(expr)
        # data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        # module.set_input("data", data)
        ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

        return measure(ftimer, False, hw_name)

# def get_conv_attr(expr):
#     assert (is_call_node(expr))
#     # note that only call node has "op" attribute corresponding to a single backend operator
#     op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span
#
#     # extract conv attributes
#     strides, padding, out_channels, dilation = \
#         list(attrs.strides), list(attrs.padding), int(attrs.channels), list(attrs.dilation)
#
#     #kernel_size = args[1].type_annotation.shape
#     kernel_size = list(map(lambda x: x.value, args[1].type_annotation.shape))
#     dtype = args[0].type_annotation.dtype
#
#     return strides, padding, out_channels, dilation, kernel_size, dtype, attrs.groups, attrs.data_layout, attrs.kernel_layout

class TensorRTCostFunc(TargetCostFunc):
    def __init__(self):
        super().__init__()

    @staticmethod
    def measure_cost(name, expr, target, hw_name):

        # Create workload
        inputs = relay.analysis.free_vars(expr)
        expr_func = relay.Function(inputs, expr)
        net, params = testing.create_workload(expr_func)

        from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
        mod, config = partition_for_tensorrt(net, params)

        # We confirm that TVM can't pass conv2d to TensorRT if it's winograd without wt
        # if name == "tensorrt_conv2d_winograd_without_weight_transform":
        #     print(name, mod["main"])
        #     sys.exit(0)

        target_str = get_build_target(hw_name)
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get(), config={'relay.ext.tensorrt.options': config}):
            lib = relay.build(mod, target=target_str, params=params)

        lib.export_library('compiled.so')

        dev = tvm.gpu(0)
        loaded_lib = tvm.runtime.load_module('compiled.so')
        module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

        setup_mod_inputs(module)
        # input_shape = get_data_shape(expr)
        # input_data = np.random.uniform(0, 1, input_shape).astype("float32")
        # module.set_input("data", input_data)
        ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

        measure_info = measure(ftimer, False, hw_name)
        return measure_info

"""
Note that CUBLAS and CUDNN goes to TVMSubGraphCostFunc_OpMeasurement.
This cost function enable op measurement through our backend pipeline where we execute ops based on our backend label.
Technically speaking, we can use that function for TensorRT (not AutoTVM due to AutoTVM schedule log) as well.
For now, we don't use it for TensorRT just to be safe. There shouldn't be an issue though.
"""


target_to_cost_func = {
    # GPU
    Target.AUTOTVM: TVMSubGraphCostFunc_AutoTVM(),
    Target.AUTOSCH: TVMSubGraphCostFunc_AutoSch(),
    #Target.CUDNN: CuDNNCostFunc(),
    Target.TENSORRT: TensorRTCostFunc(),
    Target.CUDNN: TVMSubGraphCostFunc_OpMeasurement(),
    Target.CUBLAS: TVMSubGraphCostFunc_OpMeasurement(),
    Target.TVM_DEFAULT: TVMSubGraphCostFunc_OpMeasurement(),

    # CPU
    Target.DNNL: TVMSubGraphCostFunc_OpMeasurement(),
    Target.MKL: TVMSubGraphCostFunc_OpMeasurement(),
    Target.MKLDNN: TVMSubGraphCostFunc_OpMeasurement(),
}

def get_target_cost_func(target):
    return target_to_cost_func[target].measure_cost


