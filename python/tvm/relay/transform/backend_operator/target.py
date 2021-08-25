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
# from tvm.contrib import graph_executor
from ..utility.debug_helper import printe

# only collect results whose standard deviation is below this
MAX_STANDARD_DEVIATION = 5E-04
# MAX_STANDARD_DEVIATION = 5E-03

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

cur_dir_path = Path(__file__).parent.absolute()
LOG_PATH = f"{cur_dir_path}/../logs"
BEST_MATCH_LOG = f"{LOG_PATH}/best_match"
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

def measure(ftimer, is_net, *args):
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
        printe(f"Mean, std of perf : {mean_perf}, {std_perf}")

        # If mean_perf is more than 1 ms, then we should reduce threshold not to take too long,
        # e.g., BERT or Conv3D ops
        # Otherwise, we keep MAX_STANDARD_DEVIATION no matter how small the mean_perf is.
        # MAX_STANDARD_DEVIATION much of variance shouldn't matter anyway for end-to-end perf.
        threshold = max(MAX_STANDARD_DEVIATION, MAX_STANDARD_DEVIATION*mean_perf)
        # if is_net or std_perf <= MAX_STANDARD_DEVIATION:
        if std_perf <= threshold:
            break

    return mean_perf, std_perf

# (id, parameter, name)
class Target(Enum):
    # NVIDIA GPU
    CUDNN = (1, "cuda -libs=cudnn", "cudnn")
    TENSORRT = (2, "tensorrt", "tensorrt")
    CUBLAS = (3, "cuda -libs=cublas", "cublas")
    TVM_GPU_NO_TUNING = (4, "cuda", "tvmgpu-no-tuning")
    TVM_GPU_AUTOTVM = (5, "cuda", "tvmgpu-autotvm")
    TVM_GPU_AUTOSCH = (6, "cuda", "tvmgpu-autosch")

    # Intel CPU
    TVM_CPU_AUTOTVM = (7, "llvm", "tvmcpu-autotvm")
#     ONEDNN = (8, "onednn", "onednn") # not implemented
#     TENSORFLOWXLA = (9, "tensorflowxla") # not implemented

    def __str__(self):
        return self.value[1]

    def name(self):
        return self.value[2]

target_id_to_target = {}
for target in Target:
    target_id_to_target[target.value[0]] = target

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
        target_str = target.__str__()
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

        return measure(ftimer, is_net=False)

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
            target_str = target.__str__()
            with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
                lib = relay.build_module.build(net, target=target_str, params=params)

            dev = tvm.device(str(target), 0)
            module = runtime.GraphModule(lib["default"](dev))

            # Setup execution
            setup_mod_inputs(module)
            # data_shape = get_data_shape(expr)
            # data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
            # module.set_input("data", data)
            ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
        return measure(ftimer, is_net=False)

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

        net, params = testing.create_workload(expr_func)

        # Build the subgraph
        target_str = target.__str__()

        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build_module.build(net, target=target_str, params=params)

        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        setup_mod_inputs(module)
        # data_shape = get_data_shape(expr)
        # data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        # module.set_input("data", data)
        ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

        return measure(ftimer, is_net=False)

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

        target = "cuda"
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get(), config={'relay.ext.tensorrt.options': config}):
            lib = relay.build(mod, target=target, params=params)

        lib.export_library('compiled.so')

        dev = tvm.gpu(0)
        loaded_lib = tvm.runtime.load_module('compiled.so')
        module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

        setup_mod_inputs(module)
        # input_shape = get_data_shape(expr)
        # input_data = np.random.uniform(0, 1, input_shape).astype("float32")
        # module.set_input("data", input_data)
        ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
        measure_info = measure(ftimer, is_net=False)
        return measure_info

"""
Note that CUBLAS and CUDNN goes to TVMSubGraphCostFunc_OpMeasurement.
This cost function enable op measurement through our backend pipeline where we execute ops based on our backend label.
Technically speaking, we can use that function for TensorRT (not AutoTVM due to AutoTVM schedule log) as well.
For now, we don't use it for TensorRT just to be safe. There shouldn't be an issue though.
"""


target_to_cost_func = {
    #GPU
    Target.TVM_GPU_AUTOTVM: TVMSubGraphCostFunc_AutoTVM(),
    Target.TVM_GPU_AUTOSCH: TVMSubGraphCostFunc_AutoSch(),
    #Target.CUDNN: CuDNNCostFunc(),
    Target.TENSORRT: TensorRTCostFunc(),
    Target.CUDNN: TVMSubGraphCostFunc_OpMeasurement(),
    Target.CUBLAS: TVMSubGraphCostFunc_OpMeasurement(),
    Target.TVM_GPU_NO_TUNING: TVMSubGraphCostFunc_OpMeasurement(),

    # CPU
    # Target.TVM_CPU: TVMSubGraphCostFunc_NoTuning(),
}

def get_target_cost_func(target):
    return target_to_cost_func[target].measure_cost


