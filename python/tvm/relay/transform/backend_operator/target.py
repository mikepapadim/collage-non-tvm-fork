from enum import Enum
from tvm import relay
import tvm.relay.testing as testing
import tvm
import numpy as np
# from tvm.contrib import graph_runtime as runtime
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
# NUM_REPEATS = 1 # Debug
NUM_REPEATS = 3 # Finalized one by Sung
# NUM_MEASUREMENTS_PER_REPEAT = 1 # Debug
# NUM_MEASUREMENTS_PER_REPEAT = 10 # Finalized one by Sung
NUM_MEASUREMENTS_PER_REPEAT = 20
# NUM_MEASUREMENTS_PER_REPEAT = 100
OPT_LEVEL = OptLevel(3)
EXTERNAL_COMPILERS = ['tensorrt']

cur_dir_path = Path(__file__).parent.absolute()
LOG_PATH = f"{cur_dir_path}/../logs"
BEST_MATCH_LOG = f"{LOG_PATH}/best_match"
USER_DEFINED_MATCH_LOG = f"{LOG_PATH}/user_defined_match.log"
AUTOTVM_LOG = f"{LOG_PATH}/autotvm_ops.json"
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
        std_perf = np.std(perfs)
        printe(f"Mean, std of perf : {np.mean(perfs)}, {std_perf}")

        if std_perf <= MAX_STANDARD_DEVIATION:
        #if is_net or std_perf <= MAX_STANDARD_DEVIATION:
            mean_perf = np.mean(perfs)
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
    TVM_CPU= (7, "llvm", "tvmcpu")
#     ONEDNN = (8, "onednn", "onednn") # not implemented
#     TENSORFLOWXLA = (9, "tensorflowxla") # not implemented

    def __str__(self):
        return self.value[1]

    def name(self):
        return self.value[2]


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
    def measure_cost(name, expr, target):
        # Create workload
        inputs = relay.analysis.free_vars(expr)
        expr_func = relay.Function(inputs, expr)
        net, params = testing.create_workload(expr_func)

        assert (os.path.exists(AUTOSCH_LOG))

        # AutoScheduler codes
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
    def measure_cost(name, expr, target):
        # Create workload
        inputs = relay.analysis.free_vars(expr)
        expr_func = relay.Function(inputs, expr)
        net, params = testing.create_workload(expr_func)

        assert(os.path.exists(AUTOTVM_LOG))

        # AutoTVM codes
        # Compile kernels with history best records
        with autotvm.apply_history_best(AUTOTVM_LOG):
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

class TVMSubGraphCostFunc_NoTuning(TargetCostFunc):
    def __init__(self):
        super().__init__()

    # measure the cost of running an expression on a target, in milliseconds.
    # We assume that the target has a backend operator satisfying the configuration of the expr
    @staticmethod
    def measure_cost(name, expr, target):
        # Create workload
        inputs = relay.analysis.free_vars(expr)
        expr_func = relay.Function(inputs, expr)
        net, params = testing.create_workload(expr_func)

        # Build the subgraph
        # FIXME(@Soo): We should redesign Target class to deal with new TVM build interface
        target_str = target.__str__()
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build(net, target_str, params=params)

        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        setup_mod_inputs(module)

        # data_shape = get_data_shape(expr)
        #data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        # data = get_data(expr)
        # module.set_input("data", data)
        ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

        return measure(ftimer, is_net=False)

        # target_str = target.__str__()
        # ctx = tvm.context(target_str, 0)
        # lib = relay.build_module.build(net, target_str, params=params)
        # module = runtime.GraphModule(lib["default"](ctx))
        #
        # # Setup execution
        # data_shape = get_data_shape(expr)
        # data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        # module.set_input("data", data)
        # ftimer = module.module.time_evaluator("run", ctx, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
        #
        # return measure(ftimer, is_net=False)

def get_conv_attr(expr):
    assert (is_call_node(expr))
    # note that only call node has "op" attribute corresponding to a single backend operator
    op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span

    # extract conv attributes
    strides, padding, out_channels, dilation = \
        list(attrs.strides), list(attrs.padding), int(attrs.channels), list(attrs.dilation)

    #kernel_size = args[1].type_annotation.shape
    kernel_size = list(map(lambda x: x.value, args[1].type_annotation.shape))
    dtype = args[0].type_annotation.dtype

    return strides, padding, out_channels, dilation, kernel_size, dtype, attrs.groups, attrs.data_layout, attrs.kernel_layout

class CuDNNCostFunc(TargetCostFunc):

    def __init__(self):
        super().__init__()

    @staticmethod
    def measure_cost(name, expr, target):
        from tvm.contrib import cudnn
        from tvm import te
        from tvm import tir

        # NOTE: ASSUMPTIONS
        # conv_mode = CROSS_CORELATION    (NOT CUDNN_CONVOLUTION)
        # data type = float32
        # Tensor layout = "NCHW"
        # Kernel layout = "OIHW"
        # # of groups = 1
        # nanProp_mode = 0       # CUDNN_NOT_PROPAGATE_NAN

        conv_mode = 1          # mode: CUDNN_CONVOLUTION
        activation_mode = 1    # CUDNN_RELU
        nanProp_mode = 0       # CUDNN_NOT_PROPAGATE_NAN
        full_dims = 4
        dims = full_dims-2
        conv_algo = -1         # if -1 is set, try differnt algs to pick the best one
        actvCoeff = 1e100
        groups = 1
        data_layout = 0        # CUDNN_NCHW
        dtype = 'float32'


        op_name = name[len("cudnn_"):]
        # Create workload
        inputs = relay.analysis.free_vars(expr)
        expr_func = relay.Function(inputs, expr)
        net, params = testing.create_workload(expr_func)

        # Build the subgraph
        # FIXME(@Soo): We should redesign Target class to deal with new TVM build interface
        target_str = target.__str__()
        ctx = tvm.gpu()
        # ctx = tvm.context(target_str, 0)

        data_shape = get_data_shape(expr)
        in_channels = data_shape[1]

        if "conv2d" in op_name:

            if op_name == "conv2d+biasadd+relu":
                strides, padding, out_channels, dilation, kernel_size, dtype, groups, str_data_layout, str_kernel_layout = get_conv_attr(expr.args[0].args[0])
            elif op_name == 'conv2d+relu':
                strides, padding, out_channels, dilation, kernel_size, dtype, groups, str_data_layout, str_kernel_layout = get_conv_attr(expr.args[0])
            elif op_name == 'conv2d':
                strides, padding, out_channels, dilation, kernel_size, dtype, groups, str_data_layout, str_kernel_layout = get_conv_attr(expr)
            else:
                raise Exception(f"{op_name} is not supported for CUDNN")

            if "conv2d" in op_name:
                import sys
                print("####################Conv2d Attr: ", str_data_layout, str_kernel_layout, str(groups), file=sys.stderr)
                print(expr, file=sys.stderr)
                print(groups, data_shape, out_channels, kernel_size, dtype, file=sys.stderr)


            assert(str_data_layout == "NCHW")
            assert(dtype == "float32")

            output_shape = cudnn.conv_output_shape(
                data_layout,
                padding,
                strides,
                dilation,
                list(data_shape),
                list(kernel_size),
                dtype,
                dtype,
                groups
            )

            if conv_algo == -1:
                # For now if we try to call `cudnnFindConvolutionForwardAlgorithm` when
                # using INT8 data type, CuDNN will crash down.
                # On the other hand, CuDNN only support IMPLICIT_PRECOMP_GEMM at NHWC format
                if data_layout == 1 and conv_dtype == "int32":
                    conv_algo = 1
                else:
                    conv_algo = cudnn.conv_find_algo(
                        data_layout,
                        padding,
                        strides,
                        dilation,
                        list(data_shape),
                        list(kernel_size),
                        output_shape,
                        dtype,
                        dtype,
                        groups
                )


        if op_name == "conv2d":
            # Define input tensor shapes and variables
            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            te_kernel = te.placeholder(kernel_size, name="kernel", dtype=dtype)

            cuDNN_OP = te.extern(
                output_shape,
                [te_data, te_kernel],
                lambda ins, outs: tvm.tir.call_packed(
                    "tvm.contrib.cudnn.conv2d.forward",
                      conv_mode, # mode: CUDNN_CONVOLUTION
                      data_layout, # CUDNN_TENSOR_NCHW
                      conv_algo, # ALGO
                      padding[0], padding[1],
                      strides[0], strides[1],
                      dilation[0], dilation[1],
                      ins[0], # x
                      ins[1], # w
                      outs[0], # y
                      dtype,
                      groups,
                    ),
                    name="y",
                )

            s = te.create_schedule(cuDNN_OP.op)
            func = tvm.build(s, [te_data, te_kernel, cuDNN_OP], target_str, target_host="llvm")

            data_in = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
            data = tvm.nd.array(data_in, ctx)
            weight = tvm.nd.array(params["weight"], ctx)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), ctx)

            ftimer = func.time_evaluator(func.entry_name, ctx, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
            perf = measure(ftimer, False, data, weight, output)


        elif op_name == "conv2d+relu":
            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            te_kernel = te.placeholder(kernel_size, name="kernel", dtype=dtype)

            print("Conv2d+RELU")
            cuDNN_OP = te.extern(
                output_shape,
                [te_data, te_kernel],
                lambda ins, outs: tvm.tir.call_packed(
                    "tvm.contrib.cudnn.conv2d+activation.forward",
                    conv_mode,  # mode: CUDNN_CONVOLUTION
                    data_layout,  # CUDNN_TENSOR_NCHW
                    conv_algo,
                    padding[0], padding[1],
                    strides[0], strides[1],
                    dilation[0], dilation[1],
                    dtype,
                    ins[0],  # x
                    ins[1],  # w
                    outs[0],  # y
                    groups,
                    1,  # alphas[0],
                    0,  # alphas[1],
                    1,  # alphas[0] for z
                    0,
                    activation_mode,
                    nanProp_mode,
                    actvCoeff
                ),
                name="y",
            )
            print("Build Conv2d+RELU")

            s = te.create_schedule(cuDNN_OP.op)
            func = tvm.build(s, [te_data, te_kernel, cuDNN_OP], target_str, target_host="llvm")

            data_in = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
            kernel_in = np.random.uniform(-1, 1, size=kernel_size).astype(dtype)
            data = tvm.nd.array(data_in, ctx)
            weight = tvm.nd.array(kernel_in, ctx)
            #weight = tvm.nd.array(params["weight"], ctx)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), ctx)

            print("Measure Conv2d+RELU")
            ftimer = func.time_evaluator(func.entry_name, ctx, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
            perf = measure(ftimer, False, data, weight, output)
            print(" ==> perf: ", perf)



        elif op_name == "conv2d+biasadd+relu":
            # Warning: We assuem that args[1] corresponds to bias
            bias_tensor = expr.args[0].args[1].data

            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            te_kernel = te.placeholder(kernel_size, name="kernel", dtype=dtype)
            te_z      = te.placeholder(output_shape, name="Z", dtype=dtype)

            # Note that bias is a constant and not in params cuz it's a constant
            # te_bias   = te.placeholder(params["bias"].shape, name="bias", dtype=dtype)
            te_bias = te.placeholder(bias_tensor.shape, name="bias", dtype=dtype)

            cuDNN_OP = te.extern(
                output_shape,
                [te_data, te_kernel, te_z, te_bias],
                lambda ins, outs: tvm.tir.call_packed(
                    "tvm.contrib.cudnn.conv2d+bias+activation.forward",
                    conv_mode,  # mode: CUDNN_CONVOLUTION
                    data_layout,  # CUDNN_TENSOR_NCHW
                    conv_algo,
                    padding[0], padding[1],
                    strides[0], strides[1],
                    dilation[0], dilation[1],
                    dtype,
                    ins[0],  # x
                    ins[1],  # w
                    ins[2],  # z
                    ins[3],  # bias
                    outs[0],  # y
                    groups,
                    1,  # alphas[0],
                    0,  # alphas[1],
                    1,  # alphas[0] for z
                    0,
                    activation_mode,
                    nanProp_mode,
                    actvCoeff
                ),
                name="y",
            )

            s = te.create_schedule(cuDNN_OP.op)
            func = tvm.build(s, [te_data, te_kernel, te_z, te_bias, cuDNN_OP], target_str, target_host="llvm")

            data_in = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
            data = tvm.nd.array(data_in, ctx)
            weight = tvm.nd.array(params["weight"], ctx)
            ze = tvm.nd.array(np.zeros(output_shape, dtype=dtype), ctx)
            # bias = tvm.nd.array(params["bias"], ctx)
            bias = tvm.nd.array(bias_tensor.asnumpy(), ctx)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), ctx)

            print("Measure Conv2d+BIAS+RELU")
            print(" ==> ", data_shape, kernel_size, output_shape, bias_tensor.shape)
            print(" ==> ", conv_algo, conv_mode, padding, strides, dilation)
            ftimer = func.time_evaluator(func.entry_name, ctx, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
            perf = measure(ftimer, False, data, weight, ze, bias, output)
            print(" ==> perf: ", perf)


        elif op_name == "softmax":
            axis = expr.attrs.axis

            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            output_shape = data_shape
            cuDNN_OP = te.extern(
                output_shape,
                [te_data],
                lambda ins, outs: tvm.tir.call_packed(
                    "tvm.contrib.cudnn.softmax.forward",
                      ins[0], # x
                      outs[0], # y
                      axis
                    ),
                    name="y",
                )
            s = te.create_schedule(cuDNN_OP.op)
            func = tvm.build(s, [te_data, cuDNN_OP], target_str, target_host="llvm")

            data_in = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
            data = tvm.nd.array(data_in, ctx)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), ctx)

            ftimer = func.time_evaluator(func.entry_name, ctx, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
            perf = measure(ftimer, False, data, output)


        elif op_name == "biasadd":
            axis = expr.attrs.axis
            # Warning: We assuem that args[1] corresponds to bias
            bias_tensor = expr.args[1].data

            # Note that bias is a constant and not in params cuz it's a constant
            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            # te_bias   = te.placeholder(params["bias"].shape, name="bias", dtype=dtype)
            te_bias = te.placeholder(bias_tensor.shape, name="bias", dtype=dtype)
            output_shape = data_shape

            cuDNN_OP = te.extern(
                output_shape,
                [te_data, te_bias],
                lambda ins, outs: tvm.tir.call_packed(
                    "tvm.contrib.cudnn.add",
                      ins[0], # x
                      outs[0], # y
                      1,0, # alpha, beta
                      axis
                    ),
                    name="y",
                )
            s = te.create_schedule(cuDNN_OP.op)
            func = tvm.build(s, [te_data, te_bias, cuDNN_OP], target_str, target_host="llvm")

            data_in = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
            data = tvm.nd.array(data_in, ctx)
            # bias = tvm.nd.array(params["bias"], ctx)
            bias = tvm.nd.array(bias_tensor.asnumpy(), ctx)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), ctx)

            ftimer = func.time_evaluator(func.entry_name, ctx, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
            perf = measure(ftimer, False, data, bias, output)


        elif op_name == "relu":

            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            output_shape = data_shape
            cuDNN_OP = te.extern(
                output_shape,
                [te_data],
                lambda ins, outs: tvm.tir.call_packed(
                    "tvm.contrib.cudnn.activation.forward",
                      ins[0], # x
                      outs[0], # y,
                      1,0, #alpha, beta
                      activation_mode,
                      nanProp_mode,
                      actvCoeff
                    ),
                    name="y",
                )
            s = te.create_schedule(cuDNN_OP.op)
            func = tvm.build(s, [te_data, cuDNN_OP], target_str, target_host="llvm")

            data_in = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
            data = tvm.nd.array(data_in, ctx)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), ctx)

            ftimer = func.time_evaluator(func.entry_name, ctx, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
            perf = measure(ftimer, False, data, output)


        elif op_name == "maxpool2d":
            assert(is_call_node(expr))
            # note that only call node has "op" attribute corresponding to a single backend operator
            attrs = expr.attrs

            # extract maxpool2d attributes
            #NOTE: layout, ceil is currently ignored.
            strides, padding, pool_size = list(attrs.strides), list(attrs.padding), list(attrs.pool_size)


            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            output_shape = list(data_shape)

            #outputDim = 1 + (inputDim + 2*padding - windowDim)/poolingStride;
            for i in range(dims):
                shape_imm = 1 +  tir.div(data_shape[i+2] + 2*padding[i]-pool_size[i], strides[i])
                output_shape[i+2] = shape_imm.value


            cuDNN_OP = te.extern(
                output_shape,
                [te_data],
                lambda ins, outs: tvm.tir.call_packed(
                    "tvm.contrib.cudnn.pooling.forward",
                      ins[0], # x
                      outs[0], # y
                      1, 0, # Alpha, beta
                      3, # MODE: CUDNN_POOLING_MAX_DETERMINISTIC
                      nanProp_mode,
                      pool_size[0], pool_size[1],
                      padding[0], padding[1],
                      strides[0], strides[1]
                    ),
                    name="y",
                )

            s = te.create_schedule(cuDNN_OP.op)
            func = tvm.build(s, [te_data, cuDNN_OP], target_str, target_host="llvm")

            data_in = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
            data = tvm.nd.array(data_in, ctx)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), ctx)

            ftimer = func.time_evaluator(func.entry_name, ctx, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
            perf = measure(ftimer, False, data, output)

        elif op_name == "bn":
            assert(is_tuplegetitem_node(expr))

            stat_shape = (1,in_channels,1,1)
            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            te_bn_gamma = te.placeholder(stat_shape, name="bn_gamma", dtype=dtype)
            te_bn_beta = te.placeholder(stat_shape, name="bn_beta", dtype=dtype)
            te_bn_mean = te.placeholder(stat_shape, name="bn_mean", dtype=dtype)
            te_bn_var = te.placeholder(stat_shape, name="bn_var", dtype=dtype)

            attrs = expr.tuple_value.attrs
            eps = attrs.epsilon
            output_shape = data_shape
            # NOTE: expr.attrs.center, expr.attrs.scale are currently ignored.
            assert(attrs.axis==1)


            # BN mode
            # CUDNN_BATCHNORM_PER_ACTIVATION(0): param dim should be 1xCxHxW: axis = 0
            # CUDNN_BATCHNORM_SPATIAL(1): param dim should be 1xCx1x1         axis = 1
            # CUDNN_BATCHNORM_SPATIAL_PERSISTENT(1): param dim should be 1xCx1x1

            cuDNN_OP = te.extern(
                output_shape,
                [te_data, te_bn_gamma, te_bn_beta, te_bn_mean, te_bn_var],
                lambda ins, outs: tvm.tir.call_packed(
                    "tvm.contrib.cudnn.batchnorm.forward",
                      1, #MODE
                      ins[0], # x
                      outs[0], # y
                      ins[1], # scale = gamma
                      ins[2], # bias = beta
                      ins[3], # mean
                      ins[4], # var
                      1, 0, # Alpha, beta
                      eps
                    ),
                    name="y",
                )

            s = te.create_schedule(cuDNN_OP.op)
            func = tvm.build(s, [te_data, te_bn_gamma, te_bn_beta, te_bn_mean, te_bn_var, cuDNN_OP], target_str, target_host="llvm")

            data_in = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
            data = tvm.nd.array(data_in, ctx)

            gamma = tvm.nd.array(params["bn_data_gamma"].asnumpy().reshape(stat_shape), ctx)
            beta = tvm.nd.array(params["bn_data_beta"].asnumpy().reshape(stat_shape), ctx)
            mean = tvm.nd.array(params["bn_data_moving_mean"].asnumpy().reshape(stat_shape), ctx)
            var = tvm.nd.array(params["bn_data_moving_var"].asnumpy().reshape(stat_shape), ctx)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), ctx)

            ftimer = func.time_evaluator(func.entry_name, ctx, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
            perf = measure(ftimer, False, data, gamma, beta, mean, var, output)

        else:
            # NOT IMPLEMENTED
            assert(0)

        # Note that perf contains (mean(perf), std(perf))
        return perf



class TensorRTCostFunc(TargetCostFunc):
    def __init__(self):
        super().__init__()

    @staticmethod
    def measure_cost(name, expr, target):

        # Create workload
        inputs = relay.analysis.free_vars(expr)
        expr_func = relay.Function(inputs, expr)
        net, params = testing.create_workload(expr_func)

        from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
        mod, config = partition_for_tensorrt(net, params)

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

        # from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
        # net, config = partition_for_tensorrt(net, params)
        #
        # target = "cuda"
        # with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
        #     lib = relay.build(net, target=target, params=params)
        #
        # lib.export_library('compiled.so')
        # ctx = tvm.gpu(0)
        # loaded_lib = tvm.runtime.load_module('compiled.so')
        # module = tvm.contrib.graph_runtime.GraphModule(loaded_lib['default'](ctx))
        #
        # data_shape = get_data_shape(expr)
        # data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        # module.set_input("data", data)
        # ftimer = module.module.time_evaluator("run", ctx, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)
        # return measure(ftimer)



target_to_cost_func = {
    #GPU
    Target.TVM_GPU_AUTOTVM: TVMSubGraphCostFunc_AutoTVM(),
    Target.TVM_GPU_AUTOSCH: TVMSubGraphCostFunc_AutoSch(),
    Target.TVM_GPU_NO_TUNING: TVMSubGraphCostFunc_NoTuning(),
    Target.CUDNN: CuDNNCostFunc(),
    Target.TENSORRT: TensorRTCostFunc(),
    Target.CUBLAS: TVMSubGraphCostFunc_NoTuning(),

    # CPU
    Target.TVM_CPU: TVMSubGraphCostFunc_NoTuning(),
}

def get_target_cost_func(target):
    return target_to_cost_func[target].measure_cost


