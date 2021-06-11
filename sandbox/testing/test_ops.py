import tvm
from tvm import te
import numpy as np
import tvm.contrib.graph_executor as runtime
from tvm import relay
from tvm.relay import testing
import pytest as pyt
from tvm.relay.transform.optimizer.custom_fusion_pass import *


gt_target = "cuda"

def genKey(config):
    key = ""
    for e in config:
        if isinstance(e, str):
            key += e
        else:
            key += str(e)
        key += ", "
    return key

def get_gt_net(config):
    op = config["op"]
    target = gt_target
    batch_size = config["batch_size"]
    data_shape = config["data_shape"]
    out_channels = config["out_channels"]
    kernel_size = config["kernel_size"]
    strides = config["strides"]
    padding = config["padding"]
    dilation = config["dilation"]
    groups = config["groups"]
    data_layout = config["data_layout"]
    kernel_layout = config["kernel_layout"]
    out_layout = config["out_layout"]
    out_dtype = config["out_dtype"]
    pool_size = config["pool_size"]
    axis = config["axis"]


    key = genKey(config)

    # Define input tensor shapes and variables
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight")
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")
    bias = relay.var("bias")

    # Process given operators
    if op == "conv2d":
        simple_net = relay.nn.conv2d(
            data = data,
            weight = weight,
            strides = strides,
            padding = padding,
            dilation = dilation,
            groups = groups,
            channels = out_channels,
            kernel_size = kernel_size,
            data_layout = data_layout,
            kernel_layout = kernel_layout,
            out_layout = out_layout,
            out_dtype = "",
        )
    elif op == "bn":
        simple_net = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean, bn_mvar, axis)[0]
    elif op == "relu":
        simple_net = relay.nn.relu(data)
    elif op == "conv2d+bn":
        simple_net = relay.nn.conv2d(
            data = data,
            weight = weight,
            strides = strides,
            padding = padding,
            dilation = dilation,
            groups = groups,
            channels = out_channels,
            kernel_size = kernel_size,
            data_layout = data_layout,
            kernel_layout = kernel_layout,
            out_layout = out_layout,
            out_dtype = "",
        )
        simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    elif op == "bn+relu":
        simple_net = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
        simple_net = relay.nn.relu(simple_net)

    elif op == "biasadd":
        simple_net = relay.nn.bias_add(
            data = data,
            bias = bias,
            axis = axis
        )
    elif op == "conv2d+bn+relu":
        simple_net = relay.nn.conv2d(
            data = data,
            weight = weight,
            strides = strides,
            padding = padding,
            dilation = dilation,
            groups = groups,
            channels = out_channels,
            kernel_size = kernel_size,
            data_layout = data_layout,
            kernel_layout = kernel_layout,
            out_layout = out_layout,
            out_dtype = "",
        )
        simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
        simple_net = relay.nn.relu(simple_net)
    elif op == "conv2d+bias+relu":
        simple_net = relay.nn.conv2d(
            data = data,
            weight = weight, # conv kernel
            strides = strides,
            padding = padding,
            dilation = dilation,
            groups = groups,
            channels = out_channels,
            kernel_size = kernel_size,
            data_layout = data_layout,
            kernel_layout = kernel_layout,
            out_layout = out_layout,
            out_dtype = "",
        )
        simple_net = relay.nn.bias_add(simple_net, bias)
        simple_net = relay.nn.relu(simple_net)

    elif op == "softmax":
        simple_net = relay.nn.softmax(
            data = data,
            axis = axis
        )


    elif op == "maxpool2d":
        simple_net = relay.nn.max_pool2d(
            data = data,
            pool_size = pool_size,
            strides = strides,
            padding = padding,
        )

    # Create workload
    inputs = relay.analysis.free_vars(simple_net)
    simple_net = relay.Function(inputs, simple_net)

    return simple_net



def ref_tvm_build_cudnn(config):
    def impl(neural_in):
        op = config["op"]
        target = gt_target
        batch_size = config["batch_size"]
        data_shape = config["data_shape"]
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        strides = config["strides"]
        padding = config["padding"]
        dilation = config["dilation"]
        groups = config["groups"]
        data_layout = config["data_layout"]
        kernel_layout = config["kernel_layout"]
        out_layout = config["out_layout"]
        out_dtype = config["out_dtype"]
        pool_size = config["pool_size"]
        axis = config["axis"]


        key = genKey(config)

        # Define input tensor shapes and variables
        data = relay.var("data", relay.TensorType(data_shape, "float32"))
        weight = relay.var("weight")
        bn_gamma = relay.var("bn_gamma")
        bn_beta = relay.var("bn_beta")
        bn_mmean = relay.var("bn_mean")
        bn_mvar = relay.var("bn_var")
        bias = relay.var("bias")

        # Process given operators
        if op == "conv2d":
            simple_net = relay.nn.conv2d(
                data = data,
                weight = weight,
                strides = strides,
                padding = padding,
                dilation = dilation,
                groups = groups,
                channels = out_channels,
                kernel_size = kernel_size,
                data_layout = data_layout,
                kernel_layout = kernel_layout,
                out_layout = out_layout,
                out_dtype = "",
            )
        elif op == "bn":
            simple_net = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean, bn_mvar, axis)[0]
        elif op == "relu":
            simple_net = relay.nn.relu(data)
        elif op == "conv2d+bn":
            simple_net = relay.nn.conv2d(
                data = data,
                weight = weight,
                strides = strides,
                padding = padding,
                dilation = dilation,
                groups = groups,
                channels = out_channels,
                kernel_size = kernel_size,
                data_layout = data_layout,
                kernel_layout = kernel_layout,
                out_layout = out_layout,
                out_dtype = "",
            )
            simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
        elif op == "bn+relu":
            simple_net = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
            simple_net = relay.nn.relu(simple_net)

        elif op == "biasadd":
            simple_net = relay.nn.bias_add(
                data = data,
                bias = bias,
                axis = axis
            )
        elif op == "conv2d+bn+relu":
            simple_net = relay.nn.conv2d(
                data = data,
                weight = weight,
                strides = strides,
                padding = padding,
                dilation = dilation,
                groups = groups,
                channels = out_channels,
                kernel_size = kernel_size,
                data_layout = data_layout,
                kernel_layout = kernel_layout,
                out_layout = out_layout,
                out_dtype = "",
            )
            simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
            simple_net = relay.nn.relu(simple_net)
        elif op == "conv2d+bias+relu":
            simple_net = relay.nn.conv2d(
                data = data,
                weight = weight, # conv kernel
                strides = strides,
                padding = padding,
                dilation = dilation,
                groups = groups,
                channels = out_channels,
                kernel_size = kernel_size,
                data_layout = data_layout,
                kernel_layout = kernel_layout,
                out_layout = out_layout,
                out_dtype = "",
            )
            simple_net = relay.nn.bias_add(simple_net, bias)
            simple_net = relay.nn.relu(simple_net)

        elif op == "softmax":
            simple_net = relay.nn.softmax(
                data = data,
                axis = axis
            )


        elif op == "maxpool2d":
            simple_net = relay.nn.max_pool2d(
                data = data,
                pool_size = pool_size,
                strides = strides,
                padding = padding,
            )

        # Create workload
        inputs = relay.analysis.free_vars(simple_net)
        simple_net = relay.Function(inputs, simple_net)

        simple_net, params = testing.create_workload(simple_net)
        simple_net = simple_net["main"].with_attr("CustomFusionPass", CustomFusionPass.DP)

        params = neural_in["params"]
        opt_level = 2
        target_str = 'cuda'# -libs=cudnn'
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(simple_net, target_str, params=params)#tvm.target.cuda(), params=params)

        dev = tvm.device(target_str, 0)#"cuda", 0)
        #dev = tvm.device("cuda -libs=cudnn", 0)
        #lib = relay.build_module.build(simple_net, "cuda")
        mod = runtime.GraphModule(lib["default"](dev))
        mod.set_input("data", neural_in["data"])
        mod.set_input(**params)
        mod.run()
        return mod.get_output(0)

    return impl




def ref_tvm_op_build_cudnn(config):
    from tvm.contrib import cudnn

    def check_implementation(op_name):
        if not tvm.get_global_func(op_name, allow_missing=True):
            raise Exception("Not compiled with fused cudnn support; can't build this tutorial")

    def impl(neural_in):
        op = config["op"]
        target = config["target"]
        batch_size = config["batch_size"]
        data_shape = config["data_shape"]
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        strides = config["strides"]
        padding = config["padding"]
        dilation = config["dilation"]
        groups = config["groups"]
        data_layout = config["data_layout"]
        kernel_layout = config["kernel_layout"]
        out_layout = config["out_layout"]
        out_dtype = config["out_dtype"]
        pool_size = config["pool_size"]
        axis = config["axis"]

        # NOTE: CUDNN SPECIFIC CONFIGS
        conv_mode = 1          # mode: CUDNN_CONVOLUTION
        conv_algo = -1          # pick the best performing one via measurement
        activation_mode = 1    # CUDNN_RELU
        nanProp_mode = 0   # CUDNN_NOT_PROPAGATE_NAN
        full_dims = 4
        dims = full_dims-2
        actvCoeff = 1e100
        dtype = 'float32'

        # bias_shape == (1,out_channels)


        key = genKey(config)

        # NOTE: Currently, only supports certain cases
        if data_layout == "NCHW":
            data_layout = 0
            in_channels = data_shape[1]
        else:
            assert(0)

        if len(kernel_size) == 2:
            kernel_size = [ out_channels, in_channels, *kernel_size ]


        # params
        params = neural_in["params"]
        dev = tvm.device(target, 0)


        if "conv2d" in op:

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


        # Process given operators
        if op == "conv2d+bias+relu":
            #padding, strides, dilation, _, _ = cudnn._prepare_global_func_params(dims, padding, strides, dilation)

            # Define input tensor shapes and variables
            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            te_kernel = te.placeholder(kernel_size, name="kernel", dtype=dtype)
            te_z      = te.placeholder(output_shape, name="Z", dtype=dtype)
            te_bias   = te.placeholder(params["bias"].shape, name="bias", dtype=dtype)

            cuDNN_OP = te.extern(
                output_shape,
                [te_data, te_kernel, te_z, te_bias],
                lambda ins, outs: tvm.tir.call_packed(
                    "tvm.contrib.cudnn.conv2d+bias+activation.forward",
                      conv_mode, # mode: CUDNN_CONVOLUTION
                      data_layout, # CUDNN_TENSOR_NCHW
                      conv_algo,
                      padding[0], padding[1],
                      strides[0], strides[1],
                      dilation[0], dilation[1],
                      dtype,
                      ins[0], # x
                      ins[1], # w
                      ins[2], # z
                      ins[3], # bias
                      outs[0], # y
                      groups,
                      1,#alphas[0],
                      0,#alphas[1],
                      1,#alphas[0] for z
                      0,
                      activation_mode,
                      nanProp_mode,
                      actvCoeff
                    ),
                    name="y",
                )

            s = te.create_schedule(cuDNN_OP.op)
            func = tvm.build(s, [te_data, te_kernel, te_z, te_bias, cuDNN_OP], "cuda -libs=cudnn", target_host="llvm")

            # convert np.ndarray to tvm.nd.array
            data = tvm.nd.array(neural_in["data"], dev)
            weight = tvm.nd.array(params["weight"], dev)
            ze = tvm.nd.array(np.zeros(output_shape, dtype=dtype), dev)
            bias = tvm.nd.array(params["bias"], dev)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), dev)
            func(data, weight, ze, bias, output)

        elif op == "conv2d":
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
                      conv_algo,
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
            func = tvm.build(s, [te_data, te_kernel, cuDNN_OP], "cuda -libs=cudnn", target_host="llvm")

            data = tvm.nd.array(neural_in["data"], dev)
            weight = tvm.nd.array(params["weight"], dev)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), dev)

            func(data, weight, output)

        elif op == "softmax":
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
            func = tvm.build(s, [te_data, cuDNN_OP], "cuda -libs=cudnn", target_host="llvm")
            data = tvm.nd.array(neural_in["data"], dev)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), dev)

            func(data, output)

        elif op == "biasadd" or op == "add":
            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            te_bias   = te.placeholder(params["bias"].shape, name="bias", dtype=dtype)
            output_shape = data_shape

            #assert(axis==-1 or axis==1)
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
            func = tvm.build(s, [te_data, te_bias, cuDNN_OP], "cuda -libs=cudnn", target_host="llvm")

            data = tvm.nd.array(neural_in["data"], dev)
            bias = tvm.nd.array(params["bias"], dev)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), dev)

            func(data, bias, output)


        elif op == "relu":
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
            func = tvm.build(s, [te_data, cuDNN_OP], "cuda -libs=cudnn", target_host="llvm")
            data = tvm.nd.array(neural_in["data"], dev)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), dev)

            func(data, output)



        elif op == "maxpool2d":
            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            output_shape = list(data_shape)
            #outputDim = 1 + (inputDim + 2*padding - windowDim)/poolingStride;
            for i in range(dims):
                output_shape[i+2] = int(1 + (data_shape[i+2] + 2*padding[i]-pool_size[i])/strides[i])


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
            func = tvm.build(s, [te_data, cuDNN_OP], "cuda -libs=cudnn", target_host="llvm")

            data = tvm.nd.array(neural_in["data"], dev)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), dev)

            func(data, output)


        elif op == "bn":
            stat_shape = (1,in_channels,1,1)
            te_data   = te.placeholder(data_shape, name="data", dtype=dtype)
            te_bn_gamma = te.placeholder(stat_shape, name="bn_gamma", dtype=dtype)
            te_bn_beta = te.placeholder(stat_shape, name="bn_beta", dtype=dtype)
            te_bn_mean = te.placeholder(stat_shape, name="bn_mean", dtype=dtype)
            te_bn_var = te.placeholder(stat_shape, name="bn_var", dtype=dtype)
            #axis


            eps = 1e-5
            output_shape = data_shape


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
            func = tvm.build(s, [te_data, te_bn_gamma, te_bn_beta, te_bn_mean, te_bn_var, cuDNN_OP], "cuda -libs=cudnn", target_host="llvm")

            data = tvm.nd.array(neural_in["data"], dev)
            gamma = tvm.nd.array(params["bn_gamma"].asnumpy().reshape(stat_shape), dev)
            beta = tvm.nd.array(params["bn_beta"].asnumpy().reshape(stat_shape), dev)
            mean = tvm.nd.array(params["bn_mean"].asnumpy().reshape(stat_shape), dev)
            var = tvm.nd.array(params["bn_var"].asnumpy().reshape(stat_shape), dev)
            output = tvm.nd.array(np.zeros(output_shape, dtype=dtype), dev)

            func(data, gamma, beta, mean, var, output)
        else:
            assert(0)


        return output



    check_implementation("tvm.contrib.cudnn.batchnorm.forward")
    check_implementation("tvm.contrib.cudnn.activation.forward")
    check_implementation("tvm.contrib.cudnn.add")
    check_implementation("tvm.contrib.cudnn.pooling.forward")
    check_implementation("tvm.contrib.cudnn.reduce")
    check_implementation("tvm.contrib.cudnn.scale")
    check_implementation("tvm.contrib.cudnn.conv2d+bias+activation.forward")
    check_implementation("tvm.contrib.cudnn.softmax.forward")



    return impl



def ref_impl(config):
    def impl(neural_in):
        op = config["op"]
        target = config["target"]
        batch_size = config["batch_size"]
        data_shape = config["data_shape"]
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        strides = config["strides"]
        padding = config["padding"]
        dilation = config["dilation"]
        groups = config["groups"]
        data_layout = config["data_layout"]
        kernel_layout = config["kernel_layout"]
        out_layout = config["out_layout"]
        out_dtype = config["out_dtype"]

        key = genKey(config)

        # Define input tensor shapes and variables
        data = relay.var("data", relay.TensorType(data_shape, "float32"))
        weight = relay.var("weight")
        bn_gamma = relay.var("bn_gamma")
        bn_beta = relay.var("bn_beta")
        bn_mmean = relay.var("bn_mean")
        bn_mvar = relay.var("bn_var")
        bias = relay.var("bias")

        #weight = relay.var("weight", shape=((out_channels, data_shape[1], kernel_size[0], kernel_size[1])))
        #bn_gamma = relay.var("bn_gamma", shape=(out_channels,))
        #bn_beta = relay.var("bn_beta", shape=(out_channels,))
        #bn_mmean = relay.var("bn_mean", shape=(out_channels,))
        #bn_mvar = relay.var("bn_var", shape=(out_channels,))

        # Process given operators
        if op == "conv2d":
            simple_net = relay.nn.conv2d(
                data = data,
                weight = weight,
                strides = strides,
                padding = padding,
                dilation = dilation,
                groups = groups,
                channels = out_channels,
                kernel_size = kernel_size,
                data_layout = data_layout,
                kernel_layout = kernel_layout,
                out_layout = out_layout,
                out_dtype = "",
            )
        elif op == "bn":
            simple_net = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
        elif op == "relu":
            simple_net = relay.nn.relu(data)
        elif op == "conv2d+bn":
            simple_net = relay.nn.conv2d(
                data = data,
                weight = weight,
                strides = strides,
                padding = padding,
                dilation = dilation,
                groups = groups,
                channels = out_channels,
                kernel_size = kernel_size,
                data_layout = data_layout,
                kernel_layout = kernel_layout,
                out_layout = out_layout,
                out_dtype = "",
            )
            simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
        elif op == "bn+relu":
            simple_net = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
            simple_net = relay.nn.relu(simple_net)
        elif op == "conv2d+bn+relu":
            simple_net = relay.nn.conv2d(
                data = data,
                weight = weight,
                strides = strides,
                padding = padding,
                dilation = dilation,
                groups = groups,
                channels = out_channels,
                kernel_size = kernel_size,
                data_layout = data_layout,
                kernel_layout = kernel_layout,
                out_layout = out_layout,
                out_dtype = "",
            )
            simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
            simple_net = relay.nn.relu(simple_net)

        elif op == "conv2d+bias+relu":
            simple_net = relay.nn.conv2d(
                data = data,
                weight = weight,
                strides = strides,
                padding = padding,
                dilation = dilation,
                groups = groups,
                channels = out_channels,
                kernel_size = kernel_size,
                data_layout = data_layout,
                kernel_layout = kernel_layout,
                out_layout = out_layout,
                out_dtype = "",
            )
            simple_net = relay.nn.bias_add(simple_net, bias)
            simple_net = relay.nn.relu(simple_net)



        # Create workload
        inputs = relay.analysis.free_vars(simple_net)

        simple_net = relay.Function(inputs, simple_net)

        net, _ = testing.create_workload(simple_net)

        params = neural_in["params"]
        # Bulid the subgraph
        dev = tvm.device(target, 0)
        lib = relay.build_module.build(net, target, params=params)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        module.set_input("data", neural_in["data"])

        module.run()
        # get output
        out = module.get_output(0)
        #print(out.asnumpy().shape)
        return out
    return impl

configs = [

    # ["conv2d", "cuda -libs=cudnn", 1, (1,3,224,224), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), -1, (1,16)],
    #["conv2d+bias+relu", "cuda -libs=cudnn", 1, (1,3,224,224), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), -1, (1,16)],
    # ["conv2d", "cuda -libs=cudnn", 1, (1,3,224,224), 16, (3,3), (2,2), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), -1,(1,16)],
    #["conv2d+bias+relu", "cuda -libs=cudnn", 1, (1,3,224,224), 16, (3,3), (2,2), (1,1), (1,1), 1, "NCHW", "OIHW", "", "",(2,2), -1, (1,16)],
    # ["conv2d+bias+relu", "cuda -libs=cudnn", 1, (1,3,224,224), 16, (3,3), (2,2), (1,1), (1,1), 1, "NCHW", "OIHW", "", "",(2,2), -1, (1,16)],

    # ["softmax", "cuda -libs=cudnn", 1, (1,1,4,4), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), -1, (1,16)],
    #["softmax", "cuda -libs=cudnn", 1, (1,1,4,4), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), 0, (1,16)],
    ["relu", "cuda -libs=cudnn", 1, (1,3,224,224), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), -1, (1,16)],
    #["biasadd", "cuda -libs=cudnn", 1, (13,31,224,12), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), -1, (1,16)],
    #["biasadd", "cuda -libs=cudnn", 1, (13,31,224,12), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), 1, (1,31,224)],
    #["biasadd", "cuda -libs=cudnn", 1, (13,31,224,12), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), 0, (1,31,224)],
    #["biasadd", "cuda -libs=cudnn", 1, (13,31,224,12), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), 2, (1,31,224)],
    #["maxpool2d", "cuda -libs=cudnn", 1, (1,1,4,4), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), -1, (1,16)],
    #["maxpool2d", "cuda -libs=cudnn", 1, (13,31,224,12), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), -1, (1,16)],
    #["maxpool2d", "cuda -libs=cudnn", 1, (224,224,112,12), 16, (2,2), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), -1, (1,16)],
    #["bn", "cuda -libs=cudnn", 1, (1,3,224,224), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), 1, (1,16)],
    #["bn", "cuda -libs=cudnn", 1, (11,23,24,24), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), 1, (1,16)],
    #["bn", "cuda -libs=cudnn", 1, (1,33,24,224), 16, (3,3), (1,1), (1,1), (1,1), 1, "NCHW", "OIHW", "", "", (2,2), 1, (1,16)],
]

dicts = []

for config in configs:
    # Define target backend and arguments for the subgraph
    op, target, batch_size, data_shape, out_channels, kernel_size, \
        strides, padding, dilation, groups, data_layout, kernel_layout, \
        out_layout, out_dtype, pool_size, axis, bias_shape = config
    d = {"op": op, "target": target, "batch_size": batch_size, "data_shape": data_shape, "out_channels": out_channels, "kernel_size": kernel_size, \
         "strides": strides, "padding": padding, "dilation": dilation, "groups": groups, "data_layout": data_layout, "kernel_layout": kernel_layout, \
         "out_layout": out_layout, "out_dtype": out_dtype, "bias_shape": bias_shape, "pool_size": pool_size, "axis": axis}
    dicts.append(d)

OUTPUT = "operator_cost.log"
# change this to test your ops!
# CLIENT_IMPLEMENTATION = ref_tvm_op_build_cudnn
CLIENT_IMPLEMENTATION = ref_tvm_build_cudnn
#CLIENT_IMPLEMENTATION = ref_impl
REPEAT = 1
import json
logs = {}

@pyt.mark.parametrize("config",dicts)
def test(config):
    gt_network = get_gt_net(config)

    reference_implementation = CLIENT_IMPLEMENTATION(config)

    for i in range(REPEAT):
        data = np.random.uniform(-1, 1, size=config["data_shape"]).astype("float32")
        #bias = np.random.uniform(-1, 1, size=config["bias_shape"]).astype("float32")
        net, params = testing.create_workload(gt_network)

        # Bulid the subgraph
        dev = tvm.device(gt_target, 0)
        lib = relay.build_module.build(net, gt_target, params=params)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        module.set_input("data", data)

        module.run()
        # get output
        out_gt = module.get_output(0).asnumpy()
        out_impl = reference_implementation({"data": data, "params": params}).asnumpy()


        assert(out_gt.shape == out_impl.shape)
        assert(pyt.approx(out_impl, rel=1e-7, abs=1e-7) == out_gt)

if __name__ == '__main__':
    for config in dicts:
        test(config)

    print("\n============= Completed ==================")
