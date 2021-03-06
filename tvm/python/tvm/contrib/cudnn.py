# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""External function interface to CuDNN v7 library."""
# pylint: disable-msg=C0103
import ctypes
import numpy as np
import tvm

import tvm._ffi
from tvm import te

# algos can be read from cudnn.h
_FWD_ALGOS = [
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_FWD_ALGO_COUNT",
]

_BWD_FILTER_ALGOS = [
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
    # non-deterministic
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
    # non-deterministic, algo0 with workspaceS
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",
    # not implemented
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",
]

_BWD_DATA_ALGOS = [
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
    # non-deterministic
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",
]

_ALGO_TYPE = ["fwd", "bwd_filter", "bwd_data"]


def algo_to_index(algo_type, algo_name):
    """Return a index represents the algorithm, which can be used in
    calling CuDNN function

    Parameters
    ----------
        algo_type : str
            ["fwd", "bwd_filter", "bwd_data]

        algo_name : str
            algorithm name in cudnn definition
            fwd = [
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
                "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
                "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
                "CUDNN_CONVOLUTION_FWD_ALGO_COUNT",
            ]
            bwd_filter = [
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
                # non-deterministic
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
                # non-deterministic, algo0 with workspaceS
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",
                # not implemented
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",
            ]
            bwd_data = [
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
                # non-deterministic
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",
            ]

    Returns
    -------
        algo: int
            algorithm index

    """
    idx = -1
    if algo_type == "fwd":
        idx = _FWD_ALGOS.index(algo_name)
    elif algo_type == "bwd_filter":
        idx = _BWD_FILTER_ALGOS.index(algo_name)
    elif algo_type == "bwd_data":
        idx = _BWD_DATA_ALGOS.index(algo_name)
    assert idx >= 0
    return idx


def _get_np_int32_array_handle(arr):
    """Return a void_p handle for a numpy array

    Parameters
    ----------
    arr: numpy.NDArray
        source numpy array

    Returns
    -------
    ptr:  ctypes.c_void_p
        pointer to the data
    """
    assert arr.dtype == np.int32
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    return ctypes.cast(ptr, ctypes.c_void_p)


def _prepare_global_func_params(dims, pad, stride, dilation, x_shape=None, w_shape=None):
    full_dims = dims + 2
    if x_shape:
        assert isinstance(x_shape, list)
        assert len(x_shape) == full_dims
    if w_shape:
        assert isinstance(w_shape, list)
        assert len(w_shape) == full_dims

    pad = (
        np.full(dims, pad, dtype=np.int32)
        if isinstance(pad, int)
        else np.array(pad, dtype=np.int32)
    )
    stride = (
        np.full(dims, stride, dtype=np.int32)
        if isinstance(stride, int)
        else np.array(stride, dtype=np.int32)
    )
    dilation = (
        np.full(dims, dilation, dtype=np.int32)
        if isinstance(dilation, int)
        else np.array(dilation, dtype=np.int32)
    )

    xshape = np.array(x_shape, dtype=np.int32) if x_shape else None
    wshape = np.array(w_shape, dtype=np.int32) if x_shape else None

    return pad, stride, dilation, xshape, wshape


def conv_output_shape(
    tensor_format, pad, stride, dilation, x_shape, w_shape, data_dtype, conv_dtype, groups=1
):
    """Get output shape of 2D or 3D convolution

    Paramters
    ---------
    tensor_format: int
        0: CUDNN_TENSOR_NCHW
        1: CUDNN_TENSOR_NHWC
        2: CUDNN_TENSOR_NCHW_VECT_C
    pad: int or list
        padding
    stride: int or list
        stride
    dilation: int or list
        dilation
    x_shape: list
        input shape
    w_shape: list
        weight shape
    data_dtype: str
        data type
    conv_dtype: str
        convolution type
    groups: int
        number of groups

    Returns
    -------
    oshape: list
        output shape
    """
    dims = len(x_shape)
    assert dims in (4, 5)

    pad, stride, dilation, xshape, wshape = _prepare_global_func_params(
        dims - 2, pad, stride, dilation, x_shape, w_shape
    )
    oshape = np.zeros((dims), dtype=np.int32)

    func = tvm._ffi.get_global_func("tvm.contrib.cudnn.conv.output_shape")
    func(
        tensor_format,
        dims - 2,
        _get_np_int32_array_handle(pad),
        _get_np_int32_array_handle(stride),
        _get_np_int32_array_handle(dilation),
        _get_np_int32_array_handle(xshape),
        _get_np_int32_array_handle(wshape),
        _get_np_int32_array_handle(oshape),
        data_dtype,
        conv_dtype,
        groups,
    )
    return list(oshape)


def conv_find_algo(
    tensor_format,
    pad,
    stride,
    dilation,
    x_shape,
    w_shape,
    y_shape,
    data_dtype,
    conv_dtype,
    groups=1,
):
    """Choose the best algo for the given input.

    Paramters
    ---------
    tensor_format: int
        0: CUDNN_TENSOR_NCHW
        1: CUDNN_TENSOR_NHWC
        2: CUDNN_TENSOR_NCHW_VECT_C
    pad: int or list
        padding
    stride: int or list
        stride
    dilation: int or list
        dilation
    x_shape: list
        input shape
    w_shape: list
        weight shape
    y_shape: list
        output shape
    data_dtype: str
        data type
    conv_dtype: str
        convolution type
    groups: int
        number of groups

    Returns
    -------
    algo: int
        algo chosen by CUDNN
    """
    dims = len(x_shape)
    assert dims in (4, 5)

    pad, stride, dilation, xshape, wshape = _prepare_global_func_params(
        dims - 2, pad, stride, dilation, x_shape, w_shape
    )
    yshape = np.array(y_shape, dtype=np.int32)
    func = tvm._ffi.get_global_func("tvm.contrib.cudnn.conv.find_algo")
    return func(
        tensor_format,
        dims - 2,
        _get_np_int32_array_handle(pad),
        _get_np_int32_array_handle(stride),
        _get_np_int32_array_handle(dilation),
        _get_np_int32_array_handle(xshape),
        _get_np_int32_array_handle(wshape),
        _get_np_int32_array_handle(yshape),
        data_dtype,
        conv_dtype,
        groups,
    )


def conv_forward(x, w, pad, stride, dilation, conv_mode, tensor_format, algo, conv_dtype, groups=1):
    """Create an extern op that compute 2D or 3D convolution with CuDNN

    Parameters
    ----------
    x: Tensor
        input feature map
    w: Tensor
        convolution weight
    pad: int or list
        padding
    stride: int or list
        stride
    dilation: int or list
        dilation
    conv_mode: int
        0: CUDNN_CONVOLUTION
        1: CUDNN_CROSS_CORRELATION
    tensor_format: int
        0: CUDNN_TENSOR_NCHW
        1: CUDNN_TENSOR_NHWC
        2: CUDNN_TENSOR_NCHW_VECT_C
    algo: int
        Forward algorithm, get index from ```algo_to_index``` function
        if algo == -1, the best algo will be chosen by CUDNN
    conv_dtype: str
        convolution type
    groups: int
        the number of groups

    Returns
    -------
    y: Tensor
        The result tensor
    """
    dims = len(x.shape)
    assert dims in (4, 5)

    conv_dtype = x.dtype if conv_dtype is None else conv_dtype
    pad, stride, dilation, _, _ = _prepare_global_func_params(dims - 2, pad, stride, dilation)

    x_shape = list(x.shape)

    if isinstance(x.shape[0], tvm.tir.expr.IntImm):
        oshape = conv_output_shape(
            tensor_format,
            pad,
            stride,
            dilation,
            x_shape,
            list(w.shape),
            x.dtype,
            conv_dtype,
            groups,
        )
        if algo == -1:
            # For now if we try to call `cudnnFindConvolutionForwardAlgorithm` when
            # using INT8 data type, CuDNN will crash down.
            # On the other hand, CuDNN only support IMPLICIT_PRECOMP_GEMM at NHWC format
            if tensor_format == 1 and conv_dtype == "int32":
                algo = 1
            else:
                algo = conv_find_algo(
                    tensor_format,
                    pad,
                    stride,
                    dilation,
                    list(x.shape),
                    list(w.shape),
                    oshape,
                    x.dtype,
                    conv_dtype,
                    groups,
                )
    else:
        # The dynamic batch size case, pretend this is a single batch
        x_shape[0] = 1
        oshape = conv_output_shape(
            tensor_format,
            pad,
            stride,
            dilation,
            x_shape,
            list(w.shape),
            x.dtype,
            conv_dtype,
            groups,
        )
        oshape[0] = x.shape[0]
        # This picks CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        # It seems this is the fastest among algorithms that are always applicable
        algo = 1

    if dims == 4:
        return te.extern(
            oshape,
            [x, w],
            lambda ins, outs: tvm.tir.call_packed(
                "tvm.contrib.cudnn.conv2d.forward",
                conv_mode,
                tensor_format,
                algo,
                pad[0],
                pad[1],
                stride[0],
                stride[1],
                dilation[0],
                dilation[1],
                ins[0],
                ins[1],
                outs[0],
                conv_dtype,
                groups,
            ),
            name="y",
        )

    return te.extern(
        oshape,
        [x, w],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cudnn.conv3d.forward",
            conv_mode,
            tensor_format,
            algo,
            pad[0],
            pad[1],
            pad[2],
            stride[0],
            stride[1],
            stride[2],
            dilation[0],
            dilation[1],
            dilation[2],
            ins[0],
            ins[1],
            outs[0],
            conv_dtype,
            groups,
        ),
        name="y",
    )


def softmax(x, axis=-1):
    """Compute softmax using CuDNN

    Parameters
    ----------
    x : tvm.te.Tensor
        The input tensor

    axis : int
        The axis to compute the softmax

    Returns
    -------
    ret : tvm.te.Tensor
        The result tensor
    """
    return te.extern(
        x.shape,
        [x],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cudnn.softmax.forward", ins[0], outs[0], axis
        ),
        name="y",
    )


def avg_pool2d(x, pool_size, strides, padding, pool_type, ceil_mode, data_layout, count_include_pad):
    dims = 2
    data_shape = x.shape
    output_shape = list(data_shape)
    #outputDim = 1 + (inputDim + 2*padding - windowDim)/poolingStride;
    for i in range(dims):
        output_shape[i+2] = int(1 + tvm.tir.div((data_shape[i+2] + 2*padding[i]-pool_size[i]),strides[i]))

    return te.extern(
        output_shape,
        [x],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cudnn.pooling.forward", ins[0], outs[0],
            1, 0,  # alpha, beta
            2, # MODE: CUDNN_POOLING_AVG_EXCLUDE_PADDING   # 1 -- MODE: CUDNN_POOLING_AVG_INCLUDE_PADDING
            0, # CUDNN_NOT_PROPAGATE_NAN
            pool_size[0], pool_size[1],
            padding[0], padding[1],
            strides[0], strides[1]
        ),
        name="y",
    )



def max_pool2d(x, pool_size, strides, padding, pool_type, ceil_mode, data_layout, count_include_pad):
    dims = 2
    data_shape = x.shape
    output_shape = list(data_shape)
    #outputDim = 1 + (inputDim + 2*padding - windowDim)/poolingStride;
    for i in range(dims):
        output_shape[i+2] = int(1 + tvm.tir.div((data_shape[i+2] + 2*padding[i]-pool_size[i]),strides[i]))

    return te.extern(
        output_shape,
        [x],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cudnn.pooling.forward", ins[0], outs[0],
            1, 0,  # alpha, beta
            3, # MODE: CUDNN_POOLING_MAX_DETERMINISTIC
            0, # CUDNN_NOT_PROPAGATE_NAN
            pool_size[0], pool_size[1],
            padding[0], padding[1],
            strides[0], strides[1]
        ),
        name="y",
    )




def sigmoid(x):
    return te.extern(
        x.shape,
        [x],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cudnn.activation.forward", ins[0], outs[0],
            1, # alpha
            0, # beta
            0, # mode
            0, # nanOpt
            1e100 # Coeff
        ),
        name="y",
    )

def relu(x):
    return te.extern(
        x.shape,
        [x],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cudnn.activation.forward", ins[0], outs[0],
            1, # alpha
            0, # beta
            1, # mode
            0, # nanOpt
            1e100 # Coeff
        ),
        name="y",
    )


def tanh(x):
    return te.extern(
        x.shape,
        [x],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cudnn.activation.forward", ins[0], outs[0],
            1, # alpha
            0, # beta
            2, # mode
            0, # nanOpt
            1e100 # Coeff
        ),
        name="y",
    )






def biasadd(data, bias, axis):
    return te.extern(
        data.shape,
        [bias], #inputs
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cudnn.add", ins[0], outs[0],
            1, 1, axis
        ),
        #out_buffers = [data],
        name="y",
    )



def prepare(x, w, pad, stride, dilation, tensor_format, algo, conv_dtype, groups):
    dims = len(x.shape)
    assert dims in (4, 5)

    conv_dtype = x.dtype if conv_dtype is None else conv_dtype
    pad, stride, dilation, _, _ = _prepare_global_func_params(dims - 2, pad, stride, dilation)

    x_shape = list(x.shape)

    if isinstance(x.shape[0], tvm.tir.expr.IntImm):
        oshape = conv_output_shape(
            tensor_format,
            pad,
            stride,
            dilation,
            x_shape,
            list(w.shape),
            x.dtype,
            conv_dtype,
            groups,
        )
        if algo == -1:
            # For now if we try to call `cudnnFindConvolutionForwardAlgorithm` when
            # using INT8 data type, CuDNN will crash down.
            # On the other hand, CuDNN only support IMPLICIT_PRECOMP_GEMM at NHWC format
            if tensor_format == 1 and conv_dtype == "int32":
                algo = 1
            else:
                algo = conv_find_algo(
                    tensor_format,
                    pad,
                    stride,
                    dilation,
                    list(x.shape),
                    list(w.shape),
                    oshape,
                    x.dtype,
                    conv_dtype,
                    groups,
                )
    else:
        # The dynamic batch size case, pretend this is a single batch
        x_shape[0] = 1
        oshape = conv_output_shape(
            tensor_format,
            pad,
            stride,
            dilation,
            x_shape,
            list(w.shape),
            x.dtype,
            conv_dtype,
            groups,
        )
        oshape[0] = x.shape[0]
        # This picks CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        # It seems this is the fastest among algorithms that are always applicable
        algo = 1

    return dims, conv_dtype, pad, stride, dilation, oshape, algo


def conv2d_relu(x, w, pad, stride, dilation, conv_mode, tensor_format, algo, conv_dtype, activ_mode, nan_prop_mode, actv_coeff, groups=1):

    dims, conv_dtype, pad, stride, dilation, oshape, algo = \
        prepare(x, w, pad, stride, dilation, tensor_format, algo, conv_dtype, groups)

    return te.extern(
                 oshape,
                 [x, w],
                 lambda ins, outs: tvm.tir.call_packed(
                     "tvm.contrib.cudnn.conv2d+activation.forward",
                       conv_mode, # mode: CUDNN_CONVOLUTION
                       tensor_format, # CUDNN_TENSOR_NCHW
                       algo,
                       pad[0], pad[1],
                       stride[0], stride[1],
                       dilation[0], dilation[1],
                       conv_dtype,
                       ins[0], # x
                       ins[1], # w
                       outs[0], # y
                       groups,
                       1,#alphas[0],
                       0,#alphas[1],
                       1,#alphas[0] for z
                       0,
                       activ_mode,
                       nan_prop_mode,
                       actv_coeff
                     ),
                     name="y",
            )

def conv2d_add_relu(x, w, z, b, pad, stride, dilation, conv_mode, tensor_format, algo, conv_dtype, activ_mode, nan_prop_mode, actv_coeff, groups=1):

    dims, conv_dtype, pad, stride, dilation, oshape, algo = \
        prepare(x, w, pad, stride, dilation, tensor_format, algo, conv_dtype, groups)

    return te.extern(
                 oshape,
                 [x, w, z, b],
                 lambda ins, outs: tvm.tir.call_packed(
                     "tvm.contrib.cudnn.conv2d+add+activation.forward",
                       conv_mode, # mode: CUDNN_CONVOLUTION
                       tensor_format, # CUDNN_TENSOR_NCHW
                       algo,
                       pad[0], pad[1],
                       stride[0], stride[1],
                       dilation[0], dilation[1],
                       conv_dtype,
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
                       activ_mode,
                       nan_prop_mode,
                       actv_coeff
                     ),
                     name="y",
            )


def conv2d_bias_relu(x, w, z, b, pad, stride, dilation, conv_mode, tensor_format, algo, conv_dtype, activ_mode, nan_prop_mode, actv_coeff, groups=1):

    dims, conv_dtype, pad, stride, dilation, oshape, algo = \
        prepare(x, w, pad, stride, dilation, tensor_format, algo, conv_dtype, groups)

    return te.extern(
                 oshape,
                 [x, w, z, b],
                 lambda ins, outs: tvm.tir.call_packed(
                     "tvm.contrib.cudnn.conv2d+bias+activation.forward",
                       conv_mode, # mode: CUDNN_CONVOLUTION
                       tensor_format, # CUDNN_TENSOR_NCHW
                       algo,
                       pad[0], pad[1],
                       stride[0], stride[1],
                       dilation[0], dilation[1],
                       conv_dtype,
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
                       activ_mode,
                       nan_prop_mode,
                       actv_coeff
                     ),
                     name="y",
            )


def conv3d_add_relu(x, w, z, b, pad, stride, dilation, conv_mode, tensor_format, algo, conv_dtype, activ_mode, nan_prop_mode, actv_coeff, groups=1):

    dims, conv_dtype, pad, stride, dilation, oshape, algo = \
        prepare(x, w, pad, stride, dilation, tensor_format, algo, conv_dtype, groups)

    return te.extern(
                 oshape,
                 [x, w, z, b],
                 lambda ins, outs: tvm.tir.call_packed(
                     "tvm.contrib.cudnn.conv3d+add+activation.forward",
                       conv_mode, # mode: CUDNN_CONVOLUTION
                       tensor_format, # CUDNN_TENSOR_NCHW
                       algo,
                       pad[0], pad[1], pad[2],
                       stride[0], stride[1], stride[2],
                       dilation[0], dilation[1], dilation[2],
                       conv_dtype,
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
                       activ_mode,
                       nan_prop_mode,
                       actv_coeff
                     ),
                     name="y",
            )
