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
# pylint: disable=invalid-name, unused-argument
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import tvm.ir
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

from tvm.relay.expr import Call, Constant, Tuple, GlobalVar, Var, TupleGetItem

def check_dynamism(args, op_name):
    """
    Check for dynamism inside any of the args in the op.

    Parameters
    ----------
    args : tvm.ir.container.Array
        Arguments of the op. Each of the argument shape is checked for presence of dynamic
        components.
    op_name: str
        Name of the op for debugging purposes only.
    Returns
    ----------
    ret : bool
        True if dynamism is present, False otherwise
    """
    for arg in args:
        if isinstance(arg, (Call, Var, Constant, TupleGetItem)):
            for dim_shape in arg.checked_type.shape[1:]:
                if isinstance(dim_shape, tvm.tir.expr.Any):
                    return True
        elif isinstance(arg, Tuple):
            return check_dynamism(arg.fields, op_name)
        else:
            logger.info(
                "Arg not supported in DNNL for %s with type %s",
                op_name,
                type(arg),
            )
            return True
    return False



def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


def _register_external_op_helper_with_checker(op_name, checker):
    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
        attrs, args = expr.attrs, expr.args
        # ops with dynamic shapes are offloaded to VM
        if check_dynamism(args, op_name):
            return False
        if any([x.checked_type.dtype != "float32" for x in args]):
            logger.info("Only float32 inputs are supported for DNNL.")
            return False

        return checker(attrs, args, op_name)

    return _func_wrapper


#_register_external_op_helper("add")
#_register_external_op_helper("nn.relu")

_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.conv3d")
_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.dense")
#_register_external_op_helper("subtract")
#_register_external_op_helper("multiply")


def dim_check_fn(attrs, args, op_name):
    shapes = [[int(x) if not isinstance(x, tvm.tir.expr.Any) else -1 for x in arg.checked_type.shape]   for arg in args]
    if op_name == 'add':
        for shape in shapes:
            if len(shape) < 2:
                return False

            # @sunggg: Temp solution. Selectively disable adds in NasNetA
            if shape == [1, 64, 56, 56] or shape == [1,128,28,28] or shape == [1, 256, 14, 14]:
                return False


    elif op_name == 'nn.relu':
        for shape in shapes:
            if len(shape)!=4:
                return False
    else:
        raise Exception(f"Unsupported op for dim_check_fn {op_name}")

    return True


def group_conv2d_fn(attrs, args, op_name):
    return attrs.groups == 1

_register_external_op_helper_with_checker("add", dim_check_fn)
_register_external_op_helper_with_checker("nn.relu", dim_check_fn)


def make_pattern(with_bias=True):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    return is_op("nn.relu")(conv_out)


@register_pattern_table("dnnl")
def pattern_table():
    conv2d_bias_relu_pat = ("dnnl.conv2d_bias_relu", make_pattern(with_bias=True))
    conv2d_relu_pat = ("dnnl.conv2d_relu", make_pattern(with_bias=False))
    dnnl_patterns = [conv2d_bias_relu_pat, conv2d_relu_pat]
    return dnnl_patterns



