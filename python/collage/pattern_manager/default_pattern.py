from enum import Enum, auto
from tvm.relay.dataflow_pattern import *

from .pattern_language import Pattern
from .utils import get_diamond

# Warning(@Soo): note that we ignore tuplegetitem nodes in TVM Relay,
# because they are only used to extract result of Relay's batch_norm operator

# maps op type to pattern representing it
optype_to_pattern = {
  # RESNE(X)T
  "ADD" : Pattern(is_op('add')(wildcard(), wildcard())),
  "CONV2D" : Pattern(is_op("nn.conv2d")(wildcard(), wildcard())),
  "CONV2D_WINOGRAD_WO_WT" : Pattern(is_op("nn.contrib_conv2d_winograd_without_weight_transform")(wildcard(), wildcard())),
  "RELU" : Pattern(is_op("nn.relu")(wildcard())),
  "CONV2D_RELU" : Pattern(is_op("nn.relu")(is_op("nn.conv2d")(wildcard(), wildcard()))),
  "CONV2D_WINOGRAD_WO_WT_RELU" : Pattern(is_op("nn.relu")(is_op("nn.contrib_conv2d_winograd_without_weight_transform")(wildcard(), wildcard()))),
  "CONV2D_ADD_RELU" : Pattern(is_op("nn.relu")(is_op("add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard()))),
  # "ADD_RELU" : Pattern(is_op("nn.relu")(is_op("add")(wildcard(), wildcard()))),

  # BERT
  "DENSE" : Pattern(is_op("nn.dense")(wildcard(), wildcard())),
  "RESHAPE" : Pattern(is_op("reshape")(wildcard())),
  "TRANSPOSE" : Pattern(is_op("transpose")(wildcard())),
  "BATCH_MATMUL" : Pattern(is_op("nn.batch_matmul")(wildcard(),wildcard())),
  "RESHAPE_TRANSPOSE" : Pattern(is_op("transpose")(is_op("reshape")(wildcard()))),
  "TRANSPOSE_RESHAPE" : Pattern(is_op("reshape")(is_op("transpose")(wildcard()))),
  "DENSE_RELU": Pattern(is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))),

  # NASRNN
  "TANH" : Pattern(is_op("tanh")(wildcard())),
  "SIGMOID" : Pattern(is_op("sigmoid")(wildcard())),
  "MULTIPLY" : Pattern(is_op("multiply")(wildcard(), wildcard())),
  "TUPLE_GET_ITEM_0" : Pattern(is_tuple_get_item(wildcard(), 0)),
  "TUPLE_GET_ITEM_1" : Pattern(is_tuple_get_item(wildcard(), 1)),
  "TUPLE_TWO_IDX" : Pattern(is_tuple([wildcard(), wildcard()])),
  "DENSE_RELU_ADD_SIGMOID" : Pattern(is_op("sigmoid")(is_op("add")(is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard())),is_constant()))),
  "DENSE_RELU_ADD_TANH": Pattern(is_op("tanh")(is_op("add")(is_constant(), is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))))),
  "DENSE_RELU_ADD_RELU": Pattern(is_op("nn.relu")(is_op("add")(is_constant(), is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))))),
  "MULTIPLY_TANH": Pattern(is_op("tanh")(is_op("multiply")(wildcard(), wildcard()))),
  "RELU_ADD_RELU": Pattern(is_op("nn.relu")(is_op("add")(is_constant(), is_op("nn.relu")(wildcard())))),
  "ADD_SIGMOID": Pattern(is_op("sigmoid")(is_op("add")(wildcard(), wildcard()))),
  "ADD_TANH": Pattern(is_op("tanh")(is_op("add")(wildcard(), wildcard()))),

  # NASNET-A
  "CONCAT" : Pattern(is_op("concatenate")(wildcard())),
  "BIAS_ADD" : Pattern(is_op("nn.bias_add")(wildcard(), wildcard())),
  "AVG_POOL2D" : Pattern(is_op("nn.avg_pool2d")(wildcard())),
  "MAX_POOL2D" : Pattern(is_op("nn.max_pool2d")(wildcard())),
  "TUPLE_FIVE_IDX" : Pattern(is_tuple([wildcard(), wildcard(), wildcard(), wildcard(), wildcard()])),
  "CONV2D_BIAS_RELU" : Pattern(is_op("nn.relu")(is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), wildcard()), is_constant()))),
  "CONV2D_ADD" : Pattern(is_op("add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard())),
  "AVG_POOL2D_ADD" : Pattern(is_op("add")(is_op("nn.avg_pool2d")(wildcard()), wildcard())),
  "TUPLE_FIVE_IDX_CONCAT" : Pattern(is_op("concatenate")(is_tuple([wildcard(), wildcard(), wildcard(), wildcard(), wildcard()]))),

  # ResNet-3D
  "CONV3D": Pattern(is_op("nn.conv3d")(wildcard(), wildcard())),
  "CONV3D_RELU": Pattern(is_op("nn.relu")(is_op("nn.conv3d")(wildcard(), wildcard()))),
  "CONV3D_ADD" : Pattern(is_op("add")(is_op("nn.conv3d")(wildcard(), wildcard()), wildcard())),
  "CONV3D_ADD_RELU": Pattern(is_op("nn.relu")(is_op("add")(is_op("nn.conv3d")(wildcard(), wildcard()), wildcard()))),

  # Others
  "DIAMOND" : get_diamond(),
  "BATCHNORM" : Pattern(is_tuple_get_item(is_op("nn.batch_norm")(wildcard(), wildcard(), wildcard(), wildcard(), wildcard()), 0)),
  "SOFTMAX" : Pattern(is_op("nn.softmax")(wildcard())),
  "BATCH_FLATTEN" : Pattern(is_op("nn.batch_flatten")(wildcard())),
  "GLOBAL_AVG_POOL2D" : Pattern(is_op("nn.global_avg_pool2d")(wildcard())),
  "CONV3D_BIAS_RELU" : Pattern(is_op("nn.relu")(is_op("nn.bias_add")(is_op("nn.conv3d")(wildcard(), wildcard()), is_constant()))),

  # Other Fused Ops
  "CONV2D_BN": Pattern(is_tuple_get_item(is_op("nn.batch_norm")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard(), wildcard(), wildcard(), wildcard()), 0)),
  "BN_RELU" : Pattern(is_op("nn.relu")(is_tuple_get_item(is_op("nn.batch_norm")(wildcard(), wildcard(), wildcard(), wildcard(), wildcard()), 0))),
  "CONV2D_BN_RELU" : Pattern(is_op("nn.relu")(is_tuple_get_item(is_op("nn.batch_norm")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard(), wildcard(), wildcard(), wildcard()), 0))),
  "SUBTRACT" : Pattern(is_op("subtract")(wildcard(), wildcard())),
}


# maps relay operator type to names of input vars.
relayop_to_varnames = {
  # RESNE(X)T
  "add" : ["data", "data"],
  "nn.conv2d" : ["data", "weight"],
  "nn.contrib_conv2d_winograd_without_weight_transform" : ["data", "weight"],
  "nn.relu": ["data"],

  # BERT
  "nn.dense" : ["data", "weight"],
  "reshape": ["data"],
  "transpose": ["data"],
  "nn.batch_matmul" : ["data", "data"],
  #"nn.batch_matmul" : ["x", "y"],

  # NASRNN
  "tanh": ["data"],
  "multiply": ["data", "data"],
  # "multiply": ["lhs", "rhs"],
  "sigmoid": ["data"],
  # FIXME(@Soo): How should we deal with TUPLE and TUPLE_GET_ITEM?

  # NASNET-A
  "concatenate": ["data"],
  "nn.bias_add" : ["data", "bias"],
  "nn.avg_pool2d" : ["data"],
  "nn.max_pool2d" : ["data"],
  "tuple" : ["data", "data", "data", "data", "data"],

  # RESNET_3D
  "nn.conv3d": ["data", "weight"],

  # Others
  "nn.batch_norm" : ["data", "bn_data_gamma", "bn_data_beta", "bn_data_moving_mean", "bn_data_moving_var"],
  "nn.softmax" : ["data"],
  "nn.batch_flatten" : ["data"],
  "nn.global_avg_pool2d" : ["data"],

  # DCGAN
  "image.resize": ["data"],

  # BERT_FULL
  "divide": ["data", "data"],
  "subtract": ["data", "data"],
  "sqrt": ["data"],
  "variance": ["data", "data"],
  "mean": ["data"],
}

