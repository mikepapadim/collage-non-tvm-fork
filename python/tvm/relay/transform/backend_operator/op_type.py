from enum import Enum, auto
from tvm.relay.dataflow_pattern import *

from .pattern import Pattern
from .utils import get_diamond

# Warning(@Soo): note that we ignore tuplegetitem nodes in TVM Relay,
# because they are only used to extract result of Relay's batch_norm operator

class OpType(Enum):
  # ID, name, depth
  # RESNE(X)T
  ADD = ('add', 1)
  CONV2D = ('conv2d', 1)
  CONV2D_WINOGRAD_WO_WT = ('conv2d_winograd_without_weight_transform', 1)
  RELU = ('relu', 1)
  CONV2D_RELU = ('conv2d+relu', 2)
  CONV2D_WINOGRAD_WO_WT_RELU = ('conv2d_winograd_without_weight_transform+relu', 2)
  CONV2D_ADD_RELU = ('conv2d+add+relu', 3)
  # ADD_RELU = ('add+relu', 2) # This leads to the suboptimal results

  # BERT
  DENSE = ('dense', 1)
  RESHAPE = ('reshape', 1)
  TRANSPOSE = ('transpose', 1)
  BATCH_MATMUL = ('batch_matmul', 1)
  RESHAPE_TRANSPOSE = ('reshape+transpose', 2)
  TRANSPOSE_RESHAPE = ('transpose+reshape', 2)
  DENSE_RELU = ('dense+relu', 2)

  # NASRNN
  TANH = ('tanh', 1)
  SIGMOID = ('sigmoid', 1)
  MULTIPLY = ('multiply', 1)
  TUPLE_GET_ITEM_0 = ('tuple_get_item_0', 1)
  TUPLE_GET_ITEM_1 = ('tuple_get_item_1', 1)
  TUPLE_TWO_IDX = ('tuple_two_idx', 1)
  DENSE_RELU_ADD_SIGMOID = ('dense+relu+add+sigmoid', 4)
  DENSE_RELU_ADD_TANH = ('dense+relu+add+tanh', 4)
  DENSE_RELU_ADD_RELU = ('dense+relu+add+relu', 4)
  MULTIPLY_TANH = ('multiply+tanh', 2)
  RELU_ADD_RELU = ('relu+add+relu', 3)
  ADD_SIGMOID = ('add+sigmoid', 2)
  ADD_TANH = ('add+tanh', 2)

  # NASNET-A
  CONCAT = ('concat', 1)
  BIAS_ADD = ('biasadd', 1)
  AVG_POOL2D = ('avgpool2d', 1)
  MAX_POOL2D = ('maxpool2d', 1)
  TUPLE_FIVE_IDX = ('tuple_five_idx', 1)
  CONV2D_BIAS_ADD_RELU = ('conv2d+biasadd+relu', 3)
  CONV2D_ADD = ('conv2d+add', 2)
  AVG_POOL2D_ADD = ('avgpool2d+add', 2)
  TUPLE_FIVE_IDX_CONCAT = ('tuple_five_idx+concat', 2)

  # Others
  DIAMOND = ('diamond', 6)  # Not sure yet if it works well for DP
  BN = ('bn', 1)
  SOFTMAX = ('softmax', 1)
  BATCH_FLATTEN = ('batchflatten', 1)
  GLOBAL_AVG_POOL2D = ('globalavgpool2d', 1)
  CONV2D_BN = ('conv2d+bn', 2)
  BN_RELU = ('bn+relu', 2)
  CONV2D_BN_RELU = ('conv2d+bn+relu', 3)

  def name(self):
    return self.value[0]

  def depth(self):
    return self.value[1]

# maps op type to pattern representing it
optype_to_pattern = {
  # RESNE(X)T
  OpType.ADD : Pattern(is_op('add')(wildcard(), wildcard())),
  OpType.CONV2D : Pattern(is_op("nn.conv2d")(wildcard(), wildcard())),
  OpType.CONV2D_WINOGRAD_WO_WT : Pattern(is_op("nn.contrib_conv2d_winograd_without_weight_transform")(wildcard(), wildcard())),
  OpType.RELU : Pattern(is_op("nn.relu")(wildcard())),
  OpType.CONV2D_RELU : Pattern(is_op("nn.relu")(is_op("nn.conv2d")(wildcard(), wildcard()))),
  OpType.CONV2D_WINOGRAD_WO_WT_RELU : Pattern(is_op("nn.relu")(is_op("nn.contrib_conv2d_winograd_without_weight_transform")(wildcard(), wildcard()))),
  OpType.CONV2D_ADD_RELU : Pattern(is_op("nn.relu")(is_op("add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard()))),
  # OpType.ADD_RELU : Pattern(is_op("nn.relu")(is_op("add")(wildcard(), wildcard()))),

  # BERT
  OpType.DENSE : Pattern(is_op("nn.dense")(wildcard(), wildcard())),
  OpType.RESHAPE : Pattern(is_op("reshape")(wildcard())),
  OpType.TRANSPOSE : Pattern(is_op("transpose")(wildcard())),
  OpType.BATCH_MATMUL : Pattern(is_op("nn.batch_matmul")(wildcard(),wildcard())),
  OpType.RESHAPE_TRANSPOSE : Pattern(is_op("transpose")(is_op("reshape")(wildcard()))),
  OpType.TRANSPOSE_RESHAPE : Pattern(is_op("reshape")(is_op("transpose")(wildcard()))),
  OpType.DENSE_RELU: Pattern(is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))),

  # NASRNN
  OpType.TANH : Pattern(is_op("tanh")(wildcard())),
  OpType.SIGMOID : Pattern(is_op("sigmoid")(wildcard())),
  OpType.MULTIPLY : Pattern(is_op("multiply")(wildcard(), wildcard())),
  OpType.TUPLE_GET_ITEM_0 : Pattern(is_tuple_get_item(wildcard(), 0)),
  OpType.TUPLE_GET_ITEM_1 : Pattern(is_tuple_get_item(wildcard(), 1)),
  OpType.TUPLE_TWO_IDX : Pattern(is_tuple([wildcard(), wildcard()])),
  OpType.DENSE_RELU_ADD_SIGMOID: Pattern(is_op("sigmoid")(is_op("add")(is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard())),is_constant()))),
  OpType.DENSE_RELU_ADD_TANH: Pattern(is_op("tanh")(is_op("add")(is_constant(), is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))))),
  OpType.DENSE_RELU_ADD_RELU: Pattern(is_op("nn.relu")(is_op("add")(is_constant(), is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))))),
  OpType.MULTIPLY_TANH: Pattern(is_op("tanh")(is_op("multiply")(wildcard(), wildcard()))),
  OpType.RELU_ADD_RELU: Pattern(is_op("nn.relu")(is_op("add")(is_constant(), is_op("nn.relu")(wildcard())))),
  OpType.ADD_SIGMOID: Pattern(is_op("sigmoid")(is_op("add")(wildcard(), wildcard()))),
  OpType.ADD_TANH: Pattern(is_op("tanh")(is_op("add")(wildcard(), wildcard()))),

  # NASNET-A
  OpType.CONCAT : Pattern(is_op("concatenate")(wildcard())),
  OpType.BIAS_ADD : Pattern(is_op("nn.bias_add")(wildcard(), wildcard())),
  OpType.AVG_POOL2D : Pattern(is_op("nn.avg_pool2d")(wildcard())),
  OpType.MAX_POOL2D : Pattern(is_op("nn.max_pool2d")(wildcard())),
  OpType.TUPLE_FIVE_IDX : Pattern(is_tuple([wildcard(), wildcard(), wildcard(), wildcard(), wildcard()])),
  OpType.CONV2D_BIAS_ADD_RELU : Pattern(is_op("nn.relu")(is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), wildcard()), is_constant()))),
  OpType.CONV2D_ADD : Pattern(is_op("add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard())),
  OpType.AVG_POOL2D_ADD : Pattern(is_op("add")(is_op("nn.avg_pool2d")(wildcard()), wildcard())),
  OpType.TUPLE_FIVE_IDX_CONCAT : Pattern(is_op("concatenate")(is_tuple([wildcard(), wildcard(), wildcard(), wildcard(), wildcard()]))),

  # Others
  OpType.DIAMOND : get_diamond(),
  OpType.BN : Pattern(is_tuple_get_item(is_op("nn.batch_norm")(wildcard(), wildcard(), wildcard(), wildcard(), wildcard()), 0)),
  OpType.SOFTMAX : Pattern(is_op("nn.softmax")(wildcard())),
  OpType.BATCH_FLATTEN : Pattern(is_op("nn.batch_flatten")(wildcard())),
  OpType.GLOBAL_AVG_POOL2D : Pattern(is_op("nn.global_avg_pool2d")(wildcard())),

  # Other Fused Ops
  OpType.CONV2D_BN : Pattern(is_tuple_get_item(is_op("nn.batch_norm")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard(), wildcard(), wildcard(), wildcard()), 0)),
  OpType.BN_RELU : Pattern(is_op("nn.relu")(is_tuple_get_item(is_op("nn.batch_norm")(wildcard(), wildcard(), wildcard(), wildcard(), wildcard()), 0))),
  OpType.CONV2D_BN_RELU : Pattern(is_op("nn.relu")(is_tuple_get_item(is_op("nn.batch_norm")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard(), wildcard(), wildcard(), wildcard()), 0))),
}

def add_optype_to_pattern():
  for optype, pat in optype_to_pattern.items():
    pat.set_op_type(optype)

add_optype_to_pattern()


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

  # Others
  "nn.batch_norm" : ["data", "bn_data_gamma", "bn_data_beta", "bn_data_moving_mean", "bn_data_moving_var"],
  "nn.softmax" : ["data"],
  "nn.batch_flatten" : ["data"],
  "nn.global_avg_pool2d" : ["data"],
}

