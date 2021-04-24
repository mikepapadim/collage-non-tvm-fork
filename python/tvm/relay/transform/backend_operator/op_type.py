from enum import Enum, auto
from tvm.relay.dataflow_pattern import *

from .pattern import Pattern
from .utils import get_diamond

# note that we ignore tuplegetitem nodes in TVM Relay, 
# because they are only used to extract result of Relay's batch_norm operator
class OpType(Enum):
  # ID, name, depth
  # RESNE(X)T
  ADD = (0, 'add', 1)
  CONV2D = (1, 'conv2d', 1)
  RELU = (2, 'relu', 1,)
  CONV2D_RELU = (3, 'conv2d+relu', 2)

  # BERT
  DENSE = (4, 'dense', 1)
  RESHAPE = (5, 'reshape', 1)
  TRANSPOSE = (6, 'transpose', 1)
  BATCH_MATMUL = (7, 'batch_matmul', 1)

  # NASRNN
  TANH = (8, 'tanh', 1)
  SIGMOID = (9, 'sigmoid', 1)
  MULTIPLY = (10, 'multiply', 1)
  TUPLE_GET_ITEM_0 = (11, 'tuple_get_item_0', 1)
  TUPLE_GET_ITEM_1 = (12, 'tuple_get_item_1', 1)
  TUPLE_TWO_IDX = (13, 'tuple_two_idx', 1)

  # NASNET-A
  CONCAT = (14, 'concat', 1)
  BIAS_ADD = (15, 'biasadd', 1)
  AVG_POOL2D = (16, 'avgpool2d', 1)
  MAX_POOL2D = (17, 'maxpool2d', 1)
  TUPLE_FIVE_IDX = (27, 'tuple_five_idx', 1)

  # Others
  DIAMOND = (18, 'diamond', 6)  # Not sure yet if it works well for DP
  BN = (19, 'bn', 1)
  SOFTMAX = (20, 'softmax', 1)
  BATCH_FLATTEN = (21, 'batchflatten', 1)
  GLOBAL_AVG_POOL2D = (22, 'globalavgpool2d', 1)
  CONV2D_BN = (23, 'conv2d+bn', 2)
  BN_RELU = (24, 'bn+relu', 2)
  CONV2D_BN_RELU = (25, 'conv2d+bn+relu', 3)
  CONV2D_BIAS_ADD_RELU = (26, 'conv2d+biasadd+relu', 3)

  def identifier(self):
    return self.value[0]

  def name(self):
    return self.value[1]

  def depth(self):
    return self.value[2]

# maps op type to pattern representing it
optype_to_pattern = {
  # RESNE(X)T
  OpType.ADD : Pattern(is_op('add')(wildcard(), wildcard())),
  OpType.CONV2D : Pattern(is_op("nn.conv2d")(wildcard(), wildcard())),
  OpType.RELU : Pattern(is_op("nn.relu")(wildcard())),
  OpType.CONV2D_RELU : Pattern(is_op("nn.relu")(is_op("nn.conv2d")(wildcard(), wildcard()))),

  # BERT
  OpType.DENSE : Pattern(is_op("nn.dense")(wildcard(), wildcard())),
  OpType.RESHAPE : Pattern(is_op("reshape")(wildcard())),
  OpType.TRANSPOSE : Pattern(is_op("transpose")(wildcard())),
  OpType.BATCH_MATMUL : Pattern(is_op("nn.batch_matmul")(wildcard(),wildcard())),

  # NASRNN
  OpType.TANH : Pattern(is_op("tanh")(wildcard())),
  OpType.SIGMOID : Pattern(is_op("sigmoid")(wildcard())),
  OpType.MULTIPLY : Pattern(is_op("multiply")(wildcard(), wildcard())),
  OpType.TUPLE_GET_ITEM_0 : Pattern(is_tuple_get_item(wildcard(), 0)),
  OpType.TUPLE_GET_ITEM_1 : Pattern(is_tuple_get_item(wildcard(), 1)),
  OpType.TUPLE_TWO_IDX : Pattern(is_tuple([wildcard(), wildcard()])),

  # NASNET-A
  OpType.CONCAT : Pattern(is_op("concatenate")(wildcard())),
  OpType.BIAS_ADD : Pattern(is_op("nn.bias_add")(wildcard(), wildcard())),
  OpType.AVG_POOL2D : Pattern(is_op("nn.avg_pool2d")(wildcard())),
  OpType.MAX_POOL2D : Pattern(is_op("nn.max_pool2d")(wildcard())),
  OpType.TUPLE_FIVE_IDX : Pattern(is_tuple([wildcard(), wildcard(), wildcard(), wildcard(), wildcard()])),

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
  OpType.CONV2D_BIAS_ADD_RELU : Pattern(is_op("nn.relu")(is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard()))),
}

# maps relay operator type to names of input vars. 
relayop_to_varnames = {
  # RESNE(X)T
  "add" : ["data", "data"],
  "nn.conv2d" : ["data", "weight"],
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

