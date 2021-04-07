from enum import Enum, auto
from tvm.relay.dataflow_pattern import *

from .pattern import Pattern
from .utils import get_diamond

# note that we ignore tuplegetitem nodes in TVM Relay, 
# because they are only used to extract result of Relay's batch_norm operator
class OpType(Enum):
  # ID, name, depth
  ADD = (0, 'add', 1)
  CONV2D = (1, 'conv2d', 1)
  BN = (2, 'bn', 1)
  RELU = (3, 'relu', 1,)
  SOFTMAX = (4, 'softmax', 1)
  BIAS_ADD = (5, 'biasadd', 1)
  DENSE = (6, 'dense', 1)
  BATCH_FLATTEN = (7, 'batchflatten', 1)
  GLOBAL_AVG_POOL2D = (8, 'globalavgpool2d', 1)
  MAX_POOL2D = (9, 'maxpool2d', 1)
  CONV2D_BN = (10, 'conv2d+bn', 2)
  BN_RELU = (11, 'bn+relu', 2)
  CONV2D_BN_RELU = (12, 'conv2d+bn+relu', 3)
  DIAMOND = (13, 'diamond', 6) # Not sure yet if it works well for DP
  CONV2D_BIAS_ADD_RELU = (14, 'conv2d+biasadd+relu', 3)


  def identifier(self):
    return self.value[0]

  def name(self):
    return self.value[1]

  def depth(self):
    return self.value[2]

# maps op type to pattern representing it
optype_to_pattern = {
  OpType.ADD : Pattern(is_op('add')(wildcard(), wildcard())),
  OpType.CONV2D : Pattern(is_op("nn.conv2d")(wildcard(), wildcard())),
  OpType.BN : Pattern(is_tuple_get_item(is_op("nn.batch_norm")(wildcard(), wildcard(), wildcard(), wildcard(), wildcard()), 0)),
  OpType.RELU : Pattern(is_op("nn.relu")(wildcard())),
  OpType.SOFTMAX : Pattern(is_op("nn.softmax")(wildcard())),
  OpType.BIAS_ADD : Pattern(is_op("nn.bias_add")(wildcard(), wildcard())),
  OpType.DENSE : Pattern(is_op("nn.dense")(wildcard(), wildcard())),
  OpType.BATCH_FLATTEN : Pattern(is_op("nn.batch_flatten")(wildcard())),
  OpType.GLOBAL_AVG_POOL2D : Pattern(is_op("nn.global_avg_pool2d")(wildcard())),
  OpType.MAX_POOL2D : Pattern(is_op("nn.max_pool2d")(wildcard())),
  OpType.CONV2D_BN : Pattern(is_tuple_get_item(is_op("nn.batch_norm")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard(), wildcard(), wildcard(), wildcard()), 0)),
  OpType.BN_RELU : Pattern(is_op("nn.relu")(is_tuple_get_item(is_op("nn.batch_norm")(wildcard(), wildcard(), wildcard(), wildcard(), wildcard()), 0))),
  OpType.CONV2D_BN_RELU : Pattern(is_op("nn.relu")(is_tuple_get_item(is_op("nn.batch_norm")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard(), wildcard(), wildcard(), wildcard()), 0))),
  OpType.DIAMOND : get_diamond(),
  OpType.CONV2D_BIAS_ADD_RELU : Pattern(is_op("nn.relu")(is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard()))),
}

# maps relay operator type to names of input vars. 
relayop_to_varnames = {
  "add" : ["data", "data"],
  "nn.conv2d" : ["data", "weight"],
  "nn.batch_norm" : ["data", "bn_data_gamma", "bn_data_beta", "bn_data_moving_mean", "bn_data_moving_var"],
  "nn.relu" : ["data"],
  "nn.softmax" : ["data"],
  "nn.bias_add" : ["data", "bias"],
  "nn.dense" : ["data", "weight"],
  "nn.batch_flatten" : ["data"],
  "nn.global_avg_pool2d" : ["data"],
  "nn.max_pool2d" : ["data"],
}

