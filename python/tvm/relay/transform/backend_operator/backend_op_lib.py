from tvm import relay
from collections import defaultdict

from .backend_op import BackendOp, get_optimal_backendop
from .op_config import Config, MeasuredConfigs

#from ..workloads.onnx_workloads import get_network_from_onnx
from .utils import no_constraints_func
from .target import Target
from .op_type import OpType

from tvm.relay.dataflow_pattern import *

def add_all_backend_ops_to_lib(b_op_lib, target, exclued_ops=[OpType.DIAMOND]):
  t_name = target.name()

  for op_type in OpType:
    # Skip diamond pattern for now
    if op_type in exclued_ops:# or op_type == OpType.CONV2D_BIAS_ADD_RELU:
      continue

    op_name, op_depth = op_type.name(), op_type.depth()
    b_op_lib._add_backendop(f"{t_name}_{op_name}", target, op_type, op_depth)


def add_all_backend_ops_to_lib_except_fused(b_op_lib, target):
  t_name = target.name()
  op_to_skip = [OpType.DIAMOND, OpType.ADD]  # OpType.CONV2D_BN, OpType.CONV2D_BN_RELU,
  # OpType.BN_RELU, OpType.CONV2D_BIAS_ADD_RELU

  for op_type in OpType:
    # Skip diamond pattern for now
    if op_type in op_to_skip:
      continue

    op_name, op_depth = op_type.name(), op_type.depth()
    b_op_lib._add_backendop(f"{t_name}_{op_name}", target, op_type, op_depth)

class BackendOpCostEvaluator:
  __instance = None

  @staticmethod
  def get():
    """ Static access method. """
    if BackendOpCostEvaluator.__instance == None:
      BackendOpCostEvaluator()
    return BackendOpCostEvaluator.__instance

  def __init__(self):
    """ Virtually private constructor. """
    if BackendOpCostEvaluator.__instance != None:
      raise Exception("This class is a singleton!")

    BackendOpCostEvaluator.__instance = self

  def log_backend_op_perf(self, b_op_lib, expr, target):
    assert type(target) != list

    for pattern in b_op_lib.get_all_patterns():
      if pattern.get_pattern().match(expr):
        print("PATTERN:\n", pattern.get_pattern())
        res = get_optimal_backendop(b_op_lib, expr, pattern, [target])
        if res == None:
          print("No satisfying backend operators")
        else:
          op, cost = res
          print("best backendop: %s, cost: %.5f ms" % (op, cost))


  # traverse all subgraphs of a computation graph and evaluate all matchings between backend operators and subgraphs
  def log_network_backend_ops_perf_on_target(self, b_op_lib, target, network_expr, batch_size=1):
    # Read from ONNX is deprecated because type information is not available.
    # mod, _, _, _ = get_network_from_onnx(network_name, batch_size=batch_size)
    # relay.analysis.post_order_visit(mod['main'], lambda expr: self.log_backend_op_perf(b_op_lib, expr, target))
    relay.analysis.post_order_visit(network_expr, lambda expr: self.log_backend_op_perf(b_op_lib, expr, target))




#@Sung: base class for pattern_rule
class BasePatternEngine:
  # OpKind, Comp graph, rule

  # NOTE: Ideall, OpKind should be defined by user depending on their pattern strategy. However, as we will only support TVM pattern rules and use the relay attribute of OpKind, we are not going to define it for now.

  def __init__(self, _target,  _dictOpKinds=dict(), _pattern_rules=None):
      self.target = _target
      self.dictOpKinds = _dictOpKinds
      self.fgen = _pattern_rules
      self.id = 0 # pattern id

  def _register_op_kind(self, _op, _kind):
      self.dictOpKinds[_op] = _kind

  def _register_op_dict(self, _dict):
      self.dictOpKinds = _dict

  def _register_pattern_rule(self, _pattern_rules):
      self.fgen = _pattern_rules

  def run(self, dom_tree, expr):
      # def tvm_pattern_rule(dom_tree, op_dict, expr):
      self.fgen(dom_tree, self.dictOpKinds, expr)
      pass

# library class (singleton) representing all backend operators
class BackendOpLib(object):
  __instance = None

  @staticmethod
  def get():
    """ Static access method. """
    if BackendOpLib.__instance == None:
      BackendOpLib()
    return BackendOpLib.__instance

  def __init__(self):
    """ Virtually private constructor. """
    if BackendOpLib.__instance != None:
      raise Exception("This class is a singleton!")

    # list of all backend operators
    self._measured_configs = MeasuredConfigs()
    self._measured_configs.load_from_log()

    self.all_backendops = []
    self.all_pattern_engines = []
    # dictionary that maps each pattern to list of backend ops represented by the pattern
    self.pattern_to_backendops = defaultdict(list)

    self._add_all_backendops()

    BackendOpLib.__instance = self



  #@Sung: Naming is a bit confusing
  # add->register? b_op_lib->b_op_name?
  def _add_backend_pattern_rule(self, f_engine):
      self.all_pattern_engines.append(f_engine)


  # Note that we only support ResNet50 for now
  def _add_all_backendops(self):
    # CUDNN
    # FIXME(@Soo): For ResNext, some of CUDNN convolution doesn't work.
    self._add_backendop("cudnn_conv2d", Target.CUDNN, OpType.CONV2D, 1)
    # self._add_backendop("cudnn_conv2d+relu", Target.CUDNN, OpType.CONV2D_RELU, 2)
    # self._add_backendop("cudnn_relu", Target.CUDNN, OpType.RELU, 1)
    # self._add_backendop("cudnn_biasadd", Target.CUDNN, OpType.BIAS_ADD, 1)

    # Not implemented for recording
    # self._add_backendop("cudnn_add", Target.CUDNN, OpType.ADD, 1)

    # self._add_backendop("cudnn_softmax", Target.CUDNN, OpType.SOFTMAX, 1)
    # self._add_backendop("cudnn_bn", Target.CUDNN, OpType.BN, 1)
    # measure_cost doesn't work, we need to fix this later.
    # self._add_backendop("cudnn_maxpool2d", Target.CUDNN, OpType.MAX_POOL2D, 1)
    # conv_bias_add_relu --> ResNet doesn't have this pattern, so it wouldn't be measured
    # self._add_backendop("cudnn_conv2d+biasadd+relu", Target.CUDNN, OpType.CONV2D_BIAS_ADD_RELU, 3)

    # TENSORRT
    add_all_backend_ops_to_lib(self, Target.TENSORRT, [OpType.DIAMOND, OpType.TRANSPOSE,
                                                       OpType.TUPLE_TWO_IDX, OpType.TUPLE_FIVE_IDX,
                                                       OpType.TUPLE_FIVE_IDX_CONCAT, OpType.TUPLE_GET_ITEM_0,
                                                       OpType.TUPLE_GET_ITEM_1,
                                                       OpType.BATCH_MATMUL, OpType.RESHAPE_TRANSPOSE,
                                                       OpType.TRANSPOSE_RESHAPE])

    # CUBLAS
    # TODO: Add patterns. matmul, batch matmul
    self._add_backendop("cublas_dense", Target.CUBLAS, OpType.DENSE, 1)
    self._add_backendop("cublas_batch_matmul", Target.CUBLAS, OpType.BATCH_MATMUL, 1)


    # @Sung: add TVM pattern rule
    def tvm_pattern_rule(dom_tree, op_dict, expr):
        # NOTE: Two possible choices
        # 1. Use dataflow pattern matcher in TVM.
        # e.g., op = is_op('nn.dense').has_attr({"TOpPattern": K_ELEMWISE}) in tests/python/relay/test_dataflow_pattern.py
        #       wildcard().has_attr({"TOpPattern": K_ELEMWISE})
        #   # Match call with any number of inputs
        #     call_pattern = wildcard()(None)
        #
        #  pat_is_relu = is_op("nn.relu")(None)
        #  pat_is_elemwise = wildcard().has_attr({"TOpPattern": op_dict["kElemWise"]})(None)
        #  assert(pat_is_elemwise.match(expr))
        #
        # def test_AttrPattern():
        #     op = is_op("add").has_attr({"TOpPattern": K_ELEMWISE})
        #           assert isinstance(op, AttrPattern)
        #           assert op.attrs["TOpPattern"] == K_ELEMWISE
        #
        #
        # 2. Manual implementation e.g., expr.op.get_attr("TOpPattern")
        #

        def get_op_pattern(expr):
            return expr.op.get_attr("TOpPattern")


        # Try expension. If valid, create a pattern and try further.
        def run_fuse(expr, NUM_MAX_OP = 256):
            if cur_op_type == op_dict["kOpaque"] or cur_num_op > NUM_MAX_OP:
                return None

            #tvm.relay.analysis.post_order_visit

            #// no actions needed if the current node have no dominator
            #                       if (dom_node->parent == nullptr) continue;


            # NOTE: Do not fuse tuples unitl phase 2.
            # // Do not fuse into tuple for now e.g.,   if (groups_[dom_parent_gindex]->pattern == kTuple) continue;

            # Phase 0:
            #    Path for OutEWiseFusable: conv2d --> Fuse following elem-wise ops
            #    if (group_node->pattern == kOutEWiseFusable) {}

            #    else if group_node->pattern <= kBroadcast:
            #        Pre-condition: can only be fused to parent which is injective or reduction.
            #     if (dom_node->parent != nullptr &&
            #               (dom_node->pattern <= kInjective || dom_node->pattern == kCommReduce))

            # Phase 1: group_node->pattern == kInjective || group_node->pattern == kTuple)
            # Phase 2:  Fuse injective ops into intermediate tuples, if any

        # class Config(object) in python/tvm/relay/transform/backend_operator/op_config.py

        # single op
        num_op = 1
        cur_op_type = get_op_pattern(expr)

        # Register?

        #run_fuse()




    # defined at include/tvm/relay/op_attr_types.h
    tvm_op_dict = {"kElemWise":0, "kBroadcast":1, "kInjective":2, "kCommReduce":3, "kOutEWiseFusable":4, "kTuple":7, "kOpaque":8}

    tvm_targets = [Target.TVM_GPU_AUTOTVM, Target.TVM_GPU_AUTOSCH, Target.GPU_NO_TUNING]
    for tvm_target in tvm_targets:
        tvm_pattern_engine = BasePatternEngine(tvm_target, tvm_op_dict, tvm_pattern_rule)
        self._add_backend_pattern_rule(tvm_pattern_engine)

    # TVM_GPU
    #add_all_backend_ops_to_lib(self, Target.TVM_GPU_AUTOSCH)
    #add_all_backend_ops_to_lib(self, Target.TVM_GPU_AUTOTVM)
    # add_all_backend_ops_to_lib_except_fused(backendop_lib, Target.TVM_GPU)

    # TVM_GPU_NO_TUNING
    #add_all_backend_ops_to_lib(self, Target.TVM_GPU_NO_TUNING)
    # add_all_backend_ops_to_lib_except_fused(backendop_lib, Target.TVM_GPU_NO_TUNING)

    # TVM_CPU; Exclude it for GPU testing
    # Fix: Extend this to automatically select backend library based on HW info
    # add_all_backend_ops_to_lib(backendop_lib, Target.TVM_CPU)

  # add a backend operator to the library
  def _add_backendop(self, name, target, op_type, max_depth, constraint_func = no_constraints_func):
    backendop = BackendOp(name, target, op_type, max_depth, self._measured_configs, constraint_func)
    self.all_backendops.append(backendop)
    self.pattern_to_backendops[backendop.get_pattern()].append(backendop)

  # return list of backend operators matching a pattern

  def measure_backend_ops(self, network_expr, targets, batch_size):
    assert type(targets) == list

    for target in targets:
      BackendOpCostEvaluator.get().log_network_backend_ops_perf_on_target(self, target, network_expr, batch_size)

  # return list of backend operators matching a pattern
  def get_backendops(self, pattern):
    return self.pattern_to_backendops[pattern]

  # return list of all patterns for backend operators
  def get_all_patterns(self):
    return list(self.pattern_to_backendops.keys())

  # @Sung
  # return list of all pattern rules for backend library
  def get_all_pattern_engines(self):
      return self.all_pattern_engines

  # save newly measured op perfs to the log
  def save_to_log(self):
    return self._measured_configs.save_to_log()
