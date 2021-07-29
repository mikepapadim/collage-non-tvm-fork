from tvm import relay
from collections import defaultdict

from .backend_op import BackendOp, get_optimal_backendop
from .op_config import Config, MeasuredConfigs

#from ..workloads.onnx_workloads import get_network_from_onnx
from .utils import no_constraints_func
from .target import Target
from .op_type import optype_to_pattern
from .pattern import Pattern

from tvm.relay.dataflow_pattern import *
from tvm.relay.transform.backend_operator.utils import *

def add_all_backend_ops_to_lib(b_op_lib, target, exclued_ops=["DIAMOND"]):
  t_name = target.name()

  for op_type, pattern in optype_to_pattern.items():
    # Skip diamond pattern for now
    if op_type in exclued_ops:
      continue
    b_op_lib._add_backendop(target, pattern)


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

  def log_backend_op_perf(self, b_op_lib, expr, target, hw_name):
    assert type(target) != list

    for pattern in b_op_lib.get_all_patterns():
      if pattern.get_pattern().match(expr):
        print("PATTERN:\n", pattern.get_pattern())
        res = get_optimal_backendop(b_op_lib, expr, pattern, [target], hw_name)
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
class BasePatternGenerator:
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
  def get(hw_name):
    """ Static access method. """
    if BackendOpLib.__instance == None:
      BackendOpLib(hw_name)
    return BackendOpLib.__instance

  def __init__(self, hw_name):
    """ Virtually private constructor. """
    if BackendOpLib.__instance != None:
      raise Exception("This class is a singleton!")

    # list of all backend operators
    self._measured_configs = MeasuredConfigs()
    self._measured_configs.load_from_log(hw_name)

    self.all_backendops = set()
    self.all_pattern_generators = []
    # dictionary that maps each pattern to list of backend ops represented by the pattern
    self.pattern_to_backendops = defaultdict(set)

    self._add_all_backendops()

    BackendOpLib.__instance = self


  #@Sung: Naming is a bit confusing
  # add->register? b_op_lib->b_op_name?
  def _add_backend_pattern_rule(self, f_generator):
      self.all_pattern_generators.append(f_generator)


  # Note that we only support ResNet50 for now
  def _add_all_backendops(self):
    # CUDNN
    # FIXME(@Soo): For ResNext, some of CUDNN convolution doesn't work.
    #self._add_backendop_with_key(Target.CUDNN, "CONV2D")
    # self._add_backendop_with_key(Target.CUDNN, "CONV2D_RELU")
    # self._add_backendop_with_key(Target.CUDNN, "RELU")
    # self._add_backendop_with_key(Target.CUDNN, "BIAS_ADD")

    # Not implemented for recording
    # self._add_backendop_with_key(Target.CUDNN, "ADD")

    # self._add_backendop_with_key(Target.CUDNN, "SOFTMAX")
    # self._add_backendop_with_key(Target.CUDNN, "BN")
    # measure_cost doesn't work, we need to fix this later.
    # self._add_backendop_with_key(Target.CUDNN, "MAX_POOL2D")
    # conv_bias_add_relu --> ResNet doesn't have this pattern, so it wouldn't be measured
    # self._add_backendop_with_key(Target.CUDNN, "CONV2D_BIAS_ADD_RELU")

    # TENSORRT
    add_all_backend_ops_to_lib(self, Target.TENSORRT, ["DIAMOND", "TRANSPOSE",
                                                       # "TUPLE_TWO_IDX", "TUPLE_FIVE_IDX",
                                                       # "TUPLE_FIVE_IDX_CONCAT",
                                                       # "TUPLE_GET_ITEM_0",
                                                       # "TUPLE_GET_ITEM_1",
                                                       "BATCH_MATMUL",
                                                       "RESHAPE_TRANSPOSE",
                                                       "TRANSPOSE_RESHAPE"])

    # CUBLAS
    # TODO: Add patterns. matmul, batch matmul
    #self._add_backendop_with_key(Target.CUBLAS, "DENSE")
    #self._add_backendop_with_key(Target.CUBLAS, "BATCH_MATMUL")


    # @Sung: add TVM pattern rule
    def tvm_pattern_rule(dom_tree, op_dict, expr, target=Target.TVM_GPU_AUTOTVM):
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

        # TODO:
        # Name of pattern: Serialization
        #   - post-order traversal + mark Var/Const with a special symbol $
        # Depth?
        # Relay pattern gen:

        def get_op_pattern(expr):
            return expr.op.get_attr("TOpPattern")

        # Try expension. If valid, create a pattern and try further.
        def run_fuse(src, sink, cur_pattern_type = None, NUM_MAX_OP = 256):
            if src is None:
                # Hanlde single op
                cur_pattern_type = get_op_pattern(sink)
                op_name = sink.op.name

                args = [wildcard()]*len(sink.args)
                cur_relay_pattern = is_op(op_name)(*args)
                pattern_name = op_name
            else:
                if cur_op_type == op_dict["kOpaque"] or cur_num_op > NUM_MAX_OP:
                    return None

            # Register
            self._add_backendop(target, Pattern(cur_relay_pattern))


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

        # Assume op == callnode
        if is_call_node(expr):
            run_fuse(None, expr)

        # Nice! This works
        #pattern = is_op('add')
        #print(pattern)
        #pattern = is_op('nn.relu')(wildcard(), pattern)
        #print(pattern)
        # Register?




    # defined at include/tvm/relay/op_attr_types.h
    tvm_op_dict = {"kElemWise":0, "kBroadcast":1, "kInjective":2, "kCommReduce":3, "kOutEWiseFusable":4, "kTuple":7, "kOpaque":8}

    #tvm_targets = [Target.TVM_GPU_AUTOTVM, Target.TVM_GPU_AUTOSCH, Target.GPU_NO_TUNING]
    #for tvm_target in tvm_targets:
    #    tvm_pattern_generator = BasePatternGenerator(tvm_target, tvm_op_dict, tvm_pattern_rule)
    #    self._add_backend_pattern_rule(tvm_pattern_generator)

    tvm_pattern_generator = BasePatternGenerator(Target.TVM_GPU_AUTOTVM, tvm_op_dict, tvm_pattern_rule)
    self._add_backend_pattern_rule(tvm_pattern_generator)

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
  def _add_backendop_with_key(self, target, pattern_key, constraint_func = no_constraints_func):
      self._add_backendop(target, optype_to_pattern[pattern_key], constraint_func)

  def _add_backendop(self, target, pattern, constraint_func = no_constraints_func):
    backendop = BackendOp(target, pattern, self._measured_configs, constraint_func)
    self.all_backendops.add(backendop)
    self.pattern_to_backendops[backendop.get_pattern()].add(backendop)
    #self.all_backendops.append(backendop)
    #self.pattern_to_backendops[backendop.get_pattern()].append(backendop)

  # return list of backend operators matching a pattern

  def measure_backend_ops(self, network_expr, targets, batch_size):
    assert type(targets) == list

    for target in targets:
      BackendOpCostEvaluator.get().log_network_backend_ops_perf_on_target(self, target, network_expr, batch_size)

  # return list of backend operators matching a pattern
  def get_backendops(self, pattern):
    return self.pattern_to_backendops[pattern]
    #return list(self.pattern_to_backendops[pattern])

  # return list of all patterns for backend operators
  def get_all_patterns(self):
    return list(self.pattern_to_backendops.keys())

  # @Sung
  # return list of all pattern rules for backend library
  def get_all_pattern_generators(self):
      return self.all_pattern_generators

  # save newly measured op perfs to the log
  def save_to_log(self, hw_name):
    return self._measured_configs.save_to_log(hw_name)
