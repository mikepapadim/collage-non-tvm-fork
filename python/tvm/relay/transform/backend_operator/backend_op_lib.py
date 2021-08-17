from tvm import relay
from collections import defaultdict

from .backend_op import BackendOp, get_optimal_backendop
from .op_config import Config, MeasuredConfigs

#from ..workloads.onnx_workloads import get_network_from_onnx
from .utils import no_constraints_func, get_op_pattern
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




# @ Sung: Check all nodes between src and sink by using fcheck (checks fusion conditions)
def check_path(src, sink, fcheck):
    queue = [ sink ]
    while len(queue)>0:
        node = queue.pop(0)
        if node == src:
            continue

        children = []
        if is_var_node(node) or is_constant_node(node):
            continue
        elif is_tuple_node(node):
            children = node.fields
        elif is_tuplegetitem_node(node):
            children = [ node.tuple ]
        elif is_call_node(node):
            children = node.args
        else:
            raise Exception(f"Unsupported type ({type(node)})")

        if not fcheck(node, node==src):
            return False

        queue.extend(children)

    return True

def generate_relay_pattern_node(node):
    if is_tuple_node(node):
        return is_tuple(), len(node.fields)
    elif is_tuplegetitem_node(node):
        return is_tuple_get_item, 2
    elif is_call_node(node):
        return is_op(node.op.name), len(node.args)
    elif is_constant_node(node):
        return is_constant(), 0
    elif is_var_node(node):
        return is_var(), 0
    else:
        raise Exception(f"Unsupported type ({type(node)})")


# NOTE: Seems like relay pattern matching considers pointer of pattern node.
#       Also, relay pattern string can be misleading
# for example,
#         is_conv2d = is_op('nn.conv2d')(is_var(), is_var())
#         path1 = is_op('nn.relu')(is_conv2d)
#         path2 = is_op('nn.leaky_relu')(is_conv2d)
#         diamond = is_op('add')(path1, path2)
#         --> CallPatternNode(Op(add), [CallPatternNode(Op(nn.relu), [CallPatternNode(Op(nn.conv2d), [VarPattern(),VarPattern()])]), CallPatternNode(Op(nn.leaky_relu), [CallPatternNode(Op(nn.conv2d), [VarPattern(),VarPattern()])])])
#
#         This diamond pattern does not match with the following expr
#            inp1 = relay.var('input1')
#            inp2 = relay.var('input2')
#            weight1 = relay.var('weight1')
#            weight2 = relay.var('weight2')
#            conv2d1 = relay.op.nn.conv2d(inp1, weight1)
#            conv2d2 = relay.op.nn.conv2d(inp2, weight2)
#            relu = relay.op.nn.relu(conv2d1)
#            leaky_relu = relay.op.nn.leaky_relu(conv2d2, alpha=0)
#            out = relu + leaky_relu

# dfs
def build_pattern_with_map(src, node, nodeToPatternMap):
    if node not in nodeToPatternMap:
        print(f"{node.op} is not in pattern map")
    assert(node in nodeToPatternMap)
    rpattern = nodeToPatternMap[node]
    children = []
    if is_var_node(node) or is_constant_node(node):
        pass
    elif is_tuple_node(node):
        children = node.fields
    elif is_tuplegetitem_node(node):
        children = [ node.tuple ]
    elif is_call_node(node):
        children = node.args
    else:
        raise Exception(f"Unsupported type ({type(node)})")

    if node == src:
        return rpattern

    operands = [ build_pattern_with_map(src, child, nodeToPatternMap) for child in children ]
    return rpattern(*operands)


def generate_relay_pattern(src, sink, cur_pattern_type = None, nodeToPatternMap = dict()):
    # Handle single node
    if src == sink:
        assert(cur_pattern_type is None)
        rpattern, num_operands = generate_relay_pattern_node(sink)
        operands = [wildcard() for __ in range(num_operands)]
        return rpattern(*operands), get_op_pattern(sink), 1

    # Handle multiple nodes between src and sink
    queue = [ sink ]
    cnt = 0 # count the number of new nodes
    assert(cur_pattern_type is not None)
    while len(queue)>0:
        node = queue.pop(0)
        if node == src:
            continue

        # NOTE: src should be created in the previous call
        if node not in nodeToPatternMap:
            cnt += 1
            nodeToPatternMap[node] = generate_relay_pattern_node(node)[0]

        children = []
        if is_var_node(node) or is_constant_node(node):
            continue
        elif is_tuple_node(node):
            children = node.fields
        elif is_tuplegetitem_node(node):
            children = [ node.tuple ]
        elif is_call_node(node):
            children = node.args
            cur_pattern_type = max(cur_pattern_type, get_op_pattern(node))
        else:
            raise Exception(f"Unsupported type ({type(node)})")

        queue.extend(children)

    src_pnode, num_operands = generate_relay_pattern_node(src)
    operands = [wildcard() for __ in range(num_operands)]
    nodeToPatternMap[src] = src_pnode(*operands)
    rpattern = build_pattern_with_map(src, sink, nodeToPatternMap)

    return rpattern, cur_pattern_type, cnt




#@Sung: base class for pattern_rule
class BasePatternGenerator:
  # OpKind, Comp graph, rule
  # NOTE: Ideall, OpKind should be defined by user depending on their pattern strategy.
  # However, as we will only support TVM pattern rules and use the relay attribute of OpKind, we are not going to define it for now.

  def __init__(self, _target,  _pattern_rules=None, _optype2enum=None, _enum2optype=None):
      self.target = _target
      self.fgen = _pattern_rules
      self.id = 0 # pattern id
      self.optype2enum = _optype2enum
      self.enum2optype = _enum2optype

  def _register_pattern_rule(self, _pattern_rules):
      self.fgen = _pattern_rules

  def run(self, dom_tree, expr):
      # def tvm_pattern_rule(dom_tree, op_dict, expr):
      self.fgen(expr, dom_tree, self.target, self.optype2enum, self.enum2optype)


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
    self._add_backendop_with_key(Target.CUDNN, "CONV2D")
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
    #add_all_backend_ops_to_lib(self, Target.TENSORRT, ["DIAMOND", "TRANSPOSE",
                                                       # "TUPLE_TWO_IDX", "TUPLE_FIVE_IDX",
                                                       # "TUPLE_FIVE_IDX_CONCAT",
                                                       # "TUPLE_GET_ITEM_0",
                                                       # "TUPLE_GET_ITEM_1",
   #                                                   "BATCH_MATMUL",
   #                                                   "RESHAPE_TRANSPOSE",
   #                                                   "TRANSPOSE_RESHAPE"])

    # CUBLAS
    # TODO: Add patterns. matmul, batch matmul
    self._add_backendop_with_key(Target.CUBLAS, "DENSE")
    self._add_backendop_with_key(Target.CUBLAS, "BATCH_MATMUL")

    # @Sung: add TVM pattern rule
    # TODO: Check with NASRNN.
    def tvm_pattern_rule(expr, dom_tree, target=Target.TVM_GPU_AUTOTVM, optype2enum = None, enum2optype = None):
        def run_fuse(src, sink, cur_pattern_type = None, cur_num_op = 0, nodeToPatternMap = dict()):
            assert(src is not None)
            if cur_pattern_type == optype2enum["kOpaque"] or cur_num_op > NUM_MAX_OP:
                return None

            if is_tuple_node(sink):
                # Go deeper if possible
                if sink in dom_tree:
                    run_fuse(src, dom_tree[sink], cur_pattern_type, cur_num_op, nodeToPatternMap)
                return None



            #print(f"src: {src}, sink: {sink}")
            sink_type = get_op_pattern(sink)
            # NOTE: This is current assumption. May not be true all the time.
            assert(sink_type != optype2enum["kOpaque"])
            num_nodes = 0
            cur_relay_pattern = None

            if src == sink:
                # Hanlde single op
                cur_relay_pattern, cur_pattern_type, num_nodes = generate_relay_pattern(src, sink)
            else:
                if cur_pattern_type == optype2enum["kOutEWiseFusable"]:
                    def fcheck(node, is_sink):
                        return get_op_pattern(node) <= optype2enum["kBroadcast"]
                    if sink_type <= optype2enum["kInjective"] and check_path(src, sink, fcheck):
                        cur_relay_pattern, cur_pattern_type, num_nodes = generate_relay_pattern(src, sink, cur_pattern_type, nodeToPatternMap)

                elif cur_pattern_type <= optype2enum["kBroadcast"]:
                    def fcheck(node, is_sink):
                        kind = get_op_pattern(node)
                        if not is_sink:
                            return kind <= optype2enum["kInjective"]
                        else:
                            return kind <= optype2enum["kOUtEWiseFusable"]

                    if (sink_type <= optype2enum["kCommReduce"] ) and check_path(src, sink, fcheck):
                        cur_relay_pattern, cur_pattern_type, num_nodes = generate_relay_pattern(src, sink, cur_pattern_type, nodeToPatternMap)

                elif cur_pattern_type == optype2enum["kInjective"] or cur_pattern_type == optype2enum["kTuple"]:
                    def fcheck(node, is_sink):
                        return get_op_pattern(node) <= optype2enum["kInjective"]
                    if check_path(src, sink, fcheck):
                        cur_relay_pattern, cur_pattern_type, num_nodes = generate_relay_pattern(src, sink, cur_pattern_type, nodeToPatternMap)

                else:
                    raise Exception(f"Unsupported type ({type(sink)})")


            # Invalid pattern
            if num_nodes==0 or cur_relay_pattern is None:
                return None


            cur_num_op += num_nodes
            # Register
            print(f"\t----- Register {cur_relay_pattern}")
            self._add_backendop(target, Pattern(cur_relay_pattern))

            # We may be able to expand
            #if sink in dom_tree and src_type != optype2enum["kTuple"]:
            if sink in dom_tree:
                run_fuse(src, dom_tree[sink], cur_pattern_type, cur_num_op, nodeToPatternMap)

        NUM_MAX_OP = 256
        # Assume op == callnode
        if not (is_constant_node(expr) or is_var_node(expr)):
            run_fuse(expr, expr)




    # defined at include/tvm/relay/op_attr_types.h
    tvm_enum2optype = {0:"kElemWise", 1:"kBroadcast", 2:"kInjective", 3:"kCommReduce", 4:"kOutEWiseFusable", 7:"kTuple", 8:"kOpaque"}
    tvm_optype2enum = {"kElemWise":0, "kBroadcast":1, "kInjective":2, "kCommReduce":3, "kOutEWiseFusable":4, "kTuple":7, "kOpaque":8}
    tvm_pattern_generator = BasePatternGenerator(Target.TVM_GPU_AUTOTVM, tvm_pattern_rule, tvm_optype2enum, tvm_enum2optype)
    self._add_backend_pattern_rule(tvm_pattern_generator)

    # TVM_GPU
    # add_all_backend_ops_to_lib(self, Target.TVM_GPU_AUTOSCH)
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
