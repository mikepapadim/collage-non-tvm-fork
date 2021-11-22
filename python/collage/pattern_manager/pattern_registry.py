from tvm import relay
from collections import defaultdict

from .matched_operators import BackendOp, get_optimal_backendop
from .op_config import Config, MeasuredConfigs

#from ..workloads.onnx_workloads import get_network_from_onnx
from .utils import no_constraints_func, get_op_pattern, get_args
from .cost_func import Target
from .default_pattern import optype_to_pattern
from .pattern_language import Pattern, name_relay_pattern

from tvm.relay.dataflow_pattern import *
from .utils import *

def add_all_backend_ops_to_lib(b_op_lib, target, exclued_ops=["DIAMOND"]):
  t_name = target.name()

  for pattern, pattern in optype_to_pattern.items():
    # Skip diamond pattern for now
    if pattern in exclued_ops:
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
# Start from sink
# Gotta be dfs
def check_path(src, node, fcheck, path = [], paths = []):
    path.append(node)
    if src == node:
        assert(len(path))
        paths.append(path.copy())

    elif is_var_node(node) or is_constant_node(node):
        pass
    elif fcheck(node, node==src):
        children = []
        if is_tuple_node(node):
            children = node.fields
        elif is_tuplegetitem_node(node):
            children = [ node.tuple ]
        elif is_call_node(node):
            children = node.args
        else:
            raise Exception(f"Unsupported type ({type(node)})")

        for child in children:
            check_path(src, child, fcheck, path, paths)

    out = path.pop()
    assert(node == out)



def generate_relay_pattern_node(node):
    if is_tuple_node(node):
        return is_tuple(), len(node.fields)
    #elif is_tuplegetitem_node(node):
    #    return is_tuple_get_item, 2
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


def build_pattern_with_map(src, node, nodeToPatternMap):
    if node in nodeToPatternMap:
        rpattern = nodeToPatternMap[node][0]

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
    else:
        return wildcard()


def generate_relay_pattern(src, sink, paths = None, cur_pattern_type = None, nodeToPatternMap = dict()):
    if paths is None:
        # Handle single node
        assert(not (is_constant_node(src) and is_var_node(src)))
        assert(cur_pattern_type is None)
        rpattern, num_operands = generate_relay_pattern_node(sink)
        operands = [wildcard() for __ in range(num_operands)]
        return rpattern(*operands), get_op_pattern(sink), 1

    else:
        # Handle multiple nodes
        # Create pattern node for all nodes in paths
        cnt = 0
        nodeToPatternMap = dict()
        for path in paths:
            for node in path:
                if node not in nodeToPatternMap:
                    nodeToPatternMap[node] = generate_relay_pattern_node(node)
                    cnt += 1

                # Create pattern node for const/var
                for child in get_args(node):
                    if is_constant_node(child) or is_var_node(child):
                        if child not in nodeToPatternMap:
                            nodeToPatternMap[child] = generate_relay_pattern_node(child)

        assert(src in nodeToPatternMap)
        pnode, num_operands = nodeToPatternMap[src]
        operands = [wildcard() for __ in range(num_operands)]
        nodeToPatternMap[src] = (pnode(*operands), 0) # it's zero cause we already handled.
        rpattern = build_pattern_with_map(src, sink, nodeToPatternMap)

        return rpattern, cur_pattern_type, cnt


def check_and_generate_pattern(src, sink, fcheck, cur_pattern_type, nodeToPatternMap):
    path, paths = [], []
    #print(">> check and gen")
    check_path(src, sink, fcheck, path, paths)
    if len(paths):
        return generate_relay_pattern(src, sink, paths, cur_pattern_type, nodeToPatternMap)
    else:
        return None, None, 0




#@Sung: base class for pattern_rule
class BasePatternGenerator:
  # OpKind, Comp graph, rule
  # NOTE: Ideall, OpKind should be defined by user depending on their pattern strategy.
  # However, as we will only support TVM pattern rules and use the relay attribute of OpKind, we are not going to define it for now.

  def __init__(self, _target,  _pattern_rules=None, _verify=None, _optype2enum=None, _enum2optype=None):
      self.target = _target
      self.fgen = _pattern_rules
      self.id = 0 # pattern id
      self.optype2enum = _optype2enum
      self.enum2optype = _enum2optype

      # check whether pattern satisfies the constraints
      if _verify is None:
          # Default one always returns True
          self.verify = self.default_verify
      else:
          self.verify = _verify

  def _register_pattern_rule(self, _pattern_rules):
      self.fgen = _pattern_rules

  # Always return true
  def default_verify(self, pattern):
      return True

  def run(self, dom_tree, expr):
      # def tvm_pattern_rule(dom_tree, op_dict, expr):
      self.fgen(expr, dom_tree, self.verify, self.target, self.optype2enum, self.enum2optype)


# library class (singleton) representing all backend operators
class BackendOpLib(object):
  __instance = None

  @staticmethod
  def get(hw_name):
    """ Static access method. """
    if BackendOpLib.__instance == None:
      BackendOpLib(hw_name)
    return BackendOpLib.__instance

  @staticmethod
  def destroy():
      """ Static access method. """
      BackendOpLib.__instance = None

  def __init__(self, hw_name):
    """ Virtually private constructor. """
    if BackendOpLib.__instance != None:
      raise Exception("This class is a singleton!")

    # list of all backend operators
    self._measured_configs = MeasuredConfigs()
    self._measured_configs.load_from_log(hw_name)
    # print("BACKEND OP LIG GET")

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

    # TODO(@Sung)
    # 1. Support exclusion for TRT
    # 2. conv3d, ADD, CONV2D_RELU, CONV2D_BIAS_ADD_RELU

    # FIXME(@Soo): For ResNext, some of CUDNN convolution doesn't work.
    self._add_backendop_with_key(Target.CUDNN, "CONV2D")
    self._add_backendop_with_key(Target.CUDNN, "CONV3D")

    def check_activation_constraints(config):
        dim = len(config._data_shape)
        return dim == 4 or dim == 5

    #self._add_backendop_with_key(Target.CUDNN, "SIGMOID", check_activation_constraints)
    #self._add_backendop_with_key(Target.CUDNN, "TANH", check_activation_constraints)
    self._add_backendop_with_key(Target.CUDNN, "SOFTMAX")
    self._add_backendop_with_key(Target.CUDNN, "MAX_POOL2D")
    self._add_backendop_with_key(Target.CUDNN, "AVG_POOL2D")
    # TODO:
    # self._add_backendop_with_key(Target.CUDNN, "CONV2D_ADD_RELU") # Bug at NasnetA
    #self._add_backendop_with_key(Target.CUDNN, "CONV2D_BIAS_RELU")
    #self._add_backendop_with_key(Target.CUDNN, "CONV3D_ADD_RELU")
    self._add_backendop_with_key(Target.CUDNN, "CONV3D_BIAS_RELU")
    #self._add_backendop_with_key(Target.CUDNN, "CONV2D_RELU")
    #self._add_backendop_with_key(Target.CUDNN, "RELU", check_activation_constraints) # RELU has correctness issue on ResNext


    # NOTE: cudnn ADD, BIAS_ADD cannot be supported due to the current limitation of packed function interface.
    # cudnnAddTensor() uses the last argument as both input/output.
    # self._add_backendop_with_key(Target.CUDNN, "ADD")
    # self._add_backendop_with_key(Target.CUDNN, "BIAS_ADD")

    # NOTE: BatchNorm is currently not supported. If you need it, please contact @Sung
    self._add_backendop_with_key(Target.CUDNN, "BATCHNORM")

    # DNNL, MKL, MKLDNN
    # TODO: Add patterns. matmul, batch matmul
    def check_tensor_constraints(config):
        dim1 = len(config._data_shape[0])
        dim2 = len(config._data_shape[1])
        print(f"{dim1}, {dim2}, {config._data_shape}", file=sys.stderr)
        return dim1 == 2 and dim2 == 2

    self._add_backendop_with_key(Target.MKL, "DENSE", check_tensor_constraints)
    self._add_backendop_with_key(Target.MKL, "BATCH_MATMUL")
    #self._add_backendop_with_key(Target.MKLDNN, "DENSE")


    def check_constraints_dnnl_add(config):
        for idx_shape, shape in enumerate(config._data_shape):
            if len(shape) < 2:
                return False

            # Check if all inputs have same dimensionality
            if idx_shape > 0 and len(shape) != prev_shape:
                return False
            prev_shape = len(shape)

            if shape == [1, 64, 56, 56] or shape == [1,128,28,28] or shape == [1, 256, 14, 14]:
                return False
        return True

    def check_constraints_dnnl_relu(config):
        for shape in config._data_shape:
            if len(shape) != 4:
                return False
        return True



    self._add_backendop_with_key(Target.DNNL, "CONV2D")
    self._add_backendop_with_key(Target.DNNL, "CONV3D")
    self._add_backendop_with_key(Target.DNNL, "BATCHNORM")
    self._add_backendop_with_key(Target.DNNL, "DENSE")
    # Disabled cuz it still errors out for DCGAN / NasNet-A
    #self._add_backendop_with_key(Target.DNNL, "ADD", check_constraints_dnnl_add)
    self._add_backendop_with_key(Target.DNNL, "RELU", check_constraints_dnnl_relu)

    # Unsupported error by DNNL
    #self._add_backendop_with_key(Target.DNNL, "SUBTRACT")
    #self._add_backendop_with_key(Target.DNNL, "MULTIPLY")

    # CUBLAS
    # TODO: Add patterns. matmul, batch matmul
    self._add_backendop_with_key(Target.CUBLAS, "DENSE")
    self._add_backendop_with_key(Target.CUBLAS, "BATCH_MATMUL")

    # @Sung: add TVM pattern rule
    def tvm_pattern_rule(expr, dom_tree, verify, target=Target.AUTOTVM, optype2enum = None, enum2optype = None):
        def run_fuse(src, sink, cur_pattern_type = None, cur_num_op = 0, nodeToPatternMap = dict()):
            assert(src is not None)
            if cur_pattern_type == optype2enum["kOpaque"] or cur_num_op > NUM_MAX_OP:
                return None

            if is_tuple_node(sink):
                # Go deeper if possible
                if sink in dom_tree:
                    run_fuse(src, dom_tree[sink], cur_pattern_type, cur_num_op, nodeToPatternMap)
                return None

            sink_type = get_op_pattern(sink)
            # NOTE: This is current assumption. May not be true all the time.
            # assert(sink_type != optype2enum["kOpaque"])
            num_nodes = 0
            cur_relay_pattern = None

            if src == sink:
                # Handle single op
                cur_relay_pattern, cur_pattern_type, num_nodes = generate_relay_pattern(src, sink)
            else:
                #print(f"num: {cur_num_op}, cur_pattern: {enum2optype[cur_pattern_type]}, sink_type: {enum2optype[sink_type]}\nsrc: {src}\nsink: {sink}")

                if cur_pattern_type == optype2enum["kOutEWiseFusable"]:
                    def fcheck(node, is_sink):
                        return get_op_pattern(node) <= optype2enum["kBroadcast"]

                    if sink_type <= optype2enum["kInjective"]:
                        cur_relay_pattern, cur_pattern_type, num_nodes = check_and_generate_pattern(src, sink, fcheck, cur_pattern_type, nodeToPatternMap)

                elif cur_pattern_type <= optype2enum["kBroadcast"]:
                    def fcheck(node, is_sink):
                        kind = get_op_pattern(node)
                        if not is_sink:
                            return kind <= optype2enum["kInjective"]
                        else:
                            return kind <= optype2enum["kOutEWiseFusable"]

                    if sink_type <= optype2enum["kCommReduce"]:
                        cur_relay_pattern, cur_pattern_type, num_nodes = check_and_generate_pattern(src, sink, fcheck, cur_pattern_type, nodeToPatternMap)

                elif cur_pattern_type == optype2enum["kInjective"] or cur_pattern_type == optype2enum["kTuple"]:
                    def fcheck(node, is_sink):
                        return get_op_pattern(node) <= optype2enum["kInjective"]
                    cur_relay_pattern, cur_pattern_type, num_nodes = check_and_generate_pattern(src, sink, fcheck, cur_pattern_type, nodeToPatternMap)
                elif cur_pattern_type == optype2enum["kCommReduce"] or cur_pattern_type == optype2enum["kOpaque"]:
                    pass
                else:
                    raise Exception(f"Unsupported type ({type(sink)}, {enum2optype[cur_pattern_type]}, {src})")


            # Invalid pattern
            if num_nodes==0 or cur_relay_pattern is None:
                return None


            cur_num_op += num_nodes

            if verify(cur_relay_pattern):
                # Register
                #print(f"\t----- Register {cur_relay_pattern}")

                self._add_backendop(target, Pattern(cur_relay_pattern))

                # [Deprecated - this wasn't an issue] Do not register if it is errorneous patterns
                # e.g., for bert_full, these are errorneous patterns
                # errorneous_patterns = ['0-Op(reshape)[1-Op(transpose)[2-Op(reshape)[3-Op(add)[*, *]]]]',
                #                        "0-Op(add)[1-Var/Const, 2-Op(add)[*, *]]",
                #                        "0-Op(add)[1-Op(add)[*, *], *]",
                #                        "0-Op(add)[*, 1-Op(add)[*, *]]",
                #                        "0-Op(transpose)[1-Op(reshape)[2-Op(add)[*, *]]]",
                #                        "0-Op(reshape)[1-Op(add)[*, *]]"]
                #
                # if not name_relay_pattern(cur_relay_pattern)[0] in errorneous_patterns:
                #     self._add_backendop(target, Pattern(cur_relay_pattern))
                # else:
                #     logging.info(f"The following pattern is excluded: {name_relay_pattern(cur_relay_pattern)[0]}")

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
    tvm_pattern_generator = BasePatternGenerator(Target.AUTOTVM, tvm_pattern_rule, None, tvm_optype2enum, tvm_enum2optype)
    self._add_backend_pattern_rule(tvm_pattern_generator)

    # Add TVM Default patterns
    tvm_pattern_generator = BasePatternGenerator(Target.TVM_DEFAULT, tvm_pattern_rule, None, tvm_optype2enum,
                                                 tvm_enum2optype)
    self._add_backend_pattern_rule(tvm_pattern_generator)

    # TVM_GPU
    # add_all_backend_ops_to_lib(self, Target.AUTOSCH)
    add_all_backend_ops_to_lib(self, Target.AUTOTVM)
    add_all_backend_ops_to_lib(self, Target.TVM_DEFAULT)

    # add_all_backend_ops_to_lib_except_fused(backendop_lib, Target.TVM_GPU)

    # TVM_GPU_NO_TUNING
    #add_all_backend_ops_to_lib(self, Target.TVM_GPU_NO_TUNING)
    # add_all_backend_ops_to_lib_except_fused(backendop_lib, Target.TVM_GPU_NO_TUNING)

    # TVM_CPU; Exclude it for GPU testing
    # Fix: Extend this to automatically select backend library based on HW info
    # add_all_backend_ops_to_lib(backendop_lib, Target.TVM_CPU)


    # TENSORRT
    # NOTE: Current TensorRT pattern follows TVM fusion rule for simplicity.
    # But, since BATCH_MATMUL and TRANSPOSE are not supported, we are going to exclude the patterns if they contain those illegal operators by passing verify function.
    # ops_to_exclude_trt = ["image.resize", "divide", "multiply"]
    ops_to_exclude_trt = ["image.resize"]
    # ops_to_exclude_trt = ["transpose", "image.resize", "variance", "divide", "reshape", "nn.batch_matmul", "multiply"]
    # ops_to_exclude_trt = ["transpose", "image.resize", "variance", "divide", "reshape", "nn.batch_matmul"]

    def trt_verify(pattern):
        q = [ pattern ]
        while len(q):
            cur = q.pop()

            #if isinstance(cur, WildcardPattern) or isinstance(cur, VarPattern) or isinstance(cur, ConstantPattern):
            if isinstance(cur, WildcardPattern):
                pass
            elif isinstance(cur, CallPattern):
                if isinstance(cur.op, ConstantPattern) or isinstance(cur.op, VarPattern):
                    pass
                else:
                    op_name = cur.op.expr.name

                    if op_name in ops_to_exclude_trt:
                        return False
                    q.extend(cur.args)
            elif isinstance(cur, TuplePattern):
                q.extend(cur.fields)
            elif isinstance(cur, TupleGetItemPattern):
                q.append(cur.tuple_value)
            else:
                raise Exception(f"Unexpected expression type, {type(cur)}")

        return True

    trt_pattern_generator = BasePatternGenerator(Target.TENSORRT, tvm_pattern_rule, trt_verify, tvm_optype2enum, tvm_enum2optype)
    self._add_backend_pattern_rule(trt_pattern_generator)

    #def check_constraints_batch_matmul(config):
    #    assert(len(config._data_shape) == 2)
    #    t1, t2 = config._data_shape[0], config._data_shape[1]
    #    if not (len(t1)==3 and len(t2)==3 and t1[-1] == t2[-2]):
    #        return False
    #    return True
    #self._add_backendop_with_key(Target.TENSORRT, "BATCH_MATMUL", check_constraints_batch_matmul)

    #add_all_backend_ops_to_lib(self, Target.TENSORRT, ["DIAMOND", "TRANSPOSE",
                                                       # "TUPLE_TWO_IDX", "TUPLE_FIVE_IDX",
                                                       # "TUPLE_FIVE_IDX_CONCAT",
                                                       # "TUPLE_GET_ITEM_0",
                                                       # "TUPLE_GET_ITEM_1",
   #                                                   "BATCH_MATMUL",
   #                                                   "RESHAPE_TRANSPOSE",
   #                                                   "TRANSPOSE_RESHAPE"])



  # add a backend operator to the library
  def _add_backendop_with_key(self, target, pattern_key, constraint_func = no_constraints_func):
      self._add_backendop(target, optype_to_pattern[pattern_key], constraint_func)

  def _add_backendop(self, target, pattern, constraint_func = no_constraints_func):
    backendop = BackendOp(target, pattern, self._measured_configs, constraint_func)
    self.all_backendops.add(backendop)
    self.pattern_to_backendops[backendop.get_pattern()].add(backendop)

  def measure_backend_ops(self, network_expr, targets, batch_size):
    assert type(targets) == list

    for target in targets:
      BackendOpCostEvaluator.get().log_network_backend_ops_perf_on_target(self, target, network_expr, batch_size)

  # return list of backend operators matching a pattern
  def get_backendops(self, pattern):
    return self.pattern_to_backendops[pattern]
    #return list(self.pattern_to_backendops[pattern])

  """
  Input: Target backend, backend to exclude
  Return: the list of backend operators from the given target backend while exclude them if the pattern also matches

  This is used to find a backend op assignment for a single backend baseline
  """
  def get_all_patterns_and_backend_ops_from_single_backend(self, target_backend, backend_to_exclude=None):
      # Generate op names to exclude
      op_names_to_exclude = set()
      # print(f"\n\n\nbackend OPs from {backend_to_exclude}")
      if backend_to_exclude is not None:
          for pat, b_ops in self.pattern_to_backendops.items():
              # Consider only the pattern with the depth of 1
              # We assume that fused op always include at least one of single ops supported by target single backend.
              if pat.get_depth() > 1:
                  continue

              for b_op in b_ops:
                  if b_op.get_target() in backend_to_exclude:
                      # print(b_op)
                      op_names_to_exclude |= pat.get_op_name_set()
                      break

      # Generate all patterns and backends ops from single backend while excluding ops any part of which can be matched
      # with backend_to_exclude
      pat_and_b_op = []
      for pat, b_ops in self.pattern_to_backendops.items():
          for b_op in b_ops:
              if b_op.get_target() == target_backend and op_names_to_exclude.isdisjoint(pat.get_op_name_set()):
                  pat_and_b_op.append((pat, b_op))
                  break

      pat_and_b_op.sort(key = lambda tup: tup[0].get_depth(), reverse=True)

      return pat_and_b_op

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
