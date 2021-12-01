from tvm import relay
from collections import defaultdict

from .default_patterns import str_to_pattern
from .pattern_language import Pattern, name_relay_pattern

#from tvm.relay.dataflow_pattern import *
from collage.utils import (
        no_constraints_func,
        get_op_pattern,
        get_op_pattern, 
        get_args,
        is_var_node, 
        is_constant_node, 
        is_tuple_node, 
        is_tuplegetitem_node,
        no_constraints_func
      )

from collage.backend import BackendKind
from collage.measurer.op_cost_logger import Config
import logging

# Backend pattern is a basic unit that pattern registry uses to manage (backend, pattern) pair
class BackendPattern(object):
  def __init__(self, backend, pattern, op_cost_logger, constraint_func = no_constraints_func,):
    self._backend = backend
    self._pattern = pattern 
    self._op_cost_logger = op_cost_logger 
    self._constraint_func = constraint_func

    self._op_name = pattern.get_name()
    self._name = backend.name + "_" + self._op_name
    self._depth = pattern.get_depth()
    
  def __hash__(self):
      return hash((self._name, self._constraint_func))

  def __eq__(self, other):
      return self._name == other._name and self._constraint_func == other._constraint_func

  def __repr__(self):
    return self._name

  def get_pattern(self):
    return self._pattern

  def get_backend(self):
    return self._backend

  # max depth is the depth of the longest branch
  # (chain pattern has a single branch, diamond pattern has 2 branchese)
  def get_depth(self):
    return self._depth

  def get_cost(self, expr, build_target):
    # configuration: backend operator name, operator type (which encode pattern), data shape, node attributes
    config = Config(self._name, self._op_name, expr)
    # print(config)

    # For Tuple, we do not need to measure it
    if is_tuple_node(expr) or is_tuplegetitem_node(expr):
      return 0

    # if constraints are not satisfied, return infinite cost
    if not self._constraint_func(config):
      return float('inf')#, 0

    cost_info = self._op_cost_logger.get_cost(config)

    if cost_info != None:
      logging.info("REUSED RESULT!!!!")
    else:
      logging.info("NOT REUSED!!!!")

      cost_info = self._backend.measure_cost(expr, build_target, name = self._name)
      self._op_cost_logger.save_cost(config, cost_info)

    # We use mean as a cost instead of sampling for now
    mean_cost, std_cost = cost_info
    return mean_cost


# library class (singleton) representing all backend patterns
class PatternRegistry(object):
  __instance = None

  @staticmethod
  def destroy():
      """ Static access method. """
      PatternRegistry.__instance = None

  def __init__(self, backend_registry, op_cost_logger):
    """ Virtually private constructor. """
    if PatternRegistry.__instance != None:
      raise Exception("This class should be a singleton!")


    self.backend_registry = backend_registry
    self.op_cost_logger = op_cost_logger
    # Graph-level backends will be fine-tuned with graph-level optimizer
    self.graph_level_backends = set()
    # Pattern generate will automatically add legal patterns right before optimizers
    self.backends_with_pattern_generators = set()
    # Unique set of backend patterns (backend+pattern)
    self.all_backend_patterns = set()
    # Manages a pattern <-> bakcend patterns relations
    self.pattern_to_backend_patterns = defaultdict(set)
    
    
    for name, backend_obj in self.backend_registry.items():
        # Add enumerated patterns to pattern registry
        for pattern, constraint_func in backend_obj.patterns:
          self.add_backend_pattern(backend_obj, pattern, constraint_func)

        # Add backends with pattern generator
        if backend_obj.pattern_generator is not None:
          self.backends_with_pattern_generators.add(backend_obj)
        
        # Add graph-level backends
        if backend_obj.kind == BackendKind.GRAPH_LEVEL:
          self.graph_level_backends.add(backend_obj)
    # Manage its instance
    PatternRegistry.__instance = self

  def get_registered_backends(self):
    return list(self.backend_registry.keys())

  def add_backend_pattern(self, backend, pattern, constraint_func):
    constraint_func = no_constraints_func if constraint_func is None else constraint_func
    backend_pattern = BackendPattern(
                                      backend, 
                                      pattern,
                                      self.op_cost_logger,
                                      constraint_func,
                                  )
    self.all_backend_patterns.add(backend_pattern)
    self.pattern_to_backend_patterns[backend_pattern.get_pattern()].add(backend_pattern)
  
  def get_backend_patterns(self, pattern):
    return self.pattern_to_backend_patterns[pattern]

  # save newly measured op perfs to the log
  def save_to_log(self):
    return self.op_cost_logger.save_to_log()




"""

def add_all_backend_patterns_to_lib(b_op_lib, target, exclued_ops=["DIAMOND"]):
  t_name = target.name()

  for pattern, pattern in str_to_pattern.items():
    # Skip diamond pattern for now
    if pattern in exclued_ops:
      continue
    b_op_lib._add_backend_pattern(target, pattern)

  # Note that we only support ResNet50 for now
  def _add_default_backend_patterns(self):
    # CUDNN

    # TODO(@sunggg)
    # 1. Support exclusion for TRT
    # 2. conv3d, ADD, CONV2D_RELU, CONV2D_BIAS_ADD_RELU

    # FIXME(@Soo): For ResNext, some of CUDNN convolution doesn't work.
    self._add_backend_pattern_with_key(Target.CUDNN, "CONV2D")
    self._add_backend_pattern_with_key(Target.CUDNN, "CONV3D")

    def check_activation_constraints(config):
        dim = len(config._data_shape)
        return dim == 4 or dim == 5

    #self._add_backend_pattern_with_key(Target.CUDNN, "SIGMOID", check_activation_constraints)
    #self._add_backend_pattern_with_key(Target.CUDNN, "TANH", check_activation_constraints)
    self._add_backend_pattern_with_key(Target.CUDNN, "SOFTMAX")
    self._add_backend_pattern_with_key(Target.CUDNN, "MAX_POOL2D")
    self._add_backend_pattern_with_key(Target.CUDNN, "AVG_POOL2D")
    
    # self._add_backend_pattern_with_key(Target.CUDNN, "CONV2D_ADD_RELU") # Bug at NasnetA
    #self._add_backend_pattern_with_key(Target.CUDNN, "CONV2D_BIAS_RELU")
    #self._add_backend_pattern_with_key(Target.CUDNN, "CONV3D_ADD_RELU")
    self._add_backend_pattern_with_key(Target.CUDNN, "CONV3D_BIAS_RELU")
    #self._add_backend_pattern_with_key(Target.CUDNN, "CONV2D_RELU")
    #self._add_backend_pattern_with_key(Target.CUDNN, "RELU", check_activation_constraints) # RELU has correctness issue on ResNext


    # NOTE: cudnn ADD, BIAS_ADD cannot be supported due to the current limitation of packed function interface.
    # cudnnAddTensor() uses the last argument as both input/output.
    # self._add_backend_pattern_with_key(Target.CUDNN, "ADD")
    # self._add_backend_pattern_with_key(Target.CUDNN, "BIAS_ADD")

    # NOTE: BatchNorm is currently not supported. If you need it, please contact @sunggg
    self._add_backend_pattern_with_key(Target.CUDNN, "BATCHNORM")

    # DNNL, MKL, MKLDNN
    # TODO: Add patterns. matmul, batch matmul
    def check_tensor_constraints(config):
        dim1 = len(config._data_shape[0])
        dim2 = len(config._data_shape[1])
        print(f"{dim1}, {dim2}, {config._data_shape}", file=sys.stderr)
        return dim1 == 2 and dim2 == 2

    self._add_backend_pattern_with_key(Target.MKL, "DENSE", check_tensor_constraints)
    self._add_backend_pattern_with_key(Target.MKL, "BATCH_MATMUL")
    #self._add_backend_pattern_with_key(Target.MKLDNN, "DENSE")


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

    self._add_backend_pattern_with_key(Target.DNNL, "CONV2D")
    self._add_backend_pattern_with_key(Target.DNNL, "CONV3D")
    self._add_backend_pattern_with_key(Target.DNNL, "BATCHNORM")
    self._add_backend_pattern_with_key(Target.DNNL, "DENSE")
    # Disabled cuz it still errors out for DCGAN / NasNet-A
    #self._add_backend_pattern_with_key(Target.DNNL, "ADD", check_constraints_dnnl_add)
    self._add_backend_pattern_with_key(Target.DNNL, "RELU", check_constraints_dnnl_relu)

    # Unsupported error by DNNL
    #self._add_backend_pattern_with_key(Target.DNNL, "SUBTRACT")
    #self._add_backend_pattern_with_key(Target.DNNL, "MULTIPLY")

    # CUBLAS
    # TODO: Add patterns. matmul, batch matmul
    self._add_backend_pattern_with_key(Target.CUBLAS, "DENSE")
    self._add_backend_pattern_with_key(Target.CUBLAS, "BATCH_MATMUL")

    # @sunggg: add TVM pattern rule
   


    # defined at include/tvm/relay/op_attr_types.h
    tvm_pattern_generator = BasePatternGenerator(Target.AUTOTVM, tvm_pattern_rule, None)
    self._add_backend_pattern_rule(tvm_pattern_generator)

    # Add TVM Default patterns
    tvm_pattern_generator = BasePatternGenerator(Target.TVM_DEFAULT, tvm_pattern_rule, None)
    self._add_backend_pattern_rule(tvm_pattern_generator)

    # TVM_GPU
    # add_all_backend_patterns_to_lib(self, Target.AUTOSCH)
    add_all_backend_patterns_to_lib(self, Target.AUTOTVM)
    add_all_backend_patterns_to_lib(self, Target.TVM_DEFAULT)

    # add_all_backend_patterns_to_lib_except_fused(pattern_registry, Target.TVM_GPU)

    # TVM_GPU_NO_TUNING
    # add_all_backend_patterns_to_lib(self, Target.TVM_GPU_NO_TUNING)
    # add_all_backend_patterns_to_lib_except_fused(pattern_registry, Target.TVM_GPU_NO_TUNING)

    # TVM_CPU; Exclude it for GPU testing
    # Fix: Extend this to automatically select backend library based on HW info
    # add_all_backend_patterns_to_lib(pattern_registry, Target.TVM_CPU)


    # TENSORRT
    # NOTE: Current TensorRT pattern follows TVM fusion rule for simplicity.
    # But, since BATCH_MATMUL and TRANSPOSE are not supported, we are going to exclude the patterns if they contain those illegal operators by passing verify function.
    ops_to_exclude_trt = ["image.resize"]

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

    trt_pattern_generator = BasePatternGenerator(Target.TENSORRT, tvm_pattern_rule, trt_verify)
    self._add_backend_pattern_rule(trt_pattern_generator)

"""