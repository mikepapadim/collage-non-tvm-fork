from tvm import relay
from collections import defaultdict

from .backend_op import BackendOp, get_optimal_backendop
from .op_config import Config, MeasuredConfigs

#from ..workloads.onnx_workloads import get_network_from_onnx
from .utils import no_constraints_func
from .target import Target
from .op_type import OpType

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
    # dictionary that maps each pattern to list of backend ops represented by the pattern
    self.pattern_to_backendops = defaultdict(list)

    self._add_all_backendops()

    BackendOpLib.__instance = self

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
                                                       OpType.BATCH_MATMUL, OpType.RESHAPE_TRANSPOSE,
                                                       OpType.TRANSPOSE_RESHAPE])

    # CUBLAS
    # TODO: Add patterns. matmul, batch matmul
    self._add_backendop("cublas_dense", Target.CUBLAS, OpType.DENSE, 1)
    self._add_backendop("cublas_batch_matmul", Target.CUBLAS, OpType.BATCH_MATMUL, 1)

    # TVM_GPU
    add_all_backend_ops_to_lib(self, Target.TVM_GPU_AUTOSCH)
    add_all_backend_ops_to_lib(self, Target.TVM_GPU_AUTOTVM)
    # add_all_backend_ops_to_lib_except_fused(backendop_lib, Target.TVM_GPU)

    # TVM_GPU_NO_TUNING
    add_all_backend_ops_to_lib(self, Target.TVM_GPU_NO_TUNING)
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

  # save newly measured op perfs to the log
  def save_to_log(self):
    return self._measured_configs.save_to_log()
