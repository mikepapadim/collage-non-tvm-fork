import tvm._ffi
import tvm.driver
# import tvm.relay.testing as testing
# from tvm import relay

# from ..backend_operator.record import backendop_lib
from ..backend_operator.op_config import MeasuredConfigs
from ..backend_operator.target import Target
from ..backend_operator.op_type import OpType
from ..backend_operator.backend_op_lib import BackendOpLib

from .comp_graph import ComputationGraph
from .comp_graph_optimizer import CompGraphOptimizer

from .optimizer_utils import print_matching_final

def setup_backend_op_lib(network_expr, targets, batch_size):
    backendop_lib = BackendOpLib.get()
    # backendop_lib.measure_backend_ops(network_expr, targets, batch_size)

    return backendop_lib


@tvm._ffi.register_func("relay.transform.optimizer.optimize_comp_graph")
def optimize_comp_graph(relay_expr):
    """Optimizing pass for computation graph representation (Relay IR).
    
    Parameters
    ----------
    relay_expr : tvm.relay.expr
        Relay IR for computation graph
    
    Returns
    -------
    matched_relay_expr : tvm.relay.expr
        The result matching between backend operators and Relay operators
    """

    # It is a function if you get it from last pass of Relay build
    # print("Relay expression")
    # print(relay_expr)
    if type(relay_expr) == tvm.relay.function.Function:
        relay_expr = relay_expr.body

    comp_graph = ComputationGraph(relay_expr)

    # Warning: ResNet-8 doesn't have tuned operators / CuDNN doesn't work for ResNet-8
    # target_backend = None # Consider all targets

    # targets = [Target.TENSORRT, Target.CUDNN, Target.CUBLAS, Target.TVM_GPU_NO_TUNING, Target.TVM_CPU]
    # targets = [Target.TVM_GPU_NO_TUNING, Target.TVM_GPU]
    # targets = [Target.TVM_GPU_AUTOSCH]
    # targets = [Target.TENSORRT]
    # targets = [Target.CUDNN, Target.TVM_GPU_NO_TUNING]
    # targets = [Target.TENSORRT, Target.TVM_GPU_NO_TUNING]
    targets = [Target.TVM_GPU_NO_TUNING, Target.TENSORRT, Target.CUDNN]
    batch_size = 1
    backendop_lib = setup_backend_op_lib(relay_expr, targets, batch_size)

    # Optimizing graph
    print("Computation graph created")
    optimizer = CompGraphOptimizer(backendop_lib, targets)
    print("Optimizer created")
    optimizer.optimize(comp_graph)
    print("It's optimized")
    optimized_match, post_order_match_result = optimizer.get_optimized_match(comp_graph)

    # print("Match result: ", optimized_match)
    # post_order_match_result is for debugging to check if it matches the final result from the TVM DP fusion pass
    print("Match result")
    for idx, pair in enumerate(post_order_match_result):
        print(idx, pair)
    # Debug (@soo)
    # print_matching_final(comp_graph, optimizer.loc2match)
    print("-"*40)
    # print("Optimized match")
    # print(optimized_match)

    backendop_lib.save_to_log()

    return optimized_match

# # Test script with ResNet-8
# if __name__ == "__main__":
#     batch_size = 1
#     num_class = 1000

#     # Resnet-8
# #     image_shape = (3, 28, 28)
# #     mod, params = testing.resnet.get_workload(num_layers=8, batch_size=batch_size, image_shape=image_shape)
# #     relay_expr = mod["main"].body

#     # Chain graph
#     out_channels = 16
#     batch_size = 1

#     data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
#     conv_weight = relay.var("weight")
#     dense_weight = relay.var("weight")
#     bn_gamma = relay.var("bn_gamma")
#     bn_beta = relay.var("bn_beta")
#     bn_mmean = relay.var("bn_mean")
#     bn_mvar = relay.var("bn_var")

#     simple_net = relay.nn.conv2d(
#         data=data, weight=conv_weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
#     )
# #     simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
# #     simple_net = relay.nn.relu(simple_net)
# #     simple_net = relay.nn.relu(simple_net)
#     relay_expr = simple_net
    
#     optimize_comp_graph(relay_expr)