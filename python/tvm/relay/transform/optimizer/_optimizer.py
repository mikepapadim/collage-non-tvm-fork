import tvm._ffi
import tvm.driver
# import tvm.relay.testing as testing
# from tvm import relay

# from ..backend_operator.record import backendop_lib
from ..backend_operator.backend_op import BackendOpLib
from ..backend_operator.op_config import MeasuredConfigs
from ..backend_operator.target import Target
from ..backend_operator.op_type import OpType

from .comp_graph import ComputationGraph
from .comp_graph_optimizer import CompGraphOptimizer

from .optimizer_utils import print_matching_final

def add_all_backend_ops_to_lib(b_op_lib, target):
    t_name = target.name()

    for op_type in OpType:
        # Skip diamond pattern for now
        if op_type==OpType.DIAMOND:
            continue

        op_name, op_depth = op_type.name(), op_type.depth()
        b_op_lib.add_backendop(f"{t_name}_{op_name}", target, op_type, op_depth)

def create_backendop_lib():
    measured_configs = MeasuredConfigs()
    measured_configs.load_from_log()

    backendop_lib = BackendOpLib(measured_configs)

    # CUDNN
    # TODO: discuss with Soo.
    backendop_lib.add_backendop("cudnn_conv2d", Target.CUDNN, OpType.CONV2D, 1)
    # backendop_lib.add_backendop("cudnn_softmax", Target.CUDNN, OpType.SOFTMAX, 1)
    # backendop_lib.add_backendop("cudnn_biasadd", Target.CUDNN, OpType.BIAS_ADD, 1)
    backendop_lib.add_backendop("cudnn_relu", Target.CUDNN, OpType.RELU, 1)
    # backendop_lib.add_backendop("cudnn_bn", Target.CUDNN, OpType.BN, 1)

    # measure_cost doesn't work, we need to fix this later.
    # backendop_lib.add_backendop("cudnn_maxpool2d", Target.CUDNN, OpType.MAX_POOL2D, 1)

    # conv_bias_add_relu --> ResNet doesn't have this pattern, so it wouldn't be measured
    # backendop_lib.add_backendop("cudnn_conv2d+biasadd+relu", Target.CUDNN, OpType.CONV2D_BIAS_ADD_RELU, 3)
    backendop_lib.add_backendop("cudnn_add", Target.CUDNN, OpType.ADD, 1)


    # Non-existing patterns
    #backendop_lib.add_backendop("cudnn_dense", Target.CUDNN, OpType.DENSE, 1) #TODO: matmul?
    #backendop_lib.add_backendop("cudnn_batchflatten", Target.CUDNN, OpType.BATCH_FLATTEN, 1)
    #backendop_lib.add_backendop("cudnn_globalavgpool2d", Target.CUDNN, OpType.GLOBAL_AVG_POOL2D, 1)
    #backendop_lib.add_backendop("cudnn_conv2d+bn", Target.CUDNN, OpType.CONV2D_BN, 2)
    #backendop_lib.add_backendop("cudnn_bn+relu", Target.CUDNN, OpType.BN_RELU, 2)
    #backendop_lib.add_backendop("cudnn_conv2d+bn+relu", Target.CUDNN, OpType.CONV2D_BN_RELU, 3)
    #backendop_lib.add_backendop("cudnn_diamond", Target.CUDNN, OpType.DIAMOND, 6)


    # TENSORRT
    add_all_backend_ops_to_lib(backendop_lib, Target.TENSORRT)

    # CUBLAS
    # TODO: Add patterns. matmul, batch matmul
    backendop_lib.add_backendop("cublas_dense", Target.CUBLAS, OpType.DENSE, 1)

    # TVM_GPU
    # add_all_backend_ops_to_lib(backendop_lib, Target.TVM_GPU)
    # add_all_backend_ops_to_lib_except_fused(backendop_lib, Target.TVM_GPU)

    # TVM_GPU_NO_TUNING
    add_all_backend_ops_to_lib(backendop_lib, Target.TVM_GPU_NO_TUNING)
    
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
    print("Relay expression")
    # print(relay_expr)
    if type(relay_expr) == tvm.relay.function.Function:
        relay_expr = relay_expr.body

    comp_graph = ComputationGraph(relay_expr)

    # Warning: ResNet-8 doesn't have tuned operators / CuDNN doesn't work for ResNet-8
    # target_backend = None # Consider all targets
    target_backend = [Target.TVM_GPU_NO_TUNING]
    # target_backend = [Target.TVM_GPU_NO_TUNING, Target.TENSORRT]
    backendop_lib = create_backendop_lib()

    # Optimizing graph
    print("Computation graph created")
    optimizer = CompGraphOptimizer(backendop_lib, target_backend)
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