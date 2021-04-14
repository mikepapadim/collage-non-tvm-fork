import tvm.relay.testing as testing
from tvm import relay
from ..backend_operator.target import Target
from tvm.contrib import graph_runtime as runtime
import tvm
from ._optimizer import optimize_comp_graph
from ..workloads.onnx_workloads import get_network_from_onnx
from .comp_graph import ComputationGraph
from .comp_graph_optimizer import CompGraphOptimizer
from ..backend_operator.backend_op_lib import BackendOpLib
from .optimizer_utils import print_matching_final

def get_input_network():
    # Chain graph
    out_channels = 16
    batch_size = 1

    data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
    conv_weight = relay.var("weight", relay.TensorType((out_channels, 3, 3, 3), "float32"))
    # dense_weight = relay.var("weight")
    # bn_gamma = relay.var("bn_gamma")
    # bn_beta = relay.var("bn_beta")
    # bn_mmean = relay.var("bn_mean")
    # bn_mvar = relay.var("bn_var")

    simple_net = relay.nn.relu(data)
    # simple_net = relay.nn.relu(simple_net)
    simple_net = relay.nn.conv2d(
        data=data, weight=conv_weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    )
    # simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    # simple_net = relay.nn.relu(simple_net)
    simple_net = relay.nn.relu(simple_net)

    expr = simple_net

    return expr

def get_resnet_8():
    batch_size = 1
    num_class = 1000
    # Resnet-8
    image_shape = (3, 28, 28)

    # ResNet-2
    # mod, params = testing.resnet.get_workload(num_layers=2, batch_size=batch_size, image_shape=image_shape)

    # Resnet-8
    mod, params = testing.resnet.get_workload(num_layers=8, batch_size=batch_size, image_shape=image_shape)

    expr_func = mod["main"]
    # target = Target.TVM_GPU_NO_TUNING
    # net, params = testing.create_workload(expr_func)
    #
    # # # Build the subgraph
    # target_str = target.__str__()
    # ctx = tvm.context(target_str, 0)
    # lib = relay.build_module.build(net, target_str, params=params)
    # mod = runtime.GraphModule(lib["default"](ctx))

    return expr_func.body

relay_expr = get_resnet_8()
# relay_expr = get_input_network()
comp_graph = ComputationGraph(relay_expr)

targets = [Target.CUDNN, Target.TENSORRT, Target.TVM_GPU_NO_TUNING]
batch_size = 1
backendop_lib = BackendOpLib.get()

# Optimizing graph
print("Computation graph created")
optimizer = CompGraphOptimizer(backendop_lib, targets)
print("Optimizer created")
optimizer.optimize(comp_graph)
print_matching_final(comp_graph, optimizer.loc2match)

# print(get_input_network())

# data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
# data = relay.nn.relu(data)
# op, new_args, attrs, type_args, span = data.op, data.args, data.attrs, data.type_args, data.span
# data = tvm.relay.expr.Call(op, new_args, attrs, type_args, span)
# print(type(data))
# data.fused_group_id = 100
# print(repr(data))

# optimize_comp_graph(get_input_network())
# optimize_comp_graph(get_resnet_8())

# Note that this Relay repr is different from the optimized expr in the fused DP pass of TVM.
# mod, params, shape_dict, _ = get_network_from_onnx("resnet50", batch_size=1)
# optimize_comp_graph(mod["main"])