import tvm.relay.testing as testing
from tvm import relay
from ..backend_operator.target import Target
from tvm.contrib import graph_runtime as runtime
import tvm
from ..backend_operator.utils import is_function_node
from ..workloads.onnx_workloads import get_network_from_onnx
from ..workloads.torch_workloads import get_network_from_torch

def get_concat():
    data1 = relay.var("data1", relay.TensorType((1, 3, 224, 224), "float32"))
    data2 = relay.var("data2", relay.TensorType((1, 3, 224, 224), "float32"))
    data3 = relay.var("data3", relay.TensorType((1, 3, 224, 224), "float32"))
    data4 = relay.var("data4", relay.TensorType((1, 3, 224, 224), "float32"))
    tup = relay.Tuple([data1, data2, data3, data4])
    return tup
    #concat = relay.concatenate(tup, axis=1)
    #return concat

def get_conv():
    # Chain graph
    out_channels = 16
    batch_size = 1

    data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
    conv_weight = relay.var("weight", relay.TensorType((out_channels, 3, 3, 3), "float32"))
    # bn_gamma = relay.var("bn_gamma")
    # bn_beta = relay.var("bn_beta")
    # bn_mmean = relay.var("bn_mean")
    # bn_mvar = relay.var("bn_var")

    # simple_net = relay.nn.relu(data)
    # simple_net = relay.nn.relu(simple_net)
    simple_net = relay.nn.conv2d(
        data=data, weight=conv_weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    )

    return simple_net

def get_chain_graph():
    # Chain graph
    out_channels = 16
    batch_size = 1

    data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
    conv_weight = relay.var("weight", relay.TensorType((out_channels, 3, 3, 3), "float32"))
    dense_weight = relay.var("weight", relay.TensorType((10, 802816), "float32"))
    # bn_gamma = relay.var("bn_gamma")
    # bn_beta = relay.var("bn_beta")
    # bn_mmean = relay.var("bn_mean")
    # bn_mvar = relay.var("bn_var")

    # simple_net = relay.nn.relu(data)
    # simple_net = relay.nn.relu(simple_net)
    simple_net = relay.nn.conv2d(
        data=data, weight=conv_weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    )
    simple_net = relay.nn.batch_flatten(simple_net)
    # simple_net = relay.nn.dense(simple_net, dense_weight)
    # simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    # simple_net = relay.nn.relu(simple_net)
    # simple_net = relay.nn.relu(simple_net)

    expr = simple_net
    target = Target.TVM_GPU_NO_TUNING

    inputs = relay.analysis.free_vars(expr)
    expr_func = relay.Function(inputs, expr)
    net, params = testing.create_workload(expr_func)

    return net, params


def get_resnet_8():
    batch_size = 1
    num_class = 1000
    # Resnet-8
    image_shape = (3, 28, 28)

    # ResNet-2
    # mod, params = testing.resnet.get_workload(num_layers=2, batch_size=batch_size, image_shape=image_shape)

    # Resnet-8
    mod, params = testing.resnet.get_workload(num_layers=8, batch_size=batch_size, image_shape=image_shape)
    # expr_func = mod["main"]
    # net, params = testing.create_workload(expr_func)

    return mod, params

def build_network(net, params):
    assert is_function_node(net)
    net = net.with_attr("CustomFusionPass", 1)
    # target = Target.TVM_GPU_NO_TUNING

    # FIXME(@Soo): We should redesign Target class to deal with new TVM build interface
    opt_level = 2
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(net, "cuda", params=params)

# build_network(get_chain_graph())

# We can't test this because this network include batch norm.
# mod, params = get_resnet_8()
# build_network(mod["main"], params)

# network_name = "resnet50"
# mod, params, _, _ = get_network_from_onnx(network_name, batch_size=1)
# mod, params, _, _ = get_network_from_torch("resnet_block", 1)
# mod, params, _, _ = get_network_from_torch("resnet50", 1)
# mod, params, _, _ = get_network_from_torch("resnext50_32x4d",1)
# mod, params, _, _ = get_network_from_torch("bert",1)
# mod, params, _, _ = get_network_from_torch("nasrnn",1)

# print(get_concat().attrs.axis)

from tvm.relay.dataflow_pattern import *
print(is_tuple([wildcard(), wildcard(), wildcard(), wildcard()]).match(get_concat()))

mod, params, _, _ = get_network_from_torch("nasneta",1)
build_network(mod["main"], params)
