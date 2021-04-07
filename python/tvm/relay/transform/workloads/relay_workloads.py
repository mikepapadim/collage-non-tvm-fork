import tvm
from tvm import relay
import tvm.relay.testing
import tensorflow as tf
import tvm.relay.testing.tf as tf_testing

import sys

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

#################################################################
# Define Network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.


def get_network_from_relay(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    dtype = 'float32'
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    
    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    elif name == "bert":
        # load bert protobuf

        # sym, params = relay.frontend.from_tensorflow(output_nodes)
        #config = tf.ConfigProto()
        #with tf.Session(config=config) as sess:
        model_path = '../autotune-rtx2070/tf_models/models_pb/bert'

        with tf_compat_v1.Session() as sess:

            meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], model_path)
            graph_def = meta_graph_def.graph_def
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)

            # Bert from TASO doesn't include softmax
            # graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")

        layout = None
        input_shape = (64, 1024)
        shape_dict = {"DecodeJpeg/contents": input_shape}
        mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)
        print("Compelete")

    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape