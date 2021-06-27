import tvm
import tvm.relay as relay
import tvm.relay.testing as testing
from graphviz import Digraph

# from ..workloads.onnx_workloads import get_network_from_onnx
from workloads.torch_workloads import get_network_from_torch
import argparse

from tvm.relay.transform.utility.visualize import visualize_network

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default type is string for argparse
    parser.add_argument("-n", "--network", help="name of a neural network")
    args = parser.parse_args()

    is_missing_arg = not args.network
    # is_missing_arg |= not args.target

    if is_missing_arg:
        parser.error('Make sure you input all arguments')



    # mod = get_resnet_8()
    # mod, _, _, _ = get_network_from_onnx(args.network, batch_size=1)
    mod, _, _, _ = get_network_from_torch(args.network, batch_size=1)
    visualize_network(mod["main"], args.network)






