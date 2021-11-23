import argparse
from workloads.torch_workloads import get_network_from_torch
import logging

import collage
from collage import get_build_target

def get_args():
    parser = argparse.ArgumentParser()
    # Default type is string for argparse
    parser.add_argument("-n", "--network", help="name of a neural network")
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    assert(args.network and args.hw)
    return args



if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"network: {args.network}, batch size: {args.batch_size}, hardware: {args.hw}")
    
    target = get_build_target(args.hw)
    collage_mod = collage.Collage()
    mod, params, shape_dict, _ = get_network_from_torch(args.network, args.batch_size)

    collage_mod.optimize_backend_placement(mod, target, params, shape_dict, args.network, args.hw, args.batch_size)