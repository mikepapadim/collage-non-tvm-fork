from tvm import relay
import tvm
from tvm.collage.optimizer.custom_fusion_pass import *
from workloads.torch_workloads import get_network_from_torch
from tvm import autotvm
from tvm.relay.transform.utility.debug_helper import *
from workloads.torch_workloads import *
from tvm.relay.transform.pattern_manager.target import *

import os

from measure_end_to_end import setup_attrs_ours, get_args


def plot_backend_placement(net, params, target_str, shape_dict, net_name, hw_name, batch_size):
    # Copy best backend placement
    opt_info_tag = get_opt_info_tag(net_name, hw_name, batch_size)
    source_match_path = f"{EVAL_RESULT_LOG_PATH}/{hw_name}_bs{batch_size}/best_match_{opt_info_tag}.log"
    target_match_path = get_user_defined_match_path(net_name, hw_name, batch_size)
    os.system(f"cp {source_match_path} {target_match_path}")

    net = net.with_attr("CustomFusionPass", CustomFusionPass.USER_DEFINED_FUSION)
    net = setup_attrs_ours(net, net_name, hw_name, batch_size)

    with autotvm.apply_history_best(get_autotvm_log_path(hw_name)):
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build(net, target_str, params=params)

    print(f"We successfully built the network")

if __name__ == "__main__":
    args = get_args()

    mod, params, shape_dict, _ = get_network_from_torch(args.network, args.batch_size)
    # Assign build target based on a given hw

    # Make sure you uncomment visualization code in "get_user_fusion" function in _optimizer.py
    args.target = get_build_target(args.hw)
    plot_backend_placement(mod["main"], params, args.target, shape_dict, args.network, args.hw, args.batch_size)



