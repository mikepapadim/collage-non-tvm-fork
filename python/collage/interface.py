import tvm
from tvm import relay, autotvm
import logging
import numpy as np
from tvm.contrib import graph_executor as runtime
from collage.pattern_manager.utils import is_function_node
from collage.optimizer.custom_fusion_pass import CustomFusionPass
from collage.pattern_manager.cost_func import (
    NETWORK_FUNC_ATTR, HW_FUNC_ATTR, BATCH_SIZE_ATTR, 
    NUM_MEASUREMENTS_PER_REPEAT_E2E, NUM_REPEATS_E2E,
    get_autotvm_log_path, measure
)

# AutoTVM tuning log
# logs
# backends
#    - pattern
#    - pattern rule
#    - pattern engine
# run_dp, run_two_lv


def build_and_measure_autotvm(net, params, target_str, shape_dict, hw_name):
    with autotvm.apply_history_best(get_autotvm_log_path(hw_name)):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(net, target_str, params=params)
        logging.info(f"We successfully built the network")
        # Create workload
        dev = tvm.device(target_str, 0)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        for input_name, input_shape in shape_dict.items():
            input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
            module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT_E2E, repeat=NUM_REPEATS_E2E)
    mean_perf, std_perf = measure(ftimer, True, hw_name)

    return mean_perf, std_perf, module


def measure_end_to_end_perf_autotvm(net, params, target_str, shape_dict, method_mode, net_name, hw_name, batch_size):
    assert is_function_node(net)

    if method_mode is not None:
        net = net.with_attr("CustomFusionPass", method_mode)
        net = net.with_attr(NETWORK_FUNC_ATTR, net_name)
        net = net.with_attr(HW_FUNC_ATTR, hw_name)
        net = net.with_attr(BATCH_SIZE_ATTR, batch_size)

    return build_and_measure_autotvm(net, params, target_str, shape_dict, hw_name)


class Collage:
    def __init__(self):
        self.pattern_registry = None

    def add_backend_pattern(self):
        assert 0, "Need to implement"

    def add_backend_pattern_rule(self):
        assert 0, "Need to implement"

    def add_pattern_generator(self):
        assert 0, "Need to implement"

    def add_backend_codegen(self):
        assert 0, "Need to implement"

    def optimize_backend_placement(self, mod, target, params, shape_dict, net_name, hw_name, batch_size, backends=None, method=None):
        
        mean_perf, std_perf, mod_dp = measure_end_to_end_perf_autotvm(mod["main"], params, target, shape_dict,
                                                                 CustomFusionPass.DP,
                                                                 net_name, hw_name, batch_size)
        print(f"[{net_name}] Performance of DP on {hw_name} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    