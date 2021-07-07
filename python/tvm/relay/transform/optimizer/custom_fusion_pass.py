from enum import IntEnum
from ..backend_operator.target import *

#import gc

CONFIG_VAR_USER_DEFINED_FUSION_PASS = "relay.FuseOps.UserDefinedFusion"

class CustomFusionPass(IntEnum):
    # This is for measurement
    USER_DEFINED_FUSION = 0
    DP = 1
    EXHAUSTIVE_SEARCH = 2
    TWO_LEVEL_OPT = 3

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

def measure_end_to_end_user_defined(net, params, shape_dict, target_str, net_name):
    assert is_function_node(net)

    # print(f"[measure_end_to_end_user_defined] User-defined fusion (ID: {CustomFusionPass.USER_DEFINED_FUSION})")
    net = net.with_attr("CustomFusionPass", CustomFusionPass.USER_DEFINED_FUSION)
    net = net.with_attr("NetworkName", net_name)

    # printe(f"End-To-End measure")
    # printe(f"OPT LEVEL : {OPT_LEVEL.get()}")
    with autotvm.apply_history_best(AUTOTVM_LOG):
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build(net, target_str, params=params)

        #print(f"[measure_end_to_end_user_defined] We successfully built the network")
        # Create workload
        dev = tvm.device(target_str, 0)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        for input_name, input_shape in shape_dict.items():
            input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
            module.set_input(input_name, input_data)

        ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

    perf, std = measure(ftimer, is_net=True)

    #del dev
    #del lib
    #del module
    #del ftimer
    #del input_name
    #del input_shape
    #gc.collect()

    return perf, std
