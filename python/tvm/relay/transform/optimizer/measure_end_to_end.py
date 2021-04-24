from tvm import relay
import tvm
from ..backend_operator.utils import is_function_node
from ..workloads.torch_workloads import get_network_from_torch
from ..backend_operator.target import measure, NUM_MEASUREMENTS_PER_REPEAT, NUM_REPEATS, AUTOTVM_LOG
from tvm.contrib import graph_executor as runtime
import numpy as np

from tvm import autotvm

def measure_end_to_end_perf_autotvm(net, params, target_str, shape_dict, is_ours):
    assert is_function_node(net)
    if is_ours:
        net = net.with_attr("CustomFusionPass", 1)

    with autotvm.apply_history_best(AUTOTVM_LOG):
        # FIXME(@Soo): We should redesign Target class to deal with new TVM build interface
        opt_level = 2
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(net, "cuda", params=params)

        # Create workload
        dev = tvm.device(target_str, 0)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        for input_name, input_shape in shape_dict.items():
            input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
            module.set_input(input_name, input_data)

        ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

    return measure(ftimer)

if __name__ == "__main__":
    # We can't test this because this network include batch norm.
    network_names = ["resnet50", "resnext50_32x4d", "bert", "nasrnn", "nasneta"]
    net_name = network_names[4]
    mod, params, shape_dict, _ = get_network_from_torch(net_name, 1)

    mean_perf, std_perf = measure_end_to_end_perf_autotvm(mod["main"], params, 'cuda -libs=cudnn', shape_dict, False)
    print(f"[AutoTVM+CuDNN] Performance of {net_name} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")

    mean_perf, std_perf = measure_end_to_end_perf_autotvm(mod["main"], params, 'cuda', shape_dict, False)
    print(f"[AutoTVM] Performance of {net_name} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")

    mean_perf, std_perf = measure_end_to_end_perf_autotvm(mod["main"], params, 'cuda', shape_dict, True)
    print(f"[Ours] Performance of {net_name} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")


