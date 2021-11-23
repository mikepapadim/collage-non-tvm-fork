from workloads.torch_workloads import get_network_from_torch
import numpy as np
import collage
import tvm
import logging
from tvm.contrib import graph_executor as runtime

# measurement configs
# backend configs
# logs

# [TODO] Check with soo
# 1. attrbutes
# 2. how to pass backend 
# 3. Ev search customization
    
workload = {
        "optimizer": "two-level",
        "backends": [],
        "network_name": "bert",
        "device": "rtx2070",
        "target": "cuda",
        "batch_size": 1,
        "autotvm_tuning_log": "./logs/autotvm_ops_rtx2070.json",
}


if __name__ == "__main__":
    collage_mod = collage.Module()
    network_name, batch_size, target, device = \
          workload["network_name"], workload["batch_size"], workload["target"], workload["device"]

    mod, params, shape_dict, _ = get_network_from_torch(network_name, batch_size)
    workload["mod"] = mod
    workload["params"] = params
    
    logging.basicConfig(level=logging.INFO)
    lib = collage_mod.optimize_backend_placement(**workload)

    # Create workload
    dev = tvm.device(target, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator("run", dev, number=20, repeat=20)
    perfs = np.array(ftimer().results) * 1000
    mean_perf, std_perf = np.mean(perfs), np.std(perfs)
    print(f"[{network_name}] Performance of DP on {device} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
