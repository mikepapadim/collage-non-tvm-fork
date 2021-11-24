from workloads.torch_workloads import get_network_from_torch
import numpy as np
import collage
import tvm
import logging
from tvm.contrib import graph_executor as runtime

# [NOTE]
# * Operator cost will be logged at "operator_cost.log". 
#   Since it is unreadable, Collage also dump human-readable form at "operator_cost.json"
# * Available networks: bert_full, dcgan, nasneta, resnet50_3d, resnext50_32x4d
# * Collage supports following backends by default:
#      NVIDIDIA GPUs - TVM, TensorRT, cuBLAS, cuDNN
#      Intel CPUs    - TVM, MKL, DNNL
# * For the best performance, TVM operators should be tuned with AutoTVM beforehand. 
#   Collage assuems its log is prepared at "autotvm_tuning_log.json"
#   This demo provides tuning logs for RTX2070.
# * Collage offers two optimizers: "op-level", "two-level"


# Define Collage workload
workload = {
    "optimizer": "op-level",
    "backends": ["AutoTVM", "cuDNN", "TensorRT", "cuBLAS"],
    "network_name": "dcgan",
    "device": "rtx2070",
    "target": "cuda",
    "batch_size": 1,
}

# Enable logging to skip messages during optimization. Comment this out to disable logging. 
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    collage_mod = collage.Module()
    network_name, batch_size, target, device = \
          workload["network_name"], workload["batch_size"], workload["target"], workload["device"]

    mod, params, shape_dict, _ = get_network_from_torch(network_name, batch_size)
    # Since Collage utilizes tvm as its codegen, we need to pass the following info for tvm codegen. 
    workload["mod"] = mod
    workload["params"] = params
    
    # Invoke collage optimizer
    lib = collage_mod.optimize_backend_placement(**workload)

    # Create workload
    dev = tvm.device(target, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    # Measure performance
    ftimer = module.module.time_evaluator("run", dev, number=20, repeat=20)
    perfs = np.array(ftimer().results) * 1000
    mean_perf, std_perf = np.mean(perfs), np.std(perfs)
    print(f"[{network_name}] Performance of DP on {device} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
