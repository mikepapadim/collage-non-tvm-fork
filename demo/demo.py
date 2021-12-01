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


# [TODO] 
# "resnext50_32x4d", "resnet50_3d", "bert_full"
#     falls into infinite loop (e.g.,  build_with_pattern_with_map )
# "dcgan": performance bug. 2.2ms vs 2.0 ms
# "nasneta":   File "/home/sunggg/collage/python/collage/pattern_manager/default_pattern_rules.py", line 57, in generate_relay_pattern_node
#      return is_tuple(), len(node.fields)
#      TypeError: is_tuple() missing 1 required positional argument: 'fields'
# "bert_full"

# Try trt pass
# try cpu
# try w/o pattern gen
# Define Collage workload
workload = {
    "optimizer": "op-level",
    "backends": ["autotvm", "cudnn", "tensorrt", "cublas"],
    "network_name": "dcgan", 
    "target": "cuda",
    "batch_size": 1,
}

# Enable logging to skip messages during optimization. Comment this out to disable logging. 
logging.basicConfig(level=logging.INFO)

def measure_perf(lib, target):
    # Create workload
    dev = tvm.device(target, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    # Measure performance
    ftimer = module.module.time_evaluator("run", dev, number=10, repeat=20)
    perfs = np.array(ftimer().results) * 1000
    return np.mean(perfs), np.std(perfs)

def build_with_tensorrt():
    assert 0, "Need to implement"


if __name__ == "__main__":
    collage_mod = collage.Module()
    print(f"Default backends: {collage_mod.get_registered_backends()}")

    def cg_empty():
        pass
    def custom_cost_func():
        pass
    #collage_mod.register_new_backend("TestBackend", collage.BackendKind.OP_LEVEL, cg_empty, cost_func=custom_cost_func, log="test.log")
    #print(f"Default backends: {collage_mod.get_registered_backends()}")

    # this works
    #collage_mod.backend_registry["TVM"].kwargs["tuning_log"] = "test.txt" 

    network_name, batch_size, target = \
          workload["network_name"], workload["batch_size"], workload["target"]

    mod, params, shape_dict, _ = get_network_from_torch(network_name, batch_size)
    # Since Collage utilizes tvm as its codegen, we need to pass the following info for tvm codegen. 
    workload["mod"] = mod
    workload["params"] = params

    # Invoke collage optimizer
    lib = collage_mod.optimize_backend_placement(**workload)

    mean_perf, std_perf = measure_perf(lib, target)
    print(f"[{network_name}] Performance of DP (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
