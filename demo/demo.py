from workloads.torch_workloads import get_network_from_torch
import numpy as np
import collage
import tvm
import logging
from tvm.contrib import graph_executor as runtime
from collage.analysis.visualize import visualize_network


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
# DP
#    "resnext50_32x4d", "resnet50_3d", "bert_full", "dcgan": performance bug
#    "nasneta": non-call node is somehow inserted to frontier q.
#     -- collage/optimizer/comp_graph_optimizer.py
#     -- certain ops are not saved in cost logger. e.g., TRT dense in DCGAN
# EV
#    Pass Collage context info to subprocess
#    Best placement dump
#    Reload best placement
#
# Demo
#   - Test autoscheduler
 
# Define Collage workload
workload = {
    "optimizer": "op-level", #"two-level", 
    "backends": ["autotvm", "cudnn", "cublas", "tensorrt"],
    "network_name": "resnet50_3d", #"nasneta", #"bert_full", #"dcgan", #"resnext50_32x4d", "resnet50_3d", 
    "target": "cuda",
    "batch_size": 1,
}

# Enable logging to skip messages during optimization. Comment this out to disable logging. 
logging.basicConfig(level=logging.INFO)

def measure_perf(lib, workload):
    # Create workload
    dev = tvm.device(workload["target"], 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in workload["shape_dict"].items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    # Measure performance
    ftimer = module.module.time_evaluator("run", dev, number=10, repeat=20)
    perfs = np.array(ftimer().results) * 1000
    return np.mean(perfs), np.std(perfs)

def run_with_tensorrt(workload):
    from collage.backend.default_backends import cg_TensorRT
    lib = cg_TensorRT(workload["mod"], workload["target"], workload["params"])
    return measure_perf(lib, workload)


def setup_workload(workload):
    network_name, batch_size, target = \
          workload["network_name"], workload["batch_size"], workload["target"]

    mod, params, shape_dict, _ = get_network_from_torch(network_name, batch_size)
    # Since Collage utilizes tvm as its codegen, we need to pass the following info for tvm codegen. 
    workload["mod"] = mod
    workload["params"] = params
    workload["shape_dict"] = shape_dict


if __name__ == "__main__":
    setup_workload(workload)

    trt_mean_perf, trt_std_perf = run_with_tensorrt(workload)

    collage_mod = collage.Module()
    print(f"Default backends: {collage_mod.get_registered_backends()}")

    def cg_empty():
        pass
    def custom_cost_func():
        pass
    #collage_mod.register_new_backend("TestBackend", collage.BackendKind.OP_LEVEL, cg_empty, cost_func=custom_cost_func, log="test.log")
    #print(f"Default backends: {collage_mod.get_registered_backends()}")

    # Override the default tuning log
    collage_mod.update_autotvm_tuning_log("autotvm_tuning_log_rtx2070.json")

    # Invoke collage optimizer
    lib = collage_mod.optimize_backend_placement(**workload)    
    # visualize_network(lib.ir_mod["main"], "collage_final_placement")  # This does not reflect our placement
    collage_mean_perf, collage_std_perf = measure_perf(lib, workload)
   

    print(f"Network: {workload['network_name']}")
    print(f"  Run with TensorRT (mean, std) = ({trt_mean_perf:.4f}+-{trt_std_perf:.4f})")
    print(f"  Run with Collage  (mean, std) = ({collage_mean_perf:.4f}+-{collage_std_perf:.4f})")
    print(f"  -> Speedup: {trt_mean_perf/collage_mean_perf:.4f}x")

    # Visualize backend placement optimized by Collage
    workload["input_placement_log_file"] = workload["op_level_placement_log"]
    # workload["input_placement_log_file"] = workload["graph_level_placement_log"]
    workload["placement_vis_file"] = "op_level_placement_vis"
    # workload["placement_vis_file"] = "graph_level_placement_vis"
    collage_mod.visualize_backend_placement(**workload)