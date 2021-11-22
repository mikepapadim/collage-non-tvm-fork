from tvm.relay.transform.pattern_manager.target import *
from tvm.relay.transform.optimizer.custom_fusion_pass import CustomFusionPass
from tvm.contrib import graph_executor as runtime
import argparse
from tvm import autotvm
from workloads.torch_workloads import *
from tvm.relay.transform.utility.debug_helper import *
from backend_perf_logger import *

def args_checker(args, parser):
    is_missing_arg = not args.network
    is_missing_arg |= not args.hw
    # is_missing_arg |= not args.batch_size
    # is_missing_arg |= not args.target
    # is_missing_arg |= not args.dtype

    if is_missing_arg:
        parser.error('Make sure you input all arguments')

def get_args():
    parser = argparse.ArgumentParser()
    # Default type is string for argparse
    parser.add_argument("-n", "--network", help="name of a neural network")
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    # parser.add_argument("-t", "--target", help="target device")
    # parser.add_argument("-dt", "--dtype", help="data type")

    args = parser.parse_args()

    args_checker(args, parser)
    return args

def backend_list_to_str(backends):
    backend_str_list = [b.name() for b in backends]
    return '-'.join(backend_str_list)

def measure_dp(net, params, shape_dict, args, is_perf_logging, backends):
    # Set up attributes
    net = net.with_attr("CustomFusionPass", CustomFusionPass.DP)
    net = net.with_attr(NETWORK_FUNC_ATTR, args.network)
    net = net.with_attr(HW_FUNC_ATTR, args.hw)
    net = net.with_attr(BATCH_SIZE_ATTR, args.batch_size)

    # Add backend attributes
    backend_id_list_str = [str(backend.id()) for backend in backends]
    backend_id_list_str = ",".join(backend_id_list_str)
    net = net.with_attr(BACKEND_LIST_ATTR, backend_id_list_str)

    if Target.AUTOTVM in backends:
        with autotvm.apply_history_best(get_autotvm_log_path(args.hw)):
            with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
                lib = relay.build(net, args.target, params=params)
    # If AutoTVM is not included, then use TVM fallback options
    else:
        with tvm.transform.PassContext(opt_level=OPT_LEVEL.get()):
            lib = relay.build(net, args.target, params=params)

    logging.info(f"We successfully built the network")
    # Create workload
    dev = tvm.device(args.target, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT_E2E, repeat=NUM_REPEATS_E2E)
    mean_perf, std_perf = measure(ftimer, True, args.hw)

    print(f"[{args.network}] Performance on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    if is_perf_logging:
        DPBackendPerfLogger().log_perf(args.hw, int(args.batch_size), args.network, backend_list_to_str(backends),
                                       f"{mean_perf:.4f}", f"{std_perf:.4f}")

if __name__ == "__main__":
    args = get_args()

    # Redirect output to log files
    log_dir = "e2e_measure_logs"

    # For DP,
    # setup_logging(log_dir, task_name="e2e_measure", net_name=args.network, hw_name=args.hw, batch_size=args.batch_size,
    #              logging_level=logging.INFO)
                  # logging_level=logging.WARNING)

    # For tuning time measurement, comment setup_logging above and uncomment the following codes
    logging.basicConfig(level=logging.ERROR)

    # It shows all logs. Still, it is too messy though cuz TVM logs are interrupting with our logs
    #logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.WARNING)

    # We can't test this because this network include batch norm.
    logging.info(f"batch size: {args.batch_size}")

    mod, params, shape_dict, _ = get_network_from_torch(args.network, args.batch_size)

    # Assign build target based on a given hw
    args.target = get_build_target(args.hw)
    is_perf_logging = True
    # is_perf_logging = False

    # Problem: What if TVM is not included in backends?
    # Solution(@Soo): Still match patterns of TVM, but with infinite cost
    # All cases should be fine with this solution.

    # Single backend case is all fine - use single_backend pipeline that offloads unsupported ops to TVM
    # Two backend cases
    # - Incomplete op coverages: (cuDNN, TensorRT), (cuDNN, cuBLAS), (cuBLAS, TensorRT)
    # Three backend cases
    # - Incomplete op coverages: (cuDNN, TensorRT, cuBLAS)

    import itertools
    backend_combinations = [[Target.AUTOTVM],
                            [Target.AUTOTVM, Target.CUBLAS],
                            [Target.AUTOTVM, Target.CUBLAS, Target.CUDNN],
                            [Target.AUTOTVM, Target.CUBLAS, Target.CUDNN, Target.TENSORRT]]

    for subset in backend_combinations:
        measure_dp(mod["main"], params, shape_dict, args, is_perf_logging, subset)

    # backends = [Target.CUBLAS,Target.CUDNN,Target.TENSORRT,Target.AUTOTVM]
    # Measure all possible combinations of backends
    #for L in range(1, len(backends) + 1):
    #    for subset in itertools.combinations(backends, L):
    #        measure_dp(mod["main"], params, shape_dict, args, is_perf_logging, subset)

    # Perf plot order: cuBLAS, cuDNN, TensorRT, AutoTVM (in the increasing order of op coverage)
    # It shows perf of cuBLAS, cuBLAS+cuDNN, cuBLAS+cuDNN+TensorRT, and all together
    # Or Perf plot order: AutoTVM, TensorRT, cuDNN, cuBLAS (in the decreasing order of op coverage)
