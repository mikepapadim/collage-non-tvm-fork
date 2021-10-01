import torch.nn as nn
import torch
import math
import argparse
from tqdm import tqdm
from tvm.relay.transform.backend_operator.target import *

from workloads.torch_workloads import *
from workloads.workloads import *
# import tensorflow as tf
# import time
from measure_end_to_end import log_e2e_perf
import time

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

    # Measurement related parameters
    # parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=10000)
    # parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=2000)
    parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=1000)
    parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=100)

    args = parser.parse_args()

    args_checker(args, parser)
    return args

# def measure_trt():
#     from torch2trt import torch2trt
#     import time
#
#     model_trt = torch2trt(model, [inputs])
#
#     times = []
#     for i in tqdm(range(args.discard_iter + args.iterations)):
#         torch.cuda.current_stream().synchronize()
#         t0 = time.time()
#         model_trt(inputs)
#         torch.cuda.current_stream().synchronize()
#         t1 = time.time()
#         times.append(1000.0 * (t1 - t0))
#
#     total = 0
#     for i in range(args.discard_iter, len(times)):
#         total += times[i]
#     avg = total / (args.iterations)
#     print("TensorRT: Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")

def get_torch_model_and_input(args):
    model = NETWORK_TO_TORCH_MODEL[args.network]()
    inputs = get_torch_input_data(args.network, args.batch_size)

    if get_build_target(args.hw) == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    return model, inputs


def measure_torch(model, inputs, args, is_perf_logging):
    times = []
    with torch.no_grad():
        for i in tqdm(range(args.discard_iter + args.iterations)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            model(inputs)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            times.append(start.elapsed_time(end))

    times = np.array(times)[args.discard_iter:]
    mean_perf, std_perf = np.mean(times), np.std(times)
    print(f"[{args.network}] Performance of PyTorch on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_e2e_perf(args, 'PyTorch', mean_perf, std_perf, is_perf_logging)

# Link: https://discuss.pytorch.org/t/how-to-measure-execution-time-in-pytorch/111458
# Note: CPU operations are synchronous; you can use any Python runtime profiling method like time.time().
def measure_torch_cpu(model, inputs, args, is_perf_logging):
    times = []
    with torch.no_grad():
        for i in tqdm(range(args.discard_iter + args.iterations)):
            start_time = time.time()
            model(inputs)
            times.append((time.time() - start_time)*1000.0)

    times = np.array(times)[args.discard_iter:]
    mean_perf, std_perf = np.mean(times), np.std(times)
    print(f"[{args.network}] Performance of PyTorch on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_e2e_perf(args, 'PyTorch', mean_perf, std_perf, is_perf_logging)

#######################################################################
# Warning(@Soo): Deprecated; It turns out that we need to run model code as a main (e.g., bert.py)
# not to get an error for BERT and DCGAN.
#######################################################################

# # Note that inputs are numpy array because graph mode requires different tf.constant for every execution
# def get_tf2_model_and_input(args, is_xla = False):
#     if not is_xla:
#         model = NETWORK_TO_TF2_MODEL[args.network]
#     else:
#         model = NETWORK_TO_TF2_MODEL[args.network+"_xla"]
#
#     input_shape = tuple(get_torch_input_data(args.network, args.batch_size).shape)
#     inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")
#
#     return model, inputs
#
# # Note that inputs are numpy array, not tf.constant
# def measure_tf2(model, inputs, args, method_name, is_perf_logging):
#     # Enable graph mode instead of eager execution
#     # Graph mode performs better, so it is fairer to compare against ours
#     tf.config.run_functions_eagerly(False)
#
#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = tf.compat.v1.Session(config=config)
#
#     # Warning(@Soo): It has an issue of executing CPU only for TF2.4.0 or TF2.6.0
#     # tf.compat.v1.disable_eager_execution()
#
#     with tf.device('/device:GPU:0'):
#     # with tf.device('/device:CPU:0'):
#     # with tf.device('/physical_device:GPU:0'):
#         times = []
#         for i in tqdm(range(args.discard_iter + args.iterations)):
#             t0 = time.time()
#             run_model(inputs)
#             t1 = time.time()
#             times.append(t1 - t0)
#
#     times = 1000.0 * np.array(times)[args.discard_iter:]
#     mean_perf, std_perf = np.mean(times), np.std(times)
#     print(f"[{args.network}] Performance of {method_name} on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
#     log_e2e_perf(args, 'PyTorch', mean_perf, std_perf, is_perf_logging)

if __name__ == '__main__':
    args = get_args()

    is_perf_logging = True

    # PyTorch measurement
    model, inputs = get_torch_model_and_input(args)
    if args.hw in NVIDIA_GPUS:
        measure_torch(model, inputs, args, is_perf_logging)
    elif args.hw in INTEL_CPUS:
        measure_torch_cpu(model, inputs, args, is_perf_logging)
    else:
        raise Exception(f"{args.hw} is unexpected hw, we need to set default backends for this hw.")

    #######################################################################
    # Warning(@Soo): Deprecated; It turns out that we need to run model code as a main (e.g., bert.py)
    # not to get an error for BERT and DCGAN.
    #######################################################################

    # Adjust experiment parameters for TF
    # args.iterations = 100
    # args.discard_iter = 10

    # Check if TF2 is using GPU
    # tf.debugging.set_log_device_placement(True)

    # is_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    # print(tf.config.list_physical_devices('GPU'))
    # print(tf.config.list_physical_devices())

    # TF2 measurement
    # model, inputs = get_tf2_model_and_input(args, is_xla=False)
    # measure_tf2(model, inputs, args, "TF", is_perf_logging)

    # TF2-XLA measurement
    # model, inputs = get_tf2_model_and_input(args, is_xla=True)
    # measure_tf2(model, inputs, args, "TF-XLA", is_perf_logging)
