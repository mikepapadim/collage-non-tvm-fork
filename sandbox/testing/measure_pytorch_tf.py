import torch.nn as nn
import torch
import math
import argparse
from tqdm import tqdm
from tvm.relay.transform.backend_operator.target import *

from workloads.torch_workloads import *
from workloads.workloads import *
from e2e_perf_logger import *
import tensorflow as tf
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
    parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=10000)
    parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=2000)
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


def measure_torch(model, inputs, args):
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
    E2EPerfLogger().log_perf(args.hw, args.network, 'PyTorch', mean_perf, std_perf)

# Note that inputs are numpy array because graph mode requires different tf.constant for every execution
def get_tf2_model_and_input(args, is_xla = False):
    if is_xla:
        model = NETWORK_TO_TF2_MODEL[args.network]
    else:
        model = NETWORK_TO_TF2_MODEL[args.network+"_xla"]

    input_shape = tuple(get_torch_input_data(args.network, args.batch_size).shape)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    return model, inputs

# Note that inputs are numpy array, not tf.constant
def measure_tf2(model, inputs, args, method_name):
    # Enable graph mode instead of eager execution
    # Graph mode performs better, so it is fairer to compare against ours
    tf.compat.v1.disable_eager_execution()
    tf.debugging.set_log_device_placement(True)

    with tf.device('/device:GPU:0'):
        times = []
        for i in tqdm(range(args.discard_iter + args.iterations)):
            t0 = time.time()
            model(tf.constant(inputs))
            t1 = time.time()
            times.append(t1 - t0)

    times = 1000.0 * np.array(times)[args.discard_iter:]
    mean_perf, std_perf = np.mean(times), np.std(times)
    print(f"[{args.network}] Performance of {method_name} on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    # E2EPerfLogger().log_perf(args.hw, args.network, method_name, mean_perf, std_perf)

if __name__ == '__main__':
    args = get_args()

    # PyTorch measurement
    model, inputs = get_torch_model_and_input(args)
    measure_torch(model, inputs, args)

    # Adjust experiment parameters for TF
    args.iterations = 5000
    args.discard_iter = 1000

    # TF2 measurement
    model, inputs = get_tf2_model_and_input(args, is_xla=False)
    measure_tf2(model, inputs, args, "TF")

    # TF2-XLA measurement
    model, inputs = get_tf2_model_and_input(args, is_xla=True)
    measure_tf2(model, inputs, args, "TF-XLA")