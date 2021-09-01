import argparse
import tensorflow as tf
import numpy as np
import time
import torch
# import pretrainedmodels
# import tensorflow_hub as hub
from .shared_functions import make_activation, make_conv2d, make_seperable_conv2d, make_avgpool2d, make_maxpool2d

# tf.config.run_functions_eagerly(False)
#
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

def squeeze(out_channels, input):
    return make_conv2d(input_tensor=input, filter_shape=(1,1,input.shape[1],out_channels), strides=(1,1,1,1), padding="SAME", actimode="RELU", name="squeeze")

def fit(current, input):
    return squeeze(current.shape[1], input)

def normal_cell(prev, cur, out_channels):
    cur = squeeze(out_channels, cur)
    prev = fit(cur, prev)
    ts = list()
    ts.append(make_seperable_conv2d(input_tensor=cur, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(cur)
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=cur, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_avgpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(prev)
    ts.append(make_avgpool2d(input_tensor=prev, kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_avgpool2d(input_tensor=prev, kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    assert len(ts) == 10
    outputs=list()
    for i in range(5):
        outputs.append(tf.add(ts[2*i], ts[2*i+1]))
    return tf.concat(outputs, axis=1, name="concat1")

def reduction_cell(prev, cur, out_channels):
    cur = squeeze(out_channels, cur)
    prev = fit(cur, prev)
    ts = list()
    outputs = list()
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(7,7), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=cur, out_channels=out_channels, kernels=(5,5), strides=(1,1,2,2), padding="SAME"))
    outputs.append(tf.add(ts[0], ts[1]))
    ts.append(make_maxpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(7,7), strides=(1,1,2,2), padding="SAME"))
    outputs.append(tf.add(ts[2], ts[3]))
    ts.append(make_avgpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(5,5), strides=(1,1,2,2), padding="SAME"))
    outputs.append(tf.add(ts[4], ts[5]))
    ts.append(make_maxpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=outputs[0], out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    outputs.append(tf.add(ts[6], ts[7]))
    ts.append(make_avgpool2d(input_tensor=outputs[0], kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(outputs[1])
    outputs.append(tf.add(ts[8], ts[9]))
    return tf.concat(outputs, axis=1, name="concat2")

# parser = argparse.ArgumentParser()
# parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
# parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
# parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=200)
# parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
# args = parser.parse_args()

def nasneta_tf2_model(input0):
    input = input0
    out_channels = 64
    for i in range(3):
        if i > 0:
            input = reduction_cell(prev, cur, out_channels)
        prev = input
        cur = input
        for j in range(5):
            t = normal_cell(prev, cur, out_channels)
            prev = cur
            cur = t
        out_channels *= 2
    return cur


@tf.function(experimental_compile=False)
def nasneta_tf2(input0):
    return nasneta_tf2_model(input0)

@tf.function(experimental_compile=True)
def nasneta_tf2_xla(input0):
    return nasneta_tf2_model(input0)


# times = []
# for i in range(args.discard_iter + args.iterations):
#     inputs = tf.constant(np.random.random_sample((1,64,56,56)).astype(np.float32))
#
#     t0 = time.time()
#     nasneta(inputs)
#     t1 = time.time()
#     times.append(t1 - t0)
#
# total = 0
# for i in range(args.discard_iter, len(times)):
#     total += times[i]
# avg = total / (args.iterations) * 1000.0
# print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")
