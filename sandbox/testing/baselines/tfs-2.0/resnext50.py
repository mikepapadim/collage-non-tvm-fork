import argparse
import tensorflow as tf
import numpy as np
import time
import timeit
from shared_functions import make_activation, make_conv2d, make_conv2d_bn

tf.config.run_functions_eagerly(False)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def resnext_block(input, strides, out_channels, groups, name):
    t = make_conv2d(input_tensor=input, filter_shape=(1,1,input.shape[1],out_channels), strides=(1,1,1,1), padding="SAME", actimode="RELU", name=name+"_conv1")
    t = tf.split(t, groups, axis=1, name=name+"_split")
    assert(len(t) == groups)
    for i in range(groups):
        t[i] = make_conv2d(input_tensor=t[i], filter_shape=(3,3,t[i].shape[1],out_channels//groups), strides=strides, padding="SAME", actimode="RELU", name=name+"_conv2_".format(i))
    output = tf.concat(t, axis=1, name=name+"_concat")
    t = make_conv2d(input_tensor=output, filter_shape=(1,1,output.shape[1],2*out_channels), strides=(1,1,1,1), padding="SAME", actimode="NONE", name=name+"_conv3")
    if (strides[2]>1) or (input.shape[1] != out_channels*2):
        input = make_conv2d(input_tensor=input, filter_shape=(1,1,input.shape[1],out_channels*2), strides=strides, padding="SAME", actimode="RELU", name=name+"_conv4")
    return tf.nn.relu(tf.add(input, t))

parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=5000)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
args = parser.parse_args()

@tf.function(experimental_compile=args.xla)
def resnext50(input):
    t = input
    for i in range(3):
        t = resnext_block(t, (1,1,1,1), 128, 32, "resnet_block_1_{}".format(i))
    strides=(1,1,2,2)
    for i in range(4):
        t = resnext_block(t, strides, 256, 32, "resnet_block_2_{}".format(i))
        strides=(1,1,1,1)
    strides=(1,1,2,2)
    for i in range(6):
        t = resnext_block(t, strides, 512, 32, "resnet_block_3_{}".format(i))
        strides=(1,1,1,1)
    strides=(1,1,2,2)
    for i in range(3):
        t = resnext_block(t, strides, 1024, 32, "resnet_block_4_{}".format(i))
        strides=(1,1,1,1)
    return t

times = []
for i in range(args.discard_iter + args.iterations):
    inputs = tf.constant(np.random.random_sample((1,64,56,56)).astype(np.float32))

    t0 = timeit.default_timer()
    resnext50(inputs)
    t1 = timeit.default_timer()

    times.append(t1 - t0)

total = 0
for i in range(args.discard_iter, len(times)):
    total += times[i]
avg = total / (args.iterations) * 1000.0
print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")

