import argparse
import tensorflow as tf
import numpy as np
import time
import timeit
from shared_functions import make_conv3d

tf.config.run_functions_eagerly(False)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def resnet_block(input, strides, out_channels, name):
    t = make_conv3d(input_tensor=input, filter_shape=(1,1,1,input.shape[1],out_channels), strides=(1,1,1,1,1), padding="SAME", actimode="RELU", name=name+"_conv1")
    t = make_conv3d(input_tensor=t, filter_shape=(3,3,3,out_channels,out_channels), strides=strides, padding="SAME", actimode="RELU", name=name+"_conv2")
    t = make_conv3d(input_tensor=t, filter_shape=(1,1,1,out_channels,out_channels*4), strides=(1,1,1,1,1), padding="SAME", actimode="NONE", name=name+"_conv3")
    if (strides[2]>1) or (input.shape[1] != out_channels * 4):
        input = make_conv3d(input_tensor=input, filter_shape=(1,1,1,input.shape[1],out_channels*4), strides=strides, padding="SAME", actimode="RELU", name=name+"_conv4")
    return tf.nn.relu(tf.add(input, t))

parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=5)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1)
args = parser.parse_args()

@tf.function(experimental_compile=args.xla)
def resnet50(input):
    t = input
    for i in range(3):
        t = resnet_block(t, (1,1,1,1,1), 64, "resnet_block_1_{}".format(i))
    strides=(1,1,2,2,1)
    for i in range(4):
        t = resnet_block(t, strides, 128, "resnet_block_2_{}".format(i))
        strides=(1,1,1,1,1)
    strides=(1,1,2,2,1)
    for i in range(6):
        t = resnet_block(t, strides, 256, "resnet_block_3_{}".format(i))
        strides=(1,1,1,1,1)
    strides=(1,1,2,2,1)
    for i in range(3):
        t = resnet_block(t, strides, 512, "resnet_block_4_{}".format(i))
        strides=(1,1,1,1,1)
    return t 

times = []
for i in range(args.discard_iter + args.iterations):
    inputs = tf.constant(np.random.random_sample((1,64,3,56,56)).astype(np.float32))

    t0 = timeit.default_timer()
    a = resnet50(inputs)
    t1 = timeit.default_timer()
    times.append(t1 - t0)

    print(a.numpy().shape)

total = 0
for i in range(args.discard_iter, len(times)):
    total += times[i]
avg = total / (args.iterations) * 1000.0
print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")

