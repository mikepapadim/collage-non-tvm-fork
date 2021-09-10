import argparse
import tensorflow as tf
import numpy as np
import time
from .shared_functions import make_matmul

# tf.config.run_functions_eagerly(False)
#
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

# hidden_size = 512
hidden_size = 2560
length = 5

def combine(x, h):
    w1 = make_matmul(x, hidden_size)
    w2 = make_matmul(h, hidden_size)
    return tf.add(tf.nn.relu(w1), tf.nn.relu(w2))

def nas_node(input, x):
    t = [combine(x, input) for i in range(8)]
    midt = list()
    midt.append(tf.add(tf.nn.relu(t[0]), tf.nn.sigmoid(t[3])))
    midt.append(tf.add(tf.nn.sigmoid(t[1]), tf.nn.tanh(t[2])))
    midt.append(tf.multiply(tf.nn.sigmoid(t[4]), tf.nn.tanh(t[5])))
    midt.append(tf.multiply(tf.nn.sigmoid(t[6]), tf.nn.relu(t[7])))
    midt.append(tf.add(tf.nn.sigmoid(midt[1]), tf.nn.tanh(midt[2])))
    midt.append(tf.multiply(tf.nn.tanh(midt[0]), tf.nn.tanh(midt[3])))
    midt.append(tf.multiply(tf.nn.tanh(midt[4]), tf.nn.tanh(midt[5])))
    return tf.nn.tanh(midt[6])

# parser = argparse.ArgumentParser()
# parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
# parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
# parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=5000)
# parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
# args = parser.parse_args()

def nasrnn_tf2_model(xs):
    input_dictionary = {}
    state = tf.constant(np.random.random_sample((1, hidden_size)), dtype=tf.float32)
    for i in range(length):
        state = nas_node(state, xs)#[i])
    return state

# @tf.function(jit_compile=False)
@tf.function(experimental_compile=False)
def nasrnn_tf2(xs):
    return nasrnn_tf2_model(xs)

# @tf.function(jit_compile=True)
@tf.function(experimental_compile=True)
def nasrnn_tf2_xla(xs):
    return nasrnn_tf2_model(xs)

# times = []
# for i in range(args.discard_iter + args.iterations):
#     xs = []
#     for i in range(length):
#         xs.append(
#             tf.constant(np.random.random_sample((1,hidden_size)).astype(np.float32))
#         )
#     t0 = time.time()
#     out = nasrnn(xs)
#     t1 = time.time()
#     times.append(t1 - t0)
# total = 0
# for i in range(args.discard_iter, len(times)):
#     total += times[i]
# avg = total / (args.iterations) * 1000.0
# print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")
