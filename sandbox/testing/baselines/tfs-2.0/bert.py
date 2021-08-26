import argparse
import tensorflow as tf
import numpy as np
import time
from shared_functions import make_matmul

tf.config.run_functions_eagerly(False)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def attention(input, heads):
    d_model = input.shape[1]
    d_k = d_model // heads
    assert d_model % heads == 0
    q = make_matmul(input, d_model)
    k = make_matmul(input, d_model)
    v = make_matmul(input, d_model)
    # reshape query, key, value
    q = tf.reshape(q, shape=(64,16,64))
    k = tf.reshape(k, shape=(64,16,64))
    v = tf.reshape(v, shape=(64,16,64))
    # transpose q, k, v for batched matmul
    q = tf.transpose(a=q, perm=(1,0,2))
    k = tf.transpose(a=k, perm=(1,0,2))
    v = tf.transpose(a=v, perm=(1,0,2))

    logits = tf.matmul(q, k)
    output = tf.matmul(logits, v)
    # transpose the output back
    output = tf.transpose(a=output, perm=(1,0,2))
    output = tf.reshape(output, shape=(64, 1024))
    # a final linear layer
    output = make_matmul(tf.nn.relu(make_matmul(input, 4*d_model)), d_model)
    return output

parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", default=False)
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=50)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=10)
args = parser.parse_args()

@tf.function(experimental_compile=args.xla)
def bert(input):
    t = input
    for i in range(8):
        t = attention(t, 16)
    return t

times = []
for i in range(args.discard_iter + args.iterations):
    t0 = time.time()
    res = bert(
    tf.constant(np.random.random_sample((64,1024)).astype(np.float32))
        )
    t1 = time.time()
    times.append(t1 - t0)
total = 0
for i in range(args.discard_iter, len(times)):
    total += times[i]
avg = total / (args.iterations) * 1000.0
print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")