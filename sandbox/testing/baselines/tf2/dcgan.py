import argparse
import tensorflow as tf
import numpy as np
import time
import timeit
from .shared_functions import make_activation, make_conv2d, make_conv2d_bn, make_matmul

NAME = 'dcgan'
batch_size = 1
latent_dim = 100
img_size = 256
channels = 3

def generator(input, channels):
    t = input
    init_size = img_size // 4
    l1 = make_matmul(t, 128 * init_size ** 2)
    t = tf.reshape(l1, (l1.shape[0], 128, init_size, init_size) )

    new_height = int(round(init_size * 2))
    new_width = int(round(init_size * 2))
    resized = tf.image.resize_images(t, [new_height, new_width])

    t = make_conv2d(input_tensor=resized, filter_shape=(3,3,128,128), strides=(1,1,1,1), padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)

    new_height = int(round(init_size * 2 * 2))
    new_width = int(round(init_size * 2 * 2))
    resized = tf.image.resize_images(t, [new_height, new_width])
    t = make_conv2d(input_tensor=resized, filter_shape=(3,3,64,64), strides=(1,1,1,1), padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,channels,channels), strides=(1,1,1,1), padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.tanh(t)
    return t 

def discriminator(input, channels):
    t = input
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,16,16), strides=(1,1,2,2), padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,32,32), strides=(1,1,2,2), padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,64,64), strides=(1,1,2,2), padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,128,128), strides=(1,1,2,2), padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)

    #ds_size = img_size // 2 ** 4
    adv_layer = make_matmul(t, 1)
    t = tf.nn.sigmoid(adv_layer)
    return t

def dcgan_tf2_model(input):
    t = generator(input)
    t = discriminator(t)
    return t

@tf.function(experimental_compile=False)
def dcgan(input):
    return dcgan_tf2_model(input)

@tf.function(experimental_compile=True)
def dcgan_xla(input):
    return dcgan_tf2_model(input)

# parser = argparse.ArgumentParser()
# parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
# parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
# parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=5000)
# parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
# args = parser.parse_args()

