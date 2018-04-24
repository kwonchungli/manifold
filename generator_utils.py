"""
An ugly file of helper functions to make all the pretrained
generators/discriminators work with the same API.
"""

import tensorflow as tf
import numpy as np
slim=tf.contrib.slim

## CELEBA UTILS

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)

## MNIST UTILS

class Hparams(object):
    def __init__(self):
        self.n_hidden_recog_1 = 500  # 1st layer encoder neurons
        self.n_hidden_recog_2 = 500  # 2nd layer encoder neurons
        self.n_hidden_gener_1 = 500  # 1st layer decoder neurons
        self.n_hidden_gener_2 = 500  # 2nd layer decoder neurons
        self.n_input = 784           # MNIST data input (img shape: 28*28)
        self.n_z = 20                # dimensionality of latent space
        self.transfer_fct = tf.nn.softplus

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0 / (fan_in + fan_out))
    high = constant*np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


