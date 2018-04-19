import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from .generator_utils import *
slim = tf.contrib.slim
# Copied for WGAN
#from lib_external import tflib as lib
#import lib_external.tflib.ops.linear as linear
#import lib_external.tflib.ops.conv2d as conv2d
#import lib_external.tflib.ops.batchnorm as batchnorm
#import lib_external.tflib.ops.deconv2d as deconv2d

def celebA_generator(z, hidden_num=128, output_num=3, repeat_num=4, data_format='NCHW', reuse=False):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)       
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)
        out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)
    variables = tf.contrib.framework.get_variables(vs)
    out = tf.transpose(out, [0, 2, 3, 1])
    return out, variables

def mnist_generator(z, n_z=20, n_hg1=500, n_hg2=500, n_inp=784, reuse=False):
    with tf.variable_scope("G", reuse=reuse) as scope:
        w1 = tf.get_variable('w1', initializer=xavier_init(n_z, n_hg1))
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_hg1], dtype=tf.float32))
        hidden1 = tf.nn.softplus(tf.matmul(z, w1) + b1)

        w2 = tf.get_variable('w2', initializer=xavier_init(n_hg1, n_hg2))
        b2 = tf.get_variable('b2', initializer=tf.zeros([n_hg2], dtype=tf.float32))
        hidden2 = tf.nn.softplus(tf.matmul(hidden1, w2) + b2)

        w3 = tf.get_variable('w3', initializer=xavier_init(n_hg2, n_inp))
        b3 = tf.get_variable('b3', initializer=tf.zeros([n_inp], dtype=tf.float32))
        logits = tf.matmul(hidden2, w3) + b3
        out = tf.nn.sigmoid(logits)
    variables = tf.contrib.framework.get_variables(scope)
    variables = {var.op.name.replace("G/", "gen/"): var for var in variables}
    out = tf.reshape(out, [-1, 28, 28, 1])
    return out, variables

def cifar_generator(z, n_z=128, DIM=64, reuse=False):
    with tf.variable_scope("G", reuse=reuse) as scope:
        output = linear.Linear('Generator.Input', n_z, 4*4*4*DIM, z)
        output = batchnorm.Batchnorm('Generator.BN1', [0], output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4*DIM, 4, 4])

        output = deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
        output = batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
        output = tf.nn.relu(output)

        output = deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
        output = batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
        output = tf.nn.relu(output)

        output = deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

        output = tf.tanh(output)

        variables = tf.contrib.framework.get_variables(scope)
        variables = {var.op.name.replace("G/", ""): var for var in variables}
        temp = tf.reshape(output, [-1, 3, 32, 32])
        return tf.transpose(temp,[0,2,3,1]), variables
