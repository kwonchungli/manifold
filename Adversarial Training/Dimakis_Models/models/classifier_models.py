import tensorflow as tf
from tensorlayer.layers import *
from tensorlayer.activation import *

def celebA_classifier(ims, reuse):
    with tf.variable_scope("C", reuse=reuse) as vs:
        net = InputLayer(ims)
        n_filters = 3
        for i in range(2):
            net = Conv2dLayer(net, \
                    act=tf.nn.relu, \
                    shape=[5,5,n_filters,64], \
                    name="conv_" + str(i))
            net = MaxPool2d(net, \
                    filter_size=(3,3), \
                    strides=(2,2), \
                    name="mpool_" + str(i))
            net = LocalResponseNormLayer(net, \
                    depth_radius=4, \
                    bias=1.0, \
                    alpha=0.001 / 9.0, \
                    beta=0.75, \
                    name="lrn_" + str(i))
            n_filters = 64
        net = FlattenLayer(net)
        net = DenseLayer(net, n_units=384, act=tf.nn.relu, name="d1")
        net = DenseLayer(net, n_units=192, act=tf.nn.relu, name="d2")
        net = DenseLayer(net, n_units=2, act=tf.identity, name="final")
        cla_vars = tf.contrib.framework.get_variables(vs)
        if not reuse:
            return net.outputs, tf.argmax(net.outputs, axis=1), cla_vars
    return net.outputs, tf.argmax(net.outputs, axis=1)

def mnist_classifier(ims, reuse):
    with tf.variable_scope("C", reuse=reuse) as vs:
        net = InputLayer(ims)
        n_filters = 1
        net = Conv2dLayer(net, \
                act=tf.nn.relu, \
                shape=[5,5,n_filters,64], \
                name="conv")
        net = MaxPool2d(net, \
                filter_size=(3,3), \
                strides=(2,2), \
                name="mpool")
        net = FlattenLayer(net)
        net = DenseLayer(net, n_units=384, act=tf.nn.relu, name="d1")
        net = DenseLayer(net, n_units=192, act=tf.nn.relu, name="d2")
        net = DenseLayer(net, n_units=10, act=tf.identity, name="final")
        cla_vars = tf.contrib.framework.get_variables(vs)
        if not reuse:
            return net.outputs, tf.argmax(net.outputs, axis=1), cla_vars
    return net.outputs, tf.argmax(net.outputs, axis=1)

def cifar10_classifier(im, reuse):
    with tf.variable_scope('C', reuse=reuse) as vs:
        net = InputLayer(im)
        net = Conv2dLayer(net, \
                act=tf.nn.relu, \
                shape=[5,5,3,64], \
                name="conv1")
        net = MaxPool2d(net, \
                filter_size=(3,3), \
                strides=(2,2), \
                name="pool1")
        net = LocalResponseNormLayer(net, \
                depth_radius=4, \
                bias=1.0, \
                alpha = 0.001/9.0, \
                beta = 0.75, \
                name="norm1")
        net = Conv2dLayer(net, \
                act=tf.nn.relu, \
                shape=[5,5,64,64], \
                name="conv2")
        net = LocalResponseNormLayer(net, \
                depth_radius=4, \
                bias=1.0, \
                alpha=0.001/9.0, \
                beta = 0.75, \
                name="norm2")
        net = MaxPool2d(net, \
                filter_size=(3,3), \
                strides=(2,2), \
                name="pool2")
        net = FlattenLayer(net, name="flatten_1")
        net = DenseLayer(net, n_units=384, name="local3", act=tf.nn.relu)
        net = DenseLayer(net, n_units=192, name="local4", act=tf.nn.relu)
        net = DenseLayer(net, n_units=10, name="softmax_linear", act=tf.identity)
        cla_vars = tf.contrib.framework.get_variables(vs)
        def name_fixer(var):
            return var.op.name.replace("W", "weights") \
                                .replace("b", "biases") \
                                .replace("weights_conv2d", "weights") \
                                .replace("biases_conv2d", "biases") 
        cla_vars = {name_fixer(var): var for var in cla_vars}
        if not reuse:
            return net.outputs, tf.argmax(net.outputs, axis=1), cla_vars
        return net.outputs, tf.argmax(net.outputs, axis=1)

