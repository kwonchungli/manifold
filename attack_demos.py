from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import pickle

from GAN_Network import WGAN_test, WGAN_Swiss, WGAN_MNIST
from AE_Network import FIT_AE_Swiss, FIT_AE_MNIST
from utils import file_exists, save_digit
from generator_models import celebA_generator, cifar_generator
from classifiers import *


def main():
    dataset = 'mnist'
    print('Running demo for', dataset, 'dataset')

    if dataset == 'mnist':
        myGAN = WGAN_MNIST()
        myVAE = FIT_AE_MNIST(myGAN)
        load_func = utils.MNIST_load
        eval_model = eval_mnist_model
        learning_rate = 0.001
        num_epochs = 2
        batch_size = 100
        x_size = 784
        y_size = 10
        dropout_rate = tf.placeholder_with_default(0.4, shape=())
        classifier_model_fn = lambda x: cnn_model_fn(x, dropout_rate)
    elif dataset == 'cifar10':
        myVAE = CIFAR10_AE()
        load_func = utils.cifar10_load
        eval_model = eval_cifar10_model
        learning_rate = 0.001
        num_epochs = 2
        batch_size = 100
        x_size = 784
        y_size = 10
        dropout_rate = tf.placeholder_with_default(0.4, shape=())
        classifier_model_fn = lambda x: cifar10_model_fn(x, dropout_rate)
    elif dataset == 'swiss':
        myGAN = WGAN_Swiss()
        myVAE = FIT_AE_Swiss(myGAN)
        load_func = utils.swiss_load
        eval_model = eval_swiss_model
        learning_rate = 0.1
        num_epochs = 1
        batch_size = 100
        x_size = 2
        y_size = 2
        classifier_model_fn = dnn_model_fn
    else:
        raise NotImplementedError('dataset with name ' + dataset + ' not supported yet. Please implement in the same way as for mnist')

    x = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.int32, shape=[None, y_size])
    logits = classifier_model_fn(x)
    train_op = get_train_op(logits, y, learning_rate)
    accuracy_op = get_accuracy_op(logits, y)

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        sess.run(init)
        print ('Loading GAN + AE')
        if myGAN: myGAN.restore_session(sess)
        myVAE.restore_session(sess)
        print ('Start Training Classifier')
        train_epoch, _, test_epoch = utils.load_dataset(batch_size, load_func)
        print('training model')
        train_model(sess, x, y, train_epoch, train_op, num_epochs, batch_size)
        print('testing model')
        eval_model(sess, x, y, dropout_rate, logits, test_epoch, accuracy_op, myVAE, image_path='clean_train_clean_test.png', name='clean train clean test')

if __name__ == '__main__': main()
