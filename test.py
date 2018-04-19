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
from attack_helpers import get_adv_dataset
from classifiers import *


if __name__ == '__main__':
    myGAN = WGAN_MNIST()
    myVAE = FIT_AE_MNIST(myGAN)
    
    #myGAN = WGAN_Swiss()
    #myVAE = FIT_AE_Swiss(myGAN)

    learning_rate = 0.001
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.int32, shape=[None, 10])
    dropout_rate = tf.placeholder_with_default(0.4, shape=())
    logits = cnn_model_fn(x, dropout_rate)
    train_op = get_train_op(logits, y, learning_rate)
    accuracy_op = get_accuracy_op(logits, y)

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        sess.run(init)
        myGAN.restore_session(sess)
        skip = False
        
        if skip or file_exists(myGAN.MODEL_DIRECTORY):
            print('Loading WGAN from ' + myGAN.MODEL_DIRECTORY)
            myGAN.restore_session(sess)
        else:
            print('Training WGAN')
            myGAN.train(sess)

        myVAE.restore_session(sess)
        #myVAE.train(sess)
        
        num_epochs = 1
        batch_size = 100
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        x_train = mnist.train.images
        y_train = np.asarray(mnist.train.labels, dtype=np.int32)
        x_test = mnist.test.images
        y_test = np.asarray(mnist.test.labels, dtype=np.int32)
        y_train = make_one_hot(y_train)
        y_test = make_one_hot(y_test)
        print('training model')
        train_model(sess, x, y, x_train, y_train, train_op, num_epochs, batch_size)
        print('testing model')
        eval_mnist_model(sess, x, y, dropout_rate, logits, x_test, y_test, accuracy_op, image_path='clean_train_clean_test.png', name='clean train clean test')
        print('making adv dataset')
        adversarial_x = get_adv_dataset(sess, logits, x, y, x_test, y_test)
        print('testing model with adversarial examples')
        eval_mnist_model(sess, x, y, dropout_rate, logits, adversarial_x, y_test, accuracy_op, image_path='clean_train_adv_test.png', name='clean train adv test')
        pickle.dump(adversarial_x, open('adversarial_x.p', 'wb'))
        print('cleaning adv dataset')
        cleaned_x = myVAE.autoencode_dataset(sess, adversarial_x)
        print('testing model with autoencoded adversarial examples')
        eval_mnist_model(sess, x, y, dropout_rate, logits, cleaned_x, y_test, accuracy_op, image_path='clean_train_ae_adv_test.png', name='clean train ae adv test')
        autoencoded_x = myVAE.autoencode_dataset(sess, x_test)
        pickle.dump(cleaned_x, open('cleaned_x.p', 'wb'))
        save_digit(x_test[0], 'clean.png')
        save_digit(adversarial_x[0], 'adv.png')
        save_digit(cleaned_x[0], 'cleaned.png')
        save_digit(autoencoded_x[0], 'autoencoded.png')
