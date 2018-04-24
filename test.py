from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import pickle

from GAN_Network import *
from AE_Network import *
from utils import file_exists, save_digit
from classifiers import *


if __name__ == '__main__':
    myGAN = WGAN_MNIST_V2()
    myVAE = FIT_AE_MNIST_V2(myGAN)
    
    #myGAN = WGAN_Swiss()
    #myVAE = FIT_AE_Swiss(myGAN)

    learning_rate = 0.001
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.int32, shape=[None, 10])
    
    dropout_rate = tf.placeholder_with_default(0.4, shape=())
    logits = cnn_model_fn(x, dropout_rate)
    adv_logits = cnn_model_fn(myVAE.rx, dropout_rate, True)
    
    train_op = get_train_op(logits, y, learning_rate)
    accuracy_op = get_accuracy_op(logits, y)
    adv_acc_op = get_accuracy_op(adv_logits, y)
    
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        sess.run(init)
        
        print ('Loading GAN + AE')
        myGAN.restore_session(sess)
        myVAE.restore_session(sess)
       
        print ('Start Training Classifier')
        num_epochs = 2
        batch_size = 100
        
        train_epoch, _, test_epoch = utils.load_dataset(batch_size, utils.MNIST_load)
        
        print('training model')
        train_model(sess, x, y, train_epoch, train_op, num_epochs, batch_size)
        
        print('testing model')
        eval_mnist_model(sess, x, y, dropout_rate, logits, adv_logits, test_epoch, accuracy_op, adv_acc_op, myVAE, image_path='clean_train_clean_test.png', name='clean train clean test')
        