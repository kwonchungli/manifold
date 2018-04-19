from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf

from GAN_Network import WGAN_test, WGAN_Swiss, WGAN_MNIST
from AE_Network import FIT_AE_Swiss, FIT_AE_MNIST
from utils import file_exists


if __name__ == '__main__':
    myGAN = WGAN_MNIST()
    myVAE = FIT_AE_MNIST(myGAN)
    
    #myGAN = WGAN_Swiss()
    #myVAE = FIT_AE_Swiss(myGAN)

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
        myVAE.train(sess)
