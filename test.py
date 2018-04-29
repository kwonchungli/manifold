from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import pickle

from GAN_Network import *
from AE_Network import *
from Classifier_Network import *
from utils import file_exists, save_digit
from classifiers import *
from ConvNet import ConvNet

if __name__ == '__main__':
    myGAN = WGAN_MNIST_V2()
    myVAE = FIT_AE_MNIST_V2(myGAN)
    
    #myGAN = WGAN_Swiss()
    #myVAE = FIT_AE_Swiss(myGAN)
    
    # myClass = Classifier_MNIST(0.0, myVAE)
    # myClass = Classifier_MNIST_Robust(0.3, myVAE)
    
    # myClass = Classifier_CIFAR10(0.0, myVAE)
    # myClass = Classifier_CIFAR10_Robust(0.08, myVAE)
    myClass=  Classifier_CIFAR10_MINMAX(0.03, myVAE)
    
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        sess.run(init)
        
        print ('Loading GAN + AE')
        myGAN.restore_session(sess)
        myVAE.restore_session(sess)
       
        print ('Start Training Classifier')
        myClass.restore_session(sess)
        myClass.train(sess)
        
        print('testing model')
        myClass.eval_model(sess, 0.05)
        
