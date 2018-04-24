from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import pickle

from GAN_Network import *
from AE_Network import *
from F_AAE import *


if __name__ == '__main__':
    myGAN = WGAN_CelebA()
    #myVAE = FIT_AE_CelebA(myGAN)
    
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        sess.run(init)
        
        myGAN.restore_session(sess)
        myGAN.train(sess)
        #myVAE.restore_session(sess)
        #myVAE.train(sess)
       
    