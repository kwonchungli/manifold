from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from GAN_Network import WGAN_test, WGAN_Swiss
from WAE_Network import WAE
from VAE import VAE
import tensorflow as tf

if __name__ == '__main__':
    myGAN = WGAN_Swiss()
    myVAE = WAE(myGAN)
    
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        sess.run(init)

        myGAN.restore_session(sess)
        #myGAN.train(sess)
        
        myVAE.train(sess)
        
        
