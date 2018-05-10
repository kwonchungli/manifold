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
import utils


if __name__ == '__main__':
    myGAN = WGAN_CelebA()
    print('initialized gan')
    myVAE = FIT_AE_CelebA(myGAN)
    print('initialized vae')
    
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        sess.run(init)

        myGAN.restore_session(sess)
        # myGAN.train(sess)
        # myVAE.restore_session(sess)
        myVAE.train(sess)
        
        myGAN.test_generate(sess)
      
        ################################
        gen = myGAN.get_train_gen(sess)
        batch, _ = next(gen)
        proj_img, rz = myVAE.autoencode_dataset(sess, batch)

        utils.save_images(batch.reshape(-1, 64, 64, 3), 'images/original.png')
        utils.save_images(proj_img.reshape(-1, 64, 64, 3), 'images/projection.png')
        # myGAN.train(sess)
        #myVAE.restore_session(sess)
        # myVAE.train(sess)
