from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import pickle

from GAN_Network import WGAN_test, WGAN_Swiss, WGAN_MNIST, WGAN_MNIST_V2,WGAN_MNIST_MINMAX
from AE_Network import FIT_AE_Swiss, FIT_AE_MNIST, FIT_AE_MNIST_V2,FIT_AE_MNIST_MINMAX
from utils import file_exists, save_digit
from generator_models import celebA_generator, cifar_generator
from classifiers import *



def main():
    dataset = 'mnist'
    print('Running demo for', dataset, 'dataset')


    #Set up GAN + VAE + training parameters
    train_classifier = False
    train_GAN = False

    myGAN = WGAN_MNIST_MINMAX()
    myVAE = FIT_AE_MNIST_MINMAX(myGAN)
    load_func = utils.MNIST_load
    eval_model = eval_mnist_model
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 100
    x_size = 784
    y_size = 10
    dropout_rate = tf.placeholder_with_default(0.4, shape=())
    classifier_model_fn = lambda x: cnn_model_fn(x, dropout_rate)

    x = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.int32, shape=[None, y_size])
    logits,classifier_params = classifier_model_fn(x)
    train_op = get_train_op(logits, y, learning_rate)
    accuracy_op = get_accuracy_op(logits, y)
    classifier_saver = tf.train.Saver(var_list=classifier_params, max_to_keep=1)
    classifier_model_directory = "./model_Classifier/MNIST/"

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        sess.run(init)

        if train_GAN:
            print('Training GAN')
            myGAN.train(sess)
        else:
            print('Loading GAN')
            myGAN.restore_session(sess)
            print('Loading VAE')
            myVAE.restore_session(sess)

        if train_classifier:
            print ('Start Training Classifier')
            train_epoch, _, test_epoch = utils.load_dataset(batch_size, load_func)
            print('training model')
            train_model(sess, x, y, train_epoch, train_op, num_epochs, batch_size)
            # print('testing model')
            # eval_model(sess, x, y, dropout_rate, logits, test_epoch, accuracy_op, myVAE, image_path='clean_train_clean_test.png', name='clean train clean test')
            print('saving model')
            save_model(sess,classifier_saver,classifier_model_directory)
        else:
            print('Load Classifier')
            restore_model(sess,classifier_saver,classifier_model_directory)
            print('Loaded from:')
            print(classifier_model_directory)

        # Test GAN
        # myGAN.test_generate(sess,512)

        # Test Projection Method
        # filename = './images/test_image.png'
        # n_images = myGAN.PROJ_BATCH_SIZE
        # z = myGAN.noise_gen([n_images,myGAN.get_latent_dim()])
        # images = sess.run(myGAN.Generator,feed_dict={myGAN.z_in: z})
        # utils.save_images(images.reshape(n_images,28,28),filename)
        #
        # z0 = myGAN.noise_gen([n_images,myGAN.get_latent_dim()])
        # images_hat,z_hat = myGAN.find_proj(sess,images,z0)
        # filename = './images/test_image_hat.png'
        # utils.save_images(images_hat,filename)


        # Train AE (to match projection)
        myVAE.train(sess)





if __name__ == '__main__': main()
