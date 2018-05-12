from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import pickle
import scipy

from GAN_Network import WGAN_test, WGAN_Swiss, WGAN_MNIST, WGAN_MNIST_V2,WGAN_MNIST_DIM2
# from AE_Network import FIT_AE_Swiss, FIT_AE_MNIST, FIT_AE_MNIST_V2
from utils import file_exists, save_digit
from generator_models import celebA_generator, cifar_generator
from classifiers import *


def softmax(w, t=1.0):
    npa = np.array
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist

def main():
    dataset = 'mnist'
    print('Running test with', dataset, 'dataset')

    #Set up GAN + training parameters
    train_classifier = False
    train_GAN = False

    myGAN = WGAN_MNIST_DIM2()
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
    # accuracy_op = get_accuracy_op(logits, y)
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
            print('Loaded from:')
            print(str(myGAN.MODEL_DIRECTORY))

        if train_classifier:
            print ('Start Training Classifier')
            train_epoch, _, test_epoch = utils.load_dataset(batch_size, load_func)
            print('training model')
            train_model(sess, x, y, train_epoch, train_op, num_epochs, batch_size)
            print('saving model')
            save_model(sess,classifier_saver,classifier_model_directory)
        else:
            print('Load Classifier')
            restore_model(sess,classifier_saver,classifier_model_directory)
            print('Loaded from:')
            print(classifier_model_directory)


        #--------------------------------------------------
        # Test Classifier
        #----------------------------
        accuracy_op = get_accuracy_op(logits, y)
        batch_size = 100
        train_epoch, _,test_epoch = utils.load_dataset(batch_size, load_func)
        batch_gen = utils.batch_gen(test_epoch, True, y.shape[1], num_iter=1)
        iteration = 0
        normal_avr = 0
        for images, labels in batch_gen:
            iteration += 1
            avr = sess.run(accuracy_op, feed_dict={x: images, y: labels, dropout_rate: 0.0})
            normal_avr += avr
        print("Normal Accuracy:", normal_avr / iteration)


        # -----------------------------------------------------------------------------------------------
        # Plot Generator Encodings
        # --------------------------

        # # Test images from dataset
        # n_samples = 10
        # filename = './images/test/samples.png'
        # train_epoch, _,_ = utils.load_dataset(n_samples, load_func)
        # batch_gen = utils.batch_gen(train_epoch, True, y.shape[1], num_epochs)
        # for x_train, y_train in batch_gen:
        #     rand_samples = x_train
        #     rand_labels = y_train
        # utils.save_images(rand_samples.reshape(n_samples,28,28),filename)
        #
        # # Test images from GAN
        # # n_samples = 1000
        # # filename = './images/test/samples.png'
        # # rand_samples = myGAN.test_generate(sess, n_samples, filename= filename, print_flag=False)
        #
        # # Keep Good Samples
        #
        # logits = sess.run(logits, feed_dict={x: rand_samples, dropout_rate: 0.0})
        # # threshold = 1.2
        # threshold = 10
        # good_samples = np.ndarray([0, 784])
        # good_preds = []
        # for i,logit in enumerate(logits):
        #     pred = softmax(logit)
        #     entropy = scipy.stats.entropy(pred)
        #     if entropy < threshold:
        #         good_sample = rand_samples[i].reshape(1,784)
        #         good_samples = np.concatenate((good_samples,good_sample))
        #         good_preds.append(np.argmax(pred))
        #         print("classifier output")
        #         print(np.argmax(logit))
        #         print("true class")
        #         print(np.argmax(rand_labels[i]))
        # print(good_samples.shape)
        # print(good_preds)
        #
        # # Print Good Samples
        # if good_samples.shape[0] > 0:
        #     filename = './images/test/good_samples.png'
        #     utils.save_images(good_samples.reshape(good_samples.shape[0], 28, 28), filename)

        # Keep Good Samples (Threshold Range Test)

        # logits = sess.run(logits, feed_dict={x: rand_samples, dropout_rate: 0.0})
        # threshold_range = np.linspace(1.3,1.8,8)
        # for j,threshold in enumerate(threshold_range):
        #     good_samples = np.ndarray([0, 784])
        #     for i,logit in enumerate(logits):
        #         pred = softmax(logit)
        #         entropy = scipy.stats.entropy(pred)
        #         if entropy < threshold:
        #             good_sample = rand_samples[i].reshape(1,784)
        #             good_samples = np.concatenate((good_samples,good_sample))
        #     print(good_samples.shape)
        #
        #     # Print Good Samples
        #     if good_samples.shape[0] > 0:
        #         filename = './images/test/good_samples_'+str(j+1)+'.png'
        #         utils.save_images(good_samples.reshape(good_samples.shape[0], 28, 28), filename)
        #
        # print(threshold_range)

        # -----------------------------------------------------------------------------------------------



        # -----------------------------------------------------------------------------------------------
        # Test GAN
        # --------------
        # myGAN.test_generate(sess,512)

        # Test Projection Method
        # n_images = myGAN.PROJ_BATCH_SIZE
        # filename = './images/test/test_images.png'

        # test images from generator
        # z = myGAN.noise_gen([n_images,myGAN.get_latent_dim()])
        # images = sess.run(myGAN.Generator,feed_dict={myGAN.z_in: z})
        # utils.save_images(images.reshape(n_images,28,28),filename)

        # test images from dataset
        # train_epoch, _,_ = utils.load_dataset(n_images, load_func)
        # batch_gen = utils.batch_gen(train_epoch, True, y.shape[1], num_epochs)
        # for x_train, y_train in batch_gen:
        #     images = x_train
        #     labels = y_train
        # utils.save_images(images.reshape(n_images,28,28),filename)

        # find projection
        # images_hat,z_hat = myGAN.find_proj(sess,images,z0=None,random_init=True,random_iter=4)
        # filename = './images/test/test_images_hat'+str(4)+'_'+str(myGAN.PROJ_ITER)+'.png'
        # utils.save_images(images_hat,filename)

        # for i in range(4,7,1):
        #     images_hat,z_hat = myGAN.find_proj(sess,images,z0=None,random_init=True,random_iter=i)
        #     filename = './images/test/test_images_hat'+str(i)+'_'+str(myGAN.PROJ_ITER)+'.png'
        #     utils.save_images(images_hat,filename)

        # -----------------------------------------------------------------------------------------------



if __name__ == '__main__': main()
