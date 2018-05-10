from tensorlayer.layers import *
from generator_utils import *
slim = tf.contrib.slim

import tflib as lib
import tflib.ops.linear as linear
import tflib.ops.conv2d as conv2d
import tflib.ops.layernorm as layernorm
import tflib.ops.batchnorm as batchnorm
import tflib.ops.deconv2d as deconv2d

import numpy as np
from GAN_Framework import WGAN
import utils
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf


class WGAN_test(WGAN):
    def __init__(self):
        self.data_func = utils.test_2d
        self.MODEL_DIRECTORY = "./model_WGAN/test/"
        
        super(WGAN_test, self).__init__()
        
    def noise_gen(self, noise_size):
        return np.random.normal(size=noise_size).astype('float32')
    
    def test_generate(self, sess, n_samples = 64000, filename='images/samples.png'):
        fig, ax = plt.subplots()
        
        noises = self.noise_gen((n_samples,self.get_latent_dim()))
        gen_points = sess.run(self.Generator,
                                 feed_dict={self.z_in: noises})
        
        plt.scatter(gen_points[:,0], gen_points[:,1], s=0.4, c='b', alpha=0.7)
        
        fig.savefig(filename)
        plt.close()
        
    def get_latent_dim(self):
        return 2
    def get_image_dim(self):
        return 2
    
    

    
########################################################################3    
########################################################################3
########################################################################3
########################################################################3
########################################################################3

class WGAN_Swiss(WGAN):
    def __init__(self):
        self.data_func = utils.swiss_load
        self.MODEL_DIRECTORY = "./model_WGAN/Swiss/"
        
        super(WGAN_Swiss, self).__init__()
        
    def noise_gen(self, noise_size):
        return np.random.normal(size=noise_size).astype('float32')
        #return np.random.uniform(0, 10, size=noise_size).astype('float32')
    
    def build_discriminator(self, inputs, reuse = False):
        n_hidden = 256
        with tf.variable_scope("Discriminator", reuse=reuse) as vs:
            # Hidden fully connected layer with 256 neurons
            layer_1 = tf.layers.dense(inputs, n_hidden)
            layer_1 = utils.LeakyReLU(layer_1, alpha=0.01)
            
            layer_2 = tf.layers.dense(layer_1, n_hidden)
            layer_2 = utils.LeakyReLU(layer_2, alpha=0.01)
            
            output = tf.layers.dense(layer_2, 1)
        
        variables = tf.contrib.framework.get_variables(vs)
        return tf.reshape(output, [-1])
    
    def define_learning_rate(self):
        self.train_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
                1e-4,  # Base learning rate.
                self.proj_step,  # Current index into the dataset.
                10000,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
        disc_rate = tf.train.exponential_decay(
                1e-3,  # Base learning rate.
                self.proj_step,  # Current index into the dataset.
                10000,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
        return learning_rate, disc_rate
    
    def test_generate(self, sess, n_samples = 64000, filename='images/samples.png'):
        fig = plt.figure()
        
        noises = self.noise_gen((n_samples,self.get_latent_dim()))
        gen_points = sess.run(self.Generator,
                                 feed_dict={self.z_in: noises})
        
        plt.scatter(gen_points[:,0], gen_points[:,1], s=0.4, c='b', alpha=0.7)
        """
        for i in range(10):
            t = i + np.linspace(0, 0.5, num=500)
            x, y = 0.1*t*np.cos(t), 0.1*t*np.sin(t)
            plt.scatter(x, y, c='r', s=0.1, alpha=0.5)
        """
        fig.savefig(filename)
        plt.close()
        
    def get_latent_dim(self):
        return 1
    def get_image_dim(self):
        return 2
    
########################################################################3    
########################################################################3
########################################################################3
########################################################################3
########################################################################3

class WGAN_MNIST(WGAN):
    def define_default_param(self):
        self.BATCH_SIZE = 64
        self.ITERS = 10001
        self.CRITIC_ITERS = 5
        self.PROJ_ITER = 500
        self.PROJ_BATCH_SIZE = 100
        
    def define_data_dir(self):
        self.data_func = utils.MNIST_load
        self.MODEL_DIRECTORY = "./model_WGAN/MNIST_lt3/"
        
    def noise_gen(self, noise_size):
        return np.random.normal(size=noise_size).astype('float32')
        #return np.random.uniform(0, 10, size=noise_size).astype('float32')
    
    def define_learning_rate(self):
        self.train_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
                1e-3,  # Base learning rate.
                self.train_step,  # Current index into the dataset.
                3000,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
        disc_rate = tf.train.exponential_decay(
                1e-3,  # Base learning rate.
                self.train_step,  # Current index into the dataset.
                3000,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
        return learning_rate, disc_rate
    
    # G(z)
    def build_generator(self, z, reuse=False):
        with tf.variable_scope('Generator', reuse=reuse) as vs:
            output = tf.layers.dense(z, 6*6*self.get_latent_dim())
            output = utils.LeakyReLU(output)
            
            output = tf.reshape(output, [-1, 6, 6, self.get_latent_dim()])
            
            output = tf.layers.conv2d_transpose(output, 64, 4, strides=2)
            output = utils.LeakyReLU(output)

            output = tf.layers.conv2d_transpose(output, 1, 2, strides=2)
            output = tf.nn.sigmoid(output)
            
        variables = tf.contrib.framework.get_variables(vs)
        return tf.reshape(output, [-1, self.get_image_dim()]), variables

    # D(x)
    def build_discriminator(self, x, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse) as vs:
            output = tf.reshape(x, [-1, 28, 28, 1])

            output = tf.layers.conv2d(output, 128, 5)
            output = utils.LeakyReLU(output)

            output = tf.layers.conv2d(output, 32, 5)
            output = utils.LeakyReLU(output)
            
            output = tf.layers.conv2d(output, 16, 5)
            output = utils.LeakyReLU(output)
            
            output = tf.reshape(output, [-1, 4096])
            output = tf.layers.dense(output, 256)
            
            output = utils.LeakyReLU(output)
            output = tf.layers.dense(output, 1)
            
        variables = tf.contrib.framework.get_variables(vs)
        return tf.reshape(output, [-1])
    
    def test_generate(self, sess, n_samples = 512, filename='images/samples.png'):
        noises = self.noise_gen((n_samples,self.get_latent_dim()))
        samples = sess.run(self.Generator, feed_dict={self.z_in: noises})
        
        utils.save_images(samples.reshape(n_samples, 28, 28), filename)
        
    def get_latent_dim(self):
        return 50
    def get_image_dim(self):
        return 784
    
    
###########################################################################
##########################################################################
class WGAN_MNIST_V2(WGAN_MNIST):
    def define_data_dir(self):
        self.data_func = utils.MNIST_load
        self.MODEL_DIRECTORY = "./model_WGAN/MNIST_V2/"
        
    def get_latent_dim(self):
        return 128
    
    # G(z)
    def build_generator(self, z, reuse=False):
        with tf.variable_scope('Generator', reuse=reuse) as vs:
            output = tf.layers.dense(z, 4096)
            output = tf.nn.relu(output)
            
            output = tf.reshape(output, [-1, 16, 16, 16])
            output = tf.layers.conv2d_transpose(output, 256, 5, 1)
            output = tf.nn.relu(output)
            
            output = tf.layers.conv2d_transpose(output, 128, 5, 1)
            output = tf.nn.relu(output)
            
            output = tf.layers.conv2d_transpose(output, 1, 5, 1)
            output = tf.nn.sigmoid(output)
            
        variables = tf.contrib.framework.get_variables(vs)
        return tf.reshape(output, [-1, self.get_image_dim()]), variables
    
class WGAN_MNIST_DIM2(WGAN_MNIST_V2):
    def get_latent_dim(self):
        return 2
    
    def define_data_dir(self):
        self.data_func = utils.MNIST_load
        self.MODEL_DIRECTORY = "./model_WGAN/MNIST/"
        
class WGAN_F_MNIST(WGAN_MNIST_V2):
    def define_default_param(self):
        self.BATCH_SIZE = 32
        self.ITERS = 100001
        self.CRITIC_ITERS = 5
        self.PROJ_ITER = 1000
        self.PROJ_BATCH_SIZE = 10
        
    def define_data_dir(self):
        self.data_func = utils.F_MNIST_load
        self.MODEL_DIRECTORY = "./model_WGAN/F_MNIST/"
    
    
########################################################################3
################### CIFAR10 ##############################################
#########################################################################
class WGAN_CIFAR10(WGAN):
    def define_default_param(self):
        self.BATCH_SIZE = 100
        self.ITERS = 50001
        self.CRITIC_ITERS = 5
        self.PROJ_ITER = 600
        self.PROJ_BATCH_SIZE = 100
        
    def define_data_dir(self):
        self.data_func = utils.cifar10_load
        self.MODEL_DIRECTORY = "./model_WGAN/CIFAR10/"
        
    def noise_gen(self, noise_size):
        return np.random.normal(0, 1, size=noise_size).astype('float32')
    
    # G(z)
    def build_generator(self, z, reuse=False):
        DIM = 64
        with tf.variable_scope('Generator', reuse=reuse) as vs:
            output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, z)
            output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
            output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, 4*DIM, 4, 4])

            output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
            output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
            output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)
            output = tf.tanh(output)
            
        variables = tf.contrib.framework.get_variables(vs)
        return tf.reshape(output, [-1, self.get_image_dim()]), variables
    
    def build_discriminator(self, x, reuse=False):
        dim = 64
        nonlinearity=tf.nn.relu
        output = tf.reshape(x, [-1, 64, 64, 3])
        output = tf.transpose(output, perm=[0, 3, 1, 2])

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = tf.reshape(output, [-1, 4*4*8*dim])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1])
    
    def test_generate(self, sess, n_samples = 128, filename='images/samples.png'):
        noises = self.noise_gen((n_samples,self.get_latent_dim()))
        samples = sess.run(self.Generator, feed_dict={self.z_in: noises})
        
        utils.save_images(samples.reshape(n_samples, 64, 64, 3), filename)
        
    def get_latent_dim(self):
        return 128
    def get_image_dim(self):
        return 12288

###########################################################################
##########################################################################

        
        
        
########################################################3 
### CELEBA
########################################################3
class WGAN_CelebA(WGAN_CIFAR10):
    def define_data_dir(self):
        self.data_func = utils.CelebA_load
        self.MODEL_DIRECTORY = "./model_WGAN/CelebA/"
        