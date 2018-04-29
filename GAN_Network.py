from tensorlayer.layers import *
from generator_utils import *
slim = tf.contrib.slim

import tflib as lib
import tflib.ops.linear as linear
import tflib.ops.conv2d as conv2d
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
        self.BATCH_SIZE = 128
        self.ITERS = 50001
        self.CRITIC_ITERS = 5
        self.PROJ_ITER = 150
        self.PROJ_BATCH_SIZE = 100
        
    def define_data_dir(self):
        self.MODEL_DIRECTORY = "./model_WGAN/MNIST/"
        
    def __init__(self):
        self.data_func = utils.MNIST_load
        self.define_data_dir()

        super(WGAN_MNIST, self).__init__()
        
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
        return 15
    def get_image_dim(self):
        return 784
    
    
###########################################################################
##########################################################################
class WGAN_MNIST_V2(WGAN_MNIST):
    def define_data_dir(self):
        self.MODEL_DIRECTORY = "./model_WGAN/MNIST_lt3/"
        
    def get_latent_dim(self):
        return 50
    
########################################################################3
class WGAN_CelebA(WGAN):
    def define_default_param(self):
        self.BATCH_SIZE = 128
        self.ITERS = 50001
        self.CRITIC_ITERS = 5
        self.PROJ_ITER = 150
        self.PROJ_BATCH_SIZE = 25
        
    def define_data_dir(self):
        # self.MODEL_DIRECTORY = "./model_WGAN/CelebA/"
        self.MODEL_DIRECTORY = "./model_WGAN/CelebA/"
        
    def restore_session(self, sess):
        # self.saver.restore(sess, './PreModel/Dimakis_Models/Celeb_A/data/CelebA_gen/model.ckpt-102951')
        self.saver.restore(sess, './pretrained_models/Celeb_A/data/CelebA_gen/model.ckpt-102951')

    def __init__(self):
        self.data_func = utils.CelebA_load
        self.define_data_dir()

        super(WGAN_CelebA, self).__init__()
        
    def noise_gen(self, noise_size):
        return np.random.normal(size=noise_size).astype('float32')
        #return np.random.uniform(0, 10, size=noise_size).astype('float32')
    
    def get_train_gen(self, sess):
        import data_loader as dl
        import PIL
        paths = dl.get_loader('./data/CelebA', self.BATCH_SIZE, 64, 'NHWC', 'train')
        
        def batch_gen(use_one_hot_encoding=False, out_dim=-1, num_iter=-1):
            st, it = 0, 0
            batch_x = np.zeros((self.BATCH_SIZE, self.get_image_dim()))
            crop = (50, 25, 128, 160)
            while (it < num_iter) or (num_iter < 0):
                it = it + 1
                
                for i in range(0, self.BATCH_SIZE):
                    img = PIL.Image.open(paths[st + i])
                    img = img.crop(crop)
                    img = img.resize((64, 64))
                    img = np.asarray(img, dtype=np.float32) / 255.
                    
                    batch_x[i] = img.reshape(-1, self.get_image_dim())
                
                st = (st + self.BATCH_SIZE) % (len(paths) - self.BATCH_SIZE)
                yield batch_x, None

        return batch_gen()
    
    def train(self, session):
        print 'Bad Thing Happens!!!'
        Hahahahahahahahahaha
    
    # G(z)
    def build_generator(self, z, reuse=False):
        hidden_num, output_num, repeat_num, data_format = 128, 3, 4, 'NCHW'
        
        with tf.variable_scope("G", reuse=reuse) as vs:
            num_output = int(np.prod([8, 8, hidden_num]))
            x = slim.fully_connected(z, num_output, activation_fn=None)
            x = reshape(x, 8, 8, hidden_num, data_format)
            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                if idx < repeat_num - 1:
                    x = upscale(x, 2, data_format)
            out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)
            out = tf.nn.sigmoid(out)
            out = tf.transpose(out, [0, 2, 3, 1])
            
        variables = tf.contrib.framework.get_variables(vs)
        out = tf.reshape(out, [-1, self.get_image_dim()])
        return out, variables

    # D(x)
    def define_loss(self):
        return 0, 0, 0
    
    def build_discriminator(self, x, reuse=False):
        with tf.variable_scope('Dummy', reuse=reuse) as vs:
            output = tf.reshape(x, [-1, 64, 64, 3])

            output = tf.layers.conv2d(output, 128, 4, strides = 2)
            output = utils.LeakyReLU(output)
            
            output = tf.layers.conv2d(output, 32, 3, strides = 2)
            output = utils.LeakyReLU(output)
            
            output = tf.layers.conv2d(output, 16, 2, strides = 1)
            output = utils.LeakyReLU(output)
            
            output = tf.reshape(output, [-1, 3136])
            output = tf.layers.dense(output, 256)

            output = utils.LeakyReLU(output)
            output = tf.layers.dense(output, 1)
           
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

class WGAN_MNIST_MINMAX(WGAN_MNIST):
    def define_default_param(self):
        self.BATCH_SIZE = 128
        self.ITERS = 50001
        self.CRITIC_ITERS = 5
        self.PROJ_ITER = 150
        self.PROJ_BATCH_SIZE = 128

        
        
        
