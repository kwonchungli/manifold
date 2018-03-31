import numpy as np
from GAN_Framework import WGAN
import utils
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

class WGAN_Swiss(WGAN):
    def __init__(self):
        self.data_func = utils.swiss_load
        self.MODEL_DIRECTORY = "./model_WGAN/Swiss/"
        
        super(WGAN_Swiss, self).__init__()
        
    def noise_gen(self, noise_size):
        #return np.random.normal(size=noise_size).astype('float32')
        return np.random.uniform(0, 10, size=noise_size).astype('float32')
    
    def test_generate(self, sess, n_samples = 4096, filename='samples.png'):
        fig = plt.figure()
        
        noises = self.noise_gen((n_samples,self.get_latent_dim()))
        gen_points = sess.run(self.Generator,
                                 feed_dict={self.z_in: noises})
        
        plt.scatter(gen_points[:,0], gen_points[:,1], s=0.4, c='b', alpha=0.7)
        
        import math
        x, y = 0.1*noises*np.cos(2*noises), 0.1*noises*np.sin(2*noises)
        plt.scatter(x, y, c='r', s=0.1, alpha=0.5)
        
        fig.savefig(filename)
        plt.close()
        
    def get_latent_dim(self):
        return 1
    def get_image_dim(self):
        return 2
    
    def build_generator(self, z = None, reuse = False):
        n_hidden = 256
        
        with tf.variable_scope("Generator", reuse=reuse):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.01)

            # 1st hidden layer
            w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.relu(h0)

            # 2nd hidden layer
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.relu(h1)

            # output layer-mean
            l2 = tf.layers.dense(h1, n_hidden)
            l2 = tf.nn.relu(l2)
            
            y = tf.layers.dense(l2, self.get_image_dim())
            
        return y

    def build_discriminator(self, inputs, reuse = False):
        with tf.variable_scope("Discriminator", reuse=reuse):
            # Hidden fully connected layer with 256 neurons
            layer_1 = tf.layers.dense(inputs, 256)
            layer_1 = tf.nn.relu(layer_1)
            
            layer_2 = tf.layers.dense(layer_1, 256)
            layer_2 = tf.nn.relu(layer_2)
            
            layer_3 = tf.layers.dense(layer_2, 256)
            layer_3 = tf.nn.relu(layer_3)
            
            layer_4 = tf.layers.dense(layer_3, 256)
            layer_4 = tf.nn.relu(layer_4)
            
            output = tf.layers.dense(layer_4, 1)
        
        return tf.reshape(output, [-1])
