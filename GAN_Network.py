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
    
    def test_generate(self, sess, n_samples = 64000, filename='samples.png'):
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
    
    def test_generate(self, sess, n_samples = 64000, filename='samples.png'):
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
    