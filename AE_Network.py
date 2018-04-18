import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from AE_Framework import FIT_AE

class FIT_AE_Swiss(FIT_AE): 
    def __init__(self, exGAN):
        self.data_func = utils.swiss_load
        self.MODEL_DIRECTORY = "./model_AE/Swiss/"
        self.epsilon = 0.4
        
        super(FIT_AE_Swiss, self).__init__(exGAN)
        
    def get_image_dim(self):
        return 2
    def get_latent_dim(self):
        return 1
    
    def test_generate(self, sess, train_gen, n_samples = 64000, filename='images/samples.png'):
        fig, ax = plt.subplots()
        
        noises = self.noise_gen((n_samples, self.get_latent_dim()))
        gen_points = sess.run(self.decoded,
                                 feed_dict={self.z_in: noises})
        
        plt.scatter(gen_points[:,0], gen_points[:,1], s=0.4, c='b', alpha=0.1)
        
        for i in range(5):
            batch, _ = next(train_gen)
            rx, res = sess.run([self.rx, self.z], feed_dict={self.x_hat: batch, self.x: batch})
            plt.scatter(batch[:5,0], batch[:5,1], s=10, c='r', alpha=1)
            plt.scatter(rx[:5,0], rx[:5,1], s=10, c='g', alpha=1)
        
        fig.savefig(filename)
        plt.close()
        
class FIT_AE_MNIST(FIT_AE): 
    def __init__(self, exGAN):
        self.data_func = utils.MNIST_load
        self.MODEL_DIRECTORY = "./model_AE/MNIST/"
        self.epsilon = 0.3
        
        super(FIT_AE_MNIST, self).__init__(exGAN)
        
    def get_image_dim(self):
        return 784
    def get_latent_dim(self):
        return 15
        
    def add_noise(self, batch_xs):
        noised = batch_xs + np.random.uniform(0., self.epsilon, size=batch_xs.shape)
        noised = np.clip(noised, 0., 1.)
        return noised
    
    def test_generate(self, sess, train_gen, n_samples = 512, filename='samples.png'):
        p_size = self.exGAN.PROJ_BATCH_SIZE
        for i in range(1):
            batch, _ = next(train_gen)
            batch = batch[:p_size, :]
            rx, rz = sess.run([self.rx, self.z], feed_dict={self.x_hat: batch, self.x: batch})
            
            proj_img = self.exGAN.find_proj(sess, batch, rz)
            
            utils.save_images(rx.reshape(p_size, 28, 28), 'images/reconstr.png')
            utils.save_images(batch.reshape(p_size, 28, 28), 'images/original.png')
            utils.save_images(proj_img.reshape(p_size, 28, 28), 'images/projection.png')
        
    def decoder(self, z, dim_img, n_hidden=256):
        return self.exGAN.build_generator(z, reuse=True)
    
    # Gateway
    def autoencoder(self, x_hat, x, n_hidden=256, reuse=False):
        # encoding
        mu, sigma, z = self.gaussian_MLP_encoder(x_hat, n_hidden, reuse) 

        # decoding
        y = self.exGAN.build_generator(mu, reuse=True)
        
        return z, y        