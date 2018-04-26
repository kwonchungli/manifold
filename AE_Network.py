import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from AE_Framework import FIT_AE

class FIT_AE_Swiss(FIT_AE): 
    def define_data_dir(self):
        self.MODEL_DIRECTORY = "./model_AE/Swiss/"
        
    def __init__(self, exGAN):
        self.data_func = utils.swiss_load
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
    def define_data_dir(self):
        self.MODEL_DIRECTORY = "./model_AE/MNIST/"

    def __init__(self, exGAN):
        self.data_func = utils.MNIST_load
        self.epsilon = 0.3
        
        self.define_data_dir()
        super(FIT_AE_MNIST, self).__init__(exGAN)

    def get_image_dim(self):
        return 784
    def get_latent_dim(self):
        return 15

    def add_noise(self, batch_xs):
        noised = batch_xs + np.random.normal(self.epsilon/2, self.epsilon, size=batch_xs.shape)
        noised = np.clip(noised, 0., 1.)
        return noised

    def test_generate(self, sess, train_gen, n_samples = 512, filename='samples.png'):
        p_size = self.exGAN.PROJ_BATCH_SIZE
        for i in range(1):
            batch, _ = next(train_gen)
            batch = batch[:p_size, :]
            rx, rz = sess.run([self.rx, self.z], feed_dict={self.x_hat: batch, self.x: batch})

            # proj_img = self.exGAN.find_proj(sess, batch, rz)

            utils.save_images(rx.reshape(p_size, 28, 28), 'images/reconstr.png')
            utils.save_images(batch.reshape(p_size, 28, 28), 'images/original.png')
            # utils.save_images(proj_img.reshape(p_size, 28, 28), 'images/projection.png')

    def decoder(self, z, dim_img, n_hidden=256):
        y, _ = self.exGAN.build_generator(z, reuse=True)
        return y

    # Gaussian MLP as encoder
    def gaussian_MLP_encoder(self, x, n_hidden=256, reuse=False):
        with tf.variable_scope("gaussian_MLP_encoder", reuse=reuse):
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
            gaussian_params = tf.layers.dense(output, 2*self.get_latent_dim())

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.get_latent_dim()]
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.get_latent_dim():])
            z = mean + stddev * tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)

        return mean, stddev, z

    # Gateway
    def autoencoder(self, x_hat, x, n_hidden=256, reuse=False):
        # encoding
        mu, sigma, z = self.gaussian_MLP_encoder(x_hat, n_hidden, reuse)

        # decoding
        y, _ = self.exGAN.build_generator(mu, reuse=True)
        
        return mu, y

##########################################################################
class FIT_AE_MNIST_V2(FIT_AE_MNIST):
    def define_data_dir(self):
        self.MODEL_DIRECTORY = "./model_AE/MNIST_lt3/"

    def get_latent_dim(self):
        return 50
