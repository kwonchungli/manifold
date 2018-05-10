import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from AE_Framework import FIT_AE
from ops import batch_normal, de_conv, conv2d, fully_connect, lrelu

class FIT_AE_Swiss(FIT_AE): 
    def define_data_dir(self):
        self.data_func = utils.swiss_load
        self.MODEL_DIRECTORY = "./model_AE/Swiss/"
        
    def __init__(self, exGAN):
        self.epsilon = 0.2
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
        self.data_func = utils.MNIST_load
        self.MODEL_DIRECTORY = "./model_AE/MNIST_lt3/"

    def __init__(self, exGAN):
        self.epsilon = 0.4
        super(FIT_AE_MNIST, self).__init__(exGAN)

    def get_image_dim(self):
        return 784
    def get_latent_dim(self):
        return 50

    def add_noise(self, batch_xs):
        nmax = np.clip(batch_xs + self.epsilon, 0, 1)
        nmin = np.clip(batch_xs - self.epsilon, 0, 1)
        noised = np.random.uniform(nmin, nmax)
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

##########################################################################
class FIT_AE_MNIST_V2(FIT_AE_MNIST):
    def define_data_dir(self):
        self.data_func = utils.MNIST_load
        self.MODEL_DIRECTORY = "./model_AE/MNIST_V2/"

    def get_latent_dim(self):
        return 128

##########################################################################
class FIT_AE_F_MNIST(FIT_AE_MNIST):
    def define_data_dir(self):
        self.data_func = utils.F_MNIST_load
        self.MODEL_DIRECTORY = "./model_AE/F_MNIST/"
    
    def define_default_param(self):
        self.BATCH_SIZE = 100
        self.ITERS = 50001

    def __init__(self, exGAN):
        self.epsilon = 0.3
        super(FIT_AE_MNIST, self).__init__(exGAN)
        
    def get_latent_dim(self):
        return 128

##########################################################################
class FIT_AE_CIFAR10(FIT_AE):
    def define_data_dir(self):
        self.data_func = utils.cifar10_load
        self.MODEL_DIRECTORY = "./model_AE/CIFAR10/"

    def __init__(self, exGAN):
        self.epsilon = 0.1
        super(FIT_AE_CIFAR10, self).__init__(exGAN)

    def get_image_dim(self):
        return 12228
    def get_latent_dim(self):
        return 100

    def add_noise(self, batch_xs):
        nmax = np.clip(batch_xs + self.epsilon, 0, 1)
        nmin = np.clip(batch_xs - self.epsilon, 0, 1)
        noised = np.random.uniform(nmin, nmax)
        return noised

    def test_generate(self, sess, train_gen, n_samples = 512, filename='samples.png'):
        p_size = self.exGAN.PROJ_BATCH_SIZE
        for i in range(1):
            batch, _ = next(train_gen)
            batch = batch[:p_size, :]
            rx, rz = sess.run([self.rx, self.z], feed_dict={self.x_hat: batch, self.x: batch})

            # proj_img = self.exGAN.find_proj(sess, batch, rz)

            utils.save_images(rx.reshape(p_size, 64, 64, 3), 'images/reconstr.png')
            utils.save_images(batch.reshape(p_size, 64, 64, 3), 'images/original.png')
            # utils.save_images(proj_img.reshape(p_size, 28, 28), 'images/projection.png')

    # Gaussian MLP as encoder
    def gaussian_MLP_encoder(self, x, n_hidden=256, reuse=False):
        with tf.variable_scope("gaussian_MLP_encoder", reuse=reuse):
            output = tf.reshape(x, [-1, 64, 64, 3])

            output = tf.layers.conv2d(output, 128, 5)
            output = utils.LeakyReLU(output)

            output = tf.layers.conv2d(output, 64, 5)
            output = utils.LeakyReLU(output)

            output = tf.layers.conv2d(output, 32, 5)
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
    
    
"""    
##########################################################################
class FIT_AE_MNIST_MINMAX(FIT_AE_MINMAX):
    def define_data_dir(self):
        self.data_func = utils.MNIST_load
        self.MODEL_DIRECTORY = "./model_AE/MNIST/"

    def get_image_dim(self):
        return 784

    def get_latent_dim(self):
        return 15

    def add_noise(self, batch_xs):
        noised = batch_xs + np.random.normal(self.epsilon / 2, self.epsilon, size=batch_xs.shape)
        noised = np.clip(noised, 0., 1.)
        return noised

    def test_generate(self, sess, train_gen, n_samples=512, filename='samples.png'):
        p_size = self.exGAN.PROJ_BATCH_SIZE
        for i in range(1):
            batch, _ = next(train_gen)
            batch = batch[:p_size, :]
            rx, rz = sess.run([self.rx, self.z], feed_dict={self.x_hat: batch, self.x: batch})

            # proj_img = self.exGAN.find_proj(sess, batch, rz)

            utils.save_images(rx.reshape(p_size, 28, 28), 'images/reconstr.png')
            utils.save_images(batch.reshape(p_size, 28, 28), 'images/original.png')
            # utils.save_images(proj_img.reshape(p_size, 28, 28), 'images/projection.png')

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
            gaussian_params = tf.layers.dense(output, 2 * self.get_latent_dim())

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.get_latent_dim()]
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.get_latent_dim():])
            z = mean + stddev * tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)

        return mean, stddev, z
"""

    
    
########################################################3 
### CELEBA
########################################################3

class FIT_AE_CelebA(FIT_AE_CIFAR10):
    def define_default_param(self):
        self.BATCH_SIZE = 100
        self.ITERS = 100001
        
    def define_data_dir(self):
        self.data_func = utils.CelebA_load
        self.MODEL_DIRECTORY = "./model_AE/CelebA/"

    def __init__(self, exGAN):
        self.epsilon = 0.1
        super(FIT_AE_CelebA, self).__init__(exGAN)

    def get_image_dim(self):
        return 64 * 64 * 3

    def get_latent_dim(self):
        return 128
    
    def get_train_gen(self, sess, num_epochs = 10):
        train_gen, _, _ = utils.load_dataset(self.BATCH_SIZE, self.data_func, True)
        return utils.batch_gen(train_gen)
    
    def add_noise(self, batch_xs):
        noised = batch_xs + np.random.uniform(-self.epsilon, self.epsilon, size=batch_xs.shape)
        noised = np.clip(noised, 0., 1.)
        return noised

    def test_generate(self, sess, train_gen, n_samples = 512, filename='samples.png'):
        p_size = self.exGAN.PROJ_BATCH_SIZE
        for i in range(1):
            batch, _ = next(train_gen)
            batch = batch[:p_size, :]
            rx, rz = sess.run([self.rx, self.z], feed_dict={self.x_hat: batch, self.x: batch})

            # proj_img, _ = self.exGAN.find_proj(sess, batch, rz)

            utils.save_images(rx.reshape(p_size, 64, 64, 3), 'images/reconstr.png')
            utils.save_images(batch.reshape(p_size, 64, 64, 3), 'images/original.png')
            #utils.save_images(proj_img.reshape(p_size, 64, 64, 3), 'images/projection.png')

    # Gaussian MLP as encoder
    def gaussian_MLP_encoder(self, x, n_hidden=256, reuse=False):
        with tf.variable_scope("gaussian_MLP_encoder", reuse=reuse):
            x = tf.reshape(x, [self.BATCH_SIZE, 64, 64, 3])
            
            conv1 = tf.nn.relu(batch_normal(conv2d(x, output_dim=64, name='e_c1'), scope='e_bn1'))
            conv2 = tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_bn2'))
            conv3 = tf.nn.relu(batch_normal(conv2d(conv2 , output_dim=256, name='e_c3'), scope='e_bn3'))
            conv3 = tf.reshape(conv3, [self.BATCH_SIZE, 256 * 8 * 8])
            fc1 = tf.nn.relu(batch_normal(fully_connect(conv3, output_size=1024, scope='e_f1'), scope='e_bn4'))
            z_mean = fully_connect(fc1, output_size=self.get_latent_dim(), scope='e_f2')
            
        return z_mean, None, z_mean
