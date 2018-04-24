import tensorflow as tf
from generator_models import cifar_generator
from classifiers import cifar10_classifier
import numpy as np

class CIFAR10_AE(FIT_AE):
    def define_data_dir(self):
        self.MODEL_DIRECTORY = "./model_AE/cifar10/"

    def __init__(self):
        self.data_func = utils.cifar10_load
        self.epsilon = 0.4

        self.define_data_dir()
        super(FIT_AE_CIFAR10, self).__init__(exGAN)

    def get_image_dim(self):
        return 64*64
    def get_latent_dim(self):
        return 128

    def add_noise(self, batch_xs):
        noised = batch_xs + np.random.normal(self.epsilon/2, self.epsilon/2) * (np.random.randint(4, size=batch_xs.shape) - 1)
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
      im = tf.placeholder(tf.float32,shape=[None,64,64,3])
      gen_func, gen_vars = cifar_generator(z, reuse=False)
      return gen_func

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

    def autoencoder(self, x_hat, x, n_hidden=256, reuse=False):
        mu, sigma, z = self.gaussian_MLP_encoder(x_hat, n_hidden, reuse)
        y = self.decoder(mu)
        return z, y

    def restore_session(self, sess):
      gen_saver = tf.train.Saver(gen_vars)
      gen_saver.restore(sess,'Celeb_A/data/CelebA_gen/model.ckpt-102951')

def cifar10_model_fn(x):
  cla_func, preds, cla_vars = cifar10_classifier(x, reuse=False)
  return cla_func

def eval_cifar10_model():
  cla_saver = tf.train.Saver(cla_vars)
  cla_saver.restore(sess,'Celeb_A/data/CelebA_classifier/model-999') #tf.train.latest_checkpoint('/home/jaylewis/Desktop/Dimakis_Models/Celeb_A/data/CelebA_classifier/')
