import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

BATCH_SIZE = 512 # Batch size
ITERS = 100001 # How many generator iterations to train for 

from WAE_Network import AE

class VAE(AE):
    def __init__(self):
        self.data_func = utils.test_2d
        self.MODEL_DIRECTORY = './model_WAE/TEST/'
        
        super(VAE, self).__init__()
        
    def gaussian_MLP_encoder(self, x, n_hidden=256, reuse=False):
        with tf.variable_scope("gaussian_MLP_encoder", reuse=reuse):
            layer_1 = tf.layers.dense(x, n_hidden)
            layer_1 = tf.nn.relu(layer_1)
            
            layer_2 = tf.layers.dense(layer_1, n_hidden)
            layer_2 = tf.nn.relu(layer_2)
            
            layer_3 = tf.layers.dense(layer_2, n_hidden)
            layer_3 = tf.nn.relu(layer_3)
            
            gaussian_params = tf.layers.dense(layer_3, 2*self.get_latent_dim())

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.get_image_dim()]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.get_image_dim():])

        return mean, stddev
    
    # Gateway
    def autoencoder(self, x_hat, x, n_hidden=256, reuse=False):
        # encoding
        mu, sigma = self.gaussian_MLP_encoder(x_hat, n_hidden, reuse)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        y = self.bernoulli_MLP_decoder(z, n_hidden, reuse)
        
        # loss
        marginal_likelihood = -tf.squared_difference(x, y)
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)
        
        self.ELBO = marginal_likelihood - KL_divergence
        return z, y
        
    
    def define_loss(self):
        loss = -self.ELBO
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
        return loss, train_op
        
    def train(self, sess):
        train_gen, _, _ = utils.load_dataset(BATCH_SIZE, self.data_func)
        
        # Train loop
        gen = utils.batch_gen(train_gen)
        for iteration in range(ITERS):
            batch_xs, _ = next(gen) 
            batch_noise = batch_xs

            _, tot_loss = sess.run(
                    (self.train_op, self.loss),
                        feed_dict={self.x_hat: batch_noise, self.x: batch_xs})

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 10000 == 10:
                print ('at iteration : ', iteration, ' loss : ', tot_loss)
                self.test_generate(sess, batch_xs, filename='train_samples.png')
