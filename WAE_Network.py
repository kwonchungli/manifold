import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

BATCH_SIZE = 512 # Batch size
ITERS = 100001 # How many generator iterations to train for 

class AE(object):
    def __init__(self):
        # define inputs
        self.x_hat = tf.placeholder(tf.float32, shape=[None, self.get_image_dim()], name='copy_img')
        self.x = tf.placeholder(tf.float32, shape=[None, self.get_image_dim()], name='input_img')
        self.z, self.rx = self.autoencoder(self.x_hat, self.x)
        
        # input for decoding only
        self.z_in = tf.placeholder(tf.float32, shape=[None, self.get_latent_dim()], name='latent_noise')
        self.decoded = self.decoder(self.z_in, self.get_image_dim())
        
        # optimization
        self.loss, self.train_op = self.define_loss()
        self.saver = tf.train.Saver(max_to_keep=1)
        
    def get_image_dim(self):
        return 2
    def get_latent_dim(self):
        return 1

    # Restore
    def restore_session(self, sess, checkpoint_dir = None):
        if(checkpoint_dir == None):
            checkpoint_dir = self.MODEL_DIRECTORY
            
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        
    # Gaussian MLP as encoder
    def gaussian_MLP_encoder(self, x, n_hidden=256, reuse=False):
        with tf.variable_scope("gaussian_MLP_encoder", reuse=reuse):
            layer_1 = tf.layers.dense(x, 256)
            layer_1 = tf.nn.relu(layer_1)
            
            layer_2 = tf.layers.dense(layer_1, 256)
            layer_2 = tf.nn.relu(layer_2)
            
            layer_3 = tf.layers.dense(layer_2, 256)
            layer_3 = tf.nn.relu(layer_3)
            
            gaussian_params = tf.layers.dense(layer_3, 2*self.get_latent_dim())

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.get_latent_dim()]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.get_latent_dim():])
            z = mean + stddev * tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
            
        return mean, stddev, z

    # Bernoulli MLP as decoder
    def bernoulli_MLP_decoder(self, z, n_hidden=256, reuse=False):
        with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
            # initializers
            layer_1 = tf.layers.dense(z, 256)
            layer_1 = tf.nn.relu(layer_1)
            
            layer_2 = tf.layers.dense(layer_1, 256)
            layer_2 = tf.nn.relu(layer_2)
            
            layer_3 = tf.layers.dense(layer_2, 256)
            layer_3 = tf.nn.relu(layer_3)
            
            layer_4 = tf.layers.dense(layer_3, 256)
            layer_4 = tf.nn.relu(layer_4)
            
            y = tf.layers.dense(layer_4, self.get_image_dim())

        return y
    
    # Gateway
    def autoencoder(self, x_hat, x, n_hidden=256, reuse=False):
        # encoding
        mu, sigma, z = self.gaussian_MLP_encoder(x_hat, n_hidden, reuse) 

        # decoding
        y = self.bernoulli_MLP_decoder(z, n_hidden, reuse)
        
        return z, y
    
    def define_loss(self):
        loss = tf.squared_difference(self.rx, self.x)
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
        
        return loss, train_op
        
    def decoder(self, z, dim_img, n_hidden=256):
        y = self.bernoulli_MLP_decoder(z, n_hidden, reuse=True)
        return y
    
    def test_generate(self, sess, train_gen, n_samples = 64000, filename='samples.png'):
        fig, ax = plt.subplots()
        
        noises = self.noise_gen((n_samples, self.get_latent_dim()))
        gen_points = sess.run(self.decoded,
                                 feed_dict={self.z_in: noises})
        
        plt.scatter(gen_points[:,0], gen_points[:,1], s=0.4, c='b', alpha=0.1)
        
        for i in range(5):
            batch, _ = next(train_gen)
            rx, res = sess.run([self.rx, self.z], feed_dict={self.x_hat: batch, self.x: batch})
            plt.scatter(batch[:5,0], batch[:5,1], s=10, c='r', alpha=1)
            #plt.scatter(res[:,0], 0, s=5, c='r', alpha=0.01)
            plt.scatter(rx[:5,0], rx[:5,1], s=10, c='g', alpha=1)
        
        fig.savefig(filename)
        plt.close()
                
    def train(self, sess):
        pass
        
    def noise_gen(self, noise_size):
        return np.random.normal(size=noise_size).astype('float32')

    
     
    
    
    
    
    
    
    
    
    
#########################################################################

CRITIC_ITERS = 10
from GAN_Framework import GAN

class GAN_WAE(GAN):
    def __init__(self, encoder):
        self.LAMBDA = .1
        self.data = tf.placeholder(tf.float32, shape=[None, self.get_image_dim()], name='data')
        _, _, self.qz = encoder(self.data, reuse=True)
        self.pz = tf.placeholder(tf.float32, shape=[None, self.get_latent_dim()], name='prior')
        
        self.Discriminator_fake = self.build_discriminator(self.qz)
        self.Discriminator_real = self.build_discriminator(self.pz, True)
        
        self.disc_params = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
        self.ae_cost, self.disc_cost, self.disc_train_op = self.define_loss()
        
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
    
    def define_loss(self):
        disc_fake = self.Discriminator_fake
        disc_real = self.Discriminator_real
        
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        ae_cost = -disc_cost

        alpha = tf.random_uniform(
            shape=[BATCH_SIZE,1], 
            minval=0.,
            maxval=1.
        )
        
        differences = self.qz - self.pz
        interpolates = self.pz + (alpha*differences)
        gradients = tf.gradients(self.build_discriminator(interpolates, True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += self.LAMBDA*gradient_penalty
        
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(disc_cost, var_list=self.disc_params)
        return ae_cost, disc_cost, disc_train_op
    
    def get_latent_dim(self):
        return 2
    def get_image_dim(self):
        return 2
    
    def train(self, session, train_gen, noise_gen, it):
        # cache variables
        disc_cost, disc_train_op = self.disc_cost, self.disc_train_op
        
        # Train loop
        noise_size = (BATCH_SIZE, self.get_latent_dim())
        
        # Run discriminator
        disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data, _ = next(train_gen)
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={
                    self.data: _data,
                    self.pz: noise_gen(noise_size)})
            
            if( it % 100 == 4 ):
                print ('at iteration : ', i, ' disc_loss : ', -_disc_cost)
        
##########################################################    
class WAE(AE):    
    def __init__(self, exGAN):
        self.data_func = utils.swiss_load
        self.MODEL_DIRECTORY = './model_WAE/TEST/'
        self.LAMBDA = 5
        self.exGAN = exGAN
        super(WAE, self).__init__()
    
    def decoder(self, z, dim_img, n_hidden=256):
        return self.exGAN.build_generator(z, reuse=True)
    
    # Gateway
    def autoencoder(self, x_hat, x, n_hidden=256, reuse=False):
        # encoding
        mu, sigma, z = self.gaussian_MLP_encoder(x_hat, n_hidden, reuse) 

        # decoding
        y = self.exGAN.build_generator(mu, reuse=True)
        
        return z, y
    
    def gan_penalty(self):
        # Pz = Qz test based on GAN in the Z space
        self.refGAN = GAN_WAE(self.gaussian_MLP_encoder)
        return self.refGAN.ae_cost
        
    def mmd_penalty(self, sample_qz, sample_pz):
        sigma2_p = 1
        n = BATCH_SIZE
        half_size = (n * n - n) / 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz
    
        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        # Median heuristic for the sigma^2 of Gaussian kernel
        sigma2_k = tf.nn.top_k(
            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

        res1 = tf.exp( - distances_qz / 2. / sigma2_k)
        res1 += tf.exp( - distances_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (n * n - n)
        res2 = tf.exp( - distances / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) * 2. / (n * n)
        stat = res1 - res2
        
        return tf.reduce_mean(stat)
    
    def define_loss(self):
        resconstruct_loss = tf.reduce_mean(tf.squared_difference(self.rx, self.x))
        self.res_loss = resconstruct_loss
        
        # Define GAN Loss
        self.qz = self.z
        self.pz = tf.placeholder(tf.float32, shape=[None, self.get_latent_dim()], name='latent_noise')
        matching_loss = self.mmd_penalty(self.qz, self.pz)
        
        """
        HackHack
        """
        self.rz = self.gaussian_MLP_encoder(self.decoded, reuse = True)
        self.res_loss_z = tf.reduce_mean(tf.squared_difference(self.z_in, self.rz))
        #matching_loss = self.gan_penalty()
        """ """
        
        self.loss = resconstruct_loss #+ self.LAMBDA * matching_loss
        
        self.encode_params = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        self.decode_params = self.exGAN.gen_params
        #self.decode_params = [var for var in tf.trainable_variables() if 'decoder' in var.name]
        
        self.global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
                1e-4,  # Base learning rate.
                self.global_step,  # Current index into the dataset.
                5000,  # Decay step.
                0.96,  # Decay rate.
                staircase=True)
        self.en_train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate, 
            beta1=0.5,
            beta2=0.9,
            name="auto").minimize(self.res_loss_z, var_list=self.encode_params, global_step=self.global_step)
        
        self.de_train_op = tf.train.AdamOptimizer(
            learning_rate=.5e-5, 
            beta1=0.5,
            beta2=0.9,
            name="auto2").minimize(self.loss, var_list=self.encode_params+self.decode_params)
        
        return self.loss, None
        
    def train(self, sess):
        # Dataset iterator
        train_gen, _, _ = utils.load_dataset(BATCH_SIZE, self.data_func)
        
        noise_size = (BATCH_SIZE, self.get_latent_dim())
        train_gen = utils.batch_gen(train_gen)
        
        # Train loop
        for iteration in range(ITERS):
            # HAHAHAHAHA SIBALSEKKI
            # self.refGAN.train(sess, train_gen, self.noise_gen, iteration)

            batch_xs, _ = next(train_gen)
            batch_noise = batch_xs #+ np.random.normal(0, 0.1, size=batch_xs.shape)

            _, rs_loss = sess.run(
                (self.en_train_op, self.res_loss),
                feed_dict={self.z_in: self.noise_gen(noise_size), self.x_hat: batch_noise, self.x: batch_xs, 
                           self.pz: self.noise_gen(noise_size)})
            """
            _, rs_loss = sess.run(
                (self.en_train_op, self.res_loss),
                feed_dict={self.x_hat: batch_noise, self.x: batch_xs, self.pz: self.noise_gen(noise_size), 
                           #self.refGAN.data: batch_xs, self.refGAN.pz: self.noise_gen(noise_size)
                          })
            """
            # Calculate dev loss and generate samples every 1000 iters
            if iteration % 1000 == 10:
                print ('at iteration : ', iteration, ' loss : ', rs_loss)
                self.test_generate(sess, train_gen, filename='train_samples.png')
                
        """
        for iteration in range(ITERS*2):
            batch_xs, _ = next(train_gen)
            batch_noise = batch_xs #+ np.random.normal(0, 0.1, size=batch_xs.shape)
            
            _, rs_loss = sess.run(
                    (self.de_train_op, self.res_loss),
                    feed_dict={self.x_hat: batch_noise, self.x: batch_xs, self.pz: self.noise_gen(noise_size)})

            # Calculate dev loss and generate samples every 1000 iters
            if iteration % 1000 == 10:
                print ('at iteration : ', iteration, ' loss : ', rs_loss)
                self.test_generate(sess, train_gen, filename='train_samples.png')
    
        """