import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

class AE(object):
    def define_default_param(self):
        self.BATCH_SIZE = 128
        self.ITERS = 100001
        
    def define_saver(self):
        self.saver = tf.train.Saver(var_list=self.enc_params + self.dec_params, max_to_keep=1)
        
    def __init__(self):
        self.define_default_param()

        # define inputs
        self.x_hat = tf.placeholder(tf.float32, shape=[None, self.get_image_dim()], name='copy_img')
        self.x = tf.placeholder(tf.float32, shape=[None, self.get_image_dim()], name='input_img')
        self.z, self.rx = self.autoencoder(self.x_hat, self.x)

        # input for decoding only
        self.z_in = tf.placeholder(tf.float32, shape=[None, self.get_latent_dim()], name='latent_noise')
        self.decoded = self.decoder(self.z_in, self.get_image_dim())

        # optimization
        self.loss, self.train_op = self.define_loss()

        self.enc_params = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        self.dec_params = [var for var in tf.trainable_variables() if 'decoder' in var.name]
        
        self.define_saver()        
        
    def get_image_dim(self):
        return 0
    def get_latent_dim(self):
        return 0

    # Restore
    def restore_session(self, sess, checkpoint_dir = None):
        if(checkpoint_dir == None):
            checkpoint_dir = self.MODEL_DIRECTORY

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        self.saver.restore(sess, ckpt.model_checkpoint_path)


    # Gaussian MLP as encoder
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

    def test_generate(self, sess, train_gen, n_samples = 64000, filename='images/samples.png'):
        pass

    def train(self, sess):
        pass

    def noise_gen(self, noise_size):
        return np.random.normal(size=noise_size).astype('float32')





##########################################################
class FIT_AE(AE):
    def __init__(self, exGAN):
        self.LAMBDA = 5
        self.exGAN = exGAN

        super(FIT_AE, self).__init__()

    def define_saver(self):
        self.saver = tf.train.Saver(var_list=self.enc_params, max_to_keep=1)

    def decoder(self, z, dim_img, n_hidden=256):
        return self.exGAN.build_generator(z, reuse=True)

    # Gateway
    def autoencoder(self, x_hat, x, n_hidden=256, reuse=False):
        # encoding
        mu, sigma, z = self.gaussian_MLP_encoder(x_hat, n_hidden, reuse)

        # decoding
        y = self.exGAN.build_generator(mu, reuse=True)

        return z, y

    def define_loss(self):
        # reconstruction loss in X-space
        resconstruct_loss = tf.reduce_mean(tf.norm(self.rx - self.x, ord=2, axis=1))
        self.res_loss = resconstruct_loss

        # reconstruction loss in Z-space
        noisy_x = self.decoded
        noisy_x = noisy_x + tf.random_normal(tf.shape(noisy_x), self.epsilon/2, self.epsilon, dtype=tf.float32)
        self.rz = self.gaussian_MLP_encoder(tf.clip_by_value(noisy_x, 0, 1), reuse = True)
        self.res_loss_z = tf.reduce_mean(tf.norm(self.z_in - self.rz, ord=2, axis=1))

        # get trainable params
        self.encode_params = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        self.decode_params = self.exGAN.gen_params

        # define loss & training operation
        self.global_step = tf.Variable(0)
        loss = resconstruct_loss + self.res_loss_z * 10

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
            name="auto").minimize(loss, var_list=self.encode_params, global_step=self.global_step)

        return loss, None

    def add_noise(self, batch_xs):
         return batch_xs + np.random.normal(0, self.epsilon, size=batch_xs.shape)

    def train(self, sess):
        # Dataset iterator
        train_gen, _, _ = utils.load_dataset(self.BATCH_SIZE, self.data_func)

        noise_size = (self.BATCH_SIZE, self.get_latent_dim())
        train_gen = utils.batch_gen(train_gen)

        # Train loop
        for iteration in range(self.ITERS):
            batch_xs, _ = next(train_gen)
            batch_noise = self.add_noise(batch_xs)

            _, rs_loss, rz_loss = sess.run(
                (self.en_train_op, self.res_loss, self.res_loss_z),
                feed_dict={self.z_in: self.noise_gen(noise_size), self.x_hat: batch_noise, self.x: batch_xs})

            # Calculate dev loss and generate samples every 1000 iters
            if iteration % 1000 == 10:
                print ('at iteration : ', iteration, ' loss : ', rs_loss, ', z_loss : ', rz_loss)
                self.test_generate(sess, train_gen, filename='images/train_samples.png')

            if( iteration % 10000 == 9999 ):
                print 'Saving model...'
                self.saver.save(sess, self.MODEL_DIRECTORY+'checkpoint-'+str(iteration))
                self.saver.export_meta_graph(self.MODEL_DIRECTORY+'checkpoint-'+str(iteration)+'.meta')

    def autoencode_dataset(self, sess, adversarial_x):
        i = 0
        batch_size = adversarial_x.shape[0]
        proj_img, proj_z = [], []
        while( i < batch_size ):
            ni = i + self.exGAN.PROJ_BATCH_SIZE
            rx, rz = sess.run([self.rx, self.z], feed_dict={self.x_hat: adversarial_x[i:ni, :]})
            rx, rz = self.exGAN.find_proj(sess, adversarial_x[i:ni, :], rz)

            i = ni
            proj_img.append(rx)
            proj_z.append(rz)

        return np.vstack(proj_img), np.vstack(proj_z)


##########################################################
class FIT_AE_MINMAX(FIT_AE):
    def __init__(self, exGAN):
        self.LAMBDA = 5
        self.exGAN = exGAN
        self.z_projection = tf.placeholder(tf.float32, shape=[self.exGAN.PROJ_BATCH_SIZE, self.get_latent_dim()], name='z_projection')
        self.MODEL_SAVE_DIRECTORY = './model_AE/MNIST_MINMAX/'

        super(FIT_AE_MINMAX, self).__init__(exGAN)

    def define_loss(self):

        # reconstruction loss in X-space (noisy)
        # resconstruct_loss = tf.reduce_mean(tf.norm(self.rx - self.x, ord=2, axis=1))
        # self.res_loss = resconstruct_loss

        # reconstruction loss in Z-space (noisy)
        noisy_x = self.decoded
        noisy_x = noisy_x + tf.random_normal(tf.shape(noisy_x), self.epsilon / 2, self.epsilon, dtype=tf.float32)
        self.rz = self.gaussian_MLP_encoder(tf.clip_by_value(noisy_x, 0, 1), reuse=True)
        self.res_loss_z = tf.reduce_mean(tf.norm(self.z_projection - self.rz, ord=2, axis=1))

        #----------------------------------------

        # # reconstruction loss in X-space (MinMax)
        # resconstruct_loss = tf.reduce_mean(tf.norm(self.rx - self.x, ord=2, axis=1))
        # self.res_loss = resconstruct_loss
        #
        # # reconstruction loss in Z-space (MinMax)
        # noisy_x = self.decoded
        # noisy_x = noisy_x + tf.random_normal(tf.shape(noisy_x), self.epsilon / 2, self.epsilon, dtype=tf.float32)
        # self.rz = self.gaussian_MLP_encoder(tf.clip_by_value(noisy_x, 0, 1), reuse=True)
        # self.res_loss_z = tf.reduce_mean(tf.norm(self.z_projection - self.rz, ord=2, axis=1))

        # ----------------------------------------

        # get trainable params
        self.encode_params = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        self.decode_params = self.exGAN.gen_params

        # define loss & training operation
        self.global_step = tf.Variable(0)

        # For now, forget about reconstruction loss
        # loss = resconstruct_loss + self.res_loss_z * 10
        loss = self.res_loss_z

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
            name="auto").minimize(loss, var_list=self.encode_params, global_step=self.global_step)

        return loss, None

    def add_noise(self, batch_xs):
        return batch_xs + np.random.normal(0, self.epsilon, size=batch_xs.shape)

    def train(self, sess):
        # Dataset iterator
        train_gen, _, _ = utils.load_dataset(self.BATCH_SIZE, self.data_func)

        noise_size = (self.BATCH_SIZE, self.get_latent_dim())
        train_gen = utils.batch_gen(train_gen)

        # Train loop
        for iteration in range(self.ITERS):
            batch_xs, _ = next(train_gen)
            batch_noise = self.add_noise(batch_xs)

            # z0 = self.exGAN.noise_gen([self.exGAN.PROJ_BATCH_SIZE,self.exGAN.get_latent_dim()])
            # _,zstar = self.exGAN.find_proj(sess, batch_noise, z0=z0)
            _,zstar = self.autoencode_dataset(sess,batch_noise)

            _, rs_loss, rz_loss = sess.run(
                (self.en_train_op, self.res_loss, self.res_loss_z),
                feed_dict={self.z_in: self.noise_gen(noise_size),
                           self.x_hat: batch_noise,
                           self.x: batch_xs,
                           self.z_projection: zstar})

            # Calculate dev loss and generate samples every 1000 iters
            if iteration % 1000 == 10:
                print ('at iteration : ', iteration, ' loss : ', rs_loss, ', z_loss : ', rz_loss)
                self.test_generate(sess, train_gen, filename='images/train_samples.png')

            if (iteration % 10000 == 9999):
                print 'Saving model...'
                self.saver.save(sess, self.MODEL_SAVE_DIRECTORY + 'checkpoint-' + str(iteration))
                self.saver.export_meta_graph(self.MODEL_SAVE_DIRECTORY + 'checkpoint-' + str(iteration) + '.meta')

    def autoencode_dataset(self, sess, adversarial_x):
        i = 0
        batch_size = adversarial_x.shape[0]
        proj_img, proj_z = [], []
        while (i < batch_size):
            ni = i + self.exGAN.PROJ_BATCH_SIZE
            rx, rz = sess.run([self.rx, self.z], feed_dict={self.x_hat: adversarial_x[i:ni, :]})
            rx, rz = self.exGAN.find_proj(sess, adversarial_x[i:ni, :], rz)

            i = ni
            proj_img.append(rx)
            proj_z.append(rz)

        return np.vstack(proj_img), np.vstack(proj_z)