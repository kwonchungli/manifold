import time
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf
import utils
import os.path

class GAN(object):
    def define_default_param(self):
        self.BATCH_SIZE = 128 # Batch size
        self.PROJ_ITER = 500
        self.PROJ_BATCH_SIZE = 5
        self.ITERS = 20001 # How many generator iterations to train for
        self.CRITIC_ITERS = 10 # For WGAN and WGAN-GP, number of critic iters per gen iter
    
    def define_data_dir(self):
        pass
    
    def __init__(self):
        self.define_data_dir()
        self.define_default_param()

        if not os.path.exists(self.MODEL_DIRECTORY):
            os.makedirs(self.MODEL_DIRECTORY)
            
        self.z_in = tf.placeholder(tf.float32, shape=[None, self.get_latent_dim()], name='latent_variable')
        self.data = tf.placeholder(tf.float32, shape=[None, self.get_image_dim()])

        self.Generator, self.gen_params = self.build_generator(self.z_in)
        self.Discriminator_fake = self.build_discriminator(self.Generator)
        self.Discriminator_real = self.build_discriminator(self.data, True)

        self.disc_params = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
        self.define_proj()

        self.disc_cost, self.gen_train_op, self.disc_train_op = self.define_loss()
        self.saver = tf.train.Saver(var_list=self.gen_params + self.disc_params, max_to_keep=1)

    # Restore
    def restore_session(self, sess, checkpoint_dir = None):
        if(checkpoint_dir == None):
            checkpoint_dir = self.MODEL_DIRECTORY

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    # Defining loss - different from gan to gan
    def define_loss(self):
        raise NotImplementedError
    def train(self, session):
        raise NotImplementedError

    def noise_gen(self, noise_size):
        return np.random.normal(size=noise_size).astype('float32')

    def test_generate(self, sess, n_samples = 128, filename='samples.png'):
        pass

    def get_image_dim(self):
        return 0

    def get_latent_dim(self):
        return 0

    #
    #
    #
    #
    #
    #
    #
    #
    # Deferred to sub-classes
    def define_proj(self):
        self.test_x = tf.placeholder(tf.float32, shape=[self.PROJ_BATCH_SIZE, self.get_image_dim()])
        self.z_hat = tf.get_variable('z_hat', shape=[self.PROJ_BATCH_SIZE, self.get_latent_dim()], dtype=tf.float32)
        self.out, _ = self.build_generator(self.z_hat, True)

        self.proj_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.out, self.test_x), axis=1))
        self.proj_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
                5e-2,  # Base learning rate.
                self.proj_step,  # Current index into the dataset.
                100,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)

        self.proj_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.proj_loss, var_list=self.z_hat, global_step=self.proj_step)

    def find_proj(self, sess, batch_x, z0, random_init = False, random_iter = 1):
        cost, iterat = 1.0, 0
        min_cost = 100000000.

        while( iterat < random_iter ):
            if( random_init ):
                z0 = np.random.normal(0, 1, size=(self.PROJ_BATCH_SIZE, self.get_latent_dim()))
                
            sess.run(self.z_hat.assign(z0))
            sess.run(self.proj_step.assign(0))

            for i in range(self.PROJ_ITER):
                _, cost = sess.run([self.proj_op, self.proj_loss], feed_dict={self.test_x: batch_x})
                # if( i % 50 == 0 ):
                #    print ('Projection Cost is :' , cost , 'z diff : ', np.linalg.norm(sess.run(self.z_hat[0]) - z0[0]))
                    
            if( cost < min_cost ):
                min_cost = cost
                proj_img = sess.run(self.out)
                proj_z = sess.run(self.z_hat)
                
            iterat = iterat + 1
        
        return proj_img, proj_z

""" WGAN Implementation Start """
class WGAN(GAN):
    def __init__(self):
        self.LAMBDA = .1 # Gradient penalty lambda hyperparameter
        super(WGAN, self).__init__()

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

    def define_loss(self):
        fake_data = self.Generator
        disc_fake = self.Discriminator_fake
        disc_real = self.Discriminator_real

        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        alpha = tf.random_uniform(
            shape=[self.BATCH_SIZE,1],
            minval=0.,
            maxval=1.
        )

        differences = fake_data - self.data
        interpolates = self.data + (alpha*differences)
        gradients = tf.gradients(self.build_discriminator(interpolates, True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += self.LAMBDA*gradient_penalty

        learning_rate, disc_rate = self.define_learning_rate()

        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_cost, var_list=self.gen_params, global_step=self.train_step)

        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=disc_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(disc_cost, var_list=self.disc_params)

        self.gen_cost, self.disc_cost = gen_cost, disc_cost
        return disc_cost, gen_train_op, disc_train_op

    def get_train_gen(self, sess):
        train_gen, _, _ = utils.load_dataset(self.BATCH_SIZE, self.data_func)
        return utils.batch_gen(train_gen)
        

    def train(self, session):
        # Dataset iterator
        train_gen = self.get_train_gen(session)

        # cache variables
        disc_cost, gen_train_op, disc_train_op = self.disc_cost, self.gen_train_op, self.disc_train_op

        # Train loop
        noise_size = (self.BATCH_SIZE, self.get_latent_dim())
        start_time = time.time()
        for iteration in range(self.ITERS):
            if iteration > 0:
                _ = session.run(gen_train_op,
                                feed_dict={self.z_in: self.noise_gen(noise_size)})

            # Run discriminator
            disc_iters = self.CRITIC_ITERS
            for i in range(disc_iters):
                _data, label = next(train_gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={
                        self.z_in: self.noise_gen(noise_size),
                        self.data: _data}
                )

                if( iteration % 100 == 10 ):
                    print 'disc_cost: ', -_disc_cost

            if( iteration % 100 == 10 ):
                print '---------------'
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time/iteration
                iter_left =self.ITERS-iteration
                time_left = int(avg_time * iter_left/60)
                now = datetime.datetime.now()
                print(now + datetime.timedelta(minutes=time_left))
                print('time left (minutes):'+str(time_left))
                print('ETA:'+str(now+datetime.timedelta(minutes=time_left)))
                print '-------------------------------------'

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 10:
                self.test_generate(session, filename='images/train_samples.png', print_flag=True)

            # Checkpoint
            if( iteration % 1000 == 999 ):
                print 'Saving model...'
                self.saver.save(session, self.MODEL_DIRECTORY+'checkpoint-'+str(iteration))
                self.saver.export_meta_graph(self.MODEL_DIRECTORY+'checkpoint-'+str(iteration)+'.meta')
