import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from AE_Network import *

class F_AAE_MNIST(FIT_AE_MNIST_V2):
    def define_data_dir(self):
        self.MODEL_DIRECTORY = "./model_F_AAE/MNIST/"
        
    def define_saver(self):
        self.saver = tf.train.Saver(var_list=self.enc_params + self.exGAN.gen_params + self.exGAN.disc_params, max_to_keep=1)

    def define_loss(self):
        # reconstruction loss in Z-space
        noisy_x = self.decoded
        noisy_x = noisy_x + tf.random_normal(tf.shape(noisy_x), self.epsilon/2, self.epsilon, dtype=tf.float32)
        
        self.rz = self.gaussian_MLP_encoder(tf.clip_by_value(noisy_x, 0, 1), reuse = True)
        self.res_loss_z = tf.reduce_mean(tf.norm(self.z_in - self.rz, ord=2, axis=1))
        
        # reconstruction loss in X-space
        self.disc_loss = self.exGAN.disc_cost
        self.gen_loss = self.exGAN.gen_cost + 1.0 * self.res_loss_z

        # learning rate
        self.train_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
                1e-3,  # Base learning rate.
                self.train_step,  # Current index into the dataset.
                5000,  # Decay step.
                0.96,  # Decay rate.
                staircase=True)

        # optimizer
        self.encode_params = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        
        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate, name="ex"
        ).minimize(self.gen_loss, var_list=self.exGAN.gen_params + self.encode_params, global_step=self.train_step)
        
        self.disc_train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate, name="ex"
        ).minimize(self.disc_loss, var_list=self.exGAN.disc_params)

        return self.res_loss_z, None

    def train(self, session):
        # Dataset iterator
        train_gen, _, _ = utils.load_dataset(self.BATCH_SIZE, self.data_func)
        
        noise_size = (self.BATCH_SIZE, self.get_latent_dim())
        train_gen = utils.batch_gen(train_gen)
        
        # cache variables
        disc_cost, gen_train_op, disc_train_op = self.disc_loss, self.gen_train_op, self.disc_train_op
        
        # Train loop
        noise_size = (self.BATCH_SIZE, self.get_latent_dim())
        for iteration in range(self.ITERS):
            if iteration > 0:
                this_noise = self.noise_gen(noise_size)
                _, vz = session.run([gen_train_op, self.res_loss_z], 
                                feed_dict={self.exGAN.z_in: this_noise,
                                          self.z_in: this_noise})

            # Run discriminator
            disc_iters = self.exGAN.CRITIC_ITERS
            for i in range(disc_iters):
                _data, label = next(train_gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={
                        self.exGAN.z_in: self.noise_gen(noise_size),
                        self.exGAN.data: _data}
                )
                
                if( iteration % 100 == 10 ):
                    print 'disc_cost: ', -_disc_cost

            if( iteration % 100 == 10 ):
                print '---------------res_loss_z : ', vz ,'----------------------'
                
            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 10:
                self.exGAN.test_generate(session, filename='images/train_samples.png')
                self.test_generate(session, train_gen, filename='images/train_samples.png')

            # Checkpoint
            if( iteration % 1000 == 999 ):
                print 'Saving model...'
                self.saver.save(session, self.MODEL_DIRECTORY+'checkpoint-'+str(iteration))
                self.saver.export_meta_graph(self.MODEL_DIRECTORY+'checkpoint-'+str(iteration)+'.meta')

                
class F_AAE_Swiss(FIT_AE_Swiss):
    def define_data_dir(self):
        self.MODEL_DIRECTORY = "./model_F_AAE/Swiss/"
        
    def define_saver(self):
        self.saver = tf.train.Saver(var_list=self.enc_params + self.exGAN.gen_params + self.exGAN.disc_params, max_to_keep=1)

    def define_loss(self):
        # reconstruction loss in Z-space
        noisy_x = self.decoded
        noisy_x = noisy_x + tf.random_normal(tf.shape(noisy_x), self.epsilon/2, self.epsilon, dtype=tf.float32)
        
        self.rz = self.gaussian_MLP_encoder(noisy_x, reuse = True)
        self.res_loss_z = tf.reduce_mean(tf.norm(self.z_in - self.rz, ord=2, axis=1))
        
        # reconstruction loss in X-space
        self.disc_loss = self.exGAN.disc_cost
        self.gen_loss = self.exGAN.gen_cost + 10.0 * self.res_loss_z

        # learning rate
        self.train_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
                1e-3,  # Base learning rate.
                self.train_step,  # Current index into the dataset.
                5000,  # Decay step.
                0.96,  # Decay rate.
                staircase=True)

        # optimizer
        self.encode_params = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        
        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate, name="ex"
        ).minimize(self.gen_loss, var_list=self.exGAN.gen_params + self.encode_params, global_step=self.train_step)
        
        self.disc_train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate, name="ex"
        ).minimize(self.disc_loss, var_list=self.exGAN.disc_params)

        return self.res_loss_z, None

    def train(self, session):
        # Dataset iterator
        train_gen, _, _ = utils.load_dataset(self.BATCH_SIZE, self.data_func)
        
        noise_size = (self.BATCH_SIZE, self.get_latent_dim())
        train_gen = utils.batch_gen(train_gen)
        
        # cache variables
        disc_cost, gen_train_op, disc_train_op = self.disc_loss, self.gen_train_op, self.disc_train_op
        
        # Train loop
        noise_size = (self.BATCH_SIZE, self.get_latent_dim())
        for iteration in range(self.ITERS):
            if iteration > 0:
                this_noise = self.noise_gen(noise_size)
                _, vz = session.run([gen_train_op, self.res_loss_z], 
                                feed_dict={self.exGAN.z_in: this_noise,
                                          self.z_in: this_noise})

            # Run discriminator
            disc_iters = self.exGAN.CRITIC_ITERS
            for i in range(disc_iters):
                _data, label = next(train_gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={
                        self.exGAN.z_in: self.noise_gen(noise_size),
                        self.exGAN.data: _data}
                )
                
                if( iteration % 100 == 10 ):
                    print 'disc_cost: ', -_disc_cost

            if( iteration % 100 == 10 ):
                print '---------------res_loss_z : ', vz ,'----------------------'
                
            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 10:
                self.exGAN.test_generate(session, filename='images/train_samples.png')
                self.test_generate(session, train_gen, filename='images/train_samples.png')

            # Checkpoint
            if( iteration % 1000 == 999 ):
                print 'Saving model...'
                self.saver.save(session, self.MODEL_DIRECTORY+'checkpoint-'+str(iteration))
                self.saver.export_meta_graph(self.MODEL_DIRECTORY+'checkpoint-'+str(iteration)+'.meta')                