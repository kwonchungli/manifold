import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf
import utils

BATCH_SIZE = 1024 # Batch size
ITERS = 100001 # How many generator iterations to train for 
CRITIC_ITERS = 50 # For WGAN and WGAN-GP, number of critic iters per gen iter
PROJ_ITER = 2000

class WGAN(object):
    def __init__(self):
        self.LAMBDA = .1 # Gradient penalty lambda hyperparameter
        
        self.z_in = tf.placeholder(tf.float32, shape=[None, self.get_latent_dim()], name='latent_variable')
        self.data = tf.placeholder(tf.float32, shape=[None, self.get_image_dim()])
        
        self.Generator = self.build_generator(self.z_in)
        self.Discriminator_fake = self.build_discriminator(self.Generator)
        self.Discriminator_real = self.build_discriminator(self.data, True)
        
        self.gen_params = [var for var in tf.trainable_variables() if 'Generator' in var.name]
        self.disc_params = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
        
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
        fake_data = self.Generator
        disc_fake = self.Discriminator_fake
        disc_real = self.Discriminator_real

        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        alpha = tf.random_uniform(
            shape=[BATCH_SIZE,1], 
            minval=0.,
            maxval=1.
        )
        
        differences = fake_data - self.data
        interpolates = self.data + (alpha*differences)
        gradients = tf.gradients(self.build_discriminator(interpolates, True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += self.LAMBDA*gradient_penalty

        self.proj_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
                3e-3,  # Base learning rate.
                self.proj_step,  # Current index into the dataset.
                1000,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
        disc_rate = tf.train.exponential_decay(
                1e-4,  # Base learning rate.
                self.proj_step,  # Current index into the dataset.
                2000,  # Decay step.
                0.9,  # Decay rate.
                staircase=True)
        
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate, 
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_cost, var_list=self.gen_params, global_step=self.proj_step)
        
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=disc_rate, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(disc_cost, var_list=self.disc_params)

        return disc_cost, gen_train_op, disc_train_op
    
    def train(self, session):
        # Dataset iterator
        train_gen, _, _ = utils.load_dataset(BATCH_SIZE, self.data_func)
        
        # cache variables
        disc_cost, gen_train_op, disc_train_op = self.disc_cost, self.gen_train_op, self.disc_train_op
        
        # Train loop
        noise_size = (BATCH_SIZE, self.get_latent_dim())
        for iteration in range(ITERS):
            if iteration > 0:
                _ = session.run(gen_train_op, 
                                feed_dict={self.z_in: self.noise_gen(noise_size)})

            # Run discriminator
            disc_iters = CRITIC_ITERS
            for i in range(disc_iters):
                _data, label = next(utils.batch_gen(train_gen))
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={
                        self.z_in: self.noise_gen(noise_size),
                        self.data: _data}
                )
                
                if( iteration % 100 == 10 ):
                    print 'disc_cost: ', -_disc_cost

            if( iteration % 100 == 10 ):
                print '-------------------------------------'
                
            # Calculate dev loss and generate samples every 100 iters
            if iteration % 1000 == 10:
                self.test_generate(session, filename='train_samples.png')

            # Checkpoint
            if( iteration % 1000 == 999 ):
                print 'Saving model...'
                self.saver.save(session, self.MODEL_DIRECTORY+'checkpoint-'+str(iteration))
                self.saver.export_meta_graph(self.MODEL_DIRECTORY+'checkpoint-'+str(iteration)+'.meta')

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
        self.test_x = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.OUTPUT_DIM])
        self.z_hat = tf.get_variable('z_hat', shape=[self.BATCH_SIZE, self.INPUT_DIM], dtype=tf.float32)
        self.out = self.build_generator(self.z_hat, True)
        
        self.proj_loss = tf.reduce_mean(tf.square(self.out - self.test_x))
        
        self.proj_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
                1e-2,  # Base learning rate.
                self.proj_step * self.CLASSIFIER_BATCH_SIZE,  # Current index into the dataset.
                PROJ_ITER,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
        
        self.proj_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate, 
            beta1=0.5,
            beta2=0.9
        ).minimize(self.proj_loss, var_list=self.z_hat, global_step=self.proj_step)
        
        # Takes about 30 sec
        self.grad = jacobian(self.out, self.z_hat)

    def find_proj(self, sess, batch_x):
        thresh, cost, iterat = 0.005, 1.0, 0
        while( cost > thresh ):
            sess.run(self.proj_step.assign(0))
            sess.run(self.z_hat.assign(np.random.uniform(-1, 1, size=self.z_hat.shape.as_list())))
            for i in range(PROJ_ITER):
                _, cost = sess.run([self.proj_op, self.proj_loss], feed_dict={self.test_x:batch_x})
                if( i % 500 == 0 ):
                    print ('Projection Cost is : ', cost)
                    
            thresh = thresh + 0.001
        
            lib.save_images.save_images(
            self.out.eval().reshape((self.CLASSIFIER_BATCH_SIZE, 28, 28)), 'test_proj.png'
        )
        D = sess.run(self.grad)
        return D
        