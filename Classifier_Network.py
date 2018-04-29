import numpy as np
import tensorflow as tf
import tensorlayer
from tensorlayer.layers import InputLayer, Conv2dLayer, MaxPool2d, LocalResponseNormLayer, FlattenLayer, DenseLayer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import PIL.Image

from attacks import get_adv_dataset
import utils
import attacks


class Classifier(object):
    def get_image_dim(self):
        return 0
    def get_class_num(self):
        return 0
    
    def set_model_dir(self):
        pass
        
    def __init__(self, inf_norm = 0.0, myVAE = None, batch_size = 100):
        self.x = tf.placeholder(tf.float32, shape=[None, self.get_image_dim()])
        self.y = tf.placeholder(tf.int32, shape=[None, self.get_class_num()])
        self.inf_norm = tf.placeholder_with_default(0.0001, shape=())
        self.noise_level = inf_norm
        
        self.BATCH_SIZE = batch_size
        self.VAE = myVAE
        
        self.dropout_rate = tf.placeholder_with_default(0.4, shape=())
        self.logits, self.class_vars = self.build_classifier(self.x, self.inf_norm)
        
        self.loss_op, self.train_op, self.pred_op = self.define_loss(self.logits, self.y)
        self.saver = tf.train.Saver(var_list=self.class_vars, max_to_keep=1)
        
        self.set_model_dir()
    
    def eval_model(self, sess, adv_level=0.05):
        test_gen = self.get_test_gen(sess)
        it = 0
        
        dropout_rate = self.dropout_rate
        x, y = self.x, self.y
        logits = self.logits
        accuracy_op = self.pred_op
        
        loss_1 = self.loss_op
        #loss_adv = tf.losses.softmax_cross_entropy(y, logits=adv_logits)
        #loss_z_l2 = tf.reduce_mean(tf.nn.l2_loss(myVAE.z - myVAE.z_in))

        #attack = attacks.LinfPGDAttack(loss_z_l2, myVAE.x_hat, 0.3, 30, 0.01, True)
        attack = attacks.LinfPGDAttack(loss_1, x, adv_level, 30, 0.01, True)
        #attack = attacks.LinfPGDAttack(loss_adv, myVAE.x_hat, 0.3, 30, 0.01, True)

        normal_avr, adv_avr_ref, adv_avr, reconstr_avr = 0, 0, 0, 0
        for x_test, y_test in test_gen:
            it = it + 1
            
            # Check Transfer
            #self.saver.restore(sess, './pretrained_models/cifar10/data/cifar10_classifier/model.ckpt-1000000')
            
            #adversarial_x = get_adv_dataset(sess, logits, x, y, x_test, y_test, adv_level)
            #adversarial_x = get_adv_dataset(sess, adv_logits, myVAE.x_hat, y, x_test, y_test)

            #_, z_in = myVAE.autoencode_dataset(sess, x_test)
            #adversarial_x = attack.perturb(x_test, y_test, myVAE.x_hat, y, sess, myVAE.z_in, z_in)
            
            adversarial_x = attack.perturb(x_test, y_test, x, y, sess)
            #adversarial_x = attack.perturb(x_test, y_test, myVAE.x_hat, y, sess)
            #cleaned_x, z_res = myVAE.autoencode_dataset(sess, adversarial_x)
            #print ('compare z vs z : ', (z_in[0] - z_res[0]), np.linalg.norm(z_in[0] - z_res[0]))

            #self.restore_session(sess)
            
            normal_avr += sess.run(accuracy_op, feed_dict={x: x_test, y: y_test, dropout_rate: 0.0})
            adv_avr_ref += sess.run(accuracy_op, feed_dict={x: adversarial_x, y: y_test, dropout_rate: 0.0})

            #adv_avr += sess.run(adv_acc_op, feed_dict={myVAE.x_hat: adversarial_x, y: y_test, dropout_rate: 0.0})
            #reconstr_avr += sess.run(accuracy_op, feed_dict={x: cleaned_x, y: y_test, dropout_rate: 0.0})

            if( it % 10 == 1 ):
                #test_pred = sess.run(logits, feed_dict={x: cleaned_x, y: y_test, dropout_rate: 0.0})

                #i1 = np.argmax(test_pred, 1)
                #i2 = np.argmax(y_test, 1)
                #index = np.where(np.not_equal(i1, i2))[0]

                #p_size = len(index)
                p_size = x_test.shape[0]
                wrong_x = x_test[:, :]
                wrong_adv = adversarial_x[:, :]
                #wrong_res = cleaned_x[:, :]
                
                self.test_generate(wrong_x, p_size, 'images/cl_original.png')
                self.test_generate(wrong_adv, p_size, 'images/cl_adversarial.png')
                
        print ("------------ Test ----------------")
        print("Normal Accuracy:", normal_avr / it)
        print("Normal Adversarial Accuracy:", adv_avr_ref / it)
        #print(name, "Adversarial Accuracy:", adv_avr / it)
        #print(name, "Reconstructed Accuracy:", reconstr_avr / it)

    def test_generate(self, x, n_sample, filename='images/results.png'):
        pass
        
    def build_classifier(self, im, inf_norm):
        pass
    
    def define_learning_rate(self):
        self.train_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
                1e-3,  # Base learning rate.
                self.train_step,  # Current index into the dataset.
                10000,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
        return learning_rate
    
    def define_loss(self, logits, y):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        learning_rate = self.define_learning_rate()
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        loss_op = tf.losses.softmax_cross_entropy(y, logits=logits)
        train_op = optimizer.minimize(loss_op, global_step=self.train_step)
        batch_pred = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        return loss_op, train_op, batch_pred

    def get_train_gen(self, sess, num_epochs = 10):
        train_gen, _, _ = utils.load_dataset(self.BATCH_SIZE, self.data_func)
        return utils.batch_gen(train_gen, True, self.y.shape[1], num_epochs)
    
    def get_test_gen(self, sess):
        _, _, test_gen = utils.load_dataset(self.BATCH_SIZE, self.data_func)
        return utils.batch_gen(test_gen, True, self.y.shape[1], 1)
    
    def train(self, sess):
        # Dataset iterator
        train_gen = self.get_train_gen(sess, 10)
        
        it = 0
        for x_train, y_train in train_gen:
            sess.run(self.train_op, feed_dict={self.x: x_train, self.y: y_train, self.inf_norm: self.noise_level})
            
            it = it + 1
            if ( it % 100 == 0 ):
                ls = sess.run(self.loss_op, feed_dict={self.x: x_train, self.y: y_train})
                print 'loss :', ls

            if( it % 1000 == 0 ):
                print 'Saving model...'
                self.saver.save(sess, self.filepath+'checkpoint-'+str(it))
                self.saver.export_meta_graph(self.filepath+'checkpoint-'+str(it)+'.meta')

    def restore_session(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.filepath)
        self.saver.restore(sess, ckpt.model_checkpoint_path)


########################################################################################3
##########################            CelebA            #################################3
########################################################################################3

class Classifier_CelebA(Classifier):
    def get_image_shape(self):
        return (64, 64, 3)

    def test_generate(self, x, n_sample, filename='images/results.png'):
        utils.save_images(x.reshape(n_sample, 64, 64, 3), filename)
        
    def get_image_dim(self):
        return 64 * 64 * 3

    def get_class_num(self):
        return 2

    def restore_session(self, sess):
        self.saver.restore(sess, './pretrained_models/Celeb_A/data/CelebA_classifier/model-999')

    def get_train_gen(self, sess, num_epochs = 10):
        train_gen, _, _ = utils.load_dataset(self.BATCH_SIZE, self.data_func, True)
        return utils.batch_gen(train_gen, True, self.y.shape[1], num_epochs)

    def get_test_gen(self, sess):
        _, _, test_gen = utils.load_dataset(self.BATCH_SIZE, self.data_func, True)
        return utils.batch_gen(test_gen, True, self.y.shape[1], 1)
        
    def set_model_dir(self):
        self.filepath = './model_Classifier/CelebA/'

    def __init__(self, inf_norm = 0.2, myVAE = None, batch_size = 100):
        self.data_func = utils.CelebA_load
        super(Classifier_CelebA, self).__init__(inf_norm, myVAE, batch_size)

    def build_classifier(self, im, inf_norm, reuse=False):
        with tf.variable_scope("C", reuse=reuse) as vs:
            x = tf.reshape(im, [-1, 64, 64, 3])
            #x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)
            
            net = InputLayer(x)
            n_filters = 3
            for i in range(2):
                net = Conv2dLayer(net, \
                        act=tf.nn.relu, \
                        shape=[5,5,n_filters,64], \
                        name="conv_" + str(i))
                net = MaxPool2d(net, \
                        filter_size=(3,3), \
                        strides=(2,2), \
                        name="mpool_" + str(i))
                net = LocalResponseNormLayer(net, \
                        depth_radius=4, \
                        bias=1.0, \
                        alpha=0.001 / 9.0, \
                        beta=0.75, \
                        name="lrn_" + str(i))
                n_filters = 64
            net = FlattenLayer(net)
            net = DenseLayer(net, n_units=384, act=tf.nn.relu, name="d1")
            net = DenseLayer(net, n_units=192, act=tf.nn.relu, name="d2")
            net = DenseLayer(net, n_units=2, act=tf.identity, name="final")
            cla_vars = tf.contrib.framework.get_variables(vs)
        return net.outputs, cla_vars        
    
    
########################################################################################3
##########################            MNIST            #################################3
########################################################################################3

class Classifier_MNIST(Classifier):
    def get_image_dim(self):
        return 784
    
    def get_class_num(self):
        return 10
    
    def set_model_dir(self):
        self.filepath = './model_Classifier/MNIST/'

    def test_generate(self, x, n_sample, filename='images/results.png'):
        utils.save_images(x.reshape(n_sample, 28, 28), filename)
        
    def __init__(self, inf_norm = 0.0, myVAE = None, batch_size = 100):
        self.data_func = utils.MNIST_load
        super(Classifier_MNIST, self).__init__(inf_norm, myVAE, batch_size)
    
    def build_classifier(self, im, inf_norm, reuse=False):
        with tf.variable_scope("Classifier", reuse=reuse) as vs:
            xmin = tf.clip_by_value(im - inf_norm, 0., 1.)
            xmax = tf.clip_by_value(im + inf_norm, 0., 1.)
            x = tf.random_uniform(tf.shape(im), xmin, xmax, dtype=tf.float32)

            input_layer = tf.reshape(x, [-1, 28, 28, 1])
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=self.dropout_rate)

            output = tf.layers.dense(inputs=dropout, units=10)
            cla_vars = tf.contrib.framework.get_variables(vs)
            return output, cla_vars 

class Classifier_MNIST_Robust(Classifier_MNIST):
    def set_model_dir(self):
        self.filepath = './model_Classifier/MNIST_Robust/'

########################################################################################3
########################            CIFAR10            #################################3
########################################################################################3
            
class Classifier_CIFAR10(Classifier):
    def test_generate(self, x, n_sample, filename='images/results.png'):
        utils.save_images(x.reshape(n_sample, 3, 32, 32), filename)
        
    def get_image_dim(self):
        return 3072
    
    def get_class_num(self):
        return 10
    
    #def restore_session(self, sess):
    #    self.saver.restore(sess, './pretrained_models/cifar10/data/cifar10_classifier/model.ckpt-1000000')
    
    def set_model_dir(self):
        self.filepath = './model_Classifier/CIFAR10/'
    
    def __init__(self, inf_norm = 0.0, myVAE = None, batch_size = 100):
        self.data_func = utils.cifar10_load
        super(Classifier_CIFAR10, self).__init__(inf_norm, myVAE, batch_size)
     
    def build_classifier(self, im, inf_norm, reuse=False):
        with tf.variable_scope('C', reuse=reuse) as vs:
            tensorlayer.layers.set_name_reuse(reuse)
            
            x = tf.reshape(im, [-1, 3, 32, 32])
            x = tf.transpose(x, [0, 2, 3, 1])
            
            xmin = tf.clip_by_value(x - inf_norm, 0., 1.)
            xmax = tf.clip_by_value(x + inf_norm, 0., 1.)
            x = tf.random_uniform(tf.shape(x), xmin, xmax, dtype=tf.float32)
            
            # Crop the central [height, width] of the image.
            # x = tf.image.resize_image_with_crop_or_pad(x, 24, 24)
            x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)
            
            net = InputLayer(x)
            net = Conv2dLayer(net, \
                    act=tf.nn.relu, \
                    shape=[5,5,3,64], \
                    name="conv1")
            net = MaxPool2d(net, \
                    filter_size=(3,3), \
                    strides=(2,2), \
                    name="pool1")
            net = LocalResponseNormLayer(net, \
                    depth_radius=4, \
                    bias=1.0, \
                    alpha = 0.001/9.0, \
                    beta = 0.75, \
                    name="norm1")
            net = Conv2dLayer(net, \
                    act=tf.nn.relu, \
                    shape=[5,5,64,64], \
                    name="conv2")
            net = LocalResponseNormLayer(net, \
                    depth_radius=4, \
                    bias=1.0, \
                    alpha=0.001/9.0, \
                    beta = 0.75, \
                    name="norm2")
            net = MaxPool2d(net, \
                    filter_size=(3,3), \
                    strides=(2,2), \
                    name="pool2")
            net = FlattenLayer(net, name="flatten_1")
            net = DenseLayer(net, n_units=384, name="local3", act=tf.nn.relu)

            net = DenseLayer(net, n_units=192, name="local4", act=tf.nn.relu)
            net = DenseLayer(net, n_units=10, name="softmax_linear", act=tf.identity)

            cla_vars = tf.contrib.framework.get_variables(vs)
            def name_fixer(var):
                return var.op.name.replace("W", "weights") \
                                    .replace("b", "biases") \
                                    .replace("weights_conv2d", "weights") \
                                    .replace("biases_conv2d", "biases")
            cla_vars = {name_fixer(var): var for var in cla_vars}
            return net.outputs, cla_vars

        
class Classifier_CIFAR10_Robust(Classifier_CIFAR10):
    def set_model_dir(self):
        self.filepath = './model_Classifier/CIFAR10_Robust/'

    def restore_session(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.filepath)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        
        
class Classifier_CIFAR10_MINMAX(Classifier_CIFAR10):
    def set_model_dir(self):
        self.filepath = './model_Classifier/CIFAR10_MINMAX/'
    
    def __init__(self, inf_norm = 0.08, myVAE = None, batch_size = 100):
        super(Classifier_CIFAR10_MINMAX, self).__init__(inf_norm, myVAE, batch_size)    
        self.grad = tf.gradients(self.loss_op, self.x)[0]

    def bad_example(self, sess, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        x = x_nat + np.random.uniform(-self.noise_level, self.noise_level, x_nat.shape)
        x = np.clip(x, 0., 1.)

        for i in range(20):
            grad = sess.run(self.grad, feed_dict={self.x: x, self.y: y})

            x += (self.noise_level) / 5. * np.sign(grad)
            x = np.clip(x, x_nat - self.noise_level, x_nat + self.noise_level)
            x = np.clip(x, 0., 1.) # ensure valid pixel range

        return x
    
    def train(self, sess):
        # Dataset iterator
        train_gen = self.get_train_gen(sess, 20)
        
        it = 0
        for x_train, y_train in train_gen:
            x_adv = self.bad_example(sess, x_train, y_train)
            
            sess.run(self.train_op, feed_dict={self.x: x_adv, self.y: y_train})
            it = it + 1
            if ( it % 100 == 0 ):
                ls = sess.run(self.loss_op, feed_dict={self.x: x_train, self.y: y_train})
                print 'loss :', ls

            if( it % 1000 == 0 ):
                print 'Saving model...'
                self.saver.save(sess, self.filepath+'checkpoint-'+str(it))
                self.saver.export_meta_graph(self.filepath+'checkpoint-'+str(it)+'.meta')

    def restore_session(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.filepath)
        self.saver.restore(sess, ckpt.model_checkpoint_path)