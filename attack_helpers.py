import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import PIL.Image

import utils
import attacks

from classifiers import *

def get_adv_dataset(sess, logits, x, y, x_test, y_test):
    return sess.run(attacks.fgm(x, logits, eps=0.2, ord=np.inf, targeted=False),
                    feed_dict={x: x_test, y: y_test})

def main():
    train_new_model = True
    checkpoint_dir = '.chkpts/'
    learning_rate = 0.001
    num_epochs = 1
    batch_size = 100
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    x_train = mnist.train.images
    y_train = np.asarray(mnist.train.labels, dtype=np.int32)
    x_test = mnist.test.images
    y_test = np.asarray(mnist.test.labels, dtype=np.int32)
    y_train = make_one_hot(y_train)
    y_test = make_one_hot(y_test)
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y = tf.placeholder(tf.int32, shape=[None, 10])
        dropout_rate = tf.placeholder_with_default(0.4, shape=())
        logits = cnn_model_fn(x, dropout_rate)
        train_op = get_train_op(logits, y, learning_rate)
        accuracy_op = get_accuracy_op(logits, y)
        saver = tf.train.Saver()
        if not train_new_model:
            print 'restoring model'
            restore_model(sess, saver, checkpoint_dir)
        init = tf.global_variables_initializer()
        sess.run(init)
        if train_new_model:
            print 'training model'
            train_model(sess, x, y, x_train, y_train, train_op, num_epochs, batch_size)
            save_model(sess, saver, checkpoint_dir)
        eval_mnist_model(sess, x, y, dropout_rate, logits, x_test, y_test, accuracy_op, image_path='./results.png', name='testing')
        adv_x_test = get_adv_dataset(sess, logits, x, y, x_test, y_test)
        eval_mnist_model(sess, x, y, dropout_rate, logits, adv_x_test, y_test, accuracy_op, image_path='./adv_results.png', name='adv_testing')

if __name__ == '__main__': main()
