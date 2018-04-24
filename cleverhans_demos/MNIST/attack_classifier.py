"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with Keras.
It is very similar to mnist_tutorial_tf.py, which does the same
thing but without a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper

from keras.backend import manual_variable_initialization    #solves problems with save_model / load_model (Keras)
manual_variable_initialization(True)

FLAGS = flags.FLAGS


def mnist_tutorial(train_start=0, train_end=600, test_start=0,
                   test_end=200, batch_size=128,model_name="",
                   model_dir="",print_examples=False):
    #-----------------------------------------------------------------------------
    # I) Setup (load data / model /etc.)
    #------------------------------------------------------------------------------
    """
    MNIST CleverHans tutorial
    :param model_name: filename for desired classifier model
    :param model_dir: location of classifier models
    :return: an AccuracyReport object
    """
    keras.layers.core.K.set_learning_phase(0)

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Load Model
    filepath = model_dir+model_name
    model = keras.models.load_model(filepath)
    print("Model loaded from:")
    print(filepath)
    preds = model(x)    #predictions
    print("Defined TensorFlow model graph.")

    # Evaluate the accuracy of the MNIST model on legitimate examples (for comparison)
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test,
                          args=eval_params)
    report.clean_train_clean_eval = accuracy
    print('Test accuracy on legitimate examples: %0.4f' % accuracy)


    # Print Normal Examples
    if(print_examples):
        fig2 = plt.figure()
        plt.title("Normal")
        for i in range(0,10):
            example = X_test[i:i+1,:,:,:]
            img = example[0,:,:,0]
            plt.subplot(2, 5, i + 1)
            plt.imshow(img,cmap="gray")

    #-----------------------------------------------------------------------------
    # II) Adversarial operations / evaluation
    #------------------------------------------------------------------------------

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)

    # Consider the attack to be constant
    adv_x = tf.stop_gradient(adv_x)
    preds_adv = model(adv_x)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc

    # Print Adversarial Examples
    if(print_examples):
        fig1 = plt.figure()
        for i in range(0,10):
            adversarial_example = sess.run(adv_x,feed_dict={x:X_test[i:i+1,:,:,:]})
            img = adversarial_example[0,:,:,0]
            plt.subplot(2,5,i+1)
            plt.imshow(img,cmap="gray")
        plt.show()

    return report


def main(argv=None):
    mnist_tutorial(batch_size=FLAGS.batch_size,
                   model_name=FLAGS.model_name,
                   model_dir=FLAGS.model_dir,
                   print_examples=FLAGS.print_examples)


if __name__ == '__main__':

    cwd = os.getcwd()

    flags.DEFINE_string('model_dir',cwd+"\\models\\",'Directory where to save model')
    flags.DEFINE_string('model_name',"mnist.h5",'Name of model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_bool('print_examples',False,'flag that controls if image examples are printed')
    tf.app.run()
