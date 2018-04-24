import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import PIL.Image

from attack_helpers import get_adv_dataset
import utils
import attacks

def dnn_model_fn(x):
    dense1 = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.relu)
    dense_out = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.relu)
    return tf.layers.dense(inputs=dense_out, units=2)

def cnn_model_fn(x, dropout_rate, reuse=False):
    with tf.variable_scope("Classifier", reuse=reuse):
        xn = x + tf.random_normal(tf.shape(x), dropout_rate/2, 0.001 + dropout_rate, dtype=tf.float32)
        xn = tf.clip_by_value(xn, 0, 1)
        
        input_layer = tf.reshape(xn, [-1, 28, 28, 1])
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
        dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate)

        return tf.layers.dense(inputs=dropout, units=10)

def save_model(sess, saver, checkpoint_dir):
    saver.save(sess, checkpoint_dir + 'trained_model')
    saver.export_meta_graph(checkpoint_dir + 'trained_model_graph' + '.meta')

def eval_swiss_model(sess, x, y, logits, x_test, y_test, accuracy_op, image_path=None, name=None):
    print(name, "Accuracy:", sess.run(accuracy_op, feed_dict={x: x_test, y: y_test}))
    if image_path:
        test_pred = sess.run(tf.argmax(logits, 1), feed_dict={x: x_test, y: y_test})
        test_pred_c = np.array(['b' if pred == 0 else 'r' for pred in test_pred])
        plt.scatter([x_test[:, 0]], [x_test[:, 1]], c=test_pred_c)
        plt.savefig(image_path)
        plt.figure()

        
def eval_mnist_model(sess, x, y, dropout_rate, logits, adv_logits, test_epoch, accuracy_op, adv_acc_op, myVAE, image_path=None, name=None):
    test_gen = utils.batch_gen(test_epoch, True, y.shape[1], 1)
    it = 0
    
    loss_1 = tf.losses.softmax_cross_entropy(y, logits=logits)
    loss_adv = tf.losses.softmax_cross_entropy(y, logits=adv_logits)
    loss_z_l2 = tf.reduce_mean(tf.nn.l2_loss(myVAE.z - myVAE.z_in))
    
    #attack = attacks.LinfPGDAttack(loss_z_l2, myVAE.x_hat, 0.3, 30, 0.01, True)
    #attack = attacks.LinfPGDAttack(loss_1, x, 0.3, 30, 0.01, True)
    #attack = attacks.LinfPGDAttack(loss_adv, myVAE.x_hat, 0.3, 30, 0.01, True)
    
    normal_avr, adv_avr_ref, adv_avr, reconstr_avr = 0, 0, 0, 0
    for x_test, y_test in test_gen:
        it = it + 1
        #adversarial_x = get_adv_dataset(sess, logits, x, y, x_test, y_test)
        #adversarial_x = get_adv_dataset(sess, adv_logits, myVAE.x_hat, y, x_test, y_test)
        
        #_, z_in = myVAE.autoencode_dataset(sess, x_test)
        #adversarial_x = attack.perturb(x_test, y_test, myVAE.x_hat, y, sess, myVAE.z_in, z_in)
        #adversarial_x = attack.perturb(x_test, y_test, x, y, sess)
        #adversarial_x = attack.perturb(x_test, y_test, myVAE.x_hat, y, sess)
        cleaned_x, z_res = myVAE.autoencode_dataset(sess, adversarial_x)
        
        #print ('compare z vs z : ', (z_in[0] - z_res[0]), np.linalg.norm(z_in[0] - z_res[0]))
        
        normal_avr += sess.run(accuracy_op, feed_dict={x: x_test, y: y_test, dropout_rate: 0.0})
        adv_avr_ref += sess.run(accuracy_op, feed_dict={x: adversarial_x, y: y_test, dropout_rate: 0.0})
        
        adv_avr += sess.run(adv_acc_op, feed_dict={myVAE.x_hat: adversarial_x, y: y_test, dropout_rate: 0.0})
        reconstr_avr += sess.run(accuracy_op, feed_dict={x: cleaned_x, y: y_test, dropout_rate: 0.0})
        
        if( it % 10 == 3 ):
            test_pred = sess.run(logits, feed_dict={x: cleaned_x, y: y_test, dropout_rate: 0.0})
            
            i1 = np.argmax(test_pred, 1)
            i2 = np.argmax(y_test, 1)
            index = np.where(np.not_equal(i1, i2))[0]
            
            p_size = len(index)
            p_size = x_test.shape[0]
            wrong_x = x_test[:, :]
            wrong_adv = adversarial_x[:, :]
            wrong_res = cleaned_x[:, :]
            
            utils.save_images(wrong_x.reshape(p_size, 28, 28), 'images/cl_original.png')
            utils.save_images(wrong_adv.reshape(p_size, 28, 28), 'images/cl_adversarial.png')
            utils.save_images(wrong_res.reshape(p_size, 28, 28), 'images/cl_reconstr.png')
        
    print ("------------ Test ----------------")
    print(name, "Normal Accuracy:", normal_avr / it)
    print(name, "Normal Adversarial Accuracy:", adv_avr_ref / it)
    print(name, "Adversarial Accuracy:", adv_avr / it)
    print(name, "Reconstructed Accuracy:", reconstr_avr / it)

def make_one_hot(coll):
    onehot = np.zeros((coll.shape[0], coll.max() + 1))
    onehot[np.arange(coll.shape[0]), coll] = 1
    return onehot

def restore_model(sess, saver, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt: saver.restore(sess, ckpt.model_checkpoint_path)

def get_accuracy_op(logits, y):
    correct_pred = tf.equal(tf.argmax(logits, 1),
                            tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_train_op(logits, y, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    loss_op = tf.losses.softmax_cross_entropy(y, logits=logits)
    return optimizer.minimize(loss_op)

def train_model(sess, x, y, train_epoch, train_op, num_epochs, batch_size):
    train_gen = utils.batch_gen(train_epoch, True, y.shape[1], num_epochs)
    
    for x_train, y_train in train_gen:
        sess.run(train_op, feed_dict={x: x_train, y: y_train})

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

def get_batches(colls, batch_size):
    return apply(zip, [np.array_split(coll, batch_size) for coll in colls])
    
if __name__ == '__main__': main()
