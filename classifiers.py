import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import attacks

def dnn_model_fn(x):
    dense1 = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.relu)
    dense_out = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.relu)
    return tf.layers.dense(inputs=dense_out, units=2)

def save_model(sess, saver, checkpoint_dir):
    saver.save(sess, checkpoint_dir + 'trained_model')
    saver.export_meta_graph(checkpoint_dir + 'trained_model_graph' + '.meta')

def eval_model(sess, x, y, logits, x_test, y_test, accuracy_op, image_path=None, name=None):
    print(name, "Accuracy:", sess.run(accuracy_op, feed_dict={x: x_test, y: y_test}))
    if image_path:
        test_pred = sess.run(tf.argmax(logits, 1), feed_dict={x: x_test, y: y_test})
        test_pred_c = np.array(['b' if pred == 0 else 'r' for pred in test_pred])
        plt.scatter([x_test[:, 0]], [x_test[:, 1]], c=test_pred_c)
        plt.savefig(image_path)
        plt.figure()

def get_batches(colls, batch_size):
    return apply(zip, [np.array_split(coll, batch_size) for coll in colls])

def get_adv_dataset(sess, logits, x, y, x_test, y_test):
    return sess.run(attacks.fgm(x, logits, eps=2.0, ord=np.inf, targeted=True),
                    feed_dict={x: x_test, y: y_test})

def make_one_hot(coll):
    return np.array([[0, 1] if val else [1, 0] for val in coll])

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

def train_model(sess, x, y, x_train, y_train, train_op, num_epochs, batch_size):
    for epoch_num in range(num_epochs):
        for batch_x, batch_y in get_batches([x_train, y_train], batch_size):
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

def main():
    train_new_model = True
    checkpoint_dir = '.chkpts/'
    learning_rate = 0.1
    num_epochs = 1
    batch_size = 512
    x_train, y_train, x_test, y_test = utils.swiss_load()
    y_train = make_one_hot(y_train)
    y_test = make_one_hot(y_test)
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, 2])
        y = tf.placeholder(tf.int32, shape=[None, 2])
        logits = dnn_model_fn(x)
        train_op = get_train_op(logits, y, learning_rate)
        accuracy_op = get_accuracy_op(logits, y)
        saver = tf.train.Saver()
        if not train_new_model: restore_model(sess, saver, checkpoint_dir)
        init = tf.global_variables_initializer()
        sess.run(init)
        if train_new_model:
            train_model(sess, x, y, x_train, y_train, train_op, num_epochs, batch_size)
            save_model(sess, saver, checkpoint_dir)
        eval_model(sess, x, y, logits, x_test, y_test, accuracy_op, image_path='./results.png', name='testing')
        adv_x_test = get_adv_dataset(sess, logits, x, y, x_test, y_test)
        eval_model(sess, x, y, logits, adv_x_test, y_test, accuracy_op, image_path='./adv_results.png', name='adv_testing')

if __name__ == '__main__': main()
