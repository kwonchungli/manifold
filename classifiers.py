import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import attacks

def dnn_model_fn(features, labels, mode):
    dense1 = tf.layers.dense(inputs=features['x'], units=10, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.relu)
    dense_out = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense_out, units=2)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.softmax_cross_entropy(labels, logits=logits)
    if mode == 'logits':
        return logits
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels[:, 1], predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main():
    sess = tf.Session()
    x = tf.placeholder(tf.float32, shape=[None, 2])
    y = tf.placeholder(tf.int32, shape=[None, 2])
    x_train, y_train, x_test, y_test = utils.swiss_load()
    y_train = np.array([[0, 1] if val else [1, 0] for val in y_train])
    y_test = np.array([[0, 1] if val else [1, 0] for val in y_test])
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(x_train)}, y=np.array(y_train), shuffle=True)
    classifier = tf.estimator.Estimator(model_fn=dnn_model_fn)
    classifier.train(input_fn=train_input_fn, steps=100)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(x_test)}, y=np.array(y_test), num_epochs=1, shuffle=False)
    results = classifier.evaluate(input_fn=test_input_fn)
    prediction_results = list(classifier.predict(input_fn=test_input_fn))
    preds = np.array(['b' if pred['classes'] == 0 else 'r' for pred in prediction_results])
    plt.scatter([x_test[:, 0]], [x_test[:, 1]], c=preds)
    plt.savefig('./results.png')
    accuracy_score = results["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    probas = dnn_model_fn({'x': x}, y, 'logits')
    adv_x_test = attacks.fgm(x, probas, eps=0.3, ord=np.inf, clip_min=0.0, clip_max=1.0)
    adv_y_test = y_test
    adv_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": adv_x_test}, y=np.array(adv_y_test), num_epochs=1, shuffle=False)
    adv_results = classifier.evaluate(input_fn=adv_input_fn)
    adv_prediction_results = list(classifier.predict(input_fn=adv_input_fn))
    adv_preds = np.array(['b' if pred['classes'] == 0 else 'r' for pred in adv_prediction_results])
    plt.scatter([adv_x_test[:, 0]], [adv_x_test[:, 1]], c=adv_preds)
    plt.savefig('./adv_results.png')
    adv_accuracy_score = adv_results["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


if __name__ == '__main__': main()
