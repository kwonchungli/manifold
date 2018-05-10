import tensorflow as tf
import Classifier_Network

sess = tf.InteractiveSession()

#c = Classifier_Network.Classifier_F_MNIST()
#c = Classifier_Network.Classifier_F_MNIST_Robust(0.3)
c = Classifier_Network.Classifier_F_MNIST_MINMAX(0.2)

init = tf.global_variables_initializer()
sess.run(init)

c.restore_session(sess)
#c.train(sess, 10)
c.eval_model(sess, 0.2)
