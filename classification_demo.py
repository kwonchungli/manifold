import tensorflow as tf
import Classifier_Network

sess = tf.InteractiveSession()
c = Classifier_Network.Classifier_celeba()
# init = tf.global_variables_initializer()
# sess.run(init)
c.restore_session(sess)
c.eval_model(sess)
