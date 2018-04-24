import tensorflow as tf
from models.generator_models import celebA_generator, mnist_generator
from models.classifier_models import celebA_classifier, mnist_classifier
import numpy as np

z = tf.placeholder(tf.float32,shape=[None,128])
z_2 = tf.placeholder(tf.float32,shape=[None,10])
im = tf.placeholder(tf.float32,shape=[None,64,64,3])

gen_func, gen_vars = celebA_generator(z,reuse=False)
cla_func, preds, cla_vars = celebA_classifier(im, reuse=False)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

gen_saver = tf.train.Saver(gen_vars)
cla_saver = tf.train.Saver(cla_vars)
gen_saver.restore(sess,'Celeb_A/data/CelebA_gen/model.ckpt-102951')
cla_saver.restore(sess,'Celeb_A/data/CelebA_classifier/model-999') #tf.train.latest_checkpoint('/home/jaylewis/Desktop/Dimakis_Models/Celeb_A/data/CelebA_classifier/')

logits = cla_func.eval({im: np.zeros((1,64,64,3))})
print(logits)
#logits = sess.run(cla_func,{im: np.zeros((64,64,3))})

random_im = gen_func.eval({z: np.random.randn(1,128)})
from skimage import io
io.imsave('im.png', random_im)
sess.close()
