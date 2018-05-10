import numpy as np
import tensorflow as tf

class LinfPGDAttack:
    def __init__(self, obj, x, epsilon = 0.3, pgd_iter = 40, a = 0.01, random_start = False):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
        self.epsilon = epsilon
        self.pgd_iter = pgd_iter
        self.a = a
        self.rand = random_start

        self.grad = tf.gradients(obj, x)[0]

    def perturb(self, x_nat, y, x_in, y_in, sess, plc_zin = None, z_in = None):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        feed_dict = {x_in:x_nat, y_in:y}
        if( plc_zin != None ): feed_dict[plc_zin] = z_in
        
        for i in range(self.pgd_iter):
            feed_dict = {x_in:x, y_in:y}
            if( plc_zin != None ): feed_dict[plc_zin] = z_in

            grad = sess.run(self.grad, feed_dict=feed_dict)

            x += self.a * np.sign(grad)
            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
            x = np.clip(x, 0, 1) # ensure valid pixel range

        return x
    
def model_loss(labels, model):
    op = model.op
    if op.type == "Softmax":
        logits, = op.inputs
    else:
        logits = model
    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return out

def fgm(x, preds, y=None, eps=0.3, ord=np.inf, clip_min=None, clip_max=None, targeted=False):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = model_loss(y, preds)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(xrange(1, len(x.get_shape())))
        normalized_grad = grad / tf.reduce_sum(tf.abs(grad),
                                               reduction_indices=red_ind,
                                               keep_dims=True)
    elif ord == 2:
        red_ind = list(xrange(1, len(x.get_shape())))
        square = tf.reduce_sum(tf.square(grad),
                               reduction_indices=red_ind,
                               keep_dims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are currently implemented.")

    # Multiply by constant epsilon
    scaled_grad = eps * normalized_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    
    return adv_x

def get_adv_dataset(sess, logits, x, y, x_test, y_test, eps=0.2, ord=np.inf):
    return sess.run(fgm(x, logits, eps=eps, ord=ord, clip_min=0, clip_max=1, targeted=False),
                    feed_dict={x: x_test, y: y_test})
