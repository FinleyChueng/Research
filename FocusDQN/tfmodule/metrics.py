import tensorflow as tf
import numpy as np



def DICE(labels, logits, scope=None, weights=None, ignore_BG=True, keep_batch=False):
    r'''
        Calculate the DICE metric for given predictions. We reduce it
            for each sample of a batch.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) != 3:
        raise TypeError('The labels must be 3-D tensor !!!')
    if not isinstance(logits, tf.Tensor) or len(logits.shape) != 4:
        raise TypeError('The logits must be 3-D tensor !!!')
    if weights is not None and not isinstance(weights, tf.Tensor) or len(weights.shape) != 4:
        raise TypeError('The weights must be None or 4-D tensor !!!')
    lab_shape = np.asarray(labels.get_shape().as_list()[1:3])
    logit_shape = np.asarray(logits.get_shape().as_list()[1:3])
    weight_shape = np.asarray(weights.get_shape().as_list()[1:3])
    shape_invar = (lab_shape != logit_shape).any() and (logit_shape != weight_shape).any()
    if shape_invar:
        raise TypeError('The labels, predictions and weights must be of same shape !!!')
    # Prepare data.
    category = logits.get_shape().as_list()[-1]
    labels = tf.one_hot(labels, depth=category)     # [?, h, w, cls]
    predictions = tf.one_hot(tf.argmax(tf.nn.softmax(logits, axis=-1)), depth=category)   # [?, h, w, cls]
    if weights is not None:
        weights = tf.reduce_mean(weights, axis=(1, 2))  # [?, cls]
    if ignore_BG:
        labels = labels[:, :, :, 1:]    # [?, h, w, cls-1]
        predictions = predictions[:, :, :, 1:]  # [?, h, w, cls-1]
        if weights is not None:
            weights = weights[:, 1:]
    # Calculate
    intersection = 2 * tf.multiply(labels, predictions)     # [?, h, w, cls]
    union = tf.add(labels, predictions)     # [?, h, w, cls]
    if weights is not None:
        intersection = tf.multiply(intersection, weights)   # [?, h, w, cls]
        union = tf.multiply(union, weights)     # [?, h, w, cls]
    intersection = tf.reduce_sum(intersection, axis=(1, 2, 3))  # [?]
    union = tf.reduce_sum(union, axis=(1, 2, 3))    # [?]
    dice = tf.divide(intersection, union)   # [?]
    dice = tf.where(tf.not_equal(union, 0),
                    dice,
                    tf.ones_like(dice) * 1.,
                    name='batch_dice')   # [?]
    if not keep_batch:
        dice = tf.reduce_mean(dice, name=scope)     # scalar
    # Finish.
    return dice
