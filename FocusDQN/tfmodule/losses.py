import tensorflow as tf
import numpy as np



def dice_loss(labels, logits, scope=None, weights=None, category_indep=True, ignore_BG=False):
    r'''
        Calculate the DICE loss for given labels and logits.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) != 3:
        raise TypeError('The labels must be 3-D tensor !!!')
    if not isinstance(logits, tf.Tensor) or len(logits.shape) != 4:
        raise TypeError('The logits must be 4-D tensor !!!')
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
    predictions = tf.nn.softmax(logits, axis=-1)    # [?, h, w, cls]
    if ignore_BG:
        labels = labels[:, :, :, 1:]    # [?, h, w, cls-1]
        predictions = predictions[:, :, :, 1:]  # [?, h, w, cls-1]
        if weights is not None:
            weights = weights[:, :, :, 1:]  # [?, h, w, cls-1]
    # Calculate
    intersection = 2 * tf.multiply(labels, predictions)     # [?, h, w, cls]
    union = tf.add(labels, predictions)     # [?, h, w, cls]
    if weights is not None:
        intersection = tf.multiply(intersection, weights)   # [?, h, w, cls]
        union = tf.multiply(union, weights)     # [?, h, w, cls]
    indep_axis = (1, 2) if category_indep else (1, 2, 3)
    intersection = tf.reduce_sum(intersection, axis=indep_axis)  # [?] or [?, cls]
    union = tf.reduce_sum(union, axis=indep_axis)  # [?] or [?, cls]
    # Add epsilon.
    intersection = tf.add(intersection, 1e-16)  # [?] or [?, cls]
    union = tf.add(union, 1e-16)  # [?] or [?, cls]
    dice = tf.divide(intersection, union, name='batch_dice')   # [?] or [?, cls]
    # Specify
    if category_indep:
        dice = tf.reduce_mean(dice, axis=-1)  # [?]
    dice = 1. - tf.reduce_mean(dice, name=scope)     # scalar
    # Finish.
    return dice
