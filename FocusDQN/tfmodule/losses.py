import tensorflow as tf
import numpy as np



def dice_loss(labels, logits, scope=None, weights=None):
    r'''
        Calculate the DICE loss for given labels and logits.

    Parameters:
        labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels
            and result). Each entry in labels must be an index in [0, num_classes).
        logits: Unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1}, num_classes].
        weights: Coefficients for the loss. This must be None or the same as labels.
        scope: the scope for the operations performed in computing the loss.

    Returns:
        Weighted loss tensor.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) < 1:
        raise TypeError('The labels must be at least 1-D tensor !!!')
    if not isinstance(logits, tf.Tensor) or len(logits.shape) < 2:
        raise TypeError('The logits must be at least 2-D tensor !!!')
    if weights is None or (weights is not None and not isinstance(weights, tf.Tensor)) or len(weights.shape) < 1:
        raise TypeError('The weights must be None or at least 1-D tensor !!!')
    if len(labels.shape) > 1:
        lab_shape = np.asarray(labels.get_shape().as_list())
        logit_shape = np.asarray(logits.get_shape().as_list())
        weight_shape = np.asarray(weights.get_shape().as_list())
        shape_invar = (lab_shape[1:] != logit_shape[1:-1]).any() or (lab_shape[1:] != weight_shape[1:]).any()
        if shape_invar:
            raise TypeError('The labels, predictions and weights must be of same shape !!!')
    # Prepare data.
    category = logits.get_shape().as_list()[-1]
    labels = tf.one_hot(labels, depth=category)     # [?, d1, ..., dn, cls]
    predictions = tf.nn.softmax(logits, axis=-1)    # [?, d1, ..., dn, cls]
    # Calculate.
    intersection = 2 * tf.multiply(labels, predictions)     # [?, d1, ..., dn, cls]
    union = tf.add(labels, predictions)     # [?, d1, ..., dn, cls]
    dice = tf.divide(intersection + 1e-32, union + 1e-32)   # [?, d1, ..., dn, cls]
    dice = tf.reduce_sum(dice, axis=-1, name='dice_loss')   # [?, d1, ..., dn]
    if weights is not None:
        dice = tf.multiply(dice, weights)   # [?, d1, ..., dn]
        num_present = tf.reduce_sum(tf.to_float(tf.not_equal(weights, 0.)))     # scalar
    else:
        num_present = tf.reduce_sum(tf.ones_like(dice))     # scalar
    num_present = tf.maximum(num_present, 1.)   # avoid zero
    dice = tf.divide(tf.reduce_sum(dice), num_present)  # scalar
    # Use the reverse form.
    loss = tf.subtract(1., dice, name=scope)    # scalar
    # Finish.
    return loss
