import tensorflow as tf
import numpy as np



def dice_loss(labels, logits, weights=None, scope=None, cal_mask=None, mean_axis=None):
    r'''
        Calculate the DICE loss for given labels and logits.

    Parameters:
        labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels
            and result). Each entry in labels must be an index in [0, num_classes).
        logits: Unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1}, num_classes].
        weights: Coefficients for the loss. This must be None or the same as labels.
        scope: The scope for the operations performed in computing the loss.
        cal_mask: The mask used to filter the region that used to calculate metric.
        mean_axis: The dimensions don't contributes to the calculation of metric.
            Default is "None" indicating preserve all the dimensions.

    Returns:
        Weighted loss tensor.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) < 1:
        raise TypeError('The labels must be at least 1-D tensor !!!')
    if not isinstance(logits, tf.Tensor) or len(logits.shape) < 2:
        raise TypeError('The logits must be at least 2-D tensor !!!')
    if weights is not None and not isinstance(weights, tf.Tensor):
        raise TypeError('The weights must be None or tensor !!!')
    if (cal_mask is not None and not isinstance(cal_mask, tf.Tensor)) or len(cal_mask.shape) < 2:
        raise TypeError('The cal_mask must be at least 2-D tensor !!!')
    if len(labels.shape) > 1:
        lab_shape = np.asarray(labels.get_shape().as_list())
        logit_shape = np.asarray(logits.get_shape().as_list())
        shape_invar = (lab_shape[1:] != logit_shape[1:-1]).any()
        if cal_mask is not None:
            cal_shape = np.asarray(cal_mask.get_shape().as_list())
            shape_invar = shape_invar or (logit_shape[1:-1] != cal_shape[1:-1]).any()
        if shape_invar:
            raise TypeError('The labels, logits and weights must be of same shape !!!')
    if mean_axis is not None and not isinstance(mean_axis, (list, tuple)):
        raise TypeError('The mean_axis must be a tuple or list !!!')
    if isinstance(mean_axis, (list, tuple)) and \
            not (min(mean_axis) >= -len(logits.shape) and max(mean_axis) < len(logits.shape)):
        raise ValueError('The mean_axis must match the input tensor dimension value !!!')
    if weights is not None and mean_axis is not None and (len(weights.shape) != len(mean_axis)):
        raise Exception('The dimension of weights and mean_axis should be equal !!!')
    # Prepare data.
    category = logits.get_shape().as_list()[-1]
    labels = tf.one_hot(labels, depth=category)     # [?, d1, ..., dn, cls]
    predictions = tf.nn.softmax(logits, axis=-1)    # [?, d1, ..., dn, cls]
    # Calculate the numerator and denominator.
    intersection = 2 * tf.multiply(labels, predictions)     # [?, d1, ..., dn, cls]
    union = tf.add(labels, predictions)     # [?, d1, ..., dn, cls]
    # Filter the region used to calculate.
    if cal_mask is not None:
        intersection = tf.multiply(intersection, cal_mask)  # [?, d1, ..., dn, cls]
        union = tf.multiply(union, cal_mask)  # [?, d1, ..., dn, cls]
        num_present = tf.multiply(cal_mask, labels)     # [?, d1, ..., dn, cls]
    else:
        num_present = labels    # [?, d1, ..., dn, cls]
    # Determine the axis to calculate.
    if mean_axis is not None:
        rec_axis = list(range(len(logits.shape)))
        for ax in mean_axis:
            if ax < 0:
                axis = len(logits.shape) + ax
            else:
                axis = ax
            rec_axis.remove(axis)
        rec_axis = tuple(rec_axis)
    else:
        rec_axis = None
    # Reduce sum the specific axis.
    if rec_axis is not None:
        intersection = tf.reduce_sum(intersection, axis=rec_axis)   # [?, d1, ..., dk]
        union = tf.reduce_sum(union, axis=rec_axis)     # [?, d1, ..., dk]
        num_present = tf.to_float(tf.greater(tf.reduce_sum(num_present, axis=rec_axis), 0.))    # [?, d1, ..., dk]
    # Real calculate the metric here.
    dice = tf.divide(intersection, tf.maximum(union, 1e-32), name='dice_loss')  # [?, d1, ..., dk]
    # Multiply with weights here.
    if weights is not None:
        dice = tf.multiply(dice, weights)   # [?, d1, ..., dk]
        num_present = tf.to_float(tf.greater(tf.multiply(num_present, weights), 0.))     # [?, d1, ..., dk]
    # Mean the all value to get one scalar.
    dice = tf.reduce_sum(dice)  # scalar
    num_present = tf.reduce_sum(num_present)    # scalar
    num_present = tf.maximum(num_present, 1.)   # avoid zero
    dice = tf.divide(dice, num_present)     # scalar
    # Use the reverse form.
    loss = tf.subtract(1., dice, name=scope)    # scalar
    # Finish.
    return loss



def category_CE_loss(labels, logits, weights=None, scope=None):
    r'''
        Calculate the category cross-entropy loss for given labels and logits.

    Parameters:
        labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels
            and result). Each entry in labels must be an index in [0, num_classes).
        logits: Unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1}, num_classes].
        weights: Coefficients for the loss. This must be None or the same as logits.
        scope: The scope for the operations performed in computing the loss.

    Returns:
        Weighted loss tensor.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) < 1:
        raise TypeError('The labels must be at least 1-D tensor !!!')
    if not isinstance(logits, tf.Tensor) or len(logits.shape) < 2:
        raise TypeError('The logits must be at least 2-D tensor !!!')
    if weights is not None and not isinstance(weights, tf.Tensor):
        raise TypeError('The weights must be None or tensor !!!')
    if len(labels.shape) > 1:
        lab_shape = np.asarray(labels.get_shape().as_list())
        logit_shape = np.asarray(logits.get_shape().as_list())
        shape_invar = (lab_shape[1:] != logit_shape[1:-1]).any()
        if weights is not None:
            weight_shape = np.asarray(weights.get_shape().as_list())
            shape_invar = shape_invar or (logit_shape[1:] != weight_shape[1:]).any()
        if shape_invar:
            raise TypeError('The labels, predictions and weights must be of same shape !!!')
    # Prepare data.
    category = logits.get_shape().as_list()[-1]
    labels = tf.one_hot(labels, depth=category)     # [?, d1, ..., dn, cls]
    predictions = tf.nn.softmax(logits, axis=-1)    # [?, d1, ..., dn, cls]
    # Calculate the cross-entropy loss.
    loss = tf.negative(tf.multiply(labels, tf.log(tf.maximum(predictions, 1e-32))),
                       name='category_CEloss')  # [?, d1, ..., dn, cls]
    # Multiply with weights here.
    if weights is not None:
        loss = tf.multiply(loss, weights)   # [?, d1, ..., dn, cls]
        num_present = tf.to_float(tf.not_equal(tf.multiply(weights, labels), 0.))    # [?, d1, ..., dn, cls]
    else:
        num_present = tf.to_float(tf.not_equal(labels, 0.))     # [?, d1, ..., dn, cls]
    # Mean the axises except the "class" dimension.
    rec_axis = list(range(len(logits.shape)))
    rec_axis.remove(len(logits.shape) - 1)
    loss = tf.reduce_sum(loss, axis=rec_axis)   # [cls]
    num_present = tf.reduce_sum(num_present, axis=rec_axis)     # [cls]
    num_present = tf.maximum(num_present, 1.)  # avoid zero
    loss = tf.divide(loss, num_present)     # [cls]
    # Mean the value of each class.
    loss = tf.reduce_mean(loss, name=scope)     # scalar
    # Finish.
    return loss
