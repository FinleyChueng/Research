import tensorflow as tf
import numpy as np



def DICE(labels, predictions, weights=None, scope=None, keep_axis=None):
    r'''
        Calculate the "Dice" metric for given predictions. We can reduce
            arbitrary axis, default to all dimensions.

    Parameters:
        labels: Tensor of shape [d_0, d_1, ..., d_{r-1}, num_classes] (where r
            is rank of labels and result).
        predictions: One-hot predictions. The same shape as labels.
        weights: Coefficients for the loss. This must be None or the same as labels.
        scope: the scope for the operations performed in computing metric.
        keep_axis: The dimensions to keep. Default is "None" indicating reduce
            all the dimensions.

    Returns:
        Dice metric tensor.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) < 2:
        raise TypeError('The labels must be at least 2-D tensor !!!')
    if not isinstance(predictions, tf.Tensor) or len(predictions.shape) < 2:
        raise TypeError('The predictions must be at least 2-D tensor !!!')
    if weights is None or (weights is not None and not isinstance(weights, tf.Tensor)) or len(weights.shape) < 2:
        raise TypeError('The weights must be None or 2-D tensor !!!')
    lab_shape = np.asarray(labels.get_shape().as_list())
    pred_shape = np.asarray(predictions.get_shape().as_list())
    weight_shape = np.asarray(weights.get_shape().as_list())
    shape_invar = (lab_shape[1:] != pred_shape[1:]).any() or (pred_shape[1:] != weight_shape[1:]).any()
    if shape_invar:
        raise TypeError('The labels, predictions and weights must be of same shape !!!')
    if keep_axis is not None and not isinstance(keep_axis, (list, tuple)):
        raise TypeError('The keep_axis must be a tuple or list !!!')
    if isinstance(keep_axis, (list, tuple)) and \
            not (min(keep_axis) >= -len(weights.shape) and max(keep_axis) < len(weights.shape)):
        raise ValueError('The keep axis must match the input tensor dimension value !!!')
    # Calculate.
    intersection = 2 * tf.multiply(labels, predictions)     # [?, d1, ..., dn, cls]
    union = tf.add(labels, predictions)     # [?, d1, ..., dn, cls]
    dice = tf.divide(intersection + 1e-32, union + 1e-32, name='dice')  # [?, d1, ..., dn, cls]
    # Determine the axis to reduce.
    if keep_axis is not None:
        rec_axis = list(range(len(weights.shape)))
        for ax in keep_axis:
            if ax < 0:
                axis = len(weights.shape) + ax
            else:
                axis = ax
            rec_axis.remove(axis)
        rec_axis = tuple(rec_axis)
    else:
        rec_axis = None
    # Reduce mean the specific axis.
    if weights is not None:
        weights = tf.multiply(weights, labels)  # [?, d1, ..., dn, cls]
        dice = tf.multiply(dice, weights)   # [?, d1, ..., dn, cls]
        num_present = tf.reduce_sum(tf.to_float(tf.not_equal(weights, 0.)), axis=rec_axis)  # scalar
    else:
        num_present = tf.reduce_sum(labels, axis=rec_axis)  # scalar
    num_present = tf.maximum(num_present, 1.)   # avoid zero
    dice = tf.divide(tf.reduce_sum(dice, axis=rec_axis), num_present)   # unknown
    dice = tf.stop_gradient(dice, name=scope)   # unknown
    # Finish.
    return dice



def recall(labels, predictions, weights=None, scope=None, keep_axis=None):
    r'''
        Calculate the "Recall" metric for given predictions. We can reduce
            arbitrary axis, default to all dimensions.

    Parameters:
        labels: Tensor of shape [d_0, d_1, ..., d_{r-1}, num_classes] (where r
            is rank of labels and result).
        predictions: One-hot predictions. The same shape as labels.
        weights: Coefficients for the loss. This must be None or the same as labels.
        scope: the scope for the operations performed in computing metric.
        keep_axis: The dimensions to keep. Default is "None" indicating reduce
            all the dimensions.

    Returns:
        Recall metric tensor.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) < 2:
        raise TypeError('The labels must be at least 2-D tensor !!!')
    if not isinstance(predictions, tf.Tensor) or len(predictions.shape) < 2:
        raise TypeError('The predictions must be at least 2-D tensor !!!')
    if weights is None or (weights is not None and not isinstance(weights, tf.Tensor)) or len(weights.shape) < 2:
        raise TypeError('The weights must be None or 2-D tensor !!!')
    lab_shape = np.asarray(labels.get_shape().as_list())
    pred_shape = np.asarray(predictions.get_shape().as_list())
    weight_shape = np.asarray(weights.get_shape().as_list())
    shape_invar = (lab_shape[1:] != pred_shape[1:]).any() or (pred_shape[1:] != weight_shape[1:]).any()
    if shape_invar:
        raise TypeError('The labels, predictions and weights must be of same shape !!!')
    if keep_axis is not None and not isinstance(keep_axis, (list, tuple)):
        raise TypeError('The keep_axis must be a tuple or list !!!')
    if isinstance(keep_axis, (list, tuple)) and \
            not (min(keep_axis) >= -len(weights.shape) and max(keep_axis) < len(weights.shape)):
        raise ValueError('The keep axis must match the input tensor dimension value !!!')
    # Calculate.
    intersection = tf.multiply(labels, predictions)     # [?, d1, ..., dn, cls]
    ground_truth = labels   # [?, d1, ..., dn, cls]
    recall = tf.divide(intersection + 1e-32, ground_truth + 1e-32, name='recall')   # [?, d1, ..., dn, cls]
    # Determine the axis to reduce.
    if keep_axis is not None:
        rec_axis = list(range(len(weights.shape)))
        for ax in keep_axis:
            if ax < 0:
                axis = len(weights.shape) + ax
            else:
                axis = ax
            rec_axis.remove(axis)
        rec_axis = tuple(rec_axis)
    else:
        rec_axis = None
    # Reduce mean the specific axis.
    if weights is not None:
        weights = tf.multiply(weights, labels)  # [?, d1, ..., dn, cls]
        recall = tf.multiply(recall, weights)   # [?, d1, ..., dn, cls]
        num_present = tf.reduce_sum(tf.to_float(tf.not_equal(weights, 0.)), axis=rec_axis)  # scalar
    else:
        num_present = tf.reduce_sum(labels, axis=rec_axis)  # scalar
    num_present = tf.maximum(num_present, 1.)   # avoid zero
    recall = tf.divide(tf.reduce_sum(recall, axis=rec_axis), num_present)   # unknown
    recall = tf.stop_gradient(recall, name=scope)   # unknown
    # Finish.
    return recall
