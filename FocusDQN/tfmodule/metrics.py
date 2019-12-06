import tensorflow as tf
import numpy as np



def DICE(labels, predictions, weights=None, scope=None, keep_batch=True):
    r'''
        Calculate the DICE metric for given predictions. We reduce it
            for each sample of a batch.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) != 4:
        raise TypeError('The labels must be 4-D tensor !!!')
    if not isinstance(predictions, tf.Tensor) or len(predictions.shape) != 4:
        raise TypeError('The predictions must be 4-D tensor !!!')
    if weights is not None and not isinstance(weights, tf.Tensor) or len(weights.shape) != 4:
        raise TypeError('The weights must be None or 4-D tensor !!!')
    lab_shape = np.asarray(labels.get_shape().as_list()[1:3])
    pred_shape = np.asarray(predictions.get_shape().as_list()[1:3])
    weight_shape = np.asarray(weights.get_shape().as_list()[1:3])
    shape_invar = (lab_shape != pred_shape).any() and (pred_shape != weight_shape).any()
    if shape_invar:
        raise TypeError('The labels, predictions and weights must be of same shape !!!')
    # Calculate
    intersection = 2 * tf.multiply(labels, predictions)     # [?, h, w, cls]
    union = tf.add(labels, predictions)     # [?, h, w, cls]
    if weights is not None:
        intersection = tf.multiply(intersection, weights)   # [?, h, w, cls]
        union = tf.multiply(union, weights)     # [?, h, w, cls]
        num_present = tf.reduce_sum(weights)    # scalar
    else:
        num_present = tf.reduce_sum(tf.ones_like(predictions))   # scalar
    dice = tf.divide(intersection + 1e-32, union + 1e-32, name='dice')     # [?, h, w, cls]
    # Whether to keep batch
    rec_axis = (1, 2, 3) if keep_batch else None
    dice = tf.divide(tf.reduce_sum(dice, axis=rec_axis), num_present)   # scalar or [?,]
    dice = tf.stop_gradient(dice, name=scope)   # scalar or [?]
    # Finish.
    return dice



def recall(labels, predictions, weights=None, scope=None, keep_batch=True):
    r'''
        Calculate the DICE metric for given predictions. We reduce it
            for each sample of a batch.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) != 4:
        raise TypeError('The labels must be 4-D tensor !!!')
    if not isinstance(predictions, tf.Tensor) or len(predictions.shape) != 4:
        raise TypeError('The predictions must be 4-D tensor !!!')
    if weights is not None and not isinstance(weights, tf.Tensor) or len(weights.shape) != 4:
        raise TypeError('The weights must be None or 4-D tensor !!!')
    lab_shape = np.asarray(labels.get_shape().as_list()[1:3])
    pred_shape = np.asarray(predictions.get_shape().as_list()[1:3])
    weight_shape = np.asarray(weights.get_shape().as_list()[1:3])
    shape_invar = (lab_shape != pred_shape).any() and (pred_shape != weight_shape).any()
    if shape_invar:
        raise TypeError('The labels, predictions and weights must be of same shape !!!')
    # Calculate
    intersection = tf.multiply(labels, predictions)  # [?, h, w, cls]
    ground_truth = labels
    if weights is not None:
        intersection = tf.multiply(intersection, weights)  # [?, h, w, cls]
        ground_truth = tf.multiply(ground_truth, weights)     # [?, h, w, cls]
        num_present = tf.reduce_sum(weights)  # scalar
    else:
        num_present = tf.reduce_sum(tf.ones_like(predictions))  # scalar
    recall = tf.divide(intersection + 1e-32, ground_truth + 1e-32, name='recall')   # [?, h, w, cls]
    # Whether to keep batch
    rec_axis = (1, 2, 3) if keep_batch else None
    recall = tf.divide(tf.reduce_sum(recall, axis=rec_axis), num_present)   # scalar or [?,]
    recall = tf.stop_gradient(recall, name=scope)   # scalar or [?]
    # Finish.
    return recall
