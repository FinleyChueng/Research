import tensorflow as tf
import numpy as np



def DICE(labels, predictions, weights=None, scope=None, category_indep=True, ignore_BG=False):
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
    # Prepare
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
    dice = tf.divide(intersection, union)   # [?] or [?, cls]
    dice = tf.where(tf.not_equal(union, 0),
                    dice,
                    tf.ones_like(dice) * 1.,
                    name='batch_dice')   # [?] or [?, cls]
    # Specify
    if category_indep:
        dice = tf.reduce_mean(dice, axis=-1)    # [?]
    dice = tf.stop_gradient(dice, name=scope)   # [?]
    # Finish.
    return dice



def recall(labels, predictions, weights=None, scope=None, category_indep=True, ignore_BG=False):
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
    # Prepare
    if ignore_BG:
        labels = labels[:, :, :, 1:]  # [?, h, w, cls-1]
        predictions = predictions[:, :, :, 1:]  # [?, h, w, cls-1]
        if weights is not None:
            weights = weights[:, :, :, 1:]  # [?, h, w, cls-1]
    # Calculate
    intersection = tf.multiply(labels, predictions)  # [?, h, w, cls]
    ground_truth = labels
    if weights is not None:
        intersection = tf.multiply(intersection, weights)  # [?, h, w, cls]
        ground_truth = tf.multiply(ground_truth, weights)     # [?, h, w, cls]
    indep_axis = (1, 2) if category_indep else (1, 2, 3)
    intersection = tf.reduce_sum(intersection, axis=indep_axis)  # [?] or [?, cls]
    ground_truth = tf.reduce_sum(ground_truth, axis=indep_axis)  # [?] or [?, cls]
    recall = tf.divide(intersection, ground_truth)  # [?] or [?, cls]
    recall = tf.where(tf.not_equal(ground_truth, 0),
                      recall,
                      tf.ones_like(recall) * 1.,
                      name='batch_recall')    # [?] or [?, cls]
    # Specify
    if category_indep:
        recall = tf.reduce_mean(recall, axis=-1)  # [?]
    recall = tf.stop_gradient(recall, name=scope)   # [?]
    # Finish.
    return recall


