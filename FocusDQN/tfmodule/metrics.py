import tensorflow as tf
import numpy as np



def DICE(labels, predictions, weights=None, scope=None, cal_mask=None, keep_axis=None):
    r'''
        Calculate the "Dice" metric for given predictions. We can reduce
            arbitrary axis, default to all dimensions.

    Parameters:
        labels: Tensor of shape [d_0, d_1, ..., d_{r-1}, num_classes] (where r
            is rank of labels and result).
        predictions: One-hot predictions. The same shape as labels.
        weights: Coefficients for the loss. This must be None or the same as labels.
        scope: the scope for the operations performed in computing metric.
        cal_mask: The mask used to filter the region that used to calculate metric.
        keep_axis: The dimensions to keep. Default is "None" indicating reduce
            all the dimensions.

    Returns:
        Dice metric tensor and the corresponding flag matrix.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) < 2:
        raise TypeError('The labels must be at least 2-D tensor !!!')
    if not isinstance(predictions, tf.Tensor) or len(predictions.shape) < 2:
        raise TypeError('The predictions must be at least 2-D tensor !!!')
    if weights is not None and not isinstance(weights, tf.Tensor):
        raise TypeError('The weights must be None or tensor !!!')
    if (cal_mask is not None and not isinstance(cal_mask, tf.Tensor)) or len(cal_mask.shape) < 2:
        raise TypeError('The cal_mask must be at least 2-D tensor !!!')
    if len(labels.shape) > 1:
        lab_shape = np.asarray(labels.get_shape().as_list())
        pred_shape = np.asarray(predictions.get_shape().as_list())
        shape_invar = (lab_shape[1:] != pred_shape[1:]).any()
        if cal_mask is not None:
            cal_shape = np.asarray(cal_mask.get_shape().as_list())
            shape_invar = shape_invar or (pred_shape[1:-1] != cal_shape[1:-1]).any()
        if shape_invar:
            raise TypeError('The labels, predictions and weights must be of same shape !!!')
    if keep_axis is not None and not isinstance(keep_axis, (list, tuple)):
        raise TypeError('The keep_axis must be a tuple or list !!!')
    if isinstance(keep_axis, (list, tuple)) and \
            not (min(keep_axis) >= -len(predictions.shape) and max(keep_axis) < len(predictions.shape)):
        raise ValueError('The keep_axis must match the input tensor dimension value !!!')
    if weights is not None and keep_axis is not None and (len(weights.shape) != len(keep_axis)):
        raise Exception('The dimension of weights and keep_axis should be equal !!!')
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
    if keep_axis is not None:
        rec_axis = list(range(len(predictions.shape)))
        for ax in keep_axis:
            if ax < 0:
                axis = len(predictions.shape) + ax
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
    dice = tf.divide(intersection, tf.maximum(union, 1e-32), name='dice_metric')  # [?, d1, ..., dk]
    # Multiply with weights here.
    if weights is not None:
        dice = tf.multiply(dice, weights)   # [?, d1, ..., dk]
        num_present = tf.to_float(tf.greater(tf.multiply(num_present, weights), 0.))     # [?, d1, ..., dk]
    # Stop gradients here.
    dice = tf.stop_gradient(dice, name=scope)   # [?, d1, ..., dk]
    num_present = tf.stop_gradient(num_present)     # [?, d1, ..., dk]
    # Finish.
    return dice, num_present



def recall(labels, predictions, weights=None, scope=None, cal_mask=None, keep_axis=None):
    r'''
        Calculate the "Recall" metric for given predictions. We can reduce
            arbitrary axis, default to all dimensions.

    Parameters:
        labels: Tensor of shape [d_0, d_1, ..., d_{r-1}, num_classes] (where r
            is rank of labels and result).
        predictions: One-hot predictions. The same shape as labels.
        weights: Coefficients for the loss. This must be None or the same as labels.
        scope: the scope for the operations performed in computing metric.
        cal_mask: The mask used to filter the region that used to calculate metric.
        keep_axis: The dimensions to keep. Default is "None" indicating reduce
            all the dimensions.

    Returns:
        Recall metric tensor and the corresponding flag matrix.
    '''
    # Check validity.
    if not isinstance(labels, tf.Tensor) or len(labels.shape) < 2:
        raise TypeError('The labels must be at least 2-D tensor !!!')
    if not isinstance(predictions, tf.Tensor) or len(predictions.shape) < 2:
        raise TypeError('The predictions must be at least 2-D tensor !!!')
    if weights is not None and not isinstance(weights, tf.Tensor):
        raise TypeError('The weights must be None or tensor !!!')
    if (cal_mask is not None and not isinstance(cal_mask, tf.Tensor)) or len(cal_mask.shape) < 2:
        raise TypeError('The cal_mask must be at least 2-D tensor !!!')
    if len(labels.shape) > 1:
        lab_shape = np.asarray(labels.get_shape().as_list())
        pred_shape = np.asarray(predictions.get_shape().as_list())
        shape_invar = (lab_shape[1:] != pred_shape[1:]).any()
        if cal_mask is not None:
            cal_shape = np.asarray(cal_mask.get_shape().as_list())
            shape_invar = shape_invar or (pred_shape[1:-1] != cal_shape[1:-1]).any()
        if shape_invar:
            raise TypeError('The labels, predictions and weights must be of same shape !!!')
    if keep_axis is not None and not isinstance(keep_axis, (list, tuple)):
        raise TypeError('The keep_axis must be a tuple or list !!!')
    if isinstance(keep_axis, (list, tuple)) and \
            not (min(keep_axis) >= -len(predictions.shape) and max(keep_axis) < len(predictions.shape)):
        raise ValueError('The keep_axis must match the input tensor dimension value !!!')
    if weights is not None and keep_axis is not None and (len(weights.shape) != len(keep_axis)):
        raise Exception('The dimension of weights and keep_axis should be equal !!!')
    # Calculate the numerator and denominator.
    intersection = tf.multiply(labels, predictions)     # [?, d1, ..., dn, cls]
    ground_truth = labels   # [?, d1, ..., dn, cls]
    # Filter the region used to calculate.
    if cal_mask is not None:
        intersection = tf.multiply(intersection, cal_mask)  # [?, d1, ..., dn, cls]
        ground_truth = tf.multiply(ground_truth, cal_mask)  # [?, d1, ..., dn, cls]
        num_present = tf.multiply(cal_mask, labels)     # [?, d1, ..., dn, cls]
    else:
        num_present = labels    # [?, d1, ..., dn, cls]
    # Determine the axis to calculate.
    if keep_axis is not None:
        rec_axis = list(range(len(predictions.shape)))
        for ax in keep_axis:
            if ax < 0:
                axis = len(predictions.shape) + ax
            else:
                axis = ax
            rec_axis.remove(axis)
        rec_axis = tuple(rec_axis)
    else:
        rec_axis = None
    # Reduce sum the specific axis.
    if rec_axis is not None:
        intersection = tf.reduce_sum(intersection, axis=rec_axis)   # [?, d1, ..., dk]
        ground_truth = tf.reduce_sum(ground_truth, axis=rec_axis)   # [?, d1, ..., dk]
        num_present = tf.to_float(tf.greater(tf.reduce_sum(num_present, axis=rec_axis), 0.))    # [?, d1, ..., dk]
    # Real calculate the metric here.
    recall = tf.divide(intersection, tf.maximum(ground_truth, 1e-32), name='recall_metric')     # [?, d1, ..., dk]
    # Multiply with weights here.
    if weights is not None:
        recall = tf.multiply(recall, weights)   # [?, d1, ..., dk]
        num_present = tf.to_float(tf.greater(tf.multiply(num_present, weights), 0.))    # [?, d1, ..., dk]
    # Stop gradients here.
    recall = tf.stop_gradient(recall, name=scope)   # [?, d1, ..., dk]
    num_present = tf.stop_gradient(num_present)     # [?, d1, ..., dk]
    # Finish.
    return recall, num_present
