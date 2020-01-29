import numpy as np
from sklearn import metrics
import cv2



# ------------------------------------------------------------
# This metric is self-defined.
# ------------------------------------------------------------

def BRATS_Complete(pred, label):
    r'''
        The metrics for BRATS. Complete tumor.
    '''
    completeTumor = np.zeros(pred.shape)
    syntheticData = np.zeros(label.shape)
    completeTumor[pred != 0] = 1
    syntheticData[label != 0] = 1
    v = DICE_Bi(completeTumor, syntheticData)
    return v


def BRATS_Core(pred, label):
    r'''
        The metrics for BRATS. Core tumor.
    '''
    completeTumor = np.zeros(pred.shape)
    syntheticData = np.zeros(label.shape)
    completeTumor[np.logical_not(np.logical_or((pred == 0), (pred == 2)))] = 1
    syntheticData[np.logical_not(np.logical_or((label == 0), (label == 2)))] = 1
    v = DICE_Bi(completeTumor, syntheticData)
    return v


def BRATS_Enhance(pred, label):
    r'''
        The metrics for BRATS. Enhance tumor.
    '''
    completeTumor = np.zeros(pred.shape)
    syntheticData = np.zeros(label.shape)
    completeTumor[pred == 4] = 1
    syntheticData[label == 4] = 1
    v = DICE_Bi(completeTumor, syntheticData)
    return v



# ------------------------------------------------------------
# This metric is self-defined. Mainly the "Dice" metric.
# ------------------------------------------------------------

def DICE_Bi(pred, label):
    r'''
        The simplest Dice metric form. Which is applied to
            the "Bi-Segmentation" Task.
    '''
    # Check validity.
    if not isinstance(pred, np.ndarray):
        raise ValueError('The pred must be a array !!!')
    if not isinstance(label, np.ndarray):
        raise ValueError('The label must be a array !!!')
    pred_shape = np.asarray(pred.shape)
    lab_shape = np.asarray(label.shape)
    if (pred_shape == 0).any() or (lab_shape == 0).any():
        # maybe no region
        return 0.0
    if (pred_shape != lab_shape).any():
        raise Exception('The shape of pred and label must be equal !!!')
    # Transfer to binary clazz.
    pred = pred.astype(np.bool).astype(np.int64)
    label = label.astype(np.bool).astype(np.int64)
    # Calculate.
    intersection = np.sum(pred * label)
    union = np.sum(pred + label)
    if union == 0:
        return 1.0
    else:
        return 2. * intersection / union


def prop_DICE_metric(pred, label, category, ignore_BG=True):
    r'''
        Dice metric (proportional). Treat the whole pixel as equal. The metric
            value is with respect to the pixel proportion.
    '''
    # Check validity.
    if not isinstance(pred, np.ndarray) or pred.ndim != 2:
        raise ValueError('The pred must be a 2-D array !!!')
    if not isinstance(label, np.ndarray) or label.ndim != 2:
        raise ValueError('The label must be a 2-D array !!!')
    pred_shape = np.asarray(pred.shape)
    lab_shape = np.asarray(label.shape)
    if (pred_shape == 0).any() or (lab_shape == 0).any():
        # maybe no region
        return 0.0
    if (pred_shape != lab_shape).any():
        raise Exception('The shape of pred and label must be equal !!!')
    if max(np.max(pred), np.max(label)) > category:
        raise Exception('The value of pred or label should not greater than category !!!')
    # Calculate.
    eye = np.eye(category)
    ot_pred = eye[pred]
    ot_lab = eye[label]
    if ignore_BG:
        ot_pred = ot_pred[:, :, 1:]
        ot_lab = ot_lab[:, :, 1:]
    intersection = np.sum(ot_pred * ot_lab)
    union = np.sum(ot_pred) + np.sum(ot_lab)
    if union == 0:
        return 1.0
    else:
        return 2. * intersection / union


def mean_DICE_metric(pred, label, category, ignore_BG=True):
    r'''
        Dice metric (mean). Calculate each category pixel, respectively.
            The metric value is the mean of each category Dice value.
    '''
    # Check validity.
    if not isinstance(pred, np.ndarray) or pred.ndim != 2:
        raise ValueError('The pred must be a 2-D array !!!')
    if not isinstance(label, np.ndarray) or label.ndim != 2:
        raise ValueError('The label must be a 2-D array !!!')
    pred_shape = np.asarray(pred.shape)
    lab_shape = np.asarray(label.shape)
    if (pred_shape == 0).any() or (lab_shape == 0).any():
        # maybe no region
        return 0.0
    if (pred_shape != lab_shape).any():
        raise Exception('The shape of pred and label must be equal !!!')
    if max(np.max(pred), np.max(label)) > category:
        raise Exception('The value of pred or label should not greater than category !!!')
    # Calculate and mean the values.
    vals = []
    start_idx = 1 if ignore_BG else 0
    for c in range(start_idx, category):
        ot_pred = np.asarray(pred == c, np.int64)
        ot_lab = np.asarray(label == c, np.int64)
        intersection = np.sum(ot_pred * ot_lab)
        union = np.sum(ot_pred) + np.sum(ot_lab)
        if union == 0:
            v = 1.0
        else:
            v = 2. * intersection / union
        vals.append(v)
    return np.mean(vals)



# ------------------------------------------------------------
# This metric is self-defined.
# ------------------------------------------------------------

def CBRM_Metric(predict, grand_truth):
    r'''
        Compute the self-defined metric of given predict and grand truth.

    ------------------------------------------------------------------------------------------
    Parameters:
        predict: The predict result of the specific neural network.
        grand_truth: The grand truth of the given predict result which is specific
            for the task.

    -------------------------------------------------------------------------------------------
    Return:
        The self-defined metric of the given predict result and grand truth.
    '''

    diff = grand_truth - predict
    pure_cost = np.sum(diff)    # In some extend is "Cost"
    denominator = np.sum(np.absolute(diff))

    # The factor of exponent.
    factor = pure_cost / denominator
    # Exponent.
    CBRM = np.exp(-np.absolute(factor))

    return CBRM


# ------------------------------------------------------------
# This is the SIFT-based image similarity algorithm.
# ------------------------------------------------------------

def SIFT_similarity_metric(image_A, image_B):

    img1 = np.zeros(np.shape(image_A), dtype=np.uint8)
    img1[:, :] = image_A[:, :]
    img2 = np.zeros(np.shape(image_B), dtype=np.uint8)
    img2[:, :] = image_B[:, :]

    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(len(kp1), len(kp2))

    if des1 is None or des2 is None:
        return 0

    if len(kp1) < 2 or len(kp2) < 2:
        return 0

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    print('matches...', len(matches))
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print('good', len(good))

    return len(good)



# ----------------------------------------------------------------
# This is the Difference-based (Mask) image similarity algorithm.
# ----------------------------------------------------------------
def mdifference_similarity_algorithm(mask, src):

    # Normalize the mask and the src image, so that we can normalization.
    c_mask = _normalization(mask.copy(), 255)
    c_src = _normalization(src.copy(), 255)

    # Find the position that mask != 0.
    target_pixels_pos = np.where(c_mask != 0)

    # Generate blank, and then get the target pixels.
    t_img = np.zeros(np.shape(c_src))
    t_img[target_pixels_pos] = c_src[target_pixels_pos]

    # Calculate the standard difference for whole src.
    mdiff = np.std(c_mask - t_img)

    # return cost
    return mdiff


# ------------------------------------------------------------
# This is the NMI(Normalized Mutual Information) metric.
# ------------------------------------------------------------

def mNMI_metric_4image(mask, src):
    # Normalize the mask and the src image, so that we can normalization.
    c_mask = _normalization(mask.copy(), 255)
    c_src = _normalization(src.copy(), 255)

    # Find the position that mask != 0.
    target_pixels_pos = np.where(c_mask != 0)

    # Generate blank, and then get the target pixels.
    t_img = np.zeros(np.shape(c_src))
    t_img[target_pixels_pos] = c_src[target_pixels_pos]

    vec_A = image2D_to_vector(c_mask)
    vec_B = image2D_to_vector(t_img)
    return metrics.normalized_mutual_info_score(vec_A, vec_B)

def NMI_metric_4image(image_A, image_B):
    vec_A = image2D_to_vector(image_A)
    vec_B = image2D_to_vector(image_B)
    return metrics.normalized_mutual_info_score(vec_A, vec_B)

def image2D_to_vector(src):
    # Firstly get the width and height of source image.
    sw, sh = np.shape(src)
    # Secondly declare the 1-D vector used to store the pixels information.
    vec = np.zeros(sw * sh)
    # Finally iteratively assign the pixel value to specific position.
    for x in range(sw):
        for y in range(sh):
            vec[x * sw + y] = src[x, y]

    # Return the 1-D vector.
    return vec



# ---------------------------------------------------------------------
################### PUBLIC PART #######################################
#----------------------------------------------------------------------
def _normalization(src, upper):
    denominator = np.max(src) - np.min(src)
    if denominator == 0:
        normal = np.zeros(np.shape(src))
    else:
        normal = (src - np.min(src)) / (np.max(src) - np.min(src)) * upper
    return normal

