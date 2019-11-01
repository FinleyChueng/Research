import numpy as np
from sklearn import metrics
import cv2



# ------------------------------------------------------------
# This metric is self-defined.
# ------------------------------------------------------------

def Region1(data, label):
    completeTumor = np.zeros(data.shape)
    syntheticData = np.zeros(label.shape)
    completeTumor[data != 0] = 1
    syntheticData[label != 0] = 1
    return completeTumor, syntheticData


def Region2(data, label):
    completeTumor = np.zeros(data.shape)
    syntheticData = np.zeros(label.shape)
    completeTumor[np.logical_not(np.logical_or((data == 0), (data == 2)))] = 1
    syntheticData[np.logical_not(np.logical_or((label == 0), (label == 2)))] = 1
    return completeTumor, syntheticData


def Region3(data, label):
    completeTumor = np.zeros(data.shape)
    syntheticData = np.zeros(label.shape)
    completeTumor[data == 4] = 1
    syntheticData[label == 4] = 1
    return completeTumor, syntheticData


def DICE_coef(y_true, y_pred):
    y_true_f = y_true
    y_pred_f = y_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0004) / (np.sum(y_true_f) + np.sum(y_pred_f) + 0.0004)



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
    # # #####################################
    # # visualization
    # h1, w1 = img1.shape[:2]
    # h2, w2 = img2.shape[:2]
    # view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    # view[:h1, :w1, 0] = img1
    # view[:h2, w1:, 0] = img2
    # view[:, :, 1] = view[:, :, 0]
    # view[:, :, 2] = view[:, :, 0]
    #
    # for m in good:
    #     # draw the keypoints
    #     # print m.queryIdx, m.trainIdx, m.distance
    #     color = tuple([np.random.randint(0, 255) for _ in range(3)])
    #     # print 'kp1,kp2',kp1,kp2
    #     cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])),
    #              (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), color)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(view)
    # plt.show()

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
    # mdiff = np.std(np.absolute(c_mask - t_img))
    # mdiff = np.sum(np.absolute(c_mask - t_img))

    # if len(target_pixels_pos[0]) != 0:
    #     print('################################')
    #     print(c_mask)
    #     print(c_src)
    #     print(t_img)
    #     print(mdiff)

    # return cost
    return mdiff



# ------------------------------------------------------------
# This is the Histogram-based image similarity algorithm.
# ------------------------------------------------------------
# def histogram_similarity_algorithm(image_A, image_B):
#
#     # Do not finished yet.
#
#     img1 = np.zeros(np.shape(image_A), dtype=np.uint8)
#     img1[:, :] = image_A[:, :]
#     img2 = np.zeros(np.shape(image_B), dtype=np.uint8)
#     img2[:, :] = image_B[:, :]
#
#     hist1, cvals1 = exposure.histogram(img1)
#     hist2, cvals2 = exposure.histogram(img2)
#
#     print(hist1, cvals1)
#     print(hist2, cvals2)
#     # cost = np.sum(hist1 - hist2)
#
#     # return cost
#     return 9



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


# ------------------------------------------------------------
# This version is suitable for simply coefficient calculation.
# ------------------------------------------------------------

def dice_coef(predict, grand_truth):
    r'''
        Calculate the Dice coefficient of the given predict result and grand truth.

        Specially, below shows the formulation of Dice coefficient:
            Dice = 2*TP / (FP + 2*TP + FN)

    ------------------------------------------------------------------------------------------
    Parameters:
        predict: The predict result of the specific neural network.
        grand_truth: The grand truth of the given predict result which is specific
            for the task.

    -------------------------------------------------------------------------------------------
    Return:
        The dice coefficient of the given predict result and grand truth.
    '''

    TP = true_positive(predict, grand_truth)
    FP = false_positive(predict, grand_truth)
    FN = false_negative(predict, grand_truth)
    # calculate the denominator
    denominator = FP + 2*TP + FN
    if denominator == 0:
        dice = 0
    else:
        dice = 2*TP / denominator

    return dice

def true_positive(predict, grand_truth):
    r'''
        Calculate the true-positive (TP) coefficient of the given predict result
            and grand truth.

        ------------------------------------------------------------------------------------------
        Parameters:
            predict: The predict result of the specific neural network.
            grand_truth: The grand truth of the given predict result which is specific
                for the task.

        -------------------------------------------------------------------------------------------
        Return:
            The true-positive (TP) coefficient of the given predict result and grand truth.
        '''

    # # Use vector operation to compute the TP value.
    # TP_tensor = np.logical_and(predict, grand_truth)
    # TP = np.sum(TP_tensor == True)

    # Use vector operation to compute the TP value.
    B_predict = np.asarray(predict, dtype=np.bool)
    B_gt = np.asarray(grand_truth, dtype=np.bool)
    TP_tensor = np.logical_and(B_predict, B_gt)
    TP = np.sum(TP_tensor)

    return TP

def true_negative(predict, grand_truth):
    r'''
        Calculate the true-negative (TN) coefficient of the given predict result
            and grand truth.

        ------------------------------------------------------------------------------------------
        Parameters:
            predict: The predict result of the specific neural network.
            grand_truth: The grand truth of the given predict result which is specific
                for the task.

        -------------------------------------------------------------------------------------------
        Return:
            The true-negative (TN) coefficient of the given predict result and grand truth.
        '''

    # # Use vector operation to compute the FP value.
    # TN_tensor = np.logical_or(predict, grand_truth)
    # TN = np.sum(TN_tensor == False)

    # Use vector operation to compute the FP value.
    B_predict = np.asarray(predict, dtype=np.bool)
    B_grand_truth = np.asarray(grand_truth, dtype=np.bool)
    TN_tensor = np.logical_not(np.logical_or(B_predict, B_grand_truth))
    TN = np.sum(TN_tensor)

    return TN

def false_positive(predict, grand_truth):
    r'''
        Calculate the false-positive (FP) coefficient of the given predict result
            and grand truth.

        ------------------------------------------------------------------------------------------
        Parameters:
            predict: The predict result of the specific neural network.
            grand_truth: The grand truth of the given predict result which is specific
                for the task.

        -------------------------------------------------------------------------------------------
        Return:
            The false-positive (FP) coefficient of the given predict result and grand truth.
        '''

    # # First find the all positive value in predict result, meanwhile, obtain the
    # #   corresponding grand truth value at the same position as predict result.
    # #   Generate vectors for both of them respectively.
    # pred_p_index = np.where(predict == True)
    # pred_p_tensor = predict[pred_p_index]
    # gt_cor_tensor = grand_truth[pred_p_index]
    # # Use vector operation to compute the FP value.
    # FP_tensor = np.logical_xor(pred_p_tensor, gt_cor_tensor)
    # FP = np.sum(FP_tensor)

    # First find the all positive value in predict result, meanwhile, obtain the
    #   corresponding grand truth value at the same position as predict result.
    #   Generate vectors for both of them respectively.
    B_predict = np.asarray(predict, dtype=np.bool)
    B_grand_truth = np.asarray(grand_truth, dtype=np.bool)
    pred_p_index = B_predict
    pred_p_tensor = B_predict[pred_p_index]
    gt_cor_tensor = B_grand_truth[pred_p_index]
    # Use vector operation to compute the FP value.
    FP_tensor = np.logical_xor(pred_p_tensor, gt_cor_tensor)
    FP = np.sum(FP_tensor)

    return FP

def false_negative(predict, grand_truth):
    r'''
        Calculate the false-negative (FN) coefficient of the given predict result
            and grand truth.

        ------------------------------------------------------------------------------------------
        Parameters:
            predict: The predict result of the specific neural network.
            grand_truth: The grand truth of the given predict result which is specific
                for the task.

        -------------------------------------------------------------------------------------------
        Return:
            The false-negative (FN) coefficient of the given predict result and grand truth.
        '''

    # # First find the all negative value in predict result, meanwhile, obtain the
    # #   corresponding grand truth value at the same position as predict result.
    # #   Generate vectors for both of them respectively.
    # pred_n_index = np.where(predict == False)
    # pred_n_tensor = predict[pred_n_index]
    # gt_cor_tensor = grand_truth[pred_n_index]
    # # Use vector operation to compute the FP value.
    # FN_tensor = np.logical_xor(pred_n_tensor, gt_cor_tensor)
    # FN = np.sum(FN_tensor)

    # First find the all negative value in predict result, meanwhile, obtain the
    #   corresponding grand truth value at the same position as predict result.
    #   Generate vectors for both of them respectively.
    B_predict = np.asarray(predict, dtype=np.bool)
    B_grand_truth = np.asarray(grand_truth, dtype=np.bool)
    pred_n_index = np.logical_not(B_predict)
    pred_n_tensor = B_predict[pred_n_index]
    gt_cor_tensor = B_grand_truth[pred_n_index]
    # Use vector operation to compute the FP value.
    FN_tensor = np.logical_xor(pred_n_tensor, gt_cor_tensor)
    FN = np.sum(FN_tensor)

    return FN


# ------------------------------------------------------------
# The version below will skips the "NaN" when calculating.
# ------------------------------------------------------------

def NaN_dice_coef(predict, grand_truth):
    r'''
        Calculate the Dice coefficient of the given predict result and grand truth
            under the condition that "NaN" does not participate in calculations.

        Specially, below shows the formulation of Dice coefficient:
            Dice = 2*TP / (FP + 2*TP + FN)

    ------------------------------------------------------------------------------------------
    Parameters:
        predict: The predict result of the specific neural network.
        grand_truth: The grand truth of the given predict result which is specific
            for the task.

    -------------------------------------------------------------------------------------------
    Return:
        The dice coefficient of the given predict result and grand truth.
    '''

    NaN_TP = NaN_true_positive(predict, grand_truth)
    NaN_FP = NaN_false_positive(predict, grand_truth)
    NaN_FN = NaN_false_negative(predict, grand_truth)
    # calculate the denominator
    denominator = NaN_FP + 2*NaN_TP + NaN_FN
    if denominator == 0:
        NaN_Dice = 0
    else:
        NaN_Dice = 2*NaN_TP / denominator

    return NaN_Dice

def NaN_true_positive(predict, grand_truth):
    r'''
        Calculate the true-positive (TP) coefficient of the given predict result
            and grand truth under the condition that "NaN" does not participate
            in calculations.

        ------------------------------------------------------------------------------------------
        Parameters:
            predict: The predict result of the specific neural network.
            grand_truth: The grand truth of the given predict result which is specific
                for the task.

        -------------------------------------------------------------------------------------------
        Return:
            The true-positive (TP) coefficient of the given predict result and grand truth.
        '''

    # Firstly replace the "NaN" with "zeros".
    duplication = predict.copy()
    duplication[np.where(np.isnan(predict))] = 0
    # Use vector operation to compute the TP value.
    TP_tensor = np.logical_and(duplication, grand_truth)
    TP = np.sum(TP_tensor == True)

    return TP

def NaN_true_negative(predict, grand_truth):
    r'''
        Calculate the true-negative (TN) coefficient of the given predict result
            and grand truth under the condition that "NaN" does not participate
            in calculations.

        ------------------------------------------------------------------------------------------
        Parameters:
            predict: The predict result of the specific neural network.
            grand_truth: The grand truth of the given predict result which is specific
                for the task.

        -------------------------------------------------------------------------------------------
        Return:
            The true-negative (TN) coefficient of the given predict result and grand truth.
        '''

    # Firstly replace the "NaN" with "ones".
    duplication = predict.copy()
    duplication[np.where(np.isnan(predict))] = 1
    # Use vector operation to compute the FP value.
    TN_tensor = np.logical_or(duplication, grand_truth)
    TN = np.sum(TN_tensor == False)

    return TN

def NaN_false_positive(predict, grand_truth):
    r'''
        Calculate the false-positive (FP) coefficient of the given predict result
            and grand truth under the condition that "NaN" does not participate
            in calculations.

        ------------------------------------------------------------------------------------------
        Parameters:
            predict: The predict result of the specific neural network.
            grand_truth: The grand truth of the given predict result which is specific
                for the task.

        -------------------------------------------------------------------------------------------
        Return:
            The false-positive (FP) coefficient of the given predict result and grand truth.
        '''

    # Firstly replace the "NaN" with "zeros".
    duplication = predict.copy()
    duplication[np.where(np.isnan(predict))] = 0
    # Secondly find the all positive value in predict result, meanwhile, obtain the
    #   corresponding grand truth value at the same position as predict result.
    #   Generate vectors for both of them respectively.
    pred_p_index = np.where(duplication == True)
    pred_p_tensor = duplication[pred_p_index]
    gt_cor_tensor = grand_truth[pred_p_index]
    # Use vector operation to compute the FP value.
    FP_tensor = np.logical_xor(pred_p_tensor, gt_cor_tensor)
    FP = np.sum(FP_tensor)

    return FP

def NaN_false_negative(predict, grand_truth):
    r'''
        Calculate the false-negative (FN) coefficient of the given predict result
            and grand truth under the condition that "NaN" does not participate
            in calculations.

        ------------------------------------------------------------------------------------------
        Parameters:
            predict: The predict result of the specific neural network.
            grand_truth: The grand truth of the given predict result which is specific
                for the task.

        -------------------------------------------------------------------------------------------
        Return:
            The false-negative (FN) coefficient of the given predict result and grand truth.
        '''

    # Firstly replace the "NaN" with "ones".
    duplication = predict.copy()
    duplication[np.where(np.isnan(predict))] = 1
    # Secondly find the all negative value in predict result, meanwhile, obtain the
    #   corresponding grand truth value at the same position as predict result.
    #   Generate vectors for both of them respectively.
    pred_n_index = np.where(duplication == False)
    pred_n_tensor = duplication[pred_n_index]
    gt_cor_tensor = grand_truth[pred_n_index]
    # Use vector operation to compute the FP value.
    FN_tensor = np.logical_xor(pred_n_tensor, gt_cor_tensor)
    FN = np.sum(FN_tensor)

    return FN



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

